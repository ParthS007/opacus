# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)


logger = logging.getLogger(__name__)


class PSACDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    Differentially Private Per-Sample Adaptive Clipping (DP-PSAC) algorithm
    based on a non-monotonic adaptive weight function.

    This optimizer uses per-sample adaptive clipping thresholds instead of
    a constant global clipping norm, which reduces the deviation between
    the clipped batch gradient and the true batch-averaged gradient.

    Reference: "Differentially Private Learning with Per-Sample Adaptive Clipping"
    (https://arxiv.org/abs/2212.00328)

    Unlike normalization-based approaches (Auto-S, NSGD) that use a monotonic
    weight function 1/(||g||+r) which over-weights small gradients, DP-PSAC
    uses a non-monotonic adaptive weight function that gives similar order of
    weights to samples with different gradient norms.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        r: float = 0.01,
        tau0: float = 0.1,
        tau1: float = 0.5,
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        """
        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier for differential privacy
            max_grad_norm: initial/upper bound for per-sample clipping norms.
                This serves as an upper bound; actual clipping norms adapt per-sample.
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required if ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            r: hyperparameter for the non-monotonic weight function (default: 0.01).
                Controls the shape of the adaptive weight function. Typically set to
                0.1 or smaller values like 0.01.
            tau0: lower threshold parameter for adaptive weight function (default: 0.1)
            tau1: upper threshold parameter for adaptive weight function (default: 0.5)
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )

        if r <= 0:
            raise ValueError("r must be positive")
        if tau0 <= 0 or tau1 <= 0:
            raise ValueError("tau0 and tau1 must be positive")
        if tau0 >= tau1:
            raise ValueError("tau0 must be less than tau1")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        self.r = r
        self.tau0 = tau0
        self.tau1 = tau1

        # Per-sample adaptive clipping thresholds (initialized on first use)
        # This will store the adaptive threshold for each sample in the current batch
        self._per_sample_clip_norms = None

    def _compute_non_monotonic_weight(self, grad_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute the non-monotonic adaptive weight function for per-sample gradients.

        The weight function is designed to avoid over-weighting small gradients
        while maintaining similar order of weights across different gradient norms.

        Args:
            grad_norm: per-sample gradient norms, shape [batch_size]

        Returns:
            Adaptive weights for each sample, shape [batch_size]
        """
        # Non-monotonic weight function based on the paper
        # The function gives similar order of weights to samples with different norms
        # while avoiding excessive weight on very small gradients

        # Normalize by max_grad_norm to work in relative scale
        normalized_norm = grad_norm / (self.max_grad_norm + 1e-8)

        # Non-monotonic adaptive weight function
        # For small norms (< tau0), weight increases with norm
        # For medium norms (tau0 <= norm < tau1), weight is relatively constant
        # For large norms (>= tau1), weight decreases
        weight = torch.where(
            normalized_norm < self.tau0,
            # Small gradients: weight increases with norm (non-monotonic part)
            (normalized_norm + self.r) / (self.tau0 + self.r),
            torch.where(
                normalized_norm < self.tau1,
                # Medium gradients: relatively constant weight
                1.0,
                # Large gradients: weight decreases
                (self.tau1 + self.r) / (normalized_norm + self.r),
            ),
        )

        return weight

    def _compute_per_sample_adaptive_clip_norms(
        self, per_sample_norms: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample adaptive clipping thresholds based on the non-monotonic
        weight function.

        Args:
            per_sample_norms: per-sample gradient norms, shape [batch_size]

        Returns:
            Adaptive clipping thresholds for each sample, shape [batch_size]
        """
        # Compute adaptive weights
        weights = self._compute_non_monotonic_weight(per_sample_norms)

        # Adaptive clipping threshold: scale max_grad_norm by the weight
        # This ensures samples with different norms get appropriate clipping
        adaptive_clip_norms = self.max_grad_norm * weights

        # Ensure clipping norms are within reasonable bounds
        # Lower bound: prevent division by zero and ensure minimum clipping
        min_clip_norm = self.max_grad_norm * self.r / (1.0 + self.r)
        adaptive_clip_norms = torch.clamp(
            adaptive_clip_norms, min=min_clip_norm, max=self.max_grad_norm
        )

        return adaptive_clip_norms

    def clip_and_accumulate(self):
        """
        Performs per-sample adaptive gradient clipping.
        Uses non-monotonic adaptive weight function to determine per-sample
        clipping thresholds, then clips and accumulates gradients.
        """
        if len(self.grad_samples) == 0 or len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros(
                (0,),
                device=(
                    self.grad_samples[0].device
                    if self.grad_samples
                    else torch.device("cpu")
                ),
            )
            self._per_sample_clip_norms = None
        else:
            # Compute per-parameter norms
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]

            if per_param_norms:
                target_device = per_param_norms[0].device
                per_param_norms = [norm.to(target_device) for norm in per_param_norms]

            # Compute per-sample gradient norms
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

            # Compute per-sample adaptive clipping thresholds
            self._per_sample_clip_norms = self._compute_per_sample_adaptive_clip_norms(
                per_sample_norms
            )

            # Compute per-sample clip factors using adaptive thresholds
            per_sample_clip_factor = (
                self._per_sample_clip_norms / (per_sample_norms + 1e-8)
            ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)

            # gradients should match the dtype of the optimizer parameters
            grad_sample = grad_sample.to(p.dtype)
            clip_factor_on_device = per_sample_clip_factor.to(grad_sample.device).to(
                p.dtype
            )
            grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        Adds noise to clipped gradients. The noise is calibrated based on the
        maximum clipping norm to ensure differential privacy guarantees.
        """
        # For privacy, we use the maximum clipping norm across all samples
        # This ensures the sensitivity is bounded
        if self._per_sample_clip_norms is not None:
            # Use the maximum adaptive clip norm for noise calibration
            max_adaptive_clip_norm = self._per_sample_clip_norms.max().item()
            # But we still use self.max_grad_norm as the upper bound for privacy
            # to maintain the privacy guarantee
            effective_clip_norm = min(max_adaptive_clip_norm, self.max_grad_norm)
        else:
            effective_clip_norm = self.max_grad_norm

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients and per-sample clipping state.
        """
        super().zero_grad(set_to_none)
        self._per_sample_clip_norms = None
