# Clipping Strategies Usage Guide

## Installation

To use a custom Opacus fork, install directly from the git repository:

```bash
# Install from specific branch/commit
pip install "opacus @ git+https://github.com/parths007/opacus.git@master-thesis"

# Or install in editable mode if you've cloned the repository
git clone https://github.com/parths007/opacus.git
cd opacus
git checkout master-thesis
pip install -e .

# Optional: Install with dev dependencies for development
pip install -e .[dev]
```

**Note:** The `@ git+...` syntax installs directly from the repository. Use editable mode (`-e`) if you need to modify the code and see changes immediately without reinstalling.

## Overview

Opacus supports multiple gradient clipping strategies for differentially private training. Set the `clipping` parameter in `PrivacyEngine.make_private()`.

## Available Strategies

### 1. Flat Clipping (Default)

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,  # Single float value
    clipping="flat",
)
```

### 2. Per-Layer Clipping

```python
# Provide list of max_grad_norm (one per parameter)
max_grad_norm = [1.0, 1.5, 0.8, ...]  # One value per layer
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=max_grad_norm,
    clipping="per_layer",
)
```

### 3. Automatic Clipping

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="automatic",
)
```

### 4. Automatic Per-Layer Clipping

```python
max_grad_norm = [1.0, 1.5, 0.8, ...]  # One value per layer
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=max_grad_norm,
    clipping="automatic_per_layer",
)
```

### 5. Adaptive Clipping (AdaClip)

```python
from opacus.optimizers import AdaClipDPOptimizer

# Requires additional parameters
optimizer = AdaClipDPOptimizer(
    optimizer=optimizer,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    target_unclipped_quantile=0.5,
    clipbound_learning_rate=0.2,
    max_clipbound=1e8,
    min_clipbound=1.0,
    unclipped_num_std=1.0,
    expected_batch_size=batch_size,
)
```

**Note:** For adaptive clipping with ghost clipping, use `PrivacyEngineAdaptiveClipping`:

```python
from opacus.utils.adaptive_clipping import PrivacyEngineAdaptiveClipping

privacy_engine = PrivacyEngineAdaptiveClipping()
model, optimizer, criterion, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    criterion=criterion,
    noise_multiplier=1.0,
    max_grad_norm=10.0,  # Initial clipping norm
    grad_sample_mode="ghost",
    target_unclipped_quantile=0.5,
    min_clipbound=1.0,
    max_clipbound=1e8,
    clipbound_learning_rate=0.2,
)
```

### 6. Per-Sample Adaptive Clipping (PSAC)

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="psac",
)
```

### 7. Normalized SGD

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="normalized_sgd",
)
```

## Distributed Training

For distributed training, set `distributed=True`:

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="flat",  # or "per_layer", "automatic", etc.
    distributed=True,
)
```

## Ghost Clipping (Fast Gradient Clipping)

Use `grad_sample_mode="ghost"` for memory-efficient training (only with flat clipping):

```python
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    clipping="flat",
    grad_sample_mode="ghost",
)
```
