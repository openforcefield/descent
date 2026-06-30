(training)=

# Training Force Fields

DESCENT provides a flexible framework for training SMIRNOFF force field parameters using gradient-based optimization. The framework uses SMEE (Scalable Molecular simulation with Extreme Efficiency) as the differentiable backend for evaluating force fields. This guide covers the core concepts and workflows for parameter optimization.

## Overview

The training process in DESCENT involves:

1. **Loading a SMIRNOFF force field**: Starting point for parameter optimization (e.g., OpenFF 2.0.0)
2. **Defining target data**: Reference data to fit against (QM energies, experimental properties, dimer interactions)
3. **Selecting parameters**: Which SMIRNOFF parameters to optimize (bond lengths, angles, torsions, vdW, etc.)
4. **Configuring loss functions**: How to measure agreement with targets
5. **Running optimization**: Using PyTorch optimizers to minimize the loss through differentiable force field evaluations

## Training Configuration

Training is configured using the `TrainingConfig` class or a YAML file:

```python
from descent.train import TrainingConfig, train

# Load configuration from file
config = TrainingConfig.from_file("config.yaml")

# Or create programmatically
config = TrainingConfig(
    force_field="openff-2.0.0.offxml",
    parameters=["[#6:1]-[#6:2]", "[#6:1]-[#1:2]"],
    targets=[...],
    optimizer_config={...}
)

# Run training
results = train(config)
```

## Target Functions

DESCENT supports multiple target functions that can be combined:

### Energy Targets

Fit to quantum mechanical or reference energies:

```python
from descent.targets import EnergyTarget

target = EnergyTarget(
    dataset="energies.json",
    weight=1.0,
    denominator="std"  # Normalize by standard deviation
)
```

### Thermodynamic Targets

Fit to experimental thermodynamic properties (density, enthalpy of vaporization, etc.):

```python
from descent.targets import ThermoTarget

target = ThermoTarget(
    dataset="thermo.json",
    weight=0.5,
    properties=["density", "hvap"]
)
```

### Dimer Targets

Fit to dimer interaction energies:

```python
from descent.targets import DimerTarget

target = DimerTarget(
    dataset="dimers.json",
    weight=0.3
)
```

## Optimization Strategies

DESCENT uses PyTorch optimizers for parameter optimization:

```python
from descent.optim import AdamW

optimizer = AdamW(
    lr=1e-3,
    weight_decay=1e-4
)
```

Common optimizers:
- `Adam`: Adaptive moment estimation
- `AdamW`: Adam with weight decay
- `SGD`: Stochastic gradient descent
- `LBFGS`: Limited-memory BFGS

## Loss Functions

The overall loss is a weighted combination of target losses:

```python
from descent.utils.loss import WeightedLoss

loss_fn = WeightedLoss(targets=[
    (energy_target, 1.0),
    (thermo_target, 0.5),
    (dimer_target, 0.3)
])
```

## Training Loop

The basic training loop:

```python
from descent.train import train

# Configure training
config = TrainingConfig(...)

# Run training
results = train(
    config,
    max_epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Access results
print(f"Final loss: {results['final_loss']}")
print(f"Best epoch: {results['best_epoch']}")
```

## Monitoring Progress

DESCENT provides utilities for monitoring training progress:

```python
from descent.utils.reporting import TrainingReporter

reporter = TrainingReporter(log_dir="logs")

# During training
reporter.log_epoch(epoch, loss, metrics)
reporter.log_parameters(epoch, parameters)
```

## Checkpointing

Save and resume training:

```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'parameters': parameters,
    'optimizer_state': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pt')

# Resume training
checkpoint = torch.load('checkpoint.pt')
parameters = checkpoint['parameters']
optimizer.load_state_dict(checkpoint['optimizer_state'])
```

## Best Practices

1. **Start with small learning rates**: ~1e-3 to 1e-4
2. **Use validation sets**: Monitor for overfitting
3. **Normalize targets**: Scale different target types appropriately
4. **Monitor gradients**: Check for exploding/vanishing gradients
5. **Save checkpoints**: Regularly save progress
6. **Validate results**: Test optimized parameters on held-out data

## Advanced Topics

### Custom Target Functions

Create custom target functions by subclassing `BaseTarget`:

```python
from descent.targets import BaseTarget

class CustomTarget(BaseTarget):
    def compute_loss(self, system, parameters):
        # Custom loss computation
        ...
        return loss
```

### Learning Rate Schedules

Use PyTorch learning rate schedulers:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)
```

### Regularization

Add regularization to prevent overfitting:

```python
# L2 regularization (weight decay)
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)

# Or add custom regularization to loss
loss = target_loss + lambda_reg * torch.norm(parameters)
```

## See Also

- [Targets](targets.md): Details on target functions
- [Optimization](optimization.md): Advanced optimization strategies
- [API Reference](api/train.rst): Complete API documentation
