(optimization)=

# Optimization

DESCENT uses PyTorch-based optimizers for gradient-based parameter optimization. This guide covers optimization strategies, learning rate schedules, and best practices.

## Overview

Parameter optimization in DESCENT involves:

1. **Computing the loss**: Evaluate target functions with current parameters
2. **Computing gradients**: Automatic differentiation through force field evaluations
3. **Updating parameters**: Apply optimizer step
4. **Monitoring convergence**: Track loss and parameter changes

## Optimizers

### Adam

Adaptive Moment Estimation (Adam) is the default optimizer:

```python
from descent.optim import Adam

optimizer = Adam(
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Pros:**
- Adaptive learning rates per parameter
- Works well with sparse gradients
- Fast convergence

**Cons:**
- Can generalize poorly in some cases
- May require careful tuning

### AdamW

Adam with decoupled weight decay regularization:

```python
from descent.optim import AdamW

optimizer = AdamW(
    lr=1e-3,
    weight_decay=1e-4,  # L2 regularization
    betas=(0.9, 0.999)
)
```

Recommended for most use cases.

### SGD

Stochastic Gradient Descent with momentum:

```python
from descent.optim import SGD

optimizer = SGD(
    lr=1e-2,
    momentum=0.9,
    nesterov=True
)
```

**Pros:**
- Simple and robust
- Better generalization in some cases

**Cons:**
- Requires careful learning rate tuning
- Slower convergence

### L-BFGS

Limited-memory BFGS for quasi-Newton optimization:

```python
from descent.optim import LBFGS

optimizer = LBFGS(
    lr=1.0,
    max_iter=20,
    history_size=10
)
```

**Pros:**
- Fast convergence
- Second-order information

**Cons:**
- Requires closure function
- Higher memory usage
- Not suitable for stochastic training

## Learning Rate Schedules

### Constant Learning Rate

The simplest approach:

```python
optimizer = Adam(lr=1e-3)
```

### Step Decay

Reduce learning rate at specified epochs:

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)
```

### Reduce on Plateau

Reduce when loss plateaus:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    threshold=1e-4
)
```

### Cosine Annealing

Cosine decay with warm restarts:

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2
)
```

### Exponential Decay

Exponential learning rate decay:

```python
from torch.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(
    optimizer,
    gamma=0.95
)
```

## Gradient Computation

DESCENT uses PyTorch's automatic differentiation:

```python
import torch

# Enable gradient tracking
parameters = torch.nn.Parameter(initial_params, requires_grad=True)

# Compute loss
loss = compute_loss(parameters)

# Compute gradients
loss.backward()

# Access gradients
print(parameters.grad)

# Update parameters
optimizer.step()
optimizer.zero_grad()
```

### Gradient Clipping

Prevent exploding gradients:

```python
from torch.nn.utils import clip_grad_norm_

# Clip by norm
clip_grad_norm_(parameters, max_norm=1.0)

# Or clip by value
clip_grad_value_(parameters, clip_value=0.5)
```

## Levenberg-Marquardt Optimization

For problems with structured residuals, DESCENT provides Levenberg-Marquardt optimization:

```python
from descent.optim import LevenbergMarquardt

optimizer = LevenbergMarquardt(
    lambda_init=1e-3,
    lambda_factor=10.0,
    max_iter=100
)
```

## Regularization

### L2 Regularization (Weight Decay)

Add L2 penalty to loss:

```python
# Built into AdamW
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)

# Or manually
loss = target_loss + 1e-4 * torch.sum(parameters ** 2)
```

### L1 Regularization

Add L1 penalty for sparsity:

```python
loss = target_loss + 1e-4 * torch.sum(torch.abs(parameters))
```

### Parameter Bounds

Constrain parameters to physical ranges:

```python
# Project parameters after update
with torch.no_grad():
    parameters.clamp_(min=0.0, max=10.0)
```

## Convergence Criteria

Monitor convergence using multiple criteria:

```python
def check_convergence(loss_history, param_history):
    # Loss convergence
    if abs(loss_history[-1] - loss_history[-2]) < 1e-6:
        return True
        
    # Parameter convergence
    param_change = torch.norm(param_history[-1] - param_history[-2])
    if param_change < 1e-5:
        return True
        
    # Gradient convergence
    grad_norm = torch.norm(parameters.grad)
    if grad_norm < 1e-6:
        return True
        
    return False
```

## Best Practices

1. **Start with AdamW**: Good default for most problems
2. **Use learning rate schedules**: Reduce LR when loss plateaus
3. **Monitor gradients**: Check for NaN or extreme values
4. **Clip gradients**: Prevent instability
5. **Use validation sets**: Monitor for overfitting
6. **Save checkpoints**: Regularly save best parameters
7. **Tune hyperparameters**: Use grid search or Bayesian optimization

## Hyperparameter Tuning

Common hyperparameters to tune:

- **Learning rate**: 1e-2 to 1e-5
- **Batch size**: 16 to 128
- **Weight decay**: 0 to 1e-3
- **Gradient clip**: 0.5 to 5.0
- **Target weights**: Problem-dependent

Example grid search:

```python
from itertools import product

lr_values = [1e-2, 1e-3, 1e-4]
wd_values = [0, 1e-4, 1e-3]

best_loss = float('inf')
best_params = None

for lr, wd in product(lr_values, wd_values):
    optimizer = AdamW(lr=lr, weight_decay=wd)
    loss = train(optimizer, ...)
    
    if loss < best_loss:
        best_loss = loss
        best_params = (lr, wd)
```

## Troubleshooting

### Loss not decreasing

- Check learning rate (too small?)
- Check gradients (vanishing?)
- Try different optimizer
- Simplify problem (fewer parameters)

### Loss exploding

- Reduce learning rate
- Add gradient clipping
- Check for numerical issues
- Normalize inputs

### Overfitting

- Add regularization (weight decay)
- Use validation set
- Reduce model complexity
- Get more training data

### Slow convergence

- Increase learning rate
- Use momentum/Adam
- Check gradient magnitudes
- Normalize targets

## See Also

- [Training](training.md): Training workflows
- [Targets](targets.md): Target functions
- [API Reference](api/optim.rst): Complete API documentation
