# DESCENT

*Differentiable parameter optimization for force fields*

DESCENT is a modern framework for training SMIRNOFF force field parameters against reference data using PyTorch. It provides a differentiable approach to parameter optimization, enabling efficient gradient-based training through the SMEE (Scalable Molecular simulation with Extreme Efficiency) backend.

:::{toctree}
---
caption: Getting Started
maxdepth: 1
---

installation
examples
conduct
releasehistory
:::

:::{toctree}
---
caption: User Guide
maxdepth: 1
---

training
targets
optimization
:::

:::{toctree}
---
caption: API Reference
maxdepth: 1
---

api/train
api/targets
api/optim
api/utils
:::

## Features

- **Differentiable SMIRNOFF Parameters**: Compute gradients through SMIRNOFF force field evaluations using PyTorch and SMEE
- **Flexible Target Functions**: Optimize against various target properties (QM energies, thermodynamic properties, dimer interactions)
- **Parameter Control**: Fine-grained control over which parameters to train, with support for scaling, clamping, and regularization
- **Modern API**: Clean, pythonic interface built on PyTorch
- **Extensible**: Easy to add new target functions and optimization strategies

## Quick Example

```python
import descent

# Load your training configuration
config = descent.train.TrainingConfig.from_file("config.yaml")

# Run optimization
descent.train.train(config)
```

## Citation

If you use DESCENT in your research, please cite:

```bibtex
@software{descent,
  author = {Boothroyd, Simon},
  title = {DESCENT: Differentiable parameter optimization for force fields},
  url = {https://github.com/openforcefield/descent},
  year = {2024}
}
```

## Acknowledgments

This framework benefited hugely from [ForceBalance](https://github.com/leeping/forcebalance), and significant learnings from that project and from Lee-Ping have influenced the design of DESCENT.

:::{warning}
This code is currently experimental and under active development. If you are using it, please be aware that it is not guaranteed to provide correct results, the documentation and testing may be incomplete, and the API can change without notice.
:::

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
