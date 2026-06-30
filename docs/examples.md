(examples)=

# Examples

This section contains examples and tutorials demonstrating how to use DESCENT for force field parameter optimization.

## Available Examples

The `examples/` directory in the repository contains Jupyter notebooks and scripts showing various use cases:

:::{include} ../examples/README.md
:start-line: 2
:::

## Running Examples

To run the example notebooks, you'll need to install Jupyter:

```shell-session
$ mamba install -c conda-forge jupyter
```

Then navigate to the examples directory and start Jupyter:

```shell-session
$ cd examples
$ jupyter notebook
```

## Example Workflows

### Basic Parameter Optimization

A typical DESCENT workflow involves:

1. **Define training data**: Specify reference data (e.g., QM energies, thermodynamic properties)
2. **Configure force field**: Set up the force field to optimize
3. **Select target functions**: Choose what properties to fit (energies, dimers, thermodynamics)
4. **Configure optimization**: Set up the optimizer and hyperparameters
5. **Run training**: Execute the optimization loop
6. **Analyze results**: Evaluate the fitted parameters

### Configuration File Example

DESCENT uses YAML configuration files to specify training settings:

```yaml
# Example training configuration
force_field: openff-2.0.0.offxml
parameters:
  - "[#6:1]-[#6:2]"  # C-C bonds
  - "[#6:1]-[#1:2]"  # C-H bonds

targets:
  - type: energy
    dataset: qm_energies.json
    weight: 1.0
  
  - type: thermo
    dataset: thermo_data.json
    weight: 0.5

optimizer:
  type: Adam
  lr: 0.001
  
training:
  max_epochs: 100
  batch_size: 32
```

## Additional Resources

- [Contributing Guidelines](../CONTRIBUTING.md): How to contribute examples or improvements
- [API Documentation](api/train.rst): Detailed API reference
- [GitHub Repository](https://github.com/SimonBoothroyd/descent): Source code and issue tracker
