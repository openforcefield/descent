(installation)=

# Installation

## Installing via Conda/Mamba

The recommended way to install DESCENT is via [Mamba](https://github.com/mamba-org/mamba) or [Conda](https://docs.conda.io/en/latest/):

```shell-session
$ mamba create -n descent -c conda-forge descent
$ mamba activate descent
```

If you don't have Mamba installed, you can use Conda instead:

```shell-session
$ conda create -n descent -c conda-forge descent
$ conda activate descent
```

## Installing via Pixi

DESCENT can also be installed using [Pixi](https://pixi.sh/), a modern package manager for Python and other languages. The repository includes a `pixi.toml` file:

```shell-session
$ pixi install
$ pixi shell
```

## Installing from Source

To install DESCENT from source for development purposes:

```shell-session
$ git clone https://github.com/SimonBoothroyd/descent.git
$ cd descent
$ pip install -e .
```

For development, you may want to install additional dependencies:

```shell-session
$ pip install -e ".[dev,test,docs]"
```

## Dependencies

DESCENT requires:

- Python ≥ 3.12
- PyTorch
- OpenFF Toolkit
- RDKit
- NumPy

Optional dependencies:

- Jupyter (for running example notebooks)
- Sphinx (for building documentation)

## OS Support

DESCENT is pure Python and should work on any platform that supports its dependencies. Development and testing primarily occurs on:

- macOS (x86_64 and arm64)
- Linux (x86_64)
- Windows (limited testing)

## Verifying Installation

To verify that DESCENT is installed correctly, run:

```python
import descent
print(descent.__version__)
```

Or run the test suite:

```shell-session
$ pytest descent/tests/
```

## Troubleshooting

If you encounter issues during installation:

1. **Import errors**: Ensure all dependencies are installed in the same environment
2. **OpenFF Toolkit issues**: Make sure you have RDKit or OpenEye toolkit installed
3. **PyTorch issues**: Verify PyTorch is properly installed for your system

For additional help, please [open an issue](https://github.com/SimonBoothroyd/descent/issues) on GitHub.
