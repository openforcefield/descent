# DESCENT Documentation

This directory contains the Sphinx documentation for DESCENT.

## Building the Documentation

### Using Pixi (Recommended)

This project uses [Pixi](https://pixi.sh/) for environment management. To build the documentation:

```bash
# Build HTML documentation
pixi run -e docs make_docs

# Clean build artifacts
pixi run -e docs clean_docs
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

### Manual Build

If you have the dependencies installed in your current environment:

```bash
cd docs
make html
```

### Prerequisites

The `docs` environment in `pixi.toml` includes all required dependencies:
- Sphinx ≥ 7.0
- sphinx_rtd_theme ≥ 2.0
- myst-parser ≥ 2.0
- nbsphinx ≥ 0.9
- sphinxcontrib-bibtex ≥ 2.5
- sphinx-autodoc-typehints ≥ 1.24

### Other Build Commands

```bash
make latexpdf  # Build PDF documentation
make epub      # Build EPUB documentation
make linkcheck # Check for broken links
make clean     # Remove build artifacts
```

## Documentation Structure

- `index.md` - Main documentation index
- `installation.md` - Installation guide
- `examples.md` - Examples and tutorials
- `training.md` - Training guide
- `targets.md` - Target functions guide
- `optimization.md` - Optimization guide
- `releasehistory.md` - Release notes
- `api/` - API reference documentation
- `_static/` - Static files (images, CSS)
- `_templates/` - Custom templates

## Contributing to Documentation

When adding new features:

1. Update relevant guide pages (training.md, targets.md, etc.)
2. Add docstrings to new classes/functions (NumPy style)
3. Update API reference if adding new modules
4. Add examples if appropriate
5. Build and review the documentation locally
6. Check for broken links with `make linkcheck`

## Style Guide

- Use NumPy docstring format for all docstrings
- Use Markdown for narrative documentation
- Use reStructuredText for API reference files
- Include code examples where helpful
- Add cross-references using MyST syntax: `[text](path.md)` or `` {ref}`label` ``
- Use math notation with `$...$` for inline and `$$...$$` for display math
