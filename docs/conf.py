#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# descent documentation build configuration file

import os
import sys
from importlib.util import find_spec as find_import_spec

sys.path.insert(0, os.path.abspath(".."))

import descent

# -- Project information -----------------------------------------------------

project = "DESCENT"
copyright = "2024, Simon Boothroyd"
author = "Simon Boothroyd"

# The version info for the project
version = descent.__version__ if hasattr(descent, "__version__") else "0.1.0"
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "openff_sphinx_theme",
    "myst_nb",
    # "myst_parser",  # Commented out as myst_nb handles markdown
]

# sphinx-notfound-page
# https://github.com/readthedocs/sphinx-notfound-page
# Renders a 404 page with absolute links
if find_import_spec("notfound"):
    extensions.append("notfound.extension")

    notfound_urls_prefix = "/projects/descent/en/stable/"
    notfound_context = {
        "title": "404: File Not Found",
        "body": f"""
    <h1>404: File Not Found</h1>
    <p>
        Sorry, we couldn't find that page. This often happens as a result of
        following an outdated link. Please check the
        <a href="{notfound_urls_prefix}">latest stable version</a>
        of the docs, unless you're sure you want an earlier version, and
        try using the search box or the navigation menu on the left.
    </p>
    <p>
    </p>
    """,
    }

# Autodoc settings
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}
autodoc_preserve_defaults = True
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_attr_annotations = True
napoleon_custom_sections = [("attributes", "params_style")]
napoleon_use_rtype = False
napoleon_use_param = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "openff.toolkit": (
        "https://docs.openforcefield.org/projects/toolkit/en/stable/",
        None,
    ),
}

# MyST settings
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "smartquotes",
    "replacements",
    "deflist",
]
myst_heading_anchors = 4

# Myst NB settings
# Never execute notebooks - this should be done by CI
# Output is stored in the notebook itself
nb_execution_mode = "off"

# Source files
source_suffix = [".rst", ".md", ".ipynb"]

master_doc = "index"

# Language
language = "en"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Todo settings
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

html_theme = "openff_sphinx_theme"

html_theme_options = {
    # Repository integration
    "repo_url": "https://github.com/SimonBoothroyd/descent",
    "repo_name": "descent",
    "repo_type": "github",
    "color_accent": "openff-toolkit-blue",
    "html_minify": False,
    "html_prettify": False,
    "css_minify": True,
    "master_doc": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates
templates_path = ["_templates"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    "**": ["globaltoc.html", "searchbox.html", "localtoc.html"],
}

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "descentdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

latex_documents = [
    (master_doc, "descent.tex", "DESCENT Documentation", author, "manual"),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "descent", "DESCENT Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "descent",
        "DESCENT Documentation",
        author,
        "descent",
        "Differentiable parameter optimization for force fields.",
        "Miscellaneous",
    ),
]
