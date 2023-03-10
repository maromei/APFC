# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath("../../src"))

project = "APFC"
copyright = "2022, maromei"
author = "maromei"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # use markdown
    "sphinx.ext.mathjax",  # display math in docstrings
    "sphinx.ext.napoleon",  # Numpy docstrings
    "sphinxcontrib.bibtex",  # Bibliography
    "sphinx.ext.todo",
]

todo_include_todos = True
todo_link_only = True

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "super"

templates_path = ["_templates"]
exclude_patterns = []

myst_enable_extensions = ["dollarmath", "amsmath"]

html_title = "APFC"

math_numfig = True
numfig = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
