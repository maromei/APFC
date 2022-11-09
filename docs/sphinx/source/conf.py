# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'APFC'
copyright = '2022, maromei'
author = 'maromei'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",          # use markdown
    "breathe",              # for doxygen doc
    "sphinx.ext.mathjax",   # display math in docstrings
    "sphinx.ext.napoleon"   # Numpy docstrings
]

templates_path = ['_templates']
exclude_patterns = []

breathe_projects = {"apfc": "../../doxygen/xml/"}
breathe_default_project = "apfc"

myst_enable_extensions = ["dollarmath", "amsmath"]

html_title = "APFC"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
