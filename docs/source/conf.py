# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

# Define the path to your module using Path
module_path = Path(__file__).parent.parent / "model_api" / "python"

# Insert the path to sys.path
sys.path.insert(0, str(module_path.resolve()))

project = 'InferenceSDK'
copyright = '2024, Intel OpenVINO'
author = 'Intel OpenVINO'
release = '2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.graphviz"
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
html_theme_options = {
    "logo": {
        "text": "InferenceSDK",
    },
}

breathe_projects = {"InferenceSDK":  Path(__file__).parent.parent/"build_cpp"/ "xml"}
breathe_default_project = "InferenceSDK"