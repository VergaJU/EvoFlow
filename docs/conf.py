"""Sphinx configuration for the EvoFlow documentation."""

import os
import sys

# Make the package importable when building from a checkout.
sys.path.insert(0, os.path.abspath(".."))

project = "EvoFlow"
author = "Jacopo"
copyright = "2026, Jacopo"
release = "1.0.0"
version = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "furo"
html_title = "EvoFlow documentation"

# -- Autodoc / Napoleon ------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": False,
}
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

# -- MyST / notebooks --------------------------------------------------------
# Notebooks are committed pre-executed; Read the Docs must not try to run the
# R/NicheNet/Docker parts of the workflow during a documentation build.
nb_execution_mode = "off"
nb_merge_streams = True
nb_output_stderr = "show"

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Copy button -------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
