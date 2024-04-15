# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import textwrap

project = 'libuw12'
copyright = '2024, Z. M. Williams'
author = 'Z. M. Williams'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.mathjax', 'breathe', 'exhale', 'sphinx_rtd_theme']

breathe_projects = {"libuw12": "../doxygen/xml/"}
breathe_default_project = "libuw12"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "doxygenStripFromPath":  "..",
    "rootFileTitle":         "Library API",
    # Suggested optional arguments
    "createTreeView":        True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":   textwrap.dedent('''
    INPUT = ../../
    TOC_INCLUDE_HEADINGS = 2
    EXTRACT_ALL = NO
    EXTRACT_LOCAL_CLASSES = NO
    EXCLUDE_PATTERNS = "*test*" "example"
    ''')
}


# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

templates_path = []
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
