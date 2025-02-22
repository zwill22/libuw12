import textwrap

project = "libuw12"
copyright = "2024, Z. M. Williams"
author = "Z. M. Williams"
release = "0.1.0"

extensions = [
    "sphinx.ext.mathjax",
    "breathe",
    "exhale",
    "sphinx_rtd_theme",
    "myst_parser",
]

breathe_projects = {"libuw12": "./_doxygen/xml/"}
breathe_default_project = "libuw12"

exhale_args = {
    # These arguments are required
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "doxygenStripFromPath": "..",
    "rootFileTitle": "libuw12 API",
    # Suggested optional arguments
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "verboseBuild": True,
    "exhaleDoxygenStdin": textwrap.dedent("""
    INPUT = ../
    TOC_INCLUDE_HEADINGS = 2
    EXTRACT_ALL = NO
    EXTRACT_LOCAL_CLASSES = NO
    EXCLUDE_PATTERNS = "*test*" "example" "cmake" "docs"
    PROJECT_NAME = "libuw12"
    FILE_PATTERNS = *.hpp *.h *.md
    USE_MATHJAX = TRUE
    """),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"
todo_include_todos = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
