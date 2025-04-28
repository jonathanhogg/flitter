# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Flitter'
copyright = '2025, Jonathan Hogg'
author = 'Jonathan Hogg'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser']

# templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
# html_static_path = ['_static']

myst_enable_extensions = ['dollarmath', 'deflist', 'colon_fence']

html_theme_options = {
    'github_button': True,
    'github_user': 'jonathanhogg',
    'github_repo': 'flitter',
    'description': "A functional programming language and declarative system for describing 2D and 3D visuals",
    'show_relbars': True,
}
