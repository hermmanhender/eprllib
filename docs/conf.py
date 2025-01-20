# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'eprllib'
copyright = '2024, Germán Rodolfo Henderson'
author = 'Germán Rodolfo Henderson'
release = '1.5.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # To support Google and NumPy style docstrings
    'sphinx.ext.viewcode',  # Adds links to highlighted source code
]

templates_path = ['_templates']
exclude_patterns = ['Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/hermmanhender/eprllib",
    "repository_branch": "docs",
    "path_to_docs": "/docs",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
}

html_logo = "Images/eprllib_logo.png"
html_title = "eprllib: use EnergyPlus as an environment for DRL control"
html_baseurl = "https://hermmanhender.github.io/eprllib/docs/"
