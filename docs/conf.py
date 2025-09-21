# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'intvert'
copyright = '2025, Isaac Viviano'
author = 'Isaac Viviano'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# For other params to be separated
napoleon_use_param = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True

doctest_global_setup = """
import numpy as np
import gmpy2
import intvert
"""

# rst_prolog = """
# .. include:: _links.rst
# """


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



# -- Substitutions for Cross References ------------

import re

fnames = ["mp_dft", "mp_idft", "mp_dft2", "mp_idft2", "get_coeff_classes_1D", "get_coeff_classes_2D", "select_coeffs_1D", "select_coeffs_2D", "sample_1D", "sample_2D", "invert_1D", "invert_2D"]

def process_docstring(app, what, name, obj, options, lines):
    for i in range(len(lines)):
        for fname in fnames:
            lines[i] = re.sub(f'`{fname}`', f':py:func:`{fname} <intvert.{fname}>`', lines[i])
    # You can add more complex substitution logic here

def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)