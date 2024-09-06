"""Sphinx configuration."""
project = "Multi-agent reinforcement learning for credit theory of money and #MMT."
author = "Niklas Breitenbach-Netter"
copyright = "2024, Niklas Breitenbach-Netter"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
