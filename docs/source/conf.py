"""Sphinx configuration for the TNFR Python Engine documentation."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import pypandoc
except ImportError:
    pypandoc = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
os.environ.setdefault("SPHINX_APIDOC_OPTIONS", "members,special-members,show-inheritance")
sys.path.insert(0, str(SRC_PATH))

if pypandoc is not None:
    pandoc_path = Path(pypandoc.get_pandoc_path()).resolve()
    os.environ.setdefault("PYPANDOC_PANDOC", str(pandoc_path))
    os.environ["PATH"] = f"{pandoc_path.parent}{os.pathsep}{os.environ.get('PATH', '')}"

project = "TNFR Python Engine"
author = "TNFR maintainers"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

nbsphinx_execute = "never"

myst_fence_as_directive = [
    "mermaid",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
}

autodoc_typehints = "description"
autodoc_preserve_defaults = True

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

def setup(app):  # noqa: D401
    """Hook for Sphinx customisation."""
    app.add_css_file("custom.css") if (Path(__file__).parent / "_static" / "custom.css").exists() else None
