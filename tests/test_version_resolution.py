"""Tests covering version resolution for the :mod:`tnfr` package."""

from __future__ import annotations


def test_version_constant_matches_module() -> None:
    import tnfr
    from tnfr import _version

    assert tnfr.__version__ == _version.__version__
