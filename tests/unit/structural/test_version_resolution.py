"""Tests covering version resolution for the :mod:`tnfr` package."""

from __future__ import annotations

import importlib
import sys
import types
from importlib import metadata
from importlib.metadata import PackageNotFoundError


def _clear_tnfr_modules() -> None:
    """Remove cached :mod:`tnfr` modules so import side effects re-trigger."""

    for name in [
        key for key in sys.modules if key == "tnfr" or key.startswith("tnfr.")
    ]:
        sys.modules.pop(name)


def test_version_prefers_package_metadata(monkeypatch) -> None:
    """When metadata is present it should define ``tnfr.__version__``."""

    _clear_tnfr_modules()

    expected = "9.9.9-metadata"
    monkeypatch.setattr(metadata, "version", lambda name: expected)

    tnfr = importlib.import_module("tnfr")

    assert tnfr.__version__ == expected


def test_version_falls_back_to_local_module(monkeypatch) -> None:
    """If metadata resolution fails the bundled version must be used."""

    _clear_tnfr_modules()

    expected = "9.9.9-local"
    fake_version_module = types.ModuleType("tnfr._version")
    fake_version_module.__version__ = expected

    monkeypatch.setitem(sys.modules, "tnfr._version", fake_version_module)

    def _raise(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(metadata, "version", _raise)

    tnfr = importlib.import_module("tnfr")

    assert tnfr.__version__ == expected
