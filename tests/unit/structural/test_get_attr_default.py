"""Pruebas de ``AliasAccessor.get`` con valores por defecto."""

import logging
from pathlib import Path
import importlib.util
import types
import sys

import pytest

# Import ``AliasAccessor`` without triggering package-level side effects.
pkg = types.ModuleType("tnfr")
pkg.__path__ = [str(Path(__file__).resolve().parents[3] / "src" / "tnfr")]
assert pkg.__path__  # ensure attribute is used for package resolution

_orig = sys.modules.get("tnfr")

sys.modules["tnfr"] = pkg
spec = importlib.util.spec_from_file_location(
    "tnfr.alias", Path(__file__).resolve().parents[3] / "src" / "tnfr" / "alias.py"
)
alias = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alias)  # type: ignore[union-attr]
AliasAccessor = alias.AliasAccessor

if _orig is not None:
    sys.modules["tnfr"] = _orig
else:  # pragma: no cover - cleanup when original module absent
    del sys.modules["tnfr"]


def test_get_attr_default_none_returns_none():
    d = {}
    acc = AliasAccessor(int)
    result = acc.get(d, ("x",), default=None)
    assert result is None
    assert d == {}


def test_get_attr_default_is_converted():
    d = {}
    acc = AliasAccessor(int)
    result = acc.get(d, ("x",), default="5")
    assert result == 5
    assert d == {}


def test_get_attr_default_logs_on_failure(caplog):
    d = {}
    acc = AliasAccessor(int)
    with caplog.at_level(logging.DEBUG, logger="tnfr.utils.data"):
        result = acc.get(d, ("x",), default="bad")
    assert result is None
    assert len(caplog.records) == 1
    assert "default" in caplog.text


def test_get_attr_default_strict_raises():
    d = {}
    acc = AliasAccessor(int)
    with pytest.raises(ValueError):
        acc.get(d, ("x",), default="bad", strict=True)
