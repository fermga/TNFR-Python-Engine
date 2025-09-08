"""Tests for ``get_attr`` in strict mode."""

import logging
import pytest
from tnfr.alias import AliasAccessor


def test_get_attr_logs_on_error(caplog):
    d = {"x": "abc"}
    acc = AliasAccessor(int)
    with caplog.at_level(logging.DEBUG):
        result = acc.get(d, ("x",))
    assert result is None
    assert any("Could not convert" in m for m in caplog.messages)


def test_get_attr_logs_each_error(caplog):
    d = {"x": "abc", "y": "def"}
    acc = AliasAccessor(int)
    with caplog.at_level(logging.DEBUG):
        result = acc.get(d, ("x", "y"))
    assert result is None
    messages = [m for m in caplog.messages if "Could not convert" in m]
    assert len(messages) == 2
    assert any("'x'" in m for m in messages)
    assert any("'y'" in m for m in messages)


def test_get_attr_custom_log_level(caplog):
    d = {"x": "abc"}
    acc = AliasAccessor(int)
    with caplog.at_level(logging.WARNING):
        acc.get(d, ("x",), log_level=logging.WARNING)
    assert any("Could not convert" in m for m in caplog.messages)


def test_get_attr_strict_raises():
    d = {"x": "abc"}
    acc = AliasAccessor(int)
    with pytest.raises(ValueError):
        acc.get(d, ("x",), strict=True)
