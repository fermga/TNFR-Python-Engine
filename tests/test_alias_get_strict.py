"""Tests for ``alias_get`` in strict mode."""

import logging
import pytest
from tnfr.alias import alias_get


def test_alias_get_logs_on_error(caplog):
    d = {"x": "abc"}
    with caplog.at_level(logging.DEBUG):
        result = alias_get(d, ("x",), int)
    assert result is None
    assert any("Could not convert" in m for m in caplog.messages)


def test_alias_get_custom_log_level(caplog):
    d = {"x": "abc"}
    with caplog.at_level(logging.WARNING):
        alias_get(d, ("x",), int, log_level=logging.WARNING)
    assert any("Could not convert" in m for m in caplog.messages)


def test_alias_get_strict_raises():
    d = {"x": "abc"}
    with pytest.raises(ValueError):
        alias_get(d, ("x",), int, strict=True)
