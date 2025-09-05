"""Pruebas de alias get strict."""

import logging
import pytest
from tnfr.helpers import alias_get, _validate_aliases


def test_alias_get_logs_on_error(caplog):
    d = {"x": "abc"}
    aliases = _validate_aliases(("x",))
    with caplog.at_level(logging.DEBUG):
        result = alias_get(d, aliases, int)
    assert result is None
    assert any("No se pudo convertir" in m for m in caplog.messages)


def test_alias_get_custom_log_level(caplog):
    d = {"x": "abc"}
    aliases = _validate_aliases(("x",))
    with caplog.at_level(logging.WARNING):
        alias_get(d, aliases, int, log_level=logging.WARNING)
    assert any("No se pudo convertir" in m for m in caplog.messages)


def test_alias_get_strict_raises():
    d = {"x": "abc"}
    aliases = _validate_aliases(("x",))
    with pytest.raises(ValueError):
        alias_get(d, aliases, int, strict=True)
