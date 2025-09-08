"""Pruebas de parse tokens errors."""

from __future__ import annotations

import pytest

from tnfr.cli import _parse_tokens, TOKEN_MAP


def test_parse_tokens_value_error_context():
    with pytest.raises(ValueError) as exc:
        _parse_tokens([{"WAIT": "x"}])
    msg = str(exc.value)
    assert "posición 1" in msg
    assert "WAIT" in msg
    assert isinstance(exc.value.__cause__, ValueError)


def test_parse_tokens_key_error_context(monkeypatch):
    def raiser(spec):
        return spec["missing"]

    monkeypatch.setitem(TOKEN_MAP, "RAISE", raiser)
    with pytest.raises(ValueError) as exc:
        _parse_tokens([{"RAISE": {}}])
    msg = str(exc.value)
    assert "posición 1" in msg
    assert "RAISE" in msg
    assert isinstance(exc.value.__cause__, KeyError)


def test_parse_tokens_type_error_context(monkeypatch):
    def raiser(spec):
        raise TypeError("boom")

    monkeypatch.setitem(TOKEN_MAP, "RAISE_TYPE", raiser)
    with pytest.raises(ValueError) as exc:
        _parse_tokens([{"RAISE_TYPE": {}}])
    msg = str(exc.value)
    assert "posición 1" in msg
    assert "RAISE_TYPE" in msg
    assert isinstance(exc.value.__cause__, TypeError)


def test_thol_invalid_close():
    with pytest.raises(ValueError) as exc:
        _parse_tokens([{"THOL": {"close": "XYZ"}}])
    msg = str(exc.value)
    assert "XYZ" in msg
    assert "Glyph" in msg
    assert isinstance(exc.value.__cause__, ValueError)
