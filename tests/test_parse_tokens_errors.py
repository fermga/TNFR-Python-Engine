"""Pruebas de parse tokens errors."""

from __future__ import annotations

import pytest

from tnfr import token_parser as core_token_parser
from tnfr.cli import _parse_tokens, TOKEN_MAP


def test_cli_reexports_consolidated_token_parser():
    """Ensure the CLI exposes the consolidated parser API."""

    assert _parse_tokens is core_token_parser._parse_tokens
    assert TOKEN_MAP is core_token_parser.TOKEN_MAP


def test_parse_tokens_value_error_context():
    with pytest.raises(ValueError) as exc:
        list(_parse_tokens([{"WAIT": "x"}]))
    msg = str(exc.value)
    assert msg.endswith("(pos 1, token {'WAIT': 'x'})")
    assert isinstance(exc.value.__cause__, ValueError)


def test_parse_tokens_key_error_context(monkeypatch):
    def raiser(spec):
        return spec["missing"]

    monkeypatch.setitem(TOKEN_MAP, "RAISE", raiser)
    with pytest.raises(ValueError) as exc:
        list(_parse_tokens([{"RAISE": {}}]))
    msg = str(exc.value)
    assert msg.endswith("(pos 1, token {'RAISE': {}})")
    assert isinstance(exc.value.__cause__, KeyError)


def test_parse_tokens_type_error_context(monkeypatch):
    def raiser(spec):
        raise TypeError("boom")

    monkeypatch.setitem(TOKEN_MAP, "RAISE_TYPE", raiser)
    with pytest.raises(ValueError) as exc:
        list(_parse_tokens([{"RAISE_TYPE": {}}]))
    msg = str(exc.value)
    assert msg.endswith("(pos 1, token {'RAISE_TYPE': {}})")
    assert isinstance(exc.value.__cause__, TypeError)


def test_thol_invalid_close():
    with pytest.raises(ValueError) as exc:
        list(_parse_tokens([{"THOL": {"close": "XYZ"}}]))
    msg = str(exc.value)
    assert "XYZ" in msg
    assert "Glyph" in msg
    assert isinstance(exc.value.__cause__, ValueError)


def test_parse_tokens_error_format_unified(monkeypatch):
    def raise_key(spec):
        return spec["missing"]

    def raise_value(spec):
        raise ValueError("boom")

    def raise_type(spec):
        raise TypeError("boom")

    monkeypatch.setitem(TOKEN_MAP, "RAISE_KEY", raise_key)
    monkeypatch.setitem(TOKEN_MAP, "RAISE_VALUE", raise_value)
    monkeypatch.setitem(TOKEN_MAP, "RAISE_TYPE", raise_type)

    cases = [
        ([{"FOO": 1, "BAR": 2}], "Invalid token"),
        ([{"UNKNOWN": None}], "Unrecognized token"),
        ([{"RAISE_KEY": {}}], "KeyError"),
        ([{"RAISE_VALUE": {}}], "ValueError"),
        ([{"RAISE_TYPE": {}}], "TypeError"),
    ]

    for tokens, start in cases:
        with pytest.raises(ValueError) as exc:
            list(_parse_tokens(tokens))
        msg = str(exc.value)
        assert msg.startswith(start)
        expected_end = (
            "(pos 1)" if start == "Invalid token" else f"(pos 1, token {tokens[0]!r})"
        )
        assert msg.endswith(expected_end)
