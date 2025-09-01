from __future__ import annotations

import pytest

from tnfr.cli import _parse_tokens, TOKEN_MAP


def test_parse_tokens_value_error_context():
    with pytest.raises(ValueError) as exc:
        _parse_tokens([{"WAIT": "x"}])
    msg = str(exc.value)
    assert "posición 1" in msg
    assert "WAIT" in msg


def test_parse_tokens_key_error_context(monkeypatch):
    def raiser(spec):
        return spec["missing"]

    monkeypatch.setitem(TOKEN_MAP, "RAISE", raiser)
    with pytest.raises(KeyError) as exc:
        _parse_tokens([{"RAISE": {}}])
    msg = str(exc.value)
    assert "posición 1" in msg
    assert "RAISE" in msg
