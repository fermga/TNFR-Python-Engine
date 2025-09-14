"""Pruebas de evaluación lazy en _parse_tokens."""

from __future__ import annotations

import pytest

from tnfr.cli import _parse_tokens


def test_parse_tokens_lazy_evaluation():
    it = _parse_tokens([{"WAIT": "x"}])
    assert iter(it) is it
    with pytest.raises(ValueError):
        next(it)


def test_parse_tokens_large_input():
    grande = ({"WAIT": 1} for _ in range(10000))
    assert sum(1 for _ in _parse_tokens(grande)) == 10000
