"""Tests for OperatorMetaAuto metaclass & cache telemetry.

Physics-neutral: ensures registration mechanisms do not alter semantics.
"""
from __future__ import annotations

from tnfr.operators.registry import (
    OPERATORS,
    get_operator_cache_stats,
    invalidate_operator_cache,
)
from tnfr.operators.grammar import get_grammar_cache_stats
from tnfr.operators.definitions_base import Operator


def test_metaclass_auto_registration():
    stats_before = get_operator_cache_stats()
    base_registrations = stats_before["registrations"]

    class AutoMetaTestOp(Operator):  # noqa: D401 - simple test class
        name = "auto_meta_test"
        glyph = None  # not required for registration test

    assert "auto_meta_test" in OPERATORS
    stats_after = get_operator_cache_stats()
    # registrations should have increased by at least 1
    assert stats_after["registrations"] >= base_registrations + 1


def test_cache_stats_invalidation_soft():
    pre = get_operator_cache_stats()
    soft_before = pre["soft_invalidations"]
    result = invalidate_operator_cache(hard=False)
    assert "count" in result and "cleared" in result
    post = get_operator_cache_stats()
    assert post["soft_invalidations"] == soft_before + 1


def test_cache_stats_invalidation_hard():
    pre = get_operator_cache_stats()
    hard_before = pre["hard_invalidations"]
    result = invalidate_operator_cache(hard=True)
    assert result["cleared"] >= 0
    post = get_operator_cache_stats()
    assert post["hard_invalidations"] == hard_before + 1


def test_grammar_cache_stats():
    stats = get_grammar_cache_stats()
    assert isinstance(stats, dict)
    # structure: mapping of function name -> dict with integer values
    for fn, info in stats.items():
        assert {"hits", "misses", "maxsize", "currsize"}.issubset(info.keys())
        assert all(isinstance(v, int) for v in info.values())
