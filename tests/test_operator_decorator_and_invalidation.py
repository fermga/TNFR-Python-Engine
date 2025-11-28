"""Tests for structural_operator decorator and cache invalidation.

Validates:
1. Decorator auto-registration
2. Soft invalidation preserves registry count
3. Hard invalidation clears then repopulates registry
"""

from __future__ import annotations

import pytest  # noqa: F401 (fixture access)
from tnfr.operators.registry import (
    OPERATORS,
    structural_operator,
    invalidate_operator_cache,
    discover_operators,
)
from tnfr.operators.definitions_base import Operator
from tnfr.types import Glyph


def test_structural_operator_decorator_registers():
    class TempOp(Operator):  # no decorator yet
        name = "temp_op_test"
        glyph = Glyph.AL

    # Ensure not auto-registered without decorator
    assert "temp_op_test" not in OPERATORS

    @structural_operator
    class TempOp2(Operator):  # noqa: D401 - simple test class
        name = "temp_op_test2"
        glyph = Glyph.EN

    assert "temp_op_test2" in OPERATORS
    assert OPERATORS["temp_op_test2"] is TempOp2


def test_soft_invalidation():
    discover_operators()
    before = len(OPERATORS)
    stats = invalidate_operator_cache(hard=False)
    after = len(OPERATORS)
    assert stats["cleared"] == 0
    assert stats["count"] == after
    assert after >= before  # may grow if new ops added


def test_hard_invalidation():
    discover_operators()
    before = len(OPERATORS)
    stats = invalidate_operator_cache(hard=True)
    assert stats["cleared"] == before
    assert stats["count"] >= 1  # repopulated
    assert len(OPERATORS) == stats["count"]
