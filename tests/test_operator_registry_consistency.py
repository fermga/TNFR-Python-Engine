"""Registry Consistency Tests (TNFR Canonical)

Ensures reflective auto-registration of operator classes remains complete
and aligned with glyph/function mappings. Protects canonical invariant:
Operator closure and glyph correspondence (AGENTS.md Invariant #4 & #5).
"""

from __future__ import annotations

import pytest

from tnfr.operators.grammar import (
    GLYPH_TO_FUNCTION,
    OPERATORS,
    OPERATOR_NAME_TO_CLASS,
)
from tnfr.operators.registry import discover_operators


@pytest.mark.parametrize("glyph,func_name", list(GLYPH_TO_FUNCTION.items()))
def test_all_glyph_function_names_registered(glyph, func_name):
    """Each glyph function name must map to a registered operator."""
    discover_operators()
    assert func_name in OPERATORS, (
        f"Function '{func_name}' for glyph {glyph} not registered"
    )
    cls = OPERATORS[func_name]
    assert getattr(cls, "name", None) == func_name


def test_operator_name_mapping_matches_registry():
    """Name→class mapping must mirror OPERATORS for canonical aliases."""
    discover_operators()
    assert set(OPERATOR_NAME_TO_CLASS.keys()) == set(OPERATORS.keys())
    for name, cls in OPERATORS.items():
        assert OPERATOR_NAME_TO_CLASS[name] is cls


def test_no_duplicate_operator_classes():
    """No duplicate class objects bound to the same canonical name."""
    discover_operators()
    seen = {}
    for name, cls in OPERATORS.items():
        seen.setdefault(id(cls), []).append(name)
    # If a class id maps to >1 name that's a violation
    duplicates = [names for names in seen.values() if len(names) > 1]
    assert not duplicates, (
        f"Duplicate operator class bound to multiple names: {duplicates}"
    )
