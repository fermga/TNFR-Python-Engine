"""Canonical operator set invariants.

Ensures registry exposes exactly 13 immutable operator names matching
TNFR physics. Any deviation indicates unintended dynamic mutation.
"""
from __future__ import annotations

from tnfr.operators.registry import OPERATORS

EXPECTED = {
    "emission",
    "reception",
    "coherence",
    "dissonance",
    "coupling",
    "resonance",
    "silence",
    "expansion",
    "contraction",
    "self_organization",
    "mutation",
    "transition",
    "recursivity",
}


def test_canonical_operator_count_and_names():
    assert len(OPERATORS) == 13, "Canonical operator set must be 13"
    assert set(OPERATORS.keys()) == EXPECTED, (
        "Operator names mismatch canonical set"
    )
