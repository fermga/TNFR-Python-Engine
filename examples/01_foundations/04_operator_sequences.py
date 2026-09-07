"""04 - Operator sequences: grammar and telemetry are separate decisions.

This example sends flat operator words to the canonical symbol/history
validator and compares that Boolean result with an explicitly illustrative
pressure-only response. The response is not an operator execution and is not
used to validate grammar. Its purpose is to show why a favorable instantaneous
telemetry change cannot repair an invalid word, and why a valid exploratory word
need not decrease every diagnostic at every step.

Runtime U3 phase compatibility, nested U5 structure and reference-dependent U6
drift require state beyond this flat symbol/history call.
"""

from __future__ import annotations

from dataclasses import dataclass

from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    Emission,
    Mutation,
    SelfOrganization,
    Silence,
)
from tnfr.operators.grammar_validate import validate_grammar


OPERATORS = {
    "AL": Emission,
    "IL": Coherence,
    "OZ": Dissonance,
    "UM": Coupling,
    "SHA": Silence,
    "THOL": SelfOrganization,
    "ZHIR": Mutation,
}


@dataclass(frozen=True)
class SequenceCase:
    name: str
    glyphs: tuple[str, ...]
    expected_valid: bool
    illustrative_pressure_changes: tuple[float, ...]


CASES = (
    SequenceCase(
        "U1 complete word",
        ("AL", "IL", "SHA"),
        True,
        (-0.10, -0.15, 0.0),
    ),
    SequenceCase(
        "U1 missing generator",
        ("IL", "SHA"),
        False,
        (-0.10, 0.0),
    ),
    SequenceCase(
        "U1 missing closure",
        ("AL", "IL"),
        False,
        (-0.10, -0.15),
    ),
    SequenceCase(
        "U2 balanced destabilizer",
        ("AL", "OZ", "IL", "SHA"),
        True,
        (-0.10, +0.30, -0.25, 0.0),
    ),
    SequenceCase(
        "U2 unpaid destabilizer",
        ("AL", "OZ", "SHA"),
        False,
        (-0.10, +0.30, 0.0),
    ),
    SequenceCase(
        "U4 contextual mutation",
        ("AL", "IL", "OZ", "ZHIR", "THOL", "SHA"),
        True,
        (-0.10, -0.20, +0.25, +0.40, -0.35, 0.0),
    ),
    SequenceCase(
        "U4 mutation without context",
        ("AL", "ZHIR", "SHA"),
        False,
        (-0.10, +0.40, 0.0),
    ),
    SequenceCase(
        "flat coupling word",
        ("AL", "UM", "IL", "SHA"),
        True,
        (-0.10, -0.05, -0.15, 0.0),
    ),
)


def _pressure_proxy(changes: tuple[float, ...]) -> tuple[float, float]:
    """Return before/after values of a one-node pressure-only proxy.

    ``1/(1+|pressure|)`` resembles one factor of structural coherence, but it
    omits nodal velocity and all other nodes. It is deliberately not labelled
    canonical ``C(t)``.
    """

    pressure = 0.30
    before = 1.0 / (1.0 + abs(pressure))
    for change in changes:
        pressure = max(0.01, pressure + change)
    after = 1.0 / (1.0 + abs(pressure))
    return before, after


def _validate(case: SequenceCase) -> bool:
    instances = [OPERATORS[glyph]() for glyph in case.glyphs]
    return bool(validate_grammar(instances, epi_initial=0.0))


def operator_sequences_demo() -> None:
    """Compare canonical flat grammar decisions with independent telemetry."""

    print("=" * 88)
    print("TNFR OPERATOR WORDS: CANONICAL GRAMMAR VS ILLUSTRATIVE TELEMETRY")
    print("=" * 88)
    print("The validator checks flat U1/U2/U4 symbol and history constraints.")
    print("The final column is an independent one-node pressure proxy.")
    print()
    print(
        f"{'case':<31} {'word':<29} {'grammar':>8} "
        f"{'expected':>9} {'proxy delta':>13}"
    )
    print("-" * 96)

    mismatched_decisions = 0
    for case in CASES:
        valid = _validate(case)
        before, after = _pressure_proxy(case.illustrative_pressure_changes)
        proxy_delta = after - before
        assert valid is case.expected_valid
        if valid is not (proxy_delta >= 0.0):
            mismatched_decisions += 1
        print(
            f"{case.name:<31} {' '.join(case.glyphs):<29} "
            f"{str(valid):>8} {str(case.expected_valid):>9} {proxy_delta:>+13.5f}"
        )

    print()
    print(f"Grammar/proxy sign disagreements in this finite table: {mismatched_decisions}")
    print("A positive proxy delta does not make a word valid, and a negative one")
    print("does not make it invalid. Grammar and trajectory telemetry answer")
    print("different questions and must be evaluated independently.")
    print()
    print("Scope of the six rules")
    print("----------------------")
    print("U1: generator and closure boundaries for the declared initial state.")
    print("U2: bounded destabilizer-debt policy; no general Lyapunov theorem.")
    print("U3: phase compatibility on actual coupling/resonance endpoints.")
    print("U4: trigger/handler and transformer-history requirements.")
    print("U5: hierarchy-aware coherence for nested EPI structure.")
    print("U6: before/after structural-potential drift telemetry.")


if __name__ == "__main__":
    operator_sequences_demo()
