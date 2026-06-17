"""Canonical consistency of the operator-classification sets (single source).

These tests guard the SINGLE SOURCE OF TRUTH for the U1-U6 operator
classification.  Every grammar set is derived from the per-operator
nodal-equation predicates in ``tnfr.config.physics_derivation`` and re-exported
by ``tnfr.operators.grammar_types``.  All other modules that carry a copy of
these sets (config.operator_names graduated taxonomy, math.grammar_validators
glyph sets) must agree with the canonical source — these tests fail loudly if any
of them drifts.

Background: a prior audit (June 2026) found divergent definitions —
config.operator_names listed NAV as a destabilizer and EN as a weak destabilizer,
contradicting the load-bearing validator (which uses {OZ, ZHIR, VAL}).  This
caused the static validator and the runtime mutation precondition to DISAGREE on
whether NAV/EN provide U4b bifurcation context.  These tests pin the resolution.
"""

from __future__ import annotations

from tnfr.config import physics_derivation as pd
from tnfr.operators import grammar_types as gt


# ---------------------------------------------------------------------------
# Canonical values (the absolute truth, derived from the nodal equation)
# ---------------------------------------------------------------------------

CANONICAL_GENERATORS = {"emission", "transition", "recursivity"}        # AL,NAV,REMESH
CANONICAL_CLOSURES = {"silence", "transition", "recursivity", "dissonance"}  # SHA,NAV,REMESH,OZ
CANONICAL_STABILIZERS = {"coherence", "self_organization"}              # IL, THOL
CANONICAL_DESTABILIZERS = {"dissonance", "mutation", "expansion"}       # OZ, ZHIR, VAL
CANONICAL_TRANSFORMERS = {"mutation", "self_organization"}              # ZHIR, THOL
CANONICAL_TRIGGERS = {"dissonance", "mutation"}                         # OZ, ZHIR
CANONICAL_HANDLERS = {"self_organization", "coherence"}                 # THOL, IL


class TestPhysicsDerivationIsTheSource:
    """grammar_types must equal the physics_derivation derivation, exactly."""

    def test_generators(self) -> None:
        assert gt.GENERATORS == pd.derive_start_operators_from_physics()
        assert gt.GENERATORS == CANONICAL_GENERATORS

    def test_closures(self) -> None:
        assert gt.CLOSURES == pd.derive_end_operators_from_physics()
        assert gt.CLOSURES == CANONICAL_CLOSURES

    def test_stabilizers(self) -> None:
        assert gt.STABILIZERS == pd.derive_stabilizers_from_physics()
        assert gt.STABILIZERS == CANONICAL_STABILIZERS

    def test_destabilizers(self) -> None:
        assert gt.DESTABILIZERS == pd.derive_destabilizers_from_physics()
        assert gt.DESTABILIZERS == CANONICAL_DESTABILIZERS

    def test_transformers(self) -> None:
        assert gt.TRANSFORMERS == pd.derive_transformers_from_physics()
        assert gt.TRANSFORMERS == CANONICAL_TRANSFORMERS

    def test_bifurcation_triggers(self) -> None:
        assert gt.BIFURCATION_TRIGGERS == pd.derive_bifurcation_triggers_from_physics()
        assert gt.BIFURCATION_TRIGGERS == CANONICAL_TRIGGERS

    def test_bifurcation_handlers(self) -> None:
        assert gt.BIFURCATION_HANDLERS == pd.derive_bifurcation_handlers_from_physics()
        assert gt.BIFURCATION_HANDLERS == CANONICAL_HANDLERS


class TestPredicateGrounding:
    """The per-operator predicates encode the canonical membership."""

    def test_pressure_predicate(self) -> None:
        for op in CANONICAL_DESTABILIZERS:
            assert pd.increases_structural_pressure(op)
        # NAV (controlled) and EN (integrative) and NUL must NOT destabilize
        for op in ("transition", "reception", "contraction"):
            assert not pd.increases_structural_pressure(op)

    def test_feedback_predicate(self) -> None:
        for op in CANONICAL_STABILIZERS:
            assert pd.provides_negative_feedback(op)
        assert not pd.provides_negative_feedback("dissonance")


class TestNoDivergentModule:
    """Every other module carrying these sets must agree with the source."""

    def test_operator_names_destabilizers_match(self) -> None:
        from tnfr.config import operator_names as on

        assert set(on.DESTABILIZERS) == CANONICAL_DESTABILIZERS
        # graduated split union == canonical (no NAV, no EN)
        assert set(on.DESTABILIZERS_ALL) == CANONICAL_DESTABILIZERS
        assert "transition" not in on.DESTABILIZERS_ALL   # NAV is NOT a destabilizer
        assert "reception" not in on.DESTABILIZERS_ALL    # EN is NOT a destabilizer
        assert set(on.TRANSFORMERS) == CANONICAL_TRANSFORMERS

    def test_math_grammar_validators_match(self) -> None:
        from tnfr.math import grammar_validators as mv

        destab_names = {g.value for g in mv.DESTABILIZERS}
        stab_names = {g.value for g in mv.STABILIZERS}
        # OZ, ZHIR, VAL / IL, THOL (glyph codes)
        assert destab_names == {"OZ", "ZHIR", "VAL"}
        assert stab_names == {"IL", "THOL"}

    def test_telemetry_sets_are_not_grammar(self) -> None:
        # config.constants STABILIZERS/DISRUPTORS are TELEMETRY, distinct from
        # the grammar sets — this pins that they are intentionally different.
        from tnfr.config import constants as c

        assert set(c.STABILIZERS) != CANONICAL_STABILIZERS  # {IL,RA,UM,SHA} telemetry
        assert "NAV" in c.DISRUPTORS  # telemetry disruptors include NAV (timing role)


class TestStaticRuntimeU4bAgree:
    """The static validator and the runtime mutation precondition must agree
    on which operators provide U4b bifurcation context (the fixed bug)."""

    def test_nav_not_a_destabilizer_context(self) -> None:
        from tnfr.operators.definitions import (
            Emission, Coherence, Transition, Mutation, Silence,
        )
        from tnfr.operators.grammar_validate import validate_grammar

        # [AL, IL, NAV, ZHIR, SHA]: NAV does NOT provide U4b context, so ZHIR
        # lacks a recent destabilizer -> the static validator must REJECT.
        seq = [Emission(), Coherence(), Transition(), Mutation(), Silence()]
        assert validate_grammar(seq, epi_initial=0.0) is False

    def test_oz_provides_destabilizer_context(self) -> None:
        from tnfr.operators.definitions import (
            Emission, Coherence, Dissonance, Mutation, Silence,
        )
        from tnfr.operators.grammar_validate import validate_grammar

        # [AL, IL, OZ, ZHIR, SHA]: OZ provides U4b context -> valid.
        seq = [Emission(), Coherence(), Dissonance(), Mutation(), Silence()]
        assert validate_grammar(seq, epi_initial=0.0) is True
