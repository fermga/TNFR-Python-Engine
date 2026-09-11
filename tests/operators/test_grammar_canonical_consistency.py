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

CANONICAL_GENERATORS = {"emission", "transition", "recursivity"}  # AL,NAV,REMESH
CANONICAL_CLOSURES = {
    "silence",
    "transition",
    "recursivity",
    "dissonance",
}  # SHA,NAV,REMESH,OZ
CANONICAL_STABILIZERS = {"coherence", "self_organization"}  # IL, THOL
CANONICAL_DESTABILIZERS = {"dissonance", "mutation", "expansion"}  # OZ, ZHIR, VAL
CANONICAL_TRANSFORMERS = {"mutation", "self_organization"}  # ZHIR, THOL
CANONICAL_TRIGGERS = {"dissonance", "mutation"}  # OZ, ZHIR
CANONICAL_HANDLERS = {"self_organization", "coherence"}  # THOL, IL


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


class TestBifurcationWindowEmergent:
    """The U4b window is DERIVED from the pulse relaxation (not a magic 3)."""

    def test_canonical_window_is_three(self) -> None:
        # the derivation evaluates to the canonical U4b window (nu_f=1, dt=0.5)
        from tnfr.config.operator_names import BIFURCATION_WINDOW

        assert pd.derive_bifurcation_window_from_physics() == 3
        assert BIFURCATION_WINDOW == 3

    def test_window_is_pi_band_relaxation(self) -> None:
        # window = steps for the discrete perturbation q^n to relax below the
        # coherence-band fraction 1/(pi+1); q = 1 - nu_f*dt*rho, rho = 1
        import math

        nu_f, dt, rho = 1.0, 0.5, 1.0
        q = 1.0 - nu_f * dt * rho
        band = 1.0 / (math.pi + 1.0)
        n = 1
        while q ** n >= band:
            n += 1
        assert pd.derive_bifurcation_window_from_physics(nu_f, dt) == n

    def test_window_adapts_to_dt(self) -> None:
        # a smaller step => slower per-step relaxation => a larger window
        assert (
            pd.derive_bifurcation_window_from_physics(dt=0.25)
            > pd.derive_bifurcation_window_from_physics(dt=0.5)
        )

    def test_single_emergent_window_for_all_destabilizers(self) -> None:
        # the graduated split is dropped: every destabilizer shares the SINGLE
        # emergent window (topology-independent relaxation, rho = trace/N = 1).
        # The strong/moderate/weak partition and BIFURCATION_WINDOWS dict are
        # removed -- DESTABILIZERS + BIFURCATION_WINDOW are the single sources.
        from tnfr.config import operator_names as on

        assert not hasattr(on, "BIFURCATION_WINDOWS")
        assert not hasattr(on, "DESTABILIZERS_STRONG")
        assert not hasattr(on, "DESTABILIZERS_MODERATE")
        assert not hasattr(on, "DESTABILIZERS_ALL")
        assert on.BIFURCATION_WINDOW == 3

    def test_u2_debt_capacity_is_two(self) -> None:
        # the U2 debt threshold is the geometric relaxation absorption 1/(1-q),
        # the SAME q (= 1 - nu_f*dt*rho) that sets the window -- read as a
        # capacity instead of a time
        from tnfr.config.operator_names import U2_DEBT_CAPACITY

        assert pd.derive_u2_debt_capacity_from_physics() == 2
        assert U2_DEBT_CAPACITY == 2

    def test_u2_debt_is_geometric_absorption(self) -> None:
        # capacity = floor(1/(nu_f*dt*rho)) = floor(1/(1-q)), rho = 1
        nu_f, dt, rho = 1.0, 0.5, 1.0
        relax = nu_f * dt * rho
        assert pd.derive_u2_debt_capacity_from_physics(nu_f, dt) == int(
            1.0 / relax
        )

    def test_grammar_repeat_window_is_the_relaxation_window(self) -> None:
        # the GRAMMAR repeat-avoidance window (don't re-fire a destabilizer
        # before its perturbation relaxes) = the same derived relaxation window
        from tnfr.config.defaults_core import CoreDefaults

        assert (
            CoreDefaults().GRAMMAR["window"]
            == pd.derive_bifurcation_window_from_physics()
        )


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
        # DESTABILIZERS is the single membership source (no NAV, no EN)
        assert "transition" not in on.DESTABILIZERS  # NAV is NOT a destabilizer
        assert "reception" not in on.DESTABILIZERS  # EN is NOT a destabilizer
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
            Coherence,
            Emission,
            Mutation,
            Silence,
            Transition,
        )
        from tnfr.operators.grammar_validate import validate_grammar

        # [AL, IL, NAV, ZHIR, SHA]: NAV does NOT provide U4b context, so ZHIR
        # lacks a recent destabilizer -> the static validator must REJECT.
        seq = [Emission(), Coherence(), Transition(), Mutation(), Silence()]
        assert validate_grammar(seq, epi_initial=0.0) is False

    def test_oz_provides_destabilizer_context(self) -> None:
        from tnfr.operators.definitions import (
            Coherence,
            Dissonance,
            Emission,
            Mutation,
            Silence,
        )
        from tnfr.operators.grammar_validate import validate_grammar

        # [AL, IL, OZ, ZHIR, SHA]: OZ provides U4b context -> valid.
        seq = [Emission(), Coherence(), Dissonance(), Mutation(), Silence()]
        assert validate_grammar(seq, epi_initial=0.0) is True


class TestSecondaryValidatorReadsCanonicalSets:
    """The secondary string-based validator (grammar_patterns.validate_sequence,
    exported as grammar.validate_sequence) must read the SAME canonical
    classification sets as the object-based validator.

    Both validators legitimately have different public contracts (objects->bool
    vs strings->rich result), but the SHARED U1/U2/U4b rules must agree because
    both read config.operator_names (pinned == grammar_types == physics_derivation
    by the tests above).  A June 2026 audit found the secondary validator carried
    a hardcoded inline U2 set {dissonance, mutation, expansion, CONTRACTION} that
    wrongly included NUL; these tests pin that the inline copies are gone and the
    two validators concur on the shared classification rules.
    """

    def _passes(self, names: list[str]) -> bool:
        from tnfr.operators.grammar import validate_sequence

        return validate_sequence(names).passed

    def test_nul_is_not_a_u2_destabilizer(self) -> None:
        # NUL (contraction) is the dual-lever 'both' operator, U2-NEUTRAL.
        # A bare emission->contraction->silence is rejected by the ADJACENT
        # compatibility rule, not U2, so probe NUL in a compatible context:
        # the U2 stabilizer requirement must NOT fire for NUL alone.
        from tnfr.operators.grammar import validate_sequence

        r = validate_sequence(["emission", "coupling", "contraction", "silence"])
        # Whatever the verdict, it must NOT be a missing-stabilizer (U2) failure,
        # because NUL is not a destabilizer.
        assert "missing stabilizer" not in r.message

    def test_canonical_destabilizers_trigger_u2(self) -> None:
        # Each canonical destabilizer {OZ, ZHIR, VAL} without a stabilizer must
        # fail with the U2 missing-stabilizer message.
        from tnfr.operators.grammar import validate_sequence

        for destab in ("dissonance", "expansion"):
            r = validate_sequence(["emission", destab, "silence"])
            assert r.passed is False
            assert "missing stabilizer" in r.message

    def test_destabilizer_with_stabilizer_clears_u2(self) -> None:
        # VAL + IL clears the U2 requirement (does not fail on missing stabilizer).
        assert self._passes(["emission", "expansion", "coherence", "silence"])

    def test_secondary_agrees_with_canonical_on_valid_words(self) -> None:
        # A canonical valid word passes BOTH validators.
        from tnfr.operators.definitions import Coherence, Coupling, Emission, Silence
        from tnfr.operators.grammar_validate import validate_grammar

        names = ["emission", "coupling", "coherence", "silence"]
        ops = [Emission(), Coupling(), Coherence(), Silence()]
        assert self._passes(names) is True
        assert validate_grammar(ops, epi_initial=0.0) is True
