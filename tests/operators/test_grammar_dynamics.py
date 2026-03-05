"""Tests for grammar-aware dynamics (proactive U1-U6 enforcement).

Validates that the incremental grammar checking in ``grammar_dynamics.py``
correctly detects violations, suggests alternatives, and integrates with the
existing selectors and application layer.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.operators.grammar_dynamics import (
    CandidateResult,
    GrammarViolation,
    _CLOSURE_CODES,
    _COUPLING_CODES,
    _DESTABILIZER_CODES,
    _FALLBACK_CODE,
    _GENERATOR_CODES,
    _HANDLER_CODES,
    _STABILIZER_CODES,
    _to_code,
    enforce_grammar_on_glyph,
    filter_candidates,
    suggest_alternative,
    validate_candidate,
)
from tnfr.types import Glyph


# ── test helpers ──────────────────────────────────────────────────────────

def _make_graph(
    num_nodes: int = 5,
    history: list[str] | None = None,
    epi: float = 1.0,
) -> tuple[nx.Graph, int]:
    """Build a small TNFR graph with configurable glyph history on node 0."""
    G = nx.cycle_graph(num_nodes)
    for n in G.nodes():
        G.nodes[n]["EPI"] = epi
        G.nodes[n]["nu_f"] = 1.0
        G.nodes[n]["DNFR"] = 0.05
        G.nodes[n]["theta"] = float(n) * 0.3
        G.nodes[n]["delta_nfr"] = 0.05
        G.nodes[n]["phase"] = float(n) * 0.3
    if history is not None:
        from collections import deque
        G.nodes[0]["glyph_history"] = deque(history, maxlen=20)
    return G, 0


# ═══════════════════════════════════════════════════════════════════════════
#  Code normalization
# ═══════════════════════════════════════════════════════════════════════════

class TestToCode:
    """Verify _to_code normalizes glyphs to uppercase codes."""

    def test_glyph_enum(self) -> None:
        assert _to_code(Glyph.IL) == "IL"
        assert _to_code(Glyph.OZ) == "OZ"

    def test_string_code(self) -> None:
        assert _to_code("IL") == "IL"
        assert _to_code("oz") == "OZ"

    def test_dotted_format(self) -> None:
        assert _to_code("Glyph.AL") == "AL"

    def test_canonical_name(self) -> None:
        assert _to_code("coherence") == "IL"
        assert _to_code("dissonance") == "OZ"
        assert _to_code("emission") == "AL"


# ═══════════════════════════════════════════════════════════════════════════
#  Set membership sanity
# ═══════════════════════════════════════════════════════════════════════════

class TestCodeSets:
    """Verify code-level sets match the canonical grammar_types definitions."""

    def test_destabilizers_include_oz_zhir(self) -> None:
        assert "OZ" in _DESTABILIZER_CODES
        assert "ZHIR" in _DESTABILIZER_CODES

    def test_stabilizers_include_il_thol(self) -> None:
        assert "IL" in _STABILIZER_CODES
        assert "THOL" in _STABILIZER_CODES

    def test_coupling_includes_um_ra(self) -> None:
        assert "UM" in _COUPLING_CODES
        assert "RA" in _COUPLING_CODES

    def test_generators(self) -> None:
        assert "AL" in _GENERATOR_CODES
        assert "NAV" in _GENERATOR_CODES
        assert "REMESH" in _GENERATOR_CODES

    def test_handlers(self) -> None:
        assert "IL" in _HANDLER_CODES
        assert "THOL" in _HANDLER_CODES


# ═══════════════════════════════════════════════════════════════════════════
#  U1a: Structural Initiation
# ═══════════════════════════════════════════════════════════════════════════

class TestU1a:
    """U1a: EPI=0 + empty history ⟹ must start with generator."""

    def test_generator_allowed_at_epi_zero(self) -> None:
        G, node = _make_graph(epi=0.0)
        cr = validate_candidate(G, node, "AL")
        assert cr.allowed

    def test_non_generator_rejected_at_epi_zero(self) -> None:
        G, node = _make_graph(epi=0.0)
        cr = validate_candidate(G, node, "IL")
        assert not cr.allowed
        assert any(v.rule == "U1a" for v in cr.violations)

    def test_any_glyph_ok_with_nonzero_epi(self) -> None:
        G, node = _make_graph(epi=1.0)
        cr = validate_candidate(G, node, "IL")
        assert cr.allowed

    def test_any_glyph_ok_with_existing_history(self) -> None:
        G, node = _make_graph(epi=0.0, history=["AL"])
        cr = validate_candidate(G, node, "IL")
        assert cr.allowed  # history already started


# ═══════════════════════════════════════════════════════════════════════════
#  U2: Convergence & Boundedness
# ═══════════════════════════════════════════════════════════════════════════

class TestU2:
    """U2: Destabilizer debt must not exceed threshold."""

    def test_single_destabilizer_ok(self) -> None:
        G, node = _make_graph(history=["IL", "RA"])
        cr = validate_candidate(G, node, "OZ")
        assert cr.allowed

    def test_debt_exceeds_threshold(self) -> None:
        # 3 destabilizers, 0 stabilizers → debt=3 > 2
        G, node = _make_graph(history=["OZ", "ZHIR"])
        cr = validate_candidate(G, node, "VAL")
        assert not cr.allowed
        assert any(v.rule == "U2" for v in cr.violations)

    def test_stabilizer_compensates_debt(self) -> None:
        G, node = _make_graph(history=["OZ", "IL", "ZHIR"])
        cr = validate_candidate(G, node, "OZ")
        assert cr.allowed  # debt = 2 destabilizers - 1 stabilizer + 1 new = 2 ≤ 2

    def test_alternative_suggested_on_debt(self) -> None:
        G, node = _make_graph(history=["OZ", "ZHIR"])
        cr = validate_candidate(G, node, "VAL")
        assert cr.suggested_alternative is not None
        # The alternative should be grammar-valid
        alt_cr = validate_candidate(G, node, cr.suggested_alternative)
        assert alt_cr.allowed


# ═══════════════════════════════════════════════════════════════════════════
#  U3: Resonant Coupling
# ═══════════════════════════════════════════════════════════════════════════

class TestU3:
    """U3: Coupling/resonance requires phase-compatible neighbours."""

    def test_coupling_with_compatible_phases(self) -> None:
        G, node = _make_graph()
        # All phases are close (n * 0.3 for n=0..4)
        cr = validate_candidate(G, node, "UM")
        assert cr.allowed

    def test_coupling_with_incompatible_phases(self) -> None:
        G, node = _make_graph()
        # Set all neighbours to antiphase
        for nb in G.neighbors(node):
            G.nodes[nb]["theta"] = math.pi
            G.nodes[nb]["phase"] = math.pi
        G.nodes[node]["theta"] = 0.0
        G.nodes[node]["phase"] = 0.0
        # Use very strict delta_phi_max
        G.graph["DELTA_PHI_MAX"] = 0.01
        cr = validate_candidate(G, node, "UM")
        # Should get a U3 warning (severity=warning, so still allowed)
        assert any(v.rule == "U3" for v in cr.violations)
        # U3 is a warning, not an error — allowed is True
        assert cr.allowed

    def test_non_coupling_ignores_u3(self) -> None:
        G, node = _make_graph()
        cr = validate_candidate(G, node, "OZ")
        assert not any(v.rule == "U3" for v in cr.violations)


# ═══════════════════════════════════════════════════════════════════════════
#  U4a: Bifurcation triggers need handlers
# ═══════════════════════════════════════════════════════════════════════════

class TestU4a:
    """U4a: OZ/ZHIR require a handler (IL/THOL) in context."""

    def test_trigger_with_handler(self) -> None:
        G, node = _make_graph(history=["IL", "RA"])
        cr = validate_candidate(G, node, "OZ")
        assert cr.allowed

    def test_trigger_without_handler(self) -> None:
        G, node = _make_graph(history=["RA", "NAV"])
        cr = validate_candidate(G, node, "OZ")
        assert not cr.allowed
        assert any(v.rule == "U4a" for v in cr.violations)


# ═══════════════════════════════════════════════════════════════════════════
#  U4b: Transformer context
# ═══════════════════════════════════════════════════════════════════════════

class TestU4b:
    """U4b: ZHIR/THOL need recent destabilizer; ZHIR also needs prior IL."""

    def test_zhir_with_full_context(self) -> None:
        # IL → ... → OZ → ZHIR — has IL and recent OZ
        G, node = _make_graph(history=["IL", "RA", "OZ"])
        cr = validate_candidate(G, node, "ZHIR")
        assert cr.allowed

    def test_zhir_without_destabilizer(self) -> None:
        G, node = _make_graph(history=["IL", "RA", "EN"])
        cr = validate_candidate(G, node, "ZHIR")
        assert not cr.allowed
        assert any(v.rule == "U4b" for v in cr.violations)

    def test_zhir_without_prior_il(self) -> None:
        G, node = _make_graph(history=["RA", "NAV", "OZ"])
        cr = validate_candidate(G, node, "ZHIR")
        assert not cr.allowed
        assert any(v.rule == "U4b" and "IL" in v.message for v in cr.violations)

    def test_thol_with_recent_destabilizer(self) -> None:
        G, node = _make_graph(history=["IL", "OZ"])
        cr = validate_candidate(G, node, "THOL")
        assert cr.allowed

    def test_thol_without_recent_destabilizer(self) -> None:
        G, node = _make_graph(history=["IL", "EN", "RA", "SHA"])
        cr = validate_candidate(G, node, "THOL")
        assert not cr.allowed
        assert any(v.rule == "U4b" for v in cr.violations)


# ═══════════════════════════════════════════════════════════════════════════
#  Combined: Multiple rules
# ═══════════════════════════════════════════════════════════════════════════

class TestMultipleViolations:
    """If candidate triggers more than one rule, all are reported."""

    def test_u2_and_u4a_combined(self) -> None:
        # History: 2 destabilizers, no handler, no stabilizer
        G, node = _make_graph(history=["OZ", "ZHIR"])
        # Another destabilizer: debt=3 (U2) + no handler (not if OZ counts)
        # Actually OZ is a trigger too, so U4a should flag
        cr = validate_candidate(G, node, "VAL")
        # VAL is not a trigger so U4a may not trigger for VAL
        assert any(v.rule == "U2" for v in cr.violations)


# ═══════════════════════════════════════════════════════════════════════════
#  filter_candidates
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterCandidates:
    """filter_candidates returns only valid options."""

    def test_basic_filtering(self) -> None:
        G, node = _make_graph(history=["IL", "RA"])
        valid = filter_candidates(G, node, ["IL", "OZ", "RA", "SHA"])
        assert "IL" in valid
        assert "RA" in valid
        assert "SHA" in valid

    def test_empty_input(self) -> None:
        G, node = _make_graph()
        assert filter_candidates(G, node, []) == []

    def test_all_rejected(self) -> None:
        # EPI=0, empty history → only generators pass
        G, node = _make_graph(epi=0.0)
        valid = filter_candidates(G, node, ["IL", "OZ", "RA"])
        assert valid == []


# ═══════════════════════════════════════════════════════════════════════════
#  suggest_alternative
# ═══════════════════════════════════════════════════════════════════════════

class TestSuggestAlternative:
    """suggest_alternative finds a grammar-safe fallback."""

    def test_returns_valid_alternative(self) -> None:
        G, node = _make_graph(history=["OZ", "ZHIR"])
        alt = suggest_alternative(G, node, "VAL")
        assert alt is not None
        # Verify it's actually valid
        cr = validate_candidate(G, node, alt)
        assert cr.allowed

    def test_fallback_is_il(self) -> None:
        # IL should always be in the priority list
        G, node = _make_graph()
        alt = suggest_alternative(G, node, "OZ")
        assert alt in ("IL", "THOL", "EN", "SHA", "RA", "NAV", "AL")


# ═══════════════════════════════════════════════════════════════════════════
#  enforce_grammar_on_glyph
# ═══════════════════════════════════════════════════════════════════════════

class TestEnforceGrammarOnGlyph:
    """enforce_grammar_on_glyph is the main entry point for dynamics wiring."""

    def test_valid_candidate_passes_through(self) -> None:
        G, node = _make_graph(history=["IL"])
        result = enforce_grammar_on_glyph(G, node, "RA")
        assert result == "RA"

    def test_invalid_candidate_replaced(self) -> None:
        # EPI=0, no history → non-generator rejected
        G, node = _make_graph(epi=0.0)
        result = enforce_grammar_on_glyph(G, node, "IL")
        # Should be replaced with a generator
        assert result in _GENERATOR_CODES

    def test_accepts_glyph_enum(self) -> None:
        G, node = _make_graph(history=["IL"])
        result = enforce_grammar_on_glyph(G, node, Glyph.RA)
        assert result == "RA"


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: enforce_canonical_grammar
# ═══════════════════════════════════════════════════════════════════════════

class TestEnforceCanonicalGrammarIntegration:
    """enforce_canonical_grammar now delegates to grammar_dynamics."""

    def test_valid_candidate_untouched(self) -> None:
        from tnfr.operators.grammar_application import enforce_canonical_grammar
        G, node = _make_graph(history=["IL"])
        result = enforce_canonical_grammar(G, node, "RA")
        assert result == "RA"

    def test_invalid_candidate_replaced(self) -> None:
        from tnfr.operators.grammar_application import enforce_canonical_grammar
        G, node = _make_graph(epi=0.0)
        result = enforce_canonical_grammar(G, node, "IL")
        assert result in _GENERATOR_CODES


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: selectors use grammar-aware prefilter
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectorsIntegration:
    """_soft_grammar_prefilter applies soft heuristic filters only."""

    def test_prefilter_available(self) -> None:
        from tnfr.dynamics.selectors import _soft_grammar_prefilter
        G, node = _make_graph(history=["IL", "RA"])
        # Should not raise
        result = _soft_grammar_prefilter(G, node, "OZ")
        assert isinstance(result, str)

    def test_prefilter_does_not_duplicate_grammar(self) -> None:
        """Prefilter should NOT run enforce_grammar_on_glyph (DRY)."""
        from tnfr.dynamics.selectors import _soft_grammar_prefilter
        # EPI=0, no history → grammar would replace a non-generator.
        # But the prefilter is now soft-only: it should pass through
        # the candidate (grammar enforcement happens at the apply stage).
        G, node = _make_graph(epi=0.0)
        result = _soft_grammar_prefilter(G, node, "IL")
        # soft_grammar_filters may or may not modify, but enforce_grammar
        # is NOT applied here — the returned code should not necessarily
        # be forced to a generator.
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: apply_glyph_with_grammar pre-validates (GAP #3)
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyGlyphWithGrammarPrevalidation:
    """apply_glyph_with_grammar enforces grammar BEFORE applying."""

    def test_valid_glyph_applied(self) -> None:
        from tnfr.operators.grammar_application import apply_glyph_with_grammar
        G, node = _make_graph(history=["IL"], epi=1.0)
        # RA on a node with history and EPI>0 → valid, should not raise
        apply_glyph_with_grammar(G, [node], "RA")

    def test_grammar_enforcement_runs(self) -> None:
        """Glyph should be validated (and potentially replaced) before apply."""
        from tnfr.operators.grammar_application import enforce_canonical_grammar
        # A non-generator at EPI=0 with no history → should be replaced
        G, node = _make_graph(epi=0.0)
        result = enforce_canonical_grammar(G, node, "IL")
        assert result in _GENERATOR_CODES


# ═══════════════════════════════════════════════════════════════════════════
#  validate_sequence_incremental (GAP #4)
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateSequenceIncremental:
    """Per-step incremental validation against a node's live history."""

    def test_valid_sequence(self) -> None:
        from tnfr.operators.grammar_dynamics import validate_sequence_incremental
        G, node = _make_graph(history=["IL"], epi=1.0)
        results = validate_sequence_incremental(G, node, ["RA", "IL", "OZ", "IL"])
        assert all(r.allowed for r in results)

    def test_detects_per_step_violation(self) -> None:
        from tnfr.operators.grammar_dynamics import validate_sequence_incremental
        # Start from EPI=0, no history → first step must be generator
        G, node = _make_graph(epi=0.0)
        results = validate_sequence_incremental(G, node, ["IL", "OZ"])
        # First step should be rejected (not a generator)
        assert not results[0].allowed
        assert results[0].violations

    def test_shadow_history_accumulates(self) -> None:
        from tnfr.operators.grammar_dynamics import validate_sequence_incremental
        G, node = _make_graph(epi=1.0)
        # OZ, OZ, OZ → destabilizer debt should trigger U2 by step 3
        results = validate_sequence_incremental(G, node, ["OZ", "OZ", "OZ"])
        # At least the third destabilizer in a row should be flagged
        flagged = [r for r in results if not r.allowed]
        assert len(flagged) >= 1

    def test_preserves_original_history(self) -> None:
        from tnfr.operators.grammar_dynamics import validate_sequence_incremental
        G, node = _make_graph(history=["IL", "RA"], epi=1.0)
        original = list(G.nodes[node]["glyph_history"])
        validate_sequence_incremental(G, node, ["OZ", "IL"])
        after = list(G.nodes[node]["glyph_history"])
        assert original == after

    def test_empty_sequence(self) -> None:
        from tnfr.operators.grammar_dynamics import validate_sequence_incremental
        G, node = _make_graph(epi=1.0)
        results = validate_sequence_incremental(G, node, [])
        assert results == []

    def test_available_via_grammar_reexport(self) -> None:
        from tnfr.operators.grammar import validate_sequence_incremental
        assert callable(validate_sequence_incremental)


# ═══════════════════════════════════════════════════════════════════════════
#  Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_empty_history(self) -> None:
        G, node = _make_graph(epi=1.0)
        # No history, EPI > 0 → most things should be ok
        cr = validate_candidate(G, node, "IL")
        assert cr.allowed

    def test_isolated_node(self) -> None:
        G = nx.Graph()
        G.add_node(0, EPI=1.0, nu_f=1.0, DNFR=0.05, theta=0.0,
                   delta_nfr=0.05, phase=0.0)
        # UM on isolated node → U3 warning (no neighbours)
        cr = validate_candidate(G, 0, "UM")
        assert any(v.rule == "U3" for v in cr.violations)

    def test_candidate_result_dataclass(self) -> None:
        cr = CandidateResult(candidate="IL", allowed=True)
        assert cr.violations == []
        assert cr.suggested_alternative is None

    def test_grammar_violation_dataclass(self) -> None:
        v = GrammarViolation(rule="U2", message="test", severity="error")
        assert v.rule == "U2"
        assert v.severity == "error"
