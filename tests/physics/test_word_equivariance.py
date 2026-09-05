r"""Tests for word-composition equivariance (R1, N06).

Composition closure: a grammar word built from equivariant operators is itself
equivariant (``W ρ(g) = ρ(g) W``), so it preserves the ``Fix(Γ)`` / ``Fix(Γ)^⊥``
split. The inductive step is derived; the per-operator base case is measured.
"""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Emission,
    Resonance,
    Silence,
)
from tnfr.physics import word_equivariance as we
from tnfr.physics.operator_equivariance import _test_cases
from tnfr.physics.word_equivariance import (
    WordEquivarianceResult,
    audit_word_equivariance,
    canonical_words,
    composition_closure_holds,
    word_equivariance_residual,
    word_preserves_fix,
)


def _cycle_case():
    return _test_cases()[0]  # (cycle C6, rotation sigma, node)


# --------------------------------------------------------------------------- #
# Composition closure: canonical words are equivariant
# --------------------------------------------------------------------------- #
def test_all_canonical_words_are_equivariant():
    results = audit_word_equivariance()
    assert len(results) == len(canonical_words())
    for r in results:
        assert isinstance(r, WordEquivarianceResult)
        assert r.is_equivariant, f"{r.label} residual {r.residual}"
        assert r.residual < 1e-6


def test_bootstrap_word_residual_is_zero():
    G, sigma, node = _cycle_case()
    r = word_equivariance_residual([Emission, Coupling, Coherence], G, sigma,
                                   node)
    assert r == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# The induction: every prefix stays within tolerance
# --------------------------------------------------------------------------- #
def test_composition_closure_holds_over_prefixes():
    G, sigma, node = _cycle_case()
    word = [Emission, Coupling, Coherence, Silence]
    full, worst_prefix, holds = composition_closure_holds(word, G, sigma, node)
    assert holds
    assert full < 1e-6
    assert worst_prefix < 1e-6


def test_closure_matches_single_operator_base_case():
    # a length-1 "word" reduces to the per-operator residual (base case)
    G, sigma, node = _cycle_case()
    from tnfr.physics.operator_equivariance import (
        operator_equivariance_residual,
    )
    word_r = word_equivariance_residual([Coherence], G, sigma, node)
    op_r = operator_equivariance_residual(Coherence(), G, sigma, node)
    assert word_r == pytest.approx(op_r, abs=1e-9)


# --------------------------------------------------------------------------- #
# Corollary: Fix(Γ) preservation under a Γ-symmetric word
# --------------------------------------------------------------------------- #
def test_word_preserves_fix_on_vertex_transitive_seed():
    G, _, _ = _cycle_case()
    fixed, spread = word_preserves_fix([Emission, Coupling, Coherence, Silence],
                                       G)
    assert fixed
    assert spread == pytest.approx(0.0, abs=1e-6)


def test_longer_word_still_preserves_fix():
    G, _, _ = _cycle_case()
    word = [Emission, Resonance, Coupling, Coherence, Silence]
    fixed, spread = word_preserves_fix(word, G)
    assert fixed
    assert spread < 1e-6


# --------------------------------------------------------------------------- #
# Relabel invariance of the audit
# --------------------------------------------------------------------------- #
def test_word_equivariance_relabel_invariant():
    # relabel the cycle: residual must be unchanged (equivariance is intrinsic)
    G, sigma, node = _cycle_case()
    mapping = {i: (i + 3) % 6 for i in range(6)}
    H = nx.relabel_nodes(G, mapping, copy=True)
    hsigma = {mapping[u]: mapping[v] for u, v in sigma.items()}
    word = [Emission, Coupling, Coherence]
    base = word_equivariance_residual(word, G, sigma, node)
    relabelled = word_equivariance_residual(word, H, hsigma, mapping[node])
    assert relabelled == pytest.approx(base, abs=1e-9)


def test_module_exports_complete():
    expected = {
        "WordEquivarianceResult", "word_equivariance_residual",
        "composition_closure_holds", "word_preserves_fix", "canonical_words",
        "audit_word_equivariance",
    }
    assert expected <= set(we.__all__)
