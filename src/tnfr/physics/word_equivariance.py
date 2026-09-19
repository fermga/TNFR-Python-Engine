r"""Conditional composition algebra and finite pointed-word probes (R1).

If every map O_i is equivariant on its complete compatible domain, their
composition W is equivariant: W P_sigma=P_sigma W follows by induction.
An invariant input then stays in the fixed set. Grammar admission alone does
not supply these hypotheses, and finite pointed per-operator residuals do not
prove them for all intermediate states, selectors or histories.

This module independently compares selected words and prefixes at corresponding
selected nodes on invariant fixtures. It reads the audited scalar channels on
original nodes. Its booleans describe those finite probes; no catalog-wide
symmetry theorem, arbitrary error accumulation bound or placement of analytic
Riemann S(T) in a finite symmetry complement follows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from ..alias import get_attr
from .operator_equivariance import _CHANNELS, _isolate_graph_caches, _test_cases
from .symmetry_sectors import is_orbit_constant

__all__ = [
    "WordEquivarianceResult",
    "word_equivariance_residual",
    "composition_closure_holds",
    "word_preserves_fix",
    "canonical_words",
    "audit_word_equivariance",
]


def _apply_word(word, H, node) -> None:
    """Apply the operator sequence ``word`` (classes) at ``node`` on ``H``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cls in word:
            cls()(H, node)


def word_equivariance_residual(word, G, sigma: dict, node, *, channels=None):
    r"""Residual ``max_{u,ch} |A[u] − B[σ(u)]|`` for ``W@node`` vs ``W@σ(node)``.

    ``A`` applies the whole word ``W`` at ``node``; ``B`` applies it at
    ``σ(node)``.  If every factor is equivariant and the seed is σ-invariant then
    ``W@σ(node) = σ(W@node)`` (composition closure), so the residual is ``0``.
    Each copy's inherited content-keyed caches are isolated so no isomorphic
    sibling can leak its ΔNFR prep (the R1-T01 fix).
    """
    from ..dynamics import default_compute_delta_nfr

    channels = _CHANNELS if channels is None else channels
    A = G.copy()
    B = G.copy()
    _isolate_graph_caches(A)
    _isolate_graph_caches(B)
    _apply_word(word, A, node)
    _apply_word(word, B, sigma[node])
    default_compute_delta_nfr(A)
    default_compute_delta_nfr(B)
    resid = 0.0
    for u in G.nodes():
        su = sigma[u]
        for ch in channels:
            a = float(get_attr(A.nodes[u], ch, 0.0))
            b = float(get_attr(B.nodes[su], ch, 0.0))
            resid = max(resid, abs(a - b))
    return resid


def composition_closure_holds(word, G, sigma: dict, node, *, tol: float = 1e-6):
    r"""Return the word residual, worst prefix residual and finite tolerance verdict.

    Every selected prefix is probed independently. Passing them is numerical
    evidence for this supplied word/fixture, not a proof of the all-state
    premise of the exact composition theorem."""
    worst_prefix = 0.0
    for k in range(1, len(word) + 1):
        r = word_equivariance_residual(word[:k], G, sigma, node)
        worst_prefix = max(worst_prefix, r)
    full = word_equivariance_residual(word, G, sigma, node)
    return full, worst_prefix, worst_prefix < tol and full < tol


def word_preserves_fix(word, G, *, tol: float = 1e-6):
    r"""``(is_fixed, residual)`` — ``Fix(Γ)`` preservation of a Γ-symmetric word.

    Applies ``W`` as a Γ-equivariant sweep (each factor at **every** node) on the
    orbit-constant seed, then checks the result is still orbit-constant
    (``∈ Fix(Γ)``).  This is the corollary: an equivariant word cannot move a
    symmetric state into ``Fix(Γ)^⊥``.
    """
    from ..dynamics import default_compute_delta_nfr

    H = G.copy()
    _isolate_graph_caches(H)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cls in word:
            for nd in list(H.nodes()):
                cls()(H, nd)
    default_compute_delta_nfr(H)
    residual = 0.0
    fixed = True
    for ch in _CHANNELS:
        field = {nd: float(get_attr(H.nodes[nd], ch, 0.0)) for nd in H.nodes()}
        if not is_orbit_constant(G, field, tol=tol):
            fixed = False
        vals = list(field.values())
        # orbit-constant proxy residual on a vertex-transitive seed: spread
        residual = max(residual, max(vals) - min(vals) if vals else 0.0)
    return fixed, residual


@dataclass(frozen=True)
class WordEquivarianceResult:
    r"""Per-word equivariance verdict on the audit test cases."""

    label: str
    glyphs: tuple[str, ...]
    residual: float
    is_equivariant: bool


def canonical_words():
    r"""Canonical grammar fragments + closed words (name, glyphs, classes).

    The fragments are the AGENTS.md building blocks (Bootstrap, Stabilize,
    Propagate, Explore); the closed words add a U1b closure so the audit covers
    both bare fragments and grammar-complete words.
    """
    from ..operators.definitions import (
        Coherence,
        Coupling,
        Dissonance,
        Emission,
        Mutation,
        Resonance,
        Silence,
    )

    return [
        ("Bootstrap", ("AL", "UM", "IL"), [Emission, Coupling, Coherence]),
        (
            "Bootstrap+close",
            ("AL", "UM", "IL", "SHA"),
            [Emission, Coupling, Coherence, Silence],
        ),
        ("Stabilize", ("IL", "SHA"), [Coherence, Silence]),
        ("Propagate", ("RA", "UM"), [Resonance, Coupling]),
        ("Explore", ("OZ", "ZHIR", "IL"), [Dissonance, Mutation, Coherence]),
    ]


def audit_word_equivariance(*, tol: float = 1e-6):
    r"""Audit the canonical words for composition-closed equivariance.

    Returns one :class:`WordEquivarianceResult` per word, reporting the worst
    residual over the σ-invariant test cases (cycle rotation, star leaf swap).
    """
    cases = _test_cases()
    results: list[WordEquivarianceResult] = []
    for label, glyphs, word in canonical_words():
        worst = 0.0
        for G, sigma, node in cases:
            try:
                r = word_equivariance_residual(word, G, sigma, node)
            except Exception:
                r = float("inf")
            worst = max(worst, r)
        results.append(WordEquivarianceResult(label, glyphs, worst, worst < tol))
    return results
