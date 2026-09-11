r"""Word-composition equivariance (R1, composition-closure stage).

The diffusion base case (:mod:`tnfr.physics.equivariance`) proves ``L_rw`` is
Γ-equivariant, and the per-operator audit
(:mod:`tnfr.physics.operator_equivariance`) measures each of the 13 canonical
operators equivariant on ``Fix(Γ)`` seeds.  The symmetry-sector theorem for full
TNFR *words* needs the **composition closure**: a grammar word
``W = O_k ∘ … ∘ O_1`` built from equivariant operators is itself equivariant.

**Theorem (composition closure).**  If ``O_i ∘ ρ(g) = ρ(g) ∘ O_i`` for every
automorphism ``g`` and each factor ``O_i``, then
``W ∘ ρ(g) = ρ(g) ∘ W``.  *Proof (induction on word length).*  Length 1 is the
per-operator base case.  Inductive step: for ``W' = O ∘ W`` with ``W, O``
equivariant,
``W' ρ(g) = O (W ρ(g)) = O (ρ(g) W) = (O ρ(g)) W = ρ(g) (O W) = ρ(g) W'``.  ∎

**Corollary (``Fix(Γ)`` preservation).**  An equivariant ``W`` maps ``Fix(Γ)`` into
``Fix(Γ)``: if ``ρ(g)x = x`` for all ``g`` then
``ρ(g) W(x) = W(ρ(g)x) = W(x)``.  So a symmetric configuration cannot be pushed
into the antisymmetric complement ``Fix(Γ)^⊥`` by a grammar word — the wall of
examples 117–122 and the Riemann residual.

The inductive step is **DERIVED** (the algebra above); the base case is the
per-operator **MEASURED** equivariance.  This module measures the composed
residual for canonical words (confirming the closure numerically) and the
``Fix(Γ)`` preservation of a Γ-symmetric application.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from ..alias import get_attr
from .operator_equivariance import (
    _CHANNELS,
    _isolate_graph_caches,
    _test_cases,
)
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
    r"""``(word_residual, worst_prefix_residual, holds)`` — the closure witness.

    Measures the residual of every prefix ``O_1, O_1O_2, …`` and of the full
    word.  ``holds`` is ``True`` when all prefixes (hence the whole word) stay
    within ``tol`` — the numeric image of the induction: equivariant factors
    compose to an equivariant word, the residual never escaping the tolerance.
    """
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
        ("Bootstrap+close", ("AL", "UM", "IL", "SHA"),
         [Emission, Coupling, Coherence, Silence]),
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
        results.append(
            WordEquivarianceResult(label, glyphs, worst, worst < tol)
        )
    return results
