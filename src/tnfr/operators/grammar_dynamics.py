"""Grammar-Aware Dynamics: Proactive U1-U6 enforcement in operator selection.

Bridges the grammar validation system (U1-U6) with the dynamic operator
selection layer.  Provides **incremental** grammar checking for step-by-step
dynamics, where operators are selected one at a time per node.

Physics basis
-------------
Grammar rules derive from the nodal equation ∂EPI/∂t = νf · ΔNFR(t).
Proactive enforcement prevents grammar violations *before* they corrupt graph
state, rather than detecting them reactively after the damage is done.

Incremental rule applicability
------------------------------
- **U1a** (Initiation): Checked when EPI ≈ 0 and history is empty.
- **U2**  (Convergence): Tracked via a destabilizer/stabilizer debt counter
  over a sliding window of recent history.
- **U3**  (Resonant Coupling): Phase compatibility required for UM/RA candidates.
- **U4a** (Bifurcation triggers): OZ/ZHIR require handlers in recent context.
- **U4b** (Transformer context): ZHIR/THOL need a recent destabilizer (and
  prior IL for ZHIR).
- **U1b** (Closure) and **U5/U6** are whole-sequence or telemetry checks and
  cannot be fully enforced incrementally; they are advisory here.

References
----------
- AGENTS.md §Unified Grammar (U1-U6)
- UNIFIED_GRAMMAR_RULES.md — complete derivations
- grammar_core.py — batch validator (full sequences)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .grammar_types import (
    CLOSURES,
    COUPLING_RESONANCE,
    DESTABILIZERS,
    GENERATORS,
    STABILIZERS,
    BIFURCATION_TRIGGERS,
    BIFURCATION_HANDLERS,
    TRANSFORMERS,
    GLYPH_TO_FUNCTION,
)
from ..types import Glyph

# ── glyph code ↔ canonical function name helpers ──────────────────────────

# Build fast lookup from glyph code string ("IL") → canonical name ("coherence")
_CODE_TO_NAME: dict[str, str] = {
    g.value: name for g, name in GLYPH_TO_FUNCTION.items()
}
# Reverse: canonical name → glyph code string
_NAME_TO_CODE: dict[str, str] = {
    name: g.value for g, name in GLYPH_TO_FUNCTION.items()
}

# Sets of *glyph codes* (uppercase) for fast membership tests
_DESTABILIZER_CODES = frozenset(_NAME_TO_CODE[n] for n in DESTABILIZERS if n in _NAME_TO_CODE)
_STABILIZER_CODES = frozenset(_NAME_TO_CODE[n] for n in STABILIZERS if n in _NAME_TO_CODE)
_COUPLING_CODES = frozenset(_NAME_TO_CODE[n] for n in COUPLING_RESONANCE if n in _NAME_TO_CODE)
_GENERATOR_CODES = frozenset(_NAME_TO_CODE[n] for n in GENERATORS if n in _NAME_TO_CODE)
_BIFURCATION_TRIGGER_CODES = frozenset(_NAME_TO_CODE[n] for n in BIFURCATION_TRIGGERS if n in _NAME_TO_CODE)
_HANDLER_CODES = frozenset(_NAME_TO_CODE[n] for n in BIFURCATION_HANDLERS if n in _NAME_TO_CODE)
_TRANSFORMER_CODES = frozenset(_NAME_TO_CODE[n] for n in TRANSFORMERS if n in _NAME_TO_CODE)
_CLOSURE_CODES = frozenset(_NAME_TO_CODE[n] for n in CLOSURES if n in _NAME_TO_CODE)

# Canonical fallback when all else fails — IL (Coherence) is always safe
_FALLBACK_CODE = "IL"

# How many recent glyphs to consider for incremental grammar context
_DEFAULT_WINDOW = 6


# ── data structures ───────────────────────────────────────────────────────

@dataclass(slots=True)
class GrammarViolation:
    """A single incremental grammar rule violation."""
    rule: str          # e.g. "U2", "U4b"
    message: str       # human-readable explanation
    severity: str      # "error" (must fix) or "warning" (advisory)


@dataclass(slots=True)
class CandidateResult:
    """Result of validating a single candidate glyph against recent history."""
    candidate: str
    allowed: bool
    violations: list[GrammarViolation] = field(default_factory=list)
    suggested_alternative: str | None = None


# ── core incremental checks ──────────────────────────────────────────────

def _to_code(glyph: Any) -> str:
    """Normalize a glyph to its uppercase code string (e.g. 'IL')."""
    if isinstance(glyph, Glyph):
        return glyph.value
    s = str(glyph).strip()
    # Handle "Glyph.AL" format
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    upper = s.upper()
    if upper in _CODE_TO_NAME:
        return upper
    # Try canonical name → code
    lower = s.lower()
    if lower in _NAME_TO_CODE:
        return _NAME_TO_CODE[lower]
    return upper  # best effort


def _recent_codes(G: Any, node: Any, window: int = _DEFAULT_WINDOW) -> list[str]:
    """Extract the last *window* glyph codes from the node's history."""
    nd = G.nodes[node]
    raw = nd.get("glyph_history")
    if not raw:
        return []
    items = list(raw)[-window:]
    return [_to_code(g) for g in items]


def _check_u1a(
    candidate: str,
    history: list[str],
    epi: float,
) -> GrammarViolation | None:
    """U1a: If EPI ≈ 0 and history is empty, first glyph must be a generator."""
    if history or epi > 0.0:
        return None  # not applicable
    if candidate not in _GENERATOR_CODES:
        return GrammarViolation(
            rule="U1a",
            message=(
                f"EPI=0 with empty history requires a generator "
                f"({sorted(_GENERATOR_CODES)}), got '{candidate}'."
            ),
            severity="error",
        )
    return None


def _check_u2(
    candidate: str,
    history: list[str],
) -> GrammarViolation | None:
    """U2: Convergence — destabilizer debt must not grow unbounded.

    Counts destabilizers vs stabilizers in recent history + candidate.
    If adding the candidate would leave >2 uncompensated destabilizers,
    flag a violation.
    """
    full = history + [candidate]
    destab = sum(1 for g in full if g in _DESTABILIZER_CODES)
    stab = sum(1 for g in full if g in _STABILIZER_CODES)
    debt = destab - stab
    if debt > 2:
        return GrammarViolation(
            rule="U2",
            message=(
                f"Convergence violation: {destab} destabilizers vs "
                f"{stab} stabilizers in recent window (debt={debt}). "
                f"Add a stabilizer (IL/THOL/EN) before more destabilizers."
            ),
            severity="error",
        )
    return None


def _check_u3(
    candidate: str,
    G: Any,
    node: Any,
) -> GrammarViolation | None:
    """U3: Coupling/resonance candidates require phase-compatible neighbours."""
    if candidate not in _COUPLING_CODES:
        return None
    try:
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_THETA
        theta_i = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))
        delta_phi_max = float(G.graph.get("DELTA_PHI_MAX", 1.5))
        for nb in G.neighbors(node):
            theta_j = float(get_attr(G.nodes[nb], ALIAS_THETA, 0.0))
            diff = abs(theta_i - theta_j)
            # Wrap to [0, π]
            import math
            diff = min(diff, 2 * math.pi - diff)
            if diff <= delta_phi_max:
                return None  # at least one compatible neighbour
        return GrammarViolation(
            rule="U3",
            message=(
                f"No phase-compatible neighbour for {candidate} "
                f"(all |φᵢ - φⱼ| > Δφ_max={delta_phi_max:.2f})."
            ),
            severity="warning",
        )
    except Exception:
        return None  # graceful degradation


def _check_u4a(
    candidate: str,
    history: list[str],
) -> GrammarViolation | None:
    """U4a: Bifurcation triggers need handlers in nearby context."""
    if candidate not in _BIFURCATION_TRIGGER_CODES:
        return None
    # Check if there's a handler anywhere in recent history or candidate itself
    full = history + [candidate]
    has_handler = any(g in _HANDLER_CODES for g in full)
    if not has_handler:
        return GrammarViolation(
            rule="U4a",
            message=(
                f"Bifurcation trigger '{candidate}' has no handler "
                f"({sorted(_HANDLER_CODES)}) in recent context."
            ),
            severity="error",
        )
    return None


def _check_u4b(
    candidate: str,
    history: list[str],
) -> GrammarViolation | None:
    """U4b: Transformers need recent destabilizer context.

    ZHIR also requires a prior IL (stable base).
    """
    if candidate not in _TRANSFORMER_CODES:
        return None
    # Look for a destabilizer in the last 3 operations
    recent = history[-3:] if len(history) >= 3 else history
    has_destab = any(g in _DESTABILIZER_CODES for g in recent)
    if not has_destab:
        return GrammarViolation(
            rule="U4b",
            message=(
                f"Transformer '{candidate}' requires a recent destabilizer "
                f"(within ~3 ops). Recent: {recent}."
            ),
            severity="error",
        )
    # ZHIR additionally requires prior IL
    if candidate == "ZHIR":
        has_il = "IL" in history
        if not has_il:
            return GrammarViolation(
                rule="U4b",
                message=(
                    "ZHIR (Mutation) requires prior IL (Coherence) "
                    "for a stable base before phase transformation."
                ),
                severity="error",
            )
    return None


# ── public API ────────────────────────────────────────────────────────────

def _check_violations(
    G: Any,
    node: Any,
    code: str,
    window: int = _DEFAULT_WINDOW,
) -> tuple[bool, list[GrammarViolation]]:
    """Core validation logic without alternative suggestion (avoids recursion)."""
    history = _recent_codes(G, node, window)

    # Read EPI for U1a
    try:
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_EPI
        epi = float(get_attr(G.nodes[node], ALIAS_EPI, 1.0))
    except Exception:
        epi = 1.0  # assume initialized

    violations: list[GrammarViolation] = []
    for checker in (_check_u1a, _check_u2, _check_u4a, _check_u4b):
        v = checker(code, history, epi) if checker is _check_u1a else checker(code, history)  # type: ignore[call-arg]
        if v is not None:
            violations.append(v)

    # U3 needs the graph
    v3 = _check_u3(code, G, node)
    if v3 is not None:
        violations.append(v3)

    errors = [v for v in violations if v.severity == "error"]
    allowed = len(errors) == 0
    return allowed, violations


def validate_candidate(
    G: Any,
    node: Any,
    candidate: str | Glyph,
    *,
    window: int = _DEFAULT_WINDOW,
) -> CandidateResult:
    """Check whether *candidate* is grammar-valid given the node's recent history.

    Runs incremental checks for U1a, U2, U3, U4a, U4b.  Returns a
    :class:`CandidateResult` with ``allowed=True`` if no errors are found.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node.
    node : NodeId
        Target node.
    candidate : str | Glyph
        Candidate glyph code (e.g. ``"OZ"``) or :class:`Glyph` enum.
    window : int, optional
        How many recent history entries to consider (default 6).

    Returns
    -------
    CandidateResult
        Validation result with violations and suggested alternative.
    """
    code = _to_code(candidate)
    allowed, violations = _check_violations(G, node, code, window)

    alt: str | None = None
    if not allowed:
        alt = suggest_alternative(G, node, code, window=window)

    return CandidateResult(
        candidate=code,
        allowed=allowed,
        violations=violations,
        suggested_alternative=alt,
    )


def filter_candidates(
    G: Any,
    node: Any,
    candidates: Sequence[str | Glyph],
    *,
    window: int = _DEFAULT_WINDOW,
) -> list[str]:
    """Return only grammar-valid candidates from the given list.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node.
    node : NodeId
        Target node.
    candidates : sequence of str | Glyph
        Candidate glyph codes to evaluate.
    window : int, optional
        Recent history context size.

    Returns
    -------
    list[str]
        Codes that pass incremental grammar checks (errors only; warnings pass).
    """
    result: list[str] = []
    for c in candidates:
        cr = validate_candidate(G, node, c, window=window)
        if cr.allowed:
            result.append(cr.candidate)
    return result


_PRIORITY_ORDER = ["IL", "THOL", "EN", "SHA", "RA", "NAV", "AL"]
"""Fallback priority: stabilizers first, then neutral, then generators."""


def suggest_alternative(
    G: Any,
    node: Any,
    rejected: str,
    *,
    window: int = _DEFAULT_WINDOW,
) -> str:
    """Suggest the best grammar-safe alternative when *rejected* is invalid.

    Iterates a priority-ordered list of safe candidates and returns the first
    that passes incremental grammar checks.  Falls back to ``IL`` (always safe).

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node.
    node : NodeId
        Target node.
    rejected : str
        The glyph code that was rejected.
    window : int, optional
        Recent history context size.

    Returns
    -------
    str
        A grammar-safe glyph code.
    """
    for alt in _PRIORITY_ORDER:
        if alt == rejected:
            continue
        allowed, _ = _check_violations(G, node, alt, window)
        if allowed:
            return alt
    return _FALLBACK_CODE


def enforce_grammar_on_glyph(
    G: Any,
    node: Any,
    candidate: str | Glyph,
    *,
    window: int = _DEFAULT_WINDOW,
) -> str:
    """Validate *candidate* and replace it with a safe alternative if invalid.

    Single source of truth for incremental grammar enforcement (U1-U6).
    ``enforce_canonical_grammar()`` delegates here; all application paths
    converge through this function exactly once before executing the operator.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node.
    node : NodeId
        Target node.
    candidate : str | Glyph
        Proposed glyph.
    window : int, optional
        Recent history context size.

    Returns
    -------
    str
        The validated (or replaced) glyph code.
    """
    cr = validate_candidate(G, node, candidate, window=window)
    if cr.allowed:
        return cr.candidate
    return cr.suggested_alternative or _FALLBACK_CODE


def validate_sequence_incremental(
    G: Any,
    node: Any,
    sequence: Sequence[str | Glyph],
    *,
    window: int = _DEFAULT_WINDOW,
) -> list[CandidateResult]:
    """Validate a full operator sequence step-by-step against a node's live history.

    Unlike :func:`grammar_core.validate_sequence` (which checks the sequence
    in isolation), this function considers the **actual glyph history** stored
    on the node, simulating how the sequence would be applied incrementally.

    It appends each accepted glyph to a *shadow* history copy so that later
    steps in the sequence see the effect of earlier ones.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node.
    node : NodeId
        Target node.
    sequence : sequence of str | Glyph
        Ordered glyphs to validate.
    window : int, optional
        Recent history context size.

    Returns
    -------
    list[CandidateResult]
        One result per step.  All ``allowed=True`` means the sequence is
        grammar-safe for this node in its current state.
    """
    nd = G.nodes[node]
    raw = nd.get("glyph_history")
    shadow: list[str] = [_to_code(g) for g in (list(raw) if raw else [])]

    results: list[CandidateResult] = []
    for step in sequence:
        code = _to_code(step)
        # Temporarily set shadow history on the node for the check
        original_history = nd.get("glyph_history")
        nd["glyph_history"] = shadow[-window:] if shadow else []
        try:
            cr = validate_candidate(G, node, code, window=window)
        finally:
            # Restore original history
            if original_history is not None:
                nd["glyph_history"] = original_history
            else:
                nd.pop("glyph_history", None)
        results.append(cr)
        # Accepted glyphs extend the shadow for subsequent steps
        shadow.append(code)

    return results


__all__ = [
    "GrammarViolation",
    "CandidateResult",
    "validate_candidate",
    "filter_candidates",
    "suggest_alternative",
    "enforce_grammar_on_glyph",
    "validate_sequence_incremental",
]
