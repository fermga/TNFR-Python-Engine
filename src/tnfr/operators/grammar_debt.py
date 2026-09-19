"""Shared causal bookkeeping for U2 debt and lifetime U4b context.

The capacity is derived in physics_derivation. This module implements its
discrete operator accounting: a destabilizer incurs one unit, a stabilizer
discharges one outstanding unit, and neutral operators do not discharge debt.
Earlier stabilization cannot prepay arbitrarily many future destabilizers.
This counter is grammar bookkeeping, not a measured continuous pressure field.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from numbers import Integral
from typing import Any

from .grammar_types import DESTABILIZERS, STABILIZERS, glyph_function_name

U2_DEBT_KEY = "_grammar_u2_debt"
PRIOR_COHERENCE_KEY = "_grammar_prior_coherence"


def require_replayable_history(history: Any) -> None:
    """Reject one-shot histories before read-only grammar checks consume them."""
    if isinstance(history, Iterator):
        raise ValueError(
            "glyph_history must be replayable for read-only grammar validation; "
            "materialize the iterator as a list, tuple or deque before assigning it."
        )


def advance_prior_coherence(prior: bool, operator: Any) -> bool:
    """Remember a prior IL without expiring it with the recent U4 window."""
    return prior or glyph_function_name(operator) == "coherence"


def prior_coherence_from_history(history: Iterable[Any]) -> bool:
    """Reconstruct the stable-base prerequisite from the available trace."""
    if isinstance(history, (str, bytes, bytearray)):
        return False
    try:
        tokens = iter(history)
    except TypeError:
        return False
    return any(glyph_function_name(operator) == "coherence" for operator in tokens)


def node_has_prior_coherence(node_data: Mapping[str, Any]) -> bool:
    """Read lifetime IL context, reconstructing legacy retained history."""
    value = node_data.get(PRIOR_COHERENCE_KEY)
    if isinstance(value, bool):
        return value
    return prior_coherence_from_history(node_data.get("glyph_history") or ())


def advance_debt(debt: int, operator: Any) -> int:
    """Apply one canonical operator's obligation to nonnegative U2 debt."""
    name = glyph_function_name(operator)
    if name in DESTABILIZERS:
        return debt + 1
    if name in STABILIZERS:
        return max(0, debt - 1)
    return debt


def debt_from_history(history: Iterable[Any]) -> int:
    """Reconstruct debt from an ordered history, without a sliding expiry."""
    # Match glyph_history._ensure_history when loading malformed legacy data.
    if isinstance(history, (str, bytes, bytearray)):
        return 0
    try:
        tokens = iter(history)
    except TypeError:
        return 0
    debt = 0
    for operator in tokens:
        debt = advance_debt(debt, operator)
    return debt


def node_debt(node_data: Mapping[str, Any]) -> int:
    """Read persisted debt, or reconstruct it for legacy/imported histories.

    A persisted counter preserves obligations older than the bounded trace.
    Legacy histories can reconstruct only the information they still retain.
    """
    value = node_data.get(U2_DEBT_KEY)
    if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return debt_from_history(node_data.get("glyph_history") or ())


def reset_debt_from_history(node_data: MutableMapping[str, Any]) -> int:
    """Reset accounting after deliberately replacing a node's full history.

    Use this when starting a new trace or importing replacement history into an
    existing node. A restored snapshot carrying its own debt counter needs no
    reset, since that counter retains obligations evicted from its trace.
    This resets U2 only; use ``reset_grammar_state_from_history`` when replacing
    an entire trace so that lifetime U4b context is reset as well.
    """
    debt = debt_from_history(node_data.get("glyph_history") or ())
    node_data[U2_DEBT_KEY] = debt
    return debt


def reset_grammar_state_from_history(node_data: MutableMapping[str, Any]) -> None:
    """Reconstruct U2/U4b bookkeeping after deliberate full-trace replacement.

    Preserve both persisted keys when restoring a snapshot: its bounded trace
    alone cannot recover older debt or an evicted Coherence operation.
    """
    reset_debt_from_history(node_data)
    node_data[PRIOR_COHERENCE_KEY] = prior_coherence_from_history(
        node_data.get("glyph_history") or ()
    )
