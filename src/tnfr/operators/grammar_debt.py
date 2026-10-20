"""Shared causal bookkeeping for U2 debt and lifetime U4b context.

The capacity is calibrated in physics_derivation. This module implements its
discrete operator accounting: a destabilizer incurs one unit, a stabilizer
discharges one outstanding unit, and neutral operators do not discharge debt.
Earlier stabilization cannot prepay arbitrarily many future destabilizers.
This counter is grammar bookkeeping, not a measured continuous pressure field.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .grammar_types import (
    DESTABILIZERS,
    FUNCTION_TO_GLYPH,
    STABILIZERS,
    glyph_function_name,
)

U2_DEBT_KEY = "_grammar_u2_debt"
PRIOR_COHERENCE_KEY = "_grammar_prior_coherence"
OPERATOR_STEP_KEY = "_operator_step"


@dataclass(frozen=True, slots=True)
class GrammarState:
    """Fresh admitted bookkeeping; never a cache or occurrence certificate."""

    history: tuple[str, ...]
    u2_debt: int
    prior_coherence: bool
    operator_step: int


def _operator_identifier(operator: Any) -> str:
    """Admit a recorded operator through the shared identifier owner."""
    name = glyph_function_name(operator)
    if not isinstance(name, str) or name not in FUNCTION_TO_GLYPH:
        raise ValueError("glyph_history entries must be canonical operator identifiers")
    return name


def canonical_operator_code(operator: Any) -> str:
    """Admit a shared operator identifier and return its plain glyph code."""
    return FUNCTION_TO_GLYPH[_operator_identifier(operator)].value


def require_replayable_history(history: Any) -> None:
    """Admit ordered canonical history without consuming one-shot iterators.

    ``None`` denotes an absent history for callers of this value-level helper.
    Mapping-owned state uses :func:`require_grammar_state`, which distinguishes
    an absent key from a malformed explicitly stored ``None``.
    """
    _history_names(history)


def _history_names(history: Any) -> tuple[str, ...]:
    """Admit every retained identifier once, including after an earlier IL."""
    if history is None:
        return ()
    if isinstance(history, Iterator):
        raise ValueError(
            "glyph_history must be replayable for read-only grammar validation; "
            "materialize the iterator as a list, tuple or deque before assigning it."
        )
    if not isinstance(history, (list, tuple, deque)):
        raise TypeError(
            "glyph_history must be an ordered replayable list, tuple or deque"
        )
    return tuple(_operator_identifier(operator) for operator in history)


def _node_history_names(node_data: Mapping[str, Any]) -> tuple[str, ...]:
    if "glyph_history" not in node_data:
        return ()
    history = node_data["glyph_history"]
    if history is None:
        raise TypeError("stored glyph_history must be a list, tuple or deque, not None")
    return _history_names(history)


def _admit_counter(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _admit_prior_coherence(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{PRIOR_COHERENCE_KEY} must be boolean")
    return value


def require_grammar_state(node_data: Mapping[str, Any]) -> None:
    """Admit history and authoritative bookkeeping before an execution writes.

    Only missing counters are reconstructed from the retained legacy trace.
    Invalid present counters cannot erase debt or invent lifetime context.
    """
    read_grammar_state(node_data)


def read_grammar_state(node_data: Mapping[str, Any]) -> GrammarState:
    """Re-admit a node and return one immutable normalized bookkeeping view.

    Each history identifier is admitted once. Present bookkeeping remains
    authoritative; only absent values are reconstructed from the retained
    trace. The result is local to this read and must not authorize a later
    write after the underlying state has changed.
    """
    names = _node_history_names(node_data)
    debt = (
        _admit_counter(node_data[U2_DEBT_KEY], U2_DEBT_KEY)
        if U2_DEBT_KEY in node_data
        else _debt_from_names(names)
    )
    prior = (
        _admit_prior_coherence(node_data[PRIOR_COHERENCE_KEY])
        if PRIOR_COHERENCE_KEY in node_data
        else "coherence" in names
    )
    step = (
        max(_admit_counter(node_data[OPERATOR_STEP_KEY], OPERATOR_STEP_KEY), len(names))
        if OPERATOR_STEP_KEY in node_data
        else len(names)
    )
    return GrammarState(
        tuple(FUNCTION_TO_GLYPH[name].value for name in names), debt, prior, step
    )


def advance_prior_coherence(prior: bool, operator: Any) -> bool:
    """Remember a prior IL without expiring it with the recent U4 window."""
    prior = _admit_prior_coherence(prior)
    name = _operator_identifier(operator)
    return prior or name == "coherence"


def prior_coherence_from_history(history: Iterable[Any]) -> bool:
    """Reconstruct the stable-base prerequisite from an admitted trace."""
    return "coherence" in _history_names(history)


def node_has_prior_coherence(node_data: Mapping[str, Any]) -> bool:
    """Read valid lifetime IL context, reconstructing only an absent marker."""
    names = _node_history_names(node_data)
    if PRIOR_COHERENCE_KEY in node_data:
        return _admit_prior_coherence(node_data[PRIOR_COHERENCE_KEY])
    return "coherence" in names


def advance_debt(debt: int, operator: Any) -> int:
    """Apply one admitted operator's obligation to nonnegative U2 debt."""
    debt = _admit_counter(debt, U2_DEBT_KEY)
    name = _operator_identifier(operator)
    return _advance_named_debt(debt, name)


def _advance_named_debt(debt: int, name: str) -> int:
    """Update already admitted accounting within this module only."""
    if name in DESTABILIZERS:
        return debt + 1
    if name in STABILIZERS:
        return max(0, debt - 1)
    return debt


def debt_from_history(history: Iterable[Any]) -> int:
    """Reconstruct debt from an admitted ordered trace, without sliding expiry."""
    return _debt_from_names(_history_names(history))


def _debt_from_names(names: tuple[str, ...]) -> int:
    debt = 0
    for name in names:
        debt = _advance_named_debt(debt, name)
    return debt


def node_debt(node_data: Mapping[str, Any]) -> int:
    """Read valid persisted debt, or reconstruct a missing legacy counter.

    A persisted counter preserves obligations older than the bounded trace.
    Legacy histories can reconstruct only the information they still retain.
    Invalid authoritative counters reject instead of falling back to that trace.
    """
    names = _node_history_names(node_data)
    if U2_DEBT_KEY in node_data:
        return _admit_counter(node_data[U2_DEBT_KEY], U2_DEBT_KEY)
    return _debt_from_names(names)


def reset_debt_from_history(node_data: MutableMapping[str, Any]) -> int:
    """Reset accounting after deliberately replacing a node's full history.

    Use this when starting a new trace or importing replacement history into an
    existing node. A restored snapshot carrying its own debt counter needs no
    reset, since that counter retains obligations evicted from its trace.
    This resets U2 only; use ``reset_grammar_state_from_history`` when replacing
    an entire trace so that lifetime U4b context is reset as well.
    """
    debt = _debt_from_names(_node_history_names(node_data))
    node_data[U2_DEBT_KEY] = debt
    return debt


def reset_grammar_state_from_history(node_data: MutableMapping[str, Any]) -> None:
    """Reconstruct U2/U4b bookkeeping after deliberate full-trace replacement.

    Preserve both persisted keys when restoring a snapshot: its bounded trace
    alone cannot recover older debt or an evicted Coherence operation.
    """
    names = _node_history_names(node_data)
    debt = _debt_from_names(names)
    prior = "coherence" in names
    node_data[U2_DEBT_KEY] = debt
    node_data[PRIOR_COHERENCE_KEY] = prior
