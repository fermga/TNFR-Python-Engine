"""Shared target-state admission and selected Coherence activity policies.

The default ``signed_threshold`` policy retains the historical signed-EPI
and positive-capacity bounds. ``event_domain`` admits the scalar domain of
the instantaneous pressure/phase action, including zero form and capacity.
Neither policy establishes neighbor-phase admission, grammar or occurrence.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...alias import get_attr
from ...config.thresholds import DNFR_IL_CRITICAL, EPI_IL_MIN, VF_IL_MIN
from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from .._argument_validation import (
    finite_node_real,
    finite_real,
    reject_operator_argument,
    strict_bool,
)
from .._epi_domain import require_real_scalar_epi

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = [
    "coherence_precondition_warnings",
    "diagnose_coherence_readiness",
    "emit_coherence_precondition_warnings",
    "validate_coherence_strict",
]


@dataclass(frozen=True)
class _CoherencePolicy:
    activity_policy: str
    min_epi: float | None
    min_vf: float | None
    dnfr_critical_threshold: float
    warn_zero_dnfr: bool
    warn_isolated: bool


@dataclass(frozen=True)
class _CoherenceState:
    epi: float
    vf: float
    dnfr: float
    phase: float


def _coherence_policy(G: TNFRGraph) -> _CoherencePolicy:
    """Normalize active policy inputs without coercing invalid selectors."""
    raw = G.graph.get("IL_PRECONDITIONS", {})
    if not isinstance(raw, Mapping):
        reject_operator_argument("Coherence", "IL_PRECONDITIONS must be a mapping")
    activity = raw.get("activity_policy", "signed_threshold")
    if type(activity) is not str or activity not in (
        "signed_threshold",
        "event_domain",
    ):
        reject_operator_argument(
            "Coherence",
            "IL_PRECONDITIONS.activity_policy must be 'signed_threshold' "
            "or 'event_domain'",
        )
    if activity == "event_domain":
        if "min_epi" in raw or "min_vf" in raw:
            reject_operator_argument(
                "Coherence",
                "IL_PRECONDITIONS min_epi/min_vf are incompatible with event_domain",
            )
        min_epi = min_vf = None
    else:
        min_epi = finite_real(
            raw.get("min_epi", EPI_IL_MIN),
            operator="Coherence",
            label="IL_PRECONDITIONS.min_epi",
        )
        min_vf = finite_real(
            raw.get("min_vf", VF_IL_MIN),
            operator="Coherence",
            label="IL_PRECONDITIONS.min_vf",
            lower=0.0,
        )
    return _CoherencePolicy(
        activity_policy=activity,
        min_epi=min_epi,
        min_vf=min_vf,
        dnfr_critical_threshold=finite_real(
            raw.get("dnfr_critical_threshold", DNFR_IL_CRITICAL),
            operator="Coherence",
            label="IL_PRECONDITIONS.dnfr_critical_threshold",
            lower=0.0,
        ),
        warn_zero_dnfr=strict_bool(
            raw.get("warn_zero_dnfr", True),
            operator="Coherence",
            label="IL_PRECONDITIONS.warn_zero_dnfr",
        ),
        warn_isolated=strict_bool(
            raw.get("warn_isolated", True),
            operator="Coherence",
            label="IL_PRECONDITIONS.warn_isolated",
        ),
    )


def _coherence_state(G: TNFRGraph, node: Any) -> _CoherenceState:
    """Admit authoritative scalar state before projection or comparison."""
    data = G.nodes[node]
    return _CoherenceState(
        epi=require_real_scalar_epi(
            get_attr(data, ALIAS_EPI, 0.0, strict=True, conv=lambda value: value),
            operator="Coherence",
            label="EPI state",
        ),
        vf=finite_node_real(
            data,
            ALIAS_VF,
            0.0,
            operator="Coherence",
            label="nu_f state",
            lower=0.0,
        ),
        dnfr=finite_node_real(
            data, ALIAS_DNFR, 0.0, operator="Coherence", label="DeltaNFR state"
        ),
        phase=finite_node_real(
            data, ALIAS_THETA, 0.0, operator="Coherence", label="theta state"
        ),
    )


def _coherence_activity(state: _CoherenceState, policy: _CoherencePolicy):
    """Return selected predicates, without equating a threshold with existence."""
    if policy.activity_policy == "event_domain":
        return True, True
    assert policy.min_epi is not None and policy.min_vf is not None
    return state.epi > policy.min_epi, state.vf > policy.min_vf


def _coherence_warnings(G, node, state, policy) -> tuple[str, ...]:
    """Describe only admitted stored-channel observations and local support."""
    magnitude = abs(state.dnfr)
    messages: list[str] = []
    if policy.warn_zero_dnfr and magnitude == 0.0:
        messages.append(
            f"IL warning: Node {node!r} has |ΔNFR|=0. "
            "The stored pressure channel is quiet; the phase proposal may still change."
        )
    if magnitude > policy.dnfr_critical_threshold:
        messages.append(
            f"IL warning: Node {node!r} has |ΔNFR|={magnitude:.3f} "
            f"> {policy.dnfr_critical_threshold:.3f}. This exceeds the configured "
            "advisory bound; stored-pressure contraction does not guarantee "
            "smaller refreshed pressure or future stability."
        )
    if policy.warn_isolated and G.degree(node) == 0 and len(G) > 1:
        messages.append(
            f"IL warning: Node {node!r} isolated (degree=0). "
            "No neighbor phase is available; the phase proposal only normalizes "
            "its stored representative."
        )
    return tuple(messages)


def validate_coherence_strict(
    G: TNFRGraph,
    node: Any,
    *,
    emit_warnings: bool = True,
) -> None:
    """Validate finite target state and the configured Coherence policy.

    ``signed_threshold`` requires ``EPI > min_epi`` and ``nu_f > min_vf``;
    both defaults are zero. This selected sign restriction is not a test for
    the existence of form. ``event_domain`` permits every finite signed scalar
    EPI and nonnegative capacity, with finite stored pressure and phase. It
    rejects min_epi/min_vf overrides instead of ignoring them.

    The pressure and isolation conditions are advisory. This target-only
    validator does not admit neighboring phases, proposal arithmetic, grammar,
    history, a whole word, event occurrence or future stability.
    """
    emit_warnings = strict_bool(
        emit_warnings, operator="Coherence", label="emit_warnings"
    )
    policy = _coherence_policy(G)
    state = _coherence_state(G, node)
    epi_allowed, vf_allowed = _coherence_activity(state, policy)
    if not epi_allowed:
        raise ValueError(
            f"IL precondition failed: EPI={state.epi:.3f} <= {policy.min_epi:.3f}. "
            "Stored signed form is outside the selected signed_threshold policy; "
            "negative or zero form is not absence of the NFR."
        )
    if not vf_allowed:
        raise ValueError(
            f"IL precondition failed: νf={state.vf:.3f} <= {policy.min_vf:.3f}. "
            "Capacity is outside the selected signed_threshold policy. "
            "Multiplicative scaling cannot activate zero capacity."
        )
    if emit_warnings:
        for message in _coherence_warnings(G, node, state, policy):
            warnings.warn(message, UserWarning, stacklevel=3)


def coherence_precondition_warnings(G: TNFRGraph, node: Any) -> tuple[str, ...]:
    """Return advisory messages after shared primitive and policy admission."""
    policy = _coherence_policy(G)
    state = _coherence_state(G, node)
    return _coherence_warnings(G, node, state, policy)


def emit_coherence_precondition_warnings(G: TNFRGraph, node: Any) -> None:
    """Emit the same freshly admitted warning set used by strict validation."""
    for message in coherence_precondition_warnings(G, node):
        warnings.warn(message, UserWarning, stacklevel=3)


def diagnose_coherence_readiness(G: TNFRGraph, node: Any) -> dict:
    """Report target-state policy readiness without authorizing execution.

    Compatibility keys ``epi_active`` and ``vf_active`` report the selected
    policy predicates. Under event_domain both are true after scalar
    admission, including at zero. They do not declare autonomous activity.
    Invalid state, policy scalars or flags reject rather than becoming a
    passing comparison or an unavailable observation coerced to zero.
    """
    policy = _coherence_policy(G)
    state = _coherence_state(G, node)
    epi_allowed, vf_allowed = _coherence_activity(state, policy)
    degree = G.degree(node)
    checks = {
        "epi_active": epi_allowed,
        "vf_active": vf_allowed,
        "dnfr_present": abs(state.dnfr) > 0.0,
        "dnfr_not_critical": abs(state.dnfr) <= policy.dnfr_critical_threshold,
        "has_connections": degree > 0,
    }
    ready = epi_allowed and vf_allowed
    recommendations = []
    if not epi_allowed:
        recommendations.append(
            f"The selected signed_threshold policy requires EPI > {policy.min_epi}; "
            "this rejection does not imply that signed form is absent."
        )
    if not vf_allowed:
        recommendations.append(
            f"The selected signed_threshold policy requires νf > {policy.min_vf}. "
            "Multiplicative scaling preserves zero capacity."
        )
    recommendations.extend(_coherence_warnings(G, node, state, policy))
    if ready:
        recommendations.insert(
            0,
            "Target state meets the selected IL policy; execution admission remains separate.",
        )
    return {
        "node": node,
        "ready": ready,
        "activity_policy": policy.activity_policy,
        "checks": checks,
        "values": {
            "epi": state.epi,
            "vf": state.vf,
            "dnfr": state.dnfr,
            "phase": state.phase,
            "degree": degree,
        },
        "recommendations": recommendations,
        "scope": (
            "finite_target_state_and_selected_activity_policy_only",
            "neighbor_phases_full_proposal_arithmetic_and_grammar_not_admitted",
            "no_event_occurrence_word_completion_or_future_stability_claim",
        ),
    }
