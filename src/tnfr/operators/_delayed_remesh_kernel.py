"""Immutable proposal and evidence kernel for explicit delayed REMESH mixing.

This module contracts the separately invoked temporal EPI map.  It does not
implement the advisory Recursivity glyph stage and does not advance history.

For each node, the exact-real recurrence represented by the runtime inputs is

    raw = (1 - alpha)^2 current
        + alpha (1 - alpha) local
        + alpha global.

The binary64 runtime evaluates the equivalent nested expression and then applies
one declared structural clipping policy.  Those two outputs remain separate in
every proposal so clipping cannot be mistaken for affine dynamics.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, Callable, Literal

from .._remesh_contract import (
    materialize_delayed_remesh_configuration,
    materialize_positive_diagonal_metric,
)
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..utils._structural_signature import structural_proof_signature
from ._epi_domain import require_real_scalar_epi
from .network_stage import _runtime_class_mro, _runtime_mapping_items

__all__ = [
    "DelayedRemeshNodeProposal",
    "DelayedRemeshPlan",
    "DelayedRemeshResult",
    "DelayedRemeshStabilityEvidence",
    "build_delayed_remesh_plan",
]

DelayedRemeshStatus = Literal[
    "applied", "insufficient_history", "empty_support"
]
ClipMode = Literal["hard", "soft"]

_EVIDENCE_SCOPE = (
    "One explicit delayed REMESH EPI operation with fixed selected history "
    "snapshots. Raw fields separate the exact-real affine recurrence from its "
    "nested binary64 evaluation. Weighted means and disagreement energies are "
    "observations in the declared positive diagonal metric. A finite raw "
    "global disagreement gain requires a uniform delayed offset. The bounded "
    "gain is a sufficient exact-real scalar-projection bound; it excludes "
    "binary64 rounding. No field proves repeated-map stability, history "
    "advance, pressure closure, U2 convergence, or stability from the REMESH "
    "name."
)


@dataclass(frozen=True, slots=True)
class DelayedRemeshNodeProposal:
    """One node-bound proposal derived from immutable current/history inputs."""

    node: Hashable
    epi_now: float
    epi_local: float
    epi_global: float
    raw_epi: float
    bounded_epi: float
    clipping_intervened: bool


@dataclass(frozen=True, slots=True)
class DelayedRemeshStabilityEvidence:
    """Scoped one-step observations and sufficient disagreement bounds.

    Evidence fields use finite binary64 presentation values. Planning rejects
    explicitly when an exact rational diagnostic exceeds that display range.
    """

    scope: str
    metric_weights: tuple[float, ...]
    beta: float
    gamma: float
    delta: float
    coefficient_partition_exact: bool
    max_raw_affine_rounding_residual: float
    clipped_nodes: tuple[Hashable, ...]
    observed_bounded_output_matches_raw: bool
    observed_delayed_inputs_equal_current: bool
    observed_bounded_output_equals_current: bool
    observed_equal_consensus_inputs_fixed: bool
    fixed_history_raw_map_preserves_consensus_subspace: bool | None
    observed_raw_weighted_mean_before: float | None
    observed_local_weighted_mean: float | None
    observed_global_weighted_mean: float | None
    ideal_raw_weighted_mean_after: float | None
    ideal_raw_weighted_mean_combination_exact: bool
    observed_raw_weighted_mean_after: float | None
    observed_bounded_weighted_mean_after: float | None
    observed_binary64_mean_rounding_residual: float | None
    observed_raw_weighted_mean_preserved: bool | None
    observed_bounded_weighted_mean_preserved: bool | None
    observed_current_disagreement_energy: float | None
    observed_raw_disagreement_energy: float | None
    observed_bounded_disagreement_energy: float | None
    observed_raw_disagreement_nonincreasing: bool | None
    observed_bounded_disagreement_nonincreasing: bool | None
    ideal_raw_disagreement_energy: float | None
    raw_convex_input_disagreement_bound: float | None
    raw_convex_disagreement_bound_certified: bool
    fixed_history_raw_global_disagreement_gain_upper_bound: float | None
    fixed_history_bounded_global_disagreement_gain_upper_bound: float | None

    @property
    def any_clipping_intervention(self) -> bool:
        """Whether the public bounded output differs from the raw recurrence."""

        return bool(self.clipped_nodes)

    @property
    def raw_global_disagreement_gain_certified(self) -> bool:
        """Whether the fixed-history raw map has a finite global gain bound."""

        return (
            self.fixed_history_raw_global_disagreement_gain_upper_bound
            is not None
        )

    @property
    def bounded_global_disagreement_gain_certified(self) -> bool:
        """Whether a sufficient bounded-map gain was proved in the real model."""

        return (
            self.fixed_history_bounded_global_disagreement_gain_upper_bound
            is not None
        )


@dataclass(frozen=True, slots=True)
class DelayedRemeshPlan:
    """Immutable empty/no-history no-op or all-node delayed REMESH proposal."""

    status: DelayedRemeshStatus
    node_order: tuple[Hashable, ...]
    history_length: int
    required_history_length: int
    alpha: float
    alpha_source: str
    tau_local: int
    tau_global: int
    epi_min: float
    epi_max: float
    clip_mode: ClipMode
    proposals: tuple[DelayedRemeshNodeProposal, ...]
    evidence: DelayedRemeshStabilityEvidence | None = None

    @property
    def applied(self) -> bool:
        """Whether sufficient history produced a commit-ready proposal."""

        return self.status == "applied"

    @property
    def any_clipping_intervention(self) -> bool:
        """Whether at least one proposed bounded EPI differs from raw EPI."""

        return any(proposal.clipping_intervened for proposal in self.proposals)


@dataclass(frozen=True, slots=True)
class DelayedRemeshResult:
    """Immutable result returned after a no-op or successful atomic commit."""

    status: DelayedRemeshStatus
    plan: DelayedRemeshPlan
    metadata_items: tuple[tuple[str, Any], ...] = ()
    epi_time_boundary_recorded: bool = False

    @property
    def applied(self) -> bool:
        """Whether the delayed map committed."""

        return self.status == "applied"

    @property
    def proposals(self) -> tuple[DelayedRemeshNodeProposal, ...]:
        """Return the executor-owned frozen node proposals."""

        return self.plan.proposals

    @property
    def evidence(self) -> DelayedRemeshStabilityEvidence | None:
        """Return opt-in one-step evidence, when requested."""

        return self.plan.evidence

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a detached copy of committed event metadata."""

        return dict(self.metadata_items)


def _finite_real(value: Any, label: str) -> float:
    """Return a strict finite real scalar."""

    if isinstance(value, (bool, str, bytes, bytearray, complex)):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    if not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite")
    return result


def _materialize_indexed_history(
    history: Any,
    *,
    label: str = "_epi_hist",
) -> tuple[Any, ...]:
    """Read a supported indexed history without invoking subclass methods.

    Runtime history is deliberately limited to the built-in sequence storage
    families that the graph transaction can restore, plus one-dimensional
    NumPy arrays.  Calling ``tuple(history)`` or ``history[index]`` here would
    dispatch arbitrary user overrides after preflight.
    """

    owners = _runtime_class_mro(type(history))
    if tuple in owners:
        return tuple(tuple.__iter__(history))
    if list in owners:
        return tuple(list.__iter__(history))
    if deque in owners:
        return tuple(deque.__iter__(history))
    if type(history) is range:
        return tuple(range.__iter__(history))
    if np is not None and np.ndarray in owners:
        shape_descriptor = np.ndarray.__dict__["shape"]
        flat_descriptor = np.ndarray.__dict__["flat"]
        try:
            shape = type(shape_descriptor).__get__(
                shape_descriptor,
                history,
                type(history),
            )
            if type(shape) is not tuple or len(shape) != 1:
                raise TNFRValueError(
                    f"{label} NumPy storage must be one-dimensional"
                )
            flat = type(flat_descriptor).__get__(
                flat_descriptor,
                history,
                type(history),
            )
            return tuple(flat)
        except TNFRValueError:
            raise
        except BaseException as exc:
            raise TNFRValueError(
                f"{label} must be a replayable indexed history"
            ) from exc
    raise TNFRValueError(f"{label} must be a replayable indexed history")


def _runtime_mapping_values_for_nodes(
    snapshot: Any,
    nodes: tuple[Hashable, ...],
    *,
    label: str,
) -> tuple[Any, ...]:
    """Bind raw mapping entries to graph nodes without virtual key lookup."""

    try:
        items = _runtime_mapping_items(snapshot)
    except (TNFRValueError, TypeError) as exc:
        raise TNFRValueError(f"{label} must be a node-to-EPI mapping") from exc
    if len(items) != len(nodes):
        raise TNFRValueError(
            f"{label} support must equal the current graph node support"
        )

    unmatched = list(items)
    values: list[Any] = []
    for node in nodes:
        node_signature = structural_proof_signature(node)
        match_index = None
        for index, (candidate, _value) in enumerate(unmatched):
            if candidate is node or structural_proof_signature(candidate) == node_signature:
                match_index = index
                break
            # NetworkX node support follows hash/equality semantics.  This
            # fallback is evaluated once while inputs are materialized; every
            # subsequent value read uses the raw item captured here.
            try:
                equivalent = candidate == node
            except BaseException as exc:
                raise TNFRValueError(
                    f"{label} node-key equality could not be evaluated"
                ) from exc
            boolean_result = type(equivalent) is bool or (
                np is not None and type(equivalent) is np.bool_
            )
            if boolean_result and bool(equivalent):
                try:
                    hashes_match = hash(candidate) == hash(node)
                except BaseException as exc:
                    raise TNFRValueError(
                        f"{label} node-key hash could not be evaluated"
                    ) from exc
                if hashes_match:
                    match_index = index
                    break
        if match_index is None:
            raise TNFRValueError(
                f"{label} support must equal the current graph node support"
            )
        _candidate, value = unmatched.pop(match_index)
        values.append(value)
    if unmatched:
        raise TNFRValueError(
            f"{label} support must equal the current graph node support"
        )
    return tuple(values)


def _exact_support(
    snapshot: Any,
    nodes: tuple[Hashable, ...],
    *,
    label: str,
) -> tuple[Any, ...]:
    """Require one selected snapshot to have exactly the live node support."""

    return _runtime_mapping_values_for_nodes(snapshot, nodes, label=label)


def _materialize_remesh_metric(
    raw: Mapping[Hashable, Any] | Sequence[Any] | None,
    nodes: tuple[Hashable, ...],
) -> tuple[float, ...]:
    """Freeze metric inputs through non-virtual built-in storage primitives."""

    if raw is None:
        return materialize_positive_diagonal_metric(None, nodes)
    try:
        _runtime_mapping_items(raw)
    except (TNFRValueError, TypeError):
        try:
            values = _materialize_indexed_history(
                raw,
                label="metric_weights",
            )
        except TNFRValueError as exc:
            raise TNFRValueError(
                "metric_weights must use supported mapping or indexed "
                "sequence storage"
            ) from exc
    else:
        values = _runtime_mapping_values_for_nodes(
            raw,
            nodes,
            label="metric_weights",
        )
    return materialize_positive_diagonal_metric(values, nodes)


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(value)


def _weighted_mean(
    values: tuple[Fraction, ...],
    weights: tuple[Fraction, ...],
) -> Fraction | None:
    if not values:
        return None
    total_weight = sum(weights, Fraction(0))
    return sum(
        (weight * value for weight, value in zip(weights, values, strict=True)),
        Fraction(0),
    ) / total_weight


def _disagreement(
    values: tuple[Fraction, ...],
    weights: tuple[Fraction, ...],
) -> Fraction | None:
    mean = _weighted_mean(values, weights)
    if mean is None:
        return None
    return sum(
        (
            weight * (value - mean) * (value - mean)
            for weight, value in zip(weights, values, strict=True)
        ),
        Fraction(0),
    ) / 2


def _diagnostic_float(value: Fraction, label: str) -> float:
    """Materialize one exact value in the finite binary64 evidence range."""

    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise TNFRValueError(
            f"{label} exceeds the finite binary64 diagnostic range"
        ) from exc
    if not math.isfinite(result):
        raise TNFRValueError(
            f"{label} exceeds the finite binary64 diagnostic range"
        )
    return result


def _optional_diagnostic_float(
    value: Fraction | None,
    label: str,
) -> float | None:
    return None if value is None else _diagnostic_float(value, label)


def _all_equal(values: tuple[Fraction, ...]) -> bool:
    return not values or all(value == values[0] for value in values[1:])


def _build_evidence(
    proposals: tuple[DelayedRemeshNodeProposal, ...],
    *,
    alpha: float,
    epi_min: float,
    epi_max: float,
    clip_mode: ClipMode,
    weights: tuple[float, ...],
) -> DelayedRemeshStabilityEvidence:
    """Construct exact represented-input observations and scoped real bounds."""

    alpha_q = _fraction(alpha)
    one_minus = Fraction(1) - alpha_q
    beta_q = one_minus * one_minus
    gamma_q = alpha_q * one_minus
    delta_q = alpha_q
    weights_q = tuple(_fraction(weight) for weight in weights)
    current_q = tuple(_fraction(item.epi_now) for item in proposals)
    local_q = tuple(_fraction(item.epi_local) for item in proposals)
    global_q = tuple(_fraction(item.epi_global) for item in proposals)
    runtime_raw_q = tuple(_fraction(item.raw_epi) for item in proposals)
    bounded_q = tuple(_fraction(item.bounded_epi) for item in proposals)
    ideal_raw_q = tuple(
        beta_q * current + gamma_q * local + delta_q * global_value
        for current, local, global_value in zip(
            current_q, local_q, global_q, strict=True
        )
    )
    delayed_offset_q = tuple(
        gamma_q * local + delta_q * global_value
        for local, global_value in zip(local_q, global_q, strict=True)
    )
    residuals = tuple(
        abs(runtime - ideal)
        for runtime, ideal in zip(runtime_raw_q, ideal_raw_q, strict=True)
    )
    current_mean = _weighted_mean(current_q, weights_q)
    local_mean = _weighted_mean(local_q, weights_q)
    global_mean = _weighted_mean(global_q, weights_q)
    ideal_mean = _weighted_mean(ideal_raw_q, weights_q)
    raw_mean = _weighted_mean(runtime_raw_q, weights_q)
    bounded_mean = _weighted_mean(bounded_q, weights_q)
    ideal_mean_combination = (
        None
        if current_mean is None
        else (
            beta_q * current_mean
            + gamma_q * local_mean
            + delta_q * global_mean
        )
    )
    current_energy = _disagreement(current_q, weights_q)
    raw_energy = _disagreement(runtime_raw_q, weights_q)
    bounded_energy = _disagreement(bounded_q, weights_q)
    ideal_energy = _disagreement(ideal_raw_q, weights_q)
    local_energy = _disagreement(local_q, weights_q)
    global_energy = _disagreement(global_q, weights_q)
    convex_bound = (
        None
        if current_energy is None
        else beta_q * current_energy
        + gamma_q * local_energy
        + delta_q * global_energy
    )
    offset_uniform = None if not proposals else _all_equal(delayed_offset_q)
    raw_gain = (
        _diagnostic_float(beta_q * beta_q, "fixed-history raw gain bound")
        if offset_uniform
        else None
    )
    if epi_min == epi_max:
        bounded_gain = 0.0
    elif offset_uniform:
        lipschitz = Fraction(1) if clip_mode == "hard" else Fraction(4, 3)
        bounded_gain = _diagnostic_float(
            beta_q * beta_q * lipschitz * lipschitz,
            "fixed-history bounded gain bound",
        )
    else:
        bounded_gain = None
    clipped_nodes = tuple(
        item.node for item in proposals if item.clipping_intervened
    )
    observed_consensus_fixed = bool(
        proposals
        and _all_equal(current_q + local_q + global_q)
        and runtime_raw_q == current_q
        and bounded_q == current_q
    )
    return DelayedRemeshStabilityEvidence(
        scope=_EVIDENCE_SCOPE,
        metric_weights=weights,
        beta=_diagnostic_float(beta_q, "beta coefficient"),
        gamma=_diagnostic_float(gamma_q, "gamma coefficient"),
        delta=_diagnostic_float(delta_q, "delta coefficient"),
        coefficient_partition_exact=(
            beta_q + gamma_q + delta_q == Fraction(1)
        ),
        max_raw_affine_rounding_residual=_diagnostic_float(
            max(residuals, default=Fraction(0)),
            "raw affine rounding residual",
        ),
        clipped_nodes=clipped_nodes,
        observed_bounded_output_matches_raw=(bounded_q == runtime_raw_q),
        observed_delayed_inputs_equal_current=(
            local_q == current_q and global_q == current_q
        ),
        observed_bounded_output_equals_current=(bounded_q == current_q),
        observed_equal_consensus_inputs_fixed=observed_consensus_fixed,
        fixed_history_raw_map_preserves_consensus_subspace=offset_uniform,
        observed_raw_weighted_mean_before=_optional_diagnostic_float(
            current_mean, "current weighted mean"
        ),
        observed_local_weighted_mean=_optional_diagnostic_float(
            local_mean, "local weighted mean"
        ),
        observed_global_weighted_mean=_optional_diagnostic_float(
            global_mean, "global weighted mean"
        ),
        ideal_raw_weighted_mean_after=_optional_diagnostic_float(
            ideal_mean, "ideal raw weighted mean"
        ),
        ideal_raw_weighted_mean_combination_exact=(
            ideal_mean == ideal_mean_combination
        ),
        observed_raw_weighted_mean_after=_optional_diagnostic_float(
            raw_mean, "runtime raw weighted mean"
        ),
        observed_bounded_weighted_mean_after=_optional_diagnostic_float(
            bounded_mean, "bounded weighted mean"
        ),
        observed_binary64_mean_rounding_residual=(
            None
            if ideal_mean is None
            else _diagnostic_float(
                abs(raw_mean - ideal_mean),
                "binary64 mean rounding residual",
            )
        ),
        observed_raw_weighted_mean_preserved=(
            None if current_mean is None else raw_mean == current_mean
        ),
        observed_bounded_weighted_mean_preserved=(
            None if current_mean is None else bounded_mean == current_mean
        ),
        observed_current_disagreement_energy=_optional_diagnostic_float(
            current_energy, "current disagreement energy"
        ),
        observed_raw_disagreement_energy=_optional_diagnostic_float(
            raw_energy, "runtime raw disagreement energy"
        ),
        observed_bounded_disagreement_energy=_optional_diagnostic_float(
            bounded_energy, "bounded disagreement energy"
        ),
        observed_raw_disagreement_nonincreasing=(
            None if current_energy is None else raw_energy <= current_energy
        ),
        observed_bounded_disagreement_nonincreasing=(
            None if current_energy is None else bounded_energy <= current_energy
        ),
        ideal_raw_disagreement_energy=_optional_diagnostic_float(
            ideal_energy, "ideal raw disagreement energy"
        ),
        raw_convex_input_disagreement_bound=_optional_diagnostic_float(
            convex_bound, "convex input disagreement bound"
        ),
        raw_convex_disagreement_bound_certified=(
            ideal_energy is not None
            and convex_bound is not None
            and ideal_energy <= convex_bound
        ),
        fixed_history_raw_global_disagreement_gain_upper_bound=raw_gain,
        fixed_history_bounded_global_disagreement_gain_upper_bound=bounded_gain,
    )


def build_delayed_remesh_plan(
    *,
    node_order: Sequence[Hashable],
    current_epi: Mapping[Hashable, Any],
    history: Any,
    tau_local: Any,
    tau_global: Any,
    alpha: Any,
    alpha_source: str,
    epi_min: Any,
    epi_max: Any,
    clip_mode: Any,
    clipper: Callable[..., float],
    include_stability_evidence: bool = False,
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None = None,
) -> DelayedRemeshPlan:
    """Build one immutable delayed-map plan without mutating graph state."""

    nodes = tuple(node_order)
    if len(frozenset(nodes)) != len(nodes):
        raise TNFRValueError("node_order must contain unique hashable nodes")
    configuration = materialize_delayed_remesh_configuration(
        tau_local=tau_local,
        tau_global=tau_global,
        alpha=alpha,
        alpha_source=alpha_source,
        epi_min=epi_min,
        epi_max=epi_max,
        clip_mode=clip_mode,
    )
    local_delay = configuration.tau_local
    global_delay = configuration.tau_global
    alpha_value = configuration.alpha
    alpha_source = configuration.alpha_source
    lower = configuration.epi_min
    upper = configuration.epi_max
    clip_mode = configuration.clip_mode
    if type(include_stability_evidence) is not bool:
        raise TypeError("include_stability_evidence must be a bool")
    if metric_weights is not None and not include_stability_evidence:
        raise TNFRValueError(
            "metric_weights requires include_stability_evidence=True"
        )
    weights = (
        _materialize_remesh_metric(metric_weights, nodes)
        if include_stability_evidence
        else None
    )
    history_items = _materialize_indexed_history(history)
    history_length = len(history_items)
    required = max(local_delay, global_delay) + 1
    if not nodes:
        return DelayedRemeshPlan(
            status="empty_support",
            node_order=nodes,
            history_length=history_length,
            required_history_length=required,
            alpha=alpha_value,
            alpha_source=alpha_source,
            tau_local=local_delay,
            tau_global=global_delay,
            epi_min=lower,
            epi_max=upper,
            clip_mode=clip_mode,
            proposals=(),
        )
    if history_length < required:
        return DelayedRemeshPlan(
            status="insufficient_history",
            node_order=nodes,
            history_length=history_length,
            required_history_length=required,
            alpha=alpha_value,
            alpha_source=alpha_source,
            tau_local=local_delay,
            tau_global=global_delay,
            epi_min=lower,
            epi_max=upper,
            clip_mode=clip_mode,
            proposals=(),
        )
    current = _exact_support(current_epi, nodes, label="current EPI snapshot")
    local_raw = history_items[-(local_delay + 1)]
    global_raw = history_items[-(global_delay + 1)]
    local = _exact_support(local_raw, nodes, label="local delayed EPI snapshot")
    global_snapshot = _exact_support(
        global_raw, nodes, label="global delayed EPI snapshot"
    )
    proposals_list: list[DelayedRemeshNodeProposal] = []
    for node, now_raw, local_raw_value, global_raw_value in zip(
        nodes,
        current,
        local,
        global_snapshot,
        strict=True,
    ):
        now = require_real_scalar_epi(
            now_raw, operator="Recursivity", label=f"node {node!r} EPI"
        )
        old_local = require_real_scalar_epi(
            local_raw_value,
            operator="Recursivity",
            label=f"node {node!r} local delayed EPI",
        )
        old_global = require_real_scalar_epi(
            global_raw_value,
            operator="Recursivity",
            label=f"node {node!r} global delayed EPI",
        )
        mixed_local = (1.0 - alpha_value) * now + alpha_value * old_local
        raw_epi = (1.0 - alpha_value) * mixed_local + alpha_value * old_global
        raw_epi = _finite_real(raw_epi, f"node {node!r} REMESH proposal")
        try:
            bounded_epi = clipper(
                raw_epi,
                lo=lower,
                hi=upper,
                mode=clip_mode,
                record_stats=False,
            )
        except (OverflowError, TypeError, ValueError) as exc:
            raise TNFRValueError(
                f"node {node!r} REMESH clipping failed"
            ) from exc
        bounded_epi = _finite_real(
            bounded_epi, f"node {node!r} bounded REMESH proposal"
        )
        proposals_list.append(
            DelayedRemeshNodeProposal(
                node=node,
                epi_now=now,
                epi_local=old_local,
                epi_global=old_global,
                raw_epi=raw_epi,
                bounded_epi=bounded_epi,
                clipping_intervened=(bounded_epi != raw_epi),
            )
        )
    proposals = tuple(proposals_list)
    evidence = None
    if include_stability_evidence:
        assert weights is not None
        evidence = _build_evidence(
            proposals,
            alpha=alpha_value,
            epi_min=lower,
            epi_max=upper,
            clip_mode=clip_mode,
            weights=weights,
        )
    return DelayedRemeshPlan(
        status="applied",
        node_order=nodes,
        history_length=history_length,
        required_history_length=required,
        alpha=alpha_value,
        alpha_source=alpha_source,
        tau_local=local_delay,
        tau_global=global_delay,
        epi_min=lower,
        epi_max=upper,
        clip_mode=clip_mode,
        proposals=proposals,
        evidence=evidence,
    )
