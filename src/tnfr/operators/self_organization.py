"""Atomic public implementation of SelfOrganization (THOL).

THOL materializes operational fractality only after its complete U5 proposal is
valid. The public operator separates read-only planning from commit and restores
the graph if an external monitor or a later validation rejects the operation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, ClassVar

from ..config.operator_names import SELF_ORGANIZATION
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..constants.canonical import COUPLING_GENTLE, COUPLING_MODERATE
from ..glyph_history import next_operator_step
from ..types import Glyph, TNFRGraph, real_scalar_epi
from ._argument_validation import (
    finite_node_real,
    finite_real,
    nonnegative_integer,
    reject_operator_argument,
    require_list_sink,
    strict_bool,
    validate_common_execution_arguments,
)
from .definitions_base import Operator
from .network_stage import GraphTransactionSnapshot
from ._thol_constants import THOL_CHILD_VF_DAMPING, THOL_SUB_EPI_SCALING

_OPERATOR = "Self-organization"


@dataclass(frozen=True, slots=True)
class _BifurcationProposal:
    sub_node_id: str
    sub_node_data: Mapping[str, Any]
    sub_nodes: tuple[Any, ...]
    hierarchy_children: tuple[Any, ...]
    sub_epis: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ExecutionProposal:
    d2_epi: float
    tau: float
    dnfr_after: float
    bifurcation: _BifurcationProposal | None
    subepi_amplitude_alignment: float | None
    precondition_context: Mapping[str, Any] | None
    no_bifurcation_expected: bool | None
    depth_limit_reached: Mapping[str, Any] | None
    final_parent_epi: float


_GraphSnapshot = GraphTransactionSnapshot


def _checked_sum(left: float, right: float, label: str) -> float:
    return finite_real(left + right, operator=_OPERATOR, label=label)


def _checked_product(left: float, right: float, label: str) -> float:
    return finite_real(left * right, operator=_OPERATOR, label=label)


def _configured_tau(graph_data: Mapping[str, Any], kwargs: Mapping[str, Any]) -> float:
    raw = kwargs.get("tau")
    if raw is None:
        raw = graph_data.get("BIFURCATION_THRESHOLD_TAU")
    if raw is None:
        raw = graph_data.get("THOL_BIFURCATION_THRESHOLD", 0.1)
    return finite_real(raw, operator=_OPERATOR, label="tau", lower=0.0)


def _configured_epi_bounds(graph_data: Mapping[str, Any]) -> tuple[float, float]:
    """Return the signed scalar EPI interval used by THOL proposals."""

    epi_min = finite_real(
        graph_data.get("EPI_MIN", -1.0), operator=_OPERATOR, label="EPI_MIN"
    )
    epi_max = finite_real(
        graph_data.get("EPI_MAX", 1.0), operator=_OPERATOR, label="EPI_MAX"
    )
    if epi_min > epi_max:
        reject_operator_argument(_OPERATOR, "EPI_MIN must not exceed EPI_MAX")
    return epi_min, epi_max


def _existing_list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = mapping.get(key, [])
    if not isinstance(value, list):
        reject_operator_argument(_OPERATOR, f"{key} must be a list")
    return value


def _finite_epi_value(
    value: Any,
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Validate a real scalar or canonical BEPI representation."""

    if isinstance(value, bool):
        reject_operator_argument(_OPERATOR, f"{label} must be a real EPI value")
    try:
        result = real_scalar_epi(value)
    except (OverflowError, TypeError, ValueError):
        reject_operator_argument(
            _OPERATOR, f"{label} must be a scalar or uniform-real BEPI value"
        )
    if result is None:
        reject_operator_argument(
            _OPERATOR, f"{label} must have an exact signed scalar embedding"
        )
    return finite_real(
        result,
        operator=_OPERATOR,
        label=label,
        lower=lower,
        upper=upper,
    )


def _finite_node_epi(
    node_data: Mapping[str, Any], *, label: str, default: Any = 0.0
) -> float:
    raw = default
    for key in ALIAS_EPI:
        if key in node_data:
            raw = node_data[key]
            break
    return _finite_epi_value(raw, label=label)


def _active_acceleration_history_length(node_data: Mapping[str, Any]) -> int:
    """Read only the size selected by the shared acceleration implementation."""

    from .nodal_equation import _select_acceleration_history

    source, history = _select_acceleration_history(node_data)
    if history is None:
        return 0
    if isinstance(history, (str, bytes, bytearray, Mapping, Iterator)):
        reject_operator_argument(_OPERATOR, f"{source} must be an indexed history")
    try:
        return len(history)
    except (OverflowError, TypeError):
        reject_operator_argument(_OPERATOR, f"{source} must be a sized history")


def _validate_signal_record(signals: Any, *, label: str) -> None:
    if signals is None:
        return
    if not isinstance(signals, Mapping):
        reject_operator_argument(_OPERATOR, f"{label} must be a mapping or None")
    for key in ("epi_gradient", "mean_neighbor_epi"):
        if key in signals:
            finite_real(signals[key], operator=_OPERATOR, label=f"{label}.{key}")
    if "phase_variance" in signals:
        finite_real(
            signals["phase_variance"],
            operator=_OPERATOR,
            label=f"{label}.phase_variance",
            lower=0.0,
        )
    if "coupling_strength_mean" in signals:
        finite_real(
            signals["coupling_strength_mean"],
            operator=_OPERATOR,
            label=f"{label}.coupling_strength_mean",
            lower=0.0,
            upper=1.0,
        )
    if "neighbor_count" in signals:
        nonnegative_integer(
            signals["neighbor_count"],
            operator=_OPERATOR,
            label=f"{label}.neighbor_count",
        )


def _validate_sub_epi_records(records: list[Any]) -> tuple[Mapping[str, Any], ...]:
    validated: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        label = f"sub_epis[{index}]"
        if not isinstance(record, Mapping):
            reject_operator_argument(_OPERATOR, f"{label} must be a mapping")
        if "epi" not in record:
            reject_operator_argument(_OPERATOR, f"{label}.epi is required")
        finite_real(
            record["epi"],
            operator=_OPERATOR,
            label=f"{label}.epi",
            lower=0.0,
            upper=1.0,
        )
        for key in ("vf", "tau"):
            if key in record:
                finite_real(
                    record[key],
                    operator=_OPERATOR,
                    label=f"{label}.{key}",
                    lower=0.0,
                )
        if "d2_epi" in record:
            finite_real(
                record["d2_epi"],
                operator=_OPERATOR,
                label=f"{label}.d2_epi",
            )
        for key in ("timestamp", "bifurcation_level", "cascade_depth"):
            if key in record:
                nonnegative_integer(
                    record[key], operator=_OPERATOR, label=f"{label}.{key}"
                )
        if "metabolized" in record:
            strict_bool(
                record["metabolized"],
                operator=_OPERATOR,
                label=f"{label}.metabolized",
            )
        if "hierarchy_path" in record and not isinstance(
            record["hierarchy_path"], list
        ):
            reject_operator_argument(
                _OPERATOR, f"{label}.hierarchy_path must be a list"
            )
        _validate_signal_record(record.get("network_signals"), label=label)
        validated.append(record)
    return tuple(validated)


def _retarget_bifurcation(
    proposal: _BifurcationProposal, sub_node_id: str
) -> _BifurcationProposal:
    """Return one proposal with every child-identity reference updated."""

    if (
        not proposal.sub_nodes
        or not proposal.hierarchy_children
        or not proposal.sub_epis
    ):
        raise RuntimeError("THOL bifurcation proposal has incomplete child references")
    record = dict(proposal.sub_epis[-1])
    record["node_id"] = sub_node_id
    return replace(
        proposal,
        sub_node_id=sub_node_id,
        sub_nodes=(*proposal.sub_nodes[:-1], sub_node_id),
        hierarchy_children=(*proposal.hierarchy_children[:-1], sub_node_id),
        sub_epis=(*proposal.sub_epis[:-1], record),
    )


def _merge_stage_execution_proposals(
    snapshot: TNFRGraph,
    proposals: tuple[tuple[Any, _ExecutionProposal], ...],
) -> tuple[tuple[Any, _ExecutionProposal], ...]:
    """Allocate collision-free THOL children in immutable snapshot-node rank.

    A parent's ordinary direct-call identifier remains unchanged whenever it
    does not collide. Cross-target collisions are resolved by advancing that
    parent's local suffix. Snapshot rank, rather than requested target order,
    decides which proposal retains a colliding identifier.
    """

    rank = {node: index for index, node in enumerate(snapshot.nodes)}
    if any(node not in rank for node, _proposal in proposals):
        raise RuntimeError("THOL stage proposal contains a non-snapshot target")
    ordered = sorted(proposals, key=lambda item: rank[item[0]])
    reserved = set(snapshot.nodes)
    merged: dict[Any, _ExecutionProposal] = {}
    for node, proposal in ordered:
        bifurcation = proposal.bifurcation
        if bifurcation is None:
            merged[node] = proposal
            continue
        sub_index = len(bifurcation.sub_nodes) - 1
        sub_node_id = f"{node}_sub_{sub_index}"
        while sub_node_id in reserved:
            sub_index += 1
            sub_node_id = f"{node}_sub_{sub_index}"
        reserved.add(sub_node_id)
        if sub_node_id != bifurcation.sub_node_id:
            bifurcation = _retarget_bifurcation(bifurcation, sub_node_id)
            proposal = replace(proposal, bifurcation=bifurcation)
        merged[node] = proposal
    return tuple((node, merged[node]) for node, _proposal in proposals)


class SelfOrganization(Operator):
    """Spawn a coherent sub-EPI through one atomic U5 transaction."""

    __slots__ = ()
    name: ClassVar[str] = SELF_ORGANIZATION
    glyph: ClassVar[Glyph] = Glyph.THOL

    def _validate_application_preconditions(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        """Validate the optional public gate without writing its telemetry."""

        validate_common_execution_arguments(G.graph, kw, operator=_OPERATOR)
        collect_metrics = bool(kw.get("collect_metrics", False)) or bool(
            G.graph.get("COLLECT_OPERATOR_METRICS", False)
        )
        if collect_metrics:
            require_list_sink(G.graph, "operator_metrics", operator=_OPERATOR)
        monitor = G.graph.get("integrity_monitor")
        if monitor is not None and not all(
            callable(getattr(monitor, method, None))
            for method in ("before_operator", "after_operator")
        ):
            reject_operator_argument(
                _OPERATOR,
                "integrity_monitor must provide callable before_operator and "
                "after_operator methods",
            )
        requested = kw.get("validate_preconditions", True)
        enabled = G.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False)
        if requested and enabled:
            self._validate_preconditions(G, node, **kw)

    def _validate_preconditions(
        self, G: TNFRGraph, node: Any, **kw: Any
    ) -> None:
        """Apply the legacy THOL gate as a strictly read-only check."""

        data = G.nodes[node]
        epi = _finite_node_epi(data, label="EPI")
        dnfr = finite_node_real(
            data, ALIAS_DNFR, 0.0, operator=_OPERATOR, label="DeltaNFR"
        )
        vf = finite_node_real(
            data,
            ALIAS_VF,
            0.0,
            operator=_OPERATOR,
            label="nu_f",
            lower=0.0,
        )
        min_epi = finite_real(
            G.graph.get("THOL_MIN_EPI", 0.2),
            operator=_OPERATOR,
            label="THOL_MIN_EPI",
            lower=0.0,
        )
        min_vf = finite_real(
            G.graph.get("THOL_MIN_VF", 0.1),
            operator=_OPERATOR,
            label="THOL_MIN_VF",
            lower=0.0,
        )
        if epi < min_epi:
            reject_operator_argument(
                _OPERATOR, f"EPI too low for bifurcation ({epi!r} < {min_epi!r})"
            )
        if dnfr <= 0.0:
            reject_operator_argument(
                _OPERATOR, "DeltaNFR must be positive for self-organization"
            )
        if vf < min_vf:
            reject_operator_argument(
                _OPERATOR,
                f"nu_f too low for reorganization ({vf!r} < {min_vf!r})",
            )

        min_degree = nonnegative_integer(
            G.graph.get("THOL_MIN_DEGREE", 1),
            operator=_OPERATOR,
            label="THOL_MIN_DEGREE",
        )
        allow_isolated = strict_bool(
            G.graph.get("THOL_ALLOW_ISOLATED", False),
            operator=_OPERATOR,
            label="THOL_ALLOW_ISOLATED",
        )
        degree = G.degree(node)
        if degree < min_degree and not allow_isolated:
            reject_operator_argument(
                _OPERATOR,
                f"node degree {degree} is below THOL_MIN_DEGREE {min_degree}",
            )

        d2_epi = self._compute_epi_acceleration(G, node)
        min_history = nonnegative_integer(
            G.graph.get("THOL_MIN_HISTORY_LENGTH", 3),
            operator=_OPERATOR,
            label="THOL_MIN_HISTORY_LENGTH",
        )
        if min_history < 3:
            reject_operator_argument(
                _OPERATOR, "THOL_MIN_HISTORY_LENGTH must be at least 3"
            )
        history_length = _active_acceleration_history_length(data)
        if history_length < min_history:
            reject_operator_argument(
                _OPERATOR,
                f"active EPI history has {history_length} samples; "
                f"{min_history} required",
            )

        metabolic = strict_bool(
            G.graph.get("THOL_METABOLIC_ENABLED", True),
            operator=_OPERATOR,
            label="THOL_METABOLIC_ENABLED",
        )
        if metabolic and degree == 0:
            reject_operator_argument(
                _OPERATOR, "metabolic THOL requires at least one neighbour"
            )

        tau = _configured_tau(G.graph, kw)
        logger = logging.getLogger(__name__)
        if abs(d2_epi) <= tau:
            logger.warning(
                "Node %r: THOL acceleration magnitude %.6g does not exceed tau %.6g; "
                "no sub-EPI will be generated.",
                node,
                abs(d2_epi),
                tau,
            )

        from .preconditions.mutation import record_destabilizer_context

        record_destabilizer_context(G, node, logger, record=False)

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Validate, apply and observe THOL as one graph transaction."""

        proposal = self._prepare_execution(G, node, kw)
        snapshot = _GraphSnapshot(G)
        try:
            self._execute_transaction(G, node, proposal, kw)
        except BaseException as failure:
            monitor = G.graph.get("integrity_monitor")
            discard_pending = getattr(monitor, "discard_pending_operator", None)
            if callable(discard_pending):
                try:
                    discard_pending()
                except Exception:
                    pass
            snapshot.restore_after_failure(G, failure)
            raise

    def _execute_transaction(
        self,
        G: TNFRGraph,
        node: Any,
        proposal: _ExecutionProposal,
        kw: Mapping[str, Any],
    ) -> None:
        collect_metrics = bool(kw.get("collect_metrics", False)) or bool(
            G.graph.get("COLLECT_OPERATOR_METRICS", False)
        )
        validate_equation = bool(kw.get("validate_nodal_equation", False)) or bool(
            G.graph.get("VALIDATE_NODAL_EQUATION", False)
        )
        state_before = None
        if collect_metrics or validate_equation:
            state_before = self._capture_state(G, node)

        monitor = G.graph.get("integrity_monitor")
        if monitor is not None:
            monitor.before_operator(G, node)

        from . import _validated_execution_window
        from .network_stage import (
            PointwiseStageProposal,
            _record_histories_and_patterns,
        )

        # Planning is the single source of the direct and staged structural
        # action. Canonical setters preserve alias precedence and DeltaNFR cache
        # invalidation; the shared lifecycle helper preserves glyph provenance.
        self._commit_primary_channels(G, node, proposal)
        _record_histories_and_patterns(
            G,
            (
                PointwiseStageProposal(
                    node=node,
                    glyph=self.glyph,
                    payload=proposal,
                ),
            ),
            window=_validated_execution_window(G, kw.get("window")),
        )
        self._commit_proposal(G, node, proposal)

        if monitor is not None:
            monitor.after_operator(G, node, self.name)

        if validate_equation and state_before is not None:
            from .nodal_equation import validate_nodal_equation

            validate_nodal_equation(
                G,
                node,
                epi_before=state_before["epi"],
                epi_after=proposal.final_parent_epi,
                dt=float(kw.get("dt", 1.0)),
                operator_name=self.name,
                strict=bool(G.graph.get("NODAL_EQUATION_STRICT", False)),
            )

        if collect_metrics and state_before is not None:
            metrics = self._collect_metrics(G, node, state_before)
            G.graph.setdefault("operator_metrics", []).append(metrics)

    def _prepare_execution(
        self, G: TNFRGraph, node: Any, kw: Mapping[str, Any]
    ) -> _ExecutionProposal:
        validate_common_execution_arguments(G.graph, kw, operator=_OPERATOR)
        data = G.nodes[node]
        parent_epi = _finite_node_epi(data, label="EPI")
        parent_vf = finite_node_real(
            data,
            ALIAS_VF,
            0.0,
            operator=_OPERATOR,
            label="nu_f",
            lower=0.0,
        )
        parent_theta = finite_node_real(
            data, ALIAS_THETA, 0.0, operator=_OPERATOR, label="theta"
        )
        dnfr = finite_node_real(
            data, ALIAS_DNFR, 0.0, operator=_OPERATOR, label="DeltaNFR"
        )
        # The derivative reconstructed from active EPI history is
        # authoritative. ALIAS_D2EPI is cached telemetry and may be stale.
        d2_epi = self._compute_epi_acceleration(G, node)
        tau = _configured_tau(G.graph, kw)

        from .factor_contracts import resolve_runtime_operator_factors

        factors = resolve_runtime_operator_factors(
            G.graph.get("GLYPH_FACTORS"), self.glyph, G.graph
        )
        acceleration = finite_real(
            factors["THOL_accel"],
            operator=_OPERATOR,
            label="THOL_accel",
            lower=math.nextafter(0.0, math.inf),
        )
        contribution = _checked_product(
            acceleration, d2_epi, "THOL DeltaNFR contribution"
        )
        dnfr_after = _checked_sum(dnfr, contribution, "THOL DeltaNFR proposal")

        existing = _existing_list(data, "sub_epis")
        existing_records = _validate_sub_epi_records(existing)
        self._validate_subtree(G, node, frozenset())
        bifurcation = None
        depth_limit_reached = None
        final_records = existing_records
        final_parent_epi = parent_epi
        if abs(d2_epi) > tau:
            depth_limit_reached = self._depth_limit_diagnostic(G, node)
            if depth_limit_reached is None:
                bifurcation = self._prepare_bifurcation(
                    G,
                    node,
                    parent_epi=parent_epi,
                    parent_vf=parent_vf,
                    parent_theta=parent_theta,
                    d2_epi=d2_epi,
                    tau=tau,
                    existing_records=existing_records,
                )
                final_records = bifurcation.sub_epis

        amplitude_alignment = None
        if final_records:
            from .metabolism import _subepi_amplitude_alignment_from_records

            amplitude_alignment = _subepi_amplitude_alignment_from_records(
                final_records
            )

        preconditions_active = bool(
            kw.get("validate_preconditions", True)
        ) and bool(G.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False))
        context = None
        no_bifurcation = None
        if preconditions_active:
            from .preconditions.mutation import record_destabilizer_context

            context = record_destabilizer_context(G, node, record=False)
            no_bifurcation = bifurcation is None

        collect_metrics = bool(kw.get("collect_metrics", False)) or bool(
            G.graph.get("COLLECT_OPERATOR_METRICS", False)
        )
        if collect_metrics:
            require_list_sink(G.graph, "operator_metrics", operator=_OPERATOR)
            self._validate_metric_inputs(G, node, bifurcation, final_records)

        monitor = G.graph.get("integrity_monitor")
        if monitor is not None:
            for method in ("before_operator", "after_operator"):
                if not callable(getattr(monitor, method, None)):
                    reject_operator_argument(
                        _OPERATOR,
                        f"integrity_monitor.{method} must be callable",
                    )

        self._validate_nodal_proposal(
            G,
            kw,
            epi_before=parent_epi,
            epi_after=final_parent_epi,
            vf=parent_vf,
            dnfr=dnfr_after,
        )
        return _ExecutionProposal(
            d2_epi=d2_epi,
            tau=tau,
            dnfr_after=dnfr_after,
            bifurcation=bifurcation,
            subepi_amplitude_alignment=amplitude_alignment,
            precondition_context=context,
            no_bifurcation_expected=no_bifurcation,
            depth_limit_reached=depth_limit_reached,
            final_parent_epi=final_parent_epi,
        )

    def _depth_limit_diagnostic(
        self, G: TNFRGraph, node: Any
    ) -> Mapping[str, Any] | None:
        """Return a committed diagnostic when the configured child depth is full."""

        parent_level = nonnegative_integer(
            G.nodes[node].get("_bifurcation_level", 0),
            operator=_OPERATOR,
            label="_bifurcation_level",
        )
        max_depth = nonnegative_integer(
            G.graph.get("THOL_MAX_BIFURCATION_DEPTH", 5),
            operator=_OPERATOR,
            label="THOL_MAX_BIFURCATION_DEPTH",
        )
        if parent_level < max_depth:
            return None
        require_list_sink(G.graph, "thol_depth_limits", operator=_OPERATOR)
        return {
            "node": node,
            "depth": parent_level,
            "max_depth": max_depth,
        }

    def _prepare_bifurcation(
        self,
        G: TNFRGraph,
        node: Any,
        *,
        parent_epi: float,
        parent_vf: float,
        parent_theta: float,
        d2_epi: float,
        tau: float,
        existing_records: tuple[Mapping[str, Any], ...],
    ) -> _BifurcationProposal:
        data = G.nodes[node]
        legacy_propagation = strict_bool(
            G.graph.get("THOL_PROPAGATION_ENABLED", False),
            operator=_OPERATOR,
            label="THOL_PROPAGATION_ENABLED",
        )
        if legacy_propagation:
            reject_operator_argument(
                _OPERATOR,
                "THOL network propagation is outside the DeltaNFR channel; "
                "use Resonance in a grammar-valid operator word",
            )
        metabolic_enabled = strict_bool(
            G.graph.get("THOL_METABOLIC_ENABLED", True),
            operator=_OPERATOR,
            label="THOL_METABOLIC_ENABLED",
        )
        signals = None
        if metabolic_enabled:
            for neighbor in G.neighbors(node):
                _finite_node_epi(
                    G.nodes[neighbor], label=f"neighbor {neighbor!r} EPI"
                )
                finite_node_real(
                    G.nodes[neighbor],
                    ALIAS_THETA,
                    0.0,
                    operator=_OPERATOR,
                    label=f"neighbor {neighbor!r} theta",
                )
            from .metabolism import capture_network_signals

            signals = capture_network_signals(G, node)
            _validate_signal_record(signals, label="captured network signals")

        gradient_weight = COUPLING_MODERATE
        complexity_weight = COUPLING_GENTLE
        if signals is not None:
            gradient_weight = finite_real(
                G.graph.get(
                    "THOL_METABOLIC_GRADIENT_WEIGHT", COUPLING_MODERATE
                ),
                operator=_OPERATOR,
                label="THOL_METABOLIC_GRADIENT_WEIGHT",
                lower=0.0,
                upper=1.0,
            )
            complexity_weight = finite_real(
                G.graph.get(
                    "THOL_METABOLIC_COMPLEXITY_WEIGHT", COUPLING_GENTLE
                ),
                operator=_OPERATOR,
                label="THOL_METABOLIC_COMPLEXITY_WEIGHT",
                lower=0.0,
                upper=1.0,
            )
        from .metabolism import compose_subepi_amplitude

        raw_sub_epi = finite_real(
            compose_subepi_amplitude(
                parent_epi,
                signals,
                scaling_factor=THOL_SUB_EPI_SCALING,
                gradient_weight=gradient_weight,
                complexity_weight=complexity_weight,
            ),
            operator=_OPERATOR,
            label="sub-EPI amplitude proposal",
            lower=0.0,
            upper=1.0,
        )
        epi_min, epi_max = _configured_epi_bounds(G.graph)
        child_min = max(0.0, epi_min)
        child_max = min(1.0, epi_max)
        if child_min > child_max:
            reject_operator_argument(
                _OPERATOR,
                "configured EPI interval cannot represent a nonnegative sub-EPI",
            )
        sub_epi = finite_real(
            max(child_min, min(child_max, raw_sub_epi)),
            operator=_OPERATOR,
            label="bounded sub-EPI proposal",
            lower=child_min,
            upper=child_max,
        )

        parent_level = nonnegative_integer(
            data.get("_bifurcation_level", 0),
            operator=_OPERATOR,
            label="_bifurcation_level",
        )
        hierarchy_level = nonnegative_integer(
            data.get("hierarchy_level", 0),
            operator=_OPERATOR,
            label="hierarchy_level",
        )
        parent_path = data.get("_hierarchy_path", [])
        if not isinstance(parent_path, list):
            reject_operator_argument(_OPERATOR, "_hierarchy_path must be a list")
        if node in parent_path:
            reject_operator_argument(_OPERATOR, "_hierarchy_path contains a cycle")
        child_path = [*parent_path, node]
        child_level = parent_level + 1

        sub_nodes = _existing_list(data, "sub_nodes")
        for child in sub_nodes:
            if child not in G:
                reject_operator_argument(
                    _OPERATOR, f"sub-node reference {child!r} is missing"
                )
            if G.nodes[child].get("parent_node") != node:
                reject_operator_argument(
                    _OPERATOR, f"sub-node {child!r} has inconsistent parent identity"
                )
        sub_index = len(sub_nodes)
        sub_node_id = f"{node}_sub_{sub_index}"
        while sub_node_id in G:
            sub_index += 1
            sub_node_id = f"{node}_sub_{sub_index}"

        hierarchy = G.graph.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            reject_operator_argument(_OPERATOR, "hierarchy must be a dictionary")
        if node in hierarchy:
            hierarchy_children = hierarchy[node]
            if not isinstance(hierarchy_children, list):
                reject_operator_argument(
                    _OPERATOR, "hierarchy children must be stored in a list"
                )
            if tuple(hierarchy_children) != tuple(sub_nodes):
                reject_operator_argument(
                    _OPERATOR,
                    "graph hierarchy children must match parent sub_nodes",
                )
        else:
            # sub_nodes is the node-local U5 source of truth for legacy graphs
            # that predate the redundant graph-level hierarchy index.
            hierarchy_children = sub_nodes

        timestamp = next_operator_step(data)
        child_vf = _checked_product(
            parent_vf, THOL_CHILD_VF_DAMPING, "sub-node nu_f proposal"
        )
        child_theta = finite_real(
            parent_theta % math.tau,
            operator=_OPERATOR,
            label="sub-node theta proposal",
            lower=0.0,
            upper=math.tau,
        )
        from ..constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY

        sub_node_data = {
            EPI_PRIMARY: sub_epi,
            VF_PRIMARY: child_vf,
            THETA_PRIMARY: child_theta,
            DNFR_PRIMARY: 0.0,
            "parent_node": node,
            "hierarchy_level": hierarchy_level + 1,
            "_bifurcation_level": child_level,
            "_hierarchy_path": child_path,
            "epi_history": [sub_epi],
            "glyph_history": [],
        }
        record = {
            "epi": sub_epi,
            "vf": child_vf,
            "timestamp": timestamp,
            "d2_epi": d2_epi,
            "tau": tau,
            "node_id": sub_node_id,
            "metabolized": signals is not None,
            "network_signals": signals,
            "bifurcation_level": child_level,
            "hierarchy_path": child_path,
        }
        return _BifurcationProposal(
            sub_node_id=sub_node_id,
            sub_node_data=sub_node_data,
            sub_nodes=(*sub_nodes, sub_node_id),
            hierarchy_children=(*hierarchy_children, sub_node_id),
            sub_epis=(*existing_records, record),
        )

    def _validate_metric_inputs(
        self,
        G: TNFRGraph,
        node: Any,
        bifurcation: _BifurcationProposal | None,
        final_records: tuple[Mapping[str, Any], ...],
    ) -> None:
        if not final_records:
            _existing_list(G.graph, "sub_epi")
        self._validate_subtree(G, node, frozenset())

    def _validate_subtree(
        self, G: TNFRGraph, node: Any, ancestors: frozenset[Any]
    ) -> None:
        if node in ancestors:
            reject_operator_argument(_OPERATOR, "sub-EPI hierarchy contains a cycle")
        if node not in G:
            reject_operator_argument(_OPERATOR, f"sub-node {node!r} is missing")
        next_ancestors = ancestors | {node}
        data = G.nodes[node]
        _finite_node_epi(data, label=f"sub-node {node!r} EPI")
        finite_node_real(
            data,
            ALIAS_VF,
            0.0,
            operator=_OPERATOR,
            label=f"sub-node {node!r} nu_f",
            lower=0.0,
        )
        finite_node_real(
            data,
            ALIAS_DNFR,
            0.0,
            operator=_OPERATOR,
            label=f"sub-node {node!r} DeltaNFR",
        )
        finite_node_real(
            data,
            ALIAS_THETA,
            0.0,
            operator=_OPERATOR,
            label=f"sub-node {node!r} theta",
        )
        records = _existing_list(data, "sub_epis")
        validated_records = _validate_sub_epi_records(records)
        children = _existing_list(data, "sub_nodes")
        hierarchy = G.graph.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            reject_operator_argument(_OPERATOR, "hierarchy must be a dictionary")
        if node in hierarchy:
            hierarchy_children = hierarchy[node]
            if not isinstance(hierarchy_children, list):
                reject_operator_argument(
                    _OPERATOR, "hierarchy children must be stored in a list"
                )
            if tuple(hierarchy_children) != tuple(children):
                reject_operator_argument(
                    _OPERATOR,
                    "graph hierarchy children must match parent sub_nodes",
                )
        if children:
            for child in children:
                if child not in G:
                    reject_operator_argument(
                        _OPERATOR, f"sub-node reference {child!r} is missing"
                    )
                if G.nodes[child].get("parent_node") != node:
                    reject_operator_argument(
                        _OPERATOR,
                        f"sub-node {child!r} has inconsistent parent identity",
                    )
                self._validate_subtree(G, child, next_ancestors)
        for record in validated_records:
            child = record.get("node_id")
            if child is None:
                continue
            try:
                child_exists = child in G
                listed_child = child in children
            except TypeError:
                reject_operator_argument(
                    _OPERATOR, "sub-EPI node_id must be a hashable node identifier"
                )
            if not child_exists:
                reject_operator_argument(
                    _OPERATOR, f"sub-EPI node_id reference {child!r} is missing"
                )
            # Listed children were already traversed above. Record-only legacy
            # references remain supported, but receive the same full validation.
            if not listed_child:
                self._validate_subtree(G, child, next_ancestors)

    def _validate_nodal_proposal(
        self,
        G: TNFRGraph,
        kw: Mapping[str, Any],
        *,
        epi_before: float,
        epi_after: float,
        vf: float,
        dnfr: float,
    ) -> None:
        active = bool(kw.get("validate_nodal_equation", False)) or bool(
            G.graph.get("VALIDATE_NODAL_EQUATION", False)
        )
        if not active:
            return
        dt = finite_real(
            kw.get("dt", 1.0),
            operator=_OPERATOR,
            label="dt",
            lower=math.nextafter(0.0, math.inf),
        )
        tolerance = finite_real(
            G.graph.get("NODAL_EQUATION_TOLERANCE", 1e-3),
            operator=_OPERATOR,
            label="NODAL_EQUATION_TOLERANCE",
            lower=0.0,
        )
        clip_aware = strict_bool(
            G.graph.get("NODAL_EQUATION_CLIP_AWARE", True),
            operator=_OPERATOR,
            label="NODAL_EQUATION_CLIP_AWARE",
        )
        strict = strict_bool(
            G.graph.get("NODAL_EQUATION_STRICT", False),
            operator=_OPERATOR,
            label="NODAL_EQUATION_STRICT",
        )
        measured = finite_real(
            (epi_after - epi_before) / dt,
            operator=_OPERATOR,
            label="measured nodal derivative proposal",
        )
        expected = _checked_product(vf, dnfr, "expected nodal derivative proposal")
        if clip_aware:
            epi_min, epi_max = _configured_epi_bounds(G.graph)
            theoretical = _checked_sum(
                epi_before,
                _checked_product(expected, dt, "nodal EPI increment proposal"),
                "theoretical EPI proposal",
            )
            mode = str(G.graph.get("CLIP_MODE", "hard")).lower()
            if mode not in ("hard", "soft"):
                mode = "hard"
            from ..dynamics.structural_clip import structural_clip

            expected_epi = finite_real(
                structural_clip(theoretical, lo=epi_min, hi=epi_max, mode=mode),
                operator=_OPERATOR,
                label="bounded nodal EPI proposal",
            )
            error = finite_real(
                abs(epi_after - expected_epi),
                operator=_OPERATOR,
                label="nodal equation EPI error",
                lower=0.0,
            )
        else:
            error = finite_real(
                abs(measured - expected),
                operator=_OPERATOR,
                label="nodal equation derivative error",
                lower=0.0,
            )
        if strict and error > tolerance:
            from .nodal_equation import NodalEquationViolation

            raise NodalEquationViolation(
                operator=self.name,
                measured_depi_dt=measured,
                expected_depi_dt=expected,
                tolerance=tolerance,
                details={
                    "epi_before": epi_before,
                    "epi_after": epi_after,
                    "dt": dt,
                    "vf": vf,
                    "dnfr": dnfr,
                    "error": error,
                    "clip_aware": clip_aware,
                },
            )

    def _commit_primary_channels(
        self, G: TNFRGraph, node: Any, proposal: _ExecutionProposal
    ) -> None:
        """Commit proposed nodal channels through canonical alias boundaries."""

        from ..alias import set_attr, set_dnfr

        set_attr(G.nodes[node], ALIAS_D2EPI, proposal.d2_epi)
        set_dnfr(G, node, proposal.dnfr_after)

    def _commit_support_and_hierarchy(
        self, G: TNFRGraph, node: Any, proposal: _ExecutionProposal
    ) -> None:
        """Commit the already merged structural support for one target."""

        bifurcation = proposal.bifurcation
        if bifurcation is None:
            return
        G.add_node(bifurcation.sub_node_id, **dict(bifurcation.sub_node_data))
        G.nodes[node]["sub_nodes"] = list(bifurcation.sub_nodes)
        hierarchy = G.graph.setdefault("hierarchy", {})
        hierarchy[node] = list(bifurcation.hierarchy_children)
        G.nodes[node]["sub_epis"] = list(bifurcation.sub_epis)

    def _validate_merged_stage_support(
        self,
        candidate: TNFRGraph,
        proposals: tuple[tuple[Any, _ExecutionProposal], ...],
    ) -> None:
        """Materialize and validate all merged support on a detached candidate."""

        generated: set[Any] = set()
        for node, proposal in proposals:
            bifurcation = proposal.bifurcation
            if bifurcation is not None:
                if (
                    bifurcation.sub_node_id in candidate
                    or bifurcation.sub_node_id in generated
                ):
                    raise RuntimeError(
                        "THOL stage child identifier merge is not collision-free"
                    )
                generated.add(bifurcation.sub_node_id)
            self._commit_support_and_hierarchy(candidate, node, proposal)

        hierarchy = candidate.graph.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            raise RuntimeError("THOL stage hierarchy merge changed container type")
        for node, proposal in proposals:
            self._validate_subtree(candidate, node, frozenset())
            bifurcation = proposal.bifurcation
            if bifurcation is None:
                continue
            child = bifurcation.sub_node_id
            if candidate.nodes[child].get("parent_node") != node:
                raise RuntimeError("THOL stage child has inconsistent parent identity")
            if tuple(_existing_list(candidate.nodes[node], "sub_nodes")) != (
                bifurcation.sub_nodes
            ):
                raise RuntimeError("THOL stage sub-node merge changed")
            if tuple(hierarchy.get(node, ())) != bifurcation.hierarchy_children:
                raise RuntimeError("THOL stage hierarchy merge changed")
            if tuple(_existing_list(candidate.nodes[node], "sub_epis")) != (
                bifurcation.sub_epis
            ):
                raise RuntimeError("THOL stage sub-EPI merge changed")

    def _commit_lifecycle_proposal(
        self,
        G: TNFRGraph,
        node: Any,
        proposal: _ExecutionProposal,
        *,
        emit_depth_warning: bool = True,
    ) -> None:
        """Commit target-local diagnostics and ordered graph telemetry."""

        if proposal.depth_limit_reached is not None:
            diagnostic = dict(proposal.depth_limit_reached)
            G.nodes[node]["_thol_depth_limit_reached"] = True
            G.graph.setdefault("thol_depth_limits", []).append(diagnostic)
            if emit_depth_warning:
                self._emit_depth_limit_warning(node, proposal)

        if proposal.subepi_amplitude_alignment is not None:
            G.nodes[node]["_thol_subepi_amplitude_alignment"] = (
                proposal.subepi_amplitude_alignment
            )
        if proposal.precondition_context is not None:
            G.nodes[node]["_mutation_context"] = dict(proposal.precondition_context)
        if proposal.no_bifurcation_expected is not None:
            G.nodes[node]["_thol_no_bifurcation_expected"] = (
                proposal.no_bifurcation_expected
            )

    def _emit_depth_limit_warning(
        self, node: Any, proposal: _ExecutionProposal
    ) -> None:
        """Publish one accepted depth-limit warning."""

        diagnostic = proposal.depth_limit_reached
        if diagnostic is None:
            return
        logging.getLogger(__name__).warning(
            "Node %r: THOL child depth %d reached configured maximum %d; "
            "pressure was reorganized without creating another child.",
            node,
            diagnostic["depth"],
            diagnostic["max_depth"],
        )

    def _commit_proposal(
        self, G: TNFRGraph, node: Any, proposal: _ExecutionProposal
    ) -> None:
        """Commit support and lifecycle data for the direct public call."""

        self._commit_support_and_hierarchy(G, node, proposal)
        self._commit_lifecycle_proposal(G, node, proposal)

    def _compute_epi_acceleration(self, G: TNFRGraph, node: Any) -> float:
        """Read the signed value supplied by the shared structural derivative."""

        from .nodal_equation import compute_d2epi_dt2

        return finite_real(
            compute_d2epi_dt2(G, node, store=False),
            operator=_OPERATOR,
            label="signed EPI acceleration",
        )

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        from .metrics import self_organization_metrics

        return self_organization_metrics(
            G, node, state_before["epi"], state_before["vf"]
        )


_CANONICAL_SELF_ORGANIZATION_EXECUTE = SelfOrganization._execute
