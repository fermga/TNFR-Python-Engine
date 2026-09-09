"""Exact EPI realization boundary for immutable pointwise network stages.

The shared network executor already materializes frozen proposals for the
all-target AL, SHA, VAL, NUL, ZHIR and NAV Jacobi stages.  This module consumes
those proposals; it does not independently guess an operator action from a
name or from graph factors. Every payload is replayed through the same proposal
builder against the supplied stage-start graph. AL/SHA need their shared stage
timestamp, ZHIR needs its explicitly declared ``tau``, and NAV needs its stage
time and resolved RNG stream. Callers either supply those independent builder
inputs or receive an explicit abstention.

The represented affine map is a model over the exact rational values of its
binary64 coefficients. The exact residual against the already rounded runtime
proposal is reported separately. Level A binds that observed point without any
flow hypothesis. Level B adds an exact gain in the pre-flow metric, and level C
adds the common normalized pre/post metric bridge. Binary64 usability of the
exact gain is a separate diagnostic. This is stricter than a tolerance fit and
avoids turning one observed point into a false global runtime-affinity theorem.

The certificate covers the EPI jump only.  NUL also writes stored pressure, so
its inverse-pressure-map, nodal-drive, and post-jump pure-EPI-pressure defects
are reported independently and never folded into the EPI gain.

This integration seam is intentionally not re-exported by ``tnfr.physics``.
``execute_pointwise_stage`` can opt into the certificate and computes it from
the executor's own detached stage-start graph and frozen proposal tuple before
commit. The proposal tuple remains private, so callers cannot forge a post-hoc
reconstruction from an already mutated graph.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
import math
from typing import Any

import networkx as nx

from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..mathematics.unified_numerical import np
from ..operators.factor_contracts import resolve_runtime_operator_factors
from ..types import Glyph, real_scalar_epi
from ._exact_metric import exact_vectors_proportional_if_aligned
from ._neighbor_epi_realization import validate_certificate_tolerance
from .hybrid_operator_stability import (
    AffineEPIJumpGainCertificate,
    certify_affine_epi_jump_gain,
)
from .structural_diffusion import (
    HeterogeneousDiffusionStabilityCertificate,
    structural_diffusion_operator,
    verify_heterogeneous_diffusion_stability,
)

__all__ = [
    "PointwiseEPIJumpRealizationCertificate",
    "certify_pointwise_epi_jump_realization",
]


_SUPPORTED_GLYPHS = frozenset(
    {Glyph.AL, Glyph.SHA, Glyph.VAL, Glyph.NUL, Glyph.ZHIR, Glyph.NAV}
)
_UNSET = object()
_SCOPE = (
    "CONDITIONAL three-level certificate for one executor-produced immutable "
    "all-target AL/SHA/VAL/NUL/ZHIR/NAV two-phase Jacobi proposal. Level A "
    "binds the rounded runtime EPI result to an exact rational affine map of "
    "the represented binary64 coefficients, replayed from independently "
    "declared builder inputs. Level B adds an intact positive pre-flow metric, "
    "identity-aligned node order, consensus preservation and a finite exact "
    "gain. Level C adds fixed support, a controlled logical post-state, an "
    "intact aligned post-flow metric and exact pre/post proportionality. A "
    "positive proportional metric scale denotes the same normalized geometry; "
    "it never multiplies the affine gain. Capacity-dependent rates remain in "
    "the diffusion certificates. No grammar, histories, telemetry, monitors, "
    "pressure refresh, future repetition, mixed word, or global binary64 "
    "runtime-affinity theorem is claimed. NUL stored-pressure diagnostics are "
    "separate from the EPI gain."
)


def _readonly_array(value: Any) -> Any:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _exact_vector(value: Any) -> tuple[Fraction, ...]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("pointwise EPI state must be a finite vector")
    return tuple(Fraction.from_float(float(item)) for item in array)


def _exact_matrix(value: Any) -> tuple[tuple[Fraction, ...], ...]:
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("pointwise affine map must be a finite square matrix")
    return tuple(
        tuple(Fraction.from_float(float(item)) for item in row) for row in array
    )


def _exact_matrix_vector(
    matrix: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
    offset: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (coefficient * entry for coefficient, entry in zip(row, vector)),
            Fraction(0),
        )
        + bias
        for row, bias in zip(matrix, offset)
    )


def _finite_residual_norm(values: tuple[Fraction, ...]) -> float:
    maximum = max((abs(value) for value in values), default=Fraction(0))
    try:
        return float(maximum)
    except OverflowError:
        return float("inf")


def _exactly_proportional(left: Any, right: Any) -> bool:
    left_exact = _exact_vector(left)
    right_exact = _exact_vector(right)
    return exact_vectors_proportional_if_aligned(left_exact, right_exact)


def _well_typed_conditions(value: Any) -> bool:
    """Reject truthy integer substitutions in sealed condition vectors."""

    return bool(
        type(value) is tuple
        and all(
            type(condition) is tuple
            and len(condition) == 2
            and type(condition[0]) is str
            and type(condition[1]) is bool
            for condition in value
        )
    )


def _well_typed_fraction_vector(value: Any) -> bool:
    """Reject integer and Boolean aliases for exact rational vector fields."""

    return bool(
        type(value) is tuple
        and all(type(entry) is Fraction for entry in value)
    )


def _well_typed_fraction_matrix(value: Any) -> bool:
    """Require an immutable rectangular matrix of exact ``Fraction`` values."""

    if type(value) is not tuple:
        return False
    width = len(value)
    return all(
        _well_typed_fraction_vector(row) and len(row) == width
        for row in value
    )


def _well_typed_optional_fraction_vector(value: Any) -> bool:
    """Validate an optional exact rational diagnostic vector."""

    return value is None or _well_typed_fraction_vector(value)


def _effective_clip_mode(graph: Any) -> str:
    value = str(graph.graph.get("CLIP_MODE", "hard"))
    return value if value in ("hard", "soft") else "hard"


def _raw_scalar_epi(graph: Any, node: Any) -> float:
    marker = object()
    raw = get_attr(
        graph.nodes[node],
        ALIAS_EPI,
        marker,
        strict=True,
        conv=lambda value: value,
    )
    if raw is marker:
        raise ValueError(
            "pointwise realization requires explicit EPI on every graph node"
        )
    value = real_scalar_epi(raw)
    if value is None:
        raise ValueError(
            "pointwise realization requires raw scalar or uniform-real BEPI EPI"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("pointwise realization requires finite scalar EPI")
    return result


def _finite_node_value(
    graph: Any,
    node: Any,
    aliases: tuple[str, ...],
    label: str,
) -> float:
    try:
        result = float(get_attr(graph.nodes[node], aliases, 0.0, strict=True))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite binary64 scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite binary64 scalar")
    return result


def _logical_copy(graph: Any) -> nx.Graph:
    """Rebuild ``graph`` without trusting an overridable ``copy`` method.

    Only the top-level graph, node and edge mappings are detached.  Their
    values retain NetworkX's ordinary shallow-copy semantics because this
    module writes only scalar nodal aliases on the reconstructed graph.
    """

    if not isinstance(graph, nx.Graph):
        raise TypeError("pointwise realization requires a NetworkX graph")

    directed = bool(graph.is_directed())
    multigraph = bool(graph.is_multigraph())
    graph_type: type[nx.Graph]
    if multigraph:
        graph_type = nx.MultiDiGraph if directed else nx.MultiGraph
    else:
        graph_type = nx.DiGraph if directed else nx.Graph

    nodes = tuple(graph.nodes)
    rebuilt = graph_type()
    rebuilt.graph.update(dict(graph.graph))
    for node in nodes:
        rebuilt.add_node(node)
        rebuilt.nodes[node].update(dict(graph.nodes[node]))

    if multigraph:
        edges = tuple(graph.edges(keys=True, data=True))
        for left, right, key, data in edges:
            rebuilt.add_edge(left, right, key=key)
            rebuilt.edges[left, right, key].update(dict(data))
    else:
        edges = tuple(graph.edges(data=True))
        for left, right, data in edges:
            rebuilt.add_edge(left, right)
            rebuilt.edges[left, right].update(dict(data))

    if tuple(rebuilt.nodes) != nodes:
        raise RuntimeError("logical graph reconstruction changed node order")
    if (
        rebuilt.is_directed() != directed
        or rebuilt.is_multigraph() != multigraph
    ):
        raise RuntimeError("logical graph reconstruction changed graph kind")
    if rebuilt.number_of_edges() != graph.number_of_edges():
        raise RuntimeError("logical graph reconstruction changed edge support")
    if multigraph:
        expected_edges = tuple((left, right, key) for left, right, key, _ in edges)
        if tuple(rebuilt.edges(keys=True)) != expected_edges:
            raise RuntimeError("logical graph reconstruction changed edge order")
        for left, right, key, data in edges:
            if not rebuilt.has_edge(left, right, key):
                raise RuntimeError(
                    "logical graph reconstruction changed multiedge support"
                )
            if rebuilt.edges[left, right, key] is data:
                raise RuntimeError(
                    "logical graph reconstruction retained an edge mapping"
                )
    else:
        expected_edges = tuple((left, right) for left, right, _ in edges)
        if tuple(rebuilt.edges) != expected_edges:
            raise RuntimeError("logical graph reconstruction changed edge order")
        for left, right, data in edges:
            if not rebuilt.has_edge(left, right):
                raise RuntimeError("logical graph reconstruction changed edge support")
            if rebuilt.edges[left, right] is data:
                raise RuntimeError(
                    "logical graph reconstruction retained an edge mapping"
                )

    if rebuilt.graph is graph.graph or any(
        rebuilt.nodes[node] is graph.nodes[node] for node in nodes
    ):
        raise RuntimeError("logical graph reconstruction retained mutable mappings")
    return rebuilt


def _flow_certificate(
    graph: Any, tolerance: float
) -> tuple[HeterogeneousDiffusionStabilityCertificate | None, str | None]:
    try:
        return verify_heterogeneous_diffusion_stability(
            graph, tolerance=tolerance
        ), None
    except (
        ArithmeticError,
        nx.NetworkXException,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _proposal_stamp(proposal: Any) -> tuple[Any, ...]:
    """Return the EPI/capacity fields trusted after shared-builder replay."""

    glyph = proposal.glyph
    payload = proposal.payload
    if glyph is Glyph.AL:
        return (
            glyph.value,
            proposal.node,
            payload.epi_before,
            payload.epi_after,
            payload.initialize_emission,
            payload.emission_timestamp,
        )
    if glyph is Glyph.SHA:
        return (
            glyph.value,
            proposal.node,
            payload.vf_before,
            payload.vf_after,
            payload.preserved_epi,
            payload.latency_start_time,
        )
    if glyph in (Glyph.VAL, Glyph.NUL):
        return (
            glyph.value,
            proposal.node,
            payload.requested_scale,
            payload.vf_before,
            payload.vf_after,
            payload.epi_before,
            payload.raw_epi_after,
            payload.epi_after,
            payload.write_epi,
            payload.effective_epi_scale,
            payload.edge_aware_adapted,
            payload.clip_delta,
            payload.dnfr_before,
            payload.dnfr_after,
            payload.densification_factor,
            payload.binary64_inverse_product_residual,
        )
    if glyph is Glyph.ZHIR:
        gate = payload.runtime_gate
        threshold = gate.threshold
        return (
            glyph.value,
            proposal.node,
            threshold.previous_epi,
            threshold.current_epi,
            threshold.depi_dt,
            threshold.xi,
            threshold.history_key,
            threshold.sample_interval,
            threshold.time_basis,
            threshold.physical_time_resolved,
            gate.nu_f,
            gate.minimum_nu_f,
            payload.phase.theta_before,
            payload.phase.theta_after,
            payload.operator_step,
        )
    if glyph is Glyph.NAV:
        transition = payload.transition
        jitter = payload.jitter_proposal
        jitter_stamp = None
        if jitter is not None:
            jitter_stamp = tuple(
                getattr(jitter, name, None)
                for name in (
                    "value",
                    "seed",
                    "offset",
                    "draw_index",
                    "draws_after",
                )
            )
        return (
            glyph.value,
            proposal.node,
            transition.regime,
            transition.epi_before,
            transition.vf_before,
            transition.vf_after,
            transition.theta_before,
            transition.theta_after,
            transition.dnfr_before,
            payload.handler_dnfr_after,
            payload.dnfr_after,
            jitter_stamp,
        )
    raise ValueError(f"unsupported pointwise glyph {glyph!r}")


def _replay_proposals_from_declared_inputs(
    graph: Any,
    proposals: tuple[Any, ...],
    glyph: Glyph,
    *,
    stage_timestamp: str | None,
    zhir_tau: Any,
    nav_transition_now: datetime | None,
    nav_resolved_seed: int | None,
    nav_node_offsets: Mapping[Any, int] | None,
    nav_execution_kwargs: Mapping[str, Any] | None,
) -> tuple[bool, str | None]:
    """Replay payloads from independently declared executor inputs."""

    from ..operators.network_stage import _propose_pointwise

    factors = resolve_runtime_operator_factors(
        graph.graph.get("GLYPH_FACTORS"), glyph, graph.graph
    )
    if glyph in (Glyph.AL, Glyph.SHA) and stage_timestamp is None:
        return False, "AL/SHA replay requires the executor's shared stage_timestamp"
    if glyph is Glyph.ZHIR and zhir_tau is _UNSET:
        return False, "ZHIR replay requires an explicitly declared zhir_tau"
    if glyph is Glyph.NAV and nav_transition_now is None:
        return False, "NAV replay requires the executor's shared transition time"

    nav_offsets = dict(nav_node_offsets or {})
    nav_kwargs = dict(nav_execution_kwargs or {})
    nav_uses_rng = bool(
        glyph is Glyph.NAV
        and graph.graph.get("NAV_RANDOM", True)
        and float(factors["NAV_jitter"]) > 0.0
    )
    if nav_uses_rng and nav_resolved_seed is None:
        return False, "random NAV replay requires its resolved graph seed"
    if nav_uses_rng and any(
        nav_offsets.get(proposal.node) is None for proposal in proposals
    ):
        return False, "random NAV replay requires every target's node offset"
    operator: Any = None
    if glyph is Glyph.NAV:
        from ..operators.transition import Transition

        operator = Transition()

    for proposal in proposals:
        payload = proposal.payload
        timestamp: str | None = None
        tau: Any = None
        transition_now: datetime | None = None
        node_offset: int | None = None
        execution_kwargs: Mapping[str, Any] | None = None
        if glyph in (Glyph.AL, Glyph.SHA):
            timestamp = stage_timestamp
        elif glyph is Glyph.ZHIR:
            tau = zhir_tau
        elif glyph is Glyph.NAV:
            transition_now = nav_transition_now
            node_offset = nav_offsets.get(proposal.node)
            execution_kwargs = nav_kwargs

        try:
            expected = _propose_pointwise(
                graph,
                proposal.node,
                operator,
                glyph,
                factors,
                timestamp=timestamp,
                tau=tau,
                transition_now=transition_now,
                resolved_seed=nav_resolved_seed,
                node_offset=node_offset,
                execution_kwargs=execution_kwargs,
            )
        except Exception as exc:
            raise ValueError(
                f"{glyph.value} proposal cannot be replayed from the supplied "
                "stage-start graph"
            ) from exc
        if expected != proposal or _proposal_stamp(expected) != _proposal_stamp(
            proposal
        ):
            raise ValueError(
                f"{glyph.value} proposal was replaced, tampered with, or is "
                "stale relative to the supplied stage-start graph"
            )
    return True, None


def _certificate_stamp(
    *,
    operator_name: str,
    glyph: str,
    nodes: tuple[Any, ...],
    target_nodes: tuple[Any, ...],
    stage_schedule: str,
    fixed_support_declared: bool,
    state_before: Any,
    state_after: Any,
    linear_map: Any,
    offset: Any,
    exact_linear_map: tuple[tuple[Fraction, ...], ...],
    exact_offset: tuple[Fraction, ...],
    exact_residual: tuple[Fraction, ...],
    runtime_matches_exactly: bool,
    runtime_matches_within_tolerance: bool,
    clip_mode: str,
    clip_intervention_nodes: tuple[Any, ...],
    edge_adaptation_nodes: tuple[Any, ...],
    zero_post_capacity_nodes: tuple[Any, ...],
    proposal_builder_replayed: bool,
    proposal_replay_abstention_reason: str | None,
    logical_copy_verified: bool,
    pre_flow: HeterogeneousDiffusionStabilityCertificate | None,
    post_flow: HeterogeneousDiffusionStabilityCertificate | None,
    pre_flow_node_order_matches: bool,
    post_flow_node_order_matches: bool,
    metric_abstention_reason: str | None,
    flows_certified: bool,
    metrics_proportional: bool,
    exact_post_to_pre_metric_scale: Fraction | None,
    runtime_conditions: tuple[tuple[str, bool], ...],
    pre_gain_conditions: tuple[tuple[str, bool], ...],
    bridge_conditions: tuple[tuple[str, bool], ...],
    operational_conditions: tuple[tuple[str, bool], ...],
    runtime_realization_certified: bool,
    pre_metric_gain_certified: bool,
    common_metric_bridge_certified: bool,
    operational_affine_gain_available: bool,
    exact_energy_gain_bound: Fraction | None,
    exact_nul_pressure_residual: tuple[Fraction, ...] | None,
    exact_nul_drive_residual: tuple[Fraction, ...] | None,
    exact_nul_pressure_manifold_defect: tuple[Fraction, ...] | None,
    nul_pressure_defect_norm: float | None,
    nul_pressure_abstention_reason: str | None,
    affine_jump: AffineEPIJumpGainCertificate | None,
    tolerance: float,
    scope: str,
) -> tuple[Any, ...]:
    return (
        "pointwise_epi_jump_realization_v2",
        operator_name,
        glyph,
        nodes,
        target_nodes,
        stage_schedule,
        fixed_support_declared,
        _exact_vector(state_before),
        _exact_vector(state_after),
        _exact_matrix(linear_map),
        _exact_vector(offset),
        exact_linear_map,
        exact_offset,
        exact_residual,
        runtime_matches_exactly,
        runtime_matches_within_tolerance,
        clip_mode,
        clip_intervention_nodes,
        edge_adaptation_nodes,
        zero_post_capacity_nodes,
        proposal_builder_replayed,
        proposal_replay_abstention_reason,
        logical_copy_verified,
        None if pre_flow is None else pre_flow._proof_stamp,
        None if post_flow is None else post_flow._proof_stamp,
        pre_flow_node_order_matches,
        post_flow_node_order_matches,
        metric_abstention_reason,
        flows_certified,
        metrics_proportional,
        exact_post_to_pre_metric_scale,
        runtime_conditions,
        pre_gain_conditions,
        bridge_conditions,
        operational_conditions,
        runtime_realization_certified,
        pre_metric_gain_certified,
        common_metric_bridge_certified,
        operational_affine_gain_available,
        exact_energy_gain_bound,
        exact_nul_pressure_residual,
        exact_nul_drive_residual,
        exact_nul_pressure_manifold_defect,
        nul_pressure_defect_norm,
        nul_pressure_abstention_reason,
        None if affine_jump is None else affine_jump._proof_stamp,
        tolerance,
        scope,
    )


@dataclass(frozen=True, slots=True)
class PointwiseEPIJumpRealizationCertificate:
    """Three nested proof levels for one pointwise all-target stage.

    Runtime EPI realization (A) binds the observed proposal to the represented
    affine map.  Pre-metric affine gain (B) proves a finite global disagreement
    bound in the pre-flow metric.  The common-metric bridge (C) additionally
    proves that the post-flow metric is a positive scalar multiple of the pre
    metric.  This scalar identifies the same normalized geometry and is never
    multiplied into the common-metric affine gain.  Capacity-dependent flow
    rates remain in the two diffusion certificates.
    """

    operator_name: str
    glyph: str
    nodes: tuple[Any, ...]
    target_nodes: tuple[Any, ...]
    stage_schedule: str
    fixed_support_declared: bool
    state_before: Any
    runtime_proposed_state_after: Any
    represented_linear_map: Any
    represented_offset: Any
    exact_represented_linear_map: tuple[tuple[Fraction, ...], ...]
    exact_represented_offset: tuple[Fraction, ...]
    exact_runtime_minus_represented_affine: tuple[Fraction, ...]
    runtime_matches_represented_affine_exactly: bool
    runtime_matches_represented_affine_within_tolerance: bool
    clip_mode: str
    clip_intervention_nodes: tuple[Any, ...]
    edge_adaptation_nodes: tuple[Any, ...]
    zero_post_capacity_nodes: tuple[Any, ...]
    proposal_builder_replayed_from_declared_inputs: bool
    proposal_replay_abstention_reason: str | None
    logical_copy_verified: bool
    pre_diffusion_certificate: HeterogeneousDiffusionStabilityCertificate | None
    post_diffusion_certificate: HeterogeneousDiffusionStabilityCertificate | None
    pre_flow_node_order_matches: bool
    post_flow_node_order_matches: bool
    diffusion_metric_abstention_reason: str | None
    pre_and_post_diffusion_flows_certified: bool
    pre_post_metric_exactly_proportional: bool
    exact_post_to_pre_metric_scale: Fraction | None
    exact_nul_stored_pressure_map_residual: tuple[Fraction, ...] | None
    exact_nul_nodal_drive_residual: tuple[Fraction, ...] | None
    exact_nul_stored_pressure_minus_pure_epi_pressure: (
        tuple[Fraction, ...] | None
    )
    nul_stored_pressure_defect_norm: float | None
    nul_pressure_diagnostic_abstention_reason: str | None
    runtime_epi_realization_conditions: tuple[tuple[str, bool], ...]
    pre_metric_affine_gain_conditions: tuple[tuple[str, bool], ...]
    pre_post_common_metric_bridge_conditions: tuple[tuple[str, bool], ...]
    operational_affine_gain_conditions: tuple[tuple[str, bool], ...]
    affine_jump_certificate: AffineEPIJumpGainCertificate | None
    exact_common_metric_energy_gain_bound: Fraction | None
    operational_affine_gain_available: bool
    runtime_epi_realization_certified: bool
    pre_metric_affine_gain_certified: bool
    pre_post_common_metric_bridge_certified: bool
    tolerance: float
    scope: str
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        decisive_flags = (
            self.fixed_support_declared,
            self.runtime_matches_represented_affine_exactly,
            self.runtime_matches_represented_affine_within_tolerance,
            self.proposal_builder_replayed_from_declared_inputs,
            self.logical_copy_verified,
            self.pre_flow_node_order_matches,
            self.post_flow_node_order_matches,
            self.pre_and_post_diffusion_flows_certified,
            self.pre_post_metric_exactly_proportional,
            self.operational_affine_gain_available,
            self.runtime_epi_realization_certified,
            self.pre_metric_affine_gain_certified,
            self.pre_post_common_metric_bridge_certified,
        )
        required_tuples = (
            self.nodes,
            self.target_nodes,
            self.exact_represented_linear_map,
            self.exact_represented_offset,
            self.exact_runtime_minus_represented_affine,
            self.clip_intervention_nodes,
            self.edge_adaptation_nodes,
            self.zero_post_capacity_nodes,
            self._proof_stamp,
        )
        condition_vectors = (
            self.runtime_epi_realization_conditions,
            self.pre_metric_affine_gain_conditions,
            self.pre_post_common_metric_bridge_conditions,
            self.operational_affine_gain_conditions,
        )
        required_strings = (
            self.operator_name,
            self.glyph,
            self.stage_schedule,
            self.clip_mode,
            self.scope,
        )
        optional_strings = (
            self.proposal_replay_abstention_reason,
            self.diffusion_metric_abstention_reason,
            self.nul_pressure_diagnostic_abstention_reason,
        )
        exact_gain_well_typed = bool(
            self.exact_common_metric_energy_gain_bound is None
            or (
                type(self.exact_common_metric_energy_gain_bound) is Fraction
                and self.exact_common_metric_energy_gain_bound >= 0
            )
        )
        metric_scale_well_typed = bool(
            self.exact_post_to_pre_metric_scale is None
            or (
                type(self.exact_post_to_pre_metric_scale) is Fraction
                and self.exact_post_to_pre_metric_scale > 0
            )
        )
        if (
            any(type(flag) is not bool for flag in decisive_flags)
            or any(type(value) is not tuple for value in required_tuples)
            or any(type(value) is not str for value in required_strings)
            or any(
                value is not None and type(value) is not str
                for value in optional_strings
            )
            or not _well_typed_fraction_matrix(
                self.exact_represented_linear_map
            )
            or not _well_typed_fraction_vector(
                self.exact_represented_offset
            )
            or not _well_typed_fraction_vector(
                self.exact_runtime_minus_represented_affine
            )
            or not _well_typed_optional_fraction_vector(
                self.exact_nul_stored_pressure_map_residual
            )
            or not _well_typed_optional_fraction_vector(
                self.exact_nul_nodal_drive_residual
            )
            or not _well_typed_optional_fraction_vector(
                self.exact_nul_stored_pressure_minus_pure_epi_pressure
            )
            or not exact_gain_well_typed
            or not metric_scale_well_typed
            or type(self.tolerance) is not float
            or not math.isfinite(self.tolerance)
            or not 0.0 < self.tolerance < 1.0
            or (
                self.nul_stored_pressure_defect_norm is not None
                and type(self.nul_stored_pressure_defect_norm) is not float
            )
            or not all(
                _well_typed_conditions(value) for value in condition_vectors
            )
        ):
            return False
        try:
            expected = _certificate_stamp(
                operator_name=self.operator_name,
                glyph=self.glyph,
                nodes=self.nodes,
                target_nodes=self.target_nodes,
                stage_schedule=self.stage_schedule,
                fixed_support_declared=self.fixed_support_declared,
                state_before=self.state_before,
                state_after=self.runtime_proposed_state_after,
                linear_map=self.represented_linear_map,
                offset=self.represented_offset,
                exact_linear_map=self.exact_represented_linear_map,
                exact_offset=self.exact_represented_offset,
                exact_residual=self.exact_runtime_minus_represented_affine,
                runtime_matches_exactly=(
                    self.runtime_matches_represented_affine_exactly
                ),
                runtime_matches_within_tolerance=(
                    self.runtime_matches_represented_affine_within_tolerance
                ),
                clip_mode=self.clip_mode,
                clip_intervention_nodes=self.clip_intervention_nodes,
                edge_adaptation_nodes=self.edge_adaptation_nodes,
                zero_post_capacity_nodes=self.zero_post_capacity_nodes,
                proposal_builder_replayed=(
                    self.proposal_builder_replayed_from_declared_inputs
                ),
                proposal_replay_abstention_reason=(
                    self.proposal_replay_abstention_reason
                ),
                logical_copy_verified=self.logical_copy_verified,
                pre_flow=self.pre_diffusion_certificate,
                post_flow=self.post_diffusion_certificate,
                pre_flow_node_order_matches=self.pre_flow_node_order_matches,
                post_flow_node_order_matches=self.post_flow_node_order_matches,
                metric_abstention_reason=(
                    self.diffusion_metric_abstention_reason
                ),
                flows_certified=(
                    self.pre_and_post_diffusion_flows_certified
                ),
                metrics_proportional=(
                    self.pre_post_metric_exactly_proportional
                ),
                exact_post_to_pre_metric_scale=(
                    self.exact_post_to_pre_metric_scale
                ),
                runtime_conditions=self.runtime_epi_realization_conditions,
                pre_gain_conditions=self.pre_metric_affine_gain_conditions,
                bridge_conditions=(
                    self.pre_post_common_metric_bridge_conditions
                ),
                operational_conditions=(
                    self.operational_affine_gain_conditions
                ),
                runtime_realization_certified=(
                    self.runtime_epi_realization_certified
                ),
                pre_metric_gain_certified=(
                    self.pre_metric_affine_gain_certified
                ),
                common_metric_bridge_certified=(
                    self.pre_post_common_metric_bridge_certified
                ),
                operational_affine_gain_available=(
                    self.operational_affine_gain_available
                ),
                exact_energy_gain_bound=(
                    self.exact_common_metric_energy_gain_bound
                ),
                exact_nul_pressure_residual=(
                    self.exact_nul_stored_pressure_map_residual
                ),
                exact_nul_drive_residual=self.exact_nul_nodal_drive_residual,
                exact_nul_pressure_manifold_defect=(
                    self.exact_nul_stored_pressure_minus_pure_epi_pressure
                ),
                nul_pressure_defect_norm=self.nul_stored_pressure_defect_norm,
                nul_pressure_abstention_reason=(
                    self.nul_pressure_diagnostic_abstention_reason
                ),
                affine_jump=self.affine_jump_certificate,
                tolerance=self.tolerance,
                scope=self.scope,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        nested = self.affine_jump_certificate
        pre_flow = self.pre_diffusion_certificate
        post_flow = self.post_diffusion_certificate
        nested_matches_outer = nested is None
        if nested is not None:
            try:
                from ..operators.operator_contracts import contract_for

                contract = contract_for(self.glyph)
                nested_matches_outer = bool(
                    type(nested.exact_consensus_subspace_preservation) is bool
                    and type(nested.exact_weighted_mean_preservation) is bool
                    and type(nested.finite_global_energy_gain) is bool
                    and type(nested.declared_energy_gain_bound_certified)
                    in (bool, type(None))
                    and type(nested.declared_bound_certified_by_frobenius)
                    in (bool, type(None))
                    and type(nested.declared_bound_within_tolerance)
                    in (bool, type(None))
                    and nested.operator_name == self.operator_name
                    and nested.glyph == self.glyph
                    and nested.primary_channel == contract.primary_channel.value
                    and nested.operator_scale == contract.scale.value
                    and tuple(nested.nodes) == self.nodes
                    and _exact_matrix(nested.linear_map)
                    == self.exact_represented_linear_map
                    and _exact_vector(nested.offset)
                    == self.exact_represented_offset
                    and pre_flow is not None
                    and _exact_vector(nested.metric_weights)
                    == _exact_vector(pre_flow.metric_weights)
                    and nested.exact_quotient_energy_gain_upper_bound
                    == self.exact_common_metric_energy_gain_bound
                )
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                nested_matches_outer = False
        return bool(
            self._proof_stamp == expected
            and (
                pre_flow is None
                or (
                    isinstance(
                        pre_flow, HeterogeneousDiffusionStabilityCertificate
                    )
                    and pre_flow._proof_fields_are_intact()
                )
            )
            and (
                post_flow is None
                or (
                    isinstance(
                        post_flow, HeterogeneousDiffusionStabilityCertificate
                    )
                    and post_flow._proof_fields_are_intact()
                )
            )
            and (
                nested is None
                or (
                    isinstance(nested, AffineEPIJumpGainCertificate)
                    and nested._proof_fields_are_intact()
                    and nested_matches_outer
                )
            )
        )

    @property
    def supports_runtime_affine_gain(self) -> bool:
        """Whether level B proves an exact gain in the pre-flow metric."""

        return bool(
            self._proof_fields_are_intact()
            and self.pre_metric_affine_gain_certified
            and self.affine_jump_certificate is not None
            and self.exact_common_metric_energy_gain_bound is not None
        )

    @property
    def supports_hybrid_common_metric_bridge(self) -> bool:
        """Whether level C joins the certified pre/post diffusion metrics."""

        return bool(
            self._proof_fields_are_intact()
            and self.pre_post_common_metric_bridge_certified
            and self.supports_runtime_affine_gain
        )

    @staticmethod
    def _failed(
        conditions: Any,
    ) -> tuple[str, ...]:
        if not _well_typed_conditions(conditions):
            return ("condition_vector_well_typed",)
        return tuple(name for name, passed in conditions if not passed)

    @property
    def failed_runtime_epi_realization_conditions(self) -> tuple[str, ...]:
        """Failed hypotheses for level A."""

        return self._failed(self.runtime_epi_realization_conditions)

    @property
    def failed_pre_metric_affine_gain_conditions(self) -> tuple[str, ...]:
        """Failed level A/B hypotheses for the pre-metric gain."""

        return (
            self._failed(self.runtime_epi_realization_conditions)
            + self._failed(self.pre_metric_affine_gain_conditions)
        )

    @property
    def failed_common_metric_bridge_conditions(self) -> tuple[str, ...]:
        """Failed level A/B/C hypotheses for hybrid composition."""

        return (
            self._failed(self.runtime_epi_realization_conditions)
            + self._failed(self.pre_metric_affine_gain_conditions)
            + self._failed(self.pre_post_common_metric_bridge_conditions)
        )

    @property
    def failed_operational_affine_gain_conditions(self) -> tuple[str, ...]:
        """Failed binary64-usability diagnostics for the exact gain."""

        return self._failed(self.operational_affine_gain_conditions)

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Return level-C failures and separate operational limitations."""

        failures = (
            self.failed_common_metric_bridge_conditions
            + self.failed_operational_affine_gain_conditions
        )
        if not self._proof_fields_are_intact():
            return failures + ("certificate_proof_fields_intact",)
        return failures


def certify_pointwise_epi_jump_realization(
    graph: Any,
    proposals: Iterable[Any],
    *,
    fixed_support_declared: bool,
    stage_schedule: str = "two_phase_jacobi",
    stage_timestamp: str | None = None,
    zhir_tau: Any = _UNSET,
    nav_transition_now: datetime | None = None,
    nav_resolved_seed: int | None = None,
    nav_node_offsets: Mapping[Any, int] | None = None,
    nav_execution_kwargs: Mapping[str, Any] | None = None,
    tolerance: float = 1e-10,
) -> PointwiseEPIJumpRealizationCertificate:
    """Certify one frozen pointwise stage's represented affine EPI jump.

    ``proposals`` must be the frozen ``PointwiseStageProposal`` objects built
    from the still-current stage-start graph. AL/SHA callers must declare the
    shared stage timestamp, ZHIR callers must explicitly declare ``zhir_tau``
    (where ``None`` means the configured default), and NAV callers must provide
    the time, resolved seed/offsets and execution arguments used by the
    executor. Missing exogenous inputs cause a level-A abstention.

    The executor keeps these objects private and invokes this function through
    its opt-in ``epi_jump_fixed_support_declared`` hook. Direct use remains an
    internal contract boundary rather than a post-hoc reconstruction API.
    """

    if type(fixed_support_declared) is not bool:
        raise TypeError("fixed_support_declared must be a bool")
    if not isinstance(stage_schedule, str):
        raise TypeError("stage_schedule must be a string")
    tol = validate_certificate_tolerance(tolerance)

    from ..operators.network_stage import PointwiseStageProposal
    from ..operators.operator_contracts import contract_for

    proposal_tuple = tuple(proposals)
    if not proposal_tuple:
        raise ValueError("pointwise realization requires at least one proposal")
    if any(
        not isinstance(proposal, PointwiseStageProposal)
        for proposal in proposal_tuple
    ):
        raise TypeError(
            "proposals must contain frozen PointwiseStageProposal values"
        )
    glyph = proposal_tuple[0].glyph
    if glyph not in _SUPPORTED_GLYPHS:
        raise ValueError(
            "pointwise realization supports AL, SHA, VAL, NUL, ZHIR and NAV"
        )
    if any(proposal.glyph is not glyph for proposal in proposal_tuple):
        raise ValueError("one pointwise realization cannot mix glyphs")
    for proposal in proposal_tuple:
        if getattr(proposal.payload, "glyph", None) is not glyph:
            raise ValueError("pointwise proposal payload glyph is inconsistent")
        payload_node = getattr(proposal.payload, "node", proposal.node)
        if payload_node != proposal.node:
            raise ValueError("pointwise proposal payload target is inconsistent")

    target_nodes = tuple(proposal.node for proposal in proposal_tuple)
    if len(set(target_nodes)) != len(target_nodes):
        raise ValueError("pointwise realization targets must be unique")
    nodes = tuple(graph.nodes)
    if any(node not in graph for node in target_nodes):
        raise ValueError("pointwise proposal target is absent from graph")
    by_node = {proposal.node: proposal for proposal in proposal_tuple}
    full_target_coverage = len(target_nodes) == len(nodes) and all(
        node in by_node for node in nodes
    )

    state_before_values = tuple(_raw_scalar_epi(graph, node) for node in nodes)
    state_after_values = list(state_before_values)
    vf_before_values = tuple(
        _finite_node_value(graph, node, ALIAS_VF, "nu_f") for node in nodes
    )
    vf_after_values = list(vf_before_values)
    linear = np.eye(len(nodes), dtype=float)
    offset = np.zeros(len(nodes), dtype=float)
    clip_nodes: list[Any] = []
    adaptation_nodes: list[Any] = []
    clip_mode = "inactive"

    replayed, replay_reason = _replay_proposals_from_declared_inputs(
        graph,
        proposal_tuple,
        glyph,
        stage_timestamp=stage_timestamp,
        zhir_tau=zhir_tau,
        nav_transition_now=nav_transition_now,
        nav_resolved_seed=nav_resolved_seed,
        nav_node_offsets=nav_node_offsets,
        nav_execution_kwargs=nav_execution_kwargs,
    )
    factors = resolve_runtime_operator_factors(
        graph.graph.get("GLYPH_FACTORS"), glyph, graph.graph
    )

    exact_nul_pressure_residual: list[Fraction] | None = (
        [Fraction(0) for _ in nodes] if glyph is Glyph.NUL else None
    )
    exact_nul_drive_residual: list[Fraction] | None = (
        [Fraction(0) for _ in nodes] if glyph is Glyph.NUL else None
    )

    for index, node in enumerate(nodes):
        proposal = by_node.get(node)
        if proposal is None:
            continue
        payload = proposal.payload
        if glyph is Glyph.AL:
            if payload.epi_before != state_before_values[index]:
                raise ValueError("AL proposal EPI snapshot is inconsistent")
            boost = float(factors["AL_boost"])
            raw_after = float(payload.epi_before + boost)
            state_after_values[index] = float(payload.epi_after)
            offset[index] = boost
            clip_mode = _effective_clip_mode(graph)
            if raw_after != float(payload.epi_after):
                clip_nodes.append(node)
        elif glyph is Glyph.SHA:
            if payload.preserved_epi != state_before_values[index]:
                raise ValueError("SHA proposal EPI snapshot is inconsistent")
            if payload.vf_before != vf_before_values[index]:
                raise ValueError("SHA proposal capacity snapshot is inconsistent")
            vf_after_values[index] = float(payload.vf_after)
        elif glyph in (Glyph.VAL, Glyph.NUL):
            if payload.vf_before != vf_before_values[index]:
                raise ValueError("scale proposal capacity snapshot is inconsistent")
            vf_after_values[index] = float(payload.vf_after)
            if payload.write_epi:
                if payload.epi_before != state_before_values[index]:
                    raise ValueError("scale proposal EPI snapshot is inconsistent")
                if (
                    payload.raw_epi_after is None
                    or payload.epi_after is None
                    or payload.effective_epi_scale is None
                ):
                    raise ValueError(
                        "EPI-writing scale proposal lacks its required EPI fields"
                    )
                state_after_values[index] = float(payload.epi_after)
                linear[index, index] = float(payload.requested_scale)
                clip_mode = _effective_clip_mode(graph)
                if float(payload.raw_epi_after) != float(payload.epi_after):
                    clip_nodes.append(node)
                if (
                    payload.edge_aware_adapted
                    or Fraction.from_float(float(payload.effective_epi_scale))
                    != Fraction.from_float(float(payload.requested_scale))
                ):
                    adaptation_nodes.append(node)
            if glyph is Glyph.NUL:
                if (
                    exact_nul_pressure_residual is None
                    or exact_nul_drive_residual is None
                ):
                    raise ValueError(
                        "NUL residual storage was not initialized"
                    )
                if (
                    payload.dnfr_before is None
                    or payload.dnfr_after is None
                    or payload.densification_factor is None
                ):
                    raise ValueError(
                        "NUL proposal lacks its required pressure fields"
                    )
                stored_before = _finite_node_value(
                    graph, node, ALIAS_DNFR, "NUL stored DeltaNFR"
                )
                if stored_before != payload.dnfr_before:
                    raise ValueError("NUL proposal pressure snapshot is inconsistent")
                before_q = Fraction.from_float(float(payload.dnfr_before))
                after_q = Fraction.from_float(float(payload.dnfr_after))
                inverse_q = Fraction.from_float(
                    float(payload.densification_factor)
                )
                exact_nul_pressure_residual[index] = (
                    after_q - before_q * inverse_q
                )
                exact_nul_drive_residual[index] = (
                    Fraction.from_float(float(payload.vf_after)) * after_q
                    - Fraction.from_float(float(payload.vf_before)) * before_q
                )
        elif glyph is Glyph.ZHIR:
            threshold = payload.runtime_gate.threshold
            if threshold.current_epi != state_before_values[index]:
                raise ValueError("ZHIR gate EPI snapshot is inconsistent")
            if payload.runtime_gate.nu_f != vf_before_values[index]:
                raise ValueError("ZHIR gate capacity snapshot is inconsistent")
        else:
            transition = payload.transition
            if transition.epi_before != state_before_values[index]:
                raise ValueError("NAV proposal EPI snapshot is inconsistent")
            if transition.vf_before != vf_before_values[index]:
                raise ValueError("NAV proposal capacity snapshot is inconsistent")
            vf_after_values[index] = float(transition.vf_after)

    state_before = _readonly_array(state_before_values)
    state_after = _readonly_array(state_after_values)
    represented_map = _readonly_array(linear)
    represented_offset = _readonly_array(offset)
    exact_map = _exact_matrix(represented_map)
    exact_offset = _exact_vector(represented_offset)
    exact_before = _exact_vector(state_before)
    exact_after = _exact_vector(state_after)
    exact_model_after = _exact_matrix_vector(
        exact_map, exact_before, exact_offset
    )
    exact_residual = tuple(
        runtime - model
        for runtime, model in zip(exact_after, exact_model_after)
    )
    exact_runtime_match = all(value == 0 for value in exact_residual)
    residual_norm = _finite_residual_norm(exact_residual)
    model_norm = _finite_residual_norm(exact_model_after)
    scale = max(
        1.0,
        *(abs(value) for value in state_after_values),
        model_norm,
    )
    within_tolerance = bool(
        math.isfinite(residual_norm)
        and math.isfinite(scale)
        and residual_norm <= tol * scale
    )

    row_sums = tuple(sum(row, Fraction(0)) for row in exact_map)
    exact_consensus = bool(
        row_sums
        and all(value == row_sums[0] for value in row_sums)
        and all(value == exact_offset[0] for value in exact_offset)
    )

    post_graph = _logical_copy(graph)
    logical_copy_verified = True
    for index, node in enumerate(nodes):
        set_attr(post_graph.nodes[node], ALIAS_EPI, state_after_values[index])
        set_attr(post_graph.nodes[node], ALIAS_VF, vf_after_values[index])
        proposal = by_node.get(node)
        if proposal is not None and glyph is Glyph.NUL:
            set_attr(
                post_graph.nodes[node],
                ALIAS_DNFR,
                float(proposal.payload.dnfr_after),
            )

    pre_flow, pre_reason = _flow_certificate(graph, tol)
    post_flow, post_reason = _flow_certificate(post_graph, tol)
    pre_order_matches = bool(
        pre_flow is not None and tuple(pre_flow.nodes) == nodes
    )
    post_order_matches = bool(
        post_flow is not None and tuple(post_flow.nodes) == nodes
    )
    pre_flow_certified = bool(
        pre_flow is not None
        and pre_order_matches
        and pre_flow.is_certified
        and pre_flow._proof_fields_are_intact()
    )
    post_flow_certified = bool(
        post_flow is not None
        and post_order_matches
        and post_flow.is_certified
        and post_flow._proof_fields_are_intact()
    )
    flows_certified = bool(
        pre_flow_certified and post_flow_certified
    )
    metrics_proportional = bool(
        flows_certified
        and _exactly_proportional(
            pre_flow.metric_weights, post_flow.metric_weights
        )
    )
    exact_metric_scale: Fraction | None = None
    if metrics_proportional:
        pre_metric_exact = _exact_vector(pre_flow.metric_weights)
        post_metric_exact = _exact_vector(post_flow.metric_weights)
        exact_metric_scale = post_metric_exact[0] / pre_metric_exact[0]

    metric_reason = None
    if pre_flow is None or post_flow is None:
        metric_reason = "; ".join(
            reason
            for reason in (
                None if pre_reason is None else f"pre: {pre_reason}",
                None if post_reason is None else f"post: {post_reason}",
            )
            if reason is not None
        )
    elif not pre_order_matches or not post_order_matches:
        metric_reason = (
            "pre/post diffusion node order does not match the stage identity order"
        )
    elif not flows_certified:
        metric_reason = (
            "pre/post metrics exist, but at least one represented pure-EPI "
            "diffusion flow is not certified"
        )
    elif not metrics_proportional:
        metric_reason = (
            "pre/post binary64 diffusion metrics are not exactly proportional"
        )

    zero_post_capacity = tuple(
        node
        for node, value in zip(nodes, vf_after_values)
        if value <= 0.0
    )

    exact_pressure_manifold_defect: tuple[Fraction, ...] | None = None
    pressure_defect_norm: float | None = None
    pressure_abstention_reason: str | None = None
    if glyph is Glyph.NUL:
        try:
            pressure_nodes, laplacian = structural_diffusion_operator(post_graph)
            if tuple(pressure_nodes) != nodes:
                raise ValueError("pure-EPI pressure node order changed")
            exact_laplacian = _exact_matrix(laplacian)
            exact_zero = tuple(Fraction(0) for _ in nodes)
            exact_laplacian_state = _exact_matrix_vector(
                exact_laplacian,
                exact_after,
                exact_zero,
            )
            exact_pure_pressure = tuple(
                -value for value in exact_laplacian_state
            )
            exact_stored_pressure = tuple(
                Fraction.from_float(
                    _finite_node_value(
                        post_graph, node, ALIAS_DNFR, "stored DeltaNFR"
                    )
                )
                for node in nodes
            )
            exact_pressure_manifold_defect = tuple(
                stored - pure
                for stored, pure in zip(
                    exact_stored_pressure,
                    exact_pure_pressure,
                    strict=True,
                )
            )
            pressure_defect_norm = _finite_residual_norm(
                exact_pressure_manifold_defect
            )
        except (
            ArithmeticError,
            nx.NetworkXException,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            exact_pressure_manifold_defect = None
            pressure_defect_norm = None
            pressure_abstention_reason = f"{type(exc).__name__}: {exc}"

    supported_clip_policy = clip_mode != "soft"
    gate_bound = replayed
    runtime_conditions = (
        ("two_phase_jacobi_declared", stage_schedule == "two_phase_jacobi"),
        ("all_graph_nodes_targeted_once", full_target_coverage),
        ("proposal_builder_replayed_from_declared_inputs", replayed),
        ("scalar_epi_snapshot", True),
        ("supported_hard_or_inactive_clip_policy", supported_clip_policy),
        ("runtime_clipping_inactive", not clip_nodes),
        ("edge_adaptation_inactive", not adaptation_nodes),
        ("operator_gate_bound", gate_bound),
        ("runtime_matches_represented_affine_exactly", exact_runtime_match),
    )
    runtime_certified = all(passed for _, passed in runtime_conditions)

    contract = contract_for(glyph.value)
    affine_jump: AffineEPIJumpGainCertificate | None = None
    if runtime_certified and len(nodes) >= 2 and pre_flow_certified:
        affine_jump = certify_affine_epi_jump_gain(
            contract.english_name,
            represented_map,
            pre_flow.metric_weights,
            offset=represented_offset,
            nodes=nodes,
            tolerance=tol,
        )
    operational_gain = bool(
        affine_jump is not None
        and affine_jump.supports_global_gain_theorem
        and math.isfinite(affine_jump.energy_gain_bound_for_composition)
    )
    exact_gain = (
        None
        if affine_jump is None
        else affine_jump.exact_quotient_energy_gain_upper_bound
    )
    exact_gain_available = bool(
        isinstance(exact_gain, Fraction) and exact_gain >= 0
    )
    pre_gain_conditions = (
        ("at_least_two_nodes", len(nodes) >= 2),
        ("pre_flow_node_order_matches", pre_order_matches),
        ("pre_diffusion_flow_certified", pre_flow_certified),
        ("exact_consensus_subspace_preservation", exact_consensus),
        ("affine_gain_candidate_constructed", affine_jump is not None),
        ("exact_affine_gain_bound_available", exact_gain_available),
    )
    pre_metric_gain_certified = bool(
        runtime_certified and all(passed for _, passed in pre_gain_conditions)
    )
    bridge_conditions = (
        ("fixed_support_declared", fixed_support_declared),
        ("logical_copy_verified", logical_copy_verified),
        ("post_flow_node_order_matches", post_order_matches),
        ("post_diffusion_flow_certified", post_flow_certified),
        ("pre_post_metric_exactly_proportional", metrics_proportional),
    )
    common_metric_bridge_certified = bool(
        pre_metric_gain_certified
        and all(passed for _, passed in bridge_conditions)
    )
    operational_conditions = (
        ("finite_binary64_affine_gain_display", operational_gain),
    )
    nul_pressure_tuple = (
        None
        if exact_nul_pressure_residual is None
        else tuple(exact_nul_pressure_residual)
    )
    nul_drive_tuple = (
        None
        if exact_nul_drive_residual is None
        else tuple(exact_nul_drive_residual)
    )
    proof_stamp = _certificate_stamp(
        operator_name=contract.english_name,
        glyph=glyph.value,
        nodes=nodes,
        target_nodes=target_nodes,
        stage_schedule=stage_schedule,
        fixed_support_declared=fixed_support_declared,
        state_before=state_before,
        state_after=state_after,
        linear_map=represented_map,
        offset=represented_offset,
        exact_linear_map=exact_map,
        exact_offset=exact_offset,
        exact_residual=exact_residual,
        runtime_matches_exactly=exact_runtime_match,
        runtime_matches_within_tolerance=within_tolerance,
        clip_mode=clip_mode,
        clip_intervention_nodes=tuple(clip_nodes),
        edge_adaptation_nodes=tuple(adaptation_nodes),
        zero_post_capacity_nodes=zero_post_capacity,
        proposal_builder_replayed=replayed,
        proposal_replay_abstention_reason=replay_reason,
        logical_copy_verified=logical_copy_verified,
        pre_flow=pre_flow,
        post_flow=post_flow,
        pre_flow_node_order_matches=pre_order_matches,
        post_flow_node_order_matches=post_order_matches,
        metric_abstention_reason=metric_reason,
        flows_certified=flows_certified,
        metrics_proportional=metrics_proportional,
        exact_post_to_pre_metric_scale=exact_metric_scale,
        runtime_conditions=runtime_conditions,
        pre_gain_conditions=pre_gain_conditions,
        bridge_conditions=bridge_conditions,
        operational_conditions=operational_conditions,
        runtime_realization_certified=runtime_certified,
        pre_metric_gain_certified=pre_metric_gain_certified,
        common_metric_bridge_certified=common_metric_bridge_certified,
        operational_affine_gain_available=operational_gain,
        exact_energy_gain_bound=exact_gain,
        exact_nul_pressure_residual=nul_pressure_tuple,
        exact_nul_drive_residual=nul_drive_tuple,
        exact_nul_pressure_manifold_defect=exact_pressure_manifold_defect,
        nul_pressure_defect_norm=pressure_defect_norm,
        nul_pressure_abstention_reason=pressure_abstention_reason,
        affine_jump=affine_jump,
        tolerance=tol,
        scope=_SCOPE,
    )
    return PointwiseEPIJumpRealizationCertificate(
        operator_name=contract.english_name,
        glyph=glyph.value,
        nodes=nodes,
        target_nodes=target_nodes,
        stage_schedule=stage_schedule,
        fixed_support_declared=fixed_support_declared,
        state_before=state_before,
        runtime_proposed_state_after=state_after,
        represented_linear_map=represented_map,
        represented_offset=represented_offset,
        exact_represented_linear_map=exact_map,
        exact_represented_offset=exact_offset,
        exact_runtime_minus_represented_affine=exact_residual,
        runtime_matches_represented_affine_exactly=exact_runtime_match,
        runtime_matches_represented_affine_within_tolerance=within_tolerance,
        clip_mode=clip_mode,
        clip_intervention_nodes=tuple(clip_nodes),
        edge_adaptation_nodes=tuple(adaptation_nodes),
        zero_post_capacity_nodes=zero_post_capacity,
        proposal_builder_replayed_from_declared_inputs=replayed,
        proposal_replay_abstention_reason=replay_reason,
        logical_copy_verified=logical_copy_verified,
        pre_diffusion_certificate=pre_flow,
        post_diffusion_certificate=post_flow,
        pre_flow_node_order_matches=pre_order_matches,
        post_flow_node_order_matches=post_order_matches,
        diffusion_metric_abstention_reason=metric_reason,
        pre_and_post_diffusion_flows_certified=flows_certified,
        pre_post_metric_exactly_proportional=metrics_proportional,
        exact_post_to_pre_metric_scale=exact_metric_scale,
        exact_nul_stored_pressure_map_residual=nul_pressure_tuple,
        exact_nul_nodal_drive_residual=nul_drive_tuple,
        exact_nul_stored_pressure_minus_pure_epi_pressure=(
            exact_pressure_manifold_defect
        ),
        nul_stored_pressure_defect_norm=pressure_defect_norm,
        nul_pressure_diagnostic_abstention_reason=pressure_abstention_reason,
        runtime_epi_realization_conditions=runtime_conditions,
        pre_metric_affine_gain_conditions=pre_gain_conditions,
        pre_post_common_metric_bridge_conditions=bridge_conditions,
        operational_affine_gain_conditions=operational_conditions,
        affine_jump_certificate=affine_jump,
        exact_common_metric_energy_gain_bound=exact_gain,
        operational_affine_gain_available=operational_gain,
        runtime_epi_realization_certified=runtime_certified,
        pre_metric_affine_gain_certified=pre_metric_gain_certified,
        pre_post_common_metric_bridge_certified=(
            common_metric_bridge_certified
        ),
        tolerance=tol,
        scope=_SCOPE,
        _proof_stamp=proof_stamp,
    )
