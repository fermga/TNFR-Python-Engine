"""Detached frame-aware composition of two complete sine preparations.

Exact differences of common additive origins preserve the original residual
family. The supplied bridge changes support, not node state. No event, flow,
frame inference or authentication is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from typing import TYPE_CHECKING, Any

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos
from ._sine_admission import _admit_sine_source, _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleGeometry
from .phase_cycle_geometry import _derive as _derive_phase_geometry
from .relational_observations import _ordered
from .relational_sine_comparison import _validate_comparison_labels
from .relational_sine_pattern import (
    SineRelativePattern,
    _bound_sine_pattern_from_rows,
    _difference_bounds,
)
from .relational_sine_recovery import (
    SineSectorCapture,
    _certify_sine_sector_set,
    _validate_sine_sector_capture_labels,
    certify_sine_sector_capture,
)

if TYPE_CHECKING:
    from .relational_sine_entry import SinePreparedEntry

__all__ = (
    "SinePatternComposition",
    "SinePreparedComposition",
    "SinePreparedCompositionSource",
    "assess_sine_pattern_composition",
    "assess_sine_prepared_composition",
)


@dataclass(frozen=True)
class SinePatternComposition:
    """Joint preparation availability, bridge budget and separate capture.

    Origins are right minus left common additive offsets, not observed anchor
    differences. Representative means omit the unknown left common origin;
    the phase mean additionally depends on the declared real lift. Available
    preparation does not assert capture or permission for an event.
    """

    left: SineRelativePattern
    right: SineRelativePattern
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    bridge: tuple[Any, Any]
    observation_time: Q
    edge_turn_offsets: tuple[int, ...]
    form_origin_difference: Q | None
    phase_origin_difference: Q | None
    work_allowance: Q | None
    separate_degrees: tuple[int, ...]
    joined_degrees: tuple[int, ...]
    separate_form_weights: tuple[Q, ...]
    joined_form_weights: tuple[Q, ...]
    separate_mobility: tuple[Q, ...]
    joined_mobility: tuple[Q, ...]
    component_storage_bounds: tuple[I, I]
    separate_storage_bounds: I
    bridge_form_gap_bounds: I | None
    bridge_phase_gap_bounds: I | None
    bridge_form_storage_bounds: I | None
    bridge_phase_storage_bounds: I | None
    bridge_storage_bounds: I | None
    joined_storage_bounds: I | None
    representative_weighted_form_mean_bounds: I | None
    representative_weighted_phase_mean_bounds: I | None
    joined: SineRelativePattern | None
    capture: SineSectorCapture | None
    status: str
    unavailable_reasons: tuple[str, ...]
    budget_status: str
    work_margin_bounds: I | None
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_explicit_positive_loss_sine_law_and_positive_held_capacities",
        "two_disjoint_ordered_complete_relative_sources_and_one_supplied_unit_bridge",
        "declared_common_observation_time_and_structural_clock_not_authenticated",
        "exact_right_minus_left_common_additive_origins_not_measured_anchor_gaps",
        "original_per_node_residuals_retained_with_one_unknown_global_origin",
        "source_and_joint_fields_rebuilt_from_normalized_primitives_not_cached_verdicts",
        "joint_degrees_capacity_weights_and_mobility_recomputed_after_support_change",
        "bridge_phase_storage_bounds_are_unscaled_one_minus_cosine",
        "bridge_work_separate_from_continuous_loss_and_supplied_work_allowance",
        "representative_means_relative_to_left_unknown_origin_not_absolute_observations",
        "whole_joint_state_set_handed_to_existing_capture_not_isolated_certificates",
        "unavailable_capture_is_not_instability_or_failure_of_composition",
        "no_live_edges_nodal_reset_trajectory_contact_occurrence_or_native_law_transfer",
    )

    def to_dict(self):
        """Export exact evidence and availability after checking nested labels."""
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.left)
        _validate_comparison_labels(self.right)
        _validate_label_groups(
            self.nodes,
            self.bridge,
            *self.edges,
            (self.left.reference_node, self.right.reference_node),
        )
        if self.joined is not None:
            _validate_comparison_labels(self.joined)
            _validate_label_groups((self.joined.reference_node,))
        if self.capture is not None:
            _validate_sine_sector_capture_labels(self.capture)
        return {
            "schema": "tnfr.relational-sine-pattern-composition.v1",
            "report": _project(self),
        }


def _rebuild_pattern(source):
    return _bound_sine_pattern_from_rows(
        reference_node=source.reference_node,
        reference_model=source.reference_model,
        nodes=source.nodes,
        edges=source.edges,
        neighbors=source.neighbors,
        capacity=source.capacity,
        nominal_form=source.nominal_form,
        nominal_phase=source.nominal_phase,
        form_error_bounds=source.form_error_bounds,
        phase_error_bounds=source.phase_error_bounds,
    )


def _optional_scalar(value, label, *, nonnegative=False):
    if value is None:
        return None
    value = exact_or_represented_real(value, label)
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _mean_bounds(values, errors, weights):
    if values is None:
        return None
    total = sum(weights, Q(0))
    center = sum((weight * value for weight, value in zip(weights, values)), Q(0))
    radius = sum((weight * error for weight, error in zip(weights, errors)), Q(0))
    return I((center - radius) / total, (center + radius) / total)


@dataclass(frozen=True)
class _CompositionSupport:
    geometry: PhaseCycleGeometry
    edges: tuple[tuple[Any, Any], ...]
    bridge: tuple[Any, Any]
    bridge_indices: tuple[int, int]
    neighbors: tuple[tuple[int, ...], ...]
    capacity: tuple[Q, ...]
    separate_degrees: tuple[int, ...]
    joined_degrees: tuple[int, ...]
    separate_weights: tuple[Q, ...]
    joined_weights: tuple[Q, ...]
    coefficients: tuple[float, float, float]


def _composition_support(left, right, left_edges, right_edges, bridge):
    """Admit one bridge and rebuild its geometry before interval calculations."""
    coefficients = _sine_model_coefficients(left.reference_model, positive_loss=True)
    if (
        _sine_model_coefficients(right.reference_model, positive_loss=True)
        != coefficients
    ):
        raise ValueError("component sources must have the same admitted coefficients")
    capacity = left.capacity + right.capacity
    if any(value <= 0 for value in capacity):
        raise ValueError("composition requires strictly positive held capacities")
    if set(left.nodes) & set(right.nodes):
        raise ValueError("component node labels must be disjoint")
    endpoints = _ordered(bridge, "bridge", limit=3)
    if len(endpoints) != 2:
        raise ValueError("bridge must contain one left and one right endpoint")
    try:
        valid_bridge = endpoints[0] in left.nodes and endpoints[1] in right.nodes
    except (TypeError, ValueError) as exc:
        raise ValueError("bridge must name one endpoint in each component") from exc
    if not valid_bridge:
        raise ValueError("bridge must name one left and one right endpoint")
    nodes = left.nodes + right.nodes
    positions = {node: index for index, node in enumerate(nodes)}
    split = len(left.nodes)
    bridge_indices = positions[endpoints[0]], positions[endpoints[1]]
    indices = tuple(
        sorted(
            left_edges
            + tuple((i + split, j + split) for i, j in right_edges)
            + (bridge_indices,)
        )
    )
    geometry = _derive_phase_geometry(nodes, indices)
    edges = tuple((nodes[i], nodes[j]) for i, j in geometry.edges)
    neighbors = [[] for _ in nodes]
    for i, j in geometry.edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    neighbors = tuple(tuple(row) for row in neighbors)
    degrees = tuple(map(len, neighbors))
    old_degrees = left.degrees + right.degrees
    return _CompositionSupport(
        geometry,
        edges,
        endpoints,
        bridge_indices,
        neighbors,
        capacity,
        old_degrees,
        degrees,
        tuple(Q(degree) / nu for degree, nu in zip(old_degrees, capacity)),
        tuple(Q(degree) / nu for degree, nu in zip(degrees, capacity)),
        coefficients,
    )


def _composition_edge_offsets(values, count):
    offsets = _ordered(values, "edge_turn_offsets", limit=count + 1)
    if len(offsets) != count or any(type(value) is not int for value in offsets):
        raise ValueError(
            "edge_turn_offsets requires one nonboolean integer per canonical edge"
        )
    return offsets


def _composition_work(bridge_cost, allowance):
    margin = (
        None if allowance is None or bridge_cost is None else allowance - bridge_cost
    )
    if allowance is None:
        status = "not_supplied"
    elif bridge_cost is None:
        status = "unresolved"
    elif bridge_cost.hi <= allowance:
        status = "within_allowance"
    elif bridge_cost.lo > allowance:
        status = "exceeds_allowance"
    else:
        status = "unresolved"
    return status, margin


def assess_sine_pattern_composition(
    left,
    right,
    *,
    bridge,
    observation_time,
    edge_turn_offsets,
    form_origin_difference=None,
    phase_origin_difference=None,
    work_allowance=None,
) -> SinePatternComposition:
    """Admit a joint residual family and assess its supplied bridge and capture.

    If component coordinates are nominal + C + residual, origin differences
    mean C_right-C_left. Both are exact admitted scalars on the supplied form
    and real phase charts. Unknown differences remain unavailable; intervals
    and inferred anchor measurements are not accepted as exact offsets.

    Observation time declares synchronous preparation in the same structural
    clock. The sources carry no independent timestamps. A work allowance is
    an independent nonnegative event premise; capture is evaluated on the
    whole joined family whether or not this allowance bounds the bridge cost.
    """
    if not isinstance(left, SineRelativePattern) or not isinstance(
        right, SineRelativePattern
    ):
        raise TypeError("two SineRelativePattern sources are required")
    left, left_edges = _admit_sine_source(left)
    right, right_edges = _admit_sine_source(right)
    support = _composition_support(left, right, left_edges, right_edges, bridge)
    nodes, edges = support.geometry.nodes, support.edges
    coefficients = support.coefficients
    bridge_indices = support.bridge_indices
    offsets = _composition_edge_offsets(edge_turn_offsets, len(edges))
    at = exact_or_represented_real(observation_time, "observation_time")
    if at < 0:
        raise ValueError("observation_time must be nonnegative")
    form_offset = _optional_scalar(form_origin_difference, "form_origin_difference")
    phase_offset = _optional_scalar(phase_origin_difference, "phase_origin_difference")
    allowance = _optional_scalar(work_allowance, "work_allowance", nonnegative=True)

    left, right = _rebuild_pattern(left), _rebuild_pattern(right)
    neighbors, capacity = support.neighbors, support.capacity
    degrees, old_degrees = support.joined_degrees, support.separate_degrees
    weights, old_weights = support.joined_weights, support.separate_weights
    form_errors = left.form_error_bounds + right.form_error_bounds
    phase_errors = left.phase_error_bounds + right.phase_error_bounds
    forms = (
        None
        if form_offset is None
        else left.nominal_form
        + tuple(value + form_offset for value in right.nominal_form)
    )
    phases = (
        None
        if phase_offset is None
        else left.nominal_phase
        + tuple(value + phase_offset for value in right.nominal_phase)
    )
    form_gap = (
        None
        if forms is None
        else _difference_bounds(forms, form_errors, *bridge_indices)
    )
    phase_gap = (
        None
        if phases is None
        else _difference_bounds(phases, phase_errors, *bridge_indices)
    )
    form_cost = None if form_gap is None else form_gap**2 / 2
    phase_cost = None if phase_gap is None else 1 - cos(phase_gap)
    reasons = tuple(
        label
        for value, label in (
            (form_offset, "form_origin_difference_not_supplied"),
            (phase_offset, "phase_origin_difference_not_supplied"),
        )
        if value is None
    )
    joined = capture = bridge_cost = None
    if not reasons:
        joined = _bound_sine_pattern_from_rows(
            reference_node=left.reference_node,
            reference_model=left.reference_model,
            nodes=nodes,
            edges=edges,
            neighbors=neighbors,
            capacity=capacity,
            nominal_form=forms,
            nominal_phase=phases,
            form_error_bounds=form_errors,
            phase_error_bounds=phase_errors,
        )
        bridge_cost = form_cost + Q(coefficients[2]) * phase_cost
        capture = replace(
            certify_sine_sector_capture(joined, edge_turn_offsets=offsets),
            observation_time=at,
        )
    budget_status, margin = _composition_work(bridge_cost, allowance)
    return SinePatternComposition(
        left=left,
        right=right,
        nodes=nodes,
        edges=edges,
        bridge=support.bridge,
        observation_time=at,
        edge_turn_offsets=offsets,
        form_origin_difference=form_offset,
        phase_origin_difference=phase_offset,
        work_allowance=allowance,
        separate_degrees=old_degrees,
        joined_degrees=degrees,
        separate_form_weights=old_weights,
        joined_form_weights=weights,
        separate_mobility=tuple(1 / value for value in old_weights),
        joined_mobility=tuple(1 / value for value in weights),
        component_storage_bounds=(left.storage_bounds, right.storage_bounds),
        separate_storage_bounds=left.storage_bounds + right.storage_bounds,
        bridge_form_gap_bounds=form_gap,
        bridge_phase_gap_bounds=phase_gap,
        bridge_form_storage_bounds=form_cost,
        bridge_phase_storage_bounds=phase_cost,
        bridge_storage_bounds=bridge_cost,
        joined_storage_bounds=None if joined is None else joined.storage_bounds,
        representative_weighted_form_mean_bounds=_mean_bounds(
            forms, form_errors, weights
        ),
        representative_weighted_phase_mean_bounds=_mean_bounds(
            phases, phase_errors, weights
        ),
        joined=joined,
        capture=capture,
        status="unavailable" if reasons else "available",
        unavailable_reasons=reasons,
        budget_status=budget_status,
        work_margin_bounds=margin,
    )


@dataclass(frozen=True)
class SinePreparedCompositionSource:
    """Rebuilt analytic endpoint provenance, without an observed joint state."""

    left: SinePreparedEntry
    right: SinePreparedEntry
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    bridge: tuple[Any, Any]
    left_initial_time: Q
    right_initial_time: Q
    observation_time: Q
    edge_turn_offsets: tuple[int, ...]
    form_origin_difference: Q | None
    phase_origin_difference: Q | None
    scope: tuple[str, ...] = (
        "actual_analytic_endpoints_of_two_original_relative_preparation_families",
        "components_isolated_until_equal_derived_event_time_in_one_declared_clock",
        "original_common_origin_difference_retained_with_memberwise_conserved_means",
        "internal_correlated_edge_bounds_not_properties_of_cartesian_node_boxes",
        "no_endpoint_observation_fabrication_waiting_reset_or_trajectory_replay",
    )


@dataclass(frozen=True)
class SinePreparedComposition:
    """Analytic acquisition-to-composition evidence with a separate work budget."""

    source: SinePreparedCompositionSource
    separate_degrees: tuple[int, ...]
    joined_degrees: tuple[int, ...]
    separate_form_weights: tuple[Q, ...]
    joined_form_weights: tuple[Q, ...]
    separate_mobility: tuple[Q, ...]
    joined_mobility: tuple[Q, ...]
    component_storage_bounds: tuple[I, I]
    separate_storage_bounds: I
    bridge_form_gap_bounds: I | None
    bridge_phase_gap_bounds: I | None
    bridge_form_storage_bounds: I | None
    bridge_phase_storage_bounds: I | None
    bridge_storage_bounds: I | None
    joined_storage_bounds: I | None
    endpoint_form_edge_gap_bounds: tuple[I, ...] | None
    endpoint_phase_edge_gap_bounds: tuple[I, ...] | None
    representative_weighted_form_mean_bounds: I | None
    representative_weighted_phase_mean_bounds: I | None
    capture: SineSectorCapture | None
    status: str
    unavailable_reasons: tuple[str, ...]
    work_allowance: Q | None
    budget_status: str
    work_margin_bounds: I | None
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_positive_loss_sine_law_and_positive_held_capacities_for_both_prefixes",
        "entry_endpoint_and_status_fields_rebuilt_from_source_time_and_sector_declarations",
        "full_joined_geometry_budget_before_analytic_entry_recomputation",
        "equal_exact_initial_time_plus_original_horizon_without_silent_waiting",
        "endpoint_centers_are_not_nominal_observations_or_stationary_components",
        "original_weighted_residual_mean_uncertainty_retained_per_family_member",
        "original_correlated_internal_edge_bounds_and_conservative_bridge_port_bounds",
        "postjoin_mean_uses_conserved_component_charges_plus_degree_reweighting",
        "representative_means_omit_left_common_origin_and_phase_uses_declared_lifts",
        "bridge_phase_storage_bounds_are_unscaled_one_minus_cosine",
        "whole_actual_joint_endpoint_set_passed_to_shared_sector_capture",
        "work_allowance_independent_of_acquisition_capture_and_event_occurrence",
        "no_live_edge_event_reset_flow_physical_identity_or_native_law_transfer",
    )

    @property
    def left(self):
        """Return the rebuilt left analytic entry."""
        return self.source.left

    @property
    def right(self):
        """Return the rebuilt right analytic entry."""
        return self.source.right

    @property
    def nodes(self):
        """Return the complete ordered joint support."""
        return self.source.nodes

    @property
    def edges(self):
        """Return joint edges in canonical index order."""
        return self.source.edges

    @property
    def bridge(self):
        """Return the supplied left-to-right bridge endpoints."""
        return self.source.bridge

    @property
    def observation_time(self):
        """Return the common derived endpoint time, not an observed timestamp."""
        return self.source.observation_time

    @property
    def acquisition_and_capture_certified(self):
        """Require both rebuilt acquisitions and whole-state joint capture."""
        return (
            self.left.admitted
            and self.right.admitted
            and self.capture is not None
            and self.capture.admitted
        )

    def to_dict(self):
        """Validate nested labels and project exact retained analytic evidence."""
        from ..sdk.relational_reports import _project

        _validate_sine_prepared_composition_source_labels(self.source)
        if self.capture is not None:
            _validate_sine_sector_capture_labels(self.capture)
        return {
            "schema": "tnfr.relational-sine-prepared-composition.v1",
            "report": _project(self),
        }


def _validate_sine_prepared_composition_source_labels(source):
    from ..sdk.relational_reports import _validate_label_groups
    from .relational_sine_entry import _validate_sine_prepared_entry_labels

    _validate_label_groups(source.nodes, source.bridge, *source.edges)
    _validate_sine_prepared_entry_labels(source.left)
    _validate_sine_prepared_entry_labels(source.right)


def _entry_declarations(entry):
    from .relational_sine_entry import SinePreparedEntry

    if not isinstance(entry, SinePreparedEntry) or not isinstance(
        entry.source, SineRelativePattern
    ):
        raise TypeError(
            "prepared composition requires entries from SineRelativePattern sources"
        )
    source, edges = _admit_sine_source(entry.source)
    tau = exact_or_represented_real(entry.scaled_time, "scaled_time")
    if tau < 0:
        raise ValueError("scaled_time must be nonnegative")
    if not isinstance(entry.capture, SineSectorCapture):
        raise TypeError("entry capture must retain its SineSectorCapture declaration")
    offsets = _composition_edge_offsets(entry.capture.edge_turn_offsets, len(edges))
    return source, edges, tau, offsets


def _endpoint_bridge_and_mean(left, right, support, origin, *, phase=False):
    if origin is None:
        return None, None
    split = len(left.source.nodes)
    a, b = support.bridge_indices
    b -= split
    left_weights = support.separate_weights[:split]
    right_weights = support.separate_weights[split:]
    value_name = "nominal_phase" if phase else "nominal_form"
    error_name = "phase_error_bounds" if phase else "form_error_bounds"
    means = tuple(
        _mean_bounds(
            getattr(entry.source, value_name),
            getattr(entry.source, error_name),
            weights,
        )
        for entry, weights in ((left, left_weights), (right, right_weights))
    )
    ma, mb = means[0], means[1] + origin
    boxes_name = (
        "centered_endpoint_phase_bounds" if phase else "centered_endpoint_form_bounds"
    )
    ua, ub = getattr(left, boxes_name)[a], getattr(right, boxes_name)[b]
    inverse_a = Q(1) / left.source.capacity[a]
    inverse_b = Q(1) / right.source.capacity[b]
    mean = (
        (sum(left_weights, Q(0)) + inverse_a) * ma
        + (sum(right_weights, Q(0)) + inverse_b) * mb
        + inverse_a * ua
        + inverse_b * ub
    ) / sum(support.joined_weights, Q(0))
    return mb - ma + ub - ua, mean


def _joined_endpoint_edges(left, right, support, bridge_bound, *, phase=False):
    name = (
        "endpoint_phase_edge_gap_bounds" if phase else "endpoint_form_edge_gap_bounds"
    )
    split = len(left.source.nodes)
    bounds = dict(zip(left.geometry.edges, getattr(left, name)))
    bounds.update(
        ((i + split, j + split), bound)
        for (i, j), bound in zip(right.geometry.edges, getattr(right, name))
    )
    bounds[support.bridge_indices] = bridge_bound
    return tuple(bounds[edge] for edge in support.geometry.edges)


def assess_sine_prepared_composition(
    left,
    right,
    *,
    bridge,
    left_initial_time,
    right_initial_time,
    edge_turn_offsets,
    form_origin_difference=None,
    phase_origin_difference=None,
    work_allowance=None,
) -> SinePreparedComposition:
    """Compose two analytic endpoints at their exact common declared time.

    Both entries must originate from relative patterns, including zero-radius
    patterns. Only original source primitives, scaled times and internal
    sector offsets are consumed from the entries. Endpoint boxes, horizons,
    storage and verdicts are recomputed by their owner. Components stay
    isolated for their own declared horizons; unequal endpoint times reject.

    Exact origins still mean right minus left additive common offsets of the
    initial source families. Conserved memberwise means propagate this frame
    information to the endpoints. Internal correlated edge bounds and new
    conservative bridge bounds describe the actual endpoint family, never
    every point of a fictitious independent node observation box.
    """
    from .relational_sine_entry import certify_sine_prepared_entry

    ls, le, ltau, loffsets = _entry_declarations(left)
    rs, re, rtau, roffsets = _entry_declarations(right)
    support = _composition_support(ls, rs, le, re, bridge)
    offsets = _composition_edge_offsets(edge_turn_offsets, len(support.edges))
    initial_times = tuple(
        exact_or_represented_real(value, label)
        for value, label in (
            (left_initial_time, "left_initial_time"),
            (right_initial_time, "right_initial_time"),
        )
    )
    if any(value < 0 for value in initial_times):
        raise ValueError("initial times must be nonnegative")
    e = Q(support.coefficients[0])
    event_times = tuple(
        start + tau / e for start, tau in zip(initial_times, (ltau, rtau))
    )
    if event_times[0] != event_times[1]:
        raise ValueError("derived endpoint times must be equal; no waiting is inferred")
    form_offset = _optional_scalar(form_origin_difference, "form_origin_difference")
    phase_offset = _optional_scalar(phase_origin_difference, "phase_origin_difference")
    allowance = _optional_scalar(work_allowance, "work_allowance", nonnegative=True)
    left = certify_sine_prepared_entry(
        _rebuild_pattern(ls), scaled_time=ltau, edge_turn_offsets=loffsets
    )
    right = certify_sine_prepared_entry(
        _rebuild_pattern(rs), scaled_time=rtau, edge_turn_offsets=roffsets
    )
    source = SinePreparedCompositionSource(
        left=left,
        right=right,
        nodes=support.geometry.nodes,
        edges=support.edges,
        bridge=support.bridge,
        left_initial_time=initial_times[0],
        right_initial_time=initial_times[1],
        observation_time=event_times[0],
        edge_turn_offsets=offsets,
        form_origin_difference=form_offset,
        phase_origin_difference=phase_offset,
    )
    form_gap, form_mean = _endpoint_bridge_and_mean(left, right, support, form_offset)
    phase_gap, phase_mean = _endpoint_bridge_and_mean(
        left, right, support, phase_offset, phase=True
    )
    form_cost = None if form_gap is None else form_gap**2 / 2
    phase_cost = None if phase_gap is None else 1 - cos(phase_gap)
    reasons = tuple(
        label
        for value, label in (
            (form_offset, "form_origin_difference_not_supplied"),
            (phase_offset, "phase_origin_difference_not_supplied"),
        )
        if value is None
    )
    capture = bridge_cost = form_edges = phase_edges = None
    if not reasons:
        bridge_cost = form_cost + Q(support.coefficients[2]) * phase_cost
        form_edges = _joined_endpoint_edges(left, right, support, form_gap)
        phase_edges = _joined_endpoint_edges(
            left, right, support, phase_gap, phase=True
        )
        capture = _certify_sine_sector_set(
            source=source,
            geometry=support.geometry,
            model=left.reference_model,
            capacity_bounds=tuple(I(value) for value in support.capacity),
            exact_held_capacity=support.capacity,
            form_edge_gap_bounds=form_edges,
            phase_edge_gap_bounds=phase_edges,
            edge_turn_offsets=offsets,
            uncertainty_scope="analytic_composed_endpoint_family_with_internal_norm_correlations_and_memberwise_means",
            observation_time=source.observation_time,
            weighted_mean_scope="left_common_origins_unobserved_representative_mean_bounds_retained_by_composition",
        )
    budget_status, margin = _composition_work(bridge_cost, allowance)
    component_storage = left.capture.storage_bounds, right.capture.storage_bounds
    return SinePreparedComposition(
        source=source,
        separate_degrees=support.separate_degrees,
        joined_degrees=support.joined_degrees,
        separate_form_weights=support.separate_weights,
        joined_form_weights=support.joined_weights,
        separate_mobility=tuple(1 / value for value in support.separate_weights),
        joined_mobility=tuple(1 / value for value in support.joined_weights),
        component_storage_bounds=component_storage,
        separate_storage_bounds=sum(component_storage, I(0)),
        bridge_form_gap_bounds=form_gap,
        bridge_phase_gap_bounds=phase_gap,
        bridge_form_storage_bounds=form_cost,
        bridge_phase_storage_bounds=phase_cost,
        bridge_storage_bounds=bridge_cost,
        joined_storage_bounds=None if capture is None else capture.storage_bounds,
        endpoint_form_edge_gap_bounds=form_edges,
        endpoint_phase_edge_gap_bounds=phase_edges,
        representative_weighted_form_mean_bounds=form_mean,
        representative_weighted_phase_mean_bounds=phase_mean,
        capture=capture,
        status="unavailable" if reasons else "available",
        unavailable_reasons=reasons,
        work_allowance=allowance,
        budget_status=budget_status,
        work_margin_bounds=margin,
    )
