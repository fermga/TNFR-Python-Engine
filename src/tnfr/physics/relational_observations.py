"""Read-only regional and support-change observations of relational exchange.

An explicit reference supplies real phase lifts, not a discovered equilibrium.
The full centered form and phase-error vectors retain information discarded by
regional means. Shared winding and transport observations add no evolution,
identity certificate, reduced closure or automatic selection policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from typing import Any

import networkx as nx

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..dynamics.relational import (
    RelationalExchangeField,
    RelationalExchangeModel,
    _phase_edge_storage,
    evaluate_relational_exchange,
)
from .support_transport import (
    RegionalSupportBalance,
    RegionalSupportCut,
    SupportTransportReset,
    SupportTransportSnapshot,
    _from_data,
    observe_regional_support_balance,
    observe_regional_support_cut,
    observe_support_transport,
    observe_support_transport_reset,
)
from .winding_certificates import WindingCertificate, certify_phase_winding

__all__ = (
    "RegionalRelationalWork",
    "RegionalExchangeBalance",
    "RegionalPhaseResponse",
    "RegionalPatternObservation",
    "RelationalPatternObservation",
    "observe_relational_pattern",
    "RelationalPortState",
    "RelationalAttachmentPort",
    "RelationalAttachmentObservation",
    "RelationalAttachmentSupplyAssessment",
    "observe_relational_attachment",
    "RelationalRelocationObservation",
    "observe_relational_relocation",
    "RelationalResetObservation",
    "observe_relational_reset",
    "RelationalCoefficientJetBounds",
    "bound_relational_coefficient_from_jet",
    "RelationalCoefficientSampleBounds",
    "bound_relational_coefficient_from_samples",
)


@dataclass(frozen=True)
class RelationalCoefficientJetBounds:
    """Conditional chi bounds from declared initial form-response intervals."""

    form_bounds: tuple[Fraction, Fraction]
    rate_bounds: tuple[Fraction, Fraction]
    acceleration_bounds: tuple[Fraction, Fraction]
    squared_rate_bounds: tuple[Fraction, Fraction]
    restoring_gap_bounds: tuple[Fraction, Fraction]
    coefficient_bounds: tuple[Fraction, Fraction] | None
    unavailable_reasons: tuple[str, ...]
    arithmetic_method: str
    scope: tuple[str, ...] = (
        "conditional_two_channel_relational_consensus_preparation",
        "one_nonzero_spatial_mode_with_initial_phase_consensus",
        "declared_value_rate_acceleration_intervals_in_one_gain_and_affine_clock",
        "rational_outward_bounds_with_mathematical_pi_enclosure",
        "no_graph_read_preparation_authentication_derivative_estimation_or_evolution",
        "no_automatic_precision_acceptance_physical_bridge_or_universal_coefficient",
    )


@dataclass(frozen=True)
class RelationalCoefficientSampleBounds:
    """Three-sample jet bounds conditional on a supplied smoothness/noise budget."""

    samples: tuple[Fraction, Fraction, Fraction]
    sample_step: Fraction
    sample_error_bound: Fraction
    third_derivative_bound: Fraction
    rate_estimate: Fraction
    acceleration_estimate: Fraction
    rate_error_bound: Fraction
    acceleration_error_bound: Fraction
    jet: RelationalCoefficientJetBounds
    scope: tuple[str, ...] = (
        "samples_at_relative_times_zero_h_two_h_in_one_affine_clock",
        "supplied_uniform_sample_error_and_whole_window_C3_bound",
        "exact_rational_stencils_and_shared_outward_jet_enclosure",
        "no_source_preparation_clock_noise_or_smoothness_authentication",
        "no_graph_read_evolution_fit_or_physical_admission",
    )


@dataclass(frozen=True)
class RegionalRelationalWork:
    """Sums of the captured nodal gradient-work contributions in one region.

    Positive exchange transfers phase storage toward form storage. These are
    contributions to the global storage derivative, not derivatives of an
    independently defined regional energy. Overlapping regions double count
    shared nodes; only a partition reproduces the global sums.
    """

    dissipation: Fraction
    exchange: Fraction
    form_work: Fraction
    phase_work: Fraction
    form_residual: Fraction
    phase_residual: Fraction
    balance_residual: Fraction


@dataclass(frozen=True)
class RegionalExchangeBalance:
    """Two actual weighted rates using one outward form cut on full support.

    Form weights are full-graph degree/capacity; phase weights are the captured
    phase metric/capacity. The latter rate is not the derivative of a weighted
    phase total: its metric varies with phase. Exact arithmetic on represented
    rates retains phase and pressure/rounding defects rather than setting them
    to their ideal zero values. A zero capacity inside the region makes the
    divided rates unavailable, while the cut and undivided model terms remain
    defined. No evolution, reduced closure or provenance seal is supplied.
    """

    cut: RegionalSupportCut
    form_weighted_rate: Fraction | None
    form_boundary_rate: Fraction
    form_source_rate: Fraction
    form_pressure_defect_rate: Fraction
    form_rounding_defect_rate: Fraction | None
    form_identity_residual: Fraction | None
    phase_weighted_rate: Fraction | None
    phase_boundary_rate: Fraction
    phase_rate_residual: Fraction | None
    weighted_rate_unavailable_reason: str | None


@dataclass(frozen=True)
class RegionalPhaseResponse:
    """Unweighted phase-rate balance from the captured phase mobility and form.

    Mobility is capacity divided by the positive phase metric. Population
    covariance retains its correlation with the exact form gradient; it need
    not vanish when the outward cut vanishes. The squared covariance-rate
    bound is Cauchy--Schwarz, not a calibrated threshold. Actual model rates
    retain their rounding residual. This instantaneous identity supplies no
    autonomous regional closure, measured derivative or monotone clock.
    """

    mean_mobility: Fraction
    mean_form_gradient: Fraction
    mobility_variance: Fraction
    form_gradient_variance: Fraction
    mobility_gradient_covariance: Fraction
    mean_mobility_boundary_rate: Fraction
    covariance_rate: Fraction
    covariance_rate_squared_bound: Fraction
    model_total_rate: Fraction
    rounding_residual: Fraction
    total_rate: Fraction
    mean_rate: Fraction
    identity_residual: Fraction


@dataclass(frozen=True)
class RegionalPatternObservation:
    """One supplied region's complete centered coordinates and separate offsets.

    Means are arithmetic means, whereas ``transport.mean`` uses the shared
    degree/capacity metric. The two squared norms have their respective form
    and phase units; no combined metric or recovery threshold is introduced.
    """

    nodes: tuple[Any, ...]
    form_mean: Fraction
    phase_error_mean: Fraction
    centered_form: tuple[Fraction, ...]
    centered_phase_error: tuple[Fraction, ...]
    form_norm_squared: Fraction
    phase_norm_squared: Fraction
    transport: RegionalSupportBalance | None
    transport_unavailable_reason: str | None
    work: RegionalRelationalWork | None = None
    boundary: RegionalExchangeBalance | None = None
    phase_response: RegionalPhaseResponse | None = None


@dataclass(frozen=True)
class RelationalPatternObservation:
    """Detached fresh field and observations in caller-supplied regional frames.

    ``reference_phase`` follows ``field.nodes`` and contains materialized real
    lifts. Subtracting them does not wrap angles or infer a common semicircle.
    A common change of phase offset disappears on centering; independently
    changing node representatives by full turns changes the supplied frame.
    """

    field: RelationalExchangeField
    reference_phase: tuple[float, ...]
    regions: tuple[RegionalPatternObservation, ...]
    winding: tuple[WindingCertificate, ...]
    scope: tuple[str, ...] = (
        "one_fresh_detached_relational_field_without_live_graph_writes",
        "supplied_real_reference_lifts_and_ordered_possibly_overlapping_regions",
        "exact_rational_centering_of_materialized_form_and_phase_coordinates",
        "separate_form_and_phase_units_without_a_combined_norm_or_threshold",
        "regional_transport_uses_full_support_and_independent_phase_source",
        "nodal_and_regional_work_retains_exact_represented_gradients_and_rate_defects",
        "paired_weighted_rates_share_one_cut_not_a_conserved_phase_total",
        "unweighted_phase_response_retains_mobility_form_covariance_and_rounding",
        "declared_cycle_winding_is_snapshot_geometry_not_temporal_identity",
        "no_reference_equilibrium_recovery_formation_closed_reduction_or_selection_certificate",
    )


@dataclass(frozen=True)
class RelationalPortState:
    """One supplied port's captured state and instantaneous local interface.

    ``form_gradient`` is exact represented ``q=B*x``; ``relative_resultant``
    contains the engine's materialized relative cosine/sine sums. Degree,
    gradient and resultant refer to the field's complete support. These
    summaries do not supply their own future evolution or a closed macro-node.
    """

    node: Any
    epi: float
    phase: float
    capacity: float
    degree: int
    form_gradient: Fraction
    relative_resultant: tuple[float, float]
    pressure: float
    phase_metric: float
    form_rate: float
    phase_rate: float


@dataclass(frozen=True)
class RelationalAttachmentPort:
    """Captured port before and after a hypothetical unit-support change.

    The original attachment name is retained for compatibility; relocation
    observations reuse this same before/after card.
    """

    before: RelationalPortState
    after: RelationalPortState


@dataclass(frozen=True)
class RelationalAttachmentSupplyAssessment:
    """Declared work compared with one represented hypothetical storage jump.

    The additional event-passivity premise is ``storage_change <= supplied_work``.
    Required supply uses the support comparison's captured storage difference, not a
    new physical energy or a bound on ideal trigonometric storage. Signed work
    may describe supply or extraction; its source is supplied by the caller.
    A passing balance neither authenticates that work nor selects an event.
    """

    required_supply: Fraction
    supplied_work: Fraction
    supply_margin: Fraction
    represented_balance_satisfied: bool
    scope: tuple[str, ...] = (
        "caller_supplied_work_not_authenticated",
        "passivity_is_an_additional_event_premise",
        "represented_storage_not_an_ideal_trigonometric_certificate",
        "no_occurrence_timing_or_live_support_selection",
    )


class _RelationalSupportBudget:
    """Shared declared-work accounting for captured support comparisons."""

    storage_change: Fraction

    @property
    def represented_zero_supply_passive(self) -> bool:
        """Whether captured storage satisfies the additional zero-work premise."""
        return self.storage_change <= 0

    def assess_supply(self, supplied_work: Any) -> RelationalAttachmentSupplyAssessment:
        """Assess signed declared work without evaluating or modifying any graph.

        Positive work is supply and negative work is extraction, in the same
        structural-storage units as ``storage_change``. Rational inputs remain
        exact; other real inputs follow the shared represented-real boundary.
        No default work, occurrence rule, elapsed-time credit or continuous-loss
        funding is inferred. This is arithmetic on the report, not proof of its
        provenance or of exact-real passivity from rounded phase data.
        """
        work = exact_or_represented_real(supplied_work, "supplied_work")
        margin = work - self.storage_change
        return RelationalAttachmentSupplyAssessment(
            required_supply=self.storage_change,
            supplied_work=work,
            supply_margin=margin,
            represented_balance_satisfied=margin >= 0,
        )


@dataclass(frozen=True)
class RelationalAttachmentObservation(_RelationalSupportBudget):
    """Compare two admitted components with their hypothetical joined field.

    Full component states remain available in ``components``. Change tuples
    follow ``joined.nodes`` and subtract the matching component's captured
    value from the joined value using exact represented arithmetic. Storage
    changes subtract the sum of both components; ``phase_storage_change`` is
    the unscaled cosine cost V, and total storage includes the model's beta.
    ``cut`` is directed outward from the complete left component. The shared
    ``transport_reset`` accounts for form storage only. None of these values
    records an executed support event, a trajectory or a capture certificate.
    """

    components: tuple[RelationalExchangeField, RelationalExchangeField]
    joined: RelationalExchangeField
    bridge: tuple[Any, Any]
    ports: tuple[RelationalAttachmentPort, RelationalAttachmentPort]
    form_rate_change: tuple[Fraction, ...]
    phase_rate_change: tuple[Fraction, ...]
    pressure_change: tuple[Fraction, ...]
    phase_metric_change: tuple[Fraction, ...]
    form_storage_change: Fraction
    phase_storage_change: Fraction
    storage_change: Fraction
    transport_reset: SupportTransportReset
    cut: RegionalSupportCut
    scope: tuple[str, ...] = (
        "two_disjoint_separately_admitted_connected_components",
        "supplied_ordered_unit_bridge_left_port_to_right_port",
        "acute_model_only_with_unchanged_form_phase_and_held_capacity",
        "three_fresh_native_fields_without_live_graph_writes",
        "exact_represented_differences_not_transcendental_error_bounds",
        "hypothetical_support_comparison_not_an_executed_event",
        "no_autonomous_attachment_reduced_closure_or_recovery_certificate",
    )

    @property
    def continuous_loss_change(self) -> Fraction:
        """Joined minus component loss rates, not available event work."""
        return self.joined.continuous_loss - sum(
            (field.continuous_loss for field in self.components), Fraction(0)
        )


@dataclass(frozen=True)
class RelationalRelocationObservation(_RelationalSupportBudget):
    """Two fresh fields for one supplied state-preserving bridge relocation.

    ``components`` are ordered node partitions after removing the old bridge,
    not separately evaluated fields. Their internal edges remain unchanged.
    Both cuts point outward from the first partition. Change tuples follow
    ``after.nodes``; ports contain each affected node once, in that order.
    Phase storage changes are unscaled cosine costs; total storage includes
    beta. A passive represented budget alone does not certify future identity.
    """

    before: RelationalExchangeField
    after: RelationalExchangeField
    components: tuple[tuple[Any, ...], tuple[Any, ...]]
    remove_bridge: tuple[Any, Any]
    add_bridge: tuple[Any, Any]
    ports: tuple[RelationalAttachmentPort, ...]
    form_rate_change: tuple[Fraction, ...]
    phase_rate_change: tuple[Fraction, ...]
    pressure_change: tuple[Fraction, ...]
    phase_metric_change: tuple[Fraction, ...]
    form_storage_change: Fraction
    phase_storage_change: Fraction
    storage_change: Fraction
    transport_reset: SupportTransportReset
    cut_before: RegionalSupportCut
    cut_after: RegionalSupportCut
    scope: tuple[str, ...] = (
        "supplied_graph_bridge_separates_two_nontrivial_connected_components",
        "supplied_missing_unit_edge_crosses_the_same_ordered_components",
        "all_internal_edges_and_primitive_form_phase_capacity_are_preserved",
        "acute_model_only_with_two_fresh_fields_without_live_graph_writes",
        "exact_represented_differences_not_transcendental_error_bounds",
        "hypothetical_support_comparison_not_an_executed_event",
        "no_autonomous_selection_timing_reduced_closure_or_recovery_certificate",
    )

    @property
    def continuous_loss_change(self) -> Fraction:
        """New minus old loss rate, not available event work."""
        return self.after.continuous_loss - self.before.continuous_loss


@dataclass(frozen=True)
class RelationalResetObservation(_RelationalSupportBudget):
    """Joint state/support storage accounting between two supplied snapshots.

    Form storage uses conductance; phase storage uses every bare support edge,
    including zero-conductance edges. The decomposition first changes state on
    the old support, then changes support at the new state. Phase terms are
    unscaled V; total storage multiplies them by the supplied positive beta.
    ``transport_reset.before`` retains the new state on the old support.
    Capacity and stored pressure use the shared transport snapshot, including
    its zero defaults when absent; those defaults are not measured zeros.
    Snapshot comparison does not authenticate an executed operator or certify
    that either endpoint admits the conditional unit-support relational flow.
    """

    before: SupportTransportSnapshot
    after: SupportTransportSnapshot
    phase_before: tuple[float, ...]
    phase_after: tuple[float, ...]
    edges_before: tuple[tuple[Any, Any], ...]
    edges_after: tuple[tuple[Any, Any], ...]
    storage_scale: float
    form_state_change: Fraction
    form_support_change: Fraction
    form_storage_change: Fraction
    phase_state_change: Fraction
    phase_support_change: Fraction
    phase_storage_change: Fraction
    phase_storage_before: Fraction
    phase_storage_after: Fraction
    storage_before: Fraction
    storage_after: Fraction
    storage_change: Fraction
    identity_residual: Fraction
    transport_reset: SupportTransportReset
    scope: tuple[str, ...] = (
        "supplied_nonempty_same_ordered_nodes_simple_undirected_support",
        "symmetric_nonnegative_conductance_with_disconnected_and_zero_weight_support_allowed",
        "weighted_form_storage_and_unweighted_bare_support_phase_storage",
        "state_change_on_old_support_then_support_change_at_new_state",
        "raw_phase_admission_and_shared_represented_half_sine_cost",
        "exact_represented_accounting_not_an_ideal_trigonometric_bound",
        "detached_stored_state_without_pressure_refresh_or_live_writes",
        "transport_snapshot_defaults_absent_capacity_and_stored_pressure_to_zero",
        "no_event_authentication_selection_clock_or_continuation_certificate",
    )


def observe_relational_reset(before_graph, after_graph, *, storage_scale):
    """Observe the full storage budget of a supplied joint state/support reset.

    Both graphs must have the same nonempty node order and simple undirected
    loop-free support. Shared transport admission validates scalar form,
    nonnegative capacity, stored pressure and symmetric nonnegative weights;
    absent capacity/stored pressure retain that owner's zero defaults.
    Explicit finite phase values are required at every node, including isolates.
    Graphs may be disconnected and have zero or nonunit conductances; no acute
    chamber, pressure law, held capacity or continuation admission is asserted.

    Storage is ``E_D + storage_scale * V`` with ``V`` the sum of the shared
    represented half-sine cost on every support edge. The positive scale is
    materialized through the same boundary as the relational model. The result
    separates simultaneous nodal reorganization from support work without
    crediting earlier continuous dissipation or introducing an event law.
    """
    beta, beta_q = finite_represented_real(storage_scale, "storage_scale")
    if beta <= 0:
        raise ValueError("storage_scale must be positive")
    for graph in (before_graph, after_graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError("reset endpoints must be networkx graphs")
        if (
            graph.is_directed()
            or graph.is_multigraph()
            or nx.number_of_selfloops(graph)
        ):
            raise ValueError("reset support must be simple undirected without loops")
    nodes = tuple(before_graph)
    if not nodes or nodes != tuple(after_graph):
        raise ValueError("reset endpoints require the same nonempty ordered nodes")

    def phases(graph):
        return tuple(
            finite_represented_real(
                get_attr(
                    graph.nodes[node],
                    ALIAS_THETA,
                    None,
                    conv=lambda value: value,
                    strict=True,
                ),
                "phase",
            )[0]
            for node in nodes
        )

    phase_before, phase_after = phases(before_graph), phases(after_graph)
    before = observe_support_transport(before_graph)
    after = observe_support_transport(after_graph)
    intermediate = _from_data(
        nodes,
        before.conductance,
        before.support_neighbors,
        after.epi,
        after.capacity,
        after.stored_pressure,
    )
    transport_reset = observe_support_transport_reset(intermediate, after)
    old_edges = tuple(
        (i, j) for i, row in enumerate(before.support_neighbors) for j in row if i < j
    )
    new_edges = tuple(
        (i, j) for i, row in enumerate(after.support_neighbors) for j in row if i < j
    )

    def phase_cost(phase, edges):
        total = Fraction(0)
        for i, j in edges:
            difference = finite_represented_real(
                Fraction(phase[j]) - Fraction(phase[i]), "phase difference"
            )[0]
            total += _phase_edge_storage(math.remainder(difference, math.tau))
        return total

    phase_old = phase_cost(phase_before, old_edges)
    phase_intermediate = phase_cost(phase_after, old_edges)
    phase_new = phase_cost(phase_after, new_edges)
    form_state = intermediate.dirichlet_energy - before.dirichlet_energy
    form_support = transport_reset.energy_change
    phase_state = phase_intermediate - phase_old
    phase_support = phase_new - phase_intermediate
    storage_before = before.dirichlet_energy + beta_q * phase_old
    storage_after = after.dirichlet_energy + beta_q * phase_new
    storage_change = storage_after - storage_before
    return RelationalResetObservation(
        before=before,
        after=after,
        phase_before=phase_before,
        phase_after=phase_after,
        edges_before=tuple((nodes[i], nodes[j]) for i, j in old_edges),
        edges_after=tuple((nodes[i], nodes[j]) for i, j in new_edges),
        storage_scale=beta,
        form_state_change=form_state,
        form_support_change=form_support,
        form_storage_change=after.dirichlet_energy - before.dirichlet_energy,
        phase_state_change=phase_state,
        phase_support_change=phase_support,
        phase_storage_change=phase_new - phase_old,
        phase_storage_before=phase_old,
        phase_storage_after=phase_new,
        storage_before=storage_before,
        storage_after=storage_after,
        storage_change=storage_change,
        identity_residual=(
            storage_change
            - form_state
            - form_support
            - beta_q * (phase_state + phase_support)
        ),
        transport_reset=transport_reset,
    )


def _ordered(value, label, *, limit=None):
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered iterable")
    try:
        return tuple(value) if limit is None else tuple(islice(iter(value), limit))
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered iterable") from exc


def bound_relational_coefficient_from_jet(
    *, form_bounds, rate_bounds, acceleration_bounds
) -> RelationalCoefficientJetBounds:
    """Enclose chi=m1²/[pi²*(m1²-m0*m2)] under declared preparation premises.

    Each input is an ordered pair of finite exact/represented real endpoints.
    Bounds include outward arithmetic, not unprovided measurement or derivative
    error. Unresolved signal/gap or incompatible decay returns no coefficient.
    This function neither validates a physical preparation nor reads a graph.
    """
    from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval

    def interval(raw, label):
        values = _ordered(raw, label, limit=3)
        if len(values) != 2:
            raise ValueError(f"{label} must contain two ordered endpoints")
        return I(*(exact_or_represented_real(value, label) for value in values))

    form = interval(form_bounds, "form_bounds")
    rate = interval(rate_bounds, "rate_bounds")
    acceleration = interval(acceleration_bounds, "acceleration_bounds")
    square = rate**2
    gap = square - form * acceleration
    reasons = []
    if form.contains(0):
        reasons.append("initial_form_not_separated_from_zero")
    if (form.lo > 0 and rate.lo > 0) or (form.hi < 0 and rate.hi < 0):
        reasons.append("incompatible_initial_decay")
    if gap.hi <= 0:
        reasons.append("nonpositive_restoring_gap")
    elif gap.lo <= 0:
        reasons.append("unresolved_restoring_gap")
    coefficient = None if reasons else square / (pi_interval() ** 2 * gap)
    return RelationalCoefficientJetBounds(
        form_bounds=(form.lo, form.hi),
        rate_bounds=(rate.lo, rate.hi),
        acceleration_bounds=(acceleration.lo, acceleration.hi),
        squared_rate_bounds=(square.lo, square.hi),
        restoring_gap_bounds=(gap.lo, gap.hi),
        coefficient_bounds=(
            (coefficient.lo, coefficient.hi) if coefficient is not None else None
        ),
        unavailable_reasons=tuple(reasons),
        arithmetic_method=INTERVAL_METHOD,
    )


def bound_relational_coefficient_from_samples(
    samples, *, sample_step, sample_error_bound, third_derivative_bound
) -> RelationalCoefficientSampleBounds:
    """Propagate three uniform samples and declared errors into the jet observer.

    The noise bound covers every sample and the C3 bound covers the complete
    noiseless window. Neither bound, nor uniform timing, is inferred here.
    """
    raw = _ordered(samples, "samples", limit=4)
    if len(raw) != 3:
        raise ValueError("samples must contain exactly three ordered values")
    values = tuple(exact_or_represented_real(value, "sample") for value in raw)
    step = exact_or_represented_real(sample_step, "sample_step")
    error = exact_or_represented_real(sample_error_bound, "sample_error_bound")
    third = exact_or_represented_real(third_derivative_bound, "third_derivative_bound")
    if step <= 0 or error < 0 or third < 0:
        raise ValueError("require positive sample_step and nonnegative error bounds")
    rate = (-3 * values[0] + 4 * values[1] - values[2]) / (2 * step)
    acceleration = (values[2] - 2 * values[1] + values[0]) / step**2
    rate_error = 4 * error / step + third * step**2 / 3
    acceleration_error = 4 * error / step**2 + third * step
    jet = bound_relational_coefficient_from_jet(
        form_bounds=(values[0] - error, values[0] + error),
        rate_bounds=(rate - rate_error, rate + rate_error),
        acceleration_bounds=(
            acceleration - acceleration_error,
            acceleration + acceleration_error,
        ),
    )
    return RelationalCoefficientSampleBounds(
        samples=values,
        sample_step=step,
        sample_error_bound=error,
        third_derivative_bound=third,
        rate_estimate=rate,
        acceleration_estimate=acceleration,
        rate_error_bound=rate_error,
        acceleration_error_bound=acceleration_error,
        jet=jet,
    )


def _regions(nodes, regions):
    raw = _ordered(regions, "regions")
    if not raw:
        raise ValueError("regions must contain at least one region")
    lookup = {node: index for index, node in enumerate(nodes)}
    result = []
    for region in raw:
        row = _ordered(region, "region")
        if not row:
            raise ValueError("each region must be nonempty")
        try:
            indices = tuple(lookup[node] for node in row)
        except (TypeError, KeyError) as exc:
            raise ValueError("region nodes must belong to the full support") from exc
        if len(set(indices)) != len(indices):
            raise ValueError("region nodes must be distinct")
        result.append(indices)
    return tuple(result)


def _detached_graph(field):
    """Rebuild only the captured state consumed by the existing observers."""
    graph = nx.Graph()
    for i, node in enumerate(field.nodes):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: field.epi[i],
                ALIAS_VF[0]: field.capacity[i],
                ALIAS_THETA[0]: field.phase[i],
                ALIAS_DNFR[0]: field.pressure[i],
            },
        )
    graph.add_edges_from((a, b, {"weight": 1.0}) for a, b in field.edges)
    return graph


def _port_state(field, node):
    if field.work is None or field.relative_resultant is None:
        raise RuntimeError("fresh relational field must contain port evidence")
    index = field.nodes.index(node)
    return RelationalPortState(
        node=node,
        epi=field.epi[index],
        phase=field.phase[index],
        capacity=field.capacity[index],
        degree=sum(node == a or node == b for a, b in field.edges),
        form_gradient=field.work.form_gradient[index],
        relative_resultant=field.relative_resultant[index],
        pressure=field.pressure[index],
        phase_metric=field.phase_metric[index],
        form_rate=field.form_rate[index],
        phase_rate=field.phase_rate[index],
    )


def _field_changes(before_fields, after):
    """Exact represented differences against one or several disjoint fields."""
    origins = {
        node: (field, index)
        for field in before_fields
        for index, node in enumerate(field.nodes)
    }
    changes = {
        name
        + "_change": tuple(
            Fraction(value)
            - Fraction(getattr(origins[node][0], name)[origins[node][1]])
            for node, value in zip(after.nodes, getattr(after, name), strict=True)
        )
        for name in ("form_rate", "phase_rate", "pressure", "phase_metric")
    }
    changes.update(
        {
            name
            + "_change": getattr(after, name)
            - sum((getattr(field, name) for field in before_fields), Fraction(0))
            for name in ("form_storage", "phase_storage", "storage")
        }
    )
    return changes


def observe_relational_attachment(left, right, *, model, bridge):
    """Evaluate one supplied unit bridge on detached admitted component states.

    ``left`` and ``right`` must each satisfy the native relational graph/state
    contract, with disjoint node labels. ``bridge`` is an ordered pair naming
    one left port and one right port. The selected model must use the acute
    phase domain; the joined field must independently pass that same admission.
    Invalid components or a nonacute new edge raise without changing either
    graph. Zero capacities retain the native frozen-row semantics.

    Both components are evaluated separately. Their disconnected union is
    used only for shared transport accounting, never as a relational field.
    A fresh native evaluation on the joined detached state supplies pressure,
    phase mobility and both rates. No support event is committed and no new
    pressure, phase or autonomous connection law is introduced.
    """
    if not isinstance(model, RelationalExchangeModel):
        raise TypeError("model must be a RelationalExchangeModel")
    if model.phase_domain != "acute":
        raise ValueError("relational attachment requires the acute phase domain")
    endpoints = _ordered(bridge, "bridge", limit=3)
    if len(endpoints) != 2:
        raise ValueError("bridge must contain exactly two ordered ports")
    components = (
        evaluate_relational_exchange(left, model=model),
        evaluate_relational_exchange(right, model=model),
    )
    left_nodes, right_nodes = (set(field.nodes) for field in components)
    if left_nodes & right_nodes:
        raise ValueError("component node labels must be disjoint")
    try:
        valid_ports = endpoints[0] in left_nodes and endpoints[1] in right_nodes
    except TypeError as exc:
        raise ValueError(
            "bridge ports must belong to their respective components"
        ) from exc
    if not valid_ports:
        raise ValueError("bridge ports must belong to their respective components")

    detached = nx.compose(*(_detached_graph(field) for field in components))
    # The admitted model has no forcing. Preserve that explicit declaration
    # when reconstructing only its consumed state on the detached support.
    detached.graph["GAMMA"] = {"type": "none"}
    before_transport = observe_support_transport(detached)
    detached.add_edge(*endpoints, weight=1.0)
    joined = evaluate_relational_exchange(detached, model=model)
    after_transport = observe_support_transport(_detached_graph(joined))
    reset = observe_support_transport_reset(before_transport, after_transport)
    cut = observe_regional_support_cut(after_transport, components[0].nodes)
    return RelationalAttachmentObservation(
        components=components,
        joined=joined,
        bridge=endpoints,
        ports=tuple(
            RelationalAttachmentPort(
                _port_state(field, node), _port_state(joined, node)
            )
            for field, node in zip(components, endpoints, strict=True)
        ),
        **_field_changes(components, joined),
        transport_reset=reset,
        cut=cut,
    )


def observe_relational_relocation(graph, *, model, remove_bridge, add_bridge):
    """Compare a supplied bridge relocation at unchanged primitive state.

    The acute simple unit connected input is evaluated through the native
    owner. Removing ``remove_bridge`` must leave exactly two connected
    components with at least two nodes each. Its ordered endpoints identify
    the first and second components; ``add_bridge`` must join them in that
    same order and must be absent from the original graph. All internal
    component edges remain unchanged. The new connected graph independently
    passes the same native field admission. Zero capacities remain valid.

    Only detached state is edited. No field is evaluated on the disconnected
    intermediate support. The report can describe either sign of storage
    change; its optional declared-work assessment is not an event selector,
    time law, ideal trigonometric bound or subsequent recovery certificate.
    """
    if not isinstance(model, RelationalExchangeModel):
        raise TypeError("model must be a RelationalExchangeModel")
    if model.phase_domain != "acute":
        raise ValueError("relational relocation requires the acute phase domain")
    old = _ordered(remove_bridge, "remove_bridge", limit=3)
    new = _ordered(add_bridge, "add_bridge", limit=3)
    if len(old) != 2 or len(new) != 2:
        raise ValueError("each bridge must contain exactly two ordered ports")
    before = evaluate_relational_exchange(graph, model=model)
    detached = _detached_graph(before)
    detached.graph["GAMMA"] = {"type": "none"}
    try:
        valid_nodes = all(node in detached for node in (*old, *new))
    except TypeError as exc:
        raise ValueError("bridge ports must belong to the full support") from exc
    if not valid_nodes:
        raise ValueError("bridge ports must belong to the full support")
    if not detached.has_edge(*old):
        raise ValueError("remove_bridge must be an existing edge")
    if detached.has_edge(*new):
        raise ValueError("add_bridge must be absent from the original support")
    before_transport = observe_support_transport(detached)
    detached.remove_edge(*old)
    parts = tuple(nx.connected_components(detached))
    if len(parts) != 2 or any(len(part) < 2 for part in parts):
        raise ValueError("remove_bridge must separate two nontrivial components")
    left = next(part for part in parts if old[0] in part)
    right = next(part for part in parts if old[1] in part)
    if new[0] not in left or new[1] not in right:
        raise ValueError("add_bridge must cross the same ordered components")
    components = tuple(
        tuple(node for node in before.nodes if node in part) for part in (left, right)
    )
    detached.add_edge(*new, weight=1.0)
    after = evaluate_relational_exchange(detached, model=model)
    after_transport = observe_support_transport(_detached_graph(after))
    affected = set((*old, *new))
    return RelationalRelocationObservation(
        before=before,
        after=after,
        components=components,
        remove_bridge=old,
        add_bridge=new,
        ports=tuple(
            RelationalAttachmentPort(
                _port_state(before, node), _port_state(after, node)
            )
            for node in after.nodes
            if node in affected
        ),
        **_field_changes((before,), after),
        transport_reset=observe_support_transport_reset(
            before_transport, after_transport
        ),
        cut_before=observe_regional_support_cut(before_transport, components[0]),
        cut_after=observe_regional_support_cut(after_transport, components[0]),
    )


def _regional_work(field, indices):
    work = field.work
    if work is None:
        raise RuntimeError("fresh relational field must contain work accounting")
    return RegionalRelationalWork(
        **{
            name: sum((getattr(work, name)[i] for i in indices), Fraction(0))
            for name in RegionalRelationalWork.__dataclass_fields__
        }
    )


def _regional_boundary(field, cut, degrees):
    indices = cut.region_indices
    current = cut.outward_cut_current
    e = Fraction(field.model.epi_weight)
    w = Fraction(field.model.phase_weight)
    beta = Fraction(field.model.storage_scale)
    source = sum(
        (w * degrees[i] * Fraction(field.phase_source[i]) for i in indices),
        Fraction(0),
    )
    pressure_defect = sum(
        (degrees[i] * field.pressure_split_residual[i] for i in indices),
        Fraction(0),
    )
    reason = (
        "zero_capacity_in_region"
        if any(field.capacity[i] == 0.0 for i in indices)
        else None
    )
    form_rate = phase_rate = rounding = form_residual = phase_residual = None
    if reason is None:
        form_rate = sum(
            (
                degrees[i] * Fraction(field.form_rate[i]) / Fraction(field.capacity[i])
                for i in indices
            ),
            Fraction(0),
        )
        rounding = sum(
            (
                degrees[i]
                * field.nodal_rate_rounding_defect[i]
                / Fraction(field.capacity[i])
                for i in indices
            ),
            Fraction(0),
        )
        phase_rate = sum(
            (
                Fraction(field.phase_metric[i])
                * Fraction(field.phase_rate[i])
                / Fraction(field.capacity[i])
                for i in indices
            ),
            Fraction(0),
        )
        form_residual = form_rate + e * current - source - pressure_defect - rounding
        phase_residual = phase_rate - w * current / beta
        if form_residual:
            raise RuntimeError("exact represented regional form-rate identity failed")
    return RegionalExchangeBalance(
        cut=cut,
        form_weighted_rate=form_rate,
        form_boundary_rate=-e * current,
        form_source_rate=source,
        form_pressure_defect_rate=pressure_defect,
        form_rounding_defect_rate=rounding,
        form_identity_residual=form_residual,
        phase_weighted_rate=phase_rate,
        phase_boundary_rate=w * current / beta,
        phase_rate_residual=phase_residual,
        weighted_rate_unavailable_reason=reason,
    )


def _regional_phase_response(field, cut):
    if (
        field.work is None
        or field.phase_mobility is None
        or field.phase_rate_rounding_defect is None
    ):
        raise RuntimeError(
            "fresh relational field must contain exact phase-rate evidence"
        )
    indices = cut.region_indices
    n = len(indices)
    mobility = tuple(field.phase_mobility[i] for i in indices)
    gradient = tuple(field.work.form_gradient[i] for i in indices)
    mean_mobility = sum(mobility, Fraction(0)) / n
    mean_gradient = sum(gradient, Fraction(0)) / n
    centered_mobility = tuple(a - mean_mobility for a in mobility)
    centered_gradient = tuple(q - mean_gradient for q in gradient)
    mobility_variance = sum((a * a for a in centered_mobility), Fraction(0)) / n
    gradient_variance = sum((q * q for q in centered_gradient), Fraction(0)) / n
    covariance = (
        sum((a * q for a, q in zip(centered_mobility, centered_gradient)), Fraction(0))
        / n
    )
    k = Fraction(field.model.phase_weight) / Fraction(field.model.storage_scale)
    boundary_rate = k * mean_mobility * cut.outward_cut_current
    covariance_rate = k * n * covariance
    squared_bound = k * k * n * n * mobility_variance * gradient_variance
    model_total = k * sum((a * q for a, q in zip(mobility, gradient)), Fraction(0))
    rounding = sum((field.phase_rate_rounding_defect[i] for i in indices), Fraction(0))
    total = sum((Fraction(field.phase_rate[i]) for i in indices), Fraction(0))
    residual = total - boundary_rate - covariance_rate - rounding
    if residual or model_total != boundary_rate + covariance_rate:
        raise RuntimeError("exact represented regional phase-rate identity failed")
    if covariance_rate * covariance_rate > squared_bound:
        raise RuntimeError("exact regional covariance-rate bound failed")
    return RegionalPhaseResponse(
        mean_mobility=mean_mobility,
        mean_form_gradient=mean_gradient,
        mobility_variance=mobility_variance,
        form_gradient_variance=gradient_variance,
        mobility_gradient_covariance=covariance,
        mean_mobility_boundary_rate=boundary_rate,
        covariance_rate=covariance_rate,
        covariance_rate_squared_bound=squared_bound,
        model_total_rate=model_total,
        rounding_residual=rounding,
        total_rate=total,
        mean_rate=total / n,
        identity_residual=residual,
    )


def observe_relational_pattern(
    graph,
    *,
    model: RelationalExchangeModel,
    reference_phase: Mapping,
    regions,
    cycles=(),
) -> RelationalPatternObservation:
    """Observe supplied regions using exactly one fresh relational evaluation.

    Reference phases must map the exact full node support to finite represented
    real lifts. Regions are ordered, nonempty selections without repeated nodes;
    they may overlap and need not partition support. No reference equilibrium,
    lift compatibility over time or automatic regional discovery is inferred.

    Transport is unavailable if any full-support capacity is zero, EPI weight
    is zero, or the region covers all support (in that order of precedence).
    Those are limitations of the existing regional transport contract, not
    failures of the retained geometry. Its forcing is independently supplied
    as ``w*phase_source``; native pressure-split defects are retained.

    Work and paired boundary balances also cover full-support regions and
    zero EPI weight. Their actual divided rates require positive capacity only
    inside the selected region. These more general read-outs do not broaden
    the older transport metric's admission. Each actual report populates
    ``work``, ``boundary`` and ``phase_response``; their optional defaults only
    preserve older manually constructed records. The unweighted phase response
    splits the actual phase-rate sum into mean-mobility cut, mobility/form
    covariance and exact rounding residual. It admits zero capacity and retains
    an exact squared covariance-rate bound without inventing a tolerance.

    Optional ordered cycles delegate geometric availability to the existing
    winding owner. No observation advances a graph or certifies a trajectory.
    """
    field = evaluate_relational_exchange(graph, model=model)
    if not isinstance(reference_phase, Mapping):
        raise TypeError("reference_phase must map the exact full node support")
    if set(reference_phase) != set(field.nodes):
        raise ValueError("reference_phase must match the exact full node support")
    reference = tuple(
        finite_represented_real(reference_phase[node], "reference phase")[0]
        for node in field.nodes
    )
    selections = _regions(field.nodes, regions)
    ordered_cycles = tuple(
        _ordered(cycle, "cycle") for cycle in _ordered(cycles, "cycles")
    )
    detached = _detached_graph(field)
    unavailable = (
        "zero_capacity_in_full_support"
        if any(capacity == 0.0 for capacity in field.capacity)
        else ("zero_epi_weight" if model.epi_weight == 0.0 else None)
    )
    source = observe_support_transport(detached)
    degrees = tuple(len(row) for row in source.support_neighbors)
    forcing = tuple(
        Fraction(model.phase_weight) * Fraction(value) for value in field.phase_source
    )
    observed = []
    for indices in selections:
        nodes = tuple(field.nodes[i] for i in indices)
        form = tuple(Fraction(field.epi[i]) for i in indices)
        error = tuple(
            Fraction(field.phase[i]) - Fraction(reference[i]) for i in indices
        )
        form_mean, phase_mean = sum(form) / len(form), sum(error) / len(error)
        centered_form = tuple(value - form_mean for value in form)
        centered_phase = tuple(value - phase_mean for value in error)
        reason = unavailable or (
            "full_support_region" if len(indices) == len(field.nodes) else None
        )
        transport = (
            observe_regional_support_balance(
                source, nodes, epi_weight=Fraction(model.epi_weight), forcing=forcing
            )
            if reason is None
            else None
        )
        cut = (
            transport.cut
            if transport is not None
            else observe_regional_support_cut(source, nodes)
        )
        observed.append(
            RegionalPatternObservation(
                nodes=nodes,
                form_mean=form_mean,
                phase_error_mean=phase_mean,
                centered_form=centered_form,
                centered_phase_error=centered_phase,
                form_norm_squared=sum(
                    (value**2 for value in centered_form), Fraction(0)
                ),
                phase_norm_squared=sum(
                    (value**2 for value in centered_phase), Fraction(0)
                ),
                transport=transport,
                transport_unavailable_reason=reason,
                work=_regional_work(field, indices),
                boundary=_regional_boundary(field, cut, degrees),
                phase_response=_regional_phase_response(field, cut),
            )
        )
    return RelationalPatternObservation(
        field=field,
        reference_phase=reference,
        regions=tuple(observed),
        winding=tuple(
            certify_phase_winding(detached, cycle) for cycle in ordered_cycles
        ),
    )
