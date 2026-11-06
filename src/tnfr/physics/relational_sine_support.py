"""Detached conservative sine exchange with retained conductance and bare support.

Form storage and its reciprocal phase row consume the actual conductances.
The sine pressure, phase storage and normalization consume bare neighbors.
This explicit zero-loss completion neither changes native execution nor admits
weighted data to the separate unit-support comparison and its theorems.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..dynamics.relational import (
    RelationalExchangeModel,
    _admit_relational_graph_structure,
    _relational_node_coordinates,
)
from ..mathematics._phase_resultant_chamber import relative_resultant_bounds
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from ._sine_admission import _sine_model_coefficients
from ._sine_preparation import _prepared_form_costs
from .relational_observations import _ordered
from .relational_sine_comparison import (
    _sine_phase_rate_numerators,
    _sine_phase_storage,
    _sine_work,
    _validate_comparison_labels,
)
from .support_transport import _support_components

__all__ = (
    "SineSupportExchangeComparison",
    "SineFiniteFormResponse",
    "SineInitialStatePerturbation",
    "SineStateConductancePerturbation",
    "SineSupportRecurrence",
    "SineFrozenIsolateComparison",
    "bound_relational_sine_support_exchange",
    "bound_relational_sine_frozen_isolates",
)

_LAW = "conservative_sine_support_exchange"
_FROZEN_ISOLATE_LAW = "conservative_sine_support_exchange_with_frozen_isolates"


def _conservative_coefficients(model):
    coefficients = _sine_model_coefficients(model)
    if coefficients[0] != 0:
        raise ValueError("support sine exchange requires exact zero form loss")
    return coefficients


@dataclass(frozen=True)
class SineSupportExchangeComparison:
    """Instantaneous ideal bounds for one fixed weighted-form/bare-phase law.

    ``conductances`` follows ``edges``, including supported zero-weight edges.
    ``degrees`` counts those bare edges; it is not conductance strength.
    Consumers rebuild derived fields from primitive state, law and weights.
    The report is neither a trajectory nor provenance authentication.
    """

    reference_model: RelationalExchangeModel
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    conductances: tuple[Q, ...]
    epi: tuple[Q, ...]
    phase: tuple[Q, ...]
    capacity: tuple[Q, ...]
    degrees: tuple[int, ...]
    form_gradient: tuple[Q, ...]
    relative_resultant: tuple[tuple[I, I], ...]
    inverse_phase_metric: tuple[I, ...]
    phase_sources: tuple[I, ...]
    pressure: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    form_storage: Q
    phase_storage: I
    storage: I
    dissipation: tuple[Q, ...]
    continuous_loss: Q
    form_work: tuple[I, ...]
    phase_work: tuple[I, ...]
    node_balance_residual: tuple[I, ...]
    storage_rate: I
    balance_residual: I
    law: str = _LAW
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "explicit_conservative_weighted_form_and_unweighted_support_phase_completion",
        "fixed_simple_connected_bare_support_and_finite_nonnegative_conductances",
        "bare_neighbor_degree_normalization_in_both_complete_rows",
        "held_nonnegative_capacity_exact_zero_loss_no_input_or_events",
        "supported_zero_conductance_edges_remain_in_phase_pressure_and_storage",
        "ideal_instantaneous_enclosures_at_captured_represented_coordinates",
        "computed_work_residuals_not_assigned_zero",
        "no_native_dispatch_trajectory_recovery_or_automatic_theorem_transfer",
        "no_event_selector_unique_constitutive_selection_or_physical_identification",
    )

    def phase_rate_numerators(self) -> tuple[Q, ...]:
        """Rebuild exact weighted-gradient phase rates before their pi divisor."""
        source = _rebuild_sine_support_comparison(self)
        return _sine_phase_rate_numerators(
            source.reference_model,
            source.degrees,
            source.form_gradient,
            source.capacity,
        )

    def recurrence_bounds(
        self, *, energy_ceiling, form_mean_bounds
    ) -> SineSupportRecurrence:
        """Admit a compact invariant family for almost-everywhere recurrence.

        Positive held capacities and a connected positive-conductance subgraph
        make the conserved weighted form mean and full storage control every
        real form coordinate. Supported zero conductances remain bare phase
        neighbors. Circular phases supply a compact torus, without any bound
        or recurrence claim for their real lifts. The complete vector field
        has zero divergence and preserves product form/phase volume.

        The declared positive energy ceiling and positive-width mean interval
        specify the family independently of the captured state. Its membership
        is checked separately; even admitted membership does not certify that
        particular state's recurrence, a return time or pattern formation.
        """
        return _support_recurrence_bounds(
            self, energy_ceiling=energy_ceiling, form_mean_bounds=form_mean_bounds
        )

    def form_acceleration_bounds(self) -> tuple[I, ...]:
        """Enclose the instantaneous complete-law second form derivative.

        Differentiate the form row using the declared reciprocal phase row,
        with held support, conductances, capacity, coefficients and clock.
        Bare neighbors enter the cosine sum and degree normalization; actual
        conductances enter the phase velocity through the weighted gradient.
        In particular, a zero-conductance edge still contributes phase incidence.

        Primitive source data are re-admitted and derived fields rebuilt before
        evaluation. The result is neither sampled temporal evidence nor a
        whole-window bound, trajectory, operator trigger or occurrence decision.
        """
        source = _rebuild_sine_support_comparison(self)
        _, neighbors = _support_topology(source.nodes, source.edges)
        numerators = _sine_phase_rate_numerators(
            source.reference_model,
            source.degrees,
            source.form_gradient,
            source.capacity,
        )
        _, exchange, _ = _conservative_coefficients(source.reference_model)
        pi_squared = pi_interval() ** 2
        return tuple(
            (exchange * source.capacity[index] / source.degrees[index])
            / pi_squared
            * sum(
                (
                    cos(I(source.phase[other] - source.phase[index]))
                    * (numerators[other] - numerators[index])
                    for other in adjacent
                ),
                I(0),
            )
            for index, adjacent in enumerate(neighbors)
        )

    def conditional_form_acceleration_bounds(self, *, form_radius) -> tuple[I, ...]:
        """Bound the complete-law acceleration on a centered-form ball.

        The rebuilt captured form must satisfy ``||x-mean(x)||_2 <= r``.
        Every returned interval also bounds any later state in that same ball
        under the held support, conductances, capacities, coefficients and
        clock, without inputs or events. Initial membership does not establish
        future confinement. Phase is unrestricted; zero-weight edges retain
        their bare phase incidence and degree normalization.

        This is an ideal derivative envelope, not a sampled history, numerical
        trajectory, invariant-ball certificate or operator trigger decision.
        Primitive source admission and derived-field rebuilding precede use.
        """
        source = _rebuild_sine_support_comparison(self)
        return _conditional_form_acceleration_from_admitted_source(
            source, form_radius=form_radius
        )

    def finite_form_response_bounds(self, *, duration) -> SineFiniteFormResponse:
        """Enclose finite form coordinates under this complete held law.

        For exchange coefficient w, bare sine incidence gives the global bound
        ``|x_dot_i| <= w*capacity_i/pi``. Orthogonal projection onto arithmetic
        centered forms has norm one, even when that arithmetic mean moves.
        Thus ``r(t) <= r(0) + duration*w*||capacity||_2/pi`` throughout the
        requested interval. The shared conditional acceleration envelope on
        this derived ball bounds the first-order Taylor remainder by A*h^2/2.

        Positive durations have no additional time cap. Every phase coordinate
        continues under its reciprocal weighted-gradient row; no frozen phase
        or support-event approximation is made. The globally Lipschitz full
        lifted field supplies existence for every finite duration. Nonnegative
        capacities, including zero, retain their existing admission contract.

        Source primitives and rates are rebuilt before use. These ideal
        enclosures neither evaluate a trajectory nor certify a nonzero contrast,
        operator trigger, retained measurement or numerical solver endpoint.
        """
        source = _rebuild_sine_support_comparison(self)
        horizon = exact_or_represented_real(duration, "duration")
        if horizon <= 0:
            raise ValueError("duration must be strictly positive")
        mean = sum(source.epi, Q(0)) / len(source.nodes)
        initial_radius = sqrt(
            I(sum(((value - mean) ** 2 for value in source.epi), Q(0)))
        ).hi
        _, exchange, _ = _conservative_coefficients(source.reference_model)
        global_speed = (
            exchange
            * sqrt(I(sum((value**2 for value in source.capacity), Q(0))))
            / pi_interval()
        ).hi
        window_radius = initial_radius + horizon * global_speed
        accelerations = _conditional_form_acceleration_from_admitted_source(
            source, form_radius=window_radius
        )
        remainders = tuple(value.abs_max * horizon**2 / 2 for value in accelerations)
        endpoint = tuple(
            I(form) + horizon * rate + I(-error, error)
            for form, rate, error in zip(source.epi, source.form_rates, remainders)
        )
        return SineFiniteFormResponse(
            source=source,
            duration=horizon,
            initial_centered_form_radius_upper_bound=initial_radius,
            global_form_speed_upper_bound=global_speed,
            whole_window_form_radius_upper_bound=window_radius,
            initial_form_rates=source.form_rates,
            acceleration_bounds=accelerations,
            form_remainder_upper_bounds=remainders,
            endpoint_form_bounds=endpoint,
        )

    def initial_state_perturbation_bounds(
        self, *, duration, form_error_bounds, phase_error_bounds
    ) -> SineInitialStatePerturbation:
        """Bound two full solutions differing only in their initial coordinates.

        Each error radius follows the captured node order. Form errors retain
        form units and phase errors are differences of declared radian lifts.
        Both solutions use this same held support, conductances, capacities,
        coefficients and original clock, with no inputs or events. No source
        capture or initial uncertainty distribution is inferred from the radii.

        Sine is globally one-Lipschitz. With strength s_i and bare degree d_i,
        the coupled error inequalities have coefficients
        a_i=2*w*nu_i/pi and b_i=2*w*nu_i*s_i/(beta*pi*d_i).
        Their maxima A,B bound the whole-window form and phase suprema through
        X<=e_x+A*h*Theta and Theta<=e_theta+B*h*X. A positive lower margin
        D=1-A*B*h^2 supplies the rational comparison envelope. Failure of this
        sufficient margin returns unavailable, not instability. Identical
        initial coordinates instead give zero error at every duration by
        uniqueness, independently of the comparison margin.

        Initial storage is bounded separately, edge by edge, for the complete
        supplied error box; it remains available when the time bound abstains.
        The result is not a trajectory, decay proof or comparison of two laws.
        """
        source = _rebuild_sine_support_comparison(self)
        horizon = exact_or_represented_real(duration, "duration")
        if horizon <= 0:
            raise ValueError("duration must be strictly positive")
        size = len(source.nodes)
        form_errors = _support_values(
            form_error_bounds, "form_error_bounds", size, nonnegative=True
        )
        phase_errors = _support_values(
            phase_error_bounds, "phase_error_bounds", size, nonnegative=True
        )
        common, _ = _perturbation_envelope(
            source,
            horizon,
            form_errors,
            phase_errors,
            (Q(0),) * len(source.edges),
        )
        return SineInitialStatePerturbation(
            source=source,
            duration=horizon,
            form_error_bounds=form_errors,
            phase_error_bounds=phase_errors,
            **common,
        )

    def state_and_conductance_perturbation_bounds(
        self,
        *,
        duration,
        form_error_bounds,
        phase_error_bounds,
        conductance_error_bounds,
    ) -> SineStateConductancePerturbation:
        """Bound initial-coordinate and held-conductance perturbations.

        The two complete laws share bare support, held capacities, coefficients
        and clock. Candidate conductances must be nonnegative and differ from
        the captured conductances by at most the supplied edge-ordered radii.
        Zero-conductance edges remain in the unchanged bare support. The upper
        conductances W+rho bound candidate form-to-phase feedback; changing W
        changes only the reciprocal phase row, not the bare sine form row.

        Reference form speeds v_i=w*nu_i/pi give the whole-window edge bound
        q_ij=|x_i(0)-x_j(0)|+h*(v_i+v_j). Conductance differences then supply
        phase forcing F_i=w*nu_i*sum_j(rho_ij*q_ij)/(beta*pi*d_i). The shared
        comparison kernel retains this forcing in both coupled error bounds.
        Zero state errors alone do not imply equal solutions under two laws.

        Phase errors concern declared real lifts. A consumer associating raw
        circular outputs must independently certify its integer 2*pi shifts;
        subtraction of represented math.tau is not an exact symmetry. This
        box reader does not admit or authenticate any actual candidate source,
        operator output, event error budget or source-to-flow handoff.
        """
        source = _rebuild_sine_support_comparison(self)
        horizon = exact_or_represented_real(duration, "duration")
        if horizon <= 0:
            raise ValueError("duration must be strictly positive")
        size = len(source.nodes)
        form_errors = _support_values(
            form_error_bounds, "form_error_bounds", size, nonnegative=True
        )
        phase_errors = _support_values(
            phase_error_bounds, "phase_error_bounds", size, nonnegative=True
        )
        conductance_errors = _support_values(
            conductance_error_bounds,
            "conductance_error_bounds",
            len(source.edges),
            nonnegative=True,
        )
        common, changed = _perturbation_envelope(
            source, horizon, form_errors, phase_errors, conductance_errors
        )
        if common["comparison_method"] == "zero_initial_error_uniqueness":
            common["comparison_method"] = "zero_state_and_conductance_error_uniqueness"
        return SineStateConductancePerturbation(
            source=source,
            duration=horizon,
            form_error_bounds=form_errors,
            phase_error_bounds=phase_errors,
            conductance_error_bounds=conductance_errors,
            **common,
            **changed,
        )

    def to_dict(self):
        """Export freshly rebuilt bounds; primitive data do not authenticate capture."""
        from ..sdk.relational_reports import _project

        source = _rebuild_sine_support_comparison(self)
        _validate_comparison_labels(source)
        return {
            "schema": "tnfr.relational-sine-support-comparison.v1",
            "report": _project(source),
        }


@dataclass(frozen=True)
class SineSupportRecurrence:
    """Finite-volume weighted sine family, not a chosen-orbit prediction.

    Weights d/nu use bare degrees, whereas positive conductance paths control
    form differences. The conservative n-1 path bound gives
    |x_i-mean_mu(x)| <= sqrt(2*(n-1)*energy_ceiling/min_positive_weight).
    It is a compactness bound, not a computed diameter or sharp energy barrier.
    Phase recurrence is circular; no lifted phase mean or bounded lift is used.
    """

    source: SineSupportExchangeComparison
    energy_ceiling: Q
    form_mean_bounds: tuple[Q, Q]
    form_mean_weights: tuple[Q, ...]
    normalized_mean_weights: tuple[Q, ...]
    weighted_form_mean: Q
    weighted_mean_rate_residual_bounds: I
    divergence: Q
    positive_conductance_edge_indices: tuple[int, ...]
    positive_conductance_weight_floor: Q
    path_length_upper_bound: int
    form_radius_bounds: I
    form_coordinate_bounds: I
    captured_energy_exact: Q | None
    captured_storage_bounds: I
    snapshot_family_membership: str
    snapshot_membership_reasons: tuple[str, ...]
    almost_everywhere_recurrence_certified: bool = True
    finite_positive_family_volume_certified: bool = True
    individual_recurrence_status: str = "unavailable_for_chosen_state"
    clock: str = "t"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_primitive_source_complete_held_conservative_weighted_sine_law",
        "strictly_positive_held_capacities_and_connected_positive_conductance_support",
        "supported_zero_conductances_remain_in_bare_degrees_pressure_and_phase_storage",
        "positive_energy_ceiling_and_positive_width_form_mean_slab_are_supplied",
        "full_storage_and_actual_bare_degree_over_capacity_form_mean_are_invariant",
        "zero_divergence_follows_from_separate_form_and_phase_coordinate_dependencies",
        "weighted_mean_rate_residual_is_computed_from_rebuilt_original_time_rates",
        "positive_weight_path_bound_controls_all_real_form_coordinates",
        "finite_positive_product_Lebesgue_form_and_circle_Haar_volume",
        "recurrence_is_almost_everywhere_on_the_declared_invariant_family",
        "captured_membership_is_separate_from_the_family_theorem",
        "no_individual_recurrence_verdict_for_any_selected_captured_state",
        "probability_one_requires_absolutely_continuous_preparation_in_the_family",
        "no_transfer_to_exact_energy_fixed_mean_or_other_lower_dimensional_preparations",
        "circular_phase_return_not_real_lift_recurrence_or_a_conserved_circular_mean",
        "no_return_time_periodicity_attraction_formation_or_physical_identity",
        "no_rank_spectrum_trajectory_events_or_change_to_native_execution",
    )

    def to_dict(self):
        """Rebuild the source, family bounds and membership before exporting."""
        if type(self) is not SineSupportRecurrence:
            raise TypeError("a SineSupportRecurrence is required")
        fresh = _support_recurrence_bounds(
            self.source,
            energy_ceiling=self.energy_ceiling,
            form_mean_bounds=self.form_mean_bounds,
        )
        _validate_comparison_labels(fresh.source)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-support-recurrence.v1",
            "report": _project(fresh),
        }


def _support_recurrence_bounds(source, *, energy_ceiling, form_mean_bounds):
    source = _rebuild_sine_support_comparison(source)
    maximum = exact_or_represented_real(energy_ceiling, "energy_ceiling")
    if maximum <= 0:
        raise ValueError("energy_ceiling must be strictly positive")
    raw_mean = _ordered(form_mean_bounds, "form_mean_bounds", limit=3)
    if len(raw_mean) != 2:
        raise ValueError("form_mean_bounds must contain exactly two ordered endpoints")
    lower, upper = (
        exact_or_represented_real(value, "form_mean_bounds") for value in raw_mean
    )
    if lower >= upper:
        raise ValueError("form_mean_bounds must have strictly positive width")
    if any(value <= 0 for value in source.capacity):
        raise ValueError("recurrence admission requires positive held capacities")
    edges, _ = _support_topology(source.nodes, source.edges)
    positive_edges = tuple(
        index for index, weight in enumerate(source.conductances) if weight > 0
    )
    neighbors = [[] for _ in source.nodes]
    for index in positive_edges:
        i, j = edges[index]
        neighbors[i].append(j)
        neighbors[j].append(i)
    if len(_support_components(tuple(tuple(row) for row in neighbors))) != 1:
        raise ValueError("recurrence requires connected positive-conductance support")
    weight_floor = min(source.conductances[index] for index in positive_edges)
    weights = tuple(
        Q(degree) / capacity
        for degree, capacity in zip(source.degrees, source.capacity)
    )
    total = sum(weights, Q(0))
    normalized = tuple(weight / total for weight in weights)
    mean = sum((weight * value for weight, value in zip(normalized, source.epi)), Q(0))
    mean_rate = sum(
        (weight * rate for weight, rate in zip(normalized, source.form_rates)), I(0)
    )
    if not mean_rate.lo <= 0 <= mean_rate.hi:
        raise ArithmeticError("weighted sine form mean lost its exact cancellation")
    path_bound = len(source.nodes) - 1
    radius = sqrt(I(2 * path_bound * maximum / weight_floor))
    equal_phases = all(value == source.phase[0] for value in source.phase)
    exact_energy = source.form_storage if equal_phases else None
    storage = I(exact_energy) if exact_energy is not None else source.storage
    reasons = []
    if not lower <= mean <= upper:
        membership = "outside"
        reasons.append("weighted_mean_outside_declared_slab")
    elif exact_energy is not None:
        membership = "inside" if exact_energy <= maximum else "outside"
        if membership == "outside":
            reasons.append("exact_energy_exceeds_declared_ceiling")
    elif storage.hi <= maximum:
        membership = "inside"
    elif storage.lo > maximum:
        membership = "outside"
        reasons.append("energy_lower_bound_exceeds_declared_ceiling")
    else:
        membership = "unresolved"
        reasons.append("energy_ceiling_membership_not_resolved")
    return SineSupportRecurrence(
        source=source,
        energy_ceiling=maximum,
        form_mean_bounds=(lower, upper),
        form_mean_weights=weights,
        normalized_mean_weights=normalized,
        weighted_form_mean=mean,
        weighted_mean_rate_residual_bounds=mean_rate,
        divergence=Q(0),
        positive_conductance_edge_indices=positive_edges,
        positive_conductance_weight_floor=weight_floor,
        path_length_upper_bound=path_bound,
        form_radius_bounds=radius,
        form_coordinate_bounds=I(lower - radius.hi, upper + radius.hi),
        captured_energy_exact=exact_energy,
        captured_storage_bounds=storage,
        snapshot_family_membership=membership,
        snapshot_membership_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class SineFiniteFormResponse:
    """Finite original-clock form enclosures from a complete admitted source.

    All nodal tuples follow ``source.nodes``. ``initial_form_rates`` contains
    outward numerical bounds at the initial state; the separate remainder
    bounds control the exact nonlinear flow throughout ``[0, duration]``.
    A zero-containing contrast is inconclusive, not evidence of zero response.
    """

    source: SineSupportExchangeComparison
    duration: Q
    initial_centered_form_radius_upper_bound: Q
    global_form_speed_upper_bound: Q
    whole_window_form_radius_upper_bound: Q
    initial_form_rates: tuple[I, ...]
    acceleration_bounds: tuple[I, ...]
    form_remainder_upper_bounds: tuple[Q, ...]
    endpoint_form_bounds: tuple[I, ...]
    status: str = "enclosed"
    clock: str = "t"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_primitive_source_law_support_and_capacity_rebuilt_before_use",
        "held_conservative_weighted_form_and_bare_sine_phase_law",
        "nonnegative_held_capacities_include_frozen_zero_capacity_form_rows",
        "global_form_speed_derives_centered_form_confinement_on_zero_to_duration",
        "arithmetic_centering_does_not_assume_conservation_of_the_arithmetic_mean",
        "complete_evolving_phase_row_retained_in_the_acceleration_envelope",
        "finite_form_enclosure_from_initial_rates_and_uniform_nonlinear_remainder",
        "no_inputs_events_trajectory_solver_endpoint_or_acquired_history",
        "enclosure_status_does_not_certify_a_nonzero_response_or_operator_trigger",
    )

    def to_dict(self):
        """Rebuild from primitive source and duration, discarding cached bounds."""
        if type(self) is not SineFiniteFormResponse:
            raise TypeError("a SineFiniteFormResponse is required")
        fresh = SineSupportExchangeComparison.finite_form_response_bounds(
            self.source, duration=self.duration
        )
        _validate_comparison_labels(fresh.source)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-finite-form-response.v1",
            "report": _project(fresh),
        }


@dataclass(frozen=True)
class SineInitialStatePerturbation:
    """Same-law full-coordinate error and initial storage envelopes.

    Error tuples follow ``source.nodes``. Available error bounds apply at every
    time in ``[0, duration]`` to every supplied initial error in the box, relative
    to the exact solution from the nominal source. Form and radian-phase maxima
    remain separate quantities with separate units. An unavailable time bound
    supplies no error envelope; its initial storage upper bound is independent.
    ``initial_phase_storage_upper_bound`` includes the admitted beta multiplier,
    so the two initial storage contributions sum to the full storage bound.
    """

    source: SineSupportExchangeComparison
    duration: Q
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    form_phase_lipschitz_upper_bounds: tuple[Q, ...]
    phase_form_lipschitz_upper_bounds: tuple[Q, ...]
    max_form_phase_lipschitz_upper_bound: Q
    max_phase_form_lipschitz_upper_bound: Q
    comparison_denominator: Q
    form_error_supremum_upper_bound: Q | None
    phase_error_supremum_upper_bound: Q | None
    whole_window_form_error_bounds: tuple[Q, ...] | None
    whole_window_phase_error_bounds: tuple[Q, ...] | None
    initial_form_storage_upper_bound: Q
    initial_phase_storage_upper_bound: Q
    initial_storage_upper_bound: Q
    status: str
    unavailable_reasons: tuple[str, ...]
    comparison_method: str
    clock: str = "t"
    phase_error_units: str = "radians_in_declared_real_lifts"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_complete_primitive_source_law_support_and_capacity_admission",
        "both_solutions_share_held_support_conductances_capacities_and_coefficients",
        "independent_initial_fine_form_and_radian_lift_error_box_in_source_node_order",
        "whole_window_full_nonlinear_error_without_collective_projection",
        "outward_coefficient_bounds_and_strict_rational_comparison_margin",
        "identical_initial_coordinates_have_zero_error_by_unique_complete_flow",
        "supported_zero_conductances_retain_bare_phase_incidence",
        "initial_full_storage_bound_is_separate_from_time_envelope_availability",
        "unavailable_comparison_margin_does_not_prove_instability_or_identity_loss",
        "no_inputs_events_trajectory_acquisition_decay_or_operator_admission",
    )

    def to_dict(self):
        """Rebuild bounds from the nominal source, duration and initial radii."""
        if type(self) is not SineInitialStatePerturbation:
            raise TypeError("a SineInitialStatePerturbation is required")
        fresh = SineSupportExchangeComparison.initial_state_perturbation_bounds(
            self.source,
            duration=self.duration,
            form_error_bounds=self.form_error_bounds,
            phase_error_bounds=self.phase_error_bounds,
        )
        _validate_comparison_labels(fresh.source)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-initial-state-perturbation.v1",
            "report": _project(fresh),
        }


@dataclass(frozen=True)
class SineStateConductancePerturbation:
    """Full-flow error boxes for two held laws with bounded weight differences.

    Node tuples follow ``source.nodes`` and edge tuples follow ``source.edges``.
    Candidate conductances belong to [max(0, W-rho), W+rho]. State errors are
    measured at initialization in form units and declared real radian lifts.
    A candidate's primitive membership in these boxes is a separate obligation;
    this report neither captures a candidate nor verifies an operator endpoint.
    Phase storage bounds include beta. Every available error envelope applies
    throughout [0, duration], without assuming invariance of a collective family.
    """

    source: SineSupportExchangeComparison
    duration: Q
    form_error_bounds: tuple[Q, ...]
    phase_error_bounds: tuple[Q, ...]
    conductance_error_bounds: tuple[Q, ...]
    conductance_upper_bounds: tuple[Q, ...]
    reference_form_speed_upper_bounds: tuple[Q, ...]
    reference_edge_form_contrast_upper_bounds: tuple[Q, ...]
    form_phase_lipschitz_upper_bounds: tuple[Q, ...]
    phase_form_lipschitz_upper_bounds: tuple[Q, ...]
    phase_law_forcing_upper_bounds: tuple[Q, ...]
    max_form_phase_lipschitz_upper_bound: Q
    max_phase_form_lipschitz_upper_bound: Q
    max_phase_law_forcing_upper_bound: Q
    comparison_denominator: Q
    form_error_supremum_upper_bound: Q | None
    phase_error_supremum_upper_bound: Q | None
    whole_window_form_error_bounds: tuple[Q, ...] | None
    whole_window_phase_error_bounds: tuple[Q, ...] | None
    initial_form_storage_upper_bound: Q
    initial_phase_storage_upper_bound: Q
    initial_storage_upper_bound: Q
    status: str
    unavailable_reasons: tuple[str, ...]
    comparison_method: str
    clock: str = "t"
    phase_error_units: str = "radians_in_declared_real_lifts"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_complete_primitive_reference_source_law_and_support_admission",
        "two_fixed_laws_share_bare_support_capacities_coefficients_and_clock",
        "candidate_nonnegative_conductances_lie_within_supplied_edge_error_box",
        "initial_fine_form_and_declared_radian_lift_error_box_in_source_node_order",
        "reference_global_form_speeds_bound_weight_difference_phase_forcing",
        "candidate_strength_upper_bounds_retain_nonuniform_conductance_errors",
        "whole_window_full_nonlinear_error_without_collective_projection",
        "zero_state_and_conductance_errors_give_equality_by_complete_flow_uniqueness",
        "supported_zero_conductances_retain_bare_phase_incidence",
        "initial_storage_upper_bound_uses_candidate_conductance_upper_bounds",
        "initial_storage_is_available_independently_of_time_envelope_availability",
        "unavailable_comparison_margin_does_not_prove_instability_or_identity_loss",
        "actual_candidate_and_integer_phase_lift_association_require_separate_admission",
        "no_operator_error_budget_admission_event_trajectory_acquisition_or_decay",
    )

    def to_dict(self):
        """Rebuild from reference primitives and the declared uncertainty box."""
        if type(self) is not SineStateConductancePerturbation:
            raise TypeError("a SineStateConductancePerturbation is required")
        fresh = SineSupportExchangeComparison.state_and_conductance_perturbation_bounds(
            self.source,
            duration=self.duration,
            form_error_bounds=self.form_error_bounds,
            phase_error_bounds=self.phase_error_bounds,
            conductance_error_bounds=self.conductance_error_bounds,
        )
        _validate_comparison_labels(fresh.source)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-state-conductance-perturbation.v1",
            "report": _project(fresh),
        }


def _conditional_form_acceleration_from_admitted_source(source, *, form_radius):
    """Evaluate the common envelope on a source freshly rebuilt by its caller.

    Public readers must re-admit their source before entering this helper.
    Keeping this calculation internal avoids a second full field rebuild in
    the finite-response reader, without trusting any caller-supplied cache.
    """
    radius = exact_or_represented_real(form_radius, "form_radius")
    if radius < 0:
        raise ValueError("form_radius must be nonnegative")
    mean = sum(source.epi, Q(0)) / len(source.nodes)
    if sum(((value - mean) ** 2 for value in source.epi), Q(0)) > radius**2:
        raise ValueError("captured centered form exceeds form_radius")
    edges, neighbors = _support_topology(source.nodes, source.edges)
    strength = [Q(0)] * len(source.nodes)
    for (left, right), weight in zip(edges, source.conductances):
        strength[left] += weight
        strength[right] += weight
    mobility = tuple(
        capacity / degree for capacity, degree in zip(source.capacity, source.degrees)
    )
    gradient_bounds = tuple(
        factor * weight for factor, weight in zip(mobility, strength)
    )
    _, exchange, beta = _conservative_coefficients(source.reference_model)
    common = sqrt(I(2)) * radius * (exchange**2 / beta) / pi_interval() ** 2
    result = []
    for index, adjacent in enumerate(neighbors):
        coefficient = mobility[index] * (
            sum((gradient_bounds[other] for other in adjacent), Q(0))
            + len(adjacent) * gradient_bounds[index]
        )
        upper = (common * coefficient).hi
        result.append(I(-upper, upper))
    return tuple(result)


def _perturbation_envelope(
    source, horizon, form_errors, phase_errors, conductance_errors
):
    """Shared arithmetic after complete source and error-box admission."""
    size = len(source.nodes)
    edges, _ = _support_topology(source.nodes, source.edges)
    upper_weights = tuple(
        weight + error for weight, error in zip(source.conductances, conductance_errors)
    )
    strength = [Q(0)] * size
    for (i, j), weight in zip(edges, upper_weights):
        strength[i] += weight
        strength[j] += weight
    _, exchange, beta = _conservative_coefficients(source.reference_model)
    pi = pi_interval()
    form_coefficients = tuple(
        (2 * exchange * capacity / pi).hi for capacity in source.capacity
    )
    phase_coefficients = tuple(
        (2 * exchange * capacity * weight / (beta * degree) / pi).hi
        for capacity, weight, degree in zip(source.capacity, strength, source.degrees)
    )
    reference_speeds = tuple(
        (exchange * capacity / pi).hi for capacity in source.capacity
    )
    reference_contrasts = tuple(
        abs(source.epi[i] - source.epi[j])
        + horizon * (reference_speeds[i] + reference_speeds[j])
        for i, j in edges
    )
    weighted_errors = [Q(0)] * size
    for (i, j), error, contrast in zip(edges, conductance_errors, reference_contrasts):
        weighted_errors[i] += error * contrast
        weighted_errors[j] += error * contrast
    forcing = tuple(
        (exchange * capacity * error / (beta * degree) / pi).hi
        for capacity, error, degree in zip(
            source.capacity, weighted_errors, source.degrees
        )
    )
    identical = not any(form_errors + phase_errors + conductance_errors)
    common = _coupled_supremum_envelope(
        horizon,
        form_errors,
        phase_errors,
        form_coefficients,
        phase_coefficients,
        form_forcing=(Q(0),) * size,
        phase_forcing=forcing,
        zero_error_by_uniqueness=identical,
    )

    form_storage, phase_storage = Q(0), Q(0)
    for (i, j), weight in zip(edges, upper_weights):
        form_gap = abs(source.epi[i] - source.epi[j]) + form_errors[i] + form_errors[j]
        form_storage += weight * form_gap**2 / 2
        phase_cost = (1 - cos(I(source.phase[j] - source.phase[i]))).hi
        phase_storage += min(Q(2), phase_cost + phase_errors[i] + phase_errors[j])
    common.update(
        initial_form_storage_upper_bound=form_storage,
        initial_phase_storage_upper_bound=beta * phase_storage,
        initial_storage_upper_bound=form_storage + beta * phase_storage,
    )
    changed = {
        "conductance_upper_bounds": upper_weights,
        "reference_form_speed_upper_bounds": reference_speeds,
        "reference_edge_form_contrast_upper_bounds": reference_contrasts,
        "phase_law_forcing_upper_bounds": forcing,
        "max_phase_law_forcing_upper_bound": max(forcing),
    }
    return common, changed


def _coupled_supremum_envelope(
    horizon,
    form_errors,
    phase_errors,
    form_coefficients,
    phase_coefficients,
    *,
    form_forcing,
    phase_forcing,
    zero_error_by_uniqueness=False,
):
    """Solve admitted nonnegative two-channel whole-window inequalities.

    Callers own source/law admission and proof of both coefficient and forcing
    vectors. All vectors contain equally many exact nonnegative rationals and
    the horizon is positive. Forcing denotes a proved differential comparison
    remainder, not an input installed in a physical or runtime evolution law.
    The zero-error exception additionally requires caller-proved coincidence,
    such as same-law uniqueness or a shared exact equilibrium. Zero numerical
    forcing alone does not certify that premise.
    """
    size = len(form_errors)
    form_maximum, phase_maximum = max(form_coefficients), max(phase_coefficients)
    form_forcing_maximum, phase_forcing_maximum = max(form_forcing), max(phase_forcing)
    denominator = 1 - form_maximum * phase_maximum * horizon**2
    initial_form_maximum, initial_phase_maximum = max(form_errors), max(phase_errors)
    if zero_error_by_uniqueness and any(
        form_errors + phase_errors + form_forcing + phase_forcing
    ):
        raise ArithmeticError("zero-error uniqueness requires zero errors and forcing")
    if zero_error_by_uniqueness:
        form_supremum = phase_supremum = Q(0)
        form_bounds = phase_bounds = (Q(0),) * size
    elif denominator > 0:
        form_supremum = (
            initial_form_maximum
            + horizon * form_forcing_maximum
            + form_maximum * horizon * initial_phase_maximum
            + form_maximum * horizon**2 * phase_forcing_maximum
        ) / denominator
        phase_supremum = (
            initial_phase_maximum
            + phase_maximum * horizon * initial_form_maximum
            + phase_maximum * horizon**2 * form_forcing_maximum
            + horizon * phase_forcing_maximum
        ) / denominator
        form_bounds = tuple(
            error + coefficient * horizon * phase_supremum + horizon * forcing_value
            for error, coefficient, forcing_value in zip(
                form_errors, form_coefficients, form_forcing
            )
        )
        phase_bounds = tuple(
            error + coefficient * horizon * form_supremum + horizon * forcing_value
            for error, coefficient, forcing_value in zip(
                phase_errors, phase_coefficients, phase_forcing
            )
        )
    else:
        form_supremum = phase_supremum = None
        form_bounds = phase_bounds = None

    available = zero_error_by_uniqueness or denominator > 0
    return {
        "form_phase_lipschitz_upper_bounds": form_coefficients,
        "phase_form_lipschitz_upper_bounds": phase_coefficients,
        "max_form_phase_lipschitz_upper_bound": form_maximum,
        "max_phase_form_lipschitz_upper_bound": phase_maximum,
        "comparison_denominator": denominator,
        "form_error_supremum_upper_bound": form_supremum,
        "phase_error_supremum_upper_bound": phase_supremum,
        "whole_window_form_error_bounds": form_bounds,
        "whole_window_phase_error_bounds": phase_bounds,
        "status": "enclosed" if available else "unavailable",
        "unavailable_reasons": (
            () if available else ("positive_comparison_denominator_not_certified",)
        ),
        "comparison_method": (
            "zero_initial_error_uniqueness"
            if zero_error_by_uniqueness
            else "coupled_supremum_small_gain"
        ),
    }


def _support_topology(nodes, edges, *, require_connected=True):
    """Admit ordered reciprocal bare support, retaining explicit isolates."""
    if type(nodes) is not tuple or len(nodes) < 2:
        raise ValueError("source nodes must be a complete tuple of at least two nodes")
    positions = {node: index for index, node in enumerate(nodes)}
    if len(positions) != len(nodes):
        raise ValueError("source nodes must be distinct")
    if type(edges) is not tuple or any(
        type(edge) is not tuple
        or len(edge) != 2
        or any(node not in positions for node in edge)
        for edge in edges
    ):
        raise ValueError("source edges must be pairs from its complete node set")
    index_edges = tuple((positions[left], positions[right]) for left, right in edges)
    unordered = tuple((min(i, j), max(i, j)) for i, j in index_edges)
    if any(i == j for i, j in unordered) or len(set(unordered)) != len(unordered):
        raise ValueError("source support must be simple and loop-free")
    neighbors = [[] for _ in nodes]
    for i, j in index_edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    reached, pending = {0}, [0]
    while pending:
        for neighbor in neighbors[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    if require_connected and len(reached) != len(nodes):
        raise ValueError("source bare support must be connected")
    return index_edges, tuple(tuple(row) for row in neighbors)


def _support_values(raw, label, size, *, nonnegative=False):
    raw = _ordered(raw, label, limit=size + 1)
    if len(raw) != size:
        raise ValueError(f"{label} must contain exactly {size} values")
    result = tuple(
        exact_or_represented_real(value, f"{label}[{i}]") for i, value in enumerate(raw)
    )
    if nonnegative and any(value < 0 for value in result):
        raise ValueError(f"{label} must be nonnegative")
    return result


def _support_comparison_from_rows(
    *, reference_model, nodes, edges, conductances, epi, phase, capacity
):
    """Re-admit primitive rows and derive every consumed field without a graph."""
    _, _, beta = _conservative_coefficients(reference_model)
    index_edges, neighbors = _support_topology(nodes, edges)
    conductances = _support_values(
        conductances, "conductances", len(edges), nonnegative=True
    )
    epi = _support_values(epi, "epi", len(nodes))
    phase = _support_values(phase, "phase", len(nodes))
    capacity = _support_values(capacity, "capacity", len(nodes), nonnegative=True)
    weighted_edges = tuple(
        (i, j, weight) for (i, j), weight in zip(index_edges, conductances)
    )
    costs, gradient = _prepared_form_costs(form=epi, weighted_edges=weighted_edges)
    degrees = tuple(map(len, neighbors))
    resultants = tuple(
        (I(*real), I(*imag))
        for real, imag in relative_resultant_bounds(phase, neighbors)
    )
    form_storage = sum(costs, Q(0))
    phase_storage = _sine_phase_storage(phase, index_edges)
    return SineSupportExchangeComparison(
        reference_model=reference_model,
        nodes=nodes,
        edges=edges,
        conductances=conductances,
        epi=epi,
        phase=phase,
        capacity=capacity,
        degrees=degrees,
        form_gradient=gradient,
        relative_resultant=resultants,
        form_storage=form_storage,
        phase_storage=phase_storage,
        storage=form_storage + beta * phase_storage,
        **_sine_work(
            reference_model,
            degrees,
            gradient,
            capacity,
            tuple(imag for _, imag in resultants),
        ),
    )


def _rebuild_sine_support_comparison(source):
    if type(source) is not SineSupportExchangeComparison:
        raise TypeError("a SineSupportExchangeComparison source is required")
    if source.law != _LAW:
        raise ValueError("source must declare conservative sine support exchange")
    return _support_comparison_from_rows(
        reference_model=source.reference_model,
        nodes=source.nodes,
        edges=source.edges,
        conductances=source.conductances,
        epi=source.epi,
        phase=source.phase,
        capacity=source.capacity,
    )


def _capture_support_rows(graph, reference_model, *, require_connected=True):
    """Capture all consumed primitives before any component is evaluated."""
    _conservative_coefficients(reference_model)
    _admit_relational_graph_structure(
        graph, reference_model, require_connected=require_connected
    )
    nodes, epi, phase, capacity = _relational_node_coordinates(graph)
    edges = tuple(graph.edges())
    conductances = []
    for left, right in edges:
        _, weight = finite_represented_real(
            graph.edges[left, right].get("weight", 1.0), "conductance"
        )
        if weight < 0:
            raise ValueError("conductance must be nonnegative")
        conductances.append(weight)
    return dict(
        reference_model=reference_model,
        nodes=nodes,
        edges=edges,
        conductances=tuple(conductances),
        epi=tuple(map(Q, epi)),
        phase=tuple(map(Q, phase)),
        capacity=tuple(map(Q, capacity)),
    )


def bound_relational_sine_support_exchange(graph, *, reference_model):
    """Capture weighted-form, bare-phase conservative exchange without mutation.

    With bare degree d, weighted gradient q=L_W*x and unweighted sine sum S,
    pressure is w*S/(pi*d), form rate is nu*pressure and phase rate is
    w*nu*q/(beta*pi*d). Storage is weighted Dirichlet form plus beta times
    bare-support cosine storage. Both complete rows use the original clock.

    The reference model must explicitly declare regular comparison admission
    and exact zero form loss. Every conductance is admitted before binary64
    materialization; supported zero weights remain bare phase neighbors.
    No native field, stored pressure, cache, inferred derivative or event is
    consumed. The separate unit-support reader retains its rejection contract.
    """
    return _support_comparison_from_rows(
        **_capture_support_rows(graph, reference_model)
    )


@dataclass(frozen=True)
class SineFrozenIsolateComparison:
    """Detached component law with an explicitly supplied frozen-isolate rule.

    Nontrivial components use the connected support law. Isolates retain all
    coordinates and have pressure, form rate and phase rate identically zero.
    Their inverse phase metric is unavailable, not the quotient at zero degree.
    This separate law admits neither implicit native dispatch nor event timing.
    """

    reference_model: RelationalExchangeModel
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    conductances: tuple[Q, ...]
    epi: tuple[Q, ...]
    phase: tuple[Q, ...]
    capacity: tuple[Q, ...]
    component_indices: tuple[tuple[int, ...], ...]
    isolate_indices: tuple[int, ...]
    degrees: tuple[int, ...]
    form_gradient: tuple[Q, ...]
    relative_resultant: tuple[tuple[I, I], ...]
    inverse_phase_metric: tuple[I | None, ...]
    phase_sources: tuple[I, ...]
    pressure: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    form_storage: Q
    phase_storage: I
    storage: I
    dissipation: tuple[Q, ...]
    continuous_loss: Q
    form_work: tuple[I, ...]
    phase_work: tuple[I, ...]
    node_balance_residual: tuple[I, ...]
    storage_rate: I
    balance_residual: I
    law: str = _FROZEN_ISOLATE_LAW
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "explicit_conservative_weighted_sine_law_on_each_nontrivial_component",
        "explicit_frozen_pressure_form_and_phase_rows_on_isolates",
        "fixed_simple_support_finite_nonnegative_conductances_at_least_two_nodes",
        "held_nonnegative_capacity_exact_zero_loss_no_input_or_events",
        "bare_neighbor_degrees_include_supported_zero_conductance_edges",
        "isolate_inverse_phase_metric_unavailable_no_global_positive_mean_metric",
        "edge_storage_has_no_supplied_intrinsic_isolate_energy",
        "ideal_instantaneous_enclosures_at_captured_represented_coordinates",
        "no_native_dispatch_trajectory_event_selector_or_formation_claim",
    )

    def phase_rate_numerators(self) -> tuple[Q, ...]:
        """Rebuild the phase row before pi, including explicitly frozen zeros."""
        source = _rebuild_frozen_isolate_comparison(self)
        _, w, beta = _conservative_coefficients(source.reference_model)
        return tuple(
            w * nu * q / (beta * degree) if degree else Q(0)
            for nu, q, degree in zip(
                source.capacity, source.form_gradient, source.degrees
            )
        )

    def to_dict(self):
        """Rebuild from full primitives, including isolated coordinates."""
        from ..sdk.relational_reports import _project

        source = _rebuild_frozen_isolate_comparison(self)
        _validate_comparison_labels(source)
        return {
            "schema": "tnfr.relational-sine-frozen-isolate-comparison.v1",
            "report": _project(source),
        }


def _frozen_isolate_comparison_from_rows(
    *, reference_model, nodes, edges, conductances, epi, phase, capacity
):
    """Rebuild the supplied component law without trusting cached partitions."""
    _, _, beta = _conservative_coefficients(reference_model)
    index_edges, neighbors = _support_topology(nodes, edges, require_connected=False)
    conductances = _support_values(
        conductances, "conductances", len(edges), nonnegative=True
    )
    epi = _support_values(epi, "epi", len(nodes))
    phase = _support_values(phase, "phase", len(nodes))
    capacity = _support_values(capacity, "capacity", len(nodes), nonnegative=True)
    components = _support_components(neighbors)
    isolates = tuple(row[0] for row in components if len(row) == 1)
    # Empty-neighbor sums vanish; zero rates are the supplied isolate law.
    rows = {
        "form_gradient": [Q(0)] * len(nodes),
        "relative_resultant": [(I(0), I(0))] * len(nodes),
        "inverse_phase_metric": [None] * len(nodes),
        "dissipation": [Q(0)] * len(nodes),
    }
    for name in (
        "phase_sources",
        "pressure",
        "form_rates",
        "phase_rates",
        "form_work",
        "phase_work",
        "node_balance_residual",
    ):
        rows[name] = [I(0)] * len(nodes)
    form_storage, phase_storage = Q(0), I(0)
    for component in components:
        if len(component) == 1:
            continue
        members = set(component)
        edge_indices = tuple(k for k, (i, _) in enumerate(index_edges) if i in members)
        local = _support_comparison_from_rows(
            reference_model=reference_model,
            nodes=tuple(nodes[i] for i in component),
            edges=tuple(edges[k] for k in edge_indices),
            conductances=tuple(conductances[k] for k in edge_indices),
            epi=tuple(epi[i] for i in component),
            phase=tuple(phase[i] for i in component),
            capacity=tuple(capacity[i] for i in component),
        )
        for name, values in rows.items():
            for index, value in zip(component, getattr(local, name)):
                values[index] = value
        form_storage += local.form_storage
        phase_storage += local.phase_storage
    continuous_loss = sum(rows["dissipation"], Q(0))
    storage_rate = sum(rows["form_work"], I(0)) + sum(rows["phase_work"], I(0))
    return SineFrozenIsolateComparison(
        reference_model=reference_model,
        nodes=nodes,
        edges=edges,
        conductances=conductances,
        epi=epi,
        phase=phase,
        capacity=capacity,
        component_indices=components,
        isolate_indices=isolates,
        degrees=tuple(map(len, neighbors)),
        form_storage=form_storage,
        phase_storage=phase_storage,
        storage=form_storage + beta * phase_storage,
        continuous_loss=continuous_loss,
        storage_rate=storage_rate,
        balance_residual=storage_rate + continuous_loss,
        **{name: tuple(values) for name, values in rows.items()},
    )


def _rebuild_frozen_isolate_comparison(source):
    if type(source) is not SineFrozenIsolateComparison:
        raise TypeError("a SineFrozenIsolateComparison source is required")
    if source.law != _FROZEN_ISOLATE_LAW:
        raise ValueError("source must declare conservative sine with frozen isolates")
    return _frozen_isolate_comparison_from_rows(
        reference_model=source.reference_model,
        nodes=source.nodes,
        edges=source.edges,
        conductances=source.conductances,
        epi=source.epi,
        phase=source.phase,
        capacity=source.capacity,
    )


def bound_relational_sine_frozen_isolates(graph, *, reference_model):
    """Capture an explicitly supplied frozen-isolate extension without mutation.

    The original clock and connected weighted support law apply separately to
    each nontrivial component. At isolates pressure, form rate and phase rate
    are explicitly zero while form, phase and capacity remain admitted state.
    The graph must contain at least two nodes. Storage is the sum of edge
    storages, with no supplied intrinsic isolated-node energy or mean weight.

    This detached comparison installs no law or event and never changes the
    admission of the connected support reader. A caller choosing this model
    for an event prefix must supply its refresh callback and event contract.
    """
    return _frozen_isolate_comparison_from_rows(
        **_capture_support_rows(graph, reference_model, require_connected=False)
    )
