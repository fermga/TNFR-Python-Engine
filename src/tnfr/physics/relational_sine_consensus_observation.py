"""Exact regional observations of a weighted sine consensus tangent.

The captured state supplies a complete held law, not an assumed equilibrium.
Its separately declared uniform-form/uniform-phase family has a rational
tangent after factoring the common w/pi from every original-time row. The
shared linear-observation owner determines which fine tangent coordinates
are needed for the two regional relative means; no trajectory is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from ..mathematics.linear_observation import (
    LinearObservation,
    derive_linear_observation,
)
from .relational_observations import _regions
from .relational_sine_comparison import _validate_comparison_labels
from .relational_sine_support import (
    SineFrozenIsolateComparison,
    SineSupportExchangeComparison,
    _conservative_coefficients,
    _coupled_supremum_envelope,
    _rebuild_frozen_isolate_comparison,
    _rebuild_sine_support_comparison,
    _support_topology,
    _support_values,
)

__all__ = (
    "SineConsensusObservation",
    "SineConsensusNonlinearResponse",
    "assess_sine_consensus_observation",
)

Matrix = tuple[tuple[Q, ...], ...]
Source = SineSupportExchangeComparison | SineFrozenIsolateComparison


@dataclass(frozen=True)
class SineConsensusObservation:
    """Minimal all-state linear response of two own-law regional means.

    The full coordinate order is all form increments, then all phase increments
    in declared real radian lifts, following source.nodes. The output is left
    minus right in both channels. ``linear_observation`` uses rational K; its
    reduced generator is multiplied by the same exact w/pi for original time.
    Its selected rows are coordinates, not measured physical-time derivatives.

    ``source_in_consensus_family`` checks exact equality of the normalized
    captured forms and phase lifts. False does not exclude other equilibria.
    The tangent is about the separately declared consensus family regardless
    of this flag; the captured post-event state is never projected or reset.
    """

    source: Source
    regions: tuple[tuple[Any, ...], tuple[Any, ...]]
    region_indices: tuple[tuple[int, ...], tuple[int, ...]]
    mobility: tuple[Q, ...]
    inverse_capacity_weights: tuple[Q, ...]
    region_mean_weights: tuple[Q, Q]
    relative_mean_row: tuple[Q, ...]
    bare_laplacian: Matrix
    weighted_laplacian: Matrix
    rational_generator: Matrix
    exchange_coefficient: Q
    storage_coefficient: Q
    original_time_multiplier_bounds: I
    output_rows: Matrix
    linear_observation: LinearObservation
    source_in_consensus_family: bool
    two_coordinate_closure_certified: bool
    max_rank_calls: int
    exact_identity_checks: tuple[str, ...]
    clock: str = "t"
    generator_rule: str = "J=(exchange_coefficient/pi)*rational_generator"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_held_weighted_sine_source_rebuilt_from_primitive_law_and_state",
        "separately_declared_uniform_form_and_equal_radian_lift_equilibrium_family",
        "captured_state_not_replaced_by_or_assumed_to_be_the_reference_equilibrium",
        "positive_held_capacity_and_no_isolates_for_inverse_mobility_means",
        "nonnegative_conductances_with_bare_supported_zero_weight_edges_retained",
        "left_minus_right_means_normalized_separately_by_actual_degree_over_capacity",
        "full_fine_form_then_radian_phase_tangent_state_in_source_node_order",
        "exact_common_exchange_over_pi_factor_retained_in_full_and_reduced_t_rows",
        "minimal_all_initial_state_linear_realization_by_complete_exact_row_closure",
        "no_special_preparation_or_zero_hidden_initial_state_is_assumed",
        "no_nonlinear_reduction_finite_amplitude_accuracy_or_eigenvalue_claim",
        "no_trajectory_event_occurrence_attraction_formation_or_physical_identity",
    )

    def nonlinear_response_bounds(
        self,
        *,
        duration,
        consensus_form,
        consensus_phase,
        form_amplitude_bounds,
        phase_amplitude_bounds,
    ) -> SineConsensusNonlinearResponse:
        """Bound finite sine response against its same-initial-state tangent.

        Supply the common equilibrium origins and independent node amplitude
        radii. The captured source must lie in this complete fine-state box in
        its actual declared radian lifts. No circular wrapping, projection or
        removal of hidden initial coordinates establishes membership instead.

        Shared coupled-supremum arithmetic first bounds the nonlinear form and
        phase amplitudes. The global inequality |sin(u)-u| <= |u|^3/6 then
        bounds the form-row remainder; the phase row is already exactly linear.
        The same arithmetic propagates this remainder through both channels.
        The full error projects onto both output rows and all selected exact
        observation coordinates. No trajectory or matrix exponential is used.
        """
        return _bound_consensus_nonlinear_response(
            self,
            duration=duration,
            consensus_form=consensus_form,
            consensus_phase=consensus_phase,
            form_amplitude_bounds=form_amplitude_bounds,
            phase_amplitude_bounds=phase_amplitude_bounds,
        )

    def to_dict(self):
        """Rebuild the full tangent and realization before exporting."""
        if type(self) is not SineConsensusObservation:
            raise TypeError("a SineConsensusObservation is required")
        fresh = assess_sine_consensus_observation(
            self.source, regions=self.regions, max_rank_calls=self.max_rank_calls
        )
        _validate_comparison_labels(fresh.source)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-consensus-observation.v1",
            "report": _project(fresh),
        }


@dataclass(frozen=True)
class SineConsensusNonlinearResponse:
    """Whole-box finite validity of a complete consensus tangent observation.

    Each nonlinear trajectory and its linear comparator start from the same
    full fine state in the declared box. Their reduced comparator starts at
    C*z0, including every observable hidden initialization. The retained initial
    coordinates below belong to the captured source, whose box membership is
    mandatory; they do not replace the other members of the independent box.

    Node and edge tuples follow observation.source. The cubic rate remainder
    acts only in form; the reported phase error is its dynamically propagated
    effect. Output and selected-state errors are absolute row projections of
    the full nodal errors, not relative accuracy or observed temporal records.
    """

    observation: SineConsensusObservation
    duration: Q
    consensus_form: Q
    consensus_phase: Q
    form_amplitude_bounds: tuple[Q, ...]
    phase_amplitude_bounds: tuple[Q, ...]
    initial_form_displacements: tuple[Q, ...]
    initial_phase_displacements: tuple[Q, ...]
    initial_observation_state: tuple[Q, ...]
    initial_outputs: tuple[Q, ...]
    form_phase_lipschitz_upper_bounds: tuple[Q, ...]
    phase_form_lipschitz_upper_bounds: tuple[Q, ...]
    max_form_phase_lipschitz_upper_bound: Q
    max_phase_form_lipschitz_upper_bound: Q
    comparison_denominator: Q
    form_amplitude_supremum_upper_bound: Q | None
    phase_amplitude_supremum_upper_bound: Q | None
    whole_window_form_amplitude_bounds: tuple[Q, ...] | None
    whole_window_phase_amplitude_bounds: tuple[Q, ...] | None
    edge_phase_difference_upper_bounds: tuple[Q, ...] | None
    form_cubic_remainder_upper_bounds: tuple[Q, ...] | None
    max_form_cubic_remainder_upper_bound: Q | None
    form_error_supremum_upper_bound: Q | None
    phase_error_supremum_upper_bound: Q | None
    whole_window_form_error_bounds: tuple[Q, ...] | None
    whole_window_phase_error_bounds: tuple[Q, ...] | None
    selected_state_error_upper_bounds: tuple[Q, ...] | None
    output_error_upper_bounds: tuple[Q, ...] | None
    status: str
    unavailable_reasons: tuple[str, ...]
    comparison_method: str
    clock: str = "t"
    phase_units: str = "radians_in_declared_real_lifts"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_source_complete_law_and_exact_observation_realization_before_use",
        "captured_fine_state_must_lie_in_the_supplied_consensus_amplitude_box",
        "declared_common_form_and_real_radian_origins_without_wrapping_or_projection",
        "nonlinear_and_full_tangent_solutions_share_each_complete_initial_state",
        "independent_box_retains_all_fine_including_tangent_invisible_coordinates",
        "nonlinear_amplitude_confinement_precedes_cubic_form_remainder_bound",
        "reciprocal_phase_row_is_exactly_linear_and_propagates_form_remainder",
        "same_original_time_coefficients_in_amplitude_and_error_comparisons",
        "selected_linear_state_initialization_C_z0_retains_hidden_information",
        "absolute_row_projection_bounds_outputs_and_selected_coordinates",
        "exact_zero_amplitude_is_the_common_equilibrium_at_any_positive_duration",
        "unavailable_sufficient_window_does_not_prove_instability_or_model_failure",
        "no_exact_nonlinear_quotient_attraction_event_acquisition_or_trajectory",
        "no_physical_identification_or_universal_complete_NFR_claim",
    )

    def to_dict(self):
        """Re-admit the captured box and rebuild every consumed bound and row."""
        if type(self) is not SineConsensusNonlinearResponse:
            raise TypeError("a SineConsensusNonlinearResponse is required")
        fresh = _bound_consensus_nonlinear_response(
            self.observation,
            duration=self.duration,
            consensus_form=self.consensus_form,
            consensus_phase=self.consensus_phase,
            form_amplitude_bounds=self.form_amplitude_bounds,
            phase_amplitude_bounds=self.phase_amplitude_bounds,
        )
        _validate_comparison_labels(fresh.observation.source)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-consensus-nonlinear-response.v1",
            "report": _project(fresh),
        }


def _bound_consensus_nonlinear_response(
    observation,
    *,
    duration,
    consensus_form,
    consensus_phase,
    form_amplitude_bounds,
    phase_amplitude_bounds,
):
    if type(observation) is not SineConsensusObservation:
        raise TypeError("a SineConsensusObservation is required")
    observation = assess_sine_consensus_observation(
        observation.source,
        regions=observation.regions,
        max_rank_calls=observation.max_rank_calls,
    )
    horizon = exact_or_represented_real(duration, "duration")
    if horizon <= 0:
        raise ValueError("duration must be strictly positive")
    origin_x = exact_or_represented_real(consensus_form, "consensus_form")
    origin_theta = exact_or_represented_real(consensus_phase, "consensus_phase")
    source = observation.source
    size = len(source.nodes)
    form_radii = _support_values(
        form_amplitude_bounds, "form_amplitude_bounds", size, nonnegative=True
    )
    phase_radii = _support_values(
        phase_amplitude_bounds, "phase_amplitude_bounds", size, nonnegative=True
    )
    initial_form = tuple(value - origin_x for value in source.epi)
    initial_phase = tuple(value - origin_theta for value in source.phase)
    if any(abs(value) > radius for value, radius in zip(initial_form, form_radii)):
        raise ValueError("captured source form lies outside the declared amplitude box")
    if any(abs(value) > radius for value, radius in zip(initial_phase, phase_radii)):
        raise ValueError(
            "captured source phase lift lies outside the declared amplitude box"
        )
    initial = tuple((value,) for value in initial_form + initial_phase)
    initial_selected = tuple(
        row[0]
        for row in exact_matrix_product(
            observation.linear_observation.observation, initial
        )
    )
    initial_outputs = tuple(
        row[0] for row in exact_matrix_product(observation.output_rows, initial)
    )
    rate_factor = observation.exchange_coefficient / pi_interval()
    coefficients = tuple(
        (rate_factor * sum((abs(value) for value in row), Q(0))).hi
        for row in observation.rational_generator
    )
    form_coefficients, phase_coefficients = coefficients[:size], coefficients[size:]
    zero = (Q(0),) * size
    zero_amplitude = not any(form_radii + phase_radii)
    amplitude = _coupled_supremum_envelope(
        horizon,
        form_radii,
        phase_radii,
        form_coefficients,
        phase_coefficients,
        form_forcing=zero,
        phase_forcing=zero,
        zero_error_by_uniqueness=zero_amplitude,
    )
    error_form_supremum = error_phase_supremum = None
    error_form = error_phase = edge_gaps = remainders = maximum_remainder = None
    selected_errors = output_errors = None
    if amplitude["status"] == "enclosed":
        phases = amplitude["whole_window_phase_error_bounds"]
        edges, _ = _support_topology(
            source.nodes, source.edges, require_connected=False
        )
        edge_gaps = tuple(phases[i] + phases[j] for i, j in edges)
        sums = [Q(0)] * size
        for (i, j), gap in zip(edges, edge_gaps):
            sums[i] += gap**3
            sums[j] += gap**3
        remainders = tuple(
            (rate_factor * nu * total / (6 * degree)).hi
            for nu, degree, total in zip(source.capacity, source.degrees, sums)
        )
        maximum_remainder = max(remainders)
        errors = _coupled_supremum_envelope(
            horizon,
            zero,
            zero,
            form_coefficients,
            phase_coefficients,
            form_forcing=remainders,
            phase_forcing=zero,
            zero_error_by_uniqueness=zero_amplitude,
        )
        if errors["status"] != "enclosed":
            raise ArithmeticError(
                "the shared amplitude and error window margins disagree"
            )
        error_form_supremum = errors["form_error_supremum_upper_bound"]
        error_phase_supremum = errors["phase_error_supremum_upper_bound"]
        error_form = errors["whole_window_form_error_bounds"]
        error_phase = errors["whole_window_phase_error_bounds"]
        fine_errors = error_form + error_phase

        def project(rows):
            return tuple(
                sum(
                    (abs(value) * error for value, error in zip(row, fine_errors)), Q(0)
                )
                for row in rows
            )

        selected_errors = project(observation.linear_observation.observation)
        output_errors = project(observation.output_rows)
    return SineConsensusNonlinearResponse(
        observation=observation,
        duration=horizon,
        consensus_form=origin_x,
        consensus_phase=origin_theta,
        form_amplitude_bounds=form_radii,
        phase_amplitude_bounds=phase_radii,
        initial_form_displacements=initial_form,
        initial_phase_displacements=initial_phase,
        initial_observation_state=initial_selected,
        initial_outputs=initial_outputs,
        form_phase_lipschitz_upper_bounds=form_coefficients,
        phase_form_lipschitz_upper_bounds=phase_coefficients,
        max_form_phase_lipschitz_upper_bound=amplitude[
            "max_form_phase_lipschitz_upper_bound"
        ],
        max_phase_form_lipschitz_upper_bound=amplitude[
            "max_phase_form_lipschitz_upper_bound"
        ],
        comparison_denominator=amplitude["comparison_denominator"],
        form_amplitude_supremum_upper_bound=amplitude[
            "form_error_supremum_upper_bound"
        ],
        phase_amplitude_supremum_upper_bound=amplitude[
            "phase_error_supremum_upper_bound"
        ],
        whole_window_form_amplitude_bounds=amplitude["whole_window_form_error_bounds"],
        whole_window_phase_amplitude_bounds=amplitude[
            "whole_window_phase_error_bounds"
        ],
        edge_phase_difference_upper_bounds=edge_gaps,
        form_cubic_remainder_upper_bounds=remainders,
        max_form_cubic_remainder_upper_bound=maximum_remainder,
        form_error_supremum_upper_bound=error_form_supremum,
        phase_error_supremum_upper_bound=error_phase_supremum,
        whole_window_form_error_bounds=error_form,
        whole_window_phase_error_bounds=error_phase,
        selected_state_error_upper_bounds=selected_errors,
        output_error_upper_bounds=output_errors,
        status=amplitude["status"],
        unavailable_reasons=amplitude["unavailable_reasons"],
        comparison_method=(
            "zero_amplitude_equilibrium"
            if zero_amplitude
            else "cubic_remainder_coupled_supremum"
        ),
    )


def assess_sine_consensus_observation(
    source, *, regions, max_rank_calls=4096
) -> SineConsensusObservation:
    """Derive a consensus tangent and its exact relative-mean realization.

    The reference law is conservative weighted-form/bare-phase sine exchange.
    Connected support and separately admitted disconnected component support
    are accepted through their existing source owners. Actual isolates and
    zero capacities reject because d/nu does not define the positive regional
    observation used here. Supported zero conductances remain valid and can
    produce additional neutral directions; no stability theorem is inferred.

    Exactly two ordered nonempty regions partition the full node set. At any
    uniform-form, equal-phase-lift equilibrium, let D=diag(nu/d), L the bare
    Laplacian and W the conductance Laplacian. Then J=(w/pi)*K with rational
    K=[[0,-D*L],[D*W/beta,0]]. All primitive scalars retain exact represented or
    rational admission. Factoring the positive common scalar leaves the row
    closure unchanged; no rounded pi enters rank or identity calculations.

    The shared exact rank-call budget raises on exhaustion with no partial or
    approximate realization. It is not a coefficient-size or wall-time bound.
    """
    if type(max_rank_calls) is not int:
        raise TypeError("max_rank_calls must be a positive non-boolean integer")
    if max_rank_calls < 1:
        raise ValueError("max_rank_calls must be positive")
    if type(source) is SineSupportExchangeComparison:
        source = _rebuild_sine_support_comparison(source)
    elif type(source) is SineFrozenIsolateComparison:
        source = _rebuild_frozen_isolate_comparison(source)
    else:
        raise TypeError(
            "an admitted weighted sine support or component source is required"
        )
    if any(value <= 0 for value in source.capacity):
        raise ValueError("regional inverse-mobility means require positive capacity")
    if any(degree == 0 for degree in source.degrees):
        raise ValueError("regional inverse-mobility means require no isolated nodes")
    indices = _regions(source.nodes, regions)
    flattened = tuple(i for region in indices for i in region)
    size = len(source.nodes)
    if (
        len(indices) != 2
        or len(flattened) != size
        or set(flattened) != set(range(size))
    ):
        raise ValueError(
            "two disjoint regions must contain each source node exactly once"
        )
    admitted_regions = tuple(
        tuple(source.nodes[i] for i in region) for region in indices
    )
    edges, _ = _support_topology(source.nodes, source.edges, require_connected=False)
    _, exchange, beta = _conservative_coefficients(source.reference_model)
    mobility = tuple(nu / degree for nu, degree in zip(source.capacity, source.degrees))
    weights = tuple(1 / value for value in mobility)
    region_weights = tuple(
        sum((weights[i] for i in region), Q(0)) for region in indices
    )
    relative = [Q(0)] * size
    for sign, region, total in zip((1, -1), indices, region_weights):
        for i in region:
            relative[i] = sign * weights[i] / total
    relative = tuple(relative)
    bare = [[Q(0)] * size for _ in range(size)]
    weighted = [[Q(0)] * size for _ in range(size)]
    for (i, j), conductance in zip(edges, source.conductances):
        for matrix, value in ((bare, Q(1)), (weighted, conductance)):
            matrix[i][i] += value
            matrix[j][j] += value
            matrix[i][j] -= value
            matrix[j][i] -= value
    bare = tuple(map(tuple, bare))
    weighted = tuple(map(tuple, weighted))
    zero = (Q(0),) * size
    generator = tuple(
        zero + tuple(-mobility[i] * value for value in row)
        for i, row in enumerate(bare)
    ) + tuple(
        tuple(mobility[i] * value / beta for value in row) + zero
        for i, row in enumerate(weighted)
    )
    output = (relative + zero, zero + relative)
    total_weight = sum(weights, Q(0))
    mean = tuple(value / total_weight for value in weights)
    mean_rows = (mean + zero, zero + mean)
    origins = tuple((Q(1), Q(0)) for _ in range(size)) + tuple(
        (Q(0), Q(1)) for _ in range(size)
    )
    hessian = tuple(row + zero for row in weighted) + tuple(
        zero + tuple(beta * value for value in row) for row in bare
    )
    storage_product = exact_matrix_product(hessian, generator)
    checks = {
        "both common origins lie in the full tangent kernel": exact_matrix_product(
            generator, origins
        )
        == ((Q(0), Q(0)),) * (2 * size),
        "both own-law weighted means are conserved": exact_matrix_product(
            mean_rows, generator
        )
        == (zero + zero,) * 2,
        "relative outputs discard both common origins": exact_matrix_product(
            output, origins
        )
        == ((Q(0), Q(0)),) * 2,
        "quadratic storage times the tangent is skew": all(
            storage_product[i][j] == -storage_product[j][i]
            for i in range(2 * size)
            for j in range(2 * size)
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ArithmeticError(f"consensus tangent identities failed: {failed}")
    observation = derive_linear_observation(
        generator, output, max_rank_calls=max_rank_calls
    )
    if observation.output_rank != 2:
        raise ArithmeticError("the two relative regional outputs lost independence")
    return SineConsensusObservation(
        source=source,
        regions=admitted_regions,
        region_indices=indices,
        mobility=mobility,
        inverse_capacity_weights=weights,
        region_mean_weights=region_weights,
        relative_mean_row=relative,
        bare_laplacian=bare,
        weighted_laplacian=weighted,
        rational_generator=generator,
        exchange_coefficient=exchange,
        storage_coefficient=beta,
        original_time_multiplier_bounds=exchange / pi_interval(),
        output_rows=output,
        linear_observation=observation,
        source_in_consensus_family=(
            all(value == source.epi[0] for value in source.epi)
            and all(value == source.phase[0] for value in source.phase)
        ),
        two_coordinate_closure_certified=observation.dimension == 2,
        max_rank_calls=max_rank_calls,
        exact_identity_checks=tuple(checks),
    )
