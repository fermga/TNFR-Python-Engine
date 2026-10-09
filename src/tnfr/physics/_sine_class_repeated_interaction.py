"""Exact conditional return and work bounds for repeated joined interaction.

This fixed-model proof calculator consumes a supplied nominal contrast interval.
It neither establishes that interval nor reads an acquired state or a scientific
report. All eight branches retain their own means and relative states. Entry
into the declared return family and compatibility of the mixed means remain
explicit premises; a small relative norm alone cannot establish either.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from ._sine_lyapunov import (
    _sine_lyapunov_coefficients,
    _sine_lyapunov_initial_upper,
    _sine_lyapunov_return_squared,
)
from .phase_cycle_geometry import _derive
from .relational_sine_class_mediation import _EDGES, _NODES
from .relational_sine_class_superposition import (
    _probe_coordinate_envelope,
    _unit_delay_donor_laplacian_bounds,
)

_EPSILON = Q(1, 10**32)
_DELTA = Q(1, 10**30)
_AMPLITUDE = Q(7, 10000)
_DELAY, _TOTAL = Q(1), Q(2)
_GAMMA_UPPER = Q(1, 3000)
_RADIUS = Q(1, 12)
_GAP, _RATE, _COSINE = Q(1, 145), Q(2), Q(1, 20)
_ETA_LOWER, _ETA_UPPER = Q(1, 11000000), Q(1, 9000000)
_DWELL, _DECAY_POWER = Q(1280000000000000), 256
_WORK_ALLOWANCE = Q(1, 500000)
_SIGNS = (1, -1, -1, 1)


@dataclass(frozen=True)
class _RepeatedGeometry:
    """Full support and exact weighted spectral certificates, without folding."""

    degrees: tuple[int, ...]
    laplacian: tuple[tuple[Q, ...], ...]
    centered_degree_metric: tuple[tuple[Q, ...], ...]
    normalized_gap_slack: tuple[tuple[Q, ...], ...]
    normalized_upper_slack: tuple[tuple[Q, ...], ...]
    degree_mass: int
    maximum_degree: int
    diameter: int


def _repeated_geometry() -> _RepeatedGeometry:
    edges = tuple(sorted(tuple(sorted(edge)) for edge in _EDGES))
    geometry = _derive(_NODES, edges)
    incidence = geometry.incidence
    degrees = tuple(sum(value**2 for value in row) for row in incidence)
    laplacian = tuple(
        tuple(Q(sum(x * y for x, y in zip(left, right))) for right in incidence)
        for left in incidence
    )
    mass = sum(degrees)
    centered = tuple(
        tuple(
            Q(degrees[i] * int(i == j)) - Q(degrees[i] * degrees[j], mass)
            for j in _NODES
        )
        for i in _NODES
    )
    lower = tuple(
        tuple(laplacian[i][j] - _GAP * centered[i][j] for j in _NODES) for i in _NODES
    )
    upper = tuple(
        tuple(_RATE * degrees[i] * int(i == j) - laplacian[i][j] for j in _NODES)
        for i in _NODES
    )
    adjacency = tuple(
        tuple(j for j in _NODES if i != j and laplacian[i][j]) for i in _NODES
    )
    diameter = 0
    for source in _NODES:
        distances = {source: 0}
        queue = [source]
        for node in queue:
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        if len(distances) != len(_NODES):
            raise ArithmeticError("fixed joined support must be connected")
        diameter = max(diameter, max(distances.values()))
    if (
        mass != 58
        or max(degrees) != 4
        or diameter != 10
        or _GAP != Q(4, mass * diameter)
        or not exact_symmetric_semidefinite(lower)
        or not exact_symmetric_semidefinite(upper)
    ):
        raise ArithmeticError("fixed joined weighted spectral certificate failed")
    return _RepeatedGeometry(
        degrees, laplacian, centered, lower, upper, mass, 4, diameter
    )


@dataclass(frozen=True)
class _RepeatedHistory:
    """One recurring word's bounds about that branch's carried mean leaf."""

    label: str
    first_amplitude: Q
    second_amplitude: Q
    pre_second_laplacian_bounds: tuple[Q, Q]
    first_work_bounds: tuple[Q, Q]
    second_work_bounds: tuple[Q, Q]
    after_first_radius_squared_upper_bound: Q
    after_second_radius_squared_upper_bound: Q
    after_first_excess_storage_upper_bound: Q
    after_second_excess_storage_upper_bound: Q
    final_form_d_norm_squared_upper_bound: Q
    final_phase_d_norm_squared_upper_bound: Q
    form_mean_increment: Q
    phase_mean_increment: Q
    identity_sufficient: bool
    work_within_allowances: bool
    tangent_endpoint_within_return_budget: bool


def _repeated_histories(geometry: _RepeatedGeometry) -> tuple[_RepeatedHistory, ...]:
    eps, g, a = _EPSILON, _GAMMA_UPPER, _AMPLITUDE
    initial_storage = 2 * eps**2
    barrier = _RADIUS**2 / 2700
    histories = []
    for label, first, second in (
        ("neither", Q(0), Q(0)),
        ("first_only", a, Q(0)),
        ("second_only", Q(0), a),
        ("both", a, a),
    ):
        pressure = _unit_delay_donor_laplacian_bounds(
            first_amplitude=first, endpoint_radius=eps, gamma_upper=g
        )
        first_work = (
            Q(3, 2) * first**2 - 6 * first * eps,
            Q(3, 2) * first**2 + 6 * first * eps,
        )
        second_work = tuple(Q(3, 2) * second**2 + second * p for p in pressure)
        first_x, first_y = _probe_coordinate_envelope(first, _TOTAL, eps, g)
        final_x, final_y = _probe_coordinate_envelope(first + second, _TOTAL, eps, g)
        first_radius = 27 * (first_x**2 + first_y**2)
        final_radius = 27 * (final_x**2 + final_y**2)
        first_storage = initial_storage + first_work[1]
        final_storage = first_storage + second_work[1]
        form_square = geometry.degree_mass * final_x**2
        phase_square = geometry.degree_mass * final_y**2
        histories.append(
            _RepeatedHistory(
                label,
                first,
                second,
                pressure,
                first_work,
                second_work,
                first_radius,
                final_radius,
                first_storage,
                final_storage,
                form_square,
                phase_square,
                Q(geometry.degrees[4], geometry.degree_mass) * (first + second),
                Q(0),
                bool(
                    first_radius < _RADIUS**2
                    and final_radius < _RADIUS**2
                    and first_storage < barrier
                    and final_storage < barrier
                ),
                first_work[1] <= _WORK_ALLOWANCE and second_work[1] <= _WORK_ALLOWANCE,
                max(form_square, phase_square) <= (2 * _RADIUS) ** 2,
            )
        )
    return tuple(histories)


@dataclass(frozen=True)
class _RepeatedInteraction:
    """Sufficient arithmetic, conditional on source, mean and interval premises.

    Certificates do not establish entry into K or mixed-mean compatibility.
    Relative D-norms refer to each branch's own conserved mean; no common form
    or phase offset is erased. The original acquisition is not itself K.
    """

    nominal_contrast_bounds: tuple[Q, Q]
    geometry: _RepeatedGeometry
    histories: tuple[_RepeatedHistory, ...]
    initial_excess_storage_upper_bound: Q
    storage_barrier: Q
    phase_stiffness_lower_bound: Q
    phase_stiffness_upper_bound: Q
    lyapunov_position_lower_coefficient: Q
    lyapunov_position_upper_coefficient: Q
    lyapunov_decay_rate: Q
    lyapunov_initial_upper_bound: Q
    dwell_exponent_margin: Q
    decay_upper_bound: Q
    returned_lyapunov_upper_bound: Q
    returned_form_d_norm_squared_upper_bound: Q
    returned_phase_d_norm_squared_upper_bound: Q
    return_form_squared_margin: Q
    return_phase_squared_margin: Q
    nonlinear_source_error_upper_bound: Q
    tangent_source_error_upper_bound: Q
    true_contrast_bounds: tuple[Q, Q]
    recorded_contrast_bounds: tuple[Q, Q]
    recorded_tangent_bounds: tuple[Q, Q]
    separation_margin: Q
    all_histories_trapped_conditionally: bool
    all_work_within_allowances: bool
    nonlinear_return_sufficient: bool
    tangent_return_sufficient: bool
    separation_sufficient: bool
    conditional_repeatability_sufficient: bool
    unmet_sufficient_requirements: tuple[str, ...]
    status: str
    epsilon: Q = _EPSILON
    readout_error_bound: Q = _DELTA
    first_probe_amplitude: Q = _AMPLITUDE
    second_probe_amplitude: Q = _AMPLITUDE
    delay: Q = _DELAY
    total_duration: Q = _TOTAL
    relaxation_duration: Q = _DWELL
    decay_power: int = _DECAY_POWER
    radius: Q = _RADIUS
    gamma_upper_bound: Q = _GAMMA_UPPER
    eta_lower_bound: Q = _ETA_LOWER
    eta_upper_bound: Q = _ETA_UPPER
    normalized_gap_lower_bound: Q = _GAP
    normalized_rate_upper_bound: Q = _RATE
    cosine_lower_bound: Q = _COSINE
    first_work_allowance: Q = _WORK_ALLOWANCE
    second_work_allowance: Q = _WORK_ALLOWANCE
    recurring_contact_work: Q = Q(0)
    compared_classes: tuple[tuple[int, ...], ...] = ((1, 1, 1), (1, 2, 1))
    mixed_coefficients: tuple[int, ...] = _SIGNS
    clock: str = "tau=e*t; e=1023/1024"
    scope: tuple[str, ...] = (
        "nominal_interval_is_a_supplied_complete_law_premise_not_revalidated_evidence",
        "entry_into_K_requires_each_branch_relative_D_norm_of_form_and_phase_at_most_epsilon",
        "first_acquired_word_and_initial_dwell_establish_entry_separately",
        "each_branch_keeps_its_absolute_form_and_phase_means_without_reset",
        "mixed_mean_compatibility_is_an_independent_required_premise",
        "equal_event_schedule_preserves_existing_mixed_mean_compatibility",
        "work_uses_actual_carried_pressure_and_credits_no_continuous_loss",
        "support_is_already_joined_and_no_contact_is_reapplied_between_words",
        "nonlinear_return_requires_the_certified_acute_trapping_neighborhood",
        "tangent_return_uses_its_own_carried_endpoint_and_global_quadratic_potential",
        "tangent_comparator_keeps_its_own_residual_allowance_and_is_not_reset_to_zero",
        "no_formation_autonomous_selector_reservoir_or_physical_identification_is_inferred",
    )


def _bound_repeated_interaction(
    *, nominal_contrast_lower, nominal_contrast_upper
) -> _RepeatedInteraction:
    """Rebuild fixed conditional arithmetic from two admitted interval endpoints.

    The interval must enclose the nominal full-law class contrast for the fixed
    two-event word. Endpoint admission supplies no evidence for that premise.
    Failure of a sufficient check is not a dynamical impossibility result.
    """
    lower = exact_or_represented_real(nominal_contrast_lower, "nominal_contrast_lower")
    upper = exact_or_represented_real(nominal_contrast_upper, "nominal_contrast_upper")
    if lower > upper:
        raise ValueError("nominal contrast endpoints must be ordered")
    geometry = _repeated_geometry()
    histories = _repeated_histories(geometry)
    mu, maximum, position_lower, position_upper, rate = _sine_lyapunov_coefficients(
        eta_lower=_ETA_LOWER,
        eta_upper=_ETA_UPPER,
        cosine_lower=_COSINE,
        gap_lower=_GAP,
        rate_upper=_RATE,
    )
    initial = _sine_lyapunov_initial_upper(
        eta_upper=_ETA_UPPER,
        rate_upper=_RATE,
        gap_lower=_GAP,
        position_upper=position_upper,
        form_norm_upper=2 * _RADIUS,
        phase_norm_upper=2 * _RADIUS,
    )
    decay = Q(1, 2**_DECAY_POWER)
    energy, form_square, phase_square = _sine_lyapunov_return_squared(
        initial_upper=initial,
        decay_upper=decay,
        eta_lower=_ETA_LOWER,
        gap_lower=_GAP,
        rate_upper=_RATE,
        position_lower=position_lower,
    )
    exponent_margin = rate * _DWELL - _DECAY_POWER
    form_margin = _EPSILON**2 / 4 - form_square
    phase_margin = _EPSILON**2 / 4 - phase_square
    trapped = all(history.identity_sufficient for history in histories)
    work = all(history.work_within_allowances for history in histories)
    returned = exponent_margin >= 0 and min(form_margin, phase_margin) > 0
    nonlinear_return = trapped and returned
    tangent_return = returned and all(
        history.tangent_endpoint_within_return_budget for history in histories
    )
    source = 8 * _EPSILON / (1 - 2 * _GAMMA_UPPER * _TOTAL)
    true = lower - source, upper + source
    recorded = true[0] - 8 * _DELTA, true[1] + 8 * _DELTA
    tangent = -source - 8 * _DELTA, source + 8 * _DELTA
    margin = -upper - 2 * source - 16 * _DELTA
    separation = margin > 0
    unmet = tuple(
        name
        for name, passed in (
            ("acute_trapping_not_certified", trapped),
            ("work_allowance_not_certified", work),
            ("nonlinear_half_radius_return_not_certified", nonlinear_return),
            ("tangent_half_radius_return_not_certified", tangent_return),
            ("strict_recorded_model_separation_not_certified", separation),
        )
        if not passed
    )
    return _RepeatedInteraction(
        nominal_contrast_bounds=(lower, upper),
        geometry=geometry,
        histories=histories,
        initial_excess_storage_upper_bound=2 * _EPSILON**2,
        storage_barrier=_RADIUS**2 / 2700,
        phase_stiffness_lower_bound=mu,
        phase_stiffness_upper_bound=maximum,
        lyapunov_position_lower_coefficient=position_lower,
        lyapunov_position_upper_coefficient=position_upper,
        lyapunov_decay_rate=rate,
        lyapunov_initial_upper_bound=initial,
        dwell_exponent_margin=exponent_margin,
        decay_upper_bound=decay,
        returned_lyapunov_upper_bound=energy,
        returned_form_d_norm_squared_upper_bound=form_square,
        returned_phase_d_norm_squared_upper_bound=phase_square,
        return_form_squared_margin=form_margin,
        return_phase_squared_margin=phase_margin,
        nonlinear_source_error_upper_bound=source,
        tangent_source_error_upper_bound=source,
        true_contrast_bounds=true,
        recorded_contrast_bounds=recorded,
        recorded_tangent_bounds=tangent,
        separation_margin=margin,
        all_histories_trapped_conditionally=trapped,
        all_work_within_allowances=work,
        nonlinear_return_sufficient=nonlinear_return,
        tangent_return_sufficient=tangent_return,
        separation_sufficient=separation,
        conditional_repeatability_sufficient=not unmet,
        unmet_sufficient_requirements=unmet,
        status="conditional_repeatability_sufficient" if not unmet else "bounds_only",
    )
