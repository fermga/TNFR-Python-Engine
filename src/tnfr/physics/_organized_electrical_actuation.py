"""Finite uniform-child current port with a live quarter-sine phase row.

The proof and prospective decision belong to ORGANIZED_ELECTRICAL_ACTUATION.md.
An explicit clock waveform supplies forcing without observing the actual state.
Its offset is a proof device, not a prescribed physical trajectory or a phase
jump. These detached calculations run no numerical flow or observation word.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, pi_interval
from ._organized_electrical_source import (
    _HiddenDeviationBounds,
    _integer_labels,
    _same_undirected_edges,
    _transport_hidden_deviations,
)
from ._organized_outer_winding import (
    _derive_organized_outer_resource_bounds,
    _OrganizedOuterResourceBounds,
)
from ._sine_recovery_resources import (
    _assess_sine_recovery_resources,
    _SineRecoveryResources,
)
from .relational_observations import _ordered


def _gamma():
    return Q(1, 1023) / pi_interval()


def _over_gamma(value, gamma):
    ends = (value / gamma.lo, value / gamma.hi)
    return I(min(ends), max(ends))


def _duration(value):
    normalized = _exact(value, "duration")
    if normalized <= 0:
        raise ValueError("duration must be strictly positive")
    return normalized


def _source_boxes(values, label):
    """Re-admit all five primitive enclosure coordinates before reduction."""
    raw = _ordered(values, label, limit=6)
    if len(raw) != 5:
        raise ValueError(f"{label} requires exactly five ordered intervals")
    result = []
    for index, box in enumerate(raw):
        if type(box) is not I:
            raise TypeError(f"{label}[{index}] requires a shared rational interval")
        result.append(
            I(
                _exact(box.lo, f"{label}[{index}].lo"),
                _exact(box.hi, f"{label}[{index}].hi"),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class _OrganizedElectricalPulse:
    elapsed: Q
    duration: Q
    amplitude: Q
    marked_child: int
    gamma_bounds: I
    direction: tuple[Q, ...]
    inverse_direction: tuple[Q, ...]
    phase_offset_bounds: tuple[I, ...]
    form_offset_bounds: tuple[I, ...]
    form_input_bounds: tuple[I, ...]
    phase_tau_rate_bounds: tuple[I, ...]
    form_tau_rate_bounds: tuple[I, ...]
    clock: str = "tau=(1023/1024)*t; elapsed and duration in tau units"
    scope: tuple[str, ...] = (
        "clock_and_marked_child_waveform_without_actual_state_or_feedback",
        "five_child_C5_coordinates_repeated_uniformly_on_each_C9",
        "offset_derivatives_are_not_the_complete_actual_nodal_or_phase_rates",
        "form_input_is_offset_derivative_plus_one_quarter_C5_laplacian_offset",
        "both_actual_quarter_sine_rows_remain_live",
        "finite_endpoint_input_switches_with_no_state_jump",
    )


def _evaluate_organized_electrical_pulse(*, elapsed, duration, amplitude, marked_child):
    """Enclose a fixed open-loop waveform during its closed pulse interval.

        The support is the precontact C9-by-C5 host. With r=elapsed/duration,
        s=3*r**2-2*r**3 and k=e_marked-1/5, the auxiliary offset is
        z=4*amplitude*L5^+*k*s_r'/(gamma*duration). The supplied input is
        f=z'+L5*z/4, uniform across each child's nine fine nodes. Actual phases
        are not prescribed: their departure from the command remains in the proof.
    The supplied input is zero outside the declared interval; finite endpoint
    switches in input do not jump either state coordinate. The phase command
    retains its final displacement in the continued comparison.
    """
    time = _exact(elapsed, "elapsed")
    length = _duration(duration)
    value = _exact(amplitude, "amplitude")
    if not 0 <= time <= length:
        raise ValueError("elapsed requires 0 <= elapsed <= duration")
    if type(marked_child) is not int or marked_child not in range(5):
        raise ValueError("marked_child must be an ordinary integer in 0..4")
    r = time / length
    profile, first, second = 3 * r**2 - 2 * r**3, 6 * r - 6 * r**2, 6 - 12 * r
    direction = tuple(Q(int(c == marked_child)) - Q(1, 5) for c in range(5))
    column = (Q(2, 5), Q(0), Q(-1, 5), Q(-1, 5), Q(0))
    inverse = tuple(column[(c - marked_child) % 5] for c in range(5))
    gamma = _gamma()
    offset = tuple(_over_gamma(4 * value * v * first / length, gamma) for v in inverse)
    rate = tuple(
        _over_gamma(4 * value * v * second / length**2, gamma) for v in inverse
    )
    forcing = tuple(
        _over_gamma(
            4 * value * v * second / length**2 + value * k * first / length,
            gamma,
        )
        for v, k in zip(inverse, direction)
    )
    return _OrganizedElectricalPulse(
        elapsed=time,
        duration=length,
        amplitude=value,
        marked_child=marked_child,
        gamma_bounds=gamma,
        direction=direction,
        inverse_direction=inverse,
        phase_offset_bounds=tuple(I(value * k * profile) for k in direction),
        form_offset_bounds=offset,
        form_input_bounds=forcing,
        phase_tau_rate_bounds=tuple(I(value * k * first / length) for k in direction),
        form_tau_rate_bounds=rate,
    )


@dataclass(frozen=True)
class _OrganizedElectricalPulseBounds:
    initial_form_bound: Q
    initial_hidden_form: Q
    initial_hidden_phase: Q
    outer_gap: Q
    amplitude: Q
    duration: Q
    gamma_bounds: I
    hidden: _HiddenDeviationBounds
    residual_form_upper: Q
    phase_residual_upper: Q
    endpoint_form_upper: Q
    form_change_upper: Q
    form_offset_upper: Q
    form_gradient_upper: Q
    input_upper: Q
    peak_storage_upper: Q
    continuous_loss_upper: Q
    absolute_work_upper: Q
    internal_positive_margin: Q
    internal_acute_margin: Q
    collar_margin: Q
    outer_acute_margin: Q
    unforced_form_comparison_upper: Q
    unforced_phase_comparison_upper: Q
    actual_phase_contrast_lower: Q
    continuous_loss_lower: Q
    peak_form_storage_lower: Q
    scope: tuple[str, ...] = (
        "primitive_source_bounds_require_independent_history_admission",
        "full_45_node_quarter_sine_law_on_90_unit_edges_with_held_unit_capacity",
        "identical_uniform_child_input_cancels_from_hidden_projection",
        "actual_total_phase_increment_differs_from_command_by_bounded_residual",
        "unforced_comparison_is_a_proof_device_not_an_observed_reset_branch",
        "signed_hidden_storage_is_not_an_available_work_reservoir",
        "absolute_supplied_work_bounds_positive_supplied_work_separately_from_loss",
        "no_instantaneous_word_or_reading_inverse_transfers_to_this_pulse",
    )


def _bound_organized_electrical_pulse(
    *,
    initial_form_bound,
    initial_hidden_form,
    initial_hidden_phase,
    outer_gap,
    amplitude,
    duration,
):
    """Bound one finite pulse from primitive fine and collective allowances.

    initial_form_bound is the actual fine infinity bound, hidden allowances
    are full-fine Euclidean norms, and outer_gap bounds the actual entry mean
    phase gaps with their retained lifts. A positive identity margin is a
    sufficient conditional check, not source authentication or future recovery.
    """
    initial = tuple(
        _exact(value, label)
        for label, value in (
            ("initial_form_bound", initial_form_bound),
            ("initial_hidden_form", initial_hidden_form),
            ("initial_hidden_phase", initial_hidden_phase),
            ("outer_gap", outer_gap),
        )
    )
    if min(initial) < 0:
        raise ValueError("source norm and gap bounds must be nonnegative")
    m0, r0, z0, gap = initial
    value = _exact(amplitude, "amplitude")
    length = _duration(duration)
    pi, gamma = pi_interval(), _gamma()
    g = gamma.hi
    hidden = _transport_hidden_deviations(
        initial_form=r0,
        initial_phase=z0,
        gamma_upper=g,
        laplacian_upper=8,
        duration=length,
    )
    magnitude = abs(value)
    residual = m0 + g * length
    phase = 2 * g * residual * length
    offset = 12 * magnitude / (5 * gamma.lo * length)
    gradient = 24 * magnitude / (5 * gamma.lo * length) + 8 * residual
    forcing = 48 * magnitude / (5 * gamma.lo * length**2) + 6 * magnitude / (
        5 * gamma.lo * length
    )
    comparison = g * magnitude * length / (4 * hidden.domain_margin)
    contrast = max(Q(0), magnitude - 2 * phase)
    return _OrganizedElectricalPulseBounds(
        initial_form_bound=m0,
        initial_hidden_form=r0,
        initial_hidden_phase=z0,
        outer_gap=gap,
        amplitude=value,
        duration=length,
        gamma_bounds=gamma,
        hidden=hidden,
        residual_form_upper=residual,
        phase_residual_upper=phase,
        endpoint_form_upper=residual,
        form_change_upper=(2 * residual + g) * length,
        form_offset_upper=offset,
        form_gradient_upper=gradient,
        input_upper=forcing,
        peak_storage_upper=180 * (offset + residual) ** 2 + 180,
        continuous_loss_upper=45 * length * gradient**2 / 4,
        absolute_work_upper=45 * length * gradient * forcing,
        internal_positive_margin=2 * pi.lo / 9 - 2 * hidden.phase_upper,
        internal_acute_margin=pi.lo / 2 - 2 * pi.hi / 9 - 2 * hidden.phase_upper,
        collar_margin=pi.lo / 2 - 2 * hidden.phase_upper,
        outer_acute_margin=pi.lo / 2 - gap - magnitude - 2 * z0 - 2 * phase,
        unforced_form_comparison_upper=comparison,
        unforced_phase_comparison_upper=hidden.coupling * comparison,
        actual_phase_contrast_lower=contrast,
        continuous_loss_lower=18 * contrast**2 / (g**2 * length),
        peak_form_storage_lower=12 * contrast**2 / (g**2 * length**2),
    )


@dataclass(frozen=True)
class _OrganizedElectricalActuation:
    recovery: _SineRecoveryResources
    outer: _OrganizedOuterResourceBounds
    prefix_hidden: _HiddenDeviationBounds
    pulse: _OrganizedElectricalPulseBounds
    nominal_form_upper: Q
    entry_form_upper: Q
    entry_mean_gap_upper: Q
    prerequisite_flags: tuple[tuple[str, bool], ...]
    prerequisite_margins: tuple[tuple[str, Q], ...]
    sufficient_margins: tuple[tuple[str, Q], ...]
    admitted: bool
    unresolved_conditions: tuple[str, ...]
    marked_child: int = 0
    amplitude: Q = Q(1, 10)
    duration: Q = Q(1)
    prefix_duration: Q = Q(1, 10**5)
    independent_child_count: int = 5
    source_families: tuple[str, ...] = ("outer_zero", "outer_one")
    phase_residual_budget: Q = Q(1, 500000)
    endpoint_form_budget: Q = Q(1, 500)
    form_change_budget: Q = Q(1, 250)
    peak_storage_budget: Q = Q(2 * 10**8)
    continuous_loss_budget: Q = Q(10**9)
    absolute_work_budget: Q = Q(3 * 10**8)
    scope: tuple[str, ...] = (
        "one_fresh_uniform_recovery_theorem_applied_to_five_independent_children",
        "actual_precontact_source_after_first_unforced_observation_window",
        "first_marked_positive_phase_command_replaced_by_one_finite_form_input",
        "no_actual_source_copy_reset_or_initial_state_measurement_for_actuation",
        "new_explicit_peak_loss_and_work_budgets_not_the_old_jump_budget",
        "no_remaining_observation_word_inverse_or_support_switching_admission",
        "one_analytic_assessment_no_trajectory_or_physical_experiment",
    )


def _assess_organized_electrical_actuation() -> _OrganizedElectricalActuation:
    """Assess only the committed pulse and source box, once after freeze."""
    radius, prefix = Q(1, 10**8), Q(1, 10**5)
    recovery = _assess_sine_recovery_resources(
        form_error_bound=Q(1, 10**6),
        phase_error_bound=Q(1, 10**6),
        quarter_dwell=Q(8 * 10**10),
        endpoint_radius=radius,
        decay_power=80,
    )
    outer = _derive_organized_outer_resource_bounds(recovered_radius=radius)
    geometry, gamma = outer.geometry, _gamma()
    g = gamma.hi
    forms = _source_boxes(outer.primitive_entry.endpoint_form_bounds, "nominal_form")
    gaps = _source_boxes(
        outer.primitive_entry.capture.phase_edge_gap_bounds, "nominal_gap"
    )
    nominal = max(box.abs_max for box in forms)
    r0 = _exact(outer.full_form_error_upper_bound, "full_form_error_upper_bound")
    z0 = _exact(outer.full_phase_error_upper_bound, "full_phase_error_upper_bound")
    mean_gap = max(box.abs_max for box in gaps)
    if min(nominal, r0, z0, mean_gap) < 0:
        raise ValueError("source bounds require nonnegative scalars")
    initial = nominal + r0 + g * prefix
    prefix_hidden = _transport_hidden_deviations(
        initial_form=r0,
        initial_phase=z0,
        gamma_upper=g,
        laplacian_upper=8,
        duration=prefix,
    )
    entry_gap = mean_gap + 2 * z0 + 4 * g * initial * prefix
    pulse = _bound_organized_electrical_pulse(
        initial_form_bound=initial,
        initial_hidden_form=prefix_hidden.form_upper,
        initial_hidden_phase=prefix_hidden.phase_upper,
        outer_gap=entry_gap,
        amplitude=Q(1, 10),
        duration=Q(1),
    )
    internal = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9, Q(1)) for c in range(5) for j in range(9)
    )
    external = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j, Q(1)) for c in range(5) for j in range(9)
    )
    capacities = tuple(
        _exact(value, f"capacity[{index}]")
        for index, value in enumerate(geometry.held_capacities)
    )
    flags = (
        ("five_independent_child_recoveries", recovery.recovery_certified is True),
        (
            "recovered_radius_matches",
            _exact(recovery.endpoint_radius, "recovery_radius") == radius
            and _exact(outer.recovered_child_norm_bound, "outer_radius") == radius,
        ),
        ("primitive_law_and_clock", outer.primitive_law_and_clock_certified is True),
        ("full_fine_comparison", outer.fine_error_certified is True),
        ("child_transit_identity", outer.child_winding_retained_during_transit is True),
        ("outer_one_cell", outer.outer_cell_certified is True),
        ("outer_one_capture", outer.conditional_capture_certified is True),
        ("outer_zero_capture", outer.control_conditional_capture_certified is True),
        ("outer_attachment_budget", outer.attachment_budget_certified is True),
        (
            "precontact_product_support",
            _same_undirected_edges(
                geometry.weighted_edges, internal + external, "host_edges"
            )
            and _integer_labels(geometry.node_order, tuple(range(45)))
            and _integer_labels(geometry.support_degrees, (4,) * 45),
        ),
        (
            "complete_quarter_law",
            capacities == (Q(1),) * 45
            and _exact(geometry.clock_factor, "clock_factor") == Q(1, 4)
            and geometry.phase_domain == "regular"
            and geometry.gamma_bounds == gamma
            and geometry.alpha_bounds == 2 * pi_interval() / 9,
        ),
    )
    margins = tuple(
        (label, _exact(value, label))
        for label, value in (
            ("actual_recovery_form", recovery.endpoint_form_margin),
            ("actual_recovery_phase", recovery.endpoint_phase_margin),
            ("outer_one_capture", outer.capture_margin),
            ("outer_zero_capture", outer.control_capture_margin),
            ("outer_attachment_work", outer.attachment_work_margin),
            ("outer_initial_storage", outer.initial_storage_margin),
        )
        if value is not None
    )
    flags += (("complete_numeric_source_margins", len(margins) == 6),)
    decisions = (
        ("phase_residual", Q(1, 500000) - pulse.phase_residual_upper),
        ("endpoint_form", Q(1, 500) - pulse.endpoint_form_upper),
        ("form_change", Q(1, 250) - pulse.form_change_upper),
        ("internal_positive", pulse.internal_positive_margin),
        ("internal_acute", pulse.internal_acute_margin),
        ("circular_collar", pulse.collar_margin),
        ("outer_acute", pulse.outer_acute_margin),
        ("peak_storage", Q(2 * 10**8) - pulse.peak_storage_upper),
        ("continuous_loss", Q(10**9) - pulse.continuous_loss_upper),
        ("absolute_supplied_work", Q(3 * 10**8) - pulse.absolute_work_upper),
    )
    unresolved = tuple(name for name, ok in flags if not ok) + tuple(
        name for name, value in margins + decisions if value <= 0
    )
    return _OrganizedElectricalActuation(
        recovery=recovery,
        outer=outer,
        prefix_hidden=prefix_hidden,
        pulse=pulse,
        nominal_form_upper=nominal,
        entry_form_upper=initial,
        entry_mean_gap_upper=entry_gap,
        prerequisite_flags=flags,
        prerequisite_margins=margins,
        sufficient_margins=decisions,
        admitted=not unresolved,
        unresolved_conditions=unresolved,
    )
