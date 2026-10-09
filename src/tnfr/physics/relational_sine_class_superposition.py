"""Conditional four-history superposition bounds for an acquired composite.

Two signed donor form events retain their full intervening histories under
the fixed three-C9 law. Symmetry gives a cubic central error bound, not a
positive finite mixed-response assertion. No source or trajectory is acquired.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from .relational_sine_class_mediation import _CONTACTS, _EDGES, _NODES
from .relational_sine_port_composition import _parameters

__all__ = ("SineClassSuperposition", "bound_sine_class_superposition")


@dataclass(frozen=True)
class _ClassProbeHistoryBound:
    """One actual history's conditional state, event work and identity bounds."""

    label: str
    first_probe_amplitude: Q
    second_probe_amplitude: Q
    second_event_time: Q
    total_duration: Q
    cumulative_amplitude_budget: Q
    maximum_form_coordinate_upper_bound: Q
    maximum_phase_deviation_upper_bound: Q
    pre_second_form_coordinate_upper_bound: Q
    pre_second_phase_deviation_upper_bound: Q
    first_probe_work_bounds: I
    second_probe_work_bounds: I
    first_probe_work_upper_bound: Q
    second_probe_work_upper_bound: Q
    first_probe_work_margin: Q
    second_probe_work_margin: Q
    after_first_radius_squared_upper_bound: Q
    after_second_radius_squared_upper_bound: Q
    after_first_excess_storage_upper_bound: Q
    after_second_excess_storage_upper_bound: Q
    after_first_radius_margin: Q
    after_second_radius_margin: Q
    after_first_storage_margin: Q
    after_second_storage_margin: Q
    first_form_mean_shift: Q
    final_form_mean_shift: Q
    final_form_mean_bounds: I
    final_phase_mean_bounds: I
    tangent_endpoint_discrepancy_upper_bound: Q | None
    first_probe_work_within_allowance: bool
    second_probe_work_within_allowance: bool
    work_within_allowances: bool
    first_probe_identity_certified: bool
    second_probe_identity_certified: bool
    identity_certified: bool


def _admit_probe_primitives(mediator_class, raw, maximum_duration):
    """Admit the common two-event primitives before coefficient construction."""
    if type(mediator_class) is not int or mediator_class not in (1, 2):
        raise ValueError("mediator_class must be an ordinary integer one or two")
    values = {key: exact_or_represented_real(value, key) for key, value in raw.items()}
    if any(
        value < 0
        for key, value in values.items()
        if key not in ("first_probe_amplitude", "second_probe_amplitude")
    ):
        raise ValueError(
            "times, radii, readout error and work allowances must be nonnegative"
        )
    if not 0 <= values["delay"] <= values["total_duration"] <= maximum_duration:
        raise ValueError(f"require 0<=delay<=total_duration<={maximum_duration}")
    if not 0 < values["radius"] <= Q(1, 12):
        raise ValueError("radius must lie in (0,1/12]")
    return values


@dataclass(frozen=True)
class _ClassProbeEventLedger:
    contact_work: Q
    initial_radius: Q
    initial_storage: Q
    barrier: Q
    initial_identity: bool
    contact_allowed: bool
    histories: tuple[_ClassProbeHistoryBound, ...]


def _probe_event_ledger(values, g, discrepancies):
    """Rebuild all carried event bounds; supplied discrepancy bounds are optional.

    The norm and storage proof uses heat contraction and is valid whenever
    1-2*g**2*T**2>0. Each caller admits its own proved horizon domain. A None
    discrepancy means no individual tangent comparison is supplied here.
    """
    a, b, s, t, eps, r = (
        values[key]
        for key in (
            "first_probe_amplitude",
            "second_probe_amplitude",
            "delay",
            "total_duration",
            "endpoint_radius",
            "radius",
        )
    )
    contact_work = 8 * eps**2
    initial_radius, initial_storage = 6 * eps**2, 22 * eps**2
    barrier = r**2 / 2700
    initial_identity = initial_radius < r**2 and initial_storage < barrier
    contact_allowed = contact_work <= values["contact_work_allowance"]

    def envelope(amplitude, time):
        form = (amplitude + eps + 2 * g * time * eps) / (1 - 2 * g**2 * time**2)
        phase = eps + 2 * g * time * form
        return form, phase

    histories = []
    for (label, first, second), discrepancy in zip(
        (
            ("neither", Q(0), Q(0)),
            ("first_only", a, Q(0)),
            ("second_only", Q(0), b),
            ("both", a, b),
        ),
        discrepancies,
    ):
        amplitude = abs(first) + abs(second)
        x, y = envelope(amplitude, t)
        first_x, first_y = envelope(abs(first), t)
        pre_x, pre_y = envelope(abs(first), s)
        work1center, work1error = Q(3, 2) * first**2, 6 * abs(first) * eps
        work2center, work2error = Q(3, 2) * second**2, 6 * abs(second) * pre_x
        upper1, upper2 = work1center + work1error, work2center + work2error
        z1, z2 = 27 * (first_x**2 + first_y**2), 27 * (x**2 + y**2)
        e1, e2 = initial_storage + upper1, initial_storage + upper1 + upper2
        identity1 = bool(initial_identity and z1 < r**2 and e1 < barrier)
        identity2 = bool(identity1 and z2 < r**2 and e2 < barrier)
        work1margin = values["first_probe_work_allowance"] - upper1
        work2margin = values["second_probe_work_allowance"] - upper2
        mean_error = Q(4, 58) * eps
        shift = Q(3, 58) * (first + second)
        histories.append(
            _ClassProbeHistoryBound(
                label=label,
                first_probe_amplitude=first,
                second_probe_amplitude=second,
                second_event_time=s,
                total_duration=t,
                cumulative_amplitude_budget=amplitude,
                maximum_form_coordinate_upper_bound=x,
                maximum_phase_deviation_upper_bound=y,
                pre_second_form_coordinate_upper_bound=pre_x,
                pre_second_phase_deviation_upper_bound=pre_y,
                first_probe_work_bounds=I(work1center - work1error, upper1),
                second_probe_work_bounds=I(work2center - work2error, upper2),
                first_probe_work_upper_bound=upper1,
                second_probe_work_upper_bound=upper2,
                first_probe_work_margin=work1margin,
                second_probe_work_margin=work2margin,
                after_first_radius_squared_upper_bound=z1,
                after_second_radius_squared_upper_bound=z2,
                after_first_excess_storage_upper_bound=e1,
                after_second_excess_storage_upper_bound=e2,
                after_first_radius_margin=r**2 - z1,
                after_second_radius_margin=r**2 - z2,
                after_first_storage_margin=barrier - e1,
                after_second_storage_margin=barrier - e2,
                first_form_mean_shift=Q(3, 58) * first,
                final_form_mean_shift=shift,
                final_form_mean_bounds=I(shift - mean_error, shift + mean_error),
                final_phase_mean_bounds=I(-mean_error, mean_error),
                tangent_endpoint_discrepancy_upper_bound=discrepancy,
                first_probe_work_within_allowance=work1margin >= 0,
                second_probe_work_within_allowance=work2margin >= 0,
                work_within_allowances=bool(
                    contact_allowed and work1margin >= 0 and work2margin >= 0
                ),
                first_probe_identity_certified=identity1,
                second_probe_identity_certified=identity2,
                identity_certified=identity2,
            )
        )
    return _ClassProbeEventLedger(
        contact_work,
        initial_radius,
        initial_storage,
        barrier,
        initial_identity,
        contact_allowed,
        tuple(histories),
    )


@dataclass(frozen=True)
class SineClassSuperposition:
    """Finite upper bounds and two distinct observation-error obstructions.

    All four continuations have the same full reached initial state. The
    delayed-only history first follows its own unprobed evolution. Each
    component's form and target-phase Euclidean errors are <=endpoint_radius,
    with exact separate zero sums inherited from the original preparation.
    These are conditional premises, not a formation or acquisition verdict.

    The formal coefficient is order four in a JOINT rescaling of delay and
    total_duration. Its rational signed factor is retained separately from
    gamma**4, even when a tiny numerical interval touches zero. It supplies
    no finite positive response conclusion. Identity/work flags are separate
    from the unconditional-on-those-policies response/error envelopes.
    """

    mediator_class: int
    first_probe_amplitude: Q
    second_probe_amplitude: Q
    delay: Q
    total_duration: Q
    endpoint_radius: Q
    readout_error_bound: Q
    radius: Q
    contact_work_allowance: Q
    first_probe_work_allowance: Q
    second_probe_work_allowance: Q
    classes: tuple[int, ...]
    gamma_bounds: I
    class_cosine_bounds: tuple[I, ...]
    bootstrap_margin: Q
    flow_comparison_margin: Q
    central_cubic_error_coefficient: Q
    formal_first_joint_time_integral: Q
    formal_second_joint_time_integral: Q
    formal_mixed_coefficient_rational_factor: Q
    formal_gamma_fourth_bounds: I
    formal_mixed_coefficient_bounds: I
    formal_mixed_coefficient_sign: int
    baseline_contact_gap_upper_bound: Q
    probe_contact_gap_error_upper_bound: Q
    probe_contact_gap_magnitude_lower_bound: Q
    probe_contact_gap_magnitude_upper_bound: Q
    curvature_cosine_difference_lower_candidate: Q
    oriented_instantaneous_curvature_lower_bound: Q | None
    instantaneous_curvature_orientation: int
    instantaneous_curvature_exact_zero: bool
    instantaneous_curvature_certified: bool
    nominal_mixed_upper_bound: Q
    source_mixed_error_upper_bound: Q
    true_mixed_upper_bound: Q
    readout_mixed_error_upper_bound: Q
    recorded_mixed_bounds: I
    maximum_tangent_endpoint_discrepancy_upper_bound: Q
    exact_mixed_zero: bool
    scalar_cancellation_admitted: bool
    four_record_overlap_admitted: bool
    contact_work_bounds: I
    contact_work_upper_bound: Q
    contact_work_margin: Q
    joined_initial_radius_squared_upper_bound: Q
    joined_initial_excess_storage_upper_bound: Q
    identity_barrier_lower_bound: Q
    joined_initial_identity_certified: bool
    contact_work_within_allowance: bool
    histories: tuple[_ClassProbeHistoryBound, ...]
    all_identities_certified: bool
    all_work_within_allowances: bool
    status: str
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    contacts: tuple[tuple[int, int], ...] = _CONTACTS
    donor_node: int = 4
    receiver_node: int = 22
    history_order: tuple[str, ...] = ("neither", "first_only", "second_only", "both")
    mixed_readout_coefficients: tuple[int, ...] = (1, -1, -1, 1)
    clock: str = "tau=e*t; e=1023/1024"
    formal_joint_time_order: int = 4
    formal_total_amplitude_degree: int = 3
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_full54_state_law_class_support_and_clock_in_all_four_histories",
        "one_common_reached_source_with_component_norm_bounds_and_separate_zero_sums",
        "delayed_single_probe_follows_its_unprobed_history_without_reset",
        "events_are_signed_donor_form_jumps_and_all_phase_coordinates_carry",
        "simultaneous_sign_reversal_is_odd_only_for_the_symmetric_nominal_source",
        "arbitrary_actual_residuals_are_bounded_not_made_symmetric",
        "formal_joint_time_coefficient_is_class_blind_and_not_a_finite_lower_bound",
        "instantaneous_curvature_is_a_right_derivative_after_the_second_event_not_a_measured_readout",
        "curvature_orientation_is_the_sign_of_the_second_probe_and_not_a_finite_endpoint_sign",
        "four_scalar_endpoint_readings_have_total_error_allowance_four_delta",
        "scalar_cancellation_does_not_imply_a_common_four_reading_record",
        "four_record_overlap_uses_same_source_full54_postcontact_tangent_comparator",
        "overlap_is_existential_sensor_error_compatibility_not_continuous_history_equivalence",
        "second_event_work_uses_the_actual_pre_event_history_envelope",
        "contact_and_successive_probe_work_have_separate_allowances",
        "identity_flags_require_strict_radius_and_storage_margins_at_each_event",
        "response_bounds_do_not_require_work_policy_or_identity_flags_to_pass",
        "no_formation_acquisition_solver_response_fit_or_physical_scattering_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-class-superposition.v1", "report": _project(self)}


def bound_sine_class_superposition(
    *,
    mediator_class,
    first_probe_amplitude,
    second_probe_amplitude,
    delay,
    total_duration,
    endpoint_radius,
    readout_error_bound,
    radius,
    contact_work_allowance,
    first_probe_work_allowance,
    second_probe_work_allowance,
) -> SineClassSuperposition:
    """Bound four paired histories from eleven independently admitted primitives.

    mediator_class is an ordinary integer one or two. Probe amplitudes are
    signed finite reals; all other scalars are finite and nonnegative, with
    0<=delay<=total_duration<=1/4 and 0<radius<=1/12. Exact rationals remain
    exact; shared represented-real admission rejects Boolean and nonfinite
    values before model construction. No supplied report or state reset is
    consumed. The endpoint ball and original zero sums remain hypotheses.
    """
    raw = dict(
        first_probe_amplitude=first_probe_amplitude,
        second_probe_amplitude=second_probe_amplitude,
        delay=delay,
        total_duration=total_duration,
        endpoint_radius=endpoint_radius,
        readout_error_bound=readout_error_bound,
        radius=radius,
        contact_work_allowance=contact_work_allowance,
        first_probe_work_allowance=first_probe_work_allowance,
        second_probe_work_allowance=second_probe_work_allowance,
    )
    v = _admit_probe_primitives(mediator_class, raw, Q(1, 4))
    a, b, s, t, eps, delta, r = (
        v[key]
        for key in (
            "first_probe_amplitude",
            "second_probe_amplitude",
            "delay",
            "total_duration",
            "endpoint_radius",
            "readout_error_bound",
            "radius",
        )
    )
    classes = (1, mediator_class, 1)
    gamma, cosines = _parameters(classes)
    g = gamma.hi
    d, comparison = 1 - 2 * g**2 * t**2, 1 - 3 * t
    cubic = (
        Q(8, 3)
        * g**4
        * t**4
        / (d**3 * comparison)
        * (1 + Q(2, 3) * g**2 * t**2 / comparison)
    )
    u = t - s
    integral1 = s**2 * u**2 / 2 + Q(2, 3) * s * u**3 + u**4 / 4
    integral2 = s * u**3 / 3 + u**4 / 4
    factor = (a**2 * b * integral1 + a * b**2 * integral2) / 384
    gamma4 = gamma**4
    baseline_gap = 2 * eps / (1 - 3 * s)
    ideal_form = abs(a) / (1 - 2 * g**2 * s**2)
    gap_error = 4 * g * ideal_form * (1 + 2 * g**2 * s) * s**2 + baseline_gap
    gap_lower = max(Q(0), gamma.lo * abs(a) * s / 4 - gap_error)
    gap_upper = g * abs(a) * s / 4 + gap_error
    cosine_candidate = gap_lower**2 / 3 - baseline_gap**2 / 2
    curvature_zero = a == 0 or b == 0 or s == 0
    curvature = None
    if curvature_zero:
        curvature = Q(0)
    elif gap_upper <= 1 and cosine_candidate > 0:
        curvature = gamma.lo**2 * abs(b) * cosine_candidate / 12
    nominal = cubic * ((abs(a) + abs(b)) ** 3 + abs(a) ** 3 + abs(b) ** 3)
    source_error = 4 * eps / comparison
    exact_zero = a == 0 or b == 0 or s == t
    mixed = Q(0) if exact_zero else nominal + source_error
    noise = 4 * delta
    ledger = _probe_event_ledger(
        v,
        g,
        tuple(
            cubic * amplitude**3 + 2 * eps / comparison
            for amplitude in (Q(0), abs(a), abs(b), abs(a) + abs(b))
        ),
    )
    contact_work, initial_radius, initial_storage = (
        ledger.contact_work,
        ledger.initial_radius,
        ledger.initial_storage,
    )
    barrier, initial_identity, contact_allowed, histories = (
        ledger.barrier,
        ledger.initial_identity,
        ledger.contact_allowed,
        ledger.histories,
    )
    maximum = max(
        history.tangent_endpoint_discrepancy_upper_bound for history in histories
    )
    cancellation, overlap = mixed <= noise, maximum <= delta
    return SineClassSuperposition(
        mediator_class=mediator_class,
        **v,
        classes=classes,
        gamma_bounds=gamma,
        class_cosine_bounds=cosines,
        bootstrap_margin=d,
        flow_comparison_margin=comparison,
        central_cubic_error_coefficient=cubic,
        formal_first_joint_time_integral=integral1,
        formal_second_joint_time_integral=integral2,
        formal_mixed_coefficient_rational_factor=factor,
        formal_gamma_fourth_bounds=gamma4,
        formal_mixed_coefficient_bounds=gamma4 * factor,
        formal_mixed_coefficient_sign=int(factor > 0) - int(factor < 0),
        baseline_contact_gap_upper_bound=baseline_gap,
        probe_contact_gap_error_upper_bound=gap_error,
        probe_contact_gap_magnitude_lower_bound=gap_lower,
        probe_contact_gap_magnitude_upper_bound=gap_upper,
        curvature_cosine_difference_lower_candidate=cosine_candidate,
        oriented_instantaneous_curvature_lower_bound=curvature,
        instantaneous_curvature_orientation=int(b > 0) - int(b < 0),
        instantaneous_curvature_exact_zero=curvature_zero,
        instantaneous_curvature_certified=curvature is not None and curvature > 0,
        nominal_mixed_upper_bound=nominal,
        source_mixed_error_upper_bound=source_error,
        true_mixed_upper_bound=mixed,
        readout_mixed_error_upper_bound=noise,
        recorded_mixed_bounds=I(-mixed - noise, mixed + noise),
        maximum_tangent_endpoint_discrepancy_upper_bound=maximum,
        exact_mixed_zero=exact_zero,
        scalar_cancellation_admitted=cancellation,
        four_record_overlap_admitted=overlap,
        contact_work_bounds=I(0, contact_work),
        contact_work_upper_bound=contact_work,
        contact_work_margin=v["contact_work_allowance"] - contact_work,
        joined_initial_radius_squared_upper_bound=initial_radius,
        joined_initial_excess_storage_upper_bound=initial_storage,
        identity_barrier_lower_bound=barrier,
        joined_initial_identity_certified=initial_identity,
        contact_work_within_allowance=contact_allowed,
        histories=tuple(histories),
        all_identities_certified=all(
            history.identity_certified for history in histories
        ),
        all_work_within_allowances=all(
            history.work_within_allowances for history in histories
        ),
        status=(
            "four_record_overlap"
            if overlap
            else "scalar_cancellation_only" if cancellation else "bounds_only"
        ),
    )
