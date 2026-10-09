"""Conditional nonlinear mixed-response contrast between two mediator classes.

The same rational heat coefficient serves both fixed class configurations.
Common outer-cycle and bridge channels cancel before transcendental enclosure.
This evaluates an analytic coefficient and its error budget, never a nonlinear
trajectory, an acquired source or a previously saved response.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import INTERVAL_METHOD, I
from .relational_sine_class_mediation import _CONTACTS, _EDGES, _NODES
from .relational_sine_class_nonlinear_protocol import (
    _HEAT_ORDER,
    _heat_cubic_channels,
    _heat_truncation_bounds,
    _ideal_heat_remainder,
)
from .relational_sine_class_superposition import (
    _admit_probe_primitives,
    _ClassProbeHistoryBound,
    _probe_event_ledger,
)
from .relational_sine_port_composition import _parameters

__all__ = (
    "SineClassNonlinearOrganization",
    "bound_sine_class_nonlinear_organization",
)


@dataclass(frozen=True)
class SineClassNonlinearOrganization:
    """Bounds for D=M1-M2 with separately admitted sources in the two classes.

    Each class's four histories share its complete reached source; the sources
    of the two classes need not have equal residual errors. Original component
    form/phase error caps and separate zero sums are conditional premises.
    Exact rational endpoint pairs are authoritative; interval fields are outward
    projections. An unresolved sign is a method outcome, not class independence.

    One uniform event ledger applies separately to each class. Equal bounds do
    not assert equal states, work or histories. The optional zero-contrast
    comparator supplies D=0 and its own eight scalar errors; its record-set
    separation requires 16*delta, whereas a recorded contrast sign requires 8*delta.
    """

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
    gamma_bounds: I
    mediator_class_cosine_bounds: tuple[I, I]
    mediator_cosine_difference_bounds: tuple[Q, Q]
    bootstrap_margin: Q
    source_comparison_margin: Q
    heat_cubic_channel_coefficients: tuple[Q, Q, Q]
    mediator_heat_polynomial_coefficient: Q
    heat_uniform_tail_upper_bound: Q
    mediator_heat_truncation_error_upper_bound: Q
    mediator_heat_coefficient_bounds: tuple[Q, Q]
    heat_class_contrast_coefficient_bounds: tuple[Q, Q]
    gamma_fourth_scaled_heat_contrast_bounds: tuple[Q, Q]
    per_history_ideal_remainder_upper_bounds: tuple[Q, ...]
    per_class_mixed_remainder_upper_bound: Q
    ideal_contrast_remainder_upper_bound: Q
    source_contrast_error_upper_bound: Q
    total_true_contrast_error_upper_bound: Q
    readout_contrast_error_upper_bound: Q
    true_contrast_bounds: tuple[Q, Q]
    true_contrast_interval: I
    recorded_contrast_bounds: tuple[Q, Q]
    recorded_contrast_interval: I
    predicted_orientation: int
    oriented_true_contrast_lower_bound: Q | None
    recorded_sign_margin: Q | None
    zero_contrast_comparator_separation_margin: Q | None
    strict_recorded_sign_noise_ceiling: Q | None
    true_contrast_sign_certified: bool
    recorded_contrast_sign_certified: bool
    zero_contrast_record_sets_disjoint: bool
    exact_contrast_zero: bool
    contact_work_bounds: I
    contact_work_upper_bound: Q
    contact_work_margin: Q
    joined_initial_radius_squared_upper_bound: Q
    joined_initial_excess_storage_upper_bound: Q
    identity_barrier_lower_bound: Q
    joined_initial_identity_certified: bool
    contact_work_within_allowance: bool
    uniform_history_bounds: tuple[_ClassProbeHistoryBound, ...]
    all_identities_certified: bool
    all_work_within_allowances: bool
    status: str
    compared_classes: tuple[tuple[int, ...], ...] = ((1, 1, 1), (1, 2, 1))
    class_contrast_coefficients: tuple[int, int] = (1, -1)
    history_order: tuple[str, ...] = ("neither", "first_only", "second_only", "both")
    mixed_readout_coefficients: tuple[int, ...] = (1, -1, -1, 1)
    heat_channel_order: tuple[str, ...] = ("outer_cosine", "mediator_cosine", "bridge")
    canceled_heat_channels: tuple[str, ...] = ("outer_cosine", "bridge")
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    contacts: tuple[tuple[int, int], ...] = _CONTACTS
    donor_node: int = 4
    receiver_node: int = 22
    heat_polynomial_order: int = _HEAT_ORDER
    clock: str = "tau=e*t; e=1023/1024"
    coefficient_method: str = "exact_mediator_heat_polynomial32_contraction_tail_v1"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "contrast_is_class1_mixed_response_minus_class2_mixed_response",
        "same_complete_full54_law_support_capacity_clock_and_events_for_both_classes",
        "four_histories_share_one_actual_source_within_each_class_only",
        "class_source_errors_may_differ_and_have_separate_conditional_component_zero_sums",
        "common_outer_and_bridge_heat_channels_cancel_before_interval_enclosure",
        "restricted_mediator_cubic_operator_uses_actual_full_graph_degrees",
        "analytic_heat_coefficient_is_not_a_nonlinear_flow_or_observation",
        "each_full_law_mixed_remainder_and_each_actual_source_error_are_retained",
        "eight_scalar_reading_errors_give_eight_delta_recorded_contrast_uncertainty",
        "sixteen_delta_compares_with_explicit_zero_true_contrast_and_its_own_eight_errors",
        "failure_of_sufficient_separation_does_not_prove_equality_or_record_overlap",
        "uniform_event_bounds_apply_to_each_class_without_equating_actual_histories",
        "work_and_identity_conditions_are_independent_of_contrast_resolution",
        "no_source_acquisition_cached_verdict_response_fitting_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-nonlinear-organization.v1",
            "report": _project(self),
        }


def bound_sine_class_nonlinear_organization(
    *,
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
) -> SineClassNonlinearOrganization:
    """Bound the fixed class contrast from ten primitives and no response data.

    Amplitudes are signed, 0<=delay<=total_duration<=2, 0<radius<=1/12, and
    remaining scalars are nonnegative. Shared exact-or-represented admission
    precedes coefficient arithmetic. Heat order 32 is a fixed method policy.
    No actual class source is required to equal the nominal proof reference.
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
    # The class comparison is fixed. This owner admits only the common event
    # primitives through the existing helper, not a supplied class selector.
    values = _admit_probe_primitives(1, raw, Q(2))
    a, b, s, t, eps, delta = (
        values[key]
        for key in (
            "first_probe_amplitude",
            "second_probe_amplitude",
            "delay",
            "total_duration",
            "endpoint_radius",
            "readout_error_bound",
        )
    )
    gamma, cosines = _parameters((1, 2, 1))
    g = gamma.hi
    d, ell = 1 - 2 * g**2 * t**2, 1 - 2 * g * t
    difference = cosines[0].lo - cosines[1].hi, cosines[0].hi - cosines[1].lo
    channels = _heat_cubic_channels(a, b, s, t)
    heat_tail, mediator_tail = _heat_truncation_bounds(a, b, s, t)
    mediator_bounds = channels[1] - mediator_tail, channels[1] + mediator_tail
    products = tuple(
        c * coefficient for c in difference for coefficient in mediator_bounds
    )
    coefficient_bounds = min(products), max(products)
    products = tuple(
        value**4 * coefficient
        for value in (gamma.lo, gamma.hi)
        for coefficient in coefficient_bounds
    )
    heat_bounds = min(products), max(products)
    remainders = tuple(
        _ideal_heat_remainder(amplitude, t, g)
        for amplitude in (Q(0), abs(a), abs(b), abs(a) + abs(b))
    )
    per_class_error = sum(remainders, Q(0))
    ideal_error, source_error = 2 * per_class_error, 8 * eps / ell
    error = ideal_error + source_error
    exact_zero = a == 0 or b == 0 or s == t
    true_bounds = (
        (Q(0), Q(0)) if exact_zero else (heat_bounds[0] - error, heat_bounds[1] + error)
    )
    recorded = true_bounds[0] - 8 * delta, true_bounds[1] + 8 * delta
    orientation = 1 if heat_bounds[0] > 0 else -1 if heat_bounds[1] < 0 else 0
    oriented = (
        (true_bounds[0] if orientation > 0 else -true_bounds[1])
        if orientation
        else None
    )
    sign_margin = None if oriented is None else oriented - 8 * delta
    null_margin = None if oriented is None else oriented - 16 * delta
    true_sign = oriented is not None and oriented > 0
    recorded_sign = sign_margin is not None and sign_margin > 0
    null_excluded = null_margin is not None and null_margin > 0
    ledger = _probe_event_ledger(values, g, (None,) * 4)
    return SineClassNonlinearOrganization(
        **values,
        gamma_bounds=gamma,
        mediator_class_cosine_bounds=(cosines[0], cosines[1]),
        mediator_cosine_difference_bounds=difference,
        bootstrap_margin=d,
        source_comparison_margin=ell,
        heat_cubic_channel_coefficients=channels,
        mediator_heat_polynomial_coefficient=channels[1],
        heat_uniform_tail_upper_bound=heat_tail,
        mediator_heat_truncation_error_upper_bound=mediator_tail,
        mediator_heat_coefficient_bounds=mediator_bounds,
        heat_class_contrast_coefficient_bounds=coefficient_bounds,
        gamma_fourth_scaled_heat_contrast_bounds=heat_bounds,
        per_history_ideal_remainder_upper_bounds=remainders,
        per_class_mixed_remainder_upper_bound=per_class_error,
        ideal_contrast_remainder_upper_bound=ideal_error,
        source_contrast_error_upper_bound=source_error,
        total_true_contrast_error_upper_bound=error,
        readout_contrast_error_upper_bound=8 * delta,
        true_contrast_bounds=true_bounds,
        true_contrast_interval=I(*true_bounds),
        recorded_contrast_bounds=recorded,
        recorded_contrast_interval=I(*recorded),
        predicted_orientation=orientation,
        oriented_true_contrast_lower_bound=oriented,
        recorded_sign_margin=sign_margin,
        zero_contrast_comparator_separation_margin=null_margin,
        strict_recorded_sign_noise_ceiling=oriented / 8 if true_sign else None,
        true_contrast_sign_certified=true_sign,
        recorded_contrast_sign_certified=recorded_sign,
        zero_contrast_record_sets_disjoint=null_excluded,
        exact_contrast_zero=exact_zero,
        contact_work_bounds=I(0, ledger.contact_work),
        contact_work_upper_bound=ledger.contact_work,
        contact_work_margin=values["contact_work_allowance"] - ledger.contact_work,
        joined_initial_radius_squared_upper_bound=ledger.initial_radius,
        joined_initial_excess_storage_upper_bound=ledger.initial_storage,
        identity_barrier_lower_bound=ledger.barrier,
        joined_initial_identity_certified=ledger.initial_identity,
        contact_work_within_allowance=ledger.contact_allowed,
        uniform_history_bounds=ledger.histories,
        all_identities_certified=all(h.identity_certified for h in ledger.histories),
        all_work_within_allowances=all(
            h.work_within_allowances for h in ledger.histories
        ),
        status=(
            "zero_contrast_record_sets_disjoint"
            if null_excluded
            else (
                "recorded_sign_certified"
                if recorded_sign
                else (
                    "true_sign_certified"
                    if true_sign
                    else "exact_contrast_zero" if exact_zero else "bounds_only"
                )
            )
        ),
    )
