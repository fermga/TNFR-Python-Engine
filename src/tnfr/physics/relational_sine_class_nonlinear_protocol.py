"""Finite nonlinear separation from an exact heat-cubic coefficient.

This conditional proof evaluates a finite heat polynomial, not a nonlinear
trajectory. Complete-law remainders, original source uncertainty and four
reading errors remain separate from the supplied event and identity policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial

from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from .relational_sine_class_mediation import _CONTACTS, _EDGES, _NODES
from .relational_sine_class_memory import _normalized_laplacian
from .relational_sine_class_superposition import (
    _admit_probe_primitives,
    _ClassProbeHistoryBound,
    _probe_event_ledger,
)
from .relational_sine_port_composition import _parameters

__all__ = ("SineClassNonlinearProtocol", "bound_sine_class_nonlinear_protocol")

_HEAT_ORDER = 32


def _multiply(left, right):
    result = [Q(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a:
            for j, b in enumerate(right):
                if b:
                    result[i + j] += a * b
    return tuple(result)


def _affine_compose(coefficients, constant, slope):
    """Exact coefficients of p(constant+slope*u), not sampled interpolation."""
    return tuple(
        sum(
            (
                coefficients[n] * comb(n, j) * constant ** (n - j) * slope**j
                for n in range(j, len(coefficients))
            ),
            Q(0),
        )
        for j in range(len(coefficients))
    )


@lru_cache(maxsize=1)
def _heat_geometry():
    """Rebuild the fixed Markov generator and its two exact coefficient rows."""
    degrees = tuple(sum(node in edge for edge in _EDGES) for node in _NODES)
    matrix = _normalized_laplacian(_EDGES, degrees)
    if (
        max(sum(map(abs, row), Q(0)) for row in matrix) != 2
        or any(sum(row, Q(0)) != 0 for row in matrix)
        or any(matrix[i][j] > 0 for i in _NODES for j in _NODES if i != j)
    ):
        raise ArithmeticError("fixed heat contraction geometry failed")

    def vectors(operator, node):
        terms = [tuple(Q(i == node) for i in _NODES)]
        for n in range(1, _HEAT_ORDER + 1):
            product = exact_matrix_product(operator, tuple((x,) for x in terms[-1]))
            terms.append(tuple(-row[0] / n for row in product))
        return tuple(tuple(row[i] for row in terms) for i in _NODES)

    transpose = tuple(tuple(matrix[j][i] for j in _NODES) for i in _NODES)
    return degrees, vectors(matrix, 4), vectors(transpose, 22)


def _heat_cubic_channels(a, b, delay, total):
    """Integrate three rational cosine channels after exact history cancellation."""
    if a == 0 or b == 0 or delay == total:
        return (Q(0),) * 3
    degrees, column, row = _heat_geometry()
    first, second, observation = [], [], []
    duration = total - delay
    for i in _NODES:
        shifted = _affine_compose(column[i], delay, Q(1))
        first.append(
            tuple(a * (Q(i == 4 and j == 0) - v) for j, v in enumerate(shifted))
        )
        second.append(
            tuple(b * (Q(i == 4 and j == 0) - v) for j, v in enumerate(column[i]))
        )
        observation.append(_affine_compose(row[i], duration, Q(-1)))
    channels = [Q(0), Q(0), Q(0)]
    for edge_index, (left, right) in enumerate(_EDGES):
        p = tuple(y - x for x, y in zip(first[left], first[right]))
        q = tuple(y - x for x, y in zip(second[left], second[right]))
        mixed = _multiply(_multiply(p, q), tuple(x + y for x, y in zip(p, q)))
        transport = tuple(
            x / degrees[left] - y / degrees[right]
            for x, y in zip(observation[left], observation[right])
        )
        integrand = _multiply(transport, mixed)
        integral = -sum(
            (
                coefficient * duration ** (power + 1) / (2 * (power + 1))
                for power, coefficient in enumerate(integrand)
            ),
            Q(0),
        )
        channel = 2 if edge_index >= 27 else 1 if 9 <= edge_index < 18 else 0
        channels[channel] += integral
    return tuple(channels)


def _ideal_heat_remainder(amplitude, total, g):
    """Parity-preserving full-law remainder with all gamma powers retained."""
    d, ell = 1 - 2 * g**2 * total**2, 1 - 2 * g * total
    return (
        g**6 * amplitude**3 * total**3 / d**3 * (Q(8, 3) + Q(32, 9) / ell)
        + Q(4, 15) * g**6 * amplitude**5 * total / d**5
        + Q(8, 15) * g**8 * amplitude**3 * total**5 / (d**3 * ell**2)
    )


def _heat_truncation_bounds(a, b, delay, total):
    """Shared heat-polynomial and mixed cubic contraction error bounds.

    Restricting the cubic operator to an edge subset retains this bound when
    its selected incident edges use the original full-support degrees.
    """
    heat_tail = (2 * total) ** (_HEAT_ORDER + 1) / factorial(_HEAT_ORDER + 1)
    cubic_tail = (
        4
        * (abs(a) ** 2 * abs(b) + abs(a) * abs(b) ** 2)
        * (total - delay)
        * ((1 + heat_tail) ** 4 - 1)
    )
    return heat_tail, cubic_tail


@dataclass(frozen=True)
class SineClassNonlinearProtocol:
    """A finite conditional protocol bound, without source or flow acquisition.

    Histories share one complete actual source, with the original component
    form/phase Euclidean error caps and separate zero sums. Exact rational
    endpoint pairs are authoritative; dyadic intervals are outward displays.
    Separation of the nonlinear and tangent four-record sets needs a strict
    margin above EIGHT reading-error bounds, not merely a recorded sign.
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
    source_comparison_margin: Q
    heat_cubic_channel_coefficients: tuple[Q, Q, Q]
    heat_polynomial_coefficient_bounds: tuple[Q, Q]
    heat_uniform_tail_upper_bound: Q
    heat_coefficient_truncation_error_upper_bound: Q
    heat_cubic_coefficient_bounds: tuple[Q, Q]
    gamma_fourth_scaled_heat_bounds: tuple[Q, Q]
    per_history_ideal_remainder_upper_bounds: tuple[Q, ...]
    ideal_mixed_remainder_upper_bound: Q
    source_mixed_error_upper_bound: Q
    true_mixed_bounds: tuple[Q, Q]
    true_mixed_interval: I
    recorded_mixed_bounds: tuple[Q, Q]
    recorded_mixed_interval: I
    predicted_orientation: int
    oriented_true_mixed_lower_bound: Q | None
    recorded_sign_margin: Q | None
    disjoint_record_margin: Q | None
    strict_disjoint_noise_ceiling: Q | None
    true_mixed_sign_certified: bool
    recorded_mixed_sign_certified: bool
    four_record_sets_disjoint: bool
    exact_mixed_zero: bool
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
    heat_channel_order: tuple[str, ...] = ("outer_cosine", "mediator_cosine", "bridge")
    heat_polynomial_order: int = _HEAT_ORDER
    clock: str = "tau=e*t; e=1023/1024"
    coefficient_method: str = "exact_rational_heat_polynomial32_contraction_tail_v1"
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_full54_actual_source_law_support_clock_and_class_in_four_histories",
        "component_form_and_phase_error_caps_and_separate_zero_sums_are_conditional_premises",
        "second_events_retain_all_intervening_nodal_and_hidden_coordinates",
        "heat_polynomial_is_an_analytic_coefficient_not_a_nonlinear_trajectory_solver",
        "rational_history_cancellation_precedes_transcendental_coefficient_enclosure",
        "finite_gamma_six_remainder_retains_internal_reflection_odd_modes",
        "exact_rational_endpoint_pairs_precede_optional_dyadic_interval_projection",
        "finite_response_orientation_need_not_match_the_formal_small_time_coefficient",
        "four_reading_recorded_sign_uses_four_delta_and_two_model_record_separation_eight_delta",
        "tangent_comparator_is_the_full54_postcontact_law_with_same_actual_initial_state",
        "individual_history_tangent_discrepancy_is_not_supplied_by_this_report",
        "work_and_strict_identity_flags_are_independent_of_observation_separation",
        "no_source_acquisition_reserved_response_fit_or_physical_scattering_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-nonlinear-protocol.v1",
            "report": _project(self),
        }


def bound_sine_class_nonlinear_protocol(
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
) -> SineClassNonlinearProtocol:
    """Bound a declared protocol from eleven primitives, without evaluated data.

    The class is ordinary integer one or two. Amplitudes are signed; remaining
    scalars are nonnegative, with 0<=delay<=total_duration<=2 and 0<radius<=1/12.
    Every scalar uses shared exact-or-represented admission before arithmetic.
    Heat order32 is fixed. A wide enclosure or failed sufficient separation
    does not imply linearity, absence of response or overlap of record sets.
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
    v = _admit_probe_primitives(mediator_class, raw, Q(2))
    a, b, s, t, eps, delta = (
        v[key]
        for key in (
            "first_probe_amplitude",
            "second_probe_amplitude",
            "delay",
            "total_duration",
            "endpoint_radius",
            "readout_error_bound",
        )
    )
    classes = (1, mediator_class, 1)
    gamma, cosines = _parameters(classes)
    g = gamma.hi
    d, ell = 1 - 2 * g**2 * t**2, 1 - 2 * g * t
    channels = _heat_cubic_channels(a, b, s, t)
    lower = upper = channels[2]
    for coefficient, interval in zip(channels[:2], cosines[:2]):
        products = (coefficient * interval.lo, coefficient * interval.hi)
        lower += min(products)
        upper += max(products)
    heat_tail, cubic_tail = _heat_truncation_bounds(a, b, s, t)
    coefficient_bounds = (lower - cubic_tail, upper + cubic_tail)
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
    ideal_error, source_error = sum(remainders, Q(0)), 4 * eps / ell
    exact_zero = a == 0 or b == 0 or s == t
    true_bounds = (
        (Q(0), Q(0))
        if exact_zero
        else (
            heat_bounds[0] - ideal_error - source_error,
            heat_bounds[1] + ideal_error + source_error,
        )
    )
    recorded = true_bounds[0] - 4 * delta, true_bounds[1] + 4 * delta
    orientation = 1 if heat_bounds[0] > 0 else -1 if heat_bounds[1] < 0 else 0
    oriented = (
        (true_bounds[0] if orientation > 0 else -true_bounds[1])
        if orientation
        else None
    )
    sign_margin = None if oriented is None else oriented - 4 * delta
    separation_margin = None if oriented is None else oriented - 8 * delta
    true_sign = oriented is not None and oriented > 0
    recorded_sign = sign_margin is not None and sign_margin > 0
    disjoint = separation_margin is not None and separation_margin > 0
    ledger = _probe_event_ledger(v, g, (None,) * 4)
    return SineClassNonlinearProtocol(
        mediator_class=mediator_class,
        **v,
        classes=classes,
        gamma_bounds=gamma,
        class_cosine_bounds=cosines,
        bootstrap_margin=d,
        source_comparison_margin=ell,
        heat_cubic_channel_coefficients=channels,
        heat_polynomial_coefficient_bounds=(lower, upper),
        heat_uniform_tail_upper_bound=heat_tail,
        heat_coefficient_truncation_error_upper_bound=cubic_tail,
        heat_cubic_coefficient_bounds=coefficient_bounds,
        gamma_fourth_scaled_heat_bounds=heat_bounds,
        per_history_ideal_remainder_upper_bounds=remainders,
        ideal_mixed_remainder_upper_bound=ideal_error,
        source_mixed_error_upper_bound=source_error,
        true_mixed_bounds=true_bounds,
        true_mixed_interval=I(*true_bounds),
        recorded_mixed_bounds=recorded,
        recorded_mixed_interval=I(*recorded),
        predicted_orientation=orientation,
        oriented_true_mixed_lower_bound=oriented,
        recorded_sign_margin=sign_margin,
        disjoint_record_margin=separation_margin,
        strict_disjoint_noise_ceiling=oriented / 8 if true_sign else None,
        true_mixed_sign_certified=true_sign,
        recorded_mixed_sign_certified=recorded_sign,
        four_record_sets_disjoint=disjoint,
        exact_mixed_zero=exact_zero,
        contact_work_bounds=I(0, ledger.contact_work),
        contact_work_upper_bound=ledger.contact_work,
        contact_work_margin=v["contact_work_allowance"] - ledger.contact_work,
        joined_initial_radius_squared_upper_bound=ledger.initial_radius,
        joined_initial_excess_storage_upper_bound=ledger.initial_storage,
        identity_barrier_lower_bound=ledger.barrier,
        joined_initial_identity_certified=ledger.initial_identity,
        contact_work_within_allowance=ledger.contact_allowed,
        histories=ledger.histories,
        all_identities_certified=all(h.identity_certified for h in ledger.histories),
        all_work_within_allowances=all(
            h.work_within_allowances for h in ledger.histories
        ),
        status=(
            "record_sets_disjoint"
            if disjoint
            else (
                "recorded_sign_certified"
                if recorded_sign
                else (
                    "true_sign_certified"
                    if true_sign
                    else "exact_mixed_zero" if exact_zero else "bounds_only"
                )
            )
        ),
    )
