"""Conditional receiver reflection-odd observation of two mediator classes.

The complete shared amplitude hierarchy supplies the nominal quadratic term.
A class-difference Cauchy remainder and separate actual-source allowance bound
its full-law meaning. The report acquires neither a source nor a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._sine_class_contrast import _contrast_decision, _ContrastDecision
from .relational_sine_class_cubic_response import (
    _CAUCHY_GAMMA,
    _LINEAR_NORM,
    _ORDER,
    _class_cubic_coefficients,
    _CubicParameters,
    _CubicSegment,
    _mixed_form_projection,
    _scale_by_shared_gamma,
)
from .relational_sine_class_mediation import _CONTACTS, _EDGES, _NODES
from .relational_sine_class_superposition import (
    _admit_probe_primitives,
    _ClassProbeHistoryBound,
    _probe_event_ledger,
)
from .relational_sine_port_composition import _parameters

__all__ = ("SineClassSpatialObservation", "bound_sine_class_spatial_observation")

_OBSERVATION = ((23, 1), (21, -1))
_READING_COUNT = 16


def _higher_even_contrast_remainder(amplitude, total):
    """One history's already cross-class, degree-four-and-higher allowance."""
    if amplitude == 0:
        return Q(0)
    g = _CAUCHY_GAMMA
    return Q(4096, 3) * g**7 * amplitude**4 * total**3 / (1 - 4 * g**2 * amplitude**2)


@dataclass(frozen=True)
class SineClassSpatialObservation:
    """A conditional sixteen-reading spatial contrast, with complete errors.

    Each history reads x23 and x21 independently with the supplied per-node
    error. Its four-history mixed difference is then compared between classes.
    Actual source residuals can break nominal reflection symmetry. One uniform
    work/identity ledger applies per class; it does not equate their sources.
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
    cauchy_bootstrap_margin: Q
    cauchy_radius_margin: Q
    cauchy_admitted: bool
    coefficient_evaluated: bool
    class_parameters: tuple[_CubicParameters, ...]
    class_segments: tuple[tuple[_CubicSegment, ...], ...]
    scaled_class_quadratic_bounds: tuple[tuple[Q, Q], ...] | None
    complete_quadratic_contrast_bounds: tuple[Q, Q] | None
    per_history_higher_amplitude_contrast_remainder_upper_bounds: tuple[Q, ...] | None
    higher_amplitude_contrast_error_upper_bound: Q | None
    source_contrast_error_upper_bound: Q
    decision: _ContrastDecision | None
    response_bound_available: bool
    exact_contrast_zero: bool
    uniform_history_bounds: tuple[_ClassProbeHistoryBound, ...]
    joined_initial_identity_certified: bool
    contact_work_within_allowance: bool
    all_identities_certified: bool
    all_work_within_allowances: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    compared_classes: tuple[tuple[int, ...], ...] = ((1, 1, 1), (1, 2, 1))
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    contacts: tuple[tuple[int, int], ...] = _CONTACTS
    observation_node_weights: tuple[tuple[int, int], ...] = _OBSERVATION
    reading_count: int = _READING_COUNT
    coefficient_amplitude_degree: int = 2
    coefficient_scale_gamma_power: int = 3
    time_polynomial_order: int = _ORDER
    linear_majorant_rate: Q = _LINEAR_NORM
    cauchy_gamma_upper_bound: Q = _CAUCHY_GAMMA
    clock: str = "tau=e*t; e=1023/1024"
    method: str = (
        "receiver_odd_quadratic_projection_of_complete_amplitude_time_polynomial64_v1"
    )
    interval_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "two_independent_nodal_readings_per_history_not_a_direct_difference_sensor",
        "x23_minus_x21_then_both_minus_first_minus_second_plus_neither_then_class1_minus_class2",
        "full54_coordinates_and_all_carried_amplitude_levels_are_retained",
        "nominal_reflection_odd_projection_is_even_in_the_common_amplitude_parameter",
        "receiver_quadratic_term_alone_does_not_imply_mediator_sensitivity",
        "class_difference_is_formed_before_shared_gamma_cubed_enclosure",
        "each_higher_amplitude_remainder_already_bounds_the_cross_class_difference",
        "arbitrary_actual_source_errors_are_not_cancelled_by_nominal_parity",
        "sixteen_independent_reading_errors_and_thirty_two_for_a_separately_noisy_null",
        "scalar_cancellation_does_not_imply_full_record_vector_overlap",
        "source_acquisition_identity_work_and_physical_identification_are_separate",
        "shared_full_coordinate_coefficients_can_be_prior_information_not_unseen_evidence",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-class-spatial-observation.v1",
            "report": _project(self),
        }


def bound_sine_class_spatial_observation(
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
) -> SineClassSpatialObservation:
    """Bound the receiver spatial statistic from ten freshly admitted primitives.

    Signed amplitudes, 0<=delay<=total_duration<=2 and 0<radius<=1/12 retain
    the shared event contract. Readout error applies to each separate nodal
    reading. Complex-amplitude failure withholds the full-law bound, except
    for exact event-pairing zeros. No incoming report or saved value is used.
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
    gamma, _ = _parameters((1, 2, 1))
    amplitude = abs(a) + abs(b)
    bootstrap = 1 - 16 * _CAUCHY_GAMMA**2 * t**2
    radius_margin = 1 - 2 * _CAUCHY_GAMMA * amplitude
    admitted = bootstrap > 0 and radius_margin > 0
    exact_zero = a == 0 or b == 0 or s == t
    parameters, segments = [], []
    class_bounds = contrast = remainders = higher_error = decision = None
    source_error = 16 * eps / (1 - 2 * gamma.hi * t)
    if exact_zero:
        class_bounds = ((Q(0), Q(0)),) * 2
        contrast = (Q(0), Q(0))
        decision = _contrast_decision(
            contrast, Q(0), delta, exact_zero=True, reading_count=_READING_COUNT
        )
    elif admitted:
        coefficients = []
        for mediator_class in (1, 2):
            params, history, _ = _class_cubic_coefficients(mediator_class, a, b, s, t)
            parameters.append(params)
            segments.append(history)
            coefficients.append(_mixed_form_projection(history, 1, _OBSERVATION))
        class_bounds = tuple(
            _scale_by_shared_gamma(value, gamma, 3) for value in coefficients
        )
        contrast = _scale_by_shared_gamma(coefficients[0] - coefficients[1], gamma, 3)
        remainders = tuple(
            _higher_even_contrast_remainder(value, t)
            for value in (Q(0), abs(a), abs(b), amplitude)
        )
        higher_error = sum(remainders, Q(0))
        decision = _contrast_decision(
            contrast, higher_error + source_error, delta, reading_count=_READING_COUNT
        )
    ledger = _probe_event_ledger(values, gamma.hi, (None,) * 4)
    return SineClassSpatialObservation(
        **values,
        gamma_bounds=gamma,
        cauchy_bootstrap_margin=bootstrap,
        cauchy_radius_margin=radius_margin,
        cauchy_admitted=admitted,
        coefficient_evaluated=bool(segments),
        class_parameters=tuple(parameters),
        class_segments=tuple(segments),
        scaled_class_quadratic_bounds=class_bounds,
        complete_quadratic_contrast_bounds=contrast,
        per_history_higher_amplitude_contrast_remainder_upper_bounds=remainders,
        higher_amplitude_contrast_error_upper_bound=higher_error,
        source_contrast_error_upper_bound=source_error,
        decision=decision,
        response_bound_available=decision is not None,
        exact_contrast_zero=exact_zero,
        uniform_history_bounds=ledger.histories,
        joined_initial_identity_certified=ledger.initial_identity,
        contact_work_within_allowance=ledger.contact_allowed,
        all_identities_certified=all(h.identity_certified for h in ledger.histories),
        all_work_within_allowances=all(
            h.work_within_allowances for h in ledger.histories
        ),
        status=decision.status if decision is not None else "unavailable",
        unavailable_reasons=(
            () if decision is not None else ("complex_amplitude_domain_not_admitted",)
        ),
    )
