"""A fixed, finite-precision interface for a held-sine interaction family.

Encoding reads an already available realized original state. Decoding retains
only two grid indices and fixed protocol metadata; it does not recover or
consume the discarded realization. The finite forecast is conditional on the
declared source, law, support, clock and coupling interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import _prepared_exponential_series_tail_upper
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ._sine_contact_information import _admit_contact_support_information
from ._sine_history_coupling_information import (
    _COUPLING_POLYNOMIAL_DEGREE,
    _prepared_coupling_channel_rows,
    _prepared_divided_matrix_powers,
)
from ._sine_joint_information import _prepared_sine_tube
from ._sine_preparation import _prepared_coordinate_vector
from .relational_observations import _ordered

_INTERFACE_DELTA = Q(1, 10000)
_INTERFACE_EPSILON = Q(1, 100000000)
_INTERFACE_QUANTUM = 2 * _INTERFACE_EPSILON
_INTERFACE_PREPARATION_TIME = Q(1, 4)
_INTERFACE_HORIZON = Q(1, 4)
_INTERFACE_GAMMA_BOUNDS = (Q(49, 100), Q(51, 100))
_INTERFACE_TOLERANCE = Q(5, 1000000000)
_INTERFACE_COORDINATE_BOUNDS = (
    (Q(0), Q(0)),
    (-_INTERFACE_EPSILON, _INTERFACE_EPSILON),
    (_INTERFACE_DELTA, 3 * _INTERFACE_DELTA),
    (-_INTERFACE_DELTA, _INTERFACE_DELTA),
    (Q(0), Q(0)),
    (-_INTERFACE_EPSILON, _INTERFACE_EPSILON),
    (-_INTERFACE_EPSILON, _INTERFACE_EPSILON),
    (-_INTERFACE_EPSILON, _INTERFACE_EPSILON),
)


@dataclass(frozen=True)
class _CompiledSineHistoryInterface:
    """Immutable fixed-metadata calculation; no realized source is stored."""

    coefficient_tensor: tuple
    discarded_state_error: Q
    quantization_error: Q
    coupling_error: Q
    series_tail_error: Q
    nonlinear_error: Q
    series_norm_step: Q
    series_tail: Q
    mobility_degree: Q
    phase_growth: Q
    phase_tube: Q


@dataclass(frozen=True)
class _PreparedSineHistoryInterface:
    """One packet query with a bound uniform over the entire protocol family.

    coefficient_tensor[i][j][r] is the exact coefficient of h**j * e**r
    for original coordinate i, where e=gamma-1/2. Coordinates are ordered
    (x0,x1,x2,x3,phi0,phi1,phi2,phi3). All preparation degrees have already
    been combined. The point prediction uses r=0 and the reconstructed
    original form. Reconstruction is a reference, not the reached state.

    The five uniform error components hold for every legal source and its
    encoded packet, every time and every coupling for this contact. More
    generally, the packet must reconstruct x2,x3 within epsilon of that
    source. A valid packet alone does not authenticate this association.
    At h=0 the actual matched response
    and query allowance are exactly zero; the uniform components still
    describe the whole time family. No original realization, source-dependent
    interval or affine generator is stored in this report.
    """

    packet: tuple[int, int]
    contact: tuple[int, int]
    elapsed_time: Q
    reconstructed_form: tuple[Q, ...]
    reconstructed_phase: tuple[Q, ...]
    coefficient_tensor: tuple
    uniform_discarded_state_error_upper_bound: Q
    uniform_quantization_error_upper_bound: Q
    uniform_coupling_error_upper_bound: Q
    uniform_series_tail_error_upper_bound: Q
    uniform_nonlinear_error_upper_bound: Q
    uniform_error_upper_bound: Q
    series_norm_step: Q
    series_tail_upper_bound: Q
    mobility_degree: Q
    phase_growth_bound: Q
    phase_tube_upper_bound: Q
    predicted_response: Q
    forecast_error_upper_bound: Q
    forecast_response_bounds: tuple[Q, Q]
    status: str
    response_sign: str
    polynomial_degree: int = _COUPLING_POLYNOMIAL_DEGREE
    preparation_time: Q = _INTERFACE_PREPARATION_TIME
    horizon: Q = _INTERFACE_HORIZON
    gamma_bounds: tuple[Q, Q] = _INTERFACE_GAMMA_BOUNDS
    reference_gamma: Q = Q(1, 2)
    quantum: Q = _INTERFACE_QUANTUM
    quantization_error_bound: Q = _INTERFACE_EPSILON
    tolerance: Q = _INTERFACE_TOLERANCE
    coordinate_bounds: tuple = _INTERFACE_COORDINATE_BOUNDS
    encoded_coordinate_labels: tuple = ((2, "form"), (3, "form"))
    projection_rank: int = 2
    payload_scalar_count: int = 2
    payload_bits: int = 28
    measurement_cost: int = 0
    evolved_coordinate_count: int = 0
    scope: tuple[str, ...] = (
        "already_available_realized_original_state_in_fixed_six_coordinate_box",
        "two_quantized_original_form_coordinates_without_new_physical_readings",
        "certificate_requires_packet_source_association_within_epsilon",
        "decoder_consumes_only_packet_query_and_fixed_protocol_metadata",
        "unit_support12_23_preparation_then_unit_contact01_or03",
        "original_phase_origin_and_complete_reached_state_are_not_reset",
        "one_unknown_held_gamma_common_to_both_stages_and_both_branches",
        "arithmetic_mean_of_contact_pair_on_minus_matched_off_response",
        "every_response_time_between_zero_and_one_quarter",
        "critical_degree16_reference_is_not_the_actual_reached_state",
        "signed_preparation_contraction_before_uniform_absolute_bounds",
        "discarded_state_quantization_coupling_series_and_nonlinear_errors_separate",
        "fixed_support_metadata_and_offline_full_support_preprocessing_not_payload",
        "rank2_coordinate_interface_not_exact_closure_or_minimal_rank_claim",
        "no_new_information_acquisition_trajectory_or_computational_speedup_claim",
    )


def _encode_sine_history_interface(*, form, relative_phase) -> tuple[int, int]:
    """Validate the complete original source and encode x2,x3 in 28 bits.

    Both input vectors follow node order (0,1,2,3). Their exact-or-represented
    values must lie in the fixed independent source box, including x0=phi0=0.
    Each code is in 0..10000. Nearest rounding has upward ties and absolute
    error at most epsilon. The encoder needs access to the realized source;
    admission of its uncertainty class alone does not supply this access.
    """
    x = _prepared_coordinate_vector(form, "form", 4)
    phase = _prepared_coordinate_vector(relative_phase, "relative_phase", 4)
    if any(
        not lower <= value <= upper
        for value, (lower, upper) in zip(x + phase, _INTERFACE_COORDINATE_BOUNDS)
    ):
        raise ValueError("original state must belong to the fixed interface source")
    offsets = (_INTERFACE_DELTA, -_INTERFACE_DELTA)
    return tuple(
        int((value - offset) / _INTERFACE_QUANTUM + Q(1, 2))
        for value, offset in zip(x[2:], offsets)
    )


@lru_cache(maxsize=2)
def _compiled_sine_history_interface(contact):
    """Compile one fixed protocol contact after raw query admission.

    This internal cache accepts no external matrices, reports, budgets or
    source-dependent data. Its two possible keys are canonical contacts.
    Every stored array is an immutable tuple of exact rational coefficients.
    """
    prepared = _admit_contact_support_information(
        nodes=(0, 1, 2, 3),
        old_edges=((1, 2), (2, 3)),
        old_conductances=(Q(1), Q(1)),
        capacity=(Q(1),) * 4,
        contact=contact,
        contact_weight=Q(1),
        retained_nodes=contact,
        observation_nodes=contact,
        elapsed_time=_INTERFACE_HORIZON,
        mean_kind="arithmetic",
        reading_error_bound=Q(0),
        tolerance=_INTERFACE_TOLERANCE,
    )
    degree = _COUPLING_POLYNOMIAL_DEGREE
    duration = _INTERFACE_PREPARATION_TIME + _INTERFACE_HORIZON
    gamma_upper = _INTERFACE_GAMMA_BOUNDS[1]
    mobility = 4 * max(
        -matrix[i][i]
        for matrix in (prepared.off_generator, prepared.on_generator)
        for i in range(4)
    )
    source_maxima = tuple(
        max(abs(lo), abs(hi)) for lo, hi in _INTERFACE_COORDINATE_BOUNDS
    )
    form_ceiling, phase_ceiling = max(source_maxima[:4]), max(source_maxima[4:])
    tube = _prepared_sine_tube(
        duration, mobility, form_ceiling, phase_ceiling, gamma_upper
    )
    norm_step = (1 + gamma_upper) * mobility * duration / 2
    if norm_step > 1:
        raise ValueError("interface tangent series requires norm step at most one")

    # Powers in response duration one are coefficients of h**j. Prefix
    # powers already include tau**m. No commutation between supports is used.
    prefix = _prepared_divided_matrix_powers(
        prepared.off_generator, _INTERFACE_PREPARATION_TIME, degree
    )
    off = _prepared_divided_matrix_powers(prepared.off_generator, Q(1), degree)
    on = _prepared_divided_matrix_powers(prepared.on_generator, Q(1), degree)
    channel_rows = _prepared_coupling_channel_rows(degree)
    tensor = [[[Q(0)] * (degree + 1) for _ in range(degree + 1)] for _ in range(8)]
    for response_degree in range(1, degree + 1):
        difference = tuple(
            tuple(a - b for a, b in zip(on_row, off_row))
            for on_row, off_row in zip(on[response_degree], off[response_degree])
        )
        matched_row = exact_matrix_product((prepared.observation_weights,), difference)[
            0
        ]
        for prefix_degree in range(degree + 1 - response_degree):
            row = exact_matrix_product((matched_row,), prefix[prefix_degree])[0]
            form_row, phase_row = channel_rows[response_degree + prefix_degree]
            for node, spatial in enumerate(row):
                for power, (a, b) in enumerate(zip(form_row, phase_row)):
                    tensor[node][response_degree][power] += spatial * a
                    tensor[node + 4][response_degree][power] += spatial * b
    tensor = tuple(tuple(tuple(row) for row in coordinate) for coordinate in tensor)

    horizon = _INTERFACE_HORIZON
    deviation = _INTERFACE_GAMMA_BOUNDS[1] - Q(1, 2)
    critical_row_bounds = tuple(
        sum((abs(row[0]) * horizon**j for j, row in enumerate(coordinate)), Q(0))
        for coordinate in tensor
    )
    discarded = _INTERFACE_EPSILON * sum(
        (critical_row_bounds[i] for i in (1, 5, 6, 7)), Q(0)
    )
    quantization = _INTERFACE_EPSILON * sum(
        (critical_row_bounds[i] for i in (2, 3)), Q(0)
    )
    coupling = sum(
        (
            maximum * horizon**j * deviation**r * abs(value)
            for maximum, coordinate in zip(source_maxima, tensor)
            for j, row in enumerate(coordinate)
            for r, value in enumerate(row)
            if r
        ),
        Q(0),
    )
    tail = _prepared_exponential_series_tail_upper(norm_step, degree)
    return _CompiledSineHistoryInterface(
        coefficient_tensor=tensor,
        discarded_state_error=discarded,
        quantization_error=quantization,
        coupling_error=coupling,
        series_tail_error=2 * max(source_maxima) * tail,
        nonlinear_error=2 * tube.form_error,
        series_norm_step=norm_step,
        series_tail=tail,
        mobility_degree=mobility,
        phase_growth=tube.growth,
        phase_tube=tube.phase_tube,
    )


def _decode_sine_history_interface(
    packet, *, contact, elapsed_time
) -> _PreparedSineHistoryInterface:
    """Forecast one family query using only a two-integer packet.

    contact must be (0,1) or (0,3), with the stated order; elapsed_time lies
    in [0,1/4]. No realized state, physical observation, source interval,
    affine preparation or externally compiled report is an input. All raw
    query admission precedes cached or fresh polynomial work, also at h=0.
    The certificate is conditional on this packet being the encoding of the
    claimed source, or satisfying the same two epsilon error bounds. Without
    the source, the decoder cannot authenticate that association.

    Uniform error bounds use P_gamma(z)-P_half(q) =
    [P_gamma(z)-P_half(z)] + P_half(z-q). The first term retains actual
    source maxima; the second uses the four discarded radii and two rounding
    errors. One actual matched Taylor tail and the complete general-law
    nonlinear allowance finish the bound. Insufficiently small upper bounds
    yield an unresolved result, not an impossibility claim.
    """
    packet = _ordered(packet, "packet", limit=3)
    if len(packet) != 2 or any(type(value) is not int for value in packet):
        raise TypeError("packet must contain exactly two non-boolean integers")
    if any(not 0 <= value <= 10000 for value in packet):
        raise ValueError("packet indices must lie in 0..10000")
    contact = _ordered(contact, "contact", limit=3)
    if (
        len(contact) != 2
        or any(type(value) is not int for value in contact)
        or contact not in ((0, 1), (0, 3))
    ):
        raise ValueError("contact must be exactly (0,1) or (0,3)")
    time = exact_or_represented_real(elapsed_time, "elapsed_time")
    if not 0 <= time <= _INTERFACE_HORIZON:
        raise ValueError("elapsed_time must lie in [0,1/4]")
    compiled = _compiled_sine_history_interface(contact)
    reconstructed = (
        Q(0),
        Q(0),
        _INTERFACE_DELTA + _INTERFACE_QUANTUM * packet[0],
        -_INTERFACE_DELTA + _INTERFACE_QUANTUM * packet[1],
    )
    prediction = sum(
        (
            reconstructed[i] * time**j * row[0]
            for i in (2, 3)
            for j, row in enumerate(compiled.coefficient_tensor[i])
        ),
        Q(0),
    )
    uniform_error = (
        compiled.discarded_state_error
        + compiled.quantization_error
        + compiled.coupling_error
        + compiled.series_tail_error
        + compiled.nonlinear_error
    )
    error = uniform_error if time else Q(0)
    bounds = prediction - error, prediction + error
    return _PreparedSineHistoryInterface(
        packet=packet,
        contact=contact,
        elapsed_time=time,
        reconstructed_form=reconstructed,
        reconstructed_phase=(Q(0),) * 4,
        coefficient_tensor=compiled.coefficient_tensor,
        uniform_discarded_state_error_upper_bound=compiled.discarded_state_error,
        uniform_quantization_error_upper_bound=compiled.quantization_error,
        uniform_coupling_error_upper_bound=compiled.coupling_error,
        uniform_series_tail_error_upper_bound=compiled.series_tail_error,
        uniform_nonlinear_error_upper_bound=compiled.nonlinear_error,
        uniform_error_upper_bound=uniform_error,
        series_norm_step=compiled.series_norm_step,
        series_tail_upper_bound=compiled.series_tail,
        mobility_degree=compiled.mobility_degree,
        phase_growth_bound=compiled.phase_growth,
        phase_tube_upper_bound=compiled.phase_tube,
        predicted_response=prediction,
        forecast_error_upper_bound=error,
        forecast_response_bounds=bounds,
        status="certified" if error <= _INTERFACE_TOLERANCE else "unresolved",
        response_sign=(
            "positive"
            if bounds[0] > 0
            else (
                "negative"
                if bounds[1] < 0
                else "zero" if bounds == (0, 0) else "unresolved"
            )
        ),
    )
