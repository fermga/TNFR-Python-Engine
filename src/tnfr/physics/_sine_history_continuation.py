"""A capped history record for a second held-sine interaction.

The original packet is retained together with one prescribed contact choice.
This is a finite continuation forecast with a declared history, not an update
of an autonomous reduced state. No original or reached realization is read.
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
from ._sine_history_interface import (
    _INTERFACE_COORDINATE_BOUNDS,
    _INTERFACE_DELTA,
    _INTERFACE_EPSILON,
    _INTERFACE_GAMMA_BOUNDS,
    _INTERFACE_PREPARATION_TIME,
    _INTERFACE_QUANTUM,
    _INTERFACE_TOLERANCE,
    _CompiledSineHistoryInterface,
    _PreparedSineHistoryInterface,
)
from ._sine_joint_information import _prepared_sine_tube
from .relational_observations import _ordered

_CONTINUATION_CONTACT_TIME = Q(1, 8)
_CONTINUATION_HORIZON = Q(1, 32)


@dataclass(frozen=True, kw_only=True)
class _PreparedSineHistoryContinuation(_PreparedSineHistoryInterface):
    """A source-associated packet plus exactly one contact-history bit.

    The inherited tensor and five uniform error components now describe
    the complete three-stage history and the receiver-only x0 excess. The
    first contact is removed without resetting any coordinate before the
    second on/off probe. The inherited packet field is the two-index
    component of history_record, not an additional retained measurement.

    The record must describe the actual prescribed history of the original
    source matched to its packet. Integer syntax cannot authenticate either
    association. Original and current states are absent from decoder inputs.
    The reference reconstruction remains an original-state approximation;
    it is never installed as the physical state at either support change.
    """

    history_record: tuple[int, int, int]
    first_contact: tuple[int, int]
    first_contact_duration: Q = _CONTINUATION_CONTACT_TIME
    horizon: Q = _CONTINUATION_HORIZON
    observation_nodes: tuple[int, ...] = (0,)
    observation_weights: tuple[Q, ...] = (Q(1), Q(0), Q(0), Q(0))
    payload_scalar_count: int = 3
    payload_bits: int = 29
    initial_payload_bits: int = 28
    history_payload_bits: int = 1
    recorded_contact_count: int = 1
    max_recorded_contacts: int = 1
    scope: tuple[str, ...] = (
        "original_two_index_packet_plus_one_fixed_duration_contact_choice_bit",
        "packet_source_and_record_actual_history_association_are_premises",
        "decoder_consumes_only_record_query_and_fixed_protocol_metadata",
        "old_support12_23_preparation_then_prescribed_contact01_or03",
        "first_contact_removed_state_continuously_before_second_probe",
        "second_probe_compares_receiver_x0_against_old_support_off_branch",
        "same_unknown_held_gamma_through_every_stage_and_both_final_branches",
        "complete_reached_form_phase_and_original_phase_origin_are_not_reset",
        "critical_degree16_ordered_three_stage_reference_polynomial",
        "prefix_degrees_combined_before_uniform_time_and_coupling_bounds",
        "discarded_state_quantization_coupling_series_and_nonlinear_errors_separate",
        "exactly_one_history_slot_no_unbounded_log_or_repeated_record_update",
        "fixed_metadata_and_full_support_preprocessing_are_additional_costs",
        "finite_prescribed_history_continuation_not_autonomous_reduced_state",
        "no_source_readback_new_measurement_or_event_authentication",
        "no_minimum_bit_energy_jump_passivity_or_formation_claim",
    )


def _admit_continuation_indices(values, label, size):
    """Bound the entire integer payload before any compiled calculation."""
    values = _ordered(values, label, limit=size + 1)
    if len(values) != size or any(type(value) is not int for value in values):
        raise TypeError(f"{label} must contain exactly {size} non-boolean integers")
    if any(not 0 <= value <= 10000 for value in values[:2]):
        raise ValueError("original packet indices must lie in 0..10000")
    if size == 3 and values[2] not in (0, 1):
        raise ValueError("the single contact-history bit must be zero or one")
    return values


def _admit_continuation_contact(contact, label):
    """Normalize one of the two fixed protocol ports without numeric work."""
    contact = _ordered(contact, label, limit=3)
    if (
        len(contact) != 2
        or any(type(value) is not int for value in contact)
        or contact not in ((0, 1), (0, 3))
    ):
        raise ValueError(f"{label} must be exactly (0,1) or (0,3)")
    return contact


def _record_sine_history_contact(packet, *, first_contact) -> tuple[int, int, int]:
    """Retain one declared contact choice without reading or evolving state.

    The packet is the original two-index output of the existing interface
    encoder. The fixed first-contact duration is 1/8. Bit zero denotes 01;
    bit one denotes 03. An already continued three-index record is rejected,
    so this operation cannot append an unrestricted history. No support
    event is executed or authenticated by constructing this record.
    """
    packet = _admit_continuation_indices(packet, "packet", 2)
    contact = _admit_continuation_contact(first_contact, "first_contact")
    return (*packet, int(contact == (0, 3)))


def _prepared_continuation_prefix(old_generator, first_generator, degree):
    """Combine chronological prefix degrees, retaining their matrix order."""
    initial = _prepared_divided_matrix_powers(
        old_generator, _INTERFACE_PREPARATION_TIME, degree
    )
    first = _prepared_divided_matrix_powers(
        first_generator, _CONTINUATION_CONTACT_TIME, degree
    )
    size = len(old_generator)
    combined = []
    for order in range(degree + 1):
        coefficient = [[Q(0)] * size for _ in range(size)]
        for first_order in range(order + 1):
            term = exact_matrix_product(
                first[first_order], initial[order - first_order]
            )
            for i, row in enumerate(term):
                for j, value in enumerate(row):
                    coefficient[i][j] += value
        combined.append(tuple(map(tuple, coefficient)))
    return tuple(combined)


@lru_cache(maxsize=4)
def _compiled_sine_history_continuation(first_contact, contact):
    """Compile immutable fixed-protocol coefficients, never realized state."""

    def support(selected_contact):
        return _admit_contact_support_information(
            nodes=(0, 1, 2, 3),
            old_edges=((1, 2), (2, 3)),
            old_conductances=(Q(1), Q(1)),
            capacity=(Q(1),) * 4,
            contact=selected_contact,
            contact_weight=Q(1),
            retained_nodes=selected_contact,
            observation_nodes=(0,),
            elapsed_time=_CONTINUATION_HORIZON,
            mean_kind="arithmetic",
            reading_error_bound=Q(0),
            tolerance=_INTERFACE_TOLERANCE,
        )

    current, first = support(contact), support(first_contact)
    matrices = (current.off_generator, first.on_generator, current.on_generator)
    mobility = 4 * max(-matrix[i][i] for matrix in matrices for i in range(4))
    duration = (
        _INTERFACE_PREPARATION_TIME + _CONTINUATION_CONTACT_TIME + _CONTINUATION_HORIZON
    )
    maxima = tuple(max(abs(lo), abs(hi)) for lo, hi in _INTERFACE_COORDINATE_BOUNDS)
    gamma_upper = _INTERFACE_GAMMA_BOUNDS[1]
    tube = _prepared_sine_tube(
        duration, mobility, max(maxima[:4]), max(maxima[4:]), gamma_upper
    )
    norm_step = (1 + gamma_upper) * mobility * duration / 2
    if norm_step > 1:
        raise ValueError("continuation tangent series requires norm step at most one")
    degree = _COUPLING_POLYNOMIAL_DEGREE
    prefix = _prepared_continuation_prefix(matrices[0], matrices[1], degree)
    off = _prepared_divided_matrix_powers(matrices[0], Q(1), degree)
    on = _prepared_divided_matrix_powers(matrices[2], Q(1), degree)
    channel = _prepared_coupling_channel_rows(degree)
    tensor = [[[Q(0)] * (degree + 1) for _ in range(degree + 1)] for _ in range(8)]
    for response_degree in range(1, degree + 1):
        difference = tuple(
            tuple(a - b for a, b in zip(on_row, off_row))
            for on_row, off_row in zip(on[response_degree], off[response_degree])
        )
        row = exact_matrix_product((current.observation_weights,), difference)[0]
        for prefix_degree in range(degree + 1 - response_degree):
            spatial = exact_matrix_product((row,), prefix[prefix_degree])[0]
            form, phase = channel[response_degree + prefix_degree]
            for node, value in enumerate(spatial):
                for power, (a, b) in enumerate(zip(form, phase)):
                    tensor[node][response_degree][power] += value * a
                    tensor[node + 4][response_degree][power] += value * b
    tensor = tuple(tuple(tuple(row) for row in coordinate) for coordinate in tensor)
    critical_bounds = tuple(
        sum(
            (
                abs(row[0]) * _CONTINUATION_HORIZON**j
                for j, row in enumerate(coordinate)
            ),
            Q(0),
        )
        for coordinate in tensor
    )
    discarded = _INTERFACE_EPSILON * sum(
        (critical_bounds[i] for i in (1, 5, 6, 7)), Q(0)
    )
    quantization = _INTERFACE_EPSILON * sum((critical_bounds[i] for i in (2, 3)), Q(0))
    deviation = gamma_upper - Q(1, 2)
    coupling = sum(
        (
            maximum * _CONTINUATION_HORIZON**j * deviation**r * abs(value)
            for maximum, coordinate in zip(maxima, tensor)
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
        series_tail_error=2 * max(maxima) * tail,
        nonlinear_error=2 * tube.form_error,
        series_norm_step=norm_step,
        series_tail=tail,
        mobility_degree=mobility,
        phase_growth=tube.growth,
        phase_tube=tube.phase_tube,
    )


def _decode_sine_history_continuation(
    record, *, contact, elapsed_time
) -> _PreparedSineHistoryContinuation:
    """Predict a second-probe x0 excess from a capped prescribed-history record.

    Admission precedes any cached or fresh matrix work, including at h=0.
    The original packet must match its source within the original quantizer
    errors, and its bit must match the actual first prescribed contact. The
    decoder authenticates neither premise from integer syntax. It receives
    no realized state, previous forecast, support cache or current reading.

    Old-support preparation lasts 1/4; the recorded contact lasts 1/8 and is
    then removed state-continuously. The same full reached state starts both
    second-probe branches, for every h in [0,1/32]. One unknown held coupling
    is shared across all stages. The reference is a critical degree-16
    ordered polynomial; uniform packet, parameter, series and nonlinear
    errors cover the complete actual history without a handoff reset.
    """
    record = _admit_continuation_indices(record, "record", 3)
    contact = _admit_continuation_contact(contact, "contact")
    time = exact_or_represented_real(elapsed_time, "elapsed_time")
    if not 0 <= time <= _CONTINUATION_HORIZON:
        raise ValueError("elapsed_time must lie in [0,1/32]")
    first_contact = (0, 1) if record[2] == 0 else (0, 3)
    compiled = _compiled_sine_history_continuation(first_contact, contact)
    reconstructed = (
        Q(0),
        Q(0),
        _INTERFACE_DELTA + _INTERFACE_QUANTUM * record[0],
        -_INTERFACE_DELTA + _INTERFACE_QUANTUM * record[1],
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
    return _PreparedSineHistoryContinuation(
        packet=record[:2],
        history_record=record,
        first_contact=first_contact,
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
