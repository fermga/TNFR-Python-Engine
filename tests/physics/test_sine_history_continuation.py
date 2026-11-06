"""Independent three-stage controls for one bounded history record.

Full eight-coordinate matrix polynomials preserve chronological order. The
counterexample changes only the prescribed first port, keeps the original
source and packet, and removes that edge before the identical second query.
No physical occurrence, current-state reading or autonomous update is inferred.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product, repeat
from math import factorial

import pytest

from tests.physics.test_sine_history_coupling_information import (
    _block_taylor_terms,
    _matrix_product,
)
from tests.physics.test_sine_history_information import _display, _tangent
from tests.physics.test_sine_history_interface import _physical_generator
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_history_continuation as owner
from tnfr.physics import _sine_history_interface as original

DELTA = Q(1, 10000)
EPSILON = Q(1, 100000000)
TAU = Q(1, 4)
PULSE = Q(1, 8)
HORIZON = Q(1, 32)
ETA = Q(1, 100)
CONTACTS = ((0, 1), (0, 3))
PAIRS = tuple(product(CONTACTS, repeat=2))


def _decode(
    first_contact=(0, 1), contact=(0, 1), elapsed_time=HORIZON, packet=(5000, 5000)
):
    record = owner._record_sine_history_contact(packet, first_contact=first_contact)
    return owner._decode_sine_history_continuation(
        record, contact=contact, elapsed_time=elapsed_time
    )


def _sum_matrices(left, right):
    return tuple(
        tuple(a + b for a, b in zip(arow, brow)) for arow, brow in zip(left, right)
    )


def _reflection(contact):
    return tuple(range(8)) if contact == (0, 1) else (0, 3, 2, 1, 4, 7, 6, 5)


def _port_terms(contact, duration):
    order = _reflection(contact)
    return tuple(
        tuple(
            tuple(tuple(matrix[i][j] for j in order) for i in order)
            for matrix in polynomial
        )
        for polynomial in _block_taylor_terms(True, duration)
    )


@lru_cache(maxsize=None)
def _prefix(first_contact):
    """Full8 pulse-after-old-preparation coefficients by total degree and e."""
    old = _block_taylor_terms(False, TAU)
    pulse = _port_terms(first_contact, PULSE)
    zero = tuple((Q(0),) * 8 for _ in range(8))
    result = []
    for total in range(17):
        coefficients = [zero for _ in range(total + 1)]
        for pulse_degree in range(total + 1):
            old_degree = total - pulse_degree
            for r, a in enumerate(pulse[pulse_degree]):
                for s, b in enumerate(old[old_degree]):
                    coefficients[r + s] = _sum_matrices(
                        coefficients[r + s], _matrix_product(a, b)
                    )
        result.append(tuple(coefficients))
    return tuple(result)


@lru_cache(maxsize=None)
def _tensor(first_contact, query_contact):
    prefix = _prefix(first_contact)
    off = _block_taylor_terms(False, Q(1))
    on = _port_terms(query_contact, Q(1))
    tensor = [[[Q(0)] * 17 for _ in range(17)] for _ in range(8)]
    for n in range(1, 17):
        for m in range(17 - n):
            for r, (a, b) in enumerate(zip(off[n], on[n])):
                row = tuple(y - x for x, y in zip(a[0], b[0]))
                if not any(row):
                    continue
                for s, matrix in enumerate(prefix[m]):
                    for i, value in enumerate(_matrix_product((row,), matrix)[0]):
                        tensor[i][n][r + s] += value
    return tuple(tuple(map(tuple, rows)) for rows in tensor)


def _value(tensor, state, h, e):
    return sum(
        (
            state[i] * coefficient * h**n * e**r
            for i, rows in enumerate(tensor)
            for n, row in enumerate(rows)
            for r, coefficient in enumerate(row)
        ),
        Q(0),
    )


def _tail(z):
    return z**17 / factorial(17) / (1 - z / 18)


def _closed_stage(duration, state, contact):
    if contact is None:
        return _tangent(duration, state[:4], state[4:], on=False)[0]
    order = _reflection(contact)
    reflected = tuple(state[i] for i in order)
    reached = _tangent(duration, reflected[:4], reflected[4:], on=True)[0]
    return tuple(reached[i] for i in order)


def _closed_response(state, first_contact, query_contact, h):
    prepared = _closed_stage(TAU, state, None)
    handoff = _closed_stage(PULSE, prepared, first_contact)
    # Removal changes the support and keeps precisely the same entire state.
    # Both second branches start here; off x0 is its inherited value, not0.
    on = _closed_stage(h, handoff, query_contact)
    off = _closed_stage(h, handoff, None)
    return on[0] - off[0], prepared, handoff, off


@pytest.fixture(scope="module", autouse=True)
def no_nonlinear_trajectory_committed_event_or_source_replay():
    def forbidden(*args, **kwargs):
        pytest.fail("continuation controls cannot integrate, commit or replay")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
            (external, "_derive_organized_external_interaction"),
            (external, "_assess_organized_external_interaction"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def test_second_excess_readout_has_first_order_contact_and_distinct_history_terms():
    gamma = Q(1, 2)
    old = _physical_generator((0, 1), gamma, False)
    first = _physical_generator((0, 1), gamma, True)
    second = _physical_generator((0, 3), gamma, True)
    query_difference = tuple(
        tuple(a - b for a, b in zip(arow, brow)) for arow, brow in zip(second, old)
    )
    history_difference = tuple(
        tuple(a - b for a, b in zip(arow, brow)) for arow, brow in zip(first, second)
    )
    assert any(query_difference[0])
    initial = (Q(0), Q(0), 2 * DELTA, DELTA) + (Q(0),) * 4
    mixed = _matrix_product((query_difference[0],), history_difference)[0]
    leading = sum((a * x for a, x in zip(mixed, initial)), Q(0)) * PULSE * HORIZON / 2
    assert leading == 3 * PULSE * HORIZON * DELTA / 64 > Q(1, 200000000)
    # The source was fixed analytically: its first old-preparation correction
    # in the leading spatial history direction vanishes exactly.
    spatial = (-1, -1, 0, 2)
    form_rate = tuple(
        sum((old[i][j] * initial[j] for j in range(8)), Q(0)) for i in range(4)
    )
    assert sum((a * x for a, x in zip(spatial, form_rate)), Q(0)) == 0


@pytest.mark.parametrize("first_contact,contact", PAIRS)
def test_full8_three_stage_coefficients_preserve_chronology_and_shared_gamma(
    first_contact, contact
):
    result = _decode(first_contact=first_contact, contact=contact)
    tensor = _tensor(first_contact, contact)
    assert result.coefficient_tensor == tensor
    assert all(rows[0] == (Q(0),) * 17 for rows in tensor)
    assert any(any(rows[1]) for rows in tensor)
    assert result.polynomial_degree == 16
    # Pulse and old generators do not commute; their order is observable.
    old = _physical_generator((0, 1), Q(1, 2), False)
    pulse = _physical_generator(first_contact, Q(1, 2), True)
    assert _matrix_product(pulse, old) != _matrix_product(old, pulse)
    assert tensor != _tensor((0, 3) if first_contact == (0, 1) else (0, 1), contact)


@pytest.mark.parametrize("first_contact,contact", PAIRS)
def test_uniform_entire_family_error_keeps_all_five_distinct_allowances(
    first_contact, contact, record_property
):
    result = _decode(first_contact=first_contact, contact=contact)
    tensor = _tensor(first_contact, contact)
    maxima = (Q(0), EPSILON, 3 * DELTA, DELTA, Q(0), EPSILON, EPSILON, EPSILON)
    critical = tuple(
        sum((abs(row[0]) * HORIZON**n for n, row in enumerate(rows)), Q(0))
        for rows in tensor
    )
    discarded = EPSILON * sum(critical[i] for i in (1, 5, 6, 7))
    quantization = EPSILON * (critical[2] + critical[3])
    coupling = sum(
        (
            maxima[i] * abs(coefficient) * HORIZON**n * ETA**r
            for i, rows in enumerate(tensor)
            for n, row in enumerate(rows)
            for r, coefficient in enumerate(row)
            if r
        ),
        Q(0),
    )
    total_time = TAU + PULSE + HORIZON
    assert total_time == Q(13, 32)
    gamma = Q(51, 100)
    norm_step = (1 + gamma) * total_time
    tail = 2 * (3 * DELTA) * _tail(norm_step)
    zg = gamma * total_time
    growth = 1 / (1 - zg**2 / 2)
    nonlinear = (
        2 * total_time * gamma * 2 * (EPSILON + zg * 3 * DELTA) ** 3 * growth**4 / 3
    )
    for name, expected in (
        ("uniform_discarded_state_error_upper_bound", discarded),
        ("uniform_quantization_error_upper_bound", quantization),
        ("uniform_coupling_error_upper_bound", coupling),
        ("uniform_series_tail_error_upper_bound", tail),
        ("uniform_nonlinear_error_upper_bound", nonlinear),
    ):
        assert getattr(result, name) == expected > 0
    assert result.series_norm_step == norm_step == Q(1963, 3200)
    assert result.series_tail_upper_bound == _tail(norm_step)
    assert result.phase_growth_bound == growth
    assert result.phase_tube_upper_bound == (EPSILON + zg * 3 * DELTA) * growth
    total = discarded + quantization + coupling + tail + nonlinear
    assert (
        result.uniform_error_upper_bound
        == result.forecast_error_upper_bound
        == total
        <= Q(5, 10**9)
    )
    assert result.status == "certified"
    # Absolute h/e coefficient sums prove the continuum inequality. These
    # additional values check association with a genuinely rounded source.
    actual = (
        Q(0),
        EPSILON,
        2 * DELTA + EPSILON / 3,
        DELTA - EPSILON / 2,
        Q(0),
        -EPSILON,
        EPSILON,
        -EPSILON,
    )
    packet = original._encode_sine_history_interface(
        form=actual[:4], relative_phase=actual[4:]
    )
    decoded = _decode(first_contact=first_contact, contact=contact, packet=packet)
    reconstructed = decoded.reconstructed_form + decoded.reconstructed_phase
    assert all(abs(a - b) <= EPSILON for a, b in zip(actual, reconstructed))
    for h, e in product((Q(0), HORIZON / 2, HORIZON), (-ETA, Q(0), ETA)):
        discrepancy = _value(tensor, actual, h, e) - _value(
            tensor, reconstructed, h, Q(0)
        )
        assert abs(discrepancy) <= discarded + quantization + coupling
    record_property("history_and_query", str((first_contact, contact)))
    for name, value in (
        ("discarded_state", discarded),
        ("quantization", quantization),
        ("coupling", coupling),
        ("series_tail", tail),
        ("nonlinear", nonlinear),
        ("uniform_error", total),
    ):
        record_property(name, _display(value))
    record_property(
        "uniform_scope",
        "every matching source/packet, common gamma[.49,.51], future h[0,1/32]; one recorded prescribed pulse",
    )


@pytest.mark.parametrize("first_contact,contact", PAIRS)
def test_closed_critical_stages_match_point_forecast_without_reinitializing(
    first_contact, contact
):
    result = _decode(
        first_contact=first_contact,
        contact=contact,
        packet=(7500, 7500),
        elapsed_time=HORIZON / 2,
    )
    state = result.reconstructed_form + result.reconstructed_phase
    exact, prepared, handoff, off = _closed_response(
        state, first_contact, contact, HORIZON / 2
    )
    polynomial = _value(_tensor(first_contact, contact), state, HORIZON / 2, Q(0))
    assert result.predicted_response == polynomial
    assert (
        polynomial - result.uniform_series_tail_error_upper_bound
        <= exact.lo
        <= exact.hi
        <= polynomial + result.uniform_series_tail_error_upper_bound
    )
    assert handoff[0].lo > 0
    assert off[0].lo == handoff[0].lo and off[0].hi == handoff[0].hi
    assert any(v.lo > 0 or v.hi < 0 for v in prepared[4:])
    assert any(v.lo > 0 or v.hi < 0 for v in handoff[4:])
    assert result.forecast_response_bounds == (
        polynomial - result.forecast_error_upper_bound,
        polynomial + result.forecast_error_upper_bound,
    )


def test_same_original_packet_and_present_support_need_the_prior_port_for_this_witness(
    record_property,
):
    state = (Q(0), Q(0), 2 * DELTA, DELTA) + (Q(0),) * 4
    packet = original._encode_sine_history_interface(
        form=state[:4], relative_phase=state[4:]
    )
    records = tuple(
        owner._record_sine_history_contact(packet, first_contact=p) for p in CONTACTS
    )
    assert packet == (5000, 10000) and records == ((5000, 10000, 0), (5000, 10000, 1))
    contact = (0, 3)
    # Original state, packet, inventory, common gamma, current old graph and
    # second query coincide. Actual handoff coordinates need not coincide:
    # the description supplies no new current-state readings.
    first = _closed_response(state, (0, 1), contact, HORIZON)
    second = _closed_response(state, (0, 3), contact, HORIZON)
    exact_half = (first[0] - second[0]) / 2
    first_poly = _value(_tensor((0, 1), contact), state, HORIZON, Q(0))
    second_poly = _value(_tensor((0, 3), contact), state, HORIZON, Q(0))
    polynomial_half = (first_poly - second_poly) / 2
    budget = _decode(first_contact=(0, 1), contact=contact, packet=packet)
    tail = budget.uniform_series_tail_error_upper_bound
    assert (
        polynomial_half - tail
        <= exact_half.lo
        <= exact_half.hi
        <= polynomial_half + tail
    )
    lower = polynomial_half - tail - budget.uniform_nonlinear_error_upper_bound
    assert lower > Q(5, 10**9)
    assert any(a.hi < b.lo or b.hi < a.lo for a, b in zip(first[2], second[2]))
    assert tuple(state) == (0, 0, 2 * DELTA, DELTA, 0, 0, 0, 0)
    record_property("critical_history_half_gap_polynomial", _display(polynomial_half))
    record_property("actual_history_half_gap_lower", _display(lower))
    record_property(
        "negative_scope",
        "same original source/packet/inventory and present support; first port omitted; no current coordinates supplied; critical member only",
    )


def test_one_history_slot_and_payload_cost_are_bounded_not_an_unlimited_action_log():
    packet = original._encode_sine_history_interface(
        form=(0, 0, 2 * DELTA, 0), relative_phase=(0, 0, 0, 0)
    )
    record = owner._record_sine_history_contact(
        iter(packet), first_contact=iter((0, 3))
    )
    assert record == (5000, 5000, 1) and type(record) is tuple
    with pytest.raises((TypeError, ValueError)):
        owner._record_sine_history_contact(record, first_contact=(0, 1))
    result = owner._decode_sine_history_continuation(
        iter(record), contact=iter((0, 1)), elapsed_time=HORIZON
    )
    assert result.packet == packet and result.history_record == record
    assert result.first_contact == (0, 3) and result.first_contact_duration == PULSE
    assert result.preparation_time == TAU and result.horizon == HORIZON
    assert (
        result.initial_payload_bits == 28
        and result.history_payload_bits == 1
        and result.payload_bits == 29
    )
    assert (
        result.payload_scalar_count == 3
        and result.recorded_contact_count == result.max_recorded_contacts == 1
    )
    assert result.measurement_cost == result.evolved_coordinate_count == 0
    assert result.observation_nodes == (0,) and result.observation_weights == (
        1,
        0,
        0,
        0,
    )
    assert 2 * 10001**2 < 2**28  # 29 fixed-slot bits are not minimum packed bits.
    with pytest.raises(FrozenInstanceError):
        result.history_record = (0, 0, 0)


def test_decoder_never_reacquires_original_or_current_state(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("continuation decoder attempted source/current-state readback")

    monkeypatch.setattr(original, "_encode_sine_history_interface", forbidden)
    monkeypatch.setattr(original, "_decode_sine_history_interface", forbidden)
    result = owner._decode_sine_history_continuation(
        (5000, 10000, 1), contact=(0, 3), elapsed_time=HORIZON
    )
    assert result.reconstructed_form == (0, 0, 2 * DELTA, DELTA)
    assert result.reconstructed_phase == (Q(0),) * 4
    for extra in (
        {"original_state": (0,) * 8},
        {"current_state": (0,) * 8},
        {"compiled": result},
    ):
        with pytest.raises(TypeError):
            owner._decode_sine_history_continuation(
                (5000, 10000, 1), contact=(0, 3), elapsed_time=HORIZON, **extra
            )


def test_zero_second_horizon_keeps_prior_history_and_family_bounds():
    for first_contact, contact in PAIRS:
        result = _decode(
            first_contact=first_contact, contact=contact, elapsed_time=Q(0)
        )
        assert (
            result.first_contact == first_contact
            and result.first_contact_duration == PULSE
        )
        assert result.predicted_response == result.forecast_error_upper_bound == 0
        assert result.forecast_response_bounds == (Q(0), Q(0))
        assert result.uniform_error_upper_bound > 0
        assert result.status == "certified" and result.response_sign == "zero"


def test_record_admission_does_not_upgrade_invalid_or_already_advanced_packets():
    packets = (
        (),
        (0,),
        (0, 0, 0),
        (True, 0),
        (Q(0), 0),
        (0.0, 0),
        (-1, 0),
        (0, 10001),
        repeat(0),
        {0, 1},
    )
    for packet in packets:
        with pytest.raises((TypeError, ValueError)):
            owner._record_sine_history_contact(packet, first_contact=(0, 1))
    for contact in ((1, 0), (0, 2), (False, 1), repeat(0), {0, 1}):
        with pytest.raises((TypeError, ValueError)):
            owner._record_sine_history_contact((5000, 5000), first_contact=contact)


def test_all_decoder_admission_precedes_cached_or_fresh_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid continuation reached polynomial work")

    monkeypatch.setattr(owner, "_compiled_sine_history_continuation", forbidden)
    changes = (
        {"record": ()},
        {"record": (0, 0)},
        {"record": (0, 0, 0, 0)},
        {"record": (0, 0, True)},
        {"record": (0, 0, Q(0))},
        {"record": (0, 0, -1)},
        {"record": (0, 0, 2)},
        {"record": (-1, 0, 0)},
        {"record": (0, 10001, 0)},
        {"record": repeat(0)},
        {"record": {0, 1, 2}},
        {"contact": (0, 2)},
        {"contact": (1, 0)},
        {"contact": (False, 1)},
        {"elapsed_time": True},
        {"elapsed_time": -EPSILON},
        {"elapsed_time": HORIZON + Q(1, 10**100)},
        {"elapsed_time": float("nan")},
        {"record": (0, 0, False), "elapsed_time": Q(0)},
    )
    for changed in changes:
        arguments = dict(record=(5000, 5000, 0), contact=(0, 1), elapsed_time=HORIZON)
        arguments.update(changed)
        with pytest.raises((TypeError, ValueError)):
            owner._decode_sine_history_continuation(**arguments)
