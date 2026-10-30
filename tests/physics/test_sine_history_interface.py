"""Independent controls for a fixed, quantized initial-information interface.

The two grid indices compress an admitted realized preparation. They are not
new readings, a reduced evolving state, or a fitted pair of future responses.
Only immutable pure oracle helpers are imported from prior frozen controls.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product, repeat
from math import factorial

import pytest

from tests.physics.test_sine_history_coupling_information import (
    _block_taylor_terms,
    _full_blocks,
    _matrix_product,
)
from tests.physics.test_sine_history_information import (
    _B,
    _display,
    _history_coefficients,
)
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics.linear_observation import (
    derive_coordinate_memory,
    derive_linear_observation,
)
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_history_interface as owner

DELTA = Q(1, 10000)
EPSILON = Q(1, 100000000)
HORIZON = Q(1, 4)
ETA = Q(1, 100)


def _state(**changes):
    result = dict(form=(Q(0), Q(0), 2 * DELTA, Q(0)), relative_phase=(Q(0),) * 4)
    result.update(changes)
    return result


def _decode(packet=(5000, 5000), contact=(0, 1), elapsed_time=HORIZON):
    return owner._decode_sine_history_interface(
        packet, contact=contact, elapsed_time=elapsed_time
    )


@lru_cache(maxsize=None)
def _tensor(contact):
    """Full8 Taylor products with separate h powers and one common e.

    Response terms are computed at duration1 so their time power remains
    symbolic; prefix terms already carry tau=1/4. No graph/channel factoring
    or production polynomial implementation is called by this oracle.
    """
    prefix = _block_taylor_terms(False, HORIZON)
    off, on = _block_taylor_terms(False, Q(1)), _block_taylor_terms(True, Q(1))
    tensor = [[[Q(0)] * 17 for _ in range(17)] for _ in range(8)]
    for n in range(1, 17):
        for m in range(17 - n):
            for r, (a, b) in enumerate(zip(off[n], on[n])):
                row = tuple(
                    (b[0][i] + b[1][i] - a[0][i] - a[1][i]) / 2 for i in range(8)
                )
                if not any(row):
                    continue
                for s, matrix in enumerate(prefix[m]):
                    for i, value in enumerate(_matrix_product((row,), matrix)[0]):
                        tensor[i][n][r + s] += value
    immutable = tuple(tuple(map(tuple, rows)) for rows in tensor)
    # Reflection of the old P3 support exchanges ports1 and3, both channel
    # coordinates, and the contact-pair output. The row check below verifies it.
    order = tuple(range(8)) if contact == (0, 1) else (0, 3, 2, 1, 4, 7, 6, 5)
    return tuple(immutable[i] for i in order)


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


def _physical_generator(contact, gamma, on):
    matrix = [[Q(0)] * 8 for _ in range(8)]
    edges = ((1, 2), (2, 3)) + ((contact,) if on else ())
    for i, j in edges:
        for receiver, other in ((i, j), (j, i)):
            matrix[receiver][receiver] -= Q(1, 4)
            matrix[receiver][other] += Q(1, 4)
            matrix[receiver][receiver + 4] -= gamma / 4
            matrix[receiver][other + 4] += gamma / 4
            matrix[receiver + 4][receiver] += gamma / 4
            matrix[receiver + 4][other] -= gamma / 4
    return tuple(map(tuple, matrix))


def _norm(matrix):
    return max(sum(map(abs, row), Q(0)) for row in matrix)


@pytest.fixture(scope="module", autouse=True)
def no_nonlinear_trajectory_event_or_retained_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("interface controls cannot integrate or replay a source")

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


@pytest.mark.parametrize("contact", [(0, 1), (0, 3)])
def test_full_physical_rows_verify_port_reflection_and_matched_source_gain(contact):
    gamma = Q(51, 100)
    off = _physical_generator(contact, gamma, False)
    on = _physical_generator(contact, gamma, True)
    difference = tuple(
        tuple(a - b for a, b in zip(arow, brow)) for arow, brow in zip(on, off)
    )
    observation = tuple(Q(1, 2) if i in contact else Q(0) for i in range(8))
    assert _matrix_product((observation,), difference) == ((Q(0),) * 8,)
    row = _matrix_product((observation,), off)[0]
    assert sum(map(abs, row), Q(0)) == (1 + gamma) / 4
    assert _norm(difference) == (1 + gamma) / 2
    assert max(_norm(off), _norm(on)) == 1 + gamma
    order = tuple(range(8)) if contact == (0, 1) else (0, 3, 2, 1, 4, 7, 6, 5)
    for active in (False, True):
        j0, j1 = _full_blocks(active)
        direct = _physical_generator(contact, gamma, active)
        assert direct == tuple(
            tuple(j0[i][j] + ETA * j1[i][j] for j in order) for i in order
        )
    # Duhamel plus ell*DeltaJ=0 gives the h² gain. The elementary geometric
    # exponential bound below controls EVERY h<=1/4 and gamma<=.51.
    z = (1 + gamma) * (2 * HORIZON)
    exponential = sum((z**k / factorial(k) for k in range(17)), Q(0)) + _tail(z)
    gain = (1 + gamma) ** 2 * HORIZON**2 * exponential / 16
    assert 0 < gain < Q(1, 50)
    assert EPSILON * gain < Q(2, 10**10)


def test_two_primitive_coordinates_do_not_close_the_actual_tangent():
    generator = _physical_generator((0, 1), Q(1, 2), False)
    projection = tuple(tuple(Q(i == j) for i in range(8)) for j in (2, 3))
    memory = derive_coordinate_memory(generator, (2, 3))
    hidden_index = memory.hidden_indices.index(1)
    assert tuple(row[hidden_index] for row in memory.hidden_to_visible) == (
        Q(1, 4),
        Q(0),
    )
    omitted = tuple(Q(i == 1) for i in range(8))
    assert _matrix_product(projection, tuple((x,) for x in omitted)) == (
        (Q(0),),
        (Q(0),),
    )
    derivative = _matrix_product(
        _matrix_product(projection, generator), tuple((x,) for x in omitted)
    )
    assert derivative == ((Q(1, 4),), (Q(0),))
    closure = derive_linear_observation(generator, projection)
    assert closure.output_rank == 2 and closure.dimension > 2
    assert "C J=G C" in closure.exact_identity_checks


def test_no_information_and_x2_only_descriptions_hide_an_actual_response_gap(
    record_property,
):
    plus = _state(form=(Q(0), Q(0), 2 * DELTA, DELTA))
    minus = _state(form=(Q(0), Q(0), 2 * DELTA, -DELTA))
    assert plus["form"][2] == minus["form"][2]
    assert plus["relative_phase"] == minus["relative_phase"] == (Q(0),) * 4
    plus_packet = owner._encode_sine_history_interface(**plus)
    minus_packet = owner._encode_sine_history_interface(**minus)
    assert plus_packet[0] == minus_packet[0] == 5000
    assert plus_packet[1] == 10000 and minus_packet[1] == 0
    # Port03 is the reflection of port01; its initial-x3 coefficient is
    # the independent closed critical history's initial-x1 coefficient.
    critical = _history_coefficients(HORIZON, HORIZON)[1] * DELTA
    rational_lower = DELTA * Q(161, 262144)
    assert critical.lo > rational_lower > Q(6, 10**8)
    b0, p0, g, T = 3 * DELTA, EPSILON, Q(1, 2), Q(1, 2)
    z = g * T
    c = 1 / (1 - z * z / 2)
    actual_error = 2 * T * g * 2 * (p0 + z * b0) ** 3 * c**4 / 3
    lower = rational_lower - actual_error
    assert lower > Q(1, 200000000)
    # The actual law, not an outer reached box, sends these two admitted
    # original preparations through the same history. The critical member
    # alone obstructs a uniform unknown-gamma decoder with these equal readings.
    record_property("negative_control_actual_half_gap_lower", _display(lower))
    record_property(
        "negative_scope",
        "no payload or x2-only; critical member of the unknown-law family, not every-gamma necessity",
    )


@pytest.mark.parametrize("contact", [(0, 1), (0, 3)])
def test_independent_full8_tensor_matches_all_h_and_gamma_coefficients(contact):
    result = _decode(contact=contact)
    tensor = _tensor(contact)
    assert result.coefficient_tensor == tensor
    assert len(tensor) == 8 and all(
        len(rows) == 17 and all(len(row) == 17 for row in rows) for rows in tensor
    )
    assert all(rows[0] == rows[1] == (Q(0),) * 17 for rows in tensor)
    assert result.polynomial_degree == 16
    # Reflection preserves the complete law and output association, not just
    # a scalar fit at one time. Individual h/e powers agree independently.
    other = _tensor((0, 3) if contact == (0, 1) else (0, 1))
    order = (0, 3, 2, 1, 4, 7, 6, 5)
    assert tensor == tuple(other[i] for i in order)


@pytest.mark.parametrize("contact", [(0, 1), (0, 3)])
def test_coefficient_inequalities_certify_every_time_coupling_and_packet(
    contact, record_property
):
    result = _decode(contact=contact)
    tensor = _tensor(contact)
    maxima = (Q(0), EPSILON, 3 * DELTA, DELTA, Q(0), EPSILON, EPSILON, EPSILON)
    critical = tuple(
        sum((abs(row[0]) * HORIZON**n for n, row in enumerate(rows)), Q(0))
        for rows in tensor
    )
    discarded = EPSILON * sum(critical[i] for i in (1, 5, 6, 7))
    quantization = EPSILON * (critical[2] + critical[3])
    coupling = sum(
        (
            maxima[i] * abs(value) * HORIZON**n * ETA**r
            for i, rows in enumerate(tensor)
            for n, row in enumerate(rows)
            for r, value in enumerate(row)
            if r
        ),
        Q(0),
    )
    z = Q(151, 200)
    tail = 2 * (3 * DELTA) * _tail(z)
    zg = Q(51, 200)
    growth = 1 / (1 - zg * zg / 2)
    phase = (EPSILON + zg * 3 * DELTA) * growth
    nonlinear = (
        2 * Q(1, 2) * Q(51, 100) * 2 * (EPSILON + zg * 3 * DELTA) ** 3 * growth**4 / 3
    )
    assert result.uniform_discarded_state_error_upper_bound == discarded
    assert result.uniform_quantization_error_upper_bound == quantization
    assert result.uniform_coupling_error_upper_bound == coupling
    assert result.uniform_series_tail_error_upper_bound == tail
    assert result.uniform_nonlinear_error_upper_bound == nonlinear
    assert result.series_norm_step == z and result.series_tail_upper_bound == _tail(z)
    assert (
        result.phase_growth_bound == growth and result.phase_tube_upper_bound == phase
    )
    assert discarded + quantization < Q(2, 10**10)
    total = discarded + quantization + coupling + tail + nonlinear
    assert (
        result.uniform_error_upper_bound
        == result.forecast_error_upper_bound
        == total
        <= Q(5, 10**9)
    )
    assert result.status == "certified"
    # Each |h^n e^r| is bounded by H^n eta^r. These coefficient sums apply
    # to the continuum of times and couplings and every bounded original
    # state, not merely the finite consistency checks below. The source
    # variation term includes all actual coordinates, including small phases.
    corner = (Q(0), -EPSILON, 3 * DELTA, -DELTA, Q(0), EPSILON, -EPSILON, EPSILON)
    for h, e in product((Q(0), HORIZON / 2, HORIZON), (-ETA, Q(0), ETA)):
        assert (
            abs(_value(tensor, corner, h, e) - _value(tensor, corner, h, Q(0)))
            <= coupling
        )
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
    packet = owner._encode_sine_history_interface(
        form=actual[:4], relative_phase=actual[4:]
    )
    decoded = _decode(packet=packet, contact=contact)
    reconstruction = decoded.reconstructed_form + decoded.reconstructed_phase
    assert all(abs(a - b) <= EPSILON for a, b in zip(actual, reconstruction))
    assert all(actual[i] != reconstruction[i] for i in (2, 3))
    for h, e in product((HORIZON / 2, HORIZON), (-ETA, Q(0), ETA)):
        difference = _value(tensor, actual, h, e) - _value(
            tensor, reconstruction, h, Q(0)
        )
        assert abs(difference) <= coupling + discarded + quantization
    # The encoder establishes this association. Merely valid integer codes
    # do not authenticate a claim about some unrelated external realization.
    assert coupling > 0 and nonlinear > 0 and discarded > 0 and quantization > 0
    record_property("contact", str(contact))
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
        "all admitted original states; both ports tested; every h in[0,1/4] and one common gamma in[.49,.51]",
    )


def test_exact_quantizer_endpoints_and_upward_ties_do_not_round_through_float():
    encode = owner._encode_sine_history_interface
    for value, code in ((DELTA, 0), (3 * DELTA, 10000), (2 * DELTA, 5000)):
        packet = encode(**_state(form=(Q(0), Q(0), value, value - 2 * DELTA)))
        assert packet == (code, code)
    tiny = Q(1, 10**100)
    for shift, code in ((-tiny, 0), (Q(0), 1), (tiny, 1)):
        x2 = DELTA + EPSILON + shift
        x3 = -DELTA + EPSILON + shift
        packet = encode(**_state(form=(Q(0), Q(0), x2, x3)))
        assert packet == (code, code)
        q2, q3 = DELTA + 2 * EPSILON * code, -DELTA + 2 * EPSILON * code
        assert abs(q2 - x2) <= EPSILON and abs(q3 - x3) <= EPSILON
    assert encode(
        form=iter((0, 0, 2 * DELTA, 0)), relative_phase=iter((0, 0, 0, 0))
    ) == (5000, 5000)


def test_encoder_discards_realized_coordinates_without_storing_hidden_information():
    first = _state(
        form=(0, EPSILON, 2 * DELTA, 0), relative_phase=(0, EPSILON, -EPSILON, EPSILON)
    )
    second = _state(
        form=(0, -EPSILON, 2 * DELTA, 0),
        relative_phase=(0, -EPSILON, EPSILON, -EPSILON),
    )
    a = owner._encode_sine_history_interface(**first)
    b = owner._encode_sine_history_interface(**second)
    assert type(a) is tuple and a == b == (5000, 5000)
    assert all(type(value) is int for value in a)
    # The entire instance-specific retained object is this pair of codes.
    assert len(a) == 2


def test_decoder_consumes_only_codes_query_and_fixed_metadata(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("decoder tried to reacquire an original realization")

    monkeypatch.setattr(owner, "_encode_sine_history_interface", forbidden)
    result = _decode(
        packet=iter((7500, 2500)), contact=iter((0, 3)), elapsed_time=Q(1, 8)
    )
    assert result.packet == (7500, 2500)
    assert result.reconstructed_form == (0, 0, Q(5, 2) * DELTA, -DELTA / 2)
    assert result.reconstructed_phase == (Q(0),) * 4
    with pytest.raises(TypeError):
        owner._decode_sine_history_interface(
            (5000, 5000), contact=(0, 1), elapsed_time=HORIZON, form=_state()["form"]
        )
    with pytest.raises(TypeError):
        owner._decode_sine_history_interface(
            (5000, 5000), contact=(0, 1), elapsed_time=HORIZON, compiled=result
        )
    with pytest.raises(FrozenInstanceError):
        result.packet = (0, 0)


@pytest.mark.parametrize("contact", [(0, 1), (0, 3)])
def test_query_forecast_matches_independent_critical_closed_modes(contact):
    for packet, h in (
        ((0, 10000), HORIZON),
        ((10000, 0), HORIZON / 2),
        ((5000, 5000), Q(0)),
    ):
        result = _decode(packet=packet, contact=contact, elapsed_time=h)
        state = result.reconstructed_form + result.reconstructed_phase
        expected = _value(_tensor(contact), state, h, Q(0))
        assert result.predicted_response == expected
        coefficient = _history_coefficients(HORIZON, h)
        if contact == (0, 3):
            coefficient = tuple(coefficient[i] for i in (0, 3, 2, 1, 4, 7, 6, 5))
        exact = sum((a * x for a, x in zip(coefficient, state)), _B(0))
        assert (
            expected - result.uniform_series_tail_error_upper_bound
            <= exact.lo
            <= exact.hi
            <= expected + result.uniform_series_tail_error_upper_bound
        )
        assert result.forecast_response_bounds == (
            expected - result.forecast_error_upper_bound,
            expected + result.forecast_error_upper_bound,
        )


@pytest.mark.parametrize("contact", [(0, 1), (0, 3)])
def test_zero_query_retains_the_nonzero_uniform_family_budget(contact):
    result = _decode(contact=contact, elapsed_time=Q(0))
    assert result.predicted_response == result.forecast_error_upper_bound == 0
    assert result.forecast_response_bounds == (Q(0), Q(0))
    assert result.uniform_error_upper_bound > 0
    assert result.status == "certified"
    assert result.response_sign == "zero"


def test_payload_cost_is_explicit_without_minimality_or_reduced_state_claim():
    result = _decode()
    assert result.projection_rank == result.payload_scalar_count == 2
    assert result.encoded_coordinate_labels == ((2, "form"), (3, "form"))
    assert result.quantum == 2 * EPSILON and result.quantization_error_bound == EPSILON
    assert result.payload_bits == 28 and 10001 <= 2**14
    assert (
        2**26 < 10001**2 <= 2**27
    )  # Packing could differ; 28 is this two-slot protocol.
    assert result.measurement_cost == result.evolved_coordinate_count == 0
    assert result.preparation_time == result.horizon == HORIZON
    assert result.gamma_bounds == (Q(49, 100), Q(51, 100))
    assert result.reference_gamma == Q(1, 2)


def test_encoder_admits_every_original_coordinate_not_only_the_payload():
    bad = (
        {"form": (0, 0, DELTA - EPSILON, 0)},
        {"form": (0, 0, 3 * DELTA + EPSILON, 0)},
        {"form": (0, 0, 2 * DELTA, DELTA + EPSILON)},
        {"form": (EPSILON, 0, 2 * DELTA, 0)},
        {"form": (0, 2 * EPSILON, 2 * DELTA, 0)},
        {"relative_phase": (EPSILON, 0, 0, 0)},
        {"relative_phase": (0, 0, 2 * EPSILON, 0)},
        {"relative_phase": (0, False, 0, 0)},
        {"form": (0, float("nan"), 2 * DELTA, 0)},
        {"relative_phase": (0, 0, float("inf"), 0)},
        {"form": repeat(Q(0))},
        {"relative_phase": {0, 1, 2, 3}},
    )
    for change in bad:
        with pytest.raises((TypeError, ValueError)):
            owner._encode_sine_history_interface(**_state(**change))


def test_all_query_admission_precedes_cached_or_fresh_polynomial_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid packet/query reached cached or fresh polynomial work")

    monkeypatch.setattr(owner, "_compiled_sine_history_interface", forbidden)
    bad = (
        {"packet": ()},
        {"packet": (0,)},
        {"packet": (0, 1, 2)},
        {"packet": (True, 0)},
        {"packet": (Q(0), 0)},
        {"packet": (0.0, 0)},
        {"packet": (-1, 0)},
        {"packet": (0, 10001)},
        {"packet": repeat(0)},
        {"packet": {0, 1}},
        {"contact": (1, 0)},
        {"contact": (0, 2)},
        {"contact": (False, 1)},
        {"contact": repeat(0)},
        {"elapsed_time": True},
        {"elapsed_time": -EPSILON},
        {"elapsed_time": HORIZON + Q(1, 10**100)},
        {"elapsed_time": float("nan")},
        {"packet": (False, 0), "elapsed_time": Q(0)},
    )
    for changes in bad:
        with pytest.raises((TypeError, ValueError)):
            _decode(**changes)
