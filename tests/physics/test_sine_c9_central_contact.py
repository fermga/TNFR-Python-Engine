"""Static exact-row controls for the zero-origin central C9 contact null.

These samples exercise the reflection/sign invariant subspace and its local
failure boundary. They do not propagate a state, extend the null to arbitrary
preparation errors, or evaluate a finite interaction response. Mathematical
irrational targets are not replaced by a floating approximation to pi.
"""

from collections import Counter
from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import _sine_rates, _sine_state_from_rows
from tnfr.physics.relational_sine_composition import assess_sine_pattern_composition
from tnfr.physics.relational_sine_pattern import _bound_sine_pattern_from_rows

MODEL = RelationalExchangeModel(
    1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
)
SLOPE = Q(2046, 9) * Q(355, 113) ** 2
PORTS = (4, 13)


def _odd(left):
    left = tuple(map(Q, left))
    assert len(left) == 4
    return left + (Q(0),) + tuple(-value for value in reversed(left))


def _support(size, bridge=False):
    # Fixture admission: one or two complete simple unit C9 components,
    # exact node order, reciprocal nonempty rows and one specified bridge.
    assert size in (9, 18)
    edges = tuple(
        sorted(
            (min(start + i, start + (i + 1) % 9), max(start + i, start + (i + 1) % 9))
            for start in range(0, size, 9)
            for i in range(9)
        )
    )
    if bridge:
        assert size == 18
        edges = tuple(sorted(edges + (PORTS,)))
    neighbors = [[] for _ in range(size)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return edges, tuple(map(tuple, neighbors))


def _field(forms, phases, *, bridge=False):
    forms, phases = tuple(forms), tuple(phases)
    assert len(forms) == len(phases)
    assert all(type(value) is Q for value in forms + phases)
    edges, neighbors = _support(len(forms), bridge)
    state = _sine_state_from_rows(
        tuple(range(len(forms))), edges, forms, phases, (Q(1),) * len(forms), neighbors
    )
    rates = _sine_rates(
        MODEL,
        state.degrees,
        state.form_gradient,
        state.capacity,
        tuple(imaginary for _, imaginary in state.relative_resultant),
    )
    return state, rates


def _zero_enclosure(value):
    # Independent interval sums retain small trig uncertainty; membership of
    # zero is checked alongside exact symbolic cancellation of their angles.
    # The shared static resultant reader uses a 64-bit trigonometric grid.
    assert value.contains(0) and value.abs_max < Q(1, 2**60)


def _assert_port_cancellation(state, port):
    angles = Counter(state.phase[j] - state.phase[port] for j in state.neighbors[port])
    assert angles == Counter({-angle: count for angle, count in angles.items()})
    assert state.epi[port] == state.phase[port] == 0
    assert state.form_gradient[port] == 0


@pytest.fixture(scope="module", autouse=True)
def no_trajectory():
    from tnfr.physics import relational_sine_forecast

    def forbidden(*args, **kwargs):
        pytest.fail("central contact controls must remain instantaneous")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        patch.setattr(relational_sine_forecast, "bound_sine_flow", forbidden)
        yield


@pytest.mark.parametrize("sender_class", [1, 2])
def test_nominal_phase_flat_sources_keep_zero_ports_and_product_field(sender_class):
    left = tuple(sender_class * SLOPE * (i - 4) for i in range(9))
    right = tuple(SLOPE * (i - 4) for i in range(9))
    phases = (Q(0),) * 18
    separate, before = _field(left + right, phases)
    joined, after = _field(left + right, phases, bridge=True)
    assert separate.degrees == (2,) * 18
    assert joined.degrees == tuple(3 if i in PORTS else 2 for i in range(18))
    for port in PORTS:
        _assert_port_cancellation(joined, port)
        assert before["form_rates"][port] == after["form_rates"][port] == I(0)
        assert before["phase_rates"][port] == after["phase_rates"][port] == I(0)
    assert before["form_rates"] == after["form_rates"]
    assert before["phase_rates"] == after["phase_rates"]


@pytest.mark.parametrize(
    "left_form,left_phase,right_form,right_phase",
    [
        (
            (-3, 2, -1, Q(1, 3)),
            (Q(1, 7), -2, 3, Q(-1, 5)),
            (4, -1, 5, -3),
            (-2, Q(1, 2), -1, 3),
        ),
        ((0, 0, 0, 0), (1, -1, 2, -2), (2, 4, 6, 8), (0, 0, 0, 0)),
        (
            (Q(1, 10**20), -1, 2, 0),
            (Q(1, 11), Q(2, 13), Q(-3, 17), Q(4, 19)),
            (-1, -2, -3, -4),
            (3, 1, -4, -2),
        ),
    ],
)
def test_arbitrary_odd_rows_are_invariant_and_joined_field_is_the_product(
    left_form, left_phase, right_form, right_phase
):
    forms, phases = _odd(left_form) + _odd(right_form), _odd(left_phase) + _odd(
        right_phase
    )
    separate, before = _field(forms, phases)
    joined, after = _field(forms, phases, bridge=True)
    reflection = tuple(8 - i if i < 9 else 26 - i for i in range(18))
    for port in PORTS:
        _assert_port_cancellation(separate, port)
        _assert_port_cancellation(joined, port)
        _zero_enclosure(before["form_rates"][port])
        _zero_enclosure(after["form_rates"][port])
        assert before["phase_rates"][port] == after["phase_rates"][port] == I(0)
    for name in ("form_rates", "phase_rates"):
        for i in range(18):
            assert after[name][reflection[i]] == -after[name][i]
            if i not in PORTS:
                assert after[name][i] == before[name][i]


def test_two_nominal_sender_classes_leave_the_same_arbitrary_odd_receiver_field():
    receiver_form = _odd((2, -3, 4, -5))
    receiver_phase = _odd((Q(1, 7), -1, Q(2, 5), Q(-1, 3)))
    isolated, receiver_rates = _field(receiver_form, receiver_phase)
    candidates = []
    for k in (1, 2):
        sender = tuple(k * SLOPE * (i - 4) for i in range(9))
        state, rates = _field(
            sender + receiver_form, (Q(0),) * 9 + receiver_phase, bridge=True
        )
        _assert_port_cancellation(state, PORTS[1])
        candidates.append(rates)
    for name in ("form_rates", "phase_rates"):
        assert candidates[0][name][9:] == candidates[1][name][9:]
        for i in range(9):
            if i != 4:
                assert candidates[0][name][9 + i] == receiver_rates[name][i]
    assert isolated.form_gradient[4] == 0


def test_disconnected_receiver_is_independent_even_for_nonodd_sender_rows():
    receiver_form = tuple(Q(i * i - 7) for i in range(9))
    receiver_phase = tuple(Q(i, 7) for i in range(9))
    _, expected = _field(receiver_form, receiver_phase)
    for sender in (
        tuple(Q(i**3) for i in range(9)),
        tuple(Q(1 - i, 13) for i in range(9)),
    ):
        _, actual = _field(
            sender + receiver_form, tuple(Q(i, 11) for i in range(9)) + receiver_phase
        )
        for name in ("form_rates", "phase_rates"):
            assert actual[name][9:] == expected[name]


def test_equal_nonzero_relative_phase_offset_gives_equal_initial_port_rates_only():
    offset = Q(1, 4)  # Exact ordinary nonzero angle; no represented-pi null claim.
    receiver = _odd((2, -1, 3, -4))
    mp = mpmath.mp.clone()
    mp.dps = 100
    expected = mp.sin(mp.mpf(1) / 4) / (3 * 1023 * mp.pi)
    port_rows = []
    for k in (1, 2):
        sender = tuple(k * SLOPE * (i - 4) for i in range(9))
        state, rates = _field(
            sender + receiver, (Q(0),) * 9 + (offset,) * 9, bridge=True
        )
        assert tuple(state.form_gradient[i] for i in PORTS) == (0, 0)
        scaled = tuple(rates["form_rates"][i] / Q(1023, 1024) for i in PORTS)
        assert scaled[0].lo > 0 and scaled[1].hi < 0
        for bounds, value in zip(scaled, (expected, -expected)):
            low = mp.mpf(bounds.lo.numerator) / bounds.lo.denominator
            high = mp.mpf(bounds.hi.numerator) / bounds.hi.denominator
            assert low <= value <= high
        assert tuple(rates["phase_rates"][i] for i in PORTS) == (I(0), I(0))
        port_rows.append(scaled)
    assert port_rows[0] == port_rows[1]
    # The offset breaks the zero-origin invariant product argument. Equality
    # here concerns only these initial rows, not later currents or responses.


def test_zero_bridge_gaps_alone_do_not_remove_the_degree_change():
    forms = list(_odd((-4, -3, -2, -1)) * 2)
    forms[3] += 1  # Break oddness while keeping both bridge endpoints at zero.
    phases = (Q(0),) * 18
    isolated, before = _field(tuple(forms), phases)
    joined, after = _field(tuple(forms), phases, bridge=True)
    assert forms[4] == forms[13] == 0
    assert isolated.form_gradient[4] == joined.form_gradient[4] == -1
    assert before["form_rates"][4] == I(Q(1023, 2048))
    assert after["form_rates"][4] == I(Q(341, 1024))
    assert before["phase_rates"][4].hi < after["phase_rates"][4].lo < 0


def test_existing_composition_owner_retains_bridge_admission_separate_from_capture():
    edges, neighbors = _support(9)
    sources = []
    for component, k in enumerate((1, 2)):
        nodes = tuple(range(9 * component, 9 * (component + 1)))
        sources.append(
            _bound_sine_pattern_from_rows(
                reference_node=nodes[0],
                reference_model=MODEL,
                nodes=nodes,
                edges=tuple((nodes[i], nodes[j]) for i, j in edges),
                neighbors=neighbors,
                capacity=(Q(1),) * 9,
                nominal_form=tuple(k * SLOPE * (i - 4) for i in range(9)),
                nominal_phase=(Q(0),) * 9,
                form_error_bounds=(Q(0),) * 9,
                phase_error_bounds=(Q(0),) * 9,
            )
        )
    report = assess_sine_pattern_composition(
        *sources,
        bridge=PORTS,
        observation_time=0,
        edge_turn_offsets=(0,) * 19,
        form_origin_difference=0,
        phase_origin_difference=0,
        work_allowance=0,
    )
    assert report.status == "available"
    assert report.budget_status == "within_allowance"
    assert report.bridge_form_gap_bounds == report.bridge_phase_gap_bounds == I(0)
    assert report.bridge_storage_bounds == I(0)
    assert report.separate_degrees == (2,) * 18
    assert report.joined_degrees == tuple(3 if i in PORTS else 2 for i in range(18))
    assert not report.capture.admitted
    for port in PORTS:
        assert (
            sum(
                report.joined.nominal_form[port] - report.joined.nominal_form[j]
                for j in report.joined.neighbors[port]
            )
            == 0
        )
        _zero_enclosure(report.joined.form_gradient_bounds[port])
        _zero_enclosure(report.joined.form_rate_bounds[port])
        _zero_enclosure(report.joined.phase_rate_bounds[port])
