"""Independent stationary-reduction, collective-response and scope controls."""

import json
import math
import pickle
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import relational_sine_mediation as owner
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

MODEL = RelationalExchangeModel(1, phase_domain="regular")
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _state(
    graph=None,
    *,
    forms=(3, 1, -0.5, 0.25, -1),
    phases=(0.75, 0.25, -0.5, 0.875, 0.125),
    capacities=(4, 1, 2, 3, 0.5),
):
    graph = (
        nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4)]) if graph is None else graph
    )
    for node, x, theta, nu in zip(graph, forms, phases, capacities):
        graph.nodes[node].update(EPI=x, theta=theta, nu_f=nu, delta_nfr=999)
    graph.graph.update(GAMMA={"type": "none"}, untouched={"list": [1, 2]})
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _inside(interval, value):
    assert _mp(interval.lo) <= value <= _mp(interval.hi)


@pytest.fixture(scope="module", autouse=True)
def no_native_runtime():
    def forbidden(*args, **kwargs):
        pytest.fail("a detached sine reduction must not invoke the native runtime")

    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "_field",
            "evaluate_relational_exchange",
            "_advance",
            "step_relational_exchange",
        ):
            patch.setattr(relational, name, forbidden)
        yield


@pytest.fixture(scope="module")
def sample():
    return owner.bound_relational_sine_mediation(
        _state(), mediator=0, reference_model=MODEL
    )


def test_independent_high_precision_stationary_minimum_field_and_work(sample):
    graph = _state()
    with mp.workdps(90):
        x = {i: _mp(graph.nodes[i]["EPI"]) for i in graph}
        theta = {i: _mp(graph.nodes[i]["theta"]) for i in graph}
        nu = {i: _mp(graph.nodes[i]["nu_f"]) for i in graph}
        ports = tuple(graph[0])
        x[0] = sum(x[i] for i in ports) / len(ports)
        z = sum(mp.exp(1j * theta[i]) for i in ports)
        radius = abs(z)
        theta[0] = mp.arg(z)
        assert sample.hidden_form == Q(1, 4)
        _inside(sample.resultant_magnitude, radius)
        _inside(sample.hidden_phase_curvature, radius)
        _inside(
            sample.minimum_phase_relative_to_first_port[0], mp.cos(theta[0] - theta[1])
        )
        _inside(
            sample.minimum_phase_relative_to_first_port[1], mp.sin(theta[0] - theta[1])
        )
        storage = sum(
            (x[i] - x[j]) ** 2 / 2 + 1 - mp.cos(theta[i] - theta[j])
            for i, j in graph.edges()
        )
        _inside(sample.storage, storage)
        total_loss = mp.mpf(0)
        for position, i in enumerate(sample.nodes):
            degree = graph.degree(i)
            q = sum(x[i] - x[j] for j in graph[i])
            current = sum(mp.sin(theta[j] - theta[i]) for j in graph[i])
            real = sum(mp.cos(theta[j] - theta[i]) for j in graph[i])
            source = current / (mp.pi * degree)
            pressure = (-q / degree + source) / 2
            form_rate = nu[i] * pressure
            phase_rate = nu[i] * q / (2 * mp.pi * degree)
            assert _mp(sample.form_gradient[position]) == q
            _inside(sample.relative_resultant[position][0], real)
            _inside(sample.relative_resultant[position][1], current)
            _inside(sample.phase_sources[position], source)
            _inside(sample.pressure[position], pressure)
            _inside(sample.form_rates[position], form_rate)
            _inside(sample.phase_rates[position], phase_rate)
            _inside(sample.form_work[position], q * form_rate)
            _inside(sample.phase_work[position], -current * phase_rate)
            assert sample.node_balance_residual[position].contains(0)
            total_loss += nu[i] * q**2 / (2 * degree)
        assert mp.almosteq(_mp(sample.continuous_loss), total_loss)
        _inside(sample.storage_rate, -total_loss)
        assert sample.balance_residual.contains(0)
        assert sample.balance_residual.width > 0
        assert sample.degrees == (2, 3, 1, 1)
        assert sample.edges == ((1, 2), (2, 4))


def test_two_ports_recover_exact_midpoint_and_sine_half_gap():
    graph = _state(
        nx.star_graph(2),
        forms=(9, 1, -1),
        phases=(1.5, -0.5, 0.5),
        capacities=(7, 1, 1),
    )
    report = owner.bound_relational_sine_mediation(
        graph, mediator=0, reference_model=MODEL
    )
    assert report.hidden_form == 0
    assert report.form_gradient == (1, -1)
    assert report.form_storage == 1
    with mp.workdps(90):
        _inside(report.resultant_magnitude, 2 * mp.cos(mp.mpf("0.5")))
        _inside(report.phase_storage, 2 - 2 * mp.cos(mp.mpf("0.5")))
        for position, sign in ((0, 1), (1, -1)):
            _inside(
                report.phase_sources[position], sign * mp.sin(mp.mpf("0.5")) / mp.pi
            )
    assert report.hidden_form_tracking_defect.contains(0)
    assert report.hidden_phase_tracking_defect.contains(0)


def test_three_port_reduction_has_nonadditive_collective_response():
    def rate(s, t):
        graph = _state(
            nx.star_graph(3), forms=(0,) * 4, phases=(0, 0, s, t), capacities=(1,) * 4
        )
        return owner.bound_relational_sine_mediation(
            graph, mediator=0, reference_model=MODEL
        ).form_rates[0]

    # Any f(s)+g(t) has a zero mixed rectangle. The stationary hidden
    # direction responds to both ports jointly, despite a pairwise fine law.
    mixed = rate(0.625, 0.625) - rate(0.625, 0.5) - rate(0.5, 0.625) + rate(0.5, 0.5)
    assert mixed.hi < Q(-1, 10**7)


def test_negative_absolute_resultant_has_a_minimum_without_an_arg_chart():
    # Two port phases +/-2 have a strictly negative real absolute resultant.
    # Its circular minimum exists even though principal Arg is on its cut.
    graph = _state(
        nx.star_graph(2),
        forms=(0, 1, -1),
        phases=(0, 2, -2),
        capacities=(1,) * 3,
    )
    report = owner.bound_relational_sine_mediation(
        graph, mediator=0, reference_model=MODEL
    )
    with mp.workdps(90):
        _inside(report.resultant_magnitude, -2 * mp.cos(2))
        _inside(report.phase_sources[0], mp.sin(2) / mp.pi)
        _inside(report.phase_sources[1], -mp.sin(2) / mp.pi)
    assert report.hidden_phase_curvature.lo > 0
    assert report.balance_residual.contains(0)


def test_stationary_minimum_is_not_generally_an_invariant_manifold():
    graph = _state(
        nx.star_graph(2), forms=(0.5, 1, 0), phases=(0, 0, 0), capacities=(3, 1, 2)
    )
    report = owner.bound_relational_sine_mediation(
        graph, mediator=0, reference_model=MODEL
    )
    full = bound_relational_sine_exchange(graph, reference_model=MODEL)
    assert full.form_rates[0] == full.phase_rates[0] == I(0)
    assert report.form_rates == (I(Q(-1, 4)), I(Q(1, 2)))
    assert report.hidden_form_tracking_defect == I(Q(-1, 8))
    with mp.workdps(90):
        _inside(report.hidden_phase_tracking_defect, 1 / (8 * mp.pi))
    assert report.hidden_phase_tracking_defect.lo > 0


def test_frozen_ports_recover_the_damped_pendulum_in_fast_time():
    # At held symmetric port phases +/-a, the hidden minimum is phase zero.
    # Only hidden capacity is active, so the ports are genuinely fixed.
    beta, e, w, mu = Q(2), Q(1, 4), Q(3, 4), Q(3)
    u, v, a = Q(3, 4), Q(3, 8), Q(1, 2)
    model = RelationalExchangeModel(
        float(beta), epi_weight=float(e), phase_weight=float(w), phase_domain="regular"
    )
    graph = _state(
        nx.star_graph(2),
        forms=(float(u), -1, 1),
        phases=(float(v), -float(a), float(a)),
        capacities=(float(mu), 0, 0),
    )
    full = bound_relational_sine_exchange(graph, reference_model=model)
    u_fast = full.form_rates[0] / mu
    v_fast = full.phase_rates[0] / mu
    v_acceleration = (w / (beta * pi_interval())) * u_fast
    assert full.form_rates[1:] == full.phase_rates[1:] == (I(0), I(0))
    assert full.continuous_loss / mu == e * 2 * u**2
    with mp.workdps(90):
        r = mp.cos(_mp(a))
        expected_u = -_mp(e * u) - _mp(w) * r * mp.sin(_mp(v)) / mp.pi
        expected_v = _mp(w * u / beta) / mp.pi
        expected_acceleration = (
            -_mp(e) * expected_v - _mp(w**2 / beta) * r * mp.sin(_mp(v)) / mp.pi**2
        )
        _inside(u_fast, expected_u)
        _inside(v_fast, expected_v)
        _inside(v_acceleration, expected_acceleration)


def test_source_hidden_state_and_capacity_are_retained_without_selecting_them(sample):
    graph = _state()
    graph.nodes[0].update(EPI=-20, theta=-2, nu_f=11)
    report = owner.bound_relational_sine_mediation(
        graph, mediator=0, reference_model=MODEL
    )
    for field in (
        "hidden_form",
        "phase_sources",
        "form_rates",
        "phase_rates",
        "storage",
        "continuous_loss",
    ):
        assert getattr(report, field) == getattr(sample, field)
    assert report.source_epi[0] == -20
    assert report.source_phase[0] == -2
    assert report.hidden_capacity == 11


def test_zero_visible_capacity_freezes_both_declared_rows():
    graph = _state()
    graph.nodes[1]["nu_f"] = 0
    report = owner.bound_relational_sine_mediation(
        graph, mediator=0, reference_model=MODEL
    )
    assert report.form_rates[0] == report.phase_rates[0] == I(0)
    assert report.dissipation[0] == 0
    assert not report.pressure[0].contains(0)


def test_offsets_relabeling_and_capture_are_detached(sample):
    graph = _state()
    for node in graph:
        graph.nodes[node]["EPI"] += 8
        graph.nodes[node]["theta"] += 2
    graph = nx.relabel_nodes(graph, {0: "hidden", 1: ("port", 1)})
    before = _snapshot(graph)
    report = owner.bound_relational_sine_mediation(
        graph, mediator="hidden", reference_model=MODEL
    )
    assert _snapshot(graph) == before
    assert report.hidden_form == sample.hidden_form + 8
    assert report.ports == (("port", 1), 2, 3)
    assert report.to_dict()["report"]["ports"][0] == ["port", 1]
    for field in (
        "phase_sources",
        "form_rates",
        "phase_rates",
        "storage",
        "continuous_loss",
    ):
        assert getattr(report, field) == getattr(sample, field)


@pytest.mark.parametrize("model", [None, object(), RelationalExchangeModel(1)])
def test_requires_explicit_regular_coefficient_reference(model):
    with pytest.raises(ValueError, match="regular reference"):
        owner.bound_relational_sine_mediation(
            _state(), mediator=0, reference_model=model
        )


@pytest.mark.parametrize(
    "change", ["absent", "single_port", "zero_hidden_capacity", "unresolved_resultant"]
)
def test_reduction_domain_failures_are_explicit(change):
    graph, mediator = _state(), 0
    if change == "absent":
        mediator = "absent"
    elif change == "single_port":
        mediator = 4
    elif change == "zero_hidden_capacity":
        graph.nodes[0]["nu_f"] = 0
    else:
        graph = _state(
            nx.star_graph(2), forms=(0,) * 3, phases=(0, 0, 1e300), capacities=(1,) * 3
        )
    before = _snapshot(graph)
    with pytest.raises(ValueError, match="mediator|ports|hidden capacity|resultant"):
        owner.bound_relational_sine_mediation(
            graph, mediator=mediator, reference_model=MODEL
        )
    assert _snapshot(graph) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("EPI", True),
        ("EPI", float("nan")),
        ("theta", False),
        ("theta", float("inf")),
        ("nu_f", -1),
    ],
)
def test_even_replaced_hidden_state_requires_authoritative_admission(key, value):
    graph = _state()
    graph.nodes[0][key] = value
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        owner.bound_relational_sine_mediation(graph, mediator=0, reference_model=MODEL)
    assert _snapshot(graph) == before


def test_missing_hidden_coordinate_is_not_invented():
    graph = _state()
    del graph.nodes[0]["theta"]
    with pytest.raises(ADMISSION_ERRORS):
        owner.bound_relational_sine_mediation(graph, mediator=0, reference_model=MODEL)


@pytest.mark.parametrize(
    "change", ["directed", "nonunit", "loop", "disconnected", "gamma"]
)
def test_shared_support_and_input_admission(change):
    graph = _state()
    if change == "directed":
        graph = graph.to_directed()
    elif change == "nonunit":
        graph.edges[0, 1]["weight"] = 2
    elif change == "loop":
        graph.add_edge(0, 0)
    elif change == "disconnected":
        graph.remove_edge(2, 4)
    else:
        graph.graph["GAMMA"] = {"type": "constant", "value": 1}
    with pytest.raises(ADMISSION_ERRORS):
        owner.bound_relational_sine_mediation(graph, mediator=0, reference_model=MODEL)


def test_near_cancellation_preserves_unit_bounds_and_independent_truth():
    # Represented pi is not mathematical pi. This preparation has a small
    # nonzero resultant; dependency-aware range intersections remain valid.
    angle = math.pi
    graph = _state(
        nx.star_graph(2),
        forms=(0, 1, -1),
        phases=(0, 0, angle),
        capacities=(1,) * 3,
    )
    report = owner.bound_relational_sine_mediation(
        graph, mediator=0, reference_model=MODEL
    )
    assert 0 < report.resultant_magnitude.lo <= report.resultant_magnitude.hi <= 2
    for components in (
        report.minimum_phase_relative_to_first_port,
        *report.relative_resultant,
    ):
        assert all(component.subset_of(I(-1, 1)) for component in components)
    with mp.workdps(90):
        z = 1 + mp.exp(1j * _mp(angle))
        radius, minimum = abs(z), mp.arg(z)
        _inside(report.resultant_magnitude, radius)
        _inside(report.phase_storage, 2 - radius)
        for position, phase in enumerate((0, angle)):
            _inside(
                report.relative_resultant[position][0], mp.cos(minimum - _mp(phase))
            )
            _inside(
                report.relative_resultant[position][1], mp.sin(minimum - _mp(phase))
            )
            _inside(
                report.phase_sources[position], mp.sin(minimum - _mp(phase)) / mp.pi
            )
    assert report.balance_residual.contains(0)


@dataclass(frozen=True)
class _OpaqueLabel:
    name: str


@pytest.mark.parametrize(
    "field", ["source_nodes", "nodes", "ports", "mediator", "source_edges", "edges"]
)
def test_export_admits_labels_before_generic_dataclass_projection(sample, field):
    label = _OpaqueLabel("node")
    if field == "mediator":
        changed = label
    elif field.endswith("edges"):
        changed = ((label, 1), *getattr(sample, field)[1:])
    else:
        changed = (label, *getattr(sample, field)[1:])
    report = replace(sample, **{field: changed})
    with pytest.raises(TypeError, match="JSON scalar node labels"):
        report.to_dict()


@pytest.mark.parametrize("label", [Q(1, 3), float("inf"), float("nan")])
def test_export_rejects_non_json_mediator_labels(sample, label):
    with pytest.raises(TypeError, match="JSON scalar node labels"):
        replace(sample, mediator=label).to_dict()


def test_export_preserves_source_and_conditional_scope(sample):
    payload = sample.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["schema"] == "tnfr.relational-sine-mediation.v1"
    assert payload["report"]["hidden_form"] == {"numerator": 1, "denominator": 4}
    assert payload["report"]["source_epi"][0] == {"numerator": 3, "denominator": 1}
    assert set(payload["report"]["resultant_magnitude"]) == {"lo", "hi"}
    assert "no_native_arg_pressure_derivation" in " ".join(sample.scope)
    assert "invariant_manifold" in " ".join(sample.scope)
    assert not hasattr(sample, "hidden_phase")


@pytest.mark.parametrize(
    "phases",
    [
        (0.375, -0.5, 0.5),
        (0.375, 0, 3.140625),
        (-0.25, 2, -2),
    ],
)
def test_cartesian_hidden_phase_reproduces_full_rates_without_a_resultant_angle(phases):
    # The second preparation probes cancellation with represented rational
    # radians; it is not an exact antipodal pair or a mathematical zero.
    graph = _state(
        nx.star_graph(2),
        forms=(0.75, -1, 0.5),
        phases=phases,
        capacities=(3, 1, 2),
    )
    full = bound_relational_sine_exchange(graph, reference_model=MODEL)
    with mp.workdps(90):
        hidden, left, right = map(_mp, full.phase)
        c, s = mp.cos(hidden), mp.sin(hidden)
        a = mp.cos(left) + mp.cos(right)
        b = mp.sin(left) + mp.sin(right)
        y, x_left, x_right = map(_mp, full.epi)
        mu = _mp(full.capacity[0])
        u = y - (x_left + x_right) / 2
        expected_form = mu * (-u / 2 + (b * c - a * s) / (4 * mp.pi))
        omega = mu * u / (2 * mp.pi)
        _inside(full.form_rates[0], expected_form)
        _inside(full.phase_rates[0], omega)
        # The phasor is a faithful circular coordinate. Its tangent preserves
        # the unit circle and differentiates the existing phase row directly.
        c_dot, s_dot = -omega * s, omega * c
        assert mp.almosteq(c * c_dot + s * s_dot, 0, abs_eps=mp.mpf("1e-85"))
        assert mp.almosteq(c * s_dot - s * c_dot, omega)
        if phases[1:] == (0, 3.140625):
            assert 0 < abs(mp.mpc(a, b)) < mp.mpf("0.001")
        if phases[1:] == (2, -2):
            assert a < 0
            assert b == 0


def test_uneliminated_hidden_orientation_changes_visible_rates_at_zero_form():
    reports, reductions = [], []
    for hidden_phase in (-0.5, 0.5):
        graph = _state(
            nx.star_graph(2),
            forms=(0, 0, 0),
            phases=(hidden_phase, 0, 3.140625),
            capacities=(3, 1, 2),
        )
        reports.append(bound_relational_sine_exchange(graph, reference_model=MODEL))
        reductions.append(
            owner.bound_relational_sine_mediation(
                graph, mediator=0, reference_model=MODEL
            )
        )
    negative, positive = reports
    assert negative.phase_rates == positive.phase_rates == (I(0), I(0), I(0))
    assert negative.form_rates[1].hi < 0 < positive.form_rates[1].lo
    assert positive.form_rates[2].hi < 0 < negative.form_rates[2].lo
    with mp.workdps(90):
        expected = mp.sin(mp.mpf("0.5")) / (2 * mp.pi)
        _inside(positive.form_rates[1], expected)
        _inside(negative.form_rates[1], -expected)
    # A stationary replacement sees identical visible data and consequently
    # the same reduced field. It cannot represent both supplied hidden states.
    assert reductions[0].form_rates == reductions[1].form_rates
    assert reductions[0].source_phase[0] != reductions[1].source_phase[0]
    assert reductions[0].resultant_magnitude.hi < Q(1, 1000)
