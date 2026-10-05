"""Causal pressure decomposition from one retained sine-law capture.

Independent nodal and edge formulas check the view without a solver, hidden
minimum replacement, physical bridge or replay of a frozen producer.
"""

import json
import pickle
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_comparison as comparison_owner
from tnfr.physics import relational_sine_mediation as mediation_owner
from tnfr.sdk import export_to_json

MODEL = RelationalExchangeModel(2, phase_domain="regular")


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _inside(interval, value):
    assert _mp(interval.lo) <= value <= _mp(interval.hi)


def _state(*, hidden_capacity=3):
    # The three ports have different original degrees; port 2 also reaches
    # nonport 4. Retaining only the star would change all consumed rates.
    graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4)])
    for node, x, theta, nu in zip(
        range(5),
        (0.25, 1, -0.5, 0.75, -1),
        (-0.25, 0, 0.5, -0.75, 1),
        (hidden_capacity, 1, 2, 0.5, 1.5),
    ):
        graph.nodes[node].update(EPI=x, theta=theta, nu_f=nu, delta_nfr=99)
    graph.graph.update(_t=7, retained={"history": ["unchanged"]})
    return graph


def _capture(graph, *, model=MODEL):
    return comparison_owner.bound_relational_sine_exchange(graph, reference_model=model)


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _independent_field(graph, model):
    """High-precision real nodal rows derived directly from graph incidence."""
    e, w = map(_mp, model.effective_weights)
    beta = _mp(model.storage_scale)
    x = {i: _mp(graph.nodes[i]["EPI"]) for i in graph}
    theta = {i: _mp(graph.nodes[i]["theta"]) for i in graph}
    nu = {i: _mp(graph.nodes[i]["nu_f"]) for i in graph}
    gradient, currents, pressure, fx, ft = {}, {}, {}, {}, {}
    for i in graph:
        gradient[i] = sum(x[i] - x[j] for j in graph[i])
        currents[i] = sum(mp.sin(theta[j] - theta[i]) for j in graph[i])
        pressure[i] = (-e * gradient[i] + w * currents[i] / mp.pi) / graph.degree[i]
        fx[i] = nu[i] * pressure[i]
        ft[i] = w * nu[i] * gradient[i] / (beta * mp.pi * graph.degree[i])
    return x, theta, nu, gradient, currents, pressure, fx, ft


@pytest.fixture(scope="module", autouse=True)
def no_native_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("a retained sine pressure view must not execute native dynamics")

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
    graph = _state()
    full = _capture(graph)
    return graph, full, full.mediated_pressure(mediator=0)


def test_independent_internal_and_environmental_pressure_keep_original_degrees(sample):
    graph, full, report = sample
    assert report.comparison is full
    assert report.ports == (1, 2, 3)
    assert report.port_degrees == (2, 3, 1)
    assert 4 not in report.ports and 4 in report.comparison.nodes
    with mp.workdps(90):
        x, theta, nu, q, currents, pressure, fx, ft = _independent_field(graph, MODEL)
        e, w = map(_mp, MODEL.effective_weights)
        beta = _mp(MODEL.storage_scale)
        for position, node in enumerate(report.ports):
            degree = graph.degree[node]
            gap = theta[node] - theta[0]
            contrast = x[node] - x[0]
            environmental = (-e * contrast - w * mp.sin(gap) / mp.pi) / degree
            internal = (
                sum(
                    -e * (x[node] - x[j]) + w * mp.sin(theta[j] - theta[node]) / mp.pi
                    for j in graph[node]
                    if j != 0
                )
                / degree
            )
            assert report.port_form_contrasts[position] == Q(
                graph.nodes[node]["EPI"]
            ) - Q(graph.nodes[0]["EPI"])
            _inside(report.port_phase_currents[position], mp.sin(gap))
            _inside(report.environmental_pressures[position], environmental)
            _inside(report.internal_pressures[position], internal)
            _inside(full.pressure[full.nodes.index(node)], environmental + internal)
            environmental_phase = nu[node] * w * contrast / (beta * mp.pi * degree)
            internal_phase = (
                nu[node] * w * (q[node] - contrast) / (beta * mp.pi * degree)
            )
            _inside(report.environmental_phase_rates[position], environmental_phase)
            _inside(report.internal_phase_rates[position], internal_phase)
            assert report.pressure_reconstruction_residuals[position].contains(0)
            assert report.phase_rate_reconstruction_residuals[position].contains(0)
        assert report.internal_pressures[2] == I(0)
        assert report.internal_phase_rates[2] == I(0)
        # Treating the surviving port edge as degree one is a different law.
        wrong_degree = -e * (x[1] - x[0]) - w * mp.sin(theta[1] - theta[0]) / mp.pi
        assert wrong_degree < _mp(report.environmental_pressures[0].lo)


@pytest.mark.parametrize(
    "hidden_capacity,e,w", ((3, 0.5, 0.5), (0, 0.5, 0.5), (3, 0, 1))
)
def test_hidden_rates_and_acceleration_retain_moving_ports(hidden_capacity, e, w):
    graph = _state(hidden_capacity=hidden_capacity)
    model = RelationalExchangeModel(
        2, epi_weight=e, phase_weight=w, phase_domain="regular"
    )
    full = _capture(graph, model=model)
    report = full.mediated_pressure(mediator=0)
    with mp.workdps(90):
        x, theta, nu, q, currents, pressure, fx, ft = _independent_field(graph, model)
        k = len(graph[0])
        mean_form = sum(x[node] for node in graph[0]) / k
        mean_rate = sum(fx[node] for node in graph[0]) / k
        contrast = x[0] - mean_form
        gain = nu[0] * _mp(w) / (_mp(model.storage_scale) * mp.pi)
        memory_gain = (
            nu[0] ** 2 * _mp(w) ** 2 / (_mp(model.storage_scale) * mp.pi**2 * k)
        )
        assert report.hidden_form == Q(graph.nodes[0]["EPI"])
        assert (
            report.port_mean_form
            == sum(Q(graph.nodes[node]["EPI"]) for node in graph[0]) / k
        )
        assert report.hidden_form_contrast == report.hidden_form - report.port_mean_form
        _inside(report.hidden_form_rate, fx[0])
        _inside(report.hidden_phase_rate, ft[0])
        _inside(report.port_mean_form_rate, mean_rate)
        _inside(report.hidden_form_contrast_rate, fx[0] - mean_rate)
        _inside(report.hidden_phase_acceleration, gain * (fx[0] - mean_rate))
        _inside(report.phase_gain, gain)
        _inside(report.memory_phase_gain, memory_gain)
        assert report.memory_decay == Q(hidden_capacity) * Q(e)
        # This second-order identity includes the actual changing port mean.
        _inside(
            report.hidden_phase_acceleration
            + report.memory_decay * report.hidden_phase_rate,
            memory_gain * currents[0] - gain * mean_rate,
        )
        if hidden_capacity == 0:
            assert report.hidden_form_rate == report.hidden_phase_rate == I(0)
            assert report.hidden_phase_acceleration == I(0)
            assert report.phase_gain == report.memory_phase_gain == I(0)
            assert report.hidden_loss == 0
            assert abs(pressure[0]) > mp.mpf("0.01")
        else:
            assert abs(mean_rate) > mp.mpf("0.01")
            assert abs(gain * mean_rate) > mp.mpf("0.001")
        if e == 0:
            assert report.hidden_loss == report.memory_decay == 0
            assert report.memory_phase_gain.lo > 0


def test_incident_storage_balance_retains_boundary_work_from_the_full_graph(sample):
    graph, full, report = sample
    with mp.workdps(90):
        x, theta, nu, q, currents, pressure, fx, ft = _independent_field(graph, MODEL)
        beta, e = _mp(MODEL.storage_scale), _mp(MODEL.epi_weight)
        ports = tuple(graph[0])
        form_storage = sum((x[i] - x[0]) ** 2 / 2 for i in ports)
        phase_cost = sum(1 - mp.cos(theta[i] - theta[0]) for i in ports)
        boundary_form = sum((x[i] - x[0]) * fx[i] for i in ports)
        boundary_phase = beta * sum(mp.sin(theta[i] - theta[0]) * ft[i] for i in ports)
        direct_star_rate = sum(
            (x[i] - x[0]) * (fx[i] - fx[0])
            + beta * mp.sin(theta[i] - theta[0]) * (ft[i] - ft[0])
            for i in ports
        )
        hidden_loss = e * nu[0] * q[0] ** 2 / len(ports)
        assert report.hidden_loss == Q(MODEL.epi_weight) * Q(
            graph.nodes[0]["nu_f"]
        ) * full.form_gradient[0] ** 2 / len(ports)
        assert report.incident_form_storage == sum(
            (Q(graph.nodes[i]["EPI"]) - Q(graph.nodes[0]["EPI"])) ** 2 / 2
            for i in ports
        )
        _inside(report.incident_phase_cost, phase_cost)
        _inside(report.incident_storage, form_storage + beta * phase_cost)
        _inside(report.boundary_form_work, boundary_form)
        _inside(report.boundary_phase_work, boundary_phase)
        _inside(report.boundary_work, boundary_form + boundary_phase)
        _inside(report.incident_storage_rate, direct_star_rate)
        _inside(
            report.incident_storage_rate, boundary_form + boundary_phase - hidden_loss
        )
        assert abs(boundary_form + boundary_phase) > mp.mpf("0.01")
        _inside(full.storage_rate, -_mp(full.continuous_loss))
        assert abs(direct_star_rate + _mp(full.continuous_loss)) > mp.mpf("0.01")
    assert report.balance_residual.contains(0)
    assert report.balance_residual.width > 0


def test_per_port_work_owns_signed_boundary_sums_without_recapturing_the_source(sample):
    graph, full, report = sample
    before = pickle.dumps(full)
    assert isinstance(report.port_boundary_form_work, tuple)
    assert isinstance(report.port_boundary_phase_work, tuple)
    assert isinstance(report.port_boundary_work, tuple)
    with mp.workdps(90):
        x, theta, _, _, _, _, fx, ft = _independent_field(graph, MODEL)
        beta = _mp(MODEL.storage_scale)
        for p, node in enumerate(report.ports):
            form = (x[node] - x[0]) * fx[node]
            phase = beta * mp.sin(theta[node] - theta[0]) * ft[node]
            _inside(report.port_boundary_form_work[p], form)
            _inside(report.port_boundary_phase_work[p], phase)
            _inside(report.port_boundary_work[p], form + phase)
            assert report.port_boundary_work[p] == (
                report.port_boundary_form_work[p] + report.port_boundary_phase_work[p]
            )
    assert report.boundary_form_work == sum(report.port_boundary_form_work, I(0))
    assert report.boundary_phase_work == sum(report.port_boundary_phase_work, I(0))
    assert report.boundary_work == sum(report.port_boundary_work, I(0))
    assert pickle.dumps(full) == before
    with pytest.raises(FrozenInstanceError):
        report.port_boundary_work = ()


def test_one_edge_receiver_input_is_negative_star_port_work_with_full_degree():
    graph = nx.Graph([(0, 1), (0, 3), (1, 2)])
    for i, x, theta, nu in (
        (0, Q(3, 4), Q(-1, 4), 1),
        (1, Q(-1, 2), Q(1, 2), Q(3, 2)),
        (2, Q(1, 4), Q(-1, 2), Q(1, 2)),
        (3, Q(-1, 4), Q(1, 4), 2),
    ):
        graph.nodes[i].update(EPI=x, theta=theta, nu_f=nu)
    model = RelationalExchangeModel(Q(3, 2), phase_domain="regular")
    full = _capture(graph, model=model)
    mediated = full.mediated_pressure(mediator=0)
    port = mediated.ports.index(1)
    with mp.workdps(100):
        x, theta, nu, q, _, _, fx, ft = _independent_field(graph, model)
        beta, e = _mp(model.storage_scale), _mp(model.epi_weight)
        regional_rate = (x[1] - x[2]) * (fx[1] - fx[2]) + beta * mp.sin(
            theta[1] - theta[2]
        ) * (ft[1] - ft[2])
        # FULL q1 includes the external edge and full d1=2; replacing it
        # with a degree-one isolated receiver silently changes the law.
        regional_loss = sum(e * nu[i] * q[i] ** 2 / graph.degree[i] for i in (1, 2))
        regional_input = (x[0] - x[1]) * fx[1] + beta * mp.sin(
            theta[0] - theta[1]
        ) * ft[1]
        assert mp.almosteq(regional_rate, regional_input - regional_loss)
        _inside(-mediated.port_boundary_work[port], regional_input)
        _inside(-mediated.port_boundary_form_work[port], (x[0] - x[1]) * fx[1])
        _inside(
            -mediated.port_boundary_phase_work[port],
            beta * mp.sin(theta[0] - theta[1]) * ft[1],
        )
        assert abs(regional_input) > mp.mpf("0.01")
        assert abs(regional_rate) > mp.mpf("0.01")


def test_per_port_work_appended_fields_preserve_prior_constructor_and_json(
    sample, tmp_path
):
    report = sample[2]
    payload = report.to_dict()
    assert len(payload["report"]["port_boundary_work"]) == report.port_count
    path = tmp_path / "port-work.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    previous = fields(mediation_owner.SineMediatedPressure)[:-3]
    legacy = mediation_owner.SineMediatedPressure(
        *(getattr(report, field.name) for field in previous)
    )
    assert legacy.port_boundary_form_work is None
    assert legacy.port_boundary_phase_work is None
    assert legacy.port_boundary_work is None
    assert legacy.boundary_work == report.boundary_work
    assert legacy.to_dict()["report"]["port_boundary_work"] is None


@pytest.mark.parametrize("mu", (Q(1, 2), Q(3)))
def test_equal_visible_state_with_distinct_hidden_phase_changes_actual_pressure(mu):
    reports = []
    for hidden_phase in (0, 0.5):
        graph = nx.star_graph(2)
        for node, x, theta, nu in (
            (0, 0.25, hidden_phase, float(mu)),
            (1, 1, 0, 1),
            (2, 0, 0.5, 2),
        ):
            graph.nodes[node].update(EPI=x, theta=theta, nu_f=nu, delta_nfr=99)
        reports.append(_capture(graph).mediated_pressure(mediator=0))
    first, second = reports
    with mp.workdps(90):
        signal = _mp(MODEL.phase_weight) * mp.sin(mp.mpf(1) / 2) / mp.pi
        for node, capacity in ((1, 1), (2, 2)):
            _inside(
                second.comparison.form_rates[node] - first.comparison.form_rates[node],
                capacity * signal,
            )
            assert (
                first.comparison.phase_rates[node]
                == second.comparison.phase_rates[node]
            )
        _inside(second.hidden_form_rate - first.hidden_form_rate, -_mp(mu) * signal)
        _inside(
            first.hidden_phase_rate,
            -_mp(mu) * _mp(MODEL.phase_weight) / (4 * _mp(MODEL.storage_scale) * mp.pi),
        )
    assert first.hidden_phase_rate == second.hidden_phase_rate


def test_view_reuses_the_capture_without_mutation_native_calls_or_recapture(
    monkeypatch,
):
    graph = _state()
    before = _snapshot(graph)
    full = _capture(graph)
    captured = pickle.dumps(full, protocol=5)

    def forbidden(*args, **kwargs):
        pytest.fail("the retained view must not recapture or minimize its hidden state")

    monkeypatch.setattr(comparison_owner, "_capture_sine_state", forbidden)
    monkeypatch.setattr(mediation_owner, "_capture_sine_state", forbidden)
    monkeypatch.setattr(mediation_owner, "bound_relational_sine_mediation", forbidden)
    report = full.mediated_pressure(mediator=0)
    assert report.comparison is full
    assert _snapshot(graph) == before
    assert pickle.dumps(full, protocol=5) == captured
    graph.nodes[0]["EPI"] = float("nan")
    assert full.mediated_pressure(mediator=0).hidden_form == Q(1, 4)


@pytest.mark.parametrize("kind", ("unresolved", "negative_real"))
def test_nonpositive_or_unresolved_resultant_uses_no_normalization(monkeypatch, kind):
    # Four represented phases give a resultant inside a zero-containing
    # enclosure. This is unresolved cancellation, not a claim of exact pi.
    with mp.workdps(90):
        outer = Q(201, 64)
        inner = float(mp.pi - _mp(outer))
    phases = (
        (0, inner, -inner, float(outer), -float(outer))
        if kind == "unresolved"
        else (0, 2, -2, 2, -2)
    )
    graph = nx.star_graph(4)
    for node, x, theta in zip(
        range(5),
        (0.25, 1, -0.5, 0.75, -1),
        phases,
    ):
        graph.nodes[node].update(EPI=x, theta=theta, nu_f=1, delta_nfr=99)
    full = _capture(graph)
    if kind == "unresolved":
        assert all(component.contains(0) for component in full.relative_resultant[0])
    else:
        assert full.relative_resultant[0][0].hi < 0
        assert full.relative_resultant[0][1].contains(0)

    def forbidden(*args, **kwargs):
        pytest.fail("a causal view must not divide by a hidden resultant magnitude")

    monkeypatch.setattr(mediation_owner, "sqrt", forbidden)
    monkeypatch.setattr(mediation_owner, "_unit_component", forbidden)
    report = full.mediated_pressure(mediator=0)
    with mp.workdps(90):
        x, theta, nu, q, currents, pressure, fx, ft = _independent_field(graph, MODEL)
        _inside(report.hidden_form_rate, fx[0])
        _inside(report.hidden_phase_rate, ft[0])
    assert report.balance_residual.contains(0)


@pytest.mark.parametrize("mediator", ("absent", 4))
def test_mediator_requires_membership_and_at_least_two_ports(sample, mediator):
    with pytest.raises(ValueError, match="mediator|ports"):
        sample[1].mediated_pressure(mediator=mediator)


def test_export_retains_exact_source_and_explicit_law_identity(sample):
    report = sample[2]
    encoded = report.to_dict()
    assert encoded["schema"] == "tnfr.relational-sine-mediated-pressure.v1"
    roundtrip = json.loads(json.dumps(encoded, allow_nan=False))
    assert roundtrip["report"]["mediator"] == 0
    assert roundtrip["report"]["ports"] == [1, 2, 3]
    assert roundtrip["report"]["comparison"]["law"] == sample[1].law
    assert (
        roundtrip["report"]["hidden_form"]
        == roundtrip["report"]["comparison"]["epi"][0]
    )
    assert "scope" in roundtrip["report"]


@dataclass(frozen=True)
class _OpaqueLabel:
    name: str


@pytest.mark.parametrize("field", ("mediator", "ports", "comparison"))
def test_export_rejects_unsupported_node_labels_before_projection(sample, field):
    report = sample[2]
    opaque = _OpaqueLabel("node")
    if field == "comparison":
        changed = replace(
            report.comparison, nodes=(opaque, *report.comparison.nodes[1:])
        )
    elif field == "ports":
        changed = (opaque, *report.ports[1:])
    else:
        changed = opaque
    with pytest.raises(TypeError, match="JSON scalar node labels"):
        replace(report, **{field: changed}).to_dict()
