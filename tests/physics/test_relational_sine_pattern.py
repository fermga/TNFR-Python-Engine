"""Relative pattern observations preserve symmetry, uncertainty and drift."""

import json
import pickle
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
from types import SimpleNamespace

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_observation import infer_relational_sine_hidden_state
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.sdk import export_to_json

MODEL = RelationalExchangeModel(1, phase_domain="regular")
LABELS = ("left", "center", "right")
ERRORS = (Q(1, 64), Q(1, 32), Q(1, 128))
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _graph(*, capacity=(1, 0, 2)):
    graph = nx.path_graph(LABELS)
    for node, x, phase, nu in zip(graph, (1, -0.5, 0.25), (0, 0.5, -0.25), capacity):
        graph.nodes[node].update(EPI=x, theta=phase, nu_f=nu, delta_nfr=999)
    graph.graph.update(GAMMA={"type": "none"}, unrelated={"history": [1, 2]})
    return graph


def _report(
    graph=None, *, reference="left", form_errors=(0, 0, 0), phase_errors=(0, 0, 0)
):
    return bound_relational_sine_pattern(
        _graph() if graph is None else graph,
        reference_node=reference,
        reference_model=MODEL,
        form_error_bounds=form_errors,
        phase_error_bounds=phase_errors,
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    assert _mp(bound.lo) <= value <= _mp(bound.hi)


def _oracle(graph, form=None, phase=None):
    nodes = tuple(graph)
    form = tuple(Q(graph.nodes[i]["EPI"]) for i in nodes) if form is None else form
    phase = tuple(Q(graph.nodes[i]["theta"]) for i in nodes) if phase is None else phase
    x, theta = dict(zip(nodes, map(_mp, form))), dict(zip(nodes, map(_mp, phase)))
    capacity = {i: _mp(graph.nodes[i]["nu_f"]) for i in nodes}
    e, w = map(_mp, MODEL.effective_weights)
    beta = _mp(MODEL.storage_scale)
    gradient = {i: sum(x[i] - x[j] for j in graph[i]) for i in nodes}
    current = {i: sum(mp.sin(theta[j] - theta[i]) for j in graph[i]) for i in nodes}
    pressure = {
        i: (-e * gradient[i] + w * current[i] / mp.pi) / graph.degree[i] for i in nodes
    }
    rates = {i: capacity[i] * pressure[i] for i in nodes}
    phase_rates = {
        i: capacity[i] * w * gradient[i] / (beta * mp.pi * graph.degree[i])
        for i in nodes
    }
    form_storage = sum((x[i] - x[j]) ** 2 / 2 for i, j in graph.edges)
    phase_storage = sum(1 - mp.cos(theta[j] - theta[i]) for i, j in graph.edges)
    loss = sum(e * capacity[i] * gradient[i] ** 2 / graph.degree[i] for i in nodes)
    return (
        gradient,
        current,
        pressure,
        rates,
        phase_rates,
        form_storage,
        phase_storage,
        loss,
    )


@pytest.mark.parametrize("reference", LABELS)
def test_independent_static_field_and_reference_rows(reference):
    graph = _graph()
    report = _report(graph, reference=reference)
    with mp.workdps(90):
        q, s, p, v, omega, ed, ev, loss = _oracle(graph)
        for i, node in enumerate(graph):
            _contains(report.form_gradient_bounds[i], q[node])
            _contains(report.phase_current_bounds[i], s[node])
            _contains(report.pressure_bounds[i], p[node])
            _contains(report.form_rate_bounds[i], v[node])
            _contains(report.phase_rate_bounds[i], omega[node])
            _contains(report.relative_form_rate_bounds[i], v[node] - v[reference])
            _contains(
                report.relative_phase_rate_bounds[i], omega[node] - omega[reference]
            )
        _contains(report.form_storage_bounds, ed)
        _contains(report.phase_storage_bounds, ev)
        _contains(report.storage_bounds, ed + ev)
        _contains(report.continuous_loss_bounds, loss)
        _contains(report.storage_rate_bounds, -loss)
    anchor = LABELS.index(reference)
    assert report.relative_form_bounds[anchor] == I(0)
    assert report.relative_phase_bounds[anchor] == I(0)
    assert report.relative_form_rate_bounds[anchor] == I(0)
    assert report.relative_phase_rate_bounds[anchor] == I(0)
    assert report.balance_residual_bounds.contains(0)


def test_zero_capacity_is_absolute_freezing_not_relative_freezing():
    active_reference = _report(reference="left")
    assert active_reference.form_rate_bounds[1] == I(0)
    assert active_reference.phase_rate_bounds[1] == I(0)
    assert not active_reference.relative_form_rate_bounds[1].contains(0)
    assert not active_reference.relative_phase_rate_bounds[1].contains(0)
    frozen_reference = _report(reference="center")
    assert (
        frozen_reference.relative_form_rate_bounds == frozen_reference.form_rate_bounds
    )
    assert (
        frozen_reference.relative_phase_rate_bounds
        == frozen_reference.phase_rate_bounds
    )


def test_common_form_and_phase_shifts_leave_all_pattern_fields_unchanged():
    graph = _graph()
    shifted = graph.copy()
    for node in shifted:
        shifted.nodes[node]["EPI"] += 7
        shifted.nodes[node]["theta"] -= 3
    original = _report(graph, form_errors=ERRORS, phase_errors=ERRORS)
    translated = _report(shifted, form_errors=ERRORS, phase_errors=ERRORS)
    for name in (
        "relative_form_bounds",
        "relative_phase_bounds",
        "edge_form_gap_bounds",
        "edge_phase_gap_bounds",
        "form_gradient_bounds",
        "phase_current_bounds",
        "pressure_bounds",
        "form_rate_bounds",
        "phase_rate_bounds",
        "relative_form_rate_bounds",
        "relative_phase_rate_bounds",
        "storage_bounds",
        "continuous_loss_bounds",
    ):
        assert getattr(original, name) == getattr(translated, name)


def test_edge_intervals_cancel_anchor_error_before_enclosure():
    graph = _graph()
    report = _report(graph, reference="left", form_errors=ERRORS, phase_errors=ERRORS)
    positions = {node: i for i, node in enumerate(graph)}
    for pair, form, phase in zip(
        report.edges, report.edge_form_gap_bounds, report.edge_phase_gap_bounds
    ):
        left, right = pair
        radius = ERRORS[positions[left]] + ERRORS[positions[right]]
        form_center = Q(graph.nodes[right]["EPI"]) - Q(graph.nodes[left]["EPI"])
        phase_center = Q(graph.nodes[right]["theta"]) - Q(graph.nodes[left]["theta"])
        assert form == I(form_center - radius, form_center + radius)
        assert phase == I(phase_center - radius, phase_center + radius)
    # Edge center--right excludes the reference and must not pay its noise twice.
    index = report.edges.index(("center", "right"))
    direct = report.edge_form_gap_bounds[index]
    duplicated = report.relative_form_bounds[2] - report.relative_form_bounds[1]
    assert direct.hi - direct.lo < duplicated.hi - duplicated.lo


def test_residual_sensor_errors_remain_enclosed_after_unknown_common_offsets():
    graph = _graph()
    report = _report(graph, reference="center", form_errors=ERRORS, phase_errors=ERRORS)
    form = tuple(Q(graph.nodes[i]["EPI"]) for i in graph)
    phase = tuple(Q(graph.nodes[i]["theta"]) for i in graph)
    with mp.workdps(90):
        for signs in product((-1, 1), repeat=3):
            true_form = tuple(
                value - 7 - sign * error
                for value, sign, error in zip(form, signs, ERRORS)
            )
            true_phase = tuple(
                value + 3 + sign * error
                for value, sign, error in zip(phase, signs, ERRORS)
            )
            _, _, _, v, omega, ed, ev, loss = _oracle(graph, true_form, true_phase)
            for i, node in enumerate(graph):
                _contains(
                    report.relative_form_bounds[i], _mp(true_form[i] - true_form[1])
                )
                _contains(
                    report.relative_phase_bounds[i], _mp(true_phase[i] - true_phase[1])
                )
                _contains(report.relative_form_rate_bounds[i], v[node] - v["center"])
                _contains(
                    report.relative_phase_rate_bounds[i], omega[node] - omega["center"]
                )
            _contains(report.storage_bounds, ed + ev)
            _contains(report.continuous_loss_bounds, loss)


def test_independent_regional_offsets_change_crossing_edge_dynamics():
    graph = _graph(capacity=(1, 1, 1))
    original = _report(graph)
    graph.nodes["right"]["EPI"] += 2
    changed = _report(graph)
    assert original.edge_form_gap_bounds[0] == changed.edge_form_gap_bounds[0]
    assert original.edge_form_gap_bounds[1] != changed.edge_form_gap_bounds[1]
    assert original.form_gradient_bounds[1] != changed.form_gradient_bounds[1]
    assert original.form_rate_bounds[1] != changed.form_rate_bounds[1]


def test_capture_is_detached_and_preserves_live_graph():
    graph = _graph()

    def snapshot():
        # NetworkX may lazily construct rebuildable view caches; compare
        # actual graph-owned model data rather than those implementation views.
        return pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
            protocol=5,
        )

    before = snapshot()
    report = _report(graph, form_errors=ERRORS)
    assert snapshot() == before
    retained = report.relative_form_bounds
    graph.nodes["right"]["EPI"] = 999
    assert report.relative_form_bounds == retained


@pytest.mark.parametrize(
    "invalid",
    ((0, 0), (0, 0, 0, 0), (0, -1, 0), (0, True, 0), (0, float("inf"), 0), {0, 1, 2}),
)
def test_invalid_error_vectors_reject_before_observation(invalid):
    with pytest.raises(ADMISSION_ERRORS):
        _report(form_errors=invalid)
    with pytest.raises(ADMISSION_ERRORS):
        _report(phase_errors=invalid)


@pytest.mark.parametrize(
    "change",
    ("directed", "disconnected", "weight", "gamma", "phase", "capacity", "reference"),
)
def test_invalid_support_or_declared_model_state_rejects(change):
    graph, reference = _graph(), "left"
    if change == "directed":
        graph = graph.to_directed()
    elif change == "disconnected":
        graph.remove_edge("left", "center")
    elif change == "weight":
        graph.edges["left", "center"]["weight"] = 2
    elif change == "gamma":
        graph.graph["GAMMA"] = {"type": "constant", "value": 1}
    elif change == "phase":
        graph.nodes["left"]["theta"] = float("nan")
    elif change == "capacity":
        graph.nodes["left"]["nu_f"] = -1
    else:
        reference = "absent"
    with pytest.raises(ADMISSION_ERRORS):
        _report(graph, reference=reference)


def test_exact_export_and_reference_label_validation(tmp_path):
    report = _report(form_errors=ERRORS, phase_errors=ERRORS)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-relative-pattern.v1"
    path = tmp_path / "relative-pattern.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    row = payload["report"]["relative_form_bounds"][1]
    assert (
        Q(row["lo"]["numerator"], row["lo"]["denominator"])
        == report.relative_form_bounds[1].lo
    )
    with pytest.raises(TypeError):
        replace(report, reference_node=object()).to_dict()


def test_relative_rates_are_not_the_absolute_hidden_inverse_input():
    delta = Q(1, 2)
    graph = nx.Graph()
    graph.add_node("r", EPI=0, theta=0, nu_f=1)
    graph.add_node("p", EPI=1, theta=float(delta), nu_f=1)
    with mp.workdps(90):
        e, w = map(_mp, MODEL.effective_weights)
        a, b = w / mp.pi, w / (_mp(MODEL.storage_scale) * mp.pi)
        # The actual hidden state is (x_h,theta_h)=(2,0). The moving
        # reference's rows are Vr=2e and Omega_r=-2b. Subtracting them
        # produces precisely a phantom absolute state (x_h,theta_h)=(0,0).
        v = (mp.mpf(0), -e - a * mp.sin(_mp(delta)))
        omega = (mp.mpf(0), b)

        def bounds(value):
            center, radius = Q(mp.nstr(value, 80)), Q(1, 10**12)
            return center - radius, center + radius

        result = infer_relational_sine_hidden_state(
            graph,
            ports=("r", "p"),
            reference_model=MODEL,
            form_rate_bounds=dict(zip(graph, map(bounds, v))),
            phase_rate_bounds=dict(zip(graph, map(bounds, omega))),
            source_id="deliberately-mislabeled-relative-rates",
            clock_id="fixed-clock",
            observation_time=0,
            evidence_window=(0, 0),
            forecast_start=1,
        )
    assert result.phase_rank == 2
    assert result.status == "bounded_candidate"
    assert result.hidden_form_bounds.contains(0)
    assert not result.hidden_form_bounds.contains(2)
    assert result.hidden_unit_phase_relative_to_anchor_bounds[0].contains(1)
    assert result.hidden_unit_phase_relative_to_anchor_bounds[1].contains(0)


def test_forecast_evolves_reference_before_relative_projection(tmp_path):
    graph = nx.path_graph(("r", "p"))
    graph.nodes["r"].update(EPI=1, theta=0, nu_f=1)
    graph.nodes["p"].update(EPI=0, theta=0, nu_f=2)
    report = _report(graph, reference="r", form_errors=(0, 0), phase_errors=(0, 0))
    h = Q(1, 1024)
    forecast = report.forecast(observation_time=0, end_time=h, time_step=h)
    assert forecast.admitted
    assert forecast.relative_form_bounds[0] == I(0)
    assert forecast.relative_phase_bounds[0] == I(0)
    assert forecast.reference_form_displacement_bounds.hi < 0
    assert forecast.reference_phase_displacement_bounds.lo > 0
    endpoint = forecast.full_forecast.endpoint
    assert forecast.relative_form_bounds[1] == endpoint[1] - endpoint[0]
    assert forecast.relative_phase_bounds[1] == endpoint[3] - endpoint[2]
    # Independently: storage<=1/2 gives |x_r-x_p|<=1. With nu=(1,2),
    # e=w=1/2,beta=1 this implies |x_r''|<2 and |(theta_p-theta_r)''|<2.
    # The reference displacement and relative phase obey these coarse
    # first-order Taylor bounds. No second solver or trajectory oracle runs.
    assert I(-h / 2 - h**2, -h / 2 + h**2).contains(
        forecast.reference_form_displacement_bounds
    )
    with mp.workdps(90):
        center = -3 * _mp(h) / (2 * mp.pi)
        assert center - _mp(h**2) <= _mp(forecast.relative_phase_bounds[1].lo)
        assert _mp(forecast.relative_phase_bounds[1].hi) <= center + _mp(h**2)
    payload = forecast.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-relative-forecast.v1"
    path = tmp_path / "relative-forecast.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_forecast_keeps_full_uncertainty_and_unavailable_horizon(monkeypatch):
    from tnfr.physics import relational_sine_forecast

    report = _report(reference="right", form_errors=ERRORS, phase_errors=ERRORS)
    called = []

    def stop_early(initial, **arguments):
        called.append((initial, arguments))
        # Solver wiring is independent of another scientific execution.
        # These are explicitly partial endpoint bounds, never a requested
        # horizon success; the wrapper must retain that distinction.
        return SimpleNamespace(
            endpoint=tuple(value + Q(1, 8) for value in initial[:-1]) + initial[-1:],
            admitted=False,
            status="unavailable",
            reasons=("strict_Picard_inclusion_not_resolved",),
            validated_end_time=Q(1, 4),
        )

    monkeypatch.setattr(relational_sine_forecast, "bound_sine_flow", stop_early)
    forecast = report.forecast(observation_time=0, end_time=1, time_step=Q(1, 4))
    initial, arguments = called[0]
    assert len(called) == 1
    assert initial == report.relative_form_bounds + report.relative_phase_bounds + (
        I(2),
    )
    assert initial[2] == initial[5] == I(0)
    assert initial[0].lo < initial[0].hi
    assert arguments["neighbors"] == report.neighbors
    assert arguments["visible_capacity"] == (1, 0)
    assert not forecast.admitted
    assert forecast.status == "unavailable"
    assert forecast.reasons == ("strict_Picard_inclusion_not_resolved",)
    assert forecast.full_forecast.validated_end_time == Q(1, 4)
    assert forecast.relative_form_bounds[2] == I(0)
    assert forecast.reference_form_displacement_bounds == I(Q(1, 8))


def test_forecast_rebuilds_cached_relative_boxes_from_primitive_preparation():
    graph = nx.path_graph(("r", "p"))
    graph.nodes["r"].update(EPI=0, theta=0, nu_f=0)
    graph.nodes["p"].update(EPI=1, theta=0, nu_f=0)
    report = _report(graph, reference="r", form_errors=(0, 0), phase_errors=(0, 0))
    changed = replace(
        report,
        relative_form_bounds=(I(0), I(0)),
        relative_phase_bounds=(I(0), I(99)),
    )
    h = Q(1, 1024)
    forecast = changed.forecast(observation_time=0, end_time=h, time_step=h)
    # Both held capacities vanish, so both complete-law rows vanish exactly.
    # The declared form contrast persists; cached zero contrasts cannot erase it.
    assert forecast.admitted
    assert forecast.relative_form_bounds == (I(0), I(1))
    assert forecast.relative_phase_bounds == (I(0), I(0))
    assert forecast.pattern.relative_form_bounds == (I(0), I(1))
    assert forecast.pattern.relative_phase_bounds == (I(0), I(0))
    assert changed.relative_form_bounds == (I(0), I(0))


def test_forecast_rebuilds_exact_residual_box_before_solver(monkeypatch):
    from tnfr.physics import relational_sine_forecast

    report = replace(
        _report(),
        reference_node="right",
        nominal_form=(Q(1, 10**400), Q(-2), Q(3)),
        nominal_phase=(Q(1), Q(2), Q(4)),
        form_error_bounds=(Q(1, 8), Q(1, 4), Q(1, 2)),
        phase_error_bounds=(Q(1, 16), Q(1, 8), Q(1, 4)),
    )
    captured = []

    def solver(initial, **kwargs):
        captured.append(initial)
        return SimpleNamespace(endpoint=initial)

    monkeypatch.setattr(relational_sine_forecast, "bound_sine_flow", solver)
    report.forecast(observation_time=0, end_time=1, time_step=1)
    tiny = Q(1, 10**400)
    assert captured == [
        (
            I(tiny - 3 - Q(5, 8), tiny - 3 + Q(5, 8)),
            I(-5 - Q(3, 4), -5 + Q(3, 4)),
            I(0),
            I(-3 - Q(5, 16), -3 + Q(5, 16)),
            I(-2 - Q(3, 8), -2 + Q(3, 8)),
            I(0),
            I(2),
        )
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"law": "native_arg_exchange"},
        {"capacity": (1, -1, 2)},
        {"nominal_form": (True, 0, 1)},
        {"phase_error_bounds": (0, -1, 0)},
        {"neighbors": ((1,), (0,), (1,))},
        {"reference_node": "absent"},
    ],
)
def test_forecast_rejects_invalid_source_before_solver(monkeypatch, changes):
    from tnfr.physics import relational_sine_forecast

    def forbidden(*args, **kwargs):
        pytest.fail("invalid source reached the solver")

    monkeypatch.setattr(relational_sine_forecast, "bound_sine_flow", forbidden)
    with pytest.raises(ADMISSION_ERRORS):
        replace(_report(), **changes).forecast(
            observation_time=0, end_time=1, time_step=1
        )
