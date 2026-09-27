"""Routine contracts for the opt-in, conditional relational joint solver.

Analytic small graphs check the actual shared owner. No operator policy,
autonomous-law selection, trajectory persistence or physical bridge is inferred.
"""

import math
import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr, get_theta_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import relational
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)
from tnfr.errors.contextual import TNFRUserError
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry
from tnfr.mathematics import BEPIElement
from tnfr.metrics.trig_cache import get_trig_cache
from tnfr.sdk import Network
from tnfr.types import ensure_bepi, require_finite_real_scalar_epi

ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _graph():
    graph = nx.path_graph(3)
    graph.graph.update(
        GAMMA={"type": "none"},
        DNFR_WEIGHTS={"epi": 0.1, "phase": 0.2, "vf": 0.3, "topo": 0.4},
        _t=2.0,
        untouched={"evidence": [1, 2]},
    )
    for node, form, phase, capacity in zip(
        graph,
        (0.75, 0.25, 0.5),
        (-math.pi / 6, 0.0, math.pi / 6),
        (1.0, 0.0, 2.0),
        strict=True,
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity, delta_nfr=999.0)
    return graph


def _pair(*, form=(1.0, -1.0), phase=(0.0, 0.0)):
    graph = nx.path_graph(2)
    graph.graph.update(GAMMA={"type": "none"}, _t=0.0)
    for node in graph:
        graph.nodes[node].update(
            EPI=form[node], theta=phase[node], nu_f=1.0, delta_nfr=999.0
        )
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def test_analytic_heterogeneous_path_field_is_fresh_detached_and_read_only():
    graph = _graph()
    before = _snapshot(graph)
    report = evaluate_relational_exchange(graph, model=RelationalExchangeModel(2.0))
    assert _snapshot(graph) == before
    assert report.nodes == (0, 1, 2)
    assert report.form_gradient == pytest.approx((0.5, -0.75, 0.25))
    assert report.phase_source == pytest.approx((1 / 6, 0, -1 / 6))
    assert report.phase_metric == pytest.approx((3, math.pi * math.sqrt(3), 3))
    assert report.pressure == pytest.approx((-1 / 6, 3 / 16, -5 / 24))
    assert report.form_rate == pytest.approx((-1 / 6, 0, -5 / 12))
    assert report.phase_rate == pytest.approx((1 / 24, 0, 1 / 24))
    assert float(report.continuous_loss) == pytest.approx(3 / 16)
    assert float(report.storage_rate) == pytest.approx(-3 / 16)
    assert abs(float(report.balance_residual)) < 1e-14
    with pytest.raises(FrozenInstanceError):
        report.epi = (0.0, 0.0, 0.0)
    graph.nodes[0]["EPI"] = 10.0
    assert report.epi == (0.75, 0.25, 0.5)
    assert isinstance(report.pressure, tuple)
    assert isinstance(report.edges, tuple)


def test_consensus_metric_limit_and_receiver_capacity_separation():
    graph = _graph()
    for node in graph:
        graph.nodes[node]["theta"] = 0.0
    model = RelationalExchangeModel(2.0)
    original = evaluate_relational_exchange(graph, model=model)
    assert original.phase_source == (0.0, 0.0, 0.0)
    assert original.phase_metric == pytest.approx((math.pi, 2 * math.pi, math.pi))
    assert original.phase_rate == pytest.approx(
        (1 / (8 * math.pi), 0, 1 / (8 * math.pi))
    )
    graph.nodes[1]["nu_f"] = 0.5
    changed = evaluate_relational_exchange(graph, model=model)
    assert changed.phase_rate[0] == original.phase_rate[0]
    assert changed.phase_rate[2] == original.phase_rate[2]
    assert changed.phase_rate[1] != original.phase_rate[1]


def test_pressure_uses_explicit_model_and_matches_the_native_owner():
    graph = _graph()
    before = _snapshot(graph)
    model = RelationalExchangeModel(2.0, epi_weight=3.0, phase_weight=1.0)
    report = evaluate_relational_exchange(graph, model=model)
    native = deepcopy(graph)
    native.graph["DNFR_WEIGHTS"] = {"epi": 0.75, "phase": 0.25, "vf": 0.0, "topo": 0.0}
    default_compute_delta_nfr(native)
    expected = tuple(get_attr(native.nodes[node], ALIAS_DNFR) for node in native)
    assert report.pressure == pytest.approx(expected, abs=1e-15)
    assert _snapshot(graph) == before


def test_single_step_is_simultaneous_unclipped_and_refreshes_endpoint_pressure():
    graph = _graph()
    graph.graph.update(CLIP_MODE="hard", EPI_MIN=-0.1, EPI_MAX=0.1)
    ambient_weights = deepcopy(graph.graph["DNFR_WEIGHTS"])
    model = RelationalExchangeModel(2.0)
    dt = 1 / 16
    report = step_relational_exchange(graph, model=model, dt=dt)
    expected_form = (0.75 - dt / 6, 0.25, 0.5 - 5 * dt / 12)
    expected_phase = (-math.pi / 6 + dt / 24, 0, math.pi / 6 + dt / 24)
    assert tuple(
        get_attr(graph.nodes[node], ALIAS_EPI) for node in graph
    ) == pytest.approx(expected_form)
    assert tuple(get_theta_attr(graph.nodes[node]) for node in graph) == pytest.approx(
        expected_phase
    )
    assert graph.graph["_t"] == 2.0 + dt
    assert report.t_before == 2.0
    assert report.t_after == 2.0 + dt
    assert tuple(get_attr(graph.nodes[node], ALIAS_VF) for node in graph) == (
        1.0,
        0.0,
        2.0,
    )
    assert graph.graph["DNFR_WEIGHTS"] == ambient_weights
    assert graph.graph["CLIP_MODE"] == "hard"
    assert graph.graph["untouched"] == {"evidence": [1, 2]}
    endpoint = evaluate_relational_exchange(graph, model=model)
    assert (
        tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in graph)
        == endpoint.pressure
    )
    assert report.after.pressure == endpoint.pressure
    assert report.after.pressure != report.before.pressure


@pytest.mark.parametrize(
    "arguments",
    (
        {"storage_scale": 0.0},
        {"storage_scale": 1.0, "phase_weight": 0.0},
        {"storage_scale": True},
    ),
)
def test_model_rejects_unsupported_scales_without_boolean_coercion(arguments):
    with pytest.raises(ADMISSION_ERRORS):
        RelationalExchangeModel(**arguments)


@pytest.mark.parametrize(
    "attribute,value",
    (("theta", True), ("nu_f", Q(1, 2**2000)), ("EPI", "0.5")),
)
def test_raw_last_node_rejection_is_atomic(attribute, value):
    graph = _graph()
    graph.nodes[2][attribute] = value
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


def test_rich_bepi_is_not_replaced_by_a_magnitude_even_at_zero_capacity():
    graph = _graph()
    graph.nodes[2]["EPI"] = BEPIElement((0.5, -0.5), (0.5, -0.5), (0.0, 1.0))
    graph.nodes[2]["nu_f"] = 0.0
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


def test_uniform_negative_bepi_and_legacy_aliases_keep_the_signed_chart():
    graph = _pair(form=(-0.5, -0.25))
    for node in graph:
        attributes = graph.nodes[node]
        original = attributes.pop("EPI")
        attributes[ALIAS_EPI[-1]] = ensure_bepi(original)
        attributes["phase"] = attributes.pop("theta")
        attributes["EPI_kind"] = "wave"
    get_trig_cache(graph)
    report = step_relational_exchange(
        graph, model=RelationalExchangeModel(1.0), dt=0.125
    )
    observed = tuple(
        require_finite_real_scalar_epi(
            get_attr(graph.nodes[node], ALIAS_EPI, conv=lambda raw: raw)
        )
        for node in graph
    )
    assert observed == pytest.approx((-0.484375, -0.265625))
    assert all(value < 0 for value in observed)
    assert all(ALIAS_EPI[-1] in graph.nodes[node] for node in graph)
    assert all(graph.nodes[node]["EPI_kind"] == "wave" for node in graph)
    cached_phase = get_trig_cache(graph).theta
    assert tuple(cached_phase[node] for node in graph) == pytest.approx(
        report.after.phase
    )


def test_nonzero_underflowed_nodal_rate_is_not_certified_as_equilibrium():
    graph = _pair(form=(0.25, 0.0))
    graph.nodes[0]["nu_f"] = math.ulp(0.0)
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0))
    assert _snapshot(graph) == before


def test_exact_nonzero_storage_survives_below_binary64_energy_resolution():
    small = 1e-200
    graph = _pair(form=(small, 0.0))
    report = evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0))
    assert report.form_storage == Q(small) ** 2 / 2
    assert report.storage > 0
    assert report.continuous_loss > 0
    assert float(report.storage) == 0.0
    assert float(report.continuous_loss) == 0.0
    assert all(rate != 0.0 for rate in report.form_rate)


@pytest.mark.parametrize("kind", ("directed", "multigraph", "disconnected", "weighted"))
def test_unsupported_graph_is_rejected_before_any_owned_write(kind):
    graph = _graph()
    if kind == "directed":
        graph = nx.DiGraph(graph)
    elif kind == "multigraph":
        graph = nx.MultiGraph(graph)
    elif kind == "disconnected":
        graph.remove_edge(1, 2)
    else:
        graph.edges[0, 1]["weight"] = 2.0
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


@pytest.mark.parametrize(
    "clock_arguments",
    ({"dt": True}, {"dt": Q(1, 2**2000)}, {"dt": 0.1, "t": True}),
)
def test_solver_clock_rejects_raw_invalid_values_atomically(clock_arguments):
    graph = _graph()
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(
            graph, model=RelationalExchangeModel(2.0), **clock_arguments
        )
    assert _snapshot(graph) == before


def test_overridden_none_gamma_cannot_execute_or_hide_declared_forcing(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an unforced exchange must never invoke an external Gamma callback")

    graph = _graph()
    monkeypatch.setitem(GAMMA_REGISTRY, "none", GammaEntry(forbidden, False))
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


def test_active_gamma_declaration_is_rejected_even_if_a_parameter_is_zero():
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "harmonic", "beta": 0.0}
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        evaluate_relational_exchange(graph, model=RelationalExchangeModel(2.0))
    assert _snapshot(graph) == before


def test_nonacute_candidate_endpoint_rolls_back_form_phase_clock_and_metadata():
    graph = _pair()
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(
            graph,
            model=RelationalExchangeModel(1.0, epi_weight=0, phase_weight=1),
            dt=2.0,
        )
    assert _snapshot(graph) == before


def test_failed_cache_hook_cannot_commit_a_valid_candidate_partially():
    graph = _graph()
    graph.graph["_trig_version"] = "malformed existing cache metadata"
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


def test_wrapped_acute_endpoint_cannot_hide_nonacute_euler_segment():
    graph = _pair()
    before = _snapshot(graph)
    # The two opposite phase rates produce a relative turn of approximately
    # -2*pi: its endpoint wraps near zero but its segment crosses pi/2.
    with pytest.raises(ADMISSION_ERRORS, match="initial acute lift"):
        step_relational_exchange(
            graph,
            model=RelationalExchangeModel(1.0, epi_weight=0, phase_weight=1),
            dt=math.pi**2 / 2,
        )
    assert _snapshot(graph) == before


def test_fresh_pressure_rebuilds_a_stale_maximum_with_an_invalid_node():
    graph = _graph()
    graph.graph.update(_dnfrmax=1e100, _dnfrmax_node="not_a_live_node")
    report = step_relational_exchange(
        graph, model=RelationalExchangeModel(2.0), dt=0.0625
    )
    expected = max(abs(value) for value in report.after.pressure)
    assert graph.graph["_dnfrmax"] == expected
    maximum_node = graph.graph["_dnfrmax_node"]
    assert maximum_node in graph
    assert abs(get_attr(graph.nodes[maximum_node], ALIAS_DNFR)) == expected


def test_euler_energy_increase_is_reported_without_false_continuous_monotonicity():
    graph = _pair()
    report = step_relational_exchange(
        graph, model=RelationalExchangeModel(1.0, epi_weight=0, phase_weight=1), dt=0.1
    )
    assert float(report.before.continuous_loss) == 0.0
    assert float(report.energy_change) > 0.0
    assert float(report.energy_step_defect) > 0.0
    assert report.after.epi == report.before.epi
    assert report.after.phase != report.before.phase


def test_sdk_methods_delegate_to_the_same_model_owner(monkeypatch):
    graph = _graph()
    network = Network(graph)
    model = RelationalExchangeModel(2.0)
    captured = []
    marker = object()

    def evaluate(target, *, model):
        captured.append(("evaluate", target, model))
        return marker

    def step(target, *, model, dt, t=None):
        captured.append(("step", target, model, dt, t))
        return marker

    monkeypatch.setattr(relational, "evaluate_relational_exchange", evaluate)
    monkeypatch.setattr(relational, "step_relational_exchange", step)
    assert network.relational_exchange(model) is marker
    assert network.step_relational(model, dt=0.25, t=3.0) is marker
    assert captured == [("evaluate", graph, model), ("step", graph, model, 0.25, 3.0)]
