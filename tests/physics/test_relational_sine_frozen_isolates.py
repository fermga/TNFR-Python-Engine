"""Static controls for an explicitly supplied component/frozen-isolate law.

The disconnected source is independent of THOL history or event responses.
It has a weighted triangle, a zero-conductance bare edge and two isolates.
No trajectory, autonomous occurrence or intrinsic child energy is supplied.
"""

import pickle
from dataclasses import fields, replace
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_support import (
    SineSupportExchangeComparison,
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)

_MODEL = RelationalExchangeModel(2, epi_weight=0, phase_domain="regular")
_NODES = ("frozen", 3, "b", ("leaf", 0), "zero", "tail", 9)
_EDGES = ((3, ("leaf", 0), 2), (("leaf", 0), 9, Q(1, 2)), (9, 3, 0), ("b", "tail", 0))
_ERRORS = (TypeError, ValueError, TNFRUserError)


def _graph():
    graph = nx.Graph()
    for node, x, theta, nu in zip(
        _NODES,
        (-2, Q(1, 4), Q(-1, 2), Q(3, 4), 3, Q(1, 8), Q(-3, 8)),
        (Q(1, 2), 0, Q(1, 4), Q(-3, 8), -1, Q(1, 2), Q(1, 8)),
        (2, Q(1, 2), 1, Q(3, 2), 0, 2, 1),
    ):
        graph.add_node(node, EPI=x, theta=theta, nu_f=nu, delta_nfr=99)
    graph.add_weighted_edges_from(_EDGES)
    graph.graph.update(GAMMA={"type": "none"}, untouched={"values": [1, 2]})
    return graph


def _capture(graph):
    return bound_relational_sine_frozen_isolates(graph, reference_model=_MODEL)


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("a detached frozen-isolate comparison must not run a trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


@pytest.fixture(scope="module")
def sample():
    return _capture(_graph())


def test_independent_complete_rows_storage_work_and_frozen_state(sample, mp):
    graph = _graph()
    before = _snapshot(graph)
    assert _capture(graph) == sample
    assert _snapshot(graph) == before
    assert sample.nodes == _NODES
    assert sample.component_indices == ((0,), (1, 3, 6), (2, 5), (4,))
    assert sample.isolate_indices == (0, 4)
    assert sample.degrees == (0, 2, 1, 2, 0, 1, 2)
    position = {node: i for i, node in enumerate(_NODES)}
    gradient, sine = [Q(0)] * len(_NODES), [mp.mpf(0)] * len(_NODES)
    form_storage, phase_storage = Q(0), mp.mpf(0)
    for left, right, weight in _EDGES:
        i, j = position[left], position[right]
        gap = sample.phase[j] - sample.phase[i]
        current = mp.sin(_mp(mp, gap))
        delta = sample.epi[i] - sample.epi[j]
        gradient[i] += weight * delta
        gradient[j] -= weight * delta
        sine[i] += current
        sine[j] -= current
        form_storage += weight * delta**2 / 2
        phase_storage += 1 - mp.cos(_mp(mp, gap))
    assert sample.form_gradient == tuple(gradient)
    assert sample.form_storage == form_storage
    _contains(mp, sample.phase_storage, phase_storage)
    _contains(mp, sample.storage, _mp(mp, form_storage) + 2 * phase_storage)
    expected_numerators = tuple(
        nu * q / (2 * degree) if degree else Q(0)
        for nu, q, degree in zip(sample.capacity, gradient, sample.degrees)
    )
    assert sample.phase_rate_numerators() == expected_numerators
    for i, (degree, capacity) in enumerate(zip(sample.degrees, sample.capacity)):
        if not degree:
            assert sample.inverse_phase_metric[i] is None
            assert sample.relative_resultant[i] == (I(0), I(0))
            for name in (
                "phase_sources",
                "pressure",
                "form_rates",
                "phase_rates",
                "form_work",
                "phase_work",
                "node_balance_residual",
            ):
                assert getattr(sample, name)[i] == I(0), name
            continue
        pressure = sine[i] / (mp.pi * degree)
        form_rate = _mp(mp, capacity) * pressure
        phase_rate = _mp(mp, expected_numerators[i]) / mp.pi
        for bounds, value in (
            (sample.pressure[i], pressure),
            (sample.form_rates[i], form_rate),
            (sample.phase_rates[i], phase_rate),
            (sample.form_work[i], _mp(mp, gradient[i]) * form_rate),
            (sample.phase_work[i], -2 * sine[i] * phase_rate),
        ):
            _contains(mp, bounds, value)
        assert sample.node_balance_residual[i].contains(0)
    assert (
        sample.epi[0] == -2 and sample.phase[0] == Q(1, 2) and sample.capacity[0] == 2
    )
    assert sample.continuous_loss == 0
    assert sample.dissipation == (Q(0),) * len(_NODES)
    assert sample.storage_rate.contains(0)
    assert sample.balance_residual.contains(0)


def test_connected_limit_reuses_all_existing_physical_fields_without_type_promotion():
    graph = _graph().subgraph((3, ("leaf", 0), 9)).copy()
    connected = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    extended = _capture(graph)
    assert extended.isolate_indices == ()
    assert extended.component_indices == ((0, 1, 2),)
    assert extended.law != connected.law
    for field in fields(connected):
        if field.name not in ("law", "scope"):
            assert getattr(extended, field.name) == getattr(connected, field.name)
    assert extended.phase_rate_numerators() == connected.phase_rate_numerators()
    with pytest.raises(TypeError, match="SineSupportExchangeComparison"):
        SineSupportExchangeComparison.phase_rate_numerators(extended)


def test_existing_connected_reader_and_rebuilder_keep_rejecting_disconnected_support():
    with pytest.raises(ValueError, match="connected support"):
        bound_relational_sine_support_exchange(_graph(), reference_model=_MODEL)
    graph = _graph().subgraph((3, ("leaf", 0), 9)).copy()
    connected = bound_relational_sine_support_exchange(graph, reference_model=_MODEL)
    disconnected = replace(connected, edges=(), conductances=())
    for consume in (disconnected.to_dict, disconnected.phase_rate_numerators):
        with pytest.raises(ValueError, match="must be connected"):
            consume()


def test_zero_conductance_bare_component_is_not_an_isolate(sample):
    for i in (2, 5):
        assert sample.degrees[i] == 1
        assert sample.inverse_phase_metric[i] is not None
        assert sample.form_gradient[i] == 0
        assert sample.phase_rates[i] == I(0)
        assert not sample.pressure[i].contains(0)
        assert not sample.form_rates[i].contains(0)


def test_all_isolates_retain_full_state_and_have_no_supplied_edge_storage():
    graph = _graph()
    graph.remove_edges_from(tuple(graph.edges()))
    report = _capture(graph)
    assert report.isolate_indices == tuple(range(len(graph)))
    assert report.component_indices == tuple((i,) for i in range(len(graph)))
    assert report.epi == _capture(_graph()).epi
    assert report.phase == _capture(_graph()).phase
    assert report.capacity == _capture(_graph()).capacity
    assert report.inverse_phase_metric == (None,) * len(graph)
    assert (
        report.pressure
        == report.form_rates
        == report.phase_rates
        == (I(0),) * len(graph)
    )
    assert report.phase_rate_numerators() == (Q(0),) * len(graph)
    assert report.form_storage == 0
    assert report.phase_storage == report.storage == I(0)
    assert report.storage_rate == report.balance_residual == I(0)


def test_reordered_source_keeps_original_indices_and_equivalent_node_fields(sample):
    graph = _graph()
    reordered = nx.Graph()
    reordered.graph.update(graph.graph)
    reordered.add_nodes_from(
        (node, graph.nodes[node]) for node in reversed(tuple(graph))
    )
    reordered.add_edges_from(graph.edges(data=True))
    result = _capture(reordered)
    assert result.nodes == tuple(reversed(sample.nodes))
    for name in (
        "epi",
        "phase",
        "capacity",
        "degrees",
        "pressure",
        "form_rates",
        "phase_rates",
    ):
        assert dict(zip(result.nodes, getattr(result, name))) == dict(
            zip(sample.nodes, getattr(sample, name))
        )
    assert result.storage == sample.storage
    assert result.isolate_indices == (2, 6)


@pytest.mark.parametrize("alias", (ALIAS_EPI, ALIAS_THETA, ALIAS_VF))
@pytest.mark.parametrize(
    "invalid", (True, "0.5", complex(1, 0), float("nan"), float("inf"), Q(1, 10**500))
)
def test_isolate_coordinates_are_admitted_before_any_component_or_alias_fallback(
    alias, invalid
):
    graph = _graph()
    graph.nodes["frozen"][alias[0]] = invalid
    graph.nodes["frozen"][alias[1]] = 0.5
    before = _snapshot(graph)
    with pytest.raises(_ERRORS):
        _capture(graph)
    assert _snapshot(graph) == before


@pytest.mark.parametrize("alias", (ALIAS_EPI, ALIAS_THETA, ALIAS_VF))
def test_missing_isolate_coordinate_does_not_become_an_assumed_zero(alias):
    graph = _graph()
    for key in alias:
        graph.nodes["frozen"].pop(key, None)
    with pytest.raises(_ERRORS):
        _capture(graph)


def test_negative_isolate_capacity_rejects_but_signed_form_and_phase_are_retained():
    graph = _graph()
    graph.nodes["frozen"]["nu_f"] = -1
    with pytest.raises(_ERRORS):
        _capture(graph)


@pytest.mark.parametrize("size", (0, 1))
def test_minimum_total_state_remains_two_nodes(size):
    graph = _graph().subgraph(_NODES[:size]).copy()
    with pytest.raises(ValueError, match="at least two nodes"):
        _capture(graph)


@pytest.mark.parametrize("kind", ("directed", "parallel", "self_loop"))
def test_invalid_source_support_rejects(kind):
    graph = _graph()
    if kind == "directed":
        graph = nx.DiGraph(graph)
    elif kind == "parallel":
        graph = nx.MultiGraph(graph)
    else:
        graph.add_edge("frozen", "frozen")
    with pytest.raises(_ERRORS):
        _capture(graph)


@pytest.mark.parametrize("weight", (True, -1, float("inf"), Q(1, 10**500)))
def test_invalid_component_weight_is_not_hidden_by_frozen_nodes(weight):
    graph = _graph()
    graph.edges[3, ("leaf", 0)]["weight"] = weight
    before = _snapshot(graph)
    with pytest.raises(_ERRORS):
        _capture(graph)
    assert _snapshot(graph) == before


@pytest.mark.parametrize(
    "setting",
    (
        {"use_extended_dynamics": True},
        {"GAMMA": {"type": "constant"}},
        {"GAMMA": False},
        {"GAMMA": 0},
        {"GAMMA": "none"},
        {"GAMMA": []},
    ),
)
def test_full_source_extension_and_forcing_are_not_lost_during_component_partition(
    setting,
):
    graph = _graph()
    graph.graph.update(setting)
    with pytest.raises(_ERRORS):
        _capture(graph)


def test_missing_or_explicit_none_gamma_retains_existing_absent_source_contract(sample):
    graph = _graph()
    graph.graph.pop("GAMMA")
    assert _capture(graph) == sample
    graph.graph["GAMMA"] = None
    assert _capture(graph) == sample


def test_live_gamma_registry_is_checked_before_partition_without_invoking_source(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("detached admission must not invoke supplied Gamma")

    graph = _graph()
    before = _snapshot(graph)
    monkeypatch.setitem(GAMMA_REGISTRY, "none", GammaEntry(forbidden, False))
    with pytest.raises(_ERRORS):
        _capture(graph)
    assert _snapshot(graph) == before


@pytest.mark.parametrize(
    "model",
    (
        RelationalExchangeModel(2, epi_weight=Q(1, 8), phase_domain="regular"),
        RelationalExchangeModel(2, epi_weight=0, phase_domain="acute"),
    ),
)
def test_model_is_not_reinterpreted_when_all_nodes_are_frozen(model):
    graph = _graph()
    graph.remove_edges_from(tuple(graph.edges()))
    with pytest.raises(_ERRORS):
        bound_relational_sine_frozen_isolates(graph, reference_model=model)


def test_export_and_numerators_rebuild_components_and_all_consumed_fields(sample):
    graph = _graph()
    graph.remove_edges_from(tuple(graph.edges()))
    expected = _capture(graph)
    changed = replace(
        sample,
        edges=(),
        conductances=(),
        component_indices=((0, 1, 2, 3, 4, 5, 6),),
        isolate_indices=(),
        degrees=(99,) * len(graph),
        form_gradient=(Q(99),) * len(graph),
        pressure=(I(99),) * len(graph),
        inverse_phase_metric=(I(99),) * len(graph),
        phase_rates=(I(99),) * len(graph),
        form_storage=Q(99),
        storage=I(99),
        balance_residual=I(99),
    )
    assert changed.to_dict() == expected.to_dict()
    assert changed.phase_rate_numerators() == (Q(0),) * len(graph)
    assert (
        expected.to_dict()["schema"]
        == "tnfr.relational-sine-frozen-isolate-comparison.v1"
    )


@pytest.mark.parametrize(
    "change",
    (
        {"epi": (True, *([Q(0)] * 6))},
        {"phase": (float("nan"), *([Q(0)] * 6))},
        {"capacity": (Q(-1), *([Q(0)] * 6))},
        {"conductances": (True, Q(1), Q(1), Q(1))},
        {"edges": (("frozen", "frozen"),), "conductances": (Q(1),)},
        {"law": "conservative_sine_support_exchange"},
    ),
)
def test_invalid_primitive_report_cannot_hide_behind_zero_or_cached_rows(
    sample, change
):
    forged = replace(sample, **change)
    for consume in (forged.to_dict, forged.phase_rate_numerators):
        with pytest.raises(_ERRORS):
            consume()
