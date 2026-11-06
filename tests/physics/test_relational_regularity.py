"""Static regular-domain guarantees and separately supplied rate kinematics.

No trajectory is advanced. Independent small-graph energies and high-precision
Taylor values check the outward evidence, rather than substituting native
floating-point storage for the continuous-law premise.
"""

import json
import math
import pickle
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.physics import relational_regularity as owner

MODEL = RelationalExchangeModel(1, phase_domain="regular")
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _state(graph, *, forms=None, phases=None, capacities=None):
    nodes = tuple(graph)
    forms = forms if forms is not None else (0,) * len(nodes)
    phases = phases if phases is not None else (0,) * len(nodes)
    capacities = capacities if capacities is not None else (1,) * len(nodes)
    for node, form, phase, capacity in zip(nodes, forms, phases, capacities):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity, delta_nfr=999.0)
    graph.graph.update(GAMMA={"type": "none"}, untouched={"values": [1, 2]})
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _decimal_trig(angle, *, sine=False):
    """Independent high-precision reference for the modest angles used here."""
    with localcontext() as context:
        context.prec = 100
        angle = Decimal(angle.numerator) / Decimal(angle.denominator)
        term = total = angle if sine else Decimal(1)
        for index in range(1, 80):
            denominator = (
                (2 * index) * (2 * index + 1) if sine else (2 * index - 1) * (2 * index)
            )
            term *= -(angle * angle) / denominator
            total += term
        return Q(total)


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("a static regularity observer must not advance a trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


def test_nonacute_leaf_state_gets_positive_global_regular_margins():
    report = owner.observe_relational_regularity(
        _state(nx.path_graph(2), phases=(0, 2)), model=MODEL
    )
    assert report.regularity_certified
    assert 1 < report.storage_bounds[0] <= report.storage_bounds[1] < 2
    assert all(real < 0 for real, _ in report.field.relative_resultant)
    assert min(report.phase_metric_lower_bounds) > 0
    assert 0 < report.resultant_cut_distance_lower_bound < 1
    for lower, metric in zip(
        report.phase_metric_lower_bounds, report.field.phase_metric
    ):
        assert float(lower) < metric
    assert "not_convergence" in " ".join(report.scope)


@pytest.mark.parametrize(
    "size,forms,expected_storage",
    [(3, (1, 0, 0), Q(1)), (4, (1, -0.5, 0, 0), Q(19, 8))],
)
def test_complete_graph_degree_threshold_with_independent_zero_phase_energy(
    size, forms, expected_storage
):
    report = owner.observe_relational_regularity(
        _state(nx.complete_graph(size), forms=forms), model=MODEL
    )
    # Complete-graph Dirichlet identity: (n sum(x_i^2) - sum(x_i)^2)/2.
    values = tuple(map(Q, forms))
    identity = (size * sum(x * x for x in values) - sum(values) ** 2) / 2
    assert identity == expected_storage
    assert report.storage_bounds == (identity, identity)
    assert report.regularity_storage_threshold == size - 1
    assert report.regularity_certified
    assert report.resultant_cut_distance_lower_bound == size - 1 - identity
    if size == 4:
        assert 2 < identity < report.regularity_storage_threshold


@pytest.mark.parametrize(
    "amplitude,certified",
    [(math.nextafter(2.0, 0.0), True), (2.0, False), (math.nextafter(2.0, 3.0), False)],
)
def test_storage_threshold_is_strict_and_unresolved_is_not_singularity(
    amplitude, certified
):
    report = owner.observe_relational_regularity(
        _state(nx.path_graph(2), forms=(amplitude, 0)), model=MODEL
    )
    energy = Q(amplitude) ** 2 / 2
    assert report.storage_bounds == (energy, energy)
    assert report.regularity_certified is certified
    # All resultants are exactly positive: a failed sufficient bound cannot
    # change point admission or establish an actual boundary encounter.
    assert report.field.relative_resultant == ((1.0, 0.0), (1.0, 0.0))
    if not certified:
        assert report.status == "storage_test_unresolved"
        assert report.phase_metric_lower_bounds is None
        assert report.resultant_cut_distance_lower_bound is None


def test_storage_uses_outward_mathematical_cosine_and_exact_signed_form():
    model = RelationalExchangeModel(0.5, phase_domain="regular")
    report = owner.observe_relational_regularity(
        _state(nx.path_graph(2), forms=(-0.25, 0.5), phases=(0, 2)), model=model
    )
    reference = Q(9, 32) + Q(1, 2) * (1 - _decimal_trig(Q(2)))
    lower, upper = report.storage_bounds
    assert lower < reference - Q(1, 10**80)
    assert reference + Q(1, 10**80) < upper
    assert upper - lower < Q(1, 10**12)
    assert report.regularity_certified


@pytest.mark.parametrize("capacities", [(0, 0), (0, 2), (1, 1)])
def test_capacity_changes_leave_storage_domain_bound_unchanged(capacities):
    report = owner.observe_relational_regularity(
        _state(
            nx.path_graph(2),
            forms=(0.125, -0.125),
            phases=(0, 2),
            capacities=capacities,
        ),
        model=MODEL,
    )
    assert report.regularity_certified
    assert min(report.phase_metric_lower_bounds) > 0
    assert any(value != 0 for value in report.field.pressure)
    for index, capacity in enumerate(capacities):
        if capacity == 0:
            assert report.field.form_rate[index] == report.field.phase_rate[index] == 0
        else:
            assert report.field.form_rate[index] != 0
            assert report.field.phase_rate[index] != 0
    if not any(capacities):
        assert report.resultant_speed_upper_bounds == (0, 0)
        assert report.resultant_rate_bounds == (((0, 0), (0, 0)),) * 2


def test_kinematic_bounds_consume_one_captured_direction_without_ODE_claim(monkeypatch):
    graph = _state(nx.path_graph(2), forms=(0.25, -0.25))
    original = owner.evaluate_relational_exchange
    captures = []

    def supplied_direction(*args, **kwargs):
        captures.append(1)
        return replace(original(*args, **kwargs), phase_rate=(7.0, -3.0))

    monkeypatch.setattr(owner, "evaluate_relational_exchange", supplied_direction)
    report = owner.observe_relational_regularity(graph, model=MODEL)
    assert len(captures) == 1
    assert report.resultant_rate_bounds == (((0, 0), (-10, -10)), ((0, 0), (10, 10)))
    assert report.resultant_speed_upper_bounds == (10, 10)
    assert "captured_rates_not_certified_ideal_ODE_rates" in " ".join(report.scope)
    assert "not_a_whole_time_rate_bound" in " ".join(report.scope)


def test_nonzero_phase_kinematics_enclose_independent_chain_rule():
    report = owner.observe_relational_regularity(
        _state(nx.path_graph(2), forms=(0.25, -0.125), phases=(0, 2)), model=MODEL
    )
    rates = tuple(map(Q, report.field.phase_rate))
    for index, angle in enumerate((Q(2), Q(-2))):
        difference = rates[1 - index] - rates[index]
        expected = (
            -difference * _decimal_trig(angle, sine=True),
            difference * _decimal_trig(angle),
        )
        for (lower, upper), reference in zip(
            report.resultant_rate_bounds[index], expected
        ):
            assert lower < reference < upper
        assert report.resultant_speed_upper_bounds[index] == abs(difference)


def test_detached_observation_is_node_order_covariant_and_json_safe():
    graph = _state(
        nx.path_graph(3), forms=(0.125, -0.25, 0.5), phases=(0, 0.25, -0.125)
    )
    before = _snapshot(graph)
    report = owner.observe_relational_regularity(graph, model=MODEL)
    assert _snapshot(graph) == before
    permuted = nx.Graph()
    for node in (2, 0, 1):
        permuted.add_node(node, **graph.nodes[node])
    permuted.add_edges_from(reversed(tuple(graph.edges)))
    permuted.graph.update(graph.graph)
    other = owner.observe_relational_regularity(permuted, model=MODEL)
    assert report.storage_bounds == other.storage_bounds
    assert (
        report.resultant_cut_distance_lower_bound
        == other.resultant_cut_distance_lower_bound
    )
    for attribute in (
        "phase_metric_lower_bounds",
        "resultant_rate_bounds",
        "resultant_speed_upper_bounds",
    ):
        first = dict(zip(report.field.nodes, getattr(report, attribute)))
        second = dict(zip(other.field.nodes, getattr(other, attribute)))
        assert first == second
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["schema"] == "tnfr.relational-regularity.v1"
    encoded = payload["report"]["storage_bounds"]
    assert (
        tuple(Q(item["numerator"], item["denominator"]) for item in encoded)
        == report.storage_bounds
    )
    assert payload["report"]["field"]["nodes"] == list(graph)


@pytest.mark.parametrize(
    "model",
    [
        None,
        object(),
        RelationalExchangeModel(1),
        RelationalExchangeModel(1, phase_domain="positive_resultant"),
    ],
)
def test_explicit_regular_model_is_required(model):
    with pytest.raises(ValueError, match="explicit regular"):
        owner.observe_relational_regularity(_state(nx.path_graph(2)), model=model)


@pytest.mark.parametrize(
    "kind", ["directed", "parallel", "loop", "disconnected", "singleton", "weighted"]
)
def test_invalid_support_is_rejected_by_shared_admission(kind):
    graph = nx.path_graph(2)
    if kind == "directed":
        graph = nx.DiGraph(graph)
    elif kind == "parallel":
        graph = nx.MultiGraph(graph)
    elif kind == "loop":
        graph.add_edge(0, 0)
    elif kind == "disconnected":
        graph.add_node(2)
    elif kind == "singleton":
        graph = nx.empty_graph(1)
    elif kind == "weighted":
        graph.edges[0, 1]["weight"] = 0.5
    graph = _state(graph)
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        owner.observe_relational_regularity(graph, model=MODEL)
    assert _snapshot(graph) == before
