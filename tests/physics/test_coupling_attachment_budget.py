"""One configured UM event can pay for its connection through phase relaxation.

These are endpoint budgets of actual shared operator execution, not a selected
event law, a continuous trajectory or a subsequent relational recovery claim.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_SI
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.operators.definitions import Coupling, Resonance
from tnfr.operators.network_stage import (
    _detached_stage_graph,
    execute_coupling_stage,
    execute_neighbor_stage,
)
from tnfr.physics.relational_observations import observe_relational_reset

H = math.pi / 8
ETA = 1 / 4


def _prepared(*, push=ETA, links=True):
    graph = nx.Graph(((0, 1), (2, 3)))
    graph.graph.update(
        RANDOM_SEED=17,
        UM_FUNCTIONAL_LINKS=links,
        UM_BIDIRECTIONAL=True,
        UM_CANDIDATE_COUNT=0,
        _node_sample=(2,),
        GLYPH_FACTORS={"UM_theta_push": push},
        GAMMA={"type": "none"},
        _t=0.0,
    )
    for node, phase in enumerate((0, 2 * H, 2 * H, 4 * H)):
        graph.nodes[node].update(
            EPI=0.5,
            theta=phase,
            nu_f=1.0,
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=["AL"],
        )
        graph.nodes[node][ALIAS_SI[0]] = 0.8
    return graph


@pytest.fixture(scope="module")
def cases():
    results = {}
    for name, kwargs in (
        ("coupling", {}),
        ("without_attachment", {"links": False}),
    ):
        graph = _prepared(**kwargs)
        before = deepcopy(graph)
        execution = execute_coupling_stage(graph, Coupling(), (1,))
        results[name] = before, graph, execution
    return results


def test_actual_um_pays_positive_attachment_cost_by_reducing_internal_phase_cost(cases):
    before, after, execution = cases["coupling"]
    originals = _detached_stage_graph(before), _detached_stage_graph(after)
    report = observe_relational_reset(before, after, storage_scale=1)

    assert set(after.edges) - set(before.edges) == {(1, 2)}
    assert nx.number_connected_components(before) == 2 and nx.is_connected(after)
    expected_phase = (ETA * H, (2 - ETA) * H, 2 * H, 4 * H)
    assert tuple(after.nodes[node]["theta"] for node in after) == pytest.approx(
        expected_phase, rel=0, abs=1e-15
    )
    assert tuple(after.nodes[node]["EPI"] for node in after) == (0.5,) * 4
    assert tuple(after.nodes[node]["nu_f"] for node in after) == (1.0,) * 4
    assert after.edges[1, 2]["weight"] == Q(63, 64)
    assert execution.nodes_processed == 1 and execution.glyph == "UM"
    assert before.graph["_t"] == after.graph["_t"] == 0.0
    assert tuple(after.nodes[1]["glyph_history"]) == ("AL", "UM")

    relaxation = math.cos(2 * H) - math.cos(2 * (1 - ETA) * H)
    attachment = 1 - math.cos(ETA * H)
    assert report.form_storage_change == 0
    assert float(report.phase_state_change) == pytest.approx(
        relaxation, rel=0, abs=2e-15
    )
    assert float(report.phase_support_change) == pytest.approx(
        attachment, rel=0, abs=2e-15
    )
    assert report.phase_state_change < 0 < report.phase_support_change
    assert float(report.storage_change) == pytest.approx(
        relaxation + attachment, rel=0, abs=2e-15
    )
    assert report.storage_change < 0
    assert report.storage_change == report.storage_after - report.storage_before
    assert report.form_state_change == report.form_support_change == 0
    assert report.phase_storage_change == (
        report.phase_state_change + report.phase_support_change
    )
    assert report.identity_residual == 0
    assert report.represented_zero_supply_passive
    assert report.assess_supply(0).represented_balance_satisfied
    assert nx.utils.graphs_equal(_detached_stage_graph(before), originals[0])
    assert nx.utils.graphs_equal(_detached_stage_graph(after), originals[1])


def test_frozen_postreset_triad_addition_keeps_its_positive_cost(cases):
    _, after, _ = cases["coupling"]
    pre_attachment = _detached_stage_graph(after)
    pre_attachment.remove_edge(1, 2)
    report = observe_relational_reset(pre_attachment, after, storage_scale=1)
    assert report.form_storage_change == 0
    assert report.phase_state_change == 0
    assert report.storage_change == report.phase_support_change > 0
    assert not report.represented_zero_supply_passive
    assert not report.assess_supply(0).represented_balance_satisfied
    assert report.assess_supply(report.storage_change).represented_balance_satisfied


def test_coincident_frozen_ports_and_disabled_links_separate_the_budget(cases):
    before, _, _ = cases["coupling"]
    after = deepcopy(before)
    after.add_edge(1, 2, weight=1.0)
    # This control is a supplied state-preserving attachment, not an executed
    # UM with zero push: the actual factor contract requires a positive push.
    neutral = observe_relational_reset(before, after, storage_scale=1)
    assert set(after.edges) - set(before.edges) == {(1, 2)}
    assert after.edges[1, 2]["weight"] == 1
    assert neutral.phase_state_change == neutral.phase_support_change == 0
    assert neutral.storage_change == 0 and neutral.represented_zero_supply_passive

    before, after, _ = cases["without_attachment"]
    contraction = observe_relational_reset(before, after, storage_scale=1)
    assert set(after.edges) == set(before.edges)
    assert contraction.phase_support_change == 0
    assert contraction.storage_change == contraction.phase_state_change < 0


def test_actual_um_rejects_zero_push_instead_of_silently_changing_its_contract():
    graph = _prepared(push=0.0)
    before = deepcopy(graph)
    with pytest.raises(ValueError, match="UM_theta_push"):
        execute_coupling_stage(graph, Coupling(), (1,))
    assert nx.utils.graphs_equal(graph, before)


def test_actual_nonunit_connection_does_not_inherit_unit_relational_execution(cases):
    _, after, _ = cases["coupling"]
    with pytest.raises(ValueError, match="unit conductances"):
        evaluate_relational_exchange(after, model=RelationalExchangeModel(1))


def test_actual_ra_then_um_can_pay_a_positive_form_attachment_cost():
    graph = _prepared()
    mean, contrast, mixing = Q(1, 2), Q(1, 8), Q(1, 4)
    for node, sign in enumerate((1, -1, -1, 1)):
        graph.nodes[node].update(EPI=float(mean + sign * contrast), theta=0.0)
    graph.graph["GLYPH_FACTORS"]["RA_epi_diff"] = float(mixing)
    before = deepcopy(graph)
    resonance = execute_neighbor_stage(graph, Resonance(), tuple(graph))
    after_resonance = _detached_stage_graph(graph)
    coupling = execute_coupling_stage(graph, Coupling(), (0,))
    report = observe_relational_reset(before, graph, storage_scale=1)
    attachment = observe_relational_reset(after_resonance, graph, storage_scale=1)

    contraction = 1 - 2 * mixing
    expected = tuple(mean + sign * contraction * contrast for sign in (1, -1, -1, 1))
    assert report.after.epi == expected
    assert set(graph.edges) - set(before.edges) == {(0, 2)}
    weight = Q(graph.edges[0, 2]["weight"])
    assert 0 < weight < 1
    assert resonance.nodes_processed == 4 and coupling.nodes_processed == 1
    assert report.storage_before == 4 * contrast**2
    assert report.storage_after == (4 + 2 * weight) * contraction**2 * contrast**2
    assert report.form_state_change == 4 * (contraction**2 - 1) * contrast**2
    assert report.form_support_change == 2 * weight * contraction**2 * contrast**2
    assert report.form_support_change == attachment.storage_change > 0
    assert report.phase_storage_change == 0
    assert report.storage_change < 0 and report.identity_residual == 0
    assert report.before.capacity == (1,) * 4
    assert all(capacity > 1 for capacity in report.after.capacity)
    assert report.represented_zero_supply_passive
    assert not attachment.represented_zero_supply_passive


def test_zero_conductance_support_retains_unweighted_phase_storage():
    before = _prepared(push=0)
    # The candidate is absent before and a phase neighbor afterward even when
    # its transport conductance is zero. Neither endpoint is advanced here.
    after = deepcopy(before)
    after.add_edge(0, 2, weight=0.0)
    report = observe_relational_reset(before, after, storage_scale=2)
    assert report.form_state_change == report.form_support_change == 0
    assert report.phase_state_change == 0
    assert float(report.phase_support_change) == pytest.approx(
        1 - math.cos(2 * H), rel=0, abs=2e-15
    )
    assert report.phase_support_change > 0
    assert report.storage_change == 2 * report.phase_support_change
    assert report.identity_residual == 0


@pytest.mark.parametrize("scale", (0, -1, True, math.nan, math.inf))
def test_invalid_storage_scale_cannot_certify_a_reset(cases, scale):
    before, after, _ = cases["coupling"]
    with pytest.raises((TypeError, ValueError)):
        observe_relational_reset(before, after, storage_scale=scale)


def test_changed_node_support_and_invalid_consumed_phase_reject(cases):
    before, after, _ = cases["coupling"]
    missing = _detached_stage_graph(after)
    missing.remove_node(3)
    with pytest.raises(ValueError):
        observe_relational_reset(before, missing, storage_scale=1)
    invalid = _detached_stage_graph(after)
    invalid.nodes[3]["theta"] = math.nan
    with pytest.raises((TypeError, ValueError)):
        observe_relational_reset(before, invalid, storage_scale=1)


@pytest.mark.parametrize("phase", (None, True, Q(1, 2**1200)))
def test_raw_phase_admission_does_not_invent_or_erase_coordinates(phase):
    before = _prepared()
    after = deepcopy(before)
    if phase is None:
        del after.nodes[0]["theta"]
    else:
        after.nodes[0]["theta"] = phase
    with pytest.raises((TypeError, ValueError)):
        observe_relational_reset(before, after, storage_scale=1)


@pytest.mark.parametrize("kind", ("directed", "parallel", "loop", "empty", "reordered"))
def test_reset_support_admission_is_explicit(kind):
    before = _prepared()
    after = deepcopy(before)
    if kind == "directed":
        after = nx.DiGraph(after)
    elif kind == "parallel":
        after = nx.MultiGraph(after)
    elif kind == "loop":
        after.add_edge(0, 0)
    elif kind == "empty":
        before, after = nx.Graph(), nx.Graph()
    else:
        after = nx.Graph()
        after.add_nodes_from(reversed(tuple(before.nodes(data=True))))
        after.add_edges_from(before.edges(data=True))
    with pytest.raises(ValueError):
        observe_relational_reset(before, after, storage_scale=1)


def test_storage_observation_does_not_require_an_acute_phase_field():
    before = _prepared()
    for node in before:
        before.nodes[node]["theta"] = 0.0
    after = deepcopy(before)
    after.nodes[0]["theta"] = math.pi
    report = observe_relational_reset(before, after, storage_scale=1)
    assert report.form_storage_change == 0
    assert report.phase_support_change == 0
    assert report.phase_state_change == report.storage_change == 2
