"""Static support-change controls for the supplied conditional relational law.

These tests compare fresh fields and support-event storage. They execute no
trajectory and do not select when an edge appears or certify recovery.
"""

import math
import pickle
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from benchmarks.relational_region_interaction import EPSILON, _graph
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.physics import relational_observations as owner
from tnfr.physics.relational_capture import certify_relational_capture
from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_reset,
)

MODEL = RelationalExchangeModel(1.0)
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _component(nodes, form, *, phase=0.0, capacity=(1.0, 1.0)):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edge(*nodes, weight=1.0, retained={"edge_history": [1, 2]})
    graph.graph.update(
        GAMMA={"type": "none"},
        _t=4.0,
        history={"notes": ["retained"]},
        DNFR_WEIGHTS={"epi": 0.1, "phase": 0.2, "vf": 0.3, "topo": 0.4},
    )
    for node, value, nu in zip(nodes, form, capacity, strict=True):
        graph.nodes[node].update(
            EPI=value,
            theta=phase,
            nu_f=nu,
            delta_nfr=999.0,
            history={"epi": [value]},
        )
    return graph


def _pair():
    return (
        _component((0, 1), (1.0, 0.0), capacity=(1.0, 2.0)),
        _component((2, 3), (-1.0, 0.0), capacity=(3.0, 4.0)),
    )


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _observe(left, right, *, model=MODEL, bridge=(0, 2)):
    return owner.observe_relational_attachment(left, right, model=model, bridge=bridge)


@pytest.fixture(scope="module")
def rational_cost_report():
    return _observe(*_pair())


def test_supply_margin_uses_exact_storage_without_crediting_continuous_loss(
    rational_cost_report,
):
    report = rational_cost_report
    # Equal-phase ports differ by two, so the unit bridge costs 2**2/2 = 2,
    # independently of beta and the continuous dissipation coefficient.
    assert report.storage_change == 2
    assert report.represented_zero_supply_passive is False
    tiny = Q(1, 2**1200)
    for work, margin, covered in (
        (Q(-1), Q(-3), False),
        (Q(0), Q(-2), False),
        (Q(2) - tiny, -tiny, False),
        (Q(2), Q(0), True),
        (Q(2) + tiny, tiny, True),
        (2.5, Q(1, 2), True),
    ):
        assessment = report.assess_supply(work)
        assert assessment.required_supply == 2
        assert assessment.supplied_work == Q(work)
        assert assessment.supply_margin == margin
        assert assessment.represented_balance_satisfied is covered
    assert report.continuous_loss_change == 7


@pytest.mark.parametrize(
    "invalid", (True, False, None, "2", 2 + 0j, math.inf, -math.inf, math.nan)
)
def test_supply_rejects_raw_invalid_work(rational_cost_report, invalid):
    with pytest.raises((TypeError, ValueError), match="supplied_work"):
        rational_cost_report.assess_supply(invalid)


def test_nonzero_tiny_attachment_cost_is_not_lost_at_the_supply_boundary():
    tiny = 2.0**-600
    report = _observe(
        _component((0, 1), (0.0, 0.0)),
        _component((2, 3), (tiny, tiny)),
    )
    cost = Q(1, 2**1201)
    assert float(cost) == 0.0  # This must not become the accounting input.
    assert report.storage_change == cost
    assert report.represented_zero_supply_passive is False
    assert report.assess_supply(0).supply_margin == -cost
    assert report.assess_supply(0).represented_balance_satisfied is False
    assert report.assess_supply(cost).represented_balance_satisfied is True


def test_supplied_edge_relocation_can_release_storage_without_changing_the_triad():
    # This is an accounting witness, not an attachment API or event selector.
    before_graph = nx.path_graph(4)
    for node, form in enumerate((0.0, 2.0, 1.0, 1.0)):
        before_graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0, delta_nfr=0.0)
    after_graph = before_graph.copy()
    after_graph.remove_edge(0, 1)
    after_graph.add_edge(0, 2)
    snapshots = (_snapshot(before_graph), _snapshot(after_graph))
    before = owner.evaluate_relational_exchange(before_graph, model=MODEL)
    after = owner.evaluate_relational_exchange(after_graph, model=MODEL)
    reset = observe_support_transport_reset(
        observe_support_transport(before_graph), observe_support_transport(after_graph)
    )
    assert nx.is_connected(before_graph) and nx.is_connected(after_graph)
    assert before.epi == after.epi == (0.0, 2.0, 1.0, 1.0)
    assert before.phase == after.phase == (0.0,) * 4
    assert before.capacity == after.capacity == (1.0,) * 4
    assert before.phase_storage == after.phase_storage == 0
    assert before.storage == Q(5, 2)
    assert after.storage == 1
    assert after.storage - before.storage == reset.energy_change == Q(-3, 2)
    assert reset.edge_energy_change == Q(-3, 2)
    assert reset.identity_residual == 0
    assert (_snapshot(before_graph), _snapshot(after_graph)) == snapshots


def test_supply_assessment_neither_refreshes_fields_nor_changes_retained_state(
    monkeypatch,
):
    left, right = _pair()
    report = _observe(left, right)
    before = (_snapshot(left), _snapshot(right), pickle.dumps(report))

    def unexpected_refresh(*args, **kwargs):
        pytest.fail("Supply assessment must reuse the captured storage")

    monkeypatch.setattr(owner, "evaluate_relational_exchange", unexpected_refresh)
    assessment = report.assess_supply(2)
    assert (_snapshot(left), _snapshot(right), pickle.dumps(report)) == before
    with pytest.raises(FrozenInstanceError):
        assessment.supplied_work = Q(0)
    with pytest.raises(TypeError):
        report.assess_supply()


def test_two_pairs_have_the_analytic_degree_gradient_and_rate_changes():
    left, right = _pair()
    report = _observe(left, right, model=RelationalExchangeModel(2.0))
    field = report.joined
    assert field.nodes == (0, 1, 2, 3)

    # Before attachment each endpoint has one neighbor. The new difference
    # x_0-x_2=2 changes the port gradients from (+1,-1) to (+3,-3).
    assert tuple(port.before.degree for port in report.ports) == (1, 1)
    assert tuple(port.after.degree for port in report.ports) == (2, 2)
    assert tuple(port.before.form_gradient for port in report.ports) == (1, -1)
    assert tuple(port.after.form_gradient for port in report.ports) == (3, -3)
    assert field.work.form_gradient == (3, -1, -3, 1)
    assert field.pressure == (-0.75, 0.5, 0.75, -0.5)
    assert field.form_rate == (-0.75, 1.0, 2.25, -2.0)
    assert report.form_rate_change == (Q(-1, 4), Q(0), Q(3, 4), Q(0))
    assert report.pressure_change == (Q(-1, 4), Q(0), Q(1, 4), Q(0))

    # At phase consensus H=pi*d and theta_dot=(w/beta)*nu*q/H.
    inverse_pi = 1 / Q(math.pi)
    exact_phase_rate = tuple(
        coefficient * inverse_pi for coefficient in (Q(3, 8), Q(-1, 2), Q(-9, 8), Q(1))
    )
    for actual, expected, defect in zip(
        field.phase_rate,
        exact_phase_rate,
        field.phase_rate_rounding_defect,
        strict=True,
    ):
        assert Q(actual) == expected + defect
    assert report.phase_metric_change == (Q(math.pi), Q(0), Q(math.pi), Q(0))
    assert all(port.before.relative_resultant == (1.0, 0.0) for port in report.ports)
    assert all(port.after.relative_resultant == (2.0, 0.0) for port in report.ports)

    # The shared cut and reset owners account for the one unit bridge.
    assert report.cut.region == (0, 1)
    assert report.cut.environment == (2, 3)
    assert report.cut.cut_edges == ((0, 2, Q(1)),)
    assert report.cut.outward_cut_current == 2
    assert report.form_storage_change == 2
    assert report.phase_storage_change == 0
    assert report.storage_change == 2
    assert report.transport_reset.energy_change == 2
    assert report.transport_reset.edge_energy_change == 2
    assert report.transport_reset.identity_residual == 0


def test_equal_endpoint_c5_control_changes_the_donor_field_at_zero_cut():
    # Reuse only the previously declared preparation, never its response runner.
    left, right = _graph("left"), _graph("right")
    report = _observe(left, right, bridge=(0, 5))
    donor, receiver = report.ports
    c = (math.sqrt(5.0) - 1.0) / 4.0
    h = 1.0 + 2.0 * c
    epsilon = float(EPSILON)
    assert donor.before.form_gradient == donor.after.form_gradient == -EPSILON
    assert donor.before.degree == 2 and donor.after.degree == 3
    assert receiver.before.epi == donor.before.epi == 0.0
    assert receiver.before.phase == donor.before.phase == 0.0

    # Ideal donor rates are e*epsilon/d and -w*epsilon/(pi*s), where
    # s=2*cos(2*pi/5) before and s+1 afterward. The actual phase evaluation
    # uses binary64 angles/trigonometry; this comparison is not an enclosure.
    assert donor.before.form_rate == pytest.approx(epsilon / 4, rel=0, abs=1e-15)
    assert donor.after.form_rate == pytest.approx(epsilon / 6, rel=0, abs=1e-15)
    assert float(report.form_rate_change[0]) == pytest.approx(
        -epsilon / 12, rel=0, abs=1e-15
    )
    assert donor.before.phase_rate == pytest.approx(
        -epsilon / (4 * math.pi * c), rel=0, abs=1e-15
    )
    assert donor.after.phase_rate == pytest.approx(
        -epsilon / (2 * math.pi * h), rel=0, abs=1e-15
    )
    # 2*c*(1+2*c)=1 makes the ideal phase-rate change epsilon/(2*pi).
    assert float(report.phase_rate_change[0]) == pytest.approx(
        epsilon / (2 * math.pi), rel=0, abs=1e-15
    )
    assert report.cut.outward_cut_current == 0
    assert report.form_storage_change == report.phase_storage_change == 0
    assert report.storage_change == report.transport_reset.energy_change == 0
    assert report.transport_reset.identity_residual == 0
    interiors = tuple(
        index
        for index, node in enumerate(report.joined.nodes)
        if node not in report.bridge
    )
    assert all(report.form_rate_change[i] == 0 for i in interiors)
    assert all(report.phase_rate_change[i] == 0 for i in interiors)
    # q and capacity do not change at the donor, but its degree grows 2 -> 3:
    # e*epsilon**2*(1/3 - 1/2) = -epsilon**2/12.
    assert report.continuous_loss_change == -(EPSILON**2) / 12
    assert report.represented_zero_supply_passive is True
    assert report.assess_supply(0).represented_balance_satisfied is True
    assert report.assess_supply(-Q(1, 2**1200)).represented_balance_satisfied is False
    assert report.form_rate_change[0] != 0
    assert report.phase_rate_change[0] != 0


def test_endpoint_difference_adds_exact_represented_form_and_phase_storage():
    left = _component((0, 1), (1.0, 0.0))
    right = _component((2, 3), (-1.0, 0.0), phase=0.25)
    report = _observe(left, right, model=RelationalExchangeModel(3.0))
    # Existing edges have zero phase gaps. The new bridge alone adds its
    # unit-edge cost; this uses the engine's represented sine-half expression.
    expected_form = Q(1 - (-1)) ** 2 / 2
    expected_phase = 2 * Q(math.sin(0.125)) ** 2
    assert report.form_storage_change == expected_form
    assert report.phase_storage_change == expected_phase
    assert report.storage_change == expected_form + 3 * expected_phase
    assert report.transport_reset.energy_change == expected_form
    assert report.transport_reset.identity_residual == 0
    assert report.ports[0].after.relative_resultant == pytest.approx(
        (1 + math.cos(0.25), math.sin(0.25))
    )
    assert report.ports[1].after.relative_resultant == pytest.approx(
        (1 + math.cos(0.25), -math.sin(0.25))
    )
    assert report.joined.epi == (1.0, 0.0, -1.0, 0.0)
    assert report.joined.phase == (0.0, 0.0, 0.25, 0.25)


def test_one_bridge_admission_does_not_expand_the_two_bridge_capture_theorem():
    with pytest.raises(ValueError, match="support must be"):
        certify_relational_capture(
            _graph("joined"),
            model=MODEL,
            cycles=(tuple(range(5)), tuple(range(5, 10))),
        )


def test_report_retains_exact_differences_instead_of_rounded_subtractions():
    tiny = 2.0**-55
    left = _component((0, 1), (0.0, tiny))
    right = _component((2, 3), (1.0, 0.0))
    report = _observe(left, right)
    port = report.ports[0]
    assert port.before.form_gradient == -Q(tiny)
    assert port.after.form_gradient == -1 - Q(tiny)
    assert Q(report.joined.form_gradient[0]) != port.after.form_gradient
    assert report.form_rate_change[0] == Q(port.after.form_rate) - Q(
        port.before.form_rate
    )
    assert report.form_rate_change[0] != Q(port.after.form_rate - port.before.form_rate)
    before = {
        node: (component, index)
        for component in report.components
        for index, node in enumerate(component.nodes)
    }
    for name in ("form_rate", "phase_rate", "pressure", "phase_metric"):
        for index, node in enumerate(report.joined.nodes):
            component, old_index = before[node]
            assert getattr(report, name + "_change")[index] == (
                Q(getattr(report.joined, name)[index])
                - Q(getattr(component, name)[old_index])
            )


def test_zero_capacity_preserves_pressure_cut_and_event_evidence():
    left, right = _pair()
    for graph, node in ((left, 0), (right, 2)):
        graph.nodes[node]["nu_f"] = 0.0
    report = _observe(left, right)
    assert report.cut.outward_cut_current == 2
    assert report.storage_change == 2
    for port in report.ports:
        assert port.before.capacity == port.after.capacity == 0.0
        assert port.before.form_rate == port.after.form_rate == 0.0
        assert port.before.phase_rate == port.after.phase_rate == 0.0
        assert port.after.pressure != 0.0
    assert report.transport_reset.identity_residual == 0


def test_observation_preserves_labels_order_attributes_histories_and_live_state():
    left = _component(("port", ("inside", 1)), (1.0, 0.0))
    right = _component((17, -4), (-1.0, 0.0))
    right.graph["_t"] = 8.0  # The static observer invents no common event clock.
    snapshots = tuple(_snapshot(graph) for graph in (left, right))
    report = _observe(left, right, bridge=("port", 17))
    assert tuple(_snapshot(graph) for graph in (left, right)) == snapshots
    assert report.joined.nodes == ("port", ("inside", 1), 17, -4)
    assert report.bridge == ("port", 17)
    assert report.transport_reset.before.nodes == report.transport_reset.after.nodes
    assert report.transport_reset.before.epi == report.transport_reset.after.epi
    with pytest.raises(FrozenInstanceError):
        report.bridge = (17, "port")
    with pytest.raises(FrozenInstanceError):
        report.ports[0].after.form_gradient = Q(0)
    left.nodes["port"]["EPI"] = 100.0
    assert report.joined.epi == (1.0, 0.0, -1.0, 0.0)


@pytest.mark.parametrize("bridge", [(0, 1), (2, 0), (0, 99), (0, 0), (0, 2, 3), {0, 2}])
def test_attachment_rejects_non_crossing_or_unordered_bridge_without_writes(bridge):
    left, right = _pair()
    snapshots = tuple(_snapshot(graph) for graph in (left, right))
    with pytest.raises(ADMISSION_ERRORS):
        _observe(left, right, bridge=bridge)
    assert tuple(_snapshot(graph) for graph in (left, right)) == snapshots


def test_bridge_cardinality_rejection_consumes_at_most_three_items():
    left, right = _pair()
    snapshots = tuple(_snapshot(graph) for graph in (left, right))
    consumed = []

    def bridge():
        for node in (0, 2, 3):
            consumed.append(node)
            yield node
        raise AssertionError("The invalid bridge iterator must not be exhausted")

    with pytest.raises(ADMISSION_ERRORS):
        _observe(left, right, bridge=bridge())
    assert consumed == [0, 2, 3]
    assert tuple(_snapshot(graph) for graph in (left, right)) == snapshots


@pytest.mark.parametrize(
    "defect",
    ["overlap", "disconnected", "weighted", "acute_boundary", "forcing", "capacity"],
)
def test_invalid_components_or_joined_domain_reject_atomically(defect):
    left, right = _pair()
    if defect == "overlap":
        right = nx.relabel_nodes(right, {3: 1})
    elif defect == "disconnected":
        right.remove_edge(2, 3)
    elif defect == "weighted":
        right.edges[2, 3]["weight"] = 2.0
    elif defect == "acute_boundary":
        for node in right:
            right.nodes[node]["theta"] = math.pi / 2
    elif defect == "forcing":
        right.graph["GAMMA"] = {"type": "external"}
    else:
        right.nodes[3]["nu_f"] = -1.0
    snapshots = tuple(_snapshot(graph) for graph in (left, right))
    with pytest.raises(ADMISSION_ERRORS):
        _observe(left, right)
    assert tuple(_snapshot(graph) for graph in (left, right)) == snapshots


@pytest.mark.parametrize("domain", ("acute", "positive_resultant", "regular"))
def test_attachment_uses_the_same_selected_domain_for_all_three_fields(domain):
    left, right = _pair()
    model = RelationalExchangeModel(1.0, phase_domain=domain)
    report = _observe(left, right, model=model)
    assert all(field.model == model for field in (*report.components, report.joined))


def test_regular_attachment_admits_negative_real_resultants_without_writes():
    left, right = _pair()
    left.nodes[1]["theta"] = right.nodes[3]["theta"] = 2.0
    before = tuple(_snapshot(graph) for graph in (left, right))
    with pytest.raises(ValueError, match="positive"):
        _observe(
            left,
            right,
            model=RelationalExchangeModel(1.0, phase_domain="positive_resultant"),
        )
    report = _observe(
        left, right, model=RelationalExchangeModel(1.0, phase_domain="regular")
    )
    assert all(field.relative_resultant[0][0] < 0 for field in report.components)
    assert all(
        field.pressure_path == "relative_resultant_canonical"
        for field in (*report.components, report.joined)
    )
    assert report.phase_storage_change == 0
    assert tuple(_snapshot(graph) for graph in (left, right)) == before


def test_regular_attachment_rejects_a_joined_resultant_on_the_argument_cut():
    left, right = _pair()
    left.nodes[1]["theta"] = 2.0
    right.nodes[2]["theta"] = -2.0
    before = tuple(_snapshot(graph) for graph in (left, right))
    # Both isolated pairs are regular. At joined node zero the two relative
    # phasors exp(2i)+exp(-2i) sum to a strictly negative real number.
    with pytest.raises(ValueError, match="nonpositive-real ray"):
        _observe(
            left, right, model=RelationalExchangeModel(1.0, phase_domain="regular")
        )
    assert tuple(_snapshot(graph) for graph in (left, right)) == before


def _relocation_graph():
    # Reuse only support/phase preparation; this witness perturbs node zero,
    # independently of the old interaction benchmark's node-one perturbation.
    graph = _graph("joined")
    for node in graph:
        graph.nodes[node]["EPI"] = float(EPSILON) if node == 0 else 0.0
    return graph


def _relocate(graph, *, old=(0, 5), new=(1, 6), model=MODEL):
    return owner.observe_relational_relocation(
        graph, model=model, remove_bridge=old, add_bridge=new
    )


@pytest.fixture(scope="module")
def relocation_report():
    return _relocate(_relocation_graph())


def test_c5_relocation_has_independent_storage_gradient_and_rate_oracles(
    relocation_report,
):
    report = relocation_report
    epsilon = EPSILON
    assert report.components == (tuple(range(5)), tuple(range(5, 10)))
    assert tuple(port.before.node for port in report.ports) == (0, 1, 5, 6)
    assert report.before.epi == report.after.epi == (float(epsilon),) + (0.0,) * 9
    assert report.before.phase == report.after.phase
    assert report.before.capacity == report.after.capacity == (1.0,) * 10
    assert report.before.work.form_gradient == tuple(
        epsilon * n for n in (3, -1, 0, 0, -1, -1, 0, 0, 0, 0)
    )
    assert report.after.work.form_gradient == tuple(
        epsilon * n for n in (2, -1, 0, 0, -1, 0, 0, 0, 0, 0)
    )
    assert report.before.form_storage == 3 * epsilon**2 / 2
    assert report.after.form_storage == epsilon**2
    assert report.phase_storage_change == 0
    assert report.storage_change == report.form_storage_change == -(epsilon**2) / 2
    assert report.transport_reset.energy_change == -(epsilon**2) / 2
    assert report.transport_reset.edge_energy_change == -(epsilon**2) / 2
    assert report.transport_reset.identity_residual == 0
    assert report.before.continuous_loss == 13 * epsilon**2 / 6
    assert report.after.continuous_loss == 17 * epsilon**2 / 12
    assert report.continuous_loss_change == -3 * epsilon**2 / 4
    assert report.cut_before.cut_edges == ((0, 5, Q(1)),)
    assert report.cut_after.cut_edges == ((1, 6, Q(1)),)
    assert report.cut_before.outward_cut_current == epsilon
    assert report.cut_after.outward_cut_current == 0

    # Ideal aligned-C5 formulas are compared to represented trigonometry.
    # These tolerances are not exact-real enclosures or a recovery certificate.
    c = (math.sqrt(5.0) - 1.0) / 4.0
    h = 1 + 2 * c
    factor = float(epsilon) / (2 * math.pi)
    expected_form = [0.0] * 10
    expected_form[1], expected_form[5] = -float(epsilon) / 12, -float(epsilon) / 6
    expected_phase = [0.0] * 10
    expected_phase[0] = factor * (1 / c - 3 / h)
    expected_phase[1] = factor * (1 / (2 * c) - 1 / h)
    expected_phase[5] = factor / h
    assert tuple(map(float, report.form_rate_change)) == pytest.approx(
        expected_form, rel=0, abs=1e-15
    )
    assert tuple(map(float, report.phase_rate_change)) == pytest.approx(
        expected_phase, rel=0, abs=1e-15
    )
    for i in (2, 3, 4, 7, 8, 9):
        assert report.form_rate_change[i] == report.phase_rate_change[i] == 0
        assert report.pressure_change[i] == report.phase_metric_change[i] == 0
    old_internal = set(report.before.edges) - {(0, 5)}
    assert set(report.after.edges) - {(1, 6)} == old_internal


def test_relocation_reverse_and_zero_controls_do_not_select_events(relocation_report):
    report = relocation_report
    assert report.represented_zero_supply_passive is True
    released = EPSILON**2 / 2
    assert report.assess_supply(-released).represented_balance_satisfied is True
    assert report.assess_supply(-released - Q(1, 2**1200)).supply_margin < 0
    graph = _relocation_graph()
    graph.remove_edge(0, 5)
    graph.add_edge(1, 6, weight=1.0)
    reverse = _relocate(graph, old=(1, 6), new=(0, 5))
    assert reverse.storage_change == released
    assert reverse.represented_zero_supply_passive is False
    assert reverse.assess_supply(0).represented_balance_satisfied is False
    for node in graph:
        graph.nodes[node]["EPI"] = 0.0
    zero = _relocate(graph, old=(1, 6), new=(0, 5))
    assert zero.storage_change == zero.continuous_loss_change == 0
    assert zero.represented_zero_supply_passive is True
    # A zero event budget still changes the local metric.
    assert any(zero.phase_metric_change)


def test_relocation_shared_port_and_zero_capacity_keep_full_accounting():
    left, right = _pair()
    graph = nx.compose(left, right)
    graph.add_edge(1, 2, weight=1.0)
    graph.nodes[2]["nu_f"] = 0.0
    report = _relocate(graph, old=(1, 2), new=(0, 2))
    assert tuple(port.before.node for port in report.ports) == (0, 1, 2)
    common = report.ports[2]
    assert common.before.degree == common.after.degree == 2
    assert common.before.form_gradient == -2
    assert common.after.form_gradient == -3
    assert common.before.form_rate == common.after.form_rate == 0
    assert common.before.phase_rate == common.after.phase_rate == 0
    assert report.storage_change == Q(3, 2)
    assert report.cut_before.outward_cut_current == 1
    assert report.cut_after.outward_cut_current == 2


def test_relocation_uses_two_fresh_connected_fields_without_live_writes(monkeypatch):
    graph = _relocation_graph()
    graph.graph["history"] = {"keep": [1, 2]}
    graph.nodes[0]["delta_nfr"] = 999.0
    graph.nodes[0]["history"] = {"keep": [3, 4]}
    before = _snapshot(graph)
    evaluate = owner.evaluate_relational_exchange
    calls = []

    def capture(source, **kwargs):
        assert nx.is_connected(source)
        calls.append(source)
        return evaluate(source, **kwargs)

    monkeypatch.setattr(owner, "evaluate_relational_exchange", capture)
    report = _relocate(graph)
    assert len(calls) == 2 and calls[0] is graph and calls[1] is not graph
    assert report.before.pressure[0] != 999.0
    assert _snapshot(graph) == before
    for name in ("form_rate", "phase_rate", "pressure", "phase_metric"):
        assert getattr(report, name + "_change") == tuple(
            Q(new) - Q(old)
            for old, new in zip(
                getattr(report.before, name), getattr(report.after, name)
            )
        )
    with pytest.raises(FrozenInstanceError):
        report.add_bridge = (2, 7)
    graph.nodes[0]["EPI"] = 42.0
    assert report.after.epi[0] == float(EPSILON)


@pytest.mark.parametrize(
    "old,new",
    (
        ((0, 1), (1, 6)),  # Internal cycle edge is not a graph bridge.
        ((0, 2), (1, 6)),  # Missing old edge.
        ((0, 5), (0, 2)),  # Missing but internal new edge.
        ((0, 5), (0, 1)),  # Already present new edge.
        ((0, 5), (0, 5)),  # No-op, including reverse orientation below.
        ((0, 5), (5, 0)),
        ((0, 5), (6, 1)),  # The supplied new orientation must match the cut.
        ((0, 5), (1, 99)),
        ((0, 5), (1, 1)),
        ((0, 5), ([], 6)),
        ((0, 5), (1, 6, 7)),
        ((0, 5), {1, 6}),
    ),
)
def test_invalid_relocation_ports_reject_without_writes(old, new):
    graph = _relocation_graph()
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        _relocate(graph, old=old, new=new)
    assert _snapshot(graph) == before


def test_relocation_requires_two_nontrivial_components():
    graph = nx.path_graph(4)
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=0.0, nu_f=1.0)
    before = _snapshot(graph)
    with pytest.raises(ValueError, match="nontrivial components"):
        _relocate(graph, old=(0, 1), new=(0, 2))
    assert _snapshot(graph) == before


@pytest.mark.parametrize("defect", ("model", "weight", "forcing", "acute"))
def test_relocation_native_model_admission_is_not_bypassed(defect):
    graph = _relocation_graph()
    model = MODEL
    new = (1, 6)
    if defect == "model":
        model = None
    elif defect == "weight":
        graph.edges[0, 5]["weight"] = 0.5
    elif defect == "forcing":
        graph.graph["GAMMA"] = {"type": "external"}
    else:
        new = (1, 8)  # Original support is acute; this new cross-edge is not.
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        _relocate(graph, new=new, model=model)
    assert _snapshot(graph) == before


@pytest.mark.parametrize("domain", ("acute", "positive_resultant", "regular"))
def test_relocation_uses_the_same_selected_domain_before_and_after(domain):
    graph = _relocation_graph()
    model = RelationalExchangeModel(1.0, phase_domain=domain)
    report = _relocate(graph, model=model)
    assert report.before.model == report.after.model == model


def test_regular_relocation_retains_nonacute_internal_edges_and_event_cost():
    left, right = _pair()
    left.nodes[1]["theta"] = right.nodes[3]["theta"] = 2.0
    graph = nx.compose(left, right)
    graph.add_edge(1, 2, weight=1.0)
    before = _snapshot(graph)
    report = _relocate(
        graph,
        old=(1, 2),
        new=(0, 2),
        model=RelationalExchangeModel(1.0, phase_domain="regular"),
    )
    assert report.phase_storage_change == -2 * Q(math.sin(1.0)) ** 2
    assert report.phase_storage_change < 0
    assert (
        report.before.pressure_path
        == report.after.pressure_path
        == "relative_resultant_canonical"
    )
    assert _snapshot(graph) == before


def test_regular_relocation_rechecks_the_new_resultants_before_returning():
    left, right = _pair()
    left.nodes[1]["theta"] = 2.0
    right.nodes[2]["theta"] = -2.0
    graph = nx.compose(left, right)
    graph.add_edge(1, 2, weight=1.0)
    before = _snapshot(graph)
    with pytest.raises(ValueError, match="nonpositive-real ray"):
        _relocate(
            graph,
            old=(1, 2),
            new=(0, 2),
            model=RelationalExchangeModel(1.0, phase_domain="regular"),
        )
    assert _snapshot(graph) == before
