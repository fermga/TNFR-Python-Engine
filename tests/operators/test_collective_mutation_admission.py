"""Read-only ZHIR admission and its independent temporal/grammar obligations.

The contact successor is one contextual control. Other records below are
explicitly manufactured reader inputs, not responses of that sine law, and
no Mutation event or continuous trajectory is executed by these tests.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.operators.test_collective_reception import _contact, _fields
from tests.operators.test_collective_transition import PORTS, _preparation, _transition
from tnfr.mathematics import BEPIElement
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.operators._mutation_gate import (
    mutation_threshold_sample,
    validate_mutation_runtime_gate,
)
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.mutation import diagnose_mutation_readiness
from tnfr.physics.mutation_trigger import certify_mutation_trigger
from tnfr.types import require_finite_real_scalar_epi, serialize_bepi


@pytest.fixture(scope="module")
def nav_successor():
    graph = _preparation()
    _transition(graph)
    return graph


def _reader_graph(*, temporal=True, recent_destabilizer=True):
    """Manufacture independent reader inputs without asserting their dynamics."""
    graph = nx.Graph(ZHIR_THRESHOLD_XI=0.1)
    graph.add_node(
        "receiver",
        EPI=-0.25,
        nu_f=0.5,
        delta_nfr=0.125,
        theta=0.0,
        glyph_history=["IL", "OZ"] if recent_destabilizer else ["IL", "SHA", "NAV"],
        _grammar_u2_debt=1 if recent_destabilizer else 0,
        _grammar_prior_coherence=True,
    )
    if temporal:
        graph.nodes["receiver"]["epi_time_history"] = [(10.0, -0.5), (12.0, -0.25)]
    return graph


def _certificate(node, metadata):
    return certify_mutation_trigger(
        current_epi=require_finite_real_scalar_epi(node["EPI"]),
        nu_f=node["nu_f"],
        delta_nfr=node["delta_nfr"],
        xi=metadata.get("ZHIR_THRESHOLD_XI", 0.1),
        epi_time_history=node.get("epi_time_history"),
        epi_history=node.get("epi_history"),
        legacy_epi_history=node.get("_epi_history"),
    )


def test_nav_successor_has_two_separate_missing_admission_obligations(nav_successor):
    before_nodes = deepcopy(dict(nav_successor.nodes(data=True)))
    for node in PORTS:
        data = nav_successor.nodes[node]
        assert tuple(data["glyph_history"])[-3:] == ("IL", "SHA", "NAV")
        certificate = _certificate(data, nav_successor.graph)
        assert not certificate.evidence_available
        assert certificate.reason == "missing_history"
        assert certificate.observed_depi_dt is None
        assert certificate.observed_crossed is None
        assert certificate.capacity_active
        with pytest.raises(OperatorPreconditionError, match="at least two"):
            validate_mutation_runtime_gate(data, nav_successor.graph)
        grammar = validate_candidate(nav_successor, node, "ZHIR")
        assert not grammar.allowed
        assert {item.rule for item in grammar.violations} == {"U4b"}
        readiness = diagnose_mutation_readiness(nav_successor, node)
        assert not readiness["ready"]
        assert readiness["checks"]["minimum_vf"]["passed"]
        assert readiness["checks"]["il_precedence"]["passed"]
        assert not readiness["checks"]["threshold_crossing"]["passed"]
        assert not readiness["checks"]["recent_destabilizer"]["passed"]
    assert dict(nav_successor.nodes(data=True)) == before_nodes


def test_general_positive_nodal_prediction_and_valid_grammar_do_not_supply_history():
    graph = _reader_graph(temporal=False)
    node = graph.nodes["receiver"]
    node["delta_nfr"] = 2.0
    before = deepcopy((node, graph.graph))
    certificate = _certificate(node, graph.graph)
    assert certificate.predicted_depi_dt == 1.0
    assert certificate.predicted_crossed
    assert certificate.observed_crossed is None
    assert not certificate.threshold_gate_satisfied
    assert validate_candidate(graph, "receiver", "ZHIR").allowed
    with pytest.raises(OperatorPreconditionError, match="at least two"):
        validate_mutation_runtime_gate(node, graph.graph)
    assert (node, graph.graph) == before


def test_general_observed_crossing_does_not_supply_recent_grammar_context():
    graph = _reader_graph(recent_destabilizer=False)
    node = graph.nodes["receiver"]
    before = deepcopy((node, graph.graph))
    gate = validate_mutation_runtime_gate(node, graph.graph)
    assert gate.threshold.depi_dt == 0.125
    assert gate.threshold.crossed
    assert not gate.threshold.certificate.predicted_crossed
    grammar = validate_candidate(graph, "receiver", "ZHIR")
    assert not grammar.allowed
    assert {item.rule for item in grammar.violations} == {"U4b"}
    report = diagnose_mutation_readiness(graph, "receiver")
    assert report["checks"]["threshold_crossing"]["passed"]
    assert not report["checks"]["recent_destabilizer"]["passed"]
    assert not report["ready"]
    assert (node, graph.graph) == before


def test_general_admitted_reader_state_is_not_excluded_by_one_sine_preparation():
    graph = _reader_graph()
    node = graph.nodes["receiver"]
    before = deepcopy((node, graph.graph))
    gate = validate_mutation_runtime_gate(node, graph.graph)
    assert gate.nu_f == 0.5
    assert gate.threshold.depi_dt == 0.125
    assert gate.threshold.certificate.predicted_depi_dt == 0.0625
    assert gate.threshold.certificate.rate_gap == 0.0625
    assert validate_candidate(graph, "receiver", "ZHIR").allowed
    report = diagnose_mutation_readiness(graph, "receiver")
    assert report["ready"]
    assert (node, graph.graph) == before


def test_collective_mean_or_unordered_histories_lose_target_trigger_information():
    graph = _reader_graph()
    graph.add_node("other", **deepcopy(graph.nodes["receiver"]))
    graph.nodes["other"]["epi_time_history"] = [(10.0, 0.0), (12.0, -0.25)]
    graph.add_edge("receiver", "other")
    mean_history = tuple(
        sum(Q(graph.nodes[node]["epi_time_history"][sample][1]) for node in graph) / 2
        for sample in (0, 1)
    )
    assert mean_history[1] - mean_history[0] == 0
    growing = validate_mutation_runtime_gate(graph.nodes["receiver"], graph.graph)
    assert growing.threshold.depi_dt == 0.125
    with pytest.raises(OperatorPreconditionError, match="signed dEPI/dt > xi"):
        validate_mutation_runtime_gate(graph.nodes["other"], graph.graph)
    swapped = deepcopy(graph)
    (
        swapped.nodes["receiver"]["epi_time_history"],
        swapped.nodes["other"]["epi_time_history"],
    ) = (
        swapped.nodes["other"]["epi_time_history"],
        swapped.nodes["receiver"]["epi_time_history"],
    )
    assert sorted(
        tuple(data["epi_time_history"]) for _, data in graph.nodes(data=True)
    ) == sorted(tuple(data["epi_time_history"]) for _, data in swapped.nodes(data=True))
    assert all(
        tuple(graph.nodes[node][key] for key in ("EPI", "nu_f", "delta_nfr", "theta"))
        == tuple(
            swapped.nodes[node][key] for key in ("EPI", "nu_f", "delta_nfr", "theta")
        )
        for node in graph
    )
    with pytest.raises(OperatorPreconditionError, match="signed dEPI/dt > xi"):
        validate_mutation_runtime_gate(swapped.nodes["receiver"], swapped.graph)
    assert validate_mutation_runtime_gate(
        swapped.nodes["other"], swapped.graph
    ).threshold.crossed


@pytest.mark.parametrize("serialized", [False, True])
def test_signed_collective_form_reader_preserves_negative_uniform_embedding(serialized):
    graph = _reader_graph()
    node = graph.nodes["receiver"]
    form = BEPIElement((-0.25, -0.25), (-0.25, -0.25), (0.0, 1.0))
    node["EPI"] = serialize_bepi(form) if serialized else form
    gate = validate_mutation_runtime_gate(node, graph.graph)
    assert gate.threshold.previous_epi == -0.5
    assert gate.threshold.current_epi == -0.25
    assert gate.threshold.depi_dt == 0.125
    assert gate.threshold.physical_time_resolved
    assert gate.threshold.certificate.current_endpoint_matches_state


def test_strict_threshold_does_not_accept_equal_observed_growth():
    graph = _reader_graph()
    graph.graph["ZHIR_THRESHOLD_XI"] = 0.125
    node = graph.nodes["receiver"]
    sample = mutation_threshold_sample(node, graph.graph)
    assert sample.depi_dt == sample.xi == 0.125
    assert not sample.crossed
    with pytest.raises(OperatorPreconditionError, match="signed dEPI/dt > xi"):
        validate_mutation_runtime_gate(node, graph.graph)


def test_operator_step_and_physical_histories_have_distinct_units():
    graph = _reader_graph()
    node = graph.nodes["receiver"]
    node["epi_time_history"] = [(8.0, -0.5), (12.0, -0.25)]
    node["epi_history"] = [-0.5, -0.25]
    physical = mutation_threshold_sample(node, graph.graph)
    assert physical.time_basis == "physical_time"
    assert physical.sample_interval == 4.0
    assert physical.depi_dt == 0.0625
    assert not physical.crossed
    legacy_node = deepcopy(node)
    del legacy_node["epi_time_history"]
    legacy = mutation_threshold_sample(legacy_node, graph.graph)
    assert legacy.time_basis == "legacy_unit_operator_step"
    assert legacy.sample_interval == 1.0
    assert legacy.depi_dt == 0.25
    assert legacy.crossed
    assert not legacy.physical_time_resolved
    assert legacy.certificate.current_endpoint_matches_state is None
    assert legacy.certificate.rate_gap is None


def test_reader_clock_rescaling_transforms_capacity_and_threshold():
    graph = _reader_graph()
    original = _certificate(graph.nodes["receiver"], graph.graph)
    scale = 4
    node = deepcopy(graph.nodes["receiver"])
    node["nu_f"] /= scale
    node["epi_time_history"] = [
        (scale * time, epi) for time, epi in node["epi_time_history"]
    ]
    correctly_scaled = _certificate(
        node, {"ZHIR_THRESHOLD_XI": graph.graph["ZHIR_THRESHOLD_XI"] / scale}
    )
    assert correctly_scaled.predicted_depi_dt == original.predicted_depi_dt / scale
    assert correctly_scaled.observed_depi_dt == original.observed_depi_dt / scale
    assert correctly_scaled.rate_gap == original.rate_gap / scale
    assert (
        correctly_scaled.threshold_gate_satisfied == original.threshold_gate_satisfied
    )
    unscaled_policy = _certificate(node, graph.graph)
    assert not unscaled_policy.threshold_gate_satisfied


def test_general_acute_sine_state_can_exceed_threshold_without_observed_history():
    graph = _contact()
    for node in graph:
        graph.nodes[node].update(theta=0.0 if node == 0 else 0.5, nu_f=1.0)
    fields = _fields(graph)
    threshold = Q.from_float(0.1)
    assert fields.capacity[0] > pi_interval().hi * threshold
    assert fields.form_rates[0].lo > threshold
    assert all(
        abs(fields.phase[j] - fields.phase[i]) < pi_interval().lo / 2
        for i, j in fields.edges
    )
    # All phases lie in one short global lift: no winding is imposed here.
    assert max(fields.phase) - min(fields.phase) == Q(1, 2)
    certificate = certify_mutation_trigger(
        current_epi=float(fields.epi[0]),
        nu_f=float(fields.capacity[0]),
        delta_nfr=float(fields.pressure[0].midpoint),
        xi=float(threshold),
    )
    assert certificate.predicted_crossed
    assert certificate.reason == "missing_history"
    assert certificate.observed_crossed is None
    assert not certificate.threshold_gate_satisfied
    rescaled = deepcopy(graph)
    for node in rescaled:
        rescaled.nodes[node]["nu_f"] /= 4
    slower = _fields(rescaled)
    assert slower.pressure == fields.pressure
    assert slower.form_gradient == fields.form_gradient
    assert slower.form_rates == tuple(rate / 4 for rate in fields.form_rates)
    assert slower.phase_rate_numerators() == tuple(
        rate / 4 for rate in fields.phase_rate_numerators()
    )
    for original, scaled in zip(fields.phase_rates, slower.phase_rates, strict=True):
        expected = original / 4
        assert scaled.lo <= expected.hi and expected.lo <= scaled.hi
        assert abs(scaled.midpoint - expected.midpoint) < Q(1, 10**35)


def test_equal_time_event_snapshots_are_not_an_observed_temporal_rate():
    graph = _reader_graph()
    node = graph.nodes["receiver"]
    node["epi_time_history"] = [(12.0, -0.5), (12.0, -0.25)]
    node["epi_history"] = [-0.5, -0.25]
    certificate = _certificate(node, graph.graph)
    assert certificate.reason == "non_increasing_time"
    assert certificate.observed_depi_dt is None
    assert certificate.observed_crossed is None
    with pytest.raises(OperatorPreconditionError, match="non_increasing_time"):
        validate_mutation_runtime_gate(node, graph.graph)


def test_stale_form_endpoint_does_not_fall_back_to_operator_step_history():
    graph = _reader_graph()
    node = graph.nodes["receiver"]
    node["epi_time_history"][-1] = (12.0, -0.375)
    node["epi_history"] = [-0.5, -0.25]
    certificate = _certificate(node, graph.graph)
    assert certificate.reason == "stale_physical_endpoint"
    assert not certificate.current_endpoint_matches_state
    assert certificate.observed_crossed is None
    assert certificate.source == "epi_time_history"
    with pytest.raises(OperatorPreconditionError, match="stale_physical_endpoint"):
        validate_mutation_runtime_gate(node, graph.graph)


def test_detached_time_records_retain_origin_without_authenticating_a_live_clock():
    graph = _reader_graph()
    original = _certificate(graph.nodes["receiver"], graph.graph)
    shifted_node = deepcopy(graph.nodes["receiver"])
    shifted_node["epi_time_history"] = [
        (time + 64, epi) for time, epi in shifted_node["epi_time_history"]
    ]
    shifted = _certificate(shifted_node, graph.graph)
    assert shifted.evidence.current_time - original.evidence.current_time == 64.0
    assert shifted.evidence.previous_time - original.evidence.previous_time == 64.0
    assert shifted.observed_depi_dt == original.observed_depi_dt
    assert (
        shifted.current_endpoint_matches_state
        == original.current_endpoint_matches_state
    )
    assert shifted.threshold_gate_satisfied == original.threshold_gate_satisfied


def test_observed_finite_secant_can_disagree_with_an_instantaneous_rate():
    graph = _reader_graph()
    node = graph.nodes["receiver"]
    node["delta_nfr"] = -0.5
    gate = validate_mutation_runtime_gate(node, graph.graph)
    certificate = gate.threshold.certificate
    assert Q(certificate.observed_depi_dt) == Q(1, 8)
    assert Q(certificate.predicted_depi_dt) == Q(-1, 4)
    assert Q(certificate.rate_gap) == Q(3, 8)
    assert certificate.threshold_gate_satisfied
    assert not certificate.predicted_crossed
