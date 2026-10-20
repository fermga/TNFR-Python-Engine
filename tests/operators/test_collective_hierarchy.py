"""Synthetic THOL integration controls; no retained response or flow is run.

The three-sample histories are manufactured reader inputs. Their provenance is
not the selected sine law, and passing an event gate does not establish birth
from that law. Algebraic controls independently evaluate its complete rows.
"""

from copy import deepcopy
from fractions import Fraction as Q
from math import cos, fsum, pi, sin, sqrt

import networkx as nx
import pytest

from tests.operators.test_collective_reception import (
    PERMUTATION,
    PORTS,
    _contact,
    _fields,
    _refresh,
)
from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.constants.canonical import COUPLING_GENTLE, COUPLING_MODERATE
from tnfr.mathematics import BEPIElement
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import Dissonance, SelfOrganization, Silence
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.network_stage import execute_self_organization_stage
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics import assess_u5_parent_child_coherence
from tnfr.physics.relational_sine_symmetry import assess_sine_support_involution


def _preparation(accelerations=(0.25, -0.25)):
    """Independent contact with declared synthetic temporal records/context."""
    graph = _contact()
    graph.graph["_t"] = 3.0
    for node, acceleration in zip(PORTS, accelerations, strict=True):
        form = graph.nodes[node]["EPI"]
        # Samples of an explicitly supplied quadratic, not a law trajectory.
        graph.nodes[node]["epi_time_history"] = [
            (0.0, form + 4.5 * acceleration),
            (1.0, form + 2.0 * acceleration),
            (3.0, form),
        ]
        graph.nodes[node][ALIAS_D2EPI[0]] = 999.0
    return graph


def _physical(graph):
    return (
        tuple(graph),
        tuple(graph.edges(data="weight")),
        tuple(
            (data["EPI"], data["nu_f"], data["theta"])
            for _, data in graph.nodes(data=True)
        ),
    )


def _state(graph):
    return deepcopy(
        (dict(graph.nodes(data=True)), tuple(graph.edges(data=True)), graph.graph)
    )


def _rows_and_acceleration(graph):
    """Independent edge differentiation, without the production pressure map."""
    mobility = {i: graph.nodes[i]["nu_f"] / (pi * graph.degree(i)) for i in graph}
    gradient = {
        i: fsum(
            graph.edges[i, j]["weight"]
            * (graph.nodes[i]["EPI"] - graph.nodes[j]["EPI"])
            for j in graph[i]
        )
        for i in graph
    }
    phase_rate = {i: mobility[i] * gradient[i] for i in graph}
    form_rate = {
        i: mobility[i]
        * fsum(sin(graph.nodes[j]["theta"] - graph.nodes[i]["theta"]) for j in graph[i])
        for i in graph
    }
    acceleration = {
        i: mobility[i]
        * fsum(
            cos(graph.nodes[j]["theta"] - graph.nodes[i]["theta"])
            * (phase_rate[j] - phase_rate[i])
            for j in graph[i]
        )
        for i in graph
    }
    return form_rate, phase_rate, acceleration


def test_selected_law_acceleration_has_independent_form_contrast_domain():
    graph = nx.Graph()
    graph.add_node(0, EPI=3.0, nu_f=1.0, theta=0.0)
    graph.add_node(1, EPI=1.0, nu_f=1.0, theta=pi / 3)
    graph.add_edge(0, 1, weight=1.0)
    form_rate, phase_rate, acceleration = _rows_and_acceleration(graph)
    source = _fields(graph)
    assert float(source.form_rates[0].midpoint) == pytest.approx(sqrt(3) / (2 * pi))
    assert float(source.phase_rates[0].midpoint) == pytest.approx(2 / pi)
    assert form_rate[0] == pytest.approx(float(source.form_rates[0].midpoint))
    assert phase_rate[0] == pytest.approx(float(source.phase_rates[0].midpoint))
    assert acceleration[0] == pytest.approx(-2 / pi**2)
    assert acceleration[1] == pytest.approx(2 / pi**2)
    assert abs(acceleration[0]) > 0.1
    assert not observe_structural_acceleration(graph, 0).available
    # A constitutive acceleration, even above tau, does not create a record.
    assert "epi_time_history" not in graph.nodes[0]


def test_full_word_and_no_child_event_have_separate_physical_budgets():
    graph = _preparation((1 / 16, 1 / 16))
    operators = (Dissonance(), SelfOrganization(), Silence())
    sequence = ValidatedSequence(
        operators,
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    for index, operator in enumerate(operators):
        assert all(
            validate_candidate(graph, node, operator.glyph.value).allowed
            for node in PORTS
        )
        if index == 1:
            before = _physical(graph)
            source = _fields(graph)
        stage = execute_network_operator_stage(
            graph,
            operator,
            PORTS,
            sequence_context=sequence.step(index),
            compute_delta_nfr=_refresh,
        )
        assert stage.schedule == "two_phase_jacobi"
        if index == 1:
            assert _physical(graph) == before
            after = _fields(graph)
            assert after == source
            assert not any("sub_nodes" in graph.nodes[node] for node in PORTS)
            assert all(
                get_attr(graph.nodes[node], ALIAS_D2EPI) == 1 / 16 for node in PORTS
            )
    assert all(
        tuple(graph.nodes[node]["glyph_history"])[-3:] == ("OZ", "THOL", "SHA")
        for node in PORTS
    )


def test_pressure_only_stage_changes_stored_pressure_but_no_fresh_sine_field():
    graph = _preparation()
    graph.graph["THOL_MAX_BIFURCATION_DEPTH"] = 0
    before = _physical(graph)
    source = _fields(graph)
    stored = [graph.nodes[node]["delta_nfr"] for node in PORTS]
    execute_self_organization_stage(graph, SelfOrganization(), PORTS)
    assert _physical(graph) == before
    assert _fields(graph) == source
    for node, pressure, acceleration in zip(PORTS, stored, (0.25, -0.25), strict=True):
        assert graph.nodes[node]["delta_nfr"] == pytest.approx(
            pressure + acceleration / (4 * pi)
        )
        assert get_attr(graph.nodes[node], ALIAS_D2EPI) == acceleration
    assert assess_sine_support_involution(
        _fields(graph), permutation_indices=PERMUTATION
    ).source_trajectory_reduction_certified
    _refresh(graph)
    assert [graph.nodes[node]["delta_nfr"] for node in PORTS] == stored


def test_primitive_and_public_event_agree_on_pressure_but_not_birth():
    primitive, public = _preparation(), _preparation()
    for node in PORTS:
        apply_glyph(primitive, node, "THOL")
    execute_self_organization_stage(public, SelfOrganization(), PORTS)
    assert len(primitive) == 20
    assert len(public) == 22
    assert primitive.number_of_edges() == public.number_of_edges() == 45
    for node in PORTS:
        assert primitive.nodes[node]["delta_nfr"] == public.nodes[node]["delta_nfr"]
        assert get_attr(primitive.nodes[node], ALIAS_D2EPI) == get_attr(
            public.nodes[node], ALIAS_D2EPI
        )
        assert "sub_nodes" not in primitive.nodes[node]


@pytest.fixture(scope="module")
def birth():
    graph = _preparation()
    before = deepcopy(graph)
    stage = execute_self_organization_stage(graph, SelfOrganization(), PORTS)
    return before, graph, stage


def test_public_birth_keeps_parent_component_and_creates_isolated_typed_children(birth):
    before, graph, stage = birth
    assert stage.schedule == "two_phase_jacobi"
    assert tuple(graph) == tuple(before) + ("0_sub_0", "1_sub_0")
    assert _physical(graph.subgraph(tuple(before))) == _physical(before)
    assert _fields(graph.subgraph(tuple(before))) == _fields(before)
    assert graph.graph["hierarchy"] == {0: ["0_sub_0"], 1: ["1_sub_0"]}
    for node in PORTS:
        child = graph.nodes[node]["sub_nodes"][0]
        data = graph.nodes[child]
        assert graph.degree(child) == 0
        assert data["parent_node"] == node
        assert data["_hierarchy_path"] == [node]
        assert data["hierarchy_level"] == data["_bifurcation_level"] == 1
        assert get_attr(data, ALIAS_VF) == before.nodes[node]["nu_f"] * 0.95
        assert get_attr(data, ALIAS_THETA) == before.nodes[node]["theta"] % (2 * pi)
        assert get_attr(data, ALIAS_DNFR) == 0.0
        assert data["epi_history"] == [data["EPI"]]
        assert not observe_structural_acceleration(graph, child).available
    with pytest.raises(ValueError, match="connected|isolated"):
        _fields(graph)


def test_metabolic_child_uses_bare_multiplicity_not_conductance_or_parent_mass(birth):
    before, graph, _ = birth
    for node in PORTS:
        neighbors = tuple(before[node])
        mean_form = fsum(before.nodes[j]["EPI"] for j in neighbors) / len(neighbors)
        real = fsum(cos(before.nodes[j]["theta"]) for j in neighbors) / len(neighbors)
        imag = fsum(sin(before.nodes[j]["theta"]) for j in neighbors) / len(neighbors)
        variance = 1 - sqrt(real**2 + imag**2)
        record = graph.nodes[node]["sub_epis"][0]
        expected = max(
            0,
            min(
                1,
                0.3 * before.nodes[node]["EPI"]
                + COUPLING_MODERATE * (mean_form - before.nodes[node]["EPI"])
                + COUPLING_GENTLE * variance,
            ),
        )
        assert record["epi"] == pytest.approx(expected)
        assert record["network_signals"]["neighbor_count"] == 7
        assert record["network_signals"]["mean_neighbor_epi"] == pytest.approx(
            mean_form
        )
        assert graph.nodes[node]["EPI"] == before.nodes[node]["EPI"]
    reweighted = _preparation()
    for left, right in reweighted.edges:
        reweighted.edges[left, right]["weight"] *= 0.5
    execute_self_organization_stage(reweighted, SelfOrganization(), PORTS)
    assert [reweighted.nodes[f"{i}_sub_0"]["EPI"] for i in PORTS] == [
        graph.nodes[f"{i}_sub_0"]["EPI"] for i in PORTS
    ]


def test_numeric_u5_target_remains_independent_of_valid_hierarchy(birth):
    _, graph, _ = birth
    for node in PORTS:
        # Birth supplies no measured child change rate. Missing input must not
        # become a passing numerical U5 assessment by an invented zero.
        with pytest.raises(ValueError, match="dEPI|change.rate"):
            assess_u5_parent_child_coherence(graph, node, alpha=0.25)
        supplied = deepcopy(graph)
        for member in (node, *supplied.nodes[node]["sub_nodes"]):
            supplied.nodes[member][ALIAS_DEPI[0]] = 0.0
        # These zeros are explicit synthetic observer inputs, not flow data.
        passing = assess_u5_parent_child_coherence(supplied, node, alpha=0.25)
        failing = assess_u5_parent_child_coherence(supplied, node, alpha=2.0)
        assert passing.satisfies_target
        assert not failing.satisfies_target
        assert passing.child_coherences == (1.0,)
        assert passing.parent_coherence < 1.0


def test_shared_law_refresh_rejection_rolls_back_child_and_context():
    graph = _preparation()
    before = _state(graph)
    with pytest.raises(ValueError, match="connected|isolated"):
        execute_self_organization_stage(
            graph,
            SelfOrganization(),
            PORTS,
            compute_delta_nfr=_refresh,
        )
    assert _state(graph) == before


def test_collective_mean_history_loses_both_parent_acceleration_magnitudes():
    graph = _preparation()
    mean_samples = tuple(
        sum(Q(graph.nodes[node]["epi_time_history"][sample][1]) for node in PORTS)
        / len(PORTS)
        for sample in range(3)
    )
    assert len(set(mean_samples)) == 1
    observed = tuple(
        observe_structural_acceleration(graph, node).value for node in PORTS
    )
    assert observed == (0.25, -0.25)
    assert all(abs(value) > 0.1 for value in observed)


def test_invalid_last_target_history_rejects_all_targets_atomically():
    graph = _preparation()
    graph.nodes[1]["epi_time_history"][-1] = (3.0, -123.0)
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="final EPI"):
        execute_self_organization_stage(graph, SelfOrganization(), PORTS)
    assert _state(graph) == before


@pytest.mark.parametrize("kind", ("tiny", "nonuniform", "complex"))
def test_invalid_history_embedding_rejects_whole_public_stage_before_write(kind):
    graph = _preparation()
    if kind == "tiny":
        payload = Q(1, 10**400)
        message = "underflows"
    else:
        values = (0.25, 0.5) if kind == "nonuniform" else (0.25j, 0.25j)
        payload = BEPIElement(values, values, (0.0, 1.0))
        message = "uniform-real"
    graph.nodes[1]["epi_time_history"][0] = (0.0, payload)
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match=message):
        execute_self_organization_stage(graph, SelfOrganization(), PORTS)
    assert _state(graph) == before


def test_missing_temporal_evidence_and_recent_context_are_independent():
    graph = _contact()
    assert validate_candidate(graph, 0, "THOL").allowed
    assert not observe_structural_acceleration(graph, 0).available
    with pytest.raises(OperatorPreconditionError, match="history"):
        validate_self_organization_strict(graph, 0)
    graph = _preparation()
    graph.nodes[0]["glyph_history"] = ["IL", "SHA", "NAV"]
    validate_self_organization_strict(graph, 0)
    candidate = validate_candidate(graph, 0, "THOL")
    assert not candidate.allowed
    assert {violation.rule for violation in candidate.violations} == {"U4b"}


def test_affine_clock_transforms_acceleration_threshold_and_pressure_gain():
    graph = _preparation()
    scaled = deepcopy(graph)
    scale = 4.0
    for node in PORTS:
        scaled.nodes[node]["epi_time_history"] = [
            (scale * time + 10.0, form)
            for time, form in graph.nodes[node]["epi_time_history"]
        ]
        original = observe_structural_acceleration(graph, node).value
        transformed = observe_structural_acceleration(scaled, node).value
        assert transformed == original / scale**2
        assert (abs(original) > 0.1) == (abs(transformed) > 0.1 / scale**2)
        gain = 1 / (4 * pi)
        # With nu_s=nu_t/scale and p unchanged, the event gain scales c^2.
        assert (scale**2 * gain) * transformed == gain * original
