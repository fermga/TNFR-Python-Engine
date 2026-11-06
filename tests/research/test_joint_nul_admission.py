"""Constructed common NUL eligibility, with no event or successor execution.

The strict scalar policy, incremental grammar and actual read-only stage
preflight have different scopes. None certifies the proposed numerical image,
all later wrapper effects, a complete word or an occurrence law.
"""

from copy import deepcopy
from itertools import product

import networkx as nx
import pytest

from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.operators import network_stage
from tnfr.operators.definitions import Contraction
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.preconditions import OperatorPreconditionError, validate_contraction


def _graph():
    """Two-node witness before its supplied single-parent NUL request."""
    graph = nx.Graph(
        RANDOM_SEED=17,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_FACTORS={"NUL_scale": 0.5},
        EDGE_AWARE_ENABLED=True,
        CLIP_MODE="hard",
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
        NUL_MIN_VF=0.1,
        NUL_MIN_EPI=0.1,
        NUL_MAX_DENSITY=2.0,
    )
    graph.add_node(
        0,
        EPI=2.0,
        nu_f=2.0,
        theta=0.0,
        delta_nfr=0.0,
        glyph_history=["AL", "IL"],
        _grammar_u2_debt=0,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )
    graph.add_node(1, EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.add_edge(0, 1, weight=1.0)
    return graph


def _preflight(graph):
    """Use the runtime owner up to admission; never propose or commit a reset."""
    before = deepcopy(graph)
    try:
        return network_stage._preflight_two_phase_stage(
            graph,
            Contraction(),
            (0,),
            sequence_context=None,
            execution_kwargs={},
        )
    finally:
        assert graph.graph == before.graph
        assert dict(graph.nodes) == dict(before.nodes)
        assert dict(graph.edges) == dict(before.edges)
        assert not hasattr(graph, "_last_operator_applied")


@pytest.fixture(autouse=True)
def no_event_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("common-admission controls must not execute an event")

    monkeypatch.setattr(network_stage, "execute_pointwise_stage", forbidden)
    monkeypatch.setattr(network_stage, "execute_operator_major_stage", forbidden)
    monkeypatch.setattr(network_stage, "_propose_pointwise", forbidden)


def test_joint_witness_box_keeps_common_read_only_stage_admission():
    graph = _graph()
    width = 1 / 65536
    centered_two = (2.0 - width, 2.0, 2.0 + width)
    centered_zero = (-width, 0.0, width)
    for capacity, form, pressure, phase in product(
        centered_two, centered_two, centered_zero, centered_zero
    ):
        graph.nodes[0].update(nu_f=capacity, EPI=form, delta_nfr=pressure, theta=phase)
        validate_contraction(graph, 0)
        assert validate_candidate(graph, 0, "NUL").allowed
        admission = _preflight(graph)
        assert admission.exact_stage
        assert dict(admission.snapshot.nodes) == dict(graph.nodes)
        assert admission.snapshot is not graph


def test_nominal_capacity_admission_cannot_hide_a_closed_box_threshold():
    graph = _graph()
    graph.graph["NUL_MIN_VF"] = 1.0
    for capacity in (1.5, 2.0):
        graph.nodes[0]["nu_f"] = capacity
        assert _preflight(graph).exact_stage

    graph.nodes[0]["nu_f"] = 1.0
    assert validate_candidate(graph, 0, "NUL").allowed
    with pytest.raises(OperatorPreconditionError, match="frequency at minimum"):
        _preflight(graph)
    # Strict scalar eligibility is optional in the actual wrapper. Its
    # disabled branch must not be silently treated as the stricter theorem.
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
    assert _preflight(graph).exact_stage


def test_joint_density_corner_uses_minimum_form_and_maximum_pressure_magnitude():
    graph = _graph()
    # The signed pressure with greatest magnitude is the lower endpoint.
    # Its joint corner with the minimum signed form has exact density 2.
    for capacity, form, pressure in product(
        (0.5, 1.0, 1.5), (0.25, 0.5, 0.75), (-0.5, 0.125, 0.25)
    ):
        graph.nodes[0].update(nu_f=capacity, EPI=form, delta_nfr=pressure)
        validate_contraction(graph, 0)
    graph.graph["NUL_MAX_DENSITY"] = 1.5
    graph.nodes[0].update(nu_f=1.0, EPI=0.5, delta_nfr=0.125)
    assert _preflight(graph).exact_stage
    graph.nodes[0].update(nu_f=0.5, EPI=0.25, delta_nfr=-0.5)
    assert validate_candidate(graph, 0, "NUL").allowed
    with pytest.raises(OperatorPreconditionError, match="density proxy"):
        _preflight(graph)


def test_identical_physical_state_keeps_inherited_u2_admission_obstruction():
    graph = _graph()
    assert _preflight(graph).exact_stage
    physical = tuple(
        graph.nodes[0][key] for key in ("EPI", "nu_f", "theta", "delta_nfr")
    )
    graph.nodes[0]["_grammar_u2_debt"] = U2_DEBT_CAPACITY + 1
    validate_contraction(graph, 0)
    candidate = validate_candidate(graph, 0, "NUL")
    assert not candidate.allowed
    assert {item.rule for item in candidate.violations} == {"U2"}
    with pytest.raises(StructuralGrammarError) as rejected:
        _preflight(graph)
    assert (rejected.value.rule, rejected.value.candidate) == ("U2", "NUL")
    assert (
        tuple(graph.nodes[0][key] for key in ("EPI", "nu_f", "theta", "delta_nfr"))
        == physical
    )


def test_zero_form_with_empty_history_mixes_grammar_inside_scalar_eligible_box():
    graph = _graph()
    graph.graph["NUL_MIN_EPI"] = -0.125
    graph.nodes[0].update(glyph_history=[], _operator_step=0)
    for form in (-0.125, 0.125):
        graph.nodes[0]["EPI"] = form
        validate_contraction(graph, 0)
        assert _preflight(graph).exact_stage

    graph.nodes[0]["EPI"] = 0.0
    validate_contraction(graph, 0)
    candidate = validate_candidate(graph, 0, "NUL")
    assert not candidate.allowed
    assert {item.rule for item in candidate.violations} == {"U1a"}
    with pytest.raises(StructuralGrammarError) as rejected:
        _preflight(graph)
    assert (rejected.value.rule, rejected.value.candidate) == ("U1a", "NUL")
    graph.nodes[0]["glyph_history"] = ["AL"]
    assert _preflight(graph).exact_stage


def test_stage_preflight_requires_bookkeeping_not_consumed_by_scalar_or_candidate():
    graph = _graph()
    graph.nodes[0]["_operator_step"] = True
    validate_contraction(graph, 0)
    assert validate_candidate(graph, 0, "NUL").allowed
    with pytest.raises(TypeError, match="_operator_step must be a nonnegative integer"):
        _preflight(graph)
