"""Synthetic policy and OZ/refresh controls, not retained-source acquisition.

The two-node state and optional quadratic samples are manufactured test inputs.
No retained graph, interval midpoint, trajectory, THOL proposal or child birth
is evaluated. Only the declared synthetic graph receives an actual OZ stage.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import cos, inf, nan, nextafter, pi, sin

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.constants.canonical import DELTA_PHI_MAX
from tnfr.dynamics.propagation import (
    DissonancePropagationPolicy,
    propagate_dissonance,
    resolve_dissonance_propagation_policy,
)
from tnfr.errors import TNFRValueError
from tnfr.operators._oz_config import OZStrictPolicy, resolve_oz_strict_policy
from tnfr.operators.definitions import Dissonance, SelfOrganization, Silence
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.dissonance import validate_dissonance_strict
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.research.local_capacity_history import capture, refresh_pressure
from tnfr.research.maintained_parent_um_execution import (
    _pressure_preservation_signature,
)
from tnfr.types import Glyph


def _synthetic_graph(**settings):
    graph = nx.Graph(
        RANDOM_SEED=23,
        _t=12.0,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
    )
    graph.graph.update(settings)
    for node, epi, theta, vf in (
        ("parent", 1 / 2, 1 / 8, 2.0),
        ("synthetic_child", 3 / 8, 0.0, 3.0),
    ):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: epi,
                ALIAS_THETA[0]: theta,
                ALIAS_VF[0]: vf,
                ALIAS_DNFR[0]: sin(1 / 8) / pi,
                "glyph_history": [],
                "epi_history": [epi],
            },
        )
    graph.add_edge("parent", "synthetic_child", weight=3 / 4)
    return graph


@pytest.fixture(autouse=True)
def no_retained_source_trajectory_or_thol(monkeypatch):
    from tnfr.dynamics import integrators
    from tnfr.research import maintained_sha_motion_source

    def forbidden(*args, **kwargs):
        pytest.fail("synthetic OZ controls may not acquire a trajectory or child")

    monkeypatch.setattr(maintained_sha_motion_source, "load_sha_successor", forbidden)
    monkeypatch.setattr(integrators, "rk4_coupled_step", forbidden)
    monkeypatch.setattr(SelfOrganization, "_prepare_execution", forbidden)


def test_policy_defaults_and_signed_override_are_pure_and_immutable():
    settings = {}
    policy = resolve_oz_strict_policy(settings)
    propagation = resolve_dissonance_propagation_policy(settings)
    assert policy == OZStrictPolicy(0.2, 0.8, 0.1, 1)
    assert propagation == DissonancePropagationPolicy(DELTA_PHI_MAX, 0.05)
    assert settings == {}
    with pytest.raises(FrozenInstanceError):
        policy.min_epi = 0.0
    with pytest.raises(FrozenInstanceError):
        propagation.min_propagation = 0.0

    settings = {
        "OZ_MIN_EPI": Q(-1, 4),
        "OZ_MAX_DNFR": Q(1, 2),
        "OZ_MIN_VF": 0,
        "OZ_MIN_DEGREE": 0,
        "OZ_PHASE_THRESHOLD": Q(3, 2),
        "OZ_MIN_PROPAGATION": 0,
    }
    before = deepcopy(settings)
    assert resolve_oz_strict_policy(settings) == OZStrictPolicy(-0.25, 0.5, 0.0, 0)
    assert resolve_dissonance_propagation_policy(settings) == (
        DissonancePropagationPolicy(1.5, 0.0)
    )
    assert settings == before


@pytest.mark.parametrize(
    "key", ("OZ_MIN_EPI", "OZ_MAX_DNFR", "OZ_MIN_VF", "OZ_MIN_DEGREE")
)
@pytest.mark.parametrize("value", (None, True, False, "0.1", nan, inf, -inf))
def test_invalid_strict_policies_reject_equally_in_resolver_and_live_gate(key, value):
    graph = _synthetic_graph(**{key: value})
    for read in (
        lambda: resolve_oz_strict_policy(graph.graph),
        lambda: validate_dissonance_strict(graph, "synthetic_child"),
    ):
        with pytest.raises(OperatorPreconditionError, match=key):
            read()
    assert "_oz_precondition_context" not in graph.nodes["synthetic_child"]


@pytest.mark.parametrize(
    "key,value",
    (
        ("OZ_MAX_DNFR", -0.1),
        ("OZ_MIN_VF", -0.1),
        ("OZ_MIN_DEGREE", -1),
        ("OZ_MIN_DEGREE", 1.0),
        ("OZ_MIN_DEGREE", Q(1)),
        ("OZ_MIN_EPI", Q(1, 10**400)),
    ),
)
def test_strict_policy_domains_preserve_nonnegative_integral_and_materialized_rules(
    key, value
):
    graph = _synthetic_graph(**{key: value})
    with pytest.raises(OperatorPreconditionError, match=key):
        resolve_oz_strict_policy(graph.graph)
    with pytest.raises(OperatorPreconditionError, match=key):
        validate_dissonance_strict(graph, "synthetic_child")


@pytest.mark.parametrize("settings", ({}, {"OZ_MIN_EPI": -0.25}))
def test_strict_boundaries_are_inclusive_and_degree_remains_a_warning(settings):
    graph = _synthetic_graph(**settings)
    policy = resolve_oz_strict_policy(graph.graph)
    child = graph.nodes["synthetic_child"]
    set_attr(child, ALIAS_EPI, policy.min_epi)
    set_attr(child, ALIAS_DNFR, -policy.max_dnfr)
    set_attr(child, ALIAS_VF, policy.min_vf)
    validate_dissonance_strict(graph, "synthetic_child")
    assert child["_oz_precondition_context"]["validation_passed"] is True
    for aliases, rejected in (
        (ALIAS_EPI, nextafter(policy.min_epi, -inf)),
        (ALIAS_DNFR, nextafter(-policy.max_dnfr, -inf)),
        (ALIAS_VF, nextafter(policy.min_vf, -inf)),
    ):
        previous = get_attr(child, aliases)
        set_attr(child, aliases, rejected)
        with pytest.raises(TNFRValueError, match="OZ precondition failed"):
            validate_dissonance_strict(graph, "synthetic_child")
        set_attr(child, aliases, previous)

    graph.graph["OZ_MIN_DEGREE"] = 2
    with pytest.warns(UserWarning, match="low connectivity"):
        validate_dissonance_strict(graph, "synthetic_child")


@pytest.mark.parametrize("key", ("OZ_PHASE_THRESHOLD", "OZ_MIN_PROPAGATION"))
@pytest.mark.parametrize("value", (None, True, False, "0.1", nan, inf, -inf, -0.1))
def test_invalid_propagation_policy_rejects_equally_without_neighbor_updates(
    key, value
):
    graph = _synthetic_graph(**{key: value})
    before = deepcopy(dict(graph.nodes))
    with pytest.raises(TNFRValueError):
        resolve_dissonance_propagation_policy(graph.graph)
    with pytest.raises(TNFRValueError):
        propagate_dissonance(graph, "synthetic_child", 0.1)
    assert dict(graph.nodes) == before


def test_zero_phase_policy_rejects_but_empty_neighborhood_keeps_runtime_abstention():
    graph = _synthetic_graph(OZ_PHASE_THRESHOLD=0.0)
    with pytest.raises(TNFRValueError, match="must be positive"):
        resolve_dissonance_propagation_policy(graph.graph)
    with pytest.raises(TNFRValueError, match="must be positive"):
        propagate_dissonance(graph, "synthetic_child", 0.1)
    graph.remove_edge("parent", "synthetic_child")
    assert propagate_dissonance(graph, "synthetic_child", 0.1) == set()


def _primary_state(graph):
    return (
        tuple(
            (
                node,
                *(
                    get_attr(graph.nodes[node], key)
                    for key in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF)
                ),
            )
            for node in graph
        ),
        tuple(
            (left, right, deepcopy(data))
            for left, right, data in graph.edges(data=True)
        ),
        graph.graph["_t"],
    )


def _pressures(graph):
    return {node: get_attr(graph.nodes[node], ALIAS_DNFR) for node in graph}


def _synthetic_storage(graph):
    parent, child = graph.nodes["parent"], graph.nodes["synthetic_child"]
    difference = get_attr(parent, ALIAS_EPI) - get_attr(child, ALIAS_EPI)
    angle = get_attr(parent, ALIAS_THETA) - get_attr(child, ALIAS_THETA)
    return 0.5 * graph.edges["parent", "synthetic_child"][
        "weight"
    ] * difference**2 + (1.0 - cos(angle))


def _run_synthetic_oz_refresh(graph):
    operators = (Dissonance(), SelfOrganization(), Silence())
    sequence = ValidatedSequence(
        operators,
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    raw = []

    def handoff(live):
        raw.append(
            (
                _pressures(live),
                _primary_state(live),
                _synthetic_storage(live),
                capture(live),
            )
        )
        protected = _pressure_preservation_signature(live, runtime=True)
        refresh_pressure(live)
        assert _pressure_preservation_signature(live, runtime=True) == protected

    stage = execute_network_operator_stage(
        graph,
        operators[0],
        ("synthetic_child",),
        sequence_context=sequence.step(0),
        compute_delta_nfr=handoff,
    )
    assert len(raw) == 1
    return stage, raw[0]


@pytest.mark.parametrize(
    "settings,propagates", (({}, False), ({"OZ_MIN_PROPAGATION": 0.0}, True))
)
def test_synthetic_oz_changes_raw_pressure_and_history_but_refresh_preserves_primary_law(
    settings, propagates
):
    graph = _synthetic_graph(**settings)
    refresh_pressure(graph)
    before = _primary_state(graph)
    pressure = _pressures(graph)
    storage = _synthetic_storage(graph)
    law = capture(graph)
    gain = resolve_runtime_operator_factors(None, Glyph.OZ, graph.graph)[
        "OZ_dnfr_factor"
    ]
    assert gain == (pi + 1) / pi

    stage, (raw_pressure, raw_primary, raw_storage, raw_law) = (
        _run_synthetic_oz_refresh(graph)
    )

    assert stage.schedule == "two_phase_jacobi" and stage.glyph == "OZ"
    assert stage.nodes_processed == 1
    assert raw_primary == before == _primary_state(graph)
    assert raw_storage - storage == 0.0
    assert raw_law.form_storage == law.form_storage
    assert raw_law.phase_storage == law.phase_storage
    assert raw_law.storage == law.storage
    assert _pressures(graph) == pressure
    local_after = gain * pressure["synthetic_child"]
    magnitude = local_after - pressure["synthetic_child"]
    assert raw_pressure["synthetic_child"] == local_after
    assert 0 < magnitude * 0.75 < 0.05
    expected_parent = pressure["parent"]
    if propagates:
        expected_parent += magnitude * 0.75 * (1 - 0.125 / DELTA_PHI_MAX)
    assert raw_pressure["parent"] == expected_parent
    events = graph.graph["_oz_propagation_events"]
    assert len(events) == 1
    assert events[0]["affected_nodes"] == (["parent"] if propagates else [])
    assert events[0]["affected_count"] == int(propagates)
    assert ("_oz_propagation" in graph.nodes["parent"]) is propagates
    child = graph.nodes["synthetic_child"]
    assert tuple(child["glyph_history"]) == ("OZ",)
    assert child["_operator_step"] == 1 and node_debt(child) == 1
    assert graph.nodes["parent"]["glyph_history"] == []
    assert child["epi_history"] == [3 / 8]
    assert "epi_time_history" not in child
    assert not observe_structural_acceleration(graph, "synthetic_child").available
    assert not any(key in child for key in ALIAS_D2EPI)
    assert "_bifurcation_ready" not in child


def test_synthetic_history_and_actual_oz_context_are_distinct_thol_gates():
    graph = _synthetic_graph()
    refresh_pressure(graph)
    blocked = validate_candidate(graph, "synthetic_child", "THOL")
    assert {item.rule for item in blocked.violations} == {"U4b"}
    _run_synthetic_oz_refresh(graph)
    assert validate_candidate(graph, "synthetic_child", "THOL").allowed
    with pytest.raises(OperatorPreconditionError, match="1 samples; 3 required"):
        validate_self_organization_strict(graph, "synthetic_child", emit_warnings=False)

    # Manufactured polynomial x(s)=3/8+4*s**2 tests the reader and live gates.
    # These are not acquired samples of the selected sine law or retained child.
    child = graph.nodes["synthetic_child"]
    times = (12.0, 12.125, 12.25)
    samples = tuple((time, 3 / 8 + 4 * (time - 12.0) ** 2) for time in times)
    child["epi_time_history"] = samples
    set_attr(child, ALIAS_EPI, samples[-1][1])
    graph.graph["_t"] = times[-1]
    observation = observe_structural_acceleration(graph, "synthetic_child")
    assert observation.available and observation.value == 8.0
    assert observation.source == "epi_time_history"
    assert observation.current_endpoint_matches_state is True
    validate_self_organization_strict(graph, "synthetic_child", emit_warnings=False)
    assert validate_candidate(graph, "synthetic_child", "THOL").allowed
    assert tuple(child["glyph_history"]) == ("OZ",)
    assert node_debt(child) == 1
    assert tuple(graph) == ("parent", "synthetic_child")
