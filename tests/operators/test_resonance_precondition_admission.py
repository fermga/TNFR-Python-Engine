"""RA policy admission must precede both direct and atomic-stage writes."""

from __future__ import annotations

import math
import pickle
from decimal import Decimal

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Resonance
from tnfr.operators.network_stage import execute_neighbor_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.resonance import (
    diagnose_resonance_readiness,
    validate_resonance_strict,
)
from tnfr.types import serialize_bepi


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 0.5,
                ALIAS_VF[0]: 1.0,
                ALIAS_THETA[0]: 0.0,
                ALIAS_DNFR[0]: 0.1,
                "glyph_history": ["AL", "UM"],
            }
        )
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    return graph


@pytest.mark.parametrize("staged", (False, True))
@pytest.mark.parametrize(
    "threshold",
    ("RA_MIN_SOURCE_EPI", "RA_MIN_VF", "RA_MAX_DISSONANCE", "RA_MAX_PHASE_DIFF"),
)
def test_nan_policy_rejected_before_public_execution_writes(threshold, staged):
    graph = _graph()
    graph.graph[threshold] = math.nan
    if threshold == "RA_MIN_SOURCE_EPI":
        # The old NaN comparison admitted this below-threshold source, and RA
        # changed its form through the compatible higher-amplitude neighbor.
        graph.nodes[0][ALIAS_EPI[0]] = 0.05
    before = pickle.dumps(graph, protocol=5)

    with pytest.raises(OperatorPreconditionError, match=threshold):
        if staged:
            execute_neighbor_stage(graph, Resonance(), (0,))
        else:
            Resonance()(graph, 0)

    assert pickle.dumps(graph, protocol=5) == before


@pytest.mark.parametrize("invalid", (math.inf, -0.01, True, "0.1", Decimal("1e-1000")))
def test_strict_and_readiness_share_raw_threshold_admission(invalid):
    graph = _graph()
    graph.graph["RA_MIN_SOURCE_EPI"] = invalid
    before = pickle.dumps(graph, protocol=5)
    for observe in (validate_resonance_strict, diagnose_resonance_readiness):
        with pytest.raises(OperatorPreconditionError, match="RA_MIN_SOURCE_EPI"):
            observe(graph, 0)
    assert pickle.dumps(graph, protocol=5) == before


@pytest.mark.parametrize(
    "kwargs, label",
    (
        ({"min_epi": math.nan}, "RA_MIN_SOURCE_EPI"),
        ({"max_dissonance": -1.0}, "RA_MAX_DISSONANCE"),
        ({"require_coupling": "false"}, "require_coupling"),
        ({"warn_phase_misalignment": 1}, "warn_phase_misalignment"),
    ),
)
def test_explicit_policy_arguments_obey_the_same_boundary(kwargs, label):
    with pytest.raises(OperatorPreconditionError, match=label):
        validate_resonance_strict(_graph(), 0, **kwargs)


@pytest.mark.parametrize(
    "aliases, invalid",
    (
        (ALIAS_VF, True),
        (ALIAS_VF, -0.1),
        (ALIAS_VF, Decimal("1e-1000")),
        (ALIAS_DNFR, "invalid"),
        (ALIAS_DNFR, math.nan),
        (ALIAS_THETA, True),
    ),
)
def test_invalid_consumed_state_cannot_be_reported_ready(aliases, invalid):
    graph = _graph()
    graph.nodes[0][aliases[0]] = invalid
    # A malformed authoritative alias must not fall through to a valid alias.
    if len(aliases) > 1:
        graph.nodes[0][aliases[1]] = 0.1
    before = pickle.dumps(graph, protocol=5)
    for observe in (validate_resonance_strict, diagnose_resonance_readiness):
        with pytest.raises(OperatorPreconditionError):
            observe(graph, 0)
    assert pickle.dumps(graph, protocol=5) == before


def test_signed_scalar_form_and_zero_capacity_policy_boundaries_are_retained():
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = serialize_bepi(-0.5)
    graph.nodes[0][ALIAS_VF[0]] = 0.0
    graph.nodes[0][ALIAS_DNFR[0]] = -0.25
    graph.graph.update(
        RA_MIN_SOURCE_EPI=0.5,
        RA_MIN_VF=0.0,
        RA_MAX_DISSONANCE=0.25,
        RA_MAX_PHASE_DIFF=0.0,
    )
    before = pickle.dumps(graph, protocol=5)

    validate_resonance_strict(graph, 0)
    report = diagnose_resonance_readiness(graph, 0)

    assert report["ready"] is True
    assert report["values"]["epi"] == 0.5
    assert report["values"]["vf"] == 0.0
    assert report["values"]["dnfr"] == 0.25
    assert all(value == "passed" for value in report["checks"].values())
    assert pickle.dumps(graph, protocol=5) == before


def test_below_threshold_still_fails_and_explicit_override_remains_available():
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = 0.05
    assert diagnose_resonance_readiness(graph, 0)["ready"] is False
    with pytest.raises(ValueError, match="coherent source"):
        validate_resonance_strict(graph, 0)
    validate_resonance_strict(graph, 0, min_epi=0.05)


def test_inactive_optional_policy_does_not_consume_its_thresholds():
    graph = _graph()
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=False, RA_MIN_SOURCE_EPI=math.nan
    )
    graph.nodes[0][ALIAS_EPI[0]] = 0.05

    Resonance()(graph, 0)

    assert get_attr(graph.nodes[0], ALIAS_EPI) > 0.05


@pytest.mark.parametrize("invalid", (True, math.nan, Decimal("1e-1000")))
def test_invalid_neighbor_phase_is_not_an_unavailable_success(invalid):
    graph = _graph()
    graph.nodes[1][ALIAS_THETA[0]] = invalid
    before = pickle.dumps(graph, protocol=5)
    for observe in (validate_resonance_strict, diagnose_resonance_readiness):
        with pytest.raises((TypeError, ValueError), match="phase"):
            observe(graph, 0)
    assert pickle.dumps(graph, protocol=5) == before


def test_zero_represented_resultant_makes_only_phase_warning_unavailable():
    graph = _graph()
    for node, phase in enumerate((0.0, 0.0, math.pi, -math.pi), start=1):
        graph.add_node(node, **dict(graph.nodes[0]))
        graph.nodes[node][ALIAS_THETA[0]] = phase
        graph.add_edge(0, node)
    before = pickle.dumps(graph, protocol=5)

    with pytest.warns(UserWarning, match="zero represented neighbor resultant"):
        validate_resonance_strict(graph, 0)
    report = diagnose_resonance_readiness(graph, 0)

    assert report["ready"] is True  # Other configured policy checks still pass.
    assert report["checks"]["phase_alignment"] == "unavailable"
    assert report["values"]["phase_diff"] is None
    assert any("zero represented" in item for item in report["recommendations"])
    assert pickle.dumps(graph, protocol=5) == before


@pytest.mark.parametrize("capacity", (0.0, 0.1))
def test_capacity_advice_respects_multiplicative_and_regime_limits(capacity):
    graph = _graph()
    graph.nodes[0][ALIAS_VF[0]] = capacity
    graph.graph["RA_MIN_VF"] = 0.2
    report = diagnose_resonance_readiness(graph, 0)
    advice = " ".join(report["recommendations"])

    assert report["ready"] is False
    assert "Supply admitted capacity" in advice
    assert "NAV" not in advice
    assert ("VAL (Expansion)" in advice) is (capacity > 0.0)
    if capacity == 0.0:
        assert "preserves zero capacity" in advice
    with pytest.raises(ValueError, match="Supply admitted capacity") as failure:
        validate_resonance_strict(graph, 0)
    assert "NAV" not in str(failure.value)
