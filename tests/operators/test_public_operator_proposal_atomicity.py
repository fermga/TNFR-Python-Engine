"""Public IL and NAV validate complete proposals before changing graph state."""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Coherence, Transition
from tnfr.operators.preconditions import OperatorPreconditionError


def _graph(*, history: list[str] | None = None) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(RANDOM_SEED=17, NAV_RANDOM=False)
    for index in graph:
        graph.nodes[index].update(
            {
                ALIAS_EPI[0]: 0.4 + 0.1 * index,
                ALIAS_VF[0]: 0.5,
                ALIAS_DNFR[0]: 0.4 - 0.1 * index,
                ALIAS_THETA[0]: 0.25 + index,
                "glyph_history": list(history or ["AL"]),
            }
        )
    return graph


def _snapshot(graph: nx.Graph) -> tuple[dict, dict]:
    return (
        deepcopy({node: dict(data) for node, data in graph.nodes(data=True)}),
        deepcopy(dict(graph.graph)),
    )


def _assert_unchanged(graph: nx.Graph, snapshot: tuple[dict, dict]) -> None:
    nodes_before, graph_before = snapshot
    assert {node: dict(data) for node, data in graph.nodes(data=True)} == nodes_before
    assert dict(graph.graph) == graph_before
    assert "_node_cache" not in graph.graph


@pytest.mark.parametrize(
    "value",
    [True, "0.3", -0.01, 1.01, float("nan"), float("inf")],
)
def test_il_rejects_invalid_phase_lock_strength_atomically(value) -> None:
    graph = _graph()
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="phase_locking_coefficient"):
        Coherence()(graph, 0, phase_locking_coefficient=value)

    _assert_unchanged(graph, before)


@pytest.mark.parametrize("radius", [True, -1, 1.5, "1"])
def test_il_rejects_invalid_coherence_radius_atomically(radius) -> None:
    graph = _graph()
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="coherence_radius"):
        Coherence()(graph, 0, coherence_radius=radius)

    _assert_unchanged(graph, before)


def test_il_rejects_nonfinite_neighbor_phase_before_handler_or_cache() -> None:
    graph = _graph()
    graph.nodes[1][ALIAS_THETA[0]] = float("nan")
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="neighbor theta"):
        Coherence()(graph, 0)

    _assert_unchanged(graph, before)


def test_il_rejects_malformed_precondition_config_before_warning_or_state() -> None:
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    graph.graph["IL_PRECONDITIONS"] = {"warn_isolated": 1}
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="warn_isolated"):
        Coherence()(graph, 0)

    _assert_unchanged(graph, before)


def test_il_rejects_malformed_telemetry_sink_before_handler() -> None:
    graph = _graph()
    graph.graph["IL_dnfr_reductions"] = {}
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="IL_dnfr_reductions"):
        Coherence()(graph, 0)

    _assert_unchanged(graph, before)


def test_il_endpoint_lock_normalizes_the_committed_phase() -> None:
    graph = _graph()
    graph.nodes[0][ALIAS_THETA[0]] = 4.0 * math.tau + 0.25
    graph.nodes[1][ALIAS_THETA[0]] = -3.0 * math.tau + 1.25

    Coherence()(graph, 0, phase_locking_coefficient=1.0)

    theta = graph.nodes[0][ALIAS_THETA[0]]
    assert math.isfinite(theta)
    assert 0.0 <= theta < math.tau
    assert theta == pytest.approx(1.25)
    assert graph.graph["IL_phase_locking"][-1]["theta_after"] == theta


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"vf_factor": True}, "vf_factor"),
        ({"vf_factor": -0.01}, "vf_factor"),
        ({"vf_factor": float("nan")}, "vf_factor"),
        ({"phase_shift": "0.2"}, "phase_shift"),
        ({"phase_shift": float("inf")}, "phase_shift"),
        ({"validate_preconditions": 1}, "validate_preconditions"),
    ],
)
def test_nav_rejects_invalid_public_arguments_atomically(kwargs, match) -> None:
    graph = _graph(history=["IL"])
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match=match):
        Transition()(graph, 0, **kwargs)

    _assert_unchanged(graph, before)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("NAV_STRICT", 1),
        ("NAV_RANDOM", "false"),
        ("NAV_MIN_VF", float("nan")),
        ("NAV_MAX_DNFR", -1.0),
        ("NAV_MIN_EPI_FROM_LATENCY", float("inf")),
        ("NAV_STRICT_SEQUENCE_CHECK", 1),
    ],
)
def test_nav_rejects_invalid_active_configuration_atomically(key, value) -> None:
    graph = _graph(history=["IL"])
    graph.graph[key] = value
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match=key):
        Transition()(graph, 0)

    _assert_unchanged(graph, before)


@pytest.mark.parametrize(
    ("timestamp", "match"),
    [
        ("not-a-time", "latency_start_time"),
        (datetime.now().isoformat(), "UTC offset"),
        (
            (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            "silence duration",
        ),
    ],
)
def test_nav_rejects_invalid_latency_time_without_clearing_state(
    timestamp, match
) -> None:
    graph = _graph(history=["SHA"])
    graph.nodes[0].update(
        latent=True,
        latency_start_time=timestamp,
        preserved_epi=graph.nodes[0][ALIAS_EPI[0]],
    )
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match=match):
        Transition()(graph, 0)

    _assert_unchanged(graph, before)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("preserved_epi", float("nan"), "preserved_epi"),
        ("MAX_SILENCE_DURATION", float("inf"), "MAX_SILENCE_DURATION"),
    ],
)
def test_nav_rejects_invalid_latency_scalars_without_clearing_state(
    key, value, match
) -> None:
    graph = _graph(history=["SHA"])
    graph.nodes[0].update(
        latent=True,
        latency_start_time=(
            datetime.now(timezone.utc) - timedelta(seconds=5)
        ).isoformat(),
        preserved_epi=graph.nodes[0][ALIAS_EPI[0]],
    )
    if key == "MAX_SILENCE_DURATION":
        graph.graph[key] = value
    else:
        graph.nodes[0][key] = value
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match=match):
        Transition()(graph, 0)

    _assert_unchanged(graph, before)


def test_nav_rejects_malformed_transition_sink_before_handler() -> None:
    graph = _graph(history=["IL"])
    graph.graph["_nav_transitions"] = ()
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="_nav_transitions"):
        Transition()(graph, 0)

    _assert_unchanged(graph, before)


def test_nav_handler_noop_is_rejected_before_latency_history_or_metadata() -> None:
    graph = _graph(history=["SHA"])
    graph.graph.update(
        NAV_STRICT=True,
        NAV_RANDOM=False,
        GLYPH_FACTORS={"NAV_jitter": 0.0},
    )
    graph.nodes[0].update(
        {
            ALIAS_DNFR[0]: graph.nodes[0][ALIAS_VF[0]],
            "latent": True,
            "latency_start_time": (
                datetime.now(timezone.utc) - timedelta(seconds=5)
            ).isoformat(),
            "preserved_epi": graph.nodes[0][ALIAS_EPI[0]],
        }
    )
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="must change DeltaNFR"):
        Transition()(graph, 0)

    _assert_unchanged(graph, before)


def test_nav_complete_noop_is_rejected_before_handler_history_or_metadata() -> None:
    graph = _graph(history=["IL"])
    graph.graph.update(
        NAV_STRICT=True,
        NAV_RANDOM=False,
        GLYPH_FACTORS={"NAV_jitter": 0.0},
    )
    graph.nodes[0].update(
        {
            ALIAS_VF[0]: 0.5,
            ALIAS_DNFR[0]: 0.4,
            ALIAS_THETA[0]: 0.25,
        }
    )
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="at least one nodal channel"):
        Transition()(graph, 0, vf_factor=1.0, phase_shift=0.0)

    _assert_unchanged(graph, before)


def test_nav_rejects_overflowing_proposal_before_handler() -> None:
    graph = _graph(history=["IL"])
    graph.graph.update(
        NAV_STRICT=True,
        NAV_RANDOM=False,
        GLYPH_FACTORS={"NAV_jitter": 1.0e308},
    )
    graph.nodes[0].update(
        {
            ALIAS_VF[0]: 1.0e308,
            ALIAS_DNFR[0]: 0.4,
            ALIAS_EPI[0]: 0.4,
        }
    )
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="handler proposal"):
        Transition()(graph, 0, validate_preconditions=False)

    _assert_unchanged(graph, before)


def test_nav_success_commits_finite_normalized_complete_transition() -> None:
    graph = _graph(history=["IL"])
    graph.graph.update(
        NAV_STRICT=True,
        NAV_RANDOM=False,
        GLYPH_FACTORS={"NAV_jitter": 0.1},
    )
    graph.nodes[0][ALIAS_THETA[0]] = 3.0 * math.tau + 0.25

    Transition()(graph, 0, vf_factor=1.5, phase_shift=2.0 * math.tau + 0.5)

    assert graph.nodes[0][ALIAS_VF[0]] == pytest.approx(0.75)
    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.48)
    theta = graph.nodes[0][ALIAS_THETA[0]]
    assert math.isfinite(theta)
    assert 0.0 <= theta < math.tau
    assert theta == pytest.approx(0.75)
    assert graph.nodes[0]["glyph_history"][-1] == "NAV"
    assert graph.nodes[0]["_regime_before"] == "active"
    assert graph.graph["_nav_transitions"][-1]["phase_shift"] == pytest.approx(0.5)
