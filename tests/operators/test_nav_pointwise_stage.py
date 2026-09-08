"""Executable contracts for the immutable all-target NAV stage."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import networkx as nx
import pytest

import tnfr.operators.network_stage as network_stage_module
import tnfr.operators.transition as transition_module
from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.operators.definitions import Transition
from tnfr.operators.network_stage import (
    POINTWISE_TWO_PHASE_GLYPHS,
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_operator_major_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.stage_contracts import (
    MergeLaw,
    MergeLawStatus,
    RollbackScope,
    StageResource,
    StageSchedule,
    StructuralOverlap,
    remaining_stage_contract_gaps,
    stage_contract_for,
    stage_schedule_metadata,
)
from tnfr.types import Glyph, real_scalar_epi


_PROGRESS_KEY = "_rng_jitter_progress"


def _graph(*, random_mode: bool, seed: int | None = 37) -> nx.Graph:
    graph = nx.path_graph(3)
    graph.graph.update(
        RANDOM_SEED=seed,
        NAV_RANDOM=random_mode,
        NAV_STRICT=False,
        GLYPH_FACTORS={"NAV_eta": 0.35, "NAV_jitter": 0.04},
    )
    dnfr_values = (-0.35, 0.2, 0.45)
    theta_values = (0.1, 6.1, 1.2)
    for node in graph:
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 0.2 + 0.1 * node,
                ALIAS_VF[0]: 0.5 + 0.1 * node,
                ALIAS_DNFR[0]: dnfr_values[node],
                ALIAS_THETA[0]: theta_values[node],
                "EPI_kind": "wave",
                "glyph_history": ["AL", "IL"],
            }
        )
    return graph


def _mixed_regime_graph(*, random_mode: bool) -> nx.Graph:
    graph = nx.Graph()
    labels = ("resonant-node", "latent-node", "active-node")
    graph.add_nodes_from(labels)
    graph.add_edges_from(zip(labels[:-1], labels[1:], strict=True))
    graph.graph.update(
        RANDOM_SEED=73,
        NAV_RANDOM=random_mode,
        NAV_STRICT=False,
        GLYPH_FACTORS={"NAV_eta": 0.35, "NAV_jitter": 0.04},
    )
    states = {
        "resonant-node": (0.8, 0.9, 0.45, 1.2, False),
        "latent-node": (0.2, 0.02, -0.35, 0.1, True),
        "active-node": (0.3, 0.6, 0.2, 6.1, False),
    }
    initial_draws = (4, 1, 6)
    for offset, node in enumerate(labels):
        epi, vf, dnfr, theta, latent = states[node]
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: epi,
                ALIAS_VF[0]: vf,
                ALIAS_DNFR[0]: dnfr,
                ALIAS_THETA[0]: theta,
                "EPI_kind": "wave",
                "glyph_history": ["AL", "IL"],
            }
        )
        if latent:
            graph.nodes[node]["latent"] = True
        if random_mode:
            graph.nodes[node][_PROGRESS_KEY] = {
                "seed": 73,
                "offset": offset,
                "draws": initial_draws[offset],
            }
    return graph


def _primary_state(graph: nx.Graph) -> dict[Any, tuple[object, ...]]:
    return {
        node: (
            float(real_scalar_epi(get_attr(data, ALIAS_EPI))),
            float(get_attr(data, ALIAS_VF)),
            float(get_attr(data, ALIAS_DNFR)),
            float(get_attr(data, ALIAS_THETA)),
            deepcopy(data.get(_PROGRESS_KEY)),
        )
        for node, data in graph.nodes(data=True)
    }


def _plain_state(
    graph: nx.Graph, *, omit_graph_keys: frozenset[str] = frozenset()
) -> tuple[object, ...]:
    return (
        tuple(
            (node, deepcopy(dict(data)))
            for node, data in graph.nodes(data=True)
        ),
        tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        ),
        deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key not in omit_graph_keys
            }
        ),
        (
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
    )


@pytest.mark.parametrize("random_mode", (False, True))
def test_nav_pointwise_structural_state_matches_direct_execution(
    random_mode: bool,
) -> None:
    pointwise = _graph(random_mode=random_mode)
    direct = _graph(random_mode=random_mode)

    result = execute_pointwise_stage(
        pointwise,
        Transition(),
        (0, 1, 2),
        vf_factor=1.1,
        phase_shift=0.3,
    )
    execute_operator_major_stage(
        direct,
        Transition(),
        (0, 1, 2),
        vf_factor=1.1,
        phase_shift=0.3,
    )

    assert _primary_state(pointwise) == _primary_state(direct)
    assert result.schedule == TWO_PHASE_JACOBI
    for node in pointwise:
        assert pointwise.nodes[node]["glyph_history"][-1] == "NAV"
        assert direct.nodes[node]["glyph_history"][-1] == "NAV"
        if random_mode:
            assert pointwise.nodes[node][_PROGRESS_KEY]["draws"] == 1
        else:
            assert _PROGRESS_KEY not in pointwise.nodes[node]


@pytest.mark.parametrize("random_mode", (False, True))
def test_nav_stage_matches_direct_across_regimes_and_continued_rng_streams(
    random_mode: bool,
) -> None:
    pointwise = _mixed_regime_graph(random_mode=random_mode)
    direct = _mixed_regime_graph(random_mode=random_mode)
    targets = ("active-node", "latent-node", "resonant-node")

    execute_pointwise_stage(
        pointwise,
        Transition(),
        targets,
        collect_metrics=True,
        vf_factor=1.1,
        phase_shift=0.3,
    )
    execute_operator_major_stage(
        direct,
        Transition(),
        targets,
        collect_metrics=True,
        vf_factor=1.1,
        phase_shift=0.3,
    )

    assert _primary_state(pointwise) == _primary_state(direct)
    assert pointwise.graph["_nav_transitions"] == direct.graph["_nav_transitions"]
    assert pointwise.graph["operator_metrics"] == direct.graph["operator_metrics"]
    assert pointwise.graph.get("recognized_coherence_patterns") == direct.graph.get(
        "recognized_coherence_patterns"
    )
    assert [
        event["regime_origin"] for event in pointwise.graph["_nav_transitions"]
    ] == ["active", "latent", "resonant"]
    for node in pointwise:
        assert pointwise.nodes[node]["glyph_history"] == direct.nodes[node][
            "glyph_history"
        ]
        assert get_attr(
            pointwise.nodes[node], ALIAS_SOURCE_GLYPH
        ) == get_attr(direct.nodes[node], ALIAS_SOURCE_GLYPH)
        assert pointwise.nodes[node]["_regime_before"] == direct.nodes[node][
            "_regime_before"
        ]


@pytest.mark.parametrize("random_mode", (False, True))
def test_nav_structural_state_and_node_rng_progress_are_target_order_invariant(
    random_mode: bool,
) -> None:
    forward = _graph(random_mode=random_mode)
    reverse = _graph(random_mode=random_mode)
    if random_mode:
        for graph in (forward, reverse):
            for node in graph:
                graph.nodes[node][_PROGRESS_KEY] = {
                    "seed": 37,
                    "offset": node,
                    "draws": node + 2,
                }

    forward_result = execute_pointwise_stage(
        forward,
        Transition(),
        (0, 1, 2),
        vf_factor=1.1,
        phase_shift=0.3,
    )
    reverse_result = execute_pointwise_stage(
        reverse,
        Transition(),
        (2, 1, 0),
        vf_factor=1.1,
        phase_shift=0.3,
    )

    assert _primary_state(forward) == _primary_state(reverse)
    assert forward_result.schedule == TWO_PHASE_JACOBI
    assert reverse_result.schedule == TWO_PHASE_JACOBI
    assert [event["node"] for event in forward.graph["_nav_transitions"]] == [
        0,
        1,
        2,
    ]
    assert [event["node"] for event in reverse.graph["_nav_transitions"]] == [
        2,
        1,
        0,
    ]
    if random_mode:
        for node in forward:
            assert forward.nodes[node][_PROGRESS_KEY] == reverse.nodes[node][
                _PROGRESS_KEY
            ]
            assert forward.nodes[node][_PROGRESS_KEY]["draws"] == node + 3


def test_nav_latent_targets_share_one_stage_instant() -> None:
    graph = _graph(random_mode=False)
    start = "2026-01-01T00:00:00+00:00"
    for node in graph:
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 0.2,
                ALIAS_VF[0]: 0.02,
                "latent": True,
                "latency_start_time": start,
                "preserved_epi": 0.2,
            }
        )

    execute_pointwise_stage(graph, Transition(), (2, 0, 1))

    durations = [graph.nodes[node]["silence_duration"] for node in graph]
    assert len(set(durations)) == 1
    for node in graph:
        assert "latent" not in graph.nodes[node]
        assert "latency_start_time" not in graph.nodes[node]
        assert "preserved_epi" not in graph.nodes[node]
        assert graph.nodes[node]["_regime_before"] == "latent"
    assert [
        event["regime_origin"] for event in graph.graph["_nav_transitions"]
    ] == ["latent", "latent", "latent"]


def test_nav_preflight_and_commit_observe_the_wall_clock_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingDateTime(datetime):
        calls = 0

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            instant = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
                seconds=cls.calls
            )
            cls.calls += 1
            return instant if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(network_stage_module, "datetime", CountingDateTime)
    monkeypatch.setattr(transition_module, "datetime", CountingDateTime)
    graph = _graph(random_mode=False)
    start = (
        datetime(2026, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=10)
    ).isoformat()
    for node in graph:
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 0.2,
                ALIAS_VF[0]: 0.02,
                "latent": True,
                "latency_start_time": start,
                "preserved_epi": 0.2,
            }
        )

    execute_pointwise_stage(graph, Transition(), (2, 0, 1))

    assert CountingDateTime.calls == 1
    assert {
        graph.nodes[node]["silence_duration"] for node in graph
    } == {10.0}


def test_nav_late_random_proposal_failure_leaves_every_target_unchanged() -> None:
    graph = _graph(random_mode=True)
    graph.nodes[2][_PROGRESS_KEY] = {
        "seed": 37,
        "offset": 2,
        "draws": -1,
    }
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="_rng_jitter_progress"):
        execute_pointwise_stage(graph, Transition(), (0, 1, 2))

    assert _plain_state(graph) == before


def test_nav_entropy_seed_is_not_written_before_late_proposal_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(random_mode=True, seed=None)
    graph.nodes[2][_PROGRESS_KEY] = {
        "seed": 37,
        "offset": 2,
        "draws": -1,
    }
    before = _plain_state(graph)
    observed_live_seeds: list[int | None] = []
    build_proposal = Transition._build_network_stage_proposal

    def observe_live_seed(
        self: Transition, *args: Any, **kwargs: Any
    ) -> Any:
        observed_live_seeds.append(graph.graph["RANDOM_SEED"])
        return build_proposal(self, *args, **kwargs)

    monkeypatch.setattr(
        Transition, "_build_network_stage_proposal", observe_live_seed
    )

    with pytest.raises(OperatorPreconditionError, match="_rng_jitter_progress"):
        execute_pointwise_stage(graph, Transition(), (0, 1, 2))

    assert observed_live_seeds == [None, None, None]
    assert _plain_state(graph) == before
    assert graph.graph["RANDOM_SEED"] is None


def test_nav_stale_rng_progress_rejects_commit_and_rolls_back_prior_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(random_mode=True)
    before = _plain_state(graph)
    emit_warnings = network_stage_module._emit_pointwise_precommit_warnings

    def change_progress_after_proposal(
        subject: nx.Graph, proposals: tuple[Any, ...]
    ) -> None:
        emit_warnings(subject, proposals)
        subject.nodes[1][_PROGRESS_KEY] = {
            "seed": 37,
            "offset": 1,
            "draws": 99,
        }

    monkeypatch.setattr(
        network_stage_module,
        "_emit_pointwise_precommit_warnings",
        change_progress_after_proposal,
    )

    with pytest.raises(RuntimeError, match="stale jitter proposal"):
        execute_pointwise_stage(graph, Transition(), (0, 1, 2))

    assert _plain_state(graph) == before


@pytest.mark.parametrize("seed", (37, None))
def test_nav_stale_seed_rejects_commit_before_rng_progress(
    monkeypatch: pytest.MonkeyPatch,
    seed: int | None,
) -> None:
    graph = _graph(random_mode=True, seed=seed)
    before = _plain_state(graph)
    emit_warnings = network_stage_module._emit_pointwise_precommit_warnings

    def change_seed_after_proposal(
        subject: nx.Graph, proposals: tuple[Any, ...]
    ) -> None:
        emit_warnings(subject, proposals)
        subject.graph["RANDOM_SEED"] = 99

    monkeypatch.setattr(
        network_stage_module,
        "_emit_pointwise_precommit_warnings",
        change_seed_after_proposal,
    )

    with pytest.raises(RuntimeError, match="stale NAV random seed"):
        execute_pointwise_stage(graph, Transition(), (0, 1, 2))

    assert _plain_state(graph) == before


def test_nav_pressure_callback_failure_rolls_back_all_storage() -> None:
    graph = _graph(random_mode=True)
    before = _plain_state(graph)
    calls: list[nx.Graph] = []

    def rejected_callback(subject: nx.Graph) -> None:
        calls.append(subject)
        subject.nodes[0][ALIAS_EPI[0]] = 99.0
        subject.edges[0, 1]["callback_side_effect"] = True
        subject.graph["callback_side_effect"] = True
        subject.add_node("callback-added", EPI=99.0)
        raise RuntimeError("NAV pressure callback rejected stage")

    with pytest.raises(RuntimeError, match="pressure callback rejected"):
        execute_pointwise_stage(
            graph,
            Transition(),
            (0, 1, 2),
            compute_delta_nfr=rejected_callback,
        )

    assert calls == [graph]
    assert _plain_state(graph) == before


class _RejectingMonitor:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.pending: Any = None

    def before_operator(self, graph: nx.Graph, node: Any) -> None:
        self.pending = node
        self.events.append(("before", node))

    def after_operator(self, graph: nx.Graph, node: Any, operator: str) -> None:
        self.events.append(("after", node))
        if node == 1:
            graph.graph["monitor_side_effect"] = operator
            graph.nodes[node]["monitor_side_effect"] = True
            raise RuntimeError("NAV monitor rejected second target")
        self.pending = None

    def discard_pending_operator(self) -> None:
        self.pending = None


def test_nav_monitor_failure_rolls_back_graph_and_monitor_state() -> None:
    graph = _graph(random_mode=True)
    monitor = _RejectingMonitor()
    graph.graph["integrity_monitor"] = monitor
    before = _plain_state(
        graph, omit_graph_keys=frozenset({"integrity_monitor"})
    )
    monitor_before = deepcopy(vars(monitor))

    with pytest.raises(RuntimeError, match="monitor rejected second target"):
        execute_pointwise_stage(graph, Transition(), (0, 1, 2))

    assert graph.graph["integrity_monitor"] is monitor
    assert _plain_state(
        graph, omit_graph_keys=frozenset({"integrity_monitor"})
    ) == before
    assert vars(monitor) == monitor_before


def test_nav_metric_failure_rolls_back_prior_metrics_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(random_mode=True)
    before = _plain_state(graph)

    def rejected_metric(
        self: Transition,
        subject: nx.Graph,
        node: Any,
        state_before: dict[str, Any],
    ) -> dict[str, Any]:
        if node == 1:
            subject.nodes[node]["metric_side_effect"] = True
            subject.graph["metric_side_effect"] = state_before["dnfr"]
            raise RuntimeError("NAV metric rejected second target")
        return {"operator": self.name, "node": node}

    monkeypatch.setattr(Transition, "_collect_metrics", rejected_metric)

    with pytest.raises(RuntimeError, match="metric rejected second target"):
        execute_pointwise_stage(
            graph, Transition(), (0, 1, 2), collect_metrics=True
        )

    assert _plain_state(graph) == before


def test_nav_entropy_seed_and_progress_are_restored_after_rejection() -> None:
    graph = _graph(random_mode=True, seed=None)
    before = _plain_state(graph)
    observed: dict[str, object] = {}

    def rejected_callback(subject: nx.Graph) -> None:
        observed["seed"] = subject.graph["RANDOM_SEED"]
        observed["progress"] = deepcopy(
            subject.nodes[0].get(_PROGRESS_KEY)
        )
        raise RuntimeError("reject resolved entropy seed")

    with pytest.raises(RuntimeError, match="reject resolved entropy seed"):
        execute_pointwise_stage(
            graph,
            Transition(),
            (0, 1, 2),
            compute_delta_nfr=rejected_callback,
        )

    assert type(observed["seed"]) is int
    assert isinstance(observed["progress"], dict)
    assert _plain_state(graph) == before
    assert graph.graph["RANDOM_SEED"] is None
    assert all(_PROGRESS_KEY not in graph.nodes[node] for node in graph)


def test_nav_entropy_seed_persists_and_node_streams_advance_on_success() -> None:
    graph = _graph(random_mode=True, seed=None)

    execute_pointwise_stage(graph, Transition(), (0, 1, 2))

    realized_seed = graph.graph["RANDOM_SEED"]
    assert type(realized_seed) is int
    for node in graph:
        progress = graph.nodes[node][_PROGRESS_KEY]
        assert progress == {
            "seed": realized_seed,
            "offset": node,
            "draws": 1,
        }

    execute_pointwise_stage(graph, Transition(), (2, 1, 0))

    assert graph.graph["RANDOM_SEED"] == realized_seed
    for node in graph:
        progress = graph.nodes[node][_PROGRESS_KEY]
        assert progress == {
            "seed": realized_seed,
            "offset": node,
            "draws": 2,
        }


def test_nav_runtime_schedule_matches_completed_stage_contract() -> None:
    contract = stage_contract_for(Glyph.NAV)

    assert Glyph.NAV in POINTWISE_TWO_PHASE_GLYPHS
    assert contract.current_schedule is StageSchedule.TWO_PHASE_JACOBI
    assert contract.structural_overlap is StructuralOverlap.DISJOINT_TARGET_WRITES
    assert contract.merge_law is MergeLaw.IMMUTABLE_PROPOSAL_COMMIT
    assert contract.merge_law_status is MergeLawStatus.IMPLEMENTED_AND_TESTED
    assert contract.rollback_scope is RollbackScope.STAGE
    assert contract.direct_call_rollback_scope is RollbackScope.NONE
    assert contract.structural_state_target_order_invariant is True
    assert contract.two_phase_contract_complete
    assert contract.blockers == ()
    assert {
        StageResource.TARGET_EPI,
        StageResource.TARGET_NU_F,
        StageResource.TARGET_THETA,
        StageResource.TARGET_DELTA_NFR,
        StageResource.TARGET_HISTORY,
        StageResource.TARGET_METADATA,
        StageResource.GRAPH_CONFIGURATION,
        StageResource.GRAPH_TELEMETRY,
        StageResource.GRAPH_RUNTIME,
        StageResource.RNG_PROGRESS,
        StageResource.WALL_CLOCK,
    } <= contract.read_set
    assert {
        StageResource.TARGET_NU_F,
        StageResource.TARGET_THETA,
        StageResource.TARGET_DELTA_NFR,
        StageResource.TARGET_HISTORY,
        StageResource.TARGET_METADATA,
        StageResource.GRAPH_TELEMETRY,
        StageResource.GRAPH_RUNTIME,
        StageResource.RNG_PROGRESS,
    } <= contract.write_set
    assert "per-node RNG progress" in contract.structural_state_target_order_scope
    assert "shared latency time" in contract.structural_state_target_order_scope
    assert "retain requested target order" in (
        contract.structural_state_target_order_scope
    )
    assert "NAV" not in {
        gap.glyph for gap in remaining_stage_contract_gaps()
    }

    graph = _graph(random_mode=True)
    operator = Transition()
    result = execute_pointwise_stage(graph, operator, (0, 1, 2))

    assert result.schedule == TWO_PHASE_JACOBI
    assert graph.graph[STAGE_SCHEDULE_KEY] == {
        "operator": operator.name,
        "glyph": "NAV",
        "schedule": TWO_PHASE_JACOBI,
        "nodes_processed": 3,
    }
    assert graph.graph[STAGE_CONTRACT_KEY] == stage_schedule_metadata(
        operator,
        observed_schedule=TWO_PHASE_JACOBI,
    )
    assert graph.graph[STAGE_CONTRACT_KEY][
        "executed_two_phase_contract_complete"
    ] is True
