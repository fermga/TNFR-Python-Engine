"""GPU block strategies must commit through canonical AL and RA operators."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.operators.strategies import gpu_strategies
from tnfr.operators.strategies.gpu_strategies import (
    GPUEmissionStrategy,
    GPUResonanceStrategy,
)
from tnfr.operators.strategies.strategy import (
    StrategyContext,
    StructuralFields,
)
from tnfr.node import NodeNX
from tnfr.types import real_scalar_epi


class _Backend:
    name = "test-gpu"


class _PreviewEngine:
    is_available = True
    math_backend = _Backend()

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.preview_graph = None

    def compute_delta_nfr_from_graph(self, graph):
        self.preview_graph = graph
        if self.fail:
            raise RuntimeError("preview unavailable")
        if graph:
            first = next(iter(graph))
            graph.nodes[first][ALIAS_EPI[0]] = 999.0
        return {node: 0.25 for node in graph}


class _LateAvailabilityFailureEngine(_PreviewEngine):
    def __init__(self) -> None:
        super().__init__()
        self.availability_reads = 0

    @property
    def is_available(self) -> bool:
        self.availability_reads += 1
        if self.availability_reads > 1:
            raise RuntimeError("late availability telemetry failed")
        return True


class _FailingBackend:
    @property
    def name(self) -> str:
        raise RuntimeError("late backend telemetry failed")


class _LateBackendFailureEngine(_PreviewEngine):
    math_backend = _FailingBackend()


class _NoncanonicalDirectedPreviewEngine(_PreviewEngine):
    def compute_delta_nfr_from_graph(self, graph):
        self.preview_graph = graph
        return {0: 2.0, 1: -1.0}


def _context(operator: str) -> StrategyContext:
    minimum = 200 if operator == "AL" else 100
    return StrategyContext(
        partition_id="gpu-block",
        operator_sequence_position=2,
        structural_fields=StructuralFields(
            phi_s=0.2,
            phase_gradient=0.1,
            phase_curvature=0.05,
            coherence_length=2.0,
        ),
        dispatcher_capabilities={"gpu": True},
        backend="gpu",
        block_size=minimum,
        boundary_overlap=1,
        seed=17,
    )


def _strategy(strategy_type, engine: _PreviewEngine):
    strategy = object.__new__(strategy_type)
    strategy.gpu_engine = engine
    return strategy


def _prepare(monkeypatch, strategy_type, graph, engine=None):
    monkeypatch.setattr(gpu_strategies, "HAS_GPU_ENGINE", True)
    engine = engine or _PreviewEngine()
    strategy = _strategy(strategy_type, engine)
    prepared = strategy.prepare(_context(strategy.operator), graph)
    return strategy, prepared, engine


def _node_state(graph: nx.Graph, node: int) -> tuple:
    data = graph.nodes[node]
    return (
        real_scalar_epi(get_attr(data, ALIAS_EPI)),
        get_attr(data, ALIAS_VF),
        get_attr(data, ALIAS_DNFR),
        get_attr(data, ALIAS_THETA),
        get_attr(data, ALIAS_EPI_KIND, "", conv=lambda value: value),
        tuple(data.get("glyph_history", ())),
        get_attr(data, ALIAS_SOURCE_GLYPH, "", conv=lambda value: value),
    )


def _observable_snapshot(graph: nx.Graph) -> tuple[dict, tuple, dict]:
    graph_data = {
        key: deepcopy(value)
        for key, value in graph.graph.items()
        if key != "integrity_monitor"
    }
    return (
        deepcopy({node: dict(data) for node, data in graph.nodes(data=True)}),
        deepcopy(tuple(graph.edges(data=True))),
        graph_data,
    )


def _graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph["RANDOM_SEED"] = 17
    for node in graph:
        graph.nodes[node].update(
            EPI=0.1 + 0.1 * node,
            nu_f=1.0 + node,
            delta_nfr=0.3,
            theta=0.2 * node,
            EPI_kind="wave",
        )
    return graph


def test_gpu_emission_uses_canonical_factor_and_metadata(monkeypatch) -> None:
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"AL_boost": 0.25}
    before = {node: _node_state(graph, node) for node in graph}
    strategy, prepared, engine = _prepare(
        monkeypatch, GPUEmissionStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is True
    assert result.telemetry["update_schedule"] == "two_phase_jacobi"
    assert result.telemetry["gpu_acceleration"] is False
    assert result.telemetry["backend"] == "canonical-cpu"
    assert result.telemetry["auxiliary_gpu_preview"] is True
    assert result.telemetry["auxiliary_gpu_preview_mean"] == pytest.approx(0.25)
    assert result.telemetry["auxiliary_gpu_preview_backend"] is None
    assert "mean_delta_nfr" not in result.telemetry
    assert engine.preview_graph is not graph
    for node in graph:
        after = _node_state(graph, node)
        assert after[0] == pytest.approx(before[node][0] + 0.25)
        assert after[1:4] == before[node][1:4]
        assert after[4] == "wave"
        assert after[5][-1] == "AL"
        assert after[6] == "AL"
        assert graph.nodes[node]["_emission_activated"] is True
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 1


def test_gpu_resonance_uses_canonical_factors_identity_and_provenance(
    monkeypatch,
) -> None:
    graph = _graph()
    graph.nodes[0].update(EPI=0.2, nu_f=1.0, glyph_history=["AL", "IL"])
    graph.nodes[1].update(EPI=0.6, nu_f=2.0, glyph_history=["AL", "IL"])
    graph.graph["GLYPH_FACTORS"] = {
        "RA_epi_diff": 0.5,
        "RA_vf_amplification": 0.25,
        "RA_phase_coupling": 0.5,
    }
    strategy, prepared, _engine = _prepare(
        monkeypatch, GPUResonanceStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is True
    assert result.telemetry["update_schedule"] == "two_phase_jacobi"
    assert result.telemetry["resonance_amplification"] is True
    assert result.telemetry["resonance_amplified_nodes"] == 2
    assert _node_state(graph, 0)[0] == pytest.approx(0.4)
    assert _node_state(graph, 0)[1] == pytest.approx(1.25)
    assert _node_state(graph, 0)[3] == pytest.approx(0.1)
    assert _node_state(graph, 1)[0] == pytest.approx(0.4)
    assert _node_state(graph, 1)[1] == pytest.approx(2.5)
    for node in graph:
        state = _node_state(graph, node)
        assert state[4] == "wave"
        assert state[5][-1] == "RA"
        assert state[6] == "RA"


def test_gpu_resonance_reports_inactive_frequency_amplification(monkeypatch) -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node].update(EPI=0.0, glyph_history=["AL", "IL"])
    strategy, prepared, _engine = _prepare(
        monkeypatch, GPUResonanceStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is True
    assert result.telemetry["resonance_amplification"] is False
    assert result.telemetry["resonance_amplified_nodes"] == 0


def test_gpu_resonance_runs_the_shared_pressure_refresh(monkeypatch) -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node]["glyph_history"] = ["AL", "IL"]
    calls = []

    def refresh(subject) -> None:
        calls.append(subject)
        for node in subject:
            subject.nodes[node][ALIAS_DNFR[0]] = 7.0

    graph.graph["compute_delta_nfr"] = refresh
    strategy, prepared, _engine = _prepare(
        monkeypatch, GPUResonanceStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is True
    assert calls == [graph]
    assert all(get_attr(graph.nodes[node], ALIAS_DNFR) == 7.0 for node in graph)


def test_gpu_resonance_rolls_back_a_failed_pressure_refresh(monkeypatch) -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node]["glyph_history"] = ["AL", "IL"]

    def rejected_refresh(subject) -> None:
        subject.nodes[0][ALIAS_DNFR[0]] = 99.0
        subject.graph["refresh_side_effect"] = True
        raise RuntimeError("GPU pressure refresh rejected")

    graph.graph["compute_delta_nfr"] = rejected_refresh
    before = _observable_snapshot(graph)
    strategy, prepared, _engine = _prepare(
        monkeypatch, GPUResonanceStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is False
    assert result.telemetry["rolled_back"] is True
    assert "GPU pressure refresh rejected" in result.telemetry["error"]
    assert _observable_snapshot(graph) == before


def test_gpu_resonance_rejects_phase_incompatible_block_atomically(
    monkeypatch,
) -> None:
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["AL", "IL"]
    graph.nodes[1]["glyph_history"] = ["AL", "IL"]
    graph.nodes[1][ALIAS_THETA[0]] = 2.0
    before = _observable_snapshot(graph)
    strategy, prepared, _engine = _prepare(
        monkeypatch, GPUResonanceStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is False
    assert result.telemetry["rolled_back"] is True
    assert "phase" in result.telemetry["error"].lower()
    assert _observable_snapshot(graph) == before
    assert "_node_cache" not in graph.graph


def test_gpu_resonance_rejects_identity_change_atomically(monkeypatch) -> None:
    graph = _graph()
    graph.nodes[0].update(
        EPI=-0.1, theta=0.0, glyph_history=["AL", "IL"]
    )
    graph.nodes[1].update(EPI=1.0, theta=0.1, glyph_history=["AL", "IL"])
    graph.graph["GLYPH_FACTORS"] = {"RA_epi_diff": 0.5}
    before = _observable_snapshot(graph)
    strategy, prepared, _engine = _prepare(
        monkeypatch, GPUResonanceStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is False
    assert "sign" in result.telemetry["error"].lower()
    assert _observable_snapshot(graph) == before


@dataclass
class _RejectSecondPostcondition:
    before_count: int = 0
    after_count: int = 0
    pending: int | None = None

    def before_operator(self, graph, node) -> None:
        self.before_count += 1
        self.pending = node

    def after_operator(self, graph, node, operator) -> None:
        self.after_count += 1
        if node == 1:
            graph.add_node("postcondition-side-effect", EPI=99.0)
            raise RuntimeError("postcondition rejected second node")
        self.pending = None

    def discard_pending_operator(self) -> None:
        self.pending = None


def test_gpu_block_rolls_back_prior_commits_and_monitor_state(monkeypatch) -> None:
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"AL_boost": 0.2}
    monitor = _RejectSecondPostcondition()
    graph.graph["integrity_monitor"] = monitor
    before = _observable_snapshot(graph)
    monitor_before = deepcopy(vars(monitor))
    strategy, prepared, _engine = _prepare(
        monkeypatch, GPUEmissionStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is False
    assert result.telemetry["rolled_back"] is True
    assert "postcondition rejected" in result.telemetry["error"]
    assert _observable_snapshot(graph) == before
    assert vars(monitor) == monitor_before
    assert graph.graph["integrity_monitor"] is monitor


@dataclass
class _MutateNestedCacheThenReject:
    pending: int | None = None

    def before_operator(self, graph, node) -> None:
        self.pending = node

    def after_operator(self, graph, node, operator) -> None:
        graph.graph["custom_cache"]["nested"]["value"] = 9
        raise RuntimeError("nested cache mutation rejected")

    def discard_pending_operator(self) -> None:
        self.pending = None


def test_gpu_rollback_restores_nested_runtime_mapping_contents(monkeypatch) -> None:
    graph = _graph()
    adapter = NodeNX.from_graph(graph, 0)
    node_cache = graph.graph["_node_cache"]
    custom_cache = {"nested": {"value": 1}}
    graph.graph["custom_cache"] = custom_cache
    graph.graph["integrity_monitor"] = _MutateNestedCacheThenReject()
    nodes_before = deepcopy(
        {node: dict(data) for node, data in graph.nodes(data=True)}
    )
    strategy, prepared, _engine = _prepare(
        monkeypatch, GPUEmissionStrategy, graph
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is False
    assert result.telemetry["rolled_back"] is True
    assert graph.graph["custom_cache"] is custom_cache
    assert custom_cache == {"nested": {"value": 1}}
    assert graph.graph["_node_cache"] is node_cache
    assert node_cache == {0: adapter}
    assert node_cache[0] is adapter
    assert {node: dict(data) for node, data in graph.nodes(data=True)} == nodes_before


@pytest.mark.parametrize(
    ("strategy_type", "engine", "message"),
    [
        (
            GPUEmissionStrategy,
            _LateAvailabilityFailureEngine(),
            "late availability telemetry failed",
        ),
    ],
)
def test_gpu_postcommit_telemetry_failure_rolls_back_truthfully(
    monkeypatch,
    strategy_type,
    engine,
    message: str,
) -> None:
    graph = _graph()
    if strategy_type is GPUResonanceStrategy:
        for node in graph:
            graph.nodes[node]["glyph_history"] = ["AL", "IL"]
    strategy, prepared, _engine = _prepare(
        monkeypatch,
        strategy_type,
        graph,
        engine=engine,
    )
    before = _observable_snapshot(graph)

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is False
    assert result.telemetry["rolled_back"] is True
    assert result.telemetry["rollback_error"] is None
    assert message in result.telemetry["error"]
    assert _observable_snapshot(graph) == before


def test_gpu_rollback_failure_keeps_primary_error_and_reports_secondary(
    monkeypatch,
) -> None:
    graph = _graph()
    strategy, prepared, _engine = _prepare(
        monkeypatch,
        GPUEmissionStrategy,
        graph,
        engine=_LateAvailabilityFailureEngine(),
    )
    before = _observable_snapshot(graph)
    restore = gpu_strategies.GraphTransactionSnapshot.restore

    def restore_then_fail(snapshot, live_graph) -> None:
        restore(snapshot, live_graph)
        raise RuntimeError("secondary rollback failure")

    monkeypatch.setattr(
        gpu_strategies.GraphTransactionSnapshot,
        "restore",
        restore_then_fail,
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is False
    assert result.telemetry["rolled_back"] is False
    assert result.telemetry["error"] == "late availability telemetry failed"
    assert (
        result.telemetry["rollback_error"]
        == "RuntimeError: secondary rollback failure"
    )
    assert result.warnings == [
        "Canonical Emission transaction failed: "
        "late availability telemetry failed"
    ]
    assert _observable_snapshot(graph) == before


def test_preview_without_provenance_does_not_guess_backend(monkeypatch) -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node]["glyph_history"] = ["AL", "IL"]
    strategy, prepared, _engine = _prepare(
        monkeypatch,
        GPUResonanceStrategy,
        graph,
        engine=_LateBackendFailureEngine(),
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is True
    assert result.telemetry["auxiliary_gpu_preview"] is True
    assert result.telemetry["auxiliary_gpu_preview_backend"] is None


def test_gpu_preview_failure_falls_back_to_canonical_commit(monkeypatch) -> None:
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"AL_boost": 0.1}
    strategy, prepared, _engine = _prepare(
        monkeypatch,
        GPUEmissionStrategy,
        graph,
        engine=_PreviewEngine(fail=True),
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is True
    assert result.telemetry["auxiliary_gpu_preview"] is False
    assert result.telemetry["backend"] == "canonical-cpu"
    assert result.warnings == ["Auxiliary GPU preview failed: preview unavailable"]
    for node in graph:
        assert graph.nodes[node]["glyph_history"][-1] == "AL"


def test_directed_gpu_response_is_exposed_only_as_an_auxiliary_preview(
    monkeypatch,
) -> None:
    graph = nx.DiGraph()
    graph.add_node(0, EPI=0.2, nu_f=1.0, theta=0.0, EPI_kind="wave")
    graph.add_node(1, EPI=0.1, nu_f=1.0, theta=0.1, EPI_kind="wave")
    graph.add_edge(0, 1, weight=4.0)
    graph.graph["GLYPH_FACTORS"] = {"AL_boost": 0.1}
    strategy, prepared, engine = _prepare(
        monkeypatch,
        GPUEmissionStrategy,
        graph,
        engine=_NoncanonicalDirectedPreviewEngine(),
    )

    result = strategy.apply(prepared)

    assert result.telemetry["canonical_commit"] is True
    assert result.telemetry["gpu_acceleration"] is False
    assert result.telemetry["backend"] == "canonical-cpu"
    assert result.telemetry["auxiliary_gpu_preview"] is True
    assert result.telemetry["auxiliary_gpu_preview_mean"] == pytest.approx(0.5)
    assert "mean_delta_nfr" not in result.telemetry
    assert engine.preview_graph.is_directed()
    assert engine.preview_graph.edges[0, 1]["weight"] == 4.0
    assert real_scalar_epi(graph.nodes[0]["EPI"]) == pytest.approx(0.3)
    assert real_scalar_epi(graph.nodes[1]["EPI"]) == pytest.approx(0.2)


def test_gpu_preview_preserves_directed_multiedge_data() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(0, EPI=0.2, nu_f=1.0, theta=0.0)
    graph.add_node(1, EPI=0.4, nu_f=2.0, theta=0.1)
    graph.add_edge(1, 0, key="channel", weight=3.0, length=0.5)

    preview = gpu_strategies._gpu_preview_graph(graph)

    assert isinstance(preview, nx.MultiDiGraph)
    assert preview.has_edge(1, 0, "channel")
    assert not preview.has_edge(0, 1, "channel")
    assert preview.edges[1, 0, "channel"] == {"weight": 3.0, "length": 0.5}
    preview.edges[1, 0, "channel"]["weight"] = 9.0
    assert graph.edges[1, 0, "channel"]["weight"] == 3.0


@pytest.mark.parametrize("strategy_type", [GPUEmissionStrategy, GPUResonanceStrategy])
def test_gpu_resource_estimate_marks_unmeasured_quantities_unknown(
    strategy_type,
) -> None:
    ctx = _context(strategy_type.operator)

    estimate = object.__new__(strategy_type).resource_estimate(ctx)

    expected_memory = 8 * (ctx.block_size**2 + 2 * ctx.block_size)
    assert estimate.memory_bytes == expected_memory
    assert estimate.time_ms is None
    assert estimate.delta_nfr is None
    assert estimate.phi_s_drift is None
    assert estimate.failure_risk == "unknown"


@pytest.mark.parametrize("strategy_type", [GPUEmissionStrategy, GPUResonanceStrategy])
def test_context_factor_overrides_are_explicitly_rejected(
    monkeypatch, strategy_type
) -> None:
    monkeypatch.setattr(gpu_strategies, "HAS_GPU_ENGINE", True)
    strategy = _strategy(strategy_type, _PreviewEngine())
    ctx = replace(
        _context(strategy.operator),
        dispatcher_capabilities={"gpu": True, "GLYPH_FACTORS": {"AL_boost": 0.7}},
    )

    with pytest.raises(ValueError, match="do not accept factor overrides"):
        strategy.prepare(ctx, _graph())


class _RequiredConstructorGraph(nx.MultiDiGraph):
    def __init__(self, namespace: str, incoming_graph_data=None, **attr) -> None:
        self.namespace = namespace
        super().__init__(incoming_graph_data, **attr)


def test_gpu_preview_supports_graph_subclass_with_required_constructor() -> None:
    graph = _RequiredConstructorGraph("research-cell")
    graph.graph["nested"] = {"value": 1}
    graph.add_node(0, EPI=0.2, nu_f=1.0, theta=0.0)
    graph.add_node(1, EPI=0.4, nu_f=2.0, theta=0.1)
    graph.add_edge(0, 1, key="channel", weight=2.0)

    preview = gpu_strategies._gpu_preview_graph(graph)

    assert isinstance(preview, _RequiredConstructorGraph)
    assert preview.namespace == "research-cell"
    assert preview is not graph
    preview.graph["nested"]["value"] = 9
    preview.nodes[0]["EPI"] = 7.0
    preview.edges[0, 1, "channel"]["weight"] = 8.0
    assert graph.graph["nested"]["value"] == 1
    assert graph.nodes[0]["EPI"] == pytest.approx(0.2)
    assert graph.edges[0, 1, "channel"]["weight"] == pytest.approx(2.0)


def test_gpu_proof_hash_binds_context_operator_and_state() -> None:
    graph = _graph()
    ctx = _context("AL")

    base = gpu_strategies._operation_proof_hash(
        graph, ctx, "AL", outcome="committed"
    )
    other_partition = gpu_strategies._operation_proof_hash(
        graph,
        replace(ctx, partition_id="other-block"),
        "AL",
        outcome="committed",
    )
    other_position = gpu_strategies._operation_proof_hash(
        graph,
        replace(ctx, operator_sequence_position=3),
        "AL",
        outcome="committed",
    )
    other_operator = gpu_strategies._operation_proof_hash(
        graph, ctx, "RA", outcome="committed"
    )
    graph.nodes[0]["EPI"] = 0.9
    other_state = gpu_strategies._operation_proof_hash(
        graph, ctx, "AL", outcome="committed"
    )

    assert len(base) == 64
    proofs = {base, other_partition, other_position, other_operator, other_state}
    assert len(proofs) == 5


def test_gpu_recommendation_reports_cpu_commit_and_unknown_speedup(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gpu_strategies, "HAS_GPU_ENGINE", True)
    monkeypatch.setattr(
        gpu_strategies,
        "get_unified_gpu_system",
        lambda: _PreviewEngine(),
    )

    recommendation = gpu_strategies.get_gpu_strategy_recommendations(1000)

    assert recommendation["use_gpu"] is False
    assert recommendation["preferred_operators"] == []
    assert recommendation["canonical_execution_backend"] == "canonical-cpu"
    assert recommendation["speedup_factor"] is None
    assert recommendation["memory_estimate_mb"] is None
    assert recommendation["auxiliary_preview_available"] is True
    assert recommendation["auxiliary_preview_backend"] is None
    assert recommendation["auxiliary_preview_memory_estimate_mb"] is None


@pytest.mark.parametrize("invalid", [True, -1, 1.5])
def test_gpu_recommendation_validates_graph_size(invalid) -> None:
    error = ValueError if invalid == -1 else TypeError
    with pytest.raises(error):
        gpu_strategies.get_gpu_strategy_recommendations(invalid)
