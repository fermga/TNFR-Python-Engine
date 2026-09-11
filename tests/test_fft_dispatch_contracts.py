"""Focused contracts for FFT dispatch through the public coordination layers."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.dynamics.computational_hub import (
    ComputationRequest,
    EngineType,
    TNFRComputationalHub,
)
from tnfr.dynamics.optimization_orchestrator import (
    OptimizationStrategy,
    TNFROptimizationOrchestrator,
)


def _graph() -> nx.Graph:
    graph = nx.path_graph(3)
    for node, epi in enumerate((0.0, 1.0, -0.25)):
        graph.nodes[node].update(EPI=epi, nu_f=1.0, theta=0.0, DeltaNFR=0.5)
    graph.graph["marker"] = {"nested": [1, 2]}
    return graph


def _graph_surface(graph: nx.Graph) -> tuple[object, object, object]:
    return (
        tuple((node, deepcopy(dict(graph.nodes[node]))) for node in graph.nodes),
        tuple(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        ),
        deepcopy(dict(graph.graph)),
    )


class _MutatingFFTEngine:
    def __init__(self, *, residual: float = 1e-12) -> None:
        self.calls = 0
        self.residual = residual

    def run_fft_simulation(
        self, graph: nx.Graph, num_steps: int, dt: float
    ) -> dict[str, object]:
        self.calls += 1
        graph.nodes[0]["EPI"] = 99.0
        return {
            "status": "success",
            "simulation_time": 0.25,
            "total_steps": num_steps,
            "fft_operations": num_steps,
            "cache_hits": 0,
            "steps_per_second": 40.0,
            "final_time": num_steps * dt,
            "final_coherence": 0.75,
            "final_phase_sync": 0.8,
            "max_nodal_residual": self.residual,
        }


def _orchestrator(engine: _MutatingFFTEngine) -> TNFROptimizationOrchestrator:
    orchestrator = object.__new__(TNFROptimizationOrchestrator)
    orchestrator.default_memory_budget = 512.0
    orchestrator.fft_engine = engine
    orchestrator.optimization_history = []
    orchestrator.strategy_performance = {}
    return orchestrator


@pytest.mark.parametrize(
    ("operation", "parameters", "message"),
    (
        ("temporal_evolution", {}, "only operation='epi_diffusion'"),
        (
            "epi_diffusion",
            {"pressure_model": "stored_delta_nfr"},
            "requires pressure_model='epi_diffusion'",
        ),
    ),
)
def test_orchestrator_rejects_non_diffusion_fft_dispatch_before_graph_write(
    operation: str, parameters: dict[str, object], message: str
) -> None:
    graph = _graph()
    before = _graph_surface(graph)
    engine = _MutatingFFTEngine()

    orchestrator = _orchestrator(engine)
    result = orchestrator.execute_optimization(
        graph,
        operation,
        OptimizationStrategy.SPECTRAL_FFT,
        num_steps=1,
        dt=0.1,
        **parameters,
    )

    assert result.accuracy_preserved is False
    assert message in result.details["error"]
    assert engine.calls == 0
    assert _graph_surface(graph) == before
    assert OptimizationStrategy.SPECTRAL_FFT not in orchestrator.strategy_performance


def test_main_orchestrator_entry_preserves_explicit_fft_request_for_rejection() -> None:
    graph = _graph()
    before = _graph_surface(graph)
    engine = _MutatingFFTEngine()
    orchestrator = _orchestrator(engine)

    result = orchestrator.optimize_graph_operation(
        graph,
        "general",
        strategy=OptimizationStrategy.SPECTRAL_FFT,
        num_steps=1,
    )

    assert "only operation='epi_diffusion'" in result.details["error"]
    assert engine.calls == 0
    assert _graph_surface(graph) == before


def test_orchestrator_reports_explicit_model_and_only_measured_fft_metrics() -> None:
    graph = _graph()
    engine = _MutatingFFTEngine(residual=1e-12)

    orchestrator = _orchestrator(engine)
    result = orchestrator.execute_optimization(
        graph,
        "epi_diffusion",
        OptimizationStrategy.SPECTRAL_FFT,
        num_steps=2,
        dt=0.1,
        nodal_residual_tolerance=1e-9,
    )

    assert engine.calls == 1
    assert result.accuracy_preserved is True
    assert result.speedup_factor == 1.0
    assert result.memory_used_mb == 0.0
    assert result.details["operation"] == "epi_diffusion"
    assert result.details["pressure_model"] == "epi_diffusion"
    assert result.details["performance_measurements"] == {
        "throughput_steps_per_second": 40.0,
        "speedup_factor": None,
        "memory_used_mb": None,
    }
    assert result.details["accuracy_verification"] == {
        "basis": "nodal_equation_residual",
        "max_nodal_residual": 1e-12,
        "tolerance": 1e-9,
        "passed": True,
        "baseline_comparison_performed": False,
    }
    assert OptimizationStrategy.SPECTRAL_FFT not in orchestrator.strategy_performance


def _hub(engine: _MutatingFFTEngine) -> TNFRComputationalHub:
    hub = object.__new__(TNFRComputationalHub)
    hub._engines = {EngineType.FFT_ENGINE: engine}
    hub._engine_performance = {engine_type: [] for engine_type in EngineType}
    return hub


@pytest.mark.parametrize(
    ("operation", "parameters", "message"),
    (
        ("multi_step", {"num_steps": 1}, "only operation='epi_diffusion'"),
        (
            "epi_diffusion",
            {"num_steps": 1, "pressure_model": "stored_delta_nfr"},
            "requires pressure_model='epi_diffusion'",
        ),
    ),
)
def test_hub_rejects_non_diffusion_fft_dispatch_before_graph_write(
    operation: str, parameters: dict[str, object], message: str
) -> None:
    graph = _graph()
    before = _graph_surface(graph)
    engine = _MutatingFFTEngine()
    request = ComputationRequest(
        engine_type=EngineType.FFT_ENGINE,
        operation=operation,
        graph=graph,
        parameters=parameters,
    )

    result = _hub(engine)._execute_computation(request)

    assert result.success is False
    assert message in (result.error_message or "")
    assert engine.calls == 0
    assert _graph_surface(graph) == before


def test_hub_tags_successful_fft_result_with_its_pressure_model() -> None:
    graph = _graph()
    engine = _MutatingFFTEngine()
    request = ComputationRequest(
        engine_type=EngineType.FFT_ENGINE,
        operation="epi_diffusion",
        graph=graph,
        parameters={"num_steps": 2, "dt": 0.25},
    )

    result = _hub(engine)._execute_computation(request)

    assert result.success is True
    assert result.result_data["operation"] == "epi_diffusion"
    assert result.result_data["pressure_model"] == "epi_diffusion"
    assert engine.calls == 1


def test_hub_does_not_republish_unmeasured_fft_speedup_or_memory() -> None:
    class _FFTOrchestrator:
        @staticmethod
        def analyze_optimization_profile(graph, operation):
            return object()

        @staticmethod
        def select_optimal_strategy(profile):
            return OptimizationStrategy.SPECTRAL_FFT

        @staticmethod
        def execute_optimization(graph, operation, strategy, **parameters):
            from tnfr.dynamics.optimization_orchestrator import OptimizationResult

            return OptimizationResult(
                strategy_used=OptimizationStrategy.SPECTRAL_FFT,
                execution_time=0.1,
                speedup_factor=1.0,
                cache_hits=0,
                cache_misses=1,
                memory_used_mb=0.0,
                accuracy_preserved=True,
                details={
                    "performance_measurements": {
                        "throughput_steps_per_second": 25.0,
                        "speedup_factor": None,
                        "memory_used_mb": None,
                    },
                    "accuracy_verification": {
                        "basis": "nodal_equation_residual",
                        "passed": True,
                    },
                },
            )

    hub = object.__new__(TNFRComputationalHub)
    hub._engines = {
        EngineType.OPTIMIZATION_ORCHESTRATOR: _FFTOrchestrator()
    }
    request = ComputationRequest(
        engine_type=EngineType.OPTIMIZATION_ORCHESTRATOR,
        operation="epi_diffusion",
        graph=_graph(),
    )

    payload = hub._execute_optimization_orchestrator(request)

    assert payload["speedup_factor"] is None
    assert payload["memory_used_mb"] is None
    assert payload["accuracy_verification"] == {
        "basis": "nodal_equation_residual",
        "passed": True,
    }


def test_fft_strategy_is_not_offered_for_an_unrelated_dense_operation() -> None:
    orchestrator = object.__new__(TNFROptimizationOrchestrator)
    orchestrator.default_memory_budget = 512.0
    graph = nx.complete_graph(24)

    unrelated = orchestrator.analyze_optimization_profile(graph, "temporal_evolution")
    diffusion = orchestrator.analyze_optimization_profile(graph, "epi_diffusion")

    assert OptimizationStrategy.SPECTRAL_FFT not in unrelated.available_strategies
    assert OptimizationStrategy.SPECTRAL_FFT in diffusion.available_strategies


def test_hub_marks_failed_optimization_result_as_failure() -> None:
    from tnfr.dynamics.optimization_orchestrator import OptimizationResult

    class FailedOrchestrator:
        @staticmethod
        def analyze_optimization_profile(graph, operation):
            return object()

        @staticmethod
        def select_optimal_strategy(profile):
            return OptimizationStrategy.NODAL_VECTORIZED

        @staticmethod
        def execute_optimization(graph, operation, strategy, **parameters):
            return OptimizationResult(
                strategy_used=strategy,
                execution_time=0.0,
                speedup_factor=1.0,
                cache_hits=0,
                cache_misses=0,
                memory_used_mb=0.0,
                accuracy_preserved=False,
                details={"error": "boom"},
            )

    hub = object.__new__(TNFRComputationalHub)
    hub._engines = {
        EngineType.OPTIMIZATION_ORCHESTRATOR: FailedOrchestrator()
    }
    hub._engine_performance = {kind: [] for kind in EngineType}
    request = ComputationRequest(
        engine_type=EngineType.OPTIMIZATION_ORCHESTRATOR,
        operation="general",
        graph=_graph(),
    )

    result = hub._execute_computation(request)

    assert result.success is False
    assert "boom" in (result.error_message or "")


def test_advanced_fft_reserved_dispatch_parameter_is_not_forwarded() -> None:
    class Result:
        output_data = {"filtered": True}

    class AdvancedFFT:
        def spectral_filtering(self, graph, **parameters):
            assert "spectral_operation" not in parameters
            assert parameters == {"filter_type": "highpass"}
            return Result()

    hub = object.__new__(TNFRComputationalHub)
    hub._engines = {EngineType.ADVANCED_FFT: AdvancedFFT()}
    request = ComputationRequest(
        engine_type=EngineType.ADVANCED_FFT,
        operation="harmonic_analysis",
        graph=_graph(),
        parameters={
            "spectral_operation": "spectral_filtering",
            "filter_type": "highpass",
        },
    )

    assert hub._execute_advanced_fft(request) == {"filtered": True}


def test_adelic_hub_route_calls_declared_engine_method() -> None:
    class Adelic:
        @staticmethod
        def compute_geometric_trace(*, t):
            return t + 1.0

        compute_nodal_gradient = compute_geometric_trace
        run_resonance_search = compute_geometric_trace
        compute_structural_fields = compute_geometric_trace
        step = compute_geometric_trace

    hub = object.__new__(TNFRComputationalHub)
    hub._engines = {EngineType.ADELIC_DYNAMICS: Adelic()}
    request = ComputationRequest(
        engine_type=EngineType.ADELIC_DYNAMICS,
        operation="geometric_trace",
        graph=_graph(),
        parameters={"t": 2.0},
    )

    assert hub._execute_adelic_dynamics(request) == pytest.approx(3.0)


def test_cache_services_are_not_registered_as_execution_engines() -> None:
    hub = TNFRComputationalHub(max_workers=1)
    try:
        assert EngineType.STRUCTURAL_CACHE not in hub._engines
        assert EngineType.MULTI_MODAL_CACHE not in hub._engines
    finally:
        hub.shutdown()
