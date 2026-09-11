"""Evidence and failure contracts for the optimization orchestrator."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.optimization_orchestrator import (
    OptimizationResult,
    OptimizationStrategy,
    TNFROptimizationOrchestrator,
)


def _graph(size: int = 2) -> nx.Graph:
    graph = nx.path_graph(size)
    for node in graph:
        graph.nodes[node].update(
            EPI=float(node),
            nu_f=1.0,
            theta=0.0,
            DeltaNFR=0.0,
        )
    graph.graph["payload"] = {"nested": [1, 2]}
    return graph


def _bare(**engines) -> TNFROptimizationOrchestrator:
    orchestrator = object.__new__(TNFROptimizationOrchestrator)
    orchestrator.default_memory_budget = 512.0
    orchestrator.nodal_optimizer = engines.get("nodal_optimizer")
    orchestrator.structural_cache = engines.get("structural_cache")
    orchestrator.fft_engine = engines.get("fft_engine")
    orchestrator.adelic_engine = engines.get("adelic_engine")
    orchestrator.optimization_history = []
    orchestrator.strategy_performance = {}
    orchestrator.global_cache = None
    return orchestrator


class _ProposalMapping(dict):
    metadata = {
        "stability_not_certified": True,
        "stability_evidence": "arbitrary_dt_without_step_size_certificate",
    }


class _NodalProposal:
    def __init__(self, *, perturbation: float = 0.0) -> None:
        self.hits = 3
        self.misses = 2
        self.perturbation = perturbation

    def get_optimization_stats(self):
        return {"cache_hits": self.hits, "cache_misses": self.misses}

    def compute_vectorized_nodal_evolution(self, graph, dt):
        self.hits += 1
        return _ProposalMapping(
            {
                0: (dt + self.perturbation, 0.0),
                1: (1.0 - dt, 0.0),
            }
        )


def test_nodal_route_reports_residual_and_only_counter_deltas() -> None:
    orchestrator = _bare(nodal_optimizer=_NodalProposal())

    result = orchestrator.execute_optimization(
        _graph(),
        "nodal_evolution",
        OptimizationStrategy.NODAL_VECTORIZED,
        dt=0.1,
        nodal_residual_tolerance=1e-12,
    )

    assert result.accuracy_preserved is True
    assert result.cache_hits == 1
    assert result.cache_misses == 0
    assert result.details["cache_measurements"] == {
        "hits": 1,
        "misses": 0,
        "scope": "operation_delta",
    }
    assert result.details["performance_measurements"]["speedup_factor"] is None
    assert result.details["performance_measurements"]["memory_used_mb"] is None
    assert result.details["accuracy_verification"]["max_nodal_residual"] < 1e-12
    assert result.details["state_committed"] is False
    assert result.details["stability_not_certified"] is True
    assert (
        result.details["proposal_metadata"]["stability_evidence"]
        == "arbitrary_dt_without_step_size_certificate"
    )
    assert orchestrator.strategy_performance == {}


def test_nodal_route_rejects_a_finite_but_incorrect_proposal() -> None:
    orchestrator = _bare(nodal_optimizer=_NodalProposal(perturbation=0.05))

    result = orchestrator.execute_optimization(
        _graph(),
        "nodal_evolution",
        OptimizationStrategy.NODAL_VECTORIZED,
        dt=0.1,
        nodal_residual_tolerance=1e-12,
    )

    assert result.accuracy_preserved is False
    assert result.details["accuracy_verification"]["passed"] is False
    assert result.details["accuracy_verification"]["max_nodal_residual"] > 0.1
    assert orchestrator.strategy_performance == {}


def _entry(*, xi_c: float = 0.0):
    return SimpleNamespace(
        phi_s={0: 0.0},
        grad_phi={0: 0.0},
        k_phi={0: 0.0},
        xi_c=xi_c,
        coherence=0.0,
        phase_sync=0.0,
        timestamp=0.0,
        topology_hash="state",
        spectral_basis_signature="basis",
        eigenvalues=np.zeros(1),
        eigenvectors=np.eye(1),
        coordination_nodes=[],
    )


class _StructuralCache:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.hits = 10
        self.misses = 4
        self.mismatch = mismatch

    def get_cache_stats(self):
        return {"hits": self.hits, "misses": self.misses}

    def get_structural_fields(self, graph, force_recompute=False):
        if force_recompute:
            self.misses += 1
            return _entry()
        self.hits += 1
        return _entry(xi_c=1.0 if self.mismatch else 0.0)


def test_zero_valued_structural_fields_are_verified_by_entry_equivalence() -> None:
    orchestrator = _bare(structural_cache=_StructuralCache())

    result = orchestrator.execute_optimization(
        _graph(1),
        "structural_fields",
        OptimizationStrategy.STRUCTURAL_MEMO,
    )

    assert result.accuracy_preserved is True
    assert result.cache_hits == 1
    assert result.cache_misses == 1
    assert result.details["accuracy_verification"]["entries_equal"] is True
    assert result.details["performance_measurements"] == {
        "speedup_factor": None,
        "memory_used_mb": None,
    }


def test_structural_cache_mismatch_is_a_failed_verification() -> None:
    orchestrator = _bare(structural_cache=_StructuralCache(mismatch=True))

    result = orchestrator.execute_optimization(
        _graph(1),
        "structural_fields",
        OptimizationStrategy.STRUCTURAL_MEMO,
    )

    assert result.accuracy_preserved is False
    assert result.details["accuracy_verification"]["entries_equal"] is False
    assert result.details["accuracy_verification"]["passed"] is False


class _AdelicTrace:
    def __init__(self) -> None:
        self.primes = np.asarray([2.0, 3.0])
        self.nu_f = np.log(self.primes)
        self._trace_cache = None

    def precompute_trace_landscape(self, start, end, resolution):
        times = np.linspace(start, end, resolution)
        self._trace_cache = (times, np.zeros_like(times))

    def compute_geometric_trace(self, time):
        weights = self.nu_f / np.sqrt(self.primes)
        return float(
            abs(np.sum(weights * np.exp(1j * float(time) * self.nu_f)))
        )


def test_adelic_route_measures_grid_storage_and_verifies_against_formula() -> None:
    orchestrator = _bare(adelic_engine=_AdelicTrace())

    result = orchestrator.execute_optimization(
        None,
        "trace",
        OptimizationStrategy.ADELIC_CACHE,
        landscape_resolution=17,
        verification_samples=9,
        adelic_interpolation_tolerance=1e-14,
    )

    measurements = result.details["performance_measurements"]
    assert result.accuracy_preserved is True
    assert result.speedup_factor == 1.0
    assert measurements["speedup_factor"] is None
    assert measurements["memory_used_mb"] > 0.0
    assert measurements["memory_scope"] == "managed_trace_grid_only"
    assert result.details["cache_measurements"]["hits"] is None
    assert result.details["accuracy_verification"]["max_absolute_residual"] < 1e-14


class _ResidualFFTEngine:
    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.calls = 0

    def run_fft_simulation(self, graph, num_steps, dt):
        self.calls += 1
        graph.nodes[0]["EPI"] = 99.0
        graph.graph["payload"]["nested"].append(3)
        if self.raises:
            raise RuntimeError("fft failed")
        return {
            "status": "success",
            "simulation_time": 0.01,
            "cache_hits": 0,
            "steps_per_second": 100.0,
            "max_nodal_residual": 1.0,
        }


@pytest.mark.parametrize("raises", (False, True))
def test_failed_fft_execution_restores_the_complete_graph(raises: bool) -> None:
    graph = _graph()
    before = deepcopy(graph.nodes[0]), deepcopy(graph.graph)
    orchestrator = _bare(fft_engine=_ResidualFFTEngine(raises=raises))

    result = orchestrator.execute_optimization(
        graph,
        "epi_diffusion",
        OptimizationStrategy.SPECTRAL_FFT,
        nodal_residual_tolerance=1e-6,
    )

    assert result.accuracy_preserved is False
    assert dict(graph.nodes[0]) == before[0]
    assert dict(graph.graph) == before[1]
    if raises:
        assert "fft failed" in result.details["error"]
    else:
        assert result.details["transaction_committed"] is False


def test_failed_hybrid_component_stops_before_mutating_fft_stage() -> None:
    fft = _ResidualFFTEngine()
    orchestrator = _bare(
        structural_cache=_StructuralCache(mismatch=True),
        fft_engine=fft,
    )

    result = orchestrator.execute_optimization(
        nx.complete_graph(21),
        "epi_diffusion",
        OptimizationStrategy.HYBRID,
    )

    assert result.accuracy_preserved is False
    assert fft.calls == 0
    assert result.details["combined_strategies"] == ["struct_memo"]


def test_history_learns_only_explicit_measured_speedup() -> None:
    orchestrator = _bare()
    neutral = OptimizationResult(
        strategy_used=OptimizationStrategy.NODAL_VECTORIZED,
        execution_time=0.1,
        speedup_factor=1.0,
        cache_hits=0,
        cache_misses=0,
        memory_used_mb=0.0,
        accuracy_preserved=True,
        details={
            "performance_measurements": {
                "speedup_factor": None,
                "memory_used_mb": None,
            }
        },
    )
    measured = OptimizationResult(
        strategy_used=OptimizationStrategy.NODAL_VECTORIZED,
        execution_time=0.1,
        speedup_factor=1.0,
        cache_hits=0,
        cache_misses=0,
        memory_used_mb=0.0,
        accuracy_preserved=True,
        details={
            "performance_measurements": {
                "speedup_factor": 2.0,
                "memory_used_mb": None,
            }
        },
    )

    orchestrator._update_performance_history(neutral)
    assert orchestrator.get_orchestrator_stats()["recent_avg_speedup"] is None
    orchestrator._update_performance_history(measured)

    stats = orchestrator.get_orchestrator_stats()
    assert stats["recent_avg_speedup"] == pytest.approx(2.0)
    assert stats["recent_measured_speedup_samples"] == 1
    assert stats["strategy_performance"]["nodal_vec"] == {
        "avg_speedup": 2.0,
        "max_speedup": 2.0,
        "measured_operations": 1,
    }


def test_direct_auto_execution_records_only_the_resolved_strategy_once() -> None:
    orchestrator = _bare(nodal_optimizer=_NodalProposal())
    orchestrator.analyze_optimization_profile = lambda graph, operation: object()
    orchestrator.select_optimal_strategy = (
        lambda profile: OptimizationStrategy.NODAL_VECTORIZED
    )

    result = orchestrator.execute_optimization(
        _graph(),
        "nodal_evolution",
        OptimizationStrategy.AUTO,
        dt=0.1,
    )

    assert result.strategy_used is OptimizationStrategy.NODAL_VECTORIZED
    assert len(orchestrator.optimization_history) == 1
