"""Canonical graph semantics for the unified GPU adapter."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.engines.computation import unified_gpu_system as gpu_module
from tnfr.engines.computation.unified_gpu_system import (
    GPUDeviceInfo,
    GPUOperationResult,
    TNFRUnifiedGPUSystem,
    UnifiedGPUConfig,
)
from tnfr.mathematics.unified_numerical import np


def test_graph_delta_nfr_preserves_directed_weighted_parallel_transport() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(0, EPI=0.0, nu_f=7.0, theta=0.0)
    graph.add_node(1, EPI=1.0, nu_f=2.0, theta=0.5)
    graph.add_node(2, EPI=0.0, nu_f=3.0, theta=1.0)
    graph.add_edge(0, 1, key="a", weight=2.0)
    graph.add_edge(0, 1, key="b", weight=1.0)
    graph.add_edge(0, 2, key="c", weight=1.0)
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine._operation_stats = {}

    result = engine.compute_delta_nfr_from_graph(graph)

    assert result == pytest.approx({0: 0.75, 1: 0.0, 2: 0.0})
    assert result.backend_used == "canonical-cpu"
    assert result.fallback_used is True
    assert engine._operation_stats == {
        "total_operations": 1,
        "cpu_fallbacks": 1,
    }
    assert engine._last_graph_delta_nfr_result is result


def test_graph_delta_nfr_two_node_orientation_matches_negative_laplacian() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, EPI=0.0, nu_f=1.0, theta=0.0)
    graph.add_node(1, EPI=1.0, nu_f=2.0, theta=0.4)
    graph.add_edge(0, 1, weight=3.0)
    engine = object.__new__(TNFRUnifiedGPUSystem)

    result = engine.compute_delta_nfr_from_graph(graph)

    assert result == pytest.approx({0: 1.0, 1: 0.0})


def test_dense_delta_nfr_is_normalized_epi_pressure_independent_of_capacity() -> None:
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine._operation_stats = {}

    result = engine.compute_delta_nfr_gpu(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([0.0, 1.0]),
        np.array([7.0, 2.0]),
        np.array([1.2, -0.7]),
    )

    assert result.result_data == pytest.approx([1.0, -1.0])
    assert result.backend_used == "canonical-cpu"
    assert result.fallback_used is True
    assert result.operation_metadata["nu_f_applied"] is False
    assert result.operation_metadata["phase_applied"] is False
    assert result.gpu_utilization is None
    assert result.convergence_achieved is None


def test_array_only_structural_tetrad_is_explicitly_rejected() -> None:
    engine = object.__new__(TNFRUnifiedGPUSystem)

    with pytest.raises(NotImplementedError, match="source graph"):
        engine.compute_structural_fields(np.zeros((2, 2)))


def test_gpu_result_defaults_unobserved_diagnostics_to_none() -> None:
    result = GPUOperationResult(
        result_data=np.zeros(1),
        backend_used="test",
        computation_time_ms=0.0,
        memory_usage_mb=0.0,
    )

    assert result.gpu_utilization is None
    assert result.convergence_achieved is None


def _fallback_engine(*, enabled: bool) -> TNFRUnifiedGPUSystem:
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine.config = UnifiedGPUConfig(enable_cpu_fallback=enabled)
    engine._operation_stats = {"cpu_fallbacks": 0}
    return engine


def test_dense_inputs_are_normalized_once_for_compute_and_memory() -> None:
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine._operation_stats = {}
    measured = []

    def measure(*arrays):
        measured.extend(arrays)
        assert all(isinstance(array, np.ndarray) for array in arrays)
        assert all(array.dtype == np.dtype(float) for array in arrays)
        return sum(array.nbytes for array in arrays) / (1024 * 1024)

    engine._estimate_memory_usage = measure
    result = engine.compute_delta_nfr_gpu(
        [[0, 2], [4, 0]],
        [0, 1],
        [1, 2],
        [0, -0.5],
    )

    assert result.result_data == pytest.approx([1.0, -1.0])
    assert len(measured) == 4
    assert result.memory_usage_mb == pytest.approx(80 / (1024 * 1024))


@pytest.mark.parametrize(
    ("field", "replacement", "error_type", "message"),
    [
        ("adjacency", [[0.0, True], [1.0, 0.0]], TypeError, "adjacency"),
        ("epi", [0.0, complex(1.0, 1.0)], TypeError, "EPI"),
        ("vf", [1.0, -0.1], ValueError, "nonnegative"),
        ("phase", [0.0, float("inf")], ValueError, "phase"),
    ],
)
def test_dense_inputs_reject_invalid_nodal_evidence(
    field, replacement, error_type, message
) -> None:
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine._operation_stats = {}
    values = {
        "adjacency": [[0.0, 1.0], [1.0, 0.0]],
        "epi": [0.0, 1.0],
        "vf": [1.0, 2.0],
        "phase": [0.0, 0.5],
    }
    values[field] = replacement

    with pytest.raises(error_type, match=message):
        engine.compute_delta_nfr_gpu(
            values["adjacency"],
            values["epi"],
            values["vf"],
            values["phase"],
        )


def test_execute_with_fallback_uses_a_distinct_declared_cpu_callable() -> None:
    engine = _fallback_engine(enabled=True)
    calls = []

    def gpu(values):
        calls.append("gpu")
        raise RuntimeError("device failure")

    def cpu(values):
        calls.append("cpu")
        return np.asarray(values, dtype=float) + 1.0

    result = engine.execute_with_fallback(gpu, [1.0, 2.0], cpu_fallback=cpu)

    assert calls == ["gpu", "cpu"]
    assert result.result_data == pytest.approx([2.0, 3.0])
    assert result.backend_used == "cpu_fallback"
    assert result.fallback_used is True
    assert result.gpu_utilization is None
    assert result.convergence_achieved is None
    assert engine._operation_stats["cpu_fallbacks"] == 1


def test_execute_with_fallback_preserves_cpu_result_provenance() -> None:
    engine = _fallback_engine(enabled=True)

    def gpu():
        raise RuntimeError("device failure")

    cpu_result = GPUOperationResult(
        result_data=np.asarray([3.0]),
        backend_used="canonical-cpu",
        computation_time_ms=0.1,
        memory_usage_mb=0.01,
        gpu_utilization=37.5,
        convergence_achieved=True,
    )

    result = engine.execute_with_fallback(
        gpu,
        cpu_fallback=lambda: cpu_result,
    )

    assert result is not cpu_result
    assert result.backend_used == "canonical-cpu"
    assert result.fallback_used is True
    assert result.gpu_utilization == pytest.approx(37.5)
    assert result.convergence_achieved is True
    assert result.operation_metadata["execution"] == "declared_cpu_fallback"


def test_execute_with_fallback_does_not_repeat_a_failed_callable() -> None:
    engine = _fallback_engine(enabled=True)
    calls = 0

    def failing():
        nonlocal calls
        calls += 1
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        engine.execute_with_fallback(failing)

    assert calls == 1
    assert engine._operation_stats["cpu_fallbacks"] == 0


def test_execute_with_fallback_obeys_disabled_cpu_policy() -> None:
    engine = _fallback_engine(enabled=False)
    calls = []

    def gpu():
        calls.append("gpu")
        raise RuntimeError("device failure")

    def cpu():
        calls.append("cpu")
        return np.zeros(1)

    with pytest.raises(RuntimeError, match="device failure"):
        engine.execute_with_fallback(gpu, cpu_fallback=cpu)

    assert calls == ["gpu"]
    assert engine._operation_stats["cpu_fallbacks"] == 0


def test_execute_with_fallback_rejects_same_callable_for_both_backends() -> None:
    engine = _fallback_engine(enabled=True)

    def operation():
        return np.zeros(1)

    with pytest.raises(ValueError, match="must be distinct"):
        engine.execute_with_fallback(operation, cpu_fallback=operation)


def test_compatibility_fallback_obeys_disabled_cpu_policy() -> None:
    engine = _fallback_engine(enabled=False)
    cpu_called = False

    def gpu():
        raise RuntimeError("device failure")

    def cpu():
        nonlocal cpu_called
        cpu_called = True
        return "cpu"

    with pytest.raises(RuntimeError, match="device failure"):
        engine.execute_with_gpu_fallback(gpu, cpu)

    assert cpu_called is False


def test_module_fallback_wrapper_uses_the_configured_system_policy(
    monkeypatch,
) -> None:
    engine = _fallback_engine(enabled=False)
    monkeypatch.setattr(gpu_module, "_unified_gpu_system", engine)
    cpu_called = False

    def gpu():
        raise RuntimeError("device failure")

    def cpu():
        nonlocal cpu_called
        cpu_called = True
        return "cpu"

    with pytest.raises(RuntimeError, match="device failure"):
        gpu_module.execute_with_gpu_fallback(gpu, cpu)

    assert cpu_called is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (False, None),
        (float("nan"), None),
        (-1.0, None),
        (101.0, None),
        ("42.5", 42.5),
    ],
)
def test_gpu_utilization_requires_valid_backend_evidence(raw, expected) -> None:
    observed = TNFRUnifiedGPUSystem._read_gpu_utilization(
        {"utilization_percent": raw}
    )

    if expected is None:
        assert observed is None
    else:
        assert observed == pytest.approx(expected)


def test_compute_optimal_prefers_observed_utilization_over_unknown() -> None:
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine.config = UnifiedGPUConfig(device_selection_strategy="compute_optimal")
    engine._available_devices = [
        GPUDeviceInfo(0, "unknown", "test", 10.0, 10.0, None),
        GPUDeviceInfo(1, "observed", "test", 10.0, 10.0, 25.0),
    ]

    selected = engine._select_optimal_device()

    assert selected is engine._available_devices[1]


def test_current_gpu_utilization_is_none_without_observation() -> None:
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine._current_device = None

    assert engine._get_current_gpu_utilization() is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (False, None),
        (float("nan"), None),
        (float("inf"), None),
        (-1.0, None),
        (0.0, 0.0),
        ("512.5", 512.5),
    ],
)
def test_gpu_memory_requires_finite_nonnegative_backend_evidence(
    raw, expected
) -> None:
    observed = TNFRUnifiedGPUSystem._read_gpu_memory_mb(
        {"free_memory_mb": raw}, "free_memory_mb"
    )

    if expected is None:
        assert observed is None
    else:
        assert observed == pytest.approx(expected)


def test_memory_optimal_prefers_observed_zero_over_unknown() -> None:
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine.config = UnifiedGPUConfig(device_selection_strategy="memory_optimal")
    engine._available_devices = [
        GPUDeviceInfo(0, "unknown", "test", None, None, None),
        GPUDeviceInfo(1, "observed-zero", "test", 0.0, 0.0, None),
    ]

    selected = engine._select_optimal_device()

    assert selected is engine._available_devices[1]


def test_gpu_feasibility_abstains_when_free_memory_is_unknown() -> None:
    engine = object.__new__(TNFRUnifiedGPUSystem)
    engine.config = UnifiedGPUConfig(enable_gpu_acceleration=True)
    unknown = GPUDeviceInfo(0, "unknown", "test", None, None, None)
    engine._available_devices = [unknown]
    engine._current_device = unknown

    assert engine._can_handle_gpu_operation(np.zeros(0)) is False

    observed_zero = GPUDeviceInfo(
        1, "observed-zero", "test", 0.0, 0.0, None
    )
    engine._available_devices = [observed_zero]
    engine._current_device = observed_zero
    assert engine._can_handle_gpu_operation(np.zeros(0)) is True
