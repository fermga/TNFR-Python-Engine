"""Evidence tests for sequence-FFT and graph-spectral contracts."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.advanced_fft_arithmetic import TNFRAdvancedFFTEngine
from tnfr.dynamics.fft_cache_coordinator import FFTCacheCoordinator
from tnfr.dynamics.fft_engine import FFTDynamicsEngine
from tnfr.engines.computation.unified_fft_engine import (
    TNFRUnifiedFFTEngine,
    UnifiedFFTConfig,
)
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.mathematics.spectral import igft
from tnfr.types import ensure_bepi, serialize_bepi


def _basic_engine(**overrides: object) -> TNFRUnifiedFFTEngine:
    options = {
        "auto_backend_selection": False,
        "preferred_backend": "basic",
        "enable_gpu_acceleration": False,
        "log_backend_selection": False,
    }
    options.update(overrides)
    return TNFRUnifiedFFTEngine(UnifiedFFTConfig(**options))


def test_graph_gft_reads_alias_and_preserves_uniform_bepi_sign() -> None:
    graph = nx.path_graph(2)
    graph.nodes[0]["psi"] = serialize_bepi(ensure_bepi(-0.75))
    graph.nodes[1]["EPI"] = 0.25
    for node in graph:
        graph.nodes[node]["nu_f"] = 1.0

    state = TNFRAdvancedFFTEngine(
        cache_coordinator=FFTCacheCoordinator()
    ).get_spectral_state(graph)

    np.testing.assert_allclose(
        igft(state.spectral_coeffs, state.eigenvectors),
        np.array([-0.75, 0.25]),
        atol=1e-14,
    )


def test_graph_gft_rejects_rich_epi_without_signed_scalar_chart() -> None:
    graph = nx.path_graph(2)
    graph.nodes[0]["EPI"] = BEPIElement(
        (-0.8, -0.7), (-0.8, -0.8), (0.0, 1.0)
    )
    graph.nodes[1]["EPI"] = 0.25

    with pytest.raises(TNFRValueError, match="uniform-real EPI"):
        TNFRAdvancedFFTEngine(
            cache_coordinator=FFTCacheCoordinator()
        ).get_spectral_state(graph)


def test_spectral_filter_cache_keeps_full_cutoff_precision() -> None:
    graph = nx.path_graph(3)
    for node, epi in enumerate((1.0, -0.5, 0.25)):
        graph.nodes[node]["EPI"] = epi
    engine = TNFRAdvancedFFTEngine(cache_coordinator=FFTCacheCoordinator())

    first = engine.spectral_filtering(graph, cutoff_frequency=0.12345641)
    second = engine.spectral_filtering(graph, cutoff_frequency=0.12345649)

    assert not np.array_equal(
        first.output_data["filter_response"],
        second.output_data["filter_response"],
    )


def test_sequence_fft_applies_selected_precision_and_reports_no_convergence() -> None:
    engine = _basic_engine(spectral_precision="float32")
    result = engine.compute_fft(np.array([1.0, -2.0, 0.5], dtype=np.float64))

    assert result.spectral_data.dtype == np.dtype("complex64")
    assert result.frequencies.dtype == np.dtype("float32")
    assert result.spectral_precision == "float32"
    assert result.convergence_achieved is None
    assert result.operation_metadata["effective_precision"] == "float32"
    assert result.operation_metadata["convergence_applicable"] is False


def test_normalized_cross_power_is_scale_invariant_without_absolute_epsilon() -> None:
    signal = np.array([1.0, 2.0, 4.0, 8.0])
    engine = _basic_engine()

    reference = engine.compute_cross_spectral_coherence(signal, signal)
    rescaled = engine.compute_cross_spectral_coherence(
        signal * 1.0e-9, signal * 3.0e-9
    )

    np.testing.assert_allclose(reference.spectral_data, np.ones(4))
    np.testing.assert_allclose(rescaled.spectral_data, reference.spectral_data)
    assert (
        reference.operation_metadata["reduced_semantics"]
        == "common_nonzero_spectral_support"
    )

    zero = engine.compute_cross_spectral_coherence(
        np.zeros_like(signal), signal
    )
    np.testing.assert_array_equal(zero.spectral_data, np.zeros(4))


def test_sequence_cache_key_is_injective_for_structurally_distinct_kwargs() -> None:
    engine = _basic_engine()
    signal = np.array([1.0, 2.0, 3.0])

    engine.compute_fft(signal, a="b,c:d")
    engine.compute_fft(signal, a="b", c="d")

    stats = engine.get_cache_statistics()
    assert stats["misses"] == 2
    assert stats["hits"] == 0
    assert stats["cache_size"] == 2


def test_sequence_cache_enforces_its_declared_byte_budget() -> None:
    engine = _basic_engine(max_cache_size_mb=0.001)
    signal = np.linspace(0.0, 1.0, 256)

    first = engine.compute_fft(signal)
    second = engine.compute_fft(signal)
    stats = engine.get_cache_statistics()

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert stats["oversize_skips"] == 2
    assert stats["cache_size"] == 0
    assert stats["cache_memory_estimate_mb"] <= stats["cache_limit_mb"]


@pytest.mark.parametrize(
    "values",
    (
        np.array([True, False]),
        np.array([1.0, np.nan]),
        np.array(["1", "2"]),
    ),
)
def test_sequence_fft_rejects_nonfinite_or_nonnumeric_inputs(
    values: np.ndarray,
) -> None:
    with pytest.raises(TNFRValueError, match="finite numeric scalars"):
        _basic_engine().compute_fft(values)


def test_graph_fft_euler_step_exposes_uncertified_stability_scope() -> None:
    graph = nx.path_graph(2)
    for node, epi in enumerate((0.0, 1.0)):
        graph.nodes[node].update(EPI=epi, nu_f=1.0, theta=0.0)

    result = FFTDynamicsEngine().run_fft_simulation(graph, num_steps=1, dt=0.1)

    assert result["integration_method"] == "forward_euler"
    assert result["stability_not_certified"] is True
    assert (
        result["stability_evidence"]
        == "arbitrary_dt_without_step_size_certificate"
    )
