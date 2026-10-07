"""Mathematical contracts for graph-spectral and sequence-FFT APIs."""

from __future__ import annotations

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

from tnfr.dynamics.advanced_cache_optimizer import CacheOptimizationStrategy
from tnfr.dynamics.advanced_fft_arithmetic import TNFRAdvancedFFTEngine
from tnfr.dynamics.cache_aware_fft_engine import TNFRCacheAwareFFTEngine
from tnfr.engines.computation.unified_fft_engine import (
    TNFRUnifiedFFTEngine,
    UnifiedFFTConfig,
)
from tnfr.errors import TNFRValueError
from tnfr.mathematics.spectral import get_laplacian_spectrum, gft, heat_diffusion, igft
from tnfr.physics.structural_diffusion import structural_diffusion_operator


def _irregular_weighted_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from(((0, 1, 2.0), (1, 2, 0.5), (1, 3, 3.0), (3, 4, 0.75)))
    for node, epi in enumerate((1.2, -0.3, 0.8, -1.1, 0.4)):
        graph.nodes[node]["EPI"] = epi
        graph.nodes[node]["nu_f"] = 0.5 + 0.1 * node
    return graph


def test_random_walk_gft_uses_biorthogonal_inverse() -> None:
    graph = _irregular_weighted_graph()
    eigenvalues, right_basis = get_laplacian_spectrum(graph, operator="random_walk")
    signal = np.array((1.2, -0.3, 0.8, -1.1, 0.4))
    _, laplacian = structural_diffusion_operator(graph)

    assert not np.allclose(
        right_basis.conj().T @ right_basis, np.eye(len(graph)), atol=1e-10
    )
    assert np.allclose(igft(gft(signal, right_basis), right_basis), signal)

    time = 0.23
    expected = scipy.linalg.expm(-time * laplacian) @ signal
    actual = heat_diffusion(signal, right_basis, eigenvalues, time)
    assert np.allclose(actual, expected, atol=2e-14, rtol=2e-14)


def test_partial_random_walk_spectrum_never_calls_hermitian_eigsh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _irregular_weighted_graph()

    def forbidden_eigsh(*args: object, **kwargs: object) -> object:
        raise AssertionError("eigsh cannot diagonalize L_rw")

    monkeypatch.setattr(scipy.sparse.linalg, "eigsh", forbidden_eigsh)
    eigenvalues, right_modes = get_laplacian_spectrum(
        graph, operator="random_walk", k=2
    )
    _, laplacian = structural_diffusion_operator(graph)

    assert eigenvalues.shape == (2,)
    assert right_modes.shape == (len(graph), 2)
    assert np.allclose(
        laplacian @ right_modes,
        right_modes * eigenvalues[np.newaxis, :],
        atol=2e-13,
    )
    with pytest.raises(TNFRValueError, match="partial graph spectral basis"):
        gft(np.ones(len(graph)), right_modes)


def test_partial_symmetric_spectrum_honors_n_minus_one_mode_request() -> None:
    graph = _irregular_weighted_graph()
    eigenvalues, modes = get_laplacian_spectrum(graph, k=len(graph) - 1)
    signal = np.array((1.2, -0.3, 0.8, -1.1, 0.4))

    assert eigenvalues.shape == (len(graph) - 1,)
    assert modes.shape == (len(graph), len(graph) - 1)
    assert np.all(np.diff(eigenvalues) >= 0.0)
    assert np.allclose(modes.T @ modes, np.eye(len(graph) - 1))
    assert np.allclose(
        heat_diffusion(signal, modes, eigenvalues, 0.0), modes @ modes.T @ signal
    )


def test_gft_uses_conjugate_projection_for_complex_orthonormal_basis() -> None:
    basis = np.array(((1.0, 1.0), (1.0j, -1.0j))) / np.sqrt(2.0)
    signal = np.array((2.0 + 3.0j, -1.0 + 0.5j))

    coefficients = gft(signal, basis)

    assert np.allclose(coefficients, basis.conj().T @ signal)
    assert np.allclose(igft(coefficients, basis), signal)


def test_graph_spectral_convolution_is_not_a_pointwise_nodal_product() -> None:
    graph = _irregular_weighted_graph()
    engine = TNFRAdvancedFFTEngine()
    first = np.array((1.0, -0.2, 0.4, 1.5, -0.7))
    second = np.array((0.3, 2.0, -1.0, 0.2, 0.8))
    state = engine.get_spectral_state(graph)

    result = engine.spectral_convolution(graph, first, second)
    expected = igft(
        gft(first, state.eigenvectors) * gft(second, state.eigenvectors),
        state.eigenvectors,
    )

    assert np.allclose(result.output_data, expected)
    assert not np.allclose(result.output_data, first * second)
    assert result.backend_used == "numpy_dense_gft"
    assert result.accuracy_metrics["nodal_product_equivalent"] == 0.0


def test_advanced_spectral_state_refreshes_live_epi_coefficients() -> None:
    graph = _irregular_weighted_graph()
    engine = TNFRAdvancedFFTEngine()
    first = engine.get_spectral_state(graph)
    assert first.spectral_parameter_kind == "laplacian_eigenvalue"
    assert (
        first.coherence_length_semantics == "inverse_root_spectral_centroid_heuristic"
    )
    graph.nodes[0]["EPI"] = -4.0

    second = engine.get_spectral_state(graph)
    reconstructed = igft(second.spectral_coeffs, second.eigenvectors)

    assert not np.array_equal(first.spectral_coeffs, second.spectral_coeffs)
    assert reconstructed[0] == pytest.approx(-4.0)


@pytest.mark.parametrize("requested", ("advanced", "distributed"))
def test_sequence_facade_records_fallback_from_graph_only_backend(
    requested: str,
) -> None:
    engine = TNFRUnifiedFFTEngine(
        UnifiedFFTConfig(
            auto_backend_selection=False,
            preferred_backend=requested,
            enable_gpu_acceleration=False,
            log_backend_selection=False,
        )
    )
    data = np.array((1.0, -2.0, 0.5, 3.0))

    result = engine.compute_fft(data)

    assert np.allclose(result.spectral_data, np.fft.fft(data))
    assert result.backend_used == "basic"
    assert result.operation_metadata["requested_backend"] == requested
    assert result.operation_metadata["actual_backend"] == "basic"
    assert result.operation_metadata["backend_fallback"] is True
    assert result.operation_metadata["fallback_from"] == requested


def test_sequence_convolution_has_circular_semantics_and_honest_backend() -> None:
    engine = TNFRUnifiedFFTEngine(
        UnifiedFFTConfig(enable_gpu_acceleration=False, log_backend_selection=False)
    )
    first = np.array((1.0, 2.0, 0.0, 0.0))
    second = np.array((1.0, -1.0, 0.0, 0.0))

    result = engine.compute_spectral_convolution(first, second, backend="distributed")
    expected = np.real_if_close(np.fft.ifft(np.fft.fft(first) * np.fft.fft(second)))

    assert np.allclose(result.spectral_data, expected)
    assert result.backend_used == "basic"
    assert result.operation_metadata["semantics"] == "circular_sequence_convolution"
    assert result.operation_metadata["fallback_from"] == "distributed"


def test_cross_power_diagnostic_reports_the_backend_that_computed_inputs() -> None:
    engine = TNFRUnifiedFFTEngine(
        UnifiedFFTConfig(
            auto_backend_selection=False,
            preferred_backend="advanced",
            enable_gpu_acceleration=False,
            log_backend_selection=False,
        )
    )

    result = engine.compute_cross_spectral_coherence(
        np.array((1.0, 0.0, -1.0, 0.0)),
        np.array((0.0, 1.0, 0.0, -1.0)),
    )

    assert result.backend_used == "basic"
    assert result.operation_metadata["operation"] == "modewise_normalized_cross_power"
    assert result.operation_metadata["input_backends"] == ["basic", "basic"]


def test_harmonic_compatibility_api_reports_laplacian_mode_semantics() -> None:
    graph = _irregular_weighted_graph()
    engine = TNFRAdvancedFFTEngine()

    result = engine.harmonic_analysis(graph, num_harmonics=3, window_size=7)
    data = result.output_data

    assert data["metric_semantics"] == "laplacian_mode_amplitude_ranking"
    assert data["window_size_applied"] is False
    assert data["requested_window_size"] == 7
    assert data["dominant_mode_index"] == int(
        np.argmax(result.spectral_state.amplitudes)
    )
    assert data["harmonic_amplitudes"][0] == pytest.approx(
        np.max(result.spectral_state.amplitudes)
    )


def test_cross_spectral_graph_diagnostic_rejects_incompatible_bases() -> None:
    first = _irregular_weighted_graph()
    second = _irregular_weighted_graph()
    second.add_edge(0, 4, weight=4.0)
    engine = TNFRAdvancedFFTEngine()

    with pytest.raises(TNFRValueError, match="matching Laplacian spectra"):
        engine.cross_spectral_coherence(first, second)


def test_cross_spectral_graph_diagnostic_uses_band_averaging() -> None:
    first = _irregular_weighted_graph()
    second = _irregular_weighted_graph()
    for node in second:
        second.nodes[node]["EPI"] = float(node - 2)
    engine = TNFRAdvancedFFTEngine()

    result = engine.cross_spectral_coherence(first, second, frequency_bands=2)
    data = result.output_data

    assert data["metric_semantics"] == "band_averaged_snapshot_cross_power"
    assert np.all(data["coherence_spectrum"] >= 0.0)
    assert np.all(data["coherence_spectrum"] <= 1.0)
    assert len(data["band_coherence"]) == 2


@pytest.mark.parametrize("invalid_k", (0, -1, 1.5, True))
def test_spectrum_rejects_invalid_partial_mode_counts(invalid_k: object) -> None:
    with pytest.raises(TNFRValueError, match="k must be a positive integer"):
        get_laplacian_spectrum(_irregular_weighted_graph(), k=invalid_k)


@pytest.mark.parametrize(
    "invalid_time",
    (
        -0.1,
        float("inf"),
        float("nan"),
        True,
        "1",
        Fraction(1, 10**400),
        Fraction(-1, 10**400),
    ),
)
def test_heat_diffusion_rejects_nonphysical_time(invalid_time: object) -> None:
    graph = _irregular_weighted_graph()
    eigenvalues, basis = get_laplacian_spectrum(graph)
    with pytest.raises(TNFRValueError, match="time must be finite and nonnegative"):
        heat_diffusion(np.ones(len(graph)), basis, eigenvalues, invalid_time)


def test_sequence_fft_cache_preserves_result_provenance_and_keys_dtype() -> None:
    engine = TNFRUnifiedFFTEngine(
        UnifiedFFTConfig(enable_gpu_acceleration=False, log_backend_selection=False)
    )
    float_data = np.array((1.0,), dtype=np.float64)
    same_bytes_as_integer = np.array((float_data.view(np.int64)[0],), dtype=np.int64)

    first = engine.compute_fft(float_data, backend="basic")
    second = engine.compute_fft(float_data, backend="basic")
    integer_result = engine.compute_fft(same_bytes_as_integer, backend="basic")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert integer_result.cache_hit is False
    assert engine.get_cache_statistics()["misses"] == 2


def test_highpass_and_notch_responses_preserve_the_named_bands() -> None:
    engine = TNFRAdvancedFFTEngine()
    parameters = np.linspace(0.0, 2.0, 9)

    highpass = engine._build_filter_response(parameters, "highpass", 1.0, 4)
    notch = engine._build_filter_response(parameters, "notch", 1.0, 4)

    assert highpass[0] == pytest.approx(0.0)
    assert highpass[-1] > highpass[5] > highpass[4]
    assert notch[4] == pytest.approx(0.0)
    assert notch[0] > 0.99
    assert notch[-1] > 0.99
    assert engine._calculate_attenuation_db(np.ones(3)) == pytest.approx(0.0)
    assert engine._calculate_attenuation_db(np.array((1.0, 0.1))) == pytest.approx(20.0)


class _CapturingCacheOptimizer:
    def __init__(self) -> None:
        self.strategies: list[CacheOptimizationStrategy] = []

    def optimize_cache_strategy(
        self, graph: nx.Graph, strategies: list[CacheOptimizationStrategy]
    ) -> list[object]:
        del graph
        self.strategies = list(strategies)
        return []


def test_predictive_prefetch_flag_controls_requested_cache_strategy() -> None:
    graph = _irregular_weighted_graph()
    engine = TNFRCacheAwareFFTEngine(
        enable_cache_optimization=True,
        enable_predictive_prefetch=False,
    )
    optimizer = _CapturingCacheOptimizer()
    engine.cache_optimizer = optimizer

    engine.spectral_convolution_cached(graph)

    assert CacheOptimizationStrategy.PREDICTIVE_PREFETCH not in optimizer.strategies
    assert CacheOptimizationStrategy.SPECTRAL_PERSISTENCE in optimizer.strategies


def test_public_spectrum_arrays_cannot_poison_cached_eigensystem() -> None:
    graph = nx.path_graph(3)
    eigenvalues, eigenvectors = get_laplacian_spectrum(graph)
    expected_values = eigenvalues.copy()
    expected_vectors = eigenvectors.copy()

    eigenvalues[0] = 123.0
    eigenvectors[0, 0] = 456.0
    repeated_values, repeated_vectors = get_laplacian_spectrum(graph)

    assert repeated_values is not eigenvalues
    assert repeated_vectors is not eigenvectors
    np.testing.assert_allclose(repeated_values, expected_values)
    np.testing.assert_allclose(repeated_vectors, expected_vectors)


def test_spectrum_cache_reads_the_selected_edge_weight() -> None:
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1.0, "cost")
    before, _ = get_laplacian_spectrum(graph, operator="combinatorial", weight="cost")
    graph.edges[0, 1]["cost"] = 4.0
    after, modes = get_laplacian_spectrum(
        graph, operator="combinatorial", weight="cost"
    )
    laplacian = nx.laplacian_matrix(graph, weight="cost").toarray()
    assert not np.allclose(before, after)
    np.testing.assert_allclose(after, np.linalg.eigvalsh(laplacian), atol=2e-14)
    np.testing.assert_allclose(laplacian @ modes, modes * after, atol=2e-14)


@pytest.mark.parametrize("operator", ["symmetric", "random_walk", "combinatorial"])
def test_spectrum_cache_retains_current_node_order(operator: str) -> None:
    graph = nx.path_graph(3)
    before, _ = get_laplacian_spectrum(graph, operator=operator)
    graph.remove_node(0)
    graph.add_edge(0, 1)
    assert list(graph) == [1, 2, 0]
    after, modes = get_laplacian_spectrum(graph, operator=operator)
    adjacency = nx.to_numpy_array(graph)
    degree = adjacency.sum(axis=1)
    if operator == "combinatorial":
        laplacian = np.diag(degree) - adjacency
    elif operator == "random_walk":
        laplacian = np.eye(3) - adjacency / degree[:, None]
    else:
        root = np.sqrt(degree)
        laplacian = np.eye(3) - adjacency / root[:, None] / root[None, :]
    np.testing.assert_allclose(after, before, atol=2e-14)
    np.testing.assert_allclose(laplacian @ modes, modes * after, atol=2e-14)


def test_spectrum_cache_reuses_diagonalization_after_rebuilding_operator(monkeypatch):
    from tnfr.mathematics import spectral
    from tnfr.utils.cache import invalidate_function_cache

    invalidate_function_cache(spectral._get_laplacian_spectrum_cached)
    original = scipy.linalg.eigh
    calls = []

    def observe(matrix, *args, **kwargs):
        calls.append(matrix.copy())
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "eigh", observe)
    graph = nx.path_graph(3)
    get_laplacian_spectrum(graph, operator="combinatorial")
    get_laplacian_spectrum(graph, operator="combinatorial")
    assert len(calls) == 1


def test_combinatorial_weights_are_readmitted_before_cache_lookup():
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1.0, "cost")
    get_laplacian_spectrum(graph, operator="combinatorial", weight="cost")
    graph.edges[0, 1]["cost"] = True
    with pytest.raises(TypeError, match="Laplacian edge weight"):
        get_laplacian_spectrum(graph, operator="combinatorial", weight="cost")


@pytest.mark.parametrize(
    "setting,operator", [("false", "combinatorial"), ("true", "symmetric")]
)
def test_normalized_alias_uses_shared_boolean_parsing(setting, operator):
    graph = nx.path_graph(3)
    expected, _ = get_laplacian_spectrum(graph, operator=operator)
    actual, _ = get_laplacian_spectrum(graph, normalized=setting)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="true/false"):
        get_laplacian_spectrum(graph, normalized="ambiguous")


@pytest.mark.parametrize("operator", ["symmetric", "random_walk", "combinatorial"])
@pytest.mark.parametrize("partial", [False, True])
def test_stationary_graph_modes_survive_long_heat_horizons(operator, partial):
    graph = nx.path_graph(3)
    initial = np.sqrt([1.0, 2.0, 1.0]) if operator == "symmetric" else np.ones(3)
    values, modes = get_laplacian_spectrum(
        graph, operator=operator, k=2 if partial else None
    )

    np.testing.assert_allclose(
        heat_diffusion(initial, modes, values, 1e20), initial, atol=2e-14
    )


@pytest.mark.parametrize("operator", ["symmetric", "random_walk", "combinatorial"])
def test_disconnected_stationary_modes_ignore_zero_weight_connections(operator):
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 0.0), (2, 3, 3.0)])
    graph.add_node(4)
    initial = np.array([2.0, 2.0, -3.0, -3.0, 4.0])
    values, modes = get_laplacian_spectrum(graph, operator=operator)

    assert np.count_nonzero(values == 0) == 3
    np.testing.assert_allclose(
        heat_diffusion(initial, modes, values, 1e20), initial, atol=2e-14
    )


def test_stationary_correction_preserves_genuinely_slow_nonzero_modes():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1e-20)
    values, modes = get_laplacian_spectrum(graph, operator="combinatorial")
    contrast = np.array([1.0, -1.0])

    assert values[0] == 0.0
    assert values[1] > 0.0
    np.testing.assert_allclose(
        heat_diffusion(contrast, modes, values, 5e19),
        np.exp(-1.0) * contrast,
        atol=2e-14,
    )
    # Arbitrary supplied small rates have no graph-derived zero identification.
    np.testing.assert_allclose(
        heat_diffusion(np.ones(1), np.eye(1), np.array([1e-20]), 1e20),
        [np.exp(-1.0)],
    )


def test_heat_diffusion_rejects_nonfinite_result_arithmetic():
    with pytest.raises(TNFRValueError, match="arithmetic must remain finite"):
        heat_diffusion(np.ones(1), np.eye(1), np.array([-1.0]), 1e308)


def test_unresolved_stationary_mixing_rejects_instead_of_clipping(monkeypatch):
    from tnfr.mathematics import spectral

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1e-20)
    # In a near-zero cluster an absolute-residual eigensolver may return
    # unresolved coordinate directions. Neither is the known constant mode.
    monkeypatch.setattr(
        spectral,
        "_get_laplacian_spectrum_cached",
        lambda *args: (np.array([0.0, 2e-20]), np.eye(2)),
    )
    with pytest.raises(TNFRValueError, match="cannot resolve the stationary subspace"):
        get_laplacian_spectrum(graph, operator="combinatorial")
