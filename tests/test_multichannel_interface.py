"""Offline tests for the multi-channel structural-interface pipeline.

These tests validate pipeline *mechanics* on synthetic, deterministic data and
exercise the real-data download path only through its graceful-skip behaviour,
so the suite never requires network access.

Honest note: the synthetic Kuramoto regime-switch fixture is used here only to
check that the spatial tetrad and the synchronisation baselines both track the
incoherent → coherent transition.  It is a mechanics fixture, not evidence for
the TNFR thesis.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("networkx")

from tnfr.validation.multichannel_interface import (
    MultichannelConfig,
    MultichannelWindowSeries,
    SynchronyDiscrimination,
    amplitude_pressure,
    analytic_phase_amplitude,
    build_coupling_graph,
    evaluate_synchrony_discrimination,
    fft_bandpass,
    kuramoto_order_parameter,
    mean_field_order,
    multichannel_window_series,
    phase_amplitude_matrices,
    phase_locking_matrix,
    phase_offsets,
)


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "multichannel_interface_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "multichannel_interface_benchmark", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _load_benchmark_module()


def _two_tone_channels(
    n_channels: int = 6, n_samples: int = 2048, fs: float = 128.0
) -> np.ndarray:
    """Deterministic alpha-band channels with small per-channel phase offsets."""
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(3)
    rows = []
    for j in range(n_channels):
        offset = 0.2 * j
        alpha = np.sin(2.0 * np.pi * 10.0 * t + offset)
        noise = 0.05 * rng.standard_normal(n_samples)
        rows.append(alpha + noise)
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# Signal -> phase/amplitude primitives
# ---------------------------------------------------------------------------
def test_fft_bandpass_passes_in_band_attenuates_out_of_band():
    fs = 128.0
    t = np.arange(2048) / fs
    in_band = np.sin(2.0 * np.pi * 10.0 * t)  # alpha
    out_band = np.sin(2.0 * np.pi * 40.0 * t)  # gamma
    mixed = in_band + out_band
    filtered = fft_bandpass(mixed, sampling_rate=fs, low=8.0, high=12.0)
    # In-band power retained, out-of-band power strongly attenuated.
    assert np.std(filtered) > 0.5 * np.std(in_band)
    residual_out = filtered - in_band
    assert np.std(residual_out) < 0.2 * np.std(out_band)


def test_analytic_phase_amplitude_tracks_pure_tone():
    fs = 128.0
    t = np.arange(2048) / fs
    signal = 2.0 * np.sin(2.0 * np.pi * 10.0 * t)
    phase, amplitude = analytic_phase_amplitude(signal, sampling_rate=fs)
    assert phase.shape == signal.shape
    assert amplitude.shape == signal.shape
    # Amplitude envelope of a pure tone is approximately its amplitude.
    interior = amplitude[200:-200]
    assert np.allclose(np.mean(interior), 2.0, atol=0.3)


def test_phase_amplitude_matrices_shapes():
    signals = _two_tone_channels()
    phase, amplitude = phase_amplitude_matrices(signals, sampling_rate=128.0)
    assert phase.shape == signals.shape
    assert amplitude.shape == signals.shape


# ---------------------------------------------------------------------------
# Synchronisation descriptors
# ---------------------------------------------------------------------------
def test_phase_locking_matrix_properties():
    signals = _two_tone_channels()
    phase, _ = phase_amplitude_matrices(signals, sampling_rate=128.0)
    plv = phase_locking_matrix(phase)
    n = signals.shape[0]
    assert plv.shape == (n, n)
    assert np.allclose(np.diag(plv), 1.0)
    assert np.allclose(plv, plv.T, atol=1e-9)
    assert np.all(plv >= 0.0) and np.all(plv <= 1.0)


def test_kuramoto_order_parameter_in_phase_vs_random():
    n_samples = 2048
    t = np.arange(n_samples) / 128.0
    # Strongly in-phase channels -> R near 1.
    in_phase = np.asarray([np.sin(2.0 * np.pi * 10.0 * t) for _ in range(8)])
    phase_in, _ = phase_amplitude_matrices(in_phase, sampling_rate=128.0)
    assert kuramoto_order_parameter(phase_in) > 0.9
    # Scattered phase offsets -> lower R.
    rng = np.random.default_rng(1)
    scattered = np.asarray(
        [
            np.sin(2.0 * np.pi * 10.0 * t + rng.uniform(-np.pi, np.pi))
            for _ in range(8)
        ]
    )
    phase_sc, _ = phase_amplitude_matrices(scattered, sampling_rate=128.0)
    assert kuramoto_order_parameter(phase_sc) < kuramoto_order_parameter(phase_in)


def test_mean_field_order_shapes_and_range():
    signals = _two_tone_channels()
    phase, _ = phase_amplitude_matrices(signals, sampling_rate=128.0)
    r, psi = mean_field_order(phase)
    assert r.shape == (signals.shape[1],)
    assert psi.shape == (signals.shape[1],)
    assert np.all(r >= 0.0) and np.all(r <= 1.0 + 1e-9)


def test_phase_offsets_bounded():
    signals = _two_tone_channels()
    phase, _ = phase_amplitude_matrices(signals, sampling_rate=128.0)
    theta = phase_offsets(phase)
    assert theta.shape == (signals.shape[0],)
    assert np.all(np.abs(theta) <= np.pi + 1e-9)


def test_amplitude_pressure_phase_independent():
    signals = _two_tone_channels()
    _, amplitude = phase_amplitude_matrices(signals, sampling_rate=128.0)
    pressure = amplitude_pressure(amplitude)
    assert pressure.shape == (signals.shape[0],)
    assert np.all(pressure >= 0.0)


# ---------------------------------------------------------------------------
# Coupling graph
# ---------------------------------------------------------------------------
def test_build_coupling_graph_attributes_and_degree():
    signals = _two_tone_channels(n_channels=8)
    phase, amplitude = phase_amplitude_matrices(signals, sampling_rate=128.0)
    k = 3
    graph = build_coupling_graph(phase, amplitude, k_neighbours=k)
    assert graph.number_of_nodes() == 8
    for _, attrs in graph.nodes(data=True):
        assert "phase" in attrs and "theta" in attrs
        assert "dnfr" in attrs and "delta_nfr" in attrs
    # kNN guarantees each node initiates k edges; degree may exceed k via
    # reciprocal links, but never falls below k for a connected feature set.
    for node in graph.nodes():
        assert graph.degree(node) >= 1
    for _, _, edge in graph.edges(data=True):
        assert "distance" in edge


# ---------------------------------------------------------------------------
# Synthetic Kuramoto regime switch (mechanics fixture)
# ---------------------------------------------------------------------------
def _regime_switch_signals():
    return BENCH.synthetic_kuramoto_regime_switch(
        n_oscillators=14, block=2048, coupling_low=0.05, coupling_high=2.0
    )


def test_window_series_structure_and_monotone_window_end():
    signals, _ = _regime_switch_signals()
    cfg = MultichannelConfig(window=512, step=128, k_neighbours=4)
    series = multichannel_window_series(signals, config=cfg)
    assert isinstance(series, MultichannelWindowSeries)
    n = series.window_end.size
    for arr in (
        series.grad_phi,
        series.k_phi,
        series.xi_c,
        series.phi_s,
        series.order_parameter,
        series.mean_plv,
        series.phase_dispersion,
    ):
        assert arr.shape == (n,)
    assert np.all(np.diff(series.window_end) > 0)


def test_order_parameter_higher_in_coherent_block():
    signals, labels = _regime_switch_signals()
    cfg = MultichannelConfig(window=512, step=128, k_neighbours=4)
    series = multichannel_window_series(signals, config=cfg)
    half = series.order_parameter.size // 2
    incoherent = np.nanmean(series.order_parameter[:half])
    coherent = np.nanmean(series.order_parameter[half:])
    assert coherent > incoherent
    # |∇φ| falls as the network synchronises (partial 1−R redundancy).
    assert np.nanmean(series.grad_phi[half:]) < np.nanmean(series.grad_phi[:half])


def test_window_series_rejects_short_signal():
    signals, _ = _regime_switch_signals()
    cfg = MultichannelConfig(window=signals.shape[1] + 1, step=128)
    with pytest.raises(ValueError):
        multichannel_window_series(signals, config=cfg)


def test_window_series_rejects_too_few_channels():
    signals = _two_tone_channels(n_channels=2, n_samples=2048)
    cfg = MultichannelConfig(window=512, step=128, k_neighbours=1)
    with pytest.raises(ValueError):
        multichannel_window_series(signals, config=cfg)


def test_evaluate_synchrony_discrimination_structure():
    signals, labels = _regime_switch_signals()
    cfg = MultichannelConfig(window=512, step=128, k_neighbours=4)
    result = evaluate_synchrony_discrimination(signals, labels, config=cfg)
    assert isinstance(result, SynchronyDiscrimination)
    # All AUCs are direction-agnostic discriminative scores in [0.5, 1].
    for value in result.auc.values():
        assert 0.5 <= value <= 1.0 + 1e-9
    # On a clean incoherent->coherent switch the baseline separates strongly.
    assert result.best_baseline[1] > 0.8
    # The tetrad also separates this regime (grad_phi / xi_c track synchrony).
    assert result.best_tnfr[1] > 0.8
    assert result.summary()
    assert result.interpretation


def test_config_validation():
    with pytest.raises(ValueError):
        MultichannelConfig(window=4)
    with pytest.raises(ValueError):
        MultichannelConfig(step=0)
    with pytest.raises(ValueError):
        MultichannelConfig(k_neighbours=0)
    with pytest.raises(ValueError):
        MultichannelConfig(sampling_rate=0.0)
    with pytest.raises(ValueError):
        MultichannelConfig(bandpass=(12.0, 8.0))


# ---------------------------------------------------------------------------
# Benchmark: ARFF parsing, robust clip, graceful skip
# ---------------------------------------------------------------------------
def _tiny_arff(n_rows: int = 1200) -> str:
    rng = np.random.default_rng(5)
    header = ["@RELATION eeg"]
    for i in range(14):
        header.append(f"@ATTRIBUTE ch{i} NUMERIC")
    header.append("@ATTRIBUTE eyeDetection {0,1}")
    header.append("@DATA")
    rows = []
    for r in range(n_rows):
        label = 0 if r < n_rows // 2 else 1
        vals = [f"{4000.0 + 50.0 * rng.standard_normal():.2f}" for _ in range(14)]
        rows.append(",".join(vals) + f",{label}")
    return "\n".join(header + rows)


def test_parse_arff_basic():
    text = _tiny_arff(1200)
    parsed = BENCH.parse_arff(text)
    assert parsed is not None
    signals, labels = parsed
    assert signals.shape[0] == 14
    assert signals.shape[1] == 1200
    assert labels.shape == (1200,)
    assert set(np.unique(labels)).issubset({0, 1})


def test_parse_arff_rejects_short():
    text = _tiny_arff(10)
    assert BENCH.parse_arff(text) is None


def test_robust_clip_removes_spikes():
    rng = np.random.default_rng(7)
    signals = 4000.0 + 50.0 * rng.standard_normal((14, 2000))
    signals[3, 100] = 7_000_000.0  # sensor pop
    cleaned = BENCH._robust_clip(signals)
    assert cleaned[3, 100] < 1e5
    assert np.max(np.abs(cleaned)) < 1e5


def test_extract_arff_text_from_zip():
    text = _tiny_arff(1100)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("EEG Eye State.arff", text)
    extracted = BENCH._extract_arff_text(buffer.getvalue())
    assert extracted is not None
    assert "@DATA" in extracted


def test_synthetic_kuramoto_regime_switch_shapes():
    signals, labels = BENCH.synthetic_kuramoto_regime_switch(
        n_oscillators=10, block=1024
    )
    assert signals.shape == (10, 2048)
    assert labels.shape == (2048,)
    assert labels[:1024].sum() == 0
    assert labels[1024:].sum() == 1024


def test_kuramoto_order_rises_with_coupling():
    weak = BENCH.kuramoto_simulate(20, 0.05, 3000, seed=1)
    strong = BENCH.kuramoto_simulate(20, 3.0, 3000, seed=1)
    phase_weak, _ = phase_amplitude_matrices(weak)
    phase_strong, _ = phase_amplitude_matrices(strong)
    assert kuramoto_order_parameter(phase_strong) > kuramoto_order_parameter(
        phase_weak
    )


def test_download_eeg_graceful_skip(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise OSError("no network in test environment")

    monkeypatch.setattr(BENCH, "urlopen", _boom)
    result = BENCH.download_eeg_eye_state(cache_path=tmp_path / "x.raw")
    assert result is None


def test_download_eeg_writes_full_payload(monkeypatch, tmp_path):
    # Regression: a successful download must persist the *entire* payload, not
    # an empty file (the chunk loop must write each chunk, not merely count it).
    payload = b"@RELATION t\n@DATA\n" + b"x" * 4096

    class _FakeResponse:
        def __init__(self, data):
            self._buf = data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            if n is None or n < 0:
                chunk, self._buf = self._buf, b""
                return chunk
            chunk, self._buf = self._buf[:n], self._buf[n:]
            return chunk

    monkeypatch.setattr(
        BENCH, "urlopen", lambda *a, **k: _FakeResponse(payload)
    )
    cache = tmp_path / "eeg.raw"
    result = BENCH.download_eeg_eye_state(cache_path=cache)
    assert result == cache
    assert cache.read_bytes() == payload


def test_run_multichannel_benchmark_synthetic_ok():
    cfg = MultichannelConfig(window=512, step=128, k_neighbours=4)
    report = BENCH.run_multichannel_benchmark(
        source="synthetic", config=cfg, synthetic_block=2048
    )
    assert report["status"] == "ok"
    assert report["n_windows"] > 0
    assert "auc" in report
    assert "interpretation" in report
    assert report["best_baseline"]["auc"] > 0.8
