"""Offline tests for the temporal structural-interface pipeline.

These tests validate pipeline *mechanics* on synthetic, deterministic data and
exercise the real-data download path only through its graceful-skip behaviour,
so the suite never requires network access.

Honest note: the synthetic fold fixture is used here purely to check that the
classical critical-slowing-down baselines (variance, lag-1 autocorrelation)
rise before a known transition.  It is a mechanics fixture, not evidence for the
TNFR thesis.
"""

from __future__ import annotations

import importlib.util
import io
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("networkx")

from tnfr.validation.temporal_interface import (
    EarlyWarningComparison,
    TemporalInterfaceConfig,
    WindowTetradSeries,
    build_temporal_proximity_graph,
    delay_embedding,
    evaluate_early_warning,
    hilbert_instantaneous_phase,
    kendall_tau,
    local_structural_pressure,
    rolling_lag1_autocorrelation,
    rolling_variance,
    window_tetrad_series,
)


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "temporal_interface_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "temporal_interface_benchmark", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _load_benchmark_module()


# ---------------------------------------------------------------------------
# Primitive mechanics
# ---------------------------------------------------------------------------
def test_hilbert_phase_shape_and_range():
    t = np.linspace(0.0, 4.0 * math.pi, 256)
    signal = np.sin(t)
    phase = hilbert_instantaneous_phase(signal)
    assert phase.shape == signal.shape
    assert np.all(phase >= -math.pi - 1e-9)
    assert np.all(phase <= math.pi + 1e-9)


def test_hilbert_phase_tracks_pure_tone():
    # For a pure sine the instantaneous phase should advance monotonically
    # (modulo wrapping); the unwrapped phase is near-linear in time.
    t = np.linspace(0.0, 8.0 * math.pi, 512)
    phase = np.unwrap(hilbert_instantaneous_phase(np.sin(t)))
    diffs = np.diff(phase)
    # Most increments are positive for a forward-advancing tone.
    assert np.mean(diffs > 0) > 0.9


def test_delay_embedding_shape():
    signal = np.arange(20, dtype=float)
    vectors = delay_embedding(signal, dim=3, tau=2)
    # rows = n - (dim - 1) * tau = 20 - 4 = 16
    assert vectors.shape == (16, 3)
    # First row is [x0, x2, x4].
    assert list(vectors[0]) == [0.0, 2.0, 4.0]


def test_delay_embedding_requires_enough_samples():
    with pytest.raises(ValueError):
        delay_embedding(np.arange(3, dtype=float), dim=5, tau=2)


def test_local_structural_pressure_nonnegative():
    rng = np.random.default_rng(1)
    signal = np.cumsum(rng.standard_normal(200))
    pressure = local_structural_pressure(signal, smoothing=5)
    assert pressure.shape == signal.shape
    assert np.all(pressure >= 0.0)


def test_build_graph_sets_phase_and_pressure_attributes():
    rng = np.random.default_rng(2)
    window = rng.standard_normal(120)
    phase = hilbert_instantaneous_phase(window)
    pressure = local_structural_pressure(window)
    config = TemporalInterfaceConfig(
        embedding_dim=3, embedding_tau=2, k_neighbours=6, window=120, step=30
    )
    graph = build_temporal_proximity_graph(
        window, phase=phase, pressure=pressure, config=config
    )
    assert graph.number_of_nodes() > 0
    for _, data in graph.nodes(data=True):
        assert "phase" in data and "theta" in data
        assert "dnfr" in data and "delta_nfr" in data
        assert math.isfinite(data["phase"])
        assert data["dnfr"] >= 0.0


# ---------------------------------------------------------------------------
# Rolling tetrad and baselines
# ---------------------------------------------------------------------------
def test_window_tetrad_series_structure():
    series = BENCH.synthetic_fold_transition(n=1200, transition_at=900, seed=3)
    config = TemporalInterfaceConfig(window=200, step=40)
    out = window_tetrad_series(series, config=config)
    assert isinstance(out, WindowTetradSeries)
    # window_end must be strictly increasing.
    assert np.all(np.diff(out.window_end) > 0)
    n = out.window_end.size
    for arr in (out.grad_phi, out.k_phi, out.xi_c, out.phi_s, out.variance, out.lag1_autocorr):
        assert arr.shape == (n,)


def test_window_tetrad_series_rejects_short_signal():
    config = TemporalInterfaceConfig(window=200, step=40)
    with pytest.raises(ValueError):
        window_tetrad_series(np.zeros(50), config=config)


def test_rolling_variance_rises_before_fold():
    series = BENCH.synthetic_fold_transition(n=2400, transition_at=1800, seed=4)
    var = rolling_variance(series, window=240, step=30)
    var = var[np.isfinite(var)]
    assert var.size >= 4
    # Variance late in the run (approaching the fold) exceeds the early level.
    assert np.mean(var[-3:]) > np.mean(var[:3])


def test_rolling_autocorr_rises_before_fold():
    series = BENCH.synthetic_fold_transition(n=2400, transition_at=1800, seed=5)
    ac = rolling_lag1_autocorrelation(series, window=240, step=30)
    ac = ac[np.isfinite(ac)]
    assert ac.size >= 4
    assert np.mean(ac[-3:]) > np.mean(ac[:3])


# ---------------------------------------------------------------------------
# Kendall tau
# ---------------------------------------------------------------------------
def test_kendall_tau_monotone_increasing():
    assert kendall_tau(np.arange(10, dtype=float)) == pytest.approx(1.0)


def test_kendall_tau_monotone_decreasing():
    assert kendall_tau(np.arange(10, 0, -1, dtype=float)) == pytest.approx(-1.0)


def test_kendall_tau_handles_nan_and_short():
    assert math.isnan(kendall_tau(np.array([1.0])))
    # NaNs are ignored; the remaining values are increasing.
    series = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    assert kendall_tau(series) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Early-warning evaluation
# ---------------------------------------------------------------------------
def test_evaluate_early_warning_reports_comparison():
    series = BENCH.synthetic_fold_transition(n=2400, transition_at=1800, seed=6)
    config = TemporalInterfaceConfig(window=240, step=30)
    result = evaluate_early_warning(
        series, transition_index=1800, config=config
    )
    assert isinstance(result, EarlyWarningComparison)
    assert set(result.tnfr_indicators).issubset(set(result.trends))
    assert set(result.baseline_indicators).issubset(set(result.trends))
    assert result.n_pre_transition_windows > 0
    assert isinstance(result.interpretation, str) and result.interpretation
    # On a clean single-series fold, the classical baselines should be the
    # strongest rising indicator (honest, expected calibration outcome).
    assert result.best_baseline[1] >= result.best_tnfr[1]


def test_evaluate_early_warning_summary_runs():
    series = BENCH.synthetic_fold_transition(n=1200, transition_at=900, seed=7)
    config = TemporalInterfaceConfig(window=200, step=40)
    result = evaluate_early_warning(series, transition_index=900, config=config)
    summary = result.summary()
    assert isinstance(summary, str) and summary


# ---------------------------------------------------------------------------
# Benchmark: parsing, event detection, graceful skip
# ---------------------------------------------------------------------------
def test_parse_frequency_csv_semicolon_decimal_comma():
    text = "\n".join(
        [
            "timestamp;frequency",
            "2020-01-01 00:00:00;50,012",
            "2020-01-01 00:00:01;49,998",
            "2020-01-01 00:00:02;NaN",
        ]
    )
    series = BENCH._parse_frequency_csv(text)
    finite = series[np.isfinite(series)]
    assert finite.size == 2
    assert finite[0] == pytest.approx(50.012, abs=1e-6)


def test_parse_frequency_csv_comma_dot_decimal():
    text = "\n".join(
        [
            "timestamp,frequency",
            "2020-01-01 00:00:00,50.012",
            "2020-01-01 00:00:01,49.990",
        ]
    )
    series = BENCH._parse_frequency_csv(text)
    finite = series[np.isfinite(series)]
    assert finite.size == 2
    assert finite[1] == pytest.approx(49.990, abs=1e-6)


def test_load_grid_frequency_series_from_zip(tmp_path):
    # Build an in-memory zip containing a small frequency CSV.
    rng = np.random.default_rng(8)
    lines = ["timestamp,frequency"]
    for i in range(1000):
        f = 50.0 + 0.02 * rng.standard_normal()
        lines.append(f"2020-01-01 00:00:{i:02d},{f:.4f}")
    csv_text = "\n".join(lines)
    zip_path = tmp_path / "test_Frequenz.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("202001_Frequenz.csv", csv_text)
    series = BENCH.load_grid_frequency_series(zip_path, max_points=500)
    assert series is not None
    assert series.size <= 500
    assert np.all(np.isfinite(series))
    assert 49.0 < float(np.mean(series)) < 51.0


def test_load_grid_frequency_series_rejects_garbage(tmp_path):
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"not a zip file")
    assert BENCH.load_grid_frequency_series(bad) is None


def test_detect_excursion_event_finds_interior_spike():
    series = np.full(1000, 50.0)
    series[500] = 49.0  # large interior excursion
    idx = BENCH.detect_excursion_event(series)
    assert idx == 500


def test_download_grid_frequency_graceful_skip(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise OSError("no network in test environment")

    monkeypatch.setattr(BENCH, "urlopen", _boom)
    result = BENCH.download_grid_frequency_month(
        2020, 1, cache_path=tmp_path / "x.zip"
    )
    assert result is None


def test_run_temporal_benchmark_synthetic_ok():
    config = TemporalInterfaceConfig(window=240, step=30)
    report = BENCH.run_temporal_benchmark(
        source="synthetic",
        year=2020,
        month=1,
        config=config,
        max_points=6000,
        max_bytes=80_000_000,
    )
    assert report["status"] == "ok"
    assert report["n_windows"] > 0
    assert "interpretation" in report


def test_run_temporal_benchmark_grid_graceful_skip(monkeypatch):
    monkeypatch.setattr(
        BENCH, "download_grid_frequency_month", lambda *a, **k: None
    )
    config = TemporalInterfaceConfig(window=240, step=30)
    report = BENCH.run_temporal_benchmark(
        source="grid",
        year=2020,
        month=1,
        config=config,
        max_points=6000,
        max_bytes=80_000_000,
    )
    assert report["status"] == "skipped"
    assert "reason" in report
