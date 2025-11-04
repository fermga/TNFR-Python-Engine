"""Telemetry tests covering νf estimators."""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

np = pytest.importorskip("numpy")

from tnfr.constants import merge_overrides
from tnfr.dynamics.runtime import _run_after_callbacks
from tnfr.telemetry.nu_f import ensure_nu_f_telemetry, record_nu_f_window

def _compute_expected_ci(rate: float, total_duration: float, confidence: float) -> tuple[float, float]:
    variance = rate / total_duration
    std_error = math.sqrt(variance)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    lower = max(rate - z * std_error, 0.0)
    upper = rate + z * std_error
    return lower, upper

def test_nu_f_poisson_snapshot_matches_mle(structural_rng, graph_canon) -> None:
    """Simulated Poisson counts must match the νf MLE and CI estimates."""

    rng = structural_rng
    G = graph_canon()
    merge_overrides(G, HZ_STR_BRIDGE=2.5)
    accumulator = ensure_nu_f_telemetry(G, confidence_level=0.9)

    durations = rng.uniform(0.8, 1.6, size=24)
    lam = 3.2
    counts = rng.poisson(lam * durations)

    total_reorganisations = 0
    total_duration = float(np.sum(durations))
    for count, duration in zip(counts.tolist(), durations.tolist()):
        total_reorganisations += int(count)
        accumulator.record_counts(int(count), float(duration), graph=G)

    snapshot = accumulator.snapshot(graph=G)
    assert snapshot.total_reorganisations == total_reorganisations
    assert snapshot.total_duration == pytest.approx(total_duration)

    expected_rate = total_reorganisations / total_duration
    assert snapshot.rate_hz_str == pytest.approx(expected_rate)

    expected_variance = expected_rate / total_duration
    assert snapshot.variance_hz_str == pytest.approx(expected_variance)

    expected_lower, expected_upper = _compute_expected_ci(
        expected_rate, total_duration, accumulator.confidence_level
    )
    assert snapshot.ci_lower_hz_str == pytest.approx(expected_lower)
    assert snapshot.ci_upper_hz_str == pytest.approx(expected_upper)

    bridge = 2.5
    assert snapshot.rate_hz == pytest.approx(expected_rate * bridge)
    assert snapshot.variance_hz == pytest.approx(expected_variance * (bridge**2))
    assert snapshot.ci_lower_hz == pytest.approx(expected_lower * bridge)
    assert snapshot.ci_upper_hz == pytest.approx(expected_upper * bridge)

def test_nu_f_zero_counts_retains_zero_rate(graph_canon) -> None:
    """Zero reorganisations should yield zero rate and symmetric intervals."""

    G = graph_canon()
    accumulator = ensure_nu_f_telemetry(G, confidence_level=0.95)
    accumulator.record_counts(0, 1.0, graph=G)
    snapshot = accumulator.snapshot(graph=G)

    assert snapshot.rate_hz_str == pytest.approx(0.0)
    assert snapshot.variance_hz_str == pytest.approx(0.0)
    assert snapshot.ci_lower_hz_str == pytest.approx(0.0)
    assert snapshot.ci_upper_hz_str == pytest.approx(0.0)
    assert snapshot.rate_hz == pytest.approx(0.0)
    assert snapshot.ci_lower_hz == pytest.approx(0.0)
    assert snapshot.ci_upper_hz == pytest.approx(0.0)

def test_nu_f_history_survives_runtime_summary(graph_canon) -> None:
    """History snapshots must remain accessible alongside runtime summaries."""

    G = graph_canon()
    accumulator = ensure_nu_f_telemetry(G)

    first_snapshot = accumulator.record_counts(3, 1.0, graph=G)
    second_snapshot = accumulator.record_counts(5, 2.0, graph=G)

    telemetry = G.graph.get("telemetry")
    assert isinstance(telemetry, dict)

    history = telemetry.get("nu_f_history")
    assert isinstance(history, list)
    assert history == [first_snapshot.as_payload(), second_snapshot.as_payload()]

    telemetry["nu_f_snapshot"] = second_snapshot.as_payload()
    telemetry["nu_f_bridge"] = 1.0

    _run_after_callbacks(G, step_idx=2)

    history_after_callbacks = telemetry.get("nu_f_history")
    assert isinstance(history_after_callbacks, list)
    assert history_after_callbacks == history

    summary = telemetry.get("nu_f")
    assert isinstance(summary, dict)
    assert summary["total_reorganisations"] == second_snapshot.total_reorganisations
    assert summary["total_duration"] == second_snapshot.total_duration
    assert summary["rate_hz_str"] == second_snapshot.rate_hz_str

    third_snapshot = accumulator.record_counts(2, 1.5, graph=G)

    history_final = telemetry.get("nu_f_history")
    assert isinstance(history_final, list)
    assert history_final[-1] == third_snapshot.as_payload()
    assert len(history_final) == 3

    assert telemetry.get("nu_f") is summary

def test_custom_confidence_level_survives_repeated_ensure(graph_canon) -> None:
    """Custom νf confidence levels must persist across repeated ensures."""

    G = graph_canon()
    accumulator = ensure_nu_f_telemetry(G, confidence_level=0.82)
    first_snapshot = accumulator.record_counts(4, 2.0, graph=G)

    telemetry = G.graph.get("telemetry")
    assert isinstance(telemetry, dict)

    history = telemetry.get("nu_f_history")
    assert isinstance(history, list)
    assert history[-1] == first_snapshot.as_payload()

    same_accumulator = ensure_nu_f_telemetry(G)
    assert same_accumulator is accumulator
    assert same_accumulator.confidence_level == pytest.approx(0.82)

    second_snapshot = record_nu_f_window(G, 3, 1.5)
    assert second_snapshot.confidence_level == pytest.approx(0.82)

    another_reference = ensure_nu_f_telemetry(G)
    assert another_reference is accumulator
    assert another_reference.confidence_level == pytest.approx(0.82)

    updated_history = telemetry.get("nu_f_history")
    assert isinstance(updated_history, list)
    assert updated_history[-2:] == [
        first_snapshot.as_payload(),
        second_snapshot.as_payload(),
    ]
