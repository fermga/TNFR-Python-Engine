"""Telemetry tests covering νf estimators."""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

np = pytest.importorskip("numpy")

from tnfr.constants import merge_overrides
from tnfr.telemetry.nu_f import ensure_nu_f_telemetry


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
