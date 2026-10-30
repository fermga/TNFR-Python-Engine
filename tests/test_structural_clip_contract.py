"""Boundary projection contract shared by scalar and vectorized integrators."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from tnfr.dynamics.structural_clip import (
    get_clip_stats,
    reset_clip_stats,
    structural_clip,
    structural_clip_array,
)


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_array_clipping_matches_scalar_contract(mode):
    values = np.array([-3.0, -1.0, -0.98, 0.1, 0.95, 0.98, 3.0])
    expected = [structural_clip(value, mode=mode) for value in values]
    assert structural_clip_array(values, mode=mode) == pytest.approx(
        expected, abs=1e-15
    )


def test_soft_clip_preserves_identity_core_and_lands_smoothly_at_boundary():
    assert structural_clip(0.95, mode="soft") == 0.95
    assert structural_clip(-0.95, mode="soft") == -0.95
    assert 0.98 < structural_clip(0.98, mode="soft") < 1.0
    assert -1.0 < structural_clip(-0.98, mode="soft") < -0.98
    assert structural_clip(2.0, mode="soft") == 1.0

    epsilon = 1e-6
    left_slope = (
        structural_clip(1.0, mode="soft") - structural_clip(1.0 - epsilon, mode="soft")
    ) / epsilon
    assert abs(left_slope) < 1e-3


def test_soft_clip_is_affine_equivariant_across_finite_intervals():
    normalized = np.array([-1.2, -0.98, -0.4, 0.0, 0.95, 1.3])
    reference = structural_clip_array(normalized, mode="soft")
    transformed = 12.0 + 2.0 * normalized
    actual = structural_clip_array(transformed, lo=10.0, hi=14.0, mode="soft")
    assert actual == pytest.approx(12.0 + 2.0 * reference, abs=2e-15)


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        True,
        np.bool_(True),
        "0.2",
        0.2 + 0j,
        Fraction(1, 2**2000),
    ],
)
def test_scalar_clip_rejects_nonfinite_or_nonreal_values(value):
    with pytest.raises(ValueError, match="finite real"):
        structural_clip(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lo": math.nan},
        {"hi": math.inf},
        {"lo": 2.0, "hi": 1.0},
        {"mode": "elastic"},
        {"k": 0.0},
        {"k": math.nan},
        {"lo": np.bool_(False)},
        {"lo": Fraction(1, 2**2000), "hi": Fraction(1, 2**1999)},
    ],
)
def test_scalar_and_array_clip_share_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        structural_clip(0.0, **kwargs)
    with pytest.raises(ValueError):
        structural_clip_array([0.0], **kwargs)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, math.nan],
        [0.0, math.inf],
        [True, False],
        [True, 0.5],
        [np.bool_(False), 0.5],
        [0.0, 1.0 + 0j],
        ["0.0", "1.0"],
        [0.0, Fraction(1, 2**2000)],
    ],
)
def test_array_clip_rejects_nonfinite_or_nonreal_values(values):
    with pytest.raises(ValueError, match="finite real"):
        structural_clip_array(values)


def test_clip_telemetry_records_exact_interventions_only():
    reset_clip_stats()
    structural_clip(0.5, mode="soft", record_stats=True)
    structural_clip(0.98, mode="soft", record_stats=True)
    structural_clip(2.0, mode="hard", record_stats=True)
    summary = get_clip_stats().summary()
    assert summary["soft_clips"] == 1
    assert summary["hard_clips"] == 1
    assert summary["total_adjustments"] == 2


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_array_clip_materializes_admitted_raw_scalars_without_losing_shape(mode):
    tiny = Fraction(1, 2**1074)
    raw = [[Fraction(-1, 2), tiny], [0, Fraction(1, 4)]]
    result = structural_clip_array(raw, mode=mode)
    expected = np.array([[-0.5, float(tiny)], [0.0, 0.25]])
    assert result.shape == (2, 2)
    assert np.array_equal(result, expected)
    assert result[0, 1] > 0
    assert raw[0][0] == Fraction(-1, 2)
