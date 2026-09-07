"""Exact level-set geometry of the canonical local coherence kernel."""

import math

import pytest

from tnfr.metrics.common import structural_coherence
from tnfr.physics import coherence_level_set_geometry


def test_half_coherence_is_the_unit_l1_diamond():
    result = coherence_level_set_geometry(0.5)

    assert result.l1_radius == pytest.approx(1.0)
    assert result.euclidean_radius_minimum == pytest.approx(1.0 / math.sqrt(2.0))
    assert result.euclidean_radius_maximum == pytest.approx(1.0)
    assert result.intrinsic_dimension == 1
    assert len(result.singular_points) == 4
    assert result.smooth_away_from_singular_points
    assert not result.is_globally_smooth_embedded_manifold


@pytest.mark.parametrize("pressure,velocity", [(1.0, 0.0), (0.0, -1.0), (0.4, 0.6)])
def test_every_unit_diamond_probe_has_half_coherence(pressure, velocity):
    assert structural_coherence(pressure, velocity) == pytest.approx(0.5)


def test_full_coherence_collapses_to_the_equilibrium_point():
    result = coherence_level_set_geometry(1.0)

    assert result.is_equilibrium_point
    assert result.intrinsic_dimension == 0
    assert result.singular_points == ((0.0, 0.0),)
    assert result.l1_radius == 0.0
    assert result.is_globally_smooth_embedded_manifold
    assert not result.regular_gradient_available
    assert result.regular_gradient_norm == 0.0


def test_regular_gradient_norm_follows_the_exact_kernel_derivative():
    coherence = 0.25
    result = coherence_level_set_geometry(coherence)
    assert result.regular_gradient_available
    assert result.regular_gradient_norm == pytest.approx(math.sqrt(2.0) * coherence**2)


@pytest.mark.parametrize(
    "value", [0.0, -0.1, 1.1, float("nan"), float("inf"), True]
)
def test_nonfinite_state_levels_are_rejected(value):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        coherence_level_set_geometry(value)


def test_unrepresentable_finite_level_radius_is_rejected():
    with pytest.raises(ValueError, match="radius exceeds"):
        coherence_level_set_geometry(1e-320)


def test_unrepresentable_nonzero_regular_gradient_is_not_reported_as_zero():
    with pytest.raises(ValueError, match="regular gradient norm"):
        coherence_level_set_geometry(1e-200)
