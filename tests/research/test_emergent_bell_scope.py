"""Tests for the scoped Bell/CHSH benchmark."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "emergent_bell_inequality_bound.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "emergent_bell_inequality_bound",
    _BENCHMARK_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


def _settings() -> tuple[float, float, float, float]:
    return 0.0, math.pi / 4.0, math.pi / 8.0, 3.0 * math.pi / 8.0


def test_shared_angle_model_checks_the_pointwise_chsh_bound() -> None:
    angles = np.array([0.031, 0.47, 1.13, 2.71, 5.4])
    *_, chsh, max_pointwise = _BENCHMARK._chsh_from_shared_angles(
        angles,
        *_settings(),
    )

    assert max_pointwise == 2.0
    assert chsh == pytest.approx(-2.0)


def test_chsh_bound_is_independent_of_shared_angle_distribution() -> None:
    rng = np.random.default_rng(91)
    angles = np.concatenate(
        (
            rng.vonmises(mu=0.7, kappa=12.0, size=500),
            np.full(500, 2.13),
        )
    )
    *_, chsh, max_pointwise = _BENCHMARK._chsh_from_shared_angles(
        angles,
        0.13,
        0.82,
        0.39,
        1.17,
    )

    assert max_pointwise <= 2.0
    assert abs(chsh) <= 2.0


@pytest.mark.parametrize(
    "angles",
    [
        np.array([]),
        np.array([[0.1, 0.2]]),
        np.array([0.1, math.nan]),
        np.array([0.1, math.inf]),
    ],
)
def test_shared_angle_model_rejects_undefined_samples(
    angles: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        _BENCHMARK._chsh_from_shared_angles(angles, *_settings())


def test_auxiliary_phase_generator_is_seeded_and_bounded() -> None:
    kwargs = {
        "m": 3,
        "levels": 1,
        "steps": 12,
        "dt": 0.04,
    }
    first = _BENCHMARK.fractal_resonant_hidden_variable(
        np.random.default_rng(17),
        **kwargs,
    )
    second = _BENCHMARK.fractal_resonant_hidden_variable(
        np.random.default_rng(17),
        **kwargs,
    )

    assert first == pytest.approx(second)
    phase, order = first
    assert -math.pi <= phase <= math.pi
    assert 0.0 <= order <= 1.0
