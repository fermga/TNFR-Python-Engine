"""Contracts for the assumption-explicit life time-series diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.physics.life import (
    compute_autopoietic_coefficient,
    compute_self_generation,
    detect_life_emergence,
)


def _detect(*, times=(0.0, 1.0, 2.0), rate=(0.0, 2.0, 4.0)):
    return detect_life_emergence(
        times=times,
        epi_series=np.array([0.5, 0.5, 0.5]),
        dEPI_dt=np.asarray(rate, dtype=float),
        dnfr_external=np.ones(3),
        d_dnfr_external_dt=np.zeros(3),
        epsilon=0.5,
        gamma=4.0,
        epi_max=1.0,
    )


def test_life_threshold_uses_first_sample_when_already_above() -> None:
    telemetry = _detect(rate=(2.0, 0.0, 4.0))

    assert telemetry.autopoietic_coefficient[0] > 1.0
    assert telemetry.life_threshold_time == pytest.approx(0.0)


def test_life_threshold_interpolates_first_upward_crossing() -> None:
    telemetry = _detect(rate=(0.0, 2.0, 4.0))

    assert telemetry.autopoietic_coefficient[0] < 1.0
    assert telemetry.autopoietic_coefficient[1] > 1.0
    assert 0.0 < telemetry.life_threshold_time < 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("times", (0.0, 0.0, 1.0), "strictly increasing"),
        ("times", (0.0, float("nan"), 2.0), "finite"),
        ("epi_series", np.array([0.5, -0.1, 0.5]), "nonnegative"),
        ("dEPI_dt", np.array([True, False, True]), "numeric"),
        ("dnfr_external", np.ones(2), "same shape"),
    ],
)
def test_life_diagnostics_reject_invalid_sample_contracts(
    field: str,
    value,
    message: str,
) -> None:
    kwargs = {
        "times": (0.0, 1.0, 2.0),
        "epi_series": np.array([0.5, 0.5, 0.5]),
        "dEPI_dt": np.array([0.0, 2.0, 4.0]),
        "dnfr_external": np.ones(3),
        "d_dnfr_external_dt": np.zeros(3),
        "epsilon": 0.5,
        "gamma": 4.0,
        "epi_max": 1.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        detect_life_emergence(**kwargs)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("epsilon", 1.1, "at most"),
        ("gamma", -1.0, "at least"),
        ("epi_max", 0.0, "positive"),
    ],
)
def test_life_diagnostics_reject_invalid_model_parameters(
    parameter: str,
    value: float,
    message: str,
) -> None:
    kwargs = {
        "times": (0.0, 1.0, 2.0),
        "epi_series": np.array([0.5, 0.5, 0.5]),
        "dEPI_dt": np.array([0.0, 2.0, 4.0]),
        "dnfr_external": np.ones(3),
        "d_dnfr_external_dt": np.zeros(3),
        "epsilon": 0.5,
        "gamma": 4.0,
        "epi_max": 1.0,
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=message):
        detect_life_emergence(**kwargs)


def test_life_helpers_reject_broadcasting_and_boolean_series() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_autopoietic_coefficient(
            np.ones(3),
            np.ones(1),
            np.ones(3),
        )
    with pytest.raises(ValueError, match="numeric"):
        compute_self_generation(
            np.array([True, False]),
            gamma=1.0,
            epi_max=1.0,
        )


@pytest.mark.parametrize(
    "samples",
    [
        [0.0, True, 2.0],
        [0.0, np.bool_(False), 2.0],
        ["0.0", "1.0", "2.0"],
        np.asarray([0.0, "1.0", 2.0], dtype=object),
    ],
)
def test_life_series_reject_logical_and_textual_samples_before_coercion(
    samples,
) -> None:
    with pytest.raises(ValueError, match="numeric"):
        compute_self_generation(samples, gamma=1.0, epi_max=3.0)
