"""Assumption-explicit time-series diagnostics for life-like TNFR regimes.

The functions evaluate a declared logistic self-generation model and four
dimensionless readouts on supplied samples. They do not evolve a graph, prove
autopoiesis, or classify a system as biological. The selected A(t) > 1 crossing
is an operational event whose scope is documented in
theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..mathematics.unified_numerical import np
from ._helpers import finite_real_scalar
from ._helpers import finite_real_series as _finite_series
from ._helpers import safe_div as _safe_div


@dataclass
class LifeTelemetry:
    """Finite life-model telemetry evaluated on one declared time grid."""

    times: Sequence[float]
    vitality_index: np.ndarray
    autopoietic_coefficient: np.ndarray
    self_org_index: np.ndarray
    stability_margin: np.ndarray
    life_threshold_time: float | None


def _finite_scalar(
    value: Any,
    name: str,
    *,
    lower: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Return a finite non-Boolean model parameter."""

    result = finite_real_scalar(value, name)
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if lower is not None and result < lower:
        raise ValueError(f"{name} must be at least {lower}")
    return result



def _same_shape(reference: np.ndarray, value: np.ndarray, name: str) -> None:
    """Reject implicit broadcasting between independently sampled channels."""

    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must have the same shape as the EPI series"
        )


def _finite_output(values: np.ndarray, name: str) -> np.ndarray:
    """Reject overflow rather than publishing invalid telemetry."""

    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} is non-finite for the supplied model parameters")
    return result


def compute_self_generation(
    epi_series: np.ndarray,
    gamma: float,
    epi_max: float,
) -> np.ndarray:
    """Evaluate the declared logistic term G = gamma*x*(1 - x/epi_max)."""

    epi = _finite_series(epi_series, "epi_series", nonnegative=True)
    gamma_value = _finite_scalar(gamma, "gamma", lower=0.0)
    maximum = _finite_scalar(epi_max, "epi_max", strictly_positive=True)
    return _finite_output(
        gamma_value * epi * (1.0 - epi / maximum),
        "self-generation series",
    )


def compute_autopoietic_coefficient(
    G_epi: np.ndarray,
    dEPI_dt: np.ndarray,
    dnfr_external: np.ndarray,
) -> np.ndarray:
    """Evaluate the time-local ratio A = G*dEPI_dt/(abs(p_ext)^2 + eps)."""

    generation = _finite_series(G_epi, "G_epi")
    rate = _finite_series(dEPI_dt, "dEPI_dt")
    external = _finite_series(dnfr_external, "dnfr_external")
    _same_shape(generation, rate, "dEPI_dt")
    _same_shape(generation, external, "dnfr_external")
    return _finite_output(
        _safe_div(generation * rate, np.square(np.abs(external))),
        "autopoietic coefficient",
    )


def compute_self_org_index(
    epi_series: np.ndarray,
    epsilon: float,
    gamma: float,
    epi_max: float,
    d_dnfr_external_dt: np.ndarray,
    delta: float = 1e-9,
) -> np.ndarray:
    """Evaluate the declared local sensitivity ratio S(t)."""

    epi = _finite_series(epi_series, "epi_series", nonnegative=True)
    external_rate = _finite_series(
        d_dnfr_external_dt,
        "d_dnfr_external_dt",
    )
    _same_shape(epi, external_rate, "d_dnfr_external_dt")
    feedback = _finite_scalar(epsilon, "epsilon", lower=0.0)
    if feedback > 1.0:
        raise ValueError("epsilon must be at most 1.0")
    gamma_value = _finite_scalar(gamma, "gamma", lower=0.0)
    maximum = _finite_scalar(epi_max, "epi_max", strictly_positive=True)
    regularizer = _finite_scalar(delta, "delta", lower=0.0)

    derivative = gamma_value * (1.0 - 2.0 * epi / maximum)
    return _finite_output(
        _safe_div(
            feedback * np.abs(derivative),
            np.abs(external_rate) + regularizer,
        ),
        "self-organization index",
    )


def compute_stability_margin(
    epi_series: np.ndarray,
    epi_max: float,
) -> np.ndarray:
    """Evaluate M = (x - epi_max/2)/epi_max for EPI magnitudes."""

    epi = _finite_series(epi_series, "epi_series", nonnegative=True)
    maximum = _finite_scalar(epi_max, "epi_max", strictly_positive=True)
    return _finite_output(
        (epi - 0.5 * maximum) / maximum,
        "stability margin",
    )


def detect_life_emergence(
    times: Sequence[float],
    epi_series: np.ndarray,
    dEPI_dt: np.ndarray,
    dnfr_external: np.ndarray,
    d_dnfr_external_dt: np.ndarray,
    epsilon: float,
    gamma: float,
    epi_max: float,
) -> LifeTelemetry:
    """Evaluate the declared diagnostics and locate the first A(t) > 1 event.

    The event is a sampled-model threshold with linear interpolation across the
    first upward crossing. It is not a persistence theorem or biological
    classification.
    """

    time_values = _finite_series(times, "times", nonempty=True)
    if np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    epi = _finite_series(
        epi_series,
        "epi_series",
        nonnegative=True,
        nonempty=True,
    )
    rate = _finite_series(dEPI_dt, "dEPI_dt")
    external = _finite_series(dnfr_external, "dnfr_external")
    external_rate = _finite_series(
        d_dnfr_external_dt,
        "d_dnfr_external_dt",
    )
    for name, values in (
        ("times", time_values),
        ("dEPI_dt", rate),
        ("dnfr_external", external),
        ("d_dnfr_external_dt", external_rate),
    ):
        if values.shape != epi.shape:
            raise ValueError(f"{name} must have the same shape as epi_series")

    feedback = _finite_scalar(epsilon, "epsilon", lower=0.0)
    if feedback > 1.0:
        raise ValueError("epsilon must be at most 1.0")
    gamma_value = _finite_scalar(gamma, "gamma", lower=0.0)
    maximum = _finite_scalar(epi_max, "epi_max", strictly_positive=True)

    generation = compute_self_generation(epi, gamma_value, maximum)
    autopoietic = compute_autopoietic_coefficient(
        generation,
        rate,
        external,
    )
    self_org = compute_self_org_index(
        epi,
        feedback,
        gamma_value,
        maximum,
        external_rate,
    )
    margin = compute_stability_margin(epi, maximum)

    internal_pressure = feedback * generation
    vitality = _finite_output(
        _safe_div(
            np.abs(internal_pressure),
            np.abs(internal_pressure) + np.abs(external),
        ),
        "vitality index",
    )

    threshold_time: float | None = None
    if autopoietic[0] > 1.0:
        threshold_time = float(time_values[0])
    else:
        crossings = np.flatnonzero(
            (autopoietic[:-1] <= 1.0) & (autopoietic[1:] > 1.0)
        )
        if crossings.size:
            index = int(crossings[0])
            left = float(autopoietic[index])
            right = float(autopoietic[index + 1])
            fraction = (1.0 - left) / (right - left)
            threshold_time = float(
                time_values[index]
                + fraction * (time_values[index + 1] - time_values[index])
            )

    return LifeTelemetry(
        times=tuple(float(value) for value in time_values),
        vitality_index=vitality,
        autopoietic_coefficient=autopoietic,
        self_org_index=self_org,
        stability_margin=margin,
        life_threshold_time=threshold_time,
    )


__all__ = [
    "LifeTelemetry",
    "compute_self_generation",
    "compute_autopoietic_coefficient",
    "compute_self_org_index",
    "compute_stability_margin",
    "detect_life_emergence",
]
