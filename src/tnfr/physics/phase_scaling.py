r"""Finite-size diagnostics for TNFR coherence-transition studies.

This module analyzes a pre-registered rectangular sweep: the same control
values are measured at several graph sizes with one or more replicates per
cell.  It deliberately reports slopes against node count ``N``.  Translating
those slopes into thermodynamic exponent ratios requires an independently
justified relation between ``N`` and a linear system size, plus replication
across graph families.

The routines are read-only.  They do not alter the nodal equation, operator
grammar, or the operational labels in :mod:`tnfr.physics.phase_transition`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

from ..mathematics.unified_numerical import np


@dataclass(frozen=True)
class SizePowerLawFit:
    r"""Measured fit ``observable = exp(intercept) * N**slope``.

    ``slope_standard_error`` measures only regression residuals.  It is not a
    confidence interval for a universality class.
    """

    slope: float
    intercept: float
    r_squared: float
    slope_standard_error: float | None
    point_count: int


@dataclass(frozen=True)
class PhaseScalingDiagnostic:
    """Finite-size read-out from a shared control-parameter sweep."""

    system_sizes: tuple[int, ...]
    control_values: tuple[float, ...]
    replicate_count: int
    pseudocritical_control: tuple[float | None, ...]
    pseudocritical_controls: tuple[tuple[float, ...], ...]
    peak_is_unique: tuple[bool, ...]
    all_peaks_unique: bool
    peak_at_control_boundary: tuple[bool, ...]
    all_peaks_interior: bool
    peak_susceptibility: tuple[float, ...]
    order_at_peak: tuple[float, ...]
    coherence_length_at_peak: tuple[float, ...] | None
    peak_susceptibility_sem: tuple[float | None, ...]
    order_at_peak_sem: tuple[float | None, ...]
    coherence_length_at_peak_sem: tuple[float | None, ...] | None
    susceptibility_size_fit: SizePowerLawFit | None
    order_size_fit: SizePowerLawFit | None
    coherence_length_size_fit: SizePowerLawFit | None
    nonpositive_fit_observables: tuple[str, ...]
    status: str
    limitations: tuple[str, ...]


def analyze_phase_finite_size_scaling(
    system_sizes: Sequence[int],
    control_values: Sequence[float],
    order_parameter: object,
    susceptibility: object,
    coherence_length: object | None = None,
) -> PhaseScalingDiagnostic:
    r"""Analyze a balanced finite-size phase-transition sweep.

    Parameters
    ----------
    system_sizes:
        Strictly increasing positive node counts.  At least three sizes are
        required so a size trend is more than a two-point interpolation.
    control_values:
        Strictly increasing finite values of one declared control parameter,
        shared by every size.  At least two values are required.
    order_parameter, susceptibility, coherence_length:
        Arrays shaped ``(n_sizes, n_controls)`` or
        ``(n_sizes, n_controls, n_replicates)``.  Signed order-parameter input
        is converted to magnitude before replicate averaging.  Susceptibility
        and coherence length must be non-negative.

    Returns
    -------
    PhaseScalingDiagnostic
        Pseudocritical controls (all exact sampled susceptibility maximizers),
        peak observables, replicate standard errors, and log-log slopes against
        node count.  ``pseudocritical_control`` contains a value only when the
        sampled maximizer is unique; ambiguous rows contain ``None`` and expose
        every tied value through ``pseudocritical_controls``.  Co-observables on
        a plateau are averaged across its maximizing controls within each
        replicate. A power-law fit is withheld unless every sampled size has a
        strictly positive value for that observable; no size is silently
        removed. ``nonpositive_fit_observables`` names withheld fits. These are
        finite-study measurements, not universal critical exponents.
    """
    sizes = _validate_sizes(system_sizes)
    controls = _validate_controls(control_values)
    expected_prefix = (len(sizes), len(controls))

    order = _as_replicated_array("order_parameter", order_parameter, expected_prefix)
    suscept = _as_replicated_array("susceptibility", susceptibility, expected_prefix)
    if order.shape != suscept.shape:
        raise ValueError(
            "order_parameter and susceptibility must use the same replicate count"
        )
    if np.any(suscept < 0.0):
        raise ValueError("susceptibility cannot contain negative values")

    xi = None
    if coherence_length is not None:
        xi = _as_replicated_array(
            "coherence_length", coherence_length, expected_prefix
        )
        if xi.shape != order.shape:
            raise ValueError(
                "coherence_length must use the same replicate count as the other arrays"
            )
        if np.any(xi < 0.0):
            raise ValueError("coherence_length cannot contain negative values")

    mean_suscept = _stable_nonnegative_mean(suscept, axis=2)
    maximizing_indices = tuple(
        tuple(int(index) for index in np.flatnonzero(row == np.max(row)))
        for row in mean_suscept
    )

    peak_susceptibility: list[float] = []
    order_at_peak: list[float] = []
    xi_at_peak: list[float] | None = [] if xi is not None else None
    susceptibility_sem: list[float | None] = []
    order_sem: list[float | None] = []
    xi_sem: list[float | None] | None = [] if xi is not None else None

    for size_index, indices in enumerate(maximizing_indices):
        index_array = np.asarray(indices, dtype=int)
        susceptibility_by_replicate = _stable_nonnegative_mean(
            suscept[size_index, index_array, :], axis=0
        )
        order_by_replicate = _stable_nonnegative_mean(
            np.abs(order[size_index, index_array, :]), axis=0
        )
        peak_susceptibility.append(
            float(_stable_nonnegative_mean(susceptibility_by_replicate, axis=0))
        )
        order_at_peak.append(
            float(_stable_nonnegative_mean(order_by_replicate, axis=0))
        )
        susceptibility_sem.append(_standard_error(susceptibility_by_replicate))
        order_sem.append(_standard_error(order_by_replicate))
        if xi is not None and xi_at_peak is not None:
            xi_by_replicate = _stable_nonnegative_mean(
                xi[size_index, index_array, :], axis=0
            )
            xi_at_peak.append(
                float(_stable_nonnegative_mean(xi_by_replicate, axis=0))
            )
            assert xi_sem is not None
            xi_sem.append(_standard_error(xi_by_replicate))

    try:
        size_array = np.asarray(sizes, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "system_sizes must be representable as finite floating-point values"
        ) from exc
    if not np.all(np.isfinite(size_array)) or any(
        right <= left for left, right in zip(size_array, size_array[1:])
    ):
        raise ValueError(
            "system_sizes must remain strictly increasing when represented "
            "for the numerical fit"
        )
    susceptibility_fit = _fit_positive_size_power_law(
        size_array, np.asarray(peak_susceptibility, dtype=float)
    )
    order_fit = _fit_positive_size_power_law(
        size_array, np.asarray(order_at_peak, dtype=float)
    )
    xi_fit = (
        None
        if xi_at_peak is None
        else _fit_positive_size_power_law(
            size_array, np.asarray(xi_at_peak, dtype=float)
        )
    )
    nonpositive_fit_observables = tuple(
        name
        for name, values in (
            ("peak_susceptibility", peak_susceptibility),
            ("order_at_peak", order_at_peak),
            (
                "coherence_length_at_peak",
                () if xi_at_peak is None else xi_at_peak,
            ),
        )
        if values and any(value <= 0.0 for value in values)
    )

    unique_flags = tuple(len(indices) == 1 for indices in maximizing_indices)
    boundary_flags = tuple(
        any(index in (0, len(controls) - 1) for index in indices)
        for indices in maximizing_indices
    )
    limitations = [
        "Pseudocritical controls are restricted to the sampled control grid.",
        "Slopes are measured against node count N, not an inferred linear size.",
        "One graph family cannot establish a universal transition class.",
    ]
    if any(boundary_flags):
        limitations.append(
            "At least one susceptibility maximum lies on the control-grid boundary; "
            "the corresponding pseudocritical point is not bracketed."
        )
    if not all(unique_flags):
        limitations.append(
            "At least one susceptibility maximum is a sampled plateau; no unique "
            "pseudocritical control is identified for that size."
        )
    if nonpositive_fit_observables:
        limitations.append(
            "Power-law fits were withheld for observables with a non-positive "
            "value at one or more sampled sizes; no sizes were dropped."
        )

    status_suffixes: list[str] = []
    if any(boundary_flags):
        status_suffixes.append("unbracketed")
    if not all(unique_flags):
        status_suffixes.append("ambiguous_peak")
    if nonpositive_fit_observables:
        status_suffixes.append("nonpositive_fit_input")

    return PhaseScalingDiagnostic(
        system_sizes=tuple(sizes),
        control_values=tuple(controls),
        replicate_count=int(order.shape[2]),
        pseudocritical_control=tuple(
            controls[indices[0]] if len(indices) == 1 else None
            for indices in maximizing_indices
        ),
        pseudocritical_controls=tuple(
            tuple(controls[index] for index in indices)
            for indices in maximizing_indices
        ),
        peak_is_unique=unique_flags,
        all_peaks_unique=all(unique_flags),
        peak_at_control_boundary=boundary_flags,
        all_peaks_interior=not any(boundary_flags),
        peak_susceptibility=tuple(peak_susceptibility),
        order_at_peak=tuple(order_at_peak),
        coherence_length_at_peak=(
            None if xi_at_peak is None else tuple(xi_at_peak)
        ),
        peak_susceptibility_sem=tuple(susceptibility_sem),
        order_at_peak_sem=tuple(order_sem),
        coherence_length_at_peak_sem=(
            None if xi_sem is None else tuple(xi_sem)
        ),
        susceptibility_size_fit=susceptibility_fit,
        order_size_fit=order_fit,
        coherence_length_size_fit=xi_fit,
        nonpositive_fit_observables=nonpositive_fit_observables,
        status="_".join(("finite_size_measurement", *status_suffixes)),
        limitations=tuple(limitations),
    )


def _validate_sizes(system_sizes: Sequence[int]) -> list[int]:
    sizes: list[int] = []
    for value in system_sizes:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("system_sizes must contain integer node counts")
        sizes.append(int(value))
    if len(sizes) < 3:
        raise ValueError("at least three system sizes are required")
    if any(value <= 0 for value in sizes):
        raise ValueError("system_sizes must be positive")
    if any(right <= left for left, right in zip(sizes, sizes[1:])):
        raise ValueError("system_sizes must be strictly increasing")
    return sizes


def _validate_controls(control_values: Sequence[float]) -> list[float]:
    try:
        raw_controls = tuple(control_values)
    except TypeError as exc:
        raise ValueError(
            "control_values must contain finite real coordinates"
        ) from exc
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in raw_controls
    ):
        raise ValueError("control_values must contain finite real coordinates")
    try:
        controls = [float(value) for value in raw_controls]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "control_values must contain finite real coordinates"
        ) from exc
    if len(controls) < 2:
        raise ValueError("at least two control values are required")
    if not all(math.isfinite(value) for value in controls):
        raise ValueError("control_values must contain only finite values")
    if any(right <= left for left, right in zip(controls, controls[1:])):
        raise ValueError("control_values must be strictly increasing")
    return controls


def _as_replicated_array(
    name: str,
    values: object,
    expected_prefix: tuple[int, int],
) -> object:
    try:
        object_array = np.asarray(values, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in object_array.flat
    ):
        raise ValueError(
            f"{name} must contain real numeric values, not booleans or text"
        )
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3 or array.shape[:2] != expected_prefix:
        raise ValueError(
            f"{name} must have shape {expected_prefix} or "
            f"({expected_prefix[0]}, {expected_prefix[1]}, n_replicates)"
        )
    if array.shape[2] < 1:
        raise ValueError(f"{name} must contain at least one replicate")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _standard_error(values: object) -> float | None:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return None
    mean = float(_stable_nonnegative_mean(array, axis=0))
    deviations = np.abs(array - mean)
    scale = float(np.max(deviations, initial=0.0))
    if scale == 0.0:
        return 0.0
    normalized_norm = math.hypot(*(float(value / scale) for value in deviations))
    coefficient = normalized_norm / math.sqrt(len(array) * (len(array) - 1))
    result = scale * coefficient
    if not math.isfinite(result):
        raise ValueError("replicate standard error exceeds floating-point range")
    if result == 0.0 and scale != 0.0:
        raise ValueError(
            "replicate standard error is below nonzero floating-point range"
        )
    return result


def _stable_nonnegative_mean(values: object, *, axis: int) -> object:
    """Average finite nonnegative values without overflowing their sum."""
    array = np.asarray(values, dtype=float)
    scale = np.max(array, axis=axis, keepdims=True)
    normalized = np.divide(
        array,
        scale,
        out=np.zeros_like(array, dtype=float),
        where=scale != 0.0,
    )
    fraction = np.minimum(np.mean(normalized, axis=axis), 1.0)
    result = fraction * np.squeeze(scale, axis=axis)
    if not np.all(np.isfinite(result)):
        raise ValueError("observable mean exceeds floating-point range")
    has_positive_input = np.any(array > 0.0, axis=axis)
    if np.any((result == 0.0) & has_positive_input):
        raise ValueError("observable mean is below nonzero floating-point range")
    return result


def _fit_positive_size_power_law(
    sizes: object,
    observable: object,
) -> SizePowerLawFit | None:
    x_values = np.asarray(sizes, dtype=float)
    y_values = np.asarray(observable, dtype=float)
    if np.any(x_values <= 0.0) or np.any(y_values <= 0.0):
        return None

    log_x = np.log(x_values)
    log_y = np.log(y_values)
    design = np.vstack([log_x, np.ones_like(log_x)]).T
    coefficients = np.linalg.lstsq(design, log_y, rcond=None)[0]
    predicted = design @ coefficients
    residual = log_y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (
        0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    )

    point_count = len(log_x)
    slope_standard_error: float | None = None
    centered = log_x - np.mean(log_x)
    denominator = float(np.sum(centered**2))
    if point_count > 2 and denominator > 0.0:
        residual_variance = ss_res / (point_count - 2)
        slope_standard_error = math.sqrt(max(residual_variance, 0.0) / denominator)

    return SizePowerLawFit(
        slope=float(coefficients[0]),
        intercept=float(coefficients[1]),
        r_squared=float(r_squared),
        slope_standard_error=slope_standard_error,
        point_count=point_count,
    )


__all__ = [
    "SizePowerLawFit",
    "PhaseScalingDiagnostic",
    "analyze_phase_finite_size_scaling",
]
