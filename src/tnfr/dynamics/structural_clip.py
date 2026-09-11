"""Boundary projection for real scalar EPI coordinates.

The nodal integrators call this module after evaluating
``dEPI/dt = nu_f * DeltaNFR``. Clipping is therefore a numerical boundary
policy, not an additional pressure term. Hard mode is the ordinary interval
projection. Soft mode preserves a central identity region and replaces only
the neighbourhood of each boundary by a continuously differentiable cubic
knee.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from ..config.defaults_core import CoreDefaults
from ..mathematics.unified_numerical import np

__all__ = [
    "structural_clip",
    "structural_clip_array",
    "StructuralClipStats",
    "get_clip_stats",
    "reset_clip_stats",
]


class StructuralClipStats:
    """Telemetry for structural boundary interventions."""

    def __init__(self) -> None:
        """Initialize empty statistics."""
        self.hard_clips: int = 0
        self.soft_clips: int = 0
        self.total_adjustments: int = 0
        self.max_delta_hard: float = 0.0
        self.max_delta_soft: float = 0.0
        self.sum_delta_hard: float = 0.0
        self.sum_delta_soft: float = 0.0

    def record_hard_clip(self, delta: float) -> None:
        """Record a hard-clip intervention."""
        self.hard_clips += 1
        self.total_adjustments += 1
        abs_delta = abs(delta)
        self.max_delta_hard = max(self.max_delta_hard, abs_delta)
        self.sum_delta_hard += abs_delta

    def record_soft_clip(self, delta: float) -> None:
        """Record a soft-clip intervention."""
        self.soft_clips += 1
        self.total_adjustments += 1
        abs_delta = abs(delta)
        self.max_delta_soft = max(self.max_delta_soft, abs_delta)
        self.sum_delta_soft += abs_delta

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.hard_clips = 0
        self.soft_clips = 0
        self.total_adjustments = 0
        self.max_delta_hard = 0.0
        self.max_delta_soft = 0.0
        self.sum_delta_hard = 0.0
        self.sum_delta_soft = 0.0

    def summary(self) -> dict[str, float | int]:
        """Return summary statistics as a detached dictionary."""
        return {
            "hard_clips": self.hard_clips,
            "soft_clips": self.soft_clips,
            "total_adjustments": self.total_adjustments,
            "max_delta_hard": self.max_delta_hard,
            "max_delta_soft": self.max_delta_soft,
            "avg_delta_hard": (
                self.sum_delta_hard / self.hard_clips if self.hard_clips else 0.0
            ),
            "avg_delta_soft": (
                self.sum_delta_soft / self.soft_clips if self.soft_clips else 0.0
            ),
        }


_global_stats = StructuralClipStats()


def get_clip_stats() -> StructuralClipStats:
    """Return the process-local clipping telemetry object."""
    return _global_stats


def reset_clip_stats() -> None:
    """Reset process-local clipping telemetry."""
    _global_stats.reset()


def _finite_real(value: Any, name: str) -> float:
    """Return a finite real scalar without accepting coercive text or booleans."""
    if isinstance(value, (bool, str, bytes, complex)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _validated_contract(
    lo: Any,
    hi: Any,
    mode: Any,
    k: Any,
) -> tuple[float, float, Literal["hard", "soft"], float]:
    """Validate the shared scalar/array clipping parameters."""
    lower = _finite_real(lo, "lo")
    upper = _finite_real(hi, "hi")
    if lower > upper:
        raise ValueError(f"Lower bound {lower} must be <= upper bound {upper}")
    if mode not in ("hard", "soft"):
        raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
    steepness = _finite_real(k, "k")
    if steepness <= 0.0:
        raise ValueError("k must be greater than zero")
    return lower, upper, mode, steepness


def _interval_geometry(lo: float, hi: float) -> tuple[float, float]:
    """Compute midpoint and half-width without overflowing a finite interval."""
    return lo / 2.0 + hi / 2.0, hi / 2.0 - lo / 2.0


def _soft_clip_validated(value: float, lo: float, hi: float, k: float) -> float:
    """Evaluate the validated C1 soft-knee projection for one scalar."""
    if value <= lo:
        return lo
    if value >= hi:
        return hi
    if lo == hi:
        return lo

    midpoint, half_width = _interval_geometry(lo, hi)
    magnitude = abs((value - midpoint) / half_width)

    # ``exp(-k)`` is the normalized knee width, so increasing k approaches
    # hard projection. ``-expm1(-k)`` retains accuracy for small positive k.
    knee_width = math.exp(-k)
    identity_limit = -math.expm1(-k)
    if magnitude <= identity_limit or knee_width == 0.0:
        return value

    transition = (magnitude - identity_limit) / knee_width
    # Hermite knee p(0)=0, p'(0)=1, p(1)=1, p'(1)=0.
    eased = transition + transition * transition - transition**3
    projected_magnitude = identity_limit + knee_width * eased
    projected = midpoint + math.copysign(
        projected_magnitude * half_width, value - midpoint
    )
    return max(lo, min(hi, projected))


def structural_clip(
    value: float,
    lo: float = -1.0,
    hi: float = 1.0,
    mode: Literal["hard", "soft"] = "hard",
    k: float = CoreDefaults().CLIP_SOFT_K,
    *,
    record_stats: bool = False,
) -> float:
    """Project one finite real EPI coordinate into ``[lo, hi]``.

    Hard mode returns ``min(hi, max(lo, value))``. Soft mode first normalizes
    the interval to ``[-1, 1]``. With ``a = 1 - exp(-k)``, magnitudes at or
    below ``a`` are unchanged. The remaining interval uses the cubic Hermite
    knee ``p(t) = t + t**2 - t**3`` and values outside the interval project to
    the nearest boundary. The resulting map is monotone, odd about the
    interval midpoint, bounded, and continuously differentiable at both the
    knee and the boundary. Larger ``k`` narrows the knee and approaches hard
    projection.

    ``value``, both bounds, and ``k`` must be finite real scalars; booleans,
    text, and complex values are rejected. ``k`` must be positive. These
    checks are identical in hard and soft mode so configuration errors cannot
    remain latent until a later mode change.

    Examples
    --------
    >>> structural_clip(1.1, -1.0, 1.0, mode="hard")
    1.0
    >>> structural_clip(-1.2, -1.0, 1.0, mode="soft")
    -1.0
    >>> structural_clip(0.95, -1.0, 1.0, mode="soft")
    0.95
    """
    lower, upper, resolved_mode, steepness = _validated_contract(lo, hi, mode, k)
    scalar = _finite_real(value, "value")

    if resolved_mode == "hard":
        clipped = max(lower, min(upper, scalar))
    else:
        clipped = _soft_clip_validated(scalar, lower, upper, steepness)

    if record_stats and clipped != scalar:
        if resolved_mode == "hard":
            _global_stats.record_hard_clip(clipped - scalar)
        else:
            _global_stats.record_soft_clip(clipped - scalar)
    return clipped


def structural_clip_array(
    values: Any,
    lo: float = -1.0,
    hi: float = 1.0,
    mode: Literal["hard", "soft"] = "hard",
    k: float = CoreDefaults().CLIP_SOFT_K,
):
    """Apply :func:`structural_clip` elementwise without mutating ``values``."""
    lower, upper, resolved_mode, steepness = _validated_contract(lo, hi, mode, k)
    if np is None:  # pragma: no cover - NumPy is an engine dependency
        raise RuntimeError("structural_clip_array requires NumPy")
    try:
        source = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("values must contain finite real numbers") from exc
    if source.dtype.kind not in "iuf" or not bool(np.all(np.isfinite(source))):
        raise ValueError("values must contain finite real numbers")
    array = np.asarray(source, dtype=float)

    if resolved_mode == "hard":
        return np.clip(array, lower, upper)
    if lower == upper:
        return np.full_like(array, lower, dtype=float)

    midpoint, half_width = _interval_geometry(lower, upper)
    clipped = np.clip(array, lower, upper)
    magnitude = np.abs((clipped - midpoint) / half_width)
    knee_width = math.exp(-steepness)
    identity_limit = -math.expm1(-steepness)
    if knee_width == 0.0:
        return clipped

    transition_mask = (magnitude > identity_limit) & (magnitude < 1.0)
    transition = np.zeros_like(array, dtype=float)
    transition[transition_mask] = (
        magnitude[transition_mask] - identity_limit
    ) / knee_width
    eased = transition + transition * transition - transition**3
    projected_magnitude = identity_limit + knee_width * eased
    projected = midpoint + np.copysign(
        projected_magnitude * half_width, clipped - midpoint
    )
    result = clipped.copy()
    result[transition_mask] = projected[transition_mask]
    return np.clip(result, lower, upper)
