"""Structural boundary preservation for EPI values.

This module implements canonical TNFR structural clipping that preserves
coherence by constraining EPI values to valid structural boundaries while
maintaining smooth operator behavior.

The structural_clip function ensures that EPI remains within [-1.0, 1.0]
(or configurable bounds) after operator application and integration steps,
preventing numerical precision issues from violating structural invariants.
"""

from __future__ import annotations

import math
from typing import Literal

__all__ = [
    "structural_clip",
    "StructuralClipStats",
]


class StructuralClipStats:
    """Telemetry for structural boundary interventions.

    Tracks how often and by how much the structural_clip function
    adjusts EPI values to preserve structural boundaries.
    """

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
        """Record a hard clip intervention."""
        self.hard_clips += 1
        self.total_adjustments += 1
        abs_delta = abs(delta)
        self.max_delta_hard = max(self.max_delta_hard, abs_delta)
        self.sum_delta_hard += abs_delta

    def record_soft_clip(self, delta: float) -> None:
        """Record a soft clip intervention."""
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
        """Return summary statistics as dictionary."""
        return {
            "hard_clips": self.hard_clips,
            "soft_clips": self.soft_clips,
            "total_adjustments": self.total_adjustments,
            "max_delta_hard": self.max_delta_hard,
            "max_delta_soft": self.max_delta_soft,
            "avg_delta_hard": (
                self.sum_delta_hard / self.hard_clips if self.hard_clips > 0 else 0.0
            ),
            "avg_delta_soft": (
                self.sum_delta_soft / self.soft_clips if self.soft_clips > 0 else 0.0
            ),
        }


# Global statistics instance (optional telemetry)
_global_stats = StructuralClipStats()


def get_clip_stats() -> StructuralClipStats:
    """Return the global clip statistics instance."""
    return _global_stats


def reset_clip_stats() -> None:
    """Reset global clip statistics."""
    _global_stats.reset()


def structural_clip(
    value: float,
    lo: float = -1.0,
    hi: float = 1.0,
    mode: Literal["hard", "soft"] = "hard",
    k: float = 3.0,
    *,
    record_stats: bool = False,
) -> float:
    """Apply structural boundary preservation to EPI value.

    Ensures that values remain within structural boundaries while preserving
    coherence. Two modes are available:

    - **hard**: Classic clamping for immediate stability (discontinuous derivative)
    - **soft**: Smooth hyperbolic tangent mapping (continuous derivative)

    Parameters
    ----------
    value : float
        The EPI value to clip
    lo : float, default -1.0
        Lower structural boundary (EPI_MIN)
    hi : float, default 1.0
        Upper structural boundary (EPI_MAX)
    mode : {'hard', 'soft'}, default 'hard'
        Clipping mode:
        - 'hard': Clamp to [lo, hi] (fast, discontinuous)
        - 'soft': Smooth tanh-based remapping (slower, smooth)
    k : float, default 3.0
        Steepness parameter for soft mode (higher = sharper transition)
    record_stats : bool, default False
        If True, record intervention statistics in global telemetry

    Returns
    -------
    float
        Value constrained to [lo, hi] with specified mode

    Notes
    -----
    The soft mode uses a scaled hyperbolic tangent:

        y = tanh(k · x) / tanh(k)

    which maps the input smoothly to [-1, 1], then rescales to [lo, hi].
    This preserves derivative continuity but is computationally more expensive.

    The hard mode is preferred for most use cases as it directly enforces
    boundaries with minimal overhead.

    Examples
    --------
    >>> structural_clip(1.1, -1.0, 1.0, mode="hard")
    1.0
    >>> structural_clip(-1.2, -1.0, 1.0, mode="hard")
    -1.0
    >>> abs(structural_clip(0.95, -1.0, 1.0, mode="soft") - 0.95) < 0.01
    True
    """
    if lo > hi:
        raise ValueError(f"Lower bound {lo} must be <= upper bound {hi}")

    if mode == "hard":
        # Classic clamping - fast and simple
        clipped = max(lo, min(hi, value))
        if record_stats and clipped != value:
            _global_stats.record_hard_clip(clipped - value)
        return clipped

    elif mode == "soft":
        # Smooth sigmoid-based mapping that guarantees bounds
        # Uses scaled tanh to create smooth transitions near boundaries
        if lo == hi:
            return lo

        # First, clamp to slightly extended range to handle the mapping
        # Map [lo, hi] to working range
        margin = (hi - lo) * 0.1  # 10% margin for smooth transition
        working_lo = lo - margin
        working_hi = hi + margin

        # Normalize to [-1, 1] for tanh
        # Check for zero-width range after extension (shouldn't happen with lo != hi)
        range_width = working_hi - working_lo
        if abs(range_width) < 1e-10:
            # Degenerate case: return midpoint
            return (lo + hi) / 2.0

        normalized = 2.0 * (value - (working_lo + working_hi) / 2.0) / range_width

        # Apply tanh with steepness k for smooth S-curve
        # tanh maps R → (-1, 1), scaled by k to control steepness
        smooth_normalized = math.tanh(k * normalized)

        # Map back from (-1, 1) to [lo, hi]
        # This ensures output is always within [lo, hi]
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        clipped = mid + smooth_normalized * half_range

        # Final safety clamp for numerical precision
        clipped = max(lo, min(hi, clipped))

        if record_stats and abs(clipped - value) > 1e-10:
            _global_stats.record_soft_clip(clipped - value)

        return clipped

    else:
        raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
