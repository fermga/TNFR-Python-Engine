"""Tetrad snapshot collection for rich telemetry.

This module provides purely observational telemetry for the four canonical
structural fields (Φ_s, |∇φ|, K_φ, ξ_C) without modifying operator decisions
or TNFR physics (U1-U6).

The tetrad snapshot density is controlled by `telemetry_density` config:
- "low": Basic statistics (mean, max, min)
- "medium": Add percentiles (p25, p50, p75)
- "high": Full distribution (p10, p90, p99, histograms)

Physics Invariance:
- Telemetry is READ-ONLY
- Does NOT affect operator sequences or grammar decisions
- Does NOT modify C(t), ΔNFR, or any structural dynamics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import get_telemetry_density
from ..physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)

if TYPE_CHECKING:
    import networkx as nx


def collect_tetrad_snapshot(
    G: nx.Graph,
    include_histograms: bool | None = None,
) -> dict[str, Any]:
    """Collect observational snapshot of canonical tetrad fields.

    Parameters
    ----------
    G : nx.Graph
        TNFR network with node attributes: 'ΔNFR', 'theta' (or 'phase')
    include_histograms : bool, optional
        Override telemetry_density to force histogram inclusion.
        If None, uses telemetry_density config.

    Returns
    -------
    dict
        Snapshot with keys:
        - 'phi_s': Structural potential statistics
        - 'phase_grad': Phase gradient statistics
        - 'phase_curv': Phase curvature statistics
        - 'xi_c': Coherence length (scalar or None)
        - 'metadata': Timestamp, telemetry_density, node_count

    Notes
    -----
    This function is PURELY OBSERVATIONAL:
    - Does NOT modify G or any node attributes
    - Does NOT affect operator sequences or grammar (U1-U6)
    - Does NOT change C(t), Si, or structural dynamics
    """
    density = get_telemetry_density()

    # Determine histogram inclusion
    if include_histograms is None:
        include_histograms = density == "high"

    # Collect field values
    phi_s_values = compute_structural_potential(G)  # Per-node Φ_s
    grad_values = compute_phase_gradient(G)  # Per-node |∇φ|
    curv_values = compute_phase_curvature(G)  # Per-node K_φ

    # Build snapshot
    snapshot: dict[str, Any] = {
        "phi_s": _field_statistics(phi_s_values, density, include_histograms),
        "phase_grad": _field_statistics(
            grad_values, density, include_histograms
        ),
        "phase_curv": _field_statistics(
            curv_values, density, include_histograms
        ),
        "xi_c": None,  # Filled below
        "metadata": {
            "telemetry_density": density,
            "node_count": G.number_of_nodes(),
        },
    }

    # Coherence length (expensive, single global value)
    try:
        xi_c = estimate_coherence_length(G)
        snapshot["xi_c"] = float(xi_c) if np.isfinite(xi_c) else None
    except Exception:
        snapshot["xi_c"] = None

    return snapshot


def _field_statistics(
    values: dict[int, float],
    density: str,
    include_histograms: bool,
) -> dict[str, Any]:
    """Compute statistics for a field based on telemetry density.

    Parameters
    ----------
    values : dict[int, float]
        Per-node field values
    density : str
        "low" | "medium" | "high"
    include_histograms : bool
        Whether to include histogram data

    Returns
    -------
    dict
        Statistics appropriate for density level
    """
    if not values:
        return {"mean": None, "max": None, "min": None}

    arr = np.array(list(values.values()), dtype=np.float64)
    arr = arr[np.isfinite(arr)]  # Filter out nan/inf

    if len(arr) == 0:
        return {"mean": None, "max": None, "min": None}

    # Basic statistics (all density levels)
    stats: dict[str, Any] = {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "std": float(np.std(arr)),
    }

    # Medium: Add quartiles
    if density in ("medium", "high"):
        stats["p25"] = float(np.percentile(arr, 25))
        stats["p50"] = float(np.percentile(arr, 50))
        stats["p75"] = float(np.percentile(arr, 75))

    # High: Add tail percentiles
    if density == "high":
        stats["p10"] = float(np.percentile(arr, 10))
        stats["p90"] = float(np.percentile(arr, 90))
        stats["p99"] = float(np.percentile(arr, 99))

    # Histograms (if requested)
    if include_histograms:
        counts, edges = np.histogram(arr, bins=20)
        stats["histogram"] = {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
        }

    return stats


def get_tetrad_sample_interval(base_dt: float = 1.0) -> float:
    """Compute snapshot interval based on telemetry_density.

    Parameters
    ----------
    base_dt : float
        Base timestep of simulation

    Returns
    -------
    float
        Interval (in simulation time) between tetrad snapshots

    Notes
    -----
    Sampling strategy:
    - "low": Every 10 steps (10 × base_dt)
    - "medium": Every 5 steps (5 × base_dt)
    - "high": Every step (1 × base_dt)
    """
    density = get_telemetry_density()

    if density == "high":
        return base_dt
    elif density == "medium":
        return 5.0 * base_dt
    else:  # "low"
        return 10.0 * base_dt
