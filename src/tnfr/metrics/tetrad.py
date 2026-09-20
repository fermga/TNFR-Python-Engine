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

import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ..config import get_telemetry_density
from ..mathematics.unified_numerical import np
from ..physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
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
        - 'xi_c_available', 'xi_c_provenance', 'xi_c_error': Estimator evidence
        - 'metadata': telemetry_density, node_count

    Notes
    -----
    This function is PURELY OBSERVATIONAL:
    - Does NOT evolve nodal state; field owners may maintain graph caches
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
        "phase_grad": _field_statistics(grad_values, density, include_histograms),
        "phase_curv": _field_statistics(curv_values, density, include_histograms),
        "xi_c": None,  # Filled below
        "xi_c_available": False,
        "xi_c_provenance": None,
        "xi_c_error": None,
        "metadata": {
            "telemetry_density": density,
            "node_count": G.number_of_nodes(),
        },
    }

    # Coherence length (expensive, single global value)
    try:
        estimate = estimate_coherence_length_with_provenance(G)
        provenance = asdict(estimate)
        xi_c = provenance.pop("value")
        snapshot["xi_c_provenance"] = provenance
        if math.isfinite(xi_c) and xi_c > 0.0:
            snapshot["xi_c"] = float(xi_c)
            snapshot["xi_c_available"] = True
        else:
            snapshot["xi_c_error"] = {
                "type": "UnavailableCoherenceLength",
                "message": "Estimator supplied no finite positive correlation scale",
            }
    except Exception as error:
        snapshot["xi_c_error"] = {"type": type(error).__name__, "message": str(error)}

    return snapshot


def _field_statistics(
    values: dict[int, float],
    density: str,
    include_histograms: bool,
) -> dict[str, Any]:
    """Summarize complete admitted fields; never filter out invalid nodes.

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
        Statistics appropriate for density level. Optional numerical results
        outside finite range are None with per-statistic unavailability evidence.
    """
    # Import locally: unified fields reuse metrics.common during package startup.
    from ..physics.unified import summary_statistics

    basic = summary_statistics({"field": values}).get("field")
    if basic is None:
        return {
            "mean": None,
            "max": None,
            "min": None,
            "std": None,
            "available": False,
            "error": {
                "type": "UnavailableFieldStatistics",
                "message": (
                    "Empty field"
                    if not values
                    else "Invalid field; no nodes were filtered"
                ),
            },
        }

    # Basic statistics (all density levels)
    stats: dict[str, Any] = {key: basic[key] for key in ("mean", "max", "min", "std")}
    stats.update(available=True, error=None)
    arr = np.array(list(values.values()), dtype=np.float64)
    unavailable: dict[str, dict[str, str]] = {}

    # Medium: Add quartiles
    percentiles = []
    if density in ("medium", "high"):
        percentiles.extend((25, 50, 75))

    # High: Add tail percentiles
    if density == "high":
        percentiles.extend((10, 90, 99))
    if percentiles:
        with np.errstate(over="ignore", invalid="ignore"):
            quantiles = np.percentile(arr, percentiles)
        for percentile, value in zip(percentiles, quantiles):
            key = f"p{percentile}"
            if math.isfinite(value):
                stats[key] = float(value)
            else:
                stats[key] = None
                unavailable[key] = {
                    "type": "UnrepresentablePercentileArithmetic",
                    "message": "Configured percentile calculation produced a nonfinite result",
                }

    # Histograms (if requested)
    if include_histograms:
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                counts, edges = np.histogram(arr, bins=20)
            if not bool(np.all(np.isfinite(edges))) or not bool(
                np.all(edges[1:] > edges[:-1])
            ):
                raise ValueError("Histogram edges are not finite distinct boundaries")
            if int(counts.sum()) != len(arr):
                raise ValueError("Histogram did not retain all field samples")
            stats["histogram"] = {"counts": counts.tolist(), "edges": edges.tolist()}
        except (ValueError, OverflowError, FloatingPointError, IndexError) as error:
            stats["histogram"] = None
            unavailable["histogram"] = {
                "type": type(error).__name__,
                "message": str(error),
            }

    if unavailable:
        stats["unavailable_statistics"] = unavailable

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
