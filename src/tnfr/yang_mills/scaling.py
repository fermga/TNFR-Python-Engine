r"""Y4 finite scaling diagnostics for TNFR structural gauge gaps.

Y4 studies how the finite Y1 structural gauge gap behaves across graph-size
surrogates while YMG-4 (non-Abelian derivability) remains open.  This module
therefore reports finite diagnostic evidence only.  It does not construct a
continuum limit and does not prove a Clay-strength Yang–Mills mass gap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ..constants.canonical import U6_STRUCTURAL_POTENTIAL_LIMIT
from ..mathematics.unified_numerical import np
from ..physics.gauge import compute_yang_mills_equations
from .structural_gap import build_structural_gauge_graph, compute_structural_gauge_gap
from .u6_sweep import _rescale_delta_nfr_to_u6_ratio


@dataclass(frozen=True)
class FiniteScalingPoint:
    """One graph-size point in the Y4 finite scaling diagnostic."""

    topology: str
    n: int
    seed: int
    target_u6_ratio: float
    observed_u6_ratio: float
    gap: float
    lambda0: float
    lambda1: float
    gap_verdict: str
    is_self_adjoint: bool
    gauge_invariant: bool
    gauge_spectral_deviation: float
    mean_yang_mills_residual: float
    max_yang_mills_residual: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FiniteScalingReport:
    """Aggregate Y4 finite scaling report."""

    points: tuple[FiniteScalingPoint, ...]
    grouped_scaling: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    verdict: str


def run_finite_scaling_study(
    *,
    n_values: Iterable[int] = (8, 12, 16),
    topologies: Iterable[str] = ("cycle", "complete"),
    seeds: Iterable[int] = (42, 43),
    target_u6_ratios: Iterable[float] = (0.75,),
    phase_spread: float = 0.05,
    gauge_seed: int = 42,
    curvature_weight: float = 1.0,
    confinement_weight: float = 1.0,
    tolerance: float = 1e-10,
    eigen_tolerance: float = 1e-9,
) -> FiniteScalingReport:
    """Run the Y4 finite graph-size scaling diagnostic.

    The size coordinate is the graph node count ``n``.  For each topology and
    U6 target, the report fits the finite log-log slope of mean gap vs. ``n``.
    Positive gaps across sampled sizes yield ``FINITE_SCALING_EVIDENCE``;
    observed gaps at or below tolerance yield ``GAP_COLLAPSE_OBSERVED``.
    """
    n_list = tuple(int(value) for value in n_values)
    topology_list = tuple(topologies)
    seed_list = tuple(int(value) for value in seeds)
    ratio_list = tuple(float(value) for value in target_u6_ratios)
    _validate_scaling_inputs(n_list, topology_list, seed_list, ratio_list)

    points: list[FiniteScalingPoint] = []
    for topology in topology_list:
        for target_ratio in ratio_list:
            for n in n_list:
                for seed in seed_list:
                    graph = build_structural_gauge_graph(
                        n,
                        topology=topology,
                        seed=seed,
                        phase_spread=phase_spread,
                    )
                    _rescale_delta_nfr_to_u6_ratio(graph, target_ratio)
                    result = compute_structural_gauge_gap(
                        graph,
                        gauge_seed=gauge_seed,
                        tolerance=tolerance,
                        eigen_tolerance=eigen_tolerance,
                        curvature_weight=curvature_weight,
                        confinement_weight=confinement_weight,
                    )
                    ym_eq = compute_yang_mills_equations(graph)
                    max_abs_phi_s = float(result.metadata.get("max_abs_phi_s", 0.0))
                    observed_ratio = (
                        max_abs_phi_s / U6_STRUCTURAL_POTENTIAL_LIMIT
                        if U6_STRUCTURAL_POTENTIAL_LIMIT
                        else 0.0
                    )
                    metadata = dict(result.metadata)
                    metadata.update(
                        {
                            "target_u6_ratio": float(target_ratio),
                            "observed_u6_ratio": float(observed_ratio),
                            "finite_scope": "Y4_finite_scaling_only",
                            "size_coordinate": "node_count",
                        }
                    )

                    points.append(
                        FiniteScalingPoint(
                            topology=topology,
                            n=int(n),
                            seed=int(seed),
                            target_u6_ratio=float(target_ratio),
                            observed_u6_ratio=float(observed_ratio),
                            gap=float(result.gap),
                            lambda0=float(result.lambda0),
                            lambda1=float(result.lambda1),
                            gap_verdict=result.verdict,
                            is_self_adjoint=result.is_self_adjoint,
                            gauge_invariant=result.gauge_invariant,
                            gauge_spectral_deviation=float(
                                result.gauge_spectral_deviation
                            ),
                            mean_yang_mills_residual=float(ym_eq.mean_residual),
                            max_yang_mills_residual=float(ym_eq.max_residual),
                            metadata=metadata,
                        )
                    )

    points_tuple = tuple(points)
    grouped_scaling = _group_scaling(points_tuple, eigen_tolerance)
    verdict = _classify_scaling(points_tuple, eigen_tolerance)
    summary = _summarise_scaling(points_tuple, grouped_scaling, verdict)
    return FiniteScalingReport(
        points=points_tuple,
        grouped_scaling=grouped_scaling,
        summary=summary,
        verdict=verdict,
    )


def _validate_scaling_inputs(
    n_values: tuple[int, ...],
    topologies: tuple[str, ...],
    seeds: tuple[int, ...],
    ratios: tuple[float, ...],
) -> None:
    if not n_values:
        raise ValueError("n_values must contain at least one graph size")
    if not topologies:
        raise ValueError("topologies must contain at least one topology")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if not ratios:
        raise ValueError("target_u6_ratios must contain at least one value")
    if any(n < 2 for n in n_values):
        raise ValueError("all graph sizes must be at least 2")
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("target_u6_ratios must be non-negative")


def _group_scaling(
    points: tuple[FiniteScalingPoint, ...],
    eigen_tolerance: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, float], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for point in points:
        grouped[(point.topology, point.target_u6_ratio)][point.n].append(point.gap)

    result: dict[str, dict[str, Any]] = {}
    for (topology, target_ratio), by_n in grouped.items():
        n_sorted = sorted(by_n)
        mean_gaps = [float(np.mean(by_n[n])) for n in n_sorted]
        slope = _loglog_slope(n_sorted, mean_gaps)
        min_gap = min(mean_gaps) if mean_gaps else 0.0
        key = f"{topology}|rho_u6={target_ratio:g}"
        result[key] = {
            "topology": topology,
            "target_u6_ratio": float(target_ratio),
            "n_values": n_sorted,
            "mean_gaps": mean_gaps,
            "min_mean_gap": float(min_gap),
            "loglog_slope": slope,
            "gap_decay_exponent": -slope if slope is not None else None,
            "positive_at_all_sizes": bool(mean_gaps and min_gap > eigen_tolerance),
            "scope": "finite_group_scaling_not_continuum_limit",
        }
    return result


def _classify_scaling(
    points: tuple[FiniteScalingPoint, ...],
    eigen_tolerance: float,
) -> str:
    if not points:
        return "NO_SCALING_POINTS"
    if any(not point.is_self_adjoint for point in points):
        return "SCALING_FAILED_NON_SELF_ADJOINT"
    if any(not point.gauge_invariant for point in points):
        return "SCALING_FAILED_GAUGE_VARIANCE"
    if any(point.gap <= eigen_tolerance for point in points):
        return "GAP_COLLAPSE_OBSERVED"
    return "FINITE_SCALING_EVIDENCE"


def _summarise_scaling(
    points: tuple[FiniteScalingPoint, ...],
    grouped_scaling: dict[str, dict[str, Any]],
    verdict: str,
) -> dict[str, Any]:
    gaps = [point.gap for point in points]
    return {
        "n_points": len(points),
        "n_groups": len(grouped_scaling),
        "min_gap": min(gaps) if gaps else 0.0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap": max(gaps) if gaps else 0.0,
        "all_self_adjoint": all(point.is_self_adjoint for point in points),
        "all_gauge_invariant": all(point.gauge_invariant for point in points),
        "mean_yang_mills_residual": _mean(
            point.mean_yang_mills_residual for point in points
        ),
        "max_yang_mills_residual": max(
            (point.max_yang_mills_residual for point in points),
            default=0.0,
        ),
        "verdict": verdict,
        "scope": "finite_scaling_diagnostic_not_clay_proof",
    }


def _loglog_slope(ns: list[int], gaps: list[float]) -> float | None:
    positive_pairs = [(n, gap) for n, gap in zip(ns, gaps) if gap > 0.0]
    if len(positive_pairs) < 2:
        return None
    x = np.log(np.array([n for n, _gap in positive_pairs], dtype=float))
    y = np.log(np.array([gap for _n, gap in positive_pairs], dtype=float))
    if float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return None
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(np.mean(vals)) if vals else 0.0
