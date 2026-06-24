r"""Y2 U6 confinement sweep for TNFR structural gauge gaps.

This module implements the second TNFR–Yang–Mills milestone: sweep finite
structural gauge graphs across U6-confined and U6-unconfined regimes, then
measure how the Y1 gap diagnostic behaves.

The sweep is intentionally finite and empirical.  It records correlations and
failure modes, but it does not claim a continuum mass-gap theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..constants.canonical import PHI
from ..mathematics.unified_numerical import np
from ..physics.canonical import compute_structural_potential
from ..physics.gauge import compute_gauge_curvature, compute_yang_mills_equations
from .structural_gap import build_structural_gauge_graph, compute_structural_gauge_gap


@dataclass(frozen=True)
class U6ConfinementSweepPoint:
    """One finite graph point in the Y2 U6 confinement sweep."""

    topology: str
    n: int
    seed: int
    target_u6_ratio: float
    observed_u6_ratio: float
    u6_confined: bool
    gap: float
    lambda0: float
    lambda1: float
    gap_verdict: str
    is_self_adjoint: bool
    gauge_invariant: bool
    gauge_spectral_deviation: float
    yang_mills_action: float
    gauge_coupling_constant: float
    mean_yang_mills_residual: float
    max_yang_mills_residual: float
    mean_abs_curvature: float
    max_abs_curvature: float
    curvature_active: bool
    grammar_rules_satisfied: int | None
    grammar_rules_total: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class U6ConfinementSweepReport:
    """Aggregate report for the Y2 finite U6 confinement sweep."""

    points: tuple[U6ConfinementSweepPoint, ...]
    summary: dict[str, Any]
    verdict: str


def run_u6_confinement_sweep(
    *,
    n_values: Iterable[int] = (8, 12),
    topologies: Iterable[str] = ("cycle", "complete", "watts_strogatz"),
    seeds: Iterable[int] = (42, 43),
    target_u6_ratios: Iterable[float] = (0.25, 0.75, 1.25),
    phase_spread: float = 0.05,
    gauge_seed: int = 42,
    curvature_weight: float = 1.0,
    confinement_weight: float = 1.0,
    tolerance: float = 1e-10,
    eigen_tolerance: float = 1e-9,
) -> U6ConfinementSweepReport:
    """Run the Y2 finite U6 confinement sweep.

    Parameters are sampling controls for finite diagnostics only.  The U6
    target is expressed as ``max|Φ_s| / φ`` so the canonical threshold remains
    the single value ``1.0``.  Ratios below one are U6-confined; ratios at or
    above one intentionally probe unconfined structural-potential regimes.
    """
    n_list = tuple(n_values)
    topology_list = tuple(topologies)
    seed_list = tuple(seeds)
    ratio_list = tuple(float(value) for value in target_u6_ratios)

    _validate_sweep_inputs(n_list, topology_list, seed_list, ratio_list)

    points: list[U6ConfinementSweepPoint] = []
    for n in n_list:
        for topology in topology_list:
            for seed in seed_list:
                for target_ratio in ratio_list:
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
                    curvature = compute_gauge_curvature(graph)
                    curv_abs = [abs(float(value)) for value in curvature.values()]
                    mean_abs_curv = _mean(curv_abs)
                    max_abs_curv = max(curv_abs, default=0.0)
                    max_abs_phi_s = float(result.metadata.get("max_abs_phi_s", 0.0))
                    observed_ratio = max_abs_phi_s / PHI if PHI else 0.0

                    point_metadata = dict(result.metadata)
                    point_metadata.update(
                        {
                            "target_u6_ratio": float(target_ratio),
                            "observed_u6_ratio": float(observed_ratio),
                            "curvature_active_tolerance": float(tolerance),
                            "finite_scope": "Y2_empirical_finite_graph_only",
                        }
                    )

                    points.append(
                        U6ConfinementSweepPoint(
                            topology=topology,
                            n=int(n),
                            seed=int(seed),
                            target_u6_ratio=float(target_ratio),
                            observed_u6_ratio=float(observed_ratio),
                            u6_confined=bool(observed_ratio < 1.0),
                            gap=float(result.gap),
                            lambda0=float(result.lambda0),
                            lambda1=float(result.lambda1),
                            gap_verdict=result.verdict,
                            is_self_adjoint=result.is_self_adjoint,
                            gauge_invariant=result.gauge_invariant,
                            gauge_spectral_deviation=float(
                                result.gauge_spectral_deviation
                            ),
                            yang_mills_action=float(
                                result.metadata.get("yang_mills_action", 0.0)
                            ),
                            gauge_coupling_constant=float(
                                result.metadata.get(
                                    "gauge_coupling_constant",
                                    0.0,
                                )
                            ),
                            mean_yang_mills_residual=float(ym_eq.mean_residual),
                            max_yang_mills_residual=float(ym_eq.max_residual),
                            mean_abs_curvature=float(mean_abs_curv),
                            max_abs_curvature=float(max_abs_curv),
                            curvature_active=bool(max_abs_curv > tolerance),
                            grammar_rules_satisfied=result.metadata.get(
                                "grammar_rules_satisfied"
                            ),
                            grammar_rules_total=result.metadata.get(
                                "grammar_rules_total"
                            ),
                            metadata=point_metadata,
                        )
                    )

    summary = _summarise_points(tuple(points), eigen_tolerance)
    verdict = _classify_sweep(tuple(points))
    summary["verdict"] = verdict
    return U6ConfinementSweepReport(
        points=tuple(points),
        summary=summary,
        verdict=verdict,
    )


def _validate_sweep_inputs(
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
    if any(int(n) < 2 for n in n_values):
        raise ValueError("all graph sizes must be at least 2")
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("target_u6_ratios must be non-negative")


def _rescale_delta_nfr_to_u6_ratio(G: Any, target_ratio: float) -> None:
    """Scale ``ΔNFR`` so ``max|Φ_s| / φ`` matches ``target_ratio``."""
    target_abs_phi_s = float(target_ratio) * PHI
    if target_abs_phi_s == 0.0:
        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.0
        return

    current_max = _max_abs_structural_potential(G)
    if current_max <= 1e-15:
        _install_deterministic_delta_pattern(G)
        current_max = _max_abs_structural_potential(G)

    if current_max <= 1e-15:
        return

    scale = target_abs_phi_s / current_max
    for node in G.nodes():
        delta = float(G.nodes[node].get("delta_nfr", 0.0))
        G.nodes[node]["delta_nfr"] = float(delta * scale)


def _install_deterministic_delta_pattern(G: Any) -> None:
    nodes = tuple(G.nodes())
    if not nodes:
        return
    centre = (len(nodes) - 1) / 2.0
    normaliser = max(1.0, centre)
    for idx, node in enumerate(nodes):
        signed_offset = (idx - centre) / normaliser
        if abs(signed_offset) < 1e-15:
            signed_offset = 1.0
        G.nodes[node]["delta_nfr"] = float(signed_offset)


def _max_abs_structural_potential(G: Any) -> float:
    phi_s = compute_structural_potential(G)
    return max((abs(float(value)) for value in phi_s.values()), default=0.0)


def _summarise_points(
    points: tuple[U6ConfinementSweepPoint, ...],
    eigen_tolerance: float,
) -> dict[str, Any]:
    confined = tuple(point for point in points if point.u6_confined)
    unconfined = tuple(point for point in points if not point.u6_confined)
    positive = tuple(point for point in points if point.gap > eigen_tolerance)
    curvature_active = tuple(point for point in points if point.curvature_active)

    return {
        "n_points": len(points),
        "n_confined": len(confined),
        "n_unconfined": len(unconfined),
        "positive_gap_fraction": _fraction(len(positive), len(points)),
        "confined_positive_gap_fraction": _positive_gap_fraction(
            confined,
            eigen_tolerance,
        ),
        "unconfined_positive_gap_fraction": _positive_gap_fraction(
            unconfined,
            eigen_tolerance,
        ),
        "mean_gap": _mean(point.gap for point in points),
        "mean_confined_gap": _mean(point.gap for point in confined),
        "mean_unconfined_gap": _mean(point.gap for point in unconfined),
        "mean_yang_mills_residual": _mean(
            point.mean_yang_mills_residual for point in points
        ),
        "max_yang_mills_residual": max(
            (point.max_yang_mills_residual for point in points),
            default=0.0,
        ),
        "curvature_active_fraction": _fraction(
            len(curvature_active),
            len(points),
        ),
        "u6_gap_correlation": _pearson(
            [point.observed_u6_ratio for point in points],
            [point.gap for point in points],
        ),
        "scope": "finite_graph_y2_empirical_not_clay_proof",
    }


def _classify_sweep(points: tuple[U6ConfinementSweepPoint, ...]) -> str:
    if not points:
        return "NO_SWEEP_POINTS"
    if any(not point.is_self_adjoint for point in points):
        return "SWEEP_FAILED_NON_SELF_ADJOINT"
    if any(not point.gauge_invariant for point in points):
        return "SWEEP_FAILED_GAUGE_VARIANCE"
    return "EMPIRICAL_FINITE_GRAPH_ONLY"


def _positive_gap_fraction(
    points: tuple[U6ConfinementSweepPoint, ...],
    eigen_tolerance: float,
) -> float:
    positive = sum(1 for point in points if point.gap > eigen_tolerance)
    return _fraction(positive, len(points))


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return float(np.mean(vals)) if vals else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    if float(np.std(x_arr)) <= 1e-15 or float(np.std(y_arr)) <= 1e-15:
        return None
    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    if not np.isfinite(corr):
        return None
    return float(corr)
