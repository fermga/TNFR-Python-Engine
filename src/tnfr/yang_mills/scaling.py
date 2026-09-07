r"""Y4 finite scaling diagnostics for TNFR structural gauge gaps.

Y4 studies how the finite Y1 structural gauge gap behaves across graph-size
surrogates while YMG-4 (non-Abelian derivability) remains open.  This module
therefore reports finite diagnostic evidence only.  It does not construct a
continuum limit and does not prove a Clay-strength Yang–Mills mass gap.
The legacy ``U6`` ratio names denote the single-snapshot magnitude coordinate
``max|Phi_s|/(pi/2)``. They do not assess canonical U6 potential drift.
The separate π/4 per-node magnitude-warning ratio is exposed explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ..constants.canonical import (
    PHI_S_VON_KOCH_THRESHOLD,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)
from ..mathematics.unified_numerical import np
from ..physics.gauge import compute_yang_mills_equations
from .structural_gap import (
    _finite_nonnegative,
    _finite_positive,
    build_structural_gauge_graph,
    compute_structural_gauge_gap,
)
from .u6_sweep import (
    _rescale_delta_nfr_to_u6_ratio,
    _validated_integer_values,
    _validated_ratios,
    _validated_seeds,
    _validated_topologies,
)


@dataclass(frozen=True)
class FiniteScalingPoint:
    """One graph-size point in the Y4 finite scaling diagnostic.

    ``n`` retains the requested size for API compatibility;
    ``actual_n_nodes`` is the coordinate used by the scaling fit.
    """

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
    realized_n_nodes: int | None = None

    @property
    def actual_n_nodes(self) -> int:
        """Return the realized graph size while preserving legacy ``n``."""

        return self.n if self.realized_n_nodes is None else self.realized_n_nodes

    @property
    def potential_magnitude_ratio(self) -> float:
        """Return ``max|Phi_s|`` relative to the π/4 magnitude warning."""

        return (
            self.observed_u6_ratio
            * U6_STRUCTURAL_POTENTIAL_LIMIT
            / PHI_S_VON_KOCH_THRESHOLD
        )

    @property
    def potential_magnitude_ratio_to_u6_drift_scale(self) -> float:
        """Return the legacy single-snapshot ratio using π/2."""

        return self.observed_u6_ratio

    @property
    def potential_magnitude_within_warning(self) -> bool:
        """Whether ``max|Phi_s|`` is below the selected π/4 warning."""

        return self.potential_magnitude_ratio < 1.0

    @property
    def u6_drift_assessed(self) -> bool:
        """No reference snapshot is supplied by this scaling study."""
        return False

    @property
    def mean_pure_gauge_consistency_residual(self) -> float:
        """Accurate alias for the legacy Yang--Mills residual field."""

        return self.mean_yang_mills_residual

    @property
    def max_pure_gauge_consistency_residual(self) -> float:
        """Accurate alias for the legacy Yang--Mills residual field."""

        return self.max_yang_mills_residual

    @property
    def curvature_is_numerical_residual(self) -> bool:
        """Whether curvature fields only measure exact-one-form closure error."""

        return True


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

    The size coordinate is the realized graph node count ``n``. For each
    topology and legacy π/2-normalized magnitude target, the report fits the
    finite log-log slope of mean gap vs. ``n``.
    Positive gaps across sampled sizes yield ``FINITE_SCALING_EVIDENCE``;
    observed gaps at or below tolerance yield ``GAP_COLLAPSE_OBSERVED``.
    """
    n_list = _validated_integer_values(n_values, "n_values", minimum=2)
    topology_list = _validated_topologies(topologies)
    seed_list = _validated_seeds(seeds)
    ratio_list = _validated_ratios(target_u6_ratios)
    phase_spread = _finite_nonnegative(phase_spread, "phase_spread")
    curvature_weight = _finite_nonnegative(curvature_weight, "curvature_weight")
    confinement_weight = _finite_nonnegative(
        confinement_weight,
        "confinement_weight",
    )
    tolerance = _finite_positive(tolerance, "tolerance")
    eigen_tolerance = _finite_nonnegative(eigen_tolerance, "eigen_tolerance")

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
                    actual_n = int(graph.number_of_nodes())
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
                    magnitude_ratio = (
                        max_abs_phi_s / PHI_S_VON_KOCH_THRESHOLD
                        if PHI_S_VON_KOCH_THRESHOLD
                        else 0.0
                    )
                    metadata = dict(result.metadata)
                    metadata.update(
                        {
                            "target_u6_ratio": float(target_ratio),
                            "observed_u6_ratio": float(observed_ratio),
                            "requested_n": int(n),
                            "actual_n_nodes": actual_n,
                            "target_potential_magnitude_ratio": float(
                                target_ratio
                                * U6_STRUCTURAL_POTENTIAL_LIMIT
                                / PHI_S_VON_KOCH_THRESHOLD
                            ),
                            "observed_potential_magnitude_ratio": float(
                                magnitude_ratio
                            ),
                            "potential_magnitude_within_warning": bool(
                                magnitude_ratio < 1.0
                            ),
                            "potential_magnitude_warning_threshold": float(
                                PHI_S_VON_KOCH_THRESHOLD
                            ),
                            "legacy_u6_normalization_scale": float(
                                U6_STRUCTURAL_POTENTIAL_LIMIT
                            ),
                            "u6_drift_assessed": False,
                            "u6_reference_required": True,
                            "u6_aggregation": "mean_absolute_nodewise_drift",
                            "u6_comparison": "strict_less_than",
                            "u6_definition": (
                                "mean_i |Phi_s_after(i)-Phi_s_before(i)| "
                                "< threshold"
                            ),
                            "legacy_u6_fields_are_magnitude_proxies": True,
                            "connection_scope": (
                                "derived_exact_vertex_phase_one_form"
                            ),
                            "curvature_is_numerical_residual": True,
                            "yang_mills_names_are_legacy_consistency_diagnostics": (
                                True
                            ),
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
                            realized_n_nodes=actual_n,
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


def _group_scaling(
    points: tuple[FiniteScalingPoint, ...],
    eigen_tolerance: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, float], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for point in points:
        grouped[(point.topology, point.target_u6_ratio)][
            point.actual_n_nodes
        ].append(point.gap)

    result: dict[str, dict[str, Any]] = {}
    for (topology, target_ratio), by_n in grouped.items():
        n_sorted = sorted(by_n)
        mean_gaps = [float(np.mean(by_n[n])) for n in n_sorted]
        slope = _loglog_slope(n_sorted, mean_gaps)
        min_gap = min(mean_gaps) if mean_gaps else 0.0
        # Preserve the established public key while marking its legacy scope
        # inside the value metadata.
        key = f"{topology}|rho_u6={target_ratio:g}"
        result[key] = {
            "topology": topology,
            "target_u6_ratio": float(target_ratio),
            "target_potential_magnitude_ratio": float(
                target_ratio
                * U6_STRUCTURAL_POTENTIAL_LIMIT
                / PHI_S_VON_KOCH_THRESHOLD
            ),
            "u6_drift_assessed": False,
            "legacy_u6_coordinate": True,
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
        "u6_drift_assessed": False,
        "potential_magnitude_warning_threshold": float(
            PHI_S_VON_KOCH_THRESHOLD
        ),
        "mean_yang_mills_residual": _mean(
            point.mean_yang_mills_residual for point in points
        ),
        "max_yang_mills_residual": max(
            (point.max_yang_mills_residual for point in points),
            default=0.0,
        ),
        "mean_pure_gauge_consistency_residual": _mean(
            point.mean_pure_gauge_consistency_residual for point in points
        ),
        "max_pure_gauge_consistency_residual": max(
            (
                point.max_pure_gauge_consistency_residual
                for point in points
            ),
            default=0.0,
        ),
        "u6_definition": (
            "mean_i |Phi_s_after(i)-Phi_s_before(i)| < threshold"
        ),
        "curvature_is_numerical_residual": True,
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
