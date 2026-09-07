r"""Legacy Y2 structural-potential magnitude sweep for gauge-gap diagnostics.

This module implements the second TNFR–Yang–Mills milestone: sweep finite
structural gauge graphs across a single-snapshot ``max|Phi_s|/(pi/2)``
coordinate, then measure how the Y1 gap diagnostic behaves. Public names retain
``U6`` for compatibility, but this module has no reference snapshot and
therefore does not assess the canonical two-snapshot U6 drift policy.
The actual per-node magnitude warning uses ``pi/4`` and is reported separately.

The sweep is intentionally finite and empirical.  It records correlations and
failure modes, but it does not claim a continuum mass-gap theorem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Iterable

from ..constants.canonical import (
    PHI_S_VON_KOCH_THRESHOLD,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)
from ..mathematics.unified_numerical import np
from ..physics.canonical import compute_structural_potential
from ..physics.gauge import compute_gauge_curvature, compute_yang_mills_equations
from ..rng import validate_seed
from .structural_gap import (
    _finite_nonnegative,
    _finite_positive,
    build_structural_gauge_graph,
    compute_structural_gauge_gap,
)


@dataclass(frozen=True)
class U6ConfinementSweepPoint:
    """One finite graph point in the legacy Y2 magnitude sweep.

    ``target_u6_ratio``, ``observed_u6_ratio`` and ``u6_confined`` are retained
    compatibility fields. They encode the target and observed
    ``max|Phi_s|/(pi/2)`` magnitude ratio and whether that legacy ratio is below
    one. They are neither a U6 drift assessment nor the π/4 magnitude warning.
    ``n`` retains the requested size; ``actual_n_nodes`` reports the realized
    graph size for topologies such as the square grid.
    """

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
        """Return the legacy magnitude ratio using π/2 as denominator."""

        return self.observed_u6_ratio

    @property
    def potential_magnitude_below_pi_scale(self) -> bool:
        """Compatibility alias for the π/4 magnitude-warning decision."""

        return self.potential_magnitude_within_warning

    @property
    def potential_magnitude_within_warning(self) -> bool:
        """Whether ``max|Phi_s|`` is below the selected π/4 warning."""

        return self.potential_magnitude_ratio < 1.0

    @property
    def u6_drift_assessed(self) -> bool:
        """U6 requires a reference snapshot, which this sweep does not have."""
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
    def cycle_closure_residual_active(self) -> bool:
        """Accurate alias for the legacy ``curvature_active`` field."""

        return self.curvature_active

    @property
    def curvature_is_numerical_residual(self) -> bool:
        """Whether curvature fields only measure exact-one-form closure error."""

        return True


@dataclass(frozen=True)
class U6ConfinementSweepReport:
    """Aggregate report for the legacy Y2 potential-magnitude sweep."""

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
    """Run the legacy Y2 finite structural-potential magnitude sweep.

    Parameters are sampling controls for finite diagnostics only. The legacy
    ``target_u6_ratios`` argument sets ``max|Phi_s|/(pi/2)``. Ratios below one
    are merely below the numerical scale borrowed from U6; the separate
    per-node magnitude warning is ``pi/4``. U6 itself evaluates strict
    mean-absolute before/after potential drift and is not assessed here.
    """
    n_list = _validated_integer_values(n_values, "n_values", minimum=2)
    topology_list = _validated_topologies(topologies)
    seed_list = _validated_seeds(seeds)
    ratio_list = _validated_ratios(target_u6_ratios)
    phase_spread = _finite_nonnegative(phase_spread, "phase_spread")
    gauge_seed = validate_seed(gauge_seed, allow_none=False)
    curvature_weight = _finite_nonnegative(curvature_weight, "curvature_weight")
    confinement_weight = _finite_nonnegative(
        confinement_weight,
        "confinement_weight",
    )
    tolerance = _finite_positive(tolerance, "tolerance")
    eigen_tolerance = _finite_nonnegative(eigen_tolerance, "eigen_tolerance")

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
                    curvature = compute_gauge_curvature(graph)
                    curv_abs = [abs(float(value)) for value in curvature.values()]
                    mean_abs_curv = _mean(curv_abs)
                    max_abs_curv = max(curv_abs, default=0.0)
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

                    point_metadata = dict(result.metadata)
                    point_metadata.update(
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
                            realized_n_nodes=actual_n,
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


def _materialize(values: Iterable[Any], name: str) -> tuple[Any, ...]:
    """Materialize a non-string iterable with a stable validation error."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of values")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of values") from exc
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _validated_integer_values(
    values: Iterable[Any],
    name: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    """Validate integer sampling coordinates without truncating floats."""

    raw = _materialize(values, name)
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise TypeError(f"{name} must contain only integers")
    result = tuple(int(value) for value in raw)
    if any(value < minimum for value in result):
        raise ValueError(f"all {name} values must be at least {minimum}")
    return result


def _validated_topologies(values: Iterable[Any]) -> tuple[str, ...]:
    """Validate topology labels before constructing any graph."""

    raw = _materialize(values, "topologies")
    if any(not isinstance(value, str) or not value for value in raw):
        raise TypeError("topologies must contain only nonempty strings")
    return tuple(raw)


def _validated_seeds(values: Iterable[Any]) -> tuple[int, ...]:
    """Validate deterministic seeds without float or bool coercion."""

    raw = _materialize(values, "seeds")
    return tuple(validate_seed(value, allow_none=False) for value in raw)


def _validated_ratios(values: Iterable[Any]) -> tuple[float, ...]:
    """Validate finite ratios whose squared penalty remains representable."""

    raw = _materialize(values, "target_u6_ratios")
    maximum = math.sqrt(float(np.finfo(float).max))
    result: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("target_u6_ratios must contain only real numbers")
        ratio = float(value)
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError(
                "target_u6_ratios must contain finite non-negative values"
            )
        if ratio > maximum:
            raise ValueError(
                "target_u6_ratios are too large for the squared potential penalty"
            )
        result.append(ratio)
    return tuple(result)


def _rescale_delta_nfr_to_u6_ratio(G: Any, target_ratio: float) -> None:
    """Scale pressure to a magnitude ratio; legacy U6-named helper."""
    ratio = _validated_ratios((target_ratio,))[0]
    target_abs_phi_s = ratio * U6_STRUCTURAL_POTENTIAL_LIMIT
    if target_abs_phi_s == 0.0:
        for node in G.nodes():
            G.nodes[node]["delta_nfr"] = 0.0
        return

    current_max = _max_abs_structural_potential(G)
    if current_max == 0.0:
        _install_deterministic_delta_pattern(G)
        current_max = _max_abs_structural_potential(G)

    if current_max == 0.0:
        raise RuntimeError("cannot realize a nonzero potential magnitude target")
    if not math.isfinite(current_max):
        raise ValueError("structural potential must remain finite during rescaling")

    scale = target_abs_phi_s / current_max
    scaled_pressure: dict[Any, float] = {}
    for node in G.nodes():
        delta = float(G.nodes[node].get("delta_nfr", 0.0))
        scaled = float(delta * scale)
        if not math.isfinite(scaled):
            raise ValueError(
                "rescaled structural pressure exceeds floating-point range"
            )
        scaled_pressure[node] = scaled
    for node, scaled in scaled_pressure.items():
        G.nodes[node]["delta_nfr"] = scaled


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
    within_warning = tuple(
        point for point in points if point.potential_magnitude_within_warning
    )
    outside_warning = tuple(
        point for point in points if not point.potential_magnitude_within_warning
    )
    below_legacy_scale = tuple(
        point for point in points if point.u6_confined
    )
    at_or_above_legacy_scale = tuple(
        point for point in points if not point.u6_confined
    )
    positive = tuple(point for point in points if point.gap > eigen_tolerance)
    curvature_active = tuple(point for point in points if point.curvature_active)

    return {
        "n_points": len(points),
        "n_within_potential_magnitude_warning": len(within_warning),
        "n_outside_potential_magnitude_warning": len(outside_warning),
        "n_below_legacy_u6_normalization_scale": len(below_legacy_scale),
        "n_at_or_above_legacy_u6_normalization_scale": len(
            at_or_above_legacy_scale
        ),
        # Compatibility aliases introduced with the scope correction.
        "n_below_pi_magnitude_scale": len(within_warning),
        "n_at_or_above_pi_magnitude_scale": len(outside_warning),
        # Compatibility aliases; these are magnitude groups, not U6 verdicts.
        "n_confined": len(below_legacy_scale),
        "n_unconfined": len(at_or_above_legacy_scale),
        "positive_gap_fraction": _fraction(len(positive), len(points)),
        "confined_positive_gap_fraction": _positive_gap_fraction(
            below_legacy_scale,
            eigen_tolerance,
        ),
        "unconfined_positive_gap_fraction": _positive_gap_fraction(
            at_or_above_legacy_scale,
            eigen_tolerance,
        ),
        "mean_gap": _mean(point.gap for point in points),
        "mean_within_potential_magnitude_warning_gap": _mean(
            point.gap for point in within_warning
        ),
        "mean_outside_potential_magnitude_warning_gap": _mean(
            point.gap for point in outside_warning
        ),
        "mean_below_pi_magnitude_gap": _mean(
            point.gap for point in within_warning
        ),
        "mean_at_or_above_pi_magnitude_gap": _mean(
            point.gap for point in outside_warning
        ),
        "mean_confined_gap": _mean(point.gap for point in below_legacy_scale),
        "mean_unconfined_gap": _mean(
            point.gap for point in at_or_above_legacy_scale
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
        "curvature_active_fraction": _fraction(
            len(curvature_active),
            len(points),
        ),
        "cycle_closure_residual_active_fraction": _fraction(
            len(curvature_active),
            len(points),
        ),
        "potential_magnitude_gap_correlation": _pearson(
            [point.potential_magnitude_ratio for point in points],
            [point.gap for point in points],
        ),
        "u6_gap_correlation": _pearson(
            [point.observed_u6_ratio for point in points],
            [point.gap for point in points],
        ),
        "u6_drift_assessed": False,
        "u6_definition": (
            "mean_i |Phi_s_after(i)-Phi_s_before(i)| < threshold"
        ),
        "legacy_u6_fields_are_magnitude_proxies": True,
        "curvature_is_numerical_residual": True,
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
    if not bool(np.all(np.isfinite(x_arr))) or not bool(np.all(np.isfinite(y_arr))):
        return None
    x_scale = max(float(np.max(np.abs(x_arr))), 1.0)
    y_scale = max(float(np.max(np.abs(y_arr))), 1.0)
    x_centered = x_arr / x_scale - float(np.mean(x_arr / x_scale))
    y_centered = y_arr / y_scale - float(np.mean(y_arr / y_scale))
    x_norm = float(np.linalg.norm(x_centered))
    y_norm = float(np.linalg.norm(y_centered))
    if x_norm <= 1e-15 or y_norm <= 1e-15:
        return None
    corr = float(np.dot(x_centered / x_norm, y_centered / y_norm))
    if not np.isfinite(corr):
        return None
    return max(-1.0, min(1.0, corr))
