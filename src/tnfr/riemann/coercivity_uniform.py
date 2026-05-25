r"""P22: empirical uniform-coercivity certificate for alpha = W / E.

This module upgrades pointwise positivity checks to an interval-level
diagnostic by combining:

1) sampled minimum alpha over a dense sigma grid,
2) finite-difference slope envelope (Lipschitz proxy),
3) a mesh-corrected lower bound on the full interval.

The result is still empirical (not a proof), but it is substantially
stronger than a plain sampled positivity statement.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..mathematics.unified_numerical import np
from .admissible_family_sweep import DEFAULT_TEST_FAMILIES, FamilyFactory
from .alpha_sweep import DEFAULT_GAUGES, GaugeFn
from .nodeaware_gauge_sweep import DEFAULT_NODEAWARE_GAUGES, NodeAwareGaugeFn
from .prime_ladder_hamiltonian import PrimeLadderHamiltonian
from .admissible_family_sweep import sweep_alpha_admissible_family
from .nodeaware_gauge_sweep import sweep_alpha_nodeaware


def _max_abs_slope(alpha_table: np.ndarray, sigmas: np.ndarray) -> float:
    """Return max |Δalpha/Δsigma| across all trajectories."""
    if sigmas.size < 2:
        return 0.0
    ds = np.diff(sigmas)
    flat = alpha_table.reshape((-1, sigmas.size))
    slope_max = 0.0
    for row in flat:
        if not np.all(np.isfinite(row)):
            continue
        da = np.diff(row)
        local = np.max(np.abs(da / ds))
        if float(local) > slope_max:
            slope_max = float(local)
    return slope_max


def _stratified_interval_lower_bound(
    alpha_table: np.ndarray,
    sigmas: np.ndarray,
) -> float:
    """Lower bound using per-trajectory slope envelopes.

    For each trajectory r(σ):

        lb_r = min(r) - L_r * mesh_radius,

    where L_r = max |Δr/Δσ| on that trajectory.
    Returns min_r lb_r.
    """
    if sigmas.size < 2:
        finite = alpha_table[np.isfinite(alpha_table)]
        return float(np.min(finite)) if finite.size else float("nan")

    mesh_radius = 0.5 * float(np.max(np.diff(sigmas)))
    flat = alpha_table.reshape((-1, sigmas.size))
    best = float("inf")

    for row in flat:
        if not np.all(np.isfinite(row)):
            continue
        ds = np.diff(sigmas)
        da = np.diff(row)
        L_row = float(np.max(np.abs(da / ds)))
        lb_row = float(np.min(row) - L_row * mesh_radius)
        if lb_row < best:
            best = lb_row

    return best if np.isfinite(best) else float("nan")


def _segmentwise_interval_lower_bound(
    alpha_table: np.ndarray,
    sigmas: np.ndarray,
) -> float:
    """Lower bound using segment-local Lipschitz control.

    For each trajectory and each segment [σ_i, σ_{i+1}] (linear envelope):

        lb_i = min(a_i, a_{i+1}) - |Δa/Δσ| * (Δσ/2)

    Returns the minimum over all rows and segments.
    """
    if sigmas.size < 2:
        finite = alpha_table[np.isfinite(alpha_table)]
        return float(np.min(finite)) if finite.size else float("nan")

    flat = alpha_table.reshape((-1, sigmas.size))
    best = float("inf")

    for row in flat:
        if not np.all(np.isfinite(row)):
            continue
        for i in range(sigmas.size - 1):
            ds = float(sigmas[i + 1] - sigmas[i])
            if ds <= 0.0:
                continue
            a0 = float(row[i])
            a1 = float(row[i + 1])
            slope = abs((a1 - a0) / ds)
            lb_seg = min(a0, a1) - slope * (0.5 * ds)
            if lb_seg < best:
                best = lb_seg

    return best if np.isfinite(best) else float("nan")


@dataclass(frozen=True)
class UniformCoercivityCertificate:
    """Empirical uniform-coercivity summary over [sigma_min, sigma_max]."""

    sigma_min: float
    sigma_max: float
    n_sigma: int
    sampled_alpha_min: float
    sampled_alpha_max: float
    lipschitz_proxy_max: float
    mesh_radius: float
    interval_lower_bound_global: float
    interval_lower_bound_stratified: float
    interval_lower_bound_local: float
    sampled_all_positive: bool
    interval_lower_global_positive: bool
    interval_lower_stratified_positive: bool
    interval_lower_local_positive: bool
    admissible_ok: bool
    nodeaware_ok: bool

    def summary(self) -> str:
        return (
            "UniformCoercivityCertificate("
            f"sigma=[{self.sigma_min:.3f}, {self.sigma_max:.3f}], "
            f"n_sigma={self.n_sigma}, "
            f"alpha_min_sampled={self.sampled_alpha_min:+.4e}, "
            f"alpha_max_sampled={self.sampled_alpha_max:+.4e}, "
            f"L_proxy={self.lipschitz_proxy_max:.4e}, "
            f"mesh_radius={self.mesh_radius:.4e}, "
            f"interval_lb_global={self.interval_lower_bound_global:+.4e}, "
            "interval_lb_stratified="
            f"{self.interval_lower_bound_stratified:+.4e}, "
            f"interval_lb_local={self.interval_lower_bound_local:+.4e}, "
            f"sampled_all_positive={self.sampled_all_positive}, "
            "interval_lb_global_positive="
            f"{self.interval_lower_global_positive}, "
            "interval_lb_stratified_positive="
            f"{self.interval_lower_stratified_positive}, "
            "interval_lb_local_positive="
            f"{self.interval_lower_local_positive}, "
            f"admissible_ok={self.admissible_ok}, "
            f"nodeaware_ok={self.nodeaware_ok})"
        )


def verify_uniform_coercivity_empirical(
    bundle: PrimeLadderHamiltonian,
    *,
    sigma_min: float = 0.5,
    sigma_max: float = 8.0,
    n_sigma: int = 24,
    families: dict[str, FamilyFactory] | None = None,
    gauges: dict[str, GaugeFn] | None = None,
    node_gauges: dict[str, NodeAwareGaugeFn] | None = None,
    n_zeros: int = 40,
    convergence_tol: float = 1e-12,
    max_zeros: int = 160,
) -> UniformCoercivityCertificate:
    """Build empirical interval-level coercivity certificate.

    Computes alpha surfaces from both P19 and P20 on the same sigma grid,
    then estimates an interval lower bound:

        alpha_inf_interval >= alpha_min_sampled - L_proxy * mesh_radius

    where L_proxy is the maximum finite-difference slope envelope over all
    sampled trajectories.
    """
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("sigma bounds must be strictly positive")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be > sigma_min")
    if n_sigma < 2:
        raise ValueError("n_sigma must be >= 2")

    fam = (
        dict(families)
        if families is not None
        else dict(DEFAULT_TEST_FAMILIES)
    )
    g_scalar = dict(gauges) if gauges is not None else dict(DEFAULT_GAUGES)
    g_node = (
        dict(node_gauges)
        if node_gauges is not None
        else dict(DEFAULT_NODEAWARE_GAUGES)
    )

    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)

    cert_adm = sweep_alpha_admissible_family(
        bundle,
        sigmas,
        families=fam,
        gauges=g_scalar,
        n_zeros=n_zeros,
        convergence_tol=convergence_tol,
        max_zeros=max_zeros,
    )
    cert_node = sweep_alpha_nodeaware(
        bundle,
        sigmas,
        families=fam,
        node_gauges=g_node,
        n_zeros=n_zeros,
        convergence_tol=convergence_tol,
        max_zeros=max_zeros,
    )

    alpha_a = cert_adm.alpha_table
    alpha_n = cert_node.alpha_table

    finite_a = alpha_a[np.isfinite(alpha_a)]
    finite_n = alpha_n[np.isfinite(alpha_n)]
    if finite_a.size == 0 or finite_n.size == 0:
        raise ValueError(
            "no finite alpha values available for coercivity check"
        )

    sampled_alpha_min = float(
        min(float(np.min(finite_a)), float(np.min(finite_n)))
    )
    sampled_alpha_max = float(
        max(float(np.max(finite_a)), float(np.max(finite_n)))
    )

    L_a = _max_abs_slope(alpha_a, sigmas)
    L_n = _max_abs_slope(alpha_n, sigmas)
    L_proxy = float(max(L_a, L_n))

    mesh_step_max = float(np.max(np.diff(sigmas)))
    mesh_radius = 0.5 * mesh_step_max
    interval_lb_global = sampled_alpha_min - L_proxy * mesh_radius

    lb_strat_a = _stratified_interval_lower_bound(alpha_a, sigmas)
    lb_strat_n = _stratified_interval_lower_bound(alpha_n, sigmas)
    interval_lb_stratified = float(min(lb_strat_a, lb_strat_n))

    lb_local_a = _segmentwise_interval_lower_bound(alpha_a, sigmas)
    lb_local_n = _segmentwise_interval_lower_bound(alpha_n, sigmas)
    interval_lb_local = float(min(lb_local_a, lb_local_n))

    sampled_all_positive = bool(
        cert_adm.alpha_all_positive and cert_node.alpha_all_positive
    )
    interval_lb_global_positive = bool(interval_lb_global > 0.0)
    interval_lb_stratified_positive = bool(interval_lb_stratified > 0.0)
    interval_lb_local_positive = bool(interval_lb_local > 0.0)

    return UniformCoercivityCertificate(
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        n_sigma=int(n_sigma),
        sampled_alpha_min=sampled_alpha_min,
        sampled_alpha_max=sampled_alpha_max,
        lipschitz_proxy_max=L_proxy,
        mesh_radius=mesh_radius,
        interval_lower_bound_global=float(interval_lb_global),
        interval_lower_bound_stratified=interval_lb_stratified,
        interval_lower_bound_local=interval_lb_local,
        sampled_all_positive=sampled_all_positive,
        interval_lower_global_positive=interval_lb_global_positive,
        interval_lower_stratified_positive=interval_lb_stratified_positive,
        interval_lower_local_positive=interval_lb_local_positive,
        admissible_ok=bool(cert_adm.alpha_all_positive),
        nodeaware_ok=bool(cert_node.alpha_all_positive),
    )


__all__ = [
    "UniformCoercivityCertificate",
    "verify_uniform_coercivity_empirical",
]
