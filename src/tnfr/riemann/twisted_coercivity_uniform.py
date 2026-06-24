r"""P42: empirical uniform-coercivity certificate for chi-twisted alpha.

Structural extension of P22/P23/P24 (uniform / stratified / adaptive
coercivity certificate on the zeta side) to primitive real Dirichlet
L-functions L(s, chi).

For a fixed primitive real chi (chi_3, chi_4, chi_5 in the canonical
catalogue), this module promotes pointwise positivity of

    alpha_chi(sigma; f, g) = W_chi[sigma; f] / E_TNFR_chi[sigma; f, g]

(verified pointwise by P39 admissible-family sweep and P40 node-aware
gauge sweep) to an *interval-level* empirical certificate on the
sigma-window [sigma_min, sigma_max].

The construction is canonical and exactly parallel to the zeta side:

1) sample alpha_chi on a dense log-spaced sigma grid via P39 (scalar
   gauges) and P40 (node-aware gauges);
2) compute a finite-difference Lipschitz proxy L_proxy = max
   |Delta alpha / Delta sigma| over all sampled trajectories;
3) build three interval lower bounds:
   - **global**: alpha_min - L_proxy * mesh_radius
   - **stratified**: per-trajectory min - per-trajectory slope * radius
   - **local**: segment-local linear envelope min(a0,a1) - |slope| * dsigma/2
4) optionally adaptively refine the worst segments (P24-style) by
   bisecting the smallest-margin segments and resampling both surfaces.

Honest scope
------------
P42 is an empirical interval-level diagnostic, *not* a proof.  Even
when all three lower bounds are positive and the adaptive refinement
keeps them positive, the certificate:

- does NOT prove GRH for L(s, chi);
- does NOT advance G4 = RH;
- is restricted to the finite grid (sigmas) x (families) x (gauges)
  defined by P19, P18, P20 and resampled in P39 / P40.

It closes the uniform-coercivity robustness gap on the L-track for
primitive real chi by lifting pointwise positivity to interval-level
positivity with explicit Lipschitz control.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..mathematics.unified_numerical import np
from .admissible_family_sweep import DEFAULT_TEST_FAMILIES, FamilyFactory
from .alpha_sweep import DEFAULT_GAUGES, GaugeFn
from .coercivity_uniform import (
    _max_abs_slope,
    _segmentwise_interval_lower_bound,
    _stratified_interval_lower_bound,
    _worst_segment_indices,
)
from .dirichlet_l import DirichletCharacter
from .nodeaware_gauge_sweep import DEFAULT_NODEAWARE_GAUGES, NodeAwareGaugeFn
from .twisted_admissible_family_sweep import sweep_twisted_admissible_family
from .twisted_nodeaware_gauge_sweep import sweep_twisted_nodeaware_gauge
from .twisted_prime_ladder_hamiltonian import TwistedPrimeLadderHamiltonian


@dataclass(frozen=True)
class TwistedUniformCoercivityCertificate:
    """Empirical chi-twisted uniform-coercivity summary."""

    character_name: str
    character_modulus: int
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
    n_refinement_rounds: int = 0
    n_sigma_refined: int = 0
    interval_lower_bound_local_refined: float = float("nan")
    interval_lower_local_refined_positive: bool = False

    def summary(self) -> str:
        return (
            "TwistedUniformCoercivityCertificate("
            f"chi={self.character_name}, q={self.character_modulus}, "
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
            "interval_lb_local_refined="
            f"{self.interval_lower_bound_local_refined:+.4e}, "
            f"refinement_rounds={self.n_refinement_rounds}, "
            f"n_sigma_refined={self.n_sigma_refined}, "
            f"sampled_all_positive={self.sampled_all_positive}, "
            "interval_lb_global_positive="
            f"{self.interval_lower_global_positive}, "
            "interval_lb_stratified_positive="
            f"{self.interval_lower_stratified_positive}, "
            "interval_lb_local_positive="
            f"{self.interval_lower_local_positive}, "
            "interval_lb_local_refined_positive="
            f"{self.interval_lower_local_refined_positive}, "
            f"admissible_ok={self.admissible_ok}, "
            f"nodeaware_ok={self.nodeaware_ok})"
        )


def verify_twisted_uniform_coercivity_empirical(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    *,
    sigma_min: float = 0.5,
    sigma_max: float = 8.0,
    n_sigma: int = 24,
    families: dict[str, FamilyFactory] | None = None,
    gauges: dict[str, GaugeFn] | None = None,
    node_gauges: dict[str, NodeAwareGaugeFn] | None = None,
    t_min: float = 0.5,
    t_max: float | None = None,
    initial_step: float = 0.25,
    dps: int = 30,
    refinement_rounds: int = 0,
    refinement_per_round: int = 2,
) -> TwistedUniformCoercivityCertificate:
    """Build empirical interval-level chi-twisted coercivity certificate.

    Computes alpha_chi surfaces from both P39 (scalar-gauges) and P40
    (node-aware gauges) on the same log-spaced sigma grid, then estimates
    an interval lower bound

        alpha_chi_inf_interval >= alpha_min_sampled
                                  - L_proxy * mesh_radius

    where L_proxy is the maximum finite-difference slope envelope over
    all sampled trajectories.  Optional adaptive refinement bisects the
    worst segments.

    Parameters
    ----------
    chi
        Primitive real Dirichlet character (P32).
    bundle
        P34 chi-twisted prime-ladder Hamiltonian bundle for ``chi``.
    sigma_min, sigma_max, n_sigma
        Log-spaced Gaussian-width grid (P22 convention).
    families, gauges, node_gauges
        Canonical defaults: P19 / P18 / P20.
    t_min, t_max, initial_step, dps
        Forwarded to the underlying P35 enumerator via P39 / P40.
    refinement_rounds, refinement_per_round
        Adaptive bisection control (P24-style).  ``refinement_rounds=0``
        yields the pure P22-style certificate.

    Returns
    -------
    TwistedUniformCoercivityCertificate
        Sampled minima, Lipschitz proxy, three interval lower bounds,
        adaptive-refinement counters, and aggregate positivity flags.

    Raises
    ------
    ValueError
        If sigma bounds are non-positive, ``sigma_max <= sigma_min``,
        ``n_sigma < 2``, or no finite alpha values are produced.
    """
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("sigma bounds must be strictly positive")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be > sigma_min")
    if n_sigma < 2:
        raise ValueError("n_sigma must be >= 2")

    fam = dict(families) if families is not None else dict(DEFAULT_TEST_FAMILIES)
    g_scalar = dict(gauges) if gauges is not None else dict(DEFAULT_GAUGES)
    g_node = (
        dict(node_gauges) if node_gauges is not None else dict(DEFAULT_NODEAWARE_GAUGES)
    )

    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)

    cert_adm = sweep_twisted_admissible_family(
        chi,
        bundle,
        sigmas,
        families=fam,
        gauges=g_scalar,
        t_min=t_min,
        t_max=t_max,
        initial_step=initial_step,
        dps=dps,
    )
    cert_node = sweep_twisted_nodeaware_gauge(
        chi,
        bundle,
        sigmas,
        families=fam,
        node_gauges=g_node,
        t_min=t_min,
        t_max=t_max,
        initial_step=initial_step,
        dps=dps,
    )

    alpha_a = np.asarray(cert_adm.alpha_table, dtype=float)
    alpha_n = np.asarray(cert_node.alpha_table, dtype=float)

    finite_a = alpha_a[np.isfinite(alpha_a)]
    finite_n = alpha_n[np.isfinite(alpha_n)]
    if finite_a.size == 0 or finite_n.size == 0:
        raise ValueError("no finite alpha values available for coercivity check")

    sampled_alpha_min = float(min(float(np.min(finite_a)), float(np.min(finite_n))))
    sampled_alpha_max = float(max(float(np.max(finite_a)), float(np.max(finite_n))))

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

    # --- P24-style adaptive refinement around worst segments -----------
    refined_sigmas = sigmas
    refined_alpha_a = alpha_a
    refined_alpha_n = alpha_n
    interval_lb_local_refined = interval_lb_local

    rounds = max(int(refinement_rounds), 0)
    per_round = max(int(refinement_per_round), 1)

    for _ in range(rounds):
        worst = _worst_segment_indices(
            refined_alpha_a,
            refined_alpha_n,
            refined_sigmas,
            per_round,
        )
        if not worst:
            break
        midpoints = [
            0.5 * (float(refined_sigmas[i]) + float(refined_sigmas[i + 1]))
            for i in worst
        ]
        augmented = np.unique(np.concatenate([refined_sigmas, np.asarray(midpoints)]))
        if augmented.size == refined_sigmas.size:
            break

        new_cert_adm = sweep_twisted_admissible_family(
            chi,
            bundle,
            augmented,
            families=fam,
            gauges=g_scalar,
            t_min=t_min,
            t_max=t_max,
            initial_step=initial_step,
            dps=dps,
        )
        new_cert_node = sweep_twisted_nodeaware_gauge(
            chi,
            bundle,
            augmented,
            families=fam,
            node_gauges=g_node,
            t_min=t_min,
            t_max=t_max,
            initial_step=initial_step,
            dps=dps,
        )

        refined_sigmas = augmented
        refined_alpha_a = np.asarray(new_cert_adm.alpha_table, dtype=float)
        refined_alpha_n = np.asarray(new_cert_node.alpha_table, dtype=float)

        lb_a = _segmentwise_interval_lower_bound(refined_alpha_a, refined_sigmas)
        lb_n = _segmentwise_interval_lower_bound(refined_alpha_n, refined_sigmas)
        interval_lb_local_refined = float(min(lb_a, lb_n))

    interval_lb_local_refined_positive = bool(interval_lb_local_refined > 0.0)

    return TwistedUniformCoercivityCertificate(
        character_name=str(chi.name),
        character_modulus=int(chi.modulus),
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
        n_refinement_rounds=rounds,
        n_sigma_refined=int(refined_sigmas.size),
        interval_lower_bound_local_refined=float(interval_lb_local_refined),
        interval_lower_local_refined_positive=(interval_lb_local_refined_positive),
    )


__all__ = [
    "TwistedUniformCoercivityCertificate",
    "verify_twisted_uniform_coercivity_empirical",
]
