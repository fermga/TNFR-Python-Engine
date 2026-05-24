r"""P19: admissible-family sweep for the Weil-TNFR ratio.

Extends P18 from a single Gaussian family to a small admissible-family
bundle, preserving the same gauge machinery and positivity diagnostics.

This module is intentionally operational: it does not claim a theorem on
family completeness. It provides quantitative evidence over multiple
Schwartz-even test families.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..physics.conservation import compute_energy_functional
from .alpha_sweep import DEFAULT_GAUGES, GaugeFn
from .prime_ladder_hamiltonian import PrimeLadderHamiltonian
from .weil_explicit_formula import gaussian_test_function, weil_zero_side


class AdmissibleTestFunction(Protocol):
    """Minimal protocol required by the Weil/P18-style sweep pipeline."""

    def h(self, t: float) -> float:
        """Evaluate the real-even test function on the real line."""

    def g(self, u: float) -> float:
        """Evaluate the Fourier-side profile used in prime-side terms."""

    def g_zero(self) -> float:
        """Return g(0)."""

    def h_at_half_pole(self) -> float:
        """Return h(i/2)=h(-i/2) for the chosen analytic extension."""


@dataclass(frozen=True)
class GaussianMixtureTestFunction:
    r"""Two-scale Gaussian mixture, still even, positive and Schwartz.

    .. math::

        h(t) = (1-\lambda)\,e^{-t^2/(2\sigma^2)}
             + \lambda\,e^{-t^2/(2(\beta\sigma)^2)}.

    Fourier profile under the same convention as P15:

    .. math::

        g(u) = \frac{(1-\lambda)\sigma}{\sqrt{2\pi}}
               e^{-\sigma^2u^2/2}
             + \frac{\lambda\beta\sigma}{\sqrt{2\pi}}
               e^{-(\beta\sigma)^2u^2/2}.
    """

    sigma: float
    mix: float = 0.35
    beta: float = 2.0

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be strictly positive")
        if not (0.0 <= self.mix <= 1.0):
            raise ValueError("mix must lie in [0, 1]")
        if self.beta <= 0.0:
            raise ValueError("beta must be strictly positive")

    def h(self, t: float) -> float:
        s0 = self.sigma
        s1 = self.beta * self.sigma
        g0 = math.exp(-(t * t) / (2.0 * s0 * s0))
        g1 = math.exp(-(t * t) / (2.0 * s1 * s1))
        return (1.0 - self.mix) * g0 + self.mix * g1

    def h_complex(self, t: complex) -> complex:
        s0 = self.sigma
        s1 = self.beta * self.sigma
        g0 = np.exp(-(t**2) / (2.0 * s0 * s0))
        g1 = np.exp(-(t**2) / (2.0 * s1 * s1))
        return complex((1.0 - self.mix) * g0 + self.mix * g1)

    def g(self, u: float) -> float:
        s0 = self.sigma
        s1 = self.beta * self.sigma
        nrm = 1.0 / math.sqrt(2.0 * math.pi)
        g0 = s0 * nrm * math.exp(-(s0 * s0) * u * u / 2.0)
        g1 = s1 * nrm * math.exp(-(s1 * s1) * u * u / 2.0)
        return (1.0 - self.mix) * g0 + self.mix * g1

    def g_zero(self) -> float:
        s0 = self.sigma
        s1 = self.beta * self.sigma
        nrm = 1.0 / math.sqrt(2.0 * math.pi)
        return (1.0 - self.mix) * s0 * nrm + self.mix * s1 * nrm

    def h_at_half_pole(self) -> float:
        s0 = self.sigma
        s1 = self.beta * self.sigma
        h0 = math.exp(1.0 / (8.0 * s0 * s0))
        h1 = math.exp(1.0 / (8.0 * s1 * s1))
        return (1.0 - self.mix) * h0 + self.mix * h1


def gaussian_mixture_test_function(
    sigma: float,
    *,
    mix: float = 0.35,
    beta: float = 2.0,
) -> GaussianMixtureTestFunction:
    """Construct a two-scale Gaussian mixture admissible test function."""
    return GaussianMixtureTestFunction(sigma=float(sigma), mix=mix, beta=beta)


@dataclass(frozen=True)
class Hermite2GaussianTestFunction:
    r"""Second-order Hermite-Gaussian deformation of the Gaussian family.

    .. math::

        h(t) = \Bigl(1 + \eta\,(t/\sigma)^2\Bigr)
               e^{-t^2/(2\sigma^2)},\qquad \eta \ge 0.

    This preserves evenness and Schwartz decay, while introducing a
    non-trivial polynomial envelope.
    """

    sigma: float
    eta: float = 0.25

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be strictly positive")
        if self.eta < 0.0:
            raise ValueError("eta must be non-negative")

    def h(self, t: float) -> float:
        s = self.sigma
        x = t / s
        return (1.0 + self.eta * x * x) * math.exp(-(x * x) / 2.0)

    def h_complex(self, t: complex) -> complex:
        s = self.sigma
        x = t / s
        return complex((1.0 + self.eta * x * x) * np.exp(-(x * x) / 2.0))

    def g(self, u: float) -> float:
        r"""Closed-form Fourier-side profile under the P15 convention."""
        s = self.sigma
        base = (s / math.sqrt(2.0 * math.pi))
        env = math.exp(-(s * s) * u * u / 2.0)
        poly = 1.0 + self.eta * (1.0 - (s * s) * u * u)
        return base * env * poly

    def g_zero(self) -> float:
        s = self.sigma
        return (s / math.sqrt(2.0 * math.pi)) * (1.0 + self.eta)

    def h_at_half_pole(self) -> float:
        s = self.sigma
        pole_poly = 1.0 - self.eta / (4.0 * s * s)
        return pole_poly * math.exp(1.0 / (8.0 * s * s))


def hermite2_gaussian_test_function(
    sigma: float,
    *,
    eta: float = 0.25,
) -> Hermite2GaussianTestFunction:
    """Construct second-order Hermite-Gaussian admissible test function."""
    return Hermite2GaussianTestFunction(sigma=float(sigma), eta=eta)


FamilyFactory = Callable[[float], AdmissibleTestFunction]


DEFAULT_TEST_FAMILIES: Mapping[str, FamilyFactory] = {
    "gaussian": gaussian_test_function,
    "gaussian_mixture": gaussian_mixture_test_function,
    "hermite2_gaussian": hermite2_gaussian_test_function,
}


def build_test_state_from_test_function(
    bundle: PrimeLadderHamiltonian,
    test: AdmissibleTestFunction,
    gauge: GaugeFn,
) -> nx.Graph:
    """Build a structural state using test.h(k log p) as the source profile."""
    G = bundle.graph.copy()
    pi = math.pi
    for node in G.nodes():
        p, k = node
        e_n = float(k) * math.log(float(p))
        h_val = float(test.h(e_n))
        d_val, phi_val, epi_val = gauge(h_val)
        if phi_val > pi:
            phi_val = pi
        elif phi_val < -pi:
            phi_val = -pi
        G.nodes[node]["dnfr"] = float(d_val)
        G.nodes[node]["phase"] = float(phi_val)
        G.nodes[node]["EPI"] = float(epi_val)
    return G


@dataclass(frozen=True)
class AdmissibleFamilySweepCertificate:
    """Outcome of a family × gauge × sigma alpha sweep."""

    sigmas: object
    families: tuple[str, ...]
    gauges: tuple[str, ...]
    weil_table: object
    energy_table: object
    alpha_table: object
    weil_all_positive: bool
    alpha_all_positive: bool
    alpha_min: float
    alpha_max: float
    alpha_min_sigma: float
    alpha_min_family: str
    alpha_min_gauge: str

    def summary(self) -> str:
        return (
            "AdmissibleFamilySweepCertificate("
            f"n_sigma={len(self.sigmas)}, "
            f"n_family={len(self.families)}, "
            f"n_gauge={len(self.gauges)}, "
            f"W_all_positive={self.weil_all_positive}, "
            f"alpha_all_positive={self.alpha_all_positive}, "
            f"alpha_min={self.alpha_min:+.4e} "
            f"@(sigma={self.alpha_min_sigma:.3f}, "
            f"family='{self.alpha_min_family}', "
            f"gauge='{self.alpha_min_gauge}'), "
            f"alpha_max={self.alpha_max:+.4e})"
        )


def sweep_alpha_admissible_family(
    bundle: PrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    families: Mapping[str, FamilyFactory] | None = None,
    gauges: Mapping[str, GaugeFn] | None = None,
    n_zeros: int = 60,
    convergence_tol: float = 1e-12,
    max_zeros: int = 200,
) -> AdmissibleFamilySweepCertificate:
    """Sweep alpha over sigma, admissible families, and structural gauges."""
    sigma_array = np.asarray(list(sigmas), dtype=float)
    if sigma_array.size == 0:
        raise ValueError("sigmas must be non-empty")
    if not np.all(sigma_array > 0.0):
        raise ValueError("every sigma must be strictly positive")

    family_map = (
        dict(families) if families is not None else dict(DEFAULT_TEST_FAMILIES)
    )
    if len(family_map) == 0:
        raise ValueError("families must be non-empty")

    gauge_map = dict(gauges) if gauges is not None else dict(DEFAULT_GAUGES)
    if len(gauge_map) == 0:
        raise ValueError("gauges must be non-empty")

    family_names = tuple(family_map.keys())
    gauge_names = tuple(gauge_map.keys())

    n_f = len(family_names)
    n_g = len(gauge_names)
    n_s = int(sigma_array.size)

    weil_table = np.empty((n_f, n_s), dtype=float)
    energy_table = np.empty((n_f, n_g, n_s), dtype=float)
    alpha_table = np.empty((n_f, n_g, n_s), dtype=float)

    for i, family_name in enumerate(family_names):
        mk_test = family_map[family_name]
        for j, sigma in enumerate(sigma_array):
            test = mk_test(float(sigma))
            w_val, _n_used = weil_zero_side(
                test,
                n_zeros=n_zeros,
                convergence_tol=convergence_tol,
                max_zeros=max_zeros,
            )
            w_float = float(w_val)
            weil_table[i, j] = w_float
            for k, gauge_name in enumerate(gauge_names):
                gfn = gauge_map[gauge_name]
                G = build_test_state_from_test_function(bundle, test, gfn)
                E = float(compute_energy_functional(G))
                energy_table[i, k, j] = E
                if E == 0.0:
                    if w_float > 0.0:
                        alpha_table[i, k, j] = float("inf")
                    elif w_float < 0.0:
                        alpha_table[i, k, j] = float("-inf")
                    else:
                        alpha_table[i, k, j] = float("nan")
                else:
                    alpha_table[i, k, j] = w_float / E

    weil_all_positive = bool(np.all(weil_table >= 0.0))

    finite_mask = np.isfinite(alpha_table)
    if finite_mask.any():
        finite_vals = alpha_table[finite_mask]
        alpha_min = float(finite_vals.min())
        alpha_max = float(finite_vals.max())
        idx_flat = int(np.argmin(np.where(finite_mask, alpha_table, np.inf)))
        i_min, rem = divmod(idx_flat, n_g * n_s)
        k_min, j_min = divmod(rem, n_s)
        alpha_min_sigma = float(sigma_array[j_min])
        alpha_min_family = family_names[i_min]
        alpha_min_gauge = gauge_names[k_min]
    else:
        alpha_min = float("nan")
        alpha_max = float("nan")
        alpha_min_sigma = float("nan")
        alpha_min_family = "<none>"
        alpha_min_gauge = "<none>"

    if finite_mask.any():
        finite_positive = bool(np.all(alpha_table[finite_mask] > 0.0))
    else:
        finite_positive = False
    no_negative_inf = not bool(np.any(np.isneginf(alpha_table)))
    no_nan = not bool(np.any(np.isnan(alpha_table)))
    alpha_all_positive = finite_positive and no_negative_inf and no_nan

    return AdmissibleFamilySweepCertificate(
        sigmas=sigma_array,
        families=family_names,
        gauges=gauge_names,
        weil_table=weil_table,
        energy_table=energy_table,
        alpha_table=alpha_table,
        weil_all_positive=weil_all_positive,
        alpha_all_positive=alpha_all_positive,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_min_sigma=alpha_min_sigma,
        alpha_min_family=alpha_min_family,
        alpha_min_gauge=alpha_min_gauge,
    )


__all__ = [
    "AdmissibleTestFunction",
    "GaussianMixtureTestFunction",
    "gaussian_mixture_test_function",
    "Hermite2GaussianTestFunction",
    "hermite2_gaussian_test_function",
    "FamilyFactory",
    "DEFAULT_TEST_FAMILIES",
    "build_test_state_from_test_function",
    "AdmissibleFamilySweepCertificate",
    "sweep_alpha_admissible_family",
]
