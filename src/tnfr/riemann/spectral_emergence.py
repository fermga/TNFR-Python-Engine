r"""TNFR-Riemann P29: spectral universality under canonical coupling.

This module conducts an exploratory numerical experiment on the
operator-level expression of gap G4.  It asks the question:

    Does any canonical TNFR inter-prime coupling law J_{p,q} drive the
    eigenvalue spacing statistics of the P14 prime-ladder Hamiltonian
    toward the GUE class (the conjectural universality class of the
    Riemann zeros, after Montgomery 1973 and Odlyzko 1987)?

Motivation
----------
P14 builds the prime-ladder Hamiltonian as a strictly diagonal operator
(``H_COUPLING_STRENGTH = 0``) on basis ``|p, k>``, with eigenvalues
``{k log p}``.  This construction preserves the Euler-product
orthogonality of distinct primes at the operator level, and yields
Poissonian level statistics (no level repulsion).  The Riemann zeros,
on the other hand, exhibit GUE-class level statistics to very high
empirical precision.

The honest content of gap G4 at the operator level is therefore:

    For any choice of canonical UM+RA inter-prime coupling J derived
    from the TNFR primitives (phi, gamma, pi, e) and constrained by
    U1-U6, does spec(H_P14(J)) reproduce the GUE level repulsion?

P29 does NOT answer this question definitively.  It implements a
parametric family of canonical coupling laws, sweeps their strength,
and measures how far the resulting normalised nearest-neighbour
spacing distribution lies from the Wigner surmise for GUE in
Kolmogorov-Smirnov distance.  The output is a structured numerical
report from which any reader can read off the answer for the families
considered.

Honest scope (mandatory, see AGENTS.md sec. 13.2):

    * P29 does NOT prove or close gap G4.  G4 = RH is the single
      remaining open obstruction in the TNFR-Riemann programme.
    * Convergence of level statistics to GUE under some canonical
      coupling law would constitute structural-compatibility
      evidence, not a derivation of RH.
    * Absence of such convergence across the families tested
      documents a concrete computational obstruction and constrains
      the search space for future structural arguments.
    * The Euler-product orthogonality preserved by P14 at J=0 is
      broken by any J > 0; P29 explicitly measures the trade-off
      between Euler-product fidelity (Paley-style identity at J=0,
      see P25) and GUE-universality emergence (J > 0).

Coupling families implemented
-----------------------------
All three families are built from the canonical TNFR constants
(phi, gamma, pi, e) with zero empirical fitting, per the project's
canonical-derivation policy:

* ``"kuramoto_u3"``: J_{(p,k),(q,l)} = strength * (gamma/pi) *
  exp(-|k log p - l log q|).  U3-phase-gated Kuramoto-style
  coupling: pairs with similar structural frequencies couple more
  strongly.  Implements operator UM (phase synchronisation) with
  U3 compatibility threshold gamma/pi.

* ``"phi_multiscale"``: J_{(p,k),(q,l)} = strength * phi^(-(k+l)) /
  sqrt(p*q).  THOL+REMESH multiscale law: higher echo indices and
  larger primes couple more weakly.  Implements operator THOL
  (self-organisation through sub-EPI nesting) combined with
  REMESH echo damping at rate phi.

* ``"pnt_logarithmic"``: J_{(p,k),(q,l)} = strength * gamma /
  log(1 + p*q).  Prime-Number-Theorem-aligned coupling: the
  log-weight reflects the natural density of primes around log(pq).
  Implements operator RA (resonant amplification) with PNT-weighted
  range.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P29
program (G4 operator-level exploratory diagnostic, May 2026).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import networkx as nx
import numpy as np

from .prime_ladder_hamiltonian import (
    PrimeLadderHamiltonian,
    build_prime_ladder_hamiltonian,
)


__all__ = [
    "CANONICAL_COUPLING_LAWS",
    "InterPrimeCoupling",
    "SpectralEmergenceReport",
    "build_inter_prime_coupling",
    "couple_prime_ladder_hamiltonian",
    "unfold_spectrum",
    "nearest_neighbour_spacings",
    "wigner_surmise_gue_cdf",
    "ks_distance_to_gue",
    "sweep_coupling_strength",
    "compute_spectral_emergence_report",
]


# ----------------------------------------------------------------------
# Canonical TNFR constants (locally cached to keep this module
# self-contained against constants.canonical refactors)
# ----------------------------------------------------------------------

_PHI = (1.0 + math.sqrt(5.0)) / 2.0          # 1.6180339887...
_GAMMA = 0.5772156649015329                  # Euler-Mascheroni
_PI = math.pi                                # 3.1415926535...

CANONICAL_COUPLING_LAWS: tuple[str, ...] = (
    "kuramoto_u3",
    "phi_multiscale",
    "pnt_logarithmic",
)


# ----------------------------------------------------------------------
# Coupling matrix construction
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class InterPrimeCoupling:
    """Inter-prime UM+RA coupling matrix and its canonical metadata."""

    matrix: np.ndarray
    law: str
    strength: float
    n_primes: int
    max_power: int
    frobenius_norm: float


def _coupling_value(
    law: str,
    p: int,
    k: int,
    q: int,
    l: int,
) -> float:
    """Canonical inter-prime coupling value for a single ordered pair."""
    log_p = math.log(p)
    log_q = math.log(q)
    if law == "kuramoto_u3":
        # U3-phase-gated Kuramoto: gamma/pi prefactor, exp damping in
        # frequency mismatch |k log p - l log q|.
        return (_GAMMA / _PI) * math.exp(-abs(k * log_p - l * log_q))
    if law == "phi_multiscale":
        # THOL+REMESH multiscale: phi^(-(k+l)) echo damping with
        # inverse geometric mean of primes.
        return (_PHI ** (-(k + l))) / math.sqrt(p * q)
    if law == "pnt_logarithmic":
        # PNT-aligned: gamma-weighted log range. Adding 1 inside the
        # log keeps the matrix entry finite if p*q ever reaches 1
        # (does not happen for primes >= 2, but is defensive).
        return _GAMMA / math.log(1.0 + p * q)
    raise ValueError(
        f"Unknown coupling law '{law}'. Expected one of "
        f"{CANONICAL_COUPLING_LAWS}."
    )


def build_inter_prime_coupling(
    G: nx.Graph,
    *,
    law: str,
    strength: float,
) -> InterPrimeCoupling:
    """Build canonical inter-prime UM+RA coupling matrix for ``G``.

    Only entries between nodes belonging to *distinct* primes are
    populated.  Within-prime ladder structure is left to ``H_freq``
    (and to the REMESH ladder edges already present in ``G``).

    Parameters
    ----------
    G
        Prime-ladder graph from :func:`build_prime_ladder_graph`.
        Each node label is ``(p, k)`` with ``p`` prime and
        ``k = 1, ..., max_power``.
    law
        One of :data:`CANONICAL_COUPLING_LAWS`.
    strength
        Global multiplicative prefactor.  ``strength = 0`` reproduces
        the decoupled P14 limit.

    Returns
    -------
    InterPrimeCoupling
        Hermitian real-symmetric matrix indexed by ``cached_node_list(G)``
        node ordering (matches :class:`InternalHamiltonian`).
    """
    if law not in CANONICAL_COUPLING_LAWS:
        raise ValueError(
            f"Unknown coupling law '{law}'. Expected one of "
            f"{CANONICAL_COUPLING_LAWS}."
        )
    from ..utils.cache import cached_node_list

    nodes = cached_node_list(G)
    N = len(nodes)
    J = np.zeros((N, N), dtype=float)
    for i, node_i in enumerate(nodes):
        p_i, k_i = node_i
        for j in range(i + 1, N):
            node_j = nodes[j]
            p_j, k_j = node_j
            if p_i == p_j:
                # Same prime: leave to H_freq + REMESH ladder edges.
                continue
            val = strength * _coupling_value(
                law, int(p_i), int(k_i), int(p_j), int(k_j)
            )
            J[i, j] = val
            J[j, i] = val
    primes = sorted({int(p) for p, _ in nodes})
    max_power = max(int(k) for _, k in nodes)
    return InterPrimeCoupling(
        matrix=J,
        law=law,
        strength=float(strength),
        n_primes=len(primes),
        max_power=max_power,
        frobenius_norm=float(np.linalg.norm(J, ord="fro")),
    )


def couple_prime_ladder_hamiltonian(
    ladder: PrimeLadderHamiltonian,
    coupling: InterPrimeCoupling,
) -> np.ndarray:
    """Add inter-prime coupling to P14 and return the full Hermitian H."""
    H_full = ladder.hamiltonian.H_int + coupling.matrix.astype(complex)
    # Defensive Hermiticity check (P14 + symmetric J should be Hermitian
    # by construction).
    deviation = float(np.max(np.abs(H_full - H_full.conj().T)))
    if deviation > 1e-10:
        raise RuntimeError(
            f"Coupled Hamiltonian failed Hermiticity check: {deviation:.2e}"
        )
    return H_full


# ----------------------------------------------------------------------
# Level statistics
# ----------------------------------------------------------------------


def unfold_spectrum(eigenvalues: np.ndarray) -> np.ndarray:
    """Unfold a 1D spectrum to unit mean level density.

    Uses the cumulative-count staircase ``N(E) = #{e_i <= E}`` and
    fits it with a fifth-order polynomial in ``E``; the unfolded
    levels are the polynomial evaluated at each eigenvalue.  This
    follows the standard RMT unfolding procedure (Mehta 2004,
    chap. 16).
    """
    eigs = np.sort(np.asarray(eigenvalues, dtype=float))
    n = eigs.size
    if n < 6:
        raise ValueError(
            f"Need at least 6 eigenvalues to unfold; got {n}."
        )
    counts = np.arange(1, n + 1, dtype=float)
    coeffs = np.polyfit(eigs, counts, deg=5)
    unfolded = np.polyval(coeffs, eigs)
    return unfolded


def nearest_neighbour_spacings(unfolded: np.ndarray) -> np.ndarray:
    """Normalised nearest-neighbour spacings ``s_i = u_{i+1} - u_i``."""
    u = np.sort(np.asarray(unfolded, dtype=float))
    spacings = np.diff(u)
    mean_s = spacings.mean()
    if mean_s <= 0.0:
        raise RuntimeError("Non-positive mean spacing; unfolding failed.")
    return spacings / mean_s


def wigner_surmise_gue_cdf(s: np.ndarray) -> np.ndarray:
    """GUE Wigner surmise cumulative distribution.

    The GUE Wigner surmise PDF is

        p_GUE(s) = (32 / pi^2) s^2 exp(-4 s^2 / pi),

    with CDF obtained by integration.  Implementation uses the
    closed form

        F_GUE(s) = erf(2 s / sqrt(pi)) - (4 s / pi) exp(-4 s^2 / pi).
    """
    s = np.asarray(s, dtype=float)
    from math import erf as _erf

    out = np.empty_like(s)
    for i, x in enumerate(s):
        if x < 0.0:
            out[i] = 0.0
            continue
        cdf = _erf(2.0 * x / math.sqrt(math.pi)) - (
            4.0 * x / math.pi
        ) * math.exp(-4.0 * x * x / math.pi)
        out[i] = max(0.0, min(1.0, cdf))
    return out


def poisson_cdf(s: np.ndarray) -> np.ndarray:
    """Poisson (uncorrelated) spacing CDF: ``F(s) = 1 - exp(-s)``."""
    s = np.asarray(s, dtype=float)
    return 1.0 - np.exp(-np.clip(s, 0.0, None))


def _ks_distance(
    sample: np.ndarray,
    ref_cdf: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Kolmogorov-Smirnov sup distance between empirical and reference CDFs."""
    s = np.sort(np.asarray(sample, dtype=float))
    n = s.size
    empirical = np.arange(1, n + 1, dtype=float) / n
    reference = ref_cdf(s)
    return float(np.max(np.abs(empirical - reference)))


def ks_distance_to_gue(spacings: np.ndarray) -> float:
    """KS distance between empirical spacings and the GUE Wigner surmise."""
    return _ks_distance(spacings, wigner_surmise_gue_cdf)


def ks_distance_to_poisson(spacings: np.ndarray) -> float:
    """KS distance between empirical spacings and the Poisson reference."""
    return _ks_distance(spacings, poisson_cdf)


# ----------------------------------------------------------------------
# Sweep and report
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralEmergenceReport:
    """Result of a coupling-strength sweep for one canonical law."""

    law: str
    n_primes: int
    max_power: int
    strengths: np.ndarray
    ks_to_gue: np.ndarray
    ks_to_poisson: np.ndarray
    mean_spacing_sq: np.ndarray
    coupling_frobenius: np.ndarray
    eigenvalue_range: np.ndarray  # shape (S, 2) — (min, max) per strength
    best_strength_gue: float
    best_ks_to_gue: float
    poisson_baseline_ks_gue: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def sweep_coupling_strength(
    *,
    n_primes: int,
    max_power: int,
    law: str,
    strengths: Sequence[float],
) -> SpectralEmergenceReport:
    """Sweep one canonical coupling law's strength and measure GUE distance.

    Parameters
    ----------
    n_primes
        Number of primes in the ladder.
    max_power
        REMESH echo cap.
    law
        Canonical law name (see :data:`CANONICAL_COUPLING_LAWS`).
    strengths
        Iterable of non-negative strengths to evaluate.  Should include
        0.0 to obtain the decoupled-P14 baseline.

    Returns
    -------
    SpectralEmergenceReport
    """
    strengths_arr = np.asarray(list(strengths), dtype=float)
    if np.any(strengths_arr < 0.0):
        raise ValueError("All strengths must be non-negative.")

    ladder = build_prime_ladder_hamiltonian(
        n_primes=n_primes,
        max_power=max_power,
        coupling=0.0,
    )
    G = ladder.graph

    ks_gue = np.empty(strengths_arr.size, dtype=float)
    ks_poi = np.empty(strengths_arr.size, dtype=float)
    mean_s2 = np.empty(strengths_arr.size, dtype=float)
    frob = np.empty(strengths_arr.size, dtype=float)
    eig_range = np.empty((strengths_arr.size, 2), dtype=float)

    for idx, s in enumerate(strengths_arr):
        coupling = build_inter_prime_coupling(G, law=law, strength=float(s))
        H = couple_prime_ladder_hamiltonian(ladder, coupling)
        eigvals = np.linalg.eigvalsh(H)
        unfolded = unfold_spectrum(eigvals.real)
        spacings = nearest_neighbour_spacings(unfolded)
        ks_gue[idx] = ks_distance_to_gue(spacings)
        ks_poi[idx] = ks_distance_to_poisson(spacings)
        mean_s2[idx] = float(np.mean(spacings ** 2))
        frob[idx] = coupling.frobenius_norm
        eig_range[idx, 0] = float(eigvals.real.min())
        eig_range[idx, 1] = float(eigvals.real.max())

    best_idx = int(np.argmin(ks_gue))
    # smallest strength = closest to decoupled
    baseline_idx = int(np.argmin(strengths_arr))

    notes = (
        f"baseline (strength={strengths_arr[baseline_idx]:.3g}) "
        f"KS_GUE = {ks_gue[baseline_idx]:.4f} "
        f"(Poisson-class if close to ~0.15-0.30 over this n)",
        f"best   (strength={strengths_arr[best_idx]:.3g}) "
        f"KS_GUE = {ks_gue[best_idx]:.4f}",
        "Honest scope: KS_GUE -> 0 across canonical laws would constitute "
        "structural-compatibility evidence; does NOT close gap G4.",
    )

    return SpectralEmergenceReport(
        law=law,
        n_primes=n_primes,
        max_power=max_power,
        strengths=strengths_arr,
        ks_to_gue=ks_gue,
        ks_to_poisson=ks_poi,
        mean_spacing_sq=mean_s2,
        coupling_frobenius=frob,
        eigenvalue_range=eig_range,
        best_strength_gue=float(strengths_arr[best_idx]),
        best_ks_to_gue=float(ks_gue[best_idx]),
        poisson_baseline_ks_gue=float(ks_gue[baseline_idx]),
        notes=notes,
    )


def compute_spectral_emergence_report(
    *,
    n_primes: int = 25,
    max_power: int = 4,
    laws: Sequence[str] = CANONICAL_COUPLING_LAWS,
    strengths: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
) -> dict[str, SpectralEmergenceReport]:
    """Top-level entry point: sweep every canonical law and return reports.

    Default grid is moderate (~100x100 Hamiltonian) so the experiment
    runs in seconds.  Scale up ``n_primes`` and ``max_power`` for
    higher-resolution exploration.
    """
    return {
        law: sweep_coupling_strength(
            n_primes=n_primes,
            max_power=max_power,
            law=law,
            strengths=strengths,
        )
        for law in laws
    }
