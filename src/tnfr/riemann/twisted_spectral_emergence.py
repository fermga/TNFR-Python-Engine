r"""TNFR-Riemann P47: chi-twisted spectral emergence under canonical coupling.

L-track analogue of P29 (:mod:`spectral_emergence`).  Sweeps three
canonical TNFR inter-prime coupling laws on the P34 chi-twisted
prime-ladder Hamiltonian and measures how far the resulting unfolded
nearest-neighbour spacing distribution lies from the GUE Wigner
surmise (the conjectural universality class of the non-trivial zeros
of :math:`L(s, \chi)`, by GUE-universality of Dirichlet L-functions
[Hughes-Rudnick 2003, Conrey-Snaith 2007]).

The chi-twist enters the coupling matrix multiplicatively through
:math:`\chi(p)\,\chi(q)`.  For primitive *real* characters
(:math:`\chi_3, \chi_4, \chi_5`) this factor lives in
:math:`\{-1, +1\}` on the P34 graph (zero values are excluded
automatically because primes dividing :math:`q` are removed from the
chi-twisted ladder), so the coupling matrix is real-symmetric and the
resulting full Hamiltonian remains Hermitian.

Motivation
----------
P34 builds the chi-twisted prime-ladder graph (excluding primes
:math:`p \mid q`) and P14 instantiates the canonical TNFR internal
Hamiltonian on it; in the decoupled limit
(``H_COUPLING_STRENGTH = 0``) the spectrum is strictly diagonal
:math:`\{k \log p : p \nmid q,\; k = 1, \ldots, K\}`.  This preserves
Euler-product orthogonality of distinct primes and yields Poissonian
level statistics.  The zeros of :math:`L(s, \chi)`, on the other hand,
are conjecturally GUE-distributed.

P47 asks: does any *canonical* chi-twisted inter-prime coupling law
:math:`J^{(\chi)}` derived from the TNFR primitives
:math:`(\varphi, \gamma, \pi, e)` and constrained by U1-U6 drive the
spacings of :math:`\mathrm{spec}(H_{P34}^{(\chi)} + J^{(\chi)})`
toward GUE?

Coupling families
-----------------
All three families mirror the P29 laws with an explicit chi-twist
factor :math:`\chi(p)\,\chi(q)`:

* ``"kuramoto_u3"``:
  :math:`J^{(\chi)}_{(p,k),(q,l)} =
      s \cdot \chi(p)\chi(q) \cdot (\gamma/\pi) \cdot
      \exp(-|k \log p - l \log q|)`.

* ``"phi_multiscale"``:
  :math:`J^{(\chi)}_{(p,k),(q,l)} =
      s \cdot \chi(p)\chi(q) \cdot \varphi^{-(k+l)} /
      \sqrt{p\,q}`.

* ``"pnt_logarithmic"``:
  :math:`J^{(\chi)}_{(p,k),(q,l)} =
      s \cdot \chi(p)\chi(q) \cdot \gamma /
      \log(1 + p\,q)`.

Honest scope (mandatory, see AGENTS.md sec. 13.2):

    * P47 is a diagnostic *only*.  It does NOT prove GRH for any
      :math:`L(s, \chi)` and does NOT close gap G4 = RH.
    * The level-statistics framework (unfolding, NN spacings, KS
      distance to GUE) is identical to P29 and reused verbatim from
      :mod:`spectral_emergence`; only the coupling construction is
      chi-twisted here.
    * Convergence of :math:`\mathrm{KS\_GUE} \to 0` under some
      canonical chi-twisted coupling would constitute structural-
      compatibility evidence for the GUE-universality of
      :math:`L(s, \chi)` zeros; absence thereof documents a concrete
      computational obstruction.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P47
program (chi-twisted L-track operator-level diagnostic, May 2026).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .dirichlet_l import DirichletCharacter
from .spectral_emergence import (
    ks_distance_to_gue,
    ks_distance_to_poisson,
    nearest_neighbour_spacings,
    unfold_spectrum,
)
from .twisted_prime_ladder_hamiltonian import (
    TwistedPrimeLadderHamiltonian,
    build_twisted_prime_ladder_hamiltonian,
)
from .twisted_weil_explicit_formula import character_parity

__all__ = [
    "TWISTED_CANONICAL_COUPLING_LAWS",
    "TwistedInterPrimeCoupling",
    "TwistedSpectralEmergenceReport",
    "build_twisted_inter_prime_coupling",
    "couple_twisted_prime_ladder_hamiltonian",
    "twisted_sweep_coupling_strength",
    "compute_twisted_spectral_emergence_report",
]


# ----------------------------------------------------------------------
# Canonical TNFR constants (locally cached)
# ----------------------------------------------------------------------

_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_GAMMA = 0.5772156649015329
_PI = math.pi

TWISTED_CANONICAL_COUPLING_LAWS: tuple[str, ...] = (
    "kuramoto_u3",
    "phi_multiscale",
    "pnt_logarithmic",
)


# ----------------------------------------------------------------------
# Coupling matrix construction (chi-twisted)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedInterPrimeCoupling:
    """chi-twisted inter-prime UM+RA coupling matrix with metadata."""

    matrix: np.ndarray
    law: str
    strength: float
    character_modulus: int
    character_name: str
    character_parity: int
    n_primes: int
    max_power: int
    frobenius_norm: float


def _coupling_kernel(law: str, p: int, k: int, q: int, m: int) -> float:
    """Untwisted canonical kernel (chi factor applied separately).

    ``m`` plays the role of the second ladder index ``l`` in the
    docstring formulas (variable renamed to avoid E741 ``ambiguous
    variable name 'l'``).
    """
    log_p = math.log(p)
    log_q = math.log(q)
    if law == "kuramoto_u3":
        return (_GAMMA / _PI) * math.exp(-abs(k * log_p - m * log_q))
    if law == "phi_multiscale":
        return (_PHI ** (-(k + m))) / math.sqrt(p * q)
    if law == "pnt_logarithmic":
        return _GAMMA / math.log(1.0 + p * q)
    raise ValueError(
        f"Unknown coupling law '{law}'. Expected one of "
        f"{TWISTED_CANONICAL_COUPLING_LAWS}."
    )


def build_twisted_inter_prime_coupling(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    *,
    law: str,
    strength: float,
) -> TwistedInterPrimeCoupling:
    r"""Build canonical chi-twisted inter-prime UM+RA coupling matrix.

    Entries between nodes of *distinct* primes carry the chi-twist
    prefactor :math:`\chi(p)\,\chi(q)`.  Within-prime ladder structure
    is left to :math:`\hat H_{\mathrm{freq}}` and the REMESH ladder
    edges already present in the P34 graph.

    Parameters
    ----------
    chi : DirichletCharacter
        Character defining the twist.
    bundle : TwistedPrimeLadderHamiltonian
        P34 bundle (graph + Hamiltonian) for the same character.
    law : str
        One of :data:`TWISTED_CANONICAL_COUPLING_LAWS`.
    strength : float
        Global non-negative multiplicative prefactor.  ``strength=0``
        reproduces the decoupled P34 limit.

    Returns
    -------
    TwistedInterPrimeCoupling
        Real-symmetric matrix indexed by ``cached_node_list(bundle.graph)``
        node ordering (matches :class:`InternalHamiltonian`).
    """
    if law not in TWISTED_CANONICAL_COUPLING_LAWS:
        raise ValueError(
            f"Unknown coupling law '{law}'. Expected one of "
            f"{TWISTED_CANONICAL_COUPLING_LAWS}."
        )
    if int(bundle.character_modulus) != int(chi.modulus):
        raise ValueError(
            "Character/bundle modulus mismatch: bundle "
            f"({bundle.character_modulus}) and chi ({chi.modulus})"
        )
    if strength < 0.0:
        raise ValueError("strength must be non-negative.")

    from ..utils.cache import cached_node_list

    nodes = cached_node_list(bundle.graph)
    N = len(nodes)
    chi_at = {int(p): float(chi(int(p)).real) for p, _ in nodes}

    J = np.zeros((N, N), dtype=float)
    for i, node_i in enumerate(nodes):
        p_i, k_i = node_i
        chi_pi = chi_at[int(p_i)]
        if chi_pi == 0.0:
            continue
        for j in range(i + 1, N):
            node_j = nodes[j]
            p_j, k_j = node_j
            if p_i == p_j:
                continue
            chi_pj = chi_at[int(p_j)]
            if chi_pj == 0.0:
                continue
            val = (
                strength
                * chi_pi
                * chi_pj
                * _coupling_kernel(law, int(p_i), int(k_i), int(p_j), int(k_j))
            )
            J[i, j] = val
            J[j, i] = val

    primes = sorted({int(p) for p, _ in nodes})
    max_power = max(int(k) for _, k in nodes)
    return TwistedInterPrimeCoupling(
        matrix=J,
        law=law,
        strength=float(strength),
        character_modulus=int(chi.modulus),
        character_name=str(chi.name),
        character_parity=int(character_parity(chi)),
        n_primes=len(primes),
        max_power=max_power,
        frobenius_norm=float(np.linalg.norm(J, ord="fro")),
    )


def couple_twisted_prime_ladder_hamiltonian(
    bundle: TwistedPrimeLadderHamiltonian,
    coupling: TwistedInterPrimeCoupling,
) -> np.ndarray:
    """Add chi-twisted inter-prime coupling to P34 and return full H."""
    H_full = bundle.hamiltonian.H_int + coupling.matrix.astype(complex)
    deviation = float(np.max(np.abs(H_full - H_full.conj().T)))
    if deviation > 1e-10:
        raise RuntimeError(
            "Coupled chi-twisted Hamiltonian failed Hermiticity check: "
            f"{deviation:.2e}"
        )
    return H_full


# ----------------------------------------------------------------------
# Sweep and report
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedSpectralEmergenceReport:
    """Result of a chi-twisted coupling-strength sweep for one law."""

    character_modulus: int
    character_name: str
    character_parity: int
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


def twisted_sweep_coupling_strength(
    chi: DirichletCharacter,
    *,
    n_primes: int,
    max_power: int,
    law: str,
    strengths: Sequence[float],
) -> TwistedSpectralEmergenceReport:
    """Sweep one canonical chi-twisted law's strength; measure KS to GUE.

    Parameters
    ----------
    chi : DirichletCharacter
        Character defining the twist.
    n_primes : int
        Number of primes requested (primes dividing the modulus are
        excluded from the spectrum by P34).
    max_power : int
        REMESH echo cap.
    law : str
        Canonical law name (see
        :data:`TWISTED_CANONICAL_COUPLING_LAWS`).
    strengths : sequence of float
        Iterable of non-negative strengths.  Should include 0.0 to
        obtain the decoupled-P34 baseline.

    Returns
    -------
    TwistedSpectralEmergenceReport
    """
    strengths_arr = np.asarray(list(strengths), dtype=float)
    if np.any(strengths_arr < 0.0):
        raise ValueError("All strengths must be non-negative.")

    bundle = build_twisted_prime_ladder_hamiltonian(
        chi,
        n_primes=n_primes,
        max_power=max_power,
        coupling=0.0,
    )

    S = strengths_arr.size
    ks_gue = np.empty(S, dtype=float)
    ks_poi = np.empty(S, dtype=float)
    mean_s2 = np.empty(S, dtype=float)
    frob = np.empty(S, dtype=float)
    eig_range = np.empty((S, 2), dtype=float)

    for idx, s in enumerate(strengths_arr):
        coupling = build_twisted_inter_prime_coupling(
            chi, bundle, law=law, strength=float(s)
        )
        H = couple_twisted_prime_ladder_hamiltonian(bundle, coupling)
        eigvals = np.linalg.eigvalsh(H)
        unfolded = unfold_spectrum(eigvals.real)
        spacings = nearest_neighbour_spacings(unfolded)
        ks_gue[idx] = ks_distance_to_gue(spacings)
        ks_poi[idx] = ks_distance_to_poisson(spacings)
        mean_s2[idx] = float(np.mean(spacings**2))
        frob[idx] = coupling.frobenius_norm
        eig_range[idx, 0] = float(eigvals.real.min())
        eig_range[idx, 1] = float(eigvals.real.max())

    best_idx = int(np.argmin(ks_gue))
    baseline_idx = int(np.argmin(strengths_arr))

    notes = (
        f"chi={chi.name} (q={chi.modulus}, " f"a={int(character_parity(chi))})",
        f"baseline (strength={strengths_arr[baseline_idx]:.3g}) "
        f"KS_GUE = {ks_gue[baseline_idx]:.4f}",
        f"best   (strength={strengths_arr[best_idx]:.3g}) "
        f"KS_GUE = {ks_gue[best_idx]:.4f}",
        "Honest scope: KS_GUE -> 0 across canonical chi-twisted laws "
        "would constitute structural-compatibility evidence; does NOT "
        "prove GRH for L(s, chi) and does NOT close gap G4 = RH.",
    )

    return TwistedSpectralEmergenceReport(
        character_modulus=int(chi.modulus),
        character_name=str(chi.name),
        character_parity=int(character_parity(chi)),
        law=law,
        n_primes=int(bundle.spectrum.eigenvalues.size // max_power),
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


def compute_twisted_spectral_emergence_report(
    chi: DirichletCharacter,
    *,
    n_primes: int = 25,
    max_power: int = 4,
    laws: Sequence[str] = TWISTED_CANONICAL_COUPLING_LAWS,
    strengths: Sequence[float] = (
        0.0,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
    ),
) -> dict[str, TwistedSpectralEmergenceReport]:
    """Sweep every canonical chi-twisted law for ``chi`` and return reports.

    Default grid is moderate (~80x80 Hamiltonian after P34 excludes
    primes dividing q) so the experiment runs in seconds.  Scale up
    ``n_primes`` and ``max_power`` for higher-resolution exploration.
    """
    return {
        law: twisted_sweep_coupling_strength(
            chi,
            n_primes=n_primes,
            max_power=max_power,
            law=law,
            strengths=strengths,
        )
        for law in laws
    }
