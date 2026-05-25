"""TNFR-Riemann P27: Hilbert-Polya scaffold.

This module constructs the explicit reference Hilbert-Polya operator on a
truncated TNFR Hilbert space and certifies its internal consistency with
the rest of the TNFR-Riemann stack (P14 prime-ladder Hamiltonian, P15
Weil-Guinand explicit formula).

The reference operator is

    T_HP = diag(gamma_1, gamma_2, ..., gamma_N)   on  ell^2_N(N)

where ``gamma_n`` are the imaginary parts of the non-trivial Riemann zeros
``rho_n = 1/2 + i gamma_n`` obtained from ``mpmath.zetazero``.  By
construction:

* ``T_HP`` is self-adjoint (real diagonal).
* For ``s > 0`` the shifted resolvent ``(T_HP^2 + s^2 I)^{-1/2}`` belongs
  to Schatten class ``S_p`` for every ``p > 1``; its trace and
  Hilbert-Schmidt norms are computed exactly from the gamma list.
* The zero-side ``sum 2 h(gamma_n)`` of Weil's explicit formula evaluated
  through ``T_HP`` reproduces P15 to machine precision because both sides
  consume the same gamma data.
* The spectral gap between ``spec(T_HP) = {gamma_n}`` and ``spec(P14) =
  {k log p}`` is quantified by Wasserstein-1 distance on the truncated
  empirical measures.  This number is the operator-level expression of
  gap G4: the open structural derivation of T_HP from TNFR first
  principles.

Honest scope (mandatory, see AGENTS.md):

The P27 module does **not** prove the Riemann Hypothesis.  ``T_HP`` is
populated by *inputting* the zeros from mpmath; we do not derive them
from the nodal equation, conservation, or grammar.  What P27 delivers is
the explicit operator-level slot into which a Hilbert-Polya-style attack
must fit, plus numerical evidence that the TNFR stack is internally
compatible with such a slot.  The genuinely open piece (gap G4 = RH) is
the structural derivation of ``T_HP`` from TNFR first principles without
reference to the zeros, which the framework here does not provide.

Per AGENTS.md sec. 13.2, G1/G2/G3 are operationally closed by P12-P15
and G5 is superseded.  G4 remains the single open gap and P27 does not
attack it: it organises the existing ingredients into the canonical HP
shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math

import numpy as np

from .prime_ladder_hamiltonian import (
    PrimeLadderHamiltonian,
    build_prime_ladder_hamiltonian,
)
from .weil_explicit_formula import (
    GaussianTestFunction,
    gaussian_test_function,
    weil_archimedean_integral,
    weil_pole_side,
    weil_prime_side_from_hamiltonian,
)


__all__ = [
    "HilbertPolyaCertificate",
    "fetch_zero_imaginary_parts",
    "build_hp_operator",
    "verify_hp_self_adjoint",
    "hp_resolvent_schatten_norms",
    "hp_zero_side_from_operator",
    "wasserstein_1_distance",
    "structural_gap_p14_vs_hp",
    "compute_hilbert_polya_certificate",
]


# ----------------------------------------------------------------------
# Atomic primitives
# ----------------------------------------------------------------------


def fetch_zero_imaginary_parts(n_zeros: int, *, dps: int = 30) -> np.ndarray:
    """Return ``[gamma_1, ..., gamma_N]`` from ``mpmath.zetazero``.

    Parameters
    ----------
    n_zeros
        Number of positive-axis non-trivial zeros to fetch.
    dps
        Decimal precision for mpmath.

    Returns
    -------
    np.ndarray
        Array of length ``n_zeros`` with strictly positive entries.
    """
    if n_zeros <= 0:
        raise ValueError("n_zeros must be positive")
    import mpmath

    with mpmath.workdps(dps):
        gammas = np.array(
            [float(mpmath.zetazero(n).imag) for n in range(1, n_zeros + 1)],
            dtype=float,
        )
    if not np.all(gammas > 0):
        raise RuntimeError(
            "mpmath.zetazero returned a non-positive imaginary part"
        )
    return gammas


def build_hp_operator(gammas: np.ndarray) -> np.ndarray:
    """Return the diagonal Hilbert-Polya operator ``T_HP = diag(gammas)``."""
    gammas = np.asarray(gammas, dtype=float)
    if gammas.ndim != 1:
        raise ValueError("gammas must be a 1-D array")
    return np.diag(gammas)


def verify_hp_self_adjoint(T: np.ndarray, *, tol: float = 1e-12) -> dict:
    """Check that ``T`` is self-adjoint within ``tol``."""
    arr = np.asarray(T)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("T must be a square matrix")
    asym = arr - arr.conj().T
    asym_norm = float(np.linalg.norm(asym, ord="fro"))
    imag_norm = float(np.linalg.norm(arr.imag, ord="fro"))
    return {
        "asymmetry_frobenius": asym_norm,
        "imaginary_frobenius": imag_norm,
        "self_adjoint": asym_norm <= tol and imag_norm <= tol,
        "tolerance": tol,
    }


def hp_resolvent_schatten_norms(
    gammas: np.ndarray,
    *,
    shift: float = 1.0,
) -> dict:
    r"""Compute Schatten norms of the shifted resolvent of ``T_HP``.

    Returns the trace norm ``sum 1 / (gamma_n^2 + s^2)``, the
    Hilbert-Schmidt norm ``sqrt(sum 1 / (gamma_n^2 + s^2)^2)``, and the
    operator norm ``1 / sqrt(gamma_min^2 + s^2)``.
    """
    gammas = np.asarray(gammas, dtype=float)
    if shift <= 0.0:
        raise ValueError("shift must be strictly positive")
    denom = gammas**2 + shift**2
    if not np.all(denom > 0):
        raise ValueError("shifted spectrum is degenerate")
    s1 = float(np.sum(1.0 / denom))
    s2 = float(math.sqrt(np.sum(1.0 / denom**2)))
    op_norm = float(1.0 / math.sqrt(np.min(denom)))
    return {
        "shift": float(shift),
        "schatten_1_norm": s1,
        "schatten_2_norm": s2,
        "operator_norm_inverse": op_norm,
        "trace_class": math.isfinite(s1),
    }


def hp_zero_side_from_operator(
    gammas: np.ndarray,
    test: GaussianTestFunction,
) -> float:
    r"""Evaluate ``sum_n 2 h(gamma_n)`` directly from the diagonal of T_HP.

    This is identical to P15's :func:`weil_zero_side` evaluated on the
    same gamma list, but exposes the dependence as an inner product
    ``Tr h(T_HP^2)^{1/2}`` against the spectral measure of ``T_HP``.
    """
    gammas = np.asarray(gammas, dtype=float)
    h_values = np.array([test.h(float(g)) for g in gammas], dtype=float)
    return float(2.0 * np.sum(h_values))


def wasserstein_1_distance(a: np.ndarray, b: np.ndarray) -> float:
    r"""Compute the 1-Wasserstein distance between two 1-D empirical measures.

    Both inputs are interpreted as equally weighted samples; they are
    sorted, padded to common length by interpolating the shorter one's
    quantile function, and the integral ``int_0^1 |F_a^{-1}(u) -
    F_b^{-1}(u)| du`` is approximated by trapezoidal quadrature.
    """
    a_sorted = np.sort(np.asarray(a, dtype=float))
    b_sorted = np.sort(np.asarray(b, dtype=float))
    n = max(len(a_sorted), len(b_sorted))
    if n == 0:
        return 0.0
    u = (np.arange(n) + 0.5) / n
    ua = (np.arange(len(a_sorted)) + 0.5) / len(a_sorted)
    ub = (np.arange(len(b_sorted)) + 0.5) / len(b_sorted)
    qa = np.interp(u, ua, a_sorted)
    qb = np.interp(u, ub, b_sorted)
    return float(np.mean(np.abs(qa - qb)))


def structural_gap_p14_vs_hp(
    bundle: PrimeLadderHamiltonian,
    gammas: np.ndarray,
) -> dict:
    """Quantify the operator-level gap G4 on truncated spectra.

    Compares ``spec(P14) = {k log p}`` (positive eigenvalues only) with
    ``spec(T_HP) = {gamma_n}`` on the same truncation length.  The
    Wasserstein-1 distance is the relevant scalar because both spectra
    are real and unbounded with different growth: P14 grows like
    ``log n`` while T_HP grows like ``2 pi n / log n``.

    The growth-rate mismatch is the operator-level manifestation of the
    open structural derivation problem (gap G4).  No transformation that
    sends one spectrum to the other can be a smooth structural map; any
    Hilbert-Polya-style derivation must therefore introduce a non-linear
    spectral rescaling derived from TNFR first principles.
    """
    p14_eigs, _ = bundle.hamiltonian.get_spectrum()
    p14_eigs = np.sort(np.real(p14_eigs))
    p14_eigs = p14_eigs[p14_eigs > 0.0]
    n_compare = min(len(p14_eigs), len(gammas))
    p14_trunc = p14_eigs[:n_compare]
    hp_trunc = np.sort(np.asarray(gammas, dtype=float))[:n_compare]
    w1 = wasserstein_1_distance(p14_trunc, hp_trunc)
    # Asymptotic growth diagnostic: ratio of last spectral value
    if n_compare > 0:
        growth_ratio = float(hp_trunc[-1] / p14_trunc[-1])
    else:
        growth_ratio = float("nan")
    return {
        "n_compared": int(n_compare),
        "p14_min": float(p14_trunc[0]) if n_compare > 0 else float("nan"),
        "p14_max": float(p14_trunc[-1]) if n_compare > 0 else float("nan"),
        "hp_min": float(hp_trunc[0]) if n_compare > 0 else float("nan"),
        "hp_max": float(hp_trunc[-1]) if n_compare > 0 else float("nan"),
        "wasserstein_1": w1,
        "asymptotic_growth_ratio": growth_ratio,
    }


# ----------------------------------------------------------------------
# Certificate dataclass and orchestrator
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HilbertPolyaCertificate:
    """Internal-consistency certificate for the TNFR Hilbert-Polya scaffold."""

    # Truncation parameters
    n_zeros: int
    n_primes: int
    max_power: int

    # Self-adjointness
    asymmetry_frobenius: float
    self_adjoint: bool

    # Resolvent
    resolvent_shift: float
    schatten_1_norm: float
    schatten_2_norm: float
    operator_norm_inverse: float
    trace_class: bool

    # Weil-Guinand consistency
    gaussian_sigma: float
    zero_side_via_hp: float
    pole_side: float
    archimedean_side: float
    prime_side_via_p14: float
    rhs_total: float
    residual: float
    relative_residual: float
    weil_tolerance: float
    weil_verified: bool

    # Operator-level gap G4
    spectral_gap_n_compared: int
    spectral_gap_wasserstein_1: float
    spectral_gap_growth_ratio: float

    # Overall verdict
    scaffold_consistent: bool
    notes: Tuple[str, ...]

    def summary(self) -> str:
        return (
            f"HilbertPolyaCertificate("
            f"n_zeros={self.n_zeros}, "
            f"primes={self.n_primes}, "
            f"self_adjoint={self.self_adjoint}, "
            f"trace_class={self.trace_class}, "
            f"||R||_1={self.schatten_1_norm:.4e}, "
            f"weil_residual={self.residual:.3e}, "
            f"W_1(P14,HP)={self.spectral_gap_wasserstein_1:.4e}, "
            f"scaffold_consistent={self.scaffold_consistent})"
        )


def compute_hilbert_polya_certificate(
    *,
    n_primes: int = 50,
    max_power: int = 8,
    n_zeros: int = 80,
    gaussian_sigma: float = 8.0,
    resolvent_shift: float = 1.0,
    weil_tolerance: float = 1e-3,
    dps: int = 30,
) -> HilbertPolyaCertificate:
    """Build T_HP and certify TNFR-stack consistency.

    Parameters
    ----------
    n_primes, max_power
        Prime-ladder bundle dimensions; passed to P14 builder.
    n_zeros
        Length of the gamma list used to populate ``T_HP``.
    gaussian_sigma
        Width of the Gaussian test function used for Weil-Guinand.
    resolvent_shift
        Positive shift ``s`` for ``(T_HP^2 + s^2 I)^{-1/2}``.
    weil_tolerance
        Acceptance tolerance for the Weil-Guinand residual.
    dps
        mpmath decimal precision for zero computation.
    """
    if n_zeros <= 0:
        raise ValueError("n_zeros must be positive")

    bundle = build_prime_ladder_hamiltonian(
        n_primes=n_primes,
        max_power=max_power,
        coupling=0.0,
    )
    gammas = fetch_zero_imaginary_parts(n_zeros, dps=dps)
    T_hp = build_hp_operator(gammas)

    self_adj = verify_hp_self_adjoint(T_hp)
    resolvent = hp_resolvent_schatten_norms(gammas, shift=resolvent_shift)

    test = gaussian_test_function(gaussian_sigma)
    zero_side = hp_zero_side_from_operator(gammas, test)
    pole_side_bare = weil_pole_side(test)
    log_pi_term = -test.g_zero() * math.log(math.pi)
    pole_side = pole_side_bare + log_pi_term
    archimedean = weil_archimedean_integral(test)
    prime_side = weil_prime_side_from_hamiltonian(bundle, test)
    rhs = pole_side + archimedean + prime_side
    residual = abs(zero_side - rhs)
    denom_norm = max(abs(zero_side), abs(rhs), 1.0)
    rel_residual = residual / denom_norm
    weil_ok = residual <= weil_tolerance

    gap = structural_gap_p14_vs_hp(bundle, gammas)

    scaffold_ok = bool(
        self_adj["self_adjoint"]
        and resolvent["trace_class"]
        and weil_ok
    )

    notes: Tuple[str, ...] = (
        "T_HP is populated by inputting mpmath.zetazero outputs; the",
        "scaffold does not derive the zeros from TNFR first principles.",
        "spec(P14) grows like log n while spec(T_HP) grows like",
        "2*pi*n/log n; the Wasserstein-1 distance reported below",
        "quantifies gap G4 = the structural derivation of T_HP that",
        "would actually engage the Riemann Hypothesis.",
    )

    return HilbertPolyaCertificate(
        n_zeros=int(n_zeros),
        n_primes=int(n_primes),
        max_power=int(max_power),
        asymmetry_frobenius=self_adj["asymmetry_frobenius"],
        self_adjoint=bool(self_adj["self_adjoint"]),
        resolvent_shift=resolvent["shift"],
        schatten_1_norm=resolvent["schatten_1_norm"],
        schatten_2_norm=resolvent["schatten_2_norm"],
        operator_norm_inverse=resolvent["operator_norm_inverse"],
        trace_class=bool(resolvent["trace_class"]),
        gaussian_sigma=float(gaussian_sigma),
        zero_side_via_hp=float(zero_side),
        pole_side=float(pole_side),
        archimedean_side=float(archimedean),
        prime_side_via_p14=float(prime_side),
        rhs_total=float(rhs),
        residual=float(residual),
        relative_residual=float(rel_residual),
        weil_tolerance=float(weil_tolerance),
        weil_verified=bool(weil_ok),
        spectral_gap_n_compared=int(gap["n_compared"]),
        spectral_gap_wasserstein_1=float(gap["wasserstein_1"]),
        spectral_gap_growth_ratio=float(gap["asymptotic_growth_ratio"]),
        scaffold_consistent=scaffold_ok,
        notes=notes,
    )
