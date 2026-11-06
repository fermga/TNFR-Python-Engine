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
* For ``s > 0`` the finite shifted resolvent
  ``R = (T_HP^2 + s^2 I)^{-1/2}`` has singular values
  ``1 / sqrt(gamma_n^2 + s^2)``. Its Schatten and operator norms are
  evaluated numerically on the supplied truncation. Finite ``trace_class``
  does not establish trace-class membership of an infinite operator.
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
compatible with such a slot. A derivation of ``T_HP`` independent of the
supplied zero ordinates remains absent. Such a finite construction would
still require separate infinite-dimensional and analytic arguments before
it could establish RH. Historical G4 labels refer to that research gap;
they do not certify that every other proof obligation has been resolved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

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
        raise RuntimeError("mpmath.zetazero returned a non-positive imaginary part")
    return gammas


def build_hp_operator(gammas: np.ndarray) -> np.ndarray:
    """Return the diagonal Hilbert-Polya operator ``T_HP = diag(gammas)``."""
    gammas = np.asarray(gammas, dtype=float)
    if gammas.ndim != 1:
        raise ValueError("gammas must be a 1-D array")
    return np.diag(gammas)


def verify_hp_self_adjoint(T: np.ndarray, *, tol: float = 1e-12) -> dict:
    """Check ``T=T.conj().T`` within ``tol``, retaining imaginary telemetry.

    A Hermitian matrix can have nonzero imaginary off-diagonal entries; the
    imaginary Frobenius norm is not an additional self-adjointness condition.
    """
    arr = np.asarray(T)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("T must be a square matrix")
    asym = arr - arr.conj().T
    asym_norm = float(np.linalg.norm(asym, ord="fro"))
    imag_norm = float(np.linalg.norm(arr.imag, ord="fro"))
    return {
        "asymmetry_frobenius": asym_norm,
        "imaginary_frobenius": imag_norm,
        "self_adjoint": asym_norm <= tol,
        "tolerance": tol,
    }


def hp_resolvent_schatten_norms(
    gammas: np.ndarray,
    *,
    shift: float = 1.0,
) -> dict:
    r"""Compute Schatten norms of the shifted resolvent of ``T_HP``.

    For ``R = (T_HP^2 + s^2 I)^{-1/2}``, singular values are
    ``r_n = 1 / sqrt(gamma_n^2 + s^2)``. Return ``sum r_n``,
    ``sqrt(sum r_n^2)`` and ``max r_n`` for this same operator.
    ``trace_class`` only describes the supplied finite truncation.

    Earlier implementations used the trace and Hilbert-Schmidt norms of
    ``R^2`` alongside the operator norm of ``R``. These fields now consistently
    describe ``R``; historical reports retain their original arithmetic.
    """
    gammas = _finite_real_samples(gammas)
    if not gammas.size:
        raise ValueError("gammas must contain at least one spectral value")
    if isinstance(shift, (bool, np.bool_)):
        raise ValueError("shift must be finite and strictly positive")
    try:
        shift = float(shift)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("shift must be finite and strictly positive") from exc
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("shift must be finite and strictly positive")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scale = np.maximum(np.abs(gammas), shift)
            singular_values = (1.0 / np.hypot(gammas / scale, shift / scale)) / scale
            s1 = float(np.sum(singular_values))
            op_norm = float(np.max(singular_values))
            s2 = op_norm * float(np.linalg.norm(singular_values / op_norm))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("resolvent norms exceed finite floating-point range") from exc
    if not all(math.isfinite(norm) for norm in (s1, s2, op_norm)):
        raise ValueError("resolvent norms exceed finite floating-point range")
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
    same gamma list, namely ``2 Tr h(T_HP)`` in finite spectral calculus.
    """
    gammas = np.asarray(gammas, dtype=float)
    h_values = np.array([test.h(float(g)) for g in gammas], dtype=float)
    return float(2.0 * np.sum(h_values))


def _finite_real_samples(values: np.ndarray) -> np.ndarray:
    """Admit finite real one-dimensional samples for spectral diagnostics."""
    if np.iscomplexobj(values):
        raise ValueError("samples must be finite real 1-D arrays")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("samples must be finite real 1-D arrays") from exc
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("samples must be finite real 1-D arrays")
    return array


def _empirical_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Evaluate an admitted empirical distance before range checking."""
    if left.size == right.size:
        return float(np.mean(np.abs(np.sort(left) - np.sort(right))))
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(left, right))


def _same_empirical_measure(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare atom weights exactly when a computed distance is zero."""
    left_values, left_counts = np.unique(left, return_counts=True)
    right_values, right_counts = np.unique(right, return_counts=True)
    return np.array_equal(left_values, right_values) and all(
        int(left_count) * right.size == int(right_count) * left.size
        for left_count, right_count in zip(left_counts, right_counts)
    )


def wasserstein_1_distance(a: np.ndarray, b: np.ndarray) -> float:
    r"""Compute the 1-Wasserstein distance between two 1-D empirical measures.

    Each finite sample has equal weight within its measure. Equal-size inputs
    use sorted pairwise distances; otherwise SciPy integrates the empirical
    CDF difference, retaining the discrete sample weights. Interpolating sample
    quantiles would change those measures.
    Two empty inputs return zero by the historical finite-report convention;
    a single empty input cannot define a probability measure and is rejected.
    Overflowing intermediate differences/reductions are retried after scaling;
    an unrepresentable result or a lost nonzero distance is rejected.
    """
    samples = [_finite_real_samples(values) for values in (a, b)]
    if samples[0].size == samples[1].size == 0:
        return 0.0
    if not samples[0].size or not samples[1].size:
        raise ValueError("both empirical measures must contain samples")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = _empirical_distance(*samples)
    except (FloatingPointError, OverflowError):
        result = math.inf
    if not math.isfinite(result):
        scale = max(float(np.max(np.abs(values))) for values in samples)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                normalized = _empirical_distance(
                    *(values / scale for values in samples)
                )
                result = scale * normalized
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("distance exceeds finite floating-point range") from exc
    if not math.isfinite(result):
        raise ValueError("distance exceeds finite floating-point range")
    if result == 0.0 and not _same_empirical_measure(*samples):
        raise ValueError("nonzero distance underflows to represented zero")
    return result


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

    The distance quantifies this supplied finite comparison only. It neither
    rules out smooth maps between infinite spectra nor forces a particular
    nonlinear rescaling or a unique route to a Hilbert-Polya construction.
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
        self_adj["self_adjoint"] and resolvent["trace_class"] and weil_ok
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
