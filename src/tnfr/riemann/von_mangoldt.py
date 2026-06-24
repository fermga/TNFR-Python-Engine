r"""TNFR von Mangoldt construction (P12 program — first stone).

Goal
----
Build a TNFR-native spectral object whose **weighted Dirichlet trace**
reproduces, exactly, the classical von Mangoldt series

.. math::

    -\frac{\zeta'(s)}{\zeta(s)}
      = \sum_{n=2}^{\infty} \Lambda(n)\, n^{-s}
      = \sum_{p\,\text{prime}}\sum_{k\ge 1} \log(p)\, p^{-ks},

where :math:`\Lambda(n) = \log(p)` if :math:`n = p^k` for some prime
:math:`p` and integer :math:`k \ge 1`, and :math:`\Lambda(n) = 0`
otherwise.

This is the constructive route prioritised by the May 2026 gap analysis
(see :file:`theory/TNFR_RIEMANN_RESEARCH_NOTES.md` § 7 and § 8): the
simple affine fit :math:`\zeta_H(1/2,u) \approx C\zeta_R(u+\delta)`
does not converge, so we restart from the arithmetic identity that
the bridge must respect — multiplicativity of :math:`\zeta(s)` via the
Euler product, encoded by :math:`\Lambda(n)`.

TNFR interpretation: the prime-ladder construction
--------------------------------------------------
Each prime :math:`p` is a TNFR node with a **fundamental structural
pulse** of energy :math:`\log(p)`.  The canonical REMESH operator
(recursivity, U1a / U1b in the unified grammar) generates **echoes**
at integer multiples :math:`k\cdot\log(p)`, each carrying the same
weight :math:`\log(p)` (the prime's structural emission strength).

The resulting **prime-ladder spectrum** is the disjoint union

.. math::

    \mathrm{Spec}_{TNFR} = \bigsqcup_p
      \{(\mu_{p,k}, w_{p,k}) : k = 1, 2, \dots, K_p\},
    \quad
    \mu_{p,k} = k\log(p), \quad w_{p,k} = \log(p).

Define the TNFR log-zeta-derivative as the weighted Dirichlet series

.. math::

    Z_{TNFR}(s) := \sum_{(\mu, w) \in \mathrm{Spec}_{TNFR}}
      w \cdot e^{-s\mu}.

Then by direct computation

.. math::

    Z_{TNFR}(s)
      = \sum_p \log(p) \sum_{k=1}^{K_p} p^{-ks}
      \xrightarrow[K_p\to\infty]{} \sum_p \log(p)\,
          \frac{p^{-s}}{1 - p^{-s}}
      = -\frac{\zeta'(s)}{\zeta(s)}
        \quad \text{for } \mathrm{Re}(s) > 1.

This construction has three TNFR-native features:

1. **Each prime is a node** (canonical TNFR primitive).
2. **REMESH echoes** carry the recursion across scales (operator #13).
3. **Weights = log(p)** match the structural emission strength,
   not arbitrary normalisation.

What this module does NOT do (yet)
-----------------------------------
- It does not construct an explicit self-adjoint operator on a single
  Hilbert space whose spectrum is :math:`\{k\log(p)\}` with the right
  multiplicities; it specifies the spectral data directly.  Building
  the underlying operator is § 8.2 of the research notes.
- It does not analytically continue :math:`Z_{TNFR}(s)` into
  :math:`0 < \mathrm{Re}(s) < 1`; that is § 8.3.
- It does not yet locate non-trivial zeros via this construction; that
  is § 8.4 and depends on the analytic continuation step.

Status: EXPERIMENTAL — Research prototype for TNFR-Riemann P12 program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..mathematics.unified_numerical import np
from .operator import _first_primes

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Classical helpers
    "mangoldt_lambda",
    "classical_log_zeta_derivative",
    "classical_log_zeta_derivative_matched",
    # Prime-ladder spectrum
    "PrimeLadderSpectrum",
    "build_prime_ladder_spectrum",
    "tnfr_log_zeta_derivative",
    # Verification
    "VonMangoldtReproductionResult",
    "verify_von_mangoldt_reproduction",
]


# ============================================================================
# Classical reference: Λ(n) and Σ Λ(n) n^{-s}
# ============================================================================


def mangoldt_lambda(n: int) -> float:
    r"""Classical von Mangoldt function :math:`\Lambda(n)`.

    Returns :math:`\log p` if :math:`n = p^k` for some prime
    :math:`p` and integer :math:`k \ge 1`; returns 0 otherwise.

    Parameters
    ----------
    n : int
        Integer :math:`\ge 1`.

    Returns
    -------
    float
        :math:`\Lambda(n)`.
    """
    if n < 2:
        return 0.0
    # Factor out the smallest prime factor and check it captures n entirely.
    p = 2
    while p * p <= n:
        if n % p == 0:
            # Check that n is a pure power of p.
            m = n
            while m % p == 0:
                m //= p
            return math.log(p) if m == 1 else 0.0
        p += 1 if p == 2 else 2
    # n itself is prime
    return math.log(n)


def classical_log_zeta_derivative(s: float, n_max: int) -> float:
    r"""Truncated classical sum :math:`\sum_{n=2}^{n_{\max}} \Lambda(n)\, n^{-s}`.

    Implementation iterates over prime powers :math:`p^k \le n_{\max}`
    directly (since :math:`\Lambda(n) = 0` for all non-prime-power
    :math:`n`), so cost is :math:`O(\pi(n_{\max}) \log n_{\max})`
    rather than :math:`O(n_{\max})`.

    For :math:`\mathrm{Re}(s) > 1` this converges to
    :math:`-\zeta'(s)/\zeta(s)` as :math:`n_{\max} \to \infty`.

    Parameters
    ----------
    s : float
        Real part of the spectral parameter.  Must satisfy :math:`s > 1`
        for convergence; for :math:`s \le 1` the partial sum still
        evaluates but diverges in the limit.
    n_max : int
        Upper bound of the truncation.

    Returns
    -------
    float
        Partial sum :math:`\sum_{n=2}^{n_{\max}} \Lambda(n) n^{-s}`.
    """
    if n_max < 2:
        return 0.0
    # Enumerate primes p <= n_max with a sieve, then sum log(p) * p^{-ks}
    # for every prime power p^k <= n_max.
    sieve = bytearray(b"\x01") * (n_max + 1)
    sieve[0] = sieve[1] = 0
    p = 2
    while p * p <= n_max:
        if sieve[p]:
            start = p * p
            sieve[start : n_max + 1 : p] = b"\x00" * (((n_max - start) // p) + 1)
        p += 1

    total = 0.0
    for p in range(2, n_max + 1):
        if not sieve[p]:
            continue
        log_p = math.log(p)
        # Iterate prime powers p, p^2, p^3, ... <= n_max
        pk = p
        while pk <= n_max:
            total += log_p * (pk ** (-s))
            # Guard against overflow to int(>1e18 still fine for Python ints)
            pk *= p
    return total


def classical_log_zeta_derivative_matched(
    s: float,
    primes: Sequence[int],
    max_power: int,
) -> float:
    r"""Classical sum restricted to the same prime/power set as the TNFR ladder.

    Computes :math:`\sum_{p \in \mathcal{P}} \sum_{k=1}^{K} \log(p)\, p^{-ks}`
    over the **exact** spectrum that
    :func:`build_prime_ladder_spectrum` would produce.  By construction
    this equals :func:`tnfr_log_zeta_derivative` to machine precision —
    useful as a unit-test invariant.

    Parameters
    ----------
    s : float
        Spectral parameter.
    primes : sequence of int
        Same prime list used to build the TNFR spectrum.
    max_power : int
        Same REMESH echo cap :math:`K`.

    Returns
    -------
    float
    """
    total = 0.0
    for p in primes:
        log_p = math.log(p)
        for k in range(1, max_power + 1):
            total += log_p * (p ** (-k * s))
    return total


# ============================================================================
# Prime-ladder spectrum
# ============================================================================


@dataclass(frozen=True)
class PrimeLadderSpectrum:
    r"""TNFR prime-ladder spectrum :math:`\{(k\log p, \log p)\}`.

    Encodes the disjoint union of per-prime REMESH echo ladders.
    Each entry ``(eigenvalue, weight)`` represents one structural
    echo at energy :math:`\mu_{p,k} = k\log p` with weight
    :math:`w_{p,k} = \log p`.

    Attributes
    ----------
    primes : np.ndarray
        Array of primes used (length ``n_primes``).
    max_power : int
        Maximum echo index :math:`K` (same for every prime in this
        prototype; ``max_power = 1`` reproduces only the bare-prime
        contribution :math:`\sum_p \log(p) p^{-s}`).
    eigenvalues : np.ndarray
        Flattened array of energies :math:`\mu_{p,k}`, shape
        ``(n_primes * max_power,)``.
    weights : np.ndarray
        Same shape as ``eigenvalues``; each entry is :math:`\log(p)`.
    """

    primes: np.ndarray
    max_power: int
    eigenvalues: np.ndarray
    weights: np.ndarray

    @property
    def n_primes(self) -> int:
        return int(self.primes.size)

    @property
    def size(self) -> int:
        """Number of (eigenvalue, weight) pairs in the spectrum."""
        return int(self.eigenvalues.size)


def build_prime_ladder_spectrum(
    n_primes: int,
    *,
    max_power: int = 8,
    primes: Sequence[int] | None = None,
) -> PrimeLadderSpectrum:
    r"""Construct the TNFR prime-ladder spectrum.

    Builds the spectral data

    .. math::

        \{(\mu_{p,k}, w_{p,k}) :
          p \in \mathcal{P}_{n_{\mathrm{primes}}}, \;
          k = 1, \dots, K\},
        \quad
        \mu_{p,k} = k\log p, \quad w_{p,k} = \log p.

    Parameters
    ----------
    n_primes : int
        Number of primes :math:`|\mathcal{P}|` (ignored if ``primes``
        is supplied).
    max_power : int, default 8
        Maximum REMESH echo index :math:`K`.  Larger ``max_power``
        captures higher-order prime-power contributions
        :math:`p^k` for ``k`` up to ``max_power``.  Truncation error
        for :math:`s > 1` is bounded by
        :math:`\sum_p \log(p) p^{-s(K+1)}/(1 - p^{-s})`.
    primes : sequence of int, optional
        Explicit prime list.  If given, ``n_primes`` is ignored.

    Returns
    -------
    PrimeLadderSpectrum
    """
    if max_power < 1:
        raise ValueError("max_power must be >= 1")

    if primes is None:
        if n_primes < 1:
            raise ValueError("n_primes must be >= 1")
        prime_list = _first_primes(n_primes)
    else:
        prime_list = list(primes)
        if not prime_list:
            raise ValueError("primes must be non-empty")

    p_arr = np.asarray(prime_list, dtype=float)
    log_p = np.log(p_arr)  # shape (n_primes,)

    k_arr = np.arange(1, max_power + 1, dtype=float)  # shape (max_power,)

    # eigenvalues[i, k-1] = k * log(p_i); weights[i, k-1] = log(p_i)
    mu = np.outer(log_p, k_arr)  # (n_primes, max_power)
    w = np.broadcast_to(log_p[:, None], mu.shape).copy()

    return PrimeLadderSpectrum(
        primes=np.asarray(prime_list, dtype=int),
        max_power=int(max_power),
        eigenvalues=mu.ravel(),
        weights=w.ravel(),
    )


def tnfr_log_zeta_derivative(
    spectrum: PrimeLadderSpectrum,
    s: float | complex,
) -> complex:
    r"""Weighted Dirichlet trace :math:`Z_{TNFR}(s) = \sum w\, e^{-s\mu}`.

    Evaluates the TNFR analogue of :math:`-\zeta'(s)/\zeta(s)` from the
    prime-ladder spectrum.

    Parameters
    ----------
    spectrum : PrimeLadderSpectrum
        Spectral data from :func:`build_prime_ladder_spectrum`.
    s : float or complex
        Spectral parameter.  Convergence in the prime-count limit
        requires :math:`\mathrm{Re}(s) > 1`.

    Returns
    -------
    complex
        :math:`Z_{TNFR}(s)`.  Returns a real float if ``s`` is real.
    """
    s_c = complex(s)
    # exp(-s * mu) = p^{-k s} since mu = k log p
    z = np.sum(spectrum.weights * np.exp(-s_c * spectrum.eigenvalues))
    return complex(z) if isinstance(s, complex) else float(z.real)


# ============================================================================
# Verification: prime-ladder Z_TNFR vs classical Σ Λ(n) n^{-s}
# ============================================================================


@dataclass(frozen=True)
class VonMangoldtReproductionResult:
    r"""Numerical comparison of :math:`Z_{TNFR}(s)` vs the classical series.

    Attributes
    ----------
    s_values : np.ndarray
        Real spectral parameters tested.
    n_primes : int
        Number of primes used in the TNFR spectrum.
    max_power : int
        REMESH echo cap used.
    n_max_classical : int
        Truncation bound used in the classical Dirichlet sum.
    z_tnfr : np.ndarray
        :math:`Z_{TNFR}(s)` for each ``s``.
    z_classical : np.ndarray
        Truncated classical sum :math:`\sum_{n\le N}\Lambda(n) n^{-s}`.
    abs_error : np.ndarray
        :math:`|Z_{TNFR}(s) - Z_{\mathrm{classical}}(s)|`.
    rel_error : np.ndarray
        ``abs_error / |z_classical|``.
    max_rel_error : float
        Worst-case relative error over ``s_values``.
    """

    s_values: np.ndarray
    n_primes: int
    max_power: int
    n_max_classical: int
    z_tnfr: np.ndarray
    z_classical: np.ndarray
    abs_error: np.ndarray
    rel_error: np.ndarray
    max_rel_error: float

    def summary(self) -> str:
        return (
            f"VonMangoldt reproduction:  "
            f"n_primes={self.n_primes}, max_power={self.max_power}, "
            f"n_max_classical={self.n_max_classical}, "
            f"max_rel_error={self.max_rel_error:.3e}"
        )


def verify_von_mangoldt_reproduction(
    s_values: Sequence[float],
    *,
    n_primes: int = 200,
    max_power: int = 12,
    n_max_classical: int = 100_000,
) -> VonMangoldtReproductionResult:
    r"""Numerically verify :math:`Z_{TNFR}(s) \approx \sum_{n\le N}\Lambda(n) n^{-s}`.

    Both sides are truncations of :math:`-\zeta'(s)/\zeta(s)`:

    - TNFR side: prime-ladder spectrum with ``n_primes`` primes and
      ``max_power`` echoes.
    - Classical side: direct sum over prime powers
      :math:`p^k \le n_{\max,\mathrm{classical}}` evaluated through
      :func:`classical_log_zeta_derivative`.

    Each Λ-contribution :math:`\log(p)\, p^{-ks}` corresponds **by
    construction** to exactly one entry :math:`(\mu_{p,k}, w_{p,k})`
    of the TNFR spectrum.  Therefore, when the two truncations cover
    the same set of prime powers, the sums agree exactly modulo
    floating-point rounding.  When they cover different sets, the
    discrepancy is the difference of their truncation tails relative
    to the analytic limit :math:`-\zeta'(s)/\zeta(s)`.

    Parameters
    ----------
    s_values : sequence of float
        Real spectral parameters at which to evaluate both sides.
    n_primes : int, default 200
    max_power : int, default 12
    n_max_classical : int, default 100_000
        Upper bound for the classical sieve.  Increasing this improves
        the classical reference and tightens the comparison.

    Returns
    -------
    VonMangoldtReproductionResult
    """
    spectrum = build_prime_ladder_spectrum(n_primes, max_power=max_power)

    s_arr = np.asarray(list(s_values), dtype=float)
    z_tnfr = np.array(
        [tnfr_log_zeta_derivative(spectrum, float(s)) for s in s_arr],
        dtype=float,
    )
    z_classical = np.array(
        [classical_log_zeta_derivative(float(s), n_max_classical) for s in s_arr],
        dtype=float,
    )

    abs_err = np.abs(z_tnfr - z_classical)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(
            np.abs(z_classical) > 1e-15,
            abs_err / np.abs(z_classical),
            abs_err,
        )

    return VonMangoldtReproductionResult(
        s_values=s_arr,
        n_primes=spectrum.n_primes,
        max_power=spectrum.max_power,
        n_max_classical=int(n_max_classical),
        z_tnfr=z_tnfr,
        z_classical=z_classical,
        abs_error=abs_err,
        rel_error=rel_err,
        max_rel_error=float(np.max(rel_err)),
    )
