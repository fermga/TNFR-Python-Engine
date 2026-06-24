r"""TNFR Dirichlet L-function construction (P32 program — first L-function extension).

Goal
----
Generalise the P12 von Mangoldt construction from the Riemann zeta
function to **Dirichlet L-functions**

.. math::

    L(s, \chi) = \sum_{n=1}^{\infty} \chi(n)\, n^{-s}
               = \prod_{p \text{ prime}}
                  \bigl(1 - \chi(p)\, p^{-s}\bigr)^{-1}
    \quad (\mathrm{Re}\,s > 1),

where :math:`\chi` is a Dirichlet character mod :math:`q` (a completely
multiplicative arithmetic function with period :math:`q` vanishing on
integers sharing a common factor with :math:`q`).

The associated logarithmic derivative is the **twisted von Mangoldt
series**

.. math::

    -\frac{L'(s,\chi)}{L(s,\chi)}
      = \sum_{n=1}^{\infty} \chi(n)\, \Lambda(n)\, n^{-s}
      = \sum_p \sum_{k\ge 1}
          \chi(p)^k\, \log(p)\, p^{-ks}
    \quad (\mathrm{Re}\,s > 1).

TNFR interpretation: the χ-twisted prime-ladder
-----------------------------------------------
The construction is **structurally identical** to P12 with one
modification: each prime's REMESH echo carries a **χ-twisted weight**
instead of the bare emission strength :math:`\log(p)`.

For each prime :math:`p` and REMESH echo index :math:`k \ge 1`,

.. math::

    \mu_{p,k} = k\log p
    \quad (\text{same as P12, real spectral position}),

    w_{p,k}^{(\chi)} = \chi(p)^k\, \log p
    \quad (\text{complex-valued weight}).

Primes dividing the modulus :math:`q` satisfy :math:`\chi(p) = 0` and
therefore **drop out of the spectrum entirely** — a clean structural
consequence of multiplicativity (the corresponding nodes are
decoupled by the gauge selection :math:`\chi`).

The TNFR twisted Dirichlet trace

.. math::

    Z_{TNFR}(s, \chi) := \sum_{(\mu, w) \in \mathrm{Spec}_{TNFR}(\chi)}
       w\, e^{-s\mu}
      = \sum_p \chi(p) \log(p) \sum_{k=1}^{K_p} \chi(p)^{k-1} p^{-ks}
      \xrightarrow[K_p\to\infty]{}
        \sum_p \log(p)\,
          \frac{\chi(p)\, p^{-s}}{1 - \chi(p)\, p^{-s}}
      = -\frac{L'(s,\chi)}{L(s,\chi)}.

Three TNFR-native features (inherited from P12)
-----------------------------------------------

1. **Each coprime prime is a node**; primes :math:`p \mid q` drop out
   structurally.
2. **REMESH echoes carry the χ-twisted recursion** across scales.
3. **Weights :math:`\chi(p)^k \log p`** are determined by the
   character (the gauge selection of the L-function) and the
   structural emission strength, no arbitrary normalisation.

What this module does NOT do
----------------------------

- It does not construct an explicit self-adjoint operator whose
  spectrum is :math:`\{k\log p\}_{p \nmid q}` (P32 is the spectral-data
  / Dirichlet-series layer, analogous to P12 for ζ; the operator-level
  analogue of P14 for general L-functions is future work).
- It does not analytically continue :math:`Z_{TNFR}(s,\chi)` into
  :math:`0 < \mathrm{Re}(s) < 1` (the P13 analogue for L-functions is
  future work).
- It does not locate non-trivial zeros via this construction.

Status: EXPERIMENTAL — TNFR-Riemann P32 extension to Dirichlet
L-functions.  Does NOT close gap G4 (the generalised Riemann
hypothesis is RH-equivalent in every L-function and inherits the
same arithmetic obstruction as G4 for ζ).
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
    # Character infrastructure
    "DirichletCharacter",
    "principal_character",
    "real_character_mod_3",
    "real_character_mod_4",
    "real_character_mod_5",
    # Twisted prime-ladder spectrum
    "TwistedPrimeLadderSpectrum",
    "build_twisted_prime_ladder_spectrum",
    "tnfr_log_l_derivative",
    # Classical reference
    "classical_log_l_derivative",
    "classical_log_l_derivative_matched",
    # Verification
    "DirichletLReproductionResult",
    "verify_dirichlet_l_reproduction",
]


# ============================================================================
# Dirichlet character infrastructure
# ============================================================================


@dataclass(frozen=True)
class DirichletCharacter:
    r"""Completely multiplicative arithmetic character :math:`\chi` mod :math:`q`.

    A Dirichlet character mod :math:`q` is a function
    :math:`\chi : \mathbb{Z} \to \mathbb{C}` satisfying

    1. :math:`\chi(n + q) = \chi(n)` (periodic with period :math:`q`),
    2. :math:`\chi(mn) = \chi(m)\chi(n)` (completely multiplicative),
    3. :math:`\chi(n) = 0` if :math:`\gcd(n, q) > 1`,
    4. :math:`\chi(n) \ne 0` if :math:`\gcd(n, q) = 1`.

    Internally a character is specified by its **table of values on
    a residue system** :math:`\{0, 1, \dots, q-1\}`.  The value at an
    arbitrary integer :math:`n` is recovered as
    :math:`\chi(n) = \mathrm{table}[n \bmod q]`.

    Parameters
    ----------
    modulus : int
        :math:`q \ge 1`.
    values : tuple of complex
        ``values[r] = chi(r)`` for ``r = 0, 1, ..., q-1``.
        Must have length ``modulus``.
    name : str, default ""
        Optional human-readable label.

    Notes
    -----
    This module does not enforce that the supplied values constitute a
    *valid* Dirichlet character (i.e. that they extend a homomorphism
    :math:`(\mathbb{Z}/q\mathbb{Z})^\times \to \mathbb{C}^\times`).
    Use the constructors :func:`principal_character`,
    :func:`real_character_mod_3`, :func:`real_character_mod_4`,
    :func:`real_character_mod_5` for canonical guaranteed-valid
    characters.
    """

    modulus: int
    values: tuple[complex, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if self.modulus < 1:
            raise ValueError("modulus must be >= 1")
        if len(self.values) != self.modulus:
            raise ValueError(
                f"values must have length {self.modulus} " f"(got {len(self.values)})"
            )

    def __call__(self, n: int) -> complex:
        """Evaluate :math:`\\chi(n)` for any integer ``n``."""
        return self.values[n % self.modulus]

    @property
    def is_principal(self) -> bool:
        """True iff :math:`\\chi` is the principal character mod :math:`q`."""
        for r in range(self.modulus):
            expected = 1.0 + 0j if math.gcd(r, self.modulus) == 1 else 0.0 + 0j
            if abs(self.values[r] - expected) > 1e-12:
                return False
        return True

    @property
    def is_real(self) -> bool:
        """True iff all values are real (within tolerance)."""
        return all(abs(v.imag) < 1e-12 for v in self.values)


def principal_character(modulus: int) -> DirichletCharacter:
    r"""Principal character :math:`\chi_0` mod :math:`q`.

    Defined by :math:`\chi_0(n) = 1` if :math:`\gcd(n, q) = 1` and
    :math:`\chi_0(n) = 0` otherwise.  The associated L-function is

    .. math::

        L(s, \chi_0) = \zeta(s) \prod_{p \mid q} (1 - p^{-s}),

    i.e. :math:`\zeta(s)` with the Euler factors at primes dividing
    :math:`q` removed.

    Parameters
    ----------
    modulus : int
        :math:`q \ge 1`.

    Returns
    -------
    DirichletCharacter
    """
    if modulus < 1:
        raise ValueError("modulus must be >= 1")
    vals = tuple(
        (1.0 + 0j) if math.gcd(r, modulus) == 1 else (0.0 + 0j) for r in range(modulus)
    )
    return DirichletCharacter(
        modulus=modulus,
        values=vals,
        name=f"chi_0_mod_{modulus}",
    )


def real_character_mod_3() -> DirichletCharacter:
    r"""Unique non-principal (real, primitive) character mod 3.

    Values: :math:`\chi(0)=0`, :math:`\chi(1)=1`, :math:`\chi(2)=-1`.
    This is the Legendre symbol :math:`(n / 3)` for :math:`\gcd(n,3)=1`.

    The L-function is

    .. math::

        L(s, \chi_3) = \sum_{n=1}^{\infty}
          \frac{(n/3)}{n^s}
          = 1 - 2^{-s} + 4^{-s} - 5^{-s} + 7^{-s} - 8^{-s} + \dots
    """
    return DirichletCharacter(
        modulus=3,
        values=(0.0 + 0j, 1.0 + 0j, -1.0 + 0j),
        name="chi_real_mod_3",
    )


def real_character_mod_4() -> DirichletCharacter:
    r"""Unique non-principal (real, primitive) character mod 4.

    Values: :math:`\chi(0)=0`, :math:`\chi(1)=1`, :math:`\chi(2)=0`,
    :math:`\chi(3)=-1`.  This is the Kronecker symbol giving the
    Dirichlet beta function

    .. math::

        L(s, \chi_4) = \beta(s)
          = 1 - 3^{-s} + 5^{-s} - 7^{-s} + \dots
    """
    return DirichletCharacter(
        modulus=4,
        values=(0.0 + 0j, 1.0 + 0j, 0.0 + 0j, -1.0 + 0j),
        name="chi_real_mod_4",
    )


def real_character_mod_5() -> DirichletCharacter:
    r"""Real non-principal character mod 5 (Legendre symbol :math:`(n/5)`).

    Values: :math:`\chi(0)=0, \chi(1)=1, \chi(2)=-1, \chi(3)=-1,
    \chi(4)=1`.
    """
    return DirichletCharacter(
        modulus=5,
        values=(0.0 + 0j, 1.0 + 0j, -1.0 + 0j, -1.0 + 0j, 1.0 + 0j),
        name="chi_real_mod_5",
    )


# ============================================================================
# Twisted prime-ladder spectrum
# ============================================================================


@dataclass(frozen=True)
class TwistedPrimeLadderSpectrum:
    r"""χ-twisted TNFR prime-ladder spectrum.

    Encodes the disjoint union of per-prime REMESH echo ladders with
    χ-twisted complex weights:

    .. math::

        \{(\mu_{p,k}, w_{p,k}^{(\chi)}) :
          p \in \mathcal{P}, \;
          k = 1, \dots, K\},
        \quad
        \mu_{p,k} = k\log p,
        \quad
        w_{p,k}^{(\chi)} = \chi(p)^k \log p.

    Primes dividing :math:`q` are excluded (their :math:`\chi(p) = 0`
    makes every echo vanish).

    Attributes
    ----------
    primes_active : np.ndarray
        Primes coprime to the modulus (those carrying non-zero weight).
    primes_excluded : np.ndarray
        Primes dividing the modulus, dropped from the spectrum.
    max_power : int
        Maximum echo index :math:`K`.
    eigenvalues : np.ndarray
        Real array of energies :math:`\mu_{p,k} = k\log p` over
        ``primes_active``, shape ``(n_active * max_power,)``.
    weights : np.ndarray
        Complex array of χ-twisted weights, same shape as
        ``eigenvalues``.
    character_modulus : int
        :math:`q`.
    character_name : str
        Label of the character used.
    """

    primes_active: np.ndarray
    primes_excluded: np.ndarray
    max_power: int
    eigenvalues: np.ndarray
    weights: np.ndarray
    character_modulus: int
    character_name: str

    @property
    def n_active(self) -> int:
        return int(self.primes_active.size)

    @property
    def n_excluded(self) -> int:
        return int(self.primes_excluded.size)

    @property
    def size(self) -> int:
        return int(self.eigenvalues.size)


def build_twisted_prime_ladder_spectrum(
    chi: DirichletCharacter,
    n_primes: int,
    *,
    max_power: int = 8,
    primes: Sequence[int] | None = None,
) -> TwistedPrimeLadderSpectrum:
    r"""Construct the χ-twisted TNFR prime-ladder spectrum.

    Parameters
    ----------
    chi : DirichletCharacter
        Character defining the twist.
    n_primes : int
        Number of primes to use (ignored if ``primes`` is supplied).
        Primes dividing the modulus are still counted in this total
        but excluded from the active spectrum.
    max_power : int, default 8
        Maximum REMESH echo index :math:`K`.
    primes : sequence of int, optional
        Explicit prime list.

    Returns
    -------
    TwistedPrimeLadderSpectrum
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

    active: list[int] = []
    excluded: list[int] = []
    chi_active: list[complex] = []
    for p in prime_list:
        cp = chi(p)
        if abs(cp) < 1e-15:
            excluded.append(p)
        else:
            active.append(p)
            chi_active.append(cp)

    if not active:
        raise ValueError(
            "All supplied primes divide the character modulus; "
            "spectrum would be empty."
        )

    p_arr = np.asarray(active, dtype=float)
    log_p = np.log(p_arr)  # (n_active,)
    chi_arr = np.asarray(chi_active, dtype=complex)  # (n_active,)
    k_arr = np.arange(1, max_power + 1, dtype=float)  # (max_power,)

    # μ_{p,k} = k log p
    mu = np.outer(log_p, k_arr)  # (n_active, K)

    # χ(p)^k for k=1..K → broadcast to (n_active, K)
    # use complex powers for full generality
    chi_pow = chi_arr[:, None] ** k_arr[None, :]  # (n_active, K)

    # w_{p,k} = χ(p)^k log p
    w = chi_pow * log_p[:, None]  # (n_active, K)

    return TwistedPrimeLadderSpectrum(
        primes_active=np.asarray(active, dtype=int),
        primes_excluded=np.asarray(excluded, dtype=int),
        max_power=int(max_power),
        eigenvalues=mu.ravel(),
        weights=w.ravel(),
        character_modulus=chi.modulus,
        character_name=chi.name,
    )


def tnfr_log_l_derivative(
    spectrum: TwistedPrimeLadderSpectrum,
    s: complex,
) -> complex:
    r"""Weighted χ-twisted Dirichlet trace :math:`Z_{TNFR}(s,\chi)`.

    Evaluates

    .. math::

        Z_{TNFR}(s,\chi) = \sum_{(\mu,w) \in \mathrm{Spec}_{TNFR}(\chi)}
          w\, e^{-s\mu}.

    For :math:`\mathrm{Re}\,s > 1` and :math:`K, n_{\mathrm{primes}}
    \to \infty` this converges to :math:`-L'(s,\chi)/L(s,\chi)`.

    Parameters
    ----------
    spectrum : TwistedPrimeLadderSpectrum
    s : complex

    Returns
    -------
    complex
    """
    s_c = complex(s)
    z = np.sum(spectrum.weights * np.exp(-s_c * spectrum.eigenvalues))
    return complex(z)


# ============================================================================
# Classical reference: Σ χ(n) Λ(n) n^{-s}
# ============================================================================


def classical_log_l_derivative(
    chi: DirichletCharacter,
    s: complex,
    n_max: int,
) -> complex:
    r"""Truncated classical sum :math:`\sum_{n\le N} \chi(n) \Lambda(n) n^{-s}`.

    Iterates over prime powers :math:`p^k \le N`, contributing
    :math:`\chi(p)^k \log(p)\, p^{-ks}` for each.

    Parameters
    ----------
    chi : DirichletCharacter
    s : complex
    n_max : int

    Returns
    -------
    complex
    """
    if n_max < 2:
        return 0.0 + 0j

    sieve = bytearray(b"\x01") * (n_max + 1)
    sieve[0] = sieve[1] = 0
    p = 2
    while p * p <= n_max:
        if sieve[p]:
            start = p * p
            sieve[start : n_max + 1 : p] = b"\x00" * (((n_max - start) // p) + 1)
        p += 1

    total: complex = 0.0 + 0j
    s_c = complex(s)
    for p in range(2, n_max + 1):
        if not sieve[p]:
            continue
        cp = chi(p)
        if abs(cp) < 1e-15:
            continue
        log_p = math.log(p)
        pk = p
        cp_k = cp
        while pk <= n_max:
            total += cp_k * log_p * (pk ** (-s_c))
            pk *= p
            cp_k *= cp
    return total


def classical_log_l_derivative_matched(
    chi: DirichletCharacter,
    s: complex,
    primes: Sequence[int],
    max_power: int,
) -> complex:
    r"""Classical sum restricted to the same (prime, power) set as the
    TNFR ladder.

    Parameters
    ----------
    chi : DirichletCharacter
    s : complex
    primes : sequence of int
    max_power : int

    Returns
    -------
    complex
    """
    total: complex = 0.0 + 0j
    s_c = complex(s)
    for p in primes:
        cp = chi(p)
        if abs(cp) < 1e-15:
            continue
        log_p = math.log(p)
        cp_k = cp
        for k in range(1, max_power + 1):
            total += cp_k * log_p * (p ** (-k * s_c))
            cp_k *= cp
    return total


# ============================================================================
# Verification
# ============================================================================


@dataclass(frozen=True)
class DirichletLReproductionResult:
    r"""Numerical comparison of :math:`Z_{TNFR}(s,\chi)` vs the classical
    twisted series.

    Attributes
    ----------
    character_name : str
    character_modulus : int
    s_values : np.ndarray
        Complex spectral parameters tested.
    n_active : int
        Primes actually carrying non-zero χ-weight.
    n_excluded : int
        Primes dropped (those dividing the modulus).
    max_power : int
    n_max_classical : int
    z_tnfr : np.ndarray
        :math:`Z_{TNFR}(s,\chi)` values (complex).
    z_classical : np.ndarray
        Truncated classical sum (complex).
    abs_error : np.ndarray
        :math:`|Z_{TNFR} - Z_{\mathrm{classical}}|`.
    rel_error : np.ndarray
    max_rel_error : float
    """

    character_name: str
    character_modulus: int
    s_values: np.ndarray
    n_active: int
    n_excluded: int
    max_power: int
    n_max_classical: int
    z_tnfr: np.ndarray
    z_classical: np.ndarray
    abs_error: np.ndarray
    rel_error: np.ndarray
    max_rel_error: float

    def summary(self) -> str:
        return (
            f"Dirichlet L reproduction:  "
            f"chi={self.character_name} (mod {self.character_modulus}), "
            f"n_active={self.n_active}, n_excluded={self.n_excluded}, "
            f"max_power={self.max_power}, "
            f"n_max_classical={self.n_max_classical}, "
            f"max_rel_error={self.max_rel_error:.3e}"
        )


def verify_dirichlet_l_reproduction(
    chi: DirichletCharacter,
    s_values: Sequence[complex],
    *,
    n_primes: int = 200,
    max_power: int = 12,
    n_max_classical: int = 100_000,
) -> DirichletLReproductionResult:
    r"""Numerically verify :math:`Z_{TNFR}(s,\chi) \approx
    \sum_{n\le N}\chi(n)\Lambda(n) n^{-s}`.

    The same per-prime-power correspondence as P12 holds: each TNFR
    spectral entry :math:`(\mu_{p,k}, \chi(p)^k\log p)` matches exactly
    one classical contribution :math:`\chi(p)^k\Lambda(p^k) (p^k)^{-s}`
    (since :math:`\Lambda(p^k) = \log p`).  When the two truncations
    cover the same prime-power set, the sums agree to floating-point
    precision.

    Parameters
    ----------
    chi : DirichletCharacter
    s_values : sequence of complex
        Spectral parameters; require :math:`\mathrm{Re}\,s > 1` for
        meaningful comparison to the analytic limit.
    n_primes : int, default 200
    max_power : int, default 12
    n_max_classical : int, default 100_000

    Returns
    -------
    DirichletLReproductionResult
    """
    spectrum = build_twisted_prime_ladder_spectrum(chi, n_primes, max_power=max_power)

    s_arr = np.asarray(list(s_values), dtype=complex)
    z_tnfr = np.array(
        [tnfr_log_l_derivative(spectrum, s) for s in s_arr],
        dtype=complex,
    )
    z_classical = np.array(
        [classical_log_l_derivative(chi, s, n_max_classical) for s in s_arr],
        dtype=complex,
    )

    abs_err = np.abs(z_tnfr - z_classical)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(
            np.abs(z_classical) > 1e-15,
            abs_err / np.abs(z_classical),
            abs_err,
        )

    return DirichletLReproductionResult(
        character_name=chi.name,
        character_modulus=chi.modulus,
        s_values=s_arr,
        n_active=spectrum.n_active,
        n_excluded=spectrum.n_excluded,
        max_power=spectrum.max_power,
        n_max_classical=int(n_max_classical),
        z_tnfr=z_tnfr,
        z_classical=z_classical,
        abs_error=abs_err,
        rel_error=rel_err,
        max_rel_error=float(np.max(rel_err)),
    )
