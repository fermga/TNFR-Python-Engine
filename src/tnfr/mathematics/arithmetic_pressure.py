r"""Arithmetic-pressure independence and completeness audit (R7).

The canonical arithmetic realisation of the nodal gradient is the three-channel
structural pressure (``ArithmeticTNFRFormalism.delta_nfr_value`` with unit
coefficients):

    ΔNFR(n) = (Ω(n) − 1) + (τ(n) − 2) + (σ(n)/n − (1 + 1/n)),

each channel non-negative and zero exactly at the primes.  This module audits the
three claims that are easy to conflate, computing every invariant **exactly** over
ℚ:

1. **Primality sufficiency** (PROVED, classical): each channel is individually
   ``0`` iff ``n`` is prime, and so is their sum.  A direct consequence is that
   the three-channel set is **redundant** for primality — any *single* channel
   already characterises the primes, so the realisation is **not minimal** for
   prime detection.
2. **Linear independence** (MEASURED): as real functions on the audited range the
   three channels are linearly independent (rank 3, no affine/constant relation),
   even though they are strongly *correlated* (all grow with compositeness).
   Correlation is not dependence; each channel carries distinct structural
   information (factor multiplicity, divisor count, abundance).
3. **Structural completeness** (OPEN): there is **no proof** that no fourth
   independent pressure degree exists.  Completeness is a hypothesis with
   restricted scope; a fourth channel is admitted only through the explicit gate
   :func:`admits_fourth_channel`.

Honest scope: "minimal and complete" is therefore downgraded to *"a
linearly-independent but primality-redundant three-channel set whose common zero
set is exactly the primes; completeness unproven"* (claim ``NT-P07`` OPEN).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np

from .krylov import exact_rank

__all__ = [
    "big_omega",
    "num_divisors",
    "divisor_sum",
    "channel_factorization",
    "channel_divisor",
    "channel_abundance",
    "channels",
    "arithmetic_pressure",
    "ArithmeticPressureVector",
    "pressure_vector",
    "CHANNEL_NAMES",
    "primes_in_range",
    "channel_zero_set",
    "channel_is_sufficient",
    "all_channels_sufficient",
    "channels_nonnegative",
    "pressure_zero_iff_prime",
    "channel_matrix",
    "channel_rank",
    "has_linear_relation",
    "IndependenceProof",
    "prove_functional_independence",
    "channel_correlations",
    "ablation_detects_primes",
    "minimal_channels_for_primality",
    "is_redundant_for_primality",
    "factor_class",
    "abundance_class",
    "pressure_by_class",
    "FourthChannelCriteria",
    "admits_fourth_channel",
    "completeness_proven",
    "algorithmic_primality_is_circular",
]

CHANNEL_NAMES = ("factorization", "divisor", "abundance")


# --------------------------------------------------------------------------- #
# Exact arithmetic functions (pure-Python trial division)
# --------------------------------------------------------------------------- #
def _factorization(n: int) -> dict[int, int]:
    if n < 2:
        raise ValueError("arithmetic pressure is defined for n >= 2")
    factors: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def big_omega(n: int) -> int:
    r"""``Ω(n)`` — the number of prime factors of ``n`` counted with multiplicity."""
    return sum(_factorization(n).values())


def num_divisors(n: int) -> int:
    r"""``τ(n)`` — the number of divisors of ``n``."""
    t = 1
    for e in _factorization(n).values():
        t *= e + 1
    return t


def divisor_sum(n: int) -> int:
    r"""``σ(n)`` — the sum of the divisors of ``n``."""
    s = 1
    for p, e in _factorization(n).items():
        s *= (p ** (e + 1) - 1) // (p - 1)
    return s


# --------------------------------------------------------------------------- #
# The three pressure channels (exact)
# --------------------------------------------------------------------------- #
def channel_factorization(n: int) -> Fraction:
    r"""Factorization pressure ``Ω(n) − 1`` (zero iff ``n`` prime)."""
    return Fraction(big_omega(n) - 1)


def channel_divisor(n: int) -> Fraction:
    r"""Divisor pressure ``τ(n) − 2`` (zero iff ``n`` prime)."""
    return Fraction(num_divisors(n) - 2)


def channel_abundance(n: int) -> Fraction:
    r"""Abundance pressure ``σ(n)/n − (1 + 1/n) = (σ(n) − n − 1)/n``."""
    return Fraction(divisor_sum(n) - n - 1, n)


def channels(n: int) -> tuple[Fraction, Fraction, Fraction]:
    r"""All three channels ``(Ω−1, τ−2, (σ−n−1)/n)`` (exact)."""
    return (channel_factorization(n), channel_divisor(n),
            channel_abundance(n))


def arithmetic_pressure(n: int) -> Fraction:
    r"""The total unit-coefficient pressure ``ΔNFR(n)`` (exact; ``0`` iff prime)."""
    return sum(channels(n), Fraction(0))


@dataclass(frozen=True)
class ArithmeticPressureVector:
    """The three-channel pressure of ``n`` as a structured vector.

    The scalar ``arithmetic_pressure(n)`` (mirrored by
    ``ArithmeticTNFRFormalism.delta_nfr_value``) is one **chosen** aggregation of
    these channels (their sum), not a unique or minimal basis. This vector keeps
    the channels separate so the reading is not mistaken for a minimal/complete
    basis for primality.
    """

    omega_defect: Fraction  # Ω(n) − 1  (factorization channel)
    divisor_count_defect: Fraction  # τ(n) − 2  (divisor channel)
    divisor_mass_defect: Fraction  # σ(n)/n − (1 + 1/n)  (abundance channel)

    @property
    def scalar(self) -> Fraction:
        r"""The chosen scalar aggregation (the sum) = ``arithmetic_pressure(n)``."""
        return (
            self.omega_defect
            + self.divisor_count_defect
            + self.divisor_mass_defect
        )

    def as_tuple(self) -> tuple[Fraction, Fraction, Fraction]:
        return (self.omega_defect, self.divisor_count_defect,
                self.divisor_mass_defect)


def pressure_vector(n: int) -> ArithmeticPressureVector:
    r"""The three-channel pressure of ``n`` as a structured vector (exact)."""
    c1, c2, c3 = channels(n)
    return ArithmeticPressureVector(c1, c2, c3)


_CHANNEL_FUNCS = (channel_factorization, channel_divisor, channel_abundance)


# --------------------------------------------------------------------------- #
# 1. Primality sufficiency (each channel and the sum are 0 iff prime)
# --------------------------------------------------------------------------- #
def primes_in_range(lo: int, hi: int) -> set[int]:
    r"""Primes in ``[lo, hi]`` (via ``Ω(n) = 1``)."""
    return {n for n in range(max(lo, 2), hi + 1) if big_omega(n) == 1}


def channel_zero_set(index: int, lo: int, hi: int) -> set[int]:
    r"""``{n ∈ [lo, hi] : channel_index(n) = 0}``."""
    fn = _CHANNEL_FUNCS[index]
    return {n for n in range(max(lo, 2), hi + 1) if fn(n) == 0}


def channel_is_sufficient(index: int, lo: int, hi: int) -> bool:
    r"""Whether channel ``index`` alone characterises the primes on ``[lo, hi]``."""
    return channel_zero_set(index, lo, hi) == primes_in_range(lo, hi)


def all_channels_sufficient(lo: int, hi: int) -> bool:
    r"""Whether every channel individually characterises the primes."""
    return all(channel_is_sufficient(i, lo, hi) for i in range(3))


def channels_nonnegative(lo: int, hi: int) -> bool:
    r"""Whether all three channels are ``>= 0`` on ``[lo, hi]``."""
    return all(
        all(fn(n) >= 0 for fn in _CHANNEL_FUNCS)
        for n in range(max(lo, 2), hi + 1)
    )


def pressure_zero_iff_prime(lo: int, hi: int) -> bool:
    r"""Whether ``ΔNFR(n) = 0`` exactly on the primes of ``[lo, hi]``."""
    zero = {n for n in range(max(lo, 2), hi + 1)
            if arithmetic_pressure(n) == 0}
    return zero == primes_in_range(lo, hi)


# --------------------------------------------------------------------------- #
# 2. Linear independence (rank / correlation)
# --------------------------------------------------------------------------- #
def channel_matrix(lo: int, hi: int) -> np.ndarray:
    r"""The ``(hi−lo+1) × 3`` matrix of channel values (float)."""
    rows = [
        [float(fn(n)) for fn in _CHANNEL_FUNCS]
        for n in range(max(lo, 2), hi + 1)
    ]
    return np.array(rows, dtype=float)


def channel_rank(lo: int, hi: int) -> int:
    r"""Rank of the channel matrix (3 ⇒ the channels are linearly independent)."""
    return int(np.linalg.matrix_rank(channel_matrix(lo, hi)))


def has_linear_relation(lo: int, hi: int) -> bool:
    r"""Whether any affine/constant linear relation ties the three channels.

    ``False`` iff ``rank[c1 c2 c3] = 3`` and ``rank[c1 c2 c3 1] = 4`` (no linear
    or affine dependence over the audited range).
    """
    M = channel_matrix(lo, hi)
    aug = np.column_stack([M, np.ones(len(M))])
    return not (np.linalg.matrix_rank(M) == 3
                and np.linalg.matrix_rank(aug) == 4)


@dataclass(frozen=True)
class IndependenceProof:
    """An exact witness-based proof of functional independence over ℚ."""

    witnesses: tuple[tuple[int, tuple[Fraction, Fraction, Fraction]], ...]
    rank: int
    independent: bool


def prove_functional_independence() -> IndependenceProof:
    r"""Exact proof (over ℚ) that ``a·PΩ + b·Pτ + c·Pσ = 0`` for all ``n ≥ 2``
    forces ``a = b = c = 0`` — the channels are functionally independent.

    Witness points (report §14.3): the two prime squares ``4 = 2²`` and
    ``9 = 3²`` give ``(1, 1, 1/2)`` and ``(1, 1, 1/3)``, whose difference
    ``(0, 0, 1/6)`` pins ``c = 0`` and then ``a + b = 0``; the semiprime
    ``6 = 2·3`` gives ``(1, 2, 5/6)``, which under ``c = 0`` pins ``a + 2b = 0``,
    so ``b = 0`` and ``a = 0``. The 3×3 witness matrix has **exact rank 3** over ℚ
    (fraction Gaussian elimination), which is stronger than a numerical rank.
    """
    witness_ns = (4, 9, 6)  # 2², 3², 2·3
    witnesses: list[tuple[int, tuple[Fraction, Fraction, Fraction]]] = []
    rows: list[list[Fraction]] = []
    for n in witness_ns:
        row = channels(n)
        witnesses.append((n, row))
        rows.append(list(row))
    rank = exact_rank(rows)
    return IndependenceProof(tuple(witnesses), rank, rank == 3)


def channel_correlations(lo: int, hi: int) -> np.ndarray:
    r"""The ``3 × 3`` Pearson correlation matrix of the channels.

    The channels are strongly correlated (all grow with compositeness) yet
    linearly independent — correlation is not dependence.
    """
    return np.corrcoef(channel_matrix(lo, hi).T)


# --------------------------------------------------------------------------- #
# 3. Ablation / minimality for primality
# --------------------------------------------------------------------------- #
def ablation_detects_primes(
    lo: int, hi: int, keep: tuple[int, ...]
) -> bool:
    r"""Whether the sum of the kept channels is ``0`` exactly on the primes."""
    if not keep:
        raise ValueError("keep must be non-empty")
    funcs = [_CHANNEL_FUNCS[i] for i in keep]
    zero = {
        n for n in range(max(lo, 2), hi + 1)
        if sum((fn(n) for fn in funcs), Fraction(0)) == 0
    }
    return zero == primes_in_range(lo, hi)


def minimal_channels_for_primality(lo: int, hi: int) -> int:
    r"""Smallest number of channels whose sum still detects the primes.

    Returns ``1`` when a single channel already characterises the primes (the
    realisation is redundant / non-minimal for prime detection).
    """
    for size in (1, 2, 3):
        for keep in combinations(range(3), size):
            if ablation_detects_primes(lo, hi, keep):
                return size
    return 3


def is_redundant_for_primality(lo: int, hi: int) -> bool:
    r"""Whether a proper subset of channels already detects the primes."""
    return minimal_channels_for_primality(lo, hi) < 3


# --------------------------------------------------------------------------- #
# Class-conditioned distributions
# --------------------------------------------------------------------------- #
def factor_class(n: int) -> str:
    r"""Factor-structure class: prime / prime_power / semiprime / composite_other."""
    factors = _factorization(n)
    omega = sum(factors.values())
    if omega == 1:
        return "prime"
    if len(factors) == 1:
        return "prime_power"
    if omega == 2:
        return "semiprime"
    return "composite_other"


def abundance_class(n: int) -> str:
    r"""Abundance class from ``σ(n)`` vs ``2n``: deficient / perfect / abundant."""
    s = divisor_sum(n)
    if s < 2 * n:
        return "deficient"
    if s == 2 * n:
        return "perfect"
    return "abundant"


def pressure_by_class(
    lo: int, hi: int, classifier=factor_class
) -> dict[str, dict[str, float]]:
    r"""Per-class pressure statistics ``{class: {count, mean, min, max}}``."""
    buckets: dict[str, list[float]] = {}
    for n in range(max(lo, 2), hi + 1):
        buckets.setdefault(classifier(n), []).append(
            float(arithmetic_pressure(n))
        )
    return {
        cls: {
            "count": float(len(vals)),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        for cls, vals in buckets.items()
    }


# --------------------------------------------------------------------------- #
# Completeness gate — a fourth channel is admitted only with full justification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FourthChannelCriteria:
    """The six conditions a candidate fourth pressure channel must satisfy.

    All default to ``False`` (unmet): structural completeness of the three-channel
    set is an **unproven hypothesis**, so no fourth channel is admitted and the
    "complete" language is not asserted.
    """

    independent_structural_meaning: bool = False
    not_a_function_of_existing: bool = False
    derived_from_model_not_accuracy: bool = False
    changes_diagnostic_capability: bool = False
    preserves_delta_nfr_zero_set: bool = False
    has_contract_and_tests: bool = False

    @property
    def admissible(self) -> bool:
        return (
            self.independent_structural_meaning
            and self.not_a_function_of_existing
            and self.derived_from_model_not_accuracy
            and self.changes_diagnostic_capability
            and self.preserves_delta_nfr_zero_set
            and self.has_contract_and_tests
        )


def admits_fourth_channel(criteria: FourthChannelCriteria) -> bool:
    r"""Whether a candidate fourth channel meets **every** admission criterion."""
    return criteria.admissible


def completeness_proven() -> bool:
    r"""Whether structural completeness of the three channels is proven.

    ``False``: no proof exists that no fourth independent pressure degree is
    relevant; completeness remains an open hypothesis (``NT-P07d``).
    """
    return False


def algorithmic_primality_is_circular() -> bool:
    r"""Whether using ``ΔNFR(n) = 0`` as a primality test is circular.

    ``True``: every channel is computed **from** the factorisation of ``n``
    (``Ω, τ, σ``), so the pressure presupposes the factorisation it would
    "detect". It is a structural descriptor, not a primality/factoring algorithm —
    the C5 circularity verdict is CIRCULAR (``NT-P07e``, no algorithmic claim).
    """
    return True
