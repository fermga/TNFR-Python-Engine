r"""Controlled additive-character number theory (R6).

The Goldbach representation function is the self-convolution of the prime
indicator, ``r_2 = 1_P * 1_P``.  On ``ℤ/Nℤ`` its additive-character (Fourier)
transform factorises, ``\hat r_2(k) = \hat{1_P}(k)^2`` — the circle method.  This
module asks the honest R6 question: does a *TNFR* reading of this additive phase
add a structural observable **beyond** the classical Fourier / singular-series
description, once density-matched controls are in place?

**What is proved (DERIVED).**  The cyclic Goldbach function equals
``IFFT(\hat{1_P}^2)`` exactly, so every **linear** additive-character observable of
the prime indicator — including ``r_2`` and the power spectrum — is a function of
the Fourier spectrum.  Any TNFR contribution must therefore come from a
**non-linear** phase observable and must be shown to exceed Fourier *under
controls*.

**What is measured (constructive negative).**  The candidate non-linear observable
here (phase-curvature energy, the TNFR ``K_φ`` read of the spectrum phase) shows
**no discriminating excess** over the classical power-spectrum concentration when
separating the primes from density-matched controls (Cramér, shuffle, matched
random).  Hence the claim *"TNFR additive phase adds information beyond classical
Fourier"* (``NT-P06``) is **OPEN / not established**; the tested observable gives a
negative.  No proof of Goldbach or any open problem is claimed; the target
property is never used to define the phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "prime_indicator",
    "cramer_indicator",
    "shuffled_indicator",
    "matched_random_indicator",
    "constellation_indicator",
    "fourier_spectrum",
    "cyclic_goldbach",
    "goldbach_fourier_residual",
    "power_concentration",
    "phase_curvature_energy",
    "discrimination_z",
    "ExcessResult",
    "excess_over_fourier",
]


# --------------------------------------------------------------------------- #
# Indicators (the prime signal and its density-matched controls)
# --------------------------------------------------------------------------- #
def prime_indicator(n: int) -> np.ndarray:
    r"""Indicator ``1_P`` of the primes on ``{0, …, n−1}`` via a sieve."""
    if n < 2:
        return np.zeros(max(n, 0), dtype=float)
    sieve = np.ones(n, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p:: p] = False
    return sieve.astype(float)


def cramer_indicator(n: int, *, seed: int) -> np.ndarray:
    r"""Cramér random model: ``P(n present) = 1/ln n`` (density-matched primes)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=float)
    for m in range(2, n):
        if rng.random() < 1.0 / np.log(m):
            out[m] = 1.0
    return out


def shuffled_indicator(indicator: np.ndarray, *, seed: int) -> np.ndarray:
    r"""A random permutation of the indicator (same count, no arithmetic order)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(indicator)


def matched_random_indicator(count: int, n: int, *, seed: int) -> np.ndarray:
    r"""Uniform random subset of ``{0, …, n−1}`` of exactly ``count`` elements."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=float)
    if count > 0:
        out[rng.choice(n, size=count, replace=False)] = 1.0
    return out


def constellation_indicator(n: int, offsets: tuple[int, ...]) -> np.ndarray:
    r"""Prime-constellation indicator: ``1`` at ``m`` iff ``m + o`` is prime for
    every ``o`` in ``offsets`` (``offsets = (0, 2)`` gives twin-prime starts)."""
    base = prime_indicator(n + max(offsets) + 1)
    out = np.zeros(n, dtype=float)
    for m in range(n):
        if all(base[m + o] for o in offsets):
            out[m] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Additive-character (Fourier) reading and the exact reduction
# --------------------------------------------------------------------------- #
def fourier_spectrum(indicator: np.ndarray) -> np.ndarray:
    r"""Additive-character transform ``\hat 1(k) = Σ_n 1(n) e^{-2πi kn/N}``."""
    return np.fft.fft(indicator)


def cyclic_goldbach(indicator: np.ndarray) -> np.ndarray:
    r"""Cyclic self-convolution ``r_2(m) = Σ_n 1(n) 1((m−n) mod N)`` (direct)."""
    n = len(indicator)
    support = np.flatnonzero(indicator)
    out = np.zeros(n, dtype=float)
    for a in support:
        for b in support:
            out[(a + b) % n] += 1.0
    return out


def goldbach_fourier_residual(indicator: np.ndarray) -> float:
    r"""``max |r_2 − IFFT(\hat 1^2)|`` — zero proves the additive reading is Fourier."""
    direct = cyclic_goldbach(indicator)
    spectral = np.real(np.fft.ifft(fourier_spectrum(indicator) ** 2))
    return float(np.max(np.abs(direct - spectral)))


# --------------------------------------------------------------------------- #
# Observables: classical (linear) vs TNFR candidate (non-linear phase)
# --------------------------------------------------------------------------- #
def power_concentration(indicator: np.ndarray) -> float:
    r"""Classical: peak non-DC power over mean, ``max_{k≠0}|\hat1|² / mean_{k≠0}``."""
    power = np.abs(fourier_spectrum(indicator)) ** 2
    power[0] = 0.0
    mean = float(np.mean(power))
    if mean <= 0.0:
        return 0.0
    return float(np.max(power) / mean)


def phase_curvature_energy(indicator: np.ndarray) -> float:
    r"""TNFR candidate: mean squared second difference of the spectrum phase
    (the ``K_φ`` curvature read of the additive phase)."""
    phase = np.unwrap(np.angle(fourier_spectrum(indicator)))
    curvature = np.diff(phase, 2)
    if curvature.size == 0:
        return 0.0
    return float(np.mean(curvature ** 2))


# --------------------------------------------------------------------------- #
# Controlled discrimination
# --------------------------------------------------------------------------- #
def _control_indicator(
    kind: str, count: int, n: int, *, seed: int
) -> np.ndarray:
    if kind == "matched":
        return matched_random_indicator(count, n, seed=seed)
    if kind == "cramer":
        return cramer_indicator(n, seed=seed)
    if kind == "shuffle":
        return shuffled_indicator(prime_indicator(n), seed=seed)
    raise ValueError(f"unknown control kind: {kind}")


def discrimination_z(
    summary, indicator: np.ndarray, *, kind: str, n_controls: int, seed: int
) -> float:
    r"""Z-score of ``summary(indicator)`` against a distribution of ``n_controls``
    density-matched controls of the given ``kind``."""
    count = int(indicator.sum())
    n = len(indicator)
    true_value = summary(indicator)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2 ** 31 - 1, size=n_controls)
    controls = [
        summary(_control_indicator(kind, count, n, seed=int(s)))
        for s in seeds
    ]
    mean = float(np.mean(controls))
    std = float(np.std(controls))
    if std <= 0.0:
        return 0.0
    return (true_value - mean) / std


@dataclass(frozen=True)
class ExcessResult:
    """Whether the TNFR phase observable exceeds the classical one under controls."""

    classical_z: float
    tnfr_z: float
    control_kind: str

    @property
    def tnfr_exceeds(self) -> bool:
        """``True`` only if the TNFR observable discriminates more strongly."""
        return abs(self.tnfr_z) > abs(self.classical_z)


def excess_over_fourier(
    n: int, *, seed: int = 0, n_controls: int = 30, kind: str = "matched"
) -> ExcessResult:
    r"""Compare the TNFR phase-curvature discrimination against the classical
    power-spectrum discrimination of the primes under density-matched controls.

    Returns an :class:`ExcessResult`; ``tnfr_exceeds`` is expected to be ``False``
    (constructive negative for ``NT-P06``).
    """
    primes = prime_indicator(n)
    classical_z = discrimination_z(
        power_concentration, primes, kind=kind, n_controls=n_controls, seed=seed
    )
    tnfr_z = discrimination_z(
        phase_curvature_energy, primes, kind=kind, n_controls=n_controls,
        seed=seed,
    )
    return ExcessResult(classical_z, tnfr_z, kind)
