r"""Finite cyclic additive-character diagnostics (R6).

For a binary indicator on Z/NZ, cyclic self-convolution equals the inverse DFT
of the squared DFT in exact arithmetic. The implementation compares a direct
finite count with NumPy FFT arithmetic; this is not an error theorem or an
ordinary integer Goldbach count without a no-aliasing argument.

The complete DFT is invertible. Every deterministic statistic of the same
input, linear or nonlinear, is therefore a function of it. The experiment
compares two compressed statistics, not additional information beyond Fourier.
Both convolution and power are quadratic in the input.

The candidate phase statistic is a second difference of an unwrapped FFT
phase, not the canonical graph circular-mean curvature. It depends on branch,
endpoint and zero-coefficient conventions. Shuffle and matched-random controls
preserve count; the Cramer Bernoulli control does not. Reported z-scores concern
the selected samples/statistics only. No new canonical phase law, useful
nodal evolution, prime-generation mechanism or Goldbach theorem is supplied."""

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
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return sieve.astype(float)


def cramer_indicator(n: int, *, seed: int) -> np.ndarray:
    r"""Draw independent Bernoulli entries using 1/log(m) for m>=2.

    This baseline does not condition on or preserve the finite prime count.
    Its probability expression exceeds one at m=2, so the comparison with a
    uniform random draw makes that entry present deterministically."""
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
    r"""Count ordered cyclic pairs in the support of a binary indicator.

    The implementation counts nonzero support entries and ignores their magnitude;
    it is not weighted convolution for arbitrary real-valued input. Modular
    wraparound is retained."""
    n = len(indicator)
    support = np.flatnonzero(indicator)
    out = np.zeros(n, dtype=float)
    for a in support:
        for b in support:
            out[(a + b) % n] += 1.0
    return out


def goldbach_fourier_residual(indicator: np.ndarray) -> float:
    r"""Compare direct cyclic support counts with the real inverse squared DFT.

    The convolution identity applies to binary indicators. A small numerical
    residual checks the implemented finite arithmetic; it does not prove a
    universal FFT error bound."""
    direct = cyclic_goldbach(indicator)
    spectral = np.real(np.fft.ifft(fourier_spectrum(indicator) ** 2))
    return float(np.max(np.abs(direct - spectral)))


# --------------------------------------------------------------------------- #
# Observables: classical (linear) vs TNFR candidate (non-linear phase)
# --------------------------------------------------------------------------- #
def power_concentration(indicator: np.ndarray) -> float:
    r"""Return the largest non-DC power divided by the mean after zeroing DC.

    The mean denominator includes all N entries, including the zeroed DC entry;
    it is not the mean over only N-1 non-DC bins."""
    power = np.abs(fourier_spectrum(indicator)) ** 2
    power[0] = 0.0
    mean = float(np.mean(power))
    if mean <= 0.0:
        return 0.0
    return float(np.max(power) / mean)


def phase_curvature_energy(indicator: np.ndarray) -> float:
    r"""Return mean squared interior second differences of unwrapped FFT phase.

    This calls NumPy angle/unwrap/diff, not the canonical graph K_phi owner.
    FFT zeros, phase unwrapping and omission of endpoint wrap are conventions of
    this statistic; no circular-curvature or dynamical interpretation is certified."""
    phase = np.unwrap(np.angle(fourier_spectrum(indicator)))
    curvature = np.diff(phase, 2)
    if curvature.size == 0:
        return 0.0
    return float(np.mean(curvature**2))


# --------------------------------------------------------------------------- #
# Controlled discrimination
# --------------------------------------------------------------------------- #
def _control_indicator(kind: str, count: int, n: int, *, seed: int) -> np.ndarray:
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
    r"""Compare one statistic with the selected finite control distribution.

    Shuffle and matched controls fix the number of support entries. The Cramer
    branch is an independent Bernoulli baseline and does not fix that count.
    A z-score describes this statistic and sample, not a general information or
    asymptotic number-theoretic theorem."""
    count = int(indicator.sum())
    n = len(indicator)
    true_value = summary(indicator)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_controls)
    controls = [summary(_control_indicator(kind, count, n, seed=int(s))) for s in seeds]
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
        phase_curvature_energy,
        primes,
        kind=kind,
        n_controls=n_controls,
        seed=seed,
    )
    return ExcessResult(classical_z, tnfr_z, kind)
