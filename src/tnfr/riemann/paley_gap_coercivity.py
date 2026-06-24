r"""TNFR Paley-gap coercivity diagnostic (P25 program).

Goal
----
Reframe the coercivity bottleneck (G4) using the **Paley-gap
philosophy** of Martínez Gamo, *Spectral note: Paley gap via
lambda_2 (residue circulants)*, Zenodo 10.5281/zenodo.17665853 v2
(November 2025).

The Paley-gap of the residue-circulant note has the structure

.. math::

    g(n) = \bigl|\lambda_2(\mathrm{residue\ circulant})
                 - \tfrac{n - \sqrt{n}}{2}\bigr|,

i.e. the **absolute difference between a computed spectral
quantity and a closed-form algebraic reference**.  The vanishing
:math:`g(n) = 0` selects an arithmetic structural condition
(:math:`n` prime, :math:`n \equiv 1\pmod 4`) up to the tested range
2601 — by *identity*, not by *bound*.

P25 applies the same philosophy to the TNFR-Riemann program.  The
ladder spectrum is generated three ways that **must agree** on
:math:`\mathrm{Re}(s) > 1` by construction:

1. **Route A — P12 closed form**:
   :math:`Z_{P12}(s) = \sum_{(\mu, w)} w\, e^{-s\mu}` over the
   prime-ladder spectrum
   (:func:`tnfr.riemann.von_mangoldt.tnfr_log_zeta_derivative`).

2. **Route B — P14 spectral trace**:
   :math:`Z_{P14}(s) = \mathrm{Tr}(\hat W e^{-s\hat H_{\mathrm{int}}})`
   from the prime-ladder Hamiltonian
   (:func:`tnfr.riemann.prime_ladder_hamiltonian.weighted_spectral_trace`).

3. **Reference — classical truncation**:
   :math:`Z_{\mathrm{cls}}(s) = \sum_{n \le N} \Lambda(n)\, n^{-s}`
   from :func:`tnfr.riemann.von_mangoldt.classical_log_zeta_derivative`.

Three Paley-gap quantities are then defined per :math:`\sigma`:

.. math::

    g_{P12}(\sigma) &= |Z_{P12}(\sigma) - Z_{\mathrm{cls}}(\sigma)|
        \quad\text{(P12 truncation fidelity)} \\
    g_{P14}(\sigma) &= |Z_{P14}(\sigma) - Z_{\mathrm{cls}}(\sigma)|
        \quad\text{(P14 fidelity, includes coupling deformation)} \\
    g_{\mathrm{cross}}(\sigma) &= |Z_{P14}(\sigma) - Z_{P12}(\sigma)|
        \quad\text{(pure coupling-induced deformation)}

At ``coupling = 0`` the cross gap :math:`g_{\mathrm{cross}}(\sigma)`
collapses to machine precision for every :math:`\sigma` — a
Paley-style **identity** between the closed-form construction (P12)
and the self-adjoint operator realisation (P14).  Any non-trivial
``coupling`` perturbs the Hamiltonian spectrum and produces a
measurable cross gap; this is the Paley-gap signal of structural
deformation.

Scope and honest limits
-----------------------
P25 is a **consistency diagnostic** in the Paley-gap style.  It does
**not** close G4 (RH localisation on :math:`\mathrm{Re}(s) = 1/2`).
The cross gap at ``coupling = 0`` vanishes by construction (P14 was
built to match P12 in the decoupled limit), so the zero-coupling
demo is a regression test, not a discovery.  The diagnostic value
appears at ``coupling > 0``, where the gap quantifies how strongly
inter-ladder coupling deforms the prime-ladder identity.

The Zenodo source note itself states: *reproducible; not a primality
proof*.  P25 inherits the same disclaimer at the coercivity level:
gap closure is an identity check, not a Riemann Hypothesis proof.

Notes
-----
For :math:`\sigma \le 1` the classical reference series diverges in
the :math:`N \to \infty` limit, so absolute Paley-gap magnitudes
below the line of convergence carry only relative meaning.  The
cross gap :math:`g_{\mathrm{cross}}` is well-defined for all
:math:`\sigma \in \mathbb{R}` in the finite-dimensional model
(both routes operate on the same finite spectrum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .prime_ladder_hamiltonian import PrimeLadderHamiltonian, weighted_spectral_trace
from .von_mangoldt import (
    PrimeLadderSpectrum,
    classical_log_zeta_derivative,
    tnfr_log_zeta_derivative,
)

__all__ = [
    "PaleyGapSweep",
    "paley_gap_p12",
    "paley_gap_p14",
    "paley_gap_cross",
    "sweep_paley_gap",
]


# ---------------------------------------------------------------------------
# Pointwise gaps
# ---------------------------------------------------------------------------


def paley_gap_p12(
    spectrum: PrimeLadderSpectrum,
    s: float,
    *,
    n_max_classical: int = 100_000,
) -> float:
    r"""Pointwise Paley-gap between P12 and the classical truncation.

    .. math::

        g_{P12}(\sigma) = |Z_{P12}(\sigma) - Z_{\mathrm{cls}}(\sigma)|.

    Parameters
    ----------
    spectrum : PrimeLadderSpectrum
        P12 spectrum (see :func:`build_prime_ladder_spectrum`).
    s : float
        Real spectral parameter.
    n_max_classical : int, default 100_000
        Truncation bound for the classical reference series.

    Returns
    -------
    float
        Absolute Paley-gap :math:`g_{P12}(\sigma)`.
    """
    z_p12 = tnfr_log_zeta_derivative(spectrum, float(s))
    z_cls = classical_log_zeta_derivative(float(s), int(n_max_classical))
    return float(abs(z_p12 - z_cls))


def paley_gap_p14(
    bundle: PrimeLadderHamiltonian,
    s: float,
    *,
    n_max_classical: int = 100_000,
) -> float:
    r"""Pointwise Paley-gap between P14 and the classical truncation.

    .. math::

        g_{P14}(\sigma) = |Z_{P14}(\sigma) - Z_{\mathrm{cls}}(\sigma)|.

    Parameters
    ----------
    bundle : PrimeLadderHamiltonian
        Prime-ladder Hamiltonian bundle (see
        :func:`build_prime_ladder_hamiltonian`).
    s : float
        Real spectral parameter.
    n_max_classical : int, default 100_000
        Truncation bound for the classical reference series.

    Returns
    -------
    float
        Absolute Paley-gap :math:`g_{P14}(\sigma)`.
    """
    z_p14 = weighted_spectral_trace(
        bundle.hamiltonian.H_int,
        bundle.weight_operator,
        float(s),
    )
    z_cls = classical_log_zeta_derivative(float(s), int(n_max_classical))
    return float(abs(z_p14 - z_cls))


def paley_gap_cross(
    bundle: PrimeLadderHamiltonian,
    s: float,
) -> float:
    r"""Pointwise cross Paley-gap between P14 and P12.

    .. math::

        g_{\mathrm{cross}}(\sigma) = |Z_{P14}(\sigma) - Z_{P12}(\sigma)|.

    Vanishes to machine precision when ``bundle.coupling == 0``
    (Paley-style identity between the closed-form construction and
    the self-adjoint operator realisation).

    Parameters
    ----------
    bundle : PrimeLadderHamiltonian
        Prime-ladder Hamiltonian bundle.
    s : float
        Real spectral parameter.

    Returns
    -------
    float
        Absolute cross Paley-gap :math:`g_{\mathrm{cross}}(\sigma)`.
    """
    z_p14 = weighted_spectral_trace(
        bundle.hamiltonian.H_int,
        bundle.weight_operator,
        float(s),
    )
    z_p12 = tnfr_log_zeta_derivative(bundle.spectrum, float(s))
    return float(abs(z_p14 - z_p12))


# ---------------------------------------------------------------------------
# Sweep certificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaleyGapSweep:
    r"""Paley-gap sweep over a real :math:`\sigma`-interval.

    Attributes
    ----------
    sigmas : numpy.ndarray
        Real spectral parameters tested.
    n_primes : int
        Number of primes used in the prime-ladder spectrum.
    max_power : int
        REMESH echo cap (``K`` in the P12/P14 notation).
    coupling : float
        Ladder coupling strength of the bundle.
    n_max_classical : int
        Truncation bound used for the classical reference.
    g_p12 : numpy.ndarray
        :math:`g_{P12}(\sigma)` per sigma.
    g_p14 : numpy.ndarray
        :math:`g_{P14}(\sigma)` per sigma.
    g_cross : numpy.ndarray
        :math:`g_{\mathrm{cross}}(\sigma)` per sigma.
    max_g_p12 : float
        Worst-case :math:`g_{P12}` over the sweep.
    max_g_p14 : float
        Worst-case :math:`g_{P14}` over the sweep.
    max_g_cross : float
        Worst-case :math:`g_{\mathrm{cross}}` over the sweep.

    Notes
    -----
    For ``coupling == 0`` the cross gap is at machine precision for
    every :math:`\sigma`, regardless of the classical truncation
    bound — that is the Paley-style identity.  The P12/P14 gaps
    against the classical reference both decay as ``n_primes`` and
    ``max_power`` grow (since the prime-ladder approximation to
    :math:`-\zeta'/\zeta` improves).
    """

    sigmas: np.ndarray
    n_primes: int
    max_power: int
    coupling: float
    n_max_classical: int
    g_p12: np.ndarray
    g_p14: np.ndarray
    g_cross: np.ndarray
    max_g_p12: float
    max_g_p14: float
    max_g_cross: float

    def summary(self) -> str:
        return (
            "PaleyGapSweep("
            f"sigma=[{self.sigmas[0]:.3f}, {self.sigmas[-1]:.3f}], "
            f"n_sigma={self.sigmas.size}, "
            f"n_primes={self.n_primes}, "
            f"max_power={self.max_power}, "
            f"coupling={self.coupling:.3e}, "
            f"n_max_classical={self.n_max_classical}, "
            f"max_g_p12={self.max_g_p12:.3e}, "
            f"max_g_p14={self.max_g_p14:.3e}, "
            f"max_g_cross={self.max_g_cross:.3e})"
        )


def sweep_paley_gap(
    bundle: PrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    n_max_classical: int = 100_000,
) -> PaleyGapSweep:
    r"""Sweep the three Paley-gap quantities over a real :math:`\sigma`-grid.

    Convenience driver that vectorises :func:`paley_gap_p12`,
    :func:`paley_gap_p14`, and :func:`paley_gap_cross` over the
    provided ``sigmas`` and packages the result.

    Parameters
    ----------
    bundle : PrimeLadderHamiltonian
        Prime-ladder Hamiltonian bundle.  Carries the reference P12
        spectrum, the diagonal weight operator, and the coupling
        strength used at construction.
    sigmas : sequence of float
        Real spectral parameters at which to evaluate the gaps.
    n_max_classical : int, default 100_000
        Truncation bound for the classical reference series.

    Returns
    -------
    PaleyGapSweep
    """
    s_arr = np.asarray(list(sigmas), dtype=float)
    if s_arr.size == 0:
        raise ValueError("sigmas must be non-empty")

    # Cache the classical reference once per sigma.
    z_cls = np.array(
        [classical_log_zeta_derivative(float(s), int(n_max_classical)) for s in s_arr],
        dtype=float,
    )

    z_p12 = np.array(
        [tnfr_log_zeta_derivative(bundle.spectrum, float(s)) for s in s_arr],
        dtype=float,
    )

    z_p14 = np.array(
        [
            weighted_spectral_trace(
                bundle.hamiltonian.H_int,
                bundle.weight_operator,
                float(s),
            )
            for s in s_arr
        ],
        dtype=float,
    )

    g_p12 = np.abs(z_p12 - z_cls)
    g_p14 = np.abs(z_p14 - z_cls)
    g_cross = np.abs(z_p14 - z_p12)

    return PaleyGapSweep(
        sigmas=s_arr,
        n_primes=int(bundle.spectrum.n_primes),
        max_power=int(bundle.spectrum.max_power),
        coupling=float(bundle.coupling),
        n_max_classical=int(n_max_classical),
        g_p12=g_p12,
        g_p14=g_p14,
        g_cross=g_cross,
        max_g_p12=float(np.max(g_p12)),
        max_g_p14=float(np.max(g_p14)),
        max_g_cross=float(np.max(g_cross)),
    )
