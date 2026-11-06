r"""TNFR χ-twisted Paley-gap coercivity diagnostic (P43 program).

L-track analogue of the ζ-track Paley-gap diagnostic (P25,
:mod:`tnfr.riemann.paley_gap_coercivity`).  The construction
mirrors P25 line-for-line, replacing every ζ-side ingredient by
its χ-twisted counterpart.

Goal
----
Apply the **Paley-gap philosophy** of Martínez Gamo, *Spectral
note: Paley gap via lambda_2 (residue circulants)*, Zenodo
10.5281/zenodo.17665853 v2 (November 2025), to the χ-twisted
TNFR-Riemann program.

The χ-twisted prime-ladder data are produced three ways that
**must agree** on :math:`\mathrm{Re}(s) > 1` by construction:

1. **Route A — P32 closed form**:
   :math:`Z_{P32}(s,\chi) = \sum_{(\mu,w)} w\, e^{-s\mu}` over
   the χ-twisted prime-ladder spectrum
   (:func:`tnfr.riemann.dirichlet_l.tnfr_log_l_derivative`).

2. **Route B — P34 spectral trace**:
   :math:`Z_{P34}(s,\chi) = \mathrm{Tr}(\hat W^{(\chi)}
   e^{-s\hat H_{\mathrm{int}}})` from the χ-twisted prime-ladder
   Hamiltonian
   (:func:`tnfr.riemann.twisted_prime_ladder_hamiltonian.twisted_weighted_spectral_trace`).

3. **Reference — classical truncation**:
   :math:`Z_{\mathrm{cls}}(s,\chi) =
   \sum_{n \le N} \chi(n)\,\Lambda(n)\, n^{-s}` from
   :func:`tnfr.riemann.dirichlet_l.classical_log_l_derivative`.

Three Paley-gap quantities are defined per :math:`\sigma`:

.. math::

    g_{P32}(\sigma) &= |Z_{P32}(\sigma,\chi)
                       - Z_{\mathrm{cls}}(\sigma,\chi)| \\
    g_{P34}(\sigma) &= |Z_{P34}(\sigma,\chi)
                       - Z_{\mathrm{cls}}(\sigma,\chi)| \\
    g_{\mathrm{cross}}(\sigma) &= |Z_{P34}(\sigma,\chi)
                                  - Z_{P32}(\sigma,\chi)|

At ``coupling = 0`` the cross gap :math:`g_{\mathrm{cross}}`
collapses to machine precision for every :math:`\sigma` and every
character — a Paley-style **identity** between the closed-form
construction (P32) and the self-adjoint operator realisation
(P34).  Any non-trivial ``coupling`` perturbs the χ-twisted
Hamiltonian spectrum and produces a measurable cross gap; this is
the Paley-gap signal of structural deformation transported to the
L-track.

Scope and honest limits
-----------------------
P43 is a **consistency diagnostic** in the Paley-gap style applied
to χ-twisted L-functions.  It does **not** close G4-χ (GRH
localisation on :math:`\mathrm{Re}(s) = 1/2` for :math:`L(s,\chi)`).
The cross gap at ``coupling = 0`` vanishes by construction (P34
was built to match P32 in the decoupled limit), so the
zero-coupling demo is a regression test, not a discovery.  The
diagnostic value appears at ``coupling > 0``, where the gap
quantifies how strongly inter-ladder coupling deforms the
χ-twisted prime-ladder identity.

The Zenodo source note itself states: *reproducible; not a
primality proof*.  P43 inherits the same disclaimer at the
L-track coercivity level: gap closure is an identity check, not a
Generalised Riemann Hypothesis proof.

Notes
-----
* Restricted to primitive real Dirichlet characters
  (``chi_3``, ``chi_4``, ``chi_5``) by the L-track scope of P32.
* All three quantities :math:`Z_{P32}, Z_{P34}, Z_{\mathrm{cls}}`
  are complex-valued in general; absolute differences are taken
  in :math:`\mathbb{C}`.
* For :math:`\sigma \le 1` the classical reference series diverges
  in the :math:`N \to \infty` limit, so absolute Paley-gap
  magnitudes below the line of convergence carry only relative
  meaning.  The cross gap :math:`g_{\mathrm{cross}}` is
  well-defined for all :math:`\sigma \in \mathbb{R}` in the
  finite-dimensional model (both routes operate on the same
  finite χ-twisted spectrum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dirichlet_l import (
    DirichletCharacter,
    TwistedPrimeLadderSpectrum,
    classical_log_l_derivative,
    tnfr_log_l_derivative,
)
from .twisted_prime_ladder_hamiltonian import (
    TwistedPrimeLadderHamiltonian,
    twisted_weighted_spectral_trace,
)

__all__ = [
    "TwistedPaleyGapSweep",
    "twisted_paley_gap_p32",
    "twisted_paley_gap_p34",
    "twisted_paley_gap_cross",
    "sweep_twisted_paley_gap",
]


# ---------------------------------------------------------------------------
# Pointwise gaps
# ---------------------------------------------------------------------------


def twisted_paley_gap_p32(
    spectrum: TwistedPrimeLadderSpectrum,
    chi: DirichletCharacter,
    s: float,
    *,
    n_max_classical: int = 100_000,
) -> float:
    r"""Pointwise χ-twisted Paley-gap between P32 and the classical truncation.

    .. math::

        g_{P32}(\sigma) = |Z_{P32}(\sigma,\chi)
                          - Z_{\mathrm{cls}}(\sigma,\chi)|.

    Parameters
    ----------
    spectrum : TwistedPrimeLadderSpectrum
        χ-twisted prime-ladder spectrum (P32).
    chi : DirichletCharacter
        Character defining the twist (consistent with ``spectrum``).
    s : float
        Real spectral parameter.
    n_max_classical : int, default 100_000
        Truncation bound for the classical reference series.

    Returns
    -------
    float
        Absolute Paley-gap :math:`g_{P32}(\sigma)`.
    """
    z_p32 = tnfr_log_l_derivative(spectrum, complex(s))
    z_cls = classical_log_l_derivative(chi, complex(s), int(n_max_classical))
    return float(abs(z_p32 - z_cls))


def twisted_paley_gap_p34(
    bundle: TwistedPrimeLadderHamiltonian,
    chi: DirichletCharacter,
    s: float,
    *,
    n_max_classical: int = 100_000,
) -> float:
    r"""Pointwise χ-twisted Paley-gap between P34 and the classical truncation.

    .. math::

        g_{P34}(\sigma) = |Z_{P34}(\sigma,\chi)
                          - Z_{\mathrm{cls}}(\sigma,\chi)|.

    Parameters
    ----------
    bundle : TwistedPrimeLadderHamiltonian
        P34 χ-twisted prime-ladder Hamiltonian bundle.
    chi : DirichletCharacter
        Character defining the twist (consistent with ``bundle``).
    s : float
        Real spectral parameter.
    n_max_classical : int, default 100_000
        Truncation bound for the classical reference series.

    Returns
    -------
    float
        Absolute Paley-gap :math:`g_{P34}(\sigma)`.
    """
    z_p34 = twisted_weighted_spectral_trace(
        bundle.hamiltonian.H_int,
        bundle.weight_operator,
        complex(s),
    )
    z_cls = classical_log_l_derivative(chi, complex(s), int(n_max_classical))
    return float(abs(z_p34 - z_cls))


def twisted_paley_gap_cross(
    bundle: TwistedPrimeLadderHamiltonian,
    s: float,
) -> float:
    r"""Pointwise cross χ-twisted Paley-gap between P34 and P32.

    .. math::

        g_{\mathrm{cross}}(\sigma)
            = |Z_{P34}(\sigma,\chi) - Z_{P32}(\sigma,\chi)|.

    Vanishes to machine precision when ``bundle.coupling == 0``
    (Paley-style identity between the closed-form construction P32
    and the self-adjoint operator realisation P34).

    Parameters
    ----------
    bundle : TwistedPrimeLadderHamiltonian
        P34 χ-twisted prime-ladder Hamiltonian bundle.
    s : float
        Real spectral parameter.

    Returns
    -------
    float
        Absolute cross Paley-gap :math:`g_{\mathrm{cross}}(\sigma)`.
    """
    z_p34 = twisted_weighted_spectral_trace(
        bundle.hamiltonian.H_int,
        bundle.weight_operator,
        complex(s),
    )
    z_p32 = tnfr_log_l_derivative(bundle.spectrum, complex(s))
    return float(abs(z_p34 - z_p32))


# ---------------------------------------------------------------------------
# Sweep certificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedPaleyGapSweep:
    r"""χ-twisted Paley-gap sweep over a real :math:`\sigma`-interval.

    Attributes
    ----------
    character_name : str
        Label of the χ character (``chi_3``, ``chi_4``, ``chi_5``).
    character_modulus : int
        Conductor :math:`q` of the character.
    sigmas : numpy.ndarray
        Real spectral parameters tested.
    n_primes : int
        Number of primes used in the χ-twisted prime-ladder spectrum.
    max_power : int
        REMESH echo cap (``K``).
    coupling : float
        Ladder coupling strength of the bundle.
    n_max_classical : int
        Truncation bound used for the classical reference.
    g_p32 : numpy.ndarray
        :math:`g_{P32}(\sigma)` per sigma.
    g_p34 : numpy.ndarray
        :math:`g_{P34}(\sigma)` per sigma.
    g_cross : numpy.ndarray
        :math:`g_{\mathrm{cross}}(\sigma)` per sigma.
    max_g_p32 : float
        Worst-case :math:`g_{P32}` over the sweep.
    max_g_p34 : float
        Worst-case :math:`g_{P34}` over the sweep.
    max_g_cross : float
        Worst-case :math:`g_{\mathrm{cross}}` over the sweep.

    Notes
    -----
    For ``coupling == 0`` the cross gap is at machine precision for
    every :math:`\sigma`, regardless of the classical truncation
    bound — that is the Paley-style identity on the L-track.  The
    P32/P34 gaps against the classical reference both decay as
    ``n_primes`` and ``max_power`` grow (since the χ-twisted
    prime-ladder approximation to :math:`-L'/L` improves).
    """

    character_name: str
    character_modulus: int
    sigmas: np.ndarray
    n_primes: int
    max_power: int
    coupling: float
    n_max_classical: int
    g_p32: np.ndarray
    g_p34: np.ndarray
    g_cross: np.ndarray
    max_g_p32: float
    max_g_p34: float
    max_g_cross: float

    def summary(self) -> str:
        return (
            "TwistedPaleyGapSweep("
            f"chi={self.character_name}(q={self.character_modulus}), "
            f"sigma=[{self.sigmas[0]:.3f}, {self.sigmas[-1]:.3f}], "
            f"n_sigma={self.sigmas.size}, "
            f"n_primes={self.n_primes}, "
            f"max_power={self.max_power}, "
            f"coupling={self.coupling:.3e}, "
            f"n_max_classical={self.n_max_classical}, "
            f"max_g_p32={self.max_g_p32:.3e}, "
            f"max_g_p34={self.max_g_p34:.3e}, "
            f"max_g_cross={self.max_g_cross:.3e})"
        )


def sweep_twisted_paley_gap(
    bundle: TwistedPrimeLadderHamiltonian,
    chi: DirichletCharacter,
    sigmas: Sequence[float],
    *,
    n_max_classical: int = 100_000,
) -> TwistedPaleyGapSweep:
    r"""Sweep the three χ-twisted Paley-gap quantities over a real
    :math:`\sigma`-grid.

    Convenience driver that vectorises :func:`twisted_paley_gap_p32`,
    :func:`twisted_paley_gap_p34`, and :func:`twisted_paley_gap_cross`
    over the provided ``sigmas`` and packages the result.

    Parameters
    ----------
    bundle : TwistedPrimeLadderHamiltonian
        P34 χ-twisted prime-ladder Hamiltonian bundle.  Carries the
        reference P32 spectrum, the diagonal χ-twisted weight
        operator, and the coupling strength used at construction.
    chi : DirichletCharacter
        Character defining the twist (consistent with ``bundle``).
    sigmas : sequence of float
        Real spectral parameters at which to evaluate the gaps.
    n_max_classical : int, default 100_000
        Truncation bound for the classical reference series.

    Returns
    -------
    TwistedPaleyGapSweep
    """
    s_arr = np.asarray(list(sigmas), dtype=float)
    if s_arr.size == 0:
        raise ValueError("sigmas must be non-empty")
    if int(bundle.character_modulus) != int(chi.modulus):
        raise ValueError(
            "character modulus mismatch between bundle "
            f"({bundle.character_modulus}) and chi ({chi.modulus})"
        )

    # Cache the classical reference once per sigma.
    z_cls = np.array(
        [
            classical_log_l_derivative(chi, complex(s), int(n_max_classical))
            for s in s_arr
        ],
        dtype=complex,
    )

    z_p32 = np.array(
        [tnfr_log_l_derivative(bundle.spectrum, complex(s)) for s in s_arr],
        dtype=complex,
    )

    z_p34 = np.array(
        [
            twisted_weighted_spectral_trace(
                bundle.hamiltonian.H_int,
                bundle.weight_operator,
                complex(s),
            )
            for s in s_arr
        ],
        dtype=complex,
    )

    g_p32 = np.abs(z_p32 - z_cls).astype(float)
    g_p34 = np.abs(z_p34 - z_cls).astype(float)
    g_cross = np.abs(z_p34 - z_p32).astype(float)

    return TwistedPaleyGapSweep(
        character_name=str(bundle.character_name),
        character_modulus=int(bundle.character_modulus),
        sigmas=s_arr,
        n_primes=int(bundle.spectrum.primes_active.size),
        max_power=int(bundle.spectrum.max_power),
        coupling=float(bundle.coupling),
        n_max_classical=int(n_max_classical),
        g_p32=g_p32,
        g_p34=g_p34,
        g_cross=g_cross,
        max_g_p32=float(np.max(g_p32)),
        max_g_p34=float(np.max(g_p34)),
        max_g_cross=float(np.max(g_cross)),
    )
