r"""TNFR-Riemann P36 -- Li/Keiper criterion for Dirichlet L-functions.

Structural analogue of P16 (:mod:`tnfr.riemann.li_keiper`) for general
primitive Dirichlet L-functions :math:`L(s, \chi)`.  Implements the
generalised Li-Keiper criterion (Bombieri-Lagarias 1999;
Lagarias 2007 generalisation to the Selberg class) for primitive real
``\chi`` and reuses the canonical TNFR P35 zero finder
(:func:`tnfr.riemann.twisted_weil_explicit_formula.find_dirichlet_l_zeros`)
to source the non-trivial zeros on the critical line.

Mathematical statement
----------------------

For a primitive Dirichlet character ``\chi`` mod ``q`` define the
``\chi``-twisted Li-Keiper coefficients

.. math::

    \lambda_n(\chi) \;=\; \sum_{\rho} \Bigl[ 1
                              - \bigl(1 - \tfrac{1}{\rho}\bigr)^n \Bigr],
    \qquad n \ge 1,

where ``\rho`` ranges over all non-trivial zeros of ``L(s, \chi)``,
counted with multiplicity and with the implicit pairing
``\rho \leftrightarrow \bar\rho``.  Lagarias' generalisation of Li's
1997 theorem then reads

.. math::

    \mathrm{GRH}_\chi \;\Longleftrightarrow\;
        \lambda_n(\chi) > 0 \quad \text{for every } n \ge 1.

For **primitive real** characters (Legendre / Kronecker symbols) the
functional equation pairs ``\rho`` with ``\bar\rho`` on
``\operatorname{Re}(s) = 1/2`` (under GRH), so the sum reduces to

.. math::

    \lambda_n(\chi) \;=\;
        \sum_{k} 2\, \operatorname{Re}\!\Bigl[
            1 - \bigl(1 - \tfrac{1}{\rho_k}\bigr)^n \Bigr],
    \qquad \rho_k = \tfrac{1}{2} + i\, t_k,\; t_k > 0,

structurally identical to the ``\zeta``-case (P16) once the zero list
is sourced from ``L(s, \chi)`` instead of ``\zeta(s)``.

TNFR reading
------------

In the canonical TNFR-Riemann pipeline the non-trivial zeros of
``L(s, \chi)`` appear as **resonance poles** of the ``\chi``-twisted
prime-ladder L-series after the P33 analytic continuation
(:mod:`tnfr.riemann.analytic_continuation_dirichlet`).  The P35 zero
finder (Hardy-Z bisection on ``Z_\chi(t)``) enumerates these poles on
the critical line.  Computing ``\lambda_n(\chi)`` from those locations
recasts the generalised Riemann Hypothesis for ``L(s, \chi)`` as a
TNFR-internal positivity diagnostic on the structural resonance
spectrum of the canonical P34 Hamiltonian.

Honesty disclaimer
------------------

This module does **NOT** prove GRH for any ``L(s, \chi)``.  Li's
criterion is GRH-equivalent: a finite verification of
``\lambda_n(\chi) > 0`` for ``n = 1, \ldots, N`` proves GRH only in
the limit ``N \to \infty`` with rigorous control of the truncation
error from the finite zero list.  The numerical evidence produced
here is a TNFR-native diagnostic witness, not a proof.

Public API
----------
``twisted_li_coefficients``           Compute :math:`\lambda_1, \ldots,
                                       \lambda_{n_{\max}}` from a list
                                       of upper-half-plane ``L(s, \chi)``
                                       zeros (delegates to the canonical
                                       P16 routine, which is L-function
                                       agnostic).
``TwistedLiKeiperCertificate``         Frozen result with positivity
                                       flag, zero count, and summary.
``verify_twisted_li_keiper_criterion`` End-to-end verification: enumerate
                                       ``L(s, \chi)`` zeros via P35 then
                                       check positivity of every
                                       ``\lambda_n(\chi)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..mathematics.unified_numerical import np
from .dirichlet_l import DirichletCharacter
from .li_keiper import li_coefficients_from_zeros
from .twisted_weil_explicit_formula import (
    character_parity,
    find_dirichlet_l_zeros,
)


# ---------------------------------------------------------------------------
# Thin re-export so that callers do not need to import the P16 routine
# directly.  The mathematics is identical (sum is L-function agnostic).
# ---------------------------------------------------------------------------

def twisted_li_coefficients(
    zeros_upper,
    n_max: int,
    *,
    dps: int = 50,
):
    r"""Compute :math:`\lambda_n(\chi)` for :math:`n = 1, \ldots, n_{\max}`.

    Thin wrapper around
    :func:`tnfr.riemann.li_keiper.li_coefficients_from_zeros` --
    the sum over zeros is L-function agnostic, so the only thing
    that distinguishes the ``\chi``-case from the ``\zeta``-case is the
    source of the zero list.  Kept as a named entry point so that
    callers can express their intent (``λ_n`` of an L-function) at the
    API level.

    See :func:`tnfr.riemann.li_keiper.li_coefficients_from_zeros` for
    full documentation of the algorithm and precision controls.
    """
    return li_coefficients_from_zeros(zeros_upper, n_max, dps=dps)


# ---------------------------------------------------------------------------
# Certificate dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwistedLiKeiperCertificate:
    r"""Result of a ``\chi``-twisted Li-Keiper positivity check.

    Attributes
    ----------
    character_name
        Human-readable label of the Dirichlet character (e.g.
        ``"chi_3"``).
    character_modulus
        Conductor ``q`` of ``\chi``.
    character_parity
        Parity ``a = (1 - \chi(-1))/2 \in \{0, 1\}`` of ``\chi``.
    n_max
        Highest Li index computed.
    n_zeros_used
        Number of non-trivial zeros of ``L(s, \chi)`` used in the
        truncation.
    t_max
        Upper search bound on the critical line; every zero in
        ``(0, t_{\max})`` is enumerated.
    zeros
        Real array of the ``n_zeros_used`` positive imaginary parts
        ``t_k`` (ascending).
    lambda_coefficients
        Array of shape ``(n_max,)`` with ``\lambda_n(\chi)``.
    positivity
        ``True`` iff every ``\lambda_n(\chi) > 0`` for
        ``n = 1, \ldots, n_{\max}``.
    min_lambda
        Minimum of the computed coefficients (negative value
        immediately falsifies GRH_χ within the truncation).
    """

    character_name: str
    character_modulus: int
    character_parity: int
    n_max: int
    n_zeros_used: int
    t_max: float
    zeros: np.ndarray
    lambda_coefficients: np.ndarray
    positivity: bool
    min_lambda: float

    def summary(self) -> str:
        r"""Return a multi-line human-readable summary."""
        lam = self.lambda_coefficients
        lines = [
            "Twisted Li-Keiper certificate (TNFR-Riemann P36)",
            "-" * 60,
            f"  character           = {self.character_name}",
            f"  modulus q           = {self.character_modulus}",
            f"  parity a            = {self.character_parity}",
            f"  n_max               = {self.n_max}",
            f"  n_zeros_used        = {self.n_zeros_used}",
            f"  t_max (search)      = {self.t_max:.3f}",
            f"  lambda_1            = {lam[0]:+.6e}",
            f"  lambda_{self.n_max}".ljust(22)
            + f"= {lam[-1]:+.6e}",
            f"  min_n lambda_n      = {self.min_lambda:+.6e}",
            f"  positivity (GRH_chi)= {self.positivity}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-end verification
# ---------------------------------------------------------------------------

def verify_twisted_li_keiper_criterion(
    chi: DirichletCharacter,
    *,
    n_max: int = 30,
    t_max: float = 80.0,
    t_min: float = 0.5,
    initial_step: float = 0.25,
    dps: int = 50,
    zero_search_dps: int = 30,
    character_label: str | None = None,
) -> TwistedLiKeiperCertificate:
    r"""Verify the χ-twisted Li-Keiper positivity criterion.

    Steps
    -----
    1. Enumerate every zero ``\rho_k = 1/2 + i t_k`` of ``L(s, \chi)``
       with ``0 < t_k < t_{\max}`` via P35
       (:func:`find_dirichlet_l_zeros`).
    2. Compute ``\lambda_n(\chi)`` for ``n = 1, \ldots, n_{\max}`` via
       :func:`twisted_li_coefficients` (delegates to the canonical P16
       routine, mpmath-precision-controlled).
    3. Check positivity of every coefficient.

    Parameters
    ----------
    chi : DirichletCharacter
        Primitive real character (Legendre / Kronecker symbol).
    n_max : int, default 30
        Highest Li-Keiper index to compute.
    t_max : float, default 80.0
        Upper bound of the critical-line zero search.  Must be large
        enough that the tail truncation error in
        ``\lambda_n(\chi)`` keeps the sign of every coefficient.
    t_min : float, default 0.5
        Lower bound of the zero search (must be ``> 0``).
    initial_step : float, default 0.25
        Step for the Hardy-Z sign-change scan.  Should be much
        smaller than the typical zero spacing of ``L(s, \chi)``.
    dps : int, default 50
        Mpmath working decimal precision for the ``\lambda_n``
        computation.
    zero_search_dps : int, default 30
        Mpmath working decimal precision for the Hardy-Z zero search
        (lower than ``dps`` because bisection of a real-valued
        function is intrinsically less precision-hungry than the
        cancellation-heavy ``\lambda_n`` sum).
    character_label : str, optional
        Human-readable name for the character (defaults to
        ``chi.name`` if defined, else ``f"chi mod {chi.modulus}"``).

    Returns
    -------
    TwistedLiKeiperCertificate

    Notes
    -----
    For primitive real ``\chi`` of small conductor and ``t_{\max}``
    on the order of ``80``, the first ``\sim 30`` Li-Keiper
    coefficients are positive and well-separated from zero,
    matching the numerical evidence in Lagarias 2007 and
    Bombieri-Lagarias 1999 for Dirichlet L-functions.  A single
    negative ``\lambda_n`` would falsify GRH for ``L(s, \chi)``
    within the truncation precision; the absence of one is
    consistent with GRH but does NOT prove it.
    """
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    if t_max <= t_min:
        raise ValueError("t_max must be > t_min")
    if t_min <= 0:
        raise ValueError("t_min must be > 0")

    # --- Resolve label -----------------------------------------------------
    label = character_label
    if label is None:
        label = getattr(chi, "name", None) or f"chi mod {int(chi.modulus)}"

    parity = character_parity(chi)
    q = int(chi.modulus)

    # --- Step 1: enumerate zeros via P35 ----------------------------------
    zero_ts = find_dirichlet_l_zeros(
        chi,
        t_max=t_max,
        t_min=t_min,
        initial_step=initial_step,
        dps=zero_search_dps,
    )
    if len(zero_ts) == 0:
        raise RuntimeError(
            f"No zeros of L(s, {label}) found in (t_min={t_min}, "
            f"t_max={t_max}). Increase t_max or decrease initial_step."
        )

    zeros_upper = [complex(0.5, float(t)) for t in zero_ts]
    zeros_arr = np.asarray(zero_ts, dtype=float)

    # --- Step 2: compute lambda_n via canonical P16 routine ----------------
    lambda_arr = twisted_li_coefficients(zeros_upper, n_max, dps=dps)

    # --- Step 3: positivity check -----------------------------------------
    positivity = bool(np.all(lambda_arr > 0))
    min_lambda = float(lambda_arr.min())

    return TwistedLiKeiperCertificate(
        character_name=str(label),
        character_modulus=q,
        character_parity=int(parity),
        n_max=int(n_max),
        n_zeros_used=len(zero_ts),
        t_max=float(t_max),
        zeros=zeros_arr,
        lambda_coefficients=lambda_arr,
        positivity=positivity,
        min_lambda=min_lambda,
    )


__all__ = [
    "twisted_li_coefficients",
    "TwistedLiKeiperCertificate",
    "verify_twisted_li_keiper_criterion",
]
