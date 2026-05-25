r"""TNFR-Riemann P35 — Weil/Guinand explicit formula for Dirichlet L-functions.

This module is the structural analogue of P15
(:mod:`tnfr.riemann.weil_explicit_formula`) for general primitive
Dirichlet L-functions :math:`L(s, \chi)`.  It verifies the classical
Weil-Guinand explicit formula linking the non-trivial zeros of
:math:`L(s, \chi)` to the χ-twisted prime-power spectral side computed
from the canonical TNFR P34 χ-twisted prime-ladder Hamiltonian
(:mod:`tnfr.riemann.twisted_prime_ladder_hamiltonian`).

Mathematical statement (primitive real non-principal χ mod q, parity a)
-----------------------------------------------------------------------

Let ``h`` be a real even Schwartz function on the real line with Fourier
transform

.. math::

    g(u) \;=\; \frac{1}{2\pi}\int_{-\infty}^{\infty} h(t)\,e^{-itu}\,dt.

For a primitive non-principal Dirichlet character ``\chi`` of modulus
``q`` and parity ``a = (1 - \chi(-1))/2 \in \{0, 1\}``, the
Weil-Guinand explicit formula reads

.. math::

    \sum_{\gamma} h(\gamma)
       \;=\;
       g(0)\,\log(q/\pi)
       \;+\; \frac{1}{2\pi}\!\int_{-\infty}^{\infty}\!\!
            h(t)\,\operatorname{Re}\psi\!\Bigl(\tfrac14 + \tfrac{a}{2}
                                              + \tfrac{it}{2}\Bigr) dt
       \;-\; 2\,\operatorname{Re}\!\sum_{n\ge 1}
            \frac{\chi(n)\,\Lambda(n)}{\sqrt{n}}\, g(\log n),

where ``\gamma`` runs over imaginary parts of all non-trivial zeros
``\rho = 1/2 + i\gamma`` of ``L(s, \chi)``.

Compared to the ζ analogue (P15) two structural differences appear:

* The **pole side** ``h(i/2) + h(-i/2)`` is **absent** because
  ``L(s, \chi)`` is entire for non-principal ``\chi``.
* The conductor ``q`` enters the archimedean **constant term** as
  ``g(0)\,\log(q/\pi)`` (the ζ formula has ``q = 1``, giving
  ``g(0)\log(1/\pi) = -g(0)\log\pi`` — i.e. ``log_pi`` in P15).
* The digamma argument is shifted by ``a/2`` so that the gamma
  factor of the completed L-function,
  ``\Gamma((s + a)/2)``, is reproduced correctly.

For **real** characters ``\chi`` (Legendre / Kronecker symbols) the
non-trivial zeros come in conjugate pairs ``\rho = 1/2 \pm i\gamma``,
so the zero side reduces to ``2 \sum_{\gamma > 0} h(\gamma)``.

Connection to TNFR P34
----------------------

The χ-twisted prime-power side is **exactly** a spectral functional on
the P34 Hamiltonian ``H = \operatorname{diag}(k\log p)`` with χ-twisted
complex weight operator ``W^{(\chi)} = \operatorname{diag}(\chi(p)^k \log p)``:

.. math::

    -2\,\operatorname{Re}\!\sum_{n\ge 1}
        \frac{\chi(n)\,\Lambda(n)}{\sqrt{n}}\, g(\log n)
    \;=\;
    -2\,\operatorname{Re}\,
        \operatorname{Tr}\!\bigl(\hat W^{(\chi)}
            e^{-\hat H / 2}\, g(\hat H)\bigr).

Indeed, ``n = p^k`` (with ``p \nmid q``) gives ``\chi(n) = \chi(p)^k``,
``\Lambda(n) = \log p``, ``\sqrt{n} = e^{(k\log p)/2}`` and
``g(\log n) = g(k\log p)``.  Each eigenvalue ``E_{p,k} = k\log p`` of
the P34 Hamiltonian is a node ``|p, k\rangle`` carrying complex weight
``\chi(p)^k \log p``.  This is the canonical TNFR realisation of the
prime side of Weil's formula for ``L(s, \chi)``.

Zero side
---------

Because mpmath has no direct ``dirichlet_l_zero`` analogue of
``zetazero``, the zeros of ``L(s, \chi)`` on the critical line are
located by **Hardy-Z bisection**:

.. math::

    Z_\chi(t) \;=\; e^{i\theta(t)}\,L(1/2 + it, \chi),
    \qquad
    \theta(t) = \operatorname{Im}\!\Bigl[\tfrac{it}{2}\log(q/\pi)
                + \log\Gamma\bigl(\tfrac14 + \tfrac{a}{2}
                                 + \tfrac{it}{2}\bigr)\Bigr].

For primitive real ``\chi``, ``Z_\chi`` is **real**-valued on the
critical line and its zeros coincide with the zeros of ``L(s, \chi)``
at ``s = 1/2 + it``.  A sign-change scan followed by bracket
bisection enumerates all zeros in a prescribed window
``t \in (0, t_{\max}]`` to arbitrary precision.

What this module proves and what it does not
--------------------------------------------

The Weil-Guinand explicit formula for ``L(s, \chi)`` is a *theorem*
of analytic number theory (Weil 1952), independent of RH and GRH.
What is **new** here is purely instrumental: every term of this
identity is computed **inside the canonical TNFR formalism** —
the prime side from the P34 Hamiltonian, the zero side from L-zeros
located on the critical line via Hardy-Z bisection on top of the
P33 continuation infrastructure
(:mod:`tnfr.riemann.analytic_continuation_dirichlet`).

This closes gap **G3$_\chi$** of the TNFR-Riemann programme for every
primitive Dirichlet L-function in the operational sense — there is a
canonical TNFR realisation of both sides of Weil's bridge.

This module does **not** establish gap G4 (``=`` RH) or its
generalisation **GRH**.  The zero locations are taken **as input** via
their numerical values on the critical line; their localisation on
``\operatorname{Re}(s) = 1/2`` is not derived here.  The identity itself
holds whatever the location of the zeros.

This module currently supports primitive **real** characters
(Legendre / Kronecker symbols).  Complex characters require an
extension to handle conjugate-pair zero enumeration and are left
as future work.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import mpmath
from mpmath import mp
from scipy import integrate

from ..mathematics.unified_numerical import np
from .dirichlet_l import DirichletCharacter
from .twisted_prime_ladder_hamiltonian import TwistedPrimeLadderHamiltonian
from .weil_explicit_formula import GaussianTestFunction, gaussian_test_function

__all__ = [
    "character_parity",
    "twisted_weil_constant_term",
    "twisted_weil_archimedean_integral",
    "twisted_weil_prime_side_from_hamiltonian",
    "find_dirichlet_l_zeros",
    "twisted_weil_zero_side",
    "TwistedWeilExplicitFormulaCertificate",
    "verify_twisted_weil_explicit_formula",
]


# ----------------------------------------------------------------------
# Character parity
# ----------------------------------------------------------------------


def character_parity(chi: DirichletCharacter) -> int:
    r"""Return ``a = (1 - \chi(-1))/2 \in \{0, 1\}``.

    ``a = 0`` if ``\chi`` is *even* (``\chi(-1) = +1``), ``a = 1`` if
    ``\chi`` is *odd* (``\chi(-1) = -1``).  Raises :class:`ValueError`
    if ``\chi(-1)`` is not ``\pm 1`` (which would mean ``\chi`` is not
    a primitive real character).
    """
    q = chi.modulus
    val = chi.values[(-1) % q]
    if abs(val.imag) > 1e-9:
        raise ValueError(
            f"Character {chi.name} is not real-valued at -1 "
            f"(chi(-1) = {val})."
        )
    re = float(val.real)
    if re > 0.5:
        return 0
    if re < -0.5:
        return 1
    raise ValueError(
        f"chi(-1) = {val} is not +/-1; chi may not be primitive."
    )


# ----------------------------------------------------------------------
# Archimedean and constant terms
# ----------------------------------------------------------------------


def twisted_weil_constant_term(
    chi: DirichletCharacter, test: GaussianTestFunction
) -> float:
    r"""Return the archimedean constant term ``g(0) \log(q/\pi)``.

    For ``q = 1`` this reduces to ``-g(0)\log\pi`` — i.e. the
    ``log_pi`` term used in the ζ formula (P15).
    """
    q = chi.modulus
    return test.g_zero() * math.log(q / math.pi)


def _digamma_real_part_with_shift(t: float, a: int) -> float:
    r"""Return ``\operatorname{Re}\psi(1/4 + a/2 + it/2)``."""
    val = mpmath.digamma(mpmath.mpc(0.25 + 0.5 * a, t / 2.0))
    return float(val.real)


def twisted_weil_archimedean_integral(
    chi: DirichletCharacter,
    test: GaussianTestFunction,
    *,
    integration_limit: float | None = None,
    quad_kwargs: dict | None = None,
) -> float:
    r"""Compute the archimedean digamma integral for ``L(s, \chi)``.

    Returns

    .. math::

        \frac{1}{2\pi}\int_{-\infty}^{\infty}
            h(t)\,\operatorname{Re}\psi\!\Bigl(\tfrac14 + \tfrac{a}{2}
                                              + \tfrac{it}{2}\Bigr)\,dt,

    with the parity-dependent shift ``a/2`` (``a = 0`` for even ``\chi``,
    ``a = 1`` for odd ``\chi``).  The integral is evaluated by
    :func:`scipy.integrate.quad` after truncating the domain to
    ``[-L, L]`` with ``L = 10\sigma`` so the Gaussian envelope is
    negligible at the cut.
    """
    a = character_parity(chi)
    if integration_limit is None:
        integration_limit = 10.0 * test.sigma
    kw = {"limit": 200, "epsabs": 1e-14, "epsrel": 1e-12}
    if quad_kwargs:
        kw.update(quad_kwargs)

    def integrand(t: float) -> float:
        return test.h(t) * _digamma_real_part_with_shift(t, a)

    val, _err = integrate.quad(
        integrand, -integration_limit, integration_limit, **kw
    )
    return float(val / (2.0 * math.pi))


# ----------------------------------------------------------------------
# Prime side from P34
# ----------------------------------------------------------------------


def twisted_weil_prime_side_from_hamiltonian(
    bundle: TwistedPrimeLadderHamiltonian,
    test: GaussianTestFunction,
) -> float:
    r"""Compute the χ-twisted prime side of Weil's formula from P34.

    Returns

    .. math::

        -2\,\operatorname{Re}\,
            \operatorname{Tr}\!\bigl(\hat W^{(\chi)} e^{-\hat H/2}
                                    g(\hat H)\bigr)
        \;=\;
        -2\!\sum_{(p,k):\,p\nmid q}
            \operatorname{Re}[\chi(p)^k]\,\log(p)\,p^{-k/2}\,g(k\log p),

    using the eigendecomposition of the P34 Hamiltonian carried by
    ``bundle.hamiltonian``.  At ``J_0 = 0`` (the canonical default of
    P34) the Hamiltonian is diagonal in the node basis and the trace
    collapses to a weighted sum over nodes ``(p, k)``.

    For real characters the imaginary part of every node weight
    vanishes; for complex characters only the real part of the trace
    contributes to the explicit formula.
    """
    eigvals, eigvecs = bundle.hamiltonian.get_spectrum()
    eigvals_real = np.real(eigvals)
    g_values = np.array(
        [test.g(float(e)) for e in eigvals_real], dtype=float
    )
    half_decay = np.exp(-eigvals_real / 2.0)
    # W^(chi) is diagonal complex in the node basis
    weights_diag = np.diag(bundle.weight_operator)
    # Transform to eigenbasis: <e_i|W|e_i> = sum_n |<n|e_i>|^2 * W_nn
    W_diag_eig = np.einsum(
        "ni,n,ni->i",
        np.conj(eigvecs), weights_diag, eigvecs,
    )
    contributions = np.real(W_diag_eig * half_decay * g_values)
    return float(-2.0 * np.sum(contributions))


# ----------------------------------------------------------------------
# Zero side: Hardy-Z bisection on the critical line
# ----------------------------------------------------------------------


def _chi_to_mpmath_list(chi: DirichletCharacter) -> list:
    """Convert a DirichletCharacter to mp.dirichlet's list form."""
    out: list = []
    for v in chi.values:
        if abs(v.imag) < 1e-15:
            out.append(mp.mpf(v.real))
        else:
            out.append(mp.mpc(v.real, v.imag))
    return out


def _hardy_z_chi(t, chi_list, q, a):
    r"""Evaluate the real-valued Hardy ``Z_\chi(t)``.

    .. math::

        Z_\chi(t) \;=\; e^{i\theta(t)}\, L(1/2 + it, \chi),
        \quad
        \theta(t) = \operatorname{Im}\!\bigl[\tfrac{it}{2}\log(q/\pi)
            + \log\Gamma(\tfrac14 + \tfrac{a}{2} + \tfrac{it}{2})\bigr].

    For primitive real ``\chi`` this is real-valued and its zeros
    coincide with the zeros of ``L(s, \chi)`` on the critical line.
    The input ``t`` is an ``mp.mpf``; the result is a Python ``float``.
    """
    gamma_arg = mp.mpc(mp.mpf(1) / 4 + mp.mpf(a) / 2, t / 2)
    log_phase = (
        mp.mpc(0, t / 2) * mp.log(mp.mpf(q) / mp.pi)
        + mp.loggamma(gamma_arg)
    )
    theta = log_phase.imag
    phase = mp.exp(mp.mpc(0, theta))
    L_val = mp.dirichlet(mp.mpc(mp.mpf("0.5"), t), chi_list)
    return float((phase * L_val).real)


def find_dirichlet_l_zeros(
    chi: DirichletCharacter,
    *,
    t_max: float,
    t_min: float = 0.5,
    initial_step: float = 0.25,
    dps: int = 30,
    bisection_tol: float = 1e-12,
) -> list[float]:
    r"""Locate positive-axis zeros of ``L(s, \chi)`` on the critical line.

    Implementation: Hardy-Z bisection on the real-valued function
    ``Z_\chi(t)`` (see :func:`_hardy_z_chi`).  The function is scanned
    for sign changes on a uniform grid of step ``initial_step`` over
    ``t \in [t_{\min}, t_{\max}]``, then each sign change is refined
    by bracket bisection to ``bisection_tol``.

    For primitive real ``\chi`` all complex zeros are expected (by
    GRH) to lie on ``\operatorname{Re}(s) = 1/2``, so this enumerates
    the full set on the positive axis within the search window.

    Parameters
    ----------
    chi : DirichletCharacter
        Primitive real character.
    t_max : float
        Upper bound of zero search (must be > ``t_min``).
    t_min : float, default 0.5
        Lower bound (must be > 0; ``L(s, \chi)`` has no zero at
        ``s = 1/2``, and ``\theta(0)`` is finite, so a tiny offset
        keeps the bisection clean).
    initial_step : float, default 0.25
        Step for the sign-change scan.  Should be much smaller than
        the typical zero spacing.
    dps : int, default 30
        Mpmath working decimal precision.
    bisection_tol : float, default 1e-12
        Stop bisection when the bracket width is below this tolerance.

    Returns
    -------
    list[float]
        Ascending list of zero locations ``\gamma_n > 0`` in
        ``(t_{\min}, t_{\max})``.
    """
    if t_min <= 0:
        raise ValueError("t_min must be > 0")
    if t_max <= t_min:
        raise ValueError("t_max must be > t_min")
    if initial_step <= 0:
        raise ValueError("initial_step must be > 0")

    chi_list = _chi_to_mpmath_list(chi)
    q = int(chi.modulus)
    a = character_parity(chi)

    zeros: list[float] = []
    with mp.workdps(dps):
        n_steps = max(2, int(math.ceil((t_max - t_min) / initial_step)))
        ts = [
            mp.mpf(t_min) + i * (mp.mpf(t_max) - mp.mpf(t_min)) / n_steps
            for i in range(n_steps + 1)
        ]
        prev_t = ts[0]
        prev_z = _hardy_z_chi(prev_t, chi_list, q, a)
        for t in ts[1:]:
            z = _hardy_z_chi(t, chi_list, q, a)
            if prev_z == 0.0:
                zeros.append(float(prev_t))
            elif prev_z * z < 0:
                lo, hi = prev_t, t
                f_lo = prev_z
                while hi - lo > bisection_tol:
                    mid = (lo + hi) / 2
                    f_mid = _hardy_z_chi(mid, chi_list, q, a)
                    if f_mid == 0.0:
                        lo = hi = mid
                        break
                    if f_lo * f_mid < 0:
                        hi = mid
                    else:
                        lo, f_lo = mid, f_mid
                zeros.append(float((lo + hi) / 2))
            prev_t, prev_z = t, z

    return zeros


def twisted_weil_zero_side(
    chi: DirichletCharacter,
    test: GaussianTestFunction,
    *,
    t_max: float | None = None,
    t_min: float = 0.5,
    initial_step: float = 0.25,
    dps: int = 30,
) -> tuple[float, int, list[float]]:
    r"""Return ``\sum_\gamma h(\gamma)`` over zeros of ``L(s, \chi)``.

    For primitive **real** ``\chi`` the complex zeros come in conjugate
    pairs ``\rho = 1/2 \pm i\gamma``, so the sum equals
    ``2 \sum_{\gamma > 0} h(\gamma)`` and only the positive-axis zeros
    need to be enumerated.

    Parameters
    ----------
    chi : DirichletCharacter
        Primitive real character.
    test : GaussianTestFunction
        Test function (Gaussian envelope of width ``\sigma``).
    t_max : float, optional
        Upper bound for the zero search.  Defaults to
        ``12 * test.sigma`` so the Gaussian tail at ``t = t_{\max}``
        is below ``\exp(-72) \approx 5 \times 10^{-32}``.
    t_min : float, default 0.5
    initial_step : float, default 0.25
    dps : int, default 30

    Returns
    -------
    total : float
        ``2 \sum_{\gamma > 0} h(\gamma)`` (the full zero side).
    n_zeros_used : int
        Number of positive-axis zeros included.
    zeros : list[float]
        Ascending list of ``\gamma`` values.
    """
    if t_max is None:
        t_max = 12.0 * test.sigma
    zeros = find_dirichlet_l_zeros(
        chi,
        t_min=t_min,
        t_max=t_max,
        initial_step=initial_step,
        dps=dps,
    )
    total = 2.0 * sum(test.h(g) for g in zeros)
    return float(total), len(zeros), zeros


# ----------------------------------------------------------------------
# Certificate and high-level verification driver
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedWeilExplicitFormulaCertificate:
    """Outcome of :func:`verify_twisted_weil_explicit_formula`."""

    character_name: str
    character_modulus: int
    character_parity: int
    sigma: float
    n_zeros_used: int
    zeros: tuple[float, ...]
    zero_side: float
    constant_term: float
    archimedean_side: float
    prime_side: float
    rhs_total: float
    residual: float
    relative_residual: float
    tolerance: float
    verified: bool

    def summary(self) -> str:
        return (
            f"TwistedWeilExplicitFormulaCertificate("
            f"chi={self.character_name}, q={self.character_modulus}, "
            f"a={self.character_parity}, sigma={self.sigma:.4f}, "
            f"n_zeros={self.n_zeros_used}, "
            f"zero_side={self.zero_side:.10f}, "
            f"rhs={self.rhs_total:.10f}, "
            f"residual={self.residual:.3e}, "
            f"rel={self.relative_residual:.3e}, "
            f"verified={self.verified})"
        )


def verify_twisted_weil_explicit_formula(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    *,
    sigma: float = 2.0,
    t_min: float = 0.5,
    t_max: float | None = None,
    initial_step: float = 0.25,
    tolerance: float = 1e-2,
    integration_limit: float | None = None,
    dps: int = 30,
) -> TwistedWeilExplicitFormulaCertificate:
    r"""Verify Weil-Guinand for ``L(s, \chi)`` against the P34 bundle.

    The identity verified is

    .. math::

        \sum_\gamma h(\gamma)
        \;=\;
        g(0)\log(q/\pi)
        + \frac{1}{2\pi}\!\int h(t)\,\Re\psi(\tfrac14 + \tfrac{a}{2}
                                            + \tfrac{it}{2})\,dt
        + \bigl(-2\operatorname{Re}\!\sum_n \tfrac{\chi(n)\Lambda(n)}{\sqrt n}
            g(\log n)\bigr).

    The prime side is computed from the canonical TNFR P34 Hamiltonian
    in ``bundle``; the zero side is computed by Hardy-Z bisection of
    ``L(s, \chi)`` on the critical line.  The residual is the
    difference between the two sides; ``verified`` is set when
    ``|residual| < tolerance``.

    Parameters
    ----------
    chi : DirichletCharacter
        Primitive real character.  Must agree with the character used
        to build ``bundle`` (the routine checks the modulus).
    bundle : TwistedPrimeLadderHamiltonian
        P34 bundle built via
        :func:`tnfr.riemann.build_twisted_prime_ladder_hamiltonian`
        (with ``coupling = 0`` for the exact diagonal spectrum).
    sigma : float, default 2.0
        Gaussian test-function width.
    t_min : float, default 0.5
        Lower bound of zero search.
    t_max : float, optional
        Upper bound of zero search.  Defaults to ``12 * sigma``.
    initial_step : float, default 0.25
        Hardy-Z scan step (in ``t``); must be smaller than the typical
        zero spacing.
    tolerance : float, default 1e-2
        Verification tolerance applied to ``|residual|``.
    integration_limit : float, optional
        Truncation half-width of the archimedean integral.
    dps : int, default 30
        Mpmath working precision used for Hardy-Z evaluation.

    Returns
    -------
    TwistedWeilExplicitFormulaCertificate
    """
    if bundle.character_modulus != chi.modulus:
        raise ValueError(
            f"chi.modulus ({chi.modulus}) does not match "
            f"bundle.character_modulus ({bundle.character_modulus})."
        )

    test = gaussian_test_function(sigma)
    if t_max is None:
        t_max = 12.0 * sigma

    zero_total, n_used, zeros = twisted_weil_zero_side(
        chi,
        test,
        t_min=t_min,
        t_max=t_max,
        initial_step=initial_step,
        dps=dps,
    )
    const = twisted_weil_constant_term(chi, test)
    arch = twisted_weil_archimedean_integral(
        chi, test, integration_limit=integration_limit
    )
    prime = twisted_weil_prime_side_from_hamiltonian(bundle, test)
    rhs = const + arch + prime
    residual = zero_total - rhs
    denom = max(abs(zero_total), 1e-30)
    rel = abs(residual) / denom

    return TwistedWeilExplicitFormulaCertificate(
        character_name=str(chi.name),
        character_modulus=int(chi.modulus),
        character_parity=character_parity(chi),
        sigma=float(sigma),
        n_zeros_used=int(n_used),
        zeros=tuple(zeros),
        zero_side=float(zero_total),
        constant_term=float(const),
        archimedean_side=float(arch),
        prime_side=float(prime),
        rhs_total=float(rhs),
        residual=float(residual),
        relative_residual=float(rel),
        tolerance=float(tolerance),
        verified=bool(abs(residual) < tolerance),
    )
