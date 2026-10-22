r"""TNFR-Riemann P15 — Weil/Guinand explicit formula.

This module compares finite approximations to terms of the classical
**Weil-Guinand explicit formula** using known critical-line zeros and a
declared prime-ladder Hamiltonian constructed in P14
(:mod:`tnfr.riemann.prime_ladder_hamiltonian`).

Mathematical statement
----------------------

For the entire Gaussian test functions implemented below, use the Fourier
transform convention

.. math::

    g(u) \;=\; \frac{1}{2\pi}
        \int_{-\infty}^{\infty} h(t)\, e^{-itu}\, dt.

The Weil-Guinand explicit formula reads

.. math::

    \sum_{\rho} h\!\bigl((\rho-1/2)/i\bigr)
       \;=\;
       h\!\bigl(i/2\bigr) + h\!\bigl(-i/2\bigr)
       \;-\; g(0)\,\log\pi
       \;+\; \frac{1}{2\pi}\!\int_{-\infty}^{\infty}\!\!
            h(t)\,\operatorname{Re}\psi\!\Bigl(\tfrac14 + \tfrac{it}{2}\Bigr)\, dt
       \;-\; 2 \sum_{n\ge 1} \frac{\Lambda(n)}{\sqrt{n}}\, g(\log n),

where the left-hand sum runs over all non-trivial zeros ``rho`` of
``zeta`` with multiplicity, ``psi`` is the digamma function, and
``Lambda`` is the von Mangoldt function. The argument ``(rho-1/2)/i``
is complex for an off-line zero; replacing it with the imaginary part
of every zero would assume RH. Real-even Schwartz regularity alone
would not define the complex evaluations in this formula; the chosen
entire Gaussian supplies the required analytic extension and strip decay.

Connection to TNFR P14
----------------------

With the decoupled diagonal prime-ladder construction, the prime-power
sum is a spectral functional of
``H = \operatorname{diag}(k\log p)`` with weight operator
``W = \operatorname{diag}(\log p)``:

.. math::

    -2 \sum_{n\ge 1} \frac{\Lambda(n)}{\sqrt{n}}\, g(\log n)
       \;=\;
       -2 \operatorname{Tr}\!\bigl(\hat W\, e^{-\hat H / 2}\, g(\hat H)\bigr),

since ``n = p^k`` gives ``\Lambda(n) = \log p``, ``\sqrt n =
e^{(k\log p)/2}`` and ``g(\log n) = g(k\log p)``. Each assigned eigenvalue
``E_n = k\log p`` of the P14 Hamiltonian is a node ``|p,k\rangle``
with weight ``\log p``. The finite implementation truncates this identity
to the supplied nodes. An interacting Hamiltonian need not have that
prime-ladder spectrum; its weighted trace then describes the supplied
matrix rather than the classical prime side.

Finite comparison and its boundary
----------------------------------

This module computes a residual for the Gaussian family

.. math:: h_{\sigma}(t) = \exp\!\bigl(-t^2/(2\sigma^2)\bigr).

The classical formula with complex zero arguments is unconditional.
The implemented zero side instead sums a finite known critical-line list
from ``mpmath.zetazero``. It does not enumerate arbitrary off-line zeros,
certify omitted tails or establish that the list exhausts the analytic
zero set. The integral and matrix trace also use finite numerical data.

The compatibility field ``verified`` means only that the materialized
absolute residual is below the supplied tolerance. It is not a uniform
error bound, a proof of the analytic identity, a new Hilbert-Polya bridge
or a proof of RH. Prime labels and logarithmic weights are construction
inputs, not outputs of an autonomous nodal trajectory. Current scope is
centralized in ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mpmath
from scipy import integrate

from ..mathematics.unified_numerical import np
from .prime_ladder_hamiltonian import PrimeLadderHamiltonian

__all__ = [
    "GaussianTestFunction",
    "gaussian_test_function",
    "weil_pole_side",
    "weil_archimedean_integral",
    "weil_prime_side_from_hamiltonian",
    "weil_zero_side",
    "WeilExplicitFormulaCertificate",
    "verify_weil_explicit_formula",
]


# ----------------------------------------------------------------------
# Gaussian test family
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianTestFunction:
    r"""Gaussian test function ``h(t) = exp(-t^2 / (2 sigma^2))``.

    The Fourier transform under the convention

    .. math::

        g(u) = \frac{1}{2\pi}\int h(t)\,e^{-itu}\,dt

    is also Gaussian:

    .. math:: g(u) = \frac{\sigma}{\sqrt{2\pi}}
                     \exp\!\bigl(-\sigma^2 u^2 / 2\bigr).
    """

    sigma: float

    def h(self, t: float) -> float:
        """Evaluate ``h(t)`` on the real line."""
        return math.exp(-(t**2) / (2.0 * self.sigma**2))

    def h_complex(self, t: complex) -> complex:
        """Evaluate ``h(t)`` for complex argument (analytic extension)."""
        return complex(np.exp(-(t**2) / (2.0 * self.sigma**2)))

    def g(self, u: float) -> float:
        """Evaluate the Fourier transform ``g(u)`` on the real line."""
        sigma = self.sigma
        return (sigma / math.sqrt(2.0 * math.pi)) * math.exp(-(sigma**2) * u**2 / 2.0)

    def g_zero(self) -> float:
        """Return ``g(0) = sigma / sqrt(2 pi)``."""
        return self.sigma / math.sqrt(2.0 * math.pi)

    def h_at_half_pole(self) -> float:
        r"""Return ``h(i/2) = h(-i/2) = exp(1 / (8 sigma^2))``."""
        return math.exp(1.0 / (8.0 * self.sigma**2))


def gaussian_test_function(sigma: float) -> GaussianTestFunction:
    """Construct a :class:`GaussianTestFunction` of width ``sigma``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive")
    return GaussianTestFunction(sigma=float(sigma))


# ----------------------------------------------------------------------
# Individual terms of the Weil formula
# ----------------------------------------------------------------------


def weil_pole_side(test: GaussianTestFunction) -> float:
    r"""Return the explicit-formula term ``h(i/2) + h(-i/2)``.

    These terms involve completion points 0 and 1; zeta itself has its
    simple pole only at 1, not at 0.
    """
    return 2.0 * test.h_at_half_pole()


def _digamma_real_part(t: float) -> float:
    """Return ``Re psi(1/4 + i t / 2)`` via mpmath."""
    val = mpmath.digamma(mpmath.mpc(0.25, t / 2.0))
    return float(val.real)


def weil_archimedean_integral(
    test: GaussianTestFunction,
    *,
    integration_limit: float | None = None,
    quad_kwargs: dict | None = None,
) -> float:
    r"""Compute the archimedean integral

    .. math::

        \frac{1}{2\pi}\int_{-\infty}^{\infty}
            h(t)\,\operatorname{Re}\psi\!\Bigl(\tfrac14 + \tfrac{it}{2}\Bigr)\, dt.

    The integral is evaluated by :func:`scipy.integrate.quad` after
    truncating the domain to ``[-L, L]`` with ``L`` chosen so that
    the Gaussian envelope is negligible.
    """
    if integration_limit is None:
        # h(t) decays as exp(-t^2 / (2 sigma^2)); choose L = 10 sigma
        integration_limit = 10.0 * test.sigma
    kw = {"limit": 200, "epsabs": 1e-14, "epsrel": 1e-12}
    if quad_kwargs:
        kw.update(quad_kwargs)

    def integrand(t: float) -> float:
        return test.h(t) * _digamma_real_part(t)

    val, _err = integrate.quad(integrand, -integration_limit, integration_limit, **kw)
    return float(val / (2.0 * math.pi))


def weil_prime_side_from_hamiltonian(
    bundle: PrimeLadderHamiltonian,
    test: GaussianTestFunction,
) -> float:
    r"""Compute the weighted trace for the supplied P14 matrix.

    Returns

    .. math::

        -2 \operatorname{Tr}\!\bigl(\hat W e^{-\hat H/2} g(\hat H)\bigr)
        \;=\;
        -2 \sum_{(p,k)} \log(p)\, e^{-k\log(p)/2}\, g(k\log p),

    The prime-power sum equality requires the decoupled diagonal spectrum.
    The actual trace uses the eigendecomposition of ``\hat H_{\text{int}}`` carried
    by ``bundle.hamiltonian``.  At ``J_0 = 0`` the Hamiltonian is
    diagonal and the trace collapses to a simple weighted sum over
    nodes ``(p, k)``.
    """
    eigvals, eigvecs = bundle.hamiltonian.get_spectrum()
    eigvals_real = np.real(eigvals)
    g_values = np.array([test.g(float(e)) for e in eigvals_real], dtype=float)
    half_decay = np.exp(-eigvals_real / 2.0)
    # Diagonal weight operator in the node basis
    weights_diag = np.real(np.diag(bundle.weight_operator))
    # Transform weight operator to eigenbasis: <e_i|W|e_i>
    # = sum_n |<n|e_i>|^2 W_nn
    W_diag_eig = np.einsum("ni,n,ni->i", np.conj(eigvecs), weights_diag, eigvecs).real
    contributions = W_diag_eig * half_decay * g_values
    return float(-2.0 * np.sum(contributions))


def weil_zero_side(
    test: GaussianTestFunction,
    *,
    n_zeros: int = 50,
    convergence_tol: float = 1e-12,
    max_zeros: int = 500,
) -> tuple[float, int]:
    r"""Compute a finite ``\sum_{\gamma > 0} 2 h(\gamma)`` from known line zeros.

    Each supplied critical-line zero has conjugate ``1/2 - i*gamma``;
    the real-even Gaussian gives the doubled contribution. This routine
    does not inspect possible off-line zeros or certify the omitted tail.

    Parameters
    ----------
    test
        Gaussian test function.
    n_zeros
        Initial number of zeros to use from :func:`mpmath.zetazero`.
    convergence_tol
        Stop adding zeros once the per-zero contribution falls
        below this threshold. A small last term is not a bound on the
        sum of all remaining terms.
    max_zeros
        Hard cap to prevent runaway loops.

    Returns
    -------
    total
        The truncated sum.
    n_used
        Number of upper-half-plane critical-line zeros actually used.
    """
    total = 0.0
    n_used = 0
    for n in range(1, max_zeros + 1):
        gamma_n = float(mpmath.zetazero(n).imag)
        contribution = 2.0 * test.h(gamma_n)
        total += contribution
        n_used = n
        if n >= n_zeros and contribution < convergence_tol:
            break
    return float(total), n_used


# ----------------------------------------------------------------------
# Certificate and high-level verification driver
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class WeilExplicitFormulaCertificate:
    """Finite residual record; ``verified`` is a supplied-tolerance test only."""

    sigma: float
    n_zeros_used: int
    zero_side: float
    pole_side: float
    archimedean_side: float
    prime_side: float
    rhs_total: float
    residual: float
    relative_residual: float
    tolerance: float
    verified: bool

    def summary(self) -> str:
        return (
            f"WeilExplicitFormulaCertificate("
            f"sigma={self.sigma:.4f}, "
            f"n_zeros={self.n_zeros_used}, "
            f"zero_side={self.zero_side:.10f}, "
            f"rhs={self.rhs_total:.10f}, "
            f"residual={self.residual:.3e}, "
            f"rel={self.relative_residual:.3e}, "
            f"verified={self.verified})"
        )


def verify_weil_explicit_formula(
    bundle: PrimeLadderHamiltonian,
    *,
    sigma: float = 2.0,
    n_zeros: int = 80,
    convergence_tol: float = 1e-12,
    max_zeros: int = 500,
    tolerance: float = 1e-3,
    integration_limit: float | None = None,
) -> WeilExplicitFormulaCertificate:
    r"""Compare finite zero-side and weighted-trace approximations.

    The classical comparison uses the identity

    .. math::

        \underbrace{\sum_{\rho} h((\rho-1/2)/i)}_{\text{zero side}}
        \;=\;
        \underbrace{h(i/2) + h(-i/2)}_{\text{pole side}}
        \;-\; g(0)\log\pi
        \;+\; \underbrace{\tfrac{1}{2\pi}\!\int h(t)\,\Re\psi(\tfrac14+\tfrac{it}{2})dt}_{\text{archimedean side}}
        \;+\; \underbrace{\bigl(-2\sum_n \tfrac{\Lambda(n)}{\sqrt n}g(\log n)\bigr)}_{\text{prime side from P14}}.

    The implementation uses a finite known critical-line list on the
    left and a finite matrix on the right. The classical prime-sum
    interpretation requires a decoupled diagonal prime ladder.
    ``verified`` is set when the materialized absolute residual is below
    ``tolerance``; no full-sum or rounding enclosure is returned.

    Notes
    -----
    The prime side is truncated by the finite size of ``bundle``
    (it contains only the first ``n_primes`` primes and ladder depths
    ``k <= max_power``). The Fourier Gaussian has width proportional
    to ``1/sigma``: decreasing ``sigma`` broadens the prime-side window.
    No uniform claim about which truncation dominates follows without
    bounding all omitted zero, prime and integral contributions.
    """
    test = gaussian_test_function(sigma)
    zero_total, n_used = weil_zero_side(
        test,
        n_zeros=n_zeros,
        convergence_tol=convergence_tol,
        max_zeros=max_zeros,
    )
    pole = weil_pole_side(test)
    arch = weil_archimedean_integral(test, integration_limit=integration_limit)
    prime = weil_prime_side_from_hamiltonian(bundle, test)
    log_pi_term = -test.g_zero() * math.log(math.pi)
    rhs = pole + log_pi_term + arch + prime
    residual = zero_total - rhs
    denom = max(abs(zero_total), 1e-30)
    rel = abs(residual) / denom
    return WeilExplicitFormulaCertificate(
        sigma=float(sigma),
        n_zeros_used=int(n_used),
        zero_side=float(zero_total),
        pole_side=float(pole + log_pi_term),
        archimedean_side=float(arch),
        prime_side=float(prime),
        rhs_total=float(rhs),
        residual=float(residual),
        relative_residual=float(rel),
        tolerance=float(tolerance),
        verified=bool(abs(residual) < tolerance),
    )
