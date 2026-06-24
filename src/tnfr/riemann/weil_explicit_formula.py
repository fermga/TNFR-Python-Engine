r"""TNFR-Riemann P15 — Weil/Guinand explicit formula.

This module implements a numerical verification of the classical
**Weil-Guinand explicit formula** linking the non-trivial zeros of
the Riemann zeta function to the spectrum of the canonical TNFR
prime-ladder Hamiltonian constructed in P14
(:mod:`tnfr.riemann.prime_ladder_hamiltonian`).

Mathematical statement
----------------------

Let ``h`` be a real, even Schwartz test function on the real line
with Fourier transform

.. math::

    g(u) \;=\; \frac{1}{2\pi}
        \int_{-\infty}^{\infty} h(t)\, e^{-itu}\, dt.

The Weil-Guinand explicit formula reads

.. math::

    \sum_{\gamma} h(\gamma)
       \;=\;
       h\!\bigl(i/2\bigr) + h\!\bigl(-i/2\bigr)
       \;-\; g(0)\,\log\pi
       \;+\; \frac{1}{2\pi}\!\int_{-\infty}^{\infty}\!\!
            h(t)\,\operatorname{Re}\psi\!\Bigl(\tfrac14 + \tfrac{it}{2}\Bigr)\, dt
       \;-\; 2 \sum_{n\ge 1} \frac{\Lambda(n)}{\sqrt{n}}\, g(\log n),

where the left-hand sum runs over the imaginary parts ``\gamma`` of
all non-trivial zeros ``\rho = 1/2 + i\gamma`` of ``\zeta(s)``,
``\psi`` is the digamma function and ``\Lambda`` is the von
Mangoldt function.

Connection to TNFR P14
----------------------

The prime-power sum on the right is **exactly** a spectral
functional on the P14 Hamiltonian
``H = \operatorname{diag}(k\log p)`` with weight operator
``W = \operatorname{diag}(\log p)``:

.. math::

    -2 \sum_{n\ge 1} \frac{\Lambda(n)}{\sqrt{n}}\, g(\log n)
       \;=\;
       -2 \operatorname{Tr}\!\bigl(\hat W\, e^{-\hat H / 2}\, g(\hat H)\bigr),

since ``n = p^k`` gives ``\Lambda(n) = \log p``, ``\sqrt n =
e^{(k\log p)/2}`` and ``g(\log n) = g(k\log p)``.  Each eigenvalue
``E_n = k\log p`` of the P14 Hamiltonian is a node ``|p,k\rangle``
with weight ``\log p``.  This is the canonical TNFR realisation of
the prime side of Weil's formula.

What this module proves and what it does not
--------------------------------------------

This module verifies the Weil-Guinand identity **numerically** with
a Gaussian test function family

.. math:: h_{\sigma}(t) = \exp\!\bigl(-t^2/(2\sigma^2)\bigr).

It does **not** prove the Riemann Hypothesis.  The explicit
formula is a *theorem* of analytic number theory (Riemann 1859,
Guinand 1948, Weil 1952) that holds **independently of the
location of the zeros**.  RH affects only the placement of the
``\gamma`` along the real line; the identity itself is unconditional.

What is *new* here is purely an instrumental result: the prime
side of Weil's formula is computable to machine precision from the
canonical TNFR P14 Hamiltonian, with no extra arithmetic
machinery.  This closes gap G3 of the TNFR-Riemann programme in
its operational sense — every term of Weil's bridge between
primes and zeros is exhibited inside the canonical TNFR formalism.
Gap G4 (localisation of zeros on Re(s)=1/2) is RH itself and
remains open.
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
    r"""Return ``h(i/2) + h(-i/2)`` (poles of ``\zeta`` at ``s=0,1``)."""
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
    r"""Compute the prime-power side of Weil's formula from P14.

    Returns

    .. math::

        -2 \operatorname{Tr}\!\bigl(\hat W e^{-\hat H/2} g(\hat H)\bigr)
        \;=\;
        -2 \sum_{(p,k)} \log(p)\, e^{-k\log(p)/2}\, g(k\log p),

    using the eigendecomposition of ``\hat H_{\text{int}}`` carried
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
    r"""Compute ``\sum_{\gamma > 0} 2 h(\gamma)`` over Riemann zeros.

    The sum is doubled because the trivial-zero-free zeros of
    ``\zeta`` come in conjugate pairs ``\rho = 1/2 \pm i\gamma``
    and ``h`` is real and even.

    Parameters
    ----------
    test
        Gaussian test function.
    n_zeros
        Initial number of zeros to use from :func:`mpmath.zetazero`.
    convergence_tol
        Stop adding zeros once the per-zero contribution falls
        below this threshold.
    max_zeros
        Hard cap to prevent runaway loops.

    Returns
    -------
    total
        The truncated sum.
    n_used
        Number of positive-axis zeros actually used.
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
    """Outcome of :func:`verify_weil_explicit_formula`."""

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
    r"""Verify Weil's explicit formula numerically against P14.

    The identity verified is

    .. math::

        \underbrace{\sum_{\gamma} h(\gamma)}_{\text{zero side}}
        \;=\;
        \underbrace{h(i/2) + h(-i/2)}_{\text{pole side}}
        \;-\; g(0)\log\pi
        \;+\; \underbrace{\tfrac{1}{2\pi}\!\int h(t)\,\Re\psi(\tfrac14+\tfrac{it}{2})dt}_{\text{archimedean side}}
        \;+\; \underbrace{\bigl(-2\sum_n \tfrac{\Lambda(n)}{\sqrt n}g(\log n)\bigr)}_{\text{prime side from P14}}.

    The residual is the difference of the two sides; ``verified``
    is set when ``|residual| < tolerance``.

    Notes
    -----
    The prime side is truncated by the finite size of ``bundle``
    (it contains only the first ``n_primes`` primes and echo depths
    ``k <= max_power``).  For ``sigma`` small enough that ``g`` has
    negligible support beyond the Hamiltonian's spectral support,
    the truncation error is dominated by the zero-side and
    archimedean-integral truncations rather than by the prime-side
    cutoff.
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
