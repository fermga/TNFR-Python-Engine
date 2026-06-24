r"""P17: Weil positivity bridge between the RH-equivalent Weil functional
and the canonical TNFR Lyapunov energy.

Mathematical background
-----------------------

For a real, even Schwartz test function :math:`f` with positive
Fourier-Plancherel image :math:`h(t) := |\hat f(t)|^2`, the
**Weil positivity criterion** (Weil 1952, Bombieri 2000) states

.. math::

    \text{RH}\ \Longleftrightarrow\
    W[f]\;:=\;\sum_{\gamma}\, h(\gamma)\ \ge\ 0
    \quad\text{for every admissible } f,

where the sum ranges over the imaginary parts :math:`\gamma` of the
non-trivial zeros :math:`\rho = 1/2 + i\gamma` of :math:`\zeta(s)`.
(If some :math:`\gamma` were complex, the Gaussian-type :math:`h` would
take complex values whose real part can be negative — so positivity of
:math:`W[f]` over a dense admissible family is strictly equivalent to
RH.)

This module performs two operations:

1. **Weil-positivity certificate**.  For the Gaussian admissible family
   :math:`h_\sigma(t) = \exp(-t^2/(2\sigma^2))` (this is :math:`|\hat
   f_\sigma|^2` with :math:`f_\sigma` Gaussian), it computes
   :math:`W[\sigma]` two ways — directly from the zero side and via the
   Weil-Guinand explicit formula evaluated with the P14 prime-ladder
   Hamiltonian — and reports whether :math:`W[\sigma] \ge 0`.  This is
   the RH-equivalent diagnostic, in pure TNFR form.

2. **TNFR-Lyapunov bridge**.  It defines a *canonical structural test
   state* on the P14 prime-ladder graph driven by the same Gaussian
   profile :math:`h_\sigma`, computes the canonical TNFR Lyapunov
   energy :math:`E_{\mathrm{TNFR}}[\sigma]` via
   :func:`tnfr.physics.conservation.compute_energy_functional`, and
   tabulates the ratio :math:`\alpha(\sigma) := W[\sigma]\,/\,
   E_{\mathrm{TNFR}}[\sigma]` across a grid of widths.  If
   :math:`\alpha(\sigma) > 0` uniformly across an admissible family,
   the inequality :math:`W[\sigma] \ge \alpha \cdot E_{\mathrm{TNFR}}
   [\sigma]` constitutes a TNFR-native lower-bound witness for the
   Weil positivity functional.

Honesty disclaimer
------------------
This module **does not prove** the Riemann Hypothesis.  Weil positivity
is RH-equivalent in the limit of a dense admissible family; this module
checks it numerically on a Gaussian grid.  The structural test state
defined here is one *natural* TNFR mapping of :math:`h_\sigma` to the
P14 graph, not the unique one — different mappings yield different
:math:`E_{\mathrm{TNFR}}[\sigma]`.  The bridge certificate reports
:math:`\alpha(\sigma)` for *this* mapping and serves as a structural
diagnostic, not as a theorem of analytic number theory.

Status: EXPERIMENTAL — Research prototype for TNFR-Riemann P17 program.
The deliverable closes the operational distance between the canonical
TNFR Lyapunov positivity (Structural Conservation Theorem) and the
RH-equivalent Weil positivity functional; it does *not* close gap G4
(RH itself), which would require promoting the numerical inequality
:math:`W[\sigma] \ge \alpha \cdot E_{\mathrm{TNFR}}[\sigma]` to a
theorem over a dense admissible class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..physics.conservation import compute_energy_functional
from .prime_ladder_hamiltonian import PrimeLadderHamiltonian
from .weil_explicit_formula import (
    gaussian_test_function,
    weil_archimedean_integral,
    weil_pole_side,
    weil_prime_side_from_hamiltonian,
    weil_zero_side,
)

__all__ = [
    "WeilPositivityCertificate",
    "WeilTNFRBridgeCertificate",
    "build_structural_test_state",
    "tnfr_lyapunov_of_test_state",
    "verify_weil_positivity",
    "verify_weil_tnfr_bridge",
]


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeilPositivityCertificate:
    r"""Outcome of :func:`verify_weil_positivity` for a single :math:`\sigma`.

    Attributes
    ----------
    sigma
        Width of the Gaussian admissible test function
        :math:`h_\sigma(t) = \exp(-t^2 / (2\sigma^2))`.
    weil_functional_zero_side
        :math:`W[\sigma] = \sum_{\gamma} h_\sigma(\gamma)` from the
        classical zero side (sum over Riemann zeros).
    weil_functional_explicit_formula
        Same quantity computed via the Weil-Guinand explicit formula
        using the P14 Hamiltonian for the prime side.
    explicit_formula_residual
        Difference between the two computations; a small residual
        confirms self-consistency with P15.
    n_zeros_used
        Number of positive-axis zeros used for the zero-side sum.
    positive
        Boolean: ``True`` iff :math:`W[\sigma] \ge 0` (Weil positivity
        satisfied for this test function).
    """

    sigma: float
    weil_functional_zero_side: float
    weil_functional_explicit_formula: float
    explicit_formula_residual: float
    n_zeros_used: int
    positive: bool

    def summary(self) -> str:
        return (
            "WeilPositivityCertificate("
            f"sigma={self.sigma:.4f}, "
            f"W_zero={self.weil_functional_zero_side:.6e}, "
            f"W_xf={self.weil_functional_explicit_formula:.6e}, "
            f"residual={self.explicit_formula_residual:.2e}, "
            f"n_zeros={self.n_zeros_used}, "
            f"positive={self.positive})"
        )


@dataclass(frozen=True)
class WeilTNFRBridgeCertificate:
    r"""Outcome of :func:`verify_weil_tnfr_bridge` across a :math:`\sigma` grid.

    Attributes
    ----------
    sigmas
        Grid of widths evaluated.
    weil_functional
        :math:`W[\sigma]` per width (zero-side computation).
    tnfr_lyapunov_energy
        :math:`E_{\mathrm{TNFR}}[\sigma]` per width.
    alpha
        :math:`\alpha(\sigma) = W[\sigma] / E_{\mathrm{TNFR}}[\sigma]`
        per width (``inf`` when energy is zero).
    weil_positive
        Boolean per width: ``W[\sigma] \ge 0``.
    bridge_positive
        Boolean per width: ``alpha(\sigma) > 0``.
    weil_positive_all
        Aggregate: ``all(weil_positive)``.
    bridge_positive_all
        Aggregate: ``all(bridge_positive)``.
    alpha_min
        Minimum :math:`\alpha` across the grid (lower bound candidate).
    alpha_max
        Maximum :math:`\alpha` across the grid (upper bound).
    """

    sigmas: np.ndarray
    weil_functional: np.ndarray
    tnfr_lyapunov_energy: np.ndarray
    alpha: np.ndarray
    weil_positive: np.ndarray
    bridge_positive: np.ndarray
    weil_positive_all: bool
    bridge_positive_all: bool
    alpha_min: float
    alpha_max: float

    def summary(self) -> str:
        return (
            "WeilTNFRBridgeCertificate("
            f"n_sigma={len(self.sigmas)}, "
            f"sigma_range=[{float(self.sigmas[0]):.3f}, "
            f"{float(self.sigmas[-1]):.3f}], "
            f"W_all_positive={self.weil_positive_all}, "
            f"alpha_all_positive={self.bridge_positive_all}, "
            f"alpha_min={self.alpha_min:.4e}, "
            f"alpha_max={self.alpha_max:.4e})"
        )


# ---------------------------------------------------------------------------
# Structural test state on the P14 prime-ladder graph
# ---------------------------------------------------------------------------


def build_structural_test_state(
    bundle: PrimeLadderHamiltonian,
    sigma: float,
) -> nx.Graph:
    r"""Map the Gaussian test profile :math:`h_\sigma` onto the P14 graph.

    For each prime-ladder node :math:`(p, k)` with structural energy
    :math:`E_n = k \log p`, the following structural attributes are
    written on a *copy* of ``bundle.graph``:

    * ``dnfr_(p,k) = h_\sigma(E_n) = exp(-E_n^2 / (2 sigma^2))``
      — the test profile becomes the local reorganisation pressure.
    * ``phase_(p,k) = wrap(h_\sigma(E_n))``
      — same profile drives a small phase gradient along each ladder,
      so that :math:`|\nabla\phi|` and :math:`K_\phi` are non-zero.
    * ``EPI_(p,k) = h_\sigma(E_n)``
      — primary information amplitude tracks the test profile.
    * ``nu_f_(p,k) = E_n``
      — inherited unchanged from the P14 construction.

    Rationale
    ---------
    This is a *canonical* (but not unique) TNFR realisation of the test
    function :math:`h_\sigma`.  The ``dnfr`` channel feeds
    :math:`\Phi_s = \sum_j \mathrm{dnfr}_j / d(i,j)^2`; the ``phase``
    channel feeds :math:`|\nabla\phi|` and :math:`K_\phi`; the result
    is a structural state in which every component of the tetrad
    responds to :math:`h_\sigma`.  Different mappings (e.g. encoding
    :math:`h_\sigma` only in ``phase`` or only in ``dnfr``) would
    activate different sectors of the Lyapunov functional and yield
    different :math:`E_{\mathrm{TNFR}}[\sigma]`.

    Parameters
    ----------
    bundle
        Prime-ladder Hamiltonian bundle from P14.
    sigma
        Width of the Gaussian test function (must be positive).

    Returns
    -------
    networkx.Graph
        A copy of ``bundle.graph`` with structural attributes overwritten.

    Raises
    ------
    ValueError
        If ``sigma <= 0``.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be strictly positive")

    G = bundle.graph.copy()
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)

    for node in G.nodes():
        p, k = node
        E_n = float(k) * math.log(float(p))
        h_val = math.exp(-(E_n * E_n) * inv_two_sigma_sq)
        # Wrap into [-pi, pi] (h_val is in (0, 1] so always in-range)
        phase = h_val if h_val <= math.pi else math.pi
        G.nodes[node]["dnfr"] = h_val
        G.nodes[node]["phase"] = phase
        G.nodes[node]["EPI"] = h_val
        # nu_f stays as set by build_prime_ladder_graph (k * log p)
    return G


def tnfr_lyapunov_of_test_state(
    bundle: PrimeLadderHamiltonian,
    sigma: float,
) -> float:
    r"""Compute the canonical TNFR Lyapunov energy
    :math:`E_{\mathrm{TNFR}}[\sigma]` for the structural test state.

    Equivalent to::

        G = build_structural_test_state(bundle, sigma)
        return compute_energy_functional(G)

    The Lyapunov energy is the canonical structural functional

    .. math::

        E[G] \;=\; \tfrac12\sum_i\bigl[\Phi_s^2(i) + |\nabla\phi|^2(i)
                  + K_\phi^2(i) + J_\phi^2(i) + J_{\Delta\!NFR}^2(i)\bigr],

    guaranteed non-negative by construction.  Under grammar-compliant
    evolution (U1-U6) the time derivative is non-positive (Structural
    Conservation Theorem, see :mod:`tnfr.physics.conservation`).
    """
    G = build_structural_test_state(bundle, sigma)
    return compute_energy_functional(G)


# ---------------------------------------------------------------------------
# Verification drivers
# ---------------------------------------------------------------------------


def verify_weil_positivity(
    bundle: PrimeLadderHamiltonian,
    *,
    sigma: float = 2.0,
    n_zeros: int = 80,
    convergence_tol: float = 1e-12,
    max_zeros: int = 500,
    integration_limit: float | None = None,
) -> WeilPositivityCertificate:
    r"""Verify the Weil positivity functional :math:`W[\sigma] \ge 0`.

    Computes :math:`W[\sigma] := \sum_\gamma h_\sigma(\gamma)` two ways:

    * **Zero side**: sum over Riemann zeros via :mod:`mpmath` (reuses
      :func:`weil_zero_side` from P15).
    * **Explicit formula side**: pole + archimedean + prime sides per
      the Weil-Guinand identity, with the prime side computed from the
      P14 Hamiltonian via :func:`weil_prime_side_from_hamiltonian`.

    The two computations are exposed independently; their difference
    is the consistency residual (which should be small if P15 is
    consistent for this :math:`\sigma`).

    Parameters
    ----------
    bundle
        Prime-ladder Hamiltonian bundle from P14.
    sigma
        Width of the Gaussian test function.
    n_zeros, convergence_tol, max_zeros
        Forwarded to :func:`weil_zero_side`.
    integration_limit
        Forwarded to :func:`weil_archimedean_integral`.

    Returns
    -------
    WeilPositivityCertificate
        Frozen result with both computations, residual, and positivity
        flag.
    """
    test = gaussian_test_function(sigma)

    zero_side, n_used = weil_zero_side(
        test,
        n_zeros=n_zeros,
        convergence_tol=convergence_tol,
        max_zeros=max_zeros,
    )

    pole_side = weil_pole_side(test)
    arch_side = weil_archimedean_integral(test, integration_limit=integration_limit)
    prime_side = weil_prime_side_from_hamiltonian(bundle, test)
    g_zero = test.g_zero()
    explicit_formula_rhs = (
        pole_side - g_zero * math.log(math.pi) + arch_side + prime_side
    )

    residual = float(abs(zero_side - explicit_formula_rhs))
    positive = bool(zero_side >= 0.0)

    return WeilPositivityCertificate(
        sigma=float(sigma),
        weil_functional_zero_side=float(zero_side),
        weil_functional_explicit_formula=float(explicit_formula_rhs),
        explicit_formula_residual=residual,
        n_zeros_used=int(n_used),
        positive=positive,
    )


def verify_weil_tnfr_bridge(
    bundle: PrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    n_zeros: int = 80,
    convergence_tol: float = 1e-12,
    max_zeros: int = 500,
) -> WeilTNFRBridgeCertificate:
    r"""Tabulate the TNFR-Weil positivity bridge across a :math:`\sigma` grid.

    For each :math:`\sigma` in ``sigmas``, computes

    * :math:`W[\sigma]` via :func:`verify_weil_positivity` (zero side),
    * :math:`E_{\mathrm{TNFR}}[\sigma]` via
      :func:`tnfr_lyapunov_of_test_state`,
    * :math:`\alpha(\sigma) = W[\sigma] / E_{\mathrm{TNFR}}[\sigma]`.

    A constant positive lower bound :math:`\alpha_{\min} > 0` across a
    dense admissible family would constitute a TNFR-native witness for
    the Weil positivity inequality (and hence, via Weil's equivalence,
    for the Riemann Hypothesis).  This module checks the inequality
    numerically; it does not prove uniform positivity.

    Parameters
    ----------
    bundle
        Prime-ladder Hamiltonian bundle from P14.
    sigmas
        Grid of Gaussian widths to evaluate.
    n_zeros, convergence_tol, max_zeros
        Forwarded to :func:`verify_weil_positivity` for the zero-side
        sum.

    Returns
    -------
    WeilTNFRBridgeCertificate
        Frozen result with per-:math:`\sigma` arrays and aggregate
        positivity flags.

    Raises
    ------
    ValueError
        If ``sigmas`` is empty or contains non-positive values.
    """
    sigma_array = np.array(list(sigmas), dtype=float)
    if sigma_array.size == 0:
        raise ValueError("sigmas must be non-empty")
    if np.any(sigma_array <= 0.0):
        raise ValueError("all sigmas must be strictly positive")

    n = sigma_array.size
    W_vals = np.zeros(n, dtype=float)
    E_vals = np.zeros(n, dtype=float)
    alpha_vals = np.zeros(n, dtype=float)
    weil_pos = np.zeros(n, dtype=bool)
    bridge_pos = np.zeros(n, dtype=bool)

    for i, sigma in enumerate(sigma_array):
        sigma_f = float(sigma)
        cert = verify_weil_positivity(
            bundle,
            sigma=sigma_f,
            n_zeros=n_zeros,
            convergence_tol=convergence_tol,
            max_zeros=max_zeros,
        )
        W = float(cert.weil_functional_zero_side)
        E = float(tnfr_lyapunov_of_test_state(bundle, sigma_f))

        W_vals[i] = W
        E_vals[i] = E
        if E > 0.0:
            alpha_vals[i] = W / E
        else:
            alpha_vals[i] = float("inf") if W > 0.0 else 0.0
        weil_pos[i] = bool(cert.positive)
        bridge_pos[i] = bool(alpha_vals[i] > 0.0)

    finite_alpha = alpha_vals[np.isfinite(alpha_vals)]
    if finite_alpha.size > 0:
        alpha_min = float(finite_alpha.min())
        alpha_max = float(finite_alpha.max())
    else:
        alpha_min = float("nan")
        alpha_max = float("nan")

    return WeilTNFRBridgeCertificate(
        sigmas=sigma_array,
        weil_functional=W_vals,
        tnfr_lyapunov_energy=E_vals,
        alpha=alpha_vals,
        weil_positive=weil_pos,
        bridge_positive=bridge_pos,
        weil_positive_all=bool(weil_pos.all()),
        bridge_positive_all=bool(bridge_pos.all()),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )
