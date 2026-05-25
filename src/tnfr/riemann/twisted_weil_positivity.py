r"""P37: chi-twisted Weil-TNFR positivity bridge for primitive real
Dirichlet L-functions.

Structural analogue of P17 (Weil-TNFR positivity bridge for the Riemann
zeta) extended to primitive real Dirichlet L-functions L(s, chi) via
the P34 chi-twisted prime-ladder Hamiltonian and the P35 chi-twisted
Weil-Guinand explicit formula.

Mathematical background
-----------------------

For a primitive real Dirichlet character chi (modulus q, parity
a = (1 - chi(-1))/2) and a real even Schwartz test function f with
positive Fourier-Plancherel image h(t) := |f-hat(t)|^2, the **Weil
positivity criterion for L(s, chi)** (Bombieri 2000, generalising
Weil 1952) states

    GRH_chi  <==>  W_chi[f] := sum_gamma h(gamma) >= 0
                   for every admissible f,

where the sum ranges over the imaginary parts gamma of the non-trivial
zeros rho = 1/2 + i gamma of L(s, chi).

This module performs two operations, mirroring P17 for the chi-twisted
setting:

1. **chi-twisted Weil positivity certificate**.  For the Gaussian
   admissible family h_sigma(t) = exp(-t^2/(2 sigma^2)), it computes
   W_chi[sigma] two ways -- directly from the zero side via the P35
   Hardy-Z bisection enumerator (`twisted_weil_zero_side`) and via the
   chi-twisted Weil-Guinand explicit formula evaluated with the P34
   chi-twisted prime-ladder Hamiltonian -- and reports whether
   W_chi[sigma] >= 0.  This is the GRH_chi-equivalent diagnostic, in
   pure TNFR form.

2. **chi-twisted TNFR-Lyapunov bridge**.  It defines a *canonical
   structural test state* on the P34 chi-twisted prime-ladder graph
   driven by the same Gaussian profile h_sigma, computes the canonical
   TNFR Lyapunov energy E_TNFR_chi[sigma] via
   `compute_energy_functional`, and tabulates the ratio
   alpha_chi(sigma) := W_chi[sigma] / E_TNFR_chi[sigma] across a grid
   of widths.  If alpha_chi(sigma) > 0 uniformly across an admissible
   family, the inequality W_chi[sigma] >= alpha_chi * E_TNFR_chi[sigma]
   constitutes a TNFR-native lower-bound witness for the chi-twisted
   Weil positivity functional.

Honesty disclaimer
------------------
This module **does not prove** the Generalised Riemann Hypothesis for
any L(s, chi).  Weil positivity is GRH-equivalent in the limit of a
dense admissible family; this module checks it numerically on a
Gaussian grid.  The structural test state defined here is one
*canonical* TNFR mapping of h_sigma to the P34 graph, not the unique
one.  The bridge certificate reports alpha_chi(sigma) for *this*
mapping and serves as a structural diagnostic, not as a theorem of
analytic number theory.  In particular this module does NOT advance
G4 = RH (the localisation of zeros of zeta on Re(s) = 1/2) or the
arithmetic obstruction of GRH.

Status: EXPERIMENTAL -- Research prototype for TNFR-Riemann P37 program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..physics.conservation import compute_energy_functional
from .dirichlet_l import DirichletCharacter
from .twisted_prime_ladder_hamiltonian import TwistedPrimeLadderHamiltonian
from .twisted_weil_explicit_formula import (
    character_parity,
    twisted_weil_archimedean_integral,
    twisted_weil_constant_term,
    twisted_weil_prime_side_from_hamiltonian,
    twisted_weil_zero_side,
)
from .weil_explicit_formula import gaussian_test_function

__all__ = [
    "TwistedWeilPositivityCertificate",
    "TwistedWeilTNFRBridgeCertificate",
    "build_twisted_structural_test_state",
    "twisted_tnfr_lyapunov_of_test_state",
    "verify_twisted_weil_positivity",
    "verify_twisted_weil_tnfr_bridge",
]


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedWeilPositivityCertificate:
    r"""Outcome of `verify_twisted_weil_positivity` for a single sigma.

    Attributes
    ----------
    character_name
        Label of the primitive real character chi.
    character_modulus
        Conductor q of chi.
    character_parity
        ``a = (1 - chi(-1))/2`` in {0, 1}.
    sigma
        Width of the Gaussian admissible test function
        h_sigma(t) = exp(-t^2 / (2 sigma^2)).
    weil_functional_zero_side
        W_chi[sigma] = 2 sum_{gamma > 0} h_sigma(gamma) from the
        Hardy-Z bisection enumeration of zeros of L(s, chi).
    weil_functional_explicit_formula
        Same quantity computed via the chi-twisted Weil-Guinand
        explicit formula using the P34 Hamiltonian for the prime side.
    explicit_formula_residual
        Difference between the two computations; a small residual
        confirms self-consistency with P35.
    n_zeros_used
        Number of positive-axis zeros included in the zero-side sum.
    positive
        Boolean: ``True`` iff W_chi[sigma] >= 0 (twisted Weil
        positivity satisfied for this test function).
    """

    character_name: str
    character_modulus: int
    character_parity: int
    sigma: float
    weil_functional_zero_side: float
    weil_functional_explicit_formula: float
    explicit_formula_residual: float
    n_zeros_used: int
    positive: bool

    def summary(self) -> str:
        return (
            "TwistedWeilPositivityCertificate("
            f"chi={self.character_name}, q={self.character_modulus}, "
            f"a={self.character_parity}, sigma={self.sigma:.4f}, "
            f"W_zero={self.weil_functional_zero_side:.6e}, "
            f"W_xf={self.weil_functional_explicit_formula:.6e}, "
            f"residual={self.explicit_formula_residual:.2e}, "
            f"n_zeros={self.n_zeros_used}, "
            f"positive={self.positive})"
        )


@dataclass(frozen=True)
class TwistedWeilTNFRBridgeCertificate:
    r"""Outcome of `verify_twisted_weil_tnfr_bridge` across a sigma grid.

    Attributes
    ----------
    character_name, character_modulus, character_parity
        Identifiers of the primitive real character chi.
    sigmas
        Grid of widths evaluated.
    weil_functional
        W_chi[sigma] per width (zero-side computation).
    tnfr_lyapunov_energy
        E_TNFR_chi[sigma] per width.
    alpha
        alpha_chi(sigma) = W_chi[sigma] / E_TNFR_chi[sigma] per width
        (``inf`` when energy is zero).
    weil_positive
        Boolean per width: W_chi[sigma] >= 0.
    bridge_positive
        Boolean per width: alpha_chi(sigma) > 0.
    weil_positive_all
        Aggregate: ``all(weil_positive)``.
    bridge_positive_all
        Aggregate: ``all(bridge_positive)``.
    alpha_min
        Minimum alpha_chi across the grid (lower bound candidate).
    alpha_max
        Maximum alpha_chi across the grid (upper bound).
    """

    character_name: str
    character_modulus: int
    character_parity: int
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
            "TwistedWeilTNFRBridgeCertificate("
            f"chi={self.character_name}, q={self.character_modulus}, "
            f"a={self.character_parity}, "
            f"n_sigma={len(self.sigmas)}, "
            f"sigma_range=[{float(self.sigmas[0]):.3f}, "
            f"{float(self.sigmas[-1]):.3f}], "
            f"W_all_positive={self.weil_positive_all}, "
            f"alpha_all_positive={self.bridge_positive_all}, "
            f"alpha_min={self.alpha_min:.4e}, "
            f"alpha_max={self.alpha_max:.4e})"
        )


# ---------------------------------------------------------------------------
# Structural test state on the P34 chi-twisted prime-ladder graph
# ---------------------------------------------------------------------------


def build_twisted_structural_test_state(
    bundle: TwistedPrimeLadderHamiltonian,
    sigma: float,
) -> nx.Graph:
    r"""Map the Gaussian test profile h_sigma onto the P34 graph.

    For each chi-twisted prime-ladder node (p, k) (with chi(p) != 0,
    so p does not divide q) and structural energy E_n = k log(p), the
    following structural attributes are written on a *copy* of
    ``bundle.graph``:

    * ``dnfr_(p,k) = h_sigma(E_n) = exp(-E_n^2 / (2 sigma^2))``
      -- the test profile becomes the local reorganisation pressure.
    * ``phase_(p,k) = wrap(h_sigma(E_n))``
      -- same profile drives a small phase gradient along each ladder,
      so that |grad phi| and K_phi are non-zero.
    * ``EPI_(p,k) = h_sigma(E_n)``
      -- primary information amplitude tracks the test profile.
    * ``nu_f_(p,k) = E_n``
      -- inherited unchanged from the P34 construction.

    Rationale
    ---------
    This is a *canonical* (but not unique) TNFR realisation of the
    test function h_sigma on the chi-twisted graph.  The ``dnfr``
    channel feeds Phi_s; the ``phase`` channel feeds |grad phi| and
    K_phi; the result is a structural state in which every component
    of the tetrad responds to h_sigma.  Different mappings would
    activate different sectors of the Lyapunov functional and yield
    different E_TNFR_chi[sigma].

    Parameters
    ----------
    bundle
        chi-twisted prime-ladder Hamiltonian bundle from P34.
    sigma
        Width of the Gaussian test function (must be positive).

    Returns
    -------
    networkx.Graph
        A copy of ``bundle.graph`` with structural attributes
        overwritten.

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
        phase = h_val if h_val <= math.pi else math.pi
        G.nodes[node]["dnfr"] = h_val
        G.nodes[node]["phase"] = phase
        G.nodes[node]["EPI"] = h_val
        # nu_f stays as set by build_twisted_prime_ladder_graph
    return G


def twisted_tnfr_lyapunov_of_test_state(
    bundle: TwistedPrimeLadderHamiltonian,
    sigma: float,
) -> float:
    r"""Compute the canonical TNFR Lyapunov energy E_TNFR_chi[sigma]
    for the chi-twisted structural test state.

    Equivalent to::

        G = build_twisted_structural_test_state(bundle, sigma)
        return compute_energy_functional(G)

    The Lyapunov energy is the canonical structural functional

        E[G] = (1/2) sum_i [Phi_s^2(i) + |grad phi|^2(i)
                          + K_phi^2(i) + J_phi^2(i) + J_DeltaNFR^2(i)],

    guaranteed non-negative by construction.  Under grammar-compliant
    evolution (U1-U6) the time derivative is non-positive (Structural
    Conservation Theorem).
    """
    G = build_twisted_structural_test_state(bundle, sigma)
    return compute_energy_functional(G)


# ---------------------------------------------------------------------------
# Verification drivers
# ---------------------------------------------------------------------------


def verify_twisted_weil_positivity(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    *,
    sigma: float = 2.0,
    t_min: float = 0.5,
    t_max: float | None = None,
    initial_step: float = 0.25,
    dps: int = 30,
    integration_limit: float | None = None,
) -> TwistedWeilPositivityCertificate:
    r"""Verify the chi-twisted Weil positivity functional
    W_chi[sigma] >= 0.

    Computes W_chi[sigma] := 2 sum_{gamma > 0} h_sigma(gamma) two ways:

    * **Zero side**: Hardy-Z bisection of L(s, chi) on the critical
      line (reuses `twisted_weil_zero_side` from P35).
    * **Explicit formula side**: constant + archimedean + prime sides
      per the chi-twisted Weil-Guinand identity, with the prime side
      computed from the P34 Hamiltonian via
      `twisted_weil_prime_side_from_hamiltonian`.

    The two computations are exposed independently; their difference
    is the consistency residual (which should be small if P35 is
    consistent for this sigma).

    Parameters
    ----------
    chi
        Primitive real character.  Must agree with the character used
        to build ``bundle`` (the routine checks the modulus).
    bundle
        chi-twisted prime-ladder Hamiltonian bundle from P34.
    sigma
        Width of the Gaussian test function.
    t_min, t_max, initial_step, dps
        Forwarded to `twisted_weil_zero_side`.
    integration_limit
        Forwarded to `twisted_weil_archimedean_integral`.

    Returns
    -------
    TwistedWeilPositivityCertificate
        Frozen result with both computations, residual, and
        positivity flag.

    Raises
    ------
    ValueError
        If the character modulus does not match the bundle.
    """
    if bundle.character_modulus != chi.modulus:
        raise ValueError(
            f"chi.modulus ({chi.modulus}) does not match "
            f"bundle.character_modulus ({bundle.character_modulus})."
        )

    test = gaussian_test_function(sigma)
    if t_max is None:
        t_max = 12.0 * sigma

    zero_side, n_used, _zeros = twisted_weil_zero_side(
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

    residual = float(abs(zero_side - rhs))
    positive = bool(zero_side >= 0.0)

    return TwistedWeilPositivityCertificate(
        character_name=str(chi.name),
        character_modulus=int(chi.modulus),
        character_parity=character_parity(chi),
        sigma=float(sigma),
        weil_functional_zero_side=float(zero_side),
        weil_functional_explicit_formula=float(rhs),
        explicit_formula_residual=residual,
        n_zeros_used=int(n_used),
        positive=positive,
    )


def verify_twisted_weil_tnfr_bridge(
    chi: DirichletCharacter,
    bundle: TwistedPrimeLadderHamiltonian,
    sigmas: Sequence[float],
    *,
    t_min: float = 0.5,
    t_max: float | None = None,
    initial_step: float = 0.25,
    dps: int = 30,
    integration_limit: float | None = None,
) -> TwistedWeilTNFRBridgeCertificate:
    r"""Tabulate the chi-twisted TNFR-Weil positivity bridge across a
    sigma grid.

    For each sigma in ``sigmas``, computes

    * W_chi[sigma] via `verify_twisted_weil_positivity` (zero side),
    * E_TNFR_chi[sigma] via `twisted_tnfr_lyapunov_of_test_state`,
    * alpha_chi(sigma) = W_chi[sigma] / E_TNFR_chi[sigma].

    A constant positive lower bound alpha_min > 0 across a dense
    admissible family would constitute a TNFR-native witness for the
    chi-twisted Weil positivity inequality (and hence, via Weil's
    equivalence, for GRH_chi).  This module checks the inequality
    numerically; it does **not** prove uniform positivity.

    Parameters
    ----------
    chi
        Primitive real character.  Must match ``bundle`` modulus.
    bundle
        chi-twisted prime-ladder Hamiltonian bundle from P34.
    sigmas
        Grid of Gaussian widths to evaluate.
    t_min, t_max, initial_step, dps, integration_limit
        Forwarded to `verify_twisted_weil_positivity`.  ``t_max``,
        if provided, applies uniformly to every sigma; otherwise the
        per-sigma default ``12 * sigma`` is used.

    Returns
    -------
    TwistedWeilTNFRBridgeCertificate
        Frozen result with per-sigma arrays and aggregate positivity
        flags.

    Raises
    ------
    ValueError
        If the character modulus does not match the bundle, or if
        ``sigmas`` is empty or contains non-positive values.
    """
    if bundle.character_modulus != chi.modulus:
        raise ValueError(
            f"chi.modulus ({chi.modulus}) does not match "
            f"bundle.character_modulus ({bundle.character_modulus})."
        )

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
        cert = verify_twisted_weil_positivity(
            chi,
            bundle,
            sigma=sigma_f,
            t_min=t_min,
            t_max=t_max,
            initial_step=initial_step,
            dps=dps,
            integration_limit=integration_limit,
        )
        W = float(cert.weil_functional_zero_side)
        E = float(twisted_tnfr_lyapunov_of_test_state(bundle, sigma_f))

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

    return TwistedWeilTNFRBridgeCertificate(
        character_name=str(chi.name),
        character_modulus=int(chi.modulus),
        character_parity=character_parity(chi),
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
