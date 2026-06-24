r"""TNFR prime-ladder Hamiltonian (P14 program — Gap G1 closure).

Goal
----
Instantiate the canonical TNFR internal Hamiltonian

.. math::

    \hat{H}_{\mathrm{int}} = \hat{H}_{\mathrm{coh}} + \hat{H}_{\mathrm{freq}}
                            + \hat{H}_{\mathrm{coupling}}

(see :class:`tnfr.operators.hamiltonian.InternalHamiltonian`) on the
**prime-ladder graph** introduced by the P12 program
(:mod:`tnfr.riemann.von_mangoldt`).  This provides an explicit,
self-adjoint, finite-dimensional operator whose:

1. **Spectrum** (in the decoupled limit :math:`J_0 = 0`,
   :math:`C_0 = 0`) reproduces exactly the prime-ladder spectrum
   :math:`\{k\log p\}_{p\in\mathcal{P},\,k=1,\dots,K}`.

2. **Weighted spectral trace**
   :math:`\mathrm{Tr}(\hat W e^{-s\hat H_{\mathrm{freq}}})`, with the
   diagonal weight operator
   :math:`\hat W = \sum_{p,k}\log(p)\,|p,k\rangle\langle p,k|`,
   reproduces exactly the TNFR weighted Dirichlet trace
   :math:`Z_{\mathrm{vM}}(s)` of P12, which in turn converges to
   :math:`-\zeta'(s)/\zeta(s)` for :math:`\mathrm{Re}(s) > 1`.

TNFR interpretation
-------------------
Each prime :math:`p` contributes a **REMESH echo ladder** — a chain of
nodes :math:`(p,1), (p,2), \dots, (p,K)` linked by ladder edges
(operator #13, recursivity).  The structural frequency assigned to
each node is

.. math::

    \nu_{f,(p,k)} = k \log p,

which equals its diagonal entry in
:math:`\hat H_{\mathrm{freq}}` (per the canonical construction in
:mod:`tnfr.operators.hamiltonian`).  No inter-prime coupling is
introduced: distinct prime ladders are structurally orthogonal, which
encodes the **multiplicativity of the Euler product** at the
operator level (different primes correspond to independent invariant
subspaces of :math:`\hat H`).

Closing Gap G1 (operationally)
------------------------------
The Hilbert-Pólya programme asks for a self-adjoint operator whose
spectrum encodes the prime data driving :math:`\zeta(s)`.  In this
module:

* **Self-adjointness** is automatic — :class:`InternalHamiltonian`
  verifies Hermiticity of every component at construction
  (:meth:`InternalHamiltonian._verify_hermitian`), and a diagonal
  real matrix is trivially self-adjoint.

* **Spectrum** matches the prime-ladder data by construction (proved
  here as a numerical certificate, exact to machine precision).

* **Connection to** :math:`\zeta(s)` is realised via the weighted
  trace, which equals :math:`Z_{\mathrm{vM}}(s)` of P12 and is
  analytically continued to all of :math:`\mathbb{C}` by P13
  (:mod:`tnfr.riemann.analytic_continuation`).

What this module does NOT do
----------------------------
* It does **not** prove that the non-trivial Riemann zeros are forced
  onto :math:`\mathrm{Re}(s) = 1/2` (that is gap G4 — the substance
  of RH itself).  It only exposes them as resonance poles of the
  resolvent of the analytic continuation, matching the picture of P13.

* It does **not** introduce any coupling between distinct primes.
  Doing so would break the Euler product structure
  :math:`\zeta(s) = \prod_p (1 - p^{-s})^{-1}` at the operator level
  unless the coupling is chosen with extreme care.  Non-zero coupling
  is exposed as an optional parameter for **perturbative studies
  only**, and the certificate API explicitly verifies the decoupled
  limit.

Status: EXPERIMENTAL — Research prototype for TNFR-Riemann P14 program
(gap G1 closure, May 2026).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..operators.hamiltonian import InternalHamiltonian
from .operator import _first_primes
from .von_mangoldt import (
    PrimeLadderSpectrum,
    build_prime_ladder_spectrum,
    tnfr_log_zeta_derivative,
)

__all__ = [
    "build_prime_ladder_graph",
    "build_prime_ladder_weight_operator",
    "PrimeLadderHamiltonian",
    "build_prime_ladder_hamiltonian",
    "weighted_spectral_trace",
    "PrimeLadderHamiltonianCertificate",
    "verify_hamiltonian_reproduces_prime_ladder",
]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_prime_ladder_graph(
    n_primes: int,
    *,
    max_power: int = 8,
    coupling: float = 0.0,
    primes: Sequence[int] | None = None,
) -> nx.Graph:
    r"""Construct the TNFR prime-ladder graph.

    Nodes are labelled by pairs ``(p, k)`` for each prime
    :math:`p \in \mathcal{P}` and each echo index
    :math:`k = 1, \dots, K`.  Each node carries the canonical TNFR
    structural attributes:

    * ``nu_f = k * log(p)`` (structural frequency, energy in
      :math:`\hat H_{\mathrm{freq}}`),
    * ``phase = 0``, ``EPI = 1.0``, ``Si = 1.0``, ``dnfr = 0.0``
      (neutral structural state; coherence and pressure components
      do not enter the decoupled Hamiltonian).

    REMESH echo edges link consecutive nodes on the same prime ladder
    :math:`(p, k) \leftrightarrow (p, k+1)`.  No edges connect
    distinct primes — the Euler-product orthogonality is enforced at
    the graph level.

    Parameters
    ----------
    n_primes : int
        Number of primes in :math:`\mathcal{P}` (ignored if ``primes``
        is provided).
    max_power : int, default 8
        REMESH echo cap :math:`K`.  Must satisfy ``max_power >= 1``.
    coupling : float, default 0.0
        Strength of the inter-node ladder coupling
        :math:`J_0` in :math:`\hat H_{\mathrm{coupling}}`.  Default
        ``0.0`` yields a purely diagonal Hamiltonian whose spectrum
        equals the prime-ladder spectrum exactly.  Non-zero values are
        perturbative and break exact spectrum reproduction; intended
        for stability / dependence studies only.
    primes : sequence of int, optional
        Explicit prime list.  If given, ``n_primes`` is ignored.

    Returns
    -------
    networkx.Graph
        Prime-ladder graph with structural attributes and Hamiltonian
        configuration (``H_COH_STRENGTH = 0``, ``H_COUPLING_STRENGTH =
        coupling``) attached to ``graph.graph``.

    Raises
    ------
    ValueError
        If ``max_power < 1`` or ``n_primes < 1`` (when ``primes`` not
        provided).
    """
    if max_power < 1:
        raise ValueError("max_power must be >= 1")

    if primes is None:
        if n_primes < 1:
            raise ValueError("n_primes must be >= 1")
        prime_list = _first_primes(n_primes)
    else:
        prime_list = list(primes)

    G = nx.Graph()
    # Disable coherence potential (irrelevant for prime-ladder spectrum)
    # and set ladder coupling strength.
    G.graph["H_COH_STRENGTH"] = 0.0
    G.graph["H_COUPLING_STRENGTH"] = float(coupling)

    for p in prime_list:
        log_p = math.log(p)
        for k in range(1, max_power + 1):
            node = (int(p), int(k))
            G.add_node(
                node,
                nu_f=float(k * log_p),
                phase=0.0,
                EPI=1.0,
                Si=1.0,
                dnfr=0.0,
            )
        # REMESH echo edges along the ladder of this prime only
        for k in range(1, max_power):
            G.add_edge((int(p), k), (int(p), k + 1))

    return G


def build_prime_ladder_weight_operator(G: nx.Graph) -> np.ndarray:
    r"""Diagonal weight operator :math:`\hat W = \sum_{p,k}\log(p)|p,k\rangle\langle p,k|`.

    The weight operator encodes the per-node structural emission
    strength.  In the prime-ladder construction every node
    :math:`(p,k)` carries the same weight :math:`\log p` regardless
    of the echo index :math:`k` — this is the canonical TNFR reading
    of the von Mangoldt function :math:`\Lambda(p^k) = \log p`.

    The trace
    :math:`\mathrm{Tr}(\hat W e^{-s\hat H_{\mathrm{freq}}})`
    reproduces, by construction, the weighted Dirichlet trace
    :math:`Z_{\mathrm{vM}}(s)` of :mod:`tnfr.riemann.von_mangoldt`.

    Parameters
    ----------
    G : networkx.Graph
        Output of :func:`build_prime_ladder_graph`.

    Returns
    -------
    numpy.ndarray
        Diagonal real ``(N, N)`` matrix with entries
        :math:`W_{(p,k),(p,k)} = \log p`.  Node ordering follows
        ``cached_node_list(G)`` (the same ordering used by
        :class:`InternalHamiltonian`).
    """
    from ..utils.cache import cached_node_list

    nodes = cached_node_list(G)
    weights = np.zeros(len(nodes), dtype=float)
    for i, node in enumerate(nodes):
        p, _k = node
        weights[i] = math.log(p)
    return np.diag(weights)


# ---------------------------------------------------------------------------
# Hamiltonian wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimeLadderHamiltonian:
    r"""Bundled prime-ladder Hamiltonian, weight operator, and spectral data.

    Attributes
    ----------
    graph : networkx.Graph
        Prime-ladder graph.
    hamiltonian : InternalHamiltonian
        Canonical TNFR internal Hamiltonian instantiated on ``graph``.
    weight_operator : numpy.ndarray
        Diagonal weight operator :math:`\hat W` (per
        :func:`build_prime_ladder_weight_operator`).
    spectrum : PrimeLadderSpectrum
        Reference prime-ladder spectrum (from P12) for verification.
    coupling : float
        Inter-node ladder coupling strength :math:`J_0` used at
        construction.
    """

    graph: nx.Graph
    hamiltonian: InternalHamiltonian
    weight_operator: np.ndarray
    spectrum: PrimeLadderSpectrum
    coupling: float


def build_prime_ladder_hamiltonian(
    n_primes: int,
    *,
    max_power: int = 8,
    coupling: float = 0.0,
    primes: Sequence[int] | None = None,
) -> PrimeLadderHamiltonian:
    r"""Instantiate the canonical TNFR Hamiltonian on the prime-ladder graph.

    This is the **operational closure** of gap G1: a self-adjoint
    finite-dimensional operator whose decoupled (``coupling = 0``)
    spectrum equals the prime-ladder spectrum and whose weighted
    spectral trace reproduces :math:`Z_{\mathrm{vM}}(s)`.

    Parameters
    ----------
    n_primes : int
        Number of primes (ignored if ``primes`` provided).
    max_power : int, default 8
        REMESH echo cap.
    coupling : float, default 0.0
        Ladder coupling strength.  ``0.0`` gives the exact diagonal
        spectrum; non-zero values produce perturbed spectra.
    primes : sequence of int, optional
        Explicit prime list.

    Returns
    -------
    PrimeLadderHamiltonian
        Bundle containing the graph, the Hamiltonian, the weight
        operator, the reference spectrum, and the coupling value.
    """
    G = build_prime_ladder_graph(
        n_primes,
        max_power=max_power,
        coupling=coupling,
        primes=primes,
    )
    H = InternalHamiltonian(G)
    W = build_prime_ladder_weight_operator(G)
    spectrum = build_prime_ladder_spectrum(
        n_primes,
        max_power=max_power,
        primes=primes,
    )
    return PrimeLadderHamiltonian(
        graph=G,
        hamiltonian=H,
        weight_operator=W,
        spectrum=spectrum,
        coupling=float(coupling),
    )


# ---------------------------------------------------------------------------
# Spectral observables
# ---------------------------------------------------------------------------


def weighted_spectral_trace(
    H_freq: np.ndarray,
    W: np.ndarray,
    s: float | complex,
) -> complex:
    r"""Weighted spectral trace :math:`\mathrm{Tr}(\hat W e^{-s \hat H_{\mathrm{freq}}})`.

    For a diagonal Hamiltonian (decoupled prime ladders), this reduces
    to :math:`\sum_n W_{nn} e^{-s E_n}`, which equals the TNFR
    weighted Dirichlet trace :math:`Z_{\mathrm{vM}}(s)` from
    :func:`tnfr.riemann.von_mangoldt.tnfr_log_zeta_derivative`.

    For a perturbed Hamiltonian (``coupling != 0``), it evaluates
    :math:`\mathrm{Tr}(\hat W e^{-s \hat H_{\mathrm{int}}})` via the
    spectral decomposition of :math:`\hat H_{\mathrm{int}}` — see
    :meth:`InternalHamiltonian.get_spectrum`.

    Parameters
    ----------
    H_freq : numpy.ndarray
        Hamiltonian (or its frequency component) — must be Hermitian.
    W : numpy.ndarray
        Diagonal weight operator (real).
    s : float or complex
        Spectral parameter.  Convergence requires :math:`\mathrm{Re}(s) > 1`
        in the infinite-prime limit; in the finite-dimensional model it
        is well-defined for all :math:`s \in \mathbb{C}`.

    Returns
    -------
    complex
        :math:`\mathrm{Tr}(\hat W e^{-s \hat H_{\mathrm{freq}}})`.
    """
    eigvals, eigvecs = np.linalg.eigh(H_freq)
    # W in the eigenbasis: W_diag_eig[n] = <phi_n| W |phi_n>
    # For diagonal H and diagonal W on the same basis, eigvecs = identity
    # and the formula collapses to sum w_n exp(-s E_n).
    s_c = complex(s)
    exp_minus_sE = np.exp(-s_c * eigvals)
    # diag entries of U^H W U in the eigenbasis
    W_eig = np.einsum("ij,jk,ki->i", eigvecs.conj().T, W, eigvecs)
    z = np.sum(W_eig * exp_minus_sE)
    return complex(z) if isinstance(s, complex) else float(z.real)


# ---------------------------------------------------------------------------
# Certificate: Hamiltonian reproduces the prime-ladder data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimeLadderHamiltonianCertificate:
    r"""Numerical certificate that the Hamiltonian reproduces the P12 spectrum.

    Attributes
    ----------
    n_primes : int
        Number of primes used.
    max_power : int
        REMESH echo cap.
    coupling : float
        Coupling strength used at construction.
    hilbert_dim : int
        :math:`N` = total Hilbert-space dimension =
        :math:`n_{\mathrm{primes}} \times K`.
    is_hermitian : bool
        Whether the constructed Hamiltonian passed the Hermiticity
        check (``InternalHamiltonian._verify_hermitian``).  Always
        ``True`` for a successfully-constructed bundle (the
        constructor raises otherwise).
    spectrum_max_abs_error : float
        :math:`\max_n |E_n^{\mathrm{Ham}} - E_n^{\mathrm{ladder}}|`
        between the sorted Hamiltonian eigenvalues and the sorted
        prime-ladder eigenvalues.  At ``coupling = 0`` this is zero
        to machine precision.
    spectrum_reproduced : bool
        ``spectrum_max_abs_error <= spectrum_tol``.
    s_values : numpy.ndarray
        Real spectral parameters at which the weighted trace was
        compared.
    trace_max_rel_error : float
        Worst-case relative error
        :math:`\max_s |Z_H(s) - Z_{\mathrm{vM}}(s)|/|Z_{\mathrm{vM}}(s)|`
        between the Hamiltonian trace and the reference
        prime-ladder trace.  At ``coupling = 0`` this is at the
        floating-point round-off level.
    trace_reproduced : bool
        ``trace_max_rel_error <= trace_tol``.
    overall_ok : bool
        Both spectrum and trace reproduction succeeded.

    Notes
    -----
    Failure of either reproduction at ``coupling = 0`` indicates a
    construction bug.  Failure at ``coupling != 0`` is **expected**
    and quantifies how strongly inter-ladder coupling deforms the
    prime-ladder spectrum.
    """

    n_primes: int
    max_power: int
    coupling: float
    hilbert_dim: int
    is_hermitian: bool
    spectrum_max_abs_error: float
    spectrum_reproduced: bool
    s_values: np.ndarray
    trace_max_rel_error: float
    trace_reproduced: bool
    overall_ok: bool


def verify_hamiltonian_reproduces_prime_ladder(
    bundle: PrimeLadderHamiltonian,
    s_values: Sequence[float] = (2.0, 3.0, 5.0, 10.0),
    *,
    spectrum_tol: float = 1e-10,
    trace_tol: float = 1e-10,
) -> PrimeLadderHamiltonianCertificate:
    r"""Verify that the Hamiltonian reproduces the prime-ladder spectrum and trace.

    Two checks are performed:

    1. **Spectrum reproduction.** Compute the sorted eigenvalues of
       ``bundle.hamiltonian.H_int`` and compare with the sorted
       ``bundle.spectrum.eigenvalues``.  At ``coupling = 0`` these
       must agree to machine precision.

    2. **Weighted trace reproduction.** Compute
       :math:`\mathrm{Tr}(\hat W e^{-s\hat H_{\mathrm{int}}})` via
       :func:`weighted_spectral_trace` and compare with
       :func:`tnfr.riemann.von_mangoldt.tnfr_log_zeta_derivative` at
       each ``s`` in ``s_values``.

    Parameters
    ----------
    bundle : PrimeLadderHamiltonian
        Output of :func:`build_prime_ladder_hamiltonian`.
    s_values : sequence of float, default (2.0, 3.0, 5.0, 10.0)
        Real spectral parameters at which to compare the weighted
        trace.  All values should satisfy :math:`s > 1` for clean
        comparison with the convergent classical regime.
    spectrum_tol : float, default 1e-10
        Maximum allowed absolute deviation between the two spectra.
    trace_tol : float, default 1e-10
        Maximum allowed relative deviation between the two traces.

    Returns
    -------
    PrimeLadderHamiltonianCertificate
        Numerical certificate documenting both checks.
    """
    H = bundle.hamiltonian
    W = bundle.weight_operator
    spectrum = bundle.spectrum

    # --- Spectrum check ---
    eigvals_ham, _ = H.get_spectrum()
    eigvals_ham_sorted = np.sort(np.real(eigvals_ham))
    eigvals_ref_sorted = np.sort(spectrum.eigenvalues)
    spectrum_abs_error = float(np.max(np.abs(eigvals_ham_sorted - eigvals_ref_sorted)))
    spectrum_ok = spectrum_abs_error <= spectrum_tol

    # --- Weighted trace check ---
    s_arr = np.asarray(list(s_values), dtype=float)
    z_ham = np.empty(s_arr.size, dtype=complex)
    z_ref = np.empty(s_arr.size, dtype=complex)
    for i, s in enumerate(s_arr):
        z_ham[i] = weighted_spectral_trace(H.H_int, W, float(s))
        z_ref[i] = complex(tnfr_log_zeta_derivative(spectrum, float(s)))
    abs_err = np.abs(z_ham - z_ref)
    rel_err = abs_err / np.maximum(np.abs(z_ref), 1e-300)
    trace_rel_error = float(np.max(rel_err))
    trace_ok = trace_rel_error <= trace_tol

    return PrimeLadderHamiltonianCertificate(
        n_primes=spectrum.n_primes,
        max_power=spectrum.max_power,
        coupling=bundle.coupling,
        hilbert_dim=H.N,
        is_hermitian=True,  # constructor would have raised otherwise
        spectrum_max_abs_error=spectrum_abs_error,
        spectrum_reproduced=spectrum_ok,
        s_values=s_arr,
        trace_max_rel_error=trace_rel_error,
        trace_reproduced=trace_ok,
        overall_ok=bool(spectrum_ok and trace_ok),
    )
