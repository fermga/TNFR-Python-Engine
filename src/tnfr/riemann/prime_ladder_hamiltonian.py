r"""Declared finite prime-ladder Hamiltonian and weighted trace comparisons (P14).

The constructor supplies prime labels and frequencies nu_(p,k)=k*log(p),
then instantiates the selected InternalHamiltonian coefficients. In the
decoupled frequency-only limit its real diagonal spectrum is exactly those
supplied values. The diagonal weight log(p) gives the matching finite trace
sum_(p,k) log(p)*exp(-s*k*log(p)). The corresponding infinite arithmetic series
equals -zeta'(s)/zeta(s) only on Re(s)>1 before classical continuation.

Reading supplied diagonal frequencies back as eigenvalues is a construction
identity and a numerical implementation check, not independent emergence of
prime arithmetic. Ladder edges are not an executed REMESH history. The finite
prime-ladder spectrum is not the Riemann-zero spectrum, and poles of a separate
analytic zeta evaluator are not poles of this finite Hamiltonian's resolvent.

No autonomous joint nodal law, universal catalog symmetry, physical particle
mechanism or Hilbert-Polya/RH result is derived. Optional coupling coefficients
specify a different finite model and must be reported with their provenance.
See theory/TNFR_RIEMANN_RESEARCH_NOTES.md for the current scope."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..operators.hamiltonian import InternalHamiltonian
from .nodal_pulse import first_primes as _first_primes
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
    r"""Construct a declared graph of supplied prime labels and finite ladder indices.

    Each node (p,k), 1<=k<=max_power, receives nu_f=k*log(p), phase=0, EPI=1,
    Si=1 and dnfr=0 as assigned attributes. Edges join consecutive indices on
    one ladder; they do not execute REMESH. Disconnected prime ladders are a
    construction choice, not a derived autonomous factorization mechanism.

    n_primes selects the generated prime list unless primes is supplied.
    max_power is the finite ladder depth; coupling is the configured ladder
    coefficient. The graph is used by a separate selected Hamiltonian, whose
    decoupled frequency-only spectrum reads back the assigned k*log(p) values."""
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
    r"""Instantiate the selected InternalHamiltonian on a declared finite prime ladder.

    n_primes/primes specify arithmetic labels, max_power the ladder depth,
    and the coefficient arguments the chosen finite matrix. The decoupled
    frequency-only case reproduces assigned k*log(p) entries. No REMESH history,
    autonomous graph formation or Riemann-zero Hamiltonian is certified."""
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
