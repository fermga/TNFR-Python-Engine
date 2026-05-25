r"""TNFR χ-twisted prime-ladder Hamiltonian (P34 program — structural analogue
of P14 for general Dirichlet L-functions; G1$_\chi$ operational layer).

Goal
----
Instantiate the canonical TNFR internal Hamiltonian

.. math::

    \hat{H}_{\mathrm{int}} = \hat{H}_{\mathrm{coh}} + \hat{H}_{\mathrm{freq}}
                            + \hat{H}_{\mathrm{coupling}}

(see :class:`tnfr.operators.hamiltonian.InternalHamiltonian`) on the
**χ-twisted prime-ladder graph** — the same disjoint union of per-prime
REMESH echo ladders introduced by P12 (:mod:`tnfr.riemann.von_mangoldt`)
and P14 (:mod:`tnfr.riemann.prime_ladder_hamiltonian`), but restricted
to the primes coprime to the conductor :math:`q` of a Dirichlet
character :math:`\chi`.

This is the structural analogue of P14 for general :math:`L(s,\chi)`:

1. **Spectrum** (in the decoupled limit :math:`J_0 = 0`,
   :math:`C_0 = 0`) reproduces exactly the χ-twisted prime-ladder
   spectrum :math:`\{k\log p\}_{p \nmid q,\,k = 1,\dots,K}` of P32
   (:mod:`tnfr.riemann.dirichlet_l`).

2. **Twisted weighted spectral trace**
   :math:`\mathrm{Tr}(\hat W^{(\chi)} e^{-s\hat H_{\mathrm{freq}}})`,
   with the diagonal weight operator
   :math:`\hat W^{(\chi)} = \sum_{p\nmid q,\,k} \chi(p)^k\,\log(p)\,
   |p,k\rangle\langle p,k|`, reproduces exactly the TNFR χ-twisted
   weighted Dirichlet trace of P32, which in turn converges to
   :math:`-L'(s,\chi)/L(s,\chi)` for :math:`\mathrm{Re}(s) > 1`.

TNFR interpretation
-------------------
The Hamiltonian itself is **self-adjoint** (real-symmetric, since the
energies :math:`k\log p` are real positive numbers and the bare graph
is diagonal in the decoupled limit).  The χ-twist enters **only through
the weight operator** :math:`\hat W^{(\chi)}`, which is diagonal but
generally **not Hermitian** for non-real characters: its diagonal
entries :math:`\chi(p)^k \log p` are complex when :math:`\chi(p)` is a
non-trivial root of unity.  This is the canonical structural reading of
the χ-twist: the energy ladder is character-independent, but the
emission *phase* per node carries the arithmetic information of
:math:`\chi`.

For real characters (Legendre symbol mod :math:`q`), :math:`\chi(p)^k
\in \{-1, 0, +1\}^{\mathbb{Z}}` and :math:`\hat W^{(\chi)}` is
Hermitian with real signed weights.  For complex characters,
:math:`\hat W^{(\chi)}` is a normal but non-Hermitian diagonal
operator with complex spectrum.

Closing G1$_\chi$ (operationally)
---------------------------------
P14 closed G1 for :math:`\zeta(s)` by exhibiting a self-adjoint
finite-dimensional operator whose decoupled spectrum equals the
prime-ladder data and whose weighted trace equals :math:`Z_{\mathrm{vM}}(s)`.
P34 does the analogous construction for every Dirichlet L-function:

* **Self-adjointness** of :math:`\hat H` is automatic (delegated to
  :class:`InternalHamiltonian._verify_hermitian`, identical to P14).
* **Spectrum** matches the χ-twisted prime-ladder data by construction.
* **Connection to** :math:`L(s,\chi)` is realised via the χ-twisted
  weighted trace, which equals :math:`Z_{\mathrm{TNFR}}(s,\chi)` of
  P32 and is analytically continued to all of :math:`\mathbb{C}` by
  P33 (:mod:`tnfr.riemann.analytic_continuation_dirichlet`).

What this module does NOT do
----------------------------
* It does **not** prove that the non-trivial zeros of
  :math:`L(s, \chi)` are forced onto :math:`\mathrm{Re}(s) = 1/2`
  (that is the Generalised Riemann Hypothesis — gap G4$_\chi$, the
  substance of GRH itself).  It only exposes them as resonance poles
  of the resolvent of the analytic continuation, matching the picture
  of P33.

* It does **not** introduce coupling between distinct primes; the
  Euler-product orthogonality :math:`L(s,\chi) = \prod_p (1 -
  \chi(p) p^{-s})^{-1}` is preserved at the operator level.

* It does **not** verify a numerical Weil–Guinand explicit formula
  for :math:`L(s,\chi)` — that is the future P35 (G3$_\chi$).

References
----------
* P14: :mod:`tnfr.riemann.prime_ladder_hamiltonian` (the template).
* P32: :mod:`tnfr.riemann.dirichlet_l` (provides the reference
  χ-twisted spectrum and the classical Dirichlet sum).
* P33: :mod:`tnfr.riemann.analytic_continuation_dirichlet` (provides
  the analytic continuation that connects the χ-twisted prime-ladder
  trace to :math:`-L'(s,\chi)/L(s,\chi)` on the whole complex plane).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import networkx as nx

from ..mathematics.unified_numerical import np
from ..operators.hamiltonian import InternalHamiltonian
from .dirichlet_l import (
    DirichletCharacter,
    TwistedPrimeLadderSpectrum,
    build_twisted_prime_ladder_spectrum,
    tnfr_log_l_derivative,
)
from .operator import _first_primes


__all__ = [
    "build_twisted_prime_ladder_graph",
    "build_twisted_prime_ladder_weight_operator",
    "TwistedPrimeLadderHamiltonian",
    "build_twisted_prime_ladder_hamiltonian",
    "twisted_weighted_spectral_trace",
    "TwistedPrimeLadderHamiltonianCertificate",
    "verify_twisted_hamiltonian_reproduces_prime_ladder",
]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_twisted_prime_ladder_graph(
    chi: DirichletCharacter,
    n_primes: int,
    *,
    max_power: int = 8,
    coupling: float = 0.0,
    primes: Sequence[int] | None = None,
) -> nx.Graph:
    r"""Construct the χ-twisted TNFR prime-ladder graph.

    Nodes are labelled by pairs ``(p, k)`` for each prime
    :math:`p` **coprime to** :math:`q = \chi.\mathrm{modulus}` and
    each echo index :math:`k = 1, \dots, K`.  Primes dividing
    :math:`q` are dropped at graph-construction time (their structural
    contribution to :math:`L(s, \chi)` vanishes because
    :math:`\chi(p) = 0`).

    Each node carries the canonical TNFR structural attributes:

    * ``nu_f = k * log(p)`` (structural frequency, energy in
      :math:`\hat H_{\mathrm{freq}}`),
    * ``phase = 0``, ``EPI = 1.0``, ``Si = 1.0``, ``dnfr = 0.0``
      (neutral structural state; coherence and pressure components
      do not enter the decoupled Hamiltonian).

    REMESH echo edges link consecutive nodes on the same prime ladder
    :math:`(p, k) \leftrightarrow (p, k+1)`.  No edges connect
    distinct primes — Euler-product orthogonality is enforced at the
    graph level exactly as in P14.

    Parameters
    ----------
    chi : DirichletCharacter
        Character defining the twist.  Only the modulus is used at
        the graph level (to decide which primes are active); the
        actual χ-values enter through the weight operator (see
        :func:`build_twisted_prime_ladder_weight_operator`).
    n_primes : int
        Number of primes in the seed list (ignored if ``primes`` is
        provided).  Primes dividing the modulus are still counted in
        this total but excluded from the graph.
    max_power : int, default 8
        REMESH echo cap :math:`K`.  Must satisfy ``max_power >= 1``.
    coupling : float, default 0.0
        Strength of the inter-node ladder coupling
        :math:`J_0` in :math:`\hat H_{\mathrm{coupling}}`.  Default
        ``0.0`` yields a purely diagonal Hamiltonian whose spectrum
        equals the χ-twisted prime-ladder spectrum exactly.  Non-zero
        values are perturbative and break exact spectrum
        reproduction; intended for stability / dependence studies
        only.
    primes : sequence of int, optional
        Explicit prime list.  If given, ``n_primes`` is ignored.

    Returns
    -------
    networkx.Graph
        χ-twisted prime-ladder graph with structural attributes and
        Hamiltonian configuration attached to ``graph.graph``.  The
        following extra metadata is set:

        * ``graph.graph["character_modulus"] = q``
        * ``graph.graph["character_name"] = chi.name``
        * ``graph.graph["primes_excluded"] = tuple of primes p|q``

    Raises
    ------
    ValueError
        If ``max_power < 1``, ``n_primes < 1`` (when ``primes`` not
        provided), or if every supplied prime divides the modulus.
    """
    if max_power < 1:
        raise ValueError("max_power must be >= 1")

    if primes is None:
        if n_primes < 1:
            raise ValueError("n_primes must be >= 1")
        prime_list = _first_primes(n_primes)
    else:
        prime_list = list(primes)
        if not prime_list:
            raise ValueError("primes must be non-empty")

    active: list[int] = []
    excluded: list[int] = []
    for p in prime_list:
        if abs(chi(p)) < 1e-15:
            excluded.append(int(p))
        else:
            active.append(int(p))

    if not active:
        raise ValueError(
            "All supplied primes divide the character modulus; "
            "twisted prime-ladder graph would be empty."
        )

    G = nx.Graph()
    G.graph["H_COH_STRENGTH"] = 0.0
    G.graph["H_COUPLING_STRENGTH"] = float(coupling)
    G.graph["character_modulus"] = int(chi.modulus)
    G.graph["character_name"] = str(chi.name)
    G.graph["primes_excluded"] = tuple(excluded)

    for p in active:
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
        for k in range(1, max_power):
            G.add_edge((int(p), k), (int(p), k + 1))

    return G


def build_twisted_prime_ladder_weight_operator(
    chi: DirichletCharacter,
    G: nx.Graph,
) -> np.ndarray:
    r"""Diagonal χ-twisted weight operator
    :math:`\hat W^{(\chi)} = \sum_{p\nmid q,\,k}\chi(p)^k\log(p)\,|p,k\rangle\langle p,k|`.

    The weight operator encodes the per-node structural emission
    strength **twisted by the character**.  In the χ-twisted
    prime-ladder construction every node :math:`(p,k)` carries the
    weight :math:`\chi(p)^k \log p` — this is the canonical TNFR
    reading of the **χ-twisted von Mangoldt function**
    :math:`\Lambda_\chi(p^k) = \chi(p)^k \log p`.

    Hermiticity caveat
    ------------------
    For **real characters** (Legendre symbol mod :math:`q`, e.g.
    :math:`\chi_3, \chi_4, \chi_5`), :math:`\chi(p)^k \in \{-1, +1\}`
    and :math:`\hat W^{(\chi)}` is a Hermitian diagonal operator with
    real signed entries.

    For **complex characters**, :math:`\chi(p)` is a non-trivial root
    of unity in :math:`\mathbb{C}` and :math:`\hat W^{(\chi)}` is a
    *normal* (commuting with its adjoint, since it is diagonal) but
    generally **non-Hermitian** diagonal operator with complex
    diagonal entries.  This is structurally intrinsic to the χ-twist
    and is **not** a defect of the construction: the χ-twisted
    Dirichlet series :math:`\sum_n \chi(n)\Lambda(n) n^{-s}` is itself
    complex for non-real :math:`\chi`.

    The trace
    :math:`\mathrm{Tr}(\hat W^{(\chi)} e^{-s\hat H_{\mathrm{freq}}})`
    reproduces, by construction, the χ-twisted weighted Dirichlet
    trace :math:`Z_{\mathrm{TNFR}}(s,\chi)` of
    :func:`tnfr.riemann.dirichlet_l.tnfr_log_l_derivative`.

    Parameters
    ----------
    chi : DirichletCharacter
        The same character used to build the graph.
    G : networkx.Graph
        Output of :func:`build_twisted_prime_ladder_graph` (must have
        been built with the same character; primes dividing
        :math:`q` must not appear as nodes).

    Returns
    -------
    numpy.ndarray
        Diagonal complex ``(N, N)`` matrix with entries
        :math:`W^{(\chi)}_{(p,k),(p,k)} = \chi(p)^k \log p`.  Node
        ordering follows ``cached_node_list(G)`` (the same ordering
        used by :class:`InternalHamiltonian`).
    """
    from ..utils.cache import cached_node_list

    nodes = cached_node_list(G)
    weights = np.zeros(len(nodes), dtype=complex)
    for i, node in enumerate(nodes):
        p, k = node
        cp = chi(int(p))
        weights[i] = complex(cp) ** int(k) * math.log(int(p))
    return np.diag(weights)


# ---------------------------------------------------------------------------
# Hamiltonian wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwistedPrimeLadderHamiltonian:
    r"""Bundled χ-twisted prime-ladder Hamiltonian, weight operator, and spectrum.

    Attributes
    ----------
    graph : networkx.Graph
        χ-twisted prime-ladder graph (P34).
    hamiltonian : InternalHamiltonian
        Canonical TNFR internal Hamiltonian instantiated on
        ``graph`` (identical machinery as P14; the χ-twist enters
        only through the weight operator, not through :math:`\hat H`).
    weight_operator : numpy.ndarray
        Diagonal χ-twisted weight operator :math:`\hat W^{(\chi)}`
        (complex; see
        :func:`build_twisted_prime_ladder_weight_operator`).
    spectrum : TwistedPrimeLadderSpectrum
        Reference χ-twisted prime-ladder spectrum (from P32) used
        for verification.
    coupling : float
        Inter-node ladder coupling strength :math:`J_0` used at
        construction.
    character_modulus : int
        Conductor :math:`q` of the character.
    character_name : str
        Label of the character.
    """

    graph: nx.Graph
    hamiltonian: InternalHamiltonian
    weight_operator: np.ndarray
    spectrum: TwistedPrimeLadderSpectrum
    coupling: float
    character_modulus: int
    character_name: str


def build_twisted_prime_ladder_hamiltonian(
    chi: DirichletCharacter,
    n_primes: int,
    *,
    max_power: int = 8,
    coupling: float = 0.0,
    primes: Sequence[int] | None = None,
) -> TwistedPrimeLadderHamiltonian:
    r"""Instantiate the canonical TNFR Hamiltonian on the χ-twisted prime-ladder graph.

    This is the **operational closure** of G1$_\chi$: a self-adjoint
    finite-dimensional operator whose decoupled (``coupling = 0``)
    spectrum equals the χ-twisted prime-ladder spectrum (from P32)
    and whose χ-twisted weighted spectral trace reproduces
    :math:`Z_{\mathrm{TNFR}}(s,\chi)`.

    Parameters
    ----------
    chi : DirichletCharacter
        Character defining the twist.
    n_primes : int
        Number of primes (ignored if ``primes`` provided).  Primes
        dividing the modulus are counted in this total but excluded
        from the spectrum and graph.
    max_power : int, default 8
        REMESH echo cap.
    coupling : float, default 0.0
        Ladder coupling strength.  ``0.0`` gives the exact diagonal
        spectrum; non-zero values produce perturbed spectra.
    primes : sequence of int, optional
        Explicit prime list.

    Returns
    -------
    TwistedPrimeLadderHamiltonian
        Bundle containing the graph, the Hamiltonian, the χ-twisted
        weight operator, the reference spectrum, the coupling value
        and the character identifiers.
    """
    G = build_twisted_prime_ladder_graph(
        chi,
        n_primes,
        max_power=max_power,
        coupling=coupling,
        primes=primes,
    )
    H = InternalHamiltonian(G)
    W_chi = build_twisted_prime_ladder_weight_operator(chi, G)
    spectrum = build_twisted_prime_ladder_spectrum(
        chi,
        n_primes,
        max_power=max_power,
        primes=primes,
    )
    return TwistedPrimeLadderHamiltonian(
        graph=G,
        hamiltonian=H,
        weight_operator=W_chi,
        spectrum=spectrum,
        coupling=float(coupling),
        character_modulus=int(chi.modulus),
        character_name=str(chi.name),
    )


# ---------------------------------------------------------------------------
# Spectral observables
# ---------------------------------------------------------------------------

def twisted_weighted_spectral_trace(
    H_freq: np.ndarray,
    W_chi: np.ndarray,
    s: complex,
) -> complex:
    r"""χ-twisted weighted spectral trace
    :math:`\mathrm{Tr}(\hat W^{(\chi)} e^{-s \hat H_{\mathrm{freq}}})`.

    For a diagonal Hamiltonian (decoupled prime ladders), this reduces
    to :math:`\sum_n W^{(\chi)}_{nn} e^{-s E_n}`, which equals the
    TNFR χ-twisted weighted Dirichlet trace
    :math:`Z_{\mathrm{TNFR}}(s,\chi)` from
    :func:`tnfr.riemann.dirichlet_l.tnfr_log_l_derivative`.

    For a perturbed Hamiltonian (``coupling != 0``), it evaluates
    :math:`\mathrm{Tr}(\hat W^{(\chi)} e^{-s \hat H_{\mathrm{int}}})`
    via the spectral decomposition of :math:`\hat H_{\mathrm{int}}` —
    see :meth:`InternalHamiltonian.get_spectrum`.  Since the
    Hamiltonian is Hermitian (real symmetric in the decoupled
    limit, Hermitian under generic real coupling) the eigenvectors
    are orthonormal and the formula

    .. math::

        \mathrm{Tr}(\hat W^{(\chi)} e^{-s \hat H}) =
            \sum_n \langle \phi_n | \hat W^{(\chi)} | \phi_n \rangle
            \, e^{-s E_n}

    holds exactly regardless of the (complex) non-Hermiticity of
    :math:`\hat W^{(\chi)}`.

    Parameters
    ----------
    H_freq : numpy.ndarray
        Hamiltonian (or its frequency component) — must be Hermitian.
    W_chi : numpy.ndarray
        Diagonal χ-twisted weight operator (complex in general).
    s : complex
        Spectral parameter.  Convergence requires
        :math:`\mathrm{Re}(s) > 1` in the infinite-prime limit; in
        the finite-dimensional model it is well-defined for all
        :math:`s \in \mathbb{C}`.

    Returns
    -------
    complex
        :math:`\mathrm{Tr}(\hat W^{(\chi)} e^{-s \hat H_{\mathrm{freq}}})`.
    """
    eigvals, eigvecs = np.linalg.eigh(H_freq)
    s_c = complex(s)
    exp_minus_sE = np.exp(-s_c * eigvals)
    # Diagonal entries of U^H W_chi U in the eigenbasis.
    W_eig = np.einsum("ij,jk,ki->i", eigvecs.conj().T, W_chi, eigvecs)
    z = np.sum(W_eig * exp_minus_sE)
    return complex(z)


# ---------------------------------------------------------------------------
# Certificate: Hamiltonian reproduces the χ-twisted prime-ladder data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwistedPrimeLadderHamiltonianCertificate:
    r"""Numerical certificate that the Hamiltonian reproduces the P32 χ-twisted spectrum.

    Attributes
    ----------
    character_modulus : int
        Conductor :math:`q` of the character.
    character_name : str
        Label of the character.
    n_primes_seed : int
        Number of primes initially requested at the seed level.
    n_active : int
        Number of primes actually kept (those coprime to :math:`q`).
    n_excluded : int
        Number of primes dropped (those dividing :math:`q`).
    max_power : int
        REMESH echo cap :math:`K`.
    coupling : float
        Coupling strength used at construction.
    hilbert_dim : int
        :math:`N` = total Hilbert-space dimension =
        :math:`n_{\mathrm{active}} \times K`.
    is_hermitian : bool
        Whether :math:`\hat H` passed the Hermiticity check.  Always
        ``True`` for a successfully-constructed bundle (the
        constructor raises otherwise).  Note this refers to
        :math:`\hat H`, NOT to :math:`\hat W^{(\chi)}` — the latter
        is Hermitian only for real characters.
    spectrum_max_abs_error : float
        :math:`\max_n |E_n^{\mathrm{Ham}} - E_n^{\mathrm{ladder}}|`
        between the sorted Hamiltonian eigenvalues and the sorted
        χ-twisted prime-ladder eigenvalues from P32.  At
        ``coupling = 0`` this is zero to machine precision.
    spectrum_reproduced : bool
        ``spectrum_max_abs_error <= spectrum_tol``.
    s_values : numpy.ndarray
        Complex spectral parameters at which the χ-twisted weighted
        trace was compared.
    trace_max_rel_error : float
        Worst-case relative error
        :math:`\max_s |Z_H(s,\chi) - Z_{\mathrm{TNFR}}(s,\chi)| /
        |Z_{\mathrm{TNFR}}(s,\chi)|`
        between the Hamiltonian trace and the reference χ-twisted
        prime-ladder trace from P32.  At ``coupling = 0`` this is at
        the floating-point round-off level.
    trace_reproduced : bool
        ``trace_max_rel_error <= trace_tol``.
    overall_ok : bool
        Both spectrum and trace reproduction succeeded.

    Notes
    -----
    Failure of either reproduction at ``coupling = 0`` indicates a
    construction bug.  Failure at ``coupling != 0`` is **expected**
    and quantifies how strongly inter-ladder coupling deforms the
    χ-twisted prime-ladder spectrum.
    """

    character_modulus: int
    character_name: str
    n_primes_seed: int
    n_active: int
    n_excluded: int
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


def verify_twisted_hamiltonian_reproduces_prime_ladder(
    bundle: TwistedPrimeLadderHamiltonian,
    s_values: Sequence[complex] = (2.0, 3.0, 2.0 + 1.0j, 3.0 + 2.0j, 5.0),
    *,
    n_primes_seed: int | None = None,
    spectrum_tol: float = 1e-10,
    trace_tol: float = 1e-10,
) -> TwistedPrimeLadderHamiltonianCertificate:
    r"""Verify that the χ-twisted Hamiltonian reproduces spectrum and trace.

    Two checks are performed:

    1. **Spectrum reproduction.** Compute the sorted eigenvalues of
       ``bundle.hamiltonian.H_int`` and compare with the sorted
       ``bundle.spectrum.eigenvalues`` (from P32).  At
       ``coupling = 0`` these must agree to machine precision.

    2. **χ-twisted weighted trace reproduction.** Compute
       :math:`\mathrm{Tr}(\hat W^{(\chi)} e^{-s\hat H_{\mathrm{int}}})`
       via :func:`twisted_weighted_spectral_trace` and compare with
       :func:`tnfr.riemann.dirichlet_l.tnfr_log_l_derivative` at each
       ``s`` in ``s_values``.

    Parameters
    ----------
    bundle : TwistedPrimeLadderHamiltonian
        Output of :func:`build_twisted_prime_ladder_hamiltonian`.
    s_values : sequence of complex, default
        ``(2.0, 3.0, 2.0+1j, 3.0+2j, 5.0)``
        Complex spectral parameters at which to compare the χ-twisted
        weighted trace.  All values should satisfy
        :math:`\mathrm{Re}(s) > 1` for clean comparison with the
        convergent classical regime.
    n_primes_seed : int, optional
        Seed prime count used to build the bundle (purely for
        reporting in the certificate; if ``None`` it defaults to
        ``n_active + n_excluded``).
    spectrum_tol : float, default 1e-10
        Maximum allowed absolute deviation between the two spectra.
    trace_tol : float, default 1e-10
        Maximum allowed relative deviation between the two traces.

    Returns
    -------
    TwistedPrimeLadderHamiltonianCertificate
        Numerical certificate documenting both checks.
    """
    H = bundle.hamiltonian
    W_chi = bundle.weight_operator
    spectrum = bundle.spectrum

    # --- Spectrum check ---
    eigvals_ham, _ = H.get_spectrum()
    eigvals_ham_sorted = np.sort(np.real(eigvals_ham))
    eigvals_ref_sorted = np.sort(spectrum.eigenvalues)
    spectrum_abs_error = float(
        np.max(np.abs(eigvals_ham_sorted - eigvals_ref_sorted))
    )
    spectrum_ok = spectrum_abs_error <= spectrum_tol

    # --- χ-twisted weighted trace check ---
    s_arr = np.asarray(list(s_values), dtype=complex)
    z_ham = np.empty(s_arr.size, dtype=complex)
    z_ref = np.empty(s_arr.size, dtype=complex)
    for i, s in enumerate(s_arr):
        z_ham[i] = twisted_weighted_spectral_trace(H.H_int, W_chi, complex(s))
        z_ref[i] = complex(tnfr_log_l_derivative(spectrum, complex(s)))
    abs_err = np.abs(z_ham - z_ref)
    rel_err = abs_err / np.maximum(np.abs(z_ref), 1e-300)
    trace_rel_error = float(np.max(rel_err))
    trace_ok = trace_rel_error <= trace_tol

    n_active = int(spectrum.n_active)
    n_excluded = int(spectrum.n_excluded)
    seed = (
        int(n_primes_seed)
        if n_primes_seed is not None
        else n_active + n_excluded
    )

    return TwistedPrimeLadderHamiltonianCertificate(
        character_modulus=bundle.character_modulus,
        character_name=bundle.character_name,
        n_primes_seed=seed,
        n_active=n_active,
        n_excluded=n_excluded,
        max_power=spectrum.max_power,
        coupling=bundle.coupling,
        hilbert_dim=H.N,
        is_hermitian=True,
        spectrum_max_abs_error=spectrum_abs_error,
        spectrum_reproduced=spectrum_ok,
        s_values=s_arr,
        trace_max_rel_error=trace_rel_error,
        trace_reproduced=trace_ok,
        overall_ok=bool(spectrum_ok and trace_ok),
    )
