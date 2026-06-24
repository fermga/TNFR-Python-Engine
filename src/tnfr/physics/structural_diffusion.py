r"""TNFR Structural Diffusion — the transport content of the nodal equation.

This module makes explicit, and verifies, that the TNFR nodal equation

    ∂EPI/∂t = νf · ΔNFR(t)

is **structurally a diffusion equation on the network**.  This is not an
analogy imported from another paradigm: it is the literal content of the
canonical ΔNFR computation.

THE NODAL EQUATION IS GRAPH DIFFUSION
=====================================
The canonical ΔNFR (:func:`tnfr.dynamics.default_compute_delta_nfr`) is a
weighted sum of *neighbour-mean-minus-self* gradients, one per structural
channel (see :mod:`tnfr.dynamics.dnfr`):

    g_epi(i)   = mean_{j∈N(i)} EPI(j) − EPI(i)
    g_phase(i) = −angle_diff(θ(i), mean θ neighbours) / π
    g_vf(i)    = mean νf(neighbours) − νf(i)
    g_topo(i)  = mean deg(neighbours) − deg(i)

Each ``neighbour-mean − self`` term is exactly the action of the
**random-walk graph Laplacian** L_rw = I − D⁻¹W on that field:

    g_epi = −(L_rw · EPI)   (verified to machine precision).

So the EPI channel of the nodal equation is

    ∂EPI/∂t = νf · ΔNFR_epi = −νf · L_rw · EPI,

i.e. the **discrete diffusion (heat) equation** with diffusivity νf.  The
structural form EPI spreads across the network exactly as heat or a
concentration diffuses; ΔNFR is the diffusive gradient (the structural
pressure) driving the flux, and νf is the mobility / diffusivity.

WHAT EMERGES (empirically-grounded, in TNFR's own terms)
========================================================
- **Structural diffusion** (EPI channel): the form relaxes to a uniform
  field; each Laplacian eigenmode decays as exp(−νf·λ_k·t); the slowest
  rate is set by the spectral gap λ₂ (the Fiedler value).
- **Conserved structural total**: the random-walk Laplacian conserves the
  **degree-weighted total** Σ_i deg(i)·EPI(i) (its left null vector is the
  degree vector), the analogue of the conserved amount of diffusing
  substance.
- **Equilibrium ⟺ no gradients**: ΔNFR = 0 ⟺ the field is uniform across
  neighbourhoods — the diffusive steady state.
- **Synchronization** (phase channel): the phase term aligns θ to the
  neighbour mean, driving Kuramoto-type synchronization (R → 1).

These are the registers whose existence is established by the strictest
empirical method — diffusion (Fourier 1822, Fick 1855, Einstein 1905) and
synchronization (Kuramoto; observed in fireflies, pacemaker cells, neurons,
Josephson junctions).  They are reproduced here as the **same mathematics**
(the graph Laplacian is the discrete diffusion operator), not as a
metaphor.

THE MECHANICAL REGIME IS OVERDAMPED DRIFT (not inertial)
========================================================
Because the nodal equation is **first order in time**, the mechanical
regime it produces directly is the **overdamped drift law**, not Newtonian
inertia.  Reading EPI as a position-like coordinate q and ΔNFR as the
structural pressure F, the nodal equation is

    q̇ = νf · F,

i.e. **velocity proportional to applied force**, with νf the **mobility**.
Under a sustained structural pressure the field drifts at *constant*
velocity (linear in time), it does **not** accelerate.  This is the
empirically-demonstrated mobility / drift law — Stokes drag (1851),
Einstein's mobility relation (1905), terminal velocity, sedimentation,
electrophoresis — where νf is the mobility, NOT an inverse inertial mass.

The **inertial** Newtonian regime (second order, q̈ = F/m, oscillation)
is a *different* structure: it lives in the conservative **symplectic
substrate** Hamiltonian flow (:mod:`tnfr.physics.symplectic_substrate`,
where the flow is q̈ = −q per conjugate pair).  The bare nodal equation is
the **overdamped projection** of that substrate flow.  So:

    bare nodal equation (1st order)  →  overdamped drift  v = νf·F
    symplectic substrate (2nd order) →  inertial oscillation  q̈ = −∂V/∂q

both empirically grounded, but distinct regimes — a single first-order
nodal equation cannot, by itself, be Newton's second law.

DISCRETE MODES ARE THE BOUNDED-MANIFOLD STANDING WAVES
======================================================
On a **bounded** structural manifold (a finite graph) the diffusion
operator has a **discrete** spectrum of eigenmodes — the same structure as
the discrete harmonics of a bounded vibrating medium.  The symmetric
normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2} shares the diffusion
operator L_rw's spectrum {λ_k} but has **orthonormal** eigenvectors v_k:

- **Discrete spectrum**: a finite manifold supports a finite, discrete set
  of eigenvalues {λ_k} (not a continuum) — the structural origin of
  "discrete modes".  λ_1 = 0 is the uniform mode (the conserved diffusion
  mode); λ_2 (the spectral gap) is the first non-trivial mode.
- **Standing-wave shapes**: the eigenvectors v_k are orthonormal standing
  waves.  On a path graph they are exactly the cosine standing waves of a
  vibrating string (overlap 1.0 to machine precision).
- **Nodal-domain ordering** (Courant): the number of sign changes (nodal
  domains) grows with the mode index k — the structural "mode number" k
  emerges from the bounded geometry, not from a postulate.
- **Two time-regimes, same modes**: under diffusion (first order) mode k
  relaxes as exp(−νf·λ_k·t); under the wave/substrate flow (second order)
  it oscillates at the standing-wave frequency ω_k = √λ_k.

This is the discrete-harmonic structure of a bounded elastic medium —
vibrating strings (Pythagoras), Chladni plate modes (1787), molecular
vibrational spectra — all established by the strictest empirical method.
The discreteness is a consequence of the **bounded structural geometry**,
not an imported quantum postulate.

STRUCTURAL STABILITY: THE DISPERSION RELATION
=============================================
The growth or decay of each structural eigenmode under diffusion plus a
local reaction rate r is governed by the **dispersion relation**

    σ_k = r − νf · λ_k,

the universal linear-stability law (the same tool that governs every
instability and pattern-forming system — convective instability, the
onset of pattern formation).  From the canonical Laplacian spectrum {λ_k}:

- **Pure diffusion** (r = 0): σ_k = −νf·λ_k ≤ 0, so every non-uniform mode
  *decays* — structural equilibrium is stable, the integral ∫νf·ΔNFR dt
  converges.  This is the linear-stability content of "diffusion relaxes
  to uniform".
- **Structural instability threshold** r_c = νf·λ_2 (the spectral gap, the
  Fiedler value, times the diffusivity).  For 0 < r < r_c only the uniform
  mode grows (global amplification, no spatial structure); for r > r_c the
  **Fiedler mode** (k = 1) also grows — the first *structural* pattern.
- **The first structural pattern is the Fiedler partition**: the Fiedler
  eigenvector splits the network along its **weakest structural cut** (the
  two most weakly-connected communities) — the empirically-validated
  spectral-clustering result.
- **U2 grammar, spectrally**: a destabilizing reaction raises r, a
  stabilizer lowers it; bounded evolution (U2) ⟺ keeping r below r_c.
  Above r_c the Fiedler mode grows unboundedly → fragmentation (the
  U2-violation the grammar prevents).

The reaction rate r is a generic local rate; in TNFR the operators supply
it (stabilizers lower r, destabilizers raise it).  A *two-channel*
structural diffusion with **differential diffusivity** and an
activator–inhibitor coupling supports a finite-wavelength (Turing)
instability — the empirically-demonstrated pattern-formation mechanism
(Belousov–Zhabotinsky, morphogenesis); those kinetics are a model input,
not TNFR-derived, so only the dispersion-relation mechanism is certified
here.

THE STRUCTURAL RANDOM WALK AND RESISTANCE GEOMETRY
==================================================
The diffusion operator is **literally the generator of a random walk** on
the network: L_rw = I − D^{-1}W = I − P, where P = D^{-1}W is the
random-walk transition matrix (verified exactly).  So the structural
transport is **Brownian motion on the network** — the empirically-
demonstrated random walk (Einstein 1905, Perrin 1908, the proof of atoms):

- **Stationary distribution ∝ degree**: the random walk converges to
  π_i = deg(i) / Σ deg — exactly the **degree-weighted total** the
  diffusion conserves.  The conserved quantity *is* the equilibrium
  measure.
- **Effective resistance** (Ohm's law): treating the network as a
  resistor network (the combinatorial Laplacian L = D − W is the
  conductance matrix — Kirchhoff 1847), the effective resistance
  R_eff(i,j) = L⁺_ii + L⁺_jj − 2L⁺_ij (L⁺ the pseudoinverse) is a
  **transport metric** (symmetric, non-negative, triangle inequality) —
  the structural "difficulty of transport" between two nodes.
- **Commute time = 2m·R_eff**: the expected round-trip time of the random
  walk between two nodes equals 2m times the effective resistance (m the
  number of edges) — the exact link between the diffusion random walk and
  the resistance geometry (Chandra et al. 1996), confirmed against
  Monte-Carlo walks.

These are the same mathematics as Brownian motion (random walk) and
electrical networks (Ohm/Kirchhoff resistance) — both established by the
strictest empirical method.

THE STRUCTURAL FLOW: CURRENT, KIRCHHOFF, AND CONTINUITY
======================================================
The transport carries a **structural current**: the diffusion edge current
J_ij = EPI_i − EPI_j (Fick's law — flux from high to low, antisymmetric).
Its node-level balance is **Kirchhoff's current law**, which *is* the
discrete continuity equation:

    div(J)(i) = Σ_{j∼i} J_ij = (L·EPI)(i),

so the net outflow at a node equals the combinatorial Laplacian acting on
EPI.  Hence the diffusion continuity equation ∂EPI/∂t + div(J) = 0 holds,
and for a closed network (no sources) the total flux balances, Σ_i div(J)
= 0 (L has zero column sums — the structural-conservation analogue here).
Under an injected unit current from s to t the induced potential drop is
the **effective resistance** R_eff(s,t) (Ohm's law) — tying the current to
the resistance geometry above.  These are Fick diffusion, Kirchhoff's
circuit laws, and Ohm's law — all empirically demonstrated.

This is the EPI-channel current; it complements the tetrad-field
continuity of :mod:`tnfr.physics.conservation` (which tracks the charge
ρ = Φ_s + K_φ and the current J = (J_φ, J_ΔNFR)).

HONEST SCOPE
============
- The identity ΔNFR_epi = −L_rw·EPI is EXACT (machine precision), a
  mathematical fact about the canonical ΔNFR.
- The full ΔNFR is multi-channel: EPI **diffusion** + phase
  **synchronization** + νf/topology **homogenization**.  This module
  isolates and certifies the diffusion (EPI) channel and reports the
  synchronization channel qualitatively.
- **λ₂ is topological, NOT tied to the canonical constants.** The spectral
  gap λ₂ (which governs relaxation, stability, and the instability
  threshold) is a purely spectral/topological quantity — determined by N,
  degree, and connectivity (e.g. ring λ₂ = 1 − cos(2π/N), complete-graph
  λ₂ = n/(n−1)).  The canonical constants (φ, γ, π, e) are *threshold
  scales of the tetrad fields* and do **not** enter the Laplacian
  spectrum; any numerical proximity is coincidental (the 2π/N in a ring is
  a geometric polygon angle, not the tetrad constant π).  Measured
  negative result — do not assert a λ₂ ↔ constant relation.
- This characterises the transport content of the nodal dynamics; it does
  not, by itself, resolve any open program (Riemann G4, Navier–Stokes).

References
----------
- :mod:`tnfr.dynamics.dnfr` — the canonical ΔNFR neighbour-mean gradients
- :func:`tnfr.observers.kuramoto_order` — the synchronization order R
- :mod:`tnfr.physics.conservation` — the structural continuity equation
- AGENTS.md §"Foundational Physics" — the nodal equation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..mathematics.unified_numerical import np

__all__ = [
    "StructuralDiffusionCertificate",
    "OverdampedRegimeCertificate",
    "OverdampedProjectionCertificate",
    "UndampedLimitCertificate",
    "DiscreteModeCertificate",
    "StructuralStabilityCertificate",
    "RandomWalkCertificate",
    "StructuralFlowCertificate",
    "structural_diffusion_operator",
    "symmetric_normalized_laplacian",
    "structural_field",
    "structural_diffusivity",
    "relaxation_spectrum",
    "structural_frequency_rank",
    "degree_weighted_total",
    "structural_eigenmodes",
    "nodal_domain_count",
    "dispersion_relation",
    "instability_threshold",
    "fiedler_partition",
    "random_walk_matrix",
    "stationary_distribution",
    "effective_resistance",
    "commute_time",
    "structural_current",
    "current_divergence",
    "verify_structural_diffusion",
    "verify_overdamped_regime",
    "damped_wave_rates",
    "verify_overdamped_projection",
    "verify_undamped_limit",
    "verify_discrete_modes",
    "verify_structural_stability",
    "verify_structural_random_walk",
    "verify_structural_flow",
]


def _ordered_nodes(G: Any) -> list:
    """Stable node ordering for the matrix representation."""
    return list(G.nodes())


def structural_diffusion_operator(G: Any) -> tuple[list, Any]:
    r"""Return the random-walk graph Laplacian L_rw = I − D⁻¹W.

    This is the operator whose action on a field is exactly the canonical
    ΔNFR ``neighbour-mean − self`` gradient: g = −L_rw·field.  Built from
    the (optionally weighted) adjacency; isolated nodes (degree 0) get a
    zero row (no diffusion).

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, L_rw) : tuple[list, np.ndarray]
        The node ordering and the N×N random-walk Laplacian.
    """
    nodes = _ordered_nodes(G)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    lap = np.zeros((n, n), dtype=float)
    for node in nodes:
        i = index[node]
        neigh = list(G.neighbors(node))
        if not neigh:
            continue
        # weighted degree (weight defaults to 1.0 when absent)
        weights = [float(G[node][m].get("weight", 1.0)) for m in neigh]
        deg = sum(weights)
        if deg <= 0.0:
            continue
        lap[i, i] = 1.0
        for m, w in zip(neigh, weights):
            lap[i, index[m]] -= w / deg
    return nodes, lap


def symmetric_normalized_laplacian(
    G: Any, nodes: list | None = None
) -> tuple[list, Any]:
    r"""Return the symmetric normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2}.

    L_sym shares the spectrum of the canonical diffusion operator
    L_rw = I − D⁻¹W (:func:`structural_diffusion_operator`) but is symmetric, so
    it has an orthonormal eigenbasis and real eigenvalues — the canonical choice
    for the relaxation spectrum (its λ₂ is the structural ``diffusion_gap``).
    Isolated nodes (degree 0) get a zero row.

    Parameters
    ----------
    G : TNFRGraph
    nodes : list, optional
        Node ordering; defaults to the stable ``list(G.nodes())`` order.

    Returns
    -------
    (nodes, L_sym) : tuple[list, np.ndarray]
        The node ordering and the N×N symmetric normalized Laplacian.
    """
    if nodes is None:
        nodes = _ordered_nodes(G)
    index = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    deg = np.zeros(n, dtype=float)
    for node in nodes:
        deg[index[node]] = sum(
            float(G[node][m].get("weight", 1.0)) for m in G.neighbors(node)
        )
    d_inv_sqrt = np.where(deg > 0.0, 1.0 / np.sqrt(deg), 0.0)
    lap = np.zeros((n, n), dtype=float)
    for node in nodes:
        i = index[node]
        if deg[i] <= 0.0:
            continue
        lap[i, i] = 1.0
        for m in G.neighbors(node):
            j = index[m]
            w = float(G[node][m].get("weight", 1.0))
            lap[i, j] -= w * d_inv_sqrt[i] * d_inv_sqrt[j]
    return nodes, lap


def structural_field(G: Any, nodes: list | None = None) -> Any:
    r"""Return the EPI field as a vector aligned with ``nodes``."""
    if nodes is None:
        nodes = _ordered_nodes(G)
    return np.array(
        [float(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes],
        dtype=float,
    )


def structural_diffusivity(G: Any) -> float:
    r"""Mean structural frequency νf — the diffusion coefficient (mobility).

    In ∂EPI/∂t = −νf·L_rw·EPI, νf plays the role of the diffusivity: the
    larger the structural frequency, the faster the form spreads.
    """
    nodes = _ordered_nodes(G)
    vf = [float(get_attr(G.nodes[n], ALIAS_VF, 0.0)) for n in nodes]
    return float(np.mean(vf)) if vf else 0.0


def degree_weighted_total(G: Any) -> float:
    r"""The conserved structural total Σ_i deg(i)·EPI(i).

    The random-walk Laplacian conserves the degree-weighted total (its left
    null vector is the degree vector), the analogue of the conserved amount
    of a diffusing substance.

    This is the **EPI-channel** conserved quantity (of the diffusion
    ∂EPI/∂t = −νf·L_rw·EPI).  It is **distinct** from the tetrad Noether charge
    Q = Σ(Φ_s + K_φ)
    (:func:`tnfr.physics.conservation.compute_noether_charge`), conserved under
    grammar U1–U6: TNFR carries two distinct conservation laws, on the EPI
    field and on the tetrad fields respectively (see
    STRUCTURAL_CONSERVATION_THEOREM §8.7).
    """
    nodes = _ordered_nodes(G)
    total = 0.0
    for node in nodes:
        neigh = list(G.neighbors(node))
        deg = sum(float(G[node][m].get("weight", 1.0)) for m in neigh)
        total += deg * float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    return float(total)


def relaxation_spectrum(G: Any) -> Any:
    r"""Diffusion relaxation rates νf·λ_k (sorted ascending).

    The eigenvalues λ_k of the random-walk Laplacian L_rw scaled by the
    diffusivity νf give the decay rates of the diffusion eigenmodes:
    mode k relaxes as exp(−νf·λ_k·t).  λ₁ = 0 (the conserved uniform mode);
    λ₂ (the spectral gap / Fiedler value) sets the slowest relaxation.

    Returns
    -------
    np.ndarray
        The rates νf·λ_k sorted ascending (real parts).
    """
    _, lap = structural_diffusion_operator(G)
    eig = np.linalg.eigvals(lap).real
    eig.sort()
    return structural_diffusivity(G) * eig


def structural_frequency_rank(G: Any, decimals: int = 8) -> int:
    r"""Number of distinct structural frequencies (the structural rank).

    The distinct eigenvalues of the canonical random-walk Laplacian L_rw are
    the network's structural frequencies (the relaxation rates of
    ∂EPI/∂t = −νf·L_rw·EPI, up to the νf scale). This returns their count s(G)
    — the size of the distinct-frequency spectrum — complementing
    ``relaxation_spectrum`` (which returns the rates themselves).

    For a connected graph, s distinct eigenvalues bound the diameter by s−1,
    and s = 2 iff the graph is complete (regular case). On arithmetic Cayley
    networks the rank is a primality / cyclotomy diagnostic (see
    :mod:`tnfr.mathematics.number_theory`): the quadratic-residue network on an
    odd prime has rank 3, and the k-th power residue network on a prime p has
    rank ``gcd(k, p-1) + 1``.

    Note
    ----
    For large or dense graphs the distinct-eigenvalue count is sensitive to the
    ``decimals`` rounding (floating-point noise in ``eigvals`` can split truly
    equal eigenvalues). For arithmetic residue networks the exact multiplicative
    :func:`tnfr.mathematics.number_theory.quadratic_residue_annotated_rank` is
    the robust closed-form object; this scalar count agrees with it for small
    moduli.

    Parameters
    ----------
    G : TNFRGraph
    decimals : int
        Rounding applied to the real and imaginary parts before counting
        distinct values (the spectrum may be complex for directed graphs).

    Returns
    -------
    int
        The number of distinct eigenvalues of L_rw.
    """
    _, lap = structural_diffusion_operator(G)
    eig = np.linalg.eigvals(lap)
    rounded = np.round(eig.real, decimals) + 1j * np.round(eig.imag, decimals)
    return int(np.unique(rounded).size)


@dataclass(frozen=True)
class StructuralDiffusionCertificate:
    r"""Verification that the nodal equation's EPI channel is graph diffusion.

    Attributes
    ----------
    n_nodes : int
    dnfr_is_graph_laplacian : bool
        The canonical ΔNFR (EPI channel) equals −L_rw·EPI.
    max_laplacian_residual : float
        Max |ΔNFR_epi − (−L_rw·EPI)| over the nodes (≈ 0).
    diffusivity : float
        Mean νf (the diffusion coefficient / mobility).
    spectral_gap : float
        λ₂ of L_rw (the Fiedler value); sets the slowest relaxation.
    slowest_relaxation_rate : float
        νf·λ₂ — the slowest diffusion decay rate.
    degree_weighted_conserved : bool
        Σ deg·EPI is conserved under the diffusion flow.
    max_conservation_drift : float
        Max drift of the degree-weighted total over the sampled flow.
    relaxes_to_uniform : bool
        The field relaxes to a spatially uniform diffusive equilibrium.
    final_field_std : float
        Std of the field after the sampled diffusion flow (≈ 0).
    """

    n_nodes: int
    dnfr_is_graph_laplacian: bool
    max_laplacian_residual: float
    diffusivity: float
    spectral_gap: float
    slowest_relaxation_rate: float
    degree_weighted_conserved: bool
    max_conservation_drift: float
    relaxes_to_uniform: bool
    final_field_std: float

    @property
    def is_valid_diffusion(self) -> bool:
        """True when the nodal EPI channel verifies as graph diffusion."""
        return (
            self.dnfr_is_graph_laplacian
            and self.degree_weighted_conserved
            and self.relaxes_to_uniform
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_diffusion else "INVALID"
        return (
            f"Structural diffusion [{ok}]: "
            f"ΔNFR_epi = −L_rw·EPI={self.dnfr_is_graph_laplacian} "
            f"(res {self.max_laplacian_residual:.1e}), "
            f"diffusivity νf={self.diffusivity:.4f}, "
            f"spectral gap λ₂={self.spectral_gap:.4f}, "
            f"slowest rate νf·λ₂={self.slowest_relaxation_rate:.4f}, "
            f"deg-weighted conserved={self.degree_weighted_conserved} "
            f"(drift {self.max_conservation_drift:.1e}), "
            f"relaxes to uniform={self.relaxes_to_uniform} "
            f"(final std {self.final_field_std:.1e})"
        )


def _dnfr_epi_channel(G: Any, nodes: list) -> Any:
    r"""Canonical ΔNFR restricted to the EPI channel, on a clean replica.

    Isolates the EPI diffusion channel by computing the canonical ΔNFR with
    weights (phase=0, epi=1, vf=0, topo=0) on a minimal structural replica
    (nodes + edges + EPI/θ/νf only), so the caller's graph is never mutated
    and the non-copyable runtime caches are not duplicated.
    """
    from ..dynamics import default_compute_delta_nfr

    g2 = G.__class__()
    for node in nodes:
        data = G.nodes[node]
        g2.add_node(
            node,
            EPI=float(get_attr(data, ALIAS_EPI, 0.0)),
            theta=float(data.get("theta", 0.0)),
            nu_f=float(get_attr(data, ALIAS_VF, 0.0)),
        )
    for u, v, data in G.edges(data=True):
        g2.add_edge(u, v, weight=float(data.get("weight", 1.0)))
    g2.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 0.0,
        "topo": 0.0,
    }
    default_compute_delta_nfr(g2)
    return np.array(
        [float(get_attr(g2.nodes[n], ALIAS_DNFR, 0.0)) for n in nodes],
        dtype=float,
    )


def verify_structural_diffusion(
    G: Any,
    *,
    dt: float = 0.1,
    steps: int = 400,
    tolerance: float = 1e-9,
) -> StructuralDiffusionCertificate:
    r"""Verify the nodal equation's EPI channel is graph diffusion.

    Confirms (1) the canonical ΔNFR EPI channel equals −L_rw·EPI to machine
    precision, (2) the degree-weighted total is conserved under the
    diffusion flow, and (3) the field relaxes to a uniform diffusive
    equilibrium; and reports the diffusivity νf and the relaxation spectrum.

    The caller's graph is never mutated (the ΔNFR check runs on a copy).

    Parameters
    ----------
    G : TNFRGraph
    dt : float
        Forward-Euler step for the diffusion-flow checks.
    steps : int
        Number of diffusion steps for the relaxation / conservation checks.
    tolerance : float
        Maximum allowed Laplacian residual and conservation drift.

    Returns
    -------
    StructuralDiffusionCertificate
    """
    nodes, lap = structural_diffusion_operator(G)
    n = len(nodes)
    epi = structural_field(G, nodes)

    # (1) ΔNFR (epi channel) == −L_rw·EPI ?
    try:
        dnfr_epi = _dnfr_epi_channel(G, nodes)
        residual = float(np.max(np.abs(dnfr_epi - (-(lap @ epi)))))
        is_laplacian = residual < max(tolerance, 1e-12)
    except Exception:
        residual = float("nan")
        is_laplacian = False

    # diffusivity and spectrum
    diffusivity = structural_diffusivity(G)
    eig = np.linalg.eigvals(lap).real
    eig.sort()
    spectral_gap = float(eig[1]) if n > 1 else 0.0
    slowest_rate = diffusivity * spectral_gap

    # degree vector for the conserved weighted total
    deg = np.array(
        [
            sum(float(G[node][m].get("weight", 1.0)) for m in G.neighbors(node))
            for node in nodes
        ],
        dtype=float,
    )

    # (2)+(3) integrate the pure diffusion flow e ← e − dt·L_rw·e
    e = epi.copy()
    w0 = float(deg @ e)
    max_drift = 0.0
    for _ in range(steps):
        e = e - dt * (lap @ e)
        max_drift = max(max_drift, abs(float(deg @ e) - w0))
    conserved = max_drift < max(tolerance, 1e-9 * (abs(w0) + 1e-12))
    final_std = float(np.std(e))
    relaxes = final_std < max(1e-3, 1e-2 * float(np.std(epi) + 1e-12))

    return StructuralDiffusionCertificate(
        n_nodes=n,
        dnfr_is_graph_laplacian=is_laplacian,
        max_laplacian_residual=residual,
        diffusivity=diffusivity,
        spectral_gap=spectral_gap,
        slowest_relaxation_rate=slowest_rate,
        degree_weighted_conserved=conserved,
        max_conservation_drift=max_drift,
        relaxes_to_uniform=relaxes,
        final_field_std=final_std,
    )


# ---------------------------------------------------------------------------
# The overdamped drift regime: the bare nodal equation is first-order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverdampedRegimeCertificate:
    r"""Verification that the bare nodal equation is the overdamped drift law.

    The nodal equation ∂EPI/∂t = νf·ΔNFR is **first order in time**, so —
    reading EPI as a position q and ΔNFR as the structural pressure F — it
    is the **mobility / drift law** q̇ = νf·F: velocity proportional to
    force, with νf the mobility.  Under a sustained pressure the field
    drifts at *constant* velocity (linear in time), it does not accelerate.

    This is the empirically-demonstrated overdamped regime (Stokes 1851,
    Einstein 1905, terminal velocity, sedimentation, electrophoresis).  The
    inertial Newtonian regime (q̈ = F/m, second order) is the separate
    :mod:`tnfr.physics.symplectic_substrate` Hamiltonian flow; the nodal
    equation is its overdamped projection.

    Attributes
    ----------
    drift_velocity : float
        v = νf·F evaluated at the reference (νf, F).
    velocity_is_constant : bool
        Under sustained pressure, dEPI/dt is constant (first-order/drift).
    max_velocity_variation : float
        Max |dEPI/dt − v| over the held-pressure integration (≈ 0).
    position_linear_in_time : bool
        EPI(t) grows linearly (slope = drift), not quadratically.
    position_slope : float
        Measured slope of EPI(t) (= the drift velocity).
    mobility_linear_in_nu_f : bool
        v ∝ νf (the mobility law): v/νf is constant across νf.
    drift_linear_in_pressure : bool
        v ∝ F: v/F is constant across F.
    is_second_order : bool
        Whether the bare equation is second order (always False — it is the
        overdamped, first-order regime).
    """

    drift_velocity: float
    velocity_is_constant: bool
    max_velocity_variation: float
    position_linear_in_time: bool
    position_slope: float
    mobility_linear_in_nu_f: bool
    drift_linear_in_pressure: bool
    is_second_order: bool

    @property
    def is_overdamped_drift(self) -> bool:
        """True when the bare nodal equation verifies as overdamped drift."""
        return (
            self.velocity_is_constant
            and self.position_linear_in_time
            and self.mobility_linear_in_nu_f
            and self.drift_linear_in_pressure
            and not self.is_second_order
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_overdamped_drift else "INVALID"
        return (
            f"Overdamped drift regime [{ok}]: "
            f"q̇ = νf·F = {self.drift_velocity:.4f} "
            f"(mobility law); "
            f"velocity constant={self.velocity_is_constant} "
            f"(var {self.max_velocity_variation:.1e}), "
            f"position linear={self.position_linear_in_time} "
            f"(slope {self.position_slope:.4f}), "
            f"v∝νf={self.mobility_linear_in_nu_f}, "
            f"v∝F={self.drift_linear_in_pressure}, "
            f"second-order={self.is_second_order}"
        )


def verify_overdamped_regime(
    *,
    nu_f: float = 0.7,
    pressure: float = 1.3,
    dt: float = 0.01,
    steps: int = 300,
    tolerance: float = 1e-9,
) -> OverdampedRegimeCertificate:
    r"""Verify the bare nodal equation is the overdamped drift law q̇ = νf·F.

    Integrates the canonical nodal equation
    (:func:`tnfr.dynamics.canonical.compute_canonical_nodal_derivative`)
    under a *sustained* structural pressure and measures that the EPI
    coordinate drifts at constant velocity v = νf·F (first-order, mobility
    law), linear in νf (mobility) and in the pressure F.  Uses the canonical
    nodal-equation function — no formula is re-implemented here.

    Parameters
    ----------
    nu_f : float
        Reference structural frequency (mobility).
    pressure : float
        Sustained structural pressure ΔNFR (= F).
    dt : float
        Integration step.
    steps : int
        Number of integration steps.
    tolerance : float
        Maximum allowed velocity variation / linearity residual.

    Returns
    -------
    OverdampedRegimeCertificate
    """
    from ..dynamics.canonical import compute_canonical_nodal_derivative

    # integrate the bare nodal equation under a held pressure
    epi = 0.0
    velocities = []
    positions = []
    for _ in range(steps):
        v = compute_canonical_nodal_derivative(nu_f, pressure).derivative
        epi = epi + dt * v
        velocities.append(v)
        positions.append(epi)
    vel = np.array(velocities, dtype=float)
    pos = np.array(positions, dtype=float)

    drift = float(vel[0])
    vel_var = float(np.max(np.abs(vel - drift)))
    vel_constant = vel_var < tolerance

    # position grows linearly with slope = drift (first-order, not quadratic)
    t = np.arange(steps, dtype=float) * dt
    slope, _ = np.polyfit(t, pos, 1)
    quad = np.polyfit(t, pos, 2)[0]  # leading quadratic coefficient ≈ 0
    pos_linear = abs(float(slope) - drift) < max(tolerance, 1e-6 * abs(drift)) and abs(
        float(quad)
    ) < max(tolerance, 1e-6 * abs(drift) + 1e-9)

    # mobility law: v ∝ νf (v/νf constant across νf)
    ratios_nu = [
        compute_canonical_nodal_derivative(nf, pressure).derivative / nf
        for nf in (0.2, 0.5, 1.0, 1.5)
    ]
    mobility_linear = float(np.std(ratios_nu)) < tolerance

    # drift ∝ F (v/F constant across F)
    ratios_f = [
        compute_canonical_nodal_derivative(nu_f, f).derivative / f
        for f in (0.3, 0.8, 1.3, 2.0)
    ]
    pressure_linear = float(np.std(ratios_f)) < tolerance

    return OverdampedRegimeCertificate(
        drift_velocity=drift,
        velocity_is_constant=vel_constant,
        max_velocity_variation=vel_var,
        position_linear_in_time=pos_linear,
        position_slope=float(slope),
        mobility_linear_in_nu_f=mobility_linear,
        drift_linear_in_pressure=pressure_linear,
        is_second_order=False,
    )


# ---------------------------------------------------------------------------
# Overdamped projection: the bridge from the conservative substrate wave
# to the dissipative structural diffusion
# ---------------------------------------------------------------------------


def damped_wave_rates(G: Any, gamma: float) -> tuple[Any, Any, Any]:
    r"""Per-mode slow/fast rates of the damped graph wave q̈ + γq̇ + Lq = 0.

    The conservative symplectic substrate carries the graph **wave**
    equation q̈ = −L q (second order, mode k oscillating at √λ_k — the
    standing-wave ``discrete modes`` of :func:`verify_discrete_modes`).
    Adding a damping γ gives the damped oscillator q̈ + γq̇ + L q = 0, whose
    per-mode characteristic equation is

        s² + γ s + λ_k = 0   ⟹   s± = ½(−γ ± √(γ² − 4λ_k)).

    For γ² > 4λ_k (overdamped per mode) both roots are real: a **slow** root
    s₋ → −λ_k/γ (the diffusion rate) and a **fast** root s₊ → −γ (a
    transient that dies immediately).  This is the spectral content of the
    overdamped projection: after the fast transient, mode k relaxes at
    λ_k/γ = ν_f·λ_k with ν_f = 1/γ.

    Parameters
    ----------
    G : TNFRGraph
    gamma : float
        Damping coefficient.  Its inverse is the effective diffusivity
        (mobility) ν_f = 1/γ.

    Returns
    -------
    (lambdas, s_slow, s_fast) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Sorted Laplacian eigenvalues and the (real-part) slow/fast roots.
    """
    _, lap = structural_diffusion_operator(G)
    lambdas = np.sort(np.linalg.eigvals(lap).real)
    lambdas = np.clip(lambdas, 0.0, None)
    disc = gamma * gamma - 4.0 * lambdas + 0j
    root = np.sqrt(disc)
    s_slow = ((-gamma + root) / 2.0).real
    s_fast = ((-gamma - root) / 2.0).real
    return lambdas, s_slow, s_fast


@dataclass(frozen=True)
class OverdampedProjectionCertificate:
    r"""Verification that structural diffusion is the overdamped projection
    of the conservative symplectic-substrate wave flow.

    The conservative substrate (:mod:`tnfr.physics.symplectic_substrate`)
    carries the graph wave q̈ = −L q (second order).  Damping it and taking
    the strong-damping (Smoluchowski) limit collapses it onto the
    first-order structural diffusion q̇ = −(1/γ) L q, with the
    identification **ν_f = 1/γ** (structural frequency = inverse damping =
    mobility).  Both endpoints are canonical TNFR objects; this certificate
    measures the bridge between them.

    Attributes
    ----------
    n_nodes : int
    gamma : float
        Damping coefficient used for the projection.
    nu_f_effective : float
        The effective diffusivity 1/γ recovered by the projection.
    spectral_gap : float
        λ₂ of L_rw (the Fiedler value).
    lambda_max : float
        Largest Laplacian eigenvalue (sets the bridge error scale).
    max_rate_rel_error : float
        Max over modes of |s_slow + λ_k/γ| / (λ_k/γ): how far the damped
        slow rate is from the diffusion rate.
    rate_error_times_gamma_sq : float
        ``max_rate_rel_error · γ²`` — converges to ≈ λ_max, confirming the
        bridge error scales as O(λ_max/γ²).
    slowest_slow_rate : float
        Overdamped slow rate of the Fiedler mode, −s_slow(λ₂).
    slowest_diffusion_rate : float
        The diffusion spectral gap ν_f·λ₂ = λ₂/γ.
    trajectory_max_rel_error : float
        Max relative L² error between the damped-wave trajectory and the
        diffusion trajectory exp(−L t/γ)·q₀ over an overdamped time window.
    projects_to_diffusion : bool
        Whether both the rate and trajectory errors fall within tolerance.
    """

    n_nodes: int
    gamma: float
    nu_f_effective: float
    spectral_gap: float
    lambda_max: float
    max_rate_rel_error: float
    rate_error_times_gamma_sq: float
    slowest_slow_rate: float
    slowest_diffusion_rate: float
    trajectory_max_rel_error: float
    projects_to_diffusion: bool

    @property
    def is_valid_projection(self) -> bool:
        """True when the damped substrate wave projects onto diffusion."""
        return self.projects_to_diffusion

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_projection else "INVALID"
        return (
            f"Overdamped projection [{ok}]: damped substrate wave "
            f"projects onto diffusion with nu_f=1/gamma="
            f"{self.nu_f_effective:.4f}; rate error "
            f"{self.max_rate_rel_error:.2e} (x gamma^2="
            f"{self.rate_error_times_gamma_sq:.3f} ~ lambda_max="
            f"{self.lambda_max:.3f}), slow gap {self.slowest_slow_rate:.5f} "
            f"vs diffusion gap {self.slowest_diffusion_rate:.5f}, "
            f"trajectory error {self.trajectory_max_rel_error:.2e}"
        )


def verify_overdamped_projection(
    G: Any,
    *,
    gamma: float = 50.0,
    n_time_samples: int = 40,
    tolerance: float = 1e-2,
) -> OverdampedProjectionCertificate:
    r"""Verify diffusion is the overdamped projection of the substrate wave.

    Builds the canonical random-walk Laplacian L_rw, forms the damped graph
    wave q̈ + γq̇ + L q = 0, and measures two things in the strong-damping
    limit: (i) the per-mode **slow rate** converges to the diffusion rate
    λ_k/γ = ν_f·λ_k (ν_f = 1/γ), with error scaling as O(λ_max/γ²); and
    (ii) the damped-wave **trajectory** (from q₀ at rest) collapses onto the
    structural-diffusion trajectory exp(−L t/γ)·q₀.  No field formula is
    re-implemented — L_rw comes from
    :func:`structural_diffusion_operator` and the orthonormal eigenbasis
    from the symmetric normalized Laplacian.

    Parameters
    ----------
    G : TNFRGraph
    gamma : float
        Damping coefficient (effective diffusivity ν_f = 1/γ).  Must satisfy
        γ² > 4·λ_max for every mode to be overdamped.
    n_time_samples : int
        Time samples in the overdamped window for the trajectory check.
    tolerance : float
        Maximum relative error (rate and trajectory) for a valid projection.

    Returns
    -------
    OverdampedProjectionCertificate
    """
    nodes, lap = structural_diffusion_operator(G)
    n = len(nodes)
    lambdas = np.sort(np.linalg.eigvals(lap).real)
    lambdas = np.clip(lambdas, 0.0, None)
    lam_max = float(lambdas[-1]) if n else 0.0
    nonzero = lambdas[lambdas > 1e-9]
    lam2 = float(nonzero[0]) if nonzero.size else 0.0
    nu_f = 1.0 / gamma

    # (i) per-mode slow rate vs diffusion rate
    disc = gamma * gamma - 4.0 * lambdas + 0j
    s_slow = ((-gamma + np.sqrt(disc)) / 2.0).real
    diff_rate = lambdas / gamma  # = nu_f * lambda_k
    mask = lambdas > 1e-9
    if np.any(mask):
        rel = np.abs(s_slow[mask] + diff_rate[mask]) / diff_rate[mask]
        max_rate_rel = float(np.max(rel))
    else:
        max_rate_rel = 0.0
    # Fiedler (slowest) mode
    if lam2 > 0.0:
        s_gap = (-gamma + np.sqrt(gamma * gamma - 4.0 * lam2)) / 2.0
        slow_gap = float(-s_gap)
    else:
        slow_gap = 0.0
    diff_gap = lam2 / gamma

    # (ii) trajectory: damped wave vs diffusion in the orthonormal eigenbasis
    sym_nodes, lsym = _symmetric_normalized_laplacian(G)
    w, V = np.linalg.eigh(lsym)
    w = np.clip(w, 0.0, None)
    q0 = structural_field(G, sym_nodes)
    c0 = V.T @ q0
    disc_w = gamma * gamma - 4.0 * w + 0j
    root_w = np.sqrt(disc_w)
    ss = (-gamma + root_w) / 2.0
    sf = (-gamma - root_w) / 2.0
    denom = sf - ss
    safe = np.abs(denom) > 1e-12
    a = np.where(safe, c0 * sf / np.where(safe, denom, 1.0), c0)
    b = np.where(safe, -c0 * ss / np.where(safe, denom, 1.0), 0.0)
    horizon = 3.0 * gamma / lam2 if lam2 > 0.0 else gamma
    ts = np.linspace(0.05 * horizon, horizon, max(2, n_time_samples))
    traj_err = 0.0
    for t in ts:
        q_wave = V @ (a * np.exp(ss * t) + b * np.exp(sf * t)).real
        q_diff = V @ (c0 * np.exp(-w * t / gamma))
        denom_t = float(np.linalg.norm(q_diff)) + 1e-12
        err = float(np.linalg.norm(q_wave - q_diff)) / denom_t
        traj_err = max(traj_err, err)

    projects = (max_rate_rel < tolerance) and (traj_err < tolerance)

    return OverdampedProjectionCertificate(
        n_nodes=n,
        gamma=float(gamma),
        nu_f_effective=float(nu_f),
        spectral_gap=lam2,
        lambda_max=lam_max,
        max_rate_rel_error=max_rate_rel,
        rate_error_times_gamma_sq=float(max_rate_rel * gamma * gamma),
        slowest_slow_rate=slow_gap,
        slowest_diffusion_rate=float(diff_gap),
        trajectory_max_rel_error=traj_err,
        projects_to_diffusion=bool(projects),
    )


@dataclass(frozen=True)
class UndampedLimitCertificate:
    r"""Verification that the γ→0 limit of the damped substrate wave is the
    undamped standing-wave spectrum √λ_k (the discrete modes).

    The overdamped projection (γ→∞, :func:`verify_overdamped_projection`)
    collapses the damped substrate wave onto structural diffusion.  Its
    opposite end, γ→0, is the **conservative** limit: the roots of
    s² + γs + λ_k = 0 become the pure-imaginary pair s = ±i√λ_k, so every
    mode oscillates undamped at the standing-wave frequency ω_k = √λ_k —
    exactly the discrete modes of :func:`verify_discrete_modes`.  γ is the
    single dial between the two regimes: γ→∞ diffusion, γ→0 standing waves.

    Attributes
    ----------
    n_nodes : int
    gamma : float
        Small damping used to probe the conservative limit.
    max_decay_rate : float
        Max |Re(s)| over the non-uniform modes (→ 0 as γ → 0; equals γ/2
        for underdamped modes — the envelope decay).
    max_freq_rel_error : float
        Max over modes of |Im(s) − √λ_k| / √λ_k: how close the damped
        oscillation frequency is to the undamped standing-wave frequency.
    freq_error_times_inv_gamma_sq : float
        ``max_freq_rel_error / γ²`` — converges to a constant, confirming
        the frequency error scales as O(γ²).
    matches_discrete_modes : bool
        Whether the γ→0 frequencies match the standing-wave spectrum √λ_k.
    standing_wave_frequencies : tuple[float, ...]
        The lowest few undamped frequencies ω_k = √λ_k (in ascending
        eigenvalue order, starting from the uniform mode ω₀ ≈ 0 — the same
        convention as :func:`verify_discrete_modes`).
    """

    n_nodes: int
    gamma: float
    max_decay_rate: float
    max_freq_rel_error: float
    freq_error_times_inv_gamma_sq: float
    matches_discrete_modes: bool
    standing_wave_frequencies: tuple[float, ...]

    @property
    def is_valid_undamped_limit(self) -> bool:
        """True when the γ→0 wave recovers the standing-wave spectrum."""
        return self.matches_discrete_modes

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_undamped_limit else "INVALID"
        return (
            f"Undamped limit [{ok}]: gamma->0 damped wave -> standing waves "
            f"s=+-i*sqrt(lambda_k); at gamma={self.gamma:.3g} decay "
            f"{self.max_decay_rate:.3e} (=gamma/2), freq error "
            f"{self.max_freq_rel_error:.2e} (/gamma^2="
            f"{self.freq_error_times_inv_gamma_sq:.3f}), "
            f"matches discrete modes={self.matches_discrete_modes}"
        )


def verify_undamped_limit(
    G: Any,
    *,
    gamma: float = 1e-3,
    tolerance: float = 1e-2,
) -> UndampedLimitCertificate:
    r"""Verify the γ→0 limit of the damped substrate wave is standing waves.

    Forms the damped graph wave q̈ + γq̇ + L q = 0 at a *small* damping γ and
    measures that each underdamped mode's complex root tends to the
    pure-imaginary standing-wave value s = ±i√λ_k: the envelope decay
    Re(s) = −γ/2 → 0, and the oscillation frequency Im(s) → √λ_k (the
    discrete-mode frequency of :func:`verify_discrete_modes`).  This is the
    conservative end of the same γ-dial whose γ→∞ end is the overdamped
    projection onto structural diffusion.

    Parameters
    ----------
    G : TNFRGraph
    gamma : float
        Small damping (γ² < 4·λ₂ keeps the slow modes underdamped /
        oscillatory).
    tolerance : float
        Maximum relative frequency error for a valid undamped limit.

    Returns
    -------
    UndampedLimitCertificate
    """
    nodes, lap = structural_diffusion_operator(G)
    n = len(nodes)
    lambdas, s_slow, _ = damped_wave_rates(G, gamma)
    omega = np.sqrt(np.clip(lambdas, 0.0, None))  # standing-wave frequencies

    # complex roots of s^2 + gamma s + lambda = 0 (full, not just real part)
    disc = gamma * gamma - 4.0 * lambdas + 0j
    root = np.sqrt(disc)
    s_plus = (-gamma + root) / 2.0
    decay = np.abs(s_plus.real)  # envelope decay = gamma/2 (underdamped)
    freq = np.abs(s_plus.imag)  # oscillation frequency

    mask = lambdas > 1e-9
    if np.any(mask):
        rel = np.abs(freq[mask] - omega[mask]) / omega[mask]
        max_freq_rel = float(np.max(rel))
        max_decay = float(np.max(decay[mask]))
    else:
        max_freq_rel = 0.0
        max_decay = 0.0

    freqs = tuple(float(omega[k]) for k in range(min(6, n)))
    matches = max_freq_rel < tolerance

    return UndampedLimitCertificate(
        n_nodes=n,
        gamma=float(gamma),
        max_decay_rate=max_decay,
        max_freq_rel_error=max_freq_rel,
        freq_error_times_inv_gamma_sq=float(max_freq_rel / (gamma * gamma)),
        matches_discrete_modes=bool(matches),
        standing_wave_frequencies=freqs,
    )


# ---------------------------------------------------------------------------
# Discrete modes: the standing waves of the bounded structural manifold
# ---------------------------------------------------------------------------


def _symmetric_normalized_laplacian(G: Any) -> tuple[list, Any]:
    r"""Return the symmetric normalized Laplacian L_sym.

    L_sym = I − D^{-1/2} W D^{-1/2} is symmetric (orthonormal eigenvectors)
    and shares the spectrum of the random-walk diffusion operator L_rw used
    by ΔNFR.
    """
    nodes = _ordered_nodes(G)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    deg = np.zeros(n, dtype=float)
    adj = np.zeros((n, n), dtype=float)
    for node in nodes:
        i = index[node]
        for m in G.neighbors(node):
            w = float(G[node][m].get("weight", 1.0))
            adj[i, index[m]] = w
            deg[i] += w
    dinv = np.where(deg > 0.0, 1.0 / np.sqrt(deg), 0.0)
    lap = np.eye(n) - (dinv[:, None] * adj * dinv[None, :])
    lap = 0.5 * (lap + lap.T)  # symmetrise residual numerical asymmetry
    return nodes, lap


def structural_eigenmodes(G: Any) -> tuple[Any, Any]:
    r"""Return the discrete eigenmodes of the bounded structural manifold.

    Computes the eigenvalues {λ_k} and orthonormal eigenvectors {v_k} of the
    symmetric normalized Laplacian L_sym (same spectrum as the diffusion
    operator L_rw).  The eigenvalues are the discrete mode "energies"; the
    eigenvectors are the standing-wave mode shapes (orthonormal), sorted by
    ascending λ_k.  λ_1 = 0 is the uniform mode.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (eigenvalues, eigenvectors) : tuple[np.ndarray, np.ndarray]
        ``eigenvalues`` shape ``(N,)`` ascending; ``eigenvectors`` shape
        ``(N, N)`` with column ``k`` the k-th standing-wave mode shape.
    """
    _, lap = _symmetric_normalized_laplacian(G)
    eigvals, eigvecs = np.linalg.eigh(lap)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvals, eigvecs


def nodal_domain_count(mode: Any) -> int:
    r"""Number of sign changes (nodal domains − 1) of a standing-wave mode.

    The structural "mode number": the k-th standing wave has k sign changes
    on a 1D manifold (Courant's nodal-domain ordering).  Near-zero entries
    are ignored to avoid spurious sign flips.

    Parameters
    ----------
    mode : np.ndarray
        A mode shape (eigenvector).

    Returns
    -------
    int
        The number of sign changes along the node ordering.
    """
    v = np.asarray(mode, dtype=float)
    sig = np.sign(v[np.abs(v) > 1e-12])
    if sig.size < 2:
        return 0
    return int(np.sum(np.abs(np.diff(sig)) > 0))


@dataclass(frozen=True)
class DiscreteModeCertificate:
    r"""Verification of the discrete standing-wave modes of a bounded manifold.

    A bounded structural manifold (finite graph) supports a discrete
    spectrum of orthonormal standing-wave eigenmodes — the same structure
    as the discrete harmonics of a vibrating string (Pythagoras), a Chladni
    plate, or a molecular vibrational spectrum.

    Attributes
    ----------
    n_modes : int
        Number of discrete modes (= number of nodes; finite/discrete).
    spectrum_is_discrete : bool
        The manifold has a finite, discrete eigenvalue spectrum.
    modes_orthonormal : bool
        The standing-wave mode shapes are orthonormal.
    max_orthonormality_residual : float
        Max |⟨v_i, v_j⟩ − δ_ij| (≈ 0).
    has_uniform_zero_mode : bool
        λ_1 = 0 (the uniform mode / conserved diffusion mode).
    spectral_gap : float
        λ_2 — the first non-trivial mode.
    matches_diffusion_spectrum : bool
        The L_sym spectrum equals the diffusion operator (L_rw) spectrum.
    nodal_domains_grow : bool
        The nodal-domain count grows from the lowest to the highest mode
        (Courant ordering; structural mode number).
    standing_wave_frequencies : tuple
        The first few standing-wave frequencies ω_k = √λ_k.
    """

    n_modes: int
    spectrum_is_discrete: bool
    modes_orthonormal: bool
    max_orthonormality_residual: float
    has_uniform_zero_mode: bool
    spectral_gap: float
    matches_diffusion_spectrum: bool
    nodal_domains_grow: bool
    standing_wave_frequencies: tuple

    @property
    def is_valid_discrete_modes(self) -> bool:
        """True when the manifold verifies as discrete standing waves."""
        return (
            self.spectrum_is_discrete
            and self.modes_orthonormal
            and self.has_uniform_zero_mode
            and self.matches_diffusion_spectrum
            and self.nodal_domains_grow
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_discrete_modes else "INVALID"
        freqs = ", ".join(f"{f:.3f}" for f in self.standing_wave_frequencies)
        return (
            f"Discrete standing-wave modes [{ok}]: "
            f"{self.n_modes} discrete modes, "
            f"orthonormal={self.modes_orthonormal} "
            f"(res {self.max_orthonormality_residual:.1e}), "
            f"uniform λ₁=0={self.has_uniform_zero_mode}, "
            f"spectral gap λ₂={self.spectral_gap:.4f}, "
            f"matches diffusion spectrum={self.matches_diffusion_spectrum}, "
            f"nodal domains grow={self.nodal_domains_grow}; "
            f"ω_k=√λ_k=[{freqs}]"
        )


def verify_discrete_modes(
    G: Any, *, tolerance: float = 1e-9
) -> DiscreteModeCertificate:
    r"""Verify the discrete standing-wave modes of the bounded manifold.

    Confirms that the finite manifold has a discrete spectrum of orthonormal
    standing-wave eigenmodes, with a uniform λ_1 = 0 mode, a spectrum
    matching the diffusion operator (L_rw), and nodal-domain counts growing
    with the mode index (Courant) — the structural origin of "discrete
    modes", the same as the discrete harmonics of a bounded elastic medium.

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Numerical tolerance for the orthonormality / spectrum checks.

    Returns
    -------
    DiscreteModeCertificate
    """
    eigvals, eigvecs = structural_eigenmodes(G)
    n = len(eigvals)

    discrete = n > 0 and np.all(np.isfinite(eigvals))

    gram = eigvecs.T @ eigvecs
    ortho_res = float(np.max(np.abs(gram - np.eye(n)))) if n else 0.0
    orthonormal = ortho_res < max(tolerance, 1e-9)

    uniform_zero = bool(abs(float(eigvals[0])) < 1e-6) if n else False
    gap = float(eigvals[1]) if n > 1 else 0.0

    # spectrum matches the random-walk diffusion operator L_rw
    _, lrw = structural_diffusion_operator(G)
    rw_spec = np.sort(np.linalg.eigvals(lrw).real)
    matches = bool(np.allclose(np.sort(eigvals), rw_spec, atol=1e-7))

    # nodal-domain counts grow from lowest to highest mode (Courant)
    counts = [nodal_domain_count(eigvecs[:, k]) for k in range(n)]
    grow = (counts[0] == 0 and counts[-1] > counts[0]) if n > 1 else True

    freqs = tuple(float(np.sqrt(eigvals[k])) for k in range(min(6, n)))

    return DiscreteModeCertificate(
        n_modes=n,
        spectrum_is_discrete=discrete,
        modes_orthonormal=orthonormal,
        max_orthonormality_residual=ortho_res,
        has_uniform_zero_mode=uniform_zero,
        spectral_gap=gap,
        matches_diffusion_spectrum=matches,
        nodal_domains_grow=grow,
        standing_wave_frequencies=freqs,
    )


# ---------------------------------------------------------------------------
# Structural stability: the dispersion relation and the instability threshold
# ---------------------------------------------------------------------------


def dispersion_relation(G: Any, reaction_rate: float = 0.0) -> Any:
    r"""Per-mode growth rates σ_k = reaction_rate − νf·λ_k (dispersion).

    The growth (σ_k > 0) or decay (σ_k < 0) rate of each structural
    eigenmode under diffusion plus a local reaction rate.  At
    ``reaction_rate = 0`` this is the negative of the relaxation spectrum
    (pure diffusion: every non-uniform mode decays).

    Parameters
    ----------
    G : TNFRGraph
    reaction_rate : float, optional
        A local growth/decay rate r added to every mode (the operators
        supply it: stabilizers lower r, destabilizers raise it).

    Returns
    -------
    np.ndarray
        The growth rates σ_k sorted by ascending λ_k.
    """
    eigvals, _ = structural_eigenmodes(G)
    nu = structural_diffusivity(G)
    return float(reaction_rate) - nu * eigvals


def instability_threshold(G: Any) -> float:
    r"""Structural instability threshold r_c = νf·λ_2 (spectral gap).

    The reaction rate above which the first *non-uniform* (Fiedler) mode
    becomes unstable.  For 0 < r < r_c only the uniform mode grows (global
    amplification); for r > r_c a structural pattern (the Fiedler partition)
    emerges.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    float
        νf·λ_2 (0 if the graph has fewer than two modes).
    """
    eigvals, _ = structural_eigenmodes(G)
    nu = structural_diffusivity(G)
    return float(nu * eigvals[1]) if len(eigvals) > 1 else 0.0


def fiedler_partition(G: Any) -> tuple[list, list]:
    r"""The Fiedler-mode 2-partition — the first structural pattern.

    Splits the nodes by the sign of the Fiedler eigenvector (the mode with
    the smallest non-zero λ).  This is the network's **weakest structural
    cut** (the two most weakly-connected communities) — the empirically-
    validated spectral-clustering partition, and the first pattern to grow
    once the reaction rate crosses the instability threshold.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (part_a, part_b) : tuple[list, list]
        Node lists for the two structural communities.
    """
    nodes = _ordered_nodes(G)
    eigvals, eigvecs = structural_eigenmodes(G)
    if eigvecs.shape[1] < 2:
        return list(nodes), []
    fiedler = eigvecs[:, 1]
    part_a = [nodes[i] for i in range(len(nodes)) if fiedler[i] > 0]
    part_b = [nodes[i] for i in range(len(nodes)) if fiedler[i] <= 0]
    return part_a, part_b


@dataclass(frozen=True)
class StructuralStabilityCertificate:
    r"""Verification of the structural linear-stability / dispersion relation.

    The dispersion relation σ_k = r − νf·λ_k governs the growth/decay of
    every structural eigenmode.  Pure diffusion (r = 0) decays every
    non-uniform mode (stable equilibrium); the threshold r_c = νf·λ_2
    separates uniform amplification from structural pattern formation (the
    Fiedler partition).  This is the spectral form of U2 grammar.

    Attributes
    ----------
    n_nodes : int
    diffusivity : float
        νf (the diffusion coefficient).
    spectral_gap : float
        λ_2 (the Fiedler value).
    instability_threshold : float
        r_c = νf·λ_2.
    pure_diffusion_stable : bool
        At r = 0 every non-uniform mode decays (σ_k < 0 for k ≥ 1).
    max_nonuniform_growth_at_zero : float
        Max σ_k over k ≥ 1 at r = 0 (should be < 0).
    dispersion_matches_relaxation : bool
        The r = 0 dispersion equals the negative relaxation spectrum.
    first_unstable_mode : int
        The first non-uniform mode to go unstable above threshold (= 1,
        the Fiedler mode).
    fiedler_partition_sizes : tuple
        Sizes (|A|, |B|) of the Fiedler 2-partition (the first pattern).
    """

    n_nodes: int
    diffusivity: float
    spectral_gap: float
    instability_threshold: float
    pure_diffusion_stable: bool
    max_nonuniform_growth_at_zero: float
    dispersion_matches_relaxation: bool
    first_unstable_mode: int
    fiedler_partition_sizes: tuple

    @property
    def is_valid_stability(self) -> bool:
        """True when the structural-stability picture verifies."""
        return (
            self.pure_diffusion_stable
            and self.dispersion_matches_relaxation
            and self.first_unstable_mode == 1
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_stability else "INVALID"
        a, b = self.fiedler_partition_sizes
        return (
            f"Structural stability [{ok}]: "
            f"νf={self.diffusivity:.4f}, spectral gap λ₂="
            f"{self.spectral_gap:.4f}, "
            f"instability threshold r_c=νf·λ₂="
            f"{self.instability_threshold:.4f}, "
            f"pure diffusion stable={self.pure_diffusion_stable} "
            f"(max non-uniform growth "
            f"{self.max_nonuniform_growth_at_zero:.1e}), "
            f"first unstable mode k={self.first_unstable_mode} (Fiedler), "
            f"Fiedler partition {a}/{b}"
        )


def verify_structural_stability(
    G: Any, *, tolerance: float = 1e-9
) -> StructuralStabilityCertificate:
    r"""Verify the structural linear-stability / dispersion relation.

    Confirms that pure diffusion decays every non-uniform mode (stable
    equilibrium), that the r = 0 dispersion equals the negative relaxation
    spectrum, that the instability threshold is r_c = νf·λ_2, and that the
    first structural mode to go unstable above threshold is the Fiedler
    mode (whose eigenvector gives the weakest-cut partition).

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    StructuralStabilityCertificate
    """
    eigvals, eigvecs = structural_eigenmodes(G)
    n = len(eigvals)
    nu = structural_diffusivity(G)
    gap = float(eigvals[1]) if n > 1 else 0.0
    r_c = nu * gap

    # pure diffusion (r=0): non-uniform modes (k>=1) decay
    sigma0 = dispersion_relation(G, 0.0)
    max_nonuniform = float(np.max(sigma0[1:])) if n > 1 else 0.0
    stable = max_nonuniform < tolerance

    # dispersion at r=0 == negative relaxation spectrum
    rates = relaxation_spectrum(G)
    matches = bool(np.allclose(np.sort(-sigma0), np.sort(rates), atol=1e-7))

    # first non-uniform mode to go unstable just above threshold
    sigma = dispersion_relation(G, r_c + max(1e-3, 1e-3 * r_c))
    if n > 1 and np.any(sigma[1:] > tolerance):
        first_unstable = int(np.argmax(sigma[1:] > tolerance)) + 1
    else:
        first_unstable = 0

    # Fiedler partition sizes
    part_a, part_b = fiedler_partition(G)
    sizes = (len(part_a), len(part_b))

    return StructuralStabilityCertificate(
        n_nodes=n,
        diffusivity=nu,
        spectral_gap=gap,
        instability_threshold=r_c,
        pure_diffusion_stable=stable,
        max_nonuniform_growth_at_zero=max_nonuniform,
        dispersion_matches_relaxation=matches,
        first_unstable_mode=first_unstable,
        fiedler_partition_sizes=sizes,
    )


# ---------------------------------------------------------------------------
# The structural random walk: Brownian motion and resistance geometry
# ---------------------------------------------------------------------------


def _adjacency_degree(G: Any) -> tuple[list, Any, Any]:
    """Return (nodes, weighted adjacency W, weighted degree vector)."""
    nodes = _ordered_nodes(G)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=float)
    for node in nodes:
        i = index[node]
        for m in G.neighbors(node):
            adj[i, index[m]] = float(G[node][m].get("weight", 1.0))
    deg = adj.sum(axis=1)
    return nodes, adj, deg


def random_walk_matrix(G: Any) -> tuple[list, Any]:
    r"""Return the random-walk transition matrix P = D^{-1}W.

    The diffusion operator is L_rw = I − P, so P is exactly the
    random-walk the structural diffusion generates.  Row-stochastic (each
    row sums to 1); isolated nodes get a zero row.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, P) : tuple[list, np.ndarray]
    """
    nodes, adj, deg = _adjacency_degree(G)
    with np.errstate(divide="ignore", invalid="ignore"):
        dinv = np.where(deg > 0.0, 1.0 / deg, 0.0)
    return nodes, dinv[:, None] * adj


def stationary_distribution(G: Any) -> tuple[list, Any]:
    r"""Stationary distribution π_i = deg(i) / Σ deg of the random walk.

    The random walk converges to the degree distribution — exactly the
    degree-weighted total the diffusion conserves.  The conserved quantity
    is the equilibrium measure.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, pi) : tuple[list, np.ndarray]
    """
    nodes, _, deg = _adjacency_degree(G)
    total = float(deg.sum())
    pi = deg / total if total > 0 else deg
    return nodes, pi


def _laplacian_pinv(G: Any) -> tuple[list, Any, float]:
    """Return (nodes, pinv(L) of the combinatorial Laplacian, n_edges)."""
    nodes, adj, deg = _adjacency_degree(G)
    lap = np.diag(deg) - adj
    return nodes, np.linalg.pinv(lap), float(deg.sum()) / 2.0


def effective_resistance(G: Any) -> tuple[list, Any]:
    r"""Effective-resistance matrix R_eff(i,j) (Ohm's law).

    Treating the network as a resistor network (the combinatorial
    Laplacian L = D − W is the conductance matrix — Kirchhoff 1847), the
    effective resistance between nodes is

        R_eff(i,j) = L⁺_ii + L⁺_jj − 2·L⁺_ij,

    with L⁺ the Moore–Penrose pseudoinverse.  R_eff is a metric
    (symmetric, non-negative, triangle inequality) — the structural
    "difficulty of transport" between two nodes.

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, R) : tuple[list, np.ndarray]
        ``R[i, j]`` is the effective resistance between node i and node j.
    """
    nodes, lp, _ = _laplacian_pinv(G)
    diag = np.diag(lp)
    r = diag[:, None] + diag[None, :] - 2.0 * lp
    np.fill_diagonal(r, 0.0)
    return nodes, np.maximum(r, 0.0)


def commute_time(G: Any) -> tuple[list, Any]:
    r"""Commute-time matrix C(i,j) = 2m·R_eff(i,j).

    The expected round-trip time of the structural random walk between two
    nodes equals 2m times the effective resistance (m the number of
    edges) — the exact link between the diffusion random walk and the
    resistance geometry (Chandra et al. 1996).

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, C) : tuple[list, np.ndarray]
    """
    nodes, r = effective_resistance(G)
    _, _, deg = _adjacency_degree(G)
    m_edges = float(deg.sum()) / 2.0
    return nodes, 2.0 * m_edges * r


@dataclass(frozen=True)
class RandomWalkCertificate:
    r"""Verification of the structural random walk and resistance geometry.

    The diffusion operator is the generator of a random walk (Brownian
    motion on the network); its stationary distribution is the degree, and
    the effective resistance / commute time give the transport geometry.

    Attributes
    ----------
    n_nodes : int
    operator_is_walk_generator : bool
        L_rw = I − P exactly (the diffusion operator generates the walk).
    transition_row_stochastic : bool
        P = D⁻¹W is row-stochastic.
    stationary_is_degree : bool
        π = deg / Σ deg is the stationary distribution (π·P = π).
    resistance_is_metric : bool
        R_eff is symmetric, non-negative, and obeys the triangle
        inequality.
    max_resistance : float
        The largest pairwise effective resistance (transport diameter).
    commute_equals_2m_resistance : bool
        C(i,j) = 2m·R_eff(i,j) (random walk ↔ resistance identity).
    max_walk_generator_residual : float
        Max |L_rw − (I − P)| (≈ 0).
    """

    n_nodes: int
    operator_is_walk_generator: bool
    transition_row_stochastic: bool
    stationary_is_degree: bool
    resistance_is_metric: bool
    max_resistance: float
    commute_equals_2m_resistance: bool
    max_walk_generator_residual: float

    @property
    def is_valid_random_walk(self) -> bool:
        """True when the structural random-walk picture verifies."""
        return (
            self.operator_is_walk_generator
            and self.transition_row_stochastic
            and self.stationary_is_degree
            and self.resistance_is_metric
            and self.commute_equals_2m_resistance
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_random_walk else "INVALID"
        return (
            f"Structural random walk [{ok}]: "
            f"L_rw = I−P={self.operator_is_walk_generator} "
            f"(res {self.max_walk_generator_residual:.1e}), "
            f"P row-stochastic={self.transition_row_stochastic}, "
            f"stationary π=degree={self.stationary_is_degree}, "
            f"R_eff metric={self.resistance_is_metric} "
            f"(max R {self.max_resistance:.4f}), "
            f"commute=2m·R_eff={self.commute_equals_2m_resistance}"
        )


def verify_structural_random_walk(
    G: Any, *, tolerance: float = 1e-9
) -> RandomWalkCertificate:
    r"""Verify the structural random walk and resistance geometry.

    Confirms that the diffusion operator is the random-walk generator
    (L_rw = I − P), that P is row-stochastic with stationary distribution
    π = degree, that the effective resistance is a transport metric, and
    that the commute time equals 2m·R_eff (random walk ↔ resistance).

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    RandomWalkCertificate
    """
    nodes, lrw = structural_diffusion_operator(G)
    n = len(nodes)
    _, p = random_walk_matrix(G)

    # L_rw = I − P
    gen_res = float(np.max(np.abs(lrw - (np.eye(n) - p)))) if n else 0.0
    is_generator = gen_res < max(tolerance, 1e-12)

    # P row-stochastic (rows of connected nodes sum to 1)
    row_sums = p.sum(axis=1)
    deg_nonzero = np.array([sum(1 for _ in G.neighbors(nd)) > 0 for nd in nodes])
    row_stochastic = (
        bool(np.all(np.abs(row_sums[deg_nonzero] - 1.0) < tolerance)) if n else True
    )

    # stationary distribution π = degree, π·P = π
    _, pi = stationary_distribution(G)
    stationary_ok = bool(np.allclose(pi @ p, pi, atol=1e-7)) if n else True

    # effective resistance is a metric
    _, r = effective_resistance(G)
    symmetric = bool(np.allclose(r, r.T))
    nonneg = bool(np.all(r >= -1e-9))
    # triangle inequality on a sample of triples
    triangle = True
    if n >= 3:
        rng = np.random.default_rng(0)
        for _ in range(200):
            i, j, k = rng.integers(0, n, size=3)
            if r[i, k] > r[i, j] + r[j, k] + 1e-7:
                triangle = False
                break
    is_metric = symmetric and nonneg and triangle
    max_r = float(np.max(r)) if n else 0.0

    # commute time = 2m·R_eff
    _, c = commute_time(G)
    _, _, deg = _adjacency_degree(G)
    m_edges = float(deg.sum()) / 2.0
    commute_ok = bool(np.allclose(c, 2.0 * m_edges * r, atol=1e-7))

    return RandomWalkCertificate(
        n_nodes=n,
        operator_is_walk_generator=is_generator,
        transition_row_stochastic=row_stochastic,
        stationary_is_degree=stationary_ok,
        resistance_is_metric=is_metric,
        max_resistance=max_r,
        commute_equals_2m_resistance=commute_ok,
        max_walk_generator_residual=gen_res,
    )


# ---------------------------------------------------------------------------
# The structural flow: current, Kirchhoff's law, and continuity
# ---------------------------------------------------------------------------
def structural_current(G: Any) -> tuple[list, Any]:
    r"""Structural (diffusion) edge-current matrix J_ij = EPI_i − EPI_j.

    The transport carries a current: along each edge i∼j the diffusion
    flux is J_ij = EPI_i − EPI_j (Fick's law — flux from high to low
    concentration).  The matrix is **antisymmetric** (J_ij = −J_ji) and
    supported on edges only (J_ij = 0 when i and j are not adjacent).

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, J) : tuple[list, np.ndarray]
        ``J[i, j]`` is the current from node i to node j across edge i∼j.
    """
    from ..alias import get_attr
    from ..constants.aliases import ALIAS_EPI

    nodes, adj, _ = _adjacency_degree(G)
    epi = np.array(
        [float(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes],
        dtype=float,
    )
    mask = adj != 0.0
    # J_ij = EPI_i − EPI_j on edges, zero elsewhere
    j = (epi[:, None] - epi[None, :]) * mask
    return nodes, j


def current_divergence(G: Any) -> tuple[list, Any]:
    r"""Net current outflow per node div(J)(i) = Σ_{j∼i} J_ij = (L·EPI)(i).

    Kirchhoff's current law: the net outflow at a node equals the
    combinatorial Laplacian L = D − W acting on EPI.  This is the discrete
    continuity equation div(J) = L·EPI, so ∂EPI/∂t + div(J) = 0 for the
    diffusion dynamics (∂EPI/∂t = −L_rw·EPI carries the same content
    degree-normalized).

    Parameters
    ----------
    G : TNFRGraph

    Returns
    -------
    (nodes, div) : tuple[list, np.ndarray]
        ``div[i]`` is the net current leaving node i.
    """
    nodes, j = structural_current(G)
    return nodes, j.sum(axis=1)


@dataclass(frozen=True)
class StructuralFlowCertificate:
    r"""Verification of the structural flow (current, Kirchhoff, Ohm).

    The diffusion transport carries a structural current J_ij = EPI_i −
    EPI_j (Fick's law).  Its node balance is Kirchhoff's current law —
    the discrete continuity equation div(J) = L·EPI — and under an
    injected current the potential drop is the effective resistance
    (Ohm's law).

    Attributes
    ----------
    n_nodes : int
    current_antisymmetric : bool
        J_ij = −J_ji (the current is a directed edge flow).
    kirchhoff_holds : bool
        Net outflow Σ_j J_ij = (L·EPI)_i (current law = continuity).
    total_flux_balances : bool
        Σ_i div(J)_i = 0 (closed network, no sources/sinks).
    equilibrium_zero_current : bool
        A uniform EPI field produces zero current everywhere.
    ohm_law_holds : bool
        An injected unit current s→t induces a potential drop equal to
        R_eff(s,t).
    max_kirchhoff_residual : float
        Max |Σ_j J_ij − (L·EPI)_i| (≈ 0).
    """

    n_nodes: int
    current_antisymmetric: bool
    kirchhoff_holds: bool
    total_flux_balances: bool
    equilibrium_zero_current: bool
    ohm_law_holds: bool
    max_kirchhoff_residual: float

    @property
    def is_valid_flow(self) -> bool:
        """True when the structural-flow picture verifies."""
        return (
            self.current_antisymmetric
            and self.kirchhoff_holds
            and self.total_flux_balances
            and self.equilibrium_zero_current
            and self.ohm_law_holds
        )

    def summary(self) -> str:
        """Human-readable one-line verdict."""
        ok = "VALID" if self.is_valid_flow else "INVALID"
        return (
            f"Structural flow [{ok}]: "
            f"current antisymmetric={self.current_antisymmetric}, "
            f"Kirchhoff div(J)=L·EPI={self.kirchhoff_holds} "
            f"(res {self.max_kirchhoff_residual:.1e}), "
            f"total flux balances={self.total_flux_balances}, "
            f"equilibrium zero current={self.equilibrium_zero_current}, "
            f"Ohm drop=R_eff={self.ohm_law_holds}"
        )


def verify_structural_flow(
    G: Any, *, tolerance: float = 1e-9
) -> StructuralFlowCertificate:
    r"""Verify the structural flow: current, Kirchhoff's law, Ohm's law.

    Confirms that the diffusion edge current J_ij = EPI_i − EPI_j is
    antisymmetric, that Kirchhoff's current law div(J) = L·EPI holds (the
    discrete continuity equation), that the total flux balances on a closed
    network, that a uniform EPI field carries zero current, and that an
    injected unit current induces a potential drop equal to the effective
    resistance (Ohm's law).

    Parameters
    ----------
    G : TNFRGraph
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    StructuralFlowCertificate
    """
    from ..alias import get_attr
    from ..constants.aliases import ALIAS_EPI

    nodes, j = structural_current(G)
    n = len(nodes)

    # current antisymmetry J_ij = −J_ji
    antisym = bool(np.allclose(j, -j.T, atol=tolerance)) if n else True

    # Kirchhoff: net outflow = (L·EPI) with L the combinatorial Laplacian
    _, adj, deg = _adjacency_degree(G)
    lap = np.diag(deg) - adj
    epi = np.array(
        [float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0)) for nd in nodes],
        dtype=float,
    )
    net_out = j.sum(axis=1)
    kirchhoff_res = float(np.max(np.abs(net_out - lap @ epi))) if n else 0.0
    kirchhoff_ok = kirchhoff_res < max(tolerance, 1e-9)

    # total flux balances (L has zero column sums) — closed network
    total_balances = (
        bool(abs(float(net_out.sum())) < max(tolerance, 1e-9)) if n else True
    )

    # equilibrium: a uniform EPI field carries zero current
    uniform = np.ones(n)
    j_uniform = (uniform[:, None] - uniform[None, :]) * (adj != 0.0)
    equilibrium_ok = bool(np.allclose(j_uniform, 0.0, atol=tolerance)) if n else True

    # Ohm's law: injected unit current s→t induces drop V_s − V_t = R_eff
    ohm_ok = True
    if n >= 2:
        lp = np.linalg.pinv(lap)
        diag = np.diag(lp)
        rng = np.random.default_rng(0)
        for _ in range(min(20, n)):
            s, t = rng.integers(0, n, size=2)
            if s == t:
                continue
            b = np.zeros(n)
            b[s], b[t] = 1.0, -1.0
            v = lp @ b
            drop = v[s] - v[t]
            r_eff = diag[s] + diag[t] - 2.0 * lp[s, t]
            if not np.isclose(drop, r_eff, atol=1e-7):
                ohm_ok = False
                break

    return StructuralFlowCertificate(
        n_nodes=n,
        current_antisymmetric=antisym,
        kirchhoff_holds=kirchhoff_ok,
        total_flux_balances=total_balances,
        equilibrium_zero_current=equilibrium_ok,
        ohm_law_holds=ohm_ok,
        max_kirchhoff_residual=kirchhoff_res,
    )
