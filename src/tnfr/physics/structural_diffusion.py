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

HONEST SCOPE
============
- The identity ΔNFR_epi = −L_rw·EPI is EXACT (machine precision), a
  mathematical fact about the canonical ΔNFR.
- The full ΔNFR is multi-channel: EPI **diffusion** + phase
  **synchronization** + νf/topology **homogenization**.  This module
  isolates and certifies the diffusion (EPI) channel and reports the
  synchronization channel qualitatively.
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

from ..mathematics.unified_numerical import np
from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR

__all__ = [
    "StructuralDiffusionCertificate",
    "structural_diffusion_operator",
    "structural_field",
    "structural_diffusivity",
    "relaxation_spectrum",
    "degree_weighted_total",
    "verify_structural_diffusion",
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
        weights = [
            float(G[node][m].get("weight", 1.0)) for m in neigh
        ]
        deg = sum(weights)
        if deg <= 0.0:
            continue
        lap[i, i] = 1.0
        for m, w in zip(neigh, weights):
            lap[i, index[m]] -= w / deg
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
            sum(
                float(G[node][m].get("weight", 1.0))
                for m in G.neighbors(node)
            )
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
