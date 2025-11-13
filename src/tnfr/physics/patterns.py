"""
TNFR structural pattern constructors for particle-like emergent modes.
Each constructor initializes node attributes
(theta/phase, delta_nfr/dnfr, coherence)
consistent with TNFR semantics:
- Phases wrapped to [0, 2π)
- ΔNFR controls local structural pressure; coherence = 1 / (1 + |ΔNFR|)
- No dynamics here; these are static initializations to study field signatures.

Note: These are TNFR-native patterns, not SM postulates. They provide
coherent-form prototypes to compare tetrad metrics and topological invariants.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx

_TWO_PI = 2.0 * math.pi


def _wrap_2pi(x: float) -> float:
    # map real to [0, 2π)
    y = x % _TWO_PI
    if y < 0.0:
        y += _TWO_PI
    return float(y)


def _set_baseline(G: nx.Graph, base_dnfr: float = 0.05) -> None:
    for n in G.nodes():
        G.nodes[n]["theta"] = 0.0
        G.nodes[n]["phase"] = 0.0
        G.nodes[n]["delta_nfr"] = float(base_dnfr)
        G.nodes[n]["dnfr"] = float(base_dnfr)
        G.nodes[n]["coherence"] = 1.0 / (1.0 + abs(base_dnfr))


def reset_baseline(G: nx.Graph, base_dnfr: float = 0.05) -> nx.Graph:
    """Reset all per-node telemetry to a mild baseline.
    Returns the graph for chaining.
    """
    _set_baseline(G, base_dnfr=base_dnfr)
    return G


def apply_plane_wave(G: nx.Graph, kx: float = 0.25, ky: float = 0.0) -> None:
    """Photon-like: coherent propagating phase front (Q≈0, K_φ≈0)."""
    for n in G.nodes():
        if isinstance(n, tuple) and len(n) == 2:
            i, j = n
        else:
            # fallback: use index order if not grid-like
            i = j = int(n) if isinstance(n, (int, np.integer)) else 0
        phi = _wrap_2pi(kx * i + ky * j)
        G.nodes[n]["theta"] = float(phi)
        G.nodes[n]["phase"] = float(phi)


def apply_vortex(
    G: nx.Graph,
    center: Optional[Tuple[int, int]] = None,
    dnfr_core: float = 0.2,
    decay: float = 4.0,
) -> Tuple[int, int]:
    """Electron-like: localized vortex with winding Q=±1."""
    # infer center for grid graphs
    if center is None:
        try:
            xs = [n[0] for n in G.nodes()]
            ys = [n[1] for n in G.nodes()]
            cx = int(np.median(xs))
            cy = int(np.median(ys))
            center = (cx, cy)
        except Exception:
            center = (0, 0)
    cx, cy = center
    for n in G.nodes():
        if isinstance(n, tuple) and len(n) == 2:
            i, j = n
        else:
            i = j = int(n) if isinstance(n, (int, np.integer)) else 0
        ang = math.atan2((j - cy), (i - cx))  # (-π, π]
        phi = _wrap_2pi(ang)
        r = math.hypot(i - cx, j - cy)
        dnfr = dnfr_core * math.exp(-r / max(1e-9, decay)) + 0.05
        G.nodes[n]["theta"] = float(phi)
        G.nodes[n]["phase"] = float(phi)
        G.nodes[n]["delta_nfr"] = float(dnfr)
        G.nodes[n]["dnfr"] = float(dnfr)
        G.nodes[n]["coherence"] = 1.0 / (1.0 + abs(dnfr))
    return (cx, cy)


def apply_helical_packet(
    G: nx.Graph,
    center: Optional[Tuple[int, int]] = None,
    kx: float = 0.2,
    ky: float = 0.0,
    twist: float = 0.3,
    sigma: float = 8.0,
) -> Tuple[int, int]:
    """Massive gauge-like packet: traveling phase with gentle
    helical twist around a center.
    Q≈0, with chirality (sign of twist) and local confinement by
    envelope.
    """
    if center is None:
        try:
            xs = [n[0] for n in G.nodes()]
            ys = [n[1] for n in G.nodes()]
            cx = int(np.median(xs))
            cy = int(np.median(ys))
            center = (cx, cy)
        except Exception:
            center = (0, 0)
    cx, cy = center
    for n in G.nodes():
        if isinstance(n, tuple) and len(n) == 2:
            i, j = n
        else:
            i = j = int(n) if isinstance(n, (int, np.integer)) else 0
        ang = math.atan2((j - cy), (i - cx))
        r2 = (i - cx) ** 2 + (j - cy) ** 2
        env = math.exp(-r2 / (2.0 * sigma * sigma))
        base = kx * i + ky * j
        phi = _wrap_2pi(base + twist * ang * env)
        dnfr = 0.1 * env + 0.05
        G.nodes[n]["theta"] = float(phi)
        G.nodes[n]["phase"] = float(phi)
        G.nodes[n]["delta_nfr"] = float(dnfr)
        G.nodes[n]["dnfr"] = float(dnfr)
        G.nodes[n]["coherence"] = 1.0 / (1.0 + abs(dnfr))
    return (cx, cy)


def apply_global_curvature(G: nx.Graph, a: float = 1e-3) -> None:
    """Graviton-like: very weak global curvature
    (small K_φ, large ξ_C).
    """
    for n in G.nodes():
        if isinstance(n, tuple) and len(n) == 2:
            i, j = n
        else:
            i = j = int(n) if isinstance(n, (int, np.integer)) else 0
        phi = _wrap_2pi(a * (i * i + j * j))
        G.nodes[n]["theta"] = float(phi)
        G.nodes[n]["phase"] = float(phi)


def apply_scalar_bump(
    G: nx.Graph,
    center: Optional[Tuple[int, int]] = None,
    amp: float = 0.25,
    sigma: float = 6.0,
) -> Tuple[int, int]:
    """Higgs-like: local scalar field that elevates ΔNFR around the
    center with nearly constant phase.
    """
    if center is None:
        try:
            xs = [n[0] for n in G.nodes()]
            ys = [n[1] for n in G.nodes()]
            cx = int(np.median(xs))
            cy = int(np.median(ys))
            center = (cx, cy)
        except Exception:
            center = (0, 0)
    cx, cy = center
    for n in G.nodes():
        if isinstance(n, tuple) and len(n) == 2:
            i, j = n
        else:
            i = j = int(n) if isinstance(n, (int, np.integer)) else 0
        r2 = (i - cx) ** 2 + (j - cy) ** 2
        bump = amp * math.exp(-r2 / (2.0 * sigma * sigma))
        # phase stays near-constant; we add a tiny offset to avoid degeneracy
        phi = _wrap_2pi(0.01 * bump)
        dnfr = 0.05 + abs(bump)
        G.nodes[n]["theta"] = float(phi)
        G.nodes[n]["phase"] = float(phi)
        G.nodes[n]["delta_nfr"] = float(dnfr)
        G.nodes[n]["dnfr"] = float(dnfr)
        G.nodes[n]["coherence"] = 1.0 / (1.0 + abs(dnfr))
    return (cx, cy)


def apply_quark_triplet_cluster(
    G: nx.Graph,
    centers: Optional[List[Tuple[int, int]]] = None,
    dnfr_core: float = 0.18,
    decay: float = 3.5,
) -> List[Tuple[int, int]]:
    """Quark-like cluster: three localized defects that only
    show coherent stability as an ensemble.
    Implementation: three Q=+1 vortices with nearby cores;
    their large loop has Q≈3.
    """
    if centers is None:
        # pick an equilateral-ish triangle around center
        try:
            xs = [n[0] for n in G.nodes()]
            ys = [n[1] for n in G.nodes()]
            cx = int(np.median(xs))
            cy = int(np.median(ys))
        except Exception:
            cx, cy = 0, 0
        d = 6
        centers = [(cx - d, cy), (cx + d, cy), (cx, cy + d)]
    for c in centers:
        apply_vortex(G, center=c, dnfr_core=dnfr_core, decay=decay)
    return centers


def apply_neutrino_like(G: nx.Graph, eps: float = 0.03) -> None:
    """Neutrino-like: faint mode (very low |∇φ|, Q≈0) with weak
    footprint in K_φ.
    """
    for n in G.nodes():
        if isinstance(n, tuple) and len(n) == 2:
            i, j = n
        else:
            i = j = int(n) if isinstance(n, (int, np.integer)) else 0
        phi = _wrap_2pi(eps * (i + j))
        G.nodes[n]["theta"] = float(phi)
        G.nodes[n]["phase"] = float(phi)


def apply_color_domain_lattice(
    G: nx.Graph,
    period: int = 6,
    dphi: float = math.pi / 2,
    wall_dnfr: float = 0.15,
) -> None:
    """Gluon-like surrogate: network of phase domains with walls
    (domain walls) where K_φ concentrates. Alternates phase offsets by
    periodic cells; increases ΔNFR at walls.
    """
    for n in G.nodes():
        if isinstance(n, tuple) and len(n) == 2:
            i, j = n
        else:
            i = j = int(n) if isinstance(n, (int, np.integer)) else 0
        block_i = (i // max(1, period)) % 2
        block_j = (j // max(1, period)) % 2
        offset = (block_i ^ block_j) * dphi
        phi = _wrap_2pi(offset)
        # domain walls approx when crossing boundaries of blocks
        on_boundary = (i % period == 0) or (j % period == 0)
        dnfr = 0.05 + (wall_dnfr if on_boundary else 0.0)
        G.nodes[n]["theta"] = float(phi)
        G.nodes[n]["phase"] = float(phi)
        G.nodes[n]["delta_nfr"] = float(dnfr)
        G.nodes[n]["dnfr"] = float(dnfr)
        G.nodes[n]["coherence"] = 1.0 / (1.0 + abs(dnfr))


__all__ = [
    "reset_baseline",
    "apply_plane_wave",
    "apply_vortex",
    "apply_helical_packet",
    "apply_global_curvature",
    "apply_scalar_bump",
    "apply_quark_triplet_cluster",
    "apply_neutrino_like",
    "apply_color_domain_lattice",
]


# --- Element-like radial pattern (centralization) ---------------------------

def build_element_radial_pattern(Z: int, *, seed: int = 42) -> nx.Graph:
    """Build a simple TNFR element-like radial graph for atomic number Z.

    Canonical, telemetry-only initializer placed in the physics module to
    centralize element-like pattern construction. This mirrors the examples
    helper while keeping physics-first semantics.

    Topology (heuristic for demo; not prescriptive chemistry):
    - One nucleus node connected to all inner-ring nodes (shell1)
    - Inner ring as a cycle (angular coupling)
    - If Z >= 3, add a second ring (shell2) as a cycle and add spokes
      from shell1 to shell2 every s nodes.

    Attributes seeded reproducibly:
    - 'phase' ∈ [0, 2π)
    - 'delta_nfr' in [0.2, 0.8)
    - 'role' in {nucleus, shell1, shell2}
    - 'coherence' implied by ΔNFR can be derived by callers as 1/(1+|ΔNFR|)
    """
    if Z < 1:
        Z = 1

    G = nx.Graph()
    nucleus = 0
    G.add_node(nucleus, role="nucleus")

    # Size inner shell proportional to Z (bounded below)
    n1 = max(8, 10 + 2 * Z)
    shell1 = list(range(1, 1 + n1))
    G.add_nodes_from(shell1)
    # inner cycle
    for i in range(n1):
        a = shell1[i]
        b = shell1[(i + 1) % n1]
        G.add_edge(a, b)
    # nucleus spokes
    for r in shell1:
        G.add_edge(nucleus, r)

    shell2: List[int] = []
    if Z >= 3:
        # add a second shell with mild growth
        n2 = max(10, 12 + 2 * (Z - 2))
        shell2 = list(range(1 + n1, 1 + n1 + n2))
        G.add_nodes_from(shell2)
        # outer cycle
        for i in range(n2):
            a = shell2[i]
            b = shell2[(i + 1) % n2]
            G.add_edge(a, b)
        # spokes inner↔outer every s nodes
        s = max(2, n1 // 6)
        for i in range(0, n1, s):
            inner = shell1[i]
            outer = shell2[(i * n2) // n1]
            G.add_edge(inner, outer)

    # Seeded attributes
    import random as _random
    rng = _random.Random(seed)
    for i in G.nodes():
        G.nodes[i]["phase"] = float(2.0 * math.pi * rng.random())
        G.nodes[i]["delta_nfr"] = float(0.2 + 0.6 * rng.random())
    for r in shell1:
        G.nodes[r]["role"] = "shell1"
    for r in shell2:
        G.nodes[r]["role"] = "shell2"

    return G


__all__.append("build_element_radial_pattern")
