from __future__ import annotations

import math
from typing import Any, Dict

try:  # lightweight optional
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - imported only when needed
    nx = None  # type: ignore


def _wrap_angle(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def _neighbor_phase_mean(G: Any, n: Any) -> float:
    phis = [float(G.nodes[j].get("phase", 0.0)) for j in G.neighbors(n)]
    if not phis:
        return float(G.nodes[n].get("phase", 0.0))
    c = complex(
        sum(math.cos(p) for p in phis) / len(phis),
        sum(math.sin(p) for p in phis) / len(phis),
    )
    return math.atan2(c.imag, c.real)


def apply_synthetic_activation_sequence(G: Any, *, alpha: float = 0.25, dnfr_factor: float = 0.9) -> dict:
    """Emulate minimal [AL, RA, IL] effects in a single, safe step.

    - RA-like: move phase a fraction alpha towards neighbor circular mean
    - IL-like: reduce ΔNFR magnitude by dnfr_factor
    - AL is implicit for demo purposes (no EPI mutation)

    Parameters
    ----------
    G : Graph-like
        Graph with 'phase' and 'delta_nfr' node attributes
    alpha : float
        Fraction of phase transport towards neighbor mean (0..1)
    dnfr_factor : float
        Multiplicative factor for delta_nfr (>1 increases, <1 reduces)

    Returns
    -------
    dict
        Parameters used for the step (alpha, dnfr_factor)
    """
    # Phase transport step
    new_phase: Dict[Any, float] = {}
    for n in G.nodes():
        phi = float(G.nodes[n].get("phase", 0.0))
        mn = _neighbor_phase_mean(G, n)
        d = _wrap_angle(mn - phi)
        new_phase[n] = _wrap_angle(phi + alpha * d)

    for n, v in new_phase.items():
        G.nodes[n]["phase"] = float(v)

    # Coherence step on structural pressure
    for n in G.nodes():
        dn = float(G.nodes[n].get("delta_nfr", 0.0))
        G.nodes[n]["delta_nfr"] = float(dnfr_factor * dn)

    return {"alpha": alpha, "dnfr_factor": dnfr_factor}


def build_ws_graph_with_seed(n: int = 50, k: int = 4, p: float = 0.1, seed: int = 42):
    """Build a Watts–Strogatz graph and seed 'phase'/'delta_nfr' for demos.

    Returns a NetworkX graph if networkx is available, else raises RuntimeError.
    """
    if nx is None:
        raise RuntimeError("networkx is required to build demo graph")
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    import random as _random

    rng = _random.Random(seed)
    for i in G.nodes():
        G.nodes[i]["phase"] = float(2.0 * math.pi * rng.random())
        G.nodes[i]["delta_nfr"] = float(0.2 + 0.6 * rng.random())
    return G


def build_radial_atom_graph(n_shell: int = 24, *, seed: int = 42):
    """Build a simple hydrogen-like radial graph (nucleus + ring shell).

    Topology:
    - One central node ("nucleus") connected to all ring nodes (star)
    - Ring nodes connected in a cycle (angular coupling)

    Node attributes are seeded for reproducibility:
    - 'phase' ~ Uniform[0, 2π)
    - 'delta_nfr' ~ Uniform[0.2, 0.8)

    Returns a NetworkX graph if networkx is available, else raises RuntimeError.
    """
    if nx is None:
        raise RuntimeError("networkx is required to build demo graph")
    if n_shell < 3:
        n_shell = 3

    G = nx.Graph()
    nucleus = 0
    G.add_node(nucleus)

    # Add ring nodes 1..n_shell and connect cycle
    ring_nodes = list(range(1, n_shell + 1))
    G.add_nodes_from(ring_nodes)
    # Cycle connections
    for i in range(n_shell):
        a = ring_nodes[i]
        b = ring_nodes[(i + 1) % n_shell]
        G.add_edge(a, b)
    # Star connections nucleus↔ring
    for r in ring_nodes:
        G.add_edge(nucleus, r)

    # Seeded attributes
    import random as _random
    rng = _random.Random(seed)
    for i in G.nodes():
        G.nodes[i]["phase"] = float(2.0 * math.pi * rng.random())
        G.nodes[i]["delta_nfr"] = float(0.2 + 0.6 * rng.random())

    # Tag nucleus for reference
    G.nodes[nucleus]["role"] = "nucleus"
    for r in ring_nodes:
        G.nodes[r]["role"] = "shell"

    return G


def build_element_radial_graph(Z: int, *, seed: int = 42):
    """Delegate to tnfr.physics.patterns.build_element_radial_pattern.

    This keeps examples in sync with the canonical physics implementation
    while preserving the public helper used by examples and notebooks.
    """
    if nx is None:
        raise RuntimeError("networkx is required to build demo graph")
    # Lazy import to avoid hard dependency during module import time
    from tnfr.physics.patterns import build_element_radial_pattern

    return build_element_radial_pattern(Z, seed=seed)


def build_diatomic_molecule_graph(Z1: int, Z2: int, *, seed: int = 42, bond_links: int = 1):
    """Compose two element-like radial graphs and add coupling edges to form a diatomic molecule.

    Telemetry-only builder: constructs a combined NetworkX graph with seeded 'phase' and 'delta_nfr'.
    Coupling edges emulate UM/RA at topology level; no operator dynamics are executed here.

    Roles:
      - Nodes retain 'role' from their element graph, with added 'atom' tag: 'A' or 'B'.
      - Graph attribute 'molecule' is set to f"{Z1}-{Z2}".
    """
    if nx is None:
        raise RuntimeError("networkx is required to build demo graph")

    GA = build_element_radial_graph(Z1, seed=seed)
    GB = build_element_radial_graph(Z2, seed=seed + 1)

    # Relabel B to avoid collisions
    offset = max(GA.nodes()) + 1 if GA.nodes() else 0
    GB_rel = nx.relabel_nodes(GB, {n: n + offset for n in GB.nodes()})

    # Merge graphs
    G = nx.Graph()
    G.update(GA)
    G.update(GB_rel)

    # Tag atoms
    for n in GA.nodes():
        G.nodes[n]['atom'] = 'A'
    for n in GB_rel.nodes():
        G.nodes[n]['atom'] = 'B'

    # Identify candidate bond sites: prefer shell1, else any non-nucleus
    def _candidates(H):
        shell1 = [n for n, d in H.nodes(data=True) if d.get('role') == 'shell1']
        if shell1:
            return shell1
        shell = [n for n, d in H.nodes(data=True) if d.get('role') in ('shell', 'shell2')]
        if shell:
            return shell
        # fallback: exclude nucleus
        return [n for n, d in H.nodes(data=True) if d.get('role') != 'nucleus']

    A_candidates = _candidates(GA)
    B_candidates = [n for n in GB_rel.nodes() if GB_rel.nodes[n].get('role') == 'shell1']
    if not B_candidates:
        B_candidates = _candidates(GB_rel)

    # Simple deterministic picks (seed order)
    if not A_candidates or not B_candidates:
        return G
    a0 = sorted(A_candidates)[0]
    b0 = sorted(B_candidates)[0]
    G.add_edge(a0, b0)
    # Optionally add more parallel links to emulate stronger coupling
    if bond_links > 1 and len(A_candidates) > 1 and len(B_candidates) > 1:
        a1 = sorted(A_candidates)[1]
        b1 = sorted(B_candidates)[1]
        G.add_edge(a1, b1)

    G.graph['molecule'] = f"{Z1}-{Z2}"
    return G


def build_triatomic_molecule_graph(
    Z1: int,
    Z2: int,
    Z3: int,
    *,
    seed: int = 42,
    bond_links: int = 1,
    central: str = "B",
):
    """Compose three element-like radial graphs and add coupling edges to form a triatomic molecule.

    Telemetry-only builder: constructs a combined NetworkX graph with seeded 'phase' and 'delta_nfr'.
    Coupling edges emulate UM/RA at topology level; no operator dynamics are executed here.

        Topologies (conceptual):
      - central="B" (default): V-shape with B as the central atom (links A–B and B–C)
      - central="A" or "C": use that atom as central instead

    Roles:
      - Nodes retain 'role' from their element graph, with added 'atom' tag: 'A', 'B', or 'C'.
      - Graph attribute 'molecule' is set to f"{Z1}-{Z2}-{Z3}".
            - Graph attributes include telemetry-only geometry hints:
                    - 'central_atom': one of {'A','B','C'} actually used as central
                    - 'geometry_hint': one of {'linear','bent','unknown'}
                    - 'angle_est_deg': float, rough central angle estimate (no dynamics)

    Parameters
    ----------
    Z1, Z2, Z3 : int
        Atomic numbers for the three element-like graphs.
    seed : int
        Base seed for reproducible attribute seeding (phase, delta_nfr).
    bond_links : int
        Number of parallel shell-to-shell links to add per bond (>=1).
    central : {"A","B","C"}
        Which atom serves as the central connector (default "B").
    """
    if nx is None:
        raise RuntimeError("networkx is required to build demo graph")

    GA = build_element_radial_graph(Z1, seed=seed)
    GB = build_element_radial_graph(Z2, seed=seed + 1)
    GC = build_element_radial_graph(Z3, seed=seed + 2)

    # Relabel B and C to avoid collisions with A
    offB = (max(GA.nodes()) + 1) if GA.nodes() else 0
    GB_rel = nx.relabel_nodes(GB, {n: n + offB for n in GB.nodes()})
    offC = (max(GB_rel.nodes()) + 1) if GB_rel.nodes() else (offB + 1)
    GC_rel = nx.relabel_nodes(GC, {n: n + offC for n in GC.nodes()})

    # Merge graphs
    G = nx.Graph()
    G.update(GA)
    G.update(GB_rel)
    G.update(GC_rel)

    # Tag atoms
    for n in GA.nodes():
        G.nodes[n]['atom'] = 'A'
    for n in GB_rel.nodes():
        G.nodes[n]['atom'] = 'B'
    for n in GC_rel.nodes():
        G.nodes[n]['atom'] = 'C'

    def _candidates(H):
        shell1 = [n for n, d in H.nodes(data=True) if d.get('role') == 'shell1']
        if shell1:
            return shell1
        shell = [n for n, d in H.nodes(data=True) if d.get('role') in ('shell', 'shell2')]
        if shell:
            return shell
        return [n for n, d in H.nodes(data=True) if d.get('role') != 'nucleus']

    A_cand = _candidates(GA)
    B_cand = _candidates(GB_rel)
    C_cand = _candidates(GC_rel)

    if not (A_cand and B_cand and C_cand):
        G.graph['molecule'] = f"{Z1}-{Z2}-{Z3}"
        return G

    # Choose central set and connect pairs
    central = central.upper()

    def _pick_first_two(lst: list[int]) -> list[int]:
        s = sorted(lst)
        return [s[0]] if len(s) == 1 else s[:2]

    if central == "A":
        A_sel = _pick_first_two(A_cand)
        B_sel = _pick_first_two(B_cand)
        C_sel = _pick_first_two(C_cand)
        # Connect A–B and A–C
        G.add_edge(A_sel[0], B_sel[0])
        G.add_edge(A_sel[0 if len(A_sel) == 1 else 1], C_sel[0])
        if bond_links > 1 and len(B_sel) > 1 and len(C_sel) > 1:
            G.add_edge(A_sel[0], B_sel[1])
            G.add_edge(A_sel[0 if len(A_sel) == 1 else 1], C_sel[1])
        central_atom = "A"
    elif central == "C":
        A_sel = _pick_first_two(A_cand)
        B_sel = _pick_first_two(B_cand)
        C_sel = _pick_first_two(C_cand)
        # Connect C–B and C–A
        G.add_edge(C_sel[0], B_sel[0])
        G.add_edge(C_sel[0 if len(C_sel) == 1 else 1], A_sel[0])
        if bond_links > 1 and len(B_sel) > 1 and len(A_sel) > 1:
            G.add_edge(C_sel[0], B_sel[1])
            G.add_edge(C_sel[0 if len(C_sel) == 1 else 1], A_sel[1])
        central_atom = "C"
    else:  # default central = B
        A_sel = _pick_first_two(A_cand)
        B_sel = _pick_first_two(B_cand)
        C_sel = _pick_first_two(C_cand)
        # Connect A–B and B–C
        G.add_edge(A_sel[0], B_sel[0])
        G.add_edge(B_sel[0 if len(B_sel) == 1 else 1], C_sel[0])
        if bond_links > 1 and len(A_sel) > 1 and len(C_sel) > 1:
            G.add_edge(A_sel[1], B_sel[0])
            G.add_edge(B_sel[-1], C_sel[1])
        central_atom = "B"

    # Telemetry-only geometry hint (simple heuristic, no dynamics)
    # If terminals are identical (Z1==Z3), classify based on central Z:
    #   - C (Z=6) → linear (CO2-like)
    #   - O (Z=8) → bent (H2O-like)
    # Otherwise unknown. Angle estimates are canonical demo values.
    geometry_hint = "unknown"
    angle_est_deg = 120.0
    if central_atom == "B":
        if Z1 == Z3 and Z2 == 6:  # central carbon (CO2-like)
            geometry_hint = "linear"
            angle_est_deg = 180.0
        elif Z1 == Z3 and Z2 == 8:  # central oxygen (H2O-like)
            geometry_hint = "bent"
            angle_est_deg = 104.5
        elif Z1 == Z3 and Z2 == 4:  # central beryllium (BeX2)
            geometry_hint = "linear"
            angle_est_deg = 180.0
        elif Z1 == Z3 and Z2 == 7:  # central nitrogen (NO2-like)
            geometry_hint = "bent"
            angle_est_deg = 120.0
    elif central_atom == "A":
        if Z2 == Z3 and Z1 == 6:
            geometry_hint = "linear"
            angle_est_deg = 180.0
        elif Z2 == Z3 and Z1 == 8:
            geometry_hint = "bent"
            angle_est_deg = 104.5
        elif Z2 == Z3 and Z1 == 4:
            geometry_hint = "linear"
            angle_est_deg = 180.0
        elif Z2 == Z3 and Z1 == 7:
            geometry_hint = "bent"
            angle_est_deg = 120.0
    elif central_atom == "C":
        if Z1 == Z2 and Z3 == 6:
            geometry_hint = "linear"
            angle_est_deg = 180.0
        elif Z1 == Z2 and Z3 == 8:
            geometry_hint = "bent"
            angle_est_deg = 104.5
        elif Z1 == Z2 and Z3 == 4:
            geometry_hint = "linear"
            angle_est_deg = 180.0
        elif Z1 == Z2 and Z3 == 7:
            geometry_hint = "bent"
            angle_est_deg = 120.0

    G.graph['molecule'] = f"{Z1}-{Z2}-{Z3}"
    G.graph['central_atom'] = central_atom
    G.graph['geometry_hint'] = geometry_hint
    G.graph['angle_est_deg'] = float(angle_est_deg)
    return G
