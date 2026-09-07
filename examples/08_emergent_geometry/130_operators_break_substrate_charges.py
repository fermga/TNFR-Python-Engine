#!/usr/bin/env python3
"""Example 130 — finite operator snapshot responses in substrate coordinates.

The quantities H_sub, E_geo, E_pot and the Stokes components are conserved by
the specified harmonic substrate flow.  This example performs a different
experiment: apply each engine operator to every node of one seeded graph,
recompute its fields, and compare the resulting coordinate read-outs.

The table is a finite-probe sensitivity diagnostic.  A small change is not a
conservation theorem and a large change is not symmetry breaking: neither
establishes the operator's Jacobian pullback, behavior on other states, or a
causal one-channel mechanism.  The canonical EPI/nu_f/theta/DeltaNFR channel
labels come from operator contracts and are shown only as explanatory metadata.
No universal sector partition or operator symplecticity is inferred.

References
----------
- src/tnfr/operators/definitions.py (the 13 canonical operators)
- src/tnfr/physics/symplectic_substrate.py (geometric_sector_energy,
  potential_sector_energy, polarization_vector, substrate_hamiltonian)
- examples/08_emergent_geometry/106_per_node_polarization_geometry.py (Stokes rotation)
- examples/08_emergent_geometry/126_two_layers_base_fiber.py (operators act on the fiber)
- examples/02_physics_regimes/37_operator_tetrad_synergy.py (the dual-lever structure)
- AGENTS.md "Operator-Tetrad Synergies" (Dual-Lever Structure), "Emergent Symplectic Substrate"
"""

import copy
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.constants import inject_defaults
from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.physics.symplectic_substrate import (
    extract_phase_space_point,
    geometric_sector_energy,
    polarization_vector,
    potential_sector_energy,
    substrate_hamiltonian,
)

OPS = [
    ("AL", Emission),
    ("EN", Reception),
    ("IL", Coherence),
    ("OZ", Dissonance),
    ("UM", Coupling),
    ("RA", Resonance),
    ("SHA", Silence),
    ("VAL", Expansion),
    ("NUL", Contraction),
    ("THOL", SelfOrganization),
    ("ZHIR", Mutation),
    ("NAV", Transition),
    ("REMESH", Recursivity),
]

# Primary nodal channel from the centralized operator-contract partition.
CHANNEL = {
    "UM": "theta",
    "SHA": "nu_f",
    "VAL": "nu_f",
    "IL": "dNFR",
    "OZ": "dNFR",
    "THOL": "dNFR",
    "ZHIR": "theta",
    "NAV": "dNFR",
    "NUL": "nu_f",
    "AL": "EPI",
    "EN": "EPI",
    "RA": "EPI",
    "REMESH": "EPI",
}


def _make_graph(seed=42):
    """Seeded connected probe with an explicitly U3-compatible phase band."""
    G = nx.erdos_renyi_graph(20, 0.25, seed=seed)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(1, len(comps)):
            G.add_edge(next(iter(comps[i - 1])), next(iter(comps[i])))
    inject_defaults(G)
    rng = np.random.default_rng(seed)
    for nd in G.nodes():
        G.nodes[nd]["phase"] = rng.uniform(0.8, 1.2)
        G.nodes[nd]["theta"] = G.nodes[nd]["phase"]
        G.nodes[nd]["delta_nfr"] = rng.uniform(-0.3, 0.3)
        G.nodes[nd]["nu_f"] = rng.uniform(0.8, 1.2)
    return G


def _charges(G):
    p = extract_phase_space_point(G)
    pol = polarization_vector(p)
    return {
        "H_sub": substrate_hamiltonian(p),
        "E_geo": geometric_sector_energy(p),
        "E_pot": potential_sector_energy(p),
        "P_1": pol["p_1"],
        "P_2": pol["p_2"],
        "P_3": pol["p_3"],
    }


def _apply_all(G0, cls):
    """Apply an operator to every node on a fresh copy; return the charges."""
    G = copy.deepcopy(G0)
    op = cls()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nd in list(G.nodes()):
            op(G, nd)
    return _charges(G)


def experiment_1_breaking_map():
    """Report one seeded operator-to-coordinate response table."""
    print("=" * 74)
    print("EXPERIMENT 1: Finite Operator -> Coordinate Response")
    print("=" * 74)
    print("Apply each operator to every node; measure relative snapshot changes.")
    print()
    G0 = _make_graph()
    c0 = _charges(G0)
    print(
        f"  baseline: H_sub={c0['H_sub']:.2f} E_geo={c0['E_geo']:.2f} "
        f"E_pot={c0['E_pot']:.3f}"
    )
    print()
    print(
        f"  {'op':>6} {'channel':>8} | {'dH':>6} {'dEgeo':>6} {'dEpot':>6} "
        f"{'dP1':>6} {'dP2':>6} {'dP3':>6}"
    )
    print("  " + "-" * 60)
    for glyph, cls in OPS:
        c1 = _apply_all(G0, cls)
        d = {k: abs(c1[k] - c0[k]) / (abs(c0[k]) + 1e-9) for k in c0}
        print(
            f"  {glyph:>6} {CHANNEL[glyph]:>8} | {d['H_sub']:>6.2f} "
            f"{d['E_geo']:>6.2f} {d['E_pot']:>6.2f} {d['P_1']:>6.2f} "
            f"{d['P_2']:>6.2f} {d['P_3']:>6.2f}"
        )
    print()
    print("  -> these rows describe this seed, graph, gains and application order.")
    print("     They neither classify conserved quantities nor prove operator maps.")


def experiment_2_sector_map():
    """Compare absolute coordinate-sector responses on the same finite probe."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Absolute Coordinate-Sector Response")
    print("=" * 74)
    print("E_geo lives in the phase channel Psi=(K_phi,J_phi); E_pot in the dNFR")
    print("channel (Phi_s,J_dNFR). Absolute changes (E_pot baseline is small, so")
    print("the relative view exaggerates it).")
    print()
    G0 = _make_graph()
    c0 = _charges(G0)
    print(f"  baseline ABS: E_geo={c0['E_geo']:.2f} E_pot={c0['E_pot']:.3f}")
    print()
    print(
        f"  {'op':>6} {'channel':>8} | {'|dE_geo|':>9} {'|dE_pot|':>9} "
        f"{'larger response':>14}"
    )
    print("  " + "-" * 52)
    for glyph, cls in OPS:
        c1 = _apply_all(G0, cls)
        dgeo = abs(c1["E_geo"] - c0["E_geo"])
        dpot = abs(c1["E_pot"] - c0["E_pot"])
        if dgeo < 1e-6 and dpot < 1e-6:
            sector = "no change"
        elif dgeo > dpot:
            sector = "GEOMETRIC"
        else:
            sector = "POTENTIAL"
        print(
            f"  {glyph:>6} {CHANNEL[glyph]:>8} | {dgeo:>9.4f} {dpot:>9.4f} "
            f"{sector:>14}"
        )
    print()
    print("  -> zeroes and dominant sectors are observations on this probe.")
    print("     Repeatability or exact global identities require separate tests.")


def experiment_3_lever_grouping():
    """Aggregate the finite response by canonical primary channel."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Response Grouped by Primary Nodal Channel")
    print("=" * 74)
    print("Group operators by their contract's primary nodal channel and average")
    print("the absolute coordinate response on the declared seeded probe.")
    print()
    G0 = _make_graph()
    c0 = _charges(G0)
    classes = {
        "EPI (AL/EN/RA/REMESH)": ["AL", "EN", "RA", "REMESH"],
        "nu_f (SHA/VAL/NUL)": ["SHA", "VAL", "NUL"],
        "theta (UM/ZHIR)": ["UM", "ZHIR"],
        "dNFR (IL/OZ/THOL/NAV)": ["IL", "OZ", "THOL", "NAV"],
    }
    by_glyph = {g: _apply_all(G0, cls) for g, cls in OPS}
    print(f"  {'class':38s} {'|dE_geo|':>9} {'|dE_pot|':>9} {'sector':>11}")
    print("  " + "-" * 70)
    for label, glyphs in classes.items():
        mgeo = float(np.mean([abs(by_glyph[g]["E_geo"] - c0["E_geo"]) for g in glyphs]))
        mpot = float(np.mean([abs(by_glyph[g]["E_pot"] - c0["E_pot"]) for g in glyphs]))
        if mgeo < 1e-6 and mpot < 1e-6:
            sec = "no change"
        elif mgeo > mpot:
            sec = "GEOMETRIC"
        else:
            sec = "POTENTIAL"
        print(f"  {label:38s} {mgeo:>9.4f} {mpot:>9.4f} {sec:>11}")
    print()
    print("  -> grouping can reveal probe-specific patterns, but the primary")
    print("     channel does not by itself determine a conserved-charge sector.")


def main():
    print()
    print("  TNFR Example 130: Operator Snapshot Responses")
    print("  Finite graph-field changes in auxiliary substrate coordinates")
    print("  ===============================================================")
    print()
    experiment_1_breaking_map()
    experiment_2_sector_map()
    experiment_3_lever_grouping()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("Each operator produces a reproducible coordinate response for the")
    print("declared seeded probe. The quantities are charges only along H_sub's")
    print("harmonic flow. This example does not prove operator symplecticity,")
    print("global preservation, or a universal channel-to-sector law.")


if __name__ == "__main__":
    main()
