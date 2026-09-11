#!/usr/bin/env python3
"""Example 152 — Operator contracts by primary channel and scale.

The contract registry assigns each of the 13 operators one primary state
channel (EPI, nu_f, theta, or DeltaNFR) and one execution scale (NODE or
NETWORK). These axes describe the current specification. They do not prove
that downstream fields are unchanged, that the catalog is complete, or that a
grammar role is identical to a state channel.

REMESH illustrates the network-scale boundary. Its node call is advisory; the
runtime network helper reads fixed delayed EPI snapshots and may clip the
result, while the topological helper rewrites edges from EPI. A separate finite
cyclic, unclipped fixed-delay filter has Cesaro averages that project onto
``gcd(tau_l, tau_g)``-periodic modes. That surrogate is not a runtime
``tau_g -> infinity`` limit and is referenced here rather than re-derived.

References
----------
- src/tnfr/operators/operator_contracts.py
- src/tnfr/operators/remesh.py
- theory/REMESH_INFINITY_DERIVATION.md
"""

import math
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from collections import deque

import networkx as nx
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
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
from tnfr.operators.operator_contracts import (
    OPERATOR_CONTRACTS,
    OperatorScale,
    StateChannel,
    operators_at_scale,
    operators_in_channel,
)
from tnfr.operators.remesh import apply_network_remesh, apply_topological_remesh

CLASSES = {
    "emission": Emission,
    "reception": Reception,
    "coherence": Coherence,
    "dissonance": Dissonance,
    "coupling": Coupling,
    "resonance": Resonance,
    "silence": Silence,
    "expansion": Expansion,
    "contraction": Contraction,
    "self_organization": SelfOrganization,
    "mutation": Mutation,
    "transition": Transition,
    "recursivity": Recursivity,
}

SEED = 7


def build():
    G = nx.erdos_renyi_graph(12, 0.35, seed=SEED)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(1, len(comps)):
            G.add_edge(next(iter(comps[i - 1])), next(iter(comps[i])))
    inject_defaults(G)
    rng = np.random.default_rng(SEED)
    for nd in G.nodes():
        G.nodes[nd]["EPI"] = rng.uniform(0.3, 0.6)
        G.nodes[nd]["theta"] = rng.uniform(0, 2 * math.pi)
        G.nodes[nd]["nu_f"] = rng.uniform(0.7, 1.2)
        G.nodes[nd]["delta_nfr"] = rng.uniform(-0.3, 0.3)
    return G


def node_state(G, node):
    return (
        get_attr(G.nodes[node], ALIAS_EPI, 0.0),
        get_attr(G.nodes[node], ALIAS_VF, 0.0),
        get_attr(G.nodes[node], ALIAS_THETA, 0.0),
        get_attr(G.nodes[node], ALIAS_DNFR, 0.0),
    )


def experiment_1_channel_partition():
    print("=" * 76)
    print("M1: primary-channel partition from the contract registry")
    print("=" * 76)
    print("  The 13 operators split across the four nodal-equation channels:")
    print()
    tetrad = {
        StateChannel.EPI: "the form itself",
        StateChannel.NU_F: "nu_f -> mobility (capacity lever, size-log NT arm)",
        StateChannel.THETA: "theta -> |grad phi|, K_phi (phase gradient/curvature)",
        StateChannel.DELTA_NFR: "dNFR -> Phi_s (pressure lever, count-Omega NT arm)",
    }
    for ch in StateChannel:
        ops = operators_in_channel(ch)
        print(f"  {ch.value:10s} [{tetrad[ch]}]")
        print(f"             {', '.join(ops)}")
    print()
    total = sum(len(operators_in_channel(ch)) for ch in StateChannel)
    print(f"  registry entries assigned exactly once: {total}/{len(OPERATOR_CONTRACTS)}")
    print("  -> this is specification metadata. U2/U4 grammar roles and")
    print("     realized downstream responses remain separate facts.")


def experiment_2_scale_axis():
    print()
    print("=" * 76)
    print("M2: the scale axis is U5 operational fractality (12 node + 1 network)")
    print("=" * 76)
    node_ops = operators_at_scale(OperatorScale.NODE)
    net_ops = operators_at_scale(OperatorScale.NETWORK)
    print(f"  NODE-scale ({len(node_ops)}): {', '.join(node_ops)}")
    print(f"  NETWORK-scale ({len(net_ops)}): {', '.join(net_ops)}  <- U5 fractality")
    print()
    print("  Selected two-call probe: Coherence can change node state; the")
    print("  node-level REMESH call is advisory for this fixture:")
    print()
    # A node-scale operator (Coherence) vs the network-scale REMESH at node level.
    for label, cls in (
        ("Coherence (node-scale)", Coherence),
        ("Recursivity (network-scale)", Recursivity),
    ):
        G = build()
        node = list(G.nodes())[0]
        before = node_state(G, node)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls()(G, node)
        after = node_state(G, node)
        changed = any(abs(a - b) > 1e-9 for a, b in zip(before, after))
        print(f"    {label:30s} node state changed: {changed}")
    print("  -> this illustrates the registered scale distinction; the complete")
    print("     contextual operator audit is a separate controlled protocol.")


def experiment_3_remesh_scales():
    print()
    print("=" * 76)
    print("M3: REMESH runtime paths and finite cyclic surrogate")
    print("=" * 76)
    # GLOBAL temporal: apply_network_remesh mixes EPI with its history.
    G = build()
    tau_g, tau_l, alpha = 8, 4, 0.5
    G.graph["REMESH_TAU_GLOBAL"] = tau_g
    G.graph["REMESH_TAU_LOCAL"] = tau_l
    G.graph["REMESH_ALPHA"] = alpha
    G.graph["REMESH_ALPHA_HARD"] = True
    hist = deque(maxlen=40)
    base = {n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes()}
    for s in range(25):
        hist.append(
            {n: base[n] + 0.1 * math.cos(0.3 * s + i) for i, n in enumerate(G.nodes())}
        )
    G.graph["_epi_hist"] = hist
    before = {n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        apply_network_remesh(G)
    after = {n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes()}
    n_changed = sum(1 for n in G.nodes() if abs(after[n] - before[n]) > 1e-9)
    beta, gamma, delta = (1 - alpha) ** 2, alpha * (1 - alpha), alpha
    print("  GLOBAL temporal (apply_network_remesh): mixes EPI with history")
    print(f"    nodes with EPI changed: {n_changed}/{G.number_of_nodes()}")
    print(
        f"    convex recurrence (beta,gamma,delta)=({beta},{gamma},{delta}) "
        f"sum={beta + gamma + delta} (constant-history preserving when unclipped)"
    )
    print()
    # GLOBAL topological: regenerate base from fiber.
    print("  GLOBAL topological (apply_topological_remesh): base from fiber")
    for mode in ("mst", "knn"):
        G = build()
        e_before = set(map(frozenset, G.edges()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            apply_topological_remesh(G, mode=mode, seed=SEED)
        e_after = set(map(frozenset, G.edges()))
        kept = len(e_before & e_after) / max(len(e_before), 1)
        print(
            f"    mode={mode:4s}: edges {len(e_before)}->{len(e_after)} "
            f"(kept {kept:.0%}) — topology regenerated from the EPI field"
        )
    print()
    print("  FINITE CYCLIC SURROGATE (referenced, not re-measured):")
    print("    Cesaro averages of the unclipped fixed-delay filter project onto")
    print("    gcd(tau_l,tau_g)-periodic modes. This is not a runtime tau_g->inf")
    print("    limit; fixed finite history eventually triggers the no-op guard.")
    print()
    print("  -> REMESH is the EPI-channel operator at NETWORK scale: it echoes")
    print("     the form across stored delays and can regenerate topology from")
    print("     EPI. The finite cyclic projection is a separate analysis.")


def main():
    print()
    print("#" * 76)
    print("# Example 152 - Operator Contracts by Channel and Scale")
    print("#" * 76)
    print()
    experiment_1_channel_partition()
    experiment_2_scale_axis()
    experiment_3_remesh_scales()
    print()
    print("=" * 76)
    print("Summary")
    print("=" * 76)
    print("  The canonical operator contracts reveal a two-axis structure. The")
    print("  CHANNEL axis partitions EPI / nu_f / theta / dNFR primary effects.")
    print("  SCALE axis (node vs network) is grammar rule U5: exactly one operator,")
    print("  REMESH, is network-scale -- the operational-fractality operator. REMESH")
    print("  is the EPI-channel operator whose scale is the network; it echoes the")
    print("  form across stored delays and can regenerate topology from EPI. The")
    print("  finite cyclic gcd-periodic projector is a separate surrogate. The spec")
    print("  (operator_contracts.py) is the single source of truth from which the")
    print("  audit and metadata derive. Runtime tau_g->infinity and global catalog")
    print("  completeness remain open; no open problem is closed.")
    print()


if __name__ == "__main__":
    main()
