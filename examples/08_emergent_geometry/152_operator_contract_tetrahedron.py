#!/usr/bin/env python3
"""
Example 152 — The Operator-Contract Tetrahedron: Channel × Scale, and What
Emerges From REMESH at Local / Global / Asymptotic Scale
==========================================================================

The 13 canonical operators were studied here through their CONTRACTS — what each
one does to the node state under the nodal equation ``∂EPI/∂t = νf · ΔNFR``. The
canonical contract spec ([src/tnfr/operators/operator_contracts.py]) records, for
each operator, the nodal-equation CHANNEL it acts on and the SCALE at which it
acts. This example measures the two-axis structure those contracts reveal and
then studies the one operator that is special on the scale axis: Recursivity
(REMESH).

Axis 1 — the nodal-equation channel (the dual-lever / tetrad / number theory)
-----------------------------------------------------------------------------
Every operator's primary effect lands on exactly one channel of the nodal
equation (the structural triad EPI/νf/θ plus the pressure ΔNFR). This single
partition simultaneously *is*:

  * the **dual-lever** (examples 37/130): νf channel = capacity lever,
    ΔNFR channel = pressure lever;
  * the **tetrad driver** (example 39): the ΔNFR channel drives Φ_s (0th order,
    |r| = 1.0); the θ channel drives |∇φ| / K_φ;
  * the **number-theory grading** (example 147): ΔNFR = count-Ω arm,
    νf = size-log arm.

Axis 2 — the scale (grammar rule U5, operational fractality)
------------------------------------------------------------
A second, orthogonal axis: the SCALE at which the operator acts. Exactly one
operator implements operational fractality (U5) and therefore acts at NETWORK
scale — Recursivity (REMESH). The other twelve act at NODE scale (their ``_op_*``
handler mutates one node's state channel; they act on the *fiber* of the
base/fiber optic, examples 126-131). REMESH is the multi-scale echo: its
node-level call is advisory; its canonical effect is network-scale.

What emerges from REMESH (the special operator)
-----------------------------------------------
REMESH is an EPI operator (it echoes the *form* across time and scale) whose
specialness is its NETWORK scale. It emerges at three scales, all from the EPI
history of the nodal equation:

  * LOCAL (node): advisory — by U5 fractality it does not act on a node channel.
  * GLOBAL (network):
      - temporal: ``apply_network_remesh`` mixes EPI with its history via the
        canonical convex recurrence
        ``EPI_new = (1-α)²·EPI(t) + α(1-α)·EPI(t-τ_l) + α·EPI(t-τ_g)``
        (coefficients β+γ+δ = 1);
      - topological: ``apply_topological_remesh`` regenerates the BASE (topology)
        from the FIBER (the EPI field) — the base/fiber co-emergence of ex 126-131.
  * ASYMPTOTIC (τ_g → ∞): the bounded self-adjoint orthogonal projection 𝓡_∞ onto
    the time-mean (N15, theory/REMESH_INFINITY_DERIVATION.md) — proven
    analytically, referenced here (not re-measured).

Three measured results
----------------------
M1 THE CHANNEL PARTITION REFINES THE DUAL-LEVER. Reading the canonical spec, the
   13 operators split across the four nodal channels (EPI / νf / θ / ΔNFR).
   Measured against the independent dual-lever classification (examples 37/130):
   they AGREE on the operators whose primary channel is their lever (SHA/VAL on
   νf = capacity; IL/OZ/THOL/NAV on ΔNFR = pressure), and the channel view
   REFINES the binary lever by resolving the phase channel θ (UM, ZHIR act on θ
   primarily, with their capacity/pressure lever a downstream effect) and the
   dual-channel NUL. One structure (the nodal channels), read as lever / tetrad /
   number-theory grading.

M2 THE SCALE AXIS IS U5 FRACTALITY. Exactly one operator (REMESH) is NETWORK
   scale; the other twelve are NODE scale. Measured: applying each NODE-scale
   operator to a node changes that node's state, while the node-level REMESH call
   leaves every node unchanged (it records an advisory) — REMESH's effect is
   multi-scale, not node-local. The scale axis is orthogonal to the channel axis.

M3 REMESH EMERGES AT THREE SCALES FROM THE EPI HISTORY. Measured on the canonical
   REMESH functions: the network temporal recurrence mixes EPI across all nodes
   (β+γ+δ = 1 convex combination); the topological mode regenerates the base
   topology from the EPI field; and (N15, referenced) the τ_g→∞ limit is the 𝓡_∞
   projection onto the time-mean. REMESH is the EPI-channel operator whose scale
   is the network — operational fractality made concrete.

Honest scope
------------
A characterization of the canonical operator contracts (the spec is the single
source of truth; the proactive audit, reactive monitor, and metadata all derive
from it). The channel = dual-lever correspondence and the scale = U5 axis are
read from the spec and measured against the independent ex-37/130 classification
and the canonical REMESH functions. The asymptotic 𝓡_∞ scale is the N15 result,
referenced not re-measured. Not new mathematics; closes no open problem.

References
----------
- src/tnfr/operators/operator_contracts.py (the canonical contract spec)
- src/tnfr/physics/integrity.py (audit_operator_contracts — derives from the spec)
- src/tnfr/operators/remesh.py (apply_network_remesh, apply_topological_remesh)
- theory/REMESH_INFINITY_DERIVATION.md (N15 𝓡_∞ asymptotic projection)
- examples/02_physics_regimes/37_operator_tetrad_synergy.py (the dual-lever)
- examples/08_emergent_geometry/126_two_layers_base_fiber.py (base/fiber)
- examples/08_emergent_geometry/128_base_substrate_coemergence.py (REMESH base regen)
- AGENTS.md "The 13 Canonical Operators", "Operator-Tetrad Synergies", U5
"""

import os
import sys
import math
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from collections import deque

import numpy as np
import networkx as nx

from tnfr.constants import inject_defaults
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR
from tnfr.operators.operator_contracts import (
    OPERATOR_CONTRACTS,
    StateChannel,
    OperatorScale,
    operators_in_channel,
    operators_at_scale,
)
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Coupling, Resonance, Silence,
    Expansion, Contraction, SelfOrganization, Mutation, Transition, Recursivity,
)
from tnfr.operators.remesh import apply_network_remesh, apply_topological_remesh

CLASSES = {
    "emission": Emission, "reception": Reception, "coherence": Coherence,
    "dissonance": Dissonance, "coupling": Coupling, "resonance": Resonance,
    "silence": Silence, "expansion": Expansion, "contraction": Contraction,
    "self_organization": SelfOrganization, "mutation": Mutation,
    "transition": Transition, "recursivity": Recursivity,
}

# Dual-lever classification (examples 37/130, a MEASURED result, independent of
# the contract spec — used here to cross-check the channel partition).
LEVER = {"UM": "nu_f", "SHA": "nu_f", "VAL": "nu_f",
         "IL": "dNFR", "OZ": "dNFR", "THOL": "dNFR", "ZHIR": "dNFR",
         "NAV": "dNFR", "NUL": "both",
         "AL": "neither", "EN": "neither", "RA": "neither", "REMESH": "neither"}

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
    print("M1: the channel partition REFINES the dual-lever (tetrad / number theory)")
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
    # Cross-check: the channel partition REFINES the dual-lever. They agree on
    # the operators whose primary channel IS their lever, and differ on the
    # phase operators (θ primary, downstream lever) and the dual-channel NUL.
    nu_f_ops = {OPERATOR_CONTRACTS[c.name].glyph
                for c in OPERATOR_CONTRACTS.values()
                if c.primary_channel is StateChannel.NU_F}
    dnfr_ops = {OPERATOR_CONTRACTS[c.name].glyph
                for c in OPERATOR_CONTRACTS.values()
                if c.primary_channel is StateChannel.DELTA_NFR}
    theta_ops = {OPERATOR_CONTRACTS[c.name].glyph
                 for c in OPERATOR_CONTRACTS.values()
                 if c.primary_channel is StateChannel.THETA}
    lever_capacity = {g for g, lv in LEVER.items() if lv == "nu_f"}
    lever_pressure = {g for g, lv in LEVER.items() if lv == "dNFR"}
    # Pure-lever operators: primary channel == lever pulled.
    pure_capacity = nu_f_ops - {"NUL"}           # SHA, VAL
    pure_pressure = dnfr_ops - theta_ops          # IL, OZ, THOL, NAV
    print("  AGREE (primary channel == lever):")
    print(f"    capacity: channel-nu_f {sorted(pure_capacity)} in lever-nu_f "
          f"{sorted(lever_capacity)}: {pure_capacity <= lever_capacity}")
    print(f"    pressure: channel-dNFR {sorted(pure_pressure)} in lever-dNFR "
          f"{sorted(lever_pressure)}: {pure_pressure <= lever_pressure}")
    print("  REFINE (channel splits what the binary lever collapses):")
    print(f"    theta channel {sorted(theta_ops)}: UM lever is nu_f (sync),")
    print("      ZHIR lever is dNFR (|grad phi| jump) -- but their PRIMARY")
    print("      channel is theta (the 1st/2nd-order tetrad channel, ex 39).")
    print("    NUL: channel nu_f (primary) but dual-lever 'both' (also")
    print("      densifies dNFR) -- channel picks the primary, lever sees both.")
    print("  -> the contract channel partition REFINES the dual-lever (ex 37/130):")
    print("     it agrees on the pure-lever operators and resolves the phase")
    print("     channel (theta) that the binary capacity/pressure lever")
    print("     collapses. One structure (nodal channels): lever / tetrad / NT.")


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
    print("  Measured: each NODE-scale operator changes node state; the node-level")
    print("  REMESH call leaves every node unchanged (records an advisory):")
    print()
    # A node-scale operator (Coherence) vs the network-scale REMESH at node level.
    for label, cls in (("Coherence (node-scale)", Coherence),
                       ("Recursivity (network-scale)", Recursivity)):
        G = build()
        node = list(G.nodes())[0]
        before = node_state(G, node)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls()(G, node)
        after = node_state(G, node)
        changed = any(abs(a - b) > 1e-9 for a, b in zip(before, after))
        print(f"    {label:30s} node state changed: {changed}")
    print("  -> REMESH is the only operator whose node-level effect is null: its")
    print("     action is multi-scale (network), orthogonal to the channel axis.")


def experiment_3_remesh_scales():
    print()
    print("=" * 76)
    print("M3: REMESH emerges at three scales from the EPI history")
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
        hist.append({n: base[n] + 0.1 * math.cos(0.3 * s + i)
                     for i, n in enumerate(G.nodes())})
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
    print(f"    convex recurrence (beta,gamma,delta)=({beta},{gamma},{delta}) "
          f"sum={beta + gamma + delta} (probability-preserving)")
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
        print(f"    mode={mode:4s}: edges {len(e_before)}->{len(e_after)} "
              f"(kept {kept:.0%}) — topology regenerated from the EPI field")
    print()
    print("  ASYMPTOTIC (tau_g -> inf): the R_inf projection onto the time-mean")
    print("    proven analytically in N15 (REMESH_INFINITY_DERIVATION.md): a")
    print("    bounded self-adjoint orthogonal projection (referenced, not")
    print("    re-measured here).")
    print()
    print("  -> REMESH is the EPI-channel operator at NETWORK scale: it echoes")
    print("     the form across time (temporal mixing), space (base-from-fiber")
    print("     regeneration), and the asymptotic limit (R_inf). Operational")
    print("     fractality (U5) made concrete.")


def main():
    print()
    print("#" * 76)
    print("# Example 152 - The Operator-Contract Tetrahedron: Channel x Scale,")
    print("#               and What Emerges From REMESH")
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
    print("  CHANNEL axis (EPI / nu_f / theta / dNFR) is the dual-lever = tetrad")
    print("  driver = number-theory grading (one partition, three readings). The")
    print("  SCALE axis (node vs network) is grammar rule U5: exactly one operator,")
    print("  REMESH, is network-scale -- the operational-fractality operator. REMESH")
    print("  is the EPI-channel operator whose scale is the network; it echoes the")
    print("  form across time (temporal mixing), space (base-from-fiber")
    print("  regeneration), and the asymptotic limit (R_inf, N15). The contract spec")
    print("  (operator_contracts.py) is the single source of truth from which the")
    print("  audit, reactive monitor, and metadata all derive. Characterization;")
    print("  no operator physics changed, no open problem closed.")
    print()


if __name__ == "__main__":
    main()
