#!/usr/bin/env python3
"""
Title: Social Network Dynamics - Information Propagation and Opinion Formation

Problem: In social networks, individuals form opinions through exposure to
information from their social contacts. How do opinions spread, stabilize,
or fragment? When does consensus emerge vs. polarization?

TNFR Approach: Model individuals as NFR nodes where:
- EPI represents opinion/belief structure
- νf (Hz_str) is openness to opinion change
- Phase represents alignment with prevailing narrative
- Emission models sharing information/opinions
- Reception models exposure to others' views
- Resonance models consensus building (opinions align)
- Dissonance models conflict/debate (opinions clash)

Key Operators:
- Emission (AL): Individual shares their opinion
- Reception (EN): Individual exposed to others' opinions
- Resonance (RA): Opinion alignment, consensus building
- Dissonance (OZ): Opinion conflict, debate
- Coherence (IL): Opinion stabilization
- Coupling (UM): Relationship formation, trust networks

Relevant Metrics:
- C(t): Social coherence (level of consensus)
- Si: Opinion stability (resistance to influence)
- Phase coherence: Opinion alignment across network
- ΔNFR: Pressure to change opinion

Expected Behavior:
- Initial diversity of opinions (low C(t))
- Information spreads through network
- Some opinions resonate and amplify (consensus)
- Conflicting opinions create dissonance (debate)
- Final state shows either consensus or polarization
- Stable individuals (high Si) are opinion leaders

Run:
    python docs/source/examples/social_network_dynamics.py
"""

from tnfr import create_nfr, run_sequence
from tnfr.dynamics import run
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si
from tnfr.structural import (
    Coupling,
    Coherence,
    Dissonance,
    Emission,
    Reception,
    Resonance,
    Silence,
)
from tnfr.trace import register_trace
from tnfr.constants import inject_defaults
import math


def run_example() -> None:
    """Model opinion dynamics in a social network under information flow."""
    
    print("=" * 70)
    print("TNFR Social Network Dynamics: Opinion Formation & Propagation")
    print("=" * 70)
    print()
    
    # 1. PROBLEM SETUP: Creating a social network
    # --------------------------------------------
    # Model a small online community discussing a contentious topic
    # Different personality types have different responses to information
    
    print("Phase 1: Initializing social network...")
    print("Creating 8 individuals with diverse initial opinions")
    print()
    
    # Opinion Leaders: High stability (high Si), influence others
    # Starting with strong, coherent opinions (high EPI)
    G, _ = create_nfr(
        "Leader_A",
        epi=0.45,   # Strong opinion structure
        vf=0.8,     # Moderate openness to change
        theta=0.0   # Aligned with one narrative
    )
    
    create_nfr(
        "Leader_B",
        epi=0.42,
        vf=0.75,
        theta=math.pi,  # Opposite narrative (polarized)
        graph=G
    )
    
    # Followers: Moderate stability, receptive to leaders
    create_nfr(
        "Follower_1",
        epi=0.25,   # Moderate opinion strength
        vf=1.1,     # More open to change
        theta=0.3,  # Leaning toward Leader_A
        graph=G
    )
    
    create_nfr(
        "Follower_2",
        epi=0.23,
        vf=1.15,
        theta=-0.4,  # Leaning toward Leader_A
        graph=G
    )
    
    create_nfr(
        "Follower_3",
        epi=0.27,
        vf=1.0,
        theta=2.8,   # Leaning toward Leader_B
        graph=G
    )
    
    # Undecided/Neutral individuals: Low initial opinion structure
    create_nfr(
        "Neutral_1",
        epi=0.15,    # Weak initial opinion
        vf=1.3,      # Very open to influence
        theta=1.5,   # Neutral phase
        graph=G
    )
    
    create_nfr(
        "Neutral_2",
        epi=0.18,
        vf=1.25,
        theta=1.6,
        graph=G
    )
    
    # Bridge person: Connects different groups
    create_nfr(
        "Bridge",
        epi=0.30,
        vf=0.95,
        theta=0.8,   # Between narratives
        graph=G
    )
    
    # Add social connections (who influences whom)
    # Leaders have broad influence
    G.add_edge("Leader_A", "Follower_1")
    G.add_edge("Leader_A", "Follower_2")
    G.add_edge("Leader_A", "Bridge")
    
    G.add_edge("Leader_B", "Follower_3")
    G.add_edge("Leader_B", "Bridge")
    
    # Followers connect to each other
    G.add_edge("Follower_1", "Follower_2")
    G.add_edge("Follower_1", "Neutral_1")
    G.add_edge("Follower_3", "Neutral_2")
    
    # Bridge connects different camps
    G.add_edge("Bridge", "Neutral_1")
    G.add_edge("Bridge", "Neutral_2")
    
    # Store social metadata
    roles = {
        "Leader_A": "Opinion Leader (Pro-position)",
        "Leader_B": "Opinion Leader (Anti-position)",
        "Follower_1": "Follower of Leader_A",
        "Follower_2": "Follower of Leader_A",
        "Follower_3": "Follower of Leader_B",
        "Neutral_1": "Undecided/Neutral",
        "Neutral_2": "Undecided/Neutral",
        "Bridge": "Bridge person (connects groups)",
    }
    
    for node, role in roles.items():
        G.nodes[node]["role"] = role
    
    # Inject required defaults for graph parameters
    inject_defaults(G)
    
    # Measure initial state
    C_initial, dnfr_initial, _ = compute_coherence(G, return_means=True)
    Si_initial = compute_Si(G)
    
    print(f"Initial network state:")
    print(f"  C(t) = {C_initial:.3f} (social coherence - expect low due to diversity)")
    print(f"  Mean ΔNFR = {dnfr_initial:.3f} (opinion change pressure)")
    print(f"  Network size: {G.number_of_nodes()} individuals")
    print(f"  Connections: {G.number_of_edges()} relationships")
    print()
    
    # 2. TNFR MODELING: Define behavioral protocols
    # ----------------------------------------------
    
    print("Phase 2: Defining opinion dynamics protocols...")
    print()
    
    # Opinion Leaders: Emit strongly, resist change, build consensus
    leader_protocol = [
        Emission(),      # Share strong opinions
        Reception(),     # Listen to feedback (but less affected)
        Coherence(),     # Reinforce own position
        Resonance(),     # Attract like-minded individuals
        Coupling(),      # Form strong influence networks
        Coherence(),     # Stabilize position
        Silence(),
    ]
    
    # Followers: Receptive, adopt aligned opinions
    follower_protocol = [
        Emission(),      # Share opinions (must start with emission)
        Reception(),     # Listen to leaders and peers
        Coherence(),     # Try to stabilize
        Resonance(),     # Align with dominant narrative
        Silence(),
    ]
    
    # Neutral/Undecided: Highly receptive, may experience conflict
    neutral_protocol = [
        Emission(),      # Must start with emission
        Reception(),     # Exposed to multiple viewpoints
        Coherence(),     # Attempt stabilization
        Silence(),
    ]
    
    # Bridge: Balance opposing views, experience tension
    bridge_protocol = [
        Emission(),      # Try to mediate (must start with emission)
        Reception(),     # Listen to multiple sides
        Coherence(),     # Try to maintain balance
        Resonance(),     # Facilitate dialogue
        Silence(),
    ]
    
    # 3. OPERATOR APPLICATION: Execute social dynamics
    # -------------------------------------------------
    
    print("Simulating opinion formation dynamics...")
    print()
    
    # Apply protocols
    run_sequence(G, "Leader_A", leader_protocol)
    run_sequence(G, "Leader_B", leader_protocol)
    
    run_sequence(G, "Follower_1", follower_protocol)
    run_sequence(G, "Follower_2", follower_protocol)
    run_sequence(G, "Follower_3", follower_protocol)
    
    run_sequence(G, "Neutral_1", neutral_protocol)
    run_sequence(G, "Neutral_2", neutral_protocol)
    
    run_sequence(G, "Bridge", bridge_protocol)
    
    # 4. SIMULATION: Run opinion dynamics over time
    # ----------------------------------------------
    
    print("Phase 3: Running social dynamics simulation...")
    print("(Simulating information flow and opinion evolution)")
    print()
    
    register_metrics_callbacks(G)
    register_trace(G)
    
    # Run for 12 time steps = ~12 information exchange cycles
    run(G, steps=12, dt=0.1)
    
    # 5. RESULTS INTERPRETATION
    # --------------------------
    
    print("=" * 70)
    print("RESULTS: Social Network Analysis")
    print("=" * 70)
    print()
    
    # Compute final metrics
    C_final, dnfr_final, depi_final = compute_coherence(G, return_means=True)
    Si_final = compute_Si(G)
    
    print("Network-Level Metrics:")
    print(f"  Initial C(t) = {C_initial:.3f}")
    print(f"  Final C(t) = {C_final:.3f}")
    print(f"  ΔC = {C_final - C_initial:+.3f} (consensus change)")
    print(f"  Mean ΔNFR = {dnfr_final:.3f} (residual opinion pressure)")
    print()
    
    print("Individual Opinion Stability (Sense Index):")
    if isinstance(Si_final, dict):
        for person in sorted(Si_final.keys()):
            si_val = Si_final[person]
            si_change = si_val - Si_initial.get(person, 0.0) if isinstance(Si_initial, dict) else 0.0
            role_short = roles[person].split()[0]
            print(f"  {person:15s} Si = {si_val:.3f} ({si_change:+.3f})  [{role_short}]")
    else:
        for idx, person in enumerate(sorted(G.nodes())):
            si_val = float(Si_final[idx]) if hasattr(Si_final, '__getitem__') else 0.0
            si_change = 0.0
            role_short = roles[person].split()[0]
            print(f"  {person:15s} Si = {si_val:.3f} ({si_change:+.3f})  [{role_short}]")
    print()
    
    # Social science interpretation
    print("=" * 70)
    print("SOCIAL SCIENCE INTERPRETATION")
    print("=" * 70)
    print()
    
    # Analyze consensus vs polarization
    if C_final > 0.65:
        outcome = "STRONG CONSENSUS"
        explanation = "Network converged to shared understanding"
    elif C_final > 0.45:
        outcome = "MODERATE CONSENSUS"
        explanation = "Some agreement emerged, but diversity remains"
    elif C_final > C_initial:
        outcome = "WEAK CONSENSUS"
        explanation = "Slight movement toward alignment"
    else:
        outcome = "POLARIZATION"
        explanation = "Network became more fragmented"
    
    print(f"1. Network Outcome: {outcome}")
    print(f"   {explanation}")
    print(f"   C(t): {C_initial:.3f} → {C_final:.3f}")
    print()
    
    # Identify opinion leaders (high Si + high EPI)
    if isinstance(Si_final, dict):
        avg_si = sum(Si_final.values()) / len(Si_final)
        leaders = [p for p, si in Si_final.items() if si > avg_si * 1.2]
    else:
        avg_si = float(Si_final.mean()) if hasattr(Si_final, 'mean') else 0.0
        leaders = []
    
    print(f"2. Opinion Leaders (high stability):")
    for leader in leaders:
        print(f"   • {leader}: {roles[leader]}")
    print()
    
    # Analyze information flow
    if abs(dnfr_final) < 0.3:
        stability = "STABLE - Opinions have settled"
    else:
        stability = "DYNAMIC - Opinions still evolving"
    
    print(f"3. Opinion Stability: {stability}")
    print(f"   Mean ΔNFR = {dnfr_final:.3f}")
    print()
    
    print("=" * 70)
    print("Key TNFR Insights:")
    print("=" * 70)
    print("• Individuals = NFR nodes with opinion structure (EPI)")
    print("• Opinion sharing = Emission operator propagates beliefs")
    print("• Influence = Reception operator integrates others' opinions")
    print("• Consensus = Resonance operator aligns opinion structures")
    print("• Debate = Dissonance operator creates cognitive tension")
    print("• Social coherence C(t) = Measure of consensus vs. polarization")
    print("• Stability Si = Resistance to opinion change (opinion leaders)")
    print()
    print("Social Dynamics Principles:")
    print("• High C(t) + High Si → Stable consensus (echo chamber risk)")
    print("• Low C(t) + High ΔNFR → Active debate (healthy discourse)")
    print("• Low C(t) + Low Si → Fragmentation (polarization risk)")
    print("=" * 70)


if __name__ == "__main__":
    run_example()
