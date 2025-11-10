#!/usr/bin/env python3
"""
Title: Supply Chain Resilience - Adaptive Response to Disruptions

Problem: Supply chains face disruptions (factory closures, shipping delays,
demand spikes). How do resilient supply chains adapt? What makes some
systems fragile and others robust?

TNFR Approach: Model supply chain entities as NFR nodes where:
- EPI represents operational capability/inventory structure
- νf (Hz_str) is adaptation speed (how fast can operations change)
- Phase represents synchronization with supply/demand cycles
- Mutation operator models adaptation to disruptions
- Coupling represents supply relationships
- Coherence measures supply chain stability

Key Operators:
- Coupling (UM): Supply relationships, logistics connections
- Mutation (ZHIR): Adaptation to disruptions (find alternatives)
- Reception (EN): Monitor supply/demand signals
- Emission (AL): Fulfill orders, ship products
- SelfOrganization (THOL): Find alternative supply routes
- Coherence (IL): Stabilize operations after disruption

Relevant Metrics:
- C(t): Supply chain coherence (operational stability)
- Si: Resilience (ability to maintain operations under stress)
- ΔNFR: Operational pressure (supply-demand mismatch)
- Phase coherence: Timing coordination across chain

Expected Behavior:
- Initial stable supply chain (high C(t))
- Disruption causes coherence drop (shock)
- Mutation operator enables adaptation
- Self-organization finds alternative routes
- Recovery: coherence increases as new equilibrium forms
- High Si nodes = resilient entities (bounce back faster)

Run:
    python docs/source/examples/supply_chain_resilience.py
"""

from tnfr import create_nfr, run_sequence
from tnfr.dynamics import run
from tnfr.metrics import register_metrics_callbacks
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.sense_index import compute_Si
from tnfr.structural import (
    Coherence,
    Coupling,
    Dissonance,
    Emission,
    Mutation,
    Reception,
    SelfOrganization,
    Silence,
    Transition,
)
from tnfr.trace import register_trace
from tnfr.constants import inject_defaults


def run_example() -> None:
    """Model supply chain adaptation to disruptions using TNFR operators."""
    
    print("=" * 70)
    print("TNFR Supply Chain Resilience: Adaptive Response to Disruptions")
    print("=" * 70)
    print()
    
    # 1. PROBLEM SETUP: Creating a supply chain network
    # --------------------------------------------------
    # Scenario: Electronics supply chain
    # Tiers: Raw Materials → Component Suppliers → Manufacturer → Distributor
    
    print("Phase 1: Initializing supply chain network...")
    print()
    
    # Raw Material Suppliers: Low-level inputs
    G, _ = create_nfr(
        "RawMaterial_Asia",
        epi=0.40,    # Established operations
        vf=0.85,     # Moderate adaptation speed
        theta=0.0    # Synchronized with production cycles
    )
    
    create_nfr(
        "RawMaterial_Europe",
        epi=0.38,
        vf=0.90,
        theta=0.2,
        graph=G
    )
    
    # Component Suppliers: Mid-tier manufacturing
    create_nfr(
        "Components_Taiwan",
        epi=0.45,    # Critical supplier (semiconductors)
        vf=0.70,     # Lower adaptation (specialized)
        theta=-0.1,
        graph=G
    )
    
    create_nfr(
        "Components_Mexico",
        epi=0.35,    # Alternative supplier
        vf=1.0,      # More flexible
        theta=0.3,
        graph=G
    )
    
    # Final Manufacturer: Assembly and production
    create_nfr(
        "Manufacturer_USA",
        epi=0.50,    # Complex operations
        vf=0.80,
        theta=0.0,
        graph=G
    )
    
    # Distribution Centers: Last-mile logistics
    create_nfr(
        "Distribution_East",
        epi=0.42,
        vf=1.1,      # Flexible logistics
        theta=-0.2,
        graph=G
    )
    
    create_nfr(
        "Distribution_West",
        epi=0.40,
        vf=1.15,
        theta=0.25,
        graph=G
    )
    
    # Backup/Alternative supplier (initially disconnected)
    create_nfr(
        "Backup_India",
        epi=0.30,    # Smaller capacity
        vf=1.2,      # High flexibility
        theta=0.8,   # Not yet synchronized
        graph=G
    )
    
    # Establish supply chain connections
    # Raw materials → Components
    G.add_edge("RawMaterial_Asia", "Components_Taiwan")
    G.add_edge("RawMaterial_Europe", "Components_Taiwan")
    G.add_edge("RawMaterial_Europe", "Components_Mexico")
    
    # Components → Manufacturer
    G.add_edge("Components_Taiwan", "Manufacturer_USA")
    G.add_edge("Components_Mexico", "Manufacturer_USA")
    
    # Manufacturer → Distribution
    G.add_edge("Manufacturer_USA", "Distribution_East")
    G.add_edge("Manufacturer_USA", "Distribution_West")
    
    # Note: Backup_India initially disconnected (activated during disruption)
    
    # Store supply chain metadata
    entities = {
        "RawMaterial_Asia": "Raw materials (metals, minerals) - Asia",
        "RawMaterial_Europe": "Raw materials (metals, minerals) - Europe",
        "Components_Taiwan": "Semiconductors & electronics - Taiwan",
        "Components_Mexico": "Electronics assembly - Mexico",
        "Manufacturer_USA": "Final product assembly - USA",
        "Distribution_East": "Distribution center - East Coast",
        "Distribution_West": "Distribution center - West Coast",
        "Backup_India": "Backup component supplier - India",
    }
    
    for node, description in entities.items():
        G.nodes[node]["description"] = description
    
    # Inject required defaults for graph parameters
    inject_defaults(G)
    
    # Measure initial state (before disruption)
    C_initial, dnfr_initial, _ = compute_coherence(G, return_means=True)
    Si_initial = compute_Si(G)
    
    print("Initial supply chain state (BEFORE disruption):")
    print(f"  C(t) = {C_initial:.3f} (operational stability)")
    print(f"  Mean ΔNFR = {dnfr_initial:.3f} (supply-demand balance)")
    print(f"  Network: {G.number_of_nodes()} entities, {G.number_of_edges()} connections")
    print()
    
    # 2. TNFR MODELING: Normal operations
    # ------------------------------------
    
    print("Phase 2: Establishing normal operations...")
    print()
    
    # Normal operation protocols (before disruption)
    normal_ops = [
        Emission(),      # Fulfill orders (must start with emission)
        Reception(),     # Monitor supply/demand
        Coupling(),      # Maintain relationships
        Coherence(),     # Stable operations
        Silence(),
    ]
    
    # Run normal operations for all entities
    for node in G.nodes():
        if node != "Backup_India":  # Backup not yet active
            run_sequence(G, node, normal_ops)
    
    print("Normal operations established. Measuring baseline...")
    register_metrics_callbacks(G)
    register_trace(G)
    
    # Run baseline period
    run(G, steps=5, dt=0.1)
    
    C_baseline, dnfr_baseline, _ = compute_coherence(G, return_means=True)
    print(f"Baseline C(t) = {C_baseline:.3f}")
    print()
    
    # 3. DISRUPTION EVENT: Taiwan supplier crisis
    # --------------------------------------------
    
    print("=" * 70)
    print("DISRUPTION EVENT: Components_Taiwan experiences major outage!")
    print("(Simulating earthquake, factory fire, or geopolitical crisis)")
    print("=" * 70)
    print()
    
    # Simulate disruption: Dissonance on critical supplier
    disruption_impact = [
        Emission(),      # Must start with emission
        Reception(),     # Sense the shock
        Coherence(),     # Try to stabilize
        Dissonance(),    # Major operational shock
        Mutation(),      # Forced to adapt
        Coherence(),     # Stabilize after mutation
        Silence(),
    ]
    
    run_sequence(G, "Components_Taiwan", disruption_impact)
    
    # Measure disruption impact
    C_disrupted, dnfr_disrupted, _ = compute_coherence(G, return_means=True)
    print(f"Post-disruption C(t) = {C_disrupted:.3f} (ΔC = {C_disrupted - C_baseline:.3f})")
    print()
    
    # 4. ADAPTATION RESPONSE: Supply chain reorganization
    # ----------------------------------------------------
    
    print("Phase 3: Activating adaptation protocols...")
    print()
    
    # Manufacturer must adapt to lost supplier
    manufacturer_adaptation = [
        Emission(),          # Must start with emission
        Reception(),         # Assess situation
        Coherence(),         # Stabilize before crisis response
        Dissonance(),        # Acknowledge supply crisis
        Mutation(),          # Explore alternatives
        SelfOrganization(),  # Restructure supply chain
        Silence(),           # Close self-organization
        Coupling(),          # Connect to backup supplier
        Coherence(),         # Stabilize new operations
        Silence(),
    ]
    
    # Activate backup supplier
    backup_activation = [
        Emission(),          # Begin production (must start with emission)
        Reception(),         # Understand requirements
        Coherence(),         # Stabilize
        Dissonance(),        # Acknowledge new challenge
        Mutation(),          # Adapt to new customer
        Coupling(),          # Establish connections
        Coherence(),         # Stabilize operations
        Silence(),
    ]
    
    # Alternative existing supplier scales up
    alternative_scaleup = [
        Emission(),          # Increase output (must start with emission)
        Reception(),         # See increased demand
        Coherence(),         # Stabilize
        Dissonance(),        # Acknowledge capacity challenge
        Mutation(),          # Adapt capacity
        SelfOrganization(),  # Optimize operations
        Silence(),           # Close self-organization
        Coherence(),         # Stabilize higher throughput
        Silence(),
    ]
    
    print("Manufacturer adapting to disruption...")
    run_sequence(G, "Manufacturer_USA", manufacturer_adaptation)
    
    print("Activating backup supplier (Backup_India)...")
    run_sequence(G, "Backup_India", backup_activation)
    
    # Establish new supply connection
    G.add_edge("RawMaterial_Asia", "Backup_India")
    G.add_edge("Backup_India", "Manufacturer_USA")
    print("New supply route established: RawMaterial_Asia → Backup_India → Manufacturer")
    print()
    
    print("Alternative supplier (Components_Mexico) scaling up...")
    run_sequence(G, "Components_Mexico", alternative_scaleup)
    print()
    
    # 5. RECOVERY SIMULATION
    # -----------------------
    
    print("Phase 4: Simulating recovery dynamics...")
    print()
    
    # Run recovery period
    run(G, steps=10, dt=0.1)
    
    # 6. RESULTS INTERPRETATION
    # --------------------------
    
    print("=" * 70)
    print("RESULTS: Supply Chain Resilience Analysis")
    print("=" * 70)
    print()
    
    # Compute final metrics (after recovery)
    C_final, dnfr_final, depi_final = compute_coherence(G, return_means=True)
    Si_final = compute_Si(G)
    
    print("Supply Chain Metrics Timeline:")
    print(f"  1. Baseline:      C(t) = {C_baseline:.3f}")
    print(f"  2. Disruption:    C(t) = {C_disrupted:.3f} ({C_disrupted - C_baseline:+.3f})")
    print(f"  3. Post-recovery: C(t) = {C_final:.3f} ({C_final - C_disrupted:+.3f})")
    print()
    print(f"  Net impact: ΔC = {C_final - C_baseline:+.3f}")
    print(f"  Recovery ratio: {(C_final - C_disrupted) / (C_baseline - C_disrupted) * 100:.1f}%")
    print()
    
    print("Entity Resilience (Sense Index):")
    if isinstance(Si_final, dict):
        for entity in sorted(Si_final.keys()):
            si_val = Si_final[entity]
            si_initial_val = Si_initial.get(entity, 0.0) if isinstance(Si_initial, dict) else 0.0
            change = si_val - si_initial_val
            desc_short = entities[entity].split('-')[0].strip()
            print(f"  {entity:25s} Si = {si_val:.3f} ({change:+.3f})  [{desc_short}]")
    else:
        for idx, entity in enumerate(sorted(G.nodes())):
            si_val = float(Si_final[idx]) if hasattr(Si_final, '__getitem__') else 0.0
            change = 0.0
            desc_short = entities[entity].split('-')[0].strip()
            print(f"  {entity:25s} Si = {si_val:.3f} ({change:+.3f})  [{desc_short}]")
    print()
    
    # Business interpretation
    print("=" * 70)
    print("BUSINESS INTERPRETATION")
    print("=" * 70)
    print()
    
    # Recovery assessment
    if C_final >= C_baseline * 0.95:
        recovery = "FULL RECOVERY"
        status = "Supply chain restored to normal operations"
    elif C_final >= C_disrupted * 1.5:
        recovery = "STRONG RECOVERY"
        status = "Supply chain adapted successfully"
    elif C_final > C_disrupted:
        recovery = "PARTIAL RECOVERY"
        status = "Some adaptation, but still impaired"
    else:
        recovery = "RECOVERY FAILURE"
        status = "Unable to adapt to disruption"
    
    print(f"1. Recovery Outcome: {recovery}")
    print(f"   {status}")
    print()
    
    # Identify resilient entities
    if isinstance(Si_final, dict):
        avg_si = sum(Si_final.values()) / len(Si_final)
        resilient = [e for e, si in Si_final.items() if si > avg_si * 1.1]
    else:
        avg_si = float(Si_final.mean()) if hasattr(Si_final, 'mean') else 0.0
        resilient = []
    
    print(f"2. Most Resilient Entities (Si > {avg_si * 1.1:.3f}):")
    for entity in resilient:
        print(f"   • {entity}: {entities[entity]}")
    print()
    
    # Adaptation effectiveness
    if "Backup_India" in G.nodes():
        if isinstance(Si_final, dict):
            backup_si = Si_final.get("Backup_India", 0.0)
        else:
            backup_si = 0.0
        if backup_si > 0.5:
            adaptation_status = "SUCCESSFUL - Backup supplier integrated"
        else:
            adaptation_status = "PARTIAL - Backup integration incomplete"
    else:
        adaptation_status = "NO BACKUP ACTIVATION"
    
    print(f"3. Adaptation Strategy: {adaptation_status}")
    print()
    
    print("=" * 70)
    print("Key TNFR Insights:")
    print("=" * 70)
    print("• Supply chain entities = NFR nodes with operational EPI")
    print("• Disruption = Dissonance operator (creates instability)")
    print("• Adaptation = Mutation operator (explore alternatives)")
    print("• Resilience = High Si (ability to maintain coherence under stress)")
    print("• Recovery = Self-organization (find new equilibrium)")
    print("• Network coherence C(t) = Overall supply chain stability")
    print()
    print("Resilience Principles:")
    print("  • Redundancy: Multiple suppliers → higher recovery capacity")
    print("  • Flexibility: High νf → faster adaptation to shocks")
    print("  • Coupling: Strong relationships → coordinated response")
    print("  • Si metric: Predicts which entities will fail under stress")
    print()
    print("Business Implications:")
    print("  ✓ Monitor Si to identify vulnerable supply chain nodes")
    print("  ✓ Maintain backup suppliers (even if disconnected initially)")
    print("  ✓ High νf entities recover faster (invest in flexibility)")
    print("  ✓ Track C(t) as real-time supply chain health indicator")
    print("=" * 70)


if __name__ == "__main__":
    run_example()
