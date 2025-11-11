#!/usr/bin/env python3
"""
Title: Urban Traffic Flow - Dynamic Traffic Signal Optimization

Problem: Urban traffic systems face congestion, especially during rush hours
and incidents. Static traffic light timing is inefficient. How can traffic
signals adapt dynamically to real-time conditions?

TNFR Approach: Model intersections as NFR nodes where:
- EPI represents traffic state (flow capacity, queue lengths)
- νf (Hz_str) is adaptation speed (how fast signals can adjust)
- Phase represents traffic light timing coordination
- Transition operator models traffic light state changes
- Dissonance represents congestion/bottlenecks
- Coherence measures traffic flow smoothness

Key Operators:
- Transition (NAV): Traffic light phase transitions
- Reception (EN): Monitor traffic sensors (volume, speed)
- Emission (AL): Release traffic (green light)
- Coupling (UM): Coordinate adjacent intersections
- Dissonance (OZ): Congestion, traffic jams
- Coherence (IL): Smooth traffic flow
- Mutation (ZHIR): Adapt timing patterns to conditions

Relevant Metrics:
- C(t): Traffic coherence (flow smoothness)
- Si: Intersection stability (consistent throughput)
- Phase coherence: Signal coordination (green wave)
- ΔNFR: Congestion pressure

Expected Behavior:
- Initial uncoordinated signals (low C(t))
- Traffic sensors provide feedback (Reception)
- Signals adapt timing (Transition, Mutation)
- Coordination emerges (phase alignment)
- Congestion decreases (Dissonance → Coherence)
- Final state shows smooth, coordinated flow

Run:
    python docs/source/examples/urban_traffic_flow.py
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
    Resonance,
    Silence,
)
from tnfr.trace import register_trace
from tnfr.constants import inject_defaults


def run_example() -> None:
    """Model adaptive traffic signal control using TNFR operators."""
    
    print("=" * 70)
    print("TNFR Urban Traffic Flow: Dynamic Signal Optimization")
    print("=" * 70)
    print()
    
    # 1. PROBLEM SETUP: Creating urban traffic network
    # -------------------------------------------------
    # Scenario: 3x3 grid of intersections (9 total)
    # Challenge: Morning rush hour - heavy traffic from residential to downtown
    
    print("Phase 1: Initializing urban traffic network...")
    print("Creating 9-intersection grid with traffic sensors")
    print()
    
    # Residential area intersections (light initial load)
    G, _ = create_nfr(
        "Intersection_A1",  # Northwest residential
        epi=0.25,    # Moderate traffic capacity utilization
        vf=1.0,      # Standard adaptation rate
        theta=0.0    # Traffic signal phase
    )
    
    create_nfr(
        "Intersection_A2",
        epi=0.28,
        vf=1.05,
        theta=0.5,   # Out of sync initially
        graph=G
    )
    
    create_nfr(
        "Intersection_A3",
        epi=0.22,
        vf=0.95,
        theta=-0.3,
        graph=G
    )
    
    # Mid-city intersections (moderate load)
    create_nfr(
        "Intersection_B1",
        epi=0.35,    # Higher utilization
        vf=1.1,
        theta=0.8,
        graph=G
    )
    
    create_nfr(
        "Intersection_B2",  # Central hub
        epi=0.40,    # High traffic
        vf=0.85,     # Slower adaptation (complex)
        theta=1.2,
        graph=G
    )
    
    create_nfr(
        "Intersection_B3",
        epi=0.38,
        vf=1.0,
        theta=-0.6,
        graph=G
    )
    
    # Downtown intersections (heavy load during rush hour)
    create_nfr(
        "Intersection_C1",
        epi=0.48,    # High congestion
        vf=0.90,
        theta=1.5,
        graph=G
    )
    
    create_nfr(
        "Intersection_C2",
        epi=0.52,    # Highest congestion
        vf=0.80,     # Slowest adaptation (very complex)
        theta=-1.0,
        graph=G
    )
    
    create_nfr(
        "Intersection_C3",
        epi=0.45,
        vf=0.95,
        theta=0.4,
        graph=G
    )
    
    # Establish traffic network topology (grid connections)
    # Horizontal connections
    G.add_edge("Intersection_A1", "Intersection_A2")
    G.add_edge("Intersection_A2", "Intersection_A3")
    G.add_edge("Intersection_B1", "Intersection_B2")
    G.add_edge("Intersection_B2", "Intersection_B3")
    G.add_edge("Intersection_C1", "Intersection_C2")
    G.add_edge("Intersection_C2", "Intersection_C3")
    
    # Vertical connections
    G.add_edge("Intersection_A1", "Intersection_B1")
    G.add_edge("Intersection_B1", "Intersection_C1")
    G.add_edge("Intersection_A2", "Intersection_B2")
    G.add_edge("Intersection_B2", "Intersection_C2")
    G.add_edge("Intersection_A3", "Intersection_B3")
    G.add_edge("Intersection_B3", "Intersection_C3")
    
    # Store intersection metadata
    locations = {
        "Intersection_A1": "Residential NW - Low traffic",
        "Intersection_A2": "Residential N - Low traffic",
        "Intersection_A3": "Residential NE - Low traffic",
        "Intersection_B1": "Mid-city W - Moderate traffic",
        "Intersection_B2": "Mid-city Center - High traffic",
        "Intersection_B3": "Mid-city E - Moderate traffic",
        "Intersection_C1": "Downtown SW - Heavy traffic",
        "Intersection_C2": "Downtown Center - Heaviest traffic",
        "Intersection_C3": "Downtown SE - Heavy traffic",
    }
    
    for node, location in locations.items():
        G.nodes[node]["location"] = location
    
    # Inject required defaults for graph parameters
    inject_defaults(G)
    
    # Measure initial state (before optimization)
    C_initial, dnfr_initial, _ = compute_coherence(G, return_means=True)
    Si_initial = compute_Si(G)
    
    print("Initial traffic state (BEFORE optimization):")
    print(f"  C(t) = {C_initial:.3f} (traffic flow coherence)")
    print(f"  Mean ΔNFR = {dnfr_initial:.3f} (congestion pressure)")
    print(f"  Grid: 3x3 intersections, {G.number_of_edges()} connections")
    print()
    
    # 2. TNFR MODELING: Traffic control strategies
    # ---------------------------------------------
    
    print("Phase 2: Deploying adaptive traffic control strategies...")
    print()
    
    # Residential area: Simple timing adjustments
    residential_control = [
        Emission(),      # Release traffic (green light) - must start with emission
        Reception(),     # Monitor traffic sensors
        Coherence(),     # Maintain smooth flow
        Silence(),
    ]
    
    # Mid-city: Adaptive coordination with neighbors
    midcity_control = [
        Emission(),      # Release traffic (must start with emission)
        Reception(),     # Monitor local traffic
        Resonance(),     # Coordinate with neighbors
        Coupling(),      # Synchronize phases
        Coherence(),     # Stabilize flow
        Silence(),
    ]
    
    # Downtown: Aggressive optimization, handle congestion
    downtown_control = [
        Emission(),      # Maximize throughput (must start with emission)
        Reception(),     # Monitor heavy traffic
        Coherence(),     # Stabilize before applying dissonance
        Dissonance(),    # Acknowledge congestion
        Mutation(),      # Try alternative timing patterns
        Coherence(),     # Stabilize after mutation
        Resonance(),     # Propagate successful patterns
        Coupling(),      # Coordinate with neighbors
        Coherence(),     # Stabilize under load
        Silence(),
    ]
    
    # Central hub: Complex coordination
    hub_control = [
        Emission(),      # Release traffic (must start with emission)
        Reception(),     # Monitor all directions
        Coherence(),     # Stabilize before conflicts
        Dissonance(),    # Handle conflicts
        Mutation(),      # Explore timing options
        Coherence(),     # Stabilize after mutation
        Resonance(),     # Propagate optimization
        Coupling(),      # Strong neighbor coordination
        Coherence(),     # Stabilize complex flows
        Silence(),
    ]
    
    # 3. OPERATOR APPLICATION: Deploy traffic control
    # ------------------------------------------------
    
    print("Applying control strategies by zone...")
    print()
    
    # Residential zone
    for intersection in ["Intersection_A1", "Intersection_A2", "Intersection_A3"]:
        run_sequence(G, intersection, residential_control)
    print("✓ Residential zone optimized")
    
    # Mid-city zone (except hub)
    for intersection in ["Intersection_B1", "Intersection_B3"]:
        run_sequence(G, intersection, midcity_control)
    print("✓ Mid-city zone optimized")
    
    # Central hub (special handling)
    run_sequence(G, "Intersection_B2", hub_control)
    print("✓ Central hub optimized")
    
    # Downtown zone
    for intersection in ["Intersection_C1", "Intersection_C2", "Intersection_C3"]:
        run_sequence(G, intersection, downtown_control)
    print("✓ Downtown zone optimized")
    print()
    
    # 4. SIMULATION: Run traffic dynamics
    # ------------------------------------
    
    print("Phase 3: Simulating rush hour traffic dynamics...")
    print("(Multiple signal cycles with adaptive adjustments)")
    print()
    
    register_metrics_callbacks(G)
    register_trace(G)
    
    # Run simulation: 12 time steps = ~12 signal cycles
    run(G, steps=12, dt=0.1)
    
    # 5. RESULTS INTERPRETATION
    # --------------------------
    
    print("=" * 70)
    print("RESULTS: Traffic Optimization Analysis")
    print("=" * 70)
    print()
    
    # Compute final metrics (after optimization)
    C_final, dnfr_final, depi_final = compute_coherence(G, return_means=True)
    Si_final = compute_Si(G)
    
    print("Network-Level Traffic Metrics:")
    print(f"  Initial C(t) = {C_initial:.3f}")
    print(f"  Final C(t) = {C_final:.3f}")
    print(f"  Improvement: ΔC = {C_final - C_initial:+.3f}")
    print(f"  Congestion pressure: ΔNFR = {dnfr_initial:.3f} → {dnfr_final:.3f}")
    print()
    
    print("Per-Intersection Performance:")
    print(f"{'Intersection':<20} {'Si (Stability)':<18} {'Location'}")
    print("-" * 70)
    if isinstance(Si_final, dict):
        for intersection in sorted(Si_final.keys()):
            si_val = Si_final[intersection]
            si_initial_val = Si_initial.get(intersection, 0.0) if isinstance(Si_initial, dict) else 0.0
            change = si_val - si_initial_val
            loc_short = locations[intersection].split('-')[0].strip()
            print(f"{intersection:<20} {si_val:.3f} ({change:+.3f})         {loc_short}")
    else:
        for idx, intersection in enumerate(sorted(G.nodes())):
            si_val = float(Si_final[idx]) if hasattr(Si_final, '__getitem__') else 0.0
            change = 0.0
            loc_short = locations[intersection].split('-')[0].strip()
            print(f"{intersection:<20} {si_val:.3f} ({change:+.3f})         {loc_short}")
    print()
    
    # Urban planning interpretation
    print("=" * 70)
    print("URBAN PLANNING INTERPRETATION")
    print("=" * 70)
    print()
    
    # Traffic flow assessment
    if C_final > 0.65:
        flow_status = "EXCELLENT FLOW"
        description = "Signals well-coordinated, minimal congestion"
    elif C_final > 0.45:
        flow_status = "GOOD FLOW"
        description = "Most signals coordinated, some congestion"
    elif C_final > C_initial:
        flow_status = "IMPROVED FLOW"
        description = "Better than initial, but optimization ongoing"
    else:
        flow_status = "POOR FLOW"
        description = "Optimization ineffective, congestion persists"
    
    print(f"1. Traffic Flow Status: {flow_status}")
    print(f"   {description}")
    print(f"   C(t) improvement: {(C_final - C_initial) / C_initial * 100:+.1f}%")
    print()
    
    # Identify problem intersections
    if isinstance(Si_final, dict):
        avg_si = sum(Si_final.values()) / len(Si_final)
        problem_intersections = [i for i, si in Si_final.items() if si < avg_si * 0.8]
    else:
        avg_si = float(Si_final.mean()) if hasattr(Si_final, 'mean') else 0.0
        problem_intersections = []
    
    if problem_intersections:
        print(f"2. Problem Intersections (Si < {avg_si * 0.8:.3f}):")
        for intersection in problem_intersections:
            print(f"   • {intersection}: {locations[intersection]}")
        print("   → Recommendation: Increase green time or add turn lanes")
    else:
        print("2. ✓ No critical problem intersections identified")
    print()
    
    # Coordination effectiveness
    downtown_intersections = ["Intersection_C1", "Intersection_C2", "Intersection_C3"]
    if isinstance(Si_final, dict):
        downtown_avg_si = sum(Si_final.get(i, 0.0) for i in downtown_intersections) / len(downtown_intersections)
    else:
        downtown_avg_si = avg_si
    
    if downtown_avg_si > avg_si:
        coord_status = "Downtown coordination SUCCESSFUL"
    else:
        coord_status = "Downtown needs further optimization"
    
    print(f"3. Downtown Corridor: {coord_status}")
    print(f"   Average Si = {downtown_avg_si:.3f} (network avg = {avg_si:.3f})")
    print()
    
    # Congestion reduction
    if abs(dnfr_initial) > 0.001:
        congestion_reduction = (dnfr_initial - dnfr_final) / abs(dnfr_initial) * 100
        print(f"4. Congestion Reduction: {congestion_reduction:.1f}%")
        if congestion_reduction > 30:
            print("   ✓ Significant improvement in traffic flow")
        elif congestion_reduction > 10:
            print("   ~ Moderate improvement, continue optimization")
        else:
            print("   ⚠ Limited improvement, consider infrastructure changes")
    else:
        print(f"4. Congestion Status: Initial ΔNFR was near zero (baseline: {dnfr_initial:.3f}, final: {dnfr_final:.3f})")
        if abs(dnfr_final) < abs(dnfr_initial):
            print("   ✓ System maintained low congestion levels")
        else:
            print("   ~ Congestion increased slightly")
    print()
    
    print("=" * 70)
    print("Key TNFR Insights:")
    print("=" * 70)
    print("• Intersections = NFR nodes with traffic state (EPI)")
    print("• Traffic signals = Emission operator (release vehicles)")
    print("• Sensor monitoring = Reception operator (gather data)")
    print("• Signal timing = Transition operator (phase changes)")
    print("• Coordination = Coupling + phase alignment (green wave)")
    print("• Congestion = Dissonance operator (flow breakdown)")
    print("• Adaptive timing = Mutation operator (pattern exploration)")
    print("• Flow coherence C(t) = Smoothness of traffic movement")
    print()
    print("Traffic Engineering Principles:")
    print("  • High C(t) = coordinated signals, smooth flow")
    print("  • Low ΔNFR = minimal congestion pressure")
    print("  • Phase alignment = green wave progression")
    print("  • High Si = stable throughput (good signal timing)")
    print()
    print("Optimization Strategies:")
    print("  ✓ Adaptive signals (high νf) respond faster to conditions")
    print("  ✓ Coupling enables coordination (synchronized corridors)")
    print("  ✓ Mutation explores timing alternatives under congestion")
    print("  ✓ Monitor C(t) in real-time to detect emerging congestion")
    print("  ✓ Low Si intersections need infrastructure investment")
    print()
    print("Real-World Applications:")
    print("  • Predict congestion before it occurs (falling C(t))")
    print("  • Identify which intersections to upgrade (low Si)")
    print("  • Optimize signal timing without simulation (ΔNFR-driven)")
    print("  • Coordinate emergency vehicle routing (phase manipulation)")
    print("=" * 70)


if __name__ == "__main__":
    run_example()
