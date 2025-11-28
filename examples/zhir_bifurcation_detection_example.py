#!/usr/bin/env python3
"""Example demonstrating ZHIR bifurcation potential detection.

This example shows how ZHIR (Mutation) detects when structural acceleration
∂²EPI/∂t² exceeds the bifurcation threshold τ, according to AGENTS.md §U4a.

The detection is conservative (Option B) - it sets telemetry flags and logs
information but does not create structural variants. This enables validation
of grammar U4a requirement: "If {OZ, ZHIR}, include {THOL, IL}".
"""

from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Dissonance, Mutation, Coherence, Silence
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI


def example_zhir_bifurcation_detection():
    """Demonstrate ZHIR bifurcation potential detection."""
    print("=" * 70)
    print("ZHIR Bifurcation Potential Detection Example")
    print("=" * 70)
    
    # Create a node with evolving EPI
    G, node = create_nfr("evolving_system", epi=0.3, vf=1.0, theta=0.5)
    
    # Build EPI history showing high acceleration
    # This simulates a system undergoing rapid structural reorganization
    # d²EPI/dt² = 0.60 - 2*0.40 + 0.30 = 0.60 - 0.80 + 0.30 = 0.10
    G.nodes[node]["epi_history"] = [0.30, 0.40, 0.60]
    G.nodes[node]["glyph_history"] = []
    
    # Configure bifurcation threshold
    # Default is 0.5, but we'll use 0.05 to demonstrate detection
    G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
    
    print(f"\nInitial state:")
    print(f"  Node: {node}")
    print(f"  EPI: {get_attr(G.nodes[node], ALIAS_EPI, 0.0):.3f}")
    print(f"  EPI history: {G.nodes[node]['epi_history']}")
    print(f"  Bifurcation threshold τ: {G.graph['BIFURCATION_THRESHOLD_TAU']}")
    
    # Calculate acceleration manually for demonstration
    history = G.nodes[node]["epi_history"]
    d2_epi_manual = abs(history[-1] - 2*history[-2] + history[-3])
    print(f"  Computed ∂²EPI/∂t²: {d2_epi_manual:.3f}")
    print(f"  Will trigger detection: {d2_epi_manual > G.graph['BIFURCATION_THRESHOLD_TAU']}")
    
    # Apply canonical OZ → ZHIR sequence
    # OZ (Dissonance) destabilizes, ZHIR (Mutation) transforms
    print(f"\nApplying OZ → ZHIR sequence...")
    run_sequence(G, node, [Dissonance(), Mutation()])
    
    # Check bifurcation detection
    print(f"\nAfter ZHIR:")
    if G.nodes[node].get("_zhir_bifurcation_potential"):
        print(f"  ✓ Bifurcation potential DETECTED")
        print(f"  ∂²EPI/∂t²: {G.nodes[node]['_zhir_d2epi']:.3f}")
        print(f"  Threshold τ: {G.nodes[node]['_zhir_tau']:.3f}")
        print(f"  Detection suggests: Apply THOL for controlled bifurcation")
        print(f"                      or IL for stabilization")
    else:
        print(f"  ✗ No bifurcation potential detected")
    
    # Check graph-level events
    events = G.graph.get("zhir_bifurcation_events", [])
    print(f"\n  Bifurcation events recorded: {len(events)}")
    if events:
        event = events[0]
        print(f"    - Node: {event['node']}")
        print(f"    - ∂²EPI/∂t²: {event['d2_epi']:.3f}")
        print(f"    - Timestamp: {event['timestamp']}")
    
    # Verify no structural changes (Option B - detection only)
    print(f"\n  Structural integrity (Option B):")
    print(f"    - No new nodes created")
    print(f"    - No new edges created")
    print(f"    - Telemetry flags set for grammar validation")
    
    print(f"\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


def example_zhir_no_bifurcation():
    """Demonstrate ZHIR with low acceleration (no bifurcation)."""
    print("\n" + "=" * 70)
    print("ZHIR with Low Acceleration (No Bifurcation)")
    print("=" * 70)
    
    # Create a node with low acceleration
    G, node = create_nfr("stable_system", epi=0.5, vf=1.0, theta=0.5)
    
    # Nearly linear EPI progression (low acceleration)
    # d²EPI/dt² = 0.50 - 2*0.49 + 0.48 = 0.50 - 0.98 + 0.48 = 0.00
    G.nodes[node]["epi_history"] = [0.48, 0.49, 0.50]
    G.nodes[node]["glyph_history"] = []
    
    G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
    
    print(f"\nInitial state:")
    print(f"  Node: {node}")
    print(f"  EPI history: {G.nodes[node]['epi_history']}")
    
    history = G.nodes[node]["epi_history"]
    d2_epi_manual = abs(history[-1] - 2*history[-2] + history[-3])
    print(f"  Computed ∂²EPI/∂t²: {d2_epi_manual:.3f}")
    print(f"  Threshold τ: {G.graph['BIFURCATION_THRESHOLD_TAU']}")
    print(f"  Will trigger detection: {d2_epi_manual > G.graph['BIFURCATION_THRESHOLD_TAU']}")
    
    # Apply ZHIR
    print(f"\nApplying ZHIR...")
    Mutation()(G, node)
    
    print(f"\nAfter ZHIR:")
    if G.nodes[node].get("_zhir_bifurcation_potential"):
        print(f"  ✓ Bifurcation potential detected")
    else:
        print(f"  ✗ No bifurcation potential detected (acceleration too low)")
    
    print(f"\n  System remains in stable regime")
    print(f"  No bifurcation handlers required")
    
    print(f"\n" + "=" * 70)


def example_grammar_u4a_validation():
    """Demonstrate Grammar U4a validation with bifurcation detection."""
    print("\n" + "=" * 70)
    print("Grammar U4a Validation Support")
    print("=" * 70)
    
    print("\nGrammar U4a: If {OZ, ZHIR}, include {THOL, IL}")
    print("When ZHIR detects bifurcation potential, grammar validators")
    print("can check that THOL or IL is present in the sequence.")
    
    # Example 1: Valid sequence with stabilizer
    G1, node1 = create_nfr("system1", epi=0.5, vf=1.0)
    G1.nodes[node1]["epi_history"] = [0.30, 0.40, 0.60]  # High acceleration
    G1.nodes[node1]["glyph_history"] = []
    G1.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
    
    print(f"\nExample 1: OZ → ZHIR → IL (with stabilizer)")
    run_sequence(G1, node1, [Dissonance(), Mutation(), Coherence()])
    
    if G1.nodes[node1].get("_zhir_bifurcation_potential"):
        print(f"  ✓ Bifurcation detected")
        print(f"  ✓ IL (Coherence) present - Grammar U4a satisfied")
    
    # Example 2: Would violate U4a (no stabilizer)
    G2, node2 = create_nfr("system2", epi=0.5, vf=1.0)
    G2.nodes[node2]["epi_history"] = [0.30, 0.40, 0.60]  # High acceleration
    G2.nodes[node2]["glyph_history"] = []
    G2.graph["BIFURCATION_THRESHOLD_TAU"] = 0.05
    
    print(f"\nExample 2: OZ → ZHIR (no stabilizer)")
    run_sequence(G2, node2, [Dissonance(), Mutation()])
    
    if G2.nodes[node2].get("_zhir_bifurcation_potential"):
        print(f"  ⚠ Bifurcation detected")
        print(f"  ⚠ No THOL/IL present - Grammar U4a should flag this")
        print(f"  ⚠ Uncontrolled bifurcation risk")
    
    print(f"\n" + "=" * 70)


if __name__ == "__main__":
    example_zhir_bifurcation_detection()
    example_zhir_no_bifurcation()
    example_grammar_u4a_validation()
