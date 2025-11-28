#!/usr/bin/env python3
"""Example demonstrating ZHIR threshold verification (∂EPI/∂t > ξ).

This example shows how the canonical mutation threshold requirement works
in practice, including cases where it's met and where it generates warnings.

References:
- AGENTS.md §11 (Mutation operator)
- TNFR.pdf §2.2.11 (ZHIR physics)
"""

from tnfr.structural import create_nfr
from tnfr.operators.definitions import Dissonance, Mutation, Coherence


def example_low_threshold():
    """Mutation without sufficient velocity generates warning."""
    print("\n=== Example 1: Low Velocity (∂EPI/∂t < ξ) ===")
    print("Creating node with low structural change velocity...")
    
    G, node = create_nfr("test_low", epi=0.3, vf=1.0)
    G.graph["ZHIR_THRESHOLD_XI"] = 0.1  # Canonical threshold
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True  # Enable validation
    
    # Build EPI history with low velocity
    # ∂EPI/∂t ≈ 0.30 - 0.29 = 0.01 < ξ=0.1
    G.nodes[node]["epi_history"] = [0.28, 0.29, 0.30]
    
    print(f"EPI history: {G.nodes[node]['epi_history']}")
    print(f"∂EPI/∂t ≈ {0.30 - 0.29:.3f} < ξ={G.graph['ZHIR_THRESHOLD_XI']}")
    print("\nApplying ZHIR (Mutation)...")
    
    # Apply mutation - will generate warning
    Mutation()(G, node)
    
    print(f"✓ Mutation applied")
    print(f"⚠ Warning flag set: {G.nodes[node].get('_zhir_threshold_warning', False)}")
    print("→ Mutation may lack structural justification")


def example_high_threshold():
    """Mutation with sufficient velocity succeeds cleanly."""
    print("\n=== Example 2: High Velocity (∂EPI/∂t > ξ) ===")
    print("Creating node with high structural change velocity...")
    
    G, node = create_nfr("test_high", epi=0.5, vf=1.0)
    G.graph["ZHIR_THRESHOLD_XI"] = 0.1  # Canonical threshold
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True  # Enable validation
    
    # Build EPI history with high velocity
    # ∂EPI/∂t ≈ 0.50 - 0.38 = 0.12 > ξ=0.1
    G.nodes[node]["epi_history"] = [0.25, 0.38, 0.50]
    
    print(f"EPI history: {G.nodes[node]['epi_history']}")
    print(f"∂EPI/∂t ≈ {0.50 - 0.38:.3f} > ξ={G.graph['ZHIR_THRESHOLD_XI']}")
    print("\nApplying ZHIR (Mutation)...")
    
    # Apply mutation - will succeed without warning
    Mutation()(G, node)
    
    print(f"✓ Mutation applied")
    print(f"✓ Threshold met: {G.nodes[node].get('_zhir_threshold_met', False)}")
    print("→ Phase transformation justified by structural velocity")


def example_canonical_sequence():
    """OZ → ZHIR sequence generates sufficient threshold."""
    print("\n=== Example 3: Canonical Sequence (OZ → ZHIR) ===")
    print("Canonical sequence: IL → OZ → ZHIR")
    print("OZ (Dissonance) elevates ΔNFR, increasing structural velocity...")
    
    G, node = create_nfr("test_canonical", epi=0.4, vf=1.0)
    G.graph["ZHIR_THRESHOLD_XI"] = 0.1
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True  # Enable validation
    
    # Initialize with moderate history
    G.nodes[node]["epi_history"] = [0.35, 0.38, 0.40]
    
    print(f"Initial EPI history: {G.nodes[node]['epi_history']}")
    print("\nApplying sequence: IL → OZ → ZHIR...")
    
    # Apply canonical sequence
    Coherence()(G, node)  # Stabilize
    Dissonance()(G, node)  # Increase ΔNFR (elevates velocity)
    Mutation()(G, node)  # Phase transformation
    
    print(f"✓ Sequence completed")
    
    # Check metrics
    if "operator_metrics" in G.graph:
        zhir_metrics = [m for m in G.graph["operator_metrics"] if m.get("glyph") == "ZHIR"]
        if zhir_metrics:
            m = zhir_metrics[-1]
            print(f"\nZHIR Metrics:")
            print(f"  ∂EPI/∂t = {m.get('depi_dt', 0):.3f}")
            print(f"  Threshold ξ = {m.get('threshold_xi', 0):.3f}")
            print(f"  Threshold met: {m.get('threshold_met', False)}")
            print(f"  Ratio: {m.get('threshold_ratio', 0):.2f}x")
            
            if m.get('threshold_met'):
                print("→ OZ successfully elevated velocity above threshold")


def example_metrics():
    """Show detailed threshold metrics."""
    print("\n=== Example 4: Detailed Threshold Metrics ===")
    print("Configuring detailed metrics collection...")
    
    G, node = create_nfr("test_metrics", epi=0.6, vf=1.0)
    G.graph["ZHIR_THRESHOLD_XI"] = 0.1
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True  # Enable validation
    
    # Build history with known velocity: 0.30 velocity / 0.1 threshold = 3.0x
    G.nodes[node]["epi_history"] = [0.10, 0.30, 0.60]
    
    print(f"EPI history: {G.nodes[node]['epi_history']}")
    print(f"∂EPI/∂t ≈ {0.60 - 0.30:.3f} (3.0x threshold)")
    print("\nApplying ZHIR...")
    
    Mutation()(G, node)
    
    # Display full metrics
    if "operator_metrics" in G.graph:
        m = G.graph["operator_metrics"][-1]
        print(f"\nFull ZHIR Metrics:")
        print(f"  Operator: {m['operator']}")
        print(f"  Glyph: {m['glyph']}")
        print(f"  Phase shift: {m['theta_shift']:.3f}")
        print(f"  Delta EPI: {m['delta_epi']:.3f}")
        print(f"  ---")
        print(f"  ∂EPI/∂t: {m['depi_dt']:.3f}")
        print(f"  Threshold ξ: {m['threshold_xi']:.3f}")
        print(f"  Threshold met: {m['threshold_met']}")
        print(f"  Threshold ratio: {m['threshold_ratio']:.2f}x")
        print(f"  Exceeded by: {m['threshold_exceeded_by']:.3f}")
        print(f"  ---")
        print(f"  Validated: {m['threshold_validated']}")
        print(f"  Warning: {m['threshold_warning']}")


def main():
    """Run all examples."""
    print("=" * 70)
    print("ZHIR (Mutation) Threshold Verification Examples")
    print("=" * 70)
    print("\nCanonical Requirement: θ → θ' when ∂EPI/∂t > ξ")
    print("(Phase transformation requires sufficient structural velocity)")
    
    example_low_threshold()
    example_high_threshold()
    example_canonical_sequence()
    example_metrics()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("• ZHIR threshold verification ensures mutations are justified")
    print("• Default threshold ξ=0.1 represents minimum velocity for phase change")
    print("• OZ (Dissonance) sequences naturally elevate velocity above threshold")
    print("• Warnings are logged (not blocking) for backward compatibility")
    print("• Metrics include full threshold verification telemetry")
    print("\nConfiguration:")
    print("  G.graph['ZHIR_THRESHOLD_XI'] = 0.1  # Default threshold")
    print("  G.graph['COLLECT_OPERATOR_METRICS'] = True  # Enable telemetry")
    print("=" * 70)


if __name__ == "__main__":
    main()
