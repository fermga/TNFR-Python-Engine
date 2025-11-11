"""Demonstration of extended emission metrics with structural fidelity indicators.

This script shows how to use the new AL-specific metrics to analyze
emission quality and structural effects.
"""

from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Reception, Coherence, Silence


def print_emission_metrics(metrics: dict) -> None:
    """Print emission metrics in a readable format."""
    print(f"\n{'='*60}")
    print(f"Operator: {metrics['operator']} ({metrics['glyph']})")
    print(f"{'='*60}")

    # Core structural changes
    print("\n📊 Core Structural Changes:")
    print(f"  ΔE (delta_epi):        {metrics['delta_epi']:+.3f}")
    print(f"  Δνf (delta_vf):        {metrics['delta_vf']:+.3f}")
    print(f"  ΔNFR initialized:      {metrics['dnfr_initialized']:+.3f}")
    print(f"  θ (theta):             {metrics['theta_current']:.3f} rad")

    # AL-specific quality indicators
    print("\n✨ AL-Specific Quality Indicators:")
    print(f"  Emission Quality:      {metrics['emission_quality'].upper()}")
    print(
        f"  From Latency:          {'✓' if metrics['activation_from_latency'] else '✗'}"
    )
    print(f"  Form Emergence:        {metrics['form_emergence_magnitude']:+.3f}")
    print(f"  Frequency Activated:   {'✓' if metrics['frequency_activation'] else '✗'}")
    print(
        f"  Positive ΔNFR:         {'✓' if metrics['reorganization_positive'] else '✗'}"
    )

    # Traceability
    print("\n🔍 Traceability:")
    print(f"  Emission Timestamp:    {metrics['emission_timestamp'] or 'N/A'}")
    print(
        f"  Irreversibility Flag:  {'✓' if metrics['irreversibility_marker'] else '✗'}"
    )

    # Final state
    print("\n📈 Final State:")
    print(f"  EPI:                   {metrics['epi_final']:.3f}")
    print(f"  νf:                    {metrics['vf_final']:.3f} Hz_str")
    print(f"  ΔNFR:                  {metrics['dnfr_final']:+.3f}")


def demo_latent_activation():
    """Demonstrate emission from latent state (EPI < 0.3)."""
    print("\n" + "=" * 60)
    print("DEMO 1: Emission from Latent State")
    print("=" * 60)
    print("Creating node with low EPI (0.15) - latent state")

    G, node = create_nfr("latent_node", epi=0.15, vf=1.2)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]  # First operator (Emission)
    print_emission_metrics(metrics)


def demo_active_emission():
    """Demonstrate emission from active state (EPI >= 0.3)."""
    print("\n" + "=" * 60)
    print("DEMO 2: Emission from Active State")
    print("=" * 60)
    print("Creating node with higher EPI (0.5) - active state")

    G, node = create_nfr("active_node", epi=0.5, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]  # First operator (Emission)
    print_emission_metrics(metrics)


def demo_weak_emission():
    """Demonstrate weak emission (high EPI, low νf)."""
    print("\n" + "=" * 60)
    print("DEMO 3: Weak Emission Scenario")
    print("=" * 60)
    print("Creating node with high EPI (0.75) and low νf (0.1)")

    G, node = create_nfr("weak_node", epi=0.75, vf=0.1)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][0]  # First operator (Emission)
    print_emission_metrics(metrics)


def demo_multiple_emissions():
    """Demonstrate multiple emissions tracking."""
    print("\n" + "=" * 60)
    print("DEMO 4: Multiple Emissions on Same Node")
    print("=" * 60)
    print("Applying emission twice to track reactivation")

    G, node = create_nfr("reactivation_node", epi=0.2, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # First emission
    print("\n--- First Emission ---")
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    first_metrics = G.graph["operator_metrics"][0]
    print_emission_metrics(first_metrics)

    # Second emission (reactivation)
    print("\n--- Second Emission (Reactivation) ---")
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    # Find the second Emission metric (should be at index 4: AL, EN, IL, SHA, AL)
    second_metrics = G.graph["operator_metrics"][4]
    print_emission_metrics(second_metrics)

    print("\n📝 Note: Original timestamp preserved, activation count incremented")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("EMISSION METRICS DEMONSTRATION")
    print("Extended AL-specific metrics with structural fidelity")
    print("=" * 60)

    demo_latent_activation()
    demo_active_emission()
    demo_weak_emission()
    demo_multiple_emissions()

    print("\n" + "=" * 60)
    print("SUMMARY OF NEW METRICS")
    print("=" * 60)
    print(
        """
    The extended emission metrics provide:
    
    1. emission_quality: "valid" | "weak"
       - Qualitative assessment of emission effectiveness
    
    2. activation_from_latency: bool
       - Was the node truly latent (EPI < 0.3) before emission?
    
    3. form_emergence_magnitude: float
       - Absolute EPI increment (structural form activation)
    
    4. frequency_activation: bool
       - Did νf increase (structural frequency activated)?
    
    5. reorganization_positive: bool
       - Is ΔNFR positive (positive reorganization gradient)?
    
    6. emission_timestamp: str | None
       - ISO UTC timestamp for traceability
    
    7. irreversibility_marker: bool
       - Has the node been structurally activated?
    
    These metrics enable:
    - Qualitative analysis of emission effectiveness
    - Debugging spurious activations
    - Research on emergence dynamics
    - Validation of expected structural effects
    - Full traceability with timestamps
    """
    )

    print("\nDemonstration complete! ✨")


if __name__ == "__main__":
    main()
