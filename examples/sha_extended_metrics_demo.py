"""Demo: Extended SHA (Silence) Metrics

This demo shows how to use the extended metrics for the SHA operator
to analyze structural preservation effectiveness in different scenarios.

The extended metrics enable deep analysis of:
- EPI variance during silence
- Preservation integrity quality
- Reactivation readiness assessment
- Time-to-collapse prediction
"""

from tnfr.structural import create_nfr
from tnfr.operators.metrics import silence_metrics
from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_VF, ALIAS_EPI


def print_metrics(title: str, metrics: dict) -> None:
    """Pretty print extended metrics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Operator: {metrics['operator']} ({metrics['glyph']})")
    print(f"\nCore Metrics:")
    print(f"  vf_reduction: {metrics['vf_reduction']:.4f}")
    print(f"  vf_final: {metrics['vf_final']:.4f}")
    print(f"  epi_preservation: {metrics['epi_preservation']:.4f}")
    print(f"  is_silent: {metrics['is_silent']}")
    print(f"\nExtended Metrics:")
    print(f"  epi_variance: {metrics['epi_variance']:.6f}")
    print(f"  preservation_integrity: {metrics['preservation_integrity']:.4f}")
    print(f"  reactivation_readiness: {metrics['reactivation_readiness']:.4f}")
    time_val = metrics['time_to_collapse']
    time_str = "infinite" if time_val == float('inf') else f"{time_val:.2f} steps"
    print(f"  time_to_collapse: {time_str}")


def demo_biomedical_use_case():
    """Biomedical: Sleep consolidation tracking."""
    print("\n🏥 BIOMEDICAL USE CASE: Sleep Consolidation")
    print("Tracking HRV signal during 8-hour sleep period")

    # Create node representing HRV signal
    G, node = create_nfr("hrv_signal", epi=0.6, vf=0.02)

    # Set up silence tracking
    G.nodes[node]["preserved_epi"] = 0.6
    G.nodes[node]["silence_duration"] = 8.0  # 8 hours
    G.nodes[node]["epi_history_during_silence"] = [
        0.6, 0.602, 0.598, 0.601, 0.599, 0.600
    ]  # Minimal variance (stable sleep)
    G.nodes[node]["epi_drift_rate"] = 0.001  # Very slow drift

    metrics = silence_metrics(G, node, vf_before=0.5, epi_before=0.6)
    print_metrics("Sleep Consolidation Analysis", metrics)

    print("\n📊 Interpretation:")
    print("  ✓ High preservation integrity (>0.99) = Stable sleep")
    print("  ✓ Low EPI variance (<0.01) = No sleep disturbances")
    print("  ✓ Moderate readiness (0.3-0.8) = Natural post-sleep state")
    print("  ✓ Long time-to-collapse (>100) = Healthy resilience")


def demo_cognitive_use_case():
    """Cognitive: Memory consolidation during incubation."""
    print("\n🧠 COGNITIVE USE CASE: Memory Consolidation")
    print("Memory trace during incubation period")

    # Create node representing memory trace
    G, node = create_nfr("memory_trace", epi=0.7, vf=0.03)

    # Set up perfect preservation
    G.nodes[node]["preserved_epi"] = 0.7
    G.nodes[node]["silence_duration"] = 2.0  # Brief pause
    G.nodes[node]["epi_history_during_silence"] = [0.7, 0.7, 0.7]  # Perfect stability
    G.nodes[node]["epi_drift_rate"] = 0.0  # No drift

    # Add network support (other memory traces)
    for i in range(4):
        G.add_node(f"memory_{i}")
        set_attr(G.nodes[f"memory_{i}"], ALIAS_VF, 0.5)
        set_attr(G.nodes[f"memory_{i}"], ALIAS_EPI, 0.6)
        G.add_edge(node, f"memory_{i}")

    metrics = silence_metrics(G, node, vf_before=0.5, epi_before=0.7)
    print_metrics("Memory Consolidation Analysis", metrics)

    print("\n📊 Interpretation:")
    print("  ✓ Perfect integrity (1.0) = Excellent consolidation")
    print("  ✓ Zero variance = No memory degradation")
    print("  ✓ High readiness (>0.7) = Ready for recall")
    print("  ✓ Infinite collapse time = Stable long-term memory")


def demo_social_use_case():
    """Social: Strategic pause in conflict."""
    print("\n🤝 SOCIAL USE CASE: Strategic Pause in Conflict")
    print("Conflict state paused for de-escalation")

    # Create node representing conflict state
    G, node = create_nfr("conflict_state", epi=0.4, vf=0.01)

    # Set up degrading silence (some EPI loss)
    G.nodes[node]["preserved_epi"] = 0.45
    G.nodes[node]["silence_duration"] = 15.0  # Long strategic pause
    G.nodes[node]["epi_history_during_silence"] = [
        0.45, 0.44, 0.42, 0.41, 0.40
    ]  # Gradual degradation
    G.nodes[node]["epi_drift_rate"] = 0.01  # Slow degradation

    metrics = silence_metrics(G, node, vf_before=0.8, epi_before=0.45)
    print_metrics("Strategic Pause Analysis", metrics)

    print("\n📊 Interpretation:")
    print("  ⚠ Good integrity (0.85-0.95) = Acceptable degradation")
    print("  ⚠ Some variance = Tension still present")
    print("  ⚠ Moderate readiness (<0.6) = Extended pause needed")
    print("  ⚠ Finite collapse (30-50 steps) = Act before breakdown")


def demo_preservation_failure():
    """Example: Detection of preservation failure."""
    print("\n⚠️  FAILURE DETECTION: Poor Preservation")
    print("Node with excessive EPI drift during silence")

    # Create node with poor preservation
    G, node = create_nfr("unstable_node", epi=0.2, vf=0.05)

    # Set up failing preservation
    G.nodes[node]["preserved_epi"] = 0.5  # Started at 0.5
    G.nodes[node]["silence_duration"] = 5.0
    G.nodes[node]["epi_history_during_silence"] = [
        0.5, 0.4, 0.3, 0.25, 0.2
    ]  # Rapid degradation
    G.nodes[node]["epi_drift_rate"] = 0.06  # High drift

    metrics = silence_metrics(G, node, vf_before=1.0, epi_before=0.5)
    print_metrics("Preservation Failure Detection", metrics)

    print("\n📊 Interpretation:")
    print("  ❌ Low integrity (<0.8) = PRESERVATION FAILURE")
    print("  ❌ High variance = Unstable structure")
    print("  ❌ Low readiness (<0.3) = RISKY REACTIVATION")
    print("  ❌ Short collapse time (<10) = IMMINENT COLLAPSE")
    print("\n⚡ ACTION REQUIRED: Apply IL (Coherence) before reactivation")


if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Extended SHA Metrics Demo                                 ║")
    print("║  Analyzing Structural Preservation Effectiveness           ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Run all demos
    demo_biomedical_use_case()
    demo_cognitive_use_case()
    demo_social_use_case()
    demo_preservation_failure()

    print("\n" + "="*60)
    print("✨ Demo Complete!")
    print("="*60)
    print("\nThe extended SHA metrics enable:")
    print("  • Deep analysis of preservation quality")
    print("  • Early detection of structural degradation")
    print("  • Informed reactivation timing decisions")
    print("  • Cross-domain application (biomedical, cognitive, social)")
