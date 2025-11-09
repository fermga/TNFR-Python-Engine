"""Demo script showing enhanced NAV transition metrics.

This script demonstrates the comprehensive metrics now collected by the NAV
(Transition) operator, including regime classification, scaling factors, and
latency tracking.
"""

from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Silence, Transition, Coherence
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_VF, ALIAS_THETA, ALIAS_DNFR, ALIAS_EPI
import json


def print_metrics(metrics: dict, title: str):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    
    # Core identification
    print(f"Operator: {metrics['operator']} ({metrics['glyph']})")
    
    # Regime classification
    print(f"\nRegime Classification:")
    print(f"  Origin:      {metrics['regime_origin']}")
    print(f"  Destination: {metrics['regime_destination']}")
    print(f"  Type:        {metrics['transition_type']}")
    
    # Phase metrics
    print(f"\nPhase Metrics:")
    print(f"  Magnitude:   {metrics['phase_shift_magnitude']:.4f} rad")
    print(f"  Signed:      {metrics['phase_shift_signed']:.4f} rad")
    print(f"  Final:       {metrics['theta_final']:.4f} rad")
    
    # Structural scaling
    print(f"\nStructural Scaling:")
    print(f"  νf scaling:  {metrics['vf_scaling_factor']:.4f}x")
    print(f"  ΔNFR damping: {metrics['dnfr_damping_ratio']:.4f}x")
    if metrics['epi_preservation'] is not None:
        print(f"  EPI preservation: {metrics['epi_preservation']:.4f}")
    
    # Deltas
    print(f"\nChanges:")
    print(f"  Δνf:   {metrics['delta_vf']:+.4f}")
    print(f"  ΔΔNFR: {metrics['delta_dnfr']:+.4f}")
    
    # Latency
    if metrics['latency_duration'] is not None:
        print(f"\nLatency:")
        print(f"  Duration: {metrics['latency_duration']:.6f} seconds")
    
    # Status
    print(f"\nStatus:")
    print(f"  Transition complete: {metrics['transition_complete']}")


def demo_latent_to_active():
    """Demonstrate latent → active reactivation."""
    print("\n" + "="*60)
    print("DEMO 1: Latent → Active Reactivation")
    print("="*60)
    
    # Create node and apply silence
    G, node = create_nfr("demo1", epi=0.5, vf=0.8)
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    
    print("\n1. Initial state:")
    print(f"   EPI={get_attr(G.nodes[node], ALIAS_EPI, 0.0):.4f}, "
          f"νf={get_attr(G.nodes[node], ALIAS_VF, 0.0):.4f}")
    
    # Apply silence to enter latency
    run_sequence(G, node, [Silence()])
    print("\n2. After Silence (SHA):")
    print(f"   EPI={get_attr(G.nodes[node], ALIAS_EPI, 0.0):.4f}, "
          f"νf={get_attr(G.nodes[node], ALIAS_VF, 0.0):.4f}, "
          f"latent={G.nodes[node].get('latent', False)}")
    
    # Apply transition
    Transition()(G, node)
    print("\n3. After Transition (NAV):")
    print(f"   EPI={get_attr(G.nodes[node], ALIAS_EPI, 0.0):.4f}, "
          f"νf={get_attr(G.nodes[node], ALIAS_VF, 0.0):.4f}, "
          f"latent={G.nodes[node].get('latent', False)}")
    
    # Show metrics
    metrics = G.graph["operator_metrics"][-1]
    print_metrics(metrics, "Reactivation Metrics")
    
    # Verify expectations
    assert metrics["transition_type"] == "reactivation"
    assert metrics["regime_origin"] == "latent"
    assert metrics["vf_scaling_factor"] > 1.0  # νf increased
    assert metrics["latency_duration"] is not None
    
    print("\n✓ Latent → Active reactivation successful!")


def demo_active_to_active():
    """Demonstrate active → active standard transition."""
    print("\n" + "="*60)
    print("DEMO 2: Active → Active Standard Transition")
    print("="*60)
    
    # Create node in active regime
    G, node = create_nfr("demo2", epi=0.4, vf=0.6)
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    
    print("\n1. Initial state (active regime):")
    print(f"   EPI={get_attr(G.nodes[node], ALIAS_EPI, 0.0):.4f}, "
          f"νf={get_attr(G.nodes[node], ALIAS_VF, 0.0):.4f}")
    
    # Apply transition
    Transition()(G, node)
    
    print("\n2. After Transition (NAV):")
    print(f"   EPI={get_attr(G.nodes[node], ALIAS_EPI, 0.0):.4f}, "
          f"νf={get_attr(G.nodes[node], ALIAS_VF, 0.0):.4f}")
    
    # Show metrics
    metrics = G.graph["operator_metrics"][-1]
    print_metrics(metrics, "Standard Transition Metrics")
    
    # Verify expectations
    assert metrics["transition_type"] == "regime_change"
    assert metrics["regime_origin"] == "active"
    assert metrics["regime_destination"] == "active"
    # Active transition preserves νf by default
    assert 0.9 <= metrics["vf_scaling_factor"] <= 1.1
    assert metrics["latency_duration"] is None  # No prior silence
    
    print("\n✓ Active → Active transition successful!")


def demo_resonant_to_active():
    """Demonstrate resonant → active stabilization."""
    print("\n" + "="*60)
    print("DEMO 3: Resonant → Active Stabilization")
    print("="*60)
    
    # Create node in resonant regime (high EPI and νf)
    G, node = create_nfr("demo3", epi=0.7, vf=0.9)
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    
    print("\n1. Initial state (resonant regime):")
    print(f"   EPI={get_attr(G.nodes[node], ALIAS_EPI, 0.0):.4f}, "
          f"νf={get_attr(G.nodes[node], ALIAS_VF, 0.0):.4f}")
    
    # Apply transition
    Transition()(G, node)
    
    print("\n2. After Transition (NAV):")
    print(f"   EPI={get_attr(G.nodes[node], ALIAS_EPI, 0.0):.4f}, "
          f"νf={get_attr(G.nodes[node], ALIAS_VF, 0.0):.4f}")
    
    # Show metrics
    metrics = G.graph["operator_metrics"][-1]
    print_metrics(metrics, "Stabilization Metrics")
    
    # Verify expectations
    assert metrics["regime_origin"] == "resonant"
    # Resonant transition reduces νf slightly for stability
    assert metrics["vf_scaling_factor"] < 1.0
    assert metrics["vf_scaling_factor"] >= 0.9
    
    print("\n✓ Resonant → Active stabilization successful!")


def demo_phase_wrapping():
    """Demonstrate phase wrapping at 2π boundary."""
    print("\n" + "="*60)
    print("DEMO 4: Phase Wrapping at 2π Boundary")
    print("="*60)
    
    # Create node with phase near 2π
    G, node = create_nfr("demo4", epi=0.5, vf=0.6, theta=6.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    
    theta_before = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
    print(f"\n1. Initial phase: {theta_before:.4f} rad (near 2π = {2*3.14159:.4f})")
    
    # Apply transition
    Transition()(G, node)
    
    theta_after = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
    print(f"2. Final phase:   {theta_after:.4f} rad")
    
    # Show metrics
    metrics = G.graph["operator_metrics"][-1]
    
    print(f"\nPhase Change:")
    print(f"  Raw:     {theta_after - theta_before:+.4f} rad")
    print(f"  Wrapped: {metrics['phase_shift_signed']:+.4f} rad")
    print(f"  Magnitude: {metrics['phase_shift_magnitude']:.4f} rad")
    
    # Verify wrapping
    import math
    assert abs(metrics['phase_shift_signed']) <= math.pi
    print(f"\n✓ Phase correctly wrapped to [-π, π]!")


def demo_epi_preservation():
    """Demonstrate EPI preservation tracking."""
    print("\n" + "="*60)
    print("DEMO 5: EPI Preservation Tracking")
    print("="*60)
    
    # Create node and stabilize with coherence
    G, node = create_nfr("demo5", epi=0.6, vf=0.7)
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    
    # Apply coherence then transition
    run_sequence(G, node, [Coherence()])
    epi_before_nav = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    
    Transition()(G, node)
    epi_after_nav = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
    
    # Show metrics
    metrics = G.graph["operator_metrics"][-1]
    
    print(f"\nEPI Tracking:")
    print(f"  Before NAV: {epi_before_nav:.6f}")
    print(f"  After NAV:  {epi_after_nav:.6f}")
    print(f"  Preservation ratio: {metrics['epi_preservation']:.6f}")
    print(f"  Drift: {abs(epi_after_nav - epi_before_nav):.6f}")
    
    # Verify preservation
    assert metrics['epi_preservation'] is not None
    assert 0.95 <= metrics['epi_preservation'] <= 1.05
    print(f"\n✓ EPI identity preserved (< 5% drift)!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NAV TRANSITION METRICS DEMONSTRATION")
    print("Enhanced metrics for regime classification and scaling")
    print("="*60)
    
    demo_latent_to_active()
    demo_active_to_active()
    demo_resonant_to_active()
    demo_phase_wrapping()
    demo_epi_preservation()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey Enhancements:")
    print("  ✓ Regime origin/destination classification")
    print("  ✓ Transition type (reactivation/phase_shift/regime_change)")
    print("  ✓ Phase shift magnitude with proper 2π wrapping")
    print("  ✓ Frequency scaling factor (νf_after / νf_before)")
    print("  ✓ ΔNFR damping ratio")
    print("  ✓ EPI preservation tracking")
    print("  ✓ Latency duration from SHA → NAV")
    print("  ✓ Full backward compatibility")
    print()
