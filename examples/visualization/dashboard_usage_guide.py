"""Dashboard usage guide for TNFR sequence visualization.

This guide demonstrates practical usage patterns for the visualization dashboard,
including health monitoring, optimization workflows, and educational scenarios.
"""

from tnfr.visualization import SequenceVisualizer
from tnfr.operators.grammar import validate_sequence_with_health


def health_monitoring_example():
    """Example: Monitor sequence health during development."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Health Monitoring During Development")
    print("="*70)
    
    visualizer = SequenceVisualizer()
    
    # Developer iterates on a sequence
    iterations = [
        ("Initial attempt", ["emission", "coherence", "silence"]),
        ("Add reception", ["emission", "reception", "coherence", "silence"]),
        ("Add resonance", ["emission", "reception", "coherence", "resonance", "silence"]),
    ]
    
    for iteration_name, sequence in iterations:
        print(f"\n{iteration_name}: {' → '.join(sequence)}")
        result = validate_sequence_with_health(sequence)
        
        if result.passed:
            health = result.health_metrics
            print(f"  Overall Health: {health.overall_health:.3f}")
            print(f"  Coherence: {health.coherence_index:.3f}")
            print(f"  Balance: {health.balance_score:.3f}")
            
            # Generate dashboard for detailed analysis
            fig, axes = visualizer.plot_health_dashboard(
                health,
                save_path=f"/tmp/health_monitor_{iteration_name.replace(' ', '_')}.png"
            )
            print(f"  Dashboard saved for detailed review")
        else:
            print(f"  ❌ Validation failed: {result.message}")


def optimization_workflow_example():
    """Example: Optimize sequence based on visual feedback."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Optimization Workflow")
    print("="*70)
    
    visualizer = SequenceVisualizer()
    
    # Starting sequence
    current = ["emission", "reception", "coherence", "silence"]
    print(f"\nCurrent sequence: {' → '.join(current)}")
    
    result_current = validate_sequence_with_health(current)
    health_current = result_current.health_metrics
    
    # Proposed optimization: add resonance for amplification
    optimized = ["emission", "reception", "coherence", "resonance", "silence"]
    print(f"Optimized sequence: {' → '.join(optimized)}")
    
    result_optimized = validate_sequence_with_health(optimized)
    health_optimized = result_optimized.health_metrics
    
    # Compare health metrics
    print("\n" + "-"*70)
    print("HEALTH COMPARISON")
    print("-"*70)
    print(f"{'Metric':<25} {'Current':>10} {'Optimized':>10} {'Change':>10}")
    print("-"*70)
    
    metrics = [
        ("Overall Health", health_current.overall_health, health_optimized.overall_health),
        ("Coherence Index", health_current.coherence_index, health_optimized.coherence_index),
        ("Balance Score", health_current.balance_score, health_optimized.balance_score),
        ("Sustainability", health_current.sustainability_index, health_optimized.sustainability_index),
        ("Efficiency", health_current.complexity_efficiency, health_optimized.complexity_efficiency),
    ]
    
    for metric_name, current_val, optimized_val in metrics:
        change = optimized_val - current_val
        change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"
        print(f"{metric_name:<25} {current_val:>10.3f} {optimized_val:>10.3f} {change_str:>10}")
    
    # Generate comparison visualizations
    print("\nGenerating comparison visualizations...")
    fig1, ax1 = visualizer.plot_sequence_flow(
        current, health_metrics=health_current,
        save_path="/tmp/optimization_current_flow.png"
    )
    fig2, ax2 = visualizer.plot_sequence_flow(
        optimized, health_metrics=health_optimized,
        save_path="/tmp/optimization_optimized_flow.png"
    )
    print("✓ Flow diagrams saved")
    
    fig3, axes3 = visualizer.plot_health_dashboard(
        health_optimized,
        save_path="/tmp/optimization_dashboard.png"
    )
    print("✓ Health dashboard saved")


def educational_example():
    """Example: Educational visualization of pattern types."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Educational Pattern Demonstration")
    print("="*70)
    
    visualizer = SequenceVisualizer()
    
    # Different pattern types for teaching
    educational_patterns = {
        "Activation Pattern": {
            "sequence": ["emission", "reception", "coherence", "silence"],
            "description": "Basic activation: initiate, anchor, stabilize, pause",
        },
        "Therapeutic Pattern": {
            "sequence": ["emission", "reception", "coherence", "dissonance", 
                        "self_organization", "coherence", "silence"],
            "description": "Healing sequence: controlled tension leading to reorganization",
        },
        "Resonance Pattern": {
            "sequence": ["emission", "reception", "coherence", "resonance", 
                        "coupling", "resonance", "silence"],
            "description": "Amplification: propagate and strengthen patterns",
        },
    }
    
    for pattern_name, pattern_info in educational_patterns.items():
        print(f"\n{'-'*70}")
        print(f"{pattern_name}")
        print(f"{'-'*70}")
        print(f"Description: {pattern_info['description']}")
        print(f"Sequence: {' → '.join(pattern_info['sequence'])}")
        
        result = validate_sequence_with_health(pattern_info['sequence'])
        
        if result.passed:
            health = result.health_metrics
            print(f"Health: {health.overall_health:.3f} | Pattern: {health.dominant_pattern}")
            
            # Generate educational visualizations
            safe_name = pattern_name.replace(" ", "_").lower()
            
            # Flow diagram
            fig1, ax1 = visualizer.plot_sequence_flow(
                pattern_info['sequence'],
                health_metrics=health,
                save_path=f"/tmp/edu_{safe_name}_flow.png"
            )
            
            # Pattern analysis
            fig2, ax2 = visualizer.plot_pattern_analysis(
                pattern_info['sequence'],
                pattern=health.dominant_pattern,
                save_path=f"/tmp/edu_{safe_name}_pattern.png"
            )
            
            # Frequency timeline
            fig3, ax3 = visualizer.plot_frequency_timeline(
                pattern_info['sequence'],
                save_path=f"/tmp/edu_{safe_name}_timeline.png"
            )
            
            print(f"✓ Visualizations saved for {pattern_name}")


def debugging_example():
    """Example: Visual debugging of sequence issues."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Visual Debugging")
    print("="*70)
    
    visualizer = SequenceVisualizer()
    
    # Problematic sequence with issues
    problematic = ["emission", "reception", "coherence", "dissonance", 
                   "dissonance", "mutation", "coherence", "silence"]
    
    print(f"\nProblematic sequence: {' → '.join(problematic)}")
    result = validate_sequence_with_health(problematic)
    
    if result.passed:
        health = result.health_metrics
        print(f"\n⚠️  Sequence is valid but has health issues:")
        print(f"   Overall Health: {health.overall_health:.3f}")
        print(f"   Balance Score: {health.balance_score:.3f} (low!)")
        
        if health.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in health.recommendations:
                print(f"   • {rec}")
        
        # Visual debugging
        print("\nGenerating diagnostic visualizations...")
        
        # Flow diagram shows transition issues
        fig1, ax1 = visualizer.plot_sequence_flow(
            problematic,
            health_metrics=health,
            save_path="/tmp/debug_flow.png"
        )
        print("✓ Flow diagram (check arrow colors for transition issues)")
        
        # Timeline shows frequency problems
        fig2, ax2 = visualizer.plot_frequency_timeline(
            problematic,
            save_path="/tmp/debug_timeline.png"
        )
        print("✓ Frequency timeline (check for invalid transitions)")
        
        # Dashboard quantifies the issues
        fig3, axes3 = visualizer.plot_health_dashboard(
            health,
            save_path="/tmp/debug_dashboard.png"
        )
        print("✓ Health dashboard (low metrics highlighted)")
    else:
        print(f"❌ Sequence validation failed: {result.message}")


def main():
    """Run all dashboard usage examples."""
    print("\n" + "="*70)
    print("TNFR Visualization Dashboard - Usage Guide")
    print("="*70)
    print("\nThis guide demonstrates practical workflows for using")
    print("the TNFR sequence visualization dashboard.\n")
    
    # Run all examples
    health_monitoring_example()
    optimization_workflow_example()
    educational_example()
    debugging_example()
    
    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)
    print("\nVisualization files saved to /tmp/")
    print("\nKey Takeaways:")
    print("  • Use flow diagrams to spot transition issues")
    print("  • Use dashboards for comprehensive health assessment")
    print("  • Use pattern analysis to understand structural roles")
    print("  • Use timelines to verify frequency transitions")
    print("  • Compare visualizations before/after optimization")


if __name__ == "__main__":
    main()
