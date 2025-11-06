"""Basic demonstration of TNFR sequence visualization.

This example shows how to create the four main visualization types:
1. Sequence flow diagrams
2. Health dashboards  
3. Pattern analysis
4. Frequency timelines
"""

import tempfile
from pathlib import Path

from tnfr.visualization import SequenceVisualizer
from tnfr.operators.grammar import validate_sequence_with_health


# Use system temp directory for cross-platform compatibility
OUTPUT_DIR = Path(tempfile.gettempdir()) / "tnfr_viz_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Run basic visualization examples."""
    
    # Define example sequences (must follow TNFR canonical grammar)
    sequences = {
        "activation": ["emission", "reception", "coherence", "silence"],
        "therapeutic": ["emission", "reception", "coherence", "dissonance", 
                       "self_organization", "coherence", "silence"],
        "regenerative": ["emission", "reception", "coherence", "resonance", "expansion", 
                        "coherence", "transition"],
    }
    
    # Create visualizer
    visualizer = SequenceVisualizer(figsize=(12, 8), dpi=100)
    
    for name, sequence in sequences.items():
        print(f"\n{'='*60}")
        print(f"Visualizing {name.upper()} sequence")
        print(f"{'='*60}")
        print(f"Sequence: {' → '.join(sequence)}")
        
        # Validate and get health metrics
        result = validate_sequence_with_health(sequence)
        
        if not result.passed:
            print(f"⚠️  Sequence validation failed: {result.message}")
            continue
        
        health = result.health_metrics
        print(f"\n✓ Overall Health: {health.overall_health:.3f}")
        print(f"✓ Dominant Pattern: {health.dominant_pattern}")
        print(f"✓ Coherence Index: {health.coherence_index:.3f}")
        print(f"✓ Balance Score: {health.balance_score:.3f}")
        print(f"✓ Sustainability: {health.sustainability_index:.3f}")
        
        # 1. Sequence Flow Diagram
        print("\n📊 Generating sequence flow diagram...")
        fig1, ax1 = visualizer.plot_sequence_flow(
            sequence, 
            health_metrics=health,
            save_path=str(OUTPUT_DIR / f"{name}_flow.png")
        )
        print(f"   Saved to: {OUTPUT_DIR / f'{name}_flow.png'}")
        
        # 2. Health Dashboard
        print("📊 Generating health dashboard...")
        fig2, axes2 = visualizer.plot_health_dashboard(
            health,
            save_path=str(OUTPUT_DIR / f"{name}_dashboard.png")
        )
        print(f"   Saved to: {OUTPUT_DIR / f'{name}_dashboard.png'}")
        
        # 3. Pattern Analysis
        print("📊 Generating pattern analysis...")
        fig3, ax3 = visualizer.plot_pattern_analysis(
            sequence,
            pattern=health.dominant_pattern,
            save_path=str(OUTPUT_DIR / f"{name}_pattern.png")
        )
        print(f"   Saved to: {OUTPUT_DIR / f'{name}_pattern.png'}")
        
        # 4. Frequency Timeline
        print("📊 Generating frequency timeline...")
        fig4, ax4 = visualizer.plot_frequency_timeline(
            sequence,
            save_path=str(OUTPUT_DIR / f"{name}_timeline.png")
        )
        print(f"   Saved to: {OUTPUT_DIR / f'{name}_timeline.png'}")
        
        # Show recommendations if any
        if health.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in health.recommendations:
                print(f"   • {rec}")
    
    print(f"\n{'='*60}")
    print("All visualizations complete!")
    print(f"{'='*60}")
    print(f"\nVisualization files saved to {OUTPUT_DIR}")
    print("You can view them with any image viewer.")


if __name__ == "__main__":
    main()
