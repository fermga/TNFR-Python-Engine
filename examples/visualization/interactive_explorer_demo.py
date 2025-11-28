"""Interactive exploration demo for TNFR sequence visualizer.

This example demonstrates how to use the InteractiveSequenceExplorer
programmatically for batch analysis and comparison.
"""

import sys
import tempfile
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.sequence_explorer import InteractiveSequenceExplorer


# Use system temp directory for cross-platform compatibility
OUTPUT_BASE = Path(tempfile.gettempdir()) / "tnfr_explorer_demo"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def demo_single_exploration():
    """Demonstrate single sequence exploration."""
    print("\n" + "="*70)
    print("DEMO 1: Single Sequence Exploration")
    print("="*70)
    
    explorer = InteractiveSequenceExplorer()
    
    # Analyze a therapeutic sequence
    sequence = ["emission", "reception", "coherence", "dissonance", 
                "self_organization", "coherence", "silence"]
    
    output_dir = OUTPUT_BASE / "therapeutic"
    explorer.explore_sequence(sequence, output_dir=str(output_dir))


def demo_sequence_comparison():
    """Demonstrate comparison of multiple sequences."""
    print("\n" + "="*70)
    print("DEMO 2: Sequence Comparison")
    print("="*70)
    
    explorer = InteractiveSequenceExplorer()
    
    sequences = [
        ["emission", "reception", "coherence", "silence"],
        ["emission", "reception", "coherence", "resonance", "silence"],
        ["emission", "reception", "coherence", "expansion", "coherence", "silence"],
        ["emission", "reception", "coherence", "dissonance", "mutation", "coherence", "silence"],
    ]
    
    labels = [
        "Basic Activation",
        "Activation + Resonance",
        "Activation + Expansion",
        "Activation + Mutation",
    ]
    
    output_dir = OUTPUT_BASE / "comparison"
    explorer.compare_sequences(sequences, labels=labels, output_dir=str(output_dir))


def demo_pattern_variations():
    """Demonstrate analysis of different pattern types."""
    print("\n" + "="*70)
    print("DEMO 3: Pattern Variations Analysis")
    print("="*70)
    
    explorer = InteractiveSequenceExplorer()
    
    pattern_sequences = {
        "Linear": ["emission", "reception", "coherence", "resonance", "silence"],
        "Hierarchical": ["emission", "reception", "coherence", "self_organization", 
                        "coherence", "silence"],
        "Cyclic": ["emission", "reception", "coherence", "transition", 
                   "resonance", "transition"],
        "Bifurcated": ["emission", "reception", "coherence", "dissonance", 
                      "mutation", "coherence", "silence"],
    }
    
    for pattern_name, sequence in pattern_sequences.items():
        print(f"\n{'─'*70}")
        print(f"Analyzing {pattern_name} Pattern")
        print(f"{'─'*70}")
        
        output_dir = OUTPUT_BASE / f"pattern_{pattern_name.lower()}"
        explorer.explore_sequence(sequence, output_dir=str(output_dir))


def main():
    """Run all demonstration scenarios."""
    print("\n" + "="*70)
    print("TNFR Interactive Sequence Explorer - Demonstrations")
    print("="*70)
    print("\nThis demo shows programmatic usage of the InteractiveSequenceExplorer")
    print("for batch analysis, comparison, and pattern exploration.\n")
    
    # Run demonstrations
    demo_single_exploration()
    demo_sequence_comparison()
    demo_pattern_variations()
    
    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70)
    print(f"\nCheck {OUTPUT_BASE} for visualizations.")


if __name__ == "__main__":
    main()
