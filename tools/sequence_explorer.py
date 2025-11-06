"""Interactive sequence explorer for TNFR operator sequences.

A command-line tool for exploring and analyzing TNFR operator sequences with
comprehensive visualization and health analysis.

Usage:
    python -m tools.sequence_explorer --sequence emission reception coherence silence
    python -m tools.sequence_explorer --interactive
    python -m tools.sequence_explorer --compare seq1.txt seq2.txt

Examples:
    # Analyze a single sequence
    python -m tools.sequence_explorer --sequence emission reception coherence silence

    # Interactive mode
    python -m tools.sequence_explorer --interactive

    # Compare multiple sequences
    python -m tools.sequence_explorer --compare seq1.txt seq2.txt seq3.txt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    from tnfr.visualization import SequenceVisualizer
    from tnfr.operators.grammar import validate_sequence_with_health
    from tnfr.operators.health_analyzer import SequenceHealthMetrics
except ImportError as e:
    print(f"Error: Unable to import TNFR modules: {e}", file=sys.stderr)
    print("Please ensure TNFR is installed: pip install -e .", file=sys.stderr)
    sys.exit(1)


class InteractiveSequenceExplorer:
    """Interactive explorer for TNFR operator sequences.
    
    Provides comprehensive analysis and visualization capabilities through
    a command-line interface.
    """
    
    def __init__(self):
        """Initialize the interactive sequence explorer."""
        self.visualizer = SequenceVisualizer()
    
    def explore_sequence(
        self, 
        sequence: List[str],
        output_dir: str = "/tmp",
        show_plots: bool = False,
    ) -> None:
        """Explore and analyze a single sequence comprehensively.
        
        Parameters
        ----------
        sequence : List[str]
            Sequence of operator names to analyze
        output_dir : str, optional
            Directory to save visualizations, by default "/tmp"
        show_plots : bool, optional
            Whether to show plots interactively, by default False
        """
        print(f"\n{'='*70}")
        print(f"TNFR Sequence Analysis")
        print(f"{'='*70}")
        print(f"\nSequence: {' → '.join(sequence)}")
        print(f"Length: {len(sequence)} operators\n")
        
        # Validate and analyze
        result = validate_sequence_with_health(sequence)
        
        if not result.passed:
            print(f"❌ Validation FAILED")
            print(f"   Error: {result.message}\n")
            if result.error:
                print(f"   Details: {result.error}")
            return
        
        print(f"✓ Validation PASSED")
        
        health = result.health_metrics
        
        # Display health metrics
        print(f"\n{'─'*70}")
        print("HEALTH METRICS")
        print(f"{'─'*70}")
        print(f"Overall Health:         {health.overall_health:.3f}")
        print(f"Coherence Index:        {health.coherence_index:.3f}")
        print(f"Balance Score:          {health.balance_score:.3f}")
        print(f"Sustainability:         {health.sustainability_index:.3f}")
        print(f"Complexity Efficiency:  {health.complexity_efficiency:.3f}")
        print(f"Frequency Harmony:      {health.frequency_harmony:.3f}")
        print(f"Pattern Completeness:   {health.pattern_completeness:.3f}")
        print(f"Transition Smoothness:  {health.transition_smoothness:.3f}")
        
        print(f"\n{'─'*70}")
        print("PATTERN INFORMATION")
        print(f"{'─'*70}")
        print(f"Dominant Pattern: {health.dominant_pattern}")
        
        # Display recommendations
        if health.recommendations:
            print(f"\n{'─'*70}")
            print("RECOMMENDATIONS")
            print(f"{'─'*70}")
            for i, rec in enumerate(health.recommendations, 1):
                print(f"{i}. {rec}")
        
        # Generate visualizations
        print(f"\n{'─'*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'─'*70}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sequence flow
        flow_path = output_path / "sequence_flow.png"
        print(f"📊 Sequence flow diagram... ", end="")
        self.visualizer.plot_sequence_flow(
            sequence, health_metrics=health, save_path=str(flow_path)
        )
        print(f"✓ {flow_path}")
        
        # Health dashboard
        dashboard_path = output_path / "health_dashboard.png"
        print(f"📊 Health dashboard... ", end="")
        self.visualizer.plot_health_dashboard(health, save_path=str(dashboard_path))
        print(f"✓ {dashboard_path}")
        
        # Pattern analysis
        pattern_path = output_path / "pattern_analysis.png"
        print(f"📊 Pattern analysis... ", end="")
        self.visualizer.plot_pattern_analysis(
            sequence, pattern=health.dominant_pattern, save_path=str(pattern_path)
        )
        print(f"✓ {pattern_path}")
        
        # Frequency timeline
        timeline_path = output_path / "frequency_timeline.png"
        print(f"📊 Frequency timeline... ", end="")
        self.visualizer.plot_frequency_timeline(sequence, save_path=str(timeline_path))
        print(f"✓ {timeline_path}")
        
        print(f"\n{'='*70}")
        print(f"Analysis complete! Visualizations saved to: {output_dir}")
        print(f"{'='*70}\n")
    
    def compare_sequences(
        self,
        sequences: List[List[str]],
        labels: Optional[List[str]] = None,
        output_dir: str = "/tmp",
    ) -> None:
        """Compare multiple sequences side-by-side.
        
        Parameters
        ----------
        sequences : List[List[str]]
            List of sequences to compare
        labels : List[str], optional
            Labels for each sequence
        output_dir : str, optional
            Directory to save comparison visualizations
        """
        if not sequences:
            print("No sequences to compare.")
            return
        
        if labels is None:
            labels = [f"Sequence {i+1}" for i in range(len(sequences))]
        
        print(f"\n{'='*70}")
        print(f"TNFR Sequence Comparison")
        print(f"{'='*70}\n")
        
        # Analyze all sequences
        results = []
        for label, seq in zip(labels, sequences):
            print(f"Analyzing {label}: {' → '.join(seq)}")
            result = validate_sequence_with_health(seq)
            results.append((label, seq, result))
        
        # Display comparison table
        print(f"\n{'─'*70}")
        print("COMPARISON TABLE")
        print(f"{'─'*70}")
        print(f"{'Sequence':<20} {'Valid':<8} {'Health':<10} {'Pattern':<20}")
        print(f"{'─'*70}")
        
        for label, seq, result in results:
            valid = "✓" if result.passed else "✗"
            health_str = f"{result.health_metrics.overall_health:.3f}" if result.passed else "N/A"
            pattern = result.health_metrics.dominant_pattern if result.passed else "N/A"
            print(f"{label:<20} {valid:<8} {health_str:<10} {pattern:<20}")
        
        # Rank by health
        valid_results = [(l, s, r) for l, s, r in results if r.passed]
        if valid_results:
            print(f"\n{'─'*70}")
            print("HEALTH RANKING (Valid Sequences Only)")
            print(f"{'─'*70}")
            ranked = sorted(
                valid_results,
                key=lambda x: x[2].health_metrics.overall_health,
                reverse=True
            )
            for i, (label, seq, result) in enumerate(ranked, 1):
                health = result.health_metrics.overall_health
                print(f"{i}. {label}: {health:.3f}")
        
        print(f"\n{'='*70}\n")
    
    def interactive_mode(self) -> None:
        """Run in interactive mode for exploratory analysis."""
        print("\n" + "="*70)
        print("TNFR Interactive Sequence Explorer")
        print("="*70)
        print("\nEnter operator sequences to analyze.")
        print("Commands:")
        print("  - Enter operators separated by spaces")
        print("  - 'help' - Show available operators")
        print("  - 'quit' or 'exit' - Exit explorer\n")
        
        from tnfr.config.operator_names import (
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, COUPLING,
            RESONANCE, SILENCE, EXPANSION, CONTRACTION,
            SELF_ORGANIZATION, MUTATION, TRANSITION, RECURSIVITY,
        )
        
        operators = [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, COUPLING,
            RESONANCE, SILENCE, EXPANSION, CONTRACTION,
            SELF_ORGANIZATION, MUTATION, TRANSITION, RECURSIVITY,
        ]
        
        while True:
            try:
                user_input = input("\nSequence> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ('quit', 'exit'):
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nAvailable operators:")
                    for op in operators:
                        print(f"  - {op}")
                    continue
                
                # Parse sequence
                sequence = user_input.split()
                
                if not sequence:
                    print("Error: Empty sequence")
                    continue
                
                # Analyze sequence
                self.explore_sequence(sequence)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point for sequence explorer."""
    parser = argparse.ArgumentParser(
        description="Interactive TNFR sequence explorer and analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--sequence",
        nargs="+",
        help="Sequence of operators to analyze",
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple sequences (provide file paths or sequences)",
    )
    
    parser.add_argument(
        "--output",
        default="/tmp",
        help="Output directory for visualizations (default: /tmp)",
    )
    
    args = parser.parse_args()
    
    explorer = InteractiveSequenceExplorer()
    
    if args.interactive:
        explorer.interactive_mode()
    elif args.sequence:
        explorer.explore_sequence(args.sequence, output_dir=args.output)
    elif args.compare:
        # Parse compare arguments
        sequences = []
        labels = []
        for i, item in enumerate(args.compare):
            path = Path(item)
            if path.exists() and path.is_file():
                # Read sequence from file
                with open(path) as f:
                    seq = f.read().strip().split()
                sequences.append(seq)
                labels.append(path.stem)
            else:
                # Treat as sequence name
                print(f"Warning: {item} is not a file, skipping")
        
        if sequences:
            explorer.compare_sequences(sequences, labels=labels, output_dir=args.output)
        else:
            print("No valid sequences to compare")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
