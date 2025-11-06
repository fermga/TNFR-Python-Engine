"""TNFR Therapeutic Optimization - Comparative Analysis and Recommendations.

This module provides comparative analysis of therapeutic sequences,
helping practitioners select and optimize operator patterns based on:
- Clinical context and goals
- Structural health metrics
- Pattern effectiveness
- Sustainability considerations

Demonstrates how TNFR metrics guide therapeutic decision-making.
"""

from typing import Dict, List, Tuple
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
from tnfr.config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)


# =============================================================================
# COMPARATIVE ANALYSIS UTILITIES
# =============================================================================


def compare_sequences(
    sequences: Dict[str, List[str]],
    context: str = "General therapeutic context"
) -> Dict[str, Dict]:
    """Compare multiple therapeutic sequences and provide recommendations.
    
    Parameters
    ----------
    sequences : Dict[str, List[str]]
        Dictionary mapping sequence names to operator lists
    context : str, optional
        Clinical context for the comparison
    
    Returns
    -------
    Dict[str, Dict]
        Comparison results with metrics and recommendations
    """
    results = {}
    
    print(f"\n{'=' * 70}")
    print(f"COMPARATIVE ANALYSIS: {context}")
    print(f"{'=' * 70}\n")
    
    for name, sequence in sequences.items():
        result = validate_sequence_with_health(sequence)
        
        if result.passed:
            health = result.health_metrics
            results[name] = {
                "sequence": sequence,
                "health": health,
                "passed": True,
                "overall": health.overall_health,
                "coherence": health.coherence_index,
                "balance": health.balance_score,
                "sustainability": health.sustainability_index,
                "efficiency": health.complexity_efficiency,
                "pattern": health.dominant_pattern,
                "length": len(sequence),
            }
        else:
            results[name] = {
                "passed": False,
                "message": result.message,
            }
    
    return results


def print_comparison_table(results: Dict[str, Dict]) -> None:
    """Print formatted comparison table of sequences.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Results from compare_sequences()
    """
    print(f"\n{'Sequence':<30s} {'Health':<8s} {'Balance':<8s} {'Sustain':<8s} {'Pattern':<15s}")
    print("-" * 80)
    
    for name, data in results.items():
        if data["passed"]:
            health = f"{data['overall']:.3f}"
            balance = f"{data['balance']:.3f}"
            sustain = f"{data['sustainability']:.3f}"
            pattern = data['pattern']
            
            # Color-code health status
            if data['overall'] >= 0.85:
                status = "🌟"
            elif data['overall'] >= 0.75:
                status = "✓"
            else:
                status = "⚠"
            
            print(f"{name:<30s} {status} {health:<6s} {balance:<8s} {sustain:<8s} {pattern:<15s}")
        else:
            print(f"{name:<30s} ✗ INVALID - {data['message']}")


def recommend_best_sequence(
    results: Dict[str, Dict],
    priority: str = "overall"
) -> Tuple[str, Dict]:
    """Recommend the best sequence based on specified priority.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Results from compare_sequences()
    priority : str, optional
        Optimization priority: 'overall', 'balance', 'sustainability', 'efficiency'
    
    Returns
    -------
    Tuple[str, Dict]
        Name and data of recommended sequence
    """
    valid_results = {k: v for k, v in results.items() if v["passed"]}
    
    if not valid_results:
        return None, {"error": "No valid sequences to compare"}
    
    if priority == "overall":
        best = max(valid_results.items(), key=lambda x: x[1]["overall"])
    elif priority == "balance":
        best = max(valid_results.items(), key=lambda x: x[1]["balance"])
    elif priority == "sustainability":
        best = max(valid_results.items(), key=lambda x: x[1]["sustainability"])
    elif priority == "efficiency":
        best = max(valid_results.items(), key=lambda x: x[1]["efficiency"])
    else:
        best = max(valid_results.items(), key=lambda x: x[1]["overall"])
    
    return best


# =============================================================================
# EXAMPLE 1: Crisis Intervention - Speed vs. Depth
# =============================================================================


def example_crisis_intervention_optimization():
    """Compare crisis intervention approaches: rapid vs. thorough."""
    
    sequences = {
        "Rapid Stabilization": [
            EMISSION, RECEPTION, COHERENCE, COUPLING, SILENCE
        ],
        "With Controlled Crisis": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, CONTRACTION,
            COHERENCE, COUPLING, SILENCE
        ],
        "With Brief Processing": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION,
            COHERENCE, RESONANCE, SILENCE
        ],
    }
    
    results = compare_sequences(
        sequences,
        "Crisis Intervention: Speed vs. Depth Trade-off"
    )
    
    print_comparison_table(results)
    
    # Recommendations based on different priorities
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS BY PRIORITY")
    print(f"{'=' * 70}\n")
    
    for priority in ["overall", "balance", "sustainability"]:
        best_name, best_data = recommend_best_sequence(results, priority)
        if best_data and "error" not in best_data:
            print(f"{priority.capitalize():15s}: {best_name} (score: {best_data[priority]:.3f})")
    
    print(f"\n{'=' * 70}")
    print("CLINICAL GUIDANCE")
    print(f"{'=' * 70}\n")
    print("• Rapid Stabilization: Use for acute panic, immediate safety needs")
    print("• With Controlled Crisis: Better balance, suitable for moderate distress")
    print("• With Brief Processing: Higher transformation potential, requires more time")
    print("\nRecommendation: Select based on patient stability and available session time")


# =============================================================================
# EXAMPLE 2: Trauma Processing - Exposure Intensity
# =============================================================================


def example_trauma_processing_optimization():
    """Compare trauma processing approaches with different dissonance levels."""
    
    sequences = {
        "Gradual Exposure": [
            EMISSION, RECEPTION, COHERENCE, CONTRACTION, COHERENCE,
            RESONANCE, COUPLING, SILENCE
        ],
        "Standard Exposure": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION,
            COHERENCE, RESONANCE, COUPLING, SILENCE
        ],
        "Intensive Exposure": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION,
            COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE, SILENCE
        ],
    }
    
    results = compare_sequences(
        sequences,
        "Trauma Processing: Exposure Intensity Calibration"
    )
    
    print_comparison_table(results)
    
    best_name, best_data = recommend_best_sequence(results, "overall")
    
    print(f"\n{'=' * 70}")
    print("CLINICAL GUIDANCE")
    print(f"{'=' * 70}\n")
    print("• Gradual: No direct trauma exposure, builds resources first")
    print("• Standard: Controlled single exposure with self-organization (recommended)")
    print("• Intensive: Multiple exposures with mutation - only for stabilized patients")
    print(f"\nBest Overall: {best_name} (health: {best_data['overall']:.3f})")
    print("  Rationale: Optimal balance of processing depth and patient safety")


# =============================================================================
# EXAMPLE 3: Depression Treatment - Activation Strategies
# =============================================================================


def example_depression_activation_optimization():
    """Compare depression treatment sequences with different activation strategies."""
    
    sequences = {
        "Minimal Activation": [
            EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE
        ],
        "Gradual Reactivation": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, CONTRACTION,
            COHERENCE, EXPANSION, COHERENCE, RESONANCE, SILENCE
        ],
        "Transformative Activation": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION,
            COHERENCE, EXPANSION, RESONANCE, COUPLING, SILENCE
        ],
    }
    
    results = compare_sequences(
        sequences,
        "Depression Treatment: Activation Strategy Selection"
    )
    
    print_comparison_table(results)
    
    # Compare sustainability specifically (important for depression)
    best_sustain = max(
        ((k, v) for k, v in results.items() if v["passed"]),
        key=lambda x: x[1]["sustainability"]
    )
    
    print(f"\n{'=' * 70}")
    print("CLINICAL GUIDANCE")
    print(f"{'=' * 70}\n")
    print("• Minimal: Gentle start, risk of insufficient activation")
    print("• Gradual: Better balance, sustainable progression")
    print("• Transformative: Deeper change, requires patient readiness")
    print(f"\nMost Sustainable: {best_sustain[0]} (sustain: {best_sustain[1]['sustainability']:.3f})")
    print("  Critical for depression: long-term maintenance capacity")


# =============================================================================
# EXAMPLE 4: Therapeutic Modality Comparison
# =============================================================================


def example_modality_comparison():
    """Compare different therapeutic modalities using TNFR sequences."""
    
    sequences = {
        "Supportive Therapy": [
            EMISSION, RECEPTION, COHERENCE, RESONANCE, COUPLING,
            COHERENCE, RESONANCE, SILENCE
        ],
        "CBT-Style (Restructuring)": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, CONTRACTION,
            COHERENCE, EXPANSION, COHERENCE, RESONANCE, SILENCE
        ],
        "Psychodynamic (Insight)": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION,
            COHERENCE, RESONANCE, COUPLING, SILENCE
        ],
        "Integrated Approach": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION,
            COHERENCE, EXPANSION, RESONANCE, COUPLING, SILENCE
        ],
    }
    
    results = compare_sequences(
        sequences,
        "Therapeutic Modality Structural Comparison"
    )
    
    print_comparison_table(results)
    
    best_name, best_data = recommend_best_sequence(results, "overall")
    
    print(f"\n{'=' * 70}")
    print("MODALITY ANALYSIS")
    print(f"{'=' * 70}\n")
    
    for name, data in results.items():
        if data["passed"]:
            print(f"\n{name}:")
            print(f"  Pattern: {data['pattern']}")
            print(f"  Strengths: ", end="")
            
            if data['balance'] > 0.4:
                print("balanced forces, ", end="")
            if data['sustainability'] > 0.8:
                print("highly sustainable, ", end="")
            if data['efficiency'] > 0.9:
                print("efficient, ", end="")
            if data['overall'] > 0.8:
                print("excellent health", end="")
            
            print()
    
    print(f"\n{'=' * 70}")
    print(f"Best Overall Structural Health: {best_name}")
    print(f"  Health: {best_data['overall']:.3f}")
    print(f"  Pattern: {best_data['pattern']}")
    print(f"{'=' * 70}")


# =============================================================================
# EXAMPLE 5: Optimization Recommendations
# =============================================================================


def example_sequence_optimization_guide():
    """Demonstrate how to optimize a suboptimal sequence."""
    
    print(f"\n{'=' * 70}")
    print("SEQUENCE OPTIMIZATION GUIDE")
    print(f"{'=' * 70}\n")
    
    # Original suboptimal sequence
    original = [EMISSION, RECEPTION, DISSONANCE, EXPANSION, RESONANCE]
    
    print("Original Sequence (suboptimal):")
    print(f"  {' → '.join(original)}")
    
    result_orig = validate_sequence_with_health(original)
    original_health = 0.0
    
    if result_orig.passed:
        health_orig = result_orig.health_metrics
        original_health = health_orig.overall_health
        print(f"\n  Health: {health_orig.overall_health:.3f} (⚠ FAIR)")
        print(f"  Balance: {health_orig.balance_score:.3f}")
        print(f"  Recommendations:")
        for rec in health_orig.recommendations:
            print(f"    • {rec}")
    else:
        print(f"\n  ✗ INVALID: {result_orig.message}")
    
    # Optimized sequences
    sequences = {
        "Add Initial Stabilizer": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, EXPANSION,
            COHERENCE, RESONANCE, SILENCE
        ],
        "Add Contraction for Balance": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, CONTRACTION,
            COHERENCE, RESONANCE, SILENCE
        ],
        "Add Self-Organization": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION,
            COHERENCE, RESONANCE, SILENCE
        ],
    }
    
    print(f"\n{'=' * 70}")
    print("OPTIMIZATION ATTEMPTS")
    print(f"{'=' * 70}")
    
    results = {}
    for name, seq in sequences.items():
        result = validate_sequence_with_health(seq)
        if result.passed:
            h = result.health_metrics
            results[name] = {
                "passed": True,
                "overall": h.overall_health,
                "balance": h.balance_score,
                "pattern": h.dominant_pattern,
            }
            print(f"\n{name}:")
            print(f"  {' → '.join(seq)}")
            print(f"  Health: {h.overall_health:.3f} ({h.dominant_pattern})")
            print(f"  Balance: {h.balance_score:.3f}")
            if original_health > 0:
                print(f"  Improvement: +{h.overall_health - original_health:.3f}")
        else:
            print(f"\n{name}:")
            print(f"  ✗ INVALID: {result.message}")
    
    # Recommend best
    if results:
        best = max(results.items(), key=lambda x: x[1]["overall"])
        print(f"\n{'=' * 70}")
        print(f"RECOMMENDED OPTIMIZATION: {best[0]}")
        print(f"  Final Health: {best[1]['overall']:.3f}")
        print(f"  Pattern: {best[1]['pattern']}")
        print(f"{'=' * 70}")


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================


def main():
    """Run all optimization examples."""
    print("\n" + "=" * 70)
    print("TNFR THERAPEUTIC OPTIMIZATION")
    print("Comparative Analysis and Clinical Decision Support")
    print("=" * 70)
    
    example_crisis_intervention_optimization()
    print("\n\n")
    
    example_trauma_processing_optimization()
    print("\n\n")
    
    example_depression_activation_optimization()
    print("\n\n")
    
    example_modality_comparison()
    print("\n\n")
    
    example_sequence_optimization_guide()
    
    print("\n" + "=" * 70)
    print("KEY OPTIMIZATION PRINCIPLES")
    print("=" * 70)
    print("\n1. Balance: Maintain equilibrium between stabilizers and destabilizers")
    print("2. Sustainability: Ensure sequences end with stabilizers for long-term viability")
    print("3. Efficiency: Longer sequences must provide proportional value")
    print("4. Context: Match pattern intensity to patient stability and readiness")
    print("5. Coherence: Follow TNFR grammar rules for structural integrity")
    print("\n6. Clinical Integration:")
    print("   • Use health metrics as decision-support, not replacement for judgment")
    print("   • Consider pattern type in context of therapeutic goals")
    print("   • Monitor phase synchrony (φ) in relationship/couples work")
    print("   • Respect patient's structural frequency (νf) - don't force activation")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
