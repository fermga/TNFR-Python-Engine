"""TNFR Educational Optimization - Comparative Analysis and Recommendations.

This module provides comparative analysis of educational sequences,
helping educators select and optimize operator patterns based on:
- Learning context and goals
- Structural health metrics
- Pattern effectiveness
- Sustainability considerations

Demonstrates how TNFR metrics guide pedagogical decision-making.
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
    context: str = "General learning context"
) -> Dict[str, Dict]:
    """Compare multiple educational sequences and provide recommendations.
    
    Parameters
    ----------
    sequences : Dict[str, List[str]]
        Dictionary mapping sequence names to operator lists
    context : str, optional
        Learning context for the comparison
    
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
# EXAMPLE 1: Conceptual Learning - Direct vs. Exploratory
# =============================================================================


def example_conceptual_learning_optimization():
    """Compare approaches to conceptual learning: direct instruction vs. discovery."""
    
    sequences = {
        "Direct Instruction": [
            EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE
        ],
        "Guided Discovery": [
            EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE,
            DISSONANCE, MUTATION, COHERENCE, SILENCE
        ],
        "Problem-Based Learning": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION, COHERENCE, RESONANCE, SILENCE
        ],
    }
    
    results = compare_sequences(
        sequences,
        "Conceptual Learning: Instruction Approach Comparison"
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
    print("PEDAGOGICAL GUIDANCE")
    print(f"{'=' * 70}\n")
    print("• Direct Instruction: Efficient for foundational concepts, lower transformation")
    print("• Guided Discovery: Better balance, suitable for deeper understanding")
    print("• Problem-Based: Highest transformation, requires more time and support")


# =============================================================================
# EXAMPLE 2: Skill Practice - Focus vs. Variety
# =============================================================================


def example_skill_practice_optimization():
    """Compare skill practice approaches: focused repetition vs. varied practice."""
    
    sequences = {
        "Massed Practice": [
            EMISSION, RECEPTION, COHERENCE, CONTRACTION, COHERENCE, SILENCE
        ],
        "Distributed Practice": [
            EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE,
            SILENCE, EMISSION, TRANSITION
        ],
        "Interleaved Practice": [
            EMISSION, RECEPTION, COHERENCE, EXPANSION, COUPLING,
            TRANSITION, COHERENCE, SILENCE
        ],
    }
    
    results = compare_sequences(
        sequences,
        "Skill Practice: Practice Structure Comparison"
    )
    
    print_comparison_table(results)
    
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS BY PRIORITY")
    print(f"{'=' * 70}\n")
    
    for priority in ["overall", "sustainability", "efficiency"]:
        best_name, best_data = recommend_best_sequence(results, priority)
        if best_data and "error" not in best_data:
            print(f"{priority.capitalize():15s}: {best_name} (score: {best_data[priority]:.3f})")
    
    print(f"\n{'=' * 70}")
    print("PEDAGOGICAL GUIDANCE")
    print(f"{'=' * 70}\n")
    print("• Massed Practice: Quick skill acquisition, may have lower retention")
    print("• Distributed Practice: Better long-term retention, spaced over time")
    print("• Interleaved Practice: Highest transfer, requires more cognitive effort")


# =============================================================================
# EXAMPLE 3: Feedback Integration - Immediate vs. Delayed
# =============================================================================


def example_feedback_timing_optimization():
    """Compare feedback timing approaches in learning."""
    
    sequences = {
        "Immediate Feedback": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE,
            RESONANCE, SILENCE
        ],
        "Delayed Feedback": [
            EMISSION, RECEPTION, COHERENCE, SILENCE, EMISSION,
            DISSONANCE, SELF_ORGANIZATION, COHERENCE, SILENCE
        ],
        "Elaborative Feedback": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, CONTRACTION,
            COHERENCE, EXPANSION, COUPLING, TRANSITION
        ],
    }
    
    results = compare_sequences(
        sequences,
        "Feedback Timing: Integration Strategy Comparison"
    )
    
    print_comparison_table(results)
    
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS BY PRIORITY")
    print(f"{'=' * 70}\n")
    
    for priority in ["overall", "balance"]:
        best_name, best_data = recommend_best_sequence(results, priority)
        if best_data and "error" not in best_data:
            print(f"{priority.capitalize():15s}: {best_name} (score: {best_data[priority]:.3f})")
    
    print(f"\n{'=' * 70}")
    print("PEDAGOGICAL GUIDANCE")
    print(f"{'=' * 70}\n")
    print("• Immediate Feedback: Good for procedural skills, prevents error reinforcement")
    print("• Delayed Feedback: Promotes self-organization, better for complex learning")
    print("• Elaborative Feedback: Highest integration, connects to broader context")


# =============================================================================
# EXAMPLE 4: Assessment Approaches
# =============================================================================


def example_assessment_approach_optimization():
    """Compare different assessment sequence approaches."""
    
    sequences = {
        "Formative Assessment": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE, COHERENCE,
            RECURSIVITY
        ],
        "Summative Assessment": [
            EMISSION, RECEPTION, COHERENCE, RESONANCE, TRANSITION
        ],
        "Self-Assessment": [
            EMISSION, RECEPTION, COHERENCE, DISSONANCE,
            SELF_ORGANIZATION, SILENCE, EMISSION, RECURSIVITY
        ],
    }
    
    results = compare_sequences(
        sequences,
        "Assessment Approaches: Learning Impact Comparison"
    )
    
    print_comparison_table(results)
    
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS BY PRIORITY")
    print(f"{'=' * 70}\n")
    
    for priority in ["overall", "sustainability"]:
        best_name, best_data = recommend_best_sequence(results, priority)
        if best_data and "error" not in best_data:
            print(f"{priority.capitalize():15s}: {best_name} (score: {best_data[priority]:.3f})")
    
    print(f"\n{'=' * 70}")
    print("PEDAGOGICAL GUIDANCE")
    print(f"{'=' * 70}\n")
    print("• Formative Assessment: Continuous learning feedback, supports growth")
    print("• Summative Assessment: Final evaluation, marks transition to next level")
    print("• Self-Assessment: Metacognitive development, promotes autonomous learning")


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================


def main():
    """Run all educational optimization examples."""
    print("\n" + "=" * 70)
    print("TNFR EDUCATIONAL OPTIMIZATION DEMONSTRATIONS")
    print("=" * 70)
    
    print("\n\n" + "🎯 " + "EXAMPLE 1: Conceptual Learning Approaches".center(68) + " 🎯")
    example_conceptual_learning_optimization()
    
    print("\n\n" + "🎯 " + "EXAMPLE 2: Skill Practice Structures".center(68) + " 🎯")
    example_skill_practice_optimization()
    
    print("\n\n" + "🎯 " + "EXAMPLE 3: Feedback Timing Strategies".center(68) + " 🎯")
    example_feedback_timing_optimization()
    
    print("\n\n" + "🎯 " + "EXAMPLE 4: Assessment Approaches".center(68) + " 🎯")
    example_assessment_approach_optimization()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nTNFR structural metrics provide objective guidance for pedagogical decisions:")
    print("• Overall Health: Combined quality indicator for learning effectiveness")
    print("• Balance Score: Equilibrium between challenge and support")
    print("• Sustainability: Long-term learning retention and transfer potential")
    print("• Pattern Detection: Identifies learning trajectory type")
    print("\nUse these metrics to:")
    print("1. Compare different instructional approaches objectively")
    print("2. Optimize sequences for specific learning goals")
    print("3. Predict learning outcomes and retention")
    print("4. Design evidence-based curriculum and assessment")
    print("=" * 70)


if __name__ == "__main__":
    main()
