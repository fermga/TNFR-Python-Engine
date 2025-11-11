"""Demonstration of context-guided sequence generation capabilities.

This example showcases the different ways to use the ContextualSequenceGenerator
to create, optimize, and improve TNFR operator sequences.
"""

from tnfr.tools import ContextualSequenceGenerator, list_domains, list_objectives
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer


def print_separator(title: str = "") -> None:
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)
    print()


def print_result(result: object, show_recommendations: bool = True) -> None:
    """Print generation result in a readable format."""
    print(f"Generated Sequence: {' → '.join(result.sequence)}")  # type: ignore[attr-defined]
    print(f"Health Score: {result.health_score:.3f}")  # type: ignore[attr-defined]
    print(f"Detected Pattern: {result.detected_pattern}")  # type: ignore[attr-defined]
    
    if result.domain:  # type: ignore[attr-defined]
        print(f"Domain: {result.domain}")  # type: ignore[attr-defined]
    if result.objective:  # type: ignore[attr-defined]
        print(f"Objective: {result.objective}")  # type: ignore[attr-defined]
    
    if show_recommendations and result.recommendations:  # type: ignore[attr-defined]
        print("\nRecommendations:")
        for rec in result.recommendations:  # type: ignore[attr-defined]
            print(f"  • {rec}")


def demo_domain_listing() -> None:
    """Demonstrate listing available domains and objectives."""
    print_separator("AVAILABLE DOMAINS AND OBJECTIVES")
    
    domains = list_domains()
    print("Available Application Domains:")
    for domain in domains:
        print(f"\n  {domain.upper()}")
        objectives = list_objectives(domain)
        for obj in objectives:
            print(f"    - {obj}")


def demo_context_generation() -> None:
    """Demonstrate context-based sequence generation."""
    print_separator("CONTEXT-BASED GENERATION")
    
    generator = ContextualSequenceGenerator(seed=42)
    
    # Example 1: Therapeutic crisis intervention
    print("Example 1: Therapeutic Crisis Intervention")
    print("-" * 70)
    result = generator.generate_for_context(
        domain="therapeutic",
        objective="crisis_intervention",
        min_health=0.70
    )
    print_result(result)
    
    # Example 2: Educational skill development
    print("\n\nExample 2: Educational Skill Development")
    print("-" * 70)
    result = generator.generate_for_context(
        domain="educational",
        objective="skill_development",
        min_health=0.75,
        max_length=8
    )
    print_result(result)
    
    # Example 3: Organizational team building
    print("\n\nExample 3: Organizational Team Building")
    print("-" * 70)
    result = generator.generate_for_context(
        domain="organizational",
        objective="team_building",
        min_health=0.70
    )
    print_result(result)


def demo_pattern_generation() -> None:
    """Demonstrate pattern-targeted sequence generation."""
    print_separator("PATTERN-TARGETED GENERATION")
    
    generator = ContextualSequenceGenerator(seed=42)
    
    # Example 1: BOOTSTRAP pattern
    print("Example 1: BOOTSTRAP Pattern (System Initialization)")
    print("-" * 70)
    result = generator.generate_for_pattern(
        target_pattern="BOOTSTRAP",
        max_length=5,
        min_health=0.65
    )
    print_result(result)
    
    # Example 2: STABILIZE pattern
    print("\n\nExample 2: STABILIZE Pattern (Consolidation)")
    print("-" * 70)
    result = generator.generate_for_pattern(
        target_pattern="STABILIZE",
        min_health=0.70
    )
    print_result(result)
    
    # Example 3: EXPLORE pattern
    print("\n\nExample 3: EXPLORE Pattern (Controlled Exploration)")
    print("-" * 70)
    result = generator.generate_for_pattern(
        target_pattern="EXPLORE",
        min_health=0.70
    )
    print_result(result)


def demo_sequence_improvement() -> None:
    """Demonstrate sequence improvement capabilities."""
    print_separator("SEQUENCE IMPROVEMENT")
    
    generator = ContextualSequenceGenerator(seed=42)
    analyzer = SequenceHealthAnalyzer()
    
    # Example 1: Basic sequence improvement
    print("Example 1: Improving a Basic Sequence")
    print("-" * 70)
    current = ["emission", "coherence", "silence"]
    print(f"Current Sequence: {' → '.join(current)}")
    
    current_health = analyzer.analyze_health(current)
    print(f"Current Health: {current_health.overall_health:.3f}")
    print(f"Current Pattern: {current_health.dominant_pattern}")
    
    improved, recommendations = generator.improve_sequence(
        current,
        target_health=0.80
    )
    
    improved_health = analyzer.analyze_health(improved)
    print(f"\nImproved Sequence: {' → '.join(improved)}")
    print(f"Improved Health: {improved_health.overall_health:.3f}")
    print(f"Improved Pattern: {improved_health.dominant_pattern}")
    print(f"Health Delta: {improved_health.overall_health - current_health.overall_health:+.3f}")
    
    print("\nImprovements Made:")
    for rec in recommendations:
        print(f"  • {rec}")
    
    # Example 2: Improving an imbalanced sequence
    print("\n\nExample 2: Balancing an Imbalanced Sequence")
    print("-" * 70)
    current = ["dissonance", "mutation", "expansion"]
    print(f"Current Sequence: {' → '.join(current)}")
    
    current_health = analyzer.analyze_health(current)
    print(f"Current Health: {current_health.overall_health:.3f}")
    print(f"Balance Score: {current_health.balance_score:.3f}")
    
    improved, recommendations = generator.improve_sequence(current)
    
    improved_health = analyzer.analyze_health(improved)
    print(f"\nImproved Sequence: {' → '.join(improved)}")
    print(f"Improved Health: {improved_health.overall_health:.3f}")
    print(f"Balance Score: {improved_health.balance_score:.3f}")
    
    print("\nImprovements Made:")
    for rec in recommendations:
        print(f"  • {rec}")


def demo_constraint_handling() -> None:
    """Demonstrate constraint handling in generation."""
    print_separator("CONSTRAINT HANDLING")
    
    generator = ContextualSequenceGenerator(seed=42)
    
    # Example 1: Length constraints
    print("Example 1: Strict Length Constraint")
    print("-" * 70)
    print("Generating with max_length=4...")
    result = generator.generate_for_context(
        domain="therapeutic",
        objective="process_therapy",
        max_length=4,
        min_health=0.65
    )
    print(f"Generated {len(result.sequence)} operators (constraint: ≤4)")
    print_result(result, show_recommendations=False)
    
    # Example 2: High health constraint
    print("\n\nExample 2: High Health Requirement")
    print("-" * 70)
    print("Generating with min_health=0.80...")
    result = generator.generate_for_context(
        domain="educational",
        objective="skill_development",
        min_health=0.80
    )
    print(f"Achieved health: {result.health_score:.3f} (constraint: ≥0.80)")
    print_result(result, show_recommendations=False)


def demo_deterministic_generation() -> None:
    """Demonstrate deterministic generation with seeds."""
    print_separator("DETERMINISTIC GENERATION")
    
    print("Generating with same seed (42) twice...")
    print("-" * 70)
    
    gen1 = ContextualSequenceGenerator(seed=42)
    result1 = gen1.generate_for_context(
        domain="therapeutic",
        objective="crisis_intervention"
    )
    
    gen2 = ContextualSequenceGenerator(seed=42)
    result2 = gen2.generate_for_context(
        domain="therapeutic",
        objective="crisis_intervention"
    )
    
    print(f"Generation 1: {' → '.join(result1.sequence)}")
    print(f"Generation 2: {' → '.join(result2.sequence)}")
    print(f"\nSequences match: {result1.sequence == result2.sequence}")
    print(f"Health scores match: {result1.health_score == result2.health_score}")


def main() -> None:
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TNFR SEQUENCE GENERATOR DEMO" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    
    demos = [
        ("Domain Listing", demo_domain_listing),
        ("Context-Based Generation", demo_context_generation),
        ("Pattern-Targeted Generation", demo_pattern_generation),
        ("Sequence Improvement", demo_sequence_improvement),
        ("Constraint Handling", demo_constraint_handling),
        ("Deterministic Generation", demo_deterministic_generation),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n⚠️  Error in {name}: {e}")
    
    print_separator("DEMO COMPLETE")
    print("For more information, see:")
    print("  - src/tnfr/tools/sequence_generator.py (API documentation)")
    print("  - tools/tnfr_generate (CLI tool)")
    print("  - tests/tools/test_sequence_generator.py (usage examples)")
    print()


if __name__ == "__main__":
    main()
