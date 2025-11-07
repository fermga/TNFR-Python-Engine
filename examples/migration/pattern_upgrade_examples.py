"""Pattern upgrade examples showing evolution from basic to specialized patterns.

This example demonstrates how to upgrade sequences from fundamental patterns
to more specialized domain-specific patterns in Grammar 2.0.
"""

from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.patterns import AdvancedPatternDetector, StructuralPattern
from tnfr.config.operator_names import (
    EMISSION, RECEPTION, COHERENCE, DISSONANCE, SILENCE,
    COUPLING, RESONANCE, SELF_ORGANIZATION, TRANSITION,
    MUTATION, EXPANSION, CONTRACTION, RECURSIVITY
)


def show_pattern_evolution(stages: list, title: str):
    """Show evolution of a sequence through pattern stages."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    
    detector = AdvancedPatternDetector()
    
    for i, (stage_name, sequence) in enumerate(stages, 1):
        print(f"\n--- Stage {i}: {stage_name} ---")
        print(f"Sequence: {' → '.join(sequence)}")
        
        result = validate_sequence_with_health(sequence)
        
        if result.passed:
            pattern = result.metadata.get('detected_pattern', 'unknown')
            health = result.health_metrics
            
            print(f"Pattern: {pattern.upper()}")
            print(f"Health: {health.overall_health:.2f}")
            print(f"  Coherence: {health.coherence_index:.2f} | Balance: {health.balance_score:.2f}")
            print(f"  Frequency: {health.frequency_harmony:.2f} | Sustainability: {health.sustainability_index:.2f}")


def example_1_minimal_to_hierarchical():
    """Upgrade from MINIMAL to HIERARCHICAL pattern."""
    stages = [
        ("MINIMAL - Basic activation", 
         [EMISSION, COHERENCE]),
        
        ("LINEAR - Add reception", 
         [EMISSION, RECEPTION, COHERENCE]),
        
        ("HIERARCHICAL - Add network layer",
         [EMISSION, RECEPTION, COUPLING, COHERENCE, RESONANCE]),
    ]
    
    show_pattern_evolution(stages, "Evolution: MINIMAL → LINEAR → HIERARCHICAL")
    
    print("\n📝 Evolution Path:")
    print("   MINIMAL: Simplest valid sequence")
    print("   → LINEAR: Add information capture (RECEPTION)")
    print("   → HIERARCHICAL: Add network propagation (COUPLING, RESONANCE)")


def example_2_linear_to_therapeutic():
    """Upgrade from LINEAR to THERAPEUTIC pattern."""
    stages = [
        ("LINEAR - Basic flow",
         [EMISSION, RECEPTION, COHERENCE]),
        
        ("Add destabilization",
         [EMISSION, RECEPTION, COHERENCE, DISSONANCE]),
        
        ("THERAPEUTIC - Add transformation",
         [RECEPTION, EMISSION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE]),
    ]
    
    show_pattern_evolution(stages, "Evolution: LINEAR → THERAPEUTIC")
    
    print("\n📝 Evolution Path:")
    print("   LINEAR: Basic information flow")
    print("   → Add DISSONANCE: Introduce controlled instability")
    print("   → THERAPEUTIC: Self-organization for transformation and healing")
    print("\n💡 Use Case: Therapeutic interventions, healing processes, recovery")


def example_3_stabilize_to_educational():
    """Upgrade from STABILIZE to EDUCATIONAL pattern."""
    stages = [
        ("STABILIZE - Simple consolidation",
         [COHERENCE, SILENCE]),
        
        ("Add activation",
         [EMISSION, RECEPTION, COHERENCE, SILENCE]),
        
        ("EDUCATIONAL - Add reinforcement cycles",
         [
             EMISSION, RECEPTION, COHERENCE,           # Initial learning
             TRANSITION, COHERENCE,                    # Consolidation
             RESONANCE, SILENCE                        # Reinforcement
         ]),
    ]
    
    show_pattern_evolution(stages, "Evolution: STABILIZE → EDUCATIONAL")
    
    print("\n📝 Evolution Path:")
    print("   STABILIZE: Basic pause and hold")
    print("   → Add activation: Include learning initiation")
    print("   → EDUCATIONAL: Reinforcement cycles for knowledge retention")
    print("\n💡 Use Case: Learning systems, knowledge transfer, skill development")


def example_4_hierarchical_to_organizational():
    """Upgrade from HIERARCHICAL to ORGANIZATIONAL pattern."""
    stages = [
        ("HIERARCHICAL - Basic structure",
         [EMISSION, COUPLING, COHERENCE, RESONANCE]),
        
        ("Add self-organization",
         [EMISSION, COUPLING, DISSONANCE, SELF_ORGANIZATION, COHERENCE]),
        
        ("ORGANIZATIONAL - Full coordination",
         [
             EMISSION, COUPLING, COHERENCE,            # Form network
             TRANSITION, DISSONANCE,                   # Reorganization trigger
             SELF_ORGANIZATION, COHERENCE,             # Emergent structure
             RESONANCE                                 # Propagate coordination
         ]),
    ]
    
    show_pattern_evolution(stages, "Evolution: HIERARCHICAL → ORGANIZATIONAL")
    
    print("\n📝 Evolution Path:")
    print("   HIERARCHICAL: Basic network structure")
    print("   → Add transformation: Self-organization capability")
    print("   → ORGANIZATIONAL: Coordinated emergence and propagation")
    print("\n💡 Use Case: Team coordination, organizational development, collective action")


def example_5_cyclic_to_regenerative():
    """Upgrade from CYCLIC to REGENERATIVE pattern."""
    stages = [
        ("CYCLIC - Simple repetition",
         [EMISSION, COHERENCE, EMISSION, COHERENCE]),
        
        ("Add regenerator",
         [COHERENCE, TRANSITION, EMISSION, COHERENCE, SILENCE]),
        
        ("REGENERATIVE - Self-sustaining",
         [
             COHERENCE, SILENCE,                       # Stability
             TRANSITION, EMISSION,                     # Regeneration
             RECEPTION, COHERENCE,                     # Capture
             COUPLING, RESONANCE,                      # Propagate
             SILENCE                                   # Return to stability
         ]),
    ]
    
    show_pattern_evolution(stages, "Evolution: CYCLIC → REGENERATIVE")
    
    print("\n📝 Evolution Path:")
    print("   CYCLIC: Basic repetition")
    print("   → Add TRANSITION: Enable regeneration")
    print("   → REGENERATIVE: Complete self-sustaining cycle")
    print("\n💡 Use Case: Sustainable systems, renewable processes, continuous improvement")


def example_6_creative_pattern_construction():
    """Build CREATIVE pattern from scratch."""
    stages = [
        ("Foundation - Exploration start",
         [RECEPTION, EMISSION]),
        
        ("Add variation",
         [RECEPTION, EMISSION, DISSONANCE, MUTATION]),
        
        ("CREATIVE - Full divergence/convergence",
         [
             RECEPTION, EMISSION,                      # Inspiration
             DISSONANCE, EXPANSION,                    # Exploration
             MUTATION, SELF_ORGANIZATION,              # Novel emergence
             COHERENCE, COUPLING,                      # Integration
             RESONANCE                                 # Propagation
         ]),
    ]
    
    show_pattern_evolution(stages, "Construction: CREATIVE Pattern")
    
    print("\n📝 Construction Path:")
    print("   Foundation: Capture inspiration (RECEPTION, EMISSION)")
    print("   → Explore: Variation and expansion (DISSONANCE, EXPANSION)")
    print("   → Create: Novel emergence (MUTATION, SELF_ORGANIZATION)")
    print("   → Share: Integration and propagation (COHERENCE, COUPLING, RESONANCE)")
    print("\n💡 Use Case: Creative processes, innovation, ideation sessions")


def pattern_comparison_matrix():
    """Compare all pattern types side by side."""
    print("\n" + "=" * 80)
    print("  PATTERN COMPARISON MATRIX")
    print("=" * 80)
    
    patterns_to_compare = {
        "MINIMAL": [EMISSION, COHERENCE],
        "LINEAR": [EMISSION, RECEPTION, COHERENCE],
        "HIERARCHICAL": [EMISSION, COUPLING, COHERENCE, RESONANCE],
        "THERAPEUTIC": [RECEPTION, EMISSION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE],
        "EDUCATIONAL": [EMISSION, RECEPTION, COHERENCE, TRANSITION, COHERENCE, RESONANCE, SILENCE],
        "ORGANIZATIONAL": [EMISSION, COUPLING, COHERENCE, TRANSITION, DISSONANCE, SELF_ORGANIZATION, COHERENCE, RESONANCE],
        "CREATIVE": [RECEPTION, EMISSION, DISSONANCE, EXPANSION, MUTATION, SELF_ORGANIZATION, COHERENCE, COUPLING, RESONANCE],
        "REGENERATIVE": [COHERENCE, SILENCE, TRANSITION, EMISSION, RECEPTION, COHERENCE, COUPLING, RESONANCE, SILENCE],
    }
    
    print(f"\n{'Pattern':<18} {'Length':<8} {'Health':<8} {'Coherence':<11} {'Balance':<9} {'Frequency':<11}")
    print("-" * 80)
    
    for pattern_name, sequence in patterns_to_compare.items():
        result = validate_sequence_with_health(sequence)
        if result.passed:
            h = result.health_metrics
            print(f"{pattern_name:<18} {len(sequence):<8} {h.overall_health:>6.2f}   "
                  f"{h.coherence_index:>6.2f}      {h.balance_score:>6.2f}    {h.frequency_harmony:>6.2f}")


def upgrade_recommendations():
    """Provide upgrade recommendations for common scenarios."""
    print("\n" + "=" * 80)
    print("  PATTERN UPGRADE RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n🎯 When to upgrade patterns:")
    
    print("\nMINIMAL → LINEAR:")
    print("  • When you need information capture")
    print("  • Add RECEPTION operator")
    
    print("\nLINEAR → HIERARCHICAL:")
    print("  • When working with network structures")
    print("  • Add COUPLING and RESONANCE")
    
    print("\nLINEAR → THERAPEUTIC:")
    print("  • When transformation is needed")
    print("  • Add DISSONANCE → SELF_ORGANIZATION → COHERENCE")
    
    print("\nHIERARCHICAL → ORGANIZATIONAL:")
    print("  • For coordinated team/group dynamics")
    print("  • Add TRANSITION, transformation operators")
    
    print("\nAny → REGENERATIVE:")
    print("  • For self-sustaining processes")
    print("  • Ensure: TRANSITION/RECURSIVITY, 5+ operators, balanced structure")
    
    print("\nCREATIVE pattern:")
    print("  • For innovation and exploration")
    print("  • Include: variation (DISSONANCE/MUTATION), expansion, emergence")


def main():
    """Run all pattern upgrade examples."""
    print("\n" + "█" * 80)
    print("  GRAMMAR 2.0: PATTERN UPGRADE EXAMPLES")
    print("█" * 80)
    
    example_1_minimal_to_hierarchical()
    example_2_linear_to_therapeutic()
    example_3_stabilize_to_educational()
    example_4_hierarchical_to_organizational()
    example_5_cyclic_to_regenerative()
    example_6_creative_pattern_construction()
    
    pattern_comparison_matrix()
    upgrade_recommendations()
    
    print("\n" + "=" * 80)
    print("  📚 Key Lessons:")
    print("=" * 80)
    print("  • Patterns evolve from fundamental to specialized")
    print("  • Each upgrade adds specific capabilities")
    print("  • Domain patterns optimize for specific use cases")
    print("  • Health metrics guide optimization choices")
    print("\n  Next Steps:")
    print("  1. Identify your use case and target pattern")
    print("  2. Build incrementally from simpler patterns")
    print("  3. Validate health at each stage")
    print("  4. Review docs/PATTERN_REFERENCE.md for complete catalog")
    print("=" * 80)


if __name__ == "__main__":
    main()
