"""Examples of regenerative cycles in TNFR.

This module demonstrates three types of self-sustaining regenerative cycles
as described in the R5_CICLOS_REGENERATIVOS implementation:

1. **Organizational Cycle**: Institutional evolution with autonomous renewal
2. **Educational Cycle**: Lifelong learning with fractal application
3. **Therapeutic Cycle**: Healing process with sustained regeneration

Each example shows how regenerators (NAV/REMESH/SHA) enable structural
self-sustainability and cyclic coherence.
"""

from tnfr.operators.grammar import (
    validate_sequence,
    parse_sequence,
    REGENERATORS,
    CycleType,
)
from tnfr.operators.cycle_detection import CycleDetector
from tnfr.config.operator_names import (
    COHERENCE,
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


def example_organizational_cycle():
    """Organizational self-sustaining cycle.
    
    Models: Companies in continuous evolution, living organizations.
    Pattern: AL → EN → IL → RA → UM → NAV → IL → SHA
    
    Cycle characteristics:
    - Initiative activation (AL - valid start)
    - Reception of feedback (EN)
    - Operational stability (IL - stabilizer, R2 requirement)
    - Propagation of good practices (RA - stabilizer)
    - Team synchronization (UM - stabilizer)
    - New development phase (NAV - regenerator, transformative)
    - New stability achieved (IL - stabilizer)
    - Closure pause (SHA - regenerator, valid end)
    """
    print("\n" + "="*70)
    print("ORGANIZATIONAL REGENERATIVE CYCLE")
    print("="*70)
    
    org_cycle = [
        EMISSION,      # AL: Initiative activation (R1: valid start)
        RECEPTION,     # EN: Feedback reception
        COHERENCE,     # IL: Operational stability (R2: EN→IL + stabilizer)
        RESONANCE,     # RA: Propagate practices (stabilizer)
        COUPLING,      # UM: Team sync (stabilizer)
        TRANSITION,    # NAV: New phase (regenerator - transformative)
        COHERENCE,     # IL: New stability (stabilizer)
        SILENCE,       # SHA: Closure (regenerator - meditative, R3: valid end)
    ]
    
    print("\nSequence:", " → ".join(org_cycle))
    
    # Validate the cycle
    result = validate_sequence(org_cycle)
    print(f"\n✓ Validation: {'PASSED' if result.passed else 'FAILED'}")
    
    if result.passed:
        automaton = parse_sequence(org_cycle)
        print(f"  Detected pattern: {result.metadata.get("detected_pattern", "unknown")}")
        
        # Analyze cycle structure
        detector = CycleDetector()
        analysis = detector.analyze_full_cycle(org_cycle)
        
        if analysis.is_valid_regenerative:
            print(f"\n✓ R5 Regenerative Cycle Validation: PASSED")
            print(f"  Cycle type: {analysis.cycle_type.value}")
            print(f"  Health score: {analysis.health_score:.3f}")
            print(f"  Balance: {analysis.balance_score:.3f}")
            print(f"  Diversity: {analysis.diversity_score:.3f}")
            print(f"  Coherence: {analysis.coherence_score:.3f}")
            print(f"  Regenerator position: {analysis.regenerator_position} ({org_cycle[analysis.regenerator_position]})")
            print(f"  Stabilizers before: {analysis.stabilizer_count_before}")
            print(f"  Stabilizers after: {analysis.stabilizer_count_after}")
        else:
            print(f"\n✗ R5 Validation failed: {analysis.reason}")
    else:
        print(f"  Error: {result.message}")
    
    print("\nApplication: Organizations as living networks, circular economies")


def example_educational_cycle():
    """Educational regenerative cycle.
    
    Models: Lifelong learning, transformative education.
    Pattern: AL → EN → IL → RA → REMESH → IL → UM → SILENCE
    
    Cycle characteristics:
    - Interest activation (AL - valid start)
    - Student openness/receptivity (EN)
    - Stable comprehension (IL - stabilizer, R2 requirement)
    - Knowledge sharing (RA - stabilizer)
    - Fractal application to other domains (REMESH - regenerator, recursive)
    - Integration of new understanding (IL - stabilizer)
    - Network application (UM - stabilizer)
    - Integration pause (SHA - valid end)
    """
    print("\n" + "="*70)
    print("EDUCATIONAL REGENERATIVE CYCLE")
    print("="*70)
    
    edu_cycle = [
        EMISSION,          # AL: Interest activation (R1: valid start)
        RECEPTION,         # EN: Student openness
        COHERENCE,         # IL: Stable comprehension (R2: EN→IL + stabilizer)
        RESONANCE,         # RA: Knowledge sharing (stabilizer)
        RECURSIVITY,       # REMESH: Fractal application (regenerator - recursive)
        COHERENCE,         # IL: New understanding (stabilizer)
        COUPLING,          # UM: Network application (stabilizer)
        SILENCE,           # SHA: Integration pause (R3: valid end)
    ]
    
    print("\nSequence:", " → ".join(edu_cycle))
    
    result = validate_sequence(edu_cycle)
    print(f"\n✓ Validation: {'PASSED' if result.passed else 'FAILED'}")
    
    if result.passed:
        automaton = parse_sequence(edu_cycle)
        print(f"  Detected pattern: {result.metadata.get("detected_pattern", "unknown")}")
        
        detector = CycleDetector()
        analysis = detector.analyze_full_cycle(edu_cycle)
        
        if analysis.is_valid_regenerative:
            print(f"\n✓ R5 Regenerative Cycle Validation: PASSED")
            print(f"  Cycle type: {analysis.cycle_type.value}")
            print(f"  Health score: {analysis.health_score:.3f}")
            print(f"  Regenerator: REMESH at position {analysis.regenerator_position}")
            print(f"  Stabilizers: {analysis.stabilizer_count_before} before, {analysis.stabilizer_count_after} after")
        else:
            print(f"\n✗ R5 Validation: {analysis.reason}")
    else:
        print(f"  Error: {result.message}")
    
    print("\nApplication: Lifelong learning systems, transformative pedagogy")


def example_therapeutic_cycle():
    """Therapeutic regenerative cycle.
    
    Models: Sustained healing processes, cyclical recovery.
    Pattern: AL → EN → IL → RA → REMESH → UM → IL → SHA
    
    Cycle characteristics:
    - Healing initiation (AL - valid start)
    - Therapy reception (EN)
    - Stabilization (IL - stabilizer, R2 requirement)
    - Wellbeing propagation (RA - stabilizer)
    - Recursive healing (REMESH - regenerator, fractal application)
    - Support network integration (UM - stabilizer)
    - Consolidated stability (IL - stabilizer)
    - Integration pause (SHA - closes cycle, valid end)
    """
    print("\n" + "="*70)
    print("THERAPEUTIC REGENERATIVE CYCLE")
    print("="*70)
    
    healing_cycle = [
        EMISSION,          # AL: Healing initiation (R1: valid start)
        RECEPTION,         # EN: Therapy reception
        COHERENCE,         # IL: Stabilization (R2: EN→IL + stabilizer)
        RESONANCE,         # RA: Wellbeing propagation (stabilizer)
        RECURSIVITY,       # REMESH: Recursive healing (regenerator - fractal application)
        COUPLING,          # UM: Support network (stabilizer)
        COHERENCE,         # IL: Consolidated stability (stabilizer)
        SILENCE,           # SHA: Integration pause (closes cycle, R3: valid end)
    ]
    
    print("\nSequence:", " → ".join(healing_cycle))
    
    result = validate_sequence(healing_cycle)
    print(f"\n✓ Validation: {'PASSED' if result.passed else 'FAILED'}")
    
    if result.passed:
        automaton = parse_sequence(healing_cycle)
        print(f"  Detected pattern: {result.metadata.get("detected_pattern", "unknown")}")
        
        detector = CycleDetector()
        analysis = detector.analyze_full_cycle(healing_cycle)
        
        if analysis.is_valid_regenerative:
            print(f"\n✓ R5 Regenerative Cycle Validation: PASSED")
            print(f"  Cycle type: {analysis.cycle_type.value}")
            print(f"  Health score: {analysis.health_score:.3f}")
            print(f"  Cycle emphasizes: fractal/recursive healing patterns")
            print(f"  Stabilizers: {analysis.stabilizer_count_before} before, {analysis.stabilizer_count_after} after")
        else:
            print(f"\n✗ R5 Validation: {analysis.reason}")
    else:
        print(f"  Error: {result.message}")
    
    print("\nApplication: Sustained healing, cyclical therapy, wellness maintenance")


def show_regenerator_info():
    """Display information about regenerators and cycle types."""
    print("\n" + "="*70)
    print("REGENERATORS AND CYCLE TYPES")
    print("="*70)
    
    print("\nRegenerator operators (enable self-sustaining cycles):")
    for regen in REGENERATORS:
        print(f"  • {regen}")
    
    print("\nCycle types:")
    for cycle_type in CycleType:
        print(f"  • {cycle_type.value}: {cycle_type.name}")
    
    print("\nKey characteristics of regenerative cycles:")
    print("  1. Minimum 5 operators (MIN_CYCLE_LENGTH)")
    print("  2. Must contain at least one regenerator (NAV/REMESH/SHA)")
    print("  3. Stabilizers required before AND after regenerator")
    print("  4. Health score > 0.6 (balance + diversity + coherence)")
    print("  5. Self-sustaining: can repeat without external intervention")


def main():
    """Run all regenerative cycle examples."""
    print("\n" + "#"*70)
    print("# TNFR REGENERATIVE CYCLES EXAMPLES")
    print("# R5_CICLOS_REGENERATIVOS Implementation")
    print("#"*70)
    
    show_regenerator_info()
    example_organizational_cycle()
    example_educational_cycle()
    example_therapeutic_cycle()
    
    print("\n" + "#"*70)
    print("# All examples completed")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
