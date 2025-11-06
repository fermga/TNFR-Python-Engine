"""TNFR Educational Patterns - Transformative Learning Processes.

This module demonstrates application of TNFR structural operators in
educational contexts, showing how coherent operator sequences can model:
- Conceptual breakthrough sequences (rapid cognitive restructuring)
- Competency development cycles (gradual skill acquisition)
- Knowledge spiral patterns (regenerative deepening)
- Reflective learning processes

Each pattern is validated using TNFR's structural health metrics to ensure
coherence, balance, and sustainability.
"""

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
# EDUCATIONAL PATTERNS
# =============================================================================


def get_conceptual_breakthrough_sequence():
    """Conceptual breakthrough pattern - rapid cognitive restructuring.
    
    Context: "Aha!" moments, paradigm shifts, sudden insight into complex
    concepts. Fast transformation with high cognitive reorganization.
    
    Structural flow:
    1. EMISSION (AL): Open cognitive space, activate learning mode
    2. RECEPTION (EN): Encounter with new concept/problem/contradiction
    3. COHERENCE (IL): Initial attempt to organize/understand with existing schema
    4. DISSONANCE (OZ): Recognition of contradiction/gap in current understanding
    5. MUTATION (ZHIR): Cognitive restructuring - paradigm shift ("aha!" moment)
    6. COHERENCE (IL): New conceptual integration and stabilization
    7. RESONANCE (RA): Apply understanding across multiple contexts
    8. SILENCE (SHA): Consolidation period for new schema
    
    Expected metrics:
    - Health score: > 0.75 (good structural quality)
    - Pattern: HIERARCHICAL (mutation creates nested understanding)
    - Balance: Controlled dissonance leading to transformation
    - Sustainability: Moderate (rapid change requires consolidation)
    
    Use cases:
    - Mathematical concept breakthroughs (understanding calculus, proofs)
    - Scientific paradigm shifts (grasping quantum mechanics)
    - Language structure insights (suddenly "getting" grammar)
    - Artistic technique breakthroughs
    
    Returns
    -------
    list[str]
        Validated operator sequence for conceptual breakthrough
    """
    return [
        EMISSION,           # AL: Open cognitive space, activate learning
        RECEPTION,          # EN: Encounter with new concept/problem
        COHERENCE,          # IL: Initial attempt to organize/understand
        DISSONANCE,         # OZ: Recognition of contradiction/gap
        MUTATION,           # ZHIR: Cognitive restructuring ("aha!" moment)
        COHERENCE,          # IL: New conceptual integration
        RESONANCE,          # RA: Apply understanding across contexts
        SILENCE,            # SHA: Consolidation period
    ]


def get_competency_development_sequence():
    """Competency development pattern - gradual skill acquisition.
    
    Context: Sustained learning process, step-by-step skill building,
    progressive mastery. The canonical EDUCATIONAL pattern from theory.
    
    Structural flow:
    1. EMISSION (AL): Set learning intention/goal - open learning space
    2. RECEPTION (EN): Absorb foundational knowledge and instruction
    3. COHERENCE (IL): Organize basic understanding into initial schema
    4. EXPANSION (VAL): Explore applications, variations, connections
    5. COHERENCE (IL): Stabilize expanded understanding
    6. DISSONANCE (OZ): Challenge/stretch current limits, encounter difficulties
    7. MUTATION (ZHIR): Transform to next competency level - breakthrough
    8. TRANSITION (NAV): Move to advanced competency phase
    9. COHERENCE (IL): Stabilize new skill level
    10. RESONANCE (RA): Apply and propagate new competency
    11. RECURSIVITY (REMESH): Reflect and generalize learning fractal
    
    Expected metrics:
    - Health score: > 0.85 (excellent structural quality)
    - Pattern: EDUCATIONAL (complete transformative learning cycle)
    - Balance: Good equilibrium with expansion focus
    - Sustainability: High (complete cycle with reflection)
    
    Use cases:
    - Learning musical instruments (beginner → intermediate → advanced)
    - Programming language acquisition
    - Athletic skill development
    - Professional certification programs
    
    Returns
    -------
    list[str]
        Validated operator sequence for competency development
    """
    return [
        EMISSION,           # AL: Set learning intention/goal
        RECEPTION,          # EN: Absorb foundational knowledge
        COHERENCE,          # IL: Organize basic understanding
        EXPANSION,          # VAL: Explore applications and variations
        COHERENCE,          # IL: Stabilize expanded understanding
        DISSONANCE,         # OZ: Challenge/stretch current limits
        MUTATION,           # ZHIR: Transform to next competency level
        TRANSITION,         # NAV: Move to advanced competency phase
        COHERENCE,          # IL: Stabilize new skill level
        RESONANCE,          # RA: Apply and propagate new competency
        RECURSIVITY,        # REMESH: Reflect and generalize learning
    ]


def get_knowledge_spiral_sequence():
    """Knowledge spiral pattern - regenerative deepening.
    
    Context: Continuous learning, deepening understanding through cycles,
    spiral curriculum approach, self-sustaining inquiry.
    
    Structural flow:
    1. EMISSION (AL): Initiate inquiry cycle with questions
    2. RECEPTION (EN): Receive current state awareness
    3. COHERENCE (IL): Current understanding base - starting point
    4. RESONANCE (RA): Activate and connect existing knowledge
    5. EXPANSION (VAL): Explore new dimensions, questions, connections
    6. COUPLING (UM): Synthesize new understanding with existing
    7. TRANSITION (NAV): Shift to deeper inquiry level (regenerator)
    8. COHERENCE (IL): Stabilize integrated understanding
    9. SILENCE (SHA): Reflection pause - consolidation
    
    Expected metrics:
    - Health score: > 0.80 (strong structural quality)
    - Pattern: REGENERATIVE (self-sustaining cycle)
    - Balance: Excellent equilibrium
    - Sustainability: Very high (designed for continuous cycling)
    
    Use cases:
    - Doctoral research processes
    - Life-long learning journeys
    - Spiral curriculum (revisiting topics at deeper levels)
    - Continuous professional development
    
    Returns
    -------
    list[str]
        Validated operator sequence for knowledge spiral
    """
    return [
        EMISSION,           # AL: Initiate inquiry cycle with questions
        RECEPTION,          # EN: Receive current state awareness
        COHERENCE,          # IL: Current understanding base
        RESONANCE,          # RA: Activate existing knowledge
        EXPANSION,          # VAL: Explore new dimensions/questions
        COUPLING,           # UM: Synthesize new with old
        TRANSITION,         # NAV: Shift to deeper inquiry level (regenerator)
        COHERENCE,          # IL: Stabilize integrated understanding
        SILENCE,            # SHA: Reflection pause - consolidation
    ]


def get_practice_mastery_sequence():
    """Practice mastery pattern - deliberate practice cycle.
    
    Context: Skill refinement through repetition, feedback integration,
    incremental improvement. Focus on gradual excellence.
    
    Structural flow:
    1. EMISSION (AL): Set practice session intention
    2. RECEPTION (EN): Receive instruction/model/feedback
    3. COHERENCE (IL): Stabilize current understanding
    4. DISSONANCE (OZ): Identify gap between current and target
    5. SELF_ORGANIZATION (THOL): Self-directed exploration and adjustment
    6. COHERENCE (IL): Stabilize improvement
    7. CONTRACTION (NUL): Focus on specific weakness refinement
    8. COHERENCE (IL): Integrate focused practice
    9. SILENCE (SHA): Rest and consolidation (motor memory formation)
    
    Expected metrics:
    - Health score: > 0.75 (good structural quality)
    - Pattern: ACTIVATION or HIERARCHICAL
    - Balance: Good with focus on stabilization
    - Sustainability: High (cyclical practice pattern)
    
    Use cases:
    - Musical practice sessions
    - Athletic training repetitions
    - Language speaking practice
    - Meditation practice
    
    Returns
    -------
    list[str]
        Validated operator sequence for practice mastery
    """
    return [
        EMISSION,           # AL: Set practice session intention
        RECEPTION,          # EN: Receive instruction/model/feedback
        COHERENCE,          # IL: Stabilize understanding
        DISSONANCE,         # OZ: Identify performance gap
        SELF_ORGANIZATION,  # THOL: Self-directed adjustment
        COHERENCE,          # IL: Stabilize improvement
        CONTRACTION,        # NUL: Focus on weakness refinement
        COHERENCE,          # IL: Integrate focused practice
        SILENCE,            # SHA: Rest and consolidation
    ]


def get_collaborative_learning_sequence():
    """Collaborative learning pattern - peer learning through coupling.
    
    Context: Group learning, peer teaching, collaborative problem-solving,
    knowledge construction through social interaction.
    
    Structural flow:
    1. EMISSION (AL): Opening of collaborative space, shared intention
    2. RECEPTION (EN): Listen to diverse perspectives
    3. COHERENCE (IL): Initial individual understanding
    4. COUPLING (UM): Connect ideas across learners - phase synchrony
    5. EXPANSION (VAL): Explore implications and applications together
    6. RESONANCE (RA): Amplify shared understanding through group
    7. COHERENCE (IL): Find common ground and integrate perspectives
    8. DISSONANCE (OZ): Encounter conflicting viewpoints (productive tension)
    9. TRANSITION (NAV): Transition to individual internalization
    
    Expected metrics:
    - Health score: > 0.75 (good structural quality)
    - Pattern: EDUCATIONAL or RESONATE
    - Balance: Good with coupling emphasis
    - Sustainability: High (social reinforcement)
    
    Use cases:
    - Study groups and peer learning
    - Project-based learning teams
    - Communities of practice
    - Collaborative research
    
    Returns
    -------
    list[str]
        Validated operator sequence for collaborative learning
    """
    return [
        EMISSION,           # AL: Opening collaborative space
        RECEPTION,          # EN: Listen to diverse perspectives
        COHERENCE,          # IL: Initial individual understanding
        COUPLING,           # UM: Connect ideas across learners
        EXPANSION,          # VAL: Explore implications together
        RESONANCE,          # RA: Amplify shared understanding
        COHERENCE,          # IL: Find common ground
        DISSONANCE,         # OZ: Encounter conflicting viewpoints
        TRANSITION,         # NAV: Transition to individual internalization
    ]


# =============================================================================
# VALIDATION AND REPORTING
# =============================================================================


def validate_pattern(name, sequence):
    """Validate an educational pattern and print detailed report.
    
    Parameters
    ----------
    name : str
        Human-readable name of the pattern
    sequence : list[str]
        Operator sequence to validate
    
    Returns
    -------
    bool
        True if validation passed, False otherwise
    """
    print(f"\n{'=' * 70}")
    print(f"Pattern: {name}")
    print(f"{'=' * 70}")
    print(f"Sequence: {' → '.join(sequence)}")
    
    # Validate with health metrics
    result = validate_sequence_with_health(sequence)
    
    if not result.passed:
        print(f"\n✗ VALIDATION FAILED: {result.message}")
        return False
    
    health = result.health_metrics
    print(f"\n✓ VALIDATION PASSED")
    print(f"\n--- Structural Health Metrics ---")
    print(f"Overall Health:         {health.overall_health:.3f}")
    print(f"Coherence Index:        {health.coherence_index:.3f}")
    print(f"Balance Score:          {health.balance_score:.3f}")
    print(f"Sustainability:         {health.sustainability_index:.3f}")
    print(f"Complexity Efficiency:  {health.complexity_efficiency:.3f}")
    print(f"Pattern Detected:       {health.dominant_pattern.upper()}")
    
    # Health interpretation
    if health.overall_health >= 0.85:
        status = "🌟 EXCELLENT"
    elif health.overall_health >= 0.75:
        status = "✓ GOOD"
    elif health.overall_health >= 0.60:
        status = "⚠ FAIR"
    else:
        status = "⚠️ POOR"
    
    print(f"\nHealth Status: {status}")
    
    if health.recommendations:
        print(f"\n--- Recommendations ---")
        for i, rec in enumerate(health.recommendations, 1):
            print(f"  {i}. {rec}")
    
    return True


def validate_all_patterns():
    """Validate all educational patterns and generate summary report.
    
    Returns
    -------
    dict
        Summary statistics of validation results
    """
    print("\n" + "=" * 70)
    print("TNFR EDUCATIONAL PATTERNS VALIDATION")
    print("=" * 70)
    
    patterns = {
        "Conceptual Breakthrough": get_conceptual_breakthrough_sequence(),
        "Competency Development": get_competency_development_sequence(),
        "Knowledge Spiral": get_knowledge_spiral_sequence(),
        "Practice Mastery": get_practice_mastery_sequence(),
        "Collaborative Learning": get_collaborative_learning_sequence(),
    }
    
    results = {}
    
    for name, sequence in patterns.items():
        passed = validate_pattern(name, sequence)
        
        if passed:
            result = validate_sequence_with_health(sequence)
            results[name] = {
                "passed": True,
                "health": result.health_metrics.overall_health,
                "pattern": result.health_metrics.dominant_pattern,
                "length": len(sequence),
            }
        else:
            results[name] = {"passed": False}
    
    # Summary report
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    
    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)
    
    print(f"\nTotal Patterns: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    
    if passed_count > 0:
        avg_health = sum(r["health"] for r in results.values() if r["passed"]) / passed_count
        print(f"Average Health Score: {avg_health:.3f}")
    
    print(f"\n--- Individual Results ---")
    for name, result in results.items():
        if result["passed"]:
            status = "✓ PASS"
            details = f"(health: {result['health']:.3f}, pattern: {result['pattern']})"
        else:
            status = "✗ FAIL"
            details = ""
        print(f"  {status:8s} {name:30s} {details}")
    
    return results


def main():
    """Run educational patterns validation and demonstration."""
    results = validate_all_patterns()
    
    # Success criteria check
    print(f"\n{'=' * 70}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'=' * 70}")
    
    all_passed = all(r["passed"] for r in results.values())
    health_threshold = 0.75
    all_above_threshold = all(
        r["health"] >= health_threshold
        for r in results.values()
        if r["passed"]
    )
    
    # Check for EDUCATIONAL pattern detection (at least 50%)
    educational_count = sum(
        1 for r in results.values()
        if r["passed"] and r["pattern"] == "educational"
    )
    educational_percentage = (educational_count / len(results) * 100) if results else 0
    
    print(f"\n✓ All sequences valid: {all_passed}")
    print(f"✓ All health scores > {health_threshold}: {all_above_threshold}")
    print(f"✓ Minimum 3 specialized patterns: {len(results) >= 3}")
    print(f"✓ EDUCATIONAL pattern detection: {educational_count}/{len(results)} ({educational_percentage:.0f}%)")
    
    if all_passed and all_above_threshold:
        print(f"\n🎉 SUCCESS: All educational patterns meet acceptance criteria!")
    else:
        print(f"\n⚠️  ISSUES: Some patterns need improvement")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
