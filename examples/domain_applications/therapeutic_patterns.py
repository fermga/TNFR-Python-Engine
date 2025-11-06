"""TNFR Therapeutic Patterns - Healing and Personal Transformation.

This module demonstrates application of TNFR structural operators in
therapeutic contexts, showing how coherent operator sequences can model:
- Crisis intervention and stabilization
- Gradual healing and transformation processes
- Integration and consolidation cycles
- Preventive stabilization patterns

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
# THERAPEUTIC PATTERNS
# =============================================================================


def get_crisis_intervention_sequence():
    """Crisis intervention pattern - rapid stabilization, high effectiveness.
    
    Context: Immediate crisis response, emergency containment, acute distress.
    The sequence prioritizes rapid stabilization over deep transformation.
    
    Structural flow:
    1. EMISSION (AL): Create immediate safe container, establish presence
    2. RECEPTION (EN): Receive/validate the crisis without judgment
    3. COHERENCE (IL): Emergency stabilization, grounding
    4. DISSONANCE (OZ): Acknowledge the crisis tension (brief, contained)
    5. CONTRACTION (NUL): Focus on immediate coping essentials
    6. COHERENCE (IL): Deeper stabilization
    7. COUPLING (UM): Connect to immediate support/resources
    8. SILENCE (SHA): Reparative pause, minimal integration
    
    Expected metrics:
    - Health score: > 0.70 (good structural quality)
    - Pattern: STABILIZE (coherence + silence ending)
    - Balance: Controlled dissonance with strong stabilizers
    - Sustainability: Moderate (designed for short-term intervention)
    
    Use cases:
    - Panic attack management
    - Acute grief response
    - Immediate post-trauma stabilization
    - Emergency emotional support
    
    Returns
    -------
    list[str]
        Validated operator sequence for crisis intervention
    """
    return [
        EMISSION,        # AL: Immediate safe container
        RECEPTION,       # EN: Non-judgmental reception of crisis
        COHERENCE,       # IL: Emergency stabilization
        DISSONANCE,      # OZ: Acknowledge crisis tension (brief)
        CONTRACTION,     # NUL: Focus on coping essentials
        COHERENCE,       # IL: Deeper stabilization
        COUPLING,        # UM: Connect to immediate support
        SILENCE,         # SHA: Reparative pause
    ]


def get_process_therapy_sequence():
    """Process therapy pattern - complete transformative cycle.
    
    Context: Full therapeutic process, deep personal transformation,
    structured healing journey with controlled crisis exploration.
    
    Structural flow:
    1. EMISSION (AL): Opening of therapeutic process, intention setting
    2. RECEPTION (EN): Deep empathic reception, active listening
    3. COHERENCE (IL): Establish therapeutic alliance, baseline stability
    4. DISSONANCE (OZ): Explore central tension/conflict (controlled)
    5. SELF_ORGANIZATION (THOL): Facilitate autonomous reorganization
    6. COHERENCE (IL): Integrate new insights/resources
    7. TRANSITION (NAV): Transition to new life phase
    8. SILENCE (SHA): Final consolidation and integration
    
    Expected metrics:
    - Health score: > 0.85 (excellent structural quality)
    - Pattern: THERAPEUTIC (full healing cycle detected)
    - Balance: Equilibrium between destabilizers and stabilizers
    - Sustainability: High (complete cycle with closure)
    
    Use cases:
    - Multi-session psychotherapy
    - Personal transformation programs
    - Structured healing protocols
    - Developmental transitions
    
    Returns
    -------
    list[str]
        Validated operator sequence for process therapy
    """
    return [
        EMISSION,           # AL: Therapeutic process opening
        RECEPTION,          # EN: Deep empathic reception
        COHERENCE,          # IL: Therapeutic alliance establishment
        DISSONANCE,         # OZ: Explore central conflict (controlled)
        SELF_ORGANIZATION,  # THOL: Facilitate autonomous reorganization
        COHERENCE,          # IL: Integrate new insights
        TRANSITION,         # NAV: Transition to new phase
        SILENCE,            # SHA: Final consolidation
    ]


def get_regenerative_healing_sequence():
    """Regenerative healing pattern - cyclic, self-sustainable.
    
    Context: Long-term wellness maintenance, resilience building,
    self-sustaining healing capacity that cycles autonomously.
    
    Structural flow:
    1. EMISSION (AL): Initiate wellness cycle with intention
    2. RECEPTION (EN): Receive current state awareness
    3. COHERENCE (IL): Base of personal stability
    4. RESONANCE (RA): Connect with internal resources
    5. EXPANSION (VAL): Explore growth potential
    6. COHERENCE (IL): Stabilize expanded awareness
    7. COUPLING (UM): Internal synchronization
    8. TRANSITION (NAV): New development phase (regenerator, valid end)
    
    Expected metrics:
    - Health score: > 0.80 (strong structural quality)
    - Pattern: REGENERATIVE (self-sustaining cycle)
    - Balance: Good equilibrium with expansion focus
    - Sustainability: Very high (designed for long-term cycling)
    
    Use cases:
    - Wellness maintenance programs
    - Resilience building
    - Post-therapy integration
    - Preventive mental health care
    
    Returns
    -------
    list[str]
        Validated operator sequence for regenerative healing
    """
    return [
        EMISSION,           # AL: Initiate wellness cycle
        RECEPTION,          # EN: Receive current state awareness
        COHERENCE,          # IL: Base stability
        RESONANCE,          # RA: Connect internal resources
        EXPANSION,          # VAL: Explore growth potential
        COHERENCE,          # IL: Stabilize expanded awareness
        COUPLING,           # UM: Internal synchronization
        TRANSITION,         # NAV: New development phase (valid end)
    ]


def get_insight_integration_sequence():
    """Insight integration pattern - consolidate therapeutic breakthroughs.
    
    Context: Post-insight integration, consolidating therapeutic gains,
    stabilizing new understanding into daily life.
    
    Structural flow:
    1. EMISSION (AL): Acknowledge and activate the new insight
    2. RECEPTION (EN): Receive and deeply understand the insight
    3. COHERENCE (IL): Initial stabilization of new understanding
    4. DISSONANCE (OZ): Explore tensions with old patterns
    5. SELF_ORGANIZATION (THOL): Allow new integration to self-organize
    6. RESONANCE (RA): Amplify the insight's significance
    7. COUPLING (UM): Connect insight to life contexts
    8. COHERENCE (IL): Deeper integration
    9. SILENCE (SHA): Consolidation pause
    
    Expected metrics:
    - Health score: > 0.75 (good to excellent)
    - Pattern: THERAPEUTIC (transformation with insight)
    - Balance: Equilibrium with controlled transformation
    - Sustainability: High (complete stabilization)
    
    Use cases:
    - Post-session integration
    - Breakthrough consolidation
    - Insight-to-action translation
    - Therapeutic homework
    
    Returns
    -------
    list[str]
        Validated operator sequence for insight integration
    """
    return [
        EMISSION,           # AL: Acknowledge and activate insight
        RECEPTION,          # EN: Receive and understand insight
        COHERENCE,          # IL: Initial stabilization
        DISSONANCE,         # OZ: Explore tensions with old patterns
        SELF_ORGANIZATION,  # THOL: Allow new integration
        RESONANCE,          # RA: Amplify significance
        COUPLING,           # UM: Connect to life contexts
        COHERENCE,          # IL: Deeper integration
        SILENCE,            # SHA: Consolidation pause
    ]


def get_relapse_prevention_sequence():
    """Relapse prevention pattern - maintain gains and prevent regression.
    
    Context: Addiction recovery, behavioral change maintenance,
    preventing return to previous dysfunctional patterns.
    
    Structural flow:
    1. EMISSION (AL): Activate awareness and vigilance
    2. RECEPTION (EN): Receive and monitor current state
    3. COHERENCE (IL): Current stability baseline
    4. DISSONANCE (OZ): Identify warning signs (controlled exposure)
    5. CONTRACTION (NUL): Simplify to core coping strategies
    6. COHERENCE (IL): Reestablish stability
    7. RESONANCE (RA): Reinforce healthy patterns
    8. COUPLING (UM): Connect to support networks
    9. SILENCE (SHA): Monitoring pause
    
    Expected metrics:
    - Health score: > 0.75 (good structural quality)
    - Pattern: STABILIZE or THERAPEUTIC
    - Balance: Careful dissonance control with strong stabilizers
    - Sustainability: High (designed for ongoing practice)
    
    Use cases:
    - Addiction recovery maintenance
    - Behavioral change consolidation
    - Trigger management protocols
    - Ongoing wellness monitoring
    
    Returns
    -------
    list[str]
        Validated operator sequence for relapse prevention
    """
    return [
        EMISSION,           # AL: Activate awareness and vigilance
        RECEPTION,          # EN: Receive and monitor state
        COHERENCE,          # IL: Current stability baseline
        DISSONANCE,         # OZ: Identify warning signs
        CONTRACTION,        # NUL: Simplify to core strategies
        COHERENCE,          # IL: Reestablish stability
        RESONANCE,          # RA: Reinforce healthy patterns
        COUPLING,           # UM: Connect to support networks
        SILENCE,            # SHA: Monitoring pause
    ]


# =============================================================================
# VALIDATION AND REPORTING
# =============================================================================


def validate_pattern(name, sequence):
    """Validate a therapeutic pattern and print detailed report.
    
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
    """Validate all therapeutic patterns and generate summary report.
    
    Returns
    -------
    dict
        Summary statistics of validation results
    """
    print("\n" + "=" * 70)
    print("TNFR THERAPEUTIC PATTERNS VALIDATION")
    print("=" * 70)
    
    patterns = {
        "Crisis Intervention": get_crisis_intervention_sequence(),
        "Process Therapy": get_process_therapy_sequence(),
        "Regenerative Healing": get_regenerative_healing_sequence(),
        "Insight Integration": get_insight_integration_sequence(),
        "Relapse Prevention": get_relapse_prevention_sequence(),
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
    """Run therapeutic patterns validation and demonstration."""
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
    
    print(f"\n✓ All sequences valid: {all_passed}")
    print(f"✓ All health scores > {health_threshold}: {all_above_threshold}")
    print(f"✓ Minimum 5 specialized patterns: {len(results) >= 5}")
    
    if all_passed and all_above_threshold:
        print(f"\n🎉 SUCCESS: All therapeutic patterns meet acceptance criteria!")
    else:
        print(f"\n⚠️  ISSUES: Some patterns need improvement")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
