"""TNFR Therapeutic Case Studies - Clinical Application Examples.

This module presents detailed clinical case studies demonstrating how
TNFR structural patterns apply to real therapeutic scenarios:

1. Trauma recovery - controlled desestabilization and reorganization
2. Addiction healing - expansion to contraction pattern
3. Depression emergence - reactivation from silence
4. Relationship repair - coupling through crisis

Each case includes:
- Clinical context and presenting problem
- TNFR structural interpretation
- Operator sequence selection rationale
- Expected outcomes and metrics
- Therapeutic considerations
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
# CASE STUDY 1: TRAUMA RECOVERY
# =============================================================================


def case_trauma_recovery():
    """Case Study: Trauma recovery through controlled crisis resolution.
    
    Clinical Context
    ----------------
    Patient: 32-year-old presenting with PTSD symptoms following
    a traumatic event 6 months prior. Symptoms include intrusive
    memories, hypervigilance, and avoidance behaviors. Patient has
    baseline stability but limited access to traumatic material.
    
    Therapeutic Goal
    ----------------
    Controlled exposure to traumatic memory, facilitate processing and
    reorganization, integrate new understanding. Pattern: OZ → THOL → IL
    (dissonance → self-organization → coherence).
    
    TNFR Structural Interpretation
    -------------------------------
    Trauma represents a "frozen" dissonant structure (high ΔNFR) that
    hasn't integrated. The pattern creates:
    1. Safe therapeutic container (AL, EN, IL)
    2. Controlled dissonance activation (OZ - graded exposure)
    3. Autonomous reorganization (THOL - patient's own integration)
    4. New coherent structure (IL - integrated memory)
    5. Stabilization and consolidation (RA, UM, SHA)
    
    Structural Sequence
    -------------------
    emission → reception → coherence → dissonance → self_organization
    → coherence → resonance → coupling → silence
    
    Expected Outcomes
    -----------------
    - Reduced intrusive symptoms
    - Integrated traumatic memory (no longer "frozen")
    - Increased sense of coherence and control
    - Health score: > 0.80 (therapeutic pattern)
    
    Therapeutic Considerations
    ---------------------------
    - OZ (dissonance) must be carefully titrated - not overwhelming
    - THOL (self-organization) respects patient's internal timing
    - Multiple cycles may be needed for complex trauma
    - Phase (φ) with therapist critical for safety
    """
    sequence = [
        EMISSION,           # AL: Establish safe therapeutic space
        RECEPTION,          # EN: Receive patient's narrative
        COHERENCE,          # IL: Baseline stabilization
        DISSONANCE,         # OZ: Controlled exposure to trauma
        SELF_ORGANIZATION,  # THOL: Patient's autonomous processing
        COHERENCE,          # IL: Integration of new understanding
        RESONANCE,          # RA: Amplify patient's strengths/resources
        COUPLING,           # UM: Connect to support systems
        SILENCE,            # SHA: Consolidation pause
    ]
    
    return {
        "name": "Trauma Recovery",
        "sequence": sequence,
        "presenting_problem": "PTSD with avoidance and intrusive symptoms",
        "therapeutic_goal": "Memory integration through controlled exposure",
        "key_operators": ["dissonance", "self_organization", "coherence"],
        "pattern_type": "OZ → THOL → IL (controlled crisis resolution)",
        "sessions_expected": "8-12 sessions",
    }


# =============================================================================
# CASE STUDY 2: ADDICTION HEALING
# =============================================================================


def case_addiction_healing():
    """Case Study: Addiction recovery through expansion-contraction pattern.
    
    Clinical Context
    ----------------
    Patient: 45-year-old with alcohol dependence, 2 weeks sober after
    detox. Pattern of using alcohol to manage anxiety and avoid emotions.
    Now facing "expanded" emotional awareness without previous coping mechanism.
    
    Therapeutic Goal
    ----------------
    Navigate expanded emotional awareness, identify core needs, simplify
    to essential coping strategies, build sustainable recovery structure.
    Pattern: VAL → NUL → IL (expansion → contraction → coherence).
    
    TNFR Structural Interpretation
    -------------------------------
    Addiction removal creates "expansion" (VAL) - more emotional data
    than patient can initially process. The therapeutic work:
    1. Acknowledge underlying dissonance first (OZ - pain driving addiction)
    2. Contract to essentials (NUL - core needs identification)
    3. Stabilize contracted state (IL)
    4. Controlled exploration of expanded awareness (VAL - feelings previously numbed)
    5. Stabilize expanded state (IL)
    6. Connect to recovery network (UM)
    
    Structural Sequence
    -------------------
    emission → reception → coherence → dissonance → contraction
    → coherence → expansion → coherence → coupling → silence
    
    Expected Outcomes
    -----------------
    - Reduced craving through emotional processing
    - Core coping strategies identified and practiced
    - Connection to support network (AA, therapy, family)
    - Health score: > 0.75 (stabilize pattern)
    
    Therapeutic Considerations
    ---------------------------
    - VAL (expansion) may feel overwhelming - pace carefully
    - NUL (contraction) is not suppression, it's strategic focus
    - UM (coupling) to recovery network is essential for sustainability
    - Relapse risk highest during OZ (dissonance) phase
    """
    sequence = [
        EMISSION,           # AL: Initiate recovery process
        RECEPTION,          # EN: Receive expanded emotional awareness
        COHERENCE,          # IL: Initial sobriety stabilization
        DISSONANCE,         # OZ: Identify underlying pain/needs
        CONTRACTION,        # NUL: Focus on core essential needs
        COHERENCE,          # IL: Stabilize contracted focus
        EXPANSION,          # VAL: Gradually explore emotional landscape
        COHERENCE,          # IL: Stabilize expanded awareness
        COUPLING,           # UM: Connect to recovery network
        SILENCE,            # SHA: Integration pause
    ]
    
    return {
        "name": "Addiction Healing",
        "sequence": sequence,
        "presenting_problem": "Alcohol dependence, 2 weeks post-detox",
        "therapeutic_goal": "Navigate emotional expansion, build sustainable recovery",
        "key_operators": ["expansion", "contraction", "coupling"],
        "pattern_type": "VAL → NUL → IL (expansion-contraction)",
        "sessions_expected": "12-16 sessions (initial phase)",
    }


# =============================================================================
# CASE STUDY 3: DEPRESSION EMERGENCE
# =============================================================================


def case_depression_emergence():
    """Case Study: Depression recovery through reactivation from silence.
    
    Clinical Context
    ----------------
    Patient: 28-year-old with major depressive disorder, presenting with
    anhedonia, low energy, emotional numbness. Describes feeling "frozen"
    or "shut down." Previous therapy helped stabilize crisis, now ready
    for reactivation.
    
    Therapeutic Goal
    ----------------
    Emerge from depressive "silence" (low νf), gradually reactivate
    structural frequency, reconnect with emotions and relationships.
    Pattern: SHA → AL → EN (silence → emission → reception).
    
    TNFR Structural Interpretation
    -------------------------------
    Depression as structural "silence" (SHA) - very low νf (structural
    frequency), minimal reorganization. Not absence of structure, but
    suspension of evolution. The work:
    1. Gentle activation (AL - small intention setting)
    2. Receptive capacity building (EN - allow input)
    3. Minimal stabilization (IL - initial coherence)
    4. Acknowledge resistance/difficulty (OZ - brief controlled dissonance)
    5. Contract to achievable steps (NUL - focus on what's possible)
    6. Stabilize focused approach (IL)
    7. Gradually expand reconnection (VAL - reconnect with life)
    8. Stabilize expansion (IL)
    9. Amplify positive signals (RA)
    10. Chosen rest (SHA - consolidation, not collapse)
    
    Structural Sequence
    -------------------
    emission → reception → coherence → dissonance → contraction
    → coherence → expansion → coherence → resonance → silence
    
    Expected Outcomes
    -----------------
    - Increased structural frequency (νf) - more energy, engagement
    - Reconnection with emotions (initially may be uncomfortable)
    - Rebuilding of relationships and activities
    - Health score: > 0.75 (regenerative pattern)
    
    Therapeutic Considerations
    ---------------------------
    - Start gently - don't force activation from depressive silence
    - SHA → AL transition must be gradual - honor the rest period
    - EN (reception) rebuilds capacity for input/connection
    - Multiple small cycles better than one big push
    - Risk of return to SHA if pushed too fast
    """
    sequence = [
        EMISSION,           # AL: Gentle intention activation (from silence)
        RECEPTION,          # EN: Build receptive capacity
        COHERENCE,          # IL: Minimal initial stabilization
        DISSONANCE,         # OZ: Acknowledge difficulty/resistance (brief)
        CONTRACTION,        # NUL: Focus on small achievable steps
        COHERENCE,          # IL: Stabilize focused approach
        EXPANSION,          # VAL: Gradual reconnection with life
        COHERENCE,          # IL: Stabilize expansion
        RESONANCE,          # RA: Amplify small positive signals
        SILENCE,            # SHA: Consolidation (now chosen, not default)
    ]
    
    return {
        "name": "Depression Emergence",
        "sequence": sequence,
        "presenting_problem": "Major depression with anhedonia and emotional numbness",
        "therapeutic_goal": "Reactivate from depressive silence, rebuild engagement",
        "key_operators": ["silence", "emission", "reception", "expansion"],
        "pattern_type": "SHA → AL → EN (reactivation from silence)",
        "sessions_expected": "16-24 sessions",
    }


# =============================================================================
# CASE STUDY 4: RELATIONSHIP REPAIR
# =============================================================================


def case_relationship_repair():
    """Case Study: Relationship repair through coupling-crisis-recoupling.
    
    Clinical Context
    ----------------
    Couple: Together 8 years, presenting with communication breakdown,
    increased conflict, emotional disconnection. Both committed to
    relationship but "out of sync" (phase mismatch). Recent conflict
    crisis point.
    
    Therapeutic Goal
    ----------------
    Repair phase synchrony (φ), process conflict constructively,
    establish new coupling with updated understanding. Pattern:
    UM → OZ → UM (coupling → crisis → new coupling).
    
    TNFR Structural Interpretation
    -------------------------------
    Relationship as coupled system - two nodes that must maintain phase
    coherence (φ₁ ≈ φ₂). Current state: phase mismatch causing interference
    (destructive resonance). The work:
    1. Establish therapeutic container for both (AL, EN)
    2. Stabilize individual coherence (IL - each person stable)
    3. Explore phase mismatch through dissonance (OZ - conflict as information)
    4. Allow individual reorganization (THOL - each person's process)
    5. Re-establish coupling with new understanding (UM - phase sync)
    6. Amplify new pattern (RA)
    7. Strengthen new coupling (UM - reinforce synchrony)
    8. Integration pause (SHA)
    
    Structural Sequence
    -------------------
    emission → reception → coherence → dissonance → self_organization
    → coupling → resonance → coupling → silence
    
    Expected Outcomes
    -----------------
    - Improved phase synchrony (better communication timing)
    - Conflict used constructively (OZ as information, not threat)
    - Stronger coupling through understanding (UM with awareness)
    - Health score: > 0.80 (therapeutic pattern)
    
    Therapeutic Considerations
    ---------------------------
    - Both partners need individual IL (coherence) before new UM (coupling)
    - OZ (dissonance) in session must be contained and productive
    - THOL (self-organization) respects each person's autonomy
    - New UM (coupling) is different from original - includes growth
    - Phase (φ) synchrony is ongoing work, not one-time fix
    """
    sequence = [
        EMISSION,           # AL: Create therapeutic container for couple
        RECEPTION,          # EN: Receive both perspectives
        COHERENCE,          # IL: Establish individual stability
        DISSONANCE,         # OZ: Explore conflict/phase mismatch
        SELF_ORGANIZATION,  # THOL: Allow individual reorganization
        COUPLING,           # UM: Re-establish with new understanding
        RESONANCE,          # RA: Amplify new positive patterns
        COUPLING,           # UM: Strengthen new coupling pattern
        SILENCE,            # SHA: Integration pause
    ]
    
    return {
        "name": "Relationship Repair",
        "sequence": sequence,
        "presenting_problem": "Communication breakdown, phase mismatch, increased conflict",
        "therapeutic_goal": "Repair phase synchrony, establish new coupling",
        "key_operators": ["coupling", "dissonance", "self_organization"],
        "pattern_type": "UM → OZ → UM (coupling through crisis)",
        "sessions_expected": "10-14 sessions",
    }


# =============================================================================
# VALIDATION AND REPORTING
# =============================================================================


def print_case_study(case_data):
    """Print detailed case study report with validation.
    
    Parameters
    ----------
    case_data : dict
        Case study data including sequence and context
    """
    print(f"\n{'=' * 70}")
    print(f"CASE STUDY: {case_data['name']}")
    print(f"{'=' * 70}")
    
    print(f"\nPresenting Problem:")
    print(f"  {case_data['presenting_problem']}")
    
    print(f"\nTherapeutic Goal:")
    print(f"  {case_data['therapeutic_goal']}")
    
    print(f"\nKey Operators: {', '.join(case_data['key_operators'])}")
    print(f"Pattern Type: {case_data['pattern_type']}")
    print(f"Expected Duration: {case_data['sessions_expected']}")
    
    # Validate sequence
    sequence = case_data['sequence']
    print(f"\nStructural Sequence:")
    print(f"  {' → '.join(sequence)}")
    
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
        print(f"\n--- Clinical Recommendations ---")
        for i, rec in enumerate(health.recommendations, 1):
            print(f"  {i}. {rec}")
    
    return True


def validate_all_case_studies():
    """Validate all therapeutic case studies and generate summary.
    
    Returns
    -------
    dict
        Summary statistics of validation results
    """
    print("\n" + "=" * 70)
    print("TNFR THERAPEUTIC CASE STUDIES VALIDATION")
    print("=" * 70)
    
    cases = [
        case_trauma_recovery(),
        case_addiction_healing(),
        case_depression_emergence(),
        case_relationship_repair(),
    ]
    
    results = {}
    
    for case_data in cases:
        passed = print_case_study(case_data)
        
        if passed:
            result = validate_sequence_with_health(case_data['sequence'])
            results[case_data['name']] = {
                "passed": True,
                "health": result.health_metrics.overall_health,
                "pattern": result.health_metrics.dominant_pattern,
                "length": len(case_data['sequence']),
            }
        else:
            results[case_data['name']] = {"passed": False}
    
    # Summary report
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    
    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)
    
    print(f"\nTotal Case Studies: {total_count}")
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
    print(f"✓ Minimum 4 case studies: {len(results) >= 4}")
    
    if all_passed and all_above_threshold:
        print(f"\n🎉 SUCCESS: All case studies meet acceptance criteria!")
    else:
        print(f"\n⚠️  ISSUES: Some cases need improvement")
    
    print("=" * 70)
    
    return results


def main():
    """Run all case study validations and demonstrations."""
    validate_all_case_studies()


if __name__ == "__main__":
    main()
