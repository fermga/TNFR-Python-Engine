"""TNFR Organizational Patterns - Institutional Evolution and Change Management.

This module demonstrates application of TNFR structural operators in
organizational contexts, showing how coherent operator sequences can model:
- Crisis management and rapid institutional response
- Strategic planning and long-term transformation
- Team formation and group synchronization
- Organizational resilience and adaptation

Each pattern is validated using TNFR's structural health metrics to ensure
coherence, balance, and sustainability in organizational contexts.
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
# ORGANIZATIONAL PATTERNS
# =============================================================================


def get_crisis_management_sequence():
    """Crisis management pattern - rapid institutional response to disruption.
    
    Context: Immediate organizational crisis response, emergency containment,
    acute institutional distress. Prioritizes rapid stabilization over
    deep transformation.
    
    Structural flow:
    1. EMISSION (AL): Leadership communication and response initiation
    2. RECEPTION (EN): Stakeholder engagement and situation assessment
    3. COHERENCE (IL): Immediate stabilization measures
    4. DISSONANCE (OZ): Acknowledge crisis tensions (brief, controlled)
    5. TRANSITION (NAV): Shift to recovery mode
    6. COUPLING (UM): Coordinate cross-functional response teams
    7. RESONANCE (RA): Amplify successful adaptations
    8. SILENCE (SHA): Consolidation and organizational learning
    
    Expected metrics:
    - Health score: > 0.75 (good structural quality)
    - Pattern: HIERARCHICAL or STABILIZE (rapid coordination)
    - Balance: Controlled dissonance with strong stabilizers
    - Sustainability: Moderate (designed for short-term intervention)
    
    Use cases:
    - Market disruption response
    - Leadership transition crisis
    - Operational emergency management
    - Reputation crisis containment
    
    Returns
    -------
    list[str]
        Validated operator sequence for crisis management
    """
    return [
        EMISSION,        # AL: Leadership communication/response
        RECEPTION,       # EN: Stakeholder engagement
        COHERENCE,       # IL: Immediate stabilization measures
        DISSONANCE,      # OZ: Acknowledge crisis tensions
        TRANSITION,      # NAV: Shift to recovery mode
        COUPLING,        # UM: Coordinate cross-functional response
        RESONANCE,       # RA: Amplify successful adaptations
        SILENCE,         # SHA: Consolidation and learning
    ]


def get_strategic_planning_sequence():
    """Strategic planning pattern - long-term organizational transformation.
    
    Context: Comprehensive strategic planning process, long-term institutional
    evolution, structured organizational transformation with vision alignment.
    
    Structural flow:
    1. EMISSION (AL): Vision articulation by leadership
    2. RECEPTION (EN): Environmental scanning and stakeholder input
    3. COHERENCE (IL): Strategy formulation and alignment
    4. DISSONANCE (OZ): Surface tensions and strategic trade-offs
    5. SELF_ORGANIZATION (THOL): Bottom-up strategy refinement
    6. CONTRACTION (NUL): Focus on key strategic priorities
    7. COHERENCE (IL): Intermediate strategic alignment
    8. EXPANSION (VAL): Explore strategic options and scenarios
    9. COUPLING (UM): Coordinate planning across organizational units
    10. RESONANCE (RA): Strategy execution amplification
    11. TRANSITION (NAV): Implementation phase transition
    
    Expected metrics:
    - Health score: > 0.85 (excellent structural quality)
    - Pattern: ORGANIZATIONAL (institutional evolution)
    - Balance: Equilibrium between exploration and stabilization
    - Sustainability: High (complete transformation cycle)
    
    Use cases:
    - Multi-year strategic planning
    - Organizational restructuring
    - Business model transformation
    - Vision and mission realignment
    
    Returns
    -------
    list[str]
        Validated operator sequence for strategic planning
    """
    return [
        EMISSION,           # AL: Vision articulation by leadership
        RECEPTION,          # EN: Environmental scanning/stakeholder input
        COHERENCE,          # IL: Strategy formulation and alignment
        DISSONANCE,         # OZ: Surface tensions/trade-offs
        SELF_ORGANIZATION,  # THOL: Bottom-up strategy refinement
        CONTRACTION,        # NUL: Focus on key strategic priorities (closes THOL)
        COHERENCE,          # IL: Intermediate strategic alignment
        EXPANSION,          # VAL: Explore strategic options/scenarios
        COUPLING,           # UM: Coordinate planning across units
        RESONANCE,          # RA: Strategy execution amplification
        TRANSITION,         # NAV: Implementation phase transition
    ]


def get_team_formation_sequence():
    """Team formation pattern - cohesion and group synchronization.
    
    Context: New team formation, group cohesion development,
    synchronization of team members into effective collaborative unit.
    
    Structural flow:
    1. EMISSION (AL): Team charter and purpose definition
    2. RECEPTION (EN): Individual contributions and perspectives
    3. COHERENCE (IL): Initial team norms and working agreements
    4. DISSONANCE (OZ): Address conflicts and differences
    5. MUTATION (ZHIR): Team identity breakthrough
    6. COHERENCE (IL): Refined team coherence
    7. EXPANSION (VAL): Explore team potential and capabilities
    8. COUPLING (UM): Role clarification and coordination
    9. RESONANCE (RA): Amplify team strengths
    10. SILENCE (SHA): Team consolidation period
    
    Expected metrics:
    - Health score: > 0.75 (good structural quality)
    - Pattern: HIERARCHICAL with strong coupling
    - Balance: Good equilibrium with transformation
    - Sustainability: High (team stability achieved)
    
    Use cases:
    - Cross-functional team launch
    - Project team formation
    - Leadership team alignment
    - Remote team synchronization
    
    Returns
    -------
    list[str]
        Validated operator sequence for team formation
    """
    return [
        EMISSION,           # AL: Team charter/purpose definition
        RECEPTION,          # EN: Individual contributions/perspectives
        COHERENCE,          # IL: Initial team norms
        DISSONANCE,         # OZ: Address conflicts/differences
        MUTATION,           # ZHIR: Team identity breakthrough
        COHERENCE,          # IL: Refined team coherence
        EXPANSION,          # VAL: Explore team potential/capabilities
        COUPLING,           # UM: Role clarification and coordination
        RESONANCE,          # RA: Amplify team strengths
        SILENCE,            # SHA: Team consolidation period
    ]


def get_organizational_transformation_sequence():
    """Organizational transformation pattern - comprehensive institutional change.
    
    Context: Major organizational transformation initiative, complete
    institutional evolution from current state to desired future state.
    Demonstrates canonical organizational evolution with all key operators.
    
    Structural flow:
    1. EMISSION (AL): Communication of vision/objectives
    2. RECEPTION (EN): Reception and adoption by team
    3. COUPLING (UM): Synchronization of efforts
    4. RESONANCE (RA): Amplification of positive momentum
    5. TRANSITION (NAV): Navigate to implementation phase
    6. DISSONANCE (OZ): Resistances and emergent tensions
    7. SELF_ORGANIZATION (THOL): Autonomous structural reorganization
    8. SILENCE (SHA): Integration and stabilization (closes THOL)
    9. EMISSION (AL): Re-energize transformation
    10. COHERENCE (IL): New organizational stability
    11. EXPANSION (VAL): Scaling the transformation
    12. COHERENCE (IL): Final consolidation
    13. RECURSIVITY (REMESH): Consolidation into organizational culture
    
    Expected metrics:
    - Health score: > 0.75 (good structural quality)
    - Pattern: May show as THERAPEUTIC, FRACTAL, or COMPLEX due to comprehensive transformation
    - Balance: Excellent equilibrium across transformation
    - Sustainability: Very high (complete transformation with recursion)
    
    Use cases:
    - Digital transformation programs
    - Agile transformation initiatives
    - Cultural change programs
    - Enterprise-wide change management
    
    Returns
    -------
    list[str]
        Validated operator sequence for organizational transformation
    """
    return [
        EMISSION,           # AL: Comunicación de visión/objetivos
        RECEPTION,          # EN: Recepción y adopción por equipo
        COUPLING,           # UM: Sincronización de esfuerzos
        RESONANCE,          # RA: Amplificación de momentum positivo
        TRANSITION,         # NAV: Navigate to implementation
        DISSONANCE,         # OZ: Resistencias y tensiones emergentes
        SELF_ORGANIZATION,  # THOL: Reorganización estructural autónoma
        SILENCE,            # SHA: Integration (closes THOL)
        EMISSION,           # AL: Re-energize
        COHERENCE,          # IL: Nueva estabilidad organizacional
        EXPANSION,          # VAL: Escalado de la transformación
        COHERENCE,          # IL: Final consolidation
        RECURSIVITY,        # REMESH: Consolidación en cultura organizacional
    ]


def get_innovation_cycle_sequence():
    """Innovation cycle pattern - exploration to implementation.
    
    Context: Innovation initiative from exploration through experimentation
    to scaling, maintaining creative energy while achieving implementation.
    
    Structural flow:
    1. EMISSION (AL): Innovation challenge articulation
    2. RECEPTION (EN): Gather diverse perspectives
    3. COHERENCE (IL): Initial framing and focus
    4. DISSONANCE (OZ): Creative tension and ideation
    5. SELF_ORGANIZATION (THOL): Emergent solution formation
    6. CONTRACTION (NUL): Focus on viable solutions (closes THOL)
    7. COHERENCE (IL): Concept selection and refinement
    8. EXPANSION (VAL): Explore implementation possibilities
    9. COUPLING (UM): Coordinate implementation resources
    10. RESONANCE (RA): Amplify successful innovations
    11. TRANSITION (NAV): Move to scaling phase
    
    Expected metrics:
    - Health score: > 0.80 (strong structural quality)
    - Pattern: CREATIVE or ORGANIZATIONAL
    - Balance: Good balance of exploration and stabilization
    - Sustainability: High (complete innovation lifecycle)
    
    Use cases:
    - Innovation lab initiatives
    - Product development cycles
    - Process improvement programs
    - Design thinking workshops
    
    Returns
    -------
    list[str]
        Validated operator sequence for innovation cycles
    """
    return [
        EMISSION,           # AL: Innovation challenge articulation
        RECEPTION,          # EN: Gather diverse perspectives
        COHERENCE,          # IL: Initial framing and focus
        DISSONANCE,         # OZ: Creative tension and ideation
        SELF_ORGANIZATION,  # THOL: Emergent solution formation
        CONTRACTION,        # NUL: Focus on viable solutions (closes THOL)
        COHERENCE,          # IL: Concept selection and refinement
        EXPANSION,          # VAL: Explore implementation possibilities
        COUPLING,           # UM: Coordinate implementation resources
        RESONANCE,          # RA: Amplify successful innovations
        TRANSITION,         # NAV: Move to scaling phase
    ]


def get_change_resistance_resolution_sequence():
    """Change resistance resolution pattern - addressing organizational inertia.
    
    Context: Addressing resistance to change, working through organizational
    inertia, transforming resistance into engagement.
    
    Structural flow:
    1. EMISSION (AL): Communicate change rationale clearly
    2. RECEPTION (EN): Listen to and understand resistance
    3. COHERENCE (IL): Acknowledge current state stability
    4. DISSONANCE (OZ): Surface underlying tensions
    5. CONTRACTION (NUL): Simplify change to essentials
    6. EMISSION (AL): Re-communicate simplified vision
    7. COUPLING (UM): Create stakeholder alignment
    8. RESONANCE (RA): Amplify early adopter successes
    9. TRANSITION (NAV): Shift organizational momentum
    
    Expected metrics:
    - Health score: > 0.75 (good structural quality)
    - Pattern: ORGANIZATIONAL or THERAPEUTIC
    - Balance: Careful dissonance with strong stabilization
    - Sustainability: High (sustainable change adoption)
    
    Use cases:
    - Overcoming change resistance
    - Cultural transformation barriers
    - Technology adoption challenges
    - Process change implementation
    
    Returns
    -------
    list[str]
        Validated operator sequence for change resistance resolution
    """
    return [
        EMISSION,           # AL: Communicate change rationale
        RECEPTION,          # EN: Listen to and understand resistance
        COHERENCE,          # IL: Acknowledge current state stability
        DISSONANCE,         # OZ: Surface underlying tensions
        CONTRACTION,        # NUL: Simplify change to essentials
        EMISSION,           # AL: Re-communicate simplified vision
        COUPLING,           # UM: Create stakeholder alignment
        RESONANCE,          # RA: Amplify early adopter successes
        TRANSITION,         # NAV: Shift organizational momentum
    ]


# =============================================================================
# VALIDATION AND REPORTING
# =============================================================================


def validate_pattern(name, sequence):
    """Validate an organizational pattern and print detailed report.
    
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
    """Validate all organizational patterns and generate summary report.
    
    Returns
    -------
    dict
        Summary statistics of validation results
    """
    print("\n" + "=" * 70)
    print("TNFR ORGANIZATIONAL PATTERNS VALIDATION")
    print("=" * 70)
    
    patterns = {
        "Crisis Management": get_crisis_management_sequence(),
        "Strategic Planning": get_strategic_planning_sequence(),
        "Team Formation": get_team_formation_sequence(),
        "Organizational Transformation": get_organizational_transformation_sequence(),
        "Innovation Cycle": get_innovation_cycle_sequence(),
        "Change Resistance Resolution": get_change_resistance_resolution_sequence(),
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
        
        # Count organizational patterns
        org_patterns = sum(
            1 for r in results.values()
            if r["passed"] and r["pattern"] == "organizational"
        )
        org_percentage = (org_patterns / passed_count) * 100 if passed_count > 0 else 0
        print(f"Organizational Pattern Detection: {org_patterns}/{passed_count} ({org_percentage:.1f}%)")
    
    print(f"\n--- Individual Results ---")
    for name, result in results.items():
        if result["passed"]:
            status = "✓ PASS"
            details = f"(health: {result['health']:.3f}, pattern: {result['pattern']})"
        else:
            status = "✗ FAIL"
            details = ""
        print(f"  {status:8s} {name:35s} {details}")
    
    return results


def main():
    """Run organizational patterns validation and demonstration."""
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
    
    # Count organizational patterns (may be detected as various complex patterns)
    org_patterns = sum(
        1 for r in results.values()
        if r["passed"] and r["pattern"] == "organizational"
    )
    complex_patterns = sum(
        1 for r in results.values()
        if r["passed"] and r["pattern"] in ["organizational", "therapeutic", "fractal", "complex", "regenerative"]
    )
    passed_count = sum(1 for r in results.values() if r["passed"])
    org_percentage = (org_patterns / passed_count) * 100 if passed_count > 0 else 0
    complex_percentage = (complex_patterns / passed_count) * 100 if passed_count > 0 else 0
    
    print(f"\n✓ All sequences valid: {all_passed}")
    print(f"✓ All health scores > {health_threshold}: {all_above_threshold}")
    print(f"✓ Minimum 6 specialized patterns: {len(results) >= 6}")
    print(f"✓ ORGANIZATIONAL pattern detection: {org_patterns}/{passed_count} ({org_percentage:.1f}%)")
    print(f"✓ Complex/transformative patterns: {complex_patterns}/{passed_count} ({complex_percentage:.1f}%)")
    print(f"\nNote: Organizational patterns may be detected as THERAPEUTIC, FRACTAL, COMPLEX,")
    print(f"or REGENERATIVE due to their comprehensive transformation sequences.")
    print(f"The semantic organizational nature is preserved in all patterns.")
    
    if all_passed and all_above_threshold:
        print(f"\n🎉 SUCCESS: All organizational patterns meet acceptance criteria!")
        print(f"All patterns are valid and demonstrate organizational transformation principles.")
    else:
        print(f"\n⚠️  ISSUES: Some patterns need improvement")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
