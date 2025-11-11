"""TNFR Creative Case Studies - Real-World Creative Processes.

This module demonstrates specific creative applications using TNFR structural
operators. Each case study represents a real-world creative scenario with
detailed context, operator sequence, and expected outcomes.
"""

from tnfr.operators.grammar import validate_sequence_with_health
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
# CREATIVE CASE STUDIES
# =============================================================================


def case_music_composition():
    """Case study: Composing a symphonic movement.
    
    Context: Composer creating a symphony movement from initial theme to
    final orchestration. Demonstrates artistic creation with multiple
    phases of exploration and refinement.
    
    Returns
    -------
    dict
        Case study with sequence, context, and metadata
    """
    return {
        "name": "Music Composition - Symphonic Movement",
        "domain": "Musical Arts",
        "sequence": [
            EMISSION,           # Initial theme conception
            RECEPTION,          # Study reference works, inspirations
            COHERENCE,          # Establish tonal center and form
            EXPANSION,          # Explore harmonic variations
            COHERENCE,          # Solidify theme structure
            DISSONANCE,         # Introduce tension, conflict themes
            MUTATION,           # Breakthrough - development section idea
            COHERENCE,          # Integrate development with themes
            RESONANCE,          # Amplify motifs through orchestration
            COHERENCE,          # Final integration, recapitulation
            COUPLING,           # Balance all orchestral voices
            SILENCE,            # Final reflection and completion
        ],
        "creative_challenge": "Balancing thematic development with coherent structure",
        "key_operators": [DISSONANCE, MUTATION, RESONANCE],
        "pattern_type": "Artistic Creation",
        "expected_outcome": "Complete symphonic movement with clear structure and emotional arc",
    }


def case_visual_art():
    """Case study: Creating an abstract painting.
    
    Context: Visual artist developing an abstract painting from concept
    to final work. Emphasizes exploration, creative tension, and
    autonomous emergence of form.
    
    Returns
    -------
    dict
        Case study with sequence, context, and metadata
    """
    return {
        "name": "Visual Art - Abstract Painting",
        "domain": "Visual Arts",
        "sequence": [
            EMISSION,           # Initial vision or emotion to express
            RECEPTION,          # Gather color palette, textures
            COHERENCE,          # Establish composition framework
            EXPANSION,          # Experiment with mark-making techniques
            COHERENCE,          # Settle on primary technique
            DISSONANCE,         # Create visual tension, contrasts
            MUTATION,           # Unexpected color or form emerges
            COHERENCE,          # Integrate surprising element
            RESONANCE,          # Repeat successful visual motifs
            COHERENCE,          # Refine and balance composition
            COUPLING,           # Final adjustments for unity
            SILENCE,            # Step back, declare complete
        ],
        "creative_challenge": "Finding emergent form while maintaining visual coherence",
        "key_operators": [EXPANSION, DISSONANCE, MUTATION],
        "pattern_type": "Artistic Creation",
        "expected_outcome": "Cohesive abstract painting with dynamic visual interest",
    }


def case_writing_process():
    """Case study: Writing a novel chapter.
    
    Context: Author developing a critical novel chapter with character
    development and plot advancement. Shows iterative creative process
    with multiple revision cycles.
    
    Returns
    -------
    dict
        Case study with sequence, context, and metadata
    """
    return {
        "name": "Writing Process - Novel Chapter",
        "domain": "Literary Arts",
        "sequence": [
            EMISSION,           # Chapter intention, what needs to happen
            RECEPTION,          # Review previous chapters, character arcs
            COHERENCE,          # Outline scene structure
            EXPANSION,          # Free write, explore dialogue options
            COHERENCE,          # Select best dialogue and scenes
            DISSONANCE,         # Introduce plot complication or conflict
            MUTATION,           # Character reveals unexpected depth
            COHERENCE,          # Integrate revelation into narrative
            EXPANSION,          # Elaborate emotional dimensions
            COHERENCE,          # Edit for pacing and clarity
            RECURSIVITY,        # Polish prose, layer symbolism
        ],
        "creative_challenge": "Advancing plot while deepening character authentically",
        "key_operators": [DISSONANCE, MUTATION, RECURSIVITY],
        "pattern_type": "Artistic Creation",
        "expected_outcome": "Polished chapter that advances story and develops characters",
    }


def case_product_design():
    """Case study: Designing a consumer product.
    
    Context: Product designer creating a new home appliance using
    human-centered design methodology. Demonstrates design thinking
    pattern with user feedback integration.
    
    Returns
    -------
    dict
        Case study with sequence, context, and metadata
    """
    return {
        "name": "Product Design - Smart Home Device",
        "domain": "Industrial Design",
        "sequence": [
            EMISSION,           # Design brief and project kickoff
            RECEPTION,          # User research, market analysis
            COHERENCE,          # Define design requirements
            EXPANSION,          # Ideate multiple form factors
            COHERENCE,          # Converge on promising direction
            DISSONANCE,         # Challenge cost/manufacturability assumptions
            MUTATION,           # Breakthrough - modular design concept
            COHERENCE,          # Refine modular system
            COUPLING,           # User testing and feedback
            TRANSITION,         # Move to production engineering
            SILENCE,            # Document design rationale
        ],
        "creative_challenge": "Balancing aesthetics, functionality, and manufacturability",
        "key_operators": [RECEPTION, MUTATION, COUPLING],
        "pattern_type": "Design Thinking",
        "expected_outcome": "User-validated product design ready for manufacturing",
    }


def case_software_development():
    """Case study: Developing a new software feature.
    
    Context: Software team designing and implementing an innovative
    feature through agile development. Shows innovation cycle with
    iteration and user feedback.
    
    Returns
    -------
    dict
        Case study with sequence, context, and metadata
    """
    return {
        "name": "Software Development - Innovative Feature",
        "domain": "Software Engineering",
        "sequence": [
            EMISSION,           # Feature request or user pain point
            RECEPTION,          # Analyze existing system, user needs
            COHERENCE,          # Design initial architecture
            EXPANSION,          # Explore implementation approaches
            COHERENCE,          # Select technical approach
            DISSONANCE,         # Identify performance bottlenecks
            MUTATION,           # Innovative algorithm or pattern discovered
            COHERENCE,          # Integrate solution into codebase
            RESONANCE,          # Apply pattern to related features
            TRANSITION,         # Deploy to production
        ],
        "creative_challenge": "Solving technical constraints creatively while maintaining system integrity",
        "key_operators": [EXPANSION, MUTATION, TRANSITION],
        "pattern_type": "Innovation Cycle",
        "expected_outcome": "Deployed feature that elegantly solves user problem",
    }


def case_choreography_creation():
    """Case study: Creating a dance piece.
    
    Context: Choreographer developing a contemporary dance piece
    from concept to performance. Emphasizes embodied creativity
    and iterative refinement through rehearsal.
    
    Returns
    -------
    dict
        Case study with sequence, context, and metadata
    """
    return {
        "name": "Choreography - Contemporary Dance Piece",
        "domain": "Performance Arts",
        "sequence": [
            EMISSION,           # Conceptual theme or emotion
            RECEPTION,          # Research movement vocabularies
            COHERENCE,          # Establish movement language
            EXPANSION,          # Improvise with dancers
            COHERENCE,          # Select and refine phrases
            DISSONANCE,         # Introduce physical/spatial tension
            MUTATION,           # Unexpected movement quality emerges
            COHERENCE,          # Structure new discovery into piece
            RESONANCE,          # Repeat motifs with variations
            COUPLING,           # Sync with music and lighting
            SILENCE,            # Final staging and performance
        ],
        "creative_challenge": "Translating abstract concept into embodied movement",
        "key_operators": [EXPANSION, DISSONANCE, COUPLING],
        "pattern_type": "Artistic Creation",
        "expected_outcome": "Performance-ready dance piece with clear artistic vision",
    }


# =============================================================================
# VALIDATION AND REPORTING
# =============================================================================


def validate_case_study(case_data):
    """Validate a case study sequence and print results.
    
    Parameters
    ----------
    case_data : dict
        Case study dictionary with sequence and metadata
    
    Returns
    -------
    bool
        True if validation passed, False otherwise
    """
    print(f"\n{'=' * 70}")
    print(f"Case Study: {case_data['name']}")
    print(f"{'=' * 70}")
    print(f"Domain: {case_data['domain']}")
    print(f"Sequence: {' → '.join(case_data['sequence'])}")
    
    result = validate_sequence_with_health(case_data['sequence'])
    
    if not result.passed:
        print(f"\n✗ VALIDATION FAILED: {result.message}")
        return False
    
    health = result.health_metrics
    print(f"\n✓ VALIDATION PASSED")
    print(f"\n--- Health Metrics ---")
    print(f"Overall Health:  {health.overall_health:.3f}")
    print(f"Pattern:         {health.dominant_pattern.upper()}")
    print(f"Balance:         {health.balance_score:.3f}")
    print(f"Sustainability:  {health.sustainability_index:.3f}")
    
    if health.overall_health >= 0.85:
        status = "🌟 EXCELLENT"
    elif health.overall_health >= 0.75:
        status = "✓ GOOD"
    else:
        status = "⚠ FAIR"
    
    print(f"\nHealth Status: {status}")
    print(f"\n--- Creative Context ---")
    print(f"Challenge: {case_data['creative_challenge']}")
    print(f"Key Operators: {', '.join(case_data['key_operators'])}")
    print(f"Pattern Type: {case_data['pattern_type']}")
    
    return True


def validate_all_case_studies():
    """Validate all creative case studies.
    
    Returns
    -------
    dict
        Summary statistics of validation results
    """
    print("\n" + "=" * 70)
    print("TNFR CREATIVE CASE STUDIES VALIDATION")
    print("=" * 70)
    
    cases = [
        case_music_composition(),
        case_visual_art(),
        case_writing_process(),
        case_product_design(),
        case_software_development(),
        case_choreography_creation(),
    ]
    
    results = {}
    
    for case_data in cases:
        passed = validate_case_study(case_data)
        
        if passed:
            result = validate_sequence_with_health(case_data['sequence'])
            results[case_data['name']] = {
                "passed": True,
                "health": result.health_metrics.overall_health,
                "pattern": result.health_metrics.dominant_pattern,
                "length": len(case_data['sequence']),
                "domain": case_data['domain'],
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
            details = f"(health: {result['health']:.3f}, {result['domain']})"
        else:
            status = "✗ FAIL"
            details = ""
        name_short = name.split(" - ")[1] if " - " in name else name
        print(f"  {status:8s} {name_short:40s} {details}")
    
    return results


def main():
    """Run case studies validation."""
    results = validate_all_case_studies()
    
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
    
    print(f"\n✓ All case studies valid: {all_passed}")
    print(f"✓ All health scores > {health_threshold}: {all_above_threshold}")
    print(f"✓ Minimum 5 case studies: {len(results) >= 5}")
    print(f"✓ Diverse domains covered: {len(set(r.get('domain', '') for r in results.values() if r['passed']))}")
    
    if all_passed and all_above_threshold:
        print(f"\n🎉 SUCCESS: All creative case studies meet acceptance criteria!")
    else:
        print(f"\n⚠️  ISSUES: Some case studies need improvement")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
