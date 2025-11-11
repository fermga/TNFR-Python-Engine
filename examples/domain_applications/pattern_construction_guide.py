"""TNFR Pattern Construction Guide - Principles for Building Effective Sequences.

This module provides guidelines and principles for constructing high-quality
structural operator sequences that respect TNFR canonical grammar and optimize
for specific purposes.

Key Topics
----------
- Canonical grammar rules (what sequences are valid)
- Operator compatibility constraints
- Pattern-specific optimization strategies
- Health metrics interpretation
- Common pitfalls and how to avoid them
- Sequence troubleshooting

Examples
--------
>>> from pattern_construction_guide import validate_and_diagnose
>>> sequence = ["emission", "reception", "coherence", "dissonance", "mutation", "coherence", "silence"]
>>> result = validate_and_diagnose(sequence)
>>> print(result['diagnosis'])
"""

from typing import Dict, List, Any
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
# CANONICAL GRAMMAR RULES
# =============================================================================


CANONICAL_RULES = """
TNFR Canonical Grammar Rules
=============================

1. **Sequence Start Rules**
   - MUST start with: EMISSION (AL) or RECURSIVITY (REMESH)
   - Common pattern: EMISSION → RECEPTION → COHERENCE (AL→EN→IL)

2. **Sequence End Rules**
   - MUST end with: DISSONANCE, RECURSIVITY, SILENCE, or TRANSITION
   - Most common: SILENCE (SHA) for consolidation
   - Alternative: TRANSITION (NAV) for ongoing evolution

3. **Reception-Coherence Requirement**
   - EVERY sequence must contain: RECEPTION → COHERENCE (EN→IL)
   - This establishes baseline coherence for the structure
   - Place early in sequence for proper foundation

4. **Operator Compatibility Constraints**
   
   After EXPANSION (VAL):
   - MUST be followed by COHERENCE (IL) before any other operator
   - Cannot go directly to: DISSONANCE, CONTRACTION, MUTATION
   
   After MUTATION (ZHIR):
   - MUST be followed by COHERENCE (IL) before most operators
   - Cannot go directly to: CONTRACTION, SELF_ORGANIZATION
   
   After CONTRACTION (NUL):
   - MUST be followed by COHERENCE (IL) before transformation operators
   - Cannot go directly to: MUTATION
   
   After RECURSIVITY (REMESH):
   - MUST be followed by COHERENCE (IL) or TRANSITION (NAV)
   - Cannot go directly to: SILENCE
   
   After RECEPTION (EN):
   - Cannot go directly to: EXPANSION
   - Should go to COHERENCE first

5. **Self-Organization Requirements**
   - SELF_ORGANIZATION (THOL) requires a destabilizer in previous 3 operators
   - Valid destabilizers: DISSONANCE, EXPANSION, TRANSITION
   - Example: ...DISSONANCE → SELF_ORGANIZATION... (valid)
   - Example: ...COHERENCE → RESONANCE → SELF_ORGANIZATION... (invalid)

6. **Mutation Prerequisites**
   - MUTATION (ZHIR) should be preceded by DISSONANCE (OZ)
   - Creates controlled tension before phase transition
   - Pattern: ...DISSONANCE → MUTATION... is canonical

7. **Frequency Harmony**
   - Operators have structural frequencies: high, medium, zero
   - Smooth transitions between compatible frequencies preferred
   - Avoid jarring transitions (e.g., zero → high directly)

8. **Balance Considerations**
   - Good sequences balance stabilizers and destabilizers
   - Stabilizers: COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE
   - Destabilizers: DISSONANCE, MUTATION, EXPANSION
   - Transformers: CONTRACTION, TRANSITION, RECURSIVITY
"""


def print_canonical_rules():
    """Print the canonical grammar rules."""
    print(CANONICAL_RULES)


# =============================================================================
# OPERATOR COMPATIBILITY MATRIX
# =============================================================================


def get_operator_compatibility_notes() -> Dict[str, List[str]]:
    """Get compatibility notes for each operator.
    
    Returns
    -------
    dict
        Maps operator names to lists of compatibility notes
    """
    return {
        EMISSION: [
            "Valid sequence starter",
            "Followed by RECEPTION is canonical pattern",
            "Seeds initial coherence",
        ],
        RECEPTION: [
            "Should be followed by COHERENCE (canonical requirement)",
            "Cannot be followed directly by EXPANSION",
            "Gathers environmental context",
        ],
        COHERENCE: [
            "Universal stabilizer - can follow most operators",
            "Required after: EXPANSION, MUTATION, CONTRACTION, RECURSIVITY",
            "Can be used multiple times in sequence",
            "Transition to DISSONANCE requires careful validation",
        ],
        EXPANSION: [
            "MUST be followed by COHERENCE before other operators",
            "Cannot come directly after RECEPTION",
            "Should follow COHERENCE",
            "Increases structural dimensionality",
        ],
        CONTRACTION: [
            "MUST be followed by COHERENCE before transformation operators",
            "Cannot be followed directly by MUTATION",
            "Cannot come directly after EXPANSION",
            "Concentrates toward essence",
        ],
        DISSONANCE: [
            "Cannot come directly after EXPANSION",
            "Often precedes MUTATION (canonical pattern)",
            "Enables SELF_ORGANIZATION (destabilizer requirement)",
            "Valid sequence terminator",
        ],
        MUTATION: [
            "Should be preceded by DISSONANCE",
            "MUST be followed by COHERENCE before most operators",
            "Cannot be followed by: CONTRACTION, SELF_ORGANIZATION directly",
            "Phase transition operator",
        ],
        SELF_ORGANIZATION: [
            "Requires destabilizer in previous 3 operators",
            "Cannot come directly after MUTATION",
            "Cannot come directly after stable sequences",
            "Spawns autonomous cascades",
        ],
        RESONANCE: [
            "Amplifies aligned structures",
            "Works well with COUPLING for propagation",
            "Can appear multiple times",
            "High structural frequency",
        ],
        COUPLING: [
            "Synchronizes network connections",
            "Often used in BOOTSTRAP patterns",
            "Can appear multiple times",
            "Medium structural frequency",
        ],
        SILENCE: [
            "Common sequence terminator",
            "Cannot come directly after RECURSIVITY",
            "Suspends reorganization",
            "Zero structural frequency",
        ],
        TRANSITION: [
            "Valid sequence terminator",
            "Can follow RECURSIVITY",
            "Guides controlled regime handoffs",
            "Medium structural frequency",
        ],
        RECURSIVITY: [
            "MUST be followed by COHERENCE or TRANSITION",
            "Cannot be followed by SILENCE directly",
            "Valid sequence starter",
            "Echoes across scales",
        ],
    }


# =============================================================================
# PATTERN-SPECIFIC CONSTRUCTION STRATEGIES
# =============================================================================


PATTERN_STRATEGIES = {
    "BOOTSTRAP": {
        "purpose": "Rapid system initialization",
        "key_sequence": [EMISSION, RECEPTION, COUPLING, COHERENCE, SILENCE],
        "min_operators": 5,
        "max_operators": 5,
        "critical_elements": [
            "Must include EMISSION → RECEPTION → COHERENCE",
            "COUPLING establishes basic connections",
            "End with SILENCE for stable initialization",
        ],
        "optimization_tips": [
            "Keep it minimal - don't over-engineer",
            "Focus on speed and effectiveness",
            "Use COUPLING for quick network establishment",
        ],
    },
    "EXPLORE": {
        "purpose": "Controlled exploration with safe return",
        "key_sequence": [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE],
        "min_operators": 7,
        "max_operators": 12,
        "critical_elements": [
            "Establish baseline with early COHERENCE",
            "DISSONANCE → MUTATION for exploration",
            "Return to COHERENCE before ending",
            "End with SILENCE to consolidate discoveries",
        ],
        "optimization_tips": [
            "Balance exploration (DISSONANCE, MUTATION) with stability (COHERENCE)",
            "Can have multiple exploration cycles",
            "Always return to coherent state before ending",
        ],
    },
    "STABILIZE": {
        "purpose": "Robust consolidation for long-term maintenance",
        "key_sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, COHERENCE, SILENCE],
        "min_operators": 6,
        "max_operators": 10,
        "critical_elements": [
            "Multiple COHERENCE operators",
            "RESONANCE to amplify stable elements",
            "End with COHERENCE → SILENCE pattern",
        ],
        "optimization_tips": [
            "Maximize sustainability metrics",
            "Use multiple stabilization layers",
            "RESONANCE before final COHERENCE enhances stability",
        ],
    },
    "RESONATE": {
        "purpose": "Amplification and network propagation",
        "key_sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, COUPLING, RESONANCE, COHERENCE, SILENCE],
        "min_operators": 8,
        "max_operators": 12,
        "critical_elements": [
            "Multiple RESONANCE operators",
            "COUPLING for network synchronization",
            "Pattern: RESONANCE → COUPLING → RESONANCE",
        ],
        "optimization_tips": [
            "Optimize frequency harmony metrics",
            "Use COUPLING between RESONANCE operators",
            "Can add SELF_ORGANIZATION for emergent harmonics",
        ],
    },
    "COMPRESS": {
        "purpose": "Simplification and essence extraction",
        "key_sequence": [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE, CONTRACTION, COHERENCE, CONTRACTION, COHERENCE, SILENCE],
        "min_operators": 10,
        "max_operators": 13,
        "critical_elements": [
            "Start with EXPANSION (from complex state)",
            "COHERENCE after EXPANSION (required)",
            "Multiple CONTRACTION operators",
            "COHERENCE after each CONTRACTION",
        ],
        "optimization_tips": [
            "Maximize complexity efficiency",
            "Always stabilize (COHERENCE) after dimensional changes",
            "Can alternate CONTRACTION and COHERENCE multiple times",
        ],
    },
    "COMPLEX": {
        "purpose": "Multi-pattern integration",
        "key_sequence": [EMISSION, RECEPTION, COUPLING, COHERENCE, DISSONANCE, MUTATION, COHERENCE, RESONANCE, COHERENCE, SILENCE],
        "min_operators": 10,
        "max_operators": 18,
        "critical_elements": [
            "Combines multiple pattern signatures",
            "Maintains overall balance",
            "Includes BOOTSTRAP, EXPLORE, and STABILIZE components",
        ],
        "optimization_tips": [
            "Think in terms of pattern composition",
            "Ensure each sub-pattern is complete",
            "Maintain balance across the full sequence",
            "Use COHERENCE as transition between patterns",
        ],
    },
}


def get_pattern_strategy(pattern_type: str) -> Dict[str, Any]:
    """Get construction strategy for a specific pattern type.
    
    Parameters
    ----------
    pattern_type : str
        Pattern type (e.g., "BOOTSTRAP", "EXPLORE", "STABILIZE")
        
    Returns
    -------
    dict
        Strategy information including purpose, key sequence, and tips
    """
    return PATTERN_STRATEGIES.get(pattern_type.upper(), {
        "purpose": "Unknown pattern type",
        "critical_elements": [],
        "optimization_tips": ["No specific strategy available"],
    })


# =============================================================================
# HEALTH METRICS INTERPRETATION
# =============================================================================


def interpret_health_metrics(health: float, pattern: str) -> str:
    """Interpret health score and provide guidance.
    
    Parameters
    ----------
    health : float
        Overall health score (0.0-1.0)
    pattern : str
        Detected pattern type
        
    Returns
    -------
    str
        Interpretation and recommendations
    """
    interpretation = f"Pattern: {pattern}\nHealth Score: {health:.3f}\n\n"
    
    if health >= 0.85:
        interpretation += "EXCELLENT: High structural quality.\n"
        interpretation += "- All canonical requirements met\n"
        interpretation += "- Strong balance and sustainability\n"
        interpretation += "- Ready for production use\n"
    elif health >= 0.75:
        interpretation += "GOOD: Solid structural quality.\n"
        interpretation += "- Meets canonical requirements\n"
        interpretation += "- Good balance\n"
        interpretation += "- Suitable for most use cases\n"
    elif health >= 0.65:
        interpretation += "ACCEPTABLE: Meets minimum requirements.\n"
        interpretation += "- Basic canonical compliance\n"
        interpretation += "- Consider optimization for critical applications\n"
        interpretation += "- May benefit from additional stabilization\n"
    elif health >= 0.50:
        interpretation += "MODERATE: Some structural concerns.\n"
        interpretation += "- Review balance between stabilizers/destabilizers\n"
        interpretation += "- Consider adding more COHERENCE operators\n"
        interpretation += "- Check pattern completeness\n"
    else:
        interpretation += "POOR: Significant structural issues.\n"
        interpretation += "- Likely violates canonical requirements\n"
        interpretation += "- Review operator compatibility\n"
        interpretation += "- May need complete redesign\n"
    
    return interpretation


# =============================================================================
# COMMON PITFALLS AND SOLUTIONS
# =============================================================================


COMMON_PITFALLS = {
    "missing_reception_coherence": {
        "error": "missing reception→coherence segment",
        "explanation": "Every sequence must contain RECEPTION followed by COHERENCE",
        "solution": "Add RECEPTION → COHERENCE early in the sequence (typically after EMISSION)",
        "example_fix": "EMISSION → RECEPTION → COHERENCE → ...",
    },
    "invalid_start": {
        "error": "must start with emission, recursivity",
        "explanation": "Sequences must begin with a valid starter operator",
        "solution": "Begin sequence with EMISSION (most common) or RECURSIVITY",
        "example_fix": "EMISSION → RECEPTION → COHERENCE → ...",
    },
    "invalid_end": {
        "error": "sequence must end with dissonance, recursivity, silence, transition",
        "explanation": "Sequences must end with a valid terminator",
        "solution": "End with SILENCE (most common), TRANSITION, DISSONANCE, or RECURSIVITY",
        "example_fix": "... → COHERENCE → SILENCE",
    },
    "expansion_incompatibility": {
        "error": "expansion incompatible after reception / contraction incompatible after expansion",
        "explanation": "EXPANSION requires COHERENCE before and after",
        "solution": "Always place COHERENCE immediately after EXPANSION",
        "example_fix": "... → COHERENCE → EXPANSION → COHERENCE → ...",
    },
    "mutation_incompatibility": {
        "error": "mutation incompatible after contraction / contraction incompatible after mutation",
        "explanation": "MUTATION requires COHERENCE before subsequent operations",
        "solution": "Place COHERENCE after MUTATION before other operators",
        "example_fix": "... → DISSONANCE → MUTATION → COHERENCE → ...",
    },
    "self_org_requirements": {
        "error": "self_organization requires destabilizer in previous 3 operators",
        "explanation": "SELF_ORGANIZATION needs preceding destabilization",
        "solution": "Ensure DISSONANCE, EXPANSION, or TRANSITION appears within 3 operators before SELF_ORGANIZATION",
        "example_fix": "... → DISSONANCE → SELF_ORGANIZATION → ...",
    },
    "recursivity_ending": {
        "error": "silence incompatible after recursivity",
        "explanation": "RECURSIVITY cannot be directly followed by SILENCE",
        "solution": "Add COHERENCE after RECURSIVITY before SILENCE, or use TRANSITION instead",
        "example_fix": "... → RECURSIVITY → COHERENCE → SILENCE",
    },
}


def get_pitfall_solution(error_message: str) -> Dict[str, str]:
    """Get solution for a common pitfall based on error message.
    
    Parameters
    ----------
    error_message : str
        Error message from validation
        
    Returns
    -------
    dict
        Solution information including explanation and example fix
    """
    for pitfall_id, info in COMMON_PITFALLS.items():
        if info["error"].lower() in error_message.lower():
            return info
    
    return {
        "error": error_message,
        "explanation": "Error not in common pitfalls database",
        "solution": "Review canonical grammar rules and operator compatibility",
        "example_fix": "Refer to CANONICAL_RULES documentation",
    }


# =============================================================================
# SEQUENCE VALIDATION AND DIAGNOSIS
# =============================================================================


def validate_and_diagnose(sequence: List[str]) -> Dict[str, Any]:
    """Validate a sequence and provide detailed diagnosis.
    
    Parameters
    ----------
    sequence : List[str]
        Operator sequence to validate
        
    Returns
    -------
    dict
        Comprehensive diagnosis including:
        - validation result
        - health metrics (if valid)
        - pattern detected
        - detailed diagnosis
        - recommendations
    """
    result = validate_sequence_with_health(sequence)
    
    diagnosis = {
        "passed": result.passed,
        "sequence": sequence,
        "message": result.message,
    }
    
    if result.passed and result.health_metrics:
        diagnosis["health"] = result.health_metrics.overall_health
        diagnosis["pattern"] = result.health_metrics.dominant_pattern
        diagnosis["interpretation"] = interpret_health_metrics(
            result.health_metrics.overall_health,
            result.health_metrics.dominant_pattern
        )
        diagnosis["recommendations"] = result.health_metrics.recommendations
    else:
        diagnosis["health"] = 0.0
        diagnosis["pattern"] = "invalid"
        pitfall = get_pitfall_solution(result.message)
        diagnosis["interpretation"] = f"VALIDATION FAILED\n\nError: {result.message}\n\nExplanation: {pitfall['explanation']}\n\nSolution: {pitfall['solution']}\n\nExample Fix: {pitfall['example_fix']}"
        diagnosis["recommendations"] = [pitfall["solution"]]
    
    return diagnosis


# =============================================================================
# INTERACTIVE CONSTRUCTION HELPER
# =============================================================================


def suggest_next_operators(current_sequence: List[str]) -> Dict[str, List[str]]:
    """Suggest next valid operators for a partial sequence.
    
    Parameters
    ----------
    current_sequence : List[str]
        Partial operator sequence
        
    Returns
    -------
    dict
        Categorized suggestions:
        - recommended: Best next operators
        - valid: Other valid options
        - invalid: Common invalid choices to avoid
    """
    if not current_sequence:
        return {
            "recommended": [EMISSION],
            "valid": [RECURSIVITY],
            "invalid": ["All others (must start with EMISSION or RECURSIVITY)"],
        }
    
    last_op = current_sequence[-1]
    
    # Simplified suggestions based on last operator
    suggestions = {
        EMISSION: {
            "recommended": [RECEPTION],
            "valid": [COUPLING, COHERENCE],
            "invalid": [EXPANSION, "Cannot skip to complex operators"],
        },
        RECEPTION: {
            "recommended": [COHERENCE],
            "valid": [COUPLING],
            "invalid": [EXPANSION, "Must stabilize with COHERENCE first"],
        },
        COHERENCE: {
            "recommended": [RESONANCE, SILENCE, EXPANSION],
            "valid": [COUPLING, TRANSITION, RECURSIVITY, DISSONANCE, CONTRACTION],
            "invalid": ["Most operators valid after COHERENCE"],
        },
        EXPANSION: {
            "recommended": [COHERENCE],
            "valid": [],
            "invalid": [DISSONANCE, CONTRACTION, MUTATION, "Must stabilize first"],
        },
        CONTRACTION: {
            "recommended": [COHERENCE],
            "valid": [],
            "invalid": [MUTATION, "Must stabilize before transformation"],
        },
        DISSONANCE: {
            "recommended": [MUTATION, SELF_ORGANIZATION],
            "valid": [CONTRACTION, COHERENCE],
            "invalid": [],
        },
        MUTATION: {
            "recommended": [COHERENCE],
            "valid": [],
            "invalid": [CONTRACTION, SELF_ORGANIZATION, "Must stabilize first"],
        },
        RECURSIVITY: {
            "recommended": [COHERENCE, TRANSITION],
            "valid": [],
            "invalid": [SILENCE, "Cannot end directly with SILENCE"],
        },
    }
    
    return suggestions.get(last_op, {
        "recommended": [COHERENCE, SILENCE],
        "valid": ["Context-dependent"],
        "invalid": ["Refer to compatibility matrix"],
    })


if __name__ == "__main__":
    """Interactive pattern construction guide."""
    print("=" * 70)
    print("TNFR Pattern Construction Guide")
    print("=" * 70)
    print()
    
    print("1. Canonical Grammar Rules")
    print("-" * 70)
    print(CANONICAL_RULES)
    print()
    
    print("2. Example: Validating and Diagnosing a Sequence")
    print("-" * 70)
    test_sequence = [EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, SILENCE]
    diagnosis = validate_and_diagnose(test_sequence)
    print(f"Sequence: {' → '.join(test_sequence)}")
    print(f"\nValidation: {'PASS' if diagnosis['passed'] else 'FAIL'}")
    print(f"\n{diagnosis['interpretation']}")
    print()
    
    print("3. Pattern-Specific Strategies")
    print("-" * 70)
    for pattern_name in ["BOOTSTRAP", "EXPLORE", "STABILIZE"]:
        strategy = get_pattern_strategy(pattern_name)
        print(f"\n{pattern_name}:")
        print(f"  Purpose: {strategy['purpose']}")
        print(f"  Min/Max operators: {strategy.get('min_operators', '?')}-{strategy.get('max_operators', '?')}")
        print(f"  Critical elements:")
        for element in strategy.get('critical_elements', []):
            print(f"    - {element}")
    print()
    
    print("=" * 70)
    print("For detailed operator compatibility, use:")
    print("  get_operator_compatibility_notes()")
    print("For interactive suggestions, use:")
    print("  suggest_next_operators(current_sequence)")
    print("=" * 70)
