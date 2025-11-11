"""TNFR Organizational Diagnostics - Health Assessment and Intervention Tools.

This module provides diagnostic tools for organizational health assessment,
dysfunction detection, and intervention recommendations based on TNFR
structural principles.

Key capabilities:
- Map TNFR health metrics to organizational KPIs
- Detect structural dysfunctions in organizations
- Recommend interventions based on health analysis
- Monitor organizational health over time
"""

from typing import Dict, List, Any
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
# ORGANIZATIONAL KPI MAPPING
# =============================================================================


def map_health_to_organizational_kpis(health_metrics) -> Dict[str, Any]:
    """Map TNFR health metrics to organizational KPIs.
    
    Parameters
    ----------
    health_metrics : SequenceHealthMetrics
        Health metrics from validated sequence
    
    Returns
    -------
    Dict[str, Any]
        Organizational KPIs derived from structural health metrics
    
    Examples
    --------
    >>> from tnfr.operators.grammar import validate_sequence_with_health
    >>> result = validate_sequence_with_health(["emission", "reception", "coherence", "silence"])
    >>> kpis = map_health_to_organizational_kpis(result.health_metrics)
    >>> kpis["strategic_alignment"]  # Based on coherence_index
    0.943
    """
    # Map structural metrics to organizational concepts
    kpis = {
        # Strategic alignment (coherence_index)
        "strategic_alignment": health_metrics.coherence_index,
        "alignment_rating": _interpret_alignment(health_metrics.coherence_index),
        
        # Stability vs Agility (balance_score)
        "stability_agility_balance": health_metrics.balance_score,
        "balance_rating": _interpret_balance(health_metrics.balance_score),
        
        # Institutional resilience (sustainability_index)
        "institutional_resilience": health_metrics.sustainability_index,
        "resilience_rating": _interpret_resilience(health_metrics.sustainability_index),
        
        # Operational efficiency (complexity_efficiency)
        "operational_efficiency": health_metrics.complexity_efficiency,
        "efficiency_rating": _interpret_efficiency(health_metrics.complexity_efficiency),
        
        # Overall organizational health
        "overall_health": health_metrics.overall_health,
        "health_rating": _interpret_overall_health(health_metrics.overall_health),
        
        # Pattern indicates organizational maturity
        "organizational_pattern": health_metrics.dominant_pattern,
    }
    
    return kpis


def _interpret_alignment(score: float) -> str:
    """Interpret strategic alignment score."""
    if score >= 0.9:
        return "Excellent - Strong strategic coherence"
    elif score >= 0.75:
        return "Good - Adequate strategic alignment"
    elif score >= 0.6:
        return "Fair - Some misalignment present"
    else:
        return "Poor - Significant misalignment"


def _interpret_balance(score: float) -> str:
    """Interpret stability-agility balance score."""
    if score >= 0.8:
        return "Excellent - Healthy balance"
    elif score >= 0.6:
        return "Good - Slight imbalance manageable"
    elif score >= 0.4:
        return "Fair - Imbalance creating tension"
    else:
        return "Poor - Severe imbalance"


def _interpret_resilience(score: float) -> str:
    """Interpret institutional resilience score."""
    if score >= 0.8:
        return "Excellent - Highly resilient"
    elif score >= 0.6:
        return "Good - Adequate resilience"
    elif score >= 0.4:
        return "Fair - Vulnerable to disruption"
    else:
        return "Poor - Low resilience"


def _interpret_efficiency(score: float) -> str:
    """Interpret operational efficiency score."""
    if score >= 0.85:
        return "Excellent - Highly efficient"
    elif score >= 0.7:
        return "Good - Adequate efficiency"
    elif score >= 0.55:
        return "Fair - Some inefficiency"
    else:
        return "Poor - Significant inefficiency"


def _interpret_overall_health(score: float) -> str:
    """Interpret overall organizational health."""
    if score >= 0.85:
        return "Excellent - Thriving organization"
    elif score >= 0.75:
        return "Good - Healthy organization"
    elif score >= 0.6:
        return "Fair - Some health concerns"
    else:
        return "Poor - Significant health issues"


# =============================================================================
# DYSFUNCTION DETECTION
# =============================================================================


def detect_structural_dysfunctions(sequence: List[str], health_metrics) -> List[Dict[str, Any]]:
    """Detect structural dysfunctions in organizational sequence.
    
    Parameters
    ----------
    sequence : List[str]
        Operator sequence representing organizational process
    health_metrics : SequenceHealthMetrics
        Health metrics from validated sequence
    
    Returns
    -------
    List[Dict[str, Any]]
        List of detected dysfunctions with descriptions and severities
    
    Examples
    --------
    >>> dysfunctions = detect_structural_dysfunctions(
    ...     ["emission", "dissonance", "dissonance", "transition"],
    ...     health_metrics
    ... )
    >>> len(dysfunctions) > 0  # Multiple dissonances detected
    True
    """
    dysfunctions = []
    
    # Detect excessive rigidity (too many stabilizers)
    stabilizers = sum(1 for op in sequence if op in [COHERENCE, SILENCE, RESONANCE])
    destabilizers = sum(1 for op in sequence if op in [DISSONANCE, MUTATION, EXPANSION])
    
    if stabilizers > destabilizers * 2 and len(sequence) > 5:
        dysfunctions.append({
            "type": "Excessive Rigidity",
            "severity": "Medium",
            "description": "Too many stabilizing operators relative to transformative ones",
            "impact": "Organization may resist necessary change",
            "operators": f"{stabilizers} stabilizers vs {destabilizers} destabilizers",
        })
    
    # Detect excessive chaos (too many destabilizers)
    if destabilizers > stabilizers * 1.5 and len(sequence) > 5:
        dysfunctions.append({
            "type": "Excessive Chaos",
            "severity": "High",
            "description": "Too many destabilizing operators without sufficient stabilization",
            "impact": "Organization may experience instability and burnout",
            "operators": f"{destabilizers} destabilizers vs {stabilizers} stabilizers",
        })
    
    # Detect low balance score
    if health_metrics.balance_score < 0.4:
        dysfunctions.append({
            "type": "Structural Imbalance",
            "severity": "High",
            "description": f"Balance score critically low: {health_metrics.balance_score:.2f}",
            "impact": "Severe imbalance between stability and transformation",
            "recommendation": "Add balancing operators (stabilizers if chaotic, destabilizers if rigid)",
        })
    
    # Detect low sustainability
    if health_metrics.sustainability_index < 0.5:
        dysfunctions.append({
            "type": "Low Sustainability",
            "severity": "Medium",
            "description": f"Sustainability index low: {health_metrics.sustainability_index:.2f}",
            "impact": "Changes may not persist; risk of regression",
            "recommendation": "End with stabilizers (SILENCE, COHERENCE, RECURSIVITY)",
        })
    
    # Detect missing closure
    if sequence and sequence[-1] not in [SILENCE, TRANSITION, RECURSIVITY, DISSONANCE]:
        dysfunctions.append({
            "type": "Incomplete Closure",
            "severity": "Low",
            "description": f"Sequence ends with {sequence[-1]} instead of closure operator",
            "impact": "Initiative may lack proper conclusion",
            "recommendation": "Consider ending with SILENCE, TRANSITION, or RECURSIVITY",
        })
    
    # Detect repeated dissonance without resolution
    dissonance_count = sequence.count(DISSONANCE)
    if dissonance_count > 2:
        dysfunctions.append({
            "type": "Unresolved Tensions",
            "severity": "Medium",
            "description": f"Multiple dissonances ({dissonance_count}) may indicate unresolved conflicts",
            "impact": "Ongoing organizational tension and conflict",
            "recommendation": "Add SELF_ORGANIZATION or MUTATION for transformation",
        })
    
    # Detect low operational efficiency
    if health_metrics.complexity_efficiency < 0.7:
        dysfunctions.append({
            "type": "Operational Inefficiency",
            "severity": "Low",
            "description": f"Complexity efficiency low: {health_metrics.complexity_efficiency:.2f}",
            "impact": "Process may be overly complex for intended outcome",
            "recommendation": "Simplify sequence or add CONTRACTION operator",
        })
    
    return dysfunctions


# =============================================================================
# INTERVENTION RECOMMENDATIONS
# =============================================================================


def recommend_interventions(sequence: List[str], health_metrics, dysfunctions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recommend structural interventions based on health analysis.
    
    Parameters
    ----------
    sequence : List[str]
        Current organizational operator sequence
    health_metrics : SequenceHealthMetrics
        Health metrics from validated sequence
    dysfunctions : List[Dict[str, Any]]
        Detected dysfunctions
    
    Returns
    -------
    List[Dict[str, Any]]
        Prioritized intervention recommendations
    
    Examples
    --------
    >>> interventions = recommend_interventions(sequence, health_metrics, dysfunctions)
    >>> interventions[0]["priority"]  # Highest priority first
    'High'
    """
    interventions = []
    
    # Prioritize based on dysfunction severity
    high_severity = [d for d in dysfunctions if d["severity"] == "High"]
    medium_severity = [d for d in dysfunctions if d["severity"] == "Medium"]
    
    # High priority interventions
    if health_metrics.overall_health < 0.6:
        interventions.append({
            "priority": "High",
            "intervention": "Comprehensive Organizational Health Assessment",
            "description": "Overall health critically low - conduct full diagnostic",
            "actions": [
                "Engage leadership for health assessment",
                "Survey organizational stakeholders",
                "Identify root causes of low health",
                "Develop comprehensive improvement plan",
            ],
        })
    
    if health_metrics.balance_score < 0.4:
        if sum(1 for op in sequence if op in [DISSONANCE, MUTATION]) > sum(1 for op in sequence if op in [COHERENCE, SILENCE]):
            interventions.append({
                "priority": "High",
                "intervention": "Add Stabilization Operators",
                "description": "Severe imbalance toward destabilization",
                "suggested_operators": [COHERENCE, SILENCE, RESONANCE],
                "actions": [
                    "Consolidate recent changes before new initiatives",
                    "Create stability anchors (team rituals, clear processes)",
                    "Pause new disruptive changes temporarily",
                ],
            })
        else:
            interventions.append({
                "priority": "High",
                "intervention": "Introduce Controlled Transformation",
                "description": "Excessive rigidity preventing necessary adaptation",
                "suggested_operators": [DISSONANCE, EXPANSION, SELF_ORGANIZATION],
                "actions": [
                    "Create safe spaces for innovation and experimentation",
                    "Empower teams to self-organize solutions",
                    "Challenge status quo constructively",
                ],
            })
    
    # Medium priority interventions
    if health_metrics.sustainability_index < 0.6:
        interventions.append({
            "priority": "Medium",
            "intervention": "Strengthen Change Consolidation",
            "description": "Low sustainability - changes may not persist",
            "suggested_operators": [SILENCE, RECURSIVITY, COHERENCE],
            "actions": [
                "Build in reflection and learning periods",
                "Embed changes into systems and processes",
                "Create feedback loops to monitor persistence",
            ],
        })
    
    if SELF_ORGANIZATION in sequence and CONTRACTION not in sequence and SILENCE not in sequence:
        interventions.append({
            "priority": "Medium",
            "intervention": "Close Self-Organization Phase",
            "description": "Self-organization requires proper closure",
            "suggested_operators": [CONTRACTION, SILENCE],
            "actions": [
                "Focus autonomous teams on viable solutions",
                "Harvest learnings from self-organization",
                "Consolidate emergent practices",
            ],
        })
    
    # Low priority interventions
    if health_metrics.complexity_efficiency < 0.75:
        interventions.append({
            "priority": "Low",
            "intervention": "Simplify Process",
            "description": "Operational inefficiency detected",
            "suggested_operators": [CONTRACTION],
            "actions": [
                "Identify and remove unnecessary steps",
                "Streamline decision-making processes",
                "Focus on essential activities",
            ],
        })
    
    # Sort by priority
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    interventions.sort(key=lambda x: priority_order[x["priority"]])
    
    return interventions


# =============================================================================
# COMPREHENSIVE DIAGNOSTIC REPORT
# =============================================================================


def generate_diagnostic_report(sequence: List[str]) -> Dict[str, Any]:
    """Generate comprehensive organizational diagnostic report.
    
    Parameters
    ----------
    sequence : List[str]
        Organizational operator sequence to analyze
    
    Returns
    -------
    Dict[str, Any]
        Complete diagnostic report with KPIs, dysfunctions, and recommendations
    
    Examples
    --------
    >>> sequence = ["emission", "reception", "coherence", "dissonance", "transition"]
    >>> report = generate_diagnostic_report(sequence)
    >>> report["overall_health_rating"]
    'Good - Healthy organization'
    """
    # Validate and get health metrics
    result = validate_sequence_with_health(sequence)
    
    if not result.passed:
        return {
            "valid": False,
            "error": result.message,
        }
    
    health_metrics = result.health_metrics
    
    # Generate components
    kpis = map_health_to_organizational_kpis(health_metrics)
    dysfunctions = detect_structural_dysfunctions(sequence, health_metrics)
    interventions = recommend_interventions(sequence, health_metrics, dysfunctions)
    
    return {
        "valid": True,
        "sequence": sequence,
        "sequence_length": len(sequence),
        
        # Health metrics
        "overall_health": health_metrics.overall_health,
        "overall_health_rating": kpis["health_rating"],
        
        # KPIs
        "kpis": kpis,
        
        # Dysfunctions
        "dysfunctions_detected": len(dysfunctions),
        "dysfunctions": dysfunctions,
        
        # Interventions
        "interventions_recommended": len(interventions),
        "interventions": interventions,
        
        # Summary
        "summary": {
            "health_status": "Healthy" if health_metrics.overall_health >= 0.75 else "Needs Attention",
            "critical_issues": len([d for d in dysfunctions if d["severity"] == "High"]),
            "high_priority_actions": len([i for i in interventions if i["priority"] == "High"]),
        },
    }


def print_diagnostic_report(report: Dict[str, Any]):
    """Print diagnostic report in human-readable format.
    
    Parameters
    ----------
    report : Dict[str, Any]
        Diagnostic report from generate_diagnostic_report
    """
    if not report["valid"]:
        print(f"\n✗ Invalid sequence: {report['error']}")
        return
    
    print("\n" + "=" * 70)
    print("ORGANIZATIONAL DIAGNOSTIC REPORT")
    print("=" * 70)
    
    print(f"\nSequence: {' → '.join(report['sequence'])}")
    print(f"Length: {report['sequence_length']} operators")
    
    print(f"\n--- Overall Health ---")
    print(f"Health Score: {report['overall_health']:.3f}")
    print(f"Rating: {report['overall_health_rating']}")
    print(f"Status: {report['summary']['health_status']}")
    
    print(f"\n--- Organizational KPIs ---")
    kpis = report['kpis']
    print(f"Strategic Alignment:  {kpis['strategic_alignment']:.3f} - {kpis['alignment_rating']}")
    print(f"Stability-Agility:    {kpis['stability_agility_balance']:.3f} - {kpis['balance_rating']}")
    print(f"Inst. Resilience:     {kpis['institutional_resilience']:.3f} - {kpis['resilience_rating']}")
    print(f"Operational Efficiency: {kpis['operational_efficiency']:.3f} - {kpis['efficiency_rating']}")
    
    if report['dysfunctions']:
        print(f"\n--- Dysfunctions Detected ({report['dysfunctions_detected']}) ---")
        for i, dysfunction in enumerate(report['dysfunctions'], 1):
            print(f"\n{i}. {dysfunction['type']} [{dysfunction['severity']} Severity]")
            print(f"   {dysfunction['description']}")
            print(f"   Impact: {dysfunction['impact']}")
            if "recommendation" in dysfunction:
                print(f"   → {dysfunction['recommendation']}")
    else:
        print(f"\n--- No Dysfunctions Detected ---")
        print("Organization appears structurally sound.")
    
    if report['interventions']:
        print(f"\n--- Recommended Interventions ({report['interventions_recommended']}) ---")
        for i, intervention in enumerate(report['interventions'], 1):
            print(f"\n{i}. [{intervention['priority']} Priority] {intervention['intervention']}")
            print(f"   {intervention['description']}")
            if "actions" in intervention:
                print("   Actions:")
                for action in intervention["actions"]:
                    print(f"     • {action}")
    else:
        print(f"\n--- No Interventions Needed ---")
        print("Organization is healthy. Continue current approach.")
    
    print("\n" + "=" * 70)


def main():
    """Demonstrate organizational diagnostics capabilities."""
    print("\n" + "=" * 70)
    print("TNFR ORGANIZATIONAL DIAGNOSTICS DEMONSTRATION")
    print("=" * 70)
    
    # Example 1: Healthy organization
    print("\n\nExample 1: Strategic Planning (Healthy)")
    print("-" * 70)
    healthy_sequence = [
        EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION,
        CONTRACTION, COHERENCE, EXPANSION, COUPLING, RESONANCE, TRANSITION
    ]
    report1 = generate_diagnostic_report(healthy_sequence)
    print_diagnostic_report(report1)
    
    # Example 2: Imbalanced organization (too much chaos)
    print("\n\nExample 2: Change-Fatigued Organization (Too Much Change)")
    print("-" * 70)
    chaotic_sequence = [
        EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, COHERENCE, 
        DISSONANCE, MUTATION, TRANSITION
    ]
    report2 = generate_diagnostic_report(chaotic_sequence)
    print_diagnostic_report(report2)


if __name__ == "__main__":
    main()
