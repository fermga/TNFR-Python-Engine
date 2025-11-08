"""Business domain pattern definitions."""

from ..base import PatternDefinition

# Change Management Pattern
CHANGE_MANAGEMENT = PatternDefinition(
    name="change_management",
    sequence=["emission", "dissonance", "coupling", "self_organization", "coherence"],
    description="Managing organizational change and transformation",
    use_cases=[
        "Digital transformation initiative",
        "Restructuring and reorganization",
        "Cultural change program",
    ],
    health_requirements={
        "min_coherence": 0.75,
        "min_sense_index": 0.70,
    },
    domain_context={
        "real_world_mapping": (
            "Follows Kotter's change model: emission (create urgency), "
            "dissonance (challenge status quo), coupling (build coalition), "
            "self_organization (empower action), coherence (consolidate gains)"
        ),
        "expected_outcomes": (
            "Successful change adoption, minimal resistance, "
            "sustainable new practices embedded in culture"
        ),
        "failure_modes": (
            "Skipping dissonance leads to superficial change, "
            "insufficient coupling causes isolated pockets of change, "
            "lack of coherence results in change reversal"
        ),
    },
    examples=[
        {
            "name": "Digital Transformation",
            "context": "Company-wide adoption of new digital tools and processes",
            "sequence": [
                "emission",
                "dissonance",
                "coupling",
                "self_organization",
                "coherence",
            ],
            "health_metrics": {"C_t": 0.81, "Si": 0.77},
        },
        {
            "name": "Cultural Shift",
            "context": "Moving from hierarchical to agile culture",
            "sequence": [
                "emission",
                "dissonance",
                "coupling",
                "self_organization",
                "coherence",
            ],
            "health_metrics": {"C_t": 0.79, "Si": 0.75},
        },
        {
            "name": "Merger Integration",
            "context": "Integrating two organizational cultures post-merger",
            "sequence": [
                "emission",
                "dissonance",
                "coupling",
                "self_organization",
                "coherence",
            ],
            "health_metrics": {"C_t": 0.83, "Si": 0.79},
        },
    ],
)

# Workflow Optimization Pattern
WORKFLOW_OPTIMIZATION = PatternDefinition(
    name="workflow_optimization",
    sequence=["reception", "contraction", "coupling", "resonance"],
    description="Optimizing business processes and workflows",
    use_cases=[
        "Process improvement initiative",
        "Lean/Six Sigma implementation",
        "Bottleneck elimination",
    ],
    health_requirements={
        "min_coherence": 0.75,
        "min_sense_index": 0.70,
    },
    domain_context={
        "real_world_mapping": (
            "Process optimization cycle: reception (understand current state), "
            "contraction (eliminate waste), coupling (integrate improvements), "
            "resonance (align with organizational flow)"
        ),
        "expected_outcomes": (
            "Reduced cycle time, improved efficiency, "
            "better resource utilization, sustained improvements"
        ),
        "failure_modes": (
            "Premature contraction without understanding creates new problems, "
            "lack of resonance causes local optimization at expense of system"
        ),
    },
    examples=[
        {
            "name": "Supply Chain Optimization",
            "context": "Reducing lead times and inventory costs",
            "sequence": ["reception", "contraction", "coupling", "resonance"],
            "health_metrics": {"C_t": 0.84, "Si": 0.80},
        },
        {
            "name": "Sales Process Streamlining",
            "context": "Reducing steps from lead to close",
            "sequence": ["reception", "contraction", "coupling", "resonance"],
            "health_metrics": {"C_t": 0.82, "Si": 0.78},
        },
        {
            "name": "Customer Service Improvement",
            "context": "Reducing response time and improving satisfaction",
            "sequence": ["reception", "contraction", "coupling", "resonance"],
            "health_metrics": {"C_t": 0.86, "Si": 0.81},
        },
    ],
)

# Team Alignment Pattern
TEAM_ALIGNMENT = PatternDefinition(
    name="team_alignment",
    sequence=["emission", "reception", "coupling", "resonance", "coherence"],
    description="Aligning team members around shared goals",
    use_cases=[
        "New team formation",
        "Cross-functional collaboration",
        "Strategic alignment meeting",
    ],
    health_requirements={
        "min_coherence": 0.75,
        "min_sense_index": 0.70,
    },
    domain_context={
        "real_world_mapping": (
            "Team alignment process: emission (state vision), "
            "reception (hear perspectives), coupling (find common ground), "
            "resonance (build momentum), coherence (commit to action)"
        ),
        "expected_outcomes": (
            "Shared understanding of goals, coordinated action, "
            "mutual support, high team performance"
        ),
        "failure_modes": (
            "Skipping reception leads to superficial agreement, "
            "lack of coupling causes siloed action, "
            "missing coherence results in misalignment over time"
        ),
    },
    examples=[
        {
            "name": "Strategic Planning Session",
            "context": "Leadership team aligning on annual strategy",
            "sequence": ["emission", "reception", "coupling", "resonance", "coherence"],
            "health_metrics": {"C_t": 0.85, "Si": 0.82},
        },
        {
            "name": "Cross-Functional Project Kickoff",
            "context": "Multiple departments starting shared initiative",
            "sequence": ["emission", "reception", "coupling", "resonance", "coherence"],
            "health_metrics": {"C_t": 0.80, "Si": 0.76},
        },
        {
            "name": "Team Norming",
            "context": "New team establishing working agreements",
            "sequence": ["emission", "reception", "coupling", "resonance", "coherence"],
            "health_metrics": {"C_t": 0.83, "Si": 0.79},
        },
    ],
)

# Collect all patterns
PATTERNS = {
    "change_management": CHANGE_MANAGEMENT,
    "workflow_optimization": WORKFLOW_OPTIMIZATION,
    "team_alignment": TEAM_ALIGNMENT,
}
