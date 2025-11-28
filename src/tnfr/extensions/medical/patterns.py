"""Medical domain pattern definitions."""

from ..base import PatternDefinition

# Therapeutic Alliance Pattern
THERAPEUTIC_ALLIANCE = PatternDefinition(
    name="therapeutic_alliance",
    sequence=["emission", "reception", "coherence", "resonance"],
    description="Establishing therapeutic trust and rapport",
    use_cases=[
        "Initial therapy session - building connection",
        "Re-establishing alliance after rupture",
        "Deepening therapeutic relationship",
    ],
    health_requirements={
        "min_coherence": 0.75,
        "min_sense_index": 0.70,
    },
    domain_context={
        "real_world_mapping": (
            "Maps to Carl Rogers' therapeutic alliance concept: "
            "emission (therapist presence), reception (active listening), "
            "coherence (mutual understanding), resonance (empathic attunement)"
        ),
        "expected_outcomes": (
            "Strong working alliance, patient feels heard and understood, "
            "foundation for therapeutic work established"
        ),
        "failure_modes": (
            "Premature coherence without genuine reception can feel inauthentic, "
            "lack of resonance prevents deeper connection"
        ),
    },
    examples=[
        {
            "name": "Initial Session - Trust Building",
            "context": "First meeting with new patient, establishing safety",
            "sequence": ["emission", "reception", "coherence", "resonance"],
            "health_metrics": {"C_t": 0.82, "Si": 0.76},
        },
        {
            "name": "Alliance Repair",
            "context": "After misunderstanding, rebuilding connection",
            "sequence": ["emission", "reception", "coherence", "resonance"],
            "health_metrics": {"C_t": 0.79, "Si": 0.74},
        },
        {
            "name": "Deepening Phase",
            "context": "Moving from surface to deeper therapeutic work",
            "sequence": ["emission", "reception", "coherence", "resonance"],
            "health_metrics": {"C_t": 0.85, "Si": 0.81},
        },
    ],
)

# Crisis Intervention Pattern
CRISIS_INTERVENTION = PatternDefinition(
    name="crisis_intervention",
    sequence=["dissonance", "silence", "coherence", "resonance"],
    description="Stabilizing acute emotional distress",
    use_cases=[
        "Acute anxiety or panic attack",
        "Emotional overwhelm during session",
        "Crisis situation requiring immediate stabilization",
    ],
    health_requirements={
        "min_coherence": 0.75,
        "min_sense_index": 0.70,
    },
    domain_context={
        "real_world_mapping": (
            "Follows crisis intervention model: dissonance (acknowledge distress), "
            "silence (create space/pause), coherence (stabilization techniques), "
            "resonance (empathic grounding)"
        ),
        "expected_outcomes": (
            "Reduced emotional intensity, restored sense of safety, "
            "ability to engage in problem-solving"
        ),
        "failure_modes": (
            "Skipping silence phase can escalate crisis, "
            "insufficient coherence leaves patient dysregulated"
        ),
    },
    examples=[
        {
            "name": "Panic Attack Intervention",
            "context": "Patient experiencing acute panic in session",
            "sequence": ["dissonance", "silence", "coherence", "resonance"],
            "health_metrics": {"C_t": 0.78, "Si": 0.77},
        },
        {
            "name": "Emotional Overwhelm",
            "context": "Patient flooded with difficult emotions",
            "sequence": ["dissonance", "silence", "coherence", "resonance"],
            "health_metrics": {"C_t": 0.81, "Si": 0.79},
        },
        {
            "name": "Acute Grief Response",
            "context": "Managing intense grief reaction",
            "sequence": ["dissonance", "silence", "coherence", "resonance"],
            "health_metrics": {"C_t": 0.76, "Si": 0.75},
        },
    ],
)

# Integration Phase Pattern
INTEGRATION_PHASE = PatternDefinition(
    name="integration_phase",
    sequence=["coupling", "self_organization", "expansion", "coherence"],
    description="Integrating insights and new perspectives",
    use_cases=[
        "After breakthrough moment - consolidating learning",
        "Connecting disparate experiences into coherent narrative",
        "Expanding awareness to include new perspectives",
    ],
    health_requirements={
        "min_coherence": 0.75,
        "min_sense_index": 0.70,
    },
    domain_context={
        "real_world_mapping": (
            "Reflects integration process in therapy: coupling (connecting elements), "
            "self_organization (natural meaning-making), expansion (broadening view), "
            "coherence (consolidating new understanding)"
        ),
        "expected_outcomes": (
            "Integrated self-narrative, expanded perspective, "
            "sustainable new patterns of thinking/feeling"
        ),
        "failure_modes": (
            "Premature expansion without coupling can be destabilizing, "
            "lack of final coherence leaves insights fragmented"
        ),
    },
    examples=[
        {
            "name": "Post-Breakthrough Integration",
            "context": "After major insight, integrating into broader understanding",
            "sequence": ["coupling", "self_organization", "expansion", "coherence"],
            "health_metrics": {"C_t": 0.84, "Si": 0.80},
        },
        {
            "name": "Narrative Coherence",
            "context": "Building coherent life story from fragmented experiences",
            "sequence": ["coupling", "self_organization", "expansion", "coherence"],
            "health_metrics": {"C_t": 0.83, "Si": 0.78},
        },
        {
            "name": "Perspective Expansion",
            "context": "Including previously rejected aspects of self",
            "sequence": ["coupling", "self_organization", "expansion", "coherence"],
            "health_metrics": {"C_t": 0.86, "Si": 0.82},
        },
    ],
)

# Collect all patterns
PATTERNS = {
    "therapeutic_alliance": THERAPEUTIC_ALLIANCE,
    "crisis_intervention": CRISIS_INTERVENTION,
    "integration_phase": INTEGRATION_PHASE,
}
