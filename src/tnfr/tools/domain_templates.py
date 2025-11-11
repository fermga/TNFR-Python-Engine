"""Domain-specific operator sequence templates for TNFR applications.

This module provides curated, validated operator sequences organized by
application domain (therapeutic, educational, organizational, creative).
Each template is designed to achieve specific objectives within its domain
while maintaining structural health and coherence.

All templates follow TNFR canonical principles:
- Operator closure (only canonical operators)
- Phase coherence (compatible transitions)
- Structural health (balanced forces, proper closure)
- Operational fractality (composable patterns)

Examples
--------
>>> from tnfr.tools.domain_templates import DOMAIN_TEMPLATES
>>> crisis_seq = DOMAIN_TEMPLATES["therapeutic"]["crisis_intervention"]
>>> print(crisis_seq)
['emission', 'reception', 'coherence', 'resonance', 'silence']
"""

from __future__ import annotations

from ..config.operator_names import (
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

__all__ = [
    "DOMAIN_TEMPLATES",
    "get_template",
    "list_domains",
    "list_objectives",
]


# =============================================================================
# THERAPEUTIC DOMAIN - Healing and Personal Transformation
# =============================================================================

THERAPEUTIC_TEMPLATES = {
    "crisis_intervention": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE],
        "description": "Rapid stabilization for immediate crisis response",
        "expected_health": 0.75,
        "pattern": "STABILIZE",
        "characteristics": [
            "Fast response",
            "Emergency containment",
            "Minimal transformation",
            "Quick closure",
        ],
    },
    "process_therapy": {
        "sequence": [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            SELF_ORGANIZATION,
            COHERENCE,
            TRANSITION,
            SILENCE,
        ],
        "description": "Complete transformative therapeutic cycle",
        "expected_health": 0.85,
        "pattern": "THERAPEUTIC",
        "characteristics": [
            "Deep transformation",
            "Controlled crisis exploration",
            "Autonomous reorganization",
            "Sustained integration",
        ],
    },
    "healing_cycle": {
        "sequence": [
            COHERENCE,
            RESONANCE,
            EXPANSION,
            SILENCE,
            TRANSITION,
            COHERENCE,
        ],
        "description": "Gradual healing and integration process",
        "expected_health": 0.78,
        "pattern": "REGENERATIVE",
        "characteristics": [
            "Gentle progression",
            "Capacity building",
            "Reflective pauses",
            "Sustainable growth",
        ],
    },
    "trauma_processing": {
        "sequence": [
            EMISSION,
            RECEPTION,
            COHERENCE,
            SILENCE,
            DISSONANCE,
            CONTRACTION,
            COHERENCE,
            TRANSITION,
            SILENCE,
        ],
        "description": "Safe trauma processing with containment",
        "expected_health": 0.82,
        "pattern": "THERAPEUTIC",
        "characteristics": [
            "Safety first",
            "Controlled exposure",
            "Titrated dissonance",
            "Integration focus",
        ],
    },
}


# =============================================================================
# EDUCATIONAL DOMAIN - Learning and Skill Development
# =============================================================================

EDUCATIONAL_TEMPLATES = {
    "concept_introduction": {
        "sequence": [EMISSION, RECEPTION, COHERENCE, EXPANSION, COHERENCE],
        "description": "Introduce new concepts with exploration",
        "expected_health": 0.72,
        "pattern": "BOOTSTRAP",
        "characteristics": [
            "Clear introduction",
            "Active reception",
            "Conceptual expansion",
            "Consolidation",
        ],
    },
    "skill_development": {
        "sequence": [
            RECEPTION,
            COHERENCE,
            EXPANSION,
            DISSONANCE,
            MUTATION,
            COHERENCE,
        ],
        "description": "Progressive skill building with challenge",
        "expected_health": 0.80,
        "pattern": "EDUCATIONAL",
        "characteristics": [
            "Practice focus",
            "Incremental difficulty",
            "Breakthrough moments",
            "Mastery consolidation",
        ],
    },
    "knowledge_integration": {
        "sequence": [COHERENCE, COUPLING, RESONANCE, RECURSIVITY],
        "description": "Connect and integrate multiple concepts",
        "expected_health": 0.76,
        "pattern": "RESONATE",
        "characteristics": [
            "Connection building",
            "Pattern recognition",
            "Cross-domain links",
            "Recursive understanding",
        ],
    },
    "transformative_learning": {
        "sequence": [
            RECEPTION,
            EMISSION,
            COHERENCE,
            EXPANSION,
            DISSONANCE,
            MUTATION,
            COHERENCE,
            SILENCE,
        ],
        "description": "Deep learning with paradigm shift",
        "expected_health": 0.83,
        "pattern": "EDUCATIONAL",
        "characteristics": [
            "Perspective change",
            "Challenge assumptions",
            "Cognitive restructuring",
            "Integration time",
        ],
    },
}


# =============================================================================
# ORGANIZATIONAL DOMAIN - Institutional Change and Team Dynamics
# =============================================================================

ORGANIZATIONAL_TEMPLATES = {
    "change_management": {
        "sequence": [
            TRANSITION,
            EMISSION,
            RECEPTION,
            COUPLING,
            DISSONANCE,
            SELF_ORGANIZATION,
            COHERENCE,
        ],
        "description": "Organizational transformation process",
        "expected_health": 0.81,
        "pattern": "ORGANIZATIONAL",
        "characteristics": [
            "Clear transition",
            "Stakeholder engagement",
            "Managed tension",
            "Emergent solutions",
        ],
    },
    "team_building": {
        "sequence": [EMISSION, RECEPTION, COUPLING, COHERENCE, RESONANCE],
        "description": "Build cohesive team dynamics",
        "expected_health": 0.78,
        "pattern": "RESONATE",
        "characteristics": [
            "Individual expression",
            "Active listening",
            "Connection building",
            "Shared resonance",
        ],
    },
    "crisis_response": {
        "sequence": [
            DISSONANCE,
            EMISSION,
            RECEPTION,
            COHERENCE,
            TRANSITION,
            SILENCE,
        ],
        "description": "Organizational crisis management",
        "expected_health": 0.77,
        "pattern": "STABILIZE",
        "characteristics": [
            "Acknowledge crisis",
            "Leadership clarity",
            "Rapid stabilization",
            "Strategic pause",
        ],
    },
    "innovation_cycle": {
        "sequence": [
            COHERENCE,
            EXPANSION,
            DISSONANCE,
            MUTATION,
            SELF_ORGANIZATION,
            COHERENCE,
            TRANSITION,
        ],
        "description": "Foster organizational innovation",
        "expected_health": 0.84,
        "pattern": "ORGANIZATIONAL",
        "characteristics": [
            "Stable foundation",
            "Explore possibilities",
            "Creative tension",
            "Emergent innovation",
        ],
    },
}


# =============================================================================
# CREATIVE DOMAIN - Artistic Process and Design
# =============================================================================

CREATIVE_TEMPLATES = {
    "artistic_process": {
        "sequence": [
            SILENCE,
            EMISSION,
            EXPANSION,
            DISSONANCE,
            MUTATION,
            COHERENCE,
            RECURSIVITY,
        ],
        "description": "Creative work from conception to completion",
        "expected_health": 0.82,
        "pattern": "CREATIVE",
        "characteristics": [
            "Contemplative start",
            "Initial expression",
            "Exploratory phase",
            "Iterative refinement",
        ],
    },
    "design_thinking": {
        "sequence": [
            RECEPTION,
            COHERENCE,
            EXPANSION,
            DISSONANCE,
            MUTATION,
            COHERENCE,
        ],
        "description": "Design process from empathy to prototype",
        "expected_health": 0.80,
        "pattern": "CREATIVE",
        "characteristics": [
            "User empathy",
            "Problem definition",
            "Ideation",
            "Prototype iteration",
        ],
    },
    "innovation": {
        "sequence": [
            COHERENCE,
            DISSONANCE,
            EXPANSION,
            SILENCE,
            MUTATION,
            COHERENCE,
        ],
        "description": "Innovation through creative destruction",
        "expected_health": 0.79,
        "pattern": "EXPLORE",
        "characteristics": [
            "Understand constraints",
            "Challenge assumptions",
            "Explore alternatives",
            "Breakthrough moment",
        ],
    },
    "collaborative_creation": {
        "sequence": [
            EMISSION,
            RECEPTION,
            COUPLING,
            RESONANCE,
            EXPANSION,
            SELF_ORGANIZATION,
            COHERENCE,
        ],
        "description": "Group creative process with emergent outcomes",
        "expected_health": 0.83,
        "pattern": "CREATIVE",
        "characteristics": [
            "Individual contributions",
            "Active collaboration",
            "Synergistic amplification",
            "Emergent product",
        ],
    },
}


# =============================================================================
# MASTER TEMPLATE DICTIONARY
# =============================================================================

DOMAIN_TEMPLATES: dict[str, dict[str, dict[str, object]]] = {
    "therapeutic": THERAPEUTIC_TEMPLATES,
    "educational": EDUCATIONAL_TEMPLATES,
    "organizational": ORGANIZATIONAL_TEMPLATES,
    "creative": CREATIVE_TEMPLATES,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_template(domain: str, objective: str | None = None) -> list[str]:
    """Retrieve a template sequence for a specific domain and objective.

    Parameters
    ----------
    domain : str
        Application domain (therapeutic, educational, organizational, creative).
    objective : str, optional
        Specific objective within the domain. If None, returns the first
        template in the domain.

    Returns
    -------
    list[str]
        Operator sequence as canonical operator names.

    Raises
    ------
    KeyError
        If domain or objective not found.

    Examples
    --------
    >>> template = get_template("therapeutic", "crisis_intervention")
    >>> print(template)
    ['emission', 'reception', 'coherence', 'resonance', 'silence']
    """
    if domain not in DOMAIN_TEMPLATES:
        raise KeyError(f"Domain '{domain}' not found. Available: {list(DOMAIN_TEMPLATES.keys())}")

    domain_dict = DOMAIN_TEMPLATES[domain]

    if objective is None:
        # Return first template in domain
        first_key = next(iter(domain_dict.keys()))
        return domain_dict[first_key]["sequence"]  # type: ignore[return-value]

    if objective not in domain_dict:
        raise KeyError(
            f"Objective '{objective}' not found in domain '{domain}'. "
            f"Available: {list(domain_dict.keys())}"
        )

    return domain_dict[objective]["sequence"]  # type: ignore[return-value]


def list_domains() -> list[str]:
    """List all available application domains.

    Returns
    -------
    list[str]
        List of domain names.

    Examples
    --------
    >>> domains = list_domains()
    >>> print(domains)
    ['therapeutic', 'educational', 'organizational', 'creative']
    """
    return list(DOMAIN_TEMPLATES.keys())


def list_objectives(domain: str) -> list[str]:
    """List all objectives available for a specific domain.

    Parameters
    ----------
    domain : str
        Application domain.

    Returns
    -------
    list[str]
        List of objective names for the domain.

    Raises
    ------
    KeyError
        If domain not found.

    Examples
    --------
    >>> objectives = list_objectives("therapeutic")
    >>> print(objectives)
    ['crisis_intervention', 'process_therapy', 'healing_cycle', 'trauma_processing']
    """
    if domain not in DOMAIN_TEMPLATES:
        raise KeyError(f"Domain '{domain}' not found. Available: {list(DOMAIN_TEMPLATES.keys())}")

    return list(DOMAIN_TEMPLATES[domain].keys())
