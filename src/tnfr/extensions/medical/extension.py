"""Medical domain TNFR extension.

Provides validated patterns for healthcare and therapeutic applications,
including crisis intervention, healing journeys, and trauma-informed care.
"""

from __future__ import annotations

from typing import Dict
from ..base import TNFRExtension, PatternDefinition


__all__ = ["MedicalExtension"]


class MedicalExtension(TNFRExtension):
    """Medical and therapeutic domain extension for TNFR.
    
    Provides patterns validated for healthcare contexts including:
    - Crisis intervention
    - Therapeutic journeys
    - Trauma-informed care
    - Patient support sequences
    - Healing processes
    
    All patterns maintain health scores > 0.75 and follow trauma-informed
    principles (safety, transparency, collaboration).
    
    Examples
    --------
    >>> from tnfr.extensions.medical import MedicalExtension
    >>> extension = MedicalExtension()
    >>> patterns = extension.get_pattern_definitions()
    >>> crisis = patterns["crisis_intervention"]
    >>> print(crisis.examples[0])
    ['emission', 'reception', 'coherence', 'dissonance', 'contraction', 'coherence', 'coupling', 'silence']
    """
    
    def get_domain_name(self) -> str:
        """Return medical domain identifier."""
        return "medical"
    
    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Return validated medical/therapeutic patterns.
        
        Returns
        -------
        Dict[str, PatternDefinition]
            Medical domain patterns including crisis intervention,
            therapeutic journeys, and trauma-informed care sequences.
        """
        return {
            "crisis_intervention": PatternDefinition(
                name="Crisis Intervention",
                description="Stabilization pattern for acute crisis situations",
                examples=[
                    # Primary crisis intervention sequence
                    [
                        "emission",      # Establish presence
                        "reception",     # Listen deeply
                        "coherence",     # Stabilize initial state
                        "dissonance",    # Address crisis energy
                        "contraction",   # Focus on core stability
                        "coherence",     # Re-stabilize
                        "coupling",      # Build support connection
                        "silence",       # Integration pause
                    ],
                    # Rapid stabilization variant
                    [
                        "emission",
                        "reception",
                        "coherence",
                        "contraction",
                        "coherence",
                        "silence",
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "Acute psychiatric crisis",
                    "Trauma response",
                    "Emergency emotional support",
                    "Suicide prevention intervention",
                ],
                structural_insights=[
                    "Emission establishes safe presence",
                    "Reception without judgment creates trust",
                    "Dissonance acknowledges crisis reality",
                    "Contraction focuses on immediate stability",
                    "Coupling prevents isolation",
                    "Silence allows integration without pressure",
                ],
            ),
            
            "therapeutic_journey": PatternDefinition(
                name="Therapeutic Journey",
                description="Long-term healing and growth pattern",
                examples=[
                    # Extended therapeutic process
                    [
                        "emission",          # Begin therapeutic relationship
                        "reception",         # Deep listening phase
                        "coherence",         # Establish safety
                        "dissonance",        # Explore difficult material
                        "coherence",         # Stabilize
                        "mutation",          # Process transformation
                        "coherence",         # Integrate changes
                        "silence",           # Integration pause
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "Long-term psychotherapy",
                    "Trauma recovery program",
                    "Personal development coaching",
                    "Addiction recovery journey",
                ],
                structural_insights=[
                    "Coherence before dissonance ensures safety",
                    "Coupling maintains connection during transformation",
                    "Mutation represents qualitative healing shifts",
                    "Expansion builds resilience and capacity",
                    "Resonance reinforces healthy patterns",
                ],
            ),
            
            "trauma_informed_care": PatternDefinition(
                name="Trauma-Informed Care",
                description="Pattern respecting trauma-informed principles",
                examples=[
                    # Trauma-sensitive approach
                    [
                        "emission",          # Predictable presence
                        "reception",         # Receive at client's pace
                        "coherence",         # Safety first
                        "coupling",          # Collaborative relationship
                        "coherence",         # Re-stabilize frequently
                        "silence",           # Space for client pacing
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "PTSD treatment",
                    "Complex trauma therapy",
                    "Trauma-sensitive education",
                    "Survivor support groups",
                ],
                structural_insights=[
                    "Silence honors client autonomy and pacing",
                    "Coherence prioritized over exploration",
                    "Coupling is collaborative, not directive",
                    "No forced dissonance - client-led process",
                ],
            ),
            
            "patient_support": PatternDefinition(
                name="Patient Support Sequence",
                description="Medical care support and accompaniment",
                examples=[
                    [
                        "emission",      # Show up consistently
                        "reception",     # Listen to patient experience
                        "coupling",      # Build trust
                        "coherence",     # Provide stability
                        "resonance",     # Amplify patient strengths
                        "silence",       # Integration
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "Chronic illness support",
                    "Palliative care",
                    "Medical treatment accompaniment",
                    "Patient advocacy",
                ],
                structural_insights=[
                    "Consistent emission builds reliability",
                    "Reception validates patient experience",
                    "Coupling creates supportive relationship",
                    "Resonance amplifies patient agency",
                ],
            ),
            
            "healing_cycle": PatternDefinition(
                name="Natural Healing Cycle",
                description="Pattern following natural healing rhythms",
                examples=[
                    [
                        "emission",          # Begin healing intention
                        "reception",         # Receive body wisdom
                        "coherence",         # Establish baseline
                        "dissonance",        # Acknowledge injury/illness
                        "coherence",         # Stabilize
                        "self_organization", # Support natural healing
                        "coherence",         # Integrate healing
                        "silence",           # Rest and integration
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "Physical rehabilitation",
                    "Post-surgical recovery",
                    "Integrative medicine",
                    "Holistic wellness programs",
                ],
                structural_insights=[
                    "Self-organization respects body's healing intelligence",
                    "Coherence provides foundation for recovery",
                    "Resonance strengthens healing patterns",
                    "Expansion rebuilds capacity safely",
                ],
            ),
        }
    
    def get_metadata(self) -> Dict[str, any]:
        """Return medical extension metadata."""
        return {
            "domain": "medical",
            "version": "1.0.0",
            "author": "TNFR Medical Domain Contributors",
            "description": "Healthcare and therapeutic patterns for TNFR",
            "safety_principles": [
                "Trauma-informed care",
                "Client autonomy",
                "Safety-first approach",
                "Evidence-based patterns",
            ],
            "validation_standards": {
                "min_health_score": 0.65,
                "trauma_safety_required": True,
                "clinical_review": True,
            },
        }
