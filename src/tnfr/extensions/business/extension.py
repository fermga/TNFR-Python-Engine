"""Business domain TNFR extension.

Provides validated patterns for business processes, organizational change,
sales cycles, and team dynamics.
"""

from __future__ import annotations

from typing import Dict
from ..base import TNFRExtension, PatternDefinition


__all__ = ["BusinessExtension"]


class BusinessExtension(TNFRExtension):
    """Business process and organizational domain extension for TNFR.
    
    Provides patterns validated for business contexts including:
    - Sales cycles
    - Organizational change
    - Team formation and development
    - Customer success journeys
    - Innovation processes
    
    All patterns maintain health scores > 0.75 and follow best practices
    from organizational development and business process optimization.
    
    Examples
    --------
    >>> from tnfr.extensions.business import BusinessExtension
    >>> extension = BusinessExtension()
    >>> patterns = extension.get_pattern_definitions()
    >>> sales = patterns["sales_cycle"]
    >>> print(sales.examples[0])
    ['emission', 'reception', 'coupling', 'resonance', 'coherence', 'expansion']
    """
    
    def get_domain_name(self) -> str:
        """Return business domain identifier."""
        return "business"
    
    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Return validated business process patterns.
        
        Returns
        -------
        Dict[str, PatternDefinition]
            Business domain patterns including sales cycles, organizational
            change, team development, and innovation processes.
        """
        return {
            "sales_cycle": PatternDefinition(
                name="B2B Sales Cycle",
                description="Standard enterprise sales progression pattern",
                examples=[
                    # Full B2B sales cycle
                    [
                        "emission",      # Outbound outreach
                        "reception",     # Listen to prospect needs
                        "coupling",      # Build relationship
                        "resonance",     # Align on value
                        "coherence",     # Formalize agreement
                        "silence",       # Close and integrate
                    ],
                    # Consultative sales variant
                    [
                        "emission",
                        "reception",
                        "coherence",     # Establish credibility
                        "dissonance",    # Identify pain points
                        "coherence",     # Stabilize
                        "coupling",
                        "resonance",     # Present solution
                        "coherence",     # Stabilize
                        "silence",       # Close deal
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "Enterprise software sales",
                    "Consulting service sales",
                    "High-value B2B transactions",
                    "Partnership development",
                ],
                structural_insights=[
                    "Emission initiates prospect awareness",
                    "Reception identifies real needs vs stated needs",
                    "Coupling builds trust before pitching",
                    "Resonance aligns solution with needs",
                    "Coherence formalizes commitment",
                    "Expansion represents upsell/cross-sell",
                ],
            ),
            
            "organizational_change": PatternDefinition(
                name="Organizational Change Management",
                description="Large-scale organizational transformation pattern",
                examples=[
                    # Change management process
                    [
                        "emission",          # Communicate vision
                        "reception",         # Listen to concerns
                        "coherence",         # Stabilize current state
                        "dissonance",        # Highlight need for change
                        "coherence",         # Stabilize
                        "mutation",          # Implement transformation
                        "coherence",         # Integrate new state
                        "silence",           # Integration period
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "Digital transformation",
                    "Corporate restructuring",
                    "Culture change initiatives",
                    "Process reengineering",
                ],
                structural_insights=[
                    "Coherence before dissonance reduces resistance",
                    "Coupling builds stakeholder buy-in",
                    "Mutation represents the actual change point",
                    "Resonance spreads adoption through organization",
                    "Multiple coherence phases provide stability",
                ],
            ),
            
            "team_formation": PatternDefinition(
                name="High-Performance Team Formation",
                description="Pattern for building effective teams (Tuckman model)",
                examples=[
                    # Forming -> Storming -> Norming -> Performing
                    [
                        "emission",          # Forming: Initial gathering
                        "reception",         # Listen and understand
                        "coherence",         # Establish norms
                        "dissonance",        # Storming: Surface conflicts
                        "coherence",         # Norming: Stabilize
                        "resonance",         # Performing: Sync and flow
                        "silence",           # Integration
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "New team kickoffs",
                    "Cross-functional projects",
                    "Agile team formation",
                    "Remote team building",
                ],
                structural_insights=[
                    "Dissonance (storming) is necessary and healthy",
                    "Coherence provides psychological safety",
                    "Resonance enables team flow state",
                    "Coupling precedes productive conflict",
                ],
            ),
            
            "customer_success": PatternDefinition(
                name="Customer Success Journey",
                description="Post-sale customer value realization pattern",
                examples=[
                    [
                        "emission",      # Onboarding outreach
                        "reception",     # Understand customer goals
                        "coupling",      # Build success partnership
                        "coherence",     # Establish baseline usage
                        "expansion",     # Drive adoption
                        "resonance",     # Create value moments
                        "silence",       # Integration and renewal
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "SaaS customer onboarding",
                    "Enterprise account management",
                    "Customer retention programs",
                    "Value realization initiatives",
                ],
                structural_insights=[
                    "Reception aligns on success metrics",
                    "Coupling creates partnership mindset",
                    "Expansion tied to value delivery",
                    "Resonance amplifies ROI perception",
                ],
            ),
            
            "innovation_sprint": PatternDefinition(
                name="Innovation Sprint",
                description="Rapid innovation and prototyping cycle",
                examples=[
                    [
                        "emission",          # Launch sprint
                        "reception",         # Gather insights
                        "coherence",         # Stabilize
                        "dissonance",        # Challenge assumptions
                        "self_organization", # Ideate and prototype
                        "coherence",         # Converge on concepts
                        "silence",           # Integration and review
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "Design sprints",
                    "Hackathons",
                    "Rapid prototyping",
                    "Innovation workshops",
                ],
                structural_insights=[
                    "Dissonance drives creative thinking",
                    "Self-organization enables emergence",
                    "Coherence focuses divergent ideas",
                    "Resonance validates with market",
                ],
            ),
            
            "negotiation": PatternDefinition(
                name="Win-Win Negotiation",
                description="Collaborative negotiation pattern",
                examples=[
                    [
                        "emission",      # Opening positions
                        "reception",     # Listen to interests
                        "coherence",     # Find common ground
                        "dissonance",    # Surface conflicts
                        "coherence",     # Stabilize
                        "resonance",     # Align on solution
                        "silence",       # Agreement and integration
                    ],
                ],
                min_health_score=0.65,
                use_cases=[
                    "Contract negotiations",
                    "Partnership agreements",
                    "Conflict resolution",
                    "Merger & acquisition talks",
                ],
                structural_insights=[
                    "Reception reveals true interests vs positions",
                    "Coupling humanizes the negotiation",
                    "Dissonance makes conflict explicit",
                    "Expansion creates integrative solutions",
                ],
            ),
        }
    
    def get_metadata(self) -> Dict[str, any]:
        """Return business extension metadata."""
        return {
            "domain": "business",
            "version": "1.0.0",
            "author": "TNFR Business Domain Contributors",
            "description": "Business process and organizational patterns for TNFR",
            "principles": [
                "Value creation",
                "Stakeholder alignment",
                "Sustainable growth",
                "Collaborative approach",
            ],
            "validation_standards": {
                "min_health_score": 0.65,
                "business_outcome_validation": True,
                "practitioner_review": True,
            },
        }
