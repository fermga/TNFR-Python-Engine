"""Medical domain extension for TNFR.

Provides patterns, health analyzers, and tools for medical and therapeutic
applications, focusing on therapeutic dynamics, patient progress tracking,
and intervention planning.
"""

from typing import Dict, Type
from ..base import TNFRExtension, PatternDefinition, CookbookRecipe


class MedicalExtension(TNFRExtension):
    """Extension for medical and therapeutic applications.

    This extension provides specialized patterns for clinical contexts,
    therapeutic interventions, and patient care scenarios. It includes
    health analyzers for therapeutic effectiveness and visualization
    tools for treatment journeys.

    Examples
    --------
    >>> from tnfr.extensions import registry
    >>> from tnfr.extensions.medical import MedicalExtension
    >>>
    >>> # Register extension
    >>> ext = MedicalExtension()
    >>> registry.register_extension(ext)
    >>>
    >>> # Access patterns
    >>> patterns = ext.get_pattern_definitions()
    >>> print(list(patterns.keys()))
    ['therapeutic_alliance', 'crisis_intervention', 'integration_phase']
    """

    def get_domain_name(self) -> str:
        """Return domain name identifier."""
        return "medical"

    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Return medical domain pattern definitions."""
        from .patterns import PATTERNS

        return PATTERNS

    def get_health_analyzers(self) -> Dict[str, Type]:
        """Return medical domain health analyzers."""
        from .health_analyzers import TherapeuticHealthAnalyzer

        return {
            "therapeutic": TherapeuticHealthAnalyzer,
        }

    def get_cookbook_recipes(self) -> Dict[str, CookbookRecipe]:
        """Return validated recipes for common medical scenarios."""
        from .cookbook import RECIPES

        return RECIPES

    def get_metadata(self) -> Dict[str, object]:
        """Return extension metadata."""
        return {
            "domain": "medical",
            "version": "1.0.0",
            "description": "Medical and therapeutic domain extension",
            "author": "TNFR Community",
            "patterns_count": len(self.get_pattern_definitions()),
            "use_cases": [
                "Clinical therapy sessions",
                "Crisis intervention",
                "Patient progress tracking",
                "Treatment planning",
            ],
        }
