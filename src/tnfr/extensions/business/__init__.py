"""Business domain extension for TNFR.

Provides patterns, health analyzers, and tools for business process optimization,
organizational change management, and workflow analysis using TNFR structural
operators.
"""

from typing import Dict, Type
from ..base import TNFRExtension, PatternDefinition, CookbookRecipe


class BusinessExtension(TNFRExtension):
    """Extension for business process and organizational applications.

    This extension provides specialized patterns for business contexts,
    organizational change, workflow optimization, and KPI analysis.

    Examples
    --------
    >>> from tnfr.extensions import registry
    >>> from tnfr.extensions.business import BusinessExtension
    >>>
    >>> # Register extension
    >>> ext = BusinessExtension()
    >>> registry.register_extension(ext)
    >>>
    >>> # Access patterns
    >>> patterns = ext.get_pattern_definitions()
    >>> print(list(patterns.keys()))
    ['change_management', 'workflow_optimization', 'team_alignment']
    """

    def get_domain_name(self) -> str:
        """Return domain name identifier."""
        return "business"

    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Return business domain pattern definitions."""
        from .patterns import PATTERNS

        return PATTERNS

    def get_health_analyzers(self) -> Dict[str, Type]:
        """Return business domain health analyzers."""
        from .health_analyzers import ProcessHealthAnalyzer

        return {
            "process": ProcessHealthAnalyzer,
        }

    def get_cookbook_recipes(self) -> Dict[str, CookbookRecipe]:
        """Return validated recipes for common business scenarios."""
        from .cookbook import RECIPES

        return RECIPES

    def get_metadata(self) -> Dict[str, object]:
        """Return extension metadata."""
        return {
            "domain": "business",
            "version": "1.0.0",
            "description": "Business process and organizational domain extension",
            "author": "TNFR Community",
            "patterns_count": len(self.get_pattern_definitions()),
            "use_cases": [
                "Organizational change management",
                "Workflow optimization",
                "Team alignment and collaboration",
                "Process improvement",
            ],
        }
