"""Base classes for TNFR domain extensions.

This module provides the foundation for community-contributed domain extensions
to TNFR Grammar 2.0. Extensions allow domain experts to add specialized patterns,
health analyzers, and tools while maintaining canonical TNFR invariants.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type


@dataclass
class PatternDefinition:
    """Definition of a domain-specific structural pattern.

    Captures a validated sequence of structural operators and its domain context,
    ensuring patterns maintain TNFR canonical requirements while providing
    domain-specific semantics.

    Attributes
    ----------
    name : str
        Unique identifier for the pattern within its domain.
    sequence : List[str]
        Ordered list of structural operator identifiers.
    description : str
        Human-readable explanation of what the pattern achieves.
    use_cases : List[str]
        Real-world scenarios where this pattern applies.
    health_requirements : Dict[str, float]
        Minimum health metrics (C(t), Si) required for pattern validity.
    domain_context : Dict[str, Any]
        Domain-specific metadata explaining real-world mapping.
    examples : List[Dict[str, Any]]
        Validated examples demonstrating pattern effectiveness.
    """

    name: str
    sequence: List[str]
    description: str
    use_cases: List[str] = field(default_factory=list)
    health_requirements: Dict[str, float] = field(default_factory=dict)
    domain_context: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CookbookRecipe:
    """Validated recipe for a common domain scenario.

    Provides a tested sequence configuration with parameters and expected outcomes,
    allowing practitioners to apply proven patterns to their specific contexts.

    Attributes
    ----------
    name : str
        Unique identifier for the recipe.
    description : str
        What this recipe achieves in domain terms.
    sequence : List[str]
        Structural operator sequence.
    parameters : Dict[str, Any]
        Recommended parameters (νf, phase, etc.).
    expected_health : Dict[str, float]
        Expected health metrics (C(t), Si) for successful application.
    validation : Dict[str, Any]
        Validation metadata (test count, success rate, notes).
    """

    name: str
    description: str
    sequence: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_health: Dict[str, float] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)


class TNFRExtension(ABC):
    """Abstract base class for TNFR domain extensions.

    Domain extensions provide specialized patterns, health analyzers, and tools
    for specific application domains while maintaining TNFR canonical invariants.
    All extensions must implement the core abstract methods to ensure consistent
    structure and discoverability.

    Examples
    --------
    >>> class MedicalExtension(TNFRExtension):
    ...     def get_domain_name(self) -> str:
    ...         return "medical"
    ...
    ...     def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
    ...         return {
    ...             "therapeutic_alliance": PatternDefinition(
    ...                 name="therapeutic_alliance",
    ...                 sequence=["emission", "reception", "coherence"],
    ...                 description="Establishing therapeutic trust",
    ...                 use_cases=["Initial therapy session", "Crisis intervention"],
    ...             )
    ...         }
    ...
    ...     def get_health_analyzers(self) -> Dict[str, Type]:
    ...         from .health_analyzers import TherapeuticHealthAnalyzer
    ...         return {"therapeutic": TherapeuticHealthAnalyzer}
    """

    @abstractmethod
    def get_domain_name(self) -> str:
        """Return the unique domain identifier.

        Returns
        -------
        str
            Domain name (lowercase, underscore-separated).
        """

    @abstractmethod
    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Return domain-specific pattern definitions.

        Returns
        -------
        Dict[str, PatternDefinition]
            Mapping of pattern names to their definitions.
        """

    @abstractmethod
    def get_health_analyzers(self) -> Dict[str, Type]:
        """Return domain-specific health analyzer classes.

        Returns
        -------
        Dict[str, Type]
            Mapping of analyzer names to analyzer classes.
        """

    def get_cookbook_recipes(self) -> Dict[str, CookbookRecipe]:
        """Return validated recipes for common scenarios.

        Returns
        -------
        Dict[str, CookbookRecipe]
            Mapping of recipe names to recipe definitions. Empty dict if none.
        """
        return {}

    def get_visualization_tools(self) -> Dict[str, Type]:
        """Return domain-specific visualization tools.

        Returns
        -------
        Dict[str, Type]
            Mapping of visualizer names to visualizer classes. Empty dict if none.
        """
        return {}

    def get_metadata(self) -> Dict[str, Any]:
        """Return extension metadata.

        Returns
        -------
        Dict[str, Any]
            Metadata including version, author, description, etc.
        """
        return {
            "domain": self.get_domain_name(),
            "version": "1.0.0",
            "description": self.__doc__ or "No description provided",
        }
