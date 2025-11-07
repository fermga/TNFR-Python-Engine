"""Base classes for TNFR extension system.

This module defines the abstract base class for TNFR extensions and supporting
data structures. Extensions can provide domain-specific patterns, health analyzers,
visualizers, and cookbook recipes.

Examples
--------
>>> from tnfr.extensions.base import TNFRExtension, PatternDefinition
>>> 
>>> class MedicalExtension(TNFRExtension):
...     def get_domain_name(self) -> str:
...         return "medical"
...     
...     def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
...         return {
...             "therapeutic_journey": PatternDefinition(
...                 name="Therapeutic Journey",
...                 description="Pattern for therapeutic processes",
...                 examples=[
...                     ["emission", "reception", "coherence", "coupling"],
...                 ],
...                 min_health_score=0.75,
...             )
...         }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any
from ..compat.dataclass import dataclass


__all__ = [
    "PatternDefinition",
    "TNFRExtension",
    "ExtensionRegistry",
]


@dataclass
class PatternDefinition:
    """Definition of a domain-specific pattern.
    
    Attributes
    ----------
    name : str
        Human-readable pattern name
    description : str
        Description of the pattern's purpose and use
    examples : List[List[str]]
        Example operator sequences demonstrating this pattern
    min_health_score : float
        Minimum required health score for pattern validity (default: 0.75)
    use_cases : List[str]
        Specific real-world applications (optional)
    structural_insights : List[str]
        Key structural mechanisms (optional)
    """
    name: str
    description: str
    examples: List[List[str]]
    min_health_score: float = 0.75
    use_cases: Optional[List[str]] = None
    structural_insights: Optional[List[str]] = None


class TNFRExtension(ABC):
    """Abstract base class for TNFR domain extensions.
    
    Extensions enable domain experts to contribute specialized patterns,
    health metrics, visualizations, and recipes without modifying core TNFR.
    
    All extensions must implement get_domain_name() and get_pattern_definitions().
    Other methods are optional and return empty collections by default.
    
    Examples
    --------
    >>> class BusinessExtension(TNFRExtension):
    ...     def get_domain_name(self) -> str:
    ...         return "business"
    ...     
    ...     def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
    ...         return {
    ...             "sales_cycle": PatternDefinition(
    ...                 name="Sales Cycle",
    ...                 description="Standard B2B sales progression",
    ...                 examples=[
    ...                     ["emission", "reception", "coupling", "resonance", "coherence"],
    ...                 ],
    ...             )
    ...         }
    """
    
    @abstractmethod
    def get_domain_name(self) -> str:
        """Return the domain name for this extension.
        
        Returns
        -------
        str
            Domain identifier (e.g., 'medical', 'business', 'scientific')
            Must be lowercase, alphanumeric with underscores only.
        """
        pass
    
    @abstractmethod
    def get_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Return pattern definitions for this domain.
        
        Returns
        -------
        Dict[str, PatternDefinition]
            Mapping of pattern IDs to their definitions.
            Pattern IDs should be lowercase with underscores.
        """
        pass
    
    def get_health_analyzers(self) -> Dict[str, Type[Any]]:
        """Return domain-specific health analyzer classes.
        
        Health analyzers extend the base SequenceHealthAnalyzer with
        domain-specific metrics and validation rules.
        
        Returns
        -------
        Dict[str, Type[Any]]
            Mapping of analyzer names to analyzer classes.
            Empty by default.
        """
        return {}
    
    def get_cookbook_recipes(self) -> Dict[str, Any]:
        """Return validated cookbook recipes for this domain.
        
        Returns
        -------
        Dict[str, Any]
            Mapping of recipe IDs to CookbookRecipe instances.
            Empty by default.
        """
        return {}
    
    def get_visualization_tools(self) -> Dict[str, Any]:
        """Return domain-specific visualization tools.
        
        Visualizers provide specialized views of sequences and patterns
        tailored to domain experts (e.g., therapeutic journey maps,
        organizational change timelines).
        
        Returns
        -------
        Dict[str, Any]
            Mapping of visualizer names to visualizer instances.
            Empty by default.
        """
        return {}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this extension.
        
        Returns
        -------
        Dict[str, Any]
            Extension metadata including version, author, description, etc.
        """
        return {
            "domain": self.get_domain_name(),
            "version": "1.0.0",
            "author": "Community",
            "description": f"TNFR extension for {self.get_domain_name()} domain",
        }


class ExtensionRegistry:
    """Registry for managing TNFR extensions.
    
    Provides centralized management of domain extensions, including
    registration, discovery, and pattern retrieval.
    
    Examples
    --------
    >>> from tnfr.extensions import ExtensionRegistry
    >>> registry = ExtensionRegistry()
    >>> registry.register_extension(MedicalExtension())
    >>> domains = registry.list_extensions()
    >>> patterns = registry.get_domain_patterns("medical")
    """
    
    def __init__(self) -> None:
        """Initialize empty extension registry."""
        self._extensions: Dict[str, TNFRExtension] = {}
    
    def register_extension(self, extension: TNFRExtension) -> None:
        """Register a new domain extension.
        
        Parameters
        ----------
        extension : TNFRExtension
            Extension instance to register
        
        Raises
        ------
        ValueError
            If extension domain name is invalid or already registered
        """
        domain = extension.get_domain_name()
        
        # Validate domain name
        if not domain:
            raise ValueError("Extension domain name cannot be empty")
        if not domain.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid domain name '{domain}': must be alphanumeric with underscores only"
            )
        if not domain.islower():
            raise ValueError(
                f"Invalid domain name '{domain}': must be lowercase"
            )
        
        # Check for duplicates
        if domain in self._extensions:
            raise ValueError(
                f"Extension for domain '{domain}' already registered"
            )
        
        self._extensions[domain] = extension
    
    def unregister_extension(self, domain: str) -> None:
        """Unregister an extension by domain name.
        
        Parameters
        ----------
        domain : str
            Domain name to unregister
        
        Raises
        ------
        KeyError
            If domain is not registered
        """
        if domain not in self._extensions:
            raise KeyError(f"No extension registered for domain '{domain}'")
        del self._extensions[domain]
    
    def list_extensions(self) -> List[str]:
        """List all registered extension domain names.
        
        Returns
        -------
        List[str]
            Sorted list of registered domain names
        """
        return sorted(self._extensions.keys())
    
    def get_extension(self, domain: str) -> TNFRExtension:
        """Get extension instance for a domain.
        
        Parameters
        ----------
        domain : str
            Domain name
        
        Returns
        -------
        TNFRExtension
            Extension instance
        
        Raises
        ------
        KeyError
            If domain is not registered
        """
        if domain not in self._extensions:
            raise KeyError(f"No extension registered for domain '{domain}'")
        return self._extensions[domain]
    
    def get_domain_patterns(self, domain: str) -> Dict[str, PatternDefinition]:
        """Get all pattern definitions for a domain.
        
        Parameters
        ----------
        domain : str
            Domain name
        
        Returns
        -------
        Dict[str, PatternDefinition]
            Mapping of pattern IDs to definitions
        
        Raises
        ------
        KeyError
            If domain is not registered
        """
        extension = self.get_extension(domain)
        return extension.get_pattern_definitions()
    
    def get_all_patterns(self) -> Dict[str, Dict[str, PatternDefinition]]:
        """Get all patterns from all registered extensions.
        
        Returns
        -------
        Dict[str, Dict[str, PatternDefinition]]
            Nested mapping: domain -> pattern_id -> definition
        """
        return {
            domain: ext.get_pattern_definitions()
            for domain, ext in self._extensions.items()
        }
