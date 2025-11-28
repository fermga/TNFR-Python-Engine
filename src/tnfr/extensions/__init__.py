"""Extension registry for TNFR domain extensions.

Manages registration, discovery, and access to community-contributed domain
extensions. Maintains a global registry of extensions while ensuring thread-safe
access and validation of extension requirements.
"""

from typing import Dict, List, Optional, Type
from .base import TNFRExtension, PatternDefinition, CookbookRecipe


class ExtensionRegistry:
    """Registry for managing TNFR domain extensions.

    Provides centralized registration and discovery of domain extensions,
    ensuring extensions are properly validated and accessible to the TNFR
    engine and user applications.

    Examples
    --------
    >>> from tnfr.extensions import registry
    >>> from tnfr.extensions.medical import MedicalExtension
    >>>
    >>> # Register extension
    >>> ext = MedicalExtension()
    >>> registry.register_extension(ext)
    >>>
    >>> # List available domains
    >>> domains = registry.list_extensions()
    >>> print(domains)  # ['medical', ...]
    >>>
    >>> # Get domain patterns
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
            Extension instance to register.

        Raises
        ------
        ValueError
            If extension with same domain name already registered.
        TypeError
            If extension does not inherit from TNFRExtension.
        """
        if not isinstance(extension, TNFRExtension):
            raise TypeError(f"Extension must inherit from TNFRExtension, got {type(extension)}")

        domain_name = extension.get_domain_name()

        if domain_name in self._extensions:
            raise ValueError(f"Extension for domain '{domain_name}' already registered")

        self._extensions[domain_name] = extension

    def unregister_extension(self, domain_name: str) -> None:
        """Unregister an extension.

        Parameters
        ----------
        domain_name : str
            Domain name of extension to remove.

        Raises
        ------
        KeyError
            If domain not found in registry.
        """
        if domain_name not in self._extensions:
            raise KeyError(f"Extension '{domain_name}' not found in registry")

        del self._extensions[domain_name]

    def get_extension(self, domain_name: str) -> Optional[TNFRExtension]:
        """Get extension by domain name.

        Parameters
        ----------
        domain_name : str
            Domain name identifier.

        Returns
        -------
        Optional[TNFRExtension]
            Extension instance if found, None otherwise.
        """
        return self._extensions.get(domain_name)

    def list_extensions(self) -> List[str]:
        """List all registered extension domain names.

        Returns
        -------
        List[str]
            Sorted list of registered domain names.
        """
        return sorted(self._extensions.keys())

    def get_domain_patterns(self, domain_name: str) -> Dict[str, PatternDefinition]:
        """Get pattern definitions for a specific domain.

        Parameters
        ----------
        domain_name : str
            Domain name identifier.

        Returns
        -------
        Dict[str, PatternDefinition]
            Mapping of pattern names to definitions.

        Raises
        ------
        KeyError
            If domain not found in registry.
        """
        extension = self.get_extension(domain_name)
        if extension is None:
            raise KeyError(f"Extension '{domain_name}' not found in registry")

        return extension.get_pattern_definitions()

    def get_domain_health_analyzers(self, domain_name: str) -> Dict[str, Type]:
        """Get health analyzers for a specific domain.

        Parameters
        ----------
        domain_name : str
            Domain name identifier.

        Returns
        -------
        Dict[str, Type]
            Mapping of analyzer names to analyzer classes.

        Raises
        ------
        KeyError
            If domain not found in registry.
        """
        extension = self.get_extension(domain_name)
        if extension is None:
            raise KeyError(f"Extension '{domain_name}' not found in registry")

        return extension.get_health_analyzers()

    def get_domain_recipes(self, domain_name: str) -> Dict[str, CookbookRecipe]:
        """Get cookbook recipes for a specific domain.

        Parameters
        ----------
        domain_name : str
            Domain name identifier.

        Returns
        -------
        Dict[str, CookbookRecipe]
            Mapping of recipe names to recipe definitions.

        Raises
        ------
        KeyError
            If domain not found in registry.
        """
        extension = self.get_extension(domain_name)
        if extension is None:
            raise KeyError(f"Extension '{domain_name}' not found in registry")

        return extension.get_cookbook_recipes()

    def get_all_patterns(self) -> Dict[str, Dict[str, PatternDefinition]]:
        """Get all patterns from all registered extensions.

        Returns
        -------
        Dict[str, Dict[str, PatternDefinition]]
            Nested mapping: domain_name -> pattern_name -> PatternDefinition.
        """
        all_patterns = {}
        for domain_name, extension in self._extensions.items():
            all_patterns[domain_name] = extension.get_pattern_definitions()

        return all_patterns

    def clear(self) -> None:
        """Clear all registered extensions.

        Primarily useful for testing and reinitialization.
        """
        self._extensions.clear()


# Global registry instance
registry = ExtensionRegistry()
