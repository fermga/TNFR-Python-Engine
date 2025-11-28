"""Dependency injection container for TNFR engine components.

This module provides a lightweight dependency injection container that allows
configuring and resolving engine components (validators, registries, dynamics
engines, telemetry collectors) without hard-coding implementations.

The container supports both singleton and factory registrations, enabling
flexible composition of TNFR services while maintaining clean separation of
concerns.

Examples
--------
Create a container with default implementations:

>>> container = TNFRContainer.create_default()
>>> validator = container.get(ValidationService)
>>> validator.validate_sequence(["emission", "coherence"])

Register custom implementations:

>>> container = TNFRContainer()
>>> container.register_singleton(ValidationService, CustomValidator())
>>> validator = container.get(ValidationService)
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

__all__ = ("TNFRContainer",)


class TNFRContainer:
    """Dependency injection container for TNFR engine components.

    The container manages the lifecycle of engine services, allowing
    registration of singletons (reused instances) or factories (fresh
    instances on each request). This enables flexible configuration without
    coupling client code to specific implementations.

    Examples
    --------
    Basic usage with singleton registration:

    >>> from tnfr.core.interfaces import ValidationService
    >>> class MyValidator:
    ...     def validate_sequence(self, seq):
    ...         pass
    ...     def validate_graph_state(self, graph):
    ...         pass
    >>> container = TNFRContainer()
    >>> container.register_singleton(ValidationService, MyValidator())
    >>> validator = container.get(ValidationService)

    Factory registration for per-request instances:

    >>> container.register_factory(ValidationService, MyValidator)
    >>> v1 = container.get(ValidationService)
    >>> v2 = container.get(ValidationService)
    >>> v1 is not v2  # Different instances
    True
    """

    def __init__(self) -> None:
        """Initialize an empty container."""
        self._instances: dict[Any, Any] = {}
        self._factories: dict[Any, Callable[[], Any]] = {}
        self._is_singleton: dict[Any, bool] = {}  # Track singleton status

    def register_singleton(self, interface: type[T], implementation: T) -> None:
        """Register an implementation as a singleton.

        The provided instance will be cached and returned for all subsequent
        requests for this interface.

        Parameters
        ----------
        interface : type
            The interface or abstract type to register against.
        implementation : T
            The concrete instance to return for this interface.

        Examples
        --------
        >>> container = TNFRContainer()
        >>> container.register_singleton(ValidationService, MyValidator())
        """
        # Store as a factory that returns the same instance
        self._factories[interface] = lambda: implementation
        self._is_singleton[interface] = True

    def register_factory(self, interface: type[T], factory_func: Callable[[], T]) -> None:
        """Register a factory function for creating instances.

        The factory will be called each time the interface is requested,
        allowing fresh instances or cached instances depending on the
        factory implementation.

        Parameters
        ----------
        interface : type
            The interface or abstract type to register against.
        factory_func : callable
            Function that creates and returns an instance of the interface.

        Examples
        --------
        >>> container = TNFRContainer()
        >>> container.register_factory(ValidationService, MyValidator)
        """
        self._factories[interface] = factory_func
        self._is_singleton[interface] = False

    def get(self, interface: type[T]) -> T:
        """Retrieve an instance implementing the specified interface.

        For singleton registrations, returns the cached instance. For factory
        registrations, calls the factory to produce an instance.

        Parameters
        ----------
        interface : type
            The interface type to resolve.

        Returns
        -------
        T
            Instance implementing the interface.

        Raises
        ------
        ValueError
            When no factory is registered for the interface.

        Examples
        --------
        >>> container = TNFRContainer()
        >>> container.register_singleton(ValidationService, MyValidator())
        >>> validator = container.get(ValidationService)
        """
        # Check if we have a cached singleton instance
        if interface in self._instances:
            return self._instances[interface]

        # Check if we have a factory registered
        if interface not in self._factories:
            raise ValueError(
                f"No factory registered for {interface}. "
                f"Use register_singleton() or register_factory() first."
            )

        # Call factory
        instance = self._factories[interface]()

        # Cache only if registered as singleton
        if self._is_singleton.get(interface, False):
            self._instances[interface] = instance

        return instance

    def has(self, interface: type) -> bool:
        """Check if an interface has a registered factory.

        Parameters
        ----------
        interface : type
            The interface type to check.

        Returns
        -------
        bool
            True if a factory is registered for this interface.
        """
        return interface in self._factories

    @classmethod
    def create_default(cls) -> TNFRContainer:
        """Create a container with default TNFR implementations.

        This factory method registers the standard implementations of all
        core interfaces, making it easy to get started with the TNFR engine
        without manual configuration.

        Returns
        -------
        TNFRContainer
            Container configured with default implementations.

        Examples
        --------
        >>> container = TNFRContainer.create_default()
        >>> validator = container.get(ValidationService)
        >>> # Use validator with default implementation

        Notes
        -----
        The default implementations are imported lazily to avoid circular
        dependencies. Custom implementations can be registered after creation
        to override defaults.
        """
        from .default_implementations import (
            DefaultDynamicsEngine,
            DefaultOperatorRegistry,
            DefaultTelemetryCollector,
            DefaultValidationService,
        )
        from .interfaces import (
            DynamicsEngine,
            OperatorRegistry,
            TelemetryCollector,
            ValidationService,
        )

        container = cls()

        # Register default implementations as singletons
        container.register_singleton(ValidationService, DefaultValidationService())
        container.register_singleton(OperatorRegistry, DefaultOperatorRegistry())
        container.register_singleton(DynamicsEngine, DefaultDynamicsEngine())
        container.register_singleton(TelemetryCollector, DefaultTelemetryCollector())

        return container
