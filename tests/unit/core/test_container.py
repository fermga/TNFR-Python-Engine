"""Tests for TNFRContainer dependency injection."""

import pytest

from tnfr.core.container import TNFRContainer
from tnfr.core.interfaces import ValidationService, OperatorRegistry


def test_container_initialization():
    """Verify container initializes empty."""
    container = TNFRContainer()
    assert not container.has(ValidationService)


def test_register_singleton():
    """Verify singleton registration and retrieval."""

    class MockValidator:
        def validate_sequence(self, seq):
            pass

        def validate_graph_state(self, graph):
            pass

    container = TNFRContainer()
    validator = MockValidator()
    container.register_singleton(ValidationService, validator)

    # Should return same instance
    retrieved1 = container.get(ValidationService)
    retrieved2 = container.get(ValidationService)
    assert retrieved1 is validator
    assert retrieved2 is validator


def test_register_factory():
    """Verify factory registration creates fresh instances."""

    class MockValidator:
        def __init__(self):
            self.id = id(self)

        def validate_sequence(self, seq):
            pass

        def validate_graph_state(self, graph):
            pass

    container = TNFRContainer()
    container.register_factory(ValidationService, MockValidator)

    # Should return different instances
    retrieved1 = container.get(ValidationService)
    retrieved2 = container.get(ValidationService)
    assert retrieved1.id != retrieved2.id


def test_get_unregistered_raises():
    """Verify getting unregistered interface raises ValueError."""
    container = TNFRContainer()

    with pytest.raises(ValueError, match="No factory registered"):
        container.get(ValidationService)


def test_has_check():
    """Verify has() correctly checks registration."""

    class MockValidator:
        def validate_sequence(self, seq):
            pass

        def validate_graph_state(self, graph):
            pass

    container = TNFRContainer()
    assert not container.has(ValidationService)

    container.register_singleton(ValidationService, MockValidator())
    assert container.has(ValidationService)


def test_create_default():
    """Verify create_default() registers all services."""
    container = TNFRContainer.create_default()

    # Should have all core services registered
    assert container.has(ValidationService)
    assert container.has(OperatorRegistry)

    # Should be able to retrieve them
    validator = container.get(ValidationService)
    assert isinstance(validator, ValidationService)


def test_default_services_are_singletons():
    """Verify default services are singleton instances."""
    container = TNFRContainer.create_default()

    validator1 = container.get(ValidationService)
    validator2 = container.get(ValidationService)
    assert validator1 is validator2
