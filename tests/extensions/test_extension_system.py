"""Tests for TNFR extension system."""

import pytest
from tnfr.extensions import ExtensionRegistry
from tnfr.extensions.base import TNFRExtension, PatternDefinition, CookbookRecipe


class MockExtension(TNFRExtension):
    """Mock extension for testing."""

    def get_domain_name(self) -> str:
        return "test_domain"

    def get_pattern_definitions(self):
        return {
            "test_pattern": PatternDefinition(
                name="test_pattern",
                sequence=["emission", "coherence"],
                description="Test pattern",
            )
        }

    def get_health_analyzers(self):
        return {}


def test_extension_registry_singleton():
    """Test registry is a singleton instance."""
    from tnfr.extensions import registry

    assert registry is not None
    assert isinstance(registry, ExtensionRegistry)


def test_register_extension():
    """Test registering an extension."""
    registry = ExtensionRegistry()
    ext = MockExtension()

    registry.register_extension(ext)

    assert "test_domain" in registry.list_extensions()
    assert registry.get_extension("test_domain") is ext


def test_register_duplicate_raises_error():
    """Test registering duplicate extension raises ValueError."""
    registry = ExtensionRegistry()
    ext1 = MockExtension()
    ext2 = MockExtension()

    registry.register_extension(ext1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_extension(ext2)


def test_register_invalid_type_raises_error():
    """Test registering non-TNFRExtension raises TypeError."""
    registry = ExtensionRegistry()

    with pytest.raises(TypeError, match="must inherit from TNFRExtension"):
        registry.register_extension("not an extension")


def test_unregister_extension():
    """Test unregistering an extension."""
    registry = ExtensionRegistry()
    ext = MockExtension()

    registry.register_extension(ext)
    registry.unregister_extension("test_domain")

    assert "test_domain" not in registry.list_extensions()


def test_unregister_nonexistent_raises_error():
    """Test unregistering nonexistent extension raises KeyError."""
    registry = ExtensionRegistry()

    with pytest.raises(KeyError, match="not found"):
        registry.unregister_extension("nonexistent")


def test_get_extension():
    """Test getting extension by name."""
    registry = ExtensionRegistry()
    ext = MockExtension()

    registry.register_extension(ext)
    retrieved = registry.get_extension("test_domain")

    assert retrieved is ext


def test_get_nonexistent_extension_returns_none():
    """Test getting nonexistent extension returns None."""
    registry = ExtensionRegistry()

    result = registry.get_extension("nonexistent")

    assert result is None


def test_list_extensions():
    """Test listing all registered extensions."""
    registry = ExtensionRegistry()
    ext = MockExtension()

    registry.register_extension(ext)
    extensions = registry.list_extensions()

    assert isinstance(extensions, list)
    assert "test_domain" in extensions


def test_get_domain_patterns():
    """Test getting patterns for a domain."""
    registry = ExtensionRegistry()
    ext = MockExtension()

    registry.register_extension(ext)
    patterns = registry.get_domain_patterns("test_domain")

    assert "test_pattern" in patterns
    assert patterns["test_pattern"].sequence == ["emission", "coherence"]


def test_get_domain_patterns_nonexistent_raises_error():
    """Test getting patterns for nonexistent domain raises KeyError."""
    registry = ExtensionRegistry()

    with pytest.raises(KeyError, match="not found"):
        registry.get_domain_patterns("nonexistent")


def test_get_all_patterns():
    """Test getting all patterns from all extensions."""
    registry = ExtensionRegistry()
    ext = MockExtension()

    registry.register_extension(ext)
    all_patterns = registry.get_all_patterns()

    assert "test_domain" in all_patterns
    assert "test_pattern" in all_patterns["test_domain"]


def test_clear_registry():
    """Test clearing all extensions."""
    registry = ExtensionRegistry()
    ext = MockExtension()

    registry.register_extension(ext)
    registry.clear()

    assert len(registry.list_extensions()) == 0


def test_pattern_definition_creation():
    """Test creating PatternDefinition."""
    pattern = PatternDefinition(
        name="test",
        sequence=["emission", "coherence"],
        description="Test pattern",
        use_cases=["Case 1", "Case 2"],
        health_requirements={"min_coherence": 0.75},
    )

    assert pattern.name == "test"
    assert pattern.sequence == ["emission", "coherence"]
    assert len(pattern.use_cases) == 2
    assert pattern.health_requirements["min_coherence"] == 0.75


def test_cookbook_recipe_creation():
    """Test creating CookbookRecipe."""
    recipe = CookbookRecipe(
        name="test_recipe",
        description="Test recipe",
        sequence=["emission", "coherence"],
        parameters={"nf": 1.0},
        expected_health={"min_C_t": 0.75},
        validation={"tested_cases": 10},
    )

    assert recipe.name == "test_recipe"
    assert recipe.sequence == ["emission", "coherence"]
    assert recipe.parameters["nf"] == 1.0


def test_extension_metadata():
    """Test extension metadata method."""
    ext = MockExtension()
    metadata = ext.get_metadata()

    assert "domain" in metadata
    assert metadata["domain"] == "test_domain"
