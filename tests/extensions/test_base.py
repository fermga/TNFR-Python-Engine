"""Tests for TNFR extension system base classes."""

import pytest
from tnfr.extensions.base import (
    TNFRExtension,
    ExtensionRegistry,
    PatternDefinition,
)


class MockExtension(TNFRExtension):
    """Mock extension for testing."""
    
    def get_domain_name(self) -> str:
        return "test_domain"
    
    def get_pattern_definitions(self):
        return {
            "test_pattern": PatternDefinition(
                name="Test Pattern",
                description="A test pattern",
                examples=[
                    ["emission", "reception", "coherence"],
                ],
                min_health_score=0.75,
            )
        }


class TestPatternDefinition:
    """Test PatternDefinition dataclass."""
    
    def test_create_pattern_definition(self):
        """Test creating a pattern definition."""
        pattern = PatternDefinition(
            name="Test",
            description="Test description",
            examples=[["emission", "coherence"]],
            min_health_score=0.80,
        )
        
        assert pattern.name == "Test"
        assert pattern.description == "Test description"
        assert len(pattern.examples) == 1
        assert pattern.min_health_score == 0.80
    
    def test_pattern_with_optional_fields(self):
        """Test pattern with use cases and insights."""
        pattern = PatternDefinition(
            name="Test",
            description="Test",
            examples=[["emission"]],
            use_cases=["Use case 1", "Use case 2"],
            structural_insights=["Insight 1"],
        )
        
        assert len(pattern.use_cases) == 2
        assert len(pattern.structural_insights) == 1


class TestExtensionRegistry:
    """Test extension registry."""
    
    def test_register_extension(self):
        """Test registering an extension."""
        registry = ExtensionRegistry()
        extension = MockExtension()
        
        registry.register_extension(extension)
        
        assert "test_domain" in registry.list_extensions()
    
    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate domain raises error."""
        registry = ExtensionRegistry()
        extension1 = MockExtension()
        extension2 = MockExtension()
        
        registry.register_extension(extension1)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register_extension(extension2)
    
    def test_invalid_domain_name_uppercase(self):
        """Test that uppercase domain names are rejected."""
        class BadExtension(TNFRExtension):
            def get_domain_name(self):
                return "TestDomain"
            
            def get_pattern_definitions(self):
                return {}
        
        registry = ExtensionRegistry()
        with pytest.raises(ValueError, match="must be lowercase"):
            registry.register_extension(BadExtension())
    
    def test_invalid_domain_name_special_chars(self):
        """Test that domain names with special chars are rejected."""
        class BadExtension(TNFRExtension):
            def get_domain_name(self):
                return "test-domain"
            
            def get_pattern_definitions(self):
                return {}
        
        registry = ExtensionRegistry()
        with pytest.raises(ValueError, match="alphanumeric"):
            registry.register_extension(BadExtension())
    
    def test_get_extension(self):
        """Test retrieving registered extension."""
        registry = ExtensionRegistry()
        extension = MockExtension()
        
        registry.register_extension(extension)
        retrieved = registry.get_extension("test_domain")
        
        assert retrieved is extension
    
    def test_get_nonexistent_extension_raises_error(self):
        """Test getting unregistered extension raises KeyError."""
        registry = ExtensionRegistry()
        
        with pytest.raises(KeyError):
            registry.get_extension("nonexistent")
    
    def test_unregister_extension(self):
        """Test unregistering an extension."""
        registry = ExtensionRegistry()
        extension = MockExtension()
        
        registry.register_extension(extension)
        assert "test_domain" in registry.list_extensions()
        
        registry.unregister_extension("test_domain")
        assert "test_domain" not in registry.list_extensions()
    
    def test_unregister_nonexistent_raises_error(self):
        """Test unregistering nonexistent domain raises KeyError."""
        registry = ExtensionRegistry()
        
        with pytest.raises(KeyError):
            registry.unregister_extension("nonexistent")
    
    def test_get_domain_patterns(self):
        """Test retrieving patterns for a domain."""
        registry = ExtensionRegistry()
        extension = MockExtension()
        
        registry.register_extension(extension)
        patterns = registry.get_domain_patterns("test_domain")
        
        assert "test_pattern" in patterns
        assert patterns["test_pattern"].name == "Test Pattern"
    
    def test_get_all_patterns(self):
        """Test retrieving all patterns from all domains."""
        registry = ExtensionRegistry()
        
        class Extension1(TNFRExtension):
            def get_domain_name(self):
                return "domain1"
            
            def get_pattern_definitions(self):
                return {
                    "p1": PatternDefinition(
                        name="P1", description="P1", examples=[[]]
                    )
                }
        
        class Extension2(TNFRExtension):
            def get_domain_name(self):
                return "domain2"
            
            def get_pattern_definitions(self):
                return {
                    "p2": PatternDefinition(
                        name="P2", description="P2", examples=[[]]
                    )
                }
        
        registry.register_extension(Extension1())
        registry.register_extension(Extension2())
        
        all_patterns = registry.get_all_patterns()
        
        assert "domain1" in all_patterns
        assert "domain2" in all_patterns
        assert "p1" in all_patterns["domain1"]
        assert "p2" in all_patterns["domain2"]
    
    def test_list_extensions_sorted(self):
        """Test that extensions are listed in sorted order."""
        registry = ExtensionRegistry()
        
        class ExtZ(TNFRExtension):
            def get_domain_name(self):
                return "z_domain"
            
            def get_pattern_definitions(self):
                return {}
        
        class ExtA(TNFRExtension):
            def get_domain_name(self):
                return "a_domain"
            
            def get_pattern_definitions(self):
                return {}
        
        registry.register_extension(ExtZ())
        registry.register_extension(ExtA())
        
        domains = registry.list_extensions()
        assert domains == ["a_domain", "z_domain"]


class TestTNFRExtension:
    """Test TNFRExtension base class."""
    
    def test_get_health_analyzers_default(self):
        """Test default health analyzers returns empty dict."""
        extension = MockExtension()
        analyzers = extension.get_health_analyzers()
        assert analyzers == {}
    
    def test_get_cookbook_recipes_default(self):
        """Test default cookbook recipes returns empty dict."""
        extension = MockExtension()
        recipes = extension.get_cookbook_recipes()
        assert recipes == {}
    
    def test_get_visualization_tools_default(self):
        """Test default visualization tools returns empty dict."""
        extension = MockExtension()
        tools = extension.get_visualization_tools()
        assert tools == {}
    
    def test_get_metadata_default(self):
        """Test default metadata includes basic fields."""
        extension = MockExtension()
        metadata = extension.get_metadata()
        
        assert "domain" in metadata
        assert metadata["domain"] == "test_domain"
        assert "version" in metadata
        assert "author" in metadata
        assert "description" in metadata
