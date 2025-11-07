"""Tests for TNFR Pattern Cookbook API.

Validates that the cookbook provides correct access to validated recipes
with proper filtering, search, and recommendation capabilities.
"""

import pytest
from tnfr.recipes import TNFRCookbook, CookbookRecipe


class TestTNFRCookbook:
    """Test suite for TNFRCookbook class."""
    
    @pytest.fixture
    def cookbook(self):
        """Create a cookbook instance for testing."""
        return TNFRCookbook()
    
    def test_cookbook_initialization(self, cookbook):
        """Test that cookbook initializes with all domains."""
        domains = cookbook.get_all_domains()
        assert "therapeutic" in domains
        assert "educational" in domains
        assert "organizational" in domains
        assert "creative" in domains
        assert len(domains) >= 4
    
    def test_get_recipe_valid(self, cookbook):
        """Test getting a valid recipe."""
        recipe = cookbook.get_recipe("therapeutic", "crisis_intervention")
        
        assert isinstance(recipe, CookbookRecipe)
        assert recipe.name == "Crisis Intervention"
        assert recipe.domain == "therapeutic"
        assert len(recipe.sequence) > 0
        assert recipe.health_metrics.overall_health >= 0.75
        assert len(recipe.use_cases) > 0
    
    def test_get_recipe_invalid_domain(self, cookbook):
        """Test that invalid domain raises KeyError."""
        with pytest.raises(KeyError, match="Domain .* not found"):
            cookbook.get_recipe("invalid_domain", "some_use_case")
    
    def test_get_recipe_invalid_use_case(self, cookbook):
        """Test that invalid use case raises KeyError."""
        with pytest.raises(KeyError, match="Use case .* not found"):
            cookbook.get_recipe("therapeutic", "invalid_use_case")
    
    def test_list_recipes_all(self, cookbook):
        """Test listing all recipes."""
        all_recipes = cookbook.list_recipes()
        
        # Should have recipes from all domains
        assert len(all_recipes) >= 15  # At least 15 total recipes
        
        # Should be sorted by health (descending)
        healths = [r.health_metrics.overall_health for r in all_recipes]
        assert healths == sorted(healths, reverse=True)
    
    def test_list_recipes_by_domain(self, cookbook):
        """Test filtering recipes by domain."""
        therapeutic = cookbook.list_recipes(domain="therapeutic")
        
        assert len(therapeutic) >= 5
        assert all(r.domain == "therapeutic" for r in therapeutic)
    
    def test_list_recipes_min_health(self, cookbook):
        """Test filtering by minimum health score."""
        high_quality = cookbook.list_recipes(min_health=0.85)
        
        assert len(high_quality) >= 1
        assert all(r.health_metrics.overall_health >= 0.85 for r in high_quality)
    
    def test_list_recipes_max_length(self, cookbook):
        """Test filtering by maximum sequence length."""
        short_recipes = cookbook.list_recipes(max_length=8)
        
        assert len(short_recipes) >= 1
        assert all(len(r.sequence) <= 8 for r in short_recipes)
    
    def test_list_recipes_pattern_type(self, cookbook):
        """Test filtering by pattern type."""
        therapeutic_patterns = cookbook.list_recipes(pattern_type="therapeutic")
        
        assert len(therapeutic_patterns) >= 1
        assert all(r.pattern_type == "therapeutic" for r in therapeutic_patterns)
    
    def test_list_recipes_combined_filters(self, cookbook):
        """Test combining multiple filters."""
        recipes = cookbook.list_recipes(
            domain="educational",
            min_health=0.80,
            max_length=10
        )
        
        assert all(r.domain == "educational" for r in recipes)
        assert all(r.health_metrics.overall_health >= 0.80 for r in recipes)
        assert all(len(r.sequence) <= 10 for r in recipes)
    
    def test_search_recipes_name(self, cookbook):
        """Test searching by recipe name."""
        results = cookbook.search_recipes("crisis")
        
        assert len(results) >= 1
        assert any("crisis" in r.name.lower() for r in results)
    
    def test_search_recipes_use_case(self, cookbook):
        """Test searching by use case keywords."""
        results = cookbook.search_recipes("team")
        
        assert len(results) >= 1
        # Should find in name or use cases
        for r in results:
            found = (
                "team" in r.name.lower() or
                any("team" in uc.lower() for uc in r.use_cases)
            )
            assert found
    
    def test_search_recipes_case_insensitive(self, cookbook):
        """Test that search is case-insensitive."""
        lower_results = cookbook.search_recipes("therapeutic")
        upper_results = cookbook.search_recipes("THERAPEUTIC")
        mixed_results = cookbook.search_recipes("ThErApEuTiC")
        
        assert len(lower_results) == len(upper_results)
        assert len(lower_results) == len(mixed_results)
    
    def test_search_recipes_no_results(self, cookbook):
        """Test search with no matching results."""
        results = cookbook.search_recipes("xyz_nonexistent_pattern_xyz")
        
        assert len(results) == 0
    
    def test_recommend_recipe_basic(self, cookbook):
        """Test basic recipe recommendation."""
        recipe = cookbook.recommend_recipe(
            context="Need to help team collaborate effectively",
            constraints={"min_health": 0.75}
        )
        
        assert recipe is not None
        assert recipe.health_metrics.overall_health >= 0.75
        # Should find team-related recipe
        assert "team" in recipe.name.lower() or any("team" in uc.lower() for uc in recipe.use_cases)
    
    def test_recommend_recipe_with_constraints(self, cookbook):
        """Test recommendation with multiple constraints."""
        recipe = cookbook.recommend_recipe(
            context="Emergency crisis response needed immediately",
            constraints={
                "min_health": 0.75,
                "max_length": 10,
            }
        )
        
        assert recipe is not None
        assert recipe.health_metrics.overall_health >= 0.75
        assert len(recipe.sequence) <= 10
    
    def test_recommend_recipe_domain_constraint(self, cookbook):
        """Test recommendation with domain constraint."""
        recipe = cookbook.recommend_recipe(
            context="Need to facilitate learning",
            constraints={"domain": "educational"}
        )
        
        assert recipe is not None
        assert recipe.domain == "educational"
    
    def test_recommend_recipe_no_match(self, cookbook):
        """Test recommendation with impossible constraints."""
        recipe = cookbook.recommend_recipe(
            context="some generic context",
            constraints={
                "min_health": 1.5,  # Impossible health score
            }
        )
        
        assert recipe is None
    
    def test_get_domain_summary_valid(self, cookbook):
        """Test getting summary for valid domain."""
        summary = cookbook.get_domain_summary("therapeutic")
        
        assert summary["domain"] == "therapeutic"
        assert summary["recipe_count"] >= 5
        assert 0.0 <= summary["average_health"] <= 1.0
        assert summary["health_range"][0] <= summary["health_range"][1]
        assert len(summary["patterns"]) >= 1
        assert len(summary["recipes"]) == summary["recipe_count"]
    
    def test_get_domain_summary_all_domains(self, cookbook):
        """Test getting summary for all domains."""
        domains = cookbook.get_all_domains()
        
        for domain in domains:
            summary = cookbook.get_domain_summary(domain)
            
            assert summary["recipe_count"] > 0
            assert summary["average_health"] >= 0.75  # All recipes should be quality
    
    def test_get_domain_summary_invalid(self, cookbook):
        """Test getting summary for invalid domain."""
        with pytest.raises(KeyError):
            cookbook.get_domain_summary("invalid_domain")
    
    def test_recipe_health_threshold(self, cookbook):
        """Test that all recipes meet minimum health threshold."""
        all_recipes = cookbook.list_recipes()
        
        for recipe in all_recipes:
            assert recipe.health_metrics.overall_health >= 0.75, (
                f"Recipe '{recipe.name}' has health {recipe.health_metrics.overall_health:.3f} "
                f"below threshold 0.75"
            )
    
    def test_recipe_has_required_fields(self, cookbook):
        """Test that all recipes have required fields populated."""
        all_recipes = cookbook.list_recipes()
        
        for recipe in all_recipes:
            assert recipe.name, f"Recipe missing name"
            assert recipe.domain, f"Recipe '{recipe.name}' missing domain"
            assert len(recipe.sequence) > 0, f"Recipe '{recipe.name}' has empty sequence"
            assert len(recipe.use_cases) > 0, f"Recipe '{recipe.name}' has no use cases"
            assert recipe.when_to_use, f"Recipe '{recipe.name}' missing when_to_use"
            assert recipe.health_metrics is not None, f"Recipe '{recipe.name}' missing health metrics"
    
    def test_recipe_sequences_valid(self, cookbook):
        """Test that all recipe sequences are valid operator sequences."""
        from tnfr.config.operator_names import CANONICAL_OPERATOR_NAMES
        
        all_recipes = cookbook.list_recipes()
        
        for recipe in all_recipes:
            for op in recipe.sequence:
                assert op in CANONICAL_OPERATOR_NAMES, (
                    f"Recipe '{recipe.name}' contains invalid operator '{op}'"
                )
    
    def test_domain_coverage(self, cookbook):
        """Test that all expected domains are covered."""
        expected_domains = {"therapeutic", "educational", "organizational", "creative"}
        actual_domains = set(cookbook.get_all_domains())
        
        assert expected_domains.issubset(actual_domains), (
            f"Missing domains: {expected_domains - actual_domains}"
        )
    
    def test_recipe_uniqueness(self, cookbook):
        """Test that recipes have unique names within each domain."""
        for domain in cookbook.get_all_domains():
            recipes = cookbook.list_recipes(domain=domain)
            names = [r.name for r in recipes]
            
            assert len(names) == len(set(names)), (
                f"Duplicate recipe names in {domain}: {names}"
            )


class TestCookbookIntegration:
    """Integration tests for cookbook with other TNFR components."""
    
    def test_recipe_grammar_validation(self):
        """Test that all recipes pass grammar validation."""
        from tnfr.operators.grammar import validate_sequence_with_health
        
        cookbook = TNFRCookbook()
        all_recipes = cookbook.list_recipes()
        
        for recipe in all_recipes:
            result = validate_sequence_with_health(recipe.sequence)
            
            assert result.passed, (
                f"Recipe '{recipe.name}' failed grammar validation: {result.message}"
            )
    
    def test_recipe_health_consistency(self):
        """Test that recipe health metrics are consistent with analyzer."""
        from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
        
        cookbook = TNFRCookbook()
        analyzer = SequenceHealthAnalyzer()
        
        # Test a sample recipe
        recipe = cookbook.get_recipe("therapeutic", "process_therapy")
        recalculated_health = analyzer.analyze_health(recipe.sequence)
        
        # Health should be approximately the same (allowing for floating point)
        assert abs(
            recipe.health_metrics.overall_health - recalculated_health.overall_health
        ) < 0.01
    
    def test_cookbook_with_sequence_generator(self):
        """Test cookbook integration with sequence generator."""
        try:
            from tnfr.tools.sequence_generator import SequenceGenerator
            
            cookbook = TNFRCookbook()
            generator = SequenceGenerator()
            
            # Get a recipe
            recipe = cookbook.get_recipe("educational", "conceptual_breakthrough")
            
            # Verify it can be validated by generator
            from tnfr.operators.grammar import validate_sequence_with_health
            result = validate_sequence_with_health(recipe.sequence)
            
            assert result.passed
            
        except ImportError:
            pytest.skip("SequenceGenerator not available")
