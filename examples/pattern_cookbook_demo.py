"""TNFR Pattern Cookbook - Usage Examples.

This demo shows how to use the TNFR Pattern Cookbook to find and apply
validated operator sequences for different domains and use cases.
"""

from tnfr.recipes import TNFRCookbook
from tnfr.operators.grammar import validate_sequence_with_health


def main():
    """Demonstrate TNFR Pattern Cookbook usage."""
    
    print("=" * 80)
    print("TNFR Pattern Cookbook - Usage Demo")
    print("=" * 80)
    
    # Initialize cookbook
    print("\n📚 Initializing TNFR Pattern Cookbook...")
    cookbook = TNFRCookbook()
    
    # Example 1: Get a specific recipe
    print("\n" + "=" * 80)
    print("Example 1: Get a Specific Recipe")
    print("=" * 80)
    
    recipe = cookbook.get_recipe("therapeutic", "crisis_intervention")
    
    print(f"\n✅ Recipe: {recipe.name}")
    print(f"   Domain: {recipe.domain}")
    print(f"   Pattern Type: {recipe.pattern_type}")
    print(f"   Health Score: {recipe.health_metrics.overall_health:.3f}")
    print(f"   Sequence Length: {len(recipe.sequence)} operators")
    print(f"\n   Sequence:")
    print(f"   {' → '.join(recipe.sequence)}")
    
    print(f"\n   Use Cases:")
    for i, use_case in enumerate(recipe.use_cases, 1):
        print(f"   {i}. {use_case}")
    
    print(f"\n   When to Use:")
    print(f"   {recipe.when_to_use}")
    
    print(f"\n   Health Metrics:")
    print(f"   - Balance: {recipe.health_metrics.balance_score:.3f}")
    print(f"   - Sustainability: {recipe.health_metrics.sustainability_index:.3f}")
    print(f"   - Complexity Efficiency: {recipe.health_metrics.complexity_efficiency:.3f}")
    
    # Example 2: List high-quality recipes
    print("\n" + "=" * 80)
    print("Example 2: Find High-Quality Recipes (Health > 0.85)")
    print("=" * 80)
    
    high_quality = cookbook.list_recipes(min_health=0.85)
    
    print(f"\n✅ Found {len(high_quality)} high-quality recipes:")
    for i, r in enumerate(high_quality, 1):
        print(f"\n{i}. {r.name} ({r.domain})")
        print(f"   Health: {r.health_metrics.overall_health:.3f}")
        print(f"   Pattern: {r.pattern_type}")
        print(f"   Length: {len(r.sequence)} ops")
        print(f"   Top use case: {r.use_cases[0]}")
    
    # Example 3: Search for recipes by keyword
    print("\n" + "=" * 80)
    print("Example 3: Search for 'Learning' Recipes")
    print("=" * 80)
    
    learning_recipes = cookbook.search_recipes("learning")
    
    print(f"\n✅ Found {len(learning_recipes)} recipes related to 'learning':")
    for i, r in enumerate(learning_recipes, 1):
        print(f"\n{i}. {r.name} ({r.domain})")
        print(f"   Health: {r.health_metrics.overall_health:.3f}")
        print(f"   Sequence: {' → '.join(r.sequence[:3])} ... {' → '.join(r.sequence[-2:])}")
    
    # Example 4: Get recipe recommendation
    print("\n" + "=" * 80)
    print("Example 4: Get Recommendation for Context")
    print("=" * 80)
    
    context = "Need to help a new team work together effectively on a project"
    print(f"\n📝 Context: '{context}'")
    
    recommended = cookbook.recommend_recipe(
        context=context,
        constraints={
            "min_health": 0.80,
            "max_length": 12
        }
    )
    
    if recommended:
        print(f"\n✅ Recommended Recipe: {recommended.name}")
        print(f"   Domain: {recommended.domain}")
        print(f"   Health: {recommended.health_metrics.overall_health:.3f}")
        print(f"   Length: {len(recommended.sequence)} operators")
        print(f"\n   Sequence:")
        print(f"   {' → '.join(recommended.sequence)}")
        print(f"\n   Why this recipe:")
        print(f"   {recommended.when_to_use}")
    else:
        print("\n❌ No suitable recipe found for this context")
    
    # Example 5: Domain summary
    print("\n" + "=" * 80)
    print("Example 5: Domain Summaries")
    print("=" * 80)
    
    for domain in cookbook.get_all_domains():
        summary = cookbook.get_domain_summary(domain)
        
        print(f"\n📊 {domain.upper()} Domain:")
        print(f"   Recipes: {summary['recipe_count']}")
        print(f"   Avg Health: {summary['average_health']:.3f}")
        print(f"   Health Range: {summary['health_range'][0]:.3f} - {summary['health_range'][1]:.3f}")
        print(f"   Pattern Types: {', '.join(summary['patterns'])}")
    
    # Example 6: Apply a recipe
    print("\n" + "=" * 80)
    print("Example 6: Validate and Apply a Recipe")
    print("=" * 80)
    
    recipe = cookbook.get_recipe("educational", "conceptual_breakthrough")
    
    print(f"\n📖 Recipe: {recipe.name}")
    print(f"   Sequence: {' → '.join(recipe.sequence)}")
    
    # Validate the sequence
    result = validate_sequence_with_health(recipe.sequence)
    
    print(f"\n✅ Validation Result:")
    print(f"   Valid: {result.passed}")
    print(f"   Health: {result.health_metrics.overall_health:.3f}")
    print(f"   Pattern: {result.health_metrics.dominant_pattern}")
    
    if result.health_metrics.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.health_metrics.recommendations:
            print(f"   - {rec}")
    
    # Example 7: Compare recipes
    print("\n" + "=" * 80)
    print("Example 7: Compare Similar Recipes")
    print("=" * 80)
    
    crisis_recipes = cookbook.search_recipes("crisis")
    
    if len(crisis_recipes) >= 2:
        print(f"\n🔄 Comparing {len(crisis_recipes)} crisis-related recipes:")
        
        for recipe in crisis_recipes:
            print(f"\n   {recipe.name} ({recipe.domain}):")
            print(f"   ├─ Health: {recipe.health_metrics.overall_health:.3f}")
            print(f"   ├─ Length: {len(recipe.sequence)} ops")
            print(f"   ├─ Balance: {recipe.health_metrics.balance_score:.3f}")
            print(f"   ├─ Sustainability: {recipe.health_metrics.sustainability_index:.3f}")
            print(f"   └─ Best for: {recipe.use_cases[0]}")
    
    print("\n" + "=" * 80)
    print("🎉 Pattern Cookbook Demo Complete!")
    print("=" * 80)
    print("\n💡 Next Steps:")
    print("   - Explore docs/PATTERN_COOKBOOK.md for detailed recipe documentation")
    print("   - Use cookbook.list_recipes() to browse all available patterns")
    print("   - Try cookbook.recommend_recipe() with your specific use cases")
    print("   - Adapt recipes for your domain-specific contexts")
    print("\n")


if __name__ == "__main__":
    main()
