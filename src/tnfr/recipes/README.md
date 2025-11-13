# TNFR Recipes Module

This module provides programmatic access to the TNFR Pattern Cookbook, a comprehensive library of validated operator sequences organized by application domain.

## Overview

The TNFR Pattern Cookbook offers ready-to-use, battle-tested operator sequences validated against TNFR Grammar 2.0. Each recipe includes:

- **Validated sequences**: Operators that pass canonical grammar validation
- **Health metrics**: Quantitative quality assessment (all recipes > 0.75 health)
- **Domain context**: When and how to apply the pattern
- **Use cases**: Specific real-world applications
- **Metadata**: Pattern types, structural characteristics, and more

## Quick Start

```python
from tnfr.recipes import TNFRCookbook

# Initialize the cookbook
cookbook = TNFRCookbook()

# Get a specific recipe
recipe = cookbook.get_recipe("therapeutic", "crisis_intervention")
print(f"Sequence: {' → '.join(recipe.sequence)}")
print(f"Health: {recipe.health_metrics.overall_health:.3f}")
print(f"Use cases: {recipe.use_cases}")

# List high-quality recipes
high_quality = cookbook.list_recipes(min_health=0.85)
for r in high_quality:
    print(f"{r.name} ({r.domain}): {r.health_metrics.overall_health:.3f}")

# Search for recipes
results = cookbook.search_recipes("team")
for r in results:
    print(f"Found: {r.name}")

# Get recommendation
recommended = cookbook.recommend_recipe(
    context="Need to facilitate learning breakthrough",
    constraints={"min_health": 0.80}
)
if recommended:
    print(f"Recommended: {recommended.name}")
```

## Available Domains

The cookbook covers 4 major application domains with 21+ validated recipes:

### 🏥 Therapeutic Domain (5 recipes)
- Crisis Intervention (0.79)
- Process Therapy (0.88) ⭐
- Regenerative Healing (0.80)
- Insight Integration (0.77)
- Relapse Prevention (0.76)

### 🎓 Educational Domain (5 recipes)
- Conceptual Breakthrough (0.79)
- Competency Development (0.83)
- Knowledge Spiral (0.85)
- Collaborative Learning (0.83)
- Practice Mastery (0.82)

### 🏢 Organizational Domain (6 recipes)
- Crisis Management (0.89) ⭐
- Team Formation (0.82)
- Strategic Planning (0.81)
- Innovation Cycle (0.81)
- Organizational Transformation (0.78)
- Change Resistance Resolution (0.84)

### 🎨 Creative Domain (5 recipes)
- Artistic Creation (0.78)
- Design Thinking (0.91) ⭐
- Innovation Cycle (0.82)
- Creative Flow (0.76)
- Creative Block Resolution (0.82)

⭐ = Highest health score in domain

## API Reference

### TNFRCookbook

Main class providing access to all recipes.

#### Methods

**`get_recipe(domain: str, use_case: str) -> CookbookRecipe`**

Get a specific recipe by domain and use case identifier.

```python
recipe = cookbook.get_recipe("therapeutic", "crisis_intervention")
```

**`list_recipes(domain: str = None, min_health: float = 0.0, max_length: int = None, pattern_type: str = None) -> List[CookbookRecipe]`**

List recipes with optional filtering.

```python
# All therapeutic recipes with health > 0.80
recipes = cookbook.list_recipes(
    domain="therapeutic",
    min_health=0.80
)

# Short sequences across all domains
short = cookbook.list_recipes(max_length=8)

# Regenerative patterns only
regen = cookbook.list_recipes(pattern_type="regenerative")
```

**`search_recipes(query: str) -> List[CookbookRecipe]`**

Search recipes by text query (case-insensitive).

```python
# Find all team-related recipes
team_recipes = cookbook.search_recipes("team")

# Find crisis management patterns
crisis = cookbook.search_recipes("crisis")
```

**`recommend_recipe(context: str, constraints: Dict[str, Any] = None) -> Optional[CookbookRecipe]`**

Get recipe recommendation based on context description.

```python
recommended = cookbook.recommend_recipe(
    context="Need to help new team collaborate on innovation project",
    constraints={
        "min_health": 0.80,
        "max_length": 12,
        "domain": "organizational"  # optional
    }
)
```

**`get_all_domains() -> List[str]`**

Get list of all available domains.

```python
domains = cookbook.get_all_domains()
# ['therapeutic', 'educational', 'organizational', 'creative']
```

**`get_domain_summary(domain: str) -> Dict[str, Any]`**

Get summary statistics for a domain.

```python
summary = cookbook.get_domain_summary("therapeutic")
print(f"Recipes: {summary['recipe_count']}")
print(f"Avg Health: {summary['average_health']:.3f}")
print(f"Patterns: {summary['patterns']}")
```

### CookbookRecipe

Data class representing a validated recipe.

#### Attributes

- **name** (str): Recipe display name
- **domain** (str): Application domain
- **sequence** (List[str]): Validated operator sequence
- **health_metrics** (SequenceHealthMetrics): Computed health metrics
- **use_cases** (List[str]): Real-world applications
- **when_to_use** (str): Context description
- **structural_flow** (List[str]): Operator explanations
- **key_insights** (List[str]): Success factors
- **variations** (List[RecipeVariation]): Adaptations
- **pattern_type** (str): Detected TNFR pattern

#### Example

```python
recipe = cookbook.get_recipe("educational", "conceptual_breakthrough")

print(f"Name: {recipe.name}")
print(f"Domain: {recipe.domain}")
print(f"Sequence: {' → '.join(recipe.sequence)}")
print(f"Health: {recipe.health_metrics.overall_health:.3f}")
print(f"Pattern: {recipe.pattern_type}")

for use_case in recipe.use_cases:
    print(f"- {use_case}")
```

## Examples

See `examples/pattern_cookbook_demo.py` for comprehensive usage examples including:
- Getting specific recipes
- Filtering by domain and health
- Searching by keywords
- Getting recommendations
- Comparing recipes
- Domain summaries

## Documentation

For detailed recipe documentation including:
- Complete structural flow explanations
- Use case descriptions
- Key insights and success factors
- Variations and adaptations
- Quick reference tables

See: **[docs/PATTERN_COOKBOOK.md](../../../docs/PATTERN_COOKBOOK.md)**

## Integration with Other Components

### With Grammar Validator

```python
from tnfr.recipes import TNFRCookbook
from tnfr.operators.grammar import validate_sequence_with_health

cookbook = TNFRCookbook()
recipe = cookbook.get_recipe("therapeutic", "process_therapy")

# Validate the sequence
result = validate_sequence_with_health(recipe.sequence)
print(f"Valid: {result.passed}")
print(f"Health: {result.health_metrics.overall_health:.3f}")
```

### With Sequence Generator

```python
from tnfr.recipes import TNFRCookbook
# Cookbook can serve as base for sequence generation
# See sequence_generator documentation for details
```

## Quality Standards

All recipes in the cookbook meet these quality standards:

- ✅ Pass TNFR Grammar 2.0 validation
- ✅ Health score ≥ 0.75
- ✅ Tested in domain context
- ✅ Complete documentation
- ✅ Real-world use cases identified

## Contributing

To contribute new recipes:

1. Design sequence and validate with TNFR Grammar 2.0
2. Achieve health score > 0.75
3. Document thoroughly (context, flow, insights, variations)
4. Test in real-world domain context
5. Submit with complete documentation

See cookbook documentation for detailed contribution guidelines.

## License

Part of the TNFR Python Engine. See [LICENSE.md](../../../LICENSE.md).
