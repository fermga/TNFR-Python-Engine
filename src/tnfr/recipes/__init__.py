"""TNFR Pattern Cookbook - Validated recipes organized by domain.

This module provides programmatic access to the TNFR Pattern Cookbook,
a comprehensive library of pre-validated operator sequences with health
metrics, use cases, and domain-specific context.

Examples
--------
>>> from tnfr.recipes import TNFRCookbook
>>> cookbook = TNFRCookbook()
>>> recipe = cookbook.get_recipe("therapeutic", "crisis_intervention")
>>> print(f"{recipe.name}: Health {recipe.health_metrics.overall_health:.3f}")
Crisis Intervention: Health 0.786
"""

from .cookbook import TNFRCookbook, CookbookRecipe, RecipeVariation

__all__ = [
    "TNFRCookbook",
    "CookbookRecipe",
    "RecipeVariation",
]
