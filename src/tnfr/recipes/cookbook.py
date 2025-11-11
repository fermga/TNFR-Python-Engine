"""TNFR Pattern Cookbook - Programmatic access to validated recipes.

This module provides a comprehensive library of pre-validated operator sequences
organized by domain. All recipes are validated against TNFR Grammar 2.0 and
include health metrics, use cases, and variations.

Examples
--------
>>> from tnfr.recipes import TNFRCookbook
>>> cookbook = TNFRCookbook()
>>> recipe = cookbook.get_recipe("therapeutic", "crisis_intervention")
>>> print(recipe.sequence)
['emission', 'reception', 'coherence', 'dissonance', 'contraction', 'coherence', 'coupling', 'silence']
>>> print(recipe.health_metrics.overall_health)
0.786
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from ..compat.dataclass import dataclass
from ..operators.health_analyzer import SequenceHealthMetrics, SequenceHealthAnalyzer
from ..operators.grammar import validate_sequence_with_health


__all__ = [
    "CookbookRecipe",
    "RecipeVariation",
    "TNFRCookbook",
]


@dataclass
class RecipeVariation:
    """A variation of a cookbook recipe for specific contexts.

    Attributes
    ----------
    name : str
        Name of the variation
    description : str
        What changes in this variation
    sequence : List[str]
        Modified operator sequence
    health_impact : float
        Expected change in health score (positive or negative)
    context : str
        When to use this variation
    """

    name: str
    description: str
    sequence: List[str]
    health_impact: float
    context: str


@dataclass
class CookbookRecipe:
    """A validated TNFR operator sequence recipe with full context.

    Attributes
    ----------
    name : str
        Recipe name (e.g., "Crisis Intervention")
    domain : str
        Application domain (therapeutic, educational, organizational, creative)
    sequence : List[str]
        Validated operator sequence
    health_metrics : SequenceHealthMetrics
        Computed health metrics for the sequence
    use_cases : List[str]
        Specific real-world applications
    when_to_use : str
        Context description for applying this pattern
    structural_flow : List[str]
        Operator-by-operator explanation of structural effects
    key_insights : List[str]
        Critical success factors and mechanisms
    variations : List[RecipeVariation]
        Adaptations for related contexts
    pattern_type : str
        Detected TNFR pattern type
    """

    name: str
    domain: str
    sequence: List[str]
    health_metrics: SequenceHealthMetrics
    use_cases: List[str]
    when_to_use: str
    structural_flow: List[str]
    key_insights: List[str]
    variations: List[RecipeVariation]
    pattern_type: str


class TNFRCookbook:
    """Library of validated TNFR operator sequence recipes.

    Provides programmatic access to the pattern cookbook with search,
    filtering, and recommendation capabilities.

    Examples
    --------
    >>> cookbook = TNFRCookbook()
    >>> # Get specific recipe
    >>> recipe = cookbook.get_recipe("therapeutic", "crisis_intervention")
    >>> print(f"Health: {recipe.health_metrics.overall_health:.3f}")
    Health: 0.786

    >>> # List all recipes in domain
    >>> therapeutic = cookbook.list_recipes(domain="therapeutic")
    >>> len(therapeutic)
    5

    >>> # Search by keyword
    >>> results = cookbook.search_recipes("team")
    >>> [r.name for r in results]
    ['Team Formation', 'Strategic Planning']
    """

    def __init__(self) -> None:
        """Initialize the cookbook with all validated recipes."""
        self._recipes: Dict[str, Dict[str, CookbookRecipe]] = {}
        self._analyzer = SequenceHealthAnalyzer()
        self._load_recipes()

    def _load_recipes(self) -> None:
        """Load all recipes from domain pattern modules."""
        # Import domain patterns
        try:
            from examples.domain_applications import therapeutic_patterns
            from examples.domain_applications import educational_patterns
            from examples.domain_applications import organizational_patterns
            from examples.domain_applications import creative_patterns
        except ImportError:
            # Fallback for when examples are not in path
            import sys
            from pathlib import Path

            examples_path = (
                Path(__file__).parent.parent.parent.parent / "examples" / "domain_applications"
            )
            sys.path.insert(0, str(examples_path))
            import therapeutic_patterns
            import educational_patterns
            import organizational_patterns
            import creative_patterns

        # Load therapeutic recipes
        self._load_domain_recipes(
            "therapeutic",
            therapeutic_patterns,
            [
                (
                    "crisis_intervention",
                    "Crisis Intervention",
                    [
                        "Panic attack management",
                        "Acute grief response",
                        "Immediate post-trauma stabilization",
                        "Emergency emotional support",
                    ],
                    "Immediate stabilization needed, limited time available, high-intensity crisis requiring rapid containment.",
                ),
                (
                    "process_therapy",
                    "Process Therapy",
                    [
                        "Long-term psychotherapy processes",
                        "Personal transformation work",
                        "Complex trauma resolution",
                        "Deep character structure change",
                    ],
                    "Deep change required, sufficient time and resources available, client readiness for transformative work established.",
                ),
                (
                    "regenerative_healing",
                    "Regenerative Healing",
                    [
                        "Chronic condition management",
                        "Ongoing recovery processes",
                        "Building resilience patterns",
                        "Preventive mental health work",
                    ],
                    "Long-term healing journey, building sustainable coping patterns, emphasis on self-renewal capacity.",
                ),
                (
                    "insight_integration",
                    "Insight Integration",
                    [
                        "Post-breakthrough consolidation",
                        "Integrate therapeutic insights into daily life",
                        "Stabilize sudden understanding or awareness",
                        "Connect insights to behavioral change",
                    ],
                    "After significant therapeutic breakthrough, to anchor and propagate new understanding across life domains.",
                ),
                (
                    "relapse_prevention",
                    "Relapse Prevention",
                    [
                        "Addiction recovery maintenance",
                        "Prevent regression after therapy",
                        "Maintain behavioral changes",
                        "Strengthen therapeutic gains",
                    ],
                    "Post-treatment phase, building relapse prevention skills, strengthening recovery patterns.",
                ),
            ],
        )

        # Load educational recipes
        self._load_domain_recipes(
            "educational",
            educational_patterns,
            [
                (
                    "conceptual_breakthrough",
                    "Conceptual Breakthrough",
                    [
                        "Mathematical concept breakthroughs",
                        "Scientific paradigm shifts",
                        "Language structure insights",
                        "Artistic technique breakthroughs",
                    ],
                    "Facilitating 'aha!' moments, paradigm shifts in understanding, sudden insight into complex concepts.",
                ),
                (
                    "competency_development",
                    "Competency Development",
                    [
                        "Sustained learning processes",
                        "Professional skill development",
                        "Complex skill acquisition",
                        "Career-long competency building",
                    ],
                    "Long-term skill building, step-by-step mastery progression, comprehensive competency development.",
                ),
                (
                    "knowledge_spiral",
                    "Knowledge Spiral",
                    [
                        "Iterative knowledge deepening cycles",
                        "Research and scholarly inquiry",
                        "Progressive understanding development",
                        "Cumulative learning trajectories",
                    ],
                    "Building knowledge over time, spiral curriculum design, regenerative learning cycles.",
                ),
                (
                    "collaborative_learning",
                    "Collaborative Learning",
                    [
                        "Group project work",
                        "Peer tutoring",
                        "Learning communities",
                        "Collaborative knowledge construction",
                    ],
                    "Peer learning contexts, group work, social learning environments.",
                ),
                (
                    "practice_mastery",
                    "Practice Mastery",
                    [
                        "Deliberate practice routines",
                        "Skill refinement",
                        "Performance improvement cycles",
                        "Expertise development",
                    ],
                    "Focused practice sessions, skill refinement work, performance optimization.",
                ),
            ],
        )

        # Load organizational recipes
        self._load_domain_recipes(
            "organizational",
            organizational_patterns,
            [
                (
                    "crisis_management",
                    "Crisis Management",
                    [
                        "Market disruption response",
                        "Leadership transition crisis",
                        "Operational emergency management",
                        "Reputation crisis containment",
                    ],
                    "Immediate organizational crisis, emergency institutional response, acute disruption requiring rapid coordination.",
                ),
                (
                    "team_formation",
                    "Team Formation",
                    [
                        "New team assembly",
                        "Cross-functional project initiation",
                        "Department reorganization",
                        "Merger integration",
                    ],
                    "Building new teams, establishing group coherence, creating high-performing collaborative units.",
                ),
                (
                    "strategic_planning",
                    "Strategic Planning",
                    [
                        "Comprehensive strategic planning",
                        "Vision development",
                        "Major transformation initiatives",
                        "Long-term change management",
                    ],
                    "Strategic planning processes, long-term organizational transformation, vision-driven institutional evolution.",
                ),
                (
                    "innovation_cycle",
                    "Innovation Cycle",
                    [
                        "Innovation programs",
                        "R&D project cycles",
                        "Product development sprints",
                        "Process innovation",
                    ],
                    "Innovation projects from ideation through implementation, systematic innovation programs.",
                ),
                (
                    "organizational_transformation",
                    "Organizational Transformation",
                    [
                        "Major restructuring",
                        "Culture transformation",
                        "Digital transformation",
                        "Business model evolution",
                    ],
                    "Comprehensive institutional change, transforming organizational culture and structure, fundamental business model shifts.",
                ),
                (
                    "change_resistance_resolution",
                    "Change Resistance Resolution",
                    [
                        "Overcoming resistance",
                        "Addressing opposition",
                        "Building change adoption",
                        "Managing transition conflicts",
                    ],
                    "High resistance to organizational change, need to transform opposition into engagement.",
                ),
            ],
        )

        # Load creative recipes
        self._load_domain_recipes(
            "creative",
            creative_patterns,
            [
                (
                    "artistic_creation",
                    "Artistic Creation",
                    [
                        "Painting/sculpture creation",
                        "Musical composition",
                        "Novel/screenplay writing",
                        "Choreography",
                        "Architectural design",
                    ],
                    "Complete artistic projects, major creative works requiring full creative cycle from impulse through consolidation.",
                ),
                (
                    "design_thinking",
                    "Design Thinking",
                    [
                        "Product design",
                        "Service design",
                        "UX design",
                        "Human-centered innovation",
                        "Design sprints",
                    ],
                    "Design thinking processes, human-centered problem solving, empathy-driven innovation.",
                ),
                (
                    "innovation_cycle",
                    "Innovation Cycle",
                    [
                        "Continuous innovation programs",
                        "Product pipelines",
                        "Creative R&D cycles",
                        "Innovation portfolio management",
                    ],
                    "Sustained innovation work, regenerative innovation capability building, ongoing creative renewal.",
                ),
                (
                    "creative_flow",
                    "Creative Flow",
                    [
                        "Maintaining creative momentum",
                        "Flow state cultivation",
                        "Sustained artistic practice",
                        "Creative productivity optimization",
                    ],
                    "Developing sustained creative practice, maintaining flow states, building creative momentum.",
                ),
                (
                    "creative_block_resolution",
                    "Creative Block Resolution",
                    [
                        "Overcoming writer's block",
                        "Resolving stagnation",
                        "Reinvigorating work",
                        "Breaking through plateaus",
                    ],
                    "Stuck in creative process, experiencing creative block, need breakthrough to restart creative flow.",
                ),
            ],
        )

    def _load_domain_recipes(self, domain: str, module: Any, recipe_specs: List[tuple]) -> None:
        """Load recipes for a specific domain.

        Parameters
        ----------
        domain : str
            Domain name (therapeutic, educational, organizational, creative)
        module : module
            Python module containing pattern functions
        recipe_specs : List[tuple]
            List of (function_suffix, display_name, use_cases, when_to_use) tuples
        """
        if domain not in self._recipes:
            self._recipes[domain] = {}

        for spec in recipe_specs:
            func_suffix, display_name, use_cases, when_to_use = spec

            # Get sequence function
            func_name = f"get_{func_suffix}_sequence"
            if not hasattr(module, func_name):
                continue

            func = getattr(module, func_name)
            sequence = func()

            # Validate and get health metrics
            result = validate_sequence_with_health(sequence)
            if not result.passed:
                continue

            # Create recipe
            recipe = CookbookRecipe(
                name=display_name,
                domain=domain,
                sequence=sequence,
                health_metrics=result.health_metrics,
                use_cases=use_cases,
                when_to_use=when_to_use,
                structural_flow=[],  # Could be extracted from docstring
                key_insights=[],  # Could be extracted from docstring
                variations=[],  # Future enhancement
                pattern_type=result.health_metrics.dominant_pattern,
            )

            self._recipes[domain][func_suffix] = recipe

    def get_recipe(self, domain: str, use_case: str) -> CookbookRecipe:
        """Get a specific recipe by domain and use case identifier.

        Parameters
        ----------
        domain : str
            Domain name: "therapeutic", "educational", "organizational", "creative"
        use_case : str
            Use case identifier (e.g., "crisis_intervention", "team_formation")

        Returns
        -------
        CookbookRecipe
            The requested recipe with full context and metrics

        Raises
        ------
        KeyError
            If domain or use_case not found

        Examples
        --------
        >>> cookbook = TNFRCookbook()
        >>> recipe = cookbook.get_recipe("therapeutic", "crisis_intervention")
        >>> print(recipe.name)
        Crisis Intervention
        """
        if domain not in self._recipes:
            raise KeyError(f"Domain '{domain}' not found. Available: {list(self._recipes.keys())}")

        if use_case not in self._recipes[domain]:
            raise KeyError(
                f"Use case '{use_case}' not found in '{domain}'. "
                f"Available: {list(self._recipes[domain].keys())}"
            )

        return self._recipes[domain][use_case]

    def list_recipes(
        self,
        domain: Optional[str] = None,
        min_health: float = 0.0,
        max_length: Optional[int] = None,
        pattern_type: Optional[str] = None,
    ) -> List[CookbookRecipe]:
        """List recipes with optional filtering.

        Parameters
        ----------
        domain : str, optional
            Filter by domain (therapeutic, educational, organizational, creative)
        min_health : float, default=0.0
            Minimum health score threshold
        max_length : int, optional
            Maximum sequence length
        pattern_type : str, optional
            Filter by pattern type (activation, therapeutic, regenerative, etc.)

        Returns
        -------
        List[CookbookRecipe]
            Filtered list of recipes

        Examples
        --------
        >>> cookbook = TNFRCookbook()
        >>> # Get all high-quality therapeutic recipes
        >>> recipes = cookbook.list_recipes(domain="therapeutic", min_health=0.80)
        >>> [r.name for r in recipes]
        ['Process Therapy', 'Regenerative Healing']
        """
        results = []

        domains = [domain] if domain else list(self._recipes.keys())

        for dom in domains:
            if dom not in self._recipes:
                continue

            for recipe in self._recipes[dom].values():
                # Apply filters
                if recipe.health_metrics.overall_health < min_health:
                    continue

                if max_length and len(recipe.sequence) > max_length:
                    continue

                if pattern_type and recipe.pattern_type != pattern_type:
                    continue

                results.append(recipe)

        # Sort by health score descending
        results.sort(key=lambda r: r.health_metrics.overall_health, reverse=True)

        return results

    def search_recipes(self, query: str) -> List[CookbookRecipe]:
        """Search recipes by text query across names, use cases, and context.

        Parameters
        ----------
        query : str
            Search query string (case-insensitive)

        Returns
        -------
        List[CookbookRecipe]
            Recipes matching the query, sorted by relevance

        Examples
        --------
        >>> cookbook = TNFRCookbook()
        >>> results = cookbook.search_recipes("crisis")
        >>> [r.name for r in results]
        ['Crisis Intervention', 'Crisis Management']
        """
        query_lower = query.lower()
        results = []

        for domain_recipes in self._recipes.values():
            for recipe in domain_recipes.values():
                # Search in name
                if query_lower in recipe.name.lower():
                    results.append((recipe, 3))  # High relevance
                    continue

                # Search in use cases
                if any(query_lower in uc.lower() for uc in recipe.use_cases):
                    results.append((recipe, 2))  # Medium relevance
                    continue

                # Search in when_to_use
                if query_lower in recipe.when_to_use.lower():
                    results.append((recipe, 1))  # Low relevance
                    continue

        # Sort by relevance then health
        results.sort(key=lambda x: (x[1], x[0].health_metrics.overall_health), reverse=True)

        return [r[0] for r in results]

    def recommend_recipe(
        self,
        context: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Optional[CookbookRecipe]:
        """Recommend a recipe based on context description and constraints.

        Uses keyword matching and constraint satisfaction to find the best
        matching recipe for the described context.

        Parameters
        ----------
        context : str
            Description of the situation or need
        constraints : Dict[str, Any], optional
            Additional constraints:
            - max_length: int - maximum sequence length
            - min_health: float - minimum health score
            - domain: str - restrict to specific domain
            - prefer_pattern: str - preferred pattern type

        Returns
        -------
        CookbookRecipe or None
            Best matching recipe, or None if no good match found

        Examples
        --------
        >>> cookbook = TNFRCookbook()
        >>> recipe = cookbook.recommend_recipe(
        ...     context="Need to help team work together on new project",
        ...     constraints={"min_health": 0.80, "max_length": 10}
        ... )
        >>> recipe.name
        'Team Formation'
        """
        constraints = constraints or {}

        # Start with all recipes matching constraints
        candidates = self.list_recipes(
            domain=constraints.get("domain"),
            min_health=constraints.get("min_health", 0.75),
            max_length=constraints.get("max_length"),
            pattern_type=constraints.get("prefer_pattern"),
        )

        if not candidates:
            return None

        # Extract keywords from context
        context_lower = context.lower()
        keywords = set(context_lower.split())

        # Score each candidate by keyword overlap
        scored_candidates = []
        for recipe in candidates:
            score = 0

            # Check name overlap
            name_words = set(recipe.name.lower().split())
            score += len(keywords & name_words) * 5

            # Check use cases overlap
            for use_case in recipe.use_cases:
                use_case_words = set(use_case.lower().split())
                score += len(keywords & use_case_words) * 3

            # Check when_to_use overlap
            when_words = set(recipe.when_to_use.lower().split())
            score += len(keywords & when_words) * 2

            # Boost by health score
            score += recipe.health_metrics.overall_health * 10

            scored_candidates.append((recipe, score))

        if not scored_candidates:
            return None

        # Return highest scoring recipe
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

    def get_all_domains(self) -> List[str]:
        """Get list of all available domains.

        Returns
        -------
        List[str]
            List of domain names
        """
        return list(self._recipes.keys())

    def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """Get summary statistics for a domain.

        Parameters
        ----------
        domain : str
            Domain name

        Returns
        -------
        Dict[str, Any]
            Summary with recipe count, average health, patterns, etc.
        """
        if domain not in self._recipes:
            raise KeyError(f"Domain '{domain}' not found")

        recipes = list(self._recipes[domain].values())

        if not recipes:
            return {
                "domain": domain,
                "recipe_count": 0,
                "average_health": 0.0,
                "health_range": (0.0, 0.0),
                "patterns": [],
            }

        healths = [r.health_metrics.overall_health for r in recipes]
        patterns = [r.pattern_type for r in recipes]

        return {
            "domain": domain,
            "recipe_count": len(recipes),
            "average_health": sum(healths) / len(healths),
            "health_range": (min(healths), max(healths)),
            "patterns": list(set(patterns)),
            "recipes": [r.name for r in recipes],
        }
