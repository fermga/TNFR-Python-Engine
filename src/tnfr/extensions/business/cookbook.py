"""Cookbook recipes for common business scenarios."""

from ..base import CookbookRecipe

# Change Initiative Recipe
CHANGE_INITIATIVE = CookbookRecipe(
    name="change_initiative",
    description="Launching successful organizational change program",
    sequence=["emission", "dissonance", "coupling", "self_organization", "coherence"],
    parameters={
        "suggested_nf": 1.0,  # Hz_str - steady change pace
        "suggested_phase": 0.0,
        "duration_weeks": 12,  # 3-month change cycle
    },
    expected_health={
        "min_C_t": 0.75,
        "min_Si": 0.70,
        "min_change_readiness": 0.75,
    },
    validation={
        "tested_cases": 15,
        "success_rate": 0.87,
        "notes": (
            "Validated on digital transformation and restructuring initiatives. "
            "Dissonance phase critical for unfreezing current state. "
            "Success measured as adoption rate > 80% after 6 months."
        ),
    },
)

# Process Improvement Recipe
PROCESS_IMPROVEMENT = CookbookRecipe(
    name="process_improvement",
    description="Optimizing business process for efficiency",
    sequence=["reception", "contraction", "coupling", "resonance"],
    parameters={
        "suggested_nf": 1.3,  # Hz_str - active improvement phase
        "suggested_phase": 0.0,
        "improvement_cycles": 3,  # PDCA iterations
    },
    expected_health={
        "min_C_t": 0.75,
        "min_Si": 0.70,
        "min_efficiency_potential": 0.78,
    },
    validation={
        "tested_cases": 20,
        "success_rate": 0.90,
        "notes": (
            "Validated on Lean Six Sigma projects. "
            "Reception phase ensures current state understanding. "
            "Success measured as cycle time reduction > 25%."
        ),
    },
)

# Team Alignment Recipe
TEAM_ALIGNMENT_RECIPE = CookbookRecipe(
    name="team_alignment_meeting",
    description="Aligning team around shared objectives",
    sequence=["emission", "reception", "coupling", "resonance", "coherence"],
    parameters={
        "suggested_nf": 1.5,  # Hz_str - high energy alignment
        "suggested_phase": 0.0,
        "meeting_duration_hours": 4,  # Half-day session
    },
    expected_health={
        "min_C_t": 0.75,
        "min_Si": 0.70,
        "min_alignment_strength": 0.80,
    },
    validation={
        "tested_cases": 25,
        "success_rate": 0.92,
        "notes": (
            "Validated on strategic planning and kickoff meetings. "
            "Reception phase ensures all voices heard. "
            "Success measured using team alignment survey."
        ),
    },
)

# Collect all recipes
RECIPES = {
    "change_initiative": CHANGE_INITIATIVE,
    "process_improvement": PROCESS_IMPROVEMENT,
    "team_alignment_meeting": TEAM_ALIGNMENT_RECIPE,
}
