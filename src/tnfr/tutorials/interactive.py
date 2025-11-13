"""Interactive tutorial function stubs.

This file restores minimal tutorial helpers that tests import. They return
deterministic placeholder data with the expected keys and valid ranges. The
stubs preserve public API and can be expanded with full demos later.
"""

from __future__ import annotations

from typing import Dict, Any, List

import random


def _seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)


def hello_tnfr(
    *, interactive: bool = False, random_seed: int | None = None
) -> Dict[str, Any]:
    """Return a simple readiness structure.

    Parameters
    ----------
    interactive : bool
        Placeholder for future interactive mode.
    random_seed : int | None
        Optional seed for deterministic output.
    """
    _seed(random_seed)
    return {
        "message": "TNFR tutorial environment ready",
        "interactive": interactive,
    }


def biological_example(
    *, interactive: bool = False, random_seed: int | None = None
) -> Dict[str, Any]:
    _seed(random_seed)
    # Deterministic pseudo coherence and sense indices
    coherence = 0.62
    sense_indices = {f"node_{i}": round(0.4 + i * 0.05, 3) for i in range(3)}
    return {
        "coherence": coherence,
        "sense_indices": sense_indices,
        "interpretation": "Moderate biological coherence in small triad",
    }


def social_network_example(
    *, interactive: bool = False, random_seed: int | None = None
) -> Dict[str, Any]:
    _seed(random_seed)
    return {"coherence": 0.55, "nodes": 5}


def team_communication_example(
    *, interactive: bool = False, random_seed: int | None = None
) -> Dict[str, Any]:
    _seed(random_seed)
    structures = {
        "random": {"coherence": 0.48},
        "ring": {"coherence": 0.52},
        "small_world": {"coherence": 0.57},
    }
    best = max(structures.items(), key=lambda kv: kv[1]["coherence"])[0]
    structures["best_structure"] = best
    return structures


def adaptive_ai_example(
    *, interactive: bool = False, random_seed: int | None = None
) -> Dict[str, Any]:
    _seed(random_seed)
    initial = 0.42
    final = 0.58
    trajectory: List[float] = [initial, 0.47, 0.50, 0.55, final]
    return {
        "initial_coherence": initial,
        "final_coherence": final,
        "improvement": round(final - initial, 3),
        "coherence_trajectory": trajectory,
    }


def technology_example(
    *, interactive: bool = False, random_seed: int | None = None
) -> Dict[str, Any]:
    _seed(random_seed)
    return {
        "coherence": 0.60,
        "description": "Tech diffusion pattern placeholder",
    }


def oz_dissonance_tutorial(
    *, interactive: bool = False, random_seed: int | None = None
) -> Dict[str, Any]:
    _seed(random_seed)
    return {"dissonance_pulses": 3, "coherence_after": 0.51}


def run_all_tutorials(random_seed: int | None = None) -> Dict[str, Any]:
    _seed(random_seed)
    return {
        "hello": hello_tnfr(random_seed=random_seed),
        "bio": biological_example(random_seed=random_seed),
        "social": social_network_example(random_seed=random_seed),
        "team": team_communication_example(random_seed=random_seed),
        "adaptive": adaptive_ai_example(random_seed=random_seed),
    }

