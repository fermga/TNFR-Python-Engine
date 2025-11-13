"""
TNFR Sequence Optimizer.

This module provides tools to search for optimal sequences of Glyphs that
achieve a specific objective while adhering to the Unified Grammar Rules.

It leverages the symbolic math modules to evaluate sequence properties
and guide the search process.

Core functionalities:
- Define objective functions (e.g., maximize coherence, minimize risk).
- Search for valid sequences that optimize the objective.
- Prune search space using grammar validation.

Possible algorithms:
- Genetic Algorithms
- A* Search
- Greedy Best-First Search
"""

from typing import List, Callable, Tuple

from ..types import Glyph
from .grammar_validators import (
    verify_convergence_for_sequence,
    verify_bifurcation_risk_for_sequence,
)

# Type alias for a sequence of Glyphs
Sequence = List[Glyph]

# Type alias for an objective function that scores a sequence
ObjectiveFunction = Callable[[Sequence], float]


def sample_objective_function(sequence: Sequence) -> float:
    """
    A sample objective function that rewards longer, stable sequences.
    
    - Higher score is better.
    - Penalizes sequences that fail grammar checks.
    """
    if not verify_convergence_for_sequence(sequence)[0]:
        return -1000.0  # Heavy penalty for instability

    is_safe, risk, _ = verify_bifurcation_risk_for_sequence(sequence)
    if risk > 0.5 and not is_safe:
        return -500.0  # Penalty for high, unhandled risk

    # Reward length and diversity
    score = len(sequence) + len(set(sequence))
    return float(score)


def find_optimal_sequence_greedy(
    initial_sequence: Sequence,
    possible_glyphs: List[Glyph],
    objective_fn: ObjectiveFunction,
    max_iterations: int = 100
) -> Tuple[Sequence, float]:
    """
    Finds an optimal sequence using a simple greedy best-first search.

    At each step, it tries adding each possible glyph to the current sequence
    and chooses the one that results in the highest score from the objective
    function. It stops if no single addition provides an improvement.

    Args:
        initial_sequence: The starting sequence.
        possible_glyphs: A list of glyphs that can be added.
        objective_fn: The function to maximize.
        max_iterations: The maximum number of steps to take.

    Returns:
        (best_sequence, best_score)
    """
    current_sequence = initial_sequence
    current_score = objective_fn(current_sequence)

    for _ in range(max_iterations):
        best_next_candidate = None
        best_next_score = current_score

        # Find the best single glyph to add in this iteration
        for glyph_to_add in possible_glyphs:
            candidate = current_sequence + [glyph_to_add]
            score = objective_fn(candidate)
            if score > best_next_score:
                best_next_score = score
                best_next_candidate = candidate
        
        # If the best move found improves the score, commit to it.
        # Otherwise, we have reached a local optimum and should stop.
        if best_next_candidate:
            current_sequence = best_next_candidate
            current_score = best_next_score
        else:
            break
            
    return current_sequence, current_score
