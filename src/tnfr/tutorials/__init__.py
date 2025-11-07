"""Interactive tutorials for learning TNFR step-by-step.

This module provides guided, hands-on tutorials that introduce TNFR
concepts progressively. Each tutorial:

1. Explains TNFR concepts in plain language
2. Shows working code examples
3. Displays real-time results with interpretation
4. Builds from simple to advanced concepts
5. Maintains full TNFR theoretical fidelity

All tutorials respect TNFR canonical invariants and can be run
independently or as a learning sequence.
"""

from __future__ import annotations

__all__ = [
    "hello_tnfr",
    "biological_example",
    "social_network_example",
    "technology_example",
    "team_communication_example",
    "adaptive_ai_example",
    "oz_dissonance_tutorial",
    "run_all_tutorials",
]

from .interactive import (
    hello_tnfr,
    biological_example,
    social_network_example,
    technology_example,
    team_communication_example,
    adaptive_ai_example,
    oz_dissonance_tutorial,
    run_all_tutorials,
)
