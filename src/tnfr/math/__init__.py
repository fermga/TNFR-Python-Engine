"""
TNFR Mathematical Analysis Suite.

This package provides symbolic and numerical mathematical tools for analyzing,
verifying, and optimizing TNFR (Resonant Fractal Nature Theory) dynamics.

Modules:
    symbolic: Symbolic calculus for nodal equation derivations

All tools align with TNFR physics and preserve canonical invariants.

Cross-compatibility:
    This module is also accessible through tnfr.mathematics for unified access
    to both symbolic and numerical mathematical operations.
"""

from typing import List

__version__ = "0.1.0"

# Import main symbolic functions for easy access
from .symbolic import (
    get_nodal_equation,
    solve_nodal_equation_constant_params,
    integrated_evolution_symbolic,
    check_convergence_exponential,
    compute_second_derivative_symbolic,
    evaluate_bifurcation_risk,
    latex_export,
    pretty_print,
)

# Re-export helper modules for discoverability
from . import grammar_validators, fields_symbolic, optimizer, symbolic

__all__: List[str] = [
    # Symbolic calculus
    "get_nodal_equation",
    "solve_nodal_equation_constant_params",
    "integrated_evolution_symbolic",
    "check_convergence_exponential",
    "compute_second_derivative_symbolic",
    "evaluate_bifurcation_risk",
    "latex_export",
    "pretty_print",
    # Submodules
    "symbolic",
    "grammar_validators",
    "fields_symbolic",
    "optimizer",
    # Package metadata
    "__version__",
]
