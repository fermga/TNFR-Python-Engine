"""Symbolic calculus for declared nodal rows and supplied pressure laws.

These identities retain the assumptions stated by each helper. Sequence
admission belongs to ``tnfr.operators.grammar`` and structural graph fields to
``tnfr.physics.fields``; symbolic integration does not certify an executed
trajectory. ``tnfr.mathematics`` re-exports these calculus helpers alongside
its numerical tools.
"""

from .._version import __version__
from . import symbolic

# Import main symbolic functions for easy access
from .symbolic import (
    check_convergence_exponential,
    compute_second_derivative_symbolic,
    get_nodal_equation,
    integrated_evolution_symbolic,
    latex_export,
    pretty_print,
    solve_nodal_equation_constant_params,
)

__all__: list[str] = [
    # Symbolic calculus
    "get_nodal_equation",
    "solve_nodal_equation_constant_params",
    "integrated_evolution_symbolic",
    "check_convergence_exponential",
    "compute_second_derivative_symbolic",
    "latex_export",
    "pretty_print",
    # Submodules
    "symbolic",
    # Package metadata
    "__version__",
]
