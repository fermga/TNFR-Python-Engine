"""Type stubs for canonical TNFR nodal equation implementation."""

from __future__ import annotations

from typing import NamedTuple

from ..types import GraphLike

__all__ = (
    "NodalEquationResult",
    "compute_canonical_nodal_derivative",
    "validate_structural_frequency",
    "validate_nodal_gradient",
)

class NodalEquationResult(NamedTuple):
    """Result of canonical nodal equation evaluation."""

    derivative: float
    nu_f: float
    delta_nfr: float
    validated: bool

def compute_canonical_nodal_derivative(
    nu_f: float,
    delta_nfr: float,
    *,
    validate_units: bool = True,
    graph: GraphLike | None = None,
) -> NodalEquationResult:
    """Compute ∂EPI/∂t using the canonical TNFR nodal equation."""
    ...

def validate_structural_frequency(
    nu_f: float,
    *,
    graph: GraphLike | None = None,
) -> float:
    """Validate that structural frequency is in valid range."""
    ...

def validate_nodal_gradient(
    delta_nfr: float,
    *,
    graph: GraphLike | None = None,
) -> float:
    """Validate that nodal gradient is well-defined."""
    ...
