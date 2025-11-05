"""Nodal equation validation for TNFR structural operators.

This module provides validation for the fundamental TNFR nodal equation:

    ∂EPI/∂t = νf · ΔNFR(t)

This equation governs how the Primary Information Structure (EPI) evolves
over time based on the structural frequency (νf) and internal reorganization
operator (ΔNFR). All structural operator applications must respect this
canonical relationship to maintain TNFR theoretical fidelity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR

__all__ = [
    "NodalEquationViolation",
    "validate_nodal_equation",
    "compute_expected_depi_dt",
]

# Default tolerance for nodal equation validation
DEFAULT_NODAL_EQUATION_TOLERANCE = 1e-3


class NodalEquationViolation(Exception):
    """Raised when operator application violates the nodal equation.
    
    The nodal equation ∂EPI/∂t = νf · ΔNFR(t) is the fundamental equation
    governing node evolution in TNFR. Violations indicate non-canonical
    structural transformations.
    """

    def __init__(
        self,
        operator: str,
        measured_depi_dt: float,
        expected_depi_dt: float,
        tolerance: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize nodal equation violation.

        Parameters
        ----------
        operator : str
            Name of the operator that caused the violation
        measured_depi_dt : float
            Measured ∂EPI/∂t from before/after states
        expected_depi_dt : float
            Expected ∂EPI/∂t from νf · ΔNFR(t)
        tolerance : float
            Tolerance threshold that was exceeded
        details : dict, optional
            Additional diagnostic information
        """
        self.operator = operator
        self.measured_depi_dt = measured_depi_dt
        self.expected_depi_dt = expected_depi_dt
        self.tolerance = tolerance
        self.details = details or {}
        
        error = abs(measured_depi_dt - expected_depi_dt)
        super().__init__(
            f"Nodal equation violation in {operator}: "
            f"|∂EPI/∂t_measured - νf·ΔNFR| = {error:.3e} > {tolerance:.3e}\n"
            f"  Measured: {measured_depi_dt:.6f}\n"
            f"  Expected: {expected_depi_dt:.6f}"
        )


def _get_node_attr(G: TNFRGraph, node: NodeId, aliases: tuple[str, ...], default: float = 0.0) -> float:
    """Get node attribute using alias fallback."""
    return float(get_attr(G.nodes[node], aliases, default))


def compute_expected_depi_dt(G: TNFRGraph, node: NodeId) -> float:
    """Compute expected ∂EPI/∂t from current νf and ΔNFR values.
    
    Implements the canonical TNFR nodal equation:
        ∂EPI/∂t = νf · ΔNFR(t)
    
    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to compute expected rate for
        
    Returns
    -------
    float
        Expected rate of EPI change (∂EPI/∂t)
        
    Notes
    -----
    The structural frequency (νf) is in Hz_str (structural hertz) units,
    and ΔNFR is the dimensionless internal reorganization operator.
    Their product gives the rate of structural reorganization.
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    return vf * dnfr


def validate_nodal_equation(
    G: TNFRGraph,
    node: NodeId,
    epi_before: float,
    epi_after: float,
    dt: float,
    *,
    operator_name: str = "unknown",
    tolerance: float | None = None,
    strict: bool = False,
) -> bool:
    """Validate that EPI change respects the nodal equation.
    
    Verifies that the change in EPI between before and after states
    matches the prediction from the nodal equation:
    
        ∂EPI/∂t = νf · ΔNFR(t)
    
    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node that underwent transformation
    epi_before : float
        EPI value before operator application
    epi_after : float
        EPI value after operator application
    dt : float
        Time step (typically 1.0 for discrete operator applications)
    operator_name : str, optional
        Name of the operator for error reporting
    tolerance : float, optional
        Absolute tolerance for equation validation.
        If None, uses graph configuration or default (1e-3).
    strict : bool, default False
        If True, raises NodalEquationViolation on failure.
        If False, returns validation result without raising.
        
    Returns
    -------
    bool
        True if equation is satisfied within tolerance, False otherwise
        
    Raises
    ------
    NodalEquationViolation
        If strict=True and validation fails
        
    Notes
    -----
    The nodal equation is validated using the post-transformation νf and ΔNFR
    values, as these represent the structural state after the operator effect.
    
    For discrete operator applications, dt is typically 1.0, making the
    validation equivalent to: (epi_after - epi_before) ≈ νf_after · ΔNFR_after
    
    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0, dnfr=0.1)
    >>> epi_before = G.nodes[node]["EPI"]
    >>> # Apply some transformation...
    >>> epi_after = G.nodes[node]["EPI"]
    >>> is_valid = validate_nodal_equation(G, node, epi_before, epi_after, dt=1.0)
    """
    if tolerance is None:
        # Try graph configuration first, then use default constant
        tolerance = float(G.graph.get("NODAL_EQUATION_TOLERANCE", DEFAULT_NODAL_EQUATION_TOLERANCE))
    
    # Measured rate of EPI change
    measured_depi_dt = (epi_after - epi_before) / dt if dt > 0 else 0.0
    
    # Expected rate from nodal equation: νf · ΔNFR
    # Use post-transformation values as they represent the new structural state
    expected_depi_dt = compute_expected_depi_dt(G, node)
    
    # Check if equation is satisfied within tolerance
    error = abs(measured_depi_dt - expected_depi_dt)
    is_valid = error <= tolerance
    
    if not is_valid and strict:
        vf = _get_node_attr(G, node, ALIAS_VF)
        dnfr = _get_node_attr(G, node, ALIAS_DNFR)
        
        raise NodalEquationViolation(
            operator=operator_name,
            measured_depi_dt=measured_depi_dt,
            expected_depi_dt=expected_depi_dt,
            tolerance=tolerance,
            details={
                "epi_before": epi_before,
                "epi_after": epi_after,
                "dt": dt,
                "vf": vf,
                "dnfr": dnfr,
                "error": error,
            },
        )
    
    return is_valid
