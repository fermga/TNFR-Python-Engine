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

from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_VF, ALIAS_D2EPI

__all__ = [
    "NodalEquationViolation",
    "validate_nodal_equation",
    "compute_expected_depi_dt",
    "compute_d2epi_dt2",
]

# Default tolerance for nodal equation validation
DEFAULT_NODAL_EQUATION_TOLERANCE = 1e-3
DEFAULT_NODAL_EQUATION_CLIP_AWARE = True


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


def _get_node_attr(
    G: TNFRGraph, node: NodeId, aliases: tuple[str, ...], default: float = 0.0
) -> float:
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
    clip_aware: bool | None = None,
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
    clip_aware : bool, optional
        If True, validates using structural_clip to account for boundary
        preservation: EPI_expected = structural_clip(EPI_theoretical).
        If False, uses classic mode without clip adjustment.
        If None, uses graph configuration or default (True).

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

    **Clip-aware mode** (default): When structural_clip intervenes to preserve
    boundaries, the actual EPI differs from the theoretical prediction. This
    mode accounts for boundary preservation by applying structural_clip to the
    theoretical value before comparison:

        EPI_expected = structural_clip(EPI_before + νf · ΔNFR · dt)

    This ensures validation passes when clip interventions are legitimate parts
    of the operator's structural boundary preservation.

    **Classic mode** (clip_aware=False): Validates without clip adjustment,
    useful for detecting when unexpected clipping occurs.

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

    if clip_aware is None:
        # Try graph configuration first, then use default
        clip_aware = G.graph.get("NODAL_EQUATION_CLIP_AWARE", DEFAULT_NODAL_EQUATION_CLIP_AWARE)

    # Measured rate of EPI change
    measured_depi_dt = (epi_after - epi_before) / dt if dt > 0 else 0.0

    # Expected rate from nodal equation: νf · ΔNFR
    # Use post-transformation values as they represent the new structural state
    expected_depi_dt = compute_expected_depi_dt(G, node)

    if clip_aware:
        # Clip-aware mode: apply structural_clip to theoretical EPI before comparison
        from ..dynamics.structural_clip import structural_clip

        # Get structural boundaries from graph configuration
        epi_min = float(G.graph.get("EPI_MIN", -1.0))
        epi_max = float(G.graph.get("EPI_MAX", 1.0))
        clip_mode = G.graph.get("CLIP_MODE", "hard")

        # Compute theoretical EPI based on nodal equation
        epi_theoretical = epi_before + (expected_depi_dt * dt)

        # Validate and normalize clip_mode
        clip_mode_str = str(clip_mode).lower()
        if clip_mode_str not in ("hard", "soft"):
            clip_mode_str = "hard"  # Default to safe fallback

        # Apply structural_clip to get expected EPI (what the operator should produce)
        epi_expected = structural_clip(
            epi_theoretical, lo=epi_min, hi=epi_max, mode=clip_mode_str  # type: ignore[arg-type]
        )

        # Validate against clipped expected value
        error = abs(epi_after - epi_expected)
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
                    "epi_theoretical": epi_theoretical,
                    "epi_expected": epi_expected,
                    "dt": dt,
                    "vf": vf,
                    "dnfr": dnfr,
                    "error": error,
                    "clip_aware": True,
                    "clip_intervened": abs(epi_theoretical - epi_expected) > 1e-10,
                },
            )
    else:
        # Classic mode: validate rate of change directly
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
                    "clip_aware": False,
                },
            )

    return is_valid


def compute_d2epi_dt2(G: "TNFRGraph", node: "NodeId") -> float:
    """Compute ∂²EPI/∂t² (structural acceleration).

    According to TNFR canonical theory (§2.3.3, R4), bifurcation occurs when
    structural acceleration exceeds threshold τ:
        |∂²EPI/∂t²| > τ → multiple reorganization paths viable

    This function computes the second-order time derivative of EPI using
    finite differences from the node's EPI history. The acceleration indicates
    how rapidly the rate of structural change is itself changing, which is
    the key indicator of bifurcation readiness.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier to compute acceleration for

    Returns
    -------
    float
        Structural acceleration ∂²EPI/∂t². Positive values indicate accelerating
        growth, negative values indicate accelerating contraction. Magnitude
        indicates bifurcation potential.

    Notes
    -----
    **Computation method:**

    Uses second-order finite difference approximation:
        ∂²EPI/∂t² ≈ (EPI_t - 2·EPI_{t-1} + EPI_{t-2}) / Δt²

    For discrete operator applications with Δt=1:
        ∂²EPI/∂t² ≈ EPI_t - 2·EPI_{t-1} + EPI_{t-2}

    **History requirements:**

    Requires at least 3 historical EPI values stored in node's `_epi_history`
    attribute. If insufficient history exists, returns 0.0 (no acceleration).

    The computed value is automatically stored in the node's `D2_EPI` attribute
    (using ALIAS_D2EPI aliases) for telemetry and metrics collection.

    **Physical interpretation:**

    - **d2epi ≈ 0**: Steady structural evolution (constant rate)
    - **d2epi > τ**: Positive acceleration, expanding reorganization
    - **d2epi < -τ**: Negative acceleration, collapsing reorganization
    - **|d2epi| > τ**: Bifurcation active, multiple paths viable

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Emission, Dissonance
    >>> from tnfr.operators.nodal_equation import compute_d2epi_dt2
    >>>
    >>> G, node = create_nfr("test", epi=0.2, vf=1.0)
    >>>
    >>> # Build EPI history through operator applications
    >>> Emission()(G, node)  # EPI increases
    >>> Emission()(G, node)  # EPI increases more
    >>> Dissonance()(G, node)  # Introduce instability
    >>>
    >>> # Compute acceleration
    >>> d2epi = compute_d2epi_dt2(G, node)
    >>>
    >>> # Check if bifurcation threshold exceeded
    >>> tau = G.graph.get("OZ_BIFURCATION_THRESHOLD", 0.5)
    >>> bifurcation_active = abs(d2epi) > tau

    See Also
    --------
    tnfr.dynamics.bifurcation.compute_bifurcation_score : Uses d2epi for scoring
    tnfr.operators.metrics.dissonance_metrics : Reports d2epi in OZ metrics
    tnfr.operators.preconditions.validate_dissonance : Checks d2epi for bifurcation
    """
    # Get EPI history from node
    history = G.nodes[node].get("_epi_history", [])

    if len(history) < 3:
        # Insufficient history for second derivative
        # Need at least 3 points: t-2, t-1, t
        return 0.0

    # Extract last 3 EPI values
    epi_t = history[-1]  # Current (most recent)
    epi_t1 = history[-2]  # One step ago
    epi_t2 = history[-3]  # Two steps ago

    # Second-order finite difference (assuming dt=1 for discrete operators)
    # ∂²EPI/∂t² ≈ (EPI_t - 2·EPI_{t-1} + EPI_{t-2}) / dt²
    # For dt=1: ∂²EPI/∂t² ≈ EPI_t - 2·EPI_{t-1} + EPI_{t-2}
    d2epi = epi_t - 2.0 * epi_t1 + epi_t2

    # Store in node for telemetry (using set_attr to handle aliases)
    set_attr(G.nodes[node], ALIAS_D2EPI, d2epi)

    return float(d2epi)
