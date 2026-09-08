"""Precondition validators for TNFR structural operators.

Each operator has specific requirements that must be met before execution
to maintain TNFR structural invariants. This package provides validators
for each of the 13 canonical operators.

The preconditions package has been restructured to support both legacy
imports (from ..preconditions import validate_*) and new modular imports
(from ..preconditions.emission import validate_emission_strict).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types import NodeId, TNFRGraph
    import logging

from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ...constants.canonical import VAL_MIN_EPI

__all__ = [
    "OperatorPreconditionError",
    "validate_phase_gate_u3",
    "validate_emission",
    "validate_reception",
    "validate_coherence",
    "validate_dissonance",
    "validate_coupling",
    "validate_resonance",
    "validate_silence",
    "validate_expansion",
    "validate_contraction",
    "validate_self_organization",
    "validate_mutation",
    "validate_transition",
    "validate_recursivity",
    "diagnose_coherence_readiness",
    "diagnose_resonance_readiness",
    "diagnose_mutation_readiness",
]


class OperatorPreconditionError(Exception):
    """Raised when an operator's preconditions are not met."""

    def __init__(self, operator: str, reason: str) -> None:
        """Initialize precondition error.

        Parameters
        ----------
        operator : str
            Name of the operator that failed validation
        reason : str
            Description of why the precondition failed
        """
        self.operator = operator
        self.reason = reason
        super().__init__(f"{operator}: {reason}")


from ..metrics_core import get_node_attr as _get_node_attr


def validate_emission(G: "TNFRGraph", node: "NodeId") -> None:
    """AL - Emission requires node in latent or low activation state.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If EPI is already too high for emission to be meaningful
    """
    epi = _get_node_attr(G, node, ALIAS_EPI)
    # Emission is meant to activate latent nodes, not boost already active ones
    # This is a soft threshold - configurable via graph metadata
    max_epi = float(G.graph.get("AL_MAX_EPI_FOR_EMISSION", 0.8))
    if epi >= max_epi:
        raise OperatorPreconditionError(
            "Emission", f"Node already active (EPI={epi:.3f} >= {max_epi:.3f})"
        )


def validate_reception(G: "TNFRGraph", node: "NodeId") -> None:
    """EN - Reception requires node to have neighbors to receive from.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If node has no neighbors to receive energy from
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        raise OperatorPreconditionError(
            "Reception", "Node has no neighbors to receive energy from"
        )


def validate_coherence(G: "TNFRGraph", node: "NodeId") -> None:
    """IL - Coherence requires active EPI, νf, and manageable ΔNFR.

    This function delegates to the strict validation implementation
    in coherence.py module, which provides comprehensive canonical
    precondition checks according to TNFR.pdf §2.2.1.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    ValueError
        If critical preconditions are not met (active EPI, νf, non-saturated state)

    Warnings
    --------
    UserWarning
        For suboptimal conditions (zero ΔNFR, critical ΔNFR, isolated node)

    Notes
    -----
    For backward compatibility, this function maintains the same signature
    as the legacy validate_coherence but now provides enhanced validation.

    See Also
    --------
    tnfr.operators.preconditions.coherence.validate_coherence_strict : Full implementation
    """
    from .coherence import validate_coherence_strict

    validate_coherence_strict(G, node)


def diagnose_coherence_readiness(G: "TNFRGraph", node: "NodeId") -> dict:
    """Diagnose node readiness for IL (Coherence) operator.

    Provides comprehensive diagnostic report with readiness status and
    actionable recommendations for IL operator application.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to diagnose

    Returns
    -------
    dict
        Diagnostic report with readiness status, check results, values, and recommendations

    See Also
    --------
    tnfr.operators.preconditions.coherence.diagnose_coherence_readiness : Full implementation
    """
    from .coherence import diagnose_coherence_readiness as _diagnose

    return _diagnose(G, node)


def validate_dissonance(G: "TNFRGraph", node: "NodeId") -> None:
    """OZ - Dissonance requires comprehensive structural preconditions.

    This function delegates to the strict validation implementation in
    dissonance.py module, which provides canonical precondition checks:

    1. Minimum coherence base (EPI >= threshold)
    2. ΔNFR not critically high (avoid overload)
    3. Sufficient νf for reorganization response
    4. No overload pattern (sobrecarga disonante)
    5. Network connectivity (warning)

    Also detects bifurcation readiness when ∂²EPI/∂t² > τ, enabling
    alternative structural paths (ZHIR, NUL, IL, THOL).

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If critical preconditions are not met (EPI, ΔNFR, νf, overload)

    Notes
    -----
    For backward compatibility, this function maintains the same signature
    as the legacy validate_dissonance but now provides enhanced validation.

    When bifurcation threshold is exceeded, sets node['_bifurcation_ready'] = True
    and logs the event for telemetry.

    See Also
    --------
    tnfr.operators.preconditions.dissonance.validate_dissonance_strict : Full implementation
    """
    import logging

    logger = logging.getLogger(__name__)

    # First, apply strict canonical preconditions
    # This validates EPI, ΔNFR, νf, overload, and connectivity
    from .dissonance import validate_dissonance_strict

    try:
        validate_dissonance_strict(G, node)
    except ValueError as e:
        # Convert ValueError to OperatorPreconditionError for backward compatibility
        raise OperatorPreconditionError(
            "Dissonance", str(e).replace("OZ precondition failed: ", "")
        )

    # Check bifurcation readiness using existing THOL infrastructure
    # Reuse _compute_epi_acceleration from SelfOrganization
    from ..definitions import SelfOrganization

    thol_instance = SelfOrganization()
    d2_epi = thol_instance._compute_epi_acceleration(G, node)

    # Get bifurcation threshold
    tau = float(G.graph.get("BIFURCATION_THRESHOLD_TAU", 0.5))

    # Store d²EPI for telemetry (using existing ALIAS_D2EPI)
    from ...alias import set_attr
    from ...constants.aliases import ALIAS_D2EPI

    set_attr(G.nodes[node], ALIAS_D2EPI, d2_epi)

    # Check if bifurcation threshold exceeded
    if d2_epi > tau:
        # Mark node as bifurcation-ready
        G.nodes[node]["_bifurcation_ready"] = True
        logger.info(
            f"Node {node}: bifurcation threshold exceeded "
            f"(∂²EPI/∂t²={d2_epi:.3f} > τ={tau}). "
            f"Alternative structural paths enabled."
        )
    else:
        # Clear flag if previously set
        G.nodes[node]["_bifurcation_ready"] = False


def validate_phase_gate_u3(
    G: "TNFRGraph", node: "NodeId", operator: str
) -> None:
    """U3 hard invariant: UM/RA require a phase-compatible neighbour.

    Canonical Invariant #2 / grammar U3: Coupling and Resonance are admissible
    only under the resonance condition ``|wrap(φ_i − φ_j)| ≤ Δφ_max`` with the
    canonical gate ``DELTA_PHI_MAX = π/2``.  This check runs unconditionally
    before any state mutation and **raises** (it is not a warning and cannot be
    disabled via ``VALIDATE_OPERATOR_PRECONDITIONS``).

    A runtime target must have at least one neighbour within the effective
    gate; otherwise the operator would be a no-op or couple/propagate into
    destructive phase opposition. ``UM_MAX_PHASE_DIFF`` may tighten the UM
    gate but cannot widen the hard graph limit.
    """
    from ...alias import get_attr
    from .._phase_gate import U3PhaseGateError, resolve_u3_phase_neighbors

    operator_code = (
        "UM" if str(operator).casefold() in {"um", "coupling"} else "RA"
    )
    def phase(candidate: "NodeId") -> object:
        return get_attr(
            G.nodes[candidate],
            ALIAS_THETA,
            None,
            strict=True,
            conv=lambda value: value,
        )
    try:
        resolve_u3_phase_neighbors(
            G.graph,
            phase(node),
            G.neighbors(node),
            phase_getter=phase,
            operator_code=operator_code,
        )
    except U3PhaseGateError as exc:
        raise OperatorPreconditionError(
            operator, f"U3 phase gate: {exc}"
        ) from exc


def validate_coupling(G: "TNFRGraph", node: "NodeId") -> None:
    """UM - Coupling requires active nodes with compatible phases.

    Validates comprehensive canonical preconditions for the UM (Coupling) operator
    according to TNFR theory:

    1. **Graph connectivity**: At least one other node exists for coupling
    2. **Active EPI**: Node has sufficient structural form (EPI > threshold)
    3. **Structural frequency**: Node has capacity for synchronization (νf > threshold)
    4. **Phase compatibility** (MANDATORY per Invariant #2): At least one neighbor within phase range

    Configuration Parameters
    ------------------------
    UM_MIN_EPI : float, default 0.05
        Minimum EPI magnitude required for coupling
    UM_MIN_VF : float, default 0.01
        Minimum structural frequency required for coupling
    UM_STRICT_PHASE_CHECK : bool
        Retained as legacy configuration only. It cannot disable the hard U3
        runtime invariant.
    UM_MAX_PHASE_DIFF : float, default π/2
        Maximum phase difference for compatible coupling (radians)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If node state is unsuitable for coupling:
        - Graph has no other nodes
        - EPI below threshold
        - Structural frequency below threshold
        - No phase-compatible neighbors

    Notes
    -----
    Phase compatibility is mandatory and cannot be disabled by legacy flags.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions import validate_coupling
    >>>
    >>> # Valid node for coupling
    >>> G, node = create_nfr("active", epi=0.15, vf=0.50)
    >>> validate_coupling(G, node)  # Passes
    >>>
    >>> # Invalid: EPI too low
    >>> G, node = create_nfr("inactive", epi=0.02, vf=0.50)
    >>> validate_coupling(G, node)  # Raises OperatorPreconditionError

    See Also
    --------
    Coupling : UM operator that uses this validation
    AGENTS.md : Invariant #2 (phase check mandatory)
    UNIFIED_GRAMMAR_RULES.md : U3 derivation

    [Legacy: Previously referenced EMERGENT_GRAMMAR_ANALYSIS.md RC3]
    """
    # Basic graph check - at least one other node required
    if G.number_of_nodes() <= 1:
        raise OperatorPreconditionError(
            "Coupling", "Graph has no other nodes to couple with"
        )

    # Node must be active (non-zero EPI)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    min_epi = float(G.graph.get("UM_MIN_EPI", 0.05))
    if abs(epi) < min_epi:
        raise OperatorPreconditionError(
            "Coupling",
            f"Node EPI too low for coupling (|EPI|={abs(epi):.3f} < {min_epi:.3f})",
        )

    # Node must have structural frequency capacity
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("UM_MIN_VF", 0.01))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Coupling", f"Structural frequency too low (νf={vf:.3f} < {min_vf:.3f})"
        )

    # U3 is a hard runtime invariant even when configurable preconditions are
    # disabled. Calling this validator directly must preserve that contract.
    validate_phase_gate_u3(G, node, "Coupling")


def validate_resonance(G: "TNFRGraph", node: "NodeId") -> None:
    """RA - Resonance requires comprehensive canonical preconditions.

    This function delegates to the strict validation implementation in
    resonance.py module, which provides canonical precondition checks:

    1. Coherent source EPI (minimum structural form)
    2. Network connectivity (edges for propagation)
    3. Phase compatibility with neighbors (synchronization)
    4. Controlled dissonance (stable resonance state)
    5. Sufficient νf (propagation capacity)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    ValueError
        If critical preconditions are not met (EPI, connectivity, νf, ΔNFR)

    Warnings
    --------
    UserWarning
        For suboptimal conditions (phase misalignment, isolated node)

    Notes
    -----
    For backward compatibility, this function maintains the same signature
    as the legacy validate_resonance but now provides enhanced validation.

    Typical canonical sequences that satisfy RA preconditions:
    - UM → RA: Coupling followed by propagation
    - AL → RA: Emission followed by propagation
    - IL → RA: Coherence stabilized then propagated

    See Also
    --------
    tnfr.operators.preconditions.resonance.validate_resonance_strict : Full implementation
    """
    from .resonance import validate_resonance_strict

    validate_resonance_strict(G, node)


def diagnose_resonance_readiness(G: "TNFRGraph", node: "NodeId") -> dict:
    """Diagnose node readiness for RA (Resonance) operator.

    Provides comprehensive diagnostic report with readiness status and
    actionable recommendations for RA operator application.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to diagnose

    Returns
    -------
    dict
        Diagnostic report with readiness status, check results, values, and recommendations

    See Also
    --------
    tnfr.operators.preconditions.resonance.diagnose_resonance_readiness : Full implementation
    """
    from .resonance import diagnose_resonance_readiness as _diagnose

    return _diagnose(G, node)


def validate_silence(G: "TNFRGraph", node: "NodeId") -> None:
    """SHA - Silence requires vf > 0 to reduce.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If structural frequency already near zero
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("SHA_MIN_VF", 0.01))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Silence",
            f"Structural frequency already minimal (νf={vf:.3f} < {min_vf:.3f})",
        )


def validate_expansion(G: "TNFRGraph", node: "NodeId") -> None:
    """VAL - Expansion requires comprehensive canonical preconditions.

    Canonical Requirements (TNFR Physics):
    1. **νf < max_vf**: Structural frequency below saturation
    2. **ΔNFR > 0**: Positive reorganization gradient (growth pressure)
    3. **EPI >= min_epi**: Sufficient base coherence for expansion
    4. **(Optional) Network capacity**: Check if network can support expansion

    Physical Basis:
    ----------------
    From nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

    For coherent expansion:
    - ΔNFR > 0 required: expansion needs outward pressure
    - EPI > threshold: must have coherent base to expand from
    - νf < max: must have capacity for increased reorganization

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If any precondition fails:
        - Structural frequency at maximum
        - ΔNFR non-positive (no growth pressure)
        - EPI below minimum (insufficient coherence base)
        - (Optional) Network at capacity

    Configuration Parameters
    ------------------------
    VAL_MAX_VF : float, default 10.0
        Maximum structural frequency threshold
    VAL_MIN_DNFR : float, default 1e-6
        Minimum ΔNFR for expansion (must be positive, very low to minimize breaking changes)
    VAL_MIN_EPI : float, default 1/(2π) ≈ 0.159
        Minimum EPI for coherent expansion
    VAL_CHECK_NETWORK_CAPACITY : bool, default False
        Enable network capacity validation
    VAL_MAX_NETWORK_SIZE : int, default 1000
        Maximum network size if capacity checking enabled

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions import validate_expansion
    >>>
    >>> # Valid node for expansion
    >>> G, node = create_nfr("expanding", epi=0.5, vf=2.0)
    >>> G.nodes[node]['delta_nfr'] = 0.1  # Positive ΔNFR
    >>> validate_expansion(G, node)  # Passes
    >>>
    >>> # Invalid: negative ΔNFR
    >>> G.nodes[node]['delta_nfr'] = -0.1
    >>> validate_expansion(G, node)  # Raises OperatorPreconditionError

    Notes
    -----
    VAL increases both EPI magnitude and νf, enabling exploration of new
    structural configurations while maintaining core identity (fractality).

    See Also
    --------
    Expansion : VAL operator implementation
    validate_contraction : NUL preconditions (inverse operation)
    """
    # 1. νf below maximum (existing check)
    vf = _get_node_attr(G, node, ALIAS_VF)
    max_vf = float(G.graph.get("VAL_MAX_VF", 10.0))
    if vf >= max_vf:
        raise OperatorPreconditionError(
            "Expansion",
            f"Structural frequency at maximum (νf={vf:.3f} >= {max_vf:.3f}). "
            f"Node at reorganization capacity limit.",
        )

    # 2. ΔNFR positivity check (NEW - CRITICAL)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    min_dnfr = float(G.graph.get("VAL_MIN_DNFR", 1e-6))
    if dnfr < min_dnfr:
        raise OperatorPreconditionError(
            "Expansion",
            f"ΔNFR must be positive for expansion (ΔNFR={dnfr:.3f} < {min_dnfr:.3f}). "
            f"No outward growth pressure detected. Consider OZ (Dissonance) to generate ΔNFR.",
        )

    # 3. EPI minimum check (NEW - IMPORTANT)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    min_epi = float(G.graph.get("VAL_MIN_EPI", VAL_MIN_EPI))
    if epi < min_epi:
        raise OperatorPreconditionError(
            "Expansion",
            f"EPI too low for coherent expansion (EPI={epi:.3f} < {min_epi:.3f}). "
            "Insufficient structural base. Consider AL (Emission) "
            "to source EPI first.",
        )

    # 4. Network capacity check (OPTIONAL - for large-scale systems)
    check_capacity = bool(G.graph.get("VAL_CHECK_NETWORK_CAPACITY", False))
    if check_capacity:
        max_network_size = int(G.graph.get("VAL_MAX_NETWORK_SIZE", 1000))
        current_size = G.number_of_nodes()
        if current_size >= max_network_size:
            raise OperatorPreconditionError(
                "Expansion",
                f"Network at capacity (n={current_size} >= {max_network_size}). "
                f"Cannot support further expansion. set VAL_CHECK_NETWORK_CAPACITY=False to disable.",
            )


def validate_contraction(G: "TNFRGraph", node: "NodeId") -> None:
    """NUL - Enhanced precondition validation with over-compression check.

    Canonical Requirements (TNFR Physics):
    1. **νf > min_vf**: Structural frequency above minimum for reorganization
    2. **EPI >= min_epi**: Sufficient structural form to contract safely
    3. **density <= max_density**: Not already at critical compression

    Physical Basis:
    ----------------
    From nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

    For safe contraction:
    - EPI must have sufficient magnitude (can't compress vacuum)
    - Density ρ = |ΔNFR| / EPI must not exceed critical threshold
    - Over-compression (ρ → ∞) causes structural collapse

    Density is the structural pressure per unit form. When EPI contracts
    while ΔNFR increases (canonical densification), density rises. If already
    at critical density, further contraction risks fragmentation.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If any precondition fails:
        - Structural frequency at minimum
        - EPI too low for safe contraction
        - Node already at critical density

    Configuration Parameters
    ------------------------
    NUL_MIN_VF : float, default 0.1
        Minimum structural frequency threshold
    NUL_MIN_EPI : float, default 0.1
        Minimum EPI for safe contraction
    NUL_MAX_DENSITY : float, default 10.0
        Maximum density threshold (ρ = |ΔNFR| / max(EPI, ε))

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions import validate_contraction
    >>>
    >>> # Valid node for contraction
    >>> G, node = create_nfr("contracting", epi=0.5, vf=1.0)
    >>> G.nodes[node]['delta_nfr'] = 0.2
    >>> validate_contraction(G, node)  # Passes
    >>>
    >>> # Invalid: EPI too low
    >>> G, node = create_nfr("too_small", epi=0.05, vf=1.0)
    >>> validate_contraction(G, node)  # Raises OperatorPreconditionError
    >>>
    >>> # Invalid: density too high
    >>> G, node = create_nfr("over_compressed", epi=0.1, vf=1.0)
    >>> G.nodes[node]['delta_nfr'] = 2.0  # High ΔNFR
    >>> validate_contraction(G, node)  # Raises OperatorPreconditionError

    See Also
    --------
    Contraction : NUL operator implementation
    validate_expansion : VAL preconditions (inverse operation)
    """
    vf = _get_node_attr(G, node, ALIAS_VF)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)

    # Check 1: νf must be above minimum
    min_vf = float(G.graph.get("NUL_MIN_VF", 0.1))
    if vf <= min_vf:
        raise OperatorPreconditionError(
            "Contraction",
            f"Structural frequency at minimum (νf={vf:.3f} <= {min_vf:.3f})",
        )

    # Check 2: EPI must be above minimum for contraction
    min_epi = float(G.graph.get("NUL_MIN_EPI", 0.1))
    if epi < min_epi:
        raise OperatorPreconditionError(
            "Contraction",
            f"EPI too low for safe contraction (EPI={epi:.3f} < {min_epi:.3f}). "
            f"Cannot compress structure below minimum coherent form.",
        )

    # Check 3: Density must not exceed critical threshold
    # Density ρ = |ΔNFR| / max(EPI, ε) - structural pressure per unit form
    epsilon = 1e-9
    density = abs(dnfr) / max(epi, epsilon)
    max_density = float(G.graph.get("NUL_MAX_DENSITY", 10.0))
    if density > max_density:
        raise OperatorPreconditionError(
            "Contraction",
            f"Node already at critical density (ρ={density:.3f} > {max_density:.3f}). "
            f"Further contraction risks structural collapse. "
            f"Consider IL (Coherence) to stabilize or reduce ΔNFR first.",
        )


def validate_self_organization(G: "TNFRGraph", node: "NodeId") -> None:
    """THOL - Enhanced validation: connectivity, metabolic context, acceleration.

    Self-organization requires:
    1. Sufficient EPI for bifurcation
    2. Positive reorganization pressure (ΔNFR > 0)
    3. Structural reorganization capacity (νf > 0)
    4. Network connectivity for metabolism (degree ≥ 1)
    5. EPI history for acceleration computation (≥3 points)
    6. **NEW**: Bifurcation threshold check (∂²EPI/∂t² vs τ) with telemetry

    Also detects and records the destabilizer that enabled this
    self-organization for telemetry and structural tracing purposes.

    **Bifurcation Threshold Validation (∂²EPI/∂t² > τ):**

    According to TNFR.pdf §2.2.10, THOL bifurcation occurs only when structural
    acceleration exceeds threshold τ. This function now explicitly validates this
    condition and sets telemetry flags:

    - If ∂²EPI/∂t² > τ: Bifurcation will occur (normal THOL behavior)
    - If ∂²EPI/∂t² ≤ τ: THOL executes but no sub-EPIs generated (warning logged)

    The validation is NON-BLOCKING (warning only) because THOL can meaningfully
    execute without bifurcation - it still applies coherence and metabolic effects.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If any structural requirement is not met

    Notes
    -----
    This function implements R4 Extended telemetry by analyzing the glyph_history
    to determine which destabilizer enabled the self-organization.

    Configuration Parameters
    ------------------------
    THOL_MIN_EPI : float, default 0.2
        Minimum EPI for bifurcation
    THOL_MIN_VF : float, default 0.1
        Minimum structural frequency for reorganization
    THOL_MIN_DEGREE : int, default 1
        Minimum network connectivity
    THOL_MIN_HISTORY_LENGTH : int, default 3
        Minimum EPI history for acceleration computation
    THOL_ALLOW_ISOLATED : bool, default False
        Allow isolated nodes for internal-only bifurcation
    THOL_METABOLIC_ENABLED : bool, default True
        Require metabolic network context
    BIFURCATION_THRESHOLD_TAU : float, default 0.1
        Bifurcation threshold for ∂²EPI/∂t² (see THOL_BIFURCATION_THRESHOLD)
    THOL_BIFURCATION_THRESHOLD : float, default 0.1
        Alias for BIFURCATION_THRESHOLD_TAU (operator-specific config)
    """
    import logging

    logger = logging.getLogger(__name__)

    epi = _get_node_attr(G, node, ALIAS_EPI)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    vf = _get_node_attr(G, node, ALIAS_VF)

    # 1. EPI sufficiency
    min_epi = float(G.graph.get("THOL_MIN_EPI", 0.2))
    if epi < min_epi:
        raise OperatorPreconditionError(
            "Self-organization",
            f"EPI too low for bifurcation (EPI={epi:.3f} < {min_epi:.3f})",
        )

    # 2. Reorganization pressure
    if dnfr <= 0:
        raise OperatorPreconditionError(
            "Self-organization",
            f"ΔNFR non-positive, no reorganization pressure (ΔNFR={dnfr:.3f})",
        )

    # 3. Structural frequency validation
    min_vf = float(G.graph.get("THOL_MIN_VF", 0.1))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Self-organization",
            f"Structural frequency too low for reorganization (νf={vf:.3f} < {min_vf:.3f})",
        )

    # 4. Connectivity requirement (ELEVATED FROM WARNING)
    min_degree = int(G.graph.get("THOL_MIN_DEGREE", 1))
    node_degree = G.degree(node)

    # Allow isolated THOL if explicitly enabled
    allow_isolated = bool(G.graph.get("THOL_ALLOW_ISOLATED", False))

    if node_degree < min_degree and not allow_isolated:
        raise OperatorPreconditionError(
            "Self-organization",
            f"Node insufficiently connected for network metabolism "
            f"(degree={node_degree} < {min_degree}). "
            f"set THOL_ALLOW_ISOLATED=True to enable internal-only bifurcation.",
        )

    # 5. EPI history validation (for d²EPI/dt² computation).  Use the same
    # source selector as the public acceleration diagnostic: timestamped
    # physical evidence is authoritative, followed by the canonical and
    # private unit-step compatibility histories.
    from ...errors import TNFRValueError
    from ..nodal_equation import (
        _select_acceleration_history,
        compute_d2epi_dt2,
    )

    try:
        history_source, active_history = _select_acceleration_history(
            G.nodes[node]
        )
        history_length = 0 if active_history is None else len(active_history)
    except (OverflowError, TypeError, TNFRValueError) as exc:
        raise OperatorPreconditionError(
            "Self-organization",
            "Active EPI history must be a sized, indexed history",
        ) from exc
    min_history_length = int(G.graph.get("THOL_MIN_HISTORY_LENGTH", 3))

    if history_length < min_history_length:
        raise OperatorPreconditionError(
            "Self-organization",
            f"Insufficient EPI history for acceleration computation "
            f"({history_source}: have {history_length}, need ≥{min_history_length}). "
            f"Apply operators to build history before THOL.",
        )

    try:
        d2_epi = abs(compute_d2epi_dt2(G, node, store=False))
    except TNFRValueError as exc:
        raise OperatorPreconditionError(
            "Self-organization",
            f"Invalid {history_source} acceleration evidence: {exc}",
        ) from exc

    # 6. Metabolic context validation (if metabolism enabled)
    if G.graph.get("THOL_METABOLIC_ENABLED", True):
        # If network metabolism is expected, verify neighbors exist
        if node_degree == 0:
            raise OperatorPreconditionError(
                "Self-organization",
                "Metabolic mode enabled but node is isolated. "
                "Disable THOL_METABOLIC_ENABLED or add network connections.",
            )

    # R4 Extended: Detect and record destabilizer type for telemetry
    from .mutation import record_destabilizer_context

    record_destabilizer_context(G, node, logger)

    # Bifurcation threshold validation is non-blocking: THOL can still apply
    # coherence and metabolic effects when the shared acceleration stays in
    # the closed window.

    # Get bifurcation threshold from graph configuration
    # Try BIFURCATION_THRESHOLD_TAU first (canonical), then THOL_BIFURCATION_THRESHOLD
    tau = G.graph.get("BIFURCATION_THRESHOLD_TAU")
    if tau is None:
        tau = float(G.graph.get("THOL_BIFURCATION_THRESHOLD", 0.1))
    else:
        tau = float(tau)

    # Check if bifurcation threshold will be exceeded
    if d2_epi <= tau:
        # Log warning but allow execution - THOL can be meaningful without bifurcation
        logger.warning(
            f"Node {node}: THOL applied with ∂²EPI/∂t²={d2_epi:.3f} ≤ τ={tau:.3f}. "
            f"No bifurcation will occur (empty THOL window expected). "
            f"Sub-EPIs will not be generated. "
            f"Consider stronger destabilizer (OZ, VAL) to increase acceleration."
        )
        # set telemetry flag for post-hoc analysis
        G.nodes[node]["_thol_no_bifurcation_expected"] = True
    else:
        # Clear flag if previously set
        G.nodes[node]["_thol_no_bifurcation_expected"] = False
        logger.debug(
            f"Node {node}: THOL bifurcation threshold exceeded "
            f"(∂²EPI/∂t²={d2_epi:.3f} > τ={tau:.3f}). "
            f"Sub-EPI generation expected."
        )


def validate_mutation(G: "TNFRGraph", node: "NodeId") -> None:
    """Validate the canonical ZHIR gate and optional U4b preconditions.

    The implementation is centralized in ``preconditions.mutation``.  The
    signed, strict ``dEPI/dt > xi`` trigger is always enforced; configurable
    U4b checks supplement the grammar layer when explicitly enabled.
    """
    from .mutation import validate_mutation_strict

    validate_mutation_strict(G, node)

def validate_transition(G: "TNFRGraph", node: "NodeId") -> None:
    """NAV - Comprehensive canonical preconditions for transition.

    Validates comprehensive preconditions for NAV (Transition) operator according
    to TNFR.pdf §2.3.11:

    1. **Minimum νf**: Structural frequency must exceed threshold
    2. **Controlled ΔNFR**: |ΔNFR| must be below maximum for stable transition
    3. **Regime validation**: Warns if transitioning from deep latency (EPI < 0.05)
    4. **Sequence compatibility** (optional): Warns if NAV applied after incompatible operators

    Configuration Parameters
    ------------------------
    NAV_MIN_VF : float, default 0.01
        Minimum structural frequency for transition
    NAV_MAX_DNFR : float, default 1.0
        Maximum |ΔNFR| for stable transition
    NAV_STRICT_SEQUENCE_CHECK : bool, default False
        Enable strict sequence compatibility checking
    NAV_MIN_EPI_FROM_LATENCY : float, default 0.05
        Minimum EPI for smooth transition from latency

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If node lacks necessary dynamics for transition:
        - νf below minimum threshold
        - |ΔNFR| exceeds maximum threshold (unstable state)

    Warnings
    --------
    UserWarning
        For suboptimal but valid conditions:
        - Transitioning from deep latency (EPI < 0.05)
        - NAV applied after incompatible operator (when strict checking enabled)

    Notes
    -----
    NAV requires controlled ΔNFR to prevent instability during regime transitions.
    High |ΔNFR| indicates significant reorganization pressure that could cause
    structural disruption during transition. Apply IL (Coherence) first to reduce
    ΔNFR before attempting regime transition.

    Deep latency transitions (EPI < 0.05 after SHA) benefit from prior AL (Emission)
    to provide smoother reactivation path.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions import validate_transition
    >>>
    >>> # Valid transition - controlled state
    >>> G, node = create_nfr("test", epi=0.5, vf=0.6)
    >>> G.nodes[node]['delta_nfr'] = 0.3  # Controlled ΔNFR
    >>> validate_transition(G, node)  # Passes
    >>>
    >>> # Invalid - ΔNFR too high
    >>> G.nodes[node]['delta_nfr'] = 1.5
    >>> validate_transition(G, node)  # Raises OperatorPreconditionError

    See Also
    --------
    Transition : NAV operator implementation
    validate_coherence : IL preconditions for ΔNFR reduction
    TNFR.pdf : §2.3.11 for NAV canonical requirements
    """
    import warnings

    # 1. νf minimum (existing check - preserved for backward compatibility)
    vf = _get_node_attr(G, node, ALIAS_VF)
    min_vf = float(G.graph.get("NAV_MIN_VF", 0.01))
    if vf < min_vf:
        raise OperatorPreconditionError(
            "Transition",
            f"Structural frequency too low (νf={vf:.3f} < {min_vf:.3f})",
        )

    # 2. ΔNFR positivity and bounds check (NEW - CRITICAL)
    dnfr = _get_node_attr(G, node, ALIAS_DNFR)
    max_dnfr = float(G.graph.get("NAV_MAX_DNFR", 1.0))
    if abs(dnfr) > max_dnfr:
        raise OperatorPreconditionError(
            "Transition",
            f"ΔNFR too high for stable transition (|ΔNFR|={abs(dnfr):.3f} > {max_dnfr}). "
            f"Apply IL (Coherence) first to reduce reorganization pressure.",
        )

    # 3. Regime origin validation (NEW - WARNING)
    latent = G.nodes[node].get("latent", False)
    epi = _get_node_attr(G, node, ALIAS_EPI)
    min_epi_from_latency = float(G.graph.get("NAV_MIN_EPI_FROM_LATENCY", 0.05))

    if latent and epi < min_epi_from_latency:
        # Warning: transitioning from deep latency
        warnings.warn(
            f"Node {node} in deep latency (EPI={epi:.3f} < {min_epi_from_latency:.3f}). "
            f"Consider AL (Emission) before NAV for smoother activation.",
            UserWarning,
            stacklevel=2,
        )

    # 4. Sequence compatibility check (NEW - OPTIONAL)
    if G.graph.get("NAV_STRICT_SEQUENCE_CHECK", False):
        history = G.nodes[node].get("glyph_history", [])
        if history:
            from ..grammar import glyph_function_name

            last_op = glyph_function_name(history[-1]) if history else None

            # NAV works best after stabilizers or generators
            # Valid predecessors per TNFR.pdf §2.3.11 and AGENTS.md
            valid_predecessors = {
                "emission",  # AL → NAV (activation-transition)
                "coherence",  # IL → NAV (stable-transition)
                "silence",  # SHA → NAV (latency-transition)
                "self_organization",  # THOL → NAV (bifurcation-transition)
            }

            if last_op and last_op not in valid_predecessors:
                warnings.warn(
                    f"NAV applied after {last_op}. "
                    f"More coherent after: {', '.join(sorted(valid_predecessors))}",
                    UserWarning,
                    stacklevel=2,
                )


def validate_recursivity(G: "TNFRGraph", node: "NodeId") -> None:
    """Require enough support to record a meaningful REMESH advisory.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to validate

    Raises
    ------
    OperatorPreconditionError
        If the network is too small for a REMESH advisory
    """
    # The advisory represents a network-scale request.
    min_nodes = int(G.graph.get("REMESH_MIN_NODES", 2))
    if G.number_of_nodes() < min_nodes:
        raise OperatorPreconditionError(
            "Recursivity",
            f"Network too small for remesh (n={G.number_of_nodes()} < {min_nodes})",
        )


# Import diagnostic functions from modular implementations
from .mutation import diagnose_mutation_readiness  # noqa: E402
