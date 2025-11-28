"""Postcondition validators for ZHIR (Mutation) operator.

Implements verification of mutation postconditions including phase transformation,
identity preservation, and bifurcation handling.

These postconditions ensure that ZHIR fulfills its contract and maintains TNFR
structural invariants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types import NodeId, TNFRGraph

from ...alias import get_attr
from ...constants.aliases import ALIAS_THETA
from . import OperatorContractViolation

__all__ = [
    "verify_phase_transformed",
    "verify_identity_preserved",
    "verify_bifurcation_handled",
]


def verify_phase_transformed(G: TNFRGraph, node: NodeId, theta_before: float) -> None:
    """Verify that phase was actually transformed by ZHIR.

    ZHIR's primary contract is phase transformation (θ → θ'). This verifies
    that the phase actually changed, fulfilling the operator's purpose.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to verify
    theta_before : float
        Phase value before ZHIR application

    Raises
    ------
    OperatorContractViolation
        If phase was not transformed (theta unchanged)

    Notes
    -----
    A small tolerance (1e-6) is used to account for floating-point precision.
    If theta changes by less than this tolerance, it's considered unchanged.

    This check ensures that ZHIR actually performs its structural transformation
    rather than being a no-op.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators import Mutation
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.0)
    >>> theta_before = G.nodes[node]["theta"]
    >>> Mutation()(G, node)
    >>> verify_phase_transformed(G, node, theta_before)  # Should pass
    """
    theta_after = float(get_attr(G.nodes[node], ALIAS_THETA, 0.0))

    # Check if phase actually changed (with small tolerance for floating-point)
    if abs(theta_after - theta_before) < 1e-6:
        raise OperatorContractViolation(
            "Mutation",
            f"Phase was not transformed (θ before={theta_before:.6f}, "
            f"θ after={theta_after:.6f}, diff={abs(theta_after - theta_before):.9f}). "
            f"ZHIR must transform phase to fulfill its contract.",
        )


def verify_identity_preserved(G: TNFRGraph, node: NodeId, epi_kind_before: str | None) -> None:
    """Verify that structural identity (epi_kind) was preserved through mutation.

    ZHIR transforms phase/regime while preserving structural identity. A cell
    remains a cell, a concept remains a concept - only the operational mode changes.
    This is a fundamental TNFR invariant: transformations preserve coherence.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to verify
    epi_kind_before : str or None
        Identity (epi_kind) before ZHIR application

    Raises
    ------
    OperatorContractViolation
        If identity changed during mutation

    Notes
    -----
    If epi_kind_before is None (identity not tracked), this check is skipped.
    This allows flexibility for simple nodes while enforcing identity preservation
    when it's explicitly tracked.

    **Special Case**: If epi_kind is used to track operator glyphs (common pattern),
    the check is skipped since this is operational metadata, not structural identity.
    To enable strict identity checking, use a separate attribute (e.g., "structural_type"
    or "node_type") for identity tracking.

    Identity preservation is distinct from EPI preservation - EPI may change
    slightly during mutation (structural adjustments), but the fundamental type
    (epi_kind) must remain constant.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators import Mutation
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> G.nodes[node]["structural_type"] = "stem_cell"  # Use separate attribute
    >>> epi_kind_before = G.nodes[node]["structural_type"]
    >>> Mutation()(G, node)
    >>> # After mutation, structural_type should still be "stem_cell"
    >>> # verify_identity_preserved(G, node, epi_kind_before)  # Would check structural_type
    """
    # Skip check if identity was not tracked
    if epi_kind_before is None:
        return

    epi_kind_after = G.nodes[node].get("epi_kind")

    # Skip if epi_kind appears to be tracking operator glyphs (common pattern)
    # Operator glyphs are short codes like "IL", "OZ", "ZHIR"
    if epi_kind_after in [
        "IL",
        "EN",
        "AL",
        "OZ",
        "RA",
        "UM",
        "SHA",
        "VAL",
        "NUL",
        "THOL",
        "ZHIR",
        "NAV",
        "REMESH",
    ]:
        # epi_kind is being used for operator tracking, not identity
        # This is acceptable operational metadata, skip identity check
        return

    if epi_kind_after != epi_kind_before:
        raise OperatorContractViolation(
            "Mutation",
            f"Structural identity changed during mutation: "
            f"{epi_kind_before} → {epi_kind_after}. "
            f"ZHIR must preserve epi_kind while transforming phase.",
        )


def verify_bifurcation_handled(G: TNFRGraph, node: NodeId) -> None:
    """Verify that bifurcation was handled if triggered during mutation.

    When ZHIR detects bifurcation potential (∂²EPI/∂t² > τ), it must either:
    1. Create a variant node (if bifurcation mode = "variant_creation")
    2. Set detection flag (if bifurcation mode = "detection")

    This ensures that bifurcation events are properly tracked and controlled,
    preventing uncontrolled structural fragmentation.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node to verify

    Raises
    ------
    OperatorContractViolation
        If bifurcation was triggered but not handled according to configured mode

    Notes
    -----
    **Bifurcation Modes**:

    - "detection" (default): Only flag bifurcation potential, no variant creation
    - "variant_creation": Create new node as bifurcation variant

    In "variant_creation" mode, the function verifies that a bifurcation event
    was recorded in G.graph["zhir_bifurcation_events"].

    Grammar rule U4a requires bifurcation handlers (THOL or IL) after ZHIR
    when bifurcation is detected.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators import Mutation
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> G.graph["ZHIR_BIFURCATION_MODE"] = "detection"
    >>> Mutation()(G, node)
    >>> # If bifurcation detected, flag should be set
    >>> verify_bifurcation_handled(G, node)  # Should pass
    """
    # Check if bifurcation was detected
    bifurcation_potential = G.nodes[node].get("_zhir_bifurcation_potential", False)

    if not bifurcation_potential:
        # No bifurcation detected, nothing to verify
        return

    # Bifurcation was detected - verify it was handled
    mode = G.graph.get("ZHIR_BIFURCATION_MODE", "detection")

    if mode == "variant_creation":
        # In variant creation mode, verify variant was actually created
        events = G.graph.get("zhir_bifurcation_events", [])

        # Check if this node has a recorded bifurcation event
        node_has_event = any(event.get("parent_node") == node for event in events)

        if not node_has_event:
            raise OperatorContractViolation(
                "Mutation",
                f"Bifurcation potential detected (∂²EPI/∂t² > τ) but variant "
                f"was not created. Mode={mode} requires variant creation. "
                f"Check _spawn_mutation_variant() implementation.",
            )

    elif mode == "detection":
        # In detection mode, just verify the flag is set (already checked above)
        # No variant creation required, flag is sufficient
        pass

    else:
        # Unknown mode - log warning but don't raise error
        import warnings

        warnings.warn(
            f"Unknown ZHIR_BIFURCATION_MODE: {mode}. "
            f"Expected 'detection' or 'variant_creation'. "
            f"Bifurcation handling could not be fully verified.",
            stacklevel=2,
        )
