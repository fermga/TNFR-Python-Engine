"""Strict precondition validation for OZ (Dissonance) operator.

This module implements canonical precondition validation for the Dissonance (OZ)
structural operator according to TNFR canonical theory. OZ requires specific
structural conditions to enable productive dissonance without premature collapse:

1. **Minimum coherence base (EPI)**: Node must have sufficient structural form to
   withstand disruption (EPI >= threshold)
2. **ΔNFR not critically high**: Avoid sobrecarga (overload) when ΔNFR already extreme
3. **Sufficient νf**: Structural frequency must support reorganization response
4. **No overload pattern**: Detect sobrecarga disonante (multiple OZ without resolution)
5. **Network connectivity**: Warn if isolated (bifurcation requires alternative paths)

These validations implement the canonical warning from TNFR.pdf:
> "OZ debe ir precedido por estabilización adecuada (IL). Genera nodos frágiles o
> fallidos si se aplica sobre estructuras sin coherencia interna."

References
----------
- TNFR.pdf §2.3.6: Catálogo de errores arquetípicos (Sobrecarga disonante)
- TNFR.pdf §2.3.3: Reglas sintácticas R4 (Bifurcación)
- Tabla de validación estructural: Condiciones de activación de OZ
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...types import TNFRGraph

__all__ = ["validate_dissonance_strict"]


def validate_dissonance_strict(G: TNFRGraph, node: Any) -> None:
    """Validate strict canonical preconditions for OZ (Dissonance) operator.

    According to TNFR canonical theory, Dissonance (OZ - Disonancia) requires:

    1. **Coherence base**: EPI >= threshold (sufficient structure to withstand disruption)
    2. **ΔNFR safety**: |ΔNFR| < critical (avoid overload/collapse)
    3. **Active νf**: νf >= threshold (capacity to respond to dissonance)
    4. **No overload**: < 2 consecutive OZ without resolution (IL/THOL/NUL)
    5. **Network connectivity**: degree >= 1 (warning only - enables bifurcation paths)

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : Any
        Node identifier for validation

    Raises
    ------
    ValueError
        If EPI < minimum (insufficient coherence base - apply IL first)
        If |ΔNFR| > maximum (reorganization pressure critical - apply IL first)
        If νf < minimum (cannot respond to dissonance)
        If overload detected (>= 2 OZ without resolver operators)

    Warnings
    --------
    UserWarning
        If node has low connectivity (degree < 1) - bifurcation paths limited

    Notes
    -----
    Thresholds are configurable via graph metadata:
    - ``G.graph["OZ_MIN_EPI"]``: Minimum EPI (default 0.2)
    - ``G.graph["OZ_MAX_DNFR"]``: Maximum |ΔNFR| (default 0.8)
    - ``G.graph["OZ_MIN_VF"]``: Minimum νf (default 0.1)
    - ``G.graph["OZ_MIN_DEGREE"]``: Minimum degree for warning (default 1)

    **Resolver operators** that integrate dissonance (reset overload counter):
    - IL (Coherence): Stabilizes structure
    - THOL (Self-organization): Creates sub-EPIs
    - NUL (Contraction): Simplifies structure

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.preconditions.dissonance import validate_dissonance_strict
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> G.nodes[node]["dnfr"] = 0.3
    >>> validate_dissonance_strict(G, node)  # OK - sufficient coherence base

    >>> G2, node2 = create_nfr("weak", epi=0.05, vf=1.0)
    >>> validate_dissonance_strict(G2, node2)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: OZ precondition failed: EPI=0.050 < 0.2. Insufficient coherence base.

    See Also
    --------
    tnfr.operators.preconditions : Base precondition validators
    tnfr.operators.definitions.Dissonance : Dissonance operator implementation
    """
    import warnings

    from ...alias import get_attr
    from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF

    # Get current node state
    epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
    dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
    degree = G.degree(node)

    # Get configurable thresholds
    min_epi = float(G.graph.get("OZ_MIN_EPI", 0.2))
    max_dnfr = float(G.graph.get("OZ_MAX_DNFR", 0.8))
    min_vf = float(G.graph.get("OZ_MIN_VF", 0.1))
    min_degree = int(G.graph.get("OZ_MIN_DEGREE", 1))

    # Precondition 1: Minimum coherence base (EPI)
    # OZ requires existing structure to perturb - without it, OZ causes collapse
    if epi < min_epi:
        raise ValueError(
            f"OZ precondition failed: EPI={epi:.3f} < {min_epi:.3f}. "
            f"Insufficient coherence base to withstand dissonance. "
            f"Suggestion: Apply IL (Coherence) first to stabilize node before introducing dissonance."
        )

    # Precondition 2: ΔNFR not critically high
    # Applying OZ when ΔNFR already extreme creates sobrecarga (overload) → collapse
    if abs(dnfr) > max_dnfr:
        raise ValueError(
            f"OZ precondition failed: |ΔNFR|={abs(dnfr):.3f} > {max_dnfr:.3f}. "
            f"Reorganization pressure already critical - applying OZ risks collapse. "
            f"Suggestion: Apply IL (Coherence) first to reduce ΔNFR before introducing more dissonance."
        )

    # Precondition 3: Sufficient νf for reorganization response
    # OZ triggers reorganization - node needs capacity (νf) to respond
    if vf < min_vf:
        raise ValueError(
            f"OZ precondition failed: νf={vf:.3f} < {min_vf:.3f}. "
            f"Structural frequency too low - node lacks capacity to respond to dissonance. "
            f"Suggestion: Increase νf before applying OZ."
        )

    # Precondition 4: Detect OZ overload (sobrecarga disonante)
    # Multiple OZ without resolution = entropic loop (violates R5)
    _validate_oz_no_overload(G, node)

    # Precondition 5: Network connectivity (warning only - not blocking)
    # Isolated nodes can experience dissonance but bifurcation paths are limited
    if degree < min_degree:
        warnings.warn(
            f"OZ warning: Node {node!r} has low connectivity (degree={degree} < {min_degree}). "
            f"OZ-induced bifurcation may have limited structural paths. "
            f"Consider applying UM (Coupling) first to increase network coupling.",
            UserWarning,
            stacklevel=3,
        )

    # Store validation context for telemetry
    G.nodes[node]["_oz_precondition_context"] = {
        "epi": epi,
        "dnfr": dnfr,
        "vf": vf,
        "degree": degree,
        "validation_passed": True,
    }


def _validate_oz_no_overload(G: TNFRGraph, node: Any) -> None:
    """Detect and prevent dissonance overload (sobrecarga disonante).

    Sobrecarga occurs when multiple OZ operators are applied in succession
    without intervening resolution operators (IL, THOL, NUL). This creates
    entropic loops that violate TNFR structural coherence requirements.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : Any
        Node identifier to check for overload

    Raises
    ------
    ValueError
        If >= 1 OZ application detected in recent history without resolver
        (since we're about to apply another OZ, that would make it >= 2)

    Notes
    -----
    **Resolver operators** that integrate dissonance:
    - coherence (IL): Stabilizes structure, reduces ΔNFR
    - self_organization (THOL): Creates sub-EPIs via bifurcation
    - contraction (NUL): Simplifies structure

    This check examines the last 5 operators in glyph_history to detect
    overload patterns. Since validation is called BEFORE the operator is
    applied, if we find >= 1 OZ in history without a resolver, applying
    another OZ would create overload.

    References
    ----------
    TNFR.pdf §2.3.6 - Sobrecarga disonante: "acumulación excesiva de OZ sin
    paso a mutación"
    """
    from ...operators.grammar import glyph_function_name

    # Get glyph history from node
    history = G.nodes[node].get("glyph_history", [])
    if not history:
        # No history - first operator application, cannot be overload
        return

    # Convert deque to list and examine recent history (last 5 operations)
    history_list = list(history)
    recent_history = history_list[-5:] if len(history_list) >= 5 else history_list

    # Count OZ applications in recent history
    oz_count = sum(1 for glyph in recent_history if glyph_function_name(glyph) == "dissonance")

    # If >= 1 OZ in history, check for resolver operators
    # (we're about to apply another OZ, so that would make >= 2 total)
    if oz_count >= 1:
        # Resolver operators that integrate dissonance
        resolvers = {"coherence", "self_organization", "contraction"}
        recent_names = [glyph_function_name(g) for g in recent_history]
        has_resolver = any(name in resolvers for name in recent_names)

        if not has_resolver:
            raise ValueError(
                f"OZ precondition failed: Sobrecarga disonante detected. "
                f"Found {oz_count} OZ in recent history without resolution. "
                f"Applying another OZ without resolving previous dissonance risks collapse. "
                f"Suggestion: Apply IL (Coherence), THOL (Self-organization), or NUL (Contraction) "
                f"to integrate dissonance before introducing more."
            )
