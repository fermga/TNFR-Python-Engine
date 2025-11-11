"""Algebraic properties and validation for TNFR structural operators.

Based on TNFR.pdf Section 3.2.4 - "Notación funcional de operadores glíficos".

This module implements formal validation of algebraic properties for structural
operators in the TNFR glyphic algebra, particularly focusing on SHA (Silence)
as the identity element in structural composition.

Theoretical Foundation
----------------------
From TNFR.pdf §3.2.4 (p. 227-230) and the nodal equation ∂EPI/∂t = νf · ΔNFR(t):

1. **SHA as Structural Identity**:
   SHA(g(ω)) ≈ g(ω) for structure (EPI)

   Physical basis: SHA reduces νf → 0, making ∂EPI/∂t → 0. This freezes
   structural evolution, preserving whatever structure g created.

2. **Idempotence**:
   SHA^n = SHA for all n ≥ 1

   Physical basis: Once νf ≈ 0, further SHA applications cannot reduce it more.
   The effect is saturated.

3. **Commutativity with NUL**:
   SHA ∘ NUL = NUL ∘ SHA

   Physical basis: SHA and NUL reduce orthogonal dimensions (νf vs EPI complexity).
   Order of reduction doesn't affect final state.

Category Theory Context
-----------------------
In the categorical framework (p. 231), SHA acts as identity morphism for
the structural component:
- Objects: Nodal configurations ω_i
- Morphisms: Structural operators g: ω_i → ω_j
- Identity: SHA: ω → ω (preserves structure)
- Property: SHA ∘ g ≈ g (for EPI component)

Note: SHA is NOT full identity (it modifies νf). It's identity for the
structural aspect (EPI), not the dynamic aspect (νf).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId
    from .definitions import Operator

__all__ = [
    "validate_identity_property",
    "validate_idempotence",
    "validate_commutativity_nul",
]


def validate_identity_property(
    G: TNFRGraph,
    node: NodeId,
    operator: Operator,
    tolerance: float = 0.01,
) -> bool:
    """Validate that SHA acts as identity for structure after operator.

    Tests the algebraic property: SHA(g(ω)) ≈ g(ω) for EPI

    This validates that applying SHA preserves the structural state (EPI)
    achieved by the operator. SHA acts as a "pause" that freezes νf but
    does not alter the structural form EPI.

    Physical basis: From ∂EPI/∂t = νf · ΔNFR, when SHA reduces νf → 0,
    structural evolution stops but current structure is preserved.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : NodeId
        Target node identifier
    operator : Operator
        Operator to test with SHA (must be valid generator like Emission)
    tolerance : float, optional
        Numerical tolerance for EPI comparison (default: 0.01)
        Relaxed due to grammar-required intermediate operators

    Returns
    -------
    bool
        True if identity property holds within tolerance

    Notes
    -----
    Due to TNFR grammar constraints (U1b: must end with closure,
    U2: must include stabilizer), we test identity by comparing:

    [Legacy note: Previously referenced C1-C2. See docs/grammar/DEPRECATION-INDEX.md]

    - Path 1: operator → Coherence → Dissonance (OZ terminator)
    - Path 2: operator → Coherence → Silence (SHA terminator)

    Both preserve structure after Coherence. If SHA is identity,
    EPI should be equivalent in both paths.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import Emission
    >>> from tnfr.operators.algebra import validate_identity_property
    >>> G, node = create_nfr("test", epi=0.5, vf=1.0)
    >>> validate_identity_property(G, node, Emission())  # doctest: +SKIP
    True
    """
    from ..alias import get_attr
    from ..constants.aliases import ALIAS_EPI
    from .definitions import Silence, Coherence, Dissonance
    from ..structural import run_sequence

    # Path 1: operator → Coherence → Dissonance (without SHA)
    # Valid grammar: generator → stabilizer → terminator
    G1 = G.copy()
    run_sequence(G1, node, [operator, Coherence(), Dissonance()])
    epi_without_sha = float(get_attr(G1.nodes[node], ALIAS_EPI, 0.0))

    # Path 2: operator → Coherence → Silence (SHA as terminator)
    # Valid grammar: generator → stabilizer → terminator
    G2 = G.copy()
    run_sequence(G2, node, [operator, Coherence(), Silence()])
    epi_with_sha = float(get_attr(G2.nodes[node], ALIAS_EPI, 0.0))

    # SHA should preserve the structural result (EPI) from operator → coherence
    # Both terminators should leave structure intact after stabilization
    return abs(epi_without_sha - epi_with_sha) < tolerance


def validate_idempotence(
    G: TNFRGraph,
    node: NodeId,
    tolerance: float = 0.05,
) -> bool:
    """Validate that SHA is idempotent: SHA^n = SHA.

    Tests the algebraic property: SHA(SHA(ω)) ≈ SHA(ω)

    Physical basis: Once νf ≈ 0 after first SHA, subsequent applications
    cannot reduce it further. The effect is saturated.

    Due to grammar constraints against consecutive SHA operators, we test
    idempotence by comparing SHA behavior in different sequence contexts.
    The key property: SHA always has the same characteristic effect
    (reduce νf to minimum, preserve EPI).

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : NodeId
        Target node identifier
    tolerance : float, optional
        Numerical tolerance for νf comparison (default: 0.05)

    Returns
    -------
    bool
        True if idempotence holds (consistent SHA behavior)

    Notes
    -----
    Tests SHA in two different contexts:
    - Context 1: Emission → Coherence → Silence
    - Context 2: Emission → Coherence → Resonance → Silence

    In both cases, SHA should reduce νf to near-zero and preserve EPI.
    This validates idempotent behavior: SHA effect is consistent and saturated.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.algebra import validate_idempotence
    >>> G, node = create_nfr("test", epi=0.65, vf=1.30)
    >>> validate_idempotence(G, node)  # doctest: +SKIP
    True
    """
    from ..alias import get_attr
    from ..constants.aliases import ALIAS_VF
    from .definitions import Silence, Emission, Coherence, Resonance
    from ..structural import run_sequence

    # Test 1: SHA after simple sequence
    G1 = G.copy()
    run_sequence(G1, node, [Emission(), Coherence(), Silence()])
    vf_context1 = float(get_attr(G1.nodes[node], ALIAS_VF, 0.0))

    # Test 2: SHA after longer sequence (with Resonance added)
    G2 = G.copy()
    run_sequence(G2, node, [Emission(), Coherence(), Resonance(), Silence()])
    vf_context2 = float(get_attr(G2.nodes[node], ALIAS_VF, 0.0))

    # Idempotence property: SHA behavior is consistent
    # Both νf values should be near-zero (SHA's characteristic effect)
    vf_threshold = 0.15  # SHA should reduce νf below this
    both_minimal = (vf_context1 < vf_threshold) and (vf_context2 < vf_threshold)

    # Both should be similar (consistent behavior)
    consistent = abs(vf_context1 - vf_context2) < tolerance

    return both_minimal and consistent


def validate_commutativity_nul(
    G: TNFRGraph,
    node: NodeId,
    tolerance: float = 0.02,
) -> bool:
    """Validate that SHA and NUL commute: SHA(NUL(ω)) ≈ NUL(SHA(ω)).

    Tests the algebraic property that Silence and Contraction can be applied
    in either order with equivalent results.

    Physical basis: SHA and NUL reduce orthogonal dimensions of state space:
    - SHA reduces νf (reorganization capacity)
    - NUL reduces EPI complexity (structural dimensionality)

    Since they act on independent dimensions, order doesn't matter for
    final state.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node to validate
    node : NodeId
        Target node identifier
    tolerance : float, optional
        Numerical tolerance for EPI and νf comparison (default: 0.02)

    Returns
    -------
    bool
        True if commutativity holds within tolerance

    Notes
    -----
    Tests two paths (both grammar-valid, using Transition as generator):
    1. Transition → Silence → Contraction
    2. Transition → Contraction → Silence

    The property holds if both paths result in equivalent EPI and νf values.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.algebra import validate_commutativity_nul
    >>> G, node = create_nfr("test", epi=0.55, vf=1.10)
    >>> validate_commutativity_nul(G, node)  # doctest: +SKIP
    True
    """
    from ..alias import get_attr
    from ..constants.aliases import ALIAS_EPI, ALIAS_VF
    from .definitions import Silence, Contraction, Transition
    from ..structural import run_sequence

    # Path 1: NAV → SHA → NUL (Transition then Silence then Contraction)
    G1 = G.copy()
    run_sequence(G1, node, [Transition(), Silence(), Contraction()])
    epi_sha_nul = float(get_attr(G1.nodes[node], ALIAS_EPI, 0.0))
    vf_sha_nul = float(get_attr(G1.nodes[node], ALIAS_VF, 0.0))

    # Path 2: NAV → NUL → SHA (Transition then Contraction then Silence)
    G2 = G.copy()
    run_sequence(G2, node, [Transition(), Contraction(), Silence()])
    epi_nul_sha = float(get_attr(G2.nodes[node], ALIAS_EPI, 0.0))
    vf_nul_sha = float(get_attr(G2.nodes[node], ALIAS_VF, 0.0))

    # Validate commutativity: both paths should produce similar results
    epi_commutes = abs(epi_sha_nul - epi_nul_sha) < tolerance
    vf_commutes = abs(vf_sha_nul - vf_nul_sha) < tolerance

    return epi_commutes and vf_commutes
