"""TNFR Grammar: U6 Structural Potential Validation

U6: STRUCTURAL POTENTIAL CONFINEMENT - Validate Δ Φ_s < φ (drift threshold).

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from typing import Any

from ..constants.canonical import PHI  # φ ≈ 1.618 (golden ratio)
from ..mathematics.unified_numerical import np
from .grammar_types import StructuralPotentialConfinementError

# ============================================================================
# U6: Structural Potential Confinement (CANONICAL as of 2025-11-11)
# ============================================================================


def validate_structural_potential_confinement(
    G: Any,
    phi_s_before: dict[Any, float],
    phi_s_after: dict[Any, float],
    threshold: float = PHI,  # φ (golden ratio) - canonical escape threshold
    strict: bool = True,
) -> tuple[bool, float, str]:
    """Validate U6: STRUCTURAL POTENTIAL CONFINEMENT.

    Checks that structural potential drift Δ Φ_s remains below escape threshold,
    ensuring system stays confined in potential well and avoids fragmentation.

    Canonical Status: CANONICAL (promoted 2025-11-11)
    - 2,400+ experiments, 5 topology families
    - corr(Δ Φ_s, ΔC) = -0.822 (R² ≈ 0.68)
    - Perfect universality: CV = 0.1%

    Parameters
    ----------
    G : TNFRGraph
        Network graph (used for node iteration)
    phi_s_before : dict[NodeId, float]
        Structural potential before sequence application
    phi_s_after : dict[NodeId, float]
        Structural potential after sequence application
    threshold : float, default=1.618
        Canonical threshold for Δ Φ_s (drift). Above φ (golden ratio), fragmentation risk.
        Rigorously derived from canonical TNFR constants (φ escape threshold).
    strict : bool, default=True
        If True, raises StructuralPotentialConfinementError on violation.
        If False, returns (False, drift, message) without raising.

    Returns
    -------
    valid : bool
        True if Δ Φ_s < threshold (safe regime)
    drift : float
        Measured Δ Φ_s = mean(|Φ_s_after[i] - Φ_s_before[i]|)
    message : str
        Human-readable validation result

    Raises
    ------
    StructuralPotentialConfinementError
        If Δ Φ_s ≥ threshold and strict=True

    Physical Interpretation
    -----------------------
    Φ_s minima = passive equilibrium states (potential wells).
    Grammar-valid sequences naturally maintain small Δ Φ_s (~0.6).
    Large Δ Φ_s (~3.9) indicates escape from well → fragmentation risk.

    Grammar U1-U5 acts as passive confinement mechanism (not active attractor):
    - Reduces drift by 85% (valid 0.6 vs violation 3.9)
    - No force pulling back, only resistance to escape

    Safety Criteria (Classical)
    --------------------------
    This function validates **drift** Δ Φ_s = mean(|Φ_s_after - Φ_s_before|):
    - Δ Φ_s < φ ≈ 1.618: Confined (U6 satisfied)
    - Δ Φ_s ≥ φ ≈ 1.618: Escape risk (U6 violated)
    - Valid sequences: Δ Φ_s ≈ 0.6 (37% of φ threshold)
    - Violations: Δ Φ_s ≈ 3.9 (240% of φ threshold)

    Separate criterion (absolute value, checked elsewhere):
    - |Φ_s| < 0.771 (Von Koch): Safe regime per node
    - See: PHI_S_VON_KOCH_THRESHOLD in constants/canonical.py

    Examples
    --------
    >>> from tnfr.physics.fields import compute_structural_potential
    >>> phi_before = compute_structural_potential(G)
    >>> apply_sequence(G, [Emission(), Coherence(), Silence()])
    >>> phi_after = compute_structural_potential(G)
    >>> from tnfr.config.defaults_core import STRUCTURAL_ESCAPE_THRESHOLD
    >>> valid, drift, msg = validate_structural_potential_confinement(
    ...     G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
    ... )
    >>> print(f"Valid: {valid}, Drift: {drift:.3f}")
    Valid: True, Drift: 0.583

    >>> # With strict=True (default), raises on violation
    >>> try:
    ...     validate_structural_potential_confinement(G, phi_before, phi_bad)
    ... except StructuralPotentialConfinementError as e:
    ...     print(f"U6 violation: {e}")

    References
    ----------
    - UNIFIED_GRAMMAR_RULES.md § U6: Complete physics derivation
    - docs/STRUCTURAL_FIELDS_TETRAD.md: Validation evidence
    - AGENTS.md § Structural Fields: Canonical status
    - src/tnfr/physics/fields.py: compute_structural_potential()

    """

    # Compute drift as mean absolute change
    nodes = list(G.nodes())
    if not nodes:
        return True, 0.0, "U6: No nodes, trivially satisfied"

    drifts = []
    for node in nodes:
        phi_before_i = phi_s_before.get(node, 0.0)
        phi_after_i = phi_s_after.get(node, 0.0)
        drifts.append(abs(phi_after_i - phi_before_i))

    delta_phi_s = float(np.mean(drifts))

    # Validate against threshold
    valid = delta_phi_s < threshold

    if valid:
        msg = (
            f"U6: PASS - Δ Φ_s = {delta_phi_s:.3f} < {threshold:.3f} (confined). "
            f"System remains in safe regime."
        )
        return True, delta_phi_s, msg
    else:
        msg = (
            f"U6: FAIL - Δ Φ_s = {delta_phi_s:.3f} ≥ {threshold:.3f} (escape). "
            f"Fragmentation risk. Valid sequences maintain Δ Φ_s ≈ 0.6."
        )
        if strict:
            raise StructuralPotentialConfinementError(
                delta_phi_s=delta_phi_s,
                threshold=threshold,
                sequence=None,  # Sequence not available in this context
            )
        return False, delta_phi_s, msg
