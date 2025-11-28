"""
TNFR Grammar Validators (Mathematical).

Provides formal mathematical validation for TNFR operator sequences (Glyphs)
by applying symbolic analysis from `tnfr.math.symbolic`.

Key capabilities:
- U2 CONVERGENCE: Verify integral convergence for sequences.
- U4 BIFURCATION: Analyze bifurcation risk based on sequence effects.

This module bridges the gap between computational grammar rules and the
underlying physics of the nodal equation.

Physics basis: AGENTS.md § Unified Grammar (U1-U6)
"""

from typing import List, Tuple

from ..types import Glyph
# from . import symbolic  # Not used directly, but is the conceptual basis


# ============================================================================
# GLYPH CLASSIFICATION (based on physics)
# ============================================================================

# U2: Destabilizers increase |ΔNFR|, risking divergence
DESTABILIZERS = {Glyph.OZ, Glyph.ZHIR, Glyph.VAL}

# U2: Stabilizers reduce |ΔNFR|, promoting convergence
STABILIZERS = {Glyph.IL, Glyph.THOL}

# U4a: Bifurcation triggers (high ∂²EPI/∂t²)
BIFURCATION_TRIGGERS = {Glyph.OZ, Glyph.ZHIR}

# U4a: Bifurcation handlers (control high ∂²EPI/∂t²)
BIFURCATION_HANDLERS = {Glyph.THOL, Glyph.IL}


# ============================================================================
# U2: CONVERGENCE & BOUNDEDNESS VALIDATION
# ============================================================================

def verify_convergence_for_sequence(
    sequence: List[Glyph],
    initial_growth_rate: float = 0.0,
    destabilizer_effect: float = 0.1,
    stabilizer_effect: float = -0.15,
) -> Tuple[bool, float, str]:
    """
    Verify U2 convergence for a glyph sequence.

    Models the effect of a sequence on the exponential growth rate (λ) of ΔNFR.
    A positive final λ indicates divergence risk.

    Args:
        sequence: List of TNFR Glyphs.
        initial_growth_rate: Starting growth rate λ.
        destabilizer_effect: Positive value added to λ by a destabilizer.
        stabilizer_effect: Negative value added to λ by a stabilizer.

    Returns:
        (converges, final_growth_rate, explanation)

    Physics:
        ∫ νf·ΔNFR dt must converge. If ΔNFR grows exponentially (λ > 0),
        the integral diverges. Stabilizers must counteract destabilizers
        to ensure λ ≤ 0.

    See: AGENTS.md § U2: CONVERGENCE & BOUNDEDNESS
    """
    current_growth_rate = initial_growth_rate
    
    for glyph in sequence:
        if glyph in DESTABILIZERS:
            current_growth_rate += destabilizer_effect
        elif glyph in STABILIZERS:
            current_growth_rate += stabilizer_effect
            
    converges = current_growth_rate <= 0
    
    if converges:
        explanation = (
            f"✓ U2 SATISFIED: Sequence is net-stabilizing or neutral. "
            f"Final λ = {current_growth_rate:.2f} ≤ 0."
        )
    else:
        explanation = (
            f"⚠️ U2 VIOLATION: Sequence is net-destabilizing. "
            f"Final λ = {current_growth_rate:.2f} > 0. Needs more stabilizers."
        )
        
    return converges, current_growth_rate, explanation


# ============================================================================
# U4: BIFURCATION DYNAMICS VALIDATION
# ============================================================================

def verify_bifurcation_risk_for_sequence(
    sequence: List[Glyph],
    trigger_effect: float = 0.6,
    handler_effect: float = -0.4,
    risk_threshold: float = 0.5,
    window_size: int = 3,
) -> Tuple[bool, float, str]:
    """
    Verify U4 bifurcation risk for a glyph sequence.

    Models the accumulation of bifurcation risk. Triggers (OZ, ZHIR) increase
    risk, while handlers (IL, THOL) mitigate it.

    Args:
        sequence: List of TNFR Glyphs.
        trigger_effect: Risk added by a trigger.
        handler_effect: Risk reduced by a handler.
        risk_threshold: The level of risk considered significant.

    Returns:
        (is_safe, risk_level, explanation)

    Physics:
    Uncontrolled bifurcation (high ∂²EPI/∂t²) leads to chaos.
    U4a requires that triggers be matched by handlers.
    This function provides a heuristic measure of that balance.
    """
    risk_level = 0.0
    pending_triggers: List[int] = []

    for idx, glyph in enumerate(sequence):
        if glyph in BIFURCATION_TRIGGERS:
            risk_level += trigger_effect
            pending_triggers.append(idx)
            continue

        if glyph in BIFURCATION_HANDLERS:
            matched_trigger = None
            for trigger_idx in pending_triggers:
                if idx - trigger_idx <= window_size:
                    matched_trigger = trigger_idx
                    break

            if matched_trigger is not None:
                pending_triggers.remove(matched_trigger)
            
            risk_level += handler_effect
            risk_level = max(0.0, risk_level)

    risk_level = max(0.0, risk_level)
    unhandled_triggers = len(pending_triggers) > 0
    risk_is_high = risk_level > risk_threshold

    if unhandled_triggers:
        explanation = (
            f"⚠️ U4 VIOLATION: Risk ({risk_level:.2f}) includes unhandled "
            f"triggers beyond {window_size} glyphs."
        )
    elif risk_is_high:
        explanation = (
            f"✓ U4 MITIGATED: Elevated risk ({risk_level:.2f}) is handled, "
            f"monitor for sustained acceleration."
        )
    else:
        explanation = (
            f"✓ U4 SATISFIED: Bifurcation risk is low ({risk_level:.2f})."
        )

    is_safe = not unhandled_triggers and not risk_is_high
    return is_safe, risk_level, explanation


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TNFR Mathematical Grammar Validators")
    print("=" * 70)

    # --- U2 Convergence Examples ---
    print("\n--- U2: CONVERGENCE & BOUNDEDNESS ---")
    
    # Unsafe sequence: Destabilizer without stabilizer
    unsafe_seq_u2 = [Glyph.AL, Glyph.OZ, Glyph.RA]
    print(f"\nAnalyzing sequence: {[g.value for g in unsafe_seq_u2]}")
    converges, _, explanation = verify_convergence_for_sequence(unsafe_seq_u2)
    print(explanation)

    # Safe sequence: Destabilizer followed by stabilizer
    safe_seq_u2 = [Glyph.AL, Glyph.OZ, Glyph.IL, Glyph.RA]
    print(f"\nAnalyzing sequence: {[g.value for g in safe_seq_u2]}")
    converges, _, explanation = verify_convergence_for_sequence(safe_seq_u2)
    print(explanation)

    # --- U4 Bifurcation Examples ---
    print("\n--- U4: BIFURCATION DYNAMICS ---")

    # Unsafe sequence: Trigger without handler
    unsafe_seq_u4 = [Glyph.EN, Glyph.OZ, Glyph.UM]
    print(f"\nAnalyzing sequence: {[g.value for g in unsafe_seq_u4]}")
    is_safe, risk, explanation = verify_bifurcation_risk_for_sequence(
        unsafe_seq_u4
    )
    print(explanation)

    # Safe sequence: Trigger followed by handler
    safe_seq_u4 = [Glyph.EN, Glyph.OZ, Glyph.THOL, Glyph.UM]
    print(f"\nAnalyzing sequence: {[g.value for g in safe_seq_u4]}")
    is_safe, risk, explanation = verify_bifurcation_risk_for_sequence(
        safe_seq_u4
    )
    print(explanation)
    
    print("\n" + "=" * 70)
    print("✓ Grammar validation examples completed.")
    print("=" * 70)
