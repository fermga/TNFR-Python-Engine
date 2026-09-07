"""
TNFR Grammar Validators (Mathematical).

Provides formal mathematical validation for TNFR operator sequences (Glyphs)
by applying symbolic analysis from `tnfr.math.symbolic`.

Key capabilities:
- U2 BALANCE SURROGATE: Score declared stabilizer/destabilizer effects.
- U4 BIFURCATION SURROGATE: Score declared trigger/handler labels.

This module supplies symbolic heuristics alongside computational grammar rules.
Its scalar scores do not integrate the nodal equation or certify trajectories.

Physics basis: AGENTS.md § Unified Grammar (U1-U6)
"""

from ..operators.grammar_types import BIFURCATION_HANDLERS as _HANDLER_NAMES
from ..operators.grammar_types import BIFURCATION_TRIGGERS as _TRIGGER_NAMES
from ..operators.grammar_types import DESTABILIZERS as _DESTABILIZER_NAMES
from ..operators.grammar_types import STABILIZERS as _STABILIZER_NAMES
from ..operators.grammar_types import function_name_to_glyph as _to_glyph
from ..types import Glyph

# ============================================================================
# GLYPH CLASSIFICATION — derived from the canonical single source
# ============================================================================
#
# These sets are NOT redefined here: they are converted, glyph-for-glyph, from
# the canonical operator-name sets in tnfr.operators.grammar_types (the single
# source of truth, derived in config.physics_derivation).  This guarantees this
# math-layer validator can never drift from the canonical U2/U4 grammar.


# U2: Destabilizer-debt roles through pressure, phase, or capacity stress
# ({OZ, ZHIR, VAL})
DESTABILIZERS = {_to_glyph(n) for n in _DESTABILIZER_NAMES}

# U2: Stabilizer-coverage roles ({IL, THOL})
STABILIZERS = {_to_glyph(n) for n in _STABILIZER_NAMES}

# U4a: Bifurcation triggers (high ∂²EPI/∂t²)  ({OZ, ZHIR})
BIFURCATION_TRIGGERS = {_to_glyph(n) for n in _TRIGGER_NAMES}

# U4a: Bifurcation handlers (control high ∂²EPI/∂t²)  ({THOL, IL})
BIFURCATION_HANDLERS = {_to_glyph(n) for n in _HANDLER_NAMES}

# ============================================================================
# U2: CALIBRATED NET-GROWTH SURROGATE
# ============================================================================


def verify_convergence_for_sequence(
    sequence: list[Glyph],
    initial_growth_rate: float = 0.0,
    destabilizer_effect: float = 0.1,
    stabilizer_effect: float = -0.15,
) -> tuple[bool, float, str]:
    """
    Evaluate a scalar U2 net-growth surrogate for a glyph sequence.

    The compatibility function name is retained, but the result is a heuristic
    balance score. It adds fixed effects for declared U2 roles without observing
    graph state, ordering, capacity, pressure, or elapsed time. A positive final
    λ marks an uncovered score in this surrogate; it does not prove divergence.

    Args:
        sequence: list of TNFR Glyphs.
        initial_growth_rate: Starting growth rate λ.
        destabilizer_effect: Positive value added to λ by a destabilizer.
        stabilizer_effect: Negative value added to λ by a stabilizer.

    Returns:
        (passes_surrogate, final_growth_rate, explanation)

    Scope:
        Actual convergence of ``∫νf·ΔNFR dt`` requires a specified trajectory,
        time horizon, pressure law, gains, and norm. This function establishes
        none of those conditions.

    See: AGENTS.md § U2: CONVERGENCE & BOUNDEDNESS
    """
    current_growth_rate = initial_growth_rate

    for glyph in sequence:
        if glyph in DESTABILIZERS:
            current_growth_rate += destabilizer_effect
        elif glyph in STABILIZERS:
            current_growth_rate += stabilizer_effect

    passes_surrogate = current_growth_rate <= 0

    if passes_surrogate:
        explanation = (
            f"U2 scalar surrogate passes: declared net score "
            f"λ = {current_growth_rate:.2f} ≤ 0; trajectory convergence "
            "was not evaluated."
        )
    else:
        explanation = (
            f"U2 scalar surrogate fails: declared net score "
            f"λ = {current_growth_rate:.2f} > 0. Add stabilizer coverage or "
            "evaluate the executed trajectory explicitly."
        )

    return passes_surrogate, current_growth_rate, explanation


# ============================================================================
# U4: BIFURCATION DYNAMICS VALIDATION
# ============================================================================


def verify_bifurcation_risk_for_sequence(
    sequence: list[Glyph],
    trigger_effect: float = 0.6,
    handler_effect: float = -0.4,
    risk_threshold: float = 0.5,
    window_size: int = 3,
) -> tuple[bool, float, str]:
    """
    Evaluate a scalar U4 trigger/handler surrogate for a glyph sequence.

    The score treats triggers (OZ, ZHIR) as positive increments and handlers
    (IL, THOL) as negative increments. It does not inspect EPI acceleration,
    operator state, gains, or the executed trajectory.

    Args:
        sequence: list of TNFR Glyphs.
        trigger_effect: Risk added by a trigger.
        handler_effect: Risk reduced by a handler.
        risk_threshold: The level of risk considered significant.

    Returns:
        (passes_surrogate, risk_score, explanation)

    Scope:
    U4a requires declared triggers to have handler coverage. This heuristic
    scores that label balance; it neither measures ∂²EPI/∂t² nor establishes
    bifurcation, chaos, or dynamical safety.
    """
    risk_level = 0.0
    pending_triggers: list[int] = []

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
            f"U4 scalar surrogate fails: score ({risk_level:.2f}) includes unhandled "
            f"triggers beyond {window_size} glyphs."
        )
    elif risk_is_high:
        explanation = (
            f"U4 scalar surrogate remains above its selected threshold "
            f"({risk_level:.2f}) despite handler coverage."
        )
    else:
        explanation = (
            f"U4 scalar surrogate passes with declared score {risk_level:.2f}; "
            "trajectory behavior was not evaluated."
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

    # --- U2 scalar-surrogate examples ---
    print("\n--- U2: SCALAR BALANCE SURROGATE ---")

    # Unsafe sequence: Destabilizer without stabilizer
    unsafe_seq_u2 = [Glyph.AL, Glyph.OZ, Glyph.RA]
    print(f"\nAnalyzing sequence: {[g.value for g in unsafe_seq_u2]}")
    passes_surrogate, _, explanation = verify_convergence_for_sequence(unsafe_seq_u2)
    print(explanation)

    # Safe sequence: Destabilizer followed by stabilizer
    safe_seq_u2 = [Glyph.AL, Glyph.OZ, Glyph.IL, Glyph.RA]
    print(f"\nAnalyzing sequence: {[g.value for g in safe_seq_u2]}")
    passes_surrogate, _, explanation = verify_convergence_for_sequence(safe_seq_u2)
    print(explanation)

    # --- U4 Bifurcation Examples ---
    print("\n--- U4: TRIGGER/HANDLER SURROGATE ---")

    # Unsafe sequence: Trigger without handler
    unsafe_seq_u4 = [Glyph.EN, Glyph.OZ, Glyph.UM]
    print(f"\nAnalyzing sequence: {[g.value for g in unsafe_seq_u4]}")
    is_safe, risk, explanation = verify_bifurcation_risk_for_sequence(unsafe_seq_u4)
    print(explanation)

    # Safe sequence: Trigger followed by handler
    safe_seq_u4 = [Glyph.EN, Glyph.OZ, Glyph.THOL, Glyph.UM]
    print(f"\nAnalyzing sequence: {[g.value for g in safe_seq_u4]}")
    is_safe, risk, explanation = verify_bifurcation_risk_for_sequence(safe_seq_u4)
    print(explanation)

    print("\n" + "=" * 70)
    print("✓ Grammar validation examples completed.")
    print("=" * 70)
