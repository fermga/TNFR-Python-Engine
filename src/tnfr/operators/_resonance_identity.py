"""Pure identity predicates shared by the Resonance runtime and audit."""

from __future__ import annotations

import cmath
import math
from typing import Iterable

from ..utils import angle_diff
from ._phase_gate import phase_limit_is_canonical

__all__ = [
    "RA_RUNTIME_AMPLIFICATION_TRIGGER",
    "normalize_resonance_epi_kind",
    "resonance_identity_failures",
    "resonance_kind_identity_compatible",
    "resonance_neighbor_circular_mean",
    "resonance_phase_limit_compatible",
    "resonance_proposed_epi_kind",
    "resonance_sign",
    "resonance_sign_identity_compatible",
    "validate_resonance_runtime_factors",
]


# Existing engine activation threshold for the conditional RA capacity boost.
# This is operational binary64 policy, not a derived structural constant.
RA_RUNTIME_AMPLIFICATION_TRIGGER = 1e-9


def normalize_resonance_epi_kind(value: object) -> str:
    """Normalize ``None`` and the empty string as an untracked EPI kind."""

    return "" if value is None or value == "" else str(value)


def validate_resonance_runtime_factors(
    mix_factor: float,
    vf_amplification_factor: float,
    phase_coupling_factor: float,
) -> tuple[str, ...]:
    """Return failures of the RA convex propagation/amplification domain."""

    from .factor_contracts import (
        GlyphFactorValidationError,
        validate_glyph_factor,
    )

    failures = []
    for key, value in (
        ("RA_epi_diff", mix_factor),
        ("RA_vf_amplification", vf_amplification_factor),
        ("RA_phase_coupling", phase_coupling_factor),
    ):
        try:
            validate_glyph_factor(key, value)
        except GlyphFactorValidationError as exc:
            failures.append(str(exc))
    return tuple(failures)


def resonance_phase_limit_compatible(limit: float, canonical_limit: float) -> bool:
    """Whether an RA phase gate is finite and no weaker than the canonical gate."""

    return phase_limit_is_canonical(limit, canonical_limit)


def _has_exact_antipodal_pairing(phases: tuple[float, ...]) -> bool:
    """Detect an exactly balanced unweighted antipodal phase multiset."""

    if not phases or len(phases) % 2:
        return False
    unmatched = list(range(len(phases)))
    while unmatched:
        left = unmatched.pop(0)
        partner_position = next(
            (
                position
                for position, right in enumerate(unmatched)
                if abs(angle_diff(phases[left], phases[right])) == math.pi
            ),
            None,
        )
        if partner_position is None:
            return False
        unmatched.pop(partner_position)
    return True


def resonance_neighbor_circular_mean(
    phases: Iterable[float],
) -> tuple[float | None, bool]:
    """Return RA's circular mean with an operational undefined-resultant guard.

    Exact antipodal pairing is checked in angle space because libm evaluates
    ``exp(i*pi/2)`` with a tiny real residual.  This avoids selecting an
    arbitrary direction for that sufficient cancellation class without a
    numerical tolerance.  The separate exactly-zero binary64 check catches
    further represented cancellations; this is not a universal symbolic test
    for every mathematically cancelling phase multiset.
    """

    values = tuple(phases)
    if not values:
        return None, False
    if _has_exact_antipodal_pairing(values):
        return None, False
    phasors = tuple(cmath.exp(1j * theta) for theta in values)
    resultant = complex(
        sum(value.real for value in phasors) / len(phasors),
        sum(value.imag for value in phasors) / len(phasors),
    )
    if resultant.real == 0.0 and resultant.imag == 0.0:
        return None, False
    phase = cmath.phase(resultant)
    if phase < 0.0:
        phase += 2.0 * math.pi
    return phase, True


def resonance_sign(value: float) -> int:
    """Return the strict sign class ``-1``, ``0``, or ``1``."""

    return -1 if value < 0.0 else (1 if value > 0.0 else 0)


def resonance_sign_identity_compatible(before: float, proposed: float) -> bool:
    """Whether RA avoids a strict negative-to-positive or positive-to-negative flip.

    Exact zero is the neutral boundary: arriving at zero or departing from zero
    does not invert a pre-existing nonzero sign identity.
    """

    before_sign = resonance_sign(before)
    proposed_sign = resonance_sign(proposed)
    return before_sign == 0 or proposed_sign == 0 or before_sign == proposed_sign


def resonance_kind_identity_compatible(before: str, proposed: str) -> bool:
    """Whether a proposed RA kind preserves an established nonempty kind."""

    return not before or proposed == before


def resonance_proposed_epi_kind(
    current_kind: str,
    neighbor_value_kinds: Iterable[tuple[float, str]],
    proposed_target_epi: float,
    *,
    fallback_kind: str = "RA",
) -> str:
    """Reproduce the runtime's dominant-neighbour kind selection without mutation."""

    best_kind = ""
    best_abs = 0.0
    found_neighbor = False
    for value, kind in neighbor_value_kinds:
        found_neighbor = True
        magnitude = abs(value)
        if magnitude > best_abs:
            best_abs = magnitude
            best_kind = kind

    if not found_neighbor:
        return current_kind or fallback_kind
    dominant = best_kind or fallback_kind
    proposed = dominant if best_abs > abs(proposed_target_epi) else current_kind
    return proposed or fallback_kind


def resonance_identity_failures(
    before_epi: float,
    proposed_epi: float,
    before_kind: str,
    proposed_kind: str,
) -> tuple[str, ...]:
    """Return independent failed RA identity clauses in stable order."""

    failures = []
    if not resonance_sign_identity_compatible(before_epi, proposed_epi):
        failures.append("nonzero_epi_sign_would_flip")
    if not resonance_kind_identity_compatible(before_kind, proposed_kind):
        failures.append("established_epi_kind_would_change")
    return tuple(failures)
