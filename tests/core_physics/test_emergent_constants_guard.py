"""Stage-0 guard for the emergent-constant constitution.

``EMERGENT_DERIVATION_PLAN.md`` re-derived every physics constant from ``pi`` (the
sole structural scale) and the pi-coherence band ``1/(pi+1)``, removing the
obsolete frozen phi / gamma / e decimals (golden ratio, Euler-Mascheroni, Napier).
This guard pins the migrated constants to their pi-formulas and asserts that none
has regressed to a known frozen phi/gamma/e decimal -- so the obsolete constants
cannot creep back into a nodal-physics path.

Operational engine knobs (selector magnitude thresholds, cache / perf) are exempt
by the constitution: they carry no nodal-physics meaning and live on the
magnitude scale, not on the coherence scale.
"""

from __future__ import annotations

import math

import pytest

from tnfr.constants.canonical import (
    CHANNEL_WEIGHT_PRIMARY,
    CHANNEL_WEIGHT_SECONDARY,
    CHANNEL_WEIGHT_TERTIARY,
    COHERENCE_RETENTION,
    COUPLING_FINE,
    COUPLING_GENTLE,
    COUPLING_MODERATE,
    DISSONANCE_AMPLIFICATION,
    FRAGMENTATION_THRESHOLD,
    HIGH_COHERENCE_THRESHOLD,
    MID_COHERENCE_THRESHOLD,
    PI,
)

# The frozen phi / gamma / e decimals removed by the purge (audit Sec.1 decoder).
# No nodal-physics constant may equal any of these.
FROZEN_PHI_GAMMA_E_DECIMALS = (
    0.041774,  # gamma/(pi*e*phi)
    0.067592,  # gamma/(e*pi)
    0.098503,  # gamma/(pi+e)
    0.135184,  # 2*gamma/(e*pi)
    0.155215,  # gamma/(pi+gamma)
    0.183733,  # gamma/pi
    0.330365,  # ~1/(e+gamma)
    0.381966,  # 1/phi^2
    0.463881,  # e/(pi+e)
    0.490983,  # ~gamma*phi
    0.618034,  # 1/phi
    0.737061,  # phi/(phi+gamma)
    0.750575,  # (e*phi)/(pi+e)
    0.933955,  # ~pi*e/...
    1.618034,  # phi
    2.718282,  # e
    2.803171,  # phi/gamma
    3.025733,  # (phi+1)*pi/e
    5.083204,  # phi*pi
)

# Curated nodal-physics constants (channel weights, operator gains, coherence
# levels, the coupling ladder). Operational magnitude knobs are deliberately
# excluded -- they are exempt from the constitution.
PHYSICS_CONSTANTS = {
    "FRAGMENTATION_THRESHOLD": FRAGMENTATION_THRESHOLD,
    "HIGH_COHERENCE_THRESHOLD": HIGH_COHERENCE_THRESHOLD,
    "CHANNEL_WEIGHT_PRIMARY": CHANNEL_WEIGHT_PRIMARY,
    "CHANNEL_WEIGHT_SECONDARY": CHANNEL_WEIGHT_SECONDARY,
    "CHANNEL_WEIGHT_TERTIARY": CHANNEL_WEIGHT_TERTIARY,
    "COHERENCE_RETENTION": COHERENCE_RETENTION,
    "DISSONANCE_AMPLIFICATION": DISSONANCE_AMPLIFICATION,
    "COUPLING_GENTLE": COUPLING_GENTLE,
    "COUPLING_MODERATE": COUPLING_MODERATE,
    "COUPLING_FINE": COUPLING_FINE,
    "MID_COHERENCE_THRESHOLD": MID_COHERENCE_THRESHOLD,
}


class TestCoherenceBandEmergent:
    """The coherence band is the single pi-derived quantity 1/(pi+1)."""

    def test_band_edges_are_pi_derived(self) -> None:
        assert math.isclose(
            FRAGMENTATION_THRESHOLD, 1.0 / (PI + 1.0), rel_tol=1e-12
        )
        assert math.isclose(
            HIGH_COHERENCE_THRESHOLD, PI / (PI + 1.0), rel_tol=1e-12
        )


class TestChannelWeightsEmergent:
    """DNFR_WEIGHTS / SI_WEIGHTS use the exact-normalising pi-band hierarchy."""

    def test_channel_weights_are_the_pi_band_hierarchy(self) -> None:
        assert math.isclose(
            CHANNEL_WEIGHT_PRIMARY, PI / (PI + 1.0), rel_tol=1e-12
        )
        assert math.isclose(
            CHANNEL_WEIGHT_SECONDARY, PI / (PI + 1.0) ** 2, rel_tol=1e-12
        )
        assert math.isclose(
            CHANNEL_WEIGHT_TERTIARY, 1.0 / (PI + 1.0) ** 2, rel_tol=1e-12
        )

    def test_channel_weights_normalise_exactly_to_one(self) -> None:
        total = (
            CHANNEL_WEIGHT_PRIMARY
            + CHANNEL_WEIGHT_SECONDARY
            + CHANNEL_WEIGHT_TERTIARY
        )
        assert math.isclose(total, 1.0, abs_tol=1e-12)


class TestOperatorGainsEmergent:
    """The pressure-lever gains are the coherence-band step and its reciprocal."""

    def test_pressure_lever_is_band_reciprocal(self) -> None:
        assert math.isclose(
            COHERENCE_RETENTION, PI / (PI + 1.0), rel_tol=1e-12
        )
        assert math.isclose(
            DISSONANCE_AMPLIFICATION, (PI + 1.0) / PI, rel_tol=1e-12
        )

    def test_balanced_il_oz_is_exactly_isometric(self) -> None:
        # pi/(pi+1) * (pi+1)/pi == 1 (a balanced IL.OZ preserves the norm).
        product = COHERENCE_RETENTION * DISSONANCE_AMPLIFICATION
        assert math.isclose(product, 1.0, abs_tol=1e-12)

    def test_coupling_ladder_is_pi_fractions(self) -> None:
        assert math.isclose(COUPLING_GENTLE, 1.0 / (4.0 * PI), rel_tol=1e-12)
        assert math.isclose(COUPLING_MODERATE, 1.0 / (2.0 * PI), rel_tol=1e-12)
        assert math.isclose(COUPLING_FINE, 1.0 / (8.0 * PI), rel_tol=1e-12)

    def test_mid_coherence_is_two_over_pi(self) -> None:
        assert math.isclose(MID_COHERENCE_THRESHOLD, 2.0 / PI, rel_tol=1e-12)


class TestNoFrozenPhiGammaEDecimal:
    """No nodal-physics constant may equal a removed frozen phi/gamma/e decimal."""

    @pytest.mark.parametrize("name,value", sorted(PHYSICS_CONSTANTS.items()))
    def test_constant_is_not_a_frozen_decimal(
        self, name: str, value: float
    ) -> None:
        for frozen in FROZEN_PHI_GAMMA_E_DECIMALS:
            assert not math.isclose(value, frozen, abs_tol=1e-5), (
                f"{name}={value!r} regressed to the frozen phi/gamma/e decimal "
                f"{frozen} -- physics constants must stay pi-derived"
            )
