"""Regression checks for configured defaults, not emergent physical constants.

These formulas record the current policy in constants.canonical. Expressing a
coefficient in terms of pi does not derive it from the nodal equation. Operator
composition is tested through execution contracts in tests/operators.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize("precision", [15, 80])
def test_import_preserves_caller_precision_and_binary64_constants(
    precision, source_tree_environment
):
    """Configuration imports must not alter an ongoing numerical experiment."""
    program = f"""
import sys
from pathlib import Path
import mpmath as mp
mp.mp.dps = {precision}
import tnfr
assert Path(tnfr.__file__).resolve() == Path(sys.argv[1]), tnfr.__file__
from tnfr.constants.canonical import PI, LN_2
assert mp.mp.dps == {precision}, mp.mp.dps
assert PI.hex() == '0x1.921fb54442d18p+1'
assert LN_2.hex() == '0x1.62e42fefa39efp-1'
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(Path(__file__).resolve().parents[2] / "src" / "tnfr" / "__init__.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=source_tree_environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class TestConfiguredCoherenceBand:
    """The configured band uses complementary pi-scaled defaults."""

    def test_default_band_edges(self) -> None:
        assert math.isclose(FRAGMENTATION_THRESHOLD, 1.0 / (PI + 1.0), rel_tol=1e-12)
        assert math.isclose(HIGH_COHERENCE_THRESHOLD, PI / (PI + 1.0), rel_tol=1e-12)


class TestConfiguredChannelWeights:
    """Default channel coefficients form a normalized policy."""

    def test_default_channel_weights(self) -> None:
        assert math.isclose(CHANNEL_WEIGHT_PRIMARY, PI / (PI + 1.0), rel_tol=1e-12)
        assert math.isclose(
            CHANNEL_WEIGHT_SECONDARY, PI / (PI + 1.0) ** 2, rel_tol=1e-12
        )
        assert math.isclose(
            CHANNEL_WEIGHT_TERTIARY, 1.0 / (PI + 1.0) ** 2, rel_tol=1e-12
        )

    def test_channel_weights_sum_to_one_within_rounding(self) -> None:
        total = (
            CHANNEL_WEIGHT_PRIMARY + CHANNEL_WEIGHT_SECONDARY + CHANNEL_WEIGHT_TERTIARY
        )
        assert math.isclose(total, 1.0, abs_tol=1e-12)


class TestConfiguredOperatorGains:
    """Configured scalar gains; full operators also perform other writes."""

    def test_default_pressure_gains(self) -> None:
        assert math.isclose(COHERENCE_RETENTION, PI / (PI + 1.0), rel_tol=1e-12)
        assert math.isclose(DISSONANCE_AMPLIFICATION, (PI + 1.0) / PI, rel_tol=1e-12)

    def test_scalar_gains_are_reciprocal_within_rounding(self) -> None:
        # This scalar product does not certify the full IL/OZ operator word.
        product = COHERENCE_RETENTION * DISSONANCE_AMPLIFICATION
        assert math.isclose(product, 1.0, abs_tol=1e-12)

    def test_coupling_ladder_is_pi_fractions(self) -> None:
        assert math.isclose(COUPLING_GENTLE, 1.0 / (4.0 * PI), rel_tol=1e-12)
        assert math.isclose(COUPLING_MODERATE, 1.0 / (2.0 * PI), rel_tol=1e-12)
        assert math.isclose(COUPLING_FINE, 1.0 / (8.0 * PI), rel_tol=1e-12)

    def test_mid_coherence_is_two_over_pi(self) -> None:
        assert math.isclose(MID_COHERENCE_THRESHOLD, 2.0 / PI, rel_tol=1e-12)
