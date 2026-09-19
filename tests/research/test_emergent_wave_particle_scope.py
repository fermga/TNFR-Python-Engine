"""Regression controls for the sampled-ring benchmark's applicability boundary."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "emergent_wave_particle_correspondence.py"
)
_SPEC = importlib.util.spec_from_file_location("emergent_wave_particle_scope", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


def test_mode_index_alias_preserves_phasor_but_not_unrestricted_winding_label():
    original = _BENCHMARK._ring_mode_observation(12, 1)
    aliased = _BENCHMARK._ring_mode_observation(12, 13)

    np.testing.assert_array_equal(original["phase_phasor"], aliased["phase_phasor"])
    assert aliased["mode_index"] == 13
    assert aliased["sampled_representative"] == 1
    assert aliased["certificate"].winding == 1
    assert aliased["certificate"].winding != aliased["mode_index"]
    assert aliased["eigenvalue_measured"] == original["eigenvalue_measured"]
    assert aliased["eigenvector_residual"] < 1e-12


def test_nyquist_mode_has_valid_spectral_identity_but_undefined_winding():
    observation = _BENCHMARK._ring_mode_observation(12, 6)
    certificate = observation["certificate"]

    assert observation["eigenvalue_measured"] == pytest.approx(2.0)
    assert observation["eigenvector_residual"] < 1e-12
    assert observation["sampled_representative"] is None
    assert not certificate.is_defined
    assert certificate.winding is None
    assert "branch boundary" in certificate.reason


@pytest.mark.parametrize("mode,expected", [(5, 5), (7, -5), (-1, -1)])
def test_signed_sampled_representative_matches_production_winding(mode, expected):
    observation = _BENCHMARK._ring_mode_observation(12, mode)

    assert observation["sampled_representative"] == expected
    assert observation["certificate"].is_defined
    assert observation["certificate"].winding == expected
    assert observation["certificate"].raw_winding == pytest.approx(expected)
    assert observation["eigenvector_residual"] < 1e-12


def test_defined_winding_does_not_imply_u3_admissibility():
    compatible = _BENCHMARK._ring_mode_observation(12, 1)["certificate"]
    incompatible = _BENCHMARK._ring_mode_observation(12, 5)["certificate"]

    assert compatible.is_defined and compatible.u3_admissible
    assert incompatible.is_defined and not incompatible.u3_admissible
    assert incompatible.minimum_branch_margin == pytest.approx(math.pi / 6)
    assert incompatible.minimum_u3_margin == pytest.approx(-math.pi / 3)


def test_uniform_phase_phasor_is_not_the_geometric_complex_field():
    phase_phasor, geometric_field = _BENCHMARK._uniform_phase_field_control()

    np.testing.assert_array_equal(phase_phasor, np.ones(12, dtype=complex))
    np.testing.assert_array_equal(geometric_field, np.zeros(12, dtype=complex))
    assert np.all(np.abs(phase_phasor - geometric_field) == 1.0)


def test_declared_finite_real_ripple_fixture_uses_defined_zero_windings():
    observations = _BENCHMARK._real_ripple_observations()

    assert tuple(amplitude for amplitude, _ in observations) == (0.5, 2, 5, 10, 20)
    for _, certificate in observations:
        assert certificate.cycle_nodes == tuple(range(60))
        assert certificate.is_defined
        assert certificate.winding == 0
        assert certificate.quantization_residual < 1e-12
