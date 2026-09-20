"""Domain and reproducibility tests for shared numerical utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tnfr.mathematics.unified_numerical as unified_numerical
from tnfr.errors import TNFRValueError
from tnfr.mathematics.unified_numerical import CONSTANTS, TNFRNumericalUtilities
from tnfr.utils.numeric import angle_diff, angle_diff_array


def test_zero_seed_is_preserved_and_reproducible():
    first = TNFRNumericalUtilities(seed=0)
    second = TNFRNumericalUtilities(seed=0)

    assert first.seed == 0
    assert np.array_equal(
        first.generate_random_array(5),
        second.generate_random_array(5),
    )


@pytest.mark.parametrize(
    "seed",
    [True, -1, 1.5, CONSTANTS.SEED_RANGE_MAX + 1],
)
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(TNFRValueError, match="seed must be an integer"):
        TNFRNumericalUtilities(seed=seed)


def test_operation_seed_does_not_advance_instance_rng():
    reference = TNFRNumericalUtilities(seed=7)
    subject = TNFRNumericalUtilities(seed=7)

    subject.generate_random_array(4, seed=11)
    assert np.array_equal(
        subject.generate_random_array(4),
        reference.generate_random_array(4),
    )


@pytest.mark.parametrize("size", [True, -1, (2, -1), (2, 1.5), [2, 3]])
def test_random_size_domain_is_shared_across_backends(size):
    with pytest.raises(TNFRValueError, match="size must be"):
        TNFRNumericalUtilities().generate_random_array(size)


def test_safe_divide_preserves_fractional_fallback_for_integer_input():
    utilities = TNFRNumericalUtilities()
    result = utilities.safe_divide(
        np.array([1, 2], dtype=int),
        np.array([0, 2], dtype=int),
        fallback=0.25,
    )

    assert result.dtype.kind == "f"
    assert result.tolist() == pytest.approx([0.25, 1.0])


def test_circular_mean_respects_wrap_and_rejects_undefined_samples():
    utilities = TNFRNumericalUtilities()
    mean = utilities.compute_circular_mean([math.pi - 0.01, -math.pi + 0.01])
    assert abs(abs(mean) - math.pi) < 1e-12

    with pytest.raises(TNFRValueError, match="at least one"):
        utilities.compute_circular_mean([])
    with pytest.raises(TNFRValueError, match="vanishing resultant"):
        utilities.compute_circular_mean([0.0, math.pi])


def test_phase_and_interval_domains_are_explicit():
    utilities = TNFRNumericalUtilities()
    assert utilities.normalize_phase(2.0 * math.pi) == pytest.approx(0.0)
    with pytest.raises(TNFRValueError, match="finite"):
        utilities.normalize_phase(math.inf)
    with pytest.raises(TNFRValueError, match="min_val"):
        utilities.clamp_value(0.0, 2.0, 1.0)


def test_boolean_dimension_is_not_an_integer_dimension():
    with pytest.raises(TNFRValueError, match="dims must be"):
        TNFRNumericalUtilities().kahan_sum_nd([(1.0,)], dims=True)


def test_finiteness_readout_is_a_python_boolean():
    result = TNFRNumericalUtilities().is_finite_array(np.array([1.0, 2.0]))
    assert type(result) is bool
    assert result is True


def test_safe_divide_fallback_broadcasts_and_rejects_length_mismatch(monkeypatch):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", False)
    utilities = TNFRNumericalUtilities(seed=3)

    assert utilities.safe_divide([2.0, 4.0], 2.0) == pytest.approx([1.0, 2.0])
    assert utilities.safe_divide(6.0, [2.0, 0.0], fallback=-1.0) == pytest.approx(
        [3.0, -1.0]
    )
    with pytest.raises(TNFRValueError, match="equal length"):
        utilities.safe_divide([1.0, 2.0], [1.0])


def test_statistics_identify_uninstrumented_compatibility_counters():
    utilities = TNFRNumericalUtilities(seed=5)
    utilities.generate_random_array(2)

    statistics = utilities.get_statistics()
    assert statistics["statistics_collected"] is False
    assert statistics["operation_count"] == 0
    assert statistics["total_time"] == 0.0


@pytest.mark.parametrize("numpy_available", [True, False])
@pytest.mark.parametrize("bad", [True, "0.1", 0.1 + 0j, math.inf, math.nan])
def test_circular_mean_raw_admission_agrees_across_backends(
    monkeypatch, numpy_available, bad
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    with pytest.raises(TNFRValueError, match="finite real"):
        TNFRNumericalUtilities(seed=0).compute_circular_mean([0.0, bad])


@pytest.mark.parametrize("numpy_available", [True, False])
@pytest.mark.parametrize("iterable", [True, False])
def test_unrepresentable_phase_difference_is_rejected_across_backends(
    monkeypatch, numpy_available, iterable
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    first, second = ([1e308], [-1e308]) if iterable else (1e308, -1e308)
    with pytest.raises(TNFRValueError, match="differences must be finite"):
        TNFRNumericalUtilities(seed=0).compute_phase_difference(first, second)


def test_vector_phase_difference_preserves_small_separations_and_signed_cut():
    phases = np.array([1e-16, -1e-16, math.pi, -math.pi, 1e308])
    differences = angle_diff_array(phases, 0.0, np=np)
    expected = [math.atan2(math.sin(p), math.cos(p)) for p in phases]
    assert differences == pytest.approx(expected, rel=2e-15, abs=0.0)
    assert differences[:4].tolist() == phases[:4].tolist()
    assert differences.tolist() == [angle_diff(p, 0.0) for p in phases]


def test_vector_phase_difference_mask_preserves_output_and_skips_invalid_pairs():
    output = np.array([7.0, 8.0])
    result = angle_diff_array(
        [1e-16, 1e308], [0.0, -1e308], np=np, out=output, where=[True, False]
    )
    assert result is output
    assert result.tolist() == [1e-16, 8.0]
    before = output.copy()
    with pytest.raises(TNFRValueError, match="where mask"):
        angle_diff_array([1.0, 2.0], 0.0, np=np, out=output, where=[True])
    assert np.array_equal(output, before)
    with pytest.raises(TNFRValueError, match="differences must be finite"):
        angle_diff_array([1.0, 1e308], [0.0, -1e308], np=np, out=output)
    assert np.array_equal(output, before)
