"""Domain and reproducibility tests for shared numerical utilities."""

from __future__ import annotations

import math
from fractions import Fraction

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


@pytest.mark.parametrize("numpy_available", [True, False])
@pytest.mark.parametrize("container", ["scalar", "list", "array", "object_array"])
@pytest.mark.parametrize(
    "bad",
    [True, np.bool_(False), "0.1", b"1", 0.1 + 0j, math.inf, math.nan],
)
def test_phase_readers_reject_raw_invalid_elements(
    monkeypatch, numpy_available, container, bad
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    utilities = TNFRNumericalUtilities(seed=0)
    if container == "scalar":
        phase, zero = bad, 0.0
    elif container == "list":
        phase, zero = [0.0, bad], [0.0, 0.0]
    else:
        dtype = object if container == "object_array" else None
        phase, zero = np.array([bad], dtype=dtype), np.array([0.0])

    with pytest.raises(TNFRValueError, match="finite real"):
        utilities.normalize_phase(phase)
    with pytest.raises(TNFRValueError, match="finite real"):
        utilities.compute_phase_difference(phase, zero)
    with pytest.raises(TNFRValueError, match="finite real"):
        utilities.compute_phase_difference(zero, phase)


@pytest.mark.parametrize("numpy_available", [True, False])
@pytest.mark.parametrize("bad", [Fraction(1, 10**400), 10**400])
@pytest.mark.parametrize("iterable", [True, False])
def test_phase_readers_reject_binary64_materialization_loss(
    monkeypatch, numpy_available, bad, iterable
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    utilities = TNFRNumericalUtilities(seed=0)
    phase, zero = ([bad], [0.0]) if iterable else (bad, 0.0)
    with pytest.raises(TNFRValueError, match="representable as binary64"):
        utilities.normalize_phase(phase)
    with pytest.raises(TNFRValueError, match="representable as binary64"):
        utilities.compute_phase_difference(phase, zero)


@pytest.mark.parametrize("numpy_available", [True, False])
@pytest.mark.parametrize("text", ["", b"", bytearray(b"1")])
def test_phase_readers_do_not_interpret_text_as_a_phase_sequence(
    monkeypatch, numpy_available, text
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    utilities = TNFRNumericalUtilities(seed=0)
    with pytest.raises(TNFRValueError, match="finite real"):
        utilities.normalize_phase(text)
    with pytest.raises(TNFRValueError, match="finite real"):
        utilities.compute_phase_difference(text, text)


@pytest.mark.parametrize("numpy_available", [True, False])
@pytest.mark.parametrize("container", [float, list, np.array])
def test_normalized_phase_range_includes_zero_and_excludes_rounded_upper_endpoint(
    monkeypatch, numpy_available, container
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    utilities = TNFRNumericalUtilities(seed=0)
    tiny = math.ulp(0.0)
    for phase in (-tiny, tiny, -0.0, 0.0, -math.tau, math.tau):
        source = phase if container is float else container([phase])
        result = np.asarray(utilities.normalize_phase(source))
        assert np.all(result >= 0.0)
        assert np.all(result < math.tau)
        assert np.all(result == (tiny if phase == tiny else 0.0))


@pytest.mark.parametrize("numpy_available", [True, False])
def test_phase_difference_preserves_signed_zeros_and_signed_cut(
    monkeypatch, numpy_available
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    utilities = TNFRNumericalUtilities(seed=0)
    phases = [-0.0, 0.0, -math.pi, math.pi, -math.ulp(0.0), math.ulp(0.0)]
    expected = [math.atan2(math.sin(value), math.cos(value)) for value in phases]
    actual = utilities.compute_phase_difference(phases, [0.0] * len(phases))
    assert list(actual) == expected
    assert [math.copysign(1.0, value) for value in actual] == [
        math.copysign(1.0, value) for value in expected
    ]
    for phase, result in zip(phases, expected):
        scalar = utilities.compute_phase_difference(phase, 0.0)
        assert scalar == result
        assert math.copysign(1.0, scalar) == math.copysign(1.0, result)


def test_phase_array_broadcasting_admits_exact_reals_and_preserves_input():
    utilities = TNFRNumericalUtilities(seed=0)
    phases = np.array([[Fraction(1, 4)], [np.int64(1)]], dtype=object)
    before = phases.copy()
    offsets = np.array([0.0, 0.5])
    differences = utilities.compute_phase_difference(phases, offsets)
    expected = np.array([[0.25, -0.25], [1.0, 0.5]])
    np.testing.assert_allclose(differences, expected, rtol=1e-15, atol=0.0)
    assert np.array_equal(phases, before)
    assert utilities.normalize_phase(phases).shape == phases.shape


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
@pytest.mark.parametrize(
    "bad",
    [True, "0.1", 0.1 + 0j, math.inf, math.nan, Fraction(1, 10**400), 10**400],
)
def test_circular_mean_raw_admission_agrees_across_backends(
    monkeypatch, numpy_available, bad
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    with pytest.raises(TNFRValueError, match="finite real"):
        TNFRNumericalUtilities(seed=0).compute_circular_mean([0.0, bad])


@pytest.mark.parametrize("numpy_available", [True, False])
@pytest.mark.parametrize("sign", [-1, 1])
def test_circular_mean_retains_representable_subnormal_phase(
    monkeypatch, numpy_available, sign
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    phase = sign * math.ulp(0.0)
    mean = TNFRNumericalUtilities(seed=0).compute_circular_mean(
        [Fraction.from_float(phase)]
    )
    assert mean == phase


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


@pytest.mark.parametrize("bad", [True, "0.25", 0.25 + 0j])
@pytest.mark.parametrize("container", ["scalar", "list", "array", "object_array"])
@pytest.mark.parametrize("operand", ["minuend", "subtrahend"])
@pytest.mark.parametrize("masked", [True, False])
def test_vector_phase_difference_raw_admission_is_atomic(
    bad, container, operand, masked
):
    if container == "scalar":
        phase = bad
    elif container == "list":
        phase = [0.0, bad]
    elif container == "array":
        phase = np.array([bad, bad])
    else:
        phase = np.array([0.0, bad], dtype=object)
    other = np.array([0.0, 0.5])
    inputs = (phase, other) if operand == "minuend" else (other, phase)
    output = np.array([7.0, 8.0])
    before = output.copy()
    mask = [False, True] if masked else None

    with pytest.raises(TNFRValueError, match="finite real"):
        angle_diff_array(*inputs, np=np, out=output, where=mask)

    assert np.array_equal(output, before)


@pytest.mark.parametrize("bad", [True, "0.25", 0.25 + 0j, Fraction(1, 10**400)])
def test_vector_phase_difference_does_not_admit_unselected_raw_elements(bad):
    phases = [1e-16, bad]
    output = np.array([7.0, 8.0])

    result = angle_diff_array(phases, 0.0, np=np, out=output, where=[True, False])

    assert result is output
    assert result.tolist() == [1e-16, 8.0]
    allocated = angle_diff_array(phases, 0.0, np=np, where=[True, False])
    assert allocated.tolist() == [1e-16, 0.0]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.complex64])
@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize("numpy_available", [True, False])
def test_narrow_phase_output_rejects_nonzero_loss_before_any_write(
    monkeypatch, dtype, masked, numpy_available
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    output = np.array([7.0, 8.0, 9.0], dtype=dtype)
    before = output.copy()
    mask = [True, True, False] if masked else None

    with pytest.raises(TNFRValueError, match="underflow"):
        angle_diff_array([0.25, 1e-50, 0.5], 0.0, np=np, out=output, where=mask)

    np.testing.assert_array_equal(output, before)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("numpy_available", [True, False])
def test_narrow_phase_output_preserves_subnormals_and_skips_unselected_loss(
    monkeypatch, dtype, numpy_available
):
    monkeypatch.setattr(unified_numerical, "NUMPY_AVAILABLE", numpy_available)
    smallest = float(np.finfo(dtype).smallest_subnormal)
    output = np.array([7.0, 8.0, 9.0], dtype=dtype)

    result = angle_diff_array(
        [smallest, -0.0, 1e-50],
        0.0,
        np=np,
        out=output,
        where=[True, True, False],
    )

    assert result is output
    assert float(result[0]) == smallest
    assert math.copysign(1.0, float(result[1])) == -1.0
    assert float(result[2]) == 9.0
