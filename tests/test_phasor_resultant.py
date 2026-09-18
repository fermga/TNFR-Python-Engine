"""Independent exact represented-component sums and bounded angle readouts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import permutations
import math
import sys

import pytest

from tnfr.mathematics import phasor_resultant as owner


TINY = float.fromhex("0x0.0000000000001p-1022")
MAXIMUM = sys.float_info.max


def _assert_exact_result(result, raw):
    """Oracle does not call the shared weighted-sum implementation."""
    pairs = tuple(tuple(float(value) for value in pair) for pair in raw)
    real = sum((F.from_float(pair[0]) for pair in pairs), F(0))
    imag = sum((F.from_float(pair[1]) for pair in pairs), F(0))
    assert result.count == len(pairs)
    assert result.components == pairs
    assert (result.real_sum, result.imag_sum) == (real, imag)
    assert result.joint_zero is (real == 0 and imag == 0)
    if result.joint_zero:
        assert result.scale is result.scaled_exact is result.scaled_binary64 is None
        assert result.scaled_rounding_defect is result.angle is None
        return
    scale = max(abs(real), abs(imag))
    normalized = (real/scale, imag/scale)
    rounded = tuple(map(float, normalized))
    residual = tuple(F.from_float(value)-exact for value, exact in zip(rounded, normalized, strict=True))
    assert result.scale == scale and result.scale > 0
    assert result.scaled_exact == normalized
    assert result.scaled_binary64 == rounded
    assert result.scaled_rounding_defect == residual
    assert max(map(abs, normalized)) == 1
    assert max(map(abs, rounded)) == 1.0
    assert all(math.isfinite(value) for value in rounded)
    assert math.isfinite(result.angle)
    # This checks only the displayed atan2 of the reported represented pair.
    # No exact transcendental direction or angular error bound is inferred.
    assert result.angle == math.atan2(rounded[1], rounded[0])


@pytest.mark.parametrize("pairs", (
    ((1.0, 0.0),), ((-1.0, 0.0),), ((0.0, 1.0),), ((0.0, -1.0),),
    ((1.0, 1.0), (-.25, .5)), ((0.0, -0.0), (-0.0, 0.0)),
    ((1.0, -2.0), (-1.0, 2.0)),
    ((1e16, .5), (1.0, -.5), (-1e16, 0.0)),
    ((TINY, TINY), (TINY, -TINY)),
    ((sys.float_info.min, TINY), (-sys.float_info.min, TINY)),
    ((MAXIMUM, MAXIMUM), (MAXIMUM, MAXIMUM)),
    ((MAXIMUM, -MAXIMUM), (-MAXIMUM, MAXIMUM), (TINY, -TINY)),
    ((MAXIMUM, TINY),), ((MAXIMUM, -TINY),), ((-MAXIMUM, TINY),), ((-MAXIMUM, -TINY),),
    ((TINY, MAXIMUM),), ((-TINY, MAXIMUM),),
    ((F(1, 10), F(-1, 3)), (F(2, 3), 2)),
))
def test_exact_sums_common_scaling_and_rounding_defects(pairs):
    _assert_exact_result(owner.reduce_phasor_components(pairs), pairs)


def test_every_permutation_and_reversal_preserves_the_resultant():
    pairs = ((1e16, .5), (1.0, -.5), (-1e16, 0.0), (TINY, TINY))
    baseline = owner.reduce_phasor_components(pairs)
    assert baseline.real_sum == 1+F.from_float(TINY)
    assert baseline.imag_sum == F.from_float(TINY)
    invariant = (baseline.real_sum, baseline.imag_sum, baseline.joint_zero, baseline.scale,
                 baseline.scaled_exact, baseline.scaled_binary64, baseline.scaled_rounding_defect, baseline.angle)
    for order in permutations(pairs):
        item = owner.reduce_phasor_components(order)
        _assert_exact_result(item, order)
        assert (item.real_sum, item.imag_sum, item.joint_zero, item.scale,
                item.scaled_exact, item.scaled_binary64, item.scaled_rounding_defect, item.angle) == invariant
    reversed_result = owner.reduce_phasor_components(reversed(pairs))
    assert reversed_result.real_sum == baseline.real_sum and reversed_result.imag_sum == baseline.imag_sum


def test_minor_normalized_component_may_underflow_without_losing_exact_direction_data():
    item = owner.reduce_phasor_components(((MAXIMUM, -TINY),))
    exact_minor = -F.from_float(TINY)/F.from_float(MAXIMUM)
    assert not item.joint_zero and item.scaled_exact == (F(1), exact_minor)
    assert exact_minor < 0 and item.scaled_binary64[1] == 0.0
    assert math.copysign(1.0, item.scaled_binary64[1]) == -1.0
    assert item.scaled_rounding_defect == (F(0), -exact_minor)
    assert item.angle == math.atan2(item.scaled_binary64[1], 1.0)


def test_unrepresentable_sum_is_reduced_without_float_overflow():
    item = owner.reduce_phasor_components(((MAXIMUM, MAXIMUM),)*3)
    exact = 3*F.from_float(MAXIMUM)
    assert item.real_sum == item.imag_sum == item.scale == exact
    with pytest.raises(OverflowError):
        float(exact)
    assert item.scaled_exact == (F(1), F(1)) and item.scaled_binary64 == (1.0, 1.0)
    assert item.angle == math.pi/4


def test_zero_has_no_direction_and_empty_is_not_zero(monkeypatch):
    def forbidden(*args):
        raise AssertionError("atan2 must not be called for an exact zero resultant")

    monkeypatch.setattr(owner.math, "atan2", forbidden)
    for pairs in (((1.0, -2.0), (-1.0, 2.0)), ((-0.0, 0.0),), ((TINY, -TINY), (-TINY, TINY))):
        item = owner.reduce_phasor_components(pairs)
        assert item.joint_zero and item.real_sum == item.imag_sum == 0
        assert item.angle is item.scaled_exact is item.scaled_binary64 is item.scale is None
    with pytest.raises(ValueError, match="nonempty|empty|at least"):
        owner.reduce_phasor_components(())


def test_finite_generators_materialize_once_and_result_is_immutable():
    raw = [[F(1, 10), -0.0], [-.25, 2.0], [.5, -.5]]
    saved = deepcopy(raw)
    consumed = []

    def source():
        for index, pair in enumerate(raw):
            consumed.append(index)
            yield (value for value in pair)

    result = owner.reduce_phasor_components(source())
    _assert_exact_result(result, raw)
    assert consumed == [0, 1, 2] and raw == saved
    assert isinstance(result.components, tuple) and all(isinstance(pair, tuple) for pair in result.components)
    assert math.copysign(1.0, result.components[0][1]) == 1.0
    with pytest.raises(FrozenInstanceError):
        result.count = 99
    with pytest.raises(TypeError):
        result.components[0][0] = 99
    raw[0][0] = 99
    assert result.components[0][0] == .1
    assert result.real_sum != F(1, 10)+F(1, 4)  # The contract sums represented components.


@pytest.mark.parametrize("value", (
    True, False, "1", b"1", complex(1, 0), None, float("nan"), float("inf"), -float("inf"),
    F(1, 1 << 1075), -F(1, 1 << 1075), 1 << 2048,
))
def test_invalid_or_nonrepresentable_components_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        owner.reduce_phasor_components(((value, 0.0),))
    with pytest.raises((TypeError, ValueError)):
        owner.reduce_phasor_components(((0.0, value),))


@pytest.mark.parametrize("components", (
    None, True, 1, "12", b"12", {"a": (1, 2)},
    ((1,),), ((1, 2, 3),), ((),), (1, 2), ({1, 2},), ({1: 2, 3: 4},),
))
def test_invalid_outer_or_pair_shape_is_rejected(components):
    with pytest.raises((TypeError, ValueError)):
        owner.reduce_phasor_components(components)


def test_unbounded_inner_pair_is_rejected_after_three_reads():
    seen = []

    def inner():
        for index in range(4):
            seen.append(index)
            if index == 3:
                raise AssertionError("an invalid pair must be rejected before a fourth read")
            yield 1.0

    with pytest.raises(ValueError, match="two|pair"):
        owner.reduce_phasor_components((inner(),))
    assert seen == [0, 1, 2]


def test_nonzero_rounding_defect_is_nearest_binary64_not_an_exact_angle_claim():
    item = owner.reduce_phasor_components(((3.0, 1.0),))
    exact, rounded = item.scaled_exact[1], item.scaled_binary64[1]
    assert exact == F(1, 3) and item.scaled_rounding_defect[1] != 0
    error = abs(F.from_float(rounded)-exact)
    assert error <= abs(F.from_float(math.nextafter(rounded, -math.inf))-exact)
    assert error <= abs(F.from_float(math.nextafter(rounded, math.inf))-exact)


def test_unsupported_binary64_environment_refuses_before_consuming_input(monkeypatch):
    consumed = []

    def source():
        consumed.append(True)
        yield (1.0, 0.0)

    monkeypatch.setattr(owner, "uses_ieee_binary64_rounding", lambda: False)
    with pytest.raises(ValueError, match="binary64|rounding"):
        owner.reduce_phasor_components(source())
    assert not consumed


def test_reduction_evaluates_no_sine_cosine_or_graph_dynamics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("represented-component reduction must not evaluate trigonometric components")

    monkeypatch.setattr(math, "sin", forbidden)
    monkeypatch.setattr(math, "cos", forbidden)
    original = math.atan2
    calls = []

    def angle(y, x):
        calls.append((y, x))
        return original(y, x)

    monkeypatch.setattr(owner.math, "atan2", angle)
    item = owner.reduce_phasor_components(((3.0, 4.0), (1.0, -2.0)))
    assert calls == [(0.5, 1.0)]
    assert item.real_sum == 4 and item.imag_sum == 2
