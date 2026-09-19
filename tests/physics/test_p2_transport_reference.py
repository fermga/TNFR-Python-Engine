"""Independent continuous-P2 enclosure checks, not physical admission."""

from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import product

import pytest

from tnfr._exact_time import fraction_lower_float, fraction_upper_float
from tnfr.physics.p2_transport_reference import (
    bound_fixed_reference_capacity,
    bound_fixed_reference_transport,
    bound_p2_capacity,
    bound_p2_transport,
)


def dec(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def inside(value, interval):
    assert dec(interval[0]) <= value <= dec(interval[1])


@pytest.mark.parametrize("sign", [1, -1])
def test_continuous_capacity_log_ratio_includes_independent_decimal(sign):
    initial = ((F(sign), F(sign)), (F(-sign), F(-sign)))
    final = ((F(sign, 2), F(sign, 2)), (F(-sign, 2), F(-sign, 2)))
    result = bound_p2_capacity(initial, final, elapsed_time=(F(1), F(1)))
    with localcontext() as context:
        context.prec = 90
        inside(Decimal(2).ln() / 2, result.capacity)
        # Euler finite-increment fit is 1/4, not the continuous log(2)/2.
        assert dec(result.capacity[0]) > Decimal(1) / 4
    assert result.mean_overlap == (0, 0)


def test_calibration_uncertainty_contains_every_endpoint_capacity_corner():
    initial = ((F(9, 10), F(11, 10)), (F(-11, 10), F(-9, 10)))
    final = ((F(4, 10), F(6, 10)), (F(-6, 10), F(-4, 10)))
    elapsed = F(9, 10), F(11, 10)
    result = bound_p2_capacity(initial, final, elapsed_time=elapsed)
    with localcontext() as context:
        context.prec = 90
        for x1, x2, y1, y2, time in product(*initial, *final, elapsed):
            rate = -((dec(y1) - dec(y2)) / (dec(x1) - dec(x2))).ln() / (2 * dec(time))
            inside(rate, result.capacity)


def test_transport_semigroup_contains_independent_all_corner_solutions():
    initial = ((F(-2), F(1)), (F(3), F(5)))
    capacity = F(1, 10), F(3, 10)
    times = ((F(0), F(0)), (F(1, 5), F(1, 3)), (F(1), F(2)))
    tube = bound_p2_transport(initial, capacity=capacity, elapsed_times=times)
    assert tube.samples[0].epi == initial
    assert tube.samples[0].decay == (1, 1)
    with localcontext() as context:
        context.prec = 90
        for sample in tube.samples:
            for a, b, rate, time in product(*initial, capacity, sample.elapsed_time):
                e = (-2 * dec(rate) * dec(time)).exp()
                mean = (dec(a) + dec(b)) / 2
                contrast = (dec(a) - dec(b)) * e
                inside(mean + contrast / 2, sample.epi[0])
                inside(mean - contrast / 2, sample.epi[1])
                inside(contrast, sample.contrast)
                inside(mean, sample.mean)


def test_zero_time_preserves_dependency_and_does_not_inflate_coordinate_boxes():
    initial = ((F(0), F(1)), (F(2), F(3)))
    sample = bound_p2_transport(
        initial, capacity=(1, 2), elapsed_times=[(0, 0)]
    ).samples[0]
    assert sample.epi == initial
    assert sample.mean == (1, 2)
    # Independently recombining mean and contrast would incorrectly give [-1/2,3/2].
    assert sample.contrast == (-3, -1)


def test_common_uniform_initial_state_remains_exact_for_every_capacity_and_time():
    initial = ((F(7, 3), F(7, 3)),) * 2
    tube = bound_p2_transport(initial, capacity=(1, 4), elapsed_times=[(0, 0), (1, 2)])
    assert all(
        sample.epi == initial and sample.contrast == (0, 0) for sample in tube.samples
    )


def test_wider_input_boxes_never_shrink_transport_enclosures():
    narrow = bound_p2_transport(
        ((1, 1), (-1, -1)), capacity=(F(1, 4), F(1, 4)), elapsed_times=[(1, 1)]
    )
    wide = bound_p2_transport(
        ((F(9, 10), F(11, 10)), (F(-11, 10), F(-9, 10))),
        capacity=(F(1, 5), F(3, 10)),
        elapsed_times=[(F(9, 10), F(11, 10))],
    )
    for smaller, larger in zip(narrow.samples[0].epi, wide.samples[0].epi):
        assert larger[0] <= smaller[0] <= smaller[1] <= larger[1]


@pytest.mark.parametrize(
    "initial,final,time,reason",
    [
        (((0, 0), (0, 0)), ((0, 0), (0, 0)), (1, 1), "contrast_sign"),
        (((1, 1), (-1, -1)), ((-1, -1), (1, 1)), (1, 1), "contrast_sign"),
        (((1, 1), (-1, -1)), ((1, 1), (-1, -1)), (1, 1), "strict_decay"),
        (((1, 1), (-1, -1)), ((2, 2), (-2, -2)), (1, 1), "strict_decay"),
        (((1, 1), (-1, -1)), ((1, 1), (0, 0)), (1, 1), "mean_intervals"),
        (
            ((1, 1), (-1, -1)),
            ((F(1, 2), F(1, 2)), (-F(1, 2), -F(1, 2))),
            (0, 1),
            "strictly positive",
        ),
    ],
)
def test_calibration_domain_failures_are_explicit(initial, final, time, reason):
    with pytest.raises(ValueError, match=reason):
        bound_p2_capacity(initial, final, elapsed_time=time)


@pytest.mark.parametrize(
    "capacity,times,reason",
    [
        ((0, 1), [(1, 1)], "strictly positive"),
        ((-1, -1), [(1, 1)], "strictly positive"),
        ((2, 1), [(1, 1)], "inverted"),
        ((1, 1), [(-1, 0)], "nonnegative"),
        ((1, 1), [(1, 0)], "inverted"),
        ((1, 1), [(1, 2), (0, 3)], "ordered"),
        ((1, 1), [(1, 3), (2, 2)], "ordered"),
        ((1, 1), [], "between 1 and 10000"),
        ((1, 1), [(2049, 2049)], "exponent <= 4096"),
    ],
)
def test_transport_domain_and_resource_failures(capacity, times, reason):
    with pytest.raises(ValueError, match=reason):
        bound_p2_transport(((1, 1), (-1, -1)), capacity=capacity, elapsed_times=times)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1", None])
def test_nonfinite_or_nonreal_endpoints_are_not_silently_coerced(value):
    with pytest.raises((TypeError, ValueError)):
        bound_p2_transport(
            ((value, value), (0, 0)), capacity=(1, 1), elapsed_times=[(0, 0)]
        )


def test_elapsed_materialization_stops_at_declared_sample_cap():
    visits = []

    def endless():
        while True:
            visits.append(None)
            yield (0, 0)

    with pytest.raises(ValueError, match="10000"):
        bound_p2_transport(((0, 0), (0, 0)), capacity=(1, 1), elapsed_times=endless())
    assert len(visits) == 10001


def test_inputs_are_detached_and_represented_floats_are_not_decimal_claims():
    initial = [[0.1, 0.1], [0.2, 0.2]]
    tube = bound_p2_transport(initial, capacity=(1, 1), elapsed_times=[(0, 0)])
    initial[0][0] = 999
    assert tube.samples[0].epi[0] == (F.from_float(0.1), F.from_float(0.1))
    assert tube.samples[0].epi[0][0] != F(1, 10)


def test_outward_binary64_capacity_bridge_preserves_realistic_log_enclosure():
    calibration = bound_p2_capacity(
        ((0.9, 1.1), (-1.1, -0.9)), ((0.4, 0.6), (-0.6, -0.4)), elapsed_time=(0.9, 1.1)
    )
    exact = calibration.capacity
    bounded = (
        F.from_float(fraction_lower_float(exact[0])),
        F.from_float(fraction_upper_float(exact[1])),
    )
    assert 0 < bounded[0] <= exact[0] <= exact[1] <= bounded[1]
    sample = bound_p2_transport(
        ((1, 1), (-1, -1)), capacity=bounded, elapsed_times=[(1, 1)]
    ).samples[0]
    with localcontext() as context:
        context.prec = 90
        for capacity in exact:
            inside((-2 * dec(capacity)).exp(), sample.decay)


@pytest.mark.parametrize("sign", [1, -1])
def test_fixed_reference_ratio_has_factor_one_not_common_capacity_factor_two(sign):
    fixed = bound_fixed_reference_capacity(
        (2 * sign, 2 * sign), (sign, sign), reference=(0, 0), elapsed_time=(1, 1)
    )
    common = bound_p2_capacity(
        ((sign, sign), (-sign, -sign)),
        ((F(sign, 2), F(sign, 2)), (F(-sign, 2), F(-sign, 2))),
        elapsed_time=(1, 1),
    )
    assert fixed.capacity == tuple(2 * value for value in common.capacity)
    assert fixed.reference == (0, 0)
    with localcontext() as context:
        context.prec = 90
        inside(Decimal(2).ln(), fixed.capacity)


def test_fixed_reference_calibration_does_not_require_conserved_mean():
    result = bound_fixed_reference_capacity(
        (5, 5), (3, 3), reference=(1, 1), elapsed_time=(1, 1)
    )
    assert result.initial_contrast == (4, 4)
    assert result.final_contrast == (2, 2)
    # The physical-model coordinate pair mean changes from3 to2.
    with pytest.raises(ValueError, match="mean_intervals_disjoint"):
        bound_p2_capacity(((5, 5), (1, 1)), ((3, 3), (1, 1)), elapsed_time=(1, 1))


def test_shared_reference_correlation_is_retained_as_an_outer_scope_not_feasibility():
    initial, final, reference = (F(2), F(3)), (F(1), F(6, 5)), (F(0), F(1, 10))
    result = bound_fixed_reference_capacity(
        initial, final, reference=reference, elapsed_time=(1, 1)
    )
    with localcontext() as context:
        context.prec = 90
        actual_corner_rates = []
        for x0, x1, r in product(initial, final, reference):
            rate = -((dec(x1) - dec(r)) / (dec(x0) - dec(r))).ln()
            inside(rate, result.capacity)
            actual_corner_rates.append(rate)
        # Independent numerator/denominator bounds choose incompatible r values.
        # The resulting interval is intentionally wider than shared-r extrema.
        assert dec(result.capacity[0]) < min(actual_corner_rates)
        assert dec(result.capacity[1]) > max(actual_corner_rates)


@pytest.mark.parametrize(
    "initial,reference",
    [
        ((F(3), F(5)), (F(-1), F(1))),
        ((F(-5), F(-3)), (F(-1), F(1))),
        ((F(-2), F(2)), (F(-1), F(1))),
    ],
)
def test_fixed_reference_transport_contains_independent_corner_trajectories(
    initial, reference
):
    rate = F(1, 5), F(2, 5)
    times = ((F(0), F(0)), (F(1, 4), F(1, 3)), (F(1), F(2)))
    tube = bound_fixed_reference_transport(
        initial, reference=reference, capacity=rate, elapsed_times=times
    )
    assert tube.samples[0].epi == initial
    assert all(sample.reference == reference for sample in tube.samples)
    with localcontext() as context:
        context.prec = 90
        for sample in tube.samples:
            for x0, r, nu, time in product(
                initial, reference, rate, sample.elapsed_time
            ):
                e = (-dec(nu) * dec(time)).exp()
                value = e * dec(x0) + (1 - e) * dec(r)
                inside(value, sample.epi)
                inside((dec(x0) - dec(r)) * e, sample.contrast)


def test_fixed_reference_time_zero_does_not_inflate_by_reference_uncertainty():
    sample = bound_fixed_reference_transport(
        (3, 4), reference=(-100, 100), capacity=(1, 2), elapsed_times=[(0, 0)]
    ).samples[0]
    assert sample.epi == (3, 4)
    assert sample.reference == (-100, 100)
    assert sample.decay == (1, 1)


def test_fixed_reference_mean_changes_while_reference_stays_fixed():
    first, last = bound_fixed_reference_transport(
        (4, 4), reference=(0, 0), capacity=(1, 1), elapsed_times=[(0, 0), (1, 1)]
    ).samples
    assert last.epi[1] < first.epi[0]
    assert last.reference == first.reference == (0, 0)
    assert (last.epi[1] + last.reference[1]) / 2 < 2


@pytest.mark.parametrize(
    "initial,final,reference,time,reason",
    [
        ((0, 0), (0, 0), (0, 0), (1, 1), "contrast_sign"),
        ((-1, 1), (F(1, 2), F(1, 2)), (0, 0), (1, 1), "contrast_sign"),
        ((1, 1), (-1, -1), (0, 0), (1, 1), "contrast_sign"),
        ((1, 1), (1, 1), (0, 0), (1, 1), "strict_decay"),
        ((1, 1), (2, 2), (0, 0), (1, 1), "strict_decay"),
        ((2, 2), (1, 1), (1, 0), (1, 1), "inverted"),
        ((2, 2), (1, 1), (0, 0), (0, 1), "strictly positive"),
    ],
)
def test_fixed_reference_calibration_rejects_domain_failures(
    initial,
    final,
    reference,
    time,
    reason,
):
    with pytest.raises(ValueError, match=reason):
        bound_fixed_reference_capacity(
            initial, final, reference=reference, elapsed_time=time
        )


@pytest.mark.parametrize(
    "rate,times,reason",
    [
        ((0, 1), [(1, 1)], "strictly positive"),
        ((1, 1), [(-1, 0)], "nonnegative"),
        ((1, 1), [(1, 2), (0, 3)], "ordered"),
        ((1, 1), [], "between 1 and 10000"),
        ((1, 1), [(4097, 4097)], "exponent <= 4096"),
    ],
)
def test_fixed_reference_transport_reuses_domain_and_resource_guards(
    rate, times, reason
):
    with pytest.raises(ValueError, match=reason):
        bound_fixed_reference_transport(
            (1, 1), reference=(0, 0), capacity=rate, elapsed_times=times
        )


def test_fixed_reference_inputs_are_detached_and_nonfinite_reference_rejected():
    reference = [0, 1]
    tube = bound_fixed_reference_transport(
        (2, 3), reference=reference, capacity=(1, 1), elapsed_times=[(0, 0)]
    )
    reference[0] = 100
    assert tube.samples[0].reference == (0, 1)
    with pytest.raises(ValueError, match="finite"):
        bound_fixed_reference_transport(
            (2, 3), reference=(0, float("inf")), capacity=(1, 1), elapsed_times=[(0, 0)]
        )
