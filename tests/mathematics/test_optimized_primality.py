"""Independent arithmetic controls across the optimized sieve boundary."""

import math
from fractions import Fraction

import numpy as np
import pytest

from tnfr.mathematics.optimized_primality import OptimizedTNFRPrimality
from tnfr.tools import tnfr_is_prime_cli as cli


@pytest.fixture(scope="module")
def optimizer():
    return OptimizedTNFRPrimality(backend="numpy", enable_gpu=False)


def _divisors(number):
    factors = set()
    for divisor in range(1, math.isqrt(number) + 1):
        if number % divisor == 0:
            factors.update((divisor, number // divisor))
    return factors


@pytest.mark.parametrize("number", [46337, 50021, 65536, 999983, 1_000_003])
def test_sieve_and_fallback_arithmetic_match_enumerated_divisors(optimizer, number):
    divisors = _divisors(number)
    actual_sum = optimizer._fast_divisor_sum(number)
    assert type(actual_sum) is int
    assert actual_sum == sum(divisors)
    assert optimizer._fast_divisor_count(number) == len(divisors)
    report = optimizer.is_prime_optimized(number)
    assert report.is_prime == (len(divisors) == 2)
    if len(divisors) == 2:
        assert report.delta_nfr == 0
    else:
        # 65536 = 2**16: Ω=16 and its 17 divisors are powers of two.
        expected = 15 + 0.8 * 15 + 0.6 * (sum(divisors) / number - (1 + 1 / number))
        assert report.delta_nfr == pytest.approx(expected)


def test_sieve_factor_storage_does_not_control_integer_arithmetic_precision(optimizer):
    assert optimizer.sieve_data["min_prime_factor"].dtype == np.int32
    with np.errstate(over="raise"):
        assert optimizer._fast_divisor_sum(50021) == sum(_divisors(50021))
        assert optimizer.compute_delta_nfr(50021) == 0


def test_cli_basic_and_optimized_pressure_agree_above_int32_square_boundary(
    optimizer, monkeypatch
):
    monkeypatch.setattr(cli, "_get_optimizer", lambda: optimizer)
    for number in (50021, 65536, 999983):
        basic = cli.tnfr_is_prime(number)
        optimized = cli.tnfr_is_prime(number, use_optimized=True)
        assert optimized[0] is basic[0]
        assert optimized[1] == pytest.approx(basic[1])


def test_returned_reports_cannot_mutate_cached_decisions_or_nested_metrics(monkeypatch):
    optimizer = OptimizedTNFRPrimality(sieve_limit=10)
    supplied_metrics = {"coherence": 0.25, "detail": {"samples": [1, 2]}}
    monkeypatch.setattr(
        optimizer, "_compute_structural_metrics", lambda _n: supplied_metrics
    )
    original = optimizer.is_prime_optimized(4, include_metrics=True)
    assert original.is_prime is False
    original.is_prime = True
    original.delta_nfr = 0
    original.structural_metrics["detail"]["samples"].append(3)
    repeated = optimizer.is_prime_optimized(4, include_metrics=True)
    assert repeated is not original
    assert repeated.is_prime is False
    assert repeated.delta_nfr == pytest.approx(2.1)
    assert repeated.structural_metrics == {
        "coherence": 0.25,
        "detail": {"samples": [1, 2]},
    }
    assert original.cache_hit is False
    assert repeated.cache_hit is True
    repeated.structural_metrics["coherence"] = 1
    assert (
        optimizer.is_prime_optimized(4, include_metrics=True).structural_metrics[
            "coherence"
        ]
        == 0.25
    )


def test_repeated_calls_preserve_previous_cache_hit_observations():
    optimizer = OptimizedTNFRPrimality(sieve_limit=10)
    first = optimizer.is_prime_optimized(7)
    second = optimizer.is_prime_optimized(7)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first is not second


@pytest.mark.parametrize(
    "threshold,expected", [(0.5, False), (2.1, False), (3.0, True)]
)
def test_configured_pressure_cut_is_independent_of_sieve_coverage(threshold, expected):
    for limit in (2, 10):
        optimizer = OptimizedTNFRPrimality(sieve_limit=limit)
        composite = optimizer.is_prime_optimized(4, threshold=threshold)
        assert composite.delta_nfr == pytest.approx(2.1)
        assert composite.is_prime is expected
        assert optimizer.is_prime_optimized(5, threshold=threshold).is_prime is True
        batch = optimizer.batch_test([5, 4, 4], threshold=threshold)
        assert [(row.n, row.is_prime) for row in batch] == [(4, expected), (5, True)]


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        np.bool_(True),
        "1",
        0,
        -1,
        float("nan"),
        float("inf"),
        Fraction(1, 10**400),
    ],
)
def test_threshold_admission_precedes_cache_lookup_and_batch_deduplication(invalid):
    optimizer = OptimizedTNFRPrimality(sieve_limit=10)
    optimizer.is_prime_optimized(4, threshold=1)
    retained = len(optimizer.result_cache)
    with pytest.raises((TypeError, ValueError), match="threshold"):
        optimizer.is_prime_optimized(4, threshold=invalid)
    with pytest.raises((TypeError, ValueError), match="threshold"):
        optimizer.batch_test([], threshold=invalid)
    assert len(optimizer.result_cache) == retained


@pytest.mark.parametrize("invalid", [True, np.bool_(True), 1.0, "1"])
def test_integer_admission_precedes_result_cache_and_input_equality(invalid):
    optimizer = OptimizedTNFRPrimality(sieve_limit=10)
    optimizer.is_prime_optimized(1)
    with pytest.raises(TypeError, match="integer"):
        optimizer.is_prime_optimized(invalid)
    with pytest.raises(TypeError, match="integer"):
        optimizer.compute_delta_nfr(invalid)
    with pytest.raises(TypeError, match="integer"):
        optimizer.batch_test([1, invalid])


def test_normalized_numpy_inputs_and_empty_batch_preserve_decision_contract():
    optimizer = OptimizedTNFRPrimality(sieve_limit=10)
    result = optimizer.is_prime_optimized(
        np.int64(4), threshold=Fraction(3), include_metrics="false"
    )
    assert result.is_prime is True
    assert result.structural_metrics is None
    assert type(result.n) is int
    assert optimizer.batch_test([]) == []
