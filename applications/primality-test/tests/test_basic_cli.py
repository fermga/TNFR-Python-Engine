"""Independent arithmetic and import-free CLI controls for the maintained package."""

import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tnfr_primality import cli, core, optimized  # noqa: E402


def test_default_predicate_and_cache_match_independent_sieve():
    limit = 1000
    prime = [False, False] + [True] * (limit - 1)
    for factor in range(2, limit + 1):
        if prime[factor]:
            for multiple in range(2 * factor, limit + 1, factor):
                prime[multiple] = False
    cache = optimized.OptimizedTNFRPrimality(cache_size=7)
    for n in range(2, limit + 1):
        actual, pressure = core.tnfr_is_prime(n)
        assert actual is prime[n]
        assert cache.is_prime(n) == (actual, pressure)
        assert pressure == 0 if actual else pressure > 2
    assert cache.get_statistics()["cache_size"] == 7


@pytest.mark.parametrize(
    "n,omega,tau,sigma,pressure",
    [
        (2, 1, 2, 3, 0),
        (4, 2, 3, 7, Fraction(5, 2)),
        (9, 2, 3, 13, Fraction(7, 3)),
        (6, 2, 4, 12, Fraction(23, 6)),
    ],
)
def test_component_readouts_keep_exact_arithmetic_meaning(
    n, omega, tau, sigma, pressure
):
    result = core.tnfr_component_breakdown(n)
    assert (result["omega"], result["tau"], result["sigma"]) == (omega, tau, sigma)
    assert result["delta_nfr"] == pytest.approx(float(pressure))


def test_bounded_cache_reuses_results_and_clear_restores_cold_calls(monkeypatch):
    calls = []
    original = optimized.tnfr_delta_nfr

    def observed(n):
        calls.append(n)
        return original(n)

    monkeypatch.setattr(optimized, "tnfr_delta_nfr", observed)
    cache = optimized.OptimizedTNFRPrimality(cache_size=1)
    assert cache.is_prime(17) == cache.is_prime(17) == (True, 0.0)
    assert not cache.is_prime(9)[0]
    assert not cache.is_prime(9)[0]
    assert calls == [17, 9, 9]
    cache.clear_cache()
    assert cache.get_statistics()["tests_performed"] == 0
    cache.is_prime(17)
    assert calls == [17, 9, 9, 17]


def test_benchmark_sampling_does_not_mutate_global_random_state():
    state = random.getstate()
    report = optimized.OptimizedTNFRPrimality().benchmark(120)
    assert random.getstate() == state
    assert report["total_numbers"] == 118


def test_module_json_reports_basic_results_and_undefined_pressure(capsys):
    assert cli.main(["2", "9", "0", "--json-output"]) == 0
    output = capsys.readouterr()
    rows = json.loads(output.out)
    assert output.err == ""
    assert [row["is_prime"] for row in rows] == [True, False, False]
    assert rows[1]["delta_nfr"] == pytest.approx(7 / 3)
    assert rows[2]["delta_nfr"] is None
    assert rows[2]["delta_nfr_available"] is False
    assert all(row["method"] == "Standard" for row in rows)


@pytest.mark.parametrize(
    "number,pressure",
    [(17, float("inf")), (17, float("nan")), (0, -float("inf")), (0, float("nan"))],
)
def test_json_rejects_invalid_computation_instead_of_hiding_it(
    number, pressure, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "tnfr_is_prime", lambda _: (False, pressure))
    with pytest.raises(ValueError, match="non-finite arithmetic pressure"):
        cli.main([str(number), "--json-output"])
    assert capsys.readouterr().out == ""


def test_basic_validation_json_has_finite_range_schema(capsys):
    assert cli.main(["--validate", "7", "--json-output"]) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["tested"] == result["correct"] == 6
    assert result["false_positives"] == result["false_negatives"] == 0
    assert output.err == ""


@pytest.mark.parametrize(
    "arguments,key,expected",
    [
        (["--sieve", "0"], "primes", []),
        (["--sieve", "10"], "primes", [2, 3, 5, 7]),
        (["--benchmark", "10"], "total_numbers", 8),
        (["--compare", "17", "9"], "numbers_tested", 2),
    ],
)
def test_json_modes_keep_basic_result_schemas(arguments, key, expected, capsys):
    assert cli.main([*arguments, "--json-output"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)[key] == expected
    assert output.err == ""


def test_cached_json_keeps_statistics_off_result_stream(capsys):
    assert cli.main(["17", "17", "--batch", "--stats", "--json-output"]) == 0
    output = capsys.readouterr()
    assert [row["is_prime"] for row in json.loads(output.out)] == [True, True]
    statistics = json.loads(output.err)
    assert statistics["cache_hits"] == statistics["cache_misses"] == 1


@pytest.mark.parametrize("flag", ["--advanced", "--cached", "--infrastructure-status"])
def test_retired_options_fail_explicitly_without_fallback(flag, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["17", flag])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mode,value", [("--validate", "1"), ("--benchmark", "2"), ("--sieve", "-1")]
)
def test_range_modes_reject_undefined_input_before_computation(mode, value):
    with pytest.raises(SystemExit) as error:
        cli.main([mode, value])
    assert error.value.code == 2
