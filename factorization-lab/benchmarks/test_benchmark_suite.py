"""Classical oracles and report wiring for the factorization benchmark.

These checks do not establish a physical interpretation or a speed advantage.
The actual factorizer has its own tests under factorization-lab/tests.
"""

from types import SimpleNamespace

import benchmark_expansion_suite as benchmark
import pytest


@pytest.mark.parametrize(
    "method", [benchmark.trial_division, benchmark.pollard_rho_simple]
)
def test_classical_methods(method):
    factors, runtime, iterations = method(15)
    assert sorted(factors) == [3, 5]
    assert runtime >= 0
    assert iterations >= 0


@pytest.mark.parametrize(
    "number, expected",
    [(15, [3, 5]), (105, [3, 5, 7]), (343, [7, 7, 7]), (1001, [7, 11, 13])],
)
def test_theoretical_factors(number, expected):
    assert benchmark.get_theoretical_factors(number) == expected


def test_benchmark_suites():
    for suite in benchmark.BENCHMARK_SUITES.values():
        assert suite["description"]
        assert suite["numbers"]


def test_comparative_report_preserves_results_and_computes_accuracy(monkeypatch):
    calls = []
    tnfr_result = SimpleNamespace(tnfr_certified_factors=[5, 3], runtime_ms=4.0)
    classical = [
        benchmark.ClassicalBenchmarkResult("complete", 15, True, [3, 5], 2.0, 3, ""),
        benchmark.ClassicalBenchmarkResult("incomplete", 15, False, [3], 0.5, 1, ""),
    ]

    def tnfr(number, pure_mode):
        calls.append(("tnfr", number, pure_mode))
        return tnfr_result

    def reference(number):
        calls.append(("classical", number))
        return classical

    monkeypatch.setattr(benchmark, "run_tnfr_benchmark", tnfr)
    monkeypatch.setattr(benchmark, "run_classical_benchmark", reference)
    report = benchmark.run_comparative_benchmark(15, "control", pure_mode=False)
    assert calls == [("tnfr", 15, False), ("classical", 15)]
    assert report.n == 15 and report.number_type == "control"
    assert report.theoretical_factors == [3, 5]
    assert report.tnfr_result is tnfr_result
    assert report.classical_results is classical
    assert report.tnfr_advantage == 0.5
    assert report.accuracy_comparison == {
        "complete": True,
        "incomplete": False,
        "tnfr": True,
    }


def test_comparative_report_does_not_hide_producer_exceptions(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("producer failure")

    monkeypatch.setattr(benchmark, "run_tnfr_benchmark", fail)
    with pytest.raises(RuntimeError, match="producer failure"):
        benchmark.run_comparative_benchmark(15, "failure")
