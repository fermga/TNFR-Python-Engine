"""Arithmetic CLI delegation, compatibility and truthful benchmark reports."""

import io
import re
import sys

import pytest

from tnfr.mathematics import optimized_primality as arithmetic
from tnfr.tools import tnfr_is_prime_cli as cli
from tnfr.tools import tnfr_is_prime_cli_optimized as legacy


@pytest.fixture
def optimizers(monkeypatch):
    original = arithmetic.OptimizedTNFRPrimality
    created = []

    def create(**kwargs):
        options = dict(backend="numpy", enable_gpu=False, sieve_limit=128)
        options.update(kwargs)
        optimizer = original(**options)
        created.append(optimizer)
        return optimizer

    cli._get_optimizer.cache_clear()
    monkeypatch.setattr(cli, "OptimizedTNFRPrimality", create)
    monkeypatch.setattr(arithmetic, "OptimizedTNFRPrimality", create)
    yield created
    cli._get_optimizer.cache_clear()


def _rows(output):
    rows = []
    for line in output.splitlines():
        columns = line.split()
        if len(columns) >= 3 and re.fullmatch(r"-?\d+", columns[0]):
            rows.append((int(columns[0]), columns[1], float(columns[2])))
    return rows


def test_basic_and_optimized_paths_preserve_the_arithmetic_decision(optimizers):
    for value in (-7, 0, 1, 2, 4, 9, 17, 97, 143, 169, 256):
        expected = value in (2, 17, 97)
        basic = cli.tnfr_is_prime(value)
        optimized = cli.tnfr_is_prime(value, use_optimized=True)
        assert basic[0] is expected
        assert optimized[0] is expected
        assert basic[1] == pytest.approx(optimized[1])
    assert cli.tnfr_delta_nfr(4) == pytest.approx(2.1)
    assert len(optimizers) == 1


def test_production_auto_backend_does_not_require_optional_runtime(monkeypatch):
    def unused():
        pytest.fail("CPU arithmetic must not construct an optional backend")

    monkeypatch.setattr(arithmetic, "JAXBackend", unused)
    monkeypatch.setattr(arithmetic, "TorchBackend", unused)
    optimizer = arithmetic.OptimizedTNFRPrimality(sieve_limit=16)
    assert optimizer.backend_name == "numpy"
    assert optimizer.is_prime_optimized(13).is_prime
    assert optimizer.compute_delta_nfr(25) == pytest.approx(cli.tnfr_delta_nfr(25))


def test_optimized_command_reuses_the_reported_optimizer(optimizers, capsys):
    assert cli.main(["--optimized", "--stats", "17", "4", "17"]) == 0
    output = capsys.readouterr().out
    assert _rows(output) == [(17, "True", 0.0), (4, "False", 2.1), (17, "True", 0.0)]
    assert "Optimizer Statistics:" in output
    assert "cache_size: 2" in output
    assert len(optimizers) == 1


def test_batch_retains_sorted_unique_scope_without_unused_structural_metrics(
    optimizers, capsys, monkeypatch
):
    def unused(*args, **kwargs):
        pytest.fail("performance output does not consume structural metrics")

    optimizer = cli._get_optimizer()
    monkeypatch.setattr(optimizer, "_compute_structural_metrics", unused)
    assert cli.main(["--batch", "--stats", "17", "4", "17"]) == 0
    assert _rows(capsys.readouterr().out) == [(4, "False", 2.1), (17, "True", 0.0)]
    assert len(optimizers) == 1


def test_benchmark_uses_the_actual_shared_report_schema(optimizers, capsys):
    assert cli.main(["--benchmark", "100"]) == 0
    output = capsys.readouterr().out
    assert "Numbers tested: 20" in output
    assert "Primes found: 10" in output
    assert "Largest number tested: 29" in output
    assert len(optimizers) == 1
    assert optimizers[0].sieve_data["limit"] == 100


@pytest.mark.parametrize("maximum", ["-1", "0", "99"])
def test_unsupported_benchmark_range_is_a_parser_error(maximum, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--benchmark", maximum])
    assert caught.value.code == 2
    assert "at least 100" in capsys.readouterr().err


def test_legacy_module_and_cached_flag_share_basic_owner(capsys):
    assert legacy.tnfr_delta_nfr_cached is cli.tnfr_delta_nfr_cached
    assert legacy.tnfr_is_prime(9, use_cached=True) == cli.tnfr_is_prime(9)
    assert legacy.main(["--cached", "--timing", "17", "9"]) == 0
    assert _rows(capsys.readouterr().out) == [(17, "True", 0.0), (9, "False", 2.0)]


def test_basic_override_avoids_optimizer_initialization(optimizers, capsys):
    assert cli.main(["--optimized", "--no-optimize", "17"]) == 0
    assert _rows(capsys.readouterr().out) == [(17, "True", 0.0)]
    assert optimizers == []


@pytest.mark.parametrize("options", [[], ["--timing"], ["--batch", "--stats"]])
def test_cli_output_is_portable_to_windows_text_streams(
    options, optimizers, monkeypatch
):
    raw = io.BytesIO()
    output = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", output)
    assert cli.main([*options, "17", "4"]) == 0
    output.flush()
    assert "DeltaNFR" in raw.getvalue().decode("cp1252")
