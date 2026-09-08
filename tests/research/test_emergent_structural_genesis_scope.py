"""Scope regression for the structural initialization benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "emergent_structural_genesis.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "emergent_structural_genesis",
    _BENCHMARK_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


def test_report_keeps_constructed_winding_separate_from_evolution(
    capsys,
) -> None:
    _BENCHMARK.main()
    output = capsys.readouterr().out

    assert "declared basic_activation word" in output
    assert "Directly constructed unit-winding control" in output
    assert "input fixture, not the output of M1-M3 dynamics" in output
    assert "No Kibble transition, physical" in output
    assert "causal cone is inferred" in output
