"""Scope regression for the structural initialization benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tnfr.sdk.simple import Network

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


def test_report_preserves_nonunit_coherence_and_constructed_winding_scope(
    capsys,
    monkeypatch,
) -> None:
    # Exercise report wiring with a nonunit read-out: it must not be promoted
    # to a successful C=1 claim. The real operator evolution still executes.
    monkeypatch.setattr(Network, "coherence", lambda self: 0.375)
    evolve = Network.evolve
    seeds = []

    def evolve_with_seed_evidence(self, *args, **kwargs):
        seeds.append(self.G.graph.get("RANDOM_SEED"))
        return evolve(self, *args, **kwargs)

    monkeypatch.setattr(Network, "evolve", evolve_with_seed_evidence)
    _BENCHMARK.main()
    output = capsys.readouterr().out

    assert seeds == [0] * 5
    assert "canonical C(t) = 0.3750" in output
    assert "snapshot has C(t) = 1" not in output
    assert "snapshot has W = 0" in output
    assert "declared basic_activation word" in output
    assert "Directly constructed unit-winding control" in output
    assert "input fixture, not the output of M1-M3 dynamics" in output
    assert "No Kibble transition, physical" in output
    assert "causal cone is inferred" in output
