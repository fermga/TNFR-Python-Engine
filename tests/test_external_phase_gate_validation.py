from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "external_phase_gate_validation.py"
    return _load_module(
        "external_phase_gate_validation",
        path,
    )


def test_dynamic_example_import_preserves_tnfr_module_identity():
    import tnfr
    import tnfr.utils.cache as cache_module

    root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "phase_gate_monitor_demo",
        root / "examples" / "10_applications" / "90_phase_gate_monitor_demo.py",
    )

    assert module.ROOT == root
    assert sys.modules["tnfr"] is tnfr
    assert sys.modules["tnfr.utils.cache"] is cache_module


@pytest.mark.parametrize("nodes", (14, 21, 24))
def test_sensor_scramble_preserves_histogram_and_topology_for_each_size(nodes):
    from tnfr.physics.fields import compute_phase_gradient

    root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "phase_gate_monitor_demo",
        root / "examples" / "10_applications" / "90_phase_gate_monitor_demo.py",
    )
    random_state = random.getstate()
    smooth = module.build_sensor_ring(nodes)
    scrambled = module.build_sensor_ring(nodes, scrambled=True)
    repeated = module.build_sensor_ring(nodes, scrambled=True)

    assert random.getstate() == random_state
    assert list(smooth.edges) == list(scrambled.edges)

    def phases(graph):
        return [graph.nodes[node]["theta"] for node in graph]

    assert sorted(phases(smooth)) == sorted(phases(scrambled))
    assert phases(scrambled) == phases(repeated)
    assert sum(compute_phase_gradient(scrambled).values()) > sum(
        compute_phase_gradient(smooth).values()
    )


def test_phase_gate_validation_finds_tnfr_local_advantage(tmp_path: Path):
    module = _load_benchmark_module()
    summary = module.run_validation(
        nodes=64,
        runs=8,
        output_json=tmp_path / "phase_gate_validation.json",
        output_markdown=tmp_path / "phase_gate_validation.md",
        output_html=tmp_path / "phase_gate_validation.html",
    )

    results = {row["model"]: row for row in summary["model_results"]}
    tnfr_grad = results["TNFR mean grad_phi"]["test"]
    global_r = results["Global order parameter R"]["test"]
    topology_degree = results["Topology average degree"]["test"]

    assert tnfr_grad["balanced_accuracy"] >= 0.95
    assert global_r["balanced_accuracy"] <= 0.75
    assert topology_degree["balanced_accuracy"] == 0.5

    paired = summary["paired_wave_checks"]
    assert paired["pair_count"] == 24
    assert paired["label_flips"] == 24
    assert paired["median_abs_delta_global_order_r"] < 1e-12
    assert paired["median_abs_delta_phase_histogram_entropy"] < 1e-12
    assert paired["median_abs_delta_tnfr_mean_phase_gradient"] > 0.5

    assert (tmp_path / "phase_gate_validation.json").exists()
    assert (tmp_path / "phase_gate_validation.md").exists()
    assert (tmp_path / "phase_gate_validation.html").exists()
