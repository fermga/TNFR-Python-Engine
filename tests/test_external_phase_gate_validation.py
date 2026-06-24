from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "external_phase_gate_validation.py"
    spec = importlib.util.spec_from_file_location(
        "external_phase_gate_validation", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
