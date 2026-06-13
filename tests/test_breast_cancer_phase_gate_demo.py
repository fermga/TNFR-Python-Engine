from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


pytest.importorskip("sklearn")


def _load_demo_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "10_applications" / "91_breast_cancer_phase_gate_demo.py"
    spec = importlib.util.spec_from_file_location("breast_cancer_phase_gate_demo", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wdbc_phase_gate_demo_localizes_biomedical_boundary_cases(tmp_path: Path):
    module = _load_demo_module()
    summary = module.run_demo(output_dir=tmp_path)

    assert summary["dataset"]["sector"] == "biomedical / diagnostic morphology"
    assert summary["dataset"]["samples"] == 569
    assert summary["graph"]["edges"] > summary["graph"]["nodes"]
    assert summary["phase_gate"]["violations"] > 0
    assert 0.0 < summary["phase_gate"]["edge_compliance"] < 1.0
    assert summary["review_definition"]["review_node_count"] > 0

    results = {row["score"]: row for row in summary["score_comparison"]}
    tnfr = results["TNFR phase-stress hotspot"]
    distance = results["Mean morphology-neighbour distance"]
    topology = results["Topology degree"]
    constant = results["Global constant baseline"]

    assert tnfr["auc"] >= 0.95
    assert tnfr["precision_at_review_count"] >= 0.95
    assert distance["auc"] < 0.70
    assert topology["auc"] < 0.70
    assert constant["auc"] == 0.5

    first_hotspot = summary["top_hotspots"][0]
    assert first_hotspot["incident_diagnostic_conflicts"] >= 3
    assert first_hotspot["prescription"] == ["IL", "OZ", "THOL", "SHA"]

    assert (tmp_path / "wdbc_phase_gate_demo.json").exists()
    assert (tmp_path / "wdbc_phase_gate_demo.md").exists()
    assert (tmp_path / "wdbc_phase_gate_demo.html").exists()
