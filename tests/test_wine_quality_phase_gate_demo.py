from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from urllib.error import URLError

import pytest


pytest.importorskip("sklearn")


def _load_demo_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "92_wine_quality_phase_gate_demo.py"
    spec = importlib.util.spec_from_file_location("wine_quality_phase_gate_demo", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_online_wine_quality_phase_gate_demo_prioritizes_quality_conflicts(
    tmp_path: Path,
):
    module = _load_demo_module()
    try:
        summary = module.run_demo(
            cache_path=tmp_path / "winequality-red.csv",
            output_dir=tmp_path,
            download_timeout=20.0,
        )
    except (TimeoutError, URLError, OSError) as exc:
        pytest.skip(f"UCI Wine Quality dataset unavailable: {exc}")

    assert summary["dataset"]["sector"] == "food chemistry / quality control"
    assert summary["dataset"]["samples"] == 1599
    assert summary["graph"]["edges"] > summary["graph"]["nodes"]
    assert summary["phase_gate"]["violations"] > 0
    assert 0.0 < summary["phase_gate"]["edge_compliance"] < 1.0
    assert summary["review_definition"]["review_node_count"] > 0

    results = {row["score"]: row for row in summary["score_comparison"]}
    tnfr = results["TNFR phase-stress hotspot"]
    distance = results["Mean chemistry-neighbour distance"]
    constant = results["Global constant baseline"]

    assert tnfr["auc"] >= 0.95
    assert tnfr["precision_at_review_count"] >= 0.90
    assert distance["auc"] < 0.60
    assert constant["auc"] == 0.5

    first_hotspot = summary["top_hotspots"][0]
    assert first_hotspot["incident_quality_conflicts"] >= 9
    assert first_hotspot["prescription"] == ["IL", "OZ", "THOL", "SHA"]

    assert (tmp_path / "wine_quality_phase_gate_demo.json").exists()
    assert (tmp_path / "wine_quality_phase_gate_demo.md").exists()
    assert (tmp_path / "wine_quality_phase_gate_demo.html").exists()
