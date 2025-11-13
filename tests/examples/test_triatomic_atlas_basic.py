from __future__ import annotations

import json
import os

from examples.triatomic_atlas import main as run_triatomic


def test_triatomic_atlas_outputs(tmp_path):
    # Run the atlas (writes to examples/output)
    out_html = run_triatomic()
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "output")
    base_dir = os.path.abspath(base_dir)

    # Check files exist
    html_path = out_html
    jsonl_path = os.path.join(base_dir, "triatomic_atlas.jsonl")
    csv_path = os.path.join(base_dir, "triatomic_atlas.csv")

    assert os.path.exists(html_path), f"missing html output: {html_path}"
    assert os.path.exists(jsonl_path), f"missing jsonl output: {jsonl_path}"
    assert os.path.exists(csv_path), f"missing csv output: {csv_path}"

    # JSONL contains signature and geometry fields; validate known cases
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line.strip()) for line in f if line.strip()]
    assert rows, "no JSONL rows"
    # Generic fields present
    assert "signature" in rows[0], "signature missing in JSONL row"
    assert "geometry" in rows[0] and "angle_deg" in rows[0], "geometry/angle missing in JSONL row"

    # Find specific molecules by formula
    by_formula = {r.get("formula"): r for r in rows}
    assert "HOH" in by_formula and "OCO" in by_formula, "expected HOH and OCO entries"

    h2o = by_formula["HOH"]
    co2 = by_formula["OCO"]
    assert h2o.get("geometry") == "bent", f"H2O should be bent, got {h2o.get('geometry')}"
    assert co2.get("geometry") == "linear", f"CO2 should be linear, got {co2.get('geometry')}"
    assert abs(float(h2o.get("angle_deg", 0.0)) - 104.5) < 1.0, "H2O angle estimate should be ~104.5°"
