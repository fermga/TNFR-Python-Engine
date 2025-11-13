from __future__ import annotations

import json
import os

from examples.phase_gated_coupling_demo import run_demo


def test_phase_gated_coupling_demo_outputs(tmp_path):
    html_path, jsonl_path = run_demo(threshold=0.9)

    assert os.path.exists(html_path), f"missing html output: {html_path}"
    assert os.path.exists(jsonl_path), f"missing jsonl output: {jsonl_path}"

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert len(rows) == 2, "expected two scenarios in JSONL"

    # One should be in-phase and added, the other anti-phase and blocked
    added = [r for r in rows if r.get("edge_added") is True]
    blocked = [r for r in rows if r.get("edge_added") is False]
    assert added and blocked, f"expected one added and one blocked; got added={len(added)}, blocked={len(blocked)}"

    assert "in-phase" in added[0].get("label", ""), "added scenario should be in-phase"
    assert "anti-phase" in blocked[0].get("label", ""), "blocked scenario should be anti-phase"
