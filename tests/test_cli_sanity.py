from __future__ import annotations
import json
from tnfr.cli import main


def test_cli_metrics_runs(tmp_path):
    out = tmp_path / "m.json"
    rc = main(["metrics", "--nodes", "10", "--steps", "50", "--save", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "Tg_global" in data
    assert "latency_mean" in data
