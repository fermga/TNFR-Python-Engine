from __future__ import annotations
from tnfr.cli import main
from tnfr.helpers import read_structured_file


def test_cli_metrics_runs(tmp_path):
    out = tmp_path / "m.json"
    rc = main(["metrics", "--nodes", "10", "--steps", "50", "--save", str(out)])
    assert rc == 0
    data = read_structured_file(str(out))
    assert "Tg_global" in data
    assert "latency_mean" in data


def test_cli_sequence_file(tmp_path):
    seq_file = tmp_path / "seq.json"
    seq_file.write_text("[{\"WAIT\": 1}]", encoding="utf-8")
    rc = main(["sequence", "--sequence-file", str(seq_file), "--nodes", "5"])
    assert rc == 0
