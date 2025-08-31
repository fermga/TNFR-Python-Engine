from __future__ import annotations
from tnfr.cli import main
from tnfr.helpers import read_structured_file
from tnfr import __version__


def test_cli_metrics_runs(tmp_path):
    out = tmp_path / "m.json"
    rc = main(["metrics", "--nodes", "10", "--steps", "50", "--save", str(out)])
    assert rc == 0
    data = read_structured_file(out)
    assert "Tg_global" in data
    assert "latency_mean" in data


def test_cli_sequence_file(tmp_path):
    seq_file = tmp_path / "seq.json"
    seq_file.write_text("[{\"WAIT\": 1}]", encoding="utf-8")
    rc = main(["sequence", "--sequence-file", str(seq_file), "--nodes", "5"])
    assert rc == 0


def test_cli_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert __version__ in out


def test_cli_run_erdos_p():
    rc = main(["run", "--topology", "erdos", "--p", "0.9", "--nodes", "5", "--steps", "1"])
    assert rc == 0
