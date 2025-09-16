"""Pruebas de integración para los comandos de la CLI."""

from __future__ import annotations

import json

import pytest

from tnfr.cli import main


pytestmark = pytest.mark.slow  # noqa: F841


def test_metrics_command_generates_output(tmp_path):
    out = tmp_path / "metrics.json"
    rc = main(["metrics", "--nodes", "6", "--steps", "10", "--save", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert "Tg_global" in data
    assert "latency_mean" in data


def test_sequence_command_with_file(tmp_path):
    seq_file = tmp_path / "seq.json"
    seq_file.write_text('[{"WAIT": 1}]', encoding="utf-8")
    rc = main(["sequence", "--sequence-file", str(seq_file), "--nodes", "5"])
    assert rc == 0


def test_run_command_reports_summary(capsys):
    rc = main(["run", "--nodes", "5", "--steps", "1", "--summary"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Tg global" in out
