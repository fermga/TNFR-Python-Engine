"""Pruebas de cli sanity."""
from __future__ import annotations
import argparse
from tnfr.cli import main, add_common_args, add_grammar_args, _build_graph_from_args
from tnfr.constants import METRIC_DEFAULTS
from tnfr.io_utils import read_structured_file
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


def test_grammar_args_help_group(capsys):
    parser = argparse.ArgumentParser()
    add_grammar_args(parser)
    parser.print_help()
    out = capsys.readouterr().out
    assert "Grammar" in out
    assert "--grammar.enabled" in out


def test_args_to_dict_nested_options():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_grammar_args(parser)
    args = parser.parse_args(
        [
            "--nodes",
            "5",
            "--grammar.enabled",
            "--grammar.thol_min_len",
            "7",
        ]
    )
    G = _build_graph_from_args(args)
    canon = G.graph["GRAMMAR_CANON"]
    assert canon["enabled"] is True
    assert canon["thol_min_len"] == 7
    assert METRIC_DEFAULTS["GRAMMAR_CANON"]["thol_min_len"] == 2
