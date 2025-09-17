"""Pruebas enfocadas para los flujos de la CLI y utilidades de argumentos."""

from __future__ import annotations

import argparse
import json
from collections import deque

import pytest

from tnfr.cli import main
from tnfr.cli.arguments import (
    GRAMMAR_ARG_SPECS,
    _args_to_dict,
    add_common_args,
    add_grammar_args,
)
from tnfr.cli.execution import _build_graph_from_args, _save_json
from tnfr.constants import METRIC_DEFAULTS
from tnfr import __version__


def test_cli_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert __version__ in out


def test_cli_metrics_generates_metrics_payload(tmp_path):
    out = tmp_path / "metrics.json"
    rc = main(["metrics", "--nodes", "6", "--steps", "5", "--save", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    for key in ("Tg_global", "latency_mean", "rose", "glyphogram"):
        assert key in data
    assert isinstance(data["glyphogram"], dict)
    assert all(len(v) <= 10 for v in data["glyphogram"].values())


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_cli_history_roundtrip(tmp_path, capsys, command):
    save_path = tmp_path / f"{command}-history.json"
    export_base = tmp_path / f"{command}-history"

    args: list[str] = [command, "--nodes", "5"]
    if command == "run":
        args.extend(["--steps", "1", "--summary"])
    else:
        seq_file = tmp_path / "seq.json"
        seq_file.write_text('[{"WAIT": 1}]', encoding="utf-8")
        args.extend(["--sequence-file", str(seq_file)])

    args.extend(["--save-history", str(save_path), "--export-history-base", str(export_base)])

    rc = main(args)
    assert rc == 0

    out = capsys.readouterr().out
    data_save = json.loads(save_path.read_text())
    data_export = json.loads(export_base.with_suffix(".json").read_text())

    assert "epi_support" in data_save
    assert data_save["epi_support"]
    glyphogram = data_export["glyphogram"]
    assert glyphogram["t"]

    if command == "run":
        assert "Tg global" in out
    else:
        assert "Tg global" not in out


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_cli_without_history_args(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    args: list[str] = [command, "--nodes", "5"]
    if command == "run":
        args.extend(["--steps", "0"])
    rc = main(args)
    assert rc == 0
    assert not any(tmp_path.iterdir())


def test_save_json_serializes_iterables(tmp_path):
    path = tmp_path / "data.json"
    data = {"set": {1, 2}, "tuple": (1, 2), "deque": deque([1, 2])}
    _save_json(str(path), data)
    loaded = json.loads(path.read_text())
    assert sorted(loaded["set"]) == [1, 2]
    assert loaded["tuple"] == [1, 2]
    assert loaded["deque"] == [1, 2]


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


def test_build_graph_uses_preparar_red_defaults():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_grammar_args(parser)
    args = parser.parse_args(["--nodes", "4"])

    G = _build_graph_from_args(args)

    assert G.graph.get("_tnfr_defaults_attached") is True
    history = G.graph["history"]
    assert "phase_state" in history
    assert callable(G.graph.get("compute_delta_nfr"))
    assert G.graph.get("_dnfr_hook_name") == "default_compute_delta_nfr"


def test_build_graph_attaches_observer_via_preparar_red():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_grammar_args(parser)
    args = parser.parse_args(["--nodes", "4", "--observer"])

    G = _build_graph_from_args(args)

    assert G.graph.get("_STD_OBSERVER") == "attached"


def test_args_to_dict_filters_none_values():
    parser = argparse.ArgumentParser()
    add_grammar_args(parser)
    args = parser.parse_args(["--grammar.enabled"])
    result = _args_to_dict(args, "grammar_")
    assert result == {"enabled": True}


def test_grammar_args_dest_and_default():
    parser = argparse.ArgumentParser()
    add_grammar_args(parser)
    for opt, _ in GRAMMAR_ARG_SPECS:
        action = next(a for a in parser._actions if opt in a.option_strings)
        assert action.dest == opt.lstrip("-").replace(".", "_")
        assert action.default is None
