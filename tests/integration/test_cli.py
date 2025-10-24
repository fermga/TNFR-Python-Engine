"""Integration tests covering CLI flows and argument utilities."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import deque
from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import pytest

from tnfr.cli import main
from tnfr.cli.arguments import (
    _args_to_dict,
    add_canon_toggle,
    add_common_args,
    add_grammar_args,
    add_grammar_selector_args,
    add_history_export_args,
)


def _cli_execution():
    return importlib.import_module("tnfr.cli.execution")


from tnfr import __version__
from tnfr.config.presets import get_preset
from tnfr.constants import METRIC_DEFAULTS
from tnfr.execution import CANONICAL_PRESET_NAME, basic_canonical_example

DEPRECATED_PRESET_TOKEN = "deprecated_canonical_example"


def test_cli_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert __version__ in out


@pytest.mark.parametrize(
    ("nodes", "p_arg", "expected_p"),
    [
        ("1", None, 1.0),
        ("2", "0.2", 0.2),
        ("0", None, 0.0),
    ],
)
def test_cli_run_erdos_low_nodes(monkeypatch, nodes, p_arg, expected_p):
    recorded: dict[str, float] = {}
    original = nx.gnp_random_graph

    def spy(n: int, p: float, seed: int | None = None):
        recorded.update({"n": n, "p": p, "seed": seed})
        return original(n, p, seed=seed)

    monkeypatch.setattr("tnfr.cli.execution.nx.gnp_random_graph", spy)

    args = ["run", "--nodes", nodes, "--steps", "0", "--topology", "erdos"]
    if p_arg is not None:
        args.extend(["--p", p_arg])

    rc = main(args)

    assert rc == 0
    assert recorded["n"] == int(nodes)
    assert recorded["p"] == pytest.approx(expected_p)


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_cli_invalid_preset_exits_gracefully(capsys, command):
    args = [command, "--preset", "nope"]
    if command == "run":
        args.extend(["--steps", "0"])

    rc = main(args)
    captured = capsys.readouterr()

    assert rc == 1
    assert "Unknown preset 'nope'." in captured.out
    assert "Available presets" in captured.out
    assert "Use --sequence-file" in captured.out


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_cli_legacy_preset_rejected_with_guidance(capsys, command):
    legacy = DEPRECATED_PRESET_TOKEN
    args = [command, "--preset", legacy]
    if command == "run":
        args.extend(["--steps", "0"])

    rc = main(args)
    captured = capsys.readouterr()

    assert rc == 1
    assert f"Preset not found: {legacy}" in captured.out
    assert "Legacy preset identifier" not in captured.out


def test_cli_sequence_file_missing(tmp_path, capsys):
    missing = tmp_path / "custom.json"

    rc = main(["sequence", "--sequence-file", str(missing)])
    captured = capsys.readouterr()

    assert rc == 1
    assert f"Could not read {missing}" in captured.out


def test_cli_sequence_file_invalid_json(tmp_path, capsys):
    seq_path = tmp_path / "invalid.json"
    seq_path.write_text("{invalid json}", encoding="utf-8")

    rc = main(["sequence", "--sequence-file", str(seq_path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert f"Error parsing JSON file at {seq_path}" in captured.out


def test_cli_metrics_generates_metrics_payload(monkeypatch, tmp_path):
    out = tmp_path / "metrics.json"
    sentinel_graph = object()
    recorded: dict[str, Any] = {}

    def fake_run_cli_program(args):  # noqa: ANN001 - test helper
        recorded["args_steps"] = getattr(args, "steps", None)
        return 0, sentinel_graph

    expected_summary = {
        "Tg_global": {"AL": 0.5},
        "latency_mean": 1.5,
        "rose": {"mag": 0.1},
        "glyphogram": {"t": list(range(15))},
    }

    def fake_build_summary(graph, *, series_limit=None):  # noqa: ANN001 - test helper
        recorded["graph"] = graph
        recorded["series_limit"] = series_limit
        return expected_summary, True

    monkeypatch.setattr("tnfr.cli.execution._run_cli_program", fake_run_cli_program)
    monkeypatch.setattr("tnfr.cli.execution.build_metrics_summary", fake_build_summary)

    rc = main(["metrics", "--nodes", "6", "--steps", "5", "--save", str(out)])

    assert rc == 0
    assert recorded["graph"] is sentinel_graph
    assert recorded["args_steps"] == 5
    assert recorded["series_limit"] is None
    data = json.loads(out.read_text())
    assert data == expected_summary


def test_cli_metrics_accepts_summary_limit(monkeypatch):
    sentinel_graph = object()
    recorded: dict[str, Any] = {}

    def fake_run_cli_program(args):  # noqa: ANN001 - test helper
        return 0, sentinel_graph

    def fake_build_summary(graph, *, series_limit=None):  # noqa: ANN001 - test helper
        recorded["graph"] = graph
        recorded["series_limit"] = series_limit
        return {"glyphogram": {}}, False

    monkeypatch.setattr("tnfr.cli.execution._run_cli_program", fake_run_cli_program)
    monkeypatch.setattr("tnfr.cli.execution.build_metrics_summary", fake_build_summary)

    rc = main(["metrics", "--summary-limit", "7"])

    assert rc == 0
    assert recorded["graph"] is sentinel_graph
    assert recorded["series_limit"] == 7


def test_sequence_defaults_to_canonical(monkeypatch):
    recorded: dict[str, Any] = {}
    sentinel = object()

    def fake_get_preset(name: str):
        recorded["preset_name"] = name
        return sentinel

    def fake_run_program(graph, program, args):
        recorded["program"] = program
        return object()

    monkeypatch.setattr("tnfr.cli.execution.get_preset", fake_get_preset)
    monkeypatch.setattr("tnfr.cli.execution.run_program", fake_run_program)

    rc = main(["sequence"])
    assert rc == 0
    assert recorded["preset_name"] == CANONICAL_PRESET_NAME
    assert recorded["program"] is sentinel


def test_build_basic_graph_rejects_negative_probability():
    args = argparse.Namespace(nodes=3, topology="erdos", p=-0.1, seed=None)
    with pytest.raises(ValueError, match="p must be between 0 and 1; received -0.1"):
        _cli_execution().build_basic_graph(args)


def test_build_basic_graph_rejects_unknown_topology():
    args = argparse.Namespace(nodes=3, topology="invalid", p=None, seed=None)
    message = (
        "Invalid topology 'invalid'. Accepted options are: ring, complete, erdos"
    )
    with pytest.raises(ValueError, match=message):
        _cli_execution().build_basic_graph(args)


def test_basic_canonical_example_matches_preset():
    assert basic_canonical_example() == get_preset(CANONICAL_PRESET_NAME)


def test_run_cli_program_handles_system_exit(monkeypatch):
    args = argparse.Namespace()

    def boom(_args, default=None):  # pragma: no cover - defensive default
        raise SystemExit(5)

    monkeypatch.setattr("tnfr.cli.execution.resolve_program", boom)

    code, graph = _cli_execution()._run_cli_program(args)

    assert code == 5
    assert graph is None


def test_run_cli_program_runs_and_returns_graph(monkeypatch):
    args = argparse.Namespace()
    expected_default = object()
    expected_program = object()
    provided_graph = object()
    result_graph = object()
    recorded: dict[str, Any] = {}

    def fake_resolve(_args, default=None):
        recorded["default"] = default
        return expected_program

    def fake_run_program(graph, program, _args):
        recorded["graph"] = graph
        recorded["program"] = program
        return result_graph

    monkeypatch.setattr("tnfr.cli.execution.resolve_program", fake_resolve)
    monkeypatch.setattr("tnfr.cli.execution.run_program", fake_run_program)

    code, graph = _cli_execution()._run_cli_program(
        args, default_program=expected_default, graph=provided_graph
    )

    assert code == 0
    assert graph is result_graph
    assert recorded["default"] is expected_default
    assert recorded["graph"] is provided_graph
    assert recorded["program"] is expected_program


def test_resolve_program_prefers_preset(monkeypatch):
    args = argparse.Namespace(preset="demo", sequence_file=None)
    sentinel = object()

    monkeypatch.setattr("tnfr.cli.execution.get_preset", lambda name: sentinel)

    result = _cli_execution().resolve_program(args, default=[])
    assert result is sentinel


def test_resolve_program_prefers_sequence_file(monkeypatch, tmp_path):
    seq_path = tmp_path / "custom.json"
    args = argparse.Namespace(preset=None, sequence_file=str(seq_path))
    sentinel = object()

    monkeypatch.setattr("tnfr.cli.execution._load_sequence", lambda path: sentinel)

    result = _cli_execution().resolve_program(args, default=[])
    assert result is sentinel


def test_resolve_program_uses_default_when_missing_inputs():
    args = argparse.Namespace(preset=None, sequence_file=None)
    default = basic_canonical_example()

    result = _cli_execution().resolve_program(args, default=default)
    assert result == default


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

    args.extend(
        ["--save-history", str(save_path), "--export-history-base", str(export_base)]
    )

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
        assert "Global Tg" in out
    else:
        assert "Global Tg" not in out


@pytest.mark.parametrize("command", ["run", "sequence"])
def test_cli_without_history_args(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    args: list[str] = [command, "--nodes", "5"]
    if command == "run":
        args.extend(["--steps", "0"])
    rc = main(args)
    assert rc == 0
    assert not any(tmp_path.iterdir())


def test_run_program_delegates_to_dynamics_run(monkeypatch):
    recorded: dict[str, Any] = {}

    def fake_run(
        G,
        *,
        steps: int,
        dt: float | None = None,
        use_Si: bool = True,
        apply_glyphs: bool = True,
    ) -> None:
        recorded.update(
            {
                "graph": G,
                "steps": steps,
                "dt": dt,
                "use_Si": use_Si,
                "apply_glyphs": apply_glyphs,
            }
        )

    monkeypatch.setattr("tnfr.cli.execution.run", fake_run)

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--steps", type=int, default=100)
    add_canon_toggle(parser)
    add_grammar_selector_args(parser)
    add_history_export_args(parser)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--sequence-file", type=str, default=None)
    parser.add_argument("--summary", action="store_true")

    args = parser.parse_args(["--nodes", "4", "--steps", "-2", "--dt", "0.25"])
    args.use_Si = False
    args.apply_glyphs = False

    G = _cli_execution().run_program(None, None, args)

    assert recorded["graph"] is G
    assert (
        recorded["steps"] == 0
    )  # negative steps are clamped to preserve CLI behaviour
    assert recorded["dt"] == pytest.approx(0.25)
    assert recorded["use_Si"] is False
    assert recorded["apply_glyphs"] is False


def test_save_json_serializes_iterables(tmp_path):
    path = tmp_path / "data.json"
    data = {"set": {1, 2}, "tuple": (1, 2), "deque": deque([1, 2])}
    _cli_execution()._save_json(str(path), data)
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
    G = _cli_execution()._build_graph_from_args(args)
    canon = G.graph["GRAMMAR_CANON"]
    assert canon["enabled"] is True
    assert canon["thol_min_len"] == 7
    assert METRIC_DEFAULTS["GRAMMAR_CANON"]["thol_min_len"] == 2


def test_build_graph_uses_prepare_network_defaults():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_grammar_args(parser)
    args = parser.parse_args(["--nodes", "4"])

    G = _cli_execution()._build_graph_from_args(args)

    assert G.graph.get("_tnfr_defaults_attached") is True
    history = G.graph["history"]
    assert "phase_state" in history
    assert callable(G.graph.get("compute_delta_nfr"))
    assert G.graph.get("_dnfr_hook_name") == "default_compute_delta_nfr"


def test_build_graph_attaches_observer_via_prepare_network():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_grammar_args(parser)
    args = parser.parse_args(["--nodes", "4", "--observer"])

    G = _cli_execution()._build_graph_from_args(args)

    assert G.graph.get("_STD_OBSERVER") == "attached"


def test_args_to_dict_filters_none_values():
    parser = argparse.ArgumentParser()
    add_grammar_args(parser)
    args = parser.parse_args(["--grammar.enabled"])
    result = _args_to_dict(args, "grammar_")
    assert result == {"enabled": True}
