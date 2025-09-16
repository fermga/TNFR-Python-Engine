"""Pruebas de cli sanity."""

from __future__ import annotations

import argparse
from unittest import mock

from tnfr.cli import (
    main,
    add_common_args,
    add_grammar_args,
    _build_graph_from_args,
    _args_to_dict,
)
from tnfr.cli.arguments import GRAMMAR_ARG_SPECS
from tnfr.constants import METRIC_DEFAULTS
from tnfr import __version__


def test_cli_metrics_collects_and_saves_metrics(tmp_path):
    """El subcomando metrics delega en la ejecución y persiste los datos."""

    sentinel_graph = object()
    save_path = tmp_path / "m.json"

    with (
        mock.patch(
            "tnfr.cli.execution.run_program", return_value=sentinel_graph
        ) as run_program,
        mock.patch("tnfr.cli.execution.Tg_global", return_value=0.5),
        mock.patch(
            "tnfr.cli.execution.latency_series", return_value={"value": [2.0]}
        ),
        mock.patch("tnfr.cli.execution.sigma_rose", return_value={"rose": 1}),
        mock.patch(
            "tnfr.cli.execution.glyphogram_series",
            return_value={"glyph": list(range(3))},
        ),
        mock.patch("tnfr.cli.execution._save_json") as save_json,
    ):
        rc = main(
            ["metrics", "--nodes", "10", "--steps", "50", "--save", str(save_path)]
        )

    assert rc == 0
    run_program.assert_called_once()
    run_args = run_program.call_args.args
    assert run_args[0] is None
    assert run_args[1] is None
    parsed_args = run_args[2]
    assert parsed_args.nodes == 10
    assert parsed_args.steps == 50
    save_json.assert_called_once()
    assert save_json.call_args.args[0] == str(save_path)
    payload = save_json.call_args.args[1]
    assert payload["Tg_global"] == 0.5
    assert payload["latency_mean"] == 2.0
    assert payload["rose"] == {"rose": 1}
    assert payload["glyphogram"] == {"glyph": [0, 1, 2]}


def test_cli_sequence_file_uses_loaded_program(tmp_path):
    """El comando sequence usa la secuencia cargada sin ejecutar la engine real."""

    seq_file = tmp_path / "seq.json"
    sentinel_program = object()

    with (
        mock.patch(
            "tnfr.cli.execution._load_sequence",
            return_value=sentinel_program,
        ) as load_sequence,
        mock.patch("tnfr.cli.execution.run_program") as run_program,
    ):
        rc = main(["sequence", "--sequence-file", str(seq_file), "--nodes", "5"])

    assert rc == 0
    load_sequence.assert_called_once()
    loaded_path = load_sequence.call_args.args[0]
    assert str(loaded_path) == str(seq_file)
    run_program.assert_called_once()
    run_args = run_program.call_args.args
    assert run_args[0] is None
    assert run_args[1] is sentinel_program


def test_cli_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert __version__ in out


def test_cli_run_passes_args_and_logs_summary():
    """El subcomando run construye argumentos y solicita el resumen."""

    sentinel_graph = object()

    with (
        mock.patch(
            "tnfr.cli.execution.run_program", return_value=sentinel_graph
        ) as run_program,
        mock.patch("tnfr.cli.execution._log_run_summaries") as log_summary,
    ):
        rc = main(
            [
                "run",
                "--topology",
                "erdos",
                "--p",
                "0.9",
                "--nodes",
                "5",
                "--steps",
                "1",
                "--summary",
            ]
        )

    assert rc == 0
    run_program.assert_called_once()
    run_args = run_program.call_args.args
    assert run_args[0] is None
    assert run_args[1] is None
    parsed_args = run_args[2]
    assert parsed_args.topology == "erdos"
    assert parsed_args.p == 0.9
    assert parsed_args.summary is True
    log_summary.assert_called_once()
    log_args = log_summary.call_args.args
    assert log_args[0] is sentinel_graph
    assert log_args[1] is parsed_args


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
