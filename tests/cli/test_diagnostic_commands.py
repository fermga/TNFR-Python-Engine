"""Stored CLI diagnostics retain admission and availability boundaries."""

from argparse import ArgumentParser, Namespace
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.cli import execution, main
from tnfr.cli.arguments import _add_epi_validate_parser


def _check(monkeypatch, graph, **checks):
    monkeypatch.setattr(execution, "_run_cli_program", lambda _args: (0, graph))
    options = dict(check_coherence=False, check_frequency=False, check_phase=False)
    options.update(checks)
    return execution.cmd_epi_validate(Namespace(**options))


@pytest.mark.parametrize(
    "capacity",
    [
        None,
        True,
        np.bool_(False),
        "1.0",
        float("nan"),
        float("inf"),
        -1e-12,
        Fraction(1, 10**1000),
    ],
)
def test_invalid_capacity_cannot_pass_or_use_affinity_slack(monkeypatch, capacity):
    graph = nx.empty_graph(1)
    if capacity is not None:
        graph.nodes[0][execution.VF_PRIMARY] = capacity
    assert _check(monkeypatch, graph, check_frequency=True, tolerance=1.0) == 1


def test_invalid_primary_capacity_cannot_fall_through_to_valid_alias(monkeypatch):
    graph = nx.empty_graph(1)
    graph.nodes[0][execution.VF_ALIAS_KEYS[0]] = None
    graph.nodes[0][execution.VF_ALIAS_KEYS[1]] = 1.0
    assert _check(monkeypatch, graph, check_frequency=True) == 1


def test_capacity_accepts_zero_and_valid_legacy_alias(monkeypatch):
    graph = nx.empty_graph(2)
    graph.nodes[0][execution.VF_PRIMARY] = 0.0
    graph.nodes[1][execution.VF_ALIAS_KEYS[1]] = Fraction(1, 2)
    assert _check(monkeypatch, graph, check_frequency=True) == 0


@pytest.mark.parametrize(
    "sample",
    [
        None,
        {},
        {"mean": True},
        {"mean": "0.5"},
        {"mean": float("nan")},
        {"mean": float("inf")},
        {"mean": Fraction(1, 10**1000)},
    ],
)
def test_invalid_affinity_sample_is_not_a_successful_zero(monkeypatch, sample):
    graph = nx.Graph()
    graph.graph["history"] = {"W_stats": [sample]}
    assert _check(monkeypatch, graph, check_coherence=True) == 1


@pytest.mark.parametrize("history", [None, {"W_stats": {"mean": 0.5}}])
def test_malformed_history_cannot_pass(monkeypatch, history):
    graph = nx.Graph()
    graph.graph["history"] = history
    assert _check(monkeypatch, graph, check_coherence=True) == 1


def test_affinity_tolerance_is_reported_without_coherence_claim(monkeypatch):
    messages = []
    monkeypatch.setattr(
        execution.logger, "info", lambda fmt, *args: messages.append(fmt % args)
    )
    graph = nx.Graph()
    graph.graph["history"] = {"W_stats": [{"mean": -1e-7}, {"mean": 0.5}]}
    assert _check(monkeypatch, graph, check_coherence=True, tolerance=1e-6) == 0
    output = "\n".join(messages)
    assert "Affinity sign diagnostic" in output
    assert "Coherence preserved" not in output
    assert "W_mean >= -tolerance" in output
    assert _check(monkeypatch, graph, check_coherence=True, tolerance=0.0) == 1


@pytest.mark.parametrize("check", ["check_coherence", "check_frequency", "check_phase"])
def test_no_available_observations_cannot_pass(monkeypatch, check):
    graph = nx.Graph()
    assert _check(monkeypatch, graph, **{check: True}) == 1
    assert "history" not in graph.graph


@pytest.mark.parametrize("tolerance", ["1e-6", Fraction(1, 10**1000)])
def test_tolerance_uses_shared_raw_represented_real_admission(monkeypatch, tolerance):
    def unexpected_run(_args):
        pytest.fail("an invalid tolerance must not execute the program")

    monkeypatch.setattr(execution, "_run_cli_program", unexpected_run)
    assert execution.cmd_epi_validate(Namespace(tolerance=tolerance)) == 1


def test_selectable_checks_and_empty_selection_rejection(monkeypatch):
    parser = ArgumentParser()
    _add_epi_validate_parser(parser.add_subparsers())
    args = parser.parse_args(
        ["epi.validate", "--no-check-coherence", "--no-check-phase"]
    )
    assert (args.check_coherence, args.check_frequency, args.check_phase) == (
        False,
        True,
        False,
    )
    args = parser.parse_args(
        [
            "epi.validate",
            "--no-check-coherence",
            "--no-check-frequency",
            "--no-check-phase",
        ]
    )

    def unexpected_run(_args):
        pytest.fail("an empty diagnostic selection must not execute the program")

    monkeypatch.setattr(execution, "_run_cli_program", unexpected_run)
    assert args.func(args) == 1


def test_capacity_only_selection_executes_the_real_command(capsys):
    assert (
        main(
            [
                "epi.validate",
                "--nodes",
                "2",
                "--steps",
                "0",
                "--no-check-coherence",
                "--no-check-phase",
            ]
        )
        == 0
    )
    output = capsys.readouterr().err
    assert "Finite capacity nu_f >= 0 for all 2 nodes" in output
    assert "Affinity sign diagnostic" not in output
