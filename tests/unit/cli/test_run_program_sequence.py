from __future__ import annotations

import argparse
from typing import Sequence

import networkx as nx
import pytest

from tnfr.cli import execution
from tnfr.types import ProgramTokens


@pytest.fixture()
def simple_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.graph["history"] = {"C_steps": [0]}
    return graph


@pytest.fixture()
def sample_program_tokens() -> ProgramTokens:
    tokens: ProgramTokens = [("emit", {"strength": 1.0})]
    return tokens


@pytest.fixture()
def base_args() -> argparse.Namespace:
    return argparse.Namespace(
        steps=50,
        dt=None,
        use_Si=None,
        apply_glyphs=None,
        save_history=None,
        export_history_base=None,
        export_format="json",
    )


def test_run_program_dispatches_to_play(
    simple_graph: nx.Graph,
    sample_program_tokens: ProgramTokens,
    base_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    play_calls: list[tuple[nx.Graph, Sequence[object]]] = []
    persist_calls: list[tuple[nx.Graph, argparse.Namespace]] = []
    run_called = False

    def fake_play(graph: nx.Graph, tokens: ProgramTokens) -> None:
        play_calls.append((graph, tokens))

    def fake_persist(graph: nx.Graph, args: argparse.Namespace) -> None:
        persist_calls.append((graph, args))

    def fake_run(graph: nx.Graph, *, steps: int, **_: object) -> None:
        nonlocal run_called
        run_called = True

    monkeypatch.setattr(execution, "play", fake_play)
    monkeypatch.setattr(execution, "_persist_history", fake_persist)
    monkeypatch.setattr(execution, "run", fake_run)

    result = execution.run_program(simple_graph, sample_program_tokens, base_args)

    assert result is simple_graph
    assert play_calls == [(simple_graph, sample_program_tokens)]
    assert run_called is False
    assert persist_calls == [(simple_graph, base_args)]


def test_run_program_builds_graph_before_play(
    sample_program_tokens: ProgramTokens,
    base_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built_graph = nx.Graph()
    built_graph.graph["history"] = {"phase": ["stable"]}

    build_calls: list[argparse.Namespace] = []
    play_calls: list[tuple[nx.Graph, Sequence[object]]] = []
    persist_calls: list[tuple[nx.Graph, argparse.Namespace]] = []
    run_called = False

    def fake_build(args: argparse.Namespace) -> nx.Graph:
        build_calls.append(args)
        return built_graph

    def fake_play(graph: nx.Graph, tokens: ProgramTokens) -> None:
        play_calls.append((graph, tokens))

    def fake_persist(graph: nx.Graph, args: argparse.Namespace) -> None:
        persist_calls.append((graph, args))

    def fake_run(graph: nx.Graph, *, steps: int, **_: object) -> None:
        nonlocal run_called
        run_called = True

    monkeypatch.setattr(execution, "_build_graph_from_args", fake_build)
    monkeypatch.setattr(execution, "play", fake_play)
    monkeypatch.setattr(execution, "_persist_history", fake_persist)
    monkeypatch.setattr(execution, "run", fake_run)

    result = execution.run_program(None, sample_program_tokens, base_args)

    assert build_calls == [base_args]
    assert result is built_graph
    assert play_calls == [(built_graph, sample_program_tokens)]
    assert run_called is False
    assert persist_calls == [(built_graph, base_args)]


def test_run_program_with_null_program_uses_run_mode(
    simple_graph: nx.Graph,
    base_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(**vars(base_args))
    args.steps = -5
    args.dt = 0.25
    args.use_Si = False
    args.apply_glyphs = False

    run_calls: list[tuple[nx.Graph, int, dict[str, object]]] = []

    def fake_run(graph: nx.Graph, *, steps: int, **overrides: object) -> None:
        run_calls.append((graph, steps, overrides))

    monkeypatch.setattr(execution, "run", fake_run)
    monkeypatch.setattr(execution, "_persist_history", lambda *_: None)

    result = execution.run_program(simple_graph, None, args)

    assert result is simple_graph
    assert run_calls == [
        (
            simple_graph,
            0,
            {"dt": 0.25, "use_Si": False, "apply_glyphs": False},
        )
    ]
