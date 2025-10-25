from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

from tnfr.cli import execution


@pytest.fixture()
def graph_with_history() -> tuple[nx.Graph, dict[str, Any]]:
    graph = nx.Graph()
    history: dict[str, Any] = {"C_steps": [0], "phase_state": ["stable"]}
    graph.graph["history"] = history
    return graph, history


def test_persist_history_skips_when_disabled(
    graph_with_history: tuple[nx.Graph, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, history = graph_with_history
    args = argparse.Namespace(
        save_history=None,
        export_history_base=None,
        export_format="json",
    )

    ensure_called = False

    def fake_ensure_history(G: nx.Graph) -> dict[str, Any]:
        nonlocal ensure_called
        ensure_called = True
        return history

    saved_payloads: list[tuple[str, dict[str, Any]]] = []
    exported_payloads: list[tuple[nx.Graph, str, str]] = []

    monkeypatch.setattr(execution, "ensure_history", fake_ensure_history)
    monkeypatch.setattr(
        execution,
        "_save_json",
        lambda path, data: saved_payloads.append((path, data)),
    )
    monkeypatch.setattr(
        execution,
        "export_metrics",
        lambda G, base_path, *, fmt: exported_payloads.append((G, base_path, fmt)),
    )

    execution._persist_history(graph, args)

    assert ensure_called is False
    assert saved_payloads == []
    assert exported_payloads == []
    assert graph.graph["history"] is history


def test_persist_history_saves_json(
    graph_with_history: tuple[nx.Graph, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph, history = graph_with_history
    destination = tmp_path / "history.json"
    args = argparse.Namespace(
        save_history=str(destination),
        export_history_base=None,
        export_format="json",
    )

    ensure_args: list[nx.Graph] = []
    saved_payloads: list[tuple[str, dict[str, Any]]] = []
    exported_payloads: list[tuple[nx.Graph, str, str]] = []

    def fake_ensure_history(G: nx.Graph) -> dict[str, Any]:
        ensure_args.append(G)
        return history

    def fake_save_json(path: str, data: dict[str, Any]) -> None:
        saved_payloads.append((path, data))

    monkeypatch.setattr(execution, "ensure_history", fake_ensure_history)
    monkeypatch.setattr(execution, "_save_json", fake_save_json)
    monkeypatch.setattr(
        execution,
        "export_metrics",
        lambda G, base_path, *, fmt: exported_payloads.append((G, base_path, fmt)),
    )

    execution._persist_history(graph, args)

    assert ensure_args == [graph]
    assert saved_payloads == [(str(destination), history)]
    assert exported_payloads == []
    assert graph.graph["history"] is history


def test_persist_history_exports_metrics(
    graph_with_history: tuple[nx.Graph, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph, history = graph_with_history
    base_path = tmp_path / "export"
    args = argparse.Namespace(
        save_history=None,
        export_history_base=str(base_path),
        export_format="jsonl",
    )

    ensure_args: list[nx.Graph] = []
    saved_payloads: list[tuple[str, dict[str, Any]]] = []
    exported_payloads: list[tuple[nx.Graph, str, str]] = []

    def fake_ensure_history(G: nx.Graph) -> dict[str, Any]:
        ensure_args.append(G)
        return history

    monkeypatch.setattr(execution, "ensure_history", fake_ensure_history)
    monkeypatch.setattr(
        execution,
        "_save_json",
        lambda path, data: saved_payloads.append((path, data)),
    )

    def fake_export_metrics(G: nx.Graph, base: str, *, fmt: str) -> None:
        exported_payloads.append((G, base, fmt))

    monkeypatch.setattr(execution, "export_metrics", fake_export_metrics)

    execution._persist_history(graph, args)

    assert ensure_args == [graph]
    assert saved_payloads == []
    assert exported_payloads == [(graph, str(base_path), "jsonl")]
    assert graph.graph["history"] is history
