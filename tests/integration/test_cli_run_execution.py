"""Integration coverage for the CLI ``run`` command calling the runtime."""

from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import pytest

from tnfr.cli import main


def test_cli_run_invokes_runtime_with_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``tnfr run`` forwards overrides to the runtime entry point."""

    build_args: dict[str, Any] = {}
    runtime_calls: list[dict[str, Any]] = []

    def fake_build_basic_graph(args):  # noqa: ANN001 - test helper
        build_args["nodes"] = args.nodes
        build_args["topology"] = args.topology
        graph = nx.Graph()
        graph.add_nodes_from(range(args.nodes))
        graph.graph["history"] = {"C_steps": []}
        graph.graph["hooks"] = {}
        return graph

    def fake_prepare_network(graph: nx.Graph) -> None:
        graph.graph["prepared"] = True

    def fake_register_callbacks(graph: nx.Graph) -> None:
        graph.graph["hooks"]["registered"] = True

    def fake_ensure_history(graph: nx.Graph) -> dict[str, Any]:
        return graph.graph.setdefault("history", {"C_steps": []})

    def fake_runtime_run(
        graph: nx.Graph,
        *,
        steps: int,
        dt: float | None = None,
        use_Si: bool = True,
        apply_glyphs: bool = True,
        n_jobs: dict[str, Any] | None = None,
    ) -> None:
        runtime_calls.append(
            {
                "graph": graph,
                "steps": steps,
                "dt": dt,
                "use_Si": use_Si,
                "apply_glyphs": apply_glyphs,
                "n_jobs": n_jobs,
            }
        )
        history = graph.graph.setdefault("history", {"C_steps": []})
        history.setdefault("C_steps", []).append({"step": steps, "dt": dt})
        graph.graph.setdefault("hook_metadata", {})["runtime"] = {
            "apply_glyphs": apply_glyphs,
            "n_jobs": n_jobs,
        }

    monkeypatch.setattr("tnfr.cli.execution.build_basic_graph", fake_build_basic_graph)
    monkeypatch.setattr("tnfr.cli.execution.prepare_network", fake_prepare_network)
    monkeypatch.setattr(
        "tnfr.cli.execution.register_callbacks_and_observer", fake_register_callbacks
    )
    monkeypatch.setattr("tnfr.cli.execution.ensure_history", fake_ensure_history)
    monkeypatch.setattr("tnfr.cli.execution._log_run_summaries", lambda *_, **__: None)
    monkeypatch.setattr("tnfr.dynamics.runtime.run", fake_runtime_run)
    monkeypatch.setattr("tnfr.cli.execution.run", fake_runtime_run)

    rc = main(
        [
            "run",
            "--nodes",
            "4",
            "--steps",
            "2",
            "--topology",
            "complete",
            "--dnfr-n-jobs",
            "3",
            "--dt",
            "0.125",
            "--no-use-Si",
            "--no-apply-glyphs",
        ]
    )

    assert rc == 0
    assert build_args == {"nodes": 4, "topology": "complete"}
    assert len(runtime_calls) == 1
    call = runtime_calls[0]
    assert call["steps"] == 2
    assert call["dt"] == pytest.approx(0.125)
    assert call["use_Si"] is False
    assert call["apply_glyphs"] is False
    assert call["n_jobs"] == {"dnfr_n_jobs": 3}

    graph = call["graph"]
    assert graph.graph["prepared"] is True
    assert graph.graph["hooks"]["registered"] is True
    history = graph.graph["history"]
    assert history["C_steps"]
    assert graph.graph["hook_metadata"]["runtime"] == {
        "apply_glyphs": False,
        "n_jobs": {"dnfr_n_jobs": 3},
    }
