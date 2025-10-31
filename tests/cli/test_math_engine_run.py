from __future__ import annotations

import copy

import networkx as nx  # type: ignore[import-untyped]
import pytest

from tnfr.cli import main


def test_math_engine_cli_preserves_classic_metrics(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Ensure math-engine runs report extra summaries without altering runtime state."""

    recorded_runs: list[dict[str, object]] = []

    def fake_build_basic_graph(args):  # noqa: ANN001 - test helper signature
        graph = nx.Graph()
        graph.add_node(0, EPI=0.9, vf=0.8, theta=0.1)
        graph.add_node(1, EPI=0.6, vf=0.7, theta=-0.2)
        graph.graph["COHERENCE"] = {"enabled": False}
        graph.graph["DIAGNOSIS"] = {"enabled": False}
        graph.graph["history"] = {"W_stats": [], "nodal_diag": []}
        return graph

    def fake_prepare_network(graph):  # noqa: ANN001 - integration helper
        return graph

    def fake_register_callbacks(graph):  # noqa: ANN001 - integration helper
        graph.graph.setdefault("hooks", {})["registered"] = True

    def fake_ensure_history(graph):  # noqa: ANN001 - integration helper
        return graph.graph.setdefault("history", {"W_stats": [], "nodal_diag": []})

    def fake_run(graph, *, steps, dt=None, use_Si=True, apply_glyphs=True, n_jobs=None):
        for _, data in graph.nodes(data=True):
            data["EPI"] = float(data["EPI"]) * 0.95
            data["vf"] = float(data["vf"]) + 0.02
            data["theta"] = float(data["theta"]) + 0.01
        history = graph.graph.setdefault("history", {"W_stats": [], "nodal_diag": []})
        history.setdefault("C_steps", []).append({"step": steps, "dt": dt})
        return None

    execution_mod = pytest.importorskip("tnfr.cli.execution")
    original_run_program = execution_mod.run_program

    def spy_run_program(graph, program, args):  # noqa: ANN001 - integration helper
        result_graph = original_run_program(graph, program, args)
        recorded_runs.append(
            {
                "nodes": {n: dict(data) for n, data in result_graph.nodes(data=True)},
                "history": copy.deepcopy(result_graph.graph.get("history", {})),
            }
        )
        return result_graph

    monkeypatch.setattr("tnfr.cli.execution.build_basic_graph", fake_build_basic_graph)
    monkeypatch.setattr("tnfr.cli.execution.prepare_network", fake_prepare_network)
    monkeypatch.setattr(
        "tnfr.cli.execution.register_callbacks_and_observer", fake_register_callbacks
    )
    monkeypatch.setattr("tnfr.cli.execution.ensure_history", fake_ensure_history)
    monkeypatch.setattr("tnfr.cli.execution.run", fake_run)
    monkeypatch.setattr("tnfr.dynamics.runtime.run", fake_run)
    monkeypatch.setattr("tnfr.cli.execution.play", lambda *_, **__: None)
    monkeypatch.setattr(execution_mod, "run_program", spy_run_program)

    rc_classic = main(["run", "--nodes", "2", "--steps", "1"])
    assert rc_classic == 0
    assert recorded_runs
    classic_snapshot = recorded_runs[-1]
    classic_output = capfd.readouterr().out
    assert "[MATH]" not in classic_output

    rc_math = main(
        [
            "run",
            "--nodes",
            "2",
            "--steps",
            "1",
            "--math-engine",
            "--math-coherence-spectrum",
            "0.2",
            "0.25",
            "--math-coherence-threshold",
            "0.18",
            "--math-frequency-diagonal",
            "0.4",
            "0.5",
            "--math-generator-diagonal",
            "0.0",
            "0.05",
        ]
    )
    assert rc_math == 0
    assert len(recorded_runs) >= 2
    math_snapshot = recorded_runs[-1]

    math_output = capfd.readouterr().out
    math_logs = [line for line in math_output.splitlines() if line.startswith("[MATH]")]
    assert math_logs, "math engine summaries should be logged"
    assert any("Hilbert norm" in message for message in math_logs)
    assert any("C_min" in message for message in math_logs)
    assert any("νf positivity" in message for message in math_logs)

    assert classic_snapshot["nodes"] == math_snapshot["nodes"]
    assert classic_snapshot["history"] == math_snapshot["history"]
