from __future__ import annotations

import copy
import re

import networkx as nx  # type: ignore[import-untyped]
import pytest

from tnfr.cli import main
from tnfr.constants import VF_PRIMARY
from tnfr.metrics.core import _update_nu_f_snapshot
from tnfr.telemetry.nu_f import ensure_nu_f_telemetry


def test_math_engine_cli_preserves_classic_metrics(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Ensure math-engine runs report extra summaries without altering runtime state."""

    recorded_runs: list[dict[str, object]] = []

    def fake_build_basic_graph(args):  # noqa: ANN001 - test helper signature
        graph = nx.Graph()
        graph.add_node(0, EPI=0.9, theta=0.1, **{VF_PRIMARY: 0.8})
        graph.add_node(1, EPI=0.6, theta=-0.2, **{VF_PRIMARY: 0.7})
        graph.graph["COHERENCE"] = {"enabled": False}
        graph.graph["DIAGNOSIS"] = {"enabled": False}
        graph.graph["history"] = {"W_stats": [], "nodal_diag": []}
        graph.graph["HZ_STR_BRIDGE"] = 1.75
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
            data[VF_PRIMARY] = float(data[VF_PRIMARY]) + 0.02
            data["theta"] = float(data["theta"]) + 0.01
        history = graph.graph.setdefault("history", {"W_stats": [], "nodal_diag": []})
        history.setdefault("C_steps", []).append({"step": steps, "dt": dt})
        accumulator = ensure_nu_f_telemetry(graph, confidence_level=0.9)
        accumulator.record_counts(5, 1.2, graph=graph)
        accumulator.record_counts(7, 0.8, graph=graph)
        _update_nu_f_snapshot(graph, history, record_history=True)
        telemetry = graph.graph.setdefault("telemetry", {})
        payload = telemetry.get("nu_f_snapshot")
        if isinstance(payload, dict):
            bridge_raw = telemetry.get("nu_f_bridge")
            try:
                bridge = float(bridge_raw) if bridge_raw is not None else None
            except (TypeError, ValueError):
                bridge = None
            nu_f_summary = {
                "total_reorganisations": payload.get("total_reorganisations"),
                "total_duration": payload.get("total_duration"),
                "rate_hz_str": payload.get("rate_hz_str"),
                "rate_hz": payload.get("rate_hz"),
                "variance_hz_str": payload.get("variance_hz_str"),
                "variance_hz": payload.get("variance_hz"),
                "confidence_level": payload.get("confidence_level"),
                "ci_hz_str": {
                    "lower": payload.get("ci_lower_hz_str"),
                    "upper": payload.get("ci_upper_hz_str"),
                },
                "ci_hz": {
                    "lower": payload.get("ci_lower_hz"),
                    "upper": payload.get("ci_upper_hz"),
                },
                "bridge": bridge,
            }
            telemetry["nu_f"] = nu_f_summary
            math_summary = telemetry.setdefault("math_engine", {})
            math_summary["nu_f"] = dict(nu_f_summary)
        return None

    execution_mod = pytest.importorskip("tnfr.cli.execution")
    original_run_program = execution_mod.run_program

    def spy_run_program(graph, program, args):  # noqa: ANN001 - integration helper
        result_graph = original_run_program(graph, program, args)
        recorded_runs.append(
            {
                "nodes": {n: dict(data) for n, data in result_graph.nodes(data=True)},
                "history": copy.deepcopy(result_graph.graph.get("history", {})),
                "math_cfg": result_graph.graph.get("MATH_ENGINE"),
                "telemetry": copy.deepcopy(result_graph.graph.get("telemetry", {})),
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
    nu_f_messages = [msg for msg in math_logs if "νf positivity" in msg]
    assert nu_f_messages, "νf positivity metrics must be reported"
    min_matches = [re.search(r"min=([-+]?\d*\.?\d+)", msg) for msg in nu_f_messages]
    reported_mins = [float(match.group(1)) for match in min_matches if match]
    assert reported_mins, "logged νf positivity metrics must expose a minimum value"

    telemetry = math_snapshot.get("telemetry", {})
    nu_f_summary = telemetry.get("nu_f")
    assert isinstance(nu_f_summary, dict), "runtime telemetry must expose νf summary"
    assert nu_f_summary["rate_hz_str"] is not None
    assert nu_f_summary["rate_hz"] is not None
    assert nu_f_summary["ci_hz"]["lower"] is not None
    assert nu_f_summary["ci_hz"]["upper"] is not None
    assert nu_f_summary["ci_hz_str"]["lower"] >= 0.0
    assert nu_f_summary["ci_hz_str"]["upper"] >= nu_f_summary["ci_hz_str"]["lower"]
    assert nu_f_summary["ci_hz"]["upper"] >= nu_f_summary["ci_hz"]["lower"]
    assert nu_f_summary["confidence_level"] == pytest.approx(0.9)
    assert nu_f_summary["bridge"] == pytest.approx(1.75)
    assert nu_f_summary["rate_hz"] == pytest.approx(
        nu_f_summary["rate_hz_str"] * nu_f_summary["bridge"]
    )
    assert nu_f_summary["variance_hz"] == pytest.approx(
        nu_f_summary["variance_hz_str"] * (nu_f_summary["bridge"] ** 2)
    )

    history = math_snapshot["history"]
    assert history["nu_f_rate_hz_str"][-1] == pytest.approx(nu_f_summary["rate_hz_str"])
    assert history["nu_f_rate_hz"][-1] == pytest.approx(nu_f_summary["rate_hz"])
    assert history["nu_f_ci_lower_hz_str"][-1] == pytest.approx(nu_f_summary["ci_hz_str"]["lower"])
    assert history["nu_f_ci_upper_hz_str"][-1] == pytest.approx(nu_f_summary["ci_hz_str"]["upper"])
    assert history["nu_f_ci_lower_hz"][-1] == pytest.approx(nu_f_summary["ci_hz"]["lower"])
    assert history["nu_f_ci_upper_hz"][-1] == pytest.approx(nu_f_summary["ci_hz"]["upper"])

    math_cfg = math_snapshot["math_cfg"]
    assert math_cfg is not None and math_cfg.get("enabled")
    hilbert_space = math_cfg["hilbert_space"]
    state_projector = math_cfg.get("state_projector")
    validator = math_cfg.get("validator")
    frequency_operator = math_cfg.get("frequency_operator")

    assert state_projector is not None
    assert validator is not None

    projected_freq_values = []
    for node_data in math_snapshot["nodes"].values():
        state = state_projector(
            epi=float(node_data["EPI"]),
            nu_f=float(node_data[VF_PRIMARY]),
            theta=float(node_data["theta"]),
            dim=hilbert_space.dimension,
        )
        outcome = validator.validate(
            state,
            enforce_frequency_positivity=bool(frequency_operator),
        )
        summary = outcome.summary.get("frequency")
        if isinstance(summary, dict) and "value" in summary:
            projected_freq_values.append(float(summary["value"]))

    assert projected_freq_values, "validator should supply frequency values for comparison"
    expected_min = min(projected_freq_values)
    assert reported_mins[0] == pytest.approx(expected_min)
    assert expected_min > 0.0

    assert classic_snapshot["nodes"] == math_snapshot["nodes"]
    assert classic_snapshot["history"] == math_snapshot["history"]
    assert classic_snapshot["telemetry"] == math_snapshot["telemetry"]
