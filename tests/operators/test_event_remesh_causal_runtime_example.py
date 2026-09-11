"""Regression checks for the finite causal event/REMESH example."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tnfr.operators as operators
import tnfr.operators.event_remesh_causal_runtime as causal_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "169_event_remesh_causal_runtime.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "event_remesh_causal_runtime_example",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_protocol_report():
    example = _load_example()
    result = example.run_protocol()
    return example, result, example.build_report(result)


def test_module_stub_and_facade_expose_causal_sequence_api() -> None:
    expected = {
        "CausalEventRemeshCycleReceipt",
        "EventRemeshCycleExecutionSpec",
        "ExecutedEventRemeshCycleSequence",
        "execute_event_remesh_cycle_sequence",
    }

    assert set(causal_module.__all__) == expected
    stub = Path(causal_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert "class CausalEventRemeshCycleReceipt" in stub
    assert "class EventRemeshCycleExecutionSpec" in stub
    assert "class ExecutedEventRemeshCycleSequence" in stub
    assert "def execute_event_remesh_cycle_sequence" in stub
    assert expected <= set(operators.__all__)
    for name in expected:
        assert getattr(operators, name) is getattr(causal_module, name)


@pytest.mark.parametrize(
    "imports",
    (
        "import tnfr.operators; import tnfr.physics",
        "import tnfr.physics; import tnfr.operators",
    ),
)
def test_operator_and_physics_facades_are_cold_import_order_safe(
    imports: str,
) -> None:
    environment = os.environ.copy()
    source_path = str(REPOSITORY_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not existing else os.pathsep.join((source_path, existing))
    )
    completed = subprocess.run(
        [sys.executable, "-c", imports],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_example_binds_one_graph_owned_causal_sequence(
    example_protocol_report,
) -> None:
    _, result, report = example_protocol_report

    assert result.cycle_indices == (0, 1)
    assert result.causal_cycle_order_certified
    assert result.same_graph_execution_provenance_certified
    assert result.whole_sequence_graph_state_atomic
    assert result.exact_recorded_boundary_continuity_certified
    assert result.exact_finite_energy_telescope_certified
    assert report["epi_trajectory"] == [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]
    assert report["exact_total_energy_drop"] == "0/1"
    assert all(row["binding_certified"] for row in report["receipts"])
    assert all(row["spec_schedule_identity"] for row in report["receipts"])
    assert all(row["executed_schedule_identity"] for row in report["receipts"])


def test_example_keeps_offline_and_future_scope_false(
    example_protocol_report,
) -> None:
    _, result, report = example_protocol_report

    assert report["nested_offline_scope"] == {
        "cycle_sequence_shared_graph_provenance": False,
        "runtime_telescope_shared_graph_provenance": False,
        "runtime_telescope_whole_sequence_atomicity": False,
    }
    assert report["scope"] == {
        "runtime_global_gain": False,
        "uniform_repeated_margin": False,
        "repeated_runtime_stability": False,
        "future_stability": False,
        "solver_accuracy": False,
        "solver_order": False,
        "mesh_convergence": False,
        "full_tnfr_stability": False,
        "external_side_effects_rolled_back": False,
    }
    assert not result.runtime_global_gain_certified
    assert not result.repeated_runtime_stability_certified
    assert not result.future_stability_certified


def test_main_emits_finite_json(
    example_protocol_report,
    monkeypatch,
    capsys,
) -> None:
    example, result, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: result)
    monkeypatch.setattr(example, "build_report", lambda _: report)

    example.main()
    decoded = json.loads(capsys.readouterr().out)

    assert decoded["claim"] == (
        "one finite graph-owned causal event/REMESH sequence"
    )
    assert decoded["causal_cycle_order_certified"]
    assert decoded["epi_trajectory"][-1] == decoded["epi_trajectory"][0]
