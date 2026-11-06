"""Generic manifest orchestration must not depend on an arithmetic application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from tnfr.engines.manifest import write_manifest_bundle
from tnfr.sdk import self_opt


def test_cold_graph_dry_run_never_imports_lab_or_executes_recommendations(
    tmp_path, source_tree_environment
):
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.2 + 0.1 * node, theta=0.1 * node, nu_f=1.0, delta_nfr=0.1
        )
    paths = write_manifest_bundle(
        tmp_path / "input",
        "manifest.json",
        "summary.json",
        {"operation_type": "partition"},
        {},
        [("sample", graph, {})],
    )
    original = {path: path.read_bytes() for path in (tmp_path / "input").glob("*.json")}
    code = """
import importlib.abc
import sys
from pathlib import Path

class NoArithmeticLab(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tnfr_factorization" or fullname.startswith("tnfr_factorization."):
            raise AssertionError("arithmetic application import attempted")

sys.meta_path.insert(0, NoArithmeticLab())
initial_path = sys.path[:]
from tnfr.sdk import run_partition_self_optimization
assert "scripts.run_self_optimization" not in sys.modules
assert "scripts.run_self_opt_validation" not in sys.modules
from tnfr.dynamics.optimization_orchestrator import TNFROptimizationOrchestrator
calls = []
def forbidden_execution(*args, **kwargs):
    calls.append((args, kwargs))
    raise AssertionError("dry run must not dispatch a recommendation")
TNFROptimizationOrchestrator.execute_optimization = forbidden_execution

result = run_partition_self_optimization(
    sys.argv[1], manifest_summary_path=sys.argv[2],
    output_root=sys.argv[3], base_name="sample", operation_type="partition",
)
assert set(result) == {
    "payload_root", "summary_path", "validation_report", "promotable",
    "runner_summary", "validation_summary",
}
assert result["promotable"] == {}
runner = result["runner_summary"]
assert runner["success_count"] == 1 and runner["failure_count"] == 0, runner
partition = runner["partition_results"][0]
assert partition["engine"]["dry_run"] is True
assert partition["engine"]["learning_updated"] is False
assert partition["telemetry_snapshots"]["before"] == partition["telemetry_snapshots"]["after"]
assert partition["telemetry_deltas"]["delta_c"] == 0.0
validation = result["validation_summary"]
assert validation["status_counts"] == {"validated": 0, "regressed": 0, "pending": 1}
assert validation["test_runs"] == []
assert validation["recommendations_applied"] is False
assert validation["recommendation_effects_evaluated"] is False
assert Path(result["summary_path"]).is_file()
assert Path(result["validation_report"]).is_file()
assert calls == []
assert sys.path == initial_path
assert not any(name.startswith("tnfr_factorization") for name in sys.modules)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(paths["manifest_absolute"]),
            str(paths["summary_absolute"]),
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=source_tree_environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert all(path.read_bytes() == data for path, data in original.items())


def test_installed_sdk_reports_checkout_requirement_without_creating_outputs(
    tmp_path, source_tree_environment
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    code = """
import importlib.abc
import sys
from tnfr.sdk import run_partition_self_optimization
class NoCheckoutScripts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts" or fullname.startswith("scripts."):
            raise ModuleNotFoundError("repository scripts deliberately unavailable")
        if fullname.startswith("tnfr_factorization"):
            raise AssertionError("must not try the historical application")
sys.meta_path.insert(0, NoCheckoutScripts())
try:
    run_partition_self_optimization(sys.argv[1], output_root=sys.argv[2])
except RuntimeError as exc:
    assert "repository checkout" in str(exc)
    assert "not included in the installed wheel" in str(exc)
else:
    raise AssertionError("missing checkout must raise explicitly")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(manifest), str(output)],
        capture_output=True,
        text=True,
        timeout=60,
        env=source_tree_environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not output.exists()


def test_missing_manifest_rejects_before_orchestration(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        self_opt.run_partition_self_optimization(tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("wrapper", "operation"),
    [
        ("run_pattern_discovery_optimization", "pattern_discovery"),
        ("run_fractal_partition_optimization", "fractal_partition"),
        ("run_batch_certificate_optimization", "batch_certificate_optimization"),
    ],
)
def test_specialized_wrappers_preserve_the_generic_workflow_arguments(
    monkeypatch, wrapper, operation
):
    calls = []
    expected = {"promotable": {}}

    def collect(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(self_opt, "run_partition_self_optimization", collect)
    result = getattr(self_opt, wrapper)(
        "manifest.json",
        manifest_summary_path="summary.json",
        base_name="sample",
        output_root=Path("output"),
    )
    assert result is expected
    assert calls == [
        {
            "manifest_path": "manifest.json",
            "manifest_summary_path": "summary.json",
            "base_name": "sample",
            "output_root": Path("output"),
            "operation_type": operation,
        }
    ]
