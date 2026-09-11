"""Regression checks for the causal finite-block margin example."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tnfr.physics as physics
import tnfr.physics.runtime_remesh_schedule_block_margin as block_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "170_runtime_remesh_block_margin.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "runtime_remesh_schedule_block_margin_example",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_protocol_report():
    example = _load_example()
    protocol = example.run_protocol()
    return example, protocol, example.build_report(protocol)


def test_module_stub_and_facade_expose_block_margin_api() -> None:
    expected = {
        "RuntimeRemeshScheduleBlockMarginObservation",
        "observe_executed_event_remesh_block_margin",
    }

    assert set(block_module.__all__) == expected
    stub = Path(block_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert "class RuntimeRemeshScheduleBlockMarginObservation" in stub
    assert "def observe_executed_event_remesh_block_margin" in stub
    assert expected <= set(physics.__all__)
    for name in expected:
        assert getattr(physics, name) is getattr(block_module, name)


@pytest.mark.parametrize(
    "imports",
    (
        "import tnfr.physics; import tnfr.operators",
        "import tnfr.operators; import tnfr.physics",
    ),
)
def test_block_margin_facade_is_cold_import_order_safe(imports: str) -> None:
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


def test_example_reports_exact_positive_and_zero_margins(
    example_protocol_report,
) -> None:
    _, protocol, report = example_protocol_report
    positive = protocol["positive_margin"]
    zero = protocol["zero_margin"]

    assert positive.block_observation_certified
    assert positive.same_graph_execution_provenance_certified
    assert positive.whole_sequence_graph_state_atomic
    assert positive.exact_gain_based_energy_drop_fraction_lower_bound == Fraction(
        139,
        256,
    )
    assert positive.exact_endpoint_energy_gain_upper_bound == Fraction(117, 256)
    assert positive.positive_normalized_block_margin_certified
    assert positive.strict_energy_contraction_observed

    assert zero.block_observation_certified
    assert zero.exact_gain_based_energy_drop_fraction_lower_bound == 0
    assert zero.exact_endpoint_energy_gain_upper_bound == 1
    assert not zero.positive_normalized_block_margin_certified
    assert not zero.strict_energy_contraction_observed
    assert report["positive_block"][
        "exact_gain_based_energy_drop_fraction_lower_bound"
    ] == "139/256"
    assert report["alpha_one_boundary"][
        "exact_gain_based_energy_drop_fraction_lower_bound"
    ] == "0/1"


def test_example_withholds_uniform_and_repeated_scope(
    example_protocol_report,
) -> None:
    _, protocol, report = example_protocol_report
    positive = protocol["positive_margin"]

    assert report["scope"] == {
        "absolute_uniform_margin": False,
        "uniform_normalized_forward_invariant_class_margin": False,
        "intrablock_prefix_bound": False,
        "repeated_runtime_stability": False,
        "future_stability": False,
        "runtime_global_gain": False,
        "solver_accuracy": False,
        "solver_order": False,
        "mesh_convergence": False,
        "full_tnfr_stability": False,
    }
    assert not positive.uniform_class_coercivity_certified
    assert not positive.uniform_repeated_margin_certified
    assert not positive.repeated_runtime_stability_certified
    assert not positive.future_stability_certified


def test_main_emits_finite_json(
    example_protocol_report,
    monkeypatch,
    capsys,
) -> None:
    example, protocol, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: protocol)
    monkeypatch.setattr(example, "build_report", lambda _: report)

    example.main()
    decoded = json.loads(capsys.readouterr().out)

    assert decoded["claim"] == (
        "exact margins for two causally executed finite blocks"
    )
    assert decoded["positive_block"][
        "positive_normalized_block_margin_certified"
    ]
    assert not decoded["alpha_one_boundary"][
        "positive_normalized_block_margin_certified"
    ]
