"""Regression checks for the P2 half-Reception/REMESH kernel example."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tnfr.physics as physics
import tnfr.physics.binary64_p2_reception_stability as p2_module


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "02_physics_regimes"
    / "174_binary64_p2_reception_remesh_stability.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "binary64_p2_reception_remesh_stability_example",
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


def test_module_stub_and_facade_expose_the_narrow_api() -> None:
    expected = {
        "P2HalfReceptionRemeshStabilityCertificate",
        "certify_p2_half_reception_remesh_stability",
    }
    assert set(p2_module.__all__) == expected
    stub = Path(p2_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )
    assert "class P2HalfReceptionRemeshStabilityCertificate" in stub
    assert "def certify_p2_half_reception_remesh_stability" in stub
    assert expected <= set(physics.__all__)
    for name in expected:
        assert getattr(physics, name) is getattr(p2_module, name)


@pytest.mark.parametrize(
    "imports",
    (
        "import tnfr.physics; import tnfr.operators",
        "import tnfr.operators; import tnfr.physics",
    ),
)
def test_facade_is_cold_import_order_safe(imports: str) -> None:
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


def test_example_reports_exact_zero_gain_and_finite_extinction(
    example_protocol_report,
) -> None:
    _example, protocol, report = example_protocol_report
    certificate = protocol["certificate"]

    assert certificate.p2_half_reception_remesh_stability_certificate_certified
    assert report["normalized_metric"] == ["1/4", "3/4"]
    assert report["mix_factor_hex"] == "0x1.0000000000000p-1"
    assert report["q"] == report["eta"] == report["q_eff"] == "0/1"
    assert report["extinction_horizon"] == 3
    assert report["gain_before_horizon"] == "1/1"
    assert report["gain_at_horizon"] == "0/1"
    assert report["ordinary_output"] == [-0.25, -0.25]


def test_example_keeps_signed_zero_and_execution_scope_explicit(
    example_protocol_report,
) -> None:
    _example, _protocol, report = example_protocol_report

    assert report["signed_zero_boundary"] == {
        "numeric_consensus": True,
        "output_hex": ["0x0.0p+0", "-0x0.0p+0"],
        "bit_preservation_certified": False,
    }
    assert report["scope"] == {
        "global_binary64_epi_kernel_family": True,
        "arbitrary_finite_kernel_repetition": True,
        "active_history_exact_extinction": True,
        "complete_reception_stage": False,
        "grammar_execution": False,
        "live_graph_execution": False,
        "solver_accuracy": False,
        "full_tnfr_stability": False,
    }


def test_main_emits_the_prebuilt_finite_report(
    example_protocol_report,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    example, protocol, report = example_protocol_report
    monkeypatch.setattr(example, "run_protocol", lambda: protocol)
    monkeypatch.setattr(example, "build_report", lambda _value: report)

    example.main()

    assert json.loads(capsys.readouterr().out) == report
