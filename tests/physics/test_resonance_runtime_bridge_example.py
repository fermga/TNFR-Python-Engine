"""Regression checks for the local Resonance realization example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "02_physics_regimes"
    / "164_resonance_runtime_bridge.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "resonance_runtime_bridge_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_and_result():
    example = _load_example()
    return example, example.run_protocol()


def test_example_filters_antiphase_neighbor(example_and_result):
    example, result = example_and_result
    assert result.graph_neighbors == (example.COMPATIBLE, example.INCOMPATIBLE)
    assert result.runtime_neighbors == (example.COMPATIBLE,)
    assert result.phase_incompatible_neighbors == (example.INCOMPATIBLE,)
    assert result.unweighted_runtime_neighbor_mean == 1.0
    assert result.transport_weighted_neighbor_mean == pytest.approx(0.8)
    assert result.runtime_target_value == 0.5
    assert result.identity_gate_passed


def test_example_exposes_fixed_flow_and_switching_boundary(example_and_result):
    _, result = example_and_result
    assert result.affine_jump_certificate is not None
    assert result.post_diffusion_certificate is not None
    assert result.post_diffusion_certificate.is_certified
    assert not result.pre_post_metric_exactly_proportional
    assert result.pre_post_switching_certificate is None
    assert result.pressure_refresh_required


def test_example_report_is_finite_and_scoped(example_and_result):
    example, result = example_and_result
    report = example.build_report(result)
    encoded = json.dumps(report, allow_nan=False, separators=(",", ":"))
    assert len(encoded) < 5000
    assert report["runtime"]["identity_gate_passed"]
    assert report["transport"]["post_flow_certified"]
    assert not report["hybrid_boundary"][
        "global_binary64_runtime_affinity_certified"
    ]
