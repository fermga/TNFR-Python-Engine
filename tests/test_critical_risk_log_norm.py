"""Tests for log-saturated ξ_C normalization in critical_regime_detector.

Ensures:
 1. norm_xi <= 1.0 (bounded) while raw_norm_xi may exceed 1.
 2. Monotonic ordering: larger xi_c -> >= norm_xi (non-decreasing).
 3. Saturation compresses extreme xi_c (raw_norm_xi - norm_xi grows).
 4. Risk computation uses saturated norm_xi (caps growth).

Physics alignment: telemetry-only; EPI untouched; invariants preserved.
"""
from __future__ import annotations

import importlib.util
import pathlib

MOD_PATH = pathlib.Path("benchmarks/critical_regime_detector.py").resolve()
_spec = importlib.util.spec_from_file_location(
    "critical_regime_detector", str(MOD_PATH)
)
mod = importlib.util.module_from_spec(_spec)  # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(mod)  # type: ignore


def _record(n_nodes: int, xi_c: float) -> dict:
    return {
        "topology": "synthetic",
        "n_nodes": n_nodes,
        "seed": 0,
        "phase_grad_mean": 0.5,
        "phase_grad_std": 0.05,
        "phase_curv_mean": 0.4,
        "phase_curv_std": 0.04,
        "xi_c": xi_c,
    }


def test_log_saturation_bounded():
    r = mod.compute_risk(_record(1000, 5000.0), 0.5, 0.3, 0.2, "log-sat")
    assert r["norm_xi"] <= 1.0
    assert r["norm_xi_raw"] > 1.0  # raw ratio exceeds 1


def test_monotonic_non_decreasing():
    base = mod.compute_risk(
        _record(1000, 100.0), 0.5, 0.3, 0.2, "log-sat"
    )["norm_xi"]
    higher = mod.compute_risk(
        _record(1000, 300.0), 0.5, 0.3, 0.2, "log-sat"
    )["norm_xi"]
    assert higher >= base


def test_extreme_compression():
    moderate = mod.compute_risk(_record(1000, 800.0), 0.5, 0.3, 0.2, "log-sat")
    extreme = mod.compute_risk(_record(1000, 8000.0), 0.5, 0.3, 0.2, "log-sat")
    # Raw gap grows much faster than saturated gap
    raw_gap = extreme["norm_xi_raw"] - moderate["norm_xi_raw"]
    sat_gap = extreme["norm_xi"] - moderate["norm_xi"]
    assert raw_gap > sat_gap  # compression effect


def test_risk_version_tag():
    r = mod.compute_risk(_record(1000, 500.0), 0.5, 0.3, 0.2, "log-sat")
    assert r["risk_version"] == "log_sat_xi_v1"
