"""Tests for --xi-norm flag modes (raw vs log-sat) in critical_regime_detector.

Confirms:
 1. raw mode produces norm_xi == norm_xi_raw.
 2. log-sat mode produces norm_xi <= norm_xi_raw for xi_c > n_nodes.
 3. Risk differs between modes for large xi_c (raw higher or equal).
 4. risk_version tag matches mode.
"""
from __future__ import annotations

import importlib.util
import pathlib

MOD_PATH = pathlib.Path("benchmarks/critical_regime_detector.py").resolve()
_spec = importlib.util.spec_from_file_location(
    "critical_regime_detector_modes", str(MOD_PATH)
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


def test_raw_mode_identity():
    r = mod.compute_risk(_record(1000, 5000.0), 0.5, 0.3, 0.2, "raw")
    assert r["norm_xi"] == r["norm_xi_raw"]
    assert r["risk_version"] == "raw_xi_v1"


def test_log_sat_mode_compression():
    r = mod.compute_risk(_record(1000, 5000.0), 0.5, 0.3, 0.2, "log-sat")
    assert r["norm_xi"] <= r["norm_xi_raw"]
    assert r["risk_version"] == "log_sat_xi_v1"


def test_risk_difference_large_xi():
    raw_r = mod.compute_risk(_record(1000, 8000.0), 0.5, 0.3, 0.2, "raw")
    sat_r = mod.compute_risk(_record(1000, 8000.0), 0.5, 0.3, 0.2, "log-sat")
    assert raw_r["risk"] >= sat_r["risk"]  # raw should not be lower


def test_modes_small_xi_ordering():
    raw_r = mod.compute_risk(_record(1000, 10.0), 0.5, 0.3, 0.2, "raw")
    sat_r = mod.compute_risk(_record(1000, 10.0), 0.5, 0.3, 0.2, "log-sat")
    # Saturated should be >= raw when xi_c << n_nodes (log(1+x)/log(1+n) > x/n)
    assert sat_r["norm_xi"] >= raw_r["norm_xi"]
