from __future__ import annotations

import argparse

import networkx as nx

from tnfr.cli.execution import apply_cli_config, default_glyph_selector, parametric_glyph_selector
from tnfr.constants import METRIC_DEFAULTS


def test_apply_cli_config_sets_requested_telemetry_verbosity() -> None:
    args = argparse.Namespace(
        config=None,
        dt=None,
        integrator=None,
        remesh_mode=None,
        glyph_hysteresis_window=None,
        grammar_canon=None,
        trace_verbosity="detailed",
        metrics_verbosity="basic",
        gamma_type="none",
        gamma_beta=0.0,
        gamma_R0=0.0,
    )
    G = nx.Graph()
    G.graph["TRACE"] = {"enabled": False}

    apply_cli_config(G, args)

    trace_cfg = G.graph["TRACE"]
    assert trace_cfg["verbosity"] == "detailed"
    assert trace_cfg["enabled"] is False

    metrics_cfg = G.graph["METRICS"]
    assert metrics_cfg["verbosity"] == "basic"
    assert metrics_cfg is not METRIC_DEFAULTS["METRICS"]
    assert METRIC_DEFAULTS["TRACE"]["verbosity"] == "debug"
    assert METRIC_DEFAULTS["METRICS"]["verbosity"] == "debug"


def test_apply_cli_config_switches_to_parametric_selector() -> None:
    args = argparse.Namespace(
        config=None,
        selector="param",
        gamma_type="none",
        gamma_beta=0.0,
        gamma_R0=0.0,
    )
    G = nx.Graph()
    G.graph["glyph_selector"] = default_glyph_selector

    apply_cli_config(G, args)

    assert G.graph["glyph_selector"] is parametric_glyph_selector


def test_apply_cli_config_sets_gamma_overrides() -> None:
    args = argparse.Namespace(
        config=None,
        gamma_type="custom",
        gamma_beta=2.5,
        gamma_R0=0.75,
    )
    G = nx.Graph()

    apply_cli_config(G, args)

    assert G.graph["GAMMA"] == {
        "type": "custom",
        "beta": 2.5,
        "R0": 0.75,
    }


def test_apply_cli_config_clones_defaults_when_telemetry_missing_dict() -> None:
    args = argparse.Namespace(
        config=None,
        trace_verbosity="info",
        metrics_verbosity="silent",
        gamma_type="none",
        gamma_beta=0.0,
        gamma_R0=0.0,
    )
    G = nx.Graph()
    G.graph["TRACE"] = None
    G.graph["METRICS"] = None

    apply_cli_config(G, args)

    trace_cfg = G.graph["TRACE"]
    metrics_cfg = G.graph["METRICS"]

    assert trace_cfg is not None and trace_cfg is not METRIC_DEFAULTS["TRACE"]
    assert metrics_cfg is not None and metrics_cfg is not METRIC_DEFAULTS["METRICS"]
    assert trace_cfg["verbosity"] == "info"
    assert metrics_cfg["verbosity"] == "silent"
    # Ensure defaults remain unchanged when cloning occurs.
    assert METRIC_DEFAULTS["TRACE"]["verbosity"] == "debug"
    assert METRIC_DEFAULTS["METRICS"]["verbosity"] == "debug"
