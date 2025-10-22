from __future__ import annotations

import argparse

import networkx as nx

from tnfr.cli.execution import apply_cli_config
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
