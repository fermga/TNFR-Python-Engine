"""Finite curvature variance cuts remain independent of descriptive fit quality."""

import math

import networkx as nx
import pytest

from tnfr.operators.grammar_telemetry import warn_phase_curvature_telemetry
from tnfr.physics import fields


def test_power_fit_residuals_follow_scale_keys_not_mapping_insertion_order():
    variances = {1: 4.0, 2: 1.0, 3: 4.0 / 9.0}
    ordered = fields.fit_k_phi_asymptotic_alpha(variances)
    shuffled = fields.fit_k_phi_asymptotic_alpha(
        {scale: variances[scale] for scale in (3, 1, 2)}
    )
    assert ordered["alpha"] == pytest.approx(2.0, abs=1e-10)
    assert shuffled["residuals"] == ordered["residuals"]
    assert max(map(abs, shuffled["residuals"].values())) < 2e-12


def test_omitting_reference_exponent_preserves_the_observed_fit():
    variances = {1: 4.0, 2: 1.0, 3: 4.0 / 9.0}
    selected_hint = fields.fit_k_phi_asymptotic_alpha(variances)
    no_hint = fields.fit_k_phi_asymptotic_alpha(variances, alpha_hint=None)
    assert no_hint["prediction_error"] is None
    for key in ("alpha", "c", "r_squared", "residuals"):
        assert no_hint[key] == selected_hint[key]


@pytest.mark.parametrize("hint", [None, 2.76])
def test_positive_fit_cannot_override_observed_variance_violation(monkeypatch, hint):
    monkeypatch.setattr(
        fields,
        "compute_k_phi_multiscale_variance",
        lambda graph: {1: 9.0, 2: 2.25, 3: 1.0},
    )
    report = fields.k_phi_multiscale_safety(nx.Graph(), alpha_hint=hint)
    assert report["available"]
    assert report["fit_acceptable"]
    assert report["violations"] == [1]
    assert not report["safe"]
    assert report["assessment_status"] == "warning"


@pytest.mark.parametrize("variances", [{}, {1: math.nan}, {1: math.inf}, {1: -1.0}])
def test_unavailable_variances_do_not_pass_the_cut(monkeypatch, variances):
    monkeypatch.setattr(
        fields, "compute_k_phi_multiscale_variance", lambda graph: variances
    )
    report = fields.k_phi_multiscale_safety(nx.Graph())
    assert not report["available"]
    assert not report["safe"]
    assert report["assessment_status"] == "unavailable"


def test_no_reference_hint_is_supported_by_actual_curvature_telemetry():
    graph = nx.cycle_graph(5)
    nx.set_node_attributes(graph, {node: node * 0.2 for node in graph}, "phase")
    with_hint = warn_phase_curvature_telemetry(graph)
    without_hint = warn_phase_curvature_telemetry(graph, alpha_hint=None)
    assert without_hint == with_hint
