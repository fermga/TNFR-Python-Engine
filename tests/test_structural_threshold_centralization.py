"""Structural-field monitoring policies use one canonical source."""

from __future__ import annotations

import inspect
import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.backend_config import TNFRConfig
from tnfr.constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    PHI_S_VON_KOCH_THRESHOLD,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
    XI_C_CRITICAL_RATIO,
    XI_C_WATCH_RATIO,
)
from tnfr.mathematics.unified_numerical import CONSTANTS as NUMERICAL_CONSTANTS
from tnfr.physics.interactions import em_like, strong_like, weak_like
from tnfr.telemetry.constants import (
    PHASE_CURVATURE_ABS_THRESHOLD,
    PHASE_GRADIENT_THRESHOLD,
    PHI_S_CLASSICAL_THRESHOLD,
    STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
)
from tnfr.validation.aggregator import run_structural_validation
from tnfr.validation.health import compute_structural_health


def test_public_threshold_views_alias_canonical_policies() -> None:
    config = TNFRConfig()

    assert PHASE_GRADIENT_THRESHOLD == pytest.approx(GRAD_PHI_CANONICAL_THRESHOLD)
    assert PHASE_CURVATURE_ABS_THRESHOLD == pytest.approx(K_PHI_CANONICAL_THRESHOLD)
    assert PHI_S_CLASSICAL_THRESHOLD == pytest.approx(PHI_S_VON_KOCH_THRESHOLD)
    assert STRUCTURAL_POTENTIAL_DELTA_THRESHOLD == pytest.approx(
        U6_STRUCTURAL_POTENTIAL_LIMIT
    )
    assert config.structural_potential_threshold == pytest.approx(
        PHI_S_VON_KOCH_THRESHOLD
    )
    assert config.phase_gradient_threshold == pytest.approx(
        GRAD_PHI_CANONICAL_THRESHOLD
    )
    assert config.phase_curvature_threshold == pytest.approx(K_PHI_CANONICAL_THRESHOLD)
    assert config.coherence_length_critical == pytest.approx(XI_C_CRITICAL_RATIO)
    assert NUMERICAL_CONSTANTS.STRUCTURAL_POTENTIAL_ESCAPE_THRESHOLD == pytest.approx(
        U6_STRUCTURAL_POTENTIAL_LIMIT
    )
    assert NUMERICAL_CONSTANTS.PHASE_GRADIENT_STABILITY_THRESHOLD == pytest.approx(
        GRAD_PHI_CANONICAL_THRESHOLD
    )
    assert NUMERICAL_CONSTANTS.PHASE_CURVATURE_CONFINEMENT_THRESHOLD == pytest.approx(
        K_PHI_CANONICAL_THRESHOLD
    )
    assert NUMERICAL_CONSTANTS.COHERENCE_LENGTH_CRITICAL_RATIO == pytest.approx(
        XI_C_CRITICAL_RATIO
    )


def test_validation_defaults_alias_canonical_policies() -> None:
    parameters = inspect.signature(run_structural_validation).parameters

    assert parameters["max_delta_phi_s"].default == pytest.approx(
        U6_STRUCTURAL_POTENTIAL_LIMIT
    )
    assert parameters["max_phase_gradient"].default == pytest.approx(
        GRAD_PHI_CANONICAL_THRESHOLD
    )
    assert parameters["k_phi_flag_threshold"].default == pytest.approx(
        K_PHI_CANONICAL_THRESHOLD
    )
    assert parameters["xi_c_critical_multiplier"].default == pytest.approx(
        XI_C_CRITICAL_RATIO
    )
    assert parameters["xi_c_watch_multiplier"].default == pytest.approx(
        XI_C_WATCH_RATIO
    )


def test_interaction_defaults_alias_canonical_policies() -> None:
    assert inspect.signature(em_like).parameters[
        "grad_threshold"
    ].default == pytest.approx(GRAD_PHI_CANONICAL_THRESHOLD)
    assert inspect.signature(weak_like).parameters[
        "grad_threshold"
    ].default == pytest.approx(GRAD_PHI_CANONICAL_THRESHOLD)
    assert inspect.signature(strong_like).parameters[
        "curvature_hotspot_threshold"
    ].default == pytest.approx(K_PHI_CANONICAL_THRESHOLD)


def test_validation_uses_pi_over_sixteen_instead_of_legacy_point_38() -> None:
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, {0: 0.0, 1: 0.25}, "theta")
    nx.set_node_attributes(graph, 0.0, "delta_nfr")

    canonical = run_structural_validation(graph)
    legacy_override = run_structural_validation(graph, max_phase_gradient=0.38)

    assert canonical.field_metrics["max_phase_gradient"] == pytest.approx(0.25)
    assert canonical.thresholds_exceeded["phase_gradient_max"] is True
    assert canonical.risk_level == "elevated"
    assert legacy_override.thresholds_exceeded["phase_gradient_max"] is False


def _u6_graph() -> nx.Graph:
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, 0.0, "theta")
    nx.set_node_attributes(graph, 0.0, "delta_nfr")
    return graph


@pytest.mark.parametrize(
    "baseline",
    [
        {},
        {0: 0.0},
        {2: 0.0, 3: 0.0},
        {0: 0.0, 1: 0.0, 2: 0.0},
        {0: float("nan"), 1: 0.0},
        {0: float("inf"), 1: 0.0},
        {0: True, 1: 0.0},
        {0: None, 1: 0.0},
    ],
)
def test_u6_invalid_baseline_is_unavailable_instead_of_zero_drift(baseline):
    report = run_structural_validation(
        _u6_graph(), baseline_structural_potential=baseline
    )

    assert report.status == "valid"  # Grammar status stays separate.
    assert report.field_metrics["u6_status"] == "unavailable"
    assert report.field_metrics["u6_reason"]
    assert report.field_metrics["delta_phi_s"] is None
    assert "delta_phi_s" not in report.thresholds_exceeded
    assert report.risk_level in {"elevated", "critical"}
    assert any("U6 drift unavailable" in note for note in report.notes)


def test_u6_omitted_baseline_is_not_requested():
    report = run_structural_validation(_u6_graph())
    assert report.field_metrics["u6_status"] == "not_requested"
    assert report.field_metrics["u6_reason"] is None
    assert report.field_metrics["delta_phi_s"] is None
    assert "delta_phi_s" not in report.thresholds_exceeded


@pytest.mark.parametrize("threshold, exceeded", [(1.0, False), (0.75, True)])
def test_u6_uses_complete_nodewise_mean_and_strict_shared_threshold(
    threshold, exceeded
):
    # Actual field is zero; the two absolute differences are 1/2 and 1,
    # hence mean 3/4, not max 1 or absolute mean 1/4.
    baseline = {0: 0.5, 1: -1.0}
    before = deepcopy(baseline)
    report = run_structural_validation(
        _u6_graph(),
        baseline_structural_potential=baseline,
        max_delta_phi_s=threshold,
    )
    assert report.field_metrics["delta_phi_s"] == 0.75
    assert report.field_metrics["u6_status"] == "evaluated"
    assert report.field_metrics["u6_reason"] is None
    assert report.thresholds_exceeded["delta_phi_s"] is exceeded
    assert baseline == before


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), True])
def test_u6_invalid_threshold_is_explicitly_unavailable(threshold):
    report = run_structural_validation(
        _u6_graph(),
        baseline_structural_potential={0: 0.0, 1: 0.0},
        max_delta_phi_s=threshold,
    )
    assert report.field_metrics["u6_status"] == "unavailable"
    assert report.field_metrics["delta_phi_s"] is None
    assert "delta_phi_s" not in report.thresholds_exceeded


@pytest.mark.parametrize(
    "parameter,flag",
    [
        ("max_phase_gradient", "phase_gradient_max"),
        ("k_phi_flag_threshold", "k_phi_flag"),
        ("xi_c_critical_multiplier", "xi_c_critical"),
        ("xi_c_watch_multiplier", "xi_c_watch"),
    ],
)
@pytest.mark.parametrize("invalid", [math.nan, True, "1.0", -Fraction(1, 10**400)])
def test_invalid_monitoring_threshold_is_unavailable_not_a_passing_flag(
    parameter, flag, invalid
):
    graph = _u6_graph()
    nodes_before = deepcopy(dict(graph.nodes(data=True)))
    report = run_structural_validation(graph, **{parameter: invalid})
    assert report.field_metrics["threshold_status"][flag] == "unavailable"
    assert report.field_metrics["threshold_reasons"][flag]
    assert flag not in report.thresholds_exceeded
    assert report.risk_level in {"elevated", "critical"}
    assert dict(graph.nodes(data=True)) == nodes_before


def test_empty_graph_cannot_supply_low_risk_field_evidence():
    graph = nx.Graph()
    report = run_structural_validation(graph)
    assert report.status == "valid"  # No grammar was requested.
    assert report.risk_level == "elevated"
    assert report.thresholds_exceeded == {}
    assert not any(report.field_metrics["field_availability"].values())
    assert report.field_metrics["max_phase_gradient"] is None
    assert report.field_metrics["max_k_phi"] is None
    assert report.field_metrics["xi_c"] is None
    assert graph.graph == {}


def test_undefined_curvature_keeps_independent_gradient_observation():
    graph = nx.star_graph(4)
    nx.set_node_attributes(
        graph, dict(enumerate((0.3, 0.0, 0.0, math.pi, -math.pi))), "theta"
    )
    nx.set_node_attributes(graph, 0.0, "delta_nfr")
    report = run_structural_validation(graph, max_phase_gradient=4.0)
    fields = report.field_metrics
    assert fields["field_availability"]["phase_gradient"] is True
    assert fields["field_availability"]["phase_curvature"] is False
    assert fields["phase_curvature"][0] is None
    assert fields["max_k_phi"] is None
    assert fields["max_phase_gradient"] > 0
    assert "k_phi_flag" not in report.thresholds_exceeded
    assert report.thresholds_exceeded["phase_gradient_max"] is False
    assert report.risk_level in {"elevated", "critical"}


def test_pressure_failure_does_not_erase_phase_or_certify_safety():
    graph = _u6_graph()
    graph.nodes[0]["delta_nfr"] = math.nan
    report = run_structural_validation(graph)
    assert report.field_metrics["field_availability"] == {
        "phi_s": False,
        "phase_gradient": True,
        "phase_curvature": True,
        "xi_c": False,
    }
    assert report.field_metrics["mean_structural_potential"] is None
    assert report.field_metrics["field_errors"]["phi_s"]
    assert report.risk_level == "elevated"


def test_disconnected_support_does_not_use_one_component_as_global_geometry():
    graph = nx.disjoint_union(nx.path_graph(2), nx.path_graph(3))
    nx.set_node_attributes(graph, 0.0, "theta")
    nx.set_node_attributes(graph, 0.0, "delta_nfr")
    report = run_structural_validation(graph)
    fields = report.field_metrics
    assert fields["system_diameter"] is None
    assert fields["mean_node_distance"] is None
    assert "xi_c_critical" not in report.thresholds_exceeded
    assert "xi_c_watch" not in report.thresholds_exceeded
    assert report.risk_level == "elevated"


def test_field_aggregation_uses_finite_shared_mean_and_retains_xi_provenance():
    graph = _u6_graph()
    nx.set_node_attributes(graph, 1e308, "delta_nfr")
    report = run_structural_validation(graph)
    assert report.field_metrics["phi_s"] == {0: 1e308, 1: 1e308}
    assert report.field_metrics["mean_structural_potential"] == 1e308
    assert report.field_metrics["xi_c_provenance"]["method"]


def test_zero_warning_cut_remains_a_valid_explicit_policy():
    report = run_structural_validation(_u6_graph(), max_phase_gradient=0.0)
    assert report.field_metrics["threshold_status"]["phase_gradient_max"] == "evaluated"
    assert report.thresholds_exceeded["phase_gradient_max"] is True


def test_missing_curvature_does_not_hide_independent_critical_drift():
    graph = nx.star_graph(4)
    nx.set_node_attributes(
        graph, dict(enumerate((0.3, 0.0, 0.0, math.pi, -math.pi))), "theta"
    )
    nx.set_node_attributes(graph, 0.0, "delta_nfr")
    report = compute_structural_health(
        graph, baseline_phi_s=dict.fromkeys(graph, 2.0), max_delta_phi_s=Fraction(3, 2)
    )
    assert report["field_availability"]["phase_curvature"] is False
    assert report["threshold_status"]["k_phi_flag"] == "unavailable"
    assert report["threshold_reasons"]["k_phi_flag"]
    assert report["thresholds_exceeded"]["delta_phi_s"] is True
    assert report["risk_level"] == "critical"


def test_admitted_fraction_cuts_and_large_comparison_scales_remain_evaluable():
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, dict(enumerate((0.0, 0.25, 0.0))), "theta")
    nx.set_node_attributes(graph, 0.0, "delta_nfr")
    report = run_structural_validation(
        graph,
        max_phase_gradient=Fraction(1, 8),
        k_phi_flag_threshold=Fraction(1, 8),
        xi_c_critical_multiplier=1e308,
        xi_c_watch_multiplier=1e308,
    )
    assert report.thresholds_exceeded["phase_gradient_max"] is True
    assert report.thresholds_exceeded["k_phi_flag"] is True
    assert report.thresholds_exceeded["xi_c_critical"] is False
    assert report.field_metrics["threshold_status"]["xi_c_critical"] == "evaluated"
