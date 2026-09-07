"""Structural-field monitoring policies use one canonical source."""

from __future__ import annotations

import inspect

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


def test_public_threshold_views_alias_canonical_policies() -> None:
    config = TNFRConfig()

    assert PHASE_GRADIENT_THRESHOLD == pytest.approx(
        GRAD_PHI_CANONICAL_THRESHOLD
    )
    assert PHASE_CURVATURE_ABS_THRESHOLD == pytest.approx(
        K_PHI_CANONICAL_THRESHOLD
    )
    assert PHI_S_CLASSICAL_THRESHOLD == pytest.approx(
        PHI_S_VON_KOCH_THRESHOLD
    )
    assert STRUCTURAL_POTENTIAL_DELTA_THRESHOLD == pytest.approx(
        U6_STRUCTURAL_POTENTIAL_LIMIT
    )
    assert config.structural_potential_threshold == pytest.approx(
        PHI_S_VON_KOCH_THRESHOLD
    )
    assert config.phase_gradient_threshold == pytest.approx(
        GRAD_PHI_CANONICAL_THRESHOLD
    )
    assert config.phase_curvature_threshold == pytest.approx(
        K_PHI_CANONICAL_THRESHOLD
    )
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
