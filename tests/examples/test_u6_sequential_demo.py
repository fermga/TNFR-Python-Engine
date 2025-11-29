from __future__ import annotations

import pytest

from tnfr.physics.fields import compute_structural_potential
from tnfr.operators.grammar import (
    validate_structural_potential_confinement,
    StructuralPotentialConfinementError,
)
from tnfr.examples_utils import (
    build_ws_graph_with_seed,
    apply_synthetic_activation_sequence,
)
from tnfr.config.defaults_core import STRUCTURAL_ESCAPE_THRESHOLD


def test_u6_drift_safe_under_synthetic_sequence():
    G = build_ws_graph_with_seed(n=40, k=4, p=BENCH_SMALL_NETWORK_P, seed=123)  # Canonical small network p
    phi_before = compute_structural_potential(G)
    apply_synthetic_activation_sequence(G, alpha=BENCH_ALPHA_ACTIVATION, dnfr_factor=BENCH_DNFR_FACTOR_NORMAL)  # Canonical activation
    phi_after = compute_structural_potential(G)

    ok, drift, msg = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
    )
    assert ok is True
    assert drift >= 0.0
    assert drift < STRUCTURAL_ESCAPE_THRESHOLD


def test_u6_drift_violation_with_large_dnfr_increase():
    G = build_ws_graph_with_seed(n=40, k=4, p=BENCH_SMALL_NETWORK_P, seed=456)  # Canonical small network p
    phi_before = compute_structural_potential(G)
    # Exaggerate ΔNFR to push drift upwards
    apply_synthetic_activation_sequence(G, alpha=0.0, dnfr_factor=BENCH_DNFR_FACTOR_HIGH)  # Canonical high activation
    phi_after = compute_structural_potential(G)

    ok, drift, msg = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=False
    )
    assert ok is False
    assert drift >= 0.0

    with pytest.raises(StructuralPotentialConfinementError):
        validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=STRUCTURAL_ESCAPE_THRESHOLD, strict=True
        )
