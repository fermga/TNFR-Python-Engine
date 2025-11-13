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


def test_u6_drift_safe_under_synthetic_sequence():
    G = build_ws_graph_with_seed(n=40, k=4, p=0.1, seed=123)
    phi_before = compute_structural_potential(G)
    apply_synthetic_activation_sequence(G, alpha=0.25, dnfr_factor=0.9)
    phi_after = compute_structural_potential(G)

    ok, drift, msg = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=2.0, strict=False
    )
    assert ok is True
    assert drift >= 0.0
    assert drift < 2.0


def test_u6_drift_violation_with_large_dnfr_increase():
    G = build_ws_graph_with_seed(n=40, k=4, p=0.1, seed=456)
    phi_before = compute_structural_potential(G)
    # Exaggerate ΔNFR to push drift upwards
    apply_synthetic_activation_sequence(G, alpha=0.0, dnfr_factor=4.0)
    phi_after = compute_structural_potential(G)

    ok, drift, msg = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=2.0, strict=False
    )
    assert ok is False
    assert drift >= 0.0

    with pytest.raises(StructuralPotentialConfinementError):
        validate_structural_potential_confinement(
            G, phi_before, phi_after, threshold=2.0, strict=True
        )
