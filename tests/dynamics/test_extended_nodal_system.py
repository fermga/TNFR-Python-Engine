import math

import numpy as np

from tnfr.dynamics.canonical import compute_extended_nodal_system


def test_classical_limit_and_conservation():
    """Classical derivative equals νf·ΔNFR and ∂ΔNFR/∂t=0 when ∇·J=0."""
    nu_f = 1.5
    delta_nfr = 0.4
    theta = 0.5

    # No flux transport or divergence
    result = compute_extended_nodal_system(
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        theta=theta,
        j_phi=0.0,
        j_dnfr_divergence=0.0,
        coupling_strength=1.0,
        validate_units=True,
    )

    # Classical derivative must match exactly
    assert np.isclose(
        result.classical_derivative, nu_f * delta_nfr
    ), "∂EPI/∂t must equal νf·ΔNFR in the classical limit"

    # With zero divergence, conservation gives no ΔNFR change
    assert np.isclose(
        result.dnfr_derivative, 0.0
    ), "∂ΔNFR/∂t must be 0 when ∇·J_ΔNFR = 0"


def test_phase_transport_monotonic_in_jphi():
    """∂θ/∂t increases linearly with J_φ for fixed params and κ>0."""
    nu_f = 1.0
    delta_nfr = 0.2
    theta = 0.0
    kappa = 0.8

    # Evaluate for negative, zero, positive J_φ
    res_neg = compute_extended_nodal_system(
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        theta=theta,
        j_phi=-0.2,
        j_dnfr_divergence=0.0,
        coupling_strength=kappa,
    )
    res_zero = compute_extended_nodal_system(
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        theta=theta,
        j_phi=0.0,
        j_dnfr_divergence=0.0,
        coupling_strength=kappa,
    )
    res_pos = compute_extended_nodal_system(
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        theta=theta,
        j_phi=0.2,
        j_dnfr_divergence=0.0,
        coupling_strength=kappa,
    )

    # Transport contribution is linear in J_φ with positive coefficient (γ·κ)
    assert (
        res_neg.phase_derivative
        < res_zero.phase_derivative
        < res_pos.phase_derivative
    ), (
        "Phase derivative must be monotonic increasing in J_φ for κ>0"
    )


def test_dnfr_conservation_sign_convention():
    """Sign of ∂ΔNFR/∂t follows conservation: -∇·J minus decay."""
    nu_f = 1.0
    delta_nfr = 0.1
    theta = math.pi / 4

    # Positive divergence → outflow → ΔNFR decreases (negative derivative)
    res_pos_div = compute_extended_nodal_system(
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        theta=theta,
        j_phi=0.0,
        j_dnfr_divergence=+0.3,
        coupling_strength=1.0,
    )
    assert (
        res_pos_div.dnfr_derivative < 0.0
    ), "Positive ∇·J should reduce ΔNFR (negative ∂ΔNFR/∂t)"

    # Negative divergence → inflow → ΔNFR increases (positive derivative)
    res_neg_div = compute_extended_nodal_system(
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        theta=theta,
        j_phi=0.0,
        j_dnfr_divergence=-0.3,
        coupling_strength=1.0,
    )
    assert (
        res_neg_div.dnfr_derivative > 0.0
    ), "Negative ∇·J should increase ΔNFR (positive ∂ΔNFR/∂t)"
