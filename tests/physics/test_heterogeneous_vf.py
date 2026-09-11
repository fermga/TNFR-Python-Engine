r"""Tests for heterogeneous nodal frequency.

The scalar clock-change theorem (N04) holds for a common ν_f but **fails** for a
heterogeneous D_{ν_f}(t): the generators D_{ν_f}(t)·L no longer commute, so
x(t) ≠ e^{-s(t)L} x₀. A fixed positive D_{ν_f} is still stable; uniform
time-varying stability stays open.
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.physics import heterogeneous_vf as hv
from tnfr.physics.directed_diffusion import directed_cayley_adjacency
from tnfr.physics.heterogeneous_vf import (
    certify_convex_hull_generator,
    certify_dirichlet_convergence,
    compare_fixed_stationary_measures,
    certify_heterogeneous_vf,
    finite_schedule_readout,
    fixed_generator_abscissa,
    generator_commutator_norm,
    heterogeneous_schedule,
    heterogeneous_generator,
    heterogeneous_stationary_distribution,
    piecewise_propagator,
    scalar_schedule,
    scalar_time_ansatz_residual,
    time_dependent_dirichlet_balance,
    within_initial_convex_hull,
)

W_SC = np.array([[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]],
                dtype=float)
X4 = np.array([1.0, -1.0, 0.5, -0.5]) / np.linalg.norm([1.0, -1.0, 0.5, -0.5])
T = np.linspace(0.0, 10.0, 600)


# --------------------------------------------------------------------------- #
# Commutators: scalar commutes, heterogeneous does not
# --------------------------------------------------------------------------- #
def test_scalar_frequency_generators_commute():
    # any two scalar multiples of L commute
    assert generator_commutator_norm(2.0 * np.ones(4), 3.0 * np.ones(4),
                                     W_SC) < 1e-9


def test_heterogeneous_frequency_generators_do_not_commute():
    assert generator_commutator_norm(np.array([0.5, 1.5, 1.0, 2.0]),
                                     np.array([2.0, 0.5, 1.5, 1.0]),
                                     W_SC) > 1e-3


# --------------------------------------------------------------------------- #
# The scalar-time theorem does not extend to heterogeneous capacity
# --------------------------------------------------------------------------- #
def test_heterogeneous_vf_does_not_use_scalar_time_theorem():
    common = scalar_schedule(4)
    hetero = heterogeneous_schedule(4)
    res_common = scalar_time_ansatz_residual(W_SC, X4, common, T)
    res_hetero = scalar_time_ansatz_residual(W_SC, X4, hetero, T)
    assert res_common < 1e-3            # common ν_f: clock change holds
    assert res_hetero > 1e-2            # heterogeneous: ansatz fails
    assert res_hetero > 10 * res_common


def test_common_schedule_is_an_exact_clock_change():
    res = scalar_time_ansatz_residual(
        directed_cayley_adjacency(7, {1, 2}),
        np.ones(7) / np.sqrt(7) * np.array([1, -1, 1, -1, 1, -1, 1]),
        scalar_schedule(7), T)
    assert res < 1e-3


# --------------------------------------------------------------------------- #
# Fixed-D stability (frozen generator)
# --------------------------------------------------------------------------- #
def test_fixed_heterogeneous_generator_is_stable():
    # a frozen positive D keeps -D L stable (consensus preserved, rest decays)
    for vf in ([0.5, 1.5, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [0.3, 2.0, 0.7, 1.1]):
        assert fixed_generator_abscissa(np.array(vf), W_SC) <= 1e-6


def test_fixed_heterogeneous_stationary_measure_has_zero_left_residual():
    capacity = np.array([0.5, 1.5, 1.0, 2.0])
    eta = heterogeneous_stationary_distribution(capacity, W_SC)
    generator = heterogeneous_generator(capacity, W_SC)
    assert eta.sum() == pytest.approx(1.0)
    assert np.all(eta > 0.0)
    assert np.linalg.norm(eta @ generator) < 1e-12


def test_fixed_measure_comparison_exposes_projector_mismatch():
    capacity = np.array([0.5, 1.5, 1.0, 2.0])
    comparison = compare_fixed_stationary_measures(capacity, W_SC)
    assert comparison.mismatch_norm > 0.0
    assert comparison.left_residual < 1e-12
    assert comparison.stationary_distribution != (
        comparison.capacity_weighted_distribution
    )


def test_fixed_heterogeneous_stationary_measure_rejects_zero_capacity():
    with pytest.raises(ValueError):
        heterogeneous_stationary_distribution([1.0, 0.0, 1.0, 1.0], W_SC)


def test_piecewise_nonnegative_diffusion_preserves_initial_convex_hull():
    first = np.array([0.5, 1.5, 1.0, 2.0])
    second = np.array([1.2, 0.7, 1.7, 0.6])
    x0 = np.array([-2.0, 0.5, 1.0, 3.0])
    forward = piecewise_propagator(W_SC, [(0.4, first), (0.6, second)])
    reverse = piecewise_propagator(W_SC, [(0.6, second), (0.4, first)])
    assert within_initial_convex_hull(forward, x0)
    assert within_initial_convex_hull(reverse, x0)
    assert not np.allclose(forward, reverse)


def test_metzler_certificate_proves_convex_hull_with_frozen_node():
    certificate = certify_convex_hull_generator(
        np.array([0.0, 1.5, 1.0, 2.0]), W_SC
    )
    assert certificate.is_metzler
    assert certificate.zero_row_sums
    assert certificate.preserves_convex_hull


def test_all_zero_capacity_freezes_state_without_consensus_claim():
    zero = np.zeros(4)
    generator = heterogeneous_generator(zero, W_SC)
    certificate = certify_convex_hull_generator(zero, W_SC)
    assert np.array_equal(generator, np.zeros((4, 4)))
    assert certificate.preserves_convex_hull
    assert np.array_equal(matrix := piecewise_propagator(
        W_SC, [(2.0, zero)]
    ), np.eye(4))
    assert np.array_equal(matrix @ X4, X4)


def test_mixed_zero_capacity_preserves_hull_but_not_uniform_consensus_bound():
    capacity = np.array([0.0, 1.5, 1.0, 2.0])
    generator_certificate = certify_convex_hull_generator(capacity, W_SC)
    convergence = certify_dirichlet_convergence(
        W_SC + W_SC.T,
        X4,
        capacity_lower_bound=0.0,
        capacity_upper_bound=2.0,
        integrated_min_mobility=0.0,
        tail_integral_diverges=False,
    )
    assert generator_certificate.preserves_convex_hull
    assert not convergence.asymptotic_consensus_guaranteed
    assert convergence.total_variation_bound is None


def test_rapid_switching_approaches_ordered_average_generator():
    from tnfr.physics.spectral_projectors import matrix_exponential

    first = np.array([0.5, 1.5, 1.0, 2.0])
    second = np.array([1.2, 0.7, 1.7, 0.6])
    averaged = 0.5 * (
        heterogeneous_generator(first, W_SC)
        + heterogeneous_generator(second, W_SC)
    )
    reference = matrix_exponential(-averaged)
    errors = []
    for switches in (2, 8, 32):
        duration = 1.0 / switches
        segments = [
            (duration, first if index % 2 == 0 else second)
            for index in range(switches)
        ]
        product = piecewise_propagator(W_SC, segments)
        errors.append(float(np.linalg.norm(product - reference, 2)))
    assert errors[2] < errors[1] < errors[0]


def test_time_dependent_mobility_has_exact_dirichlet_sign_identity():
    adjacency = np.array(
        [[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    state = np.array([1.0, -0.5, 2.0])
    for capacity in (
        np.array([0.0, 1.5, 0.2]),
        np.array([2.0, 0.4, 3.0]),
    ):
        balance = time_dependent_dirichlet_balance(
            adjacency, state, capacity
        )
        assert balance.identity_residual < 1e-12
        assert balance.energy_rate == pytest.approx(-balance.dissipation)
        assert balance.energy_rate <= 0.0
        assert balance.topology_term == 0.0


def test_dirichlet_hierarchy_separates_consensus_from_finite_clock():
    adjacency = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    state = np.array([0.0, 2.0, -1.0])
    convergent = certify_dirichlet_convergence(
        adjacency,
        state,
        capacity_lower_bound=0.4,
        capacity_upper_bound=2.0,
        integrated_min_mobility=3.0,
        tail_integral_diverges=True,
    )
    finite_clock = certify_dirichlet_convergence(
        adjacency,
        state,
        capacity_lower_bound=0.0,
        capacity_upper_bound=2.0,
        integrated_min_mobility=0.5,
        tail_integral_diverges=False,
    )
    assert convergent.finite_horizon_energy_bound < convergent.initial_energy
    assert convergent.asymptotic_consensus_guaranteed
    assert "CONDITIONAL" in convergent.total_variation_status
    assert convergent.total_variation_bound is not None
    assert convergent.total_variation_bound > 0.0
    assert not finite_clock.asymptotic_consensus_guaranteed
    assert "NOT_IMPLIED" in finite_clock.total_variation_status
    assert finite_clock.total_variation_bound is None


def test_dirichlet_total_variation_bound_controls_refined_trajectory():
    from tnfr.physics.spectral_projectors import matrix_exponential

    adjacency = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    state = np.array([0.0, 2.0, -1.0])
    capacity = np.array([0.5, 1.5, 1.0])
    certificate = certify_dirichlet_convergence(
        adjacency,
        state,
        capacity_lower_bound=float(np.min(capacity)),
        capacity_upper_bound=float(np.max(capacity)),
        integrated_min_mobility=5.0,
        tail_integral_diverges=True,
    )
    generator = heterogeneous_generator(capacity, adjacency)
    trajectory = np.array(
        [matrix_exponential(-time * generator) @ state
         for time in np.linspace(0.0, 20.0, 2001)]
    )
    sampled_variation = float(
        np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    )
    assert certificate.total_variation_bound is not None
    assert sampled_variation <= certificate.total_variation_bound


def test_finite_schedule_reports_exact_kernel_u6_without_interval_claim():
    import networkx as nx

    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 1.0)])
    readout = finite_schedule_readout(
        graph,
        [0.0, 2.0, -1.0],
        [(0.2, [0.5, 1.5, 1.0]), (0.3, [1.2, 0.7, 1.7])],
    )
    assert len(readout.endpoint_states) == 3
    assert len(readout.euclidean_deviations) == 3
    assert len(readout.instantaneous_weighted_deviations) == 3
    assert readout.metric_capacity_context == (
        "first_segment", "preceding_segment_1", "preceding_segment_2"
    )
    assert readout.mean_absolute_u6_drifts[0] == 0.0
    assert readout.u6_kernel.startswith("canonical_directed_weighted")
    assert readout.time_coverage == "segment_endpoints_only"
    assert "NOT_USED_AS_ENERGY" in readout.metric_derivative_status


# --------------------------------------------------------------------------- #
# Certificate + exports
# --------------------------------------------------------------------------- #
def test_certificate_marks_theorem_boundary():
    cert = certify_heterogeneous_vf(W_SC, X4, T)
    assert cert.commutator_scalar < 1e-9
    assert cert.commutator_heterogeneous > 1e-3
    assert cert.scalar_time_residual_common < 1e-3
    assert cert.scalar_time_residual_heterogeneous > 1e-2
    assert cert.scalar_time_theorem_extends is False
    assert cert.fixed_generator_stable is True
    assert "OPEN" in cert.claim_status
    assert "unmodified" in cert.claim_status


def test_module_exports_complete():
    expected = {
        "heterogeneous_generator", "generator_commutator_norm",
        "scalar_schedule", "heterogeneous_schedule", "structural_time_mean",
        "scalar_time_ansatz_residual", "fixed_generator_abscissa",
        "nonconsensus_transient_gain", "HeterogeneousVfCertificate",
        "certify_heterogeneous_vf",
    }
    assert expected <= set(hv.__all__)
