"""Tests for the structural-diffusion characterization of the nodal equation.

Verifies that the EPI channel of ΔNFR is the random-walk graph Laplacian
(the discrete diffusion operator), so the nodal equation
∂EPI/∂t = νf·ΔNFR is structurally a diffusion equation.
"""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import (
    DiscreteModeCertificate,
    OverdampedProjectionCertificate,
    OverdampedRegimeCertificate,
    RandomWalkCertificate,
    StructuralDiffusionCertificate,
    StructuralFlowCertificate,
    StructuralStabilityCertificate,
    UndampedLimitCertificate,
    commute_time,
    compute_emergent_pulse,
    compute_nodal_pulse,
    current_divergence,
    damped_wave_rates,
    degree_weighted_total,
    dispersion_relation,
    effective_resistance,
    fiedler_partition,
    instability_threshold,
    nodal_domain_count,
    random_walk_matrix,
    relaxation_spectrum,
    stationary_distribution,
    structural_current,
    structural_diffusion_operator,
    structural_diffusivity,
    structural_eigenmodes,
    structural_field,
    verify_discrete_modes,
    verify_overdamped_projection,
    verify_overdamped_regime,
    verify_structural_diffusion,
    verify_structural_flow,
    verify_structural_random_walk,
    verify_structural_stability,
    verify_undamped_limit,
)


def _canonical_graph(n: int = 60, seed: int = 11) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 6, 0.2, seed=seed)
    for node in G.nodes():
        G.nodes[node]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        G.nodes[node]["EPI"] = rng.uniform(-0.4, 0.4)
        G.nodes[node]["nu_f"] = rng.uniform(0.5, 1.5)
    default_compute_delta_nfr(G)
    return G


class TestDiffusionOperator:
    """The structural diffusion operator is the random-walk Laplacian."""

    def test_operator_shape(self) -> None:
        G = _canonical_graph(40)
        nodes, lap = structural_diffusion_operator(G)
        assert len(nodes) == 40
        assert lap.shape == (40, 40)

    def test_rows_sum_to_zero(self) -> None:
        # L_rw = I − D⁻¹W has zero row sums (diffusion conserves locally).
        G = _canonical_graph(40)
        _, lap = structural_diffusion_operator(G)
        assert np.allclose(lap.sum(axis=1), 0.0)

    def test_diagonal_is_one(self) -> None:
        G = _canonical_graph(30)
        _, lap = structural_diffusion_operator(G)
        # connected (non-isolated) nodes have diagonal 1.0
        assert np.allclose(np.diag(lap), 1.0)


class TestNodalEquationIsDiffusion:
    """The canonical ΔNFR EPI channel equals −L_rw·EPI (machine precision)."""

    def test_dnfr_epi_channel_is_graph_laplacian(self) -> None:
        G = _canonical_graph(50)
        nodes, lap = structural_diffusion_operator(G)
        epi = structural_field(G, nodes)
        # isolate the EPI channel on a clean structural replica
        from tnfr.constants.aliases import ALIAS_VF

        g2 = nx.Graph()
        for node in nodes:
            g2.add_node(
                node,
                EPI=float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)),
                theta=float(G.nodes[node].get("theta", 0.0)),
                nu_f=float(get_attr(G.nodes[node], ALIAS_VF, 0.0)),
            )
        g2.add_edges_from(G.edges())
        g2.graph["DNFR_WEIGHTS"] = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
        default_compute_delta_nfr(g2)
        dnfr = np.array([float(get_attr(g2.nodes[n], ALIAS_DNFR, 0.0)) for n in nodes])
        assert np.allclose(dnfr, -(lap @ epi), atol=1e-12)

    def test_certificate_confirms_laplacian(self) -> None:
        G = _canonical_graph(60)
        cert = verify_structural_diffusion(G)
        assert cert.dnfr_is_graph_laplacian
        assert cert.max_laplacian_residual < 1e-12


class TestDiffusionConservation:
    """Diffusion conserves the degree-weighted structural total."""

    def test_degree_weighted_total_conserved(self) -> None:
        G = _canonical_graph(60)
        cert = verify_structural_diffusion(G)
        assert cert.degree_weighted_conserved
        assert cert.max_conservation_drift < 1e-9

    def test_degree_weighted_total_value(self) -> None:
        import pytest

        G = _canonical_graph(40)
        # the helper matches a direct degree-weighted sum
        nodes, _ = structural_diffusion_operator(G)
        epi = structural_field(G, nodes)
        deg = np.array([G.degree(n) for n in nodes], dtype=float)
        assert degree_weighted_total(G) == pytest.approx(float(deg @ epi))


class TestDiffusionRelaxation:
    """The field relaxes to a uniform diffusive equilibrium."""

    def test_relaxes_to_uniform(self) -> None:
        G = _canonical_graph(60)
        cert = verify_structural_diffusion(G)
        assert cert.relaxes_to_uniform
        assert cert.final_field_std < 1e-3

    def test_spectrum_has_zero_mode(self) -> None:
        # λ₁ = 0 (the conserved uniform diffusion mode).
        G = _canonical_graph(50)
        spec = relaxation_spectrum(G)
        assert abs(float(spec[0])) < 1e-9
        assert float(spec[1]) > 0.0  # spectral gap positive

    def test_diffusivity_is_mean_vf(self) -> None:
        import pytest

        from tnfr.constants.aliases import ALIAS_VF

        G = _canonical_graph(40)
        nodes, _ = structural_diffusion_operator(G)
        vf = [float(get_attr(G.nodes[n], ALIAS_VF, 0.0)) for n in nodes]
        assert structural_diffusivity(G) == pytest.approx(float(np.mean(vf)), rel=1e-6)


class TestDiffusionCertificate:
    """The certificate aggregates the diffusion verdict."""

    def test_certificate_valid(self) -> None:
        G = _canonical_graph(60)
        cert = verify_structural_diffusion(G)
        assert isinstance(cert, StructuralDiffusionCertificate)
        assert cert.is_valid_diffusion
        assert "VALID" in cert.summary()
        assert cert.diffusivity > 0.0
        assert cert.spectral_gap > 0.0

    def test_graph_not_mutated(self) -> None:
        # verify runs the ΔNFR check on a replica; caller's graph is untouched.
        G = _canonical_graph(40)
        before = {n: float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in G.nodes()}
        weights_before = G.graph.get("DNFR_WEIGHTS")
        verify_structural_diffusion(G)
        after = {n: float(get_attr(G.nodes[n], ALIAS_DNFR, 0.0)) for n in G.nodes()}
        assert before == after
        assert G.graph.get("DNFR_WEIGHTS") == weights_before


class TestOverdampedRegime:
    """The bare nodal equation is the first-order overdamped drift law."""

    def test_drift_velocity_is_nu_f_times_pressure(self) -> None:
        import pytest

        cert = verify_overdamped_regime(nu_f=0.7, pressure=1.3)
        assert cert.drift_velocity == pytest.approx(0.7 * 1.3)

    def test_velocity_is_constant_under_sustained_pressure(self) -> None:
        # first-order: held pressure -> constant velocity (drift), not accel.
        cert = verify_overdamped_regime()
        assert cert.velocity_is_constant
        assert cert.max_velocity_variation < 1e-9

    def test_position_grows_linearly(self) -> None:
        import pytest

        cert = verify_overdamped_regime(nu_f=0.7, pressure=1.3)
        assert cert.position_linear_in_time
        # slope equals the drift velocity νf·F (not quadratic acceleration)
        assert cert.position_slope == pytest.approx(0.7 * 1.3, abs=1e-6)

    def test_mobility_law_linear_in_nu_f(self) -> None:
        cert = verify_overdamped_regime()
        assert cert.mobility_linear_in_nu_f

    def test_drift_linear_in_pressure(self) -> None:
        cert = verify_overdamped_regime()
        assert cert.drift_linear_in_pressure

    def test_is_first_order_not_inertial(self) -> None:
        cert = verify_overdamped_regime()
        assert not cert.is_second_order

    def test_certificate_valid(self) -> None:
        cert = verify_overdamped_regime()
        assert isinstance(cert, OverdampedRegimeCertificate)
        assert cert.is_overdamped_drift
        assert "VALID" in cert.summary()
        assert "mobility law" in cert.summary()


class TestOverdampedProjection:
    """Diffusion is the overdamped projection of the substrate wave flow."""

    def test_damped_wave_roots_satisfy_characteristic_equation(self) -> None:
        import pytest

        G = _canonical_graph(40)
        gamma = 50.0
        lambdas, s_slow, s_fast = damped_wave_rates(G, gamma)
        # s^2 + gamma s + lambda = 0  =>  s_slow + s_fast = -gamma,
        # s_slow * s_fast = lambda (Vieta).
        assert s_slow + s_fast == pytest.approx(-gamma * np.ones_like(lambdas))
        assert s_slow * s_fast == pytest.approx(lambdas)

    def test_slow_root_converges_to_diffusion_rate(self) -> None:
        # the slow root -> -lambda_k/gamma (the diffusion rate nu_f*lambda_k)
        G = _canonical_graph(40)
        cert = verify_overdamped_projection(G, gamma=50.0)
        assert cert.max_rate_rel_error < 1e-2

    def test_bridge_error_scales_as_inverse_gamma_squared(self) -> None:
        import pytest

        # rate_error * gamma^2 -> constant ~ lambda_max as gamma grows
        G = _canonical_graph(40)
        c1 = verify_overdamped_projection(G, gamma=50.0)
        c2 = verify_overdamped_projection(G, gamma=200.0)
        # larger gamma => the constant sits closer to lambda_max
        err1 = abs(c1.rate_error_times_gamma_sq - c1.lambda_max)
        err2 = abs(c2.rate_error_times_gamma_sq - c2.lambda_max)
        assert err2 <= err1 + 1e-9
        assert c2.rate_error_times_gamma_sq == pytest.approx(c2.lambda_max, rel=0.05)

    def test_nu_f_is_inverse_damping(self) -> None:
        import pytest

        G = _canonical_graph(40)
        cert = verify_overdamped_projection(G, gamma=40.0)
        assert cert.nu_f_effective == pytest.approx(1.0 / 40.0)

    def test_slowest_mode_matches_diffusion_spectral_gap(self) -> None:
        import pytest

        G = _canonical_graph(40)
        cert = verify_overdamped_projection(G, gamma=100.0)
        assert cert.slowest_slow_rate == pytest.approx(
            cert.slowest_diffusion_rate, rel=2e-2
        )

    def test_trajectory_collapses_onto_diffusion(self) -> None:
        G = _canonical_graph(40)
        cert = verify_overdamped_projection(G, gamma=100.0)
        assert cert.trajectory_max_rel_error < 1e-2

    def test_certificate_valid(self) -> None:
        G = _canonical_graph(40)
        cert = verify_overdamped_projection(G, gamma=50.0)
        assert isinstance(cert, OverdampedProjectionCertificate)
        assert cert.is_valid_projection
        assert "VALID" in cert.summary()


class TestUndampedLimit:
    """The gamma->0 end of the bridge: damped wave -> standing waves."""

    def test_small_gamma_recovers_standing_waves(self) -> None:
        G = _canonical_graph(40)
        cert = verify_undamped_limit(G, gamma=1e-3)
        assert cert.matches_discrete_modes
        assert cert.max_freq_rel_error < 1e-2

    def test_decay_rate_is_half_gamma(self) -> None:
        import pytest

        # underdamped envelope decay Re(s) = gamma/2
        G = _canonical_graph(40)
        gamma = 1e-3
        cert = verify_undamped_limit(G, gamma=gamma)
        assert cert.max_decay_rate == pytest.approx(gamma / 2.0, rel=1e-6)

    def test_frequency_error_scales_as_gamma_squared(self) -> None:
        # freq error / gamma^2 -> constant as gamma shrinks
        G = _canonical_graph(40)
        c1 = verify_undamped_limit(G, gamma=1e-2)
        c2 = verify_undamped_limit(G, gamma=1e-3)
        # the constant stabilises: the two normalised values agree closely
        assert (
            abs(c2.freq_error_times_inv_gamma_sq - c1.freq_error_times_inv_gamma_sq)
            < 0.05 * c1.freq_error_times_inv_gamma_sq
        )

    def test_frequencies_match_eigenmode_sqrt(self) -> None:
        import pytest

        # standing-wave frequencies are sqrt of the diffusion eigenvalues,
        # in ascending-eigenvalue order including the uniform mode (omega~0),
        # matching the verify_discrete_modes convention.
        G = _canonical_graph(40)
        cert = verify_undamped_limit(G, gamma=1e-3)
        _, lap = structural_diffusion_operator(G)
        lam = np.sort(np.clip(np.linalg.eigvals(lap).real, 0.0, None))
        expected = [float(x) ** 0.5 for x in lam[:6]]
        assert len(cert.standing_wave_frequencies) == len(expected)
        for got, exp in zip(cert.standing_wave_frequencies, expected):
            assert got == pytest.approx(exp, abs=1e-6)

    def test_certificate_valid(self) -> None:
        G = _canonical_graph(40)
        cert = verify_undamped_limit(G, gamma=1e-3)
        assert isinstance(cert, UndampedLimitCertificate)
        assert cert.is_valid_undamped_limit
        assert "VALID" in cert.summary()


class TestDiscreteModes:
    """A bounded manifold has discrete standing-wave eigenmodes."""

    def test_spectrum_is_discrete_and_finite(self) -> None:
        G = nx.path_graph(30)
        eigvals, eigvecs = structural_eigenmodes(G)
        assert eigvals.shape == (30,)
        assert eigvecs.shape == (30, 30)
        assert np.all(np.isfinite(eigvals))

    def test_modes_orthonormal(self) -> None:
        G = nx.path_graph(30)
        _, eigvecs = structural_eigenmodes(G)
        gram = eigvecs.T @ eigvecs
        assert np.allclose(gram, np.eye(30), atol=1e-9)

    def test_uniform_zero_mode(self) -> None:
        # λ₁ = 0 (uniform mode / conserved diffusion mode)
        G = nx.path_graph(30)
        eigvals, _ = structural_eigenmodes(G)
        assert abs(float(eigvals[0])) < 1e-9
        assert float(eigvals[1]) > 0.0  # spectral gap positive

    def test_path_modes_are_cosine_standing_waves(self) -> None:
        # path-graph L_sym mode k is a degree-weighted cosine standing wave
        G = nx.path_graph(40)
        _, eigvecs = structural_eigenmodes(G)
        k = 3
        i = np.arange(40)
        deg = np.array([G.degree(n) for n in G.nodes()], dtype=float)
        cos_pred = np.sqrt(deg) * np.cos(np.pi * k * (i + 0.5) / 40)
        cos_pred /= np.linalg.norm(cos_pred)
        mode = eigvecs[:, k] / np.linalg.norm(eigvecs[:, k])
        assert abs(float(np.dot(mode, cos_pred))) > 0.99

    def test_nodal_domain_count_grows_on_path(self) -> None:
        # Courant: k-th mode has exactly k sign changes on a 1D manifold
        G = nx.path_graph(40)
        _, eigvecs = structural_eigenmodes(G)
        counts = [nodal_domain_count(eigvecs[:, k]) for k in range(6)]
        assert counts == [0, 1, 2, 3, 4, 5]

    def test_spectrum_matches_diffusion_operator(self) -> None:
        # L_sym spectrum == L_rw (diffusion operator) spectrum
        G = nx.watts_strogatz_graph(40, 4, 0.2, seed=5)
        eigvals, _ = structural_eigenmodes(G)
        _, lrw = structural_diffusion_operator(G)
        rw_spec = np.sort(np.linalg.eigvals(lrw).real)
        assert np.allclose(np.sort(eigvals), rw_spec, atol=1e-7)

    def test_standing_wave_frequencies_are_sqrt_lambda(self) -> None:
        G = nx.path_graph(30)
        eigvals, _ = structural_eigenmodes(G)
        cert = verify_discrete_modes(G)
        for k, f in enumerate(cert.standing_wave_frequencies):
            assert f == __import__("pytest").approx(float(np.sqrt(eigvals[k])))

    def test_certificate_valid_across_topologies(self) -> None:
        for G in (
            nx.path_graph(40),
            nx.cycle_graph(40),
            nx.watts_strogatz_graph(40, 4, 0.2, seed=5),
        ):
            cert = verify_discrete_modes(G)
            assert isinstance(cert, DiscreteModeCertificate)
            assert cert.is_valid_discrete_modes
            assert "VALID" in cert.summary()


class TestStructuralStability:
    """The dispersion relation governs structural linear stability."""

    @staticmethod
    def _graph(builder):
        G = builder()
        for nd in G.nodes():
            G.nodes[nd]["nu_f"] = 1.0
        return G

    def test_dispersion_at_zero_is_negative_relaxation(self) -> None:
        G = self._graph(lambda: nx.watts_strogatz_graph(50, 6, 0.2, seed=7))
        sigma0 = dispersion_relation(G, 0.0)
        rates = relaxation_spectrum(G)
        assert np.allclose(np.sort(-sigma0), np.sort(rates), atol=1e-7)

    def test_pure_diffusion_decays_nonuniform_modes(self) -> None:
        # at r=0 every non-uniform mode decays (sigma_k < 0 for k >= 1)
        G = self._graph(lambda: nx.cycle_graph(40))
        sigma0 = dispersion_relation(G, 0.0)
        assert np.all(sigma0[1:] < 1e-9)

    def test_instability_threshold_is_nu_times_lambda2(self) -> None:
        import pytest

        G = self._graph(lambda: nx.watts_strogatz_graph(50, 6, 0.2, seed=7))
        eigvals, _ = structural_eigenmodes(G)
        nu = structural_diffusivity(G)
        assert instability_threshold(G) == pytest.approx(float(nu * eigvals[1]))

    def test_below_threshold_only_uniform_grows(self) -> None:
        G = self._graph(lambda: nx.cycle_graph(40))
        rc = instability_threshold(G)
        sigma = dispersion_relation(G, 0.5 * rc)
        # no non-uniform mode unstable below threshold
        assert np.all(sigma[1:] < 1e-9)

    def test_above_threshold_fiedler_grows(self) -> None:
        G = self._graph(lambda: nx.cycle_graph(40))
        rc = instability_threshold(G)
        sigma = dispersion_relation(G, rc + 0.01)
        # the Fiedler mode (k=1) is now unstable
        assert sigma[1] > 0.0

    def test_fiedler_partition_splits_barbell_evenly(self) -> None:
        # the two communities of a barbell are the weakest cut
        G = self._graph(lambda: nx.barbell_graph(20, 0))
        part_a, part_b = fiedler_partition(G)
        assert {len(part_a), len(part_b)} == {20}

    def test_certificate_valid_across_topologies(self) -> None:
        for builder in (
            lambda: nx.cycle_graph(40),
            lambda: nx.barbell_graph(20, 0),
            lambda: nx.watts_strogatz_graph(60, 6, 0.2, seed=7),
        ):
            G = self._graph(builder)
            cert = verify_structural_stability(G)
            assert isinstance(cert, StructuralStabilityCertificate)
            assert cert.is_valid_stability
            assert cert.first_unstable_mode == 1  # Fiedler
            assert cert.pure_diffusion_stable
            assert "VALID" in cert.summary()


class TestStructuralRandomWalk:
    """The diffusion operator generates a random walk with resistance."""

    def test_operator_is_walk_generator(self) -> None:
        # L_rw = I - P exactly
        G = nx.watts_strogatz_graph(40, 6, 0.2, seed=7)
        nodes, lrw = structural_diffusion_operator(G)
        _, p = random_walk_matrix(G)
        assert np.allclose(lrw, np.eye(len(nodes)) - p, atol=1e-12)

    def test_transition_is_row_stochastic(self) -> None:
        G = nx.watts_strogatz_graph(40, 6, 0.2, seed=7)
        _, p = random_walk_matrix(G)
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_stationary_is_degree(self) -> None:
        # pi proportional to degree, and pi @ P == pi
        G = nx.watts_strogatz_graph(40, 6, 0.2, seed=7)
        nodes, pi = stationary_distribution(G)
        _, p = random_walk_matrix(G)
        deg = np.array([G.degree(n) for n in nodes], dtype=float)
        assert np.allclose(pi, deg / deg.sum())
        assert np.allclose(pi @ p, pi, atol=1e-9)

    def test_effective_resistance_is_metric(self) -> None:
        G = nx.watts_strogatz_graph(40, 6, 0.2, seed=7)
        _, r = effective_resistance(G)
        assert np.allclose(r, r.T)  # symmetric
        assert np.all(r >= -1e-9)  # non-negative
        assert np.allclose(np.diag(r), 0.0)  # zero self-resistance
        # triangle inequality on a sample
        for i, j, k in [(0, 10, 20), (5, 15, 25), (1, 30, 8)]:
            assert r[i, k] <= r[i, j] + r[j, k] + 1e-7

    def test_commute_time_is_2m_resistance(self) -> None:
        G = nx.watts_strogatz_graph(40, 6, 0.2, seed=7)
        nodes, r = effective_resistance(G)
        _, c = commute_time(G)
        m = G.number_of_edges()
        assert np.allclose(c, 2.0 * m * r, atol=1e-7)

    def test_commute_time_matches_monte_carlo(self) -> None:
        # anchor: analytic commute time predicts a Monte-Carlo random walk
        G = nx.watts_strogatz_graph(30, 4, 0.2, seed=3)
        nodes, c = commute_time(G)
        idx = {n: i for i, n in enumerate(nodes)}
        A = nx.to_numpy_array(G, nodelist=nodes)
        nbrs = [np.where(A[k] > 0)[0] for k in range(len(nodes))]
        rng = np.random.default_rng(1)
        s, t = 0, 12
        total, trials = 0, 300
        for _ in range(trials):
            cur, steps, hit = s, 0, False
            while steps < 100000:
                cur = int(rng.choice(nbrs[cur]))
                steps += 1
                if not hit and cur == t:
                    hit = True
                elif hit and cur == s:
                    break
            total += steps
        mc = total / trials
        analytic = float(c[idx[s], idx[t]])
        assert abs(mc - analytic) / analytic < 0.12  # statistical tolerance

    def test_certificate_valid_across_topologies(self) -> None:
        for G in (
            nx.cycle_graph(40),
            nx.barbell_graph(20, 0),
            nx.watts_strogatz_graph(50, 6, 0.2, seed=7),
        ):
            cert = verify_structural_random_walk(G)
            assert isinstance(cert, RandomWalkCertificate)
            assert cert.is_valid_random_walk
            assert cert.operator_is_walk_generator
            assert cert.stationary_is_degree
            assert cert.commute_equals_2m_resistance
            assert "VALID" in cert.summary()


def _flow_graph(n: int = 40, seed: int = 7) -> nx.Graph:
    G = nx.watts_strogatz_graph(n, 6, 0.2, seed=seed)
    rng = np.random.default_rng(seed)
    for node in G.nodes():
        G.nodes[node]["EPI"] = float(rng.uniform(-0.4, 0.4))
    return G


class TestStructuralFlow:
    """The diffusion transport carries a Fick current obeying Kirchhoff."""

    def test_current_is_antisymmetric(self) -> None:
        # J_ij = EPI_i - EPI_j = -J_ji (directed edge flow)
        G = _flow_graph()
        _, j = structural_current(G)
        assert np.allclose(j, -j.T, atol=1e-12)

    def test_current_supported_on_edges_only(self) -> None:
        # no current across non-adjacent nodes
        G = _flow_graph()
        nodes, j = structural_current(G)
        A = nx.to_numpy_array(G, nodelist=nodes)
        assert np.allclose(j[A == 0.0], 0.0)

    def test_kirchhoff_current_law(self) -> None:
        # net outflow div(J)_i = (L*EPI)_i (continuity = current law)
        G = _flow_graph()
        nodes, div = current_divergence(G)
        A = nx.to_numpy_array(G, nodelist=nodes)
        L = np.diag(A.sum(1)) - A
        epi = np.array(
            [get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in nodes],
            dtype=float,
        )
        assert np.allclose(div, L @ epi, atol=1e-12)

    def test_total_flux_balances(self) -> None:
        # closed network: sum of net outflows = 0 (no sources)
        G = _flow_graph()
        _, div = current_divergence(G)
        assert abs(float(div.sum())) < 1e-9

    def test_equilibrium_zero_current(self) -> None:
        # a uniform EPI field carries zero current everywhere
        G = nx.watts_strogatz_graph(30, 4, 0.2, seed=5)
        for node in G.nodes():
            G.nodes[node]["EPI"] = 0.37
        _, j = structural_current(G)
        assert np.allclose(j, 0.0, atol=1e-12)
        _, div = current_divergence(G)
        assert np.allclose(div, 0.0, atol=1e-12)

    def test_ohm_law_injected_current(self) -> None:
        # injected unit current s->t induces potential drop = R_eff(s,t)
        G = _flow_graph()
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        A = nx.to_numpy_array(G, nodelist=nodes)
        L = np.diag(A.sum(1)) - A
        lp = np.linalg.pinv(L)
        _, r = effective_resistance(G)
        s, t = idx[nodes[0]], idx[nodes[15]]
        b = np.zeros(len(nodes))
        b[s], b[t] = 1.0, -1.0
        v = lp @ b
        drop = v[s] - v[t]
        assert np.isclose(drop, r[s, t], atol=1e-7)

    def test_certificate_valid_across_topologies(self) -> None:
        builders = (
            lambda: nx.cycle_graph(40),
            lambda: nx.barbell_graph(15, 0),
            lambda: nx.watts_strogatz_graph(50, 6, 0.2, seed=7),
        )
        rng = np.random.default_rng(0)
        for build in builders:
            G = build()
            for node in G.nodes():
                G.nodes[node]["EPI"] = float(rng.uniform(-0.4, 0.4))
            cert = verify_structural_flow(G)
            assert isinstance(cert, StructuralFlowCertificate)
            assert cert.is_valid_flow
            assert cert.current_antisymmetric
            assert cert.kirchhoff_holds
            assert cert.total_flux_balances
            assert cert.equilibrium_zero_current
            assert cert.ohm_law_holds
            assert "VALID" in cert.summary()


class TestEmergentPulse:
    """The emergent pulse read-out: the resonant rhythm the substrate plays."""

    def test_fundamental_is_slowest_nonuniform_resonance(self) -> None:
        import pytest

        # ring C_24: the slowest resonance is omega_2 = sqrt(1 - cos(2 pi/24))
        G = nx.cycle_graph(24)
        pulse = compute_emergent_pulse(G)
        expected = math.sqrt(1.0 - math.cos(2.0 * math.pi / 24))
        assert pulse["fundamental"] == pytest.approx(expected, rel=1e-6)

    def test_excludes_uniform_mode(self) -> None:
        # the lambda ~ 0 uniform mode is not a resonance
        G = nx.cycle_graph(24)
        pulse = compute_emergent_pulse(G)
        assert pulse["n_modes"] == 23
        assert all(w > 1e-9 for w in pulse["resonant_spectrum"])

    def test_vibration_energy_is_half_sum_lambda(self) -> None:
        import pytest

        G = nx.cycle_graph(24)
        eigvals, _ = structural_eigenmodes(G)
        pulse = compute_emergent_pulse(G)
        assert pulse["vibration_energy"] == pytest.approx(
            0.5 * float(np.sum(eigvals)), rel=1e-9
        )

    def test_fractal_signature_counts_degeneracy(self) -> None:
        # K_6: L_sym eigenvalue 6/5 has multiplicity 5 (the standard irrep
        # of S_6) -> the self-similar / fractal degeneracy signature
        G = nx.complete_graph(6)
        pulse = compute_emergent_pulse(G)
        assert pulse["spectral_multiplicity"] == 5
        assert pulse["dominant_beat"] >= 0.0


class TestNodalPulse:
    """The per-NFR pulse: each NFR oscillates; resonance couples the pulses."""

    @staticmethod
    def _pulsing_ring(n, vf, phases):
        G = nx.cycle_graph(n)
        for i, k in enumerate(G.nodes()):
            G.nodes[k].update(
                nu_f=float(vf[i]), theta=float(phases[i]), EPI=0.0
            )
        return G

    def test_phase_locked_pulses_have_unit_coherence(self) -> None:
        import pytest

        # all NFRs share one phase -> collective + local resonance = 1
        G = self._pulsing_ring(12, [1.0] * 12, [0.0] * 12)
        pulse = compute_nodal_pulse(G)
        assert pulse["phase_coherence"] == pytest.approx(1.0, abs=1e-9)
        assert pulse["mean_local_resonance"] == pytest.approx(1.0, abs=1e-9)

    def test_evenly_spread_pulses_unlock_collective_resonance(self) -> None:
        # phases evenly around the circle -> collective R = 0, yet each NFR
        # still locally resonates with its two neighbours at cos(2 pi / n)
        import pytest

        n = 8
        phases = [2.0 * math.pi * i / n for i in range(n)]
        G = self._pulsing_ring(n, [1.0] * n, phases)
        pulse = compute_nodal_pulse(G)
        assert pulse["phase_coherence"] == pytest.approx(0.0, abs=1e-9)
        assert pulse["mean_local_resonance"] == pytest.approx(
            math.cos(2.0 * math.pi / n), abs=1e-9
        )

    def test_frequency_stats_are_per_nfr_pulse_rates(self) -> None:
        import pytest

        vf = [0.5, 1.0, 1.5, 2.0]
        G = self._pulsing_ring(4, vf, [0.0] * 4)
        pulse = compute_nodal_pulse(G)
        assert pulse["mean_frequency"] == pytest.approx(1.25)
        assert pulse["frequency_spread"] == pytest.approx(float(np.std(vf)))
        assert pulse["n_pulsing"] == 4
        assert pulse["n_nodes"] == 4

    def test_resonance_gate_is_canonical_u3_bound(self) -> None:
        import pytest

        from tnfr.constants.canonical import DELTA_PHI_MAX

        G = self._pulsing_ring(6, [1.0] * 6, [0.0] * 6)
        pulse = compute_nodal_pulse(G)
        assert pulse["resonance_gate"] == pytest.approx(float(DELTA_PHI_MAX))

    def test_inactive_nfr_not_pulsing(self) -> None:
        # nu_f = 0 -> the node cannot reorganize -> not a live pulse
        G = self._pulsing_ring(4, [1.0, 0.0, 1.0, 0.0], [0.0] * 4)
        pulse = compute_nodal_pulse(G)
        assert pulse["n_pulsing"] == 2
        assert pulse["n_nodes"] == 4
