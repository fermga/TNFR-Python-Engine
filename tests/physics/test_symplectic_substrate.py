"""Tests for the emergent symplectic substrate.

Module under test: physics/symplectic_substrate.py.
Verifies that the geometry emerging from the TNFR nodal dynamics is a valid
symplectic manifold: antisymmetric non-degenerate closed 2-form, canonical
Poisson brackets, Jacobi identity, Liouville volume preservation, harmonic
Hamiltonian flow, and consistency with the canonical energy functional.
"""
from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.conservation import compute_energy_functional
from tnfr.physics.symplectic_substrate import (
    BLOCK_SYMPLECTIC_FORM,
    BLOCK_COMPLEX_STRUCTURE,
    BLOCK_COMPATIBLE_METRIC,
    background_potential,
    canonical_bracket_table,
    complex_structure_matrix,
    compatible_metric_matrix,
    evolve_substrate_flow,
    extract_phase_space_point,
    geometric_sector_energy,
    hamiltonian_vector_field,
    kahler_potential,
    liouville_divergence,
    loop_action_integral,
    noether_charges,
    potential_sector_energy,
    reduced_symplectic_form_matrix,
    substrate_flow_matrix,
    substrate_hamiltonian,
    symplectic_form_matrix,
    to_action_angle,
    to_complex_coordinates,
    verify_canonical_structure,
    verify_hermitian_structure,
    verify_integrability,
    verify_noether_conservation,
    verify_poincare_cartan,
    verify_symplectic_reduction,
    diagonal_moment_map,
)


def _canonical_graph(n: int = 30, seed: int = 5) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    for node in G.nodes():
        G.nodes[node]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        G.nodes[node]["EPI"] = rng.uniform(0.2, 0.8)
        G.nodes[node]["nu_f"] = rng.uniform(0.5, 1.5)
    default_compute_delta_nfr(G)
    return G


class TestBlockSymplecticForm:
    """The per-node canonical block J4 has the symplectic properties."""

    def test_antisymmetric(self) -> None:
        assert np.allclose(BLOCK_SYMPLECTIC_FORM.T, -BLOCK_SYMPLECTIC_FORM)

    def test_nondegenerate_determinant_one(self) -> None:
        assert abs(float(np.linalg.det(BLOCK_SYMPLECTIC_FORM)) - 1.0) < 1e-12

    def test_square_is_minus_identity(self) -> None:
        # J^2 = -I is the defining property of the canonical complex
        # structure underlying the symplectic form.
        j2 = BLOCK_SYMPLECTIC_FORM @ BLOCK_SYMPLECTIC_FORM
        assert np.allclose(j2, -np.eye(4))


class TestSymplecticFormMatrix:
    """The full 4N x 4N form is block-diagonal and symplectic."""

    def test_dimension(self) -> None:
        omega = symplectic_form_matrix(7)
        assert omega.shape == (28, 28)

    def test_antisymmetric_and_nondegenerate(self) -> None:
        omega = symplectic_form_matrix(10)
        assert np.allclose(omega.T, -omega)
        # det of a 4N block-diagonal of det-1 blocks is 1.
        assert abs(float(np.linalg.det(omega)) - 1.0) < 1e-9

    def test_rejects_zero_nodes(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            symplectic_form_matrix(0)


class TestCanonicalBrackets:
    """Poisson brackets are canonical: {q,p}=delta, others zero."""

    def test_conjugate_brackets_unity(self) -> None:
        table = canonical_bracket_table()
        assert abs(table["{qA,pA}"] - 1.0) < 1e-12
        assert abs(table["{qB,pB}"] - 1.0) < 1e-12

    def test_cross_and_self_brackets_vanish(self) -> None:
        table = canonical_bracket_table()
        assert abs(table["{qA,qB}"]) < 1e-12
        assert abs(table["{pA,pB}"]) < 1e-12
        assert abs(table["{qA,pB}"]) < 1e-12
        assert abs(table["{pA,qB}"]) < 1e-12


class TestPhaseSpaceExtraction:
    """The phase-space point is extracted from canonical fields."""

    def test_dimension_is_four_n(self) -> None:
        G = _canonical_graph(20)
        pt = extract_phase_space_point(G)
        assert pt.n_nodes == 20
        assert pt.dimension == 80
        assert pt.to_vector().shape == (80,)

    def test_vector_ordering(self) -> None:
        G = _canonical_graph(10)
        pt = extract_phase_space_point(G)
        z = pt.to_vector().reshape(pt.n_nodes, 4)
        # Per-node order is (K_phi, J_phi, Phi_s, J_dnfr).
        assert np.allclose(z[:, 0], pt.k_phi)
        assert np.allclose(z[:, 1], pt.j_phi)
        assert np.allclose(z[:, 2], pt.phi_s)
        assert np.allclose(z[:, 3], pt.j_dnfr)


class TestHamiltonianFlow:
    """The Hamiltonian flow is harmonic and volume-preserving."""

    def test_flow_is_harmonic(self) -> None:
        G = _canonical_graph(15)
        pt = extract_phase_space_point(G)
        z = pt.to_vector().reshape(pt.n_nodes, 4)
        zdot = hamiltonian_vector_field(pt).reshape(pt.n_nodes, 4)
        # q_dot = p, p_dot = -q per sector.
        expected = np.stack(
            [z[:, 1], -z[:, 0], z[:, 3], -z[:, 2]], axis=1
        )
        assert np.allclose(zdot, expected)

    def test_liouville_divergence_zero(self) -> None:
        G = _canonical_graph(25)
        pt = extract_phase_space_point(G)
        assert abs(liouville_divergence(pt)) < 1e-9


class TestEnergyConsistency:
    """H_sub + background potential reconstructs the energy functional."""

    def test_decomposition_matches_energy_functional(self) -> None:
        G = _canonical_graph(30, seed=7)
        pt = extract_phase_space_point(G)
        h_sub = substrate_hamiltonian(pt)
        u_bg = background_potential(pt)
        energy = compute_energy_functional(G)
        assert abs((h_sub + u_bg) - energy) < 1e-9

    def test_substrate_hamiltonian_nonnegative(self) -> None:
        G = _canonical_graph(20)
        pt = extract_phase_space_point(G)
        assert substrate_hamiltonian(pt) >= 0.0
        assert background_potential(pt) >= 0.0


class TestCanonicalCertificate:
    """The full certificate validates the emergent symplectic manifold."""

    def test_valid_manifold(self) -> None:
        G = _canonical_graph(30)
        cert = verify_canonical_structure(G)
        assert cert.is_valid_symplectic_manifold
        assert cert.is_antisymmetric
        assert cert.is_nondegenerate
        assert cert.is_closed
        assert cert.brackets_canonical
        assert cert.jacobi_satisfied
        assert cert.flow_is_harmonic
        assert abs(cert.liouville_divergence) < 1e-9
        assert cert.dimension == 120

    def test_certificate_summary_string(self) -> None:
        G = _canonical_graph(12)
        cert = verify_canonical_structure(G)
        summary = cert.summary()
        assert "VALID" in summary
        assert "dim=48" in summary

    def test_determinant_is_one(self) -> None:
        G = _canonical_graph(10)
        cert = verify_canonical_structure(G)
        assert abs(cert.determinant - 1.0) < 1e-12


class TestNoetherCharges:
    """Noether's theorem: substrate symmetries generate conserved charges."""

    def test_charges_split_total_energy(self) -> None:
        import pytest

        G = _canonical_graph(25)
        pt = extract_phase_space_point(G)
        charges = noether_charges(pt)
        # H_sub = E_geo + E_pot exactly.
        assert charges["time_translation"] == pytest.approx(
            charges["geometric_u1"] + charges["potential_u1"], abs=1e-12
        )
        assert charges["time_translation"] == pytest.approx(
            substrate_hamiltonian(pt), abs=1e-12
        )

    def test_sector_energies_nonnegative(self) -> None:
        G = _canonical_graph(20)
        pt = extract_phase_space_point(G)
        assert geometric_sector_energy(pt) >= 0.0
        assert potential_sector_energy(pt) >= 0.0

    def test_charges_conserved_under_flow(self) -> None:
        G = _canonical_graph(30)
        cert = verify_noether_conservation(G)
        assert cert.is_conserved
        assert cert.splits_exactly
        assert cert.max_hamiltonian_drift < 1e-9
        assert cert.max_geometric_drift < 1e-9
        assert cert.max_potential_drift < 1e-9

    def test_flow_preserves_hamiltonian(self) -> None:
        import pytest

        G = _canonical_graph(20)
        pt = extract_phase_space_point(G)
        h0 = substrate_hamiltonian(pt)
        for t in (0.5, 1.3, 2.7, 10.0):
            evolved = evolve_substrate_flow(pt, t)
            assert substrate_hamiltonian(evolved) == pytest.approx(
                h0, abs=1e-9
            )

    def test_flow_at_zero_is_identity(self) -> None:
        G = _canonical_graph(15)
        pt = extract_phase_space_point(G)
        evolved = evolve_substrate_flow(pt, 0.0)
        assert np.allclose(evolved.to_vector(), pt.to_vector())

    def test_certificate_summary(self) -> None:
        G = _canonical_graph(12)
        cert = verify_noether_conservation(G)
        summary = cert.summary()
        assert "CONSERVED" in summary
        assert "H_sub" in summary


class TestHermitianStructure:
    """The substrate carries a compatible Hermitian (flat Kähler) structure."""

    def test_complex_structure_squares_to_minus_id(self) -> None:
        j = BLOCK_COMPLEX_STRUCTURE
        assert np.allclose(j @ j, -np.eye(4))

    def test_complex_structure_is_minus_symplectic_form(self) -> None:
        assert np.allclose(BLOCK_COMPLEX_STRUCTURE, -BLOCK_SYMPLECTIC_FORM)

    def test_compatible_metric_is_identity(self) -> None:
        assert np.allclose(BLOCK_COMPATIBLE_METRIC, np.eye(4))

    def test_metric_is_positive_definite(self) -> None:
        eigvals = np.linalg.eigvalsh(BLOCK_COMPATIBLE_METRIC)
        assert np.all(eigvals > 0)

    def test_compatibility_omega_equals_jt_g(self) -> None:
        # ω(u,v) = g(Ju,v)  ⟺  Ω = Jᵀ g.
        j = BLOCK_COMPLEX_STRUCTURE
        g = BLOCK_COMPATIBLE_METRIC
        assert np.allclose(BLOCK_SYMPLECTIC_FORM, j.T @ g)

    def test_j_is_metric_orthogonal(self) -> None:
        j = BLOCK_COMPLEX_STRUCTURE
        g = BLOCK_COMPATIBLE_METRIC
        assert np.allclose(j.T @ g @ j, g)

    def test_block_matrices_dimension(self) -> None:
        assert complex_structure_matrix(5).shape == (20, 20)
        assert compatible_metric_matrix(5).shape == (20, 20)

    def test_complex_structure_acts_as_i(self) -> None:
        # J(q,p) = (−p, q) for ζ = q + i·p (multiplication by i).
        j = BLOCK_COMPLEX_STRUCTURE
        assert np.allclose(j @ np.array([1.0, 0, 0, 0]), [0, 1, 0, 0])
        assert np.allclose(j @ np.array([0, 1.0, 0, 0]), [-1, 0, 0, 0])

    def test_psi_is_geometric_complex_coordinate(self) -> None:
        from tnfr.physics.unified import compute_complex_geometric_field

        G = _canonical_graph(20)
        point = extract_phase_space_point(G)
        coords = to_complex_coordinates(point)
        psi = compute_complex_geometric_field(G)
        psi_arr = np.array(
            [psi[n] for n in point.nodes], dtype=complex
        )
        assert np.allclose(coords["geometric"], psi_arr)

    def test_kahler_potential_equals_hamiltonian(self) -> None:
        import pytest

        G = _canonical_graph(25)
        point = extract_phase_space_point(G)
        assert kahler_potential(point) == pytest.approx(
            substrate_hamiltonian(point), abs=1e-9
        )

    def test_certificate_valid(self) -> None:
        G = _canonical_graph(30)
        cert = verify_hermitian_structure(G)
        assert cert.is_valid_hermitian_structure
        assert cert.psi_is_geometric_coordinate
        assert cert.kahler_potential_matches
        assert cert.complex_dimension == 60
        assert "VALID" in cert.summary()


class TestIntegrability:
    """The substrate flow is completely integrable (Liouville–Arnold)."""

    def test_action_count_equals_dof(self) -> None:
        G = _canonical_graph(24)
        cert = verify_integrability(G)
        # 2N action variables for 2N degrees of freedom.
        assert cert.degrees_of_freedom == 48
        assert cert.n_action_variables == 48
        assert cert.n_action_variables == cert.degrees_of_freedom

    def test_actions_are_half_modulus_squared(self) -> None:
        G = _canonical_graph(20)
        point = extract_phase_space_point(G)
        aa = to_action_angle(point)
        coords = to_complex_coordinates(point)
        assert np.allclose(
            aa["action_geometric"], 0.5 * np.abs(coords["geometric"]) ** 2
        )
        assert np.allclose(
            aa["action_potential"], 0.5 * np.abs(coords["potential"]) ** 2
        )

    def test_actions_conserved_along_flow(self) -> None:
        G = _canonical_graph(24)
        point = extract_phase_space_point(G)
        aa0 = to_action_angle(point)
        evolved = evolve_substrate_flow(point, 2.3)
        aa = to_action_angle(evolved)
        assert np.allclose(aa["action_geometric"], aa0["action_geometric"])
        assert np.allclose(aa["action_potential"], aa0["action_potential"])

    def test_angles_advance_linearly(self) -> None:
        G = _canonical_graph(22)
        point = extract_phase_space_point(G)
        aa0 = to_action_angle(point)
        t = 0.7
        evolved = evolve_substrate_flow(point, t)
        aa = to_action_angle(evolved)
        # θ(t) = θ(0) − t, compared on the circle.
        delta = np.angle(
            np.exp(1j * (aa["angle_geometric"] - (aa0["angle_geometric"] - t)))
        )
        assert np.allclose(delta, 0.0, atol=1e-9)

    def test_actions_in_involution(self) -> None:
        G = _canonical_graph(24)
        cert = verify_integrability(G)
        assert cert.actions_in_involution
        assert cert.max_involution_bracket < 1e-12

    def test_sector_actions_match_noether_charges(self) -> None:
        import pytest

        G = _canonical_graph(24)
        point = extract_phase_space_point(G)
        aa = to_action_angle(point)
        assert float(np.sum(aa["action_geometric"])) == pytest.approx(
            geometric_sector_energy(point)
        )
        assert float(np.sum(aa["action_potential"])) == pytest.approx(
            potential_sector_energy(point)
        )

    def test_certificate_completely_integrable(self) -> None:
        G = _canonical_graph(30)
        cert = verify_integrability(G)
        assert cert.is_completely_integrable
        assert cert.actions_conserved
        assert cert.angles_advance_linearly
        assert cert.sector_actions_match_charges
        assert cert.degrees_of_freedom == 60
        assert "INTEGRABLE" in cert.summary()


class TestPoincareCartan:
    """The flow preserves the Poincaré–Cartan integral invariants."""

    def test_flow_matrix_is_symplectic(self) -> None:
        m = substrate_flow_matrix(4, 0.7)
        omega = symplectic_form_matrix(4)
        # 1st Poincaré invariant: M preserves ω.
        assert np.allclose(m.T @ omega @ m, omega)

    def test_flow_matrix_unit_determinant(self) -> None:
        m = substrate_flow_matrix(5, 1.3)
        # Top invariant ω^N = Liouville volume.
        assert abs(float(np.linalg.det(m)) - 1.0) < 1e-12

    def test_flow_matrix_dimension(self) -> None:
        assert substrate_flow_matrix(7, 0.4).shape == (28, 28)

    def test_flow_matrix_palindromic_spectrum(self) -> None:
        m = substrate_flow_matrix(3, 0.9)
        coeffs = np.poly(m)
        # Reciprocal symplectic spectrum → palindromic char poly.
        assert np.allclose(coeffs, coeffs[::-1], atol=1e-9)
        eig = np.linalg.eigvals(m)
        assert np.allclose(np.abs(eig), 1.0)

    def test_flow_matrix_invalid_nodes(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            substrate_flow_matrix(0, 0.5)

    def test_loop_action_integral_equals_minus_2pi_I(self) -> None:
        # ∮ p dq = −2π·I (negative enclosed area) for the torus loop.
        val = loop_action_integral(2.0)
        assert abs(val - (-2.0 * np.pi * 2.0)) < 1e-3

    def test_relative_invariant_preserved(self) -> None:
        G = _canonical_graph(24)
        cert = verify_poincare_cartan(G)
        assert cert.relative_invariant_preserved
        assert cert.max_relative_drift < 1e-9

    def test_bohr_sommerfeld_holds(self) -> None:
        G = _canonical_graph(24)
        cert = verify_poincare_cartan(G)
        # |∮ p dq| = 2π I on the action torus.
        assert cert.bohr_sommerfeld_holds

    def test_certificate_all_invariants_hold(self) -> None:
        G = _canonical_graph(30)
        cert = verify_poincare_cartan(G)
        assert cert.all_invariants_hold
        assert cert.preserves_symplectic_form
        assert cert.volume_preserved
        assert cert.char_poly_palindromic
        assert cert.phase_space_dimension == 120
        assert "ALL HOLD" in cert.summary()


class TestMarsdenWeinstein:
    """The flow's diagonal U(1) admits a Marsden–Weinstein reduction."""

    def test_moment_map_equals_hamiltonian(self) -> None:
        import pytest

        G = _canonical_graph(20)
        point = extract_phase_space_point(G)
        j = diagonal_moment_map(point)
        assert j == pytest.approx(substrate_hamiltonian(point))

    def test_moment_map_conserved(self) -> None:
        G = _canonical_graph(24)
        cert = verify_symplectic_reduction(G)
        assert cert.moment_map_conserved
        assert cert.max_moment_drift < 1e-9

    def test_reduced_dimension_is_4n_minus_2(self) -> None:
        G = _canonical_graph(20)
        cert = verify_symplectic_reduction(G)
        assert cert.phase_space_dimension == 80
        assert cert.reduced_dimension == 78

    def test_reduced_form_shape_and_determinant(self) -> None:
        # Reduced form is (4N−2)×(4N−2), det = (2N)².
        red = reduced_symplectic_form_matrix(3)
        assert red.shape == (10, 10)
        assert abs(float(np.linalg.det(red)) - 36.0) < 1e-6

    def test_reduced_form_nondegenerate(self) -> None:
        G = _canonical_graph(24)
        cert = verify_symplectic_reduction(G)
        assert cert.reduced_form_nondegenerate
        assert abs(cert.reduced_form_determinant) > 1.0

    def test_reduced_form_invalid_nodes(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            reduced_symplectic_form_matrix(0)

    def test_relative_phases_invariant(self) -> None:
        G = _canonical_graph(24)
        cert = verify_symplectic_reduction(G)
        assert cert.relative_phases_invariant

    def test_certificate_valid_reduction(self) -> None:
        G = _canonical_graph(30)
        cert = verify_symplectic_reduction(G)
        assert cert.is_valid_reduction
        assert cert.moment_map_is_hamiltonian
        assert cert.reduced_dimension == 118
        assert "VALID" in cert.summary()


class TestSubstrateIntegration:
    """The emergent substrate is exposed across SDK, telemetry, and API."""

    def test_physics_package_exports(self) -> None:
        from tnfr.physics import (
            extract_phase_space_point as _eps,
            verify_canonical_structure as _vcs,
            substrate_hamiltonian as _sh,
            symplectic_form_matrix as _sfm,
        )

        assert callable(_eps)
        assert callable(_vcs)
        assert callable(_sh)
        assert callable(_sfm)

    def test_unified_telemetry_includes_substrate(self) -> None:
        from tnfr.physics.fields import compute_unified_telemetry

        G = _canonical_graph(20)
        telemetry = compute_unified_telemetry(G)
        assert "symplectic_substrate" in telemetry
        sub = telemetry["symplectic_substrate"]
        assert sub["phase_space_dimension"] == 80
        assert abs(sub["liouville_divergence"]) < 1e-9

    def test_sdk_symplectic_substrate_method(self) -> None:
        from tnfr.sdk import TNFR, SymplecticReport

        net = TNFR.create(20).ring().evolve(2)
        report = net.symplectic_substrate()
        assert isinstance(report, SymplecticReport)
        assert report.is_valid_manifold
        assert report.phase_space_dimension == 80
        assert "VALID" in report.summary()

    def test_sdk_analyze_includes_substrate(self) -> None:
        from tnfr.sdk import TNFR

        net = TNFR.create(15).ring().evolve(2)
        analysis = TNFR.analyze(net)
        assert "symplectic_substrate" in analysis
        assert analysis["features"]["symplectic_substrate"] is True
