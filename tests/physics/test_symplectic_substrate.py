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
    BLOCK_COMPATIBLE_METRIC,
    BLOCK_COMPLEX_STRUCTURE,
    BLOCK_SYMPLECTIC_FORM,
    background_potential,
    canonical_bracket_table,
    compatible_metric_matrix,
    complex_structure_matrix,
    diagonal_moment_map,
    evolve_substrate_flow,
    extract_phase_space_point,
    geometric_sector_energy,
    hamiltonian_vector_field,
    kahler_potential,
    liouville_divergence,
    loop_action_integral,
    noether_charges,
    polarization_density,
    polarization_vector,
    potential_sector_energy,
    reduced_symplectic_form_matrix,
    substrate_flow_matrix,
    substrate_hamiltonian,
    symplectic_form_matrix,
    to_action_angle,
    to_complex_coordinates,
    verify_adiabatic_invariance,
    verify_canonical_structure,
    verify_hermitian_structure,
    verify_integrability,
    verify_noether_conservation,
    verify_poincare_cartan,
    verify_polarization_symmetry,
    verify_substrate_geometry,
    verify_symplectic_reduction,
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
        expected = np.stack([z[:, 1], -z[:, 0], z[:, 3], -z[:, 2]], axis=1)
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
            assert substrate_hamiltonian(evolved) == pytest.approx(h0, abs=1e-9)

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
        psi_arr = np.array([psi[n] for n in point.nodes], dtype=complex)
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


class TestPolarizationSymmetry:
    """The substrate carries a polarization symmetry (U(2)) with Stokes vector."""

    def test_polarization_vector_present(self) -> None:
        G = _canonical_graph(24)
        ch = polarization_vector(extract_phase_space_point(G))
        assert set(ch) == {"p_1", "p_2", "p_3", "magnitude_sq"}

    def test_p3_equals_sector_energy_difference(self) -> None:
        import pytest

        G = _canonical_graph(24)
        point = extract_phase_space_point(G)
        ch = polarization_vector(point)
        e_diff = geometric_sector_energy(point) - potential_sector_energy(point)
        assert ch["p_3"] == pytest.approx(e_diff)

    def test_magnitude_is_sum_of_squares(self) -> None:
        import pytest

        G = _canonical_graph(20)
        ch = polarization_vector(extract_phase_space_point(G))
        assert ch["magnitude_sq"] == pytest.approx(
            ch["p_1"] ** 2 + ch["p_2"] ** 2 + ch["p_3"] ** 2
        )

    def test_su2_algebra_closes(self) -> None:
        G = _canonical_graph(24)
        cert = verify_polarization_symmetry(G)
        assert cert.su2_algebra_closes
        assert cert.max_algebra_residual < 1e-6

    def test_rotation_is_symplectic(self) -> None:
        G = _canonical_graph(24)
        cert = verify_polarization_symmetry(G)
        assert cert.rotation_is_symplectic

    def test_charges_conserved_along_flow(self) -> None:
        G = _canonical_graph(24)
        cert = verify_polarization_symmetry(G)
        assert cert.charges_conserved
        assert cert.max_charge_drift < 1e-6

    def test_certificate_valid_polarization(self) -> None:
        G = _canonical_graph(30)
        cert = verify_polarization_symmetry(G)
        assert cert.is_valid_polarization_symmetry
        assert cert.p3_equals_energy_difference
        assert "VALID" in cert.summary()

    def test_full_polarization_per_node_radius_equals_energy(self) -> None:
        G = _canonical_graph(24)
        d = polarization_density(extract_phase_space_point(G))
        # |P_node| = e_node (the Poincaré sphere S²; fully polarized).
        assert np.allclose(d["radius"], d["energy"])

    def test_polarization_density_poincare_is_unit(self) -> None:
        G = _canonical_graph(20)
        d = polarization_density(extract_phase_space_point(G))
        norms = np.sqrt((d["poincare"] ** 2).sum(axis=0))
        assert np.allclose(norms, 1.0)

    def test_density_sums_to_global_charges(self) -> None:
        import pytest

        G = _canonical_graph(24)
        point = extract_phase_space_point(G)
        d = polarization_density(point)
        glob = polarization_vector(point)
        assert float(d["p_1"].sum()) == pytest.approx(glob["p_1"])
        assert float(d["p_2"].sum()) == pytest.approx(glob["p_2"])
        assert float(d["p_3"].sum()) == pytest.approx(glob["p_3"])

    def test_certificate_reports_full_polarization(self) -> None:
        G = _canonical_graph(30)
        cert = verify_polarization_symmetry(G)
        assert cert.full_polarization_holds
        assert cert.max_polarization_residual < 1e-9
        assert "fully polarized" in cert.summary()


class TestSubstrateGeometryReport:
    """The consolidated aggregator runs the whole tower in one call."""

    def test_all_structures_valid(self) -> None:
        G = _canonical_graph(30)
        report = verify_substrate_geometry(G)
        assert report.all_structures_valid
        assert report.n_nodes == 30
        assert report.phase_space_dimension == 120

    def test_bundles_seven_certificates(self) -> None:
        G = _canonical_graph(20)
        report = verify_substrate_geometry(G)
        # Each sub-certificate is the same type the individual verify returns.
        assert report.canonical.is_valid_symplectic_manifold
        assert report.noether.is_conserved
        assert report.hermitian.is_valid_hermitian_structure
        assert report.integrability.is_completely_integrable
        assert report.poincare_cartan.all_invariants_hold
        assert report.marsden_weinstein.is_valid_reduction
        assert report.polarization.is_valid_polarization_symmetry

    def test_sub_certificates_match_individual_calls(self) -> None:
        G = _canonical_graph(24)
        report = verify_substrate_geometry(G)
        # The aggregated canonical certificate matches a direct call.
        direct = verify_canonical_structure(G)
        assert (
            report.canonical.is_valid_symplectic_manifold
            == direct.is_valid_symplectic_manifold
        )
        assert report.marsden_weinstein.reduced_dimension == 4 * 24 - 2

    def test_summary_lists_all_seven(self) -> None:
        G = _canonical_graph(20)
        report = verify_substrate_geometry(G)
        summary = report.summary()
        assert "ALL VALID" in summary
        # Seven numbered structures appear in the multi-line summary.
        for k in range(1, 8):
            assert f"  {k}." in summary


class TestSubstrateIntegration:
    """The emergent substrate is exposed across SDK, telemetry, and API."""

    def test_physics_package_exports(self) -> None:
        from tnfr.physics import extract_phase_space_point as _eps
        from tnfr.physics import substrate_hamiltonian as _sh
        from tnfr.physics import symplectic_form_matrix as _sfm
        from tnfr.physics import verify_canonical_structure as _vcs

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

    def test_unified_telemetry_includes_pulse(self) -> None:
        # dual-face telemetry: the conservative pulse (the resonant rhythm)
        # alongside the dissipative canonical coherence read-out
        from tnfr.physics.fields import compute_unified_telemetry

        G = _canonical_graph(20)
        telemetry = compute_unified_telemetry(G)
        assert "pulse" in telemetry
        pulse = telemetry["pulse"]
        assert pulse["n_modes"] >= 1
        assert pulse["fundamental"] > 0.0
        assert pulse["vibration_energy"] > 0.0

    def test_unified_telemetry_includes_resonance(self) -> None:
        # the per-NFR pulse / resonance face (the source the collective
        # rhythm emerges from) alongside the collective pulse
        from tnfr.physics.fields import compute_unified_telemetry

        G = _canonical_graph(20)
        telemetry = compute_unified_telemetry(G)
        assert "resonance" in telemetry
        res = telemetry["resonance"]
        assert 0.0 <= res["phase_coherence"] <= 1.0
        assert 0.0 <= res["mean_local_resonance"] <= 1.0
        assert res["n_nodes"] == 20

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


class TestAdiabaticInvariance:
    """The substrate action is an adiabatic invariant of a slow nu_f ramp."""

    def test_slow_ramp_conserves_action(self) -> None:
        cert = verify_adiabatic_invariance()
        assert cert.is_adiabatic_invariant
        assert cert.slow_drift < 1e-2

    def test_drift_decreases_with_slowness(self) -> None:
        # the adiabatic signature: slower ramp -> smaller action drift
        cert = verify_adiabatic_invariance()
        assert cert.drift_decreases_with_slowness
        assert cert.slow_drift < cert.fast_drift

    def test_fast_ramp_breaks_invariance(self) -> None:
        # a sudden ramp (T=1) injects/extracts action: large drift
        cert = verify_adiabatic_invariance(ramp_times=(1.0, 80.0))
        assert cert.fast_drift > 0.1

    def test_drift_series_trends_down(self) -> None:
        cert = verify_adiabatic_invariance(ramp_times=(1.0, 5.0, 20.0, 80.0))
        drifts = cert.action_drifts
        # the slow end is far below the fast end (orders of magnitude)
        assert drifts[-1] < drifts[0] / 10.0

    def test_certificate_valid(self) -> None:
        from tnfr.physics.symplectic_substrate import AdiabaticInvarianceCertificate

        cert = verify_adiabatic_invariance()
        assert isinstance(cert, AdiabaticInvarianceCertificate)
        assert "VALID" in cert.summary()
        assert "clock" in cert.summary()
