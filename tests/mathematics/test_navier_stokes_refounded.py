"""Tests for the re-founded TNFR-Navier-Stokes program (two-face reading).

The linear part of incompressible NS is the diffusive (over-damped) projection
of the substrate wave (nu_f = nu); blow-up is a purely nonlinear K_phi cascade.
Closes nothing; Clay stays open.
"""

from __future__ import annotations

import math

from tnfr.navier_stokes import (
    CascadeFrontierCertificate,
    TNFRNavierStokes,
    build_torus_graph_3d,
    cascade_moment_hierarchy,
    face_of_flow,
    flow_coherence,
    measure_cascade_frontier,
    moment_ladder_closure,
    taylor_green_initial_condition_3d,
    verify_diffusive_face,
    vorticity_modal_spectrum,
)


def test_taylor_green_solver_is_faithful():
    flow = TNFRNavierStokes(8, 0.05, 1.0)
    e0 = flow.energy()
    for _ in range(20):
        flow.step(0.02)
    assert flow.energy() < e0  # viscous energy decay
    assert flow.divergence_sup() < 1e-8  # incompressible to round-off


def test_stretching_nonzero_in_3d():
    flow = TNFRNavierStokes(8, 0.02, 1.0)
    for _ in range(10):
        flow.step(0.02)
    assert abs(flow.stretching_production()) > 1e-6


def test_linear_ns_is_the_diffusive_face():
    # gamma = 1/nu; the engine certificate must be VALID and recover nu_f = nu.
    for nu in (0.05, 0.01):
        cert = verify_diffusive_face(nu)
        assert cert.is_valid_projection
        assert math.isclose(cert.nu_f_effective, nu, rel_tol=1e-9)
        fo = face_of_flow(nu)
        assert fo["overdamped"] is True
        assert "diffusive" in fo["face"]


def test_vorticity_modal_spectrum_is_well_formed():
    flow = TNFRNavierStokes(8, 0.02, 1.0)
    for _ in range(5):
        flow.step(0.02)
    sp = vorticity_modal_spectrum(flow, n_shells=10)
    assert len(sp["enstrophy_spectrum"]) == 10
    assert 0.0 <= sp["high_mode_fraction"] <= 1.0
    assert sum(sp["enstrophy_spectrum"]) > 0.0


def test_cascade_frontier_saturates_at_fixed_re_and_stays_open():
    cert = measure_cascade_frontier(
        n=8, viscosities=(0.04, 0.02), tau_target=0.05, dt=0.02
    )
    assert isinstance(cert, CascadeFrontierCertificate)
    assert len(cert.peak_debt) == 2
    assert all(cert.saturates)  # bounded debt at fixed Re (diffusive face)
    assert "Clay OPEN" in cert.summary()
    assert isinstance(cert.debt_grows_with_re, bool)


def test_helpers_preserved_for_example_106():
    g = build_torus_graph_3d(6)
    assert g.number_of_nodes() == 6**3
    u, v, w = taylor_green_initial_condition_3d(g, 1.0)
    assert u.shape == (6**3,) == v.shape == w.shape


def test_cascade_moment_hierarchy_energy_is_the_bounded_budget():
    # M_0 (energy) is the conservative-face budget, bounded by Leray; the
    # lambda-moment ladder M_1 (enstrophy), M_2 (palinstrophy) is the wall.
    flow = TNFRNavierStokes(8, 0.02, 1.0)
    m0_init = flow.energy()
    for _ in range(10):
        flow.step(0.02)
    h = cascade_moment_hierarchy(flow)
    assert h["energy_m0"] > 0.0
    assert h["enstrophy_m1"] > 0.0
    assert h["palinstrophy_m2"] > 0.0
    assert h["kmax_eta"] > 0.0
    assert isinstance(h["resolved"], bool)
    # energy is bounded by its initial value (conservative budget / Leray)
    assert h["energy_m0"] <= m0_init + 1e-9
    assert h["m1_over_m0"] > 0.0 and h["m2_over_m1"] > 0.0


def test_moment_ladder_closure_interpolation_bound():
    # The modal Cauchy-Schwarz coupling M1^2 <= M0 M2 is EXACT (s <= 1); the
    # rung read-out is well-formed. (Peak self-consistency P = 2 nu M2 and the
    # growth-phase closure ratio are measured in the benchmark / theory note.)
    flow = TNFRNavierStokes(10, 0.02, 1.0)
    for _ in range(15):
        flow.step(0.02)
    c = moment_ladder_closure(flow)
    assert c["interpolation_ok"]  # s <= 1 exactly (Cauchy-Schwarz)
    assert 0.0 < c["interpolation_saturation"] <= 1.0 + 1e-9
    assert c["palinstrophy_dissipation"] > 0.0
    assert isinstance(c["dissipation_dominates"], bool)
    assert c["closure_ratio"] == c["closure_ratio"]  # finite (not NaN)


def test_flow_coherence_is_the_universal_emergent_geometry_attractor():
    # The flow is read through the ONE universal coherence kernel (nothing NS-
    # specific added): C = 1/(1+|DeltaNFR|) with DeltaNFR = -L_rw.|omega|, and the
    # DeltaNFR=0 fixed-point predicate -- the same attractor as graph/primes/chem.
    flow = TNFRNavierStokes(10, 0.03, 1.0)
    for _ in range(30):
        flow.step(0.02)
    h = flow_coherence(flow)
    assert 0.0 < h["coherence"] <= 1.0
    assert h["coherent"]  # in the coherent band C > 1/(pi+1)
    # exact universal kernel identity C = 1/(1+mean|DeltaNFR|) (no NS formula)
    assert abs(h["coherence"] - 1.0 / (1.0 + h["mean_abs_dnfr"])) < 1e-12
    assert isinstance(h["at_equilibrium"], bool)
    assert isinstance(h["strong"], bool)
