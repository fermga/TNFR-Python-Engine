import numpy as np


def test_hypercharge_relation_Q_equals_T3_plus_Y_over_2():
    # (Q, T3, Y) tuples for representative SM particles
    cases = [
        (-1.0, -0.5, -1.0),  # electron (e-)
        (0.0, +0.5, -1.0),   # electron neutrino (nu_e)
        (+2.0/3.0, +0.5, +1.0/3.0),  # up quark (u)
        (-1.0/3.0, -0.5, +1.0/3.0),  # down quark (d)
    ]
    for Q, T3, Y in cases:
        assert np.isclose(Q, T3 + 0.5 * Y)


def test_baryon_and_lepton_numbers_consistency():
    # Baryon numbers: quark=+1/3, anti-quark=-1/3
    # Lepton numbers: lepton=+1, anti-lepton=-1
    baryon = {"u": 1.0/3.0, "d": 1.0/3.0, "u_bar": -1.0/3.0, "d_bar": -1.0/3.0}
    lepton = {"e-": 1.0, "nu_e": 1.0, "e+": -1.0, "nu_e_bar": -1.0}
    # check signs and magnitudes
    assert baryon["u"] == 1.0/3.0 and baryon["u_bar"] == -1.0/3.0
    assert lepton["e-"] == 1.0 and lepton["e+"] == -1.0


def _ckm_matrix_wolfenstein(lambda_=0.22650, A=0.790, rho=0.141, eta=0.357):
    # Wolfenstein parameterization up to O(lambda^2)
    lam = lambda_
    lam2 = lam * lam
    Vud = 1 - 0.5 * lam2
    Vus = lam
    Vub = A * lam**3 * (rho - 1j * eta)
    Vcd = -lam
    Vcs = 1 - 0.5 * lam2
    Vcb = A * lam**2
    Vtd = A * lam**3 * (1 - rho - 1j * eta)
    Vts = -A * lam**2
    Vtb = 1.0
    V = np.array(
        [
            [Vud, Vus, Vub],
            [Vcd, Vcs, Vcb],
            [Vtd, Vts, Vtb],
        ],
        dtype=complex,
    )
    return V


def _unitarity_error(U: np.ndarray) -> float:
    ident = np.eye(U.shape[0], dtype=complex)
    delta = U.conj().T @ U - ident
    return float(np.linalg.norm(delta, ord=2))


def test_ckm_unitarity_within_tolerance():
    V = _ckm_matrix_wolfenstein()
    err = _unitarity_error(V)
    # Empirically tolerant threshold for O(lambda^2)
    assert err < 2e-2


def test_pmns_unitarity_identity_matrix():
    # Simple baseline: identity PMNS is exactly unitary
    U = np.eye(3, dtype=float)
    err = _unitarity_error(U)
    assert np.isclose(err, 0.0)
