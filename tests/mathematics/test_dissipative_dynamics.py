from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics import (
    ContractiveDynamicsEngine,
    HilbertSpace,
    build_delta_nfr,
    build_lindblad_delta_nfr,
    make_coherence_operator,
)


def _steady_state_from_generator(generator: np.ndarray, dim: int) -> np.ndarray:
    evals, evecs = np.linalg.eig(generator)
    index = int(np.argmin(np.abs(evals)))
    steady_vector = evecs[:, index]
    density = steady_vector.reshape((dim, dim), order="F")
    density = 0.5 * (density + density.conj().T)
    trace = np.trace(density)
    if np.isclose(trace, 0.0):
        raise ValueError("Steady state trace collapsed; generator lacks a stationary density.")
    return density / trace


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    diff = left - right
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * float(np.sum(singular_values))


@pytest.fixture(scope="module")
def hilbert_qubit() -> HilbertSpace:
    return HilbertSpace(2)


@pytest.fixture(scope="module")
def hilbert_qutrit() -> HilbertSpace:
    return HilbertSpace(3)


def test_lindblad_generator_preserves_trace(hilbert_qubit: HilbertSpace) -> None:
    gamma = 0.35
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    generator = build_lindblad_delta_nfr(
        hamiltonian=np.zeros((2, 2), dtype=np.complex128),
        collapse_operators=[math.sqrt(gamma) * lowering],
        nu_f=1.0,
    )

    identity_vector = np.eye(hilbert_qubit.dimension, dtype=np.complex128).reshape(-1, order="F")
    assert np.allclose(
        identity_vector.conj().T @ generator, np.zeros_like(identity_vector), atol=1e-9
    )

    eigenvalues = np.linalg.eigvals(generator)
    assert np.max(eigenvalues.real) <= 1e-9


def test_lindblad_generator_rejects_dim_mismatch_with_hamiltonian(
    hilbert_qubit: HilbertSpace,
) -> None:
    dim = hilbert_qubit.dimension
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    with pytest.raises(ValueError, match="Provided dim is inconsistent"):
        build_lindblad_delta_nfr(
            hamiltonian=hamiltonian,
            collapse_operators=[np.zeros_like(hamiltonian)],
            dim=dim + 1,
        )


def test_lindblad_generator_rejects_dim_mismatch_with_collapse_operator(
    hilbert_qubit: HilbertSpace,
) -> None:
    dim = hilbert_qubit.dimension
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="Provided dim is inconsistent"):
        build_lindblad_delta_nfr(
            collapse_operators=[lowering],
            dim=dim + 1,
        )


def test_contractive_engine_preserves_trace_and_contractivity(
    hilbert_qubit: HilbertSpace,
) -> None:
    gamma = 0.4
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    generator = build_lindblad_delta_nfr(
        collapse_operators=[math.sqrt(gamma) * lowering],
        dim=hilbert_qubit.dimension,
    )

    engine = ContractiveDynamicsEngine(generator, hilbert_qubit)

    initial = np.array([[0.6, 0.25], [0.25, 0.4]], dtype=np.complex128)
    initial /= np.trace(initial)
    steady = _steady_state_from_generator(generator, hilbert_qubit.dimension)
    baseline = _trace_distance(initial, steady)
    step = engine.step(initial, dt=0.5)

    assert np.isclose(np.trace(step), 1.0, atol=1e-8)
    assert _trace_distance(step, steady) <= baseline + 1e-8
    assert np.isfinite(engine.last_contractivity_gap)


def _amplitude_damping_exact(
    density: np.ndarray,
    *,
    gamma: float,
    time: float,
) -> np.ndarray:
    e_gamma = math.exp(-gamma * time)
    sqrt_e = math.exp(-0.5 * gamma * time)

    rho00 = float(density[0, 0].real)
    rho11 = float(density[1, 1].real)
    rho01 = density[0, 1]

    evolved = np.zeros_like(density, dtype=np.complex128)
    evolved[0, 0] = rho00 + (1.0 - e_gamma) * rho11
    evolved[1, 1] = e_gamma * rho11
    evolved[0, 1] = sqrt_e * rho01
    evolved[1, 0] = np.conjugate(evolved[0, 1])
    return evolved


def test_contractive_engine_matches_qubit_ground_truth(
    hilbert_qubit: HilbertSpace,
) -> None:
    gamma = 0.7
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    generator = build_lindblad_delta_nfr(
        collapse_operators=[math.sqrt(gamma) * lowering],
        dim=hilbert_qubit.dimension,
    )
    engine = ContractiveDynamicsEngine(generator, hilbert_qubit)

    initial = np.array([[0.55, 0.2], [0.2, 0.45]], dtype=np.complex128)
    initial /= np.trace(initial)
    dt = 0.6

    evolved = engine.step(initial, dt=dt)
    expected = _amplitude_damping_exact(initial, gamma=gamma, time=dt)

    assert np.allclose(evolved, expected, atol=5e-7)


def _pure_dephasing_exact(
    density: np.ndarray,
    *,
    rates: np.ndarray,
    time: float,
) -> np.ndarray:
    dim = density.shape[0]
    evolved = np.array(density, dtype=np.complex128, copy=True)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            decay = math.exp(-0.5 * rates[i, j] * time)
            evolved[i, j] *= decay
    return evolved


def test_qutrit_pure_dephasing_matches_ground_truth(
    hilbert_qutrit: HilbertSpace,
) -> None:
    gamma = 0.25
    eigenvalues = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    projector = np.diag(eigenvalues)
    collapse = math.sqrt(gamma) * projector

    generator = build_lindblad_delta_nfr(
        collapse_operators=[collapse],
        dim=hilbert_qutrit.dimension,
    )
    engine = ContractiveDynamicsEngine(generator, hilbert_qutrit)

    initial = np.ones((3, 3), dtype=np.complex128) / 3.0
    dt = 0.8
    evolved = engine.step(initial, dt=dt)

    rate_matrix = (eigenvalues[:, None] - eigenvalues[None, :]) ** 2 * gamma
    expected = _pure_dephasing_exact(initial, rates=rate_matrix, time=dt)

    assert np.allclose(evolved, expected, atol=5e-7)

    steady = _steady_state_from_generator(generator, hilbert_qutrit.dimension)
    trajectory = engine.evolve(initial, steps=3, dt=dt)
    distances = [_trace_distance(state, steady) for state in trajectory]
    assert all(distances[k] >= distances[k + 1] - 1e-7 for k in range(len(distances) - 1))


def test_unitary_generator_remains_available(hilbert_qubit: HilbertSpace) -> None:
    generator = build_delta_nfr(hilbert_qubit.dimension)
    coherence = make_coherence_operator(
        hilbert_qubit.dimension, spectrum=np.ones(hilbert_qubit.dimension)
    )
    assert coherence.matrix.shape == (hilbert_qubit.dimension, hilbert_qubit.dimension)


def test_defective_generator_requires_scipy(hilbert_qubit: HilbertSpace) -> None:
    pytest.importorskip("scipy.linalg")

    dim = hilbert_qubit.dimension
    size = dim * dim
    generator = np.zeros((size, size), dtype=np.complex128)
    generator[0, 0] = generator[1, 1] = generator[2, 2] = generator[3, 3] = -0.5
    generator[0, 1] = 1.0
    generator[2, 3] = 1.0

    with pytest.raises(ValueError, match="not diagonalizable"):
        ContractiveDynamicsEngine(generator, hilbert_qubit, use_scipy=False)

    engine = ContractiveDynamicsEngine(generator, hilbert_qubit, use_scipy=True)
    initial = np.eye(dim, dtype=np.complex128) / dim
    evolved = engine.step(initial, dt=0.25)

    assert evolved.shape == (dim, dim)
    assert np.all(np.isfinite(evolved))
