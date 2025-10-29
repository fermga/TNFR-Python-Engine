"""Integration tests for ΔNFR generator construction."""
from __future__ import annotations

import numpy as np
import pytest

from tnfr.mathematics import build_delta_nfr


def _expected_ring_adjacency(dim: int) -> np.ndarray:
    adjacency = np.zeros((dim, dim), dtype=float)
    if dim == 1:
        return adjacency

    indices = np.arange(dim)
    adjacency[indices, (indices + 1) % dim] = 1.0
    adjacency[(indices + 1) % dim, indices] = 1.0
    return adjacency


def test_build_delta_nfr_returns_hermitian_operators():
    for topology in ("laplacian", "adjacency"):
        operator = build_delta_nfr(5, topology=topology)
        assert operator.shape == (5, 5)
        assert np.allclose(operator, operator.conj().T)


def test_build_delta_nfr_scaling_matches_ring_baselines():
    dim = 4
    nu_f = 1.5
    scale = 0.25

    expected_adjacency = _expected_ring_adjacency(dim) * (nu_f * scale)
    adjacency = build_delta_nfr(dim, topology="adjacency", nu_f=nu_f, scale=scale)
    assert np.allclose(adjacency, expected_adjacency)

    ring_laplacian = np.diag([2.0] * dim) - _expected_ring_adjacency(dim)
    expected_laplacian = ring_laplacian * (nu_f * scale)
    laplacian = build_delta_nfr(dim, topology="laplacian", nu_f=nu_f, scale=scale)
    assert np.allclose(laplacian, expected_laplacian)


def test_build_delta_nfr_reproducibility_with_seeded_noise():
    dim = 6
    seeded_first = build_delta_nfr(dim, rng=np.random.default_rng(2024))
    seeded_second = build_delta_nfr(dim, rng=np.random.default_rng(2024))
    unseeded = build_delta_nfr(dim)
    divergent = build_delta_nfr(dim, rng=np.random.default_rng(2025))

    assert np.allclose(seeded_first, seeded_second)
    assert np.allclose(seeded_first, seeded_first.conj().T)
    assert np.allclose(divergent, divergent.conj().T)
    assert not np.allclose(seeded_first, divergent)
    assert np.allclose(unseeded, unseeded.conj().T)


def test_build_delta_nfr_input_validation():
    with pytest.raises(ValueError):
        build_delta_nfr(0)

    with pytest.raises(ValueError):
        build_delta_nfr(2, topology="unknown")
