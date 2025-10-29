"""Integration tests for TNFR state projection helpers."""

from __future__ import annotations

import numpy as np

from tnfr.mathematics import BasicStateProjector


def test_basic_state_projector_shapes_and_normalisation() -> None:
    projector = BasicStateProjector()

    dim = 6
    vector = projector(epi=0.42, nu_f=1.75, theta=0.3, dim=dim)

    assert vector.shape == (dim,)
    assert np.iscomplexobj(vector)
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_basic_state_projector_is_deterministic_without_rng() -> None:
    projector = BasicStateProjector()

    first = projector(epi=0.9, nu_f=0.5, theta=1.1, dim=4)
    second = projector(epi=0.9, nu_f=0.5, theta=1.1, dim=4)

    assert np.array_equal(first, second)


def test_basic_state_projector_rng_reproducibility() -> None:
    projector = BasicStateProjector()

    seed = 2024
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    vec1 = projector(epi=0.2, nu_f=1.3, theta=0.7, dim=5, rng=rng1)
    vec2 = projector(epi=0.2, nu_f=1.3, theta=0.7, dim=5, rng=rng2)

    assert np.allclose(vec1, vec2)

    rng3 = np.random.default_rng(seed + 1)
    vec3 = projector(epi=0.2, nu_f=1.3, theta=0.7, dim=5, rng=rng3)

    assert not np.allclose(vec1, vec3)
