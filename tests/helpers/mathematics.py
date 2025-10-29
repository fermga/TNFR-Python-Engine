"""Helpers for assembling mathematical fixtures in integration tests."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from tnfr.mathematics import (
    BasicStateProjector,
    HilbertSpace,
    MathematicalDynamicsEngine,
    NFRValidator,
    build_coherence_operator,
    build_frequency_operator,
)
from tnfr.node import NodeNX
from tnfr.structural import create_nfr


def build_node_with_operators(
    *,
    epi: float = 0.9,
    nu_f: float = 0.8,
    theta: float = 0.1,
    dim: int = 3,
    coherence_value: float = 0.75,
    frequency_value: float | None = 0.5,
    enable_validation: bool = True,
) -> tuple[
    NodeNX,
    HilbertSpace,
    NFRValidator,
]:
    """Return a ``NodeNX`` configured with canonical mathematical operators."""

    G, node_id = create_nfr("math-node", epi=epi, vf=nu_f, theta=theta)
    hilbert = HilbertSpace(dim)
    coherence = build_coherence_operator(np.eye(dim) * coherence_value)
    frequency = (
        build_frequency_operator(np.eye(dim) * frequency_value)
        if frequency_value is not None
        else None
    )
    validator = NFRValidator(
        hilbert,
        coherence,
        coherence_threshold=0.0,
        frequency_operator=frequency,
    )
    node = NodeNX(
        G,
        node_id,
        state_projector=BasicStateProjector(),
        hilbert_space=hilbert,
        coherence_operator=coherence,
        frequency_operator=frequency,
        coherence_threshold=0.0,
        validator=validator if enable_validation else None,
        enable_math_validation=enable_validation,
    )
    return node, hilbert, validator


def make_dynamics_engine(
    generator: Sequence[Sequence[complex]] | np.ndarray,
    hilbert_space: HilbertSpace,
    *,
    atol: float = 1e-9,
    use_scipy: bool | None = None,
) -> MathematicalDynamicsEngine:
    """Instantiate ``MathematicalDynamicsEngine`` with canonical configuration."""

    return MathematicalDynamicsEngine(
        generator,
        hilbert_space=hilbert_space,
        atol=atol,
        use_scipy=use_scipy,
    )
