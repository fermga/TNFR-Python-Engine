"""Helpers for assembling mathematical fixtures in integration tests."""
from __future__ import annotations

import numpy as np

from tnfr.mathematics import (
    BasicStateProjector,
    HilbertSpace,
    NFRValidator,
    build_coherence_operator,
    build_frequency_operator,
)
from tnfr.node import NodeNX
from tnfr.structural import create_nfr


def build_node_with_operators(
    *,
    dim: int = 3,
    coherence_value: float = 0.75,
    frequency_value: float = 0.5,
    enable_validation: bool = True,
) -> tuple[
    NodeNX,
    HilbertSpace,
    NFRValidator,
]:
    """Return a ``NodeNX`` configured with canonical mathematical operators."""

    G, node_id = create_nfr("math-node", epi=0.9, vf=0.8, theta=0.1)
    hilbert = HilbertSpace(dim)
    coherence = build_coherence_operator(np.eye(dim) * coherence_value)
    frequency = build_frequency_operator(np.eye(dim) * frequency_value)
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
