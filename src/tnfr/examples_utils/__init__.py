"""Utilities for examples and notebooks (shared helpers).

This subpackage provides small, non-canonical helpers used in examples/notebooks
to demonstrate telemetry and sequential checks without mutating EPI.
"""

from .demo_sequences import (
    apply_synthetic_activation_sequence,
    build_ws_graph_with_seed,
    build_radial_atom_graph,
    build_element_radial_graph,
    build_diatomic_molecule_graph,
    build_triatomic_molecule_graph,
)

__all__ = [
    "apply_synthetic_activation_sequence",
    "build_ws_graph_with_seed",
    "build_radial_atom_graph",
    "build_element_radial_graph",
    "build_diatomic_molecule_graph",
    "build_triatomic_molecule_graph",
]
