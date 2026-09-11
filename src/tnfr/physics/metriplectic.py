r"""Restricted dissipative-symplectic product for TNFR graph dynamics.

This module places the specified harmonic substrate and fixed symmetric pure-EPI
diffusion in one block equation

``Xdot = J grad(H_sub) - G grad(E_D)``.

The construction is an exact metriplectic-style direct product: ``J`` is
antisymmetric, ``G`` is positive semidefinite, ``J grad(E_D)=0`` and
``G grad(H_sub)=0``.  Consequently the substrate Hamiltonian is conserved and
the EPI Dirichlet functional is nonincreasing.  The zero cross blocks are also
the honest boundary: this baseline does not derive a physical coupling between
the two sectors or classify the 13 nonlinear operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..mathematics.unified_numerical import np
from ._conductance import read_conductance
from ._helpers import finite_real_scalar
from .symplectic_substrate import (
    extract_phase_space_point,
    hamiltonian_vector_field,
    substrate_hamiltonian,
    symplectic_form_matrix,
)

__all__ = [
    "MetriplecticProductCertificate",
    "verify_metriplectic_product",
]


@dataclass(frozen=True)
class MetriplecticProductCertificate:
    """Algebraic and balance checks for the restricted product system."""

    nodes: tuple
    state_dimension: int
    poisson_tensor: Any
    dissipative_tensor: Any
    state_velocity: Any
    hamiltonian: float
    dirichlet_functional: float
    hamiltonian_derivative: float
    dirichlet_derivative: float
    poisson_antisymmetry_residual: float
    minimum_dissipative_eigenvalue: float
    poisson_dissipative_degeneracy: float
    metric_hamiltonian_degeneracy: float
    substrate_velocity_residual: float
    epi_velocity_residual: float
    stored_pressure_consistency_residual: float
    stored_pressure_matches_epi_channel: bool
    is_decoupled_metriplectic_bridge: bool
    couples_sectors: bool
    scope: str


def _real_attribute(graph: Any, node: Any, aliases: Any, name: str) -> float:
    """Read a real nodal attribute without accepting logical zero/one values."""
    raw = get_attr(
        graph.nodes[node], aliases, 0.0, strict=True, conv=lambda value: value
    )
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
        raise ValueError(f"Metriplectic product requires finite real {name}")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"Metriplectic product requires finite real {name}")
    return value


def verify_metriplectic_product(
    graph: Any, *, tolerance: float = 1e-10,
) -> MetriplecticProductCertificate:
    r"""Verify the exact direct-product bridge on one graph state.

    The first ``4N`` coordinates are the canonical substrate point ``z`` with
    ``H_sub=||z||^2/2``.  The final ``N`` coordinates are EPI ``x`` with
    ``E_D=x^T Bx/2``.  For mobility ``M=diag(nu_f/d)`` the block tensors are

    ``J=diag(J_sub,0)`` and ``G=diag(0,M)``.

    Their vector field is exactly the harmonic substrate flow together with
    ``xdot=-M Bx=nu_f DeltaNFR_epi``.  The certificate is restricted to fixed
    symmetric conductance, positive capacity and scalar EPI.
    """
    try:
        tolerance_value = finite_real_scalar(tolerance, "tolerance")
    except ValueError as exc:
        raise ValueError("tolerance must be finite and positive") from exc
    if tolerance_value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    try:
        node_count = len(graph)
    except TypeError as exc:
        raise TypeError("graph must be a finite graph-like object") from exc
    if node_count < 2:
        raise ValueError("Metriplectic product requires at least two supported nodes")
    input_nodes = frozenset(graph.nodes)
    point = extract_phase_space_point(graph)
    nodes = point.nodes
    if len(nodes) != node_count or frozenset(nodes) != input_nodes:
        raise RuntimeError("graph node set changed during metriplectic capture")
    frequency = np.asarray(
        [_real_attribute(graph, node, ALIAS_VF, "capacity") for node in nodes],
        dtype=float,
    )
    epi = np.asarray(
        [
            _real_attribute(graph, node, ALIAS_EPI, "scalar EPI")
            for node in nodes
        ],
        dtype=float,
    )
    stored_pressure = np.asarray(
        [
            _real_attribute(graph, node, ALIAS_DNFR, "DeltaNFR")
            for node in nodes
        ],
        dtype=float,
    )
    for node in nodes:
        _real_attribute(graph, node, ALIAS_THETA, "phase")
    conductance = read_conductance(graph, list(nodes), symmetric=True)
    adjacency = conductance.dense()
    strength = conductance.strength
    if np.any(strength <= 0.0):
        raise ValueError("Metriplectic product requires positive row strength")
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("Metriplectic product requires positive finite capacity")
    if not np.all(np.isfinite(epi)):
        raise ValueError("Metriplectic product requires finite scalar EPI")

    n = len(nodes)
    substrate_dimension = 4 * n
    total_dimension = substrate_dimension + n
    poisson = np.zeros((total_dimension, total_dimension), dtype=float)
    poisson[:substrate_dimension, :substrate_dimension] = symplectic_form_matrix(n)
    dissipative = np.zeros_like(poisson)
    z = point.to_vector()
    if not np.all(np.isfinite(z)):
        raise ValueError("Metriplectic product requires finite substrate fields")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mobility = frequency / strength
            dissipative[substrate_dimension:, substrate_dimension:] = np.diag(
                mobility
            )
            laplacian = np.diag(strength) - adjacency
            laplacian_epi = laplacian @ epi
            grad_h = np.concatenate((z, np.zeros(n, dtype=float)))
            grad_v = np.concatenate(
                (np.zeros(substrate_dimension, dtype=float), laplacian_epi)
            )
            velocity = poisson @ grad_h - dissipative @ grad_v
            expected_substrate = hamiltonian_vector_field(point)
            expected_epi = -mobility * laplacian_epi
            expected_pressure = -laplacian_epi / strength

            hamiltonian_derivative = float(grad_h @ velocity)
            dirichlet_derivative = float(grad_v @ velocity)
            antisymmetry = float(np.linalg.norm(poisson.T + poisson, 2))
            minimum_metric_eigenvalue = min(0.0, float(np.min(mobility)))
            poisson_degeneracy = float(np.linalg.norm(poisson @ grad_v))
            metric_degeneracy = float(np.linalg.norm(dissipative @ grad_h))
            substrate_residual = float(
                np.linalg.norm(velocity[:substrate_dimension] - expected_substrate)
            )
            epi_residual = float(
                np.linalg.norm(velocity[substrate_dimension:] - expected_epi)
            )
            pressure_residual = float(
                np.linalg.norm(stored_pressure - expected_pressure)
            )
            pressure_scale = max(
                1.0,
                float(np.linalg.norm(stored_pressure)),
                float(np.linalg.norm(expected_pressure)),
            )
            velocity_norm = float(np.linalg.norm(velocity))
            hamiltonian = substrate_hamiltonian(point)
            dirichlet_functional = float(0.5 * epi @ laplacian_epi)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            "Metriplectic product exceeds finite floating-point range"
        ) from exc

    finite_arrays = (
        mobility,
        laplacian,
        laplacian_epi,
        grad_h,
        grad_v,
        velocity,
        expected_substrate,
        expected_epi,
        expected_pressure,
    )
    finite_scalars = (
        hamiltonian_derivative,
        dirichlet_derivative,
        antisymmetry,
        minimum_metric_eigenvalue,
        poisson_degeneracy,
        metric_degeneracy,
        substrate_residual,
        epi_residual,
        pressure_residual,
        pressure_scale,
        velocity_norm,
        hamiltonian,
        dirichlet_functional,
    )
    if not all(np.all(np.isfinite(array)) for array in finite_arrays) or not all(
        np.isfinite(value) for value in finite_scalars
    ):
        raise ValueError("Metriplectic product exceeds finite floating-point range")

    scale = max(1.0, velocity_norm, abs(dirichlet_derivative))
    valid = bool(
        antisymmetry <= tolerance_value
        and minimum_metric_eigenvalue >= -tolerance_value
        and poisson_degeneracy <= tolerance_value * scale
        and metric_degeneracy <= tolerance_value * scale
        and abs(hamiltonian_derivative) <= tolerance_value * scale
        and dirichlet_derivative <= tolerance_value * scale
        and substrate_residual <= tolerance_value * scale
        and epi_residual <= tolerance_value * scale
    )
    return MetriplecticProductCertificate(
        nodes=nodes,
        state_dimension=total_dimension,
        poisson_tensor=poisson,
        dissipative_tensor=dissipative,
        state_velocity=velocity,
        hamiltonian=hamiltonian,
        dirichlet_functional=dirichlet_functional,
        hamiltonian_derivative=hamiltonian_derivative,
        dirichlet_derivative=dirichlet_derivative,
        poisson_antisymmetry_residual=antisymmetry,
        minimum_dissipative_eigenvalue=minimum_metric_eigenvalue,
        poisson_dissipative_degeneracy=poisson_degeneracy,
        metric_hamiltonian_degeneracy=metric_degeneracy,
        substrate_velocity_residual=substrate_residual,
        epi_velocity_residual=epi_residual,
        stored_pressure_consistency_residual=pressure_residual,
        stored_pressure_matches_epi_channel=(
            pressure_residual <= tolerance_value * pressure_scale
        ),
        is_decoupled_metriplectic_bridge=valid,
        couples_sectors=False,
        scope=(
            "fixed symmetric positive-capacity harmonic-substrate x pure-EPI "
            "direct product; stored DeltaNFR consistency reported separately"
        ),
    )
