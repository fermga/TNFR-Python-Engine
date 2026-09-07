r"""Executable integration boundary for the restricted TNFR S16 program.

The certificate in this module intersects four existing results on the same
pair of graph states:

* fixed, connected, symmetric pure-EPI stability with positive heterogeneous
  structural frequency;
* graph-specific reconstruction of EPI from the full structural potential and
  one conserved zero-mode scalar;
* reversible pure-EPI partition closure; and
* the declared fixed-topology structural-state quotient metric.

The two inputs are independently frozen snapshots with exactly the same node
and bare edge support.  Stored pressure and EPI-rate telemetry must agree with
the pure-EPI nodal channel before the joint Boolean can pass.  Passing therefore
certifies only the intersection above at both endpoints.  It does not certify a
trajectory between them, phase dynamics, nonlinear pressure, operator words,
REMESH/nesting, or a topology/support change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR
from ..mathematics.unified_numerical import np
from ._helpers import finite_real_scalar
from .observability import (
    EpiDiffusionReconstructionCertificate,
    epi_diffusion_reconstruction_certificate,
)
from .structural_diffusion import (
    HeterogeneousDiffusionStabilityCertificate,
    structural_diffusion_operator,
    structural_field,
    verify_heterogeneous_diffusion_stability,
)
from .structural_morphism import (
    EpiCoarseGrainingCertificate,
    certify_epi_coarse_graining,
)
from .structural_state_distance import (
    StructuralChannelScales,
    StructuralStateDistanceCertificate,
    fixed_topology_structural_state_distance,
)

__all__ = [
    "CoreResearchIntegrationCertificate",
    "certify_core_research_integration",
]


_S16_SCOPE = (
    "RESTRICTED S16 endpoint certificate for finite connected undirected "
    "simple graphs with fixed node/edge support, positive frozen heterogeneous "
    "capacity, continuous-time pure-EPI diffusion, a graph-specific full-"
    "potential reconstruction observer, a reversible partition quotient, and "
    "a declared scaled structural-state quotient metric. The Boolean composes "
    "independently frozen endpoint results; it does not certify a path between "
    "them. Phase dynamics, nonlinear pressure and operators, REMESH/nesting, "
    "changing support/topology, and history geometry remain OPEN."
)


@dataclass(frozen=True, slots=True)
class CoreResearchIntegrationCertificate:
    """Joint numerical status and every constituent S16 certificate."""

    nodes: tuple[Any, ...]
    partition: tuple[tuple[Any, ...], ...]
    left_stability: HeterogeneousDiffusionStabilityCertificate
    right_stability: HeterogeneousDiffusionStabilityCertificate
    left_reconstruction: EpiDiffusionReconstructionCertificate
    right_reconstruction: EpiDiffusionReconstructionCertificate
    left_coarse_graining: EpiCoarseGrainingCertificate
    right_coarse_graining: EpiCoarseGrainingCertificate
    structural_distance: StructuralStateDistanceCertificate
    left_pure_epi_pressure_residual_linf: float
    right_pure_epi_pressure_residual_linf: float
    left_scaled_pure_epi_pressure_residual: float
    right_scaled_pure_epi_pressure_residual: float
    left_scaled_nodal_equation_residual: float
    right_scaled_nodal_equation_residual: float
    numerical_conditions: tuple[tuple[str, bool], ...]
    joint_numerical_conditions_pass: bool
    tolerance: float
    scope: str

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Names of every condition blocking the restricted joint result."""
        return tuple(name for name, passed in self.numerical_conditions if not passed)


def _validated_tolerance(value: float) -> float:
    try:
        result = finite_real_scalar(value, "tolerance")
    except ValueError as exc:
        raise ValueError(
            "tolerance must be a finite real in the open interval (0, 1)"
        ) from exc

    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(
            "tolerance must be a finite real in the open interval (0, 1)"
        )
    return result


def _undirected_edge_support(graph: nx.Graph) -> frozenset[frozenset[Any]]:
    return frozenset(frozenset((source, target)) for source, target in graph.edges())


def _validate_shared_support(left: Any, right: Any) -> tuple[Any, ...]:
    if not isinstance(left, nx.Graph) or not isinstance(right, nx.Graph):
        raise TypeError("left and right must be NetworkX graph instances")
    if left.is_directed() or right.is_directed():
        raise ValueError("S16 integration requires two undirected graph states")
    if left.is_multigraph() or right.is_multigraph():
        raise ValueError("S16 integration requires two simple graph states")

    nodes = tuple(left)
    if set(nodes) != set(right):
        raise ValueError("S16 integration requires exactly the same node support")
    if _undirected_edge_support(left) != _undirected_edge_support(right):
        raise ValueError("S16 integration requires exactly the same bare edge support")
    return nodes


def _required_pressure(graph: nx.Graph, nodes: Sequence[Any]) -> np.ndarray:
    missing = object()
    values: list[float] = []
    for node in nodes:
        raw = get_attr(
            graph.nodes[node],
            ALIAS_DNFR,
            missing,
            conv=lambda item: item,
            strict=True,
        )
        if raw is missing:
            raise ValueError(f"node {node!r} is missing required DeltaNFR")
        values.append(finite_real_scalar(raw, f"DeltaNFR at node {node!r}"))
    return np.asarray(values, dtype=float)


def _pure_epi_pressure_residual(graph: nx.Graph) -> float:
    nodes, laplacian = structural_diffusion_operator(graph)
    field = structural_field(graph, nodes)
    stored_pressure = _required_pressure(graph, nodes)
    try:
        with np.errstate(over="raise", invalid="raise"):
            expected_pressure = -(laplacian @ field)
            residual = stored_pressure - expected_pressure
    except FloatingPointError as exc:
        raise ValueError(
            "pure-EPI pressure residual exceeds floating-point range"
        ) from exc
    if not np.all(np.isfinite(expected_pressure)) or not np.all(np.isfinite(residual)):
        raise ValueError("pure-EPI pressure residual exceeds floating-point range")
    return float(np.max(np.abs(residual), initial=0.0))


def _scaled_residual(value: float, scale: float, name: str) -> float:
    result = value / float(scale)
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds finite scaled floating-point range")
    return result


def certify_core_research_integration(
    left: nx.Graph,
    right: nx.Graph,
    partition: Iterable[Iterable[Any]],
    *,
    scales: StructuralChannelScales,
    tolerance: float = 1e-10,
    node_label_attributes: Sequence[str] = (),
    edge_label_attributes: Sequence[str] = (),
) -> CoreResearchIntegrationCertificate:
    r"""Compose the restricted S1/S2/S3/S4/S8/S9/S14 evidence chain for S16.

    ``left`` and ``right`` must have identical node ids and bare undirected edge
    support.  Conductance, structural length, EPI, structural frequency, phase,
    pressure and EPI rate may differ.  ``partition`` is materialized once and
    applied to both states.  The same relative ``tolerance`` is passed to the
    stability, reconstruction and quotient certificates and is also used for
    the scaled pure-EPI pressure and nodal-equation residuals.

    These are independently frozen endpoints. Passing does not certify an
    intervening trajectory or persistence between them.

    A failed rank, closure, balance or state-consistency condition is returned
    as a false entry in ``numerical_conditions`` and prevents
    ``joint_numerical_conditions_pass``.  Inputs outside the shared structural
    hypotheses, such as different support or a directed graph, are rejected.
    """

    relative_tolerance = _validated_tolerance(tolerance)
    nodes = _validate_shared_support(left, right)
    blocks = tuple(tuple(block) for block in partition)

    distance = fixed_topology_structural_state_distance(
        left,
        right,
        scales=scales,
        node_label_attributes=node_label_attributes,
        edge_label_attributes=edge_label_attributes,
    )
    left_stability = verify_heterogeneous_diffusion_stability(
        left, tolerance=relative_tolerance
    )
    right_stability = verify_heterogeneous_diffusion_stability(
        right, tolerance=relative_tolerance
    )
    left_reconstruction = epi_diffusion_reconstruction_certificate(
        left, tolerance=relative_tolerance
    )
    right_reconstruction = epi_diffusion_reconstruction_certificate(
        right, tolerance=relative_tolerance
    )
    left_coarse = certify_epi_coarse_graining(
        left, blocks, tolerance=relative_tolerance
    )
    right_coarse = certify_epi_coarse_graining(
        right, blocks, tolerance=relative_tolerance
    )

    left_pressure_residual = _pure_epi_pressure_residual(left)
    right_pressure_residual = _pure_epi_pressure_residual(right)
    left_scaled_pressure = _scaled_residual(
        left_pressure_residual, scales.pressure, "left pure-EPI pressure residual"
    )
    right_scaled_pressure = _scaled_residual(
        right_pressure_residual,
        scales.pressure,
        "right pure-EPI pressure residual",
    )
    left_scaled_nodal = _scaled_residual(
        distance.left_nodal_equation_residual_linf,
        scales.epi_rate,
        "left nodal-equation residual",
    )
    right_scaled_nodal = _scaled_residual(
        distance.right_nodal_equation_residual_linf,
        scales.epi_rate,
        "right nodal-equation residual",
    )

    conditions = (
        ("left_stability", bool(left_stability.is_certified)),
        ("right_stability", bool(right_stability.is_certified)),
        (
            "left_absolute_epi_reconstruction",
            bool(left_reconstruction.reconstructs_absolute_epi),
        ),
        (
            "right_absolute_epi_reconstruction",
            bool(right_reconstruction.reconstructs_absolute_epi),
        ),
        ("left_partition_closure", bool(left_coarse.nodal_closure_within_tolerance)),
        (
            "right_partition_closure",
            bool(right_coarse.nodal_closure_within_tolerance),
        ),
        (
            "left_quotient_intertwining",
            bool(left_coarse.morphism.intertwines_within_tolerance),
        ),
        (
            "right_quotient_intertwining",
            bool(right_coarse.morphism.intertwines_within_tolerance),
        ),
        (
            "left_nodal_flow_transport",
            bool(left_coarse.morphism.nodal_flow_transport_within_tolerance),
        ),
        (
            "right_nodal_flow_transport",
            bool(right_coarse.morphism.nodal_flow_transport_within_tolerance),
        ),
        (
            "fixed_topology_state_metric",
            bool(distance.exact_metric_on_isomorphism_classes),
        ),
        (
            "left_pure_epi_pressure_consistency",
            left_scaled_pressure <= relative_tolerance,
        ),
        (
            "right_pure_epi_pressure_consistency",
            right_scaled_pressure <= relative_tolerance,
        ),
        (
            "left_nodal_equation_consistency",
            left_scaled_nodal <= relative_tolerance,
        ),
        (
            "right_nodal_equation_consistency",
            right_scaled_nodal <= relative_tolerance,
        ),
    )

    return CoreResearchIntegrationCertificate(
        nodes=nodes,
        partition=blocks,
        left_stability=left_stability,
        right_stability=right_stability,
        left_reconstruction=left_reconstruction,
        right_reconstruction=right_reconstruction,
        left_coarse_graining=left_coarse,
        right_coarse_graining=right_coarse,
        structural_distance=distance,
        left_pure_epi_pressure_residual_linf=left_pressure_residual,
        right_pure_epi_pressure_residual_linf=right_pressure_residual,
        left_scaled_pure_epi_pressure_residual=left_scaled_pressure,
        right_scaled_pure_epi_pressure_residual=right_scaled_pressure,
        left_scaled_nodal_equation_residual=left_scaled_nodal,
        right_scaled_nodal_equation_residual=right_scaled_nodal,
        numerical_conditions=conditions,
        joint_numerical_conditions_pass=all(passed for _, passed in conditions),
        tolerance=relative_tolerance,
        scope=_S16_SCOPE,
    )
