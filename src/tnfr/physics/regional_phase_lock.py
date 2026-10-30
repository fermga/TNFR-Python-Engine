"""Conditional phase-locked form targets on two unit triangles and one bridge.

Fixed region-constant capacity supplies the configured free angular rates.
The U3 sine law has a cut-load threshold computed before any trajectory. Its
strict principal lock yields a nonlinear phasor source, which the existing
forced-support owner converts into a form target. No solver or autonomous
capacity/support selection is introduced here.

Exact load arithmetic, numerical transcendental estimates, an exactly odd
represented source and the production source are retained separately. The
source graph is never changed; observations do not seal runtime execution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction

import networkx as nx

from .._exact_time import finite_represented_real
from ._cycle_algebra import Vector, dot
from .forced_support import ForcedSupportBalance, derive_forced_support_balance
from .forcing_realization import NonEpiForcingObservation, capture_non_epi_forcing

__all__ = ["RegionalPhaseLockModel", "derive_regional_phase_lock"]

_SCOPE = (
    "conditional fixed unit barbell; supplied positive regional capacities and "
    "omega=nu sine phase law; fresh phase/EPI pressure only; principal lock "
    "under full edge admission; excludes capacity evolution, Gamma, events, "
    "clipping and controllers, whose runtime configuration is not validated; "
    "no autonomous NFR or infinite binary64 stability certificate"
)


@dataclass(frozen=True)
class RegionalPhaseLockModel:
    """Prospective target and separately observed representation defects.

    ``lock_status`` is ``strict``, ``at_capacity`` or ``overloaded`` according
    to the exact ideal half-pi cut load. A configured tighter gate is checked
    independently at the materialized strict target. The angles and phase
    target are binary64 estimates, not transcendental enclosures.

    ``represented_odd_balance`` solves the exact Poisson/gauge problem for
    explicitly represented analytic source estimates, with reflection signs
    inherited from the ideal formula. It does not replace production values.
    ``production_target_capture`` evaluates the actual kernel on a clean graph
    with materialized target form and phase; its forcing discrepancy,
    assembly defect and compatibility drift remain separate observations.
    The initial mean uses H=diag(degree/capacity). It fixes the displayed gauge;
    conservation during phase relaxation additionally requires zero total
    source, as supplied by reflection-antisymmetric phase trajectories. An
    arbitrary phase transient can shift this mean. No endpoint fits the target.
    """

    initial_capture: NonEpiForcingObservation
    regions: tuple[tuple, tuple]
    bridge: tuple
    regional_capacities: Vector
    coupling_strength: Fraction
    epsilon: Fraction
    common_phase_rate: Fraction
    bridge_load: Fraction
    lock_status: str
    effective_phase_gate: Fraction
    initial_weighted_mean: Fraction
    target_angles_estimate: tuple[float, float] | None
    target_phase_estimate: tuple[float, ...] | None
    target_region_gate_admitted: bool | None
    analytic_phase_gradient_estimate: tuple[float, ...] | None
    represented_odd_balance: ForcedSupportBalance | None
    target_epi: Vector | None
    production_target_capture: NonEpiForcingObservation | None
    production_forcing_discrepancy: Vector | None
    production_balance: ForcedSupportBalance | None
    scope: str


def _regions_and_bridge(graph, source, regions):
    if isinstance(regions, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("regions must be an ordered pair of ordered triples")
    rows = tuple(regions)
    if len(rows) != 2:
        raise ValueError("regions must contain two triangles")
    if any(isinstance(row, (str, bytes, bytearray, Mapping, Set)) for row in rows):
        raise TypeError("each region must be an ordered triple")
    rows = tuple(tuple(row) for row in rows)
    flat = tuple(node for row in rows for node in row)
    if (
        any(len(row) != 3 for row in rows)
        or len(set(flat)) != 6
        or set(flat) != set(source.nodes)
    ):
        raise ValueError("regions must partition all six nodes into two triples")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("regional phase lock requires a simple undirected graph")
    index = {node: i for i, node in enumerate(source.nodes)}
    left, right = ({index[node] for node in row} for row in rows)
    conductance = {(i, j): w for i, j, w in source.conductance}
    crossing = tuple((i, j) for i in left for j in right if (i, j) in conductance)
    if len(crossing) != 1:
        raise ValueError("regional phase lock requires exactly one unit bridge")
    bridge = crossing[0]
    expected = {
        (i, j): Fraction(1)
        for region in (left, right)
        for i in region
        for j in region
        if i != j
    }
    expected[bridge] = expected[bridge[::-1]] = Fraction(1)
    if conductance != expected or any(
        set(row) != {j for ii, j in expected if ii == i}
        for i, row in enumerate(source.support_neighbors)
    ):
        raise ValueError("regional phase lock requires unit triangles and unit bridge")
    return rows, (left, right), bridge


def derive_regional_phase_lock(
    graph, regions, *, coupling_strength
) -> RegionalPhaseLockModel:
    """Predict a principal regional lock before evaluating any trajectory.

    Use two ordered three-node regions whose unit complete supports meet in
    one unit bridge. Node labels and insertion order can be arbitrary. Capacity
    is positive and constant within each region, and K is positive. Only phase
    and EPI pressure channels are admitted, with positive effective EPI weight.

    For epsilon=(nu_R-nu_L)/2, cut balance gives sin(beta)=7*epsilon/K;
    the internal outer-to-bridge lag obeys sin(alpha)=2*epsilon/K. Strict load
    gives the principal angles; equality loses strict restoring force and an
    overload cannot balance the fully admitted cut. Targets are returned only
    for strict load. They describe a conditional fixed-capacity ideal model,
    not admission of other configured graph evolution.

    The target form is its H-centered Poisson profile plus the initial H mean.
    That gauge is a prospective choice, not a general asymptotic prediction:
    arbitrary phase transients can change the mean. Reflection-antisymmetric
    phase preparation preserves zero total ideal source and the chosen gauge.
    No stored pressure or measured rate defines the source. Canonical phasor
    evaluation is performed separately on one detached materialized target.
    Edge lengths do not enter this phase/form claim; no potential closure is
    inferred. The caller graph, histories and pressure caches remain untouched.
    """
    from ..operators._phase_gate import (
        resolve_u3_phase_limits,
        resolve_u3_phase_neighbors,
    )

    capture = capture_non_epi_forcing(graph)
    source = capture.snapshot
    ordered_regions, regional_indices, bridge_indices = _regions_and_bridge(
        graph, source, regions
    )
    capacities = tuple(source.capacity[next(iter(row))] for row in regional_indices)
    if any(nu <= 0 for nu in capacities) or any(
        source.capacity[i] != nu
        for row, nu in zip(regional_indices, capacities, strict=True)
        for i in row
    ):
        raise ValueError("capacity must be positive and constant within each region")
    weights = dict(capture.normalized_weights)
    if weights["epi"] <= 0 or weights["vf"] or weights["topo"]:
        raise ValueError("regional lock requires positive EPI and zero vf/topo weights")
    _, coupling = finite_represented_real(coupling_strength, "coupling_strength")
    if coupling <= 0:
        raise ValueError("coupling_strength must be positive")
    hard_gate, gate = resolve_u3_phase_limits(graph.graph, operator_code="UM")
    epsilon = (capacities[1] - capacities[0]) / 2
    omega = sum(capacities) / 2
    load = 7 * epsilon / coupling
    status = (
        "strict"
        if abs(load) < 1
        else ("at_capacity" if abs(load) == 1 else "overloaded")
    )
    metric = tuple(
        Fraction(len(row)) / nu
        for row, nu in zip(source.support_neighbors, source.capacity, strict=True)
    )
    mean = dot(metric, source.epi) / sum(metric)
    angles = phase = gradient = target = None
    admitted = odd_balance = target_capture = discrepancy = production_balance = None
    if status == "strict":
        internal_load = finite_represented_real(
            2 * epsilon / coupling, "internal locking load"
        )[0]
        bridge_load = finite_represented_real(load, "bridge locking load")[0]
        alpha, beta = math.asin(internal_load), math.asin(bridge_load)
        angles = alpha, beta
        outer = alpha / (2 * math.pi)
        central = (
            math.atan2(
                finite_represented_real(3 * epsilon / coupling, "bridge phasor sine")[
                    0
                ],
                2 * math.cos(alpha) + math.cos(beta),
            )
            / math.pi
        )
        left, right = regional_indices
        lb, rb = bridge_indices
        offsets = tuple(
            (
                -beta / 2
                if i == lb
                else (
                    beta / 2
                    if i == rb
                    else (-alpha - beta / 2 if i in left else alpha + beta / 2)
                )
            )
            for i in range(6)
        )
        phase = tuple(math.pi + offset for offset in offsets)
        gradient = tuple(
            (
                central
                if i == lb
                else (-central if i == rb else (outer if i in left else -outer))
            )
            for i in range(6)
        )
        forcing = tuple(
            weights["phase"] * Fraction.from_float(value) for value in gradient
        )
        odd_balance = derive_forced_support_balance(
            source, epi_weight=weights["epi"], forcing=forcing
        )
        target = tuple(mean + value for value in odd_balance.relative_profile)
        # Rebuild only the declared phase/form model: do not copy callbacks,
        # aliases, histories or mutable execution caches from the caller.
        # The target capture retains its own canonical edge insertion order.
        detached = nx.Graph()
        detached.graph.update(
            _dnfr_weights={name: float(value) for name, value in weights.items()},
            vectorized_dnfr=True,
            DELTA_PHI_MAX=hard_gate,
            UM_MAX_PHASE_DIFF=gate,
        )
        for node, x, theta, capacity in zip(
            source.nodes, target, phase, source.capacity, strict=True
        ):
            detached.add_node(
                node,
                EPI=finite_represented_real(x, "target EPI")[0],
                nu_f=float(capacity),
                theta=theta,
                delta_nfr=0.0,
                dEPI=0.0,
            )
        detached.add_edges_from(
            (source.nodes[i], source.nodes[j], {"weight": 1.0})
            for i, j, _ in source.conductance
            if i < j
        )
        phase_by_node = dict(zip(source.nodes, phase, strict=True))
        admitted = all(
            resolve_u3_phase_neighbors(
                detached.graph,
                phase_by_node[node],
                detached.neighbors(node),
                phase_getter=phase_by_node.__getitem__,
                operator_code="UM",
                require_compatible=False,
            ).neighbors
            == tuple(detached.neighbors(node))
            for node in source.nodes
        )
        target_capture = capture_non_epi_forcing(detached)
        discrepancy = tuple(
            actual - reference
            for actual, reference in zip(target_capture.forcing, forcing, strict=True)
        )
        production_balance = derive_forced_support_balance(
            target_capture.snapshot,
            epi_weight=weights["epi"],
            forcing=target_capture.forcing,
        )
    return RegionalPhaseLockModel(
        initial_capture=capture,
        regions=ordered_regions,
        bridge=tuple(source.nodes[i] for i in bridge_indices),
        regional_capacities=capacities,
        coupling_strength=coupling,
        epsilon=epsilon,
        common_phase_rate=omega,
        bridge_load=load,
        lock_status=status,
        effective_phase_gate=Fraction.from_float(gate),
        initial_weighted_mean=mean,
        target_angles_estimate=angles,
        target_phase_estimate=phase,
        target_region_gate_admitted=admitted,
        analytic_phase_gradient_estimate=gradient,
        represented_odd_balance=odd_balance,
        target_epi=target,
        production_target_capture=target_capture,
        production_forcing_discrepancy=discrepancy,
        production_balance=production_balance,
        scope=_SCOPE,
    )
