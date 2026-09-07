r"""Time-resolved certificate for the restricted TNFR S16 evidence chain.

The endpoint certificate in :mod:`tnfr.physics.core_research_integration`
intersects fixed-state stability, observability, quotient closure, and
structural-state geometry.  This module adds four pieces of evidence that are
meaningful only for an ordered sample path:

* every adjacent pair passes that endpoint certificate;
* every observed EPI increment agrees, on persistent node identifiers, with a
  declared forward-Euler step of the stored nodal vector field;
* every declared step is inside the frozen operator's explicit-Euler modal
  stability region; and
* the common quadratic Lyapunov value observed at the supplied samples does
  not increase beyond the declared scaled tolerance.

A common-Lyapunov certificate for the finite sampled transport family is also
required for the joint temporal Boolean.  It covers continuous pure-EPI
diffusion under arbitrary piecewise-constant switching among those sampled
regimes.  It does not observe unsampled regimes or turn the discrete residual
check into a continuous-time path proof.

The optional coarse/fine comparison uses persistent node identifiers for its
decisive EPI error.  The relabeling-quotient structural distance is retained as
a diagnostic only, because independently minimizing an isomorphism at every
time could hide an exchange of nodal identity.  Agreement on one pair of
meshes is not a convergence or order-of-accuracy theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Iterable, Sequence

import networkx as nx

from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..mathematics.unified_numerical import np
from ._helpers import finite_real_scalar
from .core_research_integration import (
    CoreResearchIntegrationCertificate,
    certify_core_research_integration,
)
from .structural_diffusion import (
    EulerRelaxationWindowDiagnostic,
    SwitchingDiffusionStabilityCertificate,
    diagnose_euler_relaxation_window,
    structural_diffusion_operator,
    verify_switching_diffusion_stability,
)
from .structural_state_distance import (
    StructuralChannelScales,
    StructuralStateDistanceCertificate,
    fixed_topology_structural_state_distance,
)

__all__ = [
    "CoreResearchTrajectoryIntervalCertificate",
    "CoreResearchTrajectoryCertificate",
    "CoreResearchRefinementSample",
    "CoreResearchRefinementComparison",
    "certify_core_research_trajectory",
    "compare_core_research_trajectory_refinement",
]


_FORWARD_EULER = "forward_euler"
_TRAJECTORY_SCOPE = (
    "RESTRICTED time-resolved S16 certificate for at least two finite, "
    "strictly time-ordered snapshots of finite connected undirected simple "
    "graphs with persistent node identifiers and fixed bare edge support. "
    "Every adjacent endpoint pair must pass the S16 certificate, every "
    "observed EPI increment must satisfy the declared forward-Euler update "
    "within the caller's scaled tolerance, every step must satisfy the frozen "
    "operator's forward-Euler modal stability condition under a separately "
    "reported relative spectral cutoff, the observed common "
    "quadratic's local increments and cumulative positive variation must stay "
    "within the path tolerance, and the finite sampled "
    "pure-EPI transport family must have the exact common metric required by "
    "the switching Lyapunov theorem. That theorem covers arbitrary piecewise-"
    "constant switching among the sampled regimes only. Intermediate "
    "unobserved regimes, continuous-time interpolation of the samples, phase "
    "dynamics, nonlinear pressure, operator words, REMESH/nesting, topology "
    "changes, and history geometry remain OPEN."
)
_REFINEMENT_SCOPE = (
    "COMMON-TIME MESH AGREEMENT diagnostic on one declared time interval. "
    "Both sampled paths must pass the restricted trajectory certificate with "
    "one partition and one set of structural scales. Equivalence of the "
    "underlying model and operator schedule must be explicitly asserted by "
    "the caller through same_dynamics_declared; it is recorded but cannot be "
    "inferred from snapshots. "
    "The decisive EPI error compares persistent node identifiers directly. "
    "The structural-state quotient distance is diagnostic and may minimize "
    "different isomorphisms at different times. A single coarse/fine "
    "comparison does not prove convergence, an asymptotic order, or behavior "
    "between the matched samples."
)


@dataclass(frozen=True, slots=True)
class CoreResearchTrajectoryIntervalCertificate:
    """S16 endpoint and discrete nodal-equation evidence for one interval."""

    index: int
    left_time: float
    right_time: float
    dt: float
    endpoint_certificate: CoreResearchIntegrationCertificate
    euler_relaxation: EulerRelaxationWindowDiagnostic
    node_residuals: tuple[tuple[Any, float], ...]
    epi_update_residual_linf: float
    scaled_epi_update_residual_linf: float
    numerical_conditions: tuple[tuple[str, bool], ...]
    interval_conditions_pass: bool

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Names of every failed condition on this sampled interval."""
        return tuple(name for name, passed in self.numerical_conditions if not passed)


@dataclass(frozen=True, slots=True)
class CoreResearchTrajectoryCertificate:
    """Joint numerical status for a sampled restricted S16 trajectory."""

    nodes: tuple[Any, ...]
    times: tuple[float, ...]
    partition: tuple[tuple[Any, ...], ...]
    integration_rule: str
    intervals: tuple[CoreResearchTrajectoryIntervalCertificate, ...]
    switching_stability: SwitchingDiffusionStabilityCertificate
    fixed_transport_operator: bool
    maximum_transport_operator_entry_difference: float
    common_lyapunov_values: tuple[float, ...]
    common_lyapunov_scaled_increments: tuple[float, ...]
    maximum_scaled_common_lyapunov_increase: float
    cumulative_scaled_common_lyapunov_positive_variation: float
    numerical_conditions: tuple[tuple[str, bool], ...]
    joint_temporal_conditions_pass: bool
    first_failed_interval_index: int | None
    first_failed_condition: str | None
    first_failed_conditions: tuple[str, ...]
    scales: StructuralChannelScales
    tolerance: float
    spectral_tolerance: float
    scope: str

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Names of failed trajectory-level conditions."""
        return tuple(name for name, passed in self.numerical_conditions if not passed)


@dataclass(frozen=True, slots=True)
class CoreResearchRefinementSample:
    """Direct and quotient diagnostics at one matched coarse/fine time."""

    coarse_index: int
    fine_index: int
    coarse_time: float
    fine_time: float
    absolute_time_offset: float
    direct_epi_differences: tuple[tuple[Any, float], ...]
    direct_epi_error_linf: float
    scaled_direct_epi_error_linf: float
    quotient_structural_distance: StructuralStateDistanceCertificate
    direct_epi_within_tolerance: bool


@dataclass(frozen=True, slots=True)
class CoreResearchRefinementComparison:
    """Agreement diagnostics for two meshes on one sampled time interval."""

    nodes: tuple[Any, ...]
    coarse_times: tuple[float, ...]
    fine_times: tuple[float, ...]
    common_time_pairs: tuple[tuple[float, float], ...]
    samples: tuple[CoreResearchRefinementSample, ...]
    coarse_trajectory: CoreResearchTrajectoryCertificate
    fine_trajectory: CoreResearchTrajectoryCertificate
    maximum_direct_epi_error: float
    maximum_scaled_direct_epi_error: float
    maximum_coarse_step: float
    maximum_fine_step: float
    all_coarse_times_matched: bool
    fine_grid_is_strict_refinement: bool
    common_time_agreement_within_tolerance: bool
    same_dynamics_declared: bool
    numerical_conditions: tuple[tuple[str, bool], ...]
    joint_refinement_conditions_pass: bool
    trajectory_tolerance: float
    spectral_tolerance: float
    agreement_tolerance: float
    time_tolerance: float
    scope: str

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Names of failed mesh-comparison conditions."""
        return tuple(name for name, passed in self.numerical_conditions if not passed)


def _validated_relative_tolerance(value: Any, name: str = "tolerance") -> float:
    try:
        result = finite_real_scalar(value, name)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a finite real in the open interval (0, 1)"
        ) from exc
    if not 0.0 < result < 1.0:
        raise ValueError(
            f"{name} must be a finite real in the open interval (0, 1)"
        )
    return result


def _validated_time_tolerance(value: Any) -> float:
    try:
        result = finite_real_scalar(value, "time_tolerance")
    except ValueError as exc:
        raise ValueError(
            "time_tolerance must be a finite nonnegative real"
        ) from exc
    if result < 0.0:
        raise ValueError("time_tolerance must be a finite nonnegative real")
    return result


def _materialize_snapshots(
    values: Iterable[nx.Graph], *, name: str
) -> tuple[nx.Graph, ...]:
    if isinstance(values, (nx.Graph, str, bytes)):
        raise TypeError(f"{name} must be an iterable of NetworkX graph snapshots")
    try:
        snapshots = tuple(values)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable of NetworkX graph snapshots"
        ) from exc
    if len(snapshots) < 2:
        raise ValueError(f"{name} must contain at least two snapshots")
    if any(not isinstance(snapshot, nx.Graph) for snapshot in snapshots):
        raise TypeError(f"{name} must contain only NetworkX graph snapshots")
    return snapshots


def _materialize_times(
    values: Iterable[float], *, expected_length: int, name: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of finite real timestamps")
    try:
        raw_times = tuple(values)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable of finite real timestamps"
        ) from exc
    if len(raw_times) != expected_length:
        raise ValueError(f"{name} length must equal the number of snapshots")
    try:
        times = tuple(
            finite_real_scalar(value, f"{name}[{index}]")
            for index, value in enumerate(raw_times)
        )
    except ValueError as exc:
        raise ValueError(f"{name} must contain only finite real timestamps") from exc
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    for left, right in zip(times, times[1:]):
        if not math.isfinite(right - left):
            raise ValueError(f"{name} interval length exceeds floating-point range")
    return times


def _materialize_partition(
    partition: Iterable[Iterable[Any]],
) -> tuple[tuple[Any, ...], ...]:
    if isinstance(partition, (str, bytes)):
        raise TypeError("partition must be an iterable of node blocks")
    try:
        raw_blocks = tuple(partition)
    except TypeError as exc:
        raise TypeError("partition must be an iterable of node blocks") from exc
    blocks: list[tuple[Any, ...]] = []
    for block in raw_blocks:
        if isinstance(block, (str, bytes)):
            raise TypeError("each partition block must be an iterable of node ids")
        try:
            blocks.append(tuple(block))
        except TypeError as exc:
            raise TypeError(
                "each partition block must be an iterable of node ids"
            ) from exc
    return tuple(blocks)


def _attribute_names(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of attribute names")
    try:
        names = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of attribute names") from exc
    if any(not isinstance(item, str) or not item for item in names):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must not contain duplicate names")
    return names


def _edge_support(graph: nx.Graph) -> frozenset[frozenset[Any]]:
    return frozenset(frozenset((source, target)) for source, target in graph.edges())


def _validate_fixed_support(
    snapshots: Sequence[nx.Graph], *, name: str
) -> tuple[Any, ...]:
    first = snapshots[0]
    if first.is_directed():
        raise ValueError(f"{name} requires undirected graph snapshots")
    if first.is_multigraph():
        raise ValueError(f"{name} requires simple graph snapshots")
    nodes = tuple(first)
    node_set = set(nodes)
    edges = _edge_support(first)
    for snapshot in snapshots[1:]:
        if snapshot.is_directed():
            raise ValueError(f"{name} requires undirected graph snapshots")
        if snapshot.is_multigraph():
            raise ValueError(f"{name} requires simple graph snapshots")
        if set(snapshot) != node_set:
            raise ValueError(f"{name} requires persistent node identifiers")
        if _edge_support(snapshot) != edges:
            raise ValueError(f"{name} requires fixed bare edge support")
    return nodes


def _required_channel(
    graph: nx.Graph,
    node: Any,
    aliases: Sequence[str],
    channel: str,
) -> float:
    data = graph.nodes[node]
    for attribute in aliases:
        if attribute in data:
            return finite_real_scalar(data[attribute], f"node {node!r} {channel}")
    raise ValueError(f"node {node!r} is missing required {channel}")


def _finite_fraction(value: Fraction, name: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds finite floating-point range")
    if result == 0.0 and value:
        raise ValueError(f"{name} is below nonzero floating-point range")
    return result


def _forward_euler_residuals(
    left: nx.Graph,
    right: nx.Graph,
    nodes: Sequence[Any],
    dt: float,
    epi_scale: float,
) -> tuple[tuple[tuple[Any, float], ...], float, float, Fraction]:
    exact_residuals: list[tuple[Any, Fraction]] = []
    for node in nodes:
        left_epi = _required_channel(left, node, ALIAS_EPI, "EPI")
        right_epi = _required_channel(right, node, ALIAS_EPI, "EPI")
        frequency = _required_channel(left, node, ALIAS_VF, "structural frequency")
        pressure = _required_channel(left, node, ALIAS_DNFR, "DeltaNFR")
        residual = (
            Fraction.from_float(right_epi)
            - Fraction.from_float(left_epi)
            - Fraction.from_float(dt)
            * Fraction.from_float(frequency)
            * Fraction.from_float(pressure)
        )
        exact_residuals.append((node, residual))

    maximum = max((abs(value) for _, value in exact_residuals), default=Fraction())
    scaled_maximum = maximum / Fraction.from_float(epi_scale)
    node_residuals = tuple(
        (node, _finite_fraction(value, "forward-Euler EPI residual"))
        for node, value in exact_residuals
    )
    return (
        node_residuals,
        _finite_fraction(maximum, "forward-Euler EPI residual"),
        _finite_fraction(scaled_maximum, "scaled forward-Euler EPI residual"),
        scaled_maximum,
    )


def _transport_operator(
    graph: nx.Graph, reference_nodes: Sequence[Any]
) -> np.ndarray:
    local_nodes, laplacian = structural_diffusion_operator(graph)
    frequency = np.asarray(
        [
            _required_channel(
                graph, node, ALIAS_VF, "structural frequency"
            )
            for node in local_nodes
        ],
        dtype=float,
    )
    with np.errstate(over="raise", invalid="raise"):
        try:
            operator = frequency[:, None] * laplacian
        except FloatingPointError as exc:
            raise ValueError(
                "transport operator exceeds finite floating-point range"
            ) from exc
    positions = {node: index for index, node in enumerate(local_nodes)}
    order = [positions[node] for node in reference_nodes]
    aligned = operator[np.ix_(order, order)]
    if not np.all(np.isfinite(aligned)):
        raise ValueError("transport operator exceeds finite floating-point range")
    return aligned


def _fixed_transport_diagnostic(
    snapshots: Sequence[nx.Graph], nodes: Sequence[Any]
) -> tuple[bool, float]:
    operators = tuple(_transport_operator(snapshot, nodes) for snapshot in snapshots)
    reference = operators[0]
    fixed = all(np.array_equal(reference, operator) for operator in operators[1:])
    maximum = 0.0
    for operator in operators[1:]:
        try:
            with np.errstate(over="raise", invalid="raise"):
                difference = np.abs(operator - reference)
        except FloatingPointError as exc:
            raise ValueError(
                "transport-operator difference exceeds finite floating-point range"
            ) from exc
        if not np.all(np.isfinite(difference)):
            raise ValueError(
                "transport-operator difference exceeds finite floating-point range"
            )
        maximum = max(maximum, float(np.max(difference, initial=0.0)))
    return fixed, maximum


def _observed_common_lyapunov(
    snapshots: Sequence[nx.Graph],
    nodes: Sequence[Any],
    switching: SwitchingDiffusionStabilityCertificate,
    epi_scale: float,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    float,
    float,
    Fraction,
    Fraction,
]:
    """Evaluate the switching certificate's common quadratic at every sample."""
    if tuple(switching.nodes) != tuple(nodes):
        raise ValueError("switching metric is not aligned with persistent node ids")
    reference_weights = tuple(
        Fraction.from_float(float(value))
        for value in switching.reference_metric_weights
    )
    reference_total = sum(reference_weights, Fraction())
    # The switching proof is expressed in ``reference_metric_weights``.
    # Normalizing those floats component by component in binary64 generally
    # produces a nearby vector that is not exactly proportional to the proved
    # metric.  Normalize once in rational arithmetic so every observed value
    # uses precisely the same quadratic, with a scale-independent unit total.
    weights = tuple(weight / reference_total for weight in reference_weights)
    weight_total = Fraction(1)
    exact_values: list[Fraction] = []
    for snapshot in snapshots:
        fields = tuple(
            Fraction.from_float(
                _required_channel(snapshot, node, ALIAS_EPI, "EPI")
            )
            for node in nodes
        )
        equilibrium = sum(
            (weight * epi for weight, epi in zip(weights, fields)),
            Fraction(),
        ) / weight_total
        value = Fraction()
        for epi, weight in zip(fields, weights):
            centered = epi - equilibrium
            value += weight * centered * centered / 2
        exact_values.append(value)

    scale_squared = Fraction.from_float(epi_scale) ** 2
    exact_scaled_increments = tuple(
        (right - left) / scale_squared
        for left, right in zip(exact_values, exact_values[1:])
    )
    maximum_increase = max(
        (increment for increment in exact_scaled_increments if increment > 0),
        default=Fraction(),
    )
    cumulative_positive_variation = sum(
        (increment for increment in exact_scaled_increments if increment > 0),
        start=Fraction(),
    )
    return (
        tuple(
            _finite_fraction(value, "observed common Lyapunov value")
            for value in exact_values
        ),
        tuple(
            _finite_fraction(
                increment, "scaled observed common Lyapunov increment"
            )
            for increment in exact_scaled_increments
        ),
        _finite_fraction(
            maximum_increase, "maximum scaled common Lyapunov increase"
        ),
        _finite_fraction(
            cumulative_positive_variation,
            "cumulative scaled common Lyapunov positive variation",
        ),
        maximum_increase,
        cumulative_positive_variation,
    )


def _validated_integration_rule(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("integration_rule must be a string")
    if value != _FORWARD_EULER:
        raise ValueError("integration_rule must be 'forward_euler'")
    return value


def certify_core_research_trajectory(
    snapshots: Iterable[nx.Graph],
    times: Iterable[float],
    partition: Iterable[Iterable[Any]],
    *,
    scales: StructuralChannelScales,
    tolerance: float = 1e-10,
    spectral_tolerance: float = 1e-12,
    integration_rule: str = _FORWARD_EULER,
    node_label_attributes: Sequence[str] = (),
    edge_label_attributes: Sequence[str] = (),
) -> CoreResearchTrajectoryCertificate:
    r"""Certify a sampled forward-Euler path inside the restricted S16 scope.

    The step convention is left-explicit:

    ``EPI[k+1] - EPI[k] = dt[k] * nu_f[k] * DeltaNFR[k]``.

    The equality is checked independently for every persistent node id.  Its
    L-infinity residual is divided by ``scales.epi`` and compared with the
    caller's relative ``tolerance``.  The stored ``dEPI`` channel remains part
    of each endpoint S16 certificate and is not substituted for the declared
    forward-Euler vector field.

    ``spectral_tolerance`` is a separate dimensionless cutoff, relative to the
    fastest decay rate, for resolving the single stationary mode.  It does not
    relax the scaled EPI update residual.

    Passing also requires modal stability of every explicit-Euler step, the
    observed non-increase of the common quadratic at the sampled times, its
    cumulative positive-variation budget over the whole path, and
    the exact common-metric switching theorem for the finite family of sampled
    pure-EPI transport operators.  A fixed transport operator is reported
    separately as a stricter diagnostic and is not required when the more
    general exact switching theorem applies.
    """
    relative_tolerance = _validated_relative_tolerance(tolerance)
    exact_relative_tolerance = Fraction.from_float(relative_tolerance)
    relative_spectral_tolerance = _validated_relative_tolerance(
        spectral_tolerance, "spectral_tolerance"
    )
    if not isinstance(scales, StructuralChannelScales):
        raise TypeError("scales must be a StructuralChannelScales instance")
    rule = _validated_integration_rule(integration_rule)
    states = _materialize_snapshots(snapshots, name="snapshots")
    sample_times = _materialize_times(
        times, expected_length=len(states), name="times"
    )
    blocks = _materialize_partition(partition)
    node_labels = _attribute_names(node_label_attributes, "node_label_attributes")
    edge_labels = _attribute_names(edge_label_attributes, "edge_label_attributes")
    nodes = _validate_fixed_support(states, name="trajectory")

    intervals: list[CoreResearchTrajectoryIntervalCertificate] = []
    first_failed_interval: int | None = None
    first_failed_conditions: tuple[str, ...] = ()
    for index, (left, right, left_time, right_time) in enumerate(
        zip(states, states[1:], sample_times, sample_times[1:])
    ):
        endpoint = certify_core_research_integration(
            left,
            right,
            blocks,
            scales=scales,
            tolerance=relative_tolerance,
            node_label_attributes=node_labels,
            edge_label_attributes=edge_labels,
        )
        dt = right_time - left_time
        euler_relaxation = diagnose_euler_relaxation_window(
            left, dt=dt, tolerance=relative_spectral_tolerance
        )
        (
            node_residuals,
            residual,
            scaled_residual,
            exact_scaled_residual,
        ) = _forward_euler_residuals(left, right, nodes, dt, float(scales.epi))
        conditions = endpoint.numerical_conditions + (
            ("forward_euler_modal_stability", euler_relaxation.is_euler_stable),
            (
                "forward_euler_epi_update",
                exact_scaled_residual <= exact_relative_tolerance,
            ),
        )
        interval_pass = all(passed for _, passed in conditions)
        interval = CoreResearchTrajectoryIntervalCertificate(
            index=index,
            left_time=left_time,
            right_time=right_time,
            dt=dt,
            endpoint_certificate=endpoint,
            euler_relaxation=euler_relaxation,
            node_residuals=node_residuals,
            epi_update_residual_linf=residual,
            scaled_epi_update_residual_linf=scaled_residual,
            numerical_conditions=conditions,
            interval_conditions_pass=interval_pass,
        )
        intervals.append(interval)
        if not first_failed_conditions and not interval_pass:
            first_failed_interval = index
            first_failed_conditions = interval.failed_conditions

    switching = verify_switching_diffusion_stability(
        states, tolerance=relative_tolerance
    )
    fixed_transport, transport_difference = _fixed_transport_diagnostic(states, nodes)
    (
        lyapunov_values,
        lyapunov_increments,
        maximum_lyapunov_increase,
        cumulative_lyapunov_positive_variation,
        exact_maximum_lyapunov_increase,
        exact_cumulative_lyapunov_positive_variation,
    ) = _observed_common_lyapunov(
        states, nodes, switching, float(scales.epi)
    )
    all_intervals_pass = all(
        interval.interval_conditions_pass for interval in intervals
    )
    conditions = (
        ("all_interval_conditions", all_intervals_pass),
        (
            "sampled_switching_common_lyapunov",
            bool(switching.supports_exact_switching_theorem),
        ),
        (
            "observed_common_lyapunov_nonincrease",
            exact_maximum_lyapunov_increase <= exact_relative_tolerance,
        ),
        (
            "cumulative_common_lyapunov_positive_variation",
            exact_cumulative_lyapunov_positive_variation
            <= exact_relative_tolerance,
        ),
    )
    joint_pass = all(passed for _, passed in conditions)
    if not first_failed_conditions:
        first_failed_conditions = tuple(
            name for name, passed in conditions if not passed
        )
    first_failed_condition = (
        first_failed_conditions[0] if first_failed_conditions else None
    )

    return CoreResearchTrajectoryCertificate(
        nodes=nodes,
        times=sample_times,
        partition=blocks,
        integration_rule=rule,
        intervals=tuple(intervals),
        switching_stability=switching,
        fixed_transport_operator=fixed_transport,
        maximum_transport_operator_entry_difference=transport_difference,
        common_lyapunov_values=lyapunov_values,
        common_lyapunov_scaled_increments=lyapunov_increments,
        maximum_scaled_common_lyapunov_increase=maximum_lyapunov_increase,
        cumulative_scaled_common_lyapunov_positive_variation=(
            cumulative_lyapunov_positive_variation
        ),
        numerical_conditions=conditions,
        joint_temporal_conditions_pass=joint_pass,
        first_failed_interval_index=first_failed_interval,
        first_failed_condition=first_failed_condition,
        first_failed_conditions=first_failed_conditions,
        scales=scales,
        tolerance=relative_tolerance,
        spectral_tolerance=relative_spectral_tolerance,
        scope=_TRAJECTORY_SCOPE,
    )


def _times_match(left: float, right: float, tolerance: float) -> bool:
    if left == right:
        return True
    return abs(left - right) <= tolerance


def _common_time_indices(
    coarse_times: Sequence[float],
    fine_times: Sequence[float],
    tolerance: float,
) -> tuple[tuple[int, int], ...]:
    coarse_candidates: list[tuple[int, tuple[int, ...]]] = []
    for coarse_index, coarse_time in enumerate(coarse_times):
        candidates = tuple(
            fine_index
            for fine_index, fine_time in enumerate(fine_times)
            if _times_match(coarse_time, fine_time, tolerance)
        )
        if len(candidates) > 1:
            raise ValueError(
                "time_tolerance creates an ambiguous coarse/fine time match"
            )
        coarse_candidates.append((coarse_index, candidates))

    selected = tuple(
        (coarse_index, candidates[0])
        for coarse_index, candidates in coarse_candidates
        if candidates
    )
    fine_counts: dict[int, int] = {}
    for _, fine_index in selected:
        fine_counts[fine_index] = fine_counts.get(fine_index, 0) + 1
    if any(count > 1 for count in fine_counts.values()):
        raise ValueError("time_tolerance creates an ambiguous coarse/fine time match")
    return selected


def _direct_epi_errors(
    coarse: nx.Graph,
    fine: nx.Graph,
    nodes: Sequence[Any],
    epi_scale: float,
) -> tuple[tuple[tuple[Any, float], ...], float, float]:
    exact_differences: list[tuple[Any, Fraction]] = []
    for node in nodes:
        coarse_epi = _required_channel(coarse, node, ALIAS_EPI, "EPI")
        fine_epi = _required_channel(fine, node, ALIAS_EPI, "EPI")
        exact_differences.append(
            (
                node,
                Fraction.from_float(coarse_epi) - Fraction.from_float(fine_epi),
            )
        )
    maximum = max(
        (abs(value) for _, value in exact_differences), default=Fraction()
    )
    scaled_maximum = maximum / Fraction.from_float(epi_scale)
    differences = tuple(
        (node, _finite_fraction(value, "direct persistent-id EPI difference"))
        for node, value in exact_differences
    )
    return (
        differences,
        _finite_fraction(maximum, "direct persistent-id EPI error"),
        _finite_fraction(
            scaled_maximum, "scaled direct persistent-id EPI error"
        ),
    )


def compare_core_research_trajectory_refinement(
    coarse_snapshots: Iterable[nx.Graph],
    coarse_times: Iterable[float],
    fine_snapshots: Iterable[nx.Graph],
    fine_times: Iterable[float],
    partition: Iterable[Iterable[Any]],
    *,
    scales: StructuralChannelScales,
    agreement_tolerance: float,
    same_dynamics_declared: bool,
    trajectory_tolerance: float = 1e-10,
    spectral_tolerance: float = 1e-12,
    time_tolerance: float = 0.0,
    node_label_attributes: Sequence[str] = (),
    edge_label_attributes: Sequence[str] = (),
) -> CoreResearchRefinementComparison:
    r"""Compare two sampled meshes only at unambiguously matched times.

    ``time_tolerance`` is an absolute tolerance in the units of ``times``.  It
    is used for endpoint and common-time matching, is included in the result,
    and must produce a one-to-one match.  Both meshes must cover the same
    sampled interval.  The function does not interpolate either mesh.

    ``common_time_agreement_within_tolerance`` uses the maximum EPI difference
    on the same node ids, scaled by ``scales.epi``, and compares it with the
    explicitly declared ``agreement_tolerance``.  The stricter
    ``trajectory_tolerance`` independently controls both S16 path certificates,
    including their forward-Euler residuals and cumulative Lyapunov budget.
    ``spectral_tolerance`` independently resolves stationary modes relative to
    each frozen spectrum. The quotient structural distance
    may select a graph isomorphism and is therefore explicitly diagnostic.  No
    returned field asserts convergence or an accuracy order.

    The joint comparison Boolean requires ``same_dynamics_declared=True``, both
    internally computed trajectory certificates, inclusion of every coarse
    time, a strictly smaller maximum fine step, and direct persistent-id EPI
    agreement. Snapshot data cannot establish that both meshes used the same
    external operator/event schedule; the explicit declaration is recorded as
    an assumption rather than inferred as evidence.
    """
    path_tolerance = _validated_relative_tolerance(
        trajectory_tolerance, "trajectory_tolerance"
    )
    path_spectral_tolerance = _validated_relative_tolerance(
        spectral_tolerance, "spectral_tolerance"
    )
    comparison_tolerance = _validated_relative_tolerance(
        agreement_tolerance, "agreement_tolerance"
    )
    matching_tolerance = _validated_time_tolerance(time_tolerance)
    if not isinstance(same_dynamics_declared, bool):
        raise TypeError("same_dynamics_declared must be a boolean")
    if not isinstance(scales, StructuralChannelScales):
        raise TypeError("scales must be a StructuralChannelScales instance")
    coarse_states = _materialize_snapshots(
        coarse_snapshots, name="coarse_snapshots"
    )
    fine_states = _materialize_snapshots(fine_snapshots, name="fine_snapshots")
    coarse_sample_times = _materialize_times(
        coarse_times,
        expected_length=len(coarse_states),
        name="coarse_times",
    )
    fine_sample_times = _materialize_times(
        fine_times,
        expected_length=len(fine_states),
        name="fine_times",
    )
    node_labels = _attribute_names(node_label_attributes, "node_label_attributes")
    edge_labels = _attribute_names(edge_label_attributes, "edge_label_attributes")
    blocks = _materialize_partition(partition)
    coarse_nodes = _validate_fixed_support(coarse_states, name="coarse trajectory")
    fine_nodes = _validate_fixed_support(fine_states, name="fine trajectory")
    if set(coarse_nodes) != set(fine_nodes):
        raise ValueError(
            "coarse and fine trajectories require persistent shared node identifiers"
        )
    if _edge_support(coarse_states[0]) != _edge_support(fine_states[0]):
        raise ValueError(
            "coarse and fine trajectories require the same bare edge support"
        )
    if not (
        _times_match(
            coarse_sample_times[0], fine_sample_times[0], matching_tolerance
        )
        and _times_match(
            coarse_sample_times[-1], fine_sample_times[-1], matching_tolerance
        )
    ):
        raise ValueError("coarse and fine meshes must cover the same time interval")

    matches = _common_time_indices(
        coarse_sample_times, fine_sample_times, matching_tolerance
    )
    if not matches or matches[0] != (0, 0) or matches[-1] != (
        len(coarse_states) - 1,
        len(fine_states) - 1,
    ):
        raise ValueError(
            "coarse and fine interval endpoints must match unambiguously"
        )

    coarse_trajectory = certify_core_research_trajectory(
        coarse_states,
        coarse_sample_times,
        blocks,
        scales=scales,
        tolerance=path_tolerance,
        spectral_tolerance=path_spectral_tolerance,
        integration_rule=_FORWARD_EULER,
        node_label_attributes=node_labels,
        edge_label_attributes=edge_labels,
    )
    fine_trajectory = certify_core_research_trajectory(
        fine_states,
        fine_sample_times,
        blocks,
        scales=scales,
        tolerance=path_tolerance,
        spectral_tolerance=path_spectral_tolerance,
        integration_rule=_FORWARD_EULER,
        node_label_attributes=node_labels,
        edge_label_attributes=edge_labels,
    )

    samples: list[CoreResearchRefinementSample] = []
    for coarse_index, fine_index in matches:
        coarse = coarse_states[coarse_index]
        fine = fine_states[fine_index]
        differences, error, scaled_error = _direct_epi_errors(
            coarse, fine, coarse_nodes, float(scales.epi)
        )
        quotient_distance = fixed_topology_structural_state_distance(
            coarse,
            fine,
            scales=scales,
            node_label_attributes=node_labels,
            edge_label_attributes=edge_labels,
        )
        offset = abs(coarse_sample_times[coarse_index] - fine_sample_times[fine_index])
        if not math.isfinite(offset):
            raise ValueError("matched time offset exceeds finite floating-point range")
        samples.append(
            CoreResearchRefinementSample(
                coarse_index=coarse_index,
                fine_index=fine_index,
                coarse_time=coarse_sample_times[coarse_index],
                fine_time=fine_sample_times[fine_index],
                absolute_time_offset=offset,
                direct_epi_differences=differences,
                direct_epi_error_linf=error,
                scaled_direct_epi_error_linf=scaled_error,
                quotient_structural_distance=quotient_distance,
                direct_epi_within_tolerance=scaled_error <= comparison_tolerance,
            )
        )

    maximum_error = max(sample.direct_epi_error_linf for sample in samples)
    maximum_scaled_error = max(
        sample.scaled_direct_epi_error_linf for sample in samples
    )
    all_coarse_matched = len(matches) == len(coarse_states)
    maximum_coarse_step = max(
        right - left
        for left, right in zip(coarse_sample_times, coarse_sample_times[1:])
    )
    maximum_fine_step = max(
        right - left
        for left, right in zip(fine_sample_times, fine_sample_times[1:])
    )
    strict_refinement = bool(
        all_coarse_matched
        and len(fine_states) > len(coarse_states)
        and maximum_fine_step < maximum_coarse_step
    )
    common_agreement = all(
        sample.direct_epi_within_tolerance for sample in samples
    )
    conditions = (
        ("same_dynamics_declared", same_dynamics_declared),
        (
            "coarse_trajectory_certificate",
            coarse_trajectory.joint_temporal_conditions_pass,
        ),
        (
            "fine_trajectory_certificate",
            fine_trajectory.joint_temporal_conditions_pass,
        ),
        ("all_coarse_times_matched", all_coarse_matched),
        ("fine_grid_is_strict_refinement", strict_refinement),
        ("common_time_direct_epi_agreement", common_agreement),
    )
    return CoreResearchRefinementComparison(
        nodes=coarse_nodes,
        coarse_times=coarse_sample_times,
        fine_times=fine_sample_times,
        common_time_pairs=tuple(
            (coarse_sample_times[coarse_index], fine_sample_times[fine_index])
            for coarse_index, fine_index in matches
        ),
        samples=tuple(samples),
        coarse_trajectory=coarse_trajectory,
        fine_trajectory=fine_trajectory,
        maximum_direct_epi_error=maximum_error,
        maximum_scaled_direct_epi_error=maximum_scaled_error,
        maximum_coarse_step=maximum_coarse_step,
        maximum_fine_step=maximum_fine_step,
        all_coarse_times_matched=all_coarse_matched,
        fine_grid_is_strict_refinement=strict_refinement,
        common_time_agreement_within_tolerance=common_agreement,
        same_dynamics_declared=same_dynamics_declared,
        numerical_conditions=conditions,
        joint_refinement_conditions_pass=all(passed for _, passed in conditions),
        trajectory_tolerance=path_tolerance,
        spectral_tolerance=path_spectral_tolerance,
        agreement_tolerance=comparison_tolerance,
        time_tolerance=matching_tolerance,
        scope=_REFINEMENT_SCOPE,
    )
