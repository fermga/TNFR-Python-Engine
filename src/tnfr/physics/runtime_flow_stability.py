r"""Exact stability evidence for one observed TNFR nodal-flow interval.

A captured endpoint retains only ordered node support, scalar EPI, structural
frequency, pressure, and effective conductance. Every finite scalar is first
materialized as binary64 and then interpreted through Fraction.from_float.

For two endpoint snapshots the certificate evaluates exactly

    x_right - x_left = duration * (nu_f * DeltaNFR_left).

Pure-EPI promotion additionally requires stable support, fixed symmetric
nonnegative conductance, unchanged positive capacity and held pressure, and
DeltaNFR_left = -L_rw x_left exactly. Runtime promotion separately requires
explicit evidence for DefaultIntegrator, Euler, one substep, absent Gamma,
inactive clipping, and disabled extended dynamics. Only then is the observed
interval identified with the rational explicit-Euler map

    A = I - duration * diag(nu_f) L_rw.

The map is analyzed in the reversible metric H = diag(d_i / nu_f_i) with the
shared exact quotient-gain machinery. This read-only result covers one supplied
interval. It does not infer runtime provenance, certify a custom integrator,
estimate solver error, or prove future, refined, or repeated schedules.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ..mathematics._neighbor_differences import edge_mean_differences
from ..mathematics.unified_numerical import np
from ..types import real_scalar_epi
from ._conductance import read_conductance
from .hybrid_operator_stability import (
    _exact_matrix_product,
    _exact_projection,
    _exact_quotient_energy_gain_upper_bound,
    _exact_weighted_frobenius_energy_bound,
)

__all__ = [
    "NodalFlowStateSnapshot",
    "NodalFlowIntervalCertificate",
    "capture_nodal_flow_state",
    "certify_observed_nodal_flow_interval",
]

ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[tuple[Fraction, ...], ...]

_SCOPE = (
    "EXACT one-interval endpoint certificate for represented binary64 values. "
    "It separately checks the rational nodal identity, frozen pure-EPI "
    "realization, declared built-in Euler conditions, IEEE-754 replay, and "
    "quotient contraction. It does not infer runtime provenance, certify "
    "custom integrators, estimate solver error, or prove future, refined, or "
    "repeated schedules."
)


def _binary64_fraction(value: Any, name: str) -> tuple[float, Fraction]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            f"{name} must contain finite real values, not booleans"
        )
    try:
        source_nonzero = bool(value != 0)
        floating = float(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            f"{name} must contain finite binary64 values"
        ) from exc
    if not math.isfinite(floating):
        raise ValueError(f"{name} must contain finite binary64 values")
    if floating == 0.0 and source_nonzero:
        raise ValueError(
            f"{name} contains a nonzero value below binary64 range"
        )
    return floating, Fraction.from_float(floating)


def _binary64_vector(
    values: Sequence[Any],
    name: str,
) -> tuple[tuple[float, ...], ExactVector]:
    floats: list[float] = []
    exact: list[Fraction] = []
    for value in values:
        floating, ratio = _binary64_fraction(value, name)
        floats.append(floating)
        exact.append(ratio)
    return tuple(floats), tuple(exact)


def _node_order(
    graph: Any,
    nodes: Iterable[Any] | None,
) -> tuple[Any, ...]:
    graph_nodes = tuple(graph)
    if nodes is None:
        return graph_nodes
    if isinstance(nodes, (str, bytes)):
        raise TypeError(
            "nodes must be an iterable of unique node identifiers"
        )
    try:
        ordered = tuple(nodes)
        unique = set(ordered)
        support = set(graph_nodes)
    except TypeError as exc:
        raise TypeError(
            "nodes must be an iterable of unique hashable identifiers"
        ) from exc
    if len(unique) != len(ordered):
        raise ValueError("nodes must contain unique identifiers")
    if unique != support:
        raise ValueError("nodes must contain the complete graph support")
    return ordered


def _conductance_observation(
    graph: Any,
    nodes: tuple[Any, ...],
    epi: tuple[float, ...],
) -> tuple[ExactMatrix, tuple[float, ...], ExactVector]:
    snapshot = read_conductance(graph, list(nodes))
    dense = snapshot.dense()
    exact_matrix = tuple(
        tuple(
            Fraction.from_float(float(value))
            for value in row
        )
        for row in dense
    )
    replay_array = edge_mean_differences(
        np.asarray(epi, dtype=float),
        snapshot.source,
        snapshot.target,
        snapshot.weight,
    )
    replay = tuple(float(value) for value in replay_array)
    exact_replay = tuple(
        Fraction.from_float(value) for value in replay
    )
    return exact_matrix, replay, exact_replay


def _is_symmetric_nonnegative(matrix: ExactMatrix) -> bool:
    return bool(
        all(
            value >= 0
            for row in matrix
            for value in row
        )
        and all(
            matrix[i][j] == matrix[j][i]
            for i in range(len(matrix))
            for j in range(i)
        )
    )


def _strength(matrix: ExactMatrix) -> ExactVector:
    return tuple(sum(row, Fraction(0)) for row in matrix)


def _pure_pressure(
    adjacency: ExactMatrix,
    degrees: ExactVector,
    epi: ExactVector,
) -> ExactVector | None:
    if any(degree <= 0 for degree in degrees):
        return None
    return tuple(
        sum(
            (
                adjacency[i][j] * (epi[j] - epi[i])
                for j in range(len(epi))
            ),
            Fraction(0),
        )
        / degrees[i]
        for i in range(len(epi))
    )


def _euler_map(
    adjacency: ExactMatrix,
    degrees: ExactVector,
    nu_f: ExactVector,
    duration: Fraction,
) -> ExactMatrix:
    size = len(nu_f)
    return tuple(
        tuple(
            (Fraction(1) if i == j else Fraction(0))
            + duration
            * nu_f[i]
            * (
                adjacency[i][j] / degrees[i]
                - (Fraction(1) if i == j else Fraction(0))
            )
            for j in range(size)
        )
        for i in range(size)
    )


def _matrix_vector(
    matrix: ExactMatrix,
    vector: ExactVector,
) -> ExactVector:
    return tuple(
        sum(
            (
                coefficient * value
                for coefficient, value in zip(row, vector)
            ),
            Fraction(0),
        )
        for row in matrix
    )


def _energy(
    epi: ExactVector,
    metric: ExactVector,
) -> Fraction:
    total = sum(metric, Fraction(0))
    center = sum(
        (
            weight * value
            for weight, value in zip(metric, epi)
        ),
        Fraction(0),
    ) / total
    return sum(
        (
            weight * (value - center) ** 2
            for weight, value in zip(metric, epi)
        ),
        Fraction(0),
    ) / 2


def _gain_bound(
    matrix: ExactMatrix,
    metric: ExactVector,
) -> Fraction:
    projection = _exact_projection(metric)
    quotient = _exact_matrix_product(
        _exact_matrix_product(projection, matrix),
        projection,
    )
    frobenius = _exact_weighted_frobenius_energy_bound(
        quotient,
        metric,
    )
    return _exact_quotient_energy_gain_upper_bound(
        quotient,
        projection,
        metric,
        frobenius,
    )


@dataclass(frozen=True, slots=True)
class NodalFlowStateSnapshot:
    """Detached state sufficient to evaluate one nodal-flow interval."""

    nodes: tuple[Any, ...]
    epi: tuple[float, ...]
    nu_f: tuple[float, ...]
    delta_nfr: tuple[float, ...]
    exact_epi: ExactVector
    exact_nu_f: ExactVector
    exact_delta_nfr: ExactVector
    conductance: ExactMatrix
    binary64_pure_epi_pressure: tuple[float, ...]
    exact_binary64_pure_epi_pressure: ExactVector


@dataclass(frozen=True, slots=True)
class NodalFlowIntervalCertificate:
    """Exact read-only evidence for one observed nodal-flow interval.

    Rational realization, frozen pure-EPI eligibility, declared runtime
    eligibility, IEEE-754 replay, and quotient contraction are separate facts.
    The exact gain is published only for an identified rational Euler map. It
    gives no custom-integrator, future-interval, or repeated-schedule theorem.
    """

    left: NodalFlowStateSnapshot
    right: NodalFlowStateSnapshot
    duration: float
    exact_duration: Fraction
    exact_nodal_equation_residual: ExactVector | None
    exact_nodal_equation_realized: bool
    stable_node_support: bool
    fixed_conductance: bool
    symmetric_nonnegative_conductance: bool
    positive_row_strength: bool
    positive_capacity: bool
    capacity_unchanged: bool
    pressure_unchanged: bool
    exact_pure_epi_pressure: ExactVector | None
    exact_pressure_residual: ExactVector | None
    exact_pure_epi_pressure_realized: bool
    exact_binary64_pressure_replay_residual: ExactVector | None
    binary64_pure_epi_pressure_realized: bool
    diffusion_conditions: tuple[tuple[str, bool], ...]
    pure_epi_diffusion_eligible: bool
    integrator_name: str | None
    method: str | None
    substeps: int | None
    gamma_is_none: bool | None
    clipping_applied: bool | None
    extended_dynamics_requested: bool | None
    runtime_conditions: tuple[tuple[str, bool], ...]
    runtime_euler_eligible: bool
    binary64_euler_replay: tuple[float, ...] | None
    exact_binary64_euler_replay_residual: ExactVector | None
    binary64_euler_replay_matches: bool
    binary64_runtime_interval_identified: bool
    exact_metric_weights: ExactVector | None
    exact_explicit_euler_map: ExactMatrix | None
    exact_explicit_euler_endpoint_residual: ExactVector | None
    explicit_euler_map_identified: bool
    exact_left_disagreement_energy: Fraction | None
    exact_right_disagreement_energy: Fraction | None
    exact_observed_disagreement_energy_gain: Fraction | None
    observed_disagreement_nonincrease: bool | None
    exact_quotient_energy_gain_upper_bound: Fraction | None
    _global_disagreement_contraction_certified: bool = field(repr=False)
    integrator_provenance_certified: bool
    future_or_repeated_schedule_stability_certified: bool
    scope: str
    _proof_stamp: tuple[Any, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def _proof_fields_are_intact(self) -> bool:
        """Detect ordinary replacement or mutation of decisive proof fields."""

        try:
            expected = _nodal_flow_interval_proof_stamp(self)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return type(self._proof_stamp) is tuple and self._proof_stamp == expected

    @property
    def global_disagreement_contraction_certified(self) -> bool:
        """Return the contraction claim only while its proof fields are intact."""

        return bool(
            self._proof_fields_are_intact()
            and self._global_disagreement_contraction_certified
        )

    @property
    def nodes(self) -> tuple[Any, ...]:
        return self.left.nodes

    @property
    def left_epi(self) -> tuple[float, ...]:
        return self.left.epi

    @property
    def right_epi(self) -> tuple[float, ...]:
        return self.right.epi

    @property
    def left_nu_f(self) -> tuple[float, ...]:
        return self.left.nu_f

    @property
    def left_delta_nfr(self) -> tuple[float, ...]:
        return self.left.delta_nfr

    @property
    def failed_diffusion_conditions(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, passed in self.diffusion_conditions
            if not passed
        )

    @property
    def failed_runtime_conditions(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, passed in self.runtime_conditions
            if not passed
        )

    @property
    def euler_map_abstention_reasons(self) -> tuple[str, ...]:
        reasons = list(self.failed_diffusion_conditions)
        reasons.extend(self.failed_runtime_conditions)
        if not self.exact_nodal_equation_realized:
            reasons.append("exact_nodal_equation_realized")
        return tuple(reasons)


_NODAL_FLOW_INTERVAL_PROOF_VERSION = "observed_nodal_flow_interval_v1"
_NODAL_FLOW_SNAPSHOT_PROOF_VERSION = "nodal_flow_state_snapshot_v1"


def _nodal_flow_snapshot_proof_signature(
    snapshot: Any,
) -> tuple[Any, ...]:
    """Freeze one endpoint by value rather than by dataclass object identity."""

    if type(snapshot) is not NodalFlowStateSnapshot:
        raise TypeError("flow certificate endpoints must be canonical snapshots")
    return (
        _NODAL_FLOW_SNAPSHOT_PROOF_VERSION,
        tuple(
            (item.name, getattr(snapshot, item.name))
            for item in fields(NodalFlowStateSnapshot)
        ),
    )


def _nodal_flow_interval_proof_stamp(
    certificate: Any,
) -> tuple[Any, ...]:
    """Snapshot every field used by interval-level theorem consumers."""

    if type(certificate) is not NodalFlowIntervalCertificate:
        raise TypeError("certificate must be a NodalFlowIntervalCertificate")
    payload = tuple(
        (item.name, getattr(certificate, item.name))
        for item in fields(NodalFlowIntervalCertificate)
        if item.name not in {"left", "right", "_proof_stamp"}
    )
    return (
        _NODAL_FLOW_INTERVAL_PROOF_VERSION,
        _nodal_flow_snapshot_proof_signature(certificate.left),
        _nodal_flow_snapshot_proof_signature(certificate.right),
        payload,
    )


def capture_nodal_flow_state(
    graph: Any,
    *,
    nodes: Iterable[Any] | None = None,
) -> NodalFlowStateSnapshot:
    """Capture only state used by the one-interval certificate.

    EPI must have a finite real scalar representation. Boolean nodal values and
    invalid conductance are rejected. The graph is neither mutated nor retained.
    """

    ordered = _node_order(graph, nodes)
    raw_epi: list[Any] = []
    raw_nu_f: list[Any] = []
    raw_pressure: list[Any] = []
    for node in ordered:
        data = graph.nodes[node]
        epi_value = get_attr(
            data,
            ALIAS_EPI,
            0.0,
            strict=True,
            conv=lambda value: value,
        )
        if isinstance(epi_value, (bool, np.bool_)):
            raise ValueError(
                "nodal-flow capture requires real scalar EPI"
            )
        try:
            scalar_epi = real_scalar_epi(epi_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "nodal-flow capture requires real scalar EPI"
            ) from exc
        if scalar_epi is None:
            raise ValueError(
                "nodal-flow capture requires real scalar EPI"
            )
        raw_epi.append(scalar_epi)
        raw_nu_f.append(
            get_attr(
                data,
                ALIAS_VF,
                0.0,
                strict=True,
                conv=lambda value: value,
            )
        )
        raw_pressure.append(
            get_attr(
                data,
                ALIAS_DNFR,
                0.0,
                strict=True,
                conv=lambda value: value,
            )
        )

    epi, exact_epi = _binary64_vector(raw_epi, "EPI")
    nu_f, exact_nu_f = _binary64_vector(raw_nu_f, "nu_f")
    pressure, exact_pressure = _binary64_vector(
        raw_pressure,
        "DeltaNFR",
    )
    conductance, replay, exact_replay = _conductance_observation(
        graph,
        ordered,
        epi,
    )
    return NodalFlowStateSnapshot(
        nodes=ordered,
        epi=epi,
        nu_f=nu_f,
        delta_nfr=pressure,
        exact_epi=exact_epi,
        exact_nu_f=exact_nu_f,
        exact_delta_nfr=exact_pressure,
        conductance=conductance,
        binary64_pure_epi_pressure=replay,
        exact_binary64_pure_epi_pressure=exact_replay,
    )


def certify_observed_nodal_flow_interval(
    left: NodalFlowStateSnapshot,
    right: NodalFlowStateSnapshot,
    *,
    duration: Any,
    integrator_name: str | None = None,
    method: str | None = None,
    substeps: int | None = None,
    gamma_is_none: bool | None = None,
    clipping_applied: bool | None = None,
    extended_dynamics_requested: bool | None = None,
) -> NodalFlowIntervalCertificate:
    """Certify one captured interval without mutating either snapshot.

    Runtime metadata is explicit and missing values cause abstention. Endpoint
    agreement never infers integrator identity. The rational map theorem and
    exact same-operation binary64 replay are reported independently.
    """

    if not isinstance(left, NodalFlowStateSnapshot):
        raise TypeError("left must be a NodalFlowStateSnapshot")
    if not isinstance(right, NodalFlowStateSnapshot):
        raise TypeError("right must be a NodalFlowStateSnapshot")
    duration_float, duration_exact = _binary64_fraction(
        duration,
        "duration",
    )
    if duration_exact < 0:
        raise ValueError("duration must be nonnegative")

    stable_support = left.nodes == right.nodes
    if stable_support:
        nodal_residual: ExactVector | None = tuple(
            right_value
            - left_value
            - duration_exact * capacity * pressure
            for left_value, right_value, capacity, pressure in zip(
                left.exact_epi,
                right.exact_epi,
                left.exact_nu_f,
                left.exact_delta_nfr,
            )
        )
        nodal_realized = all(value == 0 for value in nodal_residual)
        capacity_unchanged = left.exact_nu_f == right.exact_nu_f
        pressure_unchanged = (
            left.exact_delta_nfr == right.exact_delta_nfr
        )
    else:
        nodal_residual = None
        nodal_realized = False
        capacity_unchanged = False
        pressure_unchanged = False

    fixed_conductance = bool(
        stable_support
        and left.conductance == right.conductance
    )
    symmetric = bool(
        _is_symmetric_nonnegative(left.conductance)
        and _is_symmetric_nonnegative(right.conductance)
    )
    degrees = _strength(left.conductance)
    positive_strength = all(value > 0 for value in degrees)
    positive_capacity = all(value > 0 for value in left.exact_nu_f)

    expected_pressure = _pure_pressure(
        left.conductance,
        degrees,
        left.exact_epi,
    )
    if expected_pressure is None:
        pressure_residual = None
        pressure_realized = False
    else:
        pressure_residual = tuple(
            observed - expected
            for observed, expected in zip(
                left.exact_delta_nfr,
                expected_pressure,
            )
        )
        pressure_realized = all(
            value == 0 for value in pressure_residual
        )

    binary_pressure_residual = tuple(
        observed - expected
        for observed, expected in zip(
            left.exact_delta_nfr,
            left.exact_binary64_pure_epi_pressure,
        )
    )
    binary_pressure_realized = all(
        value == 0 for value in binary_pressure_residual
    )

    diffusion_conditions = (
        ("at_least_two_nodes", len(left.nodes) >= 2),
        ("stable_node_support", stable_support),
        ("fixed_conductance", fixed_conductance),
        ("symmetric_nonnegative_conductance", symmetric),
        ("positive_row_strength", positive_strength),
        ("positive_capacity", positive_capacity),
        ("capacity_unchanged", capacity_unchanged),
        ("pressure_unchanged", pressure_unchanged),
        ("exact_pure_epi_pressure_realized", pressure_realized),
    )
    diffusion_eligible = all(
        passed for _, passed in diffusion_conditions
    )

    valid_substeps = bool(
        isinstance(substeps, Integral)
        and not isinstance(substeps, (bool, np.bool_))
        and int(substeps) == 1
    )
    runtime_conditions = (
        ("default_integrator", integrator_name == "DefaultIntegrator"),
        ("euler_method", method == "euler"),
        ("one_substep", valid_substeps),
        ("gamma_none", gamma_is_none is True),
        ("clipping_inactive", clipping_applied is False),
        (
            "extended_dynamics_not_requested",
            extended_dynamics_requested is False,
        ),
    )
    runtime_eligible = all(
        passed for _, passed in runtime_conditions
    )

    binary_replay: tuple[float, ...] | None = None
    binary_residual: ExactVector | None = None
    binary_matches = False
    if stable_support:
        try:
            with np.errstate(
                over="raise",
                invalid="raise",
                under="ignore",
            ):
                base = np.multiply(
                    np.asarray(left.nu_f, dtype=float),
                    np.asarray(left.delta_nfr, dtype=float),
                )
                increment = np.multiply(duration_float, base)
                replay_array = np.add(
                    np.asarray(left.epi, dtype=float),
                    increment,
                )
            if np.all(np.isfinite(replay_array)):
                binary_replay = tuple(
                    float(value) for value in replay_array
                )
                exact_binary_replay = tuple(
                    Fraction.from_float(value)
                    for value in binary_replay
                )
                binary_residual = tuple(
                    observed - expected
                    for observed, expected in zip(
                        right.exact_epi,
                        exact_binary_replay,
                    )
                )
                binary_matches = all(
                    value == 0 for value in binary_residual
                )
        except FloatingPointError:
            pass

    binary_runtime_identified = bool(
        runtime_eligible
        and stable_support
        and capacity_unchanged
        and pressure_unchanged
        and binary_matches
    )
    map_identified = bool(
        nodal_realized
        and diffusion_eligible
        and runtime_eligible
    )

    metric: ExactVector | None = None
    left_energy: Fraction | None = None
    right_energy: Fraction | None = None
    observed_gain: Fraction | None = None
    observed_nonincrease: bool | None = None
    if (
        len(left.nodes) > 0
        and stable_support
        and positive_strength
        and positive_capacity
        and capacity_unchanged
    ):
        metric = tuple(
            degree / capacity
            for degree, capacity in zip(
                degrees,
                left.exact_nu_f,
            )
        )
        left_energy = _energy(left.exact_epi, metric)
        right_energy = _energy(right.exact_epi, metric)
        observed_nonincrease = right_energy <= left_energy
        if left_energy > 0:
            observed_gain = right_energy / left_energy

    exact_map: ExactMatrix | None = None
    map_residual: ExactVector | None = None
    quotient_gain: Fraction | None = None
    contracts = False
    if map_identified:
        if metric is None:
            raise RuntimeError(
                "internal error: identified Euler metric is absent"
            )
        exact_map = _euler_map(
            left.conductance,
            degrees,
            left.exact_nu_f,
            duration_exact,
        )
        mapped = _matrix_vector(exact_map, left.exact_epi)
        map_residual = tuple(
            observed - expected
            for observed, expected in zip(
                right.exact_epi,
                mapped,
            )
        )
        if any(value != 0 for value in map_residual):
            raise RuntimeError(
                "internal error: nodal and diffusion identities disagree"
            )
        quotient_gain = _gain_bound(exact_map, metric)
        contracts = quotient_gain < 1

    stored_substeps = (
        int(substeps)
        if isinstance(substeps, Integral)
        and not isinstance(substeps, (bool, np.bool_))
        else None
    )
    certificate = NodalFlowIntervalCertificate(
        left=left,
        right=right,
        duration=duration_float,
        exact_duration=duration_exact,
        exact_nodal_equation_residual=nodal_residual,
        exact_nodal_equation_realized=nodal_realized,
        stable_node_support=stable_support,
        fixed_conductance=fixed_conductance,
        symmetric_nonnegative_conductance=symmetric,
        positive_row_strength=positive_strength,
        positive_capacity=positive_capacity,
        capacity_unchanged=capacity_unchanged,
        pressure_unchanged=pressure_unchanged,
        exact_pure_epi_pressure=expected_pressure,
        exact_pressure_residual=pressure_residual,
        exact_pure_epi_pressure_realized=pressure_realized,
        exact_binary64_pressure_replay_residual=(
            binary_pressure_residual
        ),
        binary64_pure_epi_pressure_realized=(
            binary_pressure_realized
        ),
        diffusion_conditions=diffusion_conditions,
        pure_epi_diffusion_eligible=diffusion_eligible,
        integrator_name=integrator_name,
        method=method,
        substeps=stored_substeps,
        gamma_is_none=gamma_is_none,
        clipping_applied=clipping_applied,
        extended_dynamics_requested=extended_dynamics_requested,
        runtime_conditions=runtime_conditions,
        runtime_euler_eligible=runtime_eligible,
        binary64_euler_replay=binary_replay,
        exact_binary64_euler_replay_residual=binary_residual,
        binary64_euler_replay_matches=binary_matches,
        binary64_runtime_interval_identified=(
            binary_runtime_identified
        ),
        exact_metric_weights=metric,
        exact_explicit_euler_map=exact_map,
        exact_explicit_euler_endpoint_residual=map_residual,
        explicit_euler_map_identified=map_identified,
        exact_left_disagreement_energy=left_energy,
        exact_right_disagreement_energy=right_energy,
        exact_observed_disagreement_energy_gain=observed_gain,
        observed_disagreement_nonincrease=observed_nonincrease,
        exact_quotient_energy_gain_upper_bound=quotient_gain,
        _global_disagreement_contraction_certified=contracts,
        integrator_provenance_certified=False,
        future_or_repeated_schedule_stability_certified=False,
        scope=_SCOPE,
    )
    return replace(
        certificate,
        _proof_stamp=_nodal_flow_interval_proof_stamp(certificate),
    )
