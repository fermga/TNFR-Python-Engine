r"""Exact P2 reference family for three event/REMESH executions.

This module closes one deliberately small reference problem.  On the effective
two-node path with homogeneous positive capacity ``nu`` and no glyph jumps, the
pure-EPI pressure has one nonuniform mode.  For

``x0 = m * (1, 1) + d * (1, -1)`` and ``lambda = 2 * nu``,

the exact continuous amplitude at time ``T`` is ``d * exp(-lambda*T)``.  A
pressure-refreshed Euler partition with exact represented durations ``h_j``
has amplitude ``d * product_j(1 - lambda*h_j)``.  When every
``0 < lambda*h_j < 1``, exact rational arithmetic and rational exponential
enclosures certify

``0 <= exp(-lambda*T) - product_j(1-lambda*h_j)``
``   <= lambda**2/2 * sum_j(h_j**2)``
``   <= lambda**2/2 * T * max_j(h_j)``.

Proper positive subdivision strictly raises the Euler factor.  Thus the ideal
pre-REMESH error strictly decreases for nonzero ``d``.  With both REMESH delays
equal to one, the delayed row equal to ``x0`` and
``beta = (1-alpha)**2``, the ideal post-REMESH error is exactly ``beta`` times
the pre-REMESH error.  The committed binary64 endpoint is bounded by adding
the infinity norm of the bridge's signed rounding-plus-clipping residual.

The certificate binds three already executed runtime cycles with hard clipping
on one common scalar interval.  Rational enclosures are intentionally limited
to exponents at most 4096, bounding growth through the integer-exponent axis;
this is a representation limit, not a dynamical threshold. It does not bound
the total bit size induced by arbitrary Fraction inputs and is not a claim
about arbitrary glyphs, mixed channels, soft clipping, changing support or
metric, repeated execution, or binary64 asymptotic convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Any

from ..errors import TNFRValueError
from ..operators.event_remesh_runtime import EventRemeshCycleResult, _proof_value
from ..utils._structural_signature import (
    proof_stamps_are_identical,
)
from . import event_remesh_refinement as _refinement_module
from . import runtime_remesh_history_stability as _bridge_module
from .event_remesh_refinement import (
    EventRemeshEPICheckpointObservation,
    EventRemeshMeshObservation,
    EventRemeshPersistentEPIError,
    EventRemeshThreeMeshModalObservation,
    EventRemeshThreeMeshRefinementObservation,
    EventRemeshThreeMeshZHIRObservation,
)
from .remesh_history_stability import ExactVector
from .runtime_eigenmode_reference import (
    ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
    observe_executed_reversible_single_eigenmode_euler_reference,
)
from .runtime_remesh_history_stability import (
    RuntimeRemeshHistoryBridgeObservation,
)

__all__ = (
    "P2EventRemeshMeshReferenceObservation",
    "P2EventRemeshReferenceFamilyObservation",
    "observe_p2_event_remesh_reference_family",
)

_MESH_NAMES = ("coarse", "intermediate", "fine")
_MESH_PROOF_VERSION = "p2_event_remesh_mesh_reference_v1"
_FAMILY_PROOF_VERSION = "p2_event_remesh_reference_family_v1"
_SCOPE = (
    "One finite three-mesh reference family on the effective two-node path: "
    "event-free homogeneous positive-capacity pure-EPI flow, exact represented "
    "pressure-refresh durations and checkpoints, strictly stable positive "
    "Euler modal steps, tau_local=tau_global=1, the exact initial field as the "
    "delayed row, and hard REMESH clipping on one common scalar interval. "
    "Rational exponential intervals "
    "certify the ideal continuous/Euler error bound; executor-linked REMESH "
    "bridges add their signed rounding-plus-clipping residual norm. Arbitrary "
    "glyphs, mixed channels, soft clipping, changing support or metric, solver "
    "order, generic mesh convergence, repeated stability, binary64 asymptotic "
    "convergence and future behavior are outside scope."
)

_MESH_CONDITION_NAMES = (
    "cycle_and_runtime_bridge_canonical",
    "single_event_free_physical_flow",
    "effective_p2_conductance_fixed",
    "homogeneous_positive_capacity_fixed",
    "pure_epi_pressure_refreshed_at_every_boundary",
    "every_segment_runtime_euler_map_identified",
    "strict_positive_modal_step_domain",
    "exact_euler_recurrence_and_product_endpoint",
    "rational_continuous_factor_enclosure",
    "nonnegative_continuous_minus_euler_factor",
    "quadratic_factor_error_bound",
    "hmax_factor_error_bound",
    "unit_delays_and_exact_initial_delayed_row",
    "hard_clipping_common_scalar_interval",
    "exact_beta_identity",
    "ideal_remesh_affine_head_identity",
    "signed_rounding_plus_clipping_residual_identity",
    "runtime_post_remesh_triangle_bound",
)

_FAMILY_CONDITION_NAMES = (
    "canonical_three_mesh_observation_bound",
    "all_runtime_bridges_bound_by_identity",
    "common_exact_p2_reference_problem",
    "coarse_to_intermediate_proper_positive_subdivision",
    "intermediate_to_fine_proper_positive_subdivision",
    "strict_exact_euler_factor_improvement",
    "strict_exact_quadratic_bound_improvement",
    "common_rational_continuous_factor_enclosure",
    "every_mesh_reference_observation_intact",
)


def _raw_stamp_or_none(value: Any) -> tuple[Any, ...] | None:
    try:
        stamp = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return None
    return stamp if type(stamp) is tuple else None


def _compact_proof_value(value: Any) -> Any:
    nested_types = (
        EventRemeshThreeMeshRefinementObservation,
        RuntimeRemeshHistoryBridgeObservation,
    )
    mesh_type = globals().get("P2EventRemeshMeshReferenceObservation")
    if mesh_type is not None:
        nested_types = (*nested_types, mesh_type)
    if type(value) in nested_types:
        stamp = _raw_stamp_or_none(value)
        version = stamp[0] if stamp and type(stamp[0]) is str else None
        return (
            "tnfr-validated-proof-identity-v1",
            type(value).__module__,
            type(value).__qualname__,
            id(value),
            version,
        )
    if type(value) is EventRemeshCycleResult:
        stamp = _raw_stamp_or_none(value)
        return (
            "tnfr-nested-cycle-proof-v1",
            id(value),
            stamp,
        )
    if type(value) is tuple:
        return ("tuple", tuple(_compact_proof_value(item) for item in value))
    return _proof_value(value)


def _stamp_from_values(
    version: str,
    field_names: tuple[str, ...],
    values: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        version,
        tuple(
            (name, _compact_proof_value(values[name])) for name in field_names
        ),
    )


def _object_values(value: Any, expected_type: type[Any]) -> dict[str, Any]:
    if type(value) is not expected_type:
        raise TypeError("proof value must have its canonical result type")
    return {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(expected_type)
        if item.name != "_proof_stamp"
    }


def _same(left: Any, right: Any) -> bool:
    try:
        return proof_stamps_are_identical(_proof_value(left), _proof_value(right))
    except BaseException:
        return False


def _same_stamp(left: Any, right: Any) -> bool:
    return bool(
        type(left) is type(right)
        and proof_stamps_are_identical(
            _raw_stamp_or_none(left),
            _raw_stamp_or_none(right),
        )
    )


def _exact_binary64_vector(values: tuple[float, ...]) -> ExactVector:
    return tuple(Fraction.from_float(value) for value in values)


def _strict_exact_vector(value: Any, width: int = 2) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == width
        and all(type(item) is Fraction for item in value)
    )


def _strict_conditions(
    value: Any,
    expected_names: tuple[str, ...],
) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(expected_names)
        and all(
            type(item) is tuple
            and len(item) == 2
            and item[0] == expected_names[index]
            and type(item[1]) is bool
            for index, item in enumerate(value)
        )
    )


def _max_abs(value: ExactVector) -> Fraction:
    return max((abs(item) for item in value), default=Fraction(0))


def _p2_field(mean: Fraction, amplitude: Fraction) -> ExactVector:
    return mean + amplitude, mean - amplitude


def _p2_pressure(value: ExactVector) -> ExactVector:
    return value[1] - value[0], value[0] - value[1]


def _p2_conductance(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == 2
        and all(type(row) is tuple and len(row) == 2 for row in value)
        and all(type(item) is Fraction for row in value for item in row)
        and value[0][0] == 0
        and value[1][1] == 0
        and value[0][1] == value[1][0]
        and value[0][1] > 0
    )


def _partition_for_cycle(
    cycle: EventRemeshCycleResult,
    *,
    proof_already_validated: bool = False,
) -> Any:
    execution = cycle.event_execution
    if execution.schedule.operator_names:
        raise TNFRValueError("the P2 reference schedule must contain no glyphs")
    if execution.positive_flow_interval_indices != (0,):
        raise TNFRValueError(
            "the P2 reference requires exactly one positive flow interval"
        )
    evidence = execution.physical_flow_partition_evidence
    if len(evidence) != 1 or evidence[0].partition.parent_interval.index != 0:
        raise TNFRValueError(
            "the P2 reference requires one executor-owned physical partition"
        )
    partition = evidence[0]
    if not proof_already_validated and not partition._proof_fields_are_intact():
        raise TNFRValueError("physical partition evidence is not intact")
    return partition


def _validated_runtime_bridge(
    cycle: EventRemeshCycleResult,
    bridge: RuntimeRemeshHistoryBridgeObservation,
) -> RuntimeRemeshHistoryBridgeObservation:
    """Rederive one bridge while validating its cycle exactly once.

    The bridge's public integrity predicate validates the cycle before invoking
    a builder that validates it again.  A family query instead uses this
    ephemeral helper: the local bridge seal detects unsealed field mutation,
    the canonical builder validates the complete cycle and transition once,
    and the exact proof-stamp comparison rejects privately resealed values.
    Nothing is cached across calls.
    """

    if (
        type(bridge) is not RuntimeRemeshHistoryBridgeObservation
        or bridge.cycle_result is not cycle
        or not _bridge_module._sealed(bridge)
    ):
        raise TNFRValueError(
            "stored runtime REMESH bridge is forged, stale, or unbound"
        )
    expected = _bridge_module._build_runtime_remesh_history_bridge(
        cycle,
        verify_result=False,
        exact_transition_override=bridge.exact_transition,
    )
    if not _same_stamp(bridge, expected):
        raise TNFRValueError(
            "stored runtime REMESH bridge is forged, stale, or unbound"
        )
    return bridge


def _refinement_owned_proofs_are_locally_sealed(
    value: EventRemeshThreeMeshRefinementObservation,
) -> bool:
    """Check every refinement-owned seal without revalidating its cycles.

    Cycle, schedule-composition and executor evidence is validated separately
    by the runtime-bridge reconstruction.  This traversal covers direct
    mutation of every proof object constructed by the refinement observer.
    """

    try:
        if not _refinement_module._sealed(
            value,
            EventRemeshThreeMeshRefinementObservation,
            _refinement_module._THREE_MESH_PROOF_VERSION,
        ):
            return False
        meshes = (value.coarse, value.intermediate, value.fine)
        if any(
            type(mesh) is not EventRemeshMeshObservation
            or not _refinement_module._sealed(
                mesh,
                EventRemeshMeshObservation,
                _refinement_module._MESH_PROOF_VERSION,
            )
            or any(
                type(checkpoint) is not EventRemeshEPICheckpointObservation
                or not _refinement_module._sealed(
                    checkpoint,
                    EventRemeshEPICheckpointObservation,
                    _refinement_module._CHECKPOINT_PROOF_VERSION,
                )
                for checkpoint in mesh.checkpoints
            )
            for mesh in meshes
        ):
            return False
        if any(
            type(item) is not EventRemeshPersistentEPIError
            or not _refinement_module._sealed(
                item,
                EventRemeshPersistentEPIError,
                _refinement_module._ERROR_PROOF_VERSION,
            )
            for group in (
                value.coarse_intermediate_epi_errors,
                value.intermediate_fine_epi_errors,
                value.coarse_fine_epi_errors,
            )
            for item in group
        ):
            return False
        if any(
            type(item) is not EventRemeshThreeMeshZHIRObservation
            or not item._proof_fields_are_intact()
            for item in value.zhir_observations
        ):
            return False
        return all(
            type(item) is EventRemeshThreeMeshModalObservation
            and _refinement_module._sealed(
                item,
                EventRemeshThreeMeshModalObservation,
                _refinement_module._MODAL_PROOF_VERSION,
            )
            for item in value.modal_observations
        )
    except BaseException:
        return False


def _refinement_from_validated_cycles(
    cycles: tuple[EventRemeshCycleResult, ...],
) -> EventRemeshThreeMeshRefinementObservation:
    """Rederive the canonical refinement after the cycles were validated.

    This is the validation-context counterpart of the public refinement
    observer.  It deliberately omits only that observer's initial cycle
    validation, which the bridge reconstruction immediately before this call
    has already completed.
    """

    meshes = tuple(
        _refinement_module._mesh_observation(name, cycle)
        for name, cycle in zip(_MESH_NAMES, cycles, strict=True)
    )
    coarse_to_intermediate = _refinement_module._strictly_nested_boundaries(
        meshes[0], meshes[1]
    )
    intermediate_to_fine = _refinement_module._strictly_nested_boundaries(
        meshes[1], meshes[2]
    )
    if not coarse_to_intermediate or not intermediate_to_fine:
        raise TNFRValueError("P2 physical partitions are not strictly nested")
    persistent_nodes, persistent_tokens = _refinement_module._persistent_support(
        meshes[0].nodes,
        meshes[1].nodes,
        meshes[2].nodes,
    )
    typed_meshes = (meshes[0], meshes[1], meshes[2])
    _refinement_module._require_common_reference_problem(
        typed_meshes,
        persistent_tokens,
    )
    coarse_intermediate_errors = _refinement_module._pairwise_errors(
        meshes[0], meshes[1], persistent_nodes, persistent_tokens
    )
    intermediate_fine_errors = _refinement_module._pairwise_errors(
        meshes[1], meshes[2], persistent_nodes, persistent_tokens
    )
    coarse_fine_errors = _refinement_module._pairwise_errors(
        meshes[0], meshes[2], persistent_nodes, persistent_tokens
    )
    zhir, zhir_abstention = _refinement_module._physical_zhir_observations(
        meshes[0],
        meshes[1],
        meshes[2],
        persistent_nodes,
        persistent_tokens,
        None,
    )
    modal = _refinement_module._modal_observations(
        meshes[0], meshes[1], meshes[2]
    )
    supports_equal = bool(
        _refinement_module._ordered_supports_match(
            meshes[0].nodes, meshes[1].nodes
        )
        and _refinement_module._ordered_supports_match(
            meshes[1].nodes, meshes[2].nodes
        )
    )
    conditions = (
        ("coarse_cycle_proof_intact", True),
        ("intermediate_cycle_proof_intact", True),
        ("fine_cycle_proof_intact", True),
        ("same_schedule", True),
        ("every_positive_interval_physically_partitioned", True),
        ("identical_ordered_full_support", supports_equal),
        (
            "coarse_to_intermediate_strict_refinement",
            coarse_to_intermediate,
        ),
        (
            "intermediate_to_fine_strict_refinement",
            intermediate_to_fine,
        ),
        ("persistent_node_support_nonempty", bool(persistent_nodes)),
        ("initial_epi_capacity_phase_pressure_compatible", True),
        ("first_captured_post_refresh_nodal_input_compatible", True),
        ("captured_effective_conductance_compatible", True),
        ("normalized_metric_compatible", True),
        ("executor_integrator_metadata_compatible", True),
        ("pressure_callback_metadata_compatible", True),
        ("incoming_remesh_history_compatible", True),
        ("remesh_configuration_compatible", True),
    )
    value = EventRemeshThreeMeshRefinementObservation(
        coarse=meshes[0],
        intermediate=meshes[1],
        fine=meshes[2],
        persistent_nodes=persistent_nodes,
        supports_equal_across_meshes=supports_equal,
        coarse_to_intermediate_strict_refinement=coarse_to_intermediate,
        intermediate_to_fine_strict_refinement=intermediate_to_fine,
        coarse_intermediate_epi_errors=coarse_intermediate_errors,
        intermediate_fine_epi_errors=intermediate_fine_errors,
        coarse_fine_epi_errors=coarse_fine_errors,
        zhir_xi=None,
        zhir_observations=zhir,
        zhir_abstention_reason=zhir_abstention,
        modal_observations=modal,
        conditions=conditions,
    )
    return _refinement_module._seal(
        value,
        EventRemeshThreeMeshRefinementObservation,
        _refinement_module._THREE_MESH_PROOF_VERSION,
    )


def _canonical_nested_observations(
    cycles: tuple[EventRemeshCycleResult, ...],
    *,
    refinement_override: EventRemeshThreeMeshRefinementObservation | None = None,
    bridge_overrides: tuple[RuntimeRemeshHistoryBridgeObservation, ...] | None = None,
) -> tuple[
    EventRemeshThreeMeshRefinementObservation,
    tuple[RuntimeRemeshHistoryBridgeObservation, ...],
]:
    if (
        type(cycles) is not tuple
        or len(cycles) != 3
        or any(type(item) is not EventRemeshCycleResult for item in cycles)
    ):
        raise TypeError("coarse, intermediate and fine must be cycle results")

    if bridge_overrides is None:
        bridges = tuple(
            _bridge_module._build_runtime_remesh_history_bridge(
                cycle,
                verify_result=False,
            )
            for cycle in cycles
        )
    else:
        if type(bridge_overrides) is not tuple or len(bridge_overrides) != 3:
            raise TNFRValueError(
                "stored runtime REMESH bridge is forged, stale, or unbound"
            )
        bridges = tuple(
            _validated_runtime_bridge(cycle, bridge)
            for cycle, bridge in zip(cycles, bridge_overrides, strict=True)
        )

    canonical_refinement = _refinement_from_validated_cycles(cycles)
    if refinement_override is None:
        refinement = canonical_refinement
    else:
        meshes = (
            (
                refinement_override.coarse,
                refinement_override.intermediate,
                refinement_override.fine,
            )
            if type(refinement_override)
            is EventRemeshThreeMeshRefinementObservation
            else ()
        )
        if (
            type(refinement_override)
            is not EventRemeshThreeMeshRefinementObservation
            or not _refinement_owned_proofs_are_locally_sealed(
                refinement_override
            )
            or len(meshes) != 3
            or any(
                mesh.cycle_result is not cycle
                for mesh, cycle in zip(meshes, cycles, strict=True)
            )
            or not _same_stamp(refinement_override, canonical_refinement)
        ):
            raise TNFRValueError(
                "stored three-mesh observation is forged, stale, or unbound"
            )
        refinement = refinement_override
    return refinement, bridges


@dataclass(frozen=True, slots=True)
class P2EventRemeshMeshReferenceObservation:
    """One mesh row in the exact P2 event/REMESH reference family."""

    mesh_name: str
    cycle_result: EventRemeshCycleResult = field(repr=False)
    runtime_bridge: RuntimeRemeshHistoryBridgeObservation = field(repr=False)
    nodes: tuple[Any, Any]
    exact_initial_field: ExactVector
    exact_mean: Fraction
    exact_initial_amplitude: Fraction
    exact_nu_f: Fraction
    exact_lambda: Fraction
    exact_total_duration: Fraction
    exact_segment_durations: tuple[Fraction, ...]
    exact_scaled_segment_durations: tuple[Fraction, ...]
    exact_euler_segment_factors: tuple[Fraction, ...]
    exact_euler_factor: Fraction
    exact_continuous_factor_lower_bound: Fraction
    exact_continuous_factor_upper_bound: Fraction
    exact_factor_error_lower_bound: Fraction
    exact_factor_error_upper_bound: Fraction
    exact_quadratic_factor_error_upper_bound: Fraction
    exact_hmax_factor_error_upper_bound: Fraction
    exact_factor_error_symbolic_coefficients: tuple[Fraction, Fraction]
    exact_observed_pre_remesh_field: ExactVector
    exact_euler_reference_pre_remesh_field: ExactVector
    exact_pre_remesh_error_lower_bound: Fraction
    exact_pre_remesh_error_upper_bound: Fraction
    exact_pre_remesh_quadratic_error_upper_bound: Fraction
    exact_pre_remesh_hmax_error_upper_bound: Fraction
    exact_alpha: Fraction
    exact_beta: Fraction
    exact_clip_lower_bound: Fraction
    exact_clip_upper_bound: Fraction
    exact_ideal_remesh_euler_field: ExactVector
    exact_runtime_remesh_field: ExactVector
    exact_signed_rounding_residual: ExactVector
    exact_signed_clipping_residual: ExactVector
    exact_signed_total_residual: ExactVector
    exact_total_residual_linf: Fraction
    exact_ideal_post_remesh_error_symbolic_coefficients: tuple[
        Fraction,
        Fraction,
    ]
    exact_ideal_post_remesh_error_lower_bound: Fraction
    exact_ideal_post_remesh_error_upper_bound: Fraction
    exact_runtime_post_remesh_error_upper_bound: Fraction
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            current = _object_values(
                self,
                P2EventRemeshMeshReferenceObservation,
            )
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                _stamp_from_values(
                    _MESH_PROOF_VERSION,
                    _MESH_FIELD_NAMES,
                    current,
                ),
            ):
                return False
            cycle = object.__getattribute__(self, "cycle_result")
            bridge = object.__getattribute__(self, "runtime_bridge")
            if (
                type(bridge) is not RuntimeRemeshHistoryBridgeObservation
                or bridge.cycle_result is not cycle
            ):
                return False
            _validated_runtime_bridge(cycle, bridge)
            partition = _partition_for_cycle(
                cycle,
                proof_already_validated=True,
            )
            expected = _derive_mesh_values(
                object.__getattribute__(self, "mesh_name"),
                cycle,
                bridge,
                bridge_already_validated=True,
                partition_override=partition,
            )
            return proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                _stamp_from_values(
                    _MESH_PROOF_VERSION,
                    _MESH_FIELD_NAMES,
                    expected,
                ),
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def mesh_reference_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("p2_mesh_reference_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_real_euler_error_bound_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_ideal_remesh_error_scaling_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def runtime_residual_error_bound_certified(self) -> bool:
        return self._proof_fields_are_intact()


_MESH_FIELD_NAMES = tuple(
    item.name
    for item in fields(P2EventRemeshMeshReferenceObservation)
    if item.name != "_proof_stamp"
)


def _derive_mesh_values(
    mesh_name: str,
    cycle: EventRemeshCycleResult,
    bridge: RuntimeRemeshHistoryBridgeObservation,
    *,
    runtime_reference: (
        ExecutedReversibleSingleEigenmodeEulerReferenceObservation | None
    ) = None,
    runtime_partition_index: int = 0,
    runtime_reference_already_validated: bool = False,
    bridge_already_validated: bool = False,
    partition_override: Any | None = None,
) -> dict[str, Any]:
    if mesh_name not in _MESH_NAMES:
        raise TNFRValueError("mesh_name must be coarse, intermediate, or fine")
    if type(cycle) is not EventRemeshCycleResult:
        raise TypeError("cycle must be an EventRemeshCycleResult")
    if (
        type(bridge) is not RuntimeRemeshHistoryBridgeObservation
        or bridge.cycle_result is not cycle
        or (
            not bridge_already_validated
            and not bridge.bridge_observation_certified
        )
    ):
        raise TNFRValueError("runtime REMESH bridge is not canonical for its cycle")

    canonical_partition = _partition_for_cycle(
        cycle,
        proof_already_validated=partition_override is not None,
    )
    if partition_override is not None and partition_override is not canonical_partition:
        raise TNFRValueError("physical partition evidence is not bound to its cycle")
    partition = canonical_partition
    nodes = cycle.target_nodes
    if type(nodes) is not tuple or len(nodes) != 2:
        raise TNFRValueError("the P2 reference requires exactly two ordered nodes")
    initial = _exact_binary64_vector(cycle.pre_schedule_epi.epi_values)
    if len(initial) != 2:
        raise TNFRValueError("initial EPI must have exactly two coordinates")

    boundaries = partition.boundary_observations
    flows = partition.segment_flow_evidence
    first = boundaries[0].after
    capacity = first.exact_nu_f
    if len(capacity) != 2 or capacity[0] <= 0 or capacity[0] != capacity[1]:
        raise TNFRValueError(
            "the P2 reference requires homogeneous positive capacity"
        )
    nu_f = capacity[0]
    lambda_ = 2 * nu_f
    conductance = first.conductance
    if not _p2_conductance(conductance):
        raise TNFRValueError("effective conductance is not a fixed P2 edge")
    if any(
        boundary.after.exact_nu_f != capacity
        or boundary.after.conductance != conductance
        or not _p2_conductance(boundary.after.conductance)
        for boundary in boundaries
    ):
        raise TNFRValueError("capacity or effective P2 conductance changed")

    durations = tuple(
        segment.exact_duration for segment in partition.partition.segments
    )
    if not durations or any(item <= 0 for item in durations):
        raise TNFRValueError("physical segment durations must be positive")
    total_duration = sum(durations, Fraction(0))
    if total_duration != partition.partition.parent_interval.exact_duration:
        raise TNFRValueError("physical durations do not sum to the parent interval")

    if runtime_reference is None:
        runtime_binding = (
            observe_executed_reversible_single_eigenmode_euler_reference(
                (partition,)
            )
        )
        modal_index = 0
        runtime_reference_already_validated = True
    else:
        runtime_binding = runtime_reference
        modal_index = runtime_partition_index
    if (
        type(runtime_binding)
        is not ExecutedReversibleSingleEigenmodeEulerReferenceObservation
        or type(modal_index) is not int
        or modal_index < 0
        or modal_index >= len(runtime_binding.partition_observations)
        or (
            not runtime_reference_already_validated
            and not runtime_binding.runtime_reference_binding_certified
        )
    ):
        raise TNFRValueError("the runtime eigenmode reference is not canonical")
    modal_reference = runtime_binding.reference_certificate
    runtime_row = runtime_binding.partition_observations[modal_index]
    if (
        runtime_row.reference_certificate is not modal_reference
        or runtime_row.execution is not partition
        or runtime_row.partition_index != modal_index
        or runtime_row.exact_segment_durations != durations
        or runtime_row.nodes != nodes
    ):
        raise TNFRValueError(
            "runtime eigenmode row is not bound to the P2 physical partition"
        )

    mean = modal_reference.exact_weighted_mean
    mode = modal_reference.exact_centered_mode
    amplitude = mode[0]
    kernel_matches_p2 = bool(
        modal_reference.exact_conductance == conductance
        and modal_reference.exact_nu_f == capacity
        and modal_reference.exact_initial_epi == initial
        and modal_reference.exact_partitions[modal_index] == durations
        and modal_reference.exact_total_duration == total_duration
        and mean == (initial[0] + initial[1]) / 2
        and mode == (amplitude, -amplitude)
        and amplitude != 0
        and modal_reference.exact_mode_eigenvalue == lambda_
        and modal_reference.exact_mode_residual == (Fraction(0), Fraction(0))
    )
    if not kernel_matches_p2:
        raise TNFRValueError(
            "runtime data do not match the exact reversible P2 eigenmode kernel"
        )
    scaled = modal_reference.exact_scaled_partitions[modal_index]
    segment_factors = modal_reference.exact_euler_segment_factors[modal_index]
    euler_factor = modal_reference.exact_euler_factors[modal_index]

    zero = (Fraction(0), Fraction(0))
    if (
        not runtime_row.all_segment_exact_affine_maps_identified
        or any(
            residual != zero
            for residual in (
                *runtime_row.exact_pressure_realization_residuals,
                *runtime_row.exact_held_input_execution_residuals,
                *runtime_row.exact_local_runtime_defects,
            )
        )
        or runtime_row.exact_endpoint_runtime_defect != zero
    ):
        raise TNFRValueError(
            "every P2 segment must identify one exact zero-defect Euler map"
        )
    running_factor = Fraction(1)
    for index, (duration, flow) in enumerate(zip(durations, flows, strict=True)):
        expected_left = _p2_field(mean, amplitude * running_factor)
        boundary_left = boundaries[index].after
        if (
            boundary_left.exact_epi != expected_left
            or boundary_left.exact_delta_nfr != _p2_pressure(expected_left)
            or runtime_row.exact_runtime_boundary_epi[index]
            != expected_left
            or runtime_row.exact_reference_boundary_epi[index]
            != expected_left
        ):
            raise TNFRValueError(
                "represented checkpoints do not follow the exact P2 recurrence"
            )
        certificate = flow.certificate
        if (
            certificate is None
            or certificate.exact_duration != duration
            or certificate.left.exact_epi != expected_left
            or certificate.left.exact_delta_nfr != _p2_pressure(expected_left)
        ):
            raise TNFRValueError("one segment lacks exact runtime P2 Euler evidence")
        running_factor *= segment_factors[index]
        expected_right = _p2_field(mean, amplitude * running_factor)
        if (
            certificate.right.exact_epi != expected_right
            or runtime_row.exact_runtime_boundary_epi[index + 1]
            != expected_right
            or runtime_row.exact_reference_boundary_epi[index + 1]
            != expected_right
        ):
            raise TNFRValueError("one represented Euler endpoint is not exact")
    expected_pre_remesh = _p2_field(mean, amplitude * euler_factor)
    terminal = boundaries[-1].after
    observed_pre_remesh = _exact_binary64_vector(cycle.pre_remesh_epi.epi_values)
    if (
        running_factor != euler_factor
        or terminal.exact_epi != expected_pre_remesh
        or terminal.exact_delta_nfr != _p2_pressure(expected_pre_remesh)
        or observed_pre_remesh != expected_pre_remesh
    ):
        raise TNFRValueError("pre-REMESH endpoint is not the exact Euler product")

    exp_lower = modal_reference.exact_continuous_factor_lower_bound
    exp_upper = modal_reference.exact_continuous_factor_upper_bound
    error_lower = modal_reference.exact_factor_error_lower_bounds[modal_index]
    error_upper = modal_reference.exact_factor_error_upper_bounds[modal_index]
    quadratic = modal_reference.exact_quadratic_factor_error_upper_bounds[
        modal_index
    ]
    hmax = modal_reference.exact_hmax_factor_error_upper_bounds[modal_index]
    if error_lower < 0 or error_upper < error_lower:
        raise RuntimeError("rational enclosure does not prove nonnegative error")
    if error_upper > quadratic or quadratic > hmax:
        raise RuntimeError("P2 Euler error inequalities failed")

    certificate = bridge.exact_transition.certificate
    alpha = certificate.alpha
    beta = (1 - alpha) ** 2
    if (
        certificate.tau_local != 1
        or certificate.tau_global != 1
        or certificate.beta != beta
        or bridge.exact_history != (expected_pre_remesh, initial)
    ):
        raise TNFRValueError(
            "REMESH must use unit delays and the exact initial delayed row"
        )
    if cycle.remesh.plan.clip_mode != "hard":
        raise TNFRValueError("the P2 runtime reference requires hard clipping")
    clip_lower = Fraction.from_float(cycle.remesh.plan.epi_min)
    clip_upper = Fraction.from_float(cycle.remesh.plan.epi_max)
    if clip_lower >= clip_upper:
        raise TNFRValueError("the hard clipping interval must be nonempty")
    ideal_remesh = tuple(
        beta * current + (1 - beta) * delayed
        for current, delayed in zip(expected_pre_remesh, initial, strict=True)
    )
    if bridge.exact_ideal_next_field != ideal_remesh:
        raise TNFRValueError("ideal REMESH head does not realize the beta map")
    runtime_remesh = bridge.exact_runtime_bounded_next_field
    if runtime_remesh != _exact_binary64_vector(cycle.post_remesh_epi.epi_values):
        raise TNFRValueError("runtime bridge does not bind the committed endpoint")
    total_residual = tuple(
        rounding + clipping
        for rounding, clipping in zip(
            bridge.exact_rounding_residual,
            bridge.exact_clipping_residual,
            strict=True,
        )
    )
    if total_residual != bridge.exact_total_residual:
        raise RuntimeError("signed runtime residual decomposition failed")
    residual_linf = _max_abs(total_residual)
    if residual_linf != bridge.exact_max_abs_total_residual:
        raise RuntimeError("runtime residual norm is inconsistent")

    absolute_amplitude = abs(amplitude)
    pre_lower = modal_reference.exact_linf_error_lower_bounds[modal_index]
    pre_upper = modal_reference.exact_linf_error_upper_bounds[modal_index]
    pre_quadratic = (
        modal_reference.exact_linf_quadratic_error_upper_bounds[modal_index]
    )
    pre_hmax = modal_reference.exact_linf_hmax_error_upper_bounds[modal_index]
    if (
        modal_reference.exact_mode_linf_norm != absolute_amplitude
        or pre_lower != absolute_amplitude * error_lower
        or pre_upper != absolute_amplitude * error_upper
        or pre_quadratic != absolute_amplitude * quadratic
        or pre_hmax != absolute_amplitude * hmax
    ):
        raise RuntimeError("P2 error fields diverged from the exact modal kernel")
    post_lower = beta * pre_lower
    post_upper = beta * pre_upper
    runtime_post_upper = post_upper + residual_linf
    symbolic_pre = (-euler_factor, Fraction(1))
    symbolic_post = (-beta * euler_factor, beta)
    conditions = (
        ("cycle_and_runtime_bridge_canonical", True),
        ("single_event_free_physical_flow", True),
        ("effective_p2_conductance_fixed", True),
        ("homogeneous_positive_capacity_fixed", True),
        ("pure_epi_pressure_refreshed_at_every_boundary", True),
        ("every_segment_runtime_euler_map_identified", True),
        ("strict_positive_modal_step_domain", True),
        ("exact_euler_recurrence_and_product_endpoint", True),
        ("rational_continuous_factor_enclosure", True),
        ("nonnegative_continuous_minus_euler_factor", error_lower >= 0),
        ("quadratic_factor_error_bound", error_upper <= quadratic),
        ("hmax_factor_error_bound", quadratic <= hmax),
        ("unit_delays_and_exact_initial_delayed_row", True),
        ("hard_clipping_common_scalar_interval", True),
        ("exact_beta_identity", certificate.beta == beta),
        ("ideal_remesh_affine_head_identity", True),
        (
            "signed_rounding_plus_clipping_residual_identity",
            total_residual == bridge.exact_total_residual,
        ),
        (
            "runtime_post_remesh_triangle_bound",
            runtime_post_upper == beta * pre_upper + residual_linf,
        ),
    )
    if not _strict_conditions(conditions, _MESH_CONDITION_NAMES) or not all(
        passed for _, passed in conditions
    ):
        raise RuntimeError("P2 mesh reference conditions are inconsistent")

    return {
        "mesh_name": mesh_name,
        "cycle_result": cycle,
        "runtime_bridge": bridge,
        "nodes": nodes,
        "exact_initial_field": initial,
        "exact_mean": mean,
        "exact_initial_amplitude": amplitude,
        "exact_nu_f": nu_f,
        "exact_lambda": lambda_,
        "exact_total_duration": total_duration,
        "exact_segment_durations": durations,
        "exact_scaled_segment_durations": scaled,
        "exact_euler_segment_factors": segment_factors,
        "exact_euler_factor": euler_factor,
        "exact_continuous_factor_lower_bound": exp_lower,
        "exact_continuous_factor_upper_bound": exp_upper,
        "exact_factor_error_lower_bound": error_lower,
        "exact_factor_error_upper_bound": error_upper,
        "exact_quadratic_factor_error_upper_bound": quadratic,
        "exact_hmax_factor_error_upper_bound": hmax,
        "exact_factor_error_symbolic_coefficients": symbolic_pre,
        "exact_observed_pre_remesh_field": observed_pre_remesh,
        "exact_euler_reference_pre_remesh_field": expected_pre_remesh,
        "exact_pre_remesh_error_lower_bound": pre_lower,
        "exact_pre_remesh_error_upper_bound": pre_upper,
        "exact_pre_remesh_quadratic_error_upper_bound": pre_quadratic,
        "exact_pre_remesh_hmax_error_upper_bound": pre_hmax,
        "exact_alpha": alpha,
        "exact_beta": beta,
        "exact_clip_lower_bound": clip_lower,
        "exact_clip_upper_bound": clip_upper,
        "exact_ideal_remesh_euler_field": ideal_remesh,
        "exact_runtime_remesh_field": runtime_remesh,
        "exact_signed_rounding_residual": bridge.exact_rounding_residual,
        "exact_signed_clipping_residual": bridge.exact_clipping_residual,
        "exact_signed_total_residual": total_residual,
        "exact_total_residual_linf": residual_linf,
        "exact_ideal_post_remesh_error_symbolic_coefficients": symbolic_post,
        "exact_ideal_post_remesh_error_lower_bound": post_lower,
        "exact_ideal_post_remesh_error_upper_bound": post_upper,
        "exact_runtime_post_remesh_error_upper_bound": runtime_post_upper,
        "conditions": conditions,
    }


def _build_mesh(
    mesh_name: str,
    cycle: EventRemeshCycleResult,
    bridge: RuntimeRemeshHistoryBridgeObservation,
    runtime_reference: ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
    runtime_partition_index: int,
    partition: Any,
) -> P2EventRemeshMeshReferenceObservation:
    values = _derive_mesh_values(
        mesh_name,
        cycle,
        bridge,
        runtime_reference=runtime_reference,
        runtime_partition_index=runtime_partition_index,
        runtime_reference_already_validated=True,
        bridge_already_validated=True,
        partition_override=partition,
    )
    return P2EventRemeshMeshReferenceObservation(
        **values,
        _proof_stamp=_stamp_from_values(
            _MESH_PROOF_VERSION,
            _MESH_FIELD_NAMES,
            values,
        ),
    )


@dataclass(frozen=True, slots=True)
class P2EventRemeshReferenceFamilyObservation:
    """Sealed three-mesh P2 reference theorem and runtime bindings."""

    refinement: EventRemeshThreeMeshRefinementObservation = field(repr=False)
    meshes: tuple[
        P2EventRemeshMeshReferenceObservation,
        P2EventRemeshMeshReferenceObservation,
        P2EventRemeshMeshReferenceObservation,
    ]
    nodes: tuple[Any, Any]
    exact_initial_field: ExactVector
    exact_mean: Fraction
    exact_initial_amplitude: Fraction
    exact_nu_f: Fraction
    exact_lambda: Fraction
    exact_total_duration: Fraction
    exact_alpha: Fraction
    exact_beta: Fraction
    exact_clip_lower_bound: Fraction
    exact_clip_upper_bound: Fraction
    exact_continuous_factor_lower_bound: Fraction
    exact_continuous_factor_upper_bound: Fraction
    exact_euler_factors: tuple[Fraction, Fraction, Fraction]
    exact_euler_factor_improvements: tuple[Fraction, Fraction]
    exact_quadratic_factor_error_upper_bounds: tuple[
        Fraction,
        Fraction,
        Fraction,
    ]
    exact_quadratic_bound_improvements: tuple[Fraction, Fraction]
    strict_pre_remesh_error_improvement: bool
    strict_ideal_post_remesh_error_improvement: bool
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            current = _object_values(
                self,
                P2EventRemeshReferenceFamilyObservation,
            )
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                _stamp_from_values(
                    _FAMILY_PROOF_VERSION,
                    _FAMILY_FIELD_NAMES,
                    current,
                ),
            ):
                return False
            refinement = object.__getattribute__(self, "refinement")
            stored_meshes = object.__getattribute__(self, "meshes")
            if (
                type(stored_meshes) is not tuple
                or len(stored_meshes) != 3
                or any(
                    type(item) is not P2EventRemeshMeshReferenceObservation
                    for item in stored_meshes
                )
            ):
                return False
            cycles = tuple(item.cycle_result for item in stored_meshes)
            bridges = tuple(item.runtime_bridge for item in stored_meshes)
            expected = _derive_family_values(
                cycles,
                refinement_override=refinement,
                bridge_overrides=bridges,
                mesh_overrides=stored_meshes,
            )
            return proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                _stamp_from_values(
                    _FAMILY_PROOF_VERSION,
                    _FAMILY_FIELD_NAMES,
                    expected,
                ),
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def reference_family_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("p2_event_remesh_reference_family_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def exact_p2_euler_consistency_bound_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def strict_proper_subdivision_improvement_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.strict_pre_remesh_error_improvement
        )

    @property
    def exact_ideal_remesh_error_scaling_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def runtime_residual_error_bounds_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def binary64_asymptotic_convergence_certified(self) -> bool:
        return False

    @property
    def arbitrary_glyph_or_mixed_mode_certified(self) -> bool:
        return False

    @property
    def soft_clipping_certified(self) -> bool:
        return False

    @property
    def changing_support_or_metric_certified(self) -> bool:
        return False

    @property
    def solver_order_certified(self) -> bool:
        return False

    @property
    def generic_mesh_convergence_certified(self) -> bool:
        return False

    @property
    def repeated_runtime_stability_certified(self) -> bool:
        return False

    @property
    def future_stability_certified(self) -> bool:
        return False


_FAMILY_FIELD_NAMES = tuple(
    item.name
    for item in fields(P2EventRemeshReferenceFamilyObservation)
    if item.name != "_proof_stamp"
)


def _derive_family_values(
    cycles: tuple[EventRemeshCycleResult, ...],
    *,
    refinement_override: EventRemeshThreeMeshRefinementObservation | None = None,
    bridge_overrides: tuple[RuntimeRemeshHistoryBridgeObservation, ...] | None = None,
    mesh_overrides: tuple[P2EventRemeshMeshReferenceObservation, ...] | None = None,
) -> dict[str, Any]:
    refinement, bridges = _canonical_nested_observations(
        cycles,
        refinement_override=refinement_override,
        bridge_overrides=bridge_overrides,
    )
    partitions = tuple(
        _partition_for_cycle(cycle, proof_already_validated=True)
        for cycle in cycles
    )
    durations = tuple(
        tuple(segment.exact_duration for segment in item.partition.segments)
        for item in partitions
    )
    initial_fields = tuple(
        _exact_binary64_vector(cycle.pre_schedule_epi.epi_values)
        for cycle in cycles
    )
    if not initial_fields[0] == initial_fields[1] == initial_fields[2]:
        raise TNFRValueError("P2 reference meshes require one exact initial field")
    initial_states = tuple(
        item.boundary_observations[0].after for item in partitions
    )
    first_capacity = initial_states[0].exact_nu_f
    capacities = tuple(
        state.exact_nu_f for state in initial_states
    )
    if (
        len(first_capacity) != 2
        or first_capacity[0] <= 0
        or first_capacity[0] != first_capacity[1]
        or any(item != first_capacity for item in capacities[1:])
    ):
        raise TNFRValueError("P2 reference capacity is not homogeneous")
    conductances = tuple(state.conductance for state in initial_states)
    if any(
        not _p2_conductance(value) or value != conductances[0]
        for value in conductances
    ):
        raise TNFRValueError(
            "P2 reference conductance must be one fixed effective edge"
        )
    runtime_reference = (
        observe_executed_reversible_single_eigenmode_euler_reference(
            partitions
        )
    )
    modal_reference = runtime_reference.reference_certificate
    runtime_rows = runtime_reference.partition_observations
    if (
        len(runtime_rows) != 3
        or any(
            row.execution is not partition
            or row.partition_index != index
            or row.reference_certificate is not modal_reference
            for index, (row, partition) in enumerate(
                zip(runtime_rows, partitions, strict=True)
            )
        )
    ):
        raise RuntimeError(
            "the general runtime binding lost P2 partition identity"
        )
    lambda_ = modal_reference.exact_mode_eigenvalue
    exp_bounds = (
        modal_reference.exact_continuous_factor_lower_bound,
        modal_reference.exact_continuous_factor_upper_bound,
    )
    if (
        modal_reference.exact_weighted_mean
        != (initial_fields[0][0] + initial_fields[0][1]) / 2
        or modal_reference.exact_centered_mode
        != (
            (initial_fields[0][0] - initial_fields[0][1]) / 2,
            (initial_fields[0][1] - initial_fields[0][0]) / 2,
        )
        or lambda_ != 2 * first_capacity[0]
    ):
        raise RuntimeError("the general eigenmode kernel lost its exact P2 reduction")

    if mesh_overrides is None:
        meshes = tuple(
            _build_mesh(
                name,
                cycle,
                bridge,
                runtime_reference,
                index,
                partitions[index],
            )
            for index, (name, cycle, bridge) in enumerate(
                zip(_MESH_NAMES, cycles, bridges, strict=True)
            )
        )
    else:
        if type(mesh_overrides) is not tuple or len(mesh_overrides) != 3:
            raise TNFRValueError("stored P2 mesh observations are incomplete")
        expected_values = tuple(
            _derive_mesh_values(
                name,
                cycle,
                bridge,
                runtime_reference=runtime_reference,
                runtime_partition_index=index,
                runtime_reference_already_validated=True,
                bridge_already_validated=True,
                partition_override=partitions[index],
            )
            for index, (name, cycle, bridge) in enumerate(
                zip(_MESH_NAMES, cycles, bridges, strict=True)
            )
        )
        if any(
            type(mesh) is not P2EventRemeshMeshReferenceObservation
            or mesh.cycle_result is not cycle
            or mesh.runtime_bridge is not bridge
            or not proof_stamps_are_identical(
                _raw_stamp_or_none(mesh),
                _stamp_from_values(
                    _MESH_PROOF_VERSION,
                    _MESH_FIELD_NAMES,
                    _object_values(
                        mesh,
                        P2EventRemeshMeshReferenceObservation,
                    ),
                ),
            )
            or not proof_stamps_are_identical(
                _raw_stamp_or_none(mesh),
                _stamp_from_values(
                    _MESH_PROOF_VERSION,
                    _MESH_FIELD_NAMES,
                    values,
                ),
            )
            for mesh, cycle, bridge, values in zip(
                mesh_overrides,
                cycles,
                bridges,
                expected_values,
                strict=True,
            )
        ):
            raise TNFRValueError(
                "stored P2 mesh observation is forged, stale, or unbound"
            )
        meshes = mesh_overrides

    common_fields = (
        "nodes",
        "exact_initial_field",
        "exact_mean",
        "exact_initial_amplitude",
        "exact_nu_f",
        "exact_lambda",
        "exact_total_duration",
        "exact_alpha",
        "exact_beta",
        "exact_clip_lower_bound",
        "exact_clip_upper_bound",
        "exact_continuous_factor_lower_bound",
        "exact_continuous_factor_upper_bound",
    )
    if any(
        not _same(
            object.__getattribute__(mesh, name),
            object.__getattribute__(meshes[0], name),
        )
        for mesh in meshes[1:]
        for name in common_fields
    ):
        raise TNFRValueError("meshes do not realize one exact P2 reference problem")
    kernel_conditions = dict(modal_reference.conditions)
    coarse_refines = kernel_conditions["strict_proper_subdivision_chain"]
    fine_refines = coarse_refines
    factors = modal_reference.exact_euler_factors
    factor_improvements = modal_reference.exact_euler_factor_improvements
    quadratic_bounds = (
        modal_reference.exact_quadratic_factor_error_upper_bounds
    )
    quadratic_improvements = modal_reference.exact_quadratic_bound_improvements
    kernel_matches_meshes = bool(
        len(factors) == len(meshes) == 3
        and len(factor_improvements) == len(quadratic_improvements) == 2
        and tuple(mesh.exact_euler_factor for mesh in meshes) == factors
        and tuple(
            mesh.exact_quadratic_factor_error_upper_bound for mesh in meshes
        )
        == quadratic_bounds
        and all(
            mesh.exact_continuous_factor_lower_bound == exp_bounds[0]
            and mesh.exact_continuous_factor_upper_bound == exp_bounds[1]
            and mesh.exact_segment_durations
            == modal_reference.exact_partitions[index]
            and mesh.exact_scaled_segment_durations
            == modal_reference.exact_scaled_partitions[index]
            and mesh.exact_euler_segment_factors
            == modal_reference.exact_euler_segment_factors[index]
            and mesh.exact_factor_error_lower_bound
            == modal_reference.exact_factor_error_lower_bounds[index]
            and mesh.exact_factor_error_upper_bound
            == modal_reference.exact_factor_error_upper_bounds[index]
            and mesh.exact_hmax_factor_error_upper_bound
            == modal_reference.exact_hmax_factor_error_upper_bounds[index]
            for index, mesh in enumerate(meshes)
        )
    )
    if not kernel_matches_meshes:
        raise RuntimeError("P2 family fields diverged from the exact modal kernel")
    strict_factors = all(value > 0 for value in factor_improvements)
    strict_quadratic = all(value > 0 for value in quadratic_improvements)
    if not strict_factors or not strict_quadratic:
        raise RuntimeError("proper subdivision did not improve exact P2 bounds")

    first = meshes[0]
    conditions = (
        ("canonical_three_mesh_observation_bound", True),
        ("all_runtime_bridges_bound_by_identity", True),
        ("common_exact_p2_reference_problem", kernel_matches_meshes),
        (
            "coarse_to_intermediate_proper_positive_subdivision",
            coarse_refines,
        ),
        ("intermediate_to_fine_proper_positive_subdivision", fine_refines),
        ("strict_exact_euler_factor_improvement", strict_factors),
        ("strict_exact_quadratic_bound_improvement", strict_quadratic),
        (
            "common_rational_continuous_factor_enclosure",
            all(
                mesh.exact_continuous_factor_lower_bound == exp_bounds[0]
                and mesh.exact_continuous_factor_upper_bound == exp_bounds[1]
                for mesh in meshes
            ),
        ),
        ("every_mesh_reference_observation_intact", True),
    )
    if not _strict_conditions(conditions, _FAMILY_CONDITION_NAMES) or not all(
        passed for _, passed in conditions
    ):
        raise RuntimeError("P2 reference-family conditions are inconsistent")
    return {
        "refinement": refinement,
        "meshes": meshes,
        "nodes": first.nodes,
        "exact_initial_field": first.exact_initial_field,
        "exact_mean": first.exact_mean,
        "exact_initial_amplitude": first.exact_initial_amplitude,
        "exact_nu_f": first.exact_nu_f,
        "exact_lambda": first.exact_lambda,
        "exact_total_duration": first.exact_total_duration,
        "exact_alpha": first.exact_alpha,
        "exact_beta": first.exact_beta,
        "exact_clip_lower_bound": first.exact_clip_lower_bound,
        "exact_clip_upper_bound": first.exact_clip_upper_bound,
        "exact_continuous_factor_lower_bound": exp_bounds[0],
        "exact_continuous_factor_upper_bound": exp_bounds[1],
        "exact_euler_factors": factors,
        "exact_euler_factor_improvements": factor_improvements,
        "exact_quadratic_factor_error_upper_bounds": quadratic_bounds,
        "exact_quadratic_bound_improvements": quadratic_improvements,
        "strict_pre_remesh_error_improvement": strict_factors,
        "strict_ideal_post_remesh_error_improvement": bool(
            strict_factors and first.exact_beta > 0
        ),
        "conditions": conditions,
    }


def _seal_family(
    value: P2EventRemeshReferenceFamilyObservation,
) -> P2EventRemeshReferenceFamilyObservation:
    values = _object_values(value, P2EventRemeshReferenceFamilyObservation)
    return replace(
        value,
        _proof_stamp=_stamp_from_values(
            _FAMILY_PROOF_VERSION,
            _FAMILY_FIELD_NAMES,
            values,
        ),
    )


def observe_p2_event_remesh_reference_family(
    coarse: EventRemeshCycleResult,
    intermediate: EventRemeshCycleResult,
    fine: EventRemeshCycleResult,
) -> P2EventRemeshReferenceFamilyObservation:
    """Certify the scoped exact P2 three-mesh event/REMESH reference family."""

    cycles = (coarse, intermediate, fine)
    values = _derive_family_values(cycles)
    result = P2EventRemeshReferenceFamilyObservation(
        **values,
        _proof_stamp=_stamp_from_values(
            _FAMILY_PROOF_VERSION,
            _FAMILY_FIELD_NAMES,
            values,
        ),
    )
    current = _object_values(result, P2EventRemeshReferenceFamilyObservation)
    if not proof_stamps_are_identical(
        object.__getattribute__(result, "_proof_stamp"),
        _stamp_from_values(
            _FAMILY_PROOF_VERSION,
            _FAMILY_FIELD_NAMES,
            current,
        ),
    ):
        raise RuntimeError("constructed P2 reference family is inconsistent")
    return result
