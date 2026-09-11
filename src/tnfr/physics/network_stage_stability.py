r"""Scoped EPI stability certificate for repeated all-target EN/RA stages.

The SDK and supported GPU path execute Reception (EN) and Resonance (RA) as
atomic two-phase Jacobi stages: every target reads one immutable snapshot and
all accepted proposals commit together.  The local realization certificates
identify one row of the corresponding ideal-real and represented-binary64
maps.  This module assembles those rows into the full all-target map and
iterates the same proposal formulas on a detached logical graph.

Three claims remain deliberately separate:

* a finite, read-only observation of the repeated all-target runtime formula;
* an exact ideal-real fixed-map result when the relevant neighbour sets stay
  fixed and hard clipping is inactive by convex forward invariance;
* an affine disagreement-gain theorem for the represented coefficient map.

For ``alpha in [0, 1]`` every ideal-real row is a convex combination.  Hence an
initial EPI field in ``[EPI_MIN, EPI_MAX]`` remains in that interval under any
number of such stages, even if RA's U3-filtered neighbour sets change.  This
proves that *ideal-real hard clipping* is inactive.  It does not make the
two-stage binary64 mean/blend kernel globally affine: its evaluation order can
differ from a matrix product.  Exact snapshot and finite-horizon residuals are
therefore observations, never a global runtime-affinity theorem.

RA adds two further boundaries.  A fixed-map repetition requires an explicit
declaration that its U3 neighbour sets remain fixed, and repeated identity
admission is promoted only when the initial field has one nonzero sign class
and one EPI-kind class.  Capacity amplification can also change the diffusion
metric ``d_i/nu_i``; every observed pre/post metric is retained and exact
proportionality is reported separately.

Scope
-----
The certificate requires a fixed-size, connected, undirected positive-
conductance graph accepted by the heterogeneous pure-EPI diffusion theorem,
explicit scalar or uniform-real BEPI values, positive capacities, and no
isolates.  The finite trace reproduces structural proposals only; it does not
execute grammar/history bookkeeping, telemetry, monitors, or pressure refresh.
Soft or active clipping, topology changes, changing RA neighbour sets, mixed
operator words, and a global binary64 runtime-affinity proof remain outside the
fixed-map theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real
from operator import index as integer_index
from typing import Any

from ..alias import set_attr, set_attr_generic, set_attr_str
from ..constants.aliases import (
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..mathematics.unified_numerical import np
from ..operators._resonance_identity import resonance_identity_failures
from ..operators.operator_contracts import contract_for
from ..types import Glyph, serialize_bepi
from ._neighbor_epi_realization import (
    exact_binary64_matrix,
    exact_ideal_neighbor_blend_map,
    exact_matrix_vector,
    readonly_float_array,
    represented_neighbor_blend_map,
    validate_certificate_tolerance,
)
from ._exact_metric import (
    binary64_vectors_exactly_proportional as _exactly_proportional,
)
from .hybrid_operator_stability import (
    AffineEPIJumpGainCertificate,
    HybridEPIStabilityCertificate,
    certify_affine_epi_jump_gain,
    compose_hybrid_epi_stability,
)
from .reception_realization import (
    ReceptionEPIRealizationCertificate,
    certify_reception_epi_realization,
)
from .resonance_realization import (
    ResonanceEPIRealizationCertificate,
    certify_resonance_epi_realization,
)
from .structural_diffusion import (
    HeterogeneousDiffusionStabilityCertificate,
    verify_heterogeneous_diffusion_stability,
)

__all__ = [
    "AllTargetNeighborStageCertificate",
    "AllTargetNeighborStageStep",
    "NeighborStageDiffusionBridgeCertificate",
    "certify_all_target_neighbor_stage",
    "certify_reception_all_target_stage",
    "certify_resonance_all_target_stage",
    "compose_neighbor_stage_diffusion_stability",
]


_SCOPE = (
    "CONDITIONAL all-target EN/RA EPI-stage certificate. The finite trace "
    "replays the local shared realization formulas on detached graph states; "
    "it does not execute grammar, histories, telemetry, monitors, or pressure "
    "refresh. Ideal-real convex hard-clip invariance, represented affine-map "
    "gain, observed binary64 agreement, consensus drift, and diffusion-metric "
    "changes are separate results. Fixed-map RA repetition additionally "
    "requires declared fixed U3 neighbour sets and a proved sign/kind identity "
    "domain. No global binary64 runtime affinity, soft-clipping theorem, mixed "
    "word theorem, changing-support theorem, or future RA metric theorem is "
    "claimed."
)


_BRIDGE_SCOPE = (
    "CONDITIONAL one-stage EN/RA represented-affine-map followed by a strictly "
    "positive-duration fixed post-stage pure-EPI diffusion flow. The bridge "
    "revalidates node order, represented coefficients, exact row algebra, "
    "nested proof stamps and the post-stage metric before invoking the generic "
    "hybrid composer. Finite-horizon disagreement, strict contraction, "
    "post-metric weighted-mean drift and pre/post diffusion-metric change are "
    "separate results. The accepted runtime proposal is retained only as a "
    "snapshot observation: no global binary64 runtime affinity, stored-pressure "
    "refresh, repeated schedule, changing support, changing U3 gate, mixed word "
    "or inter-event trajectory is certified."
)


LocalCertificate = (
    ReceptionEPIRealizationCertificate | ResonanceEPIRealizationCertificate
)


def _positive_repetition_count(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("repetitions must be a positive integer")
    try:
        count = integer_index(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError("repetitions must be a positive integer") from exc
    if count < 1:
        raise ValueError("repetitions must be a positive integer")
    return count


def _operator_glyph(value: Any) -> Glyph:
    candidate = getattr(value, "glyph", value)
    if isinstance(candidate, Glyph):
        glyph = candidate
    else:
        try:
            contract = contract_for(str(candidate))
        except KeyError as exc:
            raise ValueError(
                "operator must identify Reception/EN or Resonance/RA"
            ) from exc
        glyph = Glyph(contract.glyph)
    if glyph not in (Glyph.EN, Glyph.RA):
        raise ValueError(
            "all-target neighbour-stage certification supports EN and RA only"
        )
    return glyph


def _exact_matrix_power(
    matrix: tuple[tuple[Fraction, ...], ...], exponent: int
) -> tuple[tuple[Fraction, ...], ...]:
    dimension = len(matrix)
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )

    def multiply(
        left: tuple[tuple[Fraction, ...], ...],
        right: tuple[tuple[Fraction, ...], ...],
    ) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(
                sum(
                    (
                        left[row][inner] * right[inner][column]
                        for inner in range(dimension)
                    ),
                    Fraction(0),
                )
                for column in range(dimension)
            )
            for row in range(dimension)
        )

    result = identity
    factor = matrix
    power = exponent
    while power:
        if power & 1:
            result = multiply(result, factor)
        power >>= 1
        if power:
            factor = multiply(factor, factor)
    return result


def _fraction_upper_float(value: Fraction) -> float:
    if value < 0:
        raise ValueError("an energy-gain bound must be nonnegative")
    try:
        result = float(value)
    except OverflowError:
        return float("inf")
    if math.isinf(result):
        return result
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, float("inf"))
    return result


def _exact_weighted_mean(
    weights: Any, state: Any
) -> Fraction:
    exact_weights = tuple(Fraction.from_float(float(value)) for value in weights)
    exact_state = tuple(Fraction.from_float(float(value)) for value in state)
    return sum(
        (weight * value for weight, value in zip(exact_weights, exact_state)),
        Fraction(0),
    ) / sum(exact_weights, Fraction(0))


def _composition_exact_gain(
    certificate: AffineEPIJumpGainCertificate,
) -> Fraction:
    """Read the precise quotient gain for repeated affine-stage algebra.

    Repeated-stage powers retain this rational factor exactly.  Hybrid
    log-space decisions separately use the upward dyadic factors exposed by
    ``HybridEPIStabilityCertificate.exact_log_composition_gain_factors``.
    """

    return certificate.exact_quotient_energy_gain_upper_bound


def _logical_copy(graph: Any) -> Any:
    """Copy graph mappings while keeping the audited input untouched."""

    try:
        return graph.copy(as_view=False)
    except TypeError:
        return graph.copy()


def _set_epi(mapping: dict[str, Any], value: float) -> None:
    set_attr_generic(mapping, ALIAS_EPI, serialize_bepi(value), conv=lambda item: item)


@dataclass(frozen=True, slots=True)
class AllTargetNeighborStageStep:
    """One immutable all-target proposal assembled from local certificates."""

    index: int
    nodes: tuple[Any, ...]
    runtime_neighbor_sets: tuple[tuple[Any, ...], ...]
    local_certificates: tuple[LocalCertificate, ...]
    state_before: Any
    runtime_unclipped_state_after: Any
    runtime_proposed_state_after: Any
    runtime_accepted_state_after: Any
    runtime_stage_admissible: bool
    atomic_rejection_nodes: tuple[Any, ...]
    clip_intervention_nodes: tuple[Any, ...]
    clip_mode: str
    epi_lower_bound: float
    epi_upper_bound: float
    mix_factor: float
    exact_mix_factor: Fraction
    ideal_real_stage_map: tuple[tuple[Fraction, ...], ...]
    represented_stage_map: Any
    exact_represented_stage_map: tuple[tuple[Fraction, ...], ...]
    exact_represented_row_sums: tuple[Fraction, ...]
    represented_map_exact_consensus_subspace_preservation: bool
    exact_runtime_proposal_minus_represented_stage: tuple[Fraction, ...]
    runtime_proposal_matches_represented_stage_exactly: bool
    runtime_proposal_matches_represented_stage_within_tolerance: bool
    ideal_real_hard_clipping_inactive_by_convexity: bool
    pre_diffusion_certificate: HeterogeneousDiffusionStabilityCertificate
    post_diffusion_certificate: HeterogeneousDiffusionStabilityCertificate
    pre_post_metric_exactly_proportional: bool
    pre_metric_affine_jump_certificate: AffineEPIJumpGainCertificate
    post_metric_affine_jump_certificate: AffineEPIJumpGainCertificate
    exact_pre_metric_runtime_weighted_mean_shift: Fraction
    exact_post_metric_runtime_weighted_mean_shift: Fraction


@dataclass(frozen=True, slots=True)
class AllTargetNeighborStageCertificate:
    """Finite repeated trace plus scoped fixed-map EN/RA conclusions."""

    operator_name: str
    glyph: str
    nodes: tuple[Any, ...]
    repetitions_requested: int
    repetitions_observed: int
    repetitions_completed: int
    fixed_support_declared: bool
    fixed_phase_neighbor_sets_declared: bool
    steps: tuple[AllTargetNeighborStageStep, ...]
    all_stages_admissible: bool
    any_runtime_clip_intervention: bool
    ideal_real_hard_clipping_inactive_for_arbitrary_repetitions: bool
    observed_runtime_neighbor_sets_constant: bool
    observed_ideal_real_maps_constant: bool
    observed_represented_maps_constant: bool
    observed_diffusion_metric_exactly_common: bool
    ra_global_sign_identity_forward_invariant: bool | None
    ra_global_kind_identity_forward_invariant: bool | None
    ideal_real_fixed_map_repetition_conditions: tuple[tuple[str, bool], ...]
    ideal_real_fixed_map_repetition_certified: bool
    represented_fixed_map_repetition_conditions: tuple[tuple[str, bool], ...]
    represented_fixed_map_repetition_certified: bool
    exact_ideal_real_repeated_map: tuple[tuple[Fraction, ...], ...] | None
    exact_represented_repeated_map: tuple[tuple[Fraction, ...], ...] | None
    exact_represented_repeated_energy_gain_bound: Fraction | None
    represented_repeated_energy_gain_bound: float | None
    represented_finite_repetition_disagreement_contraction_certified: bool | None
    represented_asymptotic_disagreement_convergence_certified: bool | None
    exact_observed_runtime_minus_represented_repeated_state: (
        tuple[Fraction, ...] | None
    )
    observed_runtime_matches_represented_repeated_state_exactly: bool | None
    global_binary64_runtime_repetition_certified: bool
    tolerance: float
    scope: str

    @property
    def failed_ideal_real_fixed_map_conditions(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, passed in self.ideal_real_fixed_map_repetition_conditions
            if not passed
        )

    @property
    def failed_represented_fixed_map_conditions(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, passed in self.represented_fixed_map_repetition_conditions
            if not passed
        )


@dataclass(frozen=True, slots=True)
class NeighborStageDiffusionBridgeCertificate:
    """One represented all-target EN/RA stage followed by fixed diffusion."""

    operator_name: str
    glyph: str
    nodes: tuple[Any, ...]
    flow_duration: float
    stage_certificate: AllTargetNeighborStageCertificate
    stage_step: AllTargetNeighborStageStep
    post_diffusion_certificate: HeterogeneousDiffusionStabilityCertificate
    post_metric_affine_jump_certificate: AffineEPIJumpGainCertificate
    hybrid_certificate: HybridEPIStabilityCertificate | None
    bridge_conditions: tuple[tuple[str, bool], ...]
    represented_model_bridge_certified: bool
    finite_horizon_disagreement_bound_certified: bool
    disagreement_contracts_over_declared_horizon: bool | None
    represented_stage_preserves_post_metric_weighted_mean_exactly: bool
    exact_runtime_post_metric_weighted_mean_shift: Fraction
    runtime_post_metric_weighted_mean_preserved_exactly: bool
    hybrid_preserves_initial_weighted_mean: bool | None
    pre_post_metric_values_identical: bool
    pre_post_metric_exactly_proportional: bool
    runtime_proposal_matches_represented_stage_exactly: bool
    runtime_proposal_matches_represented_stage_within_tolerance: bool
    runtime_pure_epi_pressure_defect: Any
    runtime_pure_epi_pressure_defect_norm: float
    runtime_pure_epi_pressure_defect_detected: bool
    pressure_refresh_required_before_runtime_flow: bool
    stored_pressure_refresh_certified: bool
    raw_epi_embedding_revalidated_without_graph: bool
    global_binary64_runtime_affinity_certified: bool
    repeated_schedule_disagreement_convergence_certified: bool | None
    tolerance: float
    scope: str

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Return unmet hypotheses of the represented one-stage bridge."""

        return tuple(
            name for name, passed in self.bridge_conditions if not passed
        )


@dataclass(frozen=True, slots=True)
class _ValidatedNeighborStage:
    step: AllTargetNeighborStageStep
    all_local_snapshots_in_affine_model_domain: bool
    represented_consensus_subspace_preserved: bool
    runtime_exact_match: bool
    runtime_match_within_tolerance: bool
    pre_post_metric_values_identical: bool
    pre_post_metric_exactly_proportional: bool
    exact_runtime_post_metric_mean_shift: Fraction
    runtime_pure_epi_pressure_defect: Any
    runtime_pure_epi_pressure_defect_norm: float
    runtime_pure_epi_pressure_defect_detected: bool
    pressure_refresh_required_before_runtime_flow: bool


def _strictly_positive_flow_duration(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("flow_duration must be a finite strictly positive real")
    try:
        source_positive = bool(value > 0)
        source_nonzero = bool(value != 0)
        duration = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "flow_duration must be a finite strictly positive real"
        ) from exc
    if not source_positive:
        raise ValueError("flow_duration must be strictly positive")
    if not math.isfinite(duration):
        raise ValueError("flow_duration must be finite")
    if source_nonzero and duration == 0.0:
        raise ValueError("flow_duration is below nonzero floating-point range")
    return duration


def _bridge_array(value: Any, name: str, shape: tuple[int, ...]) -> Any:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite binary64 array") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape!r}")
    return array


def _exact_float_vector(value: Any, name: str, size: int) -> tuple[Fraction, ...]:
    array = _bridge_array(value, name, (size,))
    return tuple(Fraction.from_float(float(entry)) for entry in array)


def _condition_values(
    conditions: Any,
    expected_names: tuple[str, ...],
    name: str,
) -> dict[str, bool]:
    if not isinstance(conditions, tuple) or len(conditions) != len(expected_names):
        raise ValueError(f"{name} has an inconsistent condition schema")
    result: dict[str, bool] = {}
    for expected_name, item in zip(expected_names, conditions, strict=True):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or item[0] != expected_name
            or not isinstance(item[1], bool)
        ):
            raise ValueError(f"{name} has an inconsistent condition schema")
        result[expected_name] = item[1]
    return result


def _validate_bridge_jump(
    jump: Any,
    *,
    name: str,
    operator_name: str,
    glyph: Glyph,
    nodes: tuple[Any, ...],
    represented_map: Any,
    metric: Any,
) -> AffineEPIJumpGainCertificate:
    if not isinstance(jump, AffineEPIJumpGainCertificate):
        raise TypeError(f"{name} must be an affine EPI jump certificate")
    if not jump._proof_fields_are_intact():
        raise ValueError(f"{name} proof fields were replaced or mutated")
    if (
        jump.operator_name != operator_name
        or jump.glyph != glyph.value
        or tuple(jump.nodes) != nodes
    ):
        raise ValueError(f"{name} operator identity or node order is inconsistent")
    dimension = len(nodes)
    jump_map = _bridge_array(
        jump.linear_map, f"{name}.linear_map", (dimension, dimension)
    )
    if exact_binary64_matrix(jump_map) != exact_binary64_matrix(represented_map):
        raise ValueError(f"{name} linear map does not match the stage map")
    exact_offset = _exact_float_vector(jump.offset, f"{name}.offset", dimension)
    if any(value != 0 for value in exact_offset):
        raise ValueError(f"{name} offset does not match the linear stage")
    exact_jump_metric = _exact_float_vector(
        jump.metric_weights, f"{name}.metric_weights", dimension
    )
    exact_flow_metric = _exact_float_vector(metric, "flow.metric_weights", dimension)
    if exact_jump_metric != exact_flow_metric:
        raise ValueError(f"{name} metric does not match its flow certificate")
    return jump


def _validate_bridge_stage_certificate(
    certificate: Any,
) -> _ValidatedNeighborStage:
    if not isinstance(certificate, AllTargetNeighborStageCertificate):
        raise TypeError(
            "stage_certificate must be an all-target EN/RA stage certificate"
        )
    certificate_tolerance = validate_certificate_tolerance(certificate.tolerance)
    glyph = _operator_glyph(certificate.glyph)
    operator_name = "Reception" if glyph is Glyph.EN else "Resonance"
    if certificate.operator_name != operator_name:
        raise ValueError("stage certificate operator name and glyph disagree")
    if (
        certificate.repetitions_requested != 1
        or certificate.repetitions_observed != 1
        or len(certificate.steps) != 1
    ):
        raise ValueError("the diffusion bridge requires exactly one certified stage")

    step = certificate.steps[0]
    nodes = tuple(certificate.nodes)
    dimension = len(nodes)
    if dimension < 2 or tuple(step.nodes) != nodes or step.index != 0:
        raise ValueError("stage certificate node order or step index is inconsistent")
    if len(step.local_certificates) != dimension:
        raise ValueError("stage certificate must contain one local row per node")

    state_before = _bridge_array(
        step.state_before, "stage_step.state_before", (dimension,)
    )
    unclipped = _bridge_array(
        step.runtime_unclipped_state_after,
        "stage_step.runtime_unclipped_state_after",
        (dimension,),
    )
    proposed = _bridge_array(
        step.runtime_proposed_state_after,
        "stage_step.runtime_proposed_state_after",
        (dimension,),
    )
    accepted = _bridge_array(
        step.runtime_accepted_state_after,
        "stage_step.runtime_accepted_state_after",
        (dimension,),
    )
    represented_map = _bridge_array(
        step.represented_stage_map,
        "stage_step.represented_stage_map",
        (dimension, dimension),
    )
    exact_map = exact_binary64_matrix(represented_map)
    if exact_map != step.exact_represented_stage_map:
        raise ValueError("stage exact represented map is inconsistent")
    exact_row_sums = tuple(sum(row, Fraction(0)) for row in exact_map)
    if exact_row_sums != step.exact_represented_row_sums:
        raise ValueError("stage exact represented row sums are inconsistent")
    consensus_preserved = all(value == 1 for value in exact_row_sums)
    if (
        step.represented_map_exact_consensus_subspace_preservation
        != consensus_preserved
    ):
        raise ValueError("stage consensus-subspace result is inconsistent")

    if glyph is Glyph.EN:
        local_type = ReceptionEPIRealizationCertificate
    else:
        local_type = ResonanceEPIRealizationCertificate
    node_indices = {node: index for index, node in enumerate(nodes)}
    exact_state = tuple(Fraction.from_float(float(value)) for value in state_before)
    local_affine_domain: list[bool] = []
    local_pressure_defects: list[Any] = []
    reference_pressure: Any = None
    ra_non_epi_structural_change = False
    rejection_nodes: list[Any] = []
    for target_index, (node, local) in enumerate(
        zip(nodes, step.local_certificates, strict=True)
    ):
        if not isinstance(local, local_type):
            raise TypeError("stage local certificate type disagrees with its glyph")
        if (
            tuple(local.nodes) != nodes
            or local.target != node
            or local.target_index != target_index
        ):
            raise ValueError("stage local certificate node order is inconsistent")
        if local.fixed_support_declared != certificate.fixed_support_declared:
            raise ValueError("stage fixed-support declarations are inconsistent")
        local_state = _bridge_array(
            local.state_before,
            f"local_certificates[{target_index}].state_before",
            (dimension,),
        )
        if not np.array_equal(local_state, state_before):
            raise ValueError("stage local certificate snapshot is inconsistent")
        local_map = _bridge_array(
            local.represented_linear_map,
            f"local_certificates[{target_index}].represented_linear_map",
            (dimension, dimension),
        )
        runtime_neighbors = tuple(local.runtime_neighbors)
        if tuple(step.runtime_neighbor_sets[target_index]) != runtime_neighbors:
            raise ValueError("stage runtime-neighbour sets are inconsistent")
        try:
            expected_neighbor_indices = tuple(
                node_indices[neighbor] for neighbor in runtime_neighbors
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "stage runtime neighbour is absent from node order"
            ) from exc
        if (
            not expected_neighbor_indices
            or tuple(local.runtime_neighbor_indices) != expected_neighbor_indices
        ):
            raise ValueError("stage runtime-neighbour indices are inconsistent")

        try:
            local_mix = float(local.mix_factor)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("stage local mix factor is invalid") from exc
        if not math.isfinite(local_mix):
            raise ValueError("stage local mix factor is invalid")
        local_exact_mix = Fraction.from_float(local_mix)
        if (
            local_exact_mix != local.exact_mix_factor
            or local_exact_mix != step.exact_mix_factor
            or local_mix != step.mix_factor
        ):
            raise ValueError("stage local mix factors are inconsistent")
        expected_ideal_map = exact_ideal_neighbor_blend_map(
            dimension,
            target_index,
            expected_neighbor_indices,
            local_exact_mix,
        )
        expected_local_map = represented_neighbor_blend_map(
            dimension,
            target_index,
            expected_neighbor_indices,
            local_mix,
        )
        if local.ideal_real_linear_map != expected_ideal_map:
            raise ValueError("stage local ideal-real map is inconsistent")
        if exact_binary64_matrix(local_map) != exact_binary64_matrix(
            expected_local_map
        ):
            raise ValueError("stage local represented map is inconsistent")
        if exact_binary64_matrix(local_map)[target_index] != exact_map[target_index]:
            raise ValueError("stage local represented row is inconsistent")
        local_unclipped = float(local.runtime_unclipped_target_value)
        local_proposed = float(
            local.runtime_target_value
            if glyph is Glyph.EN
            else local.runtime_proposed_target_value
        )
        if (
            not math.isfinite(local_unclipped)
            or not math.isfinite(local_proposed)
            or local_unclipped != float(unclipped[target_index])
            or local_proposed != float(proposed[target_index])
        ):
            raise ValueError("stage local proposal is inconsistent")

        if (
            local.clip_mode != step.clip_mode
            or float(local.epi_lower_bound) != step.epi_lower_bound
            or float(local.epi_upper_bound) != step.epi_upper_bound
        ):
            raise ValueError("stage local clipping policy is inconsistent")
        hard_clip = local.clip_mode == "hard"
        convex_mix = 0.0 <= local_mix <= 1.0
        state_in_bounds = bool(
            np.all(local_state >= float(local.epi_lower_bound))
            and np.all(local_state <= float(local.epi_upper_bound))
        )
        clipping_inactive = local_unclipped == local_proposed
        expected_ideal_conditions = (
            ("fixed_support_declared", certificate.fixed_support_declared),
            ("hard_clip_mode", hard_clip),
            ("convex_mix_factor", convex_mix),
            ("state_inside_epi_bounds", state_in_bounds),
        )
        ideal_names = tuple(name for name, _ in expected_ideal_conditions)
        declared_ideal = _condition_values(
            local.ideal_real_affine_conditions,
            ideal_names,
            f"local_certificates[{target_index}].ideal_real_affine_conditions",
        )
        if declared_ideal != dict(expected_ideal_conditions):
            raise ValueError("stage local ideal-real conditions are inconsistent")
        ideal_regime = all(passed for _, passed in expected_ideal_conditions)
        if (
            local.ideal_real_affine_regime != ideal_regime
            or local.ideal_real_hard_clipping_inactive_by_convexity
            != (hard_clip and convex_mix and state_in_bounds)
            or local.runtime_hard_clipping_inactive_at_snapshot
            != clipping_inactive
        ):
            raise ValueError("stage local affine-regime summary is inconsistent")

        exact_local_map = exact_binary64_matrix(local_map)
        local_row_sum = sum(exact_local_map[target_index], Fraction(0))
        local_consensus = local_row_sum == 1
        if (
            local.exact_represented_target_row_sum != local_row_sum
            or local.represented_map_exact_consensus_subspace_preservation
            != local_consensus
            or not local.ideal_real_consensus_subspace_preservation
        ):
            raise ValueError("stage local consensus result is inconsistent")

        local_tolerance = validate_certificate_tolerance(local.tolerance)
        exact_local_ideal_after = exact_matrix_vector(expected_ideal_map, exact_state)
        exact_local_represented_after = exact_matrix_vector(
            exact_local_map, exact_state
        )
        exact_local_proposed = Fraction.from_float(local_proposed)
        local_ideal_residual = (
            exact_local_proposed - exact_local_ideal_after[target_index]
        )
        local_represented_residual = (
            exact_local_proposed - exact_local_represented_after[target_index]
        )
        represented_residual_float = abs(float(local_represented_residual))
        represented_scale = max(
            1.0,
            abs(local_proposed),
            abs(float(exact_local_represented_after[target_index])),
        )
        local_represented_within = bool(
            math.isfinite(represented_residual_float)
            and math.isfinite(represented_scale)
            and represented_residual_float <= local_tolerance * represented_scale
        )
        if (
            local.exact_runtime_minus_ideal_target != local_ideal_residual
            or local.exact_runtime_minus_represented_affine_target
            != local_represented_residual
            or local.runtime_matches_ideal_target_exactly_at_snapshot
            != (local_ideal_residual == 0)
            or local.runtime_matches_represented_affine_exactly_at_snapshot
            != (local_represented_residual == 0)
            or local.runtime_matches_represented_affine_within_tolerance
            != local_represented_within
        ):
            raise ValueError("stage local runtime/affine result is inconsistent")

        runtime_condition_names: tuple[str, ...]
        if glyph is Glyph.EN:
            runtime_condition_names = ideal_names + (
                "scalar_epi_embedding",
                "runtime_hard_clipping_inactive_at_snapshot",
            )
        else:
            runtime_condition_names = ideal_names + (
                "u3_phase_gate",
                "scalar_epi_embedding",
                "runtime_hard_clipping_inactive_at_snapshot",
                "identity_gate_passed",
                "frequency_not_decreased",
                "post_diffusion_certificate_available",
            )
        declared_runtime = _condition_values(
            local.runtime_affine_model_domain_conditions,
            runtime_condition_names,
            f"local_certificates[{target_index}].runtime_affine_conditions",
        )
        scalar_embedding_observed = declared_runtime["scalar_epi_embedding"]

        if glyph is Glyph.RA:
            graph_neighbors = tuple(local.graph_neighbors)
            separations = _bridge_array(
                local.phase_separations,
                f"local_certificates[{target_index}].phase_separations",
                (len(graph_neighbors),),
            )
            phase_limit = float(local.phase_gate_limit)
            if not math.isfinite(phase_limit) or not 0.0 <= phase_limit <= math.pi / 2:
                raise ValueError("RA local phase gate is invalid")
            compatible_neighbors = tuple(
                neighbor
                for neighbor, separation in zip(
                    graph_neighbors, separations, strict=True
                )
                if float(separation) <= phase_limit
            )
            incompatible_neighbors = tuple(
                neighbor
                for neighbor, separation in zip(
                    graph_neighbors, separations, strict=True
                )
                if float(separation) > phase_limit
            )
            u3_gate = bool(compatible_neighbors)
            if (
                tuple(local.phase_compatible_neighbors) != compatible_neighbors
                or runtime_neighbors != compatible_neighbors
                or tuple(local.phase_incompatible_neighbors) != incompatible_neighbors
                or local.phase_gate_passed != u3_gate
            ):
                raise ValueError("RA local U3 phase-gate result is inconsistent")

            identity_failures = resonance_identity_failures(
                float(local_state[target_index]),
                local_proposed,
                str(local.epi_kind_before),
                str(local.proposed_epi_kind),
            )
            identity_passed = not identity_failures
            if (
                tuple(local.identity_gate_failures) != identity_failures
                or local.identity_gate_passed != identity_passed
                or local.runtime_operation_admissible != identity_passed
            ):
                raise ValueError("RA local admission fields are inconsistent")
            frequencies_before = _bridge_array(
                local.frequency_before,
                f"local_certificates[{target_index}].frequency_before",
                (dimension,),
            )
            proposed_frequencies = _bridge_array(
                local.proposed_frequency_after,
                f"local_certificates[{target_index}].proposed_frequency_after",
                (dimension,),
            )
            accepted_frequencies = _bridge_array(
                local.frequency_after,
                f"local_certificates[{target_index}].frequency_after",
                (dimension,),
            )
            frequency_nondecreasing = bool(
                np.all(proposed_frequencies >= frequencies_before)
            )
            expected_frequencies = (
                proposed_frequencies if identity_passed else frequencies_before
            )
            if (
                not np.array_equal(accepted_frequencies, expected_frequencies)
                or local.frequency_nondecreasing != frequency_nondecreasing
            ):
                raise ValueError("RA local frequency result is inconsistent")
            expected_identity_target = (
                local_proposed
                if identity_passed
                else float(local_state[target_index])
            )
            if float(local.runtime_target_value) != expected_identity_target:
                raise ValueError("RA local identity-gated EPI result is inconsistent")
            expected_phase = (
                float(local.proposed_phase_after)
                if identity_passed
                else float(local.phase_before)
            )
            if float(local.phase_after) != expected_phase:
                raise ValueError("RA local identity-gated phase result is inconsistent")
            ra_non_epi_structural_change = bool(
                ra_non_epi_structural_change
                or (
                    identity_passed
                    and (
                        float(local.phase_after) != float(local.phase_before)
                        or float(accepted_frequencies[target_index])
                        != float(frequencies_before[target_index])
                    )
                )
            )
            expected_runtime_conditions = expected_ideal_conditions + (
                ("u3_phase_gate", u3_gate),
                ("scalar_epi_embedding", scalar_embedding_observed),
                (
                    "runtime_hard_clipping_inactive_at_snapshot",
                    clipping_inactive,
                ),
                ("identity_gate_passed", identity_passed),
                ("frequency_not_decreased", frequency_nondecreasing),
                (
                    "post_diffusion_certificate_available",
                    local.post_diffusion_certificate is not None,
                ),
            )
            if not identity_passed:
                rejection_nodes.append(node)
            local_flow = local.post_diffusion_certificate
        else:
            expected_runtime_conditions = expected_ideal_conditions + (
                ("scalar_epi_embedding", scalar_embedding_observed),
                (
                    "runtime_hard_clipping_inactive_at_snapshot",
                    clipping_inactive,
                ),
            )
            local_flow = local.diffusion_certificate

        if declared_runtime != dict(expected_runtime_conditions):
            raise ValueError("stage local runtime-domain conditions are inconsistent")
        runtime_domain = all(passed for _, passed in expected_runtime_conditions)
        if local.runtime_snapshot_in_affine_model_domain != runtime_domain:
            raise ValueError("stage local runtime-domain summary is inconsistent")

        nested_jump = local.affine_jump_certificate
        expected_nested_jump = bool(runtime_domain and local_consensus)
        if (
            local.represented_affine_jump_eligible != expected_nested_jump
            or (nested_jump is not None) != expected_nested_jump
        ):
            raise ValueError("stage local affine-jump eligibility is inconsistent")
        if nested_jump is not None:
            if local_flow is None:
                raise ValueError("stage local affine jump lacks its flow metric")
            if (
                not isinstance(local_flow, HeterogeneousDiffusionStabilityCertificate)
                or not local_flow._proof_fields_are_intact()
                or tuple(local_flow.nodes) != nodes
            ):
                raise ValueError("stage local flow certificate is inconsistent")
            _validate_bridge_jump(
                nested_jump,
                name=f"local_certificates[{target_index}].affine_jump_certificate",
                operator_name=operator_name,
                glyph=glyph,
                nodes=nodes,
                represented_map=local_map,
                metric=local_flow.metric_weights,
            )
        local_affine_domain.append(runtime_domain)

        current_pressure = _bridge_array(
            local.current_pure_epi_pressure,
            f"local_certificates[{target_index}].current_pure_epi_pressure",
            (dimension,),
        )
        post_pressure = _bridge_array(
            local.post_reset_pure_epi_pressure,
            f"local_certificates[{target_index}].post_reset_pure_epi_pressure",
            (dimension,),
        )
        local_pressure_defect = _bridge_array(
            local.post_reset_pressure_manifold_defect,
            f"local_certificates[{target_index}].pressure_defect",
            (dimension,),
        )
        if not np.array_equal(local_pressure_defect, current_pressure - post_pressure):
            raise ValueError("stage local pure-EPI pressure defect is inconsistent")
        local_defect_norm = float(
            np.max(np.abs(local_pressure_defect), initial=0.0)
        )
        if (
            local.post_reset_pressure_manifold_defect_norm != local_defect_norm
            or local.pressure_refresh_detected_at_snapshot
            != bool(np.any(local_pressure_defect != 0.0))
        ):
            raise ValueError("stage local pressure-refresh diagnostic is inconsistent")
        if reference_pressure is None:
            reference_pressure = current_pressure
        elif not np.array_equal(reference_pressure, current_pressure):
            raise ValueError("stage local pure-EPI pressure snapshots are inconsistent")
        local_pressure_defects.append(local_pressure_defect)

    expected_rejections = tuple(rejection_nodes)
    if tuple(step.atomic_rejection_nodes) != expected_rejections:
        raise ValueError("stage atomic rejection set is inconsistent")
    admissible = not expected_rejections
    if step.runtime_stage_admissible != admissible:
        raise ValueError("stage admission result is inconsistent")
    expected_accepted = proposed if admissible else state_before
    if not np.array_equal(accepted, expected_accepted):
        raise ValueError("stage accepted state is inconsistent")
    if admissible:
        # Pure-EPI pressure is linear in EPI.  Each local certificate changes
        # one coordinate from the common snapshot, so summing its detached
        # pressure defects yields the simultaneous all-target defect.  This is
        # a binary64 snapshot diagnostic, not the local exact iff theorem.
        pressure_defect = np.sum(
            np.vstack(local_pressure_defects), axis=0, dtype=float
        )
    else:
        pressure_defect = np.zeros(dimension, dtype=float)
    if not np.all(np.isfinite(pressure_defect)):
        raise ValueError("all-target pure-EPI pressure defect exceeds binary64 range")
    pressure_defect_norm = float(
        np.max(np.abs(pressure_defect), initial=0.0)
    )
    pressure_defect_detected = bool(np.any(pressure_defect != 0.0))
    epi_state_changed = not np.array_equal(accepted, state_before)
    pressure_refresh_required = bool(
        admissible and (epi_state_changed or ra_non_epi_structural_change)
    )
    expected_completed = int(admissible)
    if (
        certificate.repetitions_completed != expected_completed
        or certificate.all_stages_admissible != admissible
    ):
        raise ValueError("outer stage admission summary is inconsistent")

    clip_nodes = tuple(
        node
        for node, raw_value, proposed_value in zip(
            nodes, unclipped, proposed, strict=True
        )
        if float(raw_value) != float(proposed_value)
    )
    if tuple(step.clip_intervention_nodes) != clip_nodes:
        raise ValueError("stage clipping intervention set is inconsistent")
    if certificate.any_runtime_clip_intervention != bool(clip_nodes):
        raise ValueError("outer stage clipping summary is inconsistent")

    exact_proposed = tuple(Fraction.from_float(float(value)) for value in proposed)
    exact_model_after = exact_matrix_vector(exact_map, exact_state)
    exact_residual = tuple(
        actual - represented
        for actual, represented in zip(exact_proposed, exact_model_after)
    )
    if exact_residual != step.exact_runtime_proposal_minus_represented_stage:
        raise ValueError("stage runtime/represented residual is inconsistent")
    exact_match = all(value == 0 for value in exact_residual)
    residual = max((abs(float(value)) for value in exact_residual), default=0.0)
    scale = max(
        1.0,
        *(abs(float(value)) for value in proposed),
        *(abs(float(value)) for value in exact_model_after),
    )
    within_tolerance = bool(
        math.isfinite(residual)
        and math.isfinite(scale)
        and residual <= certificate_tolerance * scale
    )
    if (
        step.runtime_proposal_matches_represented_stage_exactly != exact_match
        or step.runtime_proposal_matches_represented_stage_within_tolerance
        != within_tolerance
    ):
        raise ValueError("stage runtime/represented agreement result is inconsistent")

    for name, flow in (
        ("pre_diffusion_certificate", step.pre_diffusion_certificate),
        ("post_diffusion_certificate", step.post_diffusion_certificate),
    ):
        if not isinstance(flow, HeterogeneousDiffusionStabilityCertificate):
            raise TypeError(f"{name} must be a heterogeneous flow certificate")
        if not flow._proof_fields_are_intact():
            raise ValueError(f"{name} proof fields were replaced or mutated")
        if tuple(flow.nodes) != nodes:
            raise ValueError(f"{name} node order is inconsistent")

    pre_jump = _validate_bridge_jump(
        step.pre_metric_affine_jump_certificate,
        name="pre_metric_affine_jump_certificate",
        operator_name=operator_name,
        glyph=glyph,
        nodes=nodes,
        represented_map=represented_map,
        metric=step.pre_diffusion_certificate.metric_weights,
    )
    post_jump = _validate_bridge_jump(
        step.post_metric_affine_jump_certificate,
        name="post_metric_affine_jump_certificate",
        operator_name=operator_name,
        glyph=glyph,
        nodes=nodes,
        represented_map=represented_map,
        metric=step.post_diffusion_certificate.metric_weights,
    )
    del pre_jump, post_jump

    pre_metric = _exact_float_vector(
        step.pre_diffusion_certificate.metric_weights,
        "pre_diffusion_certificate.metric_weights",
        dimension,
    )
    post_metric = _exact_float_vector(
        step.post_diffusion_certificate.metric_weights,
        "post_diffusion_certificate.metric_weights",
        dimension,
    )
    metrics_identical = pre_metric == post_metric
    metrics_proportional = _exactly_proportional(
        step.pre_diffusion_certificate.metric_weights,
        step.post_diffusion_certificate.metric_weights,
    )
    if step.pre_post_metric_exactly_proportional != metrics_proportional:
        raise ValueError("stage pre/post metric result is inconsistent")

    exact_accepted = tuple(Fraction.from_float(float(value)) for value in accepted)
    pre_shift = _exact_weighted_mean(pre_metric, exact_accepted) - _exact_weighted_mean(
        pre_metric, exact_state
    )
    post_shift = _exact_weighted_mean(
        post_metric, exact_accepted
    ) - _exact_weighted_mean(post_metric, exact_state)
    if (
        pre_shift != step.exact_pre_metric_runtime_weighted_mean_shift
        or post_shift != step.exact_post_metric_runtime_weighted_mean_shift
    ):
        raise ValueError("stage exact weighted-mean shift is inconsistent")

    return _ValidatedNeighborStage(
        step=step,
        all_local_snapshots_in_affine_model_domain=all(local_affine_domain),
        represented_consensus_subspace_preserved=consensus_preserved,
        runtime_exact_match=exact_match,
        runtime_match_within_tolerance=within_tolerance,
        pre_post_metric_values_identical=metrics_identical,
        pre_post_metric_exactly_proportional=metrics_proportional,
        exact_runtime_post_metric_mean_shift=post_shift,
        runtime_pure_epi_pressure_defect=readonly_float_array(pressure_defect),
        runtime_pure_epi_pressure_defect_norm=pressure_defect_norm,
        runtime_pure_epi_pressure_defect_detected=pressure_defect_detected,
        pressure_refresh_required_before_runtime_flow=pressure_refresh_required,
    )


def _one_step(
    graph: Any,
    glyph: Glyph,
    *,
    index: int,
    fixed_support_declared: bool,
    mix_factor: Any,
    vf_amplification_factor: Any,
    phase_coupling_factor: Any,
    tolerance: float,
) -> tuple[AllTargetNeighborStageStep, Any]:
    pre_flow = verify_heterogeneous_diffusion_stability(graph, tolerance=tolerance)
    nodes = tuple(pre_flow.nodes)

    if glyph is Glyph.EN:
        local_certificates: tuple[LocalCertificate, ...] = tuple(
            certify_reception_epi_realization(
                graph,
                node,
                fixed_support_declared=fixed_support_declared,
                mix_factor=mix_factor,
                tolerance=tolerance,
            )
            for node in nodes
        )
        runtime_neighbors = tuple(
            certificate.runtime_neighbors for certificate in local_certificates
        )
        unclipped = tuple(
            certificate.runtime_unclipped_target_value
            for certificate in local_certificates
        )
        proposed = tuple(
            certificate.runtime_target_value for certificate in local_certificates
        )
        rejection_nodes: tuple[Any, ...] = ()
    else:
        local_certificates = tuple(
            certify_resonance_epi_realization(
                graph,
                node,
                fixed_support_declared=fixed_support_declared,
                mix_factor=mix_factor,
                vf_amplification_factor=vf_amplification_factor,
                phase_coupling_factor=phase_coupling_factor,
                tolerance=tolerance,
            )
            for node in nodes
        )
        runtime_neighbors = tuple(
            certificate.runtime_neighbors for certificate in local_certificates
        )
        unclipped = tuple(
            certificate.runtime_unclipped_target_value
            for certificate in local_certificates
        )
        proposed = tuple(
            certificate.runtime_proposed_target_value
            for certificate in local_certificates
        )
        rejection_nodes = tuple(
            node
            for node, certificate in zip(nodes, local_certificates)
            if not certificate.identity_gate_passed
        )

    state_before = np.asarray(local_certificates[0].state_before, dtype=float)
    if any(
        tuple(certificate.nodes) != nodes
        or not np.array_equal(certificate.state_before, state_before)
        for certificate in local_certificates
    ):
        raise RuntimeError("local realization node order or stage snapshot changed")

    admissible = not rejection_nodes
    accepted = proposed if admissible else tuple(float(value) for value in state_before)
    ideal_map = tuple(
        certificate.ideal_real_linear_map[certificate.target_index]
        for certificate in local_certificates
    )
    represented_map = np.vstack(
        [
            certificate.represented_linear_map[certificate.target_index]
            for certificate in local_certificates
        ]
    )
    exact_represented = exact_binary64_matrix(represented_map)
    exact_row_sums = tuple(sum(row, Fraction(0)) for row in exact_represented)
    represented_consensus = all(value == 1 for value in exact_row_sums)
    exact_state = tuple(Fraction.from_float(float(value)) for value in state_before)
    exact_represented_after = exact_matrix_vector(exact_represented, exact_state)
    exact_proposed = tuple(Fraction.from_float(float(value)) for value in proposed)
    exact_residual = tuple(
        actual - represented
        for actual, represented in zip(exact_proposed, exact_represented_after)
    )
    exact_match = all(value == 0 for value in exact_residual)
    residual_float = max(
        (abs(float(value)) for value in exact_residual),
        default=0.0,
    )
    scale = max(
        1.0,
        *(abs(float(value)) for value in proposed),
        *(abs(float(value)) for value in exact_represented_after),
    )
    within_tolerance = bool(
        math.isfinite(residual_float)
        and math.isfinite(scale)
        and residual_float <= tolerance * scale
    )
    clip_nodes = tuple(
        node
        for node, before_clip, after_clip in zip(nodes, unclipped, proposed)
        if before_clip != after_clip
    )

    first = local_certificates[0]
    mix = float(first.mix_factor)
    convex = 0.0 <= mix <= 1.0
    state_in_bounds = bool(
        np.all(state_before >= first.epi_lower_bound)
        and np.all(state_before <= first.epi_upper_bound)
    )
    ideal_clip_inactive = bool(
        first.clip_mode == "hard" and convex and state_in_bounds
    )

    post_graph = _logical_copy(graph)
    if admissible:
        for node, certificate, epi_after in zip(
            nodes, local_certificates, accepted
        ):
            mapping = post_graph.nodes[node]
            _set_epi(mapping, float(epi_after))
            set_attr_str(
                mapping,
                ALIAS_EPI_KIND,
                certificate.epi_kind_after,
            )
            if glyph is Glyph.RA:
                target_index = certificate.target_index
                set_attr(
                    mapping,
                    ALIAS_VF,
                    float(certificate.frequency_after[target_index]),
                )
                set_attr(mapping, ALIAS_THETA, float(certificate.phase_after))
    post_flow = verify_heterogeneous_diffusion_stability(
        post_graph, tolerance=tolerance
    )

    # The public gain API accepts an operator identifier as metadata.  Keep the
    # actual stage name while deriving both gains from the assembled map.
    operator_name = "Reception" if glyph is Glyph.EN else "Resonance"
    pre_jump = certify_affine_epi_jump_gain(
        operator_name,
        represented_map,
        pre_flow.metric_weights,
        nodes=nodes,
        tolerance=tolerance,
    )
    post_jump = certify_affine_epi_jump_gain(
        operator_name,
        represented_map,
        post_flow.metric_weights,
        nodes=nodes,
        tolerance=tolerance,
    )

    exact_accepted = tuple(Fraction.from_float(float(value)) for value in accepted)
    pre_mean_shift = _exact_weighted_mean(
        pre_flow.metric_weights, exact_accepted
    ) - _exact_weighted_mean(pre_flow.metric_weights, exact_state)
    post_mean_shift = _exact_weighted_mean(
        post_flow.metric_weights, exact_accepted
    ) - _exact_weighted_mean(post_flow.metric_weights, exact_state)

    step = AllTargetNeighborStageStep(
        index=index,
        nodes=nodes,
        runtime_neighbor_sets=runtime_neighbors,
        local_certificates=local_certificates,
        state_before=readonly_float_array(state_before),
        runtime_unclipped_state_after=readonly_float_array(unclipped),
        runtime_proposed_state_after=readonly_float_array(proposed),
        runtime_accepted_state_after=readonly_float_array(accepted),
        runtime_stage_admissible=admissible,
        atomic_rejection_nodes=rejection_nodes,
        clip_intervention_nodes=clip_nodes,
        clip_mode=first.clip_mode,
        epi_lower_bound=first.epi_lower_bound,
        epi_upper_bound=first.epi_upper_bound,
        mix_factor=mix,
        exact_mix_factor=first.exact_mix_factor,
        ideal_real_stage_map=ideal_map,
        represented_stage_map=readonly_float_array(represented_map),
        exact_represented_stage_map=exact_represented,
        exact_represented_row_sums=exact_row_sums,
        represented_map_exact_consensus_subspace_preservation=represented_consensus,
        exact_runtime_proposal_minus_represented_stage=exact_residual,
        runtime_proposal_matches_represented_stage_exactly=exact_match,
        runtime_proposal_matches_represented_stage_within_tolerance=within_tolerance,
        ideal_real_hard_clipping_inactive_by_convexity=ideal_clip_inactive,
        pre_diffusion_certificate=pre_flow,
        post_diffusion_certificate=post_flow,
        pre_post_metric_exactly_proportional=_exactly_proportional(
            pre_flow.metric_weights, post_flow.metric_weights
        ),
        pre_metric_affine_jump_certificate=pre_jump,
        post_metric_affine_jump_certificate=post_jump,
        exact_pre_metric_runtime_weighted_mean_shift=pre_mean_shift,
        exact_post_metric_runtime_weighted_mean_shift=post_mean_shift,
    )
    return step, post_graph


def certify_all_target_neighbor_stage(
    graph: Any,
    operator: Any,
    *,
    fixed_support_declared: bool,
    repetitions: Any = 1,
    fixed_phase_neighbor_sets_declared: bool = False,
    mix_factor: Any = None,
    vf_amplification_factor: Any = None,
    phase_coupling_factor: Any = None,
    tolerance: float = 1e-10,
) -> AllTargetNeighborStageCertificate:
    r"""Certify a finite repeated EN/RA all-target EPI-stage trace.

    ``fixed_support_declared`` and, for RA,
    ``fixed_phase_neighbor_sets_declared`` are external premises for the
    fixed-map theorem.  The finite trace itself is recomputed at every stage,
    so it can expose changed U3 sets, active clipping, identity rejection, and
    proof-metric changes without relying on those declarations.
    """

    if not isinstance(fixed_support_declared, bool):
        raise TypeError("fixed_support_declared must be a bool")
    if not isinstance(fixed_phase_neighbor_sets_declared, bool):
        raise TypeError("fixed_phase_neighbor_sets_declared must be a bool")
    count = _positive_repetition_count(repetitions)
    tol = validate_certificate_tolerance(tolerance)
    glyph = _operator_glyph(operator)
    if glyph is Glyph.EN and (
        vf_amplification_factor is not None or phase_coupling_factor is not None
    ):
        raise ValueError("RA-only factor overrides cannot be supplied for Reception")

    working = _logical_copy(graph)
    steps: list[AllTargetNeighborStageStep] = []
    for step_index in range(count):
        step, post_graph = _one_step(
            working,
            glyph,
            index=step_index,
            fixed_support_declared=fixed_support_declared,
            mix_factor=mix_factor,
            vf_amplification_factor=vf_amplification_factor,
            phase_coupling_factor=phase_coupling_factor,
            tolerance=tol,
        )
        steps.append(step)
        if not step.runtime_stage_admissible:
            break
        working = post_graph

    first = steps[0]
    nodes = first.nodes
    all_admissible = len(steps) == count and all(
        step.runtime_stage_admissible for step in steps
    )
    neighbor_sets_constant = all(
        step.runtime_neighbor_sets == first.runtime_neighbor_sets for step in steps
    )
    ideal_maps_constant = all(
        step.ideal_real_stage_map == first.ideal_real_stage_map for step in steps
    )
    represented_maps_constant = all(
        step.exact_represented_stage_map == first.exact_represented_stage_map
        for step in steps
    )
    reference_metric = first.pre_diffusion_certificate.metric_weights
    metric_common = all(
        _exactly_proportional(reference_metric, metric)
        for step in steps
        for metric in (
            step.pre_diffusion_certificate.metric_weights,
            step.post_diffusion_certificate.metric_weights,
        )
    )

    hard_clip_invariant = bool(
        first.clip_mode == "hard"
        and 0.0 <= first.mix_factor <= 1.0
        and np.all(first.state_before >= first.epi_lower_bound)
        and np.all(first.state_before <= first.epi_upper_bound)
    )

    if glyph is Glyph.RA:
        signs = {
            certificate.epi_sign_before
            for certificate in first.local_certificates
            if certificate.epi_sign_before != 0
        }
        kinds = {
            certificate.epi_kind_before for certificate in first.local_certificates
        }
        sign_forward: bool | None = len(signs) <= 1
        kind_forward: bool | None = len(kinds) <= 1
    else:
        sign_forward = None
        kind_forward = None

    ideal_conditions: list[tuple[str, bool]] = [
        ("fixed_support_declared", fixed_support_declared),
        ("hard_clip_mode", first.clip_mode == "hard"),
        ("convex_mix_factor", 0.0 <= first.mix_factor <= 1.0),
        (
            "initial_state_inside_epi_bounds",
            bool(
                np.all(first.state_before >= first.epi_lower_bound)
                and np.all(first.state_before <= first.epi_upper_bound)
            ),
        ),
        ("all_observed_stages_admissible", all_admissible),
        ("observed_ideal_real_maps_constant", ideal_maps_constant),
    ]
    if glyph is Glyph.RA:
        ideal_conditions.extend(
            (
                (
                    "fixed_phase_neighbor_sets_declared",
                    fixed_phase_neighbor_sets_declared,
                ),
                ("global_sign_identity_forward_invariant", bool(sign_forward)),
                ("global_kind_identity_forward_invariant", bool(kind_forward)),
            )
        )
    ideal_fixed = all(passed for _, passed in ideal_conditions)

    represented_conditions = tuple(ideal_conditions) + (
        (
            "represented_maps_constant",
            represented_maps_constant,
        ),
        (
            "represented_map_exact_consensus_subspace_preservation",
            first.represented_map_exact_consensus_subspace_preservation,
        ),
        (
            "represented_affine_gain_finite",
            first.pre_metric_affine_jump_certificate.supports_global_gain_theorem,
        ),
    )
    represented_fixed = all(passed for _, passed in represented_conditions)

    ideal_power = (
        _exact_matrix_power(first.ideal_real_stage_map, count)
        if ideal_fixed
        else None
    )
    represented_power = (
        _exact_matrix_power(first.exact_represented_stage_map, count)
        if represented_fixed
        else None
    )
    if represented_fixed:
        stage_gain = _composition_exact_gain(
            first.pre_metric_affine_jump_certificate
        )
        repeated_gain = stage_gain**count
        repeated_gain_float = _fraction_upper_float(repeated_gain)
        finite_contraction: bool | None = repeated_gain < 1
        asymptotic_convergence: bool | None = stage_gain < 1
    else:
        repeated_gain = None
        repeated_gain_float = None
        finite_contraction = None
        asymptotic_convergence = None

    repeated_residual: tuple[Fraction, ...] | None = None
    repeated_runtime_match: bool | None = None
    if represented_power is not None and all_admissible:
        exact_initial = tuple(
            Fraction.from_float(float(value)) for value in first.state_before
        )
        represented_final = exact_matrix_vector(represented_power, exact_initial)
        exact_runtime_final = tuple(
            Fraction.from_float(float(value))
            for value in steps[-1].runtime_accepted_state_after
        )
        repeated_residual = tuple(
            actual - represented
            for actual, represented in zip(exact_runtime_final, represented_final)
        )
        repeated_runtime_match = all(value == 0 for value in repeated_residual)

    return AllTargetNeighborStageCertificate(
        operator_name="Reception" if glyph is Glyph.EN else "Resonance",
        glyph=glyph.value,
        nodes=nodes,
        repetitions_requested=count,
        repetitions_observed=len(steps),
        repetitions_completed=sum(
            step.runtime_stage_admissible for step in steps
        ),
        fixed_support_declared=fixed_support_declared,
        fixed_phase_neighbor_sets_declared=fixed_phase_neighbor_sets_declared,
        steps=tuple(steps),
        all_stages_admissible=all_admissible,
        any_runtime_clip_intervention=any(
            step.clip_intervention_nodes for step in steps
        ),
        ideal_real_hard_clipping_inactive_for_arbitrary_repetitions=(
            hard_clip_invariant
        ),
        observed_runtime_neighbor_sets_constant=neighbor_sets_constant,
        observed_ideal_real_maps_constant=ideal_maps_constant,
        observed_represented_maps_constant=represented_maps_constant,
        observed_diffusion_metric_exactly_common=metric_common,
        ra_global_sign_identity_forward_invariant=sign_forward,
        ra_global_kind_identity_forward_invariant=kind_forward,
        ideal_real_fixed_map_repetition_conditions=tuple(ideal_conditions),
        ideal_real_fixed_map_repetition_certified=ideal_fixed,
        represented_fixed_map_repetition_conditions=represented_conditions,
        represented_fixed_map_repetition_certified=represented_fixed,
        exact_ideal_real_repeated_map=ideal_power,
        exact_represented_repeated_map=represented_power,
        exact_represented_repeated_energy_gain_bound=repeated_gain,
        represented_repeated_energy_gain_bound=repeated_gain_float,
        represented_finite_repetition_disagreement_contraction_certified=(
            finite_contraction
        ),
        represented_asymptotic_disagreement_convergence_certified=(
            asymptotic_convergence
        ),
        exact_observed_runtime_minus_represented_repeated_state=(
            repeated_residual
        ),
        observed_runtime_matches_represented_repeated_state_exactly=(
            repeated_runtime_match
        ),
        global_binary64_runtime_repetition_certified=False,
        tolerance=tol,
        scope=_SCOPE,
    )


def compose_neighbor_stage_diffusion_stability(
    stage_certificate: AllTargetNeighborStageCertificate,
    flow_duration: Any,
    *,
    tolerance: float = 1e-10,
) -> NeighborStageDiffusionBridgeCertificate:
    r"""Compose one certified all-target EN/RA stage with post-stage diffusion.

    The stage's represented affine coefficient map is treated as the reset and
    its fixed post-stage heterogeneous-flow certificate supplies the metric and
    decay rate.  The actual two-stage binary64 EN/RA proposal remains a finite
    snapshot observation.  A valid but out-of-domain stage returns an explicit
    abstention; malformed or internally inconsistent certificates are rejected.

    ``flow_duration`` is materialized as binary64 and must remain strictly
    positive.  Repeated schedules are intentionally not requested from the
    generic hybrid theorem because a single stage certificate does not prove
    future map, U3-neighbour-set or diffusion-metric invariance.
    """

    duration = _strictly_positive_flow_duration(flow_duration)
    tol = validate_certificate_tolerance(tolerance)
    validated = _validate_bridge_stage_certificate(stage_certificate)
    step = validated.step
    post_flow = step.post_diffusion_certificate
    post_jump = step.post_metric_affine_jump_certificate

    conditions = (
        ("fixed_support_declared", stage_certificate.fixed_support_declared),
        ("runtime_stage_admissible", step.runtime_stage_admissible),
        (
            "all_local_snapshots_in_affine_model_domain",
            validated.all_local_snapshots_in_affine_model_domain,
        ),
        ("runtime_hard_clipping_inactive", not step.clip_intervention_nodes),
        (
            "represented_stage_consensus_subspace_preservation",
            validated.represented_consensus_subspace_preserved,
        ),
        ("post_stage_pure_epi_flow_certificate", post_flow.is_certified),
        (
            "post_metric_affine_jump_gain_certificate",
            post_jump.supports_global_gain_theorem,
        ),
    )
    eligible = all(passed for _, passed in conditions)
    hybrid: HybridEPIStabilityCertificate | None = None
    if eligible:
        hybrid = compose_hybrid_epi_stability(
            post_flow,
            [post_jump],
            [0.0, duration],
            repeat_schedule=False,
            tolerance=tol,
        )
        if not hybrid.finite_horizon_disagreement_bound_certified:
            raise RuntimeError(
                "eligible neighbour-stage bridge failed hybrid recomposition"
            )

    represented_mean_preserved = bool(
        post_jump.exact_weighted_mean_preservation
    )
    runtime_mean_preserved = (
        validated.exact_runtime_post_metric_mean_shift == 0
    )
    return NeighborStageDiffusionBridgeCertificate(
        operator_name=stage_certificate.operator_name,
        glyph=stage_certificate.glyph,
        nodes=tuple(stage_certificate.nodes),
        flow_duration=duration,
        stage_certificate=stage_certificate,
        stage_step=step,
        post_diffusion_certificate=post_flow,
        post_metric_affine_jump_certificate=post_jump,
        hybrid_certificate=hybrid,
        bridge_conditions=conditions,
        represented_model_bridge_certified=bool(hybrid is not None),
        finite_horizon_disagreement_bound_certified=bool(
            hybrid is not None
            and hybrid.finite_horizon_disagreement_bound_certified
        ),
        disagreement_contracts_over_declared_horizon=(
            None
            if hybrid is None
            else hybrid.disagreement_contracts_over_declared_horizon
        ),
        represented_stage_preserves_post_metric_weighted_mean_exactly=(
            represented_mean_preserved
        ),
        exact_runtime_post_metric_weighted_mean_shift=(
            validated.exact_runtime_post_metric_mean_shift
        ),
        runtime_post_metric_weighted_mean_preserved_exactly=(
            runtime_mean_preserved
        ),
        hybrid_preserves_initial_weighted_mean=(
            None if hybrid is None else hybrid.initial_weighted_mean_preserved
        ),
        pre_post_metric_values_identical=(
            validated.pre_post_metric_values_identical
        ),
        pre_post_metric_exactly_proportional=(
            validated.pre_post_metric_exactly_proportional
        ),
        runtime_proposal_matches_represented_stage_exactly=(
            validated.runtime_exact_match
        ),
        runtime_proposal_matches_represented_stage_within_tolerance=(
            validated.runtime_match_within_tolerance
        ),
        runtime_pure_epi_pressure_defect=(
            validated.runtime_pure_epi_pressure_defect
        ),
        runtime_pure_epi_pressure_defect_norm=(
            validated.runtime_pure_epi_pressure_defect_norm
        ),
        runtime_pure_epi_pressure_defect_detected=(
            validated.runtime_pure_epi_pressure_defect_detected
        ),
        pressure_refresh_required_before_runtime_flow=(
            validated.pressure_refresh_required_before_runtime_flow
        ),
        stored_pressure_refresh_certified=False,
        raw_epi_embedding_revalidated_without_graph=False,
        global_binary64_runtime_affinity_certified=False,
        repeated_schedule_disagreement_convergence_certified=None,
        tolerance=tol,
        scope=_BRIDGE_SCOPE,
    )


def certify_reception_all_target_stage(
    graph: Any,
    *,
    fixed_support_declared: bool,
    repetitions: Any = 1,
    mix_factor: Any = None,
    tolerance: float = 1e-10,
) -> AllTargetNeighborStageCertificate:
    """Specialized wrapper for repeated all-target Reception stages."""

    return certify_all_target_neighbor_stage(
        graph,
        Glyph.EN,
        fixed_support_declared=fixed_support_declared,
        repetitions=repetitions,
        mix_factor=mix_factor,
        tolerance=tolerance,
    )


def certify_resonance_all_target_stage(
    graph: Any,
    *,
    fixed_support_declared: bool,
    repetitions: Any = 1,
    fixed_phase_neighbor_sets_declared: bool = False,
    mix_factor: Any = None,
    vf_amplification_factor: Any = None,
    phase_coupling_factor: Any = None,
    tolerance: float = 1e-10,
) -> AllTargetNeighborStageCertificate:
    """Specialized wrapper for repeated all-target Resonance stages."""

    return certify_all_target_neighbor_stage(
        graph,
        Glyph.RA,
        fixed_support_declared=fixed_support_declared,
        repetitions=repetitions,
        fixed_phase_neighbor_sets_declared=fixed_phase_neighbor_sets_declared,
        mix_factor=mix_factor,
        vf_amplification_factor=vf_amplification_factor,
        phase_coupling_factor=phase_coupling_factor,
        tolerance=tolerance,
    )
