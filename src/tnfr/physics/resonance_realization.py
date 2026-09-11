r"""Runtime realization bridge for one local Resonance (RA) update.

Resonance uses the same unweighted neighbour-EPI mean and two-stage binary64
blend as Reception, then conditionally increases the target capacity and moves
its phase toward the circular mean of all runtime neighbours.  Its stronger
operator contract also preserves structural identity.  The runtime now checks
that contract before mutation: a strict negative-to-positive or
positive-to-negative EPI crossing is rejected, as is replacement of an
established nonempty ``epi_kind``.  Exact zero is the neutral sign boundary.

This module keeps four different statements separate:

* the ideal-real convex affine EPI blend;
* the matrix assembled from represented binary64 coefficients;
* the actual two-stage binary64 target proposal at this snapshot;
* the accepted runtime snapshot after the identity gate.

The represented matrix can be passed to the affine jump theorem only when its
exact rational row sum preserves the consensus subspace and the snapshot lies
inside the declared clipping-free scalar domain.  The runtime remains a gated,
multichannel operation, so no global binary64 affinity claim is made.

When RA increases only the target capacity, the fixed post-RA pure-EPI flow has
metric ``h_i=d_i/nu_i`` different from the pre-RA metric in general.  A fixed
post-RA diffusion certificate remains available.  Arbitrary switching between
the pre/post generators is attempted only when the represented metric vectors
are exactly proportional.  The optional recovery calculation composes the
represented EPI jump with the *fixed post-RA* flow and does not bypass that
switching boundary.

Scope
-----
The bridge requires a declared fixed, undirected, connected
positive-conductance support with at least two nodes, positive finite
capacities, a non-isolated target, at least one U3-compatible neighbour, and
raw scalar or uniform-real BEPI payloads.  It is read-only.  It does not certify
finite-step integration, the graph's stored multichannel ``DeltaNFR``, soft
clipping, extrapolating EPI mixes, topology/history mutation, repeated words,
or global affinity of the gated runtime operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from types import SimpleNamespace
from typing import Any

from ..alias import get_attr, set_attr
from ..constants import DEFAULTS
from ..constants.aliases import (
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..constants.canonical import (
    COUPLING_FINE,
    COUPLING_GENTLE,
    COUPLING_MODERATE,
    DELTA_PHI_MAX,
)
from ..dynamics.structural_clip import structural_clip
from ..mathematics.unified_numerical import np
from ..operators._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
)
from ..operators._resonance_identity import (
    RA_RUNTIME_AMPLIFICATION_TRIGGER,
    normalize_resonance_epi_kind,
    resonance_identity_failures,
    resonance_neighbor_circular_mean,
    resonance_phase_limit_compatible,
    resonance_proposed_epi_kind,
    resonance_sign,
    validate_resonance_runtime_factors,
)
from ..types import ZERO_BEPI_STORAGE, ensure_bepi, real_scalar_epi
from ..utils import angle_diff
from ._exact_metric import (
    binary64_vectors_exactly_proportional as _exactly_proportional,
)
from ._helpers import finite_real_scalar
from .hybrid_operator_stability import (
    AffineEPIJumpGainCertificate,
    HybridEPIStabilityCertificate,
    certify_affine_epi_jump_gain,
    compose_hybrid_epi_stability,
)
from ._neighbor_epi_realization import (
    exact_binary64_matrix as _exact_binary64_matrix,
    exact_ideal_neighbor_blend_map as _exact_ideal_map,
    exact_matrix_vector as _exact_matrix_vector,
    fraction_float_or_infinity as _fraction_float_or_infinity,
    optional_flow_duration as _optional_duration,
    readonly_float_array as _readonly_array,
    represented_neighbor_blend_map as _represented_map,
    resolve_epi_bounds as _resolved_bounds,
    validate_certificate_tolerance as _validate_tolerance,
)
from .structural_diffusion import (
    HeterogeneousDiffusionStabilityCertificate,
    SwitchingDiffusionStabilityCertificate,
    structural_diffusion_operator,
    structural_field,
    verify_heterogeneous_diffusion_stability,
    verify_switching_diffusion_stability,
)

__all__ = [
    "ResonanceEPIRealizationCertificate",
    "certify_resonance_epi_realization",
]


_SCOPE = (
    "CONDITIONAL single-target Resonance realization on a declared fixed, "
    "undirected, connected support. Ideal-real EPI blending, the represented "
    "binary64 affine map, the two-stage binary64 proposal, and the accepted "
    "identity-gated snapshot are separate. Any nested jump/hybrid result "
    "applies only to the represented affine EPI map and fixed post-RA pure-EPI "
    "flow. Local capacity amplification generally changes d_i/nu_i, so "
    "pre/post switching is withheld unless the represented metrics are exactly "
    "proportional. Pressure uses detached pure-EPI arrays and is not the stored "
    "multichannel DeltaNFR. Finite-step flow, global runtime affinity, soft or "
    "active clipping, extrapolating mix, changing support, repeated words, and "
    "history effects remain outside scope."
)


def _resolve_factor(G: Any, supplied: Any, key: str, default: float) -> float:
    if supplied is not None:
        return finite_real_scalar(supplied, key)
    from ..operators import get_factor, get_glyph_factors

    factors = get_glyph_factors(SimpleNamespace(graph=G.graph), "RA")
    return float(get_factor(factors, key, default))


def _require_explicit_scalar_epi(
    G: Any, nodes: tuple[Any, ...]
) -> tuple[float, ...]:
    values: list[float] = []
    for node in nodes:
        mapping = G.nodes[node]
        if not any(alias in mapping for alias in ALIAS_EPI):
            raise ValueError("Resonance realization requires explicit EPI on every node")
        raw = get_attr(
            mapping,
            ALIAS_EPI,
            ZERO_BEPI_STORAGE,
            strict=True,
            conv=lambda value: value,
        )
        scalar = real_scalar_epi(raw)
        if scalar is None:
            raise ValueError(
                "Resonance realization requires raw scalar or uniform-real BEPI "
                "on every node"
            )
        if not math.isfinite(scalar):
            raise ValueError("Resonance realization requires finite scalar EPI")
        values.append(scalar)
    return tuple(values)


def _node_kind(G: Any, node: Any) -> str:
    value = get_attr(
        G.nodes[node],
        ALIAS_EPI_KIND,
        "",
        strict=True,
        conv=lambda item: item,
    )
    return normalize_resonance_epi_kind(value)


def _node_phase(G: Any, node: Any) -> float:
    return finite_real_scalar(
        get_attr(G.nodes[node], ALIAS_THETA, 0.0, strict=True),
        f"theta[{node!r}]",
    )


def _node_frequency(G: Any, node: Any) -> float:
    value = finite_real_scalar(
        get_attr(G.nodes[node], ALIAS_VF, 0.0, strict=True),
        f"nu_f[{node!r}]",
    )
    if value <= 0.0:
        raise ValueError("Resonance realization requires positive capacities")
    return value


def _exact_weighted_mean_data(
    metric: Any,
    ideal_map: tuple[tuple[Fraction, ...], ...],
    target_index: int,
    accepted_increment: Fraction,
) -> tuple[tuple[Fraction, ...], bool, Fraction]:
    exact_metric = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(metric, dtype=float)
    )
    mapped_row = tuple(
        sum(
            (
                exact_metric[row] * ideal_map[row][column]
                for row in range(len(exact_metric))
            ),
            Fraction(0),
        )
        for column in range(len(exact_metric))
    )
    defect = tuple(
        mapped - original for mapped, original in zip(mapped_row, exact_metric)
    )
    shift = (
        exact_metric[target_index]
        * accepted_increment
        / sum(exact_metric, Fraction(0))
    )
    return defect, all(value == 0 for value in defect), shift


def _weighted_mean(metric: Any, state: Any) -> float:
    weights = np.asarray(metric, dtype=float)
    weights = weights / float(np.sum(weights))
    return float(weights @ np.asarray(state, dtype=float))


@dataclass(frozen=True, slots=True)
class ResonanceEPIRealizationCertificate:
    """Read-only bridge from one RA snapshot to scoped affine-flow theorems."""

    nodes: tuple[Any, ...]
    target: Any
    target_index: int
    graph_neighbors: tuple[Any, ...]
    runtime_neighbors: tuple[Any, ...]
    runtime_neighbor_indices: tuple[int, ...]
    phase_incompatible_neighbors: tuple[Any, ...]
    fixed_support_declared: bool
    mix_factor: float
    exact_mix_factor: Fraction
    vf_amplification_factor: float
    phase_coupling_factor: float
    amplification_trigger_threshold: float
    epi_lower_bound: float
    epi_upper_bound: float
    clip_mode: str
    state_before: Any
    proposed_state_after: Any
    state_after: Any
    unweighted_runtime_neighbor_mean: float
    transport_weighted_neighbor_mean: float
    runtime_unclipped_target_value: float
    runtime_proposed_target_value: float
    runtime_target_value: float
    runtime_target_increment: float
    proposed_runtime_resonance_nontrivial: bool
    nontrivial_runtime_resonance: bool
    epi_sign_before: int
    proposed_epi_sign: int
    zero_is_neutral_sign_boundary: bool
    sign_identity_compatible: bool
    epi_kind_before: str
    proposed_epi_kind: str
    epi_kind_after: str
    kind_identity_compatible: bool
    identity_gate_failures: tuple[str, ...]
    identity_gate_passed: bool
    runtime_operation_admissible: bool
    phase_gate_limit: float
    phase_separations: Any
    phase_compatible_neighbors: tuple[Any, ...]
    phase_gate_passed: bool
    phase_before: float
    neighbor_circular_mean_phase: float | None
    neighbor_circular_mean_defined: bool
    proposed_phase_after: float
    phase_after: float
    frequency_before: Any
    proposed_frequency_after: Any
    frequency_after: Any
    frequency_amplification_active: bool
    frequency_nondecreasing: bool
    metric_weights_before: Any
    metric_weights_after: Any
    pre_post_metric_exactly_proportional: bool
    pre_diffusion_certificate: HeterogeneousDiffusionStabilityCertificate
    post_diffusion_certificate: HeterogeneousDiffusionStabilityCertificate | None
    pre_post_switching_certificate: SwitchingDiffusionStabilityCertificate | None
    pre_post_switching_theorem_certified: bool | None
    switching_abstention_reasons: tuple[str, ...]
    ideal_real_hard_clipping_inactive_by_convexity: bool
    runtime_hard_clipping_inactive_at_snapshot: bool
    ideal_real_linear_map: tuple[tuple[Fraction, ...], ...]
    represented_linear_map: Any
    exact_represented_target_row_sum: Fraction
    ideal_real_consensus_subspace_preservation: bool
    represented_map_exact_consensus_subspace_preservation: bool
    exact_runtime_minus_ideal_target: Fraction
    exact_runtime_minus_represented_affine_target: Fraction
    runtime_matches_ideal_target_exactly_at_snapshot: bool
    runtime_matches_represented_affine_exactly_at_snapshot: bool
    runtime_matches_represented_affine_within_tolerance: bool
    ideal_real_affine_conditions: tuple[tuple[str, bool], ...]
    ideal_real_affine_regime: bool
    runtime_affine_model_domain_conditions: tuple[tuple[str, bool], ...]
    runtime_snapshot_in_affine_model_domain: bool
    represented_affine_jump_eligible: bool
    affine_abstention_reasons: tuple[str, ...]
    nested_affine_certificate_uses_post_resonance_metric: bool
    global_binary64_runtime_affinity_certified: bool
    affine_jump_certificate: AffineEPIJumpGainCertificate | None
    current_pure_epi_pressure: Any
    post_reset_pure_epi_pressure: Any
    post_reset_pressure_manifold_defect: Any
    post_reset_pressure_manifold_defect_norm: float
    proposed_pressure_refresh_required: bool
    pressure_refresh_required: bool
    exact_pressure_refresh_iff_nontrivial_theorem: bool
    pressure_refresh_detected_at_snapshot: bool
    numerical_pressure_refresh_iff_nontrivial_at_snapshot: bool
    exact_ideal_post_metric_weighted_mean_linear_defect: tuple[Fraction, ...]
    ideal_real_post_metric_weighted_mean_preservation: bool
    exact_pre_metric_weighted_mean_shift: Fraction
    exact_post_metric_weighted_mean_shift: Fraction
    pre_metric_weighted_mean_before: float
    pre_metric_weighted_mean_after: float
    post_metric_weighted_mean_before: float
    post_metric_weighted_mean_after: float
    recovery_break_even_duration_estimate: float | None
    recovery_flow_duration: float | None
    hybrid_certificate: HybridEPIStabilityCertificate | None
    represented_hybrid_recovery_certified: bool | None
    tolerance: float
    scope: str

    @property
    def failed_ideal_real_affine_conditions(self) -> tuple[str, ...]:
        return tuple(
            name for name, passed in self.ideal_real_affine_conditions if not passed
        )

    @property
    def failed_runtime_affine_model_domain_conditions(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, passed in self.runtime_affine_model_domain_conditions
            if not passed
        )


def certify_resonance_epi_realization(
    G: Any,
    target: Any,
    *,
    fixed_support_declared: bool,
    mix_factor: Any = None,
    vf_amplification_factor: Any = None,
    phase_coupling_factor: Any = None,
    recovery_flow_duration: Any = None,
    tolerance: float = 1e-10,
) -> ResonanceEPIRealizationCertificate:
    r"""Audit one RA proposal without mutating ``G``.

    A failed identity gate is returned as a negative certificate whose accepted
    state equals the input state.  Directed, disconnected, isolated,
    phase-incompatible, or genuinely non-scalar inputs are rejected because
    they do not define the certified runtime domain.
    """

    if not isinstance(fixed_support_declared, bool):
        raise TypeError("fixed_support_declared must be a bool")
    tol = _validate_tolerance(tolerance)
    duration = _optional_duration(recovery_flow_duration)

    try:
        directed = bool(G.is_directed())
    except (AttributeError, TypeError) as exc:
        raise TypeError("G must expose the NetworkX graph interface") from exc
    if directed:
        raise ValueError("Resonance realization requires an undirected graph")
    if target not in G:
        raise ValueError("target must be present in the graph")

    nodes = tuple(G)
    if len(nodes) < 2:
        raise ValueError("Resonance realization requires at least two nodes")
    scalar_values = _require_explicit_scalar_epi(G, nodes)
    pre_flow = verify_heterogeneous_diffusion_stability(G, tolerance=tol)
    if tuple(pre_flow.nodes) != nodes:
        raise RuntimeError("diffusion node order changed during Resonance audit")
    node_index = {node: index for index, node in enumerate(nodes)}
    target_index = node_index[target]
    graph_neighbors = tuple(G.neighbors(target))
    if not graph_neighbors:
        raise ValueError("Resonance realization requires a non-isolated target")

    state = np.asarray(scalar_values, dtype=float)
    flow_state = structural_field(G, list(nodes))
    if not np.array_equal(state, flow_state):
        raise ValueError("Resonance scalar EPI chart disagrees with pure-EPI flow")

    mix = _resolve_factor(G, mix_factor, "RA_epi_diff", COUPLING_MODERATE)
    vf_boost = _resolve_factor(
        G,
        vf_amplification_factor,
        "RA_vf_amplification",
        COUPLING_FINE,
    )
    phase_coupling = _resolve_factor(
        G,
        phase_coupling_factor,
        "RA_phase_coupling",
        COUPLING_GENTLE,
    )
    factor_failures = validate_resonance_runtime_factors(
        mix, vf_boost, phase_coupling
    )
    if factor_failures:
        raise ValueError("; ".join(factor_failures))
    exact_mix = Fraction.from_float(mix)
    lower, upper, clip_mode = _resolved_bounds(G)

    phases = tuple(_node_phase(G, node) for node in nodes)
    phase_limit = finite_real_scalar(
        G.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX), "DELTA_PHI_MAX"
    )
    if not resonance_phase_limit_compatible(phase_limit, DELTA_PHI_MAX):
        raise ValueError(
            "DELTA_PHI_MAX must lie in the canonical interval "
            f"[0, {DELTA_PHI_MAX}]"
        )
    target_phase = phases[target_index]
    separations = tuple(
        abs(angle_diff(target_phase, phases[node_index[node]]))
        for node in graph_neighbors
    )
    runtime_neighbors = tuple(
        node
        for node, separation in zip(graph_neighbors, separations)
        if separation <= phase_limit
    )
    phase_incompatible_neighbors = tuple(
        node
        for node, separation in zip(graph_neighbors, separations)
        if separation > phase_limit
    )
    if not runtime_neighbors:
        raise ValueError(
            "Resonance realization requires at least one U3 phase-compatible neighbor"
        )
    runtime_neighbor_indices = tuple(node_index[node] for node in runtime_neighbors)

    neighbor_phases = tuple(phases[index] for index in runtime_neighbor_indices)
    neighbor_phase_mean, phase_mean_defined = resonance_neighbor_circular_mean(
        neighbor_phases
    )
    proposed_phase = target_phase
    if phase_mean_defined and neighbor_phase_mean is not None:
        proposed_phase = (
            target_phase
            + phase_coupling * angle_diff(neighbor_phase_mean, target_phase)
        ) % (2.0 * math.pi)
    if not math.isfinite(proposed_phase):
        raise ValueError("the proposed Resonance phase must remain finite")

    neighbor_values = tuple(state[index] for index in runtime_neighbor_indices)
    neighbor_mean = neighbor_epi_unweighted_mean(neighbor_values)
    raw_target = get_attr(
        G.nodes[target],
        ALIAS_EPI,
        ZERO_BEPI_STORAGE,
        strict=True,
        conv=lambda value: value,
    )
    runtime_operand = ensure_bepi(raw_target)
    unclipped_target = neighbor_epi_blend_value(runtime_operand, neighbor_mean, mix)
    proposed_target = float(
        structural_clip(
            unclipped_target,
            lo=lower,
            hi=upper,
            mode=clip_mode,
            record_stats=False,
        )
    )
    before_target = float(state[target_index])
    target_kind = _node_kind(G, target)
    neighbor_value_kinds = tuple(
        (float(state[index]), _node_kind(G, node))
        for node, index in zip(runtime_neighbors, runtime_neighbor_indices)
    )
    proposed_kind = resonance_proposed_epi_kind(
        target_kind,
        neighbor_value_kinds,
        proposed_target,
    )
    identity_failures = resonance_identity_failures(
        before_target, proposed_target, target_kind, proposed_kind
    )
    identity_passed = not identity_failures

    proposed_state = np.array(state, dtype=float, copy=True)
    proposed_state[target_index] = proposed_target
    accepted_state = (
        np.array(proposed_state, copy=True)
        if identity_passed
        else np.array(state, dtype=float, copy=True)
    )
    accepted_target = float(accepted_state[target_index])
    accepted_increment_float = accepted_target - before_target
    accepted_increment = Fraction.from_float(accepted_target) - Fraction.from_float(
        before_target
    )
    proposed_nontrivial = proposed_target != before_target
    accepted_nontrivial = identity_passed and proposed_nontrivial

    frequencies = np.asarray(tuple(_node_frequency(G, node) for node in nodes))
    amplification_active = bool(
        identity_passed
        and abs(neighbor_mean) > RA_RUNTIME_AMPLIFICATION_TRIGGER
    )
    proposed_frequencies = np.array(frequencies, dtype=float, copy=True)
    if abs(neighbor_mean) > RA_RUNTIME_AMPLIFICATION_TRIGGER:
        proposed_frequencies[target_index] = frequencies[target_index] * (1.0 + vf_boost)
    accepted_frequencies = (
        np.array(proposed_frequencies, copy=True)
        if identity_passed
        else np.array(frequencies, copy=True)
    )
    if identity_passed and (
        not np.all(np.isfinite(accepted_frequencies))
        or np.any(accepted_frequencies <= 0.0)
    ):
        raise ValueError(
            "the proposed Resonance capacity must remain finite and positive"
        )
    frequency_nondecreasing = bool(
        accepted_frequencies[target_index] >= frequencies[target_index]
    )
    accepted_phase = proposed_phase if identity_passed else target_phase
    accepted_kind = proposed_kind if identity_passed else target_kind

    ideal_map = _exact_ideal_map(
        len(nodes), target_index, runtime_neighbor_indices, exact_mix
    )
    represented_map = _represented_map(
        len(nodes), target_index, runtime_neighbor_indices, mix
    )
    represented_exact = _exact_binary64_matrix(represented_map)
    exact_state = tuple(Fraction.from_float(float(value)) for value in state)
    exact_ideal_after = _exact_matrix_vector(ideal_map, exact_state)
    exact_represented_after = _exact_matrix_vector(represented_exact, exact_state)
    exact_runtime_target = Fraction.from_float(proposed_target)
    runtime_minus_ideal = exact_runtime_target - exact_ideal_after[target_index]
    runtime_minus_represented = (
        exact_runtime_target - exact_represented_after[target_index]
    )
    represented_row_sum = sum(represented_exact[target_index], Fraction(0))
    represented_consensus = represented_row_sum == 1
    represented_residual = abs(_fraction_float_or_infinity(runtime_minus_represented))
    represented_target_float = _fraction_float_or_infinity(
        exact_represented_after[target_index]
    )
    represented_scale = max(
        1.0, abs(proposed_target), abs(represented_target_float)
    )
    represented_within = bool(
        math.isfinite(represented_residual)
        and math.isfinite(represented_scale)
        and represented_residual <= tol * represented_scale
    )

    hard_clip = clip_mode == "hard"
    convex_mix = 0.0 <= mix <= 1.0
    state_in_bounds = bool(np.all((state >= lower) & (state <= upper)))
    clipping_inactive = proposed_target == unclipped_target
    ideal_clipping_inactive = hard_clip and convex_mix and state_in_bounds
    ideal_conditions = (
        ("fixed_support_declared", fixed_support_declared),
        ("hard_clip_mode", hard_clip),
        ("convex_mix_factor", convex_mix),
        ("state_inside_epi_bounds", state_in_bounds),
    )
    ideal_regime = all(passed for _, passed in ideal_conditions)

    post_flow: HeterogeneousDiffusionStabilityCertificate | None = None
    switching: SwitchingDiffusionStabilityCertificate | None = None
    switching_reasons: list[str] = []
    post_metric = np.asarray(pre_flow.metric_weights, dtype=float)
    metric_common = False
    post_graph = None
    if identity_passed:
        post_graph = G.copy()
        set_attr(post_graph.nodes[target], ALIAS_EPI, accepted_target)
        set_attr(
            post_graph.nodes[target],
            ALIAS_VF,
            float(accepted_frequencies[target_index]),
        )
        set_attr(post_graph.nodes[target], ALIAS_THETA, accepted_phase)
        post_flow = verify_heterogeneous_diffusion_stability(post_graph, tolerance=tol)
        post_metric = np.asarray(post_flow.metric_weights, dtype=float)
        metric_common = _exactly_proportional(pre_flow.metric_weights, post_metric)
        if not fixed_support_declared:
            switching_reasons.append("fixed_support_not_declared")
        if not metric_common:
            switching_reasons.append("pre_post_metrics_not_exactly_proportional")
        if fixed_support_declared and metric_common:
            switching = verify_switching_diffusion_stability(
                [G, post_graph], tolerance=tol
            )
    else:
        switching_reasons.append("identity_gate_rejected_runtime_operation")

    runtime_conditions = ideal_conditions + (
        ("u3_phase_gate", True),
        ("scalar_epi_embedding", True),
        ("runtime_hard_clipping_inactive_at_snapshot", clipping_inactive),
        ("identity_gate_passed", identity_passed),
        ("frequency_not_decreased", frequency_nondecreasing),
        ("post_diffusion_certificate_available", post_flow is not None),
    )
    runtime_in_affine_domain = all(passed for _, passed in runtime_conditions)

    affine_jump: AffineEPIJumpGainCertificate | None = None
    if runtime_in_affine_domain and represented_consensus and post_flow is not None:
        candidate = certify_affine_epi_jump_gain(
            "Resonance",
            represented_map,
            post_flow.metric_weights,
            nodes=nodes,
            tolerance=tol,
        )
        if candidate.exact_consensus_subspace_preservation:
            affine_jump = candidate
    represented_eligible = affine_jump is not None
    affine_reasons = [name for name, passed in runtime_conditions if not passed]
    if not represented_consensus:
        affine_reasons.append("represented_binary64_row_sum_is_not_exactly_one")
    elif runtime_in_affine_domain and affine_jump is None:
        affine_reasons.append("affine_jump_exact_hypotheses_failed")

    lap_nodes, laplacian = structural_diffusion_operator(G)
    if tuple(lap_nodes) != nodes:
        raise RuntimeError("diffusion node order changed during Resonance audit")
    try:
        with np.errstate(over="raise", invalid="raise"):
            current_pressure = -(laplacian @ state)
            post_pressure = -(laplacian @ accepted_state)
            pressure_defect = current_pressure - post_pressure
            pressure_defect_norm = float(
                np.max(np.abs(pressure_defect), initial=0.0)
            )
    except FloatingPointError as exc:
        raise ValueError("pure-EPI pressure diagnostic exceeds binary64 range") from exc
    pressure_detected = bool(np.any(pressure_defect != 0.0))
    transport_neighbor_mean = float(state[target_index] + current_pressure[target_index])

    pre_defect, _pre_ideal_mean, pre_shift = _exact_weighted_mean_data(
        pre_flow.metric_weights, ideal_map, target_index, accepted_increment
    )
    post_defect, post_ideal_mean, post_shift = _exact_weighted_mean_data(
        post_metric, ideal_map, target_index, accepted_increment
    )
    del pre_defect
    pre_mean_before = _weighted_mean(pre_flow.metric_weights, state)
    pre_mean_after = _weighted_mean(pre_flow.metric_weights, accepted_state)
    post_mean_before = _weighted_mean(post_metric, state)
    post_mean_after = _weighted_mean(post_metric, accepted_state)

    break_even: float | None = None
    hybrid: HybridEPIStabilityCertificate | None = None
    hybrid_recovery: bool | None = None
    if affine_jump is not None and post_flow is not None:
        gain = affine_jump.energy_gain_bound_for_composition
        rate = post_flow.certified_exponential_rate_lower_bound
        if post_flow.is_certified and rate > 0.0:
            if gain == 0.0:
                break_even = 0.0
            elif math.isfinite(gain):
                estimate = max(0.0, math.log(gain) / rate)
                if math.isfinite(estimate):
                    break_even = estimate
        if duration is not None and post_flow.is_certified and rate > 0.0:
            hybrid = compose_hybrid_epi_stability(
                post_flow,
                [affine_jump],
                [0.0, duration],
                repeat_schedule=False,
                tolerance=tol,
            )
            hybrid_recovery = bool(
                hybrid.disagreement_contracts_over_declared_horizon
            )

    return ResonanceEPIRealizationCertificate(
        nodes=nodes,
        target=target,
        target_index=target_index,
        graph_neighbors=graph_neighbors,
        runtime_neighbors=runtime_neighbors,
        runtime_neighbor_indices=runtime_neighbor_indices,
        phase_incompatible_neighbors=phase_incompatible_neighbors,
        fixed_support_declared=fixed_support_declared,
        mix_factor=mix,
        exact_mix_factor=exact_mix,
        vf_amplification_factor=vf_boost,
        phase_coupling_factor=phase_coupling,
        amplification_trigger_threshold=RA_RUNTIME_AMPLIFICATION_TRIGGER,
        epi_lower_bound=lower,
        epi_upper_bound=upper,
        clip_mode=clip_mode,
        state_before=_readonly_array(state),
        proposed_state_after=_readonly_array(proposed_state),
        state_after=_readonly_array(accepted_state),
        unweighted_runtime_neighbor_mean=neighbor_mean,
        transport_weighted_neighbor_mean=transport_neighbor_mean,
        runtime_unclipped_target_value=unclipped_target,
        runtime_proposed_target_value=proposed_target,
        runtime_target_value=accepted_target,
        runtime_target_increment=accepted_increment_float,
        proposed_runtime_resonance_nontrivial=proposed_nontrivial,
        nontrivial_runtime_resonance=accepted_nontrivial,
        epi_sign_before=resonance_sign(before_target),
        proposed_epi_sign=resonance_sign(proposed_target),
        zero_is_neutral_sign_boundary=True,
        sign_identity_compatible=(
            "nonzero_epi_sign_would_flip" not in identity_failures
        ),
        epi_kind_before=target_kind,
        proposed_epi_kind=proposed_kind,
        epi_kind_after=accepted_kind,
        kind_identity_compatible=(
            "established_epi_kind_would_change" not in identity_failures
        ),
        identity_gate_failures=identity_failures,
        identity_gate_passed=identity_passed,
        runtime_operation_admissible=identity_passed,
        phase_gate_limit=phase_limit,
        phase_separations=_readonly_array(separations),
        phase_compatible_neighbors=runtime_neighbors,
        phase_gate_passed=True,
        phase_before=target_phase,
        neighbor_circular_mean_phase=neighbor_phase_mean,
        neighbor_circular_mean_defined=phase_mean_defined,
        proposed_phase_after=proposed_phase,
        phase_after=accepted_phase,
        frequency_before=_readonly_array(frequencies),
        proposed_frequency_after=_readonly_array(proposed_frequencies),
        frequency_after=_readonly_array(accepted_frequencies),
        frequency_amplification_active=amplification_active,
        frequency_nondecreasing=frequency_nondecreasing,
        metric_weights_before=_readonly_array(pre_flow.metric_weights),
        metric_weights_after=_readonly_array(post_metric),
        pre_post_metric_exactly_proportional=metric_common,
        pre_diffusion_certificate=pre_flow,
        post_diffusion_certificate=post_flow,
        pre_post_switching_certificate=switching,
        pre_post_switching_theorem_certified=(
            bool(switching.supports_exact_switching_theorem)
            if switching is not None
            else None
        ),
        switching_abstention_reasons=tuple(switching_reasons),
        ideal_real_hard_clipping_inactive_by_convexity=ideal_clipping_inactive,
        runtime_hard_clipping_inactive_at_snapshot=clipping_inactive,
        ideal_real_linear_map=ideal_map,
        represented_linear_map=_readonly_array(represented_map),
        exact_represented_target_row_sum=represented_row_sum,
        ideal_real_consensus_subspace_preservation=True,
        represented_map_exact_consensus_subspace_preservation=represented_consensus,
        exact_runtime_minus_ideal_target=runtime_minus_ideal,
        exact_runtime_minus_represented_affine_target=runtime_minus_represented,
        runtime_matches_ideal_target_exactly_at_snapshot=runtime_minus_ideal == 0,
        runtime_matches_represented_affine_exactly_at_snapshot=(
            runtime_minus_represented == 0
        ),
        runtime_matches_represented_affine_within_tolerance=represented_within,
        ideal_real_affine_conditions=ideal_conditions,
        ideal_real_affine_regime=ideal_regime,
        runtime_affine_model_domain_conditions=runtime_conditions,
        runtime_snapshot_in_affine_model_domain=runtime_in_affine_domain,
        represented_affine_jump_eligible=represented_eligible,
        affine_abstention_reasons=tuple(affine_reasons),
        nested_affine_certificate_uses_post_resonance_metric=True,
        global_binary64_runtime_affinity_certified=False,
        affine_jump_certificate=affine_jump,
        current_pure_epi_pressure=_readonly_array(current_pressure),
        post_reset_pure_epi_pressure=_readonly_array(post_pressure),
        post_reset_pressure_manifold_defect=_readonly_array(pressure_defect),
        post_reset_pressure_manifold_defect_norm=pressure_defect_norm,
        proposed_pressure_refresh_required=proposed_nontrivial,
        pressure_refresh_required=accepted_nontrivial,
        exact_pressure_refresh_iff_nontrivial_theorem=True,
        pressure_refresh_detected_at_snapshot=pressure_detected,
        numerical_pressure_refresh_iff_nontrivial_at_snapshot=(
            pressure_detected == accepted_nontrivial
        ),
        exact_ideal_post_metric_weighted_mean_linear_defect=post_defect,
        ideal_real_post_metric_weighted_mean_preservation=post_ideal_mean,
        exact_pre_metric_weighted_mean_shift=pre_shift,
        exact_post_metric_weighted_mean_shift=post_shift,
        pre_metric_weighted_mean_before=pre_mean_before,
        pre_metric_weighted_mean_after=pre_mean_after,
        post_metric_weighted_mean_before=post_mean_before,
        post_metric_weighted_mean_after=post_mean_after,
        recovery_break_even_duration_estimate=break_even,
        recovery_flow_duration=duration,
        hybrid_certificate=hybrid,
        represented_hybrid_recovery_certified=hybrid_recovery,
        tolerance=tol,
        scope=_SCOPE,
    )
