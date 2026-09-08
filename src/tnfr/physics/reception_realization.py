r"""Runtime realization bridge for one local Reception (EN) update.

Reception changes one EPI coordinate by blending it with the arithmetic mean
of the target's runtime neighbours and proposes the semantic EPI kind from the
dominant labelled neighbour.  On a fixed support, with a mix
``alpha in [0, 1]``, hard clipping, and an EPI field already inside its bounds,
the corresponding *ideal real* update is the affine reset

``x_i+ = (1-alpha) x_i + alpha/|N_i| sum_{j in N_i} x_j``.

In exact real arithmetic this convex combination remains in the same interval,
so hard clipping is inactive by theorem.  Whether the two-stage binary64
runtime value was unchanged by its clip is recorded as a separate snapshot
diagnostic.

That real map preserves constants, but a matrix assembled from separately
rounded binary64 coefficients need not do so exactly: the represented row sum
``fl(1-alpha) + |N_i| fl(alpha/|N_i|)`` can differ from one as a rational
identity.  The engine also evaluates a neighbour mean and then a blend rather
than a matrix product.  This module therefore keeps three objects separate:

* the exact rational ideal-real map induced by the represented ``alpha``;
* the declared matrix of represented binary64 coefficients;
* the actual shared BEPI-to-scalar runtime kernel evaluated at the snapshot.

Only the second object is passed to the affine jump theorem, and only when its
exact represented consensus hypothesis holds and every payload uses the
uniform real-scalar BEPI embedding shared by runtime and pure-EPI flow.
Agreement of the runtime kernel with the matrix is still a snapshot
diagnostic, never a global floating-point linearity claim.

The reset changes EPI without refreshing pressure.  The certificate evaluates
the canonical pure-EPI pressure before and after the reset on detached arrays.
On the required connected support, the exact real identity
``p(x)-p(x+delta e_i) = delta L_rw e_i`` proves that the pressure-manifold
defect is nonzero exactly when the runtime target coordinate changes.  The
separately reported binary64 pressure evaluation can lose a very small defect
to rounding or underflow.  This does not identify the graph's stored full
multichannel ``DeltaNFR`` with the pure-EPI channel.

Scope
-----
The bridge requires a declared fixed, undirected, connected positive-conductance
support, at least two nodes, positive capacities, a present target, and a
nonempty runtime neighbour set. EN's neighbour mean is unweighted even when
the diffusion conductance is weighted. Runtime promotion additionally requires
the uniform real-scalar BEPI embedding. General nonuniform or complex BEPI
payloads are rejected because they do not define the same real state for the
strict pure-EPI diffusion model. The optional recovery result concerns the
represented affine model composed with frozen continuous-time pure-EPI
diffusion. It does not certify soft clipping, extrapolating mix factors,
finite-step integration, other pressure channels, phase, topology changes,
operator grammar/history effects, or repeated operator words.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from types import SimpleNamespace
from typing import Any

from ..constants.aliases import ALIAS_EPI, ALIAS_EPI_KIND
from ..constants.canonical import EN_MIX_FACTOR
from ..dynamics.structural_clip import structural_clip
from ..mathematics.unified_numerical import np
from ..operators._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
    reception_proposed_epi_kind,
)
from ..types import Glyph, ZERO_BEPI_STORAGE, ensure_bepi
from ._helpers import finite_real_scalar
from ._neighbor_epi_realization import (
    exact_binary64_matrix as _exact_binary64_matrix,
    exact_ideal_neighbor_blend_map as _exact_ideal_map,
    exact_matrix_vector as _exact_matrix_vector,
    fraction_float_or_infinity as _fraction_float_or_infinity,
    optional_flow_duration as _optional_duration,
    readonly_float_array as _readonly_array,
    represented_neighbor_blend_map as _represented_map,
    require_explicit_epi as _require_explicit_epi_shared,
    resolve_epi_bounds as _resolved_bounds,
    uses_scalar_epi_embedding as _uses_scalar_epi_embedding,
    validate_certificate_tolerance as _validate_tolerance,
)
from .hybrid_operator_stability import (
    AffineEPIJumpGainCertificate,
    HybridEPIStabilityCertificate,
    certify_affine_epi_jump_gain,
    compose_hybrid_epi_stability,
)
from .structural_diffusion import (
    HeterogeneousDiffusionStabilityCertificate,
    structural_diffusion_operator,
    structural_field,
    verify_heterogeneous_diffusion_stability,
)

__all__ = [
    "ReceptionEPIRealizationCertificate",
    "certify_reception_epi_realization",
]


_SCOPE = (
    "CONDITIONAL single-target Reception realization on a declared fixed, "
    "undirected, connected support. The ideal-real affine identity, the "
    "represented binary64 coefficient map, and the two-stage scalar runtime "
    "evaluation are reported separately. Any nested jump/hybrid certificate "
    "applies only to the represented affine model. Pressure diagnostics use "
    "the pure-EPI channel on detached arrays and do not refresh or identify "
    "the graph's full multichannel DeltaNFR. The EPI-kind proposal is a "
    "snapshot runtime diagnostic outside the affine theorem. Soft clipping, "
    "nonconvex mixing, "
    "nonuniform or complex BEPI payloads, finite-step flow, grammar/history "
    "effects, changing support/capacity, other pressure channels, phase, and "
    "repeated words remain outside scope."
)


def _resolve_mix(G: Any, supplied: Any) -> float:
    if supplied is not None:
        return finite_real_scalar(supplied, "mix_factor")

    # Resolve through the same merged factor API used by ``apply_glyph_obj``.
    # The lightweight subject avoids constructing a NodeNX adapter and hence
    # avoids adding a cache entry to the audited graph.
    from ..operators import get_factor, get_glyph_factors

    factors = get_glyph_factors(SimpleNamespace(graph=G.graph), "EN")
    return float(get_factor(factors, "EN_mix", EN_MIX_FACTOR))


def _require_explicit_epi(G: Any, nodes: tuple[Any, ...]) -> None:
    _require_explicit_epi_shared(G, nodes, operator="Reception")


@dataclass(frozen=True, slots=True)
class ReceptionEPIRealizationCertificate:
    """Scoped bridge from the EN runtime formula to affine stability models."""

    nodes: tuple[Any, ...]
    target: Any
    target_index: int
    runtime_neighbors: tuple[Any, ...]
    runtime_neighbor_indices: tuple[int, ...]
    fixed_support_declared: bool
    mix_factor: float
    exact_mix_factor: Fraction
    epi_lower_bound: float
    epi_upper_bound: float
    clip_mode: str
    state_before: Any
    state_after: Any
    unweighted_runtime_neighbor_mean: float
    transport_weighted_neighbor_mean: float
    runtime_unclipped_target_value: float
    runtime_target_value: float
    runtime_target_increment: float
    nontrivial_runtime_reception: bool
    epi_kind_before: str
    epi_kind_after: str
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
    nested_affine_certificate_is_represented_model_only: bool
    global_binary64_runtime_affinity_certified: bool
    diffusion_certificate: HeterogeneousDiffusionStabilityCertificate
    affine_jump_certificate: AffineEPIJumpGainCertificate | None
    current_pure_epi_pressure: Any
    post_reset_pure_epi_pressure: Any
    post_reset_pressure_manifold_defect: Any
    post_reset_pressure_manifold_defect_norm: float
    pressure_refresh_required: bool
    exact_pressure_refresh_iff_nontrivial_theorem: bool
    pressure_refresh_detected_at_snapshot: bool
    numerical_pressure_refresh_iff_nontrivial_at_snapshot: bool
    exact_ideal_weighted_mean_linear_defect: tuple[Fraction, ...]
    ideal_real_global_weighted_mean_preservation: bool
    exact_current_weighted_mean_shift: Fraction
    weighted_mean_before: float
    weighted_mean_after: float
    weighted_mean_shift: float
    current_weighted_mean_preserved_exactly: bool
    recovery_break_even_duration_estimate: float | None
    recovery_flow_duration: float | None
    hybrid_certificate: HybridEPIStabilityCertificate | None
    represented_hybrid_recovery_certified: bool | None
    tolerance: float
    scope: str

    @property
    def failed_ideal_real_affine_conditions(self) -> tuple[str, ...]:
        """Return failed hypotheses of the clipping-free ideal-real regime."""

        return tuple(
            name for name, passed in self.ideal_real_affine_conditions if not passed
        )

    @property
    def failed_runtime_affine_model_domain_conditions(self) -> tuple[str, ...]:
        """Return failed hypotheses linking this snapshot to the model domain."""

        return tuple(
            name
            for name, passed in self.runtime_affine_model_domain_conditions
            if not passed
        )


def certify_reception_epi_realization(
    G: Any,
    target: Any,
    *,
    fixed_support_declared: bool,
    mix_factor: Any = None,
    recovery_flow_duration: Any = None,
    tolerance: float = 1e-10,
) -> ReceptionEPIRealizationCertificate:
    r"""Audit one EN update without mutating ``G``.

    ``fixed_support_declared`` is an explicit external premise: a snapshot
    cannot establish that topology and capacities remain fixed during the
    optional recovery flow.  A false declaration still returns diagnostics but
    prevents affine/hybrid theorem promotion.

    ``recovery_flow_duration``, when supplied, composes the eligible represented
    affine jump with frozen continuous-time pure-EPI diffusion.  The resulting
    hybrid certificate concerns that represented model, not exact global
    binary64 runtime arithmetic.
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
        raise ValueError("Reception realization requires an undirected graph")
    if target not in G:
        raise ValueError("target must be present in the graph")

    flow = verify_heterogeneous_diffusion_stability(G, tolerance=tol)
    nodes = tuple(flow.nodes)
    if len(nodes) < 2:
        raise ValueError("Reception realization requires at least two nodes")
    _require_explicit_epi(G, nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    target_index = node_index[target]
    runtime_neighbors = tuple(G.neighbors(target))
    if not runtime_neighbors:
        raise ValueError("Reception requires a nonempty runtime neighbor set")
    runtime_neighbor_indices = tuple(node_index[node] for node in runtime_neighbors)

    state = structural_field(G, list(nodes))
    if not np.all(np.isfinite(state)):
        raise ValueError("Reception realization requires finite scalar EPI")
    mix = _resolve_mix(G, mix_factor)
    exact_mix = Fraction.from_float(mix)
    lower, upper, clip_mode = _resolved_bounds(G)

    neighbor_values = tuple(float(state[index]) for index in runtime_neighbor_indices)
    neighbor_mean = neighbor_epi_unweighted_mean(neighbor_values)
    from ..alias import get_attr

    raw_target_epi = get_attr(
        G.nodes[target],
        ALIAS_EPI,
        ZERO_BEPI_STORAGE,
        strict=True,
        conv=lambda value: value,
    )
    runtime_epi_operand = ensure_bepi(raw_target_epi)
    unclipped_target = neighbor_epi_blend_value(
        runtime_epi_operand, neighbor_mean, mix
    )
    runtime_target = float(
        structural_clip(
            unclipped_target,
            lo=lower,
            hi=upper,
            mode=clip_mode,
            record_stats=False,
        )
    )
    state_after = np.array(state, dtype=float, copy=True)
    state_after[target_index] = runtime_target
    increment = runtime_target - float(state[target_index])
    nontrivial = runtime_target != float(state[target_index])
    clipping_inactive = runtime_target == unclipped_target

    def node_kind(node: Any) -> str:
        return str(
            get_attr(
                G.nodes[node],
                ALIAS_EPI_KIND,
                "",
                strict=True,
                conv=lambda value: value,
            )
        )

    kind_before = node_kind(target)
    kind_after = reception_proposed_epi_kind(
        kind_before,
        (
            (float(state[index]), node_kind(node))
            for node, index in zip(runtime_neighbors, runtime_neighbor_indices)
        ),
        unclipped_target_epi=unclipped_target,
        fallback_kind=Glyph.EN.value,
    )

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
    exact_runtime_target = Fraction.from_float(runtime_target)
    runtime_minus_ideal = exact_runtime_target - exact_ideal_after[target_index]
    runtime_minus_represented = (
        exact_runtime_target - exact_represented_after[target_index]
    )
    represented_row_sum = sum(represented_exact[target_index], Fraction(0))
    represented_consensus = represented_row_sum == 1
    runtime_represented_residual = abs(
        _fraction_float_or_infinity(runtime_minus_represented)
    )
    represented_target_float = _fraction_float_or_infinity(
        exact_represented_after[target_index]
    )
    runtime_represented_scale = max(
        1.0,
        abs(runtime_target),
        abs(represented_target_float),
    )
    runtime_represented_within = bool(
        math.isfinite(runtime_represented_residual)
        and math.isfinite(runtime_represented_scale)
        and runtime_represented_residual <= tol * runtime_represented_scale
    )

    hard_clip = clip_mode == "hard"
    convex_mix = 0.0 <= mix <= 1.0
    state_in_bounds = bool(np.all((state >= lower) & (state <= upper)))
    ideal_clipping_inactive = hard_clip and convex_mix and state_in_bounds
    scalar_embedding = _uses_scalar_epi_embedding(G, nodes, state)
    ideal_conditions = (
        ("fixed_support_declared", fixed_support_declared),
        ("hard_clip_mode", hard_clip),
        ("convex_mix_factor", convex_mix),
        ("state_inside_epi_bounds", state_in_bounds),
    )
    ideal_regime = all(passed for _, passed in ideal_conditions)
    runtime_conditions = ideal_conditions + (
        ("scalar_epi_embedding", scalar_embedding),
        ("runtime_hard_clipping_inactive_at_snapshot", clipping_inactive),
    )
    runtime_in_affine_domain = all(passed for _, passed in runtime_conditions)

    affine_jump: AffineEPIJumpGainCertificate | None = None
    if runtime_in_affine_domain and represented_consensus:
        candidate = certify_affine_epi_jump_gain(
            "Reception",
            represented_map,
            flow.metric_weights,
            nodes=nodes,
            tolerance=tol,
        )
        if candidate.exact_consensus_subspace_preservation:
            affine_jump = candidate
    represented_eligible = affine_jump is not None
    reasons = [name for name, passed in runtime_conditions if not passed]
    if not represented_consensus:
        reasons.append("represented_binary64_row_sum_is_not_exactly_one")
    elif runtime_in_affine_domain and affine_jump is None:
        reasons.append("affine_jump_exact_hypotheses_failed")

    lap_nodes, laplacian = structural_diffusion_operator(G)
    if tuple(lap_nodes) != nodes:
        raise RuntimeError("diffusion node order changed during Reception audit")
    try:
        with np.errstate(over="raise", invalid="raise"):
            current_pressure = -(laplacian @ state)
            post_pressure = -(laplacian @ state_after)
            pressure_defect = current_pressure - post_pressure
            pressure_defect_norm = float(
                np.max(np.abs(pressure_defect), initial=0.0)
            )
    except FloatingPointError as exc:
        raise ValueError("pure-EPI pressure diagnostic exceeds binary64 range") from exc
    pressure_detected = bool(np.any(pressure_defect != 0.0))
    pressure_refresh_required = nontrivial

    transport_neighbor_mean = float(
        state[target_index] + current_pressure[target_index]
    )

    metric_exact = tuple(
        Fraction.from_float(float(value)) for value in flow.metric_weights
    )
    exact_weighted_row = tuple(
        sum(
            (
                metric_exact[row] * ideal_map[row][column]
                for row in range(len(nodes))
            ),
            Fraction(0),
        )
        for column in range(len(nodes))
    )
    ideal_mean_defect = tuple(
        mapped - original
        for mapped, original in zip(exact_weighted_row, metric_exact)
    )
    ideal_mean_preserved = all(value == 0 for value in ideal_mean_defect)
    exact_increment = (
        Fraction.from_float(runtime_target)
        - Fraction.from_float(float(state[target_index]))
    )
    exact_mean_shift = (
        metric_exact[target_index]
        * exact_increment
        / sum(metric_exact, Fraction(0))
    )
    normalized_metric = np.asarray(flow.metric_weights, dtype=float)
    normalized_metric = normalized_metric / float(np.sum(normalized_metric))
    weighted_mean_before = float(normalized_metric @ state)
    weighted_mean_after = float(normalized_metric @ state_after)
    weighted_mean_shift = weighted_mean_after - weighted_mean_before

    break_even: float | None = None
    hybrid: HybridEPIStabilityCertificate | None = None
    hybrid_recovery: bool | None = None
    if affine_jump is not None:
        gain = affine_jump.energy_gain_bound_for_composition
        rate = flow.certified_exponential_rate_lower_bound
        # Display-only estimate: libm logarithms and division have no directed
        # rounding guarantee. The hybrid composer's exact rational/log
        # enclosure remains the sole authority for the recovery Boolean.
        if flow.is_certified and rate > 0.0:
            if gain == 0.0:
                break_even = 0.0
            elif math.isfinite(gain):
                candidate_estimate = max(0.0, math.log(gain) / rate)
                if math.isfinite(candidate_estimate):
                    break_even = candidate_estimate
        if duration is not None and flow.is_certified and rate > 0.0:
            hybrid = compose_hybrid_epi_stability(
                flow,
                [affine_jump],
                [0.0, duration],
                repeat_schedule=False,
                tolerance=tol,
            )
            hybrid_recovery = bool(
                hybrid.disagreement_contracts_over_declared_horizon
            )

    return ReceptionEPIRealizationCertificate(
        nodes=nodes,
        target=target,
        target_index=target_index,
        runtime_neighbors=runtime_neighbors,
        runtime_neighbor_indices=runtime_neighbor_indices,
        fixed_support_declared=fixed_support_declared,
        mix_factor=mix,
        exact_mix_factor=exact_mix,
        epi_lower_bound=lower,
        epi_upper_bound=upper,
        clip_mode=clip_mode,
        state_before=_readonly_array(state),
        state_after=_readonly_array(state_after),
        unweighted_runtime_neighbor_mean=neighbor_mean,
        transport_weighted_neighbor_mean=transport_neighbor_mean,
        runtime_unclipped_target_value=unclipped_target,
        runtime_target_value=runtime_target,
        runtime_target_increment=increment,
        nontrivial_runtime_reception=nontrivial,
        epi_kind_before=kind_before,
        epi_kind_after=kind_after,
        ideal_real_hard_clipping_inactive_by_convexity=(
            ideal_clipping_inactive
        ),
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
        runtime_matches_represented_affine_within_tolerance=(
            runtime_represented_within
        ),
        ideal_real_affine_conditions=ideal_conditions,
        ideal_real_affine_regime=ideal_regime,
        runtime_affine_model_domain_conditions=runtime_conditions,
        runtime_snapshot_in_affine_model_domain=runtime_in_affine_domain,
        represented_affine_jump_eligible=represented_eligible,
        affine_abstention_reasons=tuple(reasons),
        nested_affine_certificate_is_represented_model_only=True,
        global_binary64_runtime_affinity_certified=False,
        diffusion_certificate=flow,
        affine_jump_certificate=affine_jump,
        current_pure_epi_pressure=_readonly_array(current_pressure),
        post_reset_pure_epi_pressure=_readonly_array(post_pressure),
        post_reset_pressure_manifold_defect=_readonly_array(pressure_defect),
        post_reset_pressure_manifold_defect_norm=pressure_defect_norm,
        pressure_refresh_required=pressure_refresh_required,
        exact_pressure_refresh_iff_nontrivial_theorem=True,
        pressure_refresh_detected_at_snapshot=pressure_detected,
        numerical_pressure_refresh_iff_nontrivial_at_snapshot=(
            pressure_detected == nontrivial
        ),
        exact_ideal_weighted_mean_linear_defect=ideal_mean_defect,
        ideal_real_global_weighted_mean_preservation=ideal_mean_preserved,
        exact_current_weighted_mean_shift=exact_mean_shift,
        weighted_mean_before=weighted_mean_before,
        weighted_mean_after=weighted_mean_after,
        weighted_mean_shift=weighted_mean_shift,
        current_weighted_mean_preserved_exactly=exact_mean_shift == 0,
        recovery_break_even_duration_estimate=break_even,
        recovery_flow_duration=duration,
        hybrid_certificate=hybrid,
        represented_hybrid_recovery_certified=hybrid_recovery,
        tolerance=tol,
        scope=_SCOPE,
    )
