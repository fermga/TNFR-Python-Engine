"""Conditional event-aligned error after the first organized-source word.

Each ideal Reception--Coherence--Silence word acts at its own certified first
incoming energy hit. The two immediate successors are compared with their
structural timestamp difference retained. No word, trajectory, hit search or
postword continuation is executed, and supplied runtime context is not verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import bounded_represented_real, finite_represented_real
from ..config import thresholds
from ..mathematics._rational_interval import I
from ..operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from ..operators.factor_contracts import (
    canonical_glyph_factor_defaults,
    validate_glyph_factor,
)
from ._organized_source_capture import (
    _derive_organized_source_capture_barrier,
    _organized_source_offset_edge_angles,
    _OrganizedSourceCaptureBarrier,
)
from ._organized_source_child_error import (
    _bound_organized_source_child_error,
    _OrganizedSourceChildErrorBound,
)
from ._organized_source_reuse_budget import (
    _assess_organized_source_reuse_budget,
    _OrganizedSourceReuseBudget,
)

_ENERGY_CEILING = Q(1, 100)
_MEAN_CEILING = Q(1, 30)
_TARGETS = (0, 1)
# The supplied section-23 policy matches the represented default literal in
# operators.preconditions.validate_silence; it is not an inferred capacity law.
_SILENCE_PRECALL_FLOOR = 0.01


@dataclass(frozen=True)
class _OrganizedSourceChildWordErrorBound:
    """Primitive-derived inequalities under the inherited ideal-word context.

    Endpoint fields are available only if the acute/energy/mean domain, live
    numerical-policy margins and incoming first-hit certificate all pass.
    ``certified`` means this conditional mathematical implication holds; it
    authenticates neither histories, fresh stored pressure nor runtime execution.

    Form and phase endpoint bounds concern projected fine successors. Their
    maximum is ``post_primary_error_bound``; fine reconstruction additionally
    incurs the unchanged hidden radius. Because targets zero and one are
    singleton orbits, their actual fine coordinates satisfy the projected
    bounds directly. The separate timestamp bound never identifies the two
    structural times or supplies a common-time postword response.
    """

    incoming: _OrganizedSourceChildErrorBound
    barrier: _OrganizedSourceCaptureBarrier
    budget: _OrganizedSourceReuseBudget
    fine_edge_angle_bounds: tuple[I, ...]
    core_edge_angle_bounds: tuple[I, ...]
    fine_initial_mean: Q
    core_initial_mean: Q
    geometry_certified: bool
    boundary_ceiling_margin: Q
    fine_acute_margin: Q
    core_acute_margin: Q
    fine_energy_margin: Q
    core_energy_margin: Q
    mean_margin: Q
    reception_mix: Q
    phase_coefficient: Q
    pressure_retention: Q
    silence_capacity_factor: Q
    reception_form_ceiling: Q
    reception_pressure_ceiling: Q
    silence_min_capacity: Q
    form_rail_margin: Q
    reception_form_margin: Q
    reception_pressure_margin: Q
    silence_capacity_margin: Q
    domain_certified: bool
    certified: bool
    velocity_bound: Q
    timestamp_error_bound: Q | None = None
    aligned_input_error_bound: Q | None = None
    direct_hidden_phase_error_bound: Q | None = None
    post_form_error_bound: Q | None = None
    post_phase_error_bound: Q | None = None
    post_primary_error_bound: Q | None = None
    full_state_error_bound: Q | None = None
    post_energy_error_bound: Q | None = None
    successor_capacities: tuple[Q, ...] | None = None
    resultant_real_lower_bound: Q | None = None
    unavailable_reasons: tuple[str, ...] = ()
    scope: tuple[str, ...] = (
        "incoming_certificate_rebuilt_from_primitives_not_supplied_reports",
        "shared_acute_barrier_offset_edge_angles_and_one_word_rail_budget",
        "registered_default_factors_and_declared_strict_reception_policies",
        "section_23_histories_kind_ordinals_grammar_and_event_domain_are_premises",
        "pre_reception_stored_pressure_must_match_the_fresh_independent_law",
        "each_target_pair_reads_one_snapshot_per_separately_atomic_ideal_stage",
        "no_continuous_flow_between_reception_coherence_and_silence",
        "projected_successors_compared_immediately_after_their_own_first_words",
        "right_half_plane_resultant_floor_four_derived_from_the_retained_domain",
        "hidden_child_deviations_are_unchanged_by_the_marked_word",
        "successor_capacities_agree_exactly_and_structural_timestamps_stay_distinct",
        "quadratic_projected_error_requires_uniformly_positive_incoming_loss_floors",
        "no_live_admission_runtime_receipt_trajectory_or_represented_map_error_claim",
        "no_common_time_hybrid_bound_between_hits_or_postword_flow_bound",
    )


def _bound_organized_source_child_word_error(
    *, form, phase_offsets, capacities, horizon, guard_level
):
    """Certify an immediate ideal-word successor comparison from primitives.

    First rebuild the incoming first-hit bounds and independently check both
    preparations in the fixed acute cell with excess below 1/100 and absolute
    unit-capacity form mean below 1/30. The shared boundary prevents an incoming
    exit, and the existing one-word budget supplies strict numerical margins.

    Convexity of energy in this cell gives E(Pi*z)<=E(z). Section 23 then
    bounds each relevant target's relative resultant in Re(z)>4, including
    the projected fine input. An off-sector pair changes that resultant by
    at most H**2, and Arg along the right-half-plane chord changes by at most
    H**2/4. A symmetric input discrepancy D changes the five-neighbor relative
    resultant by at most 10*D. Hence the projected phase successor allowance is
    (1+5*lambda/2)*D+lambda*H**2/4. Reception is linear and nonexpansive; Silence
    preserves primary coordinates and gives equal capacities in both words.
    """
    incoming = _bound_organized_source_child_error(
        form=form,
        phase_offsets=phase_offsets,
        capacities=capacities,
        horizon=horizon,
        guard_level=guard_level,
    )
    defaults = canonical_glyph_factor_defaults()
    beta, retention, kappa = (
        Q(validate_glyph_factor(key, defaults[key]))
        for key in ("EN_mix", "IL_dnfr_factor", "SHA_vf_factor")
    )
    lam = bounded_represented_real(
        DEFAULT_PHASE_LOCKING_COEFFICIENT,
        "default phase locking coefficient",
        minimum=0,
        maximum=1,
    )[1]
    form_ceiling = finite_represented_real(
        thresholds.EPI_SATURATION_MAX, "EPI_SATURATION_MAX"
    )[1]
    pressure_ceiling = finite_represented_real(
        thresholds.DNFR_RECEPTION_MAX, "DNFR_RECEPTION_MAX"
    )[1]
    silence_floor = finite_represented_real(_SILENCE_PRECALL_FLOOR, "SHA_MIN_VF")[1]
    barrier = _derive_organized_source_capture_barrier()
    geometry = incoming.fine_initial.geometry
    geometry_ok = (
        geometry == incoming.core_initial.geometry
        and geometry == incoming.projected_state.geometry.fine_geometry
        and geometry == barrier.geometry
    )
    fine_angles = _organized_source_offset_edge_angles(geometry, incoming.phase_offsets)
    core_angles = _organized_source_offset_edge_angles(
        geometry, incoming.projected_state.phase_offsets
    )
    fine_mean = sum(incoming.form, Q(0)) / len(incoming.form)
    core_mean = sum(incoming.projected_state.form, Q(0)) / len(
        incoming.projected_state.form
    )
    mean_bound = max(abs(fine_mean), abs(core_mean))
    budget = _assess_organized_source_reuse_budget(
        reception_mix=beta,
        silence_capacity_factor=kappa,
        initial_mean_bound=mean_bound,
        energy_budget=_ENERGY_CEILING,
        word_count=1,
        initial_pair_capacity=Q(1),
    )
    boundary_margin = barrier.boundary_energy_lower_bound - _ENERGY_CEILING
    half_pi = barrier.cell_edge_gap_bounds.lo
    fine_acute = min(half_pi - angle.abs_max for angle in fine_angles)
    core_acute = min(half_pi - angle.abs_max for angle in core_angles)
    fine_energy = _ENERGY_CEILING - incoming.fine_initial_excess_bounds.hi
    core_energy = _ENERGY_CEILING - incoming.core_initial_excess_bounds.hi
    mean_margin = _MEAN_CEILING - mean_bound
    rail_margin = 1 - budget.form_absolute_upper_bound
    form_margin = form_ceiling - budget.form_absolute_upper_bound
    pressure_margin = pressure_ceiling - budget.reception_pressure_absolute_upper_bound
    capacity_margin = 1 - silence_floor
    conditions = (
        (geometry_ok, "shared_geometry_unavailable"),
        (barrier.barrier_certified, "acute_boundary_barrier_unavailable"),
        (boundary_margin > 0, "energy_ceiling_boundary_margin_unavailable"),
        (fine_acute > 0, "fine_initial_acute_cell_unavailable"),
        (core_acute > 0, "core_initial_acute_cell_unavailable"),
        (fine_energy > 0, "fine_initial_energy_ceiling_unavailable"),
        (core_energy > 0, "core_initial_energy_ceiling_unavailable"),
        (mean_margin > 0, "initial_form_mean_margin_unavailable"),
        (
            budget.unit_form_rails_certified and rail_margin > 0,
            "unit_form_rail_margin_unavailable",
        ),
        (form_margin > 0, "reception_form_ceiling_margin_unavailable"),
        (pressure_margin > 0, "reception_pressure_ceiling_margin_unavailable"),
        (capacity_margin >= 0, "silence_precall_capacity_floor_unavailable"),
    )
    reasons = tuple(reason for passed, reason in conditions if not passed)
    domain_ok = not reasons
    if not incoming.crossing_certified:
        reasons += ("incoming_crossing_unavailable",)
    certified = not reasons
    g, x = incoming.gamma_upper, incoming.form_radius_bound
    velocity = max((10 * x + 5 * g) / 4, 5 * g * x / 2)
    endpoint = {}
    if certified:
        delta = incoming.crossing_time_error_bound
        aligned = incoming.projected_state_error_bound + velocity * delta
        hidden = incoming.hidden_radius_bound
        direct_phase = lam * hidden**2 / 4
        phase_error = (1 + 5 * lam / 2) * aligned + direct_phase
        primary_error = max(aligned, phase_error)
        endpoint = dict(
            timestamp_error_bound=delta,
            aligned_input_error_bound=aligned,
            direct_hidden_phase_error_bound=direct_phase,
            post_form_error_bound=aligned,
            post_phase_error_bound=phase_error,
            post_primary_error_bound=primary_error,
            full_state_error_bound=hidden + primary_error,
            post_energy_error_bound=108 * hidden**2
            + 360 * x * aligned
            + 180 * phase_error,
            successor_capacities=tuple(
                kappa if i in _TARGETS else Q(1) for i in range(45)
            ),
            resultant_real_lower_bound=Q(4),
        )
    return _OrganizedSourceChildWordErrorBound(
        incoming=incoming,
        barrier=barrier,
        budget=budget,
        fine_edge_angle_bounds=fine_angles,
        core_edge_angle_bounds=core_angles,
        fine_initial_mean=fine_mean,
        core_initial_mean=core_mean,
        geometry_certified=geometry_ok,
        boundary_ceiling_margin=boundary_margin,
        fine_acute_margin=fine_acute,
        core_acute_margin=core_acute,
        fine_energy_margin=fine_energy,
        core_energy_margin=core_energy,
        mean_margin=mean_margin,
        reception_mix=beta,
        phase_coefficient=lam,
        pressure_retention=retention,
        silence_capacity_factor=kappa,
        reception_form_ceiling=form_ceiling,
        reception_pressure_ceiling=pressure_ceiling,
        silence_min_capacity=silence_floor,
        form_rail_margin=rail_margin,
        reception_form_margin=form_margin,
        reception_pressure_margin=pressure_margin,
        silence_capacity_margin=capacity_margin,
        domain_certified=domain_ok,
        certified=certified,
        velocity_bound=velocity,
        unavailable_reasons=reasons,
        **endpoint,
    )
