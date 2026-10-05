"""Detached cycle geometry, capture, energy obstructions and bridge removal.

The sector geometry observer reads supplied support, form, phase and storage
scale without a rate evaluation or dynamical admission. Conditional capture
owners additionally use fresh native fields to admit reflected protected
regions, a full-state local neighborhood, a joined acute winding sector or
one C5. The detachment observer evaluates the supplied joined state and its
components separately, with the shared reset owner retaining event work.
Their shared exact geometry does not project states, integrate a model,
certify binary64 steps or substitute winding telemetry for proof.
Static seeded-formation obstructions use that geometry without evaluating a field.
Energy enclosures remain separate from production storage arithmetic.
Capture concerns ideal continuation from the supplied snapshot, independently
of earlier integration error or a frozen experimental verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import TYPE_CHECKING, Any

import networkx as nx

from ..constants.aliases import ALIAS_EPI, ALIAS_THETA
from ..dynamics.relational import (
    RelationalExchangeField,
    RelationalExchangeModel,
    _epi,
    _finite,
    _raw,
    evaluate_relational_exchange,
)
from ..mathematics._phase_midpoint import _affine_interval, _pi_bounds
from ..mathematics._phase_resultant_chamber import (
    COSINE_ENCLOSURE_METHOD,
    certified_cosine_bounds,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
from .relational_observations import _detached_graph, _ordered
from .winding_certificates import WindingCertificate, certify_phase_winding

if TYPE_CHECKING:
    from .relational_observations import RelationalResetObservation

__all__ = (
    "RelationalCaptureCertificate",
    "RelationalConsensusCaptureCertificate",
    "RelationalConsensusFormationObstruction",
    "RelationalLocalCaptureCertificate",
    "RelationalSectorCaptureCertificate",
    "RelationalSectorGeometry",
    "RelationalCycleCaptureCertificate",
    "RelationalDetachmentObservation",
    "RelationalSeededFormationCase",
    "RelationalSeededFormationObstruction",
    "certify_relational_capture",
    "certify_relational_consensus_capture",
    "certify_relational_consensus_formation_obstruction",
    "certify_relational_local_capture",
    "certify_relational_sector_capture",
    "observe_relational_sector_geometry",
    "certify_relational_cycle_capture",
    "observe_relational_detachment",
    "certify_relational_seeded_formation_obstruction",
)


# Each margin is sign * (a or b) + coefficient * mathematical pi. The point
# and validated-box owners evaluate this one ledger with their own arithmetic.
_CAPTURE_RECTANGLES = (
    (
        1,
        "positive_winding",
        ((0, 1, -Q(2, 3)), (0, -1, Q(1)), (1, 1, Q(0)), (1, -1, Q(1, 2))),
    ),
    (
        0,
        "consensus",
        ((0, -1, Q(2, 3)), (0, 1, Q(2, 3)), (1, -1, Q(1, 2)), (1, 1, Q(1, 2))),
    ),
    (
        -1,
        "negative_winding",
        ((0, -1, -Q(2, 3)), (0, 1, Q(1)), (1, -1, Q(0)), (1, 1, Q(1, 2))),
    ),
)


def _capture_rectangle_candidates(a, b, affine):
    """Evaluate all strict basin margins without choosing a target sector."""
    coordinates = (a, b)
    return tuple(
        (
            sector,
            tuple(
                affine(sign * coordinates[index], coefficient)
                for index, sign, coefficient in margins
            ),
        )
        for sector, _, margins in _CAPTURE_RECTANGLES
    )


def _capture_rectangle_kind(sector):
    return next(
        (kind for value, kind, _ in _CAPTURE_RECTANGLES if value == sector), None
    )


@dataclass(frozen=True)
class RelationalCaptureCertificate:
    """Detached admission of the reflected ideal-law capture theorem.

    Coordinates ``(A,B,a,b)`` are read from the first supplied cycle relative
    to its node-three offsets. They describe the entire state only when all
    copy/reflection defects vanish exactly. Candidate rectangle margins use
    the order R+, S, R- with sector labels +1, 0, -1. R+ margins enclose
    ``a-2*pi/3``, ``pi-a``, ``b`` and ``pi/2-b``; R- substitutes ``(-a,-b)``.
    S margins enclose ``2*pi/3-a``, ``2*pi/3+a``, ``pi/2-b``, ``pi/2+b``.
    Selected margins are empty when no rectangle is certified. Both energy
    bounds use mathematical cosine at exact captured raw radians; the field's
    separately retained storage uses production floating arithmetic.
    Copy defects follow the five supplied positions; reflection defects
    concatenate the two supplied cycle orders. The field keeps graph order.

    ``admitted`` concerns the continuous conditional law from this precise
    represented state. Unavailable means this sufficient theorem was not
    admitted, not that the state cannot recover. ``target_sector`` exists
    only on full admission and describes the ideal limiting equilibrium.
    Numerical winding reports describe the current state independently; they
    need not equal that target or decide the exact rectangle/energy tests.
    """

    field: RelationalExchangeField
    cycles: tuple[tuple[Any, ...], tuple[Any, ...]]
    form_offset: Q
    phase_center: Q
    coordinates: tuple[Q, Q, Q, Q]
    copy_form_defects: tuple[Q, ...]
    copy_phase_defects: tuple[Q, ...]
    reflection_form_defects: tuple[Q, ...]
    reflection_phase_defects: tuple[Q, ...]
    exact_symmetry: bool
    unit_capacity: bool
    positive_epi_weight: bool
    candidate_rectangle_margin_bounds: tuple[tuple[int, tuple[tuple[Q, Q], ...]], ...]
    rectangle_kind: str | None
    rectangle_margin_bounds: tuple[tuple[Q, Q], ...]
    rectangle_admitted: bool
    phase_storage_bounds: tuple[Q, Q]
    storage_bounds: tuple[Q, Q]
    capture_threshold: Q
    energy_admitted: bool
    winding: tuple[WindingCertificate, WindingCertificate]
    target_sector: int | None
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = COSINE_ENCLOSURE_METHOD
    scope: tuple[str, ...] = (
        "one_fresh_detached_native_relational_field_without_live_writes",
        "supplied_two_ordered_disjoint_C5_rings_with_two_matching_adjacent_port_bridges",
        "exact_copied_and_reflected_materialized_state_with_common_form_and_phase_offsets",
        "unit_held_capacity_and_positive_epi_phase_and_storage_coefficients",
        "three_disjoint_strict_rectangles_Rplus_S_Rminus_with_exact_pi_margin_enclosures",
        "exact_form_storage_plus_certified_mathematical_cosine_phase_storage_below_7beta",
        "conditional_ideal_ODE_convergence_to_uniform_form_and_selected_phase_target_sector",
        "observed_snapshot_winding_is_distinct_from_the_prospective_ideal_target_sector",
        "no_projection_finite_solver_error_bound_future_runtime_admission_or_formation_execution",
        "no_autonomous_substrate_creation_or_physical_identity_claim",
    )

    @property
    def admitted(self) -> bool:
        """Whether every premise of the sufficient capture theorem resolved."""
        return self.status == "admitted"


def _cycles(field, cycles):
    return _cycle_support(field.nodes, field.edges, cycles)


def _cycle_support(nodes, edges, cycles):
    rows = tuple(_ordered(row, "cycle") for row in _ordered(cycles, "cycles"))
    if len(rows) != 2 or any(len(row) != 5 for row in rows):
        raise ValueError("cycles must contain exactly two ordered five-node rings")
    lookup = {node: index for index, node in enumerate(nodes)}
    try:
        indices = tuple(tuple(lookup[node] for node in row) for row in rows)
    except (TypeError, KeyError) as exc:
        raise ValueError("cycle nodes must belong to the captured support") from exc
    flat = tuple(i for row in indices for i in row)
    if len(nodes) != 10 or len(set(flat)) != 10:
        raise ValueError("the two cycles must cover the full support exactly once")
    expected = {
        frozenset((left, right))
        for row in indices
        for left, right in zip(row, row[1:] + row[:1])
    }
    expected.update(frozenset((indices[0][i], indices[1][i])) for i in (0, 1))
    actual = {frozenset((lookup[a], lookup[b])) for a, b in edges}
    if actual != expected:
        raise ValueError(
            "support must be the two supplied cycles and bridges at matching positions 0 and 1"
        )
    return rows, indices


def _phase_storage_bounds(field):
    return _phase_storage_bounds_at(field.nodes, field.edges, field.phase)


def _phase_storage_bounds_at(nodes, edges, phases):
    phase = tuple(map(Q, phases))
    position = {node: i for i, node in enumerate(nodes)}
    bounds = tuple(
        certified_cosine_bounds(phase[position[j]] - phase[position[i]])
        for i, j in edges
    )
    return (
        sum((1 - upper for _, upper in bounds), Q(0)),
        sum((1 - lower for lower, _ in bounds), Q(0)),
    )


def certify_relational_capture(
    graph, *, model: RelationalExchangeModel, cycles
) -> RelationalCaptureCertificate:
    """Certify sufficient capture on one supplied exact reflected two-C5 state.

    The model's existing graph, scalar and phase-domain admission runs first.
    ``cycles`` gives all nodes once, in two oriented five-node rings; the only
    additional edges must join matching positions zero and one. No topology,
    partition, phase lift or reflection is inferred or repaired.

    In each ring the theorem requires ``x=m+(A,-A,-B,0,B)`` and
    ``theta=c+(a,-a,-b,0,b)`` with identical copies, unit capacities and
    positive coefficients. Strict rational enclosures must place ``(a,b)``
    inside R+ = ``(2*pi/3,pi) x (0,pi/2)``, its sign reversal R-, or
    S = ``(-2*pi/3,2*pi/3) x (-pi/2,pi/2)``, and full energy below ``7*beta``.
    The admitted ideal trajectory stays in its compact regular sublevel and
    converges to ``A=B=0`` and phase coordinates ``(4*pi/5,2*pi/5)``, their
    negatives, or ``(0,0)``, respectively, preserving the common offsets.
    The target classification is derived from sufficient basin premises,
    not from current winding and not used to control the dynamics.

    Malformed inputs or unsupported engine domains raise their usual errors.
    A well-formed state outside these sufficient premises returns unavailable
    with evidence. Small nonzero symmetry defects are never tolerated away.
    """
    field = evaluate_relational_exchange(graph, model=model)
    cycles, indices = _cycles(field, cycles)
    form, phase = tuple(map(Q, field.epi)), tuple(map(Q, field.phase))
    left, right = indices
    form_offset, phase_center = form[left[3]], phase[left[3]]
    coordinates = (
        form[left[0]] - form_offset,
        form[left[4]] - form_offset,
        phase[left[0]] - phase_center,
        phase[left[4]] - phase_center,
    )
    copy_form = tuple(form[j] - form[i] for i, j in zip(left, right))
    copy_phase = tuple(phase[j] - phase[i] for i, j in zip(left, right))
    reflected_positions = (1, 0, 4, 3, 2)
    reflection_form = tuple(
        form[row[i]] + form[row[j]] - 2 * form_offset
        for row in indices
        for i, j in enumerate(reflected_positions)
    )
    reflection_phase = tuple(
        phase[row[i]] + phase[row[j]] - 2 * phase_center
        for row in indices
        for i, j in enumerate(reflected_positions)
    )
    symmetry = not any((*copy_form, *copy_phase, *reflection_form, *reflection_phase))
    unit_capacity = all(value == 1.0 for value in field.capacity)
    positive_epi = field.model.epi_weight > 0.0
    a, b = coordinates[2:]
    pi_bounds = _pi_bounds()
    candidate_bounds = _capture_rectangle_candidates(
        a,
        b,
        lambda rational, coefficient: _affine_interval(
            rational, coefficient, pi_bounds
        ),
    )
    selected = tuple(
        (sector, bounds)
        for sector, bounds in candidate_bounds
        if all(lower > 0 for lower, _ in bounds)
    )
    if len(selected) > 1:
        raise ArithmeticError("disjoint capture rectangles cannot both be admitted")
    rectangle_sector, rectangle_bounds = selected[0] if selected else (None, ())
    rectangle = bool(selected)
    rectangle_kind = _capture_rectangle_kind(rectangle_sector)
    phase_bounds = _phase_storage_bounds(field)
    beta = Q(field.model.storage_scale)
    energy_bounds = tuple(field.form_storage + beta * value for value in phase_bounds)
    threshold = 7 * beta
    energy = energy_bounds[1] < threshold
    reasons = tuple(
        reason
        for admitted, reason in (
            (symmetry, "exact_copy_reflection_not_satisfied"),
            (unit_capacity, "unit_capacity_required"),
            (positive_epi, "positive_epi_weight_required"),
            (rectangle, "strict_capture_rectangle_not_certified"),
            (energy, "strict_storage_sublevel_not_certified"),
        )
        if not admitted
    )
    detached = _detached_graph(field)
    return RelationalCaptureCertificate(
        field=field,
        cycles=cycles,
        form_offset=form_offset,
        phase_center=phase_center,
        coordinates=coordinates,
        copy_form_defects=copy_form,
        copy_phase_defects=copy_phase,
        reflection_form_defects=reflection_form,
        reflection_phase_defects=reflection_phase,
        exact_symmetry=symmetry,
        unit_capacity=unit_capacity,
        positive_epi_weight=positive_epi,
        candidate_rectangle_margin_bounds=candidate_bounds,
        rectangle_kind=rectangle_kind,
        rectangle_margin_bounds=rectangle_bounds,
        rectangle_admitted=rectangle,
        phase_storage_bounds=phase_bounds,
        storage_bounds=energy_bounds,
        capture_threshold=threshold,
        energy_admitted=energy,
        winding=tuple(certify_phase_winding(detached, row) for row in cycles),
        target_sector=rectangle_sector if not reasons else None,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class RelationalConsensusCaptureCertificate:
    """Analytic consensus capture from a bounded reflected form preparation.

    The declared theorem has beta=nu=1, e=w=1/2, exact initial phase
    consensus and copied reflected form on the supplied two-bridge support.
    Its fixed form-storage ceiling is 9. It bounds the full ideal trajectory
    through structural time one, then applies the existing strict consensus
    capture theorem. No numerical endpoint or integrated response is supplied.

    Dynamic estimates remain unavailable when any premise fails. The initial
    field and form budget remain observable. Native domain admission concerns
    that captured source; a theorem about ideal continuation does not promise
    admission of future finite steps under a configured execution guard.
    """

    initial: RelationalCaptureCertificate
    initial_form_storage: Q
    form_storage_ceiling: Q
    form_storage_margin: Q
    exact_phase_consensus: bool
    fixed_coefficients_admitted: bool
    unit_storage_scale: bool
    bootstrap_time: Q
    bootstrap_phase_absolute_upper_bounds: tuple[Q, Q] | None
    bootstrap_phase_radius_margins: tuple[Q, Q] | None
    endpoint_phase_storage_upper_bound: Q | None
    endpoint_storage_upper_bound: Q | None
    endpoint_consensus_rectangle_margin_bounds: tuple[tuple[Q, Q], ...] | None
    capture_margin: Q | None
    target_sector: int | None
    status: str
    unavailable_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "single_existing_native_capture_and_exact_copied_reflected_two_bridge_admission",
        "exact_initial_raw_phase_consensus_with_supplied_common_offsets_retained",
        "unit_held_capacity_unit_storage_scale_and_normalized_e_w_equal_one_half",
        "form_storage_at_most_nine_is_a_conditional_theorem_not_a_fitted_threshold",
        "whole_initial_unit_time_bootstrap_keeps_both_phase_coordinates_below_one_half",
        "initial_storage_equals_form_storage_only_under_admitted_phase_consensus",
        "strict_loss_and_phase_potential_bound_give_endpoint_storage_at_most_two_F_over_three",
        "endpoint_box_enters_existing_consensus_rectangle_below_its_strict_storage_barrier",
        "ideal_continuous_capture_to_consensus_excludes_maintained_unit_winding_in_this_class",
        "failed_sufficient_admission_does_not_certify_formation_or_instability",
        "no_integrated_endpoint_response_solver_or_modified_existing_strict_capture_semantics",
        "no_future_finite_step_guard_guarantee_support_selection_or_physical_identification",
    )

    @property
    def admitted(self):
        return self.status == "admitted"


def certify_relational_consensus_capture(
    graph, *, model: RelationalExchangeModel, cycles
) -> RelationalConsensusCaptureCertificate:
    """Certify a finite-time entry bound and subsequent ideal consensus capture.

    Admit the actual represented source through ``certify_relational_capture``
    once. For its exact copied reflected coordinates (A,B,a,b), the new theorem
    requires a=b=0 and F=6*A**2-4*A*B+4*B**2<=9 at beta=nu=1,e=w=1/2.
    Common form and phase offsets do not affect this conditional result.

    On 0<=t<=1 the proof gives |a|<=sqrt(3F/2)/(3*pi) and
    |b|<=sqrt(F)/(2*pi), both strictly below 1/2. At t=1 the total
    storage is at most 2F/3<=6<7 in the consensus rectangle. These
    are analytic bounds on ideal continuous evolution, not evaluated samples.
    The older initial E<7 capture certificate remains unchanged.
    """
    initial = certify_relational_capture(graph, model=model, cycles=cycles)
    field = initial.field
    form = field.form_storage
    ceiling = Q(9)
    consensus = len(set(map(Q, field.phase))) == 1
    coefficients = tuple(map(Q, field.model.effective_weights)) == (Q(1, 2), Q(1, 2))
    unit_beta = Q(field.model.storage_scale) == 1
    reasons = tuple(
        reason
        for condition, reason in (
            (initial.exact_symmetry, "exact_copy_reflection_required"),
            (consensus, "exact_initial_phase_consensus_required"),
            (initial.unit_capacity, "unit_held_capacity_required"),
            (coefficients, "normalized_epi_and_phase_weights_one_half_required"),
            (unit_beta, "unit_storage_scale_required"),
            (form <= ceiling, "initial_form_storage_at_most_nine_required"),
        )
        if not condition
    )
    phase_bounds = phase_margins = rectangle = endpoint_storage = capture_margin = None
    if not reasons:
        pi = pi_interval()
        phase_bounds = (
            (sqrt(I(Q(3, 2) * form)) / (3 * pi)).hi,
            (sqrt(I(form)) / (2 * pi)).hi,
        )
        phase_margins = tuple(Q(1, 2) - value for value in phase_bounds)
        candidates = _capture_rectangle_candidates(
            I(-phase_bounds[0], phase_bounds[0]),
            I(-phase_bounds[1], phase_bounds[1]),
            lambda coordinate, coefficient: coordinate + coefficient * pi,
        )
        rectangle = tuple(
            (value.lo, value.hi)
            for sector, margins in candidates
            if sector == 0
            for value in margins
        )
        endpoint_storage = Q(2, 3) * form
        capture_margin = initial.capture_threshold - endpoint_storage
        if (
            min(phase_margins) <= 0
            or min(value[0] for value in rectangle) <= 0
            or capture_margin <= 0
        ):
            raise ArithmeticError(
                "consensus bootstrap constants failed their strict theorem margins"
            )
    return RelationalConsensusCaptureCertificate(
        initial=initial,
        initial_form_storage=form,
        form_storage_ceiling=ceiling,
        form_storage_margin=ceiling - form,
        exact_phase_consensus=consensus,
        fixed_coefficients_admitted=coefficients,
        unit_storage_scale=unit_beta,
        bootstrap_time=Q(1),
        bootstrap_phase_absolute_upper_bounds=phase_bounds,
        bootstrap_phase_radius_margins=phase_margins,
        endpoint_phase_storage_upper_bound=Q(1, 4) * form if not reasons else None,
        endpoint_storage_upper_bound=endpoint_storage,
        endpoint_consensus_rectangle_margin_bounds=rectangle,
        capture_margin=capture_margin,
        target_sector=None if reasons else 0,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class RelationalConsensusFormationObstruction:
    """All-form exclusion of the aligned unit-winding target from consensus.

    At the admitted source, phase storage is zero and full form storage is
    at most nine. The native half-weight law keeps phase storage below the
    aligned target's cost on every regular existence interval. Arbitrary
    represented form is retained; no copy or reflection premise is imposed.
    The bound does not certify global regular continuation or convergence
    to consensus. Failed premises leave all prospective bounds unavailable.
    """

    field: RelationalExchangeField
    cycles: tuple[tuple[Any, ...], tuple[Any, ...]]
    initial_form_storage: Q
    form_storage_ceiling: Q
    form_storage_margin: Q
    exact_phase_consensus: bool
    unit_capacity: bool
    fixed_coefficients_admitted: bool
    unit_storage_scale: bool
    normalized_gap_lower_bound: Q | None
    all_regular_time_phase_storage_upper_bound: Q | None
    target_phase_storage_bounds: tuple[Q, Q] | None
    exclusion_margin: Q | None
    excluded_target_sectors: tuple[int, int] | None
    status: str
    unavailable_reasons: tuple[str, ...]
    continuation_status: str = "not_certified"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "one_fresh_detached_native_field_and_exact_two_C5_matching_adjacent_bridge_support",
        "arbitrary_full_form_without_copy_reflection_or_projection",
        "exact_initial_raw_phase_consensus_unit_held_capacity_and_storage_scale",
        "normalized_epi_phase_weights_one_half_no_input_events_or_coefficient_change",
        "full_support_form_storage_at_most_nine_including_all_bridge_costs",
        "native_argument_pressure_work_and_full_graph_gap_bound_control_phase_storage",
        "all_regular_existence_times_phase_storage_at_most_seven_initial_F_over_ten",
        "aligned_acute_unit_winding_targets_of_either_orientation_are_excluded",
        "zero_form_source_included_without_division_by_form_or_phase_storage",
        "no_global_continuation_consensus_limit_transient_winding_or_new_target_claim",
        "no_solver_response_replay_finite_step_guarantee_operator_event_or_physical_identification",
    )

    @property
    def admitted(self):
        return self.status == "admitted"


def certify_relational_consensus_formation_obstruction(
    graph, *, model: RelationalExchangeModel, cycles
) -> RelationalConsensusFormationObstruction:
    """Exclude maintained aligned unit winding for the full bounded form class.

    Reuse native source admission and the actual two-ring support. With exact
    common raw phase, beta=nu=1 and normalized e=w=1/2, the analytic theorem
    gives P(t)<=7F(0)/10 for every regular existence time when F(0)<=9.
    This excludes entry into either acute unit-winding target component and
    convergence to its aligned target. It does not prove that the native
    solution exists forever or converges to consensus. No reflected source
    coordinates, trajectory integration or numerical endpoint are used.
    """
    field = evaluate_relational_exchange(graph, model=model)
    cycles, _ = _cycles(field, cycles)
    form, ceiling = field.form_storage, Q(9)
    consensus = len(set(map(Q, field.phase))) == 1
    unit_capacity = all(Q(value) == 1 for value in field.capacity)
    coefficients = tuple(map(Q, field.model.effective_weights)) == (Q(1, 2), Q(1, 2))
    unit_beta = Q(field.model.storage_scale) == 1
    reasons = tuple(
        reason
        for condition, reason in (
            (consensus, "exact_initial_phase_consensus_required"),
            (unit_capacity, "unit_held_capacity_required"),
            (coefficients, "normalized_epi_and_phase_weights_one_half_required"),
            (unit_beta, "unit_storage_scale_required"),
            (form <= ceiling, "initial_form_storage_at_most_nine_required"),
        )
        if not condition
    )
    phase_bound = target_bounds = margin = None
    if not reasons:
        phase_bound = Q(7, 10) * form
        target_bounds = _twist_storage_bounds(_pi_bounds())
        margin = target_bounds[0] - phase_bound
        if margin <= 0:
            raise ArithmeticError(
                "consensus formation constants failed their target separation"
            )
    return RelationalConsensusFormationObstruction(
        field=field,
        cycles=cycles,
        initial_form_storage=form,
        form_storage_ceiling=ceiling,
        form_storage_margin=ceiling - form,
        exact_phase_consensus=consensus,
        unit_capacity=unit_capacity,
        fixed_coefficients_admitted=coefficients,
        unit_storage_scale=unit_beta,
        normalized_gap_lower_bound=Q(1, 9) if not reasons else None,
        all_regular_time_phase_storage_upper_bound=phase_bound,
        target_phase_storage_bounds=target_bounds,
        exclusion_margin=margin,
        excluded_target_sectors=(-1, 1) if not reasons else None,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class RelationalLocalCaptureCertificate:
    """Sufficient full-state local basin around one declared phase target.

    Form and phase means are exact means of all captured represented nodes.
    ``phase_error_affine`` follows field order and stores each centered
    phase error as ``rational + coefficient*pi``. The target's coefficients
    sum to zero, so no estimated equilibrium, coordinate projection, or
    exact reflection is required. All twenty form/phase coordinates enter
    the joint squared-norm bounds in the declared beta-one normalization.

    Admission certifies the ideal continuous law restarted at this precise
    snapshot. It does not certify the path that produced the snapshot, an
    earlier ODE's numerical error, or future binary64 steps. A negative lower
    excess-energy bound is retained: enclosing zero is not a failed test.
    Local convexity establishes exact nonnegative excess on the admitted ball.
    """

    field: RelationalExchangeField
    cycles: tuple[tuple[Any, ...], tuple[Any, ...]]
    declared_target_sector: int
    reference_phase_pi_coefficients: tuple[Q, ...]
    form_mean: Q
    phase_mean: Q
    centered_form: tuple[Q, ...]
    phase_error_affine: tuple[tuple[Q, Q], ...]
    phase_error_bounds: tuple[tuple[Q, Q], ...]
    form_norm_squared: Q
    phase_norm_squared_bounds: tuple[Q, Q]
    quotient_squared_bounds: tuple[Q, Q]
    squared_radius_threshold: Q
    radius_admitted: bool
    phase_storage_bounds: tuple[Q, Q]
    storage_bounds: tuple[Q, Q]
    reference_phase_storage_bounds: tuple[Q, Q]
    excess_storage_bounds: tuple[Q, Q]
    excess_storage_threshold: Q
    energy_admitted: bool
    unit_capacity: bool
    unit_storage_scale: bool
    positive_epi_weight: bool
    winding: tuple[WindingCertificate, WindingCertificate]
    target_sector: int | None
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = COSINE_ENCLOSURE_METHOD
    scope: tuple[str, ...] = (
        "one_fresh_detached_native_relational_field_without_live_writes",
        "same_two_C5_two_adjacent_bridge_support_with_unit_capacity_and_storage_scale",
        "declared_target_sector_minus_one_zero_or_plus_one_with_exact_pi_affine_reference",
        "all_full_state_centered_coordinates_without_symmetry_projection_or_phase_rewrapping",
        "strict_joint_norm_squared_upper_below_9_over800_and_excess_upper_below_1_over100000",
        "conditional_ideal_ODE_recovery_from_this_exact_represented_snapshot_modulo_common_offsets",
        "no_error_bound_for_an_earlier_ODE_trajectory_or_future_binary64_execution",
        "no_observed_formation_substrate_creation_or_physical_identity_claim",
    )

    @property
    def admitted(self) -> bool:
        """Whether all sufficient local-basin premises are certified."""
        return self.status == "admitted"


def _interval_square(interval):
    lower, upper = interval
    return (
        Q(0) if lower <= 0 <= upper else min(lower * lower, upper * upper),
        max(lower * lower, upper * upper),
    )


def _pi_cosine_bounds(coefficient, pi_bounds):
    # Enclose an ideal mathematical angle before applying the shared
    # cosine kernel. Its midpoint is rational; the exact radius follows from
    # the common pi enclosure and cosine's unit Lipschitz constant.
    lower, upper = _affine_interval(Q(0), coefficient, pi_bounds)
    midpoint, radius = (lower + upper) / 2, (upper - lower) / 2
    cosine_lower, cosine_upper = certified_cosine_bounds(midpoint)
    cosine_lower = max(Q(-1), cosine_lower - radius)
    cosine_upper = min(Q(1), cosine_upper + radius)
    return cosine_lower, cosine_upper


def _twist_storage_bounds(pi_bounds):
    cosine_lower, cosine_upper = _pi_cosine_bounds(Q(2, 5), pi_bounds)
    # The aligned bridges have zero cost and both rings have five twist edges.
    return 10 * (1 - cosine_upper), 10 * (1 - cosine_lower)


def _cycle_barrier_constants(pi_bounds):
    """C5 acute winding-one face cost, independently of a chosen law.

    One edge reaches pi/2; the other four sum to 3*pi/2. Convexity of
    1-cos on the closed acute chart gives 5-4*cos(3*pi/8). Return the
    original cosine enclosure as well as the resulting unscaled cost.
    """
    cosine = _pi_cosine_bounds(Q(3, 8), pi_bounds)
    return cosine, (5 - 4 * cosine[1], 5 - 4 * cosine[0])


@dataclass(frozen=True)
class RelationalSeededFormationCase:
    """One supplied support's uniform initial-energy obstruction.

    Matching ports index the ordered source and receiver cycles separately.
    The strict initial storage bound includes all supplied bridge costs.
    Its positive deficit from the two-pattern target is only a necessary
    additional storage requirement, not sufficient work, an available
    reservoir or a guarantee of formation after changing the preparation.
    """

    bridge_count: int
    matching_ports: tuple[tuple[int, int], ...]
    bridge_storage_upper_bound: Q
    initial_storage_upper_bound: Q
    initial_storage_bound_is_strict: bool
    additional_storage_gap_lower_bound: Q
    obstruction_certified: bool
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RelationalSeededFormationObstruction:
    """Conditional exclusion of a particular two-pattern formation target.

    The source is an exact winding+1 C5 twist; the receiver is a uniform-phase
    C5. Both have the same constant form and beta=capacity=1, with e=w=1/2.
    One or two matching bridges are supplied. Every relative phase admitted
    by the initial positive-real-resultant chamber obeys the energy ceiling.

    The excluded target has BOTH rings in acute winding+1 sectors, or both
    recovered winding+1 twists, under subsequent unforced fixed-support
    native evolution. Winding alone on a nonacute state is not this target.
    The report does not exclude receiver formation accompanied by source
    loss, other transient organization or preparations with form contrast.
    """

    pi_bounds: tuple[Q, Q]
    twist_cosine_bounds: tuple[Q, Q]
    source_storage_bounds: tuple[Q, Q]
    target_minimum_storage_bounds: tuple[Q, Q]
    cases: tuple[RelationalSeededFormationCase, RelationalSeededFormationCase]
    obstruction_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = COSINE_ENCLOSURE_METHOD
    scope: tuple[str, ...] = (
        "exact_winding_plus_one_C5_source_and_uniform_phase_C5_receiver",
        "same_constant_form_unit_capacity_beta_one_equal_half_pressure_weights",
        "one_or_two_supplied_matching_bridges_with_their_initial_storage_included",
        "uniform_over_all_relative_phases_in_initial_positive_real_resultant_chamber",
        "acute_initial_bridge_admission_is_a_stronger_subset",
        "initial_positive_resultant_premise_not_the_full_regular_phase_domain",
        "target_is_both_acute_winding_plus_one_sectors_or_recovered_twists_not_winding_alone",
        "native_unforced_fixed_support_continuation_has_nonincreasing_storage",
        "additional_storage_deficit_is_necessary_not_a_sufficient_formation_condition",
        "no_graph_admission_field_evaluation_trajectory_event_or_physical_claim",
    )


def certify_relational_seeded_formation_obstruction() -> (
    RelationalSeededFormationObstruction
):
    """Exclude the fixed equal-form seed's two-pattern target by storage.

    Write c=cos(2*pi/5), V*=5*(1-c). Each source port has initial relative
    resultant real part ``2*c+cos(delta_j)``. Strict positivity implies
    bridge cost ``1-cos(delta_j) < 1+2*c``. Receiver real parts are
    ``2+cos(delta_j)>=1``; nonport source real parts are ``2*c>0``.
    Thus with k in {1,2}, initial storage is strictly below
    ``V*+k*(1+2*c)``, whereas the declared target requires at least ``2*V*``.
    The difference ``5-k-(5+2*k)*c`` is strictly positive for both supports.

    All constants reuse the shared exact pi/cosine geometry. Outward upper
    initial bounds and the lower target bound certify the strict deficit;
    no relative phase is sampled or fitted. For acute initial bridges the
    stronger cost bound is k, but the report covers the larger positive-real
    initial chamber. It does not transfer to the entire regular phase domain:
    a two-bridge initial state can have negative source resultant real parts
    with nonzero imaginary parts and more bridge storage than this ceiling.
    Nor does it certify global existence of a continuation. Whenever the
    unforced native continuation exists, its storage cannot reach the target.
    """
    pi_bounds = _pi_bounds()
    cosine = _pi_cosine_bounds(Q(2, 5), pi_bounds)
    target = _twist_storage_bounds(pi_bounds)
    source = tuple(value / 2 for value in target)
    cases = []
    for count in (1, 2):
        bridge_upper = count * (1 + 2 * cosine[1])
        initial_upper = source[1] + bridge_upper
        gap = target[0] - initial_upper
        obstruction = gap > 0
        reasons = (
            ()
            if obstruction
            else ("strict_initial_to_target_storage_gap_not_certified",)
        )
        cases.append(
            RelationalSeededFormationCase(
                bridge_count=count,
                matching_ports=tuple((index, index) for index in range(count)),
                bridge_storage_upper_bound=bridge_upper,
                initial_storage_upper_bound=initial_upper,
                initial_storage_bound_is_strict=True,
                additional_storage_gap_lower_bound=gap,
                obstruction_certified=obstruction,
                unavailable_reasons=reasons,
            )
        )
    obstruction = all(case.obstruction_certified for case in cases)
    reasons = (
        () if obstruction else ("both_supplied_support_obstructions_not_certified",)
    )
    return RelationalSeededFormationObstruction(
        pi_bounds=pi_bounds,
        twist_cosine_bounds=cosine,
        source_storage_bounds=source,
        target_minimum_storage_bounds=target,
        cases=tuple(cases),
        obstruction_certified=obstruction,
        status="obstructed" if obstruction else "unavailable",
        unavailable_reasons=reasons,
    )


def certify_relational_local_capture(
    graph, *, model: RelationalExchangeModel, cycles, target_sector: int = 1
) -> RelationalLocalCaptureCertificate:
    """Check the full-state local basin of a supplied two-ring target.

    The declared sector is -1, 0 or +1, excluding Booleans. In each supplied
    cycle the ideal reference is ``sector*pi*(4/5,-4/5,-2/5,0,2/5)``.
    Real phase lifts are retained; no per-node full turns are silently added
    to improve the distance. Both rings share one common phase/form quotient.

    With beta=nu=1 and e,w>0, the section-15 energy argument admits radius
    ``r=pi/(20*sqrt(2))`` for all three targets. The fixed support has
    ``lambda_2>1/45`` and the ball has cosine margin greater than 1/10.
    Consequently its energy barrier exceeds 1/80000. The strict conservative
    gates ``norm_squared<9/800`` and ``excess<1/100000`` are therefore
    sufficient when proved by rational upper bounds. These are theorem
    premises, not adjustable numerical tolerances or an integration scheme.

    Field/topology errors raise; other unmet sufficient premises are retained
    as unavailable. Capacity/storage restrictions describe this certificate's
    scope, not the complete theorem or the model's full admitted state space.
    """
    if type(target_sector) is not int:
        raise TypeError("target_sector must be a nonboolean integer")
    if target_sector not in (-1, 0, 1):
        raise ValueError("target_sector must be -1, 0 or 1")
    field = evaluate_relational_exchange(graph, model=model)
    cycles, indices = _cycles(field, cycles)
    form, phase = tuple(map(Q, field.epi)), tuple(map(Q, field.phase))
    form_mean, phase_mean = sum(form) / len(form), sum(phase) / len(phase)
    centered_form = tuple(value - form_mean for value in form)
    reference = [Q(0)] * len(field.nodes)
    template = (Q(4, 5), Q(-4, 5), Q(-2, 5), Q(0), Q(2, 5))
    for row in indices:
        for index, coefficient in zip(row, template):
            reference[index] = target_sector * coefficient
    reference = tuple(reference)
    affine = tuple(
        (value - phase_mean, -coefficient)
        for value, coefficient in zip(phase, reference)
    )
    pi_bounds = _pi_bounds()
    phase_error_bounds = tuple(
        _affine_interval(rational, coefficient, pi_bounds)
        for rational, coefficient in affine
    )
    squares = tuple(_interval_square(interval) for interval in phase_error_bounds)
    form_norm = sum((value * value for value in centered_form), Q(0))
    phase_norm_bounds = tuple(sum((row[k] for row in squares), Q(0)) for k in (0, 1))
    quotient_bounds = tuple(form_norm + bound for bound in phase_norm_bounds)
    radius_threshold = Q(9, 800)
    radius = quotient_bounds[1] < radius_threshold
    phase_bounds = _phase_storage_bounds(field)
    beta = Q(field.model.storage_scale)
    storage_bounds = tuple(field.form_storage + beta * value for value in phase_bounds)
    reference_storage = (
        _twist_storage_bounds(pi_bounds) if target_sector else (Q(0), Q(0))
    )
    excess_bounds = (
        storage_bounds[0] - beta * reference_storage[1],
        storage_bounds[1] - beta * reference_storage[0],
    )
    energy_threshold = Q(1, 100000)
    energy = excess_bounds[1] < energy_threshold
    unit_capacity = all(value == 1.0 for value in field.capacity)
    unit_beta = beta == 1
    positive_epi = field.model.epi_weight > 0.0
    reasons = tuple(
        reason
        for admitted, reason in (
            (unit_capacity, "unit_capacity_required"),
            (unit_beta, "unit_storage_scale_required"),
            (positive_epi, "positive_epi_weight_required"),
            (radius, "strict_local_radius_not_certified"),
            (energy, "strict_local_energy_not_certified"),
        )
        if not admitted
    )
    detached = _detached_graph(field)
    return RelationalLocalCaptureCertificate(
        field=field,
        cycles=cycles,
        declared_target_sector=target_sector,
        reference_phase_pi_coefficients=reference,
        form_mean=form_mean,
        phase_mean=phase_mean,
        centered_form=centered_form,
        phase_error_affine=affine,
        phase_error_bounds=phase_error_bounds,
        form_norm_squared=form_norm,
        phase_norm_squared_bounds=phase_norm_bounds,
        quotient_squared_bounds=quotient_bounds,
        squared_radius_threshold=radius_threshold,
        radius_admitted=radius,
        phase_storage_bounds=phase_bounds,
        storage_bounds=storage_bounds,
        reference_phase_storage_bounds=reference_storage,
        excess_storage_bounds=excess_bounds,
        excess_storage_threshold=energy_threshold,
        energy_admitted=energy,
        unit_capacity=unit_capacity,
        unit_storage_scale=unit_beta,
        positive_epi_weight=positive_epi,
        winding=tuple(certify_phase_winding(detached, row) for row in cycles),
        target_sector=target_sector if not reasons else None,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class RelationalSectorGeometry:
    """Law-neutral exact sector and storage evidence on supplied unit support.

    Form, raw phase and beta are exact ratios of admitted materialized values.
    Edges retain graph order and orientation. ``admitted`` means only that
    the acute edge gaps, declared cycle periods and strict energy sublevel
    were resolved. The sublevel bounds concern states in the same sector
    below this storage ceiling; no trajectory, capture law or limiting target
    is certified. Capacity, pressure, forcing and evolution are not consumed.
    Public construction and JSON projection do not authenticate provenance.
    """

    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    epi: tuple[Q, ...]
    phase: tuple[Q, ...]
    storage_scale: Q
    form_storage: Q
    cycles: tuple[tuple[Any, ...], tuple[Any, ...]]
    declared_target_sector: int
    pi_bounds: tuple[Q, Q]
    edge_turn_candidates: tuple[int, ...]
    edge_gap_affine: tuple[tuple[Q, int], ...]
    edge_gap_bounds: tuple[tuple[Q, Q], ...]
    edge_acute_margin_bounds: tuple[tuple[tuple[Q, Q], tuple[Q, Q]], ...]
    edge_acute_admitted: tuple[bool, ...]
    acute_admitted: bool
    ring_windings: tuple[int | None, int | None]
    bridge_cycle: tuple[Any, ...]
    bridge_winding: int | None
    sector_admitted: bool
    phase_storage_bounds: tuple[Q, Q]
    storage_bounds: tuple[Q, Q]
    barrier_cosine_bounds: tuple[tuple[Q, tuple[Q, Q]], ...]
    geometric_barrier_bounds: tuple[Q, Q]
    capture_barrier_bounds: tuple[Q, Q]
    storage_margin_lower_bound: Q
    normalized_energy_margin_lower_bound: Q
    energy_admitted: bool
    sublevel_acute_margin_lower_bound: Q | None
    sublevel_resultant_real_lower_bounds: tuple[Q, ...] | None
    sublevel_phase_metric_lower_bounds: tuple[Q, ...] | None
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = COSINE_ENCLOSURE_METHOD
    scope: tuple[str, ...] = (
        "detached_materialized_signed_form_raw_phase_and_supplied_positive_storage_scale",
        "simple_unit_two_C5_two_adjacent_bridge_support",
        "exact_pi_affine_acute_gaps_cycle_periods_and_certified_cosine_storage",
        "sublevel_geometry_only_without_capacity_pressure_forcing_or_rate_evaluation",
        "sublevel_bounds_require_the_same_sector_and_storage_not_above_this_ceiling",
        "no_evolution_law_admission_future_capture_or_trajectory_error_certificate",
    )

    @property
    def admitted(self) -> bool:
        """Whether all geometric sector and storage premises resolved."""
        return self.status == "admitted"


@dataclass(frozen=True)
class RelationalSectorCaptureCertificate:
    """Sufficient full-state capture inside a declared acute winding sector.

    Edge evidence follows ``field.edges`` orientation. A candidate gap is
    ``raw_phase_difference + 2*turn*pi``; it is certified as the unique acute
    wrapped gap only when both strict acute margins have positive lower
    bounds. Each exact ring period sums the admitted additive turn integers
    along that ring, since raw differences telescope. Numerical ``winding``
    reports are separate telemetry and never establish this admission.

    The barrier is a sufficient lower bound on all faces of this acute
    sector, not a claimed sharp minimum. No reflection or local-radius test
    is imposed. The target is the circular aligned twist modulo a common
    phase offset; original nodewise full-turn representatives need not equal
    the specific raw lifts used by the local certificate.

    ``geometric_barrier_bounds`` encloses the dimensionless geometric cost;
    ``capture_barrier_bounds`` includes the declared storage scale beta.
    ``normalized_energy_margin_lower_bound`` is the certified margin divided
    by beta. Positive held capacities need not agree. The retained unit flags
    are descriptive, not admission conditions. On full admission, the deficit
    supplies uniform future acute, resultant-real and phase-metric lower
    bounds for the ideal trajectory. Nodewise bounds follow ``field.nodes``.
    Unavailable premises yield unavailable future bounds, not valid zeroes.
    """

    field: RelationalExchangeField
    cycles: tuple[tuple[Any, ...], tuple[Any, ...]]
    declared_target_sector: int
    pi_bounds: tuple[Q, Q]
    edge_turn_candidates: tuple[int, ...]
    edge_gap_affine: tuple[tuple[Q, int], ...]
    edge_gap_bounds: tuple[tuple[Q, Q], ...]
    edge_acute_margin_bounds: tuple[tuple[tuple[Q, Q], tuple[Q, Q]], ...]
    edge_acute_admitted: tuple[bool, ...]
    acute_admitted: bool
    ring_windings: tuple[int | None, int | None]
    bridge_cycle: tuple[Any, ...]
    bridge_winding: int | None
    sector_admitted: bool
    phase_storage_bounds: tuple[Q, Q]
    storage_bounds: tuple[Q, Q]
    barrier_cosine_bounds: tuple[tuple[Q, tuple[Q, Q]], ...]
    geometric_barrier_bounds: tuple[Q, Q]
    capture_barrier_bounds: tuple[Q, Q]
    storage_margin_lower_bound: Q
    normalized_energy_margin_lower_bound: Q
    energy_admitted: bool
    positive_capacity: bool
    unit_capacity: bool
    unit_storage_scale: bool
    positive_epi_weight: bool
    future_acute_margin_lower_bound: Q | None
    future_resultant_real_lower_bounds: tuple[Q, ...] | None
    future_phase_metric_lower_bounds: tuple[Q, ...] | None
    winding: tuple[WindingCertificate, WindingCertificate]
    target_sector: int | None
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = COSINE_ENCLOSURE_METHOD
    scope: tuple[str, ...] = (
        "one_fresh_detached_native_relational_field_without_live_writes",
        "same_two_C5_two_adjacent_bridge_support_with_strictly_positive_held_capacities_and_storage_scale",
        "exact_pi_affine_strict_acute_support_gaps_and_integer_cycle_periods",
        "both_ring_periods_equal_declared_plus_or_minus_one_and_bridge_square_period_zero",
        "energy_upper_below_beta_times_certified_geometric_barrier_lower_bound",
        "unit_capacity_and_unit_storage_scale_flags_are_descriptive_not_admission_conditions",
        "normalized_energy_deficit_bounds_future_acute_gaps_resultants_and_phase_metric_under_the_ideal_law",
        "sufficient_acute_sector_face_barrier_not_claimed_sharp",
        "conditional_ideal_ODE_capture_from_this_exact_represented_snapshot_without_symmetry_or_local_norm_gate",
        "circular_aligned_twist_target_modulo_common_phase_and_fixed_nodewise_full_turn_representatives",
        "no_earlier_integration_error_bound_future_binary64_execution_or_revised_frozen_prediction",
        "no_autonomous_substrate_creation_or_physical_identity_claim",
    )

    @property
    def admitted(self) -> bool:
        """Whether the exact acute sector and strict energy barrier resolved."""
        return self.status == "admitted"


def _acute_gap_evidence(difference, pi_bounds):
    midpoint = sum(pi_bounds) / 2
    # This rational quotient proposes an integer only. True-pi inequalities
    # below independently certify it; wide/unresolved intervals fail closed.
    turn = -((difference + midpoint) // (2 * midpoint))
    affine = difference, 2 * turn
    interval = _affine_interval(*affine, pi_bounds)
    margins = (
        _affine_interval(difference, 2 * turn + Q(1, 2), pi_bounds),
        _affine_interval(-difference, Q(1, 2) - 2 * turn, pi_bounds),
    )
    return turn, affine, interval, margins, all(lower > 0 for lower, _ in margins)


def _require_sector(target_sector):
    if type(target_sector) is not int:
        raise TypeError("target_sector must be a nonboolean integer")
    if target_sector not in (-1, 1):
        raise ValueError("target_sector must be -1 or 1")


def _acute_period_evidence(edges, phases, cycles, pi_bounds):
    """Shared exact edge gaps and oriented periods, without winding telemetry."""
    evidence = tuple(
        _acute_gap_evidence(phases[right] - phases[left], pi_bounds)
        for left, right in edges
    )
    turns = tuple(row[0] for row in evidence)
    admitted = tuple(row[4] for row in evidence)
    oriented = {}
    for (left, right), turn, valid in zip(edges, turns, admitted):
        oriented[left, right] = turn, valid
        oriented[right, left] = -turn, valid
    periods = []
    for cycle in cycles:
        rows = tuple(
            oriented[left, right] for left, right in zip(cycle, cycle[1:] + cycle[:1])
        )
        periods.append(
            sum(turn for turn, _ in rows) if all(ok for _, ok in rows) else None
        )
    return {
        "edge_turn_candidates": turns,
        "edge_gap_affine": tuple(row[1] for row in evidence),
        "edge_gap_bounds": tuple(row[2] for row in evidence),
        "edge_acute_margin_bounds": tuple(row[3] for row in evidence),
        "edge_acute_admitted": admitted,
        "acute_admitted": all(admitted),
    }, tuple(periods)


def _sector_geometry(nodes, edges, epi, phases, beta, cycles, target_sector):
    """One exact geometry kernel for admitted primitive snapshots."""
    cycles, _ = _cycle_support(nodes, edges, cycles)
    epi, phases, beta = tuple(map(Q, epi)), tuple(map(Q, phases)), Q(beta)
    phase = dict(zip(nodes, phases))
    form = dict(zip(nodes, epi))
    form_storage = sum(
        ((form[left] - form[right]) ** 2 / 2 for left, right in edges), Q(0)
    )
    pi_bounds = _pi_bounds()
    bridge_cycle = (cycles[0][0], cycles[0][1], cycles[1][1], cycles[1][0])
    evidence, periods = _acute_period_evidence(
        edges, phase, (*cycles, bridge_cycle), pi_bounds
    )
    ring_windings, bridge_winding = periods[:2], periods[2]
    if bridge_winding not in (None, 0):
        raise ArithmeticError("four strictly acute gaps cannot have nonzero winding")
    acute = evidence["acute_admitted"]
    sector = ring_windings == (target_sector, target_sector) and bridge_winding == 0
    phase_bounds = _phase_storage_bounds_at(nodes, edges, phases)
    storage_bounds = tuple(form_storage + beta * value for value in phase_bounds)
    first = _pi_cosine_bounds(Q(2, 5), pi_bounds)
    second, cycle_barrier = _cycle_barrier_constants(pi_bounds)
    cosines = ((Q(2, 5), first), (Q(3, 8), second))
    geometric_barrier_bounds = (
        5 - 5 * first[1] + cycle_barrier[0],
        5 - 5 * first[0] + cycle_barrier[1],
    )
    barrier_bounds = tuple(beta * value for value in geometric_barrier_bounds)
    storage_margin = barrier_bounds[0] - storage_bounds[1]
    normalized_margin = storage_margin / beta
    energy = storage_margin > 0
    reasons = tuple(
        reason
        for admitted, reason in (
            (acute, "strict_acute_edge_lifts_not_certified"),
            (sector, "declared_cycle_periods_not_certified"),
            (energy, "strict_sector_energy_barrier_not_certified"),
        )
        if not admitted
    )
    # These margins describe the geometric sublevel. A dynamical consumer
    # must independently establish that its trajectory stays in that set.
    sublevel_acute = 13 * normalized_margin if not reasons else None
    degrees = dict.fromkeys(nodes, 0)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    sublevel_metric = (
        tuple(26 * degrees[node] * normalized_margin for node in nodes)
        if not reasons
        else None
    )
    sublevel_resultant = (
        tuple(value / pi_bounds[1] for value in sublevel_metric)
        if sublevel_metric is not None
        else None
    )
    return RelationalSectorGeometry(
        nodes=nodes,
        edges=edges,
        epi=epi,
        phase=phases,
        storage_scale=beta,
        form_storage=form_storage,
        cycles=cycles,
        declared_target_sector=target_sector,
        pi_bounds=pi_bounds,
        **evidence,
        ring_windings=ring_windings,
        bridge_cycle=bridge_cycle,
        bridge_winding=bridge_winding,
        sector_admitted=sector,
        phase_storage_bounds=phase_bounds,
        storage_bounds=storage_bounds,
        barrier_cosine_bounds=cosines,
        geometric_barrier_bounds=geometric_barrier_bounds,
        capture_barrier_bounds=barrier_bounds,
        storage_margin_lower_bound=storage_margin,
        normalized_energy_margin_lower_bound=normalized_margin,
        energy_admitted=energy,
        sublevel_acute_margin_lower_bound=sublevel_acute,
        sublevel_resultant_real_lower_bounds=sublevel_resultant,
        sublevel_phase_metric_lower_bounds=sublevel_metric,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


def observe_relational_sector_geometry(
    graph, *, storage_scale, cycles, target_sector: int = 1
) -> RelationalSectorGeometry:
    """Read exact acute-sector geometry without evaluating a dynamical law.

    Only simple unit support, signed scalar form, raw phase and a supplied
    positive beta are consumed. Missing capacities, stored pressure, Gamma
    and other evolution settings neither admit nor reject this geometric
    observation. They must be admitted separately before drawing a future
    capture conclusion. The shared scalar readers preserve alias authority,
    nonfinite/Boolean rejection and representability requirements.
    """
    _require_sector(target_sector)
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("sector geometry requires simple undirected loopless support")
    beta = _finite(storage_scale, "storage_scale")
    if beta <= 0:
        raise ValueError("storage_scale must be positive")
    for _, _, data in graph.edges(data=True):
        if _finite(data.get("weight", 1.0), "conductance") != 1.0:
            raise ValueError("sector geometry requires unit conductances")
    nodes, edges = tuple(graph), tuple(graph.edges())
    epi = tuple(_epi(_raw(graph.nodes[node], ALIAS_EPI, "EPI")) for node in nodes)
    phase = tuple(
        _finite(_raw(graph.nodes[node], ALIAS_THETA, "phase"), "phase")
        for node in nodes
    )
    return _sector_geometry(nodes, edges, epi, phase, beta, cycles, target_sector)


def certify_relational_sector_capture(
    graph, *, model: RelationalExchangeModel, cycles, target_sector: int = 1
) -> RelationalSectorCaptureCertificate:
    """Check sufficient ideal reference-law capture on an acute twist sector.

    One fresh detached native field supplies the snapshot to the shared exact
    geometry kernel. Its true-pi edge gaps, integer cycle periods and rigorous
    energy sublevel are independent of the represented winding telemetry.
    The reference law additionally requires positive held capacities and
    positive form/phase coefficients. No geometric report alone admits them.

    On full admission, storage nonincrease protects this same sublevel and
    the native law converges to its aligned twist modulo common offsets.
    The Jensen deficit eta gives future acute margins greater than 13*eta,
    Re(z)>=26*degree*eta/pi and H>=26*degree*eta. These bounds do not certify
    earlier trajectory error, future binary64 steps or another phase law.
    Existing engine-domain/topology errors raise; unresolved sufficient
    premises return unavailable evidence. The flat report API is retained.
    """
    _require_sector(target_sector)
    field = evaluate_relational_exchange(graph, model=model)
    geometry = _sector_geometry(
        field.nodes,
        field.edges,
        field.epi,
        field.phase,
        field.model.storage_scale,
        cycles,
        target_sector,
    )
    positive_capacity = all(value > 0.0 for value in field.capacity)
    positive_epi = field.model.epi_weight > 0.0
    reasons = (
        tuple(
            reason
            for admitted, reason in (
                (positive_capacity, "strictly_positive_held_capacity_required"),
                (positive_epi, "positive_epi_weight_required"),
            )
            if not admitted
        )
        + geometry.unavailable_reasons
    )
    detached = _detached_graph(field)
    return RelationalSectorCaptureCertificate(
        field=field,
        cycles=geometry.cycles,
        declared_target_sector=geometry.declared_target_sector,
        pi_bounds=geometry.pi_bounds,
        edge_turn_candidates=geometry.edge_turn_candidates,
        edge_gap_affine=geometry.edge_gap_affine,
        edge_gap_bounds=geometry.edge_gap_bounds,
        edge_acute_margin_bounds=geometry.edge_acute_margin_bounds,
        edge_acute_admitted=geometry.edge_acute_admitted,
        acute_admitted=geometry.acute_admitted,
        ring_windings=geometry.ring_windings,
        bridge_cycle=geometry.bridge_cycle,
        bridge_winding=geometry.bridge_winding,
        sector_admitted=geometry.sector_admitted,
        phase_storage_bounds=geometry.phase_storage_bounds,
        storage_bounds=geometry.storage_bounds,
        barrier_cosine_bounds=geometry.barrier_cosine_bounds,
        geometric_barrier_bounds=geometry.geometric_barrier_bounds,
        capture_barrier_bounds=geometry.capture_barrier_bounds,
        storage_margin_lower_bound=geometry.storage_margin_lower_bound,
        normalized_energy_margin_lower_bound=geometry.normalized_energy_margin_lower_bound,
        energy_admitted=geometry.energy_admitted,
        positive_capacity=positive_capacity,
        unit_capacity=all(value == 1.0 for value in field.capacity),
        unit_storage_scale=geometry.storage_scale == 1,
        positive_epi_weight=positive_epi,
        future_acute_margin_lower_bound=(
            geometry.sublevel_acute_margin_lower_bound if not reasons else None
        ),
        future_resultant_real_lower_bounds=(
            geometry.sublevel_resultant_real_lower_bounds if not reasons else None
        ),
        future_phase_metric_lower_bounds=(
            geometry.sublevel_phase_metric_lower_bounds if not reasons else None
        ),
        winding=tuple(certify_phase_winding(detached, row) for row in geometry.cycles),
        target_sector=target_sector if not reasons else None,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


@dataclass(frozen=True)
class RelationalCycleCaptureCertificate:
    """Conditional acute winding-one capture on one supplied unit C5.

    Exact pi-affine gaps establish the oriented period. The mathematical
    storage upper bound must lie below beta*(5-4*cos(3*pi/8)); represented
    field storage is retained separately. Positive capacities may differ.
    This admits the native fixed-support law from this snapshot, not a
    support event, an earlier transit or arbitrary completion of the phase row.
    """

    field: RelationalExchangeField
    cycle: tuple[Any, ...]
    declared_target_sector: int
    pi_bounds: tuple[Q, Q]
    edge_turn_candidates: tuple[int, ...]
    edge_gap_affine: tuple[tuple[Q, int], ...]
    edge_gap_bounds: tuple[tuple[Q, Q], ...]
    edge_acute_margin_bounds: tuple[tuple[tuple[Q, Q], tuple[Q, Q]], ...]
    edge_acute_admitted: tuple[bool, ...]
    acute_admitted: bool
    cycle_winding: int | None
    sector_admitted: bool
    phase_storage_bounds: tuple[Q, Q]
    storage_bounds: tuple[Q, Q]
    barrier_cosine_bounds: tuple[Q, Q]
    geometric_barrier_bounds: tuple[Q, Q]
    capture_barrier_bounds: tuple[Q, Q]
    storage_margin_lower_bound: Q
    normalized_energy_margin_lower_bound: Q
    energy_admitted: bool
    positive_capacity: bool
    positive_epi_weight: bool
    target_sector: int | None
    status: str
    unavailable_reasons: tuple[str, ...]
    enclosure_method: str = COSINE_ENCLOSURE_METHOD
    scope: tuple[str, ...] = (
        "one_fresh_detached_native_field_on_exactly_the_supplied_simple_unit_C5",
        "strict_true_pi_acute_gaps_oriented_period_plus_or_minus_one",
        "mathematical_storage_upper_below_beta_times_certified_cycle_face_barrier",
        "strictly_positive_held_capacities_and_positive_epi_phase_storage_coefficients",
        "conditional_ideal_capture_to_uniform_form_and_cycle_twist_modulo_own_common_offsets",
        "no_support_event_occurrence_transit_error_or_physical_identity_claim",
    )

    @property
    def admitted(self) -> bool:
        return self.status == "admitted"


def _single_cycle_support(field, cycle):
    cycle = _ordered(cycle, "cycle", limit=6)
    if len(cycle) != 5:
        raise ValueError("cycle must contain exactly five ordered nodes")
    try:
        valid = len(field.nodes) == len(set(cycle)) == 5 and set(cycle) == set(
            field.nodes
        )
    except TypeError as exc:
        raise ValueError("cycle nodes must belong to the captured support") from exc
    if not valid:
        raise ValueError("the cycle must cover the full five-node support exactly once")
    expected = {
        frozenset((left, right)) for left, right in zip(cycle, cycle[1:] + cycle[:1])
    }
    if {frozenset(edge) for edge in field.edges} != expected:
        raise ValueError("support must be exactly the supplied five-node cycle")
    return cycle


def _cycle_capture_from_field(field, cycle, target_sector):
    cycle = _single_cycle_support(field, cycle)
    pi_bounds = _pi_bounds()
    phase = {node: Q(value) for node, value in zip(field.nodes, field.phase)}
    gaps, periods = _acute_period_evidence(field.edges, phase, (cycle,), pi_bounds)
    sector = periods[0] == target_sector
    phase_bounds = _phase_storage_bounds(field)
    beta = Q(field.model.storage_scale)
    storage_bounds = tuple(field.form_storage + beta * value for value in phase_bounds)
    cosine, geometric = _cycle_barrier_constants(pi_bounds)
    barrier = tuple(beta * value for value in geometric)
    margin = barrier[0] - storage_bounds[1]
    energy = margin > 0
    positive_capacity = all(value > 0 for value in field.capacity)
    positive_epi = field.model.epi_weight > 0
    reasons = tuple(
        reason
        for admitted, reason in (
            (positive_capacity, "strictly_positive_held_capacity_required"),
            (positive_epi, "positive_epi_weight_required"),
            (gaps["acute_admitted"], "strict_acute_edge_lifts_not_certified"),
            (sector, "declared_cycle_period_not_certified"),
            (energy, "strict_cycle_energy_barrier_not_certified"),
        )
        if not admitted
    )
    return RelationalCycleCaptureCertificate(
        field=field,
        cycle=cycle,
        declared_target_sector=target_sector,
        pi_bounds=pi_bounds,
        **gaps,
        cycle_winding=periods[0],
        sector_admitted=sector,
        phase_storage_bounds=phase_bounds,
        storage_bounds=storage_bounds,
        barrier_cosine_bounds=cosine,
        geometric_barrier_bounds=geometric,
        capture_barrier_bounds=barrier,
        storage_margin_lower_bound=margin,
        normalized_energy_margin_lower_bound=margin / beta,
        energy_admitted=energy,
        positive_capacity=positive_capacity,
        positive_epi_weight=positive_epi,
        target_sector=target_sector if not reasons else None,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )


def certify_relational_cycle_capture(
    graph, *, model: RelationalExchangeModel, cycle, target_sector: int = 1
) -> RelationalCycleCaptureCertificate:
    """Admit an acute C5 winding +/-1 and its sufficient native-law basin.

    The connected field is evaluated once. Invalid field/support admission
    raises; unresolved sufficient capacity, period or storage premises return
    unavailable evidence. The state and graph remain unchanged. This small
    certificate does not generalize the cycle length, permit disconnected
    execution or select a removal event.
    """
    _require_sector(target_sector)
    return _cycle_capture_from_field(
        evaluate_relational_exchange(graph, model=model), cycle, target_sector
    )


@dataclass(frozen=True)
class RelationalDetachmentObservation:
    """One hypothetical two-bridge cut with unchanged primitive node state.

    Changes follow ``before.nodes`` and compare the connected before field
    with fresh fields inside the two component certificates. ``reset`` owns
    event accounting. Its represented wrapped phase cost can differ from the
    native field's raw-gap cost, so field storage changes are named separately
    and their reconciliation residual is retained. No edge is removed live.
    """

    before: RelationalExchangeField
    components: tuple[
        RelationalCycleCaptureCertificate, RelationalCycleCaptureCertificate
    ]
    removed_bridges: tuple[tuple[Any, Any], tuple[Any, Any]]
    reset: RelationalResetObservation
    form_rate_change: tuple[Q, ...]
    phase_rate_change: tuple[Q, ...]
    pressure_change: tuple[Q, ...]
    phase_metric_change: tuple[Q, ...]
    field_form_storage_change: Q
    field_phase_storage_change: Q
    field_storage_change: Q
    storage_reconciliation_residual: Q
    scope: tuple[str, ...] = (
        "supplied_two_C5_rings_and_matching_bridges_at_positions_zero_and_one",
        "hypothetical_state_preserving_removal_of_both_bridges_without_live_writes",
        "one_connected_before_field_and_two_separately_admitted_component_fields",
        "disconnected_union_used_only_for_shared_reset_accounting_never_field_execution",
        "per_node_changes_follow_before_node_order_and_retain_materialized_defects",
        "reset_owns_event_budget_field_storage_differences_have_separate_representation",
        "independent_C5_capture_is_sufficient_not_an_event_selector_or_known_removal_time",
        "no_reuse_of_continuous_loss_as_event_supply_or_substrate_origin_claim",
    )

    @property
    def capture_admitted(self) -> bool:
        """Whether both post-cut native fields meet their sufficient basin."""
        return all(component.admitted for component in self.components)


def observe_relational_detachment(
    graph, *, model: RelationalExchangeModel, cycles, target_sector: int = 1
) -> RelationalDetachmentObservation:
    """Observe removal of the two supplied formation bridges, without mutation.

    Each resulting C5 must separately admit the selected native field model.
    Field admission errors raise, just as in attachment. A valid field outside
    its sufficient capture basin remains available with an unavailable cycle
    certificate. The deletion does not supply its own trigger, occurrence
    law, clock, or proof that an earlier formation trajectory reached this
    state. In particular zero removed-edge storage need not mean zero rate
    change or a safe detachment time.
    """
    from .relational_observations import _field_changes, observe_relational_reset

    _require_sector(target_sector)
    before = evaluate_relational_exchange(graph, model=model)
    cycles, _ = _cycles(before, cycles)
    bridges = tuple((cycles[0][index], cycles[1][index]) for index in (0, 1))
    source = _detached_graph(before)
    source.graph["GAMMA"] = {"type": "none"}
    detached = source.copy()
    detached.remove_edges_from(bridges)
    components = tuple(
        certify_relational_cycle_capture(
            detached.subgraph(cycle).copy(),
            model=model,
            cycle=cycle,
            target_sector=target_sector,
        )
        for cycle in cycles
    )
    # Preserve original node order for the reset snapshot while recording
    # each component's refreshed derived pressure. EPI/phase/capacity agree.
    for component in components:
        for node, data in _detached_graph(component.field).nodes(data=True):
            detached.nodes[node].update(data)
    reset = observe_relational_reset(
        source, detached, storage_scale=model.storage_scale
    )
    reversed_changes = _field_changes(
        tuple(component.field for component in components), before
    )
    changes = {
        name: tuple(-value for value in reversed_changes[name])
        for name in (
            "form_rate_change",
            "phase_rate_change",
            "pressure_change",
            "phase_metric_change",
        )
    }
    changes.update(
        {
            "field_" + name: -reversed_changes[name]
            for name in (
                "form_storage_change",
                "phase_storage_change",
                "storage_change",
            )
        }
    )
    return RelationalDetachmentObservation(
        before=before,
        components=components,
        removed_bridges=bridges,
        reset=reset,
        **changes,
        storage_reconciliation_residual=changes["field_storage_change"]
        - reset.storage_change,
    )
