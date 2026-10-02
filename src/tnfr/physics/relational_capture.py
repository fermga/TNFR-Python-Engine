"""Conditional continuous capture from detached two-ring states.

Three read-only owners admit exact reflected protected regions, a full-state
local neighborhood, or a full acute winding sector. They share fresh native
fields, topology and rigorous trigonometric bounds. They neither project
nearly symmetric states, integrate a reduced model, certify future binary64
steps, nor equate floating winding telemetry with a proven target. Exact
energy enclosures remain separate from production storage arithmetic. All
claims concern ideal continuation from the supplied snapshot, independently
of an earlier numerical trajectory's error or a frozen experimental verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from ..dynamics.relational import (
    RelationalExchangeField,
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from ..mathematics._phase_midpoint import _affine_interval, _pi_bounds
from ..mathematics._phase_resultant_chamber import (
    COSINE_ENCLOSURE_METHOD,
    certified_cosine_bounds,
)
from .relational_observations import _detached_graph, _ordered
from .winding_certificates import WindingCertificate, certify_phase_winding

__all__ = (
    "RelationalCaptureCertificate",
    "RelationalLocalCaptureCertificate",
    "RelationalSectorCaptureCertificate",
    "certify_relational_capture",
    "certify_relational_local_capture",
    "certify_relational_sector_capture",
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
    rows = tuple(_ordered(row, "cycle") for row in _ordered(cycles, "cycles"))
    if len(rows) != 2 or any(len(row) != 5 for row in rows):
        raise ValueError("cycles must contain exactly two ordered five-node rings")
    lookup = {node: index for index, node in enumerate(field.nodes)}
    try:
        indices = tuple(tuple(lookup[node] for node in row) for row in rows)
    except (TypeError, KeyError) as exc:
        raise ValueError("cycle nodes must belong to the captured support") from exc
    flat = tuple(i for row in indices for i in row)
    if len(field.nodes) != 10 or len(set(flat)) != 10:
        raise ValueError("the two cycles must cover the full support exactly once")
    expected = {
        frozenset((left, right))
        for row in indices
        for left, right in zip(row, row[1:] + row[:1])
    }
    expected.update(frozenset((indices[0][i], indices[1][i])) for i in (0, 1))
    actual = {frozenset((lookup[a], lookup[b])) for a, b in field.edges}
    if actual != expected:
        raise ValueError(
            "support must be the two supplied cycles and bridges at matching positions 0 and 1"
        )
    return rows, indices


def _phase_storage_bounds(field):
    phase = tuple(map(Q, field.phase))
    position = {node: i for i, node in enumerate(field.nodes)}
    bounds = tuple(
        certified_cosine_bounds(phase[position[j]] - phase[position[i]])
        for i, j in field.edges
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


def certify_relational_sector_capture(
    graph, *, model: RelationalExchangeModel, cycles, target_sector: int = 1
) -> RelationalSectorCaptureCertificate:
    """Check sufficient capture on the whole acute aligned-twist sector.

    The declared ring sector is +1 or -1, excluding Booleans. Each raw edge
    difference receives an integer full-turn candidate and rigorous affine-pi
    acute-margin tests. These tests, not floating remainders or the separate
    winding read-out, establish all exact cycle periods. The independent
    bridge square must have period zero; four strictly acute gaps imply this.

    On the fixed paired support, an acute ring-sector face costs at least
    ``5-4*cos(3*pi/8)`` while the other ring costs at least
    ``5-5*cos(2*pi/5)``. Bridge faces cost more. Thus the certified energy
    upper bound below a certified lower bound on their sum prevents exit.
    With positive held capacities, beta and e,w, LaSalle and strict convexity
    in the fixed cycle-period chart give the circular aligned-twist target.
    No exact copy/reflection or small full-state displacement is assumed.

    The barrier is scaled by beta. For a certified normalized deficit eta,
    the ring Jensen profile gives every future acute margin greater than
    13*eta: its boundary slope is less than 1/13. This implies per-node
    Re(z)>=26*degree*eta/pi and H>=26*degree*eta. These derived bounds
    are not a controller, capacity-evolution law or integration certificate.

    The result concerns the ideal continuous law from this snapshot. It does
    not revise previous frozen finite criteria or certify the trajectory
    producing the state. Existing engine-domain/topology errors still raise;
    unmet sufficient capture premises return detached unavailable evidence.
    """
    if type(target_sector) is not int:
        raise TypeError("target_sector must be a nonboolean integer")
    if target_sector not in (-1, 1):
        raise ValueError("target_sector must be -1 or 1")
    field = evaluate_relational_exchange(graph, model=model)
    cycles, _ = _cycles(field, cycles)
    phase = {node: Q(value) for node, value in zip(field.nodes, field.phase)}
    pi_bounds = _pi_bounds()
    evidence = tuple(
        _acute_gap_evidence(phase[right] - phase[left], pi_bounds)
        for left, right in field.edges
    )
    turns = tuple(row[0] for row in evidence)
    affine = tuple(row[1] for row in evidence)
    gaps = tuple(row[2] for row in evidence)
    margins = tuple(row[3] for row in evidence)
    admitted_edges = tuple(row[4] for row in evidence)
    oriented = {}
    for (left, right), turn, admitted in zip(field.edges, turns, admitted_edges):
        oriented[left, right] = turn, admitted
        oriented[right, left] = -turn, admitted

    def period(cycle):
        rows = tuple(
            oriented[left, right] for left, right in zip(cycle, cycle[1:] + cycle[:1])
        )
        return sum(turn for turn, _ in rows) if all(ok for _, ok in rows) else None

    ring_windings = tuple(period(row) for row in cycles)
    bridge_cycle = (cycles[0][0], cycles[0][1], cycles[1][1], cycles[1][0])
    bridge_winding = period(bridge_cycle)
    if bridge_winding not in (None, 0):
        raise ArithmeticError("four strictly acute gaps cannot have nonzero winding")
    acute = all(admitted_edges)
    sector = ring_windings == (target_sector, target_sector) and bridge_winding == 0
    phase_bounds = _phase_storage_bounds(field)
    beta = Q(field.model.storage_scale)
    storage_bounds = tuple(field.form_storage + beta * value for value in phase_bounds)
    cosines = tuple(
        (coefficient, _pi_cosine_bounds(coefficient, pi_bounds))
        for coefficient in (Q(2, 5), Q(3, 8))
    )
    first, second = cosines[0][1], cosines[1][1]
    geometric_barrier_bounds = (
        10 - 5 * first[1] - 4 * second[1],
        10 - 5 * first[0] - 4 * second[0],
    )
    barrier_bounds = tuple(beta * value for value in geometric_barrier_bounds)
    storage_margin = barrier_bounds[0] - storage_bounds[1]
    normalized_margin = storage_margin / beta
    energy = storage_margin > 0
    positive_capacity = all(value > 0.0 for value in field.capacity)
    unit_capacity = all(value == 1.0 for value in field.capacity)
    unit_beta, positive_epi = beta == 1, field.model.epi_weight > 0.0
    reasons = tuple(
        reason
        for admitted, reason in (
            (positive_capacity, "strictly_positive_held_capacity_required"),
            (positive_epi, "positive_epi_weight_required"),
            (acute, "strict_acute_edge_lifts_not_certified"),
            (sector, "declared_cycle_periods_not_certified"),
            (energy, "strict_sector_energy_barrier_not_certified"),
        )
        if not admitted
    )
    # The Jensen deficit controls the entire ideal future, not only the
    # independently retained acute margins of this snapshot.
    future_acute = 13 * normalized_margin if not reasons else None
    degrees = dict.fromkeys(field.nodes, 0)
    for left, right in field.edges:
        degrees[left] += 1
        degrees[right] += 1
    future_metric = (
        tuple(26 * degrees[node] * normalized_margin for node in field.nodes)
        if not reasons
        else None
    )
    future_resultant = (
        tuple(value / pi_bounds[1] for value in future_metric)
        if future_metric is not None
        else None
    )
    detached = _detached_graph(field)
    return RelationalSectorCaptureCertificate(
        field=field,
        cycles=cycles,
        declared_target_sector=target_sector,
        pi_bounds=pi_bounds,
        edge_turn_candidates=turns,
        edge_gap_affine=affine,
        edge_gap_bounds=gaps,
        edge_acute_margin_bounds=margins,
        edge_acute_admitted=admitted_edges,
        acute_admitted=acute,
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
        positive_capacity=positive_capacity,
        unit_capacity=unit_capacity,
        unit_storage_scale=unit_beta,
        positive_epi_weight=positive_epi,
        future_acute_margin_lower_bound=future_acute,
        future_resultant_real_lower_bounds=future_resultant,
        future_phase_metric_lower_bounds=future_metric,
        winding=tuple(certify_phase_winding(detached, row) for row in cycles),
        target_sector=target_sector if not reasons else None,
        status="unavailable" if reasons else "admitted",
        unavailable_reasons=reasons,
    )
