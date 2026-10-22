"""Sufficient continuous state of the marked weighted star and its boundary.

Four collective coordinates and six independent internal coordinates retain
all ten fine coordinates on a declared phase lift. Mean and phase-moment
observations are also exposed, without claiming that these summaries close.
This detached exact-chart analysis installs no graph, law, solver or event.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
from ._sine_weighted_star import (
    BARE_DEGREES,
    BLOCKS,
    CONDUCTANCES,
    EDGES,
    INTERNAL_EDGES,
    LAW,
    NODES,
    admit_scale,
    capacities,
    internal_form_action,
)
from .relational_sine_partition import (
    Expression,
    _difference,
    _expression_bound,
    _rational_turns,
    _sine_expression,
)
from .relational_sine_support import _support_values

__all__ = (
    "SineWeightedStarState",
    "SineWeightedStarRetention",
    "assess_sine_weighted_star_state",
    "assess_sine_weighted_star_retention",
    "reconstruct_sine_weighted_star_state",
)


def _scale(expression, coefficient):
    return _sine_expression((turn, coefficient * value) for turn, value in expression)


def _combine(*expressions):
    return _sine_expression(term for expression in expressions for term in expression)


def _phase_cost(differences):
    differences = tuple(differences)
    return _sine_expression(
        ((Q(1, 4), Q(len(differences))),)
        + tuple((turn + Q(1, 4), Q(-1)) for turn in differences)
    )


@dataclass(frozen=True)
class SineWeightedStarState:
    """Fine state, exact invertible coordinates and instantaneous dynamics.

    Primitive ``phase_turns`` are declared real lifts, not wrapped averages.
    Every phase-rate field uses radians per original structural time t.
    Expression/numerator fields omit only the indicated common 1/pi factor.
    The complex moment is M=mean(exp(i*eta)), where eta is in radians.
    The boundary node B remains dynamic and is included in every balance.
    """

    capacity_scale: Q
    form: tuple[Q, ...]
    phase_turns: tuple[Q, ...]
    capacity: tuple[Q, ...]
    collective_form: tuple[Q, Q]
    collective_phase_turns: tuple[Q, Q]
    internal_form: tuple[Q, ...]
    internal_phase_turns: tuple[Q, ...]
    form_gap: Q
    phase_gap_turns: Q
    pressure_numerator_expressions: tuple[Expression, ...]
    form_rate_numerator_expressions: tuple[Expression, ...]
    phase_rate_numerators: tuple[Q, ...]
    pressure_bounds: tuple[I, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    collective_form_rate_bounds: tuple[I, ...]
    collective_phase_rate_bounds: tuple[I, ...]
    internal_form_rate_bounds: tuple[I, ...]
    internal_phase_rate_bounds: tuple[I, ...]
    internal_phase_rate_numerators: tuple[Q, ...]
    contact_sine_mean_expression: Expression
    contact_sine_mean_bounds: I
    moment_real_expression: Expression
    moment_imag_expression: Expression
    moment_real_bounds: I
    moment_imag_bounds: I
    moment_real_rate_numerator_expression: Expression
    moment_imag_rate_numerator_expression: Expression
    moment_real_rate_bounds: I
    moment_imag_rate_bounds: I
    form_storage: Q
    phase_storage_expression: Expression
    phase_storage_bounds: I
    full_storage_bounds: I
    internal_form_storage: Q
    internal_phase_storage_bounds: I
    collective_reference_storage_bounds: I
    storage_remainder_bounds: I
    collective_reference_storage_rate_bounds: I
    storage_remainder_rate_bounds: I
    storage_rate_bounds: I
    weighted_form_mean: Q
    weighted_phase_mean_turns: Q
    weighted_form_mean_rate_bounds: I
    weighted_phase_mean_rate_bounds: I
    exact_row_reconstruction_certified: bool
    nodes: tuple[int, ...] = NODES
    blocks: tuple[tuple[int, ...], ...] = BLOCKS
    edges: tuple[tuple[int, int], ...] = EDGES
    conductances: tuple[int, ...] = CONDUCTANCES
    bare_degrees: tuple[int, ...] = BARE_DEGREES
    law: str = LAW
    clock: str = "t"
    phase_rate_units: str = "radians_per_original_structural_time"
    independent_continuous_dimension: int = 10
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "marked_four_node_star_A_with_retained_dynamic_boundary_B",
        "positive_held_capacity_fixed_support_no_forcing_loss_or_events",
        "four_collective_plus_six_internal_coordinates_are_invertible_not_a_reduction",
        "exact_rational_turn_lifts_and_signed_form_chart_not_a_live_graph_snapshot",
        "moment_and_storage_summaries_do_not_close_the_full_off_family_law",
        "phase_lift_changes_must_transport_means_internal_moment_and_phase_gap_together",
        "collective_reference_storage_and_signed_remainder_are_not_separate_invariants",
        "full_storage_does_not_determine_a_pendulum_amplitude_off_the_exact_family",
        "continuous_state_does_not_supply_grammar_history_lifecycle_or_event_selection",
        "no_nonlinear_stability_formation_minimality_or_physical_identification",
    )

    def to_dict(self):
        if type(self) is not SineWeightedStarState:
            raise TypeError("a SineWeightedStarState is required")
        fresh = assess_sine_weighted_star_state(
            capacity_scale=self.capacity_scale,
            form=self.form,
            phase_turns=self.phase_turns,
        )
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-weighted-star-state.v1", "report": _project(fresh)}


def assess_sine_weighted_star_state(
    *, form, phase_turns, capacity_scale=1
) -> SineWeightedStarState:
    """Read one supplied full state of the fixed weighted-star model.

    Forms use shared exact/represented real admission; phases are exact
    rational turns retaining the supplied lifts. This mathematical reader
    does not coerce a graph snapshot into rational phases. Instantaneous
    fields and exact identities are not a finite-time stability certificate.
    """
    scale = admit_scale(capacity_scale)
    form = _support_values(form, "form", 5)
    turns = _rational_turns(phase_turns, "phase_turns", 5)
    nu = capacities(scale)
    mobility = tuple(value / degree for value, degree in zip(nu, BARE_DEGREES))
    weights = tuple(1 / value for value in mobility)
    weight_sum = sum(weights, Q(0))
    mean_form, mean_turn = sum(form[:4], Q(0)) / 4, sum(turns[:4], Q(0)) / 4
    collective_form = (mean_form, form[4])
    collective_turns = (mean_turn, turns[4])
    xi = tuple(value - mean_form for value in form[:4])
    eta = tuple(value - mean_turn for value in turns[:4])
    gap, delta = mean_form - form[4], turns[4] - mean_turn
    pi, cache = pi_interval(), {}

    def bound(expression):
        return _expression_bound(expression, cache)

    # Independently assemble the primitive pressure and both fine rows from
    # actual incidence. Conductance enters the form gradient, not the sine sum.
    currents, gradient = [[] for _ in NODES], [Q(0) for _ in NODES]
    form_storage = Q(0)
    for (i, j), weight in zip(EDGES, CONDUCTANCES):
        difference = form[i] - form[j]
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        currents[i].append((turns[j] - turns[i], Q(1)))
        currents[j].append((turns[i] - turns[j], Q(1)))
        form_storage += weight * difference**2 / 2
    currents = tuple(_sine_expression(terms) for terms in currents)
    pressure = tuple(
        _scale(expr, Q(1, degree)) for expr, degree in zip(currents, BARE_DEGREES)
    )
    fine_form = tuple(_scale(expr, value) for expr, value in zip(currents, mobility))
    fine_phase = tuple(value * grad for value, grad in zip(mobility, gradient))

    # Reconstruct the complete field through the sufficient collective chart.
    contact = _sine_expression((delta - value, Q(1, 4)) for value in eta)
    collective_form_rows = (_scale(contact, scale), _scale(contact, -8))
    collective_phase_rows = (2 * scale * gap, -16 * gap)
    internal_terms = [[] for _ in BLOCKS[0]]
    for i, j in INTERNAL_EDGES:
        internal_terms[i].append((eta[j] - eta[i], Q(1)))
        internal_terms[j].append((eta[i] - eta[j], Q(1)))
    internal_form_rows = tuple(
        _scale(
            _difference(_sine_expression((*terms, (delta - eta[i], Q(1)))), contact),
            scale,
        )
        for i, terms in enumerate(internal_terms)
    )
    action = internal_form_action(xi)
    internal_phase_rows = tuple(scale * value for value in action)
    reconstructed_form = tuple(
        _combine(collective_form_rows[0], value) for value in internal_form_rows
    ) + (collective_form_rows[1],)
    reconstructed_phase = tuple(
        collective_phase_rows[0] + value for value in internal_phase_rows
    ) + (collective_phase_rows[1],)
    if (
        any(_difference(a, b) for a, b in zip(fine_form, reconstructed_form))
        or fine_phase != reconstructed_phase
    ):
        raise ArithmeticError("fine rows and sufficient-coordinate rows disagree")
    if any(
        _difference(row, _scale(p, value))
        for row, p, value in zip(fine_form, pressure, nu)
    ):
        raise ArithmeticError("independent pressure lost the nodal product")

    moment_real = _sine_expression((value + Q(1, 4), Q(1, 4)) for value in eta)
    moment_imag = _sine_expression((value, Q(1, 4)) for value in eta)
    moment_real_rate = _sine_expression(
        (value, -rate / 4) for value, rate in zip(eta, internal_phase_rows)
    )
    moment_imag_rate = _sine_expression(
        (value + Q(1, 4), rate / 4) for value, rate in zip(eta, internal_phase_rows)
    )
    phase_storage = _phase_cost(turns[j] - turns[i] for i, j in EDGES)
    internal_phase_storage = _phase_cost(eta[j] - eta[i] for i, j in INTERNAL_EDGES)
    internal_form_storage = (
        sum((value * rate for value, rate in zip(xi, action)), Q(0)) / 2
    )
    if form_storage != 4 * gap**2 + internal_form_storage:
        raise ArithmeticError("fine form storage lost its exact collective split")
    reference_phase = _phase_cost((delta,) * 4)
    remainder_phase = _difference(phase_storage, reference_phase)
    reference_storage_rate = _scale(
        _difference(contact, _sine_expression(((delta, Q(1)),))),
        8 * (scale + 8) * gap,
    )

    # Differentiate full storage using both complete rows; the cancellation
    # is checked symbolically, not inferred from an interval containing zero.
    storage_rate = _combine(
        *(_scale(expr, grad) for expr, grad in zip(fine_form, gradient)),
        *(_scale(expr, -rate) for expr, rate in zip(currents, fine_phase)),
    )
    form_mean_rate = _combine(
        *(_scale(expr, weight / weight_sum) for expr, weight in zip(fine_form, weights))
    )
    phase_mean_rate = (
        sum((weight * value for weight, value in zip(weights, fine_phase)), Q(0))
        / weight_sum
    )
    if storage_rate or form_mean_rate or phase_mean_rate:
        raise ArithmeticError("complete conservative rows lost a balance identity")
    return SineWeightedStarState(
        capacity_scale=scale,
        form=form,
        phase_turns=turns,
        capacity=nu,
        collective_form=collective_form,
        collective_phase_turns=collective_turns,
        internal_form=xi,
        internal_phase_turns=eta,
        form_gap=gap,
        phase_gap_turns=delta,
        pressure_numerator_expressions=pressure,
        form_rate_numerator_expressions=fine_form,
        phase_rate_numerators=fine_phase,
        pressure_bounds=tuple(bound(value) / pi for value in pressure),
        form_rate_bounds=tuple(bound(value) / pi for value in fine_form),
        phase_rate_bounds=tuple(I(value) / pi for value in fine_phase),
        collective_form_rate_bounds=tuple(
            bound(value) / pi for value in collective_form_rows
        ),
        collective_phase_rate_bounds=tuple(
            I(value) / pi for value in collective_phase_rows
        ),
        internal_form_rate_bounds=tuple(
            bound(value) / pi for value in internal_form_rows
        ),
        internal_phase_rate_bounds=tuple(
            I(value) / pi for value in internal_phase_rows
        ),
        internal_phase_rate_numerators=internal_phase_rows,
        contact_sine_mean_expression=contact,
        contact_sine_mean_bounds=bound(contact),
        moment_real_expression=moment_real,
        moment_imag_expression=moment_imag,
        moment_real_bounds=bound(moment_real),
        moment_imag_bounds=bound(moment_imag),
        moment_real_rate_numerator_expression=moment_real_rate,
        moment_imag_rate_numerator_expression=moment_imag_rate,
        moment_real_rate_bounds=bound(moment_real_rate) / pi,
        moment_imag_rate_bounds=bound(moment_imag_rate) / pi,
        form_storage=form_storage,
        phase_storage_expression=phase_storage,
        phase_storage_bounds=bound(phase_storage),
        full_storage_bounds=form_storage + bound(phase_storage),
        internal_form_storage=internal_form_storage,
        internal_phase_storage_bounds=bound(internal_phase_storage),
        collective_reference_storage_bounds=4 * gap**2 + bound(reference_phase),
        storage_remainder_bounds=internal_form_storage + bound(remainder_phase),
        collective_reference_storage_rate_bounds=bound(reference_storage_rate) / pi,
        storage_remainder_rate_bounds=-bound(reference_storage_rate) / pi,
        storage_rate_bounds=bound(storage_rate) / pi,
        weighted_form_mean=sum(
            (weight * value for weight, value in zip(weights, form)), Q(0)
        )
        / weight_sum,
        weighted_phase_mean_turns=sum(
            (weight * value for weight, value in zip(weights, turns)), Q(0)
        )
        / weight_sum,
        weighted_form_mean_rate_bounds=bound(form_mean_rate) / pi,
        weighted_phase_mean_rate_bounds=I(phase_mean_rate) / pi,
        exact_row_reconstruction_certified=True,
    )


def reconstruct_sine_weighted_star_state(
    *,
    collective_form,
    collective_phase_turns,
    internal_form,
    internal_phase_turns,
    capacity_scale=1,
) -> SineWeightedStarState:
    """Invert the sufficient chart; reject nonzero sums rather than recenter.

    The four internal entries in each channel have exactly zero sum, leaving
    three independent coordinates. Reconstruction retains the supplied lifts
    and then re-admits all fine primitives through the same state reader.
    """
    scale = admit_scale(capacity_scale)
    form = _support_values(collective_form, "collective_form", 2)
    turns = _rational_turns(collective_phase_turns, "collective_phase_turns", 2)
    xi = _support_values(internal_form, "internal_form", 4)
    eta = _rational_turns(internal_phase_turns, "internal_phase_turns", 4)
    if sum(xi, Q(0)) != 0 or sum(eta, Q(0)) != 0:
        raise ValueError("internal form and phase lifts must each sum exactly to zero")
    return assess_sine_weighted_star_state(
        capacity_scale=scale,
        form=tuple(form[0] + value for value in xi) + (form[1],),
        phase_turns=tuple(turns[0] + value for value in eta) + (turns[1],),
    )


@dataclass(frozen=True)
class SineWeightedStarRetention:
    """Sufficient finite-amplitude identity retention from full stored energy.

    Identity means each center-leaf form contrast is below 1/2 and each
    principal circular internal phase gap is below 1/4 turn. The threshold
    is a proved storage-sublevel inclusion, not proof that higher-storage
    states escape or that this supplied identity was previously acquired.
    """

    state: SineWeightedStarState
    internal_form_gaps: tuple[Q, ...]
    internal_phase_gaps_turns: tuple[Q, ...]
    source_identity_admitted: bool
    storage_margin: Q
    internal_form_gap_upper_bound: I
    internal_phase_cosine_lower_bound: Q
    all_time_identity_retention_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    form_radius: Q = Q(1, 2)
    phase_radius_turns: Q = Q(1, 4)
    effective_form_conductance: Q = Q(35, 8)
    form_boundary_storage: Q = Q(35, 64)
    phase_boundary_storage_lower_bound: Q = Q(1)
    escape_storage_lower_bound: Q = Q(35, 64)
    temporal_scope: str = "all_real_times_under_the_unchanged_complete_held_law"
    scope: tuple[str, ...] = (
        "marked_center_leaf_identity_with_fully_dynamic_retained_boundary",
        "full_storage_rebuilt_from_primitive_forms_phase_lifts_scale_and_fixed_support",
        "strict_energy_sublevel_and_global_flow_existence_not_linearization",
        "phase_boundary_uses_nonnegative_bare_edge_storage_not_signed_remainder",
        "unavailable_sufficient_bound_does_not_prove_identity_loss",
        "no_reference_orbit_tracking_attraction_formation_or_physical_identification",
        "no_new_trajectory_event_support_change_or_operator_admission",
    )

    def to_dict(self):
        if type(self) is not SineWeightedStarRetention:
            raise TypeError("a SineWeightedStarRetention is required")
        fresh = assess_sine_weighted_star_retention(self.state)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-weighted-star-retention.v1",
            "report": _project(fresh),
        }


def assess_sine_weighted_star_retention(state) -> SineWeightedStarRetention:
    """Re-admit full primitive state and apply the fixed identity boundary.

    The Dirichlet form-energy minimum at one contrast d is 35*d**2/16.
    A phase gap pi/2 costs at least one in the nonnegative full phase storage.
    Strict H < 35/64 therefore preserves the declared identity for all time
    under this fixed conservative law. Unresolved inequalities abstain.
    """
    if type(state) is not SineWeightedStarState:
        raise TypeError("a SineWeightedStarState is required")
    fresh = assess_sine_weighted_star_state(
        capacity_scale=state.capacity_scale,
        form=state.form,
        phase_turns=state.phase_turns,
    )
    form_gaps = tuple(abs(fresh.form[i] - fresh.form[j]) for i, j in INTERNAL_EDGES)
    phase_gaps = tuple(
        abs((fresh.phase_turns[j] - fresh.phase_turns[i] + Q(1, 2)) % 1 - Q(1, 2))
        for i, j in INTERNAL_EDGES
    )
    identity = all(value < Q(1, 2) for value in form_gaps) and all(
        value < Q(1, 4) for value in phase_gaps
    )
    upper = fresh.full_storage_bounds.hi
    if upper < 0:
        raise ArithmeticError("nonnegative full storage has a negative upper bound")
    margin = Q(35, 64) - upper
    if margin > 0 and not identity:
        raise ArithmeticError(
            "strict storage sublevel contradicts its identity boundary"
        )
    certified = identity and margin > 0
    reasons = () if certified else ("strict_full_storage_margin_not_certified",)
    return SineWeightedStarRetention(
        state=fresh,
        internal_form_gaps=form_gaps,
        internal_phase_gaps_turns=phase_gaps,
        source_identity_admitted=identity,
        storage_margin=margin,
        internal_form_gap_upper_bound=sqrt(I(Q(16, 35) * upper)),
        internal_phase_cosine_lower_bound=1 - upper,
        all_time_identity_retention_certified=certified,
        status="certified_identity_retention" if certified else "unavailable",
        unavailable_reasons=reasons,
    )
