"""Maintenance composition for two supplied weighted-star organizations.

One unit boundary-to-boundary edge changes both boundary normalizers. This
detached reader rebuilds the complete twenty-dimensional field and accounts
for its support work; it executes neither a runtime contact nor a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
from ._sine_weighted_star import CONDUCTANCES, EDGES, LAW
from .relational_sine_partition import (
    Expression,
    _difference,
    _expression_bound,
    _sine_expression,
)
from .relational_sine_weighted_state import (
    SineWeightedStarState,
    _combine,
    _phase_cost,
    _scale,
    assess_sine_weighted_star_retention,
)

__all__ = ("SineWeightedStarContact", "assess_sine_weighted_star_contact")

_NODES = tuple(range(10))
_EDGES = EDGES + tuple((i + 5, j + 5) for i, j in EDGES) + ((4, 9),)
_CONDUCTANCES = CONDUCTANCES * 2 + (1,)
_DEGREES = tuple(sum(i in edge for edge in _EDGES) for i in _NODES)
_BARRIER = Q(35, 64)


@dataclass(frozen=True)
class SineWeightedStarContact:
    """Actual joined-law rows, supplied support work and joint retention.

    ``left`` and ``right`` are freshly admitted isolated-reference snapshots;
    their stored rate fields remain associated with those isolated laws.
    Post-contact rates belong to this report. Numerator expressions omit the
    common 1/pi factor; all phase rates use radians per original time t.
    The retention flag applies only the strict joint-energy criterion; another
    theorem may retain an exact family when this criterion is unavailable.
    """

    left: SineWeightedStarState
    right: SineWeightedStarState
    form: tuple[Q, ...]
    phase_turns: tuple[Q, ...]
    capacity: tuple[Q, ...]
    pressure_numerator_expressions: tuple[Expression, ...]
    form_rate_numerator_expressions: tuple[Expression, ...]
    phase_rate_numerators: tuple[Q, ...]
    pressure_bounds: tuple[I, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    bridge_form_difference: Q
    bridge_phase_gap_turns: Q
    bridge_current_expression: Expression
    bridge_current_bounds: I
    component_storage_bounds: tuple[I, I]
    bridge_form_storage: Q
    bridge_phase_storage_bounds: I
    support_work_bounds: I
    form_storage: Q
    phase_storage_expression: Expression
    phase_storage_bounds: I
    full_storage_bounds: I
    component_storage_rate_numerator_expressions: tuple[Expression, Expression]
    component_storage_rate_bounds: tuple[I, I]
    bridge_storage_rate_numerator_expression: Expression
    bridge_storage_rate_bounds: I
    storage_rate_bounds: I
    postcontact_mean_weights: tuple[Q, ...]
    weighted_form_mean: Q
    weighted_phase_mean_turns: Q
    weighted_form_mean_rate_bounds: I
    weighted_phase_mean_rate_bounds: I
    source_individual_retention_certified: tuple[bool, bool]
    source_joint_identity_admitted: bool
    joint_storage_margin: Q
    internal_form_gap_upper_bound: I
    internal_phase_cosine_lower_bound: Q
    all_time_joint_identity_retention_certified: bool
    exact_chart_reconstruction_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    conductances: tuple[int, ...] = _CONDUCTANCES
    bare_degrees: tuple[int, ...] = _DEGREES
    bridge: tuple[int, int] = (4, 9)
    joint_identity_storage_boundary: Q = _BARRIER
    law: str = LAW
    clock: str = "t"
    phase_rate_units: str = "radians_per_original_structural_time"
    independent_continuous_dimension: int = 20
    arithmetic_method: str = INTERVAL_METHOD
    temporal_scope: str = "all_real_times_under_the_supplied_joined_held_law"
    scope: tuple[str, ...] = (
        "two_marked_stars_each_at_exact_scale_one_with_one_supplied_unit_boundary_edge",
        "all_twenty_fine_coordinates_retained_with_both_boundaries_dynamic",
        "left_and_right_rate_fields_are_isolated_reference_fields_not_joined_fields",
        "source_individual_certificates_apply_to_old_laws_not_postcontact_flow",
        "actual_degree_five_boundary_mobility_and_bare_phase_bridge_are_rebuilt",
        "support_insertion_work_is_supplied_and_not_financed_by_unnamed_dissipation",
        "component_storage_exchange_and_joint_conservation_use_both_complete_rows",
        "joint_sublevel_certifies_retention_higher_storage_does_not_prove_escape",
        "no_runtime_UM_event_selector_trajectory_formation_or_physical_identification",
    )

    def to_dict(self):
        if type(self) is not SineWeightedStarContact:
            raise TypeError("a SineWeightedStarContact is required")
        fresh = assess_sine_weighted_star_contact(self.left, self.right)
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-weighted-star-contact.v1",
            "report": _project(fresh),
        }


def _source(source):
    retention = assess_sine_weighted_star_retention(source)
    if retention.state.capacity_scale != 1:
        raise ValueError("weighted-star contact requires exact capacity_scale one")
    return retention


def assess_sine_weighted_star_contact(left, right) -> SineWeightedStarContact:
    """Admit two full states under the fixed unit-contact completion.

    Rebuild every source premise before consuming it. The contact is a
    mathematical support comparison at unchanged forms, phases and capacities,
    not an instruction to execute UM. This energy criterion abstains at larger
    joint storage even when both isolated certificates pass.
    """
    retained = (_source(left), _source(right))
    left, right = (item.state for item in retained)
    form = left.form + right.form
    turns = left.phase_turns + right.phase_turns
    nu = left.capacity + right.capacity
    mobility = tuple(value / degree for value, degree in zip(nu, _DEGREES))
    weights = tuple(1 / value for value in mobility)
    weight_sum = sum(weights, Q(0))
    pi, cache = pi_interval(), {}

    def bound(expression):
        return _expression_bound(expression, cache)

    # Assemble the actual complete field independently from joined incidence.
    currents, gradient = [[] for _ in _NODES], [Q(0) for _ in _NODES]
    own_currents, own_gradient = [[] for _ in _NODES], [Q(0) for _ in _NODES]
    form_storage = Q(0)
    for (i, j), weight in zip(_EDGES, _CONDUCTANCES):
        difference = form[i] - form[j]
        current = (turns[j] - turns[i], Q(1))
        reverse = (turns[i] - turns[j], Q(1))
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        currents[i].append(current)
        currents[j].append(reverse)
        form_storage += weight * difference**2 / 2
        if (i, j) != (4, 9):
            own_gradient[i] += weight * difference
            own_gradient[j] -= weight * difference
            own_currents[i].append(current)
            own_currents[j].append(reverse)
    currents = tuple(_sine_expression(value) for value in currents)
    own_currents = tuple(_sine_expression(value) for value in own_currents)
    pressure = tuple(
        _scale(expr, Q(1, degree)) for expr, degree in zip(currents, _DEGREES)
    )
    fine_form = tuple(_scale(expr, value) for expr, value in zip(currents, mobility))
    fine_phase = tuple(value * grad for value, grad in zip(mobility, gradient))
    if any(
        _difference(row, _scale(p, value))
        for row, p, value in zip(fine_form, pressure, nu)
    ):
        raise ArithmeticError("joined independent pressure lost the nodal product")

    f, delta = form[4] - form[9], turns[9] - turns[4]
    bridge_current = _sine_expression(((delta, Q(1)),))
    # The sufficient internal charts remain intact. Only the two boundary
    # rows change; these expressions retain each source's full internal state.
    reconstructed_form = list(
        left.form_rate_numerator_expressions + right.form_rate_numerator_expressions
    )
    reconstructed_phase = list(left.phase_rate_numerators + right.phase_rate_numerators)
    for index, source, sign in ((4, left, 1), (9, right, -1)):
        reconstructed_form[index] = _scale(
            _combine(
                _scale(source.contact_sine_mean_expression, -4),
                _scale(bridge_current, sign),
            ),
            Q(8, 5),
        )
        reconstructed_phase[index] = Q(8, 5) * (-8 * source.form_gap + sign * f)
    if any(
        _difference(a, b) for a, b in zip(fine_form, reconstructed_form)
    ) or fine_phase != tuple(reconstructed_phase):
        raise ArithmeticError("joined fine rows and sufficient-chart rows disagree")

    bridge_form = f**2 / 2
    bridge_phase = _phase_cost((delta,))
    phase_storage = _phase_cost(turns[j] - turns[i] for i, j in _EDGES)
    if (
        form_storage != left.form_storage + right.form_storage + bridge_form
        or _difference(
            phase_storage,
            _combine(
                left.phase_storage_expression,
                right.phase_storage_expression,
                bridge_phase,
            ),
        )
    ):
        raise ArithmeticError("joined storage lost the exact support-insertion work")

    # Differentiate each old component's full storage using the joined rows.
    component_work = tuple(
        _combine(
            *(_scale(fine_form[i], own_gradient[i]) for i in indices),
            *(_scale(own_currents[i], -fine_phase[i]) for i in indices),
        )
        for indices in (range(5), range(5, 10))
    )
    expected_work = tuple(
        _combine(
            _scale(bridge_current, -Q(64, 5) * sign * source.form_gap),
            _scale(source.contact_sine_mean_expression, Q(32, 5) * sign * f),
        )
        for source, sign in ((left, 1), (right, -1))
    )
    bridge_work = _combine(
        _scale(_difference(fine_form[4], fine_form[9]), f),
        _scale(bridge_current, fine_phase[9] - fine_phase[4]),
    )
    full_work = _combine(
        *(_scale(expr, grad) for expr, grad in zip(fine_form, gradient)),
        *(_scale(expr, -rate) for expr, rate in zip(currents, fine_phase)),
    )
    if (
        any(_difference(a, b) for a, b in zip(component_work, expected_work))
        or _combine(*component_work, bridge_work)
        or full_work
    ):
        raise ArithmeticError(
            "joined component supply lost complete storage conservation"
        )
    form_mean_rate = _combine(
        *(_scale(expr, weight / weight_sum) for expr, weight in zip(fine_form, weights))
    )
    phase_mean_rate = (
        sum((weight * value for weight, value in zip(weights, fine_phase)), Q(0))
        / weight_sum
    )
    if form_mean_rate or phase_mean_rate:
        raise ArithmeticError("joined actual-degree means lost conservation")

    full_storage = form_storage + bound(phase_storage)
    if full_storage.hi < 0:
        raise ArithmeticError("nonnegative joined storage has a negative upper bound")
    identity = all(item.source_identity_admitted for item in retained)
    margin = _BARRIER - full_storage.hi
    if margin > 0 and not identity:
        raise ArithmeticError("joined strict storage sublevel contradicts its identity")
    certified = identity and margin > 0
    return SineWeightedStarContact(
        left=left,
        right=right,
        form=form,
        phase_turns=turns,
        capacity=nu,
        pressure_numerator_expressions=pressure,
        form_rate_numerator_expressions=fine_form,
        phase_rate_numerators=fine_phase,
        pressure_bounds=tuple(bound(value) / pi for value in pressure),
        form_rate_bounds=tuple(bound(value) / pi for value in fine_form),
        phase_rate_bounds=tuple(I(value) / pi for value in fine_phase),
        bridge_form_difference=f,
        bridge_phase_gap_turns=delta,
        bridge_current_expression=bridge_current,
        bridge_current_bounds=bound(bridge_current),
        component_storage_bounds=(left.full_storage_bounds, right.full_storage_bounds),
        bridge_form_storage=bridge_form,
        bridge_phase_storage_bounds=bound(bridge_phase),
        support_work_bounds=bridge_form + bound(bridge_phase),
        form_storage=form_storage,
        phase_storage_expression=phase_storage,
        phase_storage_bounds=bound(phase_storage),
        full_storage_bounds=full_storage,
        component_storage_rate_numerator_expressions=component_work,
        component_storage_rate_bounds=tuple(
            bound(value) / pi for value in component_work
        ),
        bridge_storage_rate_numerator_expression=bridge_work,
        bridge_storage_rate_bounds=bound(bridge_work) / pi,
        storage_rate_bounds=bound(full_work) / pi,
        postcontact_mean_weights=weights,
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
        source_individual_retention_certified=tuple(
            item.all_time_identity_retention_certified for item in retained
        ),
        source_joint_identity_admitted=identity,
        joint_storage_margin=margin,
        internal_form_gap_upper_bound=sqrt(I(Q(16, 35) * full_storage.hi)),
        internal_phase_cosine_lower_bound=1 - full_storage.hi,
        all_time_joint_identity_retention_certified=certified,
        exact_chart_reconstruction_certified=True,
        status="certified_joint_identity_retention" if certified else "unavailable",
        unavailable_reasons=(
            () if certified else ("strict_joint_storage_margin_not_certified",)
        ),
    )
