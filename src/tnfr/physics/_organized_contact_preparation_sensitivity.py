"""Full-fine local preparation variations of the fixed organized contact.

The coefficients multiply epsilon and epsilon**2; the second coefficient is
one half of the second amplitude derivative. Time rows are derivatives, not
normalized Taylor coefficients. The ideal carrier evolves under both complete
rows, including its nonzero phase acceleration at the initial contact.
No source acquisition, trajectory or finite-amplitude remainder is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I as Interval,
    cos,
    pi_interval,
    sin,
)
from ._organized_alternative_contact import _geometry
from ._organized_external_interaction import _OrganizedExternalInteractionGeometry
from ._sine_class_constitutive_comparison import _edge_law
from ._sine_preparation import _prepared_coordinate_vector as _vector
from ._sine_preparation import _prepared_form_costs
from .relational_sine_entry import _unit_laplacian_action

_Row = tuple[Interval, ...]


def _sum_rows(*rows: _Row) -> _Row:
    return tuple(sum(values, Interval(0)) for values in zip(*rows))


def _scaled(row: _Row, scalar: Q | Interval) -> _Row:
    return tuple(scalar * value for value in row)


def _mean(row: _Row) -> Interval:
    return sum(row[45:], Interval(0)) / 9


@dataclass(frozen=True)
class _ContactPreparationSensitivity:
    deformation: Q
    beta_turns: Q
    beta_bounds: Interval
    computed_time_order: int
    geometry: _OrganizedExternalInteractionGeometry
    form_direction: tuple[Q, ...]
    phase_direction: tuple[Q, ...]
    child_form_direction_means: tuple[Q, ...]
    child_phase_direction_means: tuple[Q, ...]
    internal_form_direction: tuple[Q, ...]
    internal_phase_direction: tuple[Q, ...]
    pure_internal_direction: bool
    linear_donor_identically_zero_via_slot_symmetry: bool
    reflection_permutation: tuple[int, ...] | None
    direction_preserves_reflection: bool
    donor_identically_zero_via_reflection: bool
    ideal_phase_turns: tuple[Q, ...]
    ideal_current_bounds: _Row
    edge_current_derivative_bounds: tuple[tuple[Interval, ...], ...]
    form_direction_laplacian: tuple[Q, ...]
    carrier_initial_phase_acceleration_bounds: _Row
    linear_form_time_derivative_bounds: tuple[_Row, ...]
    linear_phase_time_derivative_bounds: tuple[_Row, ...]
    quadratic_form_time_derivative_bounds: tuple[_Row, ...]
    quadratic_phase_time_derivative_bounds: tuple[_Row, ...]
    linear_donor_mean_time_derivative_bounds: tuple[Interval, ...]
    quadratic_donor_mean_time_derivative_bounds: tuple[Interval, ...]
    certified_quadratic_nonzero_time_orders: tuple[int, ...]
    contact_phase_gap_second_moments: tuple[Q, ...]
    contact_form_gap_second_moments: tuple[Q, ...]
    contact_phase_laplacian_form_covariances: tuple[Q, ...]
    contact_laplacian_form_gap_second_moments: tuple[Q, ...]
    closed_linear_donor_slope_bounds: Interval
    closed_quadratic_donor_slope_bounds: Interval
    initial_form_storage_quadratic_coefficient: Q
    contact_form_storage_quadratic_coefficient: Q
    initial_storage_amplitude_coefficients: tuple[Interval, Interval, Interval]
    contact_work_amplitude_coefficients: tuple[Interval, Interval, Interval]
    marked_phase_direction_gap_maximum: Q
    initial_marked_acute_margin_lower_bound: Q
    sufficient_initial_winding_amplitude_radius: Q
    maximum_time_order: int = 3
    maximum_amplitude_order: int = 2
    amplitude_convention: str = (
        "coefficient of epsilon^a; quadratic coefficient equals second amplitude derivative divided by two"
    )
    time_convention: str = "time derivatives at tau=0; not divided by time factorial"
    protocol_revision: str = "b2801170"
    arithmetic_method: str = INTERVAL_METHOD
    clock: str = (
        "tau=(1023/1024)*t; gamma=1/(1023*pi); both complete rows have quarter factor"
    )
    scope: tuple[str, ...] = (
        "all_54_signed_form_and_lifted_phase_directions_admitted_before_arithmetic",
        "exact_global_zero_sums_fix_form_total_and_phase_gauge",
        "beta_remains_a_supplied_preparation_origin_not_a_registration_scan",
        "complete_54_node_144_edge_quarter_rate_law_with_held_unit_capacities",
        "moving_carrier_phase_acceleration_retained_in_third_time_derivatives",
        "slot_symmetry_and_reflection_flags_use_exact_primitive_relations",
        "interval_containment_of_zero_is_not_an_exact_vanishing_claim",
        "initial_winding_radius_does_not_certify_later_winding_retention",
        "storage_and_attachment_amplitude_coefficients_are_not_finite_work_budgets",
        "no_acquired_source_residual_distribution_trajectory_or_finite_discriminator",
    )


def _observe_organized_contact_preparation_sensitivity(
    *,
    deformation: object,
    beta_turns: object,
    form_direction: object,
    phase_direction: object,
    time_order: object = 3,
) -> _ContactPreparationSensitivity:
    """Observe the fixed law's first two preparation variations through order 1..3.

    All inputs, including both zero-sum constraints, precede geometry and
    trigonometry. Rational phase turns retain the supplied continuous origin;
    exact modulo-one reduction is used only to evaluate periodic edge laws.
    """
    coefficient = _exact(deformation, "deformation")
    beta = _exact(beta_turns, "beta_turns")
    u = _vector(form_direction, "form_direction", 54)
    v = _vector(phase_direction, "phase_direction", 54)
    if coefficient not in (Q(0), Q(1)):
        raise ValueError("preparation sensitivity admits only deformation zero or one")
    if type(time_order) is not int or not 1 <= time_order <= 3:
        raise ValueError("time_order must be an ordinary integer in 1..3")
    if sum(u, Q(0)) != 0 or sum(v, Q(0)) != 0:
        raise ValueError("both full preparation directions must have exact zero sum")
    order: int = time_order  # type: ignore[assignment]
    pi = pi_interval()
    geometry = _geometry(2 * pi * beta, supplied=True)
    edges = tuple((a, b) for a, b, _ in geometry.weighted_edges)
    contact_start = len(geometry.internal_edges) + len(geometry.outer_edges)
    turns = tuple(
        Q(j - 4, 9) + (Q(c - 2, 5) if c < 5 else beta)
        for c in range(6)
        for j in range(9)
    )
    jets, potentials = [], []
    for a, b in edges:
        angle = 2 * pi * ((turns[b] - turns[a]) % 1)
        potential, derivatives = _edge_law(cos(angle), sin(angle), coefficient)
        potentials.append(potential)
        jets.append(derivatives)

    def laplacian(row: tuple) -> _Row:
        return tuple(
            Interval.coerce(value) for value in _unit_laplacian_action(row, edges)
        )

    def field_derivative(derivative: int, *rows: tuple) -> _Row:
        result = [Interval(0) for _ in range(54)]
        for (a, b), jet in zip(edges, jets):
            factor: Q | Interval = Q(1)
            for row in rows:
                factor *= row[b] - row[a]
            value = jet[derivative] * factor
            result[a] += value
            result[b] -= value
        return tuple(result)

    bfactor = geometry.gamma_bounds / 4
    s = field_derivative(0)
    acceleration = _scaled(laplacian(_scaled(s, bfactor)), bfactor)
    zero = (Interval(0),) * 54
    urow, vrow = tuple(Interval(value) for value in u), tuple(
        Interval(value) for value in v
    )
    lu = tuple(_unit_laplacian_action(u, edges))
    cu = _scaled(tuple(Interval(value) for value in lu), bfactor)
    z1 = _sum_rows(
        _scaled(tuple(Interval(value) for value in lu), Q(-1, 4)),
        _scaled(field_derivative(1, v), bfactor),
    )
    w1 = _scaled(field_derivative(2, v, v), bfactor / 2)
    linear_x, linear_phase = [urow, z1], [vrow, cu]
    quadratic_x, quadratic_phase = [zero, w1], [zero, zero]
    if order >= 2:
        cz1, cw1 = _scaled(laplacian(z1), bfactor), _scaled(laplacian(w1), bfactor)
        z2 = _sum_rows(
            _scaled(laplacian(z1), Q(-1, 4)), _scaled(field_derivative(1, cu), bfactor)
        )
        w2 = _sum_rows(
            _scaled(laplacian(w1), Q(-1, 4)),
            _scaled(field_derivative(2, v, cu), bfactor),
        )
        linear_x.append(z2)
        linear_phase.append(cz1)
        quadratic_x.append(w2)
        quadratic_phase.append(cw1)
    if order >= 3:
        z3 = _sum_rows(
            _scaled(laplacian(z2), Q(-1, 4)),
            _scaled(field_derivative(1, cz1), bfactor),
            _scaled(field_derivative(2, acceleration, v), bfactor),
        )
        w3 = _sum_rows(
            _scaled(laplacian(w2), Q(-1, 4)),
            _scaled(field_derivative(1, cw1), bfactor),
            _scaled(field_derivative(2, cu, cu), bfactor),
            _scaled(field_derivative(2, v, cz1), bfactor),
            _scaled(field_derivative(3, acceleration, v, v), bfactor / 2),
        )
        linear_x.append(z3)
        linear_phase.append(_scaled(laplacian(z2), bfactor))
        quadratic_x.append(w3)
        quadratic_phase.append(_scaled(laplacian(w2), bfactor))
    means_u = tuple(sum(u[9 * c : 9 * c + 9], Q(0)) / 9 for c in range(6))
    means_v = tuple(sum(v[9 * c : 9 * c + 9], Q(0)) / 9 for c in range(6))
    internal_u = tuple(value - means_u[i // 9] for i, value in enumerate(u))
    internal_v = tuple(value - means_v[i // 9] for i, value in enumerate(v))
    internal = not any(means_u + means_v)
    permutation = None
    reflected = False
    if (10 * beta).denominator == 1:
        reflection = (4 + int(10 * beta)) % 5
        permutation = tuple(
            9 * ((reflection - c) % 5 if c < 5 else 5) + 8 - j
            for c in range(6)
            for j in range(9)
        )
        reflected = all(
            u[i] == -u[p] and v[i] == -v[p] for i, p in enumerate(permutation)
        )

    def contact_covariance(left: tuple[Q, ...], right: tuple[Q, ...]) -> tuple[Q, ...]:
        return tuple(
            sum(
                (
                    (left[45 + j] - left[9 * c + j])
                    * (right[45 + j] - right[9 * c + j])
                    for j in range(9)
                ),
                Q(0),
            )
            / 9
            for c in range(5)
        )

    vv, uu = contact_covariance(v, v), contact_covariance(u, u)
    vlu, lulu = contact_covariance(v, lu), contact_covariance(lu, lu)
    contact_jets = tuple(jets[contact_start + 9 * c] for c in range(5))
    closed_linear = -Q(3, 2) * means_u[5] - bfactor * sum(
        (jet[1] * (means_v[5] - means_v[c]) for c, jet in enumerate(contact_jets)),
        Interval(0),
    )
    closed_quadratic = (
        -bfactor
        * sum((jet[2] * moment for jet, moment in zip(contact_jets, vv)), Interval(0))
        / 2
    )
    form_costs, _ = _prepared_form_costs(form=u, weighted_edges=geometry.weighted_edges)
    phase_linear = tuple(jet[0] * (v[b] - v[a]) for (a, b), jet in zip(edges, jets))
    phase_quadratic = tuple(
        jet[1] * (v[b] - v[a]) ** 2 / 2 for (a, b), jet in zip(edges, jets)
    )

    def storage(start: int) -> tuple[Interval, Interval, Interval]:
        return (
            sum(potentials[start:], Interval(0)),
            sum(phase_linear[start:], Interval(0)),
            sum(form_costs[start:], Q(0)) + sum(phase_quadratic[start:], Interval(0)),
        )

    phase_gap_max = max(
        (abs(v[b] - v[a]) for a, b in edges[:contact_start]), default=Q(0)
    )
    alpha, outer = 2 * pi / 9, 2 * pi / 5
    internal_margin = min(alpha.lo, pi.lo / 2 - alpha.hi)
    outer_margin = min(outer.lo, pi.lo / 2 - outer.hi)
    margin = min(internal_margin, outer_margin)
    radius = min(
        (Q(1),)
        + tuple(
            (internal_margin if i < len(geometry.internal_edges) else outer_margin)
            / (2 * abs(v[b] - v[a]))
            for i, (a, b) in enumerate(edges[:contact_start])
            if v[b] != v[a]
        )
    )
    linear_mean, quadratic_mean = tuple(map(_mean, linear_x)), tuple(
        map(_mean, quadratic_x)
    )
    return _ContactPreparationSensitivity(
        deformation=coefficient,
        beta_turns=beta,
        beta_bounds=2 * pi * beta,
        computed_time_order=order,
        geometry=geometry,
        form_direction=u,
        phase_direction=v,
        child_form_direction_means=means_u,
        child_phase_direction_means=means_v,
        internal_form_direction=internal_u,
        internal_phase_direction=internal_v,
        pure_internal_direction=internal,
        linear_donor_identically_zero_via_slot_symmetry=internal,
        reflection_permutation=permutation,
        direction_preserves_reflection=reflected,
        donor_identically_zero_via_reflection=reflected,
        ideal_phase_turns=turns,
        ideal_current_bounds=s,
        edge_current_derivative_bounds=tuple(jets),
        form_direction_laplacian=lu,
        carrier_initial_phase_acceleration_bounds=acceleration,
        linear_form_time_derivative_bounds=tuple(linear_x),
        linear_phase_time_derivative_bounds=tuple(linear_phase),
        quadratic_form_time_derivative_bounds=tuple(quadratic_x),
        quadratic_phase_time_derivative_bounds=tuple(quadratic_phase),
        linear_donor_mean_time_derivative_bounds=linear_mean,
        quadratic_donor_mean_time_derivative_bounds=quadratic_mean,
        certified_quadratic_nonzero_time_orders=tuple(
            i for i, value in enumerate(quadratic_mean) if value.lo > 0 or value.hi < 0
        ),
        contact_phase_gap_second_moments=vv,
        contact_form_gap_second_moments=uu,
        contact_phase_laplacian_form_covariances=vlu,
        contact_laplacian_form_gap_second_moments=lulu,
        closed_linear_donor_slope_bounds=closed_linear,
        closed_quadratic_donor_slope_bounds=closed_quadratic,
        initial_form_storage_quadratic_coefficient=sum(form_costs, Q(0)),
        contact_form_storage_quadratic_coefficient=sum(
            form_costs[contact_start:], Q(0)
        ),
        initial_storage_amplitude_coefficients=storage(0),
        contact_work_amplitude_coefficients=storage(contact_start),
        marked_phase_direction_gap_maximum=phase_gap_max,
        initial_marked_acute_margin_lower_bound=margin,
        sufficient_initial_winding_amplitude_radius=radius,
    )
