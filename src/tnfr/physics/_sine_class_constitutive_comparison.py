"""Static comparison of complete laws with matched C9 target derivatives.

The alternative periodic potential is
V_lambda(delta)=1-cos(delta)+lambda*(1-cos(9*delta))**2/(6*9**4).
Its derivative supplies the alternative edge current. These detached observers
install no law and supply no trajectory, finite response or acquisition bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._interval_taylor import MAX_ORDER
from ..mathematics._rational_interval import I
from ._sine_class_neighbor_nonadditivity import _derive_neighbor_onset, _NeighborOnset
from .phase_response import _admit_relative_phasors
from .relational_observations import _ordered
from .relational_sine_class_mediation import _EDGES, _NODES
from .relational_sine_port_composition import _parameters


def _deformation(value):
    coefficient = _exact(value, "deformation")
    if coefficient < 0:
        raise ValueError("deformation must be nonnegative")
    return coefficient


def _edge_law(cosine, sine, coefficient):
    """Exact potential and current derivatives on an admitted unit phasor."""
    c3, s3 = 4 * cosine**3 - 3 * cosine, sine * (4 * cosine**2 - 1)
    c9, s9 = 4 * c3**3 - 3 * c3, s3 * (4 * c3**2 - 1)
    c18, s18 = c9**2 - s9**2, 2 * c9 * s9
    potential = 1 - cosine + coefficient * (1 - c9) ** 2 / (6 * 9**4)
    derivatives = (
        sine + coefficient * s9 * (1 - c9) / 2187,
        cosine + coefficient * (c9 - c18) / 243,
        -sine + coefficient * (-s9 + 2 * s18) / 27,
        -cosine + coefficient * (-c9 + 4 * c18) / 3,
    )
    return potential, derivatives


def _current_derivative_bound(deformation, derivative_order):
    """Global current derivative bound; order 0..16 is a computation policy."""
    coefficient = _deformation(deformation)
    if type(derivative_order) is not int or not 0 <= derivative_order <= MAX_ORDER:
        raise ValueError("derivative order must be an ordinary integer in 0..16")
    power = derivative_order + 1
    return 1 + coefficient * (2 * 9**power + Q(18**power, 2)) / 39366


@dataclass(frozen=True)
class _ConstitutiveState:
    """Instantaneous full27 observation in the fixed structural tau clock.

    Node phasors specify circular phases; no continuous lifts or lifted phase
    means are inferred. Gamma remains mathematical 1/(1023*pi), enclosed by I.
    All potential/current values and the positive loss are exact rationals for
    the supplied unit phasors. Rate intervals enclose that fixed complete law.
    """

    form: tuple[Q, ...]
    phase_phasors: tuple[tuple[Q, Q], ...]
    deformation: Q
    degrees: tuple[int, ...]
    gamma_bounds: I
    relative_phasors: tuple[tuple[Q, Q], ...]
    edge_potential_values: tuple[Q, ...]
    edge_current_derivatives: tuple[tuple[Q, ...], ...]
    form_gradient: tuple[Q, ...]
    phase_current_sums: tuple[Q, ...]
    form_rate_bounds: tuple[I, ...]
    phase_rate_bounds: tuple[I, ...]
    form_storage: Q
    phase_storage: Q
    storage: Q
    loss: Q
    storage_rate: Q
    storage_rate_from_rows_bounds: I
    weighted_form_charge: Q
    weighted_charge_rate_bounds: tuple[I, I]
    nodes: tuple[int, ...] = _NODES
    edges: tuple[tuple[int, int], ...] = _EDGES
    clock: str = "tau=e*t; e=1023/1024"
    capacity: tuple[Q, ...] = (Q(1),) * 27
    scope: tuple[str, ...] = (
        "fixed_joined_three_C9_support_with_actual_degrees_and_both_nodal_rows",
        "alternative_current_is_the_derivative_of_its_own_declared_potential",
        "x_tau=-D_inverse_Lx+gamma_D_inverse_S_lambda;theta_tau=gamma_D_inverse_Lx",
        "all_27_exact_unit_phasors_come_from_one_supplied_global_phase_assignment",
        "no_phase_lift_invention_state_preparation_or_law_installation",
        "storage_rate_identity_is_not_a_finite_response_or_acquisition_certificate",
        "sine_specific_remainder_and_source_bounds_are_not_transferred",
    )


def _observe_constitutive_state(*, form, phase_phasors, deformation):
    """Re-admit a complete primitive state before computing the alternative law."""
    coefficient = _deformation(deformation)
    raw_form = _ordered(form, "form", limit=28)
    raw_phasors = _ordered(phase_phasors, "phase_phasors", limit=28)
    if len(raw_form) != 27 or len(raw_phasors) != 27:
        raise ValueError("the fixed support requires 27 forms and 27 node phasors")
    values = tuple(_exact(value, f"form[{i}]") for i, value in enumerate(raw_form))
    phases = _admit_relative_phasors(raw_phasors, "phase_phasors")
    gamma, _ = _parameters(())
    degrees = tuple(sum(node in edge for edge in _EDGES) for node in _NODES)
    gradient, currents = [Q(0)] * 27, [Q(0)] * 27
    relative, potentials, derivatives = [], [], []
    form_storage = Q(0)
    for i, j in _EDGES:
        ci, si = phases[i]
        cj, sj = phases[j]
        gap = ci * cj + si * sj, ci * sj - si * cj
        potential, jet = _edge_law(*gap, coefficient)
        relative.append(gap)
        potentials.append(potential)
        derivatives.append(jet)
        difference = values[i] - values[j]
        gradient[i] += difference
        gradient[j] -= difference
        currents[i] += jet[0]
        currents[j] -= jet[0]
        form_storage += difference**2 / 2
    form_rates = tuple(
        -q / degree + gamma * current / degree
        for q, current, degree in zip(gradient, currents, degrees)
    )
    phase_rates = tuple(gamma * q / degree for q, degree in zip(gradient, degrees))
    loss = sum((q**2 / degree for q, degree in zip(gradient, degrees)), Q(0))
    rate_from_rows = sum(
        (
            q * dx - current * dy
            for q, current, dx, dy in zip(gradient, currents, form_rates, phase_rates)
        ),
        I(0),
    )
    phase_storage = sum(potentials, Q(0))
    return _ConstitutiveState(
        form=values,
        phase_phasors=phases,
        deformation=coefficient,
        degrees=degrees,
        gamma_bounds=gamma,
        relative_phasors=tuple(relative),
        edge_potential_values=tuple(potentials),
        edge_current_derivatives=tuple(derivatives),
        form_gradient=tuple(gradient),
        phase_current_sums=tuple(currents),
        form_rate_bounds=form_rates,
        phase_rate_bounds=phase_rates,
        form_storage=form_storage,
        phase_storage=phase_storage,
        storage=form_storage + phase_storage,
        loss=loss,
        storage_rate=-loss,
        storage_rate_from_rows_bounds=rate_from_rows,
        weighted_form_charge=sum((d * x for d, x in zip(degrees, values)), Q(0)),
        weighted_charge_rate_bounds=(
            -sum(gradient, Q(0)) + gamma * sum(currents, Q(0)),
            gamma * sum(gradient, Q(0)),
        ),
    )


@dataclass(frozen=True)
class _ConstitutiveOnset:
    """Local mixed time-four coefficient, without any finite-time error claim.

    At every target 2*pi*m/9 the potential perturbation has derivatives zero
    through order three and fourth derivative lambda. Thus target potential,
    current, complete tangent and quadratic field agree; cubic fields need not.
    Exact amplitude factors are retained before interval materialization.
    """

    geometry: _NeighborOnset
    deformation: Q
    donor_amplitude: Q
    receiver_amplitude: Q
    target_potential_derivative_differences: tuple[Q, ...]
    target_current_third_derivative_bounds: tuple[I, ...]
    current_derivative_upper_bounds: tuple[Q, ...]
    common_mixed_amplitude_factor: Q
    normalized_mixed_coefficient_bounds: tuple[Q, Q]
    mixed_time_four_coefficient_bounds: tuple[Q, Q]
    deformation_time_four_coefficient_bounds: tuple[Q, Q]
    time_degree: int = 4
    amplitude_degree: int = 3
    scope: tuple[str, ...] = (
        "nominal_target_source_with_simultaneous_form_inputs_at_nodes_4_and_22",
        "mediator_node_13_mixed_four_history_observation",
        "full_target_storage_equilibrium_tangent_and_quadratic_field_match",
        "higher_edge_derivatives_remain_a_supplied_constitutive_choice",
        "no_finite_horizon_sign_readout_source_or_acquisition_certificate",
        "zero_onset_coefficient_does_not_assert_an_exact_mixed_response_null",
    )


def _derive_constitutive_onset(
    *, mediator_class, deformation, donor_amplitude, receiver_amplitude
):
    """Derive the actual mixed edge weights after admitting every primitive."""
    coefficient = _deformation(deformation)
    donor = _exact(donor_amplitude, "donor_amplitude")
    receiver = _exact(receiver_amplitude, "receiver_amplitude")
    geometry = _derive_neighbor_onset(mediator_class)
    third = tuple(coefficient - cosine for cosine in geometry.channel_cosine_bounds)
    # The old exact channel factors multiply cos(delta)=-j_sine'''(delta).
    # Its two mixed monomials have the same weight by end-port reflection.
    lower = upper = difference = Q(0)
    for row, cosine in zip(
        geometry.mixed_channel_factors, geometry.channel_cosine_bounds
    ):
        if row[0] != row[1]:
            raise ArithmeticError("fixed end-port reflection coefficient mismatch")
        products = row[0] * (cosine.lo - coefficient), row[0] * (
            cosine.hi - coefficient
        )
        lower += min(products)
        upper += max(products)
        difference -= row[0] * coefficient
    amplitude = donor * receiver * (donor + receiver)
    products = tuple(
        gamma**4 * amplitude * value
        for gamma in (geometry.gamma_bounds.lo, geometry.gamma_bounds.hi)
        for value in (lower, upper)
    )
    changes = tuple(
        gamma**4 * amplitude * difference
        for gamma in (geometry.gamma_bounds.lo, geometry.gamma_bounds.hi)
    )
    return _ConstitutiveOnset(
        geometry=geometry,
        deformation=coefficient,
        donor_amplitude=donor,
        receiver_amplitude=receiver,
        target_potential_derivative_differences=(Q(0),) * 4 + (coefficient,),
        target_current_third_derivative_bounds=third,
        current_derivative_upper_bounds=tuple(
            _current_derivative_bound(coefficient, order) for order in range(4)
        ),
        common_mixed_amplitude_factor=amplitude,
        normalized_mixed_coefficient_bounds=(lower, upper),
        mixed_time_four_coefficient_bounds=(min(products), max(products)),
        deformation_time_four_coefficient_bounds=(min(changes), max(changes)),
    )
