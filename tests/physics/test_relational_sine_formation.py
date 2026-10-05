"""Independent static controls for mediated sine formation obstructions.

Necessary budgets, an initial symmetry-breaking response and a finite-time
analytic exclusion are distinct. No trajectory or reserved producer is run.
"""

import json
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction as Q
from math import inf, nextafter

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics.relational_sine_forecast import _sine_flow
from tnfr.physics.relational_sine_formation import (
    SineReceiverTransferAdmission,
    _maintained_target_obstruction,
    assess_sine_mediated_formation,
)
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_recovery import certify_sine_pattern_recovery
from tnfr.sdk import export_to_json

MODEL = RelationalExchangeModel(1, phase_domain="regular")
CANDIDATE_DONOR = tuple(Q(value, 3) for value in (10, 13, 14, 14, 13))


def _stored_model():
    model = RelationalExchangeModel(1, phase_domain="regular")
    for field, value in (
        ("epi_weight", Q(7, 19)),
        ("phase_weight", Q(5, 17)),
        ("storage_scale", Q(11, 13)),
    ):
        object.__setattr__(model, field, value)
    return model


def _report(amplitude=2, contrast=Q(1, 2), **options):
    options.setdefault("model", MODEL)
    return assess_sine_mediated_formation(
        amplitude=amplitude, capacity_contrast=contrast, **options
    )


def _explicit(donor=CANDIDATE_DONOR, hidden=Q(4, 3), **options):
    return _report(hidden, profile="explicit", donor_epi=donor, **options)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    if bound.contains(0) and abs(value) < mp.mpf("1e-80"):
        return  # Independently evaluated trigonometric cancellation at 90 dps.
    assert _mp(bound.lo) <= value <= _mp(bound.hi)


def _support():
    edges = [(offset + j, offset + (j + 1) % 5) for offset in (0, 5) for j in range(5)]
    edges += [(0, 10), (5, 10)]
    neighbors = tuple(
        tuple(j if i == node else i for i, j in edges if node in (i, j))
        for node in range(11)
    )
    return edges, neighbors


def _full_rows(form, phase, capacities, model=MODEL):
    """Independent edge equations and their first directional derivative."""
    edges, neighbors = _support()
    e, w = map(_mp, model.effective_weights)
    beta = _mp(model.storage_scale)
    a, b = w / mp.pi, w / (beta * mp.pi)
    q = [sum(form[i] - form[j] for j in row) for i, row in enumerate(neighbors)]
    currents = [
        sum(mp.sin(phase[j] - phase[i]) for j in row) for i, row in enumerate(neighbors)
    ]
    mobility = [capacities[i] / len(row) for i, row in enumerate(neighbors)]
    velocity = [mobility[i] * (-e * q[i] + a * currents[i]) for i in range(11)]
    omega = [b * mobility[i] * q[i] for i in range(11)]
    q_dot = [
        sum(velocity[i] - velocity[j] for j in row) for i, row in enumerate(neighbors)
    ]
    current_dot = [
        sum(mp.cos(phase[j] - phase[i]) * (omega[j] - omega[i]) for j in row)
        for i, row in enumerate(neighbors)
    ]
    acceleration = [
        mobility[i] * (-e * q_dot[i] + a * current_dot[i]) for i in range(11)
    ]
    phase_acceleration = [b * mobility[i] * q_dot[i] for i in range(11)]
    storage = sum(
        (form[j] - form[i]) ** 2 / 2 + beta * (1 - mp.cos(phase[j] - phase[i]))
        for i, j in edges
    )
    storage_rate = sum(
        (form[j] - form[i]) * (velocity[j] - velocity[i])
        + beta * mp.sin(phase[j] - phase[i]) * (omega[j] - omega[i])
        for i, j in edges
    )
    return velocity, omega, acceleration, phase_acceleration, storage, storage_rate


@pytest.mark.parametrize(
    "model",
    (MODEL, RelationalExchangeModel(2, phase_domain="regular"), _stored_model()),
)
@pytest.mark.parametrize("profile", ("localized", "balanced"))
def test_complete_edge_equations_initial_jets_and_storage_match_independent_oracle(
    model, profile
):
    report = _report(amplitude=Q(-7, 3), contrast=Q(2, 5), model=model, profile=profile)
    edges, neighbors = _support()
    assert set(map(frozenset, report.edges)) == set(map(frozenset, edges))
    assert report.nodes == tuple(range(11))
    assert report.degrees == tuple(map(len, neighbors))
    donor = Q(-14, 3) if profile == "balanced" else Q(0)
    assert report.initial_epi == (donor,) * 5 + (Q(0),) * 5 + (Q(-7, 3),)
    assert report.capacity[6] == Q(7, 5) and report.capacity[9] == Q(3, 5)
    turns = tuple(Q(j, 5) for j in range(5)) + (Q(0),) * 6
    assert report.initial_phase_turns == turns
    assert not any(report.initial_geometry.symbolic_sine_coefficients)
    state = (
        tuple(map(I, report.initial_epi))
        + tuple(2 * value * pi_interval() for value in turns)
        + (I(1),)
    )
    series = flow_jets(
        state,
        2,
        lambda values: _sine_flow(
            values,
            neighbors=neighbors,
            visible_capacity=report.capacity[:-1],
            model=model,
        ),
    )
    with mp.workdps(90):
        form = tuple(map(_mp, report.initial_epi))
        phase = tuple(2 * mp.pi * _mp(value) for value in turns)
        rates, omega, acceleration, phase_acceleration, energy, energy_dot = _full_rows(
            form, phase, tuple(map(_mp, report.capacity)), model
        )
        for i in range(11):
            _contains(report.initial_form_rate_bounds[i], rates[i])
            _contains(report.initial_phase_rate_bounds[i], omega[i])
            _contains(series[i][1], rates[i])
            _contains(series[11 + i][1], omega[i])
            _contains(2 * series[i][2], acceleration[i])
            _contains(2 * series[11 + i][2], phase_acceleration[i])
        odd = phase_acceleration[6] - phase_acceleration[9]
        _contains(report.phase_odd_acceleration_bounds, odd)
        assert mp.almosteq(odd, _mp(report.phase_odd_acceleration_pi_numerator) / mp.pi)
        _contains(report.initial_storage_bounds, energy)
        assert mp.almosteq(energy_dot, _mp(report.initial_storage_rate))
        loss_factor = 2 if profile == "balanced" else 8
        assert mp.almosteq(
            -energy_dot, loss_factor * _mp(model.epi_weight) * form[10] ** 2 / 3
        )
        assert report.initial_balance_residual_bounds.contains(0)


def test_balanced_preparation_reduces_loss_at_matched_energy_without_seeding_receiver():
    localized, balanced = (
        _report(profile=profile) for profile in ("localized", "balanced")
    )
    assert localized.initial_form_storage == balanced.initial_form_storage == 4
    assert localized.initial_storage_bounds == balanced.initial_storage_bounds
    assert localized.entry_storage_margin_bounds == balanced.entry_storage_margin_bounds
    assert localized.initial_phase_turns == balanced.initial_phase_turns
    assert localized.capacity == balanced.capacity
    assert balanced.initial_epi[5:10] == (Q(0),) * 5
    assert balanced.initial_phase_turns[5:10] == (Q(0),) * 5
    assert (
        balanced.initial_form_gradient == (Q(2),) + (Q(0),) * 4 + (Q(-2),) + (Q(0),) * 5
    )
    assert localized.initial_continuous_loss == 4 * balanced.initial_continuous_loss
    assert balanced.initial_continuous_loss == Q(4, 3)
    assert (
        localized.initial_form_rate_bounds[5:10]
        == balanced.initial_form_rate_bounds[5:10]
    )
    assert (
        localized.initial_phase_rate_bounds[5:10]
        == balanced.initial_phase_rate_bounds[5:10]
    )
    assert (
        localized.phase_odd_acceleration_pi_numerator
        == balanced.phase_odd_acceleration_pi_numerator
    )
    assert (
        balanced.initial_form_rate_bounds[10]
        == balanced.initial_phase_rate_bounds[10]
        == I(0)
    )
    assert localized.initial_form_rate_bounds[10].hi < 0
    assert (
        balanced.initial_form_rate_bounds[0].hi
        < 0
        < localized.initial_form_rate_bounds[0].lo
    )
    with mp.workdps(90):
        phase = tuple(2 * mp.pi * _mp(value) for value in balanced.initial_phase_turns)
        capacities = tuple(map(_mp, balanced.capacity))
        local_rows = _full_rows(
            tuple(map(_mp, localized.initial_epi)), phase, capacities
        )
        balanced_rows = _full_rows(
            tuple(map(_mp, balanced.initial_epi)), phase, capacities
        )
        # The neighboring odd response agrees, but the port's second-order
        # response changes: equal initial rates do not imply equal futures.
        assert mp.almosteq(
            local_rows[3][6] - local_rows[3][9],
            balanced_rows[3][6] - balanced_rows[3][9],
        )
        assert mp.almosteq(local_rows[3][5], 2 * balanced_rows[3][5])
        assert abs(balanced_rows[3][5]) > 0
        _, neighbors = _support()
        gradients = tuple(map(_mp, balanced.initial_form_gradient))
        gradient_rates = tuple(
            sum(balanced_rows[0][i] - balanced_rows[0][j] for j in row)
            for i, row in enumerate(neighbors)
        )
        loss_rate = (
            2
            * _mp(MODEL.epi_weight)
            * sum(
                capacities[i] * gradients[i] * gradient_rates[i] / len(row)
                for i, row in enumerate(neighbors)
            )
        )
        assert mp.almosteq(
            loss_rate,
            -2 * _mp(MODEL.epi_weight) * _mp(balanced.initial_continuous_loss),
        )


def test_balanced_initial_loss_minimum_is_only_within_the_declared_flat_region_family():
    edges, neighbors = _support()
    capacity = (Q(1),) * 6 + (Q(3, 2), Q(1), Q(1), Q(1, 2), Q(1))
    # All four independently chosen (s=D-H, r=H) pairs have F=4.
    for s, r in (
        (Q(2), Q(2)),
        (Q(-2), Q(-2)),
        (Q(14, 5), Q(2, 5)),
        (Q(2, 5), Q(14, 5)),
    ):
        form = (s + r,) * 5 + (Q(0),) * 5 + (r,)
        energy = sum((form[j] - form[i]) ** 2 / 2 for i, j in edges)
        gradients = tuple(
            sum(form[i] - form[j] for j in row) for i, row in enumerate(neighbors)
        )
        loss = Q(MODEL.epi_weight) * sum(
            capacity[i] * gradients[i] ** 2 / len(row)
            for i, row in enumerate(neighbors)
        )
        assert energy == 4
        assert loss / Q(MODEL.epi_weight) == 2 * energy / 3 + (r - s) ** 2 / 2
        assert loss >= Q(4, 3)
        assert (loss == Q(4, 3)) == (s == r)


def test_two_profile_span_has_exact_orthogonal_moments_for_every_capacity_contrast():
    edges, neighbors = _support()
    laplacian = [[Q(0) for _ in range(11)] for _ in range(11)]
    for i, j in edges:
        laplacian[i][i] += 1
        laplacian[j][j] += 1
        laplacian[i][j] -= 1
        laplacian[j][i] -= 1

    def apply(values):
        return tuple(sum(a * b for a, b in zip(row, values)) for row in laplacian)

    # K(delta)=K0+delta*K1. The two profile gradients vanish at
    # both contrast nodes, so Kq has no delta term. Checking every
    # coefficient below covers the whole admitted contrast interval.
    mobility0 = tuple(Q(1, len(row)) for row in neighbors)
    mobility1 = tuple(
        Q(1, 2) if i == 6 else Q(-1, 2) if i == 9 else Q(0) for i in range(11)
    )
    forms = ((Q(2),) * 5 + (Q(0),) * 5 + (Q(1),), (Q(0),) * 10 + (Q(1),))
    gradients = tuple(map(apply, forms))
    assert gradients == (
        (Q(1),) + (Q(0),) * 4 + (Q(-1),) + (Q(0),) * 5,
        (Q(-1),) + (Q(0),) * 4 + (Q(-1),) + (Q(0),) * 4 + (Q(2),),
    )
    assert all(k * q == 0 for qrow in gradients for k, q in zip(mobility1, qrow))
    vectors = tuple(tuple(k * q for k, q in zip(mobility0, row)) for row in gradients)
    pushed = tuple(map(apply, vectors))
    for i in range(2):
        for j in range(2):
            n = sum(
                k * qi * qj for k, qi, qj in zip(mobility0, gradients[i], gradients[j])
            )
            n_slope = sum(
                k * qi * qj for k, qi, qj in zip(mobility1, gradients[i], gradients[j])
            )
            first = sum(
                (vectors[i][a] - vectors[i][b]) * (vectors[j][a] - vectors[j][b])
                for a, b in edges
            )
            second = sum(
                k * vi * vj for k, vi, vj in zip(mobility0, pushed[i], pushed[j])
            )
            second_slope = sum(
                k * vi * vj for k, vi, vj in zip(mobility1, pushed[i], pushed[j])
            )
            assert n_slope == second_slope == 0
            if i != j:
                assert n == first == second == 0
            elif i == 0:
                assert (n, first, second) == (Q(2, 3), Q(2, 3), Q(8, 9))
            else:
                assert (n, first, second) == (Q(8, 3), Q(4), Q(58, 9))


@pytest.mark.parametrize(
    "amplitude,contrast", ((2, Q(1, 2)), (-2, Q(1, 2)), (2, Q(-1, 2)), (-2, Q(-1, 2)))
)
def test_exact_odd_response_preserves_each_sign_reversal(amplitude, contrast):
    report = _report(amplitude, contrast)
    assert report.phase_odd_acceleration_pi_numerator == -Q(amplitude) * contrast / 12
    assert report.phase_odd_acceleration_sign == (-1 if amplitude * contrast > 0 else 1)
    assert report.initial_form_storage == 4
    assert report.initial_continuous_loss == Q(16, 3)


def test_tiny_nonzero_contrast_is_not_a_reflection_or_a_zero_response():
    tiny = Q(1, 2**200)
    report = _report(2, tiny)
    assert report.capacity[6] - report.capacity[9] == 2 * tiny
    assert report.phase_odd_acceleration_sign == -1
    assert report.phase_odd_acceleration_pi_numerator == -tiny / 6
    assert report.phase_odd_acceleration_bounds.contains(0)
    assert not report.reflection_invariant
    assert "exact_receiver_reflection" not in report.exclusion_reasons


def test_reflection_and_stationary_preparation_are_separate_exact_obstructions():
    reflected = _report(10, 0)
    assert reflected.status == "excluded"
    assert reflected.reflection_invariant and not reflected.initial_equilibrium
    assert reflected.target_budget_status == reflected.entry_budget_status == "passed"
    assert reflected.exclusion_reasons == ("exact_receiver_reflection",)
    stationary = _report(0, Q(1, 2))
    assert stationary.status == "excluded" and stationary.initial_equilibrium
    assert not stationary.reflection_invariant
    assert stationary.initial_continuous_loss == 0
    assert all(value == I(0) for value in stationary.initial_form_rate_bounds)
    assert all(value == I(0) for value in stationary.initial_phase_rate_bounds)


def test_static_positive_checks_are_explicitly_insufficient_for_formation():
    report = _report(3)
    assert report.status == "passes_necessary_conditions"
    assert report.passes_necessary_conditions and report.formation_unresolved
    assert report.exclusion_reasons == report.unresolved_conditions == ()
    assert report.exclusion_time is report.early_loss_lower_bound is None
    assert not report.early_loss_exclusion_certified
    assert (
        "necessary_checks_are_not_sufficient_formation_or_a_time_to_capture"
        in report.scope
    )


def test_joint_entry_barrier_excludes_a_preparation_that_passes_target_energy():
    report = _report(Q(931, 500))
    assert report.target_storage_margin_bounds.lo > 0
    assert report.entry_storage_margin_bounds.hi < 0
    assert report.target_budget_status == "passed"
    assert report.entry_budget_status == report.status == "excluded"
    assert report.exclusion_reasons == (
        "strict_joint_acute_entry_storage_requirement",
        "maintained_target_lyapunov_obstruction",
    )
    assert not report.formation_unresolved
    assert report.phase_action_bound.necessary_entry_time_lower_bound is None
    assert report.phase_action_bound.unavailable_reasons == (
        "nonpositive_loss_allowance",
    )


def test_unresolved_entry_boundary_remains_distinct_from_maintenance_exclusion():
    with mp.workdps(110):
        near_boundary = Q(mp.nstr(mp.sqrt(5 - 4 * mp.cos(3 * mp.pi / 8)), 105))
    report = _report(near_boundary)
    assert report.entry_storage_margin_bounds.contains(0)
    assert report.target_budget_status == "passed"
    assert report.status == "excluded"
    assert not report.passes_necessary_conditions
    assert report.exclusion_reasons == ("maintained_target_lyapunov_obstruction",)
    assert report.unresolved_conditions == (
        "joint_acute_entry_storage_sign_unresolved",
    )
    assert not report.phase_action_bound.available
    assert report.phase_action_bound.necessary_entry_time_lower_bound is None
    assert report.phase_action_bound.unavailable_reasons == (
        "positive_loss_allowance_unresolved",
    )


@pytest.mark.parametrize(
    "model, amplitude, contrast",
    (
        (MODEL, Q(2), Q(1, 2)),
        (
            RelationalExchangeModel(
                2, epi_weight=3, phase_weight=1, phase_domain="regular"
            ),
            Q(3),
            Q(-2, 5),
        ),
    ),
)
def test_phase_action_uses_the_actual_receiver_and_independent_storage_allowance(
    model, amplitude, contrast
):
    report = _report(amplitude, contrast, model=model)
    action = report.phase_action_bound
    _, neighbors = _support()
    capacity = tuple(
        1 + contrast if i == 6 else 1 - contrast if i == 9 else Q(1) for i in range(11)
    )
    receiver = tuple(range(5, 10))
    maximum = max(
        capacity[i] / len(neighbors[i]) + capacity[j] / len(neighbors[j])
        for i, j in zip(receiver, receiver[1:] + receiver[:1])
    )
    assert action.receiver_edge_mobility_max == maximum
    assert action.allowable_loss_bounds is report.entry_storage_margin_bounds
    assert action.available and not action.unavailable_reasons
    assert action.horizon is None and not action.entry_through_horizon_excluded
    # The time bound never asserts formation. Another independent certificate
    # can still exclude the maintained target under its narrower law premises.
    assert not report.early_loss_exclusion_certified
    with mp.workdps(100):
        e, w = map(_mp, model.effective_weights)
        beta = _mp(model.storage_scale)
        allowance = _mp(amplitude) ** 2 - beta * (5 - 4 * mp.cos(3 * mp.pi / 8))
        b = w / (beta * mp.pi)
        exact_threshold = e * mp.pi**2 / (b**2 * _mp(maximum) * allowance)
        lower = _mp(action.necessary_entry_time_lower_bound)
        assert 0 < lower <= exact_threshold
        # Reuse the existing barrier precision; the 128-bit interval display
        # does not strengthen the underlying face-cost enclosure.
        assert abs(lower / exact_threshold - 1) < mp.mpf("1e-15")
    if model is MODEL:
        assert maximum == Q(5, 4)
        assert action.necessary_entry_time_lower_bound > 240


def test_phase_action_through_horizon_uses_strict_admission_and_keeps_tiny_coefficients():
    lower = _report().phase_action_bound.necessary_entry_time_lower_bound
    earlier = _report(exclusion_time=lower / 2).phase_action_bound
    boundary = _report(exclusion_time=lower).phase_action_bound
    assert earlier.entry_through_horizon_excluded
    assert boundary.available and not boundary.entry_through_horizon_excluded
    assert (
        earlier.necessary_entry_time_lower_bound
        == boundary.necessary_entry_time_lower_bound
    )

    model = RelationalExchangeModel(
        1, epi_weight=Q(1, 2**200), phase_weight=1, phase_domain="regular"
    )
    tiny = _report(model=model).phase_action_bound
    assert tiny.available
    assert 0 < tiny.necessary_entry_time_lower_bound < Q(1, 2**128)


def test_zero_upper_allowance_does_not_manufacture_an_infinite_entry_time():
    model = RelationalExchangeModel(Q(1, 2**200), phase_domain="regular")
    report = _report(0, model=model)
    action = report.phase_action_bound
    assert action.allowable_loss_bounds.hi == 0
    assert not action.available
    assert action.necessary_entry_time_lower_bound is None
    assert not action.entry_through_horizon_excluded
    assert action.unavailable_reasons == ("nonpositive_loss_allowance",)
    assert report.status == "excluded"


@pytest.mark.parametrize("profile", ("localized", "balanced", "explicit"))
def test_maintained_target_obstruction_uses_full_preparation_not_a_timed_entry_claim(
    profile,
):
    report = _explicit() if profile == "explicit" else _report(profile=profile)
    obstruction = report.maintained_target_obstruction
    assert obstruction.status == "excluded"
    assert obstruction.maintained_target_excluded
    assert obstruction.form_storage_coefficient == Q(4, 5)
    assert obstruction.exchange_to_loss_ratio == 1
    assert obstruction.sufficient_ratio_upper_bound == Q(3, 2)
    assert obstruction.mobility_spectral_bounds == (Q(1, 44), Q(3))
    assert not obstruction.unavailable_reasons
    assert obstruction.target_margin_bounds.lo > Q(1, 4)
    assert report.target_budget_status == report.entry_budget_status == "passed"
    assert not report.early_loss_exclusion_certified
    assert report.exclusion_reasons == ("maintained_target_lyapunov_obstruction",)
    assert not report.formation_unresolved
    with mp.workdps(100):
        edges, _ = _support()
        forms = tuple(map(_mp, report.initial_epi))
        form_storage = sum((forms[j] - forms[i]) ** 2 / 2 for i, j in edges)
        twist = 5 * (1 - mp.cos(2 * mp.pi / 5))
        _contains(
            obstruction.initial_functional_bounds, mp.mpf(4) * form_storage / 5 + twist
        )
        _contains(obstruction.target_functional_bounds, 2 * twist)
        _contains(
            obstruction.target_margin_bounds, twist - mp.mpf(4) * form_storage / 5
        )
        _contains(obstruction.mixed_term_coefficient_bounds, 1 / (2 * mp.pi))


@pytest.mark.parametrize(
    "model, contrast, reason",
    (
        (
            RelationalExchangeModel(
                1, epi_weight=1, phase_weight=3, phase_domain="regular"
            ),
            Q(1, 2),
            "requires_exchange_to_loss_ratio_strictly_between_zero_and_three_halves",
        ),
        (
            RelationalExchangeModel(2, phase_domain="regular"),
            Q(1, 2),
            "requires_unit_storage_scale",
        ),
        (MODEL, Q(-1, 2), "requires_positive_half_capacity_contrast"),
        (MODEL, Q(1, 2) + Q(1, 2**200), "requires_positive_half_capacity_contrast"),
    ),
)
def test_maintained_target_proof_does_not_transfer_outside_exact_premises(
    model, contrast, reason
):
    report = _report(model=model, contrast=contrast)
    obstruction = report.maintained_target_obstruction
    assert obstruction.status == "unavailable"
    assert obstruction.unavailable_reasons == (reason,)
    assert not obstruction.maintained_target_excluded
    assert obstruction.form_storage_coefficient is None
    assert obstruction.mixed_term_coefficient_bounds is None
    assert obstruction.mobility_spectral_bounds is None
    assert obstruction.initial_functional_bounds is None
    assert obstruction.target_functional_bounds is None
    assert obstruction.target_margin_bounds is None
    e, w = map(Q, model.effective_weights)
    assert obstruction.exchange_to_loss_ratio == w / e
    assert obstruction.sufficient_ratio_upper_bound == Q(3, 2)
    assert "maintained_target_lyapunov_obstruction" not in report.exclusion_reasons


@pytest.mark.parametrize("raw_weights", ((3, 2), (1, Q(1, 2**200))))
def test_maintained_target_ratio_family_keeps_correlated_gap_without_a_horizon(
    raw_weights,
):
    model = RelationalExchangeModel(
        1,
        epi_weight=raw_weights[0],
        phase_weight=raw_weights[1],
        phase_domain="regular",
    )
    report = _explicit(model=model)
    obstruction = report.maintained_target_obstruction
    reference = _explicit().maintained_target_obstruction
    e, w = map(Q, model.effective_weights)
    assert obstruction.exchange_to_loss_ratio == w / e
    assert 0 < obstruction.exchange_to_loss_ratio < Q(3, 2)
    assert obstruction.status == "excluded"
    assert obstruction.target_margin_bounds == reference.target_margin_bounds
    assert obstruction.initial_functional_bounds == reference.initial_functional_bounds
    assert obstruction.target_functional_bounds == reference.target_functional_bounds
    assert not report.early_loss_exclusion_certified
    assert report.phase_action_bound.horizon is None
    assert report.exclusion_reasons == ("maintained_target_lyapunov_obstruction",)
    with mp.workdps(110):
        _contains(obstruction.mixed_term_coefficient_bounds, _mp(w / e) / (2 * mp.pi))
    # Exact positive law coefficients need not be numerically separated from
    # zero by the interval grid; that does not erase the exact ratio premise.
    if raw_weights[1] < Q(1, 2**128):
        assert obstruction.mixed_term_coefficient_bounds.contains(0)


def test_maintained_ratio_admission_uses_represented_effective_weights_at_boundary():
    below = RelationalExchangeModel(
        1, epi_weight=0.4, phase_weight=0.6, phase_domain="regular"
    )
    above = RelationalExchangeModel(
        1, epi_weight=0.4, phase_weight=nextafter(0.6, inf), phase_domain="regular"
    )
    low = _report(model=below).maintained_target_obstruction
    high = _report(model=above).maintained_target_obstruction
    # Requested decimals suggest the boundary, but these two captured binary
    # laws lie strictly on opposite sides. Neither comparison uses tolerance.
    assert low.exchange_to_loss_ratio == Q(0.6) / Q(0.4) < Q(3, 2)
    assert high.exchange_to_loss_ratio == Q(nextafter(0.6, inf)) / Q(0.4) > Q(3, 2)
    assert low.status == "excluded"
    assert high.status == "unavailable"
    assert high.target_margin_bounds is None
    assert not high.maintained_target_excluded


def test_exact_ratio_endpoint_is_outside_this_sufficient_theorem_not_instability():
    # The normalized public constructor stores binary64 weights, so test the
    # exact mathematical endpoint directly at the already-admitted helper.
    reference = _report()
    at_endpoint = _maintained_target_obstruction(
        Q(2, 5),
        Q(3, 5),
        Q(1),
        Q(1, 2),
        reference.initial_form_storage,
        reference.twist_phase_storage_bounds,
    )
    assert at_endpoint.exchange_to_loss_ratio == Q(3, 2)
    assert at_endpoint.status == "unavailable"
    assert at_endpoint.unavailable_reasons == (
        "requires_exchange_to_loss_ratio_strictly_between_zero_and_three_halves",
    )
    assert not at_endpoint.maintained_target_excluded
    assert at_endpoint.initial_functional_bounds is None
    assert at_endpoint.target_margin_bounds is None


def test_raw_weight_normalization_gauge_is_identical_law_not_a_clock_change():
    first = RelationalExchangeModel(
        1, epi_weight=3, phase_weight=2, phase_domain="regular"
    )
    scaled = RelationalExchangeModel(
        1, epi_weight=6, phase_weight=4, phase_domain="regular"
    )
    assert first == scaled
    report = _explicit(model=first, exclusion_time=Q(4, 5))
    rescaled_raw = _explicit(model=scaled, exclusion_time=Q(4, 5))
    assert report == rescaled_raw
    assert report.phase_action_bound == rescaled_raw.phase_action_bound
    assert report.initial_form_rate_bounds == rescaled_raw.initial_form_rate_bounds
    assert report.initial_phase_rate_bounds == rescaled_raw.initial_phase_rate_bounds
    assert report.maintained_target_obstruction.exchange_to_loss_ratio != Q(2, 3)


def test_maintained_target_criterion_is_a_source_gap_not_a_four_unit_budget_switch():
    beyond_class_budget = _report(Q(41, 20))
    assert beyond_class_budget.initial_form_storage > 4
    assert beyond_class_budget.maintained_target_obstruction.maintained_target_excluded
    not_excluded = _report(3)
    assert not_excluded.maintained_target_obstruction.status == "not_excluded"
    assert not_excluded.maintained_target_obstruction.target_margin_bounds.hi < 0
    assert not not_excluded.maintained_target_obstruction.maintained_target_excluded
    assert (
        not_excluded.passes_necessary_conditions and not_excluded.formation_unresolved
    )

    with mp.workdps(110):
        threshold = mp.sqrt(mp.mpf(25) * (1 - mp.cos(2 * mp.pi / 5)) / 4)
        unresolved = _report(Q(mp.nstr(threshold, 105)))
    assert unresolved.maintained_target_obstruction.target_margin_bounds.contains(0)
    assert unresolved.maintained_target_obstruction.status == "unresolved"
    assert not unresolved.maintained_target_obstruction.maintained_target_excluded
    assert unresolved.status == "unresolved_bound"
    assert unresolved.unresolved_conditions == (
        "maintained_target_lyapunov_margin_unresolved",
    )


@pytest.mark.parametrize("profile", ("localized", "balanced", "explicit"))
def test_receiver_transfer_has_its_own_exact_endpoint_and_invariant_compatible_lift(
    profile,
):
    source = _explicit() if profile == "explicit" else _report(profile=profile)
    before = pickle.dumps(source)
    transfer = source.receiver_transfer()
    assert transfer.source is source
    assert pickle.dumps(source) == before
    assert transfer.target_phase_turns == (Q(0),) * 5 + tuple(
        Q(j, 5) for j in range(5)
    ) + (Q(0),)
    assert transfer.target_phase_turns != source.target_phase_turns
    assert not any(transfer.target_geometry.symbolic_sine_coefficients)
    assert transfer.compatible_common_phase_turn_offset == Q(-11, 190)
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    assert sum(weights) == Q(76, 3)
    assert sum(weights) * transfer.compatible_form_mean == sum(
        mu * x for mu, x in zip(weights, source.initial_epi)
    )
    assert (
        sum(
            mu * (turn + transfer.compatible_common_phase_turn_offset)
            for mu, turn in zip(weights, transfer.target_phase_turns)
        )
        == source.initial_weighted_phase_turn_sum
    )
    # An extra integer turn at a node changes the compatible common lift;
    # neither representative is asserted to be the future limit.
    turns = tuple(int(i == 9) for i in source.nodes)
    shifted_origin = transfer.compatible_common_phase_turn_offset - sum(
        mu * turn for mu, turn in zip(weights, turns)
    ) / sum(weights)
    assert (
        sum(
            mu * (turn + extra + shifted_origin)
            for mu, turn, extra in zip(weights, transfer.target_phase_turns, turns)
        )
        == source.initial_weighted_phase_turn_sum
    )
    assert transfer.target_storage_margin == source.initial_form_storage == 4
    assert transfer.target_functional_margin == Q(16, 5)
    assert source.status == "excluded"
    assert transfer.status == "passes_necessary_conditions"
    assert not transfer.exclusion_reasons
    assert (
        "no_initial_or_evolved_receiver_recovery_basin_entry_certified"
        in transfer.scope
    )
    with mp.workdps(95):
        phases = tuple(2 * mp.pi * _mp(t) for t in transfer.target_phase_turns)
        forms = (_mp(transfer.compatible_form_mean),) * 11
        rows = _full_rows(forms, phases, tuple(map(_mp, source.capacity)))
        assert all(abs(value) < mp.mpf("1e-90") for row in rows[:2] for value in row)
        v5 = 5 * (1 - mp.cos(2 * mp.pi / 5))
        _contains(transfer.target_storage_bounds, rows[4])
        _contains(transfer.target_functional_bounds, v5)
        _contains(transfer.target_storage_bounds, v5)


def test_receiver_transfer_reuses_captured_preparation_and_keeps_coexistence_verdict(
    monkeypatch,
):
    source = _report(exclusion_time=10)
    assert source.early_loss_exclusion_certified

    def no_new_source(*args, **kwargs):
        raise AssertionError("transfer must not repeat field or horizon evaluation")

    monkeypatch.setattr(
        "tnfr.physics.relational_sine_formation._early_loss_bounds", no_new_source
    )
    monkeypatch.setattr(
        "tnfr.physics.relational_sine_formation._sine_work", no_new_source
    )
    transfer = source.receiver_transfer()
    assert transfer.entry_through_horizon_excluded
    assert transfer.post_time_entry_deficit_lower_bound < 0
    assert not transfer.early_loss_exclusion_certified
    assert transfer.status == "passes_necessary_conditions"
    assert transfer.exclusion_reasons == ()
    assert source.status == "excluded" and source.early_loss_exclusion_certified


@pytest.mark.parametrize(
    "change",
    (
        {"initial_equilibrium": True},
        {"silent_donor_subspace": True},
        {"acute_face_phase_storage_bounds": I(100)},
        {"twist_phase_storage_bounds": I(0)},
        {"invariant_weights": (Q(0),) * 11},
        {"weighted_coordinate_mass": Q(0)},
        {"initial_weighted_phase_turn_sum": Q(999)},
        {"conserved_form_mean": Q(999)},
    ),
)
def test_receiver_transfer_rebuilds_static_evidence_before_excluding(change):
    source = _report(3)
    expected = source.receiver_transfer()
    stale = replace(source, **change)
    actual = stale.receiver_transfer()
    assert actual.source is stale
    assert actual.status == expected.status == "passes_necessary_conditions"
    assert actual.exclusion_reasons == ()
    for field in (
        "compatible_form_mean",
        "compatible_common_phase_turn_offset",
        "target_storage_bounds",
        "entry_storage_threshold_bounds",
        "entry_loss_allowance_bounds",
        "necessary_entry_time_lower_bound",
    ):
        assert getattr(actual, field) == getattr(expected, field)


def test_receiver_transfer_computes_from_admitted_represented_preparation():
    source = _report(3)
    represented = replace(
        source,
        initial_epi=tuple(map(float, source.initial_epi)),
        capacity=tuple(map(float, source.capacity)),
        initial_form_storage=float(source.initial_form_storage),
    )
    actual = represented.receiver_transfer()
    assert actual.source is represented
    assert actual.compatible_form_mean == Q(9, 38)
    assert type(actual.compatible_form_mean) is Q
    assert type(actual.donor_edge_mobility_max) is Q
    assert (
        actual.entry_loss_allowance_bounds
        == source.receiver_transfer().entry_loss_allowance_bounds
    )


@pytest.mark.parametrize(
    "change",
    (
        {"exclusion_time": True},
        {"exclusion_time": Q(0)},
        {"exclusion_time": Q(1), "combined_early_loss_lower_bound": True},
        {"exclusion_time": Q(1), "combined_early_loss_lower_bound": float("nan")},
        {"exclusion_time": Q(1), "combined_early_loss_lower_bound": Q(-1)},
        {"combined_early_loss_lower_bound": Q(0)},
    ),
)
def test_receiver_transfer_admits_retained_horizon_and_loss_evidence(change):
    with pytest.raises((TypeError, ValueError)):
        replace(_report(3), **change).receiver_transfer()


def test_transfer_phase_action_adds_disjoint_ring_costs_with_strict_horizon_control():
    initial = _report().receiver_transfer()
    assert initial.donor_edge_mobility_max == 1
    assert initial.receiver_edge_mobility_max == Q(5, 4)
    assert initial.combined_phase_action_cost == Q(29, 25)
    lower = initial.necessary_entry_time_lower_bound
    assert lower > 50
    with mp.workdps(95):
        v5 = 5 * (1 - mp.cos(2 * mp.pi / 5))
        b5 = 5 - 4 * mp.cos(3 * mp.pi / 8)
        expected = 58 * mp.pi**4 / (25 * (4 + v5 - b5))
        assert _mp(lower) <= expected
        assert expected - _mp(lower) < mp.mpf("1e-15")
        _contains(initial.entry_loss_allowance_bounds, 4 + v5 - b5)
        _contains(initial.entry_storage_threshold_bounds, b5)
    earlier = _report(exclusion_time=lower / 2).receiver_transfer()
    endpoint = _report(exclusion_time=lower).receiver_transfer()
    assert earlier.entry_through_horizon_excluded
    assert not endpoint.entry_through_horizon_excluded
    assert (
        earlier.necessary_entry_time_lower_bound
        == endpoint.necessary_entry_time_lower_bound
    )


def test_transfer_budget_boundary_and_exact_silent_controls_are_independent():
    zero = _report(0).receiver_transfer()
    assert "initial_equilibrium" in zero.exclusion_reasons
    assert zero.phase_action_status == "unavailable"
    assert zero.necessary_entry_time_lower_bound is None
    too_small = _report(Q(1, 10)).receiver_transfer()
    assert too_small.status == "excluded"
    assert too_small.entry_loss_allowance_bounds.hi < 0
    assert too_small.exclusion_reasons == (
        "strict_receiver_transfer_entry_storage_requirement",
        "exact_donor_well_retention",
        "donor_dissipative_capture",
        "weighted_receiver_localization",
    )
    enough = _report(Q(1, 8)).receiver_transfer()
    assert enough.status == "excluded"
    assert enough.entry_loss_allowance_bounds.lo > 0
    assert enough.exclusion_reasons == (
        "exact_donor_well_retention",
        "donor_dissipative_capture",
        "weighted_receiver_localization",
    )
    silent = _explicit(
        tuple(Q(value, 16) for value in (0, 1, 2, -2, -1)), hidden=0
    ).receiver_transfer()
    assert silent.source.initial_form_storage > 0
    assert silent.entry_budget_status == "passed"
    assert silent.exclusion_reasons == (
        "exact_silent_donor_sign_reflection",
        "exact_donor_well_retention",
        "donor_dissipative_capture",
        "weighted_receiver_localization",
    )
    with mp.workdps(110):
        budget = 5 * mp.cos(2 * mp.pi / 5) - 4 * mp.cos(3 * mp.pi / 8)
        borderline = Q(mp.nstr(mp.sqrt(budget), 105))
    unresolved = _report(borderline).receiver_transfer()
    assert unresolved.entry_loss_allowance_bounds.contains(0)
    assert unresolved.status == "excluded"
    assert unresolved.donor_well_retention.status == "certified"
    assert unresolved.phase_action_unavailable_reasons == (
        "positive_loss_allowance_unresolved",
    )
    assert unresolved.necessary_entry_time_lower_bound is None
    assert unresolved.unresolved_conditions == (
        "receiver_transfer_entry_storage_sign_unresolved",
    )


def test_transfer_recomputes_timed_loss_deficit_for_its_own_boundary():
    with mp.workdps(110):
        budget = 5 * mp.cos(2 * mp.pi / 5) - 4 * mp.cos(3 * mp.pi / 8)
        amplitude = Q(mp.nstr(mp.sqrt(budget + mp.mpf("1e-12")), 105))
    source = _report(amplitude, exclusion_time=Q(1, 10))
    transfer = source.receiver_transfer()
    assert transfer.entry_budget_status == "passed"
    assert transfer.entry_loss_allowance_bounds.lo > 0
    assert transfer.entry_through_horizon_excluded
    assert (
        transfer.post_time_entry_deficit_lower_bound
        == (
            source.combined_early_loss_lower_bound
            - transfer.entry_loss_allowance_bounds.hi
        )
        > 0
    )
    assert transfer.early_loss_exclusion_certified
    assert transfer.exclusion_reasons == (
        "early_dissipation_before_receiver_transfer_entry",
        "exact_donor_well_retention",
        "donor_dissipative_capture",
        "weighted_receiver_localization",
    )


@pytest.mark.parametrize(
    "model,contrast,reason",
    (
        (
            RelationalExchangeModel(
                1, epi_weight=3, phase_weight=2, phase_domain="regular"
            ),
            Q(1, 2),
            "requires_effective_half_weight_sine_law",
        ),
        (
            RelationalExchangeModel(2, phase_domain="regular"),
            Q(1, 2),
            "requires_unit_storage_scale",
        ),
        (MODEL, Q(-1, 2), "requires_positive_half_capacity_contrast"),
    ),
)
def test_receiver_transfer_respects_its_distinct_law_scope(model, contrast, reason):
    transfer = _report(model=model, contrast=contrast).receiver_transfer()
    assert transfer.status == "unavailable"
    assert transfer.hypothesis_failures == (reason,)
    assert transfer.target_functional_bounds is None
    assert transfer.target_functional_margin is None
    assert transfer.combined_phase_action_cost is None
    assert transfer.necessary_entry_time_lower_bound is None
    assert not transfer.entry_through_horizon_excluded
    assert not transfer.early_loss_exclusion_certified
    assert transfer.entry_budget_status == "unavailable"
    assert not transfer.exclusion_reasons


def test_receiver_only_target_has_a_shared_recovery_basin_without_a_reached_claim():
    transfer = _explicit().receiver_transfer()
    graph = nx.Graph()
    graph.add_nodes_from(transfer.source.nodes)
    graph.add_edges_from(transfer.source.edges)
    for i, nu in enumerate(transfer.source.capacity):
        # This is a separately supplied, represented near-target preparation.
        # It is not the original twisted-donor source or a response from it.
        graph.nodes[i].update(
            EPI=Q(1, 4096) if i == 0 else 0,
            theta=float(Q(1287 * (i - 5), 1024)) if 5 <= i < 10 else 0,
            nu_f=nu,
        )
    graph.graph["GAMMA"] = {"type": "none"}
    pattern = bound_relational_sine_pattern(
        graph,
        reference_node=0,
        reference_model=MODEL,
        form_error_bounds=(Q(1, 65536),) * 11,
        phase_error_bounds=(Q(1, 65536),) * 11,
    )
    recovery = certify_sine_pattern_recovery(
        pattern,
        target_phase_turns=transfer.target_phase_turns,
        radius=Q(1, 16),
    )
    assert recovery.admitted
    assert recovery.nodes == transfer.source.nodes
    assert recovery.edges == transfer.source.edges
    assert recovery.exact_held_capacity == transfer.source.capacity
    assert recovery.target_geometry == transfer.target_geometry
    assert recovery.excess_storage_upper_bound < Q(1, 28160)
    # The generic graph gap bound is more conservative than the separate
    # support-specific analytic basin; admission uses its actual smaller bound.
    assert 0 < recovery.excess_storage_upper_bound < recovery.barrier_lower_bound
    assert transfer.status == "passes_necessary_conditions"


def test_receiver_transfer_export_retains_source_and_target_scopes(tmp_path):
    source = _report(3)
    transfer = source.receiver_transfer()
    assert source.initial_form_storage > 4
    assert transfer.status == "passes_necessary_conditions"
    payload = transfer.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-receiver-transfer.v1"
    assert payload["report"]["source"] == source.to_dict()["report"]
    assert payload["report"]["target_functional_margin"] == {
        "numerator": 36,
        "denominator": 5,
    }
    assert payload["report"]["compatible_common_phase_turn_offset"] == {
        "numerator": -11,
        "denominator": 190,
    }
    assert (
        payload["report"]["source"]["target_phase_turns"]
        != payload["report"]["target_phase_turns"]
    )
    path = tmp_path / "receiver-transfer.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_donor_well_certifies_source_to_basin_without_replacing_transfer_action():
    source = _report(Q(7, 30))
    before = pickle.dumps(source)
    retained = source.donor_well_retention()
    transfer = source.receiver_transfer()
    assert pickle.dumps(source) == before
    assert retained.source is source
    assert retained.initial_form_storage == Q(49, 900)
    assert retained.status == "certified"
    assert retained.relative_donor_pattern_convergence_certified
    assert retained.receiver_only_targets_excluded
    assert retained.retained_phase_turns == source.initial_phase_turns
    assert retained.retained_geometry == source.initial_geometry
    assert retained.critical_phase_storage == Q(7, 2)
    assert source.initial_storage_bounds.lo > retained.critical_phase_storage
    assert retained.initial_functional_bounds.hi < retained.critical_phase_storage
    assert retained.escape_margin_bounds.lo > 0
    assert not source.silent_donor_subspace
    assert transfer.donor_well_retention == retained
    assert transfer.status == "excluded"
    assert transfer.exclusion_reasons == (
        "exact_donor_well_retention",
        "donor_dissipative_capture",
        "weighted_receiver_localization",
    )
    # The previous necessary passage budget still passes; its time lower
    # bound is not relabeled as an all-time obstruction to transient entry.
    assert transfer.entry_budget_status == "passed"
    assert transfer.phase_action_status == "available"
    assert not transfer.entry_through_horizon_excluded
    assert not transfer.early_loss_exclusion_certified
    assert transfer.necessary_entry_time_lower_bound > 0
    assert (
        "does_not_claim_all_time_winding_preservation_or_no_transient_acute_visits"
        in retained.scope
    )
    assert "no_numeric_recovery_radius_rate_deadline_or_trajectory" in retained.scope
    opposite = _report(Q(-7, 30)).donor_well_retention()
    assert opposite.status == retained.status
    assert opposite.escape_margin_bounds == retained.escape_margin_bounds


def test_donor_well_threshold_decision_survives_unresolved_display_intervals():
    with mp.workdps(190):
        fc = (25 * mp.sqrt(5) - 55) / 16
        scale = 10**170
        amplitude_below = Q(int(mp.floor(mp.sqrt(fc) * scale)), scale)
    below = _report(amplitude_below).donor_well_retention()
    above = _report(amplitude_below + Q(1, scale)).donor_well_retention()
    assert below.escape_margin_bounds.contains(0)
    assert above.escape_margin_bounds.contains(0)
    assert below.critical_form_storage_bounds.contains(below.initial_form_storage)
    assert above.critical_form_storage_bounds.contains(above.initial_form_storage)
    assert below.exact_retention_polynomial_margin > 0
    assert above.exact_retention_polynomial_margin < 0
    assert below.status == "certified"
    assert above.status == "not_certified"
    assert not above.relative_donor_pattern_convergence_certified
    assert not above.receiver_only_targets_excluded
    outside = _report(Q(1, 4)).donor_well_retention()
    assert outside.status == "not_certified"
    assert outside.escape_margin_bounds.hi < 0
    assert _report(Q(1, 4)).receiver_transfer().donor_well_retention == outside
    assert _report(0).donor_well_retention().status == "certified"


@pytest.mark.parametrize("weights", ((3, 2), (1, 1), (1, Q(1, 2**200))))
def test_donor_well_reuses_whole_positive_ratio_family(weights):
    model = RelationalExchangeModel(
        1, epi_weight=weights[0], phase_weight=weights[1], phase_domain="regular"
    )
    source = _report(Q(7, 30), model=model)
    retained = source.donor_well_retention()
    assert retained.status == "certified"
    assert retained.exchange_to_loss_ratio == Q(model.phase_weight) / Q(
        model.epi_weight
    )
    assert retained.sufficient_ratio_upper_bound == Q(3, 2)
    assert retained.initial_functional_bounds == (
        source.maintained_target_obstruction.initial_functional_bounds
    )
    if weights != (1, 1):
        # The receiver-transfer action reader keeps its narrower half-weight
        # contract even when the independent retention theorem is available.
        assert source.receiver_transfer().status == "unavailable"
        assert source.receiver_transfer().donor_well_retention == retained
        assert source.donor_dissipative_capture().status == "unavailable"


@pytest.mark.parametrize(
    "model,contrast,reason",
    (
        (
            RelationalExchangeModel(
                1, epi_weight=1, phase_weight=3, phase_domain="regular"
            ),
            Q(1, 2),
            "requires_exchange_to_loss_ratio_strictly_between_zero_and_three_halves",
        ),
        (
            RelationalExchangeModel(2, phase_domain="regular"),
            Q(1, 2),
            "requires_unit_storage_scale",
        ),
        (MODEL, Q(-1, 2), "requires_positive_half_capacity_contrast"),
    ),
)
def test_donor_well_unsupported_law_does_not_promote_a_favorable_budget(
    model, contrast, reason
):
    retained = _report(Q(7, 30), contrast, model=model).donor_well_retention()
    assert retained.exact_retention_polynomial_margin > 0
    assert retained.status == "unavailable"
    assert retained.unavailable_reasons == (reason,)
    assert retained.form_storage_coefficient is None
    assert retained.initial_functional_bounds is None
    assert retained.escape_margin_bounds is None
    assert not retained.relative_donor_pattern_convergence_certified
    assert not retained.receiver_only_targets_excluded


@pytest.mark.parametrize(
    "name,value",
    (
        ("amplitude", True),
        ("capacity_contrast", float("nan")),
        ("capacity_contrast", 1),
        ("initial_epi", (Q(0),) * 10),
        ("initial_epi", (Q(0),) * 10 + (True,)),
        ("initial_epi", (float("nan"),) * 11),
        ("initial_epi", (Q(1),) + (Q(0),) * 9 + (Q(7, 30),)),
        ("capacity", (Q(1),) * 11),
        ("initial_phase_turns", (Q(0),) * 11),
        ("nodes", (False,) + tuple(range(1, 11))),
        ("degrees", (2,) * 11),
        ("edges", ((0, 1),)),
        ("initial_form_storage", True),
        ("initial_form_storage", Q(-1)),
        ("initial_form_storage", Q(0)),
        ("model", None),
    ),
)
def test_new_basin_evidence_rejects_malformed_consumed_preparation(name, value):
    source = replace(_report(Q(7, 30)), **{name: value})
    with pytest.raises((TypeError, ValueError)):
        source.donor_well_retention()
    with pytest.raises((TypeError, ValueError)):
        source.receiver_transfer()
    with pytest.raises((TypeError, ValueError)):
        source.donor_dissipative_capture()


def test_donor_well_revalidates_geometry_without_replaying_law_or_timed_evidence(
    monkeypatch,
):
    import tnfr.physics.relational_sine_formation as owner

    source = _report(Q(7, 30), exclusion_time=Q(1, 10))
    original = owner._preparation_geometry
    count = 0

    def preparation(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    def no_evaluation(*args, **kwargs):
        raise AssertionError(
            "detached basin admission must not repeat flow or loss work"
        )

    monkeypatch.setattr(owner, "_preparation_geometry", preparation)
    monkeypatch.setattr(owner, "_sine_work", no_evaluation)
    monkeypatch.setattr(owner, "_early_loss_bounds", no_evaluation)
    monkeypatch.setattr(owner, "assess_sine_mediated_formation", no_evaluation)
    assert source.receiver_transfer().donor_well_retention.status == "certified"
    assert count == 1
    invalid_geometry = replace(source, initial_geometry=source.target_geometry)
    with pytest.raises(ValueError, match="initial_geometry"):
        invalid_geometry.donor_well_retention()
    changed_law = replace(source, law="current_squared_mobility")
    unavailable = changed_law.donor_well_retention()
    assert unavailable.unavailable_reasons == ("requires_original_normalized_sine_law",)
    assert not unavailable.relative_donor_pattern_convergence_certified
    assert changed_law.receiver_transfer().status == "unavailable"


@pytest.mark.parametrize(
    "name,value",
    (
        ("epi_weight", True),
        ("epi_weight", 0),
        ("phase_weight", float("nan")),
        ("storage_scale", True),
    ),
)
def test_formation_and_donor_well_readmit_authoritative_model_coefficients(
    name, value, monkeypatch
):
    from tnfr.physics import relational_sine_formation as owner

    model = RelationalExchangeModel(1, phase_domain="regular")
    source = _report(Q(7, 30), model=model)
    object.__setattr__(model, name, value)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid formation coefficients must reject before field work")

    monkeypatch.setattr(owner, "_sine_work", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _report(Q(7, 30), model=model)
    with pytest.raises((TypeError, ValueError)):
        source.donor_well_retention()


def test_donor_well_export_and_optional_transfer_field_keep_evidence_and_compatibility(
    tmp_path,
):
    source = _report(Q(7, 30))
    retained = source.donor_well_retention()
    payload = retained.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-donor-well-retention.v1"
    assert payload["report"]["source"] == source.to_dict()["report"]
    assert payload["report"]["initial_form_storage"] == {
        "numerator": 49,
        "denominator": 900,
    }
    path = tmp_path / "donor-well.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    transfer = source.receiver_transfer()
    assert transfer.to_dict()["report"]["donor_well_retention"] == payload["report"]
    # Every original positional slot, including scope/proof defaults, remains
    # in place. A legacy/manual report without the added evidence stays None.
    transfer_fields = fields(SineReceiverTransferAdmission)
    original_fields = transfer_fields[
        : tuple(field.name for field in transfer_fields).index("donor_well_retention")
    ]
    legacy = SineReceiverTransferAdmission(
        *(getattr(transfer, field.name) for field in original_fields)
    )
    assert legacy.donor_well_retention is None
    assert legacy.to_dict()["report"]["donor_well_retention"] is None
    assert legacy.donor_dissipative_capture is None


def test_dissipative_capture_extends_the_static_source_to_donor_basin_control():
    source = _report(Q(1, 4))
    before = pickle.dumps(source)
    captured = source.donor_dissipative_capture()
    assert captured.source is source
    assert pickle.dumps(source) == before
    assert source.donor_well_retention().status == "not_certified"
    assert captured.status == "certified"
    assert captured.horizon == Q(1, 4)
    assert captured.initial_form_storage == Q(1, 16)
    assert captured.initial_dissipative_norm_squared == Q(1, 6)
    assert captured.functional_drop_coefficient == Q(1, 25)
    assert captured.phase_path_storage_coefficient == Q(121, 38400)
    assert captured.functional_drop_lower_bound == Q(1, 150)
    assert captured.endpoint_storage_allowance == Q(13, 300)
    assert captured.phase_path_storage_allowance == Q(121, 230400)
    assert captured.endpoint_exact_polynomial_margin > 0
    assert captured.phase_path_exact_polynomial_margin > 0
    assert captured.endpoint_functional_bounds.hi < Q(7, 2)
    assert captured.phase_path_storage_bounds.hi < Q(7, 2)
    assert captured.donor_component_entry_certified
    assert captured.relative_donor_pattern_convergence_certified
    assert captured.receiver_only_targets_excluded
    assert captured.retained_geometry == source.initial_geometry
    assert captured.retained_phase_turns == source.initial_phase_turns
    transfer = source.receiver_transfer()
    assert transfer.donor_dissipative_capture == captured
    assert transfer.status == "excluded"
    assert transfer.exclusion_reasons == (
        "donor_dissipative_capture",
        "weighted_receiver_localization",
    )
    assert transfer.entry_budget_status == "passed"
    assert not transfer.early_loss_exclusion_certified
    assert not transfer.entry_through_horizon_excluded
    assert _report(0).donor_dissipative_capture().donor_component_entry_certified
    beyond = _report(Q(3, 10)).donor_dissipative_capture()
    assert beyond.status == "not_certified"
    assert beyond.endpoint_exact_polynomial_margin < 0
    assert beyond.phase_path_exact_polynomial_margin > 0
    assert not beyond.donor_component_entry_certified
    assert not beyond.relative_donor_pattern_convergence_certified
    assert not beyond.receiver_only_targets_excluded


def test_dissipative_capture_retains_preparation_direction_at_equal_form_budget():
    localized = _report(Q(1, 4)).donor_dissipative_capture()
    different = _explicit((Q(3, 10),) * 5, hidden=Q(7, 20)).donor_dissipative_capture()
    assert localized.initial_form_storage == different.initial_form_storage == Q(1, 16)
    assert different.initial_dissipative_norm_squared == Q(73, 600)
    assert different.endpoint_storage_allowance == Q(677, 15000)
    assert localized.status == "certified"
    assert different.status == "not_certified"
    assert different.phase_path_exact_polynomial_margin > 0
    assert different.endpoint_exact_polynomial_margin < 0
    assert not different.receiver_only_targets_excluded
    # This distinguishes the sufficient estimates, not the two actual limits.
    assert (
        "failed_sufficient_bounds_do_not_prove_transfer_or_donor_escape"
        in different.scope
    )


def test_dissipative_capture_has_one_exact_criterion_despite_rounded_endpoint_bounds():
    with mp.workdps(190):
        difference = (5 * mp.sqrt(5) - 11) / 4
        # Localized N=8F/3 gives A=52F/75. Bracket the exact algebraic
        # boundary with rational preparations, not a tolerance or fitted run.
        threshold = mp.sqrt(75 * difference / 52)
        scale = 10**170
        amplitude = Q(int(mp.floor(threshold * scale)), scale)
    below = _report(amplitude).donor_dissipative_capture()
    above = _report(amplitude + Q(1, scale)).donor_dissipative_capture()
    assert below.endpoint_functional_bounds.contains(Q(7, 2))
    assert above.endpoint_functional_bounds.contains(Q(7, 2))
    assert below.endpoint_exact_polynomial_margin > 0
    assert above.endpoint_exact_polynomial_margin < 0
    assert below.status == "certified"
    assert above.status == "not_certified"
    # The phase homotopy bound is a derived implication of the endpoint
    # criterion, not an independently adjustable second gate.
    assert below.phase_path_exact_polynomial_margin > 0
    assert below.phase_path_storage_bounds.hi < Q(7, 2)
    assert above.phase_path_exact_polynomial_margin > 0


@pytest.mark.parametrize(
    "model,contrast,law,reason",
    (
        (
            RelationalExchangeModel(
                1, epi_weight=3, phase_weight=2, phase_domain="regular"
            ),
            Q(1, 2),
            "normalized_sine_reciprocal_exchange",
            "requires_effective_half_weight_sine_law",
        ),
        (
            RelationalExchangeModel(2, phase_domain="regular"),
            Q(1, 2),
            "normalized_sine_reciprocal_exchange",
            "requires_unit_storage_scale",
        ),
        (
            MODEL,
            Q(-1, 2),
            "normalized_sine_reciprocal_exchange",
            "requires_positive_half_capacity_contrast",
        ),
        (
            MODEL,
            Q(1, 2),
            "current_squared_mobility",
            "requires_original_normalized_sine_law",
        ),
    ),
)
def test_dissipative_capture_keeps_its_fixed_law_scope(model, contrast, law, reason):
    source = replace(_report(Q(1, 4), contrast, model=model), law=law)
    captured = source.donor_dissipative_capture()
    assert captured.status == "unavailable"
    assert captured.unavailable_reasons == (reason,)
    assert captured.endpoint_exact_polynomial_margin > 0
    assert captured.functional_drop_lower_bound is None
    assert captured.endpoint_functional_bounds is None
    assert captured.phase_path_storage_bounds is None
    assert not captured.donor_component_entry_certified
    assert not captured.relative_donor_pattern_convergence_certified
    assert not captured.receiver_only_targets_excluded


def test_dissipative_capture_rederives_gradient_and_ignores_cached_horizon_evidence(
    monkeypatch,
):
    import tnfr.physics.relational_sine_formation as owner

    source = _report(Q(1, 4))
    expected = source.donor_dissipative_capture()
    altered = replace(
        source,
        initial_form_gradient=(Q(0),) * 11,
        directional_loss_bound=replace(
            source.directional_loss_bound, initial_dissipative_norm_squared=Q(999)
        ),
        exclusion_time=Q(8),
        combined_early_loss_lower_bound=Q(999),
    )

    def no_evaluation(*args, **kwargs):
        raise AssertionError(
            "capture reuses primitive source without a flow or horizon run"
        )

    monkeypatch.setattr(owner, "_sine_work", no_evaluation)
    monkeypatch.setattr(owner, "_directional_loss_bound", no_evaluation)
    monkeypatch.setattr(owner, "_early_loss_bounds", no_evaluation)
    monkeypatch.setattr(owner, "assess_sine_mediated_formation", no_evaluation)
    actual = altered.donor_dissipative_capture()
    assert (
        actual.initial_dissipative_norm_squared
        == expected.initial_dissipative_norm_squared
    )
    assert actual.endpoint_functional_bounds == expected.endpoint_functional_bounds
    assert actual.functional_drop_lower_bound == expected.functional_drop_lower_bound
    assert actual.horizon == expected.horizon == Q(1, 4)
    assert actual.status == expected.status == "certified"


def test_dissipative_capture_exports_and_appends_evidence_without_positional_migration(
    tmp_path,
):
    source = _report(Q(1, 4))
    captured = source.donor_dissipative_capture()
    payload = captured.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-donor-dissipative-capture.v1"
    assert payload["report"]["source"] == source.to_dict()["report"]
    assert payload["report"]["initial_dissipative_norm_squared"] == {
        "numerator": 1,
        "denominator": 6,
    }
    path = tmp_path / "donor-dissipative.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    transfer = source.receiver_transfer()
    assert (
        transfer.to_dict()["report"]["donor_dissipative_capture"] == payload["report"]
    )
    transfer_fields = fields(SineReceiverTransferAdmission)
    preceding_fields = transfer_fields[
        : tuple(field.name for field in transfer_fields).index(
            "donor_dissipative_capture"
        )
    ]
    prior = SineReceiverTransferAdmission(
        *(getattr(transfer, field.name) for field in preceding_fields)
    )
    assert prior.donor_well_retention == transfer.donor_well_retention
    assert prior.donor_dissipative_capture is None
    assert prior.to_dict()["report"]["donor_dissipative_capture"] is None


def test_receiver_barrier_order_is_conditional_not_a_transfer_exclusion():
    donor_first = _report(Q(5, 4)).receiver_transfer()
    full_budget = _report(2).receiver_transfer()
    not_ordered = _report(3).receiver_transfer()
    for report in (donor_first, full_budget, not_ordered):
        assert report.receiver_first_barrier_phase_storage == Q(7, 2)
        assert report.necessary_receiver_running_work_strict_lower_bound == Q(7, 2)
        assert report.status == "passes_necessary_conditions"
        assert report.exclusion_reasons == ()
        assert report.entry_budget_status == "passed"
        with mp.workdps(100):
            expected = 14 * mp.pi**2 / 3
            lower = _mp(report.receiver_barrier_time_loss_product_lower_bound)
            assert 0 < lower <= expected
            assert expected - lower < mp.mpf("1e-35")
    assert donor_first.donor_barrier_first_required is True
    assert donor_first.simultaneous_barrier_passage_excluded is True
    assert full_budget.donor_barrier_first_required is True
    assert full_budget.simultaneous_barrier_passage_excluded is True
    assert not_ordered.donor_barrier_first_required is False
    assert not_ordered.simultaneous_barrier_passage_excluded is False
    # A first-crossing input requirement does not evaluate actual future work.
    assert donor_first.phase_action_status == "available"
    assert donor_first.source.exclusion_time is None


def test_receiver_donor_first_order_includes_the_exact_closed_storage_boundary():
    # Two unit bridge edges give F=H**2. Admission must use exact total F,
    # not a rounded interval or the older energy-only order threshold.
    at = _report(2).receiver_transfer()
    assert at.source.initial_form_storage == 4
    assert at.donor_barrier_first_required is True
    assert at.simultaneous_barrier_passage_excluded is True
    above = _report(2 + Q(1, 2**200)).receiver_transfer()
    assert above.source.initial_form_storage > 4
    assert above.donor_barrier_first_required is False
    assert above.simultaneous_barrier_passage_excluded is False
    assert at.status == above.status == "passes_necessary_conditions"


def test_receiver_barrier_fields_are_unavailable_outside_the_existing_transfer_scope():
    model = RelationalExchangeModel(
        1, epi_weight=3, phase_weight=2, phase_domain="regular"
    )
    unsupported = _report(1, model=model).receiver_transfer()
    assert unsupported.status == "unavailable"
    assert unsupported.receiver_first_barrier_phase_storage is None
    assert unsupported.necessary_receiver_running_work_strict_lower_bound is None
    assert unsupported.donor_barrier_first_required is None
    assert unsupported.simultaneous_barrier_passage_excluded is None
    assert unsupported.receiver_barrier_time_loss_product_lower_bound is None
    assert unsupported.to_dict()["report"]["donor_barrier_first_required"] is None


def test_receiver_barrier_fields_export_without_shifting_previous_constructor_slots(
    tmp_path,
):
    transfer = _report(1).receiver_transfer()
    report = transfer.to_dict()
    assert report["report"]["receiver_first_barrier_phase_storage"] == {
        "numerator": 7,
        "denominator": 2,
    }
    assert report["report"]["donor_barrier_first_required"] is True
    path = tmp_path / "receiver-passage.json"
    export_to_json(report, path)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    transfer_fields = fields(SineReceiverTransferAdmission)
    existing_fields = transfer_fields[
        : tuple(field.name for field in transfer_fields).index(
            "receiver_first_barrier_phase_storage"
        )
    ]
    prior = SineReceiverTransferAdmission(
        *(getattr(transfer, field.name) for field in existing_fields)
    )
    assert prior.donor_dissipative_capture == transfer.donor_dissipative_capture
    assert prior.receiver_first_barrier_phase_storage is None
    assert prior.necessary_receiver_running_work_strict_lower_bound is None
    assert prior.donor_barrier_first_required is None
    assert prior.simultaneous_barrier_passage_excluded is None
    assert prior.receiver_barrier_time_loss_product_lower_bound is None


def test_exact_cycle_face_attains_joint_entry_threshold_but_receiver_only_costs_less():
    report = _report()
    edges, _ = _support()
    with mp.workdps(90):
        donor = tuple(2 * mp.pi * j / 5 for j in range(5))
        face = (0, mp.pi / 2, 7 * mp.pi / 8, 5 * mp.pi / 4, 13 * mp.pi / 8)
        phase = donor + face + (0,)
        face_energy = sum(1 - mp.cos(phase[j] - phase[i]) for i, j in edges)
        _contains(report.entry_storage_threshold_bounds, face_energy)
        assert mp.almosteq(
            face_energy, 5 * (1 - mp.cos(2 * mp.pi / 5)) + 5 - 4 * mp.cos(3 * mp.pi / 8)
        )
        receiver_only = (0,) * 5 + donor + (0,)
        receiver_energy = sum(
            1 - mp.cos(receiver_only[j] - receiver_only[i]) for i, j in edges
        )
        _contains(report.twist_phase_storage_bounds, receiver_energy)
        assert receiver_energy < _mp(report.target_storage_bounds.lo)


@pytest.mark.parametrize(
    "amplitude,contrast",
    ((2, Q(1, 2)), (-2, Q(-1, 2)), (2, Q(1, 2**200)), (Q(19, 10), Q(99, 100))),
)
def test_early_loss_excludes_signed_localized_preparations_without_any_trajectory(
    amplitude, contrast
):
    report = _report(amplitude, contrast, exclusion_time=Q(1, 5))
    assert report.status == "excluded" and report.early_loss_exclusion_certified
    assert "early_dissipation_before_joint_acute_entry" in report.exclusion_reasons
    assert report.receiver_acute_time_margin_lower_bound > 0
    assert report.receiver_gap_drift_upper_bound < Q(11, 40)
    assert report.mediator_gradient_speed_upper_bound < 6
    assert report.loss_integration_time == Q(1, 5)
    u = abs(Q(amplitude))
    coarse_loss = u**2 / 5 - 3 * u / 25 + Q(3, 125)
    assert report.early_loss_lower_bound > coarse_loss
    assert report.post_time_entry_deficit_lower_bound > 0
    with mp.workdps(90):
        energy = _mp(u) ** 2 + 5 * (1 - mp.cos(2 * mp.pi / 5))
        e, a, b = mp.mpf("0.5"), 1 / (2 * mp.pi), 1 / (2 * mp.pi)
        speeds = tuple(
            _mp(nu) * (e * mp.sqrt(2 * energy / d) + a)
            for nu, d in zip(report.capacity, report.degrees)
        )
        for certified, expected in zip(report.form_speed_upper_bounds, speeds):
            assert _mp(certified) >= expected
        gradient_speed = 2 * speeds[10] + speeds[0] + speeds[5]
        assert _mp(report.mediator_gradient_speed_upper_bound) >= gradient_speed
        phase_speeds = tuple(
            b * _mp(nu) * mp.sqrt(2 * energy / d)
            for nu, d in zip(report.capacity, report.degrees)
        )
        gap_speed = max(
            phase_speeds[i] + phase_speeds[j]
            for i, j in ((5, 6), (6, 7), (7, 8), (8, 9), (9, 5))
        )
        assert _mp(report.receiver_gap_speed_upper_bound) >= gap_speed
        tau = mp.mpf(1) / 5
        sharp_affine_loss = (
            e
            / 2
            * (
                4 * _mp(u) ** 2 * tau
                - 2 * _mp(u) * gradient_speed * tau**2
                + gradient_speed**2 * tau**3 / 3
            )
        )
        assert _mp(report.early_loss_lower_bound) <= sharp_affine_loss


def test_action_and_loss_horizons_do_not_inherit_maintenance_exclusion():
    short = _report(exclusion_time=Q(1, 100))
    assert short.receiver_acute_time_margin_lower_bound > 0
    assert short.post_time_entry_deficit_lower_bound < 0
    assert not short.early_loss_exclusion_certified
    assert short.exclusion_reasons == ("maintained_target_lyapunov_obstruction",)
    middle = _report(exclusion_time=10)
    assert middle.receiver_acute_time_margin_lower_bound < 0
    assert middle.phase_action_bound.entry_through_horizon_excluded
    assert middle.post_time_entry_deficit_lower_bound > 0
    assert middle.early_loss_exclusion_certified and middle.status == "excluded"
    long = _report(exclusion_time=1000)
    assert long.post_time_entry_deficit_lower_bound > 0
    assert long.receiver_acute_time_margin_lower_bound < 0
    assert not long.phase_action_bound.entry_through_horizon_excluded
    assert not long.early_loss_exclusion_certified
    assert long.loss_integration_time < long.exclusion_time
    assert long.exclusion_reasons == ("maintained_target_lyapunov_obstruction",)


@pytest.mark.parametrize("profile", ("localized", "balanced"))
def test_full_early_loss_uses_every_actual_initial_gradient_and_clipped_window(profile):
    report = _report(profile=profile, exclusion_time=Q(1, 5))
    _, neighbors = _support()
    assert (
        sum(report.nodal_early_loss_lower_bounds) == report.full_early_loss_lower_bound
    )
    assert report.nodal_early_loss_lower_bounds[10] == report.early_loss_lower_bound
    assert report.nodal_loss_integration_times[10] == report.loss_integration_time
    with mp.workdps(90):
        energy = 4 + 5 * (1 - mp.cos(2 * mp.pi / 5))
        e, a = mp.mpf(1) / 2, 1 / (2 * mp.pi)
        speeds = tuple(
            _mp(nu) * (e * mp.sqrt(2 * energy / len(row)) + a)
            for nu, row in zip(report.capacity, neighbors)
        )
        for i, row in enumerate(neighbors):
            gradient_speed = len(row) * speeds[i] + sum(speeds[j] for j in row)
            assert _mp(report.nodal_gradient_speed_upper_bounds[i]) >= gradient_speed
            initial = abs(_mp(report.initial_form_gradient[i]))
            stop = min(mp.mpf(1) / 5, initial / gradient_speed)
            # An independent integral of the clipped affine lower bound,
            # using unrounded 90-digit global-speed constants.
            integral = mp.quad(
                lambda t: max(0, initial - gradient_speed * t) ** 2, [0, stop]
            )
            lower_loss = e * _mp(report.capacity[i]) * integral / len(row)
            assert _mp(report.nodal_early_loss_lower_bounds[i]) <= lower_loss
            if not initial:
                assert report.nodal_loss_integration_times[i] == 0
                assert report.nodal_early_loss_lower_bounds[i] == 0
    if profile == "balanced":
        assert report.early_loss_lower_bound == report.loss_integration_time == 0
        assert report.full_early_loss_lower_bound > 0
        assert report.post_time_entry_deficit_lower_bound < 0
        assert not report.early_loss_exclusion_certified
        assert report.exclusion_reasons == ("maintained_target_lyapunov_obstruction",)
    else:
        assert report.full_early_loss_lower_bound > report.early_loss_lower_bound > 0
        assert report.early_loss_exclusion_certified


def test_balanced_global_affine_bound_does_not_certify_formation_or_inherit_localized_exclusion():
    report = _report(profile="balanced", exclusion_time=10)
    assert report.full_early_loss_lower_bound < report.entry_storage_margin_bounds.lo
    assert report.receiver_acute_time_margin_lower_bound < 0
    assert not report.early_loss_exclusion_certified
    assert report.exclusion_reasons == ("maintained_target_lyapunov_obstruction",)
    assert report.early_loss_lower_bound == 0
    assert all(
        stop < report.exclusion_time for stop in report.nodal_loss_integration_times
    )
    assert "early_dissipation_before_joint_acute_entry" not in report.exclusion_reasons


@pytest.mark.parametrize("profile", ("localized", "balanced"))
def test_directional_loss_uses_independent_full_weighted_matrix_moments(profile):
    report = _report(Q(-7, 3), Q(2, 5), profile=profile, exclusion_time=Q(3, 5))
    bound = report.directional_loss_bound
    assert bound.available and bound.unavailable_reasons == ()
    edges, neighbors = _support()
    with mp.workdps(90):
        laplacian = mp.matrix(11)
        for i, j in edges:
            laplacian[i, i] += 1
            laplacian[j, j] += 1
            laplacian[i, j] -= 1
            laplacian[j, i] -= 1
        roots = mp.diag(
            [mp.sqrt(_mp(nu) / len(row)) for nu, row in zip(report.capacity, neighbors)]
        )
        matrix = roots * laplacian * roots
        form = mp.matrix(tuple(map(_mp, report.initial_epi)))
        initial = roots * laplacian * form
        pushed = matrix * initial
        norm = mp.fdot(initial, initial)
        first = mp.fdot(initial, pushed)
        second = mp.fdot(pushed, pushed)
        assert mp.almosteq(_mp(bound.initial_dissipative_norm_squared), norm)
        assert mp.almosteq(_mp(bound.first_spectral_moment), first)
        assert mp.almosteq(_mp(bound.second_spectral_moment), second)
        assert mp.almosteq(_mp(bound.rayleigh_quotient), first / norm)
        assert mp.almosteq(_mp(bound.gamma_squared), second / norm)
        assert _mp(bound.gamma_upper_bound) >= mp.sqrt(second / norm)
        spectrum = mp.eigsy(matrix, eigvals_only=True)
        assert 1 < spectrum[spectrum.rows - 1] <= _mp(bound.spectral_upper_bound)
        # A separately assembled, nonconstant phase Hessian must obey the
        # same domination bound; no target linearization is assumed.
        hessian = mp.matrix(11)
        for i, j in edges:
            weight = mp.cos(mp.mpf(j * j - i * i) / 7)
            hessian[i, i] += weight
            hessian[j, j] += weight
            hessian[i, j] -= weight
            hessian[j, i] -= weight
        phase_spectrum = mp.eigsy(roots * hessian * roots, eigvals_only=True)
        assert max(abs(value) for value in phase_spectrum) <= _mp(
            bound.spectral_upper_bound
        )
        e, w = map(_mp, report.model.effective_weights)
        beta = _mp(report.model.storage_scale)
        c_exact = (
            mp.mpf(11)
            / 20
            * mp.sqrt(second / norm)
            * w**2
            / (beta * mp.pi**2)
            * _mp(bound.spectral_upper_bound)
        )
        assert _mp(bound.quadratic_coefficient) >= c_exact
        tau = _mp(bound.horizon)
        alpha, quadratic = map(
            _mp, (bound.linear_coefficient, bound.quadratic_coefficient)
        )
        integral = mp.quad(lambda t: (1 - alpha * t - quadratic * t**2) ** 2, [0, tau])
        assert mp.almosteq(_mp(bound.loss_lower_bound), e * norm * integral)
        assert 1 - alpha * tau - quadratic * tau**2 > 0
    assert report.combined_early_loss_lower_bound == max(
        report.full_early_loss_lower_bound, bound.loss_lower_bound
    )
    assert (
        report.combined_early_loss_lower_bound
        < report.full_early_loss_lower_bound + bound.loss_lower_bound
    )
    if profile == "balanced":
        assert bound.rayleigh_quotient == 1
        assert bound.gamma_squared == Q(4, 3)


@pytest.mark.parametrize(
    "amplitude,contrast", ((2, Q(1, 2)), (-2, Q(-99, 100)), (2, Q(1, 2**200)))
)
def test_directional_window_excludes_balanced_boundary_preparations_without_a_response_run(
    amplitude, contrast
):
    report = _report(amplitude, contrast, profile="balanced", exclusion_time=Q(3, 5))
    bound = report.directional_loss_bound
    assert bound.available
    assert bound.quadratic_coefficient < Q(77, 1080)
    assert bound.loss_lower_bound > 7 * Q(amplitude) ** 2 / 50
    assert report.full_early_loss_lower_bound < report.entry_storage_margin_bounds.lo
    assert report.combined_early_loss_lower_bound == bound.loss_lower_bound
    assert report.receiver_gap_drift_upper_bound < Q(33, 40)
    assert report.receiver_acute_time_margin_lower_bound > 0
    assert report.post_time_entry_deficit_lower_bound > 0
    assert report.status == "excluded" and report.early_loss_exclusion_certified
    assert not report.formation_unresolved


def test_directional_loss_rejects_unavailable_horizon_range_and_positive_norm_premises():
    absent = _report(profile="balanced").directional_loss_bound
    assert not absent.available and absent.loss_lower_bound is None
    assert absent.unavailable_reasons == ("exclusion_time_not_supplied",)
    zero = _report(0, profile="balanced", exclusion_time=Q(3, 5)).directional_loss_bound
    assert zero.rayleigh_quotient is zero.gamma_squared is None
    assert zero.loss_lower_bound is None
    assert "zero_initial_form_gradient" in zero.unavailable_reasons
    range_fail = _report(profile="balanced", exclusion_time=3)
    direction = range_fail.directional_loss_bound
    assert direction.hyperbolic_argument_squared_upper_bound >= 2
    assert "hyperbolic_argument_bound_exceeded" in direction.unavailable_reasons
    assert direction.loss_lower_bound is None
    assert (
        range_fail.combined_early_loss_lower_bound
        == range_fail.full_early_loss_lower_bound
    )
    dissipative = RelationalExchangeModel(
        1, epi_weight=99, phase_weight=1, phase_domain="regular"
    )
    norm_fail = _report(
        profile="balanced", exclusion_time=2, model=dissipative
    ).directional_loss_bound
    assert norm_fail.hyperbolic_argument_squared_upper_bound < Q(4, 25)
    assert norm_fail.norm_ratio_lower_bound_at_horizon < 0
    assert norm_fail.unavailable_reasons == ("positive_norm_ratio_not_certified",)
    assert norm_fail.loss_lower_bound is None


def test_tiny_exact_gradient_retains_spectral_direction_without_absolute_tolerance():
    tiny = Q(1, 2**200)
    report = _report(tiny, profile="balanced", exclusion_time=Q(3, 5))
    bound = report.directional_loss_bound
    assert bound.initial_dissipative_norm_squared == 2 * tiny**2 / 3
    assert bound.rayleigh_quotient == 1 and bound.gamma_squared == Q(4, 3)
    assert bound.available and bound.loss_lower_bound > 0
    assert report.initial_loss_per_form_storage == Q(1, 3)
    assert _report(0, profile="balanced").initial_loss_per_form_storage is None


def test_profiles_retain_their_different_conserved_form_origins_and_same_lifted_phase_mean():
    localized, balanced = (
        _report(profile=profile) for profile in ("localized", "balanced")
    )
    _, neighbors = _support()
    weights = tuple(Q(len(row)) / nu for row, nu in zip(neighbors, balanced.capacity))
    mass = sum(weights)
    for report in (localized, balanced):
        assert report.invariant_weights == weights
        assert report.weighted_coordinate_mass == mass
        form_sum = sum(
            weight * form for weight, form in zip(weights, report.initial_epi)
        )
        phase_sum = sum(
            weight * phase for weight, phase in zip(weights, report.initial_phase_turns)
        )
        assert report.initial_weighted_form_sum == form_sum
        assert report.initial_weighted_phase_turn_sum == phase_sum
        assert report.conserved_form_mean == form_sum / mass
        assert report.conserved_lifted_phase_turn_mean == phase_sum / mass
    assert balanced.conserved_form_mean == 12 * localized.conserved_form_mean
    assert (
        balanced.conserved_lifted_phase_turn_mean
        == localized.conserved_lifted_phase_turn_mean
    )
    assert balanced.to_dict()["report"]["profile"] == "balanced"
    assert localized.to_dict()["report"]["profile"] == "localized"


def test_profile_selection_is_explicit_and_does_not_coerce_invalid_values():
    assert _report() == _report(profile="localized")
    for invalid in (None, True, "", "Balanced", ["balanced"]):
        with pytest.raises(ValueError, match="profile"):
            _report(profile=invalid)


@pytest.mark.parametrize(
    "donor,hidden",
    ((CANDIDATE_DONOR, Q(4, 3)), ((Q(1),) + (Q(0),) * 4, Q(0))),
)
def test_internal_donor_receiver_jets_retain_delayed_causal_maps(donor, hidden):
    report = _explicit(donor, hidden)
    _, neighbors = _support()
    assert report.initial_epi == donor + (Q(0),) * 5 + (hidden,)
    assert not report.initial_equilibrium
    state = (
        tuple(map(I, report.initial_epi))
        + tuple(2 * value * pi_interval() for value in report.initial_phase_turns)
        + (I(1),)
    )
    series = flow_jets(
        state,
        3,
        lambda values: _sine_flow(
            values,
            neighbors=neighbors,
            visible_capacity=report.capacity[:-1],
            model=MODEL,
        ),
    )
    with mp.workdps(90):
        phase = tuple(2 * mp.pi * _mp(value) for value in report.initial_phase_turns)
        first, omega, second, theta_second, energy, energy_dot = _full_rows(
            tuple(map(_mp, report.initial_epi)), phase, tuple(map(_mp, report.capacity))
        )
        e, a, b, delta = mp.mpf(1) / 2, 1 / (2 * mp.pi), 1 / (2 * mp.pi), mp.mpf(1) / 2
        h, d0, d1, d4 = map(_mp, (hidden, donor[0], donor[1], donor[4]))
        c = e**2 - a * b
        theta_third = tuple(
            b
            * _mp(report.capacity[i])
            / len(row)
            * sum(second[i] - second[j] for j in row)
            for i, row in enumerate(neighbors)
        )
        for i in range(11):
            _contains(report.initial_form_rate_bounds[i], first[i])
            _contains(report.initial_phase_rate_bounds[i], omega[i])
            _contains(2 * series[i][2], second[i])
            _contains(2 * series[11 + i][2], theta_second[i])
            _contains(6 * series[11 + i][3], theta_third[i])
        _contains(report.initial_storage_bounds, energy)
        assert mp.almosteq(_mp(report.initial_storage_rate), energy_dot)
        assert mp.almosteq(first[5], e * h / 3)
        assert mp.almosteq(omega[5], -b * h / 3)
        assert mp.almosteq(theta_second[5], b * e * (4 * h - d0) / 6)
        assert mp.almosteq(theta_second[6] - theta_second[9], -b * e * delta * h / 3)
        assert mp.almosteq(
            theta_third[6] - theta_third[9], b * c * delta * (8 * h - d0) / 6
        )
        assert mp.almosteq(theta_third[5], b * c * (9 * d0 - d1 - d4 - 22 * h) / 18)
    if hidden == 0:
        assert report.phase_odd_acceleration_pi_numerator == 0
        assert report.initial_phase_rate_bounds[5] == I(0)
        assert series[16][2].hi < 0
        assert (series[17][3] - series[20][3]).hi < 0


@pytest.mark.parametrize("time", (Q(3, 5), Q(4, 5), Q(6, 5), Q(10)))
def test_explicit_fixed_budget_candidate_keeps_entry_and_maintenance_separate(
    time,
):
    report = _explicit(exclusion_time=time)
    bound = report.directional_loss_bound
    assert report.initial_form_storage == 4
    assert report.initial_form_gradient == (
        Q(0),
        Q(2, 3),
        Q(1, 3),
        Q(1, 3),
        Q(2, 3),
        Q(-4, 3),
        Q(0),
        Q(0),
        Q(0),
        Q(0),
        Q(-2, 3),
    )
    assert bound.initial_dissipative_norm_squared == Q(37, 27)
    assert bound.first_spectral_moment == Q(43, 54)
    assert bound.second_spectral_moment == Q(47, 54)
    assert bound.rayleigh_quotient == Q(43, 74)
    assert bound.gamma_squared == Q(47, 74)
    assert report.initial_continuous_loss == Q(37, 54)
    assert report.initial_form_rate_bounds[0] == I(0)
    assert report.initial_form_rate_bounds[10].lo > 0
    assert report.phase_odd_acceleration_pi_numerator == Q(-1, 18)
    if time == Q(6, 5):
        assert (
            report.combined_early_loss_lower_bound
            > report.entry_storage_margin_bounds.hi
        )
        assert report.status == "excluded" and report.early_loss_exclusion_certified
        assert not report.formation_unresolved
    else:
        assert (
            report.combined_early_loss_lower_bound
            < report.entry_storage_margin_bounds.lo
        )
        assert report.exclusion_reasons == ("maintained_target_lyapunov_obstruction",)
        assert not report.early_loss_exclusion_certified
    assert bound.available == (time < 2)
    if time < 2:
        assert report.receiver_acute_time_margin_lower_bound > 0
    else:
        assert bound.loss_lower_bound is None
        assert (
            report.combined_early_loss_lower_bound == report.full_early_loss_lower_bound
        )


def test_short_window_certificate_ceiling_does_not_constrain_a_longer_admitted_window():
    report = _explicit()
    edges, neighbors = _support()
    form = report.initial_epi
    q = tuple(sum(form[i] - form[j] for j in row) for i, row in enumerate(neighbors))
    mobility = tuple(nu / len(row) for nu, row in zip(report.capacity, neighbors))
    kq = tuple(k * value for k, value in zip(mobility, q))
    norm = sum(k * value**2 for k, value in zip(mobility, q))
    first = sum((kq[i] - kq[j]) ** 2 for i, j in edges)
    lkq = tuple(sum(kq[i] - kq[j] for j in row) for i, row in enumerate(neighbors))
    second = sum(k * value**2 for k, value in zip(mobility, lkq))
    assert report.directional_loss_bound.initial_dissipative_norm_squared == norm
    assert report.directional_loss_bound.first_spectral_moment == first
    assert report.directional_loss_bound.second_spectral_moment == second
    linear = first / (2 * norm)
    # Every short-window certificate (omega*tau<=2/5) has
    # tau<=4*pi/15<21/25. This only limits that certificate branch;
    # positive g(t)<=1-linear*t bounds its value, not the actual loss.
    horizon_upper = Q(21, 25)
    directional_ceiling = (
        norm
        / 2
        * (horizon_upper - linear * horizon_upper**2 + linear**2 * horizon_upper**3 / 3)
    )
    assert directional_ceiling == Q(74344921, 166500000) < Q(9, 20)
    # M_i > nu_i/(2*pi) > 7*nu_i/44 makes a LOWER gradient-speed
    # constant, hence an UPPER ceiling on each affine loss certificate.
    speed_lower = tuple(
        Q(7, 44)
        * (len(row) * report.capacity[i] + sum(report.capacity[j] for j in row))
        for i, row in enumerate(neighbors)
    )
    affine_ceiling = sum(
        Q(1, 2) * k * abs(value) ** 3 / (3 * speed)
        for k, value, speed in zip(mobility, q, speed_lower)
    )
    assert affine_ceiling == Q(385, 1458) < Q(9, 20)
    assert (
        max(directional_ceiling, affine_ceiling)
        < Q(1, 2)
        < report.entry_storage_margin_bounds.lo
    )
    with mp.workdps(90):
        assert 4 * mp.pi / 15 < _mp(horizon_upper)
        assert 1 / (2 * mp.pi) > _mp(Q(7, 44))


def test_extended_rational_cosh_window_certifies_the_candidate_without_simulation():
    report = _explicit(exclusion_time=Q(6, 5))
    bound = report.directional_loss_bound
    v = bound.hyperbolic_argument_squared_upper_bound
    assert Q(4, 25) < v < Q(9, 25)
    assert bound.cosh_upper_bound == 1 / (1 - v / 2)
    assert bound.cosh_upper_bound < Q(5, 4)
    assert bound.gamma_upper_bound < Q(4, 5)
    assert bound.quadratic_coefficient < Q(1, 24)
    assert bound.norm_ratio_lower_bound_at_horizon > Q(547, 925)
    # Independently integrate the conservative rational polynomial used by
    # the mathematical proof, rather than reuse the computed coefficient.
    tau, d, c = Q(6, 5), Q(43, 148), Q(1, 24)
    integral = (
        tau
        - d * tau**2
        + (d * d - 2 * c) * tau**3 / 3
        + d * c * tau**4 / 2
        + c * c * tau**5 / 5
    )
    coarse_loss = Q(37, 54) * integral
    assert coarse_loss == Q(7564289, 13875000)
    assert bound.loss_lower_bound > coarse_loss > report.entry_storage_margin_bounds.hi
    assert report.receiver_gap_drift_upper_bound < Q(11, 8)
    assert report.early_loss_exclusion_certified and not report.formation_unresolved
    with mp.workdps(90):
        actual_argument = 3 / (2 * mp.pi) * _mp(tau)
        assert mp.cosh(actual_argument) <= _mp(bound.cosh_upper_bound)
        actual_norm_coefficient = (
            mp.cosh(actual_argument) / 2 * mp.sqrt(mp.mpf(47) / 74) * 3 / (4 * mp.pi**2)
        )
        assert actual_norm_coefficient < _mp(bound.quadratic_coefficient)
    encoded = report.to_dict()["report"]["directional_loss_bound"]
    assert encoded["cosh_bound_method"] == bound.cosh_bound_method


def test_donor_antisymmetric_subspace_is_invariant_with_a_fixed_receiver_and_mediator():
    report = _explicit((Q(0), Q(2), Q(3), Q(-3), Q(-2)), Q(0))
    assert report.initial_form_storage > 4 and not report.initial_equilibrium
    assert report.silent_donor_subspace
    assert report.exclusion_reasons == ("exact_silent_donor_sign_reflection",)
    assert report.status == "excluded"
    assert report.phase_odd_acceleration_pi_numerator == 0
    reflection = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9, 10)
    edges, _ = _support()
    assert {frozenset((reflection[i], reflection[j])) for i, j in edges} == set(
        map(frozenset, edges)
    )
    assert tuple(report.capacity[i] for i in reflection) == report.capacity
    with mp.workdps(90):
        # Probe two points of the invariant subspace, including a deformed
        # donor phase. Equivariance plus uniqueness, rather than these finite
        # probes alone, establishes the all-time decoupling in the proof.
        for p1, p2 in ((2 * mp.pi / 5, 4 * mp.pi / 5), (mp.mpf("0.7"), mp.mpf("1.1"))):
            form = tuple(map(_mp, report.initial_epi))
            phase = (mp.mpf(0), p1, p2, -p2, -p1) + (mp.mpf(0),) * 6
            first, omega, *_ = _full_rows(form, phase, tuple(map(_mp, report.capacity)))
            for i in range(11):
                assert abs(first[i] + first[reflection[i]]) < mp.mpf("1e-80")
                assert abs(omega[i] + omega[reflection[i]]) < mp.mpf("1e-80")
            assert max(abs(value) for value in first[5:] + omega[5:]) < mp.mpf("1e-80")
    perturbed = _explicit((Q(0), Q(2), Q(3), Q(-3), Q(-2) + Q(1, 2**200)), Q(0))
    assert not perturbed.silent_donor_subspace
    assert "exact_silent_donor_sign_reflection" not in perturbed.exclusion_reasons


def test_explicit_donor_admission_preserves_exact_values_and_rejects_ambiguous_profiles():
    for bad in (
        None,
        (),
        (0,) * 4,
        (0,) * 6,
        1,
        "00000",
        {0, 1, 2, 3, 4},
        (True, 0, 0, 0, 0),
        (float("nan"), 0, 0, 0, 0),
        (1j, 0, 0, 0, 0),
    ):
        with pytest.raises((TypeError, ValueError)):
            _explicit(bad)
    for profile in ("localized", "balanced"):
        with pytest.raises(ValueError, match="donor_epi"):
            _report(profile=profile, donor_epi=(0,) * 5)
    tiny = Q(1, 2**200)
    report = _explicit((tiny, 0, 0, 0, 0), Q(0))
    assert report.initial_epi[0] == tiny
    assert not report.initial_equilibrium
    assert report.initial_form_storage == 3 * tiny**2 / 2
    assert report.directional_loss_bound.initial_dissipative_norm_squared > 0
    payload = report.to_dict()["report"]
    assert payload["profile"] == "explicit"
    assert payload["initial_epi"][0] == {"numerator": 1, "denominator": 2**200}


@pytest.mark.parametrize(
    "keyword,value",
    (
        ("amplitude", True),
        ("amplitude", float("nan")),
        ("amplitude", 1j),
        ("capacity_contrast", 1),
        ("capacity_contrast", -1),
        ("capacity_contrast", False),
        ("exclusion_time", 0),
        ("exclusion_time", -1),
        ("exclusion_time", float("inf")),
    ),
)
def test_invalid_preparation_and_clock_domains_reject(keyword, value):
    options = dict(model=MODEL, amplitude=2, capacity_contrast=Q(1, 2))
    options[keyword] = value
    with pytest.raises((TypeError, ValueError)):
        assess_sine_mediated_formation(**options)


def test_law_and_positive_dissipation_premises_are_required():
    for model in (
        None,
        RelationalExchangeModel(1),
        RelationalExchangeModel(1, epi_weight=0, phase_domain="regular"),
    ):
        with pytest.raises(ValueError):
            _report(model=model)


def test_pure_immutable_report_and_exact_atomic_export(tmp_path):
    before = pickle.dumps(MODEL, protocol=5)
    report = _report(amplitude=0.1, contrast=Q(1, 2**200), exclusion_time=Q(1, 5))
    assert report.amplitude == Q(0.1) != Q(1, 10)
    assert pickle.dumps(MODEL, protocol=5) == before
    with pytest.raises(FrozenInstanceError):
        report.amplitude = Q(7)
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-mediated-formation.v1"
    exact = payload["report"]["capacity_contrast"]
    assert exact == {"numerator": 1, "denominator": 2**200}
    assert payload["report"]["nodes"] == list(range(11))
    assert payload["report"]["law"] == "normalized_sine_reciprocal_exchange"
    path = tmp_path / "formation-obstruction.json"
    export_to_json(payload, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_phase_action_export_retains_exact_lower_bound_and_missing_manual_evidence():
    report = _report(exclusion_time=10)
    action = report.phase_action_bound
    encoded = report.to_dict()["report"]["phase_action_bound"]
    lower = action.necessary_entry_time_lower_bound
    assert encoded["necessary_entry_time_lower_bound"] == {
        "numerator": lower.numerator,
        "denominator": lower.denominator,
    }
    assert encoded["entry_through_horizon_excluded"] is True
    assert (
        encoded["allowable_loss_bounds"]
        == report.to_dict()["report"]["entry_storage_margin_bounds"]
    )
    assert "conditional_on_entry" in " ".join(encoded["scope"])
    legacy = replace(
        report, phase_action_bound=None, maintained_target_obstruction=None
    )
    assert legacy.to_dict()["report"]["phase_action_bound"] is None
    assert legacy.to_dict()["report"]["maintained_target_obstruction"] is None
    # Every original positional field retains its slot; the appended optional
    # records are unavailable when an old constructor supplies only old fields.
    old_values = tuple(
        getattr(report, field.name)
        for field in fields(report)
        if field.name not in ("phase_action_bound", "maintained_target_obstruction")
    )
    old_style = type(report)(*old_values)
    assert (
        old_style.phase_action_bound is old_style.maintained_target_obstruction is None
    )
    assert old_style.law == report.law and old_style.scope == report.scope
    maintained = report.to_dict()["report"]["maintained_target_obstruction"]
    assert maintained["form_storage_coefficient"] == {"numerator": 4, "denominator": 5}
    assert maintained["maintained_target_excluded"] is True
    assert maintained["proof_id"] == "eleven_node_sine_ratio_mixed_lyapunov"
    assert maintained["exchange_to_loss_ratio"] == {"numerator": 1, "denominator": 1}
    assert maintained["sufficient_ratio_upper_bound"] == {
        "numerator": 3,
        "denominator": 2,
    }
    obstruction = report.maintained_target_obstruction
    old_obstruction_values = tuple(
        getattr(obstruction, field.name)
        for field in fields(obstruction)
        if field.name not in ("exchange_to_loss_ratio", "sufficient_ratio_upper_bound")
    )
    old_obstruction = type(obstruction)(*old_obstruction_values)
    assert old_obstruction.exchange_to_loss_ratio is None
    assert old_obstruction.sufficient_ratio_upper_bound is None
    assert old_obstruction.proof_id == obstruction.proof_id
    assert old_obstruction.scope == obstruction.scope
