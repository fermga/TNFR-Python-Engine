"""Independent acquired-class interaction controls without source replay.

Class one and class two use the same fine sine law and contact registration.
Current form/registered phase, the persistent discrete class, actual residuals,
source preparation costs and support work remain separate information.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import product

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _sine_constitutive_inheritance as inheritance
from tnfr.physics import _sine_constitutive_recovery as recovery_owner
from tnfr.physics import _sine_winding_response as owner

ERROR = Q(1, 10**10)
EPSILON = Q(1, 10**70)
ETA = 20 * EPSILON
DELTA = Q(1, 1024)
HORIZON = Q(1, 10)
INTERFACES = ("matching", "symmetric")
CASES = tuple(product(INTERFACES, (1, 2)))


def _assess(**changes):
    arguments = dict(
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        observation_error_bound=ERROR,
    )
    arguments.update(changes)
    return owner._assess_sine_winding_response(**arguments)


def _mp(context, value):
    return context.mpf(value.numerator) / value.denominator


def _contains(context, bounds, value, width="1e-30"):
    lower, upper = (bounds.lo, bounds.hi) if hasattr(bounds, "lo") else bounds
    assert _mp(context, lower) <= value <= _mp(context, upper)
    assert _mp(context, upper - lower) < context.mpf(width)


def _display(context, value):
    return context.nstr(_mp(context, value), 25)


def _geometry(interface):
    offsets = (0,) if interface == "matching" else (-1, 0, 1)
    internal = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in (0, 1) for j in range(9))
    cross = tuple((j, 9 + (j + s) % 9) for j in range(9) for s in offsets)
    edges = internal + cross
    degree = tuple(sum(i in edge for edge in edges) for i in range(18))
    return edges, cross, degree


def _fine_rows(context, interface, winding, forms, phase_gap):
    edges, cross, degrees = _geometry(interface)
    alpha = 2 * context.pi / 9
    phase = tuple(
        winding * alpha * (i % 9 - 4) + (-1 if i < 9 else 1) * phase_gap / 2
        for i in range(18)
    )
    values = tuple(forms[i // 9] for i in range(18))
    gradient = [Q(0)] * 18
    currents = [context.mpf(0)] * 18
    work = energy = context.mpf(0)
    for i, j in edges:
        difference = values[i] - values[j]
        gradient[i] += difference
        gradient[j] -= difference
        current = context.sin(phase[j] - phase[i])
        currents[i] += current
        currents[j] -= current
        edge_energy = (
            _mp(context, difference**2 / 2) + 1 - context.cos(phase[j] - phase[i])
        )
        energy += edge_energy
        if (i, j) in cross:
            work += edge_energy
    gamma = 1 / (1023 * context.pi)
    form_rates = tuple(
        (-_mp(context, g) + gamma * s) / d
        for g, s, d in zip(gradient, currents, degrees)
    )
    phase_rates = tuple(gamma * _mp(context, g) / d for g, d in zip(gradient, degrees))
    return values, phase, tuple(gradient), form_rates, phase_rates, energy, work


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_operator_or_retained_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("winding-response controls cannot evolve or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def response():
    return _assess()


def test_default_class_one_results_scopes_and_explicit_selection_agree():
    for deformation in (Q(0), Q(1)):
        arguments = dict(
            deformation=deformation, form_error_bound=ERROR, phase_error_bound=ERROR
        )
        default = recovery_owner._assess_constitutive_recovery(**arguments)
        explicit = recovery_owner._assess_constitutive_recovery(
            **arguments, target_winding=1
        )
        assert default == explicit
        assert default.target_winding == 1
        assert default.status == "certified_constitutive_recovery"
        assert (
            default.scope[0]
            == "fresh_own_law_class_one_source_and_original_zero_sum_errors"
        )
        assert default.relaxation_duration == 10**13
        assert default.endpoint_radius == EPSILON
        for interface in INTERFACES:
            inputs = dict(interface=interface, deformation=deformation)
            assert inheritance._derive_constitutive_inheritance(
                **inputs
            ) == inheritance._derive_constitutive_inheritance(
                **inputs, target_winding=1
            )
            values = dict(
                inputs,
                form=(Q(3, 7), Q(-2, 5)),
                relative_phase_phasor=(Q(3, 5), Q(4, 5)),
            )
            assert inheritance._observe_inherited_constitutive_pair(
                **values
            ) == inheritance._observe_inherited_constitutive_pair(
                **values, target_winding=1
            )


def test_both_recoveries_use_their_own_acquired_class_and_unchanged_resources(response):
    assert tuple(item.target_winding for item in response.recoveries) == (1, 2)
    for index, report in enumerate(response.recoveries):
        source = report.source
        assert report.source_certified and report.recovery_certified
        assert source.class_order == ("winding_one", "winding_two")
        assert source.formation_certified_by_class[index]
        assert source.source_budget_certified_by_class[index]
        # Rebuild the own-class initial modified energy; choosing class zero
        # in both reports would violate this association even if both pass.
        expected = (
            Q(3, 4)
            * source.eta_bounds.hi
            * 2
            * source.endpoint_form_norm_upper_bounds[index] ** 2
        )
        expected += (
            report.lyapunov_position_upper_coefficient
            * source.endpoint_target_phase_norm_upper_bounds[index] ** 2
            / Q(1, 5)
        )
        assert report.lyapunov_initial_upper_bound == expected
        energy = expected / 2**512
        assert report.returned_lyapunov_upper_bound == energy
        assert report.endpoint_form_norm_squared_upper_bound == 4 * energy / (
            source.eta_bounds.lo * Q(1, 5)
        )
        assert (
            report.endpoint_phase_norm_squared_upper_bound
            == 2 * energy / report.lyapunov_position_lower_coefficient
        )
        assert report.endpoint_form_norm_squared_upper_bound < EPSILON**2
        assert report.endpoint_phase_norm_squared_upper_bound < EPSILON**2
        assert report.scaled_handoff_time == 100 + 10**13
        assert report.original_handoff_time == Q(1024, 1023) * (100 + 10**13)
    # The preparation means match; their nominal form-storage costs do not.
    slope = Q(2046, 9) * Q(355, 113) ** 2
    costs = tuple(36 * (k * slope) ** 2 for k in (1, 2))
    assert costs[1] == 4 * costs[0] and costs[1] < 10**9


def test_selected_class_requires_fresh_class_specific_admission(response, monkeypatch):
    fresh = recovery_owner._assess_constitutive_class_pair
    calls = []

    def only_class_one(**arguments):
        calls.append(arguments)
        source = fresh(**arguments)
        return replace(
            source,
            formation_certified_by_class=(True, False),
            source_budget_certified_by_class=(True, False),
            status="unavailable",
        )

    monkeypatch.setattr(
        recovery_owner, "_assess_constitutive_class_pair", only_class_one
    )
    arguments = dict(deformation=0, form_error_bound=ERROR, phase_error_bound=ERROR)
    first = recovery_owner._assess_constitutive_recovery(**arguments)
    second = recovery_owner._assess_constitutive_recovery(**arguments, target_winding=2)
    assert first.recovery_certified and not second.recovery_certified
    assert not second.source_certified and second.status == "unavailable"
    assert len(calls) == 2

    def only_class_two(**arguments):
        calls.append(arguments)
        source = fresh(**arguments)
        return replace(
            source,
            formation_certified_by_class=(False, True),
            source_budget_certified_by_class=(False, True),
            status="unavailable",
        )

    monkeypatch.setattr(
        recovery_owner, "_assess_constitutive_class_pair", only_class_two
    )
    second = recovery_owner._assess_constitutive_recovery(**arguments, target_winding=2)
    assert second.recovery_certified and second.source.status == "unavailable"
    assert (
        second.lyapunov_initial_upper_bound
        == response.recoveries[1].lyapunov_initial_upper_bound
    )
    assert len(calls) == 3


@pytest.mark.parametrize("interface,winding", CASES)
def test_independent_full_eighteen_node_rows_storage_and_probe_registration(
    interface, winding, mp
):
    forms, phasor = (Q(3, 7), Q(-2, 5)), (Q(3, 5), Q(4, 5))
    report = inheritance._observe_inherited_constitutive_pair(
        interface=interface,
        deformation=0,
        target_winding=winding,
        form=forms,
        relative_phase_phasor=phasor,
    )
    delta = mp.atan2(4, 3)
    edges, cross, degrees = _geometry(interface)
    _, phases, gradient, form_rates, phase_rates, energy, work = _fine_rows(
        mp, interface, winding, forms, delta
    )
    assert report.law.edges == edges and report.law.cross_edges == cross
    assert len({frozenset(edge) for edge in edges}) == len(edges)
    assert degrees == (report.law.degree,) * 18
    assert report.law.target_phase_turns == tuple(
        Q(winding * (j % 9 - 4), 9) for j in range(18)
    )
    for i in range(18):
        _contains(mp, report.form_rate_bounds[i // 9], form_rates[i])
        _contains(mp, report.phase_rate_bounds[i // 9], phase_rates[i])
    _contains(mp, report.full_storage_bounds, energy)
    loss = sum((value**2 / degrees[i] for i, value in enumerate(gradient)), Q(0))
    assert report.loss == loss and report.storage_rate == -loss
    assert abs(sum(form_rates)) < mp.mpf("1e-190")
    assert abs(sum(phase_rates)) < mp.mpf("1e-190")
    assert abs(sum(phases)) < mp.mpf("1e-190")
    q, degree = len(cross) // 9, degrees[0]
    kappa = mp.mpf(q) / degree
    gamma = 1 / (1023 * mp.pi)
    gain = (
        1 if interface == "matching" else (1 + 2 * mp.cos(2 * winding * mp.pi / 9)) / 3
    )
    assert report.law.collective_clock_factor == Q(q, degree)
    assert degree in (3, 5) and Q(1, degree) != Q(1, 4)
    _contains(mp, report.law.first_harmonic_multiplier_bounds, gain)
    _contains(
        mp,
        report.form_rate_bounds[0],
        kappa * (_mp(mp, forms[1] - forms[0]) + gamma * gain * mp.sin(delta)),
    )
    _contains(
        mp, report.phase_rate_bounds[0], -gamma * kappa * _mp(mp, forms[1] - forms[0])
    )
    ideal_cross = (
        9 * q * ((_mp(mp, forms[1] - forms[0])) ** 2 / 2 + 1 - gain * mp.cos(delta))
    )
    assert abs(work - ideal_cross) < mp.mpf("1e-190")


def test_registered_phase_and_discrete_class_are_not_supplied_by_a_circular_mean(mp):
    alpha = 2 * mp.pi / 9
    for winding in (1, 2):
        phases = tuple(winding * alpha * (j - 4) for j in range(9))
        resultant = sum((mp.exp(1j * value) for value in phases), mp.mpc(0))
        assert abs(resultant) < mp.mpf("1e-190")
        for shift in (mp.mpf(0), alpha):
            ninth = (
                sum((mp.exp(9j * (value + shift)) for value in phases), mp.mpc(0)) / 9
            )
            assert abs(ninth - 1) < mp.mpf("1e-190")
        gain = (1 + 2 * mp.cos(winding * alpha)) / 3
        assert gain * mp.sin(alpha) > mp.mpf("0.1")
    # Same X and registered Phi do not select one response across classes.
    first = inheritance._observe_inherited_constitutive_pair(
        interface="symmetric",
        deformation=0,
        form=(0, 0),
        relative_phase_phasor=(Q(3, 5), Q(4, 5)),
        target_winding=1,
    )
    second = inheritance._observe_inherited_constitutive_pair(
        interface="symmetric",
        deformation=0,
        form=(0, 0),
        relative_phase_phasor=(Q(3, 5), Q(4, 5)),
        target_winding=2,
    )
    assert (
        first.form == second.form
        and first.relative_phase_phasor == second.relative_phase_phasor
    )
    assert first.form_rate_bounds[0].lo > second.form_rate_bounds[0].hi


def test_arbitrary_isolated_waiting_keeps_actual_residuals_and_original_means(response):
    assert response.source_certified and response.isolated_waiting_certified
    assert response.recovered_radius == EPSILON
    assert response.waiting_radius == ETA
    assert response.isolated_excess_upper_bound == 4 * EPSILON**2
    assert response.isolated_coercivity_lower_bound == Q(1, 100)
    assert (
        response.isolated_excess_upper_bound / response.isolated_coercivity_lower_bound
        == ETA**2
    )
    assert response.isolated_storage_barrier == Q(1, 14400)
    assert response.isolated_waiting_storage_margin == Q(1, 14400) - 4 * EPSILON**2 > 0
    assert response.initial_form_means == (0, 0)
    assert response.supplied_phase_origins == (-DELTA / 2, DELTA / 2)
    # Independent antisymmetric edge currents leave each isolated ring's
    # means unchanged. The source rotation is present before acquisition,
    # not a post-recovery phase reset or a time-selected observation.
    for winding in (1, 2):
        initial_form = tuple(Q(winding * (j - 4)) for j in range(9))
        residual = (ETA / 8, -ETA / 8) + (Q(0),) * 7
        assert sum(initial_form) == sum(residual) == 0
        for origin in response.supplied_phase_origins:
            phases = tuple(origin + value for value in residual)
            assert sum(phases) / 9 == origin
            edge_currents = tuple(Q((3 * j) % 7 - 3, 11) for j in range(9))
            rates = [Q(0)] * 9
            for j, current in enumerate(edge_currents):
                rates[j] += current
                rates[(j + 1) % 9] -= current
            assert sum(rates) == 0
    assert ETA > EPSILON > 0


def test_actual_contact_work_retains_cancelled_first_variation_and_subgrid_remainder(
    response, mp
):
    delta = _mp(mp, DELTA)
    form = (ETA / 8, -ETA / 8) + (Q(0),) * 7 + (Q(0), ETA / 8, -ETA / 8) + (Q(0),) * 6
    residual = (
        (ETA / 10, Q(0), -ETA / 10)
        + (Q(0),) * 6
        + (Q(0), -ETA / 10, Q(0), ETA / 10)
        + (Q(0),) * 5
    )
    assert len(form) == len(residual) == 18
    for values in (form, residual):
        assert sum(values[:9]) == sum(values[9:]) == 0
        assert sum(value**2 for value in values[:9]) < ETA**2
        assert sum(value**2 for value in values[9:]) < ETA**2
    for branch in response.interfaces:
        _, cross, _ = _geometry(branch.interface)
        q = len(cross) // 9
        assert branch.attachment_work_correction_upper_bound == 4 * q * ETA**2 > 0
        for index, winding in enumerate((1, 2)):
            alpha = 2 * winding * mp.pi / 9
            nominal = tuple(
                alpha * (i % 9 - 4) + (-1 if i < 9 else 1) * delta / 2
                for i in range(18)
            )
            work = nominal_work = linear_variation = mp.mpf(0)
            for i, j in cross:
                gap = nominal[j] - nominal[i]
                variation = _mp(mp, residual[j] - residual[i])
                nominal_work += 1 - mp.cos(gap)
                work += (
                    _mp(mp, (form[j] - form[i]) ** 2 / 2) + 1 - mp.cos(gap + variation)
                )
                linear_variation += mp.sin(gap) * variation
            _contains(mp, branch.nominal_attachment_work_bounds[index], nominal_work)
            assert abs(linear_variation) < mp.mpf("1e-240")
            correction = work - nominal_work
            assert (
                0 < correction < _mp(mp, branch.attachment_work_correction_upper_bound)
            )
        if q == 3:
            assert (
                branch.nominal_attachment_work_bounds[1].lo
                > branch.nominal_attachment_work_bounds[0].hi
            )


def test_full_transverse_barrier_covers_both_classes_and_includes_probe_offsets(
    response, mp
):
    # The matching Laplacian has C9 x K2 eigenvalues; extra offset edges
    # add nonnegative squared differences, so cannot lower its gap.
    eigenvalues = tuple(
        2 - 2 * mp.cos(2 * mp.pi * j / 9) + 2 * layer
        for j in range(9)
        for layer in (0, 1)
        if (j, layer) != (0, 0)
    )
    assert min(eigenvalues) > mp.mpf(2) / 5
    assert mp.cos(4 * mp.pi / 9 + mp.sqrt(2) / 12) > mp.mpf(1) / 20
    expected_norm = Q(9, 2) * DELTA**2 + 4 * ETA**2
    assert expected_norm > 4 * ETA**2
    for branch in response.interfaces:
        assert branch.initial_joint_norm_squared_upper_bound == expected_norm
        assert branch.post_attachment_excess_upper_bound == 5 * expected_norm
        assert branch.phase_hessian_lower_bound == Q(1, 20)
        assert branch.full_graph_gap_lower_bound == Q(2, 5)
        assert branch.storage_coercivity_lower_bound == Q(1, 100)
        assert branch.storage_barrier == Q(1, 14400)
        assert branch.radius_margin == Q(1, 144) - expected_norm > 0
        assert branch.storage_margin == Q(1, 14400) - 5 * expected_norm > 0
        assert branch.identity_certified
        assert branch.duration == HORIZON
        assert branch.original_duration == HORIZON * Q(1024, 1023)
    _, matching, _ = _geometry("matching")
    _, symmetric, _ = _geometry("symmetric")
    assert set(matching) < set(symmetric)


def test_live_phase_row_and_independent_damped_kernel_keep_receiver_factor_one_half(
    response, mp
):
    gamma, delta, time = 1 / (1023 * mp.pi), _mp(mp, DELTA), _mp(mp, HORIZON)
    for branch in response.interfaces:
        q = 1 if branch.interface == "matching" else 3
        kappa = mp.mpf(q) / (2 + q)
        decay = mp.exp(-2 * kappa * time)
        _contains(mp, branch.exponential_decay_bounds, decay)
        gain = gamma * mp.sin(delta) * (1 - decay) / 2
        _contains(mp, branch.duhamel_gain_bounds, gain)
        assert branch.phase_bootstrap_margin > 0
        for index, winding in enumerate((1, 2)):
            r = mp.mpf(1) if q == 1 else (1 + 2 * mp.cos(2 * winding * mp.pi / 9)) / 3
            leading = gain * r
            nonlinear_loss = (
                mp.mpf(2) / 3 * gamma**3 * kappa**3 * r**2 * delta * time**3
            )
            assert nonlinear_loss <= _mp(mp, branch.feedback_upper_bounds[index])
            assert _mp(
                mp, branch.feedback_upper_bounds[index]
            ) - nonlinear_loss < mp.mpf("1e-30")
            lo, hi = branch.ideal_response_bounds[index]
            _contains(mp, (lo + branch.feedback_upper_bounds[index], hi), leading)
            assert lo > 0 and hi - lo > branch.feedback_upper_bounds[index] > 0
            # The frozen-phase integral solves v'=-2*kappa*v+forcing,
            # hence its coefficient is forcing/(2*kappa), not the contrast
            # twice the receiver mean. Its phase feedback is nonzero.
            frozen_derivative = kappa * gamma * r * mp.sin(delta) * decay
            assert abs(
                frozen_derivative
                + 2 * kappa * leading
                - kappa * gamma * r * mp.sin(delta)
            ) < mp.mpf("1e-190")
            forms = (Q(1, 64), Q(-1, 64))
            _, _, _, form_rates, phase_rates, _, _ = _fine_rows(
                mp, branch.interface, winding, forms, delta
            )
            v = mp.mpf(1) / 64
            assert abs(
                form_rates[0] - (-2 * kappa * v + kappa * gamma * r * mp.sin(delta))
            ) < mp.mpf("1e-190")
            assert abs(
                (phase_rates[9] - phase_rates[0]) + 4 * gamma * kappa * v
            ) < mp.mpf("1e-190")
            assert phase_rates[9] - phase_rates[0] < 0
            # A whole-interval bootstrap and integral bound, not endpoint
            # samples of an uncomputed nonlinear trajectory.
            phase_fraction_loss = 2 * gamma**2 * kappa**2 * r * time**2
            assert phase_fraction_loss < mp.mpf(1) / 2
            integrated_loss = (
                (kappa * gamma * r)
                * (2 * gamma**2 * kappa**2 * r * delta)
                * time**3
                / 3
            )
            assert abs(nonlinear_loss - integrated_loss) < mp.mpf("1e-190")


def test_finite_signed_response_preserves_common_factors_one_sided_loss_and_noise(
    response, mp, record_property
):
    assert (
        response.certified and response.status == "certified_organized_winding_response"
    )
    assert not response.unresolved_conditions
    assert response.interface_order == INTERFACES and response.class_order == (1, 2)
    assert response.response_target == Q(1, 10**9)
    gamma, delta, time = 1 / (1023 * mp.pi), _mp(mp, DELTA), _mp(mp, HORIZON)
    q3 = response.interfaces[1]
    assert q3.interface == "symmetric"
    r1 = (1 + 2 * mp.cos(2 * mp.pi / 9)) / 3
    r2 = (1 + 2 * mp.cos(4 * mp.pi / 9)) / 3
    common_gap = (
        gamma * (r1 - r2) * mp.sin(delta) * (1 - mp.exp(-mp.mpf(6) * time / 5)) / 2
    )
    _contains(
        mp,
        (
            q3.ideal_contrast_bounds[0] + q3.feedback_upper_bounds[0],
            q3.ideal_contrast_bounds[1] - q3.feedback_upper_bounds[1],
        ),
        common_gap,
    )
    assert r1 - r2 > mp.mpf(1) / 3
    assert DELTA / 369600 > Q(1, 4 * 10**8)
    assert q3.feedback_upper_bounds[0] < Q(1, 10**16)
    fine_error = Q(20, 7) * ETA
    coarse_lower = DELTA / 369600 - Q(1, 10**16) - fine_error - 2 * ERROR
    assert (
        q3.recorded_contrast_bounds[0]
        > coarse_lower
        > Q(2, 10**9)
        > response.response_target
    )
    assert q3.response_certified and not q3.unresolved_conditions
    for branch in response.interfaces:
        assert branch.fine_mean_error_upper_bound == Q(10, 7) * ETA
        assert branch.fine_contrast_error_upper_bound == fine_error > 0
        for ideal, actual, recorded in zip(
            branch.ideal_response_bounds,
            branch.actual_response_bounds,
            branch.recorded_response_bounds,
        ):
            assert actual == (ideal[0] - fine_error / 2, ideal[1] + fine_error / 2)
            assert recorded == (actual[0] - ERROR, actual[1] + ERROR)
            assert actual[0] < ideal[0] and actual[1] > ideal[1]
        assert branch.actual_contrast_bounds == (
            branch.ideal_contrast_bounds[0] - fine_error,
            branch.ideal_contrast_bounds[1] + fine_error,
        )
        assert branch.recorded_contrast_bounds == (
            branch.actual_contrast_bounds[0] - 2 * ERROR,
            branch.actual_contrast_bounds[1] + 2 * ERROR,
        )
    record_property(
        "symmetric_recorded_contrast_lower",
        _display(mp, q3.recorded_contrast_bounds[0]),
    )
    record_property("fixed_recorded_target", "1e-9")
    record_property("isolated_waiting_radius", str(ETA))
    record_property("same_probe_phase_gap", str(DELTA))
    record_property("post_contact_duration", str(HORIZON))
    with pytest.raises(FrozenInstanceError):
        response.certified = False


def test_matching_is_exact_carrier_blindness_with_nonzero_actual_allowance(
    response, mp, record_property
):
    branch = response.interfaces[0]
    assert branch.interface == "matching" and branch.response_certified
    assert (
        branch.laws[0].first_harmonic_multiplier_bounds
        == branch.laws[1].first_harmonic_multiplier_bounds
    )
    assert branch.ideal_contrast_bounds == (0, 0)
    fine = Q(20, 7) * ETA
    assert branch.actual_contrast_bounds == (-fine, fine)
    assert branch.actual_contrast_bounds != branch.ideal_contrast_bounds
    assert branch.recorded_contrast_bounds == (-fine - 2 * ERROR, fine + 2 * ERROR)
    assert branch.recorded_contrast_bounds[1] < response.response_target
    # The ideal IVPs coincide for all times; subtracting independent broad
    # response intervals would fail to preserve this exact correlation.
    assert branch.ideal_response_bounds[0] == branch.ideal_response_bounds[1]
    assert branch.ideal_response_bounds[0][1] - branch.ideal_response_bounds[0][0] > 0
    record_property(
        "matching_recorded_contrast_abs_upper",
        _display(mp, branch.recorded_contrast_bounds[1]),
    )
    record_property("matching_actual_contrast_abs_upper", str(fine))


def test_invalid_winding_rejects_before_source_or_trigonometric_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid winding reached source or trigonometric work")

    monkeypatch.setattr(recovery_owner, "_assess_constitutive_class_pair", forbidden)
    monkeypatch.setattr(inheritance, "pi_interval", forbidden)
    for winding in (True, False, 0, 3, -1, Q(1), 1.0, "1", np.int64(1), None):
        with pytest.raises((TypeError, ValueError)):
            recovery_owner._assess_constitutive_recovery(
                deformation=0,
                form_error_bound=ERROR,
                phase_error_bound=ERROR,
                target_winding=winding,
            )
        with pytest.raises((TypeError, ValueError)):
            inheritance._derive_constitutive_inheritance(
                interface="matching", deformation=0, target_winding=winding
            )
        with pytest.raises((TypeError, ValueError)):
            inheritance._observe_inherited_constitutive_pair(
                interface="symmetric",
                deformation=0,
                form=(0, 0),
                relative_phase_phasor=(1, 0),
                target_winding=winding,
            )


def test_raw_response_allowances_reject_before_all_source_and_bound_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid allowance reached source or analytical numerical work")

    for name in (
        "_assess_constitutive_recovery",
        "_derive_constitutive_inheritance",
        "pi_interval",
        "cos",
        "sin",
        "sqrt",
        "_negative_exp_bounds",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for name in ("form_error_bound", "phase_error_bound", "observation_error_bound"):
        for value in (True, float("nan"), float("inf"), 1j, -Q(1, 10**100)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: value})


def test_valid_but_unsupported_allowances_remain_unavailable():
    for name in ("form_error_bound", "phase_error_bound", "observation_error_bound"):
        report = _assess(**{name: 2 * ERROR})
        assert not report.certified and report.status == "unavailable"
        assert report.unresolved_conditions
        assert all(not branch.response_certified for branch in report.interfaces)
        if name == "observation_error_bound":
            assert report.source_certified and report.isolated_waiting_certified
            assert not report.observation_allowance_within_protocol
            assert (
                "observation_allowance_exceeds_fixed_protocol"
                in report.unresolved_conditions
            )
        else:
            assert not report.source_allowances_within_protocol
            assert not report.source_certified and not report.isolated_waiting_certified
            assert (
                "source_allowance_exceeds_fixed_protocol"
                in report.unresolved_conditions
            )


def test_fixed_width_integer_fraction_payloads_normalize_before_arithmetic(response):
    fixed = Q(np.int64(1), np.int64(10**10))
    result = _assess(
        form_error_bound=fixed, phase_error_bound=fixed, observation_error_bound=fixed
    )
    assert result == response
    for value in (
        result.form_error_bound,
        result.phase_error_bound,
        result.observation_error_bound,
    ):
        assert type(value.numerator) is type(value.denominator) is int


def test_cached_reports_and_changed_protocol_are_not_input_arguments(
    response, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported protocol input reached source work")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    for extra in (
        {"recoveries": response.recoveries},
        {"source": response.recoveries[0].source},
        {"deformation": 1},
        {"duration": 0},
        {"relative_phase_gap": 0},
        {"target_winding": 1},
        {"waiting_duration": 0},
        {"relaxation_duration": 0},
    ):
        with pytest.raises(TypeError):
            _assess(**extra)
