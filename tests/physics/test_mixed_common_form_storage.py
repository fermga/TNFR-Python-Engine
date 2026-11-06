"""Storage selection, fine port work and actual-law dissipation are distinct.

Independent edge sums and static matrix controls test the declared algebra.
Constructed chart states are not claimed to be acquired source endpoints;
no source, trajectory, measurement or operator event is evaluated here.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.physics import test_mixed_positive_interface as port
from tests.physics import test_mixed_resource_transfer_obstruction as fine
from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_common_form_storage as owner
from tnfr.physics import _sine_lyapunov as kernel_owner

ERROR = Q(1, 10**10)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
WEIGHT_ERROR = Q(2, 10**60)
HORIZON = Q(1, 2)


def _assess(**changes):
    values = dict(form_error_bounds=(ERROR, ERROR), phase_error_bounds=(ERROR, ERROR))
    values.update(changes)
    return owner._assess_mixed_common_form_storage(**values)


def _helper(**changes):
    values = dict(transport_gap_lower=Q(3), port_norm_upper=Q(2), loss_divisor=Q(9))
    values.update(changes)
    return kernel_owner._sine_noncommuting_port_coefficients(**values)


def _regional(values):
    return tuple(
        sum(values[9 * r : 9 * (r + 1)]) / 9 for r in range(4) for _ in range(9)
    )


def _relative_common(values):
    whole = sum(values) / 36
    return tuple(
        sum(values[18 * p : 18 * (p + 1)]) / 18 - whole
        for p in (0, 1)
        for _ in range(18)
    )


def _constants(mp):
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - fine._mp(mp, DELTA) / (18 * mp.pi)
    return gamma, k, 9 * k, 1 + 2 * mp.cos(mp.pi / 9)


def _weights(mp, side):
    result = tuple(
        value
        + (1 if value < fine._mp(mp, Q(7, 8)) else -1)
        * fine._mp(mp, WEIGHT_ERROR)
        * (1 + (j + side) % 3)
        / 6
        for j, value in enumerate(fine._nominal_weights(mp))
    )
    assert all(fine._mp(mp, Q(3, 4)) <= value <= 1 for value in result)
    return result


def _potential(mp, phases):
    angles = tuple(
        2 * mp.pi * fine._mp(mp, turn) + fine._mp(mp, phase)
        for turn, phase in zip(fine._turns() * 2, phases, strict=True)
    )
    reference = tuple(2 * mp.pi * fine._mp(mp, turn) for turn in fine._turns() * 2)
    result = mp.mpf(0)
    for p in (0, 1):
        for i, j, _ in fine._edges(fine._weights(0)):
            a, b = i + 18 * p, j + 18 * p
            result += mp.cos(reference[b] - reference[a]) - mp.cos(
                angles[b] - angles[a]
            )
    return result


def _storage(mp, forms, phases, weights, mu=1):
    x, hx, _, _, _ = port._oracle(mp, forms, phases, weights)
    _, _, c, _ = _constants(mp)
    return (
        fine._dot(x, hx) / 2
        + _potential(mp, phases)
        + c * mu * fine._dot(x, _relative_common(x))
    )


def _regional_storage(mp, forms, phases):
    _, _, c, coefficient = _constants(mp)
    centered = fine._center(tuple(fine._mp(mp, value) for value in forms))
    regional = _regional(centered)
    gaps = tuple(
        port._project(mp, port.PHASE_ROW, phases[18 * p : 18 * (p + 1)]) for p in (0, 1)
    )
    return c * fine._dot(regional, regional) + 9 * coefficient * sum(
        1 - mp.cos(q) for q in gaps
    )


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    def forbidden(*args, **kwargs):
        pytest.fail("storage controls cannot advance or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    result = mpmath.mp.clone()
    result.dps = 220
    return result


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_nominal_port_selection_does_not_select_unique_unit_strength_storage(
    mp, report
):
    gamma, k, c, coefficient = _constants(mp)
    a, r, s = map(mp.mpf, ("0.00003", "-0.00002", "0.00001"))
    vd, vr = (s + r) / 2, (s - r) / 2
    qd, qr = mp.mpf("0.0001"), -mp.mpf("0.00002")
    d = a - r
    for mu in (mp.mpf(0), mp.mpf("0.5"), mp.mpf(1), mp.mpf(2)):
        ad = -a + r
        vdd = -2 * k * vd - gamma * coefficient * mp.sin(qd) / 9 + d / 2
        vrd = -2 * k * vr - gamma * coefficient * mp.sin(qr) / 9 - d / 2
        derivative = 36 * c * (vd * vdd + vr * vrd) + 18 * c * mu * a * ad
        derivative += (
            9
            * coefficient
            * (mp.sin(qd) * 4 * gamma * k * vd + mp.sin(qr) * 4 * gamma * k * vr)
        )
        expected = (
            -4 * c * c * (s * s + r * r)
            + 18 * c * (1 - mu) * r * d
            - 18 * c * mu * d * d
        )
        assert abs(derivative - expected) < mp.mpf("1e-200")
        port_power = 18 * c * ((1 - mu) * a * d - d * d)
        assert abs(derivative + 8 * c * c * (vd * vd + vr * vr) - port_power) < mp.mpf(
            "1e-200"
        )
    # PSD of the two-variable full-loss form permits more than cancellation.
    for mu in (Q(1, 2), Q(2)):
        assert 8 * Q(4, 9) * mu - (1 - mu) ** 2 > 0
    for mu in (mp.mpf("0.5"), mp.mpf(2)):
        d = mp.mpf(1)
        a = 2 / (1 - mu)
        assert 18 * c * ((1 - mu) * a * d - d * d) > 0
    assert report.selected_mu == 1
    assert report.nominal_total_mu_examples == (Q(1, 2), Q(2))
    assert all(value > 0 for value in report.nominal_total_mu_margins)
    assert report.nominal_classification_certified


def test_common_origin_phase_gauges_conjugate_effort_and_coercivity(mp):
    _, _, c, _ = _constants(mp)
    forms = fine._center(tuple(Q((j * 7) % 13 - 6, 10**5) for j in range(36)))
    phases = fine._offset(SECTION / 3) + fine._offset(-SECTION / 5)
    weights = (_weights(mp, 0), _weights(mp, 1))
    x, hx, sine, dx, dy = port._oracle(mp, forms, phases, weights)
    common = _relative_common(x)
    effort = tuple(h + 2 * c * m for h, m in zip(hx, common, strict=True))
    supply = tuple(
        fine._mp(mp, value)
        for value in fine._center(tuple(Q((j * 3) % 7 - 3, 10**6) for j in range(36)))
    )
    natural = fine._dot(effort, dx) - fine._dot(sine, dy)
    supplied = fine._dot(
        effort, tuple(a + b for a, b in zip(dx, supply, strict=True))
    ) - fine._dot(sine, dy)
    assert abs(supplied - natural - fine._dot(effort, supply)) < mp.mpf("1e-200")
    assert abs(fine._dot(common, sine)) < mp.mpf("1e-200")
    value = _storage(mp, forms, phases, weights)
    shifted = tuple(value + Q(2, 7) for value in forms)
    gauged = tuple(
        value + (Q(3, 11) if j < 18 else -Q(2, 9)) for j, value in enumerate(phases)
    )
    assert abs(_storage(mp, shifted, gauged, weights) - value) < mp.mpf("1e-200")
    norm_squared = fine._dot(x, x) + sum(fine._mp(mp, y) ** 2 for y in phases)
    assert norm_squared / 5 <= value <= 9 * norm_squared


def test_hidden_fine_port_power_can_be_positive_for_every_common_coefficient(
    mp, report
):
    nominal = fine._nominal_weights(mp)
    h0 = fine._laplacian(
        tuple((i, j, fine._mp(mp, w)) for i, j, w in fine._edges(nominal))
    )
    b = (mp.mpf(1), -mp.mpf(1)) + (mp.mpf(0),) * 7
    hidden_b = (mp.mpf(0),) * 9 + b
    p = fine._matvec(h0, hidden_b)[:9]
    assert abs(sum(p)) < mp.mpf("1e-200")
    assert fine._dot(p, p) > 0
    single = tuple(
        value * mp.mpf("1e-7") for value in p + tuple(-36 * value for value in b)
    )
    forms = single + tuple(-value for value in single)
    x, hx, _, _, _ = port._oracle(mp, forms, (Q(0),) * 36, (nominal, nominal))
    gx = port._a_exchange(x)
    common = _relative_common(x)
    assert max(abs(value) for value in common) < mp.mpf("1e-200")
    assert max(abs(value) for value in _regional(x)) < mp.mpf("1e-200")
    work = -fine._dot(hx, gx)
    assert work > 0
    _, _, c, _ = _constants(mp)
    for mu in (0, 1, 2, 100):
        effort = tuple(h + 2 * c * mu * m for h, m in zip(hx, common, strict=True))
        assert abs(-fine._dot(effort, gx) - work) < mp.mpf("1e-200")
    total_loss = fine._dot(hx, hx) / 9 + fine._dot(hx, gx)
    assert total_loss >= fine._dot(x, x) / 4
    assert report.fine_port_only_obstruction_certified
    assert not report.fine_port_only_passivity_claimed
    # All states here are static chart witnesses, not preparation endpoints.


def test_noncommuting_positive_contractions_give_total_hidden_dissipation(mp, report):
    # Rank-one contractions make the -1/4 anticommutator bound sharp.
    a = mp.matrix([[1, 0], [0, 0]])
    b = mp.matrix([[mp.mpf(1) / 4, mp.sqrt(3) / 4], [mp.sqrt(3) / 4, mp.mpf(3) / 4]])
    jordan = a * b + b * a
    eigenvalues, _ = mp.eigsy(jordan)
    assert abs(eigenvalues[0] + mp.mpf(1) / 4) < mp.mpf("1e-200")
    assert mp.norm(a * b - b * a) > 0
    gap = (a + b) ** 2 - a - b + mp.eye(2) / 4
    assert min(mp.eigsy(gap, eigvals_only=True)) >= -mp.mpf("1e-200")
    assert _helper() == (Q(1, 36), Q(1, 4))
    assert report.hidden_bracket == Q(1, 36)
    assert report.hidden_coercivity_lower_bound == Q(1, 4)
    assert report.hidden_dissipation_certified
    # Regional coercivity is independent of the hidden matrix argument.
    for a0, r0, s0 in ((Q(2), Q(1), Q(0)), (Q(-3), Q(2), Q(1)), (Q(0), Q(0), Q(1))):
        norm = 9 * (a0 * a0 + r0 * r0 + s0 * s0)
        minimal_loss = 72 * (a0 - r0) ** 2 + 64 * (r0 * r0 + s0 * s0)
        assert minimal_loss >= 2 * norm


def test_actual_weighted_balance_and_six_delta_perturbation_are_complete(mp, report):
    _, _, c, _ = _constants(mp)
    forms = fine._center(tuple(Q((j * 5) % 17 - 8, 10**5) for j in range(36)))
    phases = tuple(
        base + Q((j % 3) - 1, 10**8)
        for j, base in enumerate(fine._offset(SECTION / 2) + fine._offset(-SECTION / 7))
    )
    actual = (_weights(mp, 0), _weights(mp, 1))
    nominal = (fine._nominal_weights(mp),) * 2
    x, hx, sine, dx, dy = port._oracle(mp, forms, phases, actual)
    h0x = port._oracle(mp, forms, phases, nominal)[1]
    gx, common = port._a_exchange(x), _relative_common(x)
    effort = tuple(h + 2 * c * m for h, m in zip(hx, common, strict=True))
    derivative = fine._dot(effort, dx) - fine._dot(sine, dy)
    loss = fine._dot(hx, hx) / 9 + fine._dot(effort, gx)
    assert abs(derivative + loss) < mp.mpf("1e-200")
    nominal_effort = tuple(h + 2 * c * m for h, m in zip(h0x, common, strict=True))
    nominal_loss = fine._dot(h0x, h0x) / 9 + fine._dot(nominal_effort, gx)
    norm_squared = fine._dot(x, x)
    assert abs(loss - nominal_loss) <= fine._mp(mp, 84 * WEIGHT_ERROR) * norm_squared
    assert loss >= fine._mp(mp, Q(1, 4) - 84 * WEIGHT_ERROR) * norm_squared
    assert report.nominal_dissipation_lower_bound == Q(1, 4)
    assert report.actual_weight_perturbation_upper_bound == 84 * WEIGHT_ERROR
    assert report.actual_dissipation_lower_bound == Q(1, 4) - 84 * WEIGHT_ERROR > 0
    assert report.actual_dissipation_certified


def test_fine_energy_effort_and_work_remainders_use_actual_regional_projection(mp):
    _, _, c, _ = _constants(mp)
    carrier_x = (Q(1, 10**6),) * 9 + (Q(3, 10**6),) * 9 + (-Q(2, 10**6),) * 18
    carrier_y = fine._offset(SECTION / 2) + fine._offset(-SECTION / 11)
    hidden_x = tuple(
        value
        for region in range(4)
        for value in fine._center(
            tuple(Q((j + region) % 5 - 2, 10**14) for j in range(9))
        )
    )
    hidden_y = tuple(
        value
        for region in range(4)
        for value in fine._center(
            tuple(Q((j + region) % 7 - 3, 10**10) for j in range(9))
        )
    )
    forms = tuple(a + b for a, b in zip(carrier_x, hidden_x, strict=True))
    phases = tuple(a + b for a, b in zip(carrier_y, hidden_y, strict=True))
    weights = (_weights(mp, 0), _weights(mp, 1))
    x, hx, _, _, _ = port._oracle(mp, forms, phases, weights)
    hx_hidden = tuple(fine._mp(mp, value) for value in hidden_x)
    hy_hidden = tuple(fine._mp(mp, value) for value in hidden_y)
    norm_squared = fine._dot(x, x) + sum(fine._mp(mp, y) ** 2 for y in phases)
    error = _storage(mp, forms, phases, weights) - _regional_storage(mp, forms, phases)
    upper = (
        9 * (fine._dot(hx_hidden, hx_hidden) + fine._dot(hy_hidden, hy_hidden))
        + fine._mp(mp, 7 * WEIGHT_ERROR) * norm_squared
    )
    assert abs(error) <= upper
    regional = _regional(x)
    effort = tuple(h + 2 * c * m for h, m in zip(hx, _relative_common(x), strict=True))
    coarse_effort = _regional(effort)
    effort_error = tuple(
        a - 2 * c * b for a, b in zip(coarse_effort, regional, strict=True)
    )
    assert mp.sqrt(fine._dot(effort_error, effort_error)) <= fine._mp(
        mp, 14 * WEIGHT_ERROR
    ) * mp.sqrt(norm_squared)
    work = -fine._dot(effort, port._a_exchange(x))
    coarse_work = -2 * c * fine._dot(regional, port._a_exchange(regional))
    allowance = (
        36 * fine._dot(hx_hidden, hx_hidden)
        + fine._mp(mp, 28 * WEIGHT_ERROR) * norm_squared
    )
    assert abs(work - coarse_work) <= allowance


def test_reference_storage_needs_its_own_coarse_coordinate_allowance(mp, report):
    nominal = (fine._nominal_weights(mp),) * 2
    forms = (Q(0),) * 36
    reference = fine._offset(SECTION / 2) + fine._offset(Q(0))
    actual = fine._offset(SECTION / 2 + Q(1, 10**8)) + fine._offset(Q(0))
    hidden_error = _storage(mp, forms, actual, nominal) - _regional_storage(
        mp, forms, actual
    )
    reference_error = _regional_storage(mp, forms, actual) - _regional_storage(
        mp, forms, reference
    )
    assert abs(hidden_error) < mp.mpf("1e-200")
    assert reference_error > fine._mp(mp, report.storage_remainder_upper_bound)
    assert reference_error < fine._mp(
        mp, report.projected_reference_storage_error_upper_bound
    )
    # This compares two static nearby regional states, not two allowed source
    # histories. Its purpose is to expose the missing coarse-coordinate term.
    ux, uy, norm = (
        report.form_error_upper_bound,
        report.phase_error_upper_bound,
        report.norm_upper_bound,
    )
    assert (
        report.storage_remainder_upper_bound
        == 9 * (ux * ux + uy * uy) + 7 * WEIGHT_ERROR * norm * norm
    )
    assert report.projected_effort_error_upper_bound == 14 * WEIGHT_ERROR * norm
    assert (
        report.port_power_remainder_upper_bound
        == 36 * ux * ux + 28 * WEIGHT_ERROR * norm * norm
    )
    assert (
        report.accumulated_port_work_remainder_upper_bound
        == HORIZON * report.port_power_remainder_upper_bound
    )
    assert (
        report.projected_reference_storage_error_upper_bound
        == 9 * norm * ux + 18 * norm * uy
    )
    assert (
        report.total_reference_storage_error_upper_bound
        == report.storage_remainder_upper_bound
        + report.projected_reference_storage_error_upper_bound
    )


def test_source_budget_retains_fixed_unit_port_horizon_and_no_energy_measurement(
    report,
):
    source = report.source
    assert source.observation_error_bound == 0
    assert report.form_error_upper_bound == source.positive_form_error_upper_bound
    assert report.phase_error_upper_bound == source.positive_phase_error_upper_bound
    assert report.norm_upper_bound == source.new_norm_upper_bound
    assert report.horizon == source.positive_duration == HORIZON
    assert source.contact_pairs == tuple((j, 18 + j) for j in range(9))
    assert report.source_association_certified and report.storage_domain_certified
    assert report.finite_inheritance_certified and report.certified
    assert report.storage_comparison_target == "actual_regional_projection"
    assert (
        report.work_comparison_target
        == "regional_port_work_at_actual_projected_coordinates"
    )
    assert report.extra_reference_price_included
    assert report.independently_observed_aggregates == 0
    assert report.nominal_port_mu_unique_within_ansatz
    assert report.fixed_unit_nominal_total_storage_nonunique
    assert report.fixed_port_strength == 1
    assert not report.all_port_strength_total_passivity_claimed
    assert not report.canonical_storage_selected
    assert not report.phase_potential_selected
    assert not report.physical_energy_identified
    assert not report.old_form_or_phase_record_consumed
    assert not report.state_reset_performed
    assert not report.untracked_reservoir_introduced
    assert not report.source_trajectory_evaluated
    assert not report.runtime_event_executed
    assert not report.new_all_time_joint_registration_claimed
    assert not report.empirical_validation
    # The helper itself records the limit of a stronger port: it cannot
    # silently extend the unit-strength dissipation certificate.
    assert _helper(port_norm_upper=Q(4))[0] < 0
    assert _helper(port_norm_upper=Q(4))[1] is None
    with pytest.raises(FrozenInstanceError):
        report.selected_mu = Q(2)


def test_old_form_and_phase_record_conclusions_are_not_energy_premises(
    monkeypatch, report
):
    source = report.source
    handoff = replace(
        source.handoff,
        certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        switched_record_bounds=(Q(0), Q(0)),
        status="old_phase_unused",
        unresolved_conditions=("unused",),
    )
    changed = replace(
        source,
        handoff=handoff,
        certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        positive_record_bounds=(Q(0), Q(0)),
        cut_record_bounds=(Q(0), Q(0)),
        status="old_form_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_positive_interface", lambda **kwargs: changed
    )
    result = _assess()
    assert result.certified and result.actual_dissipation_certified
    assert result.storage_remainder_upper_bound == report.storage_remainder_upper_bound
    assert (
        result.port_power_remainder_upper_bound
        == report.port_power_remainder_upper_bound
    )


def test_source_geometry_and_component_premises_are_readmitted(monkeypatch, report):
    source = report.source
    for name, value in (
        ("positive_duration", Q(1)),
        ("handoff_time", Q(0)),
        ("physical_capacities", (Q(2),) * 36),
        ("initial_donor_form_total", Q(1)),
        ("new_norm_upper_bound", source.new_norm_upper_bound / 2),
        ("positive_form_error_upper_bound", Q(0)),
        ("positive_phase_error_upper_bound", Q(0)),
        ("comparison_forcing_upper_bound", WEIGHT_ERROR),
        ("component_comparison_certified", False),
        ("state_reset_performed", True),
        ("contact_pairs", tuple((j, 18 + j) for j in range(18))),
    ):
        changed = replace(source, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_positive_interface", lambda **kwargs: changed
        )
        assert not _assess().certified
    reference = source.handoff.prefix.reference
    for name, value in (
        ("original_relative_phase_gap", SECTION),
        ("initial_form_means", (Q(0), Q(1))),
        ("actual_weight_deviation_upper_bound", 2 * WEIGHT_ERROR),
    ):
        changed_handoff = port._replace_reference(source.handoff, **{name: value})
        changed = replace(source, handoff=changed_handoff)
        monkeypatch.setattr(
            owner, "_assess_mixed_positive_interface", lambda **kwargs: changed
        )
        assert not _assess().certified
    sources = list(reference.sources)
    sources[0] = replace(sources[0], edge_weight_deviation_upper_bound=2 * WEIGHT_ERROR)
    changed_handoff = port._replace_reference(
        source.handoff,
        sources=tuple(sources),
        donor=replace(reference.donor, source=sources[0]),
    )
    changed = replace(source, handoff=changed_handoff)
    monkeypatch.setattr(
        owner, "_assess_mixed_positive_interface", lambda **kwargs: changed
    )
    assert not _assess().certified


def test_shared_noncommuting_kernel_handles_zero_boundary_negative_and_raw_domains(
    monkeypatch,
):
    assert _helper(port_norm_upper=Q(0)) == (Q(1, 9), Q(1))
    assert _helper(port_norm_upper=Q(8, 3)) == (Q(0), Q(0))
    assert _helper(port_norm_upper=Q(4)) == (-Q(1, 18), None)
    assert (
        _helper(
            transport_gap_lower=Q(np.int64(3)),
            port_norm_upper=Q(np.int64(2)),
            loss_divisor=Q(np.int64(9)),
        )
        == _helper()
    )
    for name in ("transport_gap_lower", "port_norm_upper", "loss_divisor"):
        for bad in (
            True,
            np.bool_(False),
            1,
            1.0,
            None,
            "1",
            float("nan"),
            float("inf"),
            complex(1, 2),
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _helper(**{name: bad})
        with pytest.raises(ValueError):
            _helper(**{name: Q(-1)})
    for name in ("transport_gap_lower", "loss_divisor"):
        with pytest.raises(ValueError):
            _helper(**{name: Q(0)})
    calls = []
    normalize = kernel_owner._quadratic_rational

    def capture(value, label):
        calls.append(label)
        return normalize(value, label)

    monkeypatch.setattr(kernel_owner, "_quadratic_rational", capture)
    with pytest.raises(TypeError):
        _helper(transport_gap_lower=Q(-1), loss_divisor=True)
    assert calls == ["transport_gap_lower", "port_norm_upper", "loss_divisor"]


def test_reader_raw_pairs_one_shot_and_no_storage_fitting_parameter(
    monkeypatch, report
):
    error = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    assert (
        _assess(
            form_error_bounds=(error for _ in range(2)),
            phase_error_bounds=(error for _ in range(2)),
        )
        == report
    )
    for name in (
        "mu",
        "storage",
        "energy_reading",
        "port_strength",
        "horizon",
        "reference",
        "source",
        "observation_error_bound",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
    for name in ("form_error_bounds", "phase_error_bounds"):
        values = (ERROR + Q(1, 10**100), ERROR)
        assert not _assess(**{name: values}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source or derived arithmetic")

    monkeypatch.setattr(owner, "_assess_mixed_positive_interface", forbidden)
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in (
            True,
            np.bool_(False),
            None,
            "0",
            float("nan"),
            float("inf"),
            complex(1, 2),
            -Q(1, 10**100),
        ):
            for side in (0, 1):
                values = tuple(bad if j == side else Q(0) for j in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: values})
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})
