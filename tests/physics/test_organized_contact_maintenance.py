"""Full-state modes, capture premises and unavailable continued-contact claims.

Static algebraic states verify the geometry without running a contacted
trajectory or replaying a frozen preparation producer.
"""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_contact_maintenance as owner
from tnfr.physics import _organized_external_interaction as external


_ERROR = Q(1, 10**10)
_EDGES = (
    tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(6) for j in range(9))
    + tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
    + tuple((9 * c + j, 45 + j) for c in range(5) for j in range(9))
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def _assess(**changes):
    return owner._assess_organized_contact_maintenance(
        **(
            dict(form_error_bounds=(_ERROR,) * 6, phase_error_bounds=(_ERROR,) * 6)
            | changes
        )
    )


@pytest.fixture(scope="module")
def geometry():
    return owner._derive_organized_contact_maintenance()


@pytest.fixture(scope="module")
def report():
    return _assess()


def _edge_action(mp, vector, phases=None):
    result = [mp.mpc(0)] * 54
    for left, right in _EDGES:
        weight = 1 if phases is None else mp.cos(phases[right] - phases[left])
        current = weight * (vector[left] - vector[right])
        result[left] += current
        result[right] -= current
    return tuple(result)


def test_stationary_resultants_do_not_make_unrestricted_stress_positive(mp):
    """Static controls distinguish tangent curvature from unrestricted stress."""
    alpha = 2 * mp.pi / 9
    tolerance = mp.mpf("1e-165")
    degrees = [0] * 54
    phases = tuple(alpha * (j - 4) for _ in range(6) for j in range(9))
    phasors = tuple(mp.exp(mp.j * phase) for phase in phases)
    resultants = [mp.mpc(0)] * 54
    for a, b in _EDGES:
        resultants[a] += phasors[b]
        resultants[b] += phasors[a]
        degrees[a] += 1
        degrees[b] += 1
    radial = tuple(mp.conj(z) * resultant for z, resultant in zip(phasors, resultants))
    assert degrees == [5] * 45 + [7] * 9
    assert all(abs(value.imag) < tolerance for value in radial)
    lower_bound = 5 - 2 * mp.sqrt(6)
    assert lower_bound > mp.mpf("0.1")
    for i, value in enumerate(radial):
        expected = 2 * mp.cos(alpha) + (3 if i < 45 else 5)
        assert abs(value.real - expected) < tolerance
        assert value.real > lower_bound
        assert abs(phasors[i] - resultants[i] / abs(resultants[i])) < tolerance
        # Stationarity is H*z=0, where H=diag(r)-Adj; it is not H>=0.
        assert abs(value.real * phasors[i] - resultants[i]) < tolerance
        cosine_square_sum = sum(
            mp.cos(phases[b] - phases[a]) ** 2 for a, b in _EDGES if i in (a, b)
        )
        assert cosine_square_sum >= 1
        assert value.real**2 + cosine_square_sum <= 10 * value.real
    potential = sum(1 - mp.cos(phases[b] - phases[a]) for a, b in _EDGES)
    stress_on_one = tuple(value.real - degree for value, degree in zip(radial, degrees))
    assert all(
        abs(value - (2 * mp.cos(alpha) - 2)) < tolerance for value in stress_on_one
    )
    assert abs(sum(stress_on_one) + 2 * potential) < tolerance
    assert sum(stress_on_one) < 0
    # All phase-Hessian weights are positive on this connected support.
    # The independent full-mode control below verifies the relative spectrum.
    assert min(mp.cos(phases[b] - phases[a]) for a, b in _EDGES) > 0
    vector = tuple(mp.mpf(2 * i - 53) / 108 for i in range(54))
    curvature = sum(
        v * value for v, value in zip(vector, _edge_action(mp, vector, phases))
    ).real
    form_curvature = sum(
        v * value for v, value in zip(vector, _edge_action(mp, vector))
    ).real
    assert curvature >= mp.cos(alpha) * form_curvature - tolerance
    assert form_curvature > 0

    antipodal = tuple(
        phase + (mp.pi if i >= 45 else 0) for i, phase in enumerate(phases)
    )
    anti_z = tuple(mp.exp(mp.j * phase) for phase in antipodal)
    anti_resultants = [mp.mpc(0)] * 54
    for a, b in _EDGES:
        anti_resultants[a] += anti_z[b]
        anti_resultants[b] += anti_z[a]
    anti_radial = tuple(mp.conj(z) * value for z, value in zip(anti_z, anti_resultants))
    assert all(abs(value.imag) < tolerance for value in anti_radial)
    for value in anti_radial[45:]:
        assert abs(value.real - (2 * mp.cos(alpha) - 5)) < tolerance
        assert value.real < 0
    donor_cut = tuple(mp.mpf(i >= 45) - mp.mpf(1) / 6 for i in range(54))
    negative = sum(
        value * action
        for value, action in zip(donor_cut, _edge_action(mp, donor_cut, antipodal))
    ).real
    assert abs(negative + 45) < tolerance


def test_static_slot_shift_secant_and_endpoint_hessian_defect(mp):
    """The nonstationary blind point is a secant control, not a target verdict."""
    gaps = (Q(1, 6),) * 7 + (Q(-1, 12),) * 2
    profile = tuple(sum(gaps[:j], Q(0)) for j in range(9))
    raw = tuple(
        profile[j] + (Q(c - 2, 6) if c < 5 else 0) for c in range(6) for j in range(9)
    )
    mean = sum(raw, Q(0)) / 54
    turns = tuple(value - mean for value in raw)
    shifted = tuple(
        turns[9 * c + (j + 1) % 9] + int(j == 8) - Q(1, 9)
        for c in range(6)
        for j in range(9)
    )
    assert sum(turns) == sum(shifted) == 0
    assert tuple(shifted[i] - turns[i] for i in range(54)) == tuple(
        gaps[j] - Q(1, 9) for _ in range(6) for j in range(9)
    )
    branches = tuple(-int(a % 9 == 8) for a, _ in _EDGES[:54])
    branches += tuple(-int(a // 9 == 4) for a, _ in _EDGES[54:99])
    branches += (0,) * 45
    gap_rows = []
    for state in (turns, shifted):
        edge_gaps = tuple(
            state[b] - state[a] - branch for (a, b), branch in zip(_EDGES, branches)
        )
        gap_rows.append(edge_gaps)
        assert all(-Q(1, 2) < value < Q(1, 2) for value in edge_gaps[:99])
        assert all(sum(edge_gaps[9 * c : 9 * c + 9]) == 1 for c in range(6))
        assert all(
            sum(edge_gaps[54 + 9 * c + j] for c in range(5)) == 1 for j in range(9)
        )
        for c in range(5):
            for j in range(9):
                mixed_square = (
                    edge_gaps[9 * c + j]
                    + edge_gaps[54 + 9 * c + (j + 1) % 9]
                    - edge_gaps[9 * ((c + 1) % 5) + j]
                    - edge_gaps[54 + 9 * c + j]
                )
                assert mixed_square == 0
    assert gap_rows[0][:9] == gaps
    assert gap_rows[1][:9] == gaps[1:] + gaps[:1]
    assert gap_rows[0][54:] == gap_rows[1][54:]

    phases = tuple(2 * mp.pi * _mp(mp, value) for value in turns)
    plus = tuple(2 * mp.pi * _mp(mp, value) for value in shifted)
    direction = tuple(after - before for before, after in zip(phases, plus))
    current_rows = []
    for state in (phases, plus):
        current = [mp.mpf(0)] * 54
        for a, b in _EDGES:
            value = mp.sin(state[b] - state[a])
            current[a] += value
            current[b] -= value
        current_rows.append(current)
    tolerance = mp.mpf("1e-165")
    assert abs(current_rows[0][1] - mp.sqrt(3) / 2) < tolerance
    secant = mp.mpf(0)
    endpoint_edges = mp.mpf(0)
    for old, new in zip(*gap_rows):
        a, b = 2 * mp.pi * _mp(mp, old), 2 * mp.pi * _mp(mp, new)
        secant += (mp.sin(b) - mp.sin(a)) * (b - a)
        endpoint_edges += (mp.cos(a) + mp.cos(b)) * (b - a) ** 2 / 2
    expected_secant = 3 * mp.pi * (mp.sqrt(3) + 1)
    expected_endpoint = 3 * mp.pi**2 * (mp.sqrt(3) + 1) / 4
    assert abs(secant - expected_secant) < tolerance
    assert secant > 0
    current_pairing = sum(
        h * (after - before)
        for h, before, after in zip(direction, current_rows[0], current_rows[1])
    )
    assert abs(current_pairing + secant) < tolerance
    quadratic = tuple(
        sum(
            h * action
            for h, action in zip(direction, _edge_action(mp, direction, state))
        ).real
        for state in (phases, plus)
    )
    assert all(abs(value - expected_endpoint) < tolerance for value in quadratic)
    average = sum(quadratic) / 2
    assert abs(average - endpoint_edges) < tolerance
    midpoint = tuple((a + b) / 2 for a, b in zip(phases, plus))
    midpoint_quadratic = sum(
        h * action
        for h, action in zip(direction, _edge_action(mp, direction, midpoint))
    ).real
    # Every edge with a nonzero directional gap changes by exactly pi/2.
    assert (
        abs(secant - mp.sin(mp.pi / 4) / (mp.pi / 4) * midpoint_quadratic) < tolerance
    )
    assert abs(average - mp.cos(mp.pi / 4) * midpoint_quadratic) < tolerance
    assert abs((average - secant) - expected_secant * (mp.pi / 4 - 1)) < tolerance
    assert average - secant < 0


@pytest.mark.parametrize("antipodal", (False, True))
def test_every_full_spatial_mode_against_independent_edge_matrices(
    mp, geometry, antipodal
):
    """Apply the actual 54-node Laplacian and Hessian to complex Fourier modes."""
    phases = tuple(
        2 * mp.pi * (j - 4) / 9 + (mp.pi if antipodal and c == 5 else 0)
        for c in range(6)
        for j in range(9)
    )
    gamma = 1 / (1023 * mp.pi)
    stable = unstable = neutral = 0
    for mode in geometry.modes:
        if mode.outer_mode == "common":
            outer = (mp.mpc(1),) * 6
        elif mode.outer_mode == "hub":
            outer = (mp.mpc(1),) * 5 + (mp.mpc(-5),)
        else:
            frequency = int(mode.outer_mode[4:])
            outer = tuple(mp.exp(2j * mp.pi * frequency * c / 5) for c in range(5))
            outer += (mp.mpc(0),)
        vector = tuple(
            outer[c] * mp.exp(2j * mp.pi * mode.internal_mode * j / 9)
            for c in range(6)
            for j in range(9)
        )
        norm = sum(abs(value) ** 2 for value in vector)
        laplacian_action = _edge_action(mp, vector)
        hessian_action = _edge_action(mp, vector, phases)
        laplacian = (
            sum(mp.conj(v) * action for v, action in zip(vector, laplacian_action)).real
            / norm
        )
        hessian = (
            sum(mp.conj(v) * action for v, action in zip(vector, hessian_action)).real
            / norm
        )
        for value, left, right in zip(vector, laplacian_action, hessian_action):
            assert abs(left - laplacian * value) < mp.mpf("1e-165")
            assert abs(right - hessian * value) < mp.mpf("1e-165")
        hessian_box = (
            mode.antipodal_hessian_bounds if antipodal else mode.aligned_hessian_bounds
        )
        determinant_box = (
            mode.antipodal_determinant_bounds
            if antipodal
            else mode.aligned_determinant_bounds
        )
        if (mode.internal_mode, mode.outer_mode) == (0, "common"):
            assert mode.laplacian_bounds == I(0)
            assert hessian_box == I(0)
            assert determinant_box == I(0)
            neutral += 2
            continue
        _contains(mp, mode.laplacian_bounds, laplacian)
        _contains(mp, hessian_box, hessian)
        _contains(mp, mode.trace_bounds, -laplacian / 4)
        _contains(mp, determinant_box, gamma**2 * laplacian * hessian / 16)
        # The two roots use both complete tangent rows, including quarter time.
        discriminant = laplacian**2 - 4 * gamma**2 * laplacian * hessian
        roots = (
            (-laplacian + mp.sqrt(discriminant)) / 8,
            (-laplacian - mp.sqrt(discriminant)) / 8,
        )
        stable += sum(value.real < 0 for value in roots)
        unstable += sum(value.real > 0 for value in roots)
    assert neutral == geometry.conserved_mean_modes == 2
    if antipodal:
        assert stable == geometry.antipodal_relative_stable_modes == 97
        assert unstable == geometry.antipodal_relative_unstable_modes == 9
    else:
        assert stable == geometry.aligned_relative_stable_modes == 106
        assert unstable == 0


def test_geometry_covers_actual_product_and_retains_neutral_modes(geometry):
    assert len(geometry.modes) == 54
    assert len({(m.internal_mode, m.outer_mode) for m in geometry.modes}) == 54
    assert geometry.aligned_stability_certified
    assert geometry.antipodal_instability_certified
    assert geometry.near_carrier_obstruction_certified
    neutral = next(
        mode
        for mode in geometry.modes
        if (mode.internal_mode, mode.outer_mode) == (0, "common")
    )
    assert neutral.trace_bounds == I(0)
    assert all(
        mode.antipodal_determinant_bounds.hi < 0
        for mode in geometry.modes
        if mode.outer_mode == "hub"
    )
    assert geometry.interaction_geometry.external_phase_origin_bounds == I(0)
    assert geometry.antipodal_donor_phase_origin_bounds.lo > 3


@pytest.mark.parametrize("beta", (Q(-2), Q(0), Q(7, 5)))
@pytest.mark.parametrize("fine_amplitude", (Q(0), Q(1, 100)))
def test_negative_hessian_witness_and_fine_defect_without_equilibrium_claim(
    mp, geometry, beta, fine_amplitude
):
    phases, offsets = [], []
    for c in range(6):
        for j in range(9):
            carrier = 2 * mp.pi * (c - 2) / 5 if c < 5 else _mp(mp, beta)
            fine = _mp(mp, fine_amplitude) * mp.sin(2 * mp.pi * j / 9 + c)
            offsets.append(carrier + fine)
            phases.append(2 * mp.pi * (j - 4) / 9 + carrier + fine)
    vectors = (
        tuple(mp.sin(value) for value in offsets),
        tuple(mp.cos(value) for value in offsets),
    )
    total = mp.mpf(0)
    for vector in vectors:
        mean = sum(vector) / 54
        projected = tuple(value - mean for value in vector)
        unprojected = sum(
            mp.cos(phases[b] - phases[a]) * (vector[b] - vector[a]) ** 2
            for a, b in _EDGES
        )
        quadratic = sum(
            mp.cos(phases[b] - phases[a]) * (projected[b] - projected[a]) ** 2
            for a, b in _EDGES
        )
        assert abs(quadratic - unprojected) < mp.mpf("1e-165")
        total += quadratic
    defect = sum(1 - mp.cos(offsets[b] - offsets[a]) for a, b in _EDGES[:54])
    exact_threshold = 45 * (7 - 3 * mp.sqrt(5)) / 8
    _contains(mp, geometry.internal_phase_defect_threshold_bounds, exact_threshold)
    assert defect < exact_threshold
    assert total < 2 * (defect - exact_threshold) < 0
    # These static witnesses are generally moving states, not fictitious equilibria.
    currents = [mp.mpf(0)] * 54
    for a, b in _EDGES:
        current = mp.sin(phases[b] - phases[a])
        currents[a] += current
        currents[b] -= current
    assert max(abs(value) for value in currents) > mp.mpf("0.1")


def test_actual_control_capture_keeps_source_history_and_scope(report):
    assert report.source_certified
    assert report.control_capture_certified
    assert report.control_identity_retention_certified
    assert report.control_convergence_certified
    assert report.admitted
    assert report.unresolved_conditions == ()
    assert report.relative_phase_origin == 0
    assert report.relative_preparation_dimension == 96
    assert report.protocol_revision == "4c4addf1"
    assert report.control_initial_excess_upper_bound < Q(1, 100)
    assert report.control_initial_excess_upper_bound < Q(2, 10**12)
    assert report.control_capture_barrier_lower_bound > Q(6, 25)
    assert report.control_attachment_work_upper_bound < Q(1, 10**12)
    assert all(value > 0 for _, value in report.sufficient_margins)
    assert report.outer_one_initial_defect_domain_certified
    assert report.outer_one_initial_defect_margin > 1
    assert report.outer_one_maintenance_status.startswith("unavailable_")
    assert report.other_origin_maintenance_status.startswith("unavailable_")
    assert "no_all_future_positive_individual_inner_gap_claim" in report.scope
    assert (
        "initial_defect_domain_inclusion_does_not_prove_it_is_retained" in report.scope
    )
    assert (
        report.external_interaction.observation_protocol.total_absolute_event_work_upper_bound
        > 0
    )


def test_sharper_control_bounds_derive_from_whole_word_not_generic_state_class(report):
    source = report.external_interaction
    host, observation = source.host_acquisition, source.observation_protocol
    gamma = report.geometry.interaction_geometry.gamma_bounds.hi
    duration = observation.total_elapsed_scaled_time
    b = host.full_form_error_upper_bound + gamma * duration
    e = host.full_phase_error_upper_bound + 2 * gamma * b * duration
    assert report.control_host_form_upper_bound == b
    assert report.control_host_phase_offset_upper_bound == e
    assert b < source.observation_form_upper_bound
    assert (
        e < host.full_phase_error_upper_bound + source.carried_phase_drift_upper_bound
    )
    assert (
        report.control_form_excess_upper_bound
        == 288 * report.control_joined_form_upper_bound**2
    )
    assert (
        report.control_phase_excess_upper_bound
        == 288 * report.control_joined_phase_offset_upper_bound**2
    )
    assert report.control_initial_excess_upper_bound == (
        report.control_form_excess_upper_bound + report.control_phase_excess_upper_bound
    )
    assert (
        report.internal_boundary_excess_bounds is host.internal_boundary_excess_bounds
    )
    assert report.control_capture_barrier_lower_bound == min(
        host.internal_boundary_excess_bounds.lo, 1
    )


def test_fine_state_storage_and_actual_contact_work_fit_complete_bounds(mp, report):
    """Independent node perturbations exercise all three kinds of joined edges."""
    b = _mp(mp, report.control_joined_form_upper_bound)
    e = _mp(mp, report.control_joined_phase_offset_upper_bound)
    forms = tuple(b * mp.sin(2 * mp.pi * i / 54) for i in range(54))
    errors = tuple(e * mp.cos(2 * mp.pi * 7 * i / 54) for i in range(54))
    target = tuple(2 * mp.pi * (i % 9 - 4) / 9 for i in range(54))
    actual_excess = mp.mpf(0)
    for a, z in _EDGES:
        angle = target[z] - target[a]
        actual_excess += (forms[z] - forms[a]) ** 2 / 2
        actual_excess += mp.cos(angle) - mp.cos(angle + errors[z] - errors[a])
    assert 0 <= actual_excess <= _mp(mp, report.control_initial_excess_upper_bound)
    # Attachment uses the sharper host and donor radii, rather than the joined box.
    host_b = _mp(mp, report.control_host_form_upper_bound)
    host_e = _mp(mp, report.control_host_phase_offset_upper_bound)
    donor_radius = mp.mpf("1e-70")
    work = mp.mpf(0)
    for a, z in _EDGES[99:]:
        host_form = host_b * mp.sin(a)
        donor_form = donor_radius * mp.cos(z)
        host_phase = host_e * mp.cos(a)
        donor_phase = donor_radius * mp.sin(z)
        work += (host_form - donor_form) ** 2 / 2
        work += 1 - mp.cos(host_phase - donor_phase)
    assert 0 < work <= _mp(mp, report.control_attachment_work_upper_bound)


def test_unavailable_obstruction_does_not_erase_independent_control_capture(
    monkeypatch, report
):
    monkeypatch.setattr(
        owner,
        "_assess_organized_external_interaction",
        lambda **_: report.external_interaction,
    )
    monkeypatch.setattr(
        owner,
        "_derive_organized_contact_maintenance",
        lambda: replace(report.geometry, near_carrier_obstruction_certified=False),
    )
    result = _assess()
    assert not result.outer_one_initial_defect_domain_certified
    assert result.control_capture_certified
    assert result.control_identity_retention_certified
    assert result.outer_one_maintenance_status.startswith("unavailable_")


def test_unavailable_target_stability_abstains_without_losing_geometry_scope(
    monkeypatch, report
):
    monkeypatch.setattr(
        owner,
        "_assess_organized_external_interaction",
        lambda **_: report.external_interaction,
    )
    monkeypatch.setattr(
        owner,
        "_derive_organized_contact_maintenance",
        lambda: replace(report.geometry, aligned_stability_certified=False),
    )
    result = _assess()
    assert result.source_certified
    assert not result.control_capture_certified
    assert not result.control_identity_retention_certified
    assert not result.control_convergence_certified
    assert "aligned_target_not_certified" in result.unresolved_conditions


def test_source_rebuilt_once_and_old_fixed_origin_verdict_is_not_consumed(
    monkeypatch, report
):
    source = replace(
        report.external_interaction,
        admitted=False,
        comparison_certified=False,
        identity_certified=False,
        work_certified=False,
        observed_contrast_bounds=I(-1),
    )
    calls = []

    def rebuild(**kwargs):
        calls.append(kwargs)
        return source

    monkeypatch.setattr(owner, "_assess_organized_external_interaction", rebuild)
    result = _assess()
    assert calls == [
        dict(form_error_bounds=(_ERROR,) * 6, phase_error_bounds=(_ERROR,) * 6)
    ]
    assert result.external_interaction is source
    assert result.control_capture_certified
    assert result.admitted


@pytest.mark.parametrize(
    "flag_index,condition",
    (
        (0, "host_acquisition_not_certified"),
        (1, "paired_host_control_not_certified"),
        (2, "external_recovery_not_certified"),
        (3, "preceding_sequential_observation_not_certified"),
        (4, "actual_hosts_outside_observation_class"),
    ),
)
def test_missing_actual_source_premise_abstains_in_every_capture_field(
    monkeypatch, report, flag_index, condition
):
    flags = list(report.external_interaction.source_admission_flags)
    flags[flag_index] = False
    source = replace(
        report.external_interaction,
        source_admission_flags=tuple(flags),
        source_certified=False,
    )
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **_: source
    )
    result = _assess()
    assert not result.source_certified
    assert not result.control_capture_certified
    assert not result.control_identity_retention_certified
    assert not result.control_convergence_certified
    assert not result.outer_one_initial_defect_domain_certified
    assert not result.admitted
    assert condition in result.unresolved_conditions
    assert result.geometry.aligned_stability_certified


@pytest.mark.parametrize("boundary", (I(-1), I(0), I(Q(1, 10**15))))
def test_rebuilt_boundary_must_dominate_actual_excess(monkeypatch, report, boundary):
    host = replace(
        report.external_interaction.host_acquisition,
        internal_boundary_excess_bounds=boundary,
    )
    source = replace(report.external_interaction, host_acquisition=host)
    monkeypatch.setattr(
        owner, "_assess_organized_external_interaction", lambda **_: source
    )
    result = _assess()
    assert not result.control_capture_certified
    assert not result.control_identity_retention_certified
    assert "control_capture_boundary" in result.unresolved_conditions


@pytest.mark.parametrize("parameter", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "invalid",
    (
        True,
        float("nan"),
        float("inf"),
        Decimal("0.0"),
        Q(-1, 10**100),
        _ERROR + Q(1, 10**100),
    ),
)
def test_invalid_primitive_allowance_rejected_before_source_rebuild(
    monkeypatch, parameter, invalid
):
    def forbidden(**_):
        pytest.fail("source construction occurred before primitive admission")

    monkeypatch.setattr(external, "_assess_organized_outer_winding", forbidden)
    values = [_ERROR] * 6
    values[-1] = invalid
    with pytest.raises((TypeError, ValueError)):
        _assess(**{parameter: values})


@pytest.mark.parametrize("parameter", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("length", (0, 5, 7))
def test_wrong_source_arity_rejected(monkeypatch, parameter, length):
    def forbidden(**_):
        pytest.fail("source construction occurred before vector admission")

    monkeypatch.setattr(external, "_assess_organized_outer_winding", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{parameter: (_ERROR,) * length})


def test_no_cached_report_input_or_mutable_verdict(report):
    with pytest.raises(TypeError):
        _assess(report=report)
    with pytest.raises(FrozenInstanceError):
        report.admitted = False
