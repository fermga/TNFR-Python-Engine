"""Static mixed-neighbor algebra and conditional event/source controls."""

from decimal import Decimal
from fractions import Fraction as Q

import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_class_neighbor_nonadditivity as owner


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


def _geometry():
    rings = tuple(
        tuple((9 * r + j, 9 * r + (j + 1) % 9) for j in range(9)) for r in range(3)
    )
    edges = sum(rings, ()) + ((4, 13), (13, 22))
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))
    laplacian = [[Q(0)] * 27 for _ in range(27)]
    for i, j in edges:
        laplacian[i][i] += 1
        laplacian[j][j] += 1
        laplacian[i][j] -= 1
        laplacian[j][i] -= 1
    normalized = tuple(
        tuple(value / degrees[i] for value in row) for i, row in enumerate(laplacian)
    )
    return rings, edges, degrees, tuple(map(tuple, laplacian)), normalized


def _mv(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Q(0))


def _arguments():
    return dict(
        mediator_class=2,
        donor_amplitude=Q(1, 4000),
        receiver_amplitude=Q(-1, 5000),
        horizon=Q(1, 7),
        endpoint_radius=Q(1, 10**32),
        readout_error_bound=Q(1, 10**30),
        radius=Q(1, 12),
        contact_work_allowance=Q(1, 10**12),
        donor_work_allowance=Q(1, 10**6),
        receiver_work_allowance=Q(1, 10**6),
    )


@pytest.fixture(scope="module")
def report():
    return owner._bound_neighbor_nonadditivity(**_arguments())


@pytest.mark.parametrize("mediator_class", (1, 2))
def test_exact_onset_from_independent_edge_cubic_polynomial(mediator_class):
    result = owner._derive_neighbor_onset(mediator_class)
    _, edges, degrees, _, normalized = _geometry()
    donor = tuple(row[4] for row in normalized)
    receiver = tuple(row[22] for row in normalized)
    assert donor[4] == receiver[22] == 1
    assert donor[13] == receiver[13] == Q(-1, 4)
    # Expand (a*U+b*V)^3 by its monomial coefficients at the four actual
    # mediator edges. Dividing by24 includes sine's1/6 and time integration1/4.
    channels = [[Q(0), Q(0)] for _ in range(4)]
    for edge_index, (i, j) in enumerate(edges):
        if 13 not in (i, j):
            continue
        other = j if i == 13 else i
        u, v = donor[other] - donor[13], receiver[other] - receiver[13]
        channels[min(edge_index // 9, 3)][0] -= 3 * u**2 * v / (24 * degrees[13])
        channels[min(edge_index // 9, 3)][1] -= 3 * u * v**2 / (24 * degrees[13])
    assert result.mixed_channel_factors == tuple(map(tuple, channels))
    assert result.mixed_channel_factors == (
        (Q(0), Q(0)),
        (Q(-1, 1024), Q(-1, 1024)),
        (Q(0), Q(0)),
        (Q(-15, 1024), Q(-15, 1024)),
    )


def test_mixed_quadratic_hidden_state_reenters_central_cubic_feedback():
    _, edges, degrees, _, normalized = _geometry()
    donor, receiver = tuple(row[4] for row in normalized), tuple(
        row[22] for row in normalized
    )
    sine = (Q(2, 3),) * 9 + (Q(4, 5),) * 9 + (Q(2, 3),) * 9 + (Q(0),) * 2

    def quadratic(left, right):
        out = [Q(0)] * 27
        for (i, j), s in zip(edges, sine):
            force = -s * (left[j] - left[i]) * (right[j] - right[i]) / 2
            out[i] += force / degrees[i]
            out[j] -= force / degrees[j]
        return tuple(out)

    mixed = tuple(2 * value for value in quadratic(donor, receiver))
    assert any(mixed)
    assert all(mixed[i] == 0 for i in (4, 13, 22))
    reflection = tuple(9 * (i // 9) + 8 - i % 9 for i in range(27))
    assert tuple(mixed[i] for i in reflection) == tuple(-value for value in mixed)
    hidden_phase = _mv(normalized, mixed)
    recoupling = quadratic(tuple(a + b for a, b in zip(donor, receiver)), hidden_phase)
    assert recoupling[13] != 0
    assert _dot(degrees, mixed) == _dot(degrees, recoupling) == 0


def test_exact_formal_factor_retains_subgrid_sign_without_finite_sign_claim(report):
    a, b, h = report.donor_amplitude, report.receiver_amplitude, report.horizon
    cosine = report.onset.channel_cosine_bounds[1]
    products = tuple(
        -(g**4) * (15 + c) * a * b * (a + b) * h**4 / 1024
        for g in (report.onset.gamma_bounds.lo, report.onset.gamma_bounds.hi)
        for c in (cosine.lo, cosine.hi)
    )
    assert report.formal_onset_bounds == (min(products), max(products))
    tiny = owner._bound_neighbor_nonadditivity(
        **(_arguments() | {"receiver_amplitude": Q(1, 5000), "horizon": Q(1, 10**30)})
    )
    assert (
        -Q(1, 2**128) < tiny.formal_onset_bounds[0] <= tiny.formal_onset_bounds[1] < 0
    )
    assert report.decision is None and not report.response_bound_available


def test_additive_functional_null_is_exact_without_linear_single_responses():
    hidden = Q(13, 7)
    baseline = hidden**2

    def donor(a):
        return baseline + a + hidden * a**2 + 3 * a**3

    def receiver(b):
        return baseline - b + 2 * hidden * b**3

    def additive(a, b):
        return donor(a) + receiver(b) - baseline

    a, b = Q(1, 5), Q(-2, 7)
    assert additive(a, b) - additive(a, 0) - additive(0, b) + additive(0, 0) == 0
    # This does not equate an additive sum with a shared nonlinear hidden update.
    coupled = additive(a, b) + hidden * a * b
    assert coupled - donor(a) - receiver(b) + baseline != 0


def test_source_accounting_cancels_only_shared_linear_initialization(report):
    fidelity = report.per_history_fidelity
    assert tuple(row.total_input_variation for row in fidelity) == (
        0,
        abs(report.donor_amplitude),
        abs(report.receiver_amplitude),
        abs(report.donor_amplitude) + abs(report.receiver_amplitude),
    )
    assert report.nonlinear_source_error_upper_bound == sum(
        row.nonlinear_initialization_error_upper_bound for row in fidelity
    )
    assert fidelity[0].nonlinear_initialization_error_upper_bound > 0
    assert report.higher_amplitude_error_upper_bound == sum(
        row.nominal_fifth_order_error_upper_bound for row in fidelity
    )
    assert report.nonlinear_source_error_upper_bound < 4 * report.endpoint_radius


def test_distinct_commuting_events_have_no_donor_receiver_work_cross_term(report):
    _, _, degrees, laplacian, _ = _geometry()
    a, b, eps = (
        report.donor_amplitude,
        report.receiver_amplitude,
        report.endpoint_radius,
    )
    before = tuple(eps / 2 * (int(i in (3, 21)) - int(i in (2, 20))) for i in range(27))
    after_donor = tuple(value + (a if i == 4 else 0) for i, value in enumerate(before))
    after = tuple(value + (b if i == 22 else 0) for i, value in enumerate(after_donor))
    assert laplacian[4][22] == laplacian[22][4] == 0
    assert _mv(laplacian, after_donor)[22] == _mv(laplacian, before)[22]
    work = (
        _dot(after, _mv(laplacian, after)) - _dot(before, _mv(laplacian, before))
    ) / 2
    expected = (
        a * _mv(laplacian, before)[4]
        + b * _mv(laplacian, before)[22]
        + Q(3, 2) * (a**2 + b**2)
    )
    assert work == expected
    assert (_dot(degrees, after) - _dot(degrees, before)) / 58 == Q(3, 58) * (a + b)
    assert report.pre_receiver_laplacian_absolute_bounds == (6 * eps,) * 4
    both = report.event_ledger.histories[-1]
    assert both.first_probe_work_upper_bound == Q(3, 2) * a**2 + 6 * abs(a) * eps
    assert both.second_probe_work_upper_bound == Q(3, 2) * b**2 + 6 * abs(b) * eps
    assert (
        work <= both.first_probe_work_upper_bound + both.second_probe_work_upper_bound
    )
    assert both.final_form_mean_shift == Q(3, 58) * (a + b)
    works = []
    for donor, receiver in ((0, 0), (a, 0), (0, b), (a, b)):
        state = tuple(
            value + (donor if i == 4 else receiver if i == 22 else 0)
            for i, value in enumerate(before)
        )
        works.append(
            (_dot(state, _mv(laplacian, state)) - _dot(before, _mv(laplacian, before)))
            / 2
        )
    assert works[3] - works[1] - works[2] + works[0] == 0


def test_closed_work_and_strict_storage_boundaries_remain_independent(report):
    first, second = (
        report.event_ledger.histories[-1].first_probe_work_upper_bound,
        report.event_ledger.histories[-1].second_probe_work_upper_bound,
    )
    equal = owner._bound_neighbor_nonadditivity(
        **(
            _arguments()
            | {"donor_work_allowance": first, "receiver_work_allowance": second}
        )
    )
    assert equal.event_ledger.histories[-1].work_within_allowances
    a = Q(1, 18000)
    boundary = owner._bound_neighbor_nonadditivity(
        **(
            _arguments()
            | {
                "donor_amplitude": a,
                "receiver_amplitude": a,
                "endpoint_radius": 0,
                "radius": 90 * a,
            }
        )
    )
    both = boundary.event_ledger.histories[-1]
    assert both.after_second_storage_margin == 0
    assert not both.identity_certified
    assert both.work_within_allowances


@pytest.mark.parametrize(
    "change", ({"donor_amplitude": 0}, {"receiver_amplitude": 0}, {"horizon": 0})
)
def test_exact_zero_histories_keep_four_reading_error_bound(change):
    result = owner._bound_neighbor_nonadditivity(**(_arguments() | change))
    assert result.exact_mixed_zero and result.response_bound_available
    assert result.decision.true_bounds == (0, 0)
    assert result.decision.recorded_bounds == (
        -4 * result.readout_error_bound,
        4 * result.readout_error_bound,
    )
    assert not result.decision.true_sign and not result.decision.null_excluded


def test_opposite_amplitudes_do_not_assert_actual_source_parity():
    a = Q(1, 5000)
    result = owner._bound_neighbor_nonadditivity(
        **(_arguments() | {"donor_amplitude": a, "receiver_amplitude": -a})
    )
    assert result.formal_onset_bounds == (0, 0)
    assert not result.exact_mixed_zero and result.decision is None
    assert result.nonlinear_source_error_upper_bound > 0


@pytest.fixture(scope="module", params=(1, 2))
def finite(request):
    # A static analytic witness, not a trajectory or a time-series evaluation.
    return owner._bound_neighbor_nonadditivity(
        **(
            _arguments()
            | {
                "mediator_class": request.param,
                "donor_amplitude": Q(7, 10000),
                "receiver_amplitude": Q(7, 10000),
                "horizon": Q(1, 8),
            }
        )
    )


def _multiply(left, right):
    products = tuple(a * b for a in left for b in right)
    return min(products), max(products)


def test_static_cone_rebuilt_from_independent_rational_spatial_columns(finite):
    _, edges, degrees, _, a = _geometry()
    donor, receiver = tuple(row[4] for row in a), tuple(row[22] for row in a)
    donor_second, receiver_second = _mv(a, donor), _mv(a, receiver)
    assert finite.onset.donor_phase_velocity_column == donor
    assert finite.onset.receiver_phase_velocity_column == receiver
    assert finite.onset.donor_phase_curvature_column == donor_second
    assert finite.onset.receiver_phase_curvature_column == receiver_second
    h, g = Q(1, 8), Q(1, 3000)
    d = 1 - 2 * g**2 * h**2
    error = Q(4, 3) * h**2 * (1 + g**2 / d)
    cone = finite.static_cone
    assert cone is not None and cone.tangent_phase_remainder_bound == error
    force = [(Q(0), Q(0)) for _ in range(27)]
    for index, (i, j) in enumerate(edges):
        intervals = []
        for velocity, curvature in ((donor, donor_second), (receiver, receiver_second)):
            slope = velocity[j] - velocity[i]
            end = slope - h * (curvature[j] - curvature[i]) / 2
            intervals.append((min(slope, end) - error, max(slope, end) + error))
        u, v = intervals
        for expected, stored in zip(
            intervals,
            (cone.donor_edge_phase_cones[index], cone.receiver_edge_phase_cones[index]),
        ):
            assert stored.lo <= expected[0] <= expected[1] <= stored.hi
        c = (Q(1, 6), Q(1)) if index < 27 else (Q(1), Q(1))
        current = _multiply(_multiply(_multiply(c, u), v), (u[0] + v[0], u[1] + v[1]))
        current = -current[1] / 2, -current[0] / 2
        force[i] = (
            force[i][0] + current[0] / degrees[i],
            force[i][1] + current[1] / degrees[i],
        )
        force[j] = (
            force[j][0] - current[1] / degrees[j],
            force[j][1] - current[0] / degrees[j],
        )
    for exact, stored in zip(force, cone.normalized_mixed_force_bounds):
        assert stored.lo <= exact[0] <= exact[1] <= stored.hi
        assert exact[0] - stored.lo < Q(1, 10**32)
        assert stored.hi - exact[1] < Q(1, 10**32)
    assert cone.mediator_force_upper_bound < Q(-1, 15)
    assert cone.competing_force_upper_bound < Q(1, 9)
    assert max(row.abs_max for row in cone.normalized_mixed_force_bounds) < Q(3, 20)
    # The three nonzero initial histories all have nodal max m and ||Af||<=m;
    # the joint history still has total input variation2m for Cauchy/source.
    for f in (
        tuple(Q(int(i == 4)) for i in range(27)),
        tuple(Q(int(i == 22)) for i in range(27)),
        tuple(Q(int(i in (4, 22))) for i in range(27)),
    ):
        assert max(map(abs, f)) == 1 and max(map(abs, _mv(a, f))) <= 1


def test_finite_full_bound_keeps_feedback_fifth_order_source_and_reading_budgets(
    finite,
):
    cone = finite.static_cone
    m, h, g = finite.donor_amplitude, finite.horizon, Q(1, 3000)
    d = 1 - 2 * g**2 * h**2
    extra = 3 * m**3 * (Q(4, 15) * g**6 * h**6 / d**4 + Q(1, 63) * g**8 * h**8 / d**5)
    assert finite.complete_cubic_correction_upper_bound == extra > 0
    assert cone.complete_cubic_correction_upper_bound == extra
    low, upper = cone.normalized_heat_shape_bounds
    assert low == min(row.lo for row in cone.normalized_mixed_force_bounds) / 4
    assert (
        upper
        == cone.mediator_force_upper_bound / 4
        + (cone.competing_force_upper_bound - cone.mediator_force_upper_bound) * h / 20
    )
    assert upper < Q(-7, 450)
    products = tuple(
        gamma**4 * m**3 * h**4 * endpoint
        for gamma in (finite.onset.gamma_bounds.lo, finite.onset.gamma_bounds.hi)
        for endpoint in (low, upper)
    )
    assert cone.tangent_cubic_bounds == (min(products), max(products))
    error = (
        extra
        + finite.higher_amplitude_error_upper_bound
        + finite.nonlinear_source_error_upper_bound
    )
    assert finite.decision.true_bounds == (min(products) - error, max(products) + error)
    assert Q(-39, 10**30) < finite.decision.true_bounds[0]
    assert finite.decision.true_bounds[1] < Q(-10983, 10**33)
    assert finite.readout_error_bound == Q(1, 10**30)
    assert finite.decision.recorded_bounds == (
        finite.decision.true_bounds[0] - 4 * finite.readout_error_bound,
        finite.decision.true_bounds[1] + 4 * finite.readout_error_bound,
    )
    assert finite.decision.null_separation_margin > Q(298, 10**32)
    assert finite.decision.null_excluded
    assert finite.status == "zero_contrast_record_sets_disjoint"
    assert all(
        row.identity_certified and row.work_within_allowances
        for row in finite.event_ledger.histories
    )
    assert finite.conditional_protocol_sufficient


@pytest.mark.parametrize(
    "change", ({"donor_work_allowance": 0}, {"radius": Q(1, 10000)})
)
def test_observation_verdict_does_not_replace_work_or_identity(finite, change):
    result = owner._bound_neighbor_nonadditivity(
        **(
            _arguments()
            | {
                "mediator_class": finite.mediator_class,
                "donor_amplitude": finite.donor_amplitude,
                "receiver_amplitude": finite.receiver_amplitude,
                "horizon": finite.horizon,
            }
            | change
        )
    )
    assert result.decision == finite.decision
    assert result.status == "zero_contrast_record_sets_disjoint"
    assert not result.conditional_protocol_sufficient
    if "donor_work_allowance" in change:
        assert not result.all_work_within_allowances
        assert result.all_identities_certified
    else:
        assert result.all_work_within_allowances
        assert not result.all_identities_certified


@pytest.mark.parametrize("reading_count", (4, 8))
@pytest.mark.parametrize("side", (-1, 0, 1))
def test_strict_recorded_and_independent_additive_noise_boundaries(
    finite, reading_count, side
):
    threshold = -finite.decision.true_bounds[1] / reading_count
    delta = threshold - side * Q(1, 2**600)
    result = owner._bound_neighbor_nonadditivity(
        **(
            _arguments()
            | {
                "mediator_class": finite.mediator_class,
                "donor_amplitude": finite.donor_amplitude,
                "receiver_amplitude": finite.receiver_amplitude,
                "horizon": finite.horizon,
                "readout_error_bound": delta,
            }
        )
    )
    if reading_count == 4:
        assert result.decision.recorded_sign is (side > 0)
        assert result.decision.recorded_sign_margin == reading_count * side * Q(
            1, 2**600
        )
    else:
        assert result.decision.null_excluded is (side > 0)
        assert result.decision.null_separation_margin == reading_count * side * Q(
            1, 2**600
        )


@pytest.mark.parametrize(
    "change",
    (
        {"horizon": Q(1, 8) + Q(1, 2**500)},
        {"donor_amplitude": Q(1, 4000), "receiver_amplitude": Q(1, 5000)},
        {"donor_amplitude": Q(-1, 5000), "receiver_amplitude": Q(-1, 5000)},
    ),
)
def test_unproved_finite_domains_retain_onset_and_work_without_passing_flags(change):
    result = owner._bound_neighbor_nonadditivity(
        **(
            _arguments()
            | {
                "donor_amplitude": Q(1, 5000),
                "receiver_amplitude": Q(1, 5000),
                "horizon": Q(1, 8),
            }
            | change
        )
    )
    assert result.static_cone is None
    assert result.complete_cubic_correction_upper_bound is None
    assert result.decision is None and not result.response_bound_available
    assert result.status == "finite_remainder_unavailable"
    assert len(result.event_ledger.histories) == 4


@pytest.mark.parametrize("key", tuple(_arguments()))
@pytest.mark.parametrize(
    "bad", (True, np.bool_(False), float("nan"), Decimal("1e-400"))
)
def test_admission_precedes_structural_onset(key, bad, monkeypatch):
    monkeypatch.setattr(
        owner, "_derive_neighbor_onset", lambda *_: pytest.fail("late admission")
    )
    with pytest.raises((ValueError, TypeError)):
        owner._bound_neighbor_nonadditivity(**(_arguments() | {key: bad}))


@pytest.mark.parametrize(
    "change",
    (
        {"horizon": 3},
        {"radius": 0},
        {"donor_amplitude": Q(7, 5000)},
        {"endpoint_radius": -1},
        {"readout_error_bound": -1},
        {"contact_work_allowance": -1},
        {"mediator_class": 3},
    ),
)
def test_unsupported_primitive_domains_reject_before_geometry(change, monkeypatch):
    monkeypatch.setattr(
        owner, "_derive_neighbor_onset", lambda *_: pytest.fail("late admission")
    )
    with pytest.raises(ValueError):
        owner._bound_neighbor_nonadditivity(**(_arguments() | change))
