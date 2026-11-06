"""Independent static controls for a conditional finite bridge interface.

The witnesses are supplied states, not acquired sources. Edge sums and exact
whole-window inequalities are evaluated without a trajectory or operator event.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import product
from types import SimpleNamespace

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_preparation as owner


def _arguments(**changes):
    values = dict(
        initial_form_difference_bounds=(Q(-1, 100), Q(1, 100)),
        initial_phase_gap_bounds=(Q(1, 2), Q(3, 4)),
        bridge_weight_bounds=(Q(1, 2), Q(3, 4)),
        gamma_bounds=(Q(1, 4), Q(1, 3)),
        initial_form_upper_bound=Q(1, 2),
        weighted_degree_upper_bound=Q(3),
        elapsed_time=Q(1, 100),
        donor_node_count=5,
        reading_error_bounds=(Q(1, 10**7),) * 4,
        old_edge_chart_margins=(Q(1, 4),) * 12,
    )
    values.update(changes)
    return values


def _bound(**changes):
    return owner._prepared_bridge_exchange(**_arguments(**changes))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("finite bridge interface controls cannot run events or flows")

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
            (external, "_derive_organized_external_interaction"),
            (external, "_assess_organized_external_interaction"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 150
    return context


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


def _witness(which, context, *, relabel=False):
    """Two supplied nonuniform states with distinct internal weighted supports."""
    donor_count, host_count = 5, 6 + which
    keys = tuple(range(donor_count + host_count))
    labels = {key: f"node-{101 - 7 * key}" if relabel else key for key in keys}
    donor = tuple(labels[key] for key in range(donor_count))
    host = tuple(labels[key] for key in range(donor_count, len(keys)))
    edges = []
    phase = {}
    for cycle, origin in ((donor, Q(3, 5)), (host, Q(0))):
        edges.extend((a, b, Q(1, 2)) for a, b in zip(cycle, cycle[1:] + cycle[:1]))
        phase.update(
            (node, _mp(context, origin) + 2 * context.pi * j / len(cycle))
            for j, node in enumerate(cycle)
        )
    if which:
        edges.extend(((donor[0], donor[2], Q(1, 4)), (host[0], host[3], Q(1, 4))))
    form = {labels[key]: Q(1, 8) + Q((3 * key + which) % 7 - 3, 1000) for key in keys}
    return SimpleNamespace(
        donor=donor,
        host=host,
        nodes=donor + host,
        edges=tuple(edges),
        joined=tuple(edges) + ((host[0], donor[0], Q(2 + which, 4)),),
        form=form,
        phase=phase,
        weight=Q(2 + which, 4),
        gamma=(Q(1, 4), Q(1, 3))[which],
        cycles=(donor, host),
        node_radius=Q(1, 10000),
    )


def _rows(context, witness, edges, form, phase):
    """Direct undirected edge sums, independent of production sine fields."""
    gradient = {node: context.mpf(0) for node in witness.nodes}
    sine = {node: context.mpf(0) for node in witness.nodes}
    degree = {node: Q(0) for node in witness.nodes}
    for a, b, weight in edges:
        w = _mp(context, weight)
        difference = form[a] - form[b]
        current = w * context.sin(phase[b] - phase[a])
        gradient[a] += w * difference
        gradient[b] -= w * difference
        sine[a] += current
        sine[b] -= current
        degree[a] += weight
        degree[b] += weight
    gamma = _mp(context, witness.gamma)
    dx = {node: (-gradient[node] + gamma * sine[node]) / 4 for node in witness.nodes}
    dt = {node: gamma * gradient[node] / 4 for node in witness.nodes}
    return dx, dt, degree


@pytest.mark.parametrize("which", (0, 1))
def test_distinct_supports_and_nonzero_width_states_share_the_finite_contract(
    which, mp, record_property
):
    witness = _witness(which, mp)
    # The exchange inputs and collar floor are shared. Each collar population
    # follows its own two marked cycles: eleven edges versus twelve.
    report = _bound(old_edge_chart_margins=(Q(1, 4),) * (11 + which))
    assert len(witness.donor) == 5 and len(witness.host) == 6 + which
    assert len(witness.edges) == 11 + 3 * which
    assert report.observed_sign == "negative"
    assert report.retention_status == "certified"
    # Each primitive node has a positive-width interval. Endpoint inequalities
    # below admit the complete initial box, independently of the selected static
    # points used afterward to exercise the full edge-row implementation.
    radius = witness.node_radius
    for node in witness.nodes:
        assert abs(witness.form[node]) + radius < Q(1, 2)
    a0 = witness.form[witness.host[0]] - witness.form[witness.donor[0]]
    assert Q(-1, 100) <= a0 - 2 * radius <= a0 + 2 * radius <= Q(1, 100)
    assert Q(1, 2) <= Q(3, 5) - 2 * radius < Q(3, 5) + 2 * radius <= Q(3, 4)
    for cycle in witness.cycles:
        # The exact carrier gap is 2*pi/n. The elementary bound pi>3
        # admits every interval in the node box, without trusting mp as an
        # outward certificate or merely checking the selected representatives.
        n = len(cycle)
        collar_lower = min(Q(6, n), 3 * (Q(1, 2) - Q(2, n)))
        assert Q(1, 4) + 2 * radius < collar_lower
    tolerance = mp.mpf("1e-130")
    for form_sign, phase_sign, displacement_sign in product((-1, 1), repeat=3):
        # These are static points of the independently proved whole-window
        # box, not samples from a simulated or reconstructed trajectory.
        form = {}
        phase = {}
        for index, node in enumerate(witness.nodes):
            alternating = (-1) ** index
            form[node] = _mp(mp, witness.form[node] + form_sign * alternating * radius)
            phase[node] = witness.phase[node] + _mp(
                mp, phase_sign * alternating * radius
            )
            if displacement_sign > 0:
                form[node] += _mp(
                    mp, alternating * report.form_displacement_upper_bound
                )
                phase[node] += _mp(
                    mp, alternating * report.phase_displacement_upper_bound
                )
        assert sum(form[node] for node in witness.donor) / 5 > 0
        assert sum(form[node] for node in witness.host) / len(witness.host) > 0
        assert max(abs(value) for value in form.values()) < _mp(
            mp, report.form_upper_bound
        )
        a = form[witness.host[0]] - form[witness.donor[0]]
        d = phase[witness.donor[0]] - phase[witness.host[0]]
        _contains(mp, report.form_difference_bounds, a)
        _contains(mp, report.phase_gap_bounds, d)
        for edges in (witness.edges, witness.joined):
            dx, dt, degrees = _rows(mp, witness, edges, form, phase)
            assert max(degrees.values()) <= 3
            assert max(abs(value) for value in dx.values()) < _mp(
                mp, report.form_rate_upper_bound
            )
            assert max(abs(value) for value in dt.values()) < _mp(
                mp, report.phase_rate_upper_bound
            )
            assert abs(sum(dx.values())) < tolerance
            assert abs(sum(dt.values())) < tolerance
            donor_x = sum(dx[node] for node in witness.donor)
            host_x = sum(dx[node] for node in witness.host)
            donor_t = sum(dt[node] for node in witness.donor)
            host_t = sum(dt[node] for node in witness.host)
            assert abs(donor_x + host_x) < tolerance
            assert abs(donor_t + host_t) < tolerance
            if edges == witness.edges:
                assert abs(donor_x) < tolerance and abs(donor_t) < tolerance
                continue
            gamma, weight = _mp(mp, witness.gamma), _mp(mp, witness.weight)
            assert abs(donor_x - weight * (a - gamma * mp.sin(d)) / 4) < tolerance
            assert abs(donor_t + gamma * weight * a / 4) < tolerance
            for bounds, value in (
                (report.donor_total_form_rate_bounds, donor_x),
                (report.host_total_form_rate_bounds, host_x),
                (report.donor_mean_form_rate_bounds, donor_x / 5),
                (report.donor_total_phase_rate_bounds, donor_t),
                (report.host_total_phase_rate_bounds, host_t),
                (report.donor_mean_phase_rate_bounds, donor_t / 5),
            ):
                _contains(mp, bounds, value)
            assert (
                abs(donor_x / 5 + len(witness.host) * (host_x / len(witness.host)) / 5)
                < tolerance
            )
            # Complete rows, including every background edge, determine the
            # derivative of the bridge mean current; endpoint rows alone do not.
            derivative = (
                weight
                * (
                    dx[witness.host[0]]
                    - dx[witness.donor[0]]
                    - gamma * mp.cos(d) * (dt[witness.donor[0]] - dt[witness.host[0]])
                )
                / 20
            )
            upper = (
                Q(3, 4)
                * (
                    2 * report.form_rate_upper_bound
                    + Q(2, 3) * report.phase_rate_upper_bound
                )
                / 20
            )
            assert abs(derivative) < _mp(mp, upper)
        for cycle in witness.cycles:
            gaps = []
            for index, (a, b) in enumerate(zip(cycle, cycle[1:] + cycle[:1])):
                gap = (
                    phase[b] - phase[a] + (2 * mp.pi if index == len(cycle) - 1 else 0)
                )
                assert 0 < gap < mp.pi / 2
                gaps.append(gap)
            assert abs(sum(gaps) - 2 * mp.pi) < tolerance
    record_property("donor_count", 5)
    record_property("host_count", len(witness.host))
    record_property("internal_edges", len(witness.edges))
    record_property("marked_edge_population", 11 + which)
    record_property("observed_upper", str(report.observed_increment_bounds.hi))
    record_property("retention_minimum", str(min(report.retention_margins)))


def test_integrating_independent_complete_row_majorants_gives_finite_remainder():
    values = _arguments()
    report = owner._prepared_bridge_exchange(**values)
    k, b, g, h = Q(3), Q(1, 2), Q(1, 3), Q(1, 100)
    # Heat contraction bounds B(s)=b+k*g*s/4. Direct absolute row sums
    # give the two affine speed polynomials below. Integrate their coefficients
    # exactly; no time-stepper, fitted jet or production-derived rate is used.
    form_speed = (k * (2 * b + g) / 4, k**2 * g / 8)
    phase_speed = (g * k * b / 2, k**2 * g**2 / 8)

    def integrate(coefficients):
        return sum(
            (
                value * h ** (index + 1) / (index + 1)
                for index, value in enumerate(coefficients)
            ),
            Q(0),
        )

    x, t = integrate(form_speed), integrate(phase_speed)
    assert report.form_upper_bound == b + k * g * h / 4
    assert report.form_rate_upper_bound == form_speed[0] + form_speed[1] * h
    assert report.phase_rate_upper_bound == phase_speed[0] + phase_speed[1] * h
    assert report.form_displacement_upper_bound == x
    assert report.phase_displacement_upper_bound == t
    derivative = tuple(
        Q(3, 4) * (2 * a + 2 * g * p) / 20 for a, p in zip(form_speed, phase_speed)
    )
    integrated_remainder = sum(
        (
            value * h ** (index + 2) / ((index + 1) * (index + 2))
            for index, value in enumerate(derivative)
        ),
        Q(0),
    )
    assert report.increment_remainder_upper_bound == integrated_remainder
    assert report.retention_margins == (Q(1, 4) - 2 * t,) * 12
    assert report.true_increment_bounds.contains(
        h * report.initial_donor_mean_form_rate_bounds
        + I(-integrated_remainder, integrated_remainder)
    )


def test_relabeling_and_common_form_origin_leave_the_associated_exchange_unchanged(mp):
    left, right = _witness(1, mp), _witness(1, mp, relabel=True)
    left_form = {node: _mp(mp, value) for node, value in left.form.items()}
    right_form = {node: _mp(mp, value + Q(1, 8)) for node, value in right.form.items()}
    lx, lt, ld = _rows(mp, left, left.joined, left_form, left.phase)
    rx, rt, rd = _rows(mp, right, right.joined, right_form, right.phase)
    for a, b in zip(left.nodes, right.nodes):
        assert abs(lx[a] - rx[b]) < mp.mpf("1e-130")
        assert abs(lt[a] - rt[b]) < mp.mpf("1e-130")
        assert ld[a] == rd[b]
        assert abs(right_form[b]) + _mp(mp, right.node_radius) < mp.mpf("0.5")
    assert _bound().observed_sign == "negative"


def test_donor_normalization_changes_means_and_increments_but_not_total_flux():
    five = _bound(reading_error_bounds=(0,) * 4)
    ten = _bound(donor_node_count=10, reading_error_bounds=(0,) * 4)
    assert five.donor_total_form_rate_bounds == ten.donor_total_form_rate_bounds
    assert five.host_total_form_rate_bounds == ten.host_total_form_rate_bounds
    assert five.donor_total_phase_rate_bounds == ten.donor_total_phase_rate_bounds
    assert (
        five.increment_remainder_upper_bound == 2 * ten.increment_remainder_upper_bound
    )
    for field in (
        "donor_mean_form_rate_bounds",
        "donor_mean_phase_rate_bounds",
        "true_increment_bounds",
    ):
        original, changed = getattr(five, field), getattr(ten, field)
        assert abs(original.lo - 2 * changed.lo) < Q(1, 2**118)
        assert abs(original.hi - 2 * changed.hi) < Q(1, 2**118)


def test_four_correlated_errors_keep_the_full_contrast_allowance():
    radii = (Q(1, 10**7), Q(2, 10**7), Q(3, 10**7), Q(4, 10**7))
    report = _bound(reading_error_bounds=radii)
    errors = tuple(
        signs[0] * radii[0]
        - signs[1] * radii[1]
        - signs[2] * radii[2]
        + signs[3] * radii[3]
        for signs in product((-1, 1), repeat=4)
    )
    assert max(errors) == -min(errors) == sum(radii)
    assert report.reading_error_upper_bound == sum(radii)
    assert report.observed_increment_bounds.contains(
        report.true_increment_bounds + I(min(errors), max(errors))
    )


def _diffusion_arguments(**changes):
    values = dict(
        initial_form_difference_bounds=(1, 1),
        initial_phase_gap_bounds=(0, 0),
        bridge_weight_bounds=(1, 1),
        gamma_bounds=(0, 0),
        initial_form_upper_bound=1,
        weighted_degree_upper_bound=1,
        elapsed_time=1,
        donor_node_count=1,
        reading_error_bounds=(0,) * 4,
        old_edge_chart_margins=(Q(1, 4),),
    )
    values.update(changes)
    return values


@pytest.mark.parametrize("duration", (Q(0), Q(1, 4), Q(1), Q(2)))
def test_exact_two_node_diffusion_solution_is_inside_the_finite_enclosure(duration, mp):
    report = owner._prepared_bridge_exchange(
        **_diffusion_arguments(elapsed_time=duration, old_edge_chart_margins=None)
    )
    # Direct solution for xH(0)=1/2, xD(0)=-1/2, one unit bridge:
    # a'= -a/2 and xD(h)-xD(0)=(1-exp(-h/2))/2. No integration is run.
    exact = (1 - mp.exp(-_mp(mp, duration) / 2)) / 2
    _contains(mp, report.true_increment_bounds, exact)
    assert report.phase_displacement_upper_bound == 0
    assert report.retention_status == "not_requested"


@pytest.mark.parametrize(
    "a,duration,expected",
    ((1, 1, "positive"), (-1, 1, "negative"), (1, 2, "indeterminate")),
)
def test_strict_sign_and_exact_zero_boundary_are_independent_of_retention(
    a, duration, expected
):
    report = owner._prepared_bridge_exchange(
        **_diffusion_arguments(
            initial_form_difference_bounds=(a, a),
            elapsed_time=duration,
        )
    )
    assert report.observed_sign == expected
    assert report.retention_status == "certified"
    if duration == 2:
        assert report.observed_increment_bounds.lo == 0


def test_zero_response_and_overlapping_sign_are_not_failed_identity_claims():
    exact_zero = owner._prepared_bridge_exchange(
        **_diffusion_arguments(
            initial_form_difference_bounds=(0, 0),
            initial_form_upper_bound=0,
        )
    )
    assert exact_zero.true_increment_bounds == I(0)
    assert exact_zero.observed_sign == "indeterminate"
    assert exact_zero.retention_status == "certified"
    overlapping = _bound(initial_form_difference_bounds=(-1, 1))
    assert (
        overlapping.observed_increment_bounds.lo
        < 0
        < overlapping.observed_increment_bounds.hi
    )
    assert overlapping.observed_sign == "indeterminate"
    assert overlapping.retention_status == "certified"


def test_zero_horizon_with_positive_gamma_has_zero_response_and_initial_collars():
    report = _bound(elapsed_time=0, reading_error_bounds=(0,) * 4)
    assert report.gamma_bounds[0] > 0
    assert report.form_displacement_upper_bound == 0
    assert report.phase_displacement_upper_bound == 0
    assert report.increment_remainder_upper_bound == 0
    assert report.true_increment_bounds == report.observed_increment_bounds == I(0)
    assert report.observed_sign == "indeterminate"
    assert report.retention_margins == report.old_edge_chart_margins
    assert report.retention_status == "certified"


def test_lifted_phase_interval_is_retained_across_the_circular_cut():
    report = _bound(initial_phase_gap_bounds=(3, 4))
    assert report.initial_phase_gap_bounds == (3, 4)
    assert report.phase_gap_bounds.contains(I(3, 4))
    assert report.initial_sine_bounds.lo < 0 < report.initial_sine_bounds.hi


def test_retention_is_requested_and_strict_without_changing_signal():
    baseline = _bound()
    drift = 2 * baseline.phase_displacement_upper_bound
    cases = (
        (None, "not_requested"),
        ((0,), "indeterminate"),
        ((drift,), "indeterminate"),
        ((drift / 2,), "indeterminate"),
        ((drift + Q(1, 10**6),), "certified"),
    )
    for margins, status in cases:
        report = _bound(old_edge_chart_margins=margins)
        assert report.retention_status == status
        assert report.observed_sign == baseline.observed_sign == "negative"
        assert report.observed_increment_bounds == baseline.observed_increment_bounds
        assert (report.retention_margins is None) == (margins is None)


def test_form_envelope_intersection_is_explicit_and_disjoint_bounds_reject():
    report = _bound(initial_form_difference_bounds=(-2, Q(1, 100)))
    assert report.initial_form_difference_bounds == (-2, Q(1, 100))
    assert report.effective_initial_form_difference_bounds == (-1, Q(1, 100))
    with pytest.raises(ValueError):
        _bound(initial_form_difference_bounds=(Q(1001, 1000), 2))


def test_raw_normalization_is_immutable_fresh_and_does_not_ingest_cached_verdicts():
    radii, margins = [Q(1, 10**7)] * 4, [Q(1, 4)]
    report = _bound(reading_error_bounds=radii, old_edge_chart_margins=margins)
    radii[0], margins[0] = 100, 0
    assert report.reading_error_bounds == (Q(1, 10**7),) * 4
    assert report.old_edge_chart_margins == (Q(1, 4),)
    with pytest.raises(FrozenInstanceError):
        report.observed_sign = "positive"
    assert (
        _bound(reading_error_bounds=radii, old_edge_chart_margins=margins).observed_sign
        == "indeterminate"
    )
    for bad in (report, I(0, 1), {"lo": 0, "hi": 1, "certified": True}):
        with pytest.raises((TypeError, ValueError)):
            _bound(initial_form_difference_bounds=bad)


def test_fixed_width_fraction_components_are_normalized_before_arithmetic():
    np = pytest.importorskip("numpy")
    exact = Q(2**62, 2**62 + 1)
    narrow = Q(np.int64(2**62), np.int64(2**62 + 1))
    actual = _bound(initial_form_upper_bound=narrow)
    expected = _bound(initial_form_upper_bound=exact)
    assert actual == expected
    for value in (
        actual.initial_form_upper_bound,
        actual.form_upper_bound,
        actual.increment_remainder_upper_bound,
    ):
        assert type(value.numerator) is type(value.denominator) is int


INVALID = (
    ("initial_form_difference_bounds", (True, 1)),
    ("initial_phase_gap_bounds", (0, float("nan"))),
    ("bridge_weight_bounds", (Q(1, 2), float("inf"))),
    ("gamma_bounds", (False, 1)),
    ("initial_form_upper_bound", True),
    ("initial_form_upper_bound", float("nan")),
    ("initial_form_upper_bound", "0.5"),
    ("initial_form_upper_bound", 0.5 + 0j),
    ("weighted_degree_upper_bound", float("inf")),
    ("elapsed_time", True),
    ("elapsed_time", float("nan")),
    ("donor_node_count", True),
    ("donor_node_count", 1.0),
    ("donor_node_count", 0),
    ("donor_node_count", -1),
    ("reading_error_bounds", (0, 0, 0, True)),
    ("reading_error_bounds", (0, 0, 0, float("nan"))),
    ("old_edge_chart_margins", (Q(1, 4), False)),
    ("old_edge_chart_margins", (Q(1, 4), float("inf"))),
    ("initial_form_difference_bounds", (1, -1)),
    ("initial_phase_gap_bounds", (1, 0)),
    ("bridge_weight_bounds", (0, 1)),
    ("bridge_weight_bounds", (-1, 1)),
    ("gamma_bounds", (-1, 1)),
    ("initial_form_upper_bound", -1),
    ("weighted_degree_upper_bound", Q(1, 4)),
    ("elapsed_time", -1),
    ("reading_error_bounds", (0, 0, 0, -1)),
    ("old_edge_chart_margins", (-1,)),
    ("initial_form_difference_bounds", (0,)),
    ("initial_phase_gap_bounds", (0, 1, 2)),
    ("bridge_weight_bounds", {Q(1, 2), Q(3, 4)}),
    ("gamma_bounds", "01"),
    ("reading_error_bounds", (0, 0, 0)),
    ("reading_error_bounds", (0,) * 5),
    ("old_edge_chart_margins", ()),
    ("old_edge_chart_margins", (0,) * 4097),
)


@pytest.mark.parametrize("name,value", INVALID)
def test_invalid_primitives_reject_before_interval_or_trigonometric_work(
    name, value, monkeypatch
):
    def numerical_work(*args, **kwargs):
        pytest.fail("invalid primitive reached interval or trigonometric work")

    monkeypatch.setattr(owner, "I", numerical_work)
    monkeypatch.setattr(owner, "sin", numerical_work)
    with pytest.raises((TypeError, ValueError)):
        _bound(**{name: value})


@pytest.mark.parametrize(
    "missing",
    (
        "initial_form_upper_bound",
        "gamma_bounds",
        "donor_node_count",
        "reading_error_bounds",
    ),
)
def test_missing_required_premises_cannot_be_replaced_by_defaults(missing):
    values = _arguments()
    del values[missing]
    with pytest.raises(TypeError):
        owner._prepared_bridge_exchange(**values)
