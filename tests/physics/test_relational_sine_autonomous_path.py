"""Static controls for an autonomous reflected three-node sine-law path.

The full-field enclosures below use captured dyadic states. Crossing constants
refer separately to the ideal analytic boundary a=pi/2; no rounded phase is
declared exactly antipodal and no trajectory, event or producer is executed.
"""

from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_mediation import bound_relational_sine_mediation

MODEL = RelationalExchangeModel(1, phase_domain="regular")


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _inside(interval, value):
    assert _mp(interval.lo) <= value <= _mp(interval.hi)


def _graph(*, mean, origin, form, angle, capacity, hidden_capacity):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    for node, sign in enumerate((1, 0, -1)):
        graph.nodes[node].update(
            EPI=float(mean + sign * form),
            theta=float(origin + sign * angle),
            nu_f=float(hidden_capacity if node == 1 else capacity),
            delta_nfr=99,
        )
    return graph


@pytest.mark.parametrize(
    "mean,origin,form,angle,capacity,hidden_capacity",
    (
        (Q(0), Q(0), Q(1), Q(3, 2), Q(1), Q(3)),
        (Q(2), Q(-1, 2), Q(-3, 4), Q(2), Q(1, 2), Q(7)),
        (Q(-1), Q(1, 4), Q(0), Q(1, 2), Q(2), Q(1, 2)),
        (Q(3, 8), Q(-1), Q(1, 8), Q(-2), Q(3, 2), Q(8)),
    ),
)
def test_full_sine_owner_has_the_autonomous_reflected_quotient(
    mean, origin, form, angle, capacity, hidden_capacity
):
    graph = _graph(
        mean=mean,
        origin=origin,
        form=form,
        angle=angle,
        capacity=capacity,
        hidden_capacity=hidden_capacity,
    )
    report = bound_relational_sine_exchange(graph, reference_model=MODEL)
    assert report.form_gradient == (form, 0, -form)
    assert report.form_storage == form**2
    assert report.continuous_loss == capacity * form**2
    assert report.phase_rates[1] == I(0)
    assert report.form_rates[1].contains(0)
    # The shared trigonometric owner has its own outward precision; interval
    # cancellation encloses exact zero without claiming a zero-width report.
    assert report.form_rates[1].width < Q(1, 10**16)
    with mp.workdps(90):
        p, a, nu = map(_mp, (form, angle, capacity))
        form_rate = nu * (-p / 2 - mp.sin(a) / (2 * mp.pi))
        phase_rate = nu * p / (2 * mp.pi)
        for i, sign in enumerate((1, 0, -1)):
            _inside(report.form_rates[i], sign * form_rate)
            _inside(report.phase_rates[i], sign * phase_rate)
        _inside(report.relative_resultant[1][0], 2 * mp.cos(a))
        _inside(report.relative_resultant[1][1], mp.mpf(0))
        _inside(report.storage, p**2 + 2 * (1 - mp.cos(a)))
        _inside(report.storage_rate, -nu * p**2)

    # Conservation concerns continuous lifted phase with its declared weights.
    # It neither prevents a zero resultant nor makes a stationary reduction
    # invariant through that boundary.
    weights = (1 / capacity, 2 / hidden_capacity, 1 / capacity)
    for rates in (report.form_rates, report.phase_rates):
        rate_sum = sum((weight * rate for weight, rate in zip(weights, rates)), I(0))
        assert rate_sum.contains(0)
        assert rate_sum.width < Q(1, 10**16)
    assert sum(weight * x for weight, x in zip(weights, report.epi)) == (
        sum(weights) * mean
    )
    assert sum(weight * theta for weight, theta in zip(weights, report.phase)) == (
        sum(weights) * origin
    )


@pytest.mark.parametrize("angle,resultant_sign", ((Q(3, 2), 1), (Q(2), -1)))
def test_reflection_keeps_the_hidden_state_fixed_on_both_sides_of_cancellation(
    angle, resultant_sign
):
    # These are separate static preparations, not samples claimed to lie on
    # the named crossing orbit. Symmetry applies for every finite capacity.
    reports = [
        bound_relational_sine_exchange(
            _graph(
                mean=Q(0),
                origin=Q(0),
                form=Q(1),
                angle=angle,
                capacity=Q(1),
                hidden_capacity=mu,
            ),
            reference_model=MODEL,
        )
        for mu in (Q(1, 2), Q(16))
    ]
    for report in reports:
        real, imaginary = report.relative_resultant[1]
        assert real.lo > 0 if resultant_sign > 0 else real.hi < 0
        assert imaginary.contains(0)
        assert report.form_rates[1].contains(0)
        assert report.phase_rates[1] == I(0)
        # Derivative of the actual relative resultant; no division by R.
        real_rate = -2 * sin(I(angle)) * report.phase_rates[0]
        assert real_rate.hi < 0
        with mp.workdps(90):
            _inside(real_rate, -mp.sin(_mp(angle)) / mp.pi)
    for node in (0, 2):
        assert reports[0].form_rates[node] == reports[1].form_rates[node]
        assert reports[0].phase_rates[node] == reports[1].phase_rates[node]


def test_instantaneous_minimum_replacement_changes_the_post_cancellation_field():
    angle = Q(2)
    graph = _graph(
        mean=Q(0),
        origin=Q(0),
        form=Q(1),
        angle=angle,
        capacity=Q(1),
        hidden_capacity=Q(3),
    )
    actual = bound_relational_sine_exchange(graph, reference_model=MODEL)
    minimum = bound_relational_sine_mediation(graph, mediator=1, reference_model=MODEL)
    assert actual.relative_resultant[1][0].hi < 0
    # The actual hidden angle zero is a stationary maximum. Selecting the
    # opposite minimum is a hypothetical replacement, not the autonomous row.
    assert actual.form_rates[1].contains(0)
    assert actual.phase_rates[1] == I(0)
    assert minimum.form_rates[0].lo > actual.form_rates[0].hi
    assert actual.storage.lo > minimum.storage.hi
    with mp.workdps(90):
        a = _mp(angle)
        _inside(minimum.form_rates[0], -mp.mpf(1) / 2 + mp.sin(a) / (2 * mp.pi))
        _inside(actual.storage - minimum.storage, -4 * mp.cos(a))


def test_named_rational_preparation_has_strict_analytic_crossing_margins():
    pi = pi_interval()
    assert 3 < pi.lo < pi.hi < Q(22, 7)
    initial_angle, initial_form = Q(3, 2), Q(1)
    angle_gap = pi / 2 - initial_angle
    assert 0 < angle_gap.lo < angle_gap.hi < Q(1, 14)
    # While p>=1/2, both sine and native quotient equations give
    # dp/da>=-pi-2. Integrating to pi/2 closes the first-exit bootstrap.
    form_lower = initial_form - (pi + 2) * angle_gap
    assert form_lower.lo > Q(31, 49) > Q(1, 2)
    angular_rate_lower = I(Q(31, 49)) / (2 * pi)
    assert angular_rate_lower.lo > Q(31, 308)
    crossing_time_upper = 2 * pi * angle_gap / Q(31, 49)
    assert crossing_time_upper.hi < Q(22, 31)
    # At the ideal boundary z_h'=-p/pi; the named strict p bound makes
    # this crossing transverse. No represented graph is assigned phase pi/2.
    crossing_rate_upper = -I(Q(31, 49)) / pi
    assert crossing_rate_upper.hi < Q(-31, 154)


def test_named_preparation_is_above_cancellation_but_below_the_phase_barrier():
    initial_energy = 3 - 2 * cos(I(Q(3, 2)))
    assert 2 < initial_energy.lo < initial_energy.hi < 3 < 4
    graph = _graph(
        mean=Q(0),
        origin=Q(0),
        form=Q(1),
        angle=Q(3, 2),
        capacity=Q(1),
        hidden_capacity=Q(2),
    )
    report = bound_relational_sine_exchange(graph, reference_model=MODEL)
    with mp.workdps(90):
        expected = 3 - 2 * mp.cos(mp.mpf(3) / 2)
        _inside(initial_energy, expected)
        _inside(report.storage, expected)
    assert report.continuous_loss == 1
    # The storage sublevel is confined to |a|<2*pi/3 in the initial lift.
    # The proof's largest invariant subset of p=0 is a=0 there. These static
    # controls check its constants, not a numerically observed return time.


@pytest.mark.parametrize(
    "m,p,b,a,nu,mu,beta,e,w",
    (
        (Q(1, 2), Q(3, 4), Q(1, 4), Q(3, 2), Q(1), Q(3), Q(2), Q(1, 4), Q(3, 4)),
        (Q(-1, 4), Q(1, 2), Q(-1, 2), Q(2), Q(1, 2), Q(2), Q(3, 2), Q(1, 2), Q(1, 2)),
        (Q(1, 8), Q(-1, 4), Q(7, 4), Q(-3, 4), Q(2), Q(1, 2), Q(1), Q(3, 4), Q(1, 4)),
    ),
)
def test_four_coordinate_quotient_retains_the_moving_hidden_reference(
    m, p, b, a, nu, mu, beta, e, w
):
    # m and b are mean leaf contrasts relative to the hidden node, not common
    # offsets. The hidden state is zero only at this capture; its nonzero rows
    # must be subtracted when projecting the subsequent contrast rates.
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1, "weight")
    for node, x, theta, capacity in zip(
        range(3), (m + p, Q(0), m - p), (b + a, Q(0), b - a), (nu, mu, nu)
    ):
        graph.nodes[node].update(
            EPI=float(x), theta=float(theta), nu_f=float(capacity), delta_nfr=99
        )
    model = RelationalExchangeModel(
        float(beta), epi_weight=float(e), phase_weight=float(w), phase_domain="regular"
    )
    report = bound_relational_sine_exchange(graph, reference_model=model)
    fx, ft = report.form_rates, report.phase_rates
    projected_rates = (
        (fx[0] + fx[2]) / 2 - fx[1],
        (fx[0] - fx[2]) / 2,
        (ft[0] + ft[2]) / 2 - ft[1],
        (ft[0] - ft[2]) / 2,
    )
    assert report.form_gradient == (m + p, -2 * m, m - p)
    assert report.form_storage == m**2 + p**2
    assert report.continuous_loss == 2 * e * (nu * p**2 + (nu + mu) * m**2)
    assert not ft[1].contains(0)
    with mp.workdps(90):
        mm, pp, bb, aa, nn, hh, scale, ee, ww = map(
            _mp, (m, p, b, a, nu, mu, beta, e, w)
        )
        expected_rates = (
            -(nn + hh) * (ee * mm + ww * mp.sin(bb) * mp.cos(aa) / mp.pi),
            -nn * (ee * pp + ww * mp.cos(bb) * mp.sin(aa) / mp.pi),
            (nn + hh) * ww * mm / (scale * mp.pi),
            nn * ww * pp / (scale * mp.pi),
        )
        for rate, expected in zip(projected_rates, expected_rates):
            _inside(rate, expected)
        _inside(
            report.storage,
            mm**2 + pp**2 + 2 * scale * (1 - mp.cos(aa) * mp.cos(bb)),
        )
        _inside(report.storage_rate, -2 * ee * (nn * pp**2 + (nn + hh) * mm**2))
