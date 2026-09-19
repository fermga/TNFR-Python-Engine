"""Conditional passive exchange and a finite alignment-cost reservoir.

The signed storage-residual inequality defines a class, not a runtime law. Beta
is a declared comparison coefficient, not a derived physical constant.
Fixed support, fixed positive EPI coefficient, unit capacity and a regular
positive phase metric are essential. No integration, event or phase-law
selection is performed, and finite norm integral alone is not convergence.
The residual is not identified with physical power or an external mechanism.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    P,
    _exact_generator,
    _graph,
    _nonrepeated_phase_geometry,
    _symbolic_phase_response,
)
from tnfr.physics.support_transport import observe_support_transport


def test_full_centered_storage_has_one_signed_supply_without_a_selected_phase_row():
    s = pytest.importorskip("sympy")
    lap = -s.Matrix(_exact_generator(observe_support_transport(_graph())))
    project = s.eye(6) - s.ones(6) / 6
    e, w, beta = s.symbols("e w beta", positive=True)
    mean = s.symbols("mean", real=True)
    y = project * s.Matrix(s.symbols("y0:6", real=True))
    source = s.Matrix(s.symbols("g0:6", real=True))
    phase_velocity = s.Matrix(s.symbols("omega0:6", real=True))
    metric = s.diag(*s.symbols("h0:6", positive=True))
    x = y + mean * s.ones(6, 1)
    pressure = -e * lap * x + w * source
    centered_rate = project * pressure
    # In the regular canonical chart, grad(V)=-H*g. No H_dot term belongs
    # here: the storage is ||y||^2/2+beta*V, not a phase metric norm.
    phase_potential_gradient = -metric * source
    storage_rate = y.dot(centered_rate) + beta * phase_potential_gradient.dot(
        phase_velocity
    )
    supply = w * y.dot(source) - beta * source.dot(metric * phase_velocity)
    assert s.expand(storage_rate + e * y.dot(lap * y) - supply) == 0
    assert s.expand(sum(pressure) / 6 - w * sum(source) / 6) == 0
    assert s.diff(supply, mean) == 0
    # If supply<=allowed_residual_rate, E_dot<=-e*y^T L y+that bound.
    # This includes phase-to-form and form-to-phase transfer, without choosing
    # either sign of the canonical source work or discarding the moving mean.
    # "Supply" here means only this declared storage's signed residual.
    gap = s.Rational(2, 3)
    gap_excess = lap - gap * project
    assert gap_excess == gap_excess.T
    assert gap_excess.eigenvals() == {
        s.Rational(0): 2,
        s.Rational(1, 3): 2,
        s.Rational(1): 2,
    }


def test_retained_nonreversible_response_can_exchange_work_passively_in_both_directions():
    s, graph, rows, phases = _nonrepeated_phase_geometry()
    response, squared = _symbolic_phase_response(s, phases, rows)
    reflected, _ = _symbolic_phase_response(s, tuple(-value for value in phases), rows)
    assert reflected == response
    cycle = (
        response[0, 1]
        * response[1, 2]
        * response[2, 0]
        / (response[0, 2] * response[2, 1] * response[1, 0])
    )
    assert s.simplify(cycle) == s.Rational(4, 5)
    a, gamma = s.pi / 6, s.atan(1 / (3 * s.sqrt(3)))
    source = s.Matrix([a, -a, gamma, gamma + a, gamma - a, -a]) / s.pi
    actual_source = s.Matrix(
        [
            (
                s.atan(
                    sum(s.sin(phases[j]) for j in row)
                    / sum(s.cos(phases[j]) for j in row)
                )
                - phases[i]
            )
            / s.pi
            for i, row in enumerate(rows)
        ]
    )
    assert (actual_source - source).applyfunc(s.simplify) == s.zeros(6, 1)
    gradient = s.Matrix(
        [sum(s.sin(phases[i] - phases[j]) for j in row) for i, row in enumerate(rows)]
    )
    potential = sum(
        1 - s.cos(phases[i] - phases[j])
        for i, row in enumerate(rows)
        for j in row
        if i < j
    )
    assert s.simplify(potential) == (9 - 3 * s.sqrt(3)) / 2
    assert potential > 0  # Finite initial alignment reservoir, even under reflection.
    delta = s.pi * source
    radii = tuple(s.sqrtdenest(s.sqrt(value)) for value in squared)
    metric = s.diag(*[s.pi * radii[i] * s.sin(delta[i]) / delta[i] for i in range(6)])
    assert (metric * source + gradient).applyfunc(s.simplify) == s.zeros(6, 1)
    assert s.simplify(sum(gradient)) == 0
    # 0<gamma<a ensures every |delta_i|<2a=pi/3, so r*sinc(delta)>0.
    assert 0 < 1 / (3 * s.sqrt(3)) < s.tan(a)
    assert max(phases) - min(phases) == s.pi / 3 < s.pi / 2
    assert all(value > 0 for value in squared)
    reflected_metric = s.diag(
        *[s.pi * radii[i] * s.sin(-delta[i]) / (-delta[i]) for i in range(6)]
    )
    assert (reflected_metric - metric).applyfunc(s.simplify) == s.zeros(6)
    lap = -s.Matrix(_exact_generator(observe_support_transport(graph)))
    y = s.Matrix(P * 2) / 4  # Unchanged retained form; no gain/amplitude search.
    assert lap * y == y and y.dot(y) == s.Rational(1, 4)
    e, w = s.Rational(1, 2), s.Rational(1, 4)
    beta = s.symbols("beta", positive=True)
    extra_loss = s.symbols("extra_loss", nonnegative=True)
    common_speed = s.symbols("common_speed", real=True)
    mean_rates = []
    for sign, expected_form_rate in ((1, -s.Rational(1, 12)), (-1, -s.Rational(1, 6))):
        g = sign * source
        source_work = w * y.dot(g)
        assert s.simplify(source_work) == sign * s.Rational(1, 24)
        form_rate = -e * y.dot(lap * y) + source_work
        assert s.simplify(form_rate) == expected_form_rate
        # Algebraic members of the passive inequality class, not selected
        # engine laws. Common rotation does no phase-potential work.
        omega = (
            (w / beta) * metric.inv() * y + extra_loss * g + common_speed * s.ones(6, 1)
        )
        phase_work = -beta * g.dot(metric * omega)
        expected_total = -s.Rational(1, 8) - beta * extra_loss * g.dot(metric * g)
        assert s.simplify(form_rate + phase_work - expected_total) == 0
        assert s.simplify(phase_work.subs(extra_loss, 0) + source_work) == 0
        mean_rates.append(w * sum(g) / 6)
    assert s.simplify(sum(mean_rates)) == 0
    # Reflection keeps H and R, reverses the transfer, and preserves total
    # passive loss. State-dependent skew exchange alone is not a Poisson or
    # symplectic certificate; no Jacobi identity is claimed.


def test_integrated_budget_bounds_lifetime_with_a_finite_allowed_residual():
    s = pytest.importorskip("sympy")
    e, rho = s.symbols("e rho", positive=True)
    gap = s.Rational(2, 3)
    initial, supply, endpoint, gap_integral, passive_slack = s.symbols(
        "initial allowed_residual endpoint gap_integral passive_slack", nonnegative=True
    )
    # An allowed upper bound on the cumulative signed residual gives
    # E(T)+e*gap*I+e*gap_integral+passive_slack=E(0)+supply, I=int||y||^2.
    # No literal external energy source is inferred from this comparison.
    norm_integral = (initial + supply - endpoint - e * gap_integral - passive_slack) / (
        e * gap
    )
    bound = (initial + supply) / (e * gap)
    remainder = s.simplify(bound - norm_integral)
    assert remainder == (endpoint + e * gap_integral + passive_slack) / (e * gap)
    assert remainder.is_nonnegative
    lifetime_bound = (initial + supply) / (e * gap * rho**2)
    assert s.simplify(lifetime_bound - bound / rho**2) == 0
    # If ||y||>=rho throughout [0,T], then rho^2*T<=I and T cannot exceed
    # this finite bound. Finite I alone does not prove pointwise convergence;
    # uniform form and arbitrary common-phase gauge remain outside that claim.

    # A real exact pure-EPI gap mode saturates the zero-supply balance with
    # the endpoint retained. This is an analytic owner-compatible identity,
    # not a solver run or a proposed phase feedback law.
    lap = -s.Matrix(_exact_generator(observe_support_transport(_graph())))
    direction = s.Matrix([1, 1, 1, -1, -1, -1]) / 4
    assert lap * direction == gap * direction
    time = s.symbols("time", nonnegative=True)
    energy0 = direction.dot(direction) / 2
    energy_end = energy0 * s.exp(-2 * e * gap * time)
    integrated_norm = energy0 * (1 - s.exp(-2 * e * gap * time)) / (e * gap)
    assert s.simplify(energy_end + e * gap * integrated_norm - energy0) == 0
    assert (
        s.simplify(
            s.diff(integrated_norm, time)
            - direction.dot(direction) * s.exp(-2 * e * gap * time)
        )
        == 0
    )
    assert s.limit(integrated_norm, time, s.oo) == energy0 / (e * gap)
