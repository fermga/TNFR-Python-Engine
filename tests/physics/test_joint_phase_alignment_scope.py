"""Reciprocity and conditional relaxation under the supplied law theta'=g.

The phase metric represents a vector field; it does not install a phase law
or select the undetermined joint potential. The final budget requires a
global regular path and a uniform positive metric floor. No trajectory,
pressure refresh, fitted gain or production evolution rule is constructed.
"""

from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import LIFT, NODES, _graph
from tests.physics._internal_mode_fixture import (
    _prepared_phase_metric as _prepared_metric,
)
from tests.physics._internal_mode_fixture import _unit_regional_budget as _budget
from tnfr.physics.support_transport import observe_support_transport


def test_epi_independent_psi_cannot_remove_the_prepared_reciprocal_velocity():
    s, _, _, metric, source, jacobian = _prepared_metric()
    direction = s.Matrix((1, -1, 0) * 2)
    first = s.ones(6, 1) / 2 + direction / 16
    second = first + direction / 16
    weight = s.Rational(1, 4)
    degree = 3 * s.eye(6)
    # At identical phase/capacity/support, every EPI-independent Psi has the
    # same phase gradient at both states; no particular Psi is selected.
    psi_gradient = s.Matrix(s.symbols("psi_theta0:6", real=True))
    required = tuple(
        metric.inv() * (weight * jacobian.T * degree * epi - psi_gradient)
        for epi in (first, second)
    )
    difference = s.simplify(required[1] - required[0])
    expected = -direction / (64 * s.pi * (1 + s.sqrt(3)))
    assert s.simplify(difference - expected) == s.zeros(6, 1)
    assert difference != s.zeros(6, 1)
    residuals = tuple(metric * (source - value) for value in required)
    assert s.simplify(residuals[1] - residuals[0]) == 3 * direction / (64 * s.pi)
    # The proposed x-independent theta'=g is identical at these two states.
    # If Psi cancels its residual at the first, it cannot do so at the second.


def test_phase_only_descent_does_not_erase_the_joint_reciprocal_work():
    s, graph, phases, metric, source, jacobian = _prepared_metric()
    variables = s.symbols("theta0:6", real=True)
    positions = {node: index for index, node in enumerate(NODES)}
    potential = sum(
        1 - s.cos(variables[positions[left]] - variables[positions[right]])
        for left, right in graph.edges
    )
    substitution = dict(zip(variables, phases, strict=True))
    gradient = s.Matrix(
        [s.diff(potential, value).subs(substitution) for value in variables]
    )
    assert s.simplify(gradient + metric * source) == s.zeros(6, 1)
    assert s.simplify(potential.subs(substitution)) == 5 - 2 * s.sqrt(3)
    phase_work = s.simplify((gradient.T * source)[0])
    assert phase_work == -(1 + s.sqrt(3)) / 3 < 0
    # Audit the illustrative choice Psi=V_phi without promoting it to a law.
    epi = s.ones(6, 1) / 2 + s.Matrix((1, -1, 0) * 2) / 16
    joint_gradient = gradient - s.Rational(1, 4) * jacobian.T * (3 * epi)
    joint_phase_work = s.simplify((joint_gradient.T * source)[0])
    assert s.simplify(joint_phase_work - phase_work) == 1 / (32 * s.pi) > 0
    # This is a nonzero missing cross term, not a claim that the complete
    # joint energy increases or that a reciprocal gradient law was executed.


def test_regional_owner_keeps_the_exact_young_slack_and_disagreement_loss():
    e, weight = Q(1, 2), Q(1, 4)
    phase_source = (Q(1, 6), Q(-1, 6), Q(0)) * 2
    source_squared = sum(value**2 for value in phase_source)
    rates = []
    for sign in (1, -1):
        snapshot = observe_support_transport(_graph((Q(1, 16), 0, sign * Q(1, 16), 0)))
        internal = tuple(value - Q(1, 2) for value in snapshot.epi)
        difference = tuple(
            a - b for a, b in zip(internal[:3], internal[3:], strict=True)
        )
        disagreement = sum(value**2 for value in difference)
        forcing = tuple(weight * value for value in phase_source)
        squared, rate, work = _budget(snapshot, forcing, e=e)
        assert squared == sum(value**2 for value in internal) == Q(1, 64)
        slack = e * sum(
            (y - weight * g / e) ** 2
            for y, g in zip(internal, phase_source, strict=True)
        )
        assert slack == e * squared + weight**2 * source_squared / e - work
        assert (
            rate + e * squared - weight**2 * source_squared / e
            == (-slack - 2 * e * disagreement / 3)
            < 0
        )
        rates.append(rate)
    assert rates == [Q(1, 192), Q(-5, 192)]
    # These are exact declared source vectors at snapshots. Positive initial
    # source work is compatible with the later conditional decay estimate.


def test_combined_budget_has_nonnegative_slack_under_a_uniform_phase_metric_floor():
    s = pytest.importorskip("sympy")
    e, floor = s.symbols("e h_min", positive=True)
    weight, metric_excess = s.symbols("w metric_excess", nonnegative=True)
    coefficients = s.Matrix(s.symbols("u0 v0 u1 v1", real=True))
    internal = s.Matrix(LIFT) * coefficients
    source = s.Matrix(s.symbols("g0:6", real=True))
    squared = (internal.T * internal)[0]
    difference = internal[:3, :] - internal[3:, :]
    disagreement = (difference.T * difference)[0]
    source_squared = (source.T * source)[0]
    work = 2 * weight * (internal.T * source)[0]
    rate = -2 * e * squared - 2 * e * disagreement / 3 + work
    young_slack = (
        e * ((internal - weight * source / e).T * (internal - weight * source / e))[0]
    )
    # IF theta'=g and H_i>=h_min along the entire regular path, V' has
    # this form with metric_excess=sum_i(H_i-h_min)*g_i^2 >=0.
    phase_energy_rate = -floor * source_squared - metric_excess
    combined_rate = rate + weight**2 * phase_energy_rate / (e * floor)
    remainder = combined_rate + e * squared
    expected = (
        -young_slack
        - 2 * e * disagreement / 3
        - weight**2 * metric_excess / (e * floor)
    )
    assert s.expand(remainder - expected) == 0
    assert all(
        value.is_nonnegative
        for value in (
            e,
            floor,
            weight,
            metric_excess,
            2 * e / 3,
            weight**2 / (e * floor),
        )
    )
    # Z=S+w^2*V/(e*h_min)>=0 and Z'<=-e*S give integral S<=Z(0)/e.
    # V'<=-h_min*||g||^2 also gives an L1 forcing input. Convergence of S
    # additionally uses S'<=-e*S+(w^2/e)||g||^2 and the stable scalar
    # convolution estimate; finite integral S alone would not suffice.
