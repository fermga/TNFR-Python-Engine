"""Finite observation criteria distinguish shape selection from amplitude loss.

These controls reuse the existing exact P3 diffusion and its two-rate law.
They introduce no trajectory, pressure feedback, persistence mechanism or
physical acceptance constants. Modal energy fractions are mathematical shape
read-outs, not a complete NFR identity or probability interpretation.
"""

from dataclasses import replace
from fractions import Fraction as Q

import pytest

from tests.physics.test_forced_support_shape import _observe, _reference, _source

s = pytest.importorskip("sympy")


def test_existing_p3_curve_has_an_exact_shape_survival_frontier():
    source = _source()
    reference = _reference(source)
    initial = _observe(reference)
    metric = s.diag(*reference.metric_weights)
    slow, fast = s.Matrix((1, 0, -1)), s.Matrix((1, -1, 1))
    assert (slow.T * metric * fast)[0] == 0
    assert s.Matrix(initial.state.relative_error) == (slow + fast) / 4
    z = s.Symbol("z", positive=True)  # Existing curve coordinate exp(-t).
    y = (z * slow + z**2 * fast) / 4
    squared = (y.T * metric * y)[0]
    slow_energy = (slow.T * metric * y)[0] ** 2 / (slow.T * metric * slow)[0]
    fraction = s.factor(slow_energy / squared)
    survival = s.factor(squared / initial.norm_squared)
    assert fraction == 1 / (1 + 2 * z**2)
    assert s.expand(survival) == (z**2 + 2 * z**4) / 3
    assert s.simplify(survival - (1 - fraction) / (6 * fraction**2)) == 0
    assert fraction.subs(z, 1) == Q(1, 3) and survival.subs(z, 1) == 1
    assert s.limit(fraction, z, 0, dir="+") == 1
    assert s.limit(survival, z, 0, dir="+") == 0
    assert (-z * s.diff(fraction, z)).is_positive
    assert (-z * s.diff(survival, z)).is_negative
    # Bind one algebraic point to the same production observer, preserving
    # the original mean. This is not a new graph preparation or integration.
    point = tuple(Q(value) for value in y.subs(z, Q(1, 2)))
    snapshot = replace(source, epi=tuple(initial.state.mean + value for value in point))
    observed = _observe(reference, snapshot)
    assert observed.state.mean == initial.state.mean
    assert observed.norm_squared / initial.norm_squared == Q(1, 8)
    assert fraction.subs(z, Q(1, 2)) == Q(2, 3)
    assert observed.norm_squared_rate < 0 and observed.stationary_shape is False


def test_predeclared_shape_and_survival_requirements_may_have_no_common_window():
    p = s.Symbol("p", positive=True)
    eta = s.Symbol("eta", positive=True)
    frontier = (1 - p) / (6 * p**2)
    # On 1/3<=p<1, the frontier decreases strictly from one to zero.
    assert s.factor(s.diff(frontier, p)) == (p - 2) / (6 * p**3)
    maximum_fraction = 2 / (1 + s.sqrt(1 + 24 * eta))
    assert s.simplify(frontier.subs(p, maximum_fraction) - eta) == 0
    assert maximum_fraction.subs(eta, 1) == Q(1, 3)
    assert s.limit(maximum_fraction, eta, 0, dir="+") == 1
    # Thresholds below are exact observation countercontrols, not coefficients
    # of the nodal law. At p*=2/3 the existing curve has z=1/2, t=log(2).
    required_fraction = Q(2, 3)
    available_survival = frontier.subs(p, required_fraction)
    assert available_survival == Q(1, 8)
    assert available_survival > Q(1, 16)  # A nonzero-duration window exists.
    assert available_survival == Q(1, 8)  # Only its common endpoint survives.
    assert available_survival < Q(1, 4)  # No common admissible time exists.
    assert maximum_fraction.subs(eta, Q(1, 16)) > required_fraction
    assert maximum_fraction.subs(eta, Q(1, 8)) == required_fraction
    assert maximum_fraction.subs(eta, Q(1, 4)) < required_fraction
    # Exact slow-mode shape is approached only as the original norm vanishes;
    # renormalizing a read-out does not demonstrate retained physical form.
    assert frontier.subs(p, 1) == 0


def test_two_rate_window_depends_on_inherited_rates_and_declared_observation():
    z, ratio, alpha = s.symbols("z ratio alpha", positive=True)
    p = s.Symbol("p", positive=True)
    # z=exp[-2*(r2-r1)*t], alpha=r1/(r2-r1), and ratio=E2(0)/E1(0).
    # Both modes are occupied and 0<r1<r2; no modal rate is chosen here.
    fraction = 1 / (1 + ratio * z)
    survival = z**alpha * (1 + ratio * z) / (1 + ratio)
    frontier = ((1 - p) / (ratio * p)) ** alpha / ((1 + ratio) * p)
    assert s.simplify((1 - fraction) / (ratio * fraction)) == z
    assert s.simplify(frontier.subs(p, fraction) - survival) == 0
    assert fraction.subs(z, 1) == 1 / (1 + ratio)
    assert survival.subs(z, 1) == 1
    # Positive log derivative in z proves survival falls in forward time.
    logarithmic_slope = alpha / z + ratio / (1 + ratio * z)
    assert s.simplify(s.diff(survival, z) / survival - logarithmic_slope) == 0
    assert logarithmic_slope.is_positive
    assert s.diff(fraction, z).is_negative
    # Thus for predeclared p* in [p(0),1), eta in (0,1], there is a time
    # satisfying both criteria iff eta<=frontier(p*). Strict inequality
    # gives positive duration. This is a finite observational statement,
    # not autonomous NFR formation, recovery or indefinite maintenance.
    assert s.simplify(frontier.subs({ratio: 2, alpha: 1}) - (1 - p) / (6 * p**2)) == 0
    gap = s.Symbol("gap", positive=True)
    target = s.Symbol("target", positive=True)
    formation_time = s.log(ratio * target / (1 - target)) / (2 * gap)
    target_z = (1 - target) / (ratio * target)
    assert s.simplify(fraction.subs(z, target_z) - target) == 0
    assert s.simplify(formation_time.subs(target, 1 / (1 + ratio))) == 0
