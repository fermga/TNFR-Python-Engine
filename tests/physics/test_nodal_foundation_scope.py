"""An EPI chart, magnitude observation and directed pressure are distinct."""

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.mathematics import BanachSpaceEPI, BEPIElement
from tnfr.physics.support_transport import observe_support_transport
from tnfr.types import ensure_bepi, real_scalar_epi, scalarize_epi


def _pressure(field):
    graph = nx.path_graph(len(field))
    graph.graph["DNFR_WEIGHTS"] = dict(epi=1.0, phase=0.0, vf=0.0, topo=0.0)
    for node, value in zip(graph, field):
        graph.nodes[node].update(EPI=value, nu_f=1.0, theta=0.0)
    default_compute_delta_nfr(graph)
    return observe_support_transport(graph)


def test_nonlinear_epi_relabeling_requires_a_pushforward_pressure():
    original = _pressure((1.0, 2.0))
    renamed = _pressure((1.0, 4.0))  # y=x^2, an invertible chart on x>0
    pushed_velocity = tuple(2 * x * v for x, v in zip(original.epi, original.rate))
    assert original.rate == (1, -1)
    assert pushed_velocity == (2, -4)
    assert renamed.rate == (3, -3)
    assert pushed_velocity != renamed.rate
    # A new metric/state representation cannot retain the old pressure rule
    # merely by retaining the text of x'=nu*p.


def test_positive_capacity_rescaling_cannot_repair_a_nonlinear_chart_pressure():
    original = _pressure((1.0, 2.0, 3.0))
    renamed = _pressure((1.0, 4.0, 9.0))  # Squaring is a smooth chart on x>0.
    pushed_rate = tuple(2 * x * rate for x, rate in zip(original.epi, original.rate))

    assert original.rate == (1, 0, -1)
    assert pushed_rate == (2, 0, -6)
    assert renamed.stored_pressure == (3, 1, -5)
    # The central rate must stay zero, but every strictly positive capacity
    # times the recomputed central pressure 1 is positive. This failure cannot
    # be absorbed into a positive capacity or common clock rescaling.
    assert pushed_rate[1] == 0 < renamed.stored_pressure[1]


def test_valid_square_chart_transforms_both_energy_gradient_and_mobility():
    s = pytest.importorskip("sympy")
    y0, y1 = s.symbols("y0 y1", positive=True)
    root = s.Matrix([s.sqrt(y0), s.sqrt(y1)])
    pulled_energy = (root[1] - root[0]) ** 2 / 2
    energy_gradient = s.Matrix([s.diff(pulled_energy, y) for y in (y0, y1)])
    jacobian = s.diag(2 * root[0], 2 * root[1])
    transformed_mobility = jacobian * jacobian.T  # Original P2 mobility is I.
    original_rate = s.Matrix([root[1] - root[0], root[0] - root[1]])
    transformed_rate = -transformed_mobility * energy_gradient

    assert (transformed_rate - jacobian * original_rate).applyfunc(
        s.simplify
    ) == s.zeros(2, 1)
    original = _pressure((1.0, 2.0))
    at_state = {y0: 1, y1: 4}
    pushed_live_rate = tuple(
        2 * x * rate for x, rate in zip(original.epi, original.rate)
    )
    assert tuple(transformed_rate.subs(at_state)) == pushed_live_rate
    assert (energy_gradient.dot(transformed_rate)).subs(
        at_state
    ) == original.energy_rate
    assert original.energy_rate < 0
    # No pressure is fitted from a desired response. The same original law
    # and energy are expressed in a different regular coordinate system.


def test_injective_singular_cube_encoding_admits_a_spurious_absorbing_branch():
    s = pytest.importorskip("sympy")
    t = s.Symbol("t", positive=True)
    original = _pressure((0.0, 1.0))
    assert original.rate == (1, -1)
    true_lift = ((1 - s.exp(-2 * t)) / 2, (1 + s.exp(-2 * t)) / 2)
    false_lift = (s.Integer(0), s.exp(-t))

    for lift in (true_lift, false_lift):
        encoded = tuple(x**3 for x in lift)
        assert tuple(y.subs(t, 0) for y in encoded) == (0, 1)
        for i in range(2):
            # These explicit roots are nonnegative for t>=0, so this is the
            # pushed row y_i'=3*|y_i|^(2/3)*(cbrt(y_j)-cbrt(y_i)).
            pushed_rhs = 3 * lift[i] ** 2 * (lift[1 - i] - lift[i])
            assert s.simplify(s.diff(encoded[i], t) - pushed_rhs) == 0

    true_residual = tuple(
        s.simplify(s.diff(true_lift[i], t) - (true_lift[1 - i] - true_lift[i]))
        for i in range(2)
    )
    false_residual = tuple(
        s.simplify(s.diff(false_lift[i], t) - (false_lift[1 - i] - false_lift[i]))
        for i in range(2)
    )
    assert true_residual == (0, 0)
    assert false_residual == (-s.exp(-t), 0)
    assert s.simplify(true_lift[0] ** 3).subs(t, s.log(2)) == s.Rational(27, 512)
    assert false_lift[0] ** 3 == 0
    # x->x^3 is injective, but its inverse is not differentiable at zero.
    # Retaining state information therefore does not alone make the two
    # differential equations equivalent; this is not an implemented solver.


@pytest.mark.parametrize("scale,offset", [(2.0, 3.0), (-2.0, 3.0), (0.5, -1.0)])
def test_pure_epi_gradient_respects_common_affine_chart_changes(scale, offset):
    original = _pressure((1.0, 2.0))
    renamed = _pressure(tuple(scale * x + offset for x in (1.0, 2.0)))
    assert renamed.rate == tuple(scale * v for v in original.rate)


def test_equal_metric_change_magnitudes_do_not_identify_directed_pressure():
    first, second = _pressure((0.0, 1.0)), _pressure((1.0, 0.0))
    assert tuple(abs(p) for p in first.stored_pressure) == (1, 1)
    assert tuple(abs(p) for p in second.stored_pressure) == (1, 1)
    assert first.rate == (1, -1)
    assert second.rate == (-1, 1)


def test_rich_epi_magnitude_is_not_an_injective_or_linear_state_chart():
    left = BEPIElement((1.0, -1.0), (0.0, 0.0), (0.0, 1.0))
    right = BEPIElement((-1.0, 1.0), (0.0, 0.0), (0.0, 1.0))
    assert real_scalar_epi(left) is real_scalar_epi(right) is None
    assert scalarize_epi(left) == scalarize_epi(right) == 1.0
    assert not np.array_equal(left.f_continuous, right.f_continuous)
    assert scalarize_epi(left + right) == 0.0
    assert scalarize_epi(left) + scalarize_epi(right) == 2.0


def test_uniform_real_bepi_storage_is_the_same_signed_scalar_chart():
    for value in (-2.0, 0.0, 2.0):
        stored = ensure_bepi(value)
        assert real_scalar_epi(stored) == scalarize_epi(stored) == value
        assert abs(stored) == abs(value)


def test_historical_paired_basis_does_not_span_the_full_direct_sum():
    space = BanachSpaceEPI()
    columns = []
    for continuous in range(2):
        for discrete in range(2):
            value = space.canonical_basis(
                continuous_size=2,
                discrete_size=2,
                continuous_index=continuous,
                discrete_index=discrete,
            )
            columns.append(
                tuple(value.f_continuous.real) + tuple(value.a_discrete.real)
            )
    # Every generated column lies in the proper hyperplane sum(f)=sum(a).
    assert all(column[0] + column[1] - column[2] - column[3] == 0 for column in columns)
    assert all(column != (1, 0, 0, 0) for column in columns)
