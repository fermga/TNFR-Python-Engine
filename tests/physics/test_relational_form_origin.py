"""Static controls for weighted form balance and the common-origin quotient.

Frozen preparations: K1,3 has phases (0,0,0,t), t=atan(3/4), capacities
(1,2,3,4), and zero form. The connected seven-node remote control has edges
(0,1),(0,2),(0,3),(3,4),(4,5),(4,6), phases (0,0,t,0,0,0,u), zero form,
and unit capacities; only u or node 6's capacity changes as specified below.
The clock, unit support and beta=1 are held. No trajectory or new runtime law
is installed. Symbolic ideal identities and represented native-reader checks
are separate: a binary64 angle is not an exact atan preparation certificate.
"""

import math
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

ANGLE = math.atan2(3, 4)
STAR_CAPACITIES = (1, 2, 3, 4)
REMOTE_EDGES = ((0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6))
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_weight=1, phase_domain="regular")


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("form-origin controls must remain static")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


def _state(graph, phases, capacities, forms=None):
    forms = (0,) * len(graph) if forms is None else forms
    assert len(phases) == len(capacities) == len(forms) == len(graph)
    for node, phase, capacity, form in zip(graph, phases, capacities, forms):
        graph.nodes[node].update(theta=phase, nu_f=capacity, EPI=form)
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _star(*, forms=None):
    return _state(nx.star_graph(3), (0, 0, 0, ANGLE), STAR_CAPACITIES, forms=forms)


def _remote(*, phase=0, capacity=1):
    graph = nx.Graph()
    graph.add_nodes_from(range(7))
    graph.add_edges_from(REMOTE_EDGES)
    return _state(graph, (0, 0, ANGLE, 0, 0, 0, phase), (1, 1, 1, 1, 1, 1, capacity))


def _project_represented_field(graph, field):
    """Apply only the proposed algebraic projection to captured native rates."""
    weights = tuple(
        Q(graph.degree[node]) / Q(nu) for node, nu in zip(field.nodes, field.capacity)
    )
    rates = tuple(map(Q, field.form_rate))
    total_rate = sum((mu * rate for mu, rate in zip(weights, rates)), Q(0))
    common_rate = total_rate / sum(weights)
    return weights, total_rate, common_rate, tuple(rate - common_rate for rate in rates)


def test_exact_acute_star_has_nonzero_native_weighted_form_source():
    s = pytest.importorskip("sympy")
    t, a = s.atan(s.Rational(3, 4)), s.atan(s.Rational(3, 14))
    assert s.sin(t) == s.Rational(3, 5)
    assert s.cos(t) == s.Rational(4, 5)
    assert s.sin(t) / (2 + s.cos(t)) == s.Rational(3, 14)
    # 0<a<3/14 gives 0<3a<9/14<pi/2. Tan is strictly increasing
    # there, so the exact rational comparison proves 3a<t and Qdot<0.
    tangent_three_a = s.expand_trig(s.tan(3 * a))
    assert s.simplify(tangent_three_a) == s.Rational(1737, 2366)
    assert s.Rational(1737, 2366) < s.Rational(3, 4)
    assert Q(9, 14) < Q(3, 2)  # pi>3 supplies the stated branch bound.


def test_projection_preserves_relative_rates_and_dirichlet_work_not_absolute_rate():
    s = pytest.importorskip("sympy")
    one = s.ones(4, 1)
    mu = s.Matrix([3, s.Rational(1, 2), s.Rational(1, 3), s.Rational(1, 4)])
    projection = s.eye(4) - one * mu.T / sum(mu)
    laplacian = s.Matrix([[3, -1, -1, -1], [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    differences = s.Matrix([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    assert projection * projection == projection
    assert mu.T * projection == s.zeros(1, 4)
    assert differences * projection == differences
    assert laplacian * projection == laplacian
    assert projection != s.eye(4)
    # Unchanged phase velocity and q^T*1=0 preserve the full storage rate.
    # Trajectory quotient equivalence additionally requires common-form
    # shift invariance of both original rows, tested below for the native law.


def test_shared_native_field_has_common_form_symmetry_and_unchanged_projected_work():
    model = RelationalExchangeModel(1, phase_domain="regular")
    graph = _star(forms=(0, 1, -1, 2))
    source = relational.evaluate_relational_exchange(graph, model=model)
    shifted = graph.copy()
    for node in shifted:
        shifted.nodes[node]["EPI"] += 8
    translated = relational.evaluate_relational_exchange(shifted, model=model)
    assert translated.form_rate == source.form_rate
    assert translated.phase_rate == source.phase_rate
    assert translated.storage == source.storage
    weights, total_rate, common_rate, projected = _project_represented_field(
        graph, source
    )
    assert total_rate < 0 and common_rate != 0
    assert sum(mu * rate for mu, rate in zip(weights, projected)) == 0
    q = source.work.form_gradient
    assert sum(q) == 0
    work_change = sum(
        gradient * (rate - Q(original))
        for gradient, rate, original in zip(q, projected, source.form_rate)
    )
    assert work_change == 0
    assert source.storage_rate + work_change == source.storage_rate
    for i in range(1, 4):
        assert projected[i] - projected[0] == Q(source.form_rate[i]) - Q(
            source.form_rate[0]
        )
    assert projected[0] != Q(source.form_rate[0])


def test_capacity_independent_pressure_has_no_common_lift_freedom_across_interventions():
    s = pytest.importorskip("sympy")
    differences = s.Matrix([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
    baseline = s.diag(1, 2, 3, 4)
    intervention = s.diag(2, 2, 3, 4)
    before = differences * baseline
    assert before.rank() == 3
    assert before.nullspace() == [s.Matrix((4, 2, s.Rational(4, 3), 1))]
    # A capacity-independent pressure difference must be in both kernels.
    # Varying one own capacity removes the sole common-rate freedom.
    joint = before.col_join(differences * intervention)
    assert joint.rank() == 4
    assert joint.nullspace() == []


def test_projected_native_quotient_is_not_the_sine_quotient():
    graph = _star()
    native = relational.evaluate_relational_exchange(graph, model=MODEL)
    sine = bound_relational_sine_exchange(graph, reference_model=MODEL)
    weights, total_rate, common, projected = _project_represented_field(graph, native)
    ideal_total = (3 * math.atan2(3, 14) - ANGLE) / math.pi
    assert float(total_rate) == pytest.approx(ideal_total, abs=3e-16)
    assert total_rate < -Q(1, 1000)
    # No finite common clock multiplier can turn a zero rate into this one.
    assert native.form_rate[1] == 0
    assert projected[1] == -common != 0
    assert sum(
        (mu * rate for mu, rate in zip(weights, sine.form_rates)), I(0)
    ).contains(0)
    assert native.phase_rate == (0,) * 4
    assert sine.phase_rates == (I(0),) * 4
    assert native.storage_rate == native.continuous_loss == sine.continuous_loss == 0
    assert sine.storage_rate == I(0)
    native_relative = projected[3] - projected[1]
    assert native_relative == Q(native.form_rate[3]) - Q(native.form_rate[1])
    assert float(native_relative) == pytest.approx(-4 * ANGLE / math.pi, abs=3e-16)
    sine_relative = sine.form_rates[3] - sine.form_rates[1]
    # The ideal counterpart is -4*(t-3/5)/pi<0; here the sine interval
    # encloses the field at the captured binary64 angle, not exact atan(3/4).
    assert (native_relative - sine_relative).hi < -Q(1, 25)


@pytest.mark.parametrize("change", ["remote_phase", "remote_capacity"])
def test_global_correction_changes_a_root_with_unchanged_primitive_neighborhood(change):
    before = _remote()
    after = _remote(phase=ANGLE) if change == "remote_phase" else _remote(capacity=2)
    root_ball = {0, *before.neighbors(0)}
    assert root_ball == {0, 1, 2, 3}
    assert set(before.edges) == set(after.edges)
    assert before.subgraph(root_ball).nodes(data=True) == after.subgraph(
        root_ball
    ).nodes(data=True)
    native = relational.evaluate_relational_exchange(before, model=MODEL)
    modified = relational.evaluate_relational_exchange(after, model=MODEL)
    weights, total, common, projected = _project_represented_field(before, native)
    other_weights, other_total, other_common, other_projected = (
        _project_represented_field(after, modified)
    )
    assert native.phase_source[0] == modified.phase_source[0]
    assert native.form_rate[0] == modified.form_rate[0]
    assert native.phase_rate[0] == modified.phase_rate[0]
    assert sum(weights) == 12
    if change == "remote_phase":
        assert other_total == 2 * total
        assert sum(other_weights) == 12
        assert other_common == 2 * common
    else:
        assert other_total == total
        assert sum(other_weights) == Q(23, 2)
        assert other_common / common == Q(24, 23)
    assert projected[0] != other_projected[0]


def test_single_zero_capacity_has_finite_pressure_limit_and_vanishing_rate_correction():
    s = pytest.importorskip("sympy")
    epsilon = s.Symbol("epsilon", positive=True)
    source = s.Symbol("Qdot", nonzero=True, real=True)
    # K1,3 with capacities (1,epsilon,1,1): Qdot does not depend on
    # capacities in the native model, while W=5+1/epsilon.
    common = source / (5 + 1 / epsilon)
    assert s.limit(common, epsilon, 0, dir="+") == 0
    assert s.limit(-common / epsilon, epsilon, 0, dir="+") == -source
    assert s.simplify(common / (source * epsilon)) == 1 / (5 * epsilon + 1)
    # Thus |pressure correction|<=|Qdot|/d_1 with d_1=1, not divergence.


def test_two_zero_capacities_have_path_dependent_pressure_but_same_velocity_limit():
    s = pytest.importorskip("sympy")
    epsilon, ratio = s.symbols("epsilon ratio", positive=True)
    source = s.Symbol("Qdot", nonzero=True, real=True)
    # Capacities (1,epsilon,ratio*epsilon,1) give the same limiting
    # state for every ratio, but a different pressure at the first zero leaf.
    common = source / (4 + 1 / epsilon + 1 / (ratio * epsilon))
    assert s.limit(common, epsilon, 0, dir="+") == 0
    limit = s.limit(-common / epsilon, epsilon, 0, dir="+")
    assert s.simplify(limit + source * ratio / (ratio + 1)) == 0
    assert limit.subs(ratio, 1) == -source / 2
    assert limit.subs(ratio, 2) == -2 * source / 3
    assert limit.subs(ratio, 1) != limit.subs(ratio, 2)
    # The full corrected velocity extends to the native frozen-capacity
    # field; no unique continuous pressure extension or inverse-capacity Q
    # is thereby defined on this corner.
