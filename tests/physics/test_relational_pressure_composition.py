"""Static controls for primitive pressure composition and complete-law scope.

Ideal sine intervals, independent high-precision calculations and the native
binary64 field remain distinct evidence; none of these controls runs a flow.
"""

import math
from decimal import Decimal, localcontext
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_reset,
)
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.support_transport import observe_support_transport

MODEL = RelationalExchangeModel(1, phase_domain="regular")
PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628"
    "03482534211706798214808651"
)


def _state(graph, *, phases, forms=None, capacities=None):
    count = len(graph)
    forms = (0,) * count if forms is None else forms
    capacities = (1,) * count if capacities is None else capacities
    assert len(phases) == len(forms) == len(capacities) == count
    for node, phase, form, capacity in zip(graph, phases, forms, capacities):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _read(graph, model=MODEL):
    return (
        relational.evaluate_relational_exchange(graph, model=model),
        bound_relational_sine_exchange(graph, reference_model=model),
    )


def _decimal(value):
    value = Q(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _trig(value, *, sine=False):
    """Independent Decimal series on the test's |angle| <= 1/2 domain."""
    value = _decimal(value)
    term = total = value if sine else Decimal(1)
    for k in range(1, 120):
        divisor = 2 * k * (2 * k + 1) if sine else (2 * k - 1) * 2 * k
        term *= -value * value / divisor
        total += term
    return total


def _atan(value):
    """Independent alternating series; every test argument has |value| < .6."""
    assert abs(value) < Decimal("0.6")
    power = total = value
    for k in range(1, 260):
        power *= -value * value
        total += power / (2 * k + 1)
    return total


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("pressure-composition controls must remain static")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


def test_primitive_merge_is_stronger_than_repeating_a_neighborhood():
    # A=(0,0), B=(1/2): all stars and singleton gaps are strictly acute.
    reports = tuple(
        _read(_state(nx.star_graph(len(gaps)), phases=(0, *gaps)))
        for gaps in ((0, 0), (0.5,), (0, 0, 0.5))
    )
    native_a, sine_a = reports[0]
    native_b, sine_b = reports[1]
    native_union, sine_union = reports[2]
    sine_residual = (
        3 * sine_union.phase_sources[0]
        - 2 * sine_a.phase_sources[0]
        - sine_b.phase_sources[0]
    )
    assert sine_residual.contains(0)
    assert sine_residual.width < Q(1, 10**18)
    native_residual = (
        3 * native_union.phase_source[0]
        - 2 * native_a.phase_source[0]
        - native_b.phase_source[0]
    )
    independent = (3 * math.atan2(math.sin(0.5), 2 + math.cos(0.5)) - 0.5) / math.pi
    assert native_residual == pytest.approx(independent, abs=2e-16)
    assert native_residual < -Q(1, 1000)


@pytest.mark.parametrize("copies", [2, 3])
def test_complete_synchronized_replication_preserves_both_complete_fields(copies):
    base = _state(
        nx.path_graph(3),
        phases=(0, 0.25, -0.25),
        forms=(0.5, -0.25, 1),
        capacities=(1, 2, 3),
    )
    replica = nx.Graph()
    replica.graph.update(base.graph)
    for node, data in base.nodes(data=True):
        for index in range(copies):
            replica.add_node((node, index), **data)
    for left, right in base.edges():
        for i in range(copies):
            for j in range(copies):
                replica.add_edge((left, i), (right, j))
    base_native, base_sine = _read(base)
    replica_native, replica_sine = _read(replica)
    for position, (node, _) in enumerate(replica_native.nodes):
        base_position = base_native.nodes.index(node)
        assert (
            replica_sine.degrees[position] == copies * base_sine.degrees[base_position]
        )
        assert (
            replica_sine.form_gradient[position]
            == copies * base_sine.form_gradient[base_position]
        )
        for name in ("phase_source", "form_rate", "phase_rate"):
            assert getattr(replica_native, name)[position] == pytest.approx(
                getattr(base_native, name)[base_position], rel=2e-14, abs=2e-15
            )
        for name in ("phase_sources", "form_rates", "phase_rates"):
            difference = (
                getattr(replica_sine, name)[position]
                - getattr(base_sine, name)[base_position]
            )
            assert difference.contains(0)
            assert difference.width < Q(1, 10**18)


@pytest.mark.parametrize(
    "phases,forms,capacities,epsilon",
    [
        ((0, 0.25, -0.25), (1, -0.5, 0.25), (1, 2, 3), Q(1, 4)),
        ((0, 0.5, -0.5), (1, 0, 0), (1, 1, 1), Q(1, 2)),
        (
            (0.125, 0.625, -0.125, 0.25, -0.25),
            (1, 0, -1, 0.5, -0.25),
            (2, 0, 1, 3, 0.5),
            Q(1, 2),
        ),
    ],
)
def test_acute_geometry_bounds_independent_ideal_rates_and_native_implementation(
    phases, forms, capacities, epsilon
):
    graph = _state(
        nx.star_graph(len(phases) - 1),
        phases=phases,
        forms=forms,
        capacities=capacities,
    )
    native, sine = _read(graph)
    with localcontext() as context:
        context.prec = 100
        lower = _trig(epsilon)
        eps = _decimal(epsilon)
        slack = Decimal("1e-90")  # Decimal series/rounding, not a certificate.
        source_difference_seen = phase_difference_seen = False
        for i, node in enumerate(sine.nodes):
            gaps = tuple(sine.phase[j] - sine.phase[i] for j in graph[node])
            assert all(abs(gap) <= epsilon for gap in gaps)
            real = sum((_trig(gap) for gap in gaps), Decimal(0))
            imag = sum((_trig(gap, sine=True) for gap in gaps), Decimal(0))
            angle = _atan(imag / real)
            degree = sine.degrees[i]
            attenuation = imag / (degree * angle) if angle else real / degree
            assert lower - slack <= attenuation <= 1 + slack
            native_source = angle / PI
            sine_source = imag / (PI * degree)
            difference = abs(native_source - sine_source)
            assert difference <= (1 - lower) * eps / PI + slack
            assert sine.phase_sources[i].contains(Q(sine_source))
            assert native.phase_source[i] == pytest.approx(
                float(native_source), abs=2e-15
            )
            nu, q = map(_decimal, (sine.capacity[i], sine.form_gradient[i]))
            ideal_sine_rate = nu * q / (2 * PI * degree)
            ideal_native_rate = ideal_sine_rate / attenuation
            phase_difference = abs(ideal_native_rate - ideal_sine_rate)
            assert phase_difference <= (1 / lower - 1) * abs(ideal_sine_rate) + slack
            assert sine.phase_rates[i].contains(Q(ideal_sine_rate))
            assert native.phase_rate[i] == pytest.approx(
                float(ideal_native_rate), rel=2e-14, abs=2e-15
            )
            # The source difference is also the full form-rate difference
            # after multiplying by w*nu; the diffusion rows are identical.
            observed_form_difference = native.form_rate[i] - float(
                sine.form_rates[i].midpoint
            )
            assert observed_form_difference == pytest.approx(
                float(nu * (native_source - sine_source) / 2), abs=2e-15
            )
            source_difference_seen |= difference > Decimal("1e-8")
            phase_difference_seen |= phase_difference > Decimal("1e-8")
        assert source_difference_seen and phase_difference_seen


@pytest.mark.parametrize("epi_weight", [0, 0.5])
def test_p2_diffusion_obstructs_a_common_clock_but_exchange_only_is_equivalent(
    epi_weight,
):
    model = RelationalExchangeModel(
        2, epi_weight=epi_weight, phase_weight=1 - epi_weight, phase_domain="regular"
    )
    graph = _state(nx.path_graph(2), phases=(0, 0.5), forms=(1, 0), capacities=(1, 2))
    native, sine = _read(graph, model)
    e, w = map(Q, model.effective_weights)
    beta = Q(model.storage_scale)
    c = sum(sine.capacity)
    r = sine.epi[0] - sine.epi[1]
    delta = sine.phase[1] - sine.phase[0]
    # Exact ideal determinant formula, independently using certified sin(delta).
    sine_delta = sine.relative_resultant[0][1]
    ideal_determinant = (
        c**2 * e * w * r**2 / (beta * pi_interval()) * (1 - delta / sine_delta)
    )
    native_relative_form = native.form_rate[0] - native.form_rate[1]
    native_relative_phase = native.phase_rate[1] - native.phase_rate[0]
    sine_relative_form = sine.form_rates[0] - sine.form_rates[1]
    sine_relative_phase = sine.phase_rates[1] - sine.phase_rates[0]
    # Native floats are not promoted to bounds on the ideal native field.
    represented_determinant = Q(
        native_relative_form
    ) * sine_relative_phase - sine_relative_form * Q(native_relative_phase)
    assert float(represented_determinant.midpoint) == pytest.approx(
        float(ideal_determinant.midpoint), abs=3e-15
    )
    if epi_weight:
        assert ideal_determinant.hi < -Q(1, 1000)
        assert represented_determinant.hi < -Q(1, 1000)
    else:
        assert ideal_determinant == I(0)
        attenuation = math.sin(float(delta)) / float(delta)
        for native_name, sine_name in (
            ("form_rate", "form_rates"),
            ("phase_rate", "phase_rates"),
        ):
            for native_rate, sine_rate in zip(
                getattr(native, native_name), getattr(sine, sine_name)
            ):
                assert float(sine_rate.midpoint) == pytest.approx(
                    attenuation * native_rate, rel=2e-14, abs=2e-15
                )


def test_positive_diffusion_selects_degree_over_capacity_form_weights():
    s = pytest.importorskip("sympy")
    graph = nx.star_graph(3)
    capacities = (1, 2, 3, 4)
    degrees = tuple(graph.degree[node] for node in graph)
    laplacian = s.Matrix(
        [
            [degrees[i] if i == j else -int(graph.has_edge(i, j)) for j in graph]
            for i in graph
        ]
    )
    generator = (
        -s.diag(
            *(s.Rational(nu, 2 * degree) for nu, degree in zip(capacities, degrees))
        )
        * laplacian
    )
    weights = s.Matrix([s.Rational(d, nu) for d, nu in zip(degrees, capacities)])
    nullspace = generator.T.nullspace()
    assert len(nullspace) == 1
    assert nullspace[0] / nullspace[0][0] == weights / weights[0]

    # Independent basis preparations verify the shared full form row at
    # phase consensus, where the pressure source is zero and e=1/2.
    for column in graph:
        native, sine = _read(
            _state(
                graph.copy(),
                phases=(0,) * 4,
                forms=tuple(int(i == column) for i in graph),
                capacities=capacities,
            )
        )
        for row, rate in enumerate(sine.form_rates):
            expected = Q(generator[row, column])
            assert rate.contains(expected)
            assert native.form_rate[row] == pytest.approx(float(expected), abs=2e-16)
        assert sum(
            (Q(weight) * rate for weight, rate in zip(weights, sine.form_rates)),
            I(0),
        ).contains(0)
    # Uniform arithmetic weights do not conserve the same connected flow.
    assert sum(generator[:, 0]) == 4


def test_frozen_acute_star_separates_form_balance_from_zero_storage_loss():
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    graph = _state(
        nx.star_graph(3),
        phases=(0, 0, 0, Q(1, 2)),
        capacities=(1, 2, 3, 4),
    )
    native, sine = _read(graph, model)
    alternative = sine.with_current_squared_mobility(epsilon=1)
    relative = alternative.relative_balance(reference_node=0)
    transfer = sine.regional_transfer(region=sine.nodes)
    weights = tuple(Q(d) / nu for d, nu in zip(sine.degrees, sine.capacity))
    total_weight = sum(weights)
    assert sine.nodes == (0, 1, 2, 3)
    assert weights == (Q(3), Q(1, 2), Q(1, 3), Q(1, 4))
    assert total_weight == Q(49, 12)
    assert transfer.form_weights == weights
    assert transfer.regional_form_rate_bounds == I(0)
    assert transfer.global_weighted_form_conserved
    assert transfer.global_weighted_form_rate_bounds.contains(0)
    assert transfer.global_weighted_form_rate_bounds.width < Q(1, 10**18)
    assert (
        native.continuous_loss
        == sine.continuous_loss
        == alternative.continuous_loss
        == 0
    )
    assert native.phase_rate == (0,) * 4
    assert sine.phase_rates == alternative.phase_rates == (I(0),) * 4
    assert native.balance_residual == 0
    assert sine.storage_rate == alternative.storage_rate == I(0)

    native_total_rate = sum(
        weight * Q(rate) for weight, rate in zip(weights, native.form_rate)
    )
    alternative_total_rate = total_weight * relative.weighted_form_mean_rate_bounds
    direct_alternative_rate = sum(
        (weight * rate for weight, rate in zip(weights, alternative.form_rates)), I(0)
    )
    assert relative.weighted_form_mean_rate_residual_bounds.contains(0)
    assert native_total_rate < -Q(1, 1000)
    assert alternative_total_rate.hi < -Q(1, 100)
    assert direct_alternative_rate.hi < -Q(1, 100)
    with localcontext() as context:
        context.prec = 100
        sine_gap = _trig(Q(1, 2), sine=True)
        cosine_gap = _trig(Q(1, 2))
        native_truth = (3 * _atan(sine_gap / (2 + cosine_gap)) - Decimal("0.5")) / PI
        alternative_truth = -8 * sine_gap**3 / (9 * PI)
        assert float(native_total_rate) == pytest.approx(float(native_truth), abs=2e-16)
        assert alternative_total_rate.contains(Q(alternative_truth))
        assert direct_alternative_rate.contains(Q(alternative_truth))
        assert relative.weighted_form_mean_rate_bounds.contains(
            Q(alternative_truth / _decimal(total_weight))
        )


@pytest.mark.parametrize("gap", [Q(1, 4), Q(1, 2)])
def test_p2_balance_does_not_select_constant_reciprocal_mobility(gap):
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    source = bound_relational_sine_exchange(
        _state(
            nx.path_graph(2),
            phases=(0, gap),
            forms=(Q(1, 2), -Q(1, 4)),
            capacities=(2, 3),
        ),
        reference_model=model,
    )
    alternative = source.with_current_squared_mobility(epsilon=1)
    relative = alternative.relative_balance(reference_node=0)
    for rate in (
        relative.weighted_form_mean_rate_bounds,
        relative.direct_weighted_form_mean_rate_bounds,
    ):
        assert rate.contains(0)
        assert rate.width < Q(1, 10**18)
    assert alternative.balance_residual.contains(0)
    assert alternative.continuous_loss == 0
    assert (alternative.form_rates[0] - source.form_rates[0]).lo > 0
    assert (alternative.phase_rates[0] - source.phase_rates[0]).lo > 0
    assert not relative.constant_mobility_mean_and_volume_identities_certified


def test_odd_cubic_sine_kernel_conserves_form_without_selecting_sine():
    s = pytest.importorskip("sympy")
    delta = s.Symbol("delta", real=True)
    epsilon = s.Symbol("epsilon", positive=True)
    kernel = (s.sin(delta) + epsilon * s.sin(delta) ** 3) / s.pi
    assert s.simplify(kernel + kernel.subs(delta, -delta)) == 0
    assert s.diff(kernel, delta).subs(delta, 0) == 1 / s.pi
    assert s.simplify((kernel - s.sin(delta) / s.pi).subs(delta, s.pi / 6)) == (
        epsilon / (8 * s.pi)
    )

    # This is an independently declared static source, not an installed law.
    # An irregular graph with a triangle exercises oriented internal-edge
    # cancellation beyond P2, despite the different constitutive kernel.
    graph = nx.Graph(((0, 1), (1, 2), (2, 0), (2, 3)))
    phases = s.symbols("theta0:4", real=True)
    forms = s.symbols("x0:4", real=True)
    capacities = (1, 2, 3, 4)
    total_rate = 0
    for i in graph:
        degree = graph.degree[i]
        gradient = sum(forms[i] - forms[j] for j in graph[i])
        source = (
            sum(kernel.subs(delta, phases[j] - phases[i]) for j in graph[i]) / degree
        )
        form_rate = capacities[i] * (-gradient / (2 * degree) + source / 2)
        total_rate += s.Rational(degree, capacities[i]) * form_rate
    assert s.simplify(total_rate) == 0


def test_cubic_sine_reciprocity_is_regular_through_degree_two_cancellation():
    s = pytest.importorskip("sympy")
    a, b = s.symbols("a b", real=True)
    epsilon = s.Symbol("epsilon", positive=True)

    def kernel_numerator(value):
        return value + epsilon * value**3

    single_mobility = (1 + epsilon * a**2) / s.pi
    pair_mobility = (1 + epsilon * (a**2 - a * b + b**2)) / (2 * s.pi)
    assert s.simplify(single_mobility * a - kernel_numerator(a) / s.pi) == 0
    assert (
        s.simplify(
            pair_mobility * (a + b)
            - (kernel_numerator(a) + kernel_numerator(b)) / (2 * s.pi)
        )
        == 0
    )
    assert s.expand(a**2 - a * b + b**2 - ((a - b) ** 2 + a**2 + b**2) / 2) == 0
    # The factorized mobility has a finite, positive value at S=a+b=0;
    # evaluating a quotient of source by S would lose this admitted case.
    assert s.simplify(pair_mobility.subs(b, -a)) == (1 + 3 * epsilon * a**2) / (
        2 * s.pi
    )
    assert single_mobility.subs(a, 0) == 1 / s.pi

    form_gradient, capacity, beta, weight = s.symbols("q nu beta w", positive=True)
    pressure = (kernel_numerator(a) + kernel_numerator(b)) / (2 * s.pi)
    phase_rate = weight * capacity * pair_mobility * form_gradient / beta
    exchange_work = (
        weight * capacity * form_gradient * pressure - beta * (a + b) * phase_rate
    )
    assert s.expand(exchange_work) == 0


def test_degree_three_critical_star_obstructs_the_same_cubic_phase_completion():
    # These exact sine gaps are realized by acute angles arcsin(s_i).
    # q_root=1 follows from x_root=1/3 and three zero-form leaves.
    sine_gaps = (Q(1, 4), Q(1, 4), -Q(1, 2))
    capacities = (1, 0, 0, 0)
    phase_gradients = (-sum(sine_gaps), *sine_gaps)
    gradient = sum(Q(1, 3) - Q(0) for _ in sine_gaps)
    # Only a live node can supply a phase rate after zero-capacity freezing.
    assert tuple(value for value, nu in zip(phase_gradients, capacities) if nu) == (0,)
    assert gradient == 1
    for epsilon in (Q(1), Q(1, 8), Q(7, 3)):
        # Multiplication by pi keeps the obstruction entirely rational.
        scaled_pressure = sum(value + epsilon * value**3 for value in sine_gaps) / 3
        root_form_work = capacities[0] * gradient * scaled_pressure
        assert root_form_work == -epsilon / 32
        assert root_form_work != 0
        # No finite rate on the live phase coordinate can cancel that work:
        # its coefficient is zero, whereas the source work is nonzero.


def test_cubic_pressure_admits_all_degree_exchange_after_phase_storage_changes():
    s = pytest.importorskip("sympy")
    delta, cosine = s.symbols("delta cosine", real=True)
    epsilon, beta, weight, e = s.symbols("epsilon beta w e", positive=True)
    potential = (
        1
        - s.cos(delta)
        + epsilon * (s.Rational(2, 3) - s.cos(delta) + s.cos(delta) ** 3 / 3)
    )
    kernel_numerator = s.sin(delta) + epsilon * s.sin(delta) ** 3
    assert s.simplify(s.diff(potential, delta) - kernel_numerator) == 0
    assert potential.subs(delta, 0) == 0
    assert s.diff(potential, delta, 2).subs(delta, 0) == 1
    # All factors in the added term are nonnegative for -1<=cosine<=1.
    assert (
        s.expand(
            s.Rational(2, 3)
            - cosine
            + cosine**3 / 3
            - (1 - cosine) ** 2 * (cosine + 2) / 3
        )
        == 0
    )

    graph = nx.star_graph(3)
    x, theta = s.symbols("x0:4", real=True), s.symbols("theta0:4", real=True)
    capacities, degrees = (1, 2, 3, 4), (3, 1, 1, 1)
    storage = sum(
        (x[j] - x[i]) ** 2 / 2 + beta * potential.subs(delta, theta[j] - theta[i])
        for i, j in graph.edges()
    )
    gradient = s.Matrix([s.diff(storage, z) for z in (*x, *theta)])
    form_gradient = s.Matrix([sum(x[i] - x[j] for j in graph[i]) for i in graph])
    pressure = s.Matrix(
        [
            sum(kernel_numerator.subs(delta, theta[j] - theta[i]) for j in graph[i])
            / (s.pi * degrees[i])
            for i in graph
        ]
    )
    form_rate = s.Matrix(
        [
            capacities[i] * (-e * form_gradient[i] / degrees[i] + weight * pressure[i])
            for i in graph
        ]
    )
    phase_rate = s.Matrix(
        [
            weight * capacities[i] * form_gradient[i] / (beta * s.pi * degrees[i])
            for i in graph
        ]
    )
    field = form_rate.col_join(phase_rate)
    mobility = s.diag(*(s.Rational(nu, d) for nu, d in zip(capacities, degrees)))
    zero = s.zeros(4)
    cross = weight * mobility / (beta * s.pi)
    tensor = zero.row_join(-cross).col_join(cross.row_join(zero))
    dissipation = (e * mobility).row_join(zero).col_join(zero.row_join(zero))
    assert (field - (tensor - dissipation) * gradient).applyfunc(s.trigsimp) == s.zeros(
        8, 1
    )
    loss = e * (form_gradient.T * mobility * form_gradient)[0]
    assert s.trigsimp(gradient.dot(field) + loss) == 0
    rho = mobility.inv() * s.ones(4, 1)
    assert s.simplify(rho.dot(form_rate)) == 0

    # At the earlier root-active obstruction, q_root=1 and S_root=0.
    # The changed potential has a nonzero root phase gradient which supplies
    # precisely the exchange work absent from the original cosine storage.
    sine_gaps = (s.Rational(1, 4), s.Rational(1, 4), -s.Rational(1, 2))
    changed_current = sum(value + epsilon * value**3 for value in sine_gaps)
    root_source_work = weight * changed_current / (3 * s.pi)
    root_phase_work = -beta * changed_current * weight / (3 * beta * s.pi)
    assert root_source_work == -weight * epsilon / (32 * s.pi)
    assert s.simplify(root_source_work + root_phase_work) == 0


def test_prescribed_capacity_scaling_preserves_mean_but_reweights_raw_charge():
    forms = (Q(1), Q(0), -Q(1), Q(1, 2))
    capacities = (Q(1), Q(2), Q(3), Q(4))
    graph = _state(
        nx.star_graph(3),
        phases=(0, Q(1, 4), -Q(1, 4), Q(1, 2)),
        forms=forms,
        capacities=capacities,
    )
    source = bound_relational_sine_exchange(graph, reference_model=MODEL)
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, capacities))
    total_weight = sum(weights)
    charge = sum(rho * x for rho, x in zip(weights, forms))
    mean = charge / total_weight
    fixed_weight_rate = sum(
        (rho * rate for rho, rate in zip(weights, source.form_rates)), I(0)
    )
    assert fixed_weight_rate.contains(0)
    assert fixed_weight_rate.width < Q(1, 10**18)

    # This is the jet of a prescribed common capacity factor a(t), with
    # a=1 and a_dot=2, independent of x/theta. No trajectory is evaluated.
    weight_rates = tuple(-2 * rho for rho in weights)
    charge_reweighting = sum(drho * x for drho, x in zip(weight_rates, forms))
    mean_reweighting = (
        sum(drho * (x - mean) for drho, x in zip(weight_rates, forms)) / total_weight
    )
    assert charge_reweighting == -2 * charge != 0
    assert mean_reweighting == 0
    assert (fixed_weight_rate + charge_reweighting).hi < 0

    # Both consumed rows, not just form, scale under the common factor.
    factor = Q(3, 2)
    scaled = bound_relational_sine_exchange(
        _state(
            nx.star_graph(3),
            phases=source.phase,
            forms=forms,
            capacities=tuple(factor * nu for nu in capacities),
        ),
        reference_model=MODEL,
    )
    for name in ("form_rates", "phase_rates"):
        for before, after in zip(getattr(source, name), getattr(scaled, name)):
            difference = after - factor * before
            assert difference.contains(0)
            assert difference.width < Q(1, 10**18)
    scaled_weights = tuple(Q(d) / nu for d, nu in zip(scaled.degrees, scaled.capacity))
    scaled_charge = sum(rho * x for rho, x in zip(scaled_weights, forms))
    assert scaled_charge == charge / factor
    assert scaled_charge / sum(scaled_weights) == mean

    # Prescribing only nu_0_dot=nu_0 violates the common-factor condition.
    # The independent exact reference uses the known rho_0=3 and M=67/98.
    noncommon_mean_rate = -weights[0] * (forms[0] - mean) / total_weight
    assert mean == Q(67, 98)
    assert noncommon_mean_rate == -Q(558, 2401)
    assert (fixed_weight_rate / total_weight + noncommon_mean_rate).hi < 0


def test_no_reset_support_reweighting_distinguishes_charge_and_common_origin():
    def reference(graph, forms, capacities):
        graph = _state(graph, phases=(0, 0, 0), forms=forms, capacities=capacities)
        return derive_forced_support_balance(
            observe_support_transport(graph), epi_weight=1, forcing=(0, 0, 0)
        )

    def comparison(forms, after_capacities):
        before = reference(nx.path_graph(3), forms, (1, 2, 4))
        after = reference(nx.complete_graph(3), forms, after_capacities)
        reset = observe_forced_support_reset(before, after, before.source, after.source)
        old_weights, new_weights = before.metric_weights, after.metric_weights
        charge_jump = sum(
            (new - old) * x for old, new, x in zip(old_weights, new_weights, forms)
        )
        weight_jump = sum(new_weights) - sum(old_weights)
        return reset, charge_jump, weight_jump

    forms = (Q(1), Q(0), Q(2))
    reset, charge_jump, weight_jump = comparison(forms, (1, 2, 4))
    assert reset.before_reference.metric_weights == (1, 1, Q(1, 4))
    assert reset.after_reference.metric_weights == (2, 1, Q(1, 2))
    assert reset.mean_reweighting == Q(4, 21)
    assert charge_jump == Q(3, 2)
    assert weight_jump == Q(5, 4)
    assert reset.raw_support_reset.energy_change == Q(1, 2)
    assert reset.raw_support_reset.identity_residual == 0

    shift = Q(3)
    shifted, shifted_charge_jump, shifted_weight_jump = comparison(
        tuple(x + shift for x in forms), (1, 2, 4)
    )
    assert shifted_charge_jump == charge_jump + shift * weight_jump
    assert shifted_weight_jump == weight_jump
    assert shifted.mean_reweighting == reset.mean_reweighting
    assert (
        shifted.raw_support_reset.energy_change == reset.raw_support_reset.energy_change
    )

    # An explicitly supplied degree-proportional capacity change preserves
    # rho and its total. This is a snapshot compatibility control, not an
    # executed event, capacity mechanism or disappearance of edge work.
    compensated, charge_jump, weight_jump = comparison(forms, (2, 2, 8))
    assert (
        compensated.before_reference.metric_weights
        == compensated.after_reference.metric_weights
    )
    assert compensated.mean_reweighting == charge_jump == weight_jump == 0
    assert compensated.raw_support_reset.energy_change == Q(1, 2)


def test_sine_form_charge_generates_rotation_but_is_not_a_casimir():
    s = pytest.importorskip("sympy")
    x = s.symbols("x0:3", real=True)
    theta = s.symbols("theta0:3", real=True)
    e, w, beta = s.symbols("e w beta", positive=True)
    b = w / (beta * s.pi)
    capacities, degrees = (1, 2, 4), (1, 2, 1)
    mobility = s.diag(*(s.Rational(nu, d) for nu, d in zip(capacities, degrees)))
    rho = mobility.inv() * s.ones(3, 1)
    zero = s.zeros(3)
    tensor = zero.row_join(-b * mobility).col_join((b * mobility).row_join(zero))
    charge_x_gradient = rho.col_join(s.zeros(3, 1))
    charge_theta_gradient = s.zeros(3, 1).col_join(rho)
    assert tensor * charge_x_gradient == s.zeros(3, 1).col_join(b * s.ones(3, 1))
    assert tensor * charge_theta_gradient == (-b * s.ones(3, 1)).col_join(s.zeros(3, 1))
    assert (charge_x_gradient.T * tensor * charge_theta_gradient)[0] == -b * sum(rho)
    assert tensor.det() != 0  # Positive-capacity full bracket has no null generator.

    # Differentiate the independently stated path storage, retaining the
    # dissipative form block and both exchange rows before taking charges.
    storage = sum(
        (x[j] - x[i]) ** 2 / 2 + beta * (1 - s.cos(theta[j] - theta[i]))
        for i, j in ((0, 1), (1, 2))
    )
    gradient = s.Matrix([s.diff(storage, coordinate) for coordinate in (*x, *theta)])
    dissipation = (e * mobility).row_join(zero).col_join(zero.row_join(zero))
    field = (tensor - dissipation) * gradient
    assert s.simplify((charge_x_gradient.T * field)[0]) == 0
    assert s.simplify((charge_theta_gradient.T * field)[0]) == 0

    graph = _state(
        nx.path_graph(3),
        forms=(1, 0, 2),
        phases=(Q(1, 4), 0, -Q(1, 2)),
        capacities=capacities,
    )
    report = bound_relational_sine_exchange(graph, reference_model=MODEL)
    for rates in (report.form_rates, report.phase_rates):
        contraction = sum((Q(r) * rate for r, rate in zip(rho, rates)), I(0))
        assert contraction.contains(0)
        assert contraction.width < Q(1, 10**18)
