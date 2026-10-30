"""Exact admission controls for a marked parent-child relational observation.

The frozen composition-identity-admission protocol permits these symbolic
identities and two instantaneous three-node fixtures. No retained source,
trajectory, event or proposed organization verdict is evaluated here.
The boundary-budget continuation additionally admits the symbolic center
chart and moving-parameter work identities, without a frozen-host model.
"""

from copy import deepcopy
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
import sympy as s

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics._sine_flow import _held_sine_neighbor_field
from tnfr.research.maintained_parent_um_work import _bound_state


def _symbolic_path(*, deeper=False, cut=False):
    """Derive both rows independently from reciprocal edge storage."""
    size = 4 if deeper else 3
    forms = s.symbols(f"x0:{size}", real=True)
    phases = s.symbols(f"theta0:{size}", real=True)
    capacities = s.symbols(f"nu0:{size}", positive=True)
    w, v, z = s.symbols("w v z", positive=True)
    edges = ((0, 2, v),) if cut else ((0, 1, w), (0, 2, v))
    if deeper:
        edges += ((2, 3, z),)
    degrees = tuple(sum(i in edge[:2] for edge in edges) for i in range(size))
    storage = sum(
        weight * (forms[i] - forms[j]) ** 2 / 2 + 1 - s.cos(phases[j] - phases[i])
        for i, j, weight in edges
    )
    mobility = tuple(
        nu / (s.pi * degree) if degree else s.S.Zero
        for nu, degree in zip(capacities, degrees, strict=True)
    )
    # An isolate is explicitly frozen in BOTH rows; no inverse zero degree.
    form_rates = tuple(-m * s.diff(storage, q) for m, q in zip(mobility, phases))
    phase_rates = tuple(m * s.diff(storage, q) for m, q in zip(mobility, forms))
    return SimpleNamespace(
        forms=forms,
        phases=phases,
        capacities=capacities,
        degrees=degrees,
        storage=storage,
        mobility=mobility,
        field=form_rates + phase_rates,
        coordinates=forms + phases,
        r=forms[1] - forms[0],
        delta=phases[1] - phases[0],
        U=s.sin(phases[2] - phases[0]),
        V=v * (forms[0] - forms[2]),
        w=w,
        v=v,
        z=z,
    )


def _derivative(expression, source):
    return sum(
        s.diff(expression, coordinate) * rate
        for coordinate, rate in zip(source.coordinates, source.field, strict=True)
    )


def _zero(expression):
    assert s.cancel(s.trigsimp(s.expand(expression))) == 0


@pytest.fixture(scope="module")
def pair():
    return _symbolic_path()


def test_complete_edge_storage_derives_pair_rows_and_boundary_power(pair):
    a, b = pair.mobility[:2]
    r_dot = _derivative(pair.r, pair)
    delta_dot = _derivative(pair.delta, pair)
    _zero(r_dot + (a + b) * s.sin(pair.delta) + a * pair.U)
    _zero(delta_dot - (a + b) * pair.w * pair.r + a * pair.V)
    energy = pair.w * pair.r**2 / 2 + 1 - s.cos(pair.delta)
    power = -a * (pair.w * pair.r * pair.U + s.sin(pair.delta) * pair.V)
    _zero(_derivative(energy, pair) - power)
    _zero(_derivative(pair.storage, pair))
    rho = tuple(d / nu for d, nu in zip(pair.degrees, pair.capacities, strict=True))
    for row in (pair.forms, pair.phases):
        _zero(_derivative(sum(q * weight for q, weight in zip(row, rho)), pair))


def test_angle_response_is_derivative_of_the_marked_observation(pair):
    # atan2 needs a continuous angle branch; its derivative excludes the origin.
    horizontal, vertical = pair.delta, s.sqrt(pair.w) * pair.r
    angle = s.atan2(vertical, horizontal)
    independent_derivative = _derivative(angle, pair)
    a, b = pair.mobility[:2]
    radius_squared = pair.delta**2 + pair.w * pair.r**2
    expected_numerator = -(a + b) * s.sqrt(pair.w) * (
        pair.delta * s.sin(pair.delta) + pair.w * pair.r**2
    ) + a * s.sqrt(pair.w) * (pair.r * pair.V - pair.delta * pair.U)
    _zero(independent_derivative * radius_squared - expected_numerator)


def test_radius_work_bound_retains_both_boundary_ports():
    r, delta, U, V = s.symbols("r delta U V", real=True)
    w = s.symbols("w", positive=True)
    energy = w * r**2 / 2 + 1 - s.cos(delta)
    # These nonnegative remainders prove the two inequalities used in the
    # conditional sqrt(E) work bound; no source or port bound is supplied.
    _zero(2 * energy - w * r**2 - s.sin(delta) ** 2 - (1 - s.cos(delta)) ** 2)
    _zero(
        (w * r**2 + s.sin(delta) ** 2) * (w * U**2 + V**2)
        - (w * r * U + s.sin(delta) * V) ** 2
        - w * (r * V - s.sin(delta) * U) ** 2
    )


def test_hidden_form_preserves_pair_and_global_mean_but_changes_relative_rate():
    source = _symbolic_path(deeper=True)
    eta = s.symbols("eta", real=True, nonzero=True)
    rho = tuple(d / nu for d, nu in zip(source.degrees, source.capacities))
    changed = {
        source.forms[2]: source.forms[2] + eta,
        source.forms[3]: source.forms[3] - eta * rho[2] / rho[3],
    }
    weighted_form = sum(weight * x for weight, x in zip(rho, source.forms))
    for observed in (*source.forms[:2], *source.phases, weighted_form):
        _zero(observed.subs(changed, simultaneous=True) - observed)
    delta_rate = _derivative(source.delta, source)
    difference = s.simplify(delta_rate.subs(changed, simultaneous=True) - delta_rate)
    assert difference == source.mobility[0] * source.v * eta
    assert difference != 0


def test_pair_and_instantaneous_ports_still_hide_relative_acceleration():
    source = _symbolic_path(deeper=True)
    eta = s.symbols("eta", real=True, nonzero=True)
    rho = tuple(d / nu for d, nu in zip(source.degrees, source.capacities))
    common_shift = eta * rho[3] / sum(rho)
    changed = {
        x: x - common_shift + (eta if i == 3 else 0) for i, x in enumerate(source.forms)
    }
    weighted_form = sum(weight * x for weight, x in zip(rho, source.forms))
    r_dot, delta_dot = (_derivative(q, source) for q in (source.r, source.delta))
    for observed in (
        source.r,
        source.delta,
        source.U,
        source.V,
        weighted_form,
        r_dot,
        delta_dot,
        _derivative(source.V, source),
    ):
        _zero(observed.subs(changed, simultaneous=True) - observed)
    U_dot = _derivative(source.U, source)
    difference = U_dot.subs(changed, simultaneous=True) - U_dot
    expected = (
        -eta
        * source.mobility[2]
        * source.z
        * s.cos(source.phases[2] - source.phases[0])
    )
    _zero(difference - expected)
    acceleration = _derivative(r_dot, source)
    _zero(
        acceleration.subs(changed, simultaneous=True)
        - acceleration
        + source.mobility[0] * expected
    )
    # On an acute parent/rest branch the cosine is positive, hence the two
    # accelerations differ for eta != 0. No retained source-box claim follows.
    assert s.simplify(expected) != 0


def test_matched_cut_rebuilds_degree_and_declares_both_isolate_rows(pair):
    cut = _symbolic_path(cut=True)
    assert pair.degrees == (2, 1, 1)
    assert cut.degrees == (1, 0, 1)
    assert cut.mobility[0] == 2 * pair.mobility[0]
    assert cut.field[1] == cut.field[4] == 0
    _zero(_derivative(cut.r, cut) + cut.mobility[0] * cut.U)
    _zero(_derivative(cut.delta, cut) + cut.mobility[0] * cut.V)
    # The observation keeps its supplied w even after the support edge is cut.
    energy = cut.w * cut.r**2 / 2 + 1 - s.cos(cut.delta)
    _zero(
        _derivative(energy, cut)
        + cut.mobility[0] * (cut.w * cut.r * cut.U + s.sin(cut.delta) * cut.V)
    )


MODEL = RelationalExchangeModel(1, epi_weight=0, phase_weight=1, phase_domain="regular")
PHASES = ((Q(0), Q(0), Q(0)), (Q(0), Q(1, 4), Q(1, 2)))


def _enclosure_agrees(actual, expected):
    # Independent analytic regroupings can differ in outward rounding width.
    residual = actual - expected
    assert residual.contains(0)
    assert residual.width < Q(1, 2**90)


@pytest.mark.parametrize("phases", PHASES)
def test_fixed_full_engine_rows_match_independent_pair_and_power(phases):
    forms, capacities = [Q(0), Q(1), Q(2)], [Q(2), Q(3), Q(5)]
    rows = [[(1, 1, Q(2)), (2, 1, Q(3))], [(0, 1, Q(2))], [(0, 1, Q(3))]]
    before = deepcopy((forms, capacities, rows))
    flow = _held_sine_neighbor_field(MODEL, (2, 1, 1), capacities, rows)
    result = flow(forms + list(phases))
    assert (forms, capacities, rows) == before
    pi, child_sine, rest_sine = pi_interval(), sin(I(phases[1])), sin(I(phases[2]))
    # Exact hand reduction for x=(0,1,2), nu=(2,3,5), W=(2,3).
    expected = (
        (child_sine + rest_sine) / pi,
        -3 * child_sine / pi,
        -5 * rest_sine / pi,
        -8 / pi,
        6 / pi,
        30 / pi,
    )
    for actual, target in zip(result, expected, strict=True):
        _enclosure_agrees(actual, target)
    r_dot, delta_dot = result[1] - result[0], result[4] - result[3]
    _enclosure_agrees(r_dot, (-4 * child_sine - rest_sine) / pi)
    _enclosure_agrees(delta_dot, 14 / pi)
    _enclosure_agrees(
        2 * r_dot + child_sine * delta_dot, (6 * child_sine - 2 * rest_sine) / pi
    )
    if phases == PHASES[0]:
        assert all(value.contains(0) for value in result[:3])
        assert delta_dot.lo > 0  # Zero form rates do not freeze the complete state.
    else:
        assert r_dot.hi < 0
        assert (2 * r_dot + child_sine * delta_dot).lo > 0


@pytest.mark.parametrize("phases", PHASES)
def test_fixed_cut_engine_obeys_explicit_frozen_isolate_completion(phases):
    # Same preparation and capacity; only the child edge is absent.
    cut = _bound_state(
        (Q(0), Q(1), Q(2)),
        tuple(map(I, phases)),
        tuple(map(I, (2, 3, 5))),
        ((0, 2),),
        (I(3),),
    )
    assert cut.degrees == (1, 0, 1)
    assert cut.pressure[1] == cut.form_rate[1] == cut.phase_rate[1] == I(0)
    assert cut.mean_weights[1] == I(0)
    assert not cut.positive_mean_metric
    pi, rest_sine = pi_interval(), sin(I(phases[2]))
    _enclosure_agrees(cut.form_rate[1] - cut.form_rate[0], -2 * rest_sine / pi)
    _enclosure_agrees(cut.phase_rate[1] - cut.phase_rate[0], 12 / pi)
    with pytest.raises(ValueError, match="positive ordinary integers"):
        _held_sine_neighbor_field(
            MODEL, (1, 0, 1), (2, 3, 5), (((2, 1, 3),), (), ((0, 1, 3),))
        )


@pytest.fixture(scope="module")
def center_chart():
    """Re-express the same generic full path, with an evolving deeper host."""
    source = _symbolic_path(deeper=True)
    mobility = s.symbols("m_a m_c m_j m_l", positive=True)
    coefficient_map = {
        nu: s.pi * degree * m
        for nu, degree, m in zip(
            source.capacities, source.degrees, mobility, strict=True
        )
    }
    source.field = tuple(s.simplify(q.subs(coefficient_map)) for q in source.field)
    source.mobility = mobility
    a, b = mobility[:2]
    k, gamma, mu = a + b, a / (a + b), a * b / (a + b)
    r, delta, X, theta = s.symbols("r delta X Theta", real=True)
    center_form = source.forms[0] + gamma * source.r
    center_phase = source.phases[0] + gamma * source.delta
    inverse = {
        source.forms[0]: X - gamma * r,
        source.forms[1]: X + (1 - gamma) * r,
        source.phases[0]: theta - gamma * delta,
        source.phases[1]: theta + (1 - gamma) * delta,
    }
    observations = (
        source.r,
        source.delta,
        center_form,
        center_phase,
        *source.forms[2:],
        *source.phases[2:],
    )
    coordinates = (r, delta, X, theta, *source.forms[2:], *source.phases[2:])
    rates = tuple(
        _derivative(q, source).subs(inverse, simultaneous=True) for q in observations
    )
    alpha = source.phases[2] - theta
    V0 = source.v * (X - source.forms[2])
    C, S = s.cos(alpha), s.sin(alpha)
    kappa = source.w + gamma**2 * source.v
    relative = (
        kappa * r**2 / 2
        - gamma * V0 * r
        + 1
        - s.cos(delta)
        + C * (1 - s.cos(gamma * delta))
        + S * s.sin(gamma * delta)
    )
    # The deeper edge is retained in the baseline, so this is a full graph
    # decomposition rather than a pair supplied with independent controls.
    baseline = (
        source.z * (source.forms[2] - source.forms[3]) ** 2 / 2
        + 1
        - s.cos(source.phases[3] - source.phases[2])
        + source.v * (X - source.forms[2]) ** 2 / 2
        + 1
        - C
    )
    return SimpleNamespace(
        source=source,
        coordinates=coordinates,
        field=rates,
        observations=observations,
        inverse=inverse,
        k=k,
        gamma=gamma,
        mu=mu,
        r=r,
        delta=delta,
        X=X,
        theta=theta,
        V0=V0,
        C=C,
        S=S,
        kappa=kappa,
        relative=relative,
        baseline=baseline,
    )


def _bracket(left, right, source):
    return sum(
        m * (s.diff(left, q) * s.diff(right, x) - s.diff(left, x) * s.diff(right, q))
        for x, q, m in zip(source.forms, source.phases, source.mobility, strict=True)
    )


def test_pair_center_chart_is_invertible_and_has_uncoupled_poisson_blocks(center_chart):
    chart, source = center_chart, center_chart.source
    for observed, coordinate in zip(chart.observations, chart.coordinates, strict=True):
        _zero(observed.subs(chart.inverse, simultaneous=True) - coordinate)
    r, delta, X, theta = chart.observations[:4]
    _zero(_bracket(r, delta, source) + chart.k)
    _zero(_bracket(X, theta, source) + chart.mu)
    for relative in (r, delta):
        for external in chart.observations[2:]:
            _zero(_bracket(relative, external, source))
    for center in (X, theta):
        for external in chart.observations[4:]:
            _zero(_bracket(center, external, source))
    a, b = source.mobility[:2]
    _zero(source.forms[0] / a + source.forms[1] / b - X / chart.mu)
    _zero(source.phases[0] / a + source.phases[1] / b - theta / chart.mu)


def test_complete_storage_splits_into_center_baseline_and_relative_part(center_chart):
    chart = center_chart
    full_storage = chart.source.storage.subs(chart.inverse, simultaneous=True)
    _zero(s.expand_trig(full_storage - chart.baseline - chart.relative))
    assert chart.relative.subs({chart.r: 0, chart.delta: 0}) == 0


def test_center_and_relative_rates_come_from_the_complete_law(center_chart):
    chart = center_chart
    U = chart.S * s.cos(chart.gamma * chart.delta) + chart.C * s.sin(
        chart.gamma * chart.delta
    )
    V = chart.V0 - chart.gamma * chart.source.v * chart.r
    expected = (
        -chart.k * s.diff(chart.relative, chart.delta),
        chart.k * s.diff(chart.relative, chart.r),
        chart.mu * U,
        chart.mu * V,
    )
    for actual, target in zip(chart.field[:4], expected, strict=True):
        _zero(s.expand_trig(actual - target))


def test_relative_moving_parameter_work_uses_all_evolving_host_rows(center_chart):
    chart = center_chart
    # Every derivative below uses the actual full path field pulled into the
    # invertible chart. In particular, center and host coordinates evolve.
    work = (
        -chart.gamma * chart.r * _derivative(chart.V0, chart)
        + (1 - s.cos(chart.gamma * chart.delta)) * _derivative(chart.C, chart)
        + s.sin(chart.gamma * chart.delta) * _derivative(chart.S, chart)
    )
    _zero(s.expand_trig(_derivative(chart.relative, chart) - work))
    _zero(s.expand_trig(_derivative(chart.baseline, chart) + work))
    # A deeper host form enters this work. The boundary parameters are not
    # an autonomous pair state or licensed to be frozen during continuation.
    assert s.diff(work, chart.source.forms[3]) != 0


def test_zero_baseline_relative_storage_need_not_be_nonnegative(center_chart):
    chart = center_chart
    supplied_V = s.symbols("V_nonzero", real=True, nonzero=True)
    changed = {
        chart.source.forms[2]: chart.X - supplied_V / chart.source.v,
        chart.r: chart.gamma * supplied_V / chart.kappa,
        chart.delta: 0,
    }
    value = chart.relative.subs(changed, simultaneous=True)
    expected = -chart.gamma**2 * supplied_V**2 / (2 * chart.kappa)
    _zero(value - expected)
    assert expected.is_negative
    # A square-root work budget for this K needs a new coercivity argument;
    # the previous nonnegative bridge-energy annulus cannot be reused.


def test_uniform_certificate_ceiling_has_an_exact_nonnegative_remainder():
    C, lower, margin = s.symbols("C lower margin", positive=True)
    # Writing s=lower+margin enforces the entire open domain 0<lower<s.
    initial = lower + margin
    denominator = C**2 * lower + margin
    objective = lower * margin / denominator
    ceiling = initial / (1 + C) ** 2
    remainder = (initial - (1 + C) * lower) ** 2 / ((1 + C) ** 2 * denominator)
    _zero(ceiling - objective - remainder)
    assert remainder.is_nonnegative
    # Free symbolic C=sqrt(2*pi) specializes the bound without a parameter
    # scan or selecting a scientific annulus from its relaxed optimizer.
