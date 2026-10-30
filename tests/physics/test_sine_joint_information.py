"""Independent joint-preparation controls for the complete nonlinear sine law.

The full critical tangent has closed heat-mode oracles. Actual nonlinear
responses are enclosed through a separately proved remainder, never sampled.
The deterministic affine preparation is an independent premise, not covariance.
"""

from dataclasses import FrozenInstanceError, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from functools import lru_cache
from itertools import permutations, product, repeat
from math import factorial, prod

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics import _sine_flow as flow
from tnfr.physics import _sine_joint_information as owner


@dataclass(frozen=True)
class _B:
    lo: Q
    hi: Q | None = None

    def __post_init__(self):
        object.__setattr__(self, "lo", Q(self.lo))
        object.__setattr__(self, "hi", self.lo if self.hi is None else Q(self.hi))
        assert self.lo <= self.hi

    def __add__(self, other):
        other = other if isinstance(other, _B) else _B(other)
        return _B(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return _B(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -(other if isinstance(other, _B) else _B(other))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, _B) else _B(other)
        endpoints = tuple(
            a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)
        )
        return _B(min(endpoints), max(endpoints))

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1 / Q(scalar))

    def __pow__(self, exponent):
        result = _B(1)
        for _ in range(exponent):
            result *= self
        return result


def _decay(value):
    value = Q(value)
    assert 0 <= value <= 1
    upper = sum(((-value) ** k / factorial(k) for k in range(101)), Q(0))
    return _B(upper - value**101 / factorial(101), upper)


def _sine(value):
    value = Q(value)
    if value < 0:
        return -_sine(-value)
    assert value <= 1
    upper = sum(
        ((-1) ** k * value ** (2 * k + 1) / factorial(2 * k + 1) for k in range(51)),
        Q(0),
    )
    return _B(upper - value**103 / factorial(103), upper)


def _hyperbolic(time):
    square = Q(time) ** 2 / 8
    even = sum((square**k / factorial(2 * k) for k in range(49)), Q(0))
    odd = sum((square**k / factorial(2 * k + 1) for k in range(49)), Q(0))
    even_tail = square**49 / factorial(98) / (1 - square / (99 * 100))
    odd_tail = square**49 / factorial(99) / (1 - square / (100 * 101))
    return _B(even, even + even_tail), Q(time) / 4 * _B(odd, odd + odd_tail)


def _on_heat(time, initial):
    """Independent P4 modes and their first two heat-time derivatives."""
    x0, x1, x2, x3 = map(Q, initial)
    a, b = (x0 + x3) / 2, (x1 + x2) / 2
    p, q = (x0 - x3) / 2, (x1 - x2) / 2
    mean, contrast = (a + b) / 2, (a - b) / 2
    e = _decay(time / 2)
    c, k = _hyperbolic(time)
    symmetric = (mean + contrast * e, mean - contrast * e)
    symmetric_rate = (-contrast * e / 2, contrast * e / 2)
    symmetric_acceleration = (contrast * e / 4, -contrast * e / 4)
    anti = (e * (c * p + k * (p + q)), e * (c * q + k * (p - q)))
    differentiated_mode = (
        e * (k * p / 2 + c * (p + q) / 4),
        e * (k * q / 2 + c * (p - q) / 4),
    )
    anti_rate = tuple(
        -value / 2 + derivative for value, derivative in zip(anti, differentiated_mode)
    )
    anti_acceleration = tuple(
        3 * value / 8 - derivative
        for value, derivative in zip(anti, differentiated_mode)
    )

    def lift(sym, antisym):
        return (
            sym[0] + antisym[0],
            sym[1] + antisym[1],
            sym[1] - antisym[1],
            sym[0] - antisym[0],
        )

    return (
        lift(symmetric, anti),
        lift(symmetric_rate, anti_rate),
        lift(symmetric_acceleration, anti_acceleration),
    )


def _off_heat(time, initial):
    x0, x1, x2, x3 = map(Q, initial)
    mean = (x1 + x2 + x3) / 3
    odd = (x1 - x3) / 2 * _decay(time / 4)
    even = (x1 - 2 * x2 + x3) / 6 * _decay(3 * time / 4)
    return (
        (_B(x0), mean + odd + even, mean - 2 * even, mean - odd + even),
        (_B(0), -odd / 4 - 3 * even / 4, 3 * even / 2, odd / 4 - 3 * even / 4),
        (_B(0), odd / 16 + 9 * even / 16, -9 * even / 8, -odd / 16 + 9 * even / 16),
    )


def _heat_rows(state, edges, capacity=(Q(1),) * 4):
    rows = [_B(0) for _ in state]
    for i, j in edges:
        difference = (state[j] - state[i]) / 4
        rows[i] += capacity[i] * difference
        rows[j] -= capacity[j] * difference
    return tuple(rows)


def _tangent_rows(state, edges, capacity=(Q(1),) * 4):
    form_heat = _heat_rows(state[:4], edges, capacity)
    phase_heat = _heat_rows(state[4:], edges, capacity)
    return tuple(x + theta / 2 for x, theta in zip(form_heat, phase_heat)) + tuple(
        -x / 2 for x in form_heat
    )


def _zero(value):
    assert value.lo <= 0 <= value.hi
    assert value.hi - value.lo < Q(1, 10**100)


def _contains(endpoints, oracle, tolerance=Q(1, 10**30)):
    assert endpoints[0] <= oracle.lo <= oracle.hi <= endpoints[1]
    assert endpoints[1] - endpoints[0] < tolerance


def _display(value):
    with localcontext() as context:
        context.prec = 25
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _tangent(time, initial_form, initial_phase, *, on):
    """Both critical tangent rows, including their physical-time derivatives."""
    t = Q(time) / 2
    solve = _on_heat if on else _off_heat
    form, form_rate, form_acc = solve(t, initial_form)
    phase, phase_rate, phase_acc = solve(t, initial_phase)
    summed_rate = tuple(x + p for x, p in zip(form_rate, phase_rate))
    summed_acc = tuple(x + p for x, p in zip(form_acc, phase_acc))
    state = tuple(x + t * r for x, r in zip(form, summed_rate)) + tuple(
        p - t * r for p, r in zip(phase, summed_rate)
    )
    rate = tuple(
        x + p / 2 + t * a / 2 for x, p, a in zip(form_rate, phase_rate, summed_acc)
    ) + tuple(-x / 2 - t * a / 2 for x, a in zip(form_rate, summed_acc))
    return state, rate


@lru_cache(maxsize=None)
def _coefficients(time):
    """Independent full form and phase response rows, plus heat contrast."""
    form, phase, heat = [], [], []
    zero = (Q(0),) * 4
    for j in range(4):
        basis = tuple(Q(i == j) for i in range(4))
        for initial_form, initial_phase, target in (
            (basis, zero, form),
            (zero, basis, phase),
        ):
            on, _ = _tangent(time, initial_form, initial_phase, on=True)
            off, _ = _tangent(time, initial_form, initial_phase, on=False)
            target.append((on[0] + on[1] - off[0] - off[1]) / 2)
        on, _, _ = _on_heat(Q(time) / 2, basis)
        off, _, _ = _off_heat(Q(time) / 2, basis)
        heat.append((on[0] + on[1] - off[0] - off[1]) / 2)
    return tuple(form), tuple(phase), tuple(heat)


def _arguments(kind="linked", **changes):
    delta, epsilon = Q(1, 10000), Q(1, 10**6)
    zero = (Q(0),) * 4
    form2, form3 = (delta, Q(0), Q(0), Q(0)), (Q(0), delta, Q(0), Q(0))
    if kind == "linked":
        phase2, phase3 = (-delta, Q(0), epsilon, Q(0)), (Q(0), -delta, Q(0), epsilon)
    elif kind == "crossed":
        phase2, phase3 = (Q(0), -delta, epsilon, Q(0)), (-delta, Q(0), Q(0), epsilon)
    else:
        assert kind == "independent"
        phase2, phase3 = (Q(0), Q(0), delta + epsilon, Q(0)), (
            Q(0),
            Q(0),
            Q(0),
            delta + epsilon,
        )
    values = dict(
        nodes=(0, 1, 2, 3),
        old_edges=((1, 2), (2, 3)),
        old_conductances=(Q(1), Q(1)),
        capacity=(Q(1),) * 4,
        contact=(0, 1),
        contact_weight=Q(1),
        retained_nodes=(0, 1),
        observation_nodes=(0, 1),
        phase_anchor=0,
        coordinate_labels=tuple(
            (i, channel) for channel in ("form", "relative_phase") for i in range(4)
        ),
        initial_center=(Q(0), Q(0), 2 * delta, Q(0), Q(0), Q(0), -2 * delta, Q(0)),
        initial_generators=(zero, zero, form2, form3, zero, zero, phase2, phase3),
        uncertainty_labels=("u", "v", "w2", "w3"),
        preparation_relation_cost=0 if kind == "independent" else 2,
        elapsed_time=Q(1, 4),
        mean_kind="arithmetic",
        reading_error_bound=Q(0),
        tolerance=Q(3, 10**10),
    )
    values.update(changes)
    return values


def _report(kind="linked", **changes):
    return owner._prepared_sine_joint_information(**_arguments(kind, **changes))


def _source(arguments, latent):
    return tuple(
        center + sum((v * u for v, u in zip(row, latent)), Q(0))
        for center, row in zip(
            arguments["initial_center"], arguments["initial_generators"]
        )
    )


def _determinant(matrix):
    size = len(matrix)
    return sum(
        (
            (-1)
            ** sum(order[i] > order[j] for i in range(size) for j in range(i + 1, size))
            * prod(matrix[i][order[i]] for i in range(size))
            for order in permutations(range(size))
        ),
        Q(0),
    )


def _projected_oracle(arguments):
    form, phase, _ = _coefficients(arguments["elapsed_time"])
    physical = form + phase
    coefficients = tuple(
        physical[node + (0 if channel == "form" else 4)]
        for node, channel in arguments["coordinate_labels"]
    )
    center = sum(
        (a * x for a, x in zip(coefficients, arguments["initial_center"])), _B(0)
    )
    latent = tuple(
        sum(
            (
                a * row[j]
                for a, row in zip(coefficients, arguments["initial_generators"])
            ),
            _B(0),
        )
        for j in range(len(arguments["uncertainty_labels"]))
    )
    return center, latent


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_acquired_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("joint information controls cannot integrate or execute events")

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
def reports():
    return {kind: _report(kind) for kind in ("linked", "crossed", "independent")}


@pytest.mark.parametrize("time", [Q(0), Q(1, 100), Q(1, 4)])
@pytest.mark.parametrize(
    "initial_form,initial_phase",
    [
        ((0, 0, Q(1, 5000), 0), (0, 0, Q(-1, 5000), 0)),
        ((0, 0, Q(3, 10000), Q(-1, 10000)), (0, 0, Q(-101, 10**6), Q(-99, 10**6))),
        (
            (Q(1, 10000), Q(-1, 10000), 0, 0),
            (0, Q(1, 20000), Q(-1, 10000), Q(1, 10000)),
        ),
    ],
)
def test_closed_full_tangent_keeps_nonzero_initial_form_and_both_dynamic_rows(
    time, initial_form, initial_phase
):
    for on, edges in ((True, ((0, 1), (1, 2), (2, 3))), (False, ((1, 2), (2, 3)))):
        state, derivative = _tangent(time, initial_form, initial_phase, on=on)
        for actual, expected in zip(derivative, _tangent_rows(state, edges)):
            _zero(actual - expected)
        _zero(sum(state[:4], _B(0)) - sum(initial_form))
        _zero(sum(state[4:], _B(0)) - sum(initial_phase))
        if time == 0:
            assert state == tuple(_B(x) for x in (*initial_form, *initial_phase))


def test_independent_full_rows_and_closed_heat_brackets():
    form, phase, heat = _coefficients(Q(1, 4))
    t = Q(1, 8)
    r, e = _decay(t / 4), _decay(t / 2)
    c, k = _hyperbolic(t)
    a2, a3 = (
        Q(1, 12) + r**3 / 6 - e * c / 4,
        Q(1, 12) + r / 4 - r**3 / 12 - e * c / 4 - e * k / 2,
    )
    b2, b3 = t / 8 * (e * (c - k) - r**3), t / 16 * (-r + r**3 + 2 * e * k)
    for actual, expected in (
        (heat[2], a2),
        (heat[3], a3),
        (phase[2], b2),
        (phase[3], b3),
    ):
        _zero(actual - expected)
    for a, b, combined in zip(heat, phase, form):
        _zero(a + b - combined)
    for actual, lower, upper in (
        (a2, Q(15, 6291456), Q(1, 393216)),
        (a3, Q(15, 805306368), Q(1, 50331648)),
        (b2, Q(105, 16777216), Q(1, 131072)),
        (b3, Q(105, 1610612736), Q(1, 12582912)),
    ):
        assert lower < actual.lo <= actual.hi < upper


def test_same_marginals_full_rank_and_equal_linked_crossed_volume(
    reports, record_property
):
    delta, epsilon = Q(1, 10000), Q(1, 10**6)
    determinants = {}
    marginal_rows = []
    for kind, report in reports.items():
        args = _arguments(kind)
        corners = tuple(_source(args, latent) for latent in product((-1, 1), repeat=4))
        assert len(set(corners)) == 16
        marginals = tuple(
            (min(row[i] for row in corners), max(row[i] for row in corners))
            for i in range(8)
        )
        assert report.coordinate_marginal_bounds == marginals
        marginal_rows.append(marginals)
        hidden_matrix = tuple(args["initial_generators"][i] for i in (2, 3, 6, 7))
        determinants[kind] = _determinant(hidden_matrix)
        assert determinants[kind] != 0
        assert report.uncertainty_dimension == 4
        assert report.measurement_cost == report.evolved_coordinate_count == 0
        assert report.static_coordinate_count == 8
        assert report.preparation_relation_cost == (0 if kind == "independent" else 2)
    assert marginal_rows[0] == marginal_rows[1] == marginal_rows[2]
    assert determinants["linked"] == determinants["crossed"] == delta**2 * epsilon**2
    assert determinants["independent"] == delta**2 * (delta + epsilon) ** 2
    record_property("linked_crossed_hidden_rank", "4; nonzero exact determinant")
    record_property("linked_crossed_same_volume", str(16 * delta**2 * epsilon**2))
    record_property("equal_coordinate_marginals", tuple(map(str, marginal_rows[0])))


@pytest.mark.parametrize("kind", ["linked", "crossed"])
def test_affine_images_are_the_stated_noisy_relation_bands_with_correct_association(
    kind,
):
    args = _arguments(kind)
    delta, epsilon = Q(1, 10000), Q(1, 10**6)
    samples = (*product((-1, 1), repeat=4), (Q(1, 3), Q(-2, 5), Q(3, 7), Q(-4, 9)))
    for latent in samples:
        state = _source(args, latent)
        x2, x3, p2, p3 = (state[i] for i in (2, 3, 6, 7))
        assert delta <= x2 <= 3 * delta and -delta <= x3 <= delta
        residual2, residual3 = (
            (x2 + p2, x3 + p3)
            if kind == "linked"
            else (x3 + p2 + 2 * delta, x2 + p3 - 2 * delta)
        )
        assert residual2 == epsilon * latent[2]
        assert residual3 == epsilon * latent[3]
        assert abs(residual2) <= epsilon and abs(residual3) <= epsilon
        recovered = (
            (x2 - 2 * delta) / delta,
            x3 / delta,
            residual2 / epsilon,
            residual3 / epsilon,
        )
        assert recovered == latent
    # These invertible formulas prove equivalence to the complete relation-band
    # family, not merely membership of the sampled vertices. Both relation rows
    # have independent phase pivots, so their constraint rank is two.


def test_affine_contraction_before_absolute_values_retains_joint_cancellation(
    reports, record_property
):
    delta, epsilon = Q(1, 10000), Q(1, 10**6)
    form, phase, heat = _coefficients(Q(1, 4))
    expected = {
        "linked": (
            delta * heat[2],
            delta * heat[3],
            epsilon * phase[2],
            epsilon * phase[3],
        ),
        "crossed": (
            delta * (form[2] - phase[3]),
            delta * (form[3] - phase[2]),
            epsilon * phase[2],
            epsilon * phase[3],
        ),
        "independent": (
            delta * form[2],
            delta * form[3],
            (delta + epsilon) * phase[2],
            (delta + epsilon) * phase[3],
        ),
    }
    for kind, report in reports.items():
        center, generic = _projected_oracle(_arguments(kind))
        _zero(center - 2 * delta * heat[2])
        _contains(report.tangent_center_bounds, center)
        for actual, oracle, direct in zip(
            report.tangent_generator_coefficient_bounds, expected[kind], generic
        ):
            _contains(actual, oracle)
            _zero(oracle - direct)
        signed = expected[kind]
        assert signed[0].lo > 0
        assert signed[1].hi < 0 if kind == "crossed" else signed[1].lo > 0
        radius = sum(((-v if v.hi < 0 else v) for v in signed), _B(0))
        assert (
            report.linear_information_error_bounds[0]
            <= radius.lo
            <= radius.hi
            <= report.linear_information_error_bounds[1]
        )
        record_property(
            f"{kind}_tangent_radius", tuple(_display(x) for x in (radius.lo, radius.hi))
        )
    assert (
        reports["linked"].linear_information_error_bounds[1]
        < reports["crossed"].linear_information_error_bounds[0]
    )
    assert (
        reports["linked"].linear_information_error_bounds[1]
        < reports["independent"].linear_information_error_bounds[0]
    )


@pytest.mark.parametrize("kind", ["linked", "crossed", "independent"])
def test_all_sixteen_affine_corners_and_attaining_tangent_extrema(kind, reports):
    args, report = _arguments(kind), reports[kind]
    center, latent = _projected_oracle(args)
    physical = sum(_coefficients(Q(1, 4))[:2], ())
    values = []
    for corner in product((-1, 1), repeat=4):
        state = _source(args, corner)
        direct = sum((a * x for a, x in zip(physical, state)), _B(0))
        affine = center + sum((d * u for d, u in zip(latent, corner)), _B(0))
        _zero(direct - affine)
        model = report.nonlinear_response_error_upper_bound
        assert report.forecast_response_bounds[0] <= direct.lo - model
        assert direct.hi + model <= report.forecast_response_bounds[1]
        values.append((corner, direct))
    maximum_corner = tuple(-1 if d.hi < 0 else 1 for d in latent)
    minimum_corner = tuple(-u for u in maximum_corner)
    maximum = next(value for corner, value in values if corner == maximum_corner)
    minimum = next(value for corner, value in values if corner == minimum_corner)
    radius = sum(((-d if d.hi < 0 else d) for d in latent), _B(0))
    _zero((maximum + minimum) / 2 - center)
    _zero((maximum - minimum) / 2 - radius)
    # Actual nonlinear extrema are not asserted to occur at these vertices;
    # their tangent separation transfers only after subtracting the uniform E.
    if kind != "linked":
        assert (
            (maximum - minimum) / 2
        ).lo - report.nonlinear_response_error_upper_bound > Q(14, 10**10)


def test_fixed_nonlinear_forecast_depends_on_relation_association_not_rank(
    reports, record_property
):
    linked = reports["linked"]
    assert linked.status == "certified" and linked.response_sign == "positive"
    assert linked.forecast_error_upper_bound < Q(272, 10**12) < linked.tolerance
    assert linked.forecast_response_bounds[0] > Q(22, 10**11)
    assert linked.computational_center_error_upper_bound < Q(1, 10**30)
    assert (
        linked.forecast_error_upper_bound
        == linked.linear_information_error_bounds[1]
        + linked.nonlinear_response_error_upper_bound
        + linked.computational_center_error_upper_bound
    )
    for kind in ("crossed", "independent"):
        report = reports[kind]
        assert report.status == "insufficient_information"
        assert report.information_error_lower_bound > Q(14, 10**10) > report.tolerance
    assert (
        reports["linked"].nonlinear_response_error_upper_bound
        == reports["crossed"].nonlinear_response_error_upper_bound
        == reports["independent"].nonlinear_response_error_upper_bound
    )
    record_property("fixed_response_tolerance", str(linked.tolerance))
    record_property("linked_forecast", _display(linked.predicted_response))
    record_property("linked_total_error", _display(linked.forecast_error_upper_bound))
    record_property(
        "nonlinear_remainder", _display(linked.nonlinear_response_error_upper_bound)
    )
    record_property(
        "certified_actual_response_interval",
        tuple(_display(x) for x in linked.forecast_response_bounds),
    )
    record_property(
        "crossed_unavoidable_error_lower",
        _display(reports["crossed"].information_error_lower_bound),
    )
    record_property(
        "marginal_only_unavoidable_error_lower",
        _display(reports["independent"].information_error_lower_bound),
    )
    record_property(
        "interpretation",
        "actual nonlinear bounds; full-rank affine preparations; no nonlinear endpoint evaluated",
    )


def test_nonzero_initial_form_changes_phase_tube_and_nonlinear_allowance(reports):
    report = reports["linked"]
    b0, p0, z, c = Q(3, 10000), Q(301, 10**6), Q(1, 8), Q(128, 127)
    assert report.initial_form_upper_bound == b0
    assert report.initial_phase_upper_bound == p0
    assert c - 1 - z**2 * c / 2 == 0
    phase_tube = (p0 + z * b0) * c
    forcing = 2 * phase_tube**3 / 6
    branch = Q(1, 4) * forcing * c
    assert report.off_phase_growth_bound == report.on_phase_growth_bound == c
    assert (
        report.off_phase_tube_upper_bound
        == report.on_phase_tube_upper_bound
        == phase_tube
    )
    assert (
        report.off_nonlinear_error_upper_bound
        == report.on_nonlinear_error_upper_bound
        == branch
    )
    assert (
        report.nonlinear_response_error_upper_bound
        == 2 * branch
        == (p0 + z * b0) ** 3 * c**4 / 6
    )
    assert 2 * branch < Q(7, 10**12)
    assert 2 * branch > p0**3 * c**4 / 6
    assert (129 * Q(1, 10000) + 388 * Q(1, 10**6)) / 50331648 + 2 * branch < Q(
        272, 10**12
    )
    # Initial phase zero does not license the old zero-form remainder: the
    # independently nonzero initial form drives its phase immediately.
    initial = (Q(0),) * 8
    generators = tuple((Q(1, 10000) if i == 2 else Q(0),) for i in range(8))
    pure_form = _report(
        initial_center=initial,
        initial_generators=generators,
        uncertainty_labels=("form_source",),
        preparation_relation_cost=0,
    )
    assert pure_form.initial_phase_upper_bound == 0
    assert pure_form.initial_form_upper_bound == Q(1, 10000)
    assert pure_form.nonlinear_response_error_upper_bound > 0


@pytest.mark.parametrize("z", [Q(0), Q(1, 8), Q(1, 2)])
def test_independent_positive_volterra_series_supports_nonzero_form_tube(z):
    # The two scalar Volterra equalities have phase majorant
    # P0*cosh(z)+B0*sinh(z). Positive terms and geometric tail bounds
    # independently check the rational tube and residual amplification.
    square = z**2
    even = sum((square**j / factorial(2 * j) for j in range(49)), Q(0))
    odd_ratio = sum((square**j / factorial(2 * j + 1) for j in range(49)), Q(0))
    cosh = _B(even, even + square**49 / factorial(98) / (1 - square / (99 * 100)))
    sinh_over_z = _B(
        odd_ratio,
        odd_ratio + square**49 / factorial(99) / (1 - square / (100 * 101)),
    )
    b0, p0, rational_growth = Q(3, 10000), Q(301, 10**6), 1 / (1 - square / 2)
    assert cosh.hi <= rational_growth
    assert sinh_over_z.hi <= rational_growth
    exact_majorant = p0 * cosh + b0 * z * sinh_over_z
    assert exact_majorant.hi <= (p0 + z * b0) * rational_growth
    if z:
        assert exact_majorant.lo > p0
    else:
        assert sinh_over_z == _B(1)
        assert exact_majorant == _B(p0)


@pytest.mark.parametrize("common_phase", [Q(0), Q(2, 7)])
def test_actual_coupled_field_departs_from_initial_tangent_relation(common_phase):
    delta = Q(1, 10000)
    sine = _sine(2 * delta)
    field = flow._held_quarter_sine_field(
        capacity=(1,) * 4,
        gamma=Q(1, 2),
        weighted_edges=((0, 1, 1), (1, 2, 1), (2, 3, 1)),
    )
    initial = (
        Q(0),
        Q(0),
        2 * delta,
        Q(0),
        common_phase,
        common_phase,
        common_phase - 2 * delta,
        common_phase,
    )
    actual = field(initial)
    expected_form = (
        _B(0),
        delta / 2 - sine / 8,
        -delta + sine / 4,
        delta / 2 - sine / 8,
    )
    expected_phase = (_B(0), _B(-delta / 4), _B(delta / 2), _B(-delta / 4))
    for row, expected in zip(actual, expected_form + expected_phase):
        _contains((row.lo, row.hi), expected, Q(1, 10**35))
    departure = actual[1] + actual[5]
    _contains((departure.lo, departure.hi), (2 * delta - sine) / 8, Q(1, 10**35))
    assert departure.lo > 0
    assert actual[5].hi < 0


def _paired_projection(report, arguments, *, on):
    matrix = report.on_generator if on else report.off_generator
    kernel = report.on_kernel if on else report.off_kernel
    time = report.heat_time
    by_label = {label: i for i, label in enumerate(arguments["coordinate_labels"])}
    columns = []
    for column in range(-1, len(arguments["uncertainty_labels"])):
        values = (
            arguments["initial_center"]
            if column == -1
            else tuple(row[column] for row in arguments["initial_generators"])
        )
        form = tuple(values[by_label[(i, "form")]] for i in range(4))
        phase = tuple(values[by_label[(i, "relative_phase")]] for i in range(4))
        forcing = tuple(
            form[i] + time * sum(matrix[i][j] * (form[j] + phase[j]) for j in range(4))
            for i in range(4)
        )
        columns.append(
            sum(
                (
                    _B(kernel.lower[i][j], kernel.upper[i][j])
                    * report.observation_weights[i]
                    * forcing[j]
                    for i in range(4)
                    for j in range(4)
                ),
                _B(0),
            )
        )
    return tuple(columns)


@pytest.mark.parametrize("mean_kind", ["arithmetic", "inverse_capacity"])
def test_actual_capacity_and_observation_use_full_signed_affine_projection(mean_kind):
    arguments = _arguments(
        capacity=(Q(1), Q(2), Q(3), Q(4)), elapsed_time=Q(1, 16), mean_kind=mean_kind
    )
    report = owner._prepared_sine_joint_information(**arguments)
    expected_weights = (
        (Q(1, 2), Q(1, 2), Q(0), Q(0))
        if mean_kind == "arithmetic"
        else (Q(2, 3), Q(1, 3), Q(0), Q(0))
    )
    assert report.observation_weights == expected_weights
    for matrix, edges in (
        (report.off_generator, ((1, 2), (2, 3))),
        (report.on_generator, ((0, 1), (1, 2), (2, 3))),
    ):
        for j in range(4):
            basis = tuple(_B(Q(i == j)) for i in range(4))
            expected = _heat_rows(basis, edges, arguments["capacity"])
            assert tuple(row.lo for row in expected) == tuple(
                matrix[i][j] for i in range(4)
            )
    off, on = _paired_projection(report, arguments, on=False), _paired_projection(
        report, arguments, on=True
    )
    for actual, left, right in zip(
        (report.tangent_center_bounds, *report.tangent_generator_coefficient_bounds),
        on,
        off,
    ):
        assert actual == ((left - right).lo, (left - right).hi)


def test_clock_row_and_latent_cube_equivalences_preserve_the_same_source(reports):
    baseline = reports["linked"]
    scaled = _report(capacity=(Q(3),) * 4, elapsed_time=Q(1, 12))
    fields = (
        "tangent_center_bounds",
        "linear_information_error_bounds",
        "nonlinear_response_error_upper_bound",
        "predicted_response",
        "forecast_error_upper_bound",
        "status",
        "response_sign",
    )
    for name in fields:
        assert getattr(scaled, name) == getattr(baseline, name)
    arguments = _arguments()
    row_order, column_order = (7, 2, 4, 1, 6, 0, 3, 5), (3, 1, 0, 2)
    signs = (-1, 1, -1, 1)
    rows = tuple(
        tuple(
            arguments["initial_generators"][i][j] * signs[k]
            for k, j in enumerate(column_order)
        )
        for i in row_order
    )
    permuted = _report(
        coordinate_labels=tuple(arguments["coordinate_labels"][i] for i in row_order),
        initial_center=tuple(arguments["initial_center"][i] for i in row_order),
        initial_generators=rows,
        uncertainty_labels=tuple(
            arguments["uncertainty_labels"][j] for j in column_order
        ),
    )
    for name in fields:
        assert getattr(permuted, name) == getattr(baseline, name)
    expected = tuple(
        tuple(
            value * signs[k]
            for value in (
                baseline.tangent_generator_coefficient_bounds[j]
                if signs[k] == 1
                else baseline.tangent_generator_coefficient_bounds[j][::-1]
            )
        )
        for k, j in enumerate(column_order)
    )
    assert permuted.tangent_generator_coefficient_bounds == expected
    assert permuted.coordinate_marginal_bounds == tuple(
        baseline.coordinate_marginal_bounds[i] for i in row_order
    )


def test_node_relabeling_preserves_physical_channel_association(reports):
    names = {0: "left", 1: "right", 2: "near", 3: "far"}
    args = _arguments()
    report = _report(
        nodes=("far", "right", "near", "left"),
        old_edges=(("near", "right"), ("far", "near")),
        contact=("right", "left"),
        retained_nodes=("right", "left"),
        observation_nodes=("left", "right"),
        phase_anchor="left",
        coordinate_labels=tuple(
            (names[node], channel) for node, channel in args["coordinate_labels"]
        ),
    )
    for name in (
        "tangent_center_bounds",
        "tangent_generator_coefficient_bounds",
        "linear_information_error_bounds",
        "nonlinear_response_error_upper_bound",
        "predicted_response",
        "forecast_error_upper_bound",
        "status",
        "response_sign",
    ):
        assert getattr(report, name) == getattr(reports["linked"], name)


def test_point_preparation_zero_state_and_final_reading_error_have_separate_meanings():
    point = _report(initial_generators=((),) * 8, uncertainty_labels=())
    assert (
        point.uncertainty_dimension == 0
        and point.tangent_generator_coefficient_bounds == ()
    )
    assert point.linear_information_error_bounds == (Q(0), Q(0))
    assert point.nonlinear_response_error_upper_bound > 0
    assert point.status == "certified"
    zero = _report(
        initial_center=(Q(0),) * 8,
        initial_generators=((),) * 8,
        uncertainty_labels=(),
        preparation_relation_cost=0,
    )
    assert zero.initial_form_upper_bound == zero.initial_phase_upper_bound == 0
    assert zero.forecast_response_bounds == (Q(0), Q(0))
    assert zero.response_sign == "zero"
    clean, noisy = _report(), _report(reading_error_bound=Q(1, 10**8))
    assert noisy.forecast_error_upper_bound == clean.forecast_error_upper_bound + Q(
        1, 10**8
    )
    assert noisy.information_error_lower_bound == clean.information_error_lower_bound
    assert noisy.status == noisy.response_sign == "unresolved"


def test_known_retained_coordinates_enter_center_and_cost_is_only_metadata():
    delta = Q(1, 10000)
    args = _arguments(
        initial_center=(delta, -delta, 2 * delta, 0, 0, delta / 2, -2 * delta, 0),
        initial_generators=((),) * 8,
        uncertainty_labels=(),
        preparation_relation_cost=0,
    )
    report = owner._prepared_sine_joint_information(**args)
    expected, latent = _projected_oracle(args)
    assert latent == ()
    _contains(report.tangent_center_bounds, expected)
    assert (
        report.initial_form_upper_bound == report.initial_phase_upper_bound == 2 * delta
    )
    declared_cost = owner._prepared_sine_joint_information(
        **{**args, "preparation_relation_cost": 999}
    )
    assert declared_cost.preparation_relation_cost == 999
    for name in (
        "tangent_center_bounds",
        "linear_information_error_bounds",
        "nonlinear_response_error_upper_bound",
        "forecast_error_upper_bound",
        "forecast_response_bounds",
        "status",
    ):
        assert getattr(report, name) == getattr(declared_cost, name)


def test_nonlinear_uncertainty_can_prevent_transfer_of_a_tangent_obstruction():
    args = _arguments()
    factor = Q(100)
    report = _report(
        initial_center=tuple(factor * x for x in args["initial_center"]),
        initial_generators=tuple(
            tuple(factor * x for x in row) for row in args["initial_generators"]
        ),
    )
    assert report.initial_phase_upper_bound < Q(1, 4)
    assert report.linear_information_error_bounds[0] > report.tolerance
    assert (
        report.nonlinear_response_error_upper_bound
        > report.linear_information_error_bounds[1]
    )
    assert report.information_error_lower_bound == 0
    assert report.status == "unresolved"


@pytest.mark.parametrize("zero", ["time", "contact"])
def test_identical_branch_zero_is_matched_after_all_admission(monkeypatch, zero):
    def forbidden(*args, **kwargs):
        pytest.fail("identical branches must not evaluate a kernel")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    report = _report(
        **({"elapsed_time": Q(0)} if zero == "time" else {"contact_weight": Q(0)})
    )
    assert report.off_kernel is report.on_kernel is None
    assert report.tangent_center_bounds == (Q(0), Q(0))
    assert report.tangent_generator_coefficient_bounds == ((Q(0), Q(0)),) * 4
    assert (
        report.nonlinear_response_error_upper_bound
        == report.forecast_error_upper_bound
        == 0
    )
    assert report.status == "certified" and report.response_sign == "zero"
    if zero == "contact":
        assert report.off_nonlinear_error_upper_bound > 0
        assert report.on_nonlinear_error_upper_bound > 0


def test_phase_chart_and_latent_dimension_closed_boundaries():
    args = _arguments()
    center = tuple(Q(1, 4) if i == 7 else Q(0) for i in range(8))
    report = _report(
        initial_center=center,
        initial_generators=((),) * 8,
        uncertainty_labels=(),
        elapsed_time=Q(0),
    )
    assert report.initial_phase_upper_bound == Q(1, 4)
    full_columns = tuple(row + (Q(0),) * 44 for row in args["initial_generators"])
    report = _report(
        initial_generators=full_columns,
        uncertainty_labels=tuple(range(48)),
        elapsed_time=Q(0),
    )
    assert report.uncertainty_dimension == 48
    boundary = _report(elapsed_time=Q(1))
    assert boundary.on_phase_growth_bound == boundary.off_phase_growth_bound == Q(8, 7)


def _invalid_source_cases():
    args = _arguments()
    labels, center, rows = (
        args["coordinate_labels"],
        args["initial_center"],
        args["initial_generators"],
    )

    def row_change(index, row):
        return tuple(row if i == index else current for i, current in enumerate(rows))

    return [
        {"nodes": (0, 1, 1, 3)},
        {"nodes": {0, 1, 2, 3}},
        {"nodes": repeat(0)},
        {"old_edges": ((1, 2), (2, 1))},
        {"old_conductances": (True, Q(1))},
        {"capacity": (Q(1), Q(0), Q(1), Q(1))},
        {"contact_weight": Q(-1)},
        {"retained_nodes": (0, 1, 2, 3)},
        {"retained_nodes": (0, 2)},
        {"observation_nodes": (2,)},
        {"phase_anchor": 2},
        {"phase_anchor": 9},
        {"coordinate_labels": labels[:-1]},
        {"coordinate_labels": labels[:-1] + (labels[0],)},
        {"coordinate_labels": labels[:-1] + ((9, "relative_phase"),)},
        {"coordinate_labels": labels[:-1] + ((3, "phase"),)},
        {"coordinate_labels": repeat((0, "form"))},
        {"coordinate_labels": set(labels)},
        {"initial_center": center[:-1]},
        {"initial_center": (True,) + center[1:]},
        {"initial_center": center[:2] + (float("nan"),) + center[3:]},
        {"initial_generators": rows[:-1]},
        {"initial_generators": repeat(rows[0])},
        {"initial_generators": row_change(2, (Q(0),) * 3)},
        {"initial_generators": row_change(2, repeat(Q(0)))},
        {"initial_generators": row_change(2, (True, Q(0), Q(0), Q(0)))},
        {"initial_generators": row_change(2, (float("inf"), Q(0), Q(0), Q(0)))},
        {"initial_generators": row_change(1, (Q(1, 10**100), Q(0), Q(0), Q(0)))},
        {"initial_generators": row_change(5, (Q(1, 10**100), Q(0), Q(0), Q(0)))},
        {"initial_center": center[:4] + (Q(1, 10**100),) + center[5:]},
        {"initial_generators": row_change(4, (Q(1, 10**100), Q(0), Q(0), Q(0)))},
        {"initial_center": center[:7] + (Q(1, 4),)},
        {"initial_generators": row_change(7, (Q(1, 4), Q(1, 10**100), Q(0), Q(0)))},
        {"uncertainty_labels": ("u", "u", "w2", "w3")},
        {"uncertainty_labels": ("u", "v", "w2")},
        {"uncertainty_labels": {"u", "v", "w2", "w3"}},
        {"uncertainty_labels": repeat("u")},
        {"uncertainty_labels": tuple(range(49))},
        {"preparation_relation_cost": True},
        {"preparation_relation_cost": -1},
        {"preparation_relation_cost": 2.0},
        {"preparation_relation_cost": Q(2)},
        {"elapsed_time": True},
        {"elapsed_time": Q(-1)},
        {"elapsed_time": Q(1) + Q(1, 10**100)},
        {"mean_kind": "arbitrary"},
        {"reading_error_bound": Q(-1)},
        {"tolerance": True},
    ]


@pytest.mark.parametrize("changes", _invalid_source_cases())
def test_all_primitive_associations_and_bounds_reject_before_matrix_work(
    monkeypatch, changes
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid affine preparation reached matrix work")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    monkeypatch.setattr(owner, "exact_matrix_product", forbidden)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        _report(**changes)
    if "elapsed_time" not in changes and "contact_weight" not in changes:
        with pytest.raises((ValueError, TypeError, OverflowError)):
            _report(**{**changes, "contact_weight": Q(0)})


def test_second_branch_domains_and_phase_loadings_reject_before_shortcuts(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("later admission reached an early matrix kernel")

    monkeypatch.setattr(owner, "metzler_exponential_bounds", forbidden)
    monkeypatch.setattr(owner, "exact_matrix_product", forbidden)
    with pytest.raises(ValueError):
        _report(contact_weight=Q(4), elapsed_time=Q(1, 2))
    with pytest.raises(ValueError):
        _report(contact_weight=Q(0), elapsed_time=Q(2))
    args = _arguments()
    rows = list(args["initial_generators"])
    rows[7] = (Q(1, 4), Q(1, 10**100), Q(0), Q(0))
    with pytest.raises(ValueError):
        _report(elapsed_time=Q(0), initial_generators=rows)


def test_report_is_frozen_and_ordered_source_inputs_normalize():
    args = _arguments()
    report = _report(
        nodes=[0, 1, 2, 3],
        coordinate_labels=[list(label) for label in args["coordinate_labels"]],
        initial_generators=[list(row) for row in args["initial_generators"]],
    )
    assert report.coordinate_labels == args["coordinate_labels"]
    assert report.initial_generators == args["initial_generators"]
    with pytest.raises(FrozenInstanceError):
        report.predicted_response = Q(1)
