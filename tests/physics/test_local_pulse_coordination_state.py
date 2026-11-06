"""Six local-state controls on the unchanged conservative doubled C5 law.

Sources are constructed points, not acquired preparations. First-order field
jets are directional derivatives at those points, not trajectory steps. The
controls establish information requirements, not synchronization or a clock.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import pytest
import sympy as sp

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._interval_taylor import cos as jet_cos
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics._sine_flow import _held_sine_neighbor_field

ETA = Q(1, 1024)
PROFILE = (1, 2, 3, 4, 2)
EDGES = tuple(
    sorted(
        {
            tuple(sorted((2 * i + s, 2 * ((i + 1) % 5) + t)))
            for i in range(5)
            for s, t in product((0, 1), repeat=2)
        }
    )
)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, expected):
    assert _mp(interval.lo) <= expected <= _mp(interval.hi)
    assert interval.width < Q(1, 2**65)


def _zero(interval):
    assert interval.contains(0) and interval.width < Q(1, 2**65)


def _chart(kind, sign):
    offset = tuple(sign * ETA * (Q(1 if i == 0 else 0) - Q(1, 5)) for i in range(5))
    return {
        "X": offset if kind == "form" else (Q(0),) * 5,
        "h": offset if kind == "phase" else (Q(0),) * 5,
        "u": tuple(ETA * value for value in PROFILE),
        "delta": (ETA,) * 5,
    }


def _box(chart):
    forms = tuple(
        I(chart["X"][i] + sign * chart["u"][i]) for i in range(5) for sign in (1, -1)
    )
    phases = tuple(
        pi_interval() * Q(2 * i, 5) + chart["h"][i] + sign * ETA
        for i in range(5)
        for sign in (1, -1)
    )
    return forms + phases


def _pairs(values):
    return tuple(
        tuple(
            (values[start + 2 * i] + sign * values[start + 2 * i + 1]) / 2
            for i in range(5)
        )
        for start, sign in ((0, 1), (10, 1), (0, -1), (10, -1))
    )


def _observed_rates(values, field):
    _, _, y, delta = _pairs(values)  # beta=1 in both declared witnesses.
    _, _, dy, dd = _pairs(field)
    radius = tuple(d * d + v * v for d, v in zip(delta, y))
    radial = tuple(
        2 * (d * ddot + v * vdot) for d, v, ddot, vdot in zip(delta, y, dd, dy)
    )
    angular = tuple(
        (v * ddot - d * vdot) / r
        for d, v, ddot, vdot, r in zip(delta, y, dd, dy, radius)
    )
    return radius, radial, angular


def _restoring(values):
    _, theta, _, delta = _pairs(values)
    cosine = jet_cos if isinstance(values[0], Jet) else cos
    return tuple(
        sum(
            cosine(delta[j]) * cosine(theta[j] - theta[i])
            for j in ((i - 1) % 5, (i + 1) % 5)
        )
        / 2
        for i in range(5)
    )


def _fine_oracle(chart):
    """Independent high-precision edge sums in original t, all twenty rows."""
    forms = tuple(
        _mp(chart["X"][i] + sign * chart["u"][i]) for i in range(5) for sign in (1, -1)
    )
    phases = tuple(
        2 * mp.pi * i / 5 + _mp(chart["h"][i] + sign * ETA)
        for i in range(5)
        for sign in (1, -1)
    )
    form_rate, phase_rate = [mp.mpf(0)] * 10, [mp.mpf(0)] * 10
    energy = mp.mpf(0)
    for i, j in EDGES:
        current, gap = mp.sin(phases[j] - phases[i]), forms[i] - forms[j]
        form_rate[i] += current / (4 * mp.pi)
        form_rate[j] -= current / (4 * mp.pi)
        phase_rate[i] += gap / (4 * mp.pi)
        phase_rate[j] -= gap / (4 * mp.pi)
        energy += gap**2 / 2 + 1 - mp.cos(phases[j] - phases[i])
    return forms + phases, tuple(form_rate + phase_rate), energy


@pytest.fixture(scope="module")
def witnesses():
    rows = tuple(
        tuple((j, 1, Q(1)) for j in range(10) if tuple(sorted((i, j))) in EDGES)
        for i in range(10)
    )
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    field = _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, rows)
    points = {}
    for kind, sign in product(("phase", "form"), (1, -1)):
        chart = _chart(kind, sign)
        box = _box(chart)
        points[kind, sign] = (chart, box, field(box))
    return field, points


def test_projective_observation_reconstructs_internal_pairs_except_the_tip():
    d, y = sp.symbols("delta y", real=True)
    beta = sp.symbols("beta", positive=True)
    radius = d**2 + y**2
    cosine, sine = (d**2 - y**2) / radius, -2 * d * y / radius
    assert sp.simplify(cosine**2 + sine**2 - 1) == 0
    assert sp.simplify(radius * (1 + cosine) / 2 - d**2) == 0
    assert sp.simplify(radius * (1 - cosine) / 2 - y**2) == 0
    assert sp.simplify(-radius * sine / 2 - d * y) == 0
    swap = {d: -d, y: -y}
    for expression in (
        radius,
        cosine,
        sine,
        sp.cos(d),
        beta * y**2,
        sp.sqrt(beta) * y * sp.sin(d),
    ):
        assert sp.simplify(expression.subs(swap, simultaneous=True) - expression) == 0
    assert sp.simplify(sine.subs(y, -y) + sine) == 0
    # Axis points are genuine nontips; at the tip the phase has 0/0, not angle0.
    assert (radius * cosine).subs({d: 0, y: 0}) == 0
    assert (radius * sine).subs({d: 0, y: 0}) == 0
    assert radius.subs({d: 0, y: 0}) == 0
    for first, second in ((Q(0), ETA), (ETA, Q(0)), (ETA, 2 * ETA)):
        assert first**2 + second**2 > 0
        outer = sp.Matrix((first, second)) * sp.Matrix((first, second)).T
        assert outer == sp.Matrix((-first, -second)) * sp.Matrix((-first, -second)).T
        assert outer.rank() == 1
    # u/sqrt(beta), not u, supplies the dimensionless internal form coordinate.
    assert radius.subs(
        {d: sp.Rational(1, 8), y: sp.Rational(1, 4) / sp.sqrt(4)}
    ) == sp.Rational(1, 32)


def test_observation_rates_are_pushforwards_of_all_twenty_original_time_rows(witnesses):
    d, dl, dr, gl, gr = sp.symbols("d dl dr gl gr", real=True)
    x, xl, xr, u, ul, ur = sp.symbols("X Xl Xr u ul ur", real=True)
    fine_form, fine_phase = [], []
    for sign in (1, -1):
        fine_form.append(
            sum(
                sp.sin(g + other * dj - sign * d)
                for g, dj in ((gl, dl), (gr, dr))
                for other in (1, -1)
            )
            / (4 * sp.pi)
        )
        fine_phase.append(
            sum(
                x + sign * u - xj - other * uj
                for xj, uj in ((xl, ul), (xr, ur))
                for other in (1, -1)
            )
            / (4 * sp.pi)
        )
    expected = (
        sp.cos(d) * (sp.cos(dl) * sp.sin(gl) + sp.cos(dr) * sp.sin(gr)) / (2 * sp.pi),
        -sp.sin(d) * (sp.cos(dl) * sp.cos(gl) + sp.cos(dr) * sp.cos(gr)) / (2 * sp.pi),
    )
    for sign, target in zip((1, -1), expected):
        assert (
            sp.simplify(
                sp.expand_trig((fine_form[0] + sign * fine_form[1]) / 2 - target)
            )
            == 0
        )
    assert (
        sp.simplify(
            (fine_phase[0] + fine_phase[1]) / 2 - (2 * x - xl - xr) / (2 * sp.pi)
        )
        == 0
    )
    assert sp.simplify((fine_phase[0] - fine_phase[1]) / 2 - u / sp.pi) == 0
    y, restoring, omega = sp.symbols("y F Omega", real=True)
    radial = sp.diff(d**2 + y**2, d) * omega * y - sp.diff(
        d**2 + y**2, y
    ) * omega * restoring * sp.sin(d)
    assert sp.simplify(radial - 2 * omega * y * (d - restoring * sp.sin(d))) == 0
    assert (
        sp.simplify(
            (y * omega * y + d * omega * restoring * sp.sin(d)) / (d**2 + y**2)
            - omega * (y**2 + restoring * d * sp.sin(d)) / (d**2 + y**2)
        )
        == 0
    )

    _, points = witnesses
    with mp.workdps(90):
        for chart, box, rates in points.values():
            values, oracle, _ = _fine_oracle(chart)
            assert len(rates) == len(oracle) == 20
            for actual, expected_rate in zip(rates, oracle):
                _contains(actual, expected_rate)
            radius, radial, angular = _observed_rates(box, rates)
            for i in range(5):
                y0 = (values[2 * i] - values[2 * i + 1]) / 2
                d0 = (values[10 + 2 * i] - values[11 + 2 * i]) / 2
                dy = (oracle[2 * i] - oracle[2 * i + 1]) / 2
                dd = (oracle[10 + 2 * i] - oracle[11 + 2 * i]) / 2
                _contains(radius[i], d0**2 + y0**2)
                _contains(radial[i], 2 * (d0 * dd + y0 * dy))
                _contains(angular[i], (y0 * dd - d0 * dy) / (d0**2 + y0**2))


def test_equal_local_observations_and_matched_active_sources_have_different_timing(
    witnesses,
):
    _, points = witnesses
    pi, eta = pi_interval(), I(ETA)
    alpha, radius, ceiling = 2 * pi / 5, Q(1, 10), Q(1, 1000)
    curvature = cos(alpha + sqrt(I(2)) * radius)
    gap = 5 - sqrt(I(5))
    assert curvature.lo > Q(1, 10) and gap.lo > Q(11, 4)
    assert (gap * curvature * radius**2 / 2).lo > Q(11, 8000) > ceiling
    assert Q(398, 5) * ETA**2 < radius**2
    assert 160 * ETA**2 < ceiling
    assert (alpha + 3 * eta).hi < (pi / 2).lo
    plus, minus = points["phase", 1], points["phase", -1]
    assert plus[0]["u"] == minus[0]["u"] and plus[0]["delta"] == minus[0]["delta"]
    assert len(set(PROFILE)) > 1 and PROFILE[2] != PROFILE[3]
    for chart, _, _ in (plus, minus):
        assert sum(chart["X"]) == sum(chart["h"]) == 0
        norm = 2 * sum(
            value**2 for key in ("X", "h", "u", "delta") for value in chart[key]
        )
        assert norm == Q(398, 5) * ETA**2
        assert all(d**2 + u**2 > 0 for d, u in zip(chart["delta"], chart["u"]))

    def increments(chart):
        return sorted(
            chart["h"][(i + 1) % 5] - chart["h"][i] + (t - s) * ETA
            for i in range(5)
            for s, t in product((1, -1), repeat=2)
        )

    assert increments(plus[0]) == increments(minus[0])
    a, h = sp.symbols("alpha h", real=True)
    assert (
        sp.simplify(
            sp.expand_trig(sp.cos(a - h) + sp.cos(a + h) - 2 * sp.cos(a) * sp.cos(h))
        )
        == 0
    )
    excess = 136 * eta**2 + 4 * cos(alpha) * (5 - cos(eta) ** 2 * (3 + 2 * cos(eta)))
    assert 0 < excess.lo and excess.hi < 160 * ETA**2
    with mp.workdps(90):
        ee = _mp(ETA)
        expected_energy = (
            136 * ee**2
            + 20
            - 4 * mp.cos(ee) ** 2 * mp.cos(2 * mp.pi / 5) * (3 + 2 * mp.cos(ee))
        )
        for sign, (chart, box, rates) in ((1, plus), (-1, minus)):
            _, _, actual_energy = _fine_oracle(chart)
            assert abs(actual_energy - expected_energy) < mp.mpf("1e-80")
            angular = _observed_rates(box, rates)[2]
            relative = angular[1] - angular[4]
            expected = (
                sign
                * mp.cos(ee)
                * mp.sin(2 * mp.pi / 5)
                * mp.sin(ee) ** 2
                / (5 * mp.pi * ee)
            )
            _contains(relative, expected)
            assert relative.lo > 0 if sign == 1 else relative.hi < 0


def test_instantaneous_restoring_factors_do_not_close_without_collective_form(
    witnesses,
):
    field, points = witnesses
    excess = 140 * I(ETA) ** 2 + 20 * cos(2 * pi_interval() / 5) * sin(I(ETA)) ** 2
    assert 0 < excess.lo and excess.hi < 160 * ETA**2 < Q(1, 1000)
    current, restoring_derivative, relative_acceleration = {}, {}, {}
    for sign in (1, -1):
        chart, box, rates = points["form", sign]
        assert sum(chart["X"]) == sum(chart["h"]) == 0
        assert (
            2
            * sum(value**2 for key in ("X", "h", "u", "delta") for value in chart[key])
            == Q(398, 5) * ETA**2
        )
        jets = tuple(Jet((value, rate)) for value, rate in zip(box, rates))
        field_jet = field(jets)
        angular = _observed_rates(jets, field_jet)[2]
        restoring = _restoring(jets)
        current[sign] = tuple(value.coeffs[0] for value in angular)
        restoring_derivative[sign] = restoring[1].coeffs[1] - restoring[4].coeffs[1]
        relative_acceleration[sign] = angular[1].coeffs[1] - angular[4].coeffs[1]
        _zero(angular[1].coeffs[0] - angular[4].coeffs[0])
    for a, b in zip(current[1], current[-1]):
        _zero(a - b)
    # This is exact equality of the consumed phase boxes, not a conclusion
    # inferred from overlapping numerical intervals. F never consumes X.
    assert _restoring(points["form", 1][1]) == _restoring(points["form", -1][1])
    for a, b in zip(
        _restoring(points["form", 1][1]), _restoring(points["form", -1][1])
    ):
        _zero(a - b)
    with mp.workdps(90):
        eta, alpha = _mp(ETA), 2 * mp.pi / 5
        for sign in (1, -1):
            expected_f = (
                eta * mp.sin(eta) * mp.cos(alpha) / (2 * mp.pi)
                + sign * eta * mp.cos(eta) * mp.sin(alpha) / mp.pi
            )
            expected_acceleration = mp.sin(eta) ** 2 * mp.cos(alpha) / (
                10 * mp.pi**2
            ) + sign * mp.sin(eta) * mp.cos(eta) * mp.sin(alpha) / (5 * mp.pi**2)
            _contains(restoring_derivative[sign], expected_f)
            _contains(relative_acceleration[sign], expected_acceleration)
            assert (
                relative_acceleration[sign].lo > 0
                if sign == 1
                else relative_acceleration[sign].hi < 0
            )
            _, _, energy = _fine_oracle(points["form", sign][0])
            expected_energy = 140 * eta**2 + 20 * (1 - mp.cos(eta) ** 2 * mp.cos(alpha))
            assert abs(energy - expected_energy) < mp.mpf("1e-80")
        expected_difference = (
            2 * mp.sin(eta) * mp.cos(eta) * mp.sin(alpha) / (5 * mp.pi**2)
        )
        _contains(
            relative_acceleration[1] - relative_acceleration[-1], expected_difference
        )


def test_existing_unordered_state_closes_all_consumed_rows_and_constraints(witnesses):
    R, U, C, restoring, c = sp.symbols("R U Q A c", real=True)
    invariant = C**2 - U * (1 - R**2)
    quotient_rates = (-c * C, -2 * restoring * C, -restoring * (1 - R**2) + c * U * R)
    derivative = sum(
        sp.diff(invariant, variable) * rate
        for variable, rate in zip((R, U, C), quotient_rates)
    )
    assert sp.expand(derivative) == 0
    d, u = sp.symbols("delta u", real=True)
    assert sp.simplify(invariant.subs({R: sp.cos(d), U: u**2, C: u * sp.sin(d)})) == 0
    _, points = witnesses
    for _, box, field in points.values():
        X, theta, internal, delta = _pairs(box)
        dX, dtheta, du, ddelta = _pairs(field)
        factors = _restoring(box)
        resultant, squared, product_state = (
            tuple(cos(value) for value in delta),
            tuple(value**2 for value in internal),
            tuple(u0 * sin(d0) for u0, d0 in zip(internal, delta)),
        )
        for i in range(5):
            neighbors = ((i - 1) % 5, (i + 1) % 5)
            expected_x = sum(
                resultant[i] * resultant[j] * sin(theta[j] - theta[i])
                for j in neighbors
            ) / (2 * pi_interval())
            expected_theta = sum(X[i] - X[j] for j in neighbors) / (2 * pi_interval())
            _zero(dX[i] - expected_x)
            _zero(dtheta[i] - expected_theta)
            _zero(-sin(delta[i]) * ddelta[i] + product_state[i] / pi_interval())
            _zero(
                2 * internal[i] * du[i]
                + 2 * factors[i] * product_state[i] / pi_interval()
            )
            _zero(
                du[i] * sin(delta[i])
                + internal[i] * cos(delta[i]) * ddelta[i]
                - (-factors[i] * (1 - resultant[i] ** 2) + squared[i] * resultant[i])
                / pi_interval()
            )
            assert resultant[i].lo > 0
    # The same internal observation determines R,U,Q, while X and relative
    # Theta supply their coupling. This is sufficient, not proved minimal.
    for profile in PROFILE:
        dd, yy = sp.Rational(ETA), sp.Rational(ETA * profile)
        assert sp.cos(dd) == sp.cos(-dd)
        assert yy * sp.sin(dd) == (-yy) * sp.sin(-dd)
        assert yy**2 == (-yy) ** 2


def test_relative_phase_observer_keeps_rate_clock_and_cycle_count_distinctions(
    witnesses,
):
    d, y, restoring, omega = sp.symbols("delta y F Omega", real=True)
    radius = d**2 + y**2
    zeta = (d - sp.I * y) ** 2 / radius
    angular = omega * (y**2 + restoring * d * sp.sin(d)) / radius
    dzeta = sp.diff(zeta, d) * omega * y - sp.diff(
        zeta, y
    ) * omega * restoring * sp.sin(d)
    assert sp.simplify(dzeta - 2 * sp.I * angular * zeta) == 0
    chi1, chi4, w1, w4 = sp.symbols("chi1 chi4 omega1 omega4", real=True)
    relative = sp.exp(2 * sp.I * chi1) * sp.conjugate(sp.exp(2 * sp.I * chi4))
    derivative = sp.diff(relative, chi1) * w1 + sp.diff(relative, chi4) * w4
    assert sp.simplify(derivative - 2 * sp.I * (w1 - w4) * relative) == 0
    _, points = witnesses
    rates = {
        sign: _observed_rates(points["phase", sign][1], points["phase", sign][2])[2]
        for sign in (1, -1)
    }
    _zero(rates[1][0] - rates[-1][0])
    assert rates[1][0].lo > 0 and rates[-1][0].lo > 0
    assert (rates[1][1] - rates[1][4]).lo > 0
    assert (rates[-1][1] - rates[-1][4]).hi < 0
    # A common positive speed multiplier must equal1 from component0, then
    # cannot turn one relative timing direction into the other.
    speed, scale = sp.symbols("speed scale", positive=True)
    assert sp.solve(sp.Eq(speed, scale * speed), scale) == [1]
    # At zero relative rate, acceleration relative to an advancing local
    # reading also discards every positive common reparameterization factor.
    difference, acceleration = sp.symbols("dw da", real=True)
    clock_rate = sp.symbols("gamma", positive=True)
    clock_derivative = sp.symbols("gamma_dot", real=True)
    transformed_acceleration = (
        acceleration * clock_rate - difference * clock_derivative
    ) / clock_rate**3
    invariant_acceleration = transformed_acceleration / (speed / clock_rate) ** 2
    assert (
        sp.simplify(
            invariant_acceleration.subs(difference, 0) - acceleration / speed**2
        )
        == 0
    )
    eta, alpha, pi = I(ETA), 2 * pi_interval() / 5, pi_interval()
    for sign in (1, -1):
        state = points["form", sign]
        local_speed = _observed_rates(state[1], state[2])[2][0]
        assert local_speed.lo > 0
        relative_acceleration = sin(eta) ** 2 * cos(alpha) / (
            10 * pi**2
        ) + sign * sin(eta) * cos(eta) * sin(alpha) / (5 * pi**2)
        normalized = relative_acceleration / local_speed**2
        assert normalized.lo > 0 if sign == 1 else normalized.hi < 0
    local = (sp.S(1) - 2 * sp.I) ** 2 / 5
    assert sp.simplify(local * sp.conjugate(local)) == 1  # Z_14 initially1.
    phase = sp.symbols("chi", real=True)
    for turns in (-2, 1, 3):
        assert (
            sp.simplify(
                sp.exp(2 * sp.I * (phase + turns * sp.pi)) - sp.exp(2 * sp.I * phase)
            )
            == 0
        )
        assert phase + turns * sp.pi != phase
    # Equal instantaneous projective phases retain neither these lifted
    # counts nor a history of synchronization; no artificial flow is supplied.
