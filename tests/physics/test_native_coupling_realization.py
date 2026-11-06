"""Six exact controls for capacity flow versus the finite UM component.

The scope is the ideal native C5 lock, immutable all-target capacity proposals,
and the complete observed capacity vector. No public stage, selector, runtime,
trajectory or stopping-rule fit is executed. Public active UM admits 0<s<=1;
s=0 appears only as an explicitly named analytic boundary.
"""

import pytest
import sympy as sp


def _clean(expression):
    return sp.simplify(sp.expand(expression))


def _same(left, right):
    assert left.shape == right.shape
    assert all(_clean(a - b) == 0 for a, b in zip(left, right))


@pytest.fixture(scope="module")
def comparison():
    incidence = sp.zeros(5)
    neighbors = [[] for _ in range(5)]
    for column in range(5):
        left, right = column, (column + 1) % 5
        incidence[left, column], incidence[right, column] = 1, -1
        neighbors[left].append(right)
        neighbors[right].append(left)
    laplacian = incidence * incidence.T / 2
    nu = sp.Matrix(sp.symbols("nu0:5", positive=True))
    s, kappa = sp.symbols("s kappa", positive=True)
    elapsed = sp.symbols("elapsed", nonnegative=True)
    # Build the snapshot target proposals separately from the incidence law.
    snapshot = sp.Matrix(
        [
            nu[i] + s * (sum(nu[j] for j in row) / len(row) - nu[i])
            for i, row in enumerate(neighbors)
        ]
    )
    event = snapshot.jacobian(nu)
    low, high = (5 - sp.sqrt(5)) / 4, (5 + sp.sqrt(5)) / 4
    identity = sp.eye(5)
    uniform = sp.ones(5) / 5
    low_projector = (
        laplacian * (high * identity - laplacian) / (low * (high - low))
    ).applyfunc(_clean)
    high_projector = (
        laplacian * (laplacian - low * identity) / (high * (high - low))
    ).applyfunc(_clean)
    flow = (
        uniform
        + sp.exp(-kappa * elapsed * low) * low_projector
        + sp.exp(-kappa * elapsed * high) * high_projector
    )
    return {
        "L": laplacian,
        "nu": nu,
        "s": s,
        "kappa": kappa,
        "elapsed": elapsed,
        "snapshot": snapshot,
        "event": event,
        "low": low,
        "high": high,
        "P0": uniform,
        "P_low": low_projector,
        "P_high": high_projector,
        "flow": flow,
    }


def test_snapshot_capacity_map_and_native_lock(comparison):
    law = comparison
    laplacian, s, nu, event = (law[key] for key in ("L", "s", "nu", "event"))
    _same(law["snapshot"], event * nu)
    _same(event, sp.eye(5) - s * laplacian)
    _same(sp.ones(1, 5) * event, sp.ones(1, 5))
    # s=1/(1+r), r>=0 covers exactly the public active factor interval (0,1].
    r = sp.symbols("factor_coordinate", nonnegative=True)
    admitted_event = event.subs(s, 1 / (1 + r))
    assert all(_clean(value).is_nonnegative is True for value in admitted_event)
    _same(admitted_event * sp.ones(5, 1), sp.ones(5, 1))
    origin, e, w, beta = sp.symbols("origin e w beta", positive=True)
    alpha = 2 * sp.pi / 5
    re = _clean(sp.cos(alpha) + sp.cos(-alpha))
    im = _clean(sp.sin(alpha) + sp.sin(-alpha))
    assert re.is_positive is True and im == 0
    assert (sp.pi / 2 - alpha).is_positive is True
    phase_displacement = sp.atan(im / re)
    native_source = phase_displacement / sp.pi
    native_metric = sp.pi * sp.sqrt(re**2 + im**2) * sp.sinc(phase_displacement)
    gradient = 2 * laplacian * (origin * sp.ones(5, 1))
    pressure = -e * gradient / 2 + w * native_source * sp.ones(5, 1)
    form_rate = sp.diag(*nu) * pressure
    phase_rate = w * sp.diag(*nu) * gradient / (beta * native_metric)
    _same(form_rate, sp.zeros(5, 1))
    _same(phase_rate, sp.zeros(5, 1))
    push, relief = sp.symbols("phase_push pressure_relief", positive=True)
    # Any admitted positive target-only phase push has zero displacement here.
    assert push * phase_displacement == 0
    _same((1 - relief) * pressure, sp.zeros(5, 1))
    # These identities concern structural channels, not history or occurrence.


def test_complete_c5_spectral_projectors(comparison):
    law = comparison
    laplacian, kappa, elapsed = (law[key] for key in ("L", "kappa", "elapsed"))
    projectors = (law["P0"], law["P_low"], law["P_high"])
    eigenvalues = (sp.S.Zero, law["low"], law["high"])
    _same(sum(projectors, sp.zeros(5)), sp.eye(5))
    for index, (projector, eigenvalue, rank) in enumerate(
        zip(projectors, eigenvalues, (1, 2, 2))
    ):
        _same(projector.T, projector)
        _same(projector * projector, projector)
        _same(laplacian * projector, eigenvalue * projector)
        assert _clean(sp.trace(projector) - rank) == 0
        for other_index, other in enumerate(projectors):
            if index != other_index:
                _same(projector * other, sp.zeros(5))
    assert law["low"].is_positive is True
    assert _clean(law["high"] - law["low"]).is_positive is True
    # This verifies the complete semigroup through its IVP, without invoking
    # a numerical solver or guessing one occupied spatial mode.
    _same(law["flow"].subs(elapsed, 0), sp.eye(5))
    _same(law["flow"].diff(elapsed), -kappa * laplacian * law["flow"])
    for projector, eigenvalue in zip(projectors, eigenvalues):
        _same(law["event"] * projector, (1 - law["s"] * eigenvalue) * projector)


def test_single_band_stopping_and_admitted_boundaries(comparison):
    law = comparison
    s, kappa, elapsed = (law[key] for key in ("s", "kappa", "elapsed"))
    low, high = law["low"], law["high"]
    high_threshold = _clean(1 / high)
    assert _clean(high_threshold - (1 - sp.sqrt(5) / 5)) == 0
    assert high_threshold.is_positive is True
    assert _clean(1 - high_threshold).is_positive is True
    assert _clean(1 / low - 1).is_positive is True
    assert _clean(1 - low).is_positive is True
    positive = sp.symbols("positive", positive=True)
    for eigenvalue in (low, high):
        # Positive=odds of attenuation gives 0<gain<1 without asking the CAS
        # to establish a logarithmic inequality from an undeclared domain.
        factor = positive / (eigenvalue * (1 + positive))
        duration = sp.log(1 + positive) / (kappa * eigenvalue)
        gain = _clean(1 - factor * eigenvalue)
        assert _clean(gain - 1 / (1 + positive)) == 0
        assert _clean(sp.exp(-kappa * eigenvalue * duration) - gain) == 0
        assert (
            _clean(
                sp.diff(duration, positive) - 1 / (kappa * eigenvalue * (1 + positive))
            )
            == 0
        )
        assert duration.subs(positive, 0) == 0  # Analytic boundary only.
        assert sp.exp(-kappa * elapsed * eigenvalue).is_positive is True
    assert _clean(1 - high_threshold * high) == 0
    # Above the high threshold the contrast reverses; the full capacity map
    # remains positive by the independently checked stochastic representation.
    assert _clean(1 - (high_threshold + positive) * high).is_negative is True
    assert sp.limit(sp.exp(-kappa * elapsed * high), elapsed, sp.oo) == 0
    _same(law["event"] * law["P0"], law["P0"])
    _same(law["flow"] * law["P0"], law["P0"])
    _same(law["event"].subs(s, 0), sp.eye(5))  # Not active public s=0.


def test_two_active_bands_obstruct_any_stopping_time(comparison):
    law = comparison
    s, low, high = law["s"], law["low"], law["high"]
    spectral = sp.symbols("spectral", positive=True)
    integration_variable = sp.symbols("integration_variable", nonnegative=True)
    stopping_clock = -sp.log(1 - s * spectral) / spectral
    assert _clean(sp.diff(stopping_clock, s) - 1 / (1 - s * spectral)) == 0
    assert stopping_clock.subs(s, 0) == 0
    integrand = 1 / (1 - integration_variable * spectral)
    assert (
        _clean(
            sp.diff(integrand, spectral)
            - integration_variable / (1 - integration_variable * spectral) ** 2
        )
        == 0
    )
    # The integrand difference is strictly positive on 0<u<s<1/lambda_high.
    gap = 1 / (1 - integration_variable * high) - 1 / (1 - integration_variable * low)
    expected = (
        integration_variable
        * (high - low)
        / ((1 - integration_variable * high) * (1 - integration_variable * low))
    )
    assert _clean(gap - expected) == 0
    assert _clean(high - low).is_positive is True
    odds = sp.symbols("domain_odds", positive=True)
    inside = odds / (high * (1 + odds))
    assert _clean(1 - inside * high - 1 / (1 + odds)) == 0
    denominator_low = _clean((1 - inside * low) * (1 + odds))
    assert all(
        _clean(coefficient).is_positive is True
        for coefficient in sp.Poly(denominator_low, odds).all_coeffs()
    )
    # Thus one common kappa*T cannot equal both strictly ordered clocks. This
    # argument applies pointwise, even for a state-dependent stopping duration.
    mean, amplitude = sp.symbols("mean amplitude", positive=True)
    centered = sp.eye(5)[:, 0] - sp.ones(5, 1) / 5
    witness = mean * sp.ones(5, 1) + amplitude * centered
    for projector in (law["P_low"], law["P_high"]):
        occupied = projector * witness
        assert _clean(occupied.dot(occupied) - 2 * amplitude**2 / 5) == 0
    small_amplitude = mean * odds / (2 * (1 + odds))
    admitted = witness.subs(amplitude, small_amplitude).applyfunc(_clean)
    assert all(value.is_positive is True for value in admitted)
    assert _clean(mean / 2 - small_amplitude).is_positive is True
    _same(admitted.subs(odds, 0), mean * sp.ones(5, 1))
    assert sp.limit(small_amplitude, odds, 0, dir="+") == 0


def test_regular_clock_has_the_second_order_defect(comparison):
    law = comparison
    laplacian, s, kappa, elapsed = (law[key] for key in ("L", "s", "kappa", "elapsed"))
    correction = sp.symbols("clock_correction", real=True)
    calibrated = law["flow"].subs(elapsed, (s + correction * s**2) / kappa)
    difference = calibrated - law["event"]
    _same(difference.subs(s, 0), sp.zeros(5))
    _same(difference.diff(s).subs(s, 0), sp.zeros(5))
    coefficient = difference.diff(s, 2).subs(s, 0) / 2
    _same(coefficient, laplacian**2 / 2 - correction * laplacian)
    _same(coefficient.subs(correction, 0), laplacian**2 / 2)
    for projector, eigenvalue in (
        (law["P_low"], law["low"]),
        (law["P_high"], law["high"]),
    ):
        _same(
            coefficient * projector,
            eigenvalue * (eigenvalue / 2 - correction) * projector,
        )
        _same((coefficient * projector).subs(correction, eigenvalue / 2), sp.zeros(5))
    assert _clean(law["high"] / 2 - law["low"] / 2).is_positive is True
    # The scalar remainder behind the finite natural-clock error bound has
    # d(0)=d'(0)=0 and d''(y)=exp(-y), between zero and one for y>=0.
    y = sp.symbols("y", nonnegative=True)
    remainder = sp.exp(-y) - 1 + y
    assert remainder.subs(y, 0) == 0 and sp.diff(remainder, y).subs(y, 0) == 0
    assert sp.diff(remainder, y, 2) == sp.exp(-y)


def test_optimal_clock_coefficient_and_positive_leading_defect(comparison):
    law = comparison
    low, high = law["low"], law["high"]
    low_norm, high_norm = sp.symbols(
        "low_norm_squared high_norm_squared", positive=True
    )
    correction = sp.symbols("clock_correction", real=True)
    denominator = low_norm * low**2 + high_norm * high**2
    optimum = (low_norm * low**3 + high_norm * high**3) / (2 * denominator)
    loss = (
        low_norm * low**2 * (low / 2 - correction) ** 2
        + high_norm * high**2 * (high / 2 - correction) ** 2
    )
    minimum = (
        low_norm * high_norm * low**2 * high**2 * (high - low) ** 2 / (4 * denominator)
    )
    assert _clean(loss - denominator * (correction - optimum) ** 2 - minimum) == 0
    assert _clean(sp.diff(loss, correction).subs(correction, optimum)) == 0
    assert _clean(sp.diff(loss, correction, 2) - 2 * denominator) == 0
    # Strict positivity follows from these admitted factors, not a numerical
    # minimum search or a global CAS inequality involving exponentials.
    assert low.is_positive is True and high.is_positive is True
    assert _clean(high - low).is_positive is True
    assert low_norm.is_positive is True and high_norm.is_positive is True
    assert _clean(minimum.subs(low_norm, 0)) == 0
    assert _clean(minimum.subs(high_norm, 0)) == 0
    assert _clean(optimum.subs(low_norm, 0) - high / 2) == 0
    assert _clean(optimum.subs(high_norm, 0) - low / 2) == 0
    # Independently differentiate the actual five-coordinate witness loss.
    amplitude = sp.symbols("amplitude", real=True, nonzero=True)
    centered = amplitude * (sp.eye(5)[:, 0] - sp.ones(5, 1) / 5)
    laplacian = law["L"]
    residual = (laplacian**2 / 2 - correction * laplacian) * centered
    actual_loss = sp.expand(residual.dot(residual))
    derivative = sp.Poly(sp.diff(actual_loss, correction), correction)
    actual_optimum = _clean(-derivative.nth(0) / derivative.nth(1))
    assert actual_optimum == sp.Rational(5, 6)
    actual_minimum = _clean(actual_loss.subs(correction, actual_optimum))
    assert _clean(actual_minimum - 5 * amplitude**2 / 96) == 0
    assert actual_minimum.is_positive is True
    claimed_coefficient = sp.Abs(amplitude) * sp.sqrt(30) / 24
    assert _clean(claimed_coefficient**2 - actual_minimum) == 0
    assert claimed_coefficient.is_positive is True
    # The proof owns the asymptotic infimum over all finite durations. These
    # controls verify its exact quadratic coefficient and nonzero obstruction.
