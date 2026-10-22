"""Independent analytic controls for the small-resource endpoint budget."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import factorial

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr._exact_time import exp_unit_bounds
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.physics import _organized_small_resource_information as owner


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _bound(**changes):
    return owner._bound_organized_small_resource_information(
        **(
            dict(
                phase_offset=Q(1, 100),
                form_error_bound=0,
                phase_error_bound=0,
                initial_form_contrast_error_bound=0,
                phase_registration_error_bound=0,
                endpoint_reading_error_bound=0,
                exposure_time_error_bound=0,
                relative_gain_error_bound=0,
            )
            | changes
        )
    )


def test_current_difference_is_cubic_despite_shared_linear_and_quadratic_jets():
    u = sp.Symbol("u", real=True)
    harmonic = (sp.sin(9 * u) - sp.sin(18 * u) / 2) / 2187
    factored = sp.sin(9 * u) * (1 - sp.cos(9 * u)) / 2187
    assert sp.trigsimp(harmonic - factored) == 0
    sine, alternative = sp.sin(u), sp.sin(u) + harmonic
    for derivative in range(3):
        assert sp.diff(sine, u, derivative).subs(u, 0) == sp.diff(
            alternative, u, derivative
        ).subs(u, 0)
    assert sp.diff(harmonic, u, 3).subs(u, 0) / factorial(3) == sp.Rational(1, 6)
    assert sp.diff(alternative, u, 3).subs(u, 0) == 0
    # Without the eighteenth harmonic the linear terms would not match.
    omitted = sp.sin(9 * u) / 2187
    assert sp.diff(omitted, u).subs(u, 0) != 0


def test_direct_reacting_law_difference_retains_the_shared_response():
    s, t = sp.symbols("s t", nonnegative=True)
    g = sp.Symbol("G", positive=True)
    forcing = sp.Function("F")
    difference = -g * sp.Integral((1 - sp.exp(-(s - t))) * forcing(t), (t, 0, s))
    assert (
        sp.simplify(sp.diff(difference, s, 2) + sp.diff(difference, s) + g * forcing(s))
        == 0
    )
    assert difference.subs(s, 0).doit() == sp.diff(difference, s).subs(s, 0).doit() == 0
    assert sp.integrate(sp.exp(t - 1), (t, 0, 1)) == 1 - sp.exp(-1)
    assert sp.integrate(1 - sp.exp(-(s - t)), (t, 0, s)) == s - 1 + sp.exp(-s)
    # A direct common-state comparison is essential as d tends to zero:
    # a separate O(gamma^3*d) approximation error need not resolve gamma*d^3.
    d, gamma = sp.symbols("d gamma", positive=True)
    assert sp.simplify(gamma**3 * d / (gamma * d**3)) == gamma**2 / d**2
    assert sp.limit(gamma**2 / d**2, d, 0, dir="+") == sp.oo


def test_rational_taylor_corridor_and_nominal_gap_constants():
    gamma = 1 / (1023 * pi_interval())
    g, j1 = gamma.hi**2, Q(245, 243)
    assert g < Q(1, 10**7)
    # The corridor also covers the latest allowed jitter exposure s=3/2.
    assert 1 - g * j1 * Q(3, 2) ** 2 / 2 > Q(99, 100)
    z = Q(1, 2)
    sine_factor = 1 - z**2 / 6
    cosine_factor = Q(1, 2) - z**2 / 24
    h_factor = sine_factor * cosine_factor / 3
    assert h_factor == Q(1081, 6912)
    assert h_factor * Q(99, 100) ** 3 > Q(1, 7)
    assert Q(1, 7) - g / 3 > Q(1, 8)
    assert Q(1, 6) + g / 3 < Q(1, 5)
    exp_minus_lower = sum((Q((-1) ** k, factorial(k)) for k in range(6)), Q(0))
    exp_minus_upper = sum((Q((-1) ** k, factorial(k)) for k in range(5)), Q(0))
    assert Q(1, 3) < exp_minus_lower < exp_minus_upper < Q(2, 5)
    assert (1 - exp_minus_upper) / 8 > Q(1, 20)
    assert (1 - exp_minus_lower) / 5 < Q(1, 6)
    # The bound exp(G)-1<=2G follows from comparison with a geometric series.
    assert g / (1 - g) < 2 * g
    report = _bound()
    assert report.nominal_gap_lower_bound == gamma.lo * Q(1, 100) ** 3 / 20
    assert report.nominal_gap_upper_bound == gamma.hi * Q(1, 100) ** 3 / 6
    assert report.status == "guaranteed_separation"


def test_fine_current_cancellation_and_squared_norm_bound():
    # Opposite child residuals saturate the contact-gap norm inequality.
    q = tuple(sp.symbols("q0:8", real=True))
    q += (-sum(q),)
    gaps = tuple(2 * value for value in q)
    assert sum(gaps) == 0
    fine_norm_squared = 2 * sum(value**2 for value in q)
    assert sp.expand(sum(value**2 for value in gaps) - 2 * fine_norm_squared) == 0
    current, slope, curvature = sp.symbols("current slope curvature", real=True)
    taylor_mean = (
        sum(current + slope * gap + curvature * gap**2 / 2 for gap in gaps) / 9
    )
    assert sp.expand(taylor_mean - current - curvature * fine_norm_squared / 9) == 0
    # The quadratic energy bound q^2<=4F has an exact nonnegative remainder.
    phase, scaled_form = sp.symbols("phase scaled_form", real=True)
    energy = ((phase + scaled_form) ** 2 + scaled_form**2) / 2
    assert sp.expand(4 * energy - phase**2) == sp.expand((phase + 2 * scaled_form) ** 2)
    report = _bound(form_error_bound=Q(1, 10**6), phase_error_bound=Q(1, 10**6))
    gu = report.gamma_bounds.hi
    independent_fine = ((Q(1, 10**6) + gu / 10**6) ** 2 + (gu / 10**6) ** 2) / 2
    assert report.fine_energy_upper_bound == independent_fine
    assert report.fine_current_defect_upper_bound == Q(10, 9) * 4 * independent_fine / 9


def test_rational_nuisance_bounds_follow_from_decaying_convolutions():
    s, t = sp.symbols("s t", nonnegative=True)
    decay = sp.Rational(125, 2)
    assert sp.integrate(sp.exp(-decay * t), (t, 0, sp.oo)) == 1 / decay
    convolution = sp.integrate(sp.exp(t - 1) * sp.exp(-decay * t), (t, 0, 1))
    assert sp.simplify(convolution - (sp.exp(-1) - sp.exp(-decay)) / (decay - 1)) == 0
    # Check the geometric majorant against the shared certified exponential
    # series at the two whole-window endpoints, without evaluating a response.
    gu = (1 / (1023 * pi_interval())).hi
    for endpoint in (Q(1), Q(3, 2)):
        exponent = gu**2 * Q(245, 243) * endpoint
        assert exp_unit_bounds(exponent)[1] < 1 / (1 - exponent)
    report = _bound(
        form_error_bound=Q(1, 10**6),
        phase_error_bound=Q(1, 10**6),
        initial_form_contrast_error_bound=Q(1, 10**14),
        phase_registration_error_bound=Q(1, 10**11),
        endpoint_reading_error_bound=Q(1, 10**14),
        exposure_time_error_bound=Q(1, 10**8),
        relative_gain_error_bound=Q(1, 10**8),
    )
    gu = (1 / (1023 * pi_interval())).hi
    j1, j2 = Q(245, 243), Q(10, 9)
    k = Q(decay)
    f = (
        (report.phase_error_bound + gu * report.form_error_bound) ** 2
        + (gu * report.form_error_bound) ** 2
    ) / 2
    b = j2 * 4 * f / 9
    a = Q(1, 10**11) + gu / 10**14 + gu**2 * b / k
    p1, pt = a / (1 - gu**2 * j1), a / (1 - gu**2 * j1 * (1 + Q(1, 10**8)))
    endpoint_error = (
        Q(2, 5 * 10**14) + Q(2, 3) * gu * j1 * p1 + Q(2, 5) * gu * b / (k - 1)
    )
    rmax = Q(1, 10**14) + gu * j1 * (Q(1, 100) + pt) + gu * b / k
    rate = Q(1, 10**14) + 2 * gu * j1 * (Q(1, 100) + pt) + gu * b * (1 + 1 / k)
    assert report.endpoint_phase_error_upper_bound == p1
    assert report.window_phase_error_upper_bound == pt
    assert report.endpoint_preparation_error_upper_bound == endpoint_error
    assert report.window_form_upper_bound == rmax
    assert report.window_form_rate_upper_bound == rate
    expected_total = endpoint_error + rate / 10**8 + rmax / 10**8 + Q(1, 10**14)
    assert report.observation_error_upper_bound == expected_total
    assert report.separation_margin > 0
    assert report.status == "guaranteed_separation"
    # This is a nonempty strictly positive allowance box, not only an ideal
    # noiseless limit or a response evaluated to choose its own tolerance.
    assert (
        min(
            report.form_error_bound,
            report.phase_error_bound,
            report.initial_form_contrast_error_bound,
            report.phase_registration_error_bound,
            report.endpoint_reading_error_bound,
            report.exposure_time_error_bound,
            report.relative_gain_error_bound,
        )
        > 0
    )


def test_affine_observation_needs_time_and_gain_on_the_actual_uncertain_signal():
    r, prep, timing, gain, reading = sp.symbols("r prep timing gain reading", real=True)
    observed = (1 + gain) * (r + prep + timing) + reading
    assert sp.expand(observed - r) == sp.expand(
        prep + timing + gain * (r + prep + timing) + reading
    )
    assert sp.expand(observed - r - (prep + timing + gain * r + reading)) == sp.expand(
        gain * (prep + timing)
    )
    exact = _bound()
    jitter = _bound(exposure_time_error_bound=Q(1, 2))
    gain_budget = _bound(relative_gain_error_bound=2)
    assert jitter.exposure_window == (Q(1, 2), Q(3, 2))
    assert jitter.exposure_error_upper_bound > 0
    assert gain_budget.gain_error_upper_bound == 2 * gain_budget.window_form_upper_bound
    assert jitter.status == gain_budget.status == "not_certified"
    assert exact.status == "guaranteed_separation"


def test_common_reading_witness_is_genuine_and_failed_separation_is_not_ambiguity():
    center, gap = sp.symbols("center gap", real=True)
    endpoint_sine, endpoint_alternative = center + gap / 2, center - gap / 2
    noise_sine, noise_alternative = -gap / 2, gap / 2
    assert (
        endpoint_sine + noise_sine == endpoint_alternative + noise_alternative == center
    )
    exact = _bound()
    overlap = _bound(endpoint_reading_error_bound=exact.nominal_gap_upper_bound / 2)
    assert overlap.status == "observation_class_nonidentifiable"
    assert overlap.nominal_reading_overlap_margin == 0
    undecided = _bound(endpoint_reading_error_bound=exact.nominal_gap_lower_bound / 2)
    assert undecided.separation_margin == 0
    assert undecided.nominal_reading_overlap_margin < 0
    assert undecided.status == "not_certified"
    # This large registration box destroys our separation guarantee but is
    # not reinterpreted as a proved common reading from interval overlap.
    registration = _bound(phase_registration_error_bound=Q(1, 100))
    assert registration.status == "not_certified"
    assert not hasattr(overlap, "selected_law")
    assert not hasattr(overlap, "observed_response")


def test_phase_preparation_cannot_be_cancelled_between_independent_candidates():
    nominal = _bound()
    registered = _bound(phase_registration_error_bound=Q(1, 10**5))
    assert (
        registered.endpoint_preparation_error_upper_bound
        > nominal.nominal_gap_upper_bound
    )
    assert registered.separation_margin < 0
    half = _bound(phase_offset=Q(1, 200), phase_registration_error_bound=Q(1, 10**5))
    assert (
        half.endpoint_preparation_error_upper_bound
        == registered.endpoint_preparation_error_upper_bound
    )
    assert half.nominal_gap_lower_bound == registered.nominal_gap_lower_bound / 8
    assert (
        half.nominal_attachment_work_upper_bound
        == registered.nominal_attachment_work_upper_bound / 4
    )


def test_contact_work_retains_mean_and_fine_preparation_and_structural_clock():
    r, d, a, j1 = sp.symbols("r d a j1", positive=True)
    fine_form = tuple(sp.symbols("z0:8", real=True))
    fine_form += (-sum(fine_form),)
    fine_phase = tuple(sp.symbols("q0:8", real=True))
    fine_phase += (-sum(fine_phase),)
    gap_storage = a * sum(
        (r + 2 * z) ** 2 / 2 + j1 * (d + 2 * q) ** 2 / 2
        for z, q in zip(fine_form, fine_phase)
    )
    expected = a * (
        9 * (r**2 + j1 * d**2) / 2
        + 2 * sum(z**2 for z in fine_form)
        + 2 * j1 * sum(q**2 for q in fine_phase)
    )
    assert sp.expand(gap_storage - expected) == 0
    report = _bound(
        form_error_bound=Q(1, 10**6),
        phase_error_bound=Q(1, 10**6),
        initial_form_contrast_error_bound=Q(1, 10**5),
        phase_registration_error_bound=Q(1, 1000),
    )
    bracket = Q(9, 2) * (Q(1, 10**5) ** 2 + Q(245, 243) * (Q(1, 100) + Q(1, 1000)) ** 2)
    bracket += Q(1, 10**6) ** 2 * (1 + Q(245, 243))
    assert (
        report.attachment_work_upper_bound
        == report.contact_conductance_bounds.hi * bracket
    )
    assert Q(9, 2) * (1 - Q(1, 100) ** 2 / 12) > 4
    assert Q(9, 2) * Q(245, 243) < 5
    # Both rows use d/ds=(2/a)d/dtau, so nominal s=1 is not tau=1.
    gamma = 1 / (1023 * pi_interval())
    assert report.nominal_horizon_bounds.lo <= Q(2000) / gamma.hi**2
    assert report.nominal_horizon_bounds.hi >= Q(2000) / gamma.lo**2
    assert report.fine_decay_rate_in_scaled_clock == Q(2000, 32)
    assert report.nominal_attachment_work_lower_bound > 0


def test_fine_admission_is_strict_and_unsupported_is_not_a_failed_identity():
    gu = (1 / (1023 * pi_interval())).hi
    boundary_form = Q(1, 200) / gu
    with pytest.raises(ValueError, match="F0_upper"):
        _bound(form_error_bound=boundary_form)
    admitted = _bound(form_error_bound=boundary_form * Q(999, 1000))
    assert admitted.fine_energy_margin > 0
    assert admitted.status == "not_certified"
    with pytest.raises(ValueError, match="F0_upper"):
        _bound(phase_error_bound=Q(1, 100))


def test_exact_tiny_phase_and_numpy_fraction_budgets_are_not_rounded_to_zero():
    np = pytest.importorskip("numpy")
    tiny = Q(1, 10**500)
    report = _bound(phase_offset=tiny)
    assert report.phase_offset == tiny
    assert report.nominal_gap_lower_bound > 0
    assert report.nominal_attachment_work_lower_bound > 0
    assert report.status == "guaranteed_separation"
    represented = _bound(relative_gain_error_bound=Q(np.int64(2), np.int64(3)))
    assert represented.relative_gain_error_bound == Q(2, 3)
    assert type(represented.relative_gain_error_bound.numerator) is int
    assert type(represented.relative_gain_error_bound.denominator) is int
    with pytest.raises(FrozenInstanceError):
        report.status = "selected_law"


@pytest.mark.parametrize(
    "field",
    (
        "phase_offset",
        "form_error_bound",
        "phase_error_bound",
        "initial_form_contrast_error_bound",
        "phase_registration_error_bound",
        "endpoint_reading_error_bound",
        "exposure_time_error_bound",
        "relative_gain_error_bound",
    ),
)
@pytest.mark.parametrize(
    "value", (None, True, -Q(1, 10**500), "0", 1j, float("nan"), float("inf"))
)
def test_invalid_primitives_reject_before_parameter_or_law_rebuild(
    monkeypatch, field, value
):
    monkeypatch.setattr(
        owner, "_parameters", lambda _: pytest.fail("early parameter rebuild")
    )
    monkeypatch.setattr(
        owner, "_free_probe_current_bounds", lambda _: pytest.fail("early law rebuild")
    )
    with pytest.raises((TypeError, ValueError)):
        _bound(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    (
        ("phase_offset", 0),
        ("phase_offset", Q(1, 100) + Q(1, 10**500)),
        ("exposure_time_error_bound", Q(1, 2) + Q(1, 10**500)),
    ),
)
def test_unsupported_positive_domains_reject(field, value):
    with pytest.raises(ValueError):
        _bound(**{field: value})


def test_cached_reports_do_not_replace_preparation_or_observation_primitives():
    report = _bound()
    for field in (
        "phase_offset",
        "form_error_bound",
        "phase_registration_error_bound",
        "endpoint_reading_error_bound",
    ):
        with pytest.raises((TypeError, ValueError)):
            _bound(**{field: report})
