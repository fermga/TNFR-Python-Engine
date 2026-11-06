"""Static preparation-resource controls on the complete two-child support."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.physics import _organized_free_phase_probe as owner


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _bound(**changes):
    return owner._bound_organized_free_phase_preparation(
        **(
            dict(
                initial_form_contrast_bound=0,
                continuous_form_variation_bound=0,
                jump_form_variation_bound=0,
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def edge_model():
    # Each edge contributes directly to both endpoint rows, without a
    # quotient graph or assumption that fine phases/forms agree.
    a, gamma = sp.symbols("a gamma", nonnegative=True)
    incidence = sp.zeros(27, 18)
    edges = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(2) for j in range(9))
    edges += tuple((j, 9 + j) for j in range(9))
    for row, (left, right) in enumerate(edges):
        incidence[row, left], incidence[row, right] = 1, -1
    weights = sp.diag(*([1] * 18 + [a] * 9))
    laplacian = incidence.T * weights * incidence
    contrast = sp.Matrix([[sp.Rational(1, 9)] * 9 + [-sp.Rational(1, 9)] * 9])
    return a, gamma, incidence, weights, laplacian, contrast


def test_full_edge_cancellation_and_external_source_projection(edge_model):
    a, gamma, incidence, weights, laplacian, contrast = edge_model
    x = sp.Matrix(sp.symbols("x0:18", real=True))
    source = sp.Matrix(sp.symbols("u0:18", real=True))
    currents = sp.Matrix(sp.symbols("j0:27", real=True))
    gradient = incidence.T * weights * currents
    dx = -(laplacian * x + gamma * gradient) / 4 + source
    dphase = gamma * laplacian * x / 4
    v = gamma * (contrast * x)[0]
    jbar = sum(currents[18:]) / 9
    projected_source = gamma * (sum(source[:9]) - sum(source[9:])) / 9
    assert (
        sp.expand(
            gamma * (contrast * dx)[0]
            + a * (v + gamma**2 * jbar) / 2
            - projected_source
        )
        == 0
    )
    assert sp.expand((contrast * dphase)[0] - a * v / 2) == 0
    assert sp.expand(sum(dx) - sum(source)) == 0
    # Removing the donor reaction or counting total global form as contrast
    # would lose respectively the factor two or the entire balanced source.
    balanced = sp.Matrix([1] * 9 + [-1] * 9)
    assert sum(balanced) == 0
    assert (contrast * balanced)[0] == 2
    assert (
        gamma * (contrast * balanced)[0]
        == gamma * sum(abs(value) for value in balanced) / 9
    )
    jump = sp.Matrix(
        [sp.Rational(i + 1, 9) for i in range(9)]
        + [-sp.Rational(i + 2, 11) for i in range(9)]
    )
    assert gamma * (contrast * jump)[0] == gamma * sum(abs(value) for value in jump) / 9


def test_global_current_majorant_and_invariant_barriers_need_no_fine_tube():
    theta, lam = sp.symbols("theta lam", real=True)
    expression = (
        sp.sin(theta) + lam * (sp.sin(9 * theta) - sp.sin(18 * theta) / 2) / 2187
    )
    coefficients = tuple(
        sp.expand(expression).coeff(sp.sin(k * theta)) for k in (1, 9, 18)
    )
    # On 0<=lambda<=1 the signed coefficient magnitudes sum to this affine
    # bound, whose maximum is at one, regardless of the fine phase state.
    majorant = coefficients[0] + coefficients[1] - coefficients[2]
    assert sp.diff(majorant, lam) > 0
    expected = Q(majorant.subs(lam, 1))
    report = _bound()
    assert report.current_bound == expected
    assert sum(abs(value) for _, value in report.current_coefficients) == expected
    c, excess, k, fraction = sp.symbols("c excess k fraction", nonnegative=True)
    magnitude = c + excess
    adverse_upper = -k * (magnitude - c + 2 * c * fraction)
    adverse_lower = -k * (-magnitude + c - 2 * c * fraction)
    assert sp.expand(adverse_upper) == -k * excess - 2 * k * c * fraction
    assert sp.expand(adverse_lower) == k * excess + 2 * k * c * fraction
    assert adverse_upper.is_nonpositive
    assert adverse_lower.is_nonnegative
    assert report.scaled_form_upper_bound < Q(1, 10**7)
    assert report.status == "excluded_by_resource_bound"


def test_isolated_uniform_forcing_saturates_variation_projection_without_work(
    edge_model,
):
    a, gamma, _, _, laplacian, contrast = edge_model
    amplitude, time, horizon = sp.symbols("amplitude time horizon", positive=True)
    direction = sp.Matrix([1] * 9 + [-1] * 9)
    source = amplitude * direction
    form = time * source
    isolated = laplacian.subs(a, 0)
    assert isolated * form == sp.zeros(18, 1)
    assert isolated * source == sp.zeros(18, 1)
    assert sum(source) == 0
    assert (isolated * form).dot(source) == 0
    assert (form.T * isolated * form)[0] == 0
    variation = sp.integrate(sum(abs(value) for value in source), (time, 0, horizon))
    achieved = gamma * (contrast * form.subs(time, horizon))[0]
    assert sp.simplify(achieved - gamma * variation / 9) == 0
    # This is a supplied input in a null form mode, not autonomous production
    # of contrast or a conclusion that physical preparation has zero cost.
    full_variation = Q(18) * Q(7, 3) * Q(5, 2)
    report = _bound(continuous_form_variation_bound=full_variation)
    assert (
        report.source_scaled_form_upper_bound
        == report.gamma_bounds.hi * full_variation / 9
    )
    assert report.source_scaled_form_upper_bound > 0
    assert report.scaled_form_upper_bound > _bound().scaled_form_upper_bound


def test_complete_work_balance_retains_contact_changes_and_form_sources(edge_model):
    a, gamma, incidence, weights, laplacian, _ = edge_model
    x = sp.Matrix(sp.symbols("x0:18", real=True))
    source = sp.Matrix(sp.symbols("u0:18", real=True))
    currents = sp.Matrix(sp.symbols("j0:27", real=True))
    potentials = sp.Matrix(sp.symbols("V0:27", real=True))
    adot = sp.Symbol("adot", real=True)
    gradient = incidence.T * weights * currents
    dx = -(laplacian * x + gamma * gradient) / 4 + source
    dphase = gamma * laplacian * x / 4
    storage = (x.T * laplacian * x)[0] / 2 + sum(weights * potentials)
    contact_storage = sum((x[i] - x[i + 9]) ** 2 / 2 for i in range(9))
    contact_storage += sum(potentials[18:])
    assert sp.expand(sp.diff(storage, a) - contact_storage) == 0
    observed_rate = (laplacian * x).dot(dx) + gradient.dot(dphase)
    observed_rate += adot * sp.diff(storage, a)
    expected_rate = -(laplacian * x).dot(laplacian * x) / 4
    expected_rate += (laplacian * x).dot(source) + adot * contact_storage
    assert sp.expand(observed_rate - expected_rate) == 0


def test_fresh_contact_floor_retains_arbitrary_fine_form_variance():
    mean = sp.Symbol("mean", real=True)
    fine = tuple(sp.symbols("z0:8", real=True))
    fine += (-sum(fine),)
    gaps = tuple(mean + value for value in fine)
    assert sp.expand(sum(value**2 for value in gaps) - 9 * mean**2) == sp.expand(
        sum(value**2 for value in fine)
    )
    gamma = sp.Symbol("gamma", positive=True)
    contact = gamma**2 / 1000
    floor = contact * 9 * (12 / gamma) ** 2 / 2
    assert sp.simplify(floor) == sp.Rational(81, 125)
    # Every squared residual and admitted edge potential is nonnegative;
    # neither can lower this fresh-attachment cost. Removing a contact has
    # the opposite storage jump, so this is not total work over all schedules.
    assert sp.simplify(-floor) < 0


def test_unforced_large_initial_mean_and_source_allowances_remain_separate():
    gamma = 1 / (1023 * pi_interval())
    r0 = Q(4) / gamma.hi
    continuous, jumps = Q(300), Q(17)
    report = _bound(
        initial_form_contrast_bound=r0,
        continuous_form_variation_bound=continuous,
        jump_form_variation_bound=jumps,
    )
    assert report.initial_scaled_form_upper_bound == 4
    assert report.unforced_scaled_form_upper_bound == 4
    assert report.scaled_form_upper_bound == 4 + gamma.hi * (continuous + jumps) / 9
    assert report.necessary_total_form_variation_lower_bound == 9 * (12 - 4) / gamma.hi
    supplied = _bound(initial_form_contrast_bound=Q(13) / gamma.hi)
    assert supplied.status == "not_excluded"
    assert supplied.necessary_total_form_variation_lower_bound == 0


def test_strict_resource_exclusion_and_equality_do_not_claim_reachability():
    gamma = 1 / (1023 * pi_interval())
    current_bound = Q(1) + Q(1, 2187) + Q(1, 4374)
    unforced = (gamma**2 * current_bound).hi
    threshold = 9 * (12 - unforced) / gamma.hi
    below = _bound(continuous_form_variation_bound=threshold - Q(1, 10**30))
    equality = _bound(jump_form_variation_bound=threshold)
    above = _bound(continuous_form_variation_bound=threshold + 1)
    assert below.status == "excluded_by_resource_bound"
    assert below.target_margin > 0
    assert equality.target_margin == 0
    assert equality.status == above.status == "not_excluded"
    assert equality.necessary_total_form_variation_lower_bound == threshold
    assert equality.total_form_variation_bound == threshold
    # Moving an upper allowance between continuous and jump sources does
    # not manufacture source occurrence or prescribe its chronology.
    split = _bound(
        continuous_form_variation_bound=threshold / 3,
        jump_form_variation_bound=2 * threshold / 3,
    )
    assert split.scaled_form_upper_bound == equality.scaled_form_upper_bound
    assert not hasattr(equality, "source_admitted")
    assert not hasattr(equality, "reachable")


def test_exact_tiny_and_numpy_fraction_primitives_are_normalized_and_immutable():
    np = pytest.importorskip("numpy")
    tiny = Q(1, 10**500)
    report = _bound(
        initial_form_contrast_bound=tiny,
        continuous_form_variation_bound=Q(np.int64(3), np.int64(7)),
    )
    assert report.initial_form_contrast_bound == tiny
    assert report.initial_scaled_form_upper_bound > 0
    assert report.continuous_form_variation_bound == Q(3, 7)
    for value in (
        report.initial_form_contrast_bound,
        report.continuous_form_variation_bound,
    ):
        assert type(value.numerator) is type(value.denominator) is int
    with pytest.raises(FrozenInstanceError):
        report.status = "reachable"


@pytest.mark.parametrize(
    "field",
    (
        "initial_form_contrast_bound",
        "continuous_form_variation_bound",
        "jump_form_variation_bound",
    ),
)
@pytest.mark.parametrize(
    "invalid", (None, True, -Q(1, 10**500), "0", 1j, float("nan"), float("inf"))
)
def test_missing_or_invalid_primitives_reject_before_law_rebuild(
    monkeypatch, field, invalid
):
    monkeypatch.setattr(
        owner,
        "_free_probe_current_bounds",
        lambda _: pytest.fail("early law calculation"),
    )
    with pytest.raises((TypeError, ValueError)):
        _bound(**{field: invalid})


def test_reports_do_not_replace_initial_or_source_resource_primitives():
    report = _bound()
    for field in (
        "initial_form_contrast_bound",
        "continuous_form_variation_bound",
        "jump_form_variation_bound",
    ):
        with pytest.raises((TypeError, ValueError)):
            _bound(**{field: report})
