"""Static cut and resource controls; no source or trajectory evaluation."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import combinations

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.physics import _reciprocal_form_resources as owner


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _bound(**changes):
    return owner._bound_reciprocal_form_resources(
        **(
            dict(
                initial_forms=(0,) * 18,
                marked_a=tuple(range(9)),
                marked_b=tuple(range(9, 18)),
            )
            | changes
        )
    )


def test_arbitrary_reciprocal_edges_cancel_inside_cuts_without_collective_closure():
    n = 6
    edges = tuple(combinations(range(n), 2))
    x = sp.Matrix(sp.symbols("x0:6", real=True))
    j = sp.Matrix(sp.symbols("j0:15", real=True))
    weights = sp.symbols("w0:15", nonnegative=True)
    gamma = sp.Symbol("gamma", positive=True)
    incidence = sp.zeros(len(edges), n)
    for row, (left, right) in enumerate(edges):
        incidence[row, left], incidence[row, right] = 1, -1
    laplacian = incidence.T * sp.diag(*weights) * incidence
    dx = -(laplacian * x + gamma * incidence.T * sp.diag(*weights) * j) / 4
    dtheta = gamma * laplacian * x / 4
    cut = frozenset((0, 2, 5))
    independent_cut_sum = 0
    for edge, (left, right) in enumerate(edges):
        if (left in cut) != (right in cut):
            orientation = 1 if left in cut else -1
            independent_cut_sum -= (
                orientation * weights[edge] * (x[left] - x[right] + gamma * j[edge]) / 4
            )
    assert sp.expand(sum(dx[i] for i in cut) - independent_cut_sum) == 0
    assert sp.expand(sum(dx)) == 0
    assert sp.expand(sum(dtheta)) == 0
    # This identity is on arbitrary fine coordinates. A one-sided edge loses
    # both reaction and global mean conservation even with a bounded current.
    nonreciprocal_dx = sp.Matrix(dx)
    nonreciprocal_dx[1] -= weights[0] * (x[0] - x[1] + gamma * j[0]) / 4
    assert sp.expand(sum(nonreciprocal_dx)) != 0


def test_saturated_cut_neighbors_force_every_cross_gap_inward():
    n, k = sp.symbols("n k", integer=True, positive=True)
    rho, left_slack, right_slack = sp.symbols(
        "rho left_slack right_slack", nonnegative=True
    )

    def cut(size):
        return rho * size * (n - size) / 2

    # The active k facet is exact; adjacent facets may have arbitrary slack.
    # These are the smallest inside and largest outside sorted coordinates.
    inside = cut(k) - (cut(k - 1) - left_slack)
    outside = cut(k + 1) - right_slack - cut(k)
    assert sp.expand(inside - outside) == rho + left_slack + right_slack
    gamma, current_bound, margin, gap_extra, z, weight = sp.symbols(
        "gamma current_bound margin gap_extra z weight", nonnegative=True
    )
    # At a facet rho>=gamma*J0, every outgoing edge current is bounded below
    # by -J0. z represents its nonnegative distance from that worst endpoint.
    gap = gamma * current_bound + margin + gap_extra
    current = -current_bound + z
    edge_rate = sp.expand(-weight * (gap + gamma * current) / 4)
    assert sp.expand(edge_rate + weight * (gap_extra + margin + gamma * z) / 4) == 0
    assert edge_rate.is_nonpositive
    # Neither a subcritical radius nor negative conductance has this property.
    boundary_rate = -weight * (rho + gamma * current) / 4
    adverse = {gamma: 1, current_bound: 1, z: 0}
    assert boundary_rate.subs(adverse | {weight: 1, rho: sp.Rational(1, 2)}) > 0
    assert boundary_rate.subs(adverse | {weight: -1, rho: 2}) > 0


def test_top_sum_resource_and_nine_mean_bound_are_sharp_on_staircase():
    n, k, rho = sp.symbols("n k rho", positive=True, integer=True)
    i = sp.Symbol("i", integer=True)
    prefix = sp.summation(rho * (n - 1 - 2 * i) / 2, (i, 0, k - 1))
    assert sp.expand(prefix - rho * k * (n - k) / 2) == 0
    forms = tuple(Q(7, 3) * (22 - 1 - 2 * i) / 2 for i in range(22))
    report = _bound(initial_forms=forms, marked_b=tuple(range(13, 22)))
    assert report.initial_form_mean == 0
    assert report.initial_resource == Q(7, 3)
    assert report.centered_top_sums == tuple(sum(forms[:k]) for k in range(1, 22))
    actual_contrast = (sum(forms[:9]) - sum(forms[-9:])) / 9
    assert actual_contrast == Q(7, 3) * (22 - 9)
    assert report.initial_marked_contrast == actual_contrast
    assert (
        report.scaled_contrast_upper_bound == report.gamma_bounds.hi * actual_contrast
    )


def test_cut_maximization_and_coordinate_error_factor_from_independent_subsets():
    # A nonuniform nonsorted reading has an active facet; changing that subset
    # by +eps and its complement by -eps attains the exact box supremum.
    forms = (Q(4), Q(-6), Q(3), Q(2)) + (Q(-1, 3),) * 14
    error = Q(5, 7)
    report = _bound(initial_forms=forms, form_error_bound=error)
    n = len(forms)
    mean = sum(forms) / n
    ordering = sorted(range(n), key=forms.__getitem__, reverse=True)
    resources = tuple(
        2 * sum(forms[i] - mean for i in ordering[:k]) / (k * (n - k))
        for k in range(1, n)
    )
    k = max(range(1, n), key=lambda count: resources[count - 1])
    active = frozenset(ordering[:k])
    centered_sum_coefficients = tuple(
        Q(1 if i in active else 0) - Q(k, n) for i in range(n)
    )
    error_from_dual_norm = error * sum(abs(a) for a in centered_sum_coefficients)
    resource_error = 2 * error_from_dual_norm / (k * (n - k))
    assert resource_error == 4 * error / n
    assert report.resource_error_upper_bound == resource_error
    perturbed = tuple(
        value + (error if i in active else -error) for i, value in enumerate(forms)
    )
    actual = _bound(initial_forms=perturbed)
    assert actual.initial_resource == report.initial_resource_upper_bound
    assert actual.initial_resource == max(resources) + resource_error
    # For a small cut, explicitly maximize over every subset instead of
    # treating the report's sort order as an expected-value oracle.
    all_pairs = (
        sum(forms[i] - mean for i in pair) for pair in combinations(range(n), 2)
    )
    assert report.centered_top_sums[1] == max(all_pairs)
    marked_perturbation = tuple(error if i < 9 else -error for i in range(n))
    shifted = _bound(
        initial_forms=tuple(x + dx for x, dx in zip(forms, marked_perturbation))
    )
    assert shifted.initial_marked_contrast - report.initial_marked_contrast == 2 * error
    assert report.initial_marked_contrast_error_bound == 2 * error


@pytest.mark.parametrize("n", (18, 20))
def test_zero_child_means_do_not_remove_fine_or_additional_node_resources(n):
    forms = [Q(0)] * n
    first, second = (0, 1) if n == 18 else (18, 19)
    forms[first], forms[second] = Q(100000), Q(-100000)
    report = _bound(initial_forms=forms)
    assert sum(forms[:9]) == sum(forms[9:18]) == 0
    assert report.initial_marked_contrast == 0
    # Here every nontrivial top sum is M; the extremal denominator is N-1.
    assert report.initial_resource == 2 * forms[first] / (n - 1)
    assert report.scaled_contrast_upper_bound > 12
    assert report.status == "not_excluded"
    assert _bound().status == "excluded_by_resource_bound"
    assert report.necessary_node_count_lower_bound is None
    assert not hasattr(report, "reachable")


def test_translation_and_consistent_relabeling_preserve_resource_and_marked_mean():
    forms = tuple(Q(i * i - 7 * i, 13) for i in range(23))
    a, b = tuple(range(0, 18, 2)), tuple(range(1, 18, 2))
    report = _bound(initial_forms=forms, marked_a=a, marked_b=b)
    shifted = _bound(
        initial_forms=tuple(x + Q(13, 7) for x in forms), marked_a=a, marked_b=b
    )
    permutation = tuple(reversed(range(23)))
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted = _bound(
        initial_forms=tuple(forms[i] for i in permutation),
        marked_a=tuple(inverse[i] for i in a),
        marked_b=tuple(inverse[i] for i in b),
    )
    for equivalent in (shifted, permuted):
        assert equivalent.initial_resource == report.initial_resource
        assert (
            equivalent.scaled_contrast_upper_bound == report.scaled_contrast_upper_bound
        )
        assert equivalent.initial_marked_contrast == report.initial_marked_contrast
    assert shifted.initial_form_mean - report.initial_form_mean == Q(13, 7)
    assert permuted.initial_forms == tuple(forms[i] for i in permutation)
    assert report.marked_a == a
    assert report.marked_b == b
    assert (
        report.initial_marked_contrast
        != (sum(sorted(forms, reverse=True)[:9]) - sum(sorted(forms)[:9])) / 9
    )


def test_uniform_current_bound_and_large_necessary_count_use_exact_small_inputs():
    phase, lam = sp.symbols("phase lam", real=True)
    current = sp.sin(phase) + lam * (sp.sin(9 * phase) - sp.sin(18 * phase) / 2) / 2187
    coefficients = tuple(
        sp.expand(current).coeff(sp.sin(k * phase)) for k in (1, 9, 18)
    )
    majorant = coefficients[0] + coefficients[1] - coefficients[2]
    assert sp.diff(majorant, lam) > 0
    j0 = Q(majorant.subs(lam, 1))
    gamma = 1 / (1023 * pi_interval())
    report = _bound(initial_forms=(Q(19, 11),) * 18)
    assert report.current_bound == j0
    assert report.invariant_resource_upper_bound == gamma.hi * j0
    assert report.scaled_contrast_upper_bound == gamma.hi**2 * j0 * 9
    threshold = report.necessary_node_count_lower_bound
    assert threshold > 10**8
    # This is a necessary integer lower bound obtained from an upper envelope,
    # not evidence that any network of that size attains the target.
    assert gamma.hi**2 * j0 * (threshold - 1 - 9) < 12
    assert gamma.hi**2 * j0 * (threshold - 9) >= 12
    assert report.status == "excluded_by_resource_bound"
    uncertain = _bound(form_error_bound=Q(1, 10**100))
    assert uncertain.necessary_node_count_lower_bound is None


def test_resource_target_equality_is_not_a_reachability_certificate():
    gamma = 1 / (1023 * pi_interval())
    radius = Q(12) / (9 * gamma.hi)
    forms = tuple(radius * (17 - 2 * i) / 2 for i in range(18))
    equality = _bound(initial_forms=forms)
    smaller = _bound(initial_forms=tuple(x / 2 for x in forms))
    larger = _bound(initial_forms=tuple(2 * x for x in forms))
    assert equality.initial_resource == radius
    assert equality.target_margin == 0
    assert equality.status == larger.status == "not_excluded"
    assert smaller.status == "excluded_by_resource_bound"
    assert smaller.target_margin > 0
    assert larger.target_margin < 0
    assert not hasattr(equality, "source_admitted")
    assert not hasattr(equality, "identity_retained")


def test_exact_tiny_signed_readings_numpy_fractions_and_immutable_provenance():
    np = pytest.importorskip("numpy")
    tiny = Q(1, 10**500)
    forms = [Q(0)] * 18
    forms[0], forms[1] = tiny, -tiny
    report = _bound(initial_forms=forms, form_error_bound=Q(np.int64(3), np.int64(7)))
    assert report.initial_forms[0] == tiny
    assert report.initial_forms[1] == -tiny
    assert report.initial_resource == 2 * tiny / 17
    assert report.form_error_bound == Q(3, 7)
    for value in (*report.initial_forms, report.form_error_bound):
        assert type(value.numerator) is type(value.denominator) is int
    with pytest.raises(FrozenInstanceError):
        report.status = "reachable"
    forms[0] = 999
    assert report.initial_forms[0] == tiny


def test_explicit_work_limit_accepts_boundary_and_bounds_iterable_consumption():
    report = _bound(initial_forms=(0 for _ in range(4096)))
    assert report.node_count == 4096
    assert report.scaled_contrast_upper_bound < 12
    consumed = []

    def too_many():
        for i in range(4097):
            consumed.append(i)
            yield 0
        pytest.fail("unbounded initial-form iterable consumption")

    with pytest.raises(ValueError, match="work limit"):
        _bound(initial_forms=too_many())
    assert len(consumed) == 4097


@pytest.mark.parametrize("value", (None, True, "0", 1j, float("nan"), float("inf")))
def test_invalid_form_scalar_rejects_before_current_rebuild(monkeypatch, value):
    monkeypatch.setattr(
        owner, "_free_probe_current_bounds", lambda _: pytest.fail("law rebuilt")
    )
    with pytest.raises((TypeError, ValueError)):
        _bound(initial_forms=(value,) + (0,) * 17)


@pytest.mark.parametrize(
    "value", (None, True, -Q(1, 10**500), "0", 1j, float("nan"), float("inf"))
)
def test_invalid_error_bound_rejects_before_current_rebuild(monkeypatch, value):
    monkeypatch.setattr(
        owner, "_free_probe_current_bounds", lambda _: pytest.fail("law rebuilt")
    )
    with pytest.raises((TypeError, ValueError)):
        _bound(form_error_bound=value)


@pytest.mark.parametrize(
    "forms",
    (None, "0" * 18, bytes(18), {i: 0 for i in range(18)}, set(range(18)), (0,) * 17),
)
def test_missing_unordered_or_short_initial_state_rejects(forms):
    with pytest.raises((TypeError, ValueError)):
        _bound(initial_forms=forms)


@pytest.mark.parametrize("field", ("marked_a", "marked_b"))
@pytest.mark.parametrize(
    "positions",
    (
        None,
        "012345678",
        {i: i for i in range(9)},
        set(range(9)),
        tuple(range(8)),
        tuple(range(10)),
        (0,) * 9,
        (-1,) + tuple(range(1, 9)),
        (18,) + tuple(range(1, 9)),
        (True,) + tuple(range(1, 9)),
        (0.0,) + tuple(range(1, 9)),
        (Q(0),) + tuple(range(1, 9)),
    ),
)
def test_marks_require_bounded_ordered_distinct_ordinary_integer_positions(
    field, positions
):
    with pytest.raises((TypeError, ValueError)):
        _bound(**{field: positions})


def test_overlapping_and_numpy_integer_marks_reject_without_implicit_coercion():
    with pytest.raises(ValueError, match="disjoint"):
        _bound(marked_b=tuple(range(8, 17)))
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError, match="ordinary Python integers"):
        _bound(marked_a=tuple(np.int64(i) for i in range(9)))


def test_cached_reports_never_replace_fresh_primitive_forms_or_bounds():
    report = _bound()
    for field in ("initial_forms", "marked_a", "form_error_bound"):
        with pytest.raises((TypeError, ValueError)):
            _bound(**{field: report})
