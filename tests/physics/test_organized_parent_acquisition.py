"""Independent static controls for acquisition on the fixed 36-node law."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_parent_acquisition as owner
from tnfr.physics import _repeated_organized_composition as composition

_ERROR = Q(1, 10**10)
_RECOVERED = Q(1, 10**70)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 240
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected, width="1e-28"):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
    assert _mp(mp, interval.width) < mp.mpf(width)


def _edges():
    return (
        tuple(
            (18 * g + 9 * a + j, 18 * g + 9 * a + (j + 1) % 9)
            for g in range(2)
            for a in range(2)
            for j in range(9)
        )
        + tuple((18 * g + j, 18 * g + 9 + j) for g in range(2) for j in range(9))
        + tuple((9 * a + j, 18 + 9 * a + j) for a in range(2) for j in range(9))
    )


def _laplacian(values):
    result = [0] * 36
    for left, right in _edges():
        difference = values[left] - values[right]
        result[left] += difference
        result[right] -= difference
    return tuple(result)


def _assess(**changes):
    return owner._assess_organized_parent_acquisition(
        **(
            dict(form_error_bounds=(_ERROR,) * 4, phase_error_bounds=(_ERROR,) * 4)
            | changes
        )
    )


@pytest.fixture(scope="module")
def acquisition():
    return _assess()


def test_all_four_original_children_are_fresh_and_keep_full_source(acquisition):
    report = acquisition
    assert len(report.child_sources) == 4
    assert len({id(source) for source in report.child_sources}) == 4
    assert len({id(source.source) for source in report.child_sources}) == 4
    for source in report.child_sources:
        assert source.deformation == 0
        assert source.form_error_bound == source.phase_error_bound == _ERROR
        assert source.scaled_handoff_time == 100 + 10**13
        assert source.endpoint_form_norm_squared_upper_bound < _RECOVERED**2
        assert source.endpoint_phase_norm_squared_upper_bound < _RECOVERED**2
        assert source.recovery_certified
    assert report.recovered_child_norm_bound == _RECOVERED
    assert report.initial_child_phase_origins == (
        -Q(11, 40),
        Q(9, 40),
        -Q(9, 40),
        Q(11, 40),
    )
    # Each of the four children retains two independent nine-coordinate
    # zero-mean residuals: 4*(9-1)*2=64 source directions, not four rigid nodes.
    assert report.relative_preparation_dimension == 64
    assert Q(1, 2) - 2 * _RECOVERED > Q(1, 4)
    assert report.sources_certified and report.admitted
    assert not report.unresolved_conditions
    with pytest.raises(FrozenInstanceError):
        report.sources_certified = False


def test_product_spectrum_justifies_global_heat_bound_coefficients(acquisition, mp):
    # Independent Fourier/parity eigenvectors of the actual fine support.
    eigenvalues = []
    for k in range(9):
        for parent_parity in range(2):
            for child_parity in range(2):
                vector = tuple(
                    (-1) ** (parent_parity * g + child_parity * a)
                    * mp.exp(2j * mp.pi * k * j / 9)
                    for g in range(2)
                    for a in range(2)
                    for j in range(9)
                )
                eigenvalue = (
                    2
                    - 2 * mp.cos(2 * mp.pi * k / 9)
                    + 2 * parent_parity
                    + 2 * child_parity
                )
                assert all(
                    abs(observed - eigenvalue * value) < mp.mpf("1e-220")
                    for observed, value in zip(_laplacian(vector), vector)
                )
                eigenvalues.append(eigenvalue)
    positive = [value for value in eigenvalues if value > mp.mpf("1e-220")]
    assert min(positive) > mp.mpf(1) / 3
    assert max(eigenvalues) < 8
    # Integrating the heat semigroup gives phase growth coefficient
    # gamma^2*||L||/4 <= 2*gamma^2; the form convolution costs <= 24*gamma.
    assert max(eigenvalues) / 4 < 2
    assert max(eigenvalues) / min(positive) < 24
    assert acquisition.geometry.weighted_edges == tuple(
        (left, right, Q(1)) for left, right in _edges()
    )
    assert acquisition.form_laplacian_gap_lower_bound == Q(1, 3)


def test_separable_two_pendula_match_both_fine_rows_and_slow_clock(mp):
    outer_form, inner_form = Q(1, 7), -Q(1, 9)
    outer_gap, inner_gap = Q(1, 20), Q(1, 2)
    forms = tuple(
        Q(3, 10) + (2 * g - 1) * outer_form / 2 + (2 * a - 1) * inner_form / 2
        for g in range(2)
        for a in range(2)
        for _ in range(9)
    )
    phases = tuple(
        Q(1, 10) + (2 * g - 1) * outer_gap / 2 + (2 * a - 1) * inner_gap / 2
        for g in range(2)
        for a in range(2)
        for _ in range(9)
    )
    observed = composition._observe_repeated_composition(
        interface="matching", form=forms, phase_offsets=phases
    )
    gamma = 1 / (1023 * mp.pi)
    for form, gap, form_rate, phase_rate in (
        (
            outer_form,
            outer_gap,
            observed.parent_form_rate_bounds[1] - observed.parent_form_rate_bounds[0],
            observed.parent_phase_rate_bounds[1] - observed.parent_phase_rate_bounds[0],
        ),
        (
            inner_form,
            inner_gap,
            observed.child_form_rate_bounds[1] - observed.child_form_rate_bounds[0],
            observed.child_phase_rate_bounds[1] - observed.child_phase_rate_bounds[0],
        ),
    ):
        u, d = _mp(mp, form), _mp(mp, gap)
        _contains(mp, form_rate, -(u + gamma * mp.sin(d)) / 2)
        _contains(mp, phase_rate, gamma * u / 2)
        # Transform BOTH rows under s=gamma^2*tau/2 and y=-u/gamma.
        y = -u / gamma
        _contains(mp, phase_rate / observed.geometry.gamma_bounds**2 * 2, -y)
        _contains(
            mp,
            -form_rate / observed.geometry.gamma_bounds**3 * 2,
            (mp.sin(d) - y) / gamma**2,
            width="1e-18",
        )


def test_full_error_bounds_cover_long_slow_horizon_without_fast_gronwall(
    acquisition, mp
):
    gamma = 1 / (1023 * mp.pi)
    initial = 2 * _mp(mp, _RECOVERED)
    phase = initial * (1 + gamma) * mp.exp(20)
    form = initial + 24 * gamma * phase
    assert phase <= _mp(mp, acquisition.full_phase_error_upper_bound) < mp.mpf("1e-50")
    assert form <= _mp(mp, acquisition.full_form_error_upper_bound) < mp.mpf("1e-50")
    assert acquisition.full_phase_error_upper_bound > 0
    assert acquisition.full_form_error_upper_bound > 0


def test_source_inclusive_entry_maintenance_and_resolved_handoff(acquisition, mp):
    report = acquisition
    gamma, c = 1 / (1023 * mp.pi), mp.mpf(23) / 24
    epsilon, a = gamma**2, 1 + 2 * gamma**2
    for bounds, initial, time in (
        (report.nominal_inner_gap_at_entry_bounds, mp.mpf(1) / 2, 4),
        (report.nominal_outer_gap_at_entry_bounds, mp.mpf(1) / 20, 4),
        (report.nominal_outer_gap_at_readout_bounds, mp.mpf(1) / 20, 5),
    ):
        assert _mp(mp, bounds.lo) <= initial * mp.exp(-a * time)
        assert initial * mp.exp(-c * (time - epsilon)) <= _mp(mp, bounds.hi)
    assert report.nominal_inner_gap_at_entry_bounds.hi < Q(1, 64)
    assert report.parent_phase_residual_upper_bound < Q(1, 100)
    assert report.parent_form_residual_upper_bound < Q(1, 100)
    assert report.parent_gap_upper_bound < Q(1, 4)
    # These analytic envelopes share the same initial phase gap; their
    # ratio is <1/2, retaining the source correlation through the handoff.
    assert mp.exp(4 * a - 5 * c + c * epsilon) < mp.mpf(1) / 2
    assert mp.exp(4 * a) < 60
    assert report.outer_gap_decline_lower_bound > Q(1, 10000)
    assert report.phase_prediction_error_upper_bound < Q(1, 100000)
    assert report.parent_law_phase_error_upper_bound < Q(1, 100000)
    z = report.full_phase_error_upper_bound
    r = report.full_form_error_upper_bound
    assert report.outer_gap_decline_lower_bound <= Q(1, 2400) - 4 * z
    assert _mp(mp, report.phase_prediction_error_upper_bound) >= epsilon / 10 + 4 * _mp(
        mp, z
    )
    assert _mp(mp, report.parent_law_phase_error_upper_bound) >= 8 * _mp(
        mp, z
    ) + 6 * gamma * _mp(mp, r)
    assert report.entry_certified and report.child_identities_retained
    assert report.parent_identity_retained and report.prediction_certified
    assert all(margin > 0 for _, margin in report.sufficient_margins)
    for bounds, structural_time in (
        (report.elapsed_scaled_acquisition_time_bounds, 8 / gamma**2),
        (report.elapsed_scaled_readout_time_bounds, 10 / gamma**2),
        (report.elapsed_original_acquisition_time_bounds, 8 / gamma**2 * 1024 / 1023),
        (report.elapsed_original_readout_time_bounds, 10 / gamma**2 * 1024 / 1023),
    ):
        _contains(mp, bounds, structural_time, width="1e-18")


def test_distinct_primitive_allowances_keep_child_association():
    forms = tuple(_ERROR / divisor for divisor in (1, 2, 3, 4))
    phases = tuple(_ERROR / divisor for divisor in (5, 6, 7, 8))
    report = _assess(form_error_bounds=forms, phase_error_bounds=phases)
    assert report.form_error_bounds == forms and report.phase_error_bounds == phases
    assert report.sources_certified
    assert tuple(child.form_error_bound for child in report.child_sources) == forms
    assert tuple(child.phase_error_bound for child in report.child_sources) == phases


@pytest.mark.parametrize("gap", (Q(0), Q(1, 2000), Q(1, 1000), Q(1, 20)))
def test_gradient_predictor_uses_actual_admitted_gap_not_original_preparation(gap, mp):
    predicted = owner._predict_parent_gap(gap)
    value = _mp(mp, gap)
    expected = 2 * mp.atan(mp.exp(-1) * mp.tan(value / 2))
    _contains(mp, predicted, expected)
    if gap:
        assert 0 < predicted.lo <= predicted.hi < gap
    else:
        assert predicted.lo == predicted.hi == 0


@pytest.mark.parametrize(
    "bad", (True, "0", float("nan"), float("inf"), -Q(1, 10**100), Q(1, 19))
)
def test_gradient_predictor_rejects_invalid_or_unsupported_gap(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._predict_parent_gap(bad)


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("bad", (True, -Q(1, 10**100), float("nan"), "0"))
def test_all_eight_primitives_admit_before_any_recovery(field, index, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached isolated source reconstruction")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    values = [_ERROR] * 4
    values[index] = bad
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: tuple(values)})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("bad", ((), (_ERROR,) * 3, (_ERROR,) * 5, "0000", {_ERROR}))
def test_source_shapes_reject_before_any_recovery(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid source shape reached recovery")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize("index", (2, 3))
def test_unavailable_third_or_fourth_child_withholds_source_certificate(
    index, acquisition, monkeypatch
):
    calls = []

    def unavailable_child(**kwargs):
        child = acquisition.child_sources[len(calls)]
        calls.append(kwargs)
        if len(calls) - 1 == index:
            return replace(child, recovery_certified=False, status="unavailable")
        return child

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", unavailable_child)
    report = _assess()
    assert len(calls) == 4
    assert not report.sources_certified and not report.admitted
    assert report.status == "unavailable"
    assert report.unresolved_conditions
    assert (
        report.conditional_entry_certified and report.conditional_prediction_certified
    )
    assert not report.entry_certified and not report.prediction_certified


@pytest.mark.parametrize(
    "field",
    (
        "endpoint_form_norm_squared_upper_bound",
        "endpoint_phase_norm_squared_upper_bound",
    ),
)
@pytest.mark.parametrize("value", (None, _RECOVERED**2))
def test_recovery_flag_cannot_replace_strict_consumed_endpoint_radius(
    field, value, acquisition, monkeypatch
):
    calls = []

    def changed_endpoint(**kwargs):
        child = acquisition.child_sources[len(calls)]
        calls.append(kwargs)
        return replace(child, **{field: value}) if len(calls) == 4 else child

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", changed_endpoint)
    report = _assess()
    assert report.child_sources[3].recovery_certified
    assert not report.sources_certified and not report.admitted
    assert "child_3_original_recovery_not_certified" in report.unresolved_conditions


def test_exact_prediction_margin_boundary_is_unavailable(acquisition, monkeypatch):
    monkeypatch.setattr(
        owner, "_PREDICTION_ERROR", acquisition.phase_prediction_error_upper_bound
    )
    report = _assess()
    assert dict(report.sufficient_margins)["source_specific_phase_prediction"] == 0
    assert report.sources_certified and report.entry_certified
    assert not report.conditional_prediction_certified and not report.admitted


@pytest.mark.parametrize(
    "extra",
    ("source_report", "endpoint_radius", "slow_horizon", "response", "interface"),
)
def test_frozen_protocol_and_source_evidence_cannot_be_replaced(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})


def test_attachment_work_keeps_nonzero_subgrid_source_correction(acquisition, mp):
    report = acquisition
    epsilon = _mp(mp, _RECOVERED)
    nominal = 18 * (1 - mp.cos(mp.mpf(1) / 2)) + 18 * (1 - mp.cos(mp.mpf(1) / 20))
    _contains(mp, report.attachment_nominal_work_bounds, nominal)
    assert report.attachment_residual_work_bounds == (0, 16 * _RECOVERED**2)
    assert report.complete_initial_storage_residual_bounds == (0, 32 * _RECOVERED**2)
    x = tuple(
        epsilon * (1 if (g + a) % 2 else -1) * (j - 4) / 10
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    residual = tuple(
        epsilon * (1 if a else -1) * (j * j - mp.mpf(68) / 3) / 100
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    theta = tuple(
        2 * mp.pi * (j - 4) / 9
        + mp.mpf(2 * g - 1) / 40
        + mp.mpf(2 * a - 1) / 4
        + residual[18 * g + 9 * a + j]
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    for start in range(0, 36, 9):
        assert sum(value**2 for value in x[start : start + 9]) < epsilon**2
        assert sum(value**2 for value in residual[start : start + 9]) < epsilon**2
    actual = sum(
        (x[right] - x[left]) ** 2 / 2 + 1 - mp.cos(theta[right] - theta[left])
        for left, right in _edges()[36:]
    )
    assert 0 < actual - nominal <= 16 * epsilon**2


def test_antipodal_nonentry_control_is_exact_paired_fine_state_not_a_neighborhood(mp):
    # Identical actual child states in each parent, translated by exactly pi,
    # give zero intraparent currents and identical complete rows. The relation
    # is invariant by uniqueness, even while each child's fine state moves.
    gamma = 1 / (1023 * mp.pi)
    forms = tuple(
        mp.mpf((g + 1) * (j - 4)) / 1000
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    phases = tuple(
        2 * mp.pi * (j - 4) / 9
        + mp.mpf(2 * g - 1) / 40
        + (2 * a - 1) * mp.pi / 2
        + mp.mpf((g + 1) * (j * j - mp.mpf(68) / 3)) / 10000
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )

    def rows(theta):
        gradient = _laplacian(forms)
        current = [mp.mpf(0)] * 36
        for left, right in _edges():
            value = mp.sin(theta[right] - theta[left])
            current[left] += value
            current[right] -= value
        return (
            tuple((-q + gamma * v) / 4 for q, v in zip(gradient, current)),
            tuple(gamma * q / 4 for q in gradient),
        )

    dx, dt = rows(phases)
    assert abs(sum(dx[:18]) / 18) > mp.mpf("1e-6")
    for g in range(2):
        for j in range(9):
            left, right = 18 * g + j, 18 * g + 9 + j
            assert abs(phases[right] - phases[left] - mp.pi) < mp.mpf("1e-220")
            assert abs(dx[right] - dx[left]) < mp.mpf("1e-220")
            assert abs(dt[right] - dt[left]) < mp.mpf("1e-220")
    changed = list(phases)
    changed[9] += mp.mpf(1) / 100000
    changed_dx, _ = rows(changed)
    assert abs(changed_dx[9] - changed_dx[0]) > mp.mpf("1e-12")


def test_antipodal_control_retains_its_distinct_work_and_finite_child_scope(
    acquisition, mp
):
    report = acquisition
    _contains(mp, report.antipodal_control_parent_gap_bounds, mp.pi)
    _contains(
        mp,
        report.antipodal_control_nominal_work_bounds,
        36 + 18 * (1 - mp.cos(mp.mpf(1) / 20)),
    )
    assert (
        report.antipodal_control_nominal_work_bounds.lo
        > report.attachment_work_bounds.hi
    )
    assert report.antipodal_control_residual_work_bounds == (0, 8 * _RECOVERED**2)
    assert (
        "child_winding_retention_only_through_slow_time_five_by_full_error_bound"
        in report.antipodal_control_scope
    )
    assert (
        "no_open_neighborhood_nonentry_or_antipodal_stability_claim"
        in report.antipodal_control_scope
    )
