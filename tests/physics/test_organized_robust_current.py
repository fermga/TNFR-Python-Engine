"""Synthetic controls for robust current bounds and a model-independent readout.

No acquired source, reserved corner, response or physical experiment runs.
Static reading controls solve the declared observation algebra only; they are
not supplied as realizable histories. Independent decimal trigonometry checks
the directed rational current enclosures.
"""

from copy import deepcopy
from decimal import Decimal
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
import subprocess
import sys

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_robust_current as owner


_RATIOS = (
    "capacitance_ratio",
    "conductance_ratio",
    "josephson_ratio",
    "clock_ratio",
)
_ERRORS = ("total_tracking", "plateau_tracking", "raw_readout_error", "aperture")


def _parameters(**changes):
    values = {name: Q(1) for name in _RATIOS}
    values.update({name: Q(0) for name in _ERRORS})
    values.update(changes)
    return values


@pytest.fixture(autouse=True)
def no_source_or_frozen_protocol(request, monkeypatch):
    if request.node.name.startswith("test_cold_"):
        yield
        return
    from tnfr.physics import _organized_finite_observation as finite

    def forbidden(*args, **kwargs):
        pytest.fail("a frozen history bound or inference was consumed")

    monkeypatch.setattr(finite, "_bound_organized_finite_observation", forbidden)
    monkeypatch.setattr(finite, "_infer_finite_observation_winding", forbidden)
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def nominal():
    return owner._bound_organized_robust_current(**_parameters())


def _inside(bounds, value):
    return bounds.lo <= value <= bounds.hi


def _mp(value):
    return mpmath.mpf(value.numerator) / value.denominator


def test_exact_calibration_separates_without_zeroing_live_history_error(nominal):
    assert nominal.admitted
    assert nominal.status == "conditionally_separated_current_envelopes"
    assert not nominal.unresolved_conditions
    assert nominal.separation_margin > Q(1, 2)
    assert nominal.total_elapsed_scaled_time == Q(300003, 20000)
    for name in (
        "lambda_bounds",
        "rho_bounds",
        "eta_bounds",
        "beta_bounds",
        "eta_rho_bounds",
    ):
        assert getattr(nominal, name) == I(1)
    assert nominal.damping_mismatch_upper == nominal.phase_command_mismatch_upper == 0
    for mu, model in enumerate(nominal.model_bounds, start=1):
        assert model.multiplier == mu
        assert model.finite_identity_certified
        assert all(margin > 0 for _, margin in model.identity_margins)
        assert model.amplitude_lower < mu < model.amplitude_upper
        assert model.phase_residual_upper > 0
        assert model.corrected_increment_error_upper > 0
        assert model.sample_error_upper == model.forcing_residual_integral_upper == 0


def test_coefficient_boxes_enclose_every_shared_primitive_corner():
    primitive = {
        "capacitance_ratio": I(Q(7, 8), Q(9, 8)),
        "conductance_ratio": I(Q(15, 16), Q(17, 16)),
        "josephson_ratio": I(Q(31, 32), Q(33, 32)),
        "clock_ratio": I(Q(63, 64), Q(65, 64)),
    }
    report = owner._bound_organized_robust_current(**_parameters(**primitive))
    for C, G, A, D in product(*((v.lo, v.hi) for v in primitive.values())):
        assert _inside(report.lambda_bounds, G / (C * D))
        assert _inside(report.rho_bounds, 1 / (C * D))
        assert _inside(report.eta_bounds, 1 / (A * D))
        assert _inside(report.beta_bounds, A * G**2 / (C**2 * D))
        assert _inside(report.eta_rho_bounds, 1 / (A * C * D**2))
        assert report.damping_mismatch_upper >= abs(1 - G / (C * D))
        assert report.phase_command_mismatch_upper >= abs(1 / (A * C * D**2) - 1)


def test_whole_word_offset_variation_follows_disjoint_positive_ramps(nominal):
    # s'=6r(1-r)>=0 on [0,1] and its exact integral is one. For
    # k=e_c-1/5, ||k||inf=4/5. Integrate Lz=4*a*k*s'/gamma ramp by ramp.
    amplitude_integral = Q(0)
    for c in range(5):
        k = tuple(Q(i == c) - Q(1, 5) for i in range(5))
        for multiplier in (1, -2, 1):
            amplitude_integral += 4 * abs(Q(multiplier, 10)) * max(map(abs, k))
    assert (
        nominal.offset_laplacian_integral_upper
        == amplitude_integral / nominal.gamma_bounds.lo
    )


def test_synthetic_nonzero_errors_retain_strict_separation():
    ratio = I(1 - Q(1, 10**10), 1 + Q(1, 10**10))
    report = owner._bound_organized_robust_current(
        **_parameters(
            **{name: ratio for name in _RATIOS},
            total_tracking=Q(1, 10**6),
            plateau_tracking=Q(1, 10**17),
            raw_readout_error=Q(2, 10**15),
            aperture=Q(1, 10**15),
        )
    )
    assert report.admitted
    assert report.separation_margin > Q(1, 2)
    assert report.damping_mismatch_upper > 0
    assert report.phase_command_mismatch_upper > 0
    assert all(
        m.sample_error_upper > report.raw_readout_error for m in report.model_bounds
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"total_tracking": Q(1, 10**3)},
        {"total_tracking": Q(1, 10**3), "plateau_tracking": Q(1, 10**17)},
        {"raw_readout_error": Q(1, 10**15)},
        {"aperture": Q(1, 10**14)},
        {name: I(1 - Q(1, 10**9), 1 + Q(1, 10**9)) for name in _RATIOS},
    ),
)
def test_independent_uncertainty_only_widens_sufficient_envelopes(nominal, changes):
    report = owner._bound_organized_robust_current(**_parameters(**changes))
    for old, new in zip(nominal.model_bounds, report.model_bounds):
        assert new.observation_form_upper >= old.observation_form_upper
        assert new.phase_residual_upper >= old.phase_residual_upper
        assert (
            new.corrected_increment_error_upper >= old.corrected_increment_error_upper
        )
        assert new.amplitude_lower <= old.amplitude_lower
        assert new.amplitude_upper >= old.amplitude_upper
    assert report.separation_margin < nominal.separation_margin


def test_large_readout_error_makes_bound_unavailable_not_model_excluded():
    report = owner._bound_organized_robust_current(
        **_parameters(raw_readout_error=Q(1, 100))
    )
    assert not report.admitted
    assert report.status == "unavailable"
    assert report.unresolved_conditions == (
        "radial_current_envelopes_not_strictly_separated",
    )
    assert all(model.finite_identity_certified for model in report.model_bounds)
    assert all(model.amplitude_lower == 0 for model in report.model_bounds)


def test_large_tracking_exposes_identity_failures_as_well_as_overlap():
    report = owner._bound_organized_robust_current(
        **_parameters(total_tracking=Q(10**5))
    )
    assert not report.admitted
    assert not any(model.finite_identity_certified for model in report.model_bounds)
    assert "mu_1_outer_acute_gap" in report.unresolved_conditions
    assert "mu_2_circular_collar" in report.unresolved_conditions


def test_radial_triangle_allowance_and_prior_include_all_angles(nominal):
    with mpmath.workdps(65):
        gamma = 1 / (1023 * mpmath.pi)
        for model in nominal.model_bounds:
            error = _mp(model.corrected_increment_error_upper)
            cosine_error = (
                4
                * error
                / (gamma * mpmath.mpf(".00001") * mpmath.sin(mpmath.mpf(".1")))
            )
            sine_error = (
                8
                * error
                / (gamma * mpmath.mpf(".00001") * (1 - mpmath.cos(mpmath.mpf(".1"))))
            )
            expected = mpmath.sqrt(cosine_error**2 + sine_error**2)
            assert _mp(model.radial_error_upper) >= expected
            assert _mp(model.radial_error_upper) - expected < mpmath.mpf("1e-25")
            for angle in (mpmath.mpf("-1.4"), 0, mpmath.mpf("1.4")):
                for sx, sy in product((-1, 1), repeat=2):
                    # The resultant is between cos(2*zeta) and one. Angular
                    # displacement does not change its radial model contrast.
                    for modulus in (mpmath.cos(mpmath.mpf(".000002")), 1):
                        x = (
                            model.multiplier * modulus * mpmath.cos(angle)
                            + sx * cosine_error
                        )
                        y = (
                            model.multiplier * modulus * mpmath.sin(angle)
                            + sy * sine_error
                        )
                        radius = mpmath.sqrt(x * x + y * y)
                        assert (
                            _mp(model.amplitude_lower)
                            <= radius
                            <= _mp(model.amplitude_upper)
                        )


@pytest.mark.parametrize("name", _RATIOS)
@pytest.mark.parametrize(
    "invalid",
    (
        True,
        0,
        -1,
        float("nan"),
        float("inf"),
        "1",
        Decimal("1"),
        1j,
        I(0, 1),
        Q(1, 2**200),
    ),
)
def test_calibration_requires_resolved_strictly_positive_primitive(name, invalid):
    with pytest.raises((TypeError, ValueError)):
        owner._bound_organized_robust_current(**_parameters(**{name: invalid}))


@pytest.mark.parametrize("name", _ERRORS)
@pytest.mark.parametrize(
    "invalid", (True, -1, float("nan"), float("inf"), "0", 1j, I(0, 1))
)
def test_error_allowances_reject_invalid_or_unavailable_primitives(name, invalid):
    with pytest.raises((TypeError, ValueError)):
        owner._bound_organized_robust_current(**_parameters(**{name: invalid}))


def test_plateau_tracking_cannot_exceed_complete_tracking():
    with pytest.raises(ValueError, match="plateau_tracking"):
        owner._bound_organized_robust_current(**_parameters(plateau_tracking=Q(1, 100)))


def test_aperture_has_an_explicit_plateau_boundary():
    report = owner._bound_organized_robust_current(**_parameters(aperture=Q(1, 100000)))
    assert report.aperture == report.plateau_duration
    with pytest.raises(ValueError, match="aperture"):
        owner._bound_organized_robust_current(**_parameters(aperture=Q(1, 99999)))


def _readings():
    # Set each desired corrected increment independently. Since only the
    # receiver end coordinate changes, its coefficient is 1+h/4.
    records, desired, odds, evens = [], [], [], []
    for c in range(5):
        base = Q(c - 2, 10**11)
        odd, even = Q((1, 2, -3, 0, 5)[c], 10**10), Q((2, -1, 4, 3, 0)[c], 10**11)
        row = (base, base + (even + odd) / 2, base + (even - odd) / 2)
        block = []
        for target in row:
            end = [Q(0)] * 5
            end[(c + 1) % 5] = target / (1 + Q(1, 400000))
            block.append([[Q(0)] * 5, end])
        records.append(block)
        desired.append(row)
        odds.append(odd)
        evens.append(even)
    return records, tuple(desired), tuple(odds), tuple(evens)


def test_raw_statistic_recovers_independent_linear_observation_algebra():
    records, desired, odd, even = _readings()
    report = owner._organized_raw_current_quadratures(
        readings=records, nominal_current_scale=Q(7, 3)
    )
    assert report.corrected_plateau_increments == desired
    assert report.odd_reading_differences == odd
    assert report.even_reading_combinations == even
    with mpmath.workdps(65):
        gamma, h, eps, j = (
            1 / (1023 * mpmath.pi),
            mpmath.mpf(".00001"),
            mpmath.mpf(".1"),
            mpmath.mpf(7) / 3,
        )
        for i in range(5):
            x = 2 * j * _mp(odd[i]) / (gamma * h * mpmath.sin(eps))
            y = 2 * j * _mp(even[i]) / (gamma * h * (1 - mpmath.cos(eps)))
            radius = mpmath.sqrt(x * x + y * y)
            for enclosure, expected in (
                (report.cosine_current_bounds[i], x),
                (report.sine_current_bounds[i], y),
                (report.radial_current_bounds[i], radius),
            ):
                assert _mp(enclosure.lo) <= expected <= _mp(enclosure.hi)
                assert _mp(enclosure.width) < mpmath.mpf("1e-20")


def test_statistic_uses_one_supplied_scale_and_contains_its_uncertainty():
    records = _readings()[0]
    low = owner._organized_raw_current_quadratures(
        readings=records, nominal_current_scale=2
    )
    high = owner._organized_raw_current_quadratures(
        readings=records, nominal_current_scale=3
    )
    wide = owner._organized_raw_current_quadratures(
        readings=records, nominal_current_scale=I(2, 3)
    )
    for name in (
        "cosine_current_bounds",
        "sine_current_bounds",
        "radial_current_bounds",
    ):
        for a, b, c in zip(
            getattr(low, name), getattr(high, name), getattr(wide, name)
        ):
            assert c.contains(a)
            assert c.contains(b)
    assert (
        low.corrected_plateau_increments
        == high.corrected_plateau_increments
        == wide.corrected_plateau_increments
    )


def test_constant_floating_form_does_not_change_the_current_statistic():
    records = _readings()[0]
    shifted = [
        [[[value + Q(13, 7) for value in row] for row in plateau] for plateau in block]
        for block in records
    ]
    a = owner._organized_raw_current_quadratures(
        readings=records, nominal_current_scale=1
    )
    b = owner._organized_raw_current_quadratures(
        readings=shifted, nominal_current_scale=1
    )
    assert a.corrected_plateau_increments == b.corrected_plateau_increments
    assert a.radial_current_bounds == b.radial_current_bounds


def test_zero_records_have_exact_zero_statistic_without_inventing_evidence():
    records = [[[[0] * 5 for _ in range(2)] for _ in range(3)] for _ in range(5)]
    report = owner._organized_raw_current_quadratures(
        readings=records, nominal_current_scale=1
    )
    assert report.radial_current_bounds == (I(0),) * 5
    assert not hasattr(report, "admitted")
    assert not hasattr(report, "identified_model")


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("change", (-1, 1))
def test_reading_shape_is_exact_at_every_depth(depth, change):
    records = _readings()[0]
    row = records
    for _ in range(depth):
        row = row[0]
    if change == -1:
        row.pop()
    else:
        row.append(deepcopy(row[0]))
    with pytest.raises(ValueError):
        owner._organized_raw_current_quadratures(
            readings=records, nominal_current_scale=1
        )


@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), "0", 1j))
def test_every_record_is_admitted_before_scale_or_interval_arithmetic(
    invalid, monkeypatch
):
    records = _readings()[0]
    records[4][2][1][4] = invalid

    def forbidden(*args, **kwargs):
        pytest.fail("invalid late record reached scale admission")

    monkeypatch.setattr(owner, "_positive_box", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._organized_raw_current_quadratures(
            readings=records, nominal_current_scale=1
        )


@pytest.mark.parametrize(
    "scale", (True, 0, -1, float("nan"), float("inf"), "1", I(0, 1), Q(1, 2**200))
)
def test_current_units_require_positive_admitted_scale(scale):
    with pytest.raises((TypeError, ValueError)):
        owner._organized_raw_current_quadratures(
            readings=_readings()[0], nominal_current_scale=scale
        )


def test_cold_import_has_no_acquired_source_or_solver_dependency(
    source_tree_environment,
):
    root = Path(__file__).resolve().parents[2]
    script = (
        "import importlib.abc\n"
        "import sys\n"
        "forbidden = ('tnfr.physics._organized_electrical_source', "
        "'tnfr.physics._organized_joint_resources', "
        "'tnfr.physics._organized_finite_observation_source', "
        "'tnfr.research', 'tnfr.mathematics._validated_taylor')\n"
        "class NoSource(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.startswith(forbidden):\n"
        "            raise AssertionError('unexpected source or solver: ' + fullname)\n"
        "sys.meta_path.insert(0, NoSource())\n"
        "from tnfr.physics import _organized_robust_current as owner\n"
        "report = owner._bound_organized_robust_current(capacitance_ratio=1, "
        "conductance_ratio=1, josephson_ratio=1, clock_ratio=1, "
        "total_tracking=0, plateau_tracking=0, raw_readout_error=0, aperture=0)\n"
        "assert report.admitted\n"
        "assert not [name for name in sys.modules if '_organized' in name and 'source' in name]\n"
        "assert not any(name.startswith(forbidden) for name in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=source_tree_environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
