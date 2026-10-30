"""Response wiring controls; mocked curves are not scientific evidence.

The reserved C1024 integrand is never evaluated by this suite. Independent
spectral and quadrature mathematics are exercised by their respective owners.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _localized_mobile_response as owner


def _evaluate(**changes):
    return owner._evaluate_localized_mobile_response(
        **(
            dict(
                reference_model=RelationalExchangeModel(
                    storage_scale=1,
                    epi_weight=Q(1, 2**20),
                    phase_weight=1 - Q(1, 2**20),
                    phase_domain="regular",
                ),
                node_count=1024,
                fejer_order=16,
                amplitude=Q(1, 4096),
                initial_state_error_bound=Q(1, 4096000),
                localization_radius=96,
                horizon=256,
            )
            | changes
        )
    )


@pytest.fixture(autouse=True)
def block_reserved_integrand(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("a wiring test cannot evaluate the reserved integrand")

    monkeypatch.setattr(owner, "_prepare_mobile_spectral_kernel", forbidden)


def _mock_curve(monkeypatch, function):
    calls = []

    def factory(**primitives):
        def evaluate(time):
            calls.append(time)
            return ((function(time), I(0), I(0), I(0)),)

        return SimpleNamespace(modes=(1,)), evaluate

    monkeypatch.setattr(owner, "_prepare_mobile_spectral_kernel", factory)
    return calls


def test_frozen_prior_selects_panels_before_reading_a_mocked_polynomial(monkeypatch):
    scale = Q(1, 4096) ** 3
    calls = _mock_curve(monkeypatch, lambda time: scale * (time / 256) ** 4)
    progress = []
    report = _evaluate(progress=progress.append)
    exact_integral = scale * 256 / 5
    assert report.panels == 64
    assert len(calls) == report.integrand_evaluations == 64 * 12
    assert report.quadrature_norm_bounds.contains(exact_integral)
    assert report.cubic_norm_bounds.contains(exact_integral)
    assert report.numerical_error_upper_bound < report.numerical_error_target
    assert report.nominal_interaction_certified
    assert report.uniform_interaction_certified
    assert report.nominal_defect_norm_bounds.lo > report.uniform_defect_norm_bounds.lo
    assert progress == [
        {"completed_panels": panel, "panels": 64} for panel in range(1, 65)
    ]
    with pytest.raises(FrozenInstanceError):
        report.panels = 1


def test_zero_mocked_response_is_evaluated_without_certifying_interaction(monkeypatch):
    _mock_curve(monkeypatch, lambda time: I(0))
    report = _evaluate()
    assert report.status == "evaluated"
    assert report.numerical_certified
    assert report.cubic_norm_bounds.contains(0)
    assert report.nominal_defect_norm_bounds.lo == 0
    assert report.uniform_defect_norm_bounds.lo == 0
    assert not report.nominal_interaction_certified
    assert not report.uniform_interaction_certified


def test_nominal_and_uniform_source_verdicts_remain_distinct(monkeypatch):
    # This artificial integral falls between the two predeclared thresholds.
    _mock_curve(monkeypatch, lambda time: I(Q(1, 10**12 * 256)))
    report = _evaluate()
    assert report.nominal_interaction_certified
    assert not report.uniform_interaction_certified
    assert report.uniform_defect_norm_bounds.lo == 0


def test_wide_arithmetic_never_certifies_even_a_positive_mocked_lower_bound(
    monkeypatch,
):
    _mock_curve(monkeypatch, lambda time: I(Q(1), Q(2)))
    report = _evaluate()
    assert report.nominal_defect_norm_bounds.lo > 0
    assert not report.numerical_certified
    assert not report.nominal_interaction_certified
    assert not report.uniform_interaction_certified
    assert report.status == "unavailable"
    assert "interval_arithmetic_budget_exhausted" in report.unavailable_reasons


def test_failed_prior_and_panel_limit_do_not_evaluate_an_integrand():
    report = _evaluate(max_panels=1)
    assert report.integrand_evaluations == 0
    assert report.cubic_norm_bounds is None
    assert report.unavailable_reasons == ("quadrature_panel_budget_exhausted",)
    report = _evaluate(amplitude=1)
    assert report.cubic_norm_bounds is None
    assert report.unavailable_reasons == ("analytic_response_budget_not_certified",)


def test_callback_failure_propagates_without_an_internal_retry(monkeypatch):
    calls = _mock_curve(monkeypatch, lambda time: I(0))

    def failed_progress(progress):
        raise RuntimeError("retained attempt must record this failure")

    with pytest.raises(RuntimeError, match="retained attempt"):
        _evaluate(progress=failed_progress)
    assert len(calls) == 12


@pytest.mark.parametrize("bad", [True, 1.0, Q(1), None])
def test_panel_limit_type_is_admitted_before_evaluation(bad):
    with pytest.raises(TypeError):
        _evaluate(max_panels=bad)


@pytest.mark.parametrize("bad", [0, -1, 65537])
def test_panel_limit_cannot_escape_the_frozen_work_budget(bad):
    with pytest.raises(ValueError):
        _evaluate(max_panels=bad)


def test_norm_reduction_retains_both_real_channels_and_complex_components():
    norm = owner._spectral_norm_bounds(((I(3), I(4), I(12), I(0)),))
    assert norm.contains(13)
    assert owner._expand_norm(norm, Q(14)).lo == 0
