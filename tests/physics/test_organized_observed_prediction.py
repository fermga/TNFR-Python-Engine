"""Static controls; never evaluate the declared coefficient matrix or sources."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.physics._organized_computable_prediction import (
    _ForecastStage,
    _forecast_stages,
)
from tnfr.physics._organized_observed_prediction import (
    _StaticDesign,
    _certify_dual,
    _edge_forcing,
    _observation_budget,
    _rational_proposal,
)


def test_source_budget_rebuilt_from_hidden_integral_inequalities():
    budget = _observation_budget()
    g, t = Q(1, 3000), Q(16)
    a0, z0 = Q(31, 10**9), Q(601, 10**10)
    assert budget.hidden_form_upper == a0 + 2 * g * t * budget.hidden_phase_upper
    assert budget.hidden_phase_upper == z0 + 2 * g * t * budget.hidden_form_upper
    assert (
        budget.source_error * (1 - (g * t) ** 2)
        == t * g * budget.hidden_phase_upper**2 / 9
    )
    assert 0 < budget.source_error < Q(22, 10**19)
    assert budget.historical_pair_error == 2 * (Q(1, 10**14) + budget.source_error)
    assert budget.remainder_phasor_factor < Q(37, 10**5)


def test_static_forcing_orientation_and_zero_sum():
    stage = _ForecastStage(Q(0), Q(1), "ramp", 2, Q(0), Q(1, 10))
    columns = _edge_forcing(stage, Jet((I(0), I(1), I(0), I(0))))
    for edge, column in enumerate(columns[:5]):
        assert column[edge].coeffs[0] == I(1)
        assert column[(edge + 1) % 5].coeffs[0] == I(-1)
    for column in columns:
        for degree in range(4):
            total = sum((x.coeffs[degree] for x in column), I(0))
            assert total.lo <= 0 <= total.hi
    # H2-H1 has second normalized derivative +.3; force is -B sin(B'H).
    assert columns[6][1].coeffs[2].lo <= Q(3, 10) <= columns[6][1].coeffs[2].hi
    assert columns[7][2].coeffs[2].lo <= Q(-3, 10) <= columns[7][2].coeffs[2].hi


def test_read_schedule_retains_joint_record_and_last_unobserved_ramp():
    times = tuple(
        t
        for stage in _forecast_stages()[1:]
        if stage.name == "plateau"
        for t in (stage.start, stage.start + stage.duration)
    )
    assert len(times) * 5 == 150
    assert times[0] == 200
    assert times[-1] == Q(21400015, 100000)
    assert max(b - a for a, b in zip(times, times[1:] + (Q(216),))) < 2
    assert _forecast_stages()[-2].name == "ramp"


def _independent_design():
    # Ten repetitions of an exact identity observation, not the scientific design.
    observations = tuple(
        tuple(I(int(j == i % 15)) for j in range(15)) for i in range(150)
    )
    future = tuple(tuple(I(int(i == j)) for j in range(15)) for i in range(5))
    phase = tuple(tuple(Q(int(j == i + 5)) for j in range(150)) for i in range(10))
    weights = tuple(tuple(Q(int(j == i)) for j in range(150)) for i in range(5))
    return (
        _StaticDesign((), observations, future, 0, Q(0), Q(0), True, None),
        phase,
        weights,
    )


def test_dual_certificate_rebuilds_residuals_and_does_not_trust_status():
    from dataclasses import replace

    design, phase, weights = _independent_design()
    certificate = _certify_dual(replace(design, admitted=False), phase, weights)
    assert certificate.admitted
    assert certificate.inverse_initial_defect == certificate.inverse_phasor_defect == 0
    assert certificate.future_initial_defect == certificate.future_phasor_defect == 0
    assert certificate.phase_weight_norm == certificate.future_weight_norm == 1
    assert certificate.fine_future_diameter_upper < Q(21, 10**15)
    changed = tuple(tuple(I(0) for _ in range(15)) for _ in range(150))
    unavailable = _certify_dual(
        replace(design, observation_matrix=changed), phase, weights
    )
    assert not unavailable.admitted
    assert unavailable.reason == "nonpositive_bootstrap_margin"
    assert unavailable.fine_future_diameter_upper is None


def test_dual_rejects_boolean_and_incomplete_weights():
    design, phase, weights = _independent_design()
    with pytest.raises(ValueError):
        _certify_dual(design, phase[:-1], weights)
    bad = ((True,) + phase[0][1:],) + phase[1:]
    with pytest.raises((ValueError, TypeError)):
        _certify_dual(design, bad, weights)


def test_proposal_grid_is_explicit_and_nonfinite_is_unavailable():
    assert _rational_proposal([[0.5, -0.25]]) == ((Q(1, 2), Q(-1, 4)),)
    with pytest.raises(ArithmeticError):
        _rational_proposal([[float("nan")]])


def test_benchmark_wires_fixed_assessor_without_calling_it(monkeypatch, tmp_path):
    from benchmarks import organized_observed_prediction as driver

    captured = {}

    def record(**kwargs):
        captured.update(kwargs)
        return "synthetic"

    monkeypatch.setattr(driver, "_run_single_analytic_assessment", record)
    assert driver.main(["--output", str(tmp_path)]) == 0
    assert captured["schema_name"] == "organized_observed_prediction"
    assert captured["assess"] is driver._assess
    assert captured["declaration_path"].endswith("ORGANIZED_OBSERVED_PREDICTION.md")
