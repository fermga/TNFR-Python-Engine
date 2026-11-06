"""Independent algebra and synthetic wiring; no declared record is fitted."""

from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_forecast_proposal as proposal
from tnfr.physics._organized_computable_prediction import _ForecastStage


def _synthetic_inputs():
    matrix = tuple(
        tuple(I(int(row % 15 == column)) for column in range(15)) for row in range(150)
    )
    return proposal._reading_times(), (Q(0),) * 150, matrix


def test_exact_candidate_has_unknown_common_form_and_exact_boundary():
    candidate = proposal._exact_candidate([0.01] * 5 + [0.2, -0.3, 4.0, -5.0])
    assert candidate[:5] == (Q(1, 100),) * 5
    assert sum(candidate[:5]) == Q(1, 20)
    assert candidate[5] == 0
    assert candidate[6:] == tuple(Q.from_float(x) for x in (0.2, -0.3, 4.0, -5.0))
    assert Q.from_float(0.01) > candidate[0]


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), Q(1, 10**1000)])
def test_input_admission_precedes_numpy_materialization(bad, monkeypatch):
    times, values, matrix = _synthetic_inputs()
    values = (bad,) + values[1:]

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached NumPy least squares")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    with pytest.raises((TypeError, ValueError)):
        proposal.propose_candidate(times, values, matrix)


def test_record_schedule_and_matrix_are_complete_and_exact():
    times, values, matrix = _synthetic_inputs()
    with pytest.raises(ValueError, match="schedule"):
        proposal.propose_candidate(times[:-1] + (Q(215),), values, matrix)
    with pytest.raises(ValueError, match="150"):
        proposal.propose_candidate(times, values[:-1], matrix)
    with pytest.raises(ValueError, match="15"):
        proposal.propose_candidate(times, values, (matrix[0][:-1],) + matrix[1:])
    with pytest.raises(TypeError):
        proposal.propose_candidate(
            times, values, ((True,) + matrix[0][1:],) + matrix[1:]
        )


@pytest.mark.parametrize("container", [dict.fromkeys, set, frozenset])
def test_unordered_record_containers_and_matrix_rows_reject(container, monkeypatch):
    times, values, matrix = _synthetic_inputs()

    def forbidden(*args, **kwargs):
        pytest.fail("unordered values reached NumPy materialization")

    monkeypatch.setattr(np, "array", forbidden)
    with pytest.raises(TypeError, match="ordered"):
        proposal.propose_candidate(container(times), values, matrix)
    with pytest.raises(TypeError, match="ordered"):
        proposal.propose_candidate(times, container(range(150)), matrix)
    with pytest.raises(TypeError, match="ordered"):
        proposal.propose_candidate(times, values, container(range(150)))
    with pytest.raises(TypeError, match="ordered"):
        proposal.propose_candidate(times, values, (container(range(15)),) + matrix[1:])


def test_backend_retains_numeric_build_facts_without_paths_or_commands():
    from tnfr.utils.io import json_loads

    raw = {
        "Compilers": {
            "c": {"name": "gcc", "version": "15", "commands": "private-path"}
        },
        "Machine Information": {"host": {"cpu": "x86_64", "system": "windows"}},
        "Build Dependencies": {
            "blas": {
                "name": "openblas",
                "version": "0.3.31",
                "openblas configuration": "DYNAMIC_ARCH USE64BITINT",
                "has ilp64": True,
                "include directory": "private-path",
            }
        },
        "Python Information": {"path": "private-path"},
        "SIMD Extensions": {"baseline": ["X86_V2"], "found": ["X86_V3"]},
        "Environment": {"private-token": "secret"},
    }
    module = SimpleNamespace(__version__="example", show_config=lambda **kwargs: raw)
    encoded = proposal._backend_provenance(module, module)
    assert encoded == proposal._backend_provenance(module, module)
    decoded = json_loads(encoded)
    assert decoded["arithmetic"] == "binary64"
    assert decoded["numpy"]["linear_algebra"]["blas"]["has ilp64"] is True
    assert decoded["scipy"]["compilers"]["c"] == {"name": "gcc", "version": "15"}
    assert decoded["numpy"]["simd"]["found"] == ["X86_V3"]
    assert "private" not in encoded and "secret" not in encoded


def test_numeric_field_preserves_degree_four_and_both_clock_rows():
    stage = _ForecastStage(Q(200), Q(1), "plateau")
    evaluate = proposal._numeric_carrier_field(stage)
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 5)
    rate, jacobian = evaluate(0.0, state)
    gradient = np.array([2.0, -1.0, 0.0, 0.0, -1.0])
    gamma = 1 / (1023 * np.pi)
    np.testing.assert_allclose(rate[:5], -gradient / 4, rtol=0, atol=1e-16)
    np.testing.assert_allclose(rate[5:], gamma * gradient / 4, rtol=1e-15)
    # Both conserved common coordinates remain true null directions.
    np.testing.assert_allclose(jacobian @ np.array([1.0] * 5 + [0.0] * 5), 0)
    np.testing.assert_allclose(jacobian @ np.array([0.0] * 5 + [1.0] * 5), 0)


def test_numeric_sine_orientation_and_analytic_jacobian():
    stage = _ForecastStage(Q(200), Q(1), "ramp", 2, Q(0), Q(1, 10))
    evaluate = proposal._numeric_carrier_field(stage)
    state = np.array([0.003, -0.002, 0.0, 0.001, 0.001, 0.0, 0.2, -0.4, 0.3, -0.1])
    _, jacobian = evaluate(0.25, state)
    epsilon = 1e-6
    finite_difference = np.column_stack(
        [
            (
                evaluate(0.25, state + epsilon * direction)[0]
                - evaluate(0.25, state - epsilon * direction)[0]
            )
            / (2 * epsilon)
            for direction in np.eye(10)
        ]
    )
    np.testing.assert_allclose(jacobian, finite_difference, rtol=1e-7, atol=2e-13)
    zero = np.zeros(10)
    zero[6] = np.pi / 2
    static = proposal._numeric_carrier_field(_ForecastStage(Q(200), Q(1), "plateau"))
    rate = static(0.0, zero)[0]
    np.testing.assert_allclose(
        rate[:5], np.array([1, -2, 1, 0, 0]) / (4 * 1023 * np.pi), atol=1e-19
    )


def test_historical_solver_carries_state_and_never_integrates_final_ramp(monkeypatch):
    import scipy.integrate

    calls = []

    def synthetic_solver(function, span, initial, **kwargs):
        calls.append((span, initial.copy(), kwargs))
        endpoint = initial.copy()
        endpoint[:10] += span[1]
        return SimpleNamespace(success=True, y=endpoint[:, None])

    monkeypatch.setattr(scipy.integrate, "solve_ivp", synthetic_solver)
    samples, sensitivities = proposal._historical_response(np.zeros(9))
    assert len(calls) == 29
    assert sum(span[1] for span, _, _ in calls) == pytest.approx(14.00015)
    np.testing.assert_allclose(samples[-5:], 14.00015)
    for (span, initial, kwargs), (_, following, _) in zip(calls, calls[1:]):
        np.testing.assert_allclose(following[:10], initial[:10] + span[1])
        assert span[0] == 0
        assert kwargs == dict(
            method="DOP853", rtol=1e-11, atol=1e-17, max_step=1 / 8, t_eval=[span[1]]
        )
    assert sensitivities.shape == (150, 9)
    np.testing.assert_allclose(sensitivities[:5, :5], np.eye(5))
    np.testing.assert_allclose(sensitivities[:5, 5:], 0)


def test_single_fit_uses_record_only_and_caches_matching_residual_jacobian(monkeypatch):
    import scipy.optimize

    times, values, matrix = _synthetic_inputs()
    fitted = []
    forwarded = []

    def response(parameters):
        forwarded.append(parameters.copy())
        return np.full(150, 2e-15), np.full((150, 9), 3e-5)

    def optimizer(function, initial, **kwargs):
        fitted.append((initial.copy(), kwargs))
        residual = function(initial)
        np.testing.assert_allclose(kwargs["jac"](initial.copy()), 0.3)
        return SimpleNamespace(
            success=True,
            status=3,
            nfev=1,
            x=np.array([0.01] * 5 + [0.2] * 4),
            fun=residual,
        )

    monkeypatch.setattr(proposal, "_historical_response", response)
    monkeypatch.setattr(scipy.optimize, "least_squares", optimizer)
    result = proposal.propose_candidate(times, values, matrix)
    assert len(fitted) == len(forwarded) == result.forward_calls == 1
    settings = fitted[0][1]
    assert settings["max_nfev"] == 80
    assert settings["ftol"] is settings["gtol"] is None
    assert settings["xtol"] == 1e-13
    assert settings["method"] == "trf"
    np.testing.assert_allclose(settings["x_scale"], [0.01] * 5 + [1.0] * 4)
    assert result.initial_candidate[:5] == (Q(1, 100),) * 5
    assert result.initial_candidate[5] == 0
    assert result.maximum_proposal_residual == Q.from_float(2e-15)
    assert result.reason is None


def test_nonconvergence_is_unavailable_without_restart(monkeypatch):
    import scipy.optimize

    calls = []

    def optimizer(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(
            success=False, status=0, nfev=80, fun=np.ones(150), x=np.zeros(9)
        )

    monkeypatch.setattr(scipy.optimize, "least_squares", optimizer)
    result = proposal.propose_candidate(*_synthetic_inputs())
    assert len(calls) == 1
    assert result.initial_candidate is None
    assert result.reason == "optimizer_did_not_converge"
    assert result.nfev == 80


def test_initializer_uses_oriented_phasors_and_raw_first_forms(monkeypatch):
    import scipy.optimize

    times, values, matrix = _synthetic_inputs()
    values = (Q(12, 1000), Q(-11, 1000), Q(3, 1000), Q(4, 1000), Q(5, 1000)) + values[
        5:
    ]
    coefficients = np.array([9.0] * 5 + [0, 1, -1, 0, 0] + [1, 0, 0, -1, 1])

    def linear_fit(*args, **kwargs):
        assert kwargs == {"rcond": None}
        return coefficients, None, 15, None

    def optimizer(function, initial, **kwargs):
        np.testing.assert_allclose(initial[:5], [0.01, -0.01, 0.003, 0.004, 0.005])
        np.testing.assert_allclose(initial[5:], [0, np.pi / 2, 0, np.pi])
        return SimpleNamespace(
            success=True, status=3, nfev=1, fun=np.zeros(150), x=initial
        )

    monkeypatch.setattr(np.linalg, "lstsq", linear_fit)
    monkeypatch.setattr(scipy.optimize, "least_squares", optimizer)
    result = proposal.propose_candidate(times, values, matrix)
    assert result.proposed_start[:2] == (Q(1, 100), -Q(1, 100))
    assert result.initial_candidate == result.proposed_start


def test_singular_initializer_does_not_invent_a_fallback(monkeypatch):
    import scipy.optimize

    times, values, _ = _synthetic_inputs()

    def forbidden(*args, **kwargs):
        pytest.fail("singular matrix triggered an alternative fit")

    monkeypatch.setattr(scipy.optimize, "least_squares", forbidden)
    result = proposal.propose_candidate(times, values, ((Q(0),) * 15,) * 150)
    assert result.initial_candidate is None
    assert result.reason == "static phasor initialization unavailable"
