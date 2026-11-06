"""Analytic kernels and fake-solver wiring; never run the frozen search."""

from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import organized_contact_stationary_search as owner
from tnfr.utils.io import json_loads


def test_two_node_fields_have_independent_analytic_derivatives():
    value, gradient, hessian = owner._fields(np.array([np.pi / 6]), np.array([[1.0]]))
    assert value == pytest.approx(1 - np.sqrt(3) / 2)
    assert gradient == pytest.approx([0.5])
    assert hessian == pytest.approx(np.array([[np.sqrt(3) / 2]]))


def test_edge_orientation_does_not_change_potential_derivatives():
    phase = np.array([0.2, -0.4])
    incidence = np.array([[1.0, 0.0], [-1.0, 1.0]])
    direct = owner._fields(phase, incidence)
    reversed_edges = owner._fields(phase, -incidence)
    for left, right in zip(direct, reversed_edges):
        assert left == pytest.approx(right)


def _fake_fields(phase, incidence, *, hessian=True):
    return 0.0, np.zeros_like(phase), np.eye(len(phase))


def test_oracle_enforces_budget_and_detaches_last_evaluated_phase(monkeypatch):
    calls = []

    def fields(phase, incidence, *, hessian=True):
        calls.append(phase.copy())
        return _fake_fields(phase, incidence)

    monkeypatch.setattr(owner, "_fields", fields)
    oracle = owner._Oracle(np.array([[1.0]]), 2)
    phase = np.array([0.25])
    oracle.value_gradient(phase)
    phase[0] = 0.5
    assert oracle.last_phase == pytest.approx([0.25])
    oracle.residual(phase)
    with pytest.raises(owner._BudgetExhausted):
        oracle.value_gradient(np.array([0.75]))
    assert len(calls) == oracle.calls == 2
    assert oracle.last_phase == pytest.approx([0.5])


def test_jacobian_has_its_own_hard_budget(monkeypatch):
    monkeypatch.setattr(owner, "_fields", _fake_fields)
    oracle = owner._Oracle(np.array([[1.0]]), 1)
    oracle.jacobian(np.array([0.25]))
    with pytest.raises(owner._BudgetExhausted):
        oracle.jacobian(np.array([0.5]))
    assert oracle.jacobian_calls == 1
    assert oracle.calls == 0


def _endpoint(*, residual=0.0, slack=1.0, eigenvalue=1.0):
    return {
        "phase_hex": [float(0).hex(), float(0.25).hex()],
        "potential": 0.0,
        "gradient_infinity_norm": residual,
        "tightened_chart_slack": slack,
        "grounded_hessian_minimum_eigenvalue": eigenvalue,
    }


@pytest.mark.parametrize(
    "residual,slack,accepted",
    [(0.0, 1.0, True), (1e-7, 1.0, False), (0.0, 2e-7, False)],
)
def test_screen_requires_strict_residual_and_chart_margins(residual, slack, accepted):
    assert owner._screen(_endpoint(residual=residual, slack=slack)) is accepted


def test_nonfinite_endpoint_is_rejected_before_fields(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("nonfinite endpoint reached numerical fields")

    monkeypatch.setattr(owner, "_fields", forbidden)
    with pytest.raises(ValueError, match="nonfinite optimizer endpoint"):
        owner._endpoint(np.array([np.nan]), np.array([[1.0]]), np.array([0]))


def _fake_problem(monkeypatch, endpoints):
    iterator = iter(endpoints)
    monkeypatch.setattr(owner, "_endpoint", lambda *args: next(iterator))
    monkeypatch.setattr(owner, "_fields", _fake_fields)
    return np.array([0.25]), np.array([[1.0]]), np.array([-1]), ((0, 1),)


def test_successful_boundary_optimizer_is_not_polished(monkeypatch):
    arguments = _fake_problem(monkeypatch, [_endpoint(), _endpoint(slack=0.0)])

    def minimize(function, start, *, method, jac, constraints, options):
        assert method == "SLSQP"
        assert jac is True
        assert options == {"ftol": 1e-12, "maxiter": 500}
        assert constraints.A == pytest.approx(np.array([[1.0]]))
        limit = np.pi - float(owner._MARGIN)
        assert constraints.lb == pytest.approx([-2 * np.pi - limit])
        assert constraints.ub == pytest.approx([-2 * np.pi + limit])
        function(start)
        return SimpleNamespace(x=start, success=True, status=0, message="fake", nit=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("a boundary endpoint cannot trigger root/certification")

    monkeypatch.setattr(owner, "minimize", minimize)
    monkeypatch.setattr(owner, "root", forbidden)
    monkeypatch.setattr(owner, "_certificate", forbidden)
    report = owner._attempt(*arguments)
    assert report["optimizer"]["success"]
    assert report["value_gradient_calls"] == 1
    assert report["polish"] is None
    assert report["certificate"] is None


def test_solver_failure_labels_do_not_override_independent_certificate(monkeypatch):
    arguments = _fake_problem(monkeypatch, [_endpoint(), _endpoint(), _endpoint()])

    def minimize(function, start, **kwargs):
        function(start)
        return SimpleNamespace(x=start, success=False, status=9, message="fake", nit=1)

    def root(function, phase, *, jac, method, options):
        assert method == "hybr"
        assert options == {"xtol": 1e-10, "maxfev": 600}
        function(phase)
        jac(phase)
        return SimpleNamespace(x=phase, success=False, message="fake")

    calls = []

    def certificate(*args):
        calls.append(args)
        return {"stable_identity_certified": True}

    monkeypatch.setattr(owner, "minimize", minimize)
    monkeypatch.setattr(owner, "root", root)
    monkeypatch.setattr(owner, "_certificate", certificate)
    report = owner._attempt(*arguments)
    assert not report["optimizer"]["success"]
    assert not report["polish"]["success"]
    assert report["polish"]["residual_calls"] == 1
    assert report["polish"]["jacobian_calls"] == 1
    assert report["certificate"]["stable_identity_certified"]
    assert len(calls) == 1


def test_optimizer_exhaustion_retains_last_call_without_retry(monkeypatch):
    arguments = _fake_problem(monkeypatch, [_endpoint(), _endpoint(residual=1.0)])
    invocations = []

    def minimize(function, start, **kwargs):
        invocations.append(1)
        for _ in range(5001):
            function(start)
        raise AssertionError("the hard oracle budget did not stop the fake solver")

    monkeypatch.setattr(owner, "minimize", minimize)
    report = owner._attempt(*arguments)
    assert invocations == [1]
    assert report["value_gradient_calls"] == 5000
    assert not report["optimizer"]["success"]
    assert "budget exhausted" in report["optimizer"]["message"]
    assert report["polish"] is None


def test_polished_nonpositive_curvature_is_not_a_certificate(monkeypatch):
    arguments = _fake_problem(
        monkeypatch, [_endpoint(), _endpoint(), _endpoint(eigenvalue=0.0)]
    )
    monkeypatch.setattr(
        owner,
        "minimize",
        lambda function, start, **kwargs: SimpleNamespace(
            x=start, success=True, status=0, message="fake", nit=0
        ),
    )
    monkeypatch.setattr(
        owner,
        "root",
        lambda function, phase, **kwargs: SimpleNamespace(
            x=phase, success=True, message="fake"
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("nonpositive curvature cannot pass the numeric screen")

    monkeypatch.setattr(owner, "_certificate", forbidden)
    report = owner._attempt(*arguments)
    assert report["certificate"] is None
    assert report["polish"]["endpoint"]["grounded_hessian_minimum_eigenvalue"] == 0


@pytest.mark.parametrize("branch,admitted", [(0, True), (-1, False)])
def test_certificate_wiring_readmits_exact_coordinates_and_branch(
    monkeypatch, branch, admitted
):
    captured = []

    def certificate(**arguments):
        captured.append(arguments)
        return SimpleNamespace(
            eta=Q(0),
            q0=Q(0),
            qbox=Q(1, 1000),
            root_inclusion_certified=True,
            relative_hessian_positive=True,
            stable_stationary_certified=True,
        )

    monkeypatch.setattr(owner, "_fields", _fake_fields)
    monkeypatch.setattr(owner, "_certify_organized_stationary", certificate)
    result = owner._certificate(
        np.array([0.1]), np.array([[1.0]]), np.array([branch]), ((0, 1),)
    )
    assert captured[0]["phase_center"] == (Q(0), Q.from_float(0.1))
    assert captured[0]["radius"] == Q(1, 10**7)
    assert captured[0]["inverse"] == ((Q(1),),)
    assert result["root_inclusion"]
    assert result["positive_definite"]
    assert result["marked_chamber_certified"] is admitted
    assert result["stable_identity_certified"] is admitted


def _fake_main(monkeypatch, output):
    monkeypatch.setattr(owner.sys, "argv", ["search", "--output", str(output)])
    monkeypatch.setattr(
        owner, "_git", lambda *args: "" if args[0] == "status" else "a" * 40
    )
    monkeypatch.setattr(owner.np, "show_config", lambda: print("fake numeric backend"))
    monkeypatch.setattr(
        owner,
        "_geometry",
        lambda: (((0, 1),), np.array([[1.0]]), np.array([0])),
    )
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")


def test_main_retains_records_and_stops_at_first_fake_certificate(
    monkeypatch, tmp_path
):
    output = tmp_path / "evidence"
    _fake_main(monkeypatch, output)

    def starts(*args):
        yield 0, 0.0, 1.0, np.array([0.25])
        raise AssertionError(
            "a successful certificate must stop further initialization"
        )

    def attempt(*args):
        assert (output / "attempt.json").exists()
        assert (output / "start-00.json").exists()
        return {
            "endpoint": _endpoint(),
            "certificate": {"stable_identity_certified": True},
        }

    monkeypatch.setattr(owner, "_starts", starts)
    monkeypatch.setattr(owner, "_attempt", attempt)
    owner.main()
    metadata = json_loads((output / "attempt.json").read_bytes())
    start = json_loads((output / "start-00.json").read_bytes())
    candidate = json_loads((output / "candidate-00.json").read_bytes())
    outcome = json_loads((output / "outcome.json").read_bytes())
    assert metadata["source_revision"] == "a" * 40
    assert metadata["thread_environment"]["OPENBLAS_NUM_THREADS"] == "1"
    assert metadata["evaluation_status_at_start"] == "not_evaluated"
    assert metadata["imported_sources"]
    assert start["phase_hex"] == [float(0).hex(), float(0.25).hex()]
    assert candidate["certificate"]["stable_identity_certified"]
    assert outcome == {
        "verdict": "certified_stable_stationary_identity",
        "completed": [0],
    }
    assert not (output / "candidate-01.json").exists()


def test_main_keeps_start_and_partial_record_on_exception(monkeypatch, tmp_path):
    output = tmp_path / "incomplete"
    _fake_main(monkeypatch, output)
    monkeypatch.setattr(
        owner, "_starts", lambda *args: iter([(0, 0.0, 1.0, np.array([0.25]))])
    )

    def attempt(*args):
        error = RuntimeError("fake independent certificate failure")
        error.partial_record = {"endpoint": _endpoint(), "certificate": None}
        raise error

    monkeypatch.setattr(owner, "_attempt", attempt)
    with pytest.raises(RuntimeError, match="fake independent certificate failure"):
        owner.main()
    record = json_loads((output / "incomplete.json").read_bytes())
    assert record["completed"] == []
    assert record["partial_record"]["endpoint"]["phase_hex"][1] == float(0.25).hex()
    assert (output / "start-00.json").exists()
    assert not (output / "outcome.json").exists()


def test_dirty_source_rejects_before_geometry_or_initialization(monkeypatch, tmp_path):
    output = tmp_path / "unused"
    _fake_main(monkeypatch, output)
    monkeypatch.setattr(owner, "_git", lambda *args: " M declaration.md")

    def forbidden(*args):
        raise AssertionError("dirty source must not produce a candidate")

    monkeypatch.setattr(owner, "_geometry", forbidden)
    monkeypatch.setattr(owner, "_starts", forbidden)
    with pytest.raises(RuntimeError, match="clean commit"):
        owner.main()
    assert not output.exists()
