"""Prospective source-relative contrast response under the fixed F4 law.

The symbolic controls derive the missing quadrature independently of the
benchmark's predictor. The finite experiment uses the frozen preparation,
shared pressure, nodal integration and regional observation.
Neither the source preparation nor this model check identifies a physical law.
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from fractions import Fraction as Q

import pytest

from benchmarks import source_relative_form_response as campaign


@pytest.fixture(scope="module")
def evaluated_prediction():
    # One four-trajectory producer serves all finite assertions. Analytic and
    # rational-reference predictions are prepared before the engine executes.
    return campaign.evaluate_prediction(campaign.prepare_prediction())


def test_source_relative_quadratures_close_the_fixed_affine_contrast_law():
    s = pytest.importorskip("sympy")
    u, v, cu, cv, decay, rotation = s.symbols("u v cu cv decay rotation", real=True)
    state = s.Matrix((u, v))
    velocity = s.Matrix((decay * u - rotation * v + cu, rotation * u + decay * v + cv))
    intensity = u**2 + v**2
    radial = cu * u + cv * v
    quadrature = cu * v - cv * u
    source_intensity = cu**2 + cv**2
    observables = s.Matrix((intensity, radial, quadrature))
    observed = observables.jacobian(state) * velocity
    required = s.Matrix(
        (
            2 * decay * intensity + 2 * radial,
            decay * radial - rotation * quadrature + source_intensity,
            rotation * radial + decay * quadrature,
        )
    )
    assert s.expand(observed - required) == s.zeros(3, 1)
    assert s.expand(radial**2 + quadrature**2 - source_intensity * intensity) == 0
    second_rate = s.Matrix((observed[0],)).jacobian(state) * velocity
    assert (
        s.expand(
            second_rate[0]
            - 4 * decay**2 * intensity
            - 6 * decay * radial
            + 2 * rotation * quadrature
            - 2 * source_intensity
        )
        == 0
    )
    # Intensity and its first derivative determine radial source work, but
    # do not determine the signed transverse source-relative coordinate.


def test_reflected_affine_response_has_equal_initial_rate_and_different_curvature():
    s = pytest.importorskip("sympy")
    decay, rotation = s.symbols("decay rotation", real=True)
    longitudinal, transverse, force = s.symbols(
        "longitudinal transverse force", real=True
    )
    time = s.symbols("time", real=True)
    denominator = decay**2 + rotation**2
    exponential = s.exp(decay * time)
    cosine, sine = s.cos(rotation * time), s.sin(rotation * time)
    rotation_matrix = s.Matrix(((cosine, -sine), (sine, cosine)))
    # This is the integral of exp((decay+i*rotation)*s)*force, expressed
    # in Cartesian coordinates; no engine response supplies its source.
    driven = force * s.Matrix(
        (
            (exponential * (decay * cosine + rotation * sine) - decay) / denominator,
            (exponential * (decay * sine - rotation * cosine) + rotation) / denominator,
        )
    )
    generator = s.Matrix(((decay, -rotation), (rotation, decay)))
    assert s.simplify(driven.diff(time) - generator * driven) == s.Matrix((force, 0))
    assert driven.subs(time, 0) == s.zeros(2, 1)
    plus = exponential * rotation_matrix * s.Matrix((longitudinal, transverse)) + driven
    minus = (
        exponential * rotation_matrix * s.Matrix((longitudinal, -transverse)) + driven
    )
    difference = s.simplify(plus.dot(plus) - minus.dot(minus))
    expected = (
        4
        * force
        * transverse
        * exponential
        * (decay * sine - rotation * (exponential - cosine))
        / denominator
    )
    assert s.simplify(difference - expected) == 0
    assert difference.subs(time, 0) == 0
    assert s.simplify(difference.diff(time).subs(time, 0)) == 0
    assert (
        s.simplify(difference.diff(time, 2).subs(time, 0))
        == -4 * force * transverse * rotation
    )
    assert s.simplify(difference.subs(force, 0)) == 0
    assert s.simplify(difference.subs(rotation, 0)) == 0
    assert s.simplify(difference.subs(transverse, 0)) == 0


def test_directed_triangle_pair_specialization_binds_fine_state_and_source():
    s = pytest.importorskip("sympy")
    amplitude, q = s.symbols("amplitude q", positive=True)
    mean = s.symbols("mean", real=True)
    shift = s.Matrix(((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    adjacency = s.BlockMatrix(((shift, s.eye(3)), (s.eye(3), shift))).as_explicit()
    generator = adjacency / 4 - s.eye(6) / 2
    source = s.Matrix((2 * q, -q, -q) * 2)
    plus = s.Matrix((mean, mean + amplitude, mean - amplitude) * 2)
    minus = s.Matrix((mean, mean - amplitude, mean + amplitude) * 2)
    centered_plus = plus - mean * s.ones(6, 1)
    centered_minus = minus - mean * s.ones(6, 1)
    assert centered_plus == -centered_minus
    assert centered_plus.dot(source) == centered_minus.dot(source) == 0
    assert generator * s.ones(6, 1) == s.zeros(6, 1)
    basis = s.Matrix.hstack(
        s.Matrix((1, -1, 0)) / s.sqrt(2),
        s.Matrix((1, 1, -2)) / s.sqrt(6),
    )
    regional_generator = (shift - s.eye(3)) / 4
    assert (
        s.simplify(basis.T * regional_generator * basis)
        == s.Matrix(((-3, s.sqrt(3)), (-s.sqrt(3), -3))) / 8
    )
    regional_source = s.simplify(basis.T * source[:3, 0])
    regional_plus = s.simplify(basis.T * centered_plus[:3, 0])
    assert s.simplify(regional_source.dot(regional_plus)) == 0
    assert s.simplify(s.det(s.Matrix.hstack(regional_source, regional_plus))) == (
        2 * s.sqrt(3) * amplitude * q
    )
    rates, accelerations = [], []
    for centered in (centered_plus, centered_minus):
        velocity = generator * centered + source
        acceleration = generator * velocity
        rates.append(s.expand(2 * centered.dot(velocity)))
        accelerations.append(
            s.expand(2 * (velocity.dot(velocity) + centered.dot(acceleration)))
        )
    assert (
        centered_plus.dot(centered_plus)
        == centered_minus.dot(centered_minus)
        == 4 * amplitude**2
    )
    assert rates == [-3 * amplitude**2] * 2
    assert s.expand(accelerations[0] - accelerations[1]) == 6 * amplitude * q
    # Both triangles retain equal means and contrasts. A different total
    # contrast acceleration also distinguishes every entry of their real Gram.


def test_prepared_native_source_and_observations_bind_the_reflected_control(
    evaluated_prediction,
):
    report = evaluated_prediction
    source = report["source"]
    assert source[0] > 0
    assert source[:3] == source[3:]
    assert source[0] == -2 * source[1]
    assert source[1] == source[2]
    assert sum(source) == 0
    plus = report["trajectories"]["plus"]["initial_observation"]
    minus = report["trajectories"]["minus"]["initial_observation"]
    assert plus["epi"] != minus["epi"]
    assert (
        tuple(a + b for a, b in zip(plus["epi"], minus["epi"], strict=True)) == (1,) * 6
    )
    assert plus["means"] == minus["means"] == (Q(1, 2),) * 2
    assert plus["gram_real"] == minus["gram_real"] == ((Q(1, 32),) * 2,) * 2
    assert (
        plus["gram_imag_numerator"] == minus["gram_imag_numerator"] == ((0,) * 2,) * 2
    )
    assert plus["relative_real"] == minus["relative_real"] == ((0,) * 2,) * 2
    assert plus["relative_imag_numerator"] == ((3 * source[0] / 4,) * 2,) * 2
    assert minus["relative_imag_numerator"] == ((-3 * source[0] / 4,) * 2,) * 2
    for observation in (plus, minus):
        assert observation["nodal_rate_rounding_defect"] == (0,) * 6
        assert abs(observation["intensity_rate"] + Q(3, 128)) <= Q(1, 10**12)
        assert all(
            abs(value + Q(3, 128)) <= Q(1, 10**12)
            for row in observation["gram_rate_real"]
            for value in row
        )
    # Initial equality belongs to the exact affine preparation. Fresh pressure
    # assembly is a materialized calculation with separately admitted defects.
    assert abs(plus["intensity_rate"] - minus["intensity_rate"]) <= Q(2, 10**12)


def test_reserved_future_response_exceeds_the_prospective_numerical_budget(
    evaluated_prediction,
):
    report = evaluated_prediction
    prediction = report["prediction"]
    for name, trajectory in report["trajectories"].items():
        reserved = prediction["trajectories"][name]
        endpoint = trajectory["final_observation"]
        error = max(
            abs(actual - expected)
            for actual, expected in zip(
                endpoint["epi"], reserved["euler_final"], strict=True
            )
        )
        assert error == trajectory["final_error_vs_exact_Euler"]
        assert error <= trajectory["runtime_bound"] + trajectory["source_model_bound"]
        assert trajectory["runtime_bound"] <= prediction["runtime_allowance"]
        assert trajectory["source_model_bound"] <= prediction["source_allowance"]
        assert trajectory["frozen_state_preserved"]
        assert trajectory["clipping_inactive"]
        # Independently read the represented fine-state contrast, rather than
        # accepting an intensity supplied by an unrelated report field.
        regional_means = tuple(
            sum(endpoint["epi"][offset : offset + 3]) / 3 for offset in (0, 3)
        )
        assert endpoint["means"] == regional_means
        intensity = sum(
            (value - regional_means[0]) ** 2 for value in endpoint["epi"][:3]
        )
        assert endpoint["intensity"] == intensity
        assert endpoint["gram_real"] == ((intensity,) * 2,) * 2
        assert endpoint["gram_imag_numerator"] == ((0,) * 2,) * 2
        arithmetic_error = (
            trajectory["runtime_bound"] + trajectory["source_model_bound"]
        )
        radius = prediction["form_radius_bound"]
        intensity_error = 6 * radius * arithmetic_error + 3 * arithmetic_error**2
        assert abs(intensity - reserved["euler_intensity"]) <= intensity_error
        # The complex exponential is a diagnostic estimate, not the rigorous
        # certificate. Only this comparison carries floating evaluation slack.
        assert abs(float(intensity) - reserved["continuous_intensity_estimate"]) <= (
            float(reserved["intensity_error_budget"]) + 1e-15
        )
    total_budget = sum(
        row["intensity_error_budget"] for row in prediction["trajectories"].values()
    )
    assert report["observed_gap"] >= prediction["continuous_gap_lower"] - total_budget
    assert report["observed_gap"] > prediction["observed_gap_threshold"]
    assert report["passed"]


def test_uniform_phase_ablation_removes_separation_without_changing_epichannel(
    evaluated_prediction,
):
    report = evaluated_prediction
    # The producer holds the same coefficients. Uniform phases set the source
    # to zero without renormalizing the EPI channel, unlike removing its weight.
    assert report["prediction"]["pressure_weights"] == {
        "phase": 0.5,
        "epi": 0.5,
        "vf": 0.0,
        "topo": 0.0,
    }
    radius = report["prediction"]["form_radius_bound"]
    pair_error = Q(0)
    for row in report["ablation"].values():
        assert row["source_model_bound"] == 0
        assert row["frozen_state_preserved"] and row["clipping_inactive"]
        epsilon = row["runtime_bound"]
        assert row["final_error_vs_exact_Euler"] <= epsilon
        pair_error += 6 * radius * epsilon + 3 * epsilon**2
        for endpoint in ("initial_observation", "final_observation"):
            for field in (
                "relative_real",
                "relative_imag_numerator",
                "relative_rate_real",
                "relative_rate_imag_numerator",
            ):
                assert row[endpoint][field] == ((0,) * 2,) * 2
    assert abs(report["ablation_gap"]) <= pair_error <= Q(2, 10**12)
    assert report["observed_gap"] > report["prediction"]["observed_gap_threshold"]


def test_changed_frozen_prediction_is_rejected_before_execution(monkeypatch):
    prediction = campaign.prepare_prediction()
    changed = dict(prediction, horizon=prediction["horizon"] * 2)

    def forbidden_execution(*args, **kwargs):
        raise AssertionError("changed protocol reached the engine")

    monkeypatch.setattr(campaign, "_execute", forbidden_execution)
    monkeypatch.setattr(campaign, "_graph", forbidden_execution)
    with pytest.raises(ValueError, match="frozen protocol/source"):
        campaign.evaluate_prediction(changed)


def _optimized_process(script, *arguments):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(campaign.ROOT / "src") + os.pathsep + str(campaign.ROOT)
    )
    return subprocess.run(
        [sys.executable, "-O", "-c", textwrap.dedent(script), *map(str, arguments)],
        cwd=campaign.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_optimized_execution_retains_source_arithmetic_and_held_state_admission():
    process = _optimized_process(
        """
        from fractions import Fraction as Q
        from benchmarks import source_relative_form_response as campaign

        def require_rejection(action, message):
            try:
                action()
            except ValueError as error:
                if message not in str(error):
                    raise RuntimeError(f"wrong admission: {error}") from error
            else:
                raise RuntimeError(f"optimized execution lost {message}")

        require_rejection(
            lambda: campaign._execute("plus", (Q(99),) * 6),
            "prospective source discrepancy",
        )
        update = campaign.update_epi_via_nodal_equation
        def broken_step(graph, **kwargs):
            update(graph, **kwargs)
            graph.nodes[0]["EPI"] += 0.001
        campaign.update_epi_via_nodal_equation = broken_step
        require_rejection(
            lambda: campaign._execute("plus", campaign.IDEAL_SOURCE),
            "arithmetic defects exceed",
        )
        def changed_phase(graph, **kwargs):
            update(graph, **kwargs)
            graph.nodes[0]["theta"] = 1.0
        campaign.update_epi_via_nodal_equation = changed_phase
        require_rejection(
            lambda: campaign._execute("plus", campaign.IDEAL_SOURCE),
            "held phase, capacity or support changed",
        )
        def boundary_step(graph, **kwargs):
            update(graph, **kwargs)
            graph.nodes[0]["EPI"] = 0.0
        campaign.update_epi_via_nodal_equation = boundary_step
        campaign.RUNTIME_ALLOWANCE = Q(1)
        require_rejection(
            lambda: campaign._execute("plus", campaign.IDEAL_SOURCE),
            "boundary projection may have acted",
        )
        """
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize("failure", ("decision", "exception"))
def test_cli_failure_is_recorded_and_exits_nonzero_under_optimization(
    tmp_path, failure
):
    output = tmp_path / "regression.json"
    output.with_suffix(".prediction.json").write_text("{}\n", encoding="utf-8")
    process = _optimized_process(
        """
        import sys
        from fractions import Fraction as Q
        from benchmarks import source_relative_form_response as campaign
        output, failure = sys.argv[1:]
        campaign.prepare_prediction = lambda: {}
        def evaluate(prediction):
            if failure == "exception":
                raise ValueError("injected execution boundary")
            return {"passed": False, "observed_gap": Q(0), "ablation_gap": Q(0)}
        campaign.evaluate_prediction = evaluate
        sys.argv = ["source_relative_form_response.py", "--output", output]
        raise SystemExit(campaign.main())
        """,
        output,
        failure,
    )
    assert process.returncode != 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["passed"] is False
    if failure == "exception":
        assert "injected execution boundary" in record["error"]


def test_cli_refuses_to_replace_a_retained_response_before_execution(
    tmp_path, monkeypatch
):
    output = tmp_path / "retained.json"
    original = b'{"passed": true, "original": true}\n'
    output.write_bytes(original)
    output.with_suffix(".prediction.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "prepare_prediction", lambda: {})
    monkeypatch.setattr(sys, "argv", ["campaign", "--output", str(output)])

    def forbidden_execution(*args):
        raise AssertionError("existing response reached evaluation")

    monkeypatch.setattr(campaign, "evaluate_prediction", forbidden_execution)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        campaign.main()
    assert output.read_bytes() == original


def test_first_evaluated_records_and_original_producer_remain_identical():
    directory = campaign.ROOT / "docs/assets/source_relative_form_response"
    expected_hashes = {
        "result.prediction.json": (
            "1a08e804f2ce17b0457eac26f4196e5cb2fd26e1b4374b96ee730938ec1fc022"
        ),
        "result.json": "d1a37558f4e5386470b291b648eff0cfa4b2b9a9559a267ef46de76d6be7b94a",
    }
    for name, expected in expected_hashes.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    prediction = json.loads((directory / "result.prediction.json").read_text())
    archived_source = directory / "producer.v1.py.txt"
    assert (
        hashlib.sha256(archived_source.read_bytes()).hexdigest()
        == prediction["source_sha256"]["benchmarks/source_relative_form_response.py"]
    )
