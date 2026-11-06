"""Finite prepared-state response of two conditional phase/form closures.

These controls differentiate the full cotangent connection before projecting.
They do not assume a nonlinear invariant Fourier mode or mistake prepared
tangent-line endpoints for endpoints of an executed ODE trajectory.
"""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

from benchmarks import phase_form_exchange_comparison as campaign


@pytest.fixture(scope="module")
def evaluated_prediction():
    # Replays are implementation regression checks after the first retained
    # freeze/evaluation. One producer serves all pressure/provenance checks.
    return campaign.evaluate_prediction(campaign.prepare_prediction())


@pytest.fixture(scope="module")
def cycle_reference():
    s = pytest.importorskip("sympy")
    laplacian = s.eye(6)
    metric_hessians = []
    for node in range(6):
        laplacian[node, (node - 1) % 6] = -s.Rational(1, 2)
        laplacian[node, (node + 1) % 6] = -s.Rational(1, 2)
    for node in range(6):
        displacement = -laplacian[node, :].T
        half_separation = s.zeros(6, 1)
        half_separation[(node + 1) % 6] = s.Rational(1, 2)
        half_separation[(node - 1) % 6] = -s.Rational(1, 2)
        # Exact second jet of 2*pi*cos(b)*sinc(a) at consensus. Terms of
        # degree four and higher contribute neither first nor second jets.
        metric_hessians.append(
            -2
            * s.pi
            * (half_separation * half_separation.T + displacement * displacement.T / 3)
        )
    modes = (
        s.Matrix(
            (
                1,
                s.Rational(1, 2),
                -s.Rational(1, 2),
                -1,
                -s.Rational(1, 2),
                s.Rational(1, 2),
            )
        ),
        s.Matrix(
            (
                1,
                -s.Rational(1, 2),
                -s.Rational(1, 2),
                1,
                -s.Rational(1, 2),
                -s.Rational(1, 2),
            )
        ),
    )
    values = (s.Rational(1, 2), s.Rational(3, 2))
    for mode, value in zip(modes, values, strict=True):
        assert laplacian * mode == value * mode
        assert sum(mode) == 0
    return s, laplacian, metric_hessians, modes, values


def _connection_rate_at_consensus(reference, form, exchange_scale):
    s, _, metric_hessians, _, _ = reference
    phase_rate = form / (2 * s.pi * exchange_scale)
    differential_rates = [hessian * phase_rate for hessian in metric_hessians]
    # K_ij=(x_j partial_i H_j-x_i partial_j H_i)/(H_i H_j).
    # DH=0 at consensus, so form-rate and denominator-rate terms vanish.
    return s.Matrix(
        6,
        6,
        lambda i, j: (
            form[j] * differential_rates[j][i] - form[i] * differential_rates[i][j]
        )
        / (4 * s.pi**2),
    )


def test_complete_finite_amplitude_acceleration_has_exact_modal_discriminator(
    cycle_reference,
):
    s, laplacian, _, modes, values = cycle_reference
    amplitude = s.Symbol("amplitude", real=True, nonzero=True)
    e, w, beta, eta = s.symbols("e w beta eta", positive=True)
    restoring = {"relational": [], "cotangent": []}
    phase_responses = {"relational": [], "cotangent": []}
    for mode, value in zip(modes, values, strict=True):
        form = amplitude * mode
        form_rate = -e * laplacian * form
        relational_phase = (w / (beta * s.pi)) * laplacian * form
        cotangent_phase = form / (eta * 2 * s.pi)
        connection_rate = _connection_rate_at_consensus(cycle_reference, form, eta)
        assert connection_rate + connection_rate.T == s.zeros(6)
        accelerations = {
            "relational": -e * laplacian * form_rate
            - (w / s.pi) * laplacian * relational_phase,
            "cotangent": -e * laplacian * form_rate
            - (w / s.pi) * laplacian * cotangent_phase
            + connection_rate * form / eta,
        }
        for name, phase_rate in (
            ("relational", relational_phase),
            ("cotangent", cotangent_phase),
        ):
            projection = mode.dot(accelerations[name]) / (amplitude * mode.dot(mode))
            restoring[name].append(s.simplify(e**2 * value**2 - projection))
            phase_responses[name].append(
                s.simplify(mode.dot(phase_rate) / (amplitude * mode.dot(mode)))
            )
        assert (
            s.simplify(restoring["relational"][-1] - w**2 * value**2 / (beta * s.pi**2))
            == 0
        )
        assert (
            s.simplify(restoring["cotangent"][-1] - w * value / (eta * 2 * s.pi**2))
            == 0
        )
    assert s.simplify(restoring["relational"][1] / restoring["relational"][0]) == 9
    assert s.simplify(restoring["cotangent"][1] / restoring["cotangent"][0]) == 3
    assert (
        s.simplify(phase_responses["relational"][1] / phase_responses["relational"][0])
        == 3
    )
    assert (
        s.simplify(phase_responses["cotangent"][1] / phase_responses["cotangent"][0])
        == 1
    )


def test_projected_identity_does_not_make_the_cotangent_mode_invariant(
    cycle_reference,
):
    s, _, _, modes, _ = cycle_reference
    amplitude = s.Symbol("amplitude", nonzero=True, real=True)
    form = amplitude * modes[0]
    connection_rate = _connection_rate_at_consensus(cycle_reference, form, 1)
    leakage = (connection_rate * form).applyfunc(s.simplify)
    expected = amplitude**3 * s.Matrix((-1, 1, -1, 1, -1, 1)) / (16 * s.pi**2)
    assert leakage == expected
    assert leakage != s.zeros(6, 1)
    assert modes[0].dot(leakage) == 0
    assert sum(leakage) == 0


def test_common_form_offset_and_mixed_modes_are_not_admitted_substitutions(
    cycle_reference,
):
    s, _, _, modes, _ = cycle_reference
    amplitude = s.Symbol("amplitude", nonzero=True, real=True)
    offset = s.Symbol("offset", real=True)
    offset_defects = []
    for mode in modes:
        form = amplitude * mode + offset * s.ones(6, 1)
        connection_rate = _connection_rate_at_consensus(cycle_reference, form, 1)
        offset_defects.append(
            s.simplify(mode.dot(connection_rate * form) / (amplitude * mode.dot(mode)))
        )
    assert offset_defects == [
        -5 * offset**2 / (24 * s.pi**2),
        -3 * offset**2 / (8 * s.pi**2),
    ]
    mixed = modes[0] + modes[1]
    connection_rate = _connection_rate_at_consensus(cycle_reference, mixed, 1)
    leakage = connection_rate * mixed
    assert s.simplify(mixed.dot(leakage)) == 0
    assert s.simplify(modes[0].dot(leakage) / modes[0].dot(modes[0])) == 1 / (
        8 * s.pi**2
    )
    assert s.simplify(modes[1].dot(leakage) / modes[1].dot(modes[1])) == -1 / (
        8 * s.pi**2
    )


def test_finite_tangent_line_projection_is_exact_without_a_time_step_claim(
    cycle_reference,
):
    s, laplacian, _, modes, values = cycle_reference
    amplitude, step = s.symbols("amplitude step", nonzero=True, real=True)
    e, w, beta, eta = s.symbols("e w beta eta", positive=True)
    for mode, value in zip(modes, values, strict=True):
        form = amplitude * mode
        form_rate = -e * laplacian * form
        for phase_rate, expected in (
            (
                (w / (beta * s.pi)) * laplacian * form,
                e**2 * value**2 - w**2 * value**2 / (beta * s.pi**2),
            ),
            (
                form / (eta * 2 * s.pi),
                e**2 * value**2 - w * value / (eta * 2 * s.pi**2),
            ),
        ):
            projected = []
            for sign in (-1, 1):
                prepared_form = form + sign * step * form_rate
                prepared_phase = sign * step * phase_rate
                # On the admitted acute degree-two chart, phasor Arg is
                # exactly the neighbor midpoint, so g=-L*theta/pi.
                native_rate = -e * laplacian * prepared_form
                native_rate -= (w / s.pi) * laplacian * prepared_phase
                # For the cotangent law, its additional projected Kx is
                # exactly zero because x stays collinear with this mode
                # and K is skew. A generic prepared state lacks this fact.
                projected.append(mode.dot(native_rate) / mode.dot(mode))
            finite_response = s.simplify((projected[1] - projected[0]) / (2 * step))
            assert s.simplify(finite_response / amplitude - expected) == 0
    # Admissibility bounds h by actual edge gaps; this identity says nothing
    # about the finite temporal truncation error of an evolved trajectory.


def test_actual_pressure_probes_separate_laws_inside_prospective_error_budgets(
    evaluated_prediction,
):
    report = evaluated_prediction
    prediction = report["prediction"]
    assert report["passed"]
    for model, expected in (("relational", 9), ("cotangent", 3)):
        observed = report["observed"][model]
        assert float(observed["ratio"]) == pytest.approx(expected, abs=1e-10)
        for label, values in observed["modes"].items():
            case = prediction["cases"][model][label]
            for sign, rates in values["native_rate"].items():
                probe = case["probes"][sign]
                # Independently evaluate the admitted midpoint pressure in
                # exact arithmetic on materialized inputs and pi. The full
                # cotangent vector is not claimed to equal this pressure.
                expected_rates = []
                for node in range(6):
                    previous, following = (node - 1) % 6, (node + 1) % 6
                    form_gap = (
                        probe["form"][previous] + probe["form"][following]
                    ) / 2 - probe["form"][node]
                    phase_gap = (
                        probe["phase"][previous] + probe["phase"][following]
                    ) / 2 - probe["phase"][node]
                    expected_rates.append(
                        form_gap / 2 + phase_gap / (2 * prediction["pi_represented"])
                    )
                assert tuple(map(float, rates)) == pytest.approx(
                    tuple(map(float, expected_rates)), abs=2e-15
                )
            assert values["restoring_error_bound"] <= prediction["restoring_allowance"]
            assert (
                abs(values["restoring"] - case["reference_restoring"])
                <= values["restoring_error_bound"]
            )
    assert report["observed"]["cotangent"]["ratio"] < 6
    assert report["observed"]["relational"]["ratio"] > 6
    # The scale match was declared before evaluating these two equal low-mode
    # preparations; the high-mode coefficients are never recalibrated.
    low_relational = prediction["cases"]["relational"]["low"]
    low_cotangent = prediction["cases"]["cotangent"]["low"]
    assert low_relational["probes"] == low_cotangent["probes"]
    assert (
        report["observed"]["relational"]["modes"]["low"]["native_rate"]
        == report["observed"]["cotangent"]["modes"]["low"]["native_rate"]
    )


def test_prediction_and_changed_protocol_rejection_precede_pressure(monkeypatch):
    def forbidden_probe(*args, **kwargs):
        raise AssertionError("no pressure may execute before protocol admission")

    monkeypatch.setattr(campaign, "_native_rate", forbidden_probe)
    prediction = campaign.prepare_prediction()
    changed = deepcopy(prediction)
    changed["beta"] += Q(1, 10)
    with pytest.raises(ValueError, match="frozen protocol/source"):
        campaign.evaluate_prediction(changed)
    changed = deepcopy(prediction)
    changed["cases"]["relational"]["high"]["probes"]["plus"]["form"] = (Q(0),) * 6
    with pytest.raises(ValueError, match="frozen protocol/source"):
        campaign.evaluate_prediction(changed)


def test_global_balance_near_consensus_forces_relative_phase_velocity():
    s = pytest.importorskip("sympy")
    theta = s.Matrix(s.symbols("theta0:3", real=True))
    direction = s.Matrix(s.symbols("v0:3", real=True))
    form = s.Matrix(s.symbols("x0:3", real=True))
    velocity = s.Matrix(s.symbols("omega0:3", real=True))
    epsilon = s.Symbol("epsilon", real=True)
    w, beta = s.symbols("w beta", positive=True)
    neighbors = ((1,), (0, 2), (1,))
    degree = s.diag(1, 2, 1)
    laplacian = s.Matrix(((1, -1, 0), (-1, 2, -1), (0, -1, 1)))
    capacity = s.diag(1, 0, 2)
    # Derive the phase-pressure jet independently from the neighbor phasor
    # derivative: Arg'(S)=Im(conjugate(S)*S')/|S|^2. No selected phase law
    # or nodewise work cancellation supplies this source derivative.
    source_jet = []
    for node, adjacent in enumerate(neighbors):
        resultant = sum(s.exp(s.I * epsilon * direction[j]) for j in adjacent)
        initial = resultant.subs(epsilon, 0)
        derivative = s.diff(resultant, epsilon).subs(epsilon, 0)
        argument_jet = s.im(s.conjugate(initial) * derivative) / abs(initial) ** 2
        source_jet.append((argument_jet - direction[node]) / s.pi)
    source_jet = s.Matrix(source_jet)
    phase_storage = sum(1 - s.cos(theta[j] - theta[i]) for i, j in ((0, 1), (1, 2)))
    phase_gradient = s.Matrix([s.diff(phase_storage, item) for item in theta])
    phase_gradient_jet = (
        phase_gradient.subs(dict(zip(theta, epsilon * direction, strict=True)))
        .diff(epsilon)
        .subs(epsilon, 0)
    )
    assert source_jet == -degree.inv() * laplacian * direction / s.pi
    assert phase_gradient_jet == laplacian * direction
    first_work = w * (laplacian * form).dot(capacity * source_jet)
    first_work += beta * phase_gradient_jet.dot(velocity)
    # Continuity of Omega is sufficient: the vanishing phase gradient
    # multiplies Omega(x,epsilon*v)=Omega(x,0)+o(1). No derivative of Omega
    # is required. Every independent perturbation direction must cancel.
    coefficients = s.Matrix([s.diff(first_work, item) for item in direction])
    required = (w / (beta * s.pi)) * degree.inv() * capacity * laplacian * form
    assert (coefficients - beta * laplacian * (velocity - required)).applyfunc(
        s.simplify
    ) == s.zeros(3, 1)
    assert laplacian.nullspace() == [s.ones(3, 1)]
    solution = s.solve(tuple(coefficients), tuple(velocity[:2]), dict=True)
    assert len(solution) == 1
    admitted = velocity.subs(solution[0])
    residual = (admitted - required).applyfunc(s.simplify)
    assert residual[0] == residual[1] == residual[2]
    assert required[1] == 0
    assert s.simplify(required[0] - w * (form[0] - form[1]) / (beta * s.pi)) == 0
    assert s.simplify(required[2] - 2 * w * (form[2] - form[1]) / (beta * s.pi)) == 0
    # The zero-capacity node still admits a common phase rotation: global
    # work alone determines relative phase rates, not an absolute clock.
