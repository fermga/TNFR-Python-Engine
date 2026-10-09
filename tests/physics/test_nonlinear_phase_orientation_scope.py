"""Finite-amplitude orientation in the existing optional phase-response row.

The fresh-pressure comparison keeps the complete mean and the configured
nonlinear function. It is not the optional runtime's independent pressure
evolution, a newly selected phase law, or a persistence certificate.
"""

import math
from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import NODES, _graph
from tnfr.dynamics.canonical import compute_extended_nodal_system
from tnfr.physics.extended import compute_phase_current
from tnfr.physics.forcing_realization import capture_non_epi_forcing


def _symbolic_fixture():
    s = pytest.importorskip("sympy")
    lift = s.Matrix(
        [
            [1 / s.sqrt(2), 1 / s.sqrt(6)],
            [-1 / s.sqrt(2), 1 / s.sqrt(6)],
            [0, -2 / s.sqrt(6)],
        ]
    )
    form = s.Matrix([s.Rational(1, 6), s.Rational(1, 3), -s.Rational(1, 2)])
    pressure = -form / 2
    response = pressure.applyfunc(
        lambda value: s.sin(s.pi * value) / 2 + s.Rational(3, 20) * value
    )
    return s, lift, form, pressure, response


def test_nonlinear_phase_response_generates_area_and_common_phase_from_zero_contrast():
    s, lift, form, pressure, response = _symbolic_fixture()
    epi = s.Rational(3, 5) * s.ones(3, 1) + form
    assert all(0 < value < 1 for value in epi)
    assert sum(form) == sum(pressure) == 0
    z = lift.T * form
    zeta_dot = lift.T * response
    area_rate = s.simplify(s.det(s.Matrix.hstack(z, zeta_dot)))
    expected_area = (5 * s.sqrt(6) - 3 * s.sqrt(2) - 8) / (48 * s.sqrt(3))
    assert s.simplify(area_rate - expected_area) == 0
    # All compared sides are positive: 5sqrt(6)>8+3sqrt(2) reduces, after
    # two squarings, to 17>12sqrt(2), or 289>288. No numeric sign test is used.
    assert 17**2 > 12**2 * 2
    assert s.simplify(z.dot(z)) == s.Rational(7, 18)
    mean_rate = s.simplify(sum(response) / 3)
    assert s.simplify(mean_rate - (-s.sqrt(6) + 3 * s.sqrt(2) - 2) / 24) == 0
    # sqrt(6)+2>3sqrt(2) follows from sqrt(6)>2, so this mean rate is negative.
    assert 6 > 2**2
    linear_projection = lift.T * (s.Rational(3, 20) * pressure)
    assert s.simplify(s.det(s.Matrix.hstack(z, linear_projection))) == 0
    # Initially zeta=0, J_phi=0 and mu_dot=0; beta_dot nevertheless is nonzero.
    # A fresh-pressure future would inherit an angular acceleration, not an
    # initially nonzero form angular velocity or a sustained rotating orbit.


def test_actual_phase_row_matches_the_detached_fresh_pressure_orientation_witness():
    s, lift, form, pressure, response = _symbolic_fixture()
    graph = _graph(
        (Q(-1, 12), Q(1, 4), Q(-1, 12), Q(1, 4)),
        means=(Q(3, 5), Q(3, 5)),
    )
    graph.graph["DNFR_WEIGHTS"] = dict(epi=0.5, phase=0.25, vf=0.25, topo=0)
    for node in NODES:
        graph.nodes[node]["theta"] = math.pi
    node_state = tuple(dict(graph.nodes[node]) for node in NODES)
    captured = capture_non_epi_forcing(graph)
    current = compute_phase_current(graph)
    assert tuple(current[node] for node in NODES) == (0.0,) * 6
    assert captured.forcing == (0,) * 6
    assert tuple(map(float, captured.full_kernel_pressure)) == pytest.approx(
        tuple(map(float, pressure)) * 2, abs=3e-17, rel=0
    )
    # Retain the represented kernel's defect instead of declaring it exact.
    assert max(abs(float(value)) for value in captured.kernel_pressure_defect) < 1e-16
    phase_rates = tuple(
        compute_extended_nodal_system(
            1.0, float(value), math.pi, current[node], 0.0
        ).phase_derivative
        for node, value in zip(NODES, captured.full_kernel_pressure, strict=True)
    )
    assert phase_rates == pytest.approx(
        tuple(map(float, response)) * 2, abs=2e-16, rel=0
    )
    projected = (
        (phase_rates[0] - phase_rates[1]) / math.sqrt(2),
        (phase_rates[0] + phase_rates[1] - 2 * phase_rates[2]) / math.sqrt(6),
    )
    z = tuple(map(float, lift.T * form))
    area_rate = z[0] * projected[1] - z[1] * projected[0]
    ideal_area = s.det(s.Matrix.hstack(lift.T * form, lift.T * response))
    assert area_rate > 0
    assert area_rate == pytest.approx(float(ideal_area), abs=1e-16, rel=0)
    assert math.fsum(phase_rates[:3]) / 3 == pytest.approx(
        float(sum(response) / 3), abs=3e-17, rel=0
    )
    # Only phase_derivative is interpreted: the optional independent pressure
    # derivative is not substituted for the fresh-source chain rule.
    assert tuple(dict(graph.nodes[node]) for node in NODES) == node_state


def test_projected_area_budget_retains_the_pressure_mean_inside_the_nonlinearity():
    s, lift, _, _, _ = _symbolic_fixture()
    x, y, u, v, e, k, amplitude, linear, current_gain, pressure_mean = s.symbols(
        "x y u v e k A B g p0", real=True
    )
    z, zeta = s.Matrix([x, y]), s.Matrix([u, v])
    q = -e * z - k * zeta
    centered_pressure = lift * q
    pressure = pressure_mean * s.ones(3, 1) + centered_pressure
    nonlinear = pressure.applyfunc(lambda value: s.sin(s.pi * value))
    current = s.Matrix(s.symbols("j0 j1 j2", real=True))
    phase_rate = amplitude * nonlinear + linear * pressure + current_gain * current
    zeta_rate = lift.T * phase_rate
    area = s.det(s.Matrix.hstack(z, zeta))
    actual_budget = s.det(s.Matrix.hstack(q, zeta)) + s.det(
        s.Matrix.hstack(z, zeta_rate)
    )
    expected_budget = (
        -(e + linear * k) * area
        + amplitude * s.det(s.Matrix.hstack(z, lift.T * nonlinear))
        + current_gain * s.det(s.Matrix.hstack(z, lift.T * current))
    )
    assert s.expand(actual_budget - expected_budget) == 0
    # Here p0=w*c(eta) in the canonical repeated-prism source. The following
    # identity shows why centering first does not remove its nonlinear effect.
    resolved = s.sin(s.pi * pressure_mean) * lift.T * centered_pressure.applyfunc(
        lambda value: s.cos(s.pi * value)
    ) + s.cos(s.pi * pressure_mean) * lift.T * centered_pressure.applyfunc(
        lambda value: s.sin(s.pi * value)
    )
    assert (lift.T * nonlinear - resolved).applyfunc(s.expand_trig).applyfunc(
        s.simplify
    ) == s.zeros(2, 1)
    deviation = s.Matrix([-s.Rational(1, 4), 0, s.Rational(1, 4)])
    offset_sensitivity = (
        s.pi * lift.T * deviation.applyfunc(lambda value: s.cos(s.pi * value))
    )
    assert s.simplify(offset_sensitivity[0]) == s.pi * (1 - s.sqrt(2)) / 2
    assert offset_sensitivity != s.zeros(2, 1)


def test_first_small_form_orientation_term_is_sixth_order_and_absent_from_linearization():
    s, lift, _, _, _ = _symbolic_fixture()
    x, y, e, amplitude = s.symbols("x y e A", real=True)
    z = s.Matrix([x, y])
    form = lift * z
    radius_squared = x**2 + y**2
    real_sixth = s.re((x + s.I * y) ** 6).expand()
    imaginary_sixth = s.im((x + s.I * y) ** 6).expand()
    sixth_moment = sum(value**6 for value in form)
    assert s.expand(sixth_moment - (10 * radius_squared**3 - real_sixth) / 36) == 0
    for power in (1, 3):
        projection = lift.T * form.applyfunc(lambda value: value**power)
        assert s.expand(s.det(s.Matrix.hstack(z, projection))) == 0
    fifth_projection = lift.T * form.applyfunc(lambda value: value**5)
    assert (
        s.expand(s.det(s.Matrix.hstack(z, fifth_projection)) - imaginary_sixth / 36)
        == 0
    )
    pressure_sine_jet = form.applyfunc(
        lambda value: -s.pi * e * value
        + (s.pi * e * value) ** 3 / 6
        - (s.pi * e * value) ** 5 / 120
    )
    area_jet = amplitude * s.det(s.Matrix.hstack(z, lift.T * pressure_sine_jet))
    assert (
        s.expand(area_jet + amplitude * (s.pi * e) ** 5 * imaginary_sixth / 4320) == 0
    )
    # At zeta=0 the current is zero and the linear pressure term adds no area.
    # Analytic sine's next vector term is degree seven, hence its area is O(r^8).
    # Im((x+i*y)^6)=r^6*sin(6*psi): this finite-amplitude source is invisible to
    # the contrast linearization and the cubic vector response. No recurrence
    # or autonomous-model admission follows from this instantaneous term.
