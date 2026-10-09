"""Regional orientation from the actual nonrepeated prism phase source.

These are instantaneous projections and detached pressure read-outs. A
neighboring phase preparation supplies an input, not an autonomous source
law or a certificate of sustained form. The phase-response row below remains
the existing configured formula, compared hypothetically with fresh pressure.
"""

import math
from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import NODES, _graph, _phase_support
from tnfr.dynamics.canonical import compute_extended_nodal_system
from tnfr.physics.extended import compute_phase_current
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.support_transport import observe_regional_support_balance


def _lift(s):
    return s.Matrix(
        [
            [1 / s.sqrt(2), 1 / s.sqrt(6)],
            [-1 / s.sqrt(2), 1 / s.sqrt(6)],
            [0, -2 / s.sqrt(6)],
        ]
    )


def _area(s, left, right):
    return s.det(s.Matrix.hstack(left, right))


def test_nonrepeated_projection_retains_boundary_area_and_both_mean_rates():
    s = pytest.importorskip("sympy")
    lift = _lift(s)
    e, k, alpha, linear, gain = s.symbols("e k alpha B gain", positive=True)
    mean_a, mean_b, phase_mean = s.symbols("mu_a mu_b beta_a", real=True)
    z = s.Matrix(s.symbols("z0 z1", real=True))
    other = s.Matrix(s.symbols("v0 v1", real=True))
    contrast = s.Matrix(s.symbols("eta0 eta1", real=True))
    directions = s.Matrix(s.symbols("gamma0 gamma1 gamma2", real=True))
    theta = phase_mean * s.ones(3, 1) + lift * contrast
    # gamma contains actual neighbor-phasor directions in the same regular
    # unwrapped chart. No independence or arbitrary source law is asserted.
    phase_source = (directions - theta) / s.pi
    x = mean_a * s.ones(3, 1) + lift * z
    x_other = mean_b * s.ones(3, 1) + lift * other
    epi_gradient = (sum(x) * s.ones(3, 1) - 4 * x + x_other) / 3
    pressure = e * epi_gradient + s.pi * k * phase_source
    direction_contrast = lift.T * directions
    rate = -4 * e * z / 3 + e * other / 3 - k * contrast + k * direction_contrast
    assert (lift.T * pressure - rate).applyfunc(s.expand) == s.zeros(2, 1)
    mean_rate = e * (mean_b - mean_a) / 3 + k * (sum(directions) / 3 - phase_mean)
    assert s.expand(sum(pressure) / 3 - mean_rate) == 0

    current = s.Matrix(s.symbols("j0 j1 j2", real=True))
    nonlinear = pressure.applyfunc(lambda value: s.sin(s.pi * value))
    phase_rate = alpha * nonlinear + linear * pressure + gain * current
    contrast_rate = lift.T * phase_rate
    actual = _area(s, rate, contrast) + _area(s, z, contrast_rate)
    expected = (
        -(4 * e / 3 + linear * k) * _area(s, z, contrast)
        + e * _area(s, other, contrast) / 3
        + k * _area(s, direction_contrast, contrast)
        + linear * e * _area(s, z, other) / 3
        + linear * k * _area(s, z, direction_contrast)
        + alpha * _area(s, z, lift.T * nonlinear)
        + gain * _area(s, z, lift.T * current)
    )
    assert s.expand(actual - expected) == 0

    # Unlike the repeated lift, a region's sine current has boundary flux.
    _, rows = _phase_support()
    phases = s.symbols("t0:6", real=True)
    currents = s.Matrix(
        [
            sum(s.sin(phases[j] - phases[i]) for j in row) / 3
            for i, row in enumerate(rows)
        ]
    )
    cut_current = sum(s.sin(phases[i + 3] - phases[i]) for i in range(3)) / 9
    assert s.expand(sum(currents[:3]) / 3 - cut_current) == 0
    assert s.expand(sum(currents)) == 0
    assert (
        s.expand(
            sum(phase_rate) / 3
            - alpha * sum(nonlinear) / 3
            - linear * mean_rate
            - gain * sum(current) / 3
        )
        == 0
    )


def test_strict_boundary_phase_witness_creates_local_area_from_uniform_form():
    s = pytest.importorskip("sympy")
    lift = _lift(s)
    a = s.pi / 6
    gamma = s.atan(1 / (3 * s.sqrt(3)))
    theta = s.Matrix([-a, a, 0, -a, a, a])
    graph, rows = _phase_support()
    assert all(
        abs(theta[NODES.index(i)] - theta[NODES.index(j)]) < s.pi / 2
        for i, j in graph.edges
    )
    real = s.Matrix([sum(s.cos(theta[j]) for j in row) for row in rows])
    imag = s.Matrix([sum(s.sin(theta[j]) for j in row) for row in rows])
    assert real[:3, :] == s.Matrix([1 + s.sqrt(3), 1 + s.sqrt(3), 3 * s.sqrt(3) / 2])
    assert imag[:3, :] == s.Matrix([0, 0, s.Rational(1, 2)])
    assert all(value > 0 for value in real)
    directions = s.Matrix([0, 0, gamma])
    source = (directions - theta[:3, :]) / s.pi
    assert source == s.Matrix([s.Rational(1, 6), -s.Rational(1, 6), gamma / s.pi])
    contrast = lift.T * theta[:3, :]
    weight = s.symbols("w", positive=True)
    form_rate = weight * lift.T * source
    area_rate = s.simplify(_area(s, form_rate, contrast))
    assert s.simplify(area_rate + weight * gamma / (3 * s.sqrt(3))) == 0
    assert area_rate < 0
    assert sum(weight * source) / 3 == weight * gamma / (3 * s.pi)
    # z=0 makes det(z,zeta_dot)=0 for every finite phase velocity. No phase
    # feedback choice is needed for this instantaneous source contribution.
    arbitrary_phase_rate = s.Matrix(s.symbols("v0 v1", real=True))
    assert _area(s, s.zeros(2, 1), arbitrary_phase_rate) == 0
    repeated_source = -theta[:3, :] / s.pi
    assert _area(s, weight * lift.T * repeated_source, contrast) == 0


def test_actual_capture_and_regional_owner_resolve_the_boundary_witness():
    a = math.pi / 6
    gamma = math.atan(1 / (3 * math.sqrt(3)))
    weight = 0.25
    results = []
    for repeated in (True, False):
        graph = _graph((Q(0),) * 4)
        graph.graph["DNFR_WEIGHTS"] = dict(epi=0.5, phase=weight, vf=0.25, topo=0)
        phases = (-a, a, 0, -a, a, 0 if repeated else a)
        for node, phase in zip(NODES, phases, strict=True):
            graph.nodes[node]["theta"] = phase
        before = tuple(dict(graph.nodes[node]) for node in NODES)
        capture = capture_non_epi_forcing(graph)
        assert capture.snapshot.epi_gradient == (0,) * 6
        assert capture.snapshot.capacity_gradient == (0,) * 6
        assert capture.snapshot.topology_gradient == (0,) * 6
        expected = (1 / 6, -1 / 6, 0 if repeated else gamma / math.pi)
        assert tuple(map(float, capture.phase_gradient[:3])) == pytest.approx(
            expected, abs=2e-16, rel=0
        )
        assert (
            max(abs(float(value)) for value in capture.kernel_pressure_defect) < 1e-16
        )
        regional = observe_regional_support_balance(
            capture.snapshot,
            NODES[:3],
            epi_weight=capture.epi_weight,
            forcing=capture.forcing,
        )
        assert regional.mass_boundary_rate == regional.internal_dissipation == 0
        assert regional.variance_boundary_rate == regional.model_variance_rate == 0
        assert regional.stored_mass_rate == 0  # Stored input was not refreshed.
        assert regional.mass_defect_rate == -regional.mass_forcing_rate
        assert float(regional.model_mass_rate / 9) == pytest.approx(
            0 if repeated else weight * gamma / (3 * math.pi), abs=2e-17, rel=0
        )
        pressure = tuple(map(float, capture.full_kernel_pressure))
        projected_rate = (
            (pressure[0] - pressure[1]) / math.sqrt(2),
            (pressure[0] + pressure[1] - 2 * pressure[2]) / math.sqrt(6),
        )
        contrast = (-math.sqrt(2) * a, 0.0)
        area_rate = projected_rate[0] * contrast[1] - projected_rate[1] * contrast[0]
        expected_area = 0 if repeated else -weight * gamma / (3 * math.sqrt(3))
        assert area_rate == pytest.approx(expected_area, abs=5e-17, rel=0)
        current = compute_phase_current(graph)
        phase_rates = tuple(
            compute_extended_nodal_system(
                1.0, p, phase, current[node], 0.0
            ).phase_derivative
            for node, phase, p in zip(NODES, phases, pressure, strict=True)
        )
        assert all(math.isfinite(value) for value in phase_rates)
        # The region's phase mean includes its cut-current contribution.
        cut_mean = math.fsum(math.sin(phases[i + 3] - phases[i]) for i in range(3)) / 9
        assert math.fsum(current[node] for node in NODES[:3]) / 3 == pytest.approx(
            cut_mean, abs=3e-17, rel=0
        )
        assert tuple(dict(graph.nodes[node]) for node in NODES) == before
        results.append(area_rate)
    assert results[0] == 0 and results[1] < 0
