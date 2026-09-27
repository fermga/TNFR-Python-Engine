"""Small execution controls for the conditional cotangent C8 instrument.

These tests do not regenerate the reserved 70,924-step research response.
Independent derivatives and four Euler steps check implementation and report
wiring, not a validated binary64 enclosure of the continuous trajectory.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.dnfr import default_compute_delta_nfr

_PATH = Path(__file__).resolve().parents[2] / "benchmarks/cotangent_phase_exchange.py"
_SPEC = importlib.util.spec_from_file_location("cotangent_cycle_execution_test", _PATH)
BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BENCH
_SPEC.loader.exec_module(BENCH)


def _independent_geometry(mp, phases):
    """Differentiate the defining metric with arbitrary precision, not its series."""

    def metric_at(i, coordinates):
        left, center, right = (
            coordinates[(i - 1) % 8],
            coordinates[i],
            coordinates[(i + 1) % 8],
        )
        a, b = (left + right) / 2 - center, (right - left) / 2
        return 2 * mp.pi * mp.cos(b) * mp.sinc(a)

    metric = mp.matrix([metric_at(i, phases) for i in range(8)])
    differential = mp.matrix(8, 8)
    for i in range(8):
        for j in range(8):

            def changed(value):
                coordinates = list(phases)
                coordinates[j] = value
                return metric_at(i, coordinates)

            differential[i, j] = mp.diff(changed, phases[j])
    return metric, differential


def test_consensus_geometry_has_constant_metric_and_zero_first_derivative():
    metric, differential, displacement, edges = BENCH.cycle_geometry(np.zeros(8))
    np.testing.assert_array_equal(metric, np.full(8, 2 * math.pi))
    np.testing.assert_array_equal(differential, np.zeros((8, 8)))
    np.testing.assert_array_equal(displacement, np.zeros(8))
    np.testing.assert_array_equal(edges, np.zeros(8))


@pytest.mark.parametrize("scale", [1.0, 1e-10], ids=["regular", "near-consensus"])
def test_metric_differential_matches_independent_high_precision_derivatives(scale):
    mp = pytest.importorskip("mpmath")
    phases = scale * np.array([0.11, -0.04, 0.21, 0.03, -0.17, 0.14, 0.06, -0.09])
    actual_metric, actual_differential, _, _ = BENCH.cycle_geometry(phases)
    with mp.workdps(60):
        metric, differential = _independent_geometry(
            mp, [mp.mpf(float(value)) for value in phases]
        )
        expected_metric = np.array([float(value) for value in metric])
        expected_differential = np.array(differential.tolist(), dtype=float)
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=3e-16, atol=2e-15)
    np.testing.assert_allclose(
        actual_differential, expected_differential, rtol=3e-12, atol=2e-15 * scale
    )
    # Rotation invariance and reflection are useful checks on all three local
    # derivative entries, including signs that a diagonal-only test would miss.
    np.testing.assert_allclose(actual_differential.sum(axis=1), 0, atol=3e-16 * scale)
    reflected_metric, reflected_differential, _, _ = BENCH.cycle_geometry(-phases)
    np.testing.assert_array_equal(reflected_metric, actual_metric)
    np.testing.assert_array_equal(reflected_differential, -actual_differential)


def test_actual_pressure_owner_and_complete_storage_momentum_balances():
    form = np.array([0.31, -0.12, 0.28, -0.17, 0.09, 0.22, -0.26, 0.14])
    phases = np.array([0.11, -0.04, 0.21, 0.03, -0.17, 0.14, 0.06, -0.09])
    field = BENCH.cotangent_cycle_field(form, phases)
    graph = nx.cycle_graph(8)
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.5, "phase": 0.5, "vf": 0, "topo": 0}
    for i in graph:
        set_attr(graph.nodes[i], ALIAS_EPI, float(form[i]))
        set_attr(graph.nodes[i], ALIAS_THETA, float(phases[i]))
        set_attr(graph.nodes[i], ALIAS_VF, 1.0)
    default_compute_delta_nfr(graph)
    np.testing.assert_allclose(
        field["old_pressure"],
        [get_attr(graph.nodes[i], ALIAS_DNFR) for i in graph],
        rtol=0,
        atol=4e-16,
    )
    laplacian_form = form - (np.roll(form, 1) + np.roll(form, -1)) / 2
    potential_gradient = np.zeros(8)
    for left, right in graph.edges:
        slope = math.sin(phases[left] - phases[right])
        potential_gradient[left] += slope
        potential_gradient[right] -= slope
    np.testing.assert_allclose(
        field["form_rate"] - field["old_pressure"],
        field["geometric_pressure"],
        rtol=0,
        atol=2e-17,
    )
    assert np.max(np.abs(field["geometric_pressure"])) > 1e-4
    assert float(form @ field["geometric_pressure"]) == pytest.approx(0, abs=2e-18)
    storage_rate = form @ field["form_rate"] + 0.5 * (
        potential_gradient @ field["phase_rate"]
    )
    assert storage_rate == pytest.approx(-0.5 * form @ laplacian_form, abs=8e-17)
    momentum_rate = field["metric"] @ field["form_rate"] + form @ (
        field["metric_differential"] @ field["phase_rate"]
    )
    assert momentum_rate == pytest.approx(
        -0.5 * field["metric"] @ laplacian_form, abs=5e-16
    )


def _independent_four_steps(amplitude, timestep):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(60):
        form = mp.matrix([mp.mpf(amplitude) * mp.cos(mp.pi * i / 4) for i in range(8)])
        phases, dt = mp.matrix(8, 1), mp.mpf(timestep)
        for _ in range(4):
            metric, differential = _independent_geometry(mp, list(phases))
            form_rate, phase_rate = mp.matrix(8, 1), mp.matrix(8, 1)
            for i in range(8):
                laplacian = form[i] - (form[(i - 1) % 8] + form[(i + 1) % 8]) / 2
                displacement = (phases[(i - 1) % 8] + phases[(i + 1) % 8]) / 2 - phases[
                    i
                ]
                correction = sum(
                    (form[j] * differential[j, i] - form[i] * differential[i, j])
                    * form[j]
                    / (metric[i] * metric[j])
                    for j in range(8)
                )
                form_rate[i] = -laplacian / 2 + displacement / (2 * mp.pi) + correction
                phase_rate[i] = form[i] / metric[i]
            form, phases = form + dt * form_rate, phases + dt * phase_rate
        return np.array([float(value) for value in form]), np.array(
            [float(value) for value in phases]
        )


def test_four_steps_use_shared_simultaneous_integrator_and_detached_diagnostics(
    monkeypatch,
):
    calls = []
    original = BENCH.euler_update

    def tracked(before, duration, rate):
        calls.append((before.copy(), duration, rate.copy()))
        return original(before, duration, rate)

    monkeypatch.setattr(BENCH, "euler_update", tracked)
    spec = BENCH.CotangentCycleSpec(2.0**-10, 2.0**-11, 4, (1,))
    report = BENCH.run_cotangent_cycle(spec)
    assert len(calls) == 8
    assert [sample["step"] for sample in report["samples"]] == [0, 1, 4]
    initial, first, final = report["samples"]
    ell = 1 - math.sqrt(2) / 2
    np.testing.assert_allclose(calls[0][2], -0.5 * ell * calls[0][0], atol=8e-20)
    np.testing.assert_array_equal(calls[1][0], np.zeros(8))
    np.testing.assert_allclose(calls[1][2], calls[0][0] / (2 * math.pi), atol=2e-20)
    np.testing.assert_allclose(
        first["form"],
        np.array(initial["form"]) * (1 - spec.timestep * ell / 2),
        atol=2e-19,
    )
    expected_form, expected_phase = _independent_four_steps(
        spec.amplitude, spec.timestep
    )
    np.testing.assert_allclose(final["form"], expected_form, rtol=0, atol=5e-19)
    np.testing.assert_allclose(final["phase_lift"], expected_phase, rtol=0, atol=2e-22)
    assert final["time"] == 4 * spec.timestep
    assert report["all_step_maxima"]["energy_norm"] >= 2 * spec.amplitude
    assert report["minimum_metric"] > 6.0
    assert report["execution"]["nodal_update_owner"].endswith(
        "_euler_kernel.euler_update"
    )
    assert report["execution"]["precision"] == "binary64"
    assert report["execution"]["working_source_digest"]
    assert "no validated" in report["execution"]["residual_scope"]
    for sample in report["samples"]:
        diagnostics = sample["diagnostics"]
        assert diagnostics["schema_version"] == "tnfr.diagnostics.v1"
        assert set(diagnostics["tetrad"]) == {"phi_s", "grad_phi", "k_phi", "xi_c"}
        assert diagnostics["tetrad"]["k_phi"]["complete"]
        assert "provenance" in diagnostics["tetrad"]["xi_c"]
        pressures = [row["delta_nfr"]["value"] for row in diagnostics["state"]["nodes"]]
        np.testing.assert_array_equal(
            pressures, np.array(sample["old_pressure"]) + sample["geometric_pressure"]
        )
        assert "geometric K*x" in sample["diagnostics_pressure_provenance"]
        assert sample["default_graph_pressure_residual_inf"] < 4e-16
    # Serialization retains the detached finite evidence without non-JSON NaNs.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "changes",
    [{"amplitude": True}, {"timestep": math.inf}, {"amplitude": 0.5}, {"steps": True}],
    ids=["boolean-amplitude", "nonfinite-clock", "energy-boundary", "boolean-count"],
)
def test_invalid_preparations_are_rejected(changes):
    supplied = {"amplitude": 2.0**-10, "timestep": 2.0**-11, "steps": 4}
    supplied.update(changes)
    with pytest.raises(ValueError):
        BENCH.CotangentCycleSpec(**supplied)


def test_phase_at_acute_boundary_is_not_silently_projected():
    phase = np.zeros(8)
    phase[0] = math.pi / 2
    with pytest.raises(ValueError, match="strict acute"):
        BENCH.cycle_geometry(phase)
