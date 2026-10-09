"""Prospective finite regional window under the supplied joint nodal law.

Freeze the linear prediction and nonlinear envelopes before executing the
shared phase proposal and nodal integrator. No measured endpoint refits a
coefficient. Numerical headroom is separate from ideal comparison bounds.
"""

import math
from copy import deepcopy
from fractions import Fraction as F
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest

from tests.joint_phase_helpers import barbell, execute_joint_step, triangle
from tnfr.alias import set_theta
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.epi_memory import observe_forced_support_closure
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.joint_quotient import propose_joint_nodal_phase_step
from tnfr.physics.phase_response import predict_synchronized_joint_euler
from tnfr.physics.structural_diffusion import fiedler_partition
from tnfr.physics.support_transport import observe_regional_support_euler

H, K, STEPS = F(1, 8), F(1, 2), 48
RESERVED = (32, 40, 48)
HEADROOM = 1e-12  # finite numerical comparison, not a runtime certificate
REGIONS = ((0, 1, 2), (3, 4, 5))
FIBERS = ((0, 1), (2,), (3,), (4, 5))


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _source_checks(capture):
    """Separate nonlinearity from source materialization at higher precision."""
    with mp.workdps(70):
        phase = tuple(map(_mp, capture.phase))
        source, phase_rate, nonlinear, materialization = [], [], [], []
        for i, row in enumerate(capture.snapshot.support_neighbors):
            shifts = [phase[j] - phase[i] for j in row]
            direction = mp.arg(sum(mp.exp(1j * shift) for shift in shifts))
            source.append(direction / mp.pi)
            phase_rate.append(1 + _mp(K) * sum(map(mp.sin, shifts)) / len(row))
            linear = sum(shifts) / len(row) / mp.pi
            nonlinear.append(float(source[-1] - linear))
            materialization.append(float(_mp(capture.phase_gradient[i]) - source[-1]))
    return np.array(phase_rate, dtype=float), tuple(nonlinear), tuple(materialization)


def _project(prediction, values):
    # Right modes are D-orthonormal, not Euclidean-orthonormal.
    right = np.asarray(prediction.right_modes)
    degree = np.asarray(prediction.degree_weights, dtype=float)
    return right.T @ (degree * np.asarray(values, dtype=float))


@pytest.fixture(scope="module")
def executed():
    graph = barbell()
    prediction = predict_synchronized_joint_euler(
        graph, dt=H, coupling_strength=K, steps=STEPS
    )
    assert prediction.phase_width <= F(1, 16)
    # These acceptance thresholds and times precede every nonlinear step.
    for n in RESERVED:
        sample = prediction.samples[n]
        theta, x = np.asarray(sample.phase_modes), np.asarray(sample.epi_modes)
        et = math.sqrt(14) * float(sample.phase_error_upper)
        ex = math.sqrt(14) * float(sample.epi_error_upper)
        assert (np.linalg.norm(theta[2:]) + et) / (abs(theta[1]) - et) < 0.2
        assert (np.linalg.norm(x[2:]) + ex) / (abs(x[1]) - ex) < 0.4
        assert abs(theta[1]) - et > 0.5 * abs(prediction.initial_phase_modes[1])

    captures, regional, nonlinear, materialization, staging = [], [], [], [], []
    for n in range(STEPS + 1):
        before = capture_non_epi_forcing(graph)
        captures.append(before)
        phase_rate, source_error, source_rounding = _source_checks(before)
        nonlinear.append(source_error)
        materialization.append(source_rounding)
        if n == STEPS:
            break
        step = execute_joint_step(graph, dt=H, coupling_strength=K, time=n * H)
        assert step.before == before
        phase_after = step.phase_after
        staging.append(
            np.asarray(phase_after)
            - np.asarray(before.phase, dtype=float)
            - float(H) * phase_rate
        )
        after = step.after_epi
        regional.append(
            tuple(
                observe_regional_support_euler(
                    before.snapshot,
                    after,
                    region,
                    dt=H,
                    epi_weight=before.epi_weight,
                    forcing=before.forcing,
                )
                for region in REGIONS
            )
        )
    return SimpleNamespace(
        prediction=prediction,
        captures=tuple(captures),
        regional=tuple(regional),
        nonlinear=np.asarray(nonlinear),
        materialization=np.asarray(materialization),
        staging=np.asarray(staging),
    )


def test_actual_joint_response_stays_within_prospective_nonlinear_envelopes(executed):
    prediction = executed.prediction
    for sample, capture in zip(prediction.samples, executed.captures, strict=True):
        theta = np.asarray(capture.phase, dtype=float)
        offset = theta - float(prediction.phase_origin) - sample.step * float(H)
        x = np.asarray(capture.snapshot.epi, dtype=float)
        assert np.max(np.abs(offset - sample.phase_offset)) <= (
            float(sample.phase_error_upper) + HEADROOM
        )
        assert np.max(np.abs(x - sample.epi)) <= (
            float(sample.epi_error_upper) + HEADROOM
        )
        assert max(capture.phase) - min(capture.phase) <= prediction.phase_width
        assert min(theta) >= 0 and max(theta) < math.tau
        assert np.max(np.abs(x)) < 1  # no EPI clipping in this finite control
        assert capture.stored_pressure_residual == (0,) * 6


def test_reserved_times_retain_slow_structure_after_faster_disturbances(executed):
    prediction = executed.prediction
    for n in RESERVED:
        capture = executed.captures[n]
        offset = (
            np.asarray(capture.phase, dtype=float)
            - float(prediction.phase_origin)
            - float(n * H)
        )
        theta = _project(prediction, offset)
        x = _project(prediction, capture.snapshot.epi)
        assert np.linalg.norm(theta[2:]) < 0.2 * abs(theta[1])
        assert np.linalg.norm(x[2:]) < 0.4 * abs(x[1])
        assert abs(theta[1]) > 0.5 * abs(prediction.initial_phase_modes[1])
    # A real source nonlinearity was present, not a substituted Laplacian path.
    assert np.max(np.abs(executed.nonlinear)) > 1e-9
    assert np.max(np.abs(executed.materialization)) < 2e-15
    assert np.max(np.abs(executed.staging)) < 2e-15


def test_regional_budgets_retain_bridge_flux_and_nonlinear_mean_drift(executed):
    for pair in executed.regional:
        for budget in pair:
            assert budget.mass_identity_residual == 0
            assert budget.variance_identity_residual == 0
            assert max(map(abs, budget.state_defect)) < F(1, 10**14)
            assert budget.balance.metric_weights == (2, 2, 3, 3, 2, 2)
            assert len(budget.balance.cut_edges) == 1
    weights = np.array((2, 2, 3, 3, 2, 2))
    means = [
        np.dot(weights, np.asarray(c.snapshot.epi, float)) / 14
        for c in executed.captures
    ]
    assert abs(means[-1] - means[0]) > 1e-10
    # The predictor's constant EPI mode is retained; its observed drift is error,
    # not hidden by recentering the measured field at each step.
    initial_mean = float(np.dot(weights, executed.prediction.samples[0].epi) / 14)
    assert abs(means[-1] - initial_mean) <= (
        float(executed.prediction.samples[-1].epi_error_upper) + HEADROOM
    )
    phase_mean = [
        np.dot(weights, np.asarray(c.phase, float)) / 14 for c in executed.captures
    ]
    assert abs(phase_mean[-1] - phase_mean[0] - float(STEPS * H)) < HEADROOM


def test_geometry_selects_regions_but_slow_shape_retains_bridge_endpoints():
    graph = barbell()
    result = predict_synchronized_joint_euler(graph, dt=H, coupling_strength=K, steps=0)
    expected = (
        0,
        (11 - math.sqrt(73)) / 12,
        7 / 6,
        1.5,
        1.5,
        (11 + math.sqrt(73)) / 12,
    )
    assert result.eigenvalues == pytest.approx(expected, abs=2e-15)
    assert {frozenset(block) for block in fiedler_partition(graph)} == {
        frozenset(block) for block in REGIONS
    }
    right = np.asarray(result.right_modes)
    assert right.T @ np.diag((2, 2, 3, 3, 2, 2)) @ right == pytest.approx(
        np.eye(6), abs=2e-15
    )
    assert right[2, 1] / right[0, 1] == pytest.approx((math.sqrt(73) - 5) / 6)
    assert abs(right[2, 1] - right[0, 1]) > 0.1


def test_two_regional_means_fail_closure_but_four_fibers_retain_joint_row():
    graph = barbell()
    for node, phase in enumerate(
        (F(1, 8), F(1, 8), F(3, 16), F(1, 4), F(5, 16), F(5, 16))
    ):
        set_theta(graph, node, float(phase))
    default_compute_delta_nfr(graph)
    capture = capture_non_epi_forcing(graph)
    reference = derive_forced_support_balance(
        capture.snapshot, epi_weight=capture.epi_weight, forcing=capture.forcing
    )
    two = observe_forced_support_closure(reference, REGIONS)
    assert not two.all_state_affine_closed
    assert any(two.witness.projected_rate_difference)
    four = propose_joint_nodal_phase_step(graph, FIBERS, dt=H, coupling_strength=K)
    assert four.joint.closure.all_state_affine_closed
    assert four.counted_neighbors == ((0, 1), (0, 0, 2), (1, 3, 3), (2, 3))
    assert max(map(abs, four.phase_step_defect)) < F(1, 10**14)


def test_predictor_reads_source_jacobian_without_writes_or_hiding_topology():
    graph = barbell()
    nodes = deepcopy(dict(graph.nodes(data=True)))
    edges = deepcopy(tuple(graph.edges(data=True)))
    attributes = dict(graph.graph)
    weights = deepcopy(graph.graph["_dnfr_weights"])
    spectrum = {
        name: value.copy()
        for name, value in graph.graph["_tnfr_spectrum_cache"].items()
        if isinstance(value, np.ndarray)
    }
    result = predict_synchronized_joint_euler(graph, dt=H, coupling_strength=K, steps=0)
    assert dict(graph.nodes(data=True)) == nodes
    assert tuple(graph.edges(data=True)) == edges
    assert graph.graph.keys() == attributes.keys()
    assert all(graph.graph[key] is value for key, value in attributes.items())
    assert graph.graph["_dnfr_weights"] == weights
    for name, values in spectrum.items():
        np.testing.assert_array_equal(graph.graph["_tnfr_spectrum_cache"][name], values)
    assert result.capture.snapshot.topology_gradient == (
        F(1, 2),
        F(1, 2),
        -F(2, 3),
        -F(2, 3),
        F(1, 2),
        F(1, 2),
    )
    assert dict(result.capture.normalized_weights)["topo"] == 0
    mean = result.phase_reference.mean_response
    assert mean[2] == (F(1, 3), F(1, 3), 0, F(1, 3), 0, 0)


@pytest.mark.parametrize("strength", (F(1, 4), F(1, 2)))
def test_closed_modal_powers_match_independent_fine_linear_matrix(strength):
    result = predict_synchronized_joint_euler(
        barbell(), dt=H, coupling_strength=strength, steps=7
    )
    lap = np.eye(6) - np.asarray(result.phase_reference.mean_response, dtype=float)
    matrix = np.block(
        [
            [np.eye(6) - float(H) * 0.5 * lap, -float(H) / (2 * math.pi) * lap],
            [np.zeros((6, 6)), np.eye(6) - float(H * strength) * lap],
        ]
    )
    initial = np.r_[
        np.asarray(result.capture.snapshot.epi, float),
        np.asarray(result.capture.phase, float) - float(result.phase_origin),
    ]
    expected = np.linalg.matrix_power(matrix, 7) @ initial
    assert result.samples[7].epi == pytest.approx(expected[:6], abs=5e-16)
    assert result.samples[7].phase_offset == pytest.approx(expected[6:], abs=5e-16)


@pytest.mark.parametrize("steps", (-1, 257, True, 1.5))
def test_predictor_rejects_invalid_horizon(steps):
    with pytest.raises((TypeError, ValueError)):
        predict_synchronized_joint_euler(
            barbell(), dt=H, coupling_strength=K, steps=steps
        )


@pytest.mark.parametrize("key", ("dt", "coupling_strength"))
@pytest.mark.parametrize("value", (0, -1, True, math.nan, math.inf))
def test_predictor_rejects_invalid_scalars(key, value):
    kwargs = {"dt": H, "coupling_strength": K, "steps": 0, key: value}
    with pytest.raises((TypeError, ValueError)):
        predict_synchronized_joint_euler(barbell(), **kwargs)


@pytest.mark.parametrize(
    "mutation", ("topo", "weight", "capacity", "chart", "gate", "disconnected")
)
def test_predictor_rejects_laws_outside_its_shared_laplacian_domain(mutation):
    graph = barbell()
    if mutation == "topo":
        graph.graph["_dnfr_weights"]["topo"] = 0.125
    elif mutation == "weight":
        graph.edges[2, 3]["weight"] = 0.5
    elif mutation == "capacity":
        graph.nodes[2]["nu_f"] = 1.25
    elif mutation == "chart":
        set_theta(graph, 0, 2.0)
    elif mutation == "gate":
        graph.graph["UM_MAX_PHASE_DIFF"] = 0.001
    else:
        graph.remove_edge(2, 3)
    with pytest.raises((TypeError, ValueError)):
        predict_synchronized_joint_euler(graph, dt=H, coupling_strength=K, steps=0)


def test_zero_width_exact_synchrony_has_no_nonlinearity_bound():
    graph = triangle(phase=(0.25, 0.25))
    result = predict_synchronized_joint_euler(graph, dt=H, coupling_strength=K, steps=2)
    assert all(s.phase_error_upper == s.epi_error_upper == 0 for s in result.samples)
    assert max(map(abs, result.samples[-1].phase_offset)) < HEADROOM


@pytest.mark.parametrize("strength,capacity", ((2, 1), (F(1, 2), 4)))
def test_predictor_checks_both_monotone_step_ceilings(strength, capacity):
    graph = triangle(capacity=capacity)
    with pytest.raises((TypeError, ValueError)):
        predict_synchronized_joint_euler(
            graph, dt=F(1, 2), coupling_strength=strength, steps=0
        )
