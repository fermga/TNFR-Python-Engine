"""Independent lock/source controls; small runtime residuals are not proofs.

Exact cut algebra, high-precision phase geometry and finite production steps
have separate assertions. No target is fitted and no recovery sweep is run.
"""

import math
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import mpmath as mp
import networkx as nx
import pytest

from tests.joint_phase_helpers import configure, execute_joint_step
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.canonical import compute_phase_gradient
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.phase_response import derive_phase_response, observe_phase_lock_source
from tnfr.utils import angle_diff

TOL = 3e-15  # finite binary64 comparison, not a lock-admission tolerance


def _prepare(graph, phases, capacities=None, *, weights=None):
    configure(graph, **({"weights": weights} if weights is not None else {}))
    for _, _, data in graph.edges(data=True):
        data.setdefault("weight", 1.0)
        data["length"] = 1.0
    if capacities is None:
        capacities = (1.0,) * len(graph)
    for node, phase, capacity in zip(graph, phases, capacities, strict=True):
        graph.nodes[node].update(
            EPI=0.0, theta=float(phase), nu_f=float(capacity), delta_nfr=0.0, dEPI=0.0
        )
    default_compute_delta_nfr(graph)
    return graph


def _balance(observation):
    capture = observation.capture
    return derive_forced_support_balance(
        capture.snapshot, epi_weight=capture.epi_weight, forcing=capture.forcing
    )


@contextmanager
def _unchanged(graph):
    nodes = deepcopy(dict(graph.nodes(data=True)))
    edges = deepcopy(tuple(graph.edges(data=True)))
    metadata = dict(graph.graph)
    weights = deepcopy(graph.graph["_dnfr_weights"])
    yield
    assert dict(graph.nodes(data=True)) == nodes
    assert tuple(graph.edges(data=True)) == edges
    assert graph.graph.keys() == metadata.keys()
    assert all(graph.graph[key] is value for key, value in metadata.items())
    assert graph.graph["_dnfr_weights"] == weights


def test_irregular_support_cancellation_and_cut_load_are_exact_coefficient_identities():
    graph = _prepare(nx.star_graph(3), (1, 1.25, 1.5, 1.75), (1, 2, 3, 4))
    with _unchanged(graph):
        result = observe_phase_lock_source(graph, coupling_strength=0.75)
    assert result.phase_degrees == (3, 1, 1, 1)
    assert result.common_rate == 2
    assert sum(result.full_support_sine) == 0
    assert (
        sum(
            d * r
            for d, r in zip(result.phase_degrees, result.full_support_rate_residual)
        )
        == 0
    )
    # S={center,leaf1}: internal sine terms cancel, two boundary terms remain.
    cut_load = sum(
        result.phase_degrees[i]
        * (result.common_rate - result.capture.snapshot.capacity[i])
        for i in (0, 1)
    )
    cut_residual = sum(
        result.phase_degrees[i] * result.full_support_rate_residual[i] for i in (0, 1)
    )
    boundary = F(math.sin(0.5)) + F(math.sin(0.75))
    assert cut_load + cut_residual == F(3, 4) * boundary
    assert result.actual_admitted_rate_residual == result.full_support_rate_residual
    assert any(result.sine_reduction_defect)
    assert not result.homogeneous_capacity
    with pytest.raises(FrozenInstanceError):
        result.common_rate = 0


@pytest.mark.parametrize("size", [5, 6])
def test_winding_has_nonuniform_phase_geometry_but_no_ideal_phase_source(size):
    graph = _prepare(nx.cycle_graph(size), [math.tau * j / size for j in range(size)])
    result = observe_phase_lock_source(graph, coupling_strength=0.5)
    assert result.full_u3_admission and result.strict_acute_edges_estimate
    assert result.common_rate == 1
    assert result.homogeneous_capacity and result.positive_capacities
    assert result.positive_transport_connected
    assert result.lock_source_estimate == (0.0,) * size
    assert max(map(abs, result.full_support_rate_residual)) < TOL
    assert max(map(abs, result.capture.phase_gradient)) < TOL
    assert tuple(map(float, result.relative_real)) == pytest.approx(
        (2 * math.cos(math.tau / size),) * size, abs=TOL
    )
    # This observable distinguishes phase geometry from scalar form/source.
    gradient = compute_phase_gradient(graph)
    assert tuple(gradient.values()) == pytest.approx((math.tau / size,) * size, abs=TOL)
    winding = sum(
        angle_diff(graph.nodes[(j + 1) % size]["theta"], graph.nodes[j]["theta"])
        for j in range(size)
    )
    assert winding == pytest.approx(math.tau, abs=TOL)
    step = execute_joint_step(graph, dt=F(1, 8), coupling_strength=0.5, time=0)
    assert max(map(abs, step.after_epi.epi)) < TOL
    assert step.phase_after == pytest.approx(
        tuple((float(p) + 0.125) % math.tau for p in result.capture.phase), abs=TOL
    )
    # The symbolic C6 Gram independently certifies nonzero ideal resultants.
    if size == 6:
        row = (F(1), F(1, 2), F(-1, 2), F(-1), F(-1, 2), F(1, 2))
        reference = derive_phase_response(
            cosine_gram=tuple(
                tuple(row[(j - i) % 6] for j in range(6)) for i in range(6)
            ),
            mean_neighbors=tuple(((i - 1) % 6, (i + 1) % 6) for i in range(6)),
            receiver_sources=tuple((i,) for i in range(6)),
            phase_factor=1,
        )
        assert reference.mean_resultant_squared == (1,) * 6


def test_quarter_turn_winding_is_not_admitted_as_an_exact_zero_source():
    row = (1, 0, -1, 0)
    with pytest.raises(ValueError, match="nonzero resultant"):
        derive_phase_response(
            cosine_gram=tuple(
                tuple(row[(j - i) % 4] for j in range(4)) for i in range(4)
            ),
            mean_neighbors=tuple(((i - 1) % 4, (i + 1) % 4) for i in range(4)),
            receiver_sources=tuple((i,) for i in range(4)),
            phase_factor=1,
        )
    graph = _prepare(nx.cycle_graph(4), (0, math.pi / 2, math.pi, 3 * math.pi / 2))
    result = observe_phase_lock_source(graph, coupling_strength=1)
    assert not result.strict_acute_edges_estimate
    assert result.lock_source_estimate == (None,) * 4
    assert max(map(abs, result.relative_real)) < TOL
    assert max(map(abs, result.full_support_sine)) < TOL
    # Rounded tiny resultants are retained, never promoted to ideal nondegeneracy.
    for actual, estimate, defect in zip(
        result.capture.phase_gradient,
        result.relative_phase_source_estimate,
        result.production_source_discrepancy,
    ):
        assert defect == (actual - F(estimate) if estimate is not None else None)


def test_equal_capacity_gated_free_motion_can_retain_nonzero_full_support_pressure():
    graph = _prepare(nx.path_graph(2), (1, 1.5))
    graph.graph["UM_MAX_PHASE_DIFF"] = 0.25
    result = observe_phase_lock_source(graph, coupling_strength=1)
    assert not result.full_u3_admission
    assert result.strict_acute_edges_estimate
    assert result.actual_admitted_rate_residual == (0, 0)
    assert result.full_support_rate_residual == (F(math.sin(0.5)), -F(math.sin(0.5)))
    assert result.lock_source_estimate == (None, None)
    assert tuple(map(float, result.capture.phase_gradient)) == pytest.approx(
        (0.5 / math.pi, -0.5 / math.pi), abs=TOL
    )
    step = execute_joint_step(graph, dt=0.125, coupling_strength=1, time=0)
    assert step.phase_after == (1.125, 1.625)
    assert float(step.after_epi.epi[0]) == pytest.approx(1 / (32 * math.pi), abs=TOL)
    assert step.after_epi.epi[0] > 0 > step.after_epi.epi[1]


def test_partial_gate_changes_the_phase_denominator_and_reports_reduction_separately():
    graph = _prepare(nx.path_graph(3), (1, 1.125, 1.5))
    graph.graph["UM_MAX_PHASE_DIFF"] = 0.25
    result = observe_phase_lock_source(graph, coupling_strength=0.5)
    assert result.actual_admitted_rate_residual == (
        F(math.sin(0.125)) / 2,
        -F(math.sin(0.125)) / 2,
        0,
    )
    assert (
        result.full_support_rate_residual[1]
        == (F(math.sin(0.375)) - F(math.sin(0.125))) / 4
    )
    assert result.sine_reduction_defect == (0, 0, 0)


def test_equal_capacity_off_lock_source_obeys_the_independent_acute_residual_bound():
    graph = _prepare(nx.star_graph(3), (1, 1.25, 1.5, 1.75))
    result = observe_phase_lock_source(graph, coupling_strength=0.5)
    assert result.lock_source_estimate == (0.0,) * 4
    assert any(result.full_support_rate_residual)
    assert any(result.capture.phase_gradient)
    # High-precision ideal trigonometry, not rationalized producer coefficients.
    with mp.workdps(80):
        phase = tuple(mp.mpf(v) for v in (1, 1.25, 1.5, 1.75))
        for i, row in enumerate(result.capture.snapshot.support_neighbors):
            real = sum(mp.cos(phase[j] - phase[i]) for j in row)
            imag = sum(mp.sin(phase[j] - phase[i]) for j in row)
            residual = imag / (2 * len(row))
            source = mp.atan2(imag, real) / mp.pi
            bound = abs(residual) / (mp.pi * mp.mpf("0.5") * mp.cos(mp.mpf("0.75")))
            assert abs(source) <= bound
            assert float(result.full_support_rate_residual[i]) == pytest.approx(
                float(residual), abs=TOL
            )
            assert float(result.capture.phase_gradient[i]) == pytest.approx(
                float(source), abs=TOL
            )


def test_nonacute_antipodal_boundary_is_dropped_by_real_u3():
    graph = _prepare(nx.path_graph(2), (0, math.pi))
    result = observe_phase_lock_source(graph, coupling_strength=1)
    assert not result.full_u3_admission
    assert not result.strict_acute_edges_estimate
    assert result.actual_admitted_rate_residual == (0, 0)
    assert result.lock_source_estimate == (None, None)
    assert all(float(value) < 0 for value in result.relative_real)
    assert tuple(abs(float(v)) for v in result.capture.phase_gradient) == (1, 1)


def test_heterogeneous_star_lock_has_nonzero_source_and_incompatible_stationary_epi():
    graph = _prepare(
        nx.star_graph(3),
        (math.pi, math.pi + math.pi / 6, math.pi + math.pi / 6, math.pi - math.pi / 6),
        (F(5, 6), F(3, 2), F(3, 2), F(1, 2)),
    )
    result = observe_phase_lock_source(graph, coupling_strength=1)
    assert result.full_u3_admission and result.strict_acute_edges_estimate
    assert max(map(abs, result.actual_admitted_rate_residual)) < TOL
    with mp.workdps(80):
        gamma = mp.atan(1 / (3 * mp.sqrt(3)))
        expected = tuple(
            map(float, (gamma / mp.pi, -mp.mpf(1) / 6, -mp.mpf(1) / 6, mp.mpf(1) / 6))
        )
        expected_compatibility = float((3 * gamma / mp.pi - mp.mpf(1) / 6) / 2)
    assert tuple(map(float, result.capture.phase_gradient)) == pytest.approx(
        expected, abs=TOL
    )
    assert result.lock_source_estimate == pytest.approx(expected, abs=TOL)
    balance = _balance(result)
    assert not balance.has_zero_pressure_equilibrium
    assert float(balance.compatibility_residual) == pytest.approx(
        expected_compatibility, abs=TOL
    )
    assert balance.compatibility_residual > 0 and balance.mean_drift > 0
    # One real nodal step confirms the predicted mean drive, not a new solver.
    step = execute_joint_step(graph, dt=0.125, coupling_strength=1, time=0)
    change = sum(h * x for h, x in zip(balance.metric_weights, step.after_epi.epi))
    assert float(change) == pytest.approx(
        float(balance.compatibility_residual) / 8, abs=TOL
    )


def test_phase_counts_do_not_become_transport_strengths_or_drop_zero_weight_edges():
    graph = nx.path_graph(3)
    graph[0][1]["weight"], graph[1][2]["weight"] = 2.0, 0.0
    _prepare(graph, (1, 1.25, 1.5), (1, 2, 4))
    result = observe_phase_lock_source(graph, coupling_strength=1)
    assert result.phase_degrees == (1, 2, 1)
    assert result.common_rate == F(9, 4)
    assert result.full_u3_admission
    assert not result.positive_transport_connected
    assert result.full_support_sine[2] == -F(math.sin(0.25))
    with pytest.raises(ValueError):
        _balance(result)
    graph[1][2]["weight"] = 4.0
    default_compute_delta_nfr(graph)
    connected = observe_phase_lock_source(graph, coupling_strength=1)
    assert connected.common_rate == result.common_rate
    assert connected.phase_degrees == result.phase_degrees
    assert connected.full_support_rate_residual == result.full_support_rate_residual
    assert connected.positive_transport_connected
    balance = _balance(connected)
    assert balance.strengths == (2, 6, 4)
    assert balance.compatibility_residual == sum(
        s * f for s, f in zip((2, 6, 4), connected.capture.forcing)
    )
    assert balance.compatibility_residual != sum(
        d * f for d, f in zip(connected.phase_degrees, connected.capture.forcing)
    )


@pytest.mark.parametrize("channel", ["vf", "topo"])
def test_other_pressure_is_visible_even_when_phase_source_vanishes(channel):
    weights = {"epi": 1.0, "phase": 1.0, "vf": 0.0, "topo": 0.0}
    weights[channel] = 1.0
    graph = _prepare(nx.path_graph(3), (1, 1, 1), (1, 2, 4), weights=weights)
    result = observe_phase_lock_source(graph, coupling_strength=1)
    assert result.capture.phase_gradient == (0, 0, 0)
    assert any(result.other_forcing)
    assert result.capture.forcing == result.other_forcing


def test_zero_capacity_is_observed_without_promoting_stationarity_to_equilibrium():
    graph = _prepare(nx.path_graph(2), (1, 1.5), (0, 0))
    result = observe_phase_lock_source(graph, coupling_strength=1)
    assert result.homogeneous_capacity and not result.positive_capacities
    assert any(result.capture.forcing)
    step = execute_joint_step(graph, dt=0.125, coupling_strength=1, time=0)
    assert step.after_epi.epi == (0, 0)


def test_lock_source_ratio_retains_subnormal_detuning_before_float_conversion():
    tiny = math.ulp(0.0)
    graph = _prepare(nx.path_graph(2), (0, 0), (tiny, 2 * tiny))
    result = observe_phase_lock_source(graph, coupling_strength=tiny)
    assert result.common_rate == 3 * F(tiny) / 2
    assert result.lock_source_estimate == pytest.approx(
        (math.atan(0.5) / math.pi, -math.atan(0.5) / math.pi), abs=1e-16
    )
    # This is a conditional estimate, not an observed lock; actual source is zero.
    assert result.capture.phase_gradient == (0, 0)
    assert any(result.full_support_rate_residual)


@pytest.mark.parametrize("strength", [0, -1, float("nan"), float("inf"), True])
def test_invalid_coupling_is_rejected_without_writes(strength):
    graph = _prepare(nx.path_graph(2), (1, 1))
    with _unchanged(graph):
        with pytest.raises((ValueError, TypeError)):
            observe_phase_lock_source(graph, coupling_strength=strength)


@pytest.mark.parametrize("phase", [-0.125, math.tau])
def test_noncanonical_raw_phase_is_rejected(phase):
    graph = _prepare(nx.path_graph(2), (1, 1))
    graph.nodes[0]["theta"] = phase
    with pytest.raises(ValueError, match="canonical raw phases"):
        observe_phase_lock_source(graph, coupling_strength=1)


@pytest.mark.parametrize(
    "graph",
    [
        nx.empty_graph(1),
        nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)),
        nx.cycle_graph(6, create_using=nx.DiGraph),
    ],
)
def test_disconnected_empty_or_nonreciprocal_phase_support_is_rejected(graph):
    _prepare(graph, (1,) * len(graph))
    with pytest.raises(ValueError):
        observe_phase_lock_source(graph, coupling_strength=1)
