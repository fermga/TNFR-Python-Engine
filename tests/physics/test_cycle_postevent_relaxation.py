"""Prospective cycle bounds and finite continuation of the actual UM event.

The exact-real theorem and observed binary64 path are separate evidence. A
single frozen h=1/8, K=1/2, T=32 continuation reuses production evolution;
there is no target fit, second solver, event repetition or mesh campaign.
"""

import copy
import math
from dataclasses import replace
from fractions import Fraction

import mpmath as mp
import networkx as nx
import pytest

from tests.joint_phase_helpers import (
    execute_coupling_cycle_birth,
    execute_joint_step,
    triangle,
)
from tnfr.constants import DEFAULTS
from tnfr.physics.cycle_relaxation import bound_cycle_relaxation
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.support_transport import observe_support_transport_euler
from tnfr.physics.winding_certificates import certify_phase_winding
from tnfr.utils import angle_diff


def _mp(value):
    value = Fraction(value)
    return mp.mpf(value.numerator) / value.denominator


def _measure(graph, envelope):
    source = capture_non_epi_forcing(graph)
    s = envelope.strengths
    mean = sum(a * b for a, b in zip(s, source.snapshot.epi)) / sum(s)
    disagreement = sum(a * (b - mean) ** 2 for a, b in zip(s, source.snapshot.epi))
    gaps = tuple(
        angle_diff(graph.nodes[(i + 1) % 5]["theta"], graph.nodes[i]["theta"])
        for i in range(5)
    )
    return {
        "mean": mean,
        "disagreement": disagreement,
        "gaps": gaps,
        "gap_squared": sum((d - math.tau / 5) ** 2 for d in gaps),
        "source_squared": sum(v**2 for v in source.phase_gradient),
    }


@pytest.fixture(scope="module")
def continuation():
    case = execute_coupling_cycle_birth()
    graph = case["graph"]
    # All bounds precede execution and use the actual nonunit closing edge.
    envelope = bound_cycle_relaxation(
        graph, range(5), coupling_strength=Fraction(1, 2), times=(0, 8, 16, 32)
    )
    initial_nodes = copy.deepcopy(dict(graph.nodes(data=True)))
    initial_edges = copy.deepcopy(
        dict(((i, j), data) for i, j, data in graph.edges(data=True))
    )
    observed = {0: _measure(graph, envelope)}
    peak_disagreement = Fraction(0)
    max_state_defect = Fraction(0)
    max_pressure_defect = Fraction(0)
    min_gap = min(observed[0]["gaps"])
    max_gap = max(observed[0]["gaps"])
    for index in range(256):
        step = execute_joint_step(
            graph, dt=Fraction(1, 8), coupling_strength=0.5, time=Fraction(index, 8)
        )
        assert not any(step.before.stored_pressure_residual)
        budget = observe_support_transport_euler(
            step.before.snapshot, step.after_epi, Fraction(1, 8)
        )
        assert budget.identity_residual == 0
        max_state_defect = max(max_state_defect, *map(abs, budget.state_defect))
        max_pressure_defect = max(
            max_pressure_defect, *map(abs, step.before.kernel_pressure_defect)
        )
        assert all(-1 < value < 1 for value in budget.expected_epi)
        assert all(-1 < value < 1 for value in step.after_epi.epi)
        winding = certify_phase_winding(graph, range(5))
        assert winding.winding == 1 and winding.u3_admissible
        assert winding.minimum_u3_margin > 0
        current = _measure(graph, envelope)
        # This is a finite observation with a stated rounding allowance,
        # not promotion of the continuous maximum principle to all runtime.
        assert min(current["gaps"]) >= min_gap - 2e-13
        assert max(current["gaps"]) <= max_gap + 2e-13
        peak_disagreement = max(peak_disagreement, current["disagreement"])
        if index + 1 in (64, 128, 256):
            observed[(index + 1) // 8] = current
    assert (
        dict(((i, j), data) for i, j, data in graph.edges(data=True)) == initial_edges
    )
    for node in graph:
        assert (
            graph.nodes[node]["glyph_history"] == initial_nodes[node]["glyph_history"]
        )
        assert graph.nodes[node]["nu_f"] == initial_nodes[node]["nu_f"]
    return envelope, observed, peak_disagreement, max_state_defect, max_pressure_defect


def test_post_event_continuation_matches_prospective_envelopes(continuation):
    envelope, observed, peak, state_defect, pressure_defect = continuation
    assert envelope.winding == 1
    assert envelope.initial_epi_disagreement_squared == 0
    assert envelope.mean_rate_prefactor_squared_upper > 0
    assert -1 < envelope.all_time_epi_interval[0] < envelope.weighted_mean
    assert envelope.weighted_mean < envelope.all_time_epi_interval[1] < 1
    assert state_defect < Fraction(1, 10**15)
    assert pressure_defect < Fraction(1, 10**14)
    for sample in envelope.samples:
        actual = observed[int(sample.time)]
        assert (
            actual["gap_squared"] <= float(sample.gap_deviation_squared_upper) + 2e-13
        )
        assert (
            float(actual["source_squared"])
            <= float(sample.phase_source_squared_upper) + 2e-13
        )
        assert (
            float(actual["disagreement"])
            <= float(sample.epi_disagreement_squared_upper) + 2e-13
        )
        displacement = (actual["mean"] - envelope.weighted_mean) ** 2
        assert (
            float(displacement) <= float(sample.mean_displacement_squared_upper) + 2e-13
        )
    assert observed[32]["gap_squared"] < observed[0]["gap_squared"] / 100
    assert observed[32]["source_squared"] < observed[0]["source_squared"] / 100
    assert 0 < observed[32]["disagreement"] < peak / 10
    # The closing weight genuinely drives the mean. No final target is fitted.
    assert observed[32]["mean"] > envelope.weighted_mean


def test_exact_bounds_against_independent_high_precision_algebra(continuation):
    envelope = continuation[0]
    with mp.workdps(110):
        gaps = tuple(_mp(a) + b * mp.pi for a, b in envelope.gap_affine)
        for value, interval in zip(gaps, envelope.gap_enclosures):
            assert _mp(interval[0]) <= value <= _mp(interval[1])
        centered = tuple(g - 2 * mp.pi / 5 for g in gaps)
        q2 = sum(v**2 for v in centered)
        assert q2 <= _mp(envelope.initial_gap_deviation_squared_upper)
        assert _mp(envelope.cosine_lower_bound) <= mp.cos(max(map(abs, gaps)))
        assert _mp(envelope.phase_laplacian_gap_lower_bound) <= 2 - 2 * mp.cos(
            2 * mp.pi / 5
        )
        gamma = _mp(envelope.phase_decay_rate_lower_bound)
        beta = _mp(envelope.epi_decay_rate_lower_bound)
        for sample in envelope.samples:
            t = _mp(sample.time)
            exact_integral = (mp.exp(-gamma * t) - mp.exp(-beta * t)) / (beta - gamma)
            assert exact_integral <= _mp(sample.duhamel_integral_upper)
            assert _mp(
                envelope.form_forcing_prefactor_squared_upper
            ) * exact_integral**2 <= _mp(sample.epi_disagreement_squared_upper)
            assert _mp(envelope.mean_rate_prefactor_squared_upper) * mp.exp(
                -2 * gamma * t
            ) / gamma**2 <= _mp(sample.mean_tail_squared_upper)
        # Check the source and weighted-mean identity independently of the
        # production phasor/pressure implementation, at the embedded initial state.
        source = [(gaps[i] - gaps[i - 1]) / (2 * mp.pi) for i in range(5)]
        strengths = list(map(_mp, envelope.strengths))
        derivative = sum(s * g for s, g in zip(strengths, source)) / (
            2 * sum(strengths)
        )
        assert 0 < derivative**2 <= _mp(envelope.mean_rate_prefactor_squared_upper)
    # An independent high-precision eigenvalue is a comparison, not the proof.
    # Binary64 eigenvalues cannot reliably order a sharp exact lower bound.
    with mp.workdps(110):
        s = list(map(_mp, envelope.strengths))
        b = mp.diag(s)
        for i, j, weight in envelope.capture.snapshot.conductance:
            b[i, j] -= _mp(weight)
        symmetric = mp.matrix(5)
        for i in range(5):
            for j in range(5):
                symmetric[i, j] = b[i, j] / (2 * mp.sqrt(s[i] * s[j]))
        assert (
            0
            < _mp(envelope.epi_decay_rate_lower_bound)
            <= mp.eigsy(symmetric, eigvals_only=True)[1]
        )


def test_retained_cycle_has_shape_beyond_winding_and_a_fixed_rotation_frame(
    continuation,
):
    """Use the retained endpoint; do not execute another scientific trace."""
    envelope = continuation[0]
    shape = envelope.initial_phase_shape_affine
    assert tuple(sum(row[j] for row in shape) for j in (0, 1)) == (0, 0)
    for i, (rational, coefficient) in enumerate(envelope.gap_affine):
        right, left = shape[(i + 1) % len(shape)], shape[i]
        assert (right[0] - left[0], right[1] - left[1]) == (
            rational,
            coefficient - envelope.mean_gap_pi_coefficient,
        )
    with mp.workdps(110):
        phases = [_mp(envelope.capture.phase[i]) for i in envelope.cycle_indices]
        # Independently unwrap the represented phases with high-precision pi,
        # rather than reusing the certificate's branch integers.
        gaps = [
            (phases[(i + 1) % 5] - phases[i] + mp.pi) % (2 * mp.pi) - mp.pi
            for i in range(5)
        ]
        lifted = [phases[0]]
        for gap in gaps[:-1]:
            lifted.append(lifted[-1] + gap)
        untwisted = [value - i * 2 * mp.pi / 5 for i, value in enumerate(lifted)]
        offset = mp.fsum(untwisted) / 5
        centered = [value - offset for value in untwisted]
        a, b = envelope.initial_phase_offset_affine
        assert mp.almosteq(offset, _mp(a) + _mp(b) * mp.pi)
        for exact, (a, b), (low, high) in zip(
            centered, shape, envelope.initial_phase_shape_enclosures, strict=True
        ):
            assert mp.almosteq(exact, _mp(a) + _mp(b) * mp.pi)
            if low == high:
                # The exact rational enclosure is a singleton. Independent
                # high-precision subtraction still has its own roundoff.
                assert b == 0 and low == a
                assert mp.almosteq(exact, _mp(low))
            else:
                assert _mp(low) <= exact <= _mp(high)
        shape_squared = mp.fsum(value**2 for value in centered)
        gap_squared = mp.fsum((gap - 2 * mp.pi / 5) ** 2 for gap in gaps)
        true_gap = 2 - 2 * mp.cos(2 * mp.pi / 5)
        # Winding one and initially uniform EPI do not imply a regular twist.
        assert envelope.winding == 1
        assert envelope.initial_epi_disagreement_squared == 0
        assert 0 < shape_squared <= gap_squared / true_gap
        assert gap_squared / true_gap <= _mp(
            envelope.phase_orbit_distance_squared_upper[0]
        )
        gamma = _mp(envelope.phase_decay_rate_lower_bound)
        for sample, upper in zip(
            envelope.samples, envelope.phase_orbit_distance_squared_upper, strict=True
        ):
            assert gap_squared * mp.exp(
                -2 * gamma * _mp(sample.time)
            ) / true_gap <= _mp(upper)


def test_orientation_relabeling_and_unit_weight_mean_control():
    graph = execute_coupling_cycle_birth()["graph"]
    before = repr(
        (graph.graph, list(graph.nodes(data=True)), list(graph.edges(data=True)))
    )
    forward = bound_cycle_relaxation(
        graph, range(5), coupling_strength=0.5, times=(0, 1)
    )
    assert (
        repr((graph.graph, list(graph.nodes(data=True)), list(graph.edges(data=True))))
        == before
    )
    labels = {i: f"vertex-{i}" for i in graph}
    relabeled = nx.relabel_nodes(graph, labels)
    backward = bound_cycle_relaxation(
        relabeled,
        [labels[i] for i in reversed(range(5))],
        coupling_strength=0.5,
        times=(0, 1),
    )
    assert backward.winding == -forward.winding
    assert (
        backward.initial_gap_deviation_squared_upper
        == forward.initial_gap_deviation_squared_upper
    )
    assert backward.samples == forward.samples
    assert backward.phase_orbit_distance_squared_upper == (
        forward.phase_orbit_distance_squared_upper
    )
    forward_shape = dict(zip(forward.cycle_order, forward.initial_phase_shape_affine))
    backward_shape = dict(
        zip(backward.cycle_order, backward.initial_phase_shape_affine)
    )
    assert {i: backward_shape[labels[i]] for i in graph} == forward_shape
    nx.set_edge_attributes(graph, 1.0, "weight")
    unit = bound_cycle_relaxation(graph, range(5), coupling_strength=0.5, times=(0, 1))
    assert unit.mean_rate_prefactor_squared_upper == 0
    assert unit.mean_limit_offset_upper == 0
    assert all(
        s.mean_displacement_squared_upper == s.mean_tail_squared_upper == 0
        for s in unit.samples
    )
    assert unit.epi_decay_rate_lower_bound != forward.epi_decay_rate_lower_bound


@pytest.mark.parametrize("topology_weight", [0.0, 0.25], ids=["default", "topology"])
def test_full_mix_has_the_same_envelope_when_its_extra_channels_vanish(
    continuation, topology_weight
):
    retained = continuation[0]
    source = retained.capture.snapshot
    # Reuse the initial materialized state/support projection already retained
    # by the single event/continuation fixture. This is no new event or trace.
    graph = nx.Graph()
    graph.graph.update(
        DNFR_WEIGHTS={**DEFAULTS["DNFR_WEIGHTS"], "topo": topology_weight},
        DELTA_PHI_MAX=float(retained.effective_phase_gate),
        UM_MAX_PHASE_DIFF=float(retained.effective_phase_gate),
    )
    for node, epi, capacity, phase in zip(
        source.nodes, source.epi, source.capacity, retained.capture.phase, strict=True
    ):
        graph.add_node(
            node,
            EPI=float(epi),
            nu_f=float(capacity),
            theta=float(phase),
            delta_nfr=0.0,
        )
    graph.add_edges_from(
        (source.nodes[i], source.nodes[j], {"weight": float(weight)})
        for i, j, weight in source.conductance
        if i < j
    )
    full = bound_cycle_relaxation(
        graph,
        retained.cycle_order,
        coupling_strength=retained.coupling_strength,
        times=(0, 8),
    )
    assert full.capture.snapshot.conductance == source.conductance
    assert any(weight != 1 for _, _, weight in source.conductance)
    assert full.capture.snapshot.capacity_gradient == (0,) * len(source.nodes)
    assert full.capture.snapshot.topology_gradient == (0,) * len(source.nodes)
    effective = dict(full.capture.normalized_weights)
    assert effective["vf"] > 0
    assert (effective["topo"] > 0) == (topology_weight > 0)

    reference_graph = graph.copy()
    # Explicitly preserve effective phase/EPI gains while removing the inert
    # terms. Re-normalizing a public recipe would define a different model.
    reference_graph.graph["_dnfr_weights"] = {
        name: float(value) if name in ("phase", "epi") else 0.0
        for name, value in effective.items()
    }
    reference = bound_cycle_relaxation(
        reference_graph,
        retained.cycle_order,
        coupling_strength=retained.coupling_strength,
        times=(0, 8),
    )
    assert full.capture.forcing == reference.capture.forcing
    assert full.capture.full_kernel_pressure == reference.capture.full_kernel_pressure
    assert replace(full, capture=reference.capture) == reference


def test_zero_source_and_equal_decay_rates_have_regular_bounds():
    # Unit K3 has lambda_C=3, lambda_rw=3/2. rho=0, K=1/2,
    # e=1/2, capacity=1 make the certified rates exactly equal.
    graph = triangle(phase=(0.25, 0.25), epi=(0.25, -0.125, 0.375))
    report = bound_cycle_relaxation(
        graph, range(3), coupling_strength=0.5, times=(0, 1, 2)
    )
    assert report.winding == 0
    assert report.initial_gap_deviation_squared_upper == 0
    assert report.initial_phase_offset_affine == (Fraction(1, 4), 0)
    assert report.initial_phase_shape_affine == ((0, 0),) * 3
    assert report.phase_orbit_distance_squared_upper == (0,) * 3
    assert report.phase_decay_rate_lower_bound == report.epi_decay_rate_lower_bound
    assert report.mean_limit_offset_upper == 0
    assert (
        report.samples[0].epi_disagreement_squared_upper
        == report.initial_epi_disagreement_squared
    )
    with mp.workdps(80):
        for sample in report.samples:
            t = _mp(sample.time)
            exact = t * mp.exp(-_mp(report.phase_decay_rate_lower_bound) * t)
            assert exact <= _mp(sample.duhamel_integral_upper)
    assert (
        report.samples[2].epi_disagreement_squared_upper
        < report.samples[1].epi_disagreement_squared_upper
    )


@pytest.mark.parametrize(
    "mutation", ["chord", "zero_edge", "capacity", "gate", "branch", "zero_epi"]
)
def test_outside_conditional_model_is_rejected(mutation):
    graph = execute_coupling_cycle_birth()["graph"]
    if mutation == "chord":
        graph.add_edge(0, 2, weight=1.0)
    elif mutation == "zero_edge":
        graph.edges[0, 4]["weight"] = 0.0
    elif mutation == "capacity":
        graph.nodes[0]["nu_f"] = 2.0
    elif mutation == "gate":
        graph.graph["UM_MAX_PHASE_DIFF"] = 0.1
    elif mutation == "branch":
        graph.nodes[1]["theta"] = math.pi
    else:
        graph.graph["_dnfr_weights"] = {
            "phase": 0.5,
            "epi": 0.0,
            "vf": 0.5,
            "topo": 0.0,
        }
    with pytest.raises(ValueError):
        bound_cycle_relaxation(graph, range(5), coupling_strength=0.5, times=(0, 1))


@pytest.mark.parametrize("times", [(), (1, 0), (-1,), (0,) * 258, (100000,)])
def test_evaluation_budget_and_time_domain(times):
    graph = triangle(phase=(0.25, 0.25))
    with pytest.raises(ValueError):
        bound_cycle_relaxation(graph, range(3), coupling_strength=0.5, times=times)


def test_stored_operator_pressure_is_reported_without_changing_reference():
    case = execute_coupling_cycle_birth()
    raw = bound_cycle_relaxation(
        case["raw_graph"], range(5), coupling_strength=0.5, times=(0, 1)
    )
    fresh = bound_cycle_relaxation(
        case["graph"], range(5), coupling_strength=0.5, times=(0, 1)
    )
    assert any(raw.capture.stored_pressure_residual)
    assert not any(fresh.capture.stored_pressure_residual)
    assert raw.samples == fresh.samples
    assert raw.gap_affine == fresh.gap_affine
