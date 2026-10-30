"""Frozen six-node recovery control for the geometry-limited regional lock.

Targets, horizon and acceptance follow the declared cut and restoration proof.
Every nonlinear step uses shared phase, pressure and nodal-integration owners.
The numerical allowances are finite controls, not runtime stability certificates.
"""

import math
from copy import deepcopy
from fractions import Fraction as F
from types import SimpleNamespace

import mpmath as mp
import networkx as nx
import numpy as np
import pytest

from tests.joint_phase_helpers import execute_joint_step, unit_barbell
from tnfr.alias import set_theta
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.regional_phase_lock import derive_regional_phase_lock
from tnfr.physics.support_transport import (
    observe_regional_support_euler,
    observe_support_transport_euler,
)
from tnfr.utils import angle_diff

REGIONS = ((0, 1, 2), (3, 4, 5))
H, K, EPSILON, STEPS = F(1, 8), F(1, 2), F(1, 32), 256
PHASE_PERTURBATION = tuple(F(v, 128) for v in (2, -1, 1, -1, 1, -2))
FORM_PERTURBATION = tuple(F(v, 256) for v in (31, 31, 31, -33, -33, -33))
HEADROOM = 1e-11  # separate finite binary64 allowance across phase wraps


def _model(epsilon=EPSILON, strength=K):
    graph = unit_barbell(capacities=(1 - epsilon, 1 + epsilon))
    return graph, derive_regional_phase_lock(graph, REGIONS, coupling_strength=strength)


def _norm(values, weights):
    return float(
        np.sqrt(np.dot(np.asarray(weights, float), np.asarray(values, float) ** 2))
    )


def _phase_error(capture, model, index):
    return tuple(
        angle_diff(
            float(value),
            (target + float(index * H * model.common_phase_rate)) % math.tau,
        )
        for value, target in zip(
            capture.phase, model.target_phase_estimate, strict=True
        )
    )


@pytest.fixture(
    scope="module", params=(False, True), ids=("prepared-lock", "six-node-perturbation")
)
def executed(request):
    graph, model = _model()
    assert model.lock_status == "strict" and model.target_region_gate_admitted
    # Freeze ideal recovery inequalities before examining nonlinear trajectories.
    phase_rate, form_rate = F(1759, 20480), F(31, 320)
    a, b = 1 - H * phase_rate, 1 - H * form_rate
    assert a**STEPS < F(1, 10)
    ratio_bound = b**STEPS + (a**STEPS - b**STEPS) / (23 * (form_rate - phase_rate))
    assert ratio_bound < F(1, 5)
    for i in graph:
        graph.nodes[i]["EPI"] = float(
            model.target_epi[i] + (FORM_PERTURBATION[i] if request.param else 0)
        )
        set_theta(
            graph,
            i,
            model.target_phase_estimate[i]
            + (float(PHASE_PERTURBATION[i]) if request.param else 0),
        )
    default_compute_delta_nfr(graph)
    captures, budgets, regional = [], [], []
    for n in range(STEPS):
        step = execute_joint_step(graph, dt=H, coupling_strength=K, time=n * H)
        captures.append(step.before)
        budgets.append(
            observe_support_transport_euler(step.before.snapshot, step.after_epi, H)
        )
        regional.append(
            tuple(
                observe_regional_support_euler(
                    step.before.snapshot,
                    step.after_epi,
                    region,
                    dt=H,
                    epi_weight=step.before.epi_weight,
                    forcing=step.before.forcing,
                )
                for region in REGIONS
            )
        )
    captures.append(capture_non_epi_forcing(graph))
    return SimpleNamespace(
        model=model,
        captures=tuple(captures),
        budgets=tuple(budgets),
        regional=tuple(regional),
        perturbed=request.param,
        phase_factor=a**STEPS,
        form_factor=ratio_bound,
    )


def test_six_node_restoration_matches_frozen_absolute_targets(executed):
    model = executed.model
    degree, metric = (
        model.represented_odd_balance.strengths,
        model.represented_odd_balance.metric_weights,
    )
    initial_phase = _norm(_phase_error(executed.captures[0], model, 0), degree)
    initial_form = _norm(
        tuple(
            x - z for x, z in zip(executed.captures[0].snapshot.epi, model.target_epi)
        ),
        metric,
    )
    final_phase = _norm(_phase_error(executed.captures[-1], model, STEPS), degree)
    final_form = _norm(
        tuple(
            x - z for x, z in zip(executed.captures[-1].snapshot.epi, model.target_epi)
        ),
        metric,
    )
    assert final_phase <= float(executed.phase_factor) * initial_phase + HEADROOM
    assert final_form <= float(executed.form_factor) * initial_form + HEADROOM
    if executed.perturbed:
        assert initial_phase > 0.03 and initial_form > 0.4
        assert final_phase < initial_phase / 10
        assert final_form < initial_form / 5
    else:
        assert final_phase < HEADROOM and final_form < HEADROOM
    assert max(model.target_epi) - min(model.target_epi) > F(1, 5)


def test_refresh_gates_wraps_and_signed_nodal_regional_budgets(executed):
    model = executed.model
    means = []
    metric = model.represented_odd_balance.metric_weights
    for index, capture in enumerate(executed.captures):
        assert capture.stored_pressure_residual == (0,) * 6
        assert all(0 <= phase < math.tau for phase in capture.phase)
        assert max(map(abs, capture.snapshot.epi)) < 1
        for i, neighbors in enumerate(capture.snapshot.support_neighbors):
            assert all(
                abs(angle_diff(float(capture.phase[j]), float(capture.phase[i])))
                < math.pi / 2
                for j in neighbors
            )
        error = _phase_error(capture, model, index)
        assert max(error) - min(error) <= 1 / 32 + HEADROOM
        means.append(
            sum(h * x for h, x in zip(metric, capture.snapshot.epi)) / sum(metric)
        )
    assert max(abs(float(value - means[0])) for value in means) < HEADROOM
    # Several canonical wraps occur; error uses the frozen rotating target.
    assert any(
        executed.captures[n + 1].phase[0] < executed.captures[n].phase[0]
        for n in range(STEPS)
    )
    for global_budget, pair in zip(executed.budgets, executed.regional, strict=True):
        assert global_budget.identity_residual == 0
        assert max(map(abs, global_budget.state_defect)) < F(1, 10**14)
        for budget in pair:
            assert (
                budget.mass_identity_residual == budget.variance_identity_residual == 0
            )
            assert len(budget.balance.cut_edges) == 1


def test_target_matches_independent_high_precision_phasor_and_closed_poisson_profile():
    _, model = _model()
    with mp.workdps(80):
        alpha, beta = mp.asin(mp.mpf(1) / 8), mp.asin(mp.mpf(7) / 16)
        gamma = mp.atan2(mp.mpf(3) / 16, 2 * mp.cos(alpha) + mp.cos(beta))
        u = alpha / mp.pi
        bridge = u + 3 * gamma / (2 * mp.pi)
        outer = bridge + u
        profile = [outer, outer, bridge, -bridge, -outer, -outer]
        weights = [
            mp.mpf(x.numerator) / x.denominator
            for x in model.represented_odd_balance.metric_weights
        ]
        shift = sum(h * x for h, x in zip(weights, profile)) / sum(weights)
        target = tuple(float(x - shift) for x in profile)
        assert tuple(map(float, model.target_epi)) == pytest.approx(target, abs=3e-16)
        assert model.target_angles_estimate == pytest.approx(
            (float(alpha), float(beta)), abs=2e-16
        )
    balance = model.represented_odd_balance
    assert balance.compatibility_residual == balance.profile_center_residual == 0
    assert balance.profile_residual == (0,) * 6
    assert model.bridge_load == F(7, 16)
    assert model.common_phase_rate == 1
    assert max(map(abs, model.production_forcing_discrepancy)) < F(1, 10**14)
    assert max(map(abs, model.production_target_capture.full_kernel_pressure)) < F(
        1, 10**14
    )
    # Exact reflected model compatibility is not silently imposed on production.
    actual = model.production_target_capture
    assert model.production_balance.compatibility_residual == sum(
        d * f for d, f in zip(balance.strengths, actual.forcing)
    )


def test_actual_target_phase_proposal_has_common_rate_before_any_recovery():
    graph, model = _model()
    for node, phase in zip(graph, model.target_phase_estimate):
        set_theta(graph, node, phase)
    capture = capture_non_epi_forcing(graph)
    after = propose_u3_gated_phase_step(
        graph,
        tuple(graph),
        capture.phase,
        capture.snapshot.capacity,
        dt=float(H),
        coupling_strength=float(K),
    )
    assert after == pytest.approx(
        np.asarray(capture.phase, float) + float(H), abs=1e-15
    )


@pytest.mark.parametrize(
    "strength,status", ((8, "strict"), (7, "at_capacity"), (6, "overloaded"))
)
def test_exact_cut_load_classification_and_target_availability(strength, status):
    graph = unit_barbell(capacities=(1, 3))
    model = derive_regional_phase_lock(graph, REGIONS, coupling_strength=strength)
    assert model.bridge_load == F(7, strength)
    assert model.lock_status == status
    assert (model.target_epi is not None) == (status == "strict")
    assert (model.target_phase_estimate is not None) == (status == "strict")
    if status == "overloaded":
        # Cut demand exceeds the maximal K*sin(beta) for every full-edge lock.
        assert 7 * abs(model.epsilon) > model.coupling_strength


def test_tighter_gate_is_separate_from_ideal_strict_load():
    graph, _ = _model()
    graph.graph["UM_MAX_PHASE_DIFF"] = 0.25
    model = derive_regional_phase_lock(graph, REGIONS, coupling_strength=K)
    assert model.lock_status == "strict" and not model.target_region_gate_admitted


def test_signed_detuning_and_equal_capacity_controls():
    _, positive = _model()
    _, negative = _model(-EPSILON)
    _, equal = _model(F(0))
    assert negative.target_epi == tuple(reversed(positive.target_epi))
    assert equal.bridge_load == 0 and equal.lock_status == "strict"
    assert equal.target_epi == (0,) * 6


def test_generic_phase_perturbation_can_drift_the_full_weighted_mean():
    graph, model = _model()
    for i in graph:
        graph.nodes[i]["EPI"] = float(model.target_epi[i])
        set_theta(graph, i, model.target_phase_estimate[i] + (0.05 if i == 2 else 0))
    default_compute_delta_nfr(graph)
    step = execute_joint_step(graph, dt=H, coupling_strength=K, time=0)
    degree = model.represented_odd_balance.strengths
    metric = model.represented_odd_balance.metric_weights
    demand = sum(d * f for d, f in zip(degree, step.before.forcing))
    assert abs(demand) > F(1, 10**6)
    before = sum(h * x for h, x in zip(metric, step.before.snapshot.epi)) / sum(metric)
    after = sum(h * x for h, x in zip(metric, step.after_epi.epi)) / sum(metric)
    assert abs(float(after - before)) > 1e-8
    assert float(after - before) == pytest.approx(
        float(H * demand / sum(metric)), abs=1e-16
    )


def test_turning_off_phase_pressure_does_not_preserve_the_old_form_target():
    graph, model = _model()
    for i in graph:
        graph.nodes[i]["EPI"] = float(model.target_epi[i])
        set_theta(graph, i, model.target_phase_estimate[i])
    graph.graph["_dnfr_weights"]["phase"] = 0.0  # hold e unchanged
    default_compute_delta_nfr(graph)
    step = execute_joint_step(graph, dt=H, coupling_strength=K, time=0)
    assert max(
        abs(x - y) for x, y in zip(step.after_epi.epi, step.before.snapshot.epi)
    ) > F(1, 10000)


def test_observation_does_not_mutate_state_or_shared_metadata():
    graph, _ = _model()
    nodes = deepcopy(dict(graph.nodes(data=True)))
    edges = deepcopy(tuple(graph.edges(data=True)))
    attributes = dict(graph.graph)
    weights = deepcopy(graph.graph["_dnfr_weights"])
    derive_regional_phase_lock(graph, REGIONS, coupling_strength=K)
    assert dict(graph.nodes(data=True)) == nodes
    assert tuple(graph.edges(data=True)) == edges
    assert graph.graph.keys() == attributes.keys()
    assert all(graph.graph[key] is value for key, value in attributes.items())
    assert graph.graph["_dnfr_weights"] == weights


def test_target_shape_is_independent_of_stored_pressure_and_initial_form():
    graph, baseline = _model()
    # This nonuniform change has zero H mean; it must not fit the target shape.
    for i in graph:
        graph.nodes[i]["EPI"] = float(F(1, 4) + FORM_PERTURBATION[i])
        graph.nodes[i]["delta_nfr"] = float(1000 * (i + 1))
    observed = derive_regional_phase_lock(graph, REGIONS, coupling_strength=K)
    assert observed.initial_weighted_mean == F(1, 4)
    assert (
        observed.represented_odd_balance.relative_profile
        == baseline.represented_odd_balance.relative_profile
    )
    assert observed.target_epi == tuple(x + F(1, 4) for x in baseline.target_epi)
    assert (
        observed.analytic_phase_gradient_estimate
        == baseline.analytic_phase_gradient_estimate
    )
    assert any(observed.initial_capture.stored_pressure_residual)


def test_zero_phase_pressure_retains_lock_but_not_differentiated_target():
    graph, _ = _model()
    graph.graph["_dnfr_weights"]["phase"] = 0.0
    observed = derive_regional_phase_lock(graph, REGIONS, coupling_strength=K)
    assert observed.lock_status == "strict"
    assert observed.target_angles_estimate[1] > 0
    assert observed.target_epi == (0,) * 6


def test_target_respects_arbitrary_labels_insertion_order_and_region_orientation():
    original, expected = _model()
    mapping = {i: f"n{5-i}" for i in original}
    renamed = nx.relabel_nodes(original, mapping)
    graph = nx.Graph()
    graph.graph.update(renamed.graph)
    graph.add_nodes_from(reversed(tuple(renamed.nodes(data=True))))
    graph.add_edges_from(renamed.edges(data=True))
    regions = tuple(tuple(mapping[i] for i in row) for row in REGIONS)
    actual = derive_regional_phase_lock(graph, regions, coupling_strength=K)
    by_node = dict(zip(actual.initial_capture.snapshot.nodes, actual.target_epi))
    assert tuple(by_node[mapping[i]] for i in original) == expected.target_epi


@pytest.mark.parametrize("value", (0, -1, True, math.inf, math.nan))
def test_rejects_invalid_coupling(value):
    with pytest.raises((TypeError, ValueError)):
        derive_regional_phase_lock(unit_barbell(), REGIONS, coupling_strength=value)


@pytest.mark.parametrize(
    "case", ("capacity", "epi", "vf", "topo", "weight", "support", "region")
)
def test_rejects_unsupported_model_inputs(case):
    graph, _ = _model()
    regions = REGIONS
    if case == "capacity":
        graph.nodes[0]["nu_f"] = 0.75
    elif case in ("epi", "vf", "topo"):
        graph.graph["_dnfr_weights"][case] = 0 if case == "epi" else 0.125
    elif case == "weight":
        graph.edges[2, 3]["weight"] = 0.5
    elif case == "support":
        graph.add_edge(0, 5, weight=0)
    else:
        regions = ((0, 1, 3), (2, 4, 5))
    with pytest.raises((TypeError, ValueError)):
        derive_regional_phase_lock(graph, regions, coupling_strength=K)
