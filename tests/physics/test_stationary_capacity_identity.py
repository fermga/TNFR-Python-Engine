"""Analytic stationary form/phase targets and independently changed capacity.

All targets and controls are declared before a detached production-kernel
capture. The phase-compensation controls execute one native selector batch
and coordinator invocation each; there is no integration or complete step.
No solver, pressure fit, forced operator or longer trajectory is used.
The analytic target has zero pressure over exact reals; its materialized graph
retains phase/kernel/form-rounding residuals rather than claiming exact runtime
stationarity. Stored zero pressure is supplied initialization, not evidence.
"""

import math
from copy import deepcopy
from fractions import Fraction
from functools import wraps

import networkx as nx
import pytest

from tests.physics.test_native_step_formation import _snapshot
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.dynamics import coordination, runtime, selectors
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.glyph_history import ensure_history
from tnfr.metrics.sense_index import get_Si_weights
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
    observe_forcing_capacity_difference,
)

F = Fraction


def _analytic_target(*, disconnected=False, zero_vf_weight=False):
    graph = nx.Graph()
    if disconnected:
        graph.add_nodes_from(range(5))
        graph.add_weighted_edges_from(((0, 1, 1), (2, 3, 2)))
        capacity = (F(1, 2), F(1), F(1, 2), F(1), F(3, 4))
        phase_units = (F(0), F(1, 8), F(0), F(1, 8), F(0))
    else:
        graph.add_weighted_edges_from(((0, 1, 1), (1, 2, 2)))
        capacity = (F(1, 2), F(1), F(1, 2))
        phase_units = (F(0), F(1, 8), F(0))
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=17, compute_delta_nfr=default_compute_delta_nfr, _t=0.0
    )
    if zero_vf_weight:
        graph.graph["DNFR_WEIGHTS"] = {
            **graph.graph["DNFR_WEIGHTS"],
            "vf": 0.0,
        }
    weights = {key: F(value) for key, value in graph.graph["DNFR_WEIGHTS"].items()}

    # At this particular target the path endpoints have equal phase/capacity;
    # weighted EPI and unweighted capacity neighbor rows therefore coincide.
    # This construction is not valid for arbitrary unequal endpoint capacities.
    target = tuple(
        F(3, 4)
        - weights["phase"] / weights["epi"] * phase
        - weights["vf"] / weights["epi"] * nu
        for phase, nu in zip(phase_units, capacity, strict=True)
    )
    for node, epi, phase, nu in zip(graph, target, phase_units, capacity, strict=True):
        graph.nodes[node].update(
            EPI=float(epi),
            theta=float(phase) * math.pi,
            nu_f=float(nu),
            delta_nfr=0.0,
            glyph_history=[],
        )
    return graph, target


def _capacity_copy(graph, capacity):
    result = deepcopy(graph)
    for node, value in zip(result, capacity, strict=True):
        result.nodes[node]["nu_f"] = float(value)
    return result


@pytest.fixture(scope="module")
def stationary_captures():
    graph, target = _analytic_target()
    contracted = _capacity_copy(graph, (F(7, 12), F(5, 6), F(7, 12)))
    shifted = _capacity_copy(graph, (F(3, 4), F(5, 4), F(3, 4)))
    asymmetric = _capacity_copy(graph, (F(3, 4), F(1), F(1, 2)))
    disconnected, disconnected_target = _analytic_target(disconnected=True)
    component_shifted = _capacity_copy(
        disconnected, (F(3, 4), F(5, 4), F(1), F(3, 2), F(3, 2))
    )
    uncoupled, uncoupled_target = _analytic_target(zero_vf_weight=True)
    uncoupled_contracted = _capacity_copy(uncoupled, (F(7, 12), F(5, 6), F(7, 12)))
    graphs = {
        "baseline": graph,
        "contracted": contracted,
        "shifted": shifted,
        "asymmetric": asymmetric,
        "disconnected": disconnected,
        "component_shifted": component_shifted,
        "uncoupled": uncoupled,
        "uncoupled_contracted": uncoupled_contracted,
    }
    return {
        "graphs": graphs,
        "captures": {
            name: capture_non_epi_forcing(value) for name, value in graphs.items()
        },
        "targets": (target, disconnected_target, uncoupled_target),
        "configuration": deepcopy(graph.graph["DNFR_WEIGHTS"]),
    }


def test_regular_nonzero_phase_target_is_analytic_before_kernel_observation(
    stationary_captures,
):
    case = stationary_captures
    baseline = case["captures"]["baseline"]
    assert case["configuration"] == DEFAULTS["DNFR_WEIGHTS"]
    # The prospective target used configured ratios, so verify them against
    # the effective captured coefficients before claiming its ideal balance.
    effective = dict(baseline.normalized_weights)
    for channel in ("phase", "vf"):
        assert effective[channel] / effective["epi"] == (
            F(case["configuration"][channel]) / F(case["configuration"]["epi"])
        )
    assert baseline.snapshot.epi == tuple(F(float(x)) for x in case["targets"][0])
    assert len(set(baseline.snapshot.epi)) == 2
    assert baseline.phase_gradient == pytest.approx(
        (0.125, -0.125, 0.125), rel=0, abs=2e-16
    )
    # The ideal target satisfies degree-weighted compatibility. Materialized
    # singleton phasors and certified midpoint rows round separately; retain
    # their mean defect instead of claiming exact runtime conservation.
    ideal_phase = (F(1, 8), -F(1, 8), F(1, 8))
    assert sum(d * p for d, p in zip((1, 3, 2), ideal_phase, strict=True)) == 0
    realized_balance = sum(
        d * p for d, p in zip((1, 3, 2), baseline.phase_gradient, strict=True)
    )
    rounding_budget = sum(
        d * abs(actual - ideal)
        for d, actual, ideal in zip(
            (1, 3, 2), baseline.phase_gradient, ideal_phase, strict=True
        )
    )
    assert abs(realized_balance) <= rounding_budget < F(1, 10**15)
    assert max(map(abs, baseline.full_kernel_pressure)) < F(1, 10**15)
    assert baseline.snapshot.stored_pressure == (0, 0, 0)
    assert baseline.full_kernel_pressure != baseline.snapshot.stored_pressure


def test_capacity_contraction_breaks_the_same_stationary_form_phase_target(
    stationary_captures,
):
    captures = stationary_captures["captures"]
    result = observe_forcing_capacity_difference(
        captures["baseline"], captures["contracted"]
    )
    b = dict(captures["baseline"].normalized_weights)["vf"]
    assert result.epi_offset == result.phase_offset == 0
    assert result.capacity_pressure_change == (-b / 4, b / 4, -b / 4)
    assert result.phase_realization_change == (0, 0, 0)
    assert result.component_capacity_offsets == (None,)
    assert result.identity_residual == result.stored_identity_residual == (0, 0, 0)
    assert result.fresh_pressure_change == tuple(
        predicted + defect
        for predicted, defect in zip(
            result.capacity_pressure_change, result.kernel_defect_change, strict=True
        )
    )
    assert tuple(float(p) for p in captures["contracted"].full_kernel_pressure) == (
        pytest.approx(-float(b) / 4, rel=0, abs=1e-15),
        pytest.approx(float(b) / 4, rel=0, abs=1e-15),
        pytest.approx(-float(b) / 4, rel=0, abs=1e-15),
    )
    # Supplied stored zeros do not cancel the fresh residual or certify a lock.
    assert result.stored_pressure_change == (0, 0, 0)
    assert result.stored_residual_change == tuple(
        -p for p in result.fresh_pressure_change
    )
    # A represented analytic target still has a small nonzero initial residual:
    # Delta(nu*p) = nu_after*Delta(p) + Delta(nu)*p_before, with BOTH terms.
    p_before = captures["baseline"].full_kernel_pressure
    p_after = captures["contracted"].full_kernel_pressure
    rate_difference = tuple(
        nu_after * after - nu_before * before
        for nu_after, after, nu_before, before in zip(
            result.after.capacity,
            p_after,
            result.before.capacity,
            p_before,
            strict=True,
        )
    )
    baseline_correction = tuple(
        delta * pressure
        for delta, pressure in zip(result.capacity_change, p_before, strict=True)
    )
    assert any(baseline_correction)
    assert rate_difference == tuple(
        nu * delta_p + correction
        for nu, delta_p, correction in zip(
            result.after.capacity,
            result.fresh_pressure_change,
            baseline_correction,
            strict=True,
        )
    )


def test_capacity_pressure_uses_unique_neighbors_not_transport_conductance(
    stationary_captures,
):
    captures = stationary_captures["captures"]
    result = observe_forcing_capacity_difference(
        captures["baseline"], captures["asymmetric"]
    )
    b = dict(captures["baseline"].normalized_weights)["vf"]
    assert result.capacity_pressure_change == (-b / 4, b / 8, 0)
    assert result.capacity_pressure_change[1] != b / 12
    assert result.fresh_pressure_change == pytest.approx(
        tuple(float(p) for p in (-b / 4, b / 8, 0)), rel=0, abs=2e-16
    )
    assert result.identity_residual == (0, 0, 0)


def test_a_common_additive_capacity_shift_is_invisible_to_stationary_pressure(
    stationary_captures,
):
    captures = stationary_captures["captures"]
    result = observe_forcing_capacity_difference(
        captures["baseline"], captures["shifted"]
    )
    assert result.capacity_change == (F(1, 4),) * 3
    assert result.support_components == ((0, 1, 2),)
    assert result.component_capacity_offsets == (F(1, 4),)
    assert result.capacity_pressure_change == result.fresh_pressure_change == (0, 0, 0)
    assert result.identity_residual == (0, 0, 0)
    assert result.before.capacity != result.after.capacity


def test_disconnected_support_retains_independent_component_capacity_offsets(
    stationary_captures,
):
    captures = stationary_captures["captures"]
    result = observe_forcing_capacity_difference(
        captures["disconnected"], captures["component_shifted"]
    )
    assert result.support_components == ((0, 1), (2, 3), (4,))
    assert result.component_capacity_offsets == (F(1, 4), F(1, 2), F(3, 4))
    assert len(set(result.capacity_change)) == 3
    assert result.capacity_pressure_change == result.fresh_pressure_change == (0,) * 5
    assert result.identity_residual == (0,) * 5


def test_zero_capacity_channel_weight_removes_this_stationary_obstruction(
    stationary_captures,
):
    captures = stationary_captures["captures"]
    result = observe_forcing_capacity_difference(
        captures["uncoupled"], captures["uncoupled_contracted"]
    )
    assert dict(captures["uncoupled"].normalized_weights)["vf"] == 0
    assert result.component_capacity_offsets == (None,)
    assert len(set(result.capacity_change)) > 1
    assert result.capacity_pressure_change == result.fresh_pressure_change == (0, 0, 0)
    assert result.identity_residual == (0, 0, 0)


@pytest.fixture(scope="module")
def native_phase_compensation(stationary_captures):
    source = stationary_captures["graphs"]["contracted"]
    weights = {
        key: F(value) for key, value in stationary_captures["configuration"].items()
    }
    # p_end = a*delta/pi + e*(x_center-x_end) + b*(nu_center-nu_end).
    # Halving the original capacity gap d=.5 therefore requires MORE phase
    # separation: delta_required = delta + pi*(b/a)*epsilon*d, epsilon=.5.
    required_phase_unit = F(1, 8) + weights["vf"] / weights["phase"] * F(1, 4)
    required_gap = float(required_phase_unit) * math.pi
    prepared = {
        "old_phase": deepcopy(source),
        "static_compensation": deepcopy(source),
    }
    prepared["static_compensation"].nodes[1]["theta"] = required_gap
    cases = {}
    for name, graph in prepared.items():
        initial = _snapshot(graph)
        configuration = deepcopy(
            {
                key: graph.graph[key]
                for key in (
                    "SI_WEIGHTS",
                    "SELECTOR_THRESHOLDS",
                    "GRAMMAR_CANON",
                    "GLYPH_FACTORS",
                    "AL_MAX_LAG",
                    "EN_MAX_LAG",
                    "PHASE_K_GLOBAL",
                    "PHASE_K_LOCAL",
                    "PHASE_ADAPT",
                    "EPI_MIN",
                    "EPI_MAX",
                    "CLIP_MODE",
                    "GAMMA",
                )
            }
        )
        runtime._prepare_dnfr(graph, use_Si=True)
        fresh = _snapshot(graph)
        fresh_capture = capture_non_epi_forcing(graph)
        events, choices, glyphs = [], [], []
        original_select = selectors.DefaultGlyphSelector.select
        original_glyph = selectors.apply_glyph

        @wraps(original_select)
        def select(owner, G, node):
            choice = original_select(owner, G, node)
            events.append(("select", node))
            choices.append(
                (
                    node,
                    getattr(choice, "value", choice),
                    validate_candidate(G, node, choice),
                    _snapshot(G),
                )
            )
            return choice

        @wraps(original_glyph)
        def glyph(G, node, choice, **kwargs):
            before = _snapshot(G)
            events.append(("glyph", node))
            result = original_glyph(G, node, choice, **kwargs)
            glyphs.append(
                (node, getattr(choice, "value", choice), before, _snapshot(G))
            )
            return result

        history = ensure_history(graph)
        lag_before = deepcopy(
            {key: history.get(key, {}) for key in ("since_AL", "since_EN")}
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(selectors.DefaultGlyphSelector, "select", select)
            patch.setattr(selectors, "apply_glyph", glyph)
            selectors._apply_glyphs(graph, selectors._apply_selector(graph), history)
        before_coordination = capture_non_epi_forcing(graph)
        result = coordination.coordinate_global_local_phase(graph)
        after_coordination = capture_non_epi_forcing(graph)
        cases[name] = {
            "initial": initial,
            "configuration": configuration,
            "fresh": fresh,
            "si_weights": get_Si_weights(graph),
            "fresh_capture": fresh_capture,
            "events": tuple(events),
            "choices": tuple(choices),
            "glyphs": tuple(glyphs),
            "lag_before": lag_before,
            "history": deepcopy(history),
            "before": before_coordination,
            "after": after_coordination,
            "endpoint": _snapshot(graph),
            "coordinator_result": result,
        }
    return {
        "required_phase_unit": required_phase_unit,
        "required_gap": required_gap,
        "cases": cases,
    }


def test_static_phase_compensation_is_feasible_but_not_a_native_construction(
    native_phase_compensation, stationary_captures
):
    result = native_phase_compensation
    old = result["cases"]["old_phase"]["fresh_capture"]
    compensated = result["cases"]["static_compensation"]["fresh_capture"]
    original = stationary_captures["captures"]["baseline"]
    assert compensated.snapshot.epi == old.snapshot.epi == original.snapshot.epi
    assert compensated.snapshot.capacity == old.snapshot.capacity
    assert compensated.snapshot.conductance == old.snapshot.conductance
    assert all(value > 0 for value in compensated.snapshot.capacity)
    assert compensated.phase[0] == compensated.phase[2] == old.phase[0] == old.phase[2]
    assert F(1, 8) < result["required_phase_unit"] < F(1, 2)
    assert result["required_gap"] > math.pi / 8
    assert result["required_gap"] < math.pi / 2
    weights = dict(original.normalized_weights)
    # Cheap source-domain check for the entire DECLARED epsilon<=1 recipe;
    # this does not execute another perturbation or assert its runtime fate.
    maximum_required_unit = F(1, 8) + weights["vf"] / weights["phase"] / 2
    assert result["required_phase_unit"] <= maximum_required_unit < F(1, 2)
    assert max(map(abs, compensated.full_kernel_pressure)) < F(1, 10**15)
    assert max(map(abs, old.full_kernel_pressure)) > F(1, 100)
    assert compensated.phase_gradient == pytest.approx(
        tuple(float(sign * result["required_phase_unit"]) for sign in (1, -1, 1)),
        rel=0,
        abs=2e-16,
    )
    # Both states are freshly initialized; no history produced the larger gap.
    for case in result["cases"].values():
        assert case["initial"]["glyphs"] == case["initial"]["temporal"] == ((),) * 3
        config = case["configuration"]
        assert config["GAMMA"] == {"type": "none", "beta": 0.0, "R0": 0.0}
        assert config["CLIP_MODE"] == "hard"
        assert all(
            config["EPI_MIN"] < value < config["EPI_MAX"]
            for value in case["initial"]["epi"]
        )


def test_initialized_fresh_selector_chooses_il_before_the_coordinator(
    native_phase_compensation,
):
    for case in native_phase_compensation["cases"].values():
        for key, value in case["configuration"].items():
            assert value == DEFAULTS[key]
        assert case["lag_before"] == {"since_AL": {}, "since_EN": {}}
        assert (
            case["history"]["since_AL"]
            == case["history"]["since_EN"]
            == {
                0: 1,
                1: 1,
                2: 1,
            }
        )
        assert 1 <= min(
            case["configuration"]["AL_MAX_LAG"],
            case["configuration"]["EN_MAX_LAG"],
        )
        assert case["events"] == tuple(("select", n) for n in range(3)) + tuple(
            ("glyph", n) for n in range(3)
        )
        assert all(si > 0.5 for si in case["fresh"]["si"])
        alpha, beta, _gamma = map(F, case["si_weights"])
        # On the original gap<=pi/8, even the uncontracted capacity ratio1/2
        # suffices for high-Si selection, without using the pressure contribution.
        assert alpha / 2 + F(7, 8) * beta > F(1, 2)
        assert tuple(choice for _, choice, _, _ in case["choices"]) == ("IL",) * 3
        for _, _, admission, state in case["choices"]:
            assert admission.allowed
            assert state["pressure"] == case["fresh"]["pressure"]
            assert state["si"] == case["fresh"]["si"]
        retention = case["configuration"]["GLYPH_FACTORS"]["IL_dnfr_factor"]
        for node, glyph, before, after in case["glyphs"]:
            assert glyph == "IL"
            assert after["pressure"][node] == before["pressure"][node] * retention
            for channel in ("epi", "phase", "capacity", "edges"):
                assert after[channel] == before[channel]
        assert case["endpoint"]["time"] == 0
        assert case["endpoint"]["temporal"] == ((),) * 3


def test_native_coordinator_contracts_both_gaps_instead_of_restoring_balance(
    native_phase_compensation,
):
    required = native_phase_compensation["required_gap"]
    for case in native_phase_compensation["cases"].values():
        before, after = case["before"], case["after"]
        config, history = case["configuration"], case["history"]
        assert case["coordinator_result"] is None  # Actual default legacy path.
        assert tuple(history["phase_state"]) == ("stable",)
        assert tuple(history["phase_disr"]) == (0.0,)
        k_global, k_local = history["phase_kG"][0], history["phase_kL"][0]
        assert k_global == pytest.approx(
            config["PHASE_K_GLOBAL"]
            + config["PHASE_ADAPT"]["down"]
            * (config["PHASE_ADAPT"]["kG_min"] - config["PHASE_K_GLOBAL"]),
            rel=0,
            abs=1e-17,
        )
        assert k_local == pytest.approx(
            config["PHASE_K_LOCAL"]
            + config["PHASE_ADAPT"]["down"]
            * (config["PHASE_ADAPT"]["kL_min"] - config["PHASE_K_LOCAL"]),
            rel=0,
            abs=1e-17,
        )
        gap_before = before.phase[1] - before.phase[0]
        gap_after = after.phase[1] - after.phase[0]
        assert before.phase[0] == before.phase[2]
        assert after.phase[0] == after.phase[2]
        # The common global target cancels from this signed gap difference.
        assert float(gap_after) == pytest.approx(
            (1 - k_global - 2 * k_local) * float(gap_before), rel=0, abs=2e-16
        )
        assert 0 < gap_after < gap_before
        assert float(gap_after) < required
        a = dict(after.normalized_weights)["phase"]
        predicted_endpoint_pressure = float(a) * (
            float(gap_after) / math.pi
            - float(native_phase_compensation["required_phase_unit"])
        )
        assert after.full_kernel_pressure == pytest.approx(
            tuple(sign * predicted_endpoint_pressure for sign in (1, -1, 1)),
            rel=0,
            abs=2e-16,
        )
        assert before.snapshot.epi == after.snapshot.epi
        assert before.snapshot.capacity == after.snapshot.capacity
        assert before.snapshot.conductance == after.snapshot.conductance
        assert before.snapshot.stored_pressure == after.snapshot.stored_pressure
        before_channels = dict(decompose_non_epi_forcing(before))
        after_channels = dict(decompose_non_epi_forcing(after))
        assert before_channels["vf"] == after_channels["vf"]
        assert before_channels["topo"] == after_channels["topo"]
        phase_change = tuple(
            y - x for x, y in zip(before_channels["phase"], after_channels["phase"])
        )
        assert phase_change[0] < 0 < phase_change[1]
        assert phase_change[2] == phase_change[0]
        # No refreshed source is retroactively written into the held pressure.
        assert after.full_kernel_pressure[0] < before.full_kernel_pressure[0]
        assert max(map(abs, after.full_kernel_pressure)) > F(1, 100)
        for p0, p1, delta_phase, k0, k1 in zip(
            before.full_kernel_pressure,
            after.full_kernel_pressure,
            phase_change,
            before.kernel_pressure_defect,
            after.kernel_pressure_defect,
            strict=True,
        ):
            assert p1 - p0 == delta_phase + k1 - k0
