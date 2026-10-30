"""One frozen default-runtime Mutation prefix and a capacity-only control.

The prepared K5 plus pendant has no common semicircle initially. All histories
start empty, and three native steps supply both grammar and temporal evidence.
Heterogeneous capacities are initial data, not an emergent capacity law. The
observed Mutation reduces the particular phase diameter; this witness does not
demonstrate formation of contrast or escape from a common phase chart.
"""

from copy import deepcopy
from fractions import Fraction
from functools import wraps

import networkx as nx
import pytest

import tnfr.dynamics as dynamics
from tests.physics.test_native_step_formation import _snapshot
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.dynamics import coordination, integrators, runtime, selectors
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.mathematics._phase_midpoint import _affine_interval, _pi_bounds
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    observe_forcing_dirichlet_balance,
)
from tnfr.physics.mutation_trigger import certify_mutation_trigger
from tnfr.physics.phase_chart import observe_common_phase_chart
from tnfr.physics.support_transport import (
    observe_support_transport_clipped_flow,
    observe_support_transport_euler,
)


def _prepared(*, uniform):
    graph = nx.complete_graph(5)
    graph.add_edge(1, 9)
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=17, compute_delta_nfr=default_compute_delta_nfr, _t=0.0
    )
    nx.set_edge_attributes(graph, 1.0, "weight")
    for node in graph:
        phase, epi, capacity = (
            (0.0, 0.6, 0.8)
            if node == 0
            else ((4.7, 1.0, 6.25) if node == 9 else (2.8, -1.0, 0.1))
        )
        graph.nodes[node].update(
            theta=phase,
            EPI=epi,
            nu_f=0.8 if uniform else capacity,
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=[],
        )
    return graph


def _trigger(graph, state):
    return certify_mutation_trigger(
        current_epi=state["epi"][0],
        nu_f=state["capacity"][0],
        delta_nfr=state["pressure"][0],
        epi_time_history=graph.nodes[0].get("epi_time_history"),
    )


@pytest.fixture(scope="module")
def native_prefixes():
    cases = {}
    configuration_keys = (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "SELECTOR_THRESHOLDS",
        "GLYPH_FACTORS",
        "GRAMMAR_CANON",
        "PHASE_ADAPT",
        "DT",
        "INTEGRATOR_METHOD",
        "CLIP_MODE",
        "EPI_MIN",
        "EPI_MAX",
        "CLIP_SOFT_K",
        "VF_MAX",
        "VF_ADAPT_TAU",
        "VF_ADAPT_MU",
        "AL_MAX_LAG",
        "EN_MAX_LAG",
        "GAMMA",
    )
    for name, uniform in (("heterogeneous", False), ("uniform", True)):
        graph = _prepared(uniform=uniform)
        initial = _snapshot(graph)
        chart = observe_common_phase_chart(graph)
        configuration = deepcopy({key: graph.graph[key] for key in configuration_keys})
        refreshes, senses, batches, glyph_events, flows, endpoints = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        original_pressure = runtime._refresh_delta_nfr
        original_si = dynamics.compute_Si
        original_batch = selectors._apply_glyphs
        original_select = selectors.DefaultGlyphSelector.select
        original_glyph = selectors.apply_glyph
        original_integrate = integrators.DefaultIntegrator.integrate
        original_coordinate = coordination.coordinate_global_local_phase
        selections = []
        forcing_flows, coordination_events = [], []

        @wraps(original_pressure)
        def refresh(G, **kwargs):
            before = _snapshot(G)
            result = original_pressure(G, **kwargs)
            refreshes.append((before, _snapshot(G), result))
            return result

        @wraps(original_si)
        def compute_si(G, **kwargs):
            before = _snapshot(G)
            result = original_si(G, **kwargs)
            senses.append((before, _snapshot(G)))
            return result

        @wraps(original_batch)
        def apply_batch(G, selector, history):
            before = _snapshot(G)
            record = {
                "before": before,
                "trigger": _trigger(G, before),
                "selector": selector,
                "forcing_before": capture_non_epi_forcing(G),
            }
            selection_count = len(selections)
            result = original_batch(G, selector, history)
            assert len(selections) == selection_count + 1
            record.update(selections[-1])
            record["after"] = _snapshot(G)
            record["forcing_after"] = capture_non_epi_forcing(G)
            batches.append(record)
            return result

        @wraps(original_select)
        def select(owner, G, node):
            result = original_select(owner, G, node)
            if node == 0:
                selections.append(
                    {
                        "base": getattr(result, "value", result),
                        "grammar": validate_candidate(G, node, result),
                    }
                )
            return result

        @wraps(original_glyph)
        def apply_glyph(G, node, glyph, **kwargs):
            before = _snapshot(G)
            forcing_before = capture_non_epi_forcing(G) if node == 0 else None
            result = original_glyph(G, node, glyph, **kwargs)
            if node == 0:
                glyph_events.append(
                    {
                        "glyph": getattr(glyph, "value", glyph),
                        "before": before,
                        "after": _snapshot(G),
                        "forcing_before": forcing_before,
                        "forcing_after": capture_non_epi_forcing(G),
                    }
                )
            return result

        @wraps(original_integrate)
        def integrate(owner, G, **kwargs):
            before = _snapshot(G)
            forcing_before = capture_non_epi_forcing(G)
            result = original_integrate(owner, G, **kwargs)
            flows.append((before, _snapshot(G)))
            forcing_flows.append((forcing_before, capture_non_epi_forcing(G)))
            return result

        @wraps(original_coordinate)
        def coordinate(G, *args, **kwargs):
            record = {
                "before": _snapshot(G),
                "forcing_before": capture_non_epi_forcing(G),
            }
            result = original_coordinate(G, *args, **kwargs)
            record["after"] = _snapshot(G)
            record["forcing_after"] = capture_non_epi_forcing(G)
            coordination_events.append(record)
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runtime, "_refresh_delta_nfr", refresh)
            patch.setattr(dynamics, "compute_Si", compute_si)
            patch.setattr(selectors, "_apply_glyphs", apply_batch)
            patch.setattr(selectors.DefaultGlyphSelector, "select", select)
            patch.setattr(selectors, "apply_glyph", apply_glyph)
            patch.setattr(integrators.DefaultIntegrator, "integrate", integrate)
            patch.setattr(coordination, "coordinate_global_local_phase", coordinate)
            for _ in range(3):
                runtime.step(graph)
                endpoints.append(_snapshot(graph))
        cases[name] = {
            "graph": graph,
            "initial": initial,
            "initial_chart": chart,
            "configuration": configuration,
            "refreshes": refreshes,
            "senses": senses,
            "batches": batches,
            "glyphs": glyph_events,
            "flows": flows,
            "forcing_flows": forcing_flows,
            "coordination": coordination_events,
            "endpoints": endpoints,
        }
    return cases


def test_third_step_forcing_captures_preserve_actual_owner_boundaries(native_prefixes):
    for name, case in native_prefixes.items():
        assert len(case["forcing_flows"]) == len(case["coordination"]) == 3
        glyph, batch, coordinator = (
            case["glyphs"][2],
            case["batches"][2],
            case["coordination"][2],
        )
        before, after = glyph["forcing_before"], glyph["forcing_after"]
        for field in ("nodes", "conductance", "capacity", "epi"):
            assert getattr(before.snapshot, field) == getattr(after.snapshot, field)
        assert before.epi_weight == after.epi_weight
        assert before.normalized_weights == after.normalized_weights
        assert before == batch["forcing_before"]
        if name == "heterogeneous":
            assert glyph["glyph"] == "ZHIR"
            assert before.snapshot.stored_pressure == after.snapshot.stored_pressure
            assert after.phase[0] - before.phase[0] == Fraction(1, 4)
            assert after.phase[1:] == before.phase[1:]
            assert before.phase_gradient != after.phase_gradient
            assert before.forcing != after.forcing
            assert before.full_kernel_pressure != after.full_kernel_pressure
        else:
            assert glyph["glyph"] == "IL"
            assert before.phase == after.phase
            assert before.forcing == after.forcing
            assert before.full_kernel_pressure == after.full_kernel_pressure
            assert before.snapshot.stored_pressure != after.snapshot.stored_pressure

        flow_before, flow_after = case["forcing_flows"][2]
        assert flow_before == batch["forcing_after"]
        assert flow_before.phase == flow_after.phase
        assert flow_before.forcing == flow_after.forcing
        assert flow_before.snapshot.capacity == flow_after.snapshot.capacity
        assert flow_before.snapshot.conductance == flow_after.snapshot.conductance
        assert (
            flow_before.snapshot.stored_pressure == flow_after.snapshot.stored_pressure
        )
        assert flow_before.snapshot.epi != flow_after.snapshot.epi
        assert coordinator["forcing_before"] == flow_after
        coord_before, coord_after = (
            coordinator["forcing_before"],
            coordinator["forcing_after"],
        )
        for field in ("nodes", "conductance", "capacity", "epi", "stored_pressure"):
            assert getattr(coord_before.snapshot, field) == getattr(
                coord_after.snapshot, field
            )
        assert coord_before.phase != coord_after.phase
        assert coord_before.forcing != coord_after.forcing
        assert (
            coord_after.snapshot.stored_pressure == flow_before.snapshot.stored_pressure
        )
        assert any(coord_after.stored_pressure_residual)


def test_actual_zhir_reverses_fresh_energy_rate_without_replacing_stored_pressure(
    native_prefixes,
):
    event = native_prefixes["heterogeneous"]["glyphs"][2]
    before, after = (
        observe_forcing_dirichlet_balance(event[key])
        for key in ("forcing_before", "forcing_after")
    )
    assert before.identity_residual == after.identity_residual == 0
    assert before.source.dirichlet_energy == after.source.dirichlet_energy
    assert before.stored_residual_rate == 0
    assert before.fresh_rate == before.stored_rate > 0
    assert after.fresh_rate < 0 < after.stored_rate == before.stored_rate
    assert after.stored_residual_rate > 0
    assert after.diffusion_rate == before.diffusion_rate < 0
    old_channels, new_channels = map(dict, (before.channel_rates, after.channel_rates))
    assert new_channels["phase"] < old_channels["phase"]
    assert new_channels["vf"] == old_channels["vf"]
    assert new_channels["topo"] == old_channels["topo"] == 0
    assert after.source_rate - before.source_rate == (
        new_channels["phase"] - old_channels["phase"]
    )
    assert after.modeled_rate < 0 < before.modeled_rate


@pytest.mark.parametrize("name", ["heterogeneous", "uniform"])
def test_native_coordinator_changes_fresh_work_at_the_actual_postflow_form(
    native_prefixes, name
):
    case = native_prefixes[name]
    event = case["coordination"][2]
    flow_before, flow_after = case["forcing_flows"][2]
    assert event["forcing_before"] == flow_after
    before, after = (
        observe_forcing_dirichlet_balance(event[key])
        for key in ("forcing_before", "forcing_after")
    )
    assert before.source.epi == after.source.epi == flow_after.snapshot.epi
    assert before.source.epi != flow_before.snapshot.epi
    assert before.identity_residual == after.identity_residual == 0
    assert before.source.dirichlet_energy == after.source.dirichlet_energy
    assert before.diffusion_rate == after.diffusion_rate < 0
    old_channels, new_channels = map(dict, (before.channel_rates, after.channel_rates))
    assert new_channels["vf"] == old_channels["vf"]
    assert new_channels["phase"] < old_channels["phase"]
    assert after.source_rate < before.source_rate
    assert after.fresh_rate < before.fresh_rate < 0
    assert after.stored_rate == before.stored_rate
    if name == "heterogeneous":
        assert after.stored_rate > 0


@pytest.mark.parametrize("name", ["heterogeneous", "uniform"])
def test_third_actual_held_flow_energy_budget_retains_pressure_and_endpoint_defects(
    native_prefixes, name
):
    case = native_prefixes[name]
    before, after = case["forcing_flows"][2]
    assert before == case["batches"][2]["forcing_after"]
    assert after == case["coordination"][2]["forcing_before"]
    left, right = case["flows"][2]
    elapsed = Fraction.from_float(right["time"]) - Fraction.from_float(left["time"])
    assert elapsed == Fraction(1, 2)
    budget = observe_support_transport_euler(before.snapshot, after.snapshot, elapsed)
    start = observe_forcing_dirichlet_balance(before)
    assert budget.identity_residual == start.identity_residual == 0
    assert budget.drift_term == elapsed * (
        start.diffusion_rate
        + start.source_rate
        + start.kernel_defect_rate
        + start.stored_residual_rate
    )
    assert budget.energy_change == (
        budget.drift_term + budget.quadratic_term + budget.defect_term
    )
    assert budget.energy_change == (
        after.snapshot.dirichlet_energy - before.snapshot.dirichlet_energy
    )
    assert budget.quadratic_term > 0
    assert any(budget.state_defect)
    assert start.fresh_rate < 0
    # The observed endpoint defect combines all differences from the held
    # exact-real Euler reference, including runtime rounding and clipping.
    # No raw per-substep capture isolates those contributions in this study.
    if name == "heterogeneous":
        assert start.stored_rate > 0 and budget.energy_change > 0
    else:
        assert start.stored_rate < 0 and budget.energy_change < 0


@pytest.mark.parametrize("name", ["heterogeneous", "uniform"])
def test_same_held_flow_separates_common_rail_clipping_from_remaining_endpoint_defect(
    native_prefixes, name
):
    case = native_prefixes[name]
    before, after = case["forcing_flows"][2]
    cfg = case["configuration"]
    assert cfg["CLIP_MODE"] == "hard"
    assert (cfg["EPI_MIN"], cfg["EPI_MAX"]) == (-1.0, 1.0)
    left, right = case["flows"][2]
    elapsed = Fraction.from_float(right["time"]) - Fraction.from_float(left["time"])
    budget = observe_support_transport_clipped_flow(
        before.snapshot,
        after.snapshot,
        elapsed,
        lower=cfg["EPI_MIN"],
        upper=cfg["EPI_MAX"],
    )
    assert before == case["batches"][2]["forcing_after"]
    assert after == case["coordination"][2]["forcing_before"]
    assert budget.held.identity_residual == budget.identity_residual == 0
    assert budget.unclipped_energy_change == (
        budget.held.drift_term + budget.held.quadratic_term
    )
    assert budget.clipped_energy_change == (
        budget.unclipped_energy_change + budget.clipping_term
    )
    assert budget.held.energy_change == (
        budget.clipped_energy_change + budget.implementation_term
    )
    assert budget.held.defect_term == budget.clipping_term + budget.implementation_term
    assert budget.clipping_term <= 0
    assert budget.implementation_state_defect == tuple(
        actual - clipped
        for actual, clipped in zip(after.snapshot.epi, budget.clipped_epi)
    )
    # This bound reports the retained finite endpoint only. It does not split
    # the remaining defect into individual binary64 operations or prove a
    # solver accuracy bound for future states.
    assert 0 < max(map(abs, budget.implementation_state_defect)) < Fraction(1, 10**15)
    clipped_nodes = tuple(
        node
        for node, unbounded, clipped in zip(
            before.snapshot.nodes, budget.held.expected_epi, budget.clipped_epi
        )
        if unbounded != clipped
    )
    if name == "heterogeneous":
        assert clipped_nodes == (9,)
        assert budget.held.expected_epi[-1] < budget.lower
        assert budget.clipped_epi[-1] == after.snapshot.epi[-1] == budget.lower
        assert budget.implementation_state_defect[-1] == 0
        assert budget.clipping_term < 0 < budget.clipped_energy_change
        assert budget.implementation_term > 0
    else:
        assert clipped_nodes == () and budget.clipping_term == 0
        assert budget.clipped_epi == budget.held.expected_epi
        assert budget.clipped_energy_change < 0 and budget.implementation_term < 0


def test_default_prefix_uses_supplied_geometry_and_no_fabricated_history(
    native_prefixes,
):
    positive, negative = native_prefixes["heterogeneous"], native_prefixes["uniform"]
    assert positive["initial"]["nodes"] == (0, 1, 2, 3, 4, 9)
    assert positive["initial"]["capacity"] == (0.8, 0.1, 0.1, 0.1, 0.1, 6.25)
    assert negative["initial"]["capacity"] == (0.8,) * 6
    for channel in ("epi", "phase", "pressure", "edges", "time", "glyphs", "temporal"):
        assert positive["initial"][channel] == negative["initial"][channel]
    for case in native_prefixes.values():
        assert case["initial_chart"].status == "excluded"
        assert case["initial"]["glyphs"] == case["initial"]["temporal"] == ((),) * 6
        assert case["initial"]["counters"] == (0,) * 6
        assert case["initial"]["si"] == (0.0,) * 6
        assert case["initial"]["pressure"] == (0.0,) * 6
        for key, value in case["configuration"].items():
            assert value == DEFAULTS[key]
        assert case["graph"].graph["RANDOM_SEED"] == 17
        assert case["graph"].graph["compute_delta_nfr"] is default_compute_delta_nfr
        assert "glyph_selector" not in case["graph"].graph
        assert case["graph"].graph.get("MATH_ENGINE") is None
        assert all(
            batch["selector"] is selectors.default_glyph_selector
            for batch in case["batches"]
        )
        assert [endpoint["time"] for endpoint in case["endpoints"]] == [0.5, 1.0, 1.5]
        assert all(
            endpoint["capacity"] == case["initial"]["capacity"]
            and endpoint["edges"] == case["initial"]["edges"]
            for endpoint in case["endpoints"]
        )


def test_native_fresh_pressure_and_si_select_real_il_oz_zhir_prefix(native_prefixes):
    case = native_prefixes["heterogeneous"]
    assert [batch["base"] for batch in case["batches"]] == ["OZ", "OZ", "ZHIR"]
    assert [event["glyph"] for event in case["glyphs"]] == ["IL", "OZ", "ZHIR"]
    first, second, third = case["batches"]
    assert not first["grammar"].allowed
    assert "U4a" in {violation.rule for violation in first["grammar"].violations}
    assert second["grammar"].allowed and third["grammar"].allowed
    assert third["before"]["glyphs"][0] == ("IL", "OZ")
    assert third["after"]["glyphs"][0] == ("IL", "OZ", "ZHIR")
    thresholds = case["configuration"]["SELECTOR_THRESHOLDS"]
    for index, (refresh, sense, batch) in enumerate(
        zip(case["refreshes"], case["senses"], case["batches"], strict=True)
    ):
        before, fresh, callback = refresh
        assert callback is default_compute_delta_nfr
        assert fresh["pressure"] == sense[0]["pressure"] == batch["before"]["pressure"]
        assert sense[1]["si"] == batch["before"]["si"]
        assert before["epi"] == fresh["epi"] and before["phase"] == fresh["phase"]
        p = fresh["pressure"][0]
        normalized = abs(p) / max(map(abs, fresh["pressure"]))
        assert p > 0 and sense[1]["si"][0] <= thresholds["si_lo"]
        assert (normalized > thresholds["dnfr_hi"]) == (index < 2)
    # The target's first pressure has a singleton-valued neighbor field. This
    # independent channel balance does not reconstruct pressure from its flow.
    weights = case["configuration"]["DNFR_WEIGHTS"]
    total = sum(weights.values())
    pi_low, pi_high = _pi_bounds()
    phase = Fraction.from_float(2.8)
    exact_epi_gap = Fraction(-1) - Fraction.from_float(0.6)
    exact_vf_gap = Fraction.from_float(0.1) - Fraction.from_float(0.8)
    predicted_bounds = tuple(
        float(
            Fraction.from_float(weights["phase"] / total) * phase / pi
            + Fraction.from_float(weights["epi"] / total) * exact_epi_gap
            + Fraction.from_float(weights["vf"] / total) * exact_vf_gap
        )
        for pi in (pi_high, pi_low)
    )
    actual = case["refreshes"][0][1]["pressure"][0]
    assert actual == pytest.approx(predicted_bounds[0], abs=3e-16, rel=0)
    assert predicted_bounds[0] <= predicted_bounds[1]


def test_signed_mutation_evidence_is_actual_previous_flow_not_current_product(
    native_prefixes,
):
    case = native_prefixes["heterogeneous"]
    first, second, third = case["batches"]
    assert not first["trigger"].evidence_valid
    assert first["trigger"].observed_depi_dt is None
    assert first["before"]["temporal"][0] == ((0.0, 0.6),)
    assert second["trigger"].evidence_valid
    trigger = third["trigger"]
    assert trigger.evidence_valid and trigger.physical_time_resolved
    assert trigger.source == "epi_time_history"
    assert trigger.observed_crossed and trigger.threshold_gate_satisfied
    assert not trigger.predicted_crossed
    assert trigger.predicted_depi_dt < trigger.xi < trigger.observed_depi_dt
    left, right = case["flows"][1]
    samples = third["before"]["temporal"][0]
    assert samples[-2:] == (
        (left["time"], left["epi"][0]),
        (right["time"], right["epi"][0]),
    )
    actual_secant = (right["epi"][0] - left["epi"][0]) / (right["time"] - left["time"])
    assert trigger.observed_depi_dt == actual_secant
    # The default duration contains eight rounded Euler additions. Their
    # endpoint secant need not be bitwise equal to the held nodal product.
    assert actual_secant == pytest.approx(
        left["capacity"][0] * left["pressure"][0], abs=2e-15, rel=0
    )
    assert right["epi"][0] < 1.0  # target flow is unclipped; pendant flow is not
    assert third["before"]["epi"][0] == right["epi"][0]
    abstentions = case["graph"].graph["history"].get("mutation_abstentions", ())
    assert all(item["node"] != 0 for item in abstentions)
    assert any(item["node"] in (2, 3, 4) for item in abstentions)
    assert case["graph"].graph["history"]["since_AL"][0] == 3
    assert case["graph"].graph["history"]["since_EN"][0] == 3


def _represented_chart_lifts(phases):
    """Exact true-circle lift for this retained centered runtime state only."""
    bounds = _pi_bounds()
    lifted = tuple(
        _affine_interval(Fraction.from_float(value), 2 if value < 0 else 0, bounds)
        for value in phases
    )
    return lifted


def test_executed_mutation_moves_toward_neighbors_and_contracts_this_chart(
    native_prefixes,
):
    case = native_prefixes["heterogeneous"]
    event = case["glyphs"][2]
    before, after = event["before"], event["after"]
    assert event["glyph"] == "ZHIR"
    assert after["phase"][0] == before["phase"][0] + 0.25
    assert after["phase"][1:] == before["phase"][1:]
    for channel in ("epi", "capacity", "pressure", "time", "edges", "temporal"):
        assert after[channel] == before[channel]
    assert all(
        before["phase"][0] < after["phase"][0] < before["phase"][index]
        for index in (1, 2, 3, 4)
    )
    # Native coordination stores some centered negative phases. Do not silently
    # normalize them and pretend that the canonical-input observer saw them.
    assert before["phase"][-1] < 0
    left, right = map(_represented_chart_lifts, (before["phase"], after["phase"]))
    pi_low, _ = _pi_bounds()
    assert all(left[0][1] < row[0] for row in left[1:])
    assert all(row[1] < left[-1][0] for row in left[:-1])
    diameter_before = (left[-1][0] - left[0][1], left[-1][1] - left[0][0])
    diameter_after = (right[-1][0] - right[0][1], right[-1][1] - right[0][0])
    assert 0 < diameter_after[1] < diameter_before[0]
    assert diameter_before[1] < pi_low
    assert (
        tuple(a - b for a, b in zip(diameter_before, diameter_after))
        == (Fraction(1, 4),) * 2
    )


def test_uniform_capacity_negative_keeps_default_il_despite_valid_positive_rates(
    native_prefixes,
):
    case = native_prefixes["uniform"]
    assert [batch["base"] for batch in case["batches"]] == ["IL"] * 3
    assert [event["glyph"] for event in case["glyphs"]] == ["IL"] * 3
    assert all(batch["grammar"].allowed for batch in case["batches"])
    alpha = case["configuration"]["SI_WEIGHTS"]["alpha"]
    assert all(batch["before"]["si"][0] >= alpha for batch in case["batches"])
    assert case["batches"][2]["trigger"].threshold_gate_satisfied
    assert case["batches"][2]["trigger"].observed_depi_dt > 0.1
    assert all(
        event["after"]["phase"] == event["before"]["phase"] for event in case["glyphs"]
    )
