"""Finite default-runtime formation control with transparent owner tracing.

Two independently initialized C5 states differ only in supplied adaptation
counters. Those counters do not certify past stability. The test executes the
native selector, pressure/Si refresh, integrator, adaptive phase coordinator and
REMESH gate without replacing their decisions or enabling optional mathematics.
Endpoint observations are not a repeated-runtime preservation theorem.
"""

import math
from copy import deepcopy
from functools import wraps
from importlib import import_module

import networkx as nx
import pytest

import tnfr.dynamics as dynamics
from tnfr.alias import get_attr
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics import adaptation, coordination, integrators, runtime, selectors
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_chart import observe_common_phase_chart


def _prepared(*, ready):
    graph = nx.cycle_graph(5)
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=17, compute_delta_nfr=default_compute_delta_nfr, _t=0.0
    )
    count = graph.graph["VF_ADAPT_TAU"] - 1 if ready else 0
    for node, phase, epi in zip(
        graph,
        (0.10, 0.13, 0.11, 0.17, 0.14),
        (0.125, 0.128, 0.121, 0.127, 0.123),
        strict=True,
    ):
        graph.nodes[node].update(
            theta=phase,
            EPI=epi,
            nu_f=0.3,
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=[],
            stable_count=count,
        )
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def _snapshot(graph):
    channels = {
        "epi": ALIAS_EPI,
        "phase": ALIAS_THETA,
        "capacity": ALIAS_VF,
        "pressure": ALIAS_DNFR,
        "rate": ALIAS_DEPI,
        "si": ALIAS_SI,
    }
    result = {
        name: tuple(get_attr(graph.nodes[node], aliases, 0.0) for node in graph)
        for name, aliases in channels.items()
    }
    result.update(
        nodes=tuple(graph),
        edges=tuple((i, j, deepcopy(data)) for i, j, data in graph.edges(data=True)),
        time=graph.graph.get("_t", 0.0),
        counters=tuple(graph.nodes[node].get("stable_count", 0) for node in graph),
        glyphs=tuple(tuple(graph.nodes[node]["glyph_history"]) for node in graph),
        temporal=tuple(
            tuple(graph.nodes[node].get("epi_time_history", ())) for node in graph
        ),
        remesh_history_length=len(graph.graph.get("_epi_hist", ())),
        stable_fraction_length=len(
            graph.graph.get("history", {}).get("stable_frac", ())
        ),
    )
    return result


def _install_spy(patch, owner, attribute, label, events, *, graph_index=0):
    original = getattr(owner, attribute)

    @wraps(original)
    def observed(*args, **kwargs):
        graph = args[graph_index]
        record = {"owner": label, "before": _snapshot(graph)}
        if label == "capacity_write":
            record["node"], record["value"] = args[1:3]
        events.append(record)
        result = original(*args, **kwargs)
        record["after"] = _snapshot(graph)
        record["result"] = result
        return result

    patch.setattr(owner, attribute, observed)


@pytest.fixture(scope="module")
def native_steps():
    cases = {}
    remesh = import_module("tnfr.operators.remesh")
    for name, ready in (("ordinary", False), ("counter_ready", True)):
        graph = _prepared(ready=ready)
        initial = _snapshot(graph)
        initial_chart = observe_common_phase_chart(graph)
        initial_configuration = deepcopy(
            {
                key: graph.graph.get(key)
                for key in (
                    "DNFR_WEIGHTS",
                    "SI_WEIGHTS",
                    "GLYPH_FACTORS",
                    "SELECTOR_THRESHOLDS",
                    "PHASE_ADAPT",
                    "PHASE_K_GLOBAL",
                    "PHASE_K_LOCAL",
                    "VF_ADAPT_TAU",
                    "VF_ADAPT_MU",
                    "REMESH_STABILITY_WINDOW",
                    "MATH_ENGINE",
                    "GAMMA",
                    "INTEGRATOR_METHOD",
                    "DT",
                )
            }
        )
        events = []
        with pytest.MonkeyPatch.context() as patch:
            owners = (
                (runtime, "_record_mutation_flow_boundary", "flow_history"),
                (runtime, "_refresh_delta_nfr", "pressure"),
                (dynamics, "compute_Si", "Si"),
                (selectors, "_apply_glyphs", "glyphs"),
                (coordination, "coordinate_global_local_phase", "phase_coordinator"),
                (adaptation, "adapt_vf_after_structural_stability", "capacity_gate"),
                (adaptation, "set_vf", "capacity_write"),
                (runtime, "append_remesh_epi_history_snapshot", "remesh_history"),
                (runtime, "apply_remesh_if_globally_stable", "remesh_gate"),
                (remesh, "apply_network_remesh", "network_remesh"),
            )
            for owner, attribute, label in owners:
                _install_spy(patch, owner, attribute, label, events)
            _install_spy(
                patch,
                integrators.DefaultIntegrator,
                "integrate",
                "nodal_integration",
                events,
                graph_index=1,
            )
            result = runtime.step(graph)
        cases[name] = {
            "graph": graph,
            "initial": initial,
            "initial_chart": initial_chart,
            "configuration": initial_configuration,
            "events": events,
            "result": result,
            "final": _snapshot(graph),
            "final_chart": observe_common_phase_chart(graph),
            "endpoint_forcing": capture_non_epi_forcing(graph),
        }
    return cases


def _event(case, name):
    matches = [event for event in case["events"] if event["owner"] == name]
    assert len(matches) == 1
    return matches[0]


def test_actual_default_step_order_and_enabled_owner_paths(native_steps):
    expected = [
        "flow_history",
        "pressure",
        "Si",
        "glyphs",
        "flow_history",
        "nodal_integration",
        "phase_coordinator",
        "capacity_gate",
        "flow_history",
        "remesh_history",
        "remesh_gate",
    ]
    for case in native_steps.values():
        assert case["result"] is None
        assert [
            event["owner"]
            for event in case["events"]
            if event["owner"] != "capacity_write"
        ] == expected
        cfg = case["configuration"]
        for key in (
            "DNFR_WEIGHTS",
            "SI_WEIGHTS",
            "GLYPH_FACTORS",
            "SELECTOR_THRESHOLDS",
            "PHASE_ADAPT",
            "PHASE_K_GLOBAL",
            "PHASE_K_LOCAL",
            "VF_ADAPT_TAU",
            "VF_ADAPT_MU",
            "REMESH_STABILITY_WINDOW",
            "INTEGRATOR_METHOD",
            "DT",
        ):
            assert cfg[key] == DEFAULTS[key]
        assert cfg["MATH_ENGINE"] is None and cfg["GAMMA"]["type"] == "none"
        assert case["graph"].graph["compute_delta_nfr"] is default_compute_delta_nfr
        assert case["graph"].graph["RANDOM_SEED"] == 17
        assert "glyph_selector" not in case["graph"].graph
        assert "integrator" not in case["graph"].graph


def test_pressure_is_refreshed_before_fresh_si_and_native_il_selection(native_steps):
    case = native_steps["ordinary"]
    pressure, si, glyphs = (_event(case, name) for name in ("pressure", "Si", "glyphs"))
    assert pressure["before"]["pressure"] == (0.0,) * 5
    assert any(pressure["after"]["pressure"])
    assert pressure["result"] is default_compute_delta_nfr
    before = pressure["before"]
    raw_weights = case["configuration"]["DNFR_WEIGHTS"]
    weight_sum = sum(raw_weights.values())
    phase_weight, epi_weight = (
        raw_weights[channel] / weight_sum for channel in ("phase", "epi")
    )
    expected = []
    for node in range(5):
        left, right = (node - 1) % 5, (node + 1) % 5
        # Both neighbor displacements are in one acute chart, hence the ideal
        # phasor direction is their midpoint. Runtime rounding stays explicit.
        phase_pressure = (
            (before["phase"][left] + before["phase"][right]) / 2 - before["phase"][node]
        ) / math.pi
        epi_pressure = (before["epi"][left] + before["epi"][right]) / 2 - before["epi"][
            node
        ]
        expected.append(phase_weight * phase_pressure + epi_weight * epi_pressure)
    assert pressure["after"]["pressure"] == pytest.approx(expected, abs=2e-16, rel=0)
    assert si["before"]["pressure"] == pressure["after"]["pressure"]
    assert si["after"]["si"] == glyphs["before"]["si"]
    alpha = case["configuration"]["SI_WEIGHTS"]["alpha"]
    assert all(value >= alpha for value in si["after"]["si"])
    assert glyphs["after"]["glyphs"] == (("IL",),) * 5
    retention = case["configuration"]["GLYPH_FACTORS"]["IL_dnfr_factor"]
    assert glyphs["after"]["pressure"] == tuple(
        value * retention for value in glyphs["before"]["pressure"]
    )
    for channel in ("epi", "phase", "capacity", "edges"):
        assert glyphs["after"][channel] == glyphs["before"][channel]


def test_actual_nodal_flow_and_history_use_post_glyph_held_pressure(native_steps):
    for case in native_steps.values():
        integration = _event(case, "nodal_integration")
        left, right = integration["before"], integration["after"]
        dt = case["configuration"]["DT"]
        assert right["time"] - left["time"] == dt
        assert right["epi"] != left["epi"]
        rates = tuple(
            vf * pressure for vf, pressure in zip(left["capacity"], left["pressure"])
        )
        assert right["rate"] == rates
        expected = tuple(x + dt * rate for x, rate in zip(left["epi"], rates))
        assert right["epi"] == pytest.approx(expected, abs=2e-16, rel=0)
        assert right["pressure"] == left["pressure"]
        assert right["phase"] == left["phase"]
        final = case["final"]
        for node, history in enumerate(final["temporal"]):
            assert history == (
                (0.0, case["initial"]["epi"][node]),
                (dt, final["epi"][node]),
            )
        assert final["remesh_history_length"] == 1
        assert case["graph"].graph["_epi_hist"][-1] == dict(enumerate(final["epi"]))


def test_native_coordinator_preserves_observed_chart_without_free_phase_advance(
    native_steps,
):
    for case in native_steps.values():
        phase = _event(case, "phase_coordinator")
        integration = _event(case, "nodal_integration")
        initial, final = case["initial_chart"], case["final_chart"]
        assert phase["before"]["phase"] == integration["after"]["phase"]
        assert phase["after"]["phase"] != phase["before"]["phase"]
        assert min(phase["after"]["phase"]) > min(phase["before"]["phase"])
        assert max(phase["after"]["phase"]) < max(phase["before"]["phase"])
        assert initial.admitted and final.admitted
        assert initial.cycle_periods == final.cycle_periods == (0,)
        assert final.zero_cycle_winding is True
        assert final.diameter_enclosure[1] < initial.diameter_enclosure[0]
        history = case["graph"].graph["history"]
        assert tuple(history["phase_state"]) == ("stable",)
        assert len(history["phase_kG"]) == len(history["phase_kL"]) == 1
        assert history["phase_kG"][0] != case["configuration"]["PHASE_K_GLOBAL"]
        assert history["phase_kL"][0] != case["configuration"]["PHASE_K_LOCAL"]
        assert case["final"]["edges"] == case["initial"]["edges"]
        assert case["final"]["capacity"] == case["initial"]["capacity"] == (0.3,) * 5


def test_adaptation_consumes_stored_inputs_and_uniform_capacity_is_fixed(native_steps):
    ordinary, ready = native_steps["ordinary"], native_steps["counter_ready"]
    tau = ready["configuration"]["VF_ADAPT_TAU"]
    assert ordinary["initial"]["counters"] == (0,) * 5
    assert ready["initial"]["counters"] == (tau - 1,) * 5
    for name, case in native_steps.items():
        adaptation_event = _event(case, "capacity_gate")
        left, right = adaptation_event["before"], adaptation_event["after"]
        si_event, glyphs = (_event(case, event) for event in ("Si", "glyphs"))
        assert left["si"] == si_event["after"]["si"]
        assert left["pressure"] == glyphs["after"]["pressure"]
        eps = case["graph"].graph["EPS_DNFR_STABLE"]
        si_hi = case["configuration"]["SELECTOR_THRESHOLDS"]["si_hi"]
        stable = tuple(
            abs(p) <= eps and si >= si_hi for p, si in zip(left["pressure"], left["si"])
        )
        assert stable == (False, False, False, False, True)
        expected_count = 1 if name == "ordinary" else tau
        assert right["counters"] == (0, 0, 0, 0, expected_count)
        writes = [
            event for event in case["events"] if event["owner"] == "capacity_write"
        ]
        assert len(writes) == int(name == "counter_ready")
        if writes:
            assert writes[0]["node"] == 4 and writes[0]["value"] == 0.3
        assert right["capacity"] == left["capacity"] == (0.3,) * 5
        # Runtime does not refresh pressure/Si after phase coordination. These
        # retained inputs are observable, not silently recertified as current.
        endpoint = case["endpoint_forcing"]
        assert any(endpoint.stored_pressure_residual)
        assert tuple(map(float, endpoint.snapshot.stored_pressure)) == right["pressure"]
    for channel in ("epi", "phase", "capacity", "pressure", "si", "edges"):
        assert ordinary["final"][channel] == ready["final"][channel]


def test_remesh_gate_retains_its_actual_missing_history_abstention(native_steps):
    for case in native_steps.values():
        remesh = _event(case, "remesh_gate")
        assert remesh["before"]["stable_fraction_length"] == 0
        assert case["configuration"]["REMESH_STABILITY_WINDOW"] > 0
        assert remesh["after"] == remesh["before"]
        assert all(event["owner"] != "network_remesh" for event in case["events"])
        assert "_REMESH_META" not in case["graph"].graph
        assert "_last_remesh_step" not in case["graph"].graph
