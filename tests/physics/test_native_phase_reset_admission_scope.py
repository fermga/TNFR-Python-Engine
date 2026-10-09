"""One frozen native prism decision, with real preparation and flow history.

The IL/OZ prefix, Euler intervals and terminal pressure refresh are supplied
preparation/observation operations. They are not an autonomous occurrence law.
Transparent wrappers record the actual default selector and dispatcher; they
never return a replacement glyph or populate an EPI/glyph history. A negative
Mutation result ends this case without another preparation or native decision.
"""

import json
import os
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    PROJECTION,
    _apply,
    _graph,
    _inner,
)
from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics import runtime, selectors
from tnfr.dynamics.integrators import DefaultIntegrator
from tnfr.operators import (
    build_operator_event_schedule,
    execute_operator_event_schedule,
)
from tnfr.operators.definitions import Coherence, Dissonance
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.mutation_trigger import certify_mutation_trigger

_INTERVAL = 0.125
_WEIGHTS = {"epi": 0.5, "phase": 0.25, "vf": 0.25, "topo": 0.0}


def _code(glyph):
    return getattr(glyph, "value", glyph)


def _state(graph):
    state = {"time": graph.graph.get("_t", 0.0), "nodes": tuple(graph)}
    for name, aliases in (
        ("epi", ALIAS_EPI),
        ("capacity", ALIAS_VF),
        ("phase", ALIAS_THETA),
        ("pressure", ALIAS_DNFR),
        ("sense", ALIAS_SI),
    ):
        state[name] = tuple(
            float(get_attr(graph.nodes[n], aliases, 0.0)) for n in graph
        )
    state["glyph_history"] = tuple(
        tuple(map(_code, graph.nodes[n].get("glyph_history", ()))) for n in graph
    )
    state["physical_history"] = tuple(
        tuple(graph.nodes[n].get("epi_time_history", ())) for n in graph
    )
    state["edges"] = tuple(
        (a, b, deepcopy(data)) for a, b, data in graph.edges(data=True)
    )
    return state


def _trigger(graph, node):
    data = graph.nodes[node]
    return certify_mutation_trigger(
        current_epi=get_attr(data, ALIAS_EPI, 0.0),
        nu_f=get_attr(data, ALIAS_VF, 0.0),
        delta_nfr=get_attr(data, ALIAS_DNFR, 0.0),
        xi=graph.graph["ZHIR_THRESHOLD_XI"],
        epi_time_history=data.get("epi_time_history"),
        epi_history=data.get("epi_history"),
        legacy_epi_history=data.get("_epi_history"),
    )


def _run_frozen_case():
    """Execute the single declared case; the preparation never depends on results."""
    graph = _graph()
    inject_defaults(graph)
    graph.graph["DNFR_WEIGHTS"] = dict(_WEIGHTS)
    graph.graph["INTEGRATOR_METHOD"] = "euler"
    graph.graph["RANDOM_SEED"] = 0
    graph.graph["_t"] = 0.0
    frozen_parameters = deepcopy(graph.graph)
    initial = _state(graph)
    # Actual public applications supply a stable base and recent destabilizer.
    # This prefix is openly prescribed; it is not a native selector result.
    for node in NODES:
        Coherence()(graph, node)
        Dissonance()(graph, node)
    prefixed = _state(graph)
    schedule = build_operator_event_schedule(
        (), start_time=0.0, flow_durations=(_INTERVAL,)
    )
    # The partition API requires at least two segments. Retain the frozen one
    # Euler interval through the ordinary event executor, with real shared
    # pressure refreshes at its two boundaries instead of changing the mesh.
    runtime._prepare_dnfr(graph, use_Si=False)
    preparation_left = _state(graph)
    execution = execute_operator_event_schedule(
        graph, schedule, method="euler", include_flow_certificates=True
    )
    preparation_flow_end = _state(graph)
    runtime._prepare_dnfr(graph, use_Si=False)
    preparation = _state(graph)
    trace = {"base": [], "grammar": [], "applied": [], "refresh": [], "flow": []}
    original_resolve = selectors._resolve_preselected_glyph
    original_grammar = selectors.enforce_canonical_grammar
    original_apply = selectors.apply_glyph
    original_batch = selectors._apply_glyphs
    original_refresh = runtime._prepare_dnfr
    original_integrate = DefaultIntegrator.integrate
    original_coordinate = runtime.coordination.coordinate_global_local_phase

    def resolve(G, node, selector, preselection):
        chosen = original_resolve(G, node, selector, preselection)
        if preselection is not None:
            trace["base"].append((node, _code(chosen), preselection.metrics[node]))
        return chosen

    def grammar(G, node, candidate):
        result = original_grammar(G, node, candidate)
        trace["grammar"].append((node, _code(candidate), _code(result)))
        return result

    def apply(G, node, glyph, **kwargs):
        before = _state(G)
        result = original_apply(G, node, glyph, **kwargs)
        trace["applied"].append((node, _code(glyph), before, _state(G)))
        return result

    def batch(G, selector, history):
        trace["selector"] = selector
        trace["before_batch"] = _state(G)
        trace["history_before"] = deepcopy(dict(history))
        trace["triggers"] = tuple(_trigger(G, node) for node in NODES)
        # Counterfactual admission is recorded independently of actual selection.
        trace["mutation_grammar"] = tuple(
            validate_candidate(G, node, "ZHIR") for node in NODES
        )
        trace["source_before_batch"] = capture_non_epi_forcing(G)
        result = original_batch(G, selector, history)
        trace["after_batch"] = _state(G)
        trace["source_after_batch"] = capture_non_epi_forcing(G)
        trace["history_after"] = deepcopy(dict(history))
        return result

    def refresh(G, **kwargs):
        before = _state(G)
        result = original_refresh(G, **kwargs)
        trace["refresh"].append((before, _state(G), capture_non_epi_forcing(G)))
        return result

    def integrate(self, G, **kwargs):
        before = _state(G)
        result = original_integrate(self, G, **kwargs)
        trace["flow"].append((before, _state(G), dict(kwargs)))
        return result

    def coordinate(G, *args, **kwargs):
        before = _state(G)
        result = original_coordinate(G, *args, **kwargs)
        trace["coordination"] = (before, _state(G))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selectors, "_resolve_preselected_glyph", resolve)
        patch.setattr(selectors, "enforce_canonical_grammar", grammar)
        patch.setattr(selectors, "apply_glyph", apply)
        patch.setattr(selectors, "_apply_glyphs", batch)
        patch.setattr(runtime, "_prepare_dnfr", refresh)
        patch.setattr(DefaultIntegrator, "integrate", integrate)
        patch.setattr(runtime.coordination, "coordinate_global_local_phase", coordinate)
        runtime.step(graph, dt=_INTERVAL, use_Si=True, apply_glyphs=True)
        trace["after_public_step"] = _state(graph)
        # A declared real pressure refresh, with no second decision or flow.
        runtime._prepare_dnfr(graph, use_Si=True)
    return {
        "graph": graph,
        "frozen_parameters": frozen_parameters,
        "initial": initial,
        "prefixed": prefixed,
        "preparation": preparation,
        "preparation_left": preparation_left,
        "preparation_flow_end": preparation_flow_end,
        "preparation_evidence": execution.flow_interval_evidence[0],
        "trace": trace,
        "final": _state(graph),
    }


def _portable(value):
    """Project public evidence to JSON; this does not serialize private seals."""
    if isinstance(value, Q):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _portable(item) for key, item in value.items()}
        return {
            "entries": [
                [_portable(key), _portable(item)] for key, item in value.items()
            ]
        }
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _portable(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if value is selectors.default_glyph_selector:
        return "tnfr.dynamics.selectors.default_glyph_selector"
    raise TypeError(f"Unrepresented research record type: {type(value).__name__}")


def _write_record(case, path):
    observed = any(glyph == "ZHIR" for _, glyph, *_ in case["trace"]["applied"])
    record = {
        "freeze": {
            "date": "2026-09-19",
            "initial_internal_coefficients": ["1/4", "0", "1/4", "0"],
            "fiber_means": ["1/2", "1/2"],
            "capacity": 1,
            "phase": 0,
            "prefix": "Supplied public IL then OZ at each node in NODES order",
            "preparation_intervals": [_INTERVAL],
            "preparation_adapter": "One unpartitioned executor-owned Euler interval with explicit shared boundary refreshes",
            "adapter_correction": "The initial one-segment partition construction was rejected before flow or native selection; its duration and mesh were retained",
            "initial_clock": "Explicit _t=0.0 required by event executor; no history was inserted",
            "native_step": {"dt": _INTERVAL, "use_Si": True, "apply_glyphs": True},
            "terminal_refresh": "Explicit _prepare_dnfr(use_Si=True); no second step",
            "selection": "Default native selector; no parameter/preparation search",
        },
        "phase_reset_observed": observed,
        "phase_reset_work": None,
        "phase_reset_work_status": (
            "Not measured: no Mutation occurred"
            if not observed
            else "Requires event analysis"
        ),
        "scope": "Finite execution with supplied preparation; no autonomous maintenance claim",
        "records": {key: value for key, value in case.items() if key != "graph"},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_portable(record), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def case():
    # All controls share one invocation, not a family of preparations.
    result = _run_frozen_case()
    record_path = os.environ.get("TNFR_NATIVE_PHASE_RESET_RECORD")
    if record_path:
        _write_record(result, record_path)
    return result


def test_owned_preparation_separates_positive_growth_from_instantaneous_pressure(case):
    initial, prefixed, prepared = (
        case[name] for name in ("initial", "prefixed", "preparation")
    )
    assert initial["nodes"] == NODES
    assert initial["epi"] == (0.75, 0.25, 0.5) * 2
    assert initial["glyph_history"] == ((),) * 6
    assert prefixed["glyph_history"] == (("IL", "OZ"),) * 6
    for channel in ("epi", "capacity", "phase", "edges"):
        assert prefixed[channel] == initial[channel]
    assert prepared["time"] == _INTERVAL
    assert prepared["epi"] == (47 / 64, 17 / 64, 0.5) * 2
    for index, samples in enumerate(prepared["physical_history"]):
        assert samples == (
            (0.0, initial["epi"][index]),
            (_INTERVAL, prepared["epi"][index]),
        )
    evidence = case["preparation_evidence"]
    assert evidence.integrator_provenance_certified
    assert evidence.resolved_method == "euler" and evidence.clipping_applied is False
    assert evidence.certificate.exact_nodal_equation_realized
    for index, trigger in enumerate(case["trace"]["triggers"]):
        assert trigger.evidence_valid and trigger.physical_time_resolved
        assert trigger.current_endpoint_matches_state
        assert trigger.observed_depi_dt == (-1 / 8, 1 / 8, 0)[index % 3]
        assert trigger.threshold_gate_satisfied is (index % 3 == 1)
        if index % 3 == 1:
            assert trigger.predicted_depi_dt == 15 / 128
            assert trigger.rate_gap == 1 / 128
    assert all(item.allowed for item in case["trace"]["mutation_grammar"])


def test_default_native_decision_executes_coherence_even_with_admitted_growth(case):
    trace = case["trace"]
    assert trace["selector"] is selectors.default_glyph_selector
    assert tuple(node for node, _, _ in trace["base"]) == NODES
    assert tuple(glyph for _, glyph, _ in trace["base"]) == ("IL",) * 6
    alpha = case["graph"].graph["SI_WEIGHTS"]["alpha"]
    high = case["graph"].graph["SELECTOR_THRESHOLDS"]["si_hi"]
    assert alpha > high
    assert all(si >= alpha for _, _, (si, _, _) in trace["base"])
    assert trace["grammar"] == [(node, "IL", "IL") for node in NODES]
    assert tuple(row[1] for row in trace["applied"]) == ("IL",) * 6
    assert all(trace["history_after"]["since_AL"][n] == 1 for n in NODES)
    assert all(trace["history_after"]["since_EN"][n] == 1 for n in NODES)
    assert "mutation_abstentions" not in trace["history_after"]
    assert trace["after_batch"]["glyph_history"] == (("IL", "OZ", "IL"),) * 6
    assert case["final"]["glyph_history"] == trace["after_batch"]["glyph_history"]
    assert case["final"]["phase"] == (0.0,) * 6
    assert case["final"]["capacity"] == (1.0,) * 6
    assert case["final"]["edges"] == case["initial"]["edges"]


def test_native_flow_consumes_operator_pressure_before_the_declared_refresh(case):
    trace = case["trace"]
    assert len(trace["flow"]) == 1
    before, after, config = trace["flow"][0]
    assert config["dt"] == _INTERVAL and config["method"] == "euler"
    assert before["pressure"] == trace["after_batch"]["pressure"]
    assert before["time"] == _INTERVAL and after["time"] == 2 * _INTERVAL
    for x, nu, pressure, actual in zip(
        before["epi"], before["capacity"], before["pressure"], after["epi"], strict=True
    ):
        ideal = Q(x) + Q(_INTERVAL) * Q(nu) * Q(pressure)
        assert abs(Q(actual) - ideal) <= Q(1, 2**53)
    assert len(trace["refresh"]) == 2
    for old, refreshed, observation in trace["refresh"]:
        for channel in ("epi", "phase", "capacity", "time", "physical_history"):
            assert old[channel] == refreshed[channel]
        assert observation.stored_pressure_residual == (0,) * 6
    assert trace["refresh"][-1][0] == trace["after_public_step"]
    assert case["final"]["time"] == 2 * _INTERVAL


def test_observed_nonmutation_source_has_no_phase_work_and_negative_norm_rate(case):
    trace = case["trace"]
    observations = (
        trace["source_before_batch"],
        trace["source_after_batch"],
        trace["refresh"][-1][2],
    )
    for observation in observations:
        assert observation.phase_gradient == (0,) * 6
        assert observation.forcing == (0,) * 6
        coefficients = _apply(PROJECTION, observation.snapshot.epi)
        rate = _apply(PROJECTION, observation.snapshot.rate)
        norm_rate = 2 * (
            _inner(coefficients[:2], rate[:2]) + _inner(coefficients[2:], rate[2:])
        )
        assert norm_rate < 0
    assert trace["source_after_batch"].stored_pressure_residual != (0,) * 6
    assert not any(glyph == "ZHIR" for _, glyph, *_ in trace["applied"])
