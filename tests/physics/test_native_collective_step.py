"""One admitted native call retains a supplied doubled-C5 collective family.

The three form sources and two lag profiles reuse the preceding batch study.
Before response evaluation the clock is fixed to t=1, dt=1/16 (one default
Euler substep), with ordinary/ready capacity counters and scalar/NumPy paths.
Fraction arithmetic predicts the complete block endpoint independently of
runtime kernels. Absolute binary64 error is bounded by 1e-14 in these finite
controls; pair equality, events and memory require exact agreement. This does
not establish repeated-call invariance or autonomous formation of the family.
"""

from copy import deepcopy
from fractions import Fraction as Q
from functools import wraps
from importlib import import_module
from itertools import product

import networkx as nx
import pytest

import tnfr.dynamics as dynamics
from tnfr._runtime_steps import RUNTIME_STEP_NEXT_KEY
from tnfr.alias import get_attr
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics import adaptation, coordination, integrators, runtime, selectors
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.glyph_history import push_glyph
from tnfr.metrics import sense_index
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    read_grammar_state,
)
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.types import real_scalar_epi

PAIRS = tuple((2 * block, 2 * block + 1) for block in range(5))
FORMS = (
    (0.125, 0.375, 0.125, 0.125, 0.125),
    (-0.375, 0.25, -0.125, 0.5, 0.0),
    (0.96875, 0.375, 0.125, 0.25, 0.5),
)
PROFILES = (("IL",) * 5, ("AL", "EN", "IL", "EN", "AL"))
RATES = tuple((block - 2) / 8 for block in range(5))
DT = 1 / 16
ABS_ERROR = 1e-14
CHANNELS = {
    "form": ALIAS_EPI,
    "pressure": ALIAS_DNFR,
    "phase": ALIAS_THETA,
    "capacity": ALIAS_VF,
    "rate": ALIAS_DEPI,
    "acceleration": ALIAS_D2EPI,
    "si": ALIAS_SI,
}


def _source(forms, marks, *, ready=False):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for block in range(5):
        for node in PAIRS[block]:
            for neighbor in PAIRS[(block + 1) % 5]:
                graph.add_edge(node, neighbor, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=1.0,
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        compute_delta_nfr=default_compute_delta_nfr,
        history={"since_AL": {}, "since_EN": {}},
    )
    history = graph.graph["history"]
    count = graph.graph["VF_ADAPT_TAU"] - 1 if ready else 0
    for node in graph:
        block = node // 2
        data = graph.nodes[node]
        data.update(
            EPI=forms[block],
            epi_kind="seed",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            dEPI=RATES[block],
            d2EPI=0.0,
            glyph_history=[],
            stable_count=count,
        )
        # Supplied ordinal grammar context does not certify a physical past.
        for glyph in ("AL", "IL", "EN"):
            push_glyph(data, glyph, 1)
        data[U2_DEBT_KEY] = block % 2
        history["since_AL"][node] = 5 if marks[block] == "AL" else 0
        history["since_EN"][node] = 3 if marks[block] in {"AL", "EN"} else 0
    return graph


def _snapshot(graph):
    result = {
        name: tuple(
            get_attr(graph.nodes[node], aliases, None, conv=lambda value: value)
            for node in graph
        )
        for name, aliases in CHANNELS.items()
    }
    result["form"] = tuple(real_scalar_epi(value) for value in result["form"])
    result.update(
        counters=tuple(graph.nodes[node]["stable_count"] for node in graph),
        temporal=tuple(
            tuple(graph.nodes[node].get("epi_time_history", ())) for node in graph
        ),
        time=graph.graph["_t"],
        ordinal=graph.graph.get(RUNTIME_STEP_NEXT_KEY),
        edges=tuple((i, j, dict(data)) for i, j, data in graph.edges(data=True)),
    )
    return result


def _weight(graph, key, channel):
    weights = {name: Q(value) for name, value in graph.graph[key].items()}
    return weights[channel] / sum(weights.values())


def _oracle(graph, forms, marks):
    values = list(map(Q, forms))
    coefficient = _weight(graph, "DNFR_WEIGHTS", "epi")
    pressure = tuple(
        coefficient * ((values[(j - 1) % 5] + values[(j + 1) % 5]) / 2 - x)
        for j, x in enumerate(values)
    )
    maximum = max(map(abs, pressure))
    alpha, beta, gamma = (
        _weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha + beta + gamma * (1 - abs(p) / maximum if maximum else 1)
        for p in pressure
    )
    lower, upper = Q(graph.graph["EPI_MIN"]), Q(graph.graph["EPI_MAX"])

    def clip(value):
        return min(upper, max(lower, value))

    factors = graph.graph["GLYPH_FACTORS"]
    boost, mix, retention = (
        Q(factors[key]) for key in ("AL_boost", "EN_mix", "IL_dnfr_factor")
    )
    held = list(pressure)
    for j, mark in enumerate(marks):
        if mark == "AL":
            values[j] = clip(values[j] + boost)
        elif mark == "EN":
            mean = (values[(j - 1) % 5] + values[(j + 1) % 5]) / 2
            values[j] = (1 - mix) * values[j] + mix * mean
        else:
            assert mark == "IL"
            held[j] *= retention
    endpoint = tuple(clip(x + Q(DT) * p) for x, p in zip(values, held))
    acceleration = tuple((p - Q(rate)) / Q(DT) for p, rate in zip(held, RATES))
    epsilon = Q(graph.graph["EPS_DNFR_STABLE"])
    threshold = Q(graph.graph["SELECTOR_THRESHOLDS"]["si_hi"])
    stable = tuple(abs(p) <= epsilon and s >= threshold for p, s in zip(held, si))
    return {
        "initial_pressure": pressure,
        "si": si,
        "post_event": tuple(values),
        "pressure": tuple(held),
        "form": endpoint,
        "rate": tuple(held),
        "acceleration": acceleration,
        "stable": stable,
    }


def _spy(patch, owner, attribute, name, events, *, graph_index=0):
    actual = getattr(owner, attribute)

    @wraps(actual)
    def observed(*args, **kwargs):
        graph = args[graph_index]
        event = {"name": name, "before": _snapshot(graph)}
        if name == "capacity_write":
            event["node"], event["value"] = args[1:3]
        events.append(event)
        result = actual(*args, **kwargs)
        event["after"] = _snapshot(graph)
        event["result"] = result
        return result

    patch.setattr(owner, attribute, observed)


def _execute(graph, *, scalar=False):
    events, marks = [], []
    initial = _snapshot(graph)
    # Live runtime caches may own locks. Observe declared configuration while
    # leaving all graph-owned execution, cache and history state in place.
    configuration_keys = set(DEFAULTS) | {
        "compute_delta_nfr",
        "glyph_selector",
        "integrator",
        "MATH_ENGINE",
        "GRAMMAR_REJECTION_MODE",
    }
    config = deepcopy(
        {key: value for key, value in graph.graph.items() if key in configuration_keys}
    )
    assert config["GAMMA"]["type"] == "none"
    assert integrators._uses_builtin_gamma(config["GAMMA"])
    remesh = import_module("tnfr.operators.remesh")
    with pytest.MonkeyPatch.context() as patch:
        if scalar:
            graph.graph["vectorized_dnfr"] = False
            for owner in (sense_index, selectors, integrators, coordination):
                patch.setattr(owner, "np", None)
        for owner, attribute, name in (
            (runtime, "_record_mutation_flow_boundary", "flow_history"),
            (runtime, "_refresh_delta_nfr", "pressure"),
            (dynamics, "compute_Si", "si"),
            (selectors, "_apply_glyphs", "glyphs"),
            (coordination, "coordinate_global_local_phase", "phase"),
            (adaptation, "adapt_vf_after_structural_stability", "capacity"),
            (adaptation, "set_vf", "capacity_write"),
            (runtime, "append_remesh_epi_history_snapshot", "remesh_history"),
            (runtime, "apply_remesh_if_globally_stable", "remesh_gate"),
            (remesh, "apply_network_remesh", "network_remesh"),
        ):
            _spy(patch, owner, attribute, name, events)
        _spy(
            patch,
            integrators.DefaultIntegrator,
            "integrate",
            "integration",
            events,
            graph_index=1,
        )
        apply_glyph = selectors.apply_glyph

        def admitted_glyph(graph, node, glyph, **kwargs):
            assert validate_candidate(graph, node, glyph).allowed
            result = apply_glyph(graph, node, glyph, **kwargs)
            marks.append((node, glyph))
            return result

        patch.setattr(selectors, "apply_glyph", admitted_glyph)
        result = runtime.step(graph, dt=DT)
    assert result is None
    return {
        "graph": graph,
        "initial": initial,
        "final": _snapshot(graph),
        "configuration": config,
        "events": events,
        "marks": tuple(marks),
    }


def _event(case, name):
    selected = [event for event in case["events"] if event["name"] == name]
    assert len(selected) == 1
    return selected[0]


def _lift(values):
    return tuple(float(value) for value in values for _ in range(2))


@pytest.fixture(
    scope="module",
    params=tuple(product(range(3), range(2), (False, True), (False, True))),
)
def native_case(request):
    form_index, profile_index, ready, scalar = request.param
    forms, marks = FORMS[form_index], PROFILES[profile_index]
    graph = _source(forms, marks, ready=ready)
    expected = _oracle(graph, forms, marks)
    case = _execute(graph, scalar=scalar)
    case.update(expected=expected, profile=marks, ready=ready, forms=forms)
    return case


def test_complete_call_uses_actual_default_policies_in_declared_order(native_case):
    case = native_case
    assert [
        event["name"] for event in case["events"] if event["name"] != "capacity_write"
    ] == [
        "flow_history",
        "pressure",
        "si",
        "glyphs",
        "flow_history",
        "integration",
        "phase",
        "capacity",
        "flow_history",
        "remesh_history",
        "remesh_gate",
    ]
    for key in (
        "DT",
        "DT_MIN",
        "INTEGRATOR_METHOD",
        "EPI_MIN",
        "EPI_MAX",
        "CLIP_MODE",
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
    ):
        assert case["configuration"][key] == DEFAULTS[key]
    cfg = case["configuration"]
    assert DT == cfg["DT_MIN"]
    assert cfg["INTEGRATOR_METHOD"] == "euler" and cfg["CLIP_MODE"] == "hard"
    assert cfg.get("MATH_ENGINE") is None and cfg["GAMMA"]["type"] == "none"
    assert cfg["GRAMMAR_CANON"]["enabled"] is True
    assert cfg["GRAMMAR_REJECTION_MODE"] == "raise"
    assert "glyph_selector" not in cfg and "integrator" not in cfg
    assert cfg["compute_delta_nfr"] is default_compute_delta_nfr
    assert case["marks"] == tuple(
        (node, case["profile"][node // 2]) for node in range(10)
    )
    assert case["final"]["edges"] == case["initial"]["edges"]
    assert len(case["final"]["edges"]) == 20


def test_complete_block_endpoint_and_fresh_vs_held_pressure(native_case):
    case, expected = native_case, native_case["expected"]
    pressure, si, glyphs, integration = (
        _event(case, name) for name in ("pressure", "si", "glyphs", "integration")
    )
    assert pressure["after"]["pressure"] == pytest.approx(
        _lift(expected["initial_pressure"]), abs=ABS_ERROR, rel=0
    )
    assert si["after"]["si"] == pytest.approx(
        _lift(expected["si"]), abs=ABS_ERROR, rel=0
    )
    assert (
        min(si["after"]["si"]) >= case["configuration"]["SELECTOR_THRESHOLDS"]["si_hi"]
    )
    assert glyphs["after"]["form"] == pytest.approx(
        _lift(expected["post_event"]), abs=ABS_ERROR, rel=0
    )
    assert integration["before"]["pressure"] == glyphs["after"]["pressure"]
    assert integration["after"]["pressure"] == integration["before"]["pressure"]
    for name in ("form", "pressure", "rate", "acceleration"):
        assert case["final"][name] == pytest.approx(
            _lift(expected[name]), abs=ABS_ERROR, rel=0
        )
    assert case["final"]["si"] == si["after"]["si"]
    assert case["final"]["phase"] == (0.0,) * 10
    assert case["final"]["capacity"] == (1.0,) * 10
    for snapshot in (integration["before"], integration["after"], case["final"]):
        for first, second in PAIRS:
            assert all(
                snapshot[name][first] == snapshot[name][second] for name in CHANNELS
            )
    # A new constitutive evaluation is an observation after this call, not the
    # pressure integrated by it. Evaluate it independently from final form.
    coefficient = _weight(case["graph"], "DNFR_WEIGHTS", "epi")
    values = expected["form"]
    refreshed = tuple(
        coefficient * ((values[(j - 1) % 5] + values[(j + 1) % 5]) / 2 - x)
        for j, x in enumerate(values)
    )
    assert max(abs(a - b) for a, b in zip(refreshed, expected["pressure"])) > Q(
        1, 10000
    )


def test_live_phase_capacity_and_missing_remesh_history(native_case):
    case, expected = native_case, native_case["expected"]
    graph = case["graph"]
    phase, capacity, remesh = (
        _event(case, name) for name in ("phase", "capacity", "remesh_gate")
    )
    assert phase["after"]["phase"] == phase["before"]["phase"] == (0.0,) * 10
    hist, cfg = graph.graph["history"], case["configuration"]
    assert tuple(hist["phase_state"]) == ("stable",)
    assert tuple(hist["phase_R"]) == (1.0,) and tuple(hist["phase_disr"]) == (0.0,)
    for suffix, parameter in (("G", "PHASE_K_GLOBAL"), ("L", "PHASE_K_LOCAL")):
        incoming = Q(cfg[parameter])
        policy = cfg["PHASE_ADAPT"]
        target = Q(policy[f"k{suffix}_min"])
        value = incoming + Q(policy["down"]) * (target - incoming)
        assert hist[f"phase_k{suffix}"][0] == pytest.approx(
            float(value), abs=ABS_ERROR, rel=0
        )
    assert capacity["before"]["si"] == _event(case, "si")["after"]["si"]
    assert capacity["before"]["pressure"] == _event(case, "glyphs")["after"]["pressure"]
    count = cfg["VF_ADAPT_TAU"] if case["ready"] else 1
    counts = tuple(count if stable else 0 for stable in expected["stable"])
    assert case["final"]["counters"] == tuple(
        value for value in counts for _ in range(2)
    )
    writes = [event for event in case["events"] if event["name"] == "capacity_write"]
    targets = tuple(
        node for node in range(10) if case["ready"] and expected["stable"][node // 2]
    )
    assert tuple(event["node"] for event in writes) == targets
    assert all(event["value"] == 1.0 for event in writes)
    assert remesh["before"] == remesh["after"]
    assert all(event["name"] != "network_remesh" for event in case["events"])
    assert len(hist.get("stable_frac", ())) == 0
    assert "_REMESH_META" not in graph.graph and "_last_remesh_step" not in graph.graph
    assert len(graph.graph["_epi_hist"]) == 1
    assert graph.graph["_epi_hist"][-1] == dict(enumerate(case["final"]["form"]))


def test_temporal_grammar_and_runtime_memories_retain_their_own_clocks(native_case):
    case, graph = native_case, native_case["graph"]
    before = _event(case, "integration")["before"]
    final, hist = case["final"], graph.graph["history"]
    assert case["initial"]["ordinal"] is None and final["ordinal"] == 1
    assert final["time"] == 1.0 + DT
    for event in case["events"]:
        assert event["before"]["ordinal"] == event["after"]["ordinal"] == 0
    for node in graph:
        block, data = node // 2, graph.nodes[node]
        mark = case["profile"][block]
        assert final["temporal"][node] == (
            (1.0, before["form"][node]),
            (1.0 + DT, final["form"][node]),
        )
        assert tuple(data["glyph_history"]) == ("EN", mark)
        assert data[OPERATOR_STEP_KEY] == 4 and data[PRIOR_COHERENCE_KEY] is True
        assert data[U2_DEBT_KEY] == (0 if mark == "IL" else block % 2)
        lags = (0, 3) if mark == "AL" else (1, 0 if mark == "EN" else 1)
        assert (hist["since_AL"][node], hist["since_EN"][node]) == lags
    for first, second in PAIRS:
        assert read_grammar_state(graph.nodes[first]) == read_grammar_state(
            graph.nodes[second]
        )
        assert final["temporal"][first] == final["temporal"][second]


def test_saturated_emission_retains_rate_without_fabricating_measured_motion():
    forms = (31 / 32, 63 / 64, 1.0, 15 / 16, 127 / 128)
    graph = _source(forms, ("AL",) * 5)
    expected = _oracle(graph, forms, ("AL",) * 5)
    assert expected["post_event"] == (Q(1),) * 5
    assert expected["pressure"][0] > 0 and expected["pressure"][3] > 0

    case = _execute(graph)

    assert _event(case, "integration")["before"]["form"] == (1.0,) * 10
    assert case["final"]["form"] == pytest.approx(
        _lift(expected["form"]), abs=ABS_ERROR, rel=0
    )
    assert case["final"]["rate"] == pytest.approx(
        _lift(expected["rate"]), abs=ABS_ERROR, rel=0
    )
    for node in (*PAIRS[0], *PAIRS[3]):
        assert case["final"]["form"][node] == 1.0
        assert case["final"]["rate"][node] > 0
        assert case["final"]["temporal"][node] == ((1.0, 1.0), (1.0 + DT, 1.0))


def test_unequal_qualifying_counters_survive_without_a_physical_pair_split():
    graph = _source(FORMS[0], PROFILES[0], ready=True)
    expected = _oracle(graph, FORMS[0], PROFILES[0])
    assert expected["stable"][3] is True
    graph.nodes[PAIRS[3][0]]["stable_count"] = 0

    case = _execute(graph)

    first, second = PAIRS[3]
    assert all(
        case["final"][name][first] == case["final"][name][second] for name in CHANNELS
    )
    assert case["final"]["counters"][first] == 1
    assert case["final"]["counters"][second] == graph.graph["VF_ADAPT_TAU"]
    writes = [
        event["node"] for event in case["events"] if event["name"] == "capacity_write"
    ]
    assert first not in writes and second in writes
