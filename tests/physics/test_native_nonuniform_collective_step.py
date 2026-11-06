"""Prospective controls for one nonuniform native collective step.

Before native response evaluation, retain the doubled-C5 support, pair-contiguous
order, seed 17, supplied grammar context and t=1, dt=1/16 of the preceding gate.
Case A varies phase/capacity with a strict Coherence selector margin; case B
uses a single small phase/capacity defect with explicit closed-form predictions.
Both run scalar and NumPy paths. Two boundaries expose discarded adaptation
memory and independent Transition jitter, without changing any native policy.
The absolute binary64 comparison budget is 1e-14; pair equality and discrete
history comparisons are exact. These controls establish neither generic closure
nor formation, autonomous selection, repeated-call invariance or physical identity.
"""

import hashlib
import math
import random
import struct
from fractions import Fraction as Q
from itertools import product

import pytest

from tests.physics import test_native_collective_step as base
from tnfr.constants import DEFAULTS
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    read_grammar_state,
)

PHASES = (0.0, 1 / 8, -1 / 16, 1 / 4, -1 / 8)
CAPACITIES = (3 / 4, 1.0, 7 / 8, 9 / 8, 15 / 16)
DEFECT = 1 / 1024
SMALL_PHASES = (DEFECT, 0.0, 0.0, 0.0, 0.0)
SMALL_CAPACITIES = (1 + DEFECT, 1.0, 1.0, 1.0, 1.0)
SMALL_FORMS = (1 / 8,) * 5


def _source(forms, phases, capacities, marks, *, ready=False):
    graph = base._source(forms, marks, ready=ready)
    for node in graph:
        graph.nodes[node].update(theta=phases[node // 2], nu_f=capacities[node // 2])
    return graph


def _angle(values):
    cosine = math.fsum(math.cos(value) for value in values)
    sine = math.fsum(math.sin(value) for value in values)
    assert cosine != 0.0 or sine != 0.0
    return math.atan2(sine, cosine)


def _difference(target, source):
    return math.atan2(math.sin(target - source), math.cos(target - source))


def _neighbor_mean(values, block):
    return (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2


def _oracle(graph, forms, phases, capacities, marks):
    """Predict from supplied primitives; consume no engine-produced response."""
    cfg = graph.graph
    w_phase, w_epi, w_vf = (
        float(base._weight(graph, "DNFR_WEIGHTS", name))
        for name in ("phase", "epi", "vf")
    )
    local = tuple(
        _angle((phases[(block - 1) % 5], phases[(block + 1) % 5])) for block in range(5)
    )
    contrasts = tuple(
        _difference(target, phase) for target, phase in zip(local, phases)
    )
    pressure = tuple(
        w_phase * contrasts[block] / math.pi
        + w_epi * (_neighbor_mean(forms, block) - forms[block])
        + w_vf * (_neighbor_mean(capacities, block) - capacities[block])
        for block in range(5)
    )
    alpha, beta, gamma = (
        float(base._weight(graph, "SI_WEIGHTS", name))
        for name in ("alpha", "beta", "gamma")
    )
    peak = max(map(abs, pressure))
    si = tuple(
        alpha * capacities[block] / max(capacities)
        + beta * (1 - abs(contrasts[block]) / math.pi)
        + gamma * (1 - abs(pressure[block]) / peak if peak else 1)
        for block in range(5)
    )
    lower, upper = Q(cfg["EPI_MIN"]), Q(cfg["EPI_MAX"])

    def clip(value):
        return min(upper, max(lower, value))

    boost, mix, retention = (
        Q(cfg["GLYPH_FACTORS"][key]) for key in ("AL_boost", "EN_mix", "IL_dnfr_factor")
    )
    values, held = list(map(Q, forms)), list(map(Q, pressure))
    for block, mark in enumerate(marks):
        if mark == "AL":
            values[block] = clip(values[block] + boost)
        elif mark == "EN":
            values[block] = (1 - mix) * values[block] + mix * _neighbor_mean(
                values, block
            )
        else:
            assert mark == "IL"
            held[block] *= retention
    rates = tuple(Q(capacity) * p for capacity, p in zip(capacities, held))
    endpoint = tuple(clip(x + Q(base.DT) * rate) for x, rate in zip(values, rates))
    acceleration = tuple(
        (rate - Q(previous)) / Q(base.DT) for rate, previous in zip(rates, base.RATES)
    )
    phase_order = (
        math.hypot(math.fsum(map(math.cos, phases)), math.fsum(map(math.sin, phases)))
        / 5
    )
    policy = cfg["PHASE_ADAPT"]
    assert phase_order > policy["R_hi"]
    gains = tuple(
        float(Q(cfg[key]) + Q(policy["down"]) * (Q(policy[target]) - Q(cfg[key])))
        for key, target in (("PHASE_K_GLOBAL", "kG_min"), ("PHASE_K_LOCAL", "kL_min"))
    )
    global_phase = _angle(phases)
    coordinated = tuple(
        phase + gains[0] * _difference(global_phase, phase) + gains[1] * contrast
        for phase, contrast in zip(phases, contrasts)
    )
    stable = tuple(
        abs(p) <= Q(cfg["EPS_DNFR_STABLE"]) and s >= cfg["SELECTOR_THRESHOLDS"]["si_hi"]
        for p, s in zip(held, si)
    )
    counts = tuple(
        graph.nodes[2 * block]["stable_count"] + 1 if stable[block] else 0
        for block in range(5)
    )
    mu = Q(cfg["VF_ADAPT_MU"])
    vf = tuple(map(Q, capacities))
    adapted = tuple(
        (
            (1 - mu) * value + mu * _neighbor_mean(vf, block)
            if counts[block] >= cfg["VF_ADAPT_TAU"]
            else value
        )
        for block, value in enumerate(vf)
    )
    return {
        "initial_pressure": pressure,
        "si": si,
        "post_event": tuple(values),
        "pressure": tuple(held),
        "form": endpoint,
        "rate": rates,
        "acceleration": acceleration,
        "phase": coordinated,
        "capacity": adapted,
        "counts": counts,
        "stable": stable,
        "phase_order": phase_order,
        "gains": gains,
    }


CASES = tuple(
    ("contrast", profile, ready, scalar)
    for profile, ready, scalar in product(range(2), (False, True), (False, True))
) + tuple(("small", 0, True, scalar) for scalar in (False, True))


@pytest.fixture(scope="module", params=CASES)
def nonuniform_case(request):
    kind, profile, ready, scalar = request.param
    forms, phases, capacities = (
        (base.FORMS[0], PHASES, CAPACITIES)
        if kind == "contrast"
        else (SMALL_FORMS, SMALL_PHASES, SMALL_CAPACITIES)
    )
    marks = base.PROFILES[profile]
    graph = _source(forms, phases, capacities, marks, ready=ready)
    expected = _oracle(graph, forms, phases, capacities, marks)
    # This sufficient bound fixes the selector branch before any response.
    alpha = base._weight(graph, "SI_WEIGHTS", "alpha")
    assert alpha * Q(min(capacities)) / Q(max(capacities)) > Q(1, 2)
    assert min(expected["si"]) > graph.graph["SELECTOR_THRESHOLDS"]["si_hi"]
    case = base._execute(graph, scalar=scalar)
    case.update(expected=expected, profile=marks, kind=kind, forms=forms, phases=phases)
    return case


def test_nonuniform_step_keeps_actual_default_policies_and_selected_marks(
    nonuniform_case,
):
    case = nonuniform_case
    cfg = case["configuration"]
    for key in (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "GLYPH_FACTORS",
        "SELECTOR_THRESHOLDS",
        "PHASE_ADAPT",
        "PHASE_K_GLOBAL",
        "PHASE_K_LOCAL",
        "VF_ADAPT_MU",
        "VF_ADAPT_TAU",
    ):
        assert cfg[key] == DEFAULTS[key]
    assert cfg["GRAMMAR_REJECTION_MODE"] == "raise"
    assert "glyph_selector" not in cfg and "integrator" not in cfg
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
    assert case["marks"] == tuple(
        (node, case["profile"][node // 2]) for node in range(10)
    )
    assert case["final"]["edges"] == case["initial"]["edges"]


def test_nonuniform_pressure_and_complete_endpoint_match_independent_map(
    nonuniform_case,
):
    case, expected = nonuniform_case, nonuniform_case["expected"]
    for owner, channel, predicted in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
        ("glyphs", "form", "post_event"),
    ):
        assert base._event(case, owner)["after"][channel] == pytest.approx(
            base._lift(expected[predicted]), abs=base.ABS_ERROR, rel=0
        )
    for channel in base.CHANNELS:
        assert case["final"][channel] == pytest.approx(
            base._lift(expected[channel]), abs=base.ABS_ERROR, rel=0
        )
    assert case["final"]["pressure"] == base._event(case, "glyphs")["after"]["pressure"]
    assert case["final"]["si"] == base._event(case, "si")["after"]["si"]
    for owner in ("glyphs", "integration", "phase", "capacity"):
        snapshot = base._event(case, owner)["after"]
        for first, second in base.PAIRS:
            assert all(
                snapshot[channel][first] == snapshot[channel][second]
                for channel in base.CHANNELS
            )


def test_nonuniform_phase_and_capacity_use_snapshot_laws_and_held_gates(
    nonuniform_case,
):
    case, expected = nonuniform_case, nonuniform_case["expected"]
    graph, cfg = case["graph"], case["configuration"]
    history = graph.graph["history"]
    # The runtime clamp preserves this source's centered representatives.
    # Coordination then stores its raw proposal, without a second phase wrap.
    assert base._event(case, "phase")["before"]["phase"] == base._lift(case["phases"])
    assert tuple(history["phase_state"]) == ("stable",)
    assert tuple(history["phase_disr"]) == (0.0,)
    assert history["phase_R"][0] == pytest.approx(
        expected["phase_order"], abs=base.ABS_ERROR, rel=0
    )
    assert (history["phase_kG"][0], history["phase_kL"][0]) == pytest.approx(
        expected["gains"], abs=base.ABS_ERROR, rel=0
    )
    assert (
        max(
            abs(a - b) for a, b in zip(case["final"]["phase"], case["initial"]["phase"])
        )
        > 1e-5
    )
    capacity = base._event(case, "capacity")
    assert (
        capacity["before"]["pressure"]
        == base._event(case, "glyphs")["after"]["pressure"]
    )
    assert capacity["before"]["si"] == base._event(case, "si")["after"]["si"]
    assert case["final"]["counters"] == tuple(
        value for value in expected["counts"] for _ in range(2)
    )
    writes = [event for event in case["events"] if event["name"] == "capacity_write"]
    assert tuple(event["node"] for event in writes) == tuple(
        node
        for node in range(10)
        if expected["counts"][node // 2] >= cfg["VF_ADAPT_TAU"]
    )
    remesh = base._event(case, "remesh_gate")
    assert remesh["before"] == remesh["after"]
    assert not history.get("stable_frac")
    assert len(graph.graph["_epi_hist"]) == 1
    if case["kind"] == "small":
        _assert_small_defect_prediction(case)


def test_nonuniform_grammar_and_temporal_memory_remain_pair_associated(nonuniform_case):
    case, graph = nonuniform_case, nonuniform_case["graph"]
    before = base._event(case, "integration")["before"]
    final = case["final"]
    assert final["ordinal"] == 1 and final["time"] == 1 + base.DT
    for node in graph:
        data, mark = graph.nodes[node], case["profile"][node // 2]
        assert final["temporal"][node] == (
            (1.0, before["form"][node]),
            (1 + base.DT, final["form"][node]),
        )
        assert tuple(data["glyph_history"]) == ("EN", mark)
        assert data[OPERATOR_STEP_KEY] == 4 and data[PRIOR_COHERENCE_KEY] is True
        assert data[U2_DEBT_KEY] == (0 if mark == "IL" else (node // 2) % 2)
    for first, second in base.PAIRS:
        assert read_grammar_state(graph.nodes[first]) == read_grammar_state(
            graph.nodes[second]
        )
        assert final["temporal"][first] == final["temporal"][second]
        assert final["counters"][first] == final["counters"][second]


def _assert_small_defect_prediction(case):
    graph, cfg = case["graph"], case["configuration"]
    w_phase, w_vf = (
        float(base._weight(graph, "DNFR_WEIGHTS", key)) for key in ("phase", "vf")
    )
    p0 = -DEFECT * (w_phase / math.pi + w_vf)
    pressure = (p0, -p0 / 2, 0.0, 0.0, -p0 / 2)
    assert base._event(case, "pressure")["after"]["pressure"] == pytest.approx(
        base._lift(pressure), abs=base.ABS_ERROR, rel=0
    )
    assert all(case["expected"]["stable"])
    mu, eps = Q(cfg["VF_ADAPT_MU"]), Q(DEFECT)
    adapted = (1 + (1 - mu) * eps, 1 + mu * eps / 2, 1, 1, 1 + mu * eps / 2)
    assert case["final"]["capacity"] == pytest.approx(
        base._lift(adapted), abs=base.ABS_ERROR, rel=0
    )
    kg, kl = case["expected"]["gains"]
    global_phase = math.atan2(math.sin(DEFECT), 4 + math.cos(DEFECT))
    phases = (
        (1 - kg - kl) * DEFECT + kg * global_phase,
        kg * global_phase + kl * DEFECT / 2,
        kg * global_phase,
        kg * global_phase,
        kg * global_phase + kl * DEFECT / 2,
    )
    assert case["final"]["phase"] == pytest.approx(
        base._lift(phases), abs=base.ABS_ERROR, rel=0
    )
    assert case["final"]["capacity"][0] < case["initial"]["capacity"][0]
    assert case["final"]["capacity"][2] > case["initial"]["capacity"][2]
    assert case["final"]["phase"][0] < DEFECT
    assert case["final"]["phase"][4] > 0


@pytest.mark.parametrize("scalar", (False, True))
def test_discarded_adaptation_counter_creates_predicted_capacity_pair_split(scalar):
    graph = _source(
        SMALL_FORMS, SMALL_PHASES, SMALL_CAPACITIES, base.PROFILES[0], ready=True
    )
    expected = _oracle(
        graph, SMALL_FORMS, SMALL_PHASES, SMALL_CAPACITIES, base.PROFILES[0]
    )
    assert all(expected["stable"])
    gap = -Q(graph.graph["VF_ADAPT_MU"]) * Q(DEFECT)
    graph.nodes[0]["stable_count"] = 0

    case = base._execute(graph, scalar=scalar)

    for channel in base.CHANNELS.keys() - {"capacity"}:
        assert case["final"][channel][0] == case["final"][channel][1]
    assert case["final"]["capacity"][1] - case["final"]["capacity"][0] == pytest.approx(
        float(gap), abs=base.ABS_ERROR, rel=0
    )
    assert case["final"]["counters"][:2] == (1, graph.graph["VF_ADAPT_TAU"])
    targets = [
        event["node"] for event in case["events"] if event["name"] == "capacity_write"
    ]
    assert 0 not in targets and 1 in targets


def _jitter(seed, offset, amplitude):
    """First node-local sample from the declared two-level seed derivation."""

    def digest(left, right):
        payload = struct.pack(">QQ", left & (2**64 - 1), right & (2**64 - 1))
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    return random.Random(digest(digest(seed, offset), 0)).uniform(-amplitude, amplitude)


def test_native_transition_jitter_breaks_pair_closure_from_matched_physical_memory():
    capacities, forms, phases = (1 / 4, 1.0, 1.0, 1.0, 1.0), (1 / 4,) * 5, (0.0,) * 5
    graph = _source(forms, phases, capacities, base.PROFILES[0])
    expected = _oracle(graph, forms, phases, capacities, base.PROFILES[0])
    cfg = graph.graph
    alpha, beta = (
        float(base._weight(graph, "SI_WEIGHTS", key)) for key in ("alpha", "beta")
    )
    assert (
        cfg["SELECTOR_THRESHOLDS"]["si_lo"]
        < alpha / 4 + beta
        < cfg["SELECTOR_THRESHOLDS"]["si_hi"]
    )
    assert expected["si"][0] == pytest.approx(
        alpha / 4 + beta, abs=base.ABS_ERROR, rel=0
    )
    assert min(expected["si"][1:]) > cfg["SELECTOR_THRESHOLDS"]["si_hi"]
    assert cfg["NAV_RANDOM"] is True and cfg["NAV_STRICT"] is False
    amplitude = cfg["GLYPH_FACTORS"]["NAV_jitter"]
    jitter = tuple(_jitter(17, node, amplitude) for node in (0, 1))
    assert jitter[0] < 0 < jitter[1]
    eta = cfg["GLYPH_FACTORS"]["NAV_eta"]
    deterministic = (1 - eta) * expected["initial_pressure"][0] + eta / 4
    predicted_pressure = tuple(deterministic + draw for draw in jitter)
    predicted_form = tuple(
        1 / 4 + base.DT * pressure / 4 for pressure in predicted_pressure
    )
    gap = base.DT * (jitter[1] - jitter[0]) / 4
    assert gap > 1e-4

    case = base._execute(graph, scalar=True)

    for first, second in base.PAIRS:
        assert all(
            case["initial"][channel][first] == case["initial"][channel][second]
            for channel in base.CHANNELS
        )
        assert case["initial"]["counters"][first] == case["initial"]["counters"][second]
        assert case["initial"]["temporal"][first] == case["initial"]["temporal"][second]
    assert base._event(case, "pressure")["after"]["pressure"] == pytest.approx(
        base._lift(expected["initial_pressure"]), abs=base.ABS_ERROR, rel=0
    )
    assert base._event(case, "si")["after"]["si"] == pytest.approx(
        base._lift(expected["si"]), abs=base.ABS_ERROR, rel=0
    )
    assert case["marks"] == tuple(
        (node, "NAV" if node < 2 else "IL") for node in range(10)
    )
    glyphs = base._event(case, "glyphs")
    assert glyphs["after"]["pressure"][:2] == pytest.approx(
        predicted_pressure, abs=base.ABS_ERROR, rel=0
    )
    assert glyphs["after"]["form"] == (1 / 4,) * 10
    assert case["final"]["form"][:2] == pytest.approx(
        predicted_form, abs=base.ABS_ERROR, rel=0
    )
    assert case["final"]["form"][1] - case["final"]["form"][0] == pytest.approx(
        gap, abs=base.ABS_ERROR, rel=0
    )
    assert case["final"]["phase"] == (0.0,) * 10
    assert case["final"]["capacity"] == case["initial"]["capacity"]
    for node in (0, 1):
        assert graph.nodes[node]["_rng_jitter_progress"] == {
            "seed": 17,
            "offset": node,
            "draws": 1,
        }
    assert all("_rng_jitter_progress" not in graph.nodes[node] for node in range(2, 10))
