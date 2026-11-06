"""Compatible-neighbor form association controls native Resonance capacity.

Freeze doubled C5, order 0..9, seed 17, t=1, dt=1/16, default laws, strict
grammar, supplied AL/IL/EN context and zero capacity counters. Pair zero has
form -21/32, capacity 1/4 and phase 2; pairs two/three have form -15/16 and
phase 1. Pairs one/four each have phases (0,2) and unordered forms {0,15/16}.
The two preparations swap only the latter form-phase association. The
compatible-neighbor mean is respectively zero or 15/16, turning the native
Resonance capacity trigger off/on with unchanged full-neighbor form means.

All four configurations, complete ideal-real form/capacity forecasts and phase
controller predicates precede the first reserved response (two associations,
scalar/NumPy). Fraction algebra uses represented default coefficients. Exact
alternating-series bounds certify the transitional phase controller branch;
the terminal circular angle uses an independent elementary-trig readout with
a fixed absolute 1e-14 numerical comparison allowance. This is neither a
universal binary64 certificate nor an autonomous occurrence/formation claim.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product

import pytest

from tests.physics import test_native_collective_mean as mean
from tests.physics import test_native_collective_step as base
from tests.physics import test_native_resonance_u3_boundary as boundary
from tnfr.constants import DEFAULTS
from tnfr.constants.canonical import DELTA_PHI_MAX
from tnfr.operators.grammar_debt import OPERATOR_STEP_KEY, U2_DEBT_KEY

AMPLITUDE, TARGET = Q(15, 16), Q(-21, 32)
KEYS = tuple(product((False, True), (False, True)))
MARKS = tuple((node, "RA" if node < 2 else "IL") for node in range(10))
ABS_ERROR = Q(1, 10**14)


def _source(active):
    graph = base._source(
        tuple(
            map(float, (TARGET, AMPLITUDE / 2, -AMPLITUDE, -AMPLITUDE, AMPLITUDE / 2))
        ),
        base.PROFILES[0],
    )
    for node in graph:
        block = node // 2
        if block == 0:
            form, phase, capacity = TARGET, Q(2), Q(1, 4)
        elif block in (1, 4):
            high_phase = bool(node % 2)
            form = AMPLITUDE if high_phase == active else Q(0)
            phase, capacity = Q(2 * high_phase), Q(1)
        else:
            form, phase, capacity = -AMPLITUDE, Q(1), Q(1)
        assert all(Q(float(value)) == value for value in (form, phase, capacity))
        graph.nodes[node].update(
            EPI=float(form), theta=float(phase), nu_f=float(capacity)
        )
    return graph


def _phase_certificate(graph):
    """Enclose sin(1), cos(1) by their decreasing alternating series."""
    cosine_low = sum(Q((-1) ** j, math.factorial(2 * j)) for j in range(8))
    cosine_high = cosine_low + Q(1, math.factorial(16))
    sine_low = sum(Q((-1) ** j, math.factorial(2 * j + 1)) for j in range(8))
    sine_high = sine_low + Q(1, math.factorial(17))
    assert 0 < cosine_low < cosine_high < 1 and 0 < sine_low < sine_high < 1
    # Four phases 2, four phases 1 and two phases 0 remain after RA.
    # Rotate by one radian: the resultant is (4+6*cos(1), 2*sin(1))/10.
    lower = ((4 + 6 * cosine_low) ** 2 + 4 * sine_low**2) / 100
    upper = ((4 + 6 * cosine_high) ** 2 + 4 * sine_high**2) / 100
    policy = graph.graph["PHASE_ADAPT"]
    assert Q(policy["R_lo"]) ** 2 < lower < upper < Q(policy["R_hi"]) ** 2
    assert Q(74, 100) ** 2 < lower < upper < Q(75, 100) ** 2
    gains = []
    for name, suffix in (("GLOBAL", "G"), ("LOCAL", "L")):
        current = Q(graph.graph[f"PHASE_K_{name}"])
        target = (Q(policy[f"k{suffix}_min"]) + Q(policy[f"k{suffix}_max"])) / 2
        speed = Q(policy["up"] if target > current else policy["down"])
        gains.append(current + speed * (target - current))
    assert all(0 < gain for gain in gains) and sum(gains) < 1
    return lower, upper, tuple(gains)


def _forecast(graph, active):
    """Use source algebra, never runtime pressure, selector or event kernels."""
    cfg = graph.graph
    k, b, w = (
        base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf", "phase")
    )
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    normalizer = Q(math.pi)
    assert (alpha, beta, gamma) == (w, k, b)
    compatible = AMPLITUDE if active else Q(0)
    rejected = AMPLITUDE - compatible
    adjacent_mean = (TARGET - AMPLITUDE) / 2
    target_pressure = 9 * k / 8 + 3 * b / 4 - w / normalizer
    high_pressure = k * (adjacent_mean - compatible) - 3 * b / 8 - w / (2 * normalizer)
    low_pressure = k * (adjacent_mean - rejected) - 3 * b / 8 + 3 * w / (2 * normalizer)
    middle_pressure = 3 * k * AMPLITUDE / 4
    pressure = (
        (target_pressure,) * 2
        + (low_pressure, high_pressure)
        + (middle_pressure,) * 4
        + (low_pressure, high_pressure)
    )
    peak = -high_pressure
    assert peak == max(map(abs, pressure))
    offsets = (Q(-1),) * 2 + (Q(3, 2), Q(-1, 2)) + (Q(0),) * 4 + (Q(3, 2), Q(-1, 2))
    si = tuple(
        alpha * (Q(1, 4) if node < 2 else 1)
        + beta * (1 - abs(offset) / normalizer)
        + gamma * (1 - abs(value) / peak)
        for node, (offset, value) in enumerate(zip(offsets, pressure, strict=True))
    )
    thresholds = cfg["SELECTOR_THRESHOLDS"]
    margins = (
        si[0] - Q(thresholds["si_lo"]),
        Q(thresholds["si_hi"]) - si[0],
        Q(thresholds["dnfr_hi"]) - target_pressure / peak,
        *(value - Q(thresholds["si_hi"]) for value in si[2:]),
    )
    assert 0 < target_pressure and min(margins) > Q(1, 20)
    limit = Q(cfg.get("DELTA_PHI_MAX", DELTA_PHI_MAX))
    assert 0 < limit < 2 and limit == normalizer / 2
    factors = cfg["GLYPH_FACTORS"]
    mix, boost, retention = (
        Q(factors[key])
        for key in ("RA_epi_diff", "RA_vf_amplification", "IL_dnfr_factor")
    )
    source = tuple(Q(graph.nodes[node]["EPI"]) for node in graph)
    phases = tuple(Q(graph.nodes[node]["theta"]) for node in graph)
    event_form = ((1 - mix) * TARGET + mix * compatible,) * 2 + source[2:]
    target_capacity = (1 + boost if active else Q(1)) / 4
    capacity = (target_capacity,) * 2 + (Q(1),) * 8
    held = pressure[:2] + tuple(retention * value for value in pressure[2:])
    rate = tuple(v * p for v, p in zip(capacity, held, strict=True))
    form = tuple(
        x + Q(base.DT) * velocity for x, velocity in zip(event_form, rate, strict=True)
    )
    assert Q(cfg["EPI_MIN"]) < event_form[0] < 0
    assert all(Q(cfg["EPI_MIN"]) < x < Q(cfg["EPI_MAX"]) for x in form)
    assert all(Q(cfg["VF_MIN"]) < v < Q(cfg["VF_MAX"]) for v in capacity)
    assert min(map(abs, held)) > Q(cfg["EPS_DNFR_STABLE"])
    lower, upper, gains = _phase_certificate(graph)
    # This independently evaluated trig readout is separate from the exact
    # real branch certificate above and receives the fixed numerical budget.
    global_phase = 1 + math.atan2(2 * math.sin(1), 4 + 6 * math.cos(1))
    local = (Q(1),) * 2 + (Q(3, 2),) * 2 + (Q(1),) * 4 + (Q(3, 2),) * 2
    phase = tuple(
        (1 - sum(gains)) * value + gains[0] * Q(global_phase) + gains[1] * neighbor
        for value, neighbor in zip(phases, local, strict=True)
    )
    assert all(0 < value < 2 for value in phase)
    return {
        "initial_pressure": pressure,
        "peak": peak,
        "si": si,
        "margins": margins,
        "compatible_mean": compatible,
        "event_form": event_form,
        "event_phase": phases,
        "capacity": capacity,
        "pressure": held,
        "rate": rate,
        "form": form,
        "phase": phase,
        "acceleration": tuple(
            (value - Q(base.RATES[node // 2])) / Q(base.DT)
            for node, value in enumerate(rate)
        ),
        "phase_R_squared": (lower, upper),
        "gains": gains,
    }


def _within(actual, expected, *, allowance=ABS_ERROR):
    assert abs(Q(actual) - expected) <= allowance


@pytest.fixture(scope="module")
def cases():
    # Freeze both associations and both backends before any reserved call.
    frozen = []
    for active, scalar in KEYS:
        graph = _source(active)
        configuration = deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key
                in set(DEFAULTS) | {"GRAMMAR_REJECTION_MODE", "compute_delta_nfr"}
            }
        )
        frozen.append((graph, configuration, active, scalar, _forecast(graph, active)))
    result = {}
    for graph, configuration, active, scalar, expected in frozen:
        case = boundary._execute(graph, configuration, scalar=scalar)
        case["expected"] = expected
        result[active, scalar] = case
    return result


@pytest.mark.parametrize("key", KEYS)
def test_fresh_pressure_si_and_complete_requests_precede_the_capacity_trigger(
    cases, key
):
    case, expected = cases[key], cases[key]["expected"]
    assert case["error"] is None and case["choices"] == MARKS
    for event, channel, forecast in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
    ):
        for actual, predicted in zip(
            base._event(case, event)["after"][channel], expected[forecast], strict=True
        ):
            _within(actual, predicted)
    observed_peak = max(map(abs, base._event(case, "pressure")["after"]["pressure"]))
    _within(observed_peak, expected["peak"])
    assert min(expected["margins"]) > Q(1, 20)
    assert case["prepared"] == (True,) * 10 and case["selector_cleared"] is True


@pytest.mark.parametrize("key", KEYS)
def test_live_partial_subset_controls_identity_and_capacity_amplification(cases, key):
    active, _ = key
    case, expected = cases[key], cases[key]["expected"]
    assert tuple(node for node, _, _, _ in case["u3"]) == (0, 1)
    for node, code, selection, neighbors in case["u3"]:
        assert code == "RA" and set(neighbors) == {3, 9}
        assert len(neighbors) == 2 < case["graph"].degree[node]
        assert selection.phases == (2.0, 2.0) and selection.target_phase == 2.0
        assert selection.effective_limit == float(Q(math.pi) / 2)
        before, after = case["primitives"][node][2:]
        selected_mean = sum(Q(before["form"][n]) for n in neighbors) / len(neighbors)
        assert selected_mean == expected["compatible_mean"]
        assert (
            sum(Q(before["form"][n]) for n in case["graph"].neighbors(node)) / 4
            == AMPLITUDE / 2
        )
        assert before["form"][node] < 0 and after["form"][node] < 0
        _within(after["form"][node], expected["event_form"][node])
        _within(after["capacity"][node], expected["capacity"][node])
        assert (after["capacity"][node] > before["capacity"][node]) is active
        assert before["phase"] == after["phase"]
        assert before["pressure"] == after["pressure"]
        assert case["graph"].nodes[node]["epi_kind"] == "seed"


@pytest.mark.parametrize("key", KEYS)
def test_complete_call_uses_triggered_capacity_with_the_same_held_pressure(cases, key):
    case, expected = cases[key], cases[key]["expected"]
    assert tuple((node, glyph) for node, glyph, _, _ in case["primitives"]) == MARKS
    for name in ("form", "capacity", "pressure", "rate", "acceleration", "si", "phase"):
        for actual, predicted in zip(case["final"][name], expected[name], strict=True):
            _within(actual, predicted)
    after = base._event(case, "glyphs")["after"]
    for name in ("form", "phase"):
        for actual, predicted in zip(
            after[name], expected[f"event_{name}"], strict=True
        ):
            _within(actual, predicted)
    integration = base._event(case, "integration")
    assert integration["before"]["capacity"] == case["final"]["capacity"]
    assert integration["before"]["pressure"] == integration["after"]["pressure"]
    assert case["final"]["counters"] == (0,) * 10
    assert not any(
        event["name"] in {"capacity_write", "network_remesh"}
        for event in case["events"]
    )
    assert (
        base._event(case, "capacity")["before"]["si"]
        == base._event(case, "si")["after"]["si"]
    )


@pytest.mark.parametrize("scalar", (False, True))
def test_identical_marginals_do_not_determine_the_native_capacity_trigger(
    cases, scalar
):
    off, on = cases[False, scalar], cases[True, scalar]
    assert off["initial"]["phase"] == on["initial"]["phase"]
    assert off["initial"]["capacity"] == on["initial"]["capacity"]
    assert sorted(off["initial"]["form"]) == sorted(on["initial"]["form"])
    assert mean._means(off["initial"]["form"]) == mean._means(on["initial"]["form"])
    assert (
        off["expected"]["initial_pressure"][:2]
        == on["expected"]["initial_pressure"][:2]
    )
    assert off["expected"]["si"][0] != on["expected"]["si"][0]
    assert off["expected"]["peak"] != on["expected"]["peak"]
    cfg = on["configuration"]
    boost, mix = (
        Q(cfg["GLYPH_FACTORS"][name]) for name in ("RA_vf_amplification", "RA_epi_diff")
    )
    rate_gap = boost * on["expected"]["pressure"][0] / 4
    form_gap = mix * AMPLITUDE + Q(base.DT) * rate_gap
    assert rate_gap > Q(1, 20000) and form_gap > Q(1, 10)
    for node in (0, 1):
        _within(
            Q(on["final"]["rate"][node]) - Q(off["final"]["rate"][node]),
            rate_gap,
            allowance=2 * ABS_ERROR,
        )
        _within(
            Q(on["final"]["form"][node]) - Q(off["final"]["form"][node]),
            form_gap,
            allowance=2 * ABS_ERROR,
        )
        _within(
            Q(on["final"]["capacity"][node]) - Q(off["final"]["capacity"][node]),
            boost / 4,
            allowance=2 * ABS_ERROR,
        )
    assert off["final"]["phase"] == on["final"]["phase"]


@pytest.mark.parametrize("key", KEYS)
def test_phase_controller_has_the_independently_certified_transition_branch(cases, key):
    case, expected = cases[key], cases[key]["expected"]
    history = case["graph"].graph["history"]
    assert tuple(history["phase_state"]) == ("transition",)
    assert tuple(history["phase_disr"]) == (0.0,)
    lower, upper = expected["phase_R_squared"]
    observed = Q(history["phase_R"][0])
    assert lower - 2 * ABS_ERROR <= observed**2 <= upper + 2 * ABS_ERROR
    for suffix, gain in zip(("G", "L"), expected["gains"], strict=True):
        _within(history[f"phase_k{suffix}"][0], gain)
    assert base._event(case, "glyphs")["after"]["phase"] == case["initial"]["phase"]
    assert case["final"]["phase"] != case["initial"]["phase"]
    assert all(0 < value < 2 for value in case["final"]["phase"])


@pytest.mark.parametrize("key", KEYS)
def test_declared_runtime_laws_context_and_history_are_preserved(cases, key):
    case, graph = cases[key], cases[key]["graph"]
    for name in (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "SELECTOR_THRESHOLDS",
        "GLYPH_FACTORS",
        "PHASE_ADAPT",
        "VF_ADAPT_TAU",
        "DT",
        "DT_MIN",
        "INTEGRATOR_METHOD",
        "GAMMA",
    ):
        assert case["configuration"][name] == DEFAULTS[name]
    assert case["configuration"]["GRAMMAR_REJECTION_MODE"] == "raise"
    assert case["configuration"]["GRAMMAR_CANON"]["enabled"] is True
    assert case["configuration"]["compute_delta_nfr"] is base.default_compute_delta_nfr
    assert base._event(case, "pressure")["result"] is base.default_compute_delta_nfr
    assert "glyph_selector" not in graph.graph and "integrator" not in graph.graph
    assert tuple(graph) == tuple(range(10)) and graph.graph["RANDOM_SEED"] == 17
    assert (
        case["initial"]["temporal"] == ((),) * 10
        and case["initial"]["counters"] == (0,) * 10
    )
    assert case["initial"]["ordinal"] is None and case["final"]["ordinal"] == 1
    assert case["final"]["time"] == 1 + base.DT
    assert (
        case["final"]["edges"] == case["initial"]["edges"]
        and len(case["final"]["edges"]) == 20
    )
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )
    history = graph.graph["history"]
    assert not history.get("stable_frac") and "_REMESH_META" not in graph.graph
    assert "ra_metrics" not in graph.graph and "_ra_c_tracking" not in graph.graph
    for node, data in graph.nodes(data=True):
        assert data[OPERATOR_STEP_KEY] == 4 and data[U2_DEBT_KEY] == 0
        assert data["epi_kind"] == "seed"
        assert tuple(data["glyph_history"]) == ("EN", MARKS[node][1])
        assert history["since_AL"][node] == history["since_EN"][node] == 1
        assert case["final"]["temporal"][node] == (
            (1.0, base._event(case, "integration")["before"]["form"][node]),
            (1 + base.DT, case["final"]["form"][node]),
        )
