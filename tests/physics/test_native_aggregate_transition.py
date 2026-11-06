"""One default-selected native transition on a uniform-capacity quotient.

Freeze two doubled-C5 lifts with contrasts d and -d over one coarse C5 source.
Every graph has capacity 1/2, paired phases, common kind and paired grammar/
lags. Actual lag priority selects AL, EN, IL, IL, IL; no selector is supplied.
All forecasts precede exactly one runtime.step per graph, with dt=1/16,
default pressure/Si/controller/adaptation and live built-in zero Gamma.
Next preparation and detached selection are checked without another call.
This finite conditional transition is not formation or autonomous selection;
Si, individual counters and temporal records are not identified by the quotient.
"""

import math
from functools import wraps

import networkx as nx
import pytest

from tests.physics import test_native_collective_step as native
from tnfr._runtime_steps import RUNTIME_STEP_NEXT_KEY
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.dynamics import integrators, runtime, selectors
from tnfr.dynamics.dnfr import _resolve_dnfr_weights, default_compute_delta_nfr
from tnfr.metrics.sense_index import get_Si_weights
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    canonical_operator_code,
    read_grammar_state,
    reset_grammar_state_from_history,
)

MEANS = (1 / 4, 3 / 8, 207 / 512, 7 / 16, 3 / 16)
CONTRASTS = (1 / 64, -1 / 128, 1 / 128, 1 / 64, -1 / 128)
PHASES = (1, 17 / 16, 1, 15 / 16, 1)
CAPACITY = 1 / 2
OLD_RATES = (-1 / 32, 0, 1 / 32, -1 / 16, 1 / 16)
RATE_CONTRASTS = (1 / 128, -1 / 128, 1 / 64, 1 / 128, -1 / 64)
LAG_AL, LAG_EN = (5, 0, 0, 0, 0), (3, 3, 0, 0, 0)
MARKS = ("AL", "EN", "IL", "IL", "IL")
NEXT_MARKS = ("EN", "IL", "IL", "IL", "IL")
DT, TOL = 1 / 16, 1e-14
FORM_RADIUS = 1 / 64


def _expand(means, contrasts, copies):
    if copies == 1:
        return tuple(means)
    return tuple(m + sigma * d for m, d in zip(means, contrasts) for sigma in (-1, 1))


def _means(values, copies):
    return tuple(sum(values[copies * i : copies * (i + 1)]) / copies for i in range(5))


def _neighbor(values, block):
    return (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2


def _source(copies, sign):
    graph = nx.Graph()
    graph.add_nodes_from(range(5 * copies))
    for block in range(5):
        for left in range(copies):
            for right in range(copies):
                graph.add_edge(
                    copies * block + left,
                    copies * ((block + 1) % 5) + right,
                    weight=1.0,
                )
    inject_defaults(graph)
    graph.graph.update(
        _t=1.0,
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        compute_delta_nfr=default_compute_delta_nfr,
        history={"since_AL": {}, "since_EN": {}},
    )
    forms = _expand(MEANS, tuple(sign * d for d in CONTRASTS), copies)
    rates = _expand(OLD_RATES, tuple(sign * d for d in RATE_CONTRASTS), copies)
    for n in graph:
        block = n // copies
        data = graph.nodes[n]
        data.update(
            EPI=forms[n],
            nu_f=CAPACITY,
            theta=PHASES[block],
            epi_kind="seed",
            delta_nfr=0.0,
            dEPI=rates[n],
            d2EPI=0.0,
            glyph_history=["AL", "IL"],
            stable_count=4,
            epi_time_history=[(7 / 8, forms[n] - rates[n] / 8), (1.0, forms[n])],
            **{OPERATOR_STEP_KEY: 2},
        )
        reset_grammar_state_from_history(data)
        graph.graph["history"]["since_AL"][n] = LAG_AL[block]
        graph.graph["history"]["since_EN"][n] = LAG_EN[block]
    return graph


def _readout(means, contrasts, phases, copies, weights, si_weights):
    # Two adjacent coarse phases in this common lift have their arithmetic
    # midpoint as circular mean; duplicated fine phasors have the same target.
    local = tuple(_neighbor(phases, i) for i in range(5))
    k, t = weights["epi"], weights["phase"] / math.pi
    baseline = tuple(
        k * (_neighbor(means, i) - means[i]) + t * (local[i] - phases[i])
        for i in range(5)
    )
    pressure = _expand(baseline, tuple(-k * d for d in contrasts), copies)
    peak = max(map(abs, pressure))
    alpha, beta, gamma = si_weights
    si = tuple(
        alpha
        + beta * (1 - abs(local[n // copies] - phases[n // copies]) / math.pi)
        + gamma * (1 - abs(p) / (peak or 1))
        for n, p in enumerate(pressure)
    )
    assert min(si) >= alpha > 1 / 2
    return baseline, pressure, si


def _forecast(graph, copies, sign):
    cfg = graph.graph
    for key in (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "GLYPH_FACTORS",
        "SELECTOR_THRESHOLDS",
        "PHASE_ADAPT",
        "PHASE_K_GLOBAL",
        "PHASE_K_LOCAL",
        "AL_MAX_LAG",
        "EN_MAX_LAG",
        "VF_ADAPT_TAU",
        "VF_ADAPT_MU",
        "EPS_DNFR_STABLE",
        "EPI_MIN",
        "EPI_MAX",
        "VF_MIN",
        "VF_MAX",
        "CLIP_MODE",
        "THETA_WRAP",
        "INTEGRATOR_METHOD",
        "DT_MIN",
        "GAMMA",
        "GRAMMAR_CANON",
        "REMESH_STABILITY_WINDOW",
    ):
        assert cfg[key] == DEFAULTS[key]
    assert native.DT == DT
    weights, si_weights = _resolve_dnfr_weights(graph), get_Si_weights(graph)
    d = tuple(sign * value for value in CONTRASTS)
    baseline, pressure, si = _readout(MEANS, d, PHASES, copies, weights, si_weights)
    factors = cfg["GLYPH_FACTORS"]
    boost, mix, retention = (
        factors[key] for key in ("AL_boost", "EN_mix", "IL_dnfr_factor")
    )
    event_mean = list(MEANS)
    event_mean[0] += boost
    # EN consumes the completed preceding AL pair, not the initial mean.
    event_mean[1] = (1 - mix) * MEANS[1] + mix * _neighbor(event_mean, 1)
    attenuation = (1, 1 - mix, 1, 1, 1)
    hold = (1, 1, retention, retention, retention)
    event_d = tuple(a * value for a, value in zip(attenuation, d))
    event = _expand(event_mean, event_d, copies)
    held = tuple(hold[n // copies] * p for n, p in enumerate(pressure))
    rates = tuple(CAPACITY * p for p in held)
    old_rates = _expand(
        OLD_RATES, tuple(sign * value for value in RATE_CONTRASTS), copies
    )
    acceleration = tuple((new - old) / DT for new, old in zip(rates, old_rates))
    final_mean = tuple(
        y + DT * CAPACITY * h * b for y, h, b in zip(event_mean, hold, baseline)
    )
    contrast_multiplier = tuple(
        a - DT * CAPACITY * h * weights["epi"] for a, h in zip(attenuation, hold)
    )
    final_d = tuple(factor * value for factor, value in zip(contrast_multiplier, d))
    # Frozen whole-fiber admission, stronger than checking the two fine lifts:
    # every |d_i| <= D stays inside the shared hard rails through each stage.
    assert all(abs(value) <= FORM_RADIUS for value in d)
    assert all(abs(factor) <= 1 for factor in contrast_multiplier)
    assert all(
        cfg["EPI_MIN"] + FORM_RADIUS < mean < cfg["EPI_MAX"] - FORM_RADIUS
        for mean in (*MEANS, *event_mean, *final_mean)
    )
    endpoint = _expand(final_mean, final_d, copies)
    assert all(cfg["EPI_MIN"] < value < cfg["EPI_MAX"] for value in (*event, *endpoint))

    policy = cfg["PHASE_ADAPT"]
    z = sum(complex(math.cos(p), math.sin(p)) for p in PHASES)
    resultant, angle = abs(z) / 5, math.atan2(z.imag, z.real)
    assert (
        resultant > policy["R_hi"]
    )  # Updated AL/EN/IL histories have zero disruptor load.
    gains = tuple(
        cfg[key] + policy["down"] * (policy[target] - cfg[key])
        for key, target in (("PHASE_K_GLOBAL", "kG_min"), ("PHASE_K_LOCAL", "kL_min"))
    )
    phase = tuple(
        p + gains[0] * (angle - p) + gains[1] * (_neighbor(PHASES, i) - p)
        for i, p in enumerate(PHASES)
    )
    counts = tuple(5 if abs(p) <= cfg["EPS_DNFR_STABLE"] else 0 for p in held)
    # This signed fixture exercises different eligibility with the same
    # immutable constant-capacity snapshot, whose adaptation is a fixed point.
    if copies == 1:
        assert counts == (0, 0, 5, 0, 0)
    elif sign == 1:
        assert counts == (0, 0, 0, 0, 0, 5, 0, 0, 0, 0)
    else:
        assert counts == (0, 0, 0, 0, 5, 0, 0, 0, 0, 0)
    _, next_pressure, next_si = _readout(
        final_mean, final_d, phase, copies, weights, si_weights
    )
    return dict(
        prepared=pressure,
        si=si,
        event=event,
        held=held,
        rates=rates,
        acceleration=acceleration,
        endpoint=endpoint,
        final_mean=final_mean,
        final_d=final_d,
        phase=_expand(phase, (0,) * 5, copies),
        gains=gains,
        resultant=resultant,
        counts=counts,
        next_pressure=next_pressure,
        next_si=next_si,
        alpha=si_weights[0],
        old_rates=old_rates,
    )


@pytest.fixture(scope="module")
def transitions():
    sources = tuple(
        (_source(copies, sign), copies, sign)
        for copies, sign in ((2, 1), (2, -1), (1, 0))
    )
    forecasts = tuple(_forecast(graph, copies, sign) for graph, copies, sign in sources)
    actual_step = runtime.step
    invocations = {}

    @wraps(actual_step)
    def once(graph, *args, **kwargs):
        invocations[id(graph)] = invocations.get(id(graph), 0) + 1
        assert invocations[id(graph)] == 1
        return actual_step(graph, *args, **kwargs)

    def no_campaign(*args, **kwargs):
        pytest.fail("Only one frozen native invocation per source is admitted")

    cases = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "step", once)
        patch.setattr(runtime, "run", no_campaign)
        for (graph, copies, sign), forecast in zip(sources, forecasts):
            assert (
                "glyph_selector" not in graph.graph and "integrator" not in graph.graph
            )
            assert not graph.graph.get("callbacks") and not graph.graph.get(
                "MATH_ENGINE"
            )
            gamma = integrators._get_gamma_spec(graph, strict=True)
            assert gamma["type"] == "none" and integrators._uses_builtin_gamma(gamma)
            case = native._execute(
                graph
            )  # Existing forwarding spies; one actual runtime.step.
            case.update(copies=copies, sign=sign, expected=forecast)
            runtime._prepare_dnfr(graph, use_Si=True)
            case["next_readout"] = native._snapshot(graph)
            hist = graph.graph["history"]
            next_al = {n: hist["since_AL"][n] + 1 for n in graph}
            next_en = {n: hist["since_EN"][n] + 1 for n in graph}
            selector = selectors._apply_selector(graph)
            case["next_choices"] = tuple(
                canonical_operator_code(
                    selectors._choose_glyph(
                        graph,
                        n,
                        selector,
                        True,
                        next_al,
                        next_en,
                        graph.graph["AL_MAX_LAG"],
                        graph.graph["EN_MAX_LAG"],
                    )
                )
                for n in graph
            )
            cases.append(case)
    assert tuple(invocations.values()) == (1, 1, 1)
    return tuple(cases)


def _close(actual, expected):
    assert actual == pytest.approx(expected, abs=TOL, rel=0)


def test_actual_native_selection_and_complete_stage_order(transitions):
    for case in transitions:
        copies = case["copies"]
        assert tuple(
            canonical_operator_code(mark) for _, mark in case["marks"]
        ) == tuple(mark for mark in MARKS for _ in range(copies))
        assert [
            event["name"]
            for event in case["events"]
            if event["name"] != "capacity_write"
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
        assert case["final"]["time"] == 1 + DT
        assert case["graph"].graph[RUNTIME_STEP_NEXT_KEY] == 1
        assert case["final"]["edges"] == case["initial"]["edges"]


def test_independent_prepared_and_held_rows_with_live_reception(transitions):
    for case in transitions:
        expected = case["expected"]
        _close(
            native._event(case, "pressure")["after"]["pressure"], expected["prepared"]
        )
        _close(native._event(case, "si")["after"]["si"], expected["si"])
        event = native._event(case, "glyphs")["after"]
        _close(event["form"], expected["event"])
        _close(event["pressure"], expected["held"])
        assert event["si"] == native._event(case, "si")["after"]["si"]
        factors = case["graph"].graph["GLYPH_FACTORS"]
        stale = (1 - factors["EN_mix"]) * MEANS[1] + factors["EN_mix"] * _neighbor(
            MEANS, 1
        )
        live = _means(event["form"], case["copies"])[1]
        _close(live - stale, factors["EN_mix"] * factors["AL_boost"] / 2)


def test_two_fine_lifts_share_the_actual_coarse_physical_successor(transitions):
    coarse = transitions[2]["final"]
    for case in transitions:
        expected = case["expected"]
        for channel, key in (
            ("form", "endpoint"),
            ("pressure", "held"),
            ("phase", "phase"),
            ("rate", "rates"),
            ("acceleration", "acceleration"),
        ):
            _close(case["final"][channel], expected[key])
            _close(_means(case["final"][channel], case["copies"]), coarse[channel])
        assert case["final"]["capacity"] == (CAPACITY,) * (5 * case["copies"])
        assert all(case["graph"].nodes[n]["epi_kind"] == "seed" for n in case["graph"])
    for case in transitions[:2]:
        actual_d = tuple(
            (case["final"]["form"][2 * i + 1] - case["final"]["form"][2 * i]) / 2
            for i in range(5)
        )
        _close(actual_d, case["expected"]["final_d"])
        assert all(abs(value) > 1e-4 for value in actual_d)


def test_uniform_capacity_survives_different_actual_adaptation_eligibility(transitions):
    for case in transitions:
        assert case["final"]["counters"] == case["expected"]["counts"]
        capacity = native._event(case, "capacity")
        assert capacity["before"]["capacity"] == capacity["after"]["capacity"]
        assert capacity["before"]["si"] == native._event(case, "si")["after"]["si"]
    assert transitions[0]["final"]["counters"][4:6] == (0, 5)
    assert transitions[1]["final"]["counters"][4:6] == (5, 0)
    assert transitions[2]["final"]["counters"][2] == 5
    # Prepared Si itself is not the descended state: fine pressure peaks differ.
    fine_si = _means(transitions[0]["expected"]["si"], 2)
    coarse_si = transitions[2]["expected"]["si"]
    assert max(abs(a - b) for a, b in zip(fine_si, coarse_si)) > 1e-4


def test_common_controller_and_grammar_lags_but_individual_flow_histories(transitions):
    for case in transitions:
        graph, copies, expected = case["graph"], case["copies"], case["expected"]
        hist = graph.graph["history"]
        assert hist["phase_state"][-1] == "stable" and hist["phase_disr"][-1] == 0
        _close(hist["phase_R"][-1], expected["resultant"])
        _close(
            (graph.graph["PHASE_K_GLOBAL"], graph.graph["PHASE_K_LOCAL"]),
            expected["gains"],
        )
        for n, samples in enumerate(case["final"]["temporal"]):
            block = n // copies
            state = read_grammar_state(graph.nodes[n])
            assert state.history == ("AL", "IL", MARKS[block])
            assert state.operator_step == 3 and state.u2_debt == 0
            assert hist["since_AL"][n] == (0 if block == 0 else 1)
            assert hist["since_EN"][n] == (3 if block == 0 else 0 if block == 1 else 1)
            assert len(samples) == (2 if block < 2 else 3)
            _close(samples[-2][1], expected["event"][n])
            assert samples[-2][0] == 1 and samples[-1][0] == 1 + DT
            _close((samples[-1][1] - samples[-2][1]) / DT, expected["rates"][n])


def test_terminal_archive_and_remesh_abstention_are_actual_runtime_results(transitions):
    for case in transitions:
        graph = case["graph"]
        (archive,) = graph.graph["_epi_hist"]
        assert tuple(archive) == tuple(graph)
        _close(tuple(archive.values()), case["expected"]["endpoint"])
        gate = native._event(case, "remesh_gate")
        assert gate["before"]["form"] == gate["after"]["form"]
        assert (
            len(graph.graph["history"].get("stable_frac", ()))
            < graph.graph["REMESH_STABILITY_WINDOW"]
        )
        assert "_last_remesh_step" not in graph.graph


def test_next_fresh_preparation_retains_selection_without_a_second_invocation(
    transitions,
):
    for case in transitions:
        expected, observed = case["expected"], case["next_readout"]
        _close(observed["pressure"], expected["next_pressure"])
        _close(observed["si"], expected["next_si"])
        assert min(observed["si"]) > 1 / 2
        assert case["next_choices"] == tuple(
            mark for mark in NEXT_MARKS for _ in range(case["copies"])
        )
        for channel in (
            "form",
            "phase",
            "capacity",
            "rate",
            "acceleration",
            "temporal",
            "time",
            "ordinal",
        ):
            assert observed[channel] == case["final"][channel]
