"""Prospective mean controls after Transition breaks a native pair symmetry.

Freeze the doubled-C5 support, node order 0..9, seed 17, zero phase,
capacities (1/4, 1, 1, 1, 1), initial forms 1/4, and t=1, dt=1/16.
Predict both complete native calls from these primitives before evaluating
either response. Independent BLAKE2b/Random draws use indices 0 and 1; the
second call continues the live first graph without resetting its state.

Separately supplied boundaries use the first predicted block means rounded
once to a 2**-40 grid, so all represented pair midpoints are exactly equal.
Their contrasts are frozen at zero, +/-1/64, or a single 1/2 sign crossing;
draw index 1, grammar records, lags and capacity counters are supplied equally.
They carry no temporal history and are not certificates of an acquired past.
The absolute binary64 comparison budget is 1e-14, with exact discrete checks.
These bounded controls do not establish a complete-state quotient, unlimited
iteration, autonomous formation or physical identification.
"""

import hashlib
import random
import struct
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.physics import test_native_collective_step as base
from tnfr._runtime_steps import RUNTIME_STEP_NEXT_KEY
from tnfr.constants import DEFAULTS
from tnfr.glyph_history import push_glyph
from tnfr.operators.grammar_debt import OPERATOR_STEP_KEY, read_grammar_state

MEANS = (Q(1, 4),) * 5
CAPACITIES = (Q(1, 4), Q(1), Q(1), Q(1), Q(1))
ZERO = (Q(0),) * 5
RHO = Q(1, 64)
CONTRASTS = (
    ZERO,
    (RHO, *ZERO[1:]),
    (-RHO, *ZERO[1:]),
    (RHO, -RHO, RHO, -RHO, RHO),
    (RHO,) * 5,
    (-RHO,) * 5,
)
MARKS = tuple((node, "NAV" if node < 2 else "IL") for node in range(10))


def _fine(means, contrasts):
    return tuple(value for m, d in zip(means, contrasts) for value in (m - d, m + d))


def _means(values):
    return tuple((Q(values[a]) + Q(values[b])) / 2 for a, b in base.PAIRS)


def _contrasts(values):
    return tuple((Q(values[b]) - Q(values[a])) / 2 for a, b in base.PAIRS)


def _jitter(offset, draw, amplitude):
    """Independent node/draw addressing, without a runtime RNG or response."""

    def digest(left, right):
        payload = struct.pack(">QQ", left & (2**64 - 1), right & (2**64 - 1))
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    seed = digest(digest(17, offset), draw)
    return Q(random.Random(seed).uniform(-float(amplitude), float(amplitude)))


def _forecast(graph, means, contrasts, *, draw, counts, prior_rates):
    """Derive the fine and reduced maps from admitted preparation/configuration."""
    cfg = graph.graph
    k, w = (base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf"))
    pressure_means = tuple(
        k * ((means[(i - 1) % 5] + means[(i + 1) % 5]) / 2 - means[i])
        + w * ((CAPACITIES[(i - 1) % 5] + CAPACITIES[(i + 1) % 5]) / 2 - CAPACITIES[i])
        for i in range(5)
    )
    pressure = tuple(
        value
        for b, d in zip(pressure_means, contrasts)
        for value in (b + k * d, b - k * d)
    )
    peak = max(map(abs, pressure))
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha * CAPACITIES[node // 2] + beta + gamma * (1 - abs(p) / peak)
        for node, p in enumerate(pressure)
    )
    thresholds = cfg["SELECTOR_THRESHOLDS"]
    assert all(Q(thresholds["si_lo"]) < s < Q(thresholds["si_hi"]) for s in si[:2])
    assert min(si[2:]) >= alpha + beta > Q(thresholds["si_hi"])
    assert min(abs(p) / peak for p in pressure[:2]) > Q(thresholds["dnfr_hi"])
    factors = cfg["GLYPH_FACTORS"]
    eta, h = Q(factors["NAV_eta"]), Q(factors["IL_dnfr_factor"])
    c, dt = CAPACITIES[0], Q(base.DT)
    jitter = tuple(_jitter(node, draw, factors["NAV_jitter"]) for node in (0, 1))
    signs = tuple(Q(1 if p >= 0 else -1) for p in pressure[:2])
    held = tuple(
        (1 - eta) * p + eta * c * signs[node] + jitter[node] if node < 2 else h * p
        for node, p in enumerate(pressure)
    )
    assert all(after != before for after, before in zip(held[:2], pressure[:2]))
    rates = tuple(CAPACITIES[node // 2] * p for node, p in enumerate(held))
    forms = _fine(means, contrasts)
    endpoint = tuple(x + dt * rate for x, rate in zip(forms, rates))
    assert min(forms + endpoint) > Q(cfg["EPI_MIN"])
    assert max(forms + endpoint) < Q(cfg["EPI_MAX"])
    acceleration = tuple((rate - Q(old)) / dt for rate, old in zip(rates, prior_rates))
    next_counts = tuple(
        (
            count + 1
            if abs(p) <= Q(cfg["EPS_DNFR_STABLE"]) and s >= Q(thresholds["si_hi"])
            else 0
        )
        for count, p, s in zip(counts, held, si)
    )
    assert max(next_counts) < cfg["VF_ADAPT_TAU"]
    mean_endpoint = (
        means[0]
        + dt
        * c
        * ((1 - eta) * pressure_means[0] + eta * c * sum(signs) / 2 + sum(jitter) / 2),
    ) + tuple(
        means[i] + dt * CAPACITIES[i] * h * pressure_means[i] for i in range(1, 5)
    )
    contrast_endpoint = (
        (1 - dt * c * (1 - eta) * k) * contrasts[0]
        + dt * c * (eta * c * (signs[1] - signs[0]) + jitter[1] - jitter[0]) / 2,
    ) + tuple((1 - dt * CAPACITIES[i] * h * k) * contrasts[i] for i in range(1, 5))
    assert _means(endpoint) == mean_endpoint
    assert _contrasts(endpoint) == contrast_endpoint
    return {
        "initial_pressure": pressure,
        "pressure_means": pressure_means,
        "pressure": held,
        "form": endpoint,
        "mean": mean_endpoint,
        "contrast": contrast_endpoint,
        "rate": rates,
        "acceleration": acceleration,
        "si": si,
        "counts": next_counts,
        "jitter": jitter,
        "signs": signs,
    }


def _source(means=MEANS, contrasts=ZERO, *, boundary=None):
    graph = base._source(tuple(map(float, means)), base.PROFILES[0])
    for node, form in enumerate(_fine(means, contrasts)):
        graph.nodes[node].update(EPI=float(form), nu_f=float(CAPACITIES[node // 2]))
    if boundary is not None:
        graph.graph.update(_t=1 + base.DT, **{RUNTIME_STEP_NEXT_KEY: 1})
        for node, count in enumerate(boundary["counts"]):
            data = graph.nodes[node]
            # Supplied execution metadata; no temporal samples or cached fields.
            push_glyph(data, "NAV" if node < 2 else "IL", 4)
            data["stable_count"] = count
            graph.graph["history"]["since_AL"][node] = 1
            graph.graph["history"]["since_EN"][node] = 1
            if node < 2:
                data["_rng_jitter_progress"] = {"seed": 17, "offset": node, "draws": 1}
    return graph


def _freeze_predictions():
    graph = _source()
    first = _forecast(
        graph, MEANS, ZERO, draw=0, counts=(0,) * 10, prior_rates=base._lift(base.RATES)
    )
    second = _forecast(
        graph,
        first["mean"],
        first["contrast"],
        draw=1,
        counts=first["counts"],
        prior_rates=first["rate"],
    )
    assert max(map(abs, first["contrast"])) < RHO
    assert first["signs"] == second["signs"] == (Q(1), Q(1))
    return graph, first, second


def _capture(graph, *, scalar):
    case = base._execute(graph, scalar=scalar)
    case["records"] = tuple(
        (read_grammar_state(data), deepcopy(data.get("_rng_jitter_progress")))
        for _, data in graph.nodes(data=True)
    )
    case["history"] = deepcopy(graph.graph["history"])
    case["remesh_history"] = deepcopy(graph.graph["_epi_hist"])
    return case


@pytest.fixture(scope="module", params=(False, True), ids=("numpy", "scalar"))
def continuation(request):
    graph, first, second = _freeze_predictions()
    # Both predictions are complete before the first reserved native response.
    cases = tuple(_capture(graph, scalar=request.param) for _ in range(2))
    return cases, (first, second)


def _assert_response(case, expected):
    assert case["marks"] == MARKS
    for owner, channel, predicted in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
        ("glyphs", "pressure", "pressure"),
    ):
        assert base._event(case, owner)["after"][channel] == pytest.approx(
            tuple(map(float, expected[predicted])), abs=base.ABS_ERROR, rel=0
        )
    for channel in ("form", "pressure", "rate", "acceleration", "si"):
        assert case["final"][channel] == pytest.approx(
            tuple(map(float, expected[channel])), abs=base.ABS_ERROR, rel=0
        )
    for channel, project in (("mean", _means), ("contrast", _contrasts)):
        assert tuple(map(float, project(case["final"]["form"]))) == pytest.approx(
            tuple(map(float, expected[channel])), abs=base.ABS_ERROR, rel=0
        )
    assert case["final"]["counters"] == expected["counts"]


@pytest.mark.parametrize("step", (0, 1))
def test_prospective_complete_calls_match_fine_and_collective_predictions(
    continuation, step
):
    cases, expected = continuation
    _assert_response(cases[step], expected[step])


@pytest.mark.parametrize("step", (0, 1))
def test_projection_retains_actual_runtime_and_adaptation_boundaries(
    continuation, step
):
    case = continuation[0][step]
    cfg = case["configuration"]
    for key in (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "GLYPH_FACTORS",
        "SELECTOR_THRESHOLDS",
        "NAV_RANDOM",
        "NAV_STRICT",
        "VF_ADAPT_TAU",
        "VF_ADAPT_MU",
        "DT_MIN",
    ):
        assert cfg[key] == DEFAULTS[key]
    assert cfg["GRAMMAR_REJECTION_MODE"] == "raise"
    assert "integrator" not in cfg and "glyph_selector" not in cfg
    assert [event["name"] for event in case["events"]] == [
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
    assert case["final"]["phase"] == (0.0,) * 10
    assert case["final"]["capacity"] == base._lift(CAPACITIES)
    assert case["final"]["edges"] == case["initial"]["edges"]
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )
    assert not case["history"].get("stable_frac")
    assert len(case["remesh_history"]) == step + 1
    assert case["final"]["ordinal"] == step + 1
    assert case["final"]["time"] == 1 + (step + 1) * base.DT
    for node, (grammar, progress) in enumerate(case["records"]):
        assert grammar.operator_step == 4 + step
        assert grammar.prior_coherence is True and grammar.u2_debt == 0
        assert grammar.history == ("EN",) + (MARKS[node][1],) * (step + 1)
        assert case["history"]["since_AL"][node] == step + 1
        assert case["history"]["since_EN"][node] == step + 1
        assert progress == (
            {"seed": 17, "offset": node, "draws": step + 1} if node < 2 else None
        )


def test_actual_continuation_keeps_rng_and_temporal_evidence_without_reconstruction(
    continuation,
):
    (first, second), expected = continuation
    assert first["graph"] is second["graph"]
    assert first["final"] == second["initial"]
    assert first["records"][0][1]["draws"] == 1
    assert second["records"][0][1]["draws"] == 2
    assert expected[0]["jitter"] != expected[1]["jitter"]
    for node in range(10):
        assert second["final"]["temporal"][node][-2:] == (
            (1 + base.DT, first["final"]["form"][node]),
            (1 + 2 * base.DT, second["final"]["form"][node]),
        )
    assert first["final"]["form"][0] != first["final"]["form"][1]
    assert second["final"]["form"][0] != second["final"]["form"][1]


@pytest.fixture(scope="module", params=(False, True), ids=("numpy", "scalar"))
def supplied_boundaries(request):
    _, first, _ = _freeze_predictions()
    means = tuple(Q(round(value * 2**40), 2**40) for value in first["mean"])
    contrasts = (*CONTRASTS, (Q(1, 2), *ZERO[1:]))
    graphs = tuple(_source(means, d, boundary=first) for d in contrasts)
    expected = tuple(
        _forecast(
            graph,
            means,
            d,
            draw=1,
            counts=first["counts"],
            prior_rates=base._lift(base.RATES),
        )
        for graph, d in zip(graphs, contrasts)
    )
    # Freeze every boundary prediction and guard before evaluating any response.
    k = base._weight(graphs[0], "DNFR_WEIGHTS", "epi")
    b = expected[0]["pressure_means"]
    lower, upper = b[0] - k * RHO, max(map(abs, b)) + k * RHO
    assert lower > Q(1, 25)
    assert upper < Q(3, 64)
    assert lower / upper > Q(4, 5)
    factors = graphs[0].graph["GLYPH_FACTORS"]
    eta, h, jitter_bound = (
        Q(factors[key]) for key in ("NAV_eta", "IL_dnfr_factor", "NAV_jitter")
    )
    c, dt = CAPACITIES[0], Q(base.DT)
    assert eta * (c - upper) - jitter_bound > 0
    rate_bound = max(
        c * ((1 - eta) * upper + eta * c + jitter_bound),
        max(CAPACITIES) * h * upper,
    )
    assert min(means) - RHO - dt * rate_bound > Q(1, 5)
    assert max(means) + RHO + dt * rate_bound < Q(3, 10)
    assert max(first["counts"]) <= 1
    assert max(first["counts"]) + 1 < graphs[0].graph["VF_ADAPT_TAU"]
    assert all(item["mean"] == expected[0]["mean"] for item in expected[:-1])
    assert expected[-1]["signs"] == (Q(1), Q(-1))
    assert expected[-1]["mean"][0] - expected[0]["mean"][0] == -Q(1, 512)
    for graph in graphs:
        assert _means(tuple(graph.nodes[node]["EPI"] for node in graph)) == means
        assert all("epi_time_history" not in data for _, data in graph.nodes(data=True))
        assert all(data[OPERATOR_STEP_KEY] == 4 for _, data in graph.nodes(data=True))
    cases = tuple(_capture(graph, scalar=request.param) for graph in graphs)
    return cases, expected


@pytest.mark.parametrize("variant", range(len(CONTRASTS)))
def test_supplied_internal_contrasts_share_the_predicted_next_means(
    supplied_boundaries, variant
):
    cases, expected = supplied_boundaries
    case = cases[variant]
    _assert_response(case, expected[variant])
    assert expected[variant]["mean"] == expected[0]["mean"]
    assert case["final"]["capacity"] == base._lift(CAPACITIES)
    assert case["final"]["phase"] == (0.0,) * 10
    assert not any(
        event["name"] in {"capacity_write", "network_remesh"}
        for event in case["events"]
    )
    if variant:
        assert case["final"]["form"] != cases[0]["final"]["form"]
        assert case["final"]["si"] != cases[0]["final"]["si"]


def test_same_mean_projection_does_not_identify_retained_adaptation_memory(
    supplied_boundaries,
):
    cases, expected = supplied_boundaries
    assert expected[4]["mean"] == expected[0]["mean"]
    assert cases[4]["final"]["counters"] != cases[0]["final"]["counters"]
    assert cases[4]["final"]["capacity"] == cases[0]["final"]["capacity"]


def test_same_selected_word_fails_mean_closure_after_internal_pressure_sign_crossing(
    supplied_boundaries,
):
    cases, expected = supplied_boundaries
    reference, crossed = cases[0], cases[-1]
    _assert_response(crossed, expected[-1])
    assert crossed["marks"] == reference["marks"] == MARKS
    assert crossed["records"] == reference["records"]
    assert _means(crossed["initial"]["form"]) == _means(reference["initial"]["form"])
    observed_gap = (
        _means(crossed["final"]["form"])[0] - _means(reference["final"]["form"])[0]
    )
    assert float(observed_gap) == pytest.approx(-1 / 512, abs=base.ABS_ERROR, rel=0)
    assert _means(crossed["final"]["form"])[1:] == pytest.approx(
        tuple(map(float, _means(reference["final"]["form"])[1:])),
        abs=base.ABS_ERROR,
        rel=0,
    )
    assert crossed["final"]["phase"] == reference["final"]["phase"]
    assert crossed["final"]["capacity"] == reference["final"]["capacity"]
