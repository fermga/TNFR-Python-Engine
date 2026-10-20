"""A reduced mean envelope reaches the first lag-forced Reception call.

Freeze the preceding doubled-C5 source, contiguous order, seed 17, t=1 and
dt=1/16. Predict all four calls before any native response: Transition (NAV)
on the first pair and Coherence (IL) elsewhere for calls 1..3, then Reception
(EN) everywhere when the retained reception lag reaches four. Independent
rational recurrences bound all jitter inputs in the configured amplitude;
fine forecasts separately reconstruct the actual target-associated draws.

Three supplied boundaries retain the first predicted means rounded once to
a 2**-40 grid, with contrasts zero or +/-(rho,-rho,rho,-rho,rho), rho=1/64.
They supply equal admission metadata and draw index one, without inventing
temporal evidence. All three subsequent predictions precede their responses.
The absolute binary64 budget remains 1e-14; discrete observations are exact.
This finite four-call scope does not establish a complete collective state,
indefinite closure, formation or a physical identification.
"""

from fractions import Fraction as Q

import pytest

from tests.physics import test_native_collective_mean as mean
from tests.physics import test_native_collective_step as base
from tnfr.constants import DEFAULTS

RECEPTION = tuple((node, "EN") for node in range(10))
ALTERNATING = tuple(mean.RHO * (-1) ** block for block in range(5))
BOUNDARY_CONTRASTS = (mean.ZERO, ALTERNATING, tuple(-d for d in ALTERNATING))


def _pressure(graph, means, contrasts):
    k, b = (base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf"))
    v = mean.CAPACITIES
    centers = tuple(
        k * ((means[(i - 1) % 5] + means[(i + 1) % 5]) / 2 - means[i])
        + b * ((v[(i - 1) % 5] + v[(i + 1) % 5]) / 2 - v[i])
        for i in range(5)
    )
    values = tuple(
        value for p, d in zip(centers, contrasts) for value in (p + k * d, p - k * d)
    )
    return centers, values


def _ordered_reception(values, mix):
    current = list(values)
    for i in range(5):
        current[i] = (1 - mix) * current[i] + mix * (
            current[(i - 1) % 5] + current[(i + 1) % 5]
        ) / 2
    return tuple(current)


def _reception_forecast(graph, means, contrasts, *, counts, prior_rates):
    """Ordered form jump followed by Euler with the untouched pre-jump pressure."""
    cfg = graph.graph
    k = base._weight(graph, "DNFR_WEIGHTS", "epi")
    centers, pressure = _pressure(graph, means, contrasts)
    peak = max(map(abs, pressure))
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha * mean.CAPACITIES[node // 2] + beta + gamma * (1 - abs(p) / peak)
        for node, p in enumerate(pressure)
    )
    mix, dt = Q(cfg["GLYPH_FACTORS"]["EN_mix"]), Q(base.DT)
    jumped_means = _ordered_reception(means, mix)
    jumped_contrasts = tuple((1 - mix) * d for d in contrasts)
    jumped = mean._fine(jumped_means, jumped_contrasts)
    rates = tuple(mean.CAPACITIES[node // 2] * p for node, p in enumerate(pressure))
    endpoint = tuple(x + dt * rate for x, rate in zip(jumped, rates))
    mean_endpoint = tuple(
        x + dt * v * b for x, v, b in zip(jumped_means, mean.CAPACITIES, centers)
    )
    contrast_endpoint = tuple(
        (1 - mix - dt * v * k) * d for d, v in zip(contrasts, mean.CAPACITIES)
    )
    assert mean._means(endpoint) == mean_endpoint
    assert mean._contrasts(endpoint) == contrast_endpoint
    assert min(jumped + endpoint) > Q(cfg["EPI_MIN"])
    assert max(jumped + endpoint) < Q(cfg["EPI_MAX"])
    next_counts = tuple(
        (
            count + 1
            if abs(p) <= Q(cfg["EPS_DNFR_STABLE"])
            and s >= Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
            else 0
        )
        for count, p, s in zip(counts, pressure, si)
    )
    assert max(next_counts) < cfg["VF_ADAPT_TAU"]
    simultaneous = tuple(
        (1 - mix) * x + mix * (means[(i - 1) % 5] + means[(i + 1) % 5]) / 2
        for i, x in enumerate(means)
    )
    _, refreshed = _pressure(graph, jumped_means, jumped_contrasts)
    refreshed_endpoint = tuple(
        x + dt * mean.CAPACITIES[node // 2] * p
        for node, (x, p) in enumerate(zip(jumped, refreshed))
    )
    product_contrasts = tuple(
        (1 - mix) * (1 - dt * v * k) * d for d, v in zip(contrasts, mean.CAPACITIES)
    )
    return {
        "initial_pressure": pressure,
        "pressure_means": centers,
        "pressure": pressure,
        "post_event": jumped,
        "form": endpoint,
        "mean": mean_endpoint,
        "contrast": contrast_endpoint,
        "rate": rates,
        "acceleration": tuple((r - Q(old)) / dt for r, old in zip(rates, prior_rates)),
        "si": si,
        "counts": next_counts,
        "alternatives": {
            "simultaneous_means": simultaneous,
            "refreshed_endpoint": refreshed_endpoint,
            "product_contrasts": product_contrasts,
        },
    }


def _envelope(graph, lower, upper, radii, *, reception):
    """Propagate an interval without consuming actual contrasts or sampled jitter."""
    cfg, v, dt = graph.graph, mean.CAPACITIES, Q(base.DT)
    k, b = (base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf"))
    eta, h, jitter, mix = (
        Q(cfg["GLYPH_FACTORS"][name])
        for name in ("NAV_eta", "IL_dnfr_factor", "NAV_jitter", "EN_mix")
    )
    capacity_pressure = tuple(
        b * ((v[(i - 1) % 5] + v[(i + 1) % 5]) / 2 - v[i]) for i in range(5)
    )
    pressure_lower = tuple(
        k * ((lower[(i - 1) % 5] + lower[(i + 1) % 5]) / 2 - upper[i]) + c
        for i, c in enumerate(capacity_pressure)
    )
    pressure_upper = tuple(
        k * ((upper[(i - 1) % 5] + upper[(i + 1) % 5]) / 2 - lower[i]) + c
        for i, c in enumerate(capacity_pressure)
    )
    sign_margin = pressure_lower[0] - k * radii[0]
    pressure_bound = max(
        max(abs(lo), abs(hi)) + k * radius
        for lo, hi, radius in zip(pressure_lower, pressure_upper, radii)
    )
    assert sign_margin > Q(3, 80)
    assert pressure_bound < Q(3, 64)
    assert sign_margin / pressure_bound > Q(4, 5)
    assert eta * (v[0] - pressure_bound) - jitter > 0
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    thresholds = cfg["SELECTOR_THRESHOLDS"]
    assert Q(thresholds["si_lo"]) < alpha * v[0] + beta
    assert alpha * v[0] + beta + gamma < Q(thresholds["si_hi"])
    assert alpha + beta > Q(thresholds["si_hi"])
    assert sign_margin / pressure_bound > Q(thresholds["dnfr_hi"])
    if reception:
        jumped_lower = _ordered_reception(lower, mix)
        jumped_upper = _ordered_reception(upper, mix)
        jumped_radii = tuple((1 - mix) * r for r in radii)
        next_lower = tuple(
            x + dt * capacity * p
            for x, capacity, p in zip(jumped_lower, v, pressure_lower)
        )
        next_upper = tuple(
            x + dt * capacity * p
            for x, capacity, p in zip(jumped_upper, v, pressure_upper)
        )
        next_radii = tuple(
            abs(1 - mix - dt * capacity * k) * r for capacity, r in zip(v, radii)
        )
    else:
        steps = (dt * v[0] * (1 - eta),) + tuple(
            dt * capacity * h for capacity in v[1:]
        )
        factors = tuple(1 - step * k for step in steps)
        assert min(factors) > 0

        def propagate(values, innovation):
            return tuple(
                factors[i] * values[i]
                + step * k * (values[(i - 1) % 5] + values[(i + 1) % 5]) / 2
                + step * capacity_pressure[i]
                + (dt * v[0] * (eta * v[0] + innovation) if i == 0 else 0)
                for i, step in enumerate(steps)
            )

        next_lower, next_upper = propagate(lower, -jitter), propagate(upper, jitter)
        next_radii = tuple(
            factor * r + (dt * v[0] * jitter if i == 0 else 0)
            for i, (factor, r) in enumerate(zip(factors, radii))
        )
        jumped_lower, jumped_upper, jumped_radii = lower, upper, radii
    for lo, hi, radius in zip(jumped_lower, jumped_upper, jumped_radii):
        assert lo - radius > Q(1, 5) and hi + radius < Q(3, 10)
    for lo, hi, radius in zip(next_lower, next_upper, next_radii):
        assert lo - radius > Q(1, 5) and hi + radius < Q(3, 10)
    return {
        "lower": next_lower,
        "upper": next_upper,
        "radii": next_radii,
        "sign_margin": sign_margin,
        "pressure_bound": pressure_bound,
    }


def _freeze(graph, means, contrasts, *, start, counts, prior_rates, radii):
    expected, envelopes = [], []
    lower = upper = means
    for call in range(start, 5):
        envelope = _envelope(graph, lower, upper, radii, reception=call == 4)
        if call == 4:
            prediction = _reception_forecast(
                graph, means, contrasts, counts=counts, prior_rates=prior_rates
            )
        else:
            prediction = mean._forecast(
                graph,
                means,
                contrasts,
                draw=call - 1,
                counts=counts,
                prior_rates=prior_rates,
            )
            prediction["post_event"] = mean._fine(means, contrasts)
            assert prediction["signs"] == (Q(1), Q(1))
        expected.append(prediction)
        envelopes.append(envelope)
        means, contrasts = prediction["mean"], prediction["contrast"]
        counts, prior_rates = prediction["counts"], prediction["rate"]
        lower, upper, radii = envelope["lower"], envelope["upper"], envelope["radii"]
        assert max(counts) <= call < graph.graph["VF_ADAPT_TAU"]
        assert all(lo <= m <= hi for lo, m, hi in zip(lower, means, upper))
        assert all(abs(d) <= radius for d, radius in zip(contrasts, radii))
    return tuple(expected), tuple(envelopes)


@pytest.fixture(scope="module", params=(False, True), ids=("numpy", "scalar"))
def continuation(request):
    graph = mean._source()
    expected, envelopes = _freeze(
        graph,
        mean.MEANS,
        mean.ZERO,
        start=1,
        counts=(0,) * 10,
        prior_rates=base._lift(base.RATES),
        radii=mean.ZERO,
    )
    assert expected[2]["mean"][2] == expected[2]["mean"][3]
    simultaneous = expected[3]["alternatives"]["simultaneous_means"]
    assert simultaneous[2] == simultaneous[3]
    assert expected[3]["mean"][2] - expected[3]["mean"][3] > Q(1, 8192)
    # All four forecasts and enclosure admissions are fixed before evaluation.
    cases = tuple(mean._capture(graph, scalar=request.param) for _ in range(4))
    return cases, expected, envelopes


@pytest.fixture(scope="module", params=(False, True), ids=("numpy", "scalar"))
def boundaries(request):
    graph = mean._source()
    first = mean._forecast(
        graph,
        mean.MEANS,
        mean.ZERO,
        draw=0,
        counts=(0,) * 10,
        prior_rates=base._lift(base.RATES),
    )
    centers = tuple(Q(round(value * 2**40), 2**40) for value in first["mean"])
    graphs = tuple(mean._source(centers, d, boundary=first) for d in BOUNDARY_CONTRASTS)
    forecasts = tuple(
        _freeze(
            item,
            centers,
            d,
            start=2,
            counts=first["counts"],
            prior_rates=base._lift(base.RATES),
            radii=(mean.RHO,) * 5,
        )
        for item, d in zip(graphs, BOUNDARY_CONTRASTS)
    )
    assert all(item[1] == forecasts[0][1] for item in forecasts)
    for call in range(3):
        assert all(
            item[0][call]["mean"] == forecasts[0][0][call]["mean"] for item in forecasts
        )
    for item in graphs:
        assert mean._means(tuple(item.nodes[node]["EPI"] for node in item)) == centers
        assert all("epi_time_history" not in data for _, data in item.nodes(data=True))
    # Freeze every supplied-boundary prediction before any boundary response.
    cases = tuple(
        tuple(mean._capture(item, scalar=request.param) for _ in range(3))
        for item in graphs
    )
    return cases, forecasts


def _assert_response(case, prediction, envelope, *, call):
    assert case["marks"] == (RECEPTION if call == 4 else mean.MARKS)
    for owner, channel, predicted in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
        ("glyphs", "pressure", "pressure"),
        ("glyphs", "form", "post_event"),
    ):
        assert base._event(case, owner)["after"][channel] == pytest.approx(
            tuple(map(float, prediction[predicted])), abs=base.ABS_ERROR, rel=0
        )
    for channel in ("form", "pressure", "rate", "acceleration", "si"):
        assert case["final"][channel] == pytest.approx(
            tuple(map(float, prediction[channel])), abs=base.ABS_ERROR, rel=0
        )
    for channel, projection in (("mean", mean._means), ("contrast", mean._contrasts)):
        assert tuple(map(float, projection(case["final"]["form"]))) == pytest.approx(
            tuple(map(float, prediction[channel])), abs=base.ABS_ERROR, rel=0
        )
    for value, lo, hi in zip(
        mean._means(case["final"]["form"]), envelope["lower"], envelope["upper"]
    ):
        assert float(lo) - base.ABS_ERROR <= value <= float(hi) + base.ABS_ERROR
    for value, radius in zip(mean._contrasts(case["final"]["form"]), envelope["radii"]):
        assert abs(value) <= float(radius) + base.ABS_ERROR
    assert case["final"]["counters"] == prediction["counts"]
    assert case["final"]["phase"] == (0.0,) * 10
    assert case["final"]["capacity"] == base._lift(mean.CAPACITIES)


@pytest.mark.parametrize("call", range(1, 5))
def test_complete_calls_match_independent_predictions_and_propagated_envelope(
    continuation, call
):
    cases, expected, envelopes = continuation
    _assert_response(
        cases[call - 1], expected[call - 1], envelopes[call - 1], call=call
    )


@pytest.mark.parametrize("call", range(1, 5))
def test_actual_lags_rng_history_and_runtime_policies_continue_to_reception(
    continuation, call
):
    cases, _, _ = continuation
    case, cfg = cases[call - 1], cases[call - 1]["configuration"]
    for key in (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "GLYPH_FACTORS",
        "SELECTOR_THRESHOLDS",
        "AL_MAX_LAG",
        "EN_MAX_LAG",
        "NAV_RANDOM",
        "NAV_STRICT",
        "VF_ADAPT_TAU",
        "VF_ADAPT_MU",
        "DT_MIN",
    ):
        assert cfg[key] == DEFAULTS[key]
    assert cfg["GRAMMAR_REJECTION_MODE"] == "raise"
    assert "glyph_selector" not in cfg and "integrator" not in cfg
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
    assert case["final"]["edges"] == case["initial"]["edges"]
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )
    assert not case["history"].get("stable_frac")
    assert len(case["remesh_history"]) == call
    assert case["final"]["ordinal"] == call
    assert case["final"]["time"] == 1 + call * base.DT
    for node, (grammar, progress) in enumerate(case["records"]):
        assert grammar.operator_step == 3 + call
        assert grammar.prior_coherence is True and grammar.u2_debt == 0
        expected_history = ("EN",) + (mean.MARKS[node][1],) * min(call, 3)
        if call == 4:
            expected_history += ("EN",)
        assert grammar.history == expected_history[-cfg["GLYPH_HYSTERESIS_WINDOW"] :]
        assert case["history"]["since_AL"][node] == call
        assert case["history"]["since_EN"][node] == (0 if call == 4 else call)
        assert progress == (
            {"seed": 17, "offset": node, "draws": min(call, 3)} if node < 2 else None
        )
    if call > 1:
        assert cases[call - 2]["graph"] is case["graph"]
        assert cases[call - 2]["final"] == case["initial"]


def test_reception_uses_live_neighbor_order_and_pre_jump_pressure(continuation):
    cases, expected, _ = continuation
    case, prediction = cases[-1], expected[-1]
    alternatives = prediction["alternatives"]
    correct_jump = mean._means(prediction["post_event"])
    assert max(
        abs(a - b) for a, b in zip(alternatives["simultaneous_means"], correct_jump)
    ) > Q(1, 1000000)
    assert max(
        abs(a - b)
        for a, b in zip(alternatives["refreshed_endpoint"], prediction["form"])
    ) > Q(1, 1000000)
    assert abs(alternatives["product_contrasts"][0] - prediction["contrast"][0]) > Q(
        1, 10**9
    )
    for node in range(10):
        assert case["final"]["temporal"][node] == (
            (1 + 3 * base.DT, base._event(case, "glyphs")["after"]["form"][node]),
            (1 + 4 * base.DT, case["final"]["form"][node]),
        )
    assert case["records"][0][1] == cases[-2]["records"][0][1]
    assert (
        base._event(case, "glyphs")["after"]["pressure"]
        == base._event(case, "pressure")["after"]["pressure"]
    )


@pytest.mark.parametrize("variant", range(3))
@pytest.mark.parametrize("call", range(2, 5))
def test_distinct_supplied_lifts_share_three_successive_mean_predictions(
    boundaries, variant, call
):
    cases, forecasts = boundaries
    prediction, envelope = (
        forecasts[variant][0][call - 2],
        forecasts[variant][1][call - 2],
    )
    case = cases[variant][call - 2]
    _assert_response(case, prediction, envelope, call=call)
    assert prediction["mean"] == forecasts[0][0][call - 2]["mean"]
    assert not any(
        event["name"] in {"capacity_write", "network_remesh"}
        for event in case["events"]
    )
    if variant:
        reference = cases[0][call - 2]
        assert case["final"]["form"] != reference["final"]["form"]
        assert case["final"]["temporal"] != reference["final"]["temporal"]
        assert case["final"]["counters"] != reference["final"]["counters"]
    assert case["final"]["ordinal"] == call
    assert case["final"]["time"] == 1 + call * base.DT
    if call > 2:
        assert cases[variant][call - 3]["final"] == case["initial"]


def test_fixed_contrast_box_is_not_an_invariant_of_admitted_transition_inputs():
    graph = mean._source()
    k = base._weight(graph, "DNFR_WEIGHTS", "epi")
    cfg, dt, c = graph.graph, Q(base.DT), mean.CAPACITIES[0]
    eta = Q(cfg["GLYPH_FACTORS"]["NAV_eta"])
    jitter = Q(cfg["GLYPH_FACTORS"]["NAV_jitter"])
    # An admitted boundary lift with d0=rho and opposite endpoint jitters
    # xi0=-J, xi1=J has zero mean innovation and this exact outgoing contrast.
    outgoing = (1 - dt * c * (1 - eta) * k) * mean.RHO + dt * c * jitter
    assert outgoing > mean.RHO
