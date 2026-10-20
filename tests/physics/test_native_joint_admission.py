"""Joint form/capacity bounds distinguish native admission from uncertainty.

Freeze the doubled-C5 support, order 0..9, zero phase, seed 17, t=1,
dt=1/16 and all default policies. Capacity has a first-pair 1/1024 pulse.
The positive class has M0=1/8+57/8192, other means 1/8, form radii 2**-20,
zero capacity radii, count 3 at node 2 and count 4 elsewhere. Two opposite
alternating contrasts continue for two calls: the first pair fails then
passes the inclusive held-pressure stability test, while node 2 becomes due.

The ambiguous class has all means 1/8, first-pair form radius 59/8192,
zero other radii, count 0 at node 0 and count 4 elsewhere. Opposite contrasts
produce different node-1 capacity writes in one call despite identical coarse
input and retained context. Unresolved admission retains both continuations.

All coarse enclosures, fine predictions and guards are computed before native
responses. The shared outward rational interval owner bounds the exact pair
identities; binary64 comparisons separately use the frozen absolute 1e-14
budget. These supplied finite classes do not establish an autonomous selector,
formed collective identity, unlimited closure or a generic predictor API.
"""

from fractions import Fraction as Q
from itertools import product

import pytest

from tests.physics import test_native_capacity_feedback as feedback
from tests.physics import test_native_collective_mean as mean
from tests.physics import test_native_collective_step as base
from tnfr.constants import DEFAULTS
from tnfr.mathematics._rational_interval import I

RHO = Q(1, 2**20)
POSITIVE_MEANS = (Q(1, 8) + Q(57, 8192), *feedback.MEANS[1:])
POSITIVE_RADII = (RHO,) * 5
AMBIGUOUS_RADII = (Q(59, 8192), *feedback.ZERO[1:])
COUNTS = (4, 4, 3, 4, 4, 4, 4, 4, 4, 4)
AMBIGUOUS_COUNTS = (0, *(4,) * 9)
NAMES = ("positive_plus", "positive_minus", "ambiguous_plus", "ambiguous_minus")


def _source(name):
    positive, sign = name.startswith("positive"), 1 if name.endswith("plus") else -1
    centers = POSITIVE_MEANS if positive else feedback.MEANS
    radii = POSITIVE_RADII if positive else AMBIGUOUS_RADII
    contrasts = tuple(sign * (-1) ** i * radius for i, radius in enumerate(radii))
    graph = base._source(tuple(map(float, centers)), base.PROFILES[0], ready=True)
    counts = COUNTS if positive else AMBIGUOUS_COUNTS
    for node, form in enumerate(mean._fine(centers, contrasts)):
        graph.nodes[node].update(
            EPI=float(form),
            nu_f=float(feedback.CAPACITIES[node // 2]),
            stable_count=counts[node],
        )
    return graph, {
        "means": tuple(map(I, centers)),
        "capacities": tuple(map(I, feedback.CAPACITIES)),
        "form_radii": radii,
        "capacity_radii": feedback.ZERO,
        "counts": tuple(frozenset((count,)) for count in counts),
    }


def _classify_pressure(pressure, epsilon):
    """The inclusive native predicate has three epistemic outcomes."""
    magnitude = abs(pressure)
    if magnitude.hi <= epsilon:
        return "pass"
    if magnitude.lo > epsilon:
        return "fail"
    return "undetermined"


def _coarse_step(graph, state):
    """Enclose Coherence without fine contrasts or resolved counter branches."""
    cfg = graph.graph
    m, v = state["means"], state["capacities"]
    rx, rv = state["form_radii"], state["capacity_radii"]
    k, b = (base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf"))
    h, mu, dt = (
        Q(cfg["GLYPH_FACTORS"]["IL_dnfr_factor"]),
        Q(cfg["VF_ADAPT_MU"]),
        Q(base.DT),
    )
    assert 0 <= mu <= 1
    alpha, beta = (base._weight(graph, "SI_WEIGHTS", key) for key in ("alpha", "beta"))
    minimum = min(value.lo - radius for value, radius in zip(v, rv))
    maximum = max(value.hi + radius for value, radius in zip(v, rv))
    assert 0 < minimum <= maximum
    # Zero phase and equal degree give these two channels exactly. The pressure
    # channel is nonnegative, so this bound admits all IL and Si stability gates.
    assert alpha * minimum / maximum + beta > Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
    neighbor = feedback._neighbor_mean
    centers = tuple(
        k * (neighbor(m, i) - m[i]) + b * (neighbor(v, i) - v[i]) for i in range(5)
    )
    pressure = tuple(
        h * (center + I(-k * x - b * y, k * x + b * y))
        for center, x, y in zip(centers, rx, rv)
    )
    statuses = tuple(
        _classify_pressure(value, Q(cfg["EPS_DNFR_STABLE"])) for value in pressure
    )
    counts, writes = [], []
    for node, possibilities in enumerate(state["counts"]):
        status = statuses[node // 2]
        successors = set()
        if status != "pass":
            successors.add(0)
        if status != "fail":
            successors.update(count + 1 for count in possibilities)
        counts.append(frozenset(successors))
        writes.append(frozenset(count >= cfg["VF_ADAPT_TAU"] for count in successors))

    next_m, next_v, next_rx, next_rv = [], [], [], []
    for i, (center, capacity, form_radius, capacity_radius, p) in enumerate(
        zip(m, v, rx, rv, centers)
    ):
        # E[d*e] is not set to zero: the unknown association and e**2 enter
        # the exact pair-mean identity with different signed bounds.
        covariance = k * form_radius * capacity_radius
        correction = I(-covariance, covariance + b * capacity_radius**2)
        next_m.append(center + dt * h * (capacity * p - correction))
        next_rx.append(
            (1 - dt * h * k * capacity).abs_max * form_radius
            + dt * h * (p - b * capacity).abs_max * capacity_radius
        )
        contrast = neighbor(v, i) - capacity
        capacity_options, radius_options = [], []
        for even, odd in product(writes[2 * i], writes[2 * i + 1]):
            average, difference = Q(int(even) + int(odd), 2), Q(int(odd) - int(even), 2)
            correction_radius = mu * abs(difference) * capacity_radius
            capacity_options.append(
                capacity
                + mu * average * contrast
                + I(-correction_radius, correction_radius)
            )
            radius_options.append(
                abs(1 - mu * average) * capacity_radius
                + mu * abs(difference) * contrast.abs_max
            )
        result = capacity_options[0]
        for option in capacity_options[1:]:
            result = result.hull(option)
        next_v.append(result)
        next_rv.append(max(radius_options))
    for centers_, radii_, low, high in (
        (m, rx, cfg["EPI_MIN"], cfg["EPI_MAX"]),
        (next_m, next_rx, cfg["EPI_MIN"], cfg["EPI_MAX"]),
        (next_v, next_rv, cfg["VF_MIN"], cfg["VF_MAX"]),
    ):
        assert all(
            Q(low) < value.lo - radius and value.hi + radius < Q(high)
            for value, radius in zip(centers_, radii_)
        )
    return {
        "means": tuple(next_m),
        "capacities": tuple(next_v),
        "form_radii": tuple(next_rx),
        "capacity_radii": tuple(next_rv),
        "counts": tuple(counts),
        "writes": tuple(writes),
        "pressure": pressure,
        "statuses": statuses,
    }


def _fine_step(graph, forms, capacities, counts, previous_rates):
    """Independent exact fine oracle with actual inclusive qualification/reset."""
    cfg = graph.graph
    forms, capacities = tuple(map(Q, forms)), tuple(map(Q, capacities))
    m, v = mean._means(forms), mean._means(capacities)
    k, b = (base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf"))
    h, mu, dt = (
        Q(cfg["GLYPH_FACTORS"]["IL_dnfr_factor"]),
        Q(cfg["VF_ADAPT_MU"]),
        Q(base.DT),
    )
    pressure = tuple(
        k * (feedback._neighbor_mean(m, node // 2) - x)
        + b * (feedback._neighbor_mean(v, node // 2) - capacity)
        for node, (x, capacity) in enumerate(zip(forms, capacities))
    )
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", key) for key in ("alpha", "beta", "gamma")
    )
    peak = max(map(abs, pressure))
    assert peak > 0 and min(capacities) > 0
    si = tuple(
        alpha * capacity / max(capacities) + beta + gamma * (1 - abs(p) / peak)
        for capacity, p in zip(capacities, pressure)
    )
    assert min(si) > Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
    held = tuple(h * p for p in pressure)
    rate = tuple(capacity * p for capacity, p in zip(capacities, held))
    endpoint = tuple(x + dt * r for x, r in zip(forms, rate))
    qualified = tuple(abs(p) <= Q(cfg["EPS_DNFR_STABLE"]) for p in held)
    next_counts = tuple(
        count + 1 if stable else 0 for count, stable in zip(counts, qualified)
    )
    due = tuple(count >= cfg["VF_ADAPT_TAU"] for count in next_counts)
    adapted = tuple(
        (
            (1 - mu) * capacity + mu * feedback._neighbor_mean(v, node // 2)
            if due[node]
            else capacity
        )
        for node, capacity in enumerate(capacities)
    )
    assert (
        Q(cfg["EPI_MIN"])
        < min(forms + endpoint)
        <= max(forms + endpoint)
        < Q(cfg["EPI_MAX"])
    )
    assert Q(cfg["VF_MIN"]) < min(adapted) <= max(adapted) < Q(cfg["VF_MAX"])
    return {
        "initial_pressure": pressure,
        "pressure": held,
        "si": si,
        "rate": rate,
        "acceleration": tuple(
            (r - Q(old)) / dt for r, old in zip(rate, previous_rates)
        ),
        "form": endpoint,
        "capacity": adapted,
        "counts": next_counts,
        "due": due,
        "qualified": qualified,
    }


def _assert_contained(prediction, enclosure, *, allowance=Q(0)):
    for channel, centers_key, radius_key in (
        ("form", "means", "form_radii"),
        ("capacity", "capacities", "capacity_radii"),
    ):
        for value, center in zip(
            mean._means(prediction[channel]), enclosure[centers_key]
        ):
            assert center.lo - allowance <= value <= center.hi + allowance
        for contrast, radius in zip(
            mean._contrasts(prediction[channel]), enclosure[radius_key]
        ):
            assert abs(contrast) <= radius + allowance
    for node, count in enumerate(prediction["counts"]):
        assert count in enclosure["counts"][node]


def _freeze():
    graphs, fine, bounds = {}, {}, {}
    for name in NAMES:
        graph, state = _source(name)
        forms = tuple(graph.nodes[node]["EPI"] for node in graph)
        capacities = tuple(graph.nodes[node]["nu_f"] for node in graph)
        counts = tuple(graph.nodes[node]["stable_count"] for node in graph)
        rates = base._lift(base.RATES)
        assert all("epi_time_history" not in data for _, data in graph.nodes(data=True))
        predictions, enclosures = [], []
        for _ in range(2 if name.startswith("positive") else 1):
            state = _coarse_step(graph, state)
            prediction = _fine_step(graph, forms, capacities, counts, rates)
            _assert_contained(prediction, state)
            assert all(
                prediction["due"][node] in state["writes"][node] for node in graph
            )
            predictions.append(prediction)
            enclosures.append(state)
            forms, capacities, counts, rates = (
                prediction[key] for key in ("form", "capacity", "counts", "rate")
            )
        graphs[name], fine[name], bounds[name] = (
            graph,
            tuple(predictions),
            tuple(enclosures),
        )
    assert bounds["positive_plus"] == bounds["positive_minus"]
    assert bounds["ambiguous_plus"] == bounds["ambiguous_minus"]
    first, second = bounds["positive_plus"]
    assert first["statuses"] == ("fail", "pass", "pass", "pass", "pass")
    assert second["statuses"] == ("pass",) * 5
    assert first["writes"][2] == frozenset((False,))
    assert second["writes"][2] == frozenset((True,))
    assert first["capacity_radii"][1] > 0
    positive_gap = (
        Q(base.DT)
        * Q(graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"])
        * base._weight(graph, "DNFR_WEIGHTS", "epi")
        * Q(graph.graph["VF_ADAPT_MU"])
        * feedback.EPSILON
        / 2
        * (
            1
            - Q(base.DT)
            * Q(graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"])
            * base._weight(graph, "DNFR_WEIGHTS", "epi")
        )
        * RHO
    )
    positive_means, negative_means = (
        mean._means(fine[name][1]["form"]) for name in NAMES[:2]
    )
    assert tuple(a - b for a, b in zip(positive_means, negative_means)) == (
        Q(0),
        positive_gap,
        Q(0),
        Q(0),
        Q(0),
    )
    assert positive_gap > Q(1, 10**13)
    ambiguous = bounds["ambiguous_plus"][0]
    assert ambiguous["statuses"] == ("undetermined", "pass", "pass", "pass", "pass")
    assert ambiguous["writes"][0] == frozenset((False,))
    assert ambiguous["writes"][1] == frozenset((False, True))
    assert fine["ambiguous_plus"][0]["due"][:2] == (False, False)
    assert fine["ambiguous_minus"][0]["due"][:2] == (False, True)
    gap = (
        mean._means(fine["ambiguous_plus"][0]["capacity"])[0]
        - mean._means(fine["ambiguous_minus"][0]["capacity"])[0]
    )
    assert gap == Q(graph.graph["VF_ADAPT_MU"]) * feedback.EPSILON / 2 > Q(1, 10**5)
    return graphs, fine, bounds, gap


@pytest.fixture(scope="module", params=(False, True), ids=("numpy", "scalar"))
def joint(request):
    graphs, predictions, envelopes, gap = _freeze()
    # Freeze all four preparations and all six predictions before the first
    # response on this backend. No endpoint feeds an analytical forecast.
    responses = {
        name: tuple(
            mean._capture(graph, scalar=request.param) for _ in predictions[name]
        )
        for name, graph in graphs.items()
    }
    return responses, predictions, envelopes, gap


@pytest.mark.parametrize(
    "interval, expected",
    (
        (I(-1, 1), "pass"),
        (I(1), "pass"),
        (I(-1), "pass"),
        (I(2, 3), "fail"),
        (I(-3, -2), "fail"),
        (I(1, 2), "undetermined"),
        (I(-2, -1), "undetermined"),
        (I(-2, 2), "undetermined"),
    ),
)
def test_interval_pressure_classifier_preserves_inclusive_equality(interval, expected):
    assert _classify_pressure(interval, Q(1)) == expected


@pytest.mark.parametrize("name", NAMES)
def test_complete_native_response_matches_independent_fine_forecast(joint, name):
    responses, predictions, bounds, _ = joint
    for case, expected, enclosure in zip(
        responses[name], predictions[name], bounds[name]
    ):
        assert case["marks"] == feedback.MARKS
        for owner, channel, key in (
            ("pressure", "pressure", "initial_pressure"),
            ("si", "si", "si"),
            ("glyphs", "pressure", "pressure"),
        ):
            assert base._event(case, owner)["after"][channel] == pytest.approx(
                tuple(map(float, expected[key])), abs=base.ABS_ERROR, rel=0
            )
        for channel in ("form", "pressure", "capacity", "rate", "acceleration", "si"):
            assert case["final"][channel] == pytest.approx(
                tuple(map(float, expected[channel])), abs=base.ABS_ERROR, rel=0
            )
        assert case["final"]["counters"] == expected["counts"]
        observed = {**case["final"], "counts": case["final"]["counters"]}
        _assert_contained(observed, enclosure, allowance=Q(base.ABS_ERROR))


@pytest.mark.parametrize("name", NAMES)
def test_capacity_gate_uses_held_pressure_and_prewrite_capacity_snapshot(joint, name):
    responses, predictions, _, _ = joint
    for case, expected in zip(responses[name], predictions[name]):
        integration, capacity = (
            base._event(case, owner) for owner in ("integration", "capacity")
        )
        assert (
            integration["before"]["capacity"]
            == integration["after"]["capacity"]
            == case["initial"]["capacity"]
        )
        assert capacity["before"]["form"] == capacity["after"]["form"]
        assert (
            case["final"]["pressure"]
            == base._event(case, "glyphs")["after"]["pressure"]
        )
        writes = [
            event for event in case["events"] if event["name"] == "capacity_write"
        ]
        assert tuple(event["node"] for event in writes) == tuple(
            node for node, due in enumerate(expected["due"]) if due
        )
        assert tuple(event["value"] for event in writes) == pytest.approx(
            tuple(float(expected["capacity"][event["node"]]) for event in writes),
            abs=base.ABS_ERROR,
            rel=0,
        )


@pytest.mark.parametrize("name", NAMES[:2])
def test_joint_envelope_crosses_pressure_and_due_boundaries_without_fine_lift(
    joint, name
):
    responses, predictions, enclosures, _ = joint
    first, second = responses[name]
    assert first["final"]["counters"] == (0, 0, 4, 5, 5, 5, 5, 5, 5, 5)
    assert second["final"]["counters"] == (1, 1, 5, 6, 6, 6, 6, 6, 6, 6)
    assert abs(mean._contrasts(first["final"]["capacity"])[1]) > Q(1, 10**5)
    assert second["initial"] == first["final"]
    assert second["graph"] is first["graph"]
    assert predictions[name][0]["qualified"][:2] == (False, False)
    assert predictions[name][1]["qualified"][:2] == (True, True)
    epsilon = Q(first["configuration"]["EPS_DNFR_STABLE"])
    assert all(abs(p) > epsilon for p in predictions[name][1]["initial_pressure"][:2])
    assert all(abs(p) < epsilon for p in predictions[name][1]["pressure"][:2])
    assert enclosures[name][1]["form_radii"][1] > RHO
    assert max(value.width for value in enclosures[name][1]["means"]) > 0


def test_indeterminate_admission_retains_both_actual_continuations(joint):
    responses, predictions, envelopes, gap = joint
    positive, negative = (responses[name][0] for name in NAMES[2:])
    for channel in ("form", "capacity"):
        assert mean._means(positive["initial"][channel]) == mean._means(
            negative["initial"][channel]
        )
    assert positive["initial"]["counters"] == negative["initial"]["counters"]
    assert positive["records"] == negative["records"]
    assert positive["marks"] == negative["marks"] == feedback.MARKS
    bound = envelopes["ambiguous_plus"][0]
    assert bound["counts"][0] == frozenset((0, 1))
    assert bound["counts"][1] == frozenset((0, 5))
    assert positive["final"]["counters"][:2] == (1, 0)
    assert negative["final"]["counters"][:2] == (0, 5)
    observed_gap = (
        mean._means(positive["final"]["capacity"])[0]
        - mean._means(negative["final"]["capacity"])[0]
    )
    assert float(observed_gap) == pytest.approx(float(gap), abs=base.ABS_ERROR, rel=0)
    # Treating the unresolved predicate as false would exclude the negative
    # lift's actual write; treating it as true would exclude the positive one.
    assert predictions["ambiguous_plus"][0]["due"][1] is False
    assert predictions["ambiguous_minus"][0]["due"][1] is True


def test_one_certified_coarse_continuation_encloses_distinct_actual_means(joint):
    responses, predictions, envelopes, _ = joint
    positive, negative = (responses[name] for name in NAMES[:2])
    assert envelopes[NAMES[0]] == envelopes[NAMES[1]]
    assert mean._means(positive[0]["final"]["form"]) == mean._means(
        negative[0]["final"]["form"]
    )
    for step in (0, 1):
        assert (
            positive[step]["final"]["capacity"] == negative[step]["final"]["capacity"]
        )
        assert (
            positive[step]["final"]["counters"] == negative[step]["final"]["counters"]
        )
    predicted = (
        mean._means(predictions[NAMES[0]][1]["form"])[1]
        - mean._means(predictions[NAMES[1]][1]["form"])[1]
    )
    observed = (
        mean._means(positive[1]["final"]["form"])[1]
        - mean._means(negative[1]["final"]["form"])[1]
    )
    assert observed > Q(1, 10**13)
    assert float(observed) == pytest.approx(
        float(predicted), abs=2 * base.ABS_ERROR, rel=0
    )


@pytest.mark.parametrize("name", NAMES)
def test_live_context_clock_and_default_policies_are_retained(joint, name):
    responses = joint[0][name]
    for step, case in enumerate(responses):
        cfg = case["configuration"]
        for key in (
            "DNFR_WEIGHTS",
            "SI_WEIGHTS",
            "GLYPH_FACTORS",
            "SELECTOR_THRESHOLDS",
            "EPS_DNFR_STABLE",
            "PHASE_ADAPT",
            "VF_ADAPT_TAU",
            "VF_ADAPT_MU",
            "DT_MIN",
        ):
            assert cfg[key] == DEFAULTS[key]
        assert cfg["GRAMMAR_REJECTION_MODE"] == "raise"
        assert "glyph_selector" not in cfg and "integrator" not in cfg
        assert case["final"]["phase"] == (0.0,) * 10
        assert case["final"]["edges"] == case["initial"]["edges"]
        assert case["final"]["ordinal"] == step + 1
        assert case["final"]["time"] == 1 + (step + 1) * base.DT
        assert not case["history"].get("stable_frac")
        assert len(case["remesh_history"]) == step + 1
        assert (
            base._event(case, "remesh_gate")["before"]
            == base._event(case, "remesh_gate")["after"]
        )
        assert not any(event["name"] == "network_remesh" for event in case["events"])
        assert case["history"]["phase_state"][-1] == "stable"
        assert case["history"]["phase_R"][-1] == 1.0
        for node, (grammar, progress) in enumerate(case["records"]):
            assert grammar.operator_step == 4 + step
            assert grammar.prior_coherence is True and grammar.u2_debt == 0
            assert grammar.history == ("EN",) + ("IL",) * (step + 1)
            assert progress is None
            assert case["history"]["since_AL"][node] == step + 1
            assert case["history"]["since_EN"][node] == step + 1
            assert case["final"]["temporal"][node][-2:] == (
                (1 + step * base.DT, case["initial"]["form"][node]),
                (1 + (step + 1) * base.DT, case["final"]["form"][node]),
            )
