"""Prospective capacity feedback in two complete native Coherence calls.

Freeze the doubled-C5 support, pair-contiguous node order, zero phase, seed 17,
t=1 and dt=1/16. Initial block means are 1/8; capacity is 1+1/1024 in the
first pair and 1 elsewhere. Five preparations use first-pair form contrasts
+1/1024, -1/1024 or zero with counters (0, 4), or +/-1/1024 with counters
(4, 4); all other counters are 4. Counters and ordinal grammar context are
supplied premises, without invented temporal history. Every policy is default.

All two-call predictions and wrong-law alternatives are derived with Fraction
arithmetic before any native response. Each preparation continues its own live
graph, with scalar and NumPy paths, under a fixed absolute 1e-14 comparison
budget. Exact pair identities are algebraic; floating comparisons retain that
budget. Capacity-form covariance and capacity variance affect the second mean.
The fine oracle verifies execution; its invertible pair chart is not a reduced
model. These controls establish neither formation nor unlimited mean closure.
"""

from fractions import Fraction as Q

import pytest

from tests.physics import test_native_collective_mean as mean
from tests.physics import test_native_collective_step as base
from tnfr.constants import DEFAULTS

EPSILON = Q(1, 1024)
MEANS = (Q(1, 8),) * 5
CAPACITIES = (1 + EPSILON, Q(1), Q(1), Q(1), Q(1))
ZERO = (Q(0),) * 5
PREPARATIONS = (
    ("split_positive", EPSILON, False),
    ("split_negative", -EPSILON, False),
    ("split_zero", Q(0), False),
    ("matched_positive", EPSILON, True),
    ("matched_negative", -EPSILON, True),
)
MARKS = tuple((node, "IL") for node in range(10))


def _neighbor_mean(values, block):
    return (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2


def _source(contrast, matched):
    graph = base._source(tuple(map(float, MEANS)), base.PROFILES[0], ready=True)
    forms = mean._fine(MEANS, (contrast, *ZERO[1:]))
    for node, form in enumerate(forms):
        graph.nodes[node].update(EPI=float(form), nu_f=float(CAPACITIES[node // 2]))
    if not matched:
        graph.nodes[0]["stable_count"] = 0
    return graph


def _forecast(graph, forms, capacities, counts, prior_rates):
    """Compute from preparation and the declared law, never a native response."""
    cfg = graph.graph
    forms, capacities = tuple(map(Q, forms)), tuple(map(Q, capacities))
    m, v = mean._means(forms), mean._means(capacities)
    d, e = mean._contrasts(forms), mean._contrasts(capacities)
    k, b = (base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf"))
    h, mu, dt = (
        Q(cfg["GLYPH_FACTORS"]["IL_dnfr_factor"]),
        Q(cfg["VF_ADAPT_MU"]),
        Q(base.DT),
    )
    # Fine pressures use independent neighbor means, without a runtime kernel.
    pressure = tuple(
        k * (_neighbor_mean(m, node // 2) - x)
        + b * (_neighbor_mean(v, node // 2) - capacity)
        for node, (x, capacity) in enumerate(zip(forms, capacities))
    )
    pressure_mean = tuple(
        k * (_neighbor_mean(m, i) - m[i]) + b * (_neighbor_mean(v, i) - v[i])
        for i in range(5)
    )
    peak, maximum_capacity = max(map(abs, pressure)), max(capacities)
    assert peak > 0 and min(capacities) > 0
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha * capacity / maximum_capacity + beta + gamma * (1 - abs(p) / peak)
        for capacity, p in zip(capacities, pressure)
    )
    threshold = Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
    assert alpha * min(capacities) / maximum_capacity + beta > threshold
    held = tuple(h * p for p in pressure)
    assert max(map(abs, held)) < Q(cfg["EPS_DNFR_STABLE"])
    rates = tuple(capacity * p for capacity, p in zip(capacities, held))
    endpoint = tuple(x + dt * rate for x, rate in zip(forms, rates))
    assert Q(cfg["EPI_MIN"]) < min(forms + endpoint)
    assert max(forms + endpoint) < Q(cfg["EPI_MAX"])
    next_counts = tuple(count + 1 for count in counts)
    due = tuple(count >= cfg["VF_ADAPT_TAU"] for count in next_counts)
    adapted = tuple(
        (
            (1 - mu) * capacity + mu * _neighbor_mean(v, node // 2)
            if due[node]
            else capacity
        )
        for node, capacity in enumerate(capacities)
    )
    assert Q(cfg["VF_MIN"]) < min(adapted) <= max(adapted) < Q(cfg["VF_MAX"])
    correction = tuple(ei * (k * di + b * ei) for di, ei in zip(d, e))
    reduced = tuple(
        mi + dt * h * (vi * pi - ci)
        for mi, vi, pi, ci in zip(m, v, pressure_mean, correction)
    )
    assert mean._means(endpoint) == reduced
    contrast_endpoint = tuple(
        di + dt * h * (ei * pi - vi * (k * di + b * ei))
        for di, ei, pi, vi in zip(d, e, pressure_mean, v)
    )
    assert mean._contrasts(endpoint) == contrast_endpoint
    mean_only = tuple(mi + dt * h * vi * pi for mi, vi, pi in zip(m, v, pressure_mean))
    return {
        "initial_form": forms,
        "initial_capacity": capacities,
        "initial_pressure": pressure,
        "pressure_means": pressure_mean,
        "pressure": held,
        "si": si,
        "rate": rates,
        "acceleration": tuple(
            (rate - Q(old)) / dt for rate, old in zip(rates, prior_rates)
        ),
        "form": endpoint,
        "capacity": adapted,
        "counts": next_counts,
        "due": due,
        "mean": reduced,
        "contrast": contrast_endpoint,
        "capacity_contrast": mean._contrasts(adapted),
        "correction": correction,
        "mean_only": mean_only,
    }


def _freeze_predictions():
    graphs, predictions, alternatives = {}, {}, {}
    for name, contrast, matched in PREPARATIONS:
        graph = _source(contrast, matched)
        forms = tuple(graph.nodes[node]["EPI"] for node in graph)
        capacities = tuple(graph.nodes[node]["nu_f"] for node in graph)
        counts = tuple(graph.nodes[node]["stable_count"] for node in graph)
        first = _forecast(graph, forms, capacities, counts, base._lift(base.RATES))
        second = _forecast(
            graph, first["form"], first["capacity"], first["counts"], first["rate"]
        )
        # Wrong causal alternatives are fixed before evaluating either response.
        frozen_capacity = _forecast(
            graph, first["form"], capacities, first["counts"], first["rate"]
        )
        early_capacity_form = tuple(
            x + Q(base.DT) * v * p
            for x, v, p in zip(
                first["initial_form"], first["capacity"], first["pressure"]
            )
        )
        alternatives[name] = {
            "frozen_capacity": frozen_capacity["mean"],
            "early_capacity": mean._means(early_capacity_form),
        }
        graphs[name], predictions[name] = graph, (first, second)

    source = next(iter(graphs.values()))
    cfg = source.graph
    k, b = (base._weight(source, "DNFR_WEIGHTS", name) for name in ("epi", "vf"))
    alpha = base._weight(source, "SI_WEIGHTS", "alpha")
    h, mu, dt = (
        Q(cfg["GLYPH_FACTORS"]["IL_dnfr_factor"]),
        Q(cfg["VF_ADAPT_MU"]),
        Q(base.DT),
    )
    # This convex-hull argument admits the whole stated form interval, not
    # only the five supplied preparations evaluated below.
    assert alpha / (1 + EPSILON) > Q(cfg["SELECTOR_THRESHOLDS"]["si_hi"])
    assert 0 < dt * h * (1 + EPSILON) * k < 1
    assert 0 <= mu <= 1
    radius = EPSILON
    for _ in range(2):
        assert h * (2 * k * radius + b * EPSILON) < Q(cfg["EPS_DNFR_STABLE"])
        radius += dt * h * (1 + EPSILON) * b * EPSILON
        assert Q(31, 250) < Q(1, 8) - radius
        assert Q(1, 8) + radius < Q(63, 500)
    expected_gap = (
        dt * h * mu * EPSILON * k * (1 - dt * h * k * (1 + EPSILON)) * EPSILON
    )
    positive, negative, zero = (
        predictions[f"split_{name}"] for name in ("positive", "negative", "zero")
    )
    assert positive[0]["mean"] == negative[0]["mean"] == zero[0]["mean"]
    assert positive[0]["capacity"] == negative[0]["capacity"] == zero[0]["capacity"]
    assert (
        positive[1]["mean"][0] - negative[1]["mean"][0] == expected_gap > Q(1, 10**10)
    )
    assert positive[1]["mean"][1:] == negative[1]["mean"][1:]
    assert (
        predictions["matched_positive"][1]["mean"]
        == predictions["matched_negative"][1]["mean"]
    )
    for name, first_second in predictions.items():
        first, second = first_second
        assert max(
            abs(a - b)
            for a, b in zip(first["mean"], alternatives[name]["early_capacity"])
        ) > Q(1, 10**10)
        assert max(
            abs(a - b)
            for a, b in zip(second["mean"], alternatives[name]["frozen_capacity"])
        ) > Q(1, 10**10)
        assert any(a != b for a, b in zip(first["initial_capacity"], first["capacity"]))
    variance_error = zero[1]["mean"][0] - zero[1]["mean_only"][0]
    assert variance_error == -dt * h * b * (mu * EPSILON / 2) ** 2
    assert abs(variance_error) > Q(1, 10**12)
    return graphs, predictions, alternatives, expected_gap


@pytest.fixture(scope="module", params=(False, True), ids=("numpy", "scalar"))
def feedback(request):
    graphs, predictions, alternatives, gap = _freeze_predictions()
    # Every preparation, guard and alternative is frozen before the first response.
    responses = {
        name: tuple(mean._capture(graph, scalar=request.param) for _ in range(2))
        for name, graph in graphs.items()
    }
    return responses, predictions, alternatives, gap


@pytest.mark.parametrize("name", tuple(item[0] for item in PREPARATIONS))
@pytest.mark.parametrize("step", (0, 1))
def test_complete_native_calls_match_independent_capacity_feedback(
    feedback, name, step
):
    responses, predictions, _, _ = feedback
    case, expected = responses[name][step], predictions[name][step]
    assert case["marks"] == MARKS
    for owner, channel, predicted in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
        ("glyphs", "pressure", "pressure"),
    ):
        assert base._event(case, owner)["after"][channel] == pytest.approx(
            tuple(map(float, expected[predicted])), abs=base.ABS_ERROR, rel=0
        )
    for channel in ("form", "pressure", "capacity", "rate", "acceleration", "si"):
        assert case["final"][channel] == pytest.approx(
            tuple(map(float, expected[channel])), abs=base.ABS_ERROR, rel=0
        )
    assert case["final"]["counters"] == expected["counts"]
    assert tuple(map(float, mean._means(case["final"]["form"]))) == pytest.approx(
        tuple(map(float, expected["mean"])), abs=base.ABS_ERROR, rel=0
    )


@pytest.mark.parametrize("name", tuple(item[0] for item in PREPARATIONS))
@pytest.mark.parametrize("step", (0, 1))
def test_capacity_changes_after_flow_and_is_consumed_on_the_next_call(
    feedback, name, step
):
    responses, predictions, _, _ = feedback
    case, expected = responses[name][step], predictions[name][step]
    integration, capacity = (
        base._event(case, owner) for owner in ("integration", "capacity")
    )
    assert (
        integration["before"]["capacity"]
        == integration["after"]["capacity"]
        == case["initial"]["capacity"]
    )
    assert capacity["before"]["form"] == capacity["after"]["form"]
    assert case["final"]["pressure"] == base._event(case, "glyphs")["after"]["pressure"]
    assert case["final"]["si"] == base._event(case, "si")["after"]["si"]
    writes = [event for event in case["events"] if event["name"] == "capacity_write"]
    assert tuple(event["node"] for event in writes) == tuple(
        node for node, due in enumerate(expected["due"]) if due
    )
    assert tuple(event["value"] for event in writes) == pytest.approx(
        tuple(float(expected["capacity"][event["node"]]) for event in writes),
        abs=base.ABS_ERROR,
        rel=0,
    )
    assert any(
        abs(a - b) > base.ABS_ERROR
        for a, b in zip(case["final"]["capacity"], case["initial"]["capacity"])
    )
    if step:
        assert case["initial"] == responses[name][0]["final"]
        assert case["graph"] is responses[name][0]["graph"]
        assert (
            base._event(case, "pressure")["after"]["pressure"]
            != case["initial"]["pressure"]
        )


@pytest.mark.parametrize("name", tuple(item[0] for item in PREPARATIONS))
def test_native_context_and_memory_are_retained_without_policy_substitution(
    feedback, name
):
    responses = feedback[0][name]
    for step, case in enumerate(responses):
        cfg = case["configuration"]
        for key in (
            "DNFR_WEIGHTS",
            "SI_WEIGHTS",
            "GLYPH_FACTORS",
            "SELECTOR_THRESHOLDS",
            "PHASE_ADAPT",
            "VF_ADAPT_TAU",
            "VF_ADAPT_MU",
            "DT_MIN",
        ):
            assert cfg[key] == DEFAULTS[key]
        assert cfg["GRAMMAR_REJECTION_MODE"] == "raise"
        assert "glyph_selector" not in cfg and "integrator" not in cfg
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
        assert tuple(case["history"]["phase_state"]) == ("stable",) * (step + 1)
        assert tuple(case["history"]["phase_R"]) == (1.0,) * (step + 1)
        assert tuple(case["history"]["phase_disr"]) == (0.0,) * (step + 1)
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
    for suffix, parameter in (("G", "PHASE_K_GLOBAL"), ("L", "PHASE_K_LOCAL")):
        gain = Q(DEFAULTS[parameter])
        policy = DEFAULTS["PHASE_ADAPT"]
        for step, case in enumerate(responses):
            gain += Q(policy["down"]) * (Q(policy[f"k{suffix}_min"]) - gain)
            assert case["history"][f"phase_k{suffix}"][step] == pytest.approx(
                float(gain), abs=base.ABS_ERROR, rel=0
            )


def test_matching_means_marginals_and_marks_lose_capacity_form_covariance(feedback):
    responses, predictions, _, gap = feedback
    positive, negative = (
        responses[f"split_{name}"] for name in ("positive", "negative")
    )
    for step in (0, 1):
        assert positive[step]["marks"] == negative[step]["marks"] == MARKS
        assert positive[step]["records"] == negative[step]["records"]
        assert (
            positive[step]["final"]["capacity"] == negative[step]["final"]["capacity"]
        )
        assert (
            positive[step]["final"]["counters"] == negative[step]["final"]["counters"]
        )
    for a, b in base.PAIRS:
        assert sorted(positive[0]["final"]["form"][node] for node in (a, b)) == sorted(
            negative[0]["final"]["form"][node] for node in (a, b)
        )
    assert mean._means(positive[0]["final"]["form"]) == mean._means(
        negative[0]["final"]["form"]
    )
    actual = (
        mean._means(positive[1]["final"]["form"])[0]
        - mean._means(negative[1]["final"]["form"])[0]
    )
    assert float(actual) == pytest.approx(float(gap), abs=base.ABS_ERROR, rel=0)
    assert abs(actual) > Q(1, 10**10)
    assert (
        predictions["split_positive"][1]["correction"][0]
        != predictions["split_negative"][1]["correction"][0]
    )


def test_current_covariance_and_variance_do_not_close_asymmetric_adaptation(feedback):
    responses, predictions, _, _ = feedback
    for name in ("split_positive", "split_negative"):
        first = predictions[name][0]
        initial_e = mean._contrasts(first["initial_capacity"])
        initial_d = mean._contrasts(first["initial_form"])
        assert tuple(e * d for e, d in zip(initial_e, initial_d)) == ZERO
        assert tuple(e * e for e in initial_e) == ZERO
    positive, negative = (
        predictions[f"split_{name}"][0] for name in ("positive", "negative")
    )
    covariance_positive = positive["capacity_contrast"][0] * positive["contrast"][0]
    covariance_negative = negative["capacity_contrast"][0] * negative["contrast"][0]
    assert covariance_positive == -covariance_negative != 0
    assert (
        responses["split_positive"][0]["initial"]["counters"]
        == responses["split_negative"][0]["initial"]["counters"]
    )
    assert (
        tuple(
            (a + b) / 2
            for a, b in zip(
                predictions["split_positive"][1]["mean"],
                predictions["split_negative"][1]["mean"],
            )
        )
        == predictions["split_zero"][1]["mean"]
    )


def test_closed_form_first_mean_and_capacity_updates_match_the_fine_oracle(feedback):
    responses, predictions, _, _ = feedback
    cfg = responses["split_zero"][0]["configuration"]
    weights = {name: Q(value) for name, value in cfg["DNFR_WEIGHTS"].items()}
    h, b, mu, dt = (
        Q(cfg["GLYPH_FACTORS"]["IL_dnfr_factor"]),
        weights["vf"] / sum(weights.values()),
        Q(cfg["VF_ADAPT_MU"]),
        Q(base.DT),
    )
    expected_mean = (
        Q(1, 8) - dt * h * (1 + EPSILON) * b * EPSILON,
        Q(1, 8) + dt * h * b * EPSILON / 2,
        Q(1, 8),
        Q(1, 8),
        Q(1, 8) + dt * h * b * EPSILON / 2,
    )
    expected_capacity_mean = (
        1 + EPSILON * (1 - mu / 2),
        1 + mu * EPSILON / 2,
        Q(1),
        Q(1),
        1 + mu * EPSILON / 2,
    )
    for name in ("split_positive", "split_negative", "split_zero"):
        first = predictions[name][0]
        assert first["mean"] == expected_mean
        assert mean._means(first["capacity"]) == expected_capacity_mean
        assert first["capacity_contrast"] == (-mu * EPSILON / 2, *ZERO[1:])


def test_pair_matched_adaptation_preserves_the_mean_only_reduction(feedback):
    responses, predictions, _, _ = feedback
    for step in (0, 1):
        for name in ("matched_positive", "matched_negative"):
            expected = predictions[name][step]
            assert expected["capacity_contrast"] == ZERO
            assert expected["correction"] == ZERO
            assert expected["mean"] == expected["mean_only"]
        first, second = (
            responses[name][step] for name in ("matched_positive", "matched_negative")
        )
        assert tuple(map(float, mean._means(first["final"]["form"]))) == pytest.approx(
            tuple(map(float, mean._means(second["final"]["form"]))),
            abs=base.ABS_ERROR,
            rel=0,
        )


def test_capacity_variance_survives_without_form_contrast(feedback):
    responses, predictions, _, _ = feedback
    first, second = predictions["split_zero"]
    assert first["contrast"] == ZERO
    assert first["capacity_contrast"][0] != 0
    assert second["correction"][0] > 0
    actual = mean._means(responses["split_zero"][1]["final"]["form"])[0]
    error = actual - second["mean_only"][0]
    assert float(error) == pytest.approx(
        float(second["mean"][0] - second["mean_only"][0]), abs=base.ABS_ERROR, rel=0
    )
    assert error < -Q(1, 10**12)


@pytest.mark.parametrize("name", tuple(item[0] for item in PREPARATIONS))
def test_wrong_causal_capacity_alternatives_are_separated_before_response(
    feedback, name
):
    responses, predictions, alternatives, _ = feedback
    for step, alternative in ((0, "early_capacity"), (1, "frozen_capacity")):
        observed = mean._means(responses[name][step]["final"]["form"])
        wrong = alternatives[name][alternative]
        assert max(abs(a - b) for a, b in zip(observed, wrong)) > Q(1, 10**10)
        assert predictions[name][step]["mean"] != wrong
