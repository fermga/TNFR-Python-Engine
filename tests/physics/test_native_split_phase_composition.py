"""One frozen split-phase composition checked through shared native stages.

Reuse the supplied cancellation geometry from the Resonance admission owner:
RA on twins 0/1, then IL on all other nodes, default fresh preparation and
policies, one unforced held-pressure Euler interval of length 1/16. Forecast
every stage independently before executing any of them. The shared kernels
are called in runtime order; this is not a runtime.step call, trajectory,
source-acquisition search, default-selector witness or next-call admission.
Existing runtime and joint-calculus owners retain their orchestration scope.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.physics import test_native_split_phase_resonance as resonance
from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
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
from tnfr.dynamics.dnfr import _resolve_dnfr_weights
from tnfr.metrics.sense_index import get_Si_weights
from tnfr.operators._resonance_identity import resonance_capacity_proposal
from tnfr.operators.grammar_debt import OPERATOR_STEP_KEY, U2_DEBT_KEY
from tnfr.validation import apply_canonical_clamps

DT = 1 / 16
TOL = 8e-15  # Binary64 stage checks, not a validated exact-circle enclosure.
REQUESTS = ("RA", "RA", *("IL",) * 8)


@pytest.fixture(scope="module", autouse=True)
def no_complete_native_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Only the declared shared-stage composition is admitted")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("step", "run", "_update_nodes"):
            patch.setattr(runtime, name, forbidden)
        yield


def _neighbors(node):
    block = node // 2
    return tuple(
        2 * neighbor + member
        for neighbor in ((block - 1) % 5, (block + 1) % 5)
        for member in (0, 1)
    )


def _moments(values):
    return (
        tuple((values[2 * i] + values[2 * i + 1]) / 2 for i in range(5)),
        tuple((values[2 * i + 1] - values[2 * i]) / 2 for i in range(5)),
    )


def _angle(phases):
    return math.atan2(sum(map(math.sin, phases)), sum(map(math.cos, phases)))


def _readout(x, v, theta, weights, si_weights):
    """Independent common-lift block formula, including signed phase contrast."""
    means, d = _moments(x)
    capacities, e = _moments(v)
    mu, a = _moments(theta)
    local = tuple(
        math.atan2(
            sum(math.cos(a[j]) * math.sin(mu[j]) for j in ((i - 1) % 5, (i + 1) % 5)),
            sum(math.cos(a[j]) * math.cos(mu[j]) for j in ((i - 1) % 5, (i + 1) % 5)),
        )
        for i in range(5)
    )
    k, b, t = weights["epi"], weights["vf"], weights["phase"] / math.pi
    baseline = tuple(
        k * ((means[(i - 1) % 5] + means[(i + 1) % 5]) / 2 - means[i])
        + b * ((capacities[(i - 1) % 5] + capacities[(i + 1) % 5]) / 2 - capacities[i])
        + t * (local[i] - mu[i])
        for i in range(5)
    )
    contrast = tuple(k * dx + b * dv + t * da for dx, dv, da in zip(d, e, a))
    pressure = tuple(
        baseline[i] + sign * contrast[i] for i in range(5) for sign in (1, -1)
    )
    peak = max(abs(y) + abs(z) for y, z in zip(baseline, contrast))
    vmax = max(y + abs(z) for y, z in zip(capacities, e))
    alpha, beta, gamma = si_weights
    si = tuple(
        alpha * v[n] / vmax
        + beta * (1 - abs(local[n // 2] - theta[n]) / math.pi)
        + gamma * (1 - abs(pressure[n]) / (peak or 1))
        for n in range(10)
    )
    assert max(theta) - min(theta) < math.pi
    assert all(0 < value < 1 for value in si)
    return pressure, si


def _forecast(graph):
    """Freeze every expectation from primitive state/configuration, not outputs."""
    cfg = graph.graph
    x = tuple(graph.nodes[n]["EPI"] for n in graph)
    v = tuple(graph.nodes[n]["nu_f"] for n in graph)
    theta = tuple(graph.nodes[n]["theta"] for n in graph)
    old_rates = tuple(get_attr(graph.nodes[n], ALIAS_DEPI) for n in graph)
    weights, si_weights = _resolve_dnfr_weights(graph), get_Si_weights(graph)
    pressure, si = _readout(x, v, theta, weights, si_weights)
    factors = cfg["GLYPH_FACTORS"]
    mix, coupling, boost = (
        factors[key]
        for key in ("RA_epi_diff", "RA_phase_coupling", "RA_vf_amplification")
    )
    # The two nonadjacent targets see the same unchanged neighbor snapshot.
    subsets = ((2, 8, 9), (2, 3, 8, 9))
    selected_means = tuple(
        sum(x[n] for n in subset) / len(subset) for subset in subsets
    )
    angles = tuple(_angle(tuple(theta[n] for n in subset)) for subset in subsets)
    event_x, event_v, event_theta = list(x), list(v), list(theta)
    for n in (0, 1):
        assert (
            tuple(
                sorted(
                    j for j in _neighbors(n) if abs(theta[j] - theta[n]) <= math.pi / 2
                )
            )
            == subsets[n]
        )
        event_x[n] = (1 - mix) * x[n] + mix * selected_means[n]
        event_theta[n] = (1 - coupling) * theta[n] + coupling * angles[n]
    assert selected_means == (0, 5 / 32)
    event_v[1] *= 1 + boost  # Only the second target crosses the default trigger.
    held = tuple(
        p if n < 2 else factors["IL_dnfr_factor"] * p for n, p in enumerate(pressure)
    )
    rates = tuple(capacity * p for capacity, p in zip(event_v, held))
    endpoint = tuple(value + DT * rate for value, rate in zip(event_x, rates))
    acceleration = tuple((rate - old) / DT for rate, old in zip(rates, old_rates))

    policy = cfg["PHASE_ADAPT"]
    resultant = abs(sum(complex(math.cos(t), math.sin(t)) for t in event_theta)) / 10
    # Post-event traces contain only AL, IL and RA, so disruptor load is zero.
    if resultant >= policy["R_hi"]:
        branch, targets = "stable", (policy["kG_min"], policy["kL_min"])
    elif resultant <= policy["R_lo"]:
        branch, targets = "dissonant", (
            policy["kG_max"],
            (policy["kL_min"] + policy["kL_max"]) / 2,
        )
    else:
        branch, targets = "transition", (
            (policy["kG_min"] + policy["kG_max"]) / 2,
            (policy["kL_min"] + policy["kL_max"]) / 2,
        )
    gains = tuple(
        old + policy["up" if target > old else "down"] * (target - old)
        for old, target in zip((cfg["PHASE_K_GLOBAL"], cfg["PHASE_K_LOCAL"]), targets)
    )
    global_angle = _angle(event_theta)
    final_theta = tuple(
        (1 - sum(gains)) * t
        + gains[0] * global_angle
        + gains[1] * _angle(tuple(event_theta[j] for j in _neighbors(n)))
        for n, t in enumerate(event_theta)
    )
    stable = tuple(
        value >= cfg["SELECTOR_THRESHOLDS"]["si_hi"]
        and abs(p) <= cfg["EPS_DNFR_STABLE"]
        for value, p in zip(si, held)
    )
    counts = tuple(
        graph.nodes[n]["stable_count"] + 1 if flag else 0
        for n, flag in enumerate(stable)
    )
    due = tuple(count >= cfg["VF_ADAPT_TAU"] for count in counts)
    adapted_v = tuple(
        (
            capacity
            + cfg["VF_ADAPT_MU"]
            * (sum(event_v[j] for j in _neighbors(n)) / 4 - capacity)
            if due[n]
            else capacity
        )
        for n, capacity in enumerate(event_v)
    )
    next_pressure, next_si = _readout(
        endpoint, adapted_v, final_theta, weights, si_weights
    )
    assert all(
        cfg["EPI_MIN"] < value < cfg["EPI_MAX"] for value in (*event_x, *endpoint)
    )
    assert all(
        cfg["VF_MIN"] < value < cfg["VF_MAX"] for value in (*event_v, *adapted_v)
    )
    return dict(
        initial=x,
        initial_phase=theta,
        old_rates=old_rates,
        prepared=pressure,
        si=si,
        event_x=tuple(event_x),
        event_v=tuple(event_v),
        event_theta=tuple(event_theta),
        held=held,
        rates=rates,
        endpoint=endpoint,
        acceleration=acceleration,
        branch=branch,
        resultant=resultant,
        gains=gains,
        final_theta=final_theta,
        counts=counts,
        due=due,
        adapted_v=adapted_v,
        next_pressure=next_pressure,
        next_si=next_si,
        coupling=coupling,
    )


def _snapshot(graph):
    return {
        name: tuple(get_attr(graph.nodes[n], alias) for n in graph)
        for name, alias in (
            ("x", ALIAS_EPI),
            ("v", ALIAS_VF),
            ("theta", ALIAS_THETA),
            ("pressure", ALIAS_DNFR),
            ("si", ALIAS_SI),
            ("rates", ALIAS_DEPI),
            ("acceleration", ALIAS_D2EPI),
        )
    }


@pytest.fixture(scope="module")
def composition(no_complete_native_execution):
    graph = resonance._source(cancellation=True)
    inject_defaults(graph)
    for n in graph:
        form, rate = graph.nodes[n]["EPI"], (n - 4) / 64
        set_attr(graph.nodes[n], ALIAS_DEPI, rate)
        graph.nodes[n].update(
            epi_time_history=[(7 / 8, form - rate / 8), (1.0, form)],
            stable_count=n % graph.graph["VF_ADAPT_TAU"],
        )
    expected = _forecast(graph)  # Freeze before any preparation or event.
    gamma = integrators._get_gamma_spec(graph, strict=True)
    assert gamma["type"] == "none" and integrators._uses_builtin_gamma(gamma)
    assert integrators.prepare_integration_params(graph, DT, None, "euler") == (
        DT,
        1,
        1.0,
        "euler",
    )
    original_history = tuple(tuple(graph.nodes[n]["epi_time_history"]) for n in graph)

    # Constituent shared stages in _update_nodes order, without that executor.
    runtime._record_mutation_flow_boundary(graph)
    runtime._prepare_dnfr(graph, use_Si=True)
    prepared = _snapshot(graph)
    selectors._apply_glyph_batch(
        graph, lambda _graph, n: REQUESTS[n], graph.graph.setdefault("history", {})
    )
    runtime._record_mutation_flow_boundary(graph)
    event = _snapshot(graph)
    event_history = tuple(tuple(graph.nodes[n]["epi_time_history"]) for n in graph)
    integrators.DefaultIntegrator().integrate(
        graph, dt=DT, t=None, method="euler", n_jobs=1
    )
    flow = _snapshot(graph)
    for n in graph:
        apply_canonical_clamps(graph.nodes[n], graph, n)
    clamped = _snapshot(graph)
    coordination.coordinate_global_local_phase(graph, None, None, n_jobs=1)
    adaptation.adapt_vf_after_structural_stability(graph, n_jobs=1)
    runtime._record_mutation_flow_boundary(graph)
    final = _snapshot(graph)
    # Terminal runtime stages: archive the endpoint, then independently abstain
    # from delayed REMESH because no qualifying stability window exists.
    assert not graph.graph["history"].get("stable_frac", ())
    assert graph.graph["REMESH_STABILITY_WINDOW"] > 0
    nodes_before_remesh = deepcopy(dict(graph.nodes(data=True)))
    support_before_remesh = tuple(graph.edges(data="weight"))
    runtime._update_epi_hist(graph)
    runtime._maybe_remesh(graph)
    assert dict(graph.nodes(data=True)) == nodes_before_remesh
    assert tuple(graph.edges(data="weight")) == support_before_remesh
    assert _snapshot(graph) == final
    assert not graph.graph["history"]["stable_frac"]
    assert "_last_remesh_step" not in graph.graph
    context = deepcopy(dict(graph.graph["history"]))
    runtime._prepare_dnfr(graph, use_Si=True)
    return dict(
        graph=graph,
        expected=expected,
        prepared=prepared,
        event=event,
        event_history=event_history,
        original_history=original_history,
        flow=flow,
        clamped=clamped,
        final=final,
        context=context,
        next_readout=_snapshot(graph),
    )


def _close(actual, expected):
    assert actual == pytest.approx(expected, abs=TOL, rel=0)


def test_fresh_preparation_then_unequal_resonance_triggers(composition):
    c, f = composition, composition["expected"]
    _close(c["prepared"]["pressure"], f["prepared"])
    _close(c["prepared"]["si"], f["si"])
    for key, expected in (
        ("x", "event_x"),
        ("v", "event_v"),
        ("theta", "event_theta"),
        ("pressure", "held"),
    ):
        _close(c["event"][key], f[expected])
    assert c["event"]["v"][0] == 7 / 8
    assert c["event"]["v"][1] > 9 / 8
    assert c["event"]["si"] == c["prepared"]["si"]


def test_held_flow_retains_covariance_and_associated_previous_rates(composition):
    c, f = composition, composition["expected"]
    for key, expected in (
        ("x", "endpoint"),
        ("rates", "rates"),
        ("acceleration", "acceleration"),
    ):
        _close(c["flow"][key], f[expected])
    vbar, e = _moments(f["event_v"])
    pbar, dp = _moments(f["held"])
    mean_rate, contrast_rate = _moments(c["flow"]["rates"])
    _close(mean_rate, tuple(v * p + dv * q for v, p, dv, q in zip(vbar, pbar, e, dp)))
    _close(
        contrast_rate, tuple(v * q + dv * p for v, p, dv, q in zip(vbar, pbar, e, dp))
    )
    assert abs(e[0] * dp[0]) > 1e-4
    assert abs(mean_rate[0] - vbar[0] * pbar[0]) > 1e-4
    swapped = (f["rates"][0] - f["old_rates"][1]) / DT
    assert abs(c["flow"]["acceleration"][0] - swapped) > 0.1
    assert c["clamped"] == c["flow"]  # All clipping/wrapping inactive here.


def test_history_resets_only_form_jumps_and_keeps_live_flow_secants(composition):
    c, f = composition, composition["expected"]
    for n in range(10):
        history = c["event_history"][n]
        if n < 2:
            assert len(history) == 1 and history[0][0] == 1
            _close(history[0][1], f["event_x"][n])
        else:
            assert history == c["original_history"][n]
        final = tuple(c["graph"].nodes[n]["epi_time_history"])
        assert final[:-1] == history
        assert final[-1][0] == 1 + DT
        _close((final[-1][1] - final[-2][1]) / DT, f["rates"][n])
    assert c["graph"].graph["_t"] == 1 + DT


def test_adaptive_coordinator_preserves_the_split_and_actual_trace_context(composition):
    c, f = composition, composition["expected"]
    assert f["branch"] == "transition"
    assert c["context"]["phase_state"][-1] == f["branch"]
    _close(c["context"]["phase_R"][-1], f["resultant"])
    assert c["context"]["phase_disr"][-1] == 0
    _close(
        (c["graph"].graph["PHASE_K_GLOBAL"], c["graph"].graph["PHASE_K_LOCAL"]),
        f["gains"],
    )
    _close(c["final"]["theta"], f["final_theta"])
    event_gap = (f["event_theta"][1] - f["event_theta"][0]) / 2
    final_gap = (c["final"]["theta"][1] - c["final"]["theta"][0]) / 2
    contraction = 1 - sum(f["gains"])
    _close(final_gap, contraction * event_gap)
    lower = (
        contraction
        * (1 - f["coupling"])
        * (f["initial_phase"][1] - f["initial_phase"][0])
        / 2
    )
    assert final_gap >= lower > 0
    for n in range(10):
        data = c["graph"].nodes[n]
        assert tuple(data["glyph_history"]) == ("AL", "IL", REQUESTS[n])
        assert data[OPERATOR_STEP_KEY] == 3 and data[U2_DEBT_KEY] == 0
        assert data["epi_kind"] == "seed"
        assert all(c["context"][key][n] == 1 for key in ("since_AL", "since_EN"))


def test_default_adaptation_uses_held_pressure_and_prepared_si(composition):
    c, f = composition, composition["expected"]
    assert not any(f["due"])  # No retuned pressure/Si gate to manufacture updates.
    assert tuple(c["graph"].nodes[n]["stable_count"] for n in range(10)) == f["counts"]
    _close(c["final"]["v"], f["adapted_v"])
    assert c["final"]["pressure"] == c["event"]["pressure"]
    assert c["final"]["si"] == c["prepared"]["si"]
    assert c["final"]["rates"] == c["flow"]["rates"]


def test_next_preparation_is_rebuilt_without_claiming_next_event_admission(composition):
    c, f = composition, composition["expected"]
    _close(c["next_readout"]["pressure"], f["next_pressure"])
    _close(c["next_readout"]["si"], f["next_si"])
    assert max(abs(a - b) for a, b in zip(f["next_pressure"], f["held"])) > 1e-3
    assert max(abs(a - b) for a, b in zip(f["next_si"], f["si"])) > 1e-3
    assert all(
        c["next_readout"][key] == c["final"][key]
        for key in ("x", "v", "theta", "rates", "acceleration")
    )


def test_terminal_remesh_archive_retains_support_order_without_a_stability_window(
    composition,
):
    graph, expected = composition["graph"], composition["expected"]
    (archive,) = graph.graph["_epi_hist"]
    assert tuple(archive) == tuple(graph) == tuple(range(10))
    _close(tuple(archive.values()), expected["endpoint"])
    assert len(composition["context"]["stable_frac"]) == 0
    assert (
        len(composition["context"]["stable_frac"])
        < graph.graph["REMESH_STABILITY_WINDOW"]
    )


def test_premature_adaptation_error_in_a_separate_exact_due_flag_control():
    # Algebraic specialization, not this source's prepared eligibility or a
    # native realization. Adaptation acts after the current held-flow interval.
    x, v, p = Q(1, 4), Q(9, 8), Q(1, 16)
    neighbor_v, mu, dt = Q(1), Q(1, 10), Q(1, 16)
    adapted_v = v + mu * (neighbor_v - v)
    correct = x + dt * v * p
    premature = x + dt * adapted_v * p
    assert premature - correct == dt * mu * (neighbor_v - v) * p != 0
    assert correct == Q(521, 2048)


def test_mean_capacity_cannot_inherit_one_unchanged_resonance(composition):
    c = composition
    initial_mean = (Q(7, 8) + Q(9, 8)) / 2
    boost = resonance.BOOST
    ideal_output = initial_mean + Q(boost) * Q(9, 16)
    assert initial_mean < ideal_output < (1 + Q(boost)) * initial_mean
    # Independent necessary outputs for ANY aggregate selected signal. More
    # retained state or a different aggregate grammar cannot add a third
    # capacity branch to this unchanged registered Resonance proposal.
    allowed = tuple(
        resonance_capacity_proposal(float(initial_mean), boost, active=active)[1]
        for active in (False, True)
    )
    actual_mean = sum(c["event"]["v"][:2]) / 2
    _close(actual_mean, float(ideal_output))
    assert allowed[0] < actual_mean < allowed[1]
    assert min(abs(actual_mean - endpoint) for endpoint in allowed) > 0.01
