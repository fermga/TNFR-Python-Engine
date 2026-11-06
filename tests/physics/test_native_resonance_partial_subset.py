"""A native Resonance call retains its strict compatible-neighbor subset.

Freeze doubled C5, order 0..9, seed 17, t=1, dt=1/16 and default laws. Form
means are A*(-7/8,1,-1,-1,1), A=15/16; capacity means are (1/4,1,1,1,1).
The first pair has phase 1+pi/2; pairs one and four have phases 1+/-epsilon
and forms A+/-s*rho, epsilon=rho=1/1024; other phases equal one. Both signs
and scalar/NumPy paths are frozen before the four reserved native calls.

Exact rational algebra predicts fresh pressure, Si, selected requests, the
strict U3 subset and the accepted form/capacity response. Opposite form/phase
associations have equal target pressure and Si, but different propagated form.
A common phase cone predicts controller admission and the terminal U3 domain.
Ideal-real enclosures on represented coefficients and finite binary64 checks
with absolute tolerance 1e-14 remain distinct. No subsequent call, autonomous
selection law, or formation of a collective NFR is asserted.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product

import pytest

from tests.physics import test_native_collective_step as base
from tests.physics import test_native_resonance_u3_boundary as boundary
from tnfr.constants import DEFAULTS
from tnfr.constants.canonical import DELTA_PHI_MAX
from tnfr.mathematics._rational_interval import I
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER
from tnfr.operators.grammar_debt import OPERATOR_STEP_KEY, U2_DEBT_KEY

A, RHO, EPSILON = Q(15, 16), Q(1, 1024), Q(1, 1024)
PI, LIMIT = Q(math.pi), Q(math.pi) / 2
MEANS = tuple(A * value for value in (Q(-7, 8), Q(1), Q(-1), Q(-1), Q(1)))
CAPACITIES = (Q(1, 4), Q(1), Q(1), Q(1), Q(1))
KEYS = tuple(product((-1, 1), (False, True)))
MARKS = tuple((node, "RA" if node < 2 else "IL") for node in range(10))
ABS_ERROR = Q(1, 10**14)


def _sigma(node):
    return -1 if node % 2 == 0 else 1


def _source(sign):
    graph = base._source(tuple(map(float, MEANS)), base.PROFILES[0])
    for node in graph:
        block = node // 2
        form, phase = MEANS[block], Q(1)
        if block == 0:
            phase += LIMIT
        elif block in (1, 4):
            form += sign * _sigma(node) * RHO
            phase += _sigma(node) * EPSILON
        assert Q(float(form)) == form and Q(float(phase)) == phase
        graph.nodes[node].update(
            EPI=float(form), theta=float(phase), nu_f=float(CAPACITIES[block])
        )
    return graph


def _coefficients(graph):
    k, b, w = (
        base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf", "phase")
    )
    factors = graph.graph["GLYPH_FACTORS"]
    r, c, a, h = (
        Q(factors[key])
        for key in (
            "RA_epi_diff",
            "RA_phase_coupling",
            "RA_vf_amplification",
            "IL_dnfr_factor",
        )
    )
    return k, b, w, r, c, a, h


def _common_bounds(graph):
    """Admit every paired form contrast t in [-rho,rho], without its sign."""
    cfg = graph.graph
    k, b, w, r, c, a, h = _coefficients(graph)
    z = 15 * k * A / 8 + 3 * b / 4 - w / 2
    peak = k * A
    neighbor_center = -peak - z / 2
    neighbor_radius = k * RHO + w * EPSILON / PI
    assert Q(-14, 1000) < z < Q(-13, 1000)
    assert abs(neighbor_center) + neighbor_radius < peak
    assert abs(neighbor_center) - neighbor_radius > Q(16, 100)
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", key) for key in ("alpha", "beta", "gamma")
    )
    target_si = alpha / 4 + beta / 2 + gamma * (1 - abs(z) / peak)
    other_si_floor = alpha + beta * (Q(3, 4) - EPSILON / PI)
    thresholds = cfg["SELECTOR_THRESHOLDS"]
    margins = (
        target_si - Q(thresholds["si_lo"]),
        Q(thresholds["si_hi"]) - target_si,
        Q(thresholds["dnfr_hi"]) - abs(z) / peak,
        other_si_floor - Q(thresholds["si_hi"]),
    )
    assert min(margins) > Q(1, 20)
    assert Q(cfg.get("DELTA_PHI_MAX", DELTA_PHI_MAX)) == LIMIT
    assert 0 < EPSILON < LIMIT and LIMIT + EPSILON < PI
    # The selected odd neighbors have gap L-epsilon; excluded even neighbors
    # have gap L+epsilon. The selected forms stay strictly above the trigger.
    assert A - RHO > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
    target_event = I(
        (1 - r) * MEANS[0] + r * (A - RHO), (1 - r) * MEANS[0] + r * (A + RHO)
    )
    assert Q(cfg["EPI_MIN"]) < target_event.lo < target_event.hi < 0
    target_capacity = (1 + a) / 4
    dt = Q(base.DT)
    form_bounds = []
    for node in range(10):
        block, sigma = node // 2, _sigma(node)
        if block == 0:
            bound = target_event + dt * target_capacity * z
        elif block in (1, 4):
            center = A + dt * h * (neighbor_center - sigma * w * EPSILON / PI)
            radius = (1 - dt * h * k) * RHO
            assert radius > 0
            bound = I(center - radius, center + radius)
        else:
            bound = I(-A + dt * h * peak)
        assert Q(cfg["EPI_MIN"]) < bound.lo <= bound.hi < Q(cfg["EPI_MAX"])
        form_bounds.append(bound)
    assert Q(cfg["VF_MIN"]) < target_capacity < Q(cfg["VF_MAX"])
    assert min(abs(z), h * peak, h * (abs(neighbor_center) - neighbor_radius)) > Q(
        cfg["EPS_DNFR_STABLE"]
    )

    gstar = (1 - c) * LIMIT + c * EPSILON
    post_event_diameter = gstar + EPSILON
    assert 0 < post_event_diameter < LIMIT
    # Projection onto phase one gives R >= (4+4*cos(epsilon))/10:
    # the two RA phases contribute positively since 0<gstar<pi/2.
    order_floor = Q(4, 5) - EPSILON**2 / 5
    policy = cfg["PHASE_ADAPT"]
    assert order_floor > Q(policy["R_hi"])
    gains = tuple(
        Q(cfg[f"PHASE_K_{name}"])
        + Q(policy["down"]) * (Q(policy[f"k{suffix}_min"]) - Q(cfg[f"PHASE_K_{name}"]))
        for name, suffix in (("GLOBAL", "G"), ("LOCAL", "L"))
    )
    assert 0 < gains[0] and 0 <= gains[1] and sum(gains) < 1
    return {
        "target_pressure": z,
        "target_si": target_si,
        "peak": peak,
        "margins": margins,
        "form": tuple(form_bounds),
        "phase": I(1 - EPSILON, 1 + gstar),
        "diameter": (1 - gains[0]) * post_event_diameter,
        "gains": gains,
        "order_floor": order_floor,
    }


def _forecast(graph, sign, bounds):
    """Independent full-state algebra, with no pressure/selector/operator call."""
    k, b, w, r, c, a, h = _coefficients(graph)
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", key) for key in ("alpha", "beta", "gamma")
    )
    pressure, offsets, event_form, event_phase, capacity = [], [], [], [], []
    z = bounds["target_pressure"]
    for node in range(10):
        block, sigma = node // 2, _sigma(node)
        if block == 0:
            p, offset = z, -LIMIT
            form = (1 - r) * MEANS[0] + r * (A + sign * RHO)
            phase, vf = 1 + (1 - c) * LIMIT + c * EPSILON, (1 + a) / 4
        elif block in (1, 4):
            p = -k * A - z / 2 - sigma * (k * sign * RHO + w * EPSILON / PI)
            offset = LIMIT / 2 - sigma * EPSILON
            form, phase, vf = A + sign * sigma * RHO, 1 + sigma * EPSILON, Q(1)
        else:
            p, offset, form, phase, vf = k * A, Q(0), -A, Q(1), Q(1)
        pressure.append(p)
        offsets.append(offset)
        event_form.append(form)
        event_phase.append(phase)
        capacity.append(vf)
    peak = max(map(abs, pressure))
    assert peak == bounds["peak"]
    si = tuple(
        alpha * CAPACITIES[node // 2]
        + beta * (1 - abs(offsets[node]) / PI)
        + gamma * (1 - abs(value) / peak)
        for node, value in enumerate(pressure)
    )
    assert si[0] == si[1] == bounds["target_si"]
    held = tuple(
        value if node < 2 else h * value for node, value in enumerate(pressure)
    )
    rate = tuple(vf * p for vf, p in zip(capacity, held, strict=True))
    form = tuple(x + Q(base.DT) * v for x, v in zip(event_form, rate, strict=True))
    for value, interval in zip(form, bounds["form"], strict=True):
        assert interval.lo <= value <= interval.hi
    return {
        "initial_pressure": tuple(pressure),
        "si": si,
        "event_form": tuple(event_form),
        "event_phase": tuple(event_phase),
        "capacity": tuple(capacity),
        "pressure": held,
        "rate": rate,
        "form": form,
        "acceleration": tuple(
            (value - Q(base.RATES[node // 2])) / Q(base.DT)
            for node, value in enumerate(rate)
        ),
        "selected_mean": A + sign * RHO,
        "association_gap": 2 * r * RHO,
    }


def _within(value, interval):
    assert interval.lo - ABS_ERROR <= Q(value) <= interval.hi + ABS_ERROR


@pytest.fixture(scope="module")
def cases():
    # Freeze every source, configuration and readout before any reserved call.
    frozen = []
    for sign, scalar in KEYS:
        graph = _source(sign)
        configuration = deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key
                in set(DEFAULTS) | {"GRAMMAR_REJECTION_MODE", "compute_delta_nfr"}
            }
        )
        bounds = _common_bounds(graph)
        frozen.append(
            (graph, configuration, sign, scalar, bounds, _forecast(graph, sign, bounds))
        )
    result = {}
    for graph, configuration, sign, scalar, bounds, expected in frozen:
        case = boundary._execute(graph, configuration, scalar=scalar)
        case.update(bounds=bounds, expected=expected)
        result[sign, scalar] = case
    return result


@pytest.mark.parametrize("key", KEYS)
def test_fresh_selection_and_live_partial_subset_have_independent_predictions(
    cases, key
):
    case, expected = cases[key], cases[key]["expected"]
    assert case["error"] is None and case["choices"] == MARKS
    for event, channel, forecast in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
    ):
        assert base._event(case, event)["after"][channel] == pytest.approx(
            tuple(map(float, expected[forecast])), abs=float(ABS_ERROR), rel=0
        )
    assert tuple(node for node, _, _, _ in case["u3"]) == (0, 1)
    for node, glyph, selection, neighbors in case["u3"]:
        assert glyph == "RA" and set(neighbors) == {3, 9}
        assert len(neighbors) == 2 < len(tuple(case["graph"].neighbors(node)))
        assert selection.phases == (float(1 + EPSILON),) * 2
        assert selection.target_phase == float(1 + LIMIT)
        assert selection.effective_limit == float(LIMIT)
        assert set(case["graph"].neighbors(node)) - set(neighbors) == {2, 8}
    assert tuple((node, glyph) for node, glyph, _, _ in case["primitives"]) == MARKS


@pytest.mark.parametrize("key", KEYS)
def test_resonance_uses_only_live_selected_values_and_preserves_the_other_target(
    cases, key
):
    case, expected = cases[key], cases[key]["expected"]
    for node, glyph, before, after in case["primitives"]:
        assert before["edges"] == after["edges"]
        for channel in ("form", "phase", "capacity", "pressure"):
            assert tuple(
                before[channel][other] for other in range(10) if other != node
            ) == tuple(after[channel][other] for other in range(10) if other != node)
        if glyph == "RA":
            assert after["pressure"] == before["pressure"]
            for channel in ("form", "phase", "capacity"):
                forecast = expected[
                    channel if channel == "capacity" else f"event_{channel}"
                ][node]
                assert after[channel][node] == pytest.approx(
                    float(forecast), abs=float(ABS_ERROR), rel=0
                )
            assert before["form"][node] < 0 and after["form"][node] < 0
            assert case["graph"].nodes[node]["epi_kind"] == "seed"
        else:
            assert before["form"] == after["form"] and before["phase"] == after["phase"]
    assert case["primitives"][1][2]["phase"][0] != case["initial"]["phase"][0]
    assert case["u3"][0][2].phases == case["u3"][1][2].phases


@pytest.mark.parametrize("key", KEYS)
def test_complete_flow_uses_post_ra_capacity_and_held_multichannel_pressure(cases, key):
    case, expected = cases[key], cases[key]["expected"]
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
    for channel in ("form", "pressure", "capacity", "rate", "acceleration", "si"):
        assert case["final"][channel] == pytest.approx(
            tuple(map(float, expected[channel])), abs=float(ABS_ERROR), rel=0
        )
    integration = base._event(case, "integration")
    assert integration["before"]["pressure"] == integration["after"]["pressure"]
    assert integration["before"]["capacity"] == case["final"]["capacity"]
    assert case["final"]["counters"] == (0,) * 10
    assert (
        base._event(case, "capacity")["before"]["si"]
        == base._event(case, "si")["after"]["si"]
    )
    for value, interval in zip(
        case["final"]["form"], case["bounds"]["form"], strict=True
    ):
        _within(value, interval)
    for node in (0, 1):
        old_capacity_rate = (
            Q(case["initial"]["capacity"][node]) * expected["pressure"][node]
        )
        assert abs(Q(case["final"]["rate"][node]) - old_capacity_rate) > 100 * ABS_ERROR


@pytest.mark.parametrize("scalar", (False, True))
def test_same_target_pressure_and_si_do_not_determine_partial_subset_propagation(
    cases, scalar
):
    plus, minus = cases[1, scalar], cases[-1, scalar]
    assert plus["initial"]["phase"] == minus["initial"]["phase"]
    assert plus["initial"]["capacity"] == minus["initial"]["capacity"]
    assert plus["bounds"] == minus["bounds"]
    for nodes in ((2, 3), (8, 9), (2, 3, 8, 9)):
        assert sorted(plus["initial"]["form"][node] for node in nodes) == sorted(
            minus["initial"]["form"][node] for node in nodes
        )
    for name, channel in (("pressure", "pressure"), ("si", "si")):
        left, right = (
            base._event(plus, name)["after"][channel],
            base._event(minus, name)["after"][channel],
        )
        assert left[:2] == right[:2]
    assert (
        plus["expected"]["selected_mean"] - minus["expected"]["selected_mean"]
        == 2 * RHO
    )
    predicted_gap = plus["expected"]["association_gap"]
    assert predicted_gap > Q(1, 10000)
    for node in (0, 1):
        for left, right in (
            (
                plus["primitives"][node][3]["form"][node],
                minus["primitives"][node][3]["form"][node],
            ),
            (plus["final"]["form"][node], minus["final"]["form"][node]),
        ):
            assert abs(Q(left) - Q(right) - predicted_gap) <= 2 * ABS_ERROR
    assert plus["final"]["phase"] == minus["final"]["phase"]
    assert plus["final"]["capacity"] == minus["final"]["capacity"]


@pytest.mark.parametrize("key", KEYS)
def test_phase_cone_admits_stable_controller_and_terminal_full_u3_geometry(cases, key):
    case, bounds = cases[key], cases[key]["bounds"]
    history = case["graph"].graph["history"]
    assert tuple(history["phase_state"]) == ("stable",)
    assert tuple(history["phase_disr"]) == (0.0,)
    assert Q(history["phase_R"][0]) + ABS_ERROR >= bounds["order_floor"]
    for suffix, gain in zip(("G", "L"), bounds["gains"], strict=True):
        assert abs(Q(history[f"phase_k{suffix}"][0]) - gain) <= ABS_ERROR
    phases = case["final"]["phase"]
    for value in phases:
        _within(value, bounds["phase"])
    assert (
        max(map(Q, phases)) - min(map(Q, phases)) <= bounds["diameter"] + 2 * ABS_ERROR
    )
    assert bounds["diameter"] + 2 * ABS_ERROR < LIMIT
    # The declared lift is contained in (0,2pi); all pair distances here are
    # less than pi/2. This geometry does not select a later RA occurrence.
    for left, right in case["graph"].edges:
        assert abs(phases[left] - phases[right]) < float(LIMIT)


@pytest.mark.parametrize("key", KEYS)
def test_provenance_retains_default_policies_live_memory_and_remesh_abstention(
    cases, key
):
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
    assert case["prepared"] == (True,) * 10 and case["selector_cleared"] is True
    assert "glyph_selector" not in graph.graph and "integrator" not in graph.graph
    assert tuple(graph) == tuple(range(10)) and graph.graph["RANDOM_SEED"] == 17
    assert case["initial"]["temporal"] == ((),) * 10
    assert case["initial"]["counters"] == (0,) * 10
    assert case["initial"]["ordinal"] is None and case["final"]["ordinal"] == 1
    assert case["final"]["time"] == 1 + base.DT
    assert len(case["final"]["edges"]) == 20
    assert case["final"]["edges"] == case["initial"]["edges"]
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )
    history = graph.graph["history"]
    assert not history.get("stable_frac") and "_REMESH_META" not in graph.graph
    integration = base._event(case, "integration")
    for node, data in graph.nodes(data=True):
        assert history["since_AL"][node] == history["since_EN"][node] == 1
        assert data["epi_kind"] == "seed"
        assert tuple(data["glyph_history"]) == ("EN", MARKS[node][1])
        assert data[OPERATOR_STEP_KEY] == 4 and data[U2_DEBT_KEY] == 0
        assert case["final"]["temporal"][node] == (
            (1.0, integration["before"]["form"][node]),
            (1 + base.DT, case["final"]["form"][node]),
        )
