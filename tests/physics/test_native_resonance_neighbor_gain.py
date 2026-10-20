"""An adjacent native Resonance prefix gains a previously excluded U3 block.

Freeze doubled C5, order 0..9, seed 17, t=1, dt=1/16, default laws, strict
grammar, supplied AL/IL/EN context and zero capacity counters. Block forms
are (A,-A,A,0,-A), A=15/16; capacities (1/4,1/4,1,1,1); phases are
(0,L+epsilon,2*epsilon,3,L-epsilon), L=Q(math.pi)/2, epsilon=1/1024.
An exact family certificate admits epsilon in [1/2048,1/512] with strict
initial U3 margins and unchanged prepared RA/RA/IL/IL/IL requests.

The first RA block uses only block four. Its accepted phase jump brings it
inside the later RA block's U3 neighborhood. The later event consequently
reads blocks zero and two, changing its form and phase action relative to
a pre-batch-subset forecast. The gained edge remains admitted after the
complete native phase coordinator; its relative-phase forecast is rational.

Both scalar/NumPy configurations and all forecasts precede the only two
reserved calls. Fraction identities retain represented defaults. An exact
cosine bound certifies the dissonant phase controller; absolute terminal
phases use an independent elementary-trig readout. Numerical comparisons
retain a fixed absolute 1e-14 budget, not a universal binary64 certificate.
No source, order or parameter-response sweep is performed.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.physics import test_native_collective_step as base
from tests.physics import test_native_resonance_live_prefix as prefix
from tests.physics import test_native_resonance_u3_boundary as boundary
from tnfr.constants import DEFAULTS
from tnfr.constants.canonical import DELTA_PHI_MAX
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER
from tnfr.operators.grammar_debt import OPERATOR_STEP_KEY, U2_DEBT_KEY

AMPLITUDE = Q(15, 16)
EPSILON, FAMILY = Q(1, 1024), (Q(1, 2048), Q(1, 512))
PI, LIMIT = Q(math.pi), Q(math.pi) / 2
FORMS = (AMPLITUDE, -AMPLITUDE, AMPLITUDE, Q(0), -AMPLITUDE)
CAPACITY = (Q(1, 4), Q(1, 4), Q(1), Q(1), Q(1))
MARKS = tuple((node, "RA" if node < 4 else "IL") for node in range(10))


def _source():
    graph = base._source(tuple(map(float, FORMS)), base.PROFILES[0])
    phases = (Q(0), LIMIT + EPSILON, 2 * EPSILON, Q(3), LIMIT - EPSILON)
    for node in graph:
        phase = phases[node // 2]
        assert Q(float(phase)) == phase
        graph.nodes[node].update(theta=float(phase), nu_f=float(CAPACITY[node // 2]))
    return graph


def _coefficients(graph):
    k, b, w = (
        base._weight(graph, "DNFR_WEIGHTS", name) for name in ("epi", "vf", "phase")
    )
    factors = graph.graph["GLYPH_FACTORS"]
    r, a, c, h = (
        Q(factors[name])
        for name in (
            "RA_epi_diff",
            "RA_vf_amplification",
            "RA_phase_coupling",
            "IL_dnfr_factor",
        )
    )
    return k, b, w, r, a, c, h


def _pressure(graph, epsilon):
    """Exact circular midpoints on this declared pair-constant phase chart."""
    k, b, w, _, _, _, _ = _coefficients(graph)
    offsets = (
        LIMIT,
        -LIMIT,
        (LIMIT + 3 - 3 * epsilon) / 2,
        (LIMIT + epsilon) / 2 - 3,
        Q(3, 2) - LIMIT + epsilon,
    )
    pressure = (
        -2 * k * AMPLITUDE + 3 * b / 8 + w / 2,
        2 * k * AMPLITUDE + 3 * b / 8 - w / 2,
        -3 * k * AMPLITUDE / 2 - 3 * b / 8 + w * offsets[2] / PI,
        w * offsets[3] / PI,
        3 * k * AMPLITUDE / 2 - 3 * b / 8 + w * offsets[4] / PI,
    )
    return pressure, offsets


def _gains(graph):
    policy = graph.graph["PHASE_ADAPT"]
    result = []
    for name, suffix in (("GLOBAL", "G"), ("LOCAL", "L")):
        current = Q(graph.graph[f"PHASE_K_{name}"])
        target = (
            Q(policy["kG_max"])
            if suffix == "G"
            else (Q(policy["kL_min"]) + Q(policy["kL_max"])) / 2
        )
        speed = Q(policy["up"] if target > current else policy["down"])
        result.append(current + speed * (target - current))
    assert all(value > 0 for value in result) and sum(result) < 1
    return tuple(result)


def _event_phases(graph, epsilon):
    _, _, _, _, _, c, _ = _coefficients(graph)
    first = c * (LIMIT - epsilon)
    later = (1 - c) * LIMIT + epsilon + c**2 * (LIMIT - epsilon) / 2
    return first, later, 2 * epsilon, Q(3), LIMIT - epsilon


def _family_certificate(graph):
    k, b, w, r, a, c, h = _coefficients(graph)
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    assert (alpha, beta, gamma) == (w, k, b)
    lower, upper = FAMILY
    assert 0 < lower <= EPSILON <= upper < LIMIT and 3 < PI
    assert Q(graph.graph.get("DELTA_PHI_MAX", DELTA_PHI_MAX)) == LIMIT
    # Every initial pair-neighbor phase separation is <pi, so all pressure
    # circular means are the stated arithmetic lifted midpoints.
    assert LIMIT + upper < 3 and 3 - 2 * lower < PI
    endpoints = tuple(_pressure(graph, epsilon) for epsilon in FAMILY)
    pressure_bounds = tuple(
        (min(values), max(values))
        for values in zip(*(value[0] for value in endpoints), strict=True)
    )
    safe_pressure_bounds = (
        (Q(57, 1000), Q(59, 1000)),
        (Q(-15, 1000), Q(-13, 1000)),
        (Q(27, 100), Q(28, 100)),
        (Q(-54, 100), Q(-53, 100)),
        (Q(21, 100), Q(22, 100)),
    )
    for (lo, hi), (safe_lo, safe_hi) in zip(
        pressure_bounds, safe_pressure_bounds, strict=True
    ):
        assert safe_lo < lo <= hi < safe_hi
    peak_lower, peak_upper = -pressure_bounds[3][1], -pressure_bounds[3][0]
    assert peak_lower > Q(1, 2)
    assert max(
        abs(value)
        for i, interval in enumerate(pressure_bounds)
        if i != 3
        for value in interval
    ) < Q(3, 10)
    ratio = (
        max(abs(value) for interval in pressure_bounds[:2] for value in interval)
        / peak_lower
    )
    assert ratio < Q(11, 100) < Q(graph.graph["SELECTOR_THRESHOLDS"]["dnfr_hi"])
    ra_si = alpha / 4 + beta / 2 + gamma * (1 - ratio), alpha / 4 + beta / 2 + gamma
    limits = graph.graph["SELECTOR_THRESHOLDS"]
    assert (
        Q(limits["si_lo"])
        < Q(33, 100)
        < ra_si[0]
        <= ra_si[1]
        < Q(34, 100)
        < Q(limits["si_hi"])
    )
    assert alpha > Q(3, 4) > Q(limits["si_hi"])
    assert all(abs(offset) < PI for _, offsets in endpoints for offset in offsets)
    # Initial included/excluded U3 gaps are L-epsilon and L+epsilon.
    # After the first pair, the formerly excluded adjacent pair is inside.
    assert 0 < c < 1 and c * (LIMIT - upper) - upper > Q(3, 25)
    assert 0 < r < Q(1, 2)
    assert (1 - r) * AMPLITUDE > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
    assert (1 - 2 * r) * AMPLITUDE > Q(1, 2)
    assert (1 - r) ** 2 * AMPLITUDE > Q(1, 2)
    gains = _gains(graph)
    for epsilon in FAMILY:
        first, later, small, _, fourth = _event_phases(graph, epsilon)
        # All inequalities are affine in epsilon; endpoints admit the family.
        assert 0 < small < first < later < fourth < 3
        separation = later - first
        local_difference = (separation + LIMIT - 3 * epsilon) / 2
        assert 0 < separation < LIMIT and 0 < local_difference < LIMIT
    # A post-event pair at 2*epsilon and 3 nearly cancels. Bound the other
    # three block phasors by their unit norms. The alternating-series tail
    # following degree eight is decreasing, giving a rigorous cosine upper.
    angle = Q(3, 2) - upper
    cosine_upper = sum(
        Q((-1) ** j, math.factorial(2 * j)) * angle ** (2 * j) for j in range(5)
    )
    assert 0 < cosine_upper < Q(3, 40)
    assert (
        (3 + 2 * cosine_upper) / 5 < Q(63, 100) < Q(graph.graph["PHASE_ADAPT"]["R_lo"])
    )
    held_minimum = min(
        (Q(1) if i < 2 else h) * min(map(abs, interval))
        for i, interval in enumerate(pressure_bounds)
    )
    assert all(lo * hi > 0 for lo, hi in pressure_bounds)
    assert held_minimum > Q(1, 100) > Q(graph.graph["EPS_DNFR_STABLE"])
    assert (
        Q(graph.graph["VF_MIN"]) < Q(1, 4) < (1 + a) / 4 < 1 < Q(graph.graph["VF_MAX"])
    )
    assert AMPLITUDE + Q(base.DT) * max(peak_upper * h, ratio * peak_upper) < 1
    event_forms = (
        (1 - 2 * r) * AMPLITUDE,
        -((1 - r) ** 2) * AMPLITUDE,
        AMPLITUDE,
        Q(0),
        -AMPLITUDE,
    )
    event_capacities = ((1 + a) / 4,) * 2 + (Q(1),) * 3
    for i, (form, capacity, interval) in enumerate(
        zip(event_forms, event_capacities, pressure_bounds, strict=True)
    ):
        retention = Q(1) if i < 2 else h
        # Each terminal form is affine in epsilon, so its absolute maximum
        # on the complete family is attained at one of these endpoints.
        assert max(
            abs(form + Q(base.DT) * capacity * retention * pressure)
            for pressure in interval
        ) < Q(24, 25)
    return {
        "pressure": pressure_bounds,
        "peak": (peak_lower, peak_upper),
        "ratio_ceiling": ratio,
        "ra_si": ra_si,
        "ordinary_si_floor": alpha,
        "phase_R_ceiling": Q(63, 100),
        "gained_clearance_floor": c * (LIMIT - upper) - upper,
        "terminal_gained_gap_ceiling": (1 - gains[0]) * LIMIT,
    }


def _forecast(graph):
    cfg = graph.graph
    _, _, _, r, a, c, h = _coefficients(graph)
    pressure, offsets = _pressure(graph, EPSILON)
    peak = -pressure[3]
    assert peak == max(map(abs, pressure))
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha * capacity + beta * (1 - abs(offset) / PI) + gamma * (1 - abs(p) / peak)
        for capacity, offset, p in zip(CAPACITY, offsets, pressure, strict=True)
    )
    first_form = (1 - 2 * r) * AMPLITUDE
    later_mean = (first_form + AMPLITUDE) / 2
    assert later_mean == (1 - r) * AMPLITUDE
    event_form = (first_form, -((1 - r) ** 2) * AMPLITUDE, AMPLITUDE, Q(0), -AMPLITUDE)
    capacity = ((1 + a) / 4,) * 2 + (Q(1),) * 3
    held = pressure[:2] + tuple(h * value for value in pressure[2:])
    rate = tuple(v * p for v, p in zip(capacity, held, strict=True))
    form = tuple(
        x + Q(base.DT) * velocity for x, velocity in zip(event_form, rate, strict=True)
    )
    event_phase = _event_phases(graph, EPSILON)
    local = tuple(
        (event_phase[(i - 1) % 5] + event_phase[(i + 1) % 5]) / 2 for i in range(5)
    )
    gains = _gains(graph)
    cos_sum = sum(math.cos(float(value)) for value in event_phase)
    sin_sum = sum(math.sin(float(value)) for value in event_phase)
    global_phase = Q(math.atan2(sin_sum, cos_sum))
    phase = tuple(
        (1 - sum(gains)) * value + gains[0] * global_phase + gains[1] * neighbor
        for value, neighbor in zip(event_phase, local, strict=True)
    )
    relative_phase = (1 - sum(gains)) * (event_phase[1] - event_phase[0]) + gains[1] * (
        local[1] - local[0]
    )
    assert phase[1] - phase[0] == relative_phase
    assert abs(relative_phase) < (1 - gains[0]) * LIMIT < LIMIT
    assert all(2 * EPSILON < value < 3 for value in phase)
    assert all(Q(cfg["EPI_MIN"]) < value < Q(cfg["EPI_MAX"]) for value in form)
    stale_event_form = -(1 - 2 * r) * AMPLITUDE
    stale_event_phase = (1 - c) * (LIMIT + EPSILON) + 2 * c * EPSILON
    assert event_form[1] - stale_event_form == -(r**2) * AMPLITUDE
    assert event_phase[1] - stale_event_phase == c * (event_phase[0] - 2 * EPSILON) / 2
    return {
        "initial_pressure": pressure,
        "peak": peak,
        "si": si,
        "later_mean": later_mean,
        "event_form": event_form,
        "event_phase": event_phase,
        "capacity": capacity,
        "pressure": held,
        "rate": rate,
        "form": form,
        "phase": phase,
        "relative_phase": relative_phase,
        "phase_R": Q(math.hypot(cos_sum, sin_sum) / 5),
        "gains": gains,
        "acceleration": tuple(
            (value - Q(old)) / Q(base.DT)
            for value, old in zip(rate, base.RATES, strict=True)
        ),
        "stale_event_form": stale_event_form,
        "stale_event_phase": stale_event_phase,
        "stale_form": stale_event_form + Q(base.DT) * rate[1],
        "form_gap": -(r**2) * AMPLITUDE,
        "event_phase_gap": c * (event_phase[0] - 2 * EPSILON) / 2,
    }


@pytest.fixture(scope="module")
def cases():
    frozen = []
    for scalar in (False, True):
        graph = _source()
        configuration = deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key
                in set(DEFAULTS) | {"GRAMMAR_REJECTION_MODE", "compute_delta_nfr"}
            }
        )
        frozen.append(
            (graph, configuration, scalar, _family_certificate(graph), _forecast(graph))
        )
    result = {}
    for graph, configuration, scalar, certificate, expected in frozen:
        case = boundary._execute(graph, configuration, scalar=scalar)
        case.update(certificate=certificate, expected=expected)
        result[scalar] = case
    return result


@pytest.mark.parametrize("scalar", (False, True))
def test_fresh_requests_are_admitted_over_the_strict_phase_family(cases, scalar):
    case, expected = cases[scalar], cases[scalar]["expected"]
    assert case["error"] is None and case["choices"] == MARKS
    assert case["prepared"] == (True,) * 10 and case["selector_cleared"] is True
    for event, channel, name in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
    ):
        for actual, predicted in zip(
            base._event(case, event)["after"][channel],
            prefix._pair_values(expected[name]),
            strict=True,
        ):
            prefix._within(actual, predicted)
    certificate = case["certificate"]
    assert certificate["peak"][0] <= expected["peak"] <= certificate["peak"][1]
    assert (
        certificate["ra_si"][0]
        <= min(expected["si"][:2])
        <= max(expected["si"][:2])
        <= certificate["ra_si"][1]
    )
    assert min(expected["si"][2:]) >= certificate["ordinary_si_floor"]
    for value, interval in zip(
        expected["initial_pressure"], certificate["pressure"], strict=True
    ):
        assert interval[0] <= value <= interval[1]


@pytest.mark.parametrize("scalar", (False, True))
def test_live_u3_gains_the_accepted_adjacent_pair_before_later_resonance(cases, scalar):
    case, expected = cases[scalar], cases[scalar]["expected"]
    assert tuple(node for node, _, _, _ in case["u3"]) == (0, 1, 2, 3)
    for node, code, selection, neighbors in case["u3"]:
        assert code == "RA" and selection.effective_limit == float(LIMIT)
        before, after = case["primitives"][node][2:]
        initial_neighbors = tuple(
            n
            for n in case["graph"].neighbors(node)
            if abs(Q(case["initial"]["phase"][n]) - Q(case["initial"]["phase"][node]))
            < LIMIT
        )
        assert initial_neighbors == ((8, 9) if node < 2 else (4, 5))
        assert neighbors == ((8, 9) if node < 2 else (0, 1, 4, 5))
        assert selection.target_phase == before["phase"][node]
        assert selection.phases == tuple(before["phase"][n] for n in neighbors)
        selected_mean = sum(Q(before["form"][n]) for n in neighbors) / len(neighbors)
        prefix._within(
            selected_mean, -AMPLITUDE if node < 2 else expected["later_mean"]
        )
        prefix._within(after["form"][node], expected["event_form"][node // 2])
        prefix._within(after["phase"][node], expected["event_phase"][node // 2])
        prefix._within(after["capacity"][node], expected["capacity"][node // 2])
        assert before["form"][node] * after["form"][node] > 0
        assert after["capacity"][node] > before["capacity"][node]
        assert before["pressure"] == after["pressure"]
    before_later = case["primitives"][2][2]
    assert (
        before_later["pressure"] == base._event(case, "pressure")["after"]["pressure"]
    )
    for node in (0, 1):
        gained_gap = abs(Q(before_later["phase"][2]) - Q(before_later["phase"][node]))
        assert gained_gap < LIMIT
        assert (
            LIMIT - gained_gap
            >= case["certificate"]["gained_clearance_floor"] - prefix.ABS_ERROR
        )


@pytest.mark.parametrize("scalar", (False, True))
def test_complete_successor_includes_the_changed_form_and_phase_action(cases, scalar):
    case, expected = cases[scalar], cases[scalar]["expected"]
    assert tuple((node, glyph) for node, glyph, _, _ in case["primitives"]) == MARKS
    for name in ("form", "capacity", "pressure", "rate", "acceleration", "si", "phase"):
        for actual, predicted in zip(
            case["final"][name], prefix._pair_values(expected[name]), strict=True
        ):
            prefix._within(actual, predicted)
    for name in ("form", "phase"):
        for actual, predicted in zip(
            base._event(case, "glyphs")["after"][name],
            prefix._pair_values(expected[f"event_{name}"]),
            strict=True,
        ):
            prefix._within(actual, predicted)
    integration = base._event(case, "integration")
    assert integration["before"]["capacity"] == case["final"]["capacity"]
    assert integration["before"]["pressure"] == integration["after"]["pressure"]
    for node, _, before, after in case["primitives"]:
        assert before["edges"] == after["edges"]
        for name in ("form", "phase", "capacity", "pressure"):
            assert tuple(before[name][n] for n in case["graph"] if n != node) == tuple(
                after[name][n] for n in case["graph"] if n != node
            )


@pytest.mark.parametrize("scalar", (False, True))
def test_stale_subset_misses_the_frozen_form_and_phase_discriminator(cases, scalar):
    case, expected = cases[scalar], cases[scalar]["expected"]
    assert expected["form_gap"] < -Q(1, 50)
    assert expected["event_phase_gap"] > Q(4, 1000)
    for node in (2, 3):
        after = case["primitives"][node][3]
        prefix._within(
            Q(after["form"][node]) - expected["stale_event_form"], expected["form_gap"]
        )
        prefix._within(
            Q(after["phase"][node]) - expected["stale_event_phase"],
            expected["event_phase_gap"],
        )
        prefix._within(
            Q(case["final"]["form"][node]) - expected["stale_form"],
            expected["form_gap"],
        )


@pytest.mark.parametrize("scalar", (False, True))
def test_gained_neighbor_remains_admitted_after_dissonant_phase_coordination(
    cases, scalar
):
    case, expected = cases[scalar], cases[scalar]["expected"]
    history = case["graph"].graph["history"]
    assert tuple(history["phase_state"]) == ("dissonant",)
    assert tuple(history["phase_disr"]) == (0.0,)
    prefix._within(history["phase_R"][0], expected["phase_R"])
    assert history["phase_R"][0] < float(case["certificate"]["phase_R_ceiling"])
    for suffix, gain in zip(("G", "L"), expected["gains"], strict=True):
        prefix._within(history[f"phase_k{suffix}"][0], gain)
    assert all(2 * EPSILON < Q(value) < 3 for value in case["final"]["phase"])
    for first in (0, 1):
        for later in (2, 3):
            gap = Q(case["final"]["phase"][later]) - Q(case["final"]["phase"][first])
            prefix._within(
                gap, expected["relative_phase"], allowance=2 * prefix.ABS_ERROR
            )
            assert abs(gap) < case["certificate"]["terminal_gained_gap_ceiling"] < LIMIT


@pytest.mark.parametrize("scalar", (False, True))
def test_defaults_memory_and_adaptation_are_not_replaced_by_the_u3_gain(cases, scalar):
    case, graph = cases[scalar], cases[scalar]["graph"]
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
    assert (
        case["final"]["time"] == 1 + base.DT and case["final"]["counters"] == (0,) * 10
    )
    assert (
        case["final"]["edges"] == case["initial"]["edges"]
        and len(case["final"]["edges"]) == 20
    )
    assert not any(
        event["name"] in {"capacity_write", "network_remesh"}
        for event in case["events"]
    )
    assert (
        base._event(case, "capacity")["before"]["si"]
        == base._event(case, "si")["after"]["si"]
    )
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )
    history = graph.graph["history"]
    assert not history.get("stable_frac") and "_REMESH_META" not in graph.graph
    assert "ra_metrics" not in graph.graph and "_ra_c_tracking" not in graph.graph
    for node, data in graph.nodes(data=True):
        assert data[OPERATOR_STEP_KEY] == 4 and data[U2_DEBT_KEY] == int(node // 2 == 1)
        assert data["epi_kind"] == "seed" and tuple(data["glyph_history"]) == (
            "EN",
            MARKS[node][1],
        )
        assert history["since_AL"][node] == history["since_EN"][node] == 1
        assert case["final"]["temporal"][node] == (
            (1.0, base._event(case, "integration")["before"]["form"][node]),
            (1 + base.DT, case["final"]["form"][node]),
        )
