"""Prospective native Resonance with primitive circular-phase dispersion.

Freeze doubled C5, order 0..9, seed 17, t=1, dt=1/16 and defaults. Pair form
means are (3/8,1/8,1/4,1/8,1/8), capacities (1/4,1,1,1,1), form contrasts
(-1)**i * 2**-20, and initial capacity counters zero. Two preparations have
opposite phase contrasts +/-(-1)**i / 2048 about 1+1/2048. All predictions
and common enclosures are frozen before any of the four reserved native calls
(two preparations, scalar/NumPy).

Conjugate phase pairs give exact common circular means in the ideal-real
model. Fraction algebra retains represented default coefficients, including
the represented pi pressure normalizer; actual trig, wrap and arithmetic
defects are compared separately against the fixed absolute 1e-14 budget.
The coarse enclosure allows arbitrary primitive phases in [1,1+1/1024] and
unknown form contrasts up to 2**-20. It is a conditional one-call enclosure,
not a universal floating-point certificate, occurrence law or formed NFR.
"""

import math
from fractions import Fraction as Q
from functools import wraps
from itertools import product

import pytest

import tnfr.operators as operators
from tests.physics import test_native_collective_mean as mean
from tests.physics import test_native_collective_step as base
from tnfr.constants import DEFAULTS
from tnfr.constants.canonical import DELTA_PHI_MAX
from tnfr.mathematics._rational_interval import I
from tnfr.operators.grammar_debt import OPERATOR_STEP_KEY, U2_DEBT_KEY

MEANS = (Q(3, 8), Q(1, 8), Q(1, 4), Q(1, 8), Q(1, 8))
CAPACITIES = (Q(1, 4), Q(1), Q(1), Q(1), Q(1))
RHO, DIAMETER = Q(1, 2**20), Q(1, 1024)
CENTER = 1 + DIAMETER / 2
FORM_CONTRASTS = tuple((-1) ** i * RHO for i in range(5))
MARKS = tuple((node, "RA" if node < 2 else "IL") for node in range(10))
ABS_ERROR = Q(1, 10**14)


def _neighbor(values, block):
    return (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2


def _source(sign):
    graph = base._source(tuple(map(float, MEANS)), base.PROFILES[0])
    contrasts = tuple(sign * (-1) ** i * DIAMETER / 2 for i in range(5))
    forms = mean._fine(MEANS, FORM_CONTRASTS)
    phases = mean._fine((CENTER,) * 5, contrasts)
    for node, (form, phase) in enumerate(zip(forms, phases, strict=True)):
        graph.nodes[node].update(
            EPI=float(form), theta=float(phase), nu_f=float(CAPACITIES[node // 2])
        )
    return graph, contrasts


def _coefficients(graph):
    cfg = graph.graph
    k, b, a = (
        base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf", "phase")
    )
    factors = cfg["GLYPH_FACTORS"]
    mix, boost, coupling, retention = (
        Q(factors[key])
        for key in (
            "RA_epi_diff",
            "RA_vf_amplification",
            "RA_phase_coupling",
            "IL_dnfr_factor",
        )
    )
    pressure_centers = tuple(
        k * (_neighbor(MEANS, i) - MEANS[i])
        + b * (_neighbor(CAPACITIES, i) - CAPACITIES[i])
        for i in range(5)
    )
    capacity = ((1 + boost) * CAPACITIES[0], *CAPACITIES[1:])
    event_means = ((1 - mix) * MEANS[0] + mix * _neighbor(MEANS, 0), *MEANS[1:])
    form_factors = (1 - mix, Q(1), Q(1), Q(1), Q(1))
    pressure_factors = (Q(1), retention, retention, retention, retention)
    phase_factors = (1 - coupling, Q(1), Q(1), Q(1), Q(1))
    policy = cfg["PHASE_ADAPT"]
    gains = tuple(
        Q(cfg[f"PHASE_K_{name}"])
        + Q(policy["down"]) * (Q(policy[f"k{suffix}_min"]) - Q(cfg[f"PHASE_K_{name}"]))
        for name, suffix in (("GLOBAL", "G"), ("LOCAL", "L"))
    )
    return (
        k,
        a,
        pressure_centers,
        capacity,
        event_means,
        form_factors,
        pressure_factors,
        phase_factors,
        gains,
    )


def _coarse_forecast(graph):
    """Use only the common box and retained configuration, never a fine lift."""
    (
        k,
        a,
        centers,
        capacities,
        event_means,
        form_factors,
        factors,
        phase_factors,
        gains,
    ) = _coefficients(graph)
    cfg, dt = graph.graph, Q(base.DT)
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", key) for key in ("alpha", "beta", "gamma")
    )
    # The convex phase cone has width D < pi/2, with represented pressure
    # normalizer P > 3. Twin nodes have identical neighbor sets.
    assert 3 < Q(math.pi) and 0 < DIAMETER < Q(cfg.get("DELTA_PHI_MAX", DELTA_PHI_MAX))
    error = k * RHO + a * DIAMETER / 3
    pressure = tuple(I(center - error, center + error) for center in centers)
    peak_floor = abs(centers[2]) - error
    assert peak_floor > 0
    ratio_ceiling = pressure[0].abs_max / peak_floor
    low = alpha / 4 + beta * (1 - DIAMETER / 3) + gamma * (1 - ratio_ceiling)
    high = alpha / 4 + beta + gamma
    thresholds = cfg["SELECTOR_THRESHOLDS"]
    assert Q(thresholds["si_lo"]) < low < high < Q(thresholds["si_hi"])
    assert ratio_ceiling < Q(thresholds["dnfr_hi"])
    ordinary_floor = alpha + beta * (1 - DIAMETER / 3)
    assert ordinary_floor > Q(thresholds["si_hi"])
    assert 1 - DIAMETER**2 / 8 > Q(cfg["PHASE_ADAPT"]["R_hi"])
    assert 0 < gains[0] and 0 <= gains[1] and sum(gains) < 1
    means, radii = [], []
    for center, capacity, event_mean, event_factor, factor in zip(
        centers, capacities, event_means, form_factors, factors, strict=True
    ):
        clock_factor = dt * capacity * factor
        mean_radius = clock_factor * a * DIAMETER / 3
        next_center = event_mean + clock_factor * center
        means.append(I(next_center - mean_radius, next_center + mean_radius))
        radii.append(
            abs(event_factor - clock_factor * k) * RHO + clock_factor * a * DIAMETER / 6
        )
    assert all(
        Q(cfg["EPI_MIN"]) < value.lo - radius < value.hi + radius < Q(cfg["EPI_MAX"])
        for value, radius in zip(means, radii, strict=True)
    )
    assert all(Q(cfg["VF_MIN"]) < value < Q(cfg["VF_MAX"]) for value in capacities)
    # Even an unresolved qualification cannot make an incoming zero counter
    # reach the configured five-call write threshold in this single call.
    assert cfg["VF_ADAPT_TAU"] > 1
    stable = (False,) + tuple(
        (factor * value).abs_max <= Q(cfg["EPS_DNFR_STABLE"])
        for factor, value in zip(factors[1:], pressure[1:], strict=True)
    )
    for i in (1, 2, 3):
        assert abs(factors[i] * pressure[i]).lo > Q(cfg["EPS_DNFR_STABLE"])
    assert stable == (False, False, False, False, True)
    return {
        "means": tuple(means),
        "form_radii": tuple(radii),
        "capacity": capacities,
        "phase": I(1, 1 + DIAMETER),
        "diameter": (1 - gains[0]) * DIAMETER,
        "phase_radii": tuple(
            (1 - sum(gains)) * factor * DIAMETER / 2 for factor in phase_factors
        ),
        "counters": tuple(int(value) for value in stable for _ in range(2)),
        "pressure": pressure,
        "ra_si": I(low, high),
        "il_si_floor": ordinary_floor,
    }


def _forecast(graph, phase_contrasts):
    """Exact ideal-real algebra on represented coefficients and symmetric lifts."""
    (
        k,
        a,
        centers,
        capacities,
        event_means,
        form_factors,
        factors,
        phase_factors,
        gains,
    ) = _coefficients(graph)
    dt, normalizer = Q(base.DT), Q(math.pi)
    pressure_contrasts = tuple(
        -k * d - a * t / normalizer
        for d, t in zip(FORM_CONTRASTS, phase_contrasts, strict=True)
    )
    pressure = mean._fine(centers, pressure_contrasts)
    peak = max(map(abs, pressure))
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", key) for key in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha * CAPACITIES[node // 2]
        + beta * (1 - abs(phase_contrasts[node // 2]) / normalizer)
        + gamma * (1 - abs(value) / peak)
        for node, value in enumerate(pressure)
    )
    held = tuple(factors[node // 2] * value for node, value in enumerate(pressure))
    event_contrasts = tuple(
        factor * value
        for factor, value in zip(form_factors, FORM_CONTRASTS, strict=True)
    )
    event_form = mean._fine(event_means, event_contrasts)
    rate = tuple(capacities[node // 2] * value for node, value in enumerate(held))
    form = tuple(
        value + dt * velocity for value, velocity in zip(event_form, rate, strict=True)
    )
    event_phase = mean._fine(
        (CENTER,) * 5,
        tuple(
            factor * t for factor, t in zip(phase_factors, phase_contrasts, strict=True)
        ),
    )
    phase = mean._fine(
        (CENTER,) * 5,
        tuple(
            (1 - sum(gains)) * factor * t
            for factor, t in zip(phase_factors, phase_contrasts, strict=True)
        ),
    )
    counts = tuple(
        int(
            abs(value) <= Q(graph.graph["EPS_DNFR_STABLE"])
            and s >= Q(graph.graph["SELECTOR_THRESHOLDS"]["si_hi"])
        )
        for value, s in zip(held, si, strict=True)
    )
    return {
        "initial_pressure": pressure,
        "si": si,
        "pressure": held,
        "event_form": event_form,
        "event_phase": event_phase,
        "form": form,
        "phase": phase,
        "capacity": mean._fine(capacities, (Q(0),) * 5),
        "rate": rate,
        "acceleration": tuple(
            (value - Q(base.RATES[node // 2])) / dt for node, value in enumerate(rate)
        ),
        "counters": counts,
        "gains": gains,
    }


def _within(actual, expected, *, allowance=ABS_ERROR):
    if isinstance(expected, I):
        assert expected.lo - allowance <= Q(actual) <= expected.hi + allowance
    else:
        assert abs(Q(actual) - expected) <= allowance


@pytest.fixture(scope="module")
def cases():
    # All four configurations, ideal endpoints and coarse bounds precede the
    # first reserved runtime response. A failed prediction is not recalibrated.
    frozen = []
    for sign, scalar in product((1, -1), (False, True)):
        graph, phases = _source(sign)
        bound, expected = _coarse_forecast(graph), _forecast(graph, phases)
        for i, (value, radius) in enumerate(
            zip(bound["means"], bound["form_radii"], strict=True)
        ):
            _within(mean._means(expected["form"])[i], value, allowance=Q(0))
            assert abs(mean._contrasts(expected["form"])[i]) <= radius
        frozen.append((graph, sign, scalar, expected, bound))
    result = {}
    for graph, sign, scalar, expected, bound in frozen:
        u3, primitives = [], []
        original_u3, original_glyph = (
            operators._runtime_u3_neighbors,
            base.selectors.apply_glyph,
        )

        @wraps(original_u3)
        def capture_u3(node, operator_code):
            selection, neighbors = original_u3(node, operator_code)
            u3.append(
                (
                    node.n,
                    operator_code,
                    selection,
                    tuple(neighbor.n for neighbor in neighbors),
                )
            )
            return selection, neighbors

        @wraps(original_glyph)
        def capture_glyph(graph, node, glyph, **kwargs):
            before = base._snapshot(graph)
            returned = original_glyph(graph, node, glyph, **kwargs)
            primitives.append((node, glyph, before, base._snapshot(graph)))
            return returned

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(operators, "_runtime_u3_neighbors", capture_u3)
            patch.setattr(base.selectors, "apply_glyph", capture_glyph)
            case = base._execute(graph, scalar=scalar)
        case.update(
            expected=expected, bound=bound, u3=tuple(u3), primitives=tuple(primitives)
        )
        result[sign, scalar] = case
    return result


@pytest.mark.parametrize("key", tuple(product((1, -1), (False, True))))
def test_native_selection_and_u3_are_admitted_from_the_same_common_box(cases, key):
    case = cases[key]
    assert case["marks"] == MARKS
    assert tuple(node for node, _, _, _ in case["u3"]) == (0, 1)
    for node, code, selection, neighbors in case["u3"]:
        assert code == "RA"
        assert neighbors == tuple(case["graph"].neighbors(node))
        assert len(neighbors) == 4
        assert selection.effective_limit == case["configuration"].get(
            "DELTA_PHI_MAX", DELTA_PHI_MAX
        )
        assert all(
            abs(Q(phase) - Q(selection.target_phase)) <= DIAMETER
            for phase in selection.phases
        )
        assert selection.target_phase == case["primitives"][node][2]["phase"][node]
        assert selection.phases == tuple(
            case["primitives"][node][2]["phase"][neighbor] for neighbor in neighbors
        )
    si = base._event(case, "si")["after"]["si"]
    for value in si[:2]:
        _within(value, case["bound"]["ra_si"])
    assert min(si[2:]) >= float(case["bound"]["il_si_floor"]) - float(ABS_ERROR)


@pytest.mark.parametrize("key", tuple(product((1, -1), (False, True))))
def test_frozen_joint_endpoint_includes_nonzero_primitive_phase_effects(cases, key):
    case, expected = cases[key], cases[key]["expected"]
    for name in ("form", "phase", "capacity", "pressure", "si", "rate", "acceleration"):
        for actual, predicted in zip(case["final"][name], expected[name], strict=True):
            _within(actual, predicted)
    for actual, predicted in zip(
        base._event(case, "pressure")["after"]["pressure"],
        expected["initial_pressure"],
        strict=True,
    ):
        _within(actual, predicted)
    after = base._event(case, "glyphs")["after"]
    for name in ("form", "phase"):
        for actual, predicted in zip(
            after[name], expected[f"event_{name}"], strict=True
        ):
            _within(actual, predicted)
    assert after["capacity"][0] > case["initial"]["capacity"][0]
    assert after["form"][:2] != case["initial"]["form"][:2]
    assert after["phase"][:2] != case["initial"]["phase"][:2]
    assert case["final"]["phase"] != after["phase"]
    assert case["final"]["counters"] == expected["counters"]


@pytest.mark.parametrize("key", tuple(product((1, -1), (False, True))))
def test_common_enclosure_contains_means_internal_contrasts_and_phase(cases, key):
    case, bound = cases[key], cases[key]["bound"]
    means, contrasts = mean._means(case["final"]["form"]), mean._contrasts(
        case["final"]["form"]
    )
    for value, interval, contrast, radius in zip(
        means, bound["means"], contrasts, bound["form_radii"], strict=True
    ):
        _within(value, interval)
        assert abs(contrast) <= radius + ABS_ERROR
    phases = tuple(map(Q, case["final"]["phase"]))
    for phase in phases:
        _within(phase, bound["phase"])
    assert max(phases) - min(phases) <= bound["diameter"] + 2 * ABS_ERROR
    assert max(phases) - min(phases) > 0
    for contrast, radius in zip(
        mean._contrasts(phases), bound["phase_radii"], strict=True
    ):
        assert abs(contrast) <= radius + ABS_ERROR
    assert case["final"]["counters"] == bound["counters"]
    for value, capacity in zip(
        mean._means(case["final"]["capacity"]), bound["capacity"], strict=True
    ):
        _within(value, capacity)


@pytest.mark.parametrize("key", tuple(product((1, -1), (False, True))))
def test_ordered_primitives_preserve_held_pressure_and_use_boosted_capacity(cases, key):
    case = cases[key]
    assert tuple(node for node, _, _, _ in case["primitives"]) == tuple(range(10))
    for node, glyph, before, after in case["primitives"]:
        assert before["edges"] == after["edges"]
        untouched = tuple(other for other in range(10) if other != node)
        for channel in ("form", "phase", "capacity", "pressure"):
            assert tuple(before[channel][other] for other in untouched) == tuple(
                after[channel][other] for other in untouched
            )
        if glyph == "RA":
            assert before["pressure"] == after["pressure"]
        else:
            assert before["form"] == after["form"] and before["phase"] == after["phase"]
    assert case["primitives"][1][2]["capacity"][0] > case["initial"]["capacity"][0]
    integration = base._event(case, "integration")
    assert integration["before"]["pressure"] == integration["after"]["pressure"]
    assert integration["before"]["capacity"] == case["final"]["capacity"]
    for node in (0, 1):
        observed = Q(case["final"]["rate"][node])
        old_capacity_rate = (
            Q(case["initial"]["capacity"][node]) * case["expected"]["pressure"][node]
        )
        assert abs(observed - old_capacity_rate) > 100 * ABS_ERROR
    assert not any(
        event["name"] in {"capacity_write", "network_remesh"}
        for event in case["events"]
    )
    assert (
        base._event(case, "capacity")["before"]["si"]
        == base._event(case, "si")["after"]["si"]
    )


@pytest.mark.parametrize("scalar", (False, True))
def test_opposite_primitive_phases_change_internal_forms_without_changing_means(
    cases, scalar
):
    plus, minus = cases[1, scalar], cases[-1, scalar]
    assert plus["initial"]["form"] == minus["initial"]["form"]
    assert plus["initial"]["capacity"] == minus["initial"]["capacity"]
    assert mean._means(plus["initial"]["phase"]) == mean._means(
        minus["initial"]["phase"]
    )
    assert plus["bound"] == minus["bound"]
    k, a, _, capacities, _, event_factors, factors, _, _ = _coefficients(plus["graph"])
    assert k > 0 and a > 0
    actual_plus, actual_minus = mean._contrasts(plus["final"]["form"]), mean._contrasts(
        minus["final"]["form"]
    )
    for i, (left, right) in enumerate(zip(actual_plus, actual_minus, strict=True)):
        predicted_gap = (
            -Q(base.DT)
            * capacities[i]
            * factors[i]
            * a
            * (-1) ** i
            * DIAMETER
            / Q(math.pi)
        )
        _within(left - right, predicted_gap, allowance=2 * ABS_ERROR)
        assert abs(left - right) > Q(1, 10**6)
        linear = event_factors[i] - Q(base.DT) * capacities[i] * factors[i] * k
        phase = Q(base.DT) * capacities[i] * factors[i] * a / Q(math.pi)
        assert linear > 0 and phase > 0
        variance_gap = -2 * linear * phase * RHO * DIAMETER
        # Each observed contrast differs from its ideal value by <=epsilon.
        # Bound the two squares by 2*epsilon*|observed|+epsilon**2 each.
        variance_budget = 2 * ABS_ERROR * (abs(left) + abs(right)) + 2 * ABS_ERROR**2
        _within(left**2 - right**2, variance_gap, allowance=variance_budget)
        assert left**2 < right**2
    for left, right in zip(
        mean._means(plus["final"]["form"]),
        mean._means(minus["final"]["form"]),
        strict=True,
    ):
        _within(left, right, allowance=2 * ABS_ERROR)


@pytest.mark.parametrize("key", tuple(product((1, -1), (False, True))))
def test_native_memory_configuration_and_phase_controller_remain_declared(cases, key):
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
    ):
        assert case["configuration"][name] == DEFAULTS[name]
    assert case["configuration"]["GRAMMAR_REJECTION_MODE"] == "raise"
    assert "glyph_selector" not in case["configuration"]
    assert "integrator" not in case["configuration"]
    assert case["initial"]["temporal"] == ((),) * 10
    assert case["initial"]["counters"] == (0,) * 10
    assert case["initial"]["ordinal"] is None and case["final"]["ordinal"] == 1
    assert case["final"]["time"] == 1 + base.DT
    assert case["initial"]["edges"] == case["final"]["edges"]
    hist = graph.graph["history"]
    assert tuple(hist["phase_state"]) == ("stable",)
    assert tuple(hist["phase_disr"]) == (0.0,)
    for suffix, gain in zip(("G", "L"), case["expected"]["gains"], strict=True):
        _within(hist[f"phase_k{suffix}"][0], gain)
    assert not hist.get("stable_frac")
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )
    for node, data in graph.nodes(data=True):
        assert data["epi_kind"] == "seed"
        assert tuple(data["glyph_history"]) == ("EN", MARKS[node][1])
        assert data[OPERATOR_STEP_KEY] == 4 and data[U2_DEBT_KEY] == 0
        assert case["final"]["temporal"][node] == (
            (1.0, base._event(case, "integration")["before"]["form"][node]),
            (1 + base.DT, case["final"]["form"][node]),
        )
        assert hist["since_AL"][node] == hist["since_EN"][node] == 1
