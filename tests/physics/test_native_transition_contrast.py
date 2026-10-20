"""The actual second native call requires target-associated Transition noise.

Freeze the unchanged nominal neighbor-gain preparation on doubled C5, order
0..9, seed 17, t=1, dt=1/16, strict grammar, zero Gamma and all default laws.
The budget is two first-call preparation replays, then two reserved second
calls (scalar/NumPy), with no coarse trajectory, source search or third call.
All configurations and first/second forecasts exist before any native call.
The genuine completed predecessor is retained; the previous protocol-stopped
graph and any projected surrogate are never used as preparations.

The independent first forecast and exact second pressure/Si certificate give
RA/NAV/IL/IL/IL. The live first RA block uses both neighboring blocks and
retains positive form. Fine targets 2/3 consume independently reconstructed
draw-zero innovations. Pair means and contrasts d=(odd-even)/2 predict every
node's held pressure, form, rate and acceleration; the contrast here is
negative. Primitive phase/capacity remain paired, and an exact cosine bound
certifies the transitional coordinator with disruptor load 1/15. Adaptation
does not qualify. Raw phases use an independent elementary-trig forecast.

The frozen absolute budget is 1e-14 for every physical ordinate, including
acceleration, and the mean/contrast observations. Draw values and discrete
bookkeeping are exact. Source-associated RNG reconstruction reuses the
stdlib-only oracle in test_native_collective_mean, not the production RNG.
Symbolic alternative-innovation controls execute no further native response.
This is a supplied complete-law result, not formation, scale selection,
universal stochastic closure or physical identification.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q
from functools import wraps

import pytest

import tnfr.operators as operators
from tests.physics import test_native_block_composition as composition
from tests.physics import test_native_collective_mean as mean
from tests.physics import test_native_collective_step as base
from tests.physics import test_native_ra_il_block_descent as descent
from tests.physics import test_native_resonance_neighbor_gain as gain
from tests.physics import test_native_resonance_u3_boundary as boundary
from tnfr.constants import DEFAULTS
from tnfr.observers import glyph_load
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
)

ERROR = Q(1, 10**14)
MARKS = tuple(enumerate(descent._lift(composition.SECOND_MARKS, 2)))
FIRST_TIME, SECOND_TIME = Q(17, 16), Q(9, 8)


def _neighbor(values, block):
    return (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2


def _phase_certificate(graph, first):
    """Certify the second controller using rational relative phases only."""
    phi, (kg, kl) = first["event_phase"], first["gains"]
    local = tuple(_neighbor(phi, block) for block in range(5))
    z = tuple((1 - kg - kl) * value + kl * local[i] for i, value in enumerate(phi))
    common = first["phase"][0] - z[0]
    assert all(first["phase"][i] - z[i] == common for i in range(5))
    coupling = Q(graph.graph["GLYPH_FACTORS"]["RA_phase_coupling"])
    psi = ((1 - coupling) * z[0] + coupling * _neighbor(z, 0), *z[1:])
    differences = tuple(psi[i] - psi[j] for i in range(5) for j in range(i + 1, 5))
    assert max(map(abs, differences)) < Q(12, 5)

    def cosine_polynomial(angle, terms):
        return sum(
            Q((-1) ** degree, math.factorial(2 * degree)) * angle ** (2 * degree)
            for degree in range(terms)
        )

    # After the quadratic term the alternating cosine terms decrease for
    # |angle|<12/5. Degree14 is below cos, degree12 above it.
    lower = (5 + 2 * sum(cosine_polynomial(d, 8) for d in differences)) / 25
    upper = (5 + 2 * sum(cosine_polynomial(d, 7) for d in differences)) / 25
    policy = graph.graph["PHASE_ADAPT"]
    assert Q(policy["R_lo"]) ** 2 < Q(719, 1000) ** 2 < lower
    assert lower < upper < Q(720, 1000) ** 2 < Q(policy["R_hi"]) ** 2
    traces = tuple(("EN", gain.MARKS[node][1], MARKS[node][1]) for node in range(10))
    load = Q(
        sum(mark == "NAV" for trace in traces for mark in trace), sum(map(len, traces))
    )
    assert load == Q(1, 15) < Q(policy["disr_lo"]) < Q(policy["disr_hi"])
    gains = []
    for current, suffix in zip(first["gains"], ("G", "L"), strict=True):
        target = (Q(policy[f"k{suffix}_min"]) + Q(policy[f"k{suffix}_max"])) / 2
        assert (target > current) is (suffix == "G")
        speed = Q(policy["up"] if target > current else policy["down"])
        gains.append(current + speed * (target - current))
    assert Q(61, 1000) < gains[0] < Q(62, 1000)
    assert Q(141, 1000) < gains[1] < Q(142, 1000) and sum(gains) < 1
    assert (1 - gains[0]) * (max(psi) - min(psi)) < Q(253, 125) < gain.PI
    return dict(
        relative_phase=psi,
        common=common,
        gains=tuple(gains),
        R_squared=(lower, upper),
        disruptors=load,
        traces=traces,
    )


def _forecast(graph, first):
    """Predict every second-call coordinate without consuming a response."""
    cfg = graph.graph
    fresh = composition._second_call_certificate(graph, first)
    phase_certificate = _phase_certificate(graph, first)
    x, v, theta = first["form"], first["capacity"], first["phase"]
    r, a, c, h, eta = (
        Q(cfg["GLYPH_FACTORS"][name])
        for name in (
            "RA_epi_diff",
            "RA_vf_amplification",
            "RA_phase_coupling",
            "IL_dnfr_factor",
            "NAV_eta",
        )
    )
    selected_mean = _neighbor(x, 0)
    clearances = tuple(gain.LIMIT - abs(theta[i] - theta[0]) for i in (1, 4))
    assert min(clearances) > Q(42, 100)
    assert Q(-796, 1000) < selected_mean < Q(-795, 1000)
    assert abs(selected_mean) > Q(RA_RUNTIME_AMPLIFICATION_TRIGGER)
    event_forms = ((1 - r) * x[0] + r * selected_mean, *x[1:])
    event_capacities = ((1 + a) * v[0], *v[1:])
    event_phases = ((1 - c) * theta[0] + c * _neighbor(theta, 0), *theta[1:])
    assert Q(411, 1000) < event_forms[0] < Q(412, 1000) < x[0]
    assert Q(27, 100) < event_capacities[0] < Q(271, 1000)
    assert event_phases == tuple(
        value + phase_certificate["common"]
        for value in phase_certificate["relative_phase"]
    )
    assert cfg["NAV_RANDOM"] is True and cfg["NAV_STRICT"] is False
    amplitude = Q(cfg["GLYPH_FACTORS"]["NAV_jitter"])
    innovations = tuple(mean._jitter(node, 0, amplitude) for node in (2, 3))
    innovation_mean = sum(innovations) / 2
    innovation_contrast = (innovations[1] - innovations[0]) / 2
    assert innovations[1] < 0 < innovations[0] and innovation_contrast < 0
    deterministic = (1 - eta) * fresh["pressure"][1] + eta * v[1]
    assert fresh["pressure"][1] > 0
    # These admission bounds cover the entire configured innovation interval,
    # not just the two reserved draws: NAV stays positive and changes pressure.
    assert deterministic - amplitude > Q(134, 1000)
    assert deterministic + amplitude < Q(215, 1000)
    assert deterministic - fresh["pressure"][1] - amplitude > Q(45, 1000)
    nav_pressures = tuple(deterministic + value for value in innovations)
    held_means = (
        fresh["pressure"][0],
        deterministic + innovation_mean,
        *(h * value for value in fresh["pressure"][2:]),
    )
    held_contrasts = (Q(0), innovation_contrast, Q(0), Q(0), Q(0))
    held = mean._fine(held_means, held_contrasts)
    assert held[2:4] == nav_pressures
    rates = tuple(event_capacities[node // 2] * p for node, p in enumerate(held))
    forms = tuple(
        event_forms[node // 2] + Q(base.DT) * rate for node, rate in enumerate(rates)
    )
    acceleration = tuple(
        (rate - first["rate"][node // 2]) / Q(base.DT)
        for node, rate in enumerate(rates)
    )
    # Every endpoint stays strictly interior, even for any allowed NAV draw.
    uniform_endpoints = (
        forms[:2]
        + forms[4:]
        + tuple(
            x[1] + Q(base.DT) * v[1] * (deterministic + sign * amplitude)
            for sign in (-1, 1)
        )
    )
    assert all(
        Q(cfg["EPI_MIN"]) < value < Q(cfg["EPI_MAX"]) for value in uniform_endpoints
    )
    assert all(
        Q(cfg["VF_MIN"]) < value < Q(cfg["VF_MAX"]) for value in event_capacities
    )
    assert (
        min(abs(held_means[0]), deterministic - amplitude, *map(abs, held_means[2:]))
        > Q(15, 10000)
        > Q(cfg["EPS_DNFR_STABLE"])
    )
    rate_means = tuple(
        value * capacity
        for value, capacity in zip(held_means, event_capacities, strict=True)
    )
    rate_contrasts = tuple(
        value * capacity
        for value, capacity in zip(held_contrasts, event_capacities, strict=True)
    )
    form_means = tuple(
        value + Q(base.DT) * rate
        for value, rate in zip(event_forms, rate_means, strict=True)
    )
    form_contrasts = tuple(Q(base.DT) * value for value in rate_contrasts)
    acceleration_means = tuple(
        (value - old) / Q(base.DT)
        for value, old in zip(rate_means, first["rate"], strict=True)
    )
    acceleration_contrasts = tuple(value / Q(base.DT) for value in rate_contrasts)
    assert Q(-295, 10**6) < form_contrasts[1] < Q(-293, 10**6)
    relative = phase_certificate["relative_phase"]
    cos_sum = math.fsum(math.cos(float(value)) for value in relative)
    sin_sum = math.fsum(math.sin(float(value)) for value in relative)
    target = Q(math.atan2(sin_sum, cos_sum))
    kg, kl = phase_certificate["gains"]
    phases = tuple(
        phase_certificate["common"]
        + (1 - kg - kl) * value
        + kg * target
        + kl * _neighbor(relative, i)
        for i, value in enumerate(relative)
    )
    assert all(min(event_phases) < value < max(event_phases) for value in phases)
    return {
        "fresh": fresh,
        "event_form": event_forms,
        "event_phase": event_phases,
        "event_capacity": event_capacities,
        "selected_mean": selected_mean,
        "u3_clearances": clearances,
        "innovations": innovations,
        "innovation_mean": innovation_mean,
        "innovation_contrast": innovation_contrast,
        "bare_node_innovation": mean._jitter(1, 0, amplitude),
        "amplitude": amplitude,
        "deterministic_form_mean": x[1] + Q(base.DT) * v[1] * deterministic,
        "form_diamond_radius": Q(base.DT) * v[1] * amplitude,
        "deterministic_pressure": deterministic,
        "pressure": held,
        "rate": rates,
        "form": forms,
        "acceleration": acceleration,
        "phase": descent._lift(phases, 2),
        "capacity": descent._lift(event_capacities, 2),
        "si": descent._lift(fresh["si"], 2),
        "phase_certificate": phase_certificate,
        "phase_R": Q(math.hypot(cos_sum, sin_sum) / 5),
        "means": dict(
            pressure=held_means,
            rate=rate_means,
            form=form_means,
            acceleration=acceleration_means,
        ),
        "contrasts": dict(
            pressure=held_contrasts,
            rate=rate_contrasts,
            form=form_contrasts,
            acceleration=acceleration_contrasts,
        ),
    }


def _predecessor(graph):
    """Copy consumed state/evidence without copying graph-owned cache locks."""
    return deepcopy(
        dict(
            state=base._snapshot(graph),
            memory=composition._memory(graph),
            configuration=descent._configuration(graph),
            jitter=tuple(
                graph.nodes[node].get("_rng_jitter_progress") for node in graph
            ),
        )
    )


def _second_response(graph, *, scalar):
    draws = []
    actual = operators.random_jitter

    @wraps(actual)
    def observe_draw(node, amplitude):
        result = actual(node, amplitude)
        draws.append((node.n, amplitude, result))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(operators, "random_jitter", observe_draw)
        response = boundary._execute(
            graph, descent._configuration(graph), scalar=scalar
        )
    response.update(draws=tuple(draws), memory=composition._memory(graph))
    return response


@pytest.fixture(scope="module")
def cases():
    frozen = {}
    for scalar in (False, True):
        graph = gain._source()
        if scalar:
            graph.graph["vectorized_dnfr"] = False
        gain._family_certificate(graph)
        first = gain._forecast(graph)
        frozen[scalar] = dict(
            graph=graph,
            configuration=descent._configuration(graph),
            first_forecast=first,
            expected=_forecast(graph, first),
        )
    # Both complete first/second forecasts precede any native evaluation.
    for scalar, item in frozen.items():
        preparation = boundary._execute(
            item["graph"], item["configuration"], scalar=scalar
        )
        assert preparation["error"] is None and preparation["choices"] == gain.MARKS
        for name in base.CHANNELS:
            descent._close(
                preparation["final"][name],
                descent._lift(item["first_forecast"][name], 2),
                ERROR,
            )
        predecessor = _predecessor(item["graph"])
        assert predecessor["jitter"] == (None,) * 10
        assert predecessor["state"]["ordinal"] == 1
        assert Q(predecessor["state"]["time"]) == FIRST_TIME
        item.update(preparation=preparation, predecessor=predecessor)
    # Use the actual completed graphs; copy only their evidence for inspection.
    for scalar, item in frozen.items():
        assert _predecessor(item["graph"]) == item["predecessor"]
        item["response"] = _second_response(item["graph"], scalar=scalar)
    return frozen


def test_second_call_has_independent_exact_admission_and_controller_bounds():
    graph = gain._source()
    first = gain._forecast(graph)
    expected = _forecast(graph, first)
    lower, upper = expected["phase_certificate"]["R_squared"]
    assert Q(719, 1000) ** 2 < lower < upper < Q(720, 1000) ** 2
    assert expected["fresh"]["marks"] == composition.SECOND_MARKS


def test_mean_and_contrast_innovations_cannot_be_replaced_by_one_bare_draw():
    graph = gain._source()
    expected = _forecast(graph, gain._forecast(graph))
    j0, j1 = expected["innovations"]
    midpoint, contrast = expected["innovation_mean"], expected["innovation_contrast"]
    assert j0 == midpoint - contrast and j1 == midpoint + contrast
    for substitute in (expected["bare_node_innovation"], j0, j1):
        assert abs(substitute - midpoint) > Q(1, 1000)
        replacement_mean = expected["deterministic_pressure"] + substitute
        assert abs(replacement_mean - expected["means"]["pressure"][1]) > Q(1, 1000)
    # A shared draw set deliberately to the correct mean still loses contrast.
    assert expected["contrasts"]["pressure"][1] == contrast != 0
    assert expected["contrasts"]["form"][1] < Q(-293, 10**6)
    for name in expected["means"]:
        assert (
            mean._fine(expected["means"][name], expected["contrasts"][name])
            == expected[name]
        )


def test_innovation_box_has_a_correlated_mean_contrast_diamond():
    """Certify an amplitude-box image, not reachability of every fixed-seed pair.

    The coordinate map is affine and the l1 norm is convex. Bounding its four
    rectangle corners therefore bounds the entire input box. The inverse map
    recovers the two input intervals; this preserves their joint correlation.
    """
    graph = gain._source()
    expected = _forecast(graph, gain._forecast(graph))
    amplitude = expected["amplitude"]
    for even in (-amplitude, amplitude):
        for odd in (-amplitude, amplitude):
            center, contrast = (even + odd) / 2, (odd - even) / 2
            assert abs(center) + abs(contrast) == amplitude
            assert center - contrast == even and center + contrast == odd
    center, contrast = expected["innovation_mean"], expected["innovation_contrast"]
    assert (
        abs(center) + abs(contrast)
        == max(map(abs, expected["innovations"]))
        < amplitude
    )
    offset = expected["means"]["form"][1] - expected["deterministic_form_mean"]
    scale = Q(base.DT) * expected["event_capacity"][1]
    assert offset == scale * center
    assert expected["contrasts"]["form"][1] == scale * contrast
    assert (
        abs(offset) + abs(expected["contrasts"]["form"][1])
        < expected["form_diamond_radius"]
    )


@pytest.mark.parametrize("scalar", (False, True))
def test_native_second_preparation_consumes_the_genuine_completed_predecessor(
    cases, scalar
):
    item = cases[scalar]
    case, predecessor = item["response"], item["predecessor"]
    assert case["initial"] == predecessor["state"] == item["preparation"]["final"]
    assert case["error"] is None and case["choices"] == MARKS
    assert case["prepared"] == (True,) * 10 and case["selector_cleared"] is True
    for name in ("GLOBAL", "LOCAL"):
        assert (
            case["configuration"][f"PHASE_K_{name}"]
            == predecessor["configuration"][f"PHASE_K_{name}"]
        )
    for event, name in (("pressure", "pressure"), ("si", "si")):
        descent._close(
            base._event(case, event)["after"][name],
            descent._lift(item["expected"]["fresh"][name], 2),
            ERROR,
        )
    assert tuple((node, code) for node, code, _, _ in case["primitives"]) == MARKS


@pytest.mark.parametrize("scalar", (False, True))
def test_live_resonance_prefix_precedes_transition_and_preserves_identity(
    cases, scalar
):
    case, expected = cases[scalar]["response"], cases[scalar]["expected"]
    assert tuple(node for node, _, _, _ in case["u3"]) == (0, 1)
    for node, code, selection, neighbors in case["u3"]:
        assert code == "RA" and neighbors == tuple(case["graph"].neighbors(node))
        assert neighbors == (2, 3, 8, 9)
        before, after = case["primitives"][node][2:]
        descent._close(
            (sum(Q(before["form"][n]) for n in neighbors) / 4,),
            (expected["selected_mean"],),
            ERROR,
        )
        assert before["form"][node] > 0 and after["form"][node] > 0
        for name, forecast in (
            ("form", "event_form"),
            ("phase", "event_phase"),
            ("capacity", "event_capacity"),
        ):
            descent._close((after[name][node],), (expected[forecast][0],), ERROR)
        assert after["pressure"] == before["pressure"]
        assert after["capacity"][node] > before["capacity"][node]
    assert case["primitives"][2][2]["form"][:2] == case["primitives"][1][3]["form"][:2]
    assert all(data["epi_kind"] == "seed" for data in case["graph"].nodes.values())


@pytest.mark.parametrize("scalar", (False, True))
def test_transition_draws_have_exact_target_and_progress_association(cases, scalar):
    case, expected = cases[scalar]["response"], cases[scalar]["expected"]
    assert tuple(node for node, _, _ in case["draws"]) == (2, 3)
    assert tuple(Q(value) for _, _, value in case["draws"]) == expected["innovations"]
    amplitude = case["configuration"]["GLYPH_FACTORS"]["NAV_jitter"]
    assert tuple(value for _, value, _ in case["draws"]) == (amplitude, amplitude)
    for node in (2, 3):
        before, after = case["primitives"][node][2:]
        for name in ("form", "phase", "capacity", "rate", "acceleration", "si"):
            assert before[name] == after[name]
        descent._close((after["pressure"][node],), (expected["pressure"][node],), ERROR)
        assert after["pressure"][node] != before["pressure"][node]
        assert case["graph"].nodes[node]["_rng_jitter_progress"] == dict(
            seed=17, offset=node, draws=1
        )
    assert all(
        "_rng_jitter_progress" not in case["graph"].nodes[node]
        for node in case["graph"]
        if node not in (2, 3)
    )


@pytest.mark.parametrize("scalar", (False, True))
def test_full_endpoint_and_pair_mean_contrast_match_independent_forecasts(
    cases, scalar
):
    case, expected = cases[scalar]["response"], cases[scalar]["expected"]
    for name in base.CHANNELS:
        descent._close(case["final"][name], expected[name], ERROR)
    for name in expected["means"]:
        descent._close(mean._means(case["final"][name]), expected["means"][name], ERROR)
        descent._close(
            mean._contrasts(case["final"][name]), expected["contrasts"][name], ERROR
        )
    assert case["final"]["form"][3] < case["final"]["form"][2]
    assert abs(case["final"]["form"][3] - case["final"]["form"][2]) > 0.00058
    for block, (even, odd) in enumerate(base.PAIRS):
        for name in ("phase", "capacity", "si"):
            assert case["final"][name][even] == case["final"][name][odd]
        if block != 1:
            assert all(
                case["final"][name][even] == case["final"][name][odd]
                for name in base.CHANNELS
            )
    assert case["final"]["pressure"] == base._event(case, "glyphs")["after"]["pressure"]
    assert case["final"]["si"] == base._event(case, "si")["after"]["si"]


@pytest.mark.parametrize("scalar", (False, True))
def test_phase_controller_uses_new_disruptor_load_and_retained_gains(cases, scalar):
    case, expected = cases[scalar]["response"], cases[scalar]["expected"]
    history = case["graph"].graph["history"]
    assert tuple(history["phase_state"]) == ("dissonant", "transition")
    descent._close((history["phase_R"][-1],), (expected["phase_R"],), ERROR)
    descent._close((history["phase_disr"][-1],), (Q(1, 15),), ERROR)
    descent._close(
        (history["phase_kG"][-1], history["phase_kL"][-1]),
        expected["phase_certificate"]["gains"],
        ERROR,
    )
    loads = glyph_load(case["graph"])
    assert loads["_count"] == 30
    descent._close((loads["_disruptors"],), (Q(1, 15),), ERROR)
    assert case["final"]["counters"] == (0,) * 10
    assert not any(event["name"] == "capacity_write" for event in case["events"])
    assert (
        base._event(case, "phase")["before"]["phase"]
        == base._event(case, "glyphs")["after"]["phase"]
    )


@pytest.mark.parametrize("scalar", (False, True))
def test_grammar_and_temporal_memory_keep_the_actual_two_call_origin(cases, scalar):
    item, case = cases[scalar], cases[scalar]["response"]
    graph, expected = case["graph"], item["expected"]
    assert Q(case["final"]["time"]) == SECOND_TIME and case["final"]["ordinal"] == 2
    integration = base._event(case, "integration")
    for node, data in graph.nodes(data=True):
        assert (
            tuple(data["glyph_history"])
            == expected["phase_certificate"]["traces"][node]
        )
        assert data[OPERATOR_STEP_KEY] == 5 and data[PRIOR_COHERENCE_KEY] is True
        assert data[U2_DEBT_KEY] == int(node in (2, 3))
        assert (
            graph.graph["history"]["since_AL"][node]
            == graph.graph["history"]["since_EN"][node]
            == 2
        )
        prefix = (
            ((float(FIRST_TIME), integration["before"]["form"][node]),)
            if node < 2
            else item["predecessor"]["state"]["temporal"][node]
        )
        assert case["final"]["temporal"][node] == (
            *prefix,
            (float(SECOND_TIME), case["final"]["form"][node]),
        )
    snapshots = graph.graph["_epi_hist"]
    assert len(snapshots) == 2
    assert tuple(snapshots[0].values()) == item["predecessor"]["state"]["form"]
    assert tuple(snapshots[1].values()) == case["final"]["form"]
    assert (
        not graph.graph["history"].get("stable_frac")
        and "_REMESH_META" not in graph.graph
    )
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )


@pytest.mark.parametrize("scalar", (False, True))
def test_complete_law_support_and_backend_provenance_are_unchanged(cases, scalar):
    item, case = cases[scalar], cases[scalar]["response"]
    for name in (
        "DNFR_WEIGHTS",
        "SI_WEIGHTS",
        "GLYPH_FACTORS",
        "SELECTOR_THRESHOLDS",
        "PHASE_ADAPT",
        "GAMMA",
        "INTEGRATOR_METHOD",
        "VF_ADAPT_MU",
        "VF_ADAPT_TAU",
        "NAV_RANDOM",
        "NAV_STRICT",
    ):
        assert (
            item["configuration"][name] == case["configuration"][name] == DEFAULTS[name]
        )
    assert case["configuration"]["GRAMMAR_REJECTION_MODE"] == "raise"
    assert case["configuration"]["compute_delta_nfr"] is base.default_compute_delta_nfr
    assert (
        tuple(case["graph"]) == tuple(range(10))
        and case["graph"].number_of_edges() == 20
    )
    assert case["initial"]["edges"] == case["final"]["edges"]
    assert (
        "glyph_selector" not in case["graph"].graph
        and "integrator" not in case["graph"].graph
    )
    assert (
        "ra_metrics" not in case["graph"].graph
        and "_ra_c_tracking" not in case["graph"].graph
    )
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
