"""Fresh native Resonance selection does not imply live U3 admission.

Freeze doubled C5, order 0..9, seed 17, t=1, dt=1/16, default laws and strict
grammar. Pair forms are (-7/8,1,-1,-1,1), capacities (1/4,1,1,1,1), and
initial capacity counters zero. The first pair has phase 1+pi/2+s/1024;
the remaining phases equal 1. Both signs s=+/-1 and both scalar/NumPy paths
are frozen, with independent algebraic forecasts, before four native calls.

The readout is fresh pressure/Si, all prepared choices, live U3 admission and
the admitted Resonance prefix. Conditional on admission, form/rate/capacity
are predicted through the complete call; nonlinear terminal phase is not
claimed as an exact forecast. Binary64 observations have a fixed 1e-14
absolute comparison budget. This is a finite obstruction to identifying a
selector request with execution, not a new occurrence or rollback theorem.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q
from functools import wraps
from itertools import product

import pytest

import tnfr.operators as operators
from tests.physics import test_native_collective_step as base
from tnfr.constants import DEFAULTS
from tnfr.constants.canonical import DELTA_PHI_MAX
from tnfr.operators.grammar_debt import OPERATOR_STEP_KEY, U2_DEBT_KEY
from tnfr.operators.grammar_types import StructuralGrammarError

FORMS = (Q(-7, 8), Q(1), Q(-1), Q(-1), Q(1))
CAPACITIES = (Q(1, 4), Q(1), Q(1), Q(1), Q(1))
KEYS = tuple(product((-1, 1), (False, True)))
MARKS = tuple((node, "RA" if node < 2 else "IL") for node in range(10))
EPSILON, PI = Q(1, 1024), Q(math.pi)
ABS_ERROR = 1e-14


def _source(sign):
    graph = base._source(tuple(map(float, FORMS)), base.PROFILES[0])
    gap = PI / 2 + sign * EPSILON
    for node in graph:
        phase = 1 + gap if node < 2 else Q(1)
        assert Q(float(phase)) == phase
        graph.nodes[node].update(theta=float(phase), nu_f=float(CAPACITIES[node // 2]))
    return graph, gap


def _neighbor(values, block):
    return (values[(block - 1) % 5] + values[(block + 1) % 5]) / 2


def _family_certificate(graph):
    """Bound the whole phase interval by exact monotone endpoint algebra."""
    k, b, a = (
        base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf", "phase")
    )
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    assert (a, k, b) == (alpha, beta, gamma)
    lower_gap, upper_gap = PI / 2 - EPSILON, PI / 2 + EPSILON
    assert a > 0 and k > 0 and b > 0
    lower_z = 15 * k / 8 + 3 * b / 4 - a * upper_gap / PI
    upper_z = 15 * k / 8 + 3 * b / 4 - a * lower_gap / PI
    assert Q(7, 1000) < lower_z <= upper_z < Q(9, 1000)
    # z decreases with gap; r=z/(k+z/2) increases with positive z since
    # dr/dz=k/(k+z/2)**2 > 0. Separate bounds safely discard correlation.
    lower_ratio = lower_z / (k + lower_z / 2)
    upper_ratio = upper_z / (k + upper_z / 2)
    assert 0 < lower_ratio <= upper_ratio < Q(1, 20)
    lower_si = alpha / 4 + beta * (1 - upper_gap / PI) + gamma * (1 - upper_ratio)
    upper_si = alpha / 4 + beta * (1 - lower_gap / PI) + gamma * (1 - lower_ratio)
    assert Q(1, 3) < lower_si <= upper_si < Q(17, 50)
    ordinary_floor = alpha + beta * (1 - upper_gap / (2 * PI))
    assert ordinary_floor > Q(7, 8)
    thresholds = graph.graph["SELECTOR_THRESHOLDS"]
    assert Q(thresholds["si_lo"]) < lower_si < upper_si < Q(thresholds["si_hi"])
    assert upper_ratio < Q(thresholds["dnfr_hi"])
    assert ordinary_floor > Q(thresholds["si_hi"])
    return lower_z, upper_z, lower_si, upper_si, ordinary_floor


def _forecast(graph, gap):
    """Evaluate identities from primitives without a pressure or selector call."""
    cfg = graph.graph
    k, b, a = (
        base._weight(graph, "DNFR_WEIGHTS", key) for key in ("epi", "vf", "phase")
    )
    offsets = (-gap, gap / 2, Q(0), Q(0), gap / 2)
    pressure = tuple(
        k * (_neighbor(FORMS, i) - FORMS[i])
        + b * (_neighbor(CAPACITIES, i) - CAPACITIES[i])
        + a * offsets[i] / PI
        for i in range(5)
    )
    z = 15 * k / 8 + 3 * b / 4 - a * gap / PI
    assert pressure == (z, -k - z / 2, k, k, -k - z / 2)
    peak = max(map(abs, pressure))
    alpha, beta, gamma = (
        base._weight(graph, "SI_WEIGHTS", name) for name in ("alpha", "beta", "gamma")
    )
    si = tuple(
        alpha * capacity + beta * (1 - abs(offset) / PI) + gamma * (1 - abs(p) / peak)
        for capacity, offset, p in zip(CAPACITIES, offsets, pressure, strict=True)
    )
    limits = cfg["SELECTOR_THRESHOLDS"]
    # Positive margins certify the branch without executing the selector.
    margins = (
        si[0] - Q(limits["si_lo"]),
        Q(limits["si_hi"]) - si[0],
        Q(limits["dnfr_hi"]) - abs(pressure[0]) / peak,
        *(value - Q(limits["si_hi"]) for value in si[1:]),
    )
    assert min(margins) > Q(1, 100)
    hard_limit = Q(cfg.get("DELTA_PHI_MAX", DELTA_PHI_MAX))
    assert hard_limit == PI / 2 and abs(gap - hard_limit) == EPSILON
    admitted = gap < hard_limit
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
    event_form = ((1 - mix) * FORMS[0] + mix, *FORMS[1:])
    event_phase = (1 + (1 - coupling) * gap, Q(1), Q(1), Q(1), Q(1))
    capacity = ((1 + boost) / 4, *CAPACITIES[1:])
    held = (pressure[0], *(retention * value for value in pressure[1:]))
    rate = tuple(v * p for v, p in zip(capacity, held, strict=True))
    form = tuple(x + Q(base.DT) * r for x, r in zip(event_form, rate, strict=True))
    assert Q(cfg["EPI_MIN"]) < event_form[0] < 0
    assert all(Q(cfg["EPI_MIN"]) < x < Q(cfg["EPI_MAX"]) for x in form)
    assert all(Q(cfg["VF_MIN"]) < v < Q(cfg["VF_MAX"]) for v in capacity)
    assert min(map(abs, held)) > Q(cfg["EPS_DNFR_STABLE"])
    return {
        "gap": gap,
        "initial_pressure": pressure,
        "si": si,
        "margins": margins,
        "admitted": admitted,
        "event_form": event_form,
        "event_phase": event_phase,
        "capacity": capacity,
        "pressure": held,
        "rate": rate,
        "form": form,
        "acceleration": tuple(
            (r - Q(old)) / Q(base.DT) for r, old in zip(rate, base.RATES, strict=True)
        ),
    }


def _execute(graph, configuration, *, scalar):
    """Observe an actual native failure as well as a completed invocation."""
    events, choices, gates, primitives, u3 = [], [], [], [], []
    initial = base._snapshot(graph)
    assert base.integrators._uses_builtin_gamma(configuration["GAMMA"])
    original_choice = base.selectors._resolve_preselected_glyph
    original_gate = base.selectors.enforce_canonical_grammar
    original_glyph = base.selectors.apply_glyph
    original_u3 = operators._runtime_u3_neighbors
    choice_depth = 0
    prepared = []

    @wraps(original_choice)
    def capture_choice(graph, node, *args, **kwargs):
        nonlocal choice_depth
        if choice_depth == 0:
            selector = args[0]
            assert selector is base.selectors.default_glyph_selector
            prepared.append(
                selector._prepared_graph is graph and selector._preselection is not None
            )
        choice_depth += 1
        try:
            result = original_choice(graph, node, *args, **kwargs)
        finally:
            choice_depth -= 1
        # The default selector delegates internally to this same helper.
        # Record one actual batch request, not both stack frames.
        if choice_depth == 0:
            choices.append((node, result))
        return result

    @wraps(original_gate)
    def capture_gate(graph, node, glyph, **kwargs):
        event = {"node": node, "glyph": glyph, "before": base._snapshot(graph)}
        gates.append(event)
        try:
            result = original_gate(graph, node, glyph, **kwargs)
        except StructuralGrammarError as exc:
            event["error"] = exc
            raise
        event["result"] = result
        return result

    @wraps(original_glyph)
    def capture_glyph(graph, node, glyph, **kwargs):
        before = base._snapshot(graph)
        result = original_glyph(graph, node, glyph, **kwargs)
        primitives.append((node, glyph, before, base._snapshot(graph)))
        return result

    @wraps(original_u3)
    def capture_u3(node, code):
        selection, neighbors = original_u3(node, code)
        u3.append((node.n, code, selection, tuple(other.n for other in neighbors)))
        return selection, neighbors

    error = None
    with pytest.MonkeyPatch.context() as patch:
        if scalar:
            graph.graph["vectorized_dnfr"] = False
            for owner in (
                base.sense_index,
                base.selectors,
                base.integrators,
                base.coordination,
            ):
                patch.setattr(owner, "np", None)
        for owner, attribute, name in (
            (base.runtime, "_record_mutation_flow_boundary", "flow_history"),
            (base.runtime, "_refresh_delta_nfr", "pressure"),
            (base.dynamics, "compute_Si", "si"),
            (base.selectors, "_apply_glyphs", "glyphs"),
            (base.coordination, "coordinate_global_local_phase", "phase"),
            (base.adaptation, "adapt_vf_after_structural_stability", "capacity"),
            (base.adaptation, "set_vf", "capacity_write"),
            (base.runtime, "append_remesh_epi_history_snapshot", "remesh_history"),
            (base.runtime, "apply_remesh_if_globally_stable", "remesh_gate"),
        ):
            base._spy(patch, owner, attribute, name, events)
        base._spy(
            patch,
            base.integrators.DefaultIntegrator,
            "integrate",
            "integration",
            events,
            graph_index=1,
        )
        patch.setattr(base.selectors, "_resolve_preselected_glyph", capture_choice)
        patch.setattr(base.selectors, "enforce_canonical_grammar", capture_gate)
        patch.setattr(base.selectors, "apply_glyph", capture_glyph)
        patch.setattr(operators, "_runtime_u3_neighbors", capture_u3)
        try:
            assert base.runtime.step(graph, dt=base.DT) is None
        except StructuralGrammarError as exc:
            error = exc
    return {
        "graph": graph,
        "initial": initial,
        "final": base._snapshot(graph),
        "configuration": configuration,
        "events": events,
        "choices": tuple(choices),
        "gates": tuple(gates),
        "primitives": tuple(primitives),
        "u3": tuple(u3),
        "error": error,
        "prepared": tuple(prepared),
        "selector_cleared": (
            base.selectors.default_glyph_selector._preselection is None
            and base.selectors.default_glyph_selector._prepared_graph is None
        ),
    }


@pytest.fixture(scope="module")
def cases():
    # Freeze every preparation and prediction before the first reserved call.
    frozen = []
    for sign, scalar in KEYS:
        graph, gap = _source(sign)
        configuration = deepcopy(
            {
                key: value
                for key, value in graph.graph.items()
                if key
                in set(DEFAULTS) | {"GRAMMAR_REJECTION_MODE", "compute_delta_nfr"}
            }
        )
        frozen.append(
            (
                graph,
                configuration,
                sign,
                scalar,
                _forecast(graph, gap),
                _family_certificate(graph),
            )
        )
    result = {}
    for graph, configuration, sign, scalar, expected, certificate in frozen:
        case = _execute(graph, configuration, scalar=scalar)
        case["expected"] = expected
        case["certificate"] = certificate
        result[sign, scalar] = case
    return result


@pytest.mark.parametrize("key", KEYS)
def test_fresh_pressure_and_si_predict_the_same_complete_choice_batch(cases, key):
    case, expected = cases[key], cases[key]["expected"]
    for event, channel, forecast in (
        ("pressure", "pressure", "initial_pressure"),
        ("si", "si", "si"),
    ):
        assert base._event(case, event)["after"][channel] == pytest.approx(
            base._lift(expected[forecast]), abs=ABS_ERROR, rel=0
        )
    assert case["choices"] == MARKS
    assert min(expected["margins"]) > 1000 * Q(ABS_ERROR)
    lower_z, upper_z, lower_si, upper_si, ordinary_floor = case["certificate"]
    assert lower_z <= expected["initial_pressure"][0] <= upper_z
    assert lower_si <= expected["si"][0] <= upper_si
    assert min(expected["si"][1:]) >= ordinary_floor
    assert case["gates"][0]["node"] == 0 and case["gates"][0]["glyph"] == "RA"


@pytest.mark.parametrize("key", KEYS)
def test_live_u3_distinguishes_admission_from_the_identical_requested_word(cases, key):
    case, expected = cases[key], cases[key]["expected"]
    if expected["admitted"]:
        assert case["error"] is None
        assert tuple((node, glyph) for node, glyph, _, _ in case["primitives"]) == MARKS
        assert tuple(node for node, _, _, _ in case["u3"]) == (0, 1)
        for node, glyph, selection, neighbors in case["u3"]:
            assert glyph == "RA" and neighbors == tuple(case["graph"].neighbors(node))
            assert len(neighbors) == 4 and selection.phases == (1.0,) * 4
            assert selection.effective_limit == float(PI / 2)
            assert Q(selection.target_phase) - 1 == expected["gap"]
            assert Q(selection.effective_limit) - expected["gap"] == EPSILON
    else:
        assert isinstance(case["error"], StructuralGrammarError)
        assert case["error"].rule == "U3" and case["error"].candidate == "RA"
        assert "no compatible neighbor" in str(case["error"])
        assert case["gates"][0]["error"] is case["error"]
        assert len(case["gates"]) == 1
        assert case["primitives"] == case["u3"] == ()
        assert expected["gap"] - PI / 2 == EPSILON


@pytest.mark.parametrize("scalar", (False, True))
def test_admitted_resonance_prefix_retains_identity_and_live_neighbor_snapshot(
    cases, scalar
):
    case, expected = cases[-1, scalar], cases[-1, scalar]["expected"]
    for node, glyph, before, after in case["primitives"]:
        assert before["edges"] == after["edges"]
        for channel in ("form", "phase", "capacity", "pressure"):
            assert tuple(before[channel][n] for n in range(10) if n != node) == tuple(
                after[channel][n] for n in range(10) if n != node
            )
        if glyph == "RA":
            assert after["pressure"] == before["pressure"]
            for channel in ("form", "phase", "capacity"):
                forecast = expected[
                    channel if channel == "capacity" else f"event_{channel}"
                ][0]
                assert after[channel][node] == pytest.approx(
                    float(forecast), abs=ABS_ERROR, rel=0
                )
            assert before["form"][node] < 0 and after["form"][node] < 0
            assert case["graph"].nodes[node]["epi_kind"] == "seed"
    assert case["primitives"][1][2]["phase"][0] != case["initial"]["phase"][0]
    # Twins are nonadjacent: processing the first target does not change the
    # second target's compatible-neighbor phase snapshot.
    assert case["u3"][0][2].phases == case["u3"][1][2].phases


@pytest.mark.parametrize("scalar", (False, True))
def test_rejection_precedes_primitive_flow_without_claiming_runtime_rollback(
    cases, scalar
):
    case = cases[1, scalar]
    assert [event["name"] for event in case["events"]] == [
        "flow_history",
        "pressure",
        "si",
        "glyphs",
    ]
    assert "after" not in base._event(case, "glyphs")
    for channel in (
        "form",
        "phase",
        "capacity",
        "rate",
        "acceleration",
        "counters",
        "edges",
    ):
        assert case["final"][channel] == case["initial"][channel]
    assert case["initial"]["pressure"] != case["final"]["pressure"]
    assert case["initial"]["ordinal"] is None and case["final"]["ordinal"] == 1
    assert case["final"]["time"] == 1.0
    for node in case["graph"]:
        assert case["final"]["temporal"][node] == (
            (1.0, case["initial"]["form"][node]),
        )
        assert case["graph"].nodes[node][OPERATOR_STEP_KEY] == 3
        assert case["graph"].nodes[node][U2_DEBT_KEY] == (node // 2) % 2
    assert "_epi_hist" not in case["graph"].graph


@pytest.mark.parametrize("scalar", (False, True))
def test_admitted_complete_call_uses_held_pressure_and_amplified_capacity(
    cases, scalar
):
    case, expected = cases[-1, scalar], cases[-1, scalar]["expected"]
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
            base._lift(expected[channel]), abs=ABS_ERROR, rel=0
        )
    integration = base._event(case, "integration")
    assert integration["before"]["pressure"] == integration["after"]["pressure"]
    assert integration["before"]["capacity"] == case["final"]["capacity"]
    assert case["final"]["counters"] == (0,) * 10
    assert case["final"]["time"] == 1 + base.DT
    assert case["final"]["ordinal"] == 1
    assert case["final"]["phase"] != base._event(case, "glyphs")["after"]["phase"]
    assert all(math.isfinite(value) for value in case["final"]["phase"])
    assert (
        base._event(case, "remesh_gate")["before"]
        == base._event(case, "remesh_gate")["after"]
    )
    for node in case["graph"]:
        assert case["graph"].nodes[node][OPERATOR_STEP_KEY] == 4
        assert case["final"]["temporal"][node] == (
            (1.0, integration["before"]["form"][node]),
            (1 + base.DT, case["final"]["form"][node]),
        )


@pytest.mark.parametrize("key", KEYS)
def test_provenance_keeps_default_laws_strict_grammar_and_supplied_memory(cases, key):
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
    assert case["configuration"]["GAMMA"]["type"] == "none"
    assert case["configuration"]["GRAMMAR_REJECTION_MODE"] == "raise"
    assert case["configuration"]["GRAMMAR_CANON"]["enabled"] is True
    assert case["configuration"]["compute_delta_nfr"] is base.default_compute_delta_nfr
    assert graph.graph["compute_delta_nfr"] is base.default_compute_delta_nfr
    assert base._event(case, "pressure")["result"] is base.default_compute_delta_nfr
    assert case["prepared"] == (True,) * 10 and case["selector_cleared"] is True
    assert "glyph_selector" not in graph.graph and "integrator" not in graph.graph
    assert tuple(graph) == tuple(range(10)) and graph.graph["RANDOM_SEED"] == 17
    assert len(case["final"]["edges"]) == 20
    assert case["final"]["edges"] == case["initial"]["edges"]
    history = graph.graph["history"]
    assert all(
        history["since_AL"][node] == history["since_EN"][node] == 1 for node in graph
    )
    assert len(history.get("stable_frac", ())) == 0
    assert "_REMESH_META" not in graph.graph and "_last_remesh_step" not in graph.graph
