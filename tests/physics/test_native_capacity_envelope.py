"""Fresh native selector admission limits the otherwise amplifying RA primitive.

One declared path receives one ordinary default step, with no supplied Si,
temporal history, forced operator, or custom controller. Transparent spies retain
the selection snapshot and sequential primitive boundaries. This finite case
tests the source-level capacity-envelope premises, not long-time maintenance.
The separate grammar probes check supplied states without executing operators.
"""

import math
from copy import deepcopy
from fractions import Fraction
from functools import wraps

import networkx as nx
import pytest

from tests.physics.test_native_step_formation import _snapshot
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.dynamics import runtime, selectors
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.operators.grammar_debt import U2_DEBT_KEY
from tnfr.operators.grammar_dynamics import enforce_grammar_on_glyph, validate_candidate


def _prepared():
    graph = nx.path_graph(4)
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=17, compute_delta_nfr=default_compute_delta_nfr, _t=0.0
    )
    nx.set_edge_attributes(graph, 1.0, "weight")
    for node, capacity in zip(graph, (0.3, 0.3, 0.3, 1.0), strict=True):
        graph.nodes[node].update(
            theta=0.0,
            EPI=0.125,
            nu_f=capacity,
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=[],
        )
    return graph


@pytest.fixture(scope="module")
def native_capacity_step():
    graph = _prepared()
    initial = _snapshot(graph)
    configuration = deepcopy(
        {
            name: graph.graph[name]
            for name in (
                "DNFR_WEIGHTS",
                "SI_WEIGHTS",
                "SELECTOR_THRESHOLDS",
                "GLYPH_FACTORS",
                "GRAMMAR_CANON",
                "AL_MAX_LAG",
                "EN_MAX_LAG",
                "VF_ADAPT_TAU",
                "VF_ADAPT_MU",
                "VF_MAX",
                "DT",
            )
        }
    )
    events, selections, glyphs, batches = [], [], [], []
    original_batch = selectors._apply_glyphs
    original_select = selectors.DefaultGlyphSelector.select
    original_glyph = selectors.apply_glyph

    @wraps(original_batch)
    def batch(G, selector, history):
        before = _snapshot(G)
        events.append(("batch", "before"))
        result = original_batch(G, selector, history)
        batches.append((before, _snapshot(G), selector))
        events.append(("batch", "after"))
        return result

    @wraps(original_select)
    def select(owner, G, node):
        result = original_select(owner, G, node)
        events.append(("select", node))
        selections.append((node, getattr(result, "value", result), _snapshot(G)))
        return result

    @wraps(original_glyph)
    def glyph(G, node, choice, **kwargs):
        before = _snapshot(G)
        events.append(("glyph", node))
        result = original_glyph(G, node, choice, **kwargs)
        glyphs.append((node, getattr(choice, "value", choice), before, _snapshot(G)))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selectors, "_apply_glyphs", batch)
        patch.setattr(selectors.DefaultGlyphSelector, "select", select)
        patch.setattr(selectors, "apply_glyph", glyph)
        runtime.step(graph)

    return {
        "graph": graph,
        "initial": initial,
        "configuration": configuration,
        "events": tuple(events),
        "selections": tuple(selections),
        "glyphs": tuple(glyphs),
        "batches": tuple(batches),
        "endpoint": _snapshot(graph),
    }


def test_native_ra_uses_fresh_diagnostics_and_default_unforced_selection(
    native_capacity_step,
):
    case = native_capacity_step
    graph = case["graph"]
    initial = case["initial"]
    for name, value in case["configuration"].items():
        assert value == DEFAULTS[name]
    assert graph.graph["RANDOM_SEED"] == 17
    assert graph.graph["compute_delta_nfr"] is default_compute_delta_nfr
    assert graph.graph.get("glyph_selector") is None
    assert initial["glyphs"] == ((),) * 4
    assert initial["temporal"] == ((),) * 4
    assert initial["si"] == initial["pressure"] == (0.0,) * 4
    assert len(case["batches"]) == 1
    before, _after, selector = case["batches"][0]
    assert selector is selectors.default_glyph_selector
    assert tuple(choice for _, choice, _ in case["selections"]) == (
        "RA",
        "RA",
        "NAV",
        "IL",
    )
    assert tuple(choice for _, choice, _, _ in case["glyphs"]) == (
        "RA",
        "RA",
        "NAV",
        "IL",
    )

    # Phase and form channels vanish on this declared constant initial state.
    # Only the neighbor-capacity gradient contributes to fresh pressure.
    vf_weight = case["configuration"]["DNFR_WEIGHTS"]["vf"]
    expected_pressure = (0.0, 0.0, vf_weight * 0.35, -vf_weight * 0.7)
    assert before["pressure"] == pytest.approx(expected_pressure, rel=0, abs=2e-17)
    weights = case["configuration"]["SI_WEIGHTS"]
    total = sum(weights.values())
    alpha, beta, gamma = (weights[key] / total for key in ("alpha", "beta", "gamma"))
    expected_si = tuple(
        alpha * capacity + beta + gamma * (1 - normalized_pressure)
        for capacity, normalized_pressure in zip(
            initial["capacity"], (0.0, 0.0, 0.5, 1.0), strict=True
        )
    )
    assert before["si"] == pytest.approx(expected_si, rel=0, abs=2e-16)
    assert before["capacity"] == initial["capacity"]
    assert case["endpoint"]["time"] == case["configuration"]["DT"]


def test_all_native_choices_precede_target_only_capacity_writes(native_capacity_step):
    case = native_capacity_step
    before = case["batches"][0][0]
    assert case["events"] == (
        (("batch", "before"),)
        + tuple(("select", node) for node in range(4))
        + tuple(("glyph", node) for node in range(4))
        + (("batch", "after"),)
    )
    for _, _, selection_state in case["selections"]:
        assert selection_state["capacity"] == before["capacity"]
        assert selection_state["pressure"] == before["pressure"]
        assert selection_state["si"] == before["si"]
    boost = case["configuration"]["GLYPH_FACTORS"]["RA_vf_amplification"]
    assert boost == 1 / (8 * math.pi)
    for node, glyph, live_before, live_after in case["glyphs"]:
        # Earlier native primitives do not change a later target's capacity.
        assert live_before["capacity"][node] == before["capacity"][node]
        expected = list(live_before["capacity"])
        if glyph == "RA":
            expected[node] *= 1 + boost
        assert live_after["capacity"] == tuple(expected)
        assert live_after["edges"] == live_before["edges"]
    assert case["glyphs"][1][2]["capacity"][0] > before["capacity"][0]


def test_selected_ra_amplifies_inside_the_initial_capacity_interval(
    native_capacity_step,
):
    case = native_capacity_step
    initial = case["initial"]["capacity"]
    lower, upper = min(initial), max(initial)
    config = case["configuration"]
    alpha = config["SI_WEIGHTS"]["alpha"] / sum(config["SI_WEIGHTS"].values())
    boost = config["GLYPH_FACTORS"]["RA_vf_amplification"]
    high = config["SELECTOR_THRESHOLDS"]["si_hi"]
    # Exact arithmetic on represented constants retains ample policy margin;
    # it is not an assertion that Si floating operations are rational identities.
    policy_ceiling = Fraction(high) * (1 + Fraction(boost)) / Fraction(alpha)
    assert policy_ceiling < Fraction(7, 10)
    for node, glyph, before, after in case["glyphs"]:
        assert all(lower <= value <= upper for value in after["capacity"])
        if glyph == "RA":
            assert config["SELECTOR_THRESHOLDS"]["si_lo"] < before["si"][node] < high
            assert initial[node] < after["capacity"][node] < upper
    assert case["endpoint"]["capacity"] == case["batches"][0][1]["capacity"]
    assert all(lower <= value <= upper for value in case["endpoint"]["capacity"])
    # An enclosing interval is not monotone contraction of every local contrast.
    assert initial[1] == initial[2]
    assert case["endpoint"]["capacity"][1] > case["endpoint"]["capacity"][2]


@pytest.mark.parametrize(
    ("epi", "history", "candidate", "expected"),
    (
        (0.125, (), "OZ", "IL"),
        (0.0, (), "IL", "NAV"),
        (0.0, ("IL",), "ZHIR", "IL"),
    ),
)
def test_native_grammar_fallback_does_not_add_capacity_writers(
    epi, history, candidate, expected
):
    graph = _prepared()
    graph.nodes[0].update(EPI=epi, glyph_history=list(history))
    before = deepcopy(dict(graph.nodes[0]))
    admission = validate_candidate(graph, 0, candidate)
    assert not admission.allowed
    assert admission.suggested_alternative == expected
    assert enforce_grammar_on_glyph(graph, 0, candidate) == expected
    assert graph.nodes[0] == before


def test_zero_form_with_external_overdebt_has_no_admitted_terminal_fallback():
    graph = _prepared()
    graph.nodes[0][U2_DEBT_KEY] = U2_DEBT_CAPACITY + 2
    # This externally inconsistent bookkeeping is outside initialized native
    # provenance. The terminal fallback is IL even though its U1 admission fails;
    # it must not be reported as an admitted alternative or as a new writer.
    graph.nodes[0]["EPI"] = 0.0
    admission = validate_candidate(graph, 0, "IL")
    assert not admission.allowed
    assert admission.suggested_alternative == "IL"
    assert enforce_grammar_on_glyph(graph, 0, "IL") == "IL"
    assert {violation.rule for violation in admission.violations} == {"U1a"}
