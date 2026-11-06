"""A bare C5 realizes a conditional doubled-C5 native Resonance/Coherence map.

These are current-source equivalence controls, reusing three independently
forecast preparations and one U3 rejection. Freeze all sixteen configurations
and their existing forecasts before any response: fine/coarse support, scalar
and NumPy, seed 17, t=1, dt=1/16, strict grammar and unchanged default laws.
Consecutive nonadjacent twins share every retained coordinate and memory.

The coarse graph deliberately has different valid stored pressure, Si and old
acceleration. Pressure/Si are refreshed, and admitted acceleration is ignored
by the default choice. This is not invariance of the complete graph dictionary:
initial observations, selector caches, support, node counts and samples differ.
The adaptation case additionally retains the consumed six-token grammar suffix
and lifetime counters, while discarding three valid older identifiers; both
actual appends produce the same seven-token trace before the phase controller.
Successful physical ordinates have absolute error budget 1e-14 against the
existing independent predictions and 2e-14 across the two actual runtimes.
Discrete memory and within-backend twin equality are exact. A rejected call
retains old acceleration and is compared only on its declared reached prefix.

No new production predictor, source search, retuned law, generic lumpability,
repeated-call invariance, autonomous formation or physical identification is
claimed. Complete-pair event boundaries, native direct glyphs, default selector,
zero Gamma, inactive REMESH and the declared history projection are essential.
"""

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product

import networkx as nx
import pytest

from tests.physics import test_native_collective_step as base
from tests.physics import test_native_nonuniform_collective_step as nonuniform
from tests.physics import test_native_resonance_live_prefix as prefix
from tests.physics import test_native_resonance_neighbor_gain as gain
from tests.physics import test_native_resonance_u3_boundary as boundary
from tnfr.constants import DEFAULTS
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    read_grammar_state,
)
from tnfr.operators.grammar_types import StructuralGrammarError

KINDS = ("live_prefix", "neighbor_gain", "active_adaptation", "u3_rejection")
KEYS = tuple(product(KINDS, (False, True)))
ACCEPTED = tuple(product(KINDS[:-1], (False, True)))
ERROR = Q(1, 10**14)
LONG_TRACE = ("OZ", "RA", "IL", "AL", "IL", "EN", "AL", "IL", "EN")


def _configuration(graph):
    keys = set(DEFAULTS) | {"GRAMMAR_REJECTION_MODE", "compute_delta_nfr"}
    return deepcopy({key: value for key, value in graph.graph.items() if key in keys})


def _coarse_source(fine):
    """Prepare a plain C5 from an unevaluated source, never a fine response."""
    graph = nx.Graph()
    graph.graph.update(deepcopy(fine.graph))
    for block in range(5):
        graph.add_node(block, **deepcopy(fine.nodes[2 * block]))
    for block in range(5):
        graph.add_edge(block, (block + 1) % 5, weight=1.0)
    for name in ("since_AL", "since_EN"):
        graph.graph["history"][name] = {
            block: fine.graph["history"][name][2 * block] for block in range(5)
        }
    for block, data in graph.nodes(data=True):
        # Modest admitted values keep selector normalization representable.
        # The success projection omits their left-boundary observations.
        data.update(delta_nfr=(block + 1) / 16, Si=(block + 1) / 8, d2EPI=1 / 8)
        # Admission still reads every identifier before any old prefix is lost.
        admitted = read_grammar_state(data)
        data[OPERATOR_STEP_KEY] = admitted.operator_step
        data[U2_DEBT_KEY] = admitted.u2_debt
        data[PRIOR_COHERENCE_KEY] = admitted.prior_coherence
        if len(admitted.history) > 6:
            assert DEFAULTS["GLYPH_HYSTERESIS_WINDOW"] == 7
            data["glyph_history"] = list(admitted.history[-6:])
    return graph


def _preparation(kind):
    if kind == "live_prefix":
        graph = prefix._source()
        prefix._family_certificate(graph)
        expected = prefix._forecast(graph)
        expected.update(counts=(0,) * 5, phase_R=Q(1), phase_state="stable")
    elif kind == "neighbor_gain":
        graph = gain._source()
        gain._family_certificate(graph)
        expected = gain._forecast(graph)
        expected.update(counts=(0,) * 5, phase_state="dissonant")
    elif kind == "active_adaptation":
        graph = nonuniform._source(
            nonuniform.SMALL_FORMS,
            nonuniform.SMALL_PHASES,
            nonuniform.SMALL_CAPACITIES,
            base.PROFILES[0],
            ready=True,
        )
        expected = nonuniform._oracle(
            graph,
            nonuniform.SMALL_FORMS,
            nonuniform.SMALL_PHASES,
            nonuniform.SMALL_CAPACITIES,
            base.PROFILES[0],
        )
        expected.update(
            event_form=expected["post_event"],
            phase_R=expected["phase_order"],
            phase_state="stable",
        )
        for data in graph.nodes.values():
            data["glyph_history"] = list(LONG_TRACE)
            data[OPERATOR_STEP_KEY] = 20
        assert all(expected["stable"])
        assert min(expected["si"]) > graph.graph["SELECTOR_THRESHOLDS"]["si_hi"]
    else:
        assert kind == "u3_rejection"
        graph, gap = boundary._source(1)
        boundary._family_certificate(graph)
        expected = boundary._forecast(graph, gap)
        assert expected["admitted"] is False
    marks = (
        ("IL",) * 5
        if kind == "active_adaptation"
        else ("RA",) * (1 if kind == "u3_rejection" else 2)
        + ("IL",) * (4 if kind == "u3_rejection" else 3)
    )
    expected.update(
        history_prefix=LONG_TRACE[-6:] if kind == "active_adaptation" else ("EN",),
        old_operator_step=20 if kind == "active_adaptation" else 3,
    )
    return graph, expected, marks


def _lift(values, multiplicity):
    return tuple(value for value in values for _ in range(multiplicity))


def _close(actual, expected, allowance=ERROR):
    assert len(actual) == len(expected)
    assert all(
        abs(Q(left) - Q(right)) <= allowance
        for left, right in zip(actual, expected, strict=True)
    )


@pytest.fixture(scope="module")
def cases():
    frozen = []
    for kind, scalar in KEYS:
        fine, expected, marks = _preparation(kind)
        coarse = _coarse_source(fine)
        for graph, multiplicity in ((fine, 2), (coarse, 1)):
            if scalar:
                graph.graph["vectorized_dnfr"] = False
            frozen.append(
                (
                    kind,
                    scalar,
                    multiplicity,
                    graph,
                    _configuration(graph),
                    expected,
                    marks,
                    tuple(read_grammar_state(data) for data in graph.nodes.values()),
                )
            )
    # All source/configuration objects and forecasts exist before any call.
    assert len(frozen) == 16
    responses = {}
    for (
        kind,
        scalar,
        multiplicity,
        graph,
        configuration,
        expected,
        marks,
        initial_grammar,
    ) in frozen:
        case = boundary._execute(graph, configuration, scalar=scalar)
        case.update(
            expected=expected,
            marks=marks,
            multiplicity=multiplicity,
            initial_grammar=initial_grammar,
        )
        responses.setdefault((kind, scalar), {})[multiplicity] = case
    return responses


def test_normalized_support_rows_commute_with_uniform_duplication():
    """Exact basis-row certificate, independent of runtime pressure outputs."""
    fine = prefix._source()
    coarse = _coarse_source(fine)
    for coordinate in range(5):
        values = tuple(Q(int(block == coordinate)) for block in range(5))
        for node in fine:
            block = node // 2
            fine_mean = sum(values[n // 2] for n in fine.neighbors(node)) / 4
            coarse_mean = sum(values[n] for n in coarse.neighbors(block)) / 2
            assert fine_mean == coarse_mean
            assert fine_mean - values[block] == coarse_mean - values[block]
    assert {fine.degree(node) for node in fine} == {4}
    assert {coarse.degree(node) for node in coarse} == {2}
    for graph in (fine, coarse):
        for node in graph:
            assert (
                sum(graph.degree(n) for n in graph.neighbors(node))
                == graph.degree(node) ** 2
            )


def test_suffix_projection_admits_every_discarded_grammar_identifier():
    fine, _, _ = _preparation("active_adaptation")
    fine.nodes[0]["glyph_history"][0] = "not-a-canonical-operator"
    with pytest.raises(ValueError, match="canonical operator identifiers"):
        _coarse_source(fine)
    assert len(fine.nodes[0]["glyph_history"]) == len(LONG_TRACE)
    assert fine.nodes[0][OPERATOR_STEP_KEY] == 20


@pytest.mark.parametrize("raw_step", (None, 1))
def test_suffix_projection_preserves_context_inferred_from_the_full_source(raw_step):
    fine, _, _ = _preparation("active_adaptation")
    trace = ("IL", "OZ", "OZ") + ("RA",) * 6
    for node in (0, 1):
        data = fine.nodes[node]
        data["glyph_history"] = list(trace)
        del data[U2_DEBT_KEY]
        del data[PRIOR_COHERENCE_KEY]
        if raw_step is None:
            del data[OPERATOR_STEP_KEY]
        else:
            data[OPERATOR_STEP_KEY] = raw_step
    admitted = read_grammar_state(fine.nodes[0])
    assert admitted.operator_step == 9 and admitted.u2_debt == 2
    assert admitted.prior_coherence is True
    suffix_only = read_grammar_state({"glyph_history": list(trace[-6:])})
    assert suffix_only.operator_step == 6 and suffix_only.u2_debt == 0
    assert suffix_only.prior_coherence is False

    coarse = _coarse_source(fine)

    result = read_grammar_state(coarse.nodes[0])
    assert result.history == trace[-6:]
    assert result.operator_step == admitted.operator_step
    assert result.u2_debt == admitted.u2_debt
    assert result.prior_coherence == admitted.prior_coherence
    assert fine.nodes[0]["glyph_history"] == list(trace)
    assert fine.nodes[0].get(OPERATOR_STEP_KEY) == raw_step
    assert U2_DEBT_KEY not in fine.nodes[0] and PRIOR_COHERENCE_KEY not in fine.nodes[0]


@pytest.mark.parametrize("key", KEYS)
def test_same_retained_preparation_has_distinct_support_and_discarded_fields(
    cases, key
):
    fine, coarse = cases[key][2], cases[key][1]
    for name in ("form", "capacity", "phase", "rate", "counters", "temporal"):
        assert fine["initial"][name] == _lift(coarse["initial"][name], 2)
    for name in ("pressure", "si", "acceleration"):
        assert fine["initial"][name] != _lift(coarse["initial"][name], 2)
    for node, state in enumerate(fine["initial_grammar"]):
        other = coarse["initial_grammar"][node // 2]
        assert state.u2_debt == other.u2_debt
        assert state.prior_coherence == other.prior_coherence
        assert state.operator_step == other.operator_step
        assert state.history[-6:] == other.history[-6:]
        if key[0] == "active_adaptation":
            assert state.history == LONG_TRACE and other.history == LONG_TRACE[-6:]
    for case in (fine, coarse):
        graph, cfg, multiplicity = (
            case["graph"],
            case["configuration"],
            case["multiplicity"],
        )
        assert tuple(graph) == tuple(range(5 * multiplicity))
        assert graph.number_of_edges() == 5 * multiplicity**2
        assert case["initial"]["edges"] == case["final"]["edges"]
        assert tuple(graph.graph["_node_sample"]) == tuple(graph)
        assert cfg["RANDOM_SEED"] == 17 and cfg["GRAMMAR_REJECTION_MODE"] == "raise"
        assert cfg["compute_delta_nfr"] is base.default_compute_delta_nfr
        for name in (
            "DNFR_WEIGHTS",
            "SI_WEIGHTS",
            "GLYPH_FACTORS",
            "SELECTOR_THRESHOLDS",
            "PHASE_ADAPT",
            "GAMMA",
            "VF_ADAPT_TAU",
            "VF_ADAPT_MU",
            "INTEGRATOR_METHOD",
        ):
            assert cfg[name] == DEFAULTS[name]
        assert "glyph_selector" not in graph.graph and "integrator" not in graph.graph
        assert "MATH_ENGINE" not in graph.graph
        assert "ra_metrics" not in graph.graph and "_ra_c_tracking" not in graph.graph


@pytest.mark.parametrize("key", KEYS)
def test_fresh_requests_and_diagnostics_match_existing_independent_forecasts(
    cases, key
):
    for case in cases[key].values():
        expected, multiplicity = case["expected"], case["multiplicity"]
        assert case["choices"] == tuple(enumerate(_lift(case["marks"], multiplicity)))
        assert case["prepared"] == (True,) * (5 * multiplicity)
        assert case["selector_cleared"] is True
        for event, channel, prediction in (
            ("pressure", "pressure", "initial_pressure"),
            ("si", "si", "si"),
        ):
            _close(
                base._event(case, event)["after"][channel],
                _lift(expected[prediction], multiplicity),
            )
        assert base._event(case, "pressure")["result"] is base.default_compute_delta_nfr


@pytest.mark.parametrize("key", ACCEPTED)
def test_complete_pair_event_boundaries_realize_the_coarse_ordered_prefix(cases, key):
    fine, coarse = cases[key][2], cases[key][1]
    assert fine["error"] is coarse["error"] is None
    for block, mark in enumerate(coarse["marks"]):
        fine_event, coarse_event = (
            fine["primitives"][2 * block + 1],
            coarse["primitives"][block],
        )
        assert fine_event[:2] == (2 * block + 1, mark)
        assert coarse_event[:2] == (block, mark)
        for name in ("form", "capacity", "phase", "pressure", "rate", "si"):
            _close(fine_event[3][name], _lift(coarse_event[3][name], 2), 2 * ERROR)
    for fine_event, coarse_event in zip(fine["u3"][::2], coarse["u3"], strict=True):
        fine_node, code, fine_selection, fine_neighbors = fine_event
        coarse_node, coarse_code, coarse_selection, coarse_neighbors = coarse_event
        assert fine_node == 2 * coarse_node and code == coarse_code == "RA"
        assert tuple(n // 2 for n in fine_neighbors) == _lift(coarse_neighbors, 2)
        _close(fine_selection.phases, _lift(coarse_selection.phases, 2), 2 * ERROR)
        _close(
            (fine_selection.target_phase,), (coarse_selection.target_phase,), 2 * ERROR
        )
    for case in (fine, coarse):
        _close(
            base._event(case, "glyphs")["after"]["form"],
            _lift(case["expected"]["event_form"], case["multiplicity"]),
        )


@pytest.mark.parametrize("key", ACCEPTED)
def test_complete_native_endpoints_match_forecasts_and_each_other(cases, key):
    fine, coarse = cases[key][2], cases[key][1]
    for channel in base.CHANNELS:
        for case in (fine, coarse):
            _close(
                case["final"][channel],
                _lift(case["expected"][channel], case["multiplicity"]),
            )
        _close(fine["final"][channel], _lift(coarse["final"][channel], 2), 2 * ERROR)
    for case in (fine, coarse):
        assert (
            case["final"]["pressure"]
            == base._event(case, "glyphs")["after"]["pressure"]
        )
        assert case["final"]["si"] == base._event(case, "si")["after"]["si"]
    for event in ("glyphs", "integration", "phase", "capacity"):
        snapshot = base._event(fine, event)["after"]
        for first, second in base.PAIRS:
            assert all(
                snapshot[name][first] == snapshot[name][second]
                for name in base.CHANNELS
            )


@pytest.mark.parametrize("key", ACCEPTED)
def test_global_controller_and_snapshot_capacity_adaptation_descend(cases, key):
    kind, _ = key
    for case in cases[key].values():
        graph, expected = case["graph"], case["expected"]
        history, multiplicity = graph.graph["history"], case["multiplicity"]
        assert tuple(history["phase_state"]) == (expected["phase_state"],)
        assert tuple(history["phase_disr"]) == (0.0,)
        _close(history["phase_R"], (expected["phase_R"],))
        _close((history["phase_kG"][0], history["phase_kL"][0]), expected["gains"])
        _close(
            (graph.graph["PHASE_K_GLOBAL"], graph.graph["PHASE_K_LOCAL"]),
            expected["gains"],
        )
        assert case["final"]["counters"] == _lift(expected["counts"], multiplicity)
        written = tuple(
            event["node"]
            for event in case["events"]
            if event["name"] == "capacity_write"
        )
        assert written == (tuple(graph) if kind == "active_adaptation" else ())
        assert (
            base._event(case, "capacity")["before"]["pressure"]
            == case["final"]["pressure"]
        )
        assert base._event(case, "capacity")["before"]["si"] == case["final"]["si"]
        assert not history.get("stable_frac") and "_REMESH_META" not in graph.graph
        assert (
            base._event(case, "remesh_gate")["before"]
            == base._event(case, "remesh_gate")["after"]
        )
        snapshots = graph.graph["_epi_hist"]
        assert len(snapshots) == 1 and tuple(snapshots[0]) == tuple(graph)
        assert tuple(snapshots[0].values()) == case["final"]["form"]


@pytest.mark.parametrize("key", ACCEPTED)
def test_grammar_temporal_and_lag_memory_follow_the_declared_lift(cases, key):
    fine, coarse = cases[key][2], cases[key][1]
    for case in (fine, coarse):
        graph, multiplicity = case["graph"], case["multiplicity"]
        assert case["initial"]["ordinal"] is None and case["final"]["ordinal"] == 1
        assert case["final"]["time"] == 1 + base.DT
        for node, data in graph.nodes(data=True):
            block, mark = node // multiplicity, case["marks"][node // multiplicity]
            assert data["epi_kind"] == "seed"
            assert tuple(data["glyph_history"]) == (
                *case["expected"]["history_prefix"],
                mark,
            )
            assert data[OPERATOR_STEP_KEY] == case["expected"]["old_operator_step"] + 1
            assert data[PRIOR_COHERENCE_KEY] is True
            assert data[U2_DEBT_KEY] == (block % 2 if mark == "RA" else 0)
            assert graph.graph["history"]["since_AL"][node] == 1
            assert graph.graph["history"]["since_EN"][node] == 1
            assert case["final"]["temporal"][node] == (
                (1.0, base._event(case, "integration")["before"]["form"][node]),
                (1 + base.DT, case["final"]["form"][node]),
            )
    for node in fine["graph"]:
        assert read_grammar_state(fine["graph"].nodes[node]) == read_grammar_state(
            coarse["graph"].nodes[node // 2]
        )
        fine_history, coarse_history = (
            fine["final"]["temporal"][node],
            coarse["final"]["temporal"][node // 2],
        )
        assert tuple(t for t, _ in fine_history) == tuple(t for t, _ in coarse_history)
        _close(
            tuple(x for _, x in fine_history),
            tuple(x for _, x in coarse_history),
            2 * ERROR,
        )


@pytest.mark.parametrize("scalar", (False, True))
def test_u3_rejection_descends_without_erasing_the_actual_failed_prefix(cases, scalar):
    fine, coarse = cases["u3_rejection", scalar][2], cases["u3_rejection", scalar][1]
    for case in (fine, coarse):
        error, graph = case["error"], case["graph"]
        assert isinstance(error, StructuralGrammarError)
        assert error.rule == "U3" and error.candidate == "RA"
        assert len(case["gates"]) == 1 and case["gates"][0]["node"] == 0
        assert case["primitives"] == case["u3"] == ()
        assert tuple(event["name"] for event in case["events"]) == (
            "flow_history",
            "pressure",
            "si",
            "glyphs",
        )
        assert "after" not in base._event(case, "glyphs")
        assert case["final"]["ordinal"] == 1 and case["final"]["time"] == 1.0
        for channel in (
            "form",
            "capacity",
            "phase",
            "rate",
            "acceleration",
            "counters",
        ):
            assert case["initial"][channel] == case["final"][channel]
        for node, data in graph.nodes(data=True):
            assert data[OPERATOR_STEP_KEY] == 3
            assert tuple(data["glyph_history"]) == ("EN",)
            assert case["final"]["temporal"][node] == (
                (1.0, case["initial"]["form"][node]),
            )
        assert "_epi_hist" not in graph.graph
    for channel in ("form", "capacity", "phase", "rate", "pressure", "si"):
        _close(fine["final"][channel], _lift(coarse["final"][channel], 2), 2 * ERROR)
    # Omitted old acceleration still distinguishes the full failed states.
    assert fine["final"]["acceleration"] != _lift(coarse["final"]["acceleration"], 2)
