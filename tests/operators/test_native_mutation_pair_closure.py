"""Paired-phase closure of supplied native Mutation requests.

These fixed doubled-C5 states use fresh default pressure, but their AL/IL/OZ
context and timestamped growth records are supplied preparation. They are not
acquired-history or default-selector witnesses. Direct strict admission and
the actual supplied-selector batch have distinct rejection/abstention paths.
No complete runtime call, flow, coordinator or frozen producer is executed.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr, get_attr_str, set_attr
from tnfr.constants import DEFAULTS
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_SOURCE_GLYPH, ALIAS_THETA
from tnfr.dynamics import adaptation, coordination, integrators, runtime, selectors
from tnfr.dynamics.dnfr import _resolve_dnfr_weights, default_compute_delta_nfr
from tnfr.operators import apply_glyph
from tnfr.operators._mutation_gate import validate_mutation_runtime_gate
from tnfr.operators.grammar_application import enforce_canonical_grammar
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    reset_grammar_state_from_history,
)
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.preconditions import OperatorPreconditionError

TARGETS = (0, 1)
FORM_CONTRAST = Q(1, 16)
CAPACITY_CONTRAST = Q(1, 8)
PREFIX = ("AL", "IL", "OZ")


@pytest.fixture(scope="module", autouse=True)
def no_continued_dynamics():
    def forbidden(*args, **kwargs):
        pytest.fail("Pair-event controls must not execute continued dynamics")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (runtime, "step"),
            (runtime, "run"),
            (integrators.DefaultIntegrator, "integrate"),
            (coordination, "coordinate_global_local_phase"),
            (adaptation, "adapt_vf_after_structural_stability"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


def _source(neighbor_mean_shift=Q(0)):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for block in range(5):
        for left in (2 * block, 2 * block + 1):
            for right in (2 * ((block + 1) % 5), 2 * ((block + 1) % 5) + 1):
                graph.add_edge(left, right, weight=1.0)
    means = tuple(
        Q(1, 4) + (neighbor_mean_shift if block in (1, 4) else 0) for block in range(5)
    )
    graph.graph.update(
        GRAMMAR_REJECTION_MODE="raise",
        GRAMMAR_CANON={**DEFAULTS["GRAMMAR_CANON"], "enabled": True},
        GLYPH_HYSTERESIS_WINDOW=10,
        GLYPH_SELECTOR_N_JOBS=1,
        RANDOM_SEED=20261018,
        _t=1.0,
        history={
            "since_AL": dict.fromkeys(graph, 0),
            "since_EN": dict.fromkeys(graph, 0),
        },
    )
    for node in graph:
        sign = -1 if node % 2 == 0 else 1
        form = means[node // 2] + (sign * FORM_CONTRAST if node < 2 else 0)
        capacity = 1 + (sign * CAPACITY_CONTRAST if node < 2 else 0)
        data = graph.nodes[node]
        data.update(
            EPI=float(form),
            nu_f=float(capacity),
            theta=1.0,
            epi_kind="seed",
            glyph_history=list(PREFIX),
            epi_time_history=[(0.0, float(form - Q(1, 4))), (1.0, float(form))],
            **{OPERATOR_STEP_KEY: len(PREFIX)},
        )
        reset_grammar_state_from_history(data)
        assert data[U2_DEBT_KEY] == 1 and data[PRIOR_COHERENCE_KEY]
    weights = _resolve_dnfr_weights(graph)
    k, b = Q(weights["epi"]), Q(weights["vf"])
    # Paired phases are constant; the common capacity is one and degree four.
    # Fresh target pressures are B+C and B-C, with C=k*d+b*e.
    center = k * neighbor_mean_shift
    contrast = k * FORM_CONTRAST + b * CAPACITY_CONTRAST
    expected = (center + contrast, center - contrast)
    default_compute_delta_nfr(graph, cache_size=0, n_jobs=1)
    actual = tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in TARGETS)
    assert actual == pytest.approx(tuple(map(float, expected)), rel=0, abs=2e-15)
    assert min(abs(value) for value in expected) > Q(1, 1000)
    return graph, expected


def _strict_direct(graph, node):
    admitted = enforce_canonical_grammar(graph, node, "ZHIR")
    assert admitted == "ZHIR"
    apply_glyph(graph, node, admitted, window=10)


def _phases(graph):
    return tuple(get_attr(graph.nodes[node], ALIAS_THETA) for node in TARGETS)


def _snapshot(graph):
    # Retain all semantic node/edge state and selector history. Cache membership
    # and graph object identities additionally expose rejected adapter writes.
    return (
        deepcopy(dict(graph.nodes(data=True))),
        deepcopy(tuple(graph.edges(data=True))),
        deepcopy(graph.graph["history"]),
        graph.graph["_t"],
        tuple(graph.graph.get("_node_cache", {})),
        {key: id(value) for key, value in graph.graph.items() if key != "_node_cache"},
    )


def _assert_accepted(graph, before, node, shift, *, fixed=False):
    data = graph.nodes[node]
    for key in ("EPI", "nu_f", "epi_kind", "epi_time_history"):
        assert data[key] == before[node][key]
    assert get_attr(data, ALIAS_DNFR) == get_attr(before[node], ALIAS_DNFR)
    assert tuple(data["glyph_history"]) == (*PREFIX, "ZHIR")
    assert data[OPERATOR_STEP_KEY] == 4
    assert data[U2_DEBT_KEY] == 2 and data[PRIOR_COHERENCE_KEY]
    assert get_attr_str(data, ALIAS_SOURCE_GLYPH) == "ZHIR"
    assert data["_zhir_theta_shift"] == shift
    assert data["_zhir_fixed_mode"] is fixed
    assert get_attr(data, ALIAS_THETA) == 1.0 + shift
    if not fixed:
        assert data["_zhir_theta_before"] == 1.0
        assert data["_zhir_theta_after"] == 1.0 + shift
        assert data["_zhir_regime_before"] == data["_zhir_regime_after"] == 0
        assert data["_zhir_regime_changed"] is False


@pytest.mark.parametrize("neighbor_mean_shift", (Q(0), Q(1, 2), Q(-1, 2)))
def test_fresh_pressure_signs_determine_pair_phase_closure(neighbor_mean_shift):
    graph, pressure = _source(neighbor_mean_shift)
    before = deepcopy(dict(graph.nodes(data=True)))
    shift = DEFAULTS["GLYPH_FACTORS"]["ZHIR_theta_shift_factor"] * (math.pi / 4)
    signed_shifts = tuple(math.copysign(shift, float(value)) for value in pressure)
    for node, signed_shift in zip(TARGETS, signed_shifts, strict=True):
        gate = validate_mutation_runtime_gate(graph.nodes[node], graph.graph)
        assert gate.threshold.depi_dt == 0.25 > gate.threshold.xi
        _strict_direct(graph, node)
        _assert_accepted(graph, before, node, signed_shift)
        if node == 0:
            assert graph.nodes[1] == before[1]
            assert _phases(graph) == (1.0 + signed_shift, 1.0)
    left, right = _phases(graph)
    assert (left + right) / 2 == 1 + sum(signed_shifts) / 2
    assert (right - left) / 2 == (signed_shifts[1] - signed_shifts[0]) / 2
    assert (left == right) is (neighbor_mean_shift != 0)
    assert all(graph.nodes[node] == before[node] for node in range(2, 10))
    assert graph.graph["_t"] == 1.0
    assert all(
        value == 0
        for values in graph.graph["history"].values()
        for value in values.values()
    )


def test_explicit_signed_zero_pressure_splits_the_pair_without_a_freshness_claim():
    graph, _ = _source()
    for node, value in zip(TARGETS, (0.0, -0.0), strict=True):
        set_attr(graph.nodes[node], ALIAS_DNFR, value)
    before = deepcopy(dict(graph.nodes(data=True)))
    for node, direction in zip(TARGETS, (1, -1), strict=True):
        assert math.copysign(1, get_attr(graph.nodes[node], ALIAS_DNFR)) == direction
        _strict_direct(graph, node)
        _assert_accepted(graph, before, node, direction * 0.25)
        assert math.copysign(1, get_attr(graph.nodes[node], ALIAS_DNFR)) == direction
    assert _phases(graph) == (1.25, 0.75)


def test_common_fixed_shift_preserves_pairing_despite_opposite_fresh_pressures():
    graph, pressure = _source()
    assert pressure[0] > 0 > pressure[1]
    graph.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift": 0.375}
    before = deepcopy(dict(graph.nodes(data=True)))
    for node in TARGETS:
        _strict_direct(graph, node)
        _assert_accepted(graph, before, node, 0.375, fixed=True)
    assert _phases(graph) == (1.375, 1.375)


@pytest.mark.parametrize("failure", ("history", "grammar"))
def test_direct_second_rejection_preserves_the_first_committed_mutation(failure):
    graph, _ = _source()
    if failure == "history":
        del graph.nodes[1]["epi_time_history"]
        error, reason = OperatorPreconditionError, "at least two"
    else:
        graph.nodes[1]["glyph_history"] = ["AL", "IL", "EN"]
        reset_grammar_state_from_history(graph.nodes[1])
        error, reason = StructuralGrammarError, "requires a recent destabilizer"
        assert validate_mutation_runtime_gate(
            graph.nodes[1], graph.graph
        ).threshold.crossed
    _strict_direct(graph, 0)
    after_first = _snapshot(graph)
    with pytest.raises(error, match=reason) as rejection:
        _strict_direct(graph, 1)
    if failure == "grammar":
        assert rejection.value.rule == "U4b"
    assert _snapshot(graph) == after_first
    assert _phases(graph) == (1.25, 1.0)
    assert tuple(graph.nodes[0]["glyph_history"]) == (*PREFIX, "ZHIR")
    assert graph.nodes[1][OPERATOR_STEP_KEY] == 3


def _supplied_requests(graph, node):
    return "ZHIR" if node in TARGETS else None


def test_actual_batch_substitutes_coherence_for_second_missing_history(monkeypatch):
    graph, _ = _source()
    del graph.nodes[1]["epi_time_history"]
    before = deepcopy(dict(graph.nodes(data=True)))
    dispatched = []

    def record(subject, node, glyph, **kwargs):
        apply_glyph(subject, node, glyph, **kwargs)
        dispatched.append((node, glyph))

    monkeypatch.setattr(selectors, "apply_glyph", record)
    selectors._apply_glyph_batch(graph, _supplied_requests, graph.graph["history"])
    assert dispatched == [(0, "ZHIR"), (1, "IL")]
    _assert_accepted(graph, before, 0, 0.25)
    assert _phases(graph) == (1.25, 1.0)
    second = graph.nodes[1]
    assert tuple(second["glyph_history"]) == (*PREFIX, "IL")
    assert second[OPERATOR_STEP_KEY] == 4 and second[U2_DEBT_KEY] == 0
    factor = DEFAULTS["GLYPH_FACTORS"]["IL_dnfr_factor"]
    assert get_attr(second, ALIAS_DNFR) == factor * get_attr(before[1], ALIAS_DNFR)
    assert "_zhir_theta_shift" not in second
    (abstention,) = graph.graph["history"]["mutation_abstentions"]
    assert abstention["node"] == 1
    assert abstention["requested_glyph"] == "ZHIR"
    assert abstention["applied_glyph"] == "IL"
    assert abstention["reason"] == "missing_history"
    assert abstention["observed_depi_dt"] is None
    assert abstention["time"] == graph.graph["_t"] == 1.0
    for key in ("since_AL", "since_EN"):
        assert graph.graph["history"][key] == dict.fromkeys(graph, 1)
    assert all(graph.nodes[node] == before[node] for node in range(2, 10))


def test_actual_batch_grammar_rejection_preserves_accepted_prefix(monkeypatch):
    graph, _ = _source()
    graph.nodes[1]["glyph_history"] = ["AL", "IL", "EN"]
    reset_grammar_state_from_history(graph.nodes[1])
    assert validate_mutation_runtime_gate(graph.nodes[1], graph.graph).threshold.crossed
    dispatched = []
    snapshots = []

    def record(subject, node, glyph, **kwargs):
        apply_glyph(subject, node, glyph, **kwargs)
        dispatched.append((node, glyph))
        snapshots.append(_snapshot(subject))

    monkeypatch.setattr(selectors, "apply_glyph", record)
    with pytest.raises(
        StructuralGrammarError, match="requires a recent destabilizer"
    ) as rejection:
        selectors._apply_glyph_batch(graph, _supplied_requests, graph.graph["history"])
    assert rejection.value.rule == "U4b"
    assert dispatched == [(0, "ZHIR")]
    assert _snapshot(graph) == snapshots[0]
    assert _phases(graph) == (1.25, 1.0)
    assert "mutation_abstentions" not in graph.graph["history"]
    for key in ("since_AL", "since_EN"):
        assert graph.graph["history"][key] == dict.fromkeys(graph, 1)
    assert graph.nodes[1][OPERATOR_STEP_KEY] == 3
