"""Restricted inheritance of two fine Resonance events by one coarse event.

The supplied doubled-C5 source has paired phases, nonzero associated form and
capacity contrasts, uniform strict form sign/kind and duplicated grammar.
Complete pairs 0 then 1 correspond to two supplied coarse-C5 RA requests with
the same registered factors. The second pair selects one whole neighbor block
and consumes the first pair's live update. Freeze all predictions before any
dispatch. This is an event/grammar projection, not whole-runtime equivalence,
autonomous occurrence, collective identity or a pressure/flow descent claim.
"""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import adaptation, coordination, integrators, runtime
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph
from tnfr.operators._phase_gate import resolve_u3_phase_neighbors
from tnfr.operators.grammar_application import enforce_canonical_grammar
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    read_grammar_state,
    reset_grammar_state_from_history,
)
from tnfr.operators.grammar_types import StructuralGrammarError

MEANS = (1 / 4, 1 / 2, 3 / 8, 1 / 8, 5 / 8)
FORM_CONTRASTS = (1 / 32, 1 / 16, -1 / 32, 1 / 64, -1 / 16)
CAPACITIES = (1, 9 / 8, 7 / 8, 5 / 4, 3 / 4)
CAPACITY_CONTRASTS = (1 / 8, -1 / 16, 1 / 16, -1 / 8, 1 / 32)
PHASES = (1, 5 / 4, 3, 3 / 2, 1 / 4)
FACTORS = DEFAULTS["GLYPH_FACTORS"]
MIX = FACTORS["RA_epi_diff"]
BOOST = FACTORS["RA_vf_amplification"]
COUPLING = FACTORS["RA_phase_coupling"]
TOL = 5e-15  # Represented event arithmetic, not an exact-circle enclosure.


@pytest.fixture(scope="module", autouse=True)
def no_flow_or_complete_runtime():
    def forbidden(*args, **kwargs):
        pytest.fail("Aggregate Resonance checks are supplied finite events only")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (coordination, "coordinate_global_local_phase"),
            (adaptation, "adapt_vf_after_structural_stability"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


def _configure(graph):
    inject_defaults(graph)
    graph.graph.update(
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_HYSTERESIS_WINDOW=10,
        RANDOM_SEED=20261019,
        COLLECT_RA_METRICS=True,
        TRACK_NETWORK_COHERENCE=False,
        _t=1.0,
    )


def _fine_source(sign=1):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for block in range(5):
        for left in (2 * block, 2 * block + 1):
            for right in (2 * ((block + 1) % 5), 2 * ((block + 1) % 5) + 1):
                graph.add_edge(left, right, weight=1.0)
    _configure(graph)
    for block in range(5):
        for offset, sigma in enumerate((-1, 1)):
            data = graph.nodes[2 * block + offset]
            data.update(
                EPI=sign * (MEANS[block] + sigma * FORM_CONTRASTS[block]),
                nu_f=CAPACITIES[block] + sigma * CAPACITY_CONTRASTS[block],
                theta=PHASES[block],
                # Supplied held pressure; no fresh preparation is claimed.
                delta_nfr=1 / 8 + block / 32 + sigma / 64,
                epi_kind="seed",
                glyph_history=["AL", "IL"],
                **{OPERATOR_STEP_KEY: 2},
            )
            reset_grammar_state_from_history(data)
    return graph


def _primitive(graph):
    return tuple(
        tuple(
            get_attr(graph.nodes[n], alias)
            for alias in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA)
        )
        for n in graph
    )


def _project(fine):
    rows = _primitive(fine)
    return tuple(
        tuple((a + b) / 2 for a, b in zip(rows[2 * i], rows[2 * i + 1]))
        for i in range(5)
    )


def _coarse_from_fine(fine):
    graph = nx.cycle_graph(5)
    nx.set_edge_attributes(graph, 1.0, "weight")
    _configure(graph)
    for block, (form, capacity, phase) in enumerate(_project(fine)):
        first, second = fine.nodes[2 * block], fine.nodes[2 * block + 1]
        data = graph.nodes[block]
        data.update(
            EPI=form,
            nu_f=capacity,
            theta=phase,
            delta_nfr=(get_attr(first, ALIAS_DNFR) + get_attr(second, ALIAS_DNFR)) / 2,
        )
        # On the positive domain these contexts are duplicated. Outside it,
        # this explicitly supplied representative is not an automatic quotient.
        for key in (
            "epi_kind",
            "glyph_history",
            OPERATOR_STEP_KEY,
            U2_DEBT_KEY,
            PRIOR_COHERENCE_KEY,
        ):
            data[key] = deepcopy(first[key])
    return graph


def _prediction(rows, block):
    form, capacity, phase = rows[block]
    selected = tuple(
        j
        for j in ((block - 1) % 5, (block + 1) % 5)
        if abs(rows[j][2] - phase) <= math.pi / 2
    )
    mean = sum(rows[j][0] for j in selected) / len(selected)
    angle = math.atan2(
        sum(math.sin(rows[j][2]) for j in selected),
        sum(math.cos(rows[j][2]) for j in selected),
    )
    assert abs(mean) > 1 / 10  # A wide margin above the default trigger.
    proposed = (
        (1 - MIX) * form + MIX * mean,
        (1 + BOOST) * capacity,
        (1 - COUPLING) * phase + COUPLING * angle,
    )
    assert -1 < proposed[0] < 1 and proposed[0] * form > 0
    assert 0 < proposed[1] < DEFAULTS["VF_MAX"]
    after = list(rows)
    after[block] = proposed
    return dict(
        selected=tuple(sorted(selected)),
        mean=mean,
        angle=angle,
        target=proposed,
        after=tuple(after),
    )


def _subset(graph, node):
    result = resolve_u3_phase_neighbors(
        graph.graph,
        graph.nodes[node]["theta"],
        graph.neighbors(node),
        phase_getter=lambda j: graph.nodes[j]["theta"],
        operator_code="RA",
    )
    return tuple(sorted(result.neighbors))


def _strict_ra(graph, node):
    requested = enforce_canonical_grammar(graph, node, "RA")
    assert requested == "RA"
    apply_glyph(graph, node, requested, window=10)


def _grammar(graph, node):
    return read_grammar_state(graph.nodes[node])


@pytest.fixture(scope="module", params=(1, -1), ids=("positive", "negative"))
def inherited_pairs(request, no_flow_or_complete_runtime):
    fine = _fine_source(request.param)
    coarse = _coarse_from_fine(fine)
    initial = _primitive(fine)
    rows = _project(fine)
    first = _prediction(rows, 0)
    second = _prediction(first["after"], 1)
    stale_second = _prediction(rows, 1)
    predictions = (first, second)  # Both frozen before any actual event.
    assert max(PHASES) - min(PHASES) < math.pi
    assert fine.graph["CLIP_MODE"] == coarse.graph["CLIP_MODE"] == "hard"
    assert all(
        fine.graph["EPI_MIN"] < x < fine.graph["EPI_MAX"]
        and fine.graph["VF_MIN"] < v < (1 + BOOST) * v < fine.graph["VF_MAX"]
        for x, v, _ in initial
    )
    assert first["selected"] == (1, 4) and second["selected"] == (0,)
    assert all(
        _grammar(fine, 2 * i) == _grammar(fine, 2 * i + 1) == _grammar(coarse, i)
        for i in range(5)
    )
    pressure = tuple(
        tuple(get_attr(g.nodes[n], ALIAS_DNFR) for n in g) for g in (fine, coarse)
    )
    records = []
    for block, predicted in enumerate(predictions):
        coarse_subset = _subset(coarse, block)
        fine_subsets = tuple(_subset(fine, 2 * block + member) for member in (0, 1))
        _strict_ra(fine, 2 * block)
        singleton = _primitive(fine)[2 * block : 2 * block + 2]
        singleton_grammar = tuple(
            _grammar(fine, 2 * block + member) for member in (0, 1)
        )
        _strict_ra(fine, 2 * block + 1)
        _strict_ra(coarse, block)
        records.append(
            dict(
                fine=_primitive(fine),
                coarse=_primitive(coarse),
                projection=_project(fine),
                singleton=singleton,
                singleton_grammar=singleton_grammar,
                fine_context=tuple(
                    _grammar(fine, 2 * block + member) for member in (0, 1)
                ),
                coarse_context=_grammar(coarse, block),
                predicted=predicted,
                fine_subsets=fine_subsets,
                coarse_subset=coarse_subset,
            )
        )
    return dict(
        fine=fine,
        coarse=coarse,
        initial=initial,
        records=records,
        stale_second=stale_second,
        pressure=pressure,
    )


def _close(actual, expected):
    assert actual == pytest.approx(expected, abs=TOL, rel=0)


def test_complete_pair_means_and_contrasts_match_one_registered_coarse_event(
    inherited_pairs,
):
    c = inherited_pairs
    for block, record in enumerate(c["records"]):
        for fine_mean, coarse, expected in zip(
            record["projection"], record["coarse"], record["predicted"]["after"]
        ):
            _close(fine_mean, expected)
            _close(coarse, expected)
        old_left, old_right = c["initial"][2 * block : 2 * block + 2]
        left, right = record["fine"][2 * block : 2 * block + 2]
        d_before, d_after = (old_right[0] - old_left[0]) / 2, (right[0] - left[0]) / 2
        e_before, e_after = (old_right[1] - old_left[1]) / 2, (right[1] - left[1]) / 2
        _close(d_after, (1 - MIX) * d_before)
        _close(e_after, (1 + BOOST) * e_before)
        assert 0 < abs(d_after) < abs(d_before)
        assert abs(e_after) > abs(e_before) > 0
        _close(left[2], right[2])


def test_whole_block_u3_subsets_and_adjacent_live_propagation(inherited_pairs):
    c = inherited_pairs
    for record in c["records"]:
        blocks = record["predicted"]["selected"]
        expected_fine = tuple(
            2 * block + member for block in blocks for member in (0, 1)
        )
        assert record["coarse_subset"] == blocks
        assert record["fine_subsets"] == (expected_fine, expected_fine)
    assert c["records"][1]["coarse_subset"] == (0,)  # Block 2 remains outside U3.
    live = c["records"][1]["coarse"][1]
    stale = c["stale_second"]["target"]
    assert abs(live[0] - stale[0]) > 1e-3
    assert abs(live[2] - stale[2]) > 1e-4


def test_grammar_projection_is_one_transition_only_at_complete_pair_boundaries(
    inherited_pairs,
):
    for record in inherited_pairs["records"]:
        first, second = record["singleton_grammar"]
        assert first.history == ("AL", "IL", "RA") and second.history == ("AL", "IL")
        left, right = record["singleton"]
        assert abs(left[2] - right[2]) > 1e-3  # Singleton exits the paired chart.
        singleton_phase_mean = (left[2] + right[2]) / 2
        assert abs(singleton_phase_mean - record["predicted"]["target"][2]) > 1e-3
        fine_contexts = record["fine_context"]
        assert fine_contexts[0] == fine_contexts[1] == record["coarse_context"]
        assert record["coarse_context"].operator_step == 3
        assert record["coarse_context"].u2_debt == 0
        assert record["coarse_context"].prior_coherence is True


def test_semantic_projection_does_not_identify_event_counts_or_refresh_pressure(
    inherited_pairs,
):
    c = inherited_pairs
    assert len(c["fine"].graph["ra_metrics"]) == 4
    assert len(c["coarse"].graph["ra_metrics"]) == 2
    for block, record in enumerate(c["records"]):
        coarse_metric = c["coarse"].graph["ra_metrics"][block]
        for fine_metric in c["fine"].graph["ra_metrics"][2 * block : 2 * block + 2]:
            assert (
                fine_metric["neighbors_influenced"]
                == 2 * coarse_metric["neighbors_influenced"]
            )
            _close(fine_metric["epi_propagated"], record["predicted"]["mean"])
            assert (
                fine_metric["identity_preserved"]
                and coarse_metric["identity_preserved"]
            )
    for graph, pressure in zip((c["fine"], c["coarse"]), c["pressure"]):
        assert tuple(get_attr(graph.nodes[n], ALIAS_DNFR) for n in graph) == pressure
        assert all(graph.nodes[n]["epi_kind"] == "seed" for n in graph)
        assert graph.graph["_t"] == 1
        assert all("epi_time_history" not in graph.nodes[n] for n in graph)


@pytest.mark.parametrize("failure", ("sign", "kind", "grammar"))
def test_coarse_admission_does_not_replace_fine_admission_outside_the_domain(failure):
    fine = _fine_source()
    if failure == "sign":
        fine.nodes[1]["EPI"] = -1 / 128
        fine.nodes[0]["EPI"] = 2 * MEANS[0] + 1 / 128
        error, reason = TNFRValueError, "nonzero_epi_sign_would_flip"
    elif failure == "kind":
        fine.nodes[1]["epi_kind"] = "other"
        error, reason = TNFRValueError, "established_epi_kind_would_change"
    else:
        fine.nodes[1][U2_DEBT_KEY] = U2_DEBT_CAPACITY + 1
        error, reason = StructuralGrammarError, "Convergence violation"
    # Always rebuild aggregate channels from the modified fine primitive state.
    # Heterogeneous kind/context uses an explicitly supplied first-member label.
    coarse = _coarse_from_fine(fine)
    expected = _prediction(_project(fine), 0)
    if failure == "sign":
        before = fine.nodes[1]["EPI"]
        assert before < 0 < (1 - MIX) * before + MIX * expected["mean"]
    _strict_ra(coarse, 0)
    _close(_primitive(coarse)[0], expected["target"])
    _strict_ra(fine, 0)
    prefix = deepcopy(
        (
            dict(fine.nodes(data=True)),
            tuple(fine.edges(data=True)),
            fine.graph["ra_metrics"],
        )
    )
    with pytest.raises(error, match=reason):
        _strict_ra(fine, 1)
    assert (
        dict(fine.nodes(data=True)),
        tuple(fine.edges(data=True)),
        fine.graph["ra_metrics"],
    ) == prefix
    assert _grammar(fine, 0).history == ("AL", "IL", "RA")
    assert _grammar(fine, 1).history == ("AL", "IL")
    assert len(fine.graph["ra_metrics"]) == 1
    # Target preservation does not assert rollback of graph adapter caches.
