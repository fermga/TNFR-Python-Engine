"""Supplied Resonance events retain split-phase constituent associations.

The fixed asymmetric doubled-C5 geometry has two different target-twin U3
subsets. A separate symmetric geometry tests internal phase-spread growth.
All independent event predictions are frozen before dispatch. These supplied
states and requests are not Mutation acquisition, default-selector occurrence,
a complete native call or a maintained collective identity. Existing partial-
subset/live-prefix owners retain their paired-target native evidence unchanged.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr, get_attr_str
from tnfr.constants import DEFAULTS
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics import adaptation, coordination, integrators, runtime
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph
from tnfr.operators._phase_gate import resolve_u3_phase_neighbors
from tnfr.operators._resonance_identity import RA_RUNTIME_AMPLIFICATION_TRIGGER
from tnfr.operators.grammar_application import enforce_canonical_grammar
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    U2_DEBT_KEY,
    reset_grammar_state_from_history,
)

TOL = 5e-15  # Represented event arithmetic, not an exact-circle enclosure.
PHASES = (1.0, 1.5, 2.5, 2.75, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25)
SUBSETS = ((2, 8, 9), (2, 3, 8, 9))
FACTORS = DEFAULTS["GLYPH_FACTORS"]
MIX = FACTORS["RA_epi_diff"]
COUPLING = FACTORS["RA_phase_coupling"]
BOOST = FACTORS["RA_vf_amplification"]


@pytest.fixture(scope="module", autouse=True)
def no_continuous_or_complete_native_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Split Resonance controls must remain finite event studies")

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


def _source(association=1, *, cancellation=False, symmetric=False):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for block in range(5):
        for left in (2 * block, 2 * block + 1):
            for right in (2 * ((block + 1) % 5), 2 * ((block + 1) % 5) + 1):
                graph.add_edge(left, right, weight=1.0)
    forms = [0.25] * 10
    forms[:4] = (7 / 32, 9 / 32, 0.5 - association / 8, 0.5 + association / 8)
    forms[8:] = (3 / 16, -9 / 16) if cancellation else (3 / 16, 5 / 16)
    phases = list(PHASES)
    if symmetric:
        phases[:4] = (15 / 8, 17 / 8, 0.5, 3.5)
        phases[4:8] = [2.0] * 4
        phases[8:] = (0.5, 3.5)
    graph.graph.update(
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_HYSTERESIS_WINDOW=10,
        RANDOM_SEED=20261018,
        COLLECT_RA_METRICS=True,
        TRACK_NETWORK_COHERENCE=False,
        _t=1.0,
    )
    for node in graph:
        data = graph.nodes[node]
        data.update(
            EPI=forms[node],
            nu_f=(7 / 8, 9 / 8)[node] if node < 2 else 1.0,
            theta=phases[node],
            delta_nfr=0.125,  # Supplied held pressure; no freshness claim.
            epi_kind="seed",
            glyph_history=["AL", "IL"],
            **{OPERATOR_STEP_KEY: 2},
        )
        reset_grammar_state_from_history(data)
    assert tuple(graph) == tuple(range(10))
    assert all(graph.degree(node) == 4 for node in graph)
    assert max(phases) - min(phases) < math.pi
    return graph


def _primitive(graph):
    return tuple(
        (
            get_attr(graph.nodes[node], ALIAS_EPI),
            get_attr(graph.nodes[node], ALIAS_THETA),
            get_attr(graph.nodes[node], ALIAS_VF),
        )
        for node in graph
    )


def _forecast(state, node):
    # Independent four-neighbor construction on the declared doubled cycle.
    block = node // 2
    neighbors = tuple(
        2 * neighbor + member
        for neighbor in ((block - 1) % 5, (block + 1) % 5)
        for member in (0, 1)
    )
    form, phase, capacity = state[node]
    selected = tuple(
        neighbor
        for neighbor in neighbors
        if abs(state[neighbor][1] - phase) <= math.pi / 2
    )
    assert selected
    selected_mean = sum(state[neighbor][0] for neighbor in selected) / len(selected)
    real = sum(math.cos(state[neighbor][1]) for neighbor in selected)
    imag = sum(math.sin(state[neighbor][1]) for neighbor in selected)
    angle = math.atan2(imag, real)
    if angle < 0:
        angle += math.tau
    result = (
        (1 - MIX) * form + MIX * selected_mean,
        phase + COUPLING * (angle - phase),
        (
            capacity * (1 + BOOST)
            if abs(selected_mean) > RA_RUNTIME_AMPLIFICATION_TRIGGER
            else capacity
        ),
    )
    assert -1 < result[0] < 1  # No clipping branch in these controls.
    return tuple(sorted(selected)), selected_mean, angle, result


def _shared_subset(graph, node):
    return resolve_u3_phase_neighbors(
        graph.graph,
        graph.nodes[node]["theta"],
        graph.neighbors(node),
        phase_getter=lambda neighbor: graph.nodes[neighbor]["theta"],
        operator_code="RA",
    )


def _strict_resonance(graph, node):
    requested = enforce_canonical_grammar(graph, node, "RA")
    assert requested == "RA"
    apply_glyph(graph, node, requested, window=10)


def _check_event(graph, before, node, expected):
    selected, selected_mean, _, values = expected
    assert _primitive(graph)[node] == pytest.approx(values, abs=TOL, rel=0)
    data = graph.nodes[node]
    assert get_attr_str(data, ALIAS_EPI_KIND) == "seed"
    assert get_attr(data, ALIAS_DNFR) == get_attr(before[node], ALIAS_DNFR)
    assert tuple(data["glyph_history"]) == ("AL", "IL", "RA")
    assert data[OPERATOR_STEP_KEY] == 3 and data[U2_DEBT_KEY] == 0
    assert get_attr_str(data, ALIAS_SOURCE_GLYPH) == "RA"
    metrics = graph.graph["ra_metrics"][-1]
    assert metrics["neighbors_influenced"] == len(selected)
    assert metrics["epi_propagated"] == pytest.approx(selected_mean, abs=TOL, rel=0)
    assert metrics["identity_preserved"] is True
    assert graph.graph["_t"] == 1.0


@pytest.fixture(scope="module")
def admitted_pairs():
    # Freeze both association forecasts before either source executes events.
    sources = tuple(_source(association) for association in (1, -1))
    predictions = tuple(
        tuple(_forecast(_primitive(graph), node) for node in (0, 1))
        for graph in sources
    )
    records = []
    for graph, expected in zip(sources, predictions, strict=True):
        before = deepcopy(dict(graph.nodes(data=True)))
        initial = _primitive(graph)
        for node in (0, 1):
            assert tuple(sorted(_shared_subset(graph, node).neighbors)) == SUBSETS[node]
            _strict_resonance(graph, node)
            _check_event(graph, before, node, expected[node])
            if node == 0:
                assert graph.nodes[1] == before[1]
        records.append((graph, before, initial, expected))
    return tuple(records)


def test_target_twins_have_distinct_individual_u3_subsets(admitted_pairs):
    _, _, initial, expected = admitted_pairs[0]
    assert tuple(item[0] for item in expected) == SUBSETS
    assert 2 in SUBSETS[0] and 3 not in SUBSETS[0]
    # The two neighbors belong to one pair, but lie on opposite sides of U3.
    assert abs(initial[2][1] - initial[0][1]) == 1.5 < math.pi / 2
    assert abs(initial[3][1] - initial[0][1]) == 1.75 > math.pi / 2
    assert SUBSETS[0] != SUBSETS[1]


@pytest.mark.parametrize("case", (0, 1))
def test_each_admitted_twin_uses_its_own_form_phase_and_capacity_proposal(
    admitted_pairs, case
):
    graph, before, initial, expected = admitted_pairs[case]
    for node in (0, 1):
        assert _primitive(graph)[node] == pytest.approx(
            expected[node][3], abs=TOL, rel=0
        )
        assert _primitive(graph)[node][0] > 0
        assert _primitive(graph)[node][2] > initial[node][2]
    assert len(graph.graph["ra_metrics"]) == 2
    assert all(graph.nodes[node] == before[node] for node in range(2, 10))


def test_same_block_means_do_not_determine_selected_signed_form(admitted_pairs):
    first, second = admitted_pairs
    for block in range(5):
        assert sum(first[2][node][0] for node in (2 * block, 2 * block + 1)) == sum(
            second[2][node][0] for node in (2 * block, 2 * block + 1)
        )
    assert first[3][0][1] == pytest.approx(7 / 24, abs=TOL, rel=0)
    assert second[3][0][1] == 3 / 8
    assert first[3][1][1] == second[3][1][1] == 3 / 8
    actual_difference = _primitive(second[0])[0][0] - _primitive(first[0])[0][0]
    assert actual_difference == pytest.approx(MIX / 12, abs=TOL, rel=0)
    assert actual_difference > 0.01
    assert _primitive(first[0])[1] == _primitive(second[0])[1]


def test_selected_signed_cancellation_controls_capacity_not_mean_magnitudes():
    sources = tuple(_source(sign, cancellation=True) for sign in (1, -1))
    predictions = tuple(
        tuple(_forecast(_primitive(graph), node) for node in (0, 1))
        for graph in sources
    )
    assert predictions[0][0][1] == 0
    assert predictions[1][0][1] == pytest.approx(1 / 12, abs=TOL, rel=0)
    assert predictions[0][1][1] == predictions[1][1][1] == 5 / 32
    capacities = []
    for graph, expected in zip(sources, predictions, strict=True):
        before = deepcopy(dict(graph.nodes(data=True)))
        assert sum(abs(graph.nodes[n]["EPI"]) for n in SUBSETS[0]) > 1
        for node in (0, 1):
            _strict_resonance(graph, node)
            _check_event(graph, before, node, expected[node])
        capacities.append(get_attr(graph.nodes[0], ALIAS_VF))
        assert get_attr(graph.nodes[1], ALIAS_VF) == (9 / 8) * (1 + BOOST)
    # A different trigger on each twin creates a capacity-contrast source.
    first = _primitive(sources[0])
    assert (first[0][2] + first[1][2]) / 2 == pytest.approx(1 + BOOST * 9 / 16)
    assert (first[1][2] - first[0][2]) / 2 == pytest.approx(1 / 8 + BOOST * 9 / 16)
    assert capacities == [7 / 8, (7 / 8) * (1 + BOOST)]


def test_adjacent_later_target_consumes_accepted_prefix_values():
    graph = _source()
    before = deepcopy(dict(graph.nodes(data=True)))
    initial = _primitive(graph)
    first = _forecast(initial, 0)
    after_first_prediction = list(initial)
    after_first_prediction[0] = first[3]
    second = _forecast(tuple(after_first_prediction), 2)
    stale_second = _forecast(initial, 2)
    assert second[0] == stale_second[0] == (0, 1, 4, 5)
    assert abs(second[3][0] - stale_second[3][0]) > 1e-5
    assert abs(second[3][1] - stale_second[3][1]) > 1e-8
    _strict_resonance(graph, 0)
    _check_event(graph, before, 0, first)
    assert tuple(sorted(_shared_subset(graph, 2).neighbors)) == second[0]
    _strict_resonance(graph, 2)
    _check_event(graph, before, 2, second)
    assert _primitive(graph)[2] != pytest.approx(stale_second[3], abs=TOL, rel=0)


def test_distinct_target_subsets_can_increase_internal_phase_contrast():
    graph = _source(symmetric=True)
    initial = _primitive(graph)
    expected = tuple(_forecast(initial, node) for node in (0, 1))
    assert tuple(item[0] for item in expected) == ((2, 8), (3, 9))
    assert Q(11, 8) < Q(math.pi) / 2 < Q(13, 8)
    predicted_contrast = 1 / 8 + COUPLING * 11 / 8
    for node in (0, 1):
        _strict_resonance(graph, node)
        assert _primitive(graph)[node] == pytest.approx(
            expected[node][3], abs=TOL, rel=0
        )
    contrast = (_primitive(graph)[1][1] - _primitive(graph)[0][1]) / 2
    assert contrast == pytest.approx(predicted_contrast, abs=TOL, rel=0)
    assert contrast > 1 / 8
    # The paired-neighbor convex-hull argument does not imply contraction
    # between twins that use different compatible subsets.


@pytest.mark.parametrize("coupling", (COUPLING, 1.0))
def test_common_selected_mean_attains_order_preservation_bound(coupling):
    graph = _source()
    graph.graph["GLYPH_FACTORS"] = dict(FACTORS, RA_phase_coupling=coupling)
    for neighbor in (2, 3, 8, 9):
        graph.nodes[neighbor]["theta"] = 1.25
    expected = (
        (1 - coupling) * 1.0 + coupling * 1.25,
        (1 - coupling) * 1.5 + coupling * 1.25,
    )
    for node in (0, 1):
        assert tuple(sorted(_shared_subset(graph, node).neighbors)) == (2, 3, 8, 9)
        _strict_resonance(graph, node)
    phases = tuple(_primitive(graph)[node][1] for node in (0, 1))
    assert phases == pytest.approx(expected, abs=TOL, rel=0)
    contrast = (phases[1] - phases[0]) / 2
    assert contrast == pytest.approx((1 - coupling) / 4, abs=TOL, rel=0)
    if coupling < 1:
        assert contrast > 0
    else:
        assert contrast == pytest.approx(0.0, abs=TOL, rel=0)


@pytest.mark.parametrize("failure", ("sign", "kind"))
def test_rejected_second_identity_keeps_the_accepted_first_target(failure):
    graph = _source()
    if failure == "sign":
        graph.nodes[1]["EPI"] = -1 / 128
    else:
        graph.nodes[1]["epi_kind"] = "other"
    initial = _primitive(graph)
    first, second = (_forecast(initial, node) for node in (0, 1))
    assert second[0] == SUBSETS[1]
    if failure == "sign":
        assert initial[1][0] < 0 < second[3][0]
        reason = "nonzero_epi_sign_would_flip"
    else:
        assert initial[1][0] > 0 and second[3][0] > 0
        assert initial[3][0] > abs(second[3][0])
        reason = "established_epi_kind_would_change"
    before = deepcopy(dict(graph.nodes(data=True)))
    _strict_resonance(graph, 0)
    _check_event(graph, before, 0, first)
    after_first = deepcopy(
        (
            dict(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            graph.graph["ra_metrics"],
        )
    )
    with pytest.raises(TNFRValueError, match=reason):
        _strict_resonance(graph, 1)
    assert (
        dict(graph.nodes(data=True)),
        tuple(graph.edges(data=True)),
        graph.graph["ra_metrics"],
    ) == after_first
    assert tuple(graph.nodes[1]["glyph_history"]) == ("AL", "IL")
    assert graph.nodes[1][OPERATOR_STEP_KEY] == 2
    assert graph.graph["_t"] == 1.0
    # Direct dispatch may cache adapters during validation. This is target
    # channel/history preservation, not a whole-graph transaction guarantee.
