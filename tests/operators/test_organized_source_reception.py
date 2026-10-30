"""One frozen Reception prefix on the signed-reflection source sector.

Preparation: actual ordered 45-node support, x0=-x1=1/64, other forms zero,
and the earlier opposed child-1/2 offset profile with m=1/40, epsilon=1/100.
Held unit capacities, tau clock, default EN mix and hard rails [-1,1] are
fixed before evaluating one atomic (0,1) stage and serial public calls 0->1.
AL/IL context, kind and zero stored dEPI are supplied, not acquired history.
Runtime phase is a declared binary64 rendering; the exact offset lift remains
separate. No continuous response, selector or source producer is executed.
"""

from copy import deepcopy
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.alias import get_attr, get_attr_str, set_dnfr
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.config.thresholds import DNFR_RECEPTION_MAX, EPI_SATURATION_MAX
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_EPI_KIND, ALIAS_SOURCE_GLYPH
from tnfr.dynamics import integrators, runtime
from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import Coherence, Reception, Silence
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    U2_DEBT_KEY,
    read_grammar_state,
    reset_grammar_state_from_history,
)
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as owner
from tnfr.types import Glyph, require_finite_real_scalar_epi

TARGETS = (0, 1)
ORIGIN, EPSILON, AMPLITUDE = Q(1, 40), Q(1, 100), Q(1, 64)
FORM = (AMPLITUDE, -AMPLITUDE) + (Q(0),) * 43
OFFSETS = tuple(
    ORIGIN
    + (EPSILON if c == 1 else -EPSILON if c == 2 else 0)
    * (1 if j == 0 else -1 if j == 1 else 0)
    for c in range(5)
    for j in range(9)
)
KIND = "organized-source"
MIX = Q(resolve_runtime_operator_factors(None, Glyph.EN, {})["EN_mix"])


@pytest.fixture(scope="module", autouse=True)
def no_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Reception prefix cannot run flow or rebuild a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (capture, "_assess_organized_source_capture"),
            (capture, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _forms(graph):
    return tuple(
        Q(
            require_finite_real_scalar_epi(
                get_attr(
                    graph.nodes[n], ALIAS_EPI, strict=True, conv=lambda value: value
                )
            )
        )
        for n in graph
    )


def _neighbors(geometry, node):
    return tuple(
        b if a == node else a for a, b, _ in geometry.weighted_edges if node in (a, b)
    )


def _laplacian(geometry, values):
    result = [Q(0)] * 45
    for a, b, w in geometry.weighted_edges:
        difference = w * (values[a] - values[b])
        result[a] += difference
        result[b] -= difference
    return tuple(result)


def _freeze_prediction(geometry):
    """Independent ideal endpoint, order defect and work before any dispatch."""
    # A prospective binary64 budget, not fitted to an evaluated residual:
    # h is dyadic and beta<1/4. Mean, coefficient subtraction, product and sum
    # rounding give each atomic scalar error <2*u*h (u=2^-53). The serial
    # second row inherits beta/5 of the first error, so its difference from
    # the atomic row has error <5*u*h, below the declared comparison allowance.
    assert 0 < MIX < Q(1, 4)
    assert 5 * Q(1, 2**53) * AMPLITUDE < Q(1, 10**17)
    assert all(weight == 1 for _, _, weight in geometry.weighted_edges)
    assert len(geometry.weighted_edges) == 90
    assert tuple(len(_neighbors(geometry, n)) for n in TARGETS) == (5, 5)
    gradient = _laplacian(geometry, FORM)
    assert gradient[0] == -gradient[1] == 6 * AMPLITUDE
    contraction = 1 - 6 * MIX / 5
    endpoint = (contraction * AMPLITUDE, -contraction * AMPLITUDE) + FORM[2:]
    serial_defect = -6 * MIX**2 * AMPLITUDE / 25
    serial = (endpoint[0], endpoint[1] + serial_defect) + FORM[2:]
    work = sum(
        w * ((endpoint[a] - endpoint[b]) ** 2 - (FORM[a] - FORM[b]) ** 2) / 2
        for a, b, w in geometry.weighted_edges
    )
    assert 0 < contraction < 1 and work < 0
    assert work == 6 * AMPLITUDE**2 * (contraction**2 - 1)
    assert -gradient[1] / 4 < Q(DNFR_RECEPTION_MAX)
    assert max(FORM) < Q(EPI_SATURATION_MAX)
    assert all(-1 < value < 1 for value in (*FORM, *endpoint, *serial))
    return endpoint, serial, work, serial_defect


def _read_fine(graph, geometry, represented_phase):
    """Verify the held execution fields before consuming the separate lift."""
    assert not graph.is_directed() and not graph.is_multigraph()
    assert all(type(node) is int for node in graph)
    assert tuple(graph) == geometry.node_order
    assert graph.number_of_edges() == len(geometry.weighted_edges)
    for node in graph:
        assert tuple(graph.neighbors(node)) == _neighbors(geometry, node)
        assert graph.nodes[node]["theta"] == represented_phase[node]
        assert type(graph.nodes[node]["theta"]) is float
        assert type(graph.nodes[node]["nu_f"]) is float
        assert graph.nodes[node]["nu_f"] == 1.0
    for a, b, weight in geometry.weighted_edges:
        assert type(graph[a][b]["weight"]) is float
        assert Q(graph[a][b]["weight"]) == weight
    return owner._observe_organized_source_port(
        form=_forms(graph), phase_offsets=OFFSETS
    )


def _pressure(graph):
    return tuple(graph.nodes[n]["delta_nfr"] for n in graph)


def _store_pressure(graph, observation):
    # In the declared tau clock, unit capacity makes p equal the full form row.
    for node, rate in zip(graph, observation.form_tau_rate_bounds, strict=True):
        set_dnfr(graph, node, float(rate.midpoint))


@pytest.fixture(scope="module")
def execution():
    geometry = owner._derive_organized_source_port()
    prediction = _freeze_prediction(geometry)
    represented_phase = tuple(
        float(geometry.alpha_bounds.midpoint * (n % 9 - 4) + OFFSETS[n])
        for n in range(45)
    )
    graph = nx.Graph()
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        COLLECT_OPERATOR_METRICS=True,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        RANDOM_SEED=17,
        _t=7.0,
    )
    for node in geometry.node_order:
        graph.add_node(
            node,
            EPI=float(FORM[node]),
            EPI_kind=KIND,
            theta=represented_phase[node],
            nu_f=1.0,
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=["AL", "IL"],
            **{OPERATOR_STEP_KEY: 2},
        )
        reset_grammar_state_from_history(graph.nodes[node])
    for a, b, weight in geometry.weighted_edges:
        graph.add_edge(a, b, weight=float(weight))
    initial = _read_fine(graph, geometry, represented_phase)
    _store_pressure(graph, initial)
    sequence = ValidatedSequence(
        (Reception(), Coherence(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    context = sequence.step(0)
    assert all(validate_candidate(graph, n, "EN").allowed for n in TARGETS)
    atomic, serial = deepcopy(graph), deepcopy(graph)
    calls = {"atomic": [], "serial": []}

    def refresh(which):
        def callback(candidate):
            fresh = _read_fine(candidate, geometry, represented_phase)
            calls[which].append(
                (
                    _pressure(candidate),
                    fresh,
                    deepcopy(candidate.graph["operator_metrics"]),
                )
            )
            _store_pressure(candidate, fresh)

        return callback

    with pytest.warns(UserWarning, match="no emission sources detected"):
        stage = execute_network_operator_stage(
            atomic,
            Reception(),
            TARGETS,
            sequence_context=context,
            compute_delta_nfr=refresh("atomic"),
        )
    with pytest.warns(UserWarning, match="no emission sources detected"):
        Reception()(serial, 0, sequence_context=context)
    refresh("serial")(serial)
    serial_prefix = deepcopy(serial)
    with pytest.warns(UserWarning, match="no emission sources detected"):
        Reception()(serial, 1, sequence_context=context)
    refresh("serial")(serial)
    return dict(
        geometry=geometry,
        prediction=prediction,
        phase=represented_phase,
        before=graph,
        initial=initial,
        atomic=atomic,
        serial=serial,
        serial_prefix=serial_prefix,
        stage=stage,
        calls=calls,
        context=context,
    )


def test_atomic_reception_has_nontrivial_signed_successor_with_rounding_scope(
    execution,
):
    data = execution
    graph, ideal = data["atomic"], data["prediction"][0]
    actual = _forms(graph)
    assert actual[0] == -actual[1] and 0 < actual[0] < AMPLITUDE
    assert actual[2:] == FORM[2:]
    assert max(abs(a - b) for a, b in zip(actual, ideal)) < Q(1, 10**17)
    assert actual != ideal  # Binary64 proposals are not the ideal affine map.
    restricted = owner._restrict_organized_source_reflection(
        form=actual, phase_offsets=OFFSETS
    )
    assert restricted.form == actual
    assert sum(actual) == 0
    assert data["stage"].schedule == "two_phase_jacobi"
    assert data["stage"].nodes_processed == 2
    assert all(
        item._proof_fields_are_intact() for item in data["stage"].reception_observations
    )


def test_shared_snapshot_keeps_order_kind_context_and_actual_secondary_writes(
    execution,
):
    data = execution
    before, graph = data["before"], data["atomic"]
    for index, observation in enumerate(data["stage"].reception_observations):
        node = TARGETS[index]
        neighbors = tuple(before.neighbors(node))
        assert observation.target_index == index and observation.node == node
        assert observation.neighbors == neighbors
        assert observation.neighbor_epi_values == tuple(
            float(FORM[n]) for n in neighbors
        )
        assert observation.neighbor_epi_mean == float(
            sum(FORM[n] for n in neighbors) / 5
        )
        assert observation.target_epi_before == float(FORM[node])
        assert observation.target_epi_after == float(_forms(graph)[node])
        assert (
            observation.target_epi_kind_before
            == observation.target_epi_kind_after
            == KIND
        )
        assert (
            observation.reception_sources == observation.reception_sources_after == ()
        )
        assert (
            observation.source_tracking_enabled
            and observation.reception_sources_present_after
        )
        assert observation.source_max_distance == 2
        assert observation.read_boundary == "pre_en_snapshot"
        assert not observation.auxiliary_stability_certified
    for node in graph:
        state = read_grammar_state(graph.nodes[node])
        assert state.history == (
            ("AL", "IL", "EN") if node in TARGETS else ("AL", "IL")
        )
        assert state.operator_step == (3 if node in TARGETS else 2)
        assert state.u2_debt == 0 and state.prior_coherence
        assert get_attr_str(graph.nodes[node], ALIAS_EPI_KIND) == KIND
        assert graph.nodes[node]["dEPI"] == 0.0
        assert "epi_time_history" not in graph.nodes[node]
        if node in TARGETS:
            assert get_attr_str(graph.nodes[node], ALIAS_SOURCE_GLYPH) == "EN"
            assert graph.nodes[node]["_reception_sources"] == []
        else:
            assert "_reception_sources" not in graph.nodes[node]
    assert tuple(graph.edges(data=True)) == tuple(before.edges(data=True))
    assert graph.graph["_t"] == before.graph["_t"] == 7.0
    _read_fine(graph, data["geometry"], data["phase"])


def _contains(mp, bounds, value):
    def convert(q):
        return mp.mpf(q.numerator) / q.denominator

    assert convert(bounds.lo) <= value <= convert(bounds.hi)


@pytest.mark.parametrize("path", ("atomic", "serial"))
def test_unchanged_full_law_refreshes_both_rows_and_accounts_event_work(
    execution, path
):
    data = execution
    _read_fine(data["before"], data["geometry"], data["phase"])
    _read_fine(data[path], data["geometry"], data["phase"])
    endpoint = _forms(data[path])
    jump = owner._observe_organized_source_form_jump(
        form_before=FORM, form_after=endpoint, phase_offsets=OFFSETS
    )
    displacement = tuple(b - a for a, b in zip(FORM, endpoint))
    dq = _laplacian(data["geometry"], displacement)
    work = sum(
        w * ((endpoint[a] - endpoint[b]) ** 2 - (FORM[a] - FORM[b]) ** 2) / 2
        for a, b, w in data["geometry"].weighted_edges
    )
    assert jump.form_displacement == displacement and jump.gradient_jump == dq
    assert jump.storage_jump == work < 0
    initial_gradient = _laplacian(data["geometry"], FORM)
    linear = sum(q * h for q, h in zip(initial_gradient, displacement))
    isolated = sum(
        data["geometry"].support_degrees[n] * displacement[n] ** 2 / 2 for n in TARGETS
    )
    cross = -displacement[0] * displacement[1]
    assert cross != 0 and work == linear + isolated + cross
    assert jump.before.phase_current_bounds == jump.after.phase_current_bounds
    mp = mpmath.mp.clone()
    mp.dps = 80
    gamma = 1 / (1023 * mp.pi)
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("original", mp.mpf(1023) / 1024),
        ("slow", 2 / gamma**2),
    ):
        for node in range(45):
            q = mp.mpf(dq[node].numerator) / dq[node].denominator
            _contains(
                mp,
                getattr(jump, f"form_{clock}_rate_jump_bounds")[node],
                -factor * q / 4,
            )
            _contains(
                mp,
                getattr(jump, f"phase_{clock}_rate_jump_bounds")[node],
                factor * gamma * q / 4,
            )
    assert (
        jump.form_tau_rate_jump_bounds[0].hi < 0
        or jump.form_tau_rate_jump_bounds[0].lo > 0
    )
    assert jump.phase_tau_rate_jump_bounds[0].hi < 0
    fresh = data["calls"][path][-1][1]
    assert fresh.form_gradient == _laplacian(data["geometry"], endpoint)
    assert fresh.form_tau_rate_bounds == jump.after.form_tau_rate_bounds
    assert fresh.phase_tau_rate_bounds == jump.after.phase_tau_rate_bounds
    assert _pressure(data[path]) == tuple(
        float(row.midpoint) for row in fresh.form_tau_rate_bounds
    )


def test_atomic_pressure_refresh_occurs_after_metrics_and_serial_uses_live_prefix(
    execution,
):
    data = execution
    atomic_calls, serial_calls = data["calls"]["atomic"], data["calls"]["serial"]
    assert len(atomic_calls) == 1 and len(serial_calls) == 2
    held, fresh, metrics = atomic_calls[0]
    assert held == _pressure(data["before"])
    assert fresh.form == _forms(data["atomic"])
    assert len(metrics) == 2
    for node, metric in zip(TARGETS, metrics, strict=True):
        assert metric["observed_dnfr"] == held[node]
        assert metric["dnfr_observation_boundary"] == "operator_metrics_collection"
        assert metric["source_absence_observed"] is True
        assert metric["canonical_coherence_certified"] is False
    assert serial_calls[1][0] == tuple(
        float(row.midpoint) for row in serial_calls[0][1].form_tau_rate_bounds
    )
    assert serial_calls[1][2][1]["observed_dnfr"] == serial_calls[1][0][1]
    actual, joint = _forms(data["serial"]), _forms(data["atomic"])
    ideal, defect = data["prediction"][1], data["prediction"][3]
    assert actual[0] == joint[0] and actual[1] < joint[1]
    assert abs(actual[1] - joint[1] - defect) < Q(1, 10**17)
    assert max(abs(a - b) for a, b in zip(actual, ideal)) < Q(1, 10**17)
    assert sum(actual) < 0
    with pytest.raises(ValueError, match="reflection"):
        owner._restrict_organized_source_reflection(form=actual, phase_offsets=OFFSETS)
    assert _read_fine(data["serial"], data["geometry"], data["phase"]).form == actual
    second_metric = data["serial"].graph["operator_metrics"][1]
    assert second_metric["neighbor_epi_mean"] == float(joint[0] / 5)


@pytest.mark.parametrize("failure", ("form", "pressure", "grammar"))
def test_second_target_rejection_rolls_back_the_complete_atomic_prefix(
    execution, failure
):
    graph = deepcopy(execution["before"])
    if failure == "form":
        graph.nodes[1]["EPI"] = EPI_SATURATION_MAX
    elif failure == "pressure":
        graph.nodes[1]["delta_nfr"] = DNFR_RECEPTION_MAX
    else:
        graph.nodes[1][U2_DEBT_KEY] = U2_DEBT_CAPACITY + 1
    nodes, metadata = deepcopy(dict(graph.nodes(data=True))), deepcopy(graph.graph)
    error = StructuralGrammarError if failure == "grammar" else TNFRValueError
    with pytest.raises(error):
        execute_network_operator_stage(
            graph, Reception(), TARGETS, sequence_context=execution["context"]
        )
    assert dict(graph.nodes(data=True)) == nodes
    assert graph.graph == metadata
    assert "_last_operator_applied" not in graph.__dict__


def test_public_second_target_rejection_keeps_the_already_accepted_first_event(
    execution,
):
    graph = deepcopy(execution["serial_prefix"])
    graph.nodes[1]["delta_nfr"] = DNFR_RECEPTION_MAX
    nodes = deepcopy(dict(graph.nodes(data=True)))
    metadata = deepcopy(
        {key: value for key, value in graph.graph.items() if key != "_node_cache"}
    )
    cache = dict(graph.graph["_node_cache"])
    with pytest.raises(TNFRValueError, match="EN precondition failed"):
        Reception()(graph, 1, sequence_context=execution["context"])
    assert dict(graph.nodes(data=True)) == nodes
    assert {
        key: value for key, value in graph.graph.items() if key != "_node_cache"
    } == metadata
    assert graph.graph["_node_cache"] == cache
    assert read_grammar_state(graph.nodes[0]).history == ("AL", "IL", "EN")
    assert read_grammar_state(graph.nodes[1]).history == ("AL", "IL")


@pytest.mark.parametrize("field", ("form_before", "form_after", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, float("inf"), "short"))
def test_form_jump_admits_all_raw_endpoints_before_any_observation(
    field, bad, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid endpoint reached fine-law observation")

    arguments = dict(form_before=FORM, form_after=FORM, phase_offsets=OFFSETS)
    values = arguments[field]
    arguments[field] = values[:-1] if bad == "short" else (bad,) + values[1:]
    monkeypatch.setattr(owner, "_observe_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_source_form_jump(**arguments)
