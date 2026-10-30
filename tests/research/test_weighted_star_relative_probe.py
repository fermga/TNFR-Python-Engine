"""Independent static preparation checks; no continuation or reserved response."""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.research import weighted_star_relative_probe as probe
from tnfr.research._recorded_graph_containers import decode_recorded_container
from tnfr.research.artifact_io import decode_exact_tree, encode_exact_tree
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_dumps, json_loads


def _wire(value):
    return json_dumps(encode_exact_tree(value), sort_keys=True)


@pytest.fixture(scope="module")
def preparation():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/weighted_star_registration_episode/preparation.json"
    )
    return decode_exact_tree(json_loads(path.read_bytes()))


@pytest.fixture(scope="module")
def captured(preparation):
    before = _wire(preparation)
    calls = []
    execute = probe.execute_network_operator_stage

    def stage(graph, operator, targets, **kwargs):
        calls.append((operator.glyph.value, targets, deepcopy(kwargs)))
        assert graph.graph["GRAMMAR_REJECTION_MODE"] == "raise"
        assert graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] is True
        assert graph.graph["_t"] == 7
        assert graph.graph["GAMMA"] == {"type": "none"}
        assert set(kwargs) == {"sequence_context"}
        return execute(graph, operator, targets, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("a frozen source producer must not be replayed")

    with (
        patch.object(probe, "execute_network_operator_stage", stage),
        patch("tnfr.research.weighted_star_contact.contact_source", forbidden),
        patch("tnfr.research.weighted_star_contact.execute_contact_word", forbidden),
    ):
        report = probe.prepare_weighted_star_relative_probe(preparation)
    assert _wire(preparation) == before
    return report, calls


def _changed_record(captured, branch, prefix, change):
    record = deepcopy(captured[0].record)
    row = record["branches"][branch]
    states = list(row["static_states"])
    graph = restore_recorded_graph_state(states[prefix]).graph
    change(graph)
    states[prefix] = record_graph_state(graph)
    row["static_states"] = tuple(states)
    return record


def test_only_declared_new_words_execute(captured):
    report, calls = captured
    assert [(glyph, targets) for glyph, targets, _ in calls] == [
        (glyph, (donor,))
        for donor in (5, 0)
        for _boost in (Q(1, 1024), Q(1, 512))
        for glyph in ("AL", "SHA")
    ]
    assert [(b.donor, b.boost) for b in report.branches] == [
        (5, Q(1, 1024)),
        (5, Q(1, 512)),
        (0, Q(1, 1024)),
        (0, Q(1, 512)),
    ]


@pytest.mark.parametrize("branch", range(4))
def test_exact_events_and_independent_complete_rows(captured, branch):
    report = captured[0]
    result = report.branches[branch]
    donor, boost = result.donor, result.boost
    source = report.source
    strength = sum(
        weight
        for edge, weight in zip(source.edges, source.conductances)
        if donor in edge
    )
    gradient = sum(
        weight * (source.epi[donor] - source.epi[j if i == donor else i])
        for (i, j), weight in zip(source.edges, source.conductances)
        if donor in (i, j)
    )
    assert result.actual_increment == boost
    assert result.event_work == (boost * gradient + strength * boost**2 / 2, 0)
    for index, fields in enumerate(result.prefixes):
        expected_x = list(source.epi)
        expected_nu = list(source.capacity)
        if index:
            expected_x[donor] += boost
        if index == 2:
            expected_nu[donor] /= 2
        assert fields.epi == tuple(expected_x)
        assert fields.phase == source.phase
        assert fields.capacity == tuple(expected_nu)
        assert (fields.edges, fields.conductances) == (
            source.edges,
            source.conductances,
        )
        expected_storage = Q(0)
        for (i, j), weight in zip(fields.edges, fields.conductances):
            expected_storage += weight * (expected_x[i] - expected_x[j]) ** 2 / 2
        assert fields.form_storage == expected_storage
        assert fields.phase_storage == source.phase_storage
        assert fields.storage.hi < Q(29, 64)
        numerators = []
        for node in fields.nodes:
            current, form_gradient, degree = I(0), Q(0), 0
            for (i, j), weight in zip(fields.edges, fields.conductances):
                if node not in (i, j):
                    continue
                other = j if node == i else i
                degree += 1
                current += sin(I(fields.phase[other] - fields.phase[node]))
                form_gradient += weight * (expected_x[node] - expected_x[other])
            expected_pressure = current / (degree * pi_interval())
            assert fields.pressure[node].contains(expected_pressure.midpoint)
            assert fields.form_rates[node].contains(
                (expected_nu[node] * expected_pressure).midpoint
            )
            numerators.append(expected_nu[node] * form_gradient / degree)
        assert fields.phase_rate_numerators() == tuple(numerators)
        assert fields.storage_rate.contains(0)
        assert result.stored_pressure[index] == (Q(0),) * 10
    # Stored zero pressure is retained, whereas the declared continuing law is rebuilt.
    assert result.prefixes[-1].pressure[4].lo > 0


def test_actual_pair_differences_and_receiver_ratio(captured):
    report = captured[0]
    gains = []
    for low, high in (report.branches[:2], report.branches[2:]):
        left, right = low.prefixes[-1], high.prefixes[-1]
        assert left.capacity == right.capacity
        assert left.phase == right.phase
        differences = tuple(b - a for a, b in zip(left.epi, right.epi))
        assert differences == tuple(
            Q(1, 1024) if i == low.donor else 0 for i in range(10)
        )
        gains.append(right.phase_rate_numerators()[4] - left.phase_rate_numerators()[4])
    weights = dict(zip(report.source.edges, report.source.conductances))
    assert gains[1] == -Q(13, 32) * 2 * Q(1, 1024)
    assert gains[0] / gains[1] == weights[(4, 5)] / 2
    assert (
        report.branches[0].prefixes[-1].capacity
        != report.branches[2].prefixes[-1].capacity
    )


def test_full_lifecycle_and_history_context(captured, preparation):
    source = restore_recorded_graph_state(preparation["static_states"][-1]).graph
    for branch in captured[0].record["branches"]:
        donor = branch["donor"]
        for index, state in enumerate(branch["static_states"]):
            graph = restore_recorded_graph_state(state).graph
            for node in graph:
                if node != donor:
                    assert probe._signature(graph.nodes[node]) == probe._signature(
                        source.nodes[node]
                    )
            target = graph.nodes[donor]
            assert tuple(target["glyph_history"]) == ("AL", "SHA")[:index]
            if index:
                assert target["_operator_step"] == index
                assert target["_grammar_u2_debt"] == 0
                assert target["_grammar_prior_coherence"] is False
                assert target["emission_timestamp"] == "2030-01-02T03:04:06+00:00"
                assert target["_structural_lineage"]["activation_count"] == 1
            if index == 2:
                assert target["latent"] is True
                assert target["preserved_epi"] == float(Q(5, 16) + branch["boost"])
                assert target["latency_start_time"] == target["emission_timestamp"]


def test_roundtrip_reader_never_replays_events(preparation, captured):
    wire = _wire(captured[0].record)
    record = decode_exact_tree(json_loads(wire))
    with patch.object(
        probe,
        "execute_network_operator_stage",
        side_effect=AssertionError("event replay"),
    ):
        rebuilt = probe.admit_weighted_star_relative_probe(preparation, record)
    assert rebuilt.source == captured[0].source
    assert rebuilt.branches == captured[0].branches
    assert _wire(rebuilt.record) == wire
    # The report record is detached from its input.
    record["protocol"]["structural_time"] = -1
    assert rebuilt.record["protocol"]["structural_time"] == 7


@pytest.mark.parametrize(
    "change",
    [
        lambda g: g.graph.__setitem__("EPI_LATENT_MAX", True),
        lambda g: g.graph.__setitem__("VALIDATE_OPERATOR_PRECONDITIONS", False),
        lambda g: g.graph.__setitem__("GRAMMAR_REJECTION_MODE", "warn"),
        lambda g: g.graph.__setitem__("_t", 8),
        lambda g: g.graph.__setitem__("GAMMA", {"type": "constant", "value": 1}),
    ],
)
def test_coherent_all_prefix_config_tampering_rejects(preparation, captured, change):
    record = deepcopy(captured[0].record)
    for branch in record["branches"]:
        states = []
        for state in branch["static_states"]:
            graph = restore_recorded_graph_state(state).graph
            change(graph)
            states.append(record_graph_state(graph))
        branch["static_states"] = tuple(states)
    with pytest.raises(ValueError, match="persistent prefix|Gamma"):
        probe.admit_weighted_star_relative_probe(preparation, record)


@pytest.mark.parametrize(
    "change",
    [
        lambda g: g.nodes[5].__setitem__("EPI", True),
        lambda g: g.nodes[5].__setitem__("nu_f", True),
        lambda g: g.nodes[5].__setitem__("theta", True),
        lambda g: g.nodes[5].__setitem__("delta_nfr", True),
        lambda g: g[4][5].__setitem__("weight", True),
    ],
)
def test_malformed_primitive_rejected_before_record_association(
    preparation, captured, change
):
    record = _changed_record(captured, 0, 2, change)
    with pytest.raises((ValueError, TypeError)) as error:
        probe.admit_weighted_star_relative_probe(preparation, record)
    assert "persistent prefix" not in str(error.value)


@pytest.mark.parametrize(
    "change",
    [
        lambda g: g.nodes[5].__setitem__("_operator_step", True),
        lambda g: g.nodes[5].__setitem__("_grammar_u2_debt", 1),
        lambda g: g.nodes[5].__setitem__("glyph_history", ["AL", "SHA"]),
        lambda g: g.nodes[0]["glyph_history"].append("IL"),
        lambda g: g.nodes[4]["glyph_history"].append("IL"),
        lambda g: g.nodes[5].__setitem__(
            "emission_timestamp", "2030-01-02T03:04:05+00:00"
        ),
        lambda g: g.nodes[5].__setitem__("preserved_epi", 0.5),
        lambda g: g.nodes[5].__setitem__("delta_nfr", 0.125),
        lambda g: g.nodes[6].__setitem__("theta", 0.5),
        lambda g: g[4][5].__setitem__("length", 2),
    ],
)
def test_unapproved_full_context_change_rejects(preparation, captured, change):
    record = _changed_record(captured, 0, 2, change)
    with pytest.raises(ValueError, match="persistent prefix"):
        probe.admit_weighted_star_relative_probe(preparation, record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", True),
        ("word_index", True),
        ("glyph", "IL"),
        ("event_timestamp", "2031-01-01"),
        ("structural_time", 8),
    ],
)
def test_stage_context_rejection(preparation, captured, field, value):
    record = deepcopy(captured[0].record)
    record["branches"][0]["stages"][1][field] = value
    with pytest.raises(ValueError, match="stage context"):
        probe.admit_weighted_star_relative_probe(preparation, record)


def test_source_association_and_cached_verdicts_cannot_substitute(
    preparation, captured
):
    record = deepcopy(captured[0].record)
    record["source_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source association"):
        probe.admit_weighted_star_relative_probe(preparation, record)
    record = deepcopy(captured[0].record)
    record["identity_admitted"] = True
    with pytest.raises(ValueError, match="record fields"):
        probe.admit_weighted_star_relative_probe(preparation, record)


def test_frozen_source_is_readmitted_even_if_branch_record_matches(
    preparation, captured
):
    changed = deepcopy(preparation)
    states = []
    for state in changed["static_states"]:
        graph = restore_recorded_graph_state(state).graph
        graph.graph["EPI_LATENT_MAX"] = True
        states.append(record_graph_state(graph))
    changed["static_states"] = tuple(states)
    with pytest.raises(ValueError):
        probe.admit_weighted_star_relative_probe(changed, captured[0].record)


def test_snapshots_preserve_container_types(captured):
    record = captured[0].record
    for branch in record["branches"]:
        for state in branch["static_states"]:
            data = decode_recorded_container(state)
            assert type(data["nodes"]) is tuple
            assert type(data["graph"]["VALIDATE_OPERATOR_PRECONDITIONS"]) is bool
    # Mutating one captured prefix cannot change a different branch or prefix.
    copy = deepcopy(record)
    before = _wire(copy["branches"][1])
    copy["branches"][0]["stages"][0]["target"] = -1
    assert _wire(copy["branches"][1]) == before
