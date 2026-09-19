"""Shared declared-study orchestration, detached evidence and strict JSON boundaries."""

import json
import math
import threading
from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.alias import set_dnfr, set_theta
from tnfr.sdk import (
    STUDY_TOPOLOGIES,
    TNFR,
    Network,
    StudyResult,
    StudySpec,
    diagnose_network,
    export_to_json,
    import_from_json,
    list_sequences,
    run_study,
    study,
)
from tnfr.sdk.fluent import NAMED_SEQUENCES


@pytest.mark.parametrize(
    "changes",
    [
        {"nodes": 0},
        {"nodes": True},
        {"nodes": 2.0},
        {"cycles": -1},
        {"cycles": False},
        {"cycles": "1"},
        {"seed": None},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"topology": "unregistered"},
        {"topology": []},
        {"sequence": "unregistered"},
        {"sequence": []},
        {"probability": True},
        {"probability": math.nan},
        {"probability": -0.1},
        {"probability": 1.1},
        {"name": " "},
        {"name": 3},
    ],
)
def test_recipe_rejects_invalid_domains_before_execution(changes):
    with pytest.raises((TypeError, ValueError)):
        StudySpec(**changes)


def test_recipe_is_strict_frozen_and_roundtrips():
    spec = StudySpec(nodes=3, topology="random", seed=2**32 - 1, cycles=0)
    assert StudySpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    with pytest.raises(FrozenInstanceError):
        spec.cycles = 5
    with pytest.raises(ValueError, match="Unknown study recipe keys"):
        StudySpec.from_dict({"steps": 2})
    with pytest.raises(TypeError, match="mapping"):
        StudySpec.from_dict([])


def test_catalog_reuses_registry_without_exposing_mutable_words():
    catalog = list_sequences()
    assert {entry["name"] for entry in catalog} == set(NAMED_SEQUENCES)
    for entry in catalog:
        assert entry["operators"] == NAMED_SEQUENCES[entry["name"]]
        assert "live grammar" in entry["scope"]
    selected = list_sequences("basic_activation")
    selected["operators"].clear()
    assert list_sequences("basic_activation")["operators"]
    with pytest.raises(ValueError, match="Unknown registered sequence"):
        list_sequences("missing")


def test_operator_catalog_includes_public_tokens_for_composition():
    from tnfr.operators.operator_contracts import iter_contracts

    contracts = {entry.name: entry for entry in iter_contracts()}
    assert {row["token"] for row in TNFR.operators()} == set(contracts)
    for token, contract in contracts.items():
        row = TNFR.operators(token)
        assert row["token"] == token
        assert row["name"] == contract.english_name
        assert row["glyph"] == contract.glyph


@pytest.mark.parametrize("topology", STUDY_TOPOLOGIES)
def test_zero_cycles_record_supplied_state_and_topology_without_synthetic_pressure(
    topology,
):
    result = run_study(StudySpec(nodes=3, topology=topology, cycles=0)).to_dict()
    assert result["execution"]["cycles_completed"] == 0
    assert "not elapsed physical time" in result["execution"]["clock"]
    assert result["initial_state"] == result["final"]["state"]
    assert len(result["initial_state"]["nodes"]) == 3
    for row in result["initial_state"]["nodes"]:
        assert row["epi"]["value"] == 0.0
        assert row["nu_f"]["value"] == 1.0
        assert row["phase"]["value"] == 0.0
        assert not row["delta_nfr"]["available"]
        assert row["delta_nfr"]["value"] is None
    json.dumps(result, allow_nan=False)


def test_one_cycle_matches_existing_execution_and_installs_shared_runtime_seed():
    spec = StudySpec(nodes=2, seed=37)
    result = run_study(spec).to_dict()
    expected = TNFR.create(2, name=spec.name, seed=spec.seed).ring()
    expected.G.graph["RANDOM_SEED"] = spec.seed
    expected.evolve(steps=1, sequence=spec.sequence)
    assert result["final"] == diagnose_network(expected)
    assert result["execution"]["seed"] == 37
    assert (
        result["execution"]["sequence"]["operators"] == NAMED_SEQUENCES[spec.sequence]
    )
    assert result["initial_state"] != result["final"]["state"]


def test_repeated_seeded_topology_and_word_match_in_current_runtime():
    spec = StudySpec(nodes=4, topology="random", probability=0.8, seed=7, cycles=1)
    assert (
        run_study(spec).to_dict()
        == run_study(StudySpec.from_dict(spec.to_dict())).to_dict()
    )


def test_isolated_random_node_keeps_the_actual_resonance_u3_rejection():
    from tnfr.operators.preconditions import OperatorPreconditionError

    # This declared topology has an isolate. The study wrapper must not insert
    # edges or replace the word to manufacture a successful report.
    with pytest.raises(OperatorPreconditionError, match="U3 phase gate"):
        run_study(StudySpec(nodes=4, topology="random", probability=0.5, seed=7))


def test_execution_failure_propagates_without_diagnostics_or_repair(monkeypatch):
    def refuse(self, **kwargs):
        assert self.G.graph["RANDOM_SEED"] == 12
        assert kwargs == {"steps": 1, "sequence": "basic_activation"}
        raise RuntimeError("operator refused")

    monkeypatch.setattr(Network, "evolve", refuse)
    monkeypatch.setattr(
        study, "diagnose_network", lambda _: pytest.fail("no report after failure")
    )
    with pytest.raises(RuntimeError, match="operator refused"):
        run_study(StudySpec(nodes=2, seed=12))


def test_report_is_detached_and_exports_through_existing_writer(tmp_path):
    original = {"nested": {"value": [1, 2]}}
    report = StudyResult(original)
    original["nested"]["value"].append(3)
    exported = report.to_dict()
    exported["nested"]["value"].clear()
    assert report.to_dict() == {"nested": {"value": [1, 2]}}
    destination = tmp_path / "nested" / "study.json"
    export_to_json(report, destination)
    assert import_from_json(destination) == report.to_dict()
    with pytest.raises(ValueError):
        StudyResult({"value": math.nan})
    with pytest.raises(TypeError, match="mapping"):
        StudyResult([])


def _prepared_network():
    network = TNFR.create(3).path()
    for node in network.G:
        network.G.nodes[node]["EPI"] = -1.0 - node
        set_dnfr(network.G, node, 0.5 + node)
    return network


def test_diagnostics_keep_stored_pressure_and_do_not_touch_graph_or_opaque_resource():
    network = _prepared_network()
    handle = threading.Lock()
    network.G.graph["external_handle"] = handle
    prior_nodes = deepcopy(dict(network.G.nodes(data=True)))
    prior_edges = deepcopy(list(network.G.edges(data=True)))
    prior_metadata = dict(network.G.graph)
    report = diagnose_network(network)
    assert report["nodal"][0]["value"]["expected_depi_dt"] == 0.5
    assert report["state"]["nodes"][0]["epi"]["value"] == -1.0
    assert report["tetrad"]["xi_c"]["provenance"]["method"] in {
        "autocorrelation_fit",
        "spectral_gap",
    }
    assert dict(network.G.nodes(data=True)) == prior_nodes
    assert list(network.G.edges(data=True)) == prior_edges
    assert network.G.graph == prior_metadata
    assert network.G.graph["external_handle"] is handle
    report["state"]["nodes"][0]["epi"]["value"] = 99
    assert network.G.nodes[0]["EPI"] == -1.0


def test_phase_cancellation_retains_independent_fields_and_per_node_status():
    graph = nx.star_graph(4)
    for node, phase in zip(graph, (0.3, 0.0, 0.0, math.pi, -math.pi)):
        graph.nodes[node].update(EPI=1.0, nu_f=1.0, phase=phase, delta_nfr=0.0)
    report = diagnose_network(Network(graph))
    tetrad = report["tetrad"]
    assert tetrad["k_phi"]["available"] is False
    assert tetrad["k_phi"]["complete"] is False
    assert tetrad["k_phi"]["value"][0] is None
    assert tetrad["k_phi"]["node_status"][0] == "undefined_represented_resultant"
    assert all(value is not None for value in tetrad["k_phi"]["value"][1:])
    assert tetrad["phi_s"]["available"] and tetrad["grad_phi"]["available"]
    json.dumps(report, allow_nan=False)


def test_invalid_phase_is_unavailable_without_hiding_pressure_field():
    network = _prepared_network()
    set_theta(network.G, 0, math.nan)
    report = diagnose_network(network)
    assert report["tetrad"]["k_phi"]["value"] is None
    assert report["tetrad"]["grad_phi"]["value"] is None
    assert report["tetrad"]["phi_s"]["available"]
    assert not report["state"]["nodes"][0]["phase"]["available"]
    json.dumps(report, allow_nan=False)


def test_empty_graph_retains_unavailable_xi_provenance():
    report = diagnose_network(Network(nx.Graph()))
    xi = report["tetrad"]["xi_c"]
    assert xi["value"] is None and not xi["available"]
    assert xi["provenance"]["method"] == "unavailable"
    phase = report["metrics"]["avg_phase"]
    assert phase["value"] is None and not phase["available"]
    assert phase["error"]["message"] == "Diagnostic owner reported no available value"
    json.dumps(report, allow_nan=False)


def test_each_field_owner_is_requested_once(monkeypatch):
    calls = []
    for name in (
        "compute_structural_potential",
        "observe_phase_curvature",
        "estimate_coherence_length_with_provenance",
    ):
        original = getattr(study, name)

        def counted(graph, *, original=original, name=name):
            calls.append(name)
            return original(graph)

        monkeypatch.setattr(study, name, counted)
    diagnose_network(Network(nx.Graph()))
    assert sorted(calls) == sorted(
        [
            "compute_structural_potential",
            "observe_phase_curvature",
            "estimate_coherence_length_with_provenance",
        ]
    )


def test_indexed_observations_do_not_merge_distinct_equal_display_labels():
    graph = nx.Graph()
    for node in (1, "1"):
        graph.add_node(node, EPI=0.0, nu_f=1.0, phase=0.0, delta_nfr=0.0)
    graph.add_edge(1, "1")
    report = diagnose_network(Network(graph))
    assert [row["label"] for row in report["state"]["nodes"]] == ["1", "1"]
    assert [row["index"] for row in report["state"]["nodes"]] == [0, 1]
    assert [row["value"]["node"] for row in report["nodal"]] == [0, 1]
    assert report["state"]["edges"] == [[0, 1]]


def test_guide_only_points_to_current_repository_paths(capsys):
    from pathlib import Path

    guide = TNFR.guide()
    assert capsys.readouterr().out == guide + "\n"
    root = Path(__file__).resolve().parents[2]
    for line in guide.splitlines():
        target = line.split(": ")[-1]
        if target.startswith(("docs/", "theory/", "examples/")):
            assert (root / target).is_file(), target
