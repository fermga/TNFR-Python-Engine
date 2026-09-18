"""Portable fork/protocol controls; no actual preparation or native step runs."""

from collections import deque
from fractions import Fraction as F
import hashlib
import json
import sys

import networkx as nx
import numpy as np
import pytest

from benchmarks import thol_regional_restoration as study
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.support_transport import _from_data
from tnfr.types import Glyph


def _fixture():
    graph = nx.Graph()
    parents, children = tuple(range(8)), tuple(f"child_{i}" for i in range(8))
    nodes = parents+children
    for i, node in enumerate(nodes):
        graph.add_node(node, EPI=0.2+0.01*i, nu_f=0.95, theta=0.1*i, delta_nfr=0.0,
                       glyph_history=deque(("AL", "IL"), maxlen=6),
                       epi_time_history=deque(((1.0, 0.1), (1.5, 0.2+0.01*i)), maxlen=8),
                       stable_count=0, nested={"values": [i]})
    graph.add_edges_from((i, (i+1) % 8, {"weight": 1.0}) for i in parents)
    graph.add_edges_from((p, c, {"weight": 1.0}) for p, c in zip(parents, children))
    for p, c in zip(parents, children):
        graph.nodes[p]["sub_nodes"] = [c]
        graph.nodes[c]["parent_node"] = p
    graph.graph.update(_t=1.5, _node_sample=nodes, hierarchy=dict(zip(parents, ([c] for c in children))),
                       history={"C_steps": deque((0.75, 0.8), maxlen=20)}, RANDOM_SEED=3)
    graph.graph["history_alias"] = graph.nodes[children[0]]["epi_time_history"]
    graph.graph["_node_list_cache"] = nodes
    graph._last_operator_applied = "coupling"
    prefix = {"birth": {"parent_children": tuple(zip(parents, children)), "before": {"nodes": parents}},
              "coupling": {"refreshed_forcing": "synthetic-reference-token"}}
    lineage = study.full._birth_families(graph, prefix)
    prior = {"branch": "control", "status": "executed", "endpoint": study.full._state(graph),
             "prefix": prefix, "lineage": lineage}
    index = {node: i for i, node in enumerate(nodes)}
    edges = tuple(sorted((index[u], index[v], F(1)) for u in nodes for v in graph.neighbors(u)))
    support = tuple(tuple(sorted(index[v] for v in graph.neighbors(u))) for u in nodes)
    snapshot = _from_data(nodes, edges, support, tuple(F(graph.nodes[n]["EPI"]) for n in nodes),
                          (F(0.95),)*16, (F(0),)*16)
    original = derive_forced_support_balance(snapshot, epi_weight=F(1), forcing=(F(0),)*16)
    return graph, prior, original


def test_shared_copy_restores_neighbor_order_and_detaches_aliases_and_histories():
    graph, prior, _ = _fixture()
    before = study.native._record(graph)
    (control, perturbed), evidence = study.prepare_branches(graph, prior)
    assert evidence["scientific_records_equal"]
    assert tuple(control.neighbors(7)) == tuple(perturbed.neighbors(7)) == (6, 0, "child_7")
    assert study.GRAPH_RUNTIME_CACHE_KEYS.isdisjoint(control.graph)
    assert control._last_operator_applied == graph._last_operator_applied
    for clone in (control, perturbed):
        assert clone.graph["history_alias"] is clone.nodes["child_0"]["epi_time_history"]
        assert clone.graph["history_alias"] is not graph.graph["history_alias"]
    control.nodes["child_0"]["nested"]["values"].append("fixture-only edit")
    control.graph["history_alias"].append((2.0, 0.5))
    control.graph["history"]["C_steps"].append(0.9)
    assert study.native._record(graph) == before
    assert study._scientific_record(perturbed) == evidence["scientific_source"]


@pytest.mark.parametrize("mutation", ("time", "lineage", "node_order", "pressure_hook", "selector", "integrator"))
def test_source_admission_refuses_incompatible_fork(mutation):
    graph, prior, _ = _fixture()
    if mutation == "time":
        graph.graph["_t"] = 2.25
    elif mutation == "lineage":
        graph.nodes["child_0"]["parent_node"] = 7
    elif mutation == "node_order":
        prior["endpoint"]["nodes"] = tuple(reversed(prior["endpoint"]["nodes"]))
    elif mutation == "pressure_hook":
        graph.graph["compute_delta_nfr"] = lambda graph: None
    elif mutation == "selector":
        graph.graph["glyph_selector"] = lambda *args: "IL"
    elif mutation == "integrator":
        graph.graph["integrator"] = lambda: None
    with pytest.raises(ValueError):
        study.prepare_branches(graph, prior)


def _portable_runtime(monkeypatch, *, refused=None, allowed=True, jump=0.0625, no_contrast=False):
    """Stub only execution seams; copy, records, event gates and pooling remain real."""
    graph, prior, original = _fixture()
    if no_contrast:
        for node in prior["lineage"]["children"]:
            graph.nodes[node]["EPI"] = 0.25
        prior["endpoint"] = study.full._state(graph)
    (control, perturbed), _ = study.prepare_branches(graph, prior)
    calls, analyses, event_calls, tetrads = [], [], [], []
    monkeypatch.setattr(study.full, "_reference", lambda value: original)
    monkeypatch.setattr(study.full, "_emission_admission", lambda g, targets: {
        "allowed": allowed, "targets": targets, "rows": (), "fixture": "not native admission evidence"})

    def event(g, targets):
        event_calls.append((g, targets))
        before = study.full._state(g)
        g.nodes[targets[0]]["EPI"] += jump  # Synthetic receipt fixture only.
        g.nodes[targets[0]]["glyph_history"].append("AL")
        return {"before": before, "after": study.full._state(g), "fixture": "synthetic localized write"}

    def trace(g, *, capture_generation):
        assert capture_generation is True
        ordinal = sum(item[0] is g for item in calls)
        calls.append((g, ordinal, g.graph["_t"]))
        branch = 0 if g is control else 1
        if refused == (branch, ordinal):
            return {"status": "refused", "boundaries": (), "captures": {}, "native_calls": 1,
                    "fixture": "known synthetic admission refusal"}
        g.graph["_t"] += 0.25  # Synthetic time-only record; no dynamics.
        g.graph["history"]["C_steps"].append(0.9)
        return {"status": "executed", "boundaries": (), "captures": {}, "native_calls": 1,
                "fixture": "synthetic time-only record"}

    def analyze(*args):
        analyses.append(args)
        return {"fixture": "accounting tested independently; no numerical assertion here"}

    monkeypatch.setattr(study.full, "_emission_event", event)
    monkeypatch.setattr(study.native, "_trace_step", trace)
    monkeypatch.setattr(study, "_analyze_response", analyze)
    monkeypatch.setattr(study.native, "_tetrad", lambda g: tetrads.append(g) or {"fixture": "detached readout seam"})
    return control, perturbed, prior, original, calls, analyses, event_calls, tetrads


def test_fixed_six_step_budget_localized_energy_and_lossless_packing(monkeypatch):
    control, perturbed, prior, original, calls, analyses, events, tetrads = _portable_runtime(monkeypatch)
    pool = study.window.RecordPool()
    result = study.run_branches(control, perturbed, prior, pool=pool)
    assert result["status"] == "completed" and result["attempted_native_calls"] == 12
    assert [len([x for x in calls if x[0] is g]) for g in (control, perturbed)] == [6, 6]
    assert all([row[2] for row in calls if row[0] is g] == [1.5, 1.75, 2.0, 2.25, 2.5, 2.75] for g in (control, perturbed))
    assert control.graph["_t"] == perturbed.graph["_t"] == 3
    assert len(events) == 1 and events[0] == (perturbed, ("child_0",))
    assert len(tetrads) == 4 and len(analyses) == 1
    a, b, model, children, x, y = analyses[0]
    assert len(a) == len(b) == 6 and model is original and children == prior["lineage"]["children"]
    assert sum(F(v) != F(w) for v, w in zip(x, y)) == 1
    gates = result["initial_gates"]
    hj = original.metric_weights[8]
    hb = sum(original.metric_weights[8:])
    assert gates["actual_error"] == hj*(1-hj/hb)*F(0.0625)**2/2 > 0
    for branch, raw in zip(result["branches"], (a, b), strict=True):
        assert branch["completed_steps"] == 6
        assert [study.window.expand_record(ref, pool.nodes) for ref in branch["steps"]] == study.full._payload(raw)


@pytest.mark.parametrize("options,status,event_count", (
    ({"allowed": False}, "intervention_refused", 0),
    ({"jump": 0.0}, "initial_gate_not_met", 1),
    ({"no_contrast": True}, "initial_gate_not_met", 1),
))
def test_initial_gates_stop_before_native_calls(monkeypatch, options, status, event_count):
    control, perturbed, prior, _, calls, analyses, events, tetrads = _portable_runtime(monkeypatch, **options)
    result = study.run_branches(control, perturbed, prior, pool=study.window.RecordPool())
    assert result["status"] == status and result["attempted_native_calls"] == 0
    assert result["decision"] == "inconclusive"
    assert calls == analyses == tetrads == [] and len(events) == event_count


def test_native_refusal_stops_paired_study_and_analyzes_common_prefix(monkeypatch):
    control, perturbed, prior, _, calls, analyses, _, _ = _portable_runtime(monkeypatch, refused=(1, 2))
    result = study.run_branches(control, perturbed, prior, pool=study.window.RecordPool())
    assert result["status"] == "native_refusal"
    assert result["attempted_native_calls"] == 6
    assert len([x for x in calls if x[0] is perturbed]) == 3
    assert result["branches"][1]["completed_steps"] == 2
    assert len(analyses[0][0]) == len(analyses[0][1]) == 2
    assert result["aligned_completed_steps"] == 2
    assert perturbed.graph["_t"] == 2.0


def test_unexpected_native_failure_propagates(monkeypatch):
    control, perturbed, prior, *_ = _portable_runtime(monkeypatch)

    def fail(*args, **kwargs):
        raise TypeError("unreviewed implementation error")

    monkeypatch.setattr(study.native, "_trace_step", fail)
    with pytest.raises(TypeError, match="unreviewed"):
        study.run_branches(control, perturbed, prior, pool=study.window.RecordPool())


def test_fixed_domain_analysis_refusal_retains_all_attempted_records(monkeypatch):
    control, perturbed, prior, *_ = _portable_runtime(monkeypatch)

    def unavailable(*args):
        raise ValueError("capacity changed outside fixed accounting domain")

    monkeypatch.setattr(study, "_analyze_response", unavailable)
    pool = study.window.RecordPool()
    result = study.run_branches(control, perturbed, prior, pool=pool)
    analysis = study.window.expand_record(result["analysis"], pool.nodes)
    assert result["attempted_native_calls"] == 12
    assert len(result["branches"][0]["steps"]) == len(result["branches"][1]["steps"]) == 6
    assert analysis["decision"]["outcome"] == "inconclusive"
    assert analysis["reason"] == "capacity changed outside fixed accounting domain"


def test_control_refusal_does_not_attempt_the_other_branch(monkeypatch):
    control, perturbed, prior, _, calls, analyses, _, _ = _portable_runtime(monkeypatch, refused=(0, 0))
    result = study.run_branches(control, perturbed, prior, pool=study.window.RecordPool())
    assert result["attempted_native_calls"] == len(calls) == 1
    assert result["aligned_completed_steps"] == 0
    assert analyses[0][0] == analyses[0][1] == []


def test_original_metric_domain_is_checked_before_intervention(monkeypatch):
    control, perturbed, prior, _, calls, _, events, _ = _portable_runtime(monkeypatch)
    for graph in (control, perturbed):
        graph.nodes[0]["nu_f"] = 0.0
    with pytest.raises(ValueError, match="fixed-metric domain"):
        study.run_branches(control, perturbed, prior, pool=study.window.RecordPool())
    assert calls == events == []


def test_cross_branch_mutation_is_not_silently_admitted(monkeypatch):
    control, perturbed, prior, *_ = _portable_runtime(monkeypatch)

    def bad(g, *, capture_generation):
        perturbed.nodes["child_7"]["stable_count"] += 1
        return {"status": "refused", "boundaries": (), "captures": {}}

    monkeypatch.setattr(study.native, "_trace_step", bad)
    with pytest.raises(RuntimeError, match="other branch"):
        study.run_branches(control, perturbed, prior, pool=study.window.RecordPool())


def test_one_source_replay_and_complete_report_admission_precede_fork(monkeypatch, tmp_path):
    graph, prior, _ = _fixture()
    prior["fixture_numpy_scalar"] = np.float64(0.125)
    prior["fixture_glyph_scalar"] = Glyph.IL
    path = tmp_path/"source.json"
    path.write_text(json.dumps(study.full._payload({"branches": (prior,)})), encoding="utf8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    chronology = []
    monkeypatch.setattr(study.native, "load_prior_evidence", lambda *a, **k: (
        json.loads(path.read_text()), {"path": str(path), "sha256": digest}))
    monkeypatch.setattr(study.full, "replay_response_branch", lambda branch: chronology.append(("replay", branch)) or (graph, prior))
    real_admit, real_prepare = study.native._admit_replay, study.prepare_branches

    def admit(*args):
        chronology.append(("admit",))
        return real_admit(*args)

    def prepare(*args):
        chronology.append(("fork",))
        branches, fork = real_prepare(*args)
        fork["fixture_numpy_scalar"] = np.float64(-0.0)
        fork["fixture_glyph_scalar"] = Glyph.IL
        return branches, fork

    real_pool = study.window.RecordPool

    class RecordingPool(real_pool):
        def pack(self, value):
            chronology.append(("pack", type(value["fixture_numpy_scalar"])))
            assert type(value["fixture_glyph_scalar"]) is str
            return super().pack(value)

    monkeypatch.setattr(study.native, "_admit_replay", admit)
    monkeypatch.setattr(study, "prepare_branches", prepare)
    monkeypatch.setattr(study.window, "RecordPool", RecordingPool)
    monkeypatch.setattr(study, "run_branches", lambda *a, **k: chronology.append(("continue",)) or {
        "status": "fixture", "attempted_native_calls": 0})
    result = study.run_study(path, expected_sha256=digest)
    assert chronology == [("replay", "control"), ("admit",), ("fork",),
                          ("pack", float), ("pack", float), ("continue",)]
    assert result["source_replay_calls"] == 1
    assert result["source_replay_admission"]["full_payload_equal"]
    packed = study.window.expand_record(result["replayed_control_report"], result["record_pool"])
    assert packed == json.loads(json.dumps(study.full._payload(prior), allow_nan=False))
    assert type(packed["fixture_numpy_scalar"]) is float
    assert packed["fixture_glyph_scalar"] == "IL" and type(packed["fixture_glyph_scalar"]) is str
    fork = study.window.expand_record(result["fork"], result["record_pool"])
    assert fork["fixture_numpy_scalar"].hex() == "-0x0.0p+0"
    # A changed scientific report is rejected before either fork or continuation.
    prior["endpoint"]["epi"] = tuple(0.99 for _ in prior["endpoint"]["epi"])
    chronology.clear()
    with pytest.raises(ValueError, match="scientific replay"):
        study.run_study(path, expected_sha256=digest)
    assert chronology == [("replay", "control"), ("admit",)]


@pytest.mark.parametrize("failure_at", (1, 2))
def test_source_pack_failure_prevents_all_new_branch_execution(monkeypatch, tmp_path, failure_at):
    graph, prior, _ = _fixture()
    path = tmp_path/"source.json"
    path.write_text(json.dumps(study.full._payload({"branches": (prior,)})), encoding="utf8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(study.native, "load_prior_evidence", lambda *a, **k: (
        json.loads(path.read_text()), {"path": str(path), "sha256": digest}))
    monkeypatch.setattr(study.full, "replay_response_branch", lambda branch: (graph, prior))

    class FailingPool(study.window.RecordPool):
        calls = 0

        def pack(self, value):
            self.calls += 1
            if self.calls == failure_at:
                raise TypeError("synthetic source serialization failure")
            return super().pack(value)

    def forbidden(*args, **kwargs):
        raise AssertionError("branch execution must follow both successful source packs")

    monkeypatch.setattr(study.window, "RecordPool", FailingPool)
    monkeypatch.setattr(study, "run_branches", forbidden)
    with pytest.raises(TypeError, match="source serialization"):
        study.run_study(path, expected_sha256=digest)


def test_cli_never_overwrites_source(monkeypatch, tmp_path):
    path = tmp_path/"source.json"
    path.write_text("retained bytes")
    monkeypatch.setattr(sys, "argv", ["study", "--input", str(path), "--output", str(path)])
    with pytest.raises(ValueError, match="overwrite"):
        study.main()
    assert path.read_text() == "retained bytes"


def _cli_fixture(monkeypatch, tmp_path):
    source, output = tmp_path/"source.json", tmp_path/"result.json"
    source.write_text("retained synthetic source", encoding="utf8")
    calls = []
    monkeypatch.setattr(sys, "argv", ["study", "--input", str(source), "--output", str(output)])
    monkeypatch.setattr(study, "current_git_source_provenance", lambda *args: ("a"*40, False, None))

    def synthetic_run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"protocol": study.PROTOCOL, "status": "synthetic_complete", "attempted_native_calls": 0,
                "record_pool": {}, "exact_fixture_value": F(1, 7), "fixture_glyph": Glyph.IL,
                "fixture_numpy": np.float64(0.125)}

    monkeypatch.setattr(study, "run_study", synthetic_run)
    return source, output, calls


def test_complete_cli_prevalidates_real_manifest_and_saves_complete_result(monkeypatch, tmp_path):
    source, output, calls = _cli_fixture(monkeypatch, tmp_path)
    study.main()
    checkpoint = output.with_suffix(".completed.json")
    assert len(calls) == 1 and checkpoint.read_bytes() == output.read_bytes()
    report = json.loads(output.read_bytes())
    manifest = study.CoreExperimentManifest(**report["manifest"])
    manifest.validate_for_admission()
    assert manifest.timestep == 0.25 and manifest.seed == 17
    assert report["exact_fixture_value"] == "1/7"
    assert report["fixture_glyph"] == "IL" and report["fixture_numpy"] == 0.125
    assert report["completed_checkpoint"] == str(checkpoint)
    assert source.read_text() == "retained synthetic source"


def test_cli_metadata_failure_runs_no_study(monkeypatch, tmp_path):
    _, output, calls = _cli_fixture(monkeypatch, tmp_path)
    # Exercise the real manifest validator, not a mocked constructor.
    monkeypatch.setattr(study, "current_git_source_provenance", lambda *args: ("not-a-git-sha", False, None))
    with pytest.raises(ValueError):
        study.main()
    assert calls == [] and not output.exists() and not output.with_suffix(".completed.json").exists()


def test_completed_checkpoint_survives_final_provenance_failure(monkeypatch, tmp_path):
    _, output, calls = _cli_fixture(monkeypatch, tmp_path)
    provenance_calls = []

    def provenance(*args):
        provenance_calls.append(1)
        if len(provenance_calls) == 2:
            checkpoint = output.with_suffix(".completed.json")
            assert checkpoint.is_file()
            assert json.loads(checkpoint.read_bytes())["status"] == "synthetic_complete"
            return "b"*40, False, None
        return "a"*40, False, None

    monkeypatch.setattr(study, "current_git_source_provenance", provenance)
    with pytest.raises(RuntimeError, match="completed result retained"):
        study.main()
    assert len(calls) == 1 and not output.exists()
    assert output.with_suffix(".completed.json").is_file()


def test_cli_checkpoint_cannot_overwrite_input(monkeypatch, tmp_path):
    _, output, calls = _cli_fixture(monkeypatch, tmp_path)
    source = output.with_suffix(".completed.json")
    source.write_text("retained bytes")
    monkeypatch.setattr(sys, "argv", ["study", "--input", str(source), "--output", str(output)])
    with pytest.raises(ValueError, match="overwrite"):
        study.main()
    assert calls == [] and source.read_text() == "retained bytes"
