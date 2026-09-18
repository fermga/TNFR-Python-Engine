"""Portable phase-source relevance evidence, without coordination or native steps."""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as F
import hashlib
import json
import sys

import pytest

from benchmarks import thol_regional_identity_audit as identity_owner
from benchmarks import thol_retained_phase_audit as phase_owner
from benchmarks import thol_phase_source_relevance as audit
from benchmarks.thol_pressure_feedback import _payload
from benchmarks.thol_retained_reset_audit import _literal_number
from tests.physics.test_thol_regional_identity_audit import (
    native_fixture as _native_fixture, prior as _prior_fixture,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing, decompose_non_epi_forcing
from tnfr.research.claims import ClaimStatus
from tnfr.research.core_manifests import CoreExperimentManifest


prior = _prior_fixture
native_fixture = _native_fixture


def _actual_capture(record):
    """Real synthetic capture on fixed support, never a simulated transition."""
    nodes = record["state"]["nodes"]
    graph = phase_owner._detached_graph(record, {node: node for node in nodes})
    graph.graph["_dnfr_weights"] = {
        key: _literal_number(value) for key, value in record["graph_attributes"]["_dnfr_weights"].items()}
    observation = capture_non_epi_forcing(graph)
    return observation, _payload({"available": True, "payload": {
        "observation": asdict(observation), "components": decompose_non_epi_forcing(observation)}})


@pytest.fixture
def captured_native(native_fixture):
    """Replace the synthetic endpoint's declaration with an actual readout."""
    prior, branch = deepcopy(native_fixture)
    observation, capture = _actual_capture(branch["endpoint"])
    branch["endpoint_capture"] = capture
    identity = identity_owner.audit_branch(prior, branch)
    return prior, branch, observation, _payload(identity)


@pytest.fixture
def evidence(captured_native):
    prior, branch, observation, identity = deepcopy(captured_native)
    nodes = branch["endpoint"]["state"]["nodes"]
    pairs = prior["lineage"]["parent_children"]
    mapping = {node: pairs[(i+1) % 8][j] for i, pair in enumerate(pairs) for j, node in enumerate(pair)}
    boundary = next(row for row in branch["native_trace"]["boundaries"]
                    if row["boundary"] == "coordinate_global_local_phase")
    source = boundary["before"]
    initial = phase_owner._projection(phase_owner._detached_graph(source, {node: node for node in nodes}))
    moved = phase_owner._projection(phase_owner._detached_graph(source, mapping))
    canonical = phase_owner._projection(phase_owner._detached_graph(source, mapping, node_order=nodes))
    baseline = branch["endpoint"]["state"]["phase"].copy()
    alternative = baseline.copy()
    alternative[0], alternative[8] = .5, -.25
    native_digest = "a"*64
    phase = {"selected_branch": "control", "input_evidence": {"sha256": native_digest}, "audit": {
        "source_boundary": deepcopy(boundary), "rotation": 1, "node_mapping": list(mapping.items()),
        "transported_source_equal": True, "numpy_replay_matches_archive": True,
        "base_numpy": {"initial": initial, "phase": baseline},
        "transported_numpy": {"initial": moved, "phase": baseline.copy()},
        "canonical_order_numpy": {"initial": canonical, "phase": alternative},
    }}
    identity["historical_input"] = {"sha256": native_digest}
    native = {"branches": [branch, {"branch": "child_emission"}],
              "replayed_prior_reports": [prior, {"branch": "child_emission"}]}
    return native, _payload(phase), identity, observation, native_digest


def _forbid_evolution(monkeypatch):
    from benchmarks import thol_native_runtime_response
    from tnfr.dynamics import coordination, runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("phase-source tests must not execute native or coordination dynamics")

    monkeypatch.setattr(runtime, "step", forbidden)
    monkeypatch.setattr(coordination, "coordinate_global_local_phase", forbidden)
    monkeypatch.setattr(thol_native_runtime_response, "run_study", forbidden)
    monkeypatch.setattr(phase_owner, "run_study", forbidden)


def _admit(evidence):
    native, phase, identity, _, digest = evidence
    return audit.admit_evidence(native, phase, identity, native_sha256=digest)


def test_two_captures_independent_nine_region_differences_and_fixed_terms(evidence, monkeypatch):
    _forbid_evolution(monkeypatch)
    saved = deepcopy(evidence)
    admitted = _admit(evidence)
    seen = []
    original = audit.capture_non_epi_forcing

    def capture(graph):
        item = original(graph)
        seen.append(item)
        return item

    monkeypatch.setattr(audit, "capture_non_epi_forcing", capture)
    result = audit.evaluate_admitted(admitted)
    assert len(seen) == result["forcing_capture_calls"] == 2
    assert seen[0] == evidence[3]
    assert seen[0].phase == admitted["phases"]["baseline"]
    assert seen[1].phase == admitted["phases"]["alternative"]
    assert seen[0].snapshot == seen[1].snapshot
    assert result["transported_equal_input_control"]
    assert evidence == saved
    base, alternative = seen
    snapshot = base.snapshot
    degree = tuple(sum((weight for i, _, weight in snapshot.conductance if i == k), F(0))
                   for k in range(len(snapshot.nodes)))
    metric = tuple(d/v for d, v in zip(degree, snapshot.capacity, strict=True))
    delta = tuple(a-b for a, b in zip(alternative.forcing, base.forcing, strict=True))
    kernel_delta = tuple(a-b for a, b in zip(alternative.full_kernel_pressure, base.full_kernel_pressure, strict=True))
    defect_delta = tuple(a-b for a, b in zip(alternative.kernel_pressure_defect, base.kernel_pressure_defect, strict=True))
    stored_delta = tuple(a-b for a, b in zip(alternative.stored_pressure_residual, base.stored_pressure_residual, strict=True))
    assert any(delta) and not all(delta)
    assert result["phase_component_difference"] == delta
    assert result["kernel_pressure_defect_difference"] == defect_delta
    assert result["stored_pressure_residual_difference"] == stored_delta
    assert kernel_delta == tuple(d+r for d, r in zip(delta, defect_delta, strict=True))
    assert stored_delta == tuple(-value for value in kernel_delta)
    assert len(result["regions"]) == 9 and result["regional_observations"] == 18
    for item, (label, region) in zip(result["regions"], admitted["regions"], strict=True):
        assert item["label"] == label and item["region"] == region
        ids = tuple(snapshot.nodes.index(node) for node in region)
        mean = sum((metric[i]*snapshot.epi[i] for i in ids), F(0))/sum(metric[i] for i in ids)
        mass = sum((degree[i]*delta[i] for i in ids), F(0))
        variance = sum((degree[i]*(snapshot.epi[i]-mean)*delta[i] for i in ids), F(0))
        assert item["model_weighted_total_rate_difference"] == mass
        assert item["model_variance_rate_difference"] == variance
        first, second = item["baseline"], item["alternative"]
        assert second["model_mass_rate"]-first["model_mass_rate"] == mass
        assert second["model_variance_rate"]-first["model_variance_rate"] == variance
        for name in ("mass_boundary_rate", "variance_boundary_rate", "internal_dissipation",
                     "stored_mass_rate", "stored_variance_rate", "mean", "variance"):
            assert first[name] == second[name]
        assert item["weighted_total_difference_identity_residual"] == item["variance_difference_identity_residual"] == 0
        assert item["internal_boundary_stored_rates_unchanged"]
    assert result["coordination_calls"] == result["native_calls"] == 0


def test_equal_alternative_keeps_all_nine_zero_results_without_third_capture(evidence, monkeypatch):
    _forbid_evolution(monkeypatch)
    phase = evidence[1]["audit"]
    phase["canonical_order_numpy"]["phase"] = phase["base_numpy"]["phase"].copy()
    seen = []
    original = audit.capture_non_epi_forcing

    def capture(graph):
        seen.append(tuple(graph))
        return original(graph)

    monkeypatch.setattr(audit, "capture_non_epi_forcing", capture)
    result = audit.evaluate_admitted(_admit(evidence))
    assert len(seen) == 2 and seen[0] == seen[1]
    assert not any(result["phase_component_difference"])
    assert not any(result["kernel_pressure_defect_difference"])
    assert len(result["regions"]) == 9
    assert all(row["model_weighted_total_rate_difference"] == row["model_variance_rate_difference"] == 0
               for row in result["regions"])


def test_baseline_mismatch_stops_before_alternative_capture(evidence, monkeypatch):
    admitted = _admit(evidence)
    calls = []
    original = audit.capture_non_epi_forcing

    def altered(graph):
        calls.append(tuple(graph))
        item = original(graph)
        return replace(item, full_kernel_pressure=(F(99),)*16)

    monkeypatch.setattr(audit, "capture_non_epi_forcing", altered)
    with pytest.raises(ValueError, match="baseline"):
        audit.evaluate_admitted(admitted)
    assert len(calls) == 1


@pytest.mark.parametrize("change", (
    "native_digest", "identity_digest", "branch_binding", "prior_binding", "source_boundary",
    "baseline_phase", "transported_phase", "mapping", "rotation", "canonical_order", "neighbor_order",
    "edge_weight", "endpoint_time", "callback", "backend", "weight_keys", "weight_value",
    "capture_cache", "capture_phase", "region_order", "region_membership", "region_snapshot",
    "alternative_length", "alternative_nonfinite",
))
def test_provenance_alignment_fixed_state_and_capture_admission_rejects(evidence, change):
    native, phase, identity, _, _ = evidence
    branch = native["branches"][0]
    phase_data = phase["audit"]
    endpoint = branch["endpoint"]
    if change == "native_digest":
        phase["input_evidence"]["sha256"] = "0"*64
    elif change == "identity_digest":
        identity["historical_input"]["sha256"] = "0"*64
    elif change == "branch_binding":
        identity["retained_branch"]["sha256"] = "0"*64
    elif change == "prior_binding":
        identity["retained_prior"]["sha256"] = "0"*64
    elif change == "source_boundary":
        phase_data["source_boundary"]["before"]["state"]["phase"][0] = .75
    elif change == "baseline_phase":
        phase_data["base_numpy"]["phase"][0] = .75
    elif change == "transported_phase":
        phase_data["transported_numpy"]["phase"][0] = .75
    elif change == "mapping":
        phase_data["node_mapping"][0][1] = phase_data["node_mapping"][1][1]
    elif change == "rotation":
        phase_data["rotation"] = True
    elif change == "canonical_order":
        phase_data["canonical_order_numpy"]["initial"]["nodes"].reverse()
    elif change == "neighbor_order":
        phase_data["transported_numpy"]["initial"]["ordered_neighbors"]["p1"].reverse()
    elif change == "edge_weight":
        phase_data["canonical_order_numpy"]["initial"]["edges"][0][2]["weight"] = .75
    elif change == "endpoint_time":
        endpoint["state"]["time"] = 2.0
    elif change == "callback":
        endpoint["graph_attributes"]["compute_delta_nfr"] = {"callable_module": "custom", "callable_name": "pressure"}
    elif change == "backend":
        endpoint["graph_attributes"]["vectorized_dnfr"] = False
    elif change == "weight_keys":
        del endpoint["graph_attributes"]["_dnfr_weights"]["phase"]
    elif change == "weight_value":
        endpoint["graph_attributes"]["_dnfr_weights"]["phase"] = .25
    elif change == "capture_cache":
        branch["endpoint_capture"]["payload"]["observation"]["snapshot"]["epi_gradient"][0] = "999"
    elif change == "capture_phase":
        branch["endpoint_capture"]["payload"]["observation"]["phase"][0] = "999"
    elif change == "region_order":
        identity["regions"][0], identity["regions"][1] = identity["regions"][1], identity["regions"][0]
    elif change == "region_membership":
        identity["regions"][0]["identity"]["endpoint"]["nodes"].reverse()
    elif change == "region_snapshot":
        identity["regions"][0]["finite_budget"]["after"]["capacity"][0] = "999"
    elif change == "alternative_length":
        phase_data["canonical_order_numpy"]["phase"].pop()
    elif change == "alternative_nonfinite":
        phase_data["canonical_order_numpy"]["phase"][0] = float("nan")
    else:
        raise AssertionError("unhandled control")
    with pytest.raises((ValueError, TypeError)):
        _admit(evidence)


@pytest.mark.parametrize("change", (
    "fresh_pressure", "kernel_defect", "stored_residual", "gradient_cache", "stored_pressure",
    "capacity", "phase", "weights", "forcing",
))
def test_alternative_captured_coefficients_and_residuals_are_validated(evidence, change):
    admitted = _admit(evidence)
    baseline = evidence[3]
    graph = audit._detached_graph(admitted["record"], admitted["phases"]["alternative"], admitted["config"])
    alternative = capture_non_epi_forcing(graph)
    if change == "fresh_pressure":
        alternative = replace(alternative, full_kernel_pressure=(F(999),)*16)
    elif change == "kernel_defect":
        alternative = replace(alternative, kernel_pressure_defect=(F(999),)*16)
    elif change == "stored_residual":
        alternative = replace(alternative, stored_pressure_residual=(F(999),)*16)
    elif change == "gradient_cache":
        alternative = replace(alternative, snapshot=replace(alternative.snapshot, epi_gradient=(F(999),)*16))
    elif change == "stored_pressure":
        alternative = replace(alternative, snapshot=replace(alternative.snapshot, stored_pressure=(F(999),)*16))
    elif change == "capacity":
        alternative = replace(alternative, snapshot=replace(alternative.snapshot, capacity=(F(999),)*16))
    elif change == "phase":
        alternative = replace(alternative, phase=(F(999),)*16)
    elif change == "weights":
        alternative = replace(alternative, normalized_weights=tuple((key, F(1, 4)) for key, _ in alternative.normalized_weights))
    elif change == "forcing":
        alternative = replace(alternative, forcing=(F(999),)*16)
    else:
        raise AssertionError("unhandled control")
    with pytest.raises((ValueError, TypeError)):
        audit.audit_captures(admitted, baseline, alternative)


def _write_reports(directory, evidence):
    native, phase, identity, _, _ = deepcopy(evidence)
    reports = {"native": native, "phase": phase, "identity": identity}
    claims = {"native": audit.INPUT_CLAIM, "phase": "O3.a-retained-phase-conditioning",
              "identity": "O3.a-retained-regional-identity"}
    paths, digests = {}, {}
    for name, report in reports.items():
        path = directory/f"{name}.json"
        manifest = CoreExperimentManifest(
            claim_id=claims[name], git_sha="a"*40, source_dirty=False, versions={"python": "synthetic-test"},
            graph_construction="Declared synthetic receipt for portable input tests; no executed trajectory",
            capacity_specification="Supplied positive capacities", solver="Detached readout only",
            timestep=None, seed=None, result_status=ClaimStatus.DERIVED, operator_sequence=(),
            telemetry=("fixture",), controls=("no native or coordination execution",), artifacts=(str(path),))
        manifest.validate_for_admission()
        report["manifest"] = manifest.to_dict()
        if name == "phase":
            report["input_evidence"]["sha256"] = digests["native"]
        elif name == "identity":
            report["historical_input"]["sha256"] = digests["native"]
        raw = json.dumps(report, allow_nan=False).encode()
        path.write_bytes(raw)
        paths[name], digests[name] = path, hashlib.sha256(raw).hexdigest()
    return paths, digests


def _arguments(paths, digests, output):
    return ["audit", "--input", str(paths["native"]), "--phase-input", str(paths["phase"]),
            "--identity-input", str(paths["identity"]), "--expected-sha256", digests["native"],
            "--expected-phase-sha256", digests["phase"], "--expected-identity-sha256", digests["identity"],
            "--output", str(output)]


def test_portable_cli_bindings_json_and_no_mutation(evidence, tmp_path, monkeypatch):
    _forbid_evolution(monkeypatch)
    paths, digests = _write_reports(tmp_path, evidence)
    saved = {name: path.read_bytes() for name, path in paths.items()}
    output = tmp_path/"result.json"
    monkeypatch.setattr(audit, "current_git_source_provenance", lambda *args: ("a"*40, False, None))
    monkeypatch.setattr(sys, "argv", _arguments(paths, digests, output))
    audit.main()
    result = json.loads(output.read_bytes())
    CoreExperimentManifest(**result["manifest"]).validate_for_admission()
    assert result["forcing_capture_calls"] == 2
    assert result["coordination_calls"] == result["native_calls"] == 0
    assert result["inputs"]["native"]["sha256"] == digests["native"]
    assert result["inputs"]["phase"]["sha256"] == digests["phase"]
    assert result["inputs"]["regional_identity"]["sha256"] == digests["identity"]
    for name, path in paths.items():
        assert path.read_bytes() == saved[name]
    # The referenced source boundary is a real JSON path into the pinned input.
    native = json.loads(saved["native"])
    rows = native["branches"][0]["native_trace"]["boundaries"]
    index = next(i for i, row in enumerate(rows) if row["boundary"] == "coordinate_global_local_phase")
    ref = result["source_boundary"]
    assert ref["json_path"] == f"branches[0].native_trace.boundaries[{index}]"
    canonical = json.dumps(rows[index], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert ref["sha256"] == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("which", ("native", "phase", "identity"))
def test_all_three_input_digests_fail_before_any_capture(evidence, tmp_path, monkeypatch, which):
    paths, digests = _write_reports(tmp_path, evidence)
    paths[which].write_bytes(paths[which].read_bytes()+b" ")

    def forbidden(*args, **kwargs):
        raise AssertionError("digest admission must precede any capture")

    monkeypatch.setattr(audit, "capture_non_epi_forcing", forbidden)
    with pytest.raises(ValueError, match="digest"):
        audit.run_study(paths["native"], paths["phase"], paths["identity"],
                        expected_native_sha256=digests["native"], expected_phase_sha256=digests["phase"],
                        expected_identity_sha256=digests["identity"])


@pytest.mark.parametrize("which", ("native", "phase", "identity"))
def test_cli_refuses_overwrite_of_every_input(evidence, tmp_path, monkeypatch, which):
    paths, digests = _write_reports(tmp_path, evidence)
    before = {name: path.read_bytes() for name, path in paths.items()}
    monkeypatch.setattr(sys, "argv", _arguments(paths, digests, paths[which]))

    def forbidden(*args, **kwargs):
        raise AssertionError("overwrite gate must precede source or scientific work")

    monkeypatch.setattr(audit, "current_git_source_provenance", forbidden)
    with pytest.raises(ValueError, match="overwrite"):
        audit.main()
    assert all(path.read_bytes() == before[name] for name, path in paths.items())
