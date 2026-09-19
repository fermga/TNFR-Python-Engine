"""Bounded portable version comparison; never read historical research files."""

import json
import math
import sys
from copy import deepcopy
from fractions import Fraction as F

import pytest

from benchmarks import thol_exact_phase_source_comparison as audit
from benchmarks import thol_phase_source_relevance as source_owner
from benchmarks import thol_regional_identity_audit as identity_owner
from benchmarks import thol_retained_phase_audit as phase_owner
from benchmarks.thol_preparation_policy import _literal
from benchmarks.thol_pressure_feedback import _payload
from tests.physics import test_thol_phase_source_relevance as fixtures
from tnfr.alias import get_theta_attr

prior = fixtures.prior
native_fixture = fixtures.native_fixture
captured_native = fixtures.captured_native
evidence = fixtures.evidence


@pytest.fixture
def study_input(evidence):
    native, phase, _, _, digest = deepcopy(evidence)
    branch = native["branches"][0]
    boundary = next(
        r
        for r in branch["native_trace"]["boundaries"]
        if r["boundary"] == "coordinate_global_local_phase"
    )
    nodes = branch["endpoint"]["state"]["nodes"]
    gains = (0.125, 0.0625)
    # One synthetic fixture reference, not a historical producer or native step.
    boundary["before"]["state"]["phase"] = [
        float(k % 8) * math.pi / 4 for k in range(16)
    ]
    graph = phase_owner._detached_graph(
        boundary["before"], dict(zip(nodes, nodes, strict=True))
    )
    audit.coordinate_global_local_phase(
        graph, *gains, n_jobs=1, global_reduction="legacy"
    )
    phases = [float(get_theta_attr(graph.nodes[n])) for n in nodes]
    boundary["after"]["state"]["phase"] = phases.copy()
    for row in branch["native_trace"]["boundaries"]:
        if row["ordinal"] > boundary["ordinal"]:
            row["before"]["state"]["phase"] = phases.copy()
            row["after"]["state"]["phase"] = phases.copy()
    branch["endpoint"]["state"]["phase"] = phases.copy()
    for record in (boundary["before"], boundary["after"], branch["endpoint"]):
        record["graph_attributes"].update(
            PHASE_K_GLOBAL=_literal(gains[0]), PHASE_K_LOCAL=_literal(gains[1])
        )
    observation, branch["endpoint_capture"] = fixtures._actual_capture(
        branch["endpoint"]
    )
    identity = _payload(
        identity_owner.audit_branch(native["replayed_prior_reports"][0], branch)
    )
    identity["historical_input"] = {"sha256": digest}
    old = phase["audit"]
    old["source_boundary"], old["effective_gains"] = deepcopy(boundary), list(gains)
    mapping = dict(old["node_mapping"])
    orders = (
        ("base_numpy", dict(zip(nodes, nodes, strict=True)), nodes),
        ("transported_numpy", mapping, None),
        ("canonical_order_numpy", mapping, nodes),
    )
    for name, relabel, order in orders:
        old[name]["initial"] = phase_owner._projection(
            phase_owner._detached_graph(boundary["before"], relabel, node_order=order)
        )
    old["base_numpy"]["phase"] = phases.copy()
    old["transported_numpy"]["phase"] = phases.copy()
    return native, _payload(phase), identity, observation, digest


def _inputs(evidence):
    native, phase, identity, _, digest = evidence
    admitted = source_owner.admit_evidence(
        native, phase, identity, native_sha256=digest
    )
    return (
        admitted,
        phase["audit"]["source_boundary"],
        phase["audit"]["effective_gains"],
    )


def test_three_coordinations_two_captures_fixed_inputs_and_nine_exact_rates(
    study_input, monkeypatch
):
    from tnfr.dynamics import runtime

    monkeypatch.setattr(runtime, "step", lambda *a, **k: pytest.fail("no native call"))
    admitted, boundary, gains = _inputs(study_input)
    saved = deepcopy(study_input)
    calls, captures = [], []
    original, capture = (
        audit.coordinate_global_local_phase,
        source_owner.capture_non_epi_forcing,
    )

    def coordinate(graph, *args, **kwargs):
        calls.append((kwargs["global_reduction"], tuple(graph)))
        return original(graph, *args, **kwargs)

    def observe(graph):
        result = capture(graph)
        captures.append(result)
        return result

    monkeypatch.setattr(audit, "coordinate_global_local_phase", coordinate)
    monkeypatch.setattr(source_owner, "capture_non_epi_forcing", observe)
    result = audit.compare_versions(admitted, boundary, gains)
    nodes = admitted["archived_observation"].snapshot.nodes
    assert calls == [
        ("legacy", nodes),
        (audit.VERSION, nodes),
        (audit.VERSION, tuple(reversed(nodes))),
    ]
    assert len(captures) == result["forcing_capture_calls"] == 2
    assert result["coordination_calls"] == 3 and result["native_calls"] == 0
    assert captures[0].snapshot == captures[1].snapshot
    assert captures[0] == admitted["archived_observation"]
    assert result["enumeration_control"]["aligned_raw_proposals_equal"]
    assert result["enumeration_control"]["aligned_realized_phases_equal"]
    assert (
        result["comparison_context"]["archived_alternative_phase"]
        == admitted["phases"]["alternative"]
    )
    assert (
        tuple(map(F, result["coordinator_cases"][1]["aligned_output"]))
        == captures[1].phase
    )
    assert captures[1].phase != admitted["phases"]["alternative"]
    assert study_input == saved
    first, second = captures
    snap = first.snapshot
    degree = tuple(
        sum((w for i, _, w in snap.conductance if i == k), F(0))
        for k in range(len(nodes))
    )
    h = tuple(d / nu for d, nu in zip(degree, snap.capacity, strict=True))
    delta = tuple(b - a for a, b in zip(first.forcing, second.forcing, strict=True))
    for row in result["source_comparison"]["regions"]:
        ids = tuple(nodes.index(n) for n in row["region"])
        mean = sum((h[i] * snap.epi[i] for i in ids), F(0)) / sum(h[i] for i in ids)
        assert row["model_weighted_total_rate_difference"] == sum(
            (degree[i] * delta[i] for i in ids), F(0)
        )
        assert row["model_variance_rate_difference"] == sum(
            (degree[i] * (snap.epi[i] - mean) * delta[i] for i in ids), F(0)
        )
        assert row["internal_boundary_stored_rates_unchanged"]


@pytest.mark.parametrize("change", ("boundary", "gains"))
def test_boundary_and_gain_binding_fail_before_calls(study_input, monkeypatch, change):
    admitted, boundary, gains = _inputs(study_input)
    if change == "boundary":
        boundary["before"]["state"]["phase"][0] += 0.25
    else:
        gains[0] += 0.25
    monkeypatch.setattr(
        audit,
        "coordinate_global_local_phase",
        lambda *a, **k: pytest.fail("admission before call"),
    )
    with pytest.raises(ValueError):
        audit.compare_versions(admitted, boundary, gains)


def test_legacy_mismatch_stops_after_one_call_without_pressure(
    study_input, monkeypatch
):
    admitted, boundary, gains = _inputs(study_input)
    admitted["phases"]["baseline"] = tuple(F(9) for _ in admitted["phases"]["baseline"])
    monkeypatch.setattr(
        source_owner,
        "evaluate_admitted",
        lambda *a: pytest.fail("no captures after mismatch"),
    )
    with pytest.raises(audit.ComparisonAdmissionError, match="legacy") as failure:
        audit.compare_versions(admitted, boundary, gains)
    assert failure.value.report["coordination_calls"] == 1
    assert failure.value.report["forcing_capture_calls"] == 0


@pytest.mark.parametrize("field", ("components", "local_targets"))
def test_enumeration_input_obstruction_stops_before_pressure(
    study_input, monkeypatch, field
):
    admitted, boundary, gains = _inputs(study_input)
    original = audit._run_case
    count = 0

    def changed(*args):
        nonlocal count
        result = original(*args)
        count += 1
        if count == 3:
            values = list(result[field])
            values[0] = (0.0, 0.0) if field == "components" else 9.0
            result[field] = tuple(values)
        return result

    monkeypatch.setattr(audit, "_run_case", changed)
    monkeypatch.setattr(
        source_owner,
        "evaluate_admitted",
        lambda *a: pytest.fail("no captures after obstruction"),
    )
    with pytest.raises(audit.ComparisonAdmissionError, match=field):
        audit.compare_versions(admitted, boundary, gains)


def test_portable_cli_preserves_inputs_and_serializes_evidence(
    study_input, tmp_path, monkeypatch
):
    paths, digests = fixtures._write_reports(tmp_path, study_input)
    original = {name: path.read_bytes() for name, path in paths.items()}
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        audit, "current_git_source_provenance", lambda *a: ("a" * 40, False, None)
    )
    monkeypatch.setattr(sys, "argv", fixtures._arguments(paths, digests, output))
    audit.main()
    report = json.loads(output.read_bytes())
    assert report["status"] == "completed" and report["forcing_capture_calls"] == 2
    assert report["coordinator_cases"][1]["evidence"]["version"] == audit.VERSION
    assert all(path.read_bytes() == original[name] for name, path in paths.items())


@pytest.mark.parametrize("which", ("native", "phase", "identity"))
def test_cli_refuses_all_input_overwrites(study_input, tmp_path, monkeypatch, which):
    paths, digests = fixtures._write_reports(tmp_path, study_input)
    monkeypatch.setattr(sys, "argv", fixtures._arguments(paths, digests, paths[which]))
    monkeypatch.setattr(
        audit,
        "run_study",
        lambda *a, **k: pytest.fail("no work before overwrite rejection"),
    )
    with pytest.raises(ValueError, match="overwrite"):
        audit.main()
