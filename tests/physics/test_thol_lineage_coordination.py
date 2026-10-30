"""Portable causal lineage controls and exact fixed-target accounting."""

import hashlib
import json
import platform
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest

from benchmarks.thol_pressure_feedback import _payload
from tests.physics.test_thol_distributed_target import (
    _assert_reference,
    _assert_step,
    _assert_target,
    _dot,
    _energy,
    _laplacian,
)
from tnfr.research.claims import ClaimStatus
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)


@pytest.fixture(scope="module")
def control_fixture(tmp_path_factory):
    """Fresh real controls, never an unconditional ignored-artifact dependency."""
    from benchmarks.thol_distributed_target import run_study

    controls = run_study()
    root = Path(__file__).resolve().parents[2]
    path = tmp_path_factory.mktemp("lineage_controls") / "controls.json"
    scope = ("src/tnfr", "benchmarks/thol_distributed_target.py")
    sha, dirty, digest = current_git_source_provenance(root, scope)
    manifest = CoreExperimentManifest(
        claim_id="O1.b-distributed-fixed-relative-target-response",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={"python": platform.python_version()},
        graph_construction="Fresh actual distributed C8 birth preparation in this test invocation",
        capacity_specification="Unchanged canonical THOL and UM factors",
        solver="Existing two explicit refreshed Euler partitions",
        seed=17,
        timestep=0.25,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=("coherence", "dissonance", "self_organization", "coupling"),
        telemetry=("Complete common source and exact fixed-target flow budgets",),
        controls=("Independent no-event and all-node UM continuations",),
        artifacts=(str(path),),
    )
    manifest.validate_for_admission()
    payload = {"manifest": manifest.to_dict(), "source_scope": scope, **controls}
    encoded = (json.dumps(_payload(payload), allow_nan=False) + "\n").encode()
    path.write_bytes(encoded)
    return controls, path, hashlib.sha256(encoded).hexdigest()


@pytest.fixture(scope="module")
def study(control_fixture):
    import benchmarks.thol_distributed_target as target
    from benchmarks.thol_lineage_coordination import run_study

    _, path, digest = control_fixture
    prepare = target.prepare_distributed_transport_support
    graphs = []

    def fresh(*args, **kwargs):
        graph, prefix = prepare(*args, **kwargs)
        graphs.append(graph)
        return graph, prefix

    with patch.object(target, "prepare_distributed_transport_support", fresh):
        result = run_study(control_path=path, expected_control_sha256=digest)
    assert len(graphs) == 2 and graphs[0] is not graphs[1]
    return result


def _lineage_graph():
    """Labels deliberately do not encode role or parentage."""
    parents = ("child-looking-label", 700)
    children = ("unrelated", -19)
    graph = nx.Graph()
    graph.add_nodes_from((*parents, *children))
    graph.add_edges_from(
        ((parents[0], parents[1]), (parents[0], children[0]), (parents[1], children[1]))
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=1.0, nu_f=1.0, theta=0.0, glyph_history=["IL", "OZ", "THOL"]
        )
    for parent, child in zip(parents, children, strict=True):
        graph.nodes[parent]["sub_nodes"] = [child]
        graph.nodes[child]["parent_node"] = parent
    graph.graph.update(
        hierarchy={p: [c] for p, c in zip(parents, children, strict=True)},
        _node_sample=tuple(graph),
        _t=1.0,
    )
    prefix = {
        "birth": {
            "parent_children": tuple(zip(parents, children, strict=True)),
            "before": {"nodes": parents},
        }
    }
    return graph, prefix, parents, children


def test_lineage_partition_uses_actual_parentage_not_label_conventions():
    from benchmarks.thol_distributed_target import _lineage_targets

    graph, prefix, parents, children = _lineage_graph()
    before = deepcopy(
        (
            dict(graph.graph),
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
        )
    )
    for name, expected in (("original_parents", parents), ("born_children", children)):
        selected, audit = _lineage_targets(graph, prefix, name)
        assert selected == expected
        assert audit["parents"] == parents and audit["children"] == children
        assert audit["current_nodes"] == parents + children
        assert audit["disjoint_and_exhaustive"] and audit["live_parentage_verified"]
    assert before == (
        dict(graph.graph),
        tuple(graph.nodes(data=True)),
        tuple(graph.edges(data=True)),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_child",
        "overlap",
        "uncovered",
        "parent_pointer",
        "sub_nodes",
        "hierarchy",
    ),
)
def test_corrupted_receipt_or_live_lineage_cannot_authorize_a_cohort(mutation):
    from benchmarks.thol_distributed_target import _lineage_targets

    graph, prefix, parents, children = _lineage_graph()
    if mutation == "duplicate_child":
        prefix["birth"]["parent_children"] = (
            (parents[0], children[0]),
            (parents[1], children[0]),
        )
    elif mutation == "overlap":
        prefix["birth"]["parent_children"] = (
            (parents[0], parents[1]),
            (parents[1], children[1]),
        )
    elif mutation == "uncovered":
        graph.add_node("unaccounted")
    elif mutation == "parent_pointer":
        graph.nodes[children[0]]["parent_node"] = parents[1]
    elif mutation == "sub_nodes":
        graph.nodes[parents[0]]["sub_nodes"] = [children[1]]
    else:
        graph.graph["hierarchy"][parents[0]] = [children[1]]
    with pytest.raises(ValueError):
        _lineage_targets(graph, prefix, "original_parents")


def test_admission_retains_every_target_and_does_not_enable_optional_gate(monkeypatch):
    import benchmarks.thol_distributed_target as benchmark
    from tnfr.operators.grammar_dynamics import CandidateResult, GrammarViolation

    graph, _, parents, _ = _lineage_graph()
    seen = []

    def grammar(_graph, node, glyph):
        seen.append(node)
        return CandidateResult(
            glyph,
            node != parents[0],
            (
                []
                if node != parents[0]
                else [
                    GrammarViolation(
                        "test", "Explicit synthetic grammar refusal", "error"
                    ),
                ]
            ),
        )

    def unexpected_optional(*args):
        raise AssertionError("a disabled optional gate must stay disabled")

    monkeypatch.setattr(benchmark, "validate_candidate", grammar)
    monkeypatch.setattr(benchmark, "validate_coupling", unexpected_optional)
    before = benchmark._common_source(graph)
    observation = benchmark._cohort_admission(graph, parents)
    assert seen == list(parents)
    assert observation["targets"] == parents
    assert not observation["allowed"]
    assert tuple(row["allowed"] for row in observation["candidates"]) == (False, True)
    assert all(
        row["hard_u3_allowed"]
        and not row["optional_gate_enabled"]
        and row["optional_gate_allowed"] is None
        for row in observation["candidates"]
    )
    assert observation["source_projection_unchanged"]
    assert benchmark._common_source(graph) == before


@pytest.mark.parametrize("gate", ("validate_phase_gate_u3", "validate_coupling"))
def test_only_explicit_precondition_refusals_are_retained_as_admission_outcomes(
    monkeypatch, gate
):
    import benchmarks.thol_distributed_target as benchmark
    from tnfr.operators.preconditions import OperatorPreconditionError

    graph, _, parents, _ = _lineage_graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = gate == "validate_coupling"
    calls = []

    def refused(_graph, node, *args):
        calls.append(node)
        if node == parents[0]:
            raise OperatorPreconditionError(
                "Coupling", "Explicit synthetic gate refusal"
            )

    monkeypatch.setattr(benchmark, gate, refused)
    observation = benchmark._cohort_admission(graph, parents)
    assert calls == list(parents)
    assert not observation["allowed"]
    field = (
        "hard_u3_refusal"
        if gate == "validate_phase_gate_u3"
        else "optional_gate_refusal"
    )
    assert "Explicit synthetic gate refusal" in observation["candidates"][0][field]
    assert observation["candidates"][1][field] is None
    assert tuple(row["allowed"] for row in observation["candidates"]) == (False, True)


@pytest.mark.parametrize(
    "gate,exception",
    (
        ("validate_candidate", TypeError),
        ("validate_phase_gate_u3", ValueError),
        ("validate_coupling", RuntimeError),
    ),
)
def test_unexpected_admission_failures_propagate_instead_of_becoming_refused_branches(
    monkeypatch,
    gate,
    exception,
):
    import benchmarks.thol_distributed_target as benchmark

    graph, _, parents, _ = _lineage_graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

    def broken(*args, **kwargs):
        raise exception("unexpected implementation failure")

    monkeypatch.setattr(benchmark, gate, broken)
    with pytest.raises(exception, match="unexpected implementation failure"):
        benchmark._cohort_admission(graph, parents)


def test_actual_cohorts_reproduce_complete_controls_and_the_frozen_observer(
    study, control_fixture
):
    from benchmarks.thol_lineage_coordination import COMMON_FIELDS

    controls, path, digest = control_fixture
    assert tuple(branch["branch"] for branch in study["branches"]) == (
        "original_parents",
        "born_children",
    )
    assert not study["fresh_control_trajectories_executed"]
    binding = study["retained_controls"]
    assert binding["sha256"] == digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert binding["historical_producer_preserved"]
    assert binding["historical_manifest"] == json.loads(path.read_bytes())["manifest"]
    for branch in study["branches"]:
        comparison = branch["retained_common_source_comparison"]
        assert comparison["all_fields_equal"] and comparison["fields"] == COMMON_FIELDS
        for control in controls["branches"]:
            for name in COMMON_FIELDS:
                assert branch[name] == control[name]
        original = branch["original_reference"]
        assert branch["original_reference_frozen_time"] == 0.5
        assert branch["original_reference_frozen_before_baseline_flow"]
        _assert_reference(original)
        assert (
            original["source"]
            == branch["prefix"]["coupling"]["refreshed_forcing"]["observation"][
                "snapshot"
            ]
        )
        assert branch["initial_target"]["pattern"]["error_variance"] > 0


@pytest.mark.parametrize("index", (0, 1))
def test_real_ancestry_selects_an_exhaustive_cohort_and_commits_only_its_um_history(
    study, index
):
    branch = study["branches"][index]
    assert branch["status"] == "executed"
    pairs = branch["prefix"]["birth"]["parent_children"]
    parents = tuple(parent for parent, _ in pairs)
    children = tuple(child for _, child in pairs)
    lineage = branch["lineage"]
    selected = parents if index == 0 else children
    assert parents == branch["prefix"]["birth"]["before"]["nodes"]
    assert not set(parents) & set(children)
    assert len(parents) == len(children) == 8
    assert lineage["parent_children"] == pairs
    assert lineage["parents"] == parents and lineage["children"] == children
    assert lineage["current_nodes"] == parents + children
    assert lineage["selected_targets"] == selected
    assert lineage["disjoint_and_exhaustive"] and lineage["live_parentage_verified"]
    admission = branch["admission"]
    assert admission["targets"] == selected
    assert admission["allowed"] and admission["source_projection_unchanged"]
    assert tuple(row["node"] for row in admission["candidates"]) == selected
    assert all(
        row["grammar"]["allowed"] and row["hard_u3_allowed"] and row["allowed"]
        for row in admission["candidates"]
    )
    coupling = branch["event"]["coupling"]
    before, after = coupling["before"], coupling["after_raw"]
    assert before == branch["before_optional_event"]
    assert before["nodes"] == after["nodes"] == parents + children
    assert before["epi"] == after["epi"]
    assert before["time"] == after["time"] == 1.0
    assert coupling["targets"] == selected
    assert coupling["node_sample_used"] == before["nodes"]
    assert coupling["sampling_refresh_executed"]
    assert coupling["functional_links"] and coupling["bidirectional"]
    assert coupling["candidate_limit"] == 0 and coupling["candidate_mode"] == "sample"
    assert coupling["stage_result"]["schedule"] == "two_phase_jacobi"
    assert coupling["stage_result"]["nodes_processed"] == 8
    for node in before["nodes"]:
        assert after["glyph_history"][node] == (
            before["glyph_history"][node] + (("UM",) if node in selected else ())
        )


@pytest.mark.parametrize("index,expected_new_edges", ((0, 0), (1, 8)))
def test_actual_phase_capacity_and_edges_follow_the_complete_merged_proposal(
    study,
    control_fixture,
    index,
    expected_new_edges,
):
    branch = study["branches"][index]
    coupling = branch["event"]["coupling"]
    before, after = coupling["before"], coupling["after_raw"]
    proposal = coupling["kernel_proposal"]
    assert proposal["targets"] == coupling["targets"]
    assert len(proposal["target_proposals"]) == 8
    all_proposals = {
        row["node"]: row
        for row in control_fixture[0]["branches"][1]["event"]["coupling"][
            "kernel_proposal"
        ]["target_proposals"]
    }
    assert all(
        row == all_proposals[row["node"]] for row in proposal["target_proposals"]
    )
    expected = {name: list(before[name]) for name in ("phase", "capacity", "pressure")}
    for update in proposal["node_updates"]:
        position = before["nodes"].index(update["node"])
        for key, name in (
            ("theta_after", "phase"),
            ("vf_after", "capacity"),
            ("dnfr_after", "pressure"),
        ):
            if update[key] is not None:
                expected[name][position] = update[key]
    for name, values in expected.items():
        assert after[name] == tuple(values)
    actual_edges = {
        frozenset((left, right)): data["weight"]
        for left, right, data in coupling["new_edges"]
    }
    proposed_edges = {
        frozenset((edge["left"], edge["right"])): edge["weight"]
        for edge in proposal["edges"]
    }
    assert actual_edges == proposed_edges
    assert len(actual_edges) == expected_new_edges
    assert len(after["edges"]) == 24 + expected_new_edges
    inventory = coupling["support_inventory"]
    assert (
        inventory["directed_unique_support_entries"] == 2 * len(after["edges"]) <= 100
    )
    assert len(inventory["positive_conductance_components"]) == 1
    effects = branch["proposal_and_effects"]
    assert effects["selected_target_local_proposals_match_all_node_control"]
    recipients = {node: [] for node in before["nodes"]}
    for row in proposal["target_proposals"]:
        for write in row["phase_proposals"]:
            recipients[write["node"]].append(write["source"])
    assert effects["phase_proposal_sources_by_node"] == tuple(
        {"node": node, "sources": tuple(sources)}
        for node, sources in recipients.items()
    )
    assert effects["multiply_proposed_phase_nodes"] == tuple(
        node for node, sources in recipients.items() if len(sources) > 1
    )
    for name, channel in (("phase_changes", "phase"), ("capacity_changes", "capacity")):
        for i, change in enumerate(effects[name]):
            assert change["node"] == before["nodes"][i]
            assert change["before"] == before[channel][i]
            assert change["after"] == after[channel][i]
            assert change["exact_represented_difference"] == (
                Fraction.from_float(after[channel][i])
                - Fraction.from_float(before[channel][i])
            )
    assert "not an additive decomposition" in effects["scope"]


@pytest.mark.parametrize("index", (0, 1))
def test_exact_targets_and_two_finite_flow_partitions_keep_the_original_metric(
    study, index
):
    branch = study["branches"][index]
    original = branch["original_reference"]
    for key in (
        "initial_target",
        "baseline_target",
        "post_event_target",
        "endpoint_target",
    ):
        _assert_target(branch[key], original)
    for name, start in (("baseline", 0.5), ("continuation", 1.0)):
        flow, steps = branch[name + "_flow"], branch[name + "_steps"]
        assert flow["before"]["time"] == start
        assert flow["after"]["time"] == start + 0.5
        assert flow["partition"]["segment_durations"] == (0.25, 0.25)
        assert tuple(row["time"] for row in flow["boundaries"]) == (
            start,
            start + 0.25,
            start + 0.5,
        )
        assert len(flow["segments"]) == len(steps) == 2
        assert flow["all_segment_binary64_replays_identified"]
        assert all(
            value
            for key, value in branch[name + "_capture_checks"].items()
            if key != "scope"
        )
        for segment, step in zip(flow["segments"], steps, strict=True):
            assert step["before"]["snapshot"] == segment["before_support"]
            assert step["after"]["snapshot"] == segment["after_support"]
            assert step["dt"] == Fraction(1, 4)
            _assert_step(step)
    assert branch["continuation_flow"]["before"] == branch["after_optional_event"]
    assert branch["endpoint"] == branch["continuation_flow"]["after"]
    assert branch["endpoint"]["time"] == 1.5


@pytest.mark.parametrize("index", (0, 1))
def test_zero_epi_um_preserves_instantaneous_old_target_and_exact_reset_budget(
    study, index
):
    branch = study["branches"][index]
    event = branch["event"]["exact_forced_event"]
    assert event["epi_jump"] == event["centered_epi_jump"] == (0,) * 16
    assert event["mean_epi_jump"] == 0
    assert (
        branch["baseline_target"]["pattern"] == branch["post_event_target"]["pattern"]
    )
    assert event["midpoint_pattern"] == branch["baseline_target"]["pattern"]
    assert event["before_reference"] == branch["original_reference"]
    assert event["after_reference"] == branch["post_event_target"]["reference"]
    for name in ("variance_jump_budget", "dirichlet_jump_budget"):
        assert not any(event[name].values())
    before, after = event["before"], event["after"]
    old, new = event["before_reference"], event["after_reference"]
    u, v = before["relative_error"], after["relative_error"]
    shift = tuple(b - a for a, b in zip(u, v, strict=True))
    h0, h1 = old["metric_weights"], new["metric_weights"]
    variance = event["variance_reset_budget"]
    assert (
        variance["metric_term"]
        == _dot(
            tuple(b - a for a, b in zip(h0, h1, strict=True)), tuple(x * x for x in u)
        )
        / 2
    )
    assert variance["reference_cross_term"] == _dot(
        h1, tuple(x * y for x, y in zip(u, shift, strict=True))
    )
    assert (
        variance["reference_quadratic_term"]
        == _dot(h1, tuple(x * x for x in shift)) / 2
    )
    b0, b1 = old["source"]["conductance"], new["source"]["conductance"]
    dirichlet = event["dirichlet_reset_budget"]
    assert dirichlet["metric_term"] == _energy(b1, u) - _energy(b0, u)
    assert dirichlet["reference_cross_term"] == _dot(_laplacian(b1, u), shift)
    assert dirichlet["reference_quadratic_term"] == _energy(b1, shift)
    for budget in (variance, dirichlet):
        assert budget["energy_change"] == sum(
            budget[name]
            for name in (
                "metric_term",
                "reference_cross_term",
                "reference_quadratic_term",
            )
        )
        assert budget["identity_residual"] == 0
    assert event["mean_reweighting"] == after["mean"] - before["mean"]
    assert event["mean_change"] == event["mean_reweighting"]


@pytest.mark.parametrize("index", (0, 1))
def test_signed_channel_and_same_state_rate_sums_are_independent_of_stored_pressure(
    study, index
):
    branch = study["branches"][index]
    before, after = branch["baseline_target"], branch["post_event_target"]
    metric = branch["original_reference"]["metric_weights"]
    midpoint = tuple(
        (a + b) / 2
        for a, b in zip(
            before["compatibility_residual"],
            after["compatibility_residual"],
            strict=True,
        )
    )
    old, new = dict(before["projected_rate_channels"]), dict(
        after["projected_rate_channels"]
    )
    expected = tuple(
        (
            name,
            _dot(
                metric,
                tuple(
                    mid * (b - a)
                    for mid, a, b in zip(
                        midpoint,
                        old[name],
                        new[name],
                        strict=True,
                    )
                ),
            ),
        )
        for name in old
    )
    allocation = branch["event"]["signed_target_channel_change"]
    assert allocation["channel_contributions"] == expected
    assert (
        sum(value for _, value in expected)
        == after["compatibility_energy"] - before["compatibility_energy"]
    )
    error = before["pattern"]["relative_error"]

    def rate(target):
        state = target["state"]["snapshot"]
        return _dot(
            metric,
            tuple(
                u * nu * p
                for u, nu, p in zip(
                    error, state["capacity"], state["stored_pressure"], strict=True
                )
            ),
        )

    change = branch["event"]["same_state_rate_change"]
    assert change["stored_nodal_energy_rate_change"] == rate(after) - rate(before)
    components = (
        "homogeneous_energy_rate",
        "target_source_energy_rate",
        "stored_pressure_energy_rate_defect",
    )
    for name in components:
        assert change[name + "_change"] == after[name] - before[name]
    assert (
        sum(change[name + "_change"] for name in components)
        == change["stored_nodal_energy_rate_change"]
    )
    assert change["identity_residual"] == 0
    coupling = branch["event"]["coupling"]
    raw, fresh = coupling["raw_forcing"], coupling["refreshed_forcing"]
    assert raw["observation"]["forcing"] == fresh["observation"]["forcing"]
    assert (
        raw["observation"]["snapshot"]["stored_pressure"]
        != fresh["observation"]["snapshot"]["stored_pressure"]
    )
    for capture in (raw, fresh):
        assert (
            tuple(
                sum(vector[i] for _, vector in capture["components"]) for i in range(16)
            )
            == capture["observation"]["forcing"]
        )


def test_observed_cohort_outcomes_preserve_incompatibility_and_do_not_improve_on_no_event(
    study,
):
    outcomes = {
        row["branch"]: row
        for row in (*study["control_outcomes"], *study["lineage_outcomes"])
    }
    no_event, all_nodes = outcomes["no_event"], outcomes["all_node_um"]
    parents, children = outcomes["original_parents"], outcomes["born_children"]
    assert (
        no_event["fixed_target_variance"]
        < parents["fixed_target_variance"]
        < children["fixed_target_variance"]
        < all_nodes["fixed_target_variance"]
    )
    assert no_event["target_compatible"]
    for row in (parents, children):
        assert row["status"] == "executed" and row["endpoint_time"] == 1.5
        assert not row["target_compatible"] and row["compatibility_energy"] > 0
        assert row["original_reference_sha256"] == no_event["original_reference_sha256"]
    for branch in study["branches"]:
        assert (
            branch["endpoint_target"]["pattern"]["error_variance"]
            < branch["baseline_target"]["pattern"]["error_variance"]
        )
        assert "do not prove asymptotic recovery" in branch["scope"]


@pytest.mark.parametrize(
    "mutation",
    ("bytes", "claim", "branches", "observer", "baseline", "source", "assertion"),
)
def test_retained_control_admission_rejects_wrong_bytes_or_incompatible_complete_evidence(
    control_fixture,
    tmp_path,
    mutation,
):
    from benchmarks.thol_lineage_coordination import load_control_evidence

    _, source, digest = control_fixture
    payload = json.loads(source.read_bytes())
    if mutation == "bytes":
        source = tmp_path / "wrong_bytes.json"
        source.write_bytes(b"{}")
        with pytest.raises(ValueError, match="expected digest"):
            load_control_evidence(source, expected_sha256=digest)
        return
    if mutation == "claim":
        payload["manifest"]["claim_id"] = "O1.b-unrelated"
    elif mutation == "branches":
        payload["branches"].reverse()
    elif mutation == "observer":
        payload["branches"][1]["original_reference"]["relative_profile"][0] = "0"
    elif mutation == "baseline":
        payload["branches"][1]["baseline_flow"]["after"]["time"] = 99
    elif mutation == "source":
        payload["branches"][1]["common_source"]["ordered_neighbors"].reverse()
    else:
        payload["common_causal_baseline_reproduced"] = False
    raw = json.dumps(payload, allow_nan=False).encode()
    destination = tmp_path / "mutated.json"
    destination.write_bytes(raw)
    with pytest.raises(ValueError):
        load_control_evidence(
            destination, expected_sha256=hashlib.sha256(raw).hexdigest()
        )


def test_fresh_control_comparison_rejects_changed_history_even_with_equal_scalar_metrics(
    study, control_fixture
):
    from benchmarks.thol_lineage_coordination import (
        compare_common_source,
        load_control_evidence,
    )

    _, path, digest = control_fixture
    controls, _ = load_control_evidence(path, expected_sha256=digest)
    changed = deepcopy(study["branches"][0])
    changed["common_source"]["node_sample"] = tuple(
        reversed(changed["common_source"]["node_sample"])
    )
    with pytest.raises(ValueError, match="common_source"):
        compare_common_source(changed, controls)


def test_explicit_refusal_retains_baseline_without_retry_or_continuation(monkeypatch):
    import benchmarks.thol_distributed_target as benchmark
    from tnfr.operators.grammar_dynamics import CandidateResult, GrammarViolation

    calls = []
    physical_flow = benchmark._physical_flow

    def record_flow(*args, **kwargs):
        calls.append("flow")
        return physical_flow(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_physical_flow", record_flow)
    monkeypatch.setattr(
        benchmark,
        "validate_candidate",
        lambda *args: CandidateResult(
            "UM",
            False,
            [GrammarViolation("test", "Synthetic explicit refusal", "error")],
        ),
    )
    with patch.object(
        benchmark, "_couple_parents", side_effect=AssertionError("refused UM executed")
    ):
        branch = benchmark.run_distributed_target_branch("born_children")
    assert calls == ["flow"]
    assert branch["status"] == "refused" and not branch["admission"]["allowed"]
    assert len(branch["admission"]["candidates"]) == 8
    assert (
        branch["event"]
        is branch["continuation_flow"]
        is branch["endpoint_target"]
        is None
    )
    assert branch["continuation_steps"] == ()
    assert branch["endpoint"]["time"] == 1.0
    assert (
        branch["before_optional_event"]
        == branch["after_optional_event"]
        == branch["endpoint"]
    )
    assert branch["post_event_target"] == branch["baseline_target"]


def test_cli_payload_keeps_full_literal_histories_and_bounded_claims(
    study, control_fixture, tmp_path, monkeypatch
):
    import benchmarks.thol_lineage_coordination as benchmark

    _, controls, digest = control_fixture
    output = tmp_path / "report.json"
    calls = []

    def captured_study(path, *, expected_control_sha256):
        calls.append((path, expected_control_sha256))
        return study

    monkeypatch.setattr(benchmark, "run_study", captured_study)
    monkeypatch.setattr(
        "sys.argv",
        [
            "thol_lineage_coordination",
            "--controls",
            str(controls),
            "--expected-control-sha256",
            digest,
            "--output",
            str(output),
        ],
    )
    benchmark.main()
    assert calls == [(controls, digest)]
    payload = json.loads(output.read_bytes())
    CoreExperimentManifest(**payload["manifest"]).validate_for_admission()
    assert payload["experimental_status"] == "No empirical correspondence tested"
    for branch in payload["branches"]:
        source = branch["common_source"]
        for node, attributes in source["node_attributes"]:
            history = attributes.get("glyph_history", [])
            if history and history[0] == "deque":
                history = history[2]
            assert history == source["state"]["glyph_history"][str(node)]
        for name in ("baseline_flow", "continuation_flow"):
            assert not branch[name]["solver_accuracy_certified"]
            assert not branch[name]["mesh_convergence_certified"]
            assert not branch[name]["future_or_repeated_behavior_certified"]
