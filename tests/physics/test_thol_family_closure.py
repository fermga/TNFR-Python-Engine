"""Portable retained-model tests; fixtures carry no executed-birth authority."""

import hashlib
import json
import platform
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from benchmarks.thol_distributed_target import _common_source
from benchmarks.thol_pressure_feedback import _payload, _state
from tests.physics.test_forced_epi_closure import _apply, _geometry, _subtract
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_target,
)
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.research.claims import ClaimStatus
from tnfr.research.core_manifests import (
    CoreExperimentManifest,
    current_git_source_provenance,
)

F = Fraction


def _json(value):
    return json.loads(json.dumps(_payload(value), allow_nan=False))


def _write(path, value):
    data = (json.dumps(_payload(value), allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _graph():
    graph = nx.Graph()
    graph.add_nodes_from((0, 1, 2, 3))
    graph.add_edges_from(((0, 1), (0, 2), (1, 3)))
    for node, epi, capacity in zip(
        graph, (1.75, -0.5, 0.25, 1.25), (1.0, 1.0, 0.5, 0.5), strict=True
    ):
        graph.nodes[node].update(
            EPI=epi, nu_f=capacity, theta=0.0, delta_nfr=0.0, glyph_history=[]
        )
    for parent, child in ((0, 2), (1, 3)):
        graph.nodes[parent]["sub_nodes"] = [child]
        graph.nodes[child]["parent_node"] = parent
    graph.graph.update(
        hierarchy={0: [2], 1: [3]},
        _node_sample=(0, 1, 2, 3),
        _t=1.0,
        DNFR_WEIGHTS={"phase": 1.0, "epi": 1.0, "vf": 1.0, "topo": 1.0},
    )
    return graph


def _readout(graph):
    captured = capture_non_epi_forcing(graph)
    components = decompose_non_epi_forcing(captured)
    reference = derive_forced_support_balance(
        captured.snapshot,
        epi_weight=captured.epi_weight,
        forcing=captured.forcing,
    )
    return {"observation": asdict(captured), "components": components}, reference


def _target(original, current, readout):
    from tnfr.physics.support_transport import SupportTransportSnapshot

    return asdict(
        observe_forced_support_target(
            original,
            current,
            SupportTransportSnapshot(**readout["observation"]["snapshot"]),
            forcing_components=readout["components"],
        )
    )


@pytest.fixture(scope="module")
def retained_fixture(tmp_path_factory):
    """Small complete static geometry, explicitly not a simulated THOL history."""
    directory = tmp_path_factory.mktemp("family_closure")
    control_path, lineage_path = directory / "control.json", directory / "lineage.json"
    scope = ("src/tnfr", "benchmarks/thol_family_closure.py")
    git_sha, dirty, source_hash = current_git_source_provenance(
        Path(__file__).resolve().parents[2], scope
    )

    def manifest(claim, destination):
        record = CoreExperimentManifest(
            claim_id=claim,
            git_sha=git_sha,
            source_dirty=dirty,
            dirty_source_hash=source_hash,
            versions={"python": platform.python_version()},
            result_status=ClaimStatus.MEASURED,
            graph_construction="Synthetic detached four-node observer fixture; no execution provenance",
            capacity_specification="Explicit positive capacities; no outcome fitting",
            solver="No solver or trajectory; exact readouts only",
            seed=0,
            timestep=0.25,
            operator_sequence=(),
            telemetry=("Exact affine model and supplied family relation",),
            controls=("Retained unchanged model",),
            artifacts=(str(destination),),
        )
        record.validate_for_admission()
        return record.to_dict()

    graph = _graph()
    readout, original = _readout(graph)
    target = _target(original, original, readout)
    digest = hashlib.sha256(
        json.dumps(
            _payload(asdict(original)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    state = _state(graph)
    prefix = {
        "birth": {
            "parent_children": ((0, 2), (1, 3)),
            "before": {"nodes": (0, 1)},
            "after": deepcopy(state),
        },
        "coupling": {"refreshed_forcing": readout},
    }
    common = {
        "prefix": prefix,
        "original_reference": asdict(original),
        "original_reference_sha256": digest,
        "original_reference_frozen_time": 0.5,
        "original_reference_frozen_before_baseline_flow": True,
        "initial_target": target,
        "baseline_target": target,
        "baseline_flow": {"after_forcing": readout},
        "baseline_steps": (),
        "baseline_capture_checks": {"fixture_geometry_unchanged": True},
        "common_source": _common_source(graph),
        "before_optional_event": state,
    }
    records, references = {}, {"original": original}
    for name in ("no_event", "all_node_um", "original_parents", "born_children"):
        changed = deepcopy(graph)
        if name in ("all_node_um", "original_parents"):
            for node in (0, 1):
                changed.nodes[node]["nu_f"] = 1.25
        if name in ("all_node_um", "born_children"):
            for node in (2, 3):
                changed.nodes[node]["nu_f"] = 0.75
            changed.add_edge(2, 3, weight=0.5)
        current_readout, current = _readout(changed)
        references[name] = current
        records[name] = {
            **deepcopy(common),
            "branch": name,
            "status": "executed",
            "after_optional_event": _state(changed),
            "event": (
                None
                if name == "no_event"
                else {
                    "coupling": {"refreshed_forcing": current_readout},
                }
            ),
            "post_event_target": _target(original, current, current_readout),
            "endpoint_target": _target(original, current, current_readout),
            "endpoint": _state(changed),
        }
        if name in ("original_parents", "born_children"):
            records[name]["lineage"] = {
                "parent_children": ((0, 2), (1, 3)),
                "parents": (0, 1),
                "children": (2, 3),
                "current_nodes": (0, 1, 2, 3),
                "live_parentage_verified": True,
                "disjoint_and_exhaustive": True,
                "selected_targets": (0, 1) if name == "original_parents" else (2, 3),
            }
    control = {
        "manifest": manifest(
            "O1.b-distributed-fixed-relative-target-response", control_path
        ),
        "source_scope": scope,
        "common_causal_baseline_reproduced": True,
        "branches": tuple(records[name] for name in ("no_event", "all_node_um")),
        "fixture_scope": "Common static fixture identity only; no executed trajectory claim",
    }
    control_hash = _write(control_path, control)
    lineage = {
        "manifest": manifest(
            "O1.b-lineage-scoped-UM-fixed-target-response", lineage_path
        ),
        "source_scope": scope,
        "retained_controls": {
            "path": str(control_path),
            "sha256": control_hash,
            "byte_count": control_path.stat().st_size,
            "historical_manifest": control["manifest"],
            "historical_source_scope": scope,
            "historical_producer_preserved": True,
        },
        "branches": tuple(
            records[name] for name in ("original_parents", "born_children")
        ),
        "fixture_scope": control["fixture_scope"],
    }
    lineage_hash = _write(lineage_path, lineage)
    return {
        "control_path": control_path,
        "lineage_path": lineage_path,
        "control_hash": control_hash,
        "lineage_hash": lineage_hash,
        "control": _json(control),
        "lineage": _json(lineage),
        "references": references,
    }


@pytest.fixture(scope="module")
def study(retained_fixture):
    from benchmarks.thol_family_closure import run_study

    with (
        patch(
            "tnfr.physics.epi_memory.matrix_exponential",
            side_effect=AssertionError("no exponential"),
        ),
        patch(
            "tnfr.physics.structural_morphism.matrix_exponential",
            side_effect=AssertionError("no numerical flow"),
        ),
        patch(
            "benchmarks.thol_lineage_coordination.run_distributed_target_branch",
            side_effect=AssertionError("no new trajectory"),
        ),
    ):
        return run_study(
            retained_fixture["control_path"],
            retained_fixture["lineage_path"],
            expected_control_sha256=retained_fixture["control_hash"],
            expected_lineage_sha256=retained_fixture["lineage_hash"],
        )


def test_four_retained_models_preserve_supplied_family_and_producer_identity(
    study, retained_fixture
):
    assert tuple(row["model"] for row in study["models"]) == (
        "original",
        "original_parents",
        "born_children",
        "all_node_um",
    )
    assert study["partition"]["nodes"] == (0, 1, 2, 3)
    assert (
        study["partition"]["blocks"]
        == study["partition"]["parent_children"]
        == ((0, 2), (1, 3))
    )
    assert study["partition"]["family_count"] == 2
    assert all(row["all_fields_equal"] for row in study["common_source_comparison"])
    assert (
        not study["new_trajectories_executed"]
        and not study["partition_search_performed"]
    )
    for name in ("control", "lineage"):
        binding = study[
            "retained_controls" if name == "control" else "retained_lineage"
        ]
        assert binding["sha256"] == retained_fixture[name + "_hash"]
        assert binding["historical_manifest"] == retained_fixture[name]["manifest"]
        assert (
            "Synthetic detached" in binding["historical_manifest"]["graph_construction"]
        )
        assert binding["historical_producer_preserved"]


def test_every_retained_exact_matrix_is_checked_independently(study, retained_fixture):
    from types import SimpleNamespace

    from tests.physics.test_forced_epi_closure import _assert_geometry

    blocks = study["partition"]["blocks"]
    for row in study["models"]:
        reference = retained_fixture["references"][row["model"]]
        expected = _geometry(reference, blocks)
        _assert_geometry(SimpleNamespace(**row["exact_observation"]), expected)
        assert row["observation_time"] == 1.0
        assert row["exact_observation"]["reference"] == asdict(reference)
        assert row["actual_snapshot"] == asdict(reference.source)
        assert any(row["exact_observation"]["affine_source"])
        numeric = row["numerical_crosscheck"]
        assert max(numeric["exact_cast_discrepancies"].values()) < 1e-12
        assert "not the exact closure decision" in numeric["scope"]


def test_numerical_norm_labels_and_values_mean_spectral_not_infinity_or_frobenius():
    from benchmarks.thol_family_closure import _numerical_crosscheck
    from tnfr.physics.epi_memory import observe_forced_support_closure
    from tnfr.physics.support_transport import observe_support_transport

    # Three macro coordinates and three hidden coordinates permit rank two,
    # unlike a two-block fixture where the relevant spectral/Frobenius norms
    # can coincide because constants supply a null direction.
    graph = nx.path_graph(6)
    for i in graph:
        graph.nodes[i].update(EPI=i, nu_f=1 if i % 2 == 0 else 2, delta_nfr=0, theta=0)
    reference = derive_forced_support_balance(
        observe_support_transport(graph),
        epi_weight=F(2, 5),
        forcing=tuple(F(i, 7) for i in graph),
    )
    blocks = ((0, 1), (2, 3), (4, 5))
    exact = observe_forced_support_closure(reference, blocks)
    numeric = _numerical_crosscheck(reference, blocks, exact)
    w = np.zeros((6, 6))
    for i, j, value in reference.source.conductance:
        w[i, j] = float(value)
    d = w.sum(axis=1)
    nu = np.asarray(reference.source.capacity, dtype=float)
    h = d / nu
    p = np.asarray([[float(i in block) for block in blocks] for i in graph])
    r = np.asarray(
        [
            [h[i] / sum(h[j] for j in block) if i in block else 0.0 for i in graph]
            for block in blocks
        ]
    )
    a = float(reference.epi_weight) * nu[:, None] * (np.eye(6) - w / d[:, None])
    q = np.eye(6) - p @ r
    raq, qap = (r @ a) @ q, (q @ a) @ p
    k0 = raq @ (a @ p)
    assert np.linalg.matrix_rank(raq) == 2
    assert np.linalg.norm(raq, 2) != pytest.approx(
        np.linalg.norm(raq, np.inf), rel=1e-5
    )
    assert np.linalg.norm(raq, 2) != pytest.approx(np.linalg.norm(raq, "fro"), rel=1e-5)
    for prefix, matrix in (
        ("projection_defect", raq),
        ("lift_defect", qap),
        ("zero_lag_kernel", k0),
    ):
        assert prefix + "_inf_norm" not in numeric
        assert numeric[prefix + "_spectral_norm"] == pytest.approx(
            np.linalg.norm(matrix, 2),
            rel=1e-14,
            abs=1e-16,
        )
    assert "spectral" in numeric["norm_definition"].lower()
    for name, matrix, field in (
        ("R", r, "projection"),
        ("A", a, "micro_generator"),
        ("RAQ", raq, "hidden_to_macro"),
        ("QAP", qap, "macro_to_hidden"),
        ("K0", k0, "instantaneous_kernel"),
    ):
        expected = np.linalg.norm(
            matrix - np.asarray(getattr(exact, field), dtype=float), 2
        )
        assert numeric["exact_cast_discrepancies"][name] == pytest.approx(
            expected, rel=1e-14, abs=1e-30
        )
    abar = (r @ a) @ p
    for name, mapping, source, target in (
        ("projection_intertwining", r, a, abar),
        ("lift_intertwining", p, abar, a),
    ):
        left, right = mapping @ source, target @ mapping
        residual = np.linalg.norm(left - right, 2)
        scale = max(1.0, np.linalg.norm(left, 2), np.linalg.norm(right, 2))
        assert numeric[name]["residual"] == pytest.approx(residual, rel=1e-14)
        assert numeric[name]["residual_scale"] == pytest.approx(scale, rel=1e-14)
        assert numeric[name]["relative_residual"] == pytest.approx(
            residual / scale, rel=1e-14
        )


def test_observer_reset_is_exact_reweighting_of_one_unchanged_microstate(
    study, retained_fixture
):
    references = retained_fixture["references"]
    original = references["original"]
    blocks = study["partition"]["blocks"]
    x = original.source.epi
    g0 = _geometry(original, blocks)
    projected0 = _apply(g0["R"], x)
    mean0 = sum(h * v for h, v in zip(g0["H"], x, strict=True)) / sum(g0["H"])
    u0 = tuple(v - mean0 - z for v, z in zip(x, original.relative_profile, strict=True))
    saw_nonzero_reweighting = False
    for row in study["models"]:
        reference = references[row["model"]]
        g1 = _geometry(reference, blocks)
        assert reference.source.epi == x
        reset = row["observer_reset"]
        projected1 = _apply(g1["R"], x)
        change = tuple(b - a for a, b in zip(projected0, projected1, strict=True))
        assert reset["projected_epi_before"] == projected0
        assert reset["projected_epi_after"] == projected1
        assert (
            reset["projected_epi_change"] == reset["projection_reweighting"] == change
        )
        assert change == _apply(_subtract(g1["R"], g0["R"]), x)
        mean1 = sum(h * v for h, v in zip(g1["H"], x, strict=True)) / sum(g1["H"])
        u1 = tuple(
            v - mean1 - z for v, z in zip(x, reference.relative_profile, strict=True)
        )
        assert reset["current_mean_before"] == mean0
        assert reset["current_mean_after"] == mean1
        assert reset["current_mean_reweighting"] == mean1 - mean0
        assert reset["projected_relative_error_before"] == _apply(g0["R"], u0)
        assert reset["projected_relative_error_after"] == _apply(g1["R"], u1)
        profile_shift = tuple(
            b - a
            for a, b in zip(
                _apply(g0["R"], original.relative_profile),
                _apply(g1["R"], reference.relative_profile),
                strict=True,
            )
        )
        assert reset["projected_profile_shift"] == profile_shift
        assert reset["projected_relative_error_change"] == tuple(
            raw - (mean1 - mean0) - profile
            for raw, profile in zip(change, profile_shift, strict=True)
        )
        assert reset["identity_residual"] == (0, 0)
        assert (
            row["old_target_readout"]["pattern"]
            == study["models"][0]["old_target_readout"]["pattern"]
        )
        saw_nonzero_reweighting |= any(change)
    assert saw_nonzero_reweighting


@pytest.mark.parametrize("which", ("control", "lineage"))
def test_input_content_hash_is_required_before_any_retained_evidence_is_admitted(
    retained_fixture, which
):
    from benchmarks.thol_family_closure import load_evidence

    hashes = {
        "expected_control_sha256": retained_fixture["control_hash"],
        "expected_lineage_sha256": retained_fixture["lineage_hash"],
    }
    hashes["expected_" + which + "_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        load_evidence(
            retained_fixture["control_path"], retained_fixture["lineage_path"], **hashes
        )


@pytest.mark.parametrize("value", (None, True, "", "f" * 63, "G" * 64, "A" * 64))
def test_digest_format_cannot_be_auto_admitted(value):
    from benchmarks.thol_family_closure import load_evidence

    with pytest.raises(ValueError, match="SHA256"):
        load_evidence(
            "missing_control.json",
            "missing_lineage.json",
            expected_control_sha256=value,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "producer",
        "branch_order",
        "refused",
        "control_binding",
        "parent_pointer",
        "family",
        "cached_target",
        "cached_snapshot",
        "forcing",
        "event_epi",
        "event_time",
        "original_hash",
        "event_capacity",
        "event_pressure",
        "event_phase",
    ),
)
def test_hash_bound_payload_still_requires_lineage_and_exact_model_consistency(
    retained_fixture,
    tmp_path,
    mutation,
):
    from benchmarks.thol_family_closure import run_study

    lineage = deepcopy(retained_fixture["lineage"])
    branch = lineage["branches"][0]
    if mutation == "producer":
        lineage["manifest"]["claim_id"] = "O1.b-unrelated"
    elif mutation == "branch_order":
        lineage["branches"].reverse()
    elif mutation == "refused":
        branch["status"] = "refused"
    elif mutation == "control_binding":
        lineage["retained_controls"]["historical_manifest"][
            "solver"
        ] = "wrong provenance"
    elif mutation == "parent_pointer":
        branch["common_source"]["node_attributes"][2][1]["parent_node"] = 1
    elif mutation == "family":
        branch["lineage"]["parent_children"][0][1] = 3
    elif mutation == "cached_target":
        branch["post_event_target"]["compatibility_energy"] = "99"
    elif mutation == "cached_snapshot":
        branch["event"]["coupling"]["refreshed_forcing"]["observation"]["snapshot"][
            "rate"
        ][0] = "99"
    elif mutation == "forcing":
        branch["event"]["coupling"]["refreshed_forcing"]["components"][0][1][0] = "99"
    elif mutation == "event_epi":
        branch["after_optional_event"]["epi"][0] += 1
    elif mutation == "event_time":
        branch["after_optional_event"]["time"] = 1.25
    elif mutation in ("event_capacity", "event_pressure", "event_phase"):
        branch["after_optional_event"][mutation.removeprefix("event_")][0] += 0.25
    else:
        branch["original_reference_sha256"] = "0" * 64
    path = tmp_path / "mutated.json"
    digest = _write(path, lineage)
    with pytest.raises(ValueError):
        run_study(
            retained_fixture["control_path"],
            path,
            expected_control_sha256=retained_fixture["control_hash"],
            expected_lineage_sha256=digest,
        )


def test_cli_serialization_retains_input_producers_and_no_new_trajectory_claim(
    study,
    retained_fixture,
    tmp_path,
    monkeypatch,
):
    import benchmarks.thol_family_closure as benchmark

    output = tmp_path / "analysis.json"
    called = []

    def retained(*args, **kwargs):
        called.append((args, kwargs))
        return study

    monkeypatch.setattr(benchmark, "run_study", retained)
    monkeypatch.setattr(
        "sys.argv",
        [
            "thol_family_closure",
            "--controls",
            str(retained_fixture["control_path"]),
            "--lineage",
            str(retained_fixture["lineage_path"]),
            "--expected-control-sha256",
            retained_fixture["control_hash"],
            "--expected-lineage-sha256",
            retained_fixture["lineage_hash"],
            "--output",
            str(output),
        ],
    )
    benchmark.main()
    assert len(called) == 1
    payload = json.loads(output.read_bytes())
    CoreExperimentManifest(**payload["manifest"]).validate_for_admission()
    assert (
        payload["retained_controls"]["historical_manifest"]
        == retained_fixture["control"]["manifest"]
    )
    assert (
        payload["retained_lineage"]["historical_manifest"]
        == retained_fixture["lineage"]["manifest"]
    )
    assert payload["experimental_status"] == "No empirical correspondence tested"
    assert not payload["new_trajectories_executed"]
