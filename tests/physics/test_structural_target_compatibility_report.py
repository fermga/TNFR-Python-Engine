"""Detached schema controls; synthetic records do not claim causal execution."""

import json
import sys
from copy import deepcopy
from dataclasses import asdict

import networkx as nx
import pytest

from benchmarks import structural_target_compatibility as analysis
from benchmarks.thol_pressure_feedback import _payload, _state
from tnfr.dynamics import fused_dnfr
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_pattern,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing


def _prepared(capacity):
    graph = nx.Graph()
    graph.add_edge("1", "2", weight=1.0)
    for node, nu, epi in zip(graph, capacity, (0.125, 0.25), strict=True):
        graph.nodes[node].update(EPI=epi, nu_f=nu, theta=0.0, delta_nfr=0.0)
    graph.graph["_t"] = 9.5
    default_compute_delta_nfr(graph)
    capture = capture_non_epi_forcing(graph)
    reference = derive_forced_support_balance(
        capture.snapshot,
        epi_weight=capture.epi_weight,
        forcing=capture.forcing,
    )
    return graph, capture, reference


@pytest.fixture
def parent():
    _, _, target = _prepared((1.0, 1.0))
    cases = []
    for name, capacity in (("U", (1.0, 1.0)), ("P", (1.25, 1.0)), ("F", (1.125, 1.0))):
        graph, capture, reference = _prepared(capacity)
        actual = _state(graph)
        limit = observe_forced_support_pattern(
            target,
            nodes=reference.source.nodes,
            epi=reference.relative_profile,
        )
        cases.append(
            {
                "case": name,
                "status": "measured",
                "original_target": asdict(target),
                "postevent_reference": asdict(reference),
                "initial": {"state": actual},
                "events": (
                    []
                    if name == "U"
                    else [
                        {
                            "after_forcing_capture": asdict(capture),
                            "after_refresh": {"state": actual},
                        }
                    ]
                ),
                "segments": [
                    {
                        "forcing_capture": asdict(capture),
                        "after_refresh": {"time": 9.75},
                    }
                ],
                "conditional_fixed_model_limit": {"pattern": asdict(limit)},
            }
        )
    return _payload({"cases": cases})


def test_offline_report_preserves_literal_node_ids_and_exact_channel_allocation(
    parent, monkeypatch
):
    def forbidden(**kwargs):
        raise AssertionError("offline analysis cannot materialize a fresh phase kernel")

    monkeypatch.setattr(fused_dnfr, "compute_fused_gradients_symmetric", forbidden)
    report = analysis.analyze_target_report(parent)
    u, p, f = report["cases"]
    assert report["runtime_executed"] is False
    assert u["observation"]["state"]["snapshot"]["nodes"] == ("1", "2")
    assert u["observation"]["target_compatible"] is True
    assert p["observation"]["target_compatible"] is False
    assert u["channel_capture_time"] == 9.75 and u["state_time"] == 9.5
    assert f["channel_capture_time"] == f["state_time"] == 9.5
    delta = report["feedback_channel_change"]
    assert delta["compatibility_energy_change"] < 0
    assert sum(value for _, value in delta["channel_contributions"]) == (
        f["observation"]["compatibility_energy"]
        - p["observation"]["compatibility_energy"]
    )
    assert delta["identity_residual"] == 0
    assert parent["cases"][0]["original_target"]["source"]["nodes"] == ["1", "2"]


@pytest.mark.parametrize("branch", (0, 1, 2))
@pytest.mark.parametrize("source_field", ("epi", "capacity", "stored_pressure"))
def test_stale_reference_source_cannot_replace_actual_endpoint(
    parent, branch, source_field
):
    case = parent["cases"][branch]
    case["postevent_reference"]["source"][source_field][0] = "17/16"
    with pytest.raises(ValueError, match="actual endpoint"):
        analysis.analyze_target_case(case)


def test_channel_capture_phase_must_match_held_actual_phase(parent):
    parent["cases"][1]["events"][0]["after_forcing_capture"]["phase"][0] = "1/2"
    with pytest.raises(ValueError, match="phase/forcing"):
        analysis.analyze_target_case(parent["cases"][1])


def test_recorded_limit_is_checked_against_rebuilt_reference(parent):
    parent["cases"][1]["conditional_fixed_model_limit"]["pattern"][
        "error_variance"
    ] = "0"
    with pytest.raises(ValueError, match="conditional limit"):
        analysis.analyze_target_case(parent["cases"][1])


def test_unmeasured_or_reordered_inputs_are_rejected(parent):
    missing = deepcopy(parent)
    missing["cases"][0]["status"] = "controlled_obstruction"
    with pytest.raises(ValueError, match="completed measured"):
        analysis.analyze_target_report(missing)
    with pytest.raises(ValueError, match="ordered U/P/F"):
        analysis.analyze_target_report({"cases": parent["cases"][::-1]})


def test_channel_allocation_requires_one_original_target(parent):
    before, _ = analysis.analyze_target_case(parent["cases"][0])
    changed = deepcopy(parent["cases"][1])
    changed["original_target"]["source"]["epi"][0] = "1/16"
    after, _ = analysis.analyze_target_case(changed)
    with pytest.raises(ValueError, match="identical original target"):
        analysis.compare_target_channels(before, after)


def test_main_cannot_overwrite_input_evidence(tmp_path, monkeypatch):
    source = tmp_path / "evidence.json"
    source.write_text("retained", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["target", "--input", str(source), "--output", str(source)]
    )
    with pytest.raises(ValueError, match="overwrite"):
        analysis.main()
    assert source.read_text(encoding="utf-8") == "retained"


def test_analysis_owns_its_source_scope_independently_of_producer_metadata(
    parent, tmp_path, monkeypatch
):
    parent.update(
        source_scope=["historical/producer.py"],
        manifest={
            "claim_id": "synthetic-schema-input",
            "git_sha": "1" * 40,
            "versions": {"python": "synthetic"},
            "graph_construction": "prepared fixture",
            "capacity_specification": "fixture",
            "solver": "no integration",
            "result_status": "measured",
            "source_dirty": False,
            "telemetry": ["synthetic channels"],
            "controls": ["synthetic P2"],
            "artifacts": ["synthetic-input.json"],
        },
    )
    source, output = tmp_path / "input.json", tmp_path / "derived.json"
    source.write_text(json.dumps(parent), encoding="utf-8")
    original_bytes = source.read_bytes()
    queried = []

    def provenance(root, scope):
        queried.append(tuple(scope))
        return "2" * 40, False, None

    monkeypatch.setattr(analysis, "current_git_source_provenance", provenance)
    monkeypatch.setattr(
        sys, "argv", ["target", "--input", str(source), "--output", str(output)]
    )
    analysis.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert queried == [analysis.SOURCE_SCOPE, analysis.SOURCE_SCOPE]
    assert "src/tnfr" in result["source_scope"]
    assert result["input_evidence"]["producer_source_scope"] == [
        "historical/producer.py"
    ]
    assert result["input_evidence"]["producer_manifest"]["git_sha"] == "1" * 40
    assert result["manifest"]["git_sha"] == "2" * 40
    assert source.read_bytes() == original_bytes
