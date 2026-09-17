"""The relay budget retains original evidence and a separate conditional branch."""

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks import c6_winding_relay_budget as campaign

SOURCE = Path(__file__).resolve().parents[2] / "artifacts/research" / campaign.INPUT_NAME


@pytest.fixture(scope="module")
def parent():
    if not SOURCE.is_file():
        pytest.skip("the retained original 89-cycle B43 report is unavailable")
    return json.loads(SOURCE.read_bytes())


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_winding_relay_budget(parent)


def test_actual_pre_sha_carry_is_retained_and_conditional_scope_is_explicit(report, parent):
    source = report["source"]
    assert campaign._payload(source["retained_B43_endpoint"]) == parent["B43_original_phase_tail_runtime"]["endpoint_before_SHA"]
    assert source["origin_mean"] == F(1, 2) + F(3293, 3 * 2**114)
    assert source["historical_nodal_steps_replayed"] == 356
    assert source["serialized_numerical_chain_verified"] is True
    assert source["live_execution_seal_recreated"] is False
    assert source["capacity"] == (1.,) * 6
    assert source["historical_terminal_SHA_capacity"] == (.9204225284540524,) * 6
    assert report["conditional_numerical_steps"] == 7
    assert report["conditional_elapsed_time"] == F(7, 16)
    assert report["new_live_graph_steps"] == 0
    assert report["conditional_product_relay_inclusion_certified"] is True
    assert report["conditional_family_escape_certified"] is True
    for flag in ("whole_band_exit_certified", "full_region_invariant", "indefinite_trapping_certified",
                 "general_correlated_region_excluded", "future_runtime_certified"):
        assert report[flag] is False


def test_deadline_precedes_bounded_observation_and_complete_vector_area_is_retained(report):
    certificate, observed = report["B46_two_relay_certificate"], report["B46_conditional_first_exit"]
    assert certificate["deadline"] == 10
    assert certificate["deadline_nodes"] == (4,)
    assert observed["exit_step"] == len(observed["steps"]) == 7
    assert observed["exiting_nodes"] == (4,)
    assert observed["band_failure"] is False
    assert "closure" not in certificate and "relay" not in observed
    initial = report["source"]["retained_B43_endpoint"]
    endpoint = observed["endpoint"]
    assert tuple((F(x) - F(1, 2)) * 2**54 for x in endpoint["epi"]) == (-3, 0, 2, 0, 6, -5)
    x = tuple(F(a) + b for a, b in zip(initial["epi"], initial["remainder"], strict=True))
    y = tuple(F(a) + b for a, b in zip(endpoint["epi"], endpoint["remainder"], strict=True))
    area = tuple(b - a for a, b in zip(x, y, strict=True))
    assert y == observed["exact_endpoint"]
    assert area == observed["total_nodal_area"]
    assert observed["mean_area"] == sum(area, F(0)) / 6
    assert observed["nodal_balance_residual"] == (F(0),) * 6
    for step in observed["steps"]:
        assert step["nodal_balance_residual"] == (F(0),) * 6
    assert any(area)


def test_analysis_is_immutable_and_serializes_exact_fractions(parent):
    before = deepcopy(parent)
    result = campaign.analyze_c6_winding_relay_budget(parent)
    assert parent == before
    json.dumps(campaign._payload(result), allow_nan=False)


@pytest.mark.parametrize("field,value", (("flow_count", 355), ("runtime_provenance_certified_at_capture", False)))
def test_analysis_requires_complete_retained_numeric_chain(parent, field, value):
    changed = deepcopy(parent)
    changed["B43_original_phase_tail_runtime"][field] = value
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_relay_budget(changed)


def test_public_relay_observers_use_the_same_physics_owner():
    import tnfr.physics as physics
    from tnfr.physics import c6_carried_relay

    for name in c6_carried_relay.__all__:
        assert name in physics.__all__
        assert getattr(physics, name) is getattr(c6_carried_relay, name)


def test_cli_retains_historical_input_identity_and_current_source(tmp_path, monkeypatch, parent, report):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    raw = json.dumps(parent).encode()
    source.write_bytes(raw)
    monkeypatch.setattr(campaign, "analyze_c6_winding_relay_budget", lambda _parent, **_: deepcopy(report))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    monkeypatch.setattr(sys, "argv", ["campaign", "--input", str(source), "--output", str(output)])
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["input_evidence"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["input_evidence"]["producer_manifest"] == parent["manifest"]
    assert result["input_evidence"]["producer_source_scope"] == parent["source_scope"]
    assert result["manifest"]["dirty_source_hash"] == "sha256:" + "b" * 64
    assert source.read_bytes() == raw


@pytest.mark.parametrize("change", ("source", "input"))
def test_cli_fails_closed_when_evidence_changes_during_analysis(tmp_path, monkeypatch, parent, report, change):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(parent))
    versions = iter((("a" * 40, True, "sha256:" + "b" * 64),
                     ("a" * 40, True, "sha256:" + ("c" if change == "source" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(versions))

    def analyze(_parent, **_):
        if change == "input":
            source.write_text("{}")
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_relay_budget", analyze)
    monkeypatch.setattr(sys, "argv", ["campaign", "--input", str(source), "--output", str(output)])
    with pytest.raises(RuntimeError, match="source or historical input changed"):
        campaign.main()
    assert not output.exists()


def test_cli_never_overwrites_historical_input(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["campaign", "--input", str(source), "--output", str(source)])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert source.read_text() == "{}"
