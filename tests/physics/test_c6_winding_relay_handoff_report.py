"""B47 retains the B43/B46 numeric lineage without claiming a live invocation."""

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks import c6_winding_relay_handoff as campaign
from benchmarks.c6_winding_relay_budget import replay_c6_relay_budget_evidence

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts/research"


@pytest.fixture(scope="module")
def historical():
    path = ARTIFACTS / campaign.HISTORICAL_NAME
    if not path.is_file():
        pytest.skip("the retained B43 source is unavailable")
    return path.read_bytes()


@pytest.fixture(scope="module")
def parent():
    path = ARTIFACTS / campaign.INPUT_NAME
    if not path.is_file():
        pytest.skip("the retained B46 source is unavailable")
    return json.loads(path.read_bytes())


@pytest.fixture(scope="module")
def report(parent, historical):
    return campaign.analyze_c6_winding_relay_handoff(parent, historical_bytes=historical)


def _exact(state):
    return tuple(F(x) + F(r) for x, r in zip(state["epi"], state["remainder"], strict=True))


def test_source_carry_and_conditional_lineage_remain_explicit(report, parent):
    source = report["source"]
    assert campaign._payload(source["retained_B46_endpoint"]) == parent["B46_conditional_first_exit"]["endpoint"]
    assert source["historical_nodal_steps_replayed"] == 356
    assert source["previous_conditional_steps_replayed"] == 7
    assert source["serialized_numerical_chain_verified"] is True
    assert source["live_execution_seal_recreated"] is False
    assert source["origin_mean"] == F(1, 2) + F(3296, 3 * 2**114)
    assert report["conditional_lineage_from_B46_numerically_verified"] is True
    assert report["conditional_numerical_steps"] == 197
    assert report["total_conditional_steps_after_B43"] == 204
    assert report["conditional_elapsed_time"] == F(197, 16)
    assert report["new_live_graph_steps"] == 0
    for flag in ("whole_band_exit_certified", "indefinite_trapping_certified", "future_runtime_certified"):
        assert report[flag] is False


def test_nonzero_coupled_switch_term_does_not_break_local_budget(report):
    control = report["B47_boundary_pressure_interaction"]
    assert len(control["pressure_rows"]) == 8
    assert control["node3_node4_mixed_increment"] == tuple(F(x, 2**113) for x in (0, 0, 0, -8, -8, 0))
    assert control["node0_pair_mixed_increments"] == ((F(0),) * 6,) * 2
    assert control["triple_mixed_increment"] == (F(0),) * 6
    assert control["three_independent_relays_refuted"] is True
    assert control["universal_locality_is_not_inferred_from_eight_samples"] is True
    budget = report["B47_local_relay_budget"]
    assert budget["held_nodes"] == (1, 2, 5) and budget["free_nodes"] == (3, 4)
    assert budget["deadline"] == 198 and budget["band_covers_deadline"] is True
    assert budget["budget_drift"] < 0
    coefficient = -budget["budget_correction"]
    values = tuple(row[1] + coefficient * row[0] for row in control["exact_increments"])
    assert values == (budget["budget_drift"],) * 8


def test_complete_handoff_vector_area_and_first_held_boundary(report):
    observed = report["B47_conditional_first_exit"]
    source = report["source"]["retained_B46_endpoint"]
    assert "budget" not in observed
    assert observed["exit_step"] == len(observed["steps"]) == 197
    assert len(observed["points"]) == 198
    assert observed["exiting_nodes"] == (1,) and observed["band_failure"] is False
    endpoint = observed["endpoint"]
    assert tuple((F(x) - F(1, 2)) * 2**54 for x in endpoint["epi"]) == (-3, -1, 2, 0, 8, -5)
    change = tuple(y - x for x, y in zip(_exact(source), _exact(endpoint), strict=True))
    assert change == observed["total_nodal_area"]
    assert sum(change, F(0)) / 6 == observed["mean_area"] == F(349, 3 * 2**114)
    assert observed["nodal_balance_residual"] == (F(0),) * 6
    assert all(step["nodal_balance_residual"] == (F(0),) * 6 for step in observed["steps"])
    assert report["original_preparation_plus_conditional_mean_area"] == F(1215, 2**114)
    assert report["original_preparation_plus_conditional_nodal_area"] == tuple(x - F(1, 2) for x in _exact(endpoint))
    assert any(change)


def test_analysis_does_not_mutate_inputs_and_serializes_exact_evidence(parent, historical):
    original = deepcopy(parent)
    report = campaign.analyze_c6_winding_relay_handoff(parent, historical_bytes=historical)
    assert parent == original
    json.dumps(campaign._payload(report), allow_nan=False)


@pytest.fixture(scope="module")
def retained_handoff():
    path = ARTIFACTS / "c6_winding_relay_handoff.json"
    if not path.is_file():
        pytest.skip("the retained B47 source is unavailable")
    return json.loads(path.read_bytes())


def test_replay_binds_complete_handoff_and_returns_original_carry(retained_handoff, historical):
    relay_bytes = (ARTIFACTS / campaign.INPUT_NAME).read_bytes()
    original = deepcopy(retained_handoff)
    profile, state = campaign.replay_c6_relay_handoff_evidence(
        retained_handoff, relay_bytes=relay_bytes, historical_bytes=historical,
    )
    assert state.exact_epi == _exact(retained_handoff["B47_conditional_first_exit"]["endpoint"])
    assert profile.lattice.source.phase == tuple(retained_handoff["source"]["phase"])
    assert retained_handoff == original


@pytest.mark.parametrize("path,value", (
    (("B47_conditional_first_exit", "endpoint", "remainder", 0), "0"),
    (("B47_conditional_first_exit", "steps", 0, "pressure", 0), 0.),
    (("B47_local_relay_budget", "deadline"), 199),
    (("original_preparation_plus_conditional_mean_area",), "0"),
    (("indefinite_trapping_certified",), True),
    (("input_evidence", "sha256"), "f" * 64),
    (("input_evidence", "historical_sha256"), "f" * 64),
    (("input_evidence", "producer_source_scope"), ["src"]),
    (("manifest", "claim_id"), "another-claim"),
    (("source_scope",), ["src"]),
))
def test_handoff_replay_rejects_changed_evidence(
    retained_handoff, historical, report, monkeypatch, path, value,
):
    # The complete live replay is covered above. Here isolate validation of
    # every retained field without repeating its unchanged historical work.
    monkeypatch.setattr(campaign, "analyze_c6_winding_relay_handoff", lambda *_a, **_k: deepcopy(report))
    changed = deepcopy(retained_handoff)
    entry = changed
    for key in path[:-1]:
        entry = entry[key]
    entry[path[-1]] = value
    with pytest.raises(ValueError):
        campaign.replay_c6_relay_handoff_evidence(
            changed, relay_bytes=(ARTIFACTS / campaign.INPUT_NAME).read_bytes(), historical_bytes=historical,
        )


def test_handoff_replay_requires_exact_retained_bytes(retained_handoff, historical):
    relay_bytes = (ARTIFACTS / campaign.INPUT_NAME).read_bytes()
    with pytest.raises(TypeError, match="exact retained"):
        campaign.replay_c6_relay_handoff_evidence(
            retained_handoff, relay_bytes=relay_bytes.decode(), historical_bytes=historical,
        )
    with pytest.raises(ValueError, match="input hashes"):
        campaign.replay_c6_relay_handoff_evidence(
            retained_handoff, relay_bytes=relay_bytes, historical_bytes=historical + b"\n",
        )


@pytest.mark.parametrize("path,value", (
    (("B46_conditional_first_exit", "endpoint", "remainder", 0), "0"),
    (("B46_conditional_first_exit", "steps", 0, "pressure", 0), 0.),
    (("B46_two_relay_certificate", "deadline"), 11),
    (("B46_conditional_first_exit", "mean_area"), "0"),
    (("conditional_numerical_steps",), True),
    (("source", "live_execution_seal_recreated"), True),
    (("input_evidence", "producer_source_scope"), ["src"]),
    (("input_evidence", "sha256"), "f" * 64),
    (("manifest", "claim_id"), "another-claim"),
    (("source_scope",), ["src"]),
))
def test_handoff_rejects_tampered_numeric_chain_and_provenance(parent, historical, path, value):
    changed = deepcopy(parent)
    entry = changed
    for key in path[:-1]:
        entry = entry[key]
    entry[path[-1]] = value
    with pytest.raises(ValueError):
        replay_c6_relay_budget_evidence(changed, historical_bytes=historical)


def test_handoff_requires_exact_historical_bytes(parent, historical):
    with pytest.raises(ValueError, match="input hash"):
        replay_c6_relay_budget_evidence(parent, historical_bytes=historical + b"\n")
    with pytest.raises(TypeError, match="exact retained"):
        replay_c6_relay_budget_evidence(parent, historical_bytes=historical.decode())


def test_cli_binds_both_inputs_to_current_result(tmp_path, monkeypatch, parent, historical, report):
    source, base, output = (tmp_path / name for name in ("source.json", "base.json", "output.json"))
    raw = json.dumps(parent).encode()
    source.write_bytes(raw)
    base.write_bytes(historical)
    monkeypatch.setattr(campaign, "analyze_c6_winding_relay_handoff", lambda *_a, **_k: deepcopy(report))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: ("a" * 40, True, "sha256:" + "b" * 64))
    monkeypatch.setattr(sys, "argv", ["campaign", "--input", str(source), "--historical-input", str(base), "--output", str(output)])
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["input_evidence"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["input_evidence"]["historical_sha256"] == hashlib.sha256(historical).hexdigest()
    assert result["input_evidence"]["producer_manifest"] == parent["manifest"]
    assert result["manifest"]["dirty_source_hash"] == "sha256:" + "b" * 64


@pytest.mark.parametrize("which", ("source", "historical"))
def test_cli_cannot_overwrite_either_input(tmp_path, monkeypatch, which):
    source, base = tmp_path / "source.json", tmp_path / "base.json"
    source.write_text("{}")
    base.write_text("{}")
    target = source if which == "source" else base
    monkeypatch.setattr(sys, "argv", ["campaign", "--input", str(source), "--historical-input", str(base), "--output", str(target)])
    with pytest.raises(ValueError, match="must not overwrite"):
        campaign.main()
    assert source.read_text() == base.read_text() == "{}"


@pytest.mark.parametrize("which", ("source", "historical", "code"))
def test_cli_rejects_mid_analysis_changes(tmp_path, monkeypatch, parent, historical, report, which):
    source, base, output = (tmp_path / name for name in ("source.json", "base.json", "output.json"))
    source.write_text(json.dumps(parent))
    base.write_bytes(historical)
    versions = iter((("a" * 40, True, "sha256:" + "b" * 64),
                     ("a" * 40, True, "sha256:" + ("c" if which == "code" else "b") * 64)))
    monkeypatch.setattr(campaign, "current_git_source_provenance", lambda *_: next(versions))

    def analyze(*_args, **_kwargs):
        if which != "code":
            (source if which == "source" else base).write_text("{}")
        return deepcopy(report)

    monkeypatch.setattr(campaign, "analyze_c6_winding_relay_handoff", analyze)
    monkeypatch.setattr(sys, "argv", ["campaign", "--input", str(source), "--historical-input", str(base), "--output", str(output)])
    with pytest.raises(RuntimeError, match="source or historical input changed"):
        campaign.main()
    assert not output.exists()
