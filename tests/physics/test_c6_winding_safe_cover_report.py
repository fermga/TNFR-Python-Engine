"""Report admission, original-target ownership and incomplete cover scope."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks import c6_winding_invariant_region as campaign


@pytest.fixture(scope="module")
def evidence():
    path = campaign.ROOT / "artifacts/research/c6_winding_safe_partition.json"
    if not path.is_file():
        pytest.skip("retained C6 research evidence is unavailable")
    raw = path.read_bytes()
    hint = json.loads(raw)
    from pathlib import Path
    inputs = tuple(Path(item["path"]).read_bytes() for item in hint["input_evidence"])
    assert len(inputs) == 14
    keys = ("parent_bytes", "relay_bytes", "historical_bytes", "envelope_bytes", "region_bytes",
            "excursion_bytes", "candidates_bytes", "mode_bytes", "return_bytes", "union_bytes",
            "history_candidates_bytes", "history_bytes", "memory_bytes", "cuts_bytes")
    return raw, hint, inputs, dict(zip(keys, inputs, strict=True))


@pytest.fixture(scope="module")
def report(evidence):
    raw, _, inputs, arguments = evidence
    parent = json.loads(inputs[13])
    for key in ("manifest", "source_scope", "input_evidence"):
        parent.pop(key)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "analyze_c6_winding_memory_cuts", lambda *_a, **_k: deepcopy(parent))
        return campaign.analyze_c6_winding_safe_partition(
            json.loads(inputs[0]), **arguments, priority_bytes=raw, priority_target=(28, 3, "upper"),
            excluded_predecessor_depth=1, max_partition_work=2000, query_max_intersections=1,
        )


def test_cover_report_keeps_scope_and_every_original_target(report, evidence):
    raw, _, inputs, _ = evidence
    labels = json.loads(inputs[13])["B58_remaining_first_exit_facets"]
    assert [row["target"] for row in report["B63_partition_queries"]] == labels
    assert report["B63_cover_verified"]
    assert report["B63_cover"]["schedule"]["policy"] == "priority_fifo"
    assert report["B63_cover"]["schedule"]["classification_work"] == 992
    assert report["B63_priority_evidence"] == dict(
        sha256=hashlib.sha256(raw).hexdigest(), target=dict(mask=28, node=3, direction="upper"),
        advisory_only=True, authorizes_cuts=False,
    )
    assert not report["B63_query_relation_complete"]
    assert report["excluded_first_exit_slabs"] == 40
    assert report["additional_first_exit_slabs_excluded"] == 0
    assert report["new_conditional_trajectory_steps"] == report["new_live_graph_steps"] == 0
    assert not report["indefinite_trapping_certified"] and not report["future_runtime_certified"]
    assert not any(key.startswith(("B59_", "B60_")) for key in report)
    json.dumps(campaign._payload(report), allow_nan=False)


@pytest.mark.parametrize("index", range(14))
def test_priority_hint_lineage_binds_every_parent(evidence, index):
    raw, _, inputs, _ = evidence
    changed = list(inputs)
    changed[index] += b" "
    with pytest.raises(ValueError, match="lineage"):
        campaign._c6_priority_hints(raw, (28, 3, "upper"), tuple(changed), 5000)


@pytest.mark.parametrize("target", (
    (True, 3, "upper"), (28, False, "upper"), (64, 3, "upper"),
    (28, 6, "upper"), (28, 3, "side"), [28, 3, "upper"],
))
def test_exact_priority_target_admission(evidence, target):
    raw, _, inputs, _ = evidence
    with pytest.raises(ValueError, match="priority_target"):
        campaign._c6_priority_hints(raw, target, inputs, 5000)


def test_missing_sentinel_and_oversized_hints_rejected(evidence):
    raw, hint, inputs, _ = evidence
    with pytest.raises(ValueError, match="bounded"):
        campaign._c6_priority_hints(raw, (28, 3, "upper"), inputs, 1)
    changed = deepcopy(hint)
    query = next(row["query"] for row in changed["B59_partition_queries"]
                 if row["target"] == dict(mask=28, node=3, direction="upper"))
    query["retained_endpoint_zones"].pop()
    with pytest.raises(ValueError, match="sentinel"):
        campaign._c6_priority_hints(json.dumps(changed).encode(), (28, 3, "upper"), inputs, 5000)


def test_target_without_hint_rejected_before_reconstruction(evidence, monkeypatch):
    _, _, inputs, arguments = evidence

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid advisory arguments must fail before ancestor reconstruction")
    monkeypatch.setattr(campaign, "analyze_c6_winding_memory_cuts", forbidden)
    with pytest.raises(ValueError, match="requires advisory"):
        campaign.analyze_c6_winding_safe_partition(
            json.loads(inputs[0]), **arguments, priority_target=(28, 3, "upper"),
        )


@pytest.mark.parametrize("maximum", (True, 0, 1.5))
def test_cover_budget_rejected_before_reconstruction(evidence, monkeypatch, maximum):
    raw, _, inputs, arguments = evidence

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid cover budgets must fail before ancestor reconstruction")

    monkeypatch.setattr(campaign, "analyze_c6_winding_memory_cuts", forbidden)
    with pytest.raises(ValueError, match="max_cover_work"):
        campaign.analyze_c6_winding_safe_partition(
            json.loads(inputs[0]), **arguments, priority_bytes=raw,
            priority_target=(28, 3, "upper"), max_cover_work=maximum,
        )
