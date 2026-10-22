"""Protocol admission and retained endpoint evidence without repeating evolution."""

import json
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from benchmarks import relational_capture_response as campaign
from tnfr.dynamics.relational import evaluate_relational_exchange


def test_preparation_preserves_declared_reflection_and_points_away_from_cancellation():
    graph = campaign._graph()
    before = campaign._owned_state(graph)
    field = evaluate_relational_exchange(graph, model=campaign._model())
    frame = campaign._snapshot(graph)
    assert campaign._owned_state(graph) == before
    assert frame["windings"] == (0, 0)
    assert frame["reflected"]["report"]["exact_symmetry"]
    assert not frame["capture_admitted"]
    assert min(field.resultant_real_lower_bounds) > 0
    assert field.phase_rate[0] > 0
    assert field.phase_rate[4] == 0
    # r=2B-A: its actual initial derivative retreats from b=pi/2.
    assert 2 * field.form_rate[4] - field.form_rate[0] < 0
    assert Q(8) < field.storage < Q(44, 5)


def test_protocol_rejects_changed_input_before_any_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unfrozen execution")

    monkeypatch.setattr(campaign, "_trace", forbidden)
    prediction = campaign.prepare_prediction()
    for key, value in (("capacity", (True,) * 10), ("horizon", Q(128))):
        changed = deepcopy(prediction)
        changed[key] = value
        with pytest.raises(ValueError, match="frozen protocol"):
            campaign.evaluate_prediction(changed)


def test_stop_retains_real_state_without_retry_or_fabricated_capture(monkeypatch):
    calls = []

    def rejected(graph, *, model, dt):
        calls.append(dt)
        raise ValueError("synthetic guard rejection")

    monkeypatch.setattr(campaign, "step_relational_exchange", rejected)
    trace = campaign._trace(campaign.STEP_COUNTS[0], campaign.prepare_prediction())
    assert calls == [1 / 64]
    assert trace["status"] == "stopped_by_shared_owner"
    assert trace["completed_steps"] == 0
    assert trace["stop"]["owned_state_unchanged"]
    assert trace["stop"]["rejected_proposal"] is None
    assert not trace["endpoint_capture_admitted"]
    assert not trace["observed_winding_prediction_passed"]
    assert list(trace["checkpoints"]) == ["0"]


def test_cli_freeze_is_exclusive_and_failed_evaluation_is_retained(
    tmp_path, monkeypatch
):
    output = tmp_path / "result.json"

    def failed(prediction):
        raise RuntimeError("synthetic evaluation failure")

    monkeypatch.setattr(campaign, "evaluate_prediction", failed)
    with pytest.raises(FileNotFoundError):
        campaign.main(["--output", str(output)])
    assert campaign.main(["--prepare", "--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        campaign.main(["--prepare", "--output", str(output)])
    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        campaign.main(["--output", str(output)])
    assert "synthetic evaluation failure" in json.loads(output.read_text())["error"]
    with pytest.raises(FileExistsError):
        campaign.main(["--output", str(output)])


@pytest.fixture(scope="module")
def retained():
    directory = campaign.ROOT / "docs/assets/relational_capture_response"

    def exact(item):
        return (
            Q(item["numerator"], item["denominator"])
            if set(item) == {"numerator", "denominator"}
            else item
        )

    return tuple(
        json.loads((directory / name).read_text(encoding="utf-8"), object_hook=exact)
        for name in ("result.prediction.json", "result.json")
    )


def test_retained_decisions_recompute_from_real_endpoint_without_new_evolution(
    retained,
):
    prediction, report = retained
    assert report["prediction"] == prediction
    passed = []
    for count in campaign.STEP_COUNTS:
        trace = report["traces"][str(count)]
        assert trace["initial_winding_zero"]
        assert trace["dt"] == campaign.HORIZON / count
        frames = trace["checkpoints"]
        for key, frame in frames.items():
            assert Q(frame["time"]) == int(key) * trace["dt"]
            field = frame["reflected"]["report"]["field"]
            graph = campaign._graph()
            for node in graph:
                graph.nodes[node].update(
                    EPI=field["epi"][node], theta=field["phase"][node]
                )
            replay = campaign._snapshot(graph)
            assert replay["capture_admitted"] == frame["capture_admitted"]
            assert list(replay["windings"]) == frame["windings"]
            for certificate in ("reflected", "local"):
                assert (
                    replay[certificate]["report"]["status"]
                    == frame[certificate]["report"]["status"]
                )
            assert min(field["resultant_real_lower_bounds"]) > 0
        last = frames[trace["last_admitted_checkpoint"]]
        assert trace["endpoint_capture_admitted"] == last["capture_admitted"]
        if trace["status"] == "target_basin_certified_at_checkpoint":
            assert last["capture_admitted"] and trace["stop"] is None
        elif trace["status"] == "stopped_by_shared_owner":
            assert trace["stop"]["attempted_step"] == trace["completed_steps"] + 1
            assert trace["stop"]["owned_state_unchanged"]
        else:
            assert trace["completed_steps"] == count
        after = [f for f in frames.values() if f["time"] >= 0.25]
        assert trace["observed_winding_prediction_passed"] == (
            bool(after)
            and all(
                f["windings"] == [1, 1] and all(f["winding_defined"]) for f in after
            )
        )
        passed.append(
            trace["observed_winding_prediction_passed"] and last["capture_admitted"]
        )
    assert report["finite_prediction_passed"] == all(passed)
    assert report["grid_comparisons"] == campaign._comparison(report["traces"])


def test_post_evaluation_sector_audit_keeps_original_failure_and_detached_hashes(
    monkeypatch,
):
    from benchmarks import relational_capture_audit as audit
    from tnfr.dynamics import relational

    def forbidden(*args, **kwargs):
        raise AssertionError("an endpoint audit cannot execute a trajectory")

    monkeypatch.setattr(relational, "step_relational_exchange", forbidden)
    report = audit.audit_endpoints()
    assert report["original_finite_prediction_passed"] is False
    assert report["all_endpoints_admitted"]
    assert report["runtime"]["temporal_execution"] == "none"
    for endpoint in report["endpoints"].values():
        assert endpoint["time"] == 64
        assert endpoint["original_capture_admitted"] is False
        assert endpoint["sector_capture_admitted"]
        certificate = endpoint["certificate"]["report"]
        assert certificate["status"] == "admitted"
        assert certificate["ring_windings"] == [1, 1]
        assert certificate["bridge_winding"] == 0
        margin = campaign._fraction(certificate["storage_margin_lower_bound"])
        assert margin > Q(5, 1000)
    digest = audit.EXPECTED_SHA256["result.json"]
    report["original_records_sha256"]["result.json"] = "altered"
    assert audit.EXPECTED_SHA256["result.json"] == digest


def test_frozen_source_archive_preserves_every_original_byte(retained):
    import hashlib
    import zipfile

    prediction, _ = retained
    archive_path = (
        campaign.ROOT
        / "docs/assets/relational_capture_response/source-at-evaluation.zip"
    )
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(prediction["source_sha256"])
        for name, digest in prediction["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        directory = archive_path.parent
        audit = json.loads((directory / "endpoint-capture.audit.json").read_text())
        with zipfile.ZipFile(directory / "source-at-endpoint-audit.delta.zip") as delta:
            # The small overlay retains only files changed for that later
            # audit, avoiding a second complete package archive.
            required_overlay = set()
            for name, digest in audit["source_sha256"].items():
                base_matches = (
                    name in archive.namelist()
                    and hashlib.sha256(archive.read(name)).hexdigest() == digest
                )
                if not base_matches:
                    required_overlay.add(name)
                source = archive.read(name) if base_matches else delta.read(name)
                assert hashlib.sha256(source).hexdigest() == digest
            assert set(delta.namelist()) == required_overlay
