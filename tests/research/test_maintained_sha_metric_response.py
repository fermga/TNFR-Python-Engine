"""Constructed integration controls; no retained target source is consumed."""

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from hashlib import sha256
from types import SimpleNamespace as NS
from zipfile import ZipFile

import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.research import maintained_sha_metric_response as owner
from tnfr.research.artifact_io import file_receipt, write_json_once
from tnfr.research.maintained_sha_cut_separator import CutSeparatorSource
from tnfr.research.maintained_sha_phase_evidence import PhaseReceipt
from tnfr.research.maintained_sha_preparation_response import (
    FrozenNominalEvidence,
    NominalPreparationBounds,
)


def source():
    host_edges = tuple((i, (i + 1) % 20) if i < 19 else (0, 19) for i in range(20))
    edges = host_edges + ((2, 20),)
    degrees = tuple(sum(i in edge for edge in edges) for i in range(21))
    cut_degrees = (2,) * 20 + (0,)
    form, phase = tuple(Q(i, 32) for i in range(21)), tuple(Q(i, 16) for i in range(21))
    mean_x, mean_t = sum(form[:20]) / 20, sum(phase[:20]) / 20
    return CutSeparatorSource(
        tuple(range(20)) + ("2_sub_0",),
        edges,
        (Q(1),) * 21,
        form,
        phase,
        (Q(1),) * 21,
        degrees,
        cut_degrees,
        2,
        20,
        tuple(range(20)),
        host_edges,
        (Q(1),) * 20,
        host_edges,
        (Q(1),) * 20,
        Q(1),
        Q(7) + Q(1, 2**19),
        (Q(2),) * 20,
        (Q(1, 20),) * 20,
        mean_x,
        mean_t,
        form[20] - mean_x,
        phase[20] - mean_t,
    )


def nominal():
    return NominalPreparationBounds(I(10), I(0), Q(1), Q(1))


@pytest.fixture
def comparison(monkeypatch):
    calls = []
    monkeypatch.setattr(owner, "require_phase_evidence", lambda value: value)

    def context(**kwargs):
        calls.append(("context", kwargs))
        return NS(**kwargs)

    def family(context, **kwargs):
        calls.append(("family", kwargs))
        return NS(context=context, **kwargs)

    outcomes = [
        NS(status="completed", reason=None, final_portrait_error_upper=Q(1, 8))
    ] * 2

    def run(family, **kwargs):
        calls.append(("run", kwargs))
        return outcomes[len([item for item in calls if item[0] == "run"]) - 1]

    monkeypatch.setattr(owner, "prepare_metric_context", context)
    monkeypatch.setattr(owner, "prepare_metric_family", family)
    monkeypatch.setattr(owner, "bound_metric_cells", run)
    branches = tuple(
        NS(branch=name, phase_cells=(("synthetic",),) * 256, receipts=())
        for name in ("connected", "cut_host")
    )
    return NS(calls=calls, outcomes=outcomes, branches=branches)


def test_integration_keeps_all42_asymmetric_hulls_cut_degrees_and_child_errors(
    comparison,
):
    result = owner._bound_metric_response(source(), nominal(), *comparison.branches)
    assert (
        result.status == "family_ordering_certified" and result.family_members == 3**42
    )
    assert len(result.hulls) == 42
    assert tuple(h.center for h in result.hulls) == source().epi + source().phase
    assert result.hulls[0].lower_error == result.hulls[0].upper_error == Q(1, 2**1074)
    contexts = [row for kind, row in comparison.calls if kind == "context"]
    families = [row for kind, row in comparison.calls if kind == "family"]
    assert contexts[0]["degrees"] == source().connected_degrees
    assert contexts[1]["degrees"] == (2,) * 20
    assert contexts[1]["edges"] == source().host_edges
    assert families[0]["child"] == 20 and families[1]["parent"] == 2
    assert families[1]["child_form_error"] == families[0]["form_errors"][20]
    assert families[1]["child_phase_error"] == families[0]["phase_errors"][20]
    assert len(families[0]["form_errors"]) == len(families[0]["phase_errors"]) == 21
    assert len(families[1]["form_errors"]) == len(families[1]["phase_errors"]) == 20
    for kind, row in comparison.calls:
        if kind == "run":
            assert row["durations"] == (Q(1, 512),) * 256
            assert set(row["edge_allowances"]) == {Q(1, 1024)}
    assert result.connected_radius_margin == result.cut_radius_margin == Q(63, 64)


@pytest.mark.parametrize(
    "reason", ("first_exit", "scalar_work", "initial_domain", "intermediate_budget")
)
def test_first_branch_insufficiency_stops_without_building_or_retrying_cut(
    comparison, reason
):
    comparison.outcomes[0] = NS(status="unavailable", reason=reason)
    result = owner._bound_metric_response(source(), nominal(), *comparison.branches)
    assert (
        result.status == "insufficient_bound" and result.reason == "connected_" + reason
    )
    assert result.cut_run is None and result.cut_family is None
    assert result.connected_radius_margin is result.cut_radius_margin is None
    assert [kind for kind, _ in comparison.calls] == ["context", "family", "run"]


def test_completed_arithmetic_cannot_replace_a_strict_portrait_gate(comparison):
    comparison.outcomes[1] = NS(
        status="completed", reason=None, final_portrait_error_upper=Q(1)
    )
    result = owner._bound_metric_response(source(), nominal(), *comparison.branches)
    assert result.status == "insufficient_bound"
    assert result.cut_radius_margin == 0


def test_shared_angle_gate_remains_required_after_both_radius_gates(comparison):
    low_gap = replace(nominal(), connected_clockwise=I(3))
    result = owner._bound_metric_response(source(), low_gap, *comparison.branches)
    assert result.status == "insufficient_bound"
    assert result.nominal_gap_margin < 0 < result.connected_radius_margin


def test_forged_phase_capability_rejected_before_any_family_hull(monkeypatch):
    def reject(value):
        raise ValueError("completed live phase evidence required")

    monkeypatch.setattr(owner, "require_phase_evidence", reject)
    monkeypatch.setattr(
        owner, "adjacent_coordinate_hull", lambda _: pytest.fail("width evaluated")
    )
    with pytest.raises(ValueError, match="live phase"):
        owner._bound_metric_response(source(), nominal(), object(), object())


@pytest.mark.parametrize("branch_name", ("connected", "cut_host"))
def test_marked_phase_link_uses_full_tube_and_exact_constant_child(
    monkeypatch, branch_name
):
    monkeypatch.setattr(owner, "require_phase_evidence", lambda value: value)
    original = source()
    form = tuple(I(value - Q(1, 64), value + Q(1, 64)) for value in original.epi)
    phase = tuple(I(value - Q(1, 128), value + Q(1, 128)) for value in original.phase)
    if branch_name == "cut_host":
        tube = form[:20] + phase[:20]
        x_child, t_child = I(original.epi[20]), I(original.phase[20])
    else:
        tube = form + phase
        x_child, t_child = form[20], phase[20]
    r, delta = x_child - form[2], t_child - phase[2]
    row = {"r": {"lo": r.lo, "hi": r.hi}, "delta": {"lo": delta.lo, "hi": delta.hi}}
    branch = NS(branch=branch_name, steps=(NS(tube=tube),) * 256, receipts=())
    saved = {"report": {"cells": (row,) * 256, "retained_receipts": ()}}
    owner._match_phase_observations(original, branch, saved)
    bad = dict(row, delta={"lo": delta.lo + 1, "hi": delta.hi + 1})
    saved["report"]["cells"] = (bad,) + (row,) * 255
    with pytest.raises(ValueError, match="full tube"):
        owner._match_phase_observations(original, branch, saved)


def evidence(root, inventory=()):
    empty = {"path": "unused.json", "bytes": 1, "sha256": "0" * 64}
    nominal = FrozenNominalEvidence(root, empty, empty, empty, empty, empty)
    return owner.FrozenMetricEvidence(
        nominal, empty, empty, (empty,), (empty,), inventory
    )


@pytest.mark.parametrize("bad", (True, -1, 129 * 1024**2))
def test_inventory_byte_admission_precedes_file_access(tmp_path, monkeypatch, bad):
    row = {"path": "unread.json", "bytes": bad, "sha256": "0" * 64}
    monkeypatch.setattr(
        owner, "file_receipt", lambda *a, **k: pytest.fail("opened invalid inventory")
    )
    with pytest.raises(ValueError, match="bounded receipt"):
        owner._inventory(evidence(tmp_path, (row,)))


def test_inventory_rejects_aliases_before_opening_inputs(tmp_path, monkeypatch):
    row = {"path": "same.json", "bytes": 1, "sha256": "0" * 64}
    monkeypatch.setattr(
        owner, "file_receipt", lambda *a, **k: pytest.fail("opened duplicate inventory")
    )
    with pytest.raises(ValueError, match="duplicate or aliased"):
        owner._inventory(evidence(tmp_path, (row, dict(row, path="SAME.json"))))


def test_inventory_rejects_changed_bytes_before_producer_metadata(tmp_path):
    path = tmp_path / "changed.json"
    path.write_bytes(b"first")
    row = file_receipt(path, label=path.name)
    path.write_bytes(b"other")
    with pytest.raises(ValueError, match="frozen input receipt differs"):
        owner._inventory(evidence(tmp_path, (row,)))


def complete_inventory(root):
    """Small content-association fixture, not a nominal trajectory certificate."""
    for name in (owner.DIRECT, owner.CUT, owner.RETENTION):
        (root / name).mkdir(parents=True)
    prior = root / "prior-source.json"
    write_json_once(prior, {"constructed": True})

    def row(path, base=root):
        return file_receipt(path, label=path.relative_to(base).as_posix())

    payload = b"# constructed producer source\n"
    for folder in (owner.DIRECT, owner.CUT):
        base = root / folder
        archive = base / "instrument-source.zip"
        with ZipFile(archive, "x") as zip_file:
            zip_file.writestr("producer.py", payload)
        write_json_once(
            base / "instrument-source.json",
            {
                "files": {"producer.py": sha256(payload).hexdigest()},
                "archive": row(archive, base),
            },
        )
        for name in ("result.json", "protocol.json", "attempt.json"):
            write_json_once(base / name, {"constructed": True})
        (base / "flake8.log").write_bytes(b"")
        if folder == owner.CUT:
            (base / "compare_connected.py").write_bytes(payload)
        write_json_once(
            base / "final-state.json",
            {
                "evidence": tuple(row(path, base) for path in sorted(base.iterdir())),
            },
        )
    base = root / owner.RETENTION
    for name in ("source.json", "protocol.json", "result.json", "attempt.json"):
        write_json_once(base / name, {"constructed": True})
    with ZipFile(base / "source.zip", "x") as zip_file:
        zip_file.writestr("producer.py", payload)
    write_json_once(
        base / "source-manifest.json",
        {
            "schema": "tnfr.source-manifest.v1",
            "files": (
                {
                    "path": "producer.py",
                    "bytes": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                },
            ),
        },
    )
    write_json_once(
        base / "freeze.json",
        {
            "schema": "tnfr.maintained-sha-retention-freeze.v1",
            "evaluation_status_at_freeze": "not_evaluated",
            "artifacts": tuple(
                row(base / name, base)
                for name in (
                    "source.json",
                    "protocol.json",
                    "source-manifest.json",
                    "source.zip",
                )
            ),
            "prior_artifacts": (row(prior),),
        },
    )
    direct, cut = root / owner.DIRECT, root / owner.CUT
    nominal_evidence = FrozenNominalEvidence(
        root,
        row(direct / "result.json"),
        row(direct / "final-state.json"),
        row(cut / "result.json"),
        row(cut / "final-state.json"),
        row(cut / "compare_connected.py"),
    )
    return owner.FrozenMetricEvidence(
        nominal_evidence,
        row(base / "result.json", base),
        row(base / "source.json", base),
        tuple(
            row(base / name, base)
            for name in ("protocol.json", "attempt.json", "freeze.json")
        ),
        tuple(row(cut / name, cut) for name in ("protocol.json", "attempt.json")),
        tuple(row(path) for path in sorted(root.rglob("*")) if path.is_file()),
    )


def test_complete_inventory_checks_archived_sources_and_closed_dependency_links(
    tmp_path,
):
    prepared = complete_inventory(tmp_path)
    actual = owner._inventory(prepared)
    assert actual == {row["path"]: row for row in prepared.inventory}
    removed = owner.DIRECT + "/protocol.json"
    missing = replace(
        prepared,
        inventory=tuple(row for row in prepared.inventory if row["path"] != removed),
    )
    with pytest.raises(ValueError, match="omits or changes dependency"):
        owner._inventory(missing)


def test_rehashed_source_archive_cannot_replace_frozen_member_digests(tmp_path):
    prepared = complete_inventory(tmp_path)
    path = tmp_path / owner.RETENTION / "source.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("producer.py", b"changed source\n")
    # Updating the prospective outer hash does not change the producer freeze.
    replacement = file_receipt(path, label=path.relative_to(tmp_path).as_posix())
    altered = replace(
        prepared,
        inventory=tuple(
            replacement if row["path"] == replacement["path"] else row
            for row in prepared.inventory
        ),
    )
    with pytest.raises(ValueError, match="omits or changes dependency"):
        owner._inventory(altered)


def test_zero_length_trace_is_admitted_but_its_hash_and_inventory_remain_required(
    tmp_path,
):
    prepared = complete_inventory(tmp_path)
    owner._inventory(prepared)
    label = owner.DIRECT + "/flake8.log"
    row = next(row for row in prepared.inventory if row["path"] == label)
    assert row["bytes"] == 0 and row["sha256"] == sha256(b"").hexdigest()
    (tmp_path / label).write_bytes(b"changed")
    with pytest.raises(ValueError):
        owner._inventory(prepared)


@dataclass
class _Result:
    source_receipts: tuple = ()


@pytest.mark.parametrize(
    "branch,folder", (("connected", owner.RETENTION), ("cut_host", owner.CUT))
)
def test_every_phase_input_is_bound_to_the_final_reread_inventory(
    monkeypatch, branch, folder
):
    monkeypatch.setattr(owner, "require_phase_evidence", lambda value: value)
    receipt = PhaseReceipt("steps/000.zip", 120, "a" * 64)
    evidence_branch = NS(branch=branch, receipts=(receipt,))
    path = folder + "/steps/000.zip"
    row = {"path": path, "bytes": 120, "sha256": "a" * 64}
    owner._bind_phase_inventory({path: row}, evidence_branch)
    for inventory in ({}, {path: dict(row, sha256="b" * 64)}):
        with pytest.raises(ValueError, match="phase input"):
            owner._bind_phase_inventory(inventory, evidence_branch)


@pytest.mark.parametrize("fail_branch", (None, "connected", "cut_host"))
def test_public_pipeline_admits_both_complete_proofs_before_widths(
    monkeypatch, tmp_path, fail_branch
):
    import networkx as nx

    graph = nx.complete_graph(11)
    graph.remove_edges_from(((0, 1), (0, 2)))
    graph.add_nodes_from(range(11, 21))
    graph.graph["_t"] = float(Q(7) + Q(1, 2**19))
    admitted = NS(graph=graph, repository_root=tmp_path, receipts=())
    stages = []
    monkeypatch.setattr(owner, "_inventory", lambda _: stages.append("inventory"))
    monkeypatch.setattr(
        owner,
        "validate_live_sha_successor",
        lambda _: (stages.append("fresh") or admitted),
    )
    monkeypatch.setattr(owner, "_signature", lambda _: "synthetic source")
    monkeypatch.setattr(
        owner,
        "_capture_cut_source",
        lambda *a, **k: (stages.append("capture") or source()),
    )
    monkeypatch.setattr(
        owner,
        "_verify_nominal_evidence",
        lambda *a: (stages.append("nominal") or nominal()),
    )

    def phase_reader(folder, **kwargs):
        branch = kwargs["branch"]
        stages.append(branch)
        assert len(kwargs["initial_box"]) == (42 if branch == "connected" else 40)
        if branch == fail_branch:
            raise ValueError("constructed phase rejection")
        return NS(branch=branch)

    monkeypatch.setattr(owner, "load_phase_branch", phase_reader)
    monkeypatch.setattr(
        owner, "_bind_phase_inventory", lambda *a: stages.append("phase_inventory_link")
    )
    monkeypatch.setattr(owner, "_evidence_record", lambda *a: {})
    monkeypatch.setattr(
        owner,
        "_match_phase_observations",
        lambda *a: stages.append("phase_observation_link"),
    )
    monkeypatch.setattr(
        owner,
        "_bound_metric_response",
        lambda *a: (stages.append("widths_and_metric") or _Result()),
    )
    if fail_branch:
        with pytest.raises(ValueError, match="phase rejection"):
            owner.audit_sha_metric_response(admitted, evidence=evidence(tmp_path))
        assert "widths_and_metric" not in stages
    else:
        owner.audit_sha_metric_response(admitted, evidence=evidence(tmp_path))
        assert stages == [
            "inventory",
            "fresh",
            "capture",
            "nominal",
            "connected",
            "cut_host",
            "phase_inventory_link",
            "phase_observation_link",
            "phase_inventory_link",
            "phase_observation_link",
            "widths_and_metric",
            "inventory",
        ]
