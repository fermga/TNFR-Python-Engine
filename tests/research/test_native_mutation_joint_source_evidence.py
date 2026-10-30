"""Read-only checks of the frozen 24-run joint-source attempt.

Neither the proposer nor the exact point checker is replayed. Byte/trace
associations establish retained-record consistency, not authentication.
Independent ten-constituent algebra reconstructs only the 24 retained points
and proves two failed predicates without evaluating any controller resultant.
The negative Mutation branch remains hypothetical when its sign gate fails.
"""

import gzip
import hashlib
import runpy
from collections import Counter
from dataclasses import asdict
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path

import pytest

from tests.research import test_native_mutation_feasibility as prior
from tnfr.mathematics import _phase_resultant_chamber as trigonometry
from tnfr.research import _native_mutation_feasibility as shared
from tnfr.research import _native_mutation_point as point
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

no_execution = prior.no_execution
calculus_guard = prior.calculus_guard
renewal_guard = prior.renewal_guard

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/native_mutation_joint_source"
MAX_BYTES = 32 * 1024**2
PAIRS = tuple(product(prior.BRANCHES, ("stable", "transition")))
RUNS = tuple((pair, start) for pair in PAIRS for start in range(4))


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=MAX_BYTES))
    )


def _dyadic(value):
    scale = 1 << 40
    return Q(round(Q(value) * scale), scale)


@pytest.fixture(scope="module")
def evidence_guard(renewal_guard):
    import scipy.optimize
    from scipy.stats import qmc

    _, forbidden = renewal_guard
    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (point, "check_native_mutation_point"),
            (point, "certified_cosine_bounds"),
            (trigonometry, "certified_cosine_bounds"),
            (shared, "compile_native_mutation_slice"),
            (shared, "_prepare"),
            (shared, "_cell"),
            (scipy.optimize, "minimize"),
            (qmc, "Sobol"),
            (runpy, "run_path"),
            (runpy, "run_module"),
            (prior.guards.domain.gain, "_source"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def evidence(evidence_guard):
    paths = tuple(sorted(EVIDENCE.iterdir()))
    before = tuple(file_receipt(path, max_bytes=MAX_BYTES) for path in paths)
    records = {path.name: _read(path.name) for path in paths if path.suffix == ".json"}
    yield records
    assert tuple(file_receipt(path, max_bytes=MAX_BYTES) for path in paths) == before


def test_frozen_manifests_and_archived_source_are_byte_associations(evidence):
    manifest = evidence["manifest.json"]
    assert manifest["schema"] == "native_mutation_joint_source_finite_search_v1"
    entries = manifest["files"]
    assert len({entry["path"] for entry in entries}) == len(entries) == 10
    assert {entry["path"] for entry in entries} | {"manifest.json"} == {
        path.name for path in EVIDENCE.iterdir()
    }
    for entry in entries:
        name = validate_artifact_path(entry["path"])
        assert file_receipt(EVIDENCE / name, label=name, max_bytes=MAX_BYTES) == entry
    hashes = {entry["path"]: entry["sha256"] for entry in entries}
    members = evidence["source-manifest.json"]["members"]
    verify_archive_members(EVIDENCE / "source.zip", members, max_bytes=64 * 1024**2)
    proposal = evidence["proposal-metadata.json"]
    exact = evidence["exact-metadata.json"]
    assert (
        proposal["protocol_sha256"]
        == exact["protocol_sha256"]
        == hashes["protocol.txt"]
    )
    assert (
        proposal["driver_sha256"]
        == exact["proposer_sha256"]
        == members["campaign/propose.py"]
    )
    assert exact["driver_sha256"] == members["campaign/certify.py"]
    assert exact["proposal_results_sha256"] == hashes["proposal-results.json"]
    assert (
        exact["checker_sha256"]
        == members["src/tnfr/research/_native_mutation_point.py"]
    )
    assert (
        exact["cosine_sha256"]
        == members["src/tnfr/mathematics/_phase_resultant_chamber.py"]
    )
    for name, digest in proposal["sources"].items():
        assert members[name.replace("\\", "/")] == digest
    assert exact["cosine_terms"] == 12 and exact["cosine_bits"] == 128
    assert proposal["seed"] == 20261018
    assert proposal["pairs"] == PAIRS
    assert (proposal["maxiter"], proposal["maxeval"]) == (400, 5000)
    assert (proposal["ftol"], proposal["eps"], proposal["strict_proposal_margin"]) == (
        1e-12,
        1e-7,
        1e-6,
    )


@pytest.fixture(scope="module")
def trace_summary(evidence):
    """Stream once; no expanded trace or objective callable is retained."""
    metadata = evidence["proposal-metadata.json"]
    counts, minima = Counter(), {}
    digest, size, rows, current_run = hashlib.sha256(), 0, 0, -1
    positions = {key: index for index, key in enumerate(RUNS)}
    with gzip.open(EVIDENCE / "evaluations.jsonl.gz", "rb") as stream:
        while True:
            raw = stream.readline(min(65536, MAX_BYTES - size + 1))
            if not raw:
                break
            size += len(raw)
            assert size <= MAX_BYTES and raw.endswith(b"\n")
            digest.update(raw)
            row = decode_exact_tree(json_loads(raw))
            assert set(row) == {"pair", "start", "evaluation", "unit", "loss"}
            key = row["pair"], row["start"]
            assert key in positions
            index = positions[key]
            assert index in (current_run, current_run + 1)
            current_run = index
            counts[key] += 1
            assert row["evaluation"] == counts[key] <= metadata["maxeval"]
            assert len(row["unit"]) == 13 and all(0 <= x <= 1 for x in row["unit"])
            assert row["loss"] >= 0
            if counts[key] == 1:
                assert row["unit"] == metadata["starts"][row["start"]]
            if key not in minima or row["loss"] < minima[key]["loss"]:
                minima[key] = row
            rows += 1
    return {
        "bytes": size,
        "sha256": digest.hexdigest(),
        "rows": rows,
        "counts": counts,
        "minima": minima,
    }


def test_every_trace_row_preserves_budget_and_first_minimum_selection(
    evidence, trace_summary
):
    declared = evidence["manifest.json"]["evaluation_trace_uncompressed"]
    assert {name: trace_summary[name] for name in declared} == declared
    assert trace_summary["rows"] == 71204
    proposals = evidence["proposal-results.json"]
    assert tuple((row["pair"], row["start"]) for row in proposals) == RUNS
    assert len(proposals) == 24
    for row in proposals:
        key = row["pair"], row["start"]
        minimum = trace_summary["minima"][key]
        assert row["evaluations"] == trace_summary["counts"][key]
        assert 0 <= row["iterations"] <= 400
        for name in ("loss", "unit", "evaluation"):
            assert row["best"][name] == minimum[name]
        if row["termination"]["status"] == "budget_exhausted":
            assert row["evaluations"] == 5000 and not row["termination"]["success"]
    assert (
        sum(row["termination"]["status"] == "budget_exhausted" for row in proposals)
        == 2
    )
    assert sum(row["evaluations"] for row in proposals) == 71204


def test_selected_physical_points_have_one_declared_dyadic_projection(evidence):
    metadata = evidence["proposal-metadata.json"]
    records = evidence["exact-results.json"]
    proposals = evidence["proposal-results.json"]
    assert len(records) == len(proposals) == 24
    law, premises = shared._owned_law()
    assert decode_exact_tree(asdict(law)) == metadata["law"]
    assert decode_exact_tree(asdict(premises)) == metadata["premises"]
    lower, upper = metadata["bounds"]["lower"], metadata["bounds"]["upper"]
    for proposed, exact in zip(proposals, records, strict=True):
        assert (exact["pair"], exact["start"]) == (proposed["pair"], proposed["start"])
        assert exact["proposal_loss"] == proposed["best"]["loss"]
        values = tuple(
            lo + (hi - lo) * unit
            for lo, hi, unit in zip(lower, upper, proposed["best"]["unit"], strict=True)
        )
        physical = proposed["best"]["state"]
        assert physical["capacities"] == values[:5]
        assert physical["phases"] == (0.0, *values[5:9])
        assert physical["pressure_coordinates"] == values[9:]
        assert exact["spec"]["capacities"] == tuple(
            map(_dyadic, physical["capacities"])
        )
        assert exact["spec"]["phases"] == tuple(map(_dyadic, physical["phases"]))
        assert exact["pressure_coordinates"] == tuple(
            map(_dyadic, physical["pressure_coordinates"])
        )
        assert (exact["spec"]["first_branch"], exact["spec"]["second_branch"]) == exact[
            "pair"
        ]
        report = exact["report"]
        assert report["spec"] == exact["spec"]
        assert report["pressure_coordinates"] == exact["pressure_coordinates"]
        assert (
            report["law"] == metadata["law"]
            and report["premises"] == metadata["premises"]
        )


def test_each_retained_point_fails_negative_pressure_and_complete_renewal(evidence):
    law = prior._defaults()
    owned_law, _ = shared._owned_law()
    statuses = Counter()
    for record in evidence["exact-results.json"]:
        raw_spec = record["spec"]
        capacities = shared._five(raw_spec["capacities"], "archived capacities")
        phases = shared._five(raw_spec["phases"], "archived phase lift")
        assert all(0 < value <= owned_law.capacity_rails[1] for value in capacities)
        assert phases[0] == 0
        first_branch, second_branch = (
            raw_spec["first_branch"],
            raw_spec["second_branch"],
        )
        assert type(first_branch) is str and first_branch in prior.BRANCHES
        assert type(second_branch) is str and second_branch in ("stable", "transition")
        raw_coordinates = record["pressure_coordinates"]
        assert type(raw_coordinates) is tuple and len(raw_coordinates) == 4
        coordinates = tuple(map(exact_record, raw_coordinates))
        spec = shared.NativeMutationSlice(
            capacities, phases, first_branch, second_branch
        )
        fine = prior._fine_coefficients(spec, (*coordinates, Q(0)), law)
        p, q = (values[::2] for values in fine["pressure"])
        first, mutated, terminal = (values[::2] for values in fine["phase"][1:])
        diameter1, diameter2 = max(first) - min(first), max(terminal) - min(terminal)
        # These exact inequalities establish rejection independently of
        # archived status strings or the point checker's implementation.
        assert q[0] >= 0
        assert diameter2 <= diameter1
        report = record["report"]
        geometry = {check["label"]: check for check in report["parameter_checks"]}
        for label, lhs, rhs, relation in (
            ("source_lift_width", max(phases) - min(phases), Q(3), "<="),
            ("first_common_arc", max(first) - min(first), prior.PI, "<"),
            ("mutation_common_arc", max(mutated) - min(mutated), prior.PI, "<"),
        ):
            check = geometry[label]
            assert (check["lhs"], check["rhs"], check["relation"]) == (
                lhs,
                rhs,
                relation,
            )
        assert report["pressure"] == p and report["successor_pressure"] == q
        assert sum(p) == sum(q) == 0
        assert report["first_phase"] == first
        assert report["mutation_phase"] == mutated
        assert report["terminal_phase"] == terminal
        offsets = tuple(values[::2] for values in fine["forms"])
        flat = tuple(value for boundary in offsets for value in boundary)
        interval = -1 - min(flat), 1 - max(flat)
        eta = sum(interval) / 2
        forms = tuple(tuple(value + eta for value in boundary) for boundary in offsets)
        assert report["form_offsets"] == offsets
        assert report["common_offset_interval"] == interval
        assert report["common_offset"] == eta and report["form_boundaries"] == forms
        times = (Q(1), Q(17, 16), Q(9, 8))
        histories = tuple(
            tuple((t, forms[j][i]) for j, t in enumerate(times)) for i in range(5)
        )
        assert report["temporal_histories"] == histories
        assert report["first_secants"] == tuple(
            (forms[1][i] - forms[0][i]) / prior.DT for i in range(5)
        )
        failures = {row["label"]: row for row in record["failed_details"]}
        assert (
            failures["point.successor_pressure"]["lhs"],
            failures["point.successor_pressure"]["rhs"],
            failures["point.successor_pressure"]["relation"],
        ) == (q[0], 0, "<")
        assert (
            failures["complete_renewal"]["lhs"],
            failures["complete_renewal"]["rhs"],
            failures["complete_renewal"]["relation"],
        ) == (diameter1, diameter2, "<")
        assert {"point.successor_pressure", "complete_renewal"} <= set(
            report["failed_checks"]
        )
        assert report["selected_cell_key"] is None and not report["row_checks"]
        for index, controller in enumerate(report["controller_checks"]):
            phases = spec.phases if index == 0 else mutated
            obligation = controller["obligation"]
            assert obligation["phase_differences"] == tuple(
                phases[i] - phases[j] for i, j in combinations(range(5), 2)
            )
            assert obligation["outgoing_gains"] == fine["gains"][index]
            assert (controller["terms"], controller["bits"]) == (12, 128)
        assert record["status"] == report["status"] == "rejected"
        statuses[record["status"]] += 1
    summary = evidence["result-summary.json"]
    assert statuses == summary["status_counts"] == {"rejected": 24}
    assert not summary["admitted"]
    assert summary["total_objective_evaluations"] == 71204
    assert summary["budget_exhaustions"] == 2
    assert summary["numerical_success_is_not_admission"] is True
