"""Read-only checks of the frozen IL/OZ response and its source association.

No producer is imported or invoked. Captured primitive state is re-admitted
through the weighted-law reader, and edge balances are independently rebuilt.
Hashes establish byte association, not chronology or execution authentication.
"""

from copy import deepcopy
from fractions import Fraction as Q
from math import fsum, isfinite
from pathlib import Path
from zipfile import ZipFile

import networkx as nx
import pytest

from tnfr._exact_time import finite_represented_real
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin
from tnfr.operators._epi_domain import require_real_scalar_epi
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "docs/assets/collective_pressure_handoff"
EVIDENCE = ORIGINAL / "correction_v2"
JSON_LIMIT = 32 * 1024**2
ARCHIVE_LIMIT = 32 * 1024**2
BUDGET = Q(1, 2**46)
MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
CAPTURES = (
    "inherited",
    "configured",
    "initial",
    "IL_raw",
    "IL_refreshed",
    "IL_completed",
    "OZ_raw",
    "OZ_refreshed",
    "OZ_completed",
)


def _read(name, *, directory=EVIDENCE):
    validate_artifact_path(name)
    return decode_exact_tree(
        json_loads(read_bytes_bounded(directory / name, max_bytes=JSON_LIMIT))
    )


def _receipts(rows):
    assert type(rows) is tuple and rows
    result = {}
    for row in rows:
        assert type(row) is dict and set(row) == {"path", "bytes", "sha256"}
        name = validate_artifact_path(row["path"])
        assert name not in result
        assert type(row["bytes"]) is int and 0 <= row["bytes"] <= ARCHIVE_LIMIT
        result[name] = row
    return result


def _interval(value):
    assert type(value) is dict and set(value) == {"type", "lo", "hi"}
    assert value["type"] == "tnfr.mathematics._rational_interval.I"
    lo, hi = exact_record(value["lo"]), exact_record(value["hi"])
    result = I(lo, hi)
    assert (result.lo, result.hi) == (lo, hi)
    return result


def _mapping(value):
    assert value["type"] == "builtins.dict" and set(value) == {"type", "items"}
    items = value["items"]
    result = dict(items)
    assert len(result) == len(items)
    return result


def _items(value):
    assert value["type"] in {"builtins.tuple", "builtins.list"}
    assert set(value) == {"type", "items"}
    return value["items"]


def _value(value):
    """Decode only primitive state containers, never instantiate recorded classes."""
    if value is None or type(value) in (bool, int, str, Q):
        return value
    assert type(value) is dict
    kind = value.get("type")
    if kind == "binary64":
        assert set(value) == {"type", "hex"} and type(value["hex"]) is str
        result = float.fromhex(value["hex"])
        assert isfinite(result) and result.hex() == value["hex"]
        return result
    if kind == "builtins.complex":
        assert set(value) == {"type", "real", "imaginary"}
        real, imaginary = _value(value["real"]), _value(value["imaginary"])
        assert type(real) is type(imaginary) is float
        # Preserve the uniform-real BEPI representation until shared signed
        # admission decides its domain; this is never a magnitude projection.
        return complex(real, imaginary)
    if kind == "builtins.dict":
        return {key: _value(item) for key, item in _mapping(value).items()}
    if kind in {"builtins.list", "builtins.tuple"}:
        return tuple(_value(item) for item in _items(value))
    raise AssertionError(f"unsupported primitive state tag: {kind}")


def _without_pressure(state):
    """Keep tagged bits and every persistent field outside the declared writes."""
    result = deepcopy(_mapping(state))
    graph = _mapping(result["graph"])
    result["graph"] = tuple(
        (key, value)
        for key, value in graph.items()
        if key not in {"_dnfrmax", "_dnfrmax_node"}
    )
    rows = []
    for row in _items(result["nodes"]):
        node, data = _items(row)
        rows.append(
            (
                node,
                tuple((k, v) for k, v in _mapping(data).items() if k not in ALIAS_DNFR),
            )
        )
    result["nodes"] = tuple(rows)
    return result


def _rebuild(capture):
    """Admit physical values from full snapshots, then compare exported fields."""
    state = _mapping(capture["state"])
    metadata = _mapping(state["graph"])
    graph = nx.Graph()
    graph.graph["GAMMA"] = _value(metadata["GAMMA"])
    for row in _items(state["nodes"]):
        node, encoded = _items(row)
        assert type(node) is int
        data = _mapping(encoded)

        def raw(aliases):
            selected = next(key for key in aliases if key in data)
            return _value(data[selected])

        form = require_real_scalar_epi(raw(ALIAS_EPI), operator="evidence", label="EPI")
        phase = finite_represented_real(raw(ALIAS_THETA), "phase")[0]
        capacity = finite_represented_real(raw(ALIAS_VF), "capacity")[0]
        pressure = finite_represented_real(raw(ALIAS_DNFR), "pressure")[0]
        graph.add_node(node, EPI=form, theta=phase, nu_f=capacity, delta_nfr=pressure)
    for row in _items(state["edges"]):
        left, right, encoded = _items(row)
        attributes = _value(encoded)
        graph.add_edge(left, right, **attributes)
    rebuilt = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    recorded = capture["fields"]
    for field in ("nodes", "edges", "degrees"):
        assert recorded[field] == getattr(rebuilt, field)
    for field in ("conductances", "epi", "phase", "capacity", "form_gradient"):
        assert tuple(map(exact_record, recorded[field])) == getattr(rebuilt, field)
    for field in ("pressure", "form_rates", "phase_rates"):
        assert tuple(map(_interval, recorded[field])) == getattr(rebuilt, field)
    assert _interval(recorded["storage"]) == rebuilt.storage
    assert tuple(map(exact_record, capture["stored_pressure"])) == tuple(
        Q(get_attr(graph.nodes[node], ALIAS_DNFR)) for node in rebuilt.nodes
    )
    return rebuilt


@pytest.fixture(scope="module")
def retained():
    names = (
        "protocol.json",
        "source-manifest.json",
        "freeze.json",
        "attempt.json",
        "response.json",
        "report.json",
    )
    paths = tuple(sorted(path for path in ORIGINAL.rglob("*") if path.is_file()))
    receipts = tuple(file_receipt(path, max_bytes=ARCHIVE_LIMIT) for path in paths)
    records = {name: _read(name) for name in names}
    yield records
    assert paths == tuple(
        sorted(path for path in ORIGINAL.rglob("*") if path.is_file())
    )
    assert receipts == tuple(
        file_receipt(path, max_bytes=ARCHIVE_LIMIT) for path in paths
    )


@pytest.fixture(scope="module")
def fields(retained):
    response = retained["response.json"]
    assert response["status"] == "completed", response.get("failure")
    assert tuple(response["captures"]) == CAPTURES
    return {name: _rebuild(response["captures"][name]) for name in CAPTURES}


def _verify_frozen_sources(directory, freeze, manifest):
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    artifacts = _receipts(freeze["artifacts"])
    assert set(artifacts) == {
        "protocol.json",
        "source-manifest.json",
        "source.zip",
        "evaluate.py",
    }
    for name, receipt in artifacts.items():
        assert file_receipt(directory / name, label=name) == receipt
    sources = _receipts(manifest["files"])
    assert {
        "tests/operators/test_collective_pressure_handoff.py",
        "tests/operators/test_collective_organization_contact.py",
        "src/tnfr/physics/relational_sine_support.py",
    } <= set(sources)
    verify_archive_members(
        directory / "source.zip",
        {name: receipt["sha256"] for name, receipt in sources.items()},
        max_bytes=ARCHIVE_LIMIT,
    )
    with ZipFile(directory / "source.zip") as archive:
        assert all(
            archive.getinfo(name).file_size == row["bytes"]
            for name, row in sources.items()
        )


@pytest.mark.parametrize(
    "directory, diagnostic_fragment",
    (
        (ORIGINAL, "TypeError: cannot pickle '_thread.RLock' object"),
        (
            ORIGINAL / "correction_v1",
            "TypeError: unsupported evidence type: builtins.complex",
        ),
    ),
    ids=("original-copy-failure", "first-correction-serialization-failure"),
)
def test_failed_attempt_remains_separate_from_corrective_evidence(
    directory, diagnostic_fragment
):
    _verify_frozen_sources(
        directory,
        _read("freeze.json", directory=directory),
        _read("source-manifest.json", directory=directory),
    )
    attempt = _read("attempt.json", directory=directory)
    assert attempt["freeze_sha256"] == file_receipt(directory / "freeze.json")["sha256"]
    assert attempt["maximum_evaluations"] == 1
    failure = _read("failure.json", directory=directory)
    diagnostic = failure["diagnostic"]
    assert "source_snapshot = _encode(_state(" in diagnostic
    assert diagnostic_fragment in diagnostic
    assert not (directory / "response.json").exists()
    assert not (directory / "report.json").exists()


def test_source_archive_protocol_attempt_and_response_have_matching_bytes(retained):
    _verify_frozen_sources(
        EVIDENCE, retained["freeze.json"], retained["source-manifest.json"]
    )
    prior = _receipts(retained["freeze.json"]["prior_artifacts"])
    prior_names = {
        "protocol.json",
        "evaluate.py",
        "source-manifest.json",
        "source.zip",
        "freeze.json",
        "attempt.json",
        "failure.json",
    }
    assert set(prior) == prior_names | {f"correction_v1/{name}" for name in prior_names}
    for name, receipt in prior.items():
        assert file_receipt(ORIGINAL / name, label=name) == receipt
    freeze_hash = file_receipt(EVIDENCE / "freeze.json")["sha256"]
    attempt, report = retained["attempt.json"], retained["report.json"]
    assert attempt["freeze_sha256"] == report["freeze_sha256"] == freeze_hash
    assert attempt["maximum_evaluations"] == attempt["numeric_threads"] == 1
    assert report["response"] == file_receipt(
        EVIDENCE / "response.json", label="response.json"
    )
    assert report["execution_status"] == retained["response.json"]["status"]
    checks = report["checks"]
    assert len({row["name"] for row in checks}) == len(checks) > 0
    assert report["all_checks_passed"] is all(row["passed"] for row in checks)


def test_retained_preparation_matches_the_frozen_single_word_budget(retained, fields):
    protocol, response = retained["protocol.json"], retained["response.json"]
    original = _read("protocol.json", directory=ORIGINAL)
    for key in (
        "law",
        "execution",
        "arithmetic",
        "prospective_predictions",
        "classification_rules",
    ):
        assert protocol[key] == original[key]
    assert {
        key: value
        for key, value in protocol["preparation"].items()
        if key != "preservation"
    } == {
        key: value
        for key, value in original["preparation"].items()
        if key != "preservation"
    }
    assert protocol["status_at_protocol_creation"] == "not_evaluated"
    assert protocol["preparation"]["arguments"] == (0, 1)
    assert Q(protocol["preparation"]["phase_origin"]) == Q(3, 4096)
    assert protocol["execution"]["word"] == ("coherence", "dissonance")
    assert protocol["execution"]["targets"] == (0, 1)
    assert protocol["execution"]["refreshes"] == (
        "initial_before_IL",
        "after_IL",
        "after_OZ",
    )
    assert Q(protocol["arithmetic"]["absolute_residual_allowance"]) == BUDGET
    settings = response["settings"]
    assert exact_record(settings["absolute_budget"]) == BUDGET
    assert exact_record(settings["phase_coefficient"]) == Q(0.3)
    assert exact_record(settings["OZ_MIN_PROPAGATION"]) == Q(0.05)
    assert settings["oracle_dps"] == 100 and settings["OZ_NOISE_MODE"] is False
    assert settings["OZ_ENABLE_PROPAGATION"] is True
    source = fields["inherited"]
    assert source.nodes == tuple(range(20)) and len(source.edges) == 45
    old = {
        tuple(sorted((offset + 2 * pair + a, offset + 2 * ((pair + 1) % 5) + b)))
        for offset in (0, 10)
        for pair in range(5)
        for a in (0, 1)
        for b in (0, 1)
    }
    new = {(0, 1)} | {(a, b) for a in (0, 1) for b in (10, 11)}
    assert set(source.edges) == old | new
    assert all(
        weight == 1
        for edge, weight in zip(source.edges, source.conductances)
        if edge in old
    )
    assert all(weight >= Q(3, 4) for weight in source.conductances)
    assert source.capacity == (Q(1, 2),) * 2 + (Q(1),) * 18
    assert sum(Q(d) / nu for d, nu in zip(source.degrees, source.capacity)) == 104


def test_refresh_rebuilds_pressure_and_preserves_every_other_captured_field(
    retained, fields
):
    records = retained["response.json"]["refreshes"]
    assert tuple(row["label"] for row in records) == ("initial", "IL", "OZ")
    for row in records:
        before, after = row["before"], row["after"]
        assert _without_pressure(before["state"]) == _without_pressure(after["state"])
        rebuilt = _rebuild(before)
        assert _rebuild(after) == rebuilt
        pressures = tuple(map(exact_record, after["stored_pressure"]))
        assert pressures == tuple(
            finite_represented_real(bound.midpoint, "midpoint")[1]
            for bound in rebuilt.pressure
        )
        for represented, ideal, residual in zip(
            pressures, rebuilt.pressure, row["residuals"]
        ):
            assert _interval(residual) == I(represented) - ideal
            assert _interval(residual).abs_max <= BUDGET
        metadata = _mapping(_mapping(after["state"])["graph"])
        maximum = Q(_value(metadata["_dnfrmax"]))
        witness = _value(metadata["_dnfrmax_node"])
        assert (
            maximum
            == max(map(abs, pressures))
            == abs(pressures[rebuilt.nodes.index(witness)])
        )
        for _, observation in after["acceleration"]:
            assert observation["available"] is False and observation["value"] is None
    assert fields["IL_completed"] == fields["OZ_raw"] == fields["OZ_completed"]


def test_il_phase_budget_remains_unresolved_in_the_frozen_enclosure(retained, fields):
    """Retain the failed certificate; saved point arithmetic is only a cross-check."""
    response = retained["response.json"]
    old, new = fields["initial"], fields["IL_raw"]
    coefficient = exact_record(response["settings"]["phase_coefficient"])
    measured = response["IL"]
    for i in (0, 1):
        change = arg(
            cos(I(new.phase[i] - old.phase[i])), sin(I(new.phase[i] - old.phase[i]))
        )
        defect = change - coefficient * arg(*old.relative_resultant[i])
        assert defect == _interval(measured["phase_defects"][i])
        # Each rigorous box admits zero but also exceeds the frozen allowance.
        # Neither a passed certificate nor a proved violation follows.
        assert defect.contains(0) and defect.abs_max > BUDGET
        oracle = measured["oracle100"][i]
        assert oracle["node"] == i and oracle["dps"] == 100
        point_defect = Q(oracle["represented_defect_decimal"])
        assert defect.contains(point_defect) and abs(point_defect) < BUDGET
    checks = retained["report.json"]["checks"]
    failed = tuple(row for row in checks if not row["passed"])
    assert len(failed) == 1
    assert (
        failed[0]["name"]
        == "test_full_il_phase_action_and_adjacent_edge_work_are_retained"
    )
    assert 'measured["phase_defects"]' in failed[0]["diagnostic"]
    assert retained["report.json"]["all_checks_passed"] is False


def test_il_work_and_identity_follow_from_all_retained_edges(retained, fields):
    old, new = fields["initial"], fields["IL_raw"]
    assert old.epi == new.epi and old.capacity == new.capacity
    assert old.edges == new.edges and old.conductances == new.conductances
    assert all(new.phase[i] == old.phase[i] for i in range(2, 20))
    displacement = tuple(
        arg(cos(I(new.phase[i] - old.phase[i])), sin(I(new.phase[i] - old.phase[i])))
        for i in (0, 1)
    )
    coefficient = exact_record(
        retained["response.json"]["settings"]["phase_coefficient"]
    )
    for change in displacement:
        assert not change.contains(0)
    isolated = I(0)
    for i, change in enumerate(displacement):
        for left, right in old.edges:
            if i in (left, right):
                j = right if left == i else left
                gap = I(old.phase[j] - old.phase[i])
                isolated += cos(gap) - cos(gap - change)
    left, right = displacement
    correction = (
        -4
        * cos(I(old.phase[1] - old.phase[0]) + (right - left) / 2)
        * sin(left / 2)
        * sin(right / 2)
    )
    edge_work = sum(
        (
            cos(I(old.phase[j] - old.phase[i])) - cos(I(new.phase[j] - new.phase[i]))
            for i, j in old.edges
        ),
        I(0),
    )
    work_identity = edge_work - isolated - correction
    assert work_identity.contains(0) and work_identity.abs_max <= BUDGET
    assert edge_work.hi < 0 and (new.storage - old.storage - edge_work).contains(0)
    ideal_changes = {i: coefficient * arg(*old.relative_resultant[i]) for i in (0, 1)}
    ideal_work = sum(
        (
            cos(I(old.phase[j] - old.phase[i]))
            - cos(
                I(old.phase[j] - old.phase[i])
                + ideal_changes.get(j, I(0))
                - ideal_changes.get(i, I(0))
            )
            for i, j in old.edges
        ),
        I(0),
    )
    measured = retained["response.json"]["IL"]
    assert ideal_work == _interval(measured["ideal_work"])
    assert (new.storage - old.storage - ideal_work).abs_max <= 14 * BUDGET
    actual_mean_jump = (
        sum(
            Q(degree) / capacity * arg(cos(I(after - before)), sin(I(after - before)))
            for degree, capacity, before, after in zip(
                old.degrees, old.capacity, old.phase, new.phase
            )
        )
        / 104
    )
    assert actual_mean_jump == _interval(measured["actual_mean_phase_jump"])
    assert (actual_mean_jump - 7 * sum(displacement, I(0)) / 52).abs_max <= BUDGET
    captures = retained["response.json"]["captures"]
    factor = exact_record(retained["response.json"]["settings"]["IL_dnfr_factor"])
    for i, (before, after) in enumerate(
        zip(
            captures["initial"]["stored_pressure"],
            captures["IL_raw"]["stored_pressure"],
        )
    ):
        expected = exact_record(before) * (factor if i in (0, 1) else 1)
        assert abs(exact_record(after) - expected) <= BUDGET
    assert old.phase_rates == new.phase_rates
    assert any(not (b - a).contains(0) for a, b in zip(old.form_rates, new.form_rates))
    alpha = 2 * pi_interval() / 5
    errors = tuple(
        arg(
            cos(I(phase) - (i % 10 // 2) * alpha), sin(I(phase) - (i % 10 // 2) * alpha)
        )
        for i, phase in enumerate(new.phase)
    )
    center = sum(errors, I(0)) / 20
    form_center = sum(new.epi) / 20
    norm = sum(((x - form_center) ** 2 for x in new.epi), Q(0)) + sum(
        ((e - center) ** 2 for e in errors), I(0)
    )
    assert norm.hi < Q(1, 256)
    assert (new.storage - 40 * (1 - cos(alpha))).hi < Q(1, 6400)


def test_oz_projection_and_all_candidate_products_retain_their_separate_roles(
    retained, fields
):
    response = retained["response.json"]
    captures = response["captures"]
    before, raw, after = (
        captures[name] for name in ("IL_completed", "OZ_raw", "OZ_completed")
    )
    assert before["stored_pressure"] == after["stored_pressure"]
    assert raw["stored_pressure"] != before["stored_pressure"]
    assert before["state"] != after["state"]
    assert fields["IL_completed"] == fields["OZ_raw"] == fields["OZ_completed"]
    report = fields["IL_completed"]
    settings = response["settings"]
    threshold = exact_record(settings["OZ_PHASE_THRESHOLD"])
    minimum = exact_record(settings["OZ_MIN_PROPAGATION"])
    factor = exact_record(settings["OZ_dnfr_factor"])
    pressures = tuple(map(exact_record, before["stored_pressure"]))
    local = {}
    branches = response["OZ"]["branches"]
    assert tuple(row["node"] for row in branches) == (0, 1)
    for row in branches:
        node = row["node"]
        amplitude = abs(report.pressure[node])
        branch = "amplification" if abs(pressures[node]) > Q(1e-9) else "near_zero_seed"
        assert row["represented"] == branch
        assert row["ideal"] == (
            "amplification"
            if amplitude.lo > Q(1e-9)
            else "near_zero_seed" if amplitude.hi <= Q(1e-9) else "unresolved"
        )
        assert row["ideal"] == branch
        exact = factor * pressures[node] if branch == "amplification" else Q(0.1)
        assert exact_record(row["local_exact"]) == exact
        local[node] = exact_record(row["local_represented"])
        assert local[node] == Q(float(exact))
        assert exact_record(row["local_residual"]) == local[node] - exact
        assert abs(local[node] - exact) <= BUDGET
    weights = {
        frozenset(edge): weight
        for edge, weight in zip(report.edges, report.conductances)
    }
    incoming = {node: [] for node in report.nodes}
    expected_pairs = {
        (i, j if i == u else u) for i in (0, 1) for u, j in report.edges if i in (u, j)
    }
    candidates = response["OZ"]["candidates"]
    assert {(row["source"], row["neighbor"]) for row in candidates} == expected_pairs
    assert len(candidates) == len(expected_pairs) == 14
    for row in candidates:
        source, neighbor = row["source"], row["neighbor"]
        assert exact_record(row["weight"]) == weights[frozenset((source, neighbor))]
        value = exact_record(row["magnitude"])
        exact_magnitude = abs(local[source] - pressures[source])
        assert value == Q(float(exact_magnitude))
        assert exact_record(row["magnitude_residual"]) == value - exact_magnitude
        assert abs(value - exact_magnitude) <= BUDGET
        gap = exact_record(row["wrapped_gap"])
        ideal_gap = abs(
            arg(
                cos(I(report.phase[source] - report.phase[neighbor])),
                sin(I(report.phase[source] - report.phase[neighbor])),
            )
        )
        assert _interval(row["gap_residual"]) == I(gap) - ideal_gap
        assert (I(gap) - ideal_gap).abs_max <= BUDGET
        assert row["phase_admitted"] is (gap <= threshold)
        if not row["phase_admitted"]:
            assert row["products"] == () and row["accepted"] is False
            continue
        phase_weight = exact_record(row["phase_weight"])
        phase_residual = phase_weight - (1 - gap / threshold)
        assert exact_record(row["phase_weight_residual"]) == phase_residual
        assert abs(phase_residual) <= BUDGET
        assert (
            len(row["products"])
            == len(row["product_residuals"])
            == len(row["nonzero_lost"])
            == 3
        )
        multipliers = (exact_record(row["weight"]), phase_weight, Q(1))
        for multiplier, actual, residual, lost in zip(
            multipliers, row["products"], row["product_residuals"], row["nonzero_lost"]
        ):
            exact = value * multiplier
            actual = exact_record(actual)
            assert actual == Q(float(exact))
            assert exact_record(residual) == actual - exact
            assert abs(actual - exact) <= BUDGET
            assert lost is False and not (exact != 0 and actual == 0)
            value = actual
        assert row["accepted"] is (value != 0 and value >= minimum)
        if row["accepted"]:
            incoming[neighbor].append((source, value))
    merged = tuple(
        Q(
            fsum(
                [
                    float(local.get(node, pressures[node])),
                    *(float(value) for _, value in sorted(incoming[node])),
                ]
            )
        )
        for node in report.nodes
    )
    assert tuple(map(exact_record, response["OZ"]["merged_pressure"])) == merged
    for actual, expected, residual in zip(
        raw["stored_pressure"], merged, response["OZ"]["merge_residual"]
    ):
        assert exact_record(actual) - expected == exact_record(residual)
        assert abs(exact_record(residual)) <= BUDGET
