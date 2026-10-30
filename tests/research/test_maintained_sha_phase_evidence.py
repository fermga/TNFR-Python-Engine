"""Manufactured passive records, never retained TNFR target evidence.

The constant-field Taylor rows below have independent dyadic arithmetic.
Their acceptance checks serialization and inherited-proof contracts; it does
not identify these artificial rows with the held sine differential equation.
"""

import importlib.util
import shutil
import sys
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _sine_flow
from tnfr.research import maintained_sha_motion_source
from tnfr.research import maintained_sha_phase_evidence as owner
from tnfr.research import maintained_sha_retention_evidence
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    file_receipt,
    sha256_bytes,
)
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_dumps, json_loads

STEP = Q(1, 512)
ORDER = 12
MARGIN = Q(1, 64)
METHOD = "direct_source_box_Picard_Taylor_dyadic128_v1"


@pytest.fixture(scope="module", autouse=True)
def no_source_loader_or_differential_producer():
    def forbidden(*args, **kwargs):
        pytest.fail(
            "passive constructed controls cannot invoke a source or differential producer"
        )

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (maintained_sha_motion_source, "load_sha_successor"),
            (maintained_sha_retention_evidence, "load_sha_retention"),
            (_sine_flow, "_held_sine_neighbor_field"),
            (_validated_taylor, "validated_box_taylor_step"),
            (_validated_taylor, "flow_jets"),
            (_validated_taylor, "picard_tube"),
        ):
            patch.setattr(module, name, forbidden)
        yield


def _fixture_builder():
    path = Path(__file__).with_name("phase_evidence_fixtures.py")
    spec = importlib.util.spec_from_file_location(
        "phase_evidence_control_fixtures", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_phase_fixture


@pytest.fixture(scope="module", params=("connected", "cut_host"))
def archived_branch(request, tmp_path_factory):
    folder = tmp_path_factory.mktemp(f"passive-{request.param}")
    return _fixture_builder()(folder, branch=request.param)


def _load(bundle):
    return owner.load_phase_branch(bundle["evidence_dir"], **bundle["loader_kwargs"])


def _clone(bundle, folder):
    shutil.copytree(bundle["evidence_dir"], folder)
    copied = deepcopy(bundle)
    copied["evidence_dir"] = folder
    return copied


def _read(path):
    return decode_exact_tree(json_loads(path.read_bytes()))


def _rewrite(path, value):
    path.write_text(
        json_dumps(encode_exact_tree(value), allow_nan=False) + "\n", encoding="utf-8"
    )


def _refresh_result_receipt(bundle):
    receipt = bundle["loader_kwargs"]["result_receipt"]
    path = bundle["evidence_dir"] / receipt["path"]
    bundle["loader_kwargs"]["result_receipt"] = file_receipt(
        path, label=receipt["path"]
    )


def _rewrite_first_step(bundle, mutate, *, member_override=None):
    folder = bundle["evidence_dir"]
    result_path = folder / bundle["loader_kwargs"]["result_receipt"]["path"]
    result = _read(result_path)
    connected = bundle["loader_kwargs"]["branch"] == "connected"
    row = result["records" if connected else "step_records"][0]
    path = folder / "steps" / "000.zip"
    with ZipFile(path) as archive:
        original_member = archive.namelist()[0]
        packet = decode_exact_tree(json_loads(archive.read(original_member)))
    mutate(packet)
    raw = (json_dumps(encode_exact_tree(packet), allow_nan=False) + "\n").encode()
    member = original_member if member_override is None else member_override
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member, raw)
    receipt = file_receipt(path, label="steps/000.zip")
    if connected:
        row["artifact"] = receipt
        row["member"] = {"path": member, "bytes": len(raw), "sha256": sha256_bytes(raw)}
    else:
        row["archive"] = receipt
        row["member"] = member
        row["expanded_bytes"] = len(raw)
        row["expanded_sha256"] = sha256_bytes(raw)
    _rewrite(result_path, result)
    _refresh_result_receipt(bundle)


def _interval(value):
    value = I.coerce(value)
    return {"lo": value.lo, "hi": value.hi}


def _initial(branch):
    count = 21 if branch == "connected" else 20
    # Preserve a nonprincipal common lift; wrapping these coordinates is wrong.
    return tuple(I(Q(i - 4, 32)) for i in range(count)) + tuple(
        I(7 + Q(i, 64)) for i in range(count)
    )


def _record(branch, *, index=0, initial=None):
    initial = _initial(branch) if initial is None else initial
    rates = tuple(Q((-1) ** i, 16) for i in range(len(initial)))
    increments = tuple(rate * STEP for rate in rates)
    endpoint = tuple(
        I(value.lo + change, value.hi + change)
        for value, change in zip(initial, increments, strict=True)
    )
    tube = tuple(
        I(
            value.lo + min(Q(0), change) - MARGIN,
            value.hi + max(Q(0), change) + MARGIN,
        )
        for value, change in zip(initial, increments, strict=True)
    )
    step = {
        "time": index * STEP,
        "duration": STEP,
        "order": ORDER,
        "initial_box": tuple(map(_interval, initial)),
        "tube": tuple(map(_interval, tube)),
        "series": tuple(
            (_interval(value), _interval(rate)) + (_interval(0),) * (ORDER - 1)
            for value, rate in zip(initial, rates, strict=True)
        ),
        "local_remainder_bounds": (_interval(0),) * len(initial),
        "increment": tuple(map(_interval, increments)),
        "endpoint": tuple(map(_interval, endpoint)),
        "picard_interior_margin": MARGIN,
        "domain_lower_bounds": (Q(1),),
        "method": METHOD,
    }
    record = {"index": index, "time": index * STEP, "step": step}
    if branch == "connected":
        record.update(
            tube_principal_margins=(Q(1),) * 53,
            combined_principal_margins=(Q(1),) * 53,
        )
    else:
        record.update(
            failed_tube=None,
            verification={
                "tube_rates": tuple(map(_interval, rates)),
                "picard_interior_margin": MARGIN,
                "solution_order": ORDER,
                "remainder_order": ORDER + 1,
                "series_verified": True,
                "remainder_verified": True,
                "arithmetic_verified": True,
            },
            observation={},
            accepted=True,
            reason=None,
        )
    return record, initial, tube, endpoint


def _audit(record, initial, branch, *, index=0):
    return owner._audit_step(record, index=index, initial=initial, branch=branch)


@pytest.mark.parametrize("branch", ("connected", "cut_host"))
def test_constant_field_rows_reconstruct_the_full_unwrapped_tube_without_mutation(
    branch,
):
    record, initial, tube, endpoint = _record(branch)
    before = deepcopy(record)
    report = _audit(record, initial, branch)
    assert record == before
    assert (report.index, report.time, report.duration) == (0, Q(0), STEP)
    assert report.initial == initial
    assert report.tube == tube and report.endpoint == endpoint
    assert report.phases == tube[len(initial) // 2 :]
    assert report.phases[0].lo > 6
    assert report.phases != endpoint[len(initial) // 2 :]
    assert report.producer_premises


def test_connected_picard_margin_is_inherited_while_cut_saved_rates_are_reconstructed():
    connected, initial, _, _ = _record("connected")
    connected["step"]["picard_interior_margin"] = MARGIN / 2
    inherited = _audit(connected, initial, "connected")
    cut, initial_cut, _, _ = _record("cut_host")
    checked = _audit(cut, initial_cut, "cut_host")
    assert inherited.picard_status != checked.picard_status
    assert "inherit" in inherited.picard_status.lower()
    assert "derivative" in str(inherited.producer_premises).lower()
    assert "derivative" in str(checked.producer_premises).lower()
    # Rehashing cannot repair this disagreement with the recorded full image.
    cut["step"]["picard_interior_margin"] = MARGIN / 2
    cut["verification"]["picard_interior_margin"] = MARGIN / 2
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(cut, initial_cut, "cut_host")


@pytest.mark.parametrize("branch", ("connected", "cut_host"))
def test_next_cell_must_start_at_the_exact_reconstructed_endpoint(branch):
    first, initial, _, endpoint = _record(branch)
    first_report = _audit(first, initial, branch)
    second, _, _, expected = _record(branch, index=1, initial=endpoint)
    second_report = _audit(second, first_report.endpoint, branch, index=1)
    assert second_report.endpoint == expected
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(second, initial, branch, index=1)


@pytest.mark.parametrize("branch", ("connected", "cut_host"))
@pytest.mark.parametrize(
    "location,key,value",
    (
        ("record", "index", False),
        ("record", "index", 1),
        ("record", "time", False),
        ("record", "time", STEP),
        ("step", "time", 0.0),
        ("step", "duration", STEP * 2),
        ("step", "duration", True),
        ("step", "order", True),
        ("step", "order", ORDER - 1),
        ("step", "method", "other-method"),
        ("step", "domain_lower_bounds", (Q(0),)),
        ("step", "picard_interior_margin", Q(0)),
    ),
)
def test_step_clock_method_and_domain_are_strict_primitives(
    branch, location, key, value
):
    record, initial, _, _ = _record(branch)
    target = record if location == "record" else record["step"]
    target[key] = value
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, branch)


@pytest.mark.parametrize("branch", ("connected", "cut_host"))
@pytest.mark.parametrize(
    "field",
    (
        "initial_box",
        "tube",
        "series",
        "local_remainder_bounds",
        "increment",
        "endpoint",
    ),
)
def test_every_state_and_taylor_coordinate_is_required(branch, field):
    record, initial, _, _ = _record(branch)
    record["step"][field] = record["step"][field][:-1]
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, branch)


@pytest.mark.parametrize("branch", ("connected", "cut_host"))
@pytest.mark.parametrize("field", ("initial_box", "tube", "increment", "endpoint"))
def test_changed_intervals_cannot_replace_source_or_arithmetic(branch, field):
    record, initial, _, _ = _record(branch)
    values = list(record["step"][field])
    values[0] = _interval(100)
    record["step"][field] = tuple(values)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, branch)


@pytest.mark.parametrize(
    "value",
    (
        {"lo": False, "hi": Q(0)},
        {"lo": 0.0, "hi": Q(0)},
        {"lo": Q(1), "hi": Q(0)},
        {"lo": Q(1, 3), "hi": Q(1, 3)},
        {"lo": Q(0), "hi": Q(0), "extra": 0},
    ),
)
def test_interval_endpoints_are_exact_canonical_and_not_coerced(value):
    record, initial, _, _ = _record("connected")
    record["step"]["tube"] = (value,) + record["step"]["tube"][1:]
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, "connected")


@pytest.mark.parametrize("branch", ("connected", "cut_host"))
@pytest.mark.parametrize("location", ("record", "step"))
def test_consumed_records_reject_unexpected_schema_fields(branch, location):
    record, initial, _, _ = _record(branch)
    target = record if location == "record" else record["step"]
    target["invented_scope"] = True
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, branch)


@pytest.mark.parametrize(
    "field,value",
    (
        ("accepted", False),
        ("accepted", 1),
        ("reason", "failed"),
        ("failed_tube", ()),
    ),
)
def test_cut_rejected_attempt_never_becomes_an_accepted_phase_cell(field, value):
    record, initial, _, _ = _record("cut_host")
    record[field] = value
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, "cut_host")


@pytest.mark.parametrize(
    "field,value",
    (
        ("solution_order", 11),
        ("remainder_order", 12),
        ("series_verified", 1),
        ("remainder_verified", False),
        ("arithmetic_verified", False),
    ),
)
def test_cut_verification_status_and_orders_cannot_be_forged_by_truthiness(
    field, value
):
    record, initial, _, _ = _record("cut_host")
    record["verification"][field] = value
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, "cut_host")


def test_cut_saved_rate_image_and_recorded_interior_must_match_exactly():
    record, initial, _, _ = _record("cut_host")
    rates = list(record["verification"]["tube_rates"])
    rates[0] = _interval(2)
    record["verification"]["tube_rates"] = tuple(rates)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, "cut_host")


@pytest.mark.parametrize(
    "field", ("tube_principal_margins", "combined_principal_margins")
)
def test_connected_original_branch_metadata_keeps_complete_support(field):
    record, initial, _, _ = _record("connected")
    record[field] = record[field][:-1]
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(record, initial, "connected")


def test_taylor_source_coefficient_and_declared_remainder_are_consumed():
    for field in ("series", "local_remainder_bounds"):
        record, initial, _, _ = _record("connected")
        values = list(record["step"][field])
        if field == "series":
            values[0] = (_interval(0),) + values[0][1:]
        else:
            values[0] = _interval(Q(1, 4))
        record["step"][field] = tuple(values)
        with pytest.raises((TypeError, ValueError, ArithmeticError)):
            _audit(record, initial, "connected")


@pytest.mark.parametrize("branch", ("connected", "cut_host"))
def test_arithmetic_reconstruction_uses_the_shared_owner_without_a_field_callback(
    branch, monkeypatch
):
    original = owner.reconstruct_box_taylor_arithmetic
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "reconstruct_box_taylor_arithmetic", record)
    row, initial, tube, endpoint = _record(branch)
    report = _audit(row, initial, branch)
    assert report.endpoint == endpoint
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == initial and args[1] == tube
    assert args[4] == STEP and kwargs == {"order": ORDER}


@pytest.mark.parametrize("branch", ("connected", "cut_host"))
def test_reconstruction_failure_does_not_promote_cached_endpoint(branch, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise ArithmeticError("constructed Taylor inconsistency")

    monkeypatch.setattr(owner, "reconstruct_box_taylor_arithmetic", forbidden)
    row, initial, _, _ = _record(branch)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _audit(row, initial, branch)
    assert len(calls) == 1


def test_complete_constructed_archives_return_every_original_whole_cell(
    archived_branch,
):
    report = _load(archived_branch)
    assert len(report.phase_cells) == 256
    node_count = 21 if archived_branch["loader_kwargs"]["branch"] == "connected" else 20
    assert all(len(cell) == node_count for cell in report.phase_cells)
    assert all(lo <= hi for cell in report.phase_cells for lo, hi in cell)
    assert "constant" in archived_branch["producer_scope"].lower()
    assert report.endpoint == archived_branch["expected_endpoint"]
    for value, rate, endpoint in zip(
        report.initial_box, archived_branch["rates"], report.endpoint, strict=True
    ):
        assert endpoint == I(value.lo + rate.lo / 2, value.hi + rate.hi / 2)
    assert owner.require_phase_evidence(report) is report
    with pytest.raises(ValueError):
        owner.require_phase_evidence(replace(report))


def test_each_load_rechecks_archive_bytes_instead_of_reusing_a_cached_pass(
    archived_branch, tmp_path
):
    bundle = _clone(archived_branch, tmp_path / "fresh-read")
    _load(bundle)
    path = bundle["evidence_dir"] / "steps" / "000.zip"
    path.write_bytes(path.read_bytes() + b"different retained bytes")
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


def test_source_order_and_initial_state_are_independent_admission_inputs(
    archived_branch,
):
    for field in ("node_order", "initial_box"):
        bundle = deepcopy(archived_branch)
        values = list(bundle["loader_kwargs"][field])
        if field == "node_order":
            values[0], values[1] = values[1], values[0]
        else:
            values[0] = I(values[0].lo + Q(1, 8), values[0].hi + Q(1, 8))
        bundle["loader_kwargs"][field] = tuple(values)
        with pytest.raises((TypeError, ValueError, ArithmeticError)):
            _load(bundle)


def test_extra_archive_cannot_be_hidden_by_a_completed_summary(
    archived_branch, tmp_path
):
    bundle = _clone(archived_branch, tmp_path / "extra-cell")
    folder = bundle["evidence_dir"] / "steps"
    shutil.copyfile(folder / "000.zip", folder / "256.zip")
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


@pytest.mark.parametrize(
    "location,key,value",
    (
        ("record", "time", STEP),
        ("step", "duration", STEP * 2),
        ("step", "order", 11),
        ("step", "method", "forged-but-rehashed"),
    ),
)
def test_rehashed_archives_still_require_the_original_clock_and_method(
    archived_branch, tmp_path, location, key, value
):
    bundle = _clone(archived_branch, tmp_path / "rehashed-step")

    def mutate(packet):
        target = packet if location == "record" else packet["step"]
        target[key] = value

    _rewrite_first_step(bundle, mutate)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


@pytest.mark.parametrize(
    "member", ("step-001.json", "../step-000.json", "unrelated.json")
)
def test_rehashed_zip_member_names_cannot_change_the_declared_sequence(
    archived_branch, tmp_path, member
):
    bundle = _clone(archived_branch, tmp_path / "member-sequence")
    _rewrite_first_step(bundle, lambda packet: None, member_override=member)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


def test_rehashed_duplicate_json_keys_are_rejected_before_they_can_disappear(
    archived_branch, tmp_path
):
    bundle = _clone(archived_branch, tmp_path / "duplicate-json")
    path = bundle["evidence_dir"] / bundle["loader_kwargs"]["result_receipt"]["path"]
    raw = path.read_bytes()
    assert b'"schema":' in raw
    path.write_bytes(raw.replace(b'"schema":', b'"schema":"forged","schema":', 1))
    _refresh_result_receipt(bundle)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


def test_rehashed_nonfinite_metadata_does_not_pass_shared_json_admission(
    archived_branch, tmp_path
):
    bundle = _clone(archived_branch, tmp_path / "nonfinite-json")
    path = bundle["evidence_dir"] / bundle["loader_kwargs"]["result_receipt"]["path"]
    raw = path.read_bytes().rstrip()
    assert raw.endswith(b"}")
    path.write_bytes(raw[:-1] + b',"unused_but_nonfinite":NaN}\n')
    _refresh_result_receipt(bundle)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


def test_rehashed_primitives_cannot_override_independently_bound_source(
    archived_branch, tmp_path
):
    bundle = _clone(archived_branch, tmp_path / "changed-cut-source")
    folder = bundle["evidence_dir"]
    path = folder / bundle["loader_kwargs"]["result_receipt"]["path"]
    result = _read(path)
    if bundle["loader_kwargs"]["branch"] == "cut_host":
        source = result["report"]["preparation"]["source"]
        capacities = list(source["capacity"])
        capacities[0] += Q(1, 8)
        source["capacity"] = tuple(capacities)
    else:
        source_receipt = bundle["loader_kwargs"]["source_receipt"]
        source_path = folder / source_receipt["path"]
        source = _read(source_path)
        graph = restore_recorded_graph_state(source["endpoint"]).graph
        first_node = next(iter(graph))
        graph.nodes[first_node]["nu_f"] += Q(1, 8)
        source["endpoint"] = record_graph_state(graph)
        _rewrite(source_path, source)
        new_source = file_receipt(source_path, label=source_receipt["path"])
        bundle["loader_kwargs"]["source_receipt"] = new_source
        freeze_path = folder / "freeze.json"
        freeze = _read(freeze_path)
        freeze["artifacts"] = tuple(
            new_source if row["path"] == new_source["path"] else row
            for row in freeze["artifacts"]
        )
        _rewrite(freeze_path, freeze)
        attempt_path = folder / "attempt.json"
        attempt = _read(attempt_path)
        attempt["freeze"] = file_receipt(freeze_path, label="freeze.json")
        _rewrite(attempt_path, attempt)
        result["attempt"] = file_receipt(attempt_path, label="attempt.json")
        bundle["loader_kwargs"]["producer_receipts"] = tuple(
            file_receipt(folder / row["path"], label=row["path"])
            for row in bundle["loader_kwargs"]["producer_receipts"]
        )
    _rewrite(path, result)
    _refresh_result_receipt(bundle)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


@pytest.mark.parametrize(
    "field,value",
    (
        ("capacity", True),
        ("conductances", 1.0),
        ("connected_degrees", True),
        ("cut_degrees", True),
        ("host_mean_weights", False),
        ("host_alpha", 0.5),
    ),
)
def test_expected_source_physical_scalars_are_admitted_before_any_file_read(
    archived_branch, monkeypatch, field, value
):
    bundle = deepcopy(archived_branch)
    source = bundle["loader_kwargs"]["source_binding"]["complete_source"]
    values = list(source[field])
    values[0] = value
    source[field] = tuple(values)

    def forbidden(*args, **kwargs):
        pytest.fail("malformed expected primitives must reject before evidence reads")

    monkeypatch.setattr(owner, "read_bytes_bounded", forbidden)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


@pytest.mark.parametrize(
    "field", ("form_mean", "phase_lift_mean", "form_offset", "phase_offset")
)
def test_expected_cut_means_and_offsets_are_rebuilt_from_primitives(
    archived_branch, monkeypatch, field
):
    bundle = deepcopy(archived_branch)
    source = bundle["loader_kwargs"]["source_binding"]["complete_source"]
    source[field] += Q(1, 8)

    def forbidden(*args, **kwargs):
        pytest.fail("inconsistent derived means must reject before evidence reads")

    monkeypatch.setattr(owner, "read_bytes_bounded", forbidden)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(bundle)


@pytest.mark.parametrize(
    "budget",
    (
        "MAX_ARCHIVE_BYTES",
        "MAX_MEMBER_BYTES",
        "MAX_ARCHIVES_BYTES",
        "MAX_EXPANDED_BYTES",
    ),
)
def test_archive_limits_reject_before_opening_the_first_step(
    archived_branch, monkeypatch, budget
):
    monkeypatch.setattr(owner, budget, 1)

    def forbidden(*args, **kwargs):
        pytest.fail("the complete declared archive budget must precede step access")

    monkeypatch.setattr(owner, "_read_step", forbidden)
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        _load(archived_branch)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"value":' + b"9" * 1235 + b"}",
        b"[" * 33 + b"0" + b"]" * 33,
        b'{"numerator":true,"denominator":1}',
        b'{"value":NaN}',
    ),
)
def test_numeric_and_nesting_limits_precede_exact_tag_materialization(raw):
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        owner._json(raw)


def test_raw_integer_bit_limit_checks_the_value_not_only_decimal_length():
    raw = ('{"numerator":' + str(1 << 4096) + ',"denominator":1}').encode()
    with pytest.raises(ValueError, match="bit"):
        owner._json(raw)


def test_whitespace_padded_archive_keeps_the_independent_constant_field_arithmetic(
    tmp_path,
):
    builder = _fixture_builder()
    helpers = sys.modules[builder.__module__]
    row, initial, _, endpoint = _record("cut_host")
    raw, archive, member = helpers.encoded_step(row, padding_bytes=196608)
    assert len(raw) == 196608
    path = tmp_path / "steps" / "000.zip"
    path.parent.mkdir()
    path.write_bytes(archive)
    archive_receipt = owner._receipt(file_receipt(path, label="steps/000.zip"))
    member_receipt = owner._receipt(
        {"path": member, "bytes": len(raw), "sha256": sha256_bytes(raw)}
    )
    decoded = owner._read_step(tmp_path, archive_receipt, member_receipt)
    admitted = _audit(decoded, initial, "cut_host")
    assert admitted.endpoint == endpoint


def test_connected_negative_tube_margin_can_retain_an_independent_positive_combined_bound():
    record, initial, _, _ = _record("connected")
    record["tube_principal_margins"] = (Q(-1),) + (Q(1),) * 52
    admitted = _audit(record, initial, "connected")
    assert "inherited" in admitted.picard_status
    record["combined_principal_margins"] = (Q(0),) + (Q(1),) * 52
    with pytest.raises(ValueError):
        _audit(record, initial, "connected")
