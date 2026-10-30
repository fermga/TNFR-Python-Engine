"""Static primitive-data audit of the single frozen matched selection witness.

Import and call only inside the newly reviewed/frozen supervised evaluation.
No candidate, flow, Jacobian, solver, graph executor or pytest producer is run.
The old continuous certificate is consumed by immutable source associations;
this audit recomputes only the prospectively selected algebraic/trigonometric
margins on its retained boxes. The symbolic proposed radius is never expanded.
"""

from __future__ import annotations

from fractions import Fraction as Q
from pathlib import Path

from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = "artifacts/native-selection-witness-20261028"
FINAL_RECEIPT = {
    "path": PREVIOUS + "/final-state.json",
    "bytes": 186178,
    "sha256": "c9698db64d2c9ad2e9b9bbe4c4e1eee9b7eef1c745ee8c1af3849f426aa65f4b",
}
INPUT_LIMIT = 16 * 1024**2
SOURCE_LIMIT = 128 * 1024**2
STEP = Q(1, 8)
RESULTANT_FLOOR = Q(1, 1 << 33)
CAPTURE_DEFICIT = Q(1, 1 << 30)
ARMS = (("held", 252, 1, Q(63, 2)), ("dynamic", 246, 0, Q(123, 4)))
CONSUMED_CODE = (
    "src/tnfr/research/artifact_io.py",
    "src/tnfr/research/native_capacity_selection.py",
    "src/tnfr/physics/relational_transit.py",
    "src/tnfr/mathematics/_rational_interval.py",
    "src/tnfr/mathematics/_phase_midpoint.py",
    "src/tnfr/mathematics/_phase_resultant_chamber.py",
    "src/tnfr/_exact_time.py",
    "src/tnfr/utils/io.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(name):
    _require(type(name) is str and bool(name), "invalid evidence path")
    path = (ROOT / name).resolve()
    _require(path.is_relative_to(ROOT.resolve()), "evidence path leaves repository")
    _require(path.relative_to(ROOT).as_posix() == name, "noncanonical evidence path")
    return path


def _verify(item, *, limit=INPUT_LIMIT):
    _require(set(item) == {"path", "bytes", "sha256"}, "invalid receipt schema")
    actual = file_receipt(_path(item["path"]), label=item["path"], max_bytes=limit)
    _require(actual == item, "receipt mismatch: " + item["path"])
    return actual


def _read(item, *, decode=True):
    """Hash and parse the same bounded byte read, without a stat/read race."""
    _require(set(item) == {"path", "bytes", "sha256"}, "invalid JSON receipt")
    data = read_bytes_bounded(_path(item["path"]), max_bytes=INPUT_LIMIT)
    _require(len(data) == item["bytes"], "JSON byte count changed: " + item["path"])
    _require(
        sha256_bytes(data) == item["sha256"], "JSON digest changed: " + item["path"]
    )
    value = json_loads(data)
    return decode_exact_tree(value) if decode else value


def _index(items):
    indexed = {}
    for item in items:
        _require(item["path"] not in indexed, "duplicate evidence path")
        indexed[item["path"]] = item
    return indexed


def _interval(value, interval_type):
    _require(type(value) is dict and set(value) == {"lo", "hi"}, "invalid box interval")
    lo, hi = exact_record(value["lo"]), exact_record(value["hi"])
    _require(lo <= hi, "reversed retained interval")
    result = interval_type(lo, hi)
    _require(
        (result.lo, result.hi) == (lo, hi),
        "retained interval is not on its frozen grid",
    )
    return result


def _box(values, interval_type):
    _require(
        type(values) is tuple and len(values) == 6, "box must have six coordinates"
    )
    return tuple(_interval(value, interval_type) for value in values)


def _bounds(value):
    return {"lo": value.lo, "hi": value.hi}


def _bounds_tree(value):
    if hasattr(value, "lo") and hasattr(value, "hi"):
        return _bounds(value)
    if isinstance(value, dict):
        return {key: _bounds_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_bounds_tree(item) for item in value)
    return value


def _source_admission(report):
    final = _read(FINAL_RECEIPT, decode=False)
    _require(
        final["schema"] == "tnfr.native-selection-witness-final-state.v1",
        "wrong final-state schema",
    )
    _require(final["closure_audit_passed"] is True, "prior closure did not pass")
    _require(final["paired_certificate_passed"] is True, "prior pair did not pass")
    _require(
        final["scientific_outcome"] == "paired_capture_certified",
        "prior outcome differs",
    )
    _require(
        final["physical_mapping_authorized"] is False,
        "unexpected physical source claim",
    )
    plan = _read(final["validation_plan"], decode=False)
    validation = _read(final["evaluation"], decode=False)
    _require(validation["evaluation_completed"] is True, "prior evaluation incomplete")
    _require(
        validation["paired_certificate_passed"] is True,
        "prior evaluation did not capture",
    )
    _require(validation["orientation_matched"] is True, "prior orientation differs")
    _require(validation["retry_permitted"] is False, "prior retry policy differs")
    _require(
        validation["plan"] == final["validation_plan"], "prior plan association differs"
    )
    resources = validation["resource_result"]
    _require(
        resources["reason"] is None and resources["worker_exit_code"] == 0,
        "prior worker failed",
    )
    _require(
        resources["cleanup_complete"] is True
        and resources["job_empty_after_cleanup"] is True,
        "prior cleanup incomplete",
    )
    _require(
        plan["source"] == final["evaluated_source"], "prior source manifests differ"
    )
    _require(
        plan["source_archive"] == final["evaluated_source_archive"],
        "prior source archives differ",
    )
    archive = _verify(plan["source_archive"], limit=SOURCE_LIMIT)
    sources = _index(plan["source"])
    verify_archive_members(
        _path(archive["path"]), {name: item["sha256"] for name, item in sources.items()}
    )
    consumed = tuple(
        _verify(sources[name], limit=SOURCE_LIMIT) for name in CONSUMED_CODE
    )
    report["source_associations"] = {
        "previous_final_state": FINAL_RECEIPT,
        "previous_validation_plan": final["validation_plan"],
        "previous_evaluation": final["evaluation"],
        "previous_source_archive": archive,
        "archive_member_count": len(sources),
        "current_consumed_code_matches_frozen_source": consumed,
    }
    artifacts = _index(final["retained_artifact_evidence"])
    paired = _read(validation["paired_result"])
    _require(
        paired["passed"] is True and paired["predicted_orientation_matched"] is True,
        "prior paired verdict differs",
    )
    _require(
        paired["new_candidate_count"] == 1
        and paired["parameter_scans"] == paired["prior_producer_replays"] == 0,
        "prior campaign scope differs",
    )
    _require(
        paired["physical_identification"] is False, "unexpected physical identification"
    )
    _require(
        set(paired["models"]) == {"held", "dynamic"}, "paired arm inventory differs"
    )
    candidate = _read(paired["candidate"])
    _require(
        artifacts[paired["candidate"]["path"]] == paired["candidate"],
        "candidate association differs",
    )
    report["consumed_data"] = [validation["paired_result"], paired["candidate"]]
    return artifacts, paired, candidate


def _preparation(candidate, interval_type):
    _require(
        candidate["exact_state_shared_by_both_models"] is True,
        "shared exact preparation absent",
    )
    selected = candidate["selection"]
    initial = tuple(exact_record(value) for value in selected["initial"])
    _require(len(initial) == 6, "initial state dimension differs")
    exact = selected["exact_native"]
    _require(
        set(exact) == {"A", "B", "a", "b", "s", "t"},
        "native preparation coordinates differ",
    )
    exact = {name: exact_record(value) for name, value in exact.items()}
    q, r, a, b, s, t = initial
    A, B = exact["A"], exact["B"]
    _require(
        q == 3 * A - B and r == 2 * B - A, "exact forward native embedding differs"
    )
    _require(
        A == (2 * q + r) / 5 and B == (q + 3 * r) / 5,
        "exact inverse native embedding differs",
    )
    _require(
        (a, b, s, t) == tuple(exact[name] for name in ("a", "b", "s", "t")),
        "primitive preparation differs",
    )
    _require(s == t == Q(33, 32), "capacity preparation differs")
    u = 5 - 2 * s - 2 * t
    _require(
        u == Q(7, 8) == exact_record(selected["capacity_center"]),
        "central capacity differs",
    )
    _require(exact_record(selected["epsilon"]) == Q(1, 32), "epsilon differs")
    forms = (A, -A, -B, Q(0), B) * 2
    phases = (a, -a, -b, Q(0), b) * 2
    capacities = (s, s, t, u, t) * 2
    _require(sum(capacities, Q(0)) == 10, "full capacity mean differs")
    _require(
        min(capacities) == Q(7, 8) and max(capacities) == Q(33, 32),
        "initial capacity hull differs",
    )
    _require(
        sum(forms, Q(0)) == sum(phases, Q(0)) == 0, "copied/reflected origins differ"
    )
    edges = tuple(
        (offset + index, offset + (index + 1) % 5)
        for offset in (0, 5)
        for index in range(5)
    ) + ((0, 5), (1, 6))
    return tuple(interval_type(value) for value in initial), {
        "exact_initial_six_coordinates": initial,
        "full_node_order": tuple(range(10)),
        "full_edges": edges,
        "full_form": forms,
        "full_phase_lifts": phases,
        "full_capacity": capacities,
        "capacity_mean": Q(1),
        "initial_capacity_hull": {"lo": Q(7, 8), "hi": Q(33, 32)},
        "capacity_hull_scope": "Exact initial hull; all-future preservation is the separate admitted maximum-principle theorem, not an assertion that every numerical Picard box lies in that hull.",
        "forward_inverse_embedding_checked": True,
        "identical_exact_preparation_for_both_arms": True,
    }


def _records(receipt):
    data = read_bytes_bounded(_path(receipt["path"]), max_bytes=INPUT_LIMIT)
    _require(
        len(data) == receipt["bytes"] and sha256_bytes(data) == receipt["sha256"],
        "step-log receipt differs",
    )
    lines = data.splitlines()
    _require(lines and all(line.strip() for line in lines), "empty retained step row")
    return tuple(decode_exact_tree(json_loads(line)) for line in lines)


def audit():
    """Return explicit guards and failures from one bounded static audit.

    A passing report establishes only its selected retained-data margins.
    The quantitative recovery and full-state radius theorem remains a
    separate reviewed proof. No radius denominator is ever constructed.
    """
    report = {
        "schema": "tnfr.native-selection-robustness-static-audit.v1",
        "passed": False,
        "status": "incomplete",
        "guard_selection": {
            "expected_saved_steps": 498,
            "form_contrast_abs_upper_exclusive": Q(1),
            "real_resultant_lower_inclusive": RESULTANT_FLOOR,
            "endpoint_storage_deficit_lower_inclusive": CAPTURE_DEFICIT,
            "symbolic_proposed_radius": "2^(-2^250)",
            "radius_denominator_materialized": False,
        },
        "checks_performed": 0,
        "failed_guards": [],
        "arms": {},
        "new_trajectories": 0,
        "candidate_recipe_calls": 0,
        "flow_or_solver_calls": 0,
        "historical_producer_replays": 0,
    }

    def check(condition, guard, *, arm=None, step=None, evidence=None):
        report["checks_performed"] += 1
        if not condition:
            report["failed_guards"].append(
                {"guard": guard, "arm": arm, "step": step, "evidence": evidence}
            )

    try:
        artifacts, paired, candidate = _source_admission(report)
        # Import only the reviewed primitive arithmetic and observation owners.
        # The scientific producer functions are deliberately never called.
        from tnfr.mathematics._rational_interval import I, cos
        from tnfr.research.native_capacity_selection import capture, domain

        initial, report["preparation"] = _preparation(candidate, I)
        total_steps = 0
        for name, expected_count, sector, horizon in ARMS:
            log_receipt = paired["step_logs"][name]
            _require(
                artifacts[log_receipt["path"]] == log_receipt, "log association differs"
            )
            capture_receipt = artifacts[PREVIOUS + "/" + name + "-capture.json"]
            saved_capture = _read(capture_receipt)
            rows = _records(log_receipt)
            _require(len(rows) == expected_count, "retained arm step count differs")
            model = paired["models"][name]
            _require(
                model["status"] == "captured"
                and model["steps"] == expected_count
                and model["sector"] == sector
                and model["reason"] is None,
                "paired arm metadata differs",
            )
            _require(
                exact_record(saved_capture["time"]) == horizon
                and saved_capture["accepted_steps"] == expected_count,
                "capture time/count differs",
            )
            previous = initial
            audit_rows = []
            minimum_resultant = None
            minimum_domain = None
            maximum_contrast = Q(0)
            for index, row in enumerate(rows):
                _require(
                    set(row)
                    == {
                        "time",
                        "duration",
                        "tube",
                        "endpoint",
                        "picard_interior_margin",
                        "domain_lower_bounds",
                        "propagated_initial_radii",
                        "local_remainder_bounds",
                    },
                    "unexpected retained step fields",
                )
                time, duration = exact_record(row["time"]), exact_record(
                    row["duration"]
                )
                _require(
                    time == index * STEP and duration == STEP,
                    "noncontiguous step clock",
                )
                tube, endpoint = _box(row["tube"], I), _box(row["endpoint"], I)
                _require(
                    all(value.subset_of(bound) for value, bound in zip(previous, tube)),
                    "tube does not contain preceding endpoint",
                )
                _require(
                    all(value.subset_of(bound) for value, bound in zip(endpoint, tube)),
                    "endpoint leaves its certified whole-time tube",
                )
                old_margin = exact_record(row["picard_interior_margin"])
                _require(old_margin > 0, "prior Picard margin is not strict")
                radii = tuple(
                    exact_record(value) for value in row["propagated_initial_radii"]
                )
                _require(
                    len(radii) == 6 and min(radii) >= 0,
                    "invalid retained uncertainty radii",
                )
                _box(row["local_remainder_bounds"], I)
                q, r, a, b, s, t = tube
                real = (
                    1 + cos(2 * a) + cos(a - b),
                    2 * cos(a / 2) * cos(a / 2 - b),
                    2 * cos(b),
                )
                lower = tuple(value.lo for value in real)
                full_domain = domain(tube)
                _require(
                    tuple(exact_record(value) for value in row["domain_lower_bounds"])
                    == full_domain,
                    "retained domain evidence differs from primitive recomputation",
                )
                contrast = max(q.abs_max, r.abs_max)
                embedded_form_bound = max(
                    ((2 * q + r) / 5).abs_max, ((q + 3 * r) / 5).abs_max
                )
                check(
                    contrast < 1,
                    "all_tube_abs_q_r_below_one",
                    arm=name,
                    step=index,
                    evidence=contrast,
                )
                check(
                    min(lower) >= RESULTANT_FLOOR,
                    "all_three_primitive_real_resultants_above_floor",
                    arm=name,
                    step=index,
                    evidence=lower,
                )
                check(
                    min(full_domain) > 0,
                    "recomputed_complete_domain_positive",
                    arm=name,
                    step=index,
                    evidence=full_domain,
                )
                check(
                    embedded_form_bound < 1,
                    "embedded_full_form_bound",
                    arm=name,
                    step=index,
                    evidence=embedded_form_bound,
                )
                minimum_resultant = (
                    min(lower)
                    if minimum_resultant is None
                    else min(minimum_resultant, *lower)
                )
                minimum_domain = (
                    min(full_domain)
                    if minimum_domain is None
                    else min(minimum_domain, *full_domain)
                )
                maximum_contrast = max(maximum_contrast, contrast)
                audit_rows.append(
                    {
                        "step_index": index,
                        "time": time,
                        "duration": duration,
                        "maximum_abs_q_r": contrast,
                        "embedded_form_abs_upper": embedded_form_bound,
                        "real_resultant_lower_bounds": lower,
                        "minimum_complete_domain_lower_bound": min(full_domain),
                        "retained_picard_margin": old_margin,
                        "chain_and_endpoint_inclusions_checked": True,
                    }
                )
                previous = endpoint
            _require(expected_count * STEP == horizon, "declared horizon differs")
            _require(
                previous == _box(saved_capture["endpoint"], I) == _box(model["box"], I),
                "final endpoint associations differ",
            )
            observation = capture(previous)
            deficit = Q(7) - observation["storage"].hi
            check(
                observation["sector"] == sector,
                "recomputed_capture_sector",
                arm=name,
                evidence=observation["sector"],
            )
            check(
                deficit >= CAPTURE_DEFICIT,
                "recomputed_capture_storage_deficit",
                arm=name,
                evidence=deficit,
            )
            check(
                min(observation["domain"]) > 0,
                "recomputed_endpoint_domain_positive",
                arm=name,
                evidence=observation["domain"],
            )
            check(
                bool(observation["margins"])
                and all(value.lo > 0 for value in observation["margins"]),
                "recomputed_endpoint_rectangle_strict",
                arm=name,
                evidence=_bounds_tree(observation["margins"]),
            )
            check(
                saved_capture["observation"]["sector"] == sector,
                "retained_capture_sector_association",
                arm=name,
            )
            report["arms"][name] = {
                "steps": expected_count,
                "horizon": horizon,
                "initial_box": tuple(_bounds(value) for value in initial),
                "endpoint": tuple(_bounds(value) for value in previous),
                "minimum_real_resultant_lower_bound": minimum_resultant,
                "minimum_complete_domain_lower_bound": minimum_domain,
                "maximum_abs_q_r": maximum_contrast,
                "capture_deficit_lower_bound": deficit,
                "recomputed_capture": _bounds_tree(observation),
                "step_guards": tuple(audit_rows),
                "steps_receipt": log_receipt,
                "capture_receipt": capture_receipt,
            }
            report["consumed_data"].extend((log_receipt, capture_receipt))
            total_steps += expected_count
        check(total_steps == 498, "complete_saved_step_inventory", evidence=total_steps)
        report["saved_steps_checked"] = total_steps
        report["passed"] = not report["failed_guards"]
        report["status"] = (
            "selected_static_margins_certified"
            if report["passed"]
            else "selected_static_guard_unavailable"
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OSError,
        ArithmeticError,
    ) as exc:
        report["failed_guards"].append(
            {
                "guard": "source_or_retained_data_admission",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        report["status"] = "source_or_retained_data_unavailable"
    return report


if __name__ == "__main__":
    raise SystemExit("Use audit() only from the reviewed and frozen supervised worker.")
