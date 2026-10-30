"""Bounded post-run documentary audit; never propagate or re-admit target signs.

Run only after its independent review and freeze. One certificate is read at a
time. Failure or an incomplete prefix is not a non-entry theorem. The output is
outside the attempt and belongs to a separate 120-second/8-MiB documentary
budget. The external caller must also enforce that documentary timeout.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from fractions import Fraction as Q
import math
from pathlib import Path
import re
import time

from tnfr.research.artifact_io import (
    decode_exact_tree, encode_exact_tree, exact_record, file_receipt,
    read_bytes_bounded, sha256_bytes, validate_artifact_path,
    verify_archive_members, write_json_once,
)
from tnfr.research.formed_two_port_run import (
    AUX_BYTES, CALLER_RESERVE, ORDER, STEP_BYTES, STEP_COUNT, STEP_SIZE,
    SUPERVISOR_RESERVE, declared_policy,
)
from tnfr.mathematics._validated_metric import ValidatedMetricTaylorStep
from tnfr.utils.io import json_dumps, json_loads

JSON_LIMIT = 4 * 1024**2
OUTPUT_LIMIT = 8 * 1024**2
CALLER_HALF = 32 * 1024
WORK_SECONDS = 110
TOTAL_SECONDS = 120
METHOD = "retained_spd_ball_picard_taylor_dyadic128_scalar_growth_v1"
PREMISES = (
    "flow_is_the_declared_smooth_autonomous_complete_field_on_the_whole_tube",
    "domain_margins_certify_all_consumed_model_and_growth_bound_hypotheses",
    "supremum_of_the_metric_logarithmic_Jacobian_norm_on_the_convex_tube_is_at_most_growth_rate",
    "initial_uncertainty_is_the_declared_metric_ball_not_its_coordinate_projection",
)
TARGET_NAMES = {
    "docs/assets/sine_formed_classes/two-port-compatibility-v1" + suffix
    for suffix in (".json", ".manifest.json", ".protocol.json", ".source.zip")
}
DEGREES = tuple(3 if i in (4, 5, 13, 14) else 2 for i in range(18))
METRIC = tuple(tuple(Q(d if i == j else 0) for j in range(18)) for i, d in enumerate(DEGREES))
EDGES = tuple((s + j, s + (j + 1) % 9) for s in (0, 9) for j in range(9)) + ((4, 13), (5, 14))


class DocumentaryUnavailable(ValueError):
    """Insufficient or inconsistent retained documentation, not non-entry."""


def require(condition, message):
    if not condition:
        raise DocumentaryUnavailable(message)


def qrow(values, length=18):
    require(isinstance(values, (tuple, list)) and len(values) == length, "exact row shape")
    return tuple(exact_record(v) for v in values)


def intervals(values):
    require(isinstance(values, (tuple, list)) and len(values) == 18, "interval row shape")
    result = []
    for row in values:
        require(isinstance(row, dict) and set(row) == {"lo", "hi"}, "interval field shape")
        lo, hi = exact_record(row["lo"]), exact_record(row["hi"])
        require(lo <= hi, "reversed interval")
        result.append((lo, hi))
    return tuple(result)


def check_certificate(cert, index, center, radius, projection):
    """Exact record consistency only; no Picard/Taylor/trigonometric replay."""
    require(len(cert) == 26, "complete 26-field certificate required")
    require(exact_record(cert["time"]) == index * STEP_SIZE, "time chain")
    require(exact_record(cert["duration"]) == STEP_SIZE, "fixed step duration")
    require(type(cert["order"]) is int and cert["order"] == ORDER, "fixed order")
    require(tuple(qrow(row) for row in cert["metric"]) == METRIC, "fixed full metric")
    require(qrow(cert["coordinate_projection_factors"]) == projection, "shared projection changed")
    require(qrow(cert["initial_center"]) == center and exact_record(cert["initial_radius"]) == radius, "center/radius chain")
    require(exact_record(cert["growth_rate_upper_bound"]) == 0 and exact_record(cert["growth_factor_upper_bound"]) == 1, "growth policy")
    require(cert["method"] == METHOD and tuple(cert["conditional_premises"]) == PREMISES, "method/premises")
    require(cert["growth_certificate"] is None and cert["prescribed_time_interval"] is None, "different growth/clock method")
    require(exact_record(cert["picard_interior_margin"]) > 0, "strict Picard margin")
    require(min(qrow(cert["domain_lower_bounds"], 20)) > 0, "whole-tube slacks above rho")
    initial, tube = intervals(cert["initial_box"]), intervals(cert["tube"])
    polynomial, remainder = intervals(cert["center_polynomial_bounds"]), intervals(cert["local_remainder_bounds"])
    solution, endpoint = intervals(cert["center_solution_endpoint"]), intervals(cert["endpoint"])
    coordinate_errors = qrow(cert["local_coordinate_error_bounds"])
    new_center, new_radius = qrow(cert["endpoint_center"]), exact_record(cert["endpoint_radius"])
    propagated = exact_record(cert["propagated_initial_radius"])
    local = exact_record(cert["local_metric_error_upper_bound"])
    require(radius >= 0 and propagated >= radius and local >= 0, "nonnegative propagated/local radii")
    require(new_radius >= propagated + local, "endpoint radius cannot discard error")
    require(local**2 >= sum(d * e**2 for d, e in zip(DEGREES, coordinate_errors)), "local metric-error consistency")
    for i in range(18):
        lo, hi = initial[i]
        require(lo <= center[i] - projection[i] * radius and hi >= center[i] + projection[i] * radius, "initial ball projection")
        require(tube[i][0] <= lo <= hi <= tube[i][1], "initial box outside tube")
        slo, shi = solution[i]
        require(slo <= polynomial[i][0] + remainder[i][0] and shi >= polynomial[i][1] + remainder[i][1], "remainder interval lost")
        require(new_center[i] == (slo + shi) / 2 and coordinate_errors[i] == (shi - slo) / 2, "center/error projection consistency")
        require(max(slo, tube[i][0]) <= min(shi, tube[i][1]), "endpoint and tube disjoint")
        require(endpoint[i][0] <= new_center[i] - projection[i] * new_radius and endpoint[i][1] >= new_center[i] + projection[i] * new_radius, "endpoint ball projection")
    return new_center, new_radius


class Audit:
    def __init__(self, attempt, protocol, caller):
        self.start = time.monotonic()
        self.attempt = Path(attempt).resolve(strict=True)
        self.protocol = Path(protocol).resolve(strict=True)
        self.caller = Path(caller).resolve()
        require(not self.caller.is_relative_to(self.attempt), "external caller envelope must be outside attempt")
        self.files = {}
        self.summary = {
            "schema": "tnfr.formed-two-port-documentary-audit.v1",
            "post_run_documentary_work": True, "scientific_calls": 0,
            "target_sign_replays": 0, "reference_steps_replayed": 0,
            "documentary_work_seconds": WORK_SECONDS, "documentary_total_seconds": TOTAL_SECONDS,
            "documentary_output_limit_bytes": OUTPUT_LIMIT, "json_read_limit_bytes": JSON_LIMIT,
            "attempt_directory": str(self.attempt), "verified_prefix_steps": 0,
            "source_associations_verified": False,
            "complete_reference_certificate_admitted": False,
            "non_entry_proved": False, "provenance_authenticated": False,
            "issues": [],
        }

    def clock(self):
        require(time.monotonic() - self.start < WORK_SECONDS, "documentary time budget exhausted")

    def issue(self, error):
        if len(self.summary["issues"]) < 32:
            self.summary["issues"].append(str(error)[:600])

    def path(self, name):
        name = validate_artifact_path(name)
        path = self.attempt / name
        require(path.resolve() == path and path.is_file(), "missing or redirected attempt file: " + name)
        return path

    def read(self, path, limit=JSON_LIMIT):
        self.clock()
        raw = read_bytes_bounded(path, max_bytes=limit)
        record = json_loads(raw)
        require(isinstance(record, dict), "JSON object required")
        self.clock()
        return record, {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw)}

    def local(self, name, limit=JSON_LIMIT):
        record, row = self.read(self.path(name), limit)
        row["path"] = name
        return record, row

    def bind_local(self, row):
        require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, "receipt shape")
        path = self.path(row["path"])
        require(type(row["bytes"]) is int and 0 <= row["bytes"] <= AUX_BYTES + STEP_BYTES, "receipt size")
        self.clock()
        require(file_receipt(path, label=row["path"], max_bytes=row["bytes"]) == row, "receipt bytes differ: " + row["path"])

    def inventory(self):
        for path in self.attempt.iterdir():
            self.clock()
            require(path.resolve() == path and not path.is_symlink(), "redirected attempt member")
            if path.is_dir():
                require(path.name == "steps", "unexpected attempt directory")
                members = path.iterdir()
            else:
                members = (path,)
            for item in members:
                self.clock()
                require(item.is_file() and item.resolve() == item and not item.is_symlink(), "nonregular/redirected evidence")
                name = item.relative_to(self.attempt).as_posix()
                self.files[name] = item.stat().st_size
                require(len(self.files) <= 5000, "documentary inventory limit")
        step_bytes = sum(size for name, size in self.files.items() if name.startswith("steps/"))
        auxiliary = sum(self.files.values()) - step_bytes
        external = self.caller.stat().st_size
        self.summary.update(step_bytes=step_bytes, attempt_auxiliary_bytes=auxiliary, external_envelope_bytes=external)
        require(step_bytes <= STEP_BYTES, "step quota exceeded, including partial/shared files")
        require(external <= CALLER_HALF and self.files.get("attempt-verdict.json", 0) <= CALLER_HALF, "caller partition exceeded")
        require(auxiliary + external <= AUX_BYTES, "auxiliary quota exceeded including external envelope")
        parent = sum(self.files.get(n, 0) for n in ("attempt.json", "execution.json", "worker.log"))
        require(parent <= SUPERVISOR_RESERVE and self.files.get("worker.log", 0) <= 1024**2, "supervisor quota exceeded")
        require(self.files.get("attempt.json", 0) <= 512*1024 and self.files.get("execution.json", 0) <= 512*1024, "supervisor record quota")
        require(external + self.files.get("attempt-verdict.json", 0) <= CALLER_RESERVE, "combined caller reserve exceeded")

    def sources(self):
        protocol, declared_receipt = self.read(self.protocol)
        copied, copied_receipt = self.local("protocol.json")
        require(copied == protocol and copied_receipt["sha256"] == declared_receipt["sha256"], "copied protocol differs from prospective input")
        require(protocol["schema"] == "tnfr.formed-two-port-executable-protocol.v1", "protocol schema")
        require(json_dumps(protocol["policy"], sort_keys=True) == json_dumps(encode_exact_tree(declared_policy()), sort_keys=True), "fixed prospective policy")
        rows = protocol["sources"]
        require(isinstance(rows, list) and 1 <= len(rows) <= 10000, "frozen source inventory")
        sources = {row["path"]: row for row in rows}
        require(len(sources) == len(rows) and TARGET_NAMES <= set(sources), "target/source inventory")
        entry = protocol["worker_entry"]
        require(sources.get(entry["path"]) == entry, "frozen wrapper association")
        archive = file_receipt(self.path("source.zip"), label="source.zip", max_bytes=AUX_BYTES)
        require((archive["bytes"], archive["sha256"]) == (protocol["archive"]["bytes"], protocol["archive"]["sha256"]), "frozen source archive differs")
        self.clock()
        verify_archive_members(self.path("source.zip"), {n: row["sha256"] for n, row in sources.items()}, max_bytes=AUX_BYTES)
        self.clock()
        for i, row in enumerate(protocol["reviews"]):
            actual = file_receipt(self.path(f"review-{i:02d}.md"), label=row["path"], max_bytes=row["bytes"])
            require(actual == row, "review association differs")
        controls, cr = self.local("controls.json")
        require((cr["bytes"], cr["sha256"]) == (protocol["controls"]["bytes"], protocol["controls"]["sha256"]), "control receipt differs")
        require(controls.get("passed") is True and type(controls.get("reserved_reference_steps")) is int and controls["reserved_reference_steps"] == 0, "non-transit control admission")
        admission, self.admission_receipt = self.local("admission.json")
        expected = {"protocol": copied_receipt, "archive": archive, "head": protocol["head"], "worker_entry": entry, "target_evidence": [sources[n] for n in sorted(TARGET_NAMES)]}
        require(admission["source"] == expected, "actual source association differs")
        target_rows = admission["target"]["evidence_receipts"]
        require(len(target_rows) == 4 and {r["path"]: r for r in target_rows} == {n: sources[n] for n in TARGET_NAMES}, "actually consumed target differs")
        require(admission.get("actual_source_is_flow_image_family") is True and admission.get("no_actual_state_reset") is True, "source family boundary")
        ref = decode_exact_tree(admission["reference"])
        require(tuple(ref["degrees"]) == DEGREES and tuple(ref["oriented_edges"]) == EDGES, "original source labels/degrees")
        require(tuple(qrow(row) for row in ref["metric"]) == METRIC and qrow(ref["initial_center"]) == (Q(0),)*18 and exact_record(ref["initial_radius"]) == 0, "initial reference/metric")
        require(exact_record(ref["reference_margin"]) == Q(1, 2048), "rho changed")
        start, _ = self.local("worker-start.json")
        require(start["schema"] == "tnfr.formed-two-port-attempt.v1" and json_dumps(start["policy"], sort_keys=True) == json_dumps(encode_exact_tree(declared_policy()), sort_keys=True), "worker policy")
        self.protocol_record, self.source, self.admission = protocol, expected, admission
        self.summary["source_associations_verified"] = True
        self.summary["protocol"] = declared_receipt

    def prefix(self):
        common, common_receipt = self.local("steps/certificate-fields.json")
        require(common["schema"] == "tnfr.formed-two-port-certificate-fields.v1", "shared schema")
        common = decode_exact_tree(common["fields"])
        require(set(common) == {"metric", "coordinate_projection_factors"}, "shared inventory")
        require(tuple(qrow(row) for row in common["metric"]) == METRIC, "shared metric")
        projection = qrow(common["coordinate_projection_factors"])
        require(all(p > 0 and d*p*p >= 1 for d, p in zip(DEGREES, projection)), "outward projection consistency")
        center, radius, previous = (Q(0),)*18, Q(0), None
        for index in range(STEP_COUNT):
            name = f"steps/{index:04d}.json"
            if name not in self.files:
                break
            try:
                record, receipt = self.local(name)
                require(record["schema"] == "tnfr.formed-two-port-full-step.v1" and type(record["index"]) is int and record["index"] == index, "step index/schema")
                require(record["source"] == self.source and record["previous_complete_step"] == previous, "step source/hash chain")
                require(record["shared_certificate_fields"] == common_receipt, "shared receipt differs")
                require(record["domain_values"] == "strict excess above rho, not raw acute margins", "domain observation convention")
                # Same field partition/exact decoder as read_step_certificate,
                # but reconstruct from the two already bounded byte reads.
                # Its path API would reopen with a 256-MiB cap; this audit's
                # independently declared 4-MiB JSON boundary must remain real.
                values = record["certificate"]
                expected = {f.name for f in fields(ValidatedMetricTaylorStep)}
                require(isinstance(values, dict) and set(values) == expected - set(common), "complete step field partition")
                cert = decode_exact_tree({**values, **common})
                center, radius = check_certificate(cert, index, center, radius, projection)
                self.clock()
                previous = receipt
                self.summary["verified_prefix_steps"] = index + 1
                self.summary["last_verified_step"] = previous
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.issue(f"{name}: {exc}")
                break
        self.center, self.radius, self.previous = center, radius, previous
        count = self.summary["verified_prefix_steps"]
        remaining = [name for name in self.files if name.startswith("steps/") and name != "steps/certificate-fields.json" and not re.fullmatch(r"steps/\d{4}\.json", name)]
        later = [name for name in self.files if re.fullmatch(r"steps/\d{4}\.json", name) and int(Path(name).stem) >= count]
        self.summary["unverified_step_files"] = sorted(remaining + later)[:16]
        require(not remaining and not later and count == STEP_COUNT, "incomplete or unverified reference prefix")

    def terminal(self):
        worker, worker_receipt = self.local("worker-result.json")
        require(worker["source"] == self.source and worker["last_complete_step"] == self.previous, "worker terminal source/last receipt")
        require(type(worker["completed_steps"]) is int and worker["completed_steps"] == STEP_COUNT, "worker terminal step count")
        require(worker["status"] == "certified_capture" and worker.get("conditional_on_successful_post_write_supervisor_return") is True, "worker result unavailable")
        require(type(worker["step_evidence_reserved_bytes"]) is int and worker["step_evidence_reserved_bytes"] == self.summary["step_bytes"], "successful step byte ledger")
        endpoint = decode_exact_tree(worker["endpoint"])
        require(endpoint["certified"] is True and endpoint["conditional_on_acute_reference_prefix"] is True and not endpoint["reasons"], "conditional endpoint unavailable")
        require(endpoint["form_norm_observation"] == "||P_D x||_D relative to the conserved uniform form", "relative-form observation")
        distance = exact_record(endpoint["reference_distance_upper_bound"])
        require(0 <= self.radius <= distance <= Q(1,2048), "endpoint distance/radius")
        target_intervals = intervals(self.admission["target"]["target_deviations"])
        squared_box_distance = sum(d * max(abs(c-lo), abs(c-hi))**2 for d, c, (lo, hi) in zip(DEGREES, self.center, target_intervals))
        require((distance-self.radius)**2 >= squared_box_distance, "endpoint distance not linked to retained center and target enclosure")
        require(exact_record(endpoint["reference_distance_margin"]) == Q(1,2048)-distance, "endpoint distance margin")
        for value, margin, boundary in (
            ("endpoint_form_norm_upper_bound", "form_margin", Q(1,8192)),
            ("endpoint_phase_distance_upper_bound", "phase_margin", Q(1,1024)),
            ("endpoint_excess_storage_upper_bound", "capture_storage_margin", Q(1,648000)),
        ):
            bound = exact_record(endpoint[value])
            require(0 <= bound < boundary and exact_record(endpoint[margin]) == boundary-bound, "endpoint bound/margin: " + value)
            require(endpoint[value] == decode_exact_tree(self.admission["bounds"])[value], "endpoint/bound association")
        storage = exact_record(endpoint["endpoint_excess_storage_upper_bound"])
        require(storage >= exact_record(endpoint["endpoint_form_norm_upper_bound"])**2 + exact_record(endpoint["endpoint_phase_distance_upper_bound"])**2, "storage bound loses full-state errors")
        require(exact_record(endpoint["capture_radius_squared_margin"]) == Q(1,144)-storage > 0, "capture radius margin")
        self.worker_receipt = worker_receipt

    def caller_result(self):
        """Final caller schema is frozen together with the launch wrapper."""
        envelope, caller_receipt = self.read(self.caller, CALLER_HALF)
        require(type(envelope["wrapper_exit_code"]) is int and envelope["wrapper_exit_code"] == 0 and envelope.get("wrapper_timed_out") is False, "outer wrapper did not finish successfully")
        require(envelope.get("output_error") is None and type(envelope.get("stderr_bytes")) is int and envelope["stderr_bytes"] == 0, "outer output was incomplete or had stderr")
        require(envelope.get("stdout_complete") is True and envelope.get("stderr_complete") is True and envelope.get("capture_errors") == [], "outer capture did not retain complete streams")
        returned = envelope["launcher_return"]
        require(returned["schema"] == "tnfr.formed-two-port-caller.v1" and returned["record_stage"] == "post_caller_write_result", "caller post-write stage")
        require(returned.get("execution_passed") is True and returned.get("caller_write_within_total_budget") is True, "caller write/deadline unavailable")
        elapsed = returned.get("caller_elapsed_seconds")
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed < 3600, "caller elapsed time invalid")
        require(type(returned.get("caller_total_seconds_limit")) is int and returned["caller_total_seconds_limit"] == 3600, "caller total policy")
        require(returned.get("reference_success_admitted") is False and returned.get("documentary_audit_required") is True, "caller cannot replace prefix audit")
        verdict, verdict_receipt = self.local("attempt-verdict.json", CALLER_HALF)
        require(returned["caller_verdict"] == verdict_receipt, "caller snapshot receipt")
        require(verdict["schema"] == returned["schema"] and verdict["record_stage"] == "pre_caller_write_snapshot_not_final_verdict", "caller snapshot stage")
        require(verdict["supervisor_return"] == returned["supervisor_return"] and verdict["receipts"] == returned["receipts"], "caller source snapshot changed")
        mutable_caller = {"record_stage", "caller_elapsed_seconds", "caller_write_within_total_budget", "caller_verdict", "execution_passed", "status"}
        require({k:v for k,v in verdict.items() if k not in mutable_caller} == {k:v for k,v in returned.items() if k not in mutable_caller}, "caller return differs beyond closing fields")
        require(verdict["execution_passed"] is True, "caller snapshot failed")
        require(returned["attempt_directory"] == str(self.attempt) and verdict["attempt_directory"] == str(self.attempt), "caller attempt association")
        require(returned["protocol_receipt"] == self.summary["protocol"], "caller invoked different protocol")
        receipts = returned["receipts"]
        require(set(receipts) == {"attempt", "execution_snapshot", "log", "worker_result"}, "caller receipt inventory")
        labels = {"attempt":"attempt.json", "execution_snapshot":"execution.json", "log":"worker.log", "worker_result":"worker-result.json"}
        for key, row in receipts.items():
            require(row["path"] == labels[key], "caller receipt label differs")
            self.bind_local(row)
        require(receipts["worker_result"] == self.worker_receipt, "caller consumed different worker result")
        supervisor = returned["supervisor_return"]
        require(supervisor["record_stage"] == "post_write_result_required_for_attempt_admission" and supervisor.get("passed") is True and supervisor.get("terminal_write_deadline_verified") is True, "supervisor post-write result")
        require(supervisor.get("status") == "success" and supervisor.get("error") is None and supervisor.get("watchdog_error") is None, "supervisor failure status")
        require(type(supervisor.get("wall_seconds_limit")) is int and supervisor["wall_seconds_limit"] == 3600 and type(supervisor.get("process_memory_limit_bytes")) is int and supervisor["process_memory_limit_bytes"] == 1024**3, "supervisor budget changed")
        require(supervisor.get("closing_reserve_seconds") == 10 and supervisor.get("worker_seconds_budget") == 3590, "supervisor closing policy")
        require(supervisor.get("total_budget_exhausted") is False and supervisor.get("worker_stopped_observed") is True and supervisor.get("cleanup_complete") is True, "supervisor incomplete/deadline")
        require(supervisor.get("job_assigned_before_resume") is True and supervisor.get("resumed") is True and type(supervisor.get("exit_code")) is int and supervisor["exit_code"] == 0, "supervised worker unavailable")
        require(supervisor["attempt"] == receipts["attempt"] and supervisor["log"] == receipts["log"], "supervisor consumed different evidence")
        execution, er = self.local("execution.json")
        require(er == receipts["execution_snapshot"] and execution["record_stage"] == "pre_write_snapshot_not_final_attempt_verdict", "supervisor snapshot association")
        require(execution.get("passed") is True and execution.get("status") == "success" and execution.get("terminal_write_deadline_verified") is False and execution.get("total_budget_exhausted") is False and set(execution) == set(supervisor), "supervisor snapshot must itself be successful")
        mutable = {"record_stage", "supervisor_elapsed_seconds", "terminal_write_deadline_verified", "total_budget_exhausted", "passed", "status"}
        require({k:v for k,v in execution.items() if k not in mutable} == {k:v for k,v in supervisor.items() if k not in mutable}, "supervisor return differs beyond post-write fields")
        self.summary["external_caller_envelope"] = caller_receipt
        self.summary["caller_verdict"] = verdict_receipt

    def failure_observations(self):
        """Retain bounded failure metadata even when the numerical prefix is absent."""
        for name, keys in (
            ("worker-result.json", ("status", "completed_steps", "exception_type", "reason", "non_entry_not_proved")),
            ("attempt-verdict.json", ("status", "execution_passed", "worker_status", "completed_steps", "record_stage")),
        ):
            if name not in self.files:
                continue
            try:
                record, row = self.local(name, CALLER_HALF if name == "attempt-verdict.json" else JSON_LIMIT)
                observed = {k: record[k] for k in keys if k in record and type(record[k]) in (str, int, bool, type(None))}
                observed = {k: v[:600] if isinstance(v, str) else v for k,v in observed.items()}
                self.summary[name] = {"receipt": row, "reported_values_not_accepted_as_certificate": observed}
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.issue(f"retained partial/unreadable {name}: {exc}")
        try:
            outer, row = self.read(self.caller, CALLER_HALF)
            self.summary["external_caller_envelope"] = row
            self.summary["reported_outer_exit"] = outer.get("wrapper_exit_code")
            self.summary["reported_outer_timeout"] = outer.get("wrapper_timed_out")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.issue(f"external outcome unavailable: {exc}")

    def run(self):
        try:
            self.inventory()
            self.sources()
            self.prefix()
            self.terminal()
            self.caller_result()
            self.clock()
            self.summary["complete_reference_certificate_admitted"] = True
            self.summary["status"] = "complete_conditional_documentary_certificate"
        except Exception as exc:
            # Damaged ZIP members, unsupported metadata and decode failures
            # must also leave a bounded unavailable summary. Interruptions
            # remain BaseException and are left to the external watchdog.
            self.issue(exc)
            self.summary["status"] = "incomplete_or_failed_attempt_no_nonentry_conclusion"
            self.failure_observations()
        self.summary["elapsed_before_output_seconds"] = time.monotonic()-self.start
        self.summary["boundary"] = "Content/chain consistency conditional on reviewed producing execution; no target/Taylor reproof, execution authentication, autonomous contact or physical identification."
        return self.summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--caller-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(not output.is_relative_to(args.attempt.resolve()), "documentary output must be outside preserved attempt")
    require(not output.exists(), "documentary output already exists")
    audit = Audit(args.attempt, args.protocol, args.caller_result)
    summary = audit.run()
    encoded = (json_dumps(encode_exact_tree(summary), indent=2, allow_nan=False)+"\n").encode()
    require(len(encoded) <= OUTPUT_LIMIT, "documentary output quota")
    write_json_once(output, summary)
    within = time.monotonic()-audit.start < TOTAL_SECONDS
    print(json_dumps({"summary": str(output), "post_write_within_documentary_budget": within, "reference_certificate_admitted": within and summary["complete_reference_certificate_admitted"]}))
    return 0 if within and summary["complete_reference_certificate_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
