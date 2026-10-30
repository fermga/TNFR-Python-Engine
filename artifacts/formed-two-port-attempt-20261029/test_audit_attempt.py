"""Synthetic documentary-auditor controls; no scientific producer or replay.

All source archives and certificates below are invented metadata fixtures.
They are never offered as evidence of a mathematical reference trajectory.
"""

import importlib.util
from pathlib import Path
from fractions import Fraction as Q
import zipfile

import pytest

from tnfr.research.artifact_io import file_receipt, write_json_once

spec = importlib.util.spec_from_file_location("attempt_auditor_controls_owner", Path(__file__).with_name("audit_attempt.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def no_scientific_calls(monkeypatch):
    from tnfr.mathematics import _validated_metric
    from tnfr.research import formed_two_port_reference, sine_two_port_handoff

    def forbidden(*args, **kwargs):
        pytest.fail("documentary controls must not evaluate mathematical producers")

    monkeypatch.setattr(_validated_metric, "validated_metric_taylor_step", forbidden)
    monkeypatch.setattr(_validated_metric, "_metric_error_upper", forbidden)
    monkeypatch.setattr(sine_two_port_handoff, "_target", forbidden)
    for name in ("build_formed_two_port_reference", "admit_formed_two_port_target", "formed_two_port_comparison_bounds", "admit_formed_two_port_endpoint"):
        monkeypatch.setattr(formed_two_port_reference, name, forbidden)
    monkeypatch.setattr(formed_two_port_reference.FormedTwoPortReference, "field", forbidden)
    monkeypatch.setattr(formed_two_port_reference.FormedTwoPortReference, "domain", forbidden)


def interval(lo=0, hi=0):
    return {"lo": Q(lo), "hi": Q(hi)}


def certificate(index=0, radius=Q(0)):
    zero = tuple(interval() for _ in range(18))
    return {
        "time": index*Q(1,4), "duration": Q(1,4), "order": 8,
        "metric": audit.METRIC, "initial_center": (Q(0),)*18,
        "initial_radius": radius,
        "initial_box": tuple(interval(-radius, radius) for _ in range(18)),
        "coordinate_projection_factors": (Q(1),)*18,
        "tube": tuple(interval(-1,1) for _ in range(18)),
        "picard_interior_margin": Q(1,8), "domain_lower_bounds": (Q(1,9),)*20,
        "growth_rate_upper_bound": Q(0), "growth_factor_upper_bound": Q(1),
        "propagated_initial_radius": radius,
        "center_polynomial_bounds": zero, "local_remainder_bounds": zero,
        "center_solution_endpoint": zero, "local_coordinate_error_bounds": (Q(0),)*18,
        "local_metric_error_upper_bound": Q(0), "endpoint_center": (Q(0),)*18,
        "endpoint_radius": radius,
        "endpoint": tuple(interval(-radius,radius) for _ in range(18)),
        "growth_certificate": None, "method": audit.METHOD,
        "conditional_premises": audit.PREMISES, "prescribed_time_interval": None,
    }


@pytest.fixture
def source_case(tmp_path):
    root = tmp_path / "synthetic-repository"
    root.mkdir()
    attempt = root / "attempt"
    (attempt / "steps").mkdir(parents=True)
    protocol = root / "producing-protocol.json"
    caller = root / "caller-result.json"
    write_json_once(caller, {})
    names = sorted(audit.TARGET_NAMES | {"worker.py"})
    source_rows = []
    for name in names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("Synthetic source, never imported: "+name).encode())
        source_rows.append(file_receipt(path,label=name))
    archive = root / "source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_STORED) as z:
        for row in source_rows:
            z.write(root/row["path"], row["path"])
    (attempt/"source.zip").write_bytes(archive.read_bytes())
    reviews = []
    for i in range(2):
        name = f"review-{i:02d}.md"
        (root/name).write_text("Synthetic association only, not approval.\n")
        (attempt/name).write_bytes((root/name).read_bytes())
        reviews.append(file_receipt(root/name,label=name))
    write_json_once(root/"controls.json", {"passed":True,"reserved_reference_steps":0})
    (attempt/"controls.json").write_bytes((root/"controls.json").read_bytes())
    by_name = {row["path"]:row for row in source_rows}
    record = {
        "schema":"tnfr.formed-two-port-executable-protocol.v1",
        "policy":audit.declared_policy(), "sources":source_rows,
        "archive":file_receipt(archive,label="source.zip"), "reviews":reviews,
        "controls":file_receipt(root/"controls.json",label="controls.json"),
        "head":"0"*40, "worker_entry":by_name["worker.py"],
    }
    write_json_once(protocol, record)
    (attempt/"protocol.json").write_bytes(protocol.read_bytes())
    source = {
        "protocol":file_receipt(attempt/"protocol.json",label="protocol.json"),
        "archive":file_receipt(attempt/"source.zip",label="source.zip"),
        "head":record["head"], "worker_entry":record["worker_entry"],
        "target_evidence":[by_name[n] for n in sorted(audit.TARGET_NAMES)],
    }
    reference = {
        "degrees":audit.DEGREES, "oriented_edges":audit.EDGES,
        "metric":audit.METRIC,"initial_center":(Q(0),)*18,
        "initial_radius":Q(0),"reference_margin":Q(1,2048),
    }
    admission = {
        "source":source, "reference":reference, "bounds":{},
        "target":{"evidence_receipts":source["target_evidence"],"target_deviations":tuple(interval() for _ in range(18))},
        "actual_source_is_flow_image_family":True,"no_actual_state_reset":True,
    }
    write_json_once(attempt/"admission.json", admission)
    write_json_once(attempt/"worker-start.json", {"schema":"tnfr.formed-two-port-attempt.v1","policy":audit.declared_policy()})
    instance = audit.Audit(attempt, protocol, caller)
    return instance, source


def append_step(instance, source, index=0, previous=None):
    cert = certificate(index)
    common_path = instance.attempt/"steps/certificate-fields.json"
    if not common_path.exists():
        write_json_once(common_path,{"schema":"tnfr.formed-two-port-certificate-fields.v1", "fields":{k:cert[k] for k in ("metric","coordinate_projection_factors")}})
    common = file_receipt(common_path,label="steps/certificate-fields.json")
    values = {k:v for k,v in cert.items() if k not in ("metric","coordinate_projection_factors")}
    name=f"steps/{index:04d}.json"
    write_json_once(instance.attempt/name, {
        "schema":"tnfr.formed-two-port-full-step.v1", "index":index,
        "source":source,"previous_complete_step":previous,
        "domain_values":"strict excess above rho, not raw acute margins",
        "shared_certificate_fields":common,"certificate":values,
    })
    return file_receipt(instance.attempt/name,label=name)


def test_real_source_reader_accepts_nested_json_edges_and_exact_associations(source_case):
    instance, source=source_case
    instance.inventory()
    instance.sources()
    assert instance.source == source
    assert instance.summary["source_associations_verified"]
    assert not instance.summary["complete_reference_certificate_admitted"]


def test_exact_certificate_consistency_keeps_nonzero_radius():
    value=Q(1,16)
    center,radius=audit.check_certificate(certificate(radius=value),0,(Q(0),)*18,value,(Q(1),)*18)
    assert radius == value and center == (Q(0),)*18


@pytest.mark.parametrize("change", [
    {"initial_radius":False}, {"time":Q(1,4)},
    {"endpoint_radius":Q(-1)}, {"domain_lower_bounds":(Q(0),)*20},
    {"local_coordinate_error_bounds":(Q(1),)*18},
    {"endpoint":tuple(interval(1,2) for _ in range(18))},
])
def test_certificate_rejects_coercion_chain_and_interval_loss(change):
    record=certificate(); record.update(change)
    with pytest.raises(ValueError):
        audit.check_certificate(record,0,(Q(0),)*18,Q(0),(Q(1),)*18)


def test_partial_next_json_keeps_the_verified_prefix_and_original_bytes(source_case):
    instance,source=source_case
    first=append_step(instance,source)
    partial=instance.attempt/"steps/0001.json"
    partial.write_bytes(b'{"partial":')
    result=instance.run()
    assert result["verified_prefix_steps"] == 1
    assert result["last_verified_step"] == first
    assert not result["complete_reference_certificate_admitted"] and not result["non_entry_proved"]
    assert partial.read_bytes() == b'{"partial":'


def test_changed_previous_receipt_stops_before_the_second_step(source_case):
    instance,source=source_case
    first=append_step(instance,source)
    wrong=dict(first,sha256="0"*64)
    append_step(instance,source,1,wrong)
    result=instance.run()
    assert result["verified_prefix_steps"] == 1
    assert any("hash chain" in message for message in result["issues"])


def test_zero_step_failure_never_claims_nonentry(source_case):
    instance,_=source_case
    write_json_once(instance.attempt/"worker-result.json",{"status":"unavailable","completed_steps":0,"reason":"synthetic pre-step failure","non_entry_not_proved":True})
    result=instance.run()
    assert result["verified_prefix_steps"] == 0
    assert not result["complete_reference_certificate_admitted"] and not result["non_entry_proved"]
    assert result["worker-result.json"]["reported_values_not_accepted_as_certificate"]["completed_steps"] == 0


def test_per_json_documentary_cap_does_not_expand_the_scientific_budget(source_case,monkeypatch):
    instance,source=source_case
    append_step(instance,source)
    original=instance.read
    def small_read(path,limit=audit.JSON_LIMIT):
        return original(path, min(limit,100) if Path(path).name == "0000.json" else limit)
    monkeypatch.setattr(instance,"read",small_read)
    result=instance.run()
    assert result["verified_prefix_steps"] == 0
    assert not result["non_entry_proved"]


def test_endpoint_scalar_cannot_hide_a_distant_retained_target(source_case):
    instance,source=source_case
    last=append_step(instance,source)
    instance.inventory(); instance.sources()
    instance.center,instance.radius,instance.previous=(Q(0),)*18,Q(0),last
    instance.admission["target"]["target_deviations"]=tuple(interval(1,1) for _ in range(18))
    # Exercise this consistency predicate in isolation. The synthetic terminal
    # and direct fixture state are not a completed 4096-step proof.
    endpoint={"certified":True,"conditional_on_acute_reference_prefix":True,
              "reasons":[],"form_norm_observation":"||P_D x||_D relative to the conserved uniform form",
              "reference_distance_upper_bound":Q(0),"reference_distance_margin":Q(1,2048)}
    write_json_once(instance.attempt/"worker-result.json",{
        "source":source,"last_complete_step":last,"completed_steps":4096,
        "status":"certified_capture","conditional_on_successful_post_write_supervisor_return":True,
        "step_evidence_reserved_bytes":instance.summary["step_bytes"],"endpoint":endpoint})
    with pytest.raises(ValueError,match="not linked"):
        instance.terminal()


@pytest.mark.parametrize("gate",[None,"prefix","terminal","caller_result"])
def test_success_requires_every_remaining_gate_after_real_source_admission(source_case,monkeypatch,gate):
    instance,_=source_case
    called=[]
    # This control tests only orchestration. All scientific/prefix conclusions
    # are explicit stubs, not records offered as reference evidence.
    for name in ("prefix","terminal","caller_result"):
        def stub(name=name):
            called.append(name)
            if name == gate:
                raise audit.DocumentaryUnavailable("synthetic gate unavailable")
        monkeypatch.setattr(instance,name,stub)
    result=instance.run()
    assert result["source_associations_verified"]
    assert result["complete_reference_certificate_admitted"] is (gate is None)
    assert called == ["prefix","terminal","caller_result"][:3 if gate is None else ["prefix","terminal","caller_result"].index(gate)+1]


@pytest.mark.parametrize("change",[{"wrapper_exit_code":2},{"wrapper_timed_out":True},{"stderr_bytes":1},{"output_error":"truncated"}])
def test_external_failure_cannot_be_promoted_by_a_return_flag(source_case,change):
    instance,_=source_case
    envelope={"wrapper_exit_code":0,"wrapper_timed_out":False,"stderr_bytes":0,"output_error":None,
              "stdout_complete":True,"stderr_complete":True,"capture_errors":[],
              "launcher_return":{"execution_passed":True}}
    envelope.update(change)
    instance.caller.write_text(audit.json_dumps(envelope))
    with pytest.raises(ValueError):
        instance.caller_result()
