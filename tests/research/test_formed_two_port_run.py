"""Non-transit controls of complete evidence and the fixed runner wiring."""

from dataclasses import asdict, dataclass, fields, replace
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_metric import ValidatedMetricTaylorStep
from tnfr.research import formed_two_port_run as owner
from tnfr.research.artifact_io import decode_exact_tree, encode_exact_tree, file_receipt
from tnfr.utils.io import json_loads


@pytest.fixture(autouse=True)
def no_real_reference_or_target(monkeypatch):
    from tnfr.mathematics import _validated_metric
    from tnfr.research import formed_two_port_reference

    def forbidden(*args, **kwargs):
        pytest.fail("Runner controls must not evaluate a real reference or target")

    monkeypatch.setattr(_validated_metric, "validated_metric_taylor_step", forbidden)
    for name in (
        "build_formed_two_port_reference",
        "admit_formed_two_port_target",
        "formed_two_port_comparison_bounds",
        "admit_formed_two_port_endpoint",
    ):
        monkeypatch.setattr(formed_two_port_reference, name, forbidden)


@dataclass
class SyntheticReference:
    metric: tuple = tuple(
        tuple(Q(2 if i == j else 0) for j in range(18)) for i in range(18)
    )
    initial_center: tuple = (Q(0),) * 18
    initial_radius: Q = Q(0)

    def field(self, values):
        pytest.fail("The synthetic wiring kernel must not call any field")

    def domain(self, values):
        pytest.fail("The synthetic wiring kernel must not evaluate a tube")


@dataclass
class SyntheticRecord:
    certified: bool = True
    evidence_receipts: tuple = ()


def synthetic_certificate(index=0):
    """Artificial serialization/wiring fixture, never a mathematical certificate."""
    point = (I(0),) * 18
    return ValidatedMetricTaylorStep(
        time=index * Q(1, 4),
        duration=Q(1, 4),
        order=8,
        metric=SyntheticReference().metric,
        initial_center=(Q(0),) * 18,
        initial_radius=Q(0),
        initial_box=point,
        coordinate_projection_factors=(Q(1),) * 18,
        tube=(I(-1, 1),) * 18,
        picard_interior_margin=Q(1, 8),
        domain_lower_bounds=(Q(1, 7),) * 20,
        growth_rate_upper_bound=Q(0),
        growth_factor_upper_bound=Q(1),
        propagated_initial_radius=Q(0),
        center_polynomial_bounds=point,
        local_remainder_bounds=point,
        center_solution_endpoint=point,
        local_coordinate_error_bounds=(Q(0),) * 18,
        local_metric_error_upper_bound=Q(0),
        endpoint_center=(Q(0),) * 18,
        endpoint_radius=Q(0),
        endpoint=point,
    )


def test_all_shared_certificate_fields_survive_exact_stream_roundtrip(tmp_path):
    writer = owner.EvidenceWriter(tmp_path)
    reference, certificate = SyntheticReference(), synthetic_certificate()
    certificate = replace(
        certificate,
        local_remainder_bounds=(I(Q(-1, 101), Q(2, 103)),) * 18,
        center_solution_endpoint=(I(Q(-3, 107), Q(4, 109)),) * 18,
        local_coordinate_error_bounds=(Q(5, 113),) * 18,
        local_metric_error_upper_bound=Q(6, 127),
        endpoint_center=(Q(7, 131),) * 18,
        endpoint_radius=Q(8, 137),
        endpoint=(I(Q(-9, 139), Q(10, 149)),) * 18,
    )
    shared = owner.shared_certificate_record(certificate, reference)
    receipt = writer.write("steps/certificate-fields.json", shared, step=True)
    record = owner.certificate_record(
        certificate,
        reference,
        0,
        reference.initial_center,
        reference.initial_radius,
        {"synthetic": True},
        None,
        shared=shared,
        shared_receipt=receipt,
    )
    written = writer.write("steps/0000.json", record, step=True)
    recovered = owner.read_step_certificate(tmp_path, 0)
    assert set(recovered) == {item.name for item in fields(ValidatedMetricTaylorStep)}
    assert len(recovered) == 26
    assert recovered == decode_exact_tree(encode_exact_tree(asdict(certificate)))
    assert set(record["certificate"]) == set(recovered) - {
        "metric",
        "coordinate_projection_factors",
    }
    assert written == file_receipt(
        tmp_path / "steps/0000.json", label="steps/0000.json"
    )
    assert writer.step_used == sum(
        p.stat().st_size for p in (tmp_path / "steps").iterdir()
    )


def test_shared_values_cannot_change_between_steps(tmp_path):
    reference, certificate = SyntheticReference(), synthetic_certificate()
    shared = owner.shared_certificate_record(certificate, reference)
    writer = owner.EvidenceWriter(tmp_path)
    receipt = writer.write("steps/certificate-fields.json", shared, step=True)
    altered = replace(certificate, coordinate_projection_factors=(Q(2),) * 18)
    with pytest.raises(ValueError, match="shared certificate field changed"):
        owner.certificate_record(
            altered,
            reference,
            0,
            reference.initial_center,
            reference.initial_radius,
            {},
            None,
            shared=shared,
            shared_receipt=receipt,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"time": True},
        {"duration": Q(1, 8)},
        {"order": True},
        {"growth_rate_upper_bound": False},
        {"growth_factor_upper_bound": Q(2)},
        {"picard_interior_margin": Q(0)},
        {"domain_lower_bounds": (Q(0),) * 20},
        {"initial_radius": Q(1)},
        {"method": "compact_or_unvalidated"},
        {"initial_radius": False},
        {"initial_center": (False,) * 18},
        {"endpoint_radius": 0.0},
        {"local_coordinate_error_bounds": (0.0,) * 18},
    ],
)
def test_step_policy_and_primitive_domains_are_not_coerced(change):
    reference, certificate = SyntheticReference(), synthetic_certificate()
    shared = owner.shared_certificate_record(certificate, reference)
    with pytest.raises((TypeError, ValueError)):
        owner.certificate_record(
            replace(certificate, **change),
            reference,
            0,
            reference.initial_center,
            reference.initial_radius,
            {},
            None,
            shared=shared,
            shared_receipt={},
        )


def test_budget_is_checked_before_open_and_existing_bytes_are_preserved(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(owner, "STEP_BYTES", 3)
    writer = owner.EvidenceWriter(tmp_path)
    writer.write("steps/a.json", {}, step=True)
    assert (tmp_path / "steps/a.json").read_bytes() == b"{}\n"
    with pytest.raises(owner.EvidenceBudgetExceeded):
        writer.write("steps/b.json", {}, step=True)
    assert not (tmp_path / "steps/b.json").exists()
    assert writer.step_used == 3
    with pytest.raises(FileExistsError, match="no resume"):
        owner.EvidenceWriter(tmp_path)


def test_auxiliary_reserves_leave_terminal_space_and_never_spend_parent_reserve(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(owner, "AUX_BYTES", 20)
    monkeypatch.setattr(owner, "SUPERVISOR_RESERVE", 2)
    monkeypatch.setattr(owner, "CALLER_RESERVE", 3)
    monkeypatch.setattr(owner, "TERMINAL_RESERVE", 5)
    writer = owner.EvidenceWriter(tmp_path)
    writer._write("data", b"x" * 10, step=False, terminal=False)
    with pytest.raises(owner.EvidenceBudgetExceeded):
        writer._write("overflow", b"x", step=False, terminal=False)
    writer._write("terminal", b"final", step=False, terminal=True)
    assert writer.aux_used == 15
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) == 15
    with pytest.raises(ValueError):
        writer.write("again", {}, terminal=True)


def test_partial_write_is_preserved_and_reserved_without_retry(tmp_path, monkeypatch):
    writer = owner.EvidenceWriter(tmp_path)

    def interrupted(path, callback, **kwargs):
        with Path(path).open("xb") as stream:
            stream.write(b'{"')
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(owner, "safe_write", interrupted)
    with pytest.raises(OSError):
        writer.write("steps/0000.json", {"record": 1}, step=True)
    assert (tmp_path / "steps/0000.json").read_bytes() == b'{"'
    assert writer.step_used > 2
    with pytest.raises(FileExistsError):
        owner.EvidenceWriter(tmp_path)


def test_receipt_copy_detects_change_after_the_first_receipt(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"old")
    row = file_receipt(path, label="input")
    original = owner._receipt

    def changed(root, receipt):
        result = original(root, receipt)
        path.write_bytes(b"new")
        return result

    monkeypatch.setattr(owner, "_receipt", changed)
    with pytest.raises(ValueError, match="consumed frozen bytes differ"):
        owner._receipt_data(tmp_path, row)


@pytest.mark.parametrize(
    "name", ["../escape", "steps/../../escape", "C:/outside", "steps/../other"]
)
def test_writer_rejects_redirected_or_noncanonical_paths(tmp_path, name):
    writer = owner.EvidenceWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.write(name, {}, step=name.startswith("steps/"))


def test_worker_rejects_missing_job_before_artifacts_or_science(tmp_path, monkeypatch):
    from tnfr.research import bounded_worker

    def not_bounded(**kwargs):
        raise OSError("no admitted Job")

    monkeypatch.setattr(bounded_worker, "require_current_job_limits", not_bounded)
    with pytest.raises(OSError, match="no admitted Job"):
        owner.run_reference_attempt(tmp_path, tmp_path / "absent.json", tmp_path)
    assert not list(tmp_path.iterdir())


def _wire_synthetic(monkeypatch, kernel):
    from tnfr.mathematics import _validated_metric
    from tnfr.research import bounded_worker, formed_two_port_reference

    target_receipts = tuple(
        {"path": f"synthetic-{i}", "bytes": i, "sha256": "test"} for i in range(4)
    )
    monkeypatch.setattr(
        bounded_worker,
        "require_current_job_limits",
        lambda **kwargs: {"synthetic": True},
    )
    monkeypatch.setattr(
        owner,
        "admit_execution_inputs",
        lambda *args: {"synthetic": True, "target_evidence": list(target_receipts)},
    )
    monkeypatch.setattr(
        formed_two_port_reference, "build_formed_two_port_reference", SyntheticReference
    )
    monkeypatch.setattr(
        formed_two_port_reference, "formed_two_port_comparison_bounds", SyntheticRecord
    )
    monkeypatch.setattr(
        formed_two_port_reference,
        "admit_formed_two_port_target",
        lambda *args: SyntheticRecord(evidence_receipts=target_receipts),
    )
    monkeypatch.setattr(_validated_metric, "validated_metric_taylor_step", kernel)


def test_failed_synthetic_step_preserves_exact_prefix_without_endpoint(
    tmp_path, monkeypatch
):
    calls = []

    def kernel(center, radius, metric, duration, flow, domain, **kwargs):
        calls.append(kwargs["time"])
        if len(calls) == 1:
            return synthetic_certificate(), None, None
        return None, (I(-2, 2),) * 18, "synthetic domain failure"

    _wire_synthetic(monkeypatch, kernel)
    assert (
        owner.run_reference_attempt(tmp_path, tmp_path / "synthetic.json", tmp_path)
        == 2
    )
    result = json_loads((tmp_path / "worker-result.json").read_bytes())
    assert result["completed_steps"] == 1 and result["status"] == "unavailable"
    assert calls == [Q(0), Q(1, 4)]
    assert {p.name for p in (tmp_path / "steps").iterdir()} == {
        "0000.json",
        "certificate-fields.json",
    }
    assert result["last_complete_step"] == file_receipt(
        tmp_path / "steps/0000.json", label="steps/0000.json"
    )
    assert (tmp_path / "failed-step.json").exists()


def test_full_schedule_wiring_has_no_early_success_and_no_retained_history(
    tmp_path, monkeypatch
):
    from tnfr.research import formed_two_port_reference

    calls = []
    terminals = []

    class CountingWriter:
        def __init__(self, directory):
            self.step_used, self.terminal_written = 0, False

        def write(self, name, payload, *, step=False, terminal=False):
            if step and name != "steps/certificate-fields.json":
                assert payload["index"] == self.step_used
                self.step_used += 1
            if terminal:
                self.terminal_written = True
                terminals.append(payload)
            return {"path": name, "bytes": 1, "sha256": "synthetic"}

    def kernel(center, radius, metric, duration, flow, domain, **kwargs):
        assert kwargs["time"] == len(calls) * Q(1, 4)
        calls.append(1)
        return synthetic_certificate(len(calls) - 1), None, None

    def endpoint(*args):
        assert len(calls) == 4096
        return SyntheticRecord()

    _wire_synthetic(monkeypatch, kernel)
    monkeypatch.setattr(owner, "EvidenceWriter", CountingWriter)
    monkeypatch.setattr(
        formed_two_port_reference, "admit_formed_two_port_endpoint", endpoint
    )
    assert (
        owner.run_reference_attempt(tmp_path, tmp_path / "synthetic.json", tmp_path)
        == 0
    )
    assert len(calls) == 4096 and len(terminals) == 1
    assert terminals[0]["completed_steps"] == 4096
    assert (
        terminals[0]["conditional_on_successful_post_write_supervisor_return"] is True
    )
