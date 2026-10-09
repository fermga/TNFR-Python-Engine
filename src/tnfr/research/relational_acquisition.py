"""Read-only reconstruction of a retained known-source P2 acquisition.

No producer, graph, pressure refresh or evolution is invoked. Record consistency
and the original precision decision are separate from execution authentication,
current-source replay, blind calibration and physical model admission.
"""

from __future__ import annotations

import argparse
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from .._exact_time import finite_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from ..physics.relational_observations import (
    RelationalCoefficientSampleBounds,
    bound_relational_coefficient_from_samples,
)
from ..sdk.relational_reports import relational_report_to_dict
from ..utils.io import json_dumps, json_loads
from .artifact_io import _require, encode_exact_tree
from .artifact_io import exact_record as _exact
from .artifact_io import read_bytes_bounded, sha256_bytes, verify_archive_members

__all__ = (
    "RelationalAcquisitionAudit",
    "audit_relational_coefficient_record",
    "audit_relational_coefficient_acquisition",
)

_MAX_STEPS = 4096
_MAX_RECORD_BYTES = 32 * 1024**2
_MAX_ARCHIVE_BYTES = 128 * 1024**2


def _pair(raw, *, represented=False):
    _require(isinstance(raw, (list, tuple)) and len(raw) == 2, "expected a pair")
    if represented:
        _require(all(type(value) in (int, float) for value in raw), "invalid raw state")
        values = tuple(
            finite_represented_real(value, "recorded state")[1] for value in raw
        )
        _require(
            values == tuple(map(Q, raw)), "raw state loses represented information"
        )
        return values
    return tuple(map(_exact, raw))


def _encoded(value):
    return json_dumps(value, sort_keys=True, allow_nan=False, default=encode_exact_tree)


def _same(actual, expected, label):
    _require(_encoded(actual) == _encoded(expected), f"recorded {label} differs")


@dataclass(frozen=True)
class RelationalAcquisitionAudit:
    """Detached consistency result; an original negative verdict stays negative."""

    status: str
    completed_steps: int | None = None
    acquisition_complete: bool | None = None
    recorded_passed: bool | None = None
    reconstructed_passed: bool | None = None
    sample_bounds: RelationalCoefficientSampleBounds | None = None
    unavailable_reasons: tuple[str, ...] = ()
    scope: tuple[str, ...] = (
        "known_source_unit_P2_equal_capacity_equal_two_channel_weights_v1",
        "recorded_contrast_states_rates_clock_and_error_chain_only",
        "file_audit_checks_protocol_and_archive_bytes_without_extracting_sources",
        "consistency_is_not_original_experiment_success_or_execution_authentication",
        "no_producer_evolution_current_source_replay_blind_calibration_or_physical_admission",
    )

    @property
    def consistent(self) -> bool:
        """Whether supplied evidence agrees internally, irrespective of its verdict."""
        return self.status == "consistent"

    def to_dict(self):
        """Project exact sample bounds through the existing report exporter."""
        return {
            "status": self.status,
            "completed_steps": self.completed_steps,
            "acquisition_complete": self.acquisition_complete,
            "recorded_passed": self.recorded_passed,
            "reconstructed_passed": self.reconstructed_passed,
            "sample_bounds": (
                relational_report_to_dict(self.sample_bounds)
                if self.sample_bounds is not None
                else None
            ),
            "unavailable_reasons": list(self.unavailable_reasons),
            "scope": list(self.scope),
        }


def _window(protocol):
    """Recheck the v1 proof premises independently of any producer defaults."""
    _require(
        protocol["protocol"] == "relational-p2-coefficient-acquisition-v1",
        "unsupported protocol",
    )
    for name, value in (
        ("nodes", (0, 1)),
        ("edges", ((0, 1),)),
        ("capacity", (Q(1), Q(1))),
        ("effective_weights", (Q(1, 2), Q(1, 2))),
        ("initial_phase", (Q(0), Q(0))),
        ("gamma", "absent"),
        ("events", "none; support and capacity held; no clipping or controller"),
        ("observation", "exact represented EPI[0]-EPI[1]; gain one; baseline zero"),
        ("clock", "relative structural time k*euler_step; exact dyadic grid"),
        ("pressure_refresh", "native field at every step and endpoint"),
    ):
        _same(protocol[name], value, name)
    for name, value in (
        ("pressure_path", "fused_canonical"),
        ("integrator", "native simultaneous explicit Euler"),
        ("interval_method", INTERVAL_METHOD),
        (
            "precision",
            "binary64 native states; exact Fraction defects; outward dyadic128 oracle",
        ),
    ):
        _same(protocol["runtime"][name], value, name)
    u, r = _pair(protocol["prior_rectangle"])
    _require((u, r) == (Q(1, 8), Q(1, 256)), "unsupported v1 rectangle")
    _require(_pair(protocol["prior_beta"]) == (Q(1, 2), Q(2)), "unsupported beta prior")
    _require(_exact(protocol["source_beta"]) == 1, "unsupported known source")
    _same(protocol["initial_form"], (u / 2, -u / 2), "initial form")
    step, h, horizon = map(
        _exact, (protocol["euler_step"], protocol["sample_step"], protocol["horizon"])
    )
    count, indices = protocol["step_count"], protocol["sample_indices"]
    _require(
        type(count) is int and 2 <= count <= _MAX_STEPS and count % 2 == 0,
        "invalid step count",
    )
    _require(
        0 < step and Q(float(step)) == step and horizon == 2 * h == count * step,
        "invalid represented sample grid",
    )
    _same(indices, (0, count // 2, count), "sample indices")
    _require(_exact(protocol["precision_limit"]) > 0, "invalid precision budget")
    lower = 1 - r**2 / 6
    q0, q1 = 1 / lower, r / (3 * lower**2)
    p, speed = u + r / 3, Q(2, 3) * u * q0
    u2, phase2 = p + speed / 3, Q(2, 3) * (p * q0 + u * q1 * speed)
    _require(
        horizon * speed < r and 2 * horizon < 1, "whole-window domain is unresolved"
    )
    _require(max(Q(4, 3), Q(2, 3) * (q0 + u * q1)) <= 2, "invalid Lipschitz bound")
    expected = {
        "sinc_lower": lower,
        "inverse_sinc_upper": q0,
        "inverse_sinc_derivative_upper": q1,
        "form_speed_upper": p,
        "phase_speed_upper": speed,
        "second_derivative_upper": max(u2, phase2),
        "third_form_derivative_upper": u2 + phase2 / 3,
        "lipschitz_upper": Q(2),
        "growth_factor_upper": 1 / (1 - 2 * horizon),
    }
    _same(protocol["window_bounds"], expected, "whole-window bounds")
    return step, h, horizon, count, expected


def _state(frame):
    form, phase = _pair(frame["epi"], represented=True), _pair(
        frame["phase"], represented=True
    )
    contrast = form[0] - form[1], phase[0] - phase[1]
    _require(
        abs(contrast[0]) <= Q(1, 8) and abs(contrast[1]) <= Q(1, 256),
        "state outside proof rectangle",
    )
    return contrast


def _ideal_field(state):
    # Independent pointwise observation, never a reduced evolution solver.
    u, phase = map(I, state)
    square = phase**2
    low = 1 - square / 6
    high = low + square**2 / 120
    sinc = I(low.lo, high.hi)
    pi = pi_interval()
    return -u - phase / pi, u / (pi * sinc)


def _failure_metadata(record, frame_count, count):
    _require({"stopped", "analysis_error"} <= set(record), "missing failure metadata")
    stopped, analysis_error = record["stopped"], record["analysis_error"]
    for failure in (stopped, analysis_error):
        if failure is not None:
            _require(
                isinstance(failure, Mapping)
                and isinstance(failure.get("error_type"), str)
                and bool(failure["error_type"])
                and isinstance(failure.get("error"), str),
                "invalid failure metadata",
            )
    if frame_count == count + 1:
        _require(stopped is None, "full acquisition cannot contain a stop")
    else:
        _require(stopped is not None, "partial acquisition requires a stop")
        _require(analysis_error is None, "incomplete acquisition cannot have analysis")
        if frame_count == 0:
            _require(
                set(stopped) == {"stage", "error_type", "error"}
                and stopped["stage"] == "before_acquisition",
                "empty acquisition requires a preparation failure",
            )
        else:
            _require(
                set(stopped) == {"attempted_step", "error_type", "error"}
                and type(stopped["attempted_step"]) is int
                and stopped["attempted_step"] == frame_count,
                "stop ordinal differs from the accepted prefix",
            )
    if analysis_error is not None:
        _require(
            set(analysis_error) == {"error_type", "error"},
            "invalid analysis failure metadata",
        )


def audit_relational_coefficient_record(
    record, *, protocol
) -> RelationalAcquisitionAudit:
    """Reconstruct a v1 response against its independently supplied frozen protocol.

    Contradictory or malformed evidence raises; partial or unavailable analysis
    retains no coefficient. This mapping-level check reads no artifact bytes.
    """
    _require(
        isinstance(record, Mapping) and isinstance(protocol, Mapping),
        "records require mappings",
    )
    _same(record["protocol"], protocol, "frozen protocol")
    step, h, horizon, count, bounds = _window(protocol)
    frames = record["frames"]
    _require(
        isinstance(frames, (list, tuple)) and len(frames) <= count + 1,
        "invalid frame count",
    )
    _failure_metadata(record, len(frames), count)
    for name in ("completed", "passed"):
        _require(type(record[name]) is bool, f"{name} requires a Boolean")
    if frames:
        _same(
            frames[0]["epi"], (float(Q(1, 16)), -float(Q(1, 16))), "initial nodal form"
        )
        _same(frames[0]["phase"], (0.0, 0.0), "initial phase consensus")
    field_sum = update_sum = Q(0)
    for index, frame in enumerate(frames):
        _require(
            type(frame["step"]) is int and frame["step"] == index,
            "frame ordinal differs",
        )
        _require(
            _exact(frame["time"]) == index * step, "clock differs from frozen grid"
        )
        _require(
            frame["pressure_path"] == protocol["runtime"]["pressure_path"],
            "pressure path differs",
        )
        state = _state(frame)
        if index:
            previous = _state(frames[index - 1])
            rates = _pair(frame["contrast_rates"])
            for axis, name in enumerate(("start_form_rate", "start_phase_rate")):
                raw_rates = _pair(frame[name], represented=True)
                _require(
                    rates[axis] == raw_rates[0] - raw_rates[1], "rate contrast differs"
                )
            ideal = _ideal_field(previous)
            field_error = _exact(frame["field_error_upper"])
            _require(
                field_error
                >= max(
                    max(abs(rate - value.lo), abs(rate - value.hi))
                    for rate, value in zip(rates, ideal, strict=True)
                ),
                "field error understated",
            )
            defects = tuple(
                new - old - step * rate
                for new, old, rate in zip(state, previous, rates, strict=True)
            )
            _require(_pair(frame["update_defects"]) == defects, "update defect differs")
            _require(_exact(frame["clock_defect"]) == 0, "nonzero clock defect")
            field_sum += step * field_error
            update_sum += max(map(abs, defects))
            truncation = index * bounds["second_derivative_upper"] * step**2 / 2
            for name, value in (
                ("accumulated_field_error", field_sum),
                ("accumulated_update_error", update_sum),
                ("accumulated_truncation_error", truncation),
            ):
                _require(_exact(frame[name]) == value, f"{name} differs")
        else:
            truncation = Q(0)
        _require(
            _exact(frame["continuous_contrast_error_upper"])
            == (field_sum + update_sum + truncation) / (1 - 2 * horizon),
            "continuous error bound differs",
        )
    complete = len(frames) == count + 1 and record["stopped"] is None
    _require(record["completed"] is complete, "completion flag differs")
    sample_bounds = None
    covered = precise = None
    reasons = ()
    if not complete or record["analysis_error"] is not None:
        _require(
            record["sample_analysis"] is None,
            "unavailable acquisition contains analysis",
        )
        reasons = (
            "acquisition_incomplete" if not complete else "sample_analysis_unavailable",
        )
    else:
        selected = tuple(frames[index] for index in protocol["sample_indices"])
        sample_bounds = bound_relational_coefficient_from_samples(
            tuple(_state(frame)[0] for frame in selected),
            sample_step=h,
            sample_error_bound=max(
                _exact(frame["continuous_contrast_error_upper"]) for frame in selected
            ),
            third_derivative_bound=bounds["third_form_derivative_upper"],
        )
        _same(
            record["sample_analysis"],
            relational_report_to_dict(sample_bounds),
            "sample analysis",
        )
        coefficient = sample_bounds.jet.coefficient_bounds
        if coefficient is not None:
            covered = coefficient[0] <= 1 <= coefficient[1]
            precise = coefficient[1] - coefficient[0] <= _exact(
                protocol["precision_limit"]
            )
        else:
            reasons = sample_bounds.jet.unavailable_reasons
    _require(
        record["contains_predeclared_truth"] is covered, "truth-coverage flag differs"
    )
    _require(record["width_within_budget"] is precise, "precision flag differs")
    passed = complete and covered is True and precise is True
    _require(record["passed"] is passed, "original verdict differs")
    return RelationalAcquisitionAudit(
        "consistent",
        max(0, len(frames) - 1),
        complete,
        record["passed"],
        passed,
        sample_bounds,
        reasons,
    )


def _read(path):
    return read_bytes_bounded(path, max_bytes=_MAX_RECORD_BYTES)


def _verify_archive(path, manifest):
    # Historical readers and archived helpers import this private signature.
    verify_archive_members(path, manifest, max_bytes=_MAX_ARCHIVE_BYTES)


def audit_relational_coefficient_acquisition(
    response_path,
) -> RelationalAcquisitionAudit:
    """Read a response and sibling protocol/archive; preserve all original bytes.

    Missing files return unavailable. Invalid JSON, digests or numerical evidence
    return inconsistent. A consistent result can retain a failed experiment.
    Current installed source need not match the historical source archive.
    """
    path = Path(response_path)
    frozen, archive = path.with_suffix(".protocol.json"), path.with_suffix(
        ".source.zip"
    )
    missing = tuple(str(item) for item in (path, frozen, archive) if not item.is_file())
    if missing:
        return RelationalAcquisitionAudit(
            "unavailable", unavailable_reasons=("missing_artifacts", *missing)
        )
    try:
        response_bytes, protocol_bytes = _read(path), _read(frozen)
        record, protocol = json_loads(response_bytes), json_loads(protocol_bytes)
        _require(isinstance(record, Mapping), "response must be an object")
        _require(
            record["protocol_sha256"] == sha256_bytes(protocol_bytes),
            "protocol digest differs",
        )
        _require(
            record["source_archive_sha256"] == sha256_bytes(_read(archive)),
            "archive digest differs",
        )
        _verify_archive(archive, protocol["source_sha256"])
        return audit_relational_coefficient_record(record, protocol=protocol)
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        ArithmeticError,
        RuntimeError,
        zipfile.BadZipFile,
    ) as error:
        return RelationalAcquisitionAudit(
            "inconsistent", unavailable_reasons=(str(error),)
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path)
    report = audit_relational_coefficient_acquisition(parser.parse_args(argv).response)
    print(json_dumps(report.to_dict(), sort_keys=True, allow_nan=False))
    return {"consistent": 0, "inconsistent": 1, "unavailable": 2}[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
