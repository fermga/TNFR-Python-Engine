"""Compare admitted constitutive laws against an immutable receiver reference.

This read-only consumer rebuilds source, residual, growth and winding bounds.
It neither evolves a trajectory nor regenerates the original Taylor evidence.
The preserved reference's Picard and derivative enclosures remain numerical
premises; byte associations do not authenticate their execution or chronology.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from ..mathematics._rational_interval import INTERVAL_BITS, I
from ..mathematics._validated_metric import _metric_error_upper
from ..physics._constitutive_receiver_source import _assess_constitutive_receiver_source
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    exact_record,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)

_STEM = "docs/assets/sine_formed_classes/organized-receiver-enclosure-v1"
_RESPONSE_BYTES = 741711
_RESPONSE_SHA256 = "77b38c4785a57a355aa0d86a318b7acd22702e69de317687e93222ab5758d044"
_SUMMARY_BYTES = 189956
_SUMMARY_SHA256 = "2dbb6e1b738f33f1189ba0de1436b5682c8109000d0ea400250d9a7ae763b79c"
_REFERENCE_BASE = "3e5f552bc5b8f7e8a73e2469f71bf9305032c7ef"
_METRIC = tuple(tuple(Q(2 * int(i == j)) for j in range(13)) for i in range(13))
_H = Q(1, 8)
_COUNT = 76
_DEFORMATION = Q(1, 10000)
_SOURCE_ERROR = Q(1, 10**10)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _joint_verdict(left, right):
    """Retain a known counterexample even when another interval is unavailable."""
    if left is False or right is False:
        return False
    return None if left is None or right is None else True


def _interval(record):
    _require(type(record) is dict and set(record) == {"lo", "hi"}, "interval record")
    low, high = exact_record(record["lo"]), exact_record(record["hi"])
    _require(low <= high, "reversed interval")
    return I(low, high)


def _rows(values, size, convert):
    _require(type(values) is tuple and len(values) == size, "ordered record dimension")
    return tuple(convert(value) for value in values)


@dataclass(frozen=True)
class _RetainedReferenceStep:
    time: Q
    duration: Q
    initial_center: tuple[Q, ...]
    reference_tube: tuple[I, ...]
    reference_endpoint: tuple[I, ...]
    endpoint_center: tuple[Q, ...]


def _admit_reference_step(record, *, index, initial_center):
    """Rebuild consumed endpoint arithmetic, retaining the recorded Taylor premise."""
    _require(type(record) is dict, "step record")
    _require(
        type(record["step_index"]) is int and record["step_index"] == index,
        "step order",
    )
    step = record["reference_step"]
    time = exact_record(step["time"])
    duration = exact_record(step["duration"])
    _require(time == (index - 1) * _H and duration == _H, "reference time or duration")
    _require(type(step["order"]) is int and step["order"] == 8, "reference order")
    metric = _rows(step["metric"], 13, lambda row: _rows(row, 13, exact_record))
    _require(metric == _METRIC, "reference metric")
    center = _rows(step["initial_center"], 13, exact_record)
    _require(center == initial_center, "reference center seam")
    _require(exact_record(step["initial_radius"]) == 0, "reference initial radius")
    _require(
        exact_record(step["picard_interior_margin"]) > 0, "recorded Picard premise"
    )
    _require(
        _rows(step["domain_lower_bounds"], 1, exact_record) == (Q(1),),
        "reference domain",
    )
    tube = _rows(step["tube"], 13, _interval)
    _require(
        all(value.lo <= point <= value.hi for value, point in zip(tube, center)),
        "reference tube excludes its initial state",
    )
    polynomial = _rows(step["center_polynomial_bounds"], 13, _interval)
    remainder = _rows(step["local_remainder_bounds"], 13, _interval)
    endpoint = tuple(value + error for value, error in zip(polynomial, remainder))
    _require(
        endpoint == _rows(step["center_solution_endpoint"], 13, _interval),
        "reference endpoint reconstruction",
    )
    _require(
        all(max(a.lo, b.lo) <= min(a.hi, b.hi) for a, b in zip(endpoint, tube)),
        "reference endpoint misses its tube",
    )
    next_center = _rows(step["endpoint_center"], 13, exact_record)
    _require(
        next_center == tuple(value.midpoint for value in endpoint),
        "reference recentering",
    )
    return _RetainedReferenceStep(time, duration, center, tube, endpoint, next_center)


def _read_reference(root):
    """Read only the pinned transport, then admit its exact member inventory."""
    data = read_bytes_bounded(
        Path(root) / (_STEM + ".response.zip"), max_bytes=_RESPONSE_BYTES
    )
    _require(
        len(data) == _RESPONSE_BYTES and sha256_bytes(data) == _RESPONSE_SHA256,
        "receiver reference transport differs",
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        summary_data = archive.read(_STEM + ".json")
    _require(
        len(summary_data) == _SUMMARY_BYTES
        and sha256_bytes(summary_data) == _SUMMARY_SHA256,
        "reference summary differs",
    )
    summary = decode_exact_tree(json_loads(summary_data))
    _require(
        summary["schema"] == "tnfr.organized-receiver-enclosure-outcome.v1",
        "reference schema",
    )
    _require(summary["source_base_commit"] == _REFERENCE_BASE, "reference source")
    receipts = summary["step_receipts"]
    _require(
        type(receipts) is tuple and len(receipts) == _COUNT, "reference prefix length"
    )
    hashes = {_STEM + ".json": _SUMMARY_SHA256}
    for index, receipt in enumerate(receipts, 1):
        name = f"{_STEM}.steps/step-{index:04d}.json"
        _require(receipt["path"] == name, "reference receipt sequence")
        _require(
            type(receipt["bytes"]) is int and receipt["bytes"] > 0,
            "reference receipt size",
        )
        hashes[name] = receipt["sha256"]
    verify_archive_members(io.BytesIO(data), hashes, max_bytes=10 * 1024**2)
    records = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for receipt in receipts:
            content = archive.read(receipt["path"])
            _require(len(content) == receipt["bytes"], "reference member size")
            records.append(decode_exact_tree(json_loads(content)))
    return tuple(records)


def constitutive_receiver_comparison_policy():
    """Return the fixed retained-prefix comparison, not a new trajectory protocol."""
    return {
        "deformation_bound": _DEFORMATION,
        "form_error_bound": _SOURCE_ERROR,
        "phase_error_bound": _SOURCE_ERROR,
        "release_time": Q(400),
        "reference_step": _H,
        "reference_steps": _COUNT,
        "elapsed_slow_prefix": _COUNT * _H,
        "original_prospective_maximum_horizon": Q(32),
        "cones_per_reference_step": 8,
        "interval_grid_bits": INTERVAL_BITS,
        "arithmetic": "exact_rationals_with_outward_dyadic_intervals",
        "initial_phase_error_budget": Q(21, 100000),
        "velocity_defect_budget": Q(11, 10**6),
        "reference_response_sha256": _RESPONSE_SHA256,
        "reference_source_base": _REFERENCE_BASE,
        "trajectory_evaluated": False,
        "scope": "conditional_uniform_source_and_residual_comparison_on_retained_reference",
    }


def compare_constitutive_receiver_prefix(root):
    """Rebuild uniform comparison for the original source without executing it.

    No cached source, local growth rate, error radius or winding verdict from
    the original response enters the changed-law calculation. Every local
    reference starts at its stored rational center; its endpoint recentering
    error is added once, with carried full-dimensional error never reset.
    """
    from ..physics._constitutive_receiver_comparison import (
        _compare_constitutive_receiver_step,
    )

    policy = constitutive_receiver_comparison_policy()
    source = _assess_constitutive_receiver_source(
        deformation_bound=_DEFORMATION,
        form_error_bound=_SOURCE_ERROR,
        phase_error_bound=_SOURCE_ERROR,
    )
    _require(
        source.status == "certified_uniform_constitutive_source",
        "own-law source unavailable",
    )
    _require(
        source.release_phase_error_norm_upper_bound
        < policy["initial_phase_error_budget"],
        "own-law source exceeds phase budget",
    )
    _require(
        source.uniform_velocity_defect_upper_bound < policy["velocity_defect_budget"],
        "own-law source exceeds velocity budget",
    )
    records = _read_reference(root)
    prepared = source.reference_phase_bounds[:9] + source.reference_phase_bounds[18:22]
    center = tuple(value.midpoint for value in prepared)
    rounding = _metric_error_upper(_METRIC, tuple(value.radius for value in prepared))
    radius = policy["initial_phase_error_budget"]
    _require(
        I(source.release_phase_error_norm_upper_bound + rounding).hi < radius,
        "source and reference rounding exceed phase budget",
    )
    admitted = []
    for index, record in enumerate(records, 1):
        step = _admit_reference_step(record, index=index, initial_center=center)
        admitted.append(step)
        center = step.endpoint_center
    steps = []
    receiver_zero = source.early_receiver_zero_winding_certified
    for step in admitted:
        result = _compare_constitutive_receiver_step(
            initial_center=step.initial_center,
            reference_tube=step.reference_tube,
            reference_endpoint=step.reference_endpoint,
            endpoint_center=step.endpoint_center,
            incoming_error=radius,
            deformation_bound=_DEFORMATION,
            velocity_defect_bound=policy["velocity_defect_budget"],
            duration=step.duration,
        )
        steps.append(result)
        if result.status != "certified_local_comparison":
            receiver_zero = _joint_verdict(receiver_zero, None)
            break
        radius = result.endpoint_error
        receiver_zero = _joint_verdict(receiver_zero, result.receiver_zero_throughout)
    complete = len(steps) == _COUNT and steps[-1].status == "certified_local_comparison"
    donor_loss = (
        steps[-1].donor_endpoint_nonunit
        if complete and source.release_windings == (1, 1, 0)
        else None
    )
    return {
        "schema": "tnfr.constitutive-receiver-comparison.v1",
        "policy": policy,
        "source": source,
        "initial_reference_rounding_upper_bound": rounding,
        "steps": tuple(steps),
        "complete_prefix_certified": complete,
        "receiver_zero_throughout": receiver_zero,
        "donor_loss_at_endpoint": donor_loss,
        "endpoint_phase_error": radius if complete else None,
        "trajectory_evaluated": False,
        "status": (
            "certified_uniform_donor_loss_and_receiver_zero"
            if donor_loss and receiver_zero
            else (
                "certified_uniform_receiver_zero_donor_outcome_unavailable"
                if complete and receiver_zero and donor_loss is None
                else (
                    "certified_uniform_receiver_zero_joint_donor_loss_excluded"
                    if complete and receiver_zero and donor_loss is False
                    else "comparison_outcome_unavailable"
                )
            )
        ),
        "numerical_premises": (
            "retained_reference_Picard_and_Taylor_derivative_enclosures",
            "fixed_source_support_capacity_clock_and_complete_own_law",
            "no_physical_identification_or_new_reserved_response",
        ),
    }
