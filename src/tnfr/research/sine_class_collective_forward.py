"""Independent forward policy and reconstruction of its inspected prediction.

No function here executes a causal coefficient producer or complete-law flow.
Retained coefficient generation and original acquisition are separate premises.
The forward response never receives a prediction as a numerical constraint.
"""

import io
import zipfile
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ..physics._sine_class_collective_interface import _bound_collective_interface
from ..physics._sine_class_port_prediction import _port_time_tails
from ..physics.relational_sine_class_cubic_response import _cubic_parameters
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    exact_record,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from .sine_class_collective_protocol import collective_prediction_inputs
from .sine_class_comparison_protocol import canonical_comparison_sources

PREDICTION_PATH = (
    "docs/assets/sine_formed_classes/class-collective-prediction-v1.evidence.zip"
)
PREDICTION_SHA256 = "01b443211665c027bcd5b86ace8a22a4e7ae051f34d172d5520644e6225ba16a"
_EPS, _G, _H, _A = Q(1, 10**32), Q(1, 3000), Q(1), Q(7, 10000)
_MEMBERS = {
    "design.json",
    "protocol.json",
    "source.zip",
    "freeze.json",
    "attempt.json",
    "outcome.json",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    _require(type(value) is int, "ordinary integer required")
    return value


def _pair(value):
    _require(
        isinstance(value, (tuple, list)) and len(value) == 2, "endpoint pair required"
    )
    lo, hi = map(exact_record, value)
    _require(lo <= hi, "ordered endpoints required")
    return lo, hi


def _interval(value):
    _require(
        isinstance(value, dict) and set(value) == {"lo", "hi"},
        "interval record required",
    )
    pair = _pair((value["lo"], value["hi"]))
    interval = I(*pair)
    _require(
        (interval.lo, interval.hi) == pair,
        "stored interval must lie on the declared grid",
    )
    return interval


def _series(values, size):
    _require(
        len(values) == 33 and all(len(row) == size for row in values),
        "degree32 series shape differs",
    )
    return tuple(tuple(map(_interval, row)) for row in values)


def _endpoint(series, index):
    result = series[-1][index]
    for row in reversed(series[:-1]):
        result = result * _H + row[index]
    return result


def _expand(pair, radius):
    return pair[0] - radius, pair[1] + radius


def collective_forward_sources():
    """Reuse the original two-class target and actual-family cover owner.

    This explicitly constructs absolute theta=Theta_k+y covers. The nominal
    reference is x=0, y=0; the actual-family residual radius stays separate.
    A Cartesian cover does not supply every family's zero-sum correlations.
    """
    return canonical_comparison_sources()


def collective_forward_inputs():
    """One prospectively fixed complete-law policy, not a response evaluation."""
    source = collective_forward_sources()
    return dict(
        initial_form_bounds=source["reference_form_bounds"],
        initial_phase_bounds=source["reference_phase_bounds"],
        port_impulse=(Q(0), _A, Q(0)),
        horizon=_H,
        time_step=Q(1, 16),
        order=12,
        max_steps=32,
    )


def collective_forward_policy():
    return dict(
        endpoint_radius=_EPS,
        gamma_upper=_G,
        linear_source_allowance=_EPS / (1 - 2 * _G * _H),
        reference_radius_ceiling=Q(1, 10**12),
        actual_prediction_resolution_allowance=Q(1, 10**10),
        reading_error_per_model=Q(1, 10**8),
        nominal_prediction_comparison="closed_interval_overlap_without_intersection",
        actual_prediction_comparison="whole_forward_band_in_prediction_plus_resolution_allowance",
        separation="forward_recorded_lower_strictly_above_grounded_recorded_upper",
    )


@dataclass(frozen=True)
class CollectivePredictionBands:
    mediator_class: int
    nominal_cubic_bounds: tuple[Q, Q]
    nominal_full_bounds: tuple[Q, Q]
    actual_full_bounds: tuple[Q, Q]
    nominal_comparator_bounds: tuple[Q, Q]
    actual_comparator_bounds: tuple[Q, Q]
    numerical_radius: Q
    comparator_numerical_radius: Q


def _reconstruct_prediction_report(report, mediator_class):
    """Re-admit consumed primitives and rebuild bands from retained coefficients."""
    expected = collective_prediction_inputs(mediator_class)
    _require(
        _integer(report["mediator_class"]) == mediator_class, "prediction class differs"
    )
    for key in (
        "initial_form_bounds",
        "initial_phase_bounds",
        "comparator_initial_bounds",
    ):
        _require(
            tuple(map(_pair, report[key])) == expected[key], "prediction source differs"
        )
    _require(
        tuple(map(exact_record, report["port_impulse"])) == expected["port_impulse"],
        "prediction event differs",
    )
    _require(exact_record(report["horizon"]) == _H, "prediction clock differs")
    _require(_integer(report["order"]) == 32, "prediction order differs")
    _require(
        report["arithmetic_method"] == INTERVAL_METHOD, "prediction arithmetic differs"
    )
    _require(
        report["source_coordinates"]
        == "pre-input form x and phase deviation y=theta-Theta_k at 0-",
        "prediction phase coordinates differ",
    )
    descriptor = report["descriptor"]
    _require(
        tuple(map(_integer, descriptor["port_indices"])) == (4, 13, 22, 31, 40, 49),
        "prediction observer differs",
    )
    _require(
        tuple(map(_integer, descriptor["parameter_bounds"]["classes"]))
        == (1, mediator_class, 1),
        "prediction target class differs",
    )
    parameters = descriptor["parameter_bounds"]
    expected_parameters = _cubic_parameters(mediator_class)
    _require(
        tuple(map(_integer, parameters["degrees"])) == expected_parameters.degrees,
        "prediction support degrees differ",
    )
    for key in ("gamma", "eta"):
        _require(
            _interval(parameters[key]) == getattr(expected_parameters, key),
            "prediction complete-law coefficient differs",
        )
    for key in ("edge_sines", "edge_cosines"):
        _require(
            tuple(map(_interval, parameters[key])) == getattr(expected_parameters, key),
            "prediction target geometry differs",
        )
    coefficients = report["coefficients"]
    _require(
        len(coefficients["nominal_level_coefficients"]) == 3,
        "three amplitude levels required",
    )
    levels = tuple(
        _series(rows, 54) for rows in coefficients["nominal_level_coefficients"]
    )
    for degree, rows in enumerate(levels):
        initial = tuple(I(_A if degree == 0 and i == 13 else 0) for i in range(54))
        _require(rows[0] == initial, "nominal initialization differs")
    _require(
        all(row[i] == I(0) for row in levels[1] for i in (4, 13, 22, 31, 40, 49)),
        "nominal central parity differs",
    )
    comparator = _series(coefficients["comparator_nominal_coefficients"], 6)
    _require(
        comparator[0] == tuple(I(_A if i == 1 else 0) for i in range(6)),
        "comparator event differs",
    )
    tails = _port_time_tails(_A, _H)
    nominal = _endpoint(levels[0], 13) + _endpoint(levels[2], 13)
    nominal = nominal + I(-tails[0] - tails[2], tails[0] + tails[2])
    alternative = _endpoint(comparator, 1) + I(-tails[0], tails[0])
    # Stored endpoints/flags/tails are not used to produce either enclosure.
    _require(
        nominal == _interval(report["nominal_endpoint_bounds"][1]),
        "cached prediction endpoint differs",
    )
    _require(
        alternative == _interval(report["comparator_nominal_endpoint_bounds"][1]),
        "cached comparator endpoint differs",
    )
    fidelity = _bound_collective_interface(
        total_input_variation=_A, horizon=_H, endpoint_radius=_EPS
    )
    source = _EPS / (1 - 2 * _G * _H)
    band, other = (nominal.lo, nominal.hi), (alternative.lo, alternative.hi)
    _require(
        nominal.radius <= Q(1, 10**12) and alternative.radius <= Q(1, 10**12),
        "prediction numerical budget differs",
    )
    return CollectivePredictionBands(
        mediator_class,
        band,
        _expand(band, fidelity.nominal_fifth_order_error_upper_bound),
        _expand(band, fidelity.form_error_upper_bound + source),
        other,
        _expand(other, source),
        nominal.radius,
        alternative.radius,
    )


def read_collective_prediction(root):
    """Inspect the immutable prior packet; generate no coefficients or response.

    The exact packet association supplies the retained model/derivative
    execution premise. It does not authenticate that execution or acquisition.
    Every endpoint consumed below is reconstructed from its coefficients.
    """
    path = Path(root) / PREDICTION_PATH
    data = read_bytes_bounded(path, max_bytes=2 * 1024**2)
    _require(
        len(data) == 1016399 and sha256_bytes(data) == PREDICTION_SHA256,
        "prior prediction artifact differs",
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        _require(
            len(names) == len(_MEMBERS) and set(names) == _MEMBERS,
            "prediction inventory differs",
        )
        _require(
            sum(info.file_size for info in archive.infolist()) <= 64 * 1024**2,
            "prediction expansion budget exceeded",
        )
        hashes = {name: sha256_bytes(archive.read(name)) for name in names}
        outcome = decode_exact_tree(json_loads(archive.read("outcome.json")))
    verify_archive_members(io.BytesIO(data), hashes, max_bytes=64 * 1024**2)
    _require(
        outcome["status"] == "complete"
        and outcome["full_law_response"] == "not_evaluated",
        "prior prediction scope differs",
    )
    _require(len(outcome["reports"]) == 2, "two prior classes required")
    return tuple(
        _reconstruct_prediction_report(report, k)
        for k, report in zip((1, 2), outcome["reports"])
    )


def assess_collective_forward_bounds(reference_bounds, prediction):
    """Conditional scalar comparison after independent evidence reconstruction.

    Neither prediction nor comparator changes the supplied forward interval.
    `prediction` supplies exact primitive bands, not a trusted passing flag.
    A caller consuming a report must reconstruct it before this calculation.
    """
    reference = _pair(reference_bounds)
    nominal = _pair(prediction.nominal_full_bounds)
    actual_prediction = _pair(prediction.actual_full_bounds)
    comparator = _pair(prediction.actual_comparator_bounds)
    policy = collective_forward_policy()
    actual = _expand(reference, policy["linear_source_allowance"])
    delta = policy["reading_error_per_model"]
    recorded, other_recorded = _expand(actual, delta), _expand(comparator, delta)
    radius = (reference[1] - reference[0]) / 2
    overlap = max(reference[0], nominal[0]) <= min(reference[1], nominal[1])
    window = _expand(
        actual_prediction, policy["actual_prediction_resolution_allowance"]
    )
    contained = window[0] <= actual[0] <= actual[1] <= window[1]
    margin = recorded[0] - other_recorded[1]
    narrow = radius <= policy["reference_radius_ceiling"]
    return dict(
        reference_bounds=reference,
        actual_full_bounds=actual,
        reference_radius=radius,
        reference_radius_within_budget=narrow,
        nominal_prediction_overlap=overlap,
        actual_resolution_window=window,
        actual_prediction_resolution_met=contained,
        recorded_full_bounds=recorded,
        recorded_comparator_bounds=other_recorded,
        recorded_separation_margin=margin,
        comparator_separated=margin > 0,
        all_conditions_met=narrow and overlap and contained and margin > 0,
        status=(
            "consistency_conflict"
            if not overlap
            else (
                "numerically_unresolved"
                if not narrow
                else (
                    "resolution_not_met"
                    if not contained
                    else (
                        "discrimination_unresolved"
                        if margin <= 0
                        else "all_conditions_met"
                    )
                )
            )
        ),
    )


def assess_collective_forward_report(report, prediction):
    """Re-admit a live producer report and reconstruct every consumed step.

    Incomplete evidence retains its audited prefix but has no paired verdict.
    Derivative/Picard generation remains an execution premise, not something
    proved by replaying endpoint arithmetic or reading a successful label.
    """
    from ..physics._sine_class_port_readout_evidence import _reconstruct_port_readout
    from ..physics.relational_sine_class_port_readout import _admit_port_readout_inputs

    _require(len(prediction) == 2, "two prediction classes required")
    for k, item in zip((1, 2), prediction):
        _require(_integer(item.mediator_class) == k, "prediction order differs")
    admitted = _admit_port_readout_inputs(**collective_forward_inputs())
    evidence = _reconstruct_port_readout(report, admitted)
    if not evidence.complete:
        return dict(
            status="unavailable",
            evidence=evidence,
            comparisons=None,
            all_conditions_met=False,
        )
    comparisons = tuple(
        assess_collective_forward_bounds((band.lo, band.hi), prior)
        for band, prior in zip(evidence.endpoint_bounds, prediction)
    )
    return dict(
        status=(
            "all_conditions_met"
            if all(row["all_conditions_met"] for row in comparisons)
            else "complete_with_unmet_conditions"
        ),
        evidence=evidence,
        comparisons=comparisons,
        all_conditions_met=all(row["all_conditions_met"] for row in comparisons),
    )
