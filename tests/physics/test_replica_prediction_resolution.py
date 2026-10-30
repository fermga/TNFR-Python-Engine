"""Constructed controls for the retained observer's minimax resolution.

These controls do not load the reserved prediction cells, integrate a law or
evaluate the diagnostic runner. Artificial constant Taylor records test only
admission and stored arithmetic; they are not scientific flow certificates.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import replica_prediction_resolution as resolution
from tnfr.research import replica_synchronized_comparison as comparison
from tnfr.research import replica_synchronized_prediction as prediction
from tnfr.research.artifact_io import exact_record, file_receipt, write_json_once
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE = (
    ROOT / "artifacts/replica-internal-comparison-20261026/source-certificate.json"
)
SHIFTS = (-2, -1, 0, 1, 2)
CAP = Q(1, 5)


@pytest.fixture(scope="module")
def raw_allowance():
    return json_loads(CERTIFICATE.read_bytes())


@pytest.fixture(scope="module")
def allowances(raw_allowance):
    return prediction.admit_allowance_record(raw_allowance)


def _tube(phases):
    return (Q(0),) * 5 + tuple(phases)


def _twist():
    pi = pi_interval()
    return tuple(2 * (index - 2) * pi / 5 for index in range(5))


def _contact_phases():
    return (Q(0), Q(-1, 8), Q(3, 2), Q(25, 8), Q(19, 4))


def _oracle(tube):
    """Enumerate all 5**5 assignments, independently of the production DP."""
    pi = pi_interval()
    phases = tuple(I.coerce(value) for value in tube[5:])
    distances = []
    for index, phase in enumerate(phases):
        raw = phases[(index + 1) % 5] - phase
        row = []
        for shift in SHIFTS:
            interval = raw + 2 * shift * pi
            row.append(
                max(
                    Q(0), Q(1, 100) - interval.hi, interval.lo - (pi.hi / 2 - Q(1, 100))
                )
            )
        distances.append(tuple(row))
    candidates = tuple(
        (max(distances[i][shift + 2] for i, shift in enumerate(assignment)), assignment)
        for assignment in product(SHIFTS, repeat=5)
        if sum(assignment) == 1
    )
    threshold = min(value for value, _ in candidates)
    witnesses = tuple(
        assignment for value, assignment in candidates if value == threshold
    )
    return tuple(distances), threshold, witnesses


def _changed(raw, path, value):
    changed = deepcopy(raw)
    parent = changed
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return changed


def _constructed_records(allowances):
    """Supply constant polynomials, not derivative enclosures for the law."""
    source = prediction.declared_prediction_source()
    box = source.initial_box
    tube = tuple(I(value.lo - Q(1, 8), value.hi + Q(1, 8)) for value in box)
    zeros = (I(0),) * 10
    anchor = allowances.final_state_receipt or dict(
        path=prediction.FINAL_STATE_PATH, bytes=1, sha256="0" * 64
    )
    cells = []
    for index in range(48):
        error = prediction.allowance_for_cell(index, allowances)
        step = dict(
            time=Q(index, 8),
            duration=Q(1, 8),
            order=8,
            initial_box=tuple(asdict(value) for value in box),
            tube=tuple(asdict(value) for value in tube),
            series=tuple(
                tuple(asdict(value) for value in (initial,) + (I(0),) * 8)
                for initial in box
            ),
            local_remainder_bounds=tuple(asdict(value) for value in zeros),
            increment=tuple(asdict(value) for value in zeros),
            endpoint=tuple(asdict(value) for value in box),
            picard_interior_margin=Q(1, 16),
            domain_lower_bounds=(Q(1),),
            method="direct_source_box_Picard_Taylor_dyadic128_v1",
        )
        cells.append(
            dict(
                cell_index=index,
                step=step,
                allowance=error,
                observation=asdict(prediction.observe_prediction(tube, error)),
            )
        )
    return (
        prediction.prediction_attempt(anchor),
        asdict(source),
        asdict(allowances),
        cells,
    )


def _record(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_once(path, payload)
    return file_receipt(path, label=name)


def _source_archive(root, directory, names, *, wrong_archive=False):
    sources = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        sources.append(file_receipt(path, label=name))
    name = directory + "/evaluated-source.zip"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "x", compression=ZIP_DEFLATED) as archive:
        for index, row in enumerate(sources):
            data = (root / row["path"]).read_bytes()
            if wrong_archive and index == 0:
                data = b"# deliberately different archived production owner\n"
            archive.writestr(row["path"], data)
    return sources, file_receipt(path, label=name)


def _write_bound_prediction(
    root, raw_allowance, *, forged_source=False, wrong_archive=False
):
    """Associate artificial records with bytes, without claiming execution."""
    directory = prediction.PRIOR_DIRECTORY
    sources, archive = _source_archive(
        root, directory, prediction.REQUIRED_SOURCE_NAMES
    )
    certificate = _record(root, prediction.CERTIFICATE_PATH, raw_allowance)
    prior_records = [
        certificate,
        _record(
            root,
            directory + "/validation-plan.json",
            dict(source=sources, source_archive=archive),
        ),
        _record(root, directory + "/evaluation-result.json", dict(result=certificate)),
    ]
    anchor = _record(
        root,
        prediction.FINAL_STATE_PATH,
        dict(
            schema="tnfr.replica-internal-comparison-final.v1",
            artifacts=prior_records,
            evaluated_source=sources,
            evaluated_source_archive=archive,
        ),
    )
    allowance = prediction.read_allowances(root, final_state_receipt=anchor)
    attempt, source, allowance_record, cells = _constructed_records(allowance)
    if forged_source:
        source["memberwise_image_certified"] = False
    directory = resolution.PRIOR_DIRECTORY
    sources, archive = _source_archive(
        root, directory, resolution.REQUIRED_SOURCE_NAMES, wrong_archive=wrong_archive
    )
    records = [
        _record(
            root,
            directory + "/validation-plan.json",
            dict(source=sources, source_archive=archive),
        )
    ]
    for name, value in (
        ("attempt", attempt),
        ("source", source),
        ("allowances", allowance_record),
    ):
        records.append(_record(root, directory + f"/trajectory/{name}.json", value))
    observations = []
    for index, cell in enumerate(cells):
        records.append(
            _record(root, directory + f"/trajectory/cell-{index:03d}.json", cell)
        )
        tube = tuple(I(value["lo"], value["hi"]) for value in cell["step"]["tube"])
        observations.append(prediction.observe_prediction(tube, cell["allowance"]))
    result = _record(
        root,
        directory + "/trajectory/trajectory-result.json",
        prediction.prediction_outcome(observations),
    )
    records.extend(
        (
            result,
            _record(root, directory + "/execution-result.json", dict(result=result)),
        )
    )
    return _record(
        root,
        resolution.FINAL_STATE_PATH,
        dict(
            schema="tnfr.replica-reduced-prediction-final.v1",
            artifacts=records,
            evaluated_source=sources,
            evaluated_source_archive=archive,
        ),
    )


def test_minimax_matches_an_independent_exhaustive_assignment_oracle():
    pi = pi_interval()
    examples = (
        _tube(_twist()),
        _tube(_contact_phases()),
        _tube(tuple(value + Q(3, 8) for value in _contact_phases())),
        _tube(
            tuple(value + (2 * pi if i == 0 else 0) for i, value in enumerate(_twist()))
        ),
        _tube(
            tuple(I(value - Q(1, 32), value + Q(1, 16)) for value in _contact_phases())
        ),
    )
    for tube in examples:
        distances, threshold, witnesses = _oracle(tube)
        result = resolution.resolve_prediction(tube, CAP)
        assert result.available and result.shifts == SHIFTS
        assert result.distances == distances
        assert 0 <= threshold <= CAP
        assert result.threshold_status == "within_domain"
        assert result.threshold == threshold
        assert result.minimizing_assignment in witnesses
        assert sum(result.minimizing_assignment) == 1


def test_integer_sum_constraint_cannot_be_replaced_by_independent_edge_minima():
    tube = _tube((Q(0),) * 5)
    distances, constrained, _ = _oracle(tube)
    uncoupled = max(min(row) for row in distances)
    assert uncoupled == Q(1, 100) < CAP < constrained
    assert tuple(SHIFTS[row.index(min(row))] for row in distances) == (0,) * 5
    result = resolution.resolve_prediction(tube, CAP)
    assert result.available and result.distances == distances
    assert result.threshold_status == "beyond_domain"
    assert result.threshold is None and result.strict_lower_bound == CAP
    assert result.zero_allowance_status == "separated"
    assert resolution.resolution_status(result, CAP) == "separated"


def test_exact_closed_contact_is_unresolved_and_only_strictly_smaller_error_separates():
    tube = _tube(_contact_phases())
    contact = Q(27, 200)
    result = resolution.resolve_prediction(tube, CAP)
    assert result.available and result.threshold == contact
    assert result.raw_gaps[0] == I(Q(-1, 8))
    assert resolution.resolution_status(result, contact - Q(1, 2**40)) == "separated"
    assert resolution.resolution_status(result, contact) == "unresolved"
    assert resolution.resolution_status(result, contact + Q(1, 2**40)) == "unresolved"
    frozen_contact = prediction.observe_prediction(tube, contact)
    assert frozen_contact.status == "unresolved"
    assert frozen_contact.dilated_target_lower == Q(-1, 8)
    assert 1 in frozen_contact.reachable_sums
    with pytest.raises(FrozenInstanceError):
        result.threshold = Q(0)


def test_zero_threshold_describes_the_relaxed_observation_without_a_visit_claim():
    narrow = resolution.resolve_prediction(_tube(_twist()), CAP)
    broad = resolution.resolve_prediction(_tube((I(-3, 3),) * 5), CAP)
    for result in (narrow, broad):
        assert result.available and result.threshold_status == "within_domain"
        assert result.threshold == 0
        assert result.zero_allowance_status == "unresolved"
        assert resolution.resolution_status(result, Q(0)) == "unresolved"
        assert resolution.resolution_status(result, CAP) == "unresolved"
        assert sum(result.minimizing_assignment) == 1
        assert not result.exact_state_existence_claimed
        assert not result.actual_error_measured and not result.cause_identified
        assert not result.restricted_minimum_claimed_global_outside_domain
    # Broad independent marginal intervals deliberately discard joint phase
    # correlations. A zero distance is a reader limitation, not a trajectory.
    assert all(value.lo == -6 and value.hi == 6 for value in broad.raw_gaps)


def test_beyond_cap_is_censored_and_missing_branch_domains_are_unavailable():
    tube = _tube(_contact_phases())
    low_cap = Q(1, 10)
    censored = resolution.resolve_prediction(tube, low_cap)
    assert censored.available and censored.threshold_status == "beyond_domain"
    assert censored.threshold is None and censored.strict_lower_bound == low_cap
    assert not censored.minimizing_assignment
    assert not censored.restricted_minimum_claimed_global_outside_domain
    assert resolution.resolution_status(censored, low_cap) == "separated"
    exact = resolution.resolve_prediction(tube, Q(27, 200))
    assert exact.threshold_status == "within_domain" and exact.threshold == Q(27, 200)
    assert resolution.resolution_status(exact, Q(27, 200)) == "unresolved"
    for tube, cap in (
        (_tube((Q(13), Q(0), Q(0), Q(0), Q(0))), CAP),
        (_tube((Q(20), Q(0), Q(0), Q(0), Q(0))), CAP),
        (_tube((Q(0),) * 5), Q(6)),
    ):
        unavailable = resolution.resolve_prediction(tube, cap)
        assert not unavailable.available
        assert unavailable.threshold_status == "unavailable"
        assert unavailable.threshold is None and unavailable.strict_lower_bound is None
        assert unavailable.zero_allowance_status == "unresolved"
        assert resolution.resolution_status(unavailable, Q(0)) == "unresolved"


def test_resolution_reconciles_the_frozen_dilation_on_constructed_inputs():
    examples = (
        _tube(_twist()),
        _tube(_contact_phases()),
        _tube(tuple(-value for value in _twist())),
        _tube((I(-Q(1, 16), Q(1, 16)),) * 5),
    )
    for tube in examples:
        result = resolution.resolve_prediction(tube, CAP)
        assert result.available
        for error in (Q(0), Q(1, 100), Q(1, 10), Q(27, 200), CAP):
            old = prediction.observe_prediction(tube, error)
            assert old.available
            assert resolution.resolution_status(result, error) == old.status
            if result.threshold is not None:
                assert (old.status == "separated") == (error < result.threshold)


def test_constructed_record_arithmetic_and_bound_evidence_reject_corruption(
    raw_allowance, allowances, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("an integration, field or scientific producer was called")

    monkeypatch.setattr(comparison, "source_certificate", forbidden)
    monkeypatch.setattr(prediction, "complete_field", forbidden)
    monkeypatch.setattr(prediction, "synchronized_field", forbidden)
    monkeypatch.setattr(prediction, "validated_box_taylor_step", forbidden)
    monkeypatch.setattr(prediction, "run_prediction", forbidden)
    monkeypatch.setattr(resolution, "assess_prediction_resolution", forbidden)
    records = _constructed_records(allowances)
    admitted = resolution.admit_prediction_records(*records, allowances=allowances)
    assert len(admitted.tubes) == len(admitted.observations) == 48
    assert (
        admitted.final_endpoint == prediction.declared_prediction_source().initial_box
    )
    assert all(item.status == "separated" for item in admitted.observations)
    assert not admitted.file_bound and admitted.receipts == ()
    assert admitted.endpoint_arithmetic_rebuilt
    assert not admitted.derivative_or_picard_recomputed
    assert not admitted.execution_authenticated and not admitted.prior_producer_replayed
    assert not admitted.actual_source_trajectory_read and not admitted.new_integration
    assert allowances.horizon_gap_upper == exact_record(
        raw_allowance["bounds"][1]["collective_gap_error_upper"]
    )
    corruptions = (
        ((0, "clock"), "forged clock"),
        ((0, "order"), True),
        ((1, "physical_coordinate_error"), True),
        ((1, "memberwise_image_certified"), False),
        ((2, "horizon_gap_upper"), Q(0)),
        ((3, 0, "cell_index"), True),
        ((3, 0, "allowance"), Q(0)),
        ((3, 0, "step", "duration"), Q(1, 4)),
        ((3, 0, "step", "order"), True),
        ((3, 0, "step", "picard_interior_margin"), Q(0)),
        ((3, 0, "step", "domain_lower_bounds"), (Q(0),)),
        ((3, 0, "step", "series", 0, 0, "lo"), Q(0)),
        ((3, 0, "step", "increment", 0, "hi"), Q(1)),
        ((3, 0, "step", "endpoint", 0, "lo"), Q(0)),
        ((3, 1, "step", "time"), Q(0)),
        ((3, 1, "step", "initial_box", 0, "lo"), Q(0)),
        ((3, 0, "observation", "status"), "unresolved"),
    )
    for path, value in corruptions:
        with pytest.raises((TypeError, ValueError, ArithmeticError)):
            resolution.admit_prediction_records(
                *_changed(records, path, value), allowances=allowances
            )
    for wrong_cells in (records[3][:-1], records[3] + [records[3][-1]]):
        with pytest.raises((TypeError, ValueError)):
            resolution.admit_prediction_records(
                *records[:3], wrong_cells, allowances=allowances
            )
    root = tmp_path / "valid"
    anchor = _write_bound_prediction(root, raw_allowance)
    bound = resolution.read_prediction(root, final_state_receipt=anchor)
    assert bound.file_bound and bound.final_state_receipt == anchor
    assert bound.tubes == admitted.tubes and bound.observations == admitted.observations
    assert (
        not bound.execution_authenticated and not bound.derivative_or_picard_recomputed
    )
    names = {row["path"] for row in bound.receipts}
    assert set(resolution.REQUIRED_SOURCE_NAMES) <= names
    assert resolution.FINAL_STATE_PATH in names and prediction.FINAL_STATE_PATH in names
    for name in (
        resolution.FINAL_STATE_PATH,
        resolution.PRIOR_DIRECTORY + "/evaluated-source.zip",
        resolution.PRIOR_DIRECTORY + "/trajectory/cell-000.json",
        resolution.REQUIRED_SOURCE_NAMES[0],
        prediction.CERTIFICATE_PATH,
    ):
        path = root / name
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            with pytest.raises((TypeError, ValueError)):
                resolution.read_prediction(root, final_state_receipt=anchor)
        finally:
            path.write_bytes(original)
    for wrong_anchor in (
        dict(anchor, bytes=True),
        dict(anchor, sha256="0" * 64),
        dict(anchor, path="../outside.json"),
    ):
        with pytest.raises((TypeError, ValueError)):
            resolution.read_prediction(root, final_state_receipt=wrong_anchor)
    for label, options in (
        ("forged-source", dict(forged_source=True)),
        ("wrong-archive", dict(wrong_archive=True)),
    ):
        wrong_root = tmp_path / label
        wrong_anchor = _write_bound_prediction(wrong_root, raw_allowance, **options)
        with pytest.raises((TypeError, ValueError)):
            resolution.read_prediction(wrong_root, final_state_receipt=wrong_anchor)


def test_primitive_admission_precedes_geometry_and_rejects_forged_thresholds(
    monkeypatch,
):
    valid = resolution.resolve_prediction(_tube(_contact_phases()), CAP)
    forged = deepcopy(valid)
    object.__setattr__(forged, "threshold", Q(0))
    with pytest.raises((TypeError, ValueError)):
        resolution.resolution_status(forged, Q(0))

    def forbidden(*args, **kwargs):
        raise AssertionError("geometry preceded raw input admission")

    monkeypatch.setattr(resolution, "pi_interval", forbidden)
    monkeypatch.setattr(prediction, "pi_interval", forbidden)
    for bad in (True, False, 0.0, float("nan"), float("inf"), "0", complex(0)):
        with pytest.raises((TypeError, ValueError)):
            resolution.resolve_prediction((Q(0),) * 9 + (bad,), CAP)
        with pytest.raises((TypeError, ValueError)):
            resolution.resolve_prediction((Q(0),) * 10, bad)
        with pytest.raises((TypeError, ValueError)):
            resolution.resolution_status(valid, bad)
    for error in (Q(-1), CAP + Q(1, 100)):
        with pytest.raises((TypeError, ValueError)):
            resolution.resolution_status(valid, error)
    with pytest.raises((TypeError, ValueError)):
        resolution.resolve_prediction((Q(0),) * 10, Q(-1))
    for size in (0, 9, 11):
        with pytest.raises((TypeError, ValueError)):
            resolution.resolve_prediction((Q(0),) * size, CAP)
    for lower, upper in ((False, Q(0)), (Q(1), Q(0))):
        forged_interval = I(0)
        object.__setattr__(forged_interval, "lo", lower)
        object.__setattr__(forged_interval, "hi", upper)
        with pytest.raises((TypeError, ValueError)):
            resolution.resolve_prediction((Q(0),) * 9 + (forged_interval,), CAP)

    def excess_coordinates():
        for _ in range(12):
            yield Q(0)
        raise AssertionError("state admission consumed past its bounded domain")

    with pytest.raises((TypeError, ValueError)):
        resolution.resolve_prediction(excess_coordinates(), CAP)
