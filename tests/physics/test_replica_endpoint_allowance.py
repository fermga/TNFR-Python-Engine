"""Constructed endpoint-allowance controls, without the reserved response.

Only artificial matrices receive new time evaluations. Retained time-two and
time-six allowance arithmetic may be admitted as prior evidence. Constant
Taylor fixtures and broad tubes test readers, not derivative certificates.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as Q
from math import factorial
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import mpmath
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.research import replica_endpoint_allowance as endpoint
from tnfr.research import replica_prediction_resolution as resolution
from tnfr.research import replica_synchronized_comparison as comparison
from tnfr.research import replica_synchronized_prediction as prediction
from tnfr.research.artifact_io import file_receipt, write_json_once
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE = ROOT / prediction.CERTIFICATE_PATH
TARGETS = (36, 37, 38, 39, 40)
TIMES = (Q(37, 8), Q(19, 4), Q(39, 8), Q(5), Q(41, 8))


@pytest.fixture(scope="module")
def allowances():
    return prediction.admit_allowance_record(json_loads(CERTIFICATE.read_bytes()))


def _matrix(entries=()):
    rows = [[Q(0)] * 5 for _ in range(5)]
    for i, j, value in entries:
        rows[i][j] = Q(value)
    return tuple(tuple(row) for row in rows)


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _context():
    context = mpmath.mp.clone()
    context.dps = 140
    return context


def _scalar_upper(value):
    """Independent scalar series bound with a much later truncation."""
    partial = sum((value**j / factorial(j) for j in range(65)), Q(0))
    return partial + value**65 / factorial(65) / (1 - value / 66)


def _sqrt_two_upper():
    # Rational Newton iterates remain above sqrt(2), independently of the
    # production dyadic enclosure. Five iterations give a strict fine bound.
    upper = Q(3, 2)
    for _ in range(5):
        upper = (upper + 2 / upper) / 2
    assert upper * upper > 2
    return upper


def _changed(raw, path, value):
    changed = deepcopy(raw)
    parent = changed
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return changed


def _constructed_prediction(allowances):
    """Supply constant Taylor arithmetic and five deliberately broad tubes."""
    source = prediction.declared_prediction_source()
    box, epsilon = source.initial_box, source.physical_coordinate_error
    ordinary = tuple(I(value.lo - Q(1, 8), value.hi + Q(1, 8)) for value in box)
    special = ordinary[:5] + (
        I(-epsilon, epsilon),
        I(Q(-1, 8), epsilon),
        I(-epsilon, Q(3, 2)),
        I(-epsilon, Q(25, 8)),
        I(-epsilon, Q(19, 4)),
    )
    anchor = dict(path=prediction.FINAL_STATE_PATH, bytes=1, sha256="0" * 64)
    cells = []
    for index in range(48):
        tube = special if index in TARGETS else ordinary
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
            local_remainder_bounds=tuple(asdict(I(0)) for _ in range(10)),
            increment=tuple(asdict(I(0)) for _ in range(10)),
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
    admitted = resolution.admit_prediction_records(
        prediction.prediction_attempt(anchor),
        asdict(source),
        asdict(allowances),
        cells,
        allowances=allowances,
    )
    # The outer-wrapper test substitutes only its already-tested nested reader;
    # the source, threshold and new wrapper's file checks remain executable.
    return replace(
        admitted,
        final_state_receipt=dict(
            path=resolution.FINAL_STATE_PATH, bytes=1, sha256="1" * 64
        ),
    )


def _resolution_record(admitted):
    observations = tuple(
        resolution.resolve_prediction(tube, admitted.allowances.horizon_gap_upper)
        for tube in admitted.tubes
    )
    return dict(
        evidence=asdict(admitted),
        observations=tuple(asdict(row) for row in observations),
        prior_unresolved_indices=TARGETS,
        frozen_classifications=tuple(row.status for row in admitted.observations),
    )


def _record(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_once(path, value)
    return file_receipt(path, label=name)


def _bound_resolution(root, raw, *, wrong_archive=False):
    directory = endpoint.PRIOR_DIRECTORY
    sources = []
    for name in endpoint.REQUIRED_SOURCE_NAMES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        sources.append(file_receipt(path, label=name))
    archive_name = directory + "/evaluated-source.zip"
    path = root / archive_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "x", compression=ZIP_DEFLATED) as archive:
        for index, row in enumerate(sources):
            data = (root / row["path"]).read_bytes()
            if wrong_archive and index == 0:
                data = b"# intentionally inconsistent archived owner\n"
            archive.writestr(row["path"], data)
    archive = file_receipt(path, label=archive_name)
    result = _record(root, directory + "/resolution-result.json", raw)
    records = [
        result,
        _record(
            root,
            directory + "/validation-plan.json",
            dict(source=sources, source_archive=archive),
        ),
        _record(root, directory + "/execution-result.json", dict(result=result)),
    ]
    return _record(
        root,
        endpoint.FINAL_STATE_PATH,
        dict(
            schema="tnfr.replica-prediction-resolution-final.v1",
            artifacts=records,
            evaluated_source=sources,
            evaluated_source_archive=archive,
        ),
    )


def test_artificial_positive_exponentials_match_scalar_and_nilpotent_oracles():
    context = _context()
    matrix = _matrix(((1, 1, Q(1, 4)),))
    initial = (Q(0), Q(1, 100), Q(0), Q(0), Q(0))
    scalar = endpoint.evaluate_endpoint(matrix, initial, Q(3, 2), Q(1, 4))
    true_form = context.exp(context.mpf(3) / 8) / 100
    true_gap = context.sqrt(2) * true_form
    assert true_form <= _mp(context, scalar.envelope_upper[1])
    assert true_gap <= _mp(context, scalar.uncapped_gap_upper)
    assert _mp(context, scalar.uncapped_gap_upper) - true_gap < context.mpf("1e-30")
    assert scalar.kernel.matrix == matrix and scalar.initial_envelope == initial
    assert all(value == 0 for i, value in enumerate(scalar.envelope_upper) if i != 1)
    # A two-edge nilpotent path has exp(Mt)=Id+tM+t^2*M^2/2 exactly.
    matrix = _matrix(((1, 2, Q(1, 16)), (2, 4, Q(1, 8))))
    initial = (Q(0), Q(0), Q(0), Q(0), Q(1))
    for time in (Q(1, 4), Q(1), Q(3)):
        bound = endpoint.evaluate_endpoint(matrix, initial, time, Q(1, 4))
        expected = [list(row) for row in _matrix()]
        for i in range(5):
            expected[i][i] = Q(1)
        expected[1][2], expected[2][4] = time / 16, time / 8
        expected[1][4] = time * time / 256
        assert bound.kernel.upper == tuple(tuple(row) for row in expected)
        assert bound.envelope_upper == (0, time * time / 256, time / 8, 0, 1)
        assert all(value == 0 for row in bound.kernel.tail_upper_bound for value in row)


def test_zero_horizon_and_positive_premises_use_exact_not_rounded_monotonicity():
    matrix = _matrix(((1, 2, Q(1, 16)), (2, 4, Q(1, 8))))
    initial = (Q(0), Q(0), Q(0), Q(0), Q(1))
    start = endpoint.evaluate_endpoint(matrix, initial, Q(0), Q(1, 4))
    horizon = endpoint.evaluate_endpoint(matrix, initial, Q(6), Q(1, 4))
    assert start.envelope_upper == initial
    assert start.uncapped_gap_upper == start.capped_gap_upper == 0
    assert start.all_prior_times_covered and horizon.all_prior_times_covered
    assert (
        not horizon.majorant_restarted and not horizon.numerical_bounds_assumed_monotone
    )
    assert not horizon.actual_error_measured
    assert horizon.envelope_upper[1] == Q(9, 64)
    assert horizon.capped_gap_upper <= Q(1, 4)
    # Exact toy Q(t)=t^2/256 is nondecreasing throughout [0,6]. The
    # implication concerns this function, not arbitrary rounded upper bounds.
    for left, right in ((Q(0), Q(1)), (Q(1, 3), Q(2)), (Q(5), Q(6))):
        assert right * right - left * left == (right - left) * (right + left) >= 0
    exact_constant, earlier_loose_upper, later_tight_upper = Q(1), Q(3), Q(2)
    assert exact_constant <= later_tight_upper < earlier_loose_upper
    for bad_matrix, bad_initial in (
        (_matrix(((1, 1, Q(-1)),)), initial),
        (matrix, (Q(0), Q(-1), Q(0), Q(0), Q(1))),
    ):
        with pytest.raises((TypeError, ValueError)):
            endpoint.evaluate_endpoint(bad_matrix, bad_initial, Q(1), Q(1, 4))


def test_minimum_with_an_independently_valid_cap_preserves_the_domain():
    matrix = _matrix(((1, 1, Q(1, 4)),))
    initial = (Q(0), Q(1, 100), Q(0), Q(0), Q(0))
    # Both caps bound the same artificial exact horizon error. The tight cap
    # uses independent scalar and rational-root inequalities, not the kernel.
    tight_cap = _sqrt_two_upper() * _scalar_upper(Q(3, 2)) / 100
    tight = endpoint.evaluate_endpoint(matrix, initial, Q(6), tight_cap)
    loose = endpoint.evaluate_endpoint(matrix, initial, Q(6), Q(1, 4))
    assert tight.capped_gap_upper == tight_cap < tight.uncapped_gap_upper
    assert loose.capped_gap_upper == loose.uncapped_gap_upper < Q(1, 4)
    assert tight.uncapped_gap_upper == loose.uncapped_gap_upper
    context = _context()
    actual = context.sqrt(2) * context.exp(context.mpf(3) / 2) / 100
    assert actual <= _mp(context, tight.capped_gap_upper)
    assert tight.capped_gap_upper < Q(1, 4)


def test_contact_requires_a_strict_allowance_improvement():
    threshold = Q(27, 200)
    assert (
        endpoint.classify_allowance(threshold - Q(1, 2**40), threshold) == "separated"
    )
    assert endpoint.classify_allowance(threshold, threshold) == "unresolved"
    assert (
        endpoint.classify_allowance(threshold + Q(1, 2**40), threshold) == "unresolved"
    )
    assert endpoint.classify_allowance(Q(0), threshold) == "separated"
    assert endpoint.classify_allowance(Q(0), Q(0)) == "unresolved"


def test_fixed_five_cell_schedule_preserves_the_other_forty_three_allowances():
    assert endpoint.TARGET_CELLS == TARGETS and endpoint.ENDPOINT_TIMES == TIMES
    previous = (Q(1, 100),) * 16 + (Q(1, 5),) * 32
    updates = tuple(Q(1, 10) + Q(index, 1000) for index in range(5))
    merged = endpoint.merge_allowances(previous, updates)
    assert len(merged) == 48
    assert all(merged[i] == previous[i] for i in range(48) if i not in TARGETS)
    assert tuple(merged[i] for i in TARGETS) == updates
    assert tuple(Q(i + 1, 8) for i in TARGETS) == TIMES
    assert merged[15] == previous[15] and merged[16] == previous[16]
    with pytest.raises((TypeError, ValueError)):
        endpoint.merge_allowances(previous, (Q(1),) * 5)


def test_closed_time_cover_distinguishes_full_exclusion_from_residence_exclusion():
    previous = tuple("unresolved" if i in TARGETS else "separated" for i in range(48))
    full = endpoint.endpoint_outcome(previous, ("separated",) * 5)
    partial = endpoint.endpoint_outcome(previous, ("unresolved",) * 4 + ("separated",))
    unchanged = endpoint.endpoint_outcome(previous, ("unresolved",) * 5)
    assert full["status"] == "full_horizon_target_excluded_by_endpoint_allowances"
    assert full["time_cover"]["full_horizon_separated"]
    assert partial["status"] == "half_unit_residence_excluded_by_endpoint_allowances"
    component = partial["time_cover"]["components"][0]
    assert component["end"] - component["start"] == Q(1, 2)
    assert not component["left_included"] and not component["right_included"]
    assert not partial["time_cover"]["full_horizon_separated"]
    assert partial["time_cover"]["residence_excluded"]
    assert unchanged["status"] == "endpoint_allowance_resolution_incomplete"
    assert not unchanged["time_cover"]["residence_excluded"]
    assert not unchanged["time_cover"]["existence_claimed"]


def test_bound_evidence_and_consumed_threshold_corruption_are_rejected(
    allowances, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "new endpoint evaluation or old scientific producer was called"
        )

    monkeypatch.setattr(endpoint, "evaluate_endpoint", forbidden)
    monkeypatch.setattr(endpoint, "assess_endpoint_allowances", forbidden)
    monkeypatch.setattr(comparison, "source_certificate", forbidden)
    monkeypatch.setattr(prediction, "run_prediction", forbidden)
    monkeypatch.setattr(prediction, "complete_field", forbidden)
    monkeypatch.setattr(prediction, "validated_box_taylor_step", forbidden)
    monkeypatch.setattr(resolution, "assess_prediction_resolution", forbidden)
    admitted = _constructed_prediction(allowances)
    raw = _resolution_record(admitted)
    assert (
        tuple(
            i
            for i, row in enumerate(admitted.observations)
            if row.status == "unresolved"
        )
        == TARGETS
    )
    preparation = endpoint.admit_resolution_record(raw, prediction=admitted)
    assert preparation.cell_indices == TARGETS and len(preparation.thresholds) == 5
    assert all(
        0 < value < allowances.horizon_gap_upper for value in preparation.thresholds
    )
    assert preparation.prior_threshold_arithmetic_rebuilt
    assert not preparation.file_bound and not preparation.execution_authenticated
    assert (
        not preparation.prior_producer_replayed
        and not preparation.actual_source_trajectory_read
    )
    raw_prior = json_loads(CERTIFICATE.read_bytes())
    for path, value in (
        (("bounds", 0, "kernel", "upper", 0, 0), Q(0)),
        (("bounds", 1, "kernel", "duration"), True),
    ):
        with pytest.raises((TypeError, ValueError)):
            prediction.admit_allowance_record(_changed(raw_prior, path, value))
    corruptions = (
        (("evidence", "source", "physical_coordinate_error"), True),
        (("evidence", "source", "memberwise_image_certified"), False),
        (("evidence", "allowances", "comparison_matrix"), _matrix()),
        (("evidence", "allowances", "horizon_gap_upper"), Q(0)),
        (("evidence", "final_state_receipt", "sha256"), "2" * 64),
        (("prior_unresolved_indices",), (35, 36, 37, 38, 39)),
        (("frozen_classifications",), ("separated",) * 48),
        (("observations", 36, "threshold"), Q(0)),
        (("observations", 36, "cap"), True),
        (("observations", 36, "distances"), ((Q(0),) * 5,) * 5),
        (("observations", 36, "minimizing_assignment"), (0,) * 5),
        (("observations", 36, "available"), False),
    )
    for path, value in corruptions:
        with pytest.raises((TypeError, ValueError)):
            endpoint.admit_resolution_record(
                _changed(raw, path, value), prediction=admitted
            )
    for size in (47, 49):
        changed = dict(raw, observations=(raw["observations"] * 2)[:size])
        with pytest.raises((TypeError, ValueError)):
            endpoint.admit_resolution_record(changed, prediction=admitted)
    nested_reads = []

    def nested_reader(root, *, final_state_receipt):
        assert final_state_receipt == admitted.final_state_receipt
        nested_reads.append(root)
        return admitted

    monkeypatch.setattr(endpoint, "read_prediction", nested_reader)
    root = tmp_path / "valid"
    anchor = _bound_resolution(root, raw)
    bound = endpoint.read_resolution(root, final_state_receipt=anchor)
    assert bound.file_bound and bound.final_state_receipt == anchor and nested_reads
    assert bound.thresholds == preparation.thresholds
    assert not bound.execution_authenticated
    for name in (
        endpoint.FINAL_STATE_PATH,
        endpoint.PRIOR_DIRECTORY + "/resolution-result.json",
        endpoint.PRIOR_DIRECTORY + "/evaluated-source.zip",
        endpoint.REQUIRED_SOURCE_NAMES[0],
    ):
        path = root / name
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            with pytest.raises((TypeError, ValueError)):
                endpoint.read_resolution(root, final_state_receipt=anchor)
        finally:
            path.write_bytes(original)
    for invalid_anchor in (
        dict(anchor, bytes=True),
        dict(anchor, sha256="0" * 64),
        dict(anchor, path="../outside.json"),
    ):
        with pytest.raises((TypeError, ValueError)):
            endpoint.read_resolution(root, final_state_receipt=invalid_anchor)
    for label, record, options in (
        (
            "forged-threshold",
            _changed(raw, ("observations", 36, "threshold"), Q(0)),
            {},
        ),
        ("wrong-archive", raw, dict(wrong_archive=True)),
    ):
        bad_root = tmp_path / label
        bad_anchor = _bound_resolution(bad_root, record, **options)
        with pytest.raises((TypeError, ValueError)):
            endpoint.read_resolution(bad_root, final_state_receipt=bad_anchor)


def test_raw_values_dimensions_and_records_fail_before_numerical_work(monkeypatch):
    matrix, initial = _matrix(), (Q(0),) * 5

    def forbidden(*args, **kwargs):
        raise AssertionError("numerical work preceded primitive admission")

    monkeypatch.setattr(endpoint, "nonnegative_exponential_upper", forbidden)
    monkeypatch.setattr(endpoint, "sqrt", forbidden)
    for bad in (True, False, 0.0, float("nan"), float("inf"), "0", complex(0)):
        for args in (
            (matrix, initial, bad, Q(1)),
            (matrix, initial, Q(1), bad),
            (matrix, (Q(0),) * 4 + (bad,), Q(1), Q(1)),
            (matrix[:4] + ((Q(0),) * 4 + (bad,),), initial, Q(1), Q(1)),
        ):
            with pytest.raises((TypeError, ValueError)):
                endpoint.evaluate_endpoint(*args)
        for upper, threshold in ((bad, Q(1)), (Q(0), bad)):
            with pytest.raises((TypeError, ValueError)):
                endpoint.classify_allowance(upper, threshold)
        with pytest.raises((TypeError, ValueError)):
            endpoint.merge_allowances((Q(1),) * 48, (Q(0),) * 4 + (bad,))
    for time, cap in ((Q(-1), Q(1)), (Q(7), Q(1)), (Q(1), Q(-1))):
        with pytest.raises((TypeError, ValueError)):
            endpoint.evaluate_endpoint(matrix, initial, time, cap)
    for upper, threshold in ((Q(-1), Q(0)), (Q(0), Q(-1))):
        with pytest.raises((TypeError, ValueError)):
            endpoint.classify_allowance(upper, threshold)
    with pytest.raises((TypeError, ValueError)):
        endpoint.evaluate_endpoint(matrix[:4] + ((Q(0),) * 4,), initial, Q(1), Q(1))
    for size in (0, 4, 6):
        with pytest.raises((TypeError, ValueError)):
            endpoint.evaluate_endpoint((_matrix()[0],) * size, initial, Q(1), Q(1))
        with pytest.raises((TypeError, ValueError)):
            endpoint.evaluate_endpoint(matrix, (Q(0),) * size, Q(1), Q(1))
    for previous, updates in (((Q(1),) * 47, (Q(0),) * 5), ((Q(1),) * 48, (Q(0),) * 6)):
        with pytest.raises((TypeError, ValueError)):
            endpoint.merge_allowances(previous, updates)
    previous = tuple("unresolved" if i in TARGETS else "separated" for i in range(48))
    for updates in ((True,) * 5, ("target_certified",) * 5, ("separated",) * 4):
        with pytest.raises((TypeError, ValueError)):
            endpoint.endpoint_outcome(previous, updates)
    with pytest.raises((TypeError, ValueError)):
        endpoint.endpoint_outcome(("separated",) * 48, ("separated",) * 5)

    def oversized_matrix():
        for _ in range(7):
            yield matrix[0]
        raise AssertionError("matrix admission exceeded its bounded domain")

    with pytest.raises((TypeError, ValueError)):
        endpoint.evaluate_endpoint(oversized_matrix(), initial, Q(1), Q(1))
