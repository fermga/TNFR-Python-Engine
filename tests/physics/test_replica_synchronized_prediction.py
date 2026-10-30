"""Static controls for the fixed reduced-state predictor and evidence reader.

The retained algebraic allowance is prior evidence. No control executes the
prediction runner, loads old actual-response cells or integrates either law.
Formal jets below are supplied polynomials, not a new response trajectory.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import mpmath
import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import replica_synchronized_comparison as comparison
from tnfr.research import replica_synchronized_prediction as prediction
from tnfr.research.artifact_io import exact_record, file_receipt, write_json_once
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE = (
    ROOT / "artifacts/replica-internal-comparison-20261026/source-certificate.json"
)
ORACLE_DIGITS = 140
ORACLE_TOLERANCE = Q(1, 10**110)
POINT_WIDTH_CAP = Q(1, 10**30)


@pytest.fixture(scope="module")
def raw_allowance():
    return json_loads(CERTIFICATE.read_bytes())


@pytest.fixture(scope="module")
def allowances(raw_allowance):
    return prediction.admit_allowance_record(raw_allowance)


def _context():
    context = mpmath.mp.clone()
    context.dps = ORACLE_DIGITS
    return context


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _encloses(context, interval, expected):
    tolerance = _mp(context, ORACLE_TOLERANCE)
    assert interval.width < POINT_WIDTH_CAP
    assert _mp(context, interval.lo) - tolerance <= expected
    assert expected <= _mp(context, interval.hi) + tolerance


def _tube(phases):
    return (Q(0),) * 5 + tuple(phases)


def _twist():
    pi = pi_interval()
    return tuple(2 * (index - 2) * pi / 5 for index in range(5))


def _phases_from_first_four_gaps(gaps):
    phases = [Q(0)]
    for gap in gaps:
        phases.append(phases[-1] + gap)
    return tuple(phases)


def _full_polynomial_oracle(context, coefficients):
    """Evaluate the twenty fine rows on duplicated quadratic polynomials."""
    fine = tuple(row for row in coefficients for _ in range(2))
    rates = []
    for node in range(10):
        row = [context.mpf(0)] * 3
        neighbors = tuple(
            other for other in range(10) if (node // 2 - other // 2) % 5 in (1, 4)
        )
        for other in neighbors:
            g0, g1, g2 = (
                fine[10 + other][degree] - fine[10 + node][degree]
                for degree in range(3)
            )
            row[0] += context.sin(g0)
            row[1] += context.cos(g0) * g1
            row[2] += context.cos(g0) * g2 - context.sin(g0) * g1**2 / 2
        rates.append(tuple(value / (4 * context.pi) for value in row))
    for node in range(10):
        neighbors = tuple(
            other for other in range(10) if (node // 2 - other // 2) % 5 in (1, 4)
        )
        rates.append(
            tuple(
                sum(fine[node][degree] - fine[other][degree] for other in neighbors)
                / (4 * context.pi)
                for degree in range(3)
            )
        )
    return tuple(rates)


def _changed(raw, path, value):
    changed = deepcopy(raw)
    parent = changed
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return changed


def _write_artificial_evidence(root, certificate, *, wrong_archive=False):
    """Bind known algebraic evidence to artificial content associations."""
    sources = []
    for name in prediction.REQUIRED_SOURCE_NAMES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        sources.append(file_receipt(path, label=name))
    folder = root / prediction.PRIOR_DIRECTORY
    folder.mkdir(parents=True)
    archive = folder / "evaluated-source.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as output:
        for index, row in enumerate(sources):
            data = (root / row["path"]).read_bytes()
            if wrong_archive and index == 0:
                data = b"# intentionally wrong archived production owner\n"
            output.writestr(row["path"], data)
    archive_receipt = file_receipt(archive, label=archive.relative_to(root).as_posix())
    records = []

    def record(name, payload):
        path = folder / name
        write_json_once(path, payload)
        row = file_receipt(path, label=path.relative_to(root).as_posix())
        records.append(row)
        return row

    certificate_receipt = record("source-certificate.json", certificate)
    record("validation-plan.json", dict(source=sources, source_archive=archive_receipt))
    # Execution metadata does not replace admitted certificate arithmetic.
    record("evaluation-result.json", dict(result=certificate_receipt, status="stale"))
    final = dict(
        schema="tnfr.replica-internal-comparison-final.v1",
        artifacts=records,
        evaluated_source=sources,
        evaluated_source_archive=archive_receipt,
        source=[],
        source_archive=None,
    )
    path = root / prediction.FINAL_STATE_PATH
    write_json_once(path, final)
    return file_receipt(path, label=prediction.FINAL_STATE_PATH)


def test_reduced_field_and_supplied_jets_match_the_complete_fine_oracle():
    forms = (Q(1, 5), Q(-1, 4), Q(3, 10), Q(-2, 5), Q(3, 20))
    phases = (Q(3), Q(-2), Q(1), Q(-1), Q(2))
    constants = forms + phases
    coefficients = tuple(
        (value, Q(index - 4, 20), Q(3 - index, 50))
        for index, value in enumerate(constants)
    )
    context = _context()
    oracle = _full_polynomial_oracle(
        context,
        tuple(tuple(_mp(context, value) for value in row) for row in coefficients),
    )
    flow = prediction.synchronized_field()
    scalar = flow(constants)
    jets = flow(tuple(Jet(tuple(I(value) for value in row)) for row in coefficients))
    assert len(scalar) == len(jets) == 10
    for index in range(10):
        fine_index = 2 * index
        assert jets[index].order == 2
        for degree in range(3):
            expected = (oracle[fine_index][degree] + oracle[fine_index + 1][degree]) / 2
            _encloses(context, jets[index].coeffs[degree], expected)
            assert abs(
                oracle[fine_index][degree] - oracle[fine_index + 1][degree]
            ) < _mp(context, ORACLE_TOLERANCE)
        _encloses(context, scalar[index], oracle[fine_index][0])
    # A common phase origin changes neither inherited row.
    shifted = flow(forms + tuple(value + Q(7, 3) for value in phases))
    for index, value in enumerate(shifted):
        _encloses(context, value, oracle[2 * index][0])


def test_direct_initial_image_covers_memberwise_means_with_one_rounding():
    source = prediction.declared_prediction_source()
    epsilon, slack = Q(1, 2**20), Q(1, 2**128)
    nominal = tuple(Q(570, 287) * value for value in (-1, -1, 0, 1, 1)) + (Q(0),) * 5
    assert source.nominal == nominal
    assert source.physical_coordinate_error == epsilon
    assert source.numerical_enclosure_allowance == slack
    assert len(source.initial_box) == len(source.endpoint_rounding) == 10
    for center, interval, rounding in zip(
        nominal, source.initial_box, source.endpoint_rounding
    ):
        lower, upper = center - epsilon, center + epsilon
        assert interval == I(lower, upper)
        assert interval.lo <= lower <= upper <= interval.hi
        assert center - epsilon - slack <= interval.lo
        assert interval.hi <= center + epsilon + slack
        assert rounding == (lower - interval.lo, interval.hi - upper)
        assert all(0 <= value <= slack for value in rounding)
        # Pair corners and opposite errors distinguish a full image from
        # a nominal-only source or a second independent hidden uncertainty.
        for left, right in (
            (-epsilon, -epsilon),
            (epsilon, epsilon),
            (-epsilon, epsilon),
        ):
            mean = center + (left + right) / 2
            assert interval.contains(mean)
    assert source.memberwise_image_certified
    assert source.state_dimension == 10 and not source.nominal_only
    assert not source.whole_previous_fine_computational_box_claimed
    assert not source.equal_internal_storage_claimed
    with pytest.raises(FrozenInstanceError):
        source.memberwise_image_certified = False


def test_allowance_schedule_covers_both_closed_cells_touching_time_two(allowances):
    assert prediction.allowance_for_cell(0, allowances) == allowances.early_gap_upper
    assert prediction.allowance_for_cell(15, allowances) == allowances.early_gap_upper
    assert prediction.allowance_for_cell(16, allowances) == allowances.horizon_gap_upper
    assert prediction.allowance_for_cell(47, allowances) == allowances.horizon_gap_upper
    assert 16 * Q(1, 8) == 2
    assert 0 < allowances.early_gap_upper < allowances.horizon_gap_upper
    for index in (-1, 48, True, Q(1), 1.0, "1"):
        with pytest.raises((TypeError, ValueError)):
            prediction.allowance_for_cell(index, allowances)


def test_closed_dilation_preserves_negative_gaps_and_exact_endpoint_contact():
    pi = pi_interval()
    regular = prediction.observe_prediction(_tube(_twist()), Q(0))
    assert regular.available and regular.status == "unresolved"
    assert not regular.source_membership_inferred and not regular.actual_chart_inferred
    assert not regular.positive_acquisition_claimed
    assert regular.possible_shifts == ((0,), (0,), (0,), (0,), (1,))
    assert regular.reachable_sums == (1,)
    shifted = prediction.observe_prediction(
        _tube(tuple(value + Q(7, 3) for value in _twist())), Q(0)
    )
    assert shifted.possible_shifts == regular.possible_shifts
    turned = prediction.observe_prediction(
        _tube(
            tuple(
                value + (2 * pi if index == 0 else 0)
                for index, value in enumerate(_twist())
            )
        ),
        Q(0),
    )
    assert turned.possible_shifts == ((1,), (0,), (0,), (0,), (0,))
    reversed_result = prediction.observe_prediction(
        _tube(tuple(-value for value in _twist())), Q(0)
    )
    assert reversed_result.available and reversed_result.status == "separated"
    # These signed gaps still sum to2pi but are not all positive or acute.
    # Their continued possibility requires the dilated target, not the old one.
    other = (2 * pi + Q(1, 20)) / 4
    phases = _phases_from_first_four_gaps((Q(-1, 20), other, other, other))
    undeformed = prediction.observe_prediction(_tube(phases), Q(0))
    dilated = prediction.observe_prediction(_tube(phases), Q(1, 10))
    assert undeformed.status == "separated"
    assert dilated.available and dilated.status == "unresolved"
    assert 1 in dilated.reachable_sums
    assert dilated.dilated_target_lower == Q(1, 100) - Q(1, 10)
    assert dilated.dilated_target_upper == pi.hi / 2 - Q(1, 100) + Q(1, 10)
    # Choosing E=mu+1/8 makes the lower endpoint exactly dyadic. An
    # erroneous strict-overlap test cannot hide behind interval rounding.
    allowance = Q(1, 100) + Q(1, 8)
    other = (2 * pi + Q(1, 8)) / 4
    phases = _phases_from_first_four_gaps((Q(-1, 8), other, other, other))
    contact = prediction.observe_prediction(_tube(phases), allowance)
    assert contact.raw_gaps[0] == I(Q(-1, 8))
    assert contact.dilated_target_lower == Q(-1, 8)
    assert 0 in contact.possible_shifts[0] and 1 in contact.reachable_sums
    assert contact.status == "unresolved"


def test_joint_branches_and_missing_domains_cannot_become_false_exclusions():
    small = prediction.observe_prediction(
        _tube((I(-Q(1, 16), Q(1, 16)),) * 5), Q(1, 10)
    )
    assert small.available and small.possible_shifts == ((0,),) * 5
    assert small.reachable_sums == (0,) and small.status == "separated"
    for phases, allowance in (
        ((Q(13), Q(0), Q(0), Q(0), Q(0)), Q(0)),
        ((Q(20), Q(0), Q(0), Q(0), Q(0)), Q(0)),
        ((Q(0),) * 5, Q(6)),
    ):
        result = prediction.observe_prediction(_tube(phases), allowance)
        assert not result.available and result.status == "unresolved"
        assert result.possible_shifts == () and result.reachable_sums == ()
    with pytest.raises(FrozenInstanceError):
        small.available = False


def test_partial_prefixes_keep_the_unobserved_suffix_possible():
    whole = prediction.prediction_time_cover(("separated",) * 48)
    assert whole.full_horizon_separated and whole.residence_excluded
    assert whole.components == ()
    empty = prediction.prediction_time_cover(())
    assert not empty.full_horizon_separated and not empty.residence_excluded
    assert len(empty.components) == 1
    assert (empty.components[0].start, empty.components[0].end) == (0, 6)
    assert empty.components[0].left_included and empty.components[0].right_included
    partial = prediction.prediction_time_cover(("separated",) * 47)
    assert not partial.full_horizon_separated and partial.residence_excluded
    assert len(partial.components) == 1
    assert (partial.components[0].start, partial.components[0].end) == (Q(47, 8), 6)
    assert (
        not partial.components[0].left_included and partial.components[0].right_included
    )
    boundary = prediction.prediction_time_cover(("separated",) * 44)
    assert boundary.components[0].end - boundary.components[0].start == Q(1, 2)
    assert boundary.residence_excluded
    longer = prediction.prediction_time_cover(("separated",) * 43)
    assert not longer.residence_excluded and longer.components[0].residence_possible
    for invalid in (
        ("separated",) * 49,
        (True,),
        ("target_certified",),
        ("unavailable",),
    ):
        with pytest.raises((TypeError, ValueError)):
            prediction.prediction_time_cover(invalid)


def test_retained_allowance_arithmetic_and_bound_bytes_reject_corruption(
    raw_allowance, allowances, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("an old producer, field or trajectory was replayed")

    monkeypatch.setattr(comparison, "source_certificate", forbidden)
    monkeypatch.setattr(prediction, "complete_field", forbidden)
    monkeypatch.setattr(prediction, "validated_box_taylor_step", forbidden)
    assert allowances.early_gap_upper == exact_record(
        raw_allowance["bounds"][0]["collective_gap_error_upper"]
    )
    assert allowances.horizon_gap_upper == exact_record(
        raw_allowance["bounds"][1]["collective_gap_error_upper"]
    )
    assert not allowances.file_bound and allowances.receipts == ()
    assert allowances.prior_kernel_arithmetic_rebuilt
    assert allowances.conditional_on_retained_comparison_theorem
    assert not allowances.execution_authenticated
    assert (
        not allowances.old_producer_replayed and not allowances.actual_trajectory_read
    )
    stale = deepcopy(raw_allowance)
    stale["certified"] = False
    for bound in stale["bounds"]:
        bound["target_erosion"] = dict(jointly_empty_certified=True)
    rebuilt = prediction.admit_allowance_record(stale)
    assert rebuilt == allowances
    corruptions = (
        (("source", "clock"), "another structural clock"),
        (("source", "intended_coordinate_error"), True),
        (("clock",), "rescaled time"),
        (("physical_chart_horizon",), Q(7)),
        (("comparison_matrix", 0, 2), Q(0)),
        (("initial_envelope", 0), Q(1)),
        (("bounds", 0, "time"), Q(6)),
        (("bounds", 0, "kernel", "upper", 0, 0), Q(0)),
        (("bounds", 1, "collective_gap_error_upper"), Q(0)),
        (("bounds", 0, "all_prior_times_covered"), False),
    )
    for path, value in corruptions:
        with pytest.raises((TypeError, ValueError)):
            prediction.admit_allowance_record(_changed(raw_allowance, path, value))
    root = tmp_path / "valid"
    anchor = _write_artificial_evidence(root, raw_allowance)
    admitted = prediction.read_allowances(root, final_state_receipt=anchor)
    assert admitted.file_bound and admitted.final_state_receipt == anchor
    assert admitted.early_gap_upper == allowances.early_gap_upper
    assert admitted.horizon_gap_upper == allowances.horizon_gap_upper
    names = {row["path"] for row in admitted.receipts}
    assert set(prediction.REQUIRED_SOURCE_NAMES) <= names
    assert prediction.FINAL_STATE_PATH in names and prediction.CERTIFICATE_PATH in names
    assert not admitted.execution_authenticated and not admitted.actual_trajectory_read
    for name in (
        prediction.CERTIFICATE_PATH,
        prediction.PRIOR_DIRECTORY + "/evaluated-source.zip",
        prediction.REQUIRED_SOURCE_NAMES[0],
        prediction.FINAL_STATE_PATH,
    ):
        path = root / name
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            with pytest.raises((TypeError, ValueError)):
                prediction.read_allowances(root, final_state_receipt=anchor)
        finally:
            path.write_bytes(original)
    for wrong in (
        dict(anchor, sha256="0" * 64),
        dict(anchor, bytes=True),
        dict(anchor, path="../outside.json"),
    ):
        with pytest.raises((TypeError, ValueError)):
            prediction.read_allowances(root, final_state_receipt=wrong)
    forged_root = tmp_path / "forged-source"
    forged = _changed(raw_allowance, ("source", "clock"), "forged source clock")
    forged_anchor = _write_artificial_evidence(forged_root, forged)
    with pytest.raises((TypeError, ValueError)):
        prediction.read_allowances(forged_root, final_state_receipt=forged_anchor)
    wrong_zip_root = tmp_path / "wrong-archive"
    wrong_zip_anchor = _write_artificial_evidence(
        wrong_zip_root, raw_allowance, wrong_archive=True
    )
    with pytest.raises((TypeError, ValueError)):
        prediction.read_allowances(wrong_zip_root, final_state_receipt=wrong_zip_anchor)


def test_raw_admission_precedes_fine_field_and_observer_arithmetic(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scientific work preceded input admission")

    monkeypatch.setattr(prediction, "complete_field", lambda: forbidden)
    flow = prediction.synchronized_field()
    monkeypatch.setattr(prediction, "pi_interval", forbidden)
    for bad in (True, False, 0.0, float("nan"), float("inf"), "0", complex(0)):
        invalid = (Q(0),) * 9 + (bad,)
        with pytest.raises((TypeError, ValueError)):
            flow(invalid)
        with pytest.raises((TypeError, ValueError)):
            prediction.observe_prediction(invalid, Q(0))
        with pytest.raises((TypeError, ValueError)):
            prediction.observe_prediction((Q(0),) * 10, bad)
    for size in (0, 9, 11):
        with pytest.raises((TypeError, ValueError)):
            flow((Q(0),) * size)
        with pytest.raises((TypeError, ValueError)):
            prediction.observe_prediction((Q(0),) * size, Q(0))
    for lower, upper in ((False, Q(0)), (Q(1), Q(0))):
        forged = I(0)
        object.__setattr__(forged, "lo", lower)
        object.__setattr__(forged, "hi", upper)
        with pytest.raises((TypeError, ValueError)):
            flow((Q(0),) * 9 + (forged,))
        with pytest.raises((TypeError, ValueError)):
            prediction.observe_prediction((Q(0),) * 9 + (forged,), Q(0))
    with pytest.raises((TypeError, ValueError)):
        flow((Jet.constant(0, 1),) * 9 + (Jet.constant(0, 2),))
    with pytest.raises((TypeError, ValueError)):
        prediction.observe_prediction((Q(0),) * 10, Q(-1))
