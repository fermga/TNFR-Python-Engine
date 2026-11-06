"""Detached-observer controls using only constructed boxes and records.

No retained scientific trajectory is loaded here. Artificial constant Taylor
records test association and arithmetic admission, not the TNFR differential
law or Picard inclusion. The already evaluated response is read only by the
separately frozen assessment after these controls pass.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import replica_collective_preparation as preparation
from tnfr.research import replica_target_separation as observer
from tnfr.research.artifact_io import file_receipt, write_json_once


def _state(means, *, delta=Q(1, 8)):
    """Construct a box; equal interval entries do not imply correlation."""
    forms = tuple(Q(sign, 8) for _ in range(5) for sign in (1, -1))
    phases = tuple(mean + sign * delta for mean in means for sign in (1, -1))
    return forms + phases


def _target_means():
    pi = pi_interval()
    return tuple(2 * (index - 2) * pi / 5 for index in range(5))


def _statuses(start, count, status="unresolved"):
    return tuple(
        status if start <= index < start + count else "separated" for index in range(48)
    )


@pytest.fixture(scope="module")
def source():
    return preparation.declared_source()


def _artificial_records(source):
    """Consistent constant-polynomial records, without derivative evidence."""
    initial = source.initial_box
    tube = tuple(value + I(-Q(1, 1024), Q(1, 1024)) for value in initial)
    interval = asdict
    zero = asdict(I(0))
    cells = []
    for index in range(48):
        step = dict(
            time=Q(index, 8),
            duration=Q(1, 8),
            order=8,
            initial_box=tuple(map(interval, initial)),
            tube=tuple(map(interval, tube)),
            series=tuple((interval(value),) + (zero,) * 8 for value in initial),
            local_remainder_bounds=(zero,) * 20,
            increment=(zero,) * 20,
            endpoint=tuple(map(interval, initial)),
            picard_interior_margin=Q(1, 2048),
            domain_lower_bounds=(Q(1),),
            method="direct_source_box_Picard_Taylor_dyadic128_v1",
        )
        cells.append(dict(cell_index=index, step=step))
    attempt = dict(
        schema="tnfr.replica-fixed-budget-trajectory.v1",
        horizon=Q(6),
        step=Q(1, 8),
        order=8,
        cell_count=48,
        wall_seconds=900,
        margin=Q(1, 100),
        residence_cells=4,
        phase_capture="exact_rational_interval_no_graph_recapture",
    )
    return attempt, asdict(source), tuple(cells)


def _replace_path(tree, path, value):
    if not path:
        return value
    key, *remaining = path
    if isinstance(tree, dict):
        changed = dict(tree)
        changed[key] = _replace_path(tree[key], remaining, value)
        return changed
    changed = list(tree)
    changed[key] = _replace_path(tree[key], remaining, value)
    return tuple(changed) if isinstance(tree, tuple) else changed


def _write_artificial_repository(root, source):
    """Bind constructed bytes, without claiming that their dynamics occurred."""
    producer_root = Path(__file__).resolve().parents[2]
    sources = []
    for name in observer.REQUIRED_SOURCE_NAMES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((producer_root / name).read_bytes())
        sources.append(file_receipt(path, label=name))
    folder = root / observer.PRIOR_DIRECTORY
    folder.mkdir(parents=True)
    archive = folder / "evaluated-source.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as output:
        for row in sources:
            output.writestr(row["path"], (root / row["path"]).read_bytes())
    archive_receipt = file_receipt(archive, label=archive.relative_to(root).as_posix())
    records = []

    def record(name, payload):
        path = folder / name
        write_json_once(path, payload)
        records.append(file_receipt(path, label=path.relative_to(root).as_posix()))

    record("validation-plan.json", dict(source=sources, source_archive=archive_receipt))
    # These metadata verdicts are deliberately stale. The detached consumer
    # binds their bytes, not their scientific interpretations.
    record("execution-result.json", dict(integration_complete=False))
    record("prefix-audit.json", dict(cells_completed=0, status="stale"))
    record("trajectory/trajectory-result.json", dict(absence_claimed=True))
    attempt, source_record, cells = _artificial_records(source)
    record("trajectory/attempt.json", attempt)
    record("trajectory/source.json", source_record)
    for index, cell in enumerate(cells):
        record(f"trajectory/cell-{index:03d}.json", cell)
    final = dict(
        schema="tnfr.replica-fixed-budget-formation-final.v1",
        artifacts=records,
        evaluated_source=sources,
        evaluated_source_archive=archive_receipt,
    )
    path = root / observer.FINAL_STATE_PATH
    write_json_once(path, final)
    return file_receipt(path, label=observer.FINAL_STATE_PATH)


def test_target_orientation_and_common_phase_do_not_change_the_declared_chart():
    means = _target_means()
    result = observer.observe_target(_state(means))
    assert result.available and result.status == "target_certified"
    assert result.possible_shifts == ((0,), (0,), (0,), (0,), (1,))
    assert result.reachable_sums == (1,)
    assert result.positive_observation.mean_geometry_certified
    assert not result.source_membership_inferred
    shifted = observer.observe_target(_state(tuple(value + Q(7, 3) for value in means)))
    assert shifted.status == result.status
    assert shifted.possible_shifts == result.possible_shifts
    reversed_result = observer.observe_target(_state(tuple(-value for value in means)))
    assert reversed_result.available and reversed_result.status == "separated"
    assert 1 not in reversed_result.reachable_sums
    # Whole-pair turns change the continuous representatives and integer
    # branches together, without changing the physical target.
    pi = pi_interval()
    turned = observer.observe_target(
        _state(
            tuple(
                value + (2 * pi if index == 0 else 0)
                for index, value in enumerate(means)
            )
        )
    )
    assert turned.status == "target_certified"
    assert turned.possible_shifts == ((1,), (0,), (0,), (0,), (0,))
    with pytest.raises(FrozenInstanceError):
        result.available = False


def test_possible_edge_contacts_require_a_common_winding_sum():
    # Every scalar gap can meet the positive target, but their integer
    # branches all equal zero. Telescoping rules out winding one.
    small = observer.observe_target(_state((I(-Q(1, 16), Q(1, 16)),) * 5))
    assert small.available
    assert small.possible_shifts == ((0,),) * 5
    assert small.reachable_sums == (0,)
    assert small.status == "separated"
    assert not small.positive_observation.mean_geometry_certified
    # Widening a genuine target around a boundary retains a possible sum,
    # but destroys whole-box membership. This is not an existence proof.
    means = list(_target_means())
    means[0] += I(-Q(1, 2), Q(1, 2))
    ambiguous = observer.observe_target(_state(means))
    assert ambiguous.available and ambiguous.status == "unresolved"
    assert 1 in ambiguous.reachable_sums
    assert not ambiguous.positive_observation.mean_geometry_certified
    assert not ambiguous.source_membership_inferred


def test_closed_contacts_and_unavailable_domains_cannot_become_separation(monkeypatch):
    # The exact rational lower endpoint is contained despite outward dyadic
    # rounding. Contact stays possible even when another edge excludes the
    # complete target. No equality is silently replaced by a strict overlap.
    contact = observer.observe_target(_state((Q(0), Q(1, 100), Q(0), Q(0), Q(0))))
    assert 0 in contact.possible_shifts[0]
    below = observer.observe_target(_state((Q(0), Q(1, 128), Q(0), Q(0), Q(0))))
    assert below.possible_shifts[0] == ()
    # A dyadic arithmetic fixture makes endpoint equality exact, so a
    # mistaken strict-overlap implementation cannot hide behind outward
    # rounding of 1/100. This local mechanism fixture changes no retained
    # protocol, source, trajectory or scientific target.
    with monkeypatch.context() as patch:
        patch.setattr(preparation, "MARGIN", Q(1, 8))
        patch.setattr(observer, "MARGIN", Q(1, 8))
        exact_contact = observer.observe_target(
            _state((Q(0), Q(1, 8), Q(0), Q(0), Q(0)))
        )
        assert exact_contact.positive_observation.raw_gaps[0] == I(Q(1, 8))
        assert exact_contact.possible_shifts[0] == (0,)
    for state in (
        _state((Q(0),) * 5, delta=Q(2)),
        _state((Q(13), Q(0), Q(0), Q(0), Q(0))),
        _state((Q(20), Q(0), Q(0), Q(0), Q(0))),
    ):
        result = observer.observe_target(state)
        assert not result.available and result.status == "unresolved"
        assert not result.source_membership_inferred


def test_time_cover_prices_closed_endpoints_and_nongrid_residence_windows():
    empty = observer.time_cover(("separated",) * 48)
    assert empty.components == ()
    assert empty.residence_excluded and empty.full_horizon_separated
    whole = observer.time_cover(("unresolved",) * 48)
    assert not whole.residence_excluded and not whole.full_horizon_separated
    assert len(whole.components) == 1
    full = whole.components[0]
    assert (full.start, full.end) == (0, 6)
    assert full.left_included and full.right_included and full.residence_possible
    for start, left, right in ((0, True, False), (10, False, False), (44, False, True)):
        cover = observer.time_cover(_statuses(start, 4))
        assert len(cover.components) == 1
        component = cover.components[0]
        assert (component.start, component.end) == (Q(start, 8), Q(start + 4, 8))
        assert component.left_included is left and component.right_included is right
        assert component.end - component.start == Q(1, 2)
        assert not component.residence_possible and cover.residence_excluded
        assert not cover.full_horizon_separated
    longer = observer.time_cover(_statuses(10, 5))
    component = longer.components[0]
    assert component.residence_possible and not longer.residence_excluded
    shifted_start = component.start + Q(1, 32)
    assert component.start < shifted_start < shifted_start + Q(1, 2) < component.end
    # Positive cells with unresolved neighbours preserve a valid known
    # window. Impossible target/separation contact must not certify both.
    positive = ["unresolved"] * 48
    positive[10:14] = ["target_certified"] * 4
    assert observer.time_cover(positive).positive_residence_windows == (
        (Q(10, 8), Q(14, 8)),
    )
    for invalid in (
        ("unresolved",) * 47,
        ("unresolved",) * 49,
        (True,) * 48,
        ("unavailable",) * 48,
    ):
        with pytest.raises((TypeError, ValueError)):
            observer.time_cover(invalid)
    for start in (0, 10, 44):
        with pytest.raises(ValueError):
            observer.time_cover(_statuses(start, 4, "target_certified"))
    with pytest.raises(FrozenInstanceError):
        full.residence_possible = False


def test_complete_artificial_records_rebuild_arithmetic_without_producer_replay(
    source, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("the retained reader replayed a scientific producer")

    monkeypatch.setattr(preparation, "complete_field", forbidden)
    monkeypatch.setattr(preparation, "validated_box_taylor_step", forbidden)
    records = _artificial_records(source)
    admitted = observer.admit_retained_records(*records)
    assert admitted.source == source
    assert admitted.cell_count == len(admitted.tubes) == 48
    assert admitted.initial_box == admitted.endpoint == source.initial_box
    assert not admitted.file_bound and admitted.receipts == ()
    assert not admitted.derivative_or_picard_recomputed
    assert not admitted.execution_authenticated
    for tube in admitted.tubes:
        assert len(tube) == 20
        assert all(
            initial.subset_of(bound) for initial, bound in zip(source.initial_box, tube)
        )
    dirty = deepcopy(records)
    for cell in dirty[2]:
        cell.update(
            observation={"mean_geometry_certified": True, "box_chart_certified": False},
            source_chart_used=False,
            residence_cell_certified=True,
            conditional_activity_lower_bound="unused cached claim",
        )
    assert observer.admit_retained_records(*dirty) == admitted
    with pytest.raises(FrozenInstanceError):
        admitted.file_bound = True


def test_source_clock_chain_schema_and_exact_primitives_are_re_admitted(source):
    records = _artificial_records(source)
    zero = asdict(I(0))
    corruptions = (
        ((0, "schema"), "another-protocol"),
        ((0, "horizon"), Q(7)),
        ((0, "step"), Q(1, 16)),
        ((0, "order"), True),
        ((0, "phase_capture"), "binary64_graph_recapture"),
        ((1, "nominal", 0), records[1]["nominal"][0] + Q(1, 8)),
        ((1, "clock"), "rescaled time"),
        ((1, "intended_coordinate_error"), Q(1, 2**19)),
        ((1, "initial_box", 0, "lo"), False),
        ((2, 1, "cell_index"), 0),
        ((2, 1, "step", "time"), Q(0)),
        ((2, 0, "step", "duration"), 0.125),
        ((2, 0, "step", "order"), True),
        ((2, 0, "step", "method"), "an-unbound-numerical-method"),
        ((2, 0, "step", "picard_interior_margin"), Q(0)),
        ((2, 0, "step", "domain_lower_bounds"), (True,)),
        ((2, 1, "step", "initial_box", 0), zero),
        ((2, 0, "step", "tube", 0), zero),
        ((2, 0, "step", "series", 0, 0), zero),
        ((2, 0, "step", "increment", 0), asdict(I(Q(1, 8)))),
        ((2, 0, "step", "endpoint", 0), zero),
        ((2, 0, "step", "local_remainder_bounds", 0, "lo"), Q(1, 3)),
        ((2, 0, "step", "series", 0), (zero,) * 8),
    )
    for path, value in corruptions:
        changed = _replace_path(records, path, value)
        with pytest.raises((TypeError, ValueError)):
            observer.admit_retained_records(*changed)
    for cells in (records[2][:-1], records[2] + (records[2][-1],)):
        with pytest.raises((TypeError, ValueError)):
            observer.admit_retained_records(records[0], records[1], cells)
    changed = deepcopy(records)
    changed[2][0]["step"]["unregistered_bound"] = 1
    with pytest.raises(ValueError):
        observer.admit_retained_records(*changed)


def test_all_raw_coordinates_precede_phase_work_and_invalid_types_never_alias(
    monkeypatch,
):
    def forbidden():
        raise AssertionError("phase work preceded complete input admission")

    monkeypatch.setattr(preparation, "pi_interval", forbidden)
    monkeypatch.setattr(observer, "pi_interval", forbidden)
    for value in (True, False, 0.0, float("nan"), float("inf"), "0", complex(0)):
        with pytest.raises((TypeError, ValueError)):
            observer.observe_target((Q(0),) * 19 + (value,))
    for size in (0, 19, 21):
        with pytest.raises((TypeError, ValueError)):
            observer.observe_target((Q(0),) * size)
    for lower, upper in ((False, Q(0)), (Q(1), Q(0))):
        forged = I(0)
        object.__setattr__(forged, "lo", lower)
        object.__setattr__(forged, "hi", upper)
        with pytest.raises((TypeError, ValueError)):
            observer.observe_target((Q(0),) * 19 + (forged,))


def test_file_receipts_bind_artificial_records_and_reject_changed_bytes(
    source, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("file admission invoked the differential law")

    monkeypatch.setattr(preparation, "complete_field", forbidden)
    monkeypatch.setattr(preparation, "validated_box_taylor_step", forbidden)
    anchor = _write_artificial_repository(tmp_path, source)
    admitted = observer.read_retained_trajectory(tmp_path, final_state_receipt=anchor)
    assert admitted.file_bound and admitted.cell_count == 48
    assert admitted.initial_box == admitted.endpoint == source.initial_box
    assert (
        not admitted.execution_authenticated
        and not admitted.derivative_or_picard_recomputed
    )
    names = {row["path"] for row in admitted.receipts}
    assert observer.FINAL_STATE_PATH in names
    assert all(name in names for name in observer.REQUIRED_SOURCE_NAMES)
    assert all(
        observer.PRIOR_DIRECTORY + f"/trajectory/cell-{index:03d}.json" in names
        for index in range(48)
    )
    # Rejection is based on the separately retained anchor, not a fresh
    # receipt generated from the altered payload by the consumer.
    for name in (
        observer.PRIOR_DIRECTORY + "/trajectory/cell-023.json",
        observer.PRIOR_DIRECTORY + "/trajectory/source.json",
        observer.PRIOR_DIRECTORY + "/evaluated-source.zip",
        observer.REQUIRED_SOURCE_NAMES[0],
        observer.FINAL_STATE_PATH,
    ):
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises((TypeError, ValueError)):
            observer.read_retained_trajectory(tmp_path, final_state_receipt=anchor)
        path.write_bytes(original)
    for invalid in (
        dict(anchor, path="../outside.json"),
        dict(anchor, path=observer.PRIOR_DIRECTORY + "/source.json"),
        dict(anchor, bytes=True),
        dict(anchor, sha256="0" * 64),
    ):
        with pytest.raises((TypeError, ValueError)):
            observer.read_retained_trajectory(tmp_path, final_state_receipt=invalid)
