"""Full-source uncertainty and independent entry-box conservative passage.

These are admission, exact-algebra and interval-consumer controls. No numerical
trajectory or archived scientific response is evaluated.
"""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES, _source
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics.relational_sine_entry import certify_sine_conservative_winding_entry
from tnfr.sdk.relational_reports import relational_report_to_dict

ERROR = Q(1, 4096)
ENTRY = Q(11, 512)
END = Q(3, 128)


@pytest.fixture(scope="module")
def source():
    return replace(
        _source(CYCLE_LEAF_EDGES),
        epi=(Q(0),) * 5 + tuple(Q(-192 * (i - 2)) for i in range(5)),
        phase=(Q(0),) * 10,
    )


def _assess(source, **changes):
    args = dict(
        cycle=tuple(range(5)),
        scaled_window=(ENTRY, END),
        edge_turn_offsets=(0, 0, 0, 0, -1),
        source_error_bound=ERROR,
    )
    return certify_sine_conservative_winding_entry(source, **(args | changes))


@pytest.fixture(scope="module")
def report(source):
    return _assess(source)


def test_independent_source_box_enters_and_its_full_target_box_retains(report):
    assert report.source_error_bound == ERROR
    assert report.source_initial_zero_winding_certified
    assert report.initial_winding == 0
    assert report.entry_box_contains_source_flow
    assert report.entry_box_acute_retention_certified
    assert report.acute_acquisition_certified
    assert report.certified_winding == 1
    assert report.entry_form_radius == Q(89, 4096)
    assert report.entry_phase_radius == Q(751, 1048576)
    assert report.cycle_gap_remainder_bound == Q(211, 131072)
    assert report.scaled_retention_duration == Q(1, 512)
    assert report.acute_margin_lower_bound > Q(1, 16)
    assert len(report.entry_form_bounds) == len(report.entry_phase_bounds) == 10
    assert report.initial_total_storage == 184320
    assert report.initial_storage_bounds.contains(report.initial_total_storage)
    assert report.initial_storage_bounds.width > 0


def _check_independent_target_continuation(report):
    # Apply complete graph rows directly to all target form intervals. This
    # does not assume an arbitrary target point is reachable from the source.
    neighbors = [set() for _ in range(10)]
    for i, j in CYCLE_LEAF_EDGES:
        neighbors[i].add(j)
        neighbors[j].add(i)
    h = report.scaled_retention_duration
    # Exact rational interval endpoints keep this independent oracle from
    # introducing extra outward rounding at a mathematically tight boundary.
    phase_rates = tuple(
        (
            sum(
                (
                    report.entry_form_bounds[i].lo - report.entry_form_bounds[j].hi
                    for j in row
                ),
                Q(0),
            )
            / len(row),
            sum(
                (
                    report.entry_form_bounds[i].hi - report.entry_form_bounds[j].lo
                    for j in row
                ),
                Q(0),
            )
            / len(row),
        )
        for i, row in enumerate(neighbors)
    )
    for i in range(10):
        future_form_lo = report.entry_form_bounds[i].lo - h
        future_form_hi = report.entry_form_bounds[i].hi + h
        rate_lo, rate_hi = phase_rates[i]
        future_phase_lo = report.entry_phase_bounds[i].lo + min(0, h * rate_lo) - h**2
        future_phase_hi = report.entry_phase_bounds[i].hi + max(0, h * rate_hi) + h**2
        form_center = report.source.epi[i]
        assert form_center - report.entry_box_form_remainder_bound <= future_form_lo
        assert future_form_hi <= form_center + report.entry_box_form_remainder_bound
        phase_center = report.source.phase[i]
        speed = report.initial_phase_velocity[i]
        first = phase_center + report.scaled_window[0] * speed
        last = phase_center + report.scaled_window[1] * speed
        assert (
            min(first, last) - report.entry_box_phase_remainder_bound <= future_phase_lo
        )
        assert (
            future_phase_hi <= max(first, last) + report.entry_box_phase_remainder_bound
        )
    assert report.entry_phase_radius + 2 * h * report.entry_form_radius + h**2 == (
        report.entry_box_phase_remainder_bound
    )
    assert report.entry_form_radius + h == report.entry_box_form_remainder_bound


def test_every_independent_target_box_member_has_the_reported_future_enclosure(report):
    _check_independent_target_continuation(report)


def test_non_dyadic_entry_materialization_is_retained_in_future_enclosure(source):
    source = replace(
        source,
        epi=tuple(value + Q(1, 3) for value in source.epi),
        phase=(Q(1, 7),) * len(source.nodes),
    )
    start, end = Q(1, 47), Q(1, 43)
    report = _assess(source, source_error_bound=Q(1, 4095), scaled_window=(start, end))
    assert report.entry_box_acute_retention_certified
    assert report.analytic_entry_form_radius == start + Q(1, 4095)
    assert report.analytic_entry_phase_radius == start**2 + Q(1, 4095) * (1 + 2 * start)
    assert report.entry_form_radius > report.analytic_entry_form_radius
    assert report.entry_phase_radius > report.analytic_entry_phase_radius
    assert report.node_form_remainder_bound == end + Q(1, 4095)
    assert report.node_phase_remainder_bound == end**2 + Q(1, 4095) * (1 + 2 * end)
    assert report.entry_box_form_remainder_bound > report.node_form_remainder_bound
    assert report.entry_box_phase_remainder_bound > report.node_phase_remainder_bound
    for center, interval in zip(source.epi, report.entry_form_bounds):
        assert (
            max(center - interval.lo, interval.hi - center) <= report.entry_form_radius
        )
    _check_independent_target_continuation(report)


def test_source_form_errors_require_the_linear_initial_phase_velocity_term(source):
    epsilon = ERROR
    forms = list(source.epi)
    forms[0] += epsilon
    for j in (1, 4, 5):
        forms[j] -= epsilon
    nominal = sum((source.epi[0] - source.epi[j] for j in (1, 4, 5)), Q(0)) / 3
    perturbed = sum((forms[0] - forms[j] for j in (1, 4, 5)), Q(0)) / 3
    assert perturbed - nominal == 2 * epsilon
    report = _assess(source)
    assert report.entry_phase_radius == ENTRY**2 + epsilon + ENTRY * (
        perturbed - nominal
    )


def test_initial_storage_encloses_independent_uncertain_source_states(source, report):
    with mp.workdps(80):

        def number(value):
            value = Q(value)
            return mp.mpf(value.numerator) / value.denominator

        # The two corners differ in both form and phase on every fine node.
        for orientation in (-1, 1):
            forms = [
                number(x + orientation * (1 if i % 2 else -1) * ERROR)
                for i, x in enumerate(source.epi)
            ]
            phases = [
                number(orientation * (1 if i % 3 else -1) * ERROR) for i in range(10)
            ]
            total = mp.fsum(
                (forms[j] - forms[i]) ** 2 / 2 + 1 - mp.cos(phases[j] - phases[i])
                for i, j in CYCLE_LEAF_EDGES
            )
            assert number(report.initial_storage_bounds.lo) <= total
            assert total <= number(report.initial_storage_bounds.hi)


def test_zero_error_preserves_original_nominal_certificate_and_schema(source):
    original = certify_sine_conservative_winding_entry(
        source,
        cycle=tuple(range(5)),
        scaled_window=(ENTRY, END),
        edge_turn_offsets=(0, 0, 0, 0, -1),
    )
    explicit = _assess(source, source_error_bound=0)
    assert original == explicit
    assert original.node_form_remainder_bound == END
    assert original.node_phase_remainder_bound == END**2
    assert original.initial_winding == 0
    assert original.entry_box_acute_retention_certified
    assert original.initial_storage_bounds.contains(original.initial_total_storage)
    assert original.to_dict()["schema"] == "tnfr.sine-conservative-winding-entry.v1"


def test_large_source_chart_and_wide_retention_fail_without_claiming_dynamics_failure(
    source,
):
    large = _assess(source, source_error_bound=pi_interval().hi / 2)
    assert not large.source_initial_zero_winding_certified
    assert large.initial_winding is None
    assert large.entry_box_contains_source_flow
    assert not large.acquisition_certified
    assert "whole_source_box_zero_winding_not_certified" in large.unresolved
    wide = _assess(source, scaled_window=(ENTRY, Q(1, 16)))
    assert wide.source_initial_zero_winding_certified
    assert wide.entry_box_contains_source_flow
    assert not wide.entry_box_acute_retention_certified
    assert not wide.acute_acquisition_certified


def test_enclosure_does_not_have_an_artificial_short_time_majorant_domain(source):
    report = _assess(source, scaled_window=(1, 2))
    assert report.entry_box_contains_source_flow
    assert report.entry_phase_radius == 1 + 3 * ERROR
    assert report.node_phase_remainder_bound == 4 + 5 * ERROR
    assert report.status == "unavailable"


def test_caches_and_reordered_labels_do_not_change_acquisition(source, report):
    poisoned = replace(
        source,
        phase_rates=(I(999),),
        form_rates=(),
        form_gradient=(Q(100),),
        storage=I(0),
    )
    rebuilt = _assess(poisoned)
    for name in (
        "entry_form_bounds",
        "entry_phase_bounds",
        "initial_storage_bounds",
        "cycle_principal_gap_bounds",
        "acute_acquisition_certified",
    ):
        assert getattr(rebuilt, name) == getattr(report, name)
    order = (8, 3, 0, 9, 4, 2, 6, 1, 5, 7)
    relabeled = replace(
        _source(CYCLE_LEAF_EDGES, order=order, label=lambda i: f"n:{i}"),
        epi=tuple(source.epi[i] for i in order),
        phase=tuple(source.phase[i] for i in order),
    )
    mapped = _assess(relabeled, cycle=tuple(f"n:{i}" for i in range(5)))
    assert mapped.cycle_principal_gap_bounds == report.cycle_principal_gap_bounds
    assert mapped.initial_storage_bounds == report.initial_storage_bounds
    assert mapped.entry_form_bounds == tuple(report.entry_form_bounds[i] for i in order)
    assert mapped.entry_phase_bounds == tuple(
        report.entry_phase_bounds[i] for i in order
    )


@pytest.mark.parametrize(
    "error", (False, True, -1, float("nan"), float("inf"), "1/4096", None)
)
def test_source_error_admission_rejects_invalid_coordinates(source, error):
    with pytest.raises((TypeError, ValueError)):
        _assess(source, source_error_bound=error)


def test_sdk_preserves_exact_uncertainty_and_nullable_initial_winding(source, report):
    exported = relational_report_to_dict(report)
    assert exported["report_type"] == "SineConservativeWindingEntry"
    body = exported["report"]
    assert body["source_error_bound"] == {"numerator": 1, "denominator": 4096}
    assert body["entry_phase_radius"] == {"numerator": 751, "denominator": 1048576}
    assert body["entry_box_acute_retention_certified"] is True
    unavailable = relational_report_to_dict(_assess(source, source_error_bound=2))
    assert unavailable["report"]["initial_winding"] is None
    assert unavailable["report"]["source_initial_zero_winding_certified"] is False
