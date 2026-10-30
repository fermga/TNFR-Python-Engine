"""Constructed controls for the fixed-window direct observation reader.

Point and interval preparations below are manufactured arithmetic controls,
not retained trajectories. Synthetic cell ledgers test accumulation and
availability only; they establish no complete-law trajectory or organization.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics import _validated_taylor as taylor
from tnfr.mathematics._rational_interval import I, pi_interval, sin, sqrt
from tnfr.research import maintained_sha_direct_response as owner
from tnfr.research import maintained_sha_motion as motion

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
STEP = Q(1, 512)


@pytest.fixture(autouse=True)
def no_retained_source_or_evolution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constructed direct controls cannot read or evolve a source")

    monkeypatch.setattr(owner, "load_sha_retention", forbidden)
    monkeypatch.setattr(taylor, "picard_tube", forbidden)
    for name in (
        "validate_live_sha_successor",
        "validated_box_taylor_step",
        "certify_retention",
        "prepare_sha_motion",
    ):
        monkeypatch.setattr(motion, name, forbidden)


def _overlap(actual, expected):
    assert (actual - expected).contains(0)


def _source(
    *,
    forms=(Q(-1, 8), Q(1, 8), Q(0), Q(0)),
    phases=(Q(-1, 16), Q(-1, 16), Q(-1, 16), Q(3, 16)),
    capacities=(2, 1, 2, 1),
    form_shift=Q(0),
    phase_shift=Q(0),
):
    graph = nx.Graph()
    for node, form, phase, capacity in zip(
        ("parent", "child", "rest", "remote"),
        forms,
        phases,
        capacities,
        strict=True,
    ):
        graph.add_node(
            node,
            EPI=form + form_shift,
            theta=phase + phase_shift,
            nu_f=capacity,
            delta_nfr=99,
        )
    graph.add_weighted_edges_from(
        (("parent", "child", 2), ("parent", "rest", 3), ("rest", "remote", 5))
    )
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    return owner._capture_direct_source(
        graph, parent="parent", child="child", reference_model=MODEL
    )


def _point(source):
    return tuple(map(I, source.epi + source.phase))


@pytest.fixture(scope="module")
def complete_path_case():
    source = _source()
    return source, owner._bound_cell(source, _point(source))


def test_full_parent_environment_and_original_clock_enter_the_direct_angle(
    complete_path_case,
):
    source, cell = complete_path_case
    pi, root_w = pi_interval(), sqrt(I(2))
    assert source.nodes == ("parent", "child", "rest", "remote")
    assert source.reference_conductance == 2
    assert cell.r.contains(Q(1, 4))
    assert cell.delta.contains(0)
    assert cell.r_dot.contains(0)
    # All four mobilities equal 1/pi. The external parent edge contributes
    # 3*r/(2*pi), in addition to the combined internal contribution 4*r/pi.
    _overlap(cell.delta_dot, 11 / (8 * pi))
    assert cell.radius_squared.contains(Q(1, 8))
    assert cell.radius_squared.lo > 0
    _overlap(cell.angular_numerator, -11 * root_w / (32 * pi))
    _overlap(cell.angular_rate, -11 / (2 * pi * root_w))
    _overlap(cell.angular_increment, -11 * STEP / (2 * pi * root_w))
    assert cell.orientation == "clockwise"
    assert all(cell.means_consistent)
    assert cell.form_mean.contains(0)
    assert cell.phase_mean.contains(0)
    assert cell.unavailable_reasons == ()
    with pytest.raises(FrozenInstanceError):
        cell.angular_rate = I(0)


def test_the_delta_times_form_rate_term_has_the_correct_sign_on_the_phase_axis():
    source = _source(forms=(0, 0, 0, 0), phases=(0, Q(1, 4), 0, 0))
    cell = owner._bound_cell(source, _point(source))
    pi, root_w = pi_interval(), sqrt(I(2))
    assert cell.r.contains(0)
    assert cell.delta.contains(Q(1, 4))
    _overlap(cell.r_dot, -2 * sin(I(Q(1, 4))) / pi)
    assert cell.delta_dot.contains(0)
    _overlap(cell.angular_numerator, -root_w * sin(I(Q(1, 4))) / (2 * pi))
    _overlap(cell.angular_rate, -8 * root_w * sin(I(Q(1, 4))) / pi)
    assert cell.orientation == "clockwise"


def test_unequal_parent_and_child_mobilities_weight_both_complete_rows():
    source = _source(capacities=(2, 3, 2, 1))
    cell = owner._bound_cell(source, _point(source))
    assert source.degrees == (2, 1, 2, 1)
    assert source.beta == (Q(1), Q(3), Q(1), Q(1))
    # Parent mobility is 1/pi and child mobility is 3/pi. Their phase
    # rows are -7/(8*pi) and 3/(2*pi), respectively, at this point.
    _overlap(cell.delta_dot, 19 / (8 * pi_interval()))
    _overlap(cell.angular_rate, -19 / (2 * pi_interval() * sqrt(I(2))))
    assert all(cell.means_consistent)
    phase_source = _source(
        forms=(0, 0, 0, 0),
        phases=(0, Q(1, 4), 0, 0),
        capacities=(2, 3, 2, 1),
    )
    phase_cell = owner._bound_cell(phase_source, _point(phase_source))
    _overlap(phase_cell.r_dot, -4 * sin(I(Q(1, 4))) / pi_interval())
    assert phase_cell.delta_dot.contains(0)
    assert phase_cell.orientation == "clockwise"


def test_full_environment_can_reverse_the_instantaneous_original_orientation():
    source = _source(forms=(0, 1, -10, 0), phases=(0, 0, 0, 0))
    cell = owner._bound_cell(source, _point(source))
    pi, root_w = pi_interval(), sqrt(I(2))
    assert cell.r.contains(1)
    assert cell.delta.contains(0)
    _overlap(cell.delta_dot, -26 / pi)
    _overlap(cell.angular_rate, 13 * root_w / pi)
    assert cell.orientation == "counterclockwise"
    assert cell.angular_increment.lo > 0


def test_common_form_and_phase_shifts_preserve_the_marked_observation(
    complete_path_case,
):
    _, original = complete_path_case
    shifted_source = _source(form_shift=Q(2), phase_shift=Q(1, 2))
    shifted = owner._bound_cell(shifted_source, _point(shifted_source))
    for name in (
        "r",
        "delta",
        "r_dot",
        "delta_dot",
        "radius_squared",
        "angular_numerator",
        "angular_rate",
        "angular_increment",
    ):
        _overlap(getattr(shifted, name), getattr(original, name))
    assert shifted.orientation == original.orientation
    assert all(shifted.means_consistent)
    assert shifted.form_mean.contains(2)
    assert shifted.phase_mean.contains(Q(1, 2))


def test_zero_radius_keeps_the_angle_unavailable_without_dividing():
    source = _source(forms=(0, 0, 0, 0), phases=(0, 0, 0, 0))
    cell = owner._bound_cell(source, _point(source))
    assert cell.radius_squared.lo == 0
    assert cell.angular_numerator.contains(0)
    assert cell.angular_rate is None
    assert cell.angular_increment is None
    assert cell.orientation == "unavailable"
    assert cell.unavailable_reasons


def test_a_positive_radius_can_have_an_exactly_stationary_instantaneous_angle():
    source = _source(forms=(0, 3, -4, 0), phases=(0, 0, 0, 0))
    cell = owner._bound_cell(source, _point(source))
    # The signed phase contrast combines to 4*3+3*(-4)=0 before pi.
    # This is an instantaneous complete-law state, not a stationary orbit.
    assert cell.radius_squared.contains(18)
    assert cell.radius_squared.lo > 0
    assert cell.angular_rate == I(0)
    assert cell.angular_increment == I(0)
    assert cell.orientation == "stationary"


def test_a_box_containing_zero_does_not_assert_that_its_initial_point_is_zero(
    complete_path_case,
):
    source, initial = complete_path_case
    assert initial.radius_squared.lo > 0
    tube = tuple(
        I(value - Q(1, 2), value + Q(1, 2)) for value in source.epi + source.phase
    )
    enclosed = owner._bound_cell(source, tube)
    assert enclosed.radius_squared.lo == 0
    assert enclosed.angular_rate is None
    assert enclosed.orientation == "unavailable"
    assert all(enclosed.means_consistent)


def test_each_cell_increment_uses_its_declared_structural_duration(complete_path_case):
    source, _ = complete_path_case
    short = owner._bound_cell(source, _point(source), duration=Q(1, 8))
    long = owner._bound_cell(source, _point(source), duration=Q(1, 4))
    _overlap(short.angular_rate, long.angular_rate)
    _overlap(long.angular_increment, 2 * short.angular_increment)


def _ledger(base, rates):
    """Construct numerical report inputs; this is not a claimed trajectory."""
    rows = []
    for index, rate in enumerate(rates):
        rate = I.coerce(rate)
        orientation = (
            "clockwise"
            if rate.hi < 0
            else (
                "counterclockwise"
                if rate.lo > 0
                else "stationary" if rate.lo == rate.hi == 0 else "unresolved"
            )
        )
        rows.append(
            replace(
                base,
                index=index,
                time=index * STEP,
                duration=STEP,
                angular_rate=rate,
                angular_increment=rate * STEP,
                orientation=orientation,
            )
        )
    return tuple(rows)


def test_a_net_clockwise_turn_does_not_require_clockwise_orientation_in_every_cell(
    complete_path_case,
):
    _, base = complete_path_case
    ledger = _ledger(base, (Q(-20),) * 255 + (Q(1),))
    result = owner._aggregate(ledger, full_window=True)
    assert result.full_window
    assert result.accepted_duration == Q(1, 2)
    assert result.clockwise_travel.contains(Q(5099, 512))
    assert result.angular_increment.contains(Q(-5099, 512))
    assert result.net_turn_gate.status == "pass"
    assert not result.all_clockwise
    assert dict(result.orientation_counts)["counterclockwise"] == 1
    assert result.radius_gate.status == "pass"


def test_direct_upper_bound_can_exclude_the_fixed_net_turn_without_excluding_motion(
    complete_path_case,
):
    _, base = complete_path_case
    result = owner._aggregate(_ledger(base, (Q(-1),) * 256), full_window=True)
    assert result.clockwise_travel.contains(Q(1, 2))
    assert result.net_turn_gate.status == "excluded"
    assert result.all_clockwise
    assert result.radius_gate.status == "pass"


def test_partial_prefix_and_one_unavailable_denominator_cannot_supply_a_net_turn(
    complete_path_case,
):
    source, base = complete_path_case
    full = _ledger(base, (Q(-20),) * 256)
    partial = owner._aggregate(full[:-1], full_window=True)
    assert not partial.full_window
    assert partial.net_turn_gate.status == "unavailable"
    wide = owner._bound_cell(
        source,
        tuple(I(value - 1, value + 1) for value in source.epi + source.phase),
        index=255,
        time=255 * STEP,
        duration=STEP,
    )
    unavailable = owner._aggregate(full[:-1] + (wide,), full_window=True)
    assert unavailable.full_window
    assert unavailable.angular_increment is None
    assert unavailable.clockwise_travel is None
    assert unavailable.net_turn_gate.status == "unavailable"


def test_aggregation_rejects_a_changed_cell_clock_or_grid(complete_path_case):
    _, base = complete_path_case
    ledger = _ledger(base, (Q(-20),) * 256)
    for changed in (
        replace(ledger[1], time=Q(0)),
        replace(ledger[1], duration=2 * STEP),
        replace(ledger[1], index=2),
    ):
        with pytest.raises(ValueError):
            owner._aggregate(ledger[:1] + (changed,) + ledger[2:], full_window=True)


@pytest.mark.parametrize(
    "margin,status",
    (
        (I(0), "pass"),
        (I(0, 1), "pass"),
        (I(-2, -1), "excluded"),
        (I(-1, 0), "unavailable"),
        (I(-1, 1), "unavailable"),
        (None, "unavailable"),
    ),
)
def test_net_turn_threshold_is_closed_and_exclusion_needs_a_strict_upper_gap(
    margin,
    status,
):
    assert owner._net_turn_gate(margin).status == status


def test_zero_radius_lower_bound_is_unavailable_even_at_the_closed_turn_threshold():
    assert owner._radius_gate(I(0)).status == "unavailable"
    assert owner._radius_gate(I(0, 1)).status == "unavailable"
    assert owner._radius_gate(None).status == "unavailable"
    assert owner._radius_gate(I(1, 2)).status == "pass"
    assert owner._net_turn_gate(I(0)).status == "pass"


def test_public_wrapper_reloads_the_retained_evidence_before_observing(monkeypatch):
    fresh, expected = object(), object()
    calls = []

    def load(path, *, repository_root):
        assert path == "constructed-evidence"
        assert repository_root == "constructed-repository"
        calls.append("readmit_complete_retention")
        return fresh

    def consume(value):
        assert value is fresh
        calls.append("observe_fresh_primitives")
        return expected

    monkeypatch.setattr(owner, "load_sha_retention", load)
    monkeypatch.setattr(owner, "_audit_from_fresh_retention", consume)
    assert (
        owner.audit_sha_direct_response(
            "constructed-evidence", repository_root="constructed-repository"
        )
        is expected
    )
    assert calls == ["readmit_complete_retention", "observe_fresh_primitives"]


def test_internal_consumer_rejects_a_cached_unadmitted_report():
    with pytest.raises(TypeError):
        owner._audit_from_fresh_retention(object())
