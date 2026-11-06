"""Analytic acquisition-to-composition controls, with no trajectory execution.

Frozen prefix preparations are two relabelled C5 copies, x_j=4092*(j-2),
theta_j=0, nu_j=1, beta=1, e=1023/1024, w=1/1024 and scaled time100.
They start at time0 and join at ports j=0 with zero relative common origins.
Both closing cycle edges have turn offset-1. The one nonzero residual family
uses form radius1/16 and phase radius1/65536 at every original node.

Tests distinguish the correlated actual endpoint set from its larger node
box, old prefix metrics from new bridge degrees, and acquisition/capture from
an independently supplied event-work allowance. No producer is replayed.
"""

import pickle
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_composition import assess_sine_prepared_composition
from tnfr.physics.relational_sine_entry import certify_sine_prepared_entry
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

MODEL = RelationalExchangeModel(
    1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
)
PREFIX_OFFSETS = (0, -1, 0, 0, 0)
JOINT_EDGES = (
    (0, 1),
    (0, 4),
    (0, 5),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (5, 9),
    (6, 7),
    (7, 8),
    (8, 9),
)
JOINT_OFFSETS = tuple(-1 if edge in ((0, 4), (5, 9)) else 0 for edge in JOINT_EDGES)
HORIZON = Q(102400, 1023)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_execution():
    import scipy.integrate

    from tnfr.physics import relational_sine_forecast

    def forbidden(*args, **kwargs):
        pytest.fail("prepared composition must reuse analytic bounds, not a solver")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        patch.setattr(relational_sine_forecast, "bound_sine_flow", forbidden)
        patch.setattr(scipy.integrate, "solve_ivp", forbidden)
        yield


def _graph(start):
    graph = nx.cycle_graph(tuple(range(start, start + 5)))
    for j, node in enumerate(graph):
        graph.nodes[node].update(EPI=4092 * (j - 2), theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _entry(start, *, form_error=Q(0), phase_error=Q(0)):
    source = bound_relational_sine_pattern(
        _graph(start),
        reference_node=start,
        reference_model=MODEL,
        form_error_bounds=(form_error,) * 5,
        phase_error_bounds=(phase_error,) * 5,
    )
    return certify_sine_prepared_entry(
        source, scaled_time=100, edge_turn_offsets=PREFIX_OFFSETS
    )


@pytest.fixture(scope="module")
def prepared_pair():
    return _entry(0), _entry(5)


@pytest.fixture(scope="module")
def uncertain_pair():
    return tuple(
        _entry(start, form_error=Q(1, 16), phase_error=Q(1, 65536)) for start in (0, 5)
    )


def _compose(pair, **changes):
    kwargs = dict(
        bridge=(0, 5),
        left_initial_time=Q(0),
        right_initial_time=Q(0),
        edge_turn_offsets=JOINT_OFFSETS,
        form_origin_difference=Q(0),
        phase_origin_difference=Q(0),
    )
    kwargs.update(changes)
    return assess_sine_prepared_composition(*pair, **kwargs)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def test_frozen_analytic_prefixes_handoff_to_full_joint_capture(prepared_pair):
    before = pickle.dumps(prepared_pair)
    report = _compose(prepared_pair)
    assert pickle.dumps(prepared_pair) == before
    assert report.status == "available"
    assert report.left.admitted and report.right.admitted and report.capture.admitted
    assert report.acquisition_and_capture_certified
    assert report.observation_time == report.capture.observation_time == HORIZON
    assert report.left.horizon == report.right.horizon == HORIZON
    assert report.edges == JOINT_EDGES
    assert report.capture.cycle_periods == (1, 1)
    assert report.capture.energy_margin > Q(9, 1000)
    assert report.bridge_storage_bounds.contains(0)
    assert 0 < report.bridge_storage_bounds.hi < Q(1, 10000)
    assert len(report.capture.boundary_face_lower_bounds) == 22
    assert report.budget_status == "not_supplied"
    assert report.capture.weighted_form_mean is None
    assert report.capture.weighted_phase_mean is None
    assert not hasattr(report, "joined")  # No fabricated independent observation box.


def test_prefix_mobility_and_correlated_internal_edges_survive_degree_change(
    prepared_pair,
):
    report = _compose(prepared_pair)
    assert report.separate_degrees == (2,) * 10
    assert report.joined_degrees == (3, 2, 2, 2, 2, 3, 2, 2, 2, 2)
    assert report.separate_form_weights == (Q(2),) * 10
    assert report.joined_form_weights == tuple(map(Q, report.joined_degrees))
    assert report.separate_mobility == (Q(1, 2),) * 10
    assert report.joined_mobility[0] == report.joined_mobility[5] == Q(1, 3)
    for original, rebuilt in zip(prepared_pair, (report.left, report.right)):
        assert rebuilt.mobility == (Q(1, 2),) * 5
        assert rebuilt.metric_weights == (Q(2),) * 5
        assert rebuilt.geometry == original.geometry
        assert (
            rebuilt.endpoint_form_edge_gap_bounds
            == original.endpoint_form_edge_gap_bounds
        )
        assert (
            rebuilt.endpoint_phase_edge_gap_bounds
            == original.endpoint_phase_edge_gap_bounds
        )
        for edge, form, phase in zip(
            rebuilt.source.edges,
            rebuilt.endpoint_form_edge_gap_bounds,
            rebuilt.endpoint_phase_edge_gap_bounds,
        ):
            index = report.edges.index(edge)
            assert report.endpoint_form_edge_gap_bounds[index] == form
            assert report.endpoint_phase_edge_gap_bounds[index] == phase


def test_new_mean_uses_old_component_charges_and_port_reweighting(prepared_pair):
    report = _compose(prepared_pair)
    form_ports = (
        report.left.centered_endpoint_form_bounds[0]
        + report.right.centered_endpoint_form_bounds[0]
    )
    phase_ports = (
        report.left.centered_endpoint_phase_bounds[0]
        + report.right.centered_endpoint_phase_bounds[0]
    )
    assert sum(report.joined_form_weights) == 22
    assert report.representative_weighted_form_mean_bounds == form_ports / 22
    assert report.representative_weighted_phase_mean_bounds == phase_ports / 22
    assert report.representative_weighted_form_mean_bounds.contains(0)
    assert 0 < report.representative_weighted_form_mean_bounds.hi < Q(1, 10000)
    with mp.workdps(90):
        expected_phase_center = -8 / (11 * mp.pi)
        phase_mean = report.representative_weighted_phase_mean_bounds
        assert _mp(phase_mean.lo) < expected_phase_center < _mp(phase_mean.hi)
    # Every node box allows this corner, but the actual prefix keeps old
    # weighted mean zero. Its Cartesian product is not the certified state set.
    corner = tuple(
        box.hi
        for item in (report.left, report.right)
        for box in item.centered_endpoint_form_bounds
    )
    independent_corner_mean = (
        sum(weight * value for weight, value in zip(report.joined_form_weights, corner))
        / 22
    )
    assert independent_corner_mean > report.representative_weighted_form_mean_bounds.hi


def test_event_allowance_is_independent_of_acquisition_and_capture(prepared_pair):
    zero_work = _compose(prepared_pair, work_allowance=0)
    supplied_work = _compose(prepared_pair, work_allowance=Q(1, 10000))
    assert zero_work.acquisition_and_capture_certified
    assert supplied_work.acquisition_and_capture_certified
    assert zero_work.budget_status == "unresolved"
    assert supplied_work.budget_status == "within_allowance"
    assert zero_work.joined_storage_bounds == supplied_work.joined_storage_bounds


@pytest.mark.parametrize(
    "missing", ["form_origin_difference", "phase_origin_difference"]
)
def test_missing_frame_keeps_prefix_acquisition_but_withholds_joint_state(
    prepared_pair, missing
):
    report = _compose(prepared_pair, **{missing: None})
    assert report.left.admitted and report.right.admitted
    assert report.status == "unavailable" and report.unavailable_reasons
    assert report.capture is None
    assert not report.acquisition_and_capture_certified
    assert report.bridge_storage_bounds is report.joined_storage_bounds is None


def test_time_association_uses_derived_horizons_and_explicit_starts(prepared_pair):
    left, right = prepared_pair
    forged = replace(right, scaled_time=Q(99), horizon=left.horizon)
    with pytest.raises((TypeError, ValueError)):
        _compose((left, forged))
    aligned = _compose((left, forged), right_initial_time=Q(1024, 1023))
    assert aligned.status == "available"
    assert aligned.right.scaled_time == 99
    assert aligned.right.horizon == Q(99 * 1024, 1023)
    assert aligned.observation_time == HORIZON
    assert aligned.source.right_initial_time + aligned.right.horizon == HORIZON
    with pytest.raises((TypeError, ValueError)):
        _compose(prepared_pair, right_initial_time=Q(1))


def test_exact_origins_and_declared_start_times_do_not_become_float_metadata(
    prepared_pair,
):
    report = _compose(
        prepared_pair,
        left_initial_time=Q(5, 3),
        right_initial_time=Q(5, 3),
        form_origin_difference=Q(1, 3),
        phase_origin_difference=Q(1, 7),
    )
    assert report.observation_time == HORIZON + Q(5, 3)
    assert type(report.observation_time) is Q
    assert report.source.form_origin_difference == Q(1, 3)
    assert report.source.phase_origin_difference == Q(1, 7)
    assert report.bridge_form_gap_bounds.contains(Q(1, 3))
    assert report.bridge_phase_gap_bounds.contains(Q(1, 7))
    assert report.left.horizon == report.right.horizon == HORIZON
    assert report.left.admitted and report.right.admitted
    assert not report.capture.admitted  # Known nonzero bridge cost exceeds this margin.


@pytest.mark.parametrize(
    "change",
    [
        {"status": "unavailable"},
        {"horizon": Q(0)},
        {"centered_endpoint_form_bounds": (I(0),) * 5},
        {"endpoint_form_edge_gap_bounds": (I(99),) * 5},
        {"scaled_form_radius_upper_bound": Q(0)},
        {"metric_weights": (Q(0),) * 5},
        {"nominal_weighted_form_mean": Q(999)},
    ],
)
def test_entry_caches_do_not_replace_analytic_prefix_evidence(prepared_pair, change):
    expected = _compose(prepared_pair)
    actual = _compose((replace(prepared_pair[0], **change), prepared_pair[1]))
    assert actual.acquisition_and_capture_certified
    assert actual.bridge_storage_bounds == expected.bridge_storage_bounds
    assert actual.joined_storage_bounds == expected.joined_storage_bounds
    assert (
        actual.representative_weighted_form_mean_bounds
        == expected.representative_weighted_form_mean_bounds
    )


def test_source_and_capture_caches_cannot_rescue_changed_preparation(prepared_pair):
    left, right = prepared_pair
    source = replace(
        left.source,
        relative_form_bounds=(I(999),) * 5,
        edge_phase_gap_bounds=(I(0),) * 5,
        storage_bounds=I(-1),
    )
    cached = replace(
        left,
        source=source,
        capture=replace(left.capture, status="unavailable", storage_bounds=I(-1)),
    )
    assert _compose((cached, right)).acquisition_and_capture_certified
    changed_source = replace(left.source, nominal_form=(Q(0),) * 5)
    changed = _compose((replace(left, source=changed_source), right))
    assert not changed.left.admitted
    assert not changed.acquisition_and_capture_certified
    assert changed.status == "available"


def test_original_residual_family_retains_memberwise_mean_uncertainty(uncertain_pair):
    report = _compose(uncertain_pair)
    assert report.status == "available"
    assert report.left.admitted and report.right.admitted
    for item in (report.left, report.right):
        assert item.source.form_error_bounds == (Q(1, 16),) * 5
        assert item.source.phase_error_bounds == (Q(1, 65536),) * 5
        assert item.weighted_form_mean is None
        assert item.endpoint_form_bounds is item.endpoint_phase_bounds is None
    # Uniform same-sign residuals alter the actual conserved means while
    # leaving the nominal centered profile unchanged. The new mean must
    # retain those allowed origins within the declared residual family.
    assert report.representative_weighted_form_mean_bounds.contains(Q(1, 16))
    assert report.representative_weighted_form_mean_bounds.contains(Q(-1, 16))
    assert report.bridge_form_gap_bounds.lo <= -Q(1, 8)
    assert report.bridge_form_gap_bounds.hi >= Q(1, 8)
    assert report.capture.weighted_form_mean is None


@pytest.mark.parametrize(
    "change",
    [
        {"nominal_form": (True,) * 5},
        {"phase_error_bounds": (Q(-1),) * 5},
        {"capacity": (Q(0),) * 5},
        {"capacity": (True,) * 5},
        {"degrees": (3,) * 5},
    ],
)
def test_invalid_original_primitives_reject_before_cached_admission(
    prepared_pair, change
):
    left, right = prepared_pair
    with pytest.raises((TypeError, ValueError)):
        _compose((replace(left, source=replace(left.source, **change)), right))


@pytest.mark.parametrize(
    "changes",
    [
        {"left_initial_time": True},
        {"right_initial_time": Q(-1)},
        {"left_initial_time": float("nan")},
        {"bridge": (0, 1)},
        {"phase_origin_difference": True},
        {"form_origin_difference": I(0, 1)},
        {"edge_turn_offsets": (0,) * 10},
    ],
)
def test_invalid_event_or_frame_declarations_reject(prepared_pair, changes):
    with pytest.raises((TypeError, ValueError)):
        _compose(prepared_pair, **changes)


def test_different_source_model_and_exact_comparison_inputs_are_unsupported(
    prepared_pair,
):
    left, right = prepared_pair
    other_model = RelationalExchangeModel(
        2, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    changed = replace(right, source=replace(right.source, reference_model=other_model))
    with pytest.raises((TypeError, ValueError)):
        _compose((left, changed))
    exact = bound_relational_sine_exchange(_graph(0), reference_model=MODEL)
    with pytest.raises((TypeError, ValueError)):
        _compose((replace(left, source=exact), right))
