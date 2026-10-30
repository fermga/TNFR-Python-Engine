"""Full-state capture from an independently bounded stationary reference."""

import pickle
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_reduction import (
    bound_sine_slow_phase,
    certify_sine_slow_capture,
)
from tnfr.sdk import export_to_json
from tnfr.utils.io import json_loads

MODEL = RelationalExchangeModel(
    1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
)
TARGET = tuple(Q(i - 2, 5) for i in range(5))
SIGMA = Q(1, 16)


def _graph(*, form=None, phase=None, shift_x=0, shift_theta=0):
    graph = nx.cycle_graph(5)
    for i in graph:
        graph.nodes[i].update(
            EPI=shift_x + (4092 * (i - 2) if form is None else form[i]),
            theta=shift_theta + (0 if phase is None else phase[i]),
            nu_f=1,
        )
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _source(**kwargs):
    return bound_relational_sine_exchange(_graph(**kwargs), reference_model=MODEL)


def _pattern(**kwargs):
    return bound_relational_sine_pattern(
        _graph(**kwargs),
        reference_model=MODEL,
        reference_node=0,
        form_error_bounds=(Q(1, 16),) * 5,
        phase_error_bounds=(Q(1, 65536),) * 5,
    )


def _certify(source, **kwargs):
    return certify_sine_slow_capture(
        source, **{"slow_time": SIGMA, "target_phase_turns": TARGET, **kwargs}
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.fixture(scope="module")
def capture():
    return _certify(_source())


@pytest.fixture(scope="module")
def family():
    return _certify(_pattern())


def test_published_preparation_passes_independent_full_storage_and_boundary_check(
    capture,
):
    report, slow = capture.capture, capture.slow_phase
    assert capture.admitted and report.cycle_periods == (1,)
    assert capture.target_geometry.sine_balance_status == "proved_by_odd_cancellation"
    assert report.observation_time is report.input_forecast_admitted is None
    assert slow.slow_time == SIGMA and slow.horizon_bounds.lo > 600000
    assert report.energy_margin > Q(7, 1000)
    assert report.storage_upper_bound < Q(3462, 1000)
    assert all(m.lo > Q(1, 5) for m in report.edge_acute_margin_bounds)
    assert slow.initial_storage_bounds.lo > 160000000
    # Independent exact C5 potential and boundary minimum, not a trajectory.
    with mp.workdps(95):
        target_potential = 5 * (1 - mp.cos(2 * mp.pi / 5))
        face = 1 + 4 * (1 - mp.cos(3 * mp.pi / 8))
        mismatch = mp.sqrt(20) * abs(4 / mp.pi - 2 * mp.pi / 5)
        assert _mp(capture.nominal_initial_reference_distance_bounds.lo) <= mismatch
        assert mismatch <= _mp(capture.nominal_initial_reference_distance_bounds.hi)
        assert (
            _mp(capture.target_phase_storage_bounds.lo)
            <= target_potential
            <= _mp(capture.target_phase_storage_bounds.hi)
        )
        assert _mp(report.boundary_storage_lower_bound) <= face
        assert _mp(report.storage_upper_bound) < target_potential + mp.mpf("0.0071")
    assert capture.source is slow.source is report.source


def test_critical_cancellation_retains_information_lost_by_independent_edge_boxes(
    capture,
):
    report = capture.capture
    independent = sum((1 - cos(gap) for gap in report.phase_edge_gap_bounds), I(0))
    assert independent.hi > report.boundary_phase_storage_lower_bound
    assert report.phase_storage_bounds.hi < report.boundary_phase_storage_lower_bound
    assert report.phase_storage_bounds.lo == independent.lo
    assert (
        report.storage_bounds
        == report.form_storage_bounds + report.phase_storage_bounds
    )
    assert (
        report.energy_margin
        == report.boundary_storage_lower_bound - report.storage_upper_bound
    )


def test_original_uncertain_family_keeps_memberwise_reference_and_unobserved_means(
    family,
):
    assert family.admitted and family.capture.energy_margin > Q(7, 1000)
    assert (
        family.capture.weighted_form_mean is family.capture.weighted_phase_mean is None
    )
    assert family.slow_phase.reference_initial_phase_bounds is None
    assert family.initial_reference_uncertainty_upper_bound > 0
    # Actual residual corners, with unrelated common origins, retain the
    # centered reference distance. They do not share one nominal phase path.
    with mp.workdps(95):
        alpha = 1 / (1023 * mp.pi)
        target = [2 * mp.pi * _mp(q) for q in TARGET]
        for signs in ((1, -1, 1, -1, 1), (-1, 1, -1, 1, -1)):
            x = [
                _mp(v) + sign / mp.mpf(16) + 11
                for v, sign in zip(family.source.nominal_form, signs)
            ]
            phase = [-sign / mp.mpf(65536) - 7 for sign in signs]
            xmean, mean_phase = sum(x) / 5, sum(phase) / 5
            distance = mp.sqrt(
                2
                * sum(
                    (t - mean_phase + alpha * (v - xmean) - q) ** 2
                    for t, v, q in zip(phase, x, target)
                )
            )
            assert distance <= _mp(family.initial_reference_distance_upper_bound)


def test_critical_quadratic_storage_bound_on_independent_deformations(capture):
    # Verify the analytic Hessian majorant using independent trigonometry,
    # including a deformation beyond the acute target neighborhood.
    with mp.workdps(95):
        target = [2 * mp.pi * _mp(q) for q in TARGET]
        base = 5 * (1 - mp.cos(2 * mp.pi / 5))
        for deformation in ((Q(1, 32), -Q(1, 32), 0, 0, 0), (2, -2, 1, 0, -1)):
            delta = [_mp(v) for v in deformation]
            mean = sum(delta) / 5
            phase = [t + d for t, d in zip(target, delta)]
            actual = sum(1 - mp.cos(phase[(i + 1) % 5] - phase[i]) for i in range(5))
            squared_norm = 2 * sum((d - mean) ** 2 for d in delta)
            assert actual <= base + squared_norm


def test_good_phase_geometry_does_not_override_excess_form_storage():
    source = _source(
        form=(1, -1, 0, 0, 0), phase=tuple(Q(1287 * (i - 2), 1024) for i in range(5))
    )
    report = _certify(source, slow_time=0)
    assert not report.admitted
    assert all(v.lo > 0 for v in report.capture.edge_acute_margin_bounds)
    assert (
        report.capture.phase_storage_bounds.hi
        < report.capture.boundary_phase_storage_lower_bound
    )
    assert report.capture.energy_margin < 0
    assert report.capture.unresolved_conditions == (
        "strict_total_storage_boundary_barrier_not_certified",
    )


def test_uniform_stationary_state_cannot_be_captured_in_another_sector():
    report = _certify(_source(form=(7,) * 5, phase=(0,) * 5))
    assert not report.admitted
    assert (
        "whole_source_set_strict_acute_sector_not_certified"
        in report.capture.unresolved_conditions
    )


def test_zero_time_does_not_discard_the_initial_transient(capture):
    report = _certify(capture.source, slow_time=0)
    assert report.slow_phase.composite_phase_error_upper_bound == 0
    assert report.slow_phase.phase_error_upper_bound > 0
    assert not report.admitted
    assert (
        report.actual_phase_distance_upper_bound > report.reference_distance_upper_bound
    )


def test_common_origins_and_target_origin_preserve_the_complete_certificate(capture):
    shifted = _certify(
        _source(shift_x=Q(13, 2), shift_theta=Q(-7, 4)),
        target_phase_turns=tuple(q + 7 for q in TARGET),
    )
    assert shifted.admitted
    assert (
        shifted.initial_reference_distance_upper_bound
        == capture.initial_reference_distance_upper_bound
    )
    assert shifted.capture.storage_bounds == capture.capture.storage_bounds
    assert shifted.capture.edge_turn_offsets == capture.capture.edge_turn_offsets
    assert shifted.capture.weighted_form_mean == Q(13, 2)
    assert shifted.capture.weighted_phase_mean == Q(-7, 4)


def test_target_lifts_are_declared_not_fitted_to_get_a_certificate(capture):
    changed = _certify(capture.source, target_phase_turns=(TARGET[0] + 1,) + TARGET[1:])
    assert changed.target_edge_turns == capture.target_edge_turns
    assert changed.capture.cycle_periods == capture.capture.cycle_periods
    assert not changed.admitted
    assert (
        changed.initial_reference_distance_upper_bound
        > capture.initial_reference_distance_upper_bound
    )


@pytest.mark.parametrize(
    "target",
    (
        (0,) * 4,
        (True,) * 5,
        tuple(float(q) for q in TARGET),
        (0, Q(1, 32), 0, 0, 0),
        (0, Q(1, 2), 0, 0, 0),
    ),
)
def test_invalid_or_unproved_critical_targets_reject(target):
    with pytest.raises((TypeError, ValueError)):
        _certify(_source(), target_phase_turns=target)


def test_replaced_bound_fields_cannot_forge_a_capture():
    source = _source(form=(7,) * 5, phase=(0,) * 5)
    slow = bound_sine_slow_phase(source, slow_time=SIGMA)
    forged = replace(
        slow,
        phase_error_upper_bound=Q(0),
        form_norm_upper_bound=Q(0),
        exponential_growth_bounds=I(0),
    )
    result = forged.certify_capture(target_phase_turns=TARGET)
    assert not result.admitted
    assert result.slow_phase == slow


def test_pattern_adapter_reuses_original_residuals_not_cached_storage_or_boxes(family):
    forged = replace(
        family.source,
        relative_form_bounds=(I(0),) * 5,
        relative_phase_bounds=(I(0),) * 5,
        storage_bounds=I(0),
    )
    result = forged.certify_slow_capture(slow_time=SIGMA, target_phase_turns=TARGET)
    assert result.admitted
    assert result.capture.storage_bounds == family.capture.storage_bounds
    assert (
        result.initial_reference_distance_upper_bound
        == family.initial_reference_distance_upper_bound
    )


def test_heterogeneous_capacity_and_canonical_edge_order_are_retained():
    graph = nx.Graph((("c", "a"), ("a", "b"), ("c", "b")))
    for i, node in enumerate(graph):
        graph.nodes[node].update(EPI=Q(i - 1, 1024), theta=Q(i - 1, 2048), nu_f=i + 1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(graph, reference_model=MODEL)
    result = certify_sine_slow_capture(
        source, slow_time=SIGMA, target_phase_turns=(0,) * 3
    )
    assert result.admitted and result.capture.cycle_periods == (0,)
    assert result.capture.nodes == source.nodes
    assert result.capture.exact_held_capacity == source.capacity
    assert result.capture.weighted_form_mean != sum(source.epi) / 3
    assert result.target_geometry.geometry == result.capture.geometry


def test_sdk_export_detachment_and_irrational_clock_provenance(tmp_path):
    graph = _graph()

    def snapshot():
        return pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
            protocol=5,
        )

    before = snapshot()
    source = bound_relational_sine_exchange(graph, reference_model=MODEL)
    result = _certify(source)
    assert result.capture.source is result.source is source
    assert snapshot() == before
    assert result.slow_phase.horizon_bounds.lo < result.slow_phase.horizon_bounds.hi
    assert result.capture.observation_time is None
    path = tmp_path / "slow-capture.json"
    export_to_json(result, path)
    payload = json_loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tnfr.relational-sine-slow-capture.v1"
    assert payload["report"]["capture"]["status"] == "admitted"
    assert payload["report"]["capture"]["observation_time"] is None
