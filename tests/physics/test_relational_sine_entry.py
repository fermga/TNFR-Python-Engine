"""Prepared winding acquisition: independent bounds and complete-law controls."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_prepared_entry
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.sdk import export_to_json
from tnfr.utils.io import json_loads

MODEL = RelationalExchangeModel(
    1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
)
OFFSETS = (0, -1, 0, 0, 0)


def _graph(*, sign=1, form_offset=0, phase_offset=0, capacities=None):
    graph = nx.cycle_graph(5)
    for i in graph:
        graph.nodes[i].update(
            EPI=form_offset + sign * 4092 * (i - 2),
            theta=phase_offset,
            nu_f=1 if capacities is None else capacities[i],
        )
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _source(**kwargs):
    return bound_relational_sine_exchange(_graph(**kwargs), reference_model=MODEL)


def _pattern(graph=None, *, reference=0):
    return bound_relational_sine_pattern(
        _graph() if graph is None else graph,
        reference_node=reference,
        reference_model=MODEL,
        form_error_bounds=(Q(1, 16),) * 5,
        phase_error_bounds=(Q(1, 65536),) * 5,
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.fixture(scope="module")
def entry():
    return certify_sine_prepared_entry(
        _source(), scaled_time=100, edge_turn_offsets=OFFSETS
    )


def test_phase_flat_preparation_acquires_captured_winding_with_independent_margin(
    entry,
):
    assert entry.admitted and entry.capture.admitted
    assert entry.initial_cycle_periods == (0,)
    assert entry.capture.cycle_periods == (1,)
    assert entry.scaled_time == 100 and entry.horizon == Q(102400, 1023)
    assert entry.source.form_storage == 160 * 1023**2
    assert entry.source.phase_storage.abs_max == 0
    assert entry.capture.observation_time == entry.horizon
    assert entry.capture.input_forecast_admitted is None
    assert entry.capture.weighted_form_mean == 0
    assert entry.capture.weighted_phase_mean == 0
    assert entry.capture.energy_margin > Q(1, 100)
    # Independent closed-form C5 boundary minimum and preparation-derived
    # phase profile. Neither requires an equilibrium target or a trajectory.
    with mp.workdps(90):
        barrier = 1 + 4 * (1 - mp.cos(3 * mp.pi / 8))
        v = [mp.mpf(4 * (i - 2)) / mp.pi for i in range(5)]
        potential = 4 * (1 - mp.cos(4 / mp.pi)) + 1 - mp.cos(16 / mp.pi)
        assert barrier - potential > mp.mpf("0.01345")
        assert _mp(entry.capture.storage_upper_bound) < mp.mpf("3.457")
        assert _mp(entry.capture.boundary_storage_lower_bound) <= barrier
        for box, expected in zip(entry.scaled_initial_form_bounds, v):
            assert _mp(box.lo) <= expected <= _mp(box.hi)


def test_complete_nonlinear_response_is_inside_the_analytic_endpoint(entry):
    # Independent floating integration is a regression cross-check, not the
    # certificate's proof or its producer. Both rows and all nodes evolve.
    source = entry.source
    adjacency = np.zeros((5, 5))
    for i, j in source.edges:
        adjacency[i, j] = adjacency[j, i] = 1
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    mobility = np.array([float(c / d) for c, d in zip(source.capacity, source.degrees)])
    e, w = source.reference_model.effective_weights
    beta = source.reference_model.storage_scale
    scale = w / (beta * np.pi * e)
    eta = (w / e) ** 2 / (beta * np.pi**2)

    def field(_time, state):
        z, theta = state[:5], state[5:]
        q = mobility * (laplacian @ z)
        current = (adjacency * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
        return np.concatenate((-q + eta * mobility * current, q))

    initial = np.array([float(x) * scale for x in source.epi] + [0.0] * 5)
    result = solve_ivp(
        field, (0, 100), initial, method="DOP853", rtol=2e-12, atol=2e-14
    )
    assert result.success
    final = np.concatenate((result.y[:5, -1] / scale, result.y[5:, -1]))
    boxes = entry.endpoint_form_bounds + entry.endpoint_phase_bounds
    for value, box in zip(final, boxes):
        assert float(box.lo) < value < float(box.hi)
    # The represented phase winding really changed; a nonacute passage was
    # allowed during transit, while every final principal edge gap is acute.
    phases = result.y[5:, -1]
    principal = (np.roll(phases, -1) - phases + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(principal)) < np.pi / 2
    assert np.sum(principal) == pytest.approx(2 * np.pi, abs=1e-12)


def test_uniform_source_is_stationary_and_cannot_be_certified_as_nonzero_entry():
    source = replace(_source(), epi=(Q(7),) * 5)
    result = certify_sine_prepared_entry(
        source, scaled_time=100, edge_turn_offsets=OFFSETS
    )
    assert not result.admitted
    assert all(box.lo <= 7 <= box.hi for box in result.endpoint_form_bounds)
    assert all(box.lo <= 0 <= box.hi for box in result.endpoint_phase_bounds)
    assert result.capture.unresolved_conditions


def test_sign_reversal_and_common_origins_preserve_the_actual_symmetries(entry):
    reverse = certify_sine_prepared_entry(
        _source(sign=-1), scaled_time=100, edge_turn_offsets=tuple(-v for v in OFFSETS)
    )
    shifted = certify_sine_prepared_entry(
        _source(form_offset=Q(19, 8), phase_offset=Q(-7, 4)),
        scaled_time=100,
        edge_turn_offsets=OFFSETS,
    )
    assert reverse.admitted and reverse.capture.cycle_periods == (-1,)
    assert reverse.capture.storage_bounds == entry.capture.storage_bounds
    assert shifted.admitted
    assert shifted.weighted_form_mean == Q(19, 8)
    assert shifted.common_initial_phase == Q(-7, 4)
    assert shifted.endpoint_form_edge_gap_bounds == entry.endpoint_form_edge_gap_bounds
    assert (
        shifted.endpoint_phase_edge_gap_bounds == entry.endpoint_phase_edge_gap_bounds
    )


def test_capacity_weighted_mean_and_gap_use_the_held_nonuniform_mobility():
    source = _source(capacities=(1, 2, 3, 4, 5))
    result = certify_sine_prepared_entry(
        source, scaled_time=100, edge_turn_offsets=OFFSETS
    )
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    expected = sum(h * x for h, x in zip(weights, source.epi)) / sum(weights)
    assert result.weighted_form_mean == expected != 0
    assert result.metric_weights == weights
    assert result.mobility == tuple(1 / h for h in weights)
    # Independent symmetric conjugation checks the direction of the exact
    # spectral bound; its value is not inferred from stored source fields.
    matrix = nx.laplacian_matrix(nx.cycle_graph(5)).toarray().astype(float)
    roots = np.sqrt([float(k) for k in result.mobility])
    actual = np.linalg.eigvalsh(roots[:, None] * matrix * roots[None, :])[1]
    assert 0 < float(result.weighted_gap_lower_bound) <= actual + 1e-14


def test_short_horizon_and_wrong_sector_do_not_claim_acquisition(entry):
    early = certify_sine_prepared_entry(
        entry.source, scaled_time=Q(1, 100), edge_turn_offsets=OFFSETS
    )
    wrong = certify_sine_prepared_entry(
        entry.source, scaled_time=100, edge_turn_offsets=(0,) * 5
    )
    assert not early.admitted and not wrong.admitted
    assert early.capture.unresolved_conditions


def test_cached_derived_values_are_not_consumed_as_transit_evidence(entry):
    forged = replace(
        entry.source,
        storage=I(-100),
        form_storage=Q(-1),
        phase_storage=I(-1),
        form_gradient=(Q(0),) * 5,
    )
    result = certify_sine_prepared_entry(
        forged, scaled_time=100, edge_turn_offsets=OFFSETS
    )
    assert result.admitted
    assert result.endpoint_form_bounds == entry.endpoint_form_bounds
    assert result.endpoint_phase_bounds == entry.endpoint_phase_bounds
    assert result.capture.storage_bounds == entry.capture.storage_bounds


@pytest.mark.parametrize("time", [True, -1, float("nan"), float("inf")])
def test_invalid_horizons_reject(time):
    with pytest.raises((TypeError, ValueError)):
        certify_sine_prepared_entry(
            _source(), scaled_time=time, edge_turn_offsets=OFFSETS
        )


def test_zero_time_and_exponential_work_limit_are_explicit(entry):
    zero = certify_sine_prepared_entry(
        entry.source, scaled_time=0, edge_turn_offsets=OFFSETS
    )
    assert zero.horizon == 0 and not zero.admitted
    with pytest.raises(ValueError, match="4096"):
        certify_sine_prepared_entry(
            entry.source,
            scaled_time=Q(4097) / entry.weighted_gap_lower_bound,
            edge_turn_offsets=OFFSETS,
        )


def test_changed_constitutive_ratio_is_recomputed_without_retuning_the_source(entry):
    model = RelationalExchangeModel(1, phase_domain="regular")
    result = certify_sine_prepared_entry(
        replace(entry.source, reference_model=model),
        scaled_time=100,
        edge_turn_offsets=OFFSETS,
    )
    assert result.source.epi == entry.source.epi
    assert result.coefficient_ratio == 1
    assert result.feedback_strength_bounds.lo > entry.feedback_strength_bounds.hi
    assert not result.admitted
    lossless = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    with pytest.raises(ValueError, match="positive"):
        certify_sine_prepared_entry(
            replace(entry.source, reference_model=lossless),
            scaled_time=100,
            edge_turn_offsets=OFFSETS,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": (Q(0), Q(0), Q(1), Q(0), Q(0))},
        {"capacity": (Q(1), Q(0), Q(1), Q(1), Q(1))},
        {"capacity": (True,) * 5},
        {"law": "another_law"},
        {"degrees": (3, 2, 2, 2, 2)},
    ],
)
def test_unsupported_or_forged_source_cannot_receive_transit(changes):
    with pytest.raises((TypeError, ValueError)):
        certify_sine_prepared_entry(
            replace(_source(), **changes), scaled_time=100, edge_turn_offsets=OFFSETS
        )


def test_export_retains_analytic_transit_provenance(entry, tmp_path):
    payload = entry.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-prepared-entry.v1"
    path = tmp_path / "entry.json"
    export_to_json(payload, path)
    assert json_loads(path.read_text(encoding="utf-8")) == payload
    assert "analytic" in entry.capture.uncertainty_scope
    assert entry.capture.input_forecast_requested_end_time is None


@pytest.fixture(scope="module")
def family():
    # One prospective family, with the established law and horizon unchanged.
    return _pattern().certify_prepared_entry(scaled_time=100, edge_turn_offsets=OFFSETS)


def test_entire_preparation_family_has_zero_initial_winding_and_captured_nonzero_end(
    family,
):
    assert family.admitted and family.initial_zero_winding_certified
    assert family.initial_cycle_periods == (0,)
    assert family.capture.cycle_periods == (1,)
    assert all(box.lo > 0 for box in family.initial_acute_margin_bounds)
    assert all(
        box.abs_max == Q(1, 32768) for box in family.initial_phase_edge_gap_bounds
    )
    assert family.capture.storage_upper_bound < Q(3456323, 1000000)
    assert family.capture.energy_margin > Q(12, 1000)
    # Edge-wise maxima give an independent whole-source budget. These are
    # bounds, not a claim that every edge maximum occurs simultaneously.
    assert family.initial_form_storage is None
    assert family.initial_form_storage_bounds.hi == Q(21433437701, 128)
    assert family.initial_phase_storage_bounds.hi > 0
    assert family.initial_storage_bounds.hi >= family.initial_form_storage_bounds.hi
    assert family.weighted_form_mean is family.common_initial_phase is None
    assert (
        family.capture.weighted_form_mean is family.capture.weighted_phase_mean is None
    )
    assert family.endpoint_form_bounds is family.endpoint_phase_bounds is None
    assert len(family.centered_endpoint_form_bounds) == 5
    assert len(family.centered_endpoint_phase_bounds) == 5


def test_nonflat_corner_response_respects_mean_centered_and_correlated_edge_bounds(
    family,
):
    source = family.source
    # Nonuniform independent residuals and arbitrary common origins are
    # retained. This floating response is a cross-check of the enclosure,
    # never the proof that every member succeeds.
    x = np.array([float(v) for v in source.nominal_form]) + 13 / 7
    x += np.array([1, -1, -1, 1, 1]) / 16
    theta = np.array([1, 1, -1, 1, -1]) / 65536 + 5 / 9
    adjacency = nx.to_numpy_array(nx.cycle_graph(5))
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    mobility = np.array([float(value) for value in family.mobility])
    weights = np.array([float(value) for value in family.metric_weights])
    e, w = MODEL.effective_weights
    alpha = w / (e * np.pi)
    eta = alpha**2
    mean_x = weights @ x / weights.sum()
    mean_theta = weights @ theta / weights.sum()

    def field(_time, state):
        z, phase = state[:5], state[5:]
        q = mobility * (laplacian @ z)
        current = (adjacency * np.sin(phase[None, :] - phase[:, None])).sum(axis=1)
        return np.concatenate((-q + eta * mobility * current, q))

    result = solve_ivp(
        field,
        (0, 100),
        np.concatenate((alpha * (x - mean_x), theta)),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert result.success
    final_form = result.y[:5, -1] / alpha
    final_phase = result.y[5:, -1] - mean_theta
    for values, boxes in (
        (final_form, family.centered_endpoint_form_bounds),
        (final_phase, family.centered_endpoint_phase_bounds),
    ):
        for value, box in zip(values, boxes):
            assert float(box.lo) < value < float(box.hi)
    for edge_index, (i, j) in enumerate(family.geometry.edges):
        for values, boxes in (
            (final_form, family.endpoint_form_edge_gap_bounds),
            (final_phase, family.endpoint_phase_edge_gap_bounds),
        ):
            gap = values[j] - values[i]
            assert float(boxes[edge_index].lo) < gap < float(boxes[edge_index].hi)


def test_unknown_origins_and_observation_reference_do_not_change_family_capture(family):
    # These shifts are exactly representable at graph capture; a rounded
    # non-dyadic shift could change the captured internal differences.
    shifted = _pattern(_graph(form_offset=Q(7, 8), phase_offset=Q(-13, 8)), reference=3)
    result = shifted.certify_prepared_entry(scaled_time=100, edge_turn_offsets=OFFSETS)
    assert result.admitted
    assert result.endpoint_form_edge_gap_bounds == family.endpoint_form_edge_gap_bounds
    assert (
        result.endpoint_phase_edge_gap_bounds == family.endpoint_phase_edge_gap_bounds
    )
    assert result.centered_endpoint_form_bounds == family.centered_endpoint_form_bounds
    assert (
        result.centered_endpoint_phase_bounds == family.centered_endpoint_phase_bounds
    )
    assert result.capture.weighted_form_mean is None
    assert result.endpoint_phase_bounds is None


def test_family_uses_original_residuals_not_cached_relative_rectangles(family):
    source = replace(
        family.source,
        relative_form_bounds=(I(-(10**20), 10**20),) * 5,
        relative_phase_bounds=(I(0),) * 5,
        edge_form_gap_bounds=(I(0),) * 5,
        edge_phase_gap_bounds=(I(0),) * 5,
        storage_bounds=I(-1),
    )
    result = source.certify_prepared_entry(scaled_time=100, edge_turn_offsets=OFFSETS)
    assert result.admitted
    assert result.initial_storage_bounds == family.initial_storage_bounds
    assert result.capture.storage_bounds == family.capture.storage_bounds


def test_nonflat_nominal_phase_is_retained_in_the_predicted_edges(family):
    nominal_phase = tuple(Q(value, 4096) for value in (0, 1, -1, 2, -2))
    source = replace(family.source, nominal_phase=nominal_phase)
    report = source.certify_prepared_entry(scaled_time=100, edge_turn_offsets=OFFSETS)
    assert report.admitted and report.initial_zero_winding_certified
    assert report.phase_radius_upper_bound == family.phase_radius_upper_bound
    for edge, before, after in zip(
        family.geometry.edges,
        family.endpoint_phase_edge_gap_bounds,
        report.endpoint_phase_edge_gap_bounds,
    ):
        i, j = edge
        shift = nominal_phase[j] - nominal_phase[i]
        assert after == before + shift


def test_uniform_nominal_uncertain_family_is_captured_in_consensus_not_declared_stationary():
    graph = _graph()
    for node in graph:
        graph.nodes[node]["EPI"] = Q(3)
    pattern = _pattern(graph)
    consensus = pattern.certify_sector_capture(edge_turn_offsets=(0,) * 5)
    attempted = pattern.certify_prepared_entry(
        scaled_time=100, edge_turn_offsets=OFFSETS
    )
    assert consensus.admitted and consensus.cycle_periods == (0,)
    assert not attempted.admitted
    assert attempted.initial_form_storage_bounds.hi > 0
    assert attempted.initial_phase_storage_bounds.hi > 0


def test_unresolved_initial_winding_cannot_receive_acquisition_even_from_flat_nominal(
    family,
):
    broad = replace(family.source, phase_error_bounds=(Q(1),) * 5)
    result = broad.certify_prepared_entry(scaled_time=100, edge_turn_offsets=OFFSETS)
    assert not result.initial_zero_winding_certified
    assert result.initial_cycle_periods is None and not result.admitted
    # Explicitly twisted nominal lifts are not automatically re-unwrapped.
    twisted = replace(family.source, nominal_phase=tuple(Q(5 * i, 4) for i in range(5)))
    report = twisted.certify_prepared_entry(scaled_time=100, edge_turn_offsets=OFFSETS)
    assert not report.initial_zero_winding_certified and not report.admitted


def test_zero_error_relative_source_keeps_unknown_origins(entry):
    source = replace(
        _pattern(), form_error_bounds=(Q(0),) * 5, phase_error_bounds=(Q(0),) * 5
    )
    result = source.certify_prepared_entry(scaled_time=100, edge_turn_offsets=OFFSETS)
    assert (
        result.admitted
        and result.capture.storage_bounds == entry.capture.storage_bounds
    )
    assert result.initial_form_storage == entry.initial_form_storage
    assert result.weighted_form_mean is result.common_initial_phase is None
    assert result.endpoint_form_bounds is result.endpoint_phase_bounds is None


def test_family_centering_keeps_actual_nonuniform_capacity_means():
    source = _pattern(_graph(capacities=(1, 2, 3, 4, 5)))
    result = source.certify_prepared_entry(scaled_time=100, edge_turn_offsets=OFFSETS)
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    # Two opposing vertices of the source box, each with its own conserved
    # mean. Direct full-state arithmetic checks projection and norm bounds.
    with mp.workdps(80):
        alpha = 1 / (1023 * mp.pi)
        for sign in (-1, 1):
            actual = tuple(
                x + sign * (-1) ** i * radius + Q(11, 7)
                for i, (x, radius) in enumerate(
                    zip(source.nominal_form, source.form_error_bounds)
                )
            )
            mean = sum(h * x for h, x in zip(weights, actual)) / sum(weights)
            norm_squared = sum(h * (x - mean) ** 2 for h, x in zip(weights, actual))
            assert alpha * mp.sqrt(_mp(norm_squared)) <= _mp(
                result.scaled_initial_norm_bounds.hi
            )
            for x, box in zip(actual, result.scaled_initial_form_bounds):
                value = alpha * _mp(x - mean)
                assert _mp(box.lo) <= value <= _mp(box.hi)
    assert result.weighted_form_mean is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("form_error_bounds", (True,) * 5),
        ("phase_error_bounds", (Q(-1),) * 5),
        ("form_error_bounds", (Q(0),) * 4),
        ("nominal_phase", (float("nan"),) * 5),
    ],
)
def test_family_primitive_admission_precedes_interval_arithmetic(field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(_pattern(), **{field: value}).certify_prepared_entry(
            scaled_time=100, edge_turn_offsets=OFFSETS
        )


def test_family_export_keeps_unavailable_origins_and_uncertainty(family, tmp_path):
    payload = family.to_dict()
    path = tmp_path / "family-entry.json"
    export_to_json(payload, path)
    assert json_loads(path.read_text(encoding="utf-8")) == payload
    assert payload["report"]["endpoint_form_bounds"] is None
    assert payload["report"]["weighted_form_mean"] is None
