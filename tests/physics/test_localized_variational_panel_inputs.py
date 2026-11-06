"""Synthetic panel-input wiring; no canonical 97-sample evaluation."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._interval_chebyshev import interpolate_lobatto_rows, lobatto_nodes
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _localized_variational_panel_inputs as panels


@pytest.fixture(scope="module", autouse=True)
def no_prepared_response():
    with forbid_organized_response_evaluation():
        yield


def _manufactured_evaluator(start, duration, calls):
    def evaluate(time):
        calls.append(time)
        s = 2 * (time - start) / duration - 1
        return SimpleNamespace(
            source_x=(s, s**2, 1 - s),
            source_theta=(-s, 2 * s, s**3),
            cosine_coefficient=(I(Q(1, 8)), Q(1, 4) + s / 32, I(0)),
        )

    return evaluate


def _contains_coefficients(observed, expected):
    assert len(observed) >= len(expected)
    for index, value in enumerate(observed):
        assert value.contains(expected[index] if index < len(expected) else Q(0))


def test_sampling_and_transform_preserve_signed_sources_and_affine_clock():
    calls = []
    start, duration = Q(3), Q(8)
    result = panels._sample_panel_inputs(
        _manufactured_evaluator(start, duration, calls),
        start,
        duration,
        lobatto_nodes(4),
        3,
    )
    form, phase, edge, times, elapsed, source_radius, edge_radius = result
    assert tuple(calls) == times
    assert times[0] == I(11) and times[-1] == I(3) and times[2] == I(7)
    assert len(elapsed) == 5 and all(value >= 0 for value in elapsed)
    transformed = interpolate_lobatto_rows(form + phase + edge, degree=4)
    expected = (
        (Q(0), Q(1)),
        (Q(1, 2), Q(0), Q(1, 2)),
        (Q(1), Q(-1)),
        (Q(0), Q(-1)),
        (Q(0), Q(2)),
        (Q(0), Q(3, 4), Q(0), Q(1, 4)),
        (Q(1, 8),),
        (Q(1, 4), Q(1, 32)),
        (Q(0),),
    )
    for coefficients, truth in zip(transformed, expected):
        _contains_coefficients(coefficients, truth)
    assert source_radius >= 0 and edge_radius >= 0


def test_affine_nodes_enclose_exact_nondyadic_clock_without_endpoint_spill():
    calls = []
    start, duration = Q(1, 3), Q(2, 3)
    result = panels._sample_panel_inputs(
        _manufactured_evaluator(start, duration, calls),
        start,
        duration,
        lobatto_nodes(2),
        3,
    )
    times = result[3]
    assert times[0] == I(1)
    assert times[1] == I(Q(2, 3))
    assert times[-1] == I(Q(1, 3))
    assert all(time.lo >= I(start).lo and time.hi <= 1 for time in times)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": True},
        {"start": -1},
        {"start": float("nan")},
        {"start": I(0)},
        {"duration": False},
        {"duration": 0},
        {"duration": 17},
        {"duration": float("inf")},
        {"duration": I(1)},
    ],
)
def test_invalid_clock_rejects_before_factory_or_transform(monkeypatch, kwargs):
    def forbidden(*args, **parameters):
        pytest.fail("invalid primitive clock reached preparation or transformation")

    monkeypatch.setattr(panels, "_prepare_localized_variational_inputs", forbidden)
    monkeypatch.setattr(panels, "interpolate_lobatto_rows", forbidden)
    with pytest.raises((ValueError, TypeError)):
        panels._build_localized_variational_panel_inputs(**kwargs)


def test_panel_exceeding_fresh_factory_horizon_rejects_before_sampling(monkeypatch):
    def forbidden(time):
        pytest.fail("unadmitted panel reached source evaluation")

    monkeypatch.setattr(
        panels,
        "_prepare_localized_variational_inputs",
        lambda: (SimpleNamespace(evaluation_time_bounds=I(0, 20)), forbidden),
    )
    with pytest.raises(ValueError, match="T.hi"):
        panels._build_localized_variational_panel_inputs(start=5, duration=16)


def test_sample_rows_readmit_primitives_and_detach_mutable_storage():
    supplied = [Q(1), Q(2), Q(3)]

    def evaluate(time):
        return SimpleNamespace(
            source_x=supplied,
            source_theta=[Q(0)] * 3,
            cosine_coefficient=[Q(0)] * 3,
        )

    form, *_ = panels._sample_panel_inputs(evaluate, Q(0), Q(1), (I(1), I(-1)), 3)
    supplied[0] = True
    assert form[0] == (I(1), I(1))
    with pytest.raises((ValueError, TypeError)):
        panels._sample_panel_inputs(evaluate, Q(0), Q(1), (I(1), I(-1)), 3)


def test_source_and_edge_arithmetic_radius_statistics_keep_tiny_widths():
    tiny = Q(1, 2**100)
    zero = (I(0),) * 97
    uncertain = (I(-tiny, tiny),) + zero[1:]
    source, edge = panels._coefficient_radius_bounds(
        (uncertain,), (zero,), (uncertain,)
    )
    assert source == tiny
    assert edge == 2 * tiny


@pytest.fixture(scope="module")
def manufactured_builder_result():
    # The replaced factory supplies manufactured three-node inputs only.
    # This wiring control establishes no evidence about the canonical source.
    with pytest.MonkeyPatch.context() as patch:
        preparations, calls = [], []

        def factory():
            preparations.append(True)
            return (
                SimpleNamespace(
                    node_count=3,
                    node_order=(0, 1, 2),
                    edges=((0, 1), (1, 2), (2, 0)),
                    sigma_bounds=I(Q(1, 64)),
                    horizon_bounds=I(20),
                    evaluation_time_bounds=I(0, 20),
                ),
                _manufactured_evaluator(Q(0), Q(16), calls),
            )

        patch.setattr(panels, "_prepare_localized_variational_inputs", factory)
        result = panels._build_localized_variational_panel_inputs()
        return result, tuple(preparations), tuple(calls)


def test_fixed_builder_batches_all_channels_and_retains_declared_tails(
    manufactured_builder_result,
):
    result, preparations, calls = manufactured_builder_result
    assert len(preparations) == 1
    assert len(calls) == 97
    assert calls == result.sample_time_bounds
    assert calls[0] == I(16) and calls[-1] == I(0) and calls[48] == I(8)
    assert result.start == 0 and result.duration == result.end == 16
    assert result.node_order == (0, 1, 2)
    assert result.edges == ((0, 1), (1, 2), (2, 0))
    assert result.degree == 96
    assert result.generator_error == Q(1, 2**114)
    assert result.source_error == Q(1, 2**109)
    _contains_coefficients(result.source_form_coefficients[0], (Q(0), Q(1)))
    _contains_coefficients(result.source_form_coefficients[1], (Q(1, 2), Q(0), Q(1, 2)))
    _contains_coefficients(
        result.source_phase_coefficients[2], (Q(0), Q(3, 4), Q(0), Q(1, 4))
    )
    _contains_coefficients(result.edge_coefficient_polynomials[1], (Q(1, 4), Q(1, 32)))
    assert result.source_coefficient_radius_norm_sum >= 0
    assert result.edge_coefficient_radius_operator_upper >= 0
    assert all(
        value >= 0
        for value in (
            result.factory_seconds,
            result.sampling_seconds,
            result.transform_seconds,
        )
    )
    assert not hasattr(result, "certified")
    with pytest.raises(FrozenInstanceError):
        result.source_error = Q(0)


def test_public_builder_has_no_evaluator_or_metadata_admission():
    with pytest.raises(TypeError):
        panels._build_localized_variational_panel_inputs(metadata=object())
    with pytest.raises(TypeError):
        panels._build_localized_variational_panel_inputs(evaluate=lambda time: time)
