"""Warning policies cannot turn invalid inputs or missing fields into a pass."""

import builtins
import math
from fractions import Fraction as Q
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tnfr.operators.grammar import (
    GrammarValidator,
    warn_coherence_length_telemetry,
    warn_phase_curvature_telemetry,
    warn_phase_gradient_telemetry,
)
from tnfr.physics import canonical, fields


def _pair():
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, {0: 0.0, 1: 1.0}, "theta")
    return graph


INVALID_POSITIVE = (
    True,
    np.bool_(True),
    math.nan,
    math.inf,
    -1.0,
    0.0,
    Q(1, 10**400),
    10**400,
    "1",
)


@pytest.mark.parametrize("value", INVALID_POSITIVE)
@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("kind", ("gradient", "curvature", "length"))
def test_warning_policy_admission_precedes_graph_observations(value, empty, kind):
    graph = nx.Graph() if empty else _pair()
    with pytest.raises((TypeError, ValueError)):
        if kind == "gradient":
            warn_phase_gradient_telemetry(graph, threshold=value)
        elif kind == "curvature":
            warn_phase_curvature_telemetry(
                graph, abs_threshold=value, multiscale_check=False
            )
        else:
            warn_coherence_length_telemetry(graph, regime_multipliers=(value, value))


@pytest.mark.parametrize("threshold", (Q(1, 2), Q(1), Q(2)))
def test_phase_warning_flags_and_verdict_share_the_admitted_threshold(threshold):
    graph = _pair()
    for reader, kwargs in (
        (warn_phase_gradient_telemetry, {"threshold": threshold}),
        (
            warn_phase_curvature_telemetry,
            {"abs_threshold": threshold, "multiscale_check": False},
        ),
    ):
        safe, stats, message, flagged = reader(graph, **kwargs)
        assert stats["available"]
        assert safe is (threshold > 1)
        assert flagged == ([] if threshold > 1 else [0, 1])
        assert ("PASS" in message) is safe
        observed_max = (
            stats["max"]
            if reader is warn_phase_gradient_telemetry
            else stats["max_abs"]
        )
        assert observed_max == pytest.approx(1.0)


@pytest.mark.parametrize(
    "reader,value_key",
    (
        (warn_phase_gradient_telemetry, "max"),
        (warn_phase_curvature_telemetry, "max_abs"),
    ),
)
def test_empty_phase_observation_is_unavailable(reader, value_key):
    safe, stats, message, flagged = reader(nx.Graph())
    assert not safe
    assert not stats["available"]
    assert math.isnan(stats[value_key])
    assert not flagged
    assert "unavailable" in message
    assert "PASS" not in message


@pytest.mark.parametrize(
    "reader",
    (warn_phase_gradient_telemetry, warn_phase_curvature_telemetry),
)
def test_missing_optional_field_import_never_passes(monkeypatch, reader):
    original = builtins.__import__

    def missing_fields(name, *args, **kwargs):
        if name == "physics.fields":
            raise ImportError("field owner unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_fields)
    safe, stats, message, _ = reader(_pair())
    assert not safe
    assert not stats["available"]
    assert "unavailable" in message


def test_non_import_field_defect_is_not_disguised_as_missing_dependency(monkeypatch):
    original = builtins.__import__

    def broken_fields(name, *args, **kwargs):
        if name == "physics.fields":
            raise RuntimeError("field initialization defect")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_fields)
    with pytest.raises(RuntimeError, match="initialization defect"):
        warn_phase_gradient_telemetry(_pair())


@pytest.mark.parametrize(
    "value", (True, math.nan, math.inf, Q(1, 10**400), Q(1) + Q(1, 2**100))
)
def test_consumed_multiscale_fit_policy_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError)):
        warn_phase_curvature_telemetry(nx.Graph(), fit_min_r2=value)


def test_disabled_multiscale_check_does_not_consume_fit_policy(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled multiscale owner must not be invoked")

    monkeypatch.setattr(fields, "k_phi_multiscale_safety", unexpected)
    safe, stats, _, _ = warn_phase_curvature_telemetry(
        _pair(), multiscale_check="false", alpha_hint=math.nan, fit_min_r2=math.nan
    )
    assert safe
    assert stats["available"]
    assert not stats["multiscale_checked"]


def test_unavailable_multiscale_evidence_overrides_a_cached_safe_flag(monkeypatch):
    monkeypatch.setattr(
        fields,
        "k_phi_multiscale_safety",
        lambda *args, **kwargs: {"available": False, "safe": True, "fit": {}},
    )
    safe, stats, message, _ = warn_phase_curvature_telemetry(_pair())
    assert not safe
    assert not stats["available"]
    assert not stats["multiscale_safe"]
    assert "unavailable" in message


def test_length_multiplier_order_is_checked_before_rounding():
    with pytest.raises(ValueError, match="base regime multiplier"):
        warn_coherence_length_telemetry(
            nx.Graph(), regime_multipliers=(Q(1) + Q(1, 2**100), Q(1))
        )


@pytest.mark.parametrize(
    "length,multipliers",
    ((2.0, (1.0, 1e308)), (5e-324, (0.5, 1.0))),
)
def test_path_policy_arithmetic_must_remain_representable(
    monkeypatch, length, multipliers
):
    graph = _pair()
    nx.set_edge_attributes(graph, length, "length")
    monkeypatch.setattr(
        canonical,
        "estimate_coherence_length_with_provenance",
        lambda graph: SimpleNamespace(
            value=1.0,
            method="autocorrelation_fit",
            distance_weighting="declared path geometry",
        ),
    )
    with pytest.raises(ValueError, match="path threshold"):
        warn_coherence_length_telemetry(graph, regime_multipliers=multipliers)


@pytest.mark.parametrize(
    "multipliers,severity,expected_threshold,expected_factor",
    (
        ((0.25, 3.0), "watch", "watch_threshold=2.5", "(0.25×mean_path_length)"),
        ((0.75, 3.0), "stable", "watch_threshold=7.5", "(0.75×mean_path_length)"),
        ((0.1, 0.25), "alert", "alert_threshold=2.5", "(0.25×mean_path_length)"),
    ),
)
def test_length_message_reports_the_selected_threshold(
    monkeypatch, multipliers, severity, expected_threshold, expected_factor
):
    graph = _pair()
    nx.set_edge_attributes(graph, 10.0, "length")
    monkeypatch.setattr(
        canonical,
        "estimate_coherence_length_with_provenance",
        lambda graph: SimpleNamespace(
            value=5.0,
            method="autocorrelation_fit",
            distance_weighting="declared path geometry",
        ),
    )

    safe, stats, message = warn_coherence_length_telemetry(
        graph, regime_multipliers=multipliers
    )

    assert stats["mean_path_length"] == 10.0
    assert stats["severity"] == severity
    assert safe is (severity == "stable")
    assert expected_threshold in message
    assert expected_factor in message


def test_nonblocking_aggregator_reports_invalid_policy_without_a_phase_pass():
    messages = GrammarValidator().telemetry_warnings(
        _pair(), phi_grad_threshold=math.inf, kphi_abs_threshold=math.nan
    )
    assert len(messages) == 3
    assert all("telemetry error" in message for message in messages[:2])
    assert all("PASS" not in message for message in messages[:2])
