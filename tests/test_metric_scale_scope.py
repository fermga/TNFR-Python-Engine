"""Current-state normalization and the declared numerical scale of metrics."""

from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_SI,
    ALIAS_VF,
)
from tnfr.metrics import coherence, sense_index
from tnfr.metrics.common import (
    _get_vf_dnfr_max,
    compute_coherence,
    compute_dnfr_accel_max,
    finite_mean,
    finite_mean_absolute,
    finite_pearson_correlation,
    finite_population_std,
    is_structural_equilibrium,
    normalize_dnfr,
    structural_coherence,
)
from tnfr.metrics.core import _metrics_step
from tnfr.utils.numeric import _clipped_ratio, _norm01, similarity_abs
from tnfr.validation.rules import _norm_attr, _si, get_norm, normalized_dnfr
from tnfr.validation.soft_filters import acceleration_norm


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("channel", ["capacity", "pressure"])
def test_si_normalization_uses_live_aliases_despite_stale_maxima(
    monkeypatch: pytest.MonkeyPatch, use_numpy: bool, channel: str
) -> None:
    if not use_numpy:
        monkeypatch.setattr(sense_index, "np", None)
    graph = nx.path_graph(2)
    for node, magnitude in enumerate((0.5, 1.0)):
        graph.nodes[node].update(
            {ALIAS_VF[0]: magnitude, ALIAS_DNFR[0]: magnitude, "phase": 0.0}
        )
    graph.graph.update(_vfmax=2.0, _dnfrmax=2.0)
    graph.graph["SI_WEIGHTS"] = {
        "alpha": float(channel == "capacity"),
        "beta": 0.0,
        "gamma": float(channel == "pressure"),
    }
    expected = {0: 0.5, 1: 1.0} if channel == "capacity" else {0: 0.5, 1: 0.0}

    assert sense_index.compute_Si(graph, inplace=False) == pytest.approx(expected)
    assert graph.graph["_vfmax"] == graph.graph["_dnfrmax"] == 1.0

    # A direct write bypasses the setter's incremental max-cache maintenance.
    graph.nodes[1][ALIAS_VF[0]] = 2.0
    graph.nodes[1][ALIAS_DNFR[0]] = 2.0
    expected[0] = 0.25 if channel == "capacity" else 0.75
    assert sense_index.compute_Si(graph, inplace=False) == pytest.approx(expected)
    assert graph.graph["_vfmax"] == graph.graph["_dnfrmax"] == 2.0


@pytest.mark.parametrize("node_count", [0, 2])
def test_zero_maxima_are_cached_as_zero_but_have_safe_divisors(
    node_count: int,
) -> None:
    graph = nx.empty_graph(node_count)
    graph.graph.update(_vfmax=3.0, _dnfrmax=4.0)

    assert _get_vf_dnfr_max(graph) == (1.0, 1.0)
    assert graph.graph["_vfmax"] == graph.graph["_dnfrmax"] == 0.0


@pytest.mark.parametrize("use_numpy", [True, False])
def test_uniform_capacity_unit_change_preserves_relative_si(
    monkeypatch: pytest.MonkeyPatch, use_numpy: bool
) -> None:
    if not use_numpy:
        monkeypatch.setattr(sense_index, "np", None)
    graph = nx.path_graph(2)
    for node, capacity in enumerate((0.5, 1.0)):
        graph.nodes[node].update(
            {ALIAS_VF[0]: capacity, ALIAS_DNFR[0]: 0.25, "phase": 0.0}
        )
    before = sense_index.compute_Si(graph, inplace=False)

    # This holds pressure fixed. It does not certify covariance of a pressure
    # law that itself contains an untransformed capacity-gradient coefficient.
    for _, data in graph.nodes(data=True):
        data[ALIAS_VF[0]] /= 4.0

    assert sense_index.compute_Si(graph, inplace=False) == pytest.approx(before)


def test_coherence_scale_depends_on_time_coordinate_and_stored_rate() -> None:
    graph = nx.empty_graph(1)
    graph.nodes[0].update({ALIAS_VF[0]: 2.0, ALIAS_DNFR[0]: 0.5, ALIAS_DEPI[0]: 1.0})
    assert compute_coherence(graph) == pytest.approx(2.0 / 5.0)

    # The same nodal curve in t_new=2*t has nu_new=nu/2 and rate_new=rate/2.
    # The chosen reciprocal diagnostic is not invariant to that unit change.
    graph.nodes[0][ALIAS_VF[0]] = 1.0
    graph.nodes[0][ALIAS_DEPI[0]] = 0.5
    assert compute_coherence(graph) == pytest.approx(1.0 / 2.0)

    # Reading coherence does not silently replace a recorded rate with nu*p.
    graph.nodes[0][ALIAS_VF[0]] = 0.0
    assert compute_coherence(graph) == pytest.approx(1.0 / 2.0)
    assert not is_structural_equilibrium(0.5, 0.0)


def test_network_coherence_differs_from_nodal_mean_and_projected_parent() -> None:
    graph = nx.empty_graph(2)
    for node, pressure in enumerate((1.0, -3.0)):
        graph.nodes[node].update({ALIAS_DNFR[0]: pressure, ALIAS_DEPI[0]: 0.0})

    network = compute_coherence(graph)
    nodal_mean = (structural_coherence(1.0) + structural_coherence(-3.0)) / 2.0
    projected_parent = structural_coherence((1.0 - 3.0) / 2.0)

    assert network == pytest.approx(1.0 / 3.0)
    assert nodal_mean == pytest.approx(3.0 / 8.0)
    assert projected_parent == pytest.approx(1.0 / 2.0)
    assert network < nodal_mean < projected_parent


@pytest.mark.parametrize("aliases", [ALIAS_DNFR, ALIAS_DEPI])
@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        "1",
        None,
        complex(1.0, 0.0),
        np.complex64(1.0),
        np.complex128(1.0),
    ],
)
def test_primary_coherence_rejects_invalid_authoritative_alias(
    aliases: tuple[str, ...], value: object
) -> None:
    graph = nx.empty_graph(1)
    graph.nodes[0].update({aliases[0]: value, aliases[1]: 0.0})

    # Neither a valid later alias nor the zero default may hide bad telemetry.
    with pytest.raises((TypeError, ValueError)):
        compute_coherence(graph)


def test_primary_coherence_preserves_missing_and_legacy_alias_conventions() -> None:
    graph = nx.empty_graph(1)
    assert compute_coherence(graph, return_means=True) == (1.0, 0.0, 0.0)

    graph.nodes[0][ALIAS_DNFR[1]] = -1.0
    assert compute_coherence(graph, return_means=True) == (0.5, 1.0, 0.0)
    graph.nodes[0][ALIAS_DEPI[1]] = 1.0
    assert compute_coherence(graph) == pytest.approx(1.0 / 3.0)
    assert compute_coherence(nx.Graph(), return_means=True) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("channel", ["dnfr", "depi", "eps_dnfr", "eps_depi"])
def test_equilibrium_does_not_erase_nonzero_source_at_materialization(channel):
    arguments = dict(dnfr=0.0, depi=0.0, eps_dnfr=0.0, eps_depi=0.0)
    arguments[channel] = Fraction(1, 10**400)
    with pytest.raises(ValueError, match="underflow"):
        is_structural_equilibrium(**arguments)


def test_represented_subnormal_pressure_is_not_an_exact_equilibrium():
    smallest = math.ulp(0.0)
    assert not is_structural_equilibrium(smallest, eps_dnfr=0.0)
    assert is_structural_equilibrium(smallest, eps_dnfr=smallest)
    # Binary64 rounding can still make C exactly one. That read-out must not
    # replace the explicit pressure/rate tolerance test.
    assert structural_coherence(smallest) == 1.0
    assert finite_mean_absolute((smallest, smallest), name="pressure") == smallest
    assert finite_mean((smallest, smallest)) == smallest
    assert finite_population_std((-smallest, smallest)) == smallest


def test_finite_pearson_preserves_observation_pairing():
    # Centered sums are cross=8, left_square=10, right_square=10.
    assert finite_pearson_correlation(
        iter([-2.0, -1.0, 0.0, 1.0, 2.0]), iter([2.0, 1.0, 4.0, 3.0, 5.0])
    ) == pytest.approx(0.8)


def test_finite_pearson_handles_adjacent_and_opposite_maximum_coordinates():
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    previous = math.nextafter(largest, 0.0)
    earlier = math.nextafter(previous, 0.0)
    assert finite_pearson_correlation([earlier, previous, largest], [-1, 0, 1]) == (
        pytest.approx(1.0)
    )
    assert finite_pearson_correlation([-largest, 0.0, largest], [1, 0, -1]) == (
        pytest.approx(-1.0)
    )


@pytest.mark.parametrize("slope", [-1.0, 1.0])
def test_finite_pearson_retains_subnormal_affine_variation(slope):
    smallest = math.ulp(0.0)
    left = [-smallest, 0.0, smallest]
    right = [2 * smallest + slope * 2 * value for value in left]
    assert finite_pearson_correlation(left, right) == pytest.approx(slope)


def test_finite_pearson_exposes_insufficient_or_constant_samples():
    assert finite_pearson_correlation([], []) is None
    assert finite_pearson_correlation([1.0], [2.0]) is None
    assert finite_pearson_correlation([1.0, 1.0], [0.0, 2.0]) is None
    assert finite_pearson_correlation([0.0, 2.0], [1.0, 1.0]) is None
    with pytest.raises(ValueError, match="equal sample lengths"):
        finite_pearson_correlation([], [0.0])


@pytest.mark.parametrize("invalid", [True, "1", 1 + 0j, math.nan, math.inf])
def test_finite_pearson_validates_raw_samples_even_when_unavailable(invalid):
    with pytest.raises((TypeError, ValueError)):
        finite_pearson_correlation([invalid], [0.0])
    with pytest.raises((TypeError, ValueError)):
        finite_pearson_correlation([0.0, 0.0], [0.0, invalid])


@pytest.mark.parametrize(
    "pressure",
    [
        [True, 2.0],
        np.array([False, True]),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
        np.array([1], dtype="timedelta64[s]"),
    ],
)
def test_vectorized_coherence_does_not_promote_nonreal_coordinates(pressure):
    with pytest.raises(TypeError, match="real"):
        structural_coherence(pressure)


def test_vectorized_coherence_preserves_scalar_real_domain_and_broadcasting():
    result = structural_coherence([[Fraction(1, 2)], [Fraction(3, 2)]], [0.0, 0.5])
    np.testing.assert_allclose(result, [[2 / 3, 1 / 2], [2 / 5, 1 / 3]])
    with pytest.raises(ValueError, match="underflow"):
        structural_coherence([Fraction(1, 10**400)])


@pytest.mark.parametrize("invalid", [None, math.nan, True, "2"])
def test_pressure_normalization_and_maxima_reject_invalid_authoritative_alias(
    invalid,
):
    graph = nx.empty_graph(1)
    data = graph.nodes[0]
    data.update({ALIAS_DNFR[0]: invalid, ALIAS_DNFR[1]: 2.0})
    graph.graph.update(_vfmax=7.0, _dnfrmax=8.0)
    with pytest.raises((TypeError, ValueError)):
        normalize_dnfr(data, 4.0)
    with pytest.raises((TypeError, ValueError)):
        normalize_dnfr(data, 0.0)
    with pytest.raises((TypeError, ValueError)):
        compute_dnfr_accel_max(graph)
    with pytest.raises((TypeError, ValueError)):
        _get_vf_dnfr_max(graph)
    assert graph.graph == {"_vfmax": 7.0, "_dnfrmax": 8.0}


@pytest.mark.parametrize("invalid", [math.nan, math.inf, True, "2", -1.0])
def test_pressure_normalizer_rejects_invalid_supplied_maximum(invalid):
    with pytest.raises((TypeError, ValueError)):
        normalize_dnfr({ALIAS_DNFR[0]: 2.0}, invalid)


def test_stored_maxima_and_normalization_keep_signed_magnitudes_and_missing_zero():
    graph = nx.empty_graph(2)
    graph.nodes[0].update({ALIAS_DNFR[1]: -3.0, ALIAS_D2EPI[1]: -5.0})
    graph.nodes[1][ALIAS_DNFR[0]] = 2.0
    assert compute_dnfr_accel_max(graph) == {"dnfr_max": 3.0, "accel_max": 5.0}
    assert normalize_dnfr(graph.nodes[0], 6.0) == 0.5
    assert normalize_dnfr(graph.nodes[0], 1.0) == 1.0
    assert normalize_dnfr({}, 1.0) == normalize_dnfr({}, 0.0) == 0.0
    graph.nodes[0][ALIAS_D2EPI[0]] = None
    with pytest.raises((TypeError, ValueError)):
        compute_dnfr_accel_max(graph)


@pytest.mark.parametrize("magnitude", [1e308, -1e308])
def test_bounded_normalizers_saturate_before_finite_ratio_overflow(magnitude):
    context = SimpleNamespace(norms={"dnfr_max": 1e-308, "accel_max": 1e-308})
    data = {ALIAS_DNFR[0]: magnitude, ALIAS_D2EPI[0]: magnitude}
    assert normalize_dnfr(data, 1e-308) == 1.0
    assert normalized_dnfr(context, data) == 1.0
    assert acceleration_norm(context, data) == 1.0
    assert _norm_attr(context, data, ALIAS_DNFR, "dnfr_max") == 1.0


def test_bounded_normalizers_keep_distinct_zero_and_sign_policies():
    data = {ALIAS_DNFR[0]: -0.5, ALIAS_D2EPI[0]: -0.5}
    context = SimpleNamespace(norms={"dnfr_max": 0.0, "accel_max": -1.0})
    assert normalize_dnfr(data, 0.0) == 0.0
    assert get_norm(context, "dnfr_max") == 1.0
    assert normalized_dnfr(context, data) == 0.5
    assert acceleration_norm(context, data) == 0.0
    assert _clipped_ratio(-0.5, -1.0) == 0.5
    with pytest.raises(ZeroDivisionError):
        _clipped_ratio(1.0, 0.0)


@pytest.mark.parametrize("helper", ["attribute", "acceleration", "sense"])
@pytest.mark.parametrize(
    "bad",
    [True, np.bool_(True), "0.25", math.nan, math.inf, Fraction(1, 10**400), object()],
)
def test_validation_scores_reject_raw_authoritative_alias_after_valid_read(helper, bad):
    context = SimpleNamespace(norms={"dnfr_max": 1.0, "accel_max": 1.0})
    aliases, read = {
        "attribute": (
            ALIAS_DNFR,
            lambda data: _norm_attr(context, data, ALIAS_DNFR, "dnfr_max"),
        ),
        "acceleration": (ALIAS_D2EPI, lambda data: acceleration_norm(context, data)),
        "sense": (ALIAS_SI, _si),
    }[helper]
    data = {aliases[0]: 0.25, aliases[1]: 0.75}
    assert read(data) == 0.25
    data[aliases[0]] = bad
    with pytest.raises((TypeError, ValueError)):
        read(data)
    del data[aliases[0]]
    assert read(data) == 0.75


def test_validation_scores_preserve_only_absent_channel_defaults():
    context = SimpleNamespace(norms={})
    assert _norm_attr(context, {}, ALIAS_DNFR, "dnfr_max") == 0.0
    assert acceleration_norm(context, {}) == 0.0
    assert _si({}) == 0.5


@pytest.mark.parametrize("bad", [True, "1", math.nan, math.inf, Fraction(1, 10**400)])
def test_clipped_ratio_admits_both_operands_before_saturation(bad):
    with pytest.raises((TypeError, ValueError)):
        _clipped_ratio(bad, 1e-308, absolute=True)
    with pytest.raises((TypeError, ValueError)):
        _clipped_ratio(1e308, bad, absolute=True)
    with pytest.raises((TypeError, ValueError)):
        get_norm(SimpleNamespace(norms={"accel_max": bad}), "accel_max")


def test_clipped_ratio_preserves_subnormals_and_rejects_interior_nonzero_loss():
    tiny = math.ulp(0.0)
    assert normalize_dnfr({ALIAS_DNFR[0]: -tiny}, 1.0) == tiny
    assert _clipped_ratio(tiny, tiny) == 1.0
    with pytest.raises(ValueError, match="underflow"):
        normalize_dnfr({ALIAS_DNFR[0]: tiny}, 2.0)


def test_similarity_and_interval_normalization_avoid_overflowing_differences():
    assert similarity_abs(1e308, -1e308, 0.0, 1.0) == 0.0
    assert similarity_abs(1e308, 0.0, -1e308, 1e308) == 0.5
    assert similarity_abs(0.25, 0.75, 0.0, 2.0) == 0.75
    assert similarity_abs(1.0, 3.0, 2.0, 2.0) == 1.0
    assert _norm01(0.0, -1e308, 1e308) == 0.5
    assert _norm01(1e308, -1e308, 1e308) == 1.0
    assert _norm01(-1e308, -1e308, 1e308) == 0.0
    with pytest.raises(ValueError, match="underflow"):
        _norm01(math.ulp(0.0), 0.0, 2.0)


@pytest.mark.parametrize("bad", [True, "1", math.inf, math.nan])
@pytest.mark.parametrize("position", range(4))
def test_similarity_validates_inputs_even_with_degenerate_range(bad, position):
    arguments = [0.0, 1.0, 2.0, 2.0]
    arguments[position] = bad
    with pytest.raises((TypeError, ValueError)):
        similarity_abs(*arguments)


def test_single_and_worker_si_preserve_finite_saturation():
    data = {ALIAS_VF[0]: 1e308, ALIAS_DNFR[0]: -1e308}
    options = dict(alpha=0.25, beta=0.5, gamma=0.25, vfmax=1e-308, dnfrmax=1e-308)
    assert (
        sense_index.compute_Si_node(
            "n", data, phase_dispersion=0.0, inplace=False, **options
        )
        == 0.75
    )
    assert sense_index._compute_si_python_chunk(
        [("n", (), 0.0, 1e308, -1e308)], cos_th={}, sin_th={}, **options
    ) == {"n": 0.75}


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("aliases", [ALIAS_VF, ALIAS_DNFR])
def test_si_normalization_rejects_nonzero_division_loss(
    monkeypatch, use_numpy, aliases
):
    if not use_numpy:
        monkeypatch.setattr(sense_index, "np", None)
    graph = nx.path_graph(2)
    for node, value in enumerate((math.ulp(0.0), 2.0)):
        graph.nodes[node].update(nu_f=1.0, delta_NFR=0.0, phase=0.0)
        graph.nodes[node][aliases[0]] = value
    with pytest.raises(ValueError, match="underflow"):
        sense_index.compute_Si(graph, inplace=False)


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("aliases", [ALIAS_VF, ALIAS_DNFR])
def test_si_revalidates_live_authoritative_scalar_after_cache_warmup(
    monkeypatch, use_numpy, aliases
):
    if not use_numpy:
        monkeypatch.setattr(sense_index, "np", None)
    graph = nx.path_graph(2)
    for _, data in graph.nodes(data=True):
        data.update(nu_f=1.0, delta_NFR=0.0, phase=0.0)
    sense_index.compute_Si(graph)
    graph.nodes[1].update({aliases[0]: None, aliases[1]: 1.0})
    original_senses = {node: data["Si"] for node, data in graph.nodes(data=True)}
    with pytest.raises((TypeError, ValueError)):
        sense_index.compute_Si(graph)
    assert {
        node: data["Si"] for node, data in graph.nodes(data=True)
    } == original_senses


@pytest.mark.parametrize("use_numpy", [True, False])
def test_si_rejects_negative_capacity_without_rejecting_signed_pressure(
    monkeypatch, use_numpy
):
    if not use_numpy:
        monkeypatch.setattr(sense_index, "np", None)
    graph = nx.empty_graph(1)
    data = graph.nodes[0]
    data.update({ALIAS_VF[0]: 1.0, ALIAS_DNFR[0]: -2.0, "phase": 0.0})
    sense_index.compute_Si(graph)
    previous = data["Si"]
    data[ALIAS_VF[0]] = -1.0
    with pytest.raises(ValueError, match="capacity must be nonnegative"):
        sense_index.compute_Si(graph)
    with pytest.raises(ValueError, match="capacity must be nonnegative"):
        _get_vf_dnfr_max(graph)
    with pytest.raises(ValueError, match="capacity must be nonnegative"):
        sense_index.compute_Si_node(
            0,
            data,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            vfmax=1.0,
            dnfrmax=2.0,
            phase_dispersion=0.0,
            inplace=True,
        )
    assert data["Si"] == previous


@pytest.mark.parametrize("node_count", [0, 1])
@pytest.mark.parametrize(
    "key,invalid",
    [
        ("EPS_DNFR_STABLE", True),
        ("EPS_DEPI_STABLE", "1"),
        ("EPS_DNFR_STABLE", -Fraction(1, 10**400)),
        ("EPS_DEPI_STABLE", math.nan),
    ],
)
def test_stability_configuration_is_admitted_before_history_writes(
    node_count, key, invalid
):
    graph = nx.empty_graph(node_count)
    graph.graph.update(METRICS={"enabled": True, "verbosity": "basic"})
    graph.graph[key] = invalid
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises((TypeError, ValueError)):
        _metrics_step(graph)
    assert "history" not in graph.graph
    assert dict(graph.nodes(data=True)) == before

    # Direct tracker calls share the same admission, including empty support.
    history = {}
    cuts = {"eps_dnfr": 1e-3, "eps_depi": 1e-3}
    cuts["eps_dnfr" if key == "EPS_DNFR_STABLE" else "eps_depi"] = invalid
    with pytest.raises((TypeError, ValueError)):
        coherence._track_stability(graph, history, 1.0, **cuts)
    assert history == {}
    assert dict(graph.nodes(data=True)) == before
