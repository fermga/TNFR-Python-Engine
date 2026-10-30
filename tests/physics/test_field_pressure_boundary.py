"""Strict pressure admission and detached public structural-field results."""

import math
import sys
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.config import get_precision_mode, set_precision_mode
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA
from tnfr.physics.canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
    estimate_coherence_length_with_provenance,
)
from tnfr.physics.extended import compute_dnfr_flux, compute_phase_current
from tnfr.physics.fields import compute_unified_telemetry
from tnfr.physics.telemetry import compute_structural_telemetry
from tnfr.utils.cache import get_global_cache, reset_global_cache


@pytest.fixture(autouse=True)
def _isolated_field_cache():
    previous_mode = get_precision_mode()
    set_precision_mode("standard")
    reset_global_cache()
    yield
    set_precision_mode(previous_mode)
    reset_global_cache()


def _graph():
    graph = nx.path_graph(3)
    for node, pressure in enumerate((0.0, -0.5, 1.0)):
        graph.nodes[node].update(EPI=0.25, nu_f=1.0)
        graph.nodes[node][ALIAS_DNFR[0]] = pressure
        graph.nodes[node][ALIAS_THETA[0]] = 0.2 * node
    return graph


PRESSURE_READERS = (
    compute_structural_potential,
    estimate_coherence_length,
    estimate_coherence_length_with_provenance,
    compute_dnfr_flux,
    compute_structural_telemetry,
)

PUBLIC_MAPS = {
    "phi_s": compute_structural_potential,
    "grad_phi": compute_phase_gradient,
    "curv_phi": compute_phase_curvature,
    "j_phi": compute_phase_current,
    "j_dnfr": compute_dnfr_flux,
}


@pytest.mark.parametrize("reader", PRESSURE_READERS)
@pytest.mark.parametrize("warm", (False, True), ids=("cold", "warm"))
@pytest.mark.parametrize(
    "bad,error",
    (
        (False, TypeError),
        (np.bool_(False), TypeError),
        ("0", TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (None, TypeError),
        (0j, TypeError),
    ),
)
def test_malformed_authoritative_pressure_is_rejected_before_cached_reuse(
    reader, warm, bad, error
):
    graph = _graph()
    if warm:
        reader(graph)
    # The first alias remains authoritative even when a secondary spelling
    # contains a valid value. In particular False and "0" must not reuse 0.
    graph.nodes[0][ALIAS_DNFR[-1]] = 0.0
    graph.nodes[0][ALIAS_DNFR[0]] = bad
    with pytest.raises(error):
        reader(graph)


@pytest.mark.parametrize("reader", PRESSURE_READERS)
def test_valid_first_pressure_alias_ignores_a_malformed_secondary_alias(reader):
    graph = _graph()
    before = reader(graph)
    graph.nodes[0][ALIAS_DNFR[-1]] = float("nan")
    assert reader(graph) == before


@pytest.mark.parametrize(
    "bad,error", ((False, TypeError), ("0.2", TypeError), (float("nan"), ValueError))
)
def test_phase_current_rejects_malformed_primary_phase_after_cache_hit(bad, error):
    graph = _graph()
    compute_phase_current(graph)
    graph.nodes[1][ALIAS_THETA[-1]] = 0.2
    graph.nodes[1][ALIAS_THETA[0]] = bad
    with pytest.raises(error):
        compute_phase_current(graph)


def test_invalid_pressure_does_not_hide_independent_phase_readouts():
    graph = _graph()
    readers = (compute_phase_gradient, compute_phase_curvature, compute_phase_current)
    expected = [reader(graph) for reader in readers]
    graph.nodes[0][ALIAS_DNFR[0]] = float("nan")
    assert [reader(graph) for reader in readers] == expected


def test_missing_pressure_remains_zero_and_signed_pressure_is_not_a_magnitude():
    graph = nx.path_graph(2)
    assert compute_structural_potential(graph) == {0: 0.0, 1: 0.0}
    assert compute_dnfr_flux(graph) == {0: 0.0, 1: 0.0}
    missing = compute_structural_telemetry(graph)
    assert missing["phi_s"] == missing["j_dnfr"] == {0: 0.0, 1: 0.0}
    assert math.isfinite(missing["xi_c"])

    graph.nodes[0][ALIAS_DNFR[0]] = -0.25
    graph.nodes[1][ALIAS_DNFR[-1]] = 0.5
    assert compute_structural_potential(graph) == {0: 0.5, 1: -0.25}
    assert compute_dnfr_flux(graph) == {0: 0.75, 1: -0.75}
    signed = compute_structural_telemetry(graph)
    assert signed["phi_s"] == {0: 0.5, 1: -0.25}
    assert signed["j_dnfr"] == {0: 0.75, 1: -0.75}
    assert signed["xi_c"] == estimate_coherence_length(graph)


@pytest.mark.parametrize("magnitude", (math.ulp(0.0), sys.float_info.max / 2.0))
def test_finite_extreme_pressure_with_representable_fields_remains_supported(magnitude):
    graph = nx.path_graph(2)
    graph.nodes[0][ALIAS_DNFR[0]] = magnitude
    graph.nodes[1][ALIAS_DNFR[0]] = -magnitude
    expected_potential = {0: -magnitude, 1: magnitude}
    expected_flux = {0: -2.0 * magnitude, 1: 2.0 * magnitude}
    assert compute_structural_potential(graph) == expected_potential
    assert compute_dnfr_flux(graph) == expected_flux
    result = compute_structural_telemetry(graph)
    assert result["phi_s"] == expected_potential
    assert result["j_dnfr"] == expected_flux
    assert all(math.isfinite(value) for value in result["j_dnfr"].values())
    assert math.isfinite(result["xi_c"])


@pytest.mark.parametrize("reader", PRESSURE_READERS)
@pytest.mark.parametrize("value", (Fraction(2**1024), Fraction(1, 2**1075)))
def test_nonrepresentable_pressure_does_not_become_infinity_or_zero(reader, value):
    graph = _graph()
    reader(graph)
    graph.nodes[0][ALIAS_DNFR[0]] = value
    with pytest.raises(ValueError):
        reader(graph)


@pytest.mark.parametrize("name,reader", PUBLIC_MAPS.items())
def test_public_maps_are_detached_and_cannot_poison_structural_telemetry(name, reader):
    graph = _graph()
    baseline = deepcopy(reader(graph))
    telemetry_before = deepcopy(compute_structural_telemetry(graph))
    cache = get_global_cache()
    before_hits = cache.hits
    public = reader(graph)
    assert cache.hits > before_hits
    assert public == baseline
    public[0] = 12345.0
    public["caller_only"] = -99.0

    assert reader(graph) == baseline
    assert compute_structural_telemetry(graph)[name] == telemetry_before[name]
    public.clear()
    assert reader(graph) == baseline


def test_structural_telemetry_nested_maps_cannot_poison_direct_or_later_reads():
    graph = _graph()
    direct = {name: deepcopy(reader(graph)) for name, reader in PUBLIC_MAPS.items()}
    baseline = deepcopy(compute_structural_telemetry(graph))
    public = compute_structural_telemetry(graph)
    for name in PUBLIC_MAPS:
        public[name][0] = 12345.0
        public[name]["caller_only"] = -99.0
    public["xi_c"] = -1.0

    assert compute_structural_telemetry(graph) == baseline
    for name, reader in PUBLIC_MAPS.items():
        assert reader(graph) == direct[name]


def test_unified_telemetry_canonical_and_extended_views_are_detached():
    graph = _graph()
    baseline = deepcopy(compute_structural_telemetry(graph))
    phase_current = deepcopy(compute_phase_current(graph))
    pressure_flux = deepcopy(compute_dnfr_flux(graph))
    unified = compute_unified_telemetry(graph)
    for name in PUBLIC_MAPS:
        unified["canonical"][name].clear()
    unified["extended_canonical"]["phase_current"].clear()
    unified["extended_canonical"]["dnfr_flux"].clear()

    assert compute_structural_telemetry(graph) == baseline
    assert compute_phase_current(graph) == phase_current
    assert compute_dnfr_flux(graph) == pressure_flux
    repeated = compute_unified_telemetry(graph)
    assert repeated["canonical"] == baseline
    assert repeated["extended_canonical"]["phase_current"] == phase_current
    assert repeated["extended_canonical"]["dnfr_flux"] == pressure_flux
