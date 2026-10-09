"""Capacity/synchrony snapshots do not count observed oscillations."""

import math
import sys
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_THETA, ALIAS_VF
from tnfr.physics.structural_diffusion import (
    compute_emergent_pulse,
    compute_nodal_pulse,
)
from tnfr.sdk.simple import Network


def _snapshot(capacities):
    graph = nx.cycle_graph(len(capacities))
    for node, capacity in enumerate(capacities):
        graph.nodes[node].update(EPI=0.5, nu_f=capacity, theta=0.0, delta_nfr=0.0)
    return graph


def test_every_positive_represented_capacity_counts_without_activity_cutoff():
    graph = _snapshot(
        (
            0.0,
            -0.0,
            math.ulp(0.0),
            math.nextafter(1e-9, 0.0),
            1e-9,
            math.nextafter(1e-9, math.inf),
            1.0,
        )
    )
    before = deepcopy(dict(graph.nodes(data=True)))

    result = compute_nodal_pulse(graph)

    assert result["n_pulsing"] == 5
    assert result["n_nodes"] == 7
    assert dict(graph.nodes(data=True)) == before


def test_sdk_count_is_capacity_presence_even_at_zero_nodal_rate():
    graph = _snapshot((math.ulp(0.0), 0.25, 0.5, 1.0))
    result = Network(graph).resonance()

    assert result["n_pulsing"] == 4
    assert all(
        data["nu_f"] * data["delta_nfr"] == 0.0 for _, data in graph.nodes(data=True)
    )
    assert result["phase_coherence"] == pytest.approx(1.0)
    assert result["mean_local_resonance"] == pytest.approx(1.0)

    # The same supplied phases retain their synchrony with no capacity.
    nx.set_node_attributes(graph, 0.0, "nu_f")
    inactive = Network(graph).resonance()
    assert inactive["n_pulsing"] == 0
    assert inactive["phase_coherence"] == result["phase_coherence"]
    assert inactive["mean_local_resonance"] == result["mean_local_resonance"]


def test_empty_snapshot_retains_legacy_schema_without_active_nodes():
    result = compute_nodal_pulse(nx.Graph())

    assert set(result) == {
        "mean_frequency",
        "frequency_spread",
        "phase_coherence",
        "mean_local_resonance",
        "resonance_gate",
        "n_pulsing",
        "n_nodes",
    }
    assert result["n_pulsing"] == result["n_nodes"] == 0


@pytest.mark.parametrize("aliases", (ALIAS_VF, ALIAS_THETA), ids=("capacity", "phase"))
@pytest.mark.parametrize(
    "invalid", (math.nan, math.inf, -math.inf, True, np.bool_(False), "0", None, [0])
)
def test_invalid_authoritative_alias_cannot_be_replaced_with_zero_or_secondary(
    aliases, invalid, monkeypatch
):
    graph = _snapshot((0.5, 0.5, 0.5))
    graph.nodes[0].update({alias: 0.25 for alias in aliases})
    graph.nodes[0][aliases[0]] = invalid

    def unexpected_synchrony(_graph):
        pytest.fail("invalid state must be rejected before synchrony is calculated")

    monkeypatch.setattr("tnfr.gamma.kuramoto_R_psi", unexpected_synchrony)
    with pytest.raises((TypeError, ValueError)):
        compute_nodal_pulse(graph)


def test_negative_capacity_is_not_a_quiet_node():
    graph = _snapshot((-1.0, 0.5, 0.5))
    with pytest.raises(ValueError, match="nonnegative"):
        compute_nodal_pulse(graph)


@pytest.mark.parametrize(
    ("capacities", "mean", "spread"),
    (
        ((sys.float_info.max,) * 4, sys.float_info.max, 0.0),
        (
            (0.0, sys.float_info.max, 0.0, sys.float_info.max),
            sys.float_info.max / 2,
            sys.float_info.max / 2,
        ),
        (
            (math.nextafter(sys.float_info.max, 0.0), sys.float_info.max) * 2,
            math.nextafter(sys.float_info.max, 0.0),
            math.ulp(sys.float_info.max) / 2,
        ),
    ),
    ids=("constant-maximum", "full-range", "adjacent-maximum"),
)
def test_finite_extreme_capacity_statistics_do_not_overflow(capacities, mean, spread):
    with np.errstate(over="raise", invalid="raise"):
        result = compute_nodal_pulse(_snapshot(capacities))

    assert math.isfinite(result["mean_frequency"])
    assert result["mean_frequency"] == pytest.approx(mean, rel=2e-16)
    assert result["frequency_spread"] == spread
    assert result["phase_coherence"] == pytest.approx(1.0)
    assert result["mean_local_resonance"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "owner",
    (
        "tnfr.gamma.kuramoto_R_psi",
        "tnfr.metrics.coherence.coherence_matrix",
        "tnfr.metrics.coherence.local_phase_sync_weighted",
    ),
)
def test_failed_synchrony_owner_is_not_reported_as_zero(owner, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("synchrony computation failed")

    monkeypatch.setattr(owner, unavailable)
    with pytest.raises(RuntimeError, match="synchrony computation failed"):
        compute_nodal_pulse(_snapshot((0.5, 0.5, 0.5)))


@pytest.mark.parametrize("channel", ("global", "local"))
def test_nonfinite_synchrony_result_is_rejected(channel, monkeypatch):
    if channel == "global":
        monkeypatch.setattr("tnfr.gamma.kuramoto_R_psi", lambda graph: (math.nan, 0.0))
    else:
        monkeypatch.setattr(
            "tnfr.metrics.coherence.local_phase_sync_weighted",
            lambda *args, **kwargs: math.inf,
        )
    with pytest.raises(ValueError, match="finite"):
        compute_nodal_pulse(_snapshot((0.5, 0.5, 0.5)))


def test_disabled_affinity_has_no_available_local_synchrony():
    graph = _snapshot((0.5, 0.5, 0.5))
    graph.graph["COHERENCE"] = {"enabled": False}
    with pytest.raises(ValueError, match="unavailable"):
        compute_nodal_pulse(graph)


@pytest.mark.parametrize("invalid", (-1, True, np.bool_(False), 1.0, "2", None))
def test_auxiliary_wave_mode_count_requires_a_nonnegative_integer(invalid, monkeypatch):
    def unexpected_spectrum(_graph):
        pytest.fail("mode count must be validated before the spectrum is calculated")

    monkeypatch.setattr(
        "tnfr.physics.structural_diffusion._cached_eigenvalues", unexpected_spectrum
    )
    with pytest.raises(ValueError, match="n_modes"):
        compute_emergent_pulse(nx.path_graph(4), n_modes=invalid)


def test_auxiliary_wave_zero_or_truncated_output_retains_full_spectrum_statistics():
    graph = nx.path_graph(4)
    full = compute_emergent_pulse(graph)
    empty = compute_emergent_pulse(graph, n_modes=0)
    truncated = compute_emergent_pulse(graph, n_modes=np.int64(1))

    assert full["n_modes"] == 3
    assert empty["resonant_spectrum"] == []
    assert truncated["resonant_spectrum"] == full["resonant_spectrum"][:1]
    assert {
        key: value for key, value in empty.items() if key != "resonant_spectrum"
    } == {key: value for key, value in full.items() if key != "resonant_spectrum"}
    assert truncated["n_modes"] == full["n_modes"]
