"""Storage chart membership is independent of scalar temporal complexity.

All temporal inputs are synthetic arrays; these tests execute no dynamics.
"""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics import BEPIElement
from tnfr.riemann import epi_type_signature as diagnostic
from tnfr.types import ensure_bepi, serialize_bepi_json


@pytest.mark.parametrize("value", (-3.0, 0.0, 2.5))
def test_nonzero_discrete_scalar_embedding_is_not_non_scalar_storage(value):
    embedded = ensure_bepi(value)
    assert diagnostic._bepi_storage_fraction(
        (value, embedded, serialize_bepi_json(embedded)),
    ) == (0.0, 0, 3)


@pytest.mark.parametrize(
    "continuous,discrete",
    (
        ((1.0, -1.0), (0.0, 0.0)),
        ((1.0j, -1.0j), (0.0, 0.0)),
        ((1.0, 1.0), (1.0, 2.0)),
        ((1.0, np.nextafter(1.0, 2.0)), (1.0, 1.0)),
    ),
)
def test_resolved_nonuniform_or_complex_content_cannot_hide_in_equal_magnitudes(
    continuous,
    discrete,
):
    element = BEPIElement(continuous, discrete, (0.0, 1.0))
    # Even a very large historical atol cannot discard represented variation.
    for value in (element, serialize_bepi_json(element)):
        assert diagnostic._bepi_storage_fraction((value,), atol=100.0) == (1.0, 1, 1)


def test_storage_fraction_counts_a_mixed_finite_generator_once():
    rich = BEPIElement((1.0, -1.0), (0.0, 0.0), (0.0, 1.0))
    assert diagnostic._bepi_storage_fraction(
        value for value in (0.0, ensure_bepi(-2.0), rich, serialize_bepi_json(rich))
    ) == (0.5, 2, 4)
    assert diagnostic._bepi_storage_fraction(iter(())) == (0.0, 0, 0)


@pytest.mark.parametrize(
    "invalid",
    (
        True,
        np.bool_(False),
        "1.0",
        object(),
        np.inf,
        np.nan,
        SimpleNamespace(f_continuous=(0.0, 1.0), a_discrete=(0.0, 0.0)),
        {"continuous": [0.0, np.inf], "discrete": [0.0], "grid": [0.0, 1.0]},
    ),
)
def test_unsupported_or_nonfinite_storage_is_not_silently_classified_scalar(invalid):
    with pytest.raises((TypeError, ValueError)):
        diagnostic._bepi_storage_fraction((invalid,))


def test_mutated_nonfinite_bepi_array_is_rejected():
    element = ensure_bepi(1.0)
    element.f_continuous[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        diagnostic._bepi_storage_fraction((element,))


def _synthetic_probe(monkeypatch, trajectories, storage):
    graph = nx.path_graph(len(storage))
    for node, value in zip(graph, storage, strict=True):
        graph.nodes[node]["psi"] = value  # exercise canonical alias precedence
    monkeypatch.setattr(diagnostic, "_build_canonical_demo_graph", lambda *args: graph)
    monkeypatch.setattr(diagnostic, "_evolve_and_collect", lambda *args: trajectories)
    return graph


def test_broadband_scalar_signal_never_becomes_vector_necessity(monkeypatch):
    # An ordinary scalar impulse has a broad temporal spectrum; it remains R-valued.
    scalar = np.zeros(65)
    scalar[0] = 1.0
    trajectories = np.tile(scalar, (3, 1))
    _synthetic_probe(monkeypatch, trajectories, [0.0, ensure_bepi(-1.0), 2.0])
    result = diagnostic.compute_epi_type_signature(n_nodes=3, n_steps=64, n_bins=16)
    assert result.signature > 0.8
    assert result.effective_modes > 8
    assert result.storage_bepi_count == 0
    assert result.verdict == "REAL_SCALAR_STORAGE"
    assert not result.diagnostics["dimensional_necessity_assessed"]
    assert not result.diagnostics["minimal_realization_assessed"]
    assert "no type necessity" in result.summary()


def test_constant_scalar_and_broad_spectrum_share_only_the_storage_verdict(monkeypatch):
    _synthetic_probe(monkeypatch, np.full((3, 9), -2.0), [-2.0] * 3)
    result = diagnostic.compute_epi_type_signature(n_nodes=3, n_steps=8, n_bins=4)
    assert result.signature == 0.0
    assert result.effective_modes == 1.0
    assert result.verdict == "REAL_SCALAR_STORAGE"


def test_zero_entropy_does_not_erase_observed_non_scalar_storage(monkeypatch):
    rich = BEPIElement((1.0, -1.0), (0.0, 0.0), (0.0, 1.0))
    _synthetic_probe(monkeypatch, np.ones((3, 9)), [rich, 1.0, ensure_bepi(1.0)])
    result = diagnostic.compute_epi_type_signature(n_nodes=3, n_steps=8, n_bins=4)
    assert result.signature == 0.0
    assert result.storage_bepi_count == 1
    assert result.storage_bepi_fraction == 1 / 3
    assert result.verdict == "NONSCALAR_BEPI_STORAGE_OBSERVED"


def test_deprecated_thresholds_are_metadata_not_type_or_tolerance_tests(monkeypatch):
    _synthetic_probe(monkeypatch, np.ones((3, 9)), [1.0] * 3)
    result = diagnostic.compute_epi_type_signature(
        n_nodes=3,
        n_steps=8,
        n_bins=4,
        scalar_threshold=-100,
        bepi_threshold=-100,
        storage_atol=100,
    )
    assert result.verdict == "REAL_SCALAR_STORAGE"
    assert result.diagnostics["scalar_threshold"] == -100
    assert result.diagnostics["bepi_threshold"] == -100
    assert result.diagnostics["storage_atol"] == 100
    assert result.diagnostics["legacy_thresholds_used_for_verdict"] is False
    assert result.diagnostics["storage_tolerance_used"] is False


def test_invalid_primary_storage_cannot_fall_back_to_valid_alias(monkeypatch):
    graph = _synthetic_probe(monkeypatch, np.ones((3, 9)), [1.0] * 3)
    graph.nodes[0]["EPI"] = object()
    with pytest.raises(TypeError):
        diagnostic.compute_epi_type_signature(n_nodes=3, n_steps=8, n_bins=4)


def test_nonfinite_temporal_readout_cannot_return_a_storage_certificate(monkeypatch):
    trajectories = np.ones((3, 9))
    trajectories[0, 1] = np.nan
    _synthetic_probe(monkeypatch, trajectories, [1.0] * 3)
    with pytest.raises(ValueError, match="finite two-dimensional"):
        diagnostic.compute_epi_type_signature(n_nodes=3, n_steps=8, n_bins=4)
