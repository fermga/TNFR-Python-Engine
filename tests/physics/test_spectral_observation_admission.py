"""Primitive and arithmetic boundaries of finite real spectral observations."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.physics import spectral_conservation as spectral
from tnfr.physics.conservation import ConservationSnapshot

_READERS = ("balance", "ward", "energy", "sectors", "drift", "classification")
_CONSUMED = {
    "balance": "charge_density",
    "ward": "charge_density",
    "energy": "phi_s",
    "sectors": "phi_s",
    "drift": "phi_s",
    "classification": "divergence",
}
_FIELDS = (
    "charge_density",
    "divergence",
    "phi_s",
    "k_phi",
    "j_phi",
    "j_dnfr",
    "grad_phi",
)


def _snapshot(nodes=(0,), *, charge=0.0, phi=0.0, divergence=0.0):
    values = {field: {node: 0.0 for node in nodes} for field in _FIELDS}
    for field, value in (
        ("charge_density", charge),
        ("phi_s", phi),
        ("divergence", divergence),
    ):
        values[field] = {node: value for node in nodes}
    return ConservationSnapshot(**values)


def _call(reader, before, after, graph, **kwargs):
    if reader == "balance":
        return spectral.verify_spectral_conservation_balance(
            before, after, graph, **kwargs
        )
    if reader == "ward":
        return spectral.compute_spectral_ward_identity(before, after, "supplied", graph)
    if reader == "energy":
        return spectral.compute_spectral_structural_energy(
            before, after, graph, **kwargs
        )
    if reader == "sectors":
        return spectral.decompose_spectral_sectors(graph, after)
    if reader == "drift":
        return spectral.compute_spectral_energy_conservation(before, after, graph)
    return spectral.classify_spectral_modes(graph, after, **kwargs)


@pytest.mark.parametrize("reader", ["balance", "energy"])
@pytest.mark.parametrize(
    "invalid",
    [0.0, -1.0, np.nan, np.inf, True, np.bool_(True), "1", Fraction(1, 10**400)],
)
def test_invalid_time_cannot_create_a_conservation_or_stability_flag(
    monkeypatch, reader, invalid
):
    def unexpected(graph):
        pytest.fail("invalid time must reject before observing the spectral chart")

    monkeypatch.setattr(spectral, "get_laplacian_spectrum", unexpected)
    with pytest.raises(ValueError, match="dt"):
        _call(
            reader,
            _snapshot(),
            _snapshot(charge=1.0, phi=1.0),
            nx.empty_graph(1),
            dt=invalid,
        )


@pytest.mark.parametrize(
    ("reader", "parameter"),
    [
        ("balance", "tolerance"),
        ("energy", "stability_threshold"),
        ("classification", "threshold"),
    ],
)
@pytest.mark.parametrize(
    "invalid", [-1.0, np.nan, np.inf, True, np.bool_(True), "1", Fraction(1, 10**400)]
)
def test_invalid_policy_cannot_certify_nonzero_observations(reader, parameter, invalid):
    with pytest.raises(ValueError, match=parameter):
        _call(
            reader,
            _snapshot(),
            _snapshot(charge=1.0, phi=1.0, divergence=0.25),
            nx.empty_graph(1),
            **{parameter: invalid},
        )


@pytest.mark.parametrize("reader", _READERS)
@pytest.mark.parametrize(
    "invalid",
    [
        True,
        np.bool_(True),
        "1",
        1.0 + 0.0j,
        np.nan,
        np.inf,
        Fraction(1, 10**400),
        10**400,
    ],
)
def test_original_consumed_scalars_are_admitted_before_projection(
    monkeypatch, reader, invalid
):
    field = _CONSUMED[reader]
    after = replace(_snapshot(), **{field: {0: invalid}})

    def unexpected(*args):
        pytest.fail("an invalid primitive observation reached a projection")

    monkeypatch.setattr(spectral, "gft", unexpected)
    with pytest.raises(ValueError, match=field):
        _call(reader, _snapshot(), after, nx.empty_graph(1))


@pytest.mark.parametrize("reader", _READERS)
def test_unused_map_values_are_not_new_observation_premises(reader):
    unused = (
        "j_phi"
        if reader in ("balance", "ward", "sectors", "classification")
        else "charge_density"
    )
    snapshot = replace(_snapshot(), **{unused: {0: np.nan}})
    _call(reader, snapshot, snapshot, nx.empty_graph(1))


@pytest.mark.parametrize("reader", _READERS)
def test_empty_observation_cannot_supply_a_passing_diagnostic(reader):
    with pytest.raises(ValueError, match="nonempty"):
        _call(reader, _snapshot(()), _snapshot(()), nx.empty_graph(0))


@pytest.mark.parametrize("reader", _READERS)
@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[np.nan]]),
        np.array([[True]]),
        np.array([[1.0j]]),
        np.array([[2.0]]),
        np.eye(2),
    ],
)
def test_report_requires_a_full_finite_real_orthonormal_chart(
    monkeypatch, reader, invalid
):
    monkeypatch.setattr(
        spectral, "get_laplacian_spectrum", lambda graph: (np.array([0.0]), invalid)
    )
    with pytest.raises(ValueError, match="eigenbasis"):
        _call(reader, _snapshot(), _snapshot(), nx.empty_graph(1))


@pytest.mark.parametrize(
    "invalid",
    [np.array([np.nan]), np.array([-1.0]), np.array([True]), np.array([0.0, 1.0])],
)
def test_reported_eigenvalues_are_admitted(monkeypatch, invalid):
    monkeypatch.setattr(
        spectral, "get_laplacian_spectrum", lambda graph: (invalid, np.eye(1))
    )
    with pytest.raises(ValueError, match="eigenvalues"):
        _call("balance", _snapshot(), _snapshot(), nx.empty_graph(1))


@pytest.mark.parametrize("reader", _READERS)
@pytest.mark.parametrize(
    "invalid",
    [
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1.0j]),
        np.array([]),
        np.ones(2),
    ],
)
def test_bad_transform_results_reject_before_labels(monkeypatch, reader, invalid):
    monkeypatch.setattr(spectral, "gft", lambda signal, basis: invalid)
    with pytest.raises(ValueError, match="spectral coefficients"):
        _call(reader, _snapshot(), _snapshot(), nx.empty_graph(1))


@pytest.mark.parametrize("reader", _READERS[:-1])
@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
def test_nonrepresentable_squared_fields_do_not_become_energy_evidence(
    reader, magnitude
):
    with pytest.raises(ValueError, match="squared"):
        _call(
            reader,
            _snapshot(),
            _snapshot(charge=magnitude, phi=magnitude),
            nx.empty_graph(1),
        )


@pytest.mark.parametrize(
    ("reader", "after"),
    [("balance", _snapshot(charge=1e-150)), ("energy", _snapshot(phi=1e-75))],
)
def test_nonzero_modal_rates_cannot_underflow_into_passing_flags(reader, after):
    with pytest.raises(ValueError, match="modal rate"):
        _call(reader, _snapshot(), after, nx.empty_graph(1), dt=1e200)


def test_valid_constant_rationals_materialize_before_spectral_arithmetic():
    graph = nx.empty_graph(1)
    before = _snapshot()
    after = _snapshot(
        charge=Fraction(1, 2), phi=Fraction(1, 2), divergence=Fraction(1, 4)
    )
    balance = spectral.verify_spectral_conservation_balance(
        before, after, graph, dt=Fraction(1, 2), tolerance=0.0
    )
    np.testing.assert_array_equal(balance.mode_sources, [1.125])
    energy = spectral.compute_spectral_structural_energy(
        before, after, graph, dt=Fraction(1, 2), stability_threshold=Fraction(1, 4)
    )
    np.testing.assert_array_equal(energy.mode_energies_after, [0.125])
    assert energy.total_derivative == 0.25
    assert energy.is_spectrally_stable
    assert energy.n_unstable_modes == 0


@pytest.mark.parametrize(
    ("value", "threshold"), [(0.0, 0.0), (-0.0, 0.0), (0.5, 0.5), (-0.5, 0.5)]
)
def test_classification_includes_zero_and_exact_tolerance_boundary(value, threshold):
    result = spectral.classify_spectral_modes(
        nx.empty_graph(1), _snapshot(divergence=value), threshold=threshold
    )
    assert result["mode_labels"] == ["conserved"]
    assert result["n_conserved"] == 1


def test_default_zero_median_keeps_zero_divergence_conserved():
    result = spectral.classify_spectral_modes(nx.empty_graph(1), _snapshot())
    assert result["mode_labels"] == ["conserved"]


def test_large_finite_divergence_uses_finite_mean_and_rms():
    snapshot = _snapshot(divergence=1e308)
    result = spectral.verify_spectral_conservation_balance(
        snapshot, snapshot, nx.empty_graph(1), tolerance=0.0
    )
    np.testing.assert_array_equal(result.div_spectrum_mean, [1e308])
    np.testing.assert_array_equal(result.mode_sources, [1e308])
    assert result.overall_spectral_quality == pytest.approx(1e-308, rel=1e-14, abs=0.0)
    assert result.n_conserved_modes == 0


def test_nonzero_mean_divergence_cannot_underflow_to_a_passing_balance():
    before = _snapshot(divergence=np.nextafter(0.0, 1.0))
    with pytest.raises(ValueError, match="mean divergence.*underflows"):
        spectral.verify_spectral_conservation_balance(
            before, _snapshot(), nx.empty_graph(1)
        )


@pytest.mark.parametrize("size", [1, 2, 3])
def test_unobserved_bands_are_unavailable_instead_of_passing(size):
    graph = nx.path_graph(size)
    snapshot = _snapshot(tuple(graph))
    result = spectral.verify_spectral_conservation_balance(snapshot, snapshot, graph)
    assert result.conservation_quality_by_band == {
        "low": 1.0,
        "mid": None,
        "high": None,
    }


def test_default_even_median_avoids_avoidable_intermediate_overflow(monkeypatch):
    graph = nx.empty_graph(2)
    monkeypatch.setattr(
        spectral, "get_laplacian_spectrum", lambda graph: (np.zeros(2), np.eye(2))
    )
    result = spectral.classify_spectral_modes(
        graph, _snapshot((0, 1), divergence=1e308)
    )
    assert result["mode_labels"] == ["conserved", "conserved"]


def test_sector_ratio_retains_explicit_negligible_denominator_sentinel():
    result = spectral.decompose_spectral_sectors(nx.empty_graph(1), _snapshot(phi=1.0))
    assert result.potential_sector_energy == 1.0
    assert result.geometric_sector_energy == 0.0
    assert result.sector_ratio == np.inf
    assert result.dominant_sector == "potential"


def test_reciprocal_directed_graph_keeps_an_orthonormal_physics_chart():
    graph = nx.complete_graph(3).to_directed()
    snapshot = replace(
        _snapshot(tuple(graph)), charge_density={0: -1.0, 1: 0.0, 2: 1.0}
    )
    result = spectral.verify_spectral_conservation_balance(snapshot, snapshot, graph)
    assert result.parseval_before == pytest.approx(2.0)
    np.testing.assert_allclose(
        result.eigenvectors.T @ result.eigenvectors, np.eye(3), atol=1e-12
    )


@pytest.mark.parametrize("order", ["C", "F"])
def test_chart_admission_preserves_contiguous_projection_rounding(monkeypatch, order):
    rng = np.random.default_rng(483)
    basis = np.array(np.linalg.qr(rng.normal(size=(32, 32)))[0], order=order)
    signal = rng.normal(size=32)
    # Every orthonormal chart diagonalizes an edgeless graph's zero Laplacian.
    graph = nx.empty_graph(32)
    snapshot = replace(_snapshot(tuple(graph)), phi_s=dict(enumerate(signal)))
    expected = spectral.gft(signal, basis)
    monkeypatch.setattr(
        spectral, "get_laplacian_spectrum", lambda graph: (np.zeros(32), basis)
    )
    result = spectral.decompose_spectral_sectors(graph, snapshot)
    np.testing.assert_array_equal(result.phi_s_spectrum, expected)
