"""Failure, abstention and root-versus-growth controls for signal diagnostics."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import runpy

import networkx as nx
import numpy as np
import pytest

from tnfr.validation import signal_confrontation as sc


def _spatial_signal(values):
    # This is the eigenvalue-one mode of the symmetric P3 Laplacian.
    return np.array([values, np.zeros_like(values), -values])


def _fit(values):
    return sc._modal_roots_from_graph(nx.path_graph(3), _spatial_signal(values))


@pytest.mark.parametrize("length", [1, 2, 7])
def test_short_finite_window_abstains(length):
    data = _spatial_signal(np.arange(length, dtype=float))
    result = sc.confront_signal(data)
    assert result.modal_diagnostic.status == "unresolved"
    assert "8 time samples" in result.modal_diagnostic.reason
    assert result.wave_fraction is None
    assert result.diffusive_face_valid is None
    assert sc.emergent_wave_fraction(data) is None
    assert "unresolved" in result.summary()
    assert "WAVE" not in result.summary()


@pytest.mark.parametrize("value", [0.0, 4.0, 1e250])
def test_constant_signal_has_no_modal_evidence(value):
    result = sc.diagnose_modal_roots(np.full((3, 32), value))
    assert result.status == "unresolved"
    assert result.complex_root_fraction is None
    assert result.legacy_diffusive_face is None


def test_pure_uniform_graph_mode_does_not_become_diffusion():
    data = np.tile(np.sin(np.arange(80)), (4, 1))
    result = sc._modal_roots_from_graph(nx.cycle_graph(4), data)
    assert result.status == "unresolved"
    assert "nontrivial" in result.reason


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("reader", [
    sc.confront_signal, sc.diagnose_modal_roots, sc.emergent_wave_fraction,
    sc.estimate_quality_factor, sc.nodal_prediction_skill,
])
def test_nonfinite_observations_are_rejected_before_any_transform(bad, reader):
    data = np.zeros((3, 32))
    data[1, 20] = bad
    with pytest.raises(ValueError, match="finite"):
        reader(data)


@pytest.mark.parametrize("data", [
    np.empty((3, 0)), np.ones((2, 32)), np.ones(32),
    np.ones((3, 32), dtype=complex) * (1 + 1j),
])
def test_invalid_signal_shape_or_domain_is_rejected(data):
    with pytest.raises(ValueError):
        sc.confront_signal(data)


def test_modal_solver_exception_is_explicit_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("forced AR fit failure")

    monkeypatch.setattr(sc.np.linalg, "lstsq", fail)
    data = _spatial_signal(np.sin(np.arange(80)))
    result = sc.confront_signal(data)
    assert result.modal_diagnostic.status == "failure"
    assert "forced AR fit failure" in result.modal_diagnostic.reason
    assert result.wave_fraction is None
    assert result.diffusive_face_valid is None
    assert sc.emergent_wave_fraction(data) is None
    assert "failure" in result.summary()
    assert "WAVE" not in result.summary()
    payload = result.to_dict()
    assert payload["diffusive_face_valid"] is None
    assert payload["wave_fraction"] is None
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_eigensolver_and_graph_identity_errors_cannot_classify(monkeypatch):
    data = _spatial_signal(np.sin(np.arange(80)))
    graph = nx.relabel_nodes(nx.path_graph(3), {0: 5})
    report = sc._modal_roots_from_graph(graph, data)
    assert report.status == "failure"
    assert "channel indices" in report.reason

    def fail(*args, **kwargs):
        raise RuntimeError("forced eigensystem failure")

    monkeypatch.setattr(sc.np.linalg, "eigh", fail)
    report = sc.diagnose_modal_roots(data)
    assert report.status == "failure"
    assert "eigensystem failure" in report.reason
    assert report.legacy_diffusive_face is None


@pytest.mark.parametrize("radius,stability", [
    (0.97, "decaying"), (1.0, "unit_boundary"), (1.03, "growing"),
])
def test_complex_roots_do_not_imply_energy_conservation(radius, stability):
    t = np.arange(160)
    result = _fit(radius ** t * np.sin(0.8 * t))
    assert result.status == "resolved"
    assert result.root_classification == "complex_dominated"
    assert result.stability == stability
    if stability == "growing":
        assert result.legacy_diffusive_face is None
    else:
        assert result.legacy_diffusive_face is False


def test_real_growing_roots_are_not_relaxation():
    t = np.arange(80)
    result = _fit(1.08 ** t + 0.7 ** t)
    assert result.status == "resolved"
    assert result.root_classification == "real_dominated"
    assert result.stability == "growing"
    assert result.growing_modes == 1
    assert result.legacy_diffusive_face is None


@pytest.mark.parametrize("values", [
    np.arange(80, dtype=float), (-1.0) ** np.arange(80),
])
def test_repeated_or_rank_deficient_roots_are_unresolved(values):
    result = _fit(values)
    assert result.status == "unresolved"
    assert "rank-deficient or repeated-root" in result.reason
    assert result.unresolved_modes == 1
    assert result.complex_root_fraction is None


def test_unresolved_energetic_mode_is_not_dropped_from_the_verdict():
    t = np.arange(80, dtype=float)
    # Two independent P3 modes: identified sinusoid plus a repeated-root ramp.
    data = (_spatial_signal(np.sin(0.8 * t))
            + np.array([t, -np.sqrt(2) * t, t]))
    result = sc._modal_roots_from_graph(nx.path_graph(3), data)
    assert result.fitted_modes == result.unresolved_modes == 1
    assert result.status == "unresolved"
    assert result.complex_root_fraction is None
    assert result.legacy_diffusive_face is None


def test_nonfinite_solver_result_abstains(monkeypatch):
    monkeypatch.setattr(sc.np.linalg, "lstsq", lambda *args, **kwargs: (
        np.array([np.nan, 0.0, 0.0]), np.array([]), 3, np.ones(3),
    ))
    result = _fit(np.sin(np.arange(80)))
    assert result.status == "failure"
    assert "nonfinite AR(2)" in result.reason
    assert result.legacy_diffusive_face is None


def test_graph_construction_failure_has_a_reason(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("forced graph failure")

    monkeypatch.setattr(sc, "build_coupling_graph", fail)
    result = sc.diagnose_modal_roots(_spatial_signal(np.sin(np.arange(80))))
    assert result.status == "failure"
    assert "forced graph failure" in result.reason
    assert result.complex_root_fraction is None


def test_finite_amplitude_scaling_preserves_root_diagnostic():
    values = np.sin(np.arange(160) * 0.8)
    base = _fit(values)
    for scale in [1e-250, 1e250]:
        actual = _fit(values * scale)
        assert actual == base


def test_unknown_legacy_report_and_infinite_spectral_length_serialize():
    base = sc.confront_signal(_spatial_signal(np.sin(np.arange(80))))
    report = replace(
        base, modal_diagnostic=None, wave_fraction=None,
        diffusive_face_valid=None, xi_c=float("inf"),
    )
    assert "unresolved" in report.summary()
    assert "WAVE" not in report.summary()
    payload = report.to_dict()
    assert payload["xi_c"] is None
    assert payload["nonfinite_readouts"] == ["xi_c"]
    assert payload["modal_diagnostic"] is None
    json.dumps(payload, allow_nan=False)


def test_same_window_skill_is_explicitly_descriptive():
    result = sc.nodal_prediction_skill(np.ones((3, 32)))
    assert result.nodal_skill == result.ar1_skill == 0.0
    assert result.evaluation_scope == "same_window_descriptive_fit"
    assert asdict(result)["evaluation_scope"] == "same_window_descriptive_fit"
    assert "not held-out" in result.summary()
    assert result.capacity_domain == "inactive_boundary"


def test_negative_legacy_coefficient_is_exposed_not_clipped():
    result = sc.nodal_prediction_skill(_spatial_signal(1.02 ** np.arange(80)))
    assert result.diffusivity < 0.0
    assert result.capacity_domain == "negative_outside_model"
    assert "negative_outside_model" in result.summary()


def test_example_handles_constant_modal_abstention(monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / (
        "examples/10_applications/159_empirical_confrontation_pipeline.py"
    )
    namespace = runpy.run_path(str(path))
    main = namespace["main"]
    monkeypatch.setitem(main.__globals__, "load_signal", lambda: (
        np.ones((3, 32)), "constant diagnostic control",
    ))
    main()
    output = capsys.readouterr().out
    assert "unresolved" in output
    assert "complex-root fraction = unavailable" in output
    assert "no physical regime certificate" in output
    assert "same-window fit" in output
    assert "Separate synthetic P2 forecast-boundary demonstration" in output
    assert "not_admitted_by_this_score" in output
    assert "WAVE (under-damped)" not in output
