r"""Tests for the finite S13 non-normal pressure-prediction audit."""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.physics import (
    NonnormalPredictionCertificate,
    NonnormalPredictorRecord,
    benchmark_nonnormal_prediction,
    deterministic_directed_family,
    measure_nonnormal_pressure_prediction,
)
from tnfr.physics.spectral_projectors import matrix_exponential
from tnfr.physics.transient_u2 import restricted_generator


@pytest.fixture(scope="module")
def benchmark() -> NonnormalPredictionCertificate:
    return benchmark_nonnormal_prediction(deterministic_directed_family())


def test_family_is_reproducible_and_has_strongly_connected_backbone():
    first = deterministic_directed_family(size=5)
    second = deterministic_directed_family(size=5)
    assert len(first) == 5
    for left, right in zip(first, second):
        assert left.shape == (10, 10)
        assert np.array_equal(left, right)
        assert all(left[i, (i + 1) % len(left)] > 0.0 for i in range(len(left)))
        assert restricted_generator(left).shape == (9, 9)


def test_pressure_obeys_same_restricted_semigroup():
    weights = deterministic_directed_family(size=5)[4]
    laplacian = restricted_generator(weights)
    pressure0 = np.linspace(-1.0, 1.0, len(laplacian))
    epi0 = np.linalg.solve(laplacian, -pressure0)
    time = 0.7
    propagator = matrix_exponential(-time * laplacian)

    pressure_from_epi = -laplacian @ (propagator @ epi0)
    pressure_direct = propagator @ pressure0
    assert np.allclose(pressure_from_epi, pressure_direct, atol=1e-10)


def test_single_graph_api_exposes_both_prediction_rules():
    weights = deterministic_directed_family(size=5)[4]
    record = measure_nonnormal_pressure_prediction(weights, index=7)
    assert record.index == 7
    assert record.spectral_abscissa < 0.0
    assert record.numerical_abscissa > 0.0
    assert not record.spectral_rule_predicts_burst
    assert record.lognorm_sign_estimate == "positive"
    assert record.lognorm_numerical_sign_status == "resolved_positive"
    assert record.lognorm_rule_predicts_burst
    assert record.measured_pressure_burst


def test_measured_peak_is_attained_by_a_pressure_witness(benchmark):
    record = benchmark.records[4]
    assert record.measured_pressure_burst
    laplacian = restricted_generator(deterministic_directed_family()[4])
    propagator = matrix_exponential(-record.peak_time_structural * laplacian)
    _, _, right = np.linalg.svd(propagator)
    witness = right[0]
    witness_gain = np.linalg.norm(propagator @ witness) / np.linalg.norm(witness)
    assert witness_gain == pytest.approx(record.peak_pressure_gain, rel=1e-10)


def test_default_family_separates_asymptotic_and_transient_rules(benchmark):
    assert benchmark.family_size == 16
    assert benchmark.calibration_size == 8
    assert benchmark.holdout_size == 8
    assert benchmark.measured_burst_count == 4
    assert benchmark.resolved_lognorm_sign_count == benchmark.family_size
    assert benchmark.unresolved_lognorm_sign_count == 0
    assert benchmark.all_lognorm_signs_numerically_resolved
    assert (
        benchmark.lognorm_verification_status
        == "all_signs_resolved_and_scan_consistent"
    )
    assert benchmark.all_spectrally_stable
    assert benchmark.scan_consistent_with_lognorm_theorem
    assert benchmark.records[0].lognorm_numerical_sign_status == (
        "resolved_nonpositive"
    )
    assert benchmark.records[0].lognorm_rule_predicts_burst is False

    # A stable-spectrum-only rule labels all cases safe and misses every burst.
    assert benchmark.spectral_rule_accuracy == pytest.approx(0.75)
    assert benchmark.spectral_rule_balanced_accuracy == pytest.approx(0.5)
    # The logarithmic-norm sign is the exact linear transient criterion.
    assert benchmark.lognorm_rule_accuracy == pytest.approx(1.0)
    assert benchmark.lognorm_rule_balanced_accuracy == pytest.approx(1.0)


def test_analytic_rule_remains_exact_on_predeclared_holdout(benchmark):
    assert benchmark.calibration_spectral_accuracy == pytest.approx(0.75)
    assert benchmark.holdout_spectral_accuracy == pytest.approx(0.75)
    assert benchmark.calibration_lognorm_accuracy == pytest.approx(1.0)
    assert benchmark.holdout_lognorm_accuracy == pytest.approx(1.0)
    assert "no fitted coefficients" in benchmark.split_rule


def test_nonnormal_predictors_rank_gain_better_in_this_finite_family(benchmark):
    assert (
        benchmark.spearman_numerical_abscissa_vs_gain
        > benchmark.spearman_spectral_abscissa_vs_gain
    )
    assert (
        benchmark.spearman_kreiss_bound_vs_gain
        > benchmark.spearman_spectral_abscissa_vs_gain
    )
    # Commutator size detects non-normality but does not order transient risk here.
    assert (
        benchmark.spearman_numerical_abscissa_vs_gain
        > benchmark.spearman_normality_residual_vs_gain
    )
    for record in benchmark.records:
        assert record.kreiss_lower_bound <= (
            record.peak_pressure_gain * (1.0 + 1e-6) + record.tolerance
        )


def test_claim_status_keeps_exact_measured_and_open_scopes_separate(benchmark):
    assert "EXACT" in benchmark.claim_status
    assert "MEASURED" in benchmark.claim_status
    assert "OPEN" in benchmark.claim_status
    assert "p=-L_rw x" in benchmark.pressure_semigroup_identity


def test_near_zero_lognorm_sign_abstains_from_exact_numerical_verification():
    family = deterministic_directed_family()
    # This fixed blend lies at a sign crossing for mu_2.  Its residual is many
    # orders below the matrix-derived backward-error tolerance.
    blend = 0.011165391908707193
    unresolved = (1.0 - blend) * family[0] + blend * family[4]

    record = measure_nonnormal_pressure_prediction(
        unresolved, t_max=0.1, samples=3, resolvent_grid=2
    )
    assert abs(record.numerical_abscissa) <= record.tolerance
    assert record.lognorm_sign_estimate in {"negative", "zero", "positive"}
    assert record.lognorm_numerical_sign_status == "unresolved_near_zero"
    assert record.lognorm_rule_predicts_burst is None

    certificate = benchmark_nonnormal_prediction(
        (unresolved,) * 4, t_max=0.1, samples=3, resolvent_grid=2
    )
    assert certificate.resolved_lognorm_sign_count == 0
    assert certificate.unresolved_lognorm_sign_count == 4
    assert not certificate.all_lognorm_signs_numerically_resolved
    assert not certificate.scan_consistent_with_lognorm_theorem
    assert certificate.lognorm_verification_status.startswith("unresolved_")
    assert np.isnan(certificate.lognorm_rule_accuracy)
    assert np.isnan(certificate.lognorm_rule_balanced_accuracy)
    assert "NUMERICALLY UNRESOLVED" in certificate.claim_status
    assert (
        "no exact finite-family verification is claimed" in certificate.claim_status
    )


def test_row_rescaling_does_not_change_prediction():
    family = deterministic_directed_family(size=3)
    weights = family[0]
    scaled = np.diag(np.geomspace(0.1, 10.0, len(weights))) @ weights
    certificate = benchmark_nonnormal_prediction(
        (weights, scaled, family[1], family[2]), samples=51, resolvent_grid=6
    )
    original, rescaled = certificate.records[:2]
    assert rescaled.spectral_abscissa == pytest.approx(
        original.spectral_abscissa, abs=1e-10
    )
    assert rescaled.numerical_abscissa == pytest.approx(
        original.numerical_abscissa, abs=1e-10
    )
    assert rescaled.peak_pressure_gain == pytest.approx(
        original.peak_pressure_gain, abs=1e-10
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 0},
        {"size": True},
        {"seed": True},
        {"nodes": 1},
        {"extra_edge_probability": True},
        {"extra_edge_probability": "0.2"},
        {"extra_edge_probability": 0.2j},
        {"extra_edge_probability": -0.1},
        {"extra_edge_probability": 1.1},
        {"log10_weight_span": True},
        {"log10_weight_span": "6.0"},
        {"log10_weight_span": 6.0j},
        {"log10_weight_span": -1.0},
        {"log10_weight_span": 1e4},
    ],
)
def test_family_rejects_invalid_design_parameters(kwargs):
    with pytest.raises(ValueError):
        deterministic_directed_family(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t_max": 0.0},
        {"t_max": True},
        {"t_max": "30.0"},
        {"t_max": 30.0j},
        {"t_max": float("inf")},
        {"t_max": 1e308},
        {"samples": 1},
        {"samples": True},
        {"resolvent_grid": 1},
        {"resolvent_grid": True},
    ],
)
def test_benchmark_rejects_invalid_scan_parameters(kwargs):
    with pytest.raises(ValueError):
        benchmark_nonnormal_prediction(deterministic_directed_family(size=4), **kwargs)


def test_benchmark_requires_both_partitions():
    with pytest.raises(ValueError, match="at least four"):
        benchmark_nonnormal_prediction(deterministic_directed_family(size=3))


def test_strongly_connected_numerically_unresolved_gap_is_not_misclassified():
    epsilon = 1e-12
    weights = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0 - epsilon, 0.0, epsilon],
            [epsilon, 0.0, 1.0 - epsilon],
        ]
    )

    with pytest.raises(ValueError, match="numerically unresolved"):
        measure_nonnormal_pressure_prediction(weights)


@pytest.mark.parametrize("index", [True, np.bool_(True), 0.5])
def test_single_graph_rejects_noninteger_index(index):
    with pytest.raises(ValueError, match="index must be an integer"):
        measure_nonnormal_pressure_prediction(
            deterministic_directed_family(size=1)[0], index=index
        )


def test_single_graph_rejects_boolean_conductance():
    with pytest.raises(ValueError, match="not booleans"):
        measure_nonnormal_pressure_prediction(
            np.array([[False, True], [True, False]])
        )


def test_public_records_are_typed_and_indexed(benchmark):
    assert isinstance(benchmark.records[0], NonnormalPredictorRecord)
    assert [record.index for record in benchmark.records] == list(range(16))
