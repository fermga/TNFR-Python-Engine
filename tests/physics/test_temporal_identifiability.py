"""Finite-probe tests for TNFR temporal operator identifiability."""

from __future__ import annotations

import math

import pytest

from tnfr.mathematics.unified_numerical import np
from tnfr.operators.operator_contracts import contract_identifiability_certificate
from tnfr.physics import (
    certify_signature_noise_margin,
    certify_temporal_signature_matrix,
    identify_nearest_signature,
    probe_canonical_operator_identifiability,
)
from tnfr.physics.temporal_identifiability import _wrapped_phase_delta


def test_supplied_matrix_reports_exact_partition_and_numerical_ranks() -> None:
    certificate = certify_temporal_signature_matrix(
        ((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        ("a", "b", "c", "d"),
        feature_names=("first", "second"),
    )

    assert certificate.equivalence_classes == (("a", "b"), ("c",), ("d",))
    assert certificate.ambiguous_groups == (("a", "b"),)
    assert certificate.uniquely_identifiable == ("c", "d")
    assert certificate.matrix_rank == 2
    assert certificate.affine_rank == 2
    assert not certificate.all_rows_identifiable
    assert certificate.claim_status.startswith("EXACT row partition")


def test_declared_decimal_quantization_defines_row_equivalence() -> None:
    exact = certify_temporal_signature_matrix(
        ((0.12341,), (0.12342,)),
        ("left", "right"),
    )
    quantized = certify_temporal_signature_matrix(
        ((0.12341,), (0.12342,)),
        ("left", "right"),
        quantization_decimals=3,
    )

    assert exact.all_rows_identifiable
    assert quantized.ambiguous_groups == (("left", "right"),)


def test_noise_margin_is_half_minimum_scaled_separation() -> None:
    certificate = certify_signature_noise_margin(
        ((0.0, 0.0), (2.0, 0.0), (0.0, 4.0)),
        ("origin", "horizontal", "vertical"),
        feature_names=("x", "y"),
        feature_scales=(2.0, 4.0),
        norm="linf",
    )

    assert certificate.minimum_pairwise_distance == pytest.approx(1.0)
    assert certificate.certified_noise_radius == pytest.approx(0.5)
    assert certificate.all_prototypes_distinct
    assert certificate.strict_bound_required
    assert certificate.minimum_distance_is_conservative_lower_bound
    assert set(certificate.closest_pairs) == {
        ("origin", "horizontal"),
        ("origin", "vertical"),
        ("horizontal", "vertical"),
    }


def test_l2_distance_preserves_subnormal_separation() -> None:
    certificate = certify_signature_noise_margin(
        ((0.0, 0.0), (1.0e-200, 0.0)),
        ("origin", "tiny"),
        norm="l2",
    )

    assert certificate.all_prototypes_distinct
    assert certificate.minimum_pairwise_distance == pytest.approx(1.0e-200)
    assert certificate.certified_noise_radius == pytest.approx(5.0e-201)


def test_large_feature_scale_prevents_false_difference_overflow() -> None:
    certificate = certify_signature_noise_margin(
        ((-1.0e308,), (1.0e308,)),
        ("negative", "positive"),
        feature_scales=(1.0e308,),
    )

    assert certificate.minimum_pairwise_distance == pytest.approx(2.0)
    assert certificate.certified_noise_radius == pytest.approx(1.0)


def test_unrepresentable_scaled_separation_is_rejected() -> None:
    with pytest.raises(ValueError, match="below floating-point range"):
        certify_signature_noise_margin(
            ((0.0,), (math.ulp(0.0),)),
            ("origin", "smallest"),
            feature_scales=(1.0e308,),
        )


def test_nearest_signature_requires_strict_open_noise_ball() -> None:
    certificate = certify_signature_noise_margin(
        ((0.0,), (2.0,)),
        ("left", "right"),
    )

    inside = identify_nearest_signature((0.9,), certificate)
    midpoint = identify_nearest_signature((1.0,), certificate)
    outside = identify_nearest_signature((-1.5,), certificate)

    assert inside.robustly_identified
    assert inside.identified_label == "left"
    assert inside.nearest_distance == pytest.approx(0.9)
    assert not midpoint.robustly_identified
    assert midpoint.identified_label is None
    assert midpoint.nearest_labels == ("left", "right")
    assert not outside.robustly_identified
    assert outside.identified_label == "left"


def test_l2_midpoint_is_excluded_using_exact_binary_input_geometry() -> None:
    certificate = certify_signature_noise_margin(
        ((0.0, 0.0), (1.0, 1.0)),
        ("lower", "upper"),
        norm="l2",
    )

    midpoint = identify_nearest_signature((0.5, 0.5), certificate)

    assert midpoint.nearest_labels == ("lower", "upper")
    assert not midpoint.within_certified_radius
    assert not midpoint.robustly_identified


def test_duplicate_prototypes_have_zero_noise_margin() -> None:
    certificate = certify_signature_noise_margin(
        ((1.0, 2.0), (1.0, 2.0), (3.0, 4.0)),
        ("a", "b", "c"),
        norm="l2",
    )

    assert certificate.minimum_pairwise_distance == 0.0
    assert certificate.certified_noise_radius == 0.0
    assert not certificate.all_prototypes_distinct
    result = identify_nearest_signature((1.0, 2.0), certificate)
    assert result.nearest_labels == ("a", "b")
    assert not result.robustly_identified


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"signatures": ((1.0,),), "labels": ("one",)}, ValueError),
        (
            {
                "signatures": ((1.0,), (2.0,)),
                "labels": ("one", "two"),
                "feature_scales": (0.0,),
            },
            ValueError,
        ),
        (
            {
                "signatures": ((1.0,), (2.0,)),
                "labels": ("one", "two"),
                "norm": "l1",
            },
            ValueError,
        ),
        (
            {
                "signatures": ((1.0,), (2.0,)),
                "labels": ("one", "two"),
                "feature_scales": (True,),
            },
            TypeError,
        ),
    ],
)
def test_noise_margin_rejects_undefined_domains(kwargs, error) -> None:
    with pytest.raises(error):
        certify_signature_noise_margin(**kwargs)


@pytest.mark.parametrize(
    ("signatures", "labels", "feature_names", "error"),
    [
        ((), (), None, ValueError),
        (((1.0,), (2.0,)), ("one",), None, ValueError),
        (((1.0,), (2.0,)), ("same", "same"), None, ValueError),
        (((1.0, 2.0),), ("one",), ("only",), ValueError),
        (((1.0, math.nan),), ("one",), None, ValueError),
        (((True,),), ("one",), None, TypeError),
        ((("1.0",),), ("one",), None, TypeError),
        (((1.0,),), "one", None, TypeError),
    ],
)
def test_signature_matrix_rejects_malformed_domains(
    signatures, labels, feature_names, error
) -> None:
    with pytest.raises(error):
        certify_temporal_signature_matrix(
            signatures,
            labels,
            feature_names=feature_names,
        )


@pytest.mark.parametrize("decimals", [True, -1, 16, 2.5])
def test_signature_matrix_rejects_invalid_quantization(decimals) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify_temporal_signature_matrix(
            ((1.0,),),
            ("one",),
            quantization_decimals=decimals,
        )


def test_extreme_finite_rows_do_not_overflow_affine_rank() -> None:
    certificate = certify_temporal_signature_matrix(
        ((-1.0e308, 0.0), (1.0e308, 0.0), (0.0, 1.0e308)),
        ("left", "right", "top"),
    )

    assert certificate.matrix_rank == 2
    assert certificate.affine_rank == 2
    assert certificate.all_rows_identifiable


def test_temporal_phase_delta_uses_half_open_canonical_wrapping() -> None:
    wrapped = _wrapped_phase_delta(
        np.asarray((math.pi, -math.pi, 3.0 * math.pi, 0.25)),
        np.zeros(4),
    )

    assert wrapped == pytest.approx((-math.pi, -math.pi, -math.pi, 0.25))
    assert np.all(wrapped >= -math.pi)
    assert np.all(wrapped < math.pi)


def test_canonical_probe_is_reproducible_and_executes_all_operators() -> None:
    first = probe_canonical_operator_identifiability(
        probe_seeds=(7, 29), n_nodes=6
    )
    second = probe_canonical_operator_identifiability(
        probe_seeds=(7, 29), n_nodes=6
    )

    assert first == second
    assert len(first.requested_operators) == 13
    assert first.executed_operators == first.requested_operators
    assert len(first.algebra.signatures) == 13
    assert len(first.algebra.feature_names) == 2 * 17
    assert first.frames_per_probe == 1
    assert first.algebra.all_rows_identifiable
    assert first.unresolved_groups == ()
    assert first.claim_status.startswith("MEASURED on deterministic")
    assert first.noise_margin is not None
    assert first.noise_margin.all_prototypes_distinct
    assert first.noise_margin.certified_noise_radius > 0.0


def test_probe_noise_margin_uses_the_declared_quantization() -> None:
    certificate = probe_canonical_operator_identifiability(
        probe_seeds=(7,),
        n_nodes=6,
        quantization_decimals=0,
    )

    assert certificate.noise_margin is not None
    assert (
        certificate.noise_margin.all_prototypes_distinct
        is certificate.algebra.all_rows_identifiable
    )
    assert certificate.noise_margin.signatures == tuple(
        tuple(round(value, 0) for value in row)
        for row in certificate.algebra.signatures
    )


@pytest.mark.parametrize("observation", [(True,), ("0.0",)])
def test_nearest_signature_rejects_bool_and_string_coercion(observation) -> None:
    certificate = certify_signature_noise_margin(
        ((0.0,), (1.0,)),
        ("zero", "one"),
    )

    with pytest.raises(TypeError):
        identify_nearest_signature(observation, certificate)


def test_temporal_channels_separate_silence_from_contraction_collision() -> None:
    categorical = contract_identifiability_certificate()
    assert ("Silence", "Contraction") in categorical.ambiguous_groups

    temporal = probe_canonical_operator_identifiability(probe_seeds=(7,), n_nodes=6)
    rows = dict(zip(temporal.algebra.labels, temporal.algebra.signatures, strict=True))
    silence = rows["Silence"]
    contraction = rows["Contraction"]

    # Both lower νf on this probe, while only Contraction changes EPI.  The
    # quantitative temporal channel therefore resolves the categorical tie.
    assert silence[1] < 0.0
    assert contraction[1] < 0.0
    assert silence[0] == pytest.approx(0.0, abs=1e-15)
    assert abs(contraction[0]) > 1e-6
    assert temporal.silence_contraction_separated


def test_operator_identifiers_are_not_signature_features() -> None:
    certificate = probe_canonical_operator_identifiability(
        probe_seeds=(11,), n_nodes=5
    )
    feature_text = " ".join(certificate.algebra.feature_names).lower()
    for label in certificate.algebra.labels:
        assert label.lower() not in feature_text
    for glyph in (
        "al",
        "en",
        "il",
        "oz",
        "um",
        "ra",
        "sha",
        "val",
        "nul",
        "thol",
        "zhir",
        "nav",
        "remesh",
    ):
        assert f"_{glyph}_" not in f"_{feature_text}_"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probe_seeds": ()},
        {"probe_seeds": (7, 7)},
        {"probe_seeds": (True,)},
        {"probe_seeds": (1.5,)},
        {"n_nodes": 2},
        {"n_nodes": True},
        {"n_nodes": 4.5},
        {"rank_tolerance": -1.0},
        {"rank_tolerance": math.inf},
    ],
)
def test_canonical_probe_validates_experiment_domain(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        probe_canonical_operator_identifiability(**kwargs)
