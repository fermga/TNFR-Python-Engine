"""Tests for transform contracts and composite EPI regularity diagnostics."""

from __future__ import annotations

from fractions import Fraction
from typing import Callable

import pytest

np = pytest.importorskip("numpy")

from tnfr.errors import TNFRValueError
from tnfr.mathematics import (
    COMPOSITE_EPI_REGULARITY_KIND,
    COMPOSITE_EPI_REGULARITY_PROVENANCE,
    BEPIElement,
    evaluate_coherence_transform,
    evaluate_composite_epi_regularity_transform,
    transforms,
)


@pytest.mark.parametrize(
    "callable_obj",
    [
        transforms.build_isometry_factory,
        transforms.validate_norm_preservation,
    ],
)
def test_contracts_pending_implementation(callable_obj: Callable[..., object]) -> None:
    call_args = {
        transforms.build_isometry_factory: dict(source_dimension=2, target_dimension=2),
        transforms.validate_norm_preservation: dict(
            transform=lambda data: data,
            probes=[[1.0, 0.0]],
            metric=lambda data: 1.0,
        ),
    }[callable_obj]

    with pytest.raises(NotImplementedError) as excinfo:
        callable_obj(**call_args)

    assert "not implemented" in str(excinfo.value).lower()


def test_regularity_trend_accepts_unbounded_increasing_values() -> None:
    report = transforms.assess_composite_epi_regularity_trend([1.2, 2.6, 8.4])

    assert report.is_monotonic
    assert report.violations == ()
    assert report.regularity_values == (1.2, 2.6, 8.4)
    assert report.metric_kind == COMPOSITE_EPI_REGULARITY_KIND
    assert report.provenance == COMPOSITE_EPI_REGULARITY_PROVENANCE


def test_regularity_trend_processes_bepi_sequence() -> None:
    grid = np.linspace(0.0, 1.0, 129)
    sequence = [
        BEPIElement(
            np.sin(frequency * np.pi * grid).astype(np.complex128),
            np.zeros(1, dtype=np.complex128),
            grid,
        )
        for frequency in (1.0, 2.0, 4.0)
    ]

    report = transforms.assess_composite_epi_regularity_trend(sequence)

    assert report.is_monotonic
    assert len(report.regularity_values) == 3
    assert report.regularity_values[0] < report.regularity_values[-1]


def test_regularity_trend_detects_drop_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        report = transforms.assess_composite_epi_regularity_trend(
            [1.0, 0.92, 0.95], tolerated_drop=0.03
        )

    assert not report.is_monotonic
    assert report.violations
    first = report.violations[0]
    assert first.kind == "drop"
    assert first.drop == pytest.approx(0.08)
    assert first.metric_kind == COMPOSITE_EPI_REGULARITY_KIND
    assert "composite epi regularity drop" in caplog.text.lower()


def test_regularity_trend_flags_plateau_when_forbidden() -> None:
    report = transforms.assess_composite_epi_regularity_trend(
        [1.0, 1.0, 1.02], allow_plateaus=False
    )

    assert not report.is_monotonic
    assert report.violations[0].kind == "plateau"


def test_regularity_transform_evaluator_reports_metric_identity() -> None:
    grid = np.linspace(0.0, 1.0, 4)
    element = BEPIElement(
        np.array([0.2 + 0.0j, -0.1 + 0.05j, 0.05 + 0.02j, 0.0 + 0.0j]),
        np.array([0.3 + 0.0j, -0.2 + 0.0j], dtype=np.complex128),
        grid,
    )

    def lift(element: BEPIElement) -> BEPIElement:
        return element.compose(lambda values: 1.2 * values)

    result = evaluate_composite_epi_regularity_transform(element, lift, kappa=1.0)
    assert result.satisfied
    assert result.regularity_after >= result.regularity_before
    assert result.metric_kind == COMPOSITE_EPI_REGULARITY_KIND
    assert result.provenance == COMPOSITE_EPI_REGULARITY_PROVENANCE

    ratio = result.ratio
    failing = evaluate_composite_epi_regularity_transform(
        element, lift, kappa=ratio + 1e-6, tolerance=0.0
    )
    assert not failing.satisfied
    assert failing.deficit > 0

    forgiving = evaluate_composite_epi_regularity_transform(
        element, lift, kappa=ratio + 1e-6, tolerance=1e-3
    )
    assert forgiving.satisfied


def test_historical_coherence_names_are_regularity_aliases() -> None:
    report = transforms.ensure_coherence_monotonicity([1.5, 2.0])
    assert report.coherence_values == report.regularity_values

    grid = np.linspace(0.0, 1.0, 4)
    element = BEPIElement(
        np.ones(4, dtype=np.complex128),
        np.zeros(1, dtype=np.complex128),
        grid,
    )
    evaluation = evaluate_coherence_transform(element, lambda value: value)
    assert evaluation.coherence_before == evaluation.regularity_before
    assert evaluation.coherence_after == evaluation.regularity_after
    assert evaluation.metric_kind == COMPOSITE_EPI_REGULARITY_KIND


@pytest.mark.parametrize(
    "values", [np.array([0.0]), np.array([0.0, 1.0, 2.0]), (Fraction(1, 4), 1.0)]
)
def test_regularity_trend_accepts_numeric_sequences_without_truth_testing(values):
    report = transforms.assess_composite_epi_regularity_trend(values)
    assert report.is_monotonic
    assert report.regularity_values == tuple(float(value) for value in values)


@pytest.mark.parametrize("values", [[], (), np.array([])])
def test_regularity_trend_rejects_empty_sequences(values):
    with pytest.raises(TNFRValueError, match="at least one entry"):
        transforms.assess_composite_epi_regularity_trend(values)


@pytest.mark.parametrize("values", ["12", b"12", bytearray(b"12"), {0: 9.0, 1: 1.0}])
def test_regularity_trend_rejects_non_numeric_sequence_containers(values):
    with pytest.raises(TypeError, match="sequence of regularity entries"):
        transforms.assess_composite_epi_regularity_trend(values)


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), "1.0", 1 + 0j, np.inf, np.nan, Fraction(1, 10**400)],
)
def test_regularity_trend_admits_values_before_materializing_them(value):
    with pytest.raises(TNFRValueError, match="finite representable real scalar"):
        transforms.assess_composite_epi_regularity_trend([value])


def test_regularity_trend_rejects_nested_numeric_arrays():
    with pytest.raises(TNFRValueError, match="finite representable real scalar"):
        transforms.assess_composite_epi_regularity_trend(np.array([[1.0], [2.0]]))


@pytest.mark.parametrize("name", ["atol", "tolerated_drop"])
@pytest.mark.parametrize(
    "value", [True, "0", -1.0, np.inf, np.nan, Fraction(1, 10**400)]
)
def test_regularity_trend_uses_shared_tolerance_admission(name, value):
    with pytest.raises(TNFRValueError, match=name):
        transforms.assess_composite_epi_regularity_trend([1.0, 2.0], **{name: value})


def test_regularity_trend_reports_normalized_tolerances():
    report = transforms.assess_composite_epi_regularity_trend(
        [1.0, 0.75], tolerated_drop=Fraction(1, 4), atol=np.float64(0)
    )
    assert report.is_monotonic
    assert report.tolerated_drop == 0.25
    assert type(report.tolerated_drop) is float
    assert type(report.atol) is float


@pytest.mark.parametrize("flag", [False, "false", "off", "0"])
def test_regularity_trend_parses_disabled_plateaus(flag):
    report = transforms.ensure_coherence_monotonicity([1.0, 1.0], allow_plateaus=flag)
    assert report.allow_plateaus is False
    assert not report.is_monotonic
    assert report.violations[0].kind == "plateau"


def test_regularity_trend_rejects_ambiguous_boolean_text():
    with pytest.raises(ValueError, match="true/false"):
        transforms.assess_composite_epi_regularity_trend([1.0], allow_plateaus="maybe")


def test_regularity_trend_rejects_nonfinite_computed_bepi_value():
    element = BEPIElement([0.0, 0.0], [1e308] * 4, [0.0, 1.0])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(TNFRValueError, match="finite"):
            transforms.assess_composite_epi_regularity_trend([element])


def test_regularity_drop_threshold_is_not_rounded_into_acceptance():
    report = transforms.assess_composite_epi_regularity_trend(
        [1.0, np.nextafter(1.0, 0.0)], tolerated_drop=2**-54, atol=0.0
    )
    assert not report.is_monotonic
    assert report.violations[0].kind == "drop"
    assert report.violations[0].drop == 2**-53


def test_regularity_growth_threshold_is_not_rounded_into_a_plateau():
    report = transforms.assess_composite_epi_regularity_trend(
        [1.0, np.nextafter(1.0, np.inf)], allow_plateaus=False, atol=3 * 2**-54
    )
    assert report.is_monotonic


def test_regularity_trend_handles_large_finite_comparison_arithmetic():
    increasing = transforms.assess_composite_epi_regularity_trend(
        [-1e308, 1e308], allow_plateaus=False
    )
    tolerated = transforms.assess_composite_epi_regularity_trend(
        [1e308, -1e308], tolerated_drop=1e308, atol=1e308
    )
    assert increasing.is_monotonic
    assert tolerated.is_monotonic


def test_regularity_trend_rejects_an_unrepresentable_reported_drop():
    with pytest.raises(TNFRValueError, match="regularity drop"):
        transforms.assess_composite_epi_regularity_trend([1e308, -1e308])
