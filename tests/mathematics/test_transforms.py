"""Tests for transform contracts and composite EPI regularity diagnostics."""

from __future__ import annotations

from typing import Callable

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics import (
    BEPIElement,
    COMPOSITE_EPI_REGULARITY_KIND,
    COMPOSITE_EPI_REGULARITY_PROVENANCE,
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

    message = str(excinfo.value).lower()
    for fragment in ("phase", "2"):
        assert fragment in message


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
