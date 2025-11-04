"""Tests documenting the transforms contract behaviour."""

from __future__ import annotations

from typing import Callable

import pytest

np = pytest.importorskip("numpy")

from tnfr.mathematics import BEPIElement, evaluate_coherence_transform, transforms

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

def test_ensure_coherence_monotonicity_accepts_increasing_values() -> None:
    report = transforms.ensure_coherence_monotonicity([0.5, 0.6, 0.8])

    assert report.is_monotonic
    assert report.violations == ()
    assert tuple(report.coherence_values) == (0.5, 0.6, 0.8)

def test_ensure_coherence_monotonicity_processes_bepi_sequence() -> None:
    grid = np.linspace(0.0, 1.0, 4)
    base = BEPIElement(
        np.array([0.1 + 0.0j, 0.2 + 0.1j, -0.05 + 0.05j, 0.0 + 0.0j]),
        np.array([0.5 + 0.0j, 0.1 + 0.0j], dtype=np.complex128),
        grid,
    )

    def scale(element: BEPIElement, factor: float) -> BEPIElement:
        return element.compose(lambda values: factor * values)

    sequence = [scale(base, factor) for factor in (1.0, 1.05, 1.1)]
    report = transforms.ensure_coherence_monotonicity(sequence)

    assert report.is_monotonic
    assert len(report.coherence_values) == 3
    assert report.coherence_values[0] < report.coherence_values[-1]

def test_ensure_coherence_monotonicity_detects_drop_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        report = transforms.ensure_coherence_monotonicity([1.0, 0.92, 0.95], tolerated_drop=0.03)

    assert not report.is_monotonic
    assert report.violations
    first = report.violations[0]
    assert first.kind == "drop"
    assert first.drop == pytest.approx(0.08)
    assert "coherence drop" in caplog.text.lower()

def test_ensure_coherence_monotonicity_flags_plateau_when_forbidden() -> None:
    report = transforms.ensure_coherence_monotonicity([1.0, 1.0, 1.02], allow_plateaus=False)

    assert not report.is_monotonic
    assert report.violations[0].kind == "plateau"

def test_coherence_transform_evaluator_validates_kappa() -> None:
    grid = np.linspace(0.0, 1.0, 4)
    element = BEPIElement(
        np.array([0.2 + 0.0j, -0.1 + 0.05j, 0.05 + 0.02j, 0.0 + 0.0j]),
        np.array([0.3 + 0.0j, -0.2 + 0.0j], dtype=np.complex128),
        grid,
    )

    def lift(element: BEPIElement) -> BEPIElement:
        return element.compose(lambda values: 1.2 * values)

    result = evaluate_coherence_transform(element, lift, kappa=1.0)
    assert result.satisfied
    assert result.coherence_after >= result.coherence_before

    ratio = result.ratio
    failing = evaluate_coherence_transform(element, lift, kappa=ratio + 1e-6, tolerance=0.0)
    assert not failing.satisfied
    assert failing.deficit > 0

    forgiving = evaluate_coherence_transform(element, lift, kappa=ratio + 1e-6, tolerance=1e-3)
    assert forgiving.satisfied
