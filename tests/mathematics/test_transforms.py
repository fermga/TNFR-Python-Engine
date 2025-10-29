"""Tests documenting the transforms contract behaviour."""

from __future__ import annotations

from typing import Callable

import pytest

from tnfr.mathematics import transforms


@pytest.mark.parametrize(
    "callable_obj",
    [
        transforms.build_isometry_factory,
        transforms.validate_norm_preservation,
        transforms.ensure_coherence_monotonicity,
    ],
)
def test_transform_contracts_raise_not_implemented(callable_obj: Callable[..., object]) -> None:
    call_args = {
        transforms.build_isometry_factory: dict(source_dimension=2, target_dimension=2),
        transforms.validate_norm_preservation: dict(
            transform=lambda data: data,
            probes=[[1.0, 0.0]],
            metric=lambda data: 1.0,
        ),
        transforms.ensure_coherence_monotonicity: dict(coherence_series=[1.0, 1.0]),
    }[callable_obj]

    with pytest.raises(NotImplementedError) as excinfo:
        callable_obj(**call_args)

    message = str(excinfo.value)
    expected_fragments = {
        transforms.build_isometry_factory: ["Phase 2", "isometry"],
        transforms.validate_norm_preservation: ["Phase 2", "norm"],
        transforms.ensure_coherence_monotonicity: ["Phase 2", "coherence"],
    }[callable_obj]
    lowered = message.lower()
    for fragment in expected_fragments:
        assert fragment.lower() in lowered
