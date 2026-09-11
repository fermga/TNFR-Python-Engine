r"""Exact finite block margins for causally executed event/REMESH cycles.

Every adjacent row in a runtime schedule/REMESH telescope satisfies

``D_j = V_j - V_{j+1} = K_j + S_j``,

where ``K_j`` is the gain-based energy-drop lower bound and ``S_j`` is
the nonnegative represented-schedule slack.  Summing a nonempty contiguous
block gives ``D = K + S`` without multiplying fixed-history REMESH gains.
When ``V_before > 0``, ``kappa = K / V_before`` is a block-specific lower
drop fraction and ``1 - kappa`` bounds the observed endpoint energy ratio.

The input must be an intact :class:`ExecutedEventRemeshCycleSequence`.  This
binds the finite algebra to one same-invocation causal execution and its outer
graph transaction.  The result is still one observed block: it establishes no
uniform class coercivity, repeated or future stability, global executable gain,
solver accuracy or order, mesh convergence, or full TNFR stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from fractions import Fraction
from typing import Any

from ..errors import TNFRValueError
from ..operators.event_remesh_causal_runtime import (
    ExecutedEventRemeshCycleSequence,
)
from ..utils._structural_signature import proof_stamps_are_identical
from .runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation,
    RuntimeRemeshScheduleSequenceObservation,
)

__all__ = (
    "RuntimeRemeshScheduleBlockMarginObservation",
    "observe_executed_event_remesh_block_margin",
)


_PROOF_VERSION = "runtime_remesh_schedule_block_margin_v1"
_SCOPE = (
    "One exact nonempty contiguous block of adjacent schedule/REMESH energy "
    "balances selected from one intact causally executed finite sequence. "
    "The source execution, telescope and selected boundary objects are bound "
    "by process-local identity and revalidated from their authoritative proof "
    "relations. The summed lower bound, schedule slack, observed drop and "
    "normalized block diagnostics apply only to this recorded block. Uniform "
    "class coercivity, a repeated-cycle margin, repeated or future stability, "
    "a global executable gain, solver accuracy or order, mesh convergence and "
    "full TNFR stability are not certified."
)
_CONDITION_NAMES = (
    "source_execution_intact",
    "source_causal_execution_provenance_intact",
    "source_runtime_telescope_bound_by_identity",
    "nonempty_contiguous_boundary_block",
    "selected_boundaries_bound_by_identity",
    "every_selected_boundary_intact",
    "exact_augmented_energies_nonnegative",
    "exact_intermediate_augmented_energies_continuous",
    "exact_finite_block_energy_drop_telescope",
    "exact_block_lower_bound_and_slack_identity",
    "every_selected_schedule_slack_nonnegative",
    "exact_normalized_block_diagnostics_consistent",
    "exact_endpoint_energy_gain_upper_bound_satisfied",
)


def _raw_proof_stamp(value: Any) -> tuple[Any, ...] | None:
    try:
        stamp = object.__getattribute__(value, "_proof_stamp")
    except BaseException:
        return None
    return stamp if type(stamp) is tuple else None


def _execution_is_intact(value: Any) -> bool:
    if type(value) is not ExecutedEventRemeshCycleSequence:
        return False
    try:
        return bool(
            ExecutedEventRemeshCycleSequence._proof_fields_are_intact(value)
            is True
        )
    except BaseException:
        return False


def _strict_conditions(value: Any) -> bool:
    return bool(
        type(value) is tuple
        and len(value) == len(_CONDITION_NAMES)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bool
            and item[0] == _CONDITION_NAMES[index]
            for index, item in enumerate(value)
        )
    )


def _observation_values(
    value: "RuntimeRemeshScheduleBlockMarginObservation",
) -> dict[str, Any]:
    if type(value) is not RuntimeRemeshScheduleBlockMarginObservation:
        raise TypeError("observation must have its canonical result type")
    return {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(RuntimeRemeshScheduleBlockMarginObservation)
        if item.name != "_proof_stamp"
    }


def _proof_stamp_from_values(values: dict[str, Any]) -> tuple[Any, ...]:
    source = values["source_execution"]
    boundaries = values["boundaries"]
    return (
        _PROOF_VERSION,
        ("source-execution", id(source), _raw_proof_stamp(source)),
        values["start_boundary"],
        values["boundary_count"],
        tuple(
            ("runtime-telescope-boundary", id(item), _raw_proof_stamp(item))
            for item in boundaries
        )
        if type(boundaries) is tuple
        else ("invalid-boundary-container",),
        values["exact_augmented_energy_before"],
        values["exact_augmented_energy_after"],
        values["exact_gain_based_energy_drop_lower_bound"],
        values["exact_schedule_augmented_energy_gain_slack"],
        values["exact_energy_drop"],
        values["exact_gain_based_energy_drop_fraction_lower_bound"],
        values["exact_observed_energy_drop_fraction"],
        values["exact_endpoint_energy_gain_upper_bound"],
        values["conditions"],
    )


def _validated_boundary_range(
    total: int,
    start_boundary: int,
    boundary_count: int | None,
) -> tuple[int, int]:
    if type(start_boundary) is not int:
        raise TypeError("start_boundary must be an integer and not bool")
    if boundary_count is not None and type(boundary_count) is not int:
        raise TypeError("boundary_count must be an integer, None, and not bool")
    if start_boundary < 0 or start_boundary >= total:
        raise TNFRValueError("start_boundary is outside the runtime telescope")
    count = total - start_boundary if boundary_count is None else boundary_count
    if count <= 0:
        raise TNFRValueError("boundary_count must select a nonempty block")
    stop = start_boundary + count
    if stop > total:
        raise TNFRValueError("selected boundary block exceeds the runtime telescope")
    return count, stop


def _derive_values(
    source: ExecutedEventRemeshCycleSequence,
    start_boundary: int,
    boundary_count: int | None,
    *,
    boundary_overrides: (
        tuple[RuntimeRemeshScheduleBoundaryObservation, ...] | None
    ) = None,
    source_already_validated: bool = False,
) -> dict[str, Any]:
    if type(source) is not ExecutedEventRemeshCycleSequence:
        raise TypeError(
            "execution must be an ExecutedEventRemeshCycleSequence"
        )
    if not source_already_validated and not _execution_is_intact(source):
        raise TNFRValueError(
            "executed cycle sequence is unsealed, tampered, or inconsistent"
        )

    telescope = object.__getattribute__(source, "runtime_telescope")
    observed_sequence = object.__getattribute__(source, "observed_sequence")
    # The exact execution validator above recursively rederives this telescope
    # and all its boundaries. Reuse that result within this call; validation
    # trust is never retained between calls.
    if type(telescope) is not RuntimeRemeshScheduleSequenceObservation:
        raise TNFRValueError(
            "executed cycle sequence has no intact runtime telescope"
        )
    if telescope.source_sequence is not observed_sequence:
        raise TNFRValueError(
            "runtime telescope is not identity-bound to the executed sequence"
        )
    all_boundaries = telescope.boundaries
    if type(all_boundaries) is not tuple or not all_boundaries:
        raise TNFRValueError("runtime telescope has no adjacent boundary")

    count, stop = _validated_boundary_range(
        len(all_boundaries),
        start_boundary,
        boundary_count,
    )
    expected_boundaries = all_boundaries[start_boundary:stop]
    if boundary_overrides is None:
        selected = expected_boundaries
    else:
        selected = boundary_overrides
        if (
            type(selected) is not tuple
            or len(selected) != count
            or any(
                observed is not expected
                for observed, expected in zip(
                    selected,
                    expected_boundaries,
                    strict=True,
                )
            )
        ):
            raise TNFRValueError(
                "stored block boundaries are not the selected telescope objects"
            )
    selected_boundaries_intact = bool(
        selected
        and all(
            type(item) is RuntimeRemeshScheduleBoundaryObservation
            for item in selected
        )
    )
    if not selected_boundaries_intact:
        raise TNFRValueError(
            "selected runtime telescope boundary proof is not intact"
        )

    scalar_names = (
        "exact_augmented_energy_before",
        "exact_augmented_energy_after",
        "exact_energy_drop",
        "exact_gain_based_energy_drop_lower_bound",
        "exact_schedule_augmented_energy_gain_slack",
    )
    if any(
        type(getattr(boundary, name)) is not Fraction
        for boundary in selected
        for name in scalar_names
    ):
        raise TNFRValueError("selected boundary energy values must be exact")

    before = selected[0].exact_augmented_energy_before
    after = selected[-1].exact_augmented_energy_after
    lower_bound = sum(
        (
            item.exact_gain_based_energy_drop_lower_bound
            for item in selected
        ),
        Fraction(0),
    )
    slack = sum(
        (
            item.exact_schedule_augmented_energy_gain_slack
            for item in selected
        ),
        Fraction(0),
    )
    drop = sum(
        (item.exact_energy_drop for item in selected),
        Fraction(0),
    )
    continuous = all(
        left.exact_augmented_energy_after
        == right.exact_augmented_energy_before
        for left, right in zip(selected, selected[1:])
    )
    nonnegative_energies = bool(
        before >= 0
        and after >= 0
        and all(
            item.exact_augmented_energy_before >= 0
            and item.exact_augmented_energy_after >= 0
            for item in selected
        )
    )
    nonnegative_slacks = all(
        item.exact_schedule_augmented_energy_gain_slack >= 0
        for item in selected
    )

    if before == 0:
        lower_fraction = None
        observed_fraction = None
        endpoint_gain_upper_bound = None
        normalized_consistent = True
        endpoint_bound_satisfied = True
    else:
        lower_fraction = lower_bound / before
        observed_fraction = drop / before
        endpoint_gain_upper_bound = Fraction(1) - lower_fraction
        normalized_consistent = bool(
            observed_fraction == Fraction(1) - after / before
            and endpoint_gain_upper_bound
            == Fraction(1) - lower_bound / before
        )
        endpoint_bound_satisfied = bool(
            after / before <= endpoint_gain_upper_bound
        )

    conditions = (
        ("source_execution_intact", True),
        (
            "source_causal_execution_provenance_intact",
            True,
        ),
        (
            "source_runtime_telescope_bound_by_identity",
            telescope is source.runtime_telescope
            and telescope.source_sequence is source.observed_sequence,
        ),
        (
            "nonempty_contiguous_boundary_block",
            count > 0
            and stop <= len(all_boundaries)
            and len(selected) == count,
        ),
        (
            "selected_boundaries_bound_by_identity",
            all(
                observed is expected
                for observed, expected in zip(
                    selected,
                    expected_boundaries,
                    strict=True,
                )
            ),
        ),
        (
            "every_selected_boundary_intact",
            selected_boundaries_intact,
        ),
        ("exact_augmented_energies_nonnegative", nonnegative_energies),
        (
            "exact_intermediate_augmented_energies_continuous",
            continuous,
        ),
        (
            "exact_finite_block_energy_drop_telescope",
            drop == before - after,
        ),
        (
            "exact_block_lower_bound_and_slack_identity",
            drop == lower_bound + slack,
        ),
        (
            "every_selected_schedule_slack_nonnegative",
            nonnegative_slacks,
        ),
        (
            "exact_normalized_block_diagnostics_consistent",
            normalized_consistent,
        ),
        (
            "exact_endpoint_energy_gain_upper_bound_satisfied",
            endpoint_bound_satisfied,
        ),
    )
    if not _strict_conditions(conditions) or not all(
        passed for _name, passed in conditions
    ):
        failed = ", ".join(name for name, passed in conditions if not passed)
        raise TNFRValueError(f"runtime REMESH block margin failed: {failed}")

    return {
        "source_execution": source,
        "start_boundary": start_boundary,
        "boundary_count": count,
        "boundaries": selected,
        "exact_augmented_energy_before": before,
        "exact_augmented_energy_after": after,
        "exact_gain_based_energy_drop_lower_bound": lower_bound,
        "exact_schedule_augmented_energy_gain_slack": slack,
        "exact_energy_drop": drop,
        "exact_gain_based_energy_drop_fraction_lower_bound": lower_fraction,
        "exact_observed_energy_drop_fraction": observed_fraction,
        "exact_endpoint_energy_gain_upper_bound": endpoint_gain_upper_bound,
        "conditions": conditions,
    }


@dataclass(frozen=True, slots=True)
class RuntimeRemeshScheduleBlockMarginObservation:
    """Sealed exact margin for one causally executed finite boundary block."""

    source_execution: ExecutedEventRemeshCycleSequence = field(
        repr=False,
        compare=False,
    )
    start_boundary: int
    boundary_count: int
    boundaries: tuple[RuntimeRemeshScheduleBoundaryObservation, ...] = field(
        repr=False,
        compare=False,
    )
    exact_augmented_energy_before: Fraction
    exact_augmented_energy_after: Fraction
    exact_gain_based_energy_drop_lower_bound: Fraction
    exact_schedule_augmented_energy_gain_slack: Fraction
    exact_energy_drop: Fraction
    exact_gain_based_energy_drop_fraction_lower_bound: Fraction | None
    exact_observed_energy_drop_fraction: Fraction | None
    exact_endpoint_energy_gain_upper_bound: Fraction | None
    conditions: tuple[tuple[str, bool], ...]
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        try:
            current = _observation_values(self)
            current_stamp = _proof_stamp_from_values(current)
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                current_stamp,
            ):
                return False
            source = object.__getattribute__(self, "source_execution")
            if not _execution_is_intact(source):
                return False
            expected = _derive_values(
                source,
                object.__getattribute__(self, "start_boundary"),
                object.__getattribute__(self, "boundary_count"),
                boundary_overrides=object.__getattribute__(self, "boundaries"),
                source_already_validated=True,
            )
            expected_stamp = _proof_stamp_from_values(expected)
            return proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                expected_stamp,
            )
        except BaseException:
            return False

    @property
    def scope(self) -> str:
        return _SCOPE

    @property
    def block_observation_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def same_graph_execution_provenance_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def whole_sequence_graph_state_atomic(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def exact_finite_block_balance_certified(self) -> bool:
        return self._proof_fields_are_intact()

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        if not self._proof_fields_are_intact():
            return ("runtime_remesh_schedule_block_margin_proof_fields_intact",)
        return tuple(name for name, passed in self.conditions if not passed)

    @property
    def energy_nonincrease_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact() and self.exact_energy_drop >= 0
        )

    @property
    def energy_nonincrease_sufficiently_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_gain_based_energy_drop_lower_bound >= 0
        )

    @property
    def positive_normalized_block_margin_certified(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_augmented_energy_before > 0
            and self.exact_gain_based_energy_drop_lower_bound > 0
            and self.exact_gain_based_energy_drop_fraction_lower_bound
            == self.exact_gain_based_energy_drop_lower_bound
            / self.exact_augmented_energy_before
        )

    @property
    def strict_energy_contraction_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_augmented_energy_after
            < self.exact_augmented_energy_before
        )

    @property
    def zero_energy_preservation_observed(self) -> bool:
        return bool(
            self._proof_fields_are_intact()
            and self.exact_augmented_energy_before == 0
            and self.exact_augmented_energy_after == 0
        )

    @property
    def uniform_class_coercivity_certified(self) -> bool:
        return False

    @property
    def uniform_repeated_margin_certified(self) -> bool:
        return False

    @property
    def repeated_runtime_stability_certified(self) -> bool:
        return False

    @property
    def future_stability_certified(self) -> bool:
        return False

    @property
    def runtime_global_gain_certified(self) -> bool:
        return False

    @property
    def full_tnfr_stability_certified(self) -> bool:
        return False

    @property
    def solver_accuracy_certified(self) -> bool:
        return False

    @property
    def solver_order_certified(self) -> bool:
        return False

    @property
    def mesh_convergence_certified(self) -> bool:
        return False


def observe_executed_event_remesh_block_margin(
    execution: ExecutedEventRemeshCycleSequence,
    *,
    start_boundary: int = 0,
    boundary_count: int | None = None,
) -> RuntimeRemeshScheduleBlockMarginObservation:
    """Derive an exact margin for a contiguous causal execution block."""

    values = _derive_values(execution, start_boundary, boundary_count)
    result = RuntimeRemeshScheduleBlockMarginObservation(
        **values,
        _proof_stamp=_proof_stamp_from_values(values),
    )
    if not result._proof_fields_are_intact():
        raise RuntimeError(
            "constructed runtime REMESH block-margin proof is inconsistent"
        )
    return result
