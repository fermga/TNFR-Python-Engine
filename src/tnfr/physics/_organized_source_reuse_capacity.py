"""Detached capacity limits for repeated Silence (SHA) under held capacities.

The exact branch uses the represented initial capacity and factor as rational
parameters. The represented branch applies one correctly rounded binary64
multiplication per prospective call, matching the scalar SHA proposal without
invoking any event. Both require the inclusive positive pre-call floor.

This is a capacity-only necessary bound on reuse of Reception, Coherence and
Silence. Neither grammar, other live preconditions, organized-state retention
nor occurrence of a word follows from a capacity prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from numbers import Integral

from .._exact_time import (
    bounded_represented_real,
    represented_fraction,
    represented_fraction_as_float,
)
from ..operators.factor_contracts import resolve_operator_factors

_MAXIMUM_CAPACITY_CALLS = 4096


@dataclass(frozen=True)
class _SilenceCapacityLimit:
    """One arithmetic branch, including unresolved and non-floor outcomes."""

    status: str
    accepted_calls: int
    current_capacity: Q
    last_accepted_pre_capacity: Q | None
    first_blocked_call: int | None
    first_blocked_pre_capacity: Q | None
    blocked_rounded_product: Q | None = None

    @property
    def last_admissible_call(self) -> int | None:
        """Return an established terminal count, never a truncated prefix."""
        if self.first_blocked_call is None:
            return None
        return self.accepted_calls


@dataclass(frozen=True)
class _OrganizedSourceReuseCapacity:
    """Exact and represented conditional budgets with no event installation."""

    initial_capacity: Q
    silence_capacity_factor: Q
    minimum_capacity: Q
    maximum_calls: int
    factor_source: str
    exact: _SilenceCapacityLimit
    represented: _SilenceCapacityLimit
    represented_capacities: tuple[Q, ...]
    exact_capacity_at_represented_count: Q
    represented_minus_exact_at_represented_count: Q
    scope: tuple[str, ...] = (
        "positive_initial_capacity_positive_factor_strictly_below_one",
        "positive_inclusive_pre_call_capacity_floor",
        "capacity_held_between_silence_calls_no_replenishment",
        "exact_branch_uses_represented_parameters_without_rounding_products",
        "represented_branch_rounds_each_product_to_nearest_binary64",
        "capacity_only_not_grammar_other_preconditions_retention_or_occurrence",
        "a_later_silence_rejection_does_not_rollback_earlier_word_stages",
        "no_operator_event_continuous_trajectory_or_saved_response_evaluation",
        "budget_exhaustion_and_stagnation_do_not_establish_unlimited_reuse",
    )


def _positive_represented(value, label):
    _, exact = bounded_represented_real(value, label, minimum=0)
    if exact <= 0:
        raise ValueError(f"{label} must be strictly positive")
    return exact


def _capacity_branch(*, initial, factor, floor, budget, rounded):
    """Evaluate a finite conditional scalar recurrence from admitted values."""
    current, last_pre, count = initial, None, 0
    path = [current]
    exact_at_count = initial
    while True:
        # Resolve the very next floor violation even after the final permitted
        # multiplication. Reading this boundary is not an additional call.
        if current < floor:
            return (
                _SilenceCapacityLimit(
                    "floor_blocked", count, current, last_pre, count + 1, current
                ),
                tuple(path),
                exact_at_count,
            )
        if count == budget:
            return (
                _SilenceCapacityLimit(
                    "budget_exhausted", count, current, last_pre, None, None
                ),
                tuple(path),
                exact_at_count,
            )
        product = current * factor
        after = (
            represented_fraction(
                represented_fraction_as_float(product, "SHA capacity product"),
                "SHA represented capacity",
            )
            if rounded
            else product
        )
        if after == 0:
            # The runtime rejects a positive attenuated product rounded to
            # zero. Do not count this uncommitted proposal as an accepted call.
            return (
                _SilenceCapacityLimit(
                    "product_underflow",
                    count,
                    current,
                    last_pre,
                    count + 1,
                    current,
                    after,
                ),
                tuple(path),
                exact_at_count,
            )
        last_pre, current = current, after
        count += 1
        exact_at_count = exact_at_count * factor if rounded else current
        if rounded:
            path.append(current)
        if rounded and after == last_pre:
            return (
                _SilenceCapacityLimit(
                    "represented_stagnation", count, current, last_pre, None, None
                ),
                tuple(path),
                exact_at_count,
            )


def _assess_organized_source_reuse_capacity(
    *,
    minimum_capacity,
    maximum_calls,
    initial_capacity=1.0,
    silence_capacity_factor=None,
):
    """Bound repeated SHA capacity admission under the supplied held model.

    ``maximum_calls`` is an explicit work budget in ``[1, 4096]``. A branch
    that reaches it without resolving its next failure reports an unresolved
    prefix. Factor zero is valid for some runtime SHA configurations but lies
    outside this positive-capacity model.

    No graph is consumed: callers must separately establish that Reception,
    Coherence, any inter-event dynamics and all other activity hold the target
    capacity, and must retain every other admission and retention obligation.
    """
    if isinstance(maximum_calls, bool) or not isinstance(maximum_calls, Integral):
        raise TypeError("maximum_calls must be an integer")
    budget = int(maximum_calls)
    if not 1 <= budget <= _MAXIMUM_CAPACITY_CALLS:
        raise ValueError(f"maximum_calls must lie in [1, {_MAXIMUM_CAPACITY_CALLS}]")
    initial = _positive_represented(initial_capacity, "initial_capacity")
    floor = _positive_represented(minimum_capacity, "minimum_capacity")
    overrides = (
        None
        if silence_capacity_factor is None
        else {"SHA_vf_factor": silence_capacity_factor}
    )
    resolved = resolve_operator_factors(
        overrides, "SHA", active_keys=("SHA_vf_factor",)
    )
    factor = _positive_represented(resolved["SHA_vf_factor"], "silence_capacity_factor")
    arguments = {
        "initial": initial,
        "factor": factor,
        "floor": floor,
        "budget": budget,
    }
    exact, _, _ = _capacity_branch(**arguments, rounded=False)
    represented, path, exact_at_count = _capacity_branch(**arguments, rounded=True)
    return _OrganizedSourceReuseCapacity(
        initial_capacity=initial,
        silence_capacity_factor=factor,
        minimum_capacity=floor,
        maximum_calls=budget,
        factor_source=(
            "canonical_factor_registry"
            if silence_capacity_factor is None
            else "explicit_admitted_override"
        ),
        exact=exact,
        represented=represented,
        represented_capacities=path,
        exact_capacity_at_represented_count=exact_at_count,
        represented_minus_exact_at_represented_count=(
            represented.current_capacity - exact_at_count
        ),
    )
