"""Timestamped capacity secants, without a constitutive capacity law.

The samples use the graph's explicitly recorded runtime time. Two samples give
an interval secant; three give the change between consecutive interval rates
divided by their midpoint separation. These observations can include unresolved
events between endpoints and are not exact pointwise time derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, MutableMapping

from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_D2VF, ALIAS_DVF, ALIAS_VF
from ..types import GraphLike

_SAMPLES_KEY = "_capacity_rate_samples"
_DIAGNOSTIC_KEY = "capacity_rate_diagnostic"
CapacitySample = tuple[float, float]


def _samples(raw: Any) -> tuple[CapacitySample, ...]:
    """Validate the bounded retained observations without coercing chronology."""
    if not isinstance(raw, (tuple, list)) or len(raw) > 3:
        raise ValueError("capacity rate samples must contain at most three pairs")
    samples: list[CapacitySample] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("capacity rate samples must be (time, capacity) pairs")
        time = finite_represented_real(item[0], f"capacity sample {index} time")[0]
        capacity = finite_represented_real(item[1], f"capacity sample {index} value")[0]
        if samples and time <= samples[-1][0]:
            raise ValueError("retained capacity sample times must increase strictly")
        samples.append((time, capacity))
    return tuple(samples)


def _represented_or_none(exact: Fraction) -> float | None:
    """Expose only finite rates whose nonzero value survives representation."""
    try:
        return finite_represented_real(exact, "capacity diagnostic")[0]
    except ValueError:
        return None


def _secant(left: CapacitySample, right: CapacitySample) -> Fraction:
    return (Fraction.from_float(right[1]) - Fraction.from_float(left[1])) / (
        Fraction.from_float(right[0]) - Fraction.from_float(left[0])
    )


@dataclass(frozen=True)
class CapacityRateObservation:
    """One finite-sample read-out, with availability separate from numeric zero."""

    sample_time: float | None
    samples: tuple[CapacitySample, ...]
    status: str
    rate: float | None = None
    second_difference: float | None = None
    rate_status: str = "unavailable"
    second_difference_status: str = "unavailable"

    def as_payload(self) -> dict[str, Any]:
        advancing = self.status == "advancing_time"
        return {
            "time_basis": "graph_runtime_time",
            "sample_time": self.sample_time,
            "sample_count": len(self.samples),
            "status": self.status,
            "rate": self.rate,
            "rate_status": self.rate_status,
            "second_difference": self.second_difference,
            "second_difference_status": self.second_difference_status,
            "rate_interval": (
                [self.samples[-2][0], self.samples[-1][0]]
                if advancing and len(self.samples) >= 2
                else None
            ),
            "second_difference_times": (
                [sample[0] for sample in self.samples]
                if advancing and len(self.samples) == 3
                else None
            ),
        }


def observe_capacity_rates(
    capacity: float | None,
    sample_time: float | None,
    previous_samples: Any = (),
) -> CapacityRateObservation:
    """Observe supplied samples without advancing capacity or inventing time.

    ``None`` explicitly denotes a missing source in this detached interface.
    Graph adapters distinguish absent attributes from malformed supplied values.
    Signed finite capacities are accepted as numerical observations; this does
    not certify the physical admissibility of a capacity state.

    A changed same-time sample resets the history at the right-hand state.
    An identical same-time observation preserves history but creates no new
    interval. Backward time is rejected. Untimestamped legacy derivatives are
    never accepted as prior samples.
    """
    samples = _samples(previous_samples)
    time = (
        None
        if sample_time is None
        else finite_represented_real(sample_time, "capacity observation time")[0]
    )
    value = (
        None
        if capacity is None
        else finite_represented_real(capacity, "capacity observation value")[0]
    )
    if time is not None and samples and time < samples[-1][0]:
        raise ValueError("runtime time precedes the latest capacity sample")
    if time is None:
        return CapacityRateObservation(None, (), "missing_time")
    if value is None:
        return CapacityRateObservation(time, (), "missing_capacity")

    current = (time, value)
    if not samples:
        return CapacityRateObservation(time, (current,), "initial_sample")
    if time == samples[-1][0]:
        if value != samples[-1][1]:
            return CapacityRateObservation(time, (current,), "same_time_jump")
        return CapacityRateObservation(time, samples, "duplicate_time")

    samples = (samples + (current,))[-3:]
    exact_rate = _secant(samples[-2], samples[-1])
    rate = _represented_or_none(exact_rate)
    second = None
    second_status = "insufficient_samples"
    if len(samples) == 3:
        # Midpoint separation = (t_n - t_(n-2))/2. Exact represented times
        # avoid collapse of rounded midpoint timestamps at large time origins.
        span = Fraction.from_float(samples[-1][0]) - Fraction.from_float(samples[0][0])
        exact_second = 2 * (exact_rate - _secant(samples[0], samples[1])) / span
        second = _represented_or_none(exact_second)
        second_status = "available" if second is not None else "unrepresentable"
    return CapacityRateObservation(
        time,
        samples,
        "advancing_time",
        rate,
        second,
        "available" if rate is not None else "unrepresentable",
        second_status,
    )


def plan_capacity_rate_observations(
    G: GraphLike,
) -> dict[Any, CapacityRateObservation]:
    """Validate every node before a caller commits any diagnostic writes."""
    time = (
        finite_represented_real(G.graph["_t"], "graph runtime time")[0]
        if "_t" in G.graph
        else None
    )
    result: dict[Any, CapacityRateObservation] = {}
    for node, data in G.nodes(data=True):
        if any(alias in data for alias in ALIAS_VF):
            raw_capacity = get_attr(
                data, ALIAS_VF, strict=True, conv=lambda value: value
            )
            capacity = finite_represented_real(raw_capacity, "stored capacity")[0]
        else:
            capacity = None
        result[node] = observe_capacity_rates(
            capacity, time, data.get(_SAMPLES_KEY, ())
        )
    return result


def _write_optional_aliases(
    data: MutableMapping[str, Any], aliases: tuple[str, ...], value: float | None
) -> None:
    """Keep existing diagnostic spellings consistent, including unavailable."""
    present = [alias for alias in aliases if alias in data]
    for alias in present or [aliases[0]]:
        data[alias] = value


def commit_capacity_rate_observations(
    G: GraphLike, observations: Mapping[Any, CapacityRateObservation]
) -> None:
    """Commit a prevalidated observation batch; no nodal channel is changed."""
    for node, observation in observations.items():
        data = G.nodes[node]
        data[_SAMPLES_KEY] = observation.samples
        data[_DIAGNOSTIC_KEY] = observation.as_payload()
        _write_optional_aliases(data, ALIAS_DVF, observation.rate)
        _write_optional_aliases(data, ALIAS_D2VF, observation.second_difference)
        data.pop("_prev_vf", None)
        data.pop("_prev_dvf", None)


def aggregate_capacity_rates(
    observations: Mapping[Any, CapacityRateObservation],
) -> tuple[float | None, dict[str, Any]]:
    """Require complete second-difference coverage for the historical B mean."""
    count = len(observations)
    first_count = sum(item.rate is not None for item in observations.values())
    seconds = tuple(
        item.second_difference
        for item in observations.values()
        if item.second_difference is not None
    )
    mean = None
    status = "empty" if not count else "incomplete"
    if count and len(seconds) == count:
        exact_mean = (
            sum((Fraction.from_float(value) for value in seconds), Fraction()) / count
        )
        mean = _represented_or_none(exact_mean)
        status = "complete" if mean is not None else "aggregate_unrepresentable"
    return mean, {
        "node_count": count,
        "first_available": first_count,
        "second_available": len(seconds),
        "status": status,
    }
