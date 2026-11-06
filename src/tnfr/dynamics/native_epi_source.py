"""Opt-in EPI endpoint accounting of an actually executed native batch.

These frozen observations describe represented state and dispatch boundaries.
They authenticate neither a canonical implementation nor a later permission.
A native batch supplies no positive elapsed interval or growth-rate evidence.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .._exact_time import finite_represented_real
from ..constants.aliases import ALIAS_EPI
from ..types import require_finite_real_scalar_epi

__all__ = ("NativeEPIEndpoint", "NativeEPIDispatch", "NativeEPIBatchObservation")

ExactVector = tuple[Fraction, ...]
_MISSING = object()


@dataclass(frozen=True, slots=True)
class NativeEPIEndpoint:
    """One inert read of ordered node storage, signed form and runtime clock.

    Storage identifiers are local observations, not portable identity proofs.
    Opaque node labels are retained without recursively freezing their objects.
    Missing or unsupported form is unavailable, never projected or zero-filled.
    """

    nodes: tuple[Any, ...] | None
    node_storage_ids: tuple[int, ...] | None
    exact_epi: ExactVector | None
    exact_time: Fraction | None
    unavailability_reasons: tuple[str, ...]

    @property
    def available(self) -> bool:
        return (
            not self.unavailability_reasons
            and self.nodes is not None
            and self.node_storage_ids is not None
            and self.exact_epi is not None
            and self.exact_time is not None
        )


def _same_population(left: NativeEPIEndpoint, right: NativeEPIEndpoint) -> bool:
    return (
        left.nodes is not None
        and right.nodes is not None
        and len(left.nodes) == len(right.nodes)
        and all(a is b for a, b in zip(left.nodes, right.nodes, strict=True))
        and left.node_storage_ids == right.node_storage_ids
    )


def _difference(after: ExactVector, before: ExactVector) -> ExactVector:
    return tuple(b - a for a, b in zip(before, after, strict=True))


@dataclass(frozen=True, slots=True)
class NativeEPIDispatch:
    """Actual direct-dispatch endpoints, including a primitive that raised.

    The glyph is a dispatched label, not authentication of the callable used.
    ``returned`` distinguishes a completed call from visible partial effects.
    It does not certify later bookkeeping or whole-batch success.
    """

    index: int
    node: Any
    glyph: Any
    before: NativeEPIEndpoint
    after: NativeEPIEndpoint
    returned: bool
    failure_type: str | None

    @property
    def exact_increment(self) -> ExactVector | None:
        if (
            not self.before.available
            or not self.after.available
            or not _same_population(self.before, self.after)
        ):
            return None
        for endpoint in (self.before, self.after):
            if (
                type(endpoint.exact_epi) is not tuple
                or len(endpoint.exact_epi) != len(endpoint.nodes)
                or any(type(value) is not Fraction for value in endpoint.exact_epi)
            ):
                raise ValueError("dispatch EPI endpoint requires one Fraction per node")
        return _difference(self.after.exact_epi, self.before.exact_epi)


@dataclass(frozen=True, slots=True)
class NativeEPIBatchObservation:
    """Exact accounting of captured boundaries, separate from execution success.

    Totals require full scalar coverage, one ordered node/storage population
    and one unchanged represented clock. A failed prefix can satisfy that
    accounting domain while ``completed`` remains false. Variation sums the
    absolute captured increments, not an unobserved within-call trajectory.
    Even zero outside-dispatch variation cannot exclude an unseen round trip.
    """

    initial: NativeEPIEndpoint
    final: NativeEPIEndpoint
    dispatches: tuple[NativeEPIDispatch, ...]
    completed: bool
    failure_type: str | None
    available: bool
    unavailability_reasons: tuple[str, ...]
    clock_preserved: bool | None
    exact_net_increment: ExactVector | None = None
    exact_dispatch_increment: ExactVector | None = None
    exact_dispatch_variation: ExactVector | None = None
    exact_outside_dispatch_increment: ExactVector | None = None
    exact_outside_dispatch_variation: ExactVector | None = None
    exact_outside_dispatch_increments: tuple[ExactVector, ...] | None = None


class _UnsupportedCaptureValue(ValueError):
    pass


def _safe_scalar_types() -> tuple[type, ...]:
    from ..mathematics.unified_numerical import np

    names = (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "longdouble",
        "complex64",
        "complex128",
        "clongdouble",
        "bool_",
    )
    return (bool, int, float, complex, Fraction) + tuple(
        kind for name in names if isinstance(kind := getattr(np, name, None), type)
    )


def _safe_capture_value(value: Any, *, depth: int = 0) -> Any:
    """Copy known representations without invoking input-defined protocols.

    The shared scalar owner still decides mathematical admission. Restricting
    observation to inert storage avoids invoking an unrelated EPI coercion hook
    when the actual operator does not consume that coordinate.
    """
    from ..mathematics import BEPIElement
    from ..mathematics.unified_numerical import np
    from ..operators.network_stage import (
        _runtime_mapping_items,
        _runtime_stored_attribute,
    )

    if depth > 8:
        raise _UnsupportedCaptureValue("nested EPI capture representation")
    kind = type(value)
    if any(kind is scalar for scalar in _safe_scalar_types()):
        if kind is Fraction and (
            type(value.numerator) is not int or type(value.denominator) is not int
        ):
            raise _UnsupportedCaptureValue("unsupported Fraction components")
        return value
    if kind is tuple or kind is list:
        return tuple(_safe_capture_value(item, depth=depth + 1) for item in value)
    if kind is getattr(np, "ndarray", None):
        if value.dtype.kind not in "biufc":
            raise _UnsupportedCaptureValue("unsupported EPI array dtype")
        return _safe_capture_value(value.tolist(), depth=depth + 1)
    if kind is BEPIElement:
        return {
            key: _safe_capture_value(
                _runtime_stored_attribute(value, attribute), depth=depth + 1
            )
            for key, attribute in (
                ("continuous", "f_continuous"),
                ("discrete", "a_discrete"),
                ("grid", "x_grid"),
            )
        }
    # Mapping base readers bypass subclass get/items/iteration overrides.
    # Ignore unused metadata rather than coercing it during an EPI observation.
    if any(
        base is dict or base is OrderedDict
        for base in type.__getattribute__(kind, "__mro__")
    ):
        return {
            key: _safe_capture_value(item, depth=depth + 1)
            for key, item in _runtime_mapping_items(value)
            if type(key) is str
            and key in ("continuous", "discrete", "grid", "real", "imag")
        }
    raise _UnsupportedCaptureValue("unsupported EPI capture representation")


def _capture_endpoint(graph: Any, retained: list[Any]) -> NativeEPIEndpoint:
    from ..operators.network_stage import (
        _runtime_mapping_items,
        _runtime_mapping_known_value,
        _runtime_stored_attribute,
    )

    reasons: list[str] = []
    exact_time = None
    try:
        graph_data = _runtime_stored_attribute(graph, "graph")
        raw_time = _runtime_mapping_known_value(graph_data, "_t", 0.0)
        if not any(type(raw_time) is scalar for scalar in _safe_scalar_types()):
            raise _UnsupportedCaptureValue("unsupported clock representation")
        safe_time = _safe_capture_value(raw_time)
        exact_time = finite_represented_real(safe_time, "native capture clock")[1]
    except (TypeError, ValueError, OverflowError):
        reasons.append("invalid_clock")

    try:
        node_outer = _runtime_stored_attribute(graph, "_node")
        entries = _runtime_mapping_items(node_outer)
    except (TypeError, ValueError, OverflowError):
        return NativeEPIEndpoint(
            None, None, None, exact_time, tuple((*reasons, "unsupported_storage"))
        )
    nodes = tuple(node for node, _data in entries)
    storage_ids = tuple(id(data) for _node, data in entries)
    retained.extend(data for _node, data in entries)
    forms: list[Fraction] = []
    for _node, data in entries:
        try:
            raw = _MISSING
            for alias in ALIAS_EPI:
                raw = _runtime_mapping_known_value(data, alias, _MISSING)
                if raw is not _MISSING:
                    break
            if raw is _MISSING:
                reasons.append("missing_epi")
                continue
            safe_value = _safe_capture_value(raw)
            form = require_finite_real_scalar_epi(safe_value)
            forms.append(Fraction.from_float(form))
        except _UnsupportedCaptureValue:
            reasons.append("unsupported_epi_storage")
        except (TypeError, ValueError, OverflowError):
            reasons.append("invalid_epi")
    return NativeEPIEndpoint(
        nodes,
        storage_ids,
        tuple(forms) if len(forms) == len(nodes) else None,
        exact_time,
        tuple(dict.fromkeys(reasons)),
    )


def _sum_vectors(vectors: tuple[ExactVector, ...], dimension: int) -> ExactVector:
    return tuple(
        sum((vector[i] for vector in vectors), Fraction(0)) for i in range(dimension)
    )


def _batch_observation(
    initial, final, dispatches, completed, failure_type, coverage_failed
):
    endpoints = (
        (initial,)
        + tuple(
            endpoint for item in dispatches for endpoint in (item.before, item.after)
        )
        + (final,)
    )
    reasons = []
    if coverage_failed:
        reasons.append("dispatch_capture_failed")
    if any(not endpoint.available for endpoint in endpoints):
        reasons.append("endpoint_unavailable")
    if any(not _same_population(initial, endpoint) for endpoint in endpoints):
        reasons.append("node_population_or_storage_changed")
    clock_preserved = (
        all(endpoint.exact_time == initial.exact_time for endpoint in endpoints)
        if all(endpoint.exact_time is not None for endpoint in endpoints)
        else None
    )
    if clock_preserved is False:
        reasons.append("clock_changed")
    common = dict(
        initial=initial,
        final=final,
        dispatches=dispatches,
        completed=completed,
        failure_type=failure_type,
        available=not reasons,
        unavailability_reasons=tuple(reasons),
        clock_preserved=clock_preserved,
    )
    if reasons:
        return NativeEPIBatchObservation(**common)
    dimension = len(initial.nodes)
    increments = tuple(item.exact_increment for item in dispatches)
    gaps = []
    previous = initial.exact_epi
    for item in dispatches:
        gaps.append(_difference(item.before.exact_epi, previous))
        previous = item.after.exact_epi
    gaps.append(_difference(final.exact_epi, previous))
    gaps = tuple(gaps)
    net = _difference(final.exact_epi, initial.exact_epi)
    dispatched = _sum_vectors(increments, dimension)
    outside = _sum_vectors(gaps, dimension)
    if net != tuple(a + b for a, b in zip(dispatched, outside, strict=True)):
        raise RuntimeError("native EPI captured boundary accounting is inconsistent")
    return NativeEPIBatchObservation(
        **common,
        exact_net_increment=net,
        exact_dispatch_increment=dispatched,
        exact_dispatch_variation=_sum_vectors(
            tuple(tuple(abs(value) for value in row) for row in increments), dimension
        ),
        exact_outside_dispatch_increment=outside,
        exact_outside_dispatch_variation=_sum_vectors(
            tuple(tuple(abs(value) for value in row) for row in gaps), dimension
        ),
        exact_outside_dispatch_increments=gaps,
    )


def _failure_type(failure: BaseException | None) -> str | None:
    return None if failure is None else type.__getattribute__(type(failure), "__name__")


class _NativeEPIBatchCapture:
    """Call-local recorder; no detached report is accepted as authority."""

    def __init__(self, graph):
        self.graph = graph
        self.retained: list[Any] = []
        self.initial = _capture_endpoint(graph, self.retained)
        self.dispatches: list[NativeEPIDispatch] = []
        self.coverage_failed = False

    def before_dispatch(self):
        return _capture_endpoint(self.graph, self.retained)

    def after_dispatch(self, node, glyph, before, failure=None):
        self.dispatches.append(
            NativeEPIDispatch(
                len(self.dispatches),
                node,
                glyph,
                before,
                _capture_endpoint(self.graph, self.retained),
                failure is None,
                _failure_type(failure),
            )
        )

    def finish(self, failure=None):
        final = _capture_endpoint(self.graph, self.retained)
        return _batch_observation(
            self.initial,
            final,
            tuple(self.dispatches),
            failure is None,
            _failure_type(failure),
            self.coverage_failed,
        )
