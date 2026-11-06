"""Conditional composition of a represented NUL image and a sine-law bound.

The consumer rebuilds its own reset and successor family from primitive boxes.
Its consistency checks bind the declared nominal initial vectors to that image;
they do not prove the external whole-cell solution, live grammar/history,
target-selection or event-admission premises. No graph event or trajectory is
executed. Optional scalar-policy eligibility remains a separate shared check.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import nul_reset
from . import sine_capacity_metric as capacity_metric
from . import sine_preparation_metric as metric

__all__ = ("NulSuccessorRun", "bound_nul_successor_cells")


@dataclass(frozen=True)
class NulSuccessorRun:
    """Rebuilt image and conditional arithmetic, without an admission verdict.

    Source and successor vectors retain absolute form origins and phase lifts.
    ``run.status == 'completed'`` has precisely the conditional meaning of the
    capacity-comparison owner. No nominal observation-radius or angle-gap gate
    is established by this report. Reports are never accepted as input premises.
    """

    reset: nul_reset.NulCapacityResetEnclosure
    source_form: tuple
    source_phase: tuple
    source_capacity: tuple
    successor_form: tuple
    successor_phase: tuple
    successor_capacity: tuple
    nominal_form: tuple
    nominal_phase: tuple
    nominal_capacities: tuple
    child_form: nul_reset.NulCoordinateBounds | None
    child_phase: nul_reset.NulCoordinateBounds | None
    family: metric.MetricFamily
    run: capacity_metric.CapacityMetricRun


def _coordinate_batch(values, size, label):
    raw = metric._items(values, label, metric.MAX_METRIC_NODES)
    if len(raw) != size:
        raise ValueError(f"{label} has the wrong component dimension")
    return tuple(
        nul_reset._coordinate_bounds(value, f"{label}[{index}]")
        for index, value in enumerate(raw)
    )


def _represented_triple(bounds):
    # Each rational was admitted from a genuine float and round-trips exactly.
    return tuple(float(value) for value in (bounds.lower, bounds.nominal, bounds.upper))


def _nominal(values, actual, label, budget):
    raw = metric._items(values, label, len(actual))
    if len(raw) != len(actual):
        raise ValueError(f"{label} has the wrong component dimension")
    admitted = tuple(metric._input(value, label, budget) for value in raw)
    if admitted != tuple(bounds.nominal for bounds in actual):
        raise ValueError(f"{label} differs from the actual nominal NUL image")
    return admitted


def _cells(values, size, label, budget):
    raw = metric._items(values, label, metric.MAX_METRIC_CELLS)
    if not raw:
        raise ValueError(f"{label} requires a nonempty whole-cell sequence")
    return tuple(metric._pairs(cell, size, label, budget) for cell in raw)


def _errors(bounds):
    return bounds.lower_error, bounds.upper_error


def bound_nul_successor_cells(
    *,
    factor,
    parent,
    form_bounds,
    phase_bounds,
    capacity_bounds,
    pressure_bounds,
    edges,
    conductances,
    degrees,
    reference_weight,
    nominal_form,
    nominal_phase,
    nominal_capacities,
    phase_cells,
    form_cells,
    edge_allowances,
    durations,
    child=None,
    child_form_bounds=None,
    child_phase_bounds=None,
    epi_min=-1.0,
    epi_max=1.0,
):
    """Bind one marked hard NUL image to its own conditional successor bound.

    Form, phase and capacity inputs are complete node-ordered triples
    ``(lower, nominal, upper)`` of genuine finite binary64 floats. Only the
    marked parent's stored-pressure triple is consumed by the reset. The
    same factor and rails apply to every member; phases and untargeted state
    are unchanged. Positive successor capacities are held during each member's
    subsequent full-sine flow, on the supplied fixed support and clock.

    The independently declared nominal vectors require exact int/Fraction
    entries and must match the actual reset image. The first whole-cell boxes
    must contain that initial point. These necessary consistency checks do
    not authenticate the boxes as a solution of the successor law. All their
    whole-time, derivative/remainder and source-association premises remain
    external, as do common stage admission and fixed wrapper context.

    For a connected portrait supply a distinct in-component child. Otherwise
    supply both external-child coordinate triples under its independent
    constant law. Both nominal coordinates and asymmetric errors are retained;
    the external child is not a target. No observation radius or net-angle
    claim follows without its separately established nominal premises.
    """
    raw_capacity = metric._items(
        capacity_bounds, "capacity_bounds", metric.MAX_METRIC_NODES
    )
    size = len(raw_capacity)
    if size < 2:
        raise ValueError("a dynamic component requires between 2 and 21 nodes")
    parent = metric._index(parent, size, "parent")
    source_capacity = _coordinate_batch(raw_capacity, size, "capacity_bounds")
    source_form = _coordinate_batch(form_bounds, size, "form_bounds")
    source_phase = _coordinate_batch(phase_bounds, size, "phase_bounds")
    if any(bounds.lower <= 0 for bounds in source_capacity):
        raise ValueError("all source capacity lower endpoints must be positive")
    if child is None:
        child_form = nul_reset._coordinate_bounds(child_form_bounds, "child form")
        child_phase = nul_reset._coordinate_bounds(child_phase_bounds, "child phase")
    else:
        child = metric._index(child, size, "child")
        if (
            child == parent
            or child_form_bounds is not None
            or child_phase_bounds is not None
        ):
            raise ValueError(
                "connected portraits require a distinct child and no external state"
            )
        child_form = child_phase = None
    image = nul_reset.bound_nul_capacity_reset(
        factor=factor,
        capacity_bounds=_represented_triple(source_capacity[parent]),
        form_bounds=_represented_triple(source_form[parent]),
        pressure_bounds=pressure_bounds,
        epi_min=epi_min,
        epi_max=epi_max,
    )
    successor_form = tuple(
        image.image_form if index == parent else bounds
        for index, bounds in enumerate(source_form)
    )
    successor_capacity = tuple(
        image.image_capacity if index == parent else bounds
        for index, bounds in enumerate(source_capacity)
    )
    budget = metric._Budget()
    nominal_form = _nominal(nominal_form, successor_form, "nominal_form", budget)
    nominal_phase = _nominal(nominal_phase, source_phase, "nominal_phase", budget)
    nominal_capacities = _nominal(
        nominal_capacities, successor_capacity, "nominal_capacities", budget
    )
    phases = _cells(phase_cells, size, "phase_cells", budget)
    forms = _cells(form_cells, size, "form_cells", budget)
    if len(phases) != len(forms):
        raise ValueError("phase and form cells must have the same count")
    if any(
        not lo <= value <= hi
        for nominal, first in ((nominal_form, forms[0]), (nominal_phase, phases[0]))
        for value, (lo, hi) in zip(nominal, first)
    ):
        raise ValueError("first whole-cell boxes must contain the actual nominal image")
    context = metric.prepare_metric_context(
        capacities=nominal_capacities,
        edges=edges,
        conductances=conductances,
        degrees=degrees,
    )
    family = metric.prepare_metric_family(
        context,
        form_errors=tuple(_errors(bounds) for bounds in successor_form),
        phase_errors=tuple(_errors(bounds) for bounds in source_phase),
        parent=parent,
        reference_weight=reference_weight,
        child=child,
        child_form_error=None if child_form is None else _errors(child_form),
        child_phase_error=None if child_phase is None else _errors(child_phase),
    )
    run = capacity_metric.bound_capacity_metric_cells(
        family,
        capacity_intervals=tuple(
            (bounds.lower, bounds.upper) for bounds in successor_capacity
        ),
        phase_cells=phases,
        form_cells=forms,
        edge_allowances=edge_allowances,
        durations=durations,
    )
    return NulSuccessorRun(
        image,
        source_form,
        source_phase,
        source_capacity,
        successor_form,
        source_phase,
        successor_capacity,
        nominal_form,
        nominal_phase,
        nominal_capacities,
        child_form,
        child_phase,
        family,
        run,
    )
