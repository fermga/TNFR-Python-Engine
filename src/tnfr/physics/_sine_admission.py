"""Shared primitive admission for conditional complete sine-law readers.

Model and source admission perform no theorem, trajectory or provenance
authentication. The source check is budget-neutral; the sector wrapper adds
the geometry owner's existing finite support budget. Runtime source classes
are imported only inside consumers to keep this owner independent of recovery
and avoid cycles through report adapters.
"""

from __future__ import annotations

from dataclasses import replace

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import (
    RelationalExchangeModel,
    _relational_model_coefficients,
)
from ..mathematics._rational_interval import I
from .relational_observations import _ordered


def _require_regular_sine_model(model):
    """Require the declared reference model before inspecting its coefficients."""
    if (
        not isinstance(model, RelationalExchangeModel)
        or model.phase_domain != "regular"
    ):
        raise ValueError("an explicit regular reference model is required")


def _sine_model_coefficients(model, *, positive_loss=False):
    """Admit authoritative represented coefficients without renormalization.

    The basic complete sine law admits zero loss. Analytic weighted preparation
    additionally requires positive loss; capacity admission remains with the
    consuming source or source-independent domain. No field is evaluated.
    """
    _require_regular_sine_model(model)
    loss, exchange, beta = _relational_model_coefficients(model)
    if positive_loss:
        if beta <= 0 or loss <= 0 or exchange <= 0:
            raise ValueError(
                "analytic sine domain requires positive form loss, exchange and storage"
            )
    return loss, exchange, beta


def _admit_sine_source(source):
    """Revalidate consumed primitives without trusting derived report fields.

    This consumer does not authenticate a capture or replay a forecast. It
    admits the declared complete state set and checks its model, support and
    held-capacity association. This owner imposes
    no phase-geometry or solver work cap; consumers retain their own budgets.
    """
    from .relational_sine_comparison import SineExchangeComparison
    from .relational_sine_pattern import SineRelativeForecast, SineRelativePattern

    if not isinstance(
        source, (SineExchangeComparison, SineRelativePattern, SineRelativeForecast)
    ):
        raise TypeError(
            "a SineExchangeComparison, SineRelativePattern or SineRelativeForecast is required"
        )
    pattern = source.pattern if isinstance(source, SineRelativeForecast) else source
    if not isinstance(pattern, (SineExchangeComparison, SineRelativePattern)):
        raise TypeError("forecast pattern must be a SineRelativePattern")
    if isinstance(source, SineRelativeForecast) and not isinstance(
        pattern, SineRelativePattern
    ):
        raise TypeError("forecast pattern must be a SineRelativePattern")
    model = pattern.reference_model
    _require_regular_sine_model(model)
    if pattern.law != "normalized_sine_reciprocal_exchange":
        raise ValueError("source must declare the complete normalized sine law")
    coefficients = _sine_model_coefficients(model)
    nodes = pattern.nodes
    if type(nodes) is not tuple or len(nodes) < 2:
        raise ValueError(
            "source nodes must be a complete ordered tuple of at least two nodes"
        )
    positions = {node: index for index, node in enumerate(nodes)}
    if len(positions) != len(nodes):
        raise ValueError("source nodes must be distinct")
    if type(pattern.edges) is not tuple or any(
        type(edge) is not tuple
        or len(edge) != 2
        or any(node not in positions for node in edge)
        for edge in pattern.edges
    ):
        raise ValueError("source edges must be pairs from its complete node set")
    edges = tuple(
        sorted(
            (
                min(positions[left], positions[right]),
                max(positions[left], positions[right]),
            )
            for left, right in pattern.edges
        )
    )
    if any(left == right for left, right in edges) or len(set(edges)) != len(edges):
        raise ValueError("source support must be simple and loop-free")
    size = len(nodes)
    neighbors = [[] for _ in nodes]
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    reached, pending = {0}, [0]
    while pending:
        for node in neighbors[pending.pop()]:
            if node not in reached:
                reached.add(node)
                pending.append(node)
    if len(reached) != size:
        raise ValueError("source support must be connected")
    degrees = tuple(map(len, neighbors))
    if (
        type(pattern.degrees) is not tuple
        or any(type(value) is not int for value in pattern.degrees)
        or pattern.degrees != degrees
    ):
        raise ValueError("source degrees must match its complete support")

    def values(raw, label, *, nonnegative=False):
        raw = _ordered(raw, label, limit=size + 1)
        if len(raw) != size:
            raise ValueError(f"{label} must contain one value per source node")
        result = tuple(
            exact_or_represented_real(value, f"{label}[{i}]")
            for i, value in enumerate(raw)
        )
        if nonnegative and any(value < 0 for value in result):
            raise ValueError(f"{label} must be nonnegative")
        return result

    def validate_neighbors(rows):
        if (
            type(rows) is not tuple
            or len(rows) != size
            or any(
                type(row) is not tuple
                or any(type(value) is not int for value in row)
                or tuple(sorted(row)) != tuple(sorted(expected))
                for row, expected in zip(rows, neighbors)
            )
        ):
            raise ValueError("source neighbor rows must match its complete support")

    capacity = values(pattern.capacity, "capacity", nonnegative=True)
    if isinstance(pattern, SineExchangeComparison):
        admitted = replace(
            pattern,
            epi=values(pattern.epi, "epi"),
            phase=values(pattern.phase, "phase"),
            capacity=capacity,
        )
    else:
        validate_neighbors(pattern.neighbors)
        if pattern.reference_node not in positions:
            raise ValueError("source reference node must belong to its support")
        admitted = replace(
            pattern,
            nominal_form=values(pattern.nominal_form, "nominal_form"),
            nominal_phase=values(pattern.nominal_phase, "nominal_phase"),
            capacity=capacity,
            form_error_bounds=values(
                pattern.form_error_bounds, "form_error_bounds", nonnegative=True
            ),
            phase_error_bounds=values(
                pattern.phase_error_bounds, "phase_error_bounds", nonnegative=True
            ),
        )
    if isinstance(source, SineRelativeForecast):
        from .relational_sine_forecast import SineForecast

        full = source.full_forecast
        if not isinstance(full, SineForecast):
            raise TypeError("a complete SineForecast endpoint is required")
        # Admit both declarations before comparing them: dataclass equality
        # alone treats Boolean coefficients as equal to zero or one.
        if _sine_model_coefficients(full.model) != coefficients:
            raise ValueError("full forecast model must match its pattern")
        validate_neighbors(full.neighbors)
        if full.freeze_hidden is not False:
            raise ValueError("a changed-capacity freeze_hidden forecast is unsupported")
        times = tuple(
            exact_or_represented_real(getattr(full, label), label)
            for label in ("observation_time", "validated_end_time", "end_time")
        )
        if not 0 <= times[0] <= times[1] <= times[2] or times[0] == times[2]:
            raise ValueError("forecast actual time must lie in its declared horizon")
        visible = _ordered(full.visible_capacity, "visible_capacity", limit=size)
        if len(visible) != size - 1:
            raise ValueError("forecast requires one held capacity per visible node")
        visible = tuple(
            exact_or_represented_real(value, "visible_capacity") for value in visible
        )
        if any(value < 0 for value in visible):
            raise ValueError("forecast visible capacities must be nonnegative")
        if visible != capacity[:-1]:
            raise ValueError("forecast visible capacities must match its pattern")
        initial = _ordered(full.initial_box, "initial_box", limit=2 * size + 2)
        if len(initial) != 2 * size + 1:
            raise ValueError(
                "forecast initial layout must match the complete source state"
            )
        initial_capacity = I.coerce(initial[-1])
        if initial_capacity != I(capacity[-1]):
            raise ValueError("forecast initial capacity must match its pattern")
        endpoint = _ordered(full.endpoint, "endpoint", limit=2 * size + 2)
        if len(endpoint) != 2 * size + 1:
            raise ValueError(
                "forecast endpoint layout must match the complete source state"
            )
        endpoint = tuple(I.coerce(value) for value in endpoint)
        if endpoint[-1].lo < 0:
            raise ValueError("forecast held-capacity interval must be nonnegative")
        # Capacity is a constant augmented coordinate. Its enclosure may widen
        # under interval arithmetic, but cannot discard the declared held value.
        # This association check does not authenticate or replay solver history.
        if not endpoint[-1].contains(capacity[-1]):
            raise ValueError("forecast endpoint capacity must contain its held value")
        admitted = replace(
            source,
            pattern=admitted,
            full_forecast=replace(
                full,
                visible_capacity=visible,
                initial_box=initial[:-1] + (initial_capacity,),
                endpoint=endpoint,
                observation_time=times[0],
                validated_end_time=times[1],
                end_time=times[2],
            ),
        )
    return admitted, edges


def _sector_source_admission(source):
    """Apply the sector geometry's work budget after shared source admission."""
    from .phase_cycle_geometry import _derive
    from .relational_sine_pattern import SineRelativeForecast

    admitted, edges = _admit_sine_source(source)
    pattern = (
        admitted.pattern if isinstance(admitted, SineRelativeForecast) else admitted
    )
    return admitted, _derive(pattern.nodes, edges)
