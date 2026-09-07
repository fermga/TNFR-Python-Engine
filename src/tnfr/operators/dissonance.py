"""Dissonance (OZ) operator.

Purpose: inject controlled instability; widen dnfr; test bifurcation.
Physics: raises structural pressure; may push second derivative over tau.
Grammar: destabilizer (U2); trigger (U4a); closure-capable.
Effects: dnfr up (the direct OZ channel); epi, vf and phase unaffected.
Preconditions: sufficient epi/vf; dnfr below critical; prior stability.
Typical: OZ->IL; OZ->THOL; IL->OZ->THOL growth cycle; AL->OZ->RA.
Avoid: OZ->SHA; repeated OZ without IL/THOL containment.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import DISSONANCE
from ..constants.aliases import ALIAS_DNFR
from ..errors import TNFRValueError
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


_MISSING = object()
_PROPAGATION_EVENTS_KEY = "_oz_propagation_events"


@dataclass(frozen=True, slots=True)
class _LocalDissonancePlan:
    """Exact low-level OZ result computed on isolated graph storage."""

    dnfr_before: float
    dnfr_after: float
    magnitude: float
    realized_seed: Any = _MISSING


@dataclass(slots=True)
class _MappingRollback:
    """Restore one mapping in place while preserving original value objects."""

    mapping: MutableMapping[Any, Any]
    items: dict[Any, Any]
    sequences: tuple[tuple[list[Any] | deque[Any], tuple[Any, ...]], ...]
    mappings: tuple[tuple[MutableMapping[Any, Any], tuple[tuple[Any, Any], ...]], ...]

    @classmethod
    def capture(
        cls,
        mapping: MutableMapping[Any, Any],
        *,
        sequence_keys: tuple[str, ...] = (),
        mapping_keys: tuple[str, ...] = (),
    ) -> "_MappingRollback":
        items = dict(mapping)
        sequences: list[tuple[list[Any] | deque[Any], tuple[Any, ...]]] = []
        for key in sequence_keys:
            value = mapping.get(key, _MISSING)
            if isinstance(value, (list, deque)):
                sequences.append((value, tuple(value)))

        mappings: list[
            tuple[MutableMapping[Any, Any], tuple[tuple[Any, Any], ...]]
        ] = []
        for key in mapping_keys:
            value = mapping.get(key, _MISSING)
            if isinstance(value, MutableMapping):
                mappings.append((value, tuple(value.items())))
        return cls(mapping, items, tuple(sequences), tuple(mappings))

    def restore(self) -> None:
        """Undo added/replaced keys and in-place sequence/cache mutations."""

        for sequence, values in self.sequences:
            sequence.clear()
            sequence.extend(values)
        for mapping, items in self.mappings:
            mapping.clear()
            mapping.update(items)

        for key in tuple(self.mapping):
            if key not in self.items:
                self.mapping.pop(key, None)
        for key, value in self.items.items():
            if self.mapping.get(key, _MISSING) is not value:
                self.mapping[key] = value


@dataclass(slots=True)
class _ObjectRollback:
    """Shallow object-state rollback with in-place mutable containers."""

    target: Any
    attributes: dict[str, Any]
    sequences: tuple[tuple[list[Any] | deque[Any], tuple[Any, ...]], ...]
    mappings: tuple[tuple[MutableMapping[Any, Any], tuple[tuple[Any, Any], ...]], ...]
    nested: tuple["_ObjectRollback", ...]

    @classmethod
    def capture(
        cls, target: Any, *, nested_names: tuple[str, ...] = ()
    ) -> "_ObjectRollback | None":
        try:
            attributes = dict(vars(target))
        except TypeError:
            return None

        sequences = []
        mappings = []
        for value in attributes.values():
            if isinstance(value, (list, deque)):
                sequences.append((value, tuple(value)))
            elif isinstance(value, MutableMapping):
                mappings.append((value, tuple(value.items())))
        nested = tuple(
            snapshot
            for name in nested_names
            if (snapshot := cls.capture(getattr(target, name, None))) is not None
        )
        return cls(
            target,
            attributes,
            tuple(sequences),
            tuple(mappings),
            nested,
        )

    def restore(self) -> None:
        for snapshot in self.nested:
            snapshot.restore()
        for sequence, values in self.sequences:
            sequence.clear()
            sequence.extend(values)
        for mapping, items in self.mappings:
            mapping.clear()
            mapping.update(items)

        attributes = vars(self.target)
        for name in tuple(attributes):
            if name not in self.attributes:
                delattr(self.target, name)
        for name, value in self.attributes.items():
            if getattr(self.target, name, _MISSING) is not value:
                setattr(self.target, name, value)


def _plan_local_dissonance(G: TNFRGraph, node: Any) -> _LocalDissonancePlan:
    """Run the registered low-level OZ formula against isolated storage."""

    import networkx as nx

    from ..node import NodeNX
    from ..utils.cache import GRAPH_RUNTIME_CACHE_KEYS
    from . import GLYPH_OPERATIONS, get_glyph_factors

    probe = nx.Graph()
    probe.graph.update(G.graph)
    for key in GRAPH_RUNTIME_CACHE_KEYS:
        probe.graph.pop(key, None)
    probe.add_nodes_from(G.nodes)
    probe.nodes[node].update(G.nodes[node])

    probe_node = NodeNX(probe, node)
    dnfr_before = float(probe_node.dnfr)
    factors = get_glyph_factors(probe_node, Glyph.OZ)
    GLYPH_OPERATIONS[Glyph.OZ](probe_node, factors)
    dnfr_after = float(probe_node.dnfr)
    magnitude = abs(dnfr_after - dnfr_before)
    if not all(math.isfinite(value) for value in (dnfr_before, dnfr_after, magnitude)):
        raise TNFRValueError("OZ local plan must remain finite")

    original_seed = G.graph.get("RANDOM_SEED", _MISSING)
    probe_seed = probe.graph.get("RANDOM_SEED", _MISSING)
    realized_seed = _MISSING
    if (
        (original_seed is _MISSING or original_seed is None)
        and probe_seed is not _MISSING
        and probe_seed is not None
    ):
        realized_seed = probe_seed
    return _LocalDissonancePlan(
        dnfr_before,
        dnfr_after,
        magnitude,
        realized_seed,
    )


class Dissonance(Operator):
    """Raise dnfr; induce exploratory instability; probe bifurcation.

    Contracts: must increase dnfr; follow with IL/THOL. Avoid SHA
    immediately. See also: Coherence, SelfOrganization, Mutation.
    """

    __slots__ = ()
    name: ClassVar[str] = DISSONANCE
    glyph: ClassVar[Glyph] = Glyph.OZ

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Apply OZ with optional network propagation.

        Parameters
        ----------
        G : TNFRGraph
            Graph storing TNFR nodes
        node : Any
            Target node identifier
        **kw : Any
            Additional keyword arguments:
            - propagate_to_network: enable propagation (default True)
            - propagation_mode: phase_weighted | uniform | frequency_weighted
            - Other arguments forwarded to base Operator.__call__
        """
        from ._argument_validation import require_list_sink, strict_bool

        if "propagate_to_network" in kw:
            propagate = strict_bool(
                kw["propagate_to_network"],
                operator=self.name,
                label="propagate_to_network",
            )
        else:
            propagate = strict_bool(
                G.graph.get("OZ_ENABLE_PROPAGATION", True),
                operator=self.name,
                label="OZ_ENABLE_PROPAGATION",
            )
        if not propagate:
            super()._execute(G, node, **kw)
            return

        # The probe reuses the registered low-level OZ formula on isolated
        # storage. This gives the exact deterministic/noise result without
        # consuming live jitter progress or creating live adapter caches.
        local_plan = _plan_local_dissonance(G, node)
        if local_plan.magnitude == 0.0:
            super()._execute(G, node, **kw)
            return

        from ..dynamics.propagation import (
            _commit_dissonance_propagation,
            _prepare_dissonance_propagation,
            _rollback_dissonance_propagation,
        )

        propagation_plan = _prepare_dissonance_propagation(
            G,
            node,
            local_plan.magnitude,
            propagation_mode=kw.get("propagation_mode", "phase_weighted"),
            dnfr_overrides={node: local_plan.dnfr_after},
        )

        # Both append-only graph sinks are validated before local OZ. The
        # operator metric is still collected at its historical lifecycle point,
        # before network propagation.
        require_list_sink(
            G.graph, _PROPAGATION_EVENTS_KEY, operator=self.name
        )
        collect_metrics = kw.get("collect_metrics", False) or G.graph.get(
            "COLLECT_OPERATOR_METRICS", False
        )
        if collect_metrics:
            require_list_sink(G.graph, "operator_metrics", operator=self.name)

        node_rollback = _MappingRollback.capture(
            G.nodes[node], sequence_keys=("glyph_history",)
        )
        graph_rollback = _MappingRollback.capture(
            G.graph,
            sequence_keys=(
                "operator_metrics",
                _PROPAGATION_EVENTS_KEY,
                "recognized_coherence_patterns",
            ),
            mapping_keys=("_node_cache", "_node_cache_weak"),
        )
        monitor_rollback = _ObjectRollback.capture(
            G.graph.get("integrity_monitor"), nested_names=("_summary",)
        )
        node_list_cache_rollback = _ObjectRollback.capture(
            G.graph.get("_node_list_cache")
        )

        propagation_committed = False
        try:
            if local_plan.realized_seed is not _MISSING:
                G.graph["RANDOM_SEED"] = local_plan.realized_seed

            super()._execute(G, node, **kw)

            dnfr_after = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
            actual_magnitude = abs(dnfr_after - local_plan.dnfr_before)
            if (
                dnfr_after != local_plan.dnfr_after
                or actual_magnitude != local_plan.magnitude
            ):
                raise TNFRValueError(
                    "OZ local execution diverged from its validated plan"
                )

            affected = _commit_dissonance_propagation(propagation_plan)
            propagation_committed = True
            event = {
                "source": node,
                "magnitude": actual_magnitude,
                "affected_nodes": list(affected),
                "affected_count": len(affected),
            }
            sink = G.graph.get(_PROPAGATION_EVENTS_KEY)
            if sink is None:
                sink = []
                G.graph[_PROPAGATION_EVENTS_KEY] = sink
            sink.append(event)
        except BaseException:
            try:
                if propagation_committed:
                    _rollback_dissonance_propagation(propagation_plan)
            finally:
                if monitor_rollback is not None:
                    monitor_rollback.restore()
                if node_list_cache_rollback is not None:
                    node_list_cache_rollback.restore()
                node_rollback.restore()
                graph_rollback.restore()
            raise

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate OZ-specific preconditions."""
        from .preconditions import validate_dissonance

        validate_dissonance(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect OZ-specific metrics."""
        from .metrics import dissonance_metrics

        return dissonance_metrics(
            G,
            node,
            state_before["dnfr"],
            state_before["theta"],
        )
