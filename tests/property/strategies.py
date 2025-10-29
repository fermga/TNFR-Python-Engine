"""Reusable Hypothesis strategies for TNFR property tests.

The helpers centralise NetworkX graph generation together with TNFR
initialisation steps so property-based tests can focus on assertions.

For continuous integration the recommended Hypothesis configuration uses
``deadline=None`` and ``max_examples=25`` to balance coverage and runtime.
Use :data:`PROPERTY_TEST_SETTINGS` as a decorator on property tests to keep
runs aligned with CI expectations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import math

import networkx as nx
from hypothesis import assume, settings, strategies as st
from hypothesis.strategies import SearchStrategy

from hypothesis_networkx import graph_builder

from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_KEY,
    VF_PRIMARY,
    inject_defaults,
)
from tnfr.dynamics import set_delta_nfr_hook
from tnfr.initialization import init_node_attrs

__all__ = (
    "DEFAULT_PROPERTY_MAX_EXAMPLES",
    "PROPERTY_TEST_SETTINGS",
    "HookConfig",
    "ClusteredGraph",
    "PhaseGraph",
    "PhaseNeighbourhood",
    "PhaseBulkScenario",
    "create_nfr",
    "nested_structured_mappings",
    "prepare_network",
    "homogeneous_graphs",
    "phase_graphs",
    "two_cluster_graphs",
    "phase_neighbourhoods",
    "phase_bulk_scenarios",
)

# Hypothesis guidance for CI runs.
DEFAULT_PROPERTY_MAX_EXAMPLES = 25
PROPERTY_TEST_SETTINGS = settings(deadline=None, max_examples=DEFAULT_PROPERTY_MAX_EXAMPLES)

DeltaHook = Callable[..., None]


@dataclass(frozen=True)
class HookConfig:
    """Description for :func:`set_delta_nfr_hook` selections."""

    func: DeltaHook
    name: str | None = None
    note: str | None = None

    def install(self, graph: nx.Graph) -> None:
        """Register ``func`` on ``graph`` via :func:`set_delta_nfr_hook`."""

        set_delta_nfr_hook(graph, self.func, name=self.name, note=self.note)


def _zero_delta(G: nx.Graph, *_args: Any, **_kwargs: Any) -> None:
    """Default hook that clears ΔNFR contributions for every node."""

    for node in G.nodes:
        G.nodes[node][DNFR_PRIMARY] = 0.0


DEFAULT_HOOKS: tuple[HookConfig, ...] = (
    HookConfig(
        func=_zero_delta,
        name="zero_delta",
        note="ΔNFR is reset to zero for deterministic property runs.",
    ),
)


def _normalise_override(value: SearchStrategy[Any] | Any) -> SearchStrategy[Any]:
    """Turn literal values into Hypothesis strategies when required."""

    return value if isinstance(value, SearchStrategy) else st.just(value)


_KEY_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


def _homogeneous_lists(strategy: SearchStrategy[Any]) -> SearchStrategy[list[Any]]:
    return st.lists(strategy, max_size=4)


def _list_variants(base: SearchStrategy[list[Any]]) -> SearchStrategy[Any]:
    return st.one_of(
        base,
        st.builds(tuple, base),
        st.builds(deque, base),
        st.builds(lambda values: set(values), base),
    )


def _scalar_sequences() -> SearchStrategy[Any]:
    integers = _homogeneous_lists(st.integers(min_value=-1_000, max_value=1_000))
    floats = _homogeneous_lists(
        st.floats(
            min_value=-1_000.0,
            max_value=1_000.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    text = _homogeneous_lists(
        st.text(alphabet=_KEY_ALPHABET, min_size=0, max_size=8)
    )
    return _list_variants(st.one_of(integers, floats, text))


def nested_structured_mappings() -> SearchStrategy[dict[str, Any]]:
    """Return nested dictionaries using TNFR-compatible scalar collections."""

    keys = st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=8)
    scalars = st.one_of(
        st.integers(min_value=-1_000, max_value=1_000),
        st.floats(
            min_value=-1_000.0,
            max_value=1_000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.text(alphabet=_KEY_ALPHABET, min_size=0, max_size=12),
    )
    base_values = st.one_of(scalars, _scalar_sequences())

    def extend(children: SearchStrategy[Any]) -> SearchStrategy[dict[str, Any]]:
        return st.dictionaries(
            keys,
            st.one_of(base_values, children),
            max_size=4,
        )

    return st.recursive(
        st.dictionaries(keys, base_values, max_size=4),
        extend,
        max_leaves=10,
    )


def create_nfr(
    *,
    include_weight: bool = True,
    weight: SearchStrategy[float] | None = None,
    overrides: Mapping[str, SearchStrategy[Any] | Any] | None = None,
) -> SearchStrategy[dict[str, Any]]:
    """Return a strategy yielding node attribute dictionaries.

    Parameters
    ----------
    include_weight:
        When ``True`` each node carries a ``weight`` attribute. Disable it
        when weight-less nodes are preferred.
    weight:
        Override the default weight strategy (a bounded positive float).
    overrides:
        Map of additional attributes to attach to every node. Values can be
        fixed literals or Hypothesis strategies.
    """

    fields: dict[str, SearchStrategy[Any]] = {}
    if include_weight:
        fields["weight"] = weight or st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )

    if overrides:
        for key, value in overrides.items():
            fields[key] = _normalise_override(value)

    if not fields:
        return st.just({})
    return st.fixed_dictionaries(fields)


def prepare_network(
    *,
    min_nodes: int = 2,
    max_nodes: int = 8,
    min_edges: int = 0,
    max_edges: int | None = None,
    connected: bool = True,
    graph_type: type[nx.Graph] = nx.Graph,
    self_loops: bool = False,
    node_factory: SearchStrategy[Mapping[str, Any]] | None = None,
    edge_factory: SearchStrategy[Mapping[str, Any]] | None = None,
    node_keys: SearchStrategy[Any] | None = None,
    graph_overrides: Mapping[str, Any] | None = None,
    hook_options: Sequence[HookConfig | None] | None = None,
    require_hook: bool = False,
    init_nodes: bool = True,
    override_defaults: bool = False,
) -> SearchStrategy[nx.Graph]:
    """Return a strategy yielding graphs prepared for TNFR property tests.

    The generated graphs already contain TNFR defaults, optionally initialised
    node attributes, and can register a ΔNFR hook chosen from ``hook_options``.
    Pass ``require_hook=True`` to force installing one of the candidates.
    """

    node_data_strategy = node_factory or create_nfr()
    edge_data_strategy = edge_factory or st.fixed_dictionaries({})

    builder = graph_builder(
        node_data=node_data_strategy,
        edge_data=edge_data_strategy,
        node_keys=node_keys,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        min_edges=min_edges,
        max_edges=max_edges,
        graph_type=graph_type,
        self_loops=self_loops,
        connected=connected,
    )

    if hook_options is None:
        available_hooks: tuple[HookConfig, ...] = DEFAULT_HOOKS
        allow_none = True
    else:
        allow_none = any(option is None for option in hook_options)
        available_hooks = tuple(option for option in hook_options if option is not None)

    if require_hook and not available_hooks:
        msg = "require_hook=True but no hook options were provided"
        raise ValueError(msg)

    hook_strategies: list[SearchStrategy[HookConfig | None]] = []
    if not require_hook and (allow_none or not available_hooks):
        hook_strategies.append(st.just(None))
    if available_hooks:
        hook_strategies.append(st.sampled_from(available_hooks))

    if not hook_strategies:
        hook_strategy: SearchStrategy[HookConfig | None] = st.just(None)
    elif len(hook_strategies) == 1:
        hook_strategy = hook_strategies[0]
    else:
        hook_strategy = st.one_of(*hook_strategies)

    overrides_strategy = st.just(dict(graph_overrides or {}))

    return st.builds(
        _finalise_graph,
        builder,
        overrides_strategy,
        hook_strategy,
        st.just(init_nodes),
        st.just(override_defaults),
    )


def _finalise_graph(
    graph: nx.Graph,
    overrides: Mapping[str, Any],
    hook_cfg: HookConfig | None,
    init_nodes: bool,
    override_defaults: bool,
) -> nx.Graph:
    """Apply TNFR defaults, optional overrides and hook registration."""

    inject_defaults(graph, override=override_defaults)
    if overrides:
        for key, value in overrides.items():
            graph.graph[key] = value

    if init_nodes:
        init_node_attrs(graph, override=True)

    if hook_cfg is not None:
        hook_cfg.install(graph)

    return graph


@dataclass(frozen=True)
class ClusteredGraph:
    """Description of a graph partitioned into two clusters."""

    graph: nx.Graph
    clusters: tuple[tuple[Any, ...], tuple[Any, ...]]


@dataclass(frozen=True)
class PhaseGraph:
    """Container combining a prepared graph with its phase baseline."""

    graph: nx.Graph
    base_phases: Mapping[Any, float]
    offset: float


@dataclass(frozen=True)
class PhaseNeighbourhood:
    """Container with finite neighbour angles and derived trig mappings."""

    neighbours: tuple[Any, ...]
    angles: Mapping[Any, float]
    weights: Mapping[Any, float]
    cos_map: Mapping[Any, float]
    sin_map: Mapping[Any, float]
    fallback: float


@dataclass(frozen=True)
class PhaseBulkScenario:
    """Description of graph-wide phase information for vectorised metrics."""

    edge_src: tuple[int, ...]
    edge_dst: tuple[int, ...]
    cos_values: tuple[float, ...]
    sin_values: tuple[float, ...]
    theta_values: tuple[float, ...]
    node_count: int
    neighbour_counts: tuple[int, ...]


def _bounded_attr() -> st.SearchStrategy[float]:
    return st.floats(
        min_value=-2.0,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
    )


def _finite_angle() -> st.SearchStrategy[float]:
    return st.floats(
        min_value=-4.0 * math.pi,
        max_value=4.0 * math.pi,
        allow_nan=False,
        allow_infinity=False,
    )


def _finite_weight() -> st.SearchStrategy[float]:
    return st.floats(
        min_value=0.0,
        max_value=5.0,
        allow_nan=False,
        allow_infinity=False,
    )


@st.composite
def phase_neighbourhoods(
    draw,
    *,
    min_neighbours: int = 0,
    max_neighbours: int = 6,
    fallback: st.SearchStrategy[float] | None = None,
) -> PhaseNeighbourhood:
    """Return finite neighbour lists together with cos/sin lookup tables."""

    count = draw(
        st.integers(min_value=min_neighbours, max_value=max_neighbours)
    )
    labels = draw(
        st.lists(
            st.text(alphabet=_KEY_ALPHABET, min_size=1, max_size=6),
            min_size=count,
            max_size=count,
            unique=True,
        )
    )

    angles = draw(
        st.lists(_finite_angle(), min_size=count, max_size=count)
    )
    weights = draw(
        st.lists(_finite_weight(), min_size=count, max_size=count)
    )
    fallback_strategy = fallback or _finite_angle()
    fallback_value = draw(fallback_strategy)

    angle_map = {label: angle for label, angle in zip(labels, angles)}
    weight_map = {label: weight for label, weight in zip(labels, weights)}
    cos_map = {label: math.cos(angle) for label, angle in angle_map.items()}
    sin_map = {label: math.sin(angle) for label, angle in angle_map.items()}

    return PhaseNeighbourhood(
        neighbours=tuple(labels),
        angles=angle_map,
        weights=weight_map,
        cos_map=cos_map,
        sin_map=sin_map,
        fallback=fallback_value,
    )


@st.composite
def phase_bulk_scenarios(
    draw,
    *,
    min_nodes: int = 0,
    max_nodes: int = 6,
    max_neighbours_per_node: int = 6,
) -> PhaseBulkScenario:
    """Return graph-scale phase descriptions for vectorised phase metrics."""

    node_count = draw(
        st.integers(min_value=min_nodes, max_value=max_nodes)
    )
    if node_count == 0:
        return PhaseBulkScenario(
            edge_src=(),
            edge_dst=(),
            cos_values=(),
            sin_values=(),
            theta_values=(),
            node_count=0,
            neighbour_counts=(),
        )

    theta_values = draw(
        st.lists(_finite_angle(), min_size=node_count, max_size=node_count)
    )
    cos_values = [math.cos(theta) for theta in theta_values]
    sin_values = [math.sin(theta) for theta in theta_values]

    edge_src: list[int] = []
    edge_dst: list[int] = []
    neighbour_counts = [0] * node_count

    index_strategy = st.integers(min_value=0, max_value=node_count - 1)

    for dst in range(node_count):
        neighbour_sample = draw(
            st.lists(
                index_strategy,
                min_size=0,
                max_size=max_neighbours_per_node,
            )
        )
        for src in neighbour_sample:
            edge_src.append(src)
            edge_dst.append(dst)
            neighbour_counts[dst] += 1

    return PhaseBulkScenario(
        edge_src=tuple(edge_src),
        edge_dst=tuple(edge_dst),
        cos_values=tuple(cos_values),
        sin_values=tuple(sin_values),
        theta_values=tuple(theta_values),
        node_count=node_count,
        neighbour_counts=tuple(neighbour_counts),
    )


@st.composite
def homogeneous_graphs(
    draw,
    *,
    min_nodes: int = 2,
    max_nodes: int = 8,
) -> nx.Graph:
    """Graphs whose nodes share the same EPI and νf values."""

    graph = draw(
        prepare_network(
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            connected=True,
        )
    )
    epi_value = draw(_bounded_attr())
    vf_value = draw(_bounded_attr())
    for _node, data in graph.nodes(data=True):
        data[EPI_PRIMARY] = epi_value
        data[VF_PRIMARY] = vf_value
        data[DNFR_PRIMARY] = 0.0
    return graph


@st.composite
def phase_graphs(
    draw,
    *,
    min_nodes: int = 2,
    max_nodes: int = 8,
    phase_strategy: SearchStrategy[float] | None = None,
    offset: SearchStrategy[float] | None = None,
    hook_options: Sequence[HookConfig | None] | None = None,
    require_hook: bool = False,
) -> PhaseGraph:
    """Return graphs with explicit phase assignments and a shared offset."""

    graph = draw(
        prepare_network(
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            connected=True,
            hook_options=hook_options,
            require_hook=require_hook,
        )
    )
    offset_value = draw(offset) if offset is not None else 0.0
    base_phase_strategy = phase_strategy or _finite_angle()
    base_phases: dict[Any, float] = {}

    for node, data in graph.nodes(data=True):
        base_value = draw(base_phase_strategy)
        base_phases[node] = base_value
        data[THETA_KEY] = base_value + offset_value
        data[DNFR_PRIMARY] = 0.0

    return PhaseGraph(
        graph=graph,
        base_phases=MappingProxyType(base_phases),
        offset=offset_value,
    )


@st.composite
def two_cluster_graphs(
    draw,
    *,
    min_cluster: int = 2,
    max_cluster: int = 4,
) -> ClusteredGraph:
    """Return a complete bi-cluster graph with contrasting EPI/νf."""

    size_a = draw(st.integers(min_value=min_cluster, max_value=max_cluster))
    size_b = size_a

    graph = nx.complete_bipartite_graph(size_a, size_b)
    inject_defaults(graph, override=False)
    init_node_attrs(graph, override=True)

    left_nodes = tuple(range(size_a))
    right_nodes = tuple(range(size_a, size_a + size_b))

    epi_left = draw(_bounded_attr())
    epi_right = draw(_bounded_attr())
    vf_left = draw(_bounded_attr())
    vf_right = draw(_bounded_attr())

    # Ensure the clusters are distinguishable to produce a gradient.
    assume(abs(epi_left - epi_right) + abs(vf_left - vf_right) > 1e-6)

    for node in left_nodes:
        data = graph.nodes[node]
        data[EPI_PRIMARY] = epi_left
        data[VF_PRIMARY] = vf_left
        data[DNFR_PRIMARY] = 0.0
    for node in right_nodes:
        data = graph.nodes[node]
        data[EPI_PRIMARY] = epi_right
        data[VF_PRIMARY] = vf_right
        data[DNFR_PRIMARY] = 0.0

    return ClusteredGraph(graph=graph, clusters=(left_nodes, right_nodes))
