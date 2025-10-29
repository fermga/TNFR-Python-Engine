"""Reusable Hypothesis strategies for TNFR property tests.

The helpers centralise NetworkX graph generation together with TNFR
initialisation steps so property-based tests can focus on assertions.

For continuous integration the recommended Hypothesis configuration uses
``deadline=None`` and ``max_examples=25`` to balance coverage and runtime.
Use :data:`PROPERTY_TEST_SETTINGS` as a decorator on property tests to keep
runs aligned with CI expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
from hypothesis import assume, settings, strategies as st
from hypothesis.strategies import SearchStrategy

from hypothesis_networkx import graph_builder

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.dynamics import set_delta_nfr_hook
from tnfr.initialization import init_node_attrs

__all__ = (
    "DEFAULT_PROPERTY_MAX_EXAMPLES",
    "PROPERTY_TEST_SETTINGS",
    "HookConfig",
    "ClusteredGraph",
    "create_nfr",
    "prepare_network",
    "homogeneous_graphs",
    "two_cluster_graphs",
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


def _bounded_attr() -> st.SearchStrategy[float]:
    return st.floats(
        min_value=-2.0,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
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
