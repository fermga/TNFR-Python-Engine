"""Graph change tracking for intelligent cache invalidation.

This module provides hooks to track structural changes in TNFR graphs and
trigger selective cache invalidation based on which properties changed.
"""

from __future__ import annotations

from typing import Any, Optional

from .hierarchical_cache import TNFRHierarchicalCache

__all__ = ["GraphChangeTracker", "track_node_property_update"]


class GraphChangeTracker:
    """Track graph modifications for selective cache invalidation.
    
    Installs hooks into graph modification methods to automatically invalidate
    affected cache entries when structural properties change.
    
    Parameters
    ----------
    cache : TNFRHierarchicalCache
        The cache instance to invalidate.
    
    Attributes
    ----------
    topology_changes : int
        Count of topology modifications (add/remove node/edge).
    property_changes : int
        Count of node property modifications.
    
    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.caching import TNFRHierarchicalCache, GraphChangeTracker
    >>> cache = TNFRHierarchicalCache()
    >>> G = nx.Graph()
    >>> tracker = GraphChangeTracker(cache)
    >>> tracker.track_graph_changes(G)
    >>> # Now modifications to G will trigger cache invalidation
    >>> cache.set("key1", 1, CacheLevel.GRAPH_STRUCTURE, {'graph_topology'})
    >>> G.add_node("n1")  # Invalidates graph_topology cache entries
    >>> cache.get("key1", CacheLevel.GRAPH_STRUCTURE)  # Returns None
    """
    
    def __init__(self, cache: TNFRHierarchicalCache):
        self._cache = cache
        self.topology_changes = 0
        self.property_changes = 0
        self._tracked_graphs: set[int] = set()
    
    def track_graph_changes(self, graph: Any) -> None:
        """Install hooks to track changes in a graph.
        
        Wraps the graph's add_node, remove_node, add_edge, and remove_edge
        methods to trigger cache invalidation.
        
        Parameters
        ----------
        graph : GraphLike
            The graph to monitor for changes.
        
        Notes
        -----
        This uses monkey-patching to intercept graph modifications. The
        original methods are preserved and called after invalidation.
        """
        graph_id = id(graph)
        if graph_id in self._tracked_graphs:
            return  # Already tracking this graph
        
        self._tracked_graphs.add(graph_id)
        
        # Store original methods
        original_add_node = graph.add_node
        original_remove_node = graph.remove_node
        original_add_edge = graph.add_edge
        original_remove_edge = graph.remove_edge
        
        # Create tracked versions
        def tracked_add_node(node_id: Any, **attrs: Any) -> None:
            result = original_add_node(node_id, **attrs)
            self._on_topology_change()
            return result
        
        def tracked_remove_node(node_id: Any) -> None:
            result = original_remove_node(node_id)
            self._on_topology_change()
            return result
        
        def tracked_add_edge(u: Any, v: Any, **attrs: Any) -> None:
            result = original_add_edge(u, v, **attrs)
            self._on_topology_change()
            return result
        
        def tracked_remove_edge(u: Any, v: Any) -> None:
            result = original_remove_edge(u, v)
            self._on_topology_change()
            return result
        
        # Replace methods
        graph.add_node = tracked_add_node
        graph.remove_node = tracked_remove_node
        graph.add_edge = tracked_add_edge
        graph.remove_edge = tracked_remove_edge
        
        # Store reference to tracker for property changes
        if hasattr(graph, 'graph'):
            graph.graph['_tnfr_change_tracker'] = self
    
    def on_node_property_change(
        self,
        node_id: Any,
        property_name: str,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
    ) -> None:
        """Notify tracker of a node property change.
        
        Parameters
        ----------
        node_id : Any
            The node whose property changed.
        property_name : str
            Name of the property that changed (e.g., 'epi', 'vf', 'phase').
        old_value : Any, optional
            Previous value (for logging/debugging).
        new_value : Any, optional
            New value (for logging/debugging).
        
        Notes
        -----
        This should be called explicitly when node properties are modified
        outside of the graph's standard API (e.g., G.nodes[n]['epi'] = value).
        """
        # Invalidate node-specific dependency
        dep_key = f"node_{property_name}_{node_id}"
        self._cache.invalidate_by_dependency(dep_key)
        
        # Invalidate global property dependency
        global_dep = f"all_node_{property_name}"
        self._cache.invalidate_by_dependency(global_dep)
        
        # Invalidate derived metrics for this node
        if property_name in ['epi', 'vf', 'phase', 'delta_nfr']:
            self._cache.invalidate_by_dependency(f"derived_metrics_{node_id}")
        
        self.property_changes += 1
    
    def _on_topology_change(self) -> None:
        """Handle topology modifications (add/remove node/edge)."""
        # Invalidate topology-dependent caches
        self._cache.invalidate_by_dependency('graph_topology')
        self._cache.invalidate_by_dependency('node_neighbors')
        self._cache.invalidate_by_dependency('adjacency_matrix')
        
        self.topology_changes += 1
    
    def reset_counters(self) -> None:
        """Reset change counters."""
        self.topology_changes = 0
        self.property_changes = 0


def track_node_property_update(
    graph: Any,
    node_id: Any,
    property_name: str,
    new_value: Any,
) -> None:
    """Helper to track node property updates.
    
    Updates the node property and notifies the change tracker if one is
    attached to the graph.
    
    Parameters
    ----------
    graph : GraphLike
        The graph containing the node.
    node_id : Any
        The node to update.
    property_name : str
        Property name to update.
    new_value : Any
        New value for the property.
    
    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.caching import TNFRHierarchicalCache, GraphChangeTracker
    >>> from tnfr.caching.invalidation import track_node_property_update
    >>> cache = TNFRHierarchicalCache()
    >>> G = nx.Graph()
    >>> G.add_node("n1", epi=0.5)
    >>> tracker = GraphChangeTracker(cache)
    >>> tracker.track_graph_changes(G)
    >>> # Use helper to update and invalidate
    >>> track_node_property_update(G, "n1", "epi", 0.7)
    """
    # Get old value
    old_value = graph.nodes[node_id].get(property_name)
    
    # Update property
    graph.nodes[node_id][property_name] = new_value
    
    # Notify tracker if present
    if hasattr(graph, 'graph'):
        tracker = graph.graph.get('_tnfr_change_tracker')
        if isinstance(tracker, GraphChangeTracker):
            tracker.on_node_property_change(
                node_id, property_name, old_value, new_value
            )
