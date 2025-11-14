"""Hierarchical bifurcation visualization for THOL operator.

Provides utilities to visualize and inspect nested THOL bifurcation structures,
supporting operational fractality analysis and debugging of complex emergent
hierarchies.

TNFR Canonical Principle
-------------------------
From THOL_ENCAPSULATION_GUIDE.md:

    "THOL windows can be nested to arbitrary depth, reflecting TNFR's
    fractal structure."

This module implements ASCII tree visualization of bifurcation hierarchies,
enabling rapid inspection of multi-level emergent structures without external
visualization dependencies.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

import sys

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI

__all__ = [
    "print_bifurcation_hierarchy",
    "get_bifurcation_hierarchy_text",
    "get_hierarchy_info",
]


def print_bifurcation_hierarchy(
    G: TNFRGraph,
    node: NodeId,
    indent: int = 0,
    max_depth: int | None = None,
    stream: TextIO | None = None,
) -> None:
    """Print ASCII tree of bifurcation hierarchy.

    Recursively traverses THOL bifurcations and displays them as an ASCII
    tree structure, showing EPI values and bifurcation levels at each node.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing bifurcation structure
    node : NodeId
        Root node to start visualization from
    indent : int, optional
        Current indentation level (used internally for recursion), by default 0
    max_depth : int | None, optional
        Maximum depth to display (None = unlimited), by default None

    Parameters
    ----------
    stream : TextIO, optional
        Destination for the ASCII tree (default: sys.stdout).

    Notes
    -----
    TNFR Principle: Operational fractality (Invariant #7) enables recursive
    bifurcation. This visualization makes hierarchical structures visible for:
    - Debugging nested THOL sequences
    - Validating depth constraints
    - Analyzing emergent patterns
    - Educational demonstrations

    The format uses standard tree characters:
    - ├─ for intermediate branches
    - └─ for last branch at each level
    - │  for vertical continuation

    Examples
    --------
    >>> # Simple single-level bifurcation
    >>> print_bifurcation_hierarchy(G, node)
    Node 0 (EPI=0.82, level=0)
    ├─ Sub-EPI 1 (epi=0.21, level=1)
    └─ Sub-EPI 2 (epi=0.18, level=1)

    >>> # Multi-level nested bifurcation
    >>> print_bifurcation_hierarchy(G, node)
    Node 0 (EPI=0.82, level=0)
    ├─ Sub-EPI 1 (epi=0.21, level=1)
    │  ├─ Sub-Sub-EPI 1.1 (epi=0.05, level=2)
    │  └─ Sub-Sub-EPI 1.2 (epi=0.08, level=2)
    └─ Sub-EPI 2 (epi=0.18, level=1)

    >>> # Limit depth display
    >>> print_bifurcation_hierarchy(G, node, max_depth=1)
    Node 0 (EPI=0.82, level=0)
    ├─ Sub-EPI 1 (epi=0.21, level=1) [...]
    └─ Sub-EPI 2 (epi=0.18, level=1) [...]
    """
    output = stream or sys.stdout

    # Check depth limit
    if max_depth is not None and indent >= max_depth:
        return

    # Get node information
    node_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    node_level = G.nodes[node].get("_bifurcation_level", 0)

    # Print current node
    prefix = "  " * indent
    _write_line(
        output,
        f"{prefix}Node {node} (EPI={node_epi:.2f}, level={node_level})",
    )

    # Get sub-EPIs
    sub_epis = G.nodes[node].get("sub_epis", [])

    if not sub_epis:
        return

    # Print each sub-EPI
    for i, sub_epi in enumerate(sub_epis):
        is_last = i == len(sub_epis) - 1
        branch = "└─" if is_last else "├─"
        continuation = "  " if is_last else "│ "

        sub_level = sub_epi.get("bifurcation_level", 1)
        sub_epi_value = sub_epi.get("epi", 0.0)

        # Check if we'll hit depth limit on next level
        truncated = ""
        if max_depth is not None and indent + 1 >= max_depth:
            # Check if sub-EPI has further nesting
            sub_node_id = sub_epi.get("node_id")
            if sub_node_id and sub_node_id in G.nodes:
                sub_has_children = bool(
                    G.nodes[sub_node_id].get("sub_epis", [])
                )
                if sub_has_children:
                    truncated = " [...]"

        _write_line(
            output,
            f"{prefix}{branch} Sub-EPI {i+1} "
            f"(epi={sub_epi_value:.2f}, level={sub_level}){truncated}",
        )

        # Recurse into sub-node if it exists and we haven't hit depth limit
        sub_node_id = sub_epi.get("node_id")
        if sub_node_id and sub_node_id in G.nodes:
            if max_depth is None or indent + 1 < max_depth:
                # Print continuation line for all but last child
                if not is_last:
                    child_sub_epis = G.nodes[sub_node_id].get("sub_epis", [])
                    if child_sub_epis:
                        # Prepare indentation for child's children
                        child_indent = indent + 1
                        "  " * child_indent
                        # Print vertical continuation
                        _write_line(output, f"{prefix}{continuation}")
                        # Recurse with continuation context
                        _print_sub_hierarchy(
                            G,
                            sub_node_id,
                            child_indent,
                            parent_continuation=continuation,
                            parent_prefix=prefix,
                            max_depth=max_depth,
                            stream=output,
                        )
                else:
                    # Last child - no continuation line
                    child_sub_epis = G.nodes[sub_node_id].get("sub_epis", [])
                    if child_sub_epis:
                        child_indent = indent + 1
                        _print_sub_hierarchy(
                            G,
                            sub_node_id,
                            child_indent,
                            parent_continuation="  ",
                            parent_prefix=prefix,
                            max_depth=max_depth,
                            stream=output,
                        )


def _print_sub_hierarchy(
    G: TNFRGraph,
    node: NodeId,
    indent: int,
    parent_continuation: str,
    parent_prefix: str,
    max_depth: int | None,
    stream: TextIO,
) -> None:
    """Helper to print sub-hierarchy with proper indentation.

    Internal function handling continuation lines for nested hierarchies.
    """
    sub_epis = G.nodes[node].get("sub_epis", [])

    if not sub_epis:
        return

    if max_depth is not None and indent >= max_depth:
        return

    for i, sub_epi in enumerate(sub_epis):
        is_last = i == len(sub_epis) - 1
        branch = "└─" if is_last else "├─"
        continuation = "  " if is_last else "│ "

        sub_level = sub_epi.get("bifurcation_level", 1)
        sub_epi_value = sub_epi.get("epi", 0.0)

        # Build prefix with parent continuation
        full_prefix = parent_prefix + parent_continuation

        truncated = ""
        if max_depth is not None and indent + 1 >= max_depth:
            sub_node_id = sub_epi.get("node_id")
            if sub_node_id and sub_node_id in G.nodes:
                sub_has_children = bool(
                    G.nodes[sub_node_id].get("sub_epis", [])
                )
                if sub_has_children:
                    truncated = " [...]"

        _write_line(
            stream,
            f"{full_prefix}{branch} Sub-EPI {i+1} "
            f"(epi={sub_epi_value:.2f}, level={sub_level}){truncated}",
        )

        # Recurse if node exists
        sub_node_id = sub_epi.get("node_id")
        if sub_node_id and sub_node_id in G.nodes:
            if max_depth is None or indent + 1 < max_depth:
                child_sub_epis = G.nodes[sub_node_id].get("sub_epis", [])
                if child_sub_epis:
                    _print_sub_hierarchy(
                        G,
                        sub_node_id,
                        indent + 1,
                        parent_continuation=parent_continuation + continuation,
                        parent_prefix=parent_prefix,
                        max_depth=max_depth,
                        stream=stream,
                    )


def _write_line(stream: TextIO, message: str) -> None:
    """Write a message with newline to the provided stream."""

    stream.write(f"{message}\n")


def get_bifurcation_hierarchy_text(
    G: TNFRGraph,
    node: NodeId,
    max_depth: int | None = None,
) -> str:
    """Get bifurcation hierarchy as a formatted text string.

    Convenience wrapper around print_bifurcation_hierarchy that captures
    the ASCII tree output and returns it as a string, suitable for
    embedding in UIs, notebooks, or reports.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing bifurcation structure
    node : NodeId
        Root node to start visualization from
    max_depth : int | None, optional
        Maximum depth to display (None = unlimited), by default None

    Returns
    -------
    str
        Formatted ASCII tree representation of the bifurcation hierarchy

    Examples
    --------
    >>> hierarchy_text = get_bifurcation_hierarchy_text(G, node)
    >>> print(hierarchy_text)
    Node 0 (EPI=0.82, level=0)
    ├─ Sub-EPI 1 (epi=0.21, level=1)
    └─ Sub-EPI 2 (epi=0.18, level=1)

    >>> # Embed in a UI or notebook
    >>> from IPython.display import display, HTML
    >>> display(HTML(f"<pre>{hierarchy_text}</pre>"))  # doctest: +SKIP
    """
    buffer = StringIO()
    print_bifurcation_hierarchy(G, node, max_depth=max_depth, stream=buffer)
    return buffer.getvalue()


def get_hierarchy_info(G: TNFRGraph, node: NodeId) -> dict:
    """Get hierarchical bifurcation information for a node.

    Returns structured data about the bifurcation hierarchy, useful for
    programmatic analysis and testing.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing bifurcation structure
    node : NodeId
        Node to analyze

    Returns
    -------
    dict
        Hierarchy information containing:
        - node: Node identifier
        - epi: Current EPI value
        - bifurcation_level: Current bifurcation level
        - hierarchy_path: List of ancestor nodes
        - sub_epi_count: Number of direct sub-EPIs
        - max_depth: Maximum bifurcation depth from this node
        - total_descendants: Total number of sub-EPIs at all levels

    Examples
    --------
    >>> info = get_hierarchy_info(G, node)
    >>> info["bifurcation_level"]
    0
    >>> info["max_depth"]
    2
    >>> info["total_descendants"]
    4
    """
    from ..operators.metabolism import compute_hierarchical_depth

    node_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    node_level = G.nodes[node].get("_bifurcation_level", 0)
    hierarchy_path = G.nodes[node].get("_hierarchy_path", [])
    sub_epis = G.nodes[node].get("sub_epis", [])

    # Compute max depth
    max_depth = compute_hierarchical_depth(G, node)

    # Count total descendants recursively
    total_descendants = len(sub_epis)
    for sub_epi in sub_epis:
        sub_node_id = sub_epi.get("node_id")
        if sub_node_id and sub_node_id in G.nodes:
            # Recurse into sub-node
            sub_info = get_hierarchy_info(G, sub_node_id)
            total_descendants += sub_info["total_descendants"]

    return {
        "node": node,
        "epi": node_epi,
        "bifurcation_level": node_level,
        "hierarchy_path": hierarchy_path,
        "sub_epi_count": len(sub_epis),
        "max_depth": max_depth,
        "total_descendants": total_descendants,
    }
