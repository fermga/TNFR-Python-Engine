"""Visualization tools for THOL cascade dynamics.

Provides plotting functions to visualize cascade propagation across networks,
temporal evolution of cascades, and collective emergence patterns.

TNFR Canonical Principle
-------------------------
From "El pulso que nos atraviesa" (TNFR Manual, §2.2.10):

    "THOL actúa como modulador central de plasticidad. Es el glifo que
    permite a la red reorganizar su topología sin intervención externa."

These visualizations make cascade dynamics observable and traceable,
enabling scientific validation and debugging of self-organization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import TNFRGraph

import matplotlib.pyplot as plt

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI
from ..utils import get_logger

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

__all__ = [
    "plot_cascade_propagation",
    "plot_cascade_timeline",
]


logger = get_logger(__name__)


def plot_cascade_propagation(G: TNFRGraph, figsize: tuple[int, int] = (12, 8)):
    """Visualize THOL cascade propagation across network.

    Creates network diagram with:
    - Node size = EPI magnitude
    - Node color = bifurcation occurred (red) or not (blue)
    - Edge thickness = coupling strength
    - Arrows = propagation direction

    Parameters
    ----------
    G : TNFRGraph
        Graph with THOL propagation history
    figsize : tuple[int, int], default (12, 8)
        Figure size in inches (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the cascade visualization

    Notes
    -----
    TNFR Principle: Cascade propagation shows how self-organization
    spreads through phase-aligned neighbors. Red nodes = bifurcation source,
    blue nodes = unaffected. Arrow thickness = propagation strength.

    Examples
    --------
    >>> # After running THOL sequence with cascades
    >>> fig = plot_cascade_propagation(G)
    >>> fig.savefig("cascade_propagation.png")
    >>> plt.show()
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for cascade visualization")

    propagations = G.graph.get("thol_propagations", [])

    fig, ax = plt.subplots(figsize=figsize)

    # Identify nodes that bifurcated (source nodes in propagations)
    bifurcated_nodes = set()
    for prop in propagations:
        bifurcated_nodes.add(prop["source_node"])

    # Node colors: red = bifurcated, lightblue = normal
    node_colors = [
        "red" if n in bifurcated_nodes else "lightblue"
        for n in G.nodes
    ]

    # Node sizes based on EPI magnitude
    node_sizes = []
    for n in G.nodes:
        epi = float(get_attr(G.nodes[n], ALIAS_EPI, 0.5))
        node_sizes.append(1000 * epi)

    # Compute layout
    pos = nx.spring_layout(G, seed=42)

    # Draw network structure
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
        alpha=0.8,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

    # Draw propagation arrows
    for prop in propagations:
        source = prop["source_node"]
        for target, strength in prop["propagations"]:
            if source in pos and target in pos:
                ax.annotate(
                    "",
                    xy=pos[target],
                    xytext=pos[source],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="red",
                        lw=2 * strength,
                        alpha=0.7,
                    ),
                )

    ax.set_title("THOL Cascade Propagation", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_cascade_timeline(G: TNFRGraph, figsize: tuple[int, int] = (10, 5)):
    """Plot temporal evolution of cascade events.

    Creates scatter plot showing:
    - X-axis: Timestamp (operator sequence step)
    - Y-axis: Number of propagation targets
    - Size: Indicates cascade magnitude

    Parameters
    ----------
    G : TNFRGraph
        Graph with THOL propagation history
    figsize : tuple[int, int], default (10, 5)
        Figure size in inches (width, height)

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object containing the timeline, or None if no cascades

    Notes
    -----
    TNFR Principle: Temporal evolution reveals cascade patterns.
    Spikes indicate strong propagation events; clusters indicate
    sustained collective reorganization.

    Examples
    --------
    >>> # After running THOL sequence with cascades
    >>> fig = plot_cascade_timeline(G)
    >>> if fig:
    ...     fig.savefig("cascade_timeline.png")
    ...     plt.show()
    """
    propagations = G.graph.get("thol_propagations", [])

    if not propagations:
        logger.info("Cascade timeline skipped: no propagation events recorded")
        return None

    timestamps = [p["timestamp"] for p in propagations]
    cascade_sizes = [len(p["propagations"]) for p in propagations]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(timestamps, cascade_sizes, s=100, alpha=0.7, color="darkred")
    ax.plot(timestamps, cascade_sizes, linestyle="--", alpha=0.5, color="gray")

    ax.set_xlabel("Timestamp (operator sequence step)", fontsize=12)
    ax.set_ylabel("Propagation Targets", fontsize=12)
    ax.set_title("THOL Cascade Evolution", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cascade_metrics_summary(
    G: TNFRGraph,
    node_metrics: dict[Any, dict[str, Any]],
    figsize: tuple[int, int] = (14, 6),
):
    """Create comprehensive cascade metrics dashboard.

    Creates multi-panel visualization showing:
    - Panel 1: Cascade depth distribution
    - Panel 2: Sub-EPI coherence over time
    - Panel 3: Metabolic activity index

    Parameters
    ----------
    G : TNFRGraph
        Graph with THOL history
    node_metrics : dict
        Dictionary mapping node IDs to their THOL metrics
    figsize : tuple[int, int], default (14, 6)
        Figure size in inches (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the dashboard

    Notes
    -----
    TNFR Principle: Complete observability requires multiple metrics.
    This dashboard provides holistic view of self-organization dynamics.

    Examples
    --------
    >>> # Collect metrics during sequence
    >>> metrics_by_node = {}
    >>> for node in G.nodes:
    ...     metrics_by_node[node] = self_organization_metrics(G, node, ...)
    >>> fig = plot_cascade_metrics_summary(G, metrics_by_node)
    >>> fig.savefig("cascade_metrics_dashboard.png")
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Cascade depth distribution
    depths = [m.get("cascade_depth", 0) for m in node_metrics.values()]
    axes[0].hist(
        depths,
        bins=range(max(depths) + 2),
        alpha=0.7,
        color="steelblue",
    )
    axes[0].set_xlabel("Cascade Depth", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(
        "Cascade Depth Distribution",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].grid(alpha=0.3)

    # Panel 2: Sub-EPI coherence
    coherences = [m.get("subepi_coherence", 0) for m in node_metrics.values()]
    node_ids = list(node_metrics.keys())
    axes[1].bar(
        range(len(node_ids)),
        coherences,
        alpha=0.7,
        color="forestgreen",
    )
    axes[1].set_xlabel("Node Index", fontsize=11)
    axes[1].set_ylabel("Coherence [0,1]", fontsize=11)
    axes[1].set_title(
        "Sub-EPI Collective Coherence",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axhline(
        0.5,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Threshold",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel 3: Metabolic activity index
    activities = [
        m.get("metabolic_activity_index", 0)
        for m in node_metrics.values()
    ]
    axes[2].bar(
        range(len(node_ids)),
        activities,
        alpha=0.7,
        color="darkorange",
    )
    axes[2].set_xlabel("Node Index", fontsize=11)
    axes[2].set_ylabel("Activity [0,1]", fontsize=11)
    axes[2].set_title(
        "Metabolic Activity Index",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig
