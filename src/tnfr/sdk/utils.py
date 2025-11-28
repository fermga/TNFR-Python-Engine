"""Utility functions for TNFR SDK.

This module provides helper functions for common operations,
analysis, and visualization support.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

__all__ = [
    "compare_networks",
    "compute_network_statistics",
    "export_to_json",
    "import_from_json",
]


def compare_networks(
    networks: Dict[str, Any],
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare metrics across multiple networks.

    Parameters
    ----------
    networks : Dict[str, NetworkResults]
        Dictionary mapping network names to their results.
    metrics : List[str], optional
        List of metrics to compare. If None, compares all available.
        Options: 'coherence', 'avg_si', 'avg_delta_nfr', 'node_count'

    Returns
    -------
    Dict[str, Dict[str, float]]
        Comparison table with metrics for each network.

    Examples
    --------
    >>> from tnfr.sdk import TNFRExperimentBuilder
    >>> comparison = TNFRExperimentBuilder.compare_topologies(30, 5)
    >>> from tnfr.sdk.utils import compare_networks
    >>> stats = compare_networks(comparison)
    """
    if metrics is None:
        metrics = ["coherence", "avg_si", "avg_delta_nfr", "node_count"]

    comparison = {}
    for name, results in networks.items():
        comparison[name] = {}

        if "coherence" in metrics:
            comparison[name]["coherence"] = results.coherence

        if "avg_si" in metrics:
            si_values = list(results.sense_indices.values())
            comparison[name]["avg_si"] = sum(si_values) / len(si_values) if si_values else 0.0

        if "avg_delta_nfr" in metrics:
            dnfr_values = list(results.delta_nfr.values())
            comparison[name]["avg_delta_nfr"] = (
                sum(dnfr_values) / len(dnfr_values) if dnfr_values else 0.0
            )

        if "node_count" in metrics:
            comparison[name]["node_count"] = len(results.sense_indices)

    return comparison


def compute_network_statistics(results: Any) -> Dict[str, float]:
    """Compute extended statistics for a network.

    Parameters
    ----------
    results : NetworkResults
        Results from network measurement.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed statistics.

    Examples
    --------
    >>> from tnfr.sdk import TNFRNetwork
    >>> network = TNFRNetwork().add_nodes(10).connect_nodes(0.3)
    >>> results = network.apply_sequence("basic_activation").measure()
    >>> from tnfr.sdk.utils import compute_network_statistics
    >>> stats = compute_network_statistics(results)
    """
    si_values = list(results.sense_indices.values())
    dnfr_values = list(results.delta_nfr.values())

    stats = {
        "coherence": results.coherence,
        "node_count": len(si_values),
    }

    if si_values:
        stats["avg_si"] = sum(si_values) / len(si_values)
        stats["min_si"] = min(si_values)
        stats["max_si"] = max(si_values)

        if HAS_NUMPY:
            stats["std_si"] = float(np.std(si_values))
        else:
            mean_si = stats["avg_si"]
            variance = sum((x - mean_si) ** 2 for x in si_values) / len(si_values)
            stats["std_si"] = variance**0.5

    if dnfr_values:
        stats["avg_delta_nfr"] = sum(dnfr_values) / len(dnfr_values)
        stats["min_delta_nfr"] = min(dnfr_values)
        stats["max_delta_nfr"] = max(dnfr_values)

        if HAS_NUMPY:
            stats["std_delta_nfr"] = float(np.std(dnfr_values))
        else:
            mean_dnfr = stats["avg_delta_nfr"]
            variance = sum((x - mean_dnfr) ** 2 for x in dnfr_values) / len(dnfr_values)
            stats["std_delta_nfr"] = variance**0.5

    if results.avg_vf is not None:
        stats["avg_vf"] = results.avg_vf

    if results.avg_phase is not None:
        stats["avg_phase"] = results.avg_phase

    return stats


def export_to_json(
    network_data: Any,
    filepath: Path | str,
    indent: int = 2,
) -> None:
    """Export network data to JSON file.

    Parameters
    ----------
    network_data : TNFRNetwork or NetworkResults or dict
        Network data to export. If TNFRNetwork, calls export_to_dict().
        If NetworkResults, calls to_dict(). Otherwise uses dict directly.
    filepath : Path or str
        Path where JSON file should be saved.
    indent : int, default=2
        JSON indentation level for readability.

    Examples
    --------
    >>> from tnfr.sdk import TNFRNetwork
    >>> from tnfr.sdk.utils import export_to_json
    >>> network = TNFRNetwork().add_nodes(10).connect_nodes(0.3)
    >>> export_to_json(network, "network.json")
    """
    filepath = Path(filepath)

    # Convert to dict if needed
    if hasattr(network_data, "export_to_dict"):
        data = network_data.export_to_dict()
    elif hasattr(network_data, "to_dict"):
        data = network_data.to_dict()
    else:
        data = network_data

    # Write JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def import_from_json(filepath: Path | str) -> Dict[str, Any]:
    """Import network data from JSON file.

    Parameters
    ----------
    filepath : Path or str
        Path to JSON file to load.

    Returns
    -------
    Dict[str, Any]
        Dictionary with network data.

    Examples
    --------
    >>> from tnfr.sdk.utils import import_from_json
    >>> data = import_from_json("network.json")
    >>> print(data['metadata']['nodes'])
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def format_comparison_table(
    comparison: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> str:
    """Format network comparison as a readable table.

    Parameters
    ----------
    comparison : Dict[str, Dict[str, float]]
        Comparison data from compare_networks().
    metrics : List[str], optional
        Metrics to include in table. If None, uses all available.

    Returns
    -------
    str
        Formatted table string.

    Examples
    --------
    >>> from tnfr.sdk import TNFRExperimentBuilder
    >>> from tnfr.sdk.utils import compare_networks, format_comparison_table
    >>> results = TNFRExperimentBuilder.compare_topologies(20, 3)
    >>> comp = compare_networks(results)
    >>> print(format_comparison_table(comp))
    """
    if not comparison:
        return "No networks to compare"

    # Get all available metrics if not specified
    if metrics is None:
        metrics = list(next(iter(comparison.values())).keys())

    # Build table
    lines = []

    # Header
    header = f"{'Network':<20} " + " ".join(f"{m:>12}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for network_name, data in sorted(comparison.items()):
        values = " ".join(f"{data.get(m, 0.0):>12.3f}" for m in metrics)
        lines.append(f"{network_name:<20} {values}")

    return "\n".join(lines)


def suggest_sequence_for_goal(goal: str) -> Tuple[str, str]:
    """Suggest operator sequence for a specific goal.

    Parameters
    ----------
    goal : str
        Description of the goal. Options:
        - "activation", "start", "initialize"
        - "stabilize", "consolidate"
        - "explore", "diverge"
        - "synchronize", "align", "coordinate"
        - "mutate", "innovate", "create"

    Returns
    -------
    Tuple[str, str]
        (sequence_name, description) tuple.

    Examples
    --------
    >>> from tnfr.sdk.utils import suggest_sequence_for_goal
    >>> seq, desc = suggest_sequence_for_goal("stabilize")
    >>> print(f"Use: {seq}")
    >>> print(f"Description: {desc}")
    """
    goal_lower = goal.lower()

    suggestions = {
        "activation": (
            "basic_activation",
            "Initiates network with emission, reception, coherence, and resonance",
        ),
        "start": (
            "basic_activation",
            "Initiates network with emission, reception, coherence, and resonance",
        ),
        "initialize": (
            "basic_activation",
            "Initiates network with emission, reception, coherence, and resonance",
        ),
        "stabilize": (
            "stabilization",
            "Establishes and maintains coherent structure with recursivity",
        ),
        "consolidate": (
            "consolidation",
            "Consolidates structure with recursive coherence and silence",
        ),
        "explore": (
            "exploration",
            "Explores phase space with dissonance and transition",
        ),
        "diverge": (
            "exploration",
            "Explores phase space with dissonance and transition",
        ),
        "synchronize": (
            "network_sync",
            "Synchronizes nodes through coupling and resonance",
        ),
        "align": ("network_sync", "Synchronizes nodes through coupling and resonance"),
        "coordinate": (
            "network_sync",
            "Synchronizes nodes through coupling and resonance",
        ),
        "mutate": (
            "creative_mutation",
            "Generates variation through dissonance and mutation",
        ),
        "innovate": (
            "creative_mutation",
            "Generates variation through dissonance and mutation",
        ),
        "create": (
            "creative_mutation",
            "Generates variation through dissonance and mutation",
        ),
    }

    return suggestions.get(goal_lower, ("basic_activation", "Default: basic activation sequence"))
