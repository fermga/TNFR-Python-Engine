#!/usr/bin/env python3
"""Compare shared field readouts on reproducibly supplied graph snapshots.

No operators or dynamics are executed. Phases, capacities, form and pressure
are initialization inputs, not a generated physical state. Derived fields and
sample correlations describe those inputs; temporal conservation is unavailable.

Run with ``--plot output.png`` for an optional static comparison figure.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from tnfr.constants import inject_defaults
from tnfr.physics.fields import compute_unified_telemetry


def build_snapshots(seed: int = 17) -> dict[str, nx.Graph]:
    """Initialize three declared supports; no evolution or parameter fitting."""
    generator = random.Random(seed)
    graphs = {
        "Path (3 nodes)": nx.path_graph(3),
        "Cycle (8 nodes)": nx.cycle_graph(8),
        "Barbell (7 nodes)": nx.barbell_graph(3, 1),
    }
    for graph in graphs.values():
        inject_defaults(graph)
        graph.graph["RANDOM_SEED"] = seed
        for node in graph:
            graph.nodes[node].update(
                EPI=generator.uniform(-0.5, 0.5),
                nu_f=generator.uniform(0.5, 1.5),
                theta=generator.uniform(-0.6, 0.6),
                delta_nfr=generator.uniform(-0.3, 0.3),
            )
    return graphs


def analyze_snapshots(seed: int = 17) -> dict[str, dict[str, Any]]:
    """Use the shared composite owner once per supplied snapshot."""
    return {
        name: compute_unified_telemetry(graph)
        for name, graph in build_snapshots(seed).items()
    }


def print_snapshot_report(reports: dict[str, dict[str, Any]]) -> None:
    """Describe snapshots without interpreting them as a temporal experiment."""
    for name, report in reports.items():
        complex_field = report["complex_field"]
        tensors = report["tensor_invariants"]
        print(f"\n{name}")
        print(f"  K_phi / J_phi sample correlation: {complex_field['correlation']:.6f}")
        print(f"  Mean Psi magnitude: {np.mean(complex_field['psi_magnitude']):.6f}")
        print(
            f"  Sum of raw quadratic densities: {np.sum(tensors['energy_density']):.6f}"
        )
        print("  Temporal conservation: unavailable (single snapshot)")


def create_visualization(reports: dict[str, dict[str, Any]], output_path: Path) -> Path:
    """Plot descriptive snapshot statistics, without conservation thresholds."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    names = list(reports)
    correlations = [reports[name]["complex_field"]["correlation"] for name in names]
    energies = [
        float(np.sum(reports[name]["tensor_invariants"]["energy_density"]))
        for name in names
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    figure.suptitle("Supplied graph snapshots: descriptive field readouts")
    axes[0].bar(names, correlations)
    axes[0].set_title("K_phi / J_phi sample correlation")
    axes[1].bar(names, energies)
    axes[1].set_title("Sum of raw quadratic densities")
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=20)
    axes[2].set_title("Temporal conservation: unavailable")
    axes[2].text(
        0.5,
        0.5,
        "No paired snapshots or clock were measured.\n"
        "No conservation or stability claim follows.",
        ha="center",
        va="center",
        transform=axes[2].transAxes,
    )
    axes[2].set_axis_off()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args(argv)
    print(f"Supplied-state field comparison; seed={args.seed}; no evolution")
    reports = analyze_snapshots(args.seed)
    print_snapshot_report(reports)
    if args.plot is not None:
        print(f"Figure: {create_visualization(reports, args.plot)}")


if __name__ == "__main__":
    main()
