"""Plot summaries for Integrated Force Regime Study and Methods Battery.

This script reads:
- results/integrated_force_study_summary.json
- results/field_methods_battery_summary.json

and produces a few quick visualizations under results/plots/.

It gracefully degrades if matplotlib is not installed.
"""
# flake8: noqa
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
INTEGRATED_SUMMARY = RESULTS_DIR / "integrated_force_study_summary.json"
METHODS_SUMMARY = RESULTS_DIR / "field_methods_battery_summary.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required summary: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as e:  # pragma: no cover - optional dependency
        print("matplotlib not available (plots will be skipped):", e)
        return None


def plot_regime_counts(integrated: Dict[str, Any], plt) -> None:
    # For each topology, gather counts of regimes
    topologies = [k for k in integrated.keys() if k not in ("unknown",)]
    regimes = sorted({r for topo in topologies for r in integrated[topo].get("regimes", {}).keys()})

    data = {r: [integrated[topo].get("regimes", {}).get(r, 0) for topo in topologies] for r in regimes}

    import numpy as np
    x = np.arange(len(topologies))
    width = 0.8 / max(1, len(regimes))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(regimes):
        ax.bar(x + i * width, data[r], width=width, label=r)
    ax.set_xticks(x + width * (len(regimes) - 1) / 2)
    ax.set_xticklabels(topologies, rotation=30, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Force Regime Counts per Topology")
    ax.legend(frameon=False)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "regime_counts_per_topology.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Wrote:", out)


def plot_composite_global(integrated: Dict[str, Any], plt) -> None:
    topologies = [k for k in integrated.keys() if k not in ("unknown",)]
    means = []
    stds = []
    for topo in topologies:
        comp = integrated[topo].get("composite_field_metrics", {})
        sg = comp.get("S_global", {})
        means.append(float(sg.get("mean", 0.0)))
        stds.append(float(sg.get("std", 0.0)))

    import numpy as np
    x = np.arange(len(topologies))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=30, ha="right")
    ax.set_ylabel("S_global (mean ± std)")
    ax.set_title("Composite Global Stress by Topology")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "composite_global_stress.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Wrote:", out)


def plot_grad_vs_kphi_corr(integrated: Dict[str, Any], plt) -> None:
    topologies = [k for k in integrated.keys() if k not in ("unknown",)]
    vals = [float(integrated[topo].get("correlations", {}).get("grad_phi__abs_k_phi", 0.0)) for topo in topologies]

    import numpy as np
    x = np.arange(len(topologies))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=30, ha="right")
    ax.set_ylabel("corr(grad_phi, |K_phi|)")
    ax.set_title("Gradient–Curvature Correlation by Topology")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "grad_vs_kphi_correlation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Wrote:", out)


def plot_methods_battery(methods: Dict[str, Any], plt) -> None:
    # Visualize mean corr/MAD/RMSE for gradient and curvature across topologies
    topologies = list(methods.keys())

    grad_corr = [float(methods[t].get("grad_corr", {}).get("mean", 0.0)) for t in topologies]
    grad_mad = [float(methods[t].get("grad_mad", {}).get("mean", 0.0)) for t in topologies]
    grad_rmse = [float(methods[t].get("grad_rmse", {}).get("mean", 0.0)) for t in topologies]

    curv_corr = [float(methods[t].get("curv_corr", {}).get("mean", 0.0)) for t in topologies]
    curv_mad = [float(methods[t].get("curv_mad", {}).get("mean", 0.0)) for t in topologies]
    curv_rmse = [float(methods[t].get("curv_rmse", {}).get("mean", 0.0)) for t in topologies]

    import numpy as np
    x = np.arange(len(topologies))
    width = 0.25

    # Gradient metrics
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, grad_corr, width=width, label="grad corr")
    ax.bar(x, grad_mad, width=width, label="grad MAD")
    ax.bar(x + width, grad_rmse, width=width, label="grad RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=30, ha="right")
    ax.set_title("Methods Battery (Gradient): corr / MAD / RMSE")
    ax.legend(frameon=False)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "methods_battery_gradient.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Wrote:", out)

    # Curvature metrics
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, curv_corr, width=width, label="curv corr")
    ax.bar(x, curv_mad, width=width, label="curv MAD")
    ax.bar(x + width, curv_rmse, width=width, label="curv RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=30, ha="right")
    ax.set_title("Methods Battery (Curvature): corr / MAD / RMSE")
    ax.legend(frameon=False)
    out = PLOTS_DIR / "methods_battery_curvature.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Wrote:", out)


def main() -> int:
    integrated = _load_json(INTEGRATED_SUMMARY)
    methods = _load_json(METHODS_SUMMARY)
    plt = _try_import_matplotlib()
    if plt is None:
        print("Skipping plots; install matplotlib to enable visualization.")
        return 0

    plot_regime_counts(integrated, plt)
    plot_composite_global(integrated, plt)
    plot_grad_vs_kphi_corr(integrated, plt)
    plot_methods_battery(methods, plt)

    print("All plots written to:", PLOTS_DIR)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
