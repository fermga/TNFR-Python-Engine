#!/usr/bin/env python3
"""Example 93 — TNFR Structural Interface Theory (offline "try it").

Structural Interface Theory studies graph-local boundaries where neighbouring
nodes are close under the graph relation but differ sharply in phase, state, or
regime, and expresses the diagnosis as a grammar-valid operator prescription.

This example runs fully offline and deterministically, comparing supplied
spatial and signal preparations through the shared observation adapters:

  A. Static spatial — records -> k-NN graph -> injected binary phase ->
     per-node interface stress, compared against classical graph-local
     baselines on a *non-circular* target (planted boundary nodes), plus a
     grammar-valid operator prescription.

  B. Multi-channel — independently generated sinusoids in two concatenated
     blocks, with dispersed and aligned phases. The phase-locking graph is
     reconstructed from signals; no coupling law or TNFR evolution is executed.

Honest scope
------------
- The static interface stress is, by construction, related to the classical
  k-NN label-disagreement baseline (both are shown side by side; TNFR is not
  claimed to dominate the strongest global baseline on hard data).
- Local phase stress and global phase order can correlate. Different field
  formulas alone do not establish independent or additional predictive value.
- The multichannel adapter currently returns ξ_C without fit/fallback
  provenance, so the displayed value cannot identify a measured correlation
  length. Its amplitude-pressure proxy is a supplied observation map.

Run:
    python examples/10_applications/93_structural_interface_demo.py

Documentation:
    docs/STRUCTURAL_INTERFACE_THEORY.md
"""
from __future__ import annotations

import sys
from pathlib import Path

# Math symbols (∇φ, K_φ, ξ_C) are used in the output; ensure UTF-8 stdout on
# consoles that default to a narrow code page (e.g. Windows cp1252).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - best-effort console setup
        pass

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from tnfr.validation.multichannel_interface import (  # noqa: E402
    MultichannelConfig,
    evaluate_synchrony_discrimination,
    kuramoto_order_parameter,
    multichannel_window_series,
    phase_amplitude_matrices,
)
from tnfr.validation.structural_interface import (  # noqa: E402
    build_knn_graph,
    encode_phase_from_binary_state,
    evaluate_interface_scores,
    full_baseline_score_maps,
    interface_score_maps,
    score_structural_interfaces,
)

# ---------------------------------------------------------------------------
# Section A — static spatial interface scoring
# ---------------------------------------------------------------------------


def build_two_cluster_records(
    *, per_cluster: int = 40, n_boundary: int = 10, seed: int = 0
) -> tuple[list[dict[str, float]], set[int]]:
    """Two well-separated clusters plus a thin planted boundary band.

    The boundary band holds samples that sit between the clusters and carry the
    minority label of whichever side they fall on; these are the planted *true
    interface* nodes used as a non-circular review target.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []

    for _ in range(per_cluster):
        x, y = rng.normal(0.0, 0.6, size=2)
        records.append({"x": float(x), "y": float(y), "label": 0})
    for _ in range(per_cluster):
        x, y = rng.normal(0.0, 0.6, size=2)
        records.append({"x": float(x + 5.0), "y": float(y), "label": 1})

    boundary: set[int] = set()
    for _ in range(n_boundary):
        x = float(rng.uniform(2.0, 3.0))
        y = float(rng.normal(0.0, 0.6))
        # Label as the *far* cluster -> a genuine local boundary conflict.
        label = 0 if x > 2.5 else 1
        records.append({"x": x, "y": y, "label": label})
        boundary.add(len(records) - 1)

    return records, boundary


def run_static_spatial() -> None:
    print("=" * 64)
    print("A. Static spatial interface scoring (synthetic two clusters)")
    print("=" * 64)

    records, boundary = build_two_cluster_records()
    graph = build_knn_graph(
        records, ["x", "y"], k=6, node_attributes=["label", "x", "y"]
    )
    encode_phase_from_binary_state(graph, "label", positive_value=1)

    scores = score_structural_interfaces(graph, state_key="label")
    tnfr_maps = {"tnfr_phase_stress": interface_score_maps(scores)}
    baseline_maps = full_baseline_score_maps(graph, state_key="label", feature_key="x")
    score_maps = {**tnfr_maps, **baseline_maps}

    # Non-circular target: the planted boundary nodes (NOT raw disagreement).
    labels = {node: (node in boundary) for node in graph.nodes()}
    evaluation = evaluate_interface_scores(labels, score_maps)

    print(
        f"\nNodes: {evaluation['total_nodes']}  "
        f"planted interface nodes: {evaluation['review_node_count']}"
    )
    print("\nRanking power (ROC-AUC) against planted boundary target:")
    rows = sorted(evaluation["score_comparison"], key=lambda r: r["auc"], reverse=True)
    for row in rows:
        print(f"  {row['score']:<28} AUC={row['auc']:.3f}")

    top = max(scores, key=lambda s: s.tnfr_stress)
    print(
        f"\nHighest-stress node {top.node}: "
        f"|∇φ|={top.phase_gradient:.3f} |K_φ|={top.abs_curvature:.3f} "
        f"violations={top.incident_violation_count}"
    )
    print(f"Configured operator suggestion: {' -> '.join(top.prescription)}")
    print(
        "\nThese rankings concern the supplied planted target only. "
        "The operator suggestion is contextual telemetry; it is not executed "
        "and does not establish live-state or complete-word admission."
    )


# ---------------------------------------------------------------------------
# Section B — supplied sinusoidal blocks and signal-derived observations
# ---------------------------------------------------------------------------


def synthetic_regime_switch(
    *, n_channels: int = 8, block: int = 2048, fs: float = 64.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate supplied dispersed-phase and aligned-phase sinusoidal blocks."""
    rng = np.random.default_rng(seed)
    t = np.arange(block) / fs
    f0 = 6.0

    incoherent = np.array(
        [
            np.sin(
                2.0 * np.pi * (f0 + rng.normal(0.0, 0.6)) * t
                + rng.uniform(0.0, 2.0 * np.pi)
            )
            for _ in range(n_channels)
        ]
    )
    base = 2.0 * np.pi * f0 * t
    coherent = np.array(
        [np.sin(base + rng.normal(0.0, 0.05)) for _ in range(n_channels)]
    )

    signals = np.concatenate([incoherent, coherent], axis=1)
    labels = np.concatenate([np.zeros(block, dtype=int), np.ones(block, dtype=int)])
    return signals, labels


def run_multichannel() -> None:
    print("\n" + "=" * 64)
    print("B. Supplied sinusoidal blocks (dispersed phases -> aligned phases)")
    print("=" * 64)

    signals, labels = synthetic_regime_switch()
    cfg = MultichannelConfig(window=512, step=128, k_neighbours=4)

    # Per-regime global synchrony (Kuramoto R) for context.
    half = signals.shape[1] // 2
    phase_lo, _ = phase_amplitude_matrices(signals[:, :half])
    phase_hi, _ = phase_amplitude_matrices(signals[:, half:])
    r_lo = kuramoto_order_parameter(phase_lo)
    r_hi = kuramoto_order_parameter(phase_hi)
    print(f"\nKuramoto order parameter R: incoherent={r_lo:.3f}  coherent={r_hi:.3f}")

    series = multichannel_window_series(signals, config=cfg)
    data = series.as_dict()

    def _fmt(value: float) -> str:
        return "n/a" if value != value else f"{value:.3f}"  # NaN-safe

    print(
        "\nWindowed field means (first vs last supplied signal window):\n"
        f"  |∇φ|: {data['grad_phi'][0]:.3f} -> {data['grad_phi'][-1]:.3f}\n"
        f"  K_φ:  {data['k_phi'][0]:.3f} -> {data['k_phi'][-1]:.3f}\n"
        f"  ξ_C:  {_fmt(data['xi_c'][0])} -> {_fmt(data['xi_c'][-1])}"
    )

    discrimination = evaluate_synchrony_discrimination(signals, labels, config=cfg)
    print("\n" + discrimination.summary())
    print(
        "\nThe adapter does not retain ξ_C fit/fallback provenance; its value\n"
        "alone is not an identified correlation length. This finite ranking\n"
        "does not establish independent predictive information or superiority\n"
        "on measured data. Signal preparation supplies the apparent regime\n"
        "change; no autonomous synchronization or nodal dynamics is tested."
    )


def main() -> None:
    run_static_spatial()
    run_multichannel()
    print("\nSee docs/STRUCTURAL_INTERFACE_THEORY.md for the full programme.")


if __name__ == "__main__":
    main()
