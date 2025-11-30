"""TNFR–Riemann benchmark: sigma-critical regression.

This benchmark reproduces the sandbox described in
`theory/TNFR_RIEMANN_RESEARCH_NOTES.md` by scanning the toy operator
``H_TNFR`` over a grid of ``sigma`` values and estimating the point where
the smallest eigenvalue crosses zero.  The estimate should stay close to
``sigma = 0.5`` as the prime path graph grows, reflecting the conjectured
critical-line confinement.

Two artifacts are produced by default:

1. newline-delimited JSON telemetry with per-sigma eigen spectra.
2. a JSON summary reporting the estimated ``sigma_c`` per graph size.

Usage (PowerShell):

    python benchmarks/riemann_program.py \
        --graph-sizes 8,12,16 --sigma-grid 0.35,0.45,0.5,0.55,0.65 \
        --telemetry results/riemann_program/telemetry/sigma_scan.jsonl \
        --summary results/riemann_program/sigma_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
import math

import numpy as np

# Ensure local src is importable ------------------------------------------------
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from tnfr.riemann.operator import (  # type: ignore  # noqa: E402
    build_prime_path_graph,
    build_h_tnfr,
)
from tnfr.riemann.telemetry import (  # type: ignore  # noqa: E402
    RiemannTelemetryRecord,
    compute_field_aggregates,
    write_telemetry_records,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def estimate_sigma_critical(sigmas: Sequence[float], mins: Sequence[float]) -> float:
    """Linear interpolation of the zero crossing for the min eigenvalue."""

    if len(sigmas) != len(mins):
        raise ValueError("sigmas and mins must have identical length")

    for idx in range(len(sigmas) - 1):
        y1 = mins[idx]
        y2 = mins[idx + 1]
        if y1 == y2:
            continue
        if (y1 <= 0.0 and y2 >= 0.0) or (y1 >= 0.0 and y2 <= 0.0):
            x1 = sigmas[idx]
            x2 = sigmas[idx + 1]
            # Linear interpolation between (x1, y1) and (x2, y2)
            slope = (y2 - y1) / (x2 - x1)
            if slope == 0.0:
                return float(x1)
            return float(x1 - y1 / slope)

    # Fallback: choose sigma with the smallest absolute min eigenvalue
    best_idx = int(np.argmin(np.abs(mins)))
    return float(sigmas[best_idx])


def _assign_prime_fields(G: Any, sigma: float) -> None:
    """Populate phase and ΔNFR attributes for structural field telemetry."""

    for _, data in G.nodes(data=True):
        label = float(data.get("label", 1.0))
        phase = (sigma * math.log(label)) % (2 * math.pi)
        data["phase"] = phase
        data["theta"] = phase
        data["delta_nfr"] = float((sigma - 0.5) * math.log(label))


def sweep_min_eigenvalues(
    graph_size: int,
    sigmas: Sequence[float],
) -> tuple[list[RiemannTelemetryRecord], list[float]]:
    """Compute eigen spectra for each sigma and return telemetry records."""

    G = build_prime_path_graph(graph_size)
    records: list[RiemannTelemetryRecord] = []
    min_values: list[float] = []

    for sigma in sigmas:
        _assign_prime_fields(G, float(sigma))
        H, _ = build_h_tnfr(G, sigma=sigma)
        eigenvalues = np.linalg.eigvalsh(H)
        min_value = float(np.min(eigenvalues))
        min_values.append(min_value)
        field_metrics = compute_field_aggregates(G)
        records.append(
            RiemannTelemetryRecord(
                graph_size=graph_size,
                sigma=float(sigma),
                min_eigenvalue=min_value,
                eigenvalues=[float(val) for val in eigenvalues],
                **field_metrics,
            )
        )

    return records, min_values


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TNFR–Riemann sigma-critical benchmark"
    )
    parser.add_argument(
        "--graph-sizes",
        type=str,
        default="8,12,16",
        help="Comma-separated list of prime graph sizes to evaluate.",
    )
    parser.add_argument(
        "--sigma-grid",
        type=str,
        default="0.35,0.45,0.5,0.55,0.65",
        help="Comma-separated sigma grid used for interpolation.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.08,
        help="Maximum allowed |sigma_c - 0.5| deviation before flagging.",
    )
    parser.add_argument(
        "--telemetry",
        type=str,
        default="results/riemann_program/telemetry/sigma_scan.jsonl",
        help="Output path for newline-delimited telemetry records.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="results/riemann_program/sigma_summary.json",
        help="Output path for JSON summary report.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    graph_sizes = [int(x) for x in args.graph_sizes.split(',') if x.strip()]
    sigma_grid = [float(x) for x in args.sigma_grid.split(',') if x.strip()]

    telemetry_records: list[RiemannTelemetryRecord] = []
    summary: list[dict[str, float | int | bool]] = []

    for graph_size in graph_sizes:
        records, min_values = sweep_min_eigenvalues(graph_size, sigma_grid)
        sigma_c = estimate_sigma_critical(sigma_grid, min_values)
        passes = abs(sigma_c - 0.5) <= args.tolerance

        for rec in records:
            if rec.sigma == sigma_c:
                rec.estimate_sigma_critical = sigma_c
        telemetry_records.extend(records)
        telemetry_records.append(
            RiemannTelemetryRecord(
                graph_size=graph_size,
                sigma=sigma_c,
                min_eigenvalue=min(min_values, key=abs),
                eigenvalues=[],
                estimate_sigma_critical=sigma_c,
                notes="summary",
            )
        )

        summary.append(
            {
                "graph_size": graph_size,
                "sigma_critical": sigma_c,
                "distance_from_half": abs(sigma_c - 0.5),
                "tolerance": args.tolerance,
                "passes": passes,
            }
        )

    telemetry_path = write_telemetry_records(telemetry_records, args.telemetry)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    failing = [item for item in summary if not item["passes"]]
    if failing:
        print(
            "[TNFR Riemann] sigma-critical benchmark FAILED for graph sizes:",
            ", ".join(str(item["graph_size"]) for item in failing),
        )
        print(f"Telemetry: {telemetry_path}")
        print(f"Summary:   {summary_path}")
        return 1

    print("[TNFR Riemann] sigma-critical benchmark passed for all graph sizes.")
    print(f"Telemetry: {telemetry_path}")
    print(f"Summary:   {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(main(sys.argv[1:]))
