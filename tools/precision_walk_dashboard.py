"""Aggregate precision-walk summary files into a tetrad dashboard."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    phi = data.get("fields", {}).get("phi_s", {})
    xi = data.get("xi_c", {})
    return {
        "file": str(path),
        "topology": data.get("topology"),
        "steps": data.get("steps"),
        "n": data.get("n"),
        "oz_fraction": _infer_oz_fraction(path.name),
        "phi_mean": phi.get("corr_mean"),
        "phi_std": phi.get("corr_std"),
        "xi_corr": xi.get("corr"),
        "xi_corr_d": xi.get("corr_d"),
    }


def _infer_oz_fraction(name: str) -> float | None:
    parts = name.split("_")
    for idx, token in enumerate(parts):
        if token.startswith("frac"):
            try:
                clean = token.replace("frac", "")
                for suffix in (".summary", ".json", ".jsonl"):
                    clean = clean.replace(suffix, "")
                return float(clean)
            except ValueError:
                return None
    return None


def aggregate(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_topology: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        topo = entry.get("topology") or "unknown"
        by_topology.setdefault(topo, []).append(entry)

    topo_stats: Dict[str, Dict[str, float | None]] = {}
    for topo, vals in by_topology.items():
        def _avg(key: str) -> float | None:
            series = []
            for v in vals:
                value = v.get(key)
                if isinstance(value, (int, float)) and not math.isnan(value):
                    series.append(float(value))
            return mean(series) if series else None

        topo_stats[topo] = {
            "phi_mean_avg": _avg("phi_mean"),
            "phi_std_avg": _avg("phi_std"),
            "xi_corr_avg": _avg("xi_corr"),
        }

    return {
        "runs": entries,
        "by_topology": topo_stats,
    }


def write_markdown(entries: List[Dict[str, Any]], output: Path) -> None:
    header = (
        "| File | Topology | Steps | Phi_s mean | Phi_s std | Xi_C corr |\n"
        "|------|----------|-------|------------|------------|-----------|\n"
    )
    lines = [header]
    for entry in entries:
        line = (
            f"| {Path(entry['file']).name} | {entry.get('topology')} | "
            f"{entry.get('steps')} | {_fmt(entry.get('phi_mean'))} | "
            f"{_fmt(entry.get('phi_std'))} | {_fmt(entry.get('xi_corr'))} |\n"
        )
        lines.append(line)
    output.write_text("".join(lines))


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value != value:  # NaN check
            return "NA"
        return f"{value:+.2f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build precision walk tetrad dashboard"
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="Summary JSON files",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/precision_walk_dashboard.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("benchmarks/results/precision_walk_dashboard.md"),
    )
    args = parser.parse_args()

    entries = [load_summary(path) for path in args.summaries]
    entries.sort(key=lambda e: (str(e.get("topology")), e.get("steps", 0)))
    agg = aggregate(entries)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(agg, indent=2))
    write_markdown(entries, args.md_out)


if __name__ == "__main__":
    main()
