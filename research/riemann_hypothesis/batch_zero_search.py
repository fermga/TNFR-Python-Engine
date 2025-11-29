#!/usr/bin/env python3
"""TNFR batch runner for systematic RH zero scans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from search_new_zeros import ZeroSearchConfig, TNFRZeroSearcher


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load and validate a manifest describing scan segments."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list of segment objects")
    if not data:
        raise ValueError("Manifest is empty")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry #{idx + 1} is not an object")
        if "t_min" not in item or "t_max" not in item:
            raise ValueError(f"Manifest entry #{idx + 1} missing 't_min'/'t_max'")
    return data


def build_config(entry: Dict[str, Any], output_dir: Path, export_csv: bool) -> ZeroSearchConfig:
    """Construct ZeroSearchConfig for a manifest entry."""
    label = entry.get("label")
    if not label:
        label = f"range_{entry['t_min']}_{entry['t_max']}"
    base_name = entry.get("output_basename", label)
    results_path = output_dir / f"{base_name}.json"
    csv_path = (output_dir / f"{base_name}.csv") if export_csv else None
    defaults = ZeroSearchConfig()

    return ZeroSearchConfig(
        t_min=float(entry["t_min"]),
        t_max=float(entry["t_max"]),
        step=float(entry.get("step", defaults.step)),
        refinement_window=float(entry.get("refinement_window", defaults.refinement_window)),
        refinement_iterations=int(entry.get("refinement_iterations", defaults.refinement_iterations)),
        discriminant_threshold=float(entry.get("threshold", defaults.discriminant_threshold)),
        zeta_threshold=float(entry.get("zeta_threshold", defaults.zeta_threshold)),
        lambda_coeff=float(entry.get("lambda_coeff", defaults.lambda_coeff)),
        results_path=results_path,
        csv_path=csv_path,
    )


def run_config(config: ZeroSearchConfig) -> Dict[str, Any]:
    """Execute a single scan configuration and return summary data."""
    searcher = TNFRZeroSearcher(config)
    candidates, telemetry = searcher.scan_range()
    summary = searcher.summarize_candidates(candidates)
    searcher.save_results(candidates, summary, telemetry)
    searcher.save_csv(candidates)
    return {
        "results_path": str(config.results_path),
        "csv_path": str(config.csv_path) if config.csv_path else None,
        "summary": summary,
        "telemetry": telemetry.to_dict(),
        "candidate_count": len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple TNFR zero scans from a manifest")
    parser.add_argument("--manifest", required=True, help="Path to JSON manifest with scan segments")
    parser.add_argument("--output-dir", default="batch_zero_results",
                        help="Directory where result files will be written")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionally limit number of manifest entries to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned runs without executing scans")
    parser.add_argument("--disable-csv", action="store_true",
                        help="Skip CSV exports even if supported")
    parser.add_argument("--summary", default=None,
                        help="Optional path for aggregated summary JSON")

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(manifest_path)

    if args.limit is not None:
        entries = entries[:args.limit]

    summaries: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        config = build_config(entry, output_dir, export_csv=not args.disable_csv)
        label = entry.get("label", config.results_path.stem)
        print(f"[{idx}/{len(entries)}] Scanning {config.t_min} ≤ t ≤ {config.t_max} (label={label})")
        if args.dry_run:
            print("  Dry run enabled; skipping execution")
            summaries.append({
                "label": label,
                "results_path": str(config.results_path),
                "csv_path": str(config.csv_path) if config.csv_path else None,
                "dry_run": True,
            })
            continue
        result = run_config(config)
        summaries.append({"label": label, **result})

    summary_path = Path(args.summary) if args.summary else (output_dir / "batch_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"📦 Batch summary saved to {summary_path}")


if __name__ == "__main__":
    main()
