#!/usr/bin/env python3
"""Unified RH zero database builder.

Aggregates known Riemann zeta zeros from the built-in TNFR catalog and any
number of external sources (CSV/JSON/TXT) into a canonical JSON/CSV dataset
with metadata and gap statistics suitable for validation pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rh_zeros_database import RHZerosDatabase


@dataclass
class SourceSpec:
    label: str
    path: Path
    fmt: str

    def describe(self) -> Dict[str, str]:
        return {"label": self.label, "path": str(self.path), "format": self.fmt}


@dataclass
class ZeroRecord:
    index: int
    height: float
    source: str

    def to_dict(self) -> Dict[str, float]:
        return {"index": self.index, "height": self.height, "source": self.source}


def parse_source_arg(argument: str) -> SourceSpec:
    """Parse --source entries of the form 'path=...,label=...,format=...'"""
    parts: Dict[str, str] = {}
    for chunk in argument.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid source spec '{argument}' (missing '=')")
        key, value = chunk.split("=", 1)
        parts[key.strip().lower()] = value.strip()

    if "path" not in parts:
        raise ValueError("Source spec must include 'path'")

    path = Path(parts["path"])
    fmt = parts.get("format")
    if fmt is None:
        fmt = path.suffix.lstrip(".").lower()
    label = parts.get("label", path.stem)
    if not label:
        raise ValueError("Source label cannot be empty")

    return SourceSpec(label=label, path=path, fmt=fmt)


def load_builtin(count: Optional[int]) -> Tuple[List[float], Dict[str, str]]:
    db = RHZerosDatabase()
    zeros = [complex_zero.imag for complex_zero in db.get_zeros_complex(count)]
    meta = {
        "label": "tnfr_builtin",
        "count": len(zeros),
        "description": "First 100 zeros from Odlyzko high-precision tables",
    }
    return zeros, meta


def load_csv(path: Path) -> List[float]:
    with path.open("r", encoding="utf-8") as handle:
        peek = handle.readline()
        handle.seek(0)
        if "," in peek:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV {path} lacks header row")
            candidate_fields = [
                "t",
                "imag",
                "imaginary",
                "height",
                "value",
                "im",
            ]
            field = next((f for f in candidate_fields if f in reader.fieldnames), None)
            if field is None:
                raise ValueError(
                    f"CSV {path} must contain one of {candidate_fields}, got {reader.fieldnames}"
                )
            return [float(row[field]) for row in reader if row.get(field)]
        else:
            handle.seek(0)
            reader = csv.reader(handle)
            values = []
            for row in reader:
                if not row:
                    continue
                values.append(float(row[0]))
            return values


def load_json(path: Path) -> List[float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [float(item) for item in payload]
    if isinstance(payload, dict):
        if "zeros" in payload:
            zeros = payload["zeros"]
            if isinstance(zeros, list):
                heights: List[float] = []
                for entry in zeros:
                    if isinstance(entry, dict):
                        for key in ("t", "imag", "imaginary", "height"):
                            if key in entry:
                                heights.append(float(entry[key]))
                                break
                    else:
                        heights.append(float(entry))
                return heights
        if "imaginary_parts" in payload:
            return [float(v) for v in payload["imaginary_parts"]]
    raise ValueError(f"JSON file {path} is not in a supported format")


def load_txt(path: Path) -> List[float]:
    values: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in stripped.split():
                values.append(float(token))
    return values


LOADERS = {
    "csv": load_csv,
    "tsv": load_csv,
    "json": load_json,
    "jsn": load_json,
    "txt": load_txt,
    "dat": load_txt,
}


def ingest_sources(
    specs: List[SourceSpec],
    include_builtin: bool,
    builtin_count: Optional[int],
) -> Tuple[List[ZeroRecord], List[Dict[str, str]]]:
    zeros: List[ZeroRecord] = []
    metadata: List[Dict[str, str]] = []

    index = 1
    if include_builtin:
        builtin_values, builtin_meta = load_builtin(builtin_count)
        metadata.append(builtin_meta)
        for height in builtin_values:
            zeros.append(ZeroRecord(index=index, height=float(height), source=builtin_meta["label"]))
            index += 1

    for spec in specs:
        loader = LOADERS.get(spec.fmt.lower())
        if loader is None:
            raise ValueError(f"Unsupported format '{spec.fmt}' for source {spec.path}")
        values = loader(spec.path)
        if not values:
            continue
        values.sort()
        for height in values:
            zeros.append(ZeroRecord(index=index, height=float(height), source=spec.label))
            index += 1
        metadata.append(spec.describe())

    zeros.sort(key=lambda record: record.height)
    for idx, record in enumerate(zeros, start=1):
        record.index = idx
    return zeros, metadata


def compute_gap_stats(heights: Iterable[float]) -> Dict[str, float]:
    heights = list(heights)
    if len(heights) < 2:
        return {"count": len(heights)}
    gaps = [b - a for a, b in zip(heights[:-1], heights[1:])]
    return {
        "count": len(heights),
        "min_gap": min(gaps),
        "max_gap": max(gaps),
        "mean_gap": statistics.mean(gaps),
        "median_gap": statistics.median(gaps),
        "std_gap": statistics.pstdev(gaps),
        "range": heights[-1] - heights[0],
        "density": len(heights) / (heights[-1] - heights[0]) if heights[-1] > heights[0] else math.inf,
    }


def build_payload(zeros: List[ZeroRecord], sources: List[Dict[str, str]]) -> Dict[str, object]:
    heights = [record.height for record in zeros]
    stats = compute_gap_stats(heights)
    payload = {
        "metadata": {
            "total_zeros": len(zeros),
            "sources": sources,
            "stats": stats,
        },
        "zeros": [record.to_dict() for record in zeros],
    }
    return payload


def write_outputs(payload: Dict[str, object], json_path: Path, csv_path: Optional[Path]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"💾 Unified zero database written to {json_path}")

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "height", "source"])
            writer.writeheader()
            writer.writerows(payload["zeros"])
        print(f"📄 CSV export written to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified RH zero database")
    parser.add_argument("--source", action="append", default=[],
                        help="External source spec 'path=...,label=...,format=csv|json|txt'")
    parser.add_argument("--include-builtin", action="store_true", default=False,
                        help="Include TNFR built-in first 100 zeros")
    parser.add_argument("--builtin-count", type=int, default=None,
                        help="Limit how many built-in zeros are included")
    parser.add_argument("--output", default="data/rh_zeros_unified.json",
                        help="Path to output JSON file")
    parser.add_argument("--csv-output", default="data/rh_zeros_unified.csv",
                        help="Optional CSV export path (omit by setting to '')")
    parser.add_argument("--max-zeros", type=int, default=None,
                        help="Optional maximum number of zeros to retain after aggregation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [parse_source_arg(entry) for entry in args.source]
    zeros, metadata = ingest_sources(specs, include_builtin=args.include_builtin,
                                     builtin_count=args.builtin_count)

    if args.max_zeros is not None:
        zeros = zeros[:args.max_zeros]

    payload = build_payload(zeros, metadata)

    csv_path = Path(args.csv_output) if args.csv_output else None
    write_outputs(payload, Path(args.output), csv_path)

    print("📊 Database summary:")
    for key, value in payload["metadata"]["stats"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
