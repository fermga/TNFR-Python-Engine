#!/usr/bin/env python3
"""Utility for generating TNFR zero-scan manifest files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class ManifestEntry:
    label: str
    t_min: float
    t_max: float
    step: float
    threshold: float
    refinement_window: float
    refinement_iterations: int
    zeta_threshold: float
    lambda_coeff: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneratorConfig:
    t_min: float
    t_max: float
    chunk_size: float
    overlap: float
    label_prefix: str
    step: float
    threshold: float
    refinement_window: float
    refinement_iterations: int
    zeta_threshold: float
    lambda_coeff: float
    output_path: Path
    dry_run: bool


def chunk_ranges(config: GeneratorConfig) -> List[ManifestEntry]:
    """Split [t_min, t_max] into overlapping chunks."""
    ranges: List[ManifestEntry] = []
    start = config.t_min
    idx = 0
    while start < config.t_max:
        end = min(start + config.chunk_size, config.t_max)
        label = f"{config.label_prefix}_{idx+1:03d}_{start:.3f}_{end:.3f}"
        ranges.append(ManifestEntry(
            label=label,
            t_min=round(start, 12),
            t_max=round(end, 12),
            step=config.step,
            threshold=config.threshold,
            refinement_window=config.refinement_window,
            refinement_iterations=config.refinement_iterations,
            zeta_threshold=config.zeta_threshold,
            lambda_coeff=config.lambda_coeff,
        ))
        idx += 1
        next_start = end - config.overlap
        if next_start <= start:
            raise ValueError("Overlap is too large; next chunk start does not advance")
        start = next_start
        if end >= config.t_max:
            break
    return ranges


def write_manifest(entries: List[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [entry.to_dict() for entry in entries]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"📝 Manifest written to {path} ({len(entries)} segments)")


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Generate manifest JSON for batch zero scans")
    parser.add_argument("--t-min", type=float, required=True, help="Start of height range")
    parser.add_argument("--t-max", type=float, required=True, help="End of height range")
    parser.add_argument("--chunk-size", type=float, default=None,
                        help="Size of each chunk (exclusive of overlap)")
    parser.add_argument("--chunk-count", type=int, default=None,
                        help="Number of chunks; alternative to chunk size")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Overlap between consecutive chunks")
    parser.add_argument("--step", type=float, default=0.5, help="Sampling step inside each chunk")
    parser.add_argument("--threshold", type=float, default=1e-5,
                        help="Discriminant threshold per chunk")
    parser.add_argument("--refinement-window", type=float, default=1.0,
                        help="Refinement window size")
    parser.add_argument("--refinement-iterations", type=int, default=6,
                        help="Refinement iterations per candidate")
    parser.add_argument("--zeta-threshold", type=float, default=1e-8,
                        help="Validation threshold for |ζ|")
    parser.add_argument("--lambda-coeff", type=float, default=0.05462277217684343,
                        help="λ coefficient to embed in manifest")
    parser.add_argument("--label-prefix", default="segment",
                        help="Prefix for generated labels")
    parser.add_argument("--output", default="manifests/generated_manifest.json",
                        help="Output file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show generated segments without writing file")

    args = parser.parse_args()
    if args.t_max <= args.t_min:
        parser.error("--t-max must be greater than --t-min")
    span = args.t_max - args.t_min
    if args.chunk_size is None and args.chunk_count is None:
        parser.error("Specify either --chunk-size or --chunk-count")
    if args.chunk_size is not None and args.chunk_count is not None:
        parser.error("Choose only one of --chunk-size or --chunk-count")
    if args.chunk_size is not None:
        chunk_size = args.chunk_size
    else:
        if args.chunk_count <= 0:
            parser.error("--chunk-count must be positive")
        chunk_size = span / args.chunk_count
    if chunk_size <= 0:
        parser.error("Chunk size must be positive")
    if args.overlap < 0:
        parser.error("--overlap cannot be negative")
    if args.overlap >= chunk_size:
        parser.error("--overlap must be smaller than chunk size")

    return GeneratorConfig(
        t_min=args.t_min,
        t_max=args.t_max,
        chunk_size=chunk_size,
        overlap=args.overlap,
        label_prefix=args.label_prefix,
        step=args.step,
        threshold=args.threshold,
        refinement_window=args.refinement_window,
        refinement_iterations=args.refinement_iterations,
        zeta_threshold=args.zeta_threshold,
        lambda_coeff=args.lambda_coeff,
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )


def main() -> None:
    config = parse_args()
    entries = chunk_ranges(config)
    if config.dry_run:
        print("🧪 Dry run; manifest not written")
        for entry in entries:
            print(json.dumps(entry.to_dict(), indent=2))
        return
    write_manifest(entries, config.output_path)


if __name__ == "__main__":
    main()
