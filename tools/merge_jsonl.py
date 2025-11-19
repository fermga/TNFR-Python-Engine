"""Merge multiple JSONL files into one, optionally gzipped.

Usage:
    python tools/merge_jsonl.py --out results/merged.jsonl `
        input1.jsonl input2.jsonl
    python tools/merge_jsonl.py --out results/merged.jsonl.gz --gzip `
        results/shard_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import gzip
from pathlib import Path
from typing import Iterable


def merge_files(
    out_path: Path,
    inputs: Iterable[Path],
    gzip_out: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_out:
        with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
            for p in inputs:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out_f.write(line)
    else:
        with open(out_path, "w", encoding="utf-8") as out_f:
            for p in inputs:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out_f.write(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge JSONL files")
    ap.add_argument("inputs", nargs="+", help="Input JSONL files or globs")
    ap.add_argument("--out", required=True, help="Output file path")
    ap.add_argument("--gzip", action="store_true", help="Gzip-compress output")
    args = ap.parse_args()

    # Expand globs
    input_paths = []
    for pat in args.inputs:
        matched = glob.glob(pat)
        if not matched:
            raise FileNotFoundError(f"No files match: {pat}")
        input_paths.extend(matched)

    inputs = [Path(p) for p in sorted(set(input_paths))]
    out_path = Path(args.out)
    gzip_out = bool(args.gzip or out_path.suffix == ".gz")

    merge_files(out_path, inputs, gzip_out)
    print(f"[OK] Merged {len(inputs)} files → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
