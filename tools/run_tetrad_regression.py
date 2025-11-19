"""
Run a fixed set of representative TNFR telemetry/benchmark runs and optionally
compare against a baseline JSONL within tolerances.

- Produces JSONL outputs suitable for CI regression checks.
- Does not modify TNFR dynamics; purely observational.

Usage examples (PowerShell):

    # Generate a fresh reference sweep (JSONL written to --output)
    python tools/run_tetrad_regression.py \
        --output results/regression/bifurcation_smoke.jsonl \
        --nodes 24 --seeds 1 --topologies ring \
        --oz-intensity-grid 0.1,0.2 --vf-grid 0.9

    # Compare a new run to an existing baseline
    python tools/run_tetrad_regression.py \
        --output results/regression/new.jsonl \
        --baseline results/regression/bifurcation_smoke.jsonl \
        --nodes 24 --seeds 1 --topologies ring \
        --oz-intensity-grid 0.1,0.2 --vf-grid 0.9 --tolerance 1e-3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "bifurcation_landscape.py"


def _parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (
        isinstance(x, float) and (x != x)
    )


def _compare_records(
    a: Dict[str, Any], b: Dict[str, Any], tol: float
) -> Tuple[bool, List[str]]:
    diffs: List[str] = []
    ok = True
    # Compare intersection of keys; ignore extra keys
    keys = set(a.keys()) & set(b.keys())
    for k in sorted(keys):
        va, vb = a[k], b[k]
        if _is_number(va) and _is_number(vb):
            if abs(float(va) - float(vb)) > tol:
                ok = False
                diffs.append(f"num-diff {k}: {va} vs {vb} (tol {tol})")
        elif isinstance(va, str) and isinstance(vb, str):
            if va != vb:
                ok = False
                diffs.append(f"str-diff {k}: '{va}' vs '{vb}'")
        # For lists/dicts, skip deep compare in smoke harness
    return ok, diffs


def run_bifurcation_smoke(args: argparse.Namespace) -> Path:
    cmd = [
        sys.executable,
        str(BENCH),
        "--nodes",
        str(args.nodes),
        "--seeds",
        str(args.seeds),
        "--topologies",
        args.topologies,
        "--oz-intensity-grid",
        args.oz_intensity_grid,
        "--vf-grid",
        args.vf_grid,
        # Intentionally omit --quiet to capture JSONL on stdout
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bifurcation_landscape failed: {proc.stderr[:4000]}"
        )

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # best-effort JSON filter: skip non-JSON lines
            if line.startswith("{") and line.endswith("}"):
                f.write(line + "\n")
    return out_path


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TNFR tetrad regression harness")
    p.add_argument(
        "--output",
        default="results/regression/bifurcation_smoke.jsonl",
        help="Path to write JSONL results (relative to repo root)",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline JSONL to compare against",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numeric tolerance for comparisons",
    )

    # Minimal knobs for the smoke sweep
    p.add_argument("--nodes", type=int, default=24)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--topologies", default="ring")
    p.add_argument("--oz-intensity-grid", default="0.1,0.2")
    p.add_argument("--vf-grid", default="0.9,1.0")

    args = p.parse_args(argv)

    out_path = run_bifurcation_smoke(args)

    if args.baseline:
        base = ROOT / args.baseline
        if not base.exists():
            print(json.dumps({
                "status": "baseline-missing",
                "baseline": str(base)
            }))
            return 2
        new_records = _parse_jsonl(out_path)
        base_records = _parse_jsonl(base)
        if len(new_records) != len(base_records):
            print(json.dumps({
                "status": "length-mismatch",
                "new": len(new_records),
                "baseline": len(base_records)
            }))
            return 3
        all_ok = True
        all_diffs: List[str] = []
        for i, (nr, br) in enumerate(zip(new_records, base_records)):
            ok, diffs = _compare_records(nr, br, args.tolerance)
            if not ok:
                all_ok = False
                all_diffs.extend([f"record {i}: {d}" for d in diffs])
        print(json.dumps({
            "status": "ok" if all_ok else "diffs",
            "records": len(new_records),
            "tolerance": args.tolerance,
            "diffs": all_diffs[:50],
        }))
        return 0 if all_ok else 4

    # No baseline: report where results were written
    print(json.dumps({
        "status": "written",
        "output": str(out_path)
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
