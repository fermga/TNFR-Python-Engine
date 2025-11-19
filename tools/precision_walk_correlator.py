"""Compute correlation summaries for precision walk JSONL outputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    mean_a = fmean(a)
    mean_b = fmean(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0.0 or den_b == 0.0:
        return float("nan")
    return num / (den_a * den_b)


def diffs(seq: List[float]) -> List[float]:
    return [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]


def _paired(
    base: Iterable[float], other: Iterable[float]
) -> Tuple[List[float], List[float]]:
    a_vals: List[float] = []
    b_vals: List[float] = []
    for a, b in zip(base, other):
        fa = float(a)
        fb = float(b)
        if math.isnan(fa) or math.isnan(fb):
            continue
        a_vals.append(fa)
        b_vals.append(fb)
    return a_vals, b_vals


def correlate(
    base: Iterable[float], other: Iterable[float]
) -> Optional[float]:
    a_vals, b_vals = _paired(base, other)
    if len(a_vals) < 2:
        return None
    value = pearson(a_vals, b_vals)
    return value if not math.isnan(value) else None


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    integral = [float(r["metrics"]["integral_vf_dnfr"]) for r in records]
    vf_mean = [float(r["metrics"]["vf_dnfr_mean"]) for r in records]
    deltas = diffs(integral)
    summary: Dict[str, Any] = {
        "steps": len(records),
        "topology": records[0].get("topology"),
        "n": records[0].get("n"),
        "dt": records[0].get("dt"),
        "seed": records[0].get("seed"),
        "integral_vs_vf_mean": correlate(integral, vf_mean),
        "fields": {},
        "stratification": {},
    }

    field_specs = {
        "phase_curv": ["mean", "std"],
        "phase_grad": ["mean", "std"],
        "phi_s": ["mean", "std", "max", "abs_max"],
    }
    for field, stats in field_specs.items():
        field_data = records[0]["metrics"].get(field)
        if not isinstance(field_data, dict):
            continue
        field_entry: Dict[str, Optional[float]] = {}
        for stat in stats:
            seq = [float(r["metrics"][field][stat]) for r in records]
            field_entry[f"corr_{stat}"] = correlate(integral, seq)
            field_entry[f"corr_d_{stat}"] = correlate(deltas, diffs(seq))
        summary["fields"][field] = field_entry

    xi_seq = [float(r["metrics"].get("xi_c", float("nan"))) for r in records]
    summary["xi_c"] = {
        "corr": correlate(integral, xi_seq),
        "corr_d": correlate(deltas, diffs(xi_seq)),
    }

    bins = ["top5", "next15", "mid60", "bottom20"]
    for bin_name in bins:
        seqs: Dict[str, List[float]] = {
            "curv_mean": [],
            "grad_mean": [],
        }
        has_bin = True
        for rec in records:
            strat = rec.get("node_stratification")
            if not strat or bin_name not in strat:
                has_bin = False
                break
            seqs["curv_mean"].append(
                float(strat[bin_name].get("curv_mean", float("nan")))
            )
            seqs["grad_mean"].append(
                float(strat[bin_name].get("grad_mean", float("nan")))
            )
        if not has_bin:
            continue
        bin_entry: Dict[str, Optional[float]] = {}
        for key, seq in seqs.items():
            if any(math.isnan(x) for x in seq):
                continue
            bin_entry[f"corr_{key}"] = correlate(integral, seq)
            if len(seq) >= 2:
                bin_entry[f"corr_d_{key}"] = correlate(deltas, diffs(seq))
        summary["stratification"][bin_name] = bin_entry

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize precision walk correlations"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to precision walk JSONL",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to write JSON summary",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    summary = summarize(records)
    summary["source"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
