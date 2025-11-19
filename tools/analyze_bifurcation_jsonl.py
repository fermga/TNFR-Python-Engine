import argparse
import json
import math
import os
from typing import Any
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


def iter_jsonl(paths: List[str]) -> Iterable[Dict]:
    for p in paths:
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.lower().endswith(".jsonl"):
                    yield from iter_jsonl([os.path.join(p, name)])
            continue
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except OSError:
            continue


def summarize_numeric(
    records: Iterable[Dict],
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, int],
]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    # We'll collect for keys that appear numeric at least once
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                fv = float(v)
                sums[k] += fv
                counts[k] += 1
                if k not in mins or fv < mins[k]:
                    mins[k] = fv
                if k not in maxs or fv > maxs[k]:
                    maxs[k] = fv
    means = {k: (sums[k] / counts[k]) for k in counts if counts[k] > 0}
    return means, mins, maxs, counts


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze TNFR bifurcation JSONL outputs (merged or per-shard)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more JSONL files or directories containing JSONL files."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Optional: limit number of records processed (0 = all)."
        ),
    )
    args = parser.parse_args(argv)

    paths = args.paths
    records: List[Dict] = []
    total = 0
    for obj in iter_jsonl(paths):
        records.append(obj)
        total += 1
        if args.limit and total >= args.limit:
            break

    if not records:
        print("No records to analyze.")
        return 2

    # Classifications
    cls_counter: Counter[str] = Counter()
    handlers_true = 0
    handlers_present_count = 0
    topologies: Counter[str] = Counter()
    oz_values: Counter[str] = Counter()
    vf_values: Counter[str] = Counter()
    seeds: Counter[str] = Counter()

    def get_any(
        d: Dict[str, Any], keys: List[str], default: Any = None
    ) -> Any:
        for k in keys:
            if k in d:
                return d[k]
        return default

    for rec in records:
        cls = str(get_any(rec, ["classification", "class"], "unknown"))
        cls_counter[cls] += 1

        hp = get_any(
            rec,
            ["handlers_present", "handlers", "handlers_ok"],
            None,
        )
        if hp is not None:
            handlers_present_count += 1
            if bool(hp):
                handlers_true += 1

        topo = get_any(
            rec,
            ["topology", "topology_name", "graph_topology"],
            "?",
        )
        if topo is not None:
            topologies[str(topo)] += 1

        oz = get_any(rec, ["oz_intensity", "oz", "dissonance_intensity"], None)
        if oz is not None:
            oz_values[str(oz)] += 1

        vf = get_any(rec, ["vf", "nu_f", "structural_frequency"], None)
        if vf is not None:
            vf_values[str(vf)] += 1

        seed = get_any(rec, ["seed", "rng_seed"], None)
        if seed is not None:
            seeds[str(seed)] += 1

    # Numeric summaries
    means, mins, maxs, counts = summarize_numeric(records)

    print("=== Bifurcation Sweep Summary ===")
    print(f"Records: {len(records)}")
    if topologies:
        print("Topologies:")
        for k, v in topologies.most_common():
            print(f"  - {k}: {v}")
    if oz_values:
        print("OZ intensities:")
        for k, v in oz_values.most_common():
            print(f"  - {k}: {v}")
    if vf_values:
        print("νf values:")
        for k, v in vf_values.most_common():
            print(f"  - {k}: {v}")
    if seeds:
        print("Seeds:")
        for k, v in seeds.most_common():
            print(f"  - {k}: {v}")

    print("Classifications:")
    for k, v in cls_counter.most_common():
        pct = (100.0 * v / len(records)) if records else 0.0
        print(f"  - {k}: {v} ({pct:.1f}%)")

    if handlers_present_count:
        rate = 100.0 * handlers_true / handlers_present_count
        print(
            f"Handlers present: {handlers_true}/"
            f"{handlers_present_count} ({rate:.1f}%)"
        )

    # Highlight canonical telemetry fields if present
    highlight_keys = [
        "delta_phi_s",
        "delta_phase_gradient_max",
        "delta_phase_curvature_max",
        "coherence_length_ratio",
        "delta_dnfr_variance",
        "bifurcation_score_max",
        "structural_potential_shift",
    ]

    def print_stat(k: str) -> None:
        if k in counts and counts[k] > 0:
            print(
                f"  - {k}: count={counts[k]}, "
                f"mean={means.get(k, float('nan')):.6g}, "
                f"min={mins.get(k, float('nan')):.6g}, "
                f"max={maxs.get(k, float('nan')):.6g}"
            )

    print("Key metrics:")
    printed_any = False
    for k in highlight_keys:
        if k in counts:
            printed_any = True
            print_stat(k)
    if not printed_any:
        # Fallback: show top few numeric keys by coverage
        print("  (no predefined keys found; showing top numeric fields)")
        for k, c in sorted(
            counts.items(), key=lambda kv: kv[1], reverse=True
        )[:7]:
            print_stat(k)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
