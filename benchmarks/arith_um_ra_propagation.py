#!/usr/bin/env python3
"""
UM/RA propagation benchmark on Arithmetic TNFR Network.

- Builds arithmetic network up to N (default 1000)
- Computes phases, applies UM (coupling) and runs RA (resonance) from primes
- Prints simple metrics and optionally writes JSONL with activation snapshots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=1000, help="Max number (inclusive)")
    p.add_argument("--steps", type=int, default=6, help="Resonance steps")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--decay", type=float, default=0.1)
    p.add_argument("--dphi", type=float, default=1.57079632679, help="Δφ_max (radians)")
    p.add_argument("--out", type=str, default="", help="Optional JSONL output path")
    args = p.parse_args()

    net = ArithmeticTNFRNetwork(args.N)
    net.compute_phase(store=True)

    hist = net.resonance_from_primes(
        steps=args.steps,
        init_value=1.0,
        gain=args.gain,
        decay=args.decay,
        delta_phi_max=args.dphi,
        normalize=True,
    )

    # Metrics per step
    records: List[Dict] = []
    for t, act in enumerate(hist):
        rec = {"t": t}
        rec.update(net.resonance_metrics(act))
        records.append(rec)

    # Print summary
    print("UM/RA Propagation Summary:")
    print(f"  N = {args.N}, steps = {args.steps}, gain = {args.gain}, decay = {args.decay}, Δφ_max = {args.dphi}")
    for r in records:
        print(f"  t={r['t']}: mean_act={r['mean_activation']:.4f}, frac>=0.5={r['fraction_ge_0_5']:.4f}, corr_primes={r['corr_with_primes']:.4f}")

    # Optional JSONL
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for t, act in enumerate(hist):
                line = {
                    "t": t,
                    "metrics": records[t],
                    "activation": {int(k): float(v) for k, v in act.items()},
                    "params": {
                        "N": args.N,
                        "steps": args.steps,
                        "gain": args.gain,
                        "decay": args.decay,
                        "delta_phi_max": args.dphi,
                    },
                }
                f.write(json.dumps(line) + "\n")
        print(f"Wrote JSONL: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
