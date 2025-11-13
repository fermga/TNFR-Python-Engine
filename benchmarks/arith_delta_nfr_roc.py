#!/usr/bin/env python3
"""
ΔNFR threshold calibration with ROC/AUC up to N=10k/100k and cross-validation.

Computes τ(n), σ(n), ω(n) for n ≤ N via sieve-like methods; then ΔNFR; then ROC curves
against ground-truth primality from a prime sieve. Exports JSON with AUC and fold metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------- Sieve utilities -----------------------


def sieve_primes(N: int) -> np.ndarray:
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(N ** 0.5) + 1):
        if is_prime[p]:
            is_prime[p * p : N + 1 : p] = False
    return is_prime


def sieve_tau_sigma(N: int) -> Tuple[np.ndarray, np.ndarray]:
    tau = np.zeros(N + 1, dtype=np.int32)
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        tau[d::d] += 1
    return tau, sigma


def sieve_spf(N: int) -> np.ndarray:
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            for j in range(i * i, N + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    # fill remaining with self where zero (primes below sqrt region)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
    return spf


def omega_from_spf(spf: np.ndarray) -> np.ndarray:
    N = len(spf) - 1
    omega = np.zeros(N + 1, dtype=np.int32)
    for n in range(2, N + 1):
        x = n
        cnt = 0
        while x > 1:
            p = spf[x]
            cnt += 1
            x //= p
        omega[n] = cnt
    return omega

# ----------------------- ΔNFR/ROC ------------------------------


def delta_nfr_array(N: int, tau: np.ndarray, sigma: np.ndarray, omega: np.ndarray, *,
                    zeta=1.0, eta=0.8, theta=0.6) -> np.ndarray:
    n = np.arange(N + 1, dtype=np.float64)
    n[0] = 1.0
    dnfr = (zeta * (omega - 1)) + (eta * (tau - 2)) + (theta * (sigma / n - (1.0 + 1.0 / n)))
    dnfr[:2] = np.inf
    return dnfr


def roc_points(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # label 1 = prime, 0 = composite; we predict prime if score <= thr (low ΔNFR)
    tpr = []
    fpr = []
    P = labels.sum()
    Nn = len(labels) - P
    for thr in thresholds:
        pred = (scores <= thr)
        TP = int(np.logical_and(pred, labels == 1).sum())
        FP = int(np.logical_and(pred, labels == 0).sum())
        tpr.append(TP / P if P > 0 else 0.0)
        fpr.append(FP / Nn if Nn > 0 else 0.0)
    return np.array(fpr), np.array(tpr)


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    # trapezoidal rule; assumes fpr sorted with thresholds
    order = np.argsort(fpr)
    xf, yf = fpr[order], tpr[order]
    # Prefer numpy.trapezoid if available, fallback to trapz
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(yf, xf)
    else:
        area = np.trapz(yf, xf)
    return float(area)


def kfold_blocks(N: int, k: int) -> List[Tuple[int, int]]:
    # Return k contiguous blocks [start, end] covering [2..N]
    size = (N - 1) // k + 1
    blocks = []
    start = 2
    for _ in range(k):
        end = min(N, start + size - 1)
        blocks.append((start, end))
        start = end + 1
        if start > N:
            break
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=str, default="", help="Optional JSON output path")
    args = ap.parse_args()

    N = args.N
    is_prime = sieve_primes(N)
    tau, sigma = sieve_tau_sigma(N)
    spf = sieve_spf(N)
    omega = omega_from_spf(spf)

    dnfr = delta_nfr_array(N, tau, sigma, omega)
    # Use labels on 2..N
    idx = np.arange(2, N + 1)
    y = is_prime[idx].astype(np.int32)
    scores = dnfr[idx]

    # thresholds spanning [min, max]
    thr_min, thr_max = float(np.min(scores[np.isfinite(scores)])), float(np.max(scores[np.isfinite(scores)]))
    thresholds = np.linspace(thr_min, thr_max, num=256, dtype=np.float64)

    # Global ROC/AUC
    fpr, tpr = roc_points(scores, y, thresholds)
    auc_all = auc(fpr, tpr)

    # k-fold contiguous blocks (train to choose thr; test to evaluate)
    blocks = kfold_blocks(N, args.folds)
    fold_metrics: List[Dict] = []
    for (s, e) in blocks:
        mask_test = (idx >= s) & (idx <= e)
        mask_train = ~mask_test
        # choose best threshold on train (max F1 by threshold sweep)
        best_thr = None
        best_f1 = -1.0
        for thr in thresholds:
            pred = (scores[mask_train] <= thr)
            TP = int(np.logical_and(pred, y[mask_train] == 1).sum())
            FP = int(np.logical_and(pred, y[mask_train] == 0).sum())
            FN = int(np.logical_and(~pred, y[mask_train] == 1).sum())
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        # evaluate on test
        pred_test = (scores[mask_test] <= best_thr)
        TP = int(np.logical_and(pred_test, y[mask_test] == 1).sum())
        FP = int(np.logical_and(pred_test, y[mask_test] == 0).sum())
        FN = int(np.logical_and(~pred_test, y[mask_test] == 1).sum())
        TN = int(np.logical_and(~pred_test, y[mask_test] == 0).sum())
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_test = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fold_metrics.append({
            "block": [int(s), int(e)],
            "best_thr": best_thr,
            "precision": prec,
            "recall": rec,
            "f1": f1_test,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        })

    result = {
        "N": N,
        "auc_global": auc_all,
        "folds": args.folds,
        "blocks": blocks,
        "fold_metrics": fold_metrics,
        "note": "ROC computed with prime=score<=thr. Expect near-perfect separation since primes have ΔNFR=0."
    }

    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result), encoding="utf-8")
        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
