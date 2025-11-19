"""Prime detection benchmark: TNFR ΔNFR vs trial division vs Miller-Rabin.

Run:
    python benchmarks/prime_detection_comparison.py --max 10000 --threshold 0.2

Outputs timing and precision / recall metrics for TNFR candidate detection.
"""
from __future__ import annotations
import time
import math
import argparse
from statistics import mean
from typing import List, Tuple, Dict

try:
    from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork
except ImportError:
    raise SystemExit("Cannot import ArithmeticTNFRNetwork; check PYTHONPATH")

# ----------------------- Baseline Algorithms -----------------------

def trial_division_is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True

# Miller-Rabin deterministic for 64-bit range using known bases
# For simplicity we implement a probabilistic version with fixed bases

def miller_rabin_is_prime(n: int) -> bool:
    if n < 2:
        return False
    # small primes shortcut
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        if n == p:
            return True
        if n % p == 0:
            return n == p
    # write n-1 as d*2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # bases chosen for reasonable coverage
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

# ----------------------- TNFR Prime Candidate -----------------------

def tnfr_candidates(
    max_number: int,
    threshold: float,
) -> Tuple[List[int], Dict[int, float]]:
    """Return TNFR prime candidates and ΔNFR map."""
    net = ArithmeticTNFRNetwork(max_number=max_number)
    cand = net.detect_prime_candidates(delta_nfr_threshold=threshold)
    return [n for n, _ in cand], {n: dnfr for n, dnfr in cand}

# ----------------------------- Metrics -----------------------------

def evaluate_accuracy(
    truth: List[int],
    predicted: List[int],
) -> Dict[str, float]:
    truth_set = set(truth)
    pred_set = set(predicted)
    tp = len(truth_set & pred_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }

# ----------------------------- Benchmark Runner -----------------------------

def benchmark(
    max_number: int,
    threshold: float,
    runs: int,
) -> None:
    numbers = list(range(2, max_number + 1))

    # Ground truth via Miller-Rabin for range; trial division only timed.
    truth = [n for n in numbers if miller_rabin_is_prime(n)]

    td_times, mr_times, tnfr_times = [], [], []

    # Trial division timing
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = [trial_division_is_prime(n) for n in numbers]
        td_times.append(time.perf_counter() - t0)

    # Miller-Rabin timing
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = [miller_rabin_is_prime(n) for n in numbers]
        mr_times.append(time.perf_counter() - t0)

    # TNFR candidate detection timing
    for _ in range(runs):
        t0 = time.perf_counter()
        cand, _map = tnfr_candidates(max_number, threshold)
        tnfr_times.append(time.perf_counter() - t0)

    acc = evaluate_accuracy(truth, cand)

    print("=== PRIME DETECTION COMPARISON ===")
    print(f"Range: 2..{max_number}")
    print(f"TNFR threshold: {threshold}")
    print("--- Timing (seconds, mean over runs) ---")
    print(f"Trial Division: {mean(td_times):.4f}")
    print(f"Miller-Rabin : {mean(mr_times):.4f}")
    print(f"TNFR ΔNFR     : {mean(tnfr_times):.4f}")
    print("--- Accuracy vs Miller-Rabin truth ---")
    print(
        f"Precision: {acc['precision']:.4f}  Recall: {acc['recall']:.4f}"
    )
    print(f"F1: {acc['f1']:.4f}")
    print(f"TP: {acc['tp']}  FP: {acc['fp']}  FN: {acc['fn']}")

    # Simple guidance
    if acc['recall'] < 0.85:
        print("[GUIDE] Consider increasing threshold to improve recall.")
    if acc['precision'] < 0.75:
        print(
            "[GUIDE] Consider decreasing threshold to reduce false positives."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TNFR ΔNFR prime candidate timing vs classical methods"
        )
    )
    parser.add_argument(
        "--max", type=int, default=5000, help="Maximum number (inclusive)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="ΔNFR threshold for candidate"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Repeats for timing"
    )
    args = parser.parse_args()
    benchmark(args.max, args.threshold, args.runs)
    

if __name__ == "__main__":
    main()
