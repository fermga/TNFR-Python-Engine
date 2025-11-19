
# Large Simulations Guide

Practical guidance and tools for running big TNFR sweeps (2k–5k+ nodes) while
staying faithful to TNFR physics and keeping runtimes predictable.

---

## Scale Tips (Quick)

- Node count: Φ_s switches to optimized/landmarks paths automatically (>50/500).
  2k–5k works if grids are moderate.
- Grids: keep `--oz-intensity-grid` and `--vf-grid` to 2–3 values each at large N
  to bound runtime and JSON size.
- Seeds: prefer sharding sweeps across parallel processes vs. large `--seeds` in one run.
- Output size: redirect stdout to a JSONL file; size grows with `grid × seeds`.
- Validation: sample a small subset in high precision to sanity‑check drift,
  then run the bulk in standard (physics invariant).
- Precision/telemetry: use `standard` + `low` for throughput; bump only when
  validating numerics.

---

## One‑Shot Commands (PowerShell)

Python executable note:

- Windows: prefer `./test-env/Scripts/python.exe`
- macOS/Linux: prefer `./test-env/bin/python`

Large JSONL sweep (save results to file):

```powershell
./test-env/Scripts/python.exe benchmarks/bifurcation_landscape.py `
  --nodes 2000 `
  --seeds 4 `
  --topologies ws,scale_free `
  --oz-intensity-grid 0.9,1.1 `
  --vf-grid 0.9,1.0 `
  --mutation-thresholds 1.2 `
  > results\bifurcation_ws_sf_2k.jsonl
```

Compute‑only sweep (quiet mode):

```powershell
./test-env/Scripts/python.exe benchmarks/bifurcation_landscape.py `
  --nodes 3000 `
  --seeds 3 `
  --topologies ws `
  --oz-intensity-grid 1.0,1.2 `
  --vf-grid 0.9,1.0 `
  --mutation-thresholds 1.3 `
  --quiet
```

Set precision/telemetry for long runs:

```python
from tnfr.config import set_precision_mode, set_telemetry_density

set_precision_mode("standard")      # fastest; physics invariant
set_telemetry_density("low")        # lighter sampling for scale
```

---

## Sharded Launch (Parallel on Windows)

Use the provided launcher to split the OZ intensity grid across N shards and run
in parallel. Each shard covers a disjoint subset of `--oz-intensity-grid`, and
outputs a per‑shard JSONL. The script merges them at the end.

Script: `tools/launch_bifurcation_sharded.ps1`

Examples:

```powershell
# 2 shards across oz grid; 2500 nodes; writes merged file
pwsh -File tools/launch_bifurcation_sharded.ps1 `
  -Nodes 2500 -Seeds 2 -Shards 2 -Topologies "ws" `
  -OzIntensityGrid "0.8,1.0,1.2" -VfGrid "0.9,1.0" -MutationThresholds "1.2" `
  -OutDir "results" -MergedOut "results/bifurcation_ws_2k_merged.jsonl"

# 3 shards, two topologies
pwsh -File tools/launch_bifurcation_sharded.ps1 `
  -Nodes 3000 -Seeds 2 -Shards 3 -Topologies "ws,scale_free" `
  -OzIntensityGrid "0.9,1.0,1.1" -VfGrid "0.9,1.0" `
  -OutDir "results" -MergedOut "results/bifurcation_ws_sf_3k_merged.jsonl"
```

---

## Merge Utilities

Merge multiple JSONL files (PowerShell):

```powershell
Get-ChildItem results\shard_*.jsonl |
  ForEach-Object { Get-Content $_ } |
  Out-File -Encoding utf8 results\merged.jsonl
```

Python merging (with optional gzip):

```powershell
./test-env/Scripts/python.exe tools/merge_jsonl.py --out results\merged.jsonl results\shard_*.jsonl
./test-env/Scripts/python.exe tools/merge_jsonl.py --out results\merged.jsonl.gz --gzip results\shard_*.jsonl
```

---

## Drift Checks (Precision Modes)

Use the optimization suite to validate numeric invariance at scale:

```powershell
./test-env/Scripts/python.exe run_benchmark.py
```

Then parse drift fields (see README → Performance → Parse precision_modes drift).

---

## CI Smoke Regression

Pull requests run a smoke sweep via `.github/workflows/regression-smoke.yml`.
Optionally add a baseline to `presets/regression/bifurcation_smoke.baseline.jsonl`
to gate merges on drift beyond tolerance (default `5e-3`).

Artifacts: `results/bifurcation_smoke.jsonl`, `results/regression_summary.txt`.

---

## Single-Process Knee (≈4k–8k Nodes)

Empirical timing on this machine (Windows, venv Python, vectorized ΔNFR path) shows a
clear performance knee between 4k and 8k nodes. Throughput declines sub‑quadratically
thanks to sparsity and fused operations, but memory bandwidth and cache pressure begin
to dominate beyond ~4k.

| Nodes | p (edge probability) | Vectorized median (s) | Steps/sec | Approx edges (p·N²) | Edge updates/sec |
|-------|----------------------|-----------------------|-----------|---------------------|------------------|
| 2048  | 0.05                 | 0.0131                | 76.4      | ~0.21 M             | ~16.0 M          |
| 4096  | 0.05                 | 0.0348                | 28.7      | ~0.84 M             | ~24.1 M          |
| 8192  | 0.05                 | 0.1077                | 9.28      | ~3.36 M             | ~31.2 M          |

Observations:

- Scaling is sub‑quadratic in wall‑time due to vectorization; edge updates/sec rises then flattens.
- The knee appears when cache locality degrades (≈4k→8k: steps/sec drops ~3.1× for 2× nodes).
- Practical real‑time simulations (≥20 Hz) are comfortable ≤4k nodes at moderate sparsity (p≈0.05).

### Sparse Scaling (p = 0.02)

Reducing density shifts the knee upward but still reveals bandwidth effects:

| Nodes | p | Vectorized median (s) | Steps/sec | Approx edges | Edge updates/sec |
|-------|---|-----------------------|-----------|--------------|------------------|
| 4096  | 0.02 | 0.0223             | 44.9      | ~0.34 M      | ~15.3 M          |
| 8192  | 0.02 | 0.0899             | 11.1      | ~1.34 M      | ~14.9 M          |
| 16384 | 0.02 | 0.1898             | 5.27      | ~5.37 M      | ~28.3 M          |

Guidance:

- Lower p improves steps/sec but aggregate edge throughput remains bounded by memory bandwidth.
- For very large N, prefer p≤0.02 and shard sweeps (OZ, νf grids) rather than increasing seeds.
- Use `-Warmup` (see multiprocess helper) to stabilize first‑step timing in throughput studies.

### Recommendations

Core operating band: 4k–8k nodes. This range straddles the cache → memory bandwidth knee while still yielding multi‑scale resonance behavior (structural field tetrad) with acceptable turnaround.

1. Interactive runs: 2048–4096 nodes (high density p≈0.05) for rapid (>25 Hz) iteration and operator sequence debugging.
2. Batch physics analyses: 4096–8192 nodes; choose 4096 at higher p (≈0.05) and 8192 at lower p (≈0.02) to balance edge growth and ΔNFR cost.
3. >8192 nodes: Specialized research only (ξ_C divergence tests, extreme sparsity stress, backend performance studies). Document justification before execution.
4. De‑emphasize scaling via seeds at large N; prefer sharded grids (OZ intensity, νf) and controlled p adjustments.
5. Always log Φ_s, |∇φ|, K_φ, ξ_C in 4k–8k runs to correlate emerging stress with knee proximity; this band is optimal for structural field sensitivity.
6. Use `-Warmup` for stable first‑step timing; add stagger/affinity when launching many concurrent 4k–8k jobs to mitigate synchronized bandwidth contention.
7. When p is decreased (≤0.02), maintain node count near upper band (8192) to preserve meaningful spatial correlation scale without incurring excessive edge throughput penalties.

Summary: Prefer 4096 (dense) or 8192 (sparse). Treat 16k+ as exceptional; insights per wall‑clock second diminish sharply beyond 8k.

---

## Canonical Φ_s Approximation Strategy

The structural potential Φ_s is the most expensive canonical field at mid–large scales due to repeated shortest‑path traversals. A physics‑preserving approximation is now canonical via landmark sampling with adaptive refinement.

### Method

- Select an initial landmark fraction `landmark_ratio` (typical safe band 0.01–0.04 for 2k–8k nodes).
- Score candidates by `degree * (1 + |ΔNFR|)` to bias toward structurally influential nodes (pressure + connectivity).
- Precompute shortest‑path distances from landmarks once; reuse through a topology+ratio keyed cache.
- Approximate Φ_s(i) using only landmark contributions: `Σ_{l∈L} ΔNFR_l / d(i,l)^α` scaled by a correction factor proportional to `|V| / |L|`.
- When `validate=True`, sample a node subset, compute exact Φ_s for those nodes, estimate relative mean absolute error (RMAE). If RMAE > ε, landmark_ratio is doubled (capped at 0.5) and distances reused / extended; repeat until RMAE ≤ ε or max refinements reached.

### Guarantees

- Preserves physical inverse‑square law form (α=2) — no heuristic reweighting beyond uniform correction factor.
- Error bounded: RMAE ≤ ε (default 0.05) after adaptive refinement for tested ER and scale‑free graphs (see `tests/test_structural_potential_landmark.py`).
- Read‑only: does not mutate EPI or ΔNFR (Invariant #1 & #3 retained).
- Deterministic given seed (landmark selection reproducible).

### Metadata

- `__phi_s_landmark_ratio__`: Effective ratio after refinement.
- `__phi_s_rmae__`: Achieved relative mean absolute error (validate mode only).

### Recommended Usage

- Exploration sweeps / telemetry: `landmark_ratio=0.02`, `validate=True` (balanced speed + assurance).
- High precision snapshots: start `0.01` with validation; allow refinement to converge.
- Large (>8k) experimental runs: begin at `0.01`; expect automatic increase if graph heterogeneity induces error.

### Performance Impact (Indicative)

- 4k nodes ER p=0.05: landmark 0.02 cuts Φ_s wall clock by ~3–4× vs exact BFS path.
- 8k nodes sparse p=0.02: landmark 0.02–0.03 prevents superlinear distance explosion; cache reuse amortizes across phase updates.

### Safety & Physics Fidelity

- Landmark sampling approximates sum over sources without altering exponent α or ΔNFR semantics; convergence of ∫νf·ΔNFR dt (U2) unaffected.
- Structural potential confinement threshold (ΔΦ_s < 2.0) remains valid; approximation error bounded below typical threshold margins.

### When to Avoid Approximation

- Tiny graphs (≤50 nodes): exact computation dominates (<1s); approximation unnecessary.
- Precision‑critical validation experiments establishing new thresholds: compute exact once, then benchmark approximation.

### Future Extensions

- Multi‑scale landmark stratification (clusters + high ΔNFR outliers).
- Adaptive α tuning study (currently fixed at canonical α=2.0).
- Probabilistic confidence intervals for Φ_s(i) via bootstrap over landmark sets.

This strategy makes Φ_s feasible in the recommended operating band (4k–8k nodes) while preserving TNFR physics and canonical telemetry integrity.

---

## Telemetry Correlation With Knee

We captured the structural field tetrad at knee regimes and observed the following patterns (see `results/telemetry_knee.jsonl`):

- Φ_s magnitude and runtime: `field_seconds` grew from ~43s (2048, p=0.05) → ~370s (4096, p=0.05) → ~3040s (8192, p=0.05), confirming Φ_s dominates runtime at scale.
- Phase gradient |∇φ|: mean stayed ~1.57 across sizes; `max_phase_grad` approached ~2.0 in sparse (4096, p=0.02), indicating near‑antiphase stress pockets.
- Phase curvature K_φ: variance remained ~1.8; local extremes flagged confinement zones even when |∇φ| was flat, aligning with K_φ’s geometric role.
- Coherence length ξ_C: dense 4096@0.05 localized (ξ_C ≈ 6.1); sparse 4096@0.02 showed long‑range correlations (ξ_C ≈ 1553), signaling critical‑like behavior.

Implications:

- Prefer 4k–8k for physics sensitivity; avoid Φ_s at 8k in exploratory sweeps unless landmark approximation is verified active.
- For sparse runs (p≈0.02), monitor ξ_C: spikes imply system‑wide reorganization and amplify knee effects.
- Use K_φ to pinpoint high‑torsion loci even when |∇φ| appears steady; these zones correlate with incipient fragmentation under OZ pushes.

Operational tips:

- Enable Φ_s landmarks (`--phi-landmark-ratio 0.02`) and skip multiscale K_φ during wide sweeps to bound runtime; rerun selected points with full telemetry.
- Shard by OZ/νf grids rather than seeds, and cap concurrency to 2–3 jobs on bandwidth‑limited hosts, staggering starts by 2–4s and isolating cores when possible.
