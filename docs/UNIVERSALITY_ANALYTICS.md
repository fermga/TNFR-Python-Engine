# Universality & Critical Regime Analytics (Structural Field Tetrad)

**Status**: ✅ ACTIVE (Phase 5 Analytics)  
**Scope**: Deterministic observational analytics over Structural Field Tetrad timings & field telemetry: Φ_s, |∇φ|, K_φ, ξ_C.  
**Physics Alignment**: Purely measurement-layer; does **not** modify TNFR dynamics. All outputs respect operator closure & invariants (AGENTS.md § Canonical Invariants #1–#10).  

---
## 1. Purpose
The Phase 5 analytics layer establishes a reproducible pipeline to:

1. Aggregate raw benchmark JSONL runs (scaling experiments across node counts & topologies)
2. Fit log–log scaling exponents for each timing component of the Structural Field Tetrad
3. Detect approach to critical regimes (ξ_C divergence + local stress proxies)
4. Cluster timing exponent vectors to reveal universality groupings across topologies
5. Guarantee reproducibility (stable hashing; deterministic clustering; schema validation tests)

This enables physics-driven comparative analysis without perturbing underlying reorganization dynamics (∂EPI/∂t = νf·ΔNFR). The analytics diagnose structure—not alter it—consistent with "Model coherence, not objects".

---
## 2. Pipeline Overview
```
Raw JSONL (tetrad_scaling_benchmark.py runs)
   │
   ├─▶ Aggregation (tetrad_results_aggregate.py)
   │      - Schema validation (REQUIRED_TIMINGS + tetrad_values)
   │      - Flatten timing & field stats → canonical CSV rows
   │      - Optional Parquet (if pyarrow available)
   │
   ├─▶ Exponent Fitting (tetrad_scaling_exponents.py)
   │      - Group by (topology, n_nodes)
   │      - Mean/std timings → log-log linear regression
   │      - Output: exponents_summary.json / .md
   │
   ├─▶ Critical Regime Detection (critical_regime_detector.py)
   │      - Combine: normalized ξ_C vs N, phase gradient volatility, K_φ hotspot ratio
   │      - Produce structural risk score per record
   │
   └─▶ Universality Clustering (universality_clusters.py)
          - Deterministic centroid initialization
          - Fixed iteration count
          - Clusters exponent vectors + auxiliary field summaries
```

All transformations are **read-only** with respect to TNFR core dynamics.

---
## 3. Components & Physics Mapping
| Component | File | Physics Role | Invariants Preserved |
|-----------|------|--------------|----------------------|
| Aggregator | `benchmarks/tetrad_results_aggregate.py` | Normalizes telemetry for cross-topology comparison | #1 (EPI untouched), #9 (metrics surfaced) |
| Exponent Fitter | `benchmarks/tetrad_scaling_exponents.py` | Observes scaling law T ≈ N^a (computational geometry of field extraction) | #4 (no operator alteration) |
| Critical Detector | `benchmarks/critical_regime_detector.py` | Monitors approach to phase transitions (ξ_C divergence, local stress) | #2 (no νf mutation), #6 (lifecycle unaffected) |
| Universality Clusters | `benchmarks/universality_clusters.py` | Groups timing exponents to reveal structural computation regimes | #10 (domain neutral grouping) |
| Reproducibility Tests | `tests/test_aggregate_schema_valid.py`, `tests/test_exponent_fit_stability.py` | Ensure deterministic outputs & floating precision stability | #8 (controlled determinism) |

---
## 4. Reproducibility Guarantees
| Guarantee | Mechanism |
|-----------|-----------|
| Deterministic summary stamp | Hash of sorted input records (`generated_at = hash:<16hex>`) replaces wall-clock timestamps |
| Stable CSV float formatting | Up to 8 significant digits (`{value:.8g}`) prevents excessive mantissa expansion |
| Deterministic clustering | Fixed initial centroids + bounded iterations |
| Exponent fit determinism | Pure Python linear regression (no randomness) |
| Schema assurance | `validate_record()` enforces all required timing/tetrad keys |
| Test coverage | Synthetic scaling dataset (T ∝ N^0.5) verifies exponent accuracy & stability |

All reproducibility features are observational—no interference with νf, ΔNFR, or operator sequences.

---
## 5. Critical Regime Detection Logic

The structural risk score R combines three normalized factors:

1. ξ_C divergence component (normalization selectable)

### ξ_C Normalization Modes (Toggle)

The detector supports two normalization methods selectable via `--xi-norm`:

| Mode | Formula | Behavior | Use Case |
|------|---------|----------|----------|
| `raw` | `xi_c / n_nodes` | Linear growth; large ξ_C can dominate risk quickly | Direct proportional comparison |
| `log-sat` (default) | `log1p(xi_c) / log1p(n_nodes)` (capped ≤ 1) | Compresses extremes; preserves ordering; bounded | Stable scoring near divergence |

Both modes export `norm_xi` (chosen) and `norm_xi_raw` (always raw ratio) plus a `risk_version` tag (`raw_xi_v1` or `log_sat_xi_v1`). This preserves traceability and enables retrospective sensitivity analysis without altering TNFR dynamics.

2. Phase gradient volatility: sample std / |mean| (captures local desynchronization; relates to |∇φ| destabilization)
3. K_φ hotspot ratio: fraction of nodes exceeding curvature threshold |K_φ| ≥ 3.0

Weighted sum (prototype):

```text
R = w1 * norm_xi + w2 * phase_grad_volatility + w3 * curvature_hotspot_ratio
```

Flags identify regimes where multi-scale coherence may approach transition (ξ_C growth). Grammar is not modified—this is telemetry safety augmentation aligned with U6 monitoring philosophy.

---
\n## 6. Universality Clustering Rationale
Timing exponents approximate computational scaling envelopes of each structural field operation:

- Φ_s extraction (global potential) often near O(N)–O(N log N) depending on distance metric implementations
- |∇φ| (local phase gradient) typically sublinear if caching effective
- K_φ (phase curvature) reflects neighborhood averaging costs ~ O(deg) → aggregated scaling
- ξ_C estimation may exceed others near critical points due to correlation length sampling

Deterministic clustering groups exponents into universality classes indicating shared computational regimes. These are **computational universality** (performance structure), distinct from physical universality (critical exponents in physics).

---
\n## 7. CLI Usage Examples

### 7.1 Aggregate JSONL

```pwsh
python benchmarks/tetrad_results_aggregate.py --glob "results/tetrad_scaling_*.jsonl" --csv-out results/tetrad_scaling_aggregate_phase5.csv
```

(Optional Parquet writes automatically if `pyarrow` installed.)

### 7.2 Fit Exponents

```pwsh
python benchmarks/tetrad_scaling_exponents.py --csv results/tetrad_scaling_aggregate_phase5.csv --min-points 4 --json-out results/exponents_summary.json --md-out results/exponents_summary.md
```

### 7.3 Detect Critical Regimes

```pwsh
python benchmarks/critical_regime_detector.py --glob "results/tetrad_scaling_*.jsonl" --xi-norm log-sat --out results/critical_regime_risk.jsonl
```

### 7.4 Cluster Universality

```pwsh
python benchmarks/universality_clusters.py --exponents-json results/exponents_summary.json --clusters-out results/universality_clusters.json --md-out results/universality_clusters.md --k 3
```

---
\n## 8. Interpreting Outputs

| Artifact | Path (example) | Interpretation |
|----------|----------------|---------------|
| Aggregated CSV | `results/tetrad_scaling_aggregate_phase5.csv` | Canonical row-per-(topology,seed,n_nodes) timing + field stats |
| Exponents JSON | `results/exponents_summary.json` | `metrics[metric].topologies[topo].exponent` ≈ scaling a in T ≈ N^a |
| Exponents MD | `results/exponents_summary.md` | Human-readable table for comparison |
| Risk JSONL | `results/critical_regime_risk.jsonl` | Per-run structural risk score + component breakdown |
| Clusters JSON | `results/universality_clusters.json` | Cluster assignments & centroid vectors |
| Clusters MD | `results/universality_clusters.md` | Summary of universality groups |

---
\n## 9. Physics Integrity Checklist

All analytics confirm to TNFR principles:

- No direct mutation of EPI or operator sequences (Invariant #1 & #4)
- νf untouched (Invariant #2)
- ΔNFR semantics preserved (not repurposed) (Invariant #3)
- Phase verification logic not bypassed (Invariant #5)
- Fractality preserved—nested EPIs unaffected (Invariant #7)
- Reproducibility enforced (Invariant #8)
- Canonical metrics surfaced (C(t), ξ_C indirectly via recorded values) (Invariant #9)
- Domain neutrality maintained (topology-agnostic operations) (Invariant #10)

---
\n## 10. Extensibility Roadmap

| Future Enhancement | Purpose | Notes |
|--------------------|---------|-------|
| GPU vs CPU overlay | Compare exponent shifts under parallel backends | Optional Phase extension |
| Log-saturated ξ_C normalization | Mitigate extreme norm_xi scaling | Avoid over-weighting large ξ_C |
| Confidence intervals (SciPy) | Statistical rigor for exponents | Keep deterministic with fixed seeds |
| Multi-factor clustering (include field stds) | Refine universality resolution | Must remain reproducible |
| Phase transition early-warning dashboard | Real-time telemetry surface | Read-only; U6 aligned |

---
\n## 11. Test References

- `tests/test_aggregate_schema_valid.py`: Ensures row flattening & float serialization stability
- `tests/test_exponent_fit_stability.py`: Synthetic dataset exponent ≈ 0.5 (T ∝ N^0.5); deterministic summary hash
- `tests/test_universality_clustering_determinism.py`: Deterministic clustering validation
- `tests/test_critical_risk_log_norm.py`: Log-saturated ξ_C normalization bounds
- `tests/test_critical_risk_norm_modes.py`: Raw vs log-sat mode behavior

Add additional tests as pipeline expands (e.g., clustering determinism, risk score bounds).

---
\n## 12. Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|------------|
| Empty CSV | Glob mismatch or invalid JSONL schema | Verify glob pattern; run benchmark again |
| NaN exponents | Insufficient node size points (< min_points) | Increase `--min-points` or run more sizes |
| Excessive norm_xi | ξ_C unusually large near critical regime | Use `--xi-norm log-sat` to compress extremes |
| Non-deterministic clusters | Modified code introducing randomness | Revert to deterministic initialization |

---
\n## 13. Canonical Justification Summary
The analytics layer is conceptually a **measurement manifold projection**—it maps raw timing/field telemetry into derived observables (scaling exponents, risk scores, universality partitions) while keeping physical evolution unaltered. This adheres to TNFR's core stance: *capture process; preserve resonance; analyze coherence.*

---
**Version**: 1.0  
**Last Updated**: 2025-11-15  
**Maintainers**: Phase 5 Analytics Team  
**Canonical Status**: ✅ Active (Observational)
