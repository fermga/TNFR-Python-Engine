# TNFR Factoring Playbook

## Purpose

This playbook codifies the canonical procedure for factorizing any integer `n > 1`
inside the factorization lab. Every step maps directly to TNFR physics:
Paley graph instrumentation, nodal decoding, operator strategies, and
TNFR-native verification metrics.

## Prerequisites

1. Install the TNFR engine: `pip install -e .` from the repository root.
2. Ensure FFT backends are available (`tnfr.dynamics.advanced_fft_arithmetic` is loaded
   automatically; distributed backends are optional).
3. Export `TNFR_PARTITION_*` overrides only if you need custom block sizes.
4. Keep `make.cmd smoke-tests` passing—this validates the self-optimizing engine,
   arithmetic formalism, and grammar rules before factoring new numbers.

## Canonical workflow

1. **Graph construction**
   - Call `SpectralPaleyFactorizer.analyze(n)`.
   - The factorizer lifts `n` to a Paley/Jacobi-compatible modulus, attaches EPI/νf
     annotations, and records Laplacian eigenvalues via FFT.
2. **Partition planning**
   - `plan_paley_partitions` deterministically divides the graph using
     `PartitionPlannerConfig`. Telemetry (Φ_s, |∇φ|, K_φ, ξ_C) is stored per block.
3. **Nodal decoding**
   - Each partition executes `[UM, RA, IL, THOL]`. The nodal decoder records ΔNFR
     attenuation, coherence ratios, and inferred periodicity.
4. **Operator strategy synthesis**
   - The self-optimizing engine proposes canonical sequences (glyphs) and per-partition
     strategies for {AL, IL, RA, SHA}. These are preserved in `operator_strategy_plan`.
5. **TNFR verification**
   - `_verify_factors_tnfr` collects deterministic invariants:
     ΔNFR gain, Φ_s confinement vs parent, coherence windows, and periodicity locks.
     Factors whose partitions satisfy the criteria become `tnfr_certified_factors`.
6. **Certificate emission**
   - Invoke `analyze(..., trace_certificates=True)` to persist:
     `partition_states`, `strategy_plan_snapshot`, `nodal_decoding_snapshot`,
     `tnfr_verification_snapshot`, and the invariant report.
7. **(Optional) arithmetic cross-check**
   - Classical gcd/divisibility remains available for diagnostics but is no longer the
     canonical proof. Treat `tnfr_verification_snapshot` as the decisive evidence.

## Automation recipes

### Single run with certificate
```pwsh
python - <<'PY'
from tnfr_factorization import SpectralPaleyFactorizer
factorizer = SpectralPaleyFactorizer()
result = factorizer.analyze(221, trace_certificates=True)
print(result.tnfr_certified_factors)
print(result.certificate_path)
PY
```

### Full-spectrum benchmark with range sweep
```pwsh
python factorization-lab/benchmarks/full_spectrum_factorization.py \
  --output full_spectrum_range.json \
  --range-start 150 --range-stop 190
```
Each entry in `factorization-lab/results/benchmarks/` includes partition traces and the
serialized TNFR verification report for downstream audits.

### Failure telemetry controls

`SpectralPaleyFactorizer` automatically records failed attempts in
`results/failure_telemetry/` so operators can study Laplacian gaps,
partition coverage, and verifier bottlenecks. Control this behavior via
the `TNFR_FAILURE_TELEMETRY` environment variable:

| Value | Effect |
|-------|--------|
| unset / `1` / `true` / `on` | Failure telemetry stays enabled (default) |
| `0` / `false` / `off` | Disables telemetry recording entirely |

To redirect artifacts, pass `failure_telemetry_root=Path(...)` when constructing
`SpectralPaleyFactorizer`. This keeps the JSON artifacts and manifest under a
per-run scratch directory while the default production pipelines continue to
log into `results/failure_telemetry/`.

## Verification criteria (summary)

| Criterion | Threshold |
|-----------|-----------|
| ΔNFR gain | ≥ 0.15 relative drop inside `[UM, RA, IL, THOL]` |
| Coherence | 0.72 ≤ `coherence_ratio` ≤ 1.38 |
| Φ_s delta  | \|Φ_s(local) − Φ_s(parent)\| ≤ 0.35 |
| Evidence   | ≥ 3 of the five flags (`stabilized`, `coherence`, `dnfr_gain`, `phi`, `periodic`) per partition |
| Coverage   | ≥ 50% of partitions referencing a factor must endorse it |

Factors passing the criteria appear in `tnfr_certified_factors` and the certificate's
`tnfr_verification_snapshot`. Use these lists for downstream automation (e.g. deciding
whether to cascade into higher-level operators or to log a divergence).

## Troubleshooting

| Symptom | Likely cause | Remedy |
|---------|--------------|--------|
| `tnfr_certified_factors` empty | Partitions failed ΔNFR or coherence criteria | Inspect partition certificates (results/certificates/partitioned/…) for the offending block; adjust partition target size or run dissonance probes (OZ) before IL |
| `tnfr_verification_snapshot` missing | `trace_certificates=False` or nodal decoder disabled | Re-run with `trace_certificates=True`; ensure `[UM, RA, IL, THOL]` is active in `nodal_operator_sequence` |
| High `phi_delta_max` values | Parent Φ_s too low/high or partitions mis-sized | Tune `TNFR_PARTITION_TARGET_SIZE` or apply `THOL` stabilizers earlier |
| Self-optimizing engine unavailable | Optional dependency missing | Fallback sequence (`emission→coupling→resonance→coherence→silence`) still valid, but consider installing `tnfr.engines.self_optimization` |

## References

- `theory/UNIFIED_GRAMMAR_RULES.md` — nodal decoder completeness & grammar proofs.
- `theory/TNFR_NODAL_FACTOR_COMPLETENESS.md` — full statement of the TNFR-only
  factorization theorem and verifier thresholds.
- `docs/OPERATOR_CERTIFICATES.md` — certificate schema and TNFR evidence fields.
- `factorization-lab/benchmarks/full_spectrum_factorization.py` — automation entry point.
- `factorization-lab/tests/test_spectral_paley.py` — regression coverage for decoder,
  certificates, and verification metrics.
