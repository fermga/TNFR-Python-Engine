# Operator-Sequence Certificates for Spectral Factorization

## Purpose

Provide TNFR-native evidence that the Paley graph for `n` reorganizes coherently through
validated operator sequences, producing sub-EPIs that correspond to the recovered
factors. Certificates must include per-partition TNFR telemetry, operator-strategy plans,
and invariant reports so downstream agents can replay the full sequence.

## Canonical pipeline

1. **Input** `n`
2. **Construct** Paley/Jacobi graph and annotate nodes for FFT
3. **Partition** via deterministic planner (configurable target size/overlap)
4. **Execute** `[UM, RA, IL, THOL]` on each partition (nodal decoder)
5. **Plan** AL/IL/RA/SHA strategies per partition from `StrategyRegistry`
6. **Emit** enriched certificate with:
   - partition before/after TNFR fields (Φ_s, |∇φ|, K_φ, ξ_C, ΔNFR)
   - nodal decoder outcomes (coherence ratios, inferred factors)
   - operator strategy plan snapshot + nodal decoding snapshot
   - invariant report citing U1–U6 + canonical invariants (#1–#6)
7. **Optionally verify** candidates via arithmetic gcd (supporting evidence)

## Certificate contents (JSON)

- `operators`, `canonical_operators`: sequence recommended/applied
- `telemetry`: global TNFR metrics (Φ_s, |∇φ|, K_φ, ξ_C, ΔNFR)
- `pressure_components`: three-term ΔNFR decomposition — factorization, divisor, sigma pressures (§6)
- `dual_lever_analysis`: capacity vs pressure lever classification of the operator sequence (§8)
- `conservation_proxies`: Noether charge Q = Φ_s + K_φ, Lyapunov energy E = 0.5·(Φ_s² + |∇φ|² + K_φ²)
- `arithmetic_epi`, `arithmetic_nu_f`: nodal equation components for the target integer (§5)
- `partition_states`: map `partition_id -> {before, after, node_count, boundary_count, candidate_factors}`
- `strategy_plan_snapshot`: subset of `operator_strategy_plan` covering AL/IL/RA/SHA
- `nodal_decoding_snapshot`: per-partition decoder log (sequence, ΔNFR convergence, dynamic factors)
- `invariant_report`: structured proof of U1–U6 + canonical invariants (#1–#6)
- `tnfr_verification_snapshot`: deterministic certification data (per-factor endorsements, ΔNFR gains, coherence windows) that replaces the old gcd-only confirmation
- `partition_manifest*` references + per-partition certificate files

Certificates live in `results/certificates/`, and partition artifacts are stored under
`results/certificates/partitioned/`. Threshold controls (`TNFR_PARTITION_OUTPUT_DIR`,
`TNFR_PARTITION_FILELIST_THRESHOLD`) determine where the files land.

> **Verification update**: gcd/divisibility checks remain available for diagnostics, but
> TNFR publication artifacts now treat the `tnfr_verification_snapshot` as the canonical
> proof of factor identity. A factor is "accepted" when its partitions satisfy the
> deterministic ΔNFR/coherence/Φ_s criteria captured in that snapshot.

## Workflow summary

```text
n → Paley/Jacobi graph → partitions + telemetry → nodal decoder [UM,RA,IL,THOL]
→ strategy plan selection → certificate emission (partition states + invariants)
→ (optional) arithmetic gcd verification
```

## Operational guidance

- Always call `SpectralPaleyFactorizer.analyze(..., trace_certificates=True)` when you need
  reproducible evidence. Certificates should accompany benchmark runs (`paley_gap_*`,
  `full_spectrum_factorization`) and CI validations.
- Inspect `partition_states` to see how each block satisfied ΔNFR convergence and coherence
  matching. The `nodal_state` metadata is the authoritative view of the local EPI evolution.
- Use `invariant_report` to confirm the run satisfied TNFR grammar and canonical invariants
  without re-running the analysis. If any rule is not “tracked”, the result should be
  considered provisional.

## Future extensions

- Attach raw operator logs (beyond the nodal decoder) when additional sequences are introduced.
- Include visual artifacts (plots of Φ_s, |∇φ|, K_φ, ξ_C per partition) referenced from the
  certificate metadata.
- Integrate certificate summaries into Zenodo release bundles so every dataset ships with
  structural evidence.
