# Self-Optimization Pipeline & Validation

**Status**: ✅ ACTIVE — integrated with Paley/Spectral partitions and SDK auto-optimization

**Scope**: End-to-end workflow that takes partition manifests, produces dry-run recommendations via `scripts/run_self_optimization.py`, validates them via `scripts/run_self_opt_validation.py`, and feeds qualified operator strategies into downstream orchestration (`operator_strategy_plan`).

---

## 1. Workflow Overview

1. **Generate recommendations**
   - Run `scripts/run_self_optimization.py` against a partition manifest to produce dry-run payloads under `results/self_optimization/` (one JSON per partition).
2. **Review payloads & telemetry**
   - Inspect per-partition JSON for ΔΦ_s, ΔC(t), Si, and operator sequences recorded by the engine.
3. **Validate targeted pytest suites**
   - Execute `scripts/run_self_opt_validation.py` so each recommendation family is mapped to the correct pytest subset (e.g., Paley spectral tests).
4. **Promote successful plans**
   - Merge validated recommendations into the canonical `operator_strategy_plan` for the relevant topology (see §5).
5. **Dashboard + regression checks**
   - Push the summary JSON to dashboards to visualize ΔΦ_s/ΔC(t) per partition and re-run smoke suites listed in the review checklist.

---

## 2. Runner Usage Examples

### 2.1 Generate dry-run recommendations

```powershell
pwsh> "C:/Program Files/Python313/python.exe" scripts/run_self_optimization.py \`
    --manifest manifests/paley/_manifest.json \`
    --output-dir results/self_optimization/paley_v3 \`
    --operation-type paley_partition \`
    --max-workers 4 \`
    --capture-snapshots \`
    --summary results/self_optimization/paley_v3_summary.json
```

Key flags:
- `--manifest` (required) — path to `_manifest.json` describing partitions.
- `--output-dir` — storage for per-partition payloads + telemetry.
- `--operation-type` — label recorded in each payload (`operation_type`).
- `--max-workers` — parallelism for partition processing.
- `--capture-snapshots` — forces telemetry capture even when not in dry-run mode.
- `--summary` — writes aggregated summary JSON for CI artifacts.

### 2.2 Validate recommendations against pytest subsets

```powershell
pwsh> "C:/Program Files/Python313/python.exe" scripts/run_self_opt_validation.py \`
    --payload-root results/self_optimization/paley_v3 \`
    --report results/self_optimization_validation.json \`
    --fail-on-regression
```

Important arguments:
- `--payload-root` — root directory containing seed folders (`seed_alpha/**.json`).
- `--report` — summary output (defaults to `results/self_optimization_validation.json`).
- `--fail-on-regression` — exits with status 1 if any pytest subset fails.
- `--pytest-cmd` / `--pytest-args` — optional overrides for custom runners.

---

## 3. Directory Layout

```
results/
  self_optimization/
    seed_alpha/
      paley_partition_0001.json
      paley_partition_0001.json.sha256
      ...
    seed_beta/
      ...
    paley_v3_summary.json
  self_optimization_validation.json
```

- **Seed folders** (e.g., `seed_alpha`) group payloads produced in a single runner invocation.
- **Partition payloads** encode dry-run recommendations, telemetry, and candidate operator plans.
- **`.sha256`** files mirror payloads for integrity verification.
- **Summary JSON** aggregates per-run statistics (copied to CI artifacts).
- **Validation report** lists every discovered payload, mapped tests, and exit codes.

---

## 4. JSON Schemas (Field Reference)

### 4.1 Partition payload (`results/self_optimization/**/paley_partition_XXXX.json`)

```json
{
  "metadata": {
    "operation_type": "paley_partition",
    "partition_id": "0001",
    "manifest": "manifests/paley/_manifest.json",
    "seed": 41,
    "sequence": ["AL", "UM", "IL", "SHA"]
  },
  "engine": {
    "dry_run": true,
    "operator_strategy_plan": {
      "per_partition": {...},
      "stabilizers": ["IL", "THOL"],
      "resonance_budget": 0.18
    },
    "candidate": {...},
    "telemetry_snapshots": [
      {
        "partition_id": "0001",
        "delta_phi_s": 0.42,
        "delta_c": +0.07,
        "fields": {"phi_s": 0.63, "grad_phi": 0.12, "k_phi": 1.46, "xi_c": 0.88}
      }
    ]
  },
  "telemetry": {
    "delta_phi_s": 0.42,
    "delta_c": 0.07,
    "coherence": 0.82,
    "sense_index": 0.76
  },
  "candidate_factors": [ ... ]
}
```

Field guarantees:
- `metadata.operation_type` always filled by the runner.
- `engine.operator_strategy_plan` is `null` if no promotable strategy was produced.
- `telemetry` merges manifest-level telemetry with per-partition measurements.
- `telemetry_snapshots` exist whenever `--capture-snapshots` or dry-run mode is active.

### 4.2 Validation summary (`results/self_optimization_validation.json`)

```json
{
  "payload_root": "results/self_optimization/paley_v3",
  "total_recommendations": 12,
  "status_counts": {"validated": 8, "regressed": 2, "pending": 2},
  "test_runs": [
    {
      "tests": ["factorization-lab/tests/test_spectral_paley.py"],
      "exit_code": 0,
      "status": "validated",
      "count": 8
    }
  ],
  "results": [
    {
      "path": "results/self_optimization/.../paley_partition_0001.json",
      "operation_type": "paley_partition",
      "tests": ["factorization-lab/tests/test_spectral_paley.py"],
      "status": "validated",
      "exit_code": 0,
      "metadata": {"partition_id": "0001", ...}
    },
    {
      "operation_type": "unknown",
      "tests": [],
      "status": "pending"
    }
  ]
}
```

Status meanings:
- `validated` — pytest subset passed.
- `regressed` — pytest subset failed (exit code != 0).
- `pending` — no mapped tests for `operation_type`; requires manual review or new OPERATION_TESTS entry.

---

## 5. Promotion Guidance (`operator_strategy_plan`)

1. **Confirm validation status** — only promote partitions marked `validated`.
2. **Inspect plan fidelity** — ensure `engine.operator_strategy_plan` includes AL/IL/RA/SHA coverage expected by [`docs/OPERATOR_STRATEGY_SPEC.md`](OPERATOR_STRATEGY_SPEC.md).
3. **Telemetry guardrails** — ΔΦ_s < 0.6 and ΔC(t) ≥ +0.05 preferred. Reject payloads that violate structural thresholds defined in `docs/STRUCTURAL_FIELDS_TETRAD.md`.
4. **Embed plan** — copy the `operator_strategy_plan` block into the target manifest or orchestration layer (e.g., `tnfr_factorization.spectral_paley` per-partition registry).
5. **Record provenance** — append manifest ID, git SHA, and validation report path to the promotion log so downstream tooling can trace decisions.

> **Tip**: `factorization-lab/tests/test_spectral_paley.py` asserts that promoted plans expose `per_partition` entries. Use the test suite to catch incomplete promotions before merging.

---

## 6. Review Checklist

| Step | Purpose | Command / Artifact |
|------|---------|--------------------|
| Telemetry sanity | Confirm Φ_s, |∇φ|, K_φ, ξ_C remain within canonical bounds | Inspect `telemetry_snapshots` or run `tnfr.validation.aggregator` on the source graph |
| ΔΦ_s / ΔC(t) trend | Ensure coherence gains match potential drift | Plot via dashboard instructions (§7) |
| Operator coverage | Verify sequences comply with U1-U6 grammar | Re-run `tnfr.validation.grammar.collect_grammar_errors` on promoted sequences |
| Validation sweep | Ensure targeted pytest suites cover new payloads | `python scripts/run_self_opt_validation.py --fail-on-regression` |
| Regression nets | Keep smoke targets green | `./make.cmd smoke-tests` plus `pytest tests/scripts/test_run_self_opt_validation.py` |
| Wiring audit | Confirm manifests / plan registries updated | `git diff` on manifest + `factorization-lab` modules |

Document review outcomes next to each partition ID in the manifest summary so auditors can replay the decision process.

---

## 7. Dashboard Guidance (ΔΦ_s vs ΔC(t))

To visualize each partition's structural movement, extend the existing `HealthVisualizationDashboard` (see `docs/STRUCTURAL_HEALTH.md`) with an overlay that plots ΔΦ_s on the x-axis and ΔC(t) on the y-axis:

```python
from docs.examples.structural_dashboard import HealthVisualizationDashboard
import pandas as pd

def plot_partition_dispersion(summary_json: str):
    payload = json.loads(Path(summary_json).read_text())
    rows = []
    for entry in payload["results"]:
        tel = entry["metadata"].get("telemetry") or {}
        rows.append({
            "partition": entry["metadata"].get("partition_id"),
            "delta_phi_s": tel.get("delta_phi_s"),
            "delta_c": tel.get("delta_c"),
            "status": entry["status"] or "pending",
        })
    df = pd.DataFrame(rows).dropna()
    HealthVisualizationDashboard().scatter_delta_phi_vs_coherence(df)
```

- **Color code** points by `status` so regressed partitions surface immediately.
- **Threshold lines**: add vertical line at ΔΦ_s = 1.0 and horizontal line at ΔC(t) = 0 to emphasize acceptable quadrants.
- **Reuse existing charts** like coherence trajectories or Φ_s histograms by feeding the same dataframe.

Store rendered dashboards under `results/reports/self_opt/` to accompany validation reports.

---

## 8. References

- [`scripts/run_self_optimization.py`](../scripts/run_self_optimization.py)
- [`scripts/run_self_opt_validation.py`](../scripts/run_self_opt_validation.py)
- [`docs/OPERATOR_STRATEGY_SPEC.md`](OPERATOR_STRATEGY_SPEC.md)
- [`docs/STRUCTURAL_HEALTH.md`](STRUCTURAL_HEALTH.md)
- [`docs/STRUCTURAL_FIELDS_TETRAD.md`](STRUCTURAL_FIELDS_TETRAD.md)

Keeping this pipeline documented ensures TNFR recommendations remain reproducible, physics-aligned, and ready for integration into production operator strategies.
