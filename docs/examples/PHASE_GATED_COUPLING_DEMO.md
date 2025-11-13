# Phase-Gated Coupling Demo (U3) — Telemetry-only

Status: Preview • Read-only telemetry • Explicit U3 phase verification gate

This example demonstrates explicit phase verification (U3) for coupling between two atom-like TNFR graphs.
It runs two scenarios with deterministic phases:

1) In-phase terminals (compatible at threshold 0.9) → coupling edge added
2) Anti-phase terminals (incompatible at threshold 0.9) → coupling edge blocked

We compute the Structural Field Tetrad telemetry and perform a sequential ΔΦ_s check (U6, read-only) in each scenario.

HTML exports include a compact Safety Triad panel (telemetry-only) annotating thresholds: ΔΦ_s (confinement), |∇φ| (local stability), |K_φ| (curvature safety), plus a brief dataset summary.

---

## Run it

- VS Code Task: "Run Phase-Gated Coupling Demo (script)"
- VS Code Task: "Export Phase-Gated Coupling Demo (HTML - classic)" (cellbook)
- Outputs: `examples/output/phase_gated_coupling_demo.html` and `.jsonl` (script), and `results/reports/TNFR_Phase_Gated_Coupling_Demo.html` (notebook)

Optional terminal:

```pwsh
$env:PYTHONPATH = (Resolve-Path -Path ./src).Path
& "C:/Program Files/Python313/python.exe" examples/phase_gated_coupling_demo.py
```

---

## What it shows

- U3: Coupling edge is created only if `is_phase_compatible(θ_a, θ_b, threshold)` is True
- Telemetry-only: No prescriptive changes to dynamics; ΔΦ_s, |∇φ|, K_φ, ξ_C are measured and reported
- Deterministic phases highlight the contrast between compatible vs incompatible cases
- Quick plots: nodes colored by `atom` tag (A/B), titles annotate whether the gated edge was added

---

## Notes

- This demo uses `tnfr.metrics.phase_compatibility.is_phase_compatible` and `compute_phase_coupling_strength`
- The threshold is set to 0.9 by default (very strict) to clearly show gating behavior
- For richer experiments, integrate this gate into operator sequences; here it remains a minimal, isolated showcase
