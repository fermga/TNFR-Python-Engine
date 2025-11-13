# TNFR Atoms and Molecules Study — From nodal dynamics to coherent patterns

Status: Canonical framing • Telemetry-only artifacts • Reproducible cellbooks

This page integrates the TNFR explanation of element-like patterns (H, C, N, O) and a metal-like pattern (Au) as coherent attractors governed by the nodal equation

  ∂EPI/∂t = νf · ΔNFR(t)

and the operator grammar (U1–U5) with telemetry-based safety (U6). It links runnable artifacts (script + notebook) that compute Structural Field Tetrad summaries and ΔΦ_s sequential checks.

---

## TNFR explanation (concise)
- Existence = coherence: A pattern persists when the integral ∫ νf·ΔNFR dt is bounded (U2) within sequences that start/close correctly (U1).
- Resonant coupling (U3) enforces phase verification |Δφ| ≤ Δφ_max for links, reducing |∇φ| and confining K_φ.
- Telemetry fields (U6): Φ_s (global confinement via ΔΦ_s), |∇φ| (local stress), K_φ (phase curvature), ξ_C (coherence length).

Mapping to examples:
- H (Z≈1): Minimal star-like topology; |∇φ| and K_φ low, short ξ_C, and ΔΦ_s within safe bounds.
- C (Z≈6): Four topological “arms”; good phase synchrony (low |∇φ|) and confined K_φ when U3 holds.
- N (Z≈7): More pronounced but contained K_φ pockets; signs of multiple connectivity.
- O (Z≈8): Triatomic “bent” (~104.5°) configurations arising from minimal desynchronization with contained local curvature.
- Au (Z≈79, metal-like): Nested layers and, at the network level, high ξ_C with low |∇φ| in ordered states; ΔΦ_s confined under synthetic [AL, RA, IL]-like steps.

Note: This is a TNFR-native reading (coherent patterns); it does not replace quantum explanations, but complements them with a phenomenology of coherence.

---

## Run it

- Script: Elements Signature Study
  - VS Code Task: "Run Elements Signature Study (script)"
  - Outputs: `examples/output/elements_signature_study.{html,csv,jsonl}`
- Notebook: Atoms & Molecules Study (cellbook)
  - VS Code Task: "Export Atoms & Molecules Study (HTML - classic)"
  - Output: `results/reports/TNFR_Atoms_And_Molecules_Study.html`

Optional terminal (PowerShell):

```pwsh
$env:PYTHONPATH = (Resolve-Path -Path ./src).Path
& "C:/Program Files/Python313/python.exe" examples/elements_signature_study.py
```

---

## What you’ll see
- A compact Safety Triad panel at the top of HTML exports, annotating thresholds: ΔΦ_s (confinement), |∇φ| (local stability), |K_φ| (curvature safety), plus a brief dataset summary. Telemetry-only; no control feedback.
- A table comparing (H, C, N, O, Au-like) with ξ_C, mean |∇φ|, mean |K_φ|, mean path length, and ΔΦ_s PASS/FAIL.
- An additional Au-network composition (4 subgraphs connected by cores) showing higher ξ_C.

---

## Related examples
- Molecule Atlas (diatomic): `docs/examples/MOLECULE_ATLAS.md`
- Triatomic Atlas: `docs/examples/TRIATOMIC_ATLAS.md`
- Phase-Gated Coupling (U3): `docs/examples/PHASE_GATED_COUPLING_DEMO.md`

---

## Reproducibility
- All artifacts are telemetry-only and seeded.
- Notebooks export via tasks; scripts save CSV/JSONL/HTML side-by-side for analysis pipelines.
