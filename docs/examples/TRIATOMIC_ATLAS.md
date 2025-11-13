# Triatomic Molecule Atlas (TNFR) — Telemetry-only

Status: Preview • Read-only telemetry • Extends Molecule Atlas to triatomics

This example composes three element-like radial graphs and adds coupling edges to emulate bonding (UM/RA at topology level; no operator dynamics executed here). It computes the Structural Field Tetrad and a canonical U6 sequential ΔΦ_s check per molecule.

---

## What it does

- Builds triatomic graphs via `build_triatomic_molecule_graph(Z1, Z2, Z3, seed, bond_links=1, central='B')`
- Computes the Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
- Runs a short synthetic activation ([AL, RA, IL]-like) and evaluates the U6 ΔΦ_s check
- Writes HTML for quick browsing plus JSONL/CSV for analysis, including a `signature` classification
- Adds telemetry-only geometry hints for triatomics (central atom, geometry, angle estimate)
 - HTML exports include a compact Safety Triad panel (telemetry-only) annotating thresholds: ΔΦ_s (confinement), |∇φ| (local stability), |K_φ| (curvature safety), plus a brief dataset summary.

---

## How to run

### From VS Code tasks

Use the Command Palette → "Run Task":

- Run Triatomic Atlas (script)
- Export Triatomic Atlas (HTML - classic)

Outputs:
- HTML/CSV/JSONL in `examples/output/`
- Executed notebook HTML in `results/reports/`

### From the terminal (optional)

```pwsh
$env:PYTHONPATH = (Resolve-Path -Path ./src).Path
& "C:/Program Files/Python313/python.exe" examples/triatomic_atlas.py
```

---

## Outputs

- HTML: `examples/output/triatomic_atlas.html`
- JSONL: `examples/output/triatomic_atlas.jsonl`
- CSV: `examples/output/triatomic_atlas.csv`
- Notebook (parity): `notebooks/TNFR_Triatomic_Atlas.ipynb` → export via task above
 - Notebook (parity): `notebooks/TNFR_Triatomic_Atlas.ipynb` → export via task above
 
See also: `docs/examples/ATOMS_AND_MOLECULES_STUDY.md` for a TNFR framing that links the nodal equation and operator grammar to element-like patterns (H, C, N, O) and a metal-like (Au) network, with runnable script/notebook.

Columns:
- formula (e.g., H2O appears as HOH), central atom (A/B/C used), geometry (linear/bent/unknown), angle_deg (estimated), signature (localized-confined/confined/stressed/critical/runaway), ξ_C, mean |∇φ|, mean |K_φ|, mean path length, local fraction, ΔΦ_s PASS/FAIL, ΔΦ_s value

---

## TNFR view of triatomic “bonding” (read-only)

- UM (Coupling) + RA (Resonance) + IL (Coherence) remain the conceptual triad behind stable links
- In this demo, we reflect UM/RA by adding edges between candidate shell nodes with a selectable central atom (default: B); IL is emulated in the synthetic step reducing |ΔNFR|
- U3 phase verification is conceptually required for real coupling; here we monitor |∇φ|/K_φ as safety telemetry instead of enforcing phase gates
- Geometry tagging is heuristic and telemetry-only: if terminals are identical and central is C (Z=6), classify linear (CO2-like); if central is O (Z=8), classify bent (H2O-like); otherwise unknown. Angle estimates are canonical demo values (180° for linear, ~104.5° for bent). No dynamics are affected.

---

## Next steps

- Minimal geometry: angle-like heuristics for different central atoms (e.g., bent vs linear motifs) — still telemetry-only
- Phase-gated coupling (explicit U3 check) in a controlled example
- Richer visuals: per-atom colors and per-atom sub-telemetry

---

## Notebook parity (reproducible cellbook)

- Path: `notebooks/TNFR_Triatomic_Atlas.ipynb`
- Cells compute the Structural Field Tetrad, perform ΔΦ_s sequential check (U6 telemetry), derive a telemetry-only signature, and save CSV/JSONL summaries to `examples/output/`.
- Export HTML: use the VS Code task “Export Triatomic Atlas (HTML - classic)” to generate `results/reports/TNFR_Triatomic_Atlas.html`.
