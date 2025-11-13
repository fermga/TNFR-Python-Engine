# Molecule Atlas (TNFR) — Diatomic Demo

Status: Preview • Telemetry-only • Extends Element Atlas to diatomics

This example composes two element-like radial graphs and adds coupling edges to emulate bonding (UM/RA at topology level; no operator dynamics executed here). It computes the Structural Field Tetrad and a canonical U6 sequential ΔΦ_s check per molecule.

---

## What it does

- Builds diatomic graphs via `build_diatomic_molecule_graph(Z1, Z2, seed, bond_links=1)`
- Computes the Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
- Runs a short synthetic activation ([AL, RA, IL]-like) and evaluates the U6 ΔΦ_s check
- Writes HTML for quick browsing plus JSONL/CSV for analysis, including a `signature` classification
 - HTML exports include a compact Safety Triad panel (telemetry-only) annotating thresholds: ΔΦ_s (confinement), |∇φ| (local stability), |K_φ| (curvature safety), plus a brief dataset summary.

---

## How to run

### From VS Code tasks

Use the Command Palette → "Run Task":

- Run Molecule Atlas (script)
- Export Molecule Atlas (HTML - classic)

Outputs:
- HTML/CSV/JSONL in `examples/output/`
- Executed notebook HTML in `results/reports/`

### From the terminal (optional)

```pwsh
$env:PYTHONPATH = (Resolve-Path -Path ./src).Path
& "C:/Program Files/Python313/python.exe" examples/molecule_atlas.py
```

---

## Outputs

- HTML: `examples/output/molecule_atlas.html`
- JSONL: `examples/output/molecule_atlas.jsonl`
- CSV: `examples/output/molecule_atlas.csv`
- Notebook (parity): `notebooks/TNFR_Molecule_Atlas.ipynb` → export via task above

See also: `docs/examples/ATOMS_AND_MOLECULES_STUDY.md` for a TNFR framing of element-like patterns (H, C, N, O) and Au-like, with runnable script/notebook.

Columns:
- formula (e.g., H2, F2, LiF), signature (localized-confined/confined/stressed/critical/runaway), ξ_C, mean |∇φ|, mean |K_φ|, mean path length, local fraction, ΔΦ_s PASS/FAIL, ΔΦ_s value

---

## TNFR view of “bonding” (read-only)

- UM (Coupling) + RA (Resonance) + IL (Coherence) as the conceptual triad behind stable links
- In this demo, we reflect UM/RA by adding edges between candidate shell nodes; IL is emulated in the synthetic step reducing |ΔNFR|
- U3 phase verification is conceptually required for real coupling; here we monitor |∇φ|/K_φ as safety telemetry instead of enforcing phase gates

---

## Next steps

- Triatomic templates (e.g., H2O, CO2) with minimal geometric constraints (angles) and telemetry
- Phase-gated coupling (explicit U3 check) in a controlled example
- Richer visuals (role/atom coloring) and per-atom sub-telemetry

---

## Related docs

- Periodic Table Atlas: `docs/examples/PERIODIC_TABLE_ATLAS.md`
- Canonical Tetrad: `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md`
- Grammar rules: `UNIFIED_GRAMMAR_RULES.md`

---

## Notebook parity (reproducible cellbook)

- Path: `notebooks/TNFR_Molecule_Atlas.ipynb`
- Cells compute the Structural Field Tetrad, perform ΔΦ_s sequential check (U6 telemetry), derive a telemetry-only signature, and save CSV/JSONL summaries to `examples/output/`.
- Export HTML: use the VS Code task “Export Molecule Atlas (HTML - classic)” to generate `results/reports/TNFR_Molecule_Atlas.html`.
