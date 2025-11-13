# Periodic Table Atlas (TNFR)

Status: Preview • Reproducible • Uses CANONICAL Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C)

This example reconstructs element-like rows (Z = 1..10) using radial graphs and reports the Structural Field Tetrad alongside a canonical U6 sequential ΔΦ_s validator. It’s a read-only diagnostic pipeline: no new grammar rules are introduced (U1–U5 remain prescriptive; U6 is telemetry-only).

---

## What it does

- Builds element-like radial topologies via shared helpers:
  - `build_element_radial_graph(Z, seed)` and `build_radial_atom_graph(n_shell, seed)`
- Computes the Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
- Runs a short synthetic activation step (∼[AL, RA, IL]-like) and evaluates the canonical U6 sequential ΔΦ_s check
- Writes HTML for quick browsing plus JSONL/CSV for programmatic analysis

---

## How to run

### From VS Code tasks

Use the Command Palette → "Run Task":

- Run Periodic Table Atlas (script)
- Export Periodic Table Atlas (HTML - classic)

Outputs:
- HTML/CSV/JSONL in `examples/output/`
- Notebook HTML in `results/reports/`

### From the terminal (optional)

```pwsh
# Ensure PYTHONPATH includes src
$env:PYTHONPATH = (Resolve-Path -Path ./src).Path
# Run the script (writes HTML/CSV/JSONL)
& "C:/Program Files/Python313/python.exe" examples/periodic_table_atlas.py
```

---

## Outputs

- HTML: `examples/output/periodic_table_atlas.html`
- JSONL: `examples/output/periodic_table_atlas.jsonl`
- CSV: `examples/output/periodic_table_atlas.csv`
- Grouped CSV (by signature): `examples/output/periodic_table_atlas_by_signature.csv`
- Summary JSON (by signature): `examples/output/periodic_table_atlas_summary.json`

Each row includes:
- Z, seed, topology metadata (shell counts, sizes)
- Tetrad summaries: mean |∇φ|, mean |K_φ|, ξ_C, Φ_s aggregates
- Locality fraction, ΔΦ_s sequential PASS/FAIL
- A descriptive signature (telemetry-only): one of localized-confined, confined, stressed, critical, runaway

The HTML includes:
- A footer table “Summary by Signature” with count, PASS rate, and mean metrics per signature.
- A short legend listing canonical thresholds (|∇φ|, |K_φ|, ΔΦ_s) and a distribution by signature.

Note: Thresholds/messages are centralized in `src/tnfr/telemetry/constants.py`.

---

## Structural Field Tetrad (U6 telemetry)

- Φ_s (global potential): monitor ΔΦ_s < 2.0
- |∇φ| (local desynchronization): < 0.38 for stable operation
- K_φ (geometric curvature): |K_φ| < 3.0; optional multiscale check via `k_phi_multiscale_safety(G, alpha_hint=2.76)`
- ξ_C (coherence length): regime monitoring; warns near criticality

APIs (from `tnfr.physics.fields`):
- `compute_structural_potential(G)`
- `compute_phase_gradient(G)`
- `compute_phase_curvature(G)`, `k_phi_multiscale_safety(G)`
- `estimate_coherence_length(G)`

---

## Sequential ΔΦ_s validator (U6)

The script performs a short synthetic activation and measures ΔΦ_s before/after. PASS indicates confinement (ΔΦ_s below the escape threshold); FAIL flags potential runaway.

- Implementation: shared helper `apply_synthetic_activation_sequence(G, ...)`
- Validation: `src/tnfr/operators/grammar.py` U6 check
- Tests: `tests/examples/test_u6_sequential_demo.py`

---

## Reproducibility & caching

- Seeded builders: deterministic topologies for given (Z, seed)
- Caching: integrated via the repo cache manager in the script to speed up repeated runs
- Tests: `tests/examples/test_periodic_table_basic.py` ensures outputs exist and can be regenerated

---

## Notebook parity

- Notebook: `notebooks/TNFR_Periodic_Table_Atlas.ipynb`
  - Mirrors the script’s telemetry, sequential validator, and exports
  - Use the VS Code task “Export Periodic Table Atlas (HTML - classic)” for a one-click HTML export

---

## Limitations and next steps

- Classification is descriptive: current summaries focus on Tetrad metrics, locality, and ΔΦ_s outcomes; no prescriptive rules beyond U1–U5 are added
- Future enhancements (non-breaking):
  - Family/group labels inferred from structural signatures (mean/var |∇φ|, K_φ hotspots, ξ_C by role)
  - Richer visuals by role/shell in the HTML
  - Expanded tests covering notebook export parity

---

## Related docs

- Canonical Tetrad: `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md`
- Physics module overview: `src/tnfr/physics/README.md`
- Grammar rules: `UNIFIED_GRAMMAR_RULES.md`
