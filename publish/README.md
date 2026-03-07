# TNFR Publications

This directory contains self-contained, Zenodo-ready publication packages
for TNFR research results.

Each subdirectory is an independent, minimal reproducible unit that can be
deposited as-is on [Zenodo](https://zenodo.org/) or similar archives.

## Publications

| # | Directory | Title | Status |
|---|-----------|-------|--------|
| 1 | `001-structural-conservation/` | A Structural Continuity Law for Grammar-Constrained Dynamics in TNFR | Draft |

## Package Standard

Every publication package must contain:

| File | Purpose |
|------|---------|
| `README.md` | Reproduction instructions (2–3 commands) |
| `CITATION.cff` | Machine-readable citation metadata |
| `preprint.md` | Preprint source (Markdown + LaTeX math) |
| `requirements.txt` | Pinned Python dependencies |
| `src/run_experiment.py` | Single script reproducing the central experiment |
| `results/` | Generated metrics (CSV) and figures (PNG) |
| `tests/` | Claim-level test assertions |

## Editorial Strategy

```
Zenodo (minimal reproducible package)
  → Medium (accessible narrative)
    → Extended preprint (arXiv / journal submission)
```

Each stage builds on the previous one. Claims are intentionally narrow
and experimentally grounded.
