# Operator Completeness Search (Research)

This document describes the end-to-end pipeline we use to probe operator coverage in TNFR and surface potential "gaps"—regions in the structural dynamics space that are under-expressed by existing canonical operators or compositions.

## Physical Basis

We ground the analysis in the TNFR nodal equation:

∂EPI/∂t = νf · ΔNFR(t)

- Form (EPI) evolves only via structural operators.
- Frequency (νf) is the reorganization capacity (Hz_str).
- ΔNFR is the internal reorganization gradient (structural pressure).

Sequences must obey the unified grammar (U1–U6). In particular:
- U1: Initiate with a generator and end with a closure.
- U2: Destabilizers (OZ/ZHIR/VAL) require stabilizers (IL/THOL).
- U3: Coupling and resonance require phase compatibility.
- U4–U6: Bifurcation control, multi-scale coherence, and structural potential confinement.

We monitor the Structural Field Tetrad (canonical):
- Φ_s (global potential), |∇φ| (phase gradient), K_φ (phase curvature), ξ_C (coherence length)

## Pipeline Overview

Implemented in `notebooks/Operator_Completeness_Search.ipynb`.

- Canonical generator: Produces sequences that satisfy U1–U4 using the in-repo validator.
- Measurement: Applies sequences to a base network and records changes in:
  - coherence, sense_index, Φ_s, |∇φ|, K_φ, ξ_C, phase_sync, epi_mag
- Caching: Per-sequence JSON cache keyed by SHA1 to avoid recomputation.
- Chunking: 10×1000 (configurable) with per-chunk manifests and a global manifest.
- Grammar slices: Heuristics to tag sequences by destabilization, stabilization, coupling, and length.
- Analytics:
  - Slice summaries and heatmaps
  - PCA (≥3 PCs) + KMeans with silhouette selection
  - Gap heuristics (outliers, sparsity, contract checks)

## Artifacts

Outputs are written under:

- results/operator_search/full/
  - effects_chunk_*.parquet (or .csv)
  - effects_all.parquet (or .csv)
  - grammar_slices_chunk_*.parquet; grammar_slices_all.parquet
  - heatmap_grad_phi_by_destab_stab.png
  - heatmap_coherence_by_len_coupling.png
  - pca_explained_variance.json
  - kmeans_silhouette.json
  - kmeans_centroids.parquet (or .csv)
  - pca_kmeans_scatter.png
  - manifest_chunk_*.json, manifest_all.json

## How to Reproduce

- Open the notebook and set `USE_REAL_TNFR=True`, `QUICK_RUN=False`.
- Run the chunked driver cell. The disk cache accelerates re-runs.
- Heatmaps and clustering cells export figures and tables automatically.

## Interpreting Results

- Heatmaps:
  - grad_phi mean by {destabilizer × stabilizer-after-destabilizer} highlights local stress containment (U2).
  - coherence mean by {sequence length × coupling} surfaces stability vs connectivity trade-offs.
- PCA + KMeans:
  - Explained variance summarizes dominant axes in structural effects.
  - Silhouette-driven k reveals coherent clusters of operator effects; centroids in original feature space aid interpretation.
- Gaps:
  - Outliers (z-score), sparse regions (KDE), and contract deviations suggest under-expressed dynamics that may motivate operator refinements or new families—subject to canonicity checks.

## Canonicity Checks Before Proposals

- Map any proposed behavior to existing operators or justify a new operator physically.
- Demonstrate preservation of invariants and U1–U6 compliance.
- Provide tests: monotonicity (coherence), bifurcation handlers, propagation, multi-scale, reproducibility.

## References

- AGENTS.md (Canonical guidance)
- UNIFIED_GRAMMAR_RULES.md (derivations)
- docs/TNFR_FORCES_EMERGENCE.md (field emergence)
- src/tnfr/* (operators, dynamics, metrics)
