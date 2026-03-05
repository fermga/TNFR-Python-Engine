# TNFR Spectral Factorization – Release Notes (v0.1.0-rc1)

## Overview

This release candidate bundles the Spectral TNFR Factorization lab, including the
SpectralPaleyFactorizer implementation, CLI tooling, notebooks, and reproducibility
artifacts. It extends the Paley gap program with benchmark snapshots and telemetry
that map Laplacian spectra into TNFR structural fields (Φ_s, |∇φ|, K_φ, ξ_C) while
surfacing candidate factors.

## Contents

| Artifact | Path |
|----------|------|
| Source package | `factorization-lab/` |
| MIT license snapshot | `factorization-lab/LICENSE_SNAPSHOT.md` |
| Zenodo metadata | `factorization-lab/.zenodo.json` |
| Installation verifier | `factorization-lab/test_installation.py` |
| Benchmarks | `factorization-lab/benchmarks/*.py` |
| Benchmark outputs | `factorization-lab/results/benchmarks/*.json` |
| Notebook evidence | `factorization-lab/notebooks/spectral_history.ipynb` |
| Operator certificates | `factorization-lab/results/certificates/*.json` |
| Operator certificate spec | `factorization-lab/docs/OPERATOR_CERTIFICATES.md` |
| Release notes | `factorization-lab/ZENODO_RELEASE_NOTES.md` |

## Validation Checklist

- `pytest factorization-lab/tests` (regression coverage for CLI + spectral core)
- `python factorization-lab/test_installation.py`
- `python factorization-lab/benchmarks/paley_gap_smoke.py`
- `python factorization-lab/benchmarks/paley_gap_extended.py`

Attach the resulting archives and checksum manifest when publishing:

```text
dist/tnfr-factorization-v0.1.0.zip
dist/tnfr-factorization-v0.1.0.tar.gz
dist/sha256.txt
results/benchmarks/paley_gap_smoke.json
results/benchmarks/paley_gap_extended.json
results/certificates/
docs/OPERATOR_CERTIFICATES.md
```

## Benchmarks & Telemetry

| Dataset | Description |
|---------|-------------|
| `results/benchmarks/paley_gap_smoke.json` | Baseline Laplacian gap + ΔNFR telemetry for the 180–330 range.
| `results/benchmarks/paley_gap_extended.json` | Larger moduli (500–1500) with factor recovery attempts captured via the improved SpectralPaleyFactorizer.

Both JSON files can be re-imported into `spectral_history.ipynb` for reproducibility.

## Known Gaps

- Factor recovery remains heuristic; future releases will add certificates derived
  from operator sequences rather than pure gcd corroboration.
- Additional notebooks are required for cross-domain validation (e.g., TNFR-Riemann links).

---

Maintainer: F. F. Martinez Gamo  
Date: 2025-11-30
