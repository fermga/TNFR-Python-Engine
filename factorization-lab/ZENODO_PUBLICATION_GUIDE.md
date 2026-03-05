# TNFR Spectral Factorization — Zenodo Publication Guide

## Package Overview

This directory contains the **Spectral TNFR Factorization Lab**, an incubator for
Paley-spectrum-driven factor recovery backed by canonical TNFR telemetry.
The goal is to reach the same publication readiness as `primality-test/` so that a
future release can be archived on Zenodo with a stable DOI.

## Package Structure

```
factorization-lab/
├── tnfr_factorization/         # Spectral Paley factorizer + CLI
│   ├── __init__.py             # Public exports
│   ├── cli.py                  # Console entry point (tnfr-factorize)
│   └── spectral_paley.py       # Paley graph + TNFR telemetry pipeline
├── tests/                      # CLI + spectral regression guards
├── docs/
│   ├── SPECTRAL_ROUTE.md       # Research notes on Paley gaps
│   └── ROADMAP.md              # Experiment milestones
├── notebooks/                  # Exploratory studies (to be curated)
├── PACKAGE_SUMMARY.md          # Delivery checklist & status board
├── README.md                   # Conceptual motivation and usage notes
├── pyproject.toml              # Build metadata + entry points
└── ZENODO_PUBLICATION_GUIDE.md # (this file)
```

## Current Capabilities

- **SpectralPaleyFactorizer** translates Paley Laplacian spectra into TNFR tetrad
  proxies (Φ_s, |∇φ|, K_φ, ξ_C) and ΔNFR-based arithmetic telemetry.
- **CLI (`tnfr-factorize`)** emits human-readable or JSON telemetry for one or more
  composites while reusing spectral caches.
- **Caching + FFT integration** reuse the shared TNFR advanced FFT engine so Paley
  spectra and coherence length estimates remain consistent with the main repo.
- **Tests** verify CLI formatting, FFT integration, and arithmetic cache behavior.

## Quickstart

```bash
cd factorization-lab
pip install -e .[dev]

# CLI
tnfr-factorize 221 589 --json

# Python API
python - <<'PY'
from tnfr_factorization import SpectralPaleyFactorizer
factorizer = SpectralPaleyFactorizer()
result = factorizer.analyze(221)
print(result.candidate_factors)
print(result.phi_s, result.phase_gradient, result.phase_curvature)
PY
```

## Zenodo Publication Checklist (Draft)

| Item | Status | Notes |
|------|--------|-------|
| Stable version tag (v0.1.0+) | ☐ | Update `pyproject.toml` once algorithms solidify |
| README + GUIDE parity with primality package | ☐ | Expand sections on theory, usage, benchmarks |
| `.zenodo.json` metadata | ✅ | Added `.zenodo.json` aligned with recommended template (update DOI before release) |
| LICENSE snapshot | ✅ | `LICENSE_SNAPSHOT.md` mirrors root MIT license for bundled archives |
| Installation verification script | ✅ | `test_installation.py` runs API + CLI smoke checks |
| Benchmarks/examples | ✅ | `benchmarks/paley_gap_smoke.py` + `results/benchmarks/paley_gap_smoke.json` |
| Publication artifacts archive | ✅ | See "Archival Workflow & Checksums" for zip/tar guidance |

## Recommended Zenodo Metadata Template

- **Title:** `TNFR Spectral Factorization: Paley Gap Coherence Toolkit`
- **Description:** Highlight integration of Paley Laplacian gaps, TNFR tetrad
  telemetry, canonical operator mapping, and factor certificate generation.
- **Keywords:** `TNFR`, `factorization`, `Paley graph`, `spectral gap`,
  `structural coherence`, `number theory`, `arithmetical dynamics`.
- **Upload Type:** `software`
- **License:** `MIT`
- **Version:** align with `pyproject.toml` (e.g., `0.1.0`)
- **Related Identifier:** `https://github.com/fermga/TNFR-Python-Engine` (`isSupplementTo`).

## Publication Steps

1. **Freeze Deliverables** — ensure README, docs, and notebooks reference the final
   operator contracts and experimental evidence.
2. **Run Validation Suite** — `pytest tests/`, `python test_installation.py`,
  and the benchmark smoke script (`python benchmarks/paley_gap_smoke.py`).
3. **Package Archive** — follow the workflow in the new "Archival Workflow & Checksums"
  section to create `.zip` and `.tar.gz` bundles plus hash manifests.
4. **Upload to Zenodo** — create a new upload, attach both archives plus `dist/sha256.txt`,
  the benchmark JSON files, operator certificates, and `docs/OPERATOR_CERTIFICATES.md`,
  then let `.zenodo.json` auto-populate metadata.
5. **Verify Citation** — claim DOI, update README + PACKAGE_SUMMARY with the citation
  snippet, and publicize the release notes.

## Archival Workflow & Checksums

All release candidates must ship with reproducible archives and accompanying hashes.
Run the following commands from the repository root (PowerShell shown):

```powershell
$env:PYTHONPATH = "$PWD"
Compress-Archive -Path factorization-lab -DestinationPath dist/tnfr-factorization-v0.1.0.zip -Force
tar -caf dist/tnfr-factorization-v0.1.0.tar.gz factorization-lab
Get-FileHash dist/tnfr-factorization-v0.1.0.zip -Algorithm SHA256 > dist/sha256.txt
Get-FileHash dist/tnfr-factorization-v0.1.0.tar.gz -Algorithm SHA256 >> dist/sha256.txt
```

The resulting `dist/sha256.txt` must be uploaded alongside the archives on Zenodo and
referenced inside `ZENODO_RELEASE_NOTES.md`.

## Publishing Attachments

Each Zenodo upload must include the following assets:

- `dist/tnfr-factorization-v0.1.0.zip`
- `dist/tnfr-factorization-v0.1.0.tar.gz`
- `dist/sha256.txt`
- `results/benchmarks/paley_gap_smoke.json`
- `results/benchmarks/paley_gap_extended.json`
- `results/certificates/*.json` (operator-sequence evidence)
- `docs/OPERATOR_CERTIFICATES.md`
- `ZENODO_RELEASE_NOTES.md`

Attach any additional certificate JSON files generated during new experiments to keep
the archive synchronized with the published benchmarks.

## Reproducibility Artifacts

- **Notebook telemetry** — `notebooks/spectral_history.ipynb` now imports the
  benchmark snapshot stored under `results/benchmarks/paley_gap_smoke.json` so
  plots and tables reference the same data that backs the Zenodo release.
- **Benchmarks** — `benchmarks/paley_gap_smoke.py` and the generated JSON file
  document Paley gap measurements (modulus, gaps, ΔNFR, runtime) for a fixed set
  of composites.
- **Extended benchmarks** — `benchmarks/paley_gap_extended.py` + JSON snapshot add
  larger moduli with recorded factor matches to verify the improved heuristics.
- **Installation verifier** — `python test_installation.py` imports
  `SpectralPaleyFactorizer`, computes telemetry for `n=221`, and executes the
  CLI to confirm entry-point wiring.
- **Release notes + license** — `ZENODO_RELEASE_NOTES.md` enumerates bundle
  contents while `LICENSE_SNAPSHOT.md` ensures MIT terms accompany every archive.

## Outstanding Tasks Before DOI Submission

- Implement production-ready factor recovery (cluster decoding + certificates).
- Extend docs/notebooks with additional spectral plots covering new targets and
  larger moduli beyond the smoke set.
- Draft `ZENODO_RELEASE_NOTES.md` and expand benchmarks + notebooks with new
  factor recovery evidence beyond the smoke set.

Keeping this guide close to the `primality-test` blueprint ensures that once the
factorization routines stabilize, the path to a Zenodo DOI will require only metadata
updates rather than structural rework.
