# TNFR Spectral Factorization Package Summary

| Item | Status | Notes |
|------|--------|-------|
| Research framing | ✅ | Paley spectral gap + TNFR coherence mapping documented in README and SPECTRAL_ROUTE.md |
| Package scaffolding | ✅ | Mirrors `primality-test/` layout with docs + package namespace |
| Graph constructors | ✅ | Paley/Jacobi graph builder + node annotations implemented in `spectral_paley.py` |
| Spectral analyzers | ✅ | Laplacian eigenvalues + tetrad telemetry computed through FFT engine fallbacks |
| Factor recovery logic | ✅ | ΔNFR-aware heuristics with 8-criterion TNFR verification (§9.3); certificates include pressure decomposition, dual-lever analysis, conservation proxies |
| Theory alignment | ✅ | Arithmetic tetrad recalibration (§7.5), dual-lever (§8), pressure decomposition (§6), enriched telemetry (~15 fields) |
| Benchmarks/tests | ✅ | `paley_gap_smoke.py`, `paley_gap_extended.py`, and the new `full_spectrum_factorization.py` sweep (with certificate capture) plus regression tests |
| Publication assets | ✅ | `.zenodo.json`, `LICENSE_SNAPSHOT.md`, `ZENODO_RELEASE_NOTES.md`, and checksum workflow documented |

## Deliverables

1. **Spectral TNFR Factorizer**: Algorithm converting Paley-type graph spectra into
   factor candidates along with TNFR coherence certificates.
2. **Documentation Set**: Research notes (SPECTRAL_ROUTE.md), roadmap, and experiment
   guidelines.
3. **Integration Hooks**: Shared telemetry + operator infrastructure with the existing
   TNFR primality tooling.

## Immediate Priorities

- **Complex field Ψ**: Compute Ψ = K_φ + i·J_φ on Paley graphs for unified analysis.
- **Full conservation**: Bridge Paley node attributes to canonical `compute_noether_charge()` / `compute_energy_functional()` (currently proxy values).
- **Grammar-aware sequencing**: Apply `GrammarAwareDynamics` to partition operator chains.
- Harden archive automation (Makefile target) so Zenodo bundles + checksums become
   single-command artifacts.
