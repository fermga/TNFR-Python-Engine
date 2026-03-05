# TNFR Spectral Factorization Package Summary

| Item | Status | Notes |
|------|--------|-------|
| Research framing | ✅ | Paley spectral gap + TNFR coherence mapping documented in README and SPECTRAL_ROUTE.md |
| Package scaffolding | ✅ | Mirrors `primality-test/` layout with docs + package namespace |
| Graph constructors | ✅ | Paley/Jacobi graph builder + node annotations implemented in `spectral_paley.py` |
| Spectral analyzers | ✅ | Laplacian eigenvalues + tetrad telemetry computed through FFT engine fallbacks |
| Factor recovery logic | 🟡 | Enhanced ΔNFR-aware heuristics surface true factors for benchmark semiprimes; certificates TBD |
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

- Add operator-driven cluster decoding so factor recovery emits certificates rather than
   gcd corroboration alone.
- Extend benchmarks beyond biprimes (e.g., triprimes, power composites) and chart
   coherence responses.
- Harden archive automation (Makefile target) so Zenodo bundles + checksums become
   single-command artifacts.
