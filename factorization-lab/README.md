# TNFR spectral factorization lab

This directory is an experimental research lab for extracting factor candidates
from quadratic-residue graph spectra, partitions and TNFR telemetry. It is not a
proof of a general factorization theorem and does not establish a complexity
improvement over classical algorithms.

The lab distinguishes three outputs:

- `candidate_factors`: candidates produced by the spectral and partition pipeline;
- `tnfr_certified_factors`: candidates satisfying the lab's configured structural
  acceptance criteria;
- arithmetic divisibility or `gcd` checks: the conclusive verification that a
  candidate is a factor of `n`.

Structural certification is diagnostic evidence. It must not be described as a
replacement for arithmetic verification.

## Public API

Run from the lab directory so its local package is importable:

```bash
cd factorization-lab
python -m tnfr_factorization.cli 221
```

```python
from tnfr_factorization import factorize

result = factorize(221, pure=False, trace=False)
print(result.candidate_factors)
print(result.tnfr_certified_factors)
```

`pure=True` disables arithmetic refinement and therefore returns experimental
structural candidates. `trace=True` emits certificate and partition artifacts;
their paths are exposed on `FactorizationResult`.

For lower-level analysis use `SpectralPaleyFactorizer` and
`SpectralAnalysisResult`. The implementation and dataclasses in
`tnfr_factorization/` are the API source of truth.

## Pipeline

1. Construct a quadratic-residue graph for the selected modulus.
2. Compute spectral and structural diagnostics.
3. Partition the graph and infer periodic candidate factors.
4. Apply the configured structural verifier.
5. Optionally refine and verify candidates arithmetically.
6. Record reproducibility metadata when tracing is enabled.

The fields `Phi_s`, phase-gradient magnitude, `K_phi` and `xi_C` retain the
scopes defined by the main engine. Lab thresholds are operational parameters;
they are not universal TNFR constants.

## Verification

```bash
python -m pytest factorization-lab/tests factorization-lab/test_installation.py -q
python factorization-lab/benchmarks/paley_gap_smoke.py
```

Benchmark results apply only to the recorded inputs, environment and seed. The
test suite checks deterministic construction, candidate handling, certificate
serialization and resistance to selected false-positive cases; it does not prove
completeness for arbitrary integers.

## Canonical references

- [TNFR number theory](../theory/TNFR_NUMBER_THEORY.md)
- [Diagnostic and grammar scope](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md)
- [Structural field definitions](../docs/STRUCTURAL_FIELDS_TETRAD.md)
- [Unified grammar](../theory/UNIFIED_GRAMMAR_RULES.md)

The root [LICENSE.md](../LICENSE.md) is authoritative for this repository.
