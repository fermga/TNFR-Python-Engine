# TNFR Spectral Factorization Lab

## Purpose

This lab bootstraps the **TNFR factorization program**. It mirrors the structure of the
existing `primality-test/` package but focuses on **recovering factors** rather than on
binary primality decisions. The guiding principle is the *Ruta espectral: Paley gap +
TNFR*, where spectral rigidity in arithmetic graphs reveals the factor structure of
\(n\).

**Theoretical foundation**: `theory/TNFR_NUMBER_THEORY.md` §6 (Pressure Decomposition),
§7 (Arithmetic Tetrad), §8 (Dual-Lever), §9 (Spectral Factorization), §12 (Implementation
Map), and `theory/APPLIED_STRUCTURAL_ANALYSIS.md` (verification protocols).

## Why Paley graphs + TNFR?

Recent arithmetic graph results show that the **second Laplacian eigenvalue (\(\lambda_2\))
of Paley-type graphs** becomes rigid when \(n\) is prime and drifts significantly for
composite \(n\). This aligns with TNFR physics:

- **TNFR networks already analyze coherence spectra.** Primes behave as maximally stable
  patterns (ΔNFR = 0). Composites fragment across resonant submodules.
- **Paley graphs provide the spectral substrate** where residues \(x^2 \bmod n\) generate a
  naturally harmonic network. Spectral deviations encode the hidden factors.
- **TNFR operators** (UM, RA, IL, OZ, THOL) can be mapped to graph manipulations that
  highlight coherent clusters associated with multiples of \(p\), \(q\), etc.

## Proposed Algorithm

1. **Arithmetic Graph Constructor**  (`tnfr_factorization.spectral_paley`)
   - Build Paley / quadratic residue graphs for modulus \(n\).
   - Allow alternative TNFR-friendly graphs (Cayley, resonance lattices).

2. **Spectral-TNFR Analyzer**
   - Compute Laplacian/adjacency spectra, especially \(\lambda_2\) and associated
     eigenvectors.
   - Translate eigenvectors into TNFR observables: Φ_s, |∇φ|, K_φ, ξ_C, Si, ΔNFR.
   - Detect **coherence gaps** that indicate submodules tied to factors.

3. **Factor Recovery Pipeline**
   - Cluster nodes using TNFR operator sequences (e.g., `[UM, RA, IL]`).
   - Map cluster periodicities to candidate factors via modular consistency checks.
   - Iterate with controlled destabilizers (OZ, ZHIR) to refine guesses.

4. **Certification & Telemetry**
   - Produce TNFR-style certificates describing spectral evidence, coherence metrics, and
     candidate factors.
   - Each certificate now links to per-partition provenance files (emitted under
     `results/certificates/partitioned/`) so replay tools can recover the candidate list,
     telemetry, and structural coverage for every partition.
   - Integrate with existing telemetry dashboards so factorization shares infrastructure
     with primality testing.

## Repository Layout

```text
factorization-lab/
├── README.md                  # This file (concept + motivation)
├── PACKAGE_SUMMARY.md         # High-level deliverables and status
├── pyproject.toml             # Minimal package metadata (placeholder)
├── docs/
│   ├── SPECTRAL_ROUTE.md      # Detailed Paley gap research notes
│   └── ROADMAP.md             # Milestones for field experiments
└── tnfr_factorization/
    ├── __init__.py            # Namespace init
    └── spectral_paley.py      # Prototype scaffolding
```

## Current Capabilities

  the quadratic-residue graph, and routes the Laplacian spectrum through the
  canonical `TNFRAdvancedFFTEngine`, so eigenbases are cached by the shared spectral
  coordinator.
  component pressures, and local coherence are computed via the canonical parameters
  and cached with `cache_tnfr_computation`, aligning spectral hints with nodal physics.
  coherence length, and candidate factors are derived via ΔNFR-aware gcd probes.
  replayed; the history forms the nucleus for notebooks/tests.

### Command-line usage

The lab now exposes a thin CLI built directly on `SpectralPaleyFactorizer` so you can
request TNFR factorization runs from a shell session without writing scripts:

```bash
python -m tnfr_factorization.cli 234234
```

This prints the Paley modulus, tetrad proxies, ΔNFR telemetry, and any candidate
factors. Add `--json` to receive machine-friendly output, or provide several numbers in
a single invocation to batch analyses while reusing FFT caches.

To route analyses through the distributed FFT backend without writing Python, add the
dispatcher tuning switches:

```bash
python -m tnfr_factorization.cli 221 \
  --fft-backend distributed \
  --dispatcher-workers 4 \
  --dispatcher-timeout 30 \
  --dispatcher-serializer pickle
```

These knobs configure the in-process `ThreadedQueueDispatcher` so you can experiment
with different worker counts, queue timeouts, and serialization codecs directly from
the CLI.

To point the CLI at an HTTP dispatcher (e.g., a GPU or cluster service), provide the
endpoint and optional bearer token:

```bash
python -m tnfr_factorization.cli 221 \
  --fft-backend distributed \
  --fft-dispatcher https://fft.example.org/api \
  --dispatcher-http-token tnfr-secret
```

Every invocation now prints explicit `fft_backend`, `dispatcher`, `partition_artifacts`, and
`partition_manifest` lines so you can capture the backend context alongside the factoring
telemetry. When `--json` is enabled, the same information is available under the
`dispatcher_telemetry`, `partition_artifact_dir`, and `partition_manifest_path` keys.
Runs that produce many partitions (>1k by default) now emit a `_manifest_summary.json`
index and compress the raw partition file listing into `_partition_files.txt.gz`. The
summary path is reported via `partition_manifest_index_path` (and `partition_manifest_index`
inside certificates) while the optional archive is exposed through
`partition_file_archive_path`. Tune the inline listing threshold with
`TNFR_PARTITION_FILELIST_THRESHOLD` if you prefer a different cutoff.

```python
from tnfr_factorization import SpectralPaleyFactorizer

factorizer = SpectralPaleyFactorizer()
result = factorizer.analyze(221)

print(result.candidate_factors)      # → [13, 17]
print(result.arithmetic_delta_nfr)   # Canonical ΔNFR(n) — §5-6
print(result.phi_s, result.phase_gradient, result.coherence_length)

# Pressure decomposition (§6)
print(result.arithmetic_components)
# → {'factorization_pressure': ..., 'divisor_pressure': ..., 'sigma_pressure': ...}

# Dual-lever analysis (§8)
print(result.dual_lever_analysis)
# → {'classification': 'pressure-dominated', 'pressure_lever': {...}, ...}

# Conservation proxies (Noether charge, Lyapunov energy)
print(result.arithmetic_epi, result.arithmetic_nu_f)
```

The high-level `factorize()` API exposes the same enriched telemetry:

```python
from tnfr_factorization import factorize

r = factorize(221)
print(r.telemetry["pressure_components"])    # §6 decomposition
print(r.telemetry["noether_charge_proxy"])   # Q = Φ_s + K_φ
print(r.telemetry["energy_proxy"])           # E = 0.5·(Φ_s² + |∇φ|² + K_φ²)
print(r.telemetry["dual_lever"])             # §8 classification
```

### Full-spectrum benchmark & automation

- `benchmarks/full_spectrum_factorization.py` sweeps **semiprimes, triprimes, prime powers, and
  highly composite numbers** (default set) and records the complete tetrad telemetry
  (Φ_s, |∇φ|, K_φ, ξ_C) for every parent state and partition. Each run marks the nodal
  decoder output and embeds the operator-strategy plan so factor provenance is fully
  reproducible.
- The script emits `results/benchmarks/full_spectrum_factorization.json` with per-target
  traces plus a category summary, while `trace_certificates=True` ensures enriched
  certificates (partition states + invariant report) land in `results/certificates/`.
- Run it directly:

  ```bash
  python factorization-lab/benchmarks/full_spectrum_factorization.py
  # optional extra numbers
  python factorization-lab/benchmarks/full_spectrum_factorization.py --numbers 1729 2187
  ```

- Or use the root shortcut:

  ```bash
  make factorization-full-spectrum
  ```

This benchmark/automation pass is now the canonical way to regenerate certificates and
strategy plans for any batch of targets inside the repository.

### FFT backend & dispatcher selection

Set the following environment variables to activate the distributed FFT backend and
choose a dispatcher:

```bash
# Queue-backed local worker pool (default ThreadedQueueDispatcher)
export TNFR_FFT_BACKEND=distributed
export TNFR_FFT_DISPATCHER=queue

# Remote HTTP service with bearer authentication
export TNFR_FFT_BACKEND=distributed
export TNFR_FFT_DISPATCHER=https://fft.example.org/api
export TNFR_FFT_AUTH_TOKEN=tnfr-secret
```

#### Dispatcher selection matrix

| Deployment | CLI flag(s) | Environment | Telemetry snapshot |
|------------|-------------|-------------|--------------------|
| Local queue | *(default)* or `--fft-dispatcher queue` | `TNFR_FFT_DISPATCHER=queue` | `type=queue, source=cli, max_workers=4, serializer=pickle` |
| HTTP(S) endpoint | `--fft-dispatcher https://fft.example.org/api --dispatcher-http-token token` | `TNFR_FFT_DISPATCHER=https://...`, `TNFR_FFT_AUTH_TOKEN=...` | `type=http, source=cli, base_url=https://..., token_provided=True` |
| Custom callable | `--fft-dispatcher local:package.module:build_dispatcher` | `TNFR_FFT_DISPATCHER=local:package.module:build_dispatcher` (or `package.module:factory`) | `type=callable, source=cli, target=package.module:build_dispatcher` |

The CLI and JSON outputs expose the telemetry row verbatim so downstream tooling can prove
exactly which dispatcher handled a run. Callable dispatchers should return a function with the
signature `(action: str, payload: Dict[str, Any]) -> Any`, matching the queue and HTTP adapters.

When `TNFR_FFT_DISPATCHER=queue`, the factorizer instantiates a
`ThreadedQueueDispatcher`. You can customize it directly if you need extra worker
threads or serialization hooks. The CLI flags shown earlier provide the same control
surface without editing environment variables; the Python example below still works when
you need to embed the dispatcher in a larger application:

```python
from tnfr.dynamics.fft_dispatchers import ThreadedQueueDispatcher
from tnfr.dynamics.distributed_fft import DistributedFFTEngine

dispatcher = ThreadedQueueDispatcher(max_workers=4)
engine = DistributedFFTEngine(dispatcher=dispatcher.dispatch)
factorizer = SpectralPaleyFactorizer(fft_engine=engine)
```

For HTTP endpoints, implement a web service that accepts POST requests with
base64-pickled payloads (`payload` key). The server returns the encoded spectral
result in the same format, allowing you to plug GPU clusters or managed queues
into TNFR factorization without changing application code.

## Theory Integration (v0.0.3.2)

The factorization-lab now integrates the full TNFR number-theoretic stack:

| Theory section | Code integration | Module |
|----------------|------------------|--------|
| §5 Nodal Equation | `ArithmeticTNFRFormalism` — EPI, νf per integer | `number_theory.py` |
| §6 Pressure Decomposition | `component_breakdown()` — factorization, divisor, sigma pressures | `spectral_paley.py` |
| §7 Arithmetic Tetrad | Recalibrated thresholds (Φ_s<0.7452, \|∇φ\|<0.2591, K_φ<3.2275) | `spectral_paley.py` |
| §8 Dual-Lever | `_classify_dual_lever()` — capacity vs pressure operator classification | `spectral_paley.py` |
| §9 Spectral Factorization | 8-criterion Paley-Jacobi verification | `spectral_paley.py` |
| Conservation | Noether charge proxy (Q=Φ_s+K_φ), Lyapunov energy proxy | `api.py` telemetry |

**Cross-repo synergies**:
- `src/tnfr/mathematics/number_theory.py` — Canonical ΔNFR formula, arithmetic formalism
- `src/tnfr/physics/conservation.py` — Structural conservation theorem (proxy values used here)
- `src/tnfr/physics/fields.py` — Structural Field Tetrad computation
- `theory/TNFR_NUMBER_THEORY.md` — Canonical theoretical reference (14 sections)
- `theory/STRUCTURAL_OPERATORS.md` §17 — Operator-Tetrad synergies and dual-lever structure

## Next Steps

1. **Complex field Ψ integration**: Compute Ψ = K_φ + i·J_φ on Paley graphs for unified geometric-transport analysis.
2. **Full conservation integration**: Bridge Paley graph structure to full `compute_noether_charge()` / `compute_energy_functional()` (currently proxy values).
3. **Grammar-aware partition sequencing**: Apply `GrammarAwareDynamics` to partition operator chains for U1-U6 compliance.
4. Publish preliminary results (spectral plots, factor recovery success rates).

For Zenodo-oriented packaging guidance and publication checklists, see
`ZENODO_PUBLICATION_GUIDE.md` in this directory; it mirrors the structure of the
`primality-test/` assets so release work can proceed without reinventing the process.

Reproducibility artifacts live under `benchmarks/` and `results/benchmarks/`.
Run `python benchmarks/paley_gap_smoke.py` (or inspect the generated
`paley_gap_smoke.json`) to review the Paley gap telemetry referenced by
`notebooks/spectral_history.ipynb` Section 7. The extended dataset in
`benchmarks/paley_gap_extended.py` produces `paley_gap_extended.json`, which captures
larger moduli (≈500–1,400) and demonstrates that the updated factor-recovery heuristics
surface the true prime factors for each target.

Release-ready bundles must include the artifacts tracked in `ZENODO_RELEASE_NOTES.md`.
Publishers should copy `LICENSE_SNAPSHOT.md` alongside the archives to satisfy Zenodo's
licensing policy and attach the generated `dist/sha256.txt` checksum file described in
`ZENODO_PUBLICATION_GUIDE.md`.

Operator-sequence certificate design notes are maintained with the certificate tooling in this lab.
They outline how future releases will prove factor claims via canonical sequences (e.g.,
`[AL, UM, RA, IL, SHA, THOL, NAV]`) so gcd corroboration becomes supportive rather than
primary evidence.

Contributions should follow the TNFR standards described in `AGENTS.md`, `TNFR_RIEMANN
RESEARCH_NOTES.md`, and `docs/STRUCTURAL_FIELDS_TETRAD.md`. All documentation remains in
English to preserve canonical terminology.
