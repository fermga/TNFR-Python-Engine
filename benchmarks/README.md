# Microbenchmarks

This directory hosts **targeted microbenchmarks** for hot paths in the TNFR
engine. Each script isolates a specific optimisation and contrasts it with a
reference implementation. Use them to validate performance regressions after
refactoring core operators.

## Usage

```bash
PYTHONPATH=src python benchmarks/<script>.py
```

* Set `PYTHONPATH=src` so the interpreter can import the in-repo `tnfr`
  package without installing it.
* Scripts emit warnings for optional dependencies (NumPy, YAML, orjson). Those
  warnings are harmless during ad-hoc benchmarks.

## Maintained benchmarks

| Script | Focus | Notes |
| --- | --- | --- |
| `cached_abs_max.py` | Cache-aware updates for absolute maxima (`tnfr.alias.set_attr_with_max`). | Demonstrates how cached maxima avoid scanning the graph via `multi_recompute_abs_max` on every assignment. |
| `collect_attr.py` | Vectorised collection of nodal attributes (`tnfr.alias.collect_attr`). | Requires NumPy; the script exits gracefully when the module is unavailable. |
| `contractive_vs_unitary.py` | Unitary vs. Lindblad ΔNFR evolution (`tnfr.mathematics.MathematicalDynamicsEngine` vs. `ContractiveDynamicsEngine`). | Compares wall-clock timings and Frobenius contractivity after repeated semigroup steps. |
| `evolution_backend_speedup.py` | Backend comparison for evolution engines (`MathematicalDynamicsEngine`, `ContractiveDynamicsEngine`). | Measures per-backend timings, speed-up ratios, and persists JSON artefacts for reproducibility. |
| `default_compute_delta_nfr.py` | Core ΔNFR update speed (`tnfr.dynamics.default_compute_delta_nfr`). | Runs multiple passes on random graphs and reports best/median/mean/worst timings. Accepts `--profile` to dump per-function timings. |
| `compute_dnfr_benchmark.py` | `_compute_dnfr` vectorised vs. fallback execution. | Explores how graph size/density impacts the NumPy and pure-Python paths, reporting summary stats and speed-up ratios. |
| `compute_si_profile.py` | Sense Index profiling (`tnfr.metrics.sense_index.compute_Si`). | Captures cProfile stats for NumPy and pure-Python runs, exporting `.pstats` or JSON summaries. |
| `full_pipeline_profile.py` | Full telemetry + ΔNFR pipeline profiling (`compute_Si`, `_prepare_dnfr_data`, `_compute_dnfr_common`, `default_compute_delta_nfr`). | Produces paired `.pstats` and JSON reports for vectorised and fallback runs, supports multi-configuration chunk/worker sweeps, and records per-operator wall-clock summaries. |
| `neighbor_phase_mean.py` | Fast phase averaging for neighbourhoods (`tnfr.metrics.trig.neighbor_phase_mean`). | Includes a `NodeNX`-based reference to highlight the benefit of the shared `trig_cache` module. |
| `prepare_dnfr_data.py` | ΔNFR data preparation reuse (`tnfr.dynamics._prepare_dnfr_data`). | Exercises cache reuse when assembling phase/EPI/νf arrays. |
| `neighbor_accumulation_comparison.py` | Broadcast neighbour accumulation (`tnfr.dynamics.dnfr._accumulate_neighbors_numpy`). | Benchmarks the single `np.add.at` accumulator against the legacy stack kernel; on 320 random nodes (p=0.65) with Python 3.11/NumPy 2.3.4 it delivered ~1.9× lower median runtime (0.097 s vs 0.185 s). |

### Evolution backend speed-ups

Use the evolution benchmark to compare how the mathematics backends handle the unitary and contractive engines. The script honours the new CLI selector (`--backends`) and the `TNFR_MATH_BACKEND` environment variable, automatically skipping adapters when the corresponding dependencies (JAX, PyTorch) are missing. Results are reproducible thanks to the explicit RNG seed and optional JSON export:

```bash
PYTHONPATH=src python benchmarks/evolution_backend_speedup.py \
  --sizes 2 4 8 --steps 16 --repeats 3 --dt 0.05 \
  --output results/evolution_backends.json
```

Sample output on Python 3.11 with NumPy 2.3.4 (JAX/PyTorch unavailable) illustrates the reported tables:

Unitary mean time (milliseconds per run)

| dim | numpy |
| --- | --- |
| 2 | 1.808 ms |
| 4 | 0.872 ms |

Contractive mean time (milliseconds per run)

| dim | numpy |
| --- | --- |
| 2 | 1.774 ms |
| 4 | 3.437 ms |

Unitary speed-up vs. NumPy baseline

| dim | numpy |
| --- | --- |
| 2 | 1.000 x |
| 4 | 1.000 x |

Contractive speed-up vs. NumPy baseline

| dim | numpy |
| --- | --- |
| 2 | 1.000 x |
| 4 | 1.000 x |

When additional backends are available, their columns appear automatically in all four tables (unitary timing, contractive timing, and their respective speed-up ratios).

### Broadcast accumulator regression check

The performance test suite now covers the vectorised accumulator that relies on
`numpy.bincount` to collapse neighbour contributions. Run the slow-marked test
to validate both speed and numerical parity against the pure-Python path:

```bash
PYTHONPATH=src pytest tests/performance/test_dynamics_performance.py \
  -k broadcast_neighbor_accumulator_stays_faster_and_correct -m slow
```

The command prints per-test timings; the assertion requires the NumPy path to
complete at least 10 % faster while matching the fallback values within
`1e-9` relative/absolute tolerance. Use it to capture before/after measurements
when iterating on `_accumulate_neighbors_broadcasted` or related kernels.

## Retired scripts

Older benchmarks covering glyph history trimming, usage counters, or glyph
timing updates have been removed because the optimised paths now mirror the
reference implementations. Keeping them would create maintenance noise without
providing actionable performance signals.

## Profiling workflows

### ΔNFR default pipeline

Run the benchmark with profiling enabled to capture cumulative timings for the
entire ΔNFR pipeline. Choose the output format that best fits your tooling:

```bash
PYTHONPATH=src python benchmarks/default_compute_delta_nfr.py \
  --nodes 320 --edge-probability 0.22 --repeats 3 \
  --profile profiles/dnfr_default.pstats
```

* Use `--profile-format json` to export an ordered JSON array with the
  `totaltime` and `inlinetime` for each function.
* Inspect `.pstats` files via `python -m pstats profiles/dnfr_default.pstats` and
  sort on `cumtime` (cumulative time) or `tottime` (self time) to spot hotspots.

### Sense Index vectorised vs. fallback paths

The profiling script mirrors the setup from `tests/performance/test_sense_performance.py`
to compare vectorised and pure-Python Si computations:

```bash
PYTHONPATH=src python benchmarks/compute_si_profile.py \
  --nodes 512 --loops 8 --format json --output-dir profiles
```

The command writes two files:

* `compute_Si_numpy.*` – profile captured when NumPy is available.
* `compute_Si_python.*` – profile captured with NumPy disabled, exercising the
  fallback path.

Inspect the top entries sorted by `cumtime` (cumulative time per function) to
spot the phases consuming most wall-clock time. Compare both outputs to confirm
that vectorisation shifts time into array primitives rather than Python loops.

### Full pipeline profiling (Si + ΔNFR)

```
PYTHONPATH=src python benchmarks/full_pipeline_profile.py \
  --nodes 384 --edge-probability 0.28 --loops 6 --output-dir profiles \
  --si-chunk-sizes auto 2048 --dnfr-chunk-sizes auto 4096
```

The profiler iterates over the Cartesian product of the requested Si and ΔNFR
chunk sizes (and, when provided, worker counts via `--si-workers` and
`--dnfr-workers`). Each combination is tagged as `cfgXX` and encoded in the
artefact name, for example
`full_pipeline_vectorized_cfg01_si_auto_dn_auto_siw_auto_dnw_auto.json`.

For every configuration + execution-mode pair the script writes matching
`.pstats` and `.json` files. The JSON schema now includes:

* `configuration` – the label, ordinal, textual description, and raw knob
  values (`SI_CHUNK_SIZE`, `DNFR_CHUNK_SIZE`, `SI_N_JOBS`, `DNFR_N_JOBS`).
* `metadata` – runtime context covering vectorisation, graph size, and the
  requested/resolved chunk sizes and worker counts applied to the seeded graph.
* `operator_totals` – raw wall-clock totals for each explicit operator call in
  the benchmark loop.
* `operator_timings` – totals and per-loop averages per operator (mirrored under
  `manual_timings` for backwards compatibility).
* `compute_Si_breakdown` – wall-clock totals, per-loop averages, and execution
  path counts for the Sense Index sub-stages (cache rebuilds, vectorised
  neighbour aggregation, normalisation/clamp, and in-place writes). Use the
  `path_counts` entry to verify whether the NumPy kernels or the pure-Python
  fallback handled the run.
* `target_functions` – cumulative profiler statistics (`cumtime`, `totaltime`)
  for the four canonical operators. Use this section to compare how much time
  each function spends (including callees) in vectorised vs. fallback modes.
* `rows` – the complete, sorted profiler table, matching the `.pstats` export.
  Inspect it when a hotspot needs deeper call-tree analysis.

Contrast the vectorised and fallback JSON summaries to confirm that NumPy shifts
most cumulative time from `_compute_dnfr_common` into array-based kernels and
that `compute_Si` benefits from the same optimisation. The console output now
prints the same Sense Index breakdown, making it easier to spot whether cache
reconstruction or neighbour aggregation dominates the runtime. Cross-check the
`path_counts` field—vectorised runs should report NumPy activity while fallback
profiles stay in the Python bucket. Significant regressions should surface as
higher `cumtime` values or inflated per-loop wall-clock figures across both the
top-level operator timings and the Si sub-stages.

### `_compute_dnfr` vectorisation checks

Use the microbenchmark to compare the NumPy and pure-Python branches directly
across multiple graph sizes and densities:

```bash
PYTHONPATH=src python benchmarks/compute_dnfr_benchmark.py \
  --nodes 192 384 768 --edge-probabilities 0.05 0.12 0.3 --repeats 8
```

The output prints a compact table per configuration with:

* **Vectorised best/median/mean/worst** – summary timings in seconds when
  NumPy is available.
* **Fallback best/median/mean/worst** – the pure-Python execution statistics.
* **Ratio (fallback ÷ vectorized)** – the average slow-down when vectorisation
  is disabled. Ratios significantly above `1.00×` confirm that dense kernels are
  exercised and provide a guardrail for regressions.

Pass `--force-dense` to ensure the dense accumulator path is exercised when the
automatic heuristics might prefer sparse accumulation. The script gracefully
degrades to reporting only the fallback timings whenever NumPy is unavailable.

### Chunked execution switches

Both benchmarks honour the new batching knobs exposed by the engine:

* Set `graph.graph["SI_CHUNK_SIZE"] = 2048` (or pass
  `chunk_size=2048` when calling `compute_Si`) inside
  `compute_si_profile.py` to process nodes in deterministic batches. This is
  helpful when profiling large (>10k nodes) graphs on memory-constrained
  machines.
* Set `graph.graph["DNFR_CHUNK_SIZE"] = 4096` before invoking
  `default_compute_delta_nfr` to bound the accumulator size in the ΔNFR
  benchmark. Larger chunks favour throughput, while smaller ones keep the
  temporary buffers inside stricter memory budgets.

Leave both settings unset for medium-sized runs; the automatic heuristics use
CPU availability and a conservative memory estimate to choose a balanced chunk
size.
