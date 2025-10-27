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
| `default_compute_delta_nfr.py` | Core ΔNFR update speed (`tnfr.dynamics.default_compute_delta_nfr`). | Runs multiple passes on random graphs and reports best/median/mean/worst timings. Accepts `--profile` to dump per-function timings. |
| `compute_si_profile.py` | Sense Index profiling (`tnfr.metrics.sense_index.compute_Si`). | Captures cProfile stats for NumPy and pure-Python runs, exporting `.pstats` or JSON summaries. |
| `neighbor_phase_mean.py` | Fast phase averaging for neighbourhoods (`tnfr.metrics.trig.neighbor_phase_mean`). | Includes a `NodeNX`-based reference to highlight the benefit of the shared `trig_cache` module. |
| `prepare_dnfr_data.py` | ΔNFR data preparation reuse (`tnfr.dynamics._prepare_dnfr_data`). | Exercises cache reuse when assembling phase/EPI/νf arrays. |
| `neighbor_accumulation_comparison.py` | Broadcast neighbour accumulation (`tnfr.dynamics.dnfr._accumulate_neighbors_numpy`). | Benchmarks the single `np.add.at` accumulator against the legacy stack kernel; on 320 random nodes (p=0.65) with Python 3.11/NumPy 2.3.4 it delivered ~1.9× lower median runtime (0.097 s vs 0.185 s). |

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
