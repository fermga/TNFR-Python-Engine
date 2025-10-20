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
| `default_compute_delta_nfr.py` | Core ΔNFR update speed (`tnfr.dynamics.default_compute_delta_nfr`). | Runs multiple passes on random graphs and reports best/median/mean/worst timings. |
| `neighbor_phase_mean.py` | Fast phase averaging for neighbourhoods (`tnfr.metrics.trig.neighbor_phase_mean`). | Includes a `NodeNX`-based reference to highlight the benefit of the shared `trig_cache` module. |
| `prepare_dnfr_data.py` | ΔNFR data preparation reuse (`tnfr.dynamics._prepare_dnfr_data`). | Exercises cache reuse when assembling phase/EPI/νf arrays. |
| `neighbor_accumulation_comparison.py` | Broadcast neighbour accumulation (`tnfr.dynamics.dnfr._accumulate_neighbors_numpy`). | Benchmarks the single `np.add.at` accumulator against the legacy stack kernel; on 320 random nodes (p=0.65) with Python 3.11/NumPy 2.3.4 it delivered ~1.9× lower median runtime (0.097 s vs 0.185 s). |

## Retired scripts

Older benchmarks covering glyph history trimming, usage counters, or glyph
timing updates have been removed because the optimised paths now mirror the
reference implementations. Keeping them would create maintenance noise without
providing actionable performance signals.
