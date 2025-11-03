# Reproducibility Infrastructure

This document describes the reproducibility infrastructure for TNFR benchmarks.

## Overview

The TNFR project now includes deterministic pipeline execution and artifact traceability through:
- Global seed management for benchmarks
- SHA256 checksum generation for all outputs
- Manifest-based verification
- CI integration for reproducibility testing

## Quick Start

### Run all benchmarks with deterministic seeds

```bash
make reproduce
```

This will:
1. Run all configured benchmarks with seed=42 (default)
2. Generate output artifacts in `artifacts/` directory
3. Create `artifacts/manifest.json` with checksums

### Verify checksums

```bash
make reproduce-verify
```

This verifies that all artifacts match the checksums in the manifest.

## Advanced Usage

### Run specific benchmarks

```bash
python scripts/run_reproducible_benchmarks.py \
  --benchmarks comprehensive_cache_profiler full_pipeline_profile \
  --seed 123 \
  --output-dir my_artifacts
```

### Custom verification

```bash
python scripts/run_reproducible_benchmarks.py \
  --verify my_artifacts/manifest.json \
  --verbose
```

## Configured Benchmarks

The following benchmarks are configured for reproducible execution:

1. **comprehensive_cache_profiler** - Tracks buffer allocation effectiveness across TNFR hot paths
2. **full_pipeline_profile** - Full telemetry + ΔNFR pipeline profiling
3. **cache_hot_path_profiler** - Cache metrics for hot execution paths
4. **compute_si_profile** - Sense Index profiling (vectorized vs fallback)

## Manifest Format

The manifest file (`manifest.json`) contains:

```json
{
  "seed": 42,
  "benchmarks": {
    "benchmark_name": {
      "status": "success",
      "output_files": ["path/to/output.json"],
      "checksums": {
        "output.json": "sha256_checksum_here"
      }
    }
  }
}
```

## CI Integration

The reproducibility CI workflow (`.github/workflows/reproducibility.yml`) runs on:
- Push to main/master
- Pull requests
- Manual trigger

It verifies that benchmarks:
1. Complete successfully
2. Generate valid output artifacts
3. Produce consistent manifest structure

## TNFR Compliance

This infrastructure follows TNFR principles:

- **Controlled Determinism** (Invariant #8): Seeds ensure reproducible execution
- **Structural Traceability**: Checksums provide artifact verification
- **Operational Fractality**: No changes to core TNFR operators
- **Trans-scale Neutrality**: Infrastructure works across all benchmark scales

## Troubleshooting

### Benchmark fails to run

Check that all dependencies are installed:
```bash
pip install .[test,numpy,yaml,orjson]
```

### Checksum mismatch

Some benchmarks may include timing information that varies between runs. This is expected behavior. The important part is:
1. Benchmarks run successfully with consistent seeds
2. Structural outputs are deterministic
3. Manifest structure is valid

### Missing benchmark script

Ensure you're running from the repository root and the benchmark script exists in `benchmarks/`.

## Adding New Benchmarks

To add a new benchmark to the reproducibility suite:

1. Ensure the benchmark script accepts `--seed` parameter
2. Add configuration to `BENCHMARK_CONFIGS` in `scripts/run_reproducible_benchmarks.py`
3. Test with: `python scripts/run_reproducible_benchmarks.py --benchmarks your_benchmark`
4. Update this documentation

## References

- [scripts/README.md](../scripts/README.md) - Script documentation
- [benchmarks/README.md](../benchmarks/README.md) - Benchmark usage guide
- [AGENTS.md](../AGENTS.md) - TNFR paradigm compliance
