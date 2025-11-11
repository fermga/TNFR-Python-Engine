# Command Line Interface (CLI) Reference

This page documents selected CLI entry points for the TNFR Python Engine. It consolidates content migrated from the repository root `README.md` on 2025-11-11.

> Tip: Run any command with `--help` to see available options.

---
## Profile Pipeline
Profile an end-to-end resonance computation pipeline and generate reports.

```pwsh
# Example (PowerShell)
tnfr profile-pipeline \
  --nodes 120 --edge-probability 0.28 --loops 3 \
  --si-chunk-sizes auto 48 --dnfr-chunk-sizes auto \
  --output-dir profiles/pipeline
```

Artifacts:
- Python `.pstats` files for detailed callgraph inspection
- JSON summaries with timings per stage (ΔNFR, Si, accumulation, cache hits)
- Optional flamegraphs when supported

Notes:
- Use GPU backends with the appropriate extras (`compute-jax` or `compute-torch`) to compare performance.
- Keep grammar constraints in mind when tuning loops or destabilizers; telemetry must remain bounded (U2) and confinement respected (U6: Δ Φ_s < 2.0).

---
## Environment and Secrets
Configuration values (e.g., cache endpoints) should be loaded via the secure helpers:

```python
from tnfr.secure_config import load_redis_config, get_cache_secret
redis_cfg = load_redis_config()
```

Never commit credentials. See the Security guides for more.

---
## Related Guides
- Performance Optimization: `advanced/PERFORMANCE_OPTIMIZATION.md`
- Scalability: `SCALABILITY.md`
- Benchmarks: `benchmarks/`
- Security Guides: `security/`
