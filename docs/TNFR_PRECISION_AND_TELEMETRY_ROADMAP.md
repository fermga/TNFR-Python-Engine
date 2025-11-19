# TNFR Precision & Telemetry Roadmap

**Objective:** Reinvest newly available computational capacity (Φ_s and highly optimized TNFR arithmetic) into **greater numerical precision**, **richer passive telemetry**, and **broader experimental coverage** without modifying canonical grammar (U1–U6) or TNFR physics.

- **Grammar:** Closed at U1–U6 (complete).
- **Invariants:** Must remain intact (see `AGENTS.md` and `UNIFIED_GRAMMAR_RULES.md`).
- **Scope:** Precision modes, telemetry, experiments, diagnostics, and benchmarks only (READ-ONLY relative to dynamics).

> Language Policy: This roadmap is fully translated to English to comply with the English-only documentation requirement.

---

## Phase 1 – Central configuration and Φ_s high precision ✅ COMPLETED

**Completion Date:** 2025-11-15

### 1.1 Central mode configuration ✅

**Implemented file:** `src/tnfr/config/precision_modes.py`

**Integration:** Exported from `src/tnfr/config/__init__.py`

**Public API:**
```python
from tnfr.config import (
    PrecisionMode, TelemetryDensity, DiagnosticsLevel,
    get_precision_mode, set_precision_mode,
    get_telemetry_density, set_telemetry_density,
    get_diagnostics_level, set_diagnostics_level,
)
```

**Implemented Modes:**
- `precision_mode: Literal["standard", "high", "research"]` (default: "standard")
- `telemetry_density: Literal["low", "medium", "high"]` (default: "low")
- `diagnostics_level: Literal["off", "basic", "rich"]` (default: "off")

**Tests:** ✅ `tests/test_config_precision.py`

- 3 tests PASS
- Default values verified
- Setters/getters validated
- Invalid value rejection verified

---

### 1.2 Φ_s (STRUCTURAL POTENTIAL) high precision ✅

**Modified file:** `src/tnfr/physics/canonical.py`

**Implementation:**
```python
def _get_precision_dtype() -> type:
    """Return dtype according to precision_mode."""
    mode = get_precision_mode()
    if mode == "research":
        return np.longdouble  # 80-bit on x86
    else:
        return np.float64

def _compute_phi_s_exact(...):
    dtype = _get_precision_dtype()
    mode = get_precision_mode()
    
    if mode in ("high", "research"):
        # Log-space for numerical stability
        log_contrib = log(|ΔNFR|) - α·log(d)
        contrib = exp(log_contrib)
    else:
        # Standard: direct computation
        contrib = ΔNFR / d^α
```

**Tests:** ✅ `tests/test_precision_phi_s.py`

- 6 tests PASS
- Validated standard/high correlation > 0.99
- **U6 invariant preserved:** identical decisions across modes
- Coherence C(t) stable (diff < 1%)
- Finite values in research mode

**Demo:** ✅ `examples/demo_precision_modes.py`
```
Results (15 nodes):
Standard: Φ_s = -1.816939
High:     Φ_s = -1.817148
Research: Φ_s = -1.817356

U6 drift: 0.0004 (< 2.0 threshold)
→ U6 decision identical ✓
```

**Phase 1 Conclusion:**

- ✅ Precision infrastructure fully functional
- ✅ Φ_s integrated with 3 precision modes
- ✅ TNFR physics (U1–U6) preserved
- ✅ Tests complete and demo functional

---

## Phase 2 – |∇φ|, K_φ and ξ_C high precision ✅ COMPLETED

**Completion Date:** 2025-11-15

### 2.1 Phase gradient |∇φ| ✅

**Modified file:** `src/tnfr/physics/canonical.py`

**Implementation:**
```python
def _compute_phase_gradient_and_curvature(...):
    dtype = _get_precision_dtype()
    
    # Use precision-aware dtype for all calculations
    phi_i = dtype(phases[i])
    neigh_phases = np.array([phases[j] for j in neighbors], dtype=dtype)
    
    # Phase wrapping with correct dtype
    pi_typed = dtype(np.pi)
    wrapped_diffs = (diffs + pi_typed) % (2 * pi_typed) - pi_typed
```

**Tests:** ✅ `tests/test_precision_tetrad.py`

- Correlation standard/high > 0.999
- Max gradient node identical across modes
- Isolated nodes have zero gradient in all modes

### 2.2 Phase curvature K_φ ✅

**Implementation:**

- Uses `dtype` from `get_precision_mode()` for neighborhood calculations
- Improved precision in circular mean of neighbor phases

**Tests:** ✅ `tests/test_precision_tetrad.py`

- Correlation standard/high > 0.999
- Signs of K_φ preserved across modes
- Qualitative pattern identical

### 2.3 Coherence length ξ_C ✅

**Implementation:**
```python
# Precision-aware sampling
if mode == "research":
    sample_threshold = 100
    min_samples = 30
elif mode == "high":
    sample_threshold = 75
    min_samples = 20
else:  # standard
    sample_threshold = 50
    min_samples = 20
```

**Tests:** ✅ `tests/test_precision_tetrad.py`

- **UPDATE 2025-11-15**: Fixed ξ_C test graph generator
  - Problem: Random ΔNFR doesn't guarantee exponential decay
  - Solution: Ring topology with distance-dependent ΔNFR structure
  - Result: **8/8 tests PASS, 0 skipped**
- Research mode uses more samples for better fit
- Finite values when computable

**Phase 2 Conclusion:**
- ✅ All canonical fields (Φ_s, |∇φ|, K_φ, ξ_C) integrated with precision_mode
- ✅ Correlation > 0.99 for all fields across modes
- ✅ Invariants preserved: qualitative decisions identical
- ✅ **8/8 tests PASS, 0 SKIPPED** (previously 6/8 with 2 skipped due to faulty graph generator)

---

## Phase 3 – Rich passive telemetry (tetrad) [PREVIOUS PLAN]

### 2.1 Phase gradient |∇φ|

**File:** `src/tnfr/physics/fields.py`

**Function:** `compute_phase_gradient(G, ...)`

**Changes:**

- Use float64 internally when `precision_mode != "standard"`.
- Ensure correct phase wrapping (mod 2π) independent of precision.

**Tests:** `tests/physics/test_phase_gradient_precision.py`

- Small graph with artificial phases:
  - `standard` vs `high`: same max-gradient node and qualitative structure.

---

### 2.2 Phase curvature K_φ

**File:** `src/tnfr/physics/fields.py`

**Function:** `compute_phase_curvature(G, ...)`

**Changes:**

- In high precision mode, compute neighborhood averages and differences in float64.

**Tests:** `tests/physics/test_phase_curvature_precision.py`

- Simple graphs (line, star) with interpretable K_φ pattern.
- `standard` vs `high`: same extreme nodes (sign and order), only numeric differences.

---

### 2.3 Coherence length ξ_C

**Function:** `estimate_coherence_length(G, ...)` (in `fields.py` or dedicated module)

**Changes:**

- Add parameters controlled by `precision_mode`:
  - More distance radii/shells.
  - More samples (if random sampling is used).
  - In "research" mode allow weighted log–log fit or more robust strategies.

**Tests:** `tests/physics/test_xi_c_precision.py`

- Synthetic curves with approximately known ξ_C.
- In `high`/`research` modes: lower estimator variance without changing regime classification (local vs critical).

---

## Phase 3 – Rich passive telemetry (tetrad) ✅ COMPLETED

**Completion Date:** 2025-11-15

### 3.1 Tetrad snapshot helper ✅

**Implemented file:** `src/tnfr/metrics/tetrad.py` (193 lines)

**Key functions:**
- `collect_tetrad_snapshot(G, include_histograms=None) -> Dict`
  - Passive observation of the 4 canonical fields
  - Φ_s, |∇φ|, K_φ, ξ_C (global values: mean, max, min)
  - Density controlled by `telemetry_density`
  
- `get_tetrad_sample_interval(base_dt=1.0) -> float`
  - Computes sampling interval based on density
  - "low": 10× base_dt, "medium": 5× base_dt, "high": 1× base_dt

**Implemented detail levels:**

**Low Density:**

- Basic stats: mean, max, min, std
- No percentiles
- No histograms
- Sampling interval: 10× base timestep

**Medium Density:**

- Basic stats + quartiles (p25, p50, p75)
- No histograms
- Sampling interval: 5× base timestep

**High Density:**

- Basic stats + quartiles + tail percentiles (p10, p90, p99)
- Histograms included (20 bins)
- Sampling interval: 1× base timestep (each step)

**Export:** Public module via `src/tnfr/metrics/__init__.py`
```python
from tnfr.metrics import (
    collect_tetrad_snapshot,
    get_tetrad_sample_interval,
)
```

**Tests:** ✅ `tests/test_tetrad_snapshot.py` (291 lines)
- 10/10 tests PASS
- Snapshot structure validated
- Statistics per density level correct
- Histogram override functional
- **Physical invariance preserved:** Snapshot is READ-ONLY
- Consistency across densities: core values identical
- Sampling interval correct per density
- Handles empty graphs and isolated nodes

**Demo:** ✅ `examples/demo_tetrad_telemetry.py` (206 lines)
```
Results (30 nodes, Watts-Strogatz):

LOW Density:
  Phi_s: mean=-2.8016, max=0.5894, min=-6.0583
  Sample interval: 1.00 (10× base_dt)

MEDIUM Density:
  Phi_s: mean=-2.8016, p25=-4.3656, p50=-2.9992, p75=-1.5138
  Sample interval: 0.50 (5× base_dt)

HIGH Density:
  Phi_s: mean=-2.8016, p10=-4.8965, p90=0.3391, p99=0.5826
  Histograms: 20 bins
  Sample interval: 0.10 (1× base_dt)

[OK] DNFR unchanged across telemetry collection
[OK] Telemetry is READ-ONLY (no operator changes)
[OK] U1-U6 grammar preserved
```

**Phase 3 Conclusion:**
- ✅ Fully functional tetrad snapshot system
- ✅ 3 density levels implemented (low/medium/high)
- ✅ Sampling intervals auto-scale
- ✅ Purely observational (READ-ONLY); absolute physics invariance
- ✅ 10/10 tests PASS, demo functional
- ✅ Public API integrated in `tnfr.metrics`

---

## Phase 4 – Scale topologies & expand experimental coverage ✅ COMPLETED

**Completion Date:** 2025-11-15

### 4.0 Summary of Achievements

- ✅ Core benchmarks now parameterized via CLI (`--nodes`, `--seeds`, `--topologies`, `--high-resolution`, `--quiet`).
- ✅ Added `--dry-run` to `coherence_length_critical_exponent.py` for fast CI parameter validation.
- ✅ High resolution flags activate extended grids (extra distances in asymptotic freedom; additional thresholds in confinement zones).
- ✅ Tetrad universality correlation script: `benchmarks/analyze_tetrad_universality.py` (Φ_s, |∇φ|, K_φ, ξ_C vs C(t), Si per topology).
- ✅ ξ_C precision comparison script: `benchmarks/analyze_xi_c_precision.py` (drift statistics and qualitative mode consistency).
- ✅ CLI tests: `tests/benchmarks/test_cli_params.py` (3/3 PASS minimal configs; uses `--dry-run` for critical xi_C path).
- ✅ Clean lint after refactors (< 79 char lines). No TNFR dynamics mutation: purely passive experimental enrichment.

### 4.1 Parameterize network size and seeds (IMPLEMENTED)

**Modified files:**

- `benchmarks/asymptotic_freedom_test.py` (new argparse parser, extended scales with `--high-resolution`).
- `benchmarks/confinement_zones_test.py` (argparse parser, extra thresholds in high resolution mode).
- `benchmarks/coherence_length_critical_exponent.py` (extended parser + `--dry-run`).

**Added flags:**

```text
--nodes <int>            # network size
--seeds <int>            # number of runs / seeds
--topologies <list>      # subset of topologies
--high-resolution        # activates extended grid/thresholds
--quiet                  # silences output (CI)
--dry-run                # (xi_C only) validate and exit quickly
```

### 4.2 Increase experimental resolution (IMPLEMENTED)

- Asymptotic freedom: additional scales [6, 8, 9, 12, 15] for better α fit.
- Confinement zones: additional thresholds [2.5, 3.25, 3.75, 5.25, 6.5] around revised 3.0 value.
- Critical xi_C: external grid support already present; behavior preserved via existing parser.

### 4.3 Analysis scripts (IMPLEMENTED)

**`analyze_tetrad_universality.py`**

- Input: JSONL/CSV with tetrad/coherence metrics.
- Output: per-topology JSONL with correlations and predictive ranking.
- Runs entirely in passive (READ-ONLY) mode.

**`analyze_xi_c_precision.py`**

- Input: JSONL of critical (xi_C) experiments with field `precision_mode`.
- Metrics: global mean ξ_C, drift range, qualitative consistency (relative drift < 5%).
- Output: table + consolidated JSONL.

### 4.4 CLI tests and validation (IMPLEMENTED)

- File: `tests/benchmarks/test_cli_params.py`
- Coverage:
  - Asymptotic freedom (basic parametrization)
  - Confinement zones (basic parametrization)
  - Coherence length critical exponent (`--dry-run` for speed)
- Result: 3/3 PASS (<5 s)

### 4.5 No physical changes

- Grammar U1–U6 intact.
- Invariants preserved (configuration and external analysis only).
- No operation alters ΔNFR, ν_f, phases or EPI: all passive.

**Phase 4 Conclusion:** Scaling and experimental flexibility achieved; infrastructure ready for bifurcation phase (Phase 5) and regression/documentation (Phase 6) with expanded analytical base.

**New file:** `src/tnfr/metrics/tetrad.py`

**Function:** `collect_tetrad_snapshot(G, ...) -> Dict`

**Responsibility:** build a purely observational snapshot of canonical fields:

- Φ_s, |∇φ|, K_φ, ξ_C (global values: mean, max, min)
- If `telemetry_density == "high"`:
  - Percentiles (p25, p50, p75, p90, p99)
  - Tail size and optional discretized histograms of ΔNFR and phases

**Tests:** `tests/metrics/test_tetrad_snapshot.py`

- Same network, same configuration:
  - Verify snapshot contains all expected fields
  - With `telemetry_density="high"` more detail but same semantics

---

### 3.2 Dynamics integration

**Files:** `src/tnfr/dynamics/*.py`

**Changes:**

- Introduce parameter `tetrad_sample_interval` derived from `telemetry_density`.
- At each `tetrad_sample_interval` steps:
  - Call `collect_tetrad_snapshot(G, ...)`.
  - Store/emit snapshot (internal list, JSONL stream, etc.).
- **Restriction:** telemetry must not modify operators or U1–U6 decisions.

**Tests:** `tests/metrics/test_tetrad_snapshot_density.py`

- Same simulation with `telemetry_density="low"` vs `"high"`:
  - Same ΔNFR(t), C(t), operator choices and outcomes
  - Only number/size of snapshots increases

---

## Phase 4 – Scale topologies & expand experimental coverage (legacy plan translated)

### 4.1 Parameterize network size and seeds

**Files:** key scripts in `benchmarks/`, e.g.:
- `benchmarks/coherence_length_critical_exponent.py`
- `benchmarks/asymptotic_freedom_test.py`
- `benchmarks/confinement_zones_test.py`

**Changes:**

- Add CLI flags:
  - `--nodes` (e.g. 1000, 10000, 100000)
  - `--seeds` (number of seeds)
  - `--topologies ring,ws,scale_free,grid,...`
- Keep defaults equivalent to current behavior.

### 4.2 Increase experimental resolution

**Add flag** `--high-resolution` to:

- Densify parameter grid around critical regions (intensity, couplings, etc.)
- Increase seeds per grid point

**Recommended analysis scripts:**
- `benchmarks/analyze_tetrad_universality.py`:
  - Read results (JSONL/CSV) from multiple experiments
  - Compute correlations between tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and C(t), Si per topology/size
- `benchmarks/analyze_xi_c_precision.py`:
  - Compare critical exponents and errors under different precision modes

**Tests:** `tests/benchmarks/test_cli_params.py`

- Run minimal benchmark versions with flags to ensure correct parsing and successful completion.

---

## Phase 5 – Fine-grained bifurcations & intensive exploration ✅ COMPLETED

**Completion Date:** 2025-11-15

### 5.1 Real bifurcation metrics and CLI sweeps ✅

**Implemented files:**

- `benchmarks/bifurcation_metrics.py`
  - FieldSnapshot capture (Φ_s, |∇φ|, K_φ, ξ_C)
  - Topology builder fixed for scale-free via Barabási–Albert
  - Initialization injects light ΔNFR variability for benchmarks/tests (read-only to dynamics; preserves invariants)
  - Operator sequence: OZ (+optional ZHIR) with IL/THOL handlers; records `handlers_present`
  - Metrics: `delta_phi_s`, `delta_phase_gradient_max`, `delta_phase_curvature_max`, `coherence_length_ratio`, `delta_dnfr_variance`, `bifurcation_score_max`, `handlers_present`, `classification`

- `benchmarks/bifurcation_landscape.py`
  - Full CLI sweep producing JSONL per grid point
  - Threshold flags: `--bifurcation-score-threshold`, `--phase-gradient-spike`, `--phase-curvature-spike`, `--coherence-length-amplification`, `--dnfr-variance-increase`, `--structural-potential-shift`, `--fragmentation-coherence-threshold`
  - Core grids: `--nodes`, `--seeds`, `--topologies`, `--oz-intensity-grid`, `--vf-grid`, `--mutation-thresholds`
  - Modes: `--dry-run` (validate params only), `--quiet` (suppresses JSONL; omit when piping output)

**Classification states:** `none | incipient | bifurcation | fragmentation`

**Notes on canonicity:**

- Metrics are computed from pre/post snapshots around operator sequences; no direct EPI mutation outside operators
- Initialization ΔNFR perturbation is benchmark/test-only to produce measurable deltas; operators remain the sole state-changing mechanism
- Phase verification and handlers (IL/THOL) enforced per U2–U4

**Tests:** ✅

- `tests/benchmarks/test_bifurcation_metrics.py` (21/21 PASS): deltas, handler detection, classification transitions, reproducibility
- `tests/benchmarks/test_cli_params.py` (5/5 PASS): includes real-run JSONL parsing (omit `--quiet`)

**Validation sweeps:**

- Moderate (ring, WS): classifications predominantly `none`; `handlers_present=true` throughout; scores ~0.145–0.247
- Aggressive (ring): `incipient` dominant; coherence decreases as expected; scores up to ~0.319

**Example commands:**

```pwsh
# Minimal sweep with JSONL output
python benchmarks/bifurcation_landscape.py --nodes 30 --seeds 2 --topologies ring,ws `
  --oz-intensity-grid 0.1,0.2,0.3 --vf-grid 0.8,1.0 `
  --bifurcation-score-threshold 0.28 --phase-gradient-spike 0.35 `
  --phase-curvature-spike 3.0 --coherence-length-amplification 1.5

# Parameter validation only (no runs)
python benchmarks/bifurcation_landscape.py --dry-run --nodes 20 --seeds 1
```

### 5.2 Intensive exploration & sensitivity mode (scaffold) 🚧

**Planned (tracked for Phase 6+):**

- `src/tnfr/experiments/exploration.py`
  - `run_exploration_batch(...)` for vectorized sweeps across configs/topologies
  - `sensitivity_scan(...)` perturbing ν_f, initial ΔNFR, phases; read-only telemetry outputs
- Tests: seed reproducibility, bounded response to small perturbations

---

## Phase 6 – Benchmarks, regressions & documentation

### 6.1 New tracks in benchmark_optimization_tracks.py

**File:** `benchmarks/benchmark_optimization_tracks.py`

**Suggested Tracks:**

- `precision_modes`:
  - Compare time and variance of Φ_s, ξ_C, |∇φ|, K_φ in `precision_mode="standard"` vs `"high"`
- `telemetry_pipeline` (extended):
  - Measure overhead with dense telemetry and callbacks enabled

### 6.2 Regression harness

**New script:** `tools/run_tetrad_regression.py`

**Function:**

- Run a fixed set of representative simulations/benchmarks
- Store reference snapshots (JSONL) of tetrad, C(t), Si, etc.
- Compare new runs against reference within numerical tolerances

---

### 6.3 Documentation updates

**Files to update:**

- `AGENTS.md`:
  - Brief section on `precision_mode`, `telemetry_density`, `diagnostics_level`
  - Reinforce that these modes do not modify U1–U6 or TNFR physics
- `docs/grammar/07-MIGRATION-AND-EVOLUTION.md`:
  - Add "Capacity reinvestment" section describing improvements as engine evolution, not grammar change
- `DOCUMENTATION_INDEX.md` and U6/ξ_C specific docs (e.g., `docs/XI_C_CANONICAL_PROMOTION.md`, `docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md`):
  - Link to new experiments and analysis tools

---

### 6.4 Final validation

**Recommended commands:**

```pwsh
# Main benchmarks
python run_benchmark.py

# Basic tests
pytest

# Tests with high precision / telemetry mode (if CLI or env vars)
# Example (depending on implementation):
$env:TNFR_PRECISION_MODE = "high"
pytest tests/physics -q
```

**Acceptance criteria:**

- All tests still pass
- `run_benchmark.py` remains green with reasonable improvement reporting
- Grammar (U1–U6) and physical invariants behave identically; only resolution, coverage and telemetry increase

---

## Summary

This roadmap enables:

- Intensification of numerical precision for Φ_s, |∇φ|, K_φ and ξ_C
- Increased telemetry density and richness without affecting dynamics
- Scaling of topologies and validation experiments far beyond previous limits
- Addition of diagnostics and exploration/sensitivity modes reinforcing confidence in model robustness
- Preservation of canonical grammar (U1–U6) and TNFR physics fully intact
