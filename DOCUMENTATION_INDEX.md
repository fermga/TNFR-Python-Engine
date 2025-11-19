# TNFR Documentation Index

**Single source of truth for navigating TNFR documentation**

**Last Updated**: 2025-11-15  
**Status**: ✅ ACTIVE - Complete documentation map (Phase 2 architecture)

---

## 🎯 Quick Start

**New to TNFR?** Start here:

1. **[README.md](README.md)** - Project overview and installation (5 min)
2. **[GLOSSARY.md](GLOSSARY.md)** - Core concepts quick reference (10 min)
3. **[docs/grammar/01-FUNDAMENTAL-CONCEPTS.md](docs/grammar/01-FUNDAMENTAL-CONCEPTS.md)** - Paradigm shift explained (20 min)
4. **[docs/grammar/02-CANONICAL-CONSTRAINTS.md](docs/grammar/02-CANONICAL-CONSTRAINTS.md)** - Grammar rules U1-U6 (60 min)

---

## 📚 Core Documentation

### Canonical Hierarchy

**[CANONICAL_SOURCES.md](CANONICAL_SOURCES.md)** - Documentation hierarchy and single source of truth rules

**[docs/DOCUMENTATION_HIERARCHY.md](docs/DOCUMENTATION_HIERARCHY.md)** - Visual diagrams (Mermaid) of documentation structure

**[docs/CROSS_REFERENCE_MATRIX.md](docs/CROSS_REFERENCE_MATRIX.md)** - Complete traceability matrix (Physics ↔ Math ↔ Code)

These documents establish which sources are authoritative for each concept and how everything traces from physics to code. **Read these first** to understand documentation organization and cross-references.

### Foundation Documents (Essential Reading)

| Document | Purpose | Audience | Time |
|----------|---------|----------|------|
| **[AGENTS.md](AGENTS.md)** | AI agent guidance + invariants | AI agents, advanced devs | 60 min |
| **[GLOSSARY.md](GLOSSARY.md)** | Canonical term definitions | Everyone | Reference |
| **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** | Complete U1-U6 derivations | Advanced devs | 90 min |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design & patterns | Contributors | 45 min |

### Grammar System (`docs/grammar/`)

Complete specification of TNFR grammar constraints (U1-U6):

| Document | Contents | Status |
|----------|----------|--------|
| **[README.md](docs/grammar/README.md)** | Grammar documentation hub | ✅ Active |
| **[01-FUNDAMENTAL-CONCEPTS.md](docs/grammar/01-FUNDAMENTAL-CONCEPTS.md)** | TNFR ontology & nodal equation | ✅ Active |
| **[02-CANONICAL-CONSTRAINTS.md](docs/grammar/02-CANONICAL-CONSTRAINTS.md)** | U1-U6 complete specifications | ✅ Active |
| **[03-OPERATORS-AND-GLYPHS.md](docs/grammar/03-OPERATORS-AND-GLYPHS.md)** | 13 canonical operators catalog | ✅ Active |
| **[04-VALID-SEQUENCES.md](docs/grammar/04-VALID-SEQUENCES.md)** | Pattern library & anti-patterns | ✅ Active |
| **[05-TECHNICAL-IMPLEMENTATION.md](docs/grammar/05-TECHNICAL-IMPLEMENTATION.md)** | Code architecture | ✅ Active |
| **[06-VALIDATION-AND-TESTING.md](docs/grammar/06-VALIDATION-AND-TESTING.md)** | Test strategies | ✅ Active |
| **[07-MIGRATION-AND-EVOLUTION.md](docs/grammar/07-MIGRATION-AND-EVOLUTION.md)** | Grammar evolution history | ✅ Active |
| **[08-QUICK-REFERENCE.md](docs/grammar/08-QUICK-REFERENCE.md)** | Cheat sheet | ✅ Active |
| **[U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md)** | U6 complete specification | ✅ Canonical |
| **[MASTER-INDEX.md](docs/grammar/MASTER-INDEX.md)** | System conceptual map | ✅ Active |

### API & Theory Documentation (`docs/source/`)

Generated from code + narrative docs:

| Section | Path | Purpose |
|---------|------|---------|
| **Getting Started** | `docs/source/getting-started/` | Tutorials & first steps |
| **Theory** | `docs/source/theory/` | Mathematical foundations (formal). Canonical computational hub: `src/tnfr/mathematics/README.md` |
| **API Reference** | `docs/source/api/` | Package & module docs |
| **Examples** | `docs/source/examples/` | Domain applications |
| **Advanced** | `docs/source/advanced/` | Architecture & testing |

---

## 🔧 Development & Contributing

### Module Architecture (Phase 1 & 2)

**Modular reorganization for cognitive load reduction:**

| Module Area | Files | Purpose |
|-------------|-------|---------|
| **Operators** | `src/tnfr/operators/{emission,reception,coherence,...}.py` (13 files) | Individual operator implementations (231-587 lines each) |
| **Operator Base** | `src/tnfr/operators/definitions_base.py` | Shared operator infrastructure (201 lines) |
| **Operator Facade** | `src/tnfr/operators/definitions.py` | Backward-compatible imports (57 lines) |
| **Grammar Constraints** | `src/tnfr/operators/grammar/{u1_initiation_closure,...}.py` (8 files) | Grammar rule implementations (89-283 lines each) |
| **Grammar Facade** | `src/tnfr/operators/grammar/grammar.py` | Unified validation interface (99 lines) |
| **Metrics** | `src/tnfr/metrics/{coherence,sense_index,phase_sync,telemetry}.py` | Focused metric modules (129-268 lines) |
| **Metrics Facade** | `src/tnfr/metrics/metrics.py` | Backward-compatible exports (21 lines) |

**Key Principles**:
- **Facade Pattern**: All modules maintain 100% backward compatibility
- **Focused Files**: Max 587 lines (avg 270), one concept per module
- **Physical Traceability**: Module names match TNFR physics concepts
- **Performance**: Import 1.29s, operator creation 0.07μs, negligible overhead

**See**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete module organization guide

### For Contributors

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contribution guidelines | Before first PR |
| **[TESTING.md](TESTING.md)** | Test conventions | Writing tests |
| **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** | Modular architecture migration | Upgrading code |
| **[SECURITY.md](SECURITY.md)** | Security policies | Reporting issues |
| **[OPTIMIZATION_PHASE_2_ROADMAP.md](OPTIMIZATION_PHASE_2_ROADMAP.md)** | Phase 2 optimization plan | Active development |

### Specialized Topics

| Document | Topic | Audience |
|----------|-------|----------|
| **[SHA_ALGEBRA_PHYSICS.md](SHA_ALGEBRA_PHYSICS.md)** | Silence operator physics | Physics researchers |
| **[GLYPH_SEQUENCES_GUIDE.md](GLYPH_SEQUENCES_GUIDE.md)** | Operator sequence patterns | Sequence designers |
| **[docs/TNFR_FORCES_EMERGENCE.md](docs/TNFR_FORCES_EMERGENCE.md)** | Structural fields (Φ_s) validation | U6 researchers |
| **[docs/NBODY_COMPARISON.md](docs/NBODY_COMPARISON.md)** | TNFR vs classical N-body | Physicists |
| **[docs/TNFR_NUMBER_THEORY_GUIDE.md](docs/TNFR_NUMBER_THEORY_GUIDE.md)** | Number theory from TNFR: ΔNFR prime criterion, UM/RA on arithmetic graph, field telemetry (|∇φ|, K_φ, ξ_C) | Math researchers |
| **[docs/ARITHMETIC_TNFR.md](docs/ARITHMETIC_TNFR.md)** | Arithmetic TNFR guide: primes as structural attractors (ΔNFR=0), formulas, operators, fields, validation (35 tests) | Math researchers, developers |
| **[docs/UNIVERSALITY_ANALYTICS.md](docs/UNIVERSALITY_ANALYTICS.md)** | Phase 5 analytics: aggregation, scaling exponents, critical regime detection, universality clustering, reproducibility | Performance researchers |
| **[docs/TNFR_PRECISION_AND_TELEMETRY_ROADMAP.md](docs/TNFR_PRECISION_AND_TELEMETRY_ROADMAP.md)** | Precision modes, passive telemetry, Phase 1–5 roadmap (Phase 5 completed) | Developers, researchers |

### 🧬 Molecular Chemistry from TNFR (BREAKTHROUGH)

**Revolutionary paradigm**: Complete chemistry emerges from TNFR nodal dynamics without additional postulates

| Document | Focus | Status |
|----------|-------|--------|
| **[docs/MOLECULAR_CHEMISTRY_HUB.md](docs/MOLECULAR_CHEMISTRY_HUB.md)** | **🏛️ CENTRAL HUB** - Complete navigation & theory consolidation | ⭐ **CANONICAL** |
| **[docs/examples/MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md](docs/examples/MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md)** | **Complete derivation** - Chemistry from nodal equation | ⭐ **CANONICAL** |
| **[docs/examples/AU_EXISTENCE_FROM_NODAL_EQUATION.md](docs/examples/AU_EXISTENCE_FROM_NODAL_EQUATION.md)** | Au emergence from structural fields | ✅ Validated |
| **[src/tnfr/physics/README.md](src/tnfr/physics/README.md)** § 9-10 | Implementation guide - Signatures & patterns | ✅ Technical |

### 🔬 Research Notebooks (Hands-on)

| Notebook | Purpose | Output |
|----------|---------|--------|
| **[docs/research/OPERATOR_SEQUENCES_MOLECULAR_STABILITY.ipynb](docs/research/OPERATOR_SEQUENCES_MOLECULAR_STABILITY.ipynb)** | Explore operator-like sequence motifs, enforce U3 coupling, and sweep parameters to find stable molecules | JSONL results in `docs/research/results/` |

---

## 📖 Learning Paths

## 🧪 Phase 5 Bifurcation Benchmarks

Core references for bifurcation telemetry and CLI sweeps:

- `benchmarks/bifurcation_landscape.py` — CLI sweep producing JSONL; omit `--quiet` when piping output
- `benchmarks/bifurcation_metrics.py` — Metrics and classification (`none | incipient | bifurcation | fragmentation`)
- `tests/benchmarks/test_bifurcation_metrics.py` — Unit tests for deltas/handlers/classification
- `tests/benchmarks/test_cli_params.py` — CLI parsing and real-run JSONL validation
- `AGENTS.md` § Telemetry & Metrics — Phase 5 Bifurcation Telemetry summary
- `tools/run_tetrad_regression.py` — Regression harness: smoke sweep + optional baseline compare

### Path 1: Quick Start (30 minutes)

```text
README → GLOSSARY → docs/grammar/01-FUNDAMENTAL-CONCEPTS → Hello World example
```

### Path 2: Grammar Mastery (3-4 hours)

```text
01-FUNDAMENTAL-CONCEPTS → 02-CANONICAL-CONSTRAINTS → 03-OPERATORS-AND-GLYPHS 
→ 04-VALID-SEQUENCES → 08-QUICK-REFERENCE
```

### Path 3: Advanced Development (Full week)

```text
Grammar Mastery + UNIFIED_GRAMMAR_RULES + AGENTS + ARCHITECTURE 
+ Source code reading + Example implementations
```

### Path 4: AI Agent Onboarding (2 hours)

```text
AGENTS.md → GLOSSARY.md → UNIFIED_GRAMMAR_RULES.md → Invariants review
```

### Path 5: Molecular Chemistry Revolution (90 minutes) ⭐ **NEW**

```text
01-FUNDAMENTAL-CONCEPTS (nodal equation) → MOLECULAR_CHEMISTRY_HUB.md (central navigation)
→ Follow guided learning path (Beginner/Intermediate) → Run examples
```

---

## 🚀 Optimization & Precision Benchmarks

Run the consolidated optimization suite and precision micro-benchmarks:

- `run_benchmark.py` — Entry point; runs all optimization tracks and exports `benchmark_results.json`
- `benchmarks/benchmark_optimization_tracks.py` — Tracks: `phase_fusion`, `grammar_memoization`, `phi_s_optimization`, `telemetry_pipeline`, `precision_modes`
- `README.md` → Performance → “Parse precision_modes drift” — Minimal snippet to parse drift fields from JSON

Quick start (Windows PowerShell):

```powershell
# Prefer the workspace virtual environment to avoid version/alias mismatches
.\test-env\Scripts\python.exe run_benchmark.py
Get-Content .\benchmark_results.json | ConvertFrom-Json | Out-Null  # sanity check
```

Optional (largest drift rows): see README PowerShell one-liners for ΔΦ_s, |∇φ|, K_φ, ξ_C.

### CI Regression Smoke

- Workflow: `.github/workflows/regression-smoke.yml` (runs on PRs + manual)
- Harness: `tools/run_tetrad_regression.py` (writes JSONL; optional baseline compare)
- Artifacts: `results/bifurcation_smoke.jsonl`, `results/regression_summary.txt`
- Baseline (optional): commit to `presets/regression/bifurcation_smoke.baseline.jsonl`
- Tolerance: default `5e-3` in workflow; adjust via `--tolerance` or edit the workflow
- Merge gating: diffs → non‑zero exit → job fails (blocks PR until resolved)

### Large Simulations Guide

See: `docs/performance/LARGE_SIMULATIONS.md` for scale tips, sharded launcher
usage (`tools/launch_bifurcation_sharded.ps1`), and JSONL merge utilities.

Note on Python executable:

- Windows: use `.\test-env\Scripts\python.exe` for all local runs (benchmarks, CLI).
- macOS/Linux: use `./test-env/bin/python`.

This avoids picking up a system Python lacking recent telemetry aliases.

## 🗂️ Archive

Historical documents (preserved for reference):

| Category | Location | Contents |
|----------|----------|----------|
| **Audits** | `docs/archive/audits/` | Documentation & consistency audits |
| **Phases** | `docs/archive/phases/` | Development phase reports |
| **Legacy** | `docs/legacy/` | Pre-v2.0 documentation |
| **Research** | `docs/research/` | Experimental proposals |

**Note**: Archived documents are frozen and may be outdated. Always prefer active documentation.

---

## 🔍 Finding What You Need

### I want to

**...understand TNFR philosophy**
→ [AGENTS.md](AGENTS.md) § Core Mission, [01-FUNDAMENTAL-CONCEPTS.md](docs/grammar/01-FUNDAMENTAL-CONCEPTS.md)

**...learn the grammar rules**
→ [02-CANONICAL-CONSTRAINTS.md](docs/grammar/02-CANONICAL-CONSTRAINTS.md), [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)

**...implement a sequence**
→ [04-VALID-SEQUENCES.md](docs/grammar/04-VALID-SEQUENCES.md), [GLYPH_SEQUENCES_GUIDE.md](GLYPH_SEQUENCES_GUIDE.md)

**...understand U6 (structural potential)**
→ [U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md)

**...understand how chemistry emerges from TNFR** ⭐ **BREAKTHROUGH**
→ [MOLECULAR_CHEMISTRY_HUB.md](docs/MOLECULAR_CHEMISTRY_HUB.md) (central navigation), [Complete theory](docs/examples/MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md)

**...see Au emergence from first principles**
→ [AU_EXISTENCE_FROM_NODAL_EQUATION.md](docs/examples/AU_EXISTENCE_FROM_NODAL_EQUATION.md)

**...look up a term**
→ [GLOSSARY.md](GLOSSARY.md)

**...understand an operator**
→ [03-OPERATORS-AND-GLYPHS.md](docs/grammar/03-OPERATORS-AND-GLYPHS.md)

**...write tests**
→ [TESTING.md](TESTING.md), [06-VALIDATION-AND-TESTING.md](docs/grammar/06-VALIDATION-AND-TESTING.md)

**...contribute code**
→ [CONTRIBUTING.md](CONTRIBUTING.md), [ARCHITECTURE.md](ARCHITECTURE.md)

**...migrate from old grammar**
→ [07-MIGRATION-AND-EVOLUTION.md](docs/grammar/07-MIGRATION-AND-EVOLUTION.md)

**...explore bifurcations/fragmentation with metrics**
→ [TNFR Precision & Telemetry Roadmap](docs/TNFR_PRECISION_AND_TELEMETRY_ROADMAP.md) § Phase 5, `benchmarks/bifurcation_landscape.py`

---

## 📡 Telemetry Snapshot Quickstart

Collect the Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) efficiently to identify the performance knee and monitor safety.

- Tool: `benchmarks/telemetry_snapshot.py`
- Output: JSONL in `benchmarks/results/`

Fast knee-light sweep (sampled Φ_s, skip heavy metrics):

```powershell
$env:PYTHONPATH="src"
./test-env/Scripts/python.exe benchmarks/telemetry_snapshot.py `
  --sizes 1000 2000 4000 `
  --topologies er ws ba `
  --seeds 42 `
  --avg-degree 6 `
  --alpha 2.0 `
  --landmark-ratio 0.02 `
  --phi-s-sample 128 `
  --phi-s-cap-landmarks 512 `
  --skip-xi-c --skip-kphi `
  --output benchmarks/results/telemetry_snapshot_knee_light.jsonl
```

Accuracy spot-check (canonical Φ_s with validation):

```powershell
$env:PYTHONPATH="src"
./test-env/Scripts/python.exe benchmarks/telemetry_snapshot.py `
  --sizes 4000 6000 8000 `
  --topologies er `
  --seeds 42 `
  --avg-degree 6 `
  --alpha 2.0 `
  --landmark-ratio 0.02 `
  --validate --sample-size 64 --max-refinements 2 `
  --phi-s-sample 128 `
  --phi-s-cap-landmarks 1024 `
  --skip-xi-c --skip-kphi `
  --output benchmarks/results/telemetry_snapshot_validate.jsonl
```

Recommended defaults:

- Φ_s (fast scans): `--landmark-ratio 0.02`, `--phi-s-sample 128`, `--phi-s-cap-landmarks 512`.
- Φ_s (accuracy-critical): add `--validate --sample-size 64 --max-refinements 2`; cap landmarks at `1024` for 4k–8k.
- Skip flags: `--skip-xi-c --skip-kphi` for knee scans; capture ξ_C and multiscale K_φ in a limited follow-up run.

Canonical telemetry thresholds (interpret post-stabilization):

- |∇φ|: keep below ≈ 0.38 for stable operation (random init will exceed).
- K_φ: local alert at `|K_φ| ≥ 3.0` (confinement/fault zones).
- Φ_s: watch `meta.phi_s_rmae`; target ≤ 1.0 on validation runs.

Guardrail:

- For practicality, avoid n > 4000 in routine runs. Tools now skip sizes above 4000 unless `--allow-large` is provided.

---

## 🧭 Fundamentals Precision Walk

High-precision evolution of the canonical nodal equation with per-step tetrad telemetry and U2 integral tracing.

- Tool: `benchmarks/fundamentals_precision_walk.py`
- Output: JSONL per step under `benchmarks/results/`

Example (PowerShell):

```powershell
$env:PYTHONPATH="src"
./test-env/Scripts/python.exe benchmarks/fundamentals_precision_walk.py `
  --n 2000 --topology er `
  --steps 50 --dt 0.01 `
  --avg-degree 6 --seed 42 `
  --landmark-ratio 0.02 --validate --sample-size 64 --max-refinements 2 `
  --output benchmarks/results/precision_walk_er_2000.jsonl
```

What it records per step:

- Φ_s, |∇φ|, K_φ, ξ_C summaries
- Integral proxy: `integral_vf_dnfr = ∫ νf·ΔNFR dt` (global, mean per step)
- `coherence_length_ratio = ξ_C / mean_distance`

Use this to study convergence (U2), stabilization (U2/U4), and field interplay during evolution at high precision.

### OZ→IL High-Intensity Correlation Sweep (2025-11-17)

- Configuration: `n=2000`, `dt=0.01`, `steps=15`, `oz_fraction=0.2`, `oz_every=1`, ER/WS/BA topologies with identical seeds and avg_degree 6.
- Extended dynamics (`--use-extended-dynamics`) now auto-enable whenever `--oz-il` is set so θ and ΔNFR continue evolving while OZ/IL pulses run; per-step cache invalidation keeps Φ_s, |∇φ|, K_φ, ξ_C fresh.
- Output JSONL snapshots live in `benchmarks/results/precision_walk_{er|ws|ba}_ozil_hi.jsonl` with summary table captured at `benchmarks/results/ozil_hi_correlation_summary.md`.
- Observed correlations (vf·ΔNFR integral vs curvature): ER shows mild negative coupling (mean K_phi ≈ -0.35), WS flips to strong positive (+0.76) while the variance term goes negative (-0.42), BA highlights hub-driven variance growth (std K_phi ≈ +0.54) and large negative deltas (Δ std K_phi ≈ -0.60).
- `--export-node-fields` now logs `curv_mean`, `curv_std`, `grad_mean`, and `grad_std` per degree cohort (top5/next15/mid60/bottom20), enabling gradient-vs-curvature correlation studies directly from the JSONL.
- BA at oz_fraction 0.3 for 25 steps (`precision_walk_ba_ozil_steps25_frac30.jsonl`) shows curvature Δ correlations staying weak while gradient variance flips sign; see `benchmarks/results/ozil_hi_correlation_summary.md` for full tables.
- Long-run BA sweep (45 steps, oz_fraction 0.35) stored at `benchmarks/results/precision_walk_ba_ozil_steps45_frac35.jsonl` produces JSON summaries via `tools/precision_walk_correlator.py`, e.g. `*.summary.json`, so correlation numbers are versioned alongside telemetry.
- Use `./test-env/Scripts/python.exe tools/precision_walk_correlator.py <input.jsonl> <output.summary.json>` to regenerate correlation packs for any precision-walk export.
- Clustered OZ bursts are available via `--oz-bursts N`; e.g. BA runs with `--oz-bursts 3` (45 steps, oz_fraction 0.35) show gradient Δ correlations turning strongly negative while curvature Δ remains near zero.
- `tools/precision_walk_correlator.py` now reports the full tetrad (Φ_s, |∇φ|, K_φ, ξ_C) plus stratified bins so Φ_s/ξ_C correlations are captured alongside gradients/curvature.
- Degree-stratified BA run (`benchmarks/results/precision_walk_ba_ozil_hi_strat.jsonl`, `--export-node-fields`) shows leaves (bottom 20%) keep a strong negative level correlation (-0.63) while hubs nearly decouple (-0.08); Δ correlations stay weakly positive, so the hub/leaf sign flip does not materialize at oz_fraction 0.2.
- Extending WS to 30 steps with oz_fraction 0.3 (`precision_walk_ws_ozil_steps30_frac30.jsonl`) drives `corr(K_phi_std, vf·ΔNFR)` to ≈+0.99 and keeps Δ correlations small (≈0.09), meaning curvature variance finally locks to structural pressure when OZ pulses persist long enough.

Reproduce (PowerShell):

```powershell
$env:PYTHONPATH="src"
./test-env/Scripts/python.exe benchmarks/fundamentals_precision_walk.py `
  --n 2000 --steps 15 --dt 0.01 --avg-degree 6 --oz-il `
  --oz-fraction 0.2 --oz-every 1 --landmark-ratio 0.02 `
  --topology ws --seed 42 --output benchmarks/results/precision_walk_ws_ozil_hi.jsonl
```

---

## 📊 Documentation Quality Status

**Language**: ✅ 100% English (0 Spanish)  
**U6 Status**: ✅ Canonical (2,400+ experiments validated)  
**Grammar Coverage**: ✅ Complete (U1-U6 fully documented)  
**Cross-References**: ✅ Comprehensive bidirectional linking  
**Broken Links**: ✅ 91% reduced (637 → 58)  
**Single Source of Truth**: ✅ Established (AGENTS + UNIFIED_GRAMMAR_RULES + GLOSSARY)

**Last Audit**: 2025-11-11 ([Report](docs/archive/audits/DOCUMENTATION_AUDIT_REPORT.md))

---

## 🔄 Maintenance

This index is actively maintained. If you find:

- Broken links
- Missing documents
- Outdated information
- Unclear navigation

Please open an issue or PR.

**Maintainers**: Keep this index updated when adding/moving/removing major documentation files.

---

**Version**: 3.0  
**Canonical Status**: ✅ ACTIVE  
**Next Review**: 2026-02-11

