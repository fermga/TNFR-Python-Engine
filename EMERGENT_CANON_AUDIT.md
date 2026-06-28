# Emergent Canon Audit — consolidating the post-φ/γ/e TNFR base

**Status**: COMPLETE — superseded by `EMERGENT_DERIVATION_PLAN.md`, which carried
the φ/γ/e re-derivation through Stages 0–5 and now enforces it with a regression
guard (`tests/core_physics/test_emergent_constants_guard.py`) · **Started**:
2026-06-27 · **Completed**: 2026-06-28 · **Owner**: TNFR refactor

## 0. Why this exists

The obsolete constants **φ (golden ratio), γ (Euler–Mascheroni), e (Napier)** were
removed from the engine: only **π** is a genuine structural scale. But the prior
purge only *removed the φ/γ/e formulas from the code/comments* — it left the
**numeric values frozen** (e.g. `K_TOP_FALLBACK = 2.803171` was `φ/γ`;
`MATH_DELTA_NFR_THRESHOLD = 0.067592` was `γ/(eπ)`). These frozen decimals are now
**magic numbers with no emergent meaning**, and a few are outright **non-physical**
(see below). Since all prior TNFR results depended on these constants and their
interactions, the entire numeric base must be **re-derived and reinterpreted** so
that every value either:

1. **emerges** from TNFR structure/dynamics (π, the spectral gap λ₂, the nodal
   coherence `C(t) = 1/(1+mean|ΔNFR|+mean|dEPI|)`, the operator contracts, the
   grammar U1–U6, the derivative-tower tetrad), **or**
2. is an **honest free parameter** (a clean structural default — unit, π-fraction,
   0, or a simple rational — explicitly labeled *tunable*), **or**
3. is an **operational engine knob** (cache/speedup/scoring; *not* nodal physics)
   set to a **plain round value**, explicitly labeled, **or**
4. is **eliminated** (meaningless / dead / redundant).

This file tracks what is done and what remains. **Hold commit/push until a green
milestone is confirmed by the user.** Test command:

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONPATH=(Resolve-Path -Path ./src).Path; & ".\.venv312\Scripts\python.exe" -m pytest -q --no-header -p no:cacheprovider
```

Baseline before this audit: **2201 passed, 12 skipped, 0 failed**.

## 1. Decoder — the frozen φ/γ/e decimals (the "tell")

These recurring decimals across `constants/canonical.py` reveal a φ/γ/e origin and
must be replaced. (Non-exhaustive; any 6-decimal value tied to these is suspect.)

| Frozen decimal | Was (φ/γ/e) | Frozen decimal | Was (φ/γ/e) |
|---|---|---|---|
| `0.041774` | γ/(π·e·φ) | `0.737061` | φ/(φ+γ) |
| `0.067592` | γ/(e·π) | `0.750575` | (e·φ)/(π+e) |
| `0.098503` | γ/(π+e) | `0.618034` | 1/φ |
| `0.135184` | 2γ/(eπ) | `0.381966` | 1/φ² (= 2−φ) |
| `0.155215` | γ/(π+γ) | `1.618034` | φ |
| `0.183733` | γ/π | `2.718282` | e |
| `0.463881` | e/(π+e) | `2.803171` | φ/γ |
| `0.490983` | ~γ·φ/… | `3.025733` | (φ+1)·π/e |
| `0.933955` | ~π·e/… | `5.083204` | φ·π |
| `0.330365` | ~1/(e+γ) | `8539.734223` | large φ/γ/e artifact |

## 2. Categories (counts approximate)

- **A — Emergent / clean (KEEP, ~40):** π-derived (π/2, π/4, 0.9π, 1/(π+1),
  π/(π+1), 1/(2π), 1/(4π), 1/(8π), 2π, sin/cos of π-multiples, exp(LN_2)=2),
  unit/integer free params (0.9, 1.05, 0.10, 0.5, ±1, 1.0, 2, 6, 50, 100, 256),
  honest tolerances (1e-10, 1e-8, 1e-6).
- **B — Frozen φ/γ/e PHYSICS values (RE-DERIVE, ~25):** enter the nodal eq /
  operators / grammar / fields / telemetry. **Top priority** — these define the
  paradigm. (Table in §4.)
- **C — Frozen φ/γ/e ENGINE knobs (CLEAN VALUES, ~200):** scoring weights, cache
  sizes, speedups, prediction horizons, domain (medical/business/viz/example/cli)
  thresholds. Not nodal physics → plain round values or relocation/elimination.
- **D — Inline magic numbers** in `physics/`, `dynamics/`, `operators/`,
  `metrics/` modules (not named constants): audit separately.

## 3. Staged plan

- [x] **Stage 0** — Full inventory + categorization + this tracking file.
- [x] **Stage 1** — **Tier-1 canonical physics re-derivation** (§4). DONE
      (suite green 2201). All Tier-1 rows re-derived or eliminated; the
      `PHYSICS_*_DEPENDENCY` empirical-fit coefficients were reclassified to
      Tier-2 (Stage 2). The canonical physics base is now emergent.
- [x] **Stage 2** — Tier-2 engine-knob clean-up (§5): **DONE**. 180 frozen φ/γ/e
      decimals rounded to clean operational values; all dead domain constants
      (MEDICAL_/BUSINESS_/VIZ_/EXAMPLE_/BENCH_/THERAP_/CLI_/SCRIPT_/TOOL_/UTILS_)
      and the dead `CANONICAL_CONSTANTS` registry + PHASE_8/9 dicts ELIMINATED;
      engine-used `SDK_*` relocated into `sdk/builders.py`. canonical.py ~770 →
      ~565 lines. Suite green 2201.
- [x] **Stage 3** — Inline magic-number audit in modules (Category D). DONE
      (core modules were ~95% clean). Fixed the residual φ/γ/e artifacts +
      operator-gain inconsistencies; suite green 2201.
- [x] **Stage 4** — Reinterpretation: updated `theory/TNFR_NUMBER_THEORY.md`
      §7.5 (the stale arith-trio "recalibration" table → the canonical π-derived
      tetrad thresholds) and added a CHANGELOG `[Unreleased]` consolidation entry.
      DONE.

## 4. Tier-1 PHYSICS constants — derivation table (Stage 1)

Legend: ☐ pending · ☑ done. "Derivation" = the emergent/honest replacement.

| Constant | Was | → Derivation (emergent / honest) | Status |
|---|---|---|---|
| `K_PHI_THRESHOLD` (arith) | 3.227450 | **NON-PHYSICAL (>π, unreachable)** → π (or eliminate the arith-recalibrated trio) | ☐ |
| `PHI_S_THRESHOLD` (arith) | 0.745219 | π/4 (canonical) or eliminate | ☐ |
| `GRAD_PHI_THRESHOLD` (arith) | 0.259117 | eliminate / canonical heuristic | ☐ |
| `MAX_STRUCTURAL_FREQUENCY` | 5.083204 (φπ) | **2π** (one phase cycle, = VF_MAX) | ☐ |
| `MIN_STRUCTURAL_FREQUENCY` | 0.183733 (γ/π) | 0.0 floor (νf=0 death) or 1/(2π) tunable | ☐ |
| `MATH_DELTA_NFR_THRESHOLD` | 0.067592 (γ/(eπ)) | **0.5** separator (prime ⟺ ΔNFR=0 exact) | ☐ |
| `MATH_DELTA_NFR_THRESHOLD_2X` | 0.135184 | 1.0 (= 2×0.5) or eliminate | ☐ |
| `K_TOP_FALLBACK_CANONICAL` | 2.803171 (φ/γ) | clean spectral fallback (π-fraction or 1.0) | ☐ |
| `K_TOP_MIN_CANONICAL` | 0.041774 | 1/(8π) (= KL_MIN floor) | ☐ |
| `K_TOP_MAX_CANONICAL` | 0.933955 | π-fraction near 1 (tunable) | ☐ |
| `GLYPH_SELECTOR_MARGIN_CANONICAL` | 0.041774 | **= KL_MIN (1/(8π))** (comment already claims this) | ☐ |
| `AU_CURVATURE_PERMISSIVE_THRESHOLD` | 3.025733 ((φ+1)π/e) | **0.96·π** (permissive |K_φ|, between 0.9π and π) | ☐ |
| `CRITICAL_EXPONENT` | 0.183733 (γ/π) | honest heuristic regime scale (decouple from GRAD_PHI) | ☐ |
| `GRAD_PHI_CANONICAL_THRESHOLD` | 0.183733 (γ/π) | heuristic |∇φ| early-warning, clean π-fraction (tunable) | ☐ |
| `INTEGRATORS_EPI_MARGIN_CANONICAL` | 0.098503 | 0.1 (tunable margin) | ☐ |
| `INTEGRATORS_J_PHI_SCALE_CANONICAL` | 0.098503 | 0.1 (tunable scale) | ☐ |
| `INTEGRATORS_SYNTHETIC_DIV_CANONICAL` | -0.155215 | clean (−1/(2π) or −0.15) | ☐ |
| `INTEGRATORS_FLUX_FALLBACK_CANONICAL` | 0.737061 | UM_COMPAT = π/(π+1) | ☐ |
| `INTEGRATORS_SIGMOID_OFFSET_CANONICAL` | 0.618034 | 0.5 (unit midpoint) | ☐ |
| `FEEDBACK_COHERENCE_TOL_LOW` | 0.13937 | 0.15 (tunable) | ☐ |
| `FEEDBACK_COHERENCE_TOL_HIGH` | 0.098503 | 0.1 (tunable) | ☐ |
| `FEEDBACK_EPI_THRESHOLD` | 0.330365 | 1/3 (tunable) | ☐ |
| `DYNAMICS_ADELIC_DRIFT_CANONICAL` | 0.098503 | 0.1 | ☐ |
| `DYNAMICS_ADELIC_DT_STEP_CANONICAL` | 0.041774 | DT_MIN = 1/16 | ☐ |
| `DYNAMICS_SI_HI_THRESHOLD_CANONICAL` | 0.618034 | π/(π+1) high-coherence (or 0.75 tunable) | ☐ |
| `MATH_COHERENCE_MIN_CANONICAL` | 0.098503 | 0.1 | ☐ |
| `MATH_TOLERANCE_CANONICAL` | 8539.734223 | **honest tolerance** (verify usage; likely a large cap) | ☐ |
| `PHYSICS_HOTSPOT_FRACTION_CANONICAL` | 0.098503 | 0.1 | ☐ |
| `CONFIG_INIT_VF_MEAN / VF_STD / EPI_LATENT_MAX / VF_BASAL / EPSILON_MIN` | various | clean (0.5 / 0.15 / 0.5 / 0.5 / 0.1) | ☐ |
| `PHYSICS_*_DEPENDENCY` (N_NODES/K_DEGREE/P_REWIRE/EXPECTED_CORR) | various | reclassify → Tier-2 (empirical fits, not nodal physics) | ☐ |

Each row: verify consumers + run the suite for the affected group before moving on.
Some thresholds are consumed in tests that pin the old value — update those tests
to the new emergent value (documenting the structural rationale).

## 5. Tier-2 engine knobs — bulk rule (Stage 2)

For every Category-C constant (OPT_ORCH_*, PATTERN_*, FFT_*, PATTERNS_*,
SELF_OPT_*, MULTIMODAL_*, CACHE_OPT_*, INTEGRATION_*, NODAL_OPT_*, STRUCT_CACHE_*,
CYCLE_* frozen, ALGEBRA_*, EMERGENT_*, OPERATORS_PATTERN_*, SDK_*, MEDICAL_*,
BUSINESS_*, VIZ_*, UTILS_* frozen, EXAMPLE_*, BENCH_*, THERAP_*, CLI_*, SCRIPT_*,
TOOL_*, MIN_BUSINESS_COHERENCE):

- Replace the frozen φ/γ/e decimal with a **plain round operational value** that
  preserves the intended magnitude (e.g. `2.803171 → 2.8`, `0.737061 → 0.75`,
  `0.618034 → 0.6`, `8.539734 → 8.5`), keeping the `# operational tuning (not TNFR
  physics)` label. Drop the false precision.
- Where a knob is already `= π` or `= UM_COMPAT_THRESHOLD` etc., keep it.
- **Eliminate** dead/duplicate aliases and empty registry plumbing.
- **Consider relocating** domain-application constants (MEDICAL_/BUSINESS_/
  EXAMPLE_/THERAP_/CLI_/SCRIPT_/TOOL_/VIZ_) out of `constants/canonical.py` into
  their own modules — they are not TNFR canon. (Decide with user.)

## 6. Status log

- **2026-06-27** — Stage 0 complete: full canonical.py inventory (~300 constants),
  categorized; this tracker created. Beginning Stage 1 (Tier-1 physics).
- **2026-06-27** — **Stage 1 COMPLETE** (suite green 2201 passed / 0 failed, twice):
  - π-derived: `MAX_STRUCTURAL_FREQUENCY=2π`, `MIN_STRUCTURAL_FREQUENCY=1/(2π)`,
    `AU_CURVATURE_PERMISSIVE=0.96π`, `GLYPH_SELECTOR_MARGIN=1/(8π)`,
    `CRITICAL_EXPONENT=π/16`, `GRAD_PHI_CANONICAL_THRESHOLD=π/16`,
    `DYNAMICS_SI_HI=π/(π+1)`, `K_TOP_MIN=1/(8π)`, `K_TOP_MAX=1.0`, `K_TOP_FALLBACK=π`.
  - Genuine emergent derivation: `MATH_DELTA_NFR_THRESHOLD=0.5` (the unit-gap
    midpoint — prime ⟺ ΔNFR=0 exactly, every composite ΔNFR>1, from the unit-coeff
    arithmetic ΔNFR formula ζ(ω−1)+η(τ−2)+θ(σ/n−(1+1/n))); `_2X=2×base=1.0`.
  - Clean free-params (tunable): INTEGRATORS_* (0.1/0.1/−0.15/0.75/0.5),
    FEEDBACK_* (0.15/0.1/1/3; DNFR auto), DYNAMICS_ADELIC_* (0.1, 1/16),
    MATH_COHERENCE_MIN=0.1, MATH_TOLERANCE=1e4 (→PRECISION=100),
    PHYSICS_HOTSPOT_FRACTION=0.1, CONFIG_* (0.5/0.15/0.5/0.5/0.1).
  - Eliminated vestigial arith trio (`PHI_S_THRESHOLD`/`GRAD_PHI_THRESHOLD`/
    `K_PHI_THRESHOLD`; the last was 3.2275 > π — non-physical/unreachable).
  - No tests pinned values (verified via consumer map); CRITICAL_EXPONENT's
    zeta-bridge test is self-consistent. Stale `theory/TNFR_NUMBER_THEORY.md`
    table (lines ~311-313) referencing the removed arith trio → fix in Stage 4.
- Beginning Stage 2 (Tier-2 engine knobs, ~200 frozen φ/γ/e decimals).
- **2026-06-27** — **Stage 2 value-cleanup COMPLETE** (suite green 2201 / 0 failed):
  180 frozen φ/γ/e engine-knob decimals rounded to clean ≤2-decimal operational
  values (e.g. 2.803171→2.8, 0.737061→0.74, 0.618034→0.62, 0.098503→0.1,
  −0.155215→−0.16, 8.539734→8.5). 3 tiny measured values (1.8e-05, 0.00147,
  0.001352) correctly left. Aliases auto-followed their rounded bases. ZERO test
  changes needed. Remaining: Stage 2b relocation (domain constants — needs user
  decision), Stage 3 (inline magic numbers in modules), Stage 4 (doc
  reinterpretation incl. the stale TNFR_NUMBER_THEORY arith-trio table).
- **2026-06-27** — **Stage 2b COMPLETE** (suite green 2201 / 0 failed): domain
  constants were all DEAD (no imports anywhere) → eliminated MEDICAL_/BUSINESS_/
  VIZ_/EXAMPLE_/BENCH_/THERAP_/CLI_/SCRIPT_/TOOL_/UTILS_ (~45 constants) + the dead
  `CANONICAL_CONSTANTS` registry and PHASE_8/9 dicts. `SDK_*` (used only by
  sdk/builders.py) relocated into that module as plain locals; 4 unused SDK_
  dropped. canonical.py ~770 → ~565 lines. Beginning Stage 3 (inline magic
  numbers in physics/dynamics/operators/metrics modules).
- **2026-06-27** — **Stage 3 COMPLETE** (suite green 2201 / 0 failed): core modules
  were ~95% clean. Fixed residual artifacts: `operators/__init__.py` gains
  `0.9015`/`1.0676` → canonical `SHA_VF_FACTOR`/`NUL_SCALE_FACTOR`/`VAL_SCALE_FACTOR`;
  `bifurcation.py` `vf/2.71828`(e)→`vf/2.0`, `epi/0.922442`→`epi/0.9`, weights
  →0.46/0.26/0.14; `cycle_detection.py` `0.098503`/`0.167944`→0.1/0.17;
  `variational.py` `16.18034`(10·φ)→16.0; `signatures.py` Au drift `2.19525`→0.7·π.
  Beginning Stage 4 (doc reinterpretation).
- **2026-06-27** — **Stage 4 COMPLETE / AUDIT CLOSED** (suite green 2201 / 0 failed):
  rewrote `theory/TNFR_NUMBER_THEORY.md` §7.5 to the canonical π-derived tetrad
  thresholds (dropping the removed arith trio incl. the non-physical 3.2275);
  added the emergent-canon consolidation entry to `CHANGELOG.md [Unreleased]`. The
  engine's canonical base is now fully emergent (π / spectral / nodal) or honest
  free-parameters. **Out of scope (flagged):** the auxiliary `primality-test/` and
  `factorization-lab/` subprojects keep their own constants (some still reference
  φ/γ/e) — separate from the engine. Commit/push held pending user confirmation.
- **2026-06-27** — **Stage 5: THIRD AUDIT** (errors / contradictions / redundancies /
  synergies; suite green 2201 / 0 failed). Found + fixed:
  - **Contradiction (frequency bounds defined 4 ways):** canonical's
    `MIN/MAX_STRUCTURAL_FREQUENCY` (2π, 1/(2π)) duplicated `VF_MIN/VF_MAX` and were
    *misused* in `lifecycle.py` as the `mutation_dnfr` ΔNFR threshold. Removed the
    pair; `lifecycle.mutation_dnfr` → plain `5.0` (½ collapse). (`unified_numerical`
    / `validation` keep a labeled *loose practical* [0, 1000] range, superseded by
    the 2π dynamics clamp — left as-is.)
  - **Error (K_TOP fallback outside its own clamp):** `K_TOP_FALLBACK` was π (3.14)
    but the clamp is [1/(8π), 1.0], so `topology.py` returned π unclamped in one
    path and 1.0 in another. Fixed: `K_TOP_FALLBACK = K_TOP_MAX = 1.0` (consistent).
  - **Stale comments (value ≠ comment):** `REMESH_SIMILARITY`/`FEEDBACK_TARGET`
    `≈0.7371`→0.7616; `PHASE_GRADIENT_THRESHOLD` `≈0.1837`→0.196; `MATH_PRECISION`
    `≈85.4`→100; `defaults_core VF_MAX` `≈5.0832`→2π; the tetrad "three of four
    first-principles" header (Φ_s/K_φ are π-derived, |∇φ| is heuristic).
  - **Example artifact:** `examples/.../97_goldbach` hardcoded `DELTA_PHI_MAX=0.1837`
    (=γ/π, also a naming error — canonical Δφ_max is π/2) → `PHASE_MATCH_THRESHOLD=0.2`.
  - **Synergy (redundancy consolidation):** introduced the single-sourced coherence
    band `FRAGMENTATION_THRESHOLD = 1/(π+1)` and `HIGH_COHERENCE_THRESHOLD = π/(π+1)`;
    the six members of the `1/(π+1)` / `π/(π+1)` families now reference them.
  - **Flagged (not changed):** ~120 operational engine knobs (CACHE_OPT_/INTEGRATION_/
    SELF_OPT_/OPT_ORCH_/PATTERNS_/NODAL_OPT_/etc.) still live in canonical.py — they
    are engine-used (can't eliminate) but are NOT TNFR physics; relocating them to
    their consuming modules (like SDK) is the next optimization. The `0.1` value
    recurs ~18× as an operational neutral default (independent tunables — fine).
- **2026-06-27** — **Stage 6: OPERATIONAL-KNOB RELOCATION** (suite green 2201 / 0
  failed). Acted on the Stage-5 flag: split the ~150 operational engine-tuning
  knobs out of `canonical.py` into a new dedicated module
  `src/tnfr/constants/operational.py` (explicitly "engine tuning, NOT physics";
  imports only `PI` from canonical — one-way dependency, canonical never imports
  operational). A parallel `engines/constants/operational.py` star-shim mirrors the
  existing `canonical` shim. **Result:** `canonical.py` 239 → **89 numeric constants
  (pure structural / physics)**; `operational.py` holds the **150** moved knobs;
  union reproduces the original snapshot **exactly** (0 leaks, 0 drift, verified via
  a name→value golden snapshot before/after). Families moved: OPT_ORCH_,
  MULTIMODAL_CACHE_, FFT_ (cutoffs/importance/engine/coordination), ARITHMETIC_FFT_,
  PATTERNS_, SELF_OPT_, EMERGENT_ (centralization), CACHE_OPT_, UNIFIED_CACHE_,
  INTEGRATION_, NODAL_OPT_, STRUCT_CACHE_, CYCLE_ (balance/fallback/min-health),
  PATTERN_*_WEIGHT, ALGEBRA_*_TOLERANCE, PHYSICS_* (network-study), OPERATORS_*
  (pattern scoring), MIN_BUSINESS_COHERENCE. **Kept in canonical** (structural):
  the π-derived aliases PHYSICS_GRAD_THRESHOLD (π/16) / PHYSICS_CURVATURE_HOTSPOT
  (0.9π) / PHYSICS_HOTSPOT_FRACTION, CYCLE_OPTIMAL_BALANCE (1/(π+1)),
  THERAPEUTIC_EXCELLENT (sin π/3), INTEGRATORS_*, FEEDBACK_*, tetrad/phase/VF/EPI/KL/
  DT bounds, operator gains. **26 consumers** redirected (`canonical`→`operational`),
  mixed files split to preserve their structural imports (PI, CYCLE_OPTIMAL_BALANCE,
  tetrad aliases); `backend_config.py` and `physics/interactions.py` unchanged (they
  use only kept structural aliases). Bonus: consolidating the "one-import-per-block"
  style dropped dozens of now-wrong `π/e` / `4·φ²/π²` era comments.
- **2026-06-27** — **Stage 7: THIRD DEEP AUDIT** (frozen φ/γ/e residue +
  internal coherence; suite green 2201 / 0 failed). Method: diffed the committed
  pre-purge `HEAD` canonical.py (still φ/γ/e) against current to find values the
  purge *froze* (replaced a φ/γ/e expression with its evaluated decimal, mislabeled
  "tunable") and stale comments citing the old value. Findings + fixes:
  - **Coherence violation (load-bearing) — NUL densification:** `canonical.py` states
    `NUL_DENSIFICATION_FACTOR = 1/λ ≈ 1.111` ("DERIVED, conserves the nodal-equation
    product νf·ΔNFR"), but `operators/__init__.py` hardcoded `densification_default =
    1.35` ("extra amplification beyond 1/λ") citing the stale 0.9015. Reconciled the
    operator to the canonical `1/λ` and fixed the contradictory comment.
  - **Frozen transcendentals → clean free values:** `FEEDBACK_LEARNING_RATE` 0.043
    (=e^(−π)) → 0.05; `FEEDBACK_TAU_ADAPTIVE` 0.155 (=γ/(π+γ)) → 0.15;
    `AU_CURVATURE_PERMISSIVE` 0.96·π (0.96=(φ+1)/e) → 0.95·π;
    `STRESS_NORM` 0.0985 (=γ/(π+e)) → 0.1; `unified_numerical` `MIN_BUSINESS_COHERENCE`
    0.750575 (=(e·φ)/(π+e)) → 0.75 and `PHASE_GRADIENT_STABILITY` 0.183733 (=γ/π) → π/16;
    six `RemeshDefaults`/`GLYPH_THRESHOLDS`/`SELECTOR_WEIGHTS` `round(0.92.., 3)` frozen
    round-of-literal artifacts → plain decimals; the misleading `_EXP_NEG_PI` alias →
    `_FEEDBACK_LR`; the `(e*phi)/(pi+e)` strong-coherence formula in SDK docstrings removed.
  - **De-obfuscation:** `STRUCTURAL_ESCAPE = math.exp(LN_2)` → plain `2.0` (= EPI span).
  - **VAL_MIN_EPI** precondition fallback `0.2` → the canonical `1/(2π)`.
  - **~55 stale comments** citing old φ/γ/e values updated to the current value across
    `defaults_core`, `bifurcation` (code/doc contradiction: docstring still said
    "νf/e", "EPI/0.922", old %), `patterns`, physics (`interactions`, `signatures`,
    `phase_transition`, `canonical`, `integrity`, `calibration`), `riemann/zeta_bridge`,
    telemetry, grammar, and the FFT/cache operational modules; the most pervasive was
    `≈ 0.1837` (γ/π) → `≈ 0.196` (π/16, ×~16).
  - **Confirmed NOT used as an exponent:** `CRITICAL_EXPONENT` (π/16) is a heuristic
    reference scale (function arg / spectral buffer), never `x ** CRITICAL_EXPONENT`.
  - **Legitimately kept:** the `riemann/` golden-ratio / Euler-Mascheroni / Napier
    values are *explicitly labeled non-resonant probe frequencies* ("NOT TNFR
    structural scales"), the correct use of those numbers as negative controls.
  - **Flagged (user decision):** AGENTS.md §7 cites strong-coherence cut `C > 0.7506`
    (= frozen (e·φ)/(π+e)); code now uses the clean operational `0.75`. Recommend
    updating AGENTS.md to `0.75`, or promoting the cut to the canonical band gate
    `π/(π+1) ≈ 0.7616`. Also `PHI_S_GOLDEN_THRESHOLD` (riemann alias for π/2) is a
    misnomer (no longer "golden").
- **2026-06-27** — **Stage 8: DOCUMENTATION → EMERGENT CANONICITY** (suite green
  2201 / 0 failed). Resolved the Stage-7 flag by **promoting** the documented values
  to the genuinely-emergent π-derived ones across AGENTS.md + the
  `.github/agents/my-agent.md` mirror + theory/docs/examples + code docstrings:
  - **Arithmetic error fixed:** `π/(π+1)` is `0.7585`, **not** `0.7616` (the latter
    does not complement `1/(π+1)=0.2415` to 1). Corrected the comment value in
    `canonical.py` (×5) and `defaults_core.py`, and everywhere in the docs.
  - **Strong-coherence cut → emergent band gate:** `0.7506` (frozen (e·φ)/(π+e))
    → `π/(π+1) ≈ 0.7585`. The SDK `COHERENCE_STRONG` was re-aliased from the
    operational `MIN_BUSINESS_COHERENCE` (0.75) to the emergent
    `HIGH_COHERENCE_THRESHOLD = π/(π+1)`; `MIN_BUSINESS_COHERENCE` (0.75) stays the
    separate operational business-health knob. Fixed the now-self-contradictory
    CONTRIBUTING.md "correct vs arbitrary" example (it had `MIN_BUSINESS_COHERENCE`
    = the "correct" value but it now equals the "arbitrary" 0.75) → uses the
    emergent gate.
  - **Φ_s confinement → π-derived (was φ-framed):** drift `Δ Φ_s < π/2 ≈ 1.571`
    (half phase-wrap, was φ ≈ 1.618), per-node `|Φ_s| < π/4 ≈ 0.785` (quarter
    phase-wrap, was empirical 0.7711) — in AGENTS.md §3/§6/§7, `GLOSSARY`,
    `PHYSICS_VERIFICATION` (canonicity MODERATE→HIGH), `physics/__init__` (+ its
    doctest), `conservation`, `variational`, `grammar_u6`, `telemetry/constants`,
    and the two physics-regime examples.
  - **SPIRAL_ATTRACTORS_AND_LOGARITHMIC_DYNAMICS.md — DELETED** (with its broken
    demo `examples/02_physics_regimes/32_spiral_attractors_demo.py`). It was a
    pre-purge φ/γ/e "four constants" thesis whose distinctive claims were false:
    the "golden ratio as dynamical attractor" check was **circular** (set
    `b = 2·ln(φ)/π` by hand, then "verified" ratios = φ), the demo **imported the
    purged φ/γ/e constants** (could not run — so §6.3 "Status: Validated" was
    false), and §7 ("the fourth constant γ") is pure pre-purge paradigm. The only
    true content (log spirals from rotation + growth, free `b = νf·k/ω`) is trivial
    and non-distinctive. References removed from `theory/README.md`,
    `FUNDAMENTAL_THEORY.md`, `examples/README.md`.
  - **Verified-correct (left as-is):** the website brief and several theory files
    (`FUNDAMENTAL_THEORY`, `MINIMAL_STRUCTURAL_DEGREES`, `GLOSSARY` §VON_KOCH)
    already document the π/2–π/4 supersession; the `riemann/` probes stay.
- **2026-06-27** — **Stage 9: FULL theory/ DOC AUDIT** (every `theory/*.md` checked
  for residual φ/γ/e false claims; yang_mills tests green 18/18; no engine code
  changed except cosmetic docstrings). **Verdict: no further deletions needed** —
  SPIRAL_ATTRACTORS (Stage 8) was the only doc whose *thesis* was false; all 26
  remaining docs are genuinely emergent and only carried scattered stale φ/γ/e
  refs. Fixes applied (the code was already clean — these were doc-only, except 2
  yang_mills docstrings):
  - **FUNDAMENTAL_THEORY / EXTENDED_FIELDS / GAUGE_SYMMETRY / PHYSICAL_REGIME:**
    `Δ Φ_s < φ` → `π/2`; `Si < 1.5/(π+γ)` → `Si < 0.4`; AU permissive curvature
    `(φ+1)π/e` → `0.95π`; `V < ½φ²·N` → `½(π/2)²·N`; heuristic `γ/π` → `π/16`.
  - **STRUCTURAL_OPERATORS (≈23 refs):** all operator-gain tables/formulas → the
    operational `defaults_core.py` values — IL `φ/(φ+γ)≈0.737`→`0.75`; OZ/NUL-dens
    `φ/γ≈2.803`→ OZ `2.0`, NUL densification `1/λ≈1.111` (was wrongly `2.803`);
    VAL `1+γ/(π·e)≈1.067`→`1.05`; SHA/NUL scale `1−γ/(π+e)≈0.902`→`0.9`;
    RA `e^{−φ}≈0.198` and ZHIR `φ/(e+γ)≈0.489` → labeled operational/free.
  - **STRUCTURAL_CONSERVATION:** IL/OZ Lyapunov proof ρ_IL `0.737`→`0.75`, ρ_OZ
    `2.803`→`2.0` (product `1.5>1`, argument preserved); `|Φ_s|<φ`→`π/2`; the full
    13-operator energy table (VAL 1.05, AL 0.10, SHA 0.9, NUL-dens 1.111) + the
    sequence-contractiveness example `(1+3.0)(1−0.438)⁴≈0.40<1`.
  - **STRUCTURAL_STABILITY / TNFR_VARIATIONAL_PRINCIPLE / TNFR_NUMBER_THEORY:** IL/OZ
    rows → operational; Mutation ΔNFR `φ×π≈5.083`→`5.0`; `γ_c=γ/π`, noise floor,
    chirality → operational calibrated; `|Φ_s|<φ`→`π/2`; `|∇φ|` threshold `γ/π`→`π/16`;
    "Tetrahedral confinement" label → "U6 structural-potential confinement".
  - **TNFR_YANG_MILLS** (+ `src/tnfr/yang_mills/structural_gap.py` 2 docstrings):
    V_U6 "normalised by φ²" / "Φ_s²/φ²" / `ρ_U6 = max|Φ_s|/φ` → `(π/2)²` /
    `U6_STRUCTURAL_POTENTIAL_LIMIT` (the code already computed `/(π/2)²`; only the
    docstrings + note were stale).
  - **CATALOG_TYPE_HYGIENE_PROGRAMME:** removed the false "Universal Tetrahedral
    Correspondence γ ↔ |∇φ|" claim and the `FUNDAMENTAL_THEORY — Universal
    Tetrahedral Correspondence` cross-ref; stale `canonical.py:506`→`:289`.
  - **TNFR_RIEMANN_RESEARCH_NOTES (20 refs — the subagent missed these):** the
    refuted **"Universal Tetrahedral Correspondence"** (φ↔Φ_s, γ↔|∇φ|, π↔K_φ, e↔ξ_C)
    purged throughout — the explicit mapping → the minimal structural-field tetrad
    (only π structural); the three inter-prime kernels relabeled **exploratory, not
    canonical**; δ_coh `derived from UTC` → exploratory γ/π prefactor; `U6 ΔΦ_s<φ`→`π/2`;
    DNFR/SI weights `(φ/γ/π/e-derived)` → operational tunable (+ stale `:57/:65/:150`
    line anchors → `:85/:93/:186`); SELECTOR_WEIGHTS `{π/(π+e),1/(π+1),γ/(π+1)}` →
    the operational `{0.536, 1/(π+1)≈0.241, 0.139}`; 8 narrative "nor from the
    Universal Tetrahedral Correspondence" tool-lists → "structural-field tetrad".
  - **Confirmed clean (no changes):** MINIMAL_STRUCTURAL_DEGREES, MATHEMATICAL_DYNAMICS_BASIS,
    GLOSSARY (§VON_KOCH deprecation honest), UNIFIED_GRAMMAR_RULES, DISSIPATIVE_AND_OPEN_SYSTEMS,
    EMERGENT_ONTOLOGY, REMESH_INFINITY_DERIVATION, APPLIED_STRUCTURAL_ANALYSIS,
    NUCLEUS_A/B, TNFR_BSD/HODGE/P_VS_NP/NAVIER_STOKES research notes (NAVIER_STOKES
    `2.803702` is simulation data; Riemann `sin(γ_em·T)` is an explicit non-resonant
    Euler-Mascheroni control — both legitimate). `defaults_core.py`/`canonical.py`
    already fully purged (weights labeled "tunable"; GAMMA/PHI/E intentionally absent).
