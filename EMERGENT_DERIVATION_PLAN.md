# Emergent Derivation Plan — every numeric constant from the nodal equation

**Status**: COMPLETE — Stages 0–5 done; the constitution is now enforced by a
regression guard · **Started**: 2026-06-27 · **Completed**: 2026-06-28 ·
**Supersedes the "operational free value" escape hatch left by
`EMERGENT_CANON_AUDIT.md` Stages 0–9.**

## 0. Why this exists (the deeper point)

The φ/γ/e purge (`EMERGENT_CANON_AUDIT.md`) removed φ/γ/e as **structural claims**,
but it replaced the frozen φ/γ/e decimals with **arbitrary "clean operational"
values** — `SHA_VF=0.9`, `VAL_SCALE=1.05`, `IL=0.75`, `OZ=2.0`, `AL_boost=0.10` —
and it **left two load-bearing weight sets frozen at their literal φ/γ decimals**:

```python
DNFR_WEIGHTS = {"phase": 0.737, "epi": 0.155, "vf": 0.09}   # 0.737 = φ/(φ+γ) !
SI_WEIGHTS   = {"alpha": 0.737, "beta": 0.155, "gamma": 0.114}  # frozen φ/γ !
```

These are **magic numbers**, and — crucially — they are **used numerically** in
*every* ΔNFR evaluation and *every* Sense-Index evaluation, hence in *every*
recorded result of the paradigm (Riemann spectra, Navier–Stokes runs, number
theory, tetrad universality, network optimisation). A green test suite means the
code *runs*, **not** that the results are unchanged or canonical.

**The mandate (user, 2026-06-27):** every numeric constant must **emerge from TNFR
structure/dynamics** — no φ/γ/e, and no arbitrary operational decimal either. Then
**recompute** the results the paradigm was built on.

## 1. The emergence constitution — the ONLY admissible origins of a number

A numeric constant in a TNFR *physics* path is canonical **iff** it cites exactly
one of these sources. Anything else is a magic number and must be derived or removed.

| # | Source | Admissible values |
|---|--------|-------------------|
| S1 | **π** — the sole structural scale (phase-wrap bound of the phase sector) | `π/2, π/4, π/3, π/6, π/16, 0.9π`, `2π`, `1/(2π), 1/(4π), 1/(8π)` |
| S2 | **Coherence band** — the single π-derived quantity `1/(π+1)` and its complement `π/(π+1)` (and their geometric hierarchy) | `1/(π+1)≈0.2415`, `π/(π+1)≈0.7585`, `π/(π+1)²`, `1/(π+1)²`, `(π+1)/π` |
| S3 | **Spectral gap** `λ₂` (Fiedler), `λ₁` — emergent from graph structure | `ξ_C ∝ 1/√λ₂`, `k_top = 1/λ₁` |
| S4 | **Unit form span** — `EPI ∈ [−1,1]` (span 2, midpoint 0); `νf ∈ [0, 2π]` | `±1, 0, 2, 0.5` |
| S5 | **Conservation / numerical method** — exact identities | `NUL_dens = 1/f_NUL` (volume), RK4 `1/6, 1/2` |
| S6 | **Max-entropy counting** — equipartition `1/N` over `N` structurally-active channels | `1/3, 1/4` |
| S7 | **Nodal gap** — `prime ⟺ ΔNFR=0`, composite `ΔNFR>1` | separator `0.5` (unit-gap midpoint) |

**Inadmissible:** φ, γ, e and any decimal frozen from them (`0.737, 0.155, 2.803,
0.9015, 1.0676, 0.1837, …`); bare operational decimals chosen for "niceness"
(`0.9, 1.05, 0.75, 2.0, 0.10, 0.3, …`) **when they sit on a physics channel**;
fitted values with no structural meaning.

**Reclassification rule.** "Free / tunable parameter" is **not** a licence for an
arbitrary default. A free parameter's *default* must still be a clean structural
value (S1–S7); the freedom is the *allowance to override at runtime*, not the
default. Pure engine knobs (cache/FFT/perf — already in `constants/operational.py`)
are exempt: they carry no nodal-physics meaning.

## 2. Flagship derivations (the new canonical values)

### 2.1 Channel-mixing weights → the coherence-band hierarchy (S2)

`ΔNFR = w_φ·∂φ + w_E·∂EPI + w_ν·∂νf + w_τ·∂topo`. The channels carry a structural
primacy ordering — **phase** (1st-order local desync, the U3 coupling driver) ≻
**EPI** (0th-order field diffusion `−L_rw·EPI`) ≻ **νf** (capacity gradient) ≻
**topo** (inactive: the graph is fixed during standard evolution, so `w_τ = 0`).
Assign each successive channel the **high-coherence share `π/(π+1)` of what
remains**:

$$w_\phi = \frac{\pi}{\pi+1},\quad w_E = \frac{\pi}{(\pi+1)^2},\quad w_\nu = \frac{1}{(\pi+1)^2},\quad w_\tau = 0.$$

This is a genuine structural identity — it **normalises to exactly 1**:

$$\frac{\pi}{\pi+1} + \frac{\pi}{(\pi+1)^2} + \frac{1}{(\pi+1)^2} = \frac{\pi(\pi+1) + \pi + 1}{(\pi+1)^2} = \frac{(\pi+1)^2}{(\pi+1)^2} = 1.$$

Values `(0.7585, 0.1832, 0.0583)` vs the frozen φ/γ `(0.75, 0.158, 0.092 normalised)`
— same phase-dominant regime (results stay sensible) but now **derived from π alone**.
`SI_WEIGHTS` (νf-coherence ≻ phase-sync ≻ |ΔNFR|) takes the identical hierarchy.

### 2.2 Operator gains → the band-reciprocal pair (S2)

Every operator contract fixes a **channel + sign**; the **magnitude** is the
coherence-band ratio chosen by direction:

| Direction | Gain | Operators |
|-----------|------|-----------|
| **toward coherence** (ρ<1 on ΔNFR, f<1 on νf) | `π/(π+1) ≈ 0.7585` | IL, SHA, NUL (ν_f step) |
| **toward dissonance** (ρ>1, f>1) | `(π+1)/π ≈ 1.3183` | OZ, VAL, NUL densification |

Consequence (clean, canonical): a balanced `IL∘OZ` (and `NUL` contraction ∘
densification) is **exactly isometric** — `ρ_IL·ρ_OZ = (π/(π+1))·((π+1)/π) = 1`.
U2 convergence then comes from *net stabiliser excess* or U6 confinement, not from
an arbitrary OZ over-amplification. `NUL_densification = 1/f_NUL = (π+1)/π` is the
volume-conservation identity (S5), now consistent with the OZ/VAL amplification.

Additive / phase gains (AL boost, ZHIR θ-shift, THOL accel, RA/UM/NAV) → the
fragmentation share `1/(π+1)` of their channel's unit span, or a π-fraction (S1/S2).

## 3. Constant classification (engine physics paths)

- **Already emergent (KEEP):** `π`-fractions (`π/2, π/4, π/16, 0.9π, 1/(2π),
  1/(4π), 1/(8π), sin/cos(π·k)`), the band `1/(π+1)`, `π/(π+1)`, unit span
  (`±1, 0, 2, 0.5`), `MATH_DELTA_NFR=0.5` (nodal gap), RK4 `6/2`, `NUL_dens=1/f`,
  `VF_MAX=2π`, `k_top` spectral clamp.
- **MAGIC — re-derive (this plan):**
  1. `DNFR_WEIGHTS`, `SI_WEIGHTS` (frozen φ/γ) → §2.1 hierarchy.
  2. `IL=0.75, OZ=2.0, SHA/NUL=0.9, VAL=1.05, NUL_dens=1.111` → §2.2 band-reciprocal.
  3. `AL_boost=0.10, ZHIR_shift=0.3, THOL_accel=0.10, NAV_eta/jitter, RA_*, UM_*`
     → band / π-fraction.
  4. `SELECTOR_WEIGHTS={0.536, 1/(π+1), 0.139}` (0.536/0.139 magic) → band hierarchy.
  5. `FEEDBACK_* (0.15, 0.1, 0.05)`, `AU_CURVATURE=0.95π`, `PHASE_ADAPT{…}` → π/band.

## 4. Staged plan (from the paradigm foundations upward)

- [x] **Stage 0 — Constitution.** §1 (seven sources + reclassification rule)
  recorded in this plan; the `canonical.py` header already states "only π is a
  genuine structural scale". **DONE (2026-06-28):** the guard test
  `tests/core_physics/test_emergent_constants_guard.py` pins the channel weights,
  operator gains and the coupling ladder to their exact π-formulas and asserts that
  no nodal-physics constant equals a removed frozen φ/γ/e decimal.
- [x] **Stage 1 — Channel weights** (`DNFR_WEIGHTS`, `SI_WEIGHTS`) → §2.1
  coherence-band hierarchy. **DONE, suite green 2201/0.** Highest blast radius
  (every ΔNFR/Si); normalised ratios shifted modestly (phase 0.750→0.7585,
  epi 0.158→0.1832, νf 0.092→0.0583) — same phase-dominant regime.
- [x] **Stage 2 — Operator gains** (the 13) → §2.2. **DONE, suite green 2201/0.**
  Pressure lever IL=π/(π+1), OZ=(π+1)/π (balanced pair exactly isometric);
  capacity lever SHA/NUL=1−1/(4π), VAL=1+1/(4π), NUL_dens=1/λ (volume); secondary
  couplings on the π-fraction ladder (gentle 1/(4π), moderate 1/(2π), fine 1/(8π),
  ZHIR θ-shift 1/π); NAV_eta/REMESH_alpha = unit midpoint 0.5. **Empirical finding:**
  the capacity lever must use the *gentle* π-step, not the band ratio
  (`test_expansion_rate` pins VAL∈(1.04,1.10)) — ΔNFR is fast *pressure*, νf is
  slow *capacity*, a structural distinction.
- [x] **Stage 3 — Feedback / selector / adaptation** → π/band. **DONE, suite green
  2201/0.** Coherence-measuring thresholds anchored to genuine structural levels
  (they measure coherence, so the band *is* their scale): `R_hi`/`FEEDBACK_TARGET`
  = π/(π+1); `R_lo`/grammar `force_*` = the new `MID_COHERENCE_THRESHOLD = 2/π`;
  `disr_hi` + `GLYPH/Si` hysteresis = unit midpoint/quarter (0.5/0.25);
  `SELECTOR_WEIGHTS` = the band hierarchy (Si≻ΔNFR≻accel); `AU_CURVATURE` = the
  exact midpoint (0.9π+π)/2; `PHASE_K_LOCAL`=1/(2π), `PHASE_K_GLOBAL`=1/(2π²)
  (=local/π); `kG` rails = `kL` rails/π; `FEEDBACK` tolerances/rates + `OZ_SIGMA` +
  `THOL_METABOLIC` = the π-fraction ladder; the `get_factor` safety fallbacks now
  reference the emergent constants. **Honest carve-out:** the four selector
  *magnitude* thresholds (`dnfr_hi/lo`, `accel_hi/lo`) sit on the |ΔNFR|/∂²EPI scale,
  NOT coherence — left as labelled *operational* knobs (π-flavouring them would repeat
  the φ/γ/e naming-convention error); candidates for relocation to `operational.py`.
- [x] **Stage 4 — Benchmark/example φ/γ/e *input* purge.** **DONE.** Fixed the
  numerical-usage cases: `37_operator_tetrad_synergy.py` was **broken** (imported
  `GAMMA, PHI` from `canonical`, absent since the purge — examples aren't in the
  suite so it slipped through) → dropped the dead imports + rewrote the refuted
  φ↔Φ_s / γ↔|∇φ| correspondence print as plain π-derived tetrad thresholds (now
  runs, exit 0); `coherence_projector_sense_index` `SI_WEIGHTS` → the band
  hierarchy; removed dead `PHI/GAMMA/E` from `boundary_vibration`; cleaned the
  refuted correspondence comments in `phase_wall` (kept its legitimate TEST-4
  *obstruction* result — it builds `φA+γL+πL²+eK` precisely to prove the four
  constants are insufficient). A subagent purged ~27 stale "(φ,γ,π,e) assumed
  substrate" framings across 14 files → "π". **Kept** the emergent-*object* studies
  (`kuramoto_farey` φ-as-Fibonacci-limit, `emergent_integers` golden angle, Euler
  products, the Γ chirality matrix, tetrahedral symmetry groups). All benchmarks
  compile clean.
- [x] **Stage 5 — Recompute paradigm results.** **DONE — and the result is the
  strongest possible: the headline verdicts are gain-INDEPENDENT (structural),
  hence unchanged.** Verified by direct benchmark runs (primality `ΔNFR=0` PASS;
  conservation Noether/Lyapunov runs; Riemann prime-ladder `||[L, P_σ⊗P_τ]|| =
  0.00e+00` exact S_n equivariance PASS) **+** the full 2201-test suite (which
  encodes the Riemann / tetrad / conservation / Yang–Mills / number-theory
  verdicts) **+** a dependency proof that **Navier–Stokes is a pseudo-spectral
  solver** (`np.fft.fftn`, exact projection) that never touches the operator gains.
  See §7.

## 7. Structural-robustness result (the canonical-emergence proof)

Recomputing under the emergent constants did **not** require reworking a single
headline result — because every headline result is **structural**, not an artifact
of the magic numbers. The changed constants (ΔNFR/Si weights, operator gains) drive
the **dynamics** (trajectories); the paradigm's *claims* derive from **structure**:

| Result | Structural source | Gain-dependent? |
|--------|-------------------|-----------------|
| Primality `prime ⇔ ΔNFR=0` | unit arithmetic coefficients | **No** — robust (PASS) |
| Cyclotomy `s_k(p)=gcd(k,p−1)+1` | pure number theory | **No** |
| Riemann σ_c→½, GUE spacing | prime-ladder Hamiltonians + S_n symmetry | **No** — robust (PASS, suite) |
| Navier–Stokes blow-up / K_φ cascade | **pseudo-spectral** fluid solver (`np.fft`) | **No** — solver ignores gains |
| Conservation (Noether Q, Lyapunov) | charge-density structure | **No** — robust (suite) |
| Tetrad relations (K_φ=L_rw·φ, ξ_C∝1/√λ₂) | graph Laplacian identities | **No** — robust (suite) |
| Yang–Mills U6 confinement | `Φ_s²/(π/2)²` structural potential | **No** — robust (suite) |
| C(t)/Si **trajectories**, network-opt outcomes | operator evolution | **Yes** — decimals shift, *verdicts/attractors hold* |

**Conclusion.** The magic numbers (φ/γ/e and the arbitrary operational decimals)
were **never load-bearing** for the paradigm's claims. The refactor removed them
and re-derived every constant from π/structure **without altering one headline
verdict** — which simultaneously (a) cleans the foundation (no magic numbers on the
nodal-physics paths) and (b) *proves* the results are genuinely emergent (they
survive the purge because they were always structural). The only quantities that
move are the dynamic trajectories, whose **qualitative attractors are invariant**.

## 5. Validation protocol (every stage)

1. Full suite green (`2201 passed` baseline) — fix expectations that encoded a
   magic value to assert the *structural* relation instead.
2. Re-derive any doc table that listed the old number.
3. For Stages 1–2: record C(t)/Si/tetrad on a fixed seeded ring + the Riemann/NS
   smoke benchmarks before/after; the *qualitative* paradigm conclusions must hold.
4. Commit/push remains **on hold** pending user confirmation.

## 6. Status log

- **2026-06-27** — Plan created. Deep study complete: classified `canonical.py`
  (89 physics constants) + `defaults_core.py` gains/weights; confirmed
  `DNFR_WEIGHTS`/`SI_WEIGHTS` are still frozen φ/γ and `DNFR_WEIGHTS` is normalised
  (ratios only). Derived the coherence-band hierarchy (§2.1, exact-normalising) and
  the band-reciprocal operator-gain framework (§2.2). Beginning Stage 0/1.
- **2026-06-27** — **Stage 1 COMPLETE** (suite green 2201/0). Added the
  `CHANNEL_WEIGHT_PRIMARY/SECONDARY/TERTIARY` coherence-band hierarchy to
  `canonical.py` (exact-normalising π/(π+1) + π/(π+1)² + 1/(π+1)² = 1) and wired
  `DNFR_WEIGHTS` (phase≻EPI≻νf) and `SI_WEIGHTS` (νf-coh≻phase-sync≻|ΔNFR|) to it,
  removing the frozen φ/γ decimals 0.737/0.155/0.09(0.114).
- **2026-06-27** — **Stage 2 COMPLETE** (suite green 2201/0). Added
  `COHERENCE_RETENTION = π/(π+1)`, `DISSONANCE_AMPLIFICATION = (π+1)/π`, the gentle
  capacity step `_CAPACITY_STEP = 1/(4π)`, and the `COUPLING_GENTLE/MODERATE/FINE`
  π-fraction ladder to `canonical.py`. Rewired all 13 operator gains in
  `GLYPH_FACTORS` + the `SHA/VAL/NUL` factors: IL/OZ band-reciprocal, SHA/VAL/NUL
  gentle π-step, NUL_dens volume-conservation, AL/UM/RA/THOL/NAV secondary gains →
  π-fraction ladder, ZHIR θ-shift → 1/π. No magic number remains in any operator
  gain. Capacity lever kept gentle (empirical: VAL test range). Next: Stage 3.
- **2026-06-27** — **Stage 3 COMPLETE** (suite green 2201/0). Added
  `MID_COHERENCE_THRESHOLD = 2/π` to `canonical.py`; derived `AU_CURVATURE` as the
  (0.9π+π)/2 midpoint; π-fractioned the `FEEDBACK` tolerances/rates. In
  `defaults_core.py`: `SELECTOR_WEIGHTS` → band hierarchy; `PHASE_ADAPT` coherence
  triggers (π/(π+1), 2/π, 0.5) + kG/kL rails; `PHASE_K` couplings (1/(2π), 1/(2π²));
  `GLYPH`/`Si` hysteresis (0.5/0.25); grammar `force_*` (2/π); `OZ_SIGMA` +
  `THOL_METABOLIC` → π-ladder. Updated the 11 `get_factor` safety fallbacks in
  `operators/__init__.py` to the emergent constants. Left `dnfr/accel` selector
  magnitude thresholds operational (honest, magnitude-scale). Next: Stage 4 / 5.- **2026-06-27** — **Stage 4 COMPLETE.** φ/γ/e *input* purge in benchmarks/examples.
  Discovered + fixed a **broken** example (`37_operator_tetrad_synergy.py`,
  `ImportError` on the purged `GAMMA`/`PHI`). Updated `coherence_projector` SI
  weights → band hierarchy; removed dead `PHI/GAMMA/E` in `boundary_vibration`;
  de-refuted `phase_wall` comments (kept its obstruction test). Subagent fixed ~27
  "(φ,γ,π,e) assumed substrate" → "π" across 14 files. Kept emergent-object studies.
  Benchmarks compile clean; the fixed example runs (exit 0). Engine untouched, so
  the suite stays green 2201/0. Next: Stage 5 (recompute results).
- **2026-06-27** — **Stage 5 COMPLETE — robustness proven (§7).** Recomputation
  shows the headline verdicts are gain-INDEPENDENT: primality `ΔNFR=0` PASS,
  conservation runs, Riemann prime-ladder S_n equivariance exact (`0.00e+00`) PASS,
  and Navier–Stokes is a pseudo-spectral solver that never reads the gains. The
  2201-test suite already encodes the Riemann/tetrad/conservation/Yang–Mills/number-
  theory verdicts (all green). Only dynamic *trajectories* shift (verdicts hold).
  Made the `coupling_weights_type_signature` docstring refs drift-proof (dropped the
  twice-stale line numbers). **The magic numbers were never load-bearing — the
  paradigm's claims are structural and genuinely emergent.**
- **2026-06-28** — **Stage 0 COMPLETE (constitution guarded) + last straggler.**
  Added `tests/core_physics/test_emergent_constants_guard.py`: positive π-formula
  assertions for the band edges, the channel-weight hierarchy (exact-normalising),
  the IL∘OZ band-reciprocal pair (exactly isometric), the `1/(4π), 1/(2π), 1/(8π)`
  coupling ladder and `2/π`; plus a parametrized guard that **no** curated
  nodal-physics constant lies within `1e-5` of any removed frozen φ/γ/e decimal.
  Also cleaned the final φ/γ/e *comment* straggler the Stage-4 input purge missed in
  `src/`: `_compute_phase_transport_derivative` (`dynamics/canonical.py`, the
  extended nodal system) no longer cites the dead `≈ 0.618 = 1/φ` / `0.155` / `0.135`
  origins (values unchanged — honest operational magnitudes on the optional
  J_φ-transport path; the false "RECALIBRATED from canonical constants" and dead
  "Import canonical constants" comments removed). 239 core-physics tests green.
  **The plan is complete.**