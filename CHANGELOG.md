# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.0.3.6] - 2026-09-19 — Nodal research integration and documentation consolidation

### Nodal dynamics and research

- Integrated shared phase/form observations, geometry realization and joint
  quotient contracts with explicit hypotheses and representation boundaries.
- Added finite-state, phase-origin, reflection, hidden-form, support and memory
  regression cases; retained counterexamples and unavailable-field evidence.
- Unified signed scalar EPI admission and clarified continuous integration,
  prescribed forcing, operator jumps and observed temporal evidence.
- Included the post-0.0.3.5 diffusion, finite REMESH/Reception, THOL and grammar
  work already integrated on main. Its exact and finite-runtime claims retain
  their specified domains; the current execution plan owns their detailed status.

### Documentation and diagnostics

- Consolidated theory, architecture, developer instructions and the research
  queue; archived long historical notebooks and superseded proposals unchanged.
- Corrected unsupported conservation, complete-state, autonomous-emergence and
  arithmetic claims in guides, source explanations and release metadata.
- Removed the educational manuscript and its site navigation; maintained
  technical guides and examples own the current explanations.
- Generate the operator contract table from the registry and reject metadata
  drift. Validate local/current-main references, directory fragments and
  reference-style links; retain checks under optimized Python.
- Build one MkDocs site with working mathematical typesetting, diagrams,
  current guide navigation and historical downloads.

### Maintenance and compatibility

- Added a shared SDK study API and the `tnfr network`, `operators` and
  `sequences` commands, also available through `python -m tnfr`. Validated JSON
  recipes use existing topology builders and operator words; reports retain
  input state, versions, diagnostic availability and estimator provenance.
- Added a runnable study example and one CLI/SDK guide. Recipe cycles count
  operator words, not physical time; recipes are not complete engine checkpoints.
- Keep CLI logs on stderr without replacing application logging handlers.
  Structured stdout is portable to Windows code pages, and report files use
  the shared atomic JSON writer. Reject conflicting inputs and invalid step
  counts before execution. Remove unavailable legacy profiling commands.
- Removed dead workflow/profiler entry points and two obsolete benchmarks;
  current owners and original/archive hashes are recorded in the retirement manifests.
- Reduced documentation dependencies to the active MkDocs toolchain and release
  tooling to build/Twine. Removed unused, contradictory automatic-version settings.
- Retained the existing GitHub-release-triggered PyPI publication route.
  Publication metadata and the release tag must match the package version.
  Release preparation verifies the selected commit; the workflow itself does
  not provide a numerical stability certificate or upload to Zenodo.
- Scalar-only operations explicitly reject complex/nonuniform EPI payloads.
  Diagnostic thresholds, fitted coherence length and auxiliary models retain
  their documented scope; review the current API contracts when upgrading.
- Repaired advanced primality CLI validation-schema handling and separated
  human-readable progress from JSON output. These are interface corrections,
  not changes to the primality calculations.
- Fixed a missing plotting-padding reference, preserved compatibility reexports
  explicitly and applied the configured Python formatting tools.
- Declare SciPy as a core dependency for the SDK's spectral import path.
  Missing optional system-memory telemetry no longer prevents CPU computation
  interfaces from loading.
- Keep spectral-cache writes from signature readouts off the caller's graph,
  including snapshot-only calls and graphs with an existing cache.
- Handle the empty disagreement sector explicitly in matrix-exponential and
  transient-gain calculations. Accept UTC `Z` timestamps consistently on Python
  3.10 while preserving missing, nonmonotone and mixed-timezone evidence. Compare
  FFT phase preservation by circular separation at the existing tolerance.
- Make CLI byte-lineage tests independent of ignored historical artifacts,
  remove a pytest-benchmark fixture-name collision and compare spectral
  estimates against an analytic reference. CI retains per-Python JUnit reports.
- Classified the expensive C6 historical report producers as slow tests;
  their lightweight input-validation and independent exact-model checks remain
  in the default selection. No producer result is inferred from that classification.
- Retired self-contained toy tests, copied verifier criteria, duplicate test
  runners and obsolete curvature/timing campaigns. Retained their useful
  contracts in tests that call the actual engine and preserved retirement reasons.
- Replaced introductory toy dynamics with the shared SDK entry point. Removed
  misleading standalone demonstrations and kept explicit model boundaries for
  retained research examples.
- Centralized example loading and CLI checks; retained independent numerical
  assertions while reducing duplicated cold-import processes. Parallel CI keeps
  each test file together so module fixtures run once per worker.
- Reuse the temporal evaluation's existing window count instead of recomputing
  its complete field series. Test report wiring separately from numerical work.
- Share diagnostic graph preparation while retaining fresh mutable graphs.
  Remove ineffective test resets for absent subsystems and retain the actual
  cache, callback, RNG and transaction isolation controls.
- Compose testing/development extras from their existing dependency groups.
  `make test` follows the configured pytest suite; research producers have
  separate explicit commands. Benchmark wrapper checks no longer accept arbitrary
  exceptions as successful tests.

### Evidence and open scope

Release-specific validation belongs to the GitHub release notes and attached
receipts. The scientific objective of deriving autonomous persistent patterns
and connecting them to independently specified terrestrial measurements remains
open. This release does not establish general C6 stability, a complete state
basis, a Millennium conjecture or physical particle emergence.

The detailed pre-release development log is preserved in the
[historical snapshot](theory/research/archive/DEVELOPMENT_LOG_PRE_0.0.3.6_2026-09-19.txt).
Descriptions in older release entries below record their original context;
current mathematical authority belongs to the [theory index](theory/README.md).

## [0.0.3.5] - 2026-06-24 — Tetrad correspondence audit & emergent redesign

A computational audit of the "Universal Tetrahedral Correspondence" found that
only **π** is a genuine structural scale; the four-constant correspondence
(φ↔Φ_s, γ↔|∇φ|, e↔ξ_C) is mostly an **organizing overlay**. Several thresholds
asserted as "derived" were empirical, inert (magic), or measured false. This
work corrects the claims and replaces magic thresholds with emergent,
system-measured quantities. The nodal equation, the 13 operators, and grammar
U1–U6 are unchanged.

### Corrected (canonicity claims)

- **Only π is a genuine structural scale** — the phase-wrap bound shared by BOTH
  `|∇φ|` and `K_φ` (both are means of wrapped angles, ≤ π). γ, e, φ are
  recoverable as mathematical identities but are NOT the structural scales of
  their tetrad fields. `K_φ` matches `L_rw·φ` only on a smooth, consistent
  unwrapped branch with matching normalization; the reported correlation is a
  finite-protocol measurement. The exact wrapped computation is nonlinear;
  fitted `ξ_C` uses `1/√λ₂` as a spectral comparison/fallback (not base e).
- **`|∇φ|` bound corrected** from `γ/π ≈ 0.1837` to the phase-wrap bound `0.9π`
  in `physics/variational.py`, symmetric with `K_φ`. The measured synchronization
  onset is ≈ 0.29 and σ-dependent, NOT the constant γ/π; γ/π is retained
  elsewhere only as a heuristic early-warning level, explicitly labelled
  non-derived.
- **`derive_tetrad_threshold_values`** rows re-statused: π `geometric`; φ, γ, e
  `overlay` (recoverable identities, not structural scales).
- **`ARCHITECTURE.md`, `.zenodo.json`, `CONTRIBUTING.md`, `theory/README.md`,
  `docs/STRUCTURAL_FIELDS_TETRAD.md`, `constants/canonical.py`** — removed the
  "Universal Tetrahedral Correspondence foundation / 100% derived / zero
  empirical tuning / verified to machine precision" claims; replaced with the
  honest tiering (π genuine; γ/e/φ notational overlay).
- **Second audit pass (repo-wide)** — removed the remaining "Universal
  Tetrahedral Correspondence / canonical derivation / Kuramoto critical
  coupling" claims and internal contradictions across `ARCHITECTURE.md`
  (the "Mathematical Purity / 497 magic numbers eliminated / zero empirical"
  sections), `README.md`, `theory/FUNDAMENTAL_THEORY.md`, `theory/GLOSSARY.md`,
  `theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md`, `AGENTS.md` (+ mirror),
  `docs/grammar/PHYSICS_VERIFICATION.md`, and the central constants modules
  (`telemetry/constants.py`, `mathematics/unified_numerical.py`,
  `config/defaults_core.py`, `physics/signatures.py`,
  `operators/grammar_telemetry.py`, `physics/emergent_chemistry.py`) plus four
  benchmarks. Exposed cosmetic "derivations" (e.g. `MIN_BUSINESS_SENSE_INDEX =
  1/φ + 0.082 ≈ 0.700`, `⌊φ×10⌋ = 16`) as calibrated/notational values.

### Changed (emergent replacement of magic thresholds)

- **`physics/phase_transition.py` redesigned** around a standardized signed-mean
  ratio. The "universal critical exponent γ_c = γ/π" was **measured false**
  (the fitted exponent is protocol-dependent), and the classification noise
  floor `(γ/π)²` was **proven inert** (it sat in a two-order-of-magnitude gap;
  sweeping it changed no classification). Removed the magic constants `GAMMA_C`,
  `ORDER_PARAMETER_NOISE_FLOOR`, `CHIRALITY_THRESHOLD` and the
  `theoretical_exponent` field; added `symmetry_zscore(mean, var, n) =
  |mean|/√(Var/N)` and the selected cut `Z_SIGNIFICANCE = 1`. The later S6
  audit recorded above corrects its interpretation: coupled graph nodes do not
  justify an independent-sample significance test, so the ratio is an
  operational spatial-imbalance diagnostic.
- **Named γ/π constants relabelled** as heuristic / non-derived in
  `constants/canonical.py` (`CRITICAL_EXPONENT`, `GRAD_PHI_CANONICAL_THRESHOLD`,
  `PHASE_GRADIENT_THRESHOLD_CANONICAL`) and `mathematics/unified_numerical.py`;
  `gauge.py`, `emergent_chemistry.py`, `interactions.py` regime thresholds
  marked calibrated/heuristic, not derived.

### Notes

- **Third audit pass (repo-wide, exhaustive)** — removed the remaining
  "Universal Tetrahedral Correspondence / canonical derivation / zero empirical
  fitting" claims across the two subprojects (`primality-test/`,
  `factorization-lab/`), examples, benchmarks, the `mathematical_purity` tests
  (now check genuine bounds, not the refuted mapping), all theory docs
  (`FUNDAMENTAL_THEORY.md §4` and `GLOSSARY` renamed to "structural-field
  tetrad"), and ~40 in-code combo comments (`defaults_core.py`, `bifurcation.py`,
  `cycle_detection.py`, `number_theory.py`, etc.) now marked notational. Exposed
  the primality coefficients (ζ=φγ, η=(γ/φ)π, θ=1/φ) as combos chosen to
  approximate the original empirical values (ζ=1.0, η=0.8, θ=0.6).
- The gauge force-regime classification and emergent-chemistry excitation scale
  were flagged here for full emergent rework; that rework is now complete — see
  "Emergent redesign" below.
- Full test suite green (2196 passed) after all three passes.

### Consequences audit (computational impact, 2026-06-21)

A measure-first review of whether *calculations* (not just narrative) depended
on the refuted tetrad values. Main finding: **the significant results are
robust**, because they emerge from STRUCTURE (integer orderings, exact zeros,
relative scores) rather than the scale values (γ/π, etc.):

- **Emergent chemistry** (periodic table, magic numbers 2/10/18/36/54/86,
  octet): the aufbau filling order uses only the integers (n+l, n); the octet
  is the exact ΔNFR=0 zero — both independent of any scale coefficient. The
  `nu_excitation` (=γ/π), `nu_0`, `coherence_gap` fields were DEAD (defined,
  never consumed) and were removed; only `theta_valence` enters, as a positive
  scale (the zero is robust to its value).
- **Number-theory primality** (n prime ⟺ ΔNFR=0): each pressure term vanishes
  individually for primes (Ω−1, τ−2, σ/n−(1+1/n) are all 0), so the result is
  independent of the coefficients ζ, η, θ.
- **Gauge interaction regimes**: `dominant_regime` is decided by relative
  scores, NOT the γ/π threshold; `above_threshold` (which used γ/π) is metadata.
- **Riemann ζ-bridge buffer** γ/π: a regularisation shift whose exact value is
  immaterial. **K_φ asymptotic exponent** α≈2.76: a measured fit, unrelated.

Real consequences corrected:
- `PHASE_CURVATURE_ABS_THRESHOLD = φ×π ≈ 5.083` was a **non-physical** K_φ bound
  (|K_φ| ≤ π by phase wrap, so any check using it was a no-op); it was dead code,
  corrected to `0.9π ≈ 2.827`.
- `STRUCTURAL_STABILITY_AND_DYNAMICS.md §2.2` still described the old γ_c
  classification table; updated to the emergent z-score rule.
- Removed the dead chemistry scale parameters; fixed a residual
  `MATHEMATICAL_DYNAMICS_BASIS.md` |∇φ| claim. Full suite green (2196 passed).

### Emergent redesign (gauge regimes + chemistry excitation, 2026-06-21)

Completed the full emergent rework of the two studies whose conceptual base was
the (now-refuted) four-constant overlay. Both were rebuilt to rest on STRUCTURE
alone (measure-first; no value replaced by another magic value):

- **Gauge interaction-regime classification** (`physics/gauge.py`): removed the
  three overlay threshold constants `REGIME_DOMINANCE_THRESHOLD` (1/φ),
  `REGIME_STRONG_THRESHOLD` (γ/π, "Kuramoto critical coupling in gauge") and the
  unused `REGIME_SECONDARY_THRESHOLD` (γ/(π+γ)). The per-sector `above_threshold`
  activity flags now use a single parameter-free criterion: a sector is *active*
  when its normalised score exceeds the equipartition share `1/N_REGIMES = 0.25`
  (the maximum-entropy reference, derived from the number of gauge sectors — the
  four structural channels of the tetrad). Uniform across all four sectors; no
  overlay constant. New public symbols `N_REGIMES`, `REGIME_ACTIVITY_SHARE`
  replace the removed thresholds. `dominant_regime` (relative `max` of scores)
  is unchanged — it was already robust. Measured: the criterion always flags the
  dominant sector and additionally marks genuine co-active secondaries.
- **Emergent-chemistry valence scale** (`physics/emergent_chemistry.py`): removed
  the last free scale parameter (`theta_valence = 1/φ`) and the now-trivial
  `EmergentChemistryParameters` dataclass. `ΔNFR_chem(Z)` is now the **integer**
  structural distance of the outer shell to a closed configuration, in natural
  units (one subshell step = 1) — the exact chemical analogue of primality
  `ΔNFR(n)=0`. Noble gases (2,10,18,36,54,86) → ΔNFR=0; halogens/alkali → 1;
  oxygen → 2; carbon → 4. Magic numbers and the octet are unchanged (they always
  emerged from the integer (n+l) ordering and the exact zero).
- **Measured (chemistry):** tested whether the (n+l) filling order could emerge
  from the raw Laplacian spectrum of a concentric multi-shell ("onion") manifold.
  It does NOT — Madelung ordering reflects electron-electron screening absent
  from a free graph Laplacian. (n+l) is therefore documented honestly as an
  integer excitation-count rule (total radial+angular quanta), not a spectral
  derivation and not a constant correspondence.
- Tests updated (`test_gauge.py`: `TestRegimeActivityCriterion`, equipartition
  consistency). Full suite green (2195 passed, 2 skipped).

## [0.0.3.4] - 2026-06-17

This release consolidates the emergent-geometry program, centralizes the
operator/grammar/contract layer onto single canonical sources, opens three new
TNFR-native Millennium-problem programs, and refactors the documentation to the
current engine state. The 13-operator catalog, grammar U1–U6, and the nodal
equation are unchanged; everything below either *measures* structure the nodal
equation already contains or removes duplication. Full suite: 2043 passed, 2
skipped.

### Symplectic Substrate (historical entry; current scope corrected)

This release introduced an auxiliary ambient **symplectic phase space**
initialized from graph-field snapshots. Later audits established that its exact
harmonic-flow identities do not derive the full nodal equation or certify the
engine operators.

- **New module**: `src/tnfr/physics/symplectic_substrate.py` — phase space
  `P = ℝ^{4N}` with conjugate pairs `(K_φ, J_φ)` (geometric) and `(Φ_s, J_ΔNFR)`
  (potential); symplectic 2-form `ω` (antisymmetric, non-degenerate, closed);
  canonical Poisson brackets; `H_sub = ½Σ(K_φ²+J_φ²+Φ_s²+J_ΔNFR²)` equal to the
  structural-energy snapshot exactly after adding the held-fixed gradient
  background; Liouville `div(X_H)=0` for the specified harmonic flow. Each
  engine operator would require a separate Jacobian pullback test.
- **Auxiliary model structure tower** (each measured to machine precision):
  Noether charges (time-translation → `H_sub`; geometric U(1) → `E_geo = ½Σ|Ψ|²`;
  potential U(1) → `E_pot`); the compatible Hermitian / flat-Kähler triple
  `(ω, J, g)` with `J = −ω` — so the `i` in `Ψ = K_φ + i·J_φ` *is* the complex
  structure the substrate induces; complete integrability (action–angle,
  Liouville–Arnold); Poincaré–Cartan integral invariants; Marsden–Weinstein
  symplectic reduction; and the hidden **U(2) polarization symmetry** whose
  SU(2) part supplies three conserved **Stokes parameters** on the per-node
  Poincaré sphere (classical wave polarization — Stokes 1852 / Poincaré 1892 —
  not isospin or qubits).
- **Historical threshold overlay**: `derive_tetrad_threshold_values` reconstructs
  φ, γ, and e from chosen mathematical identities. They are organizational
  overlays rather than derived TNFR field scales; π remains the phase primitive.
- **Consolidated entry point**: `verify_substrate_geometry(G)` bundles all
  certificates into a `SubstrateGeometryReport`.
- **SDK**: `Network.symplectic_substrate()` + `SymplecticReport`, in
  `TNFR.analyze()`.
- **Honest scope**: a flat, constant-coefficient linear Kähler backbone — a
  consolidation of geometry already implied by `conservation.py` +
  `variational.py`; it does not resolve any open program.
- **Demonstrations**: `examples/08_emergent_geometry/98`, `106`, `114`.

### Emergent Geometry — Structural Diffusion (transport layer)

The EPI channel of the canonical ΔNFR is the random-walk graph Laplacian
`−L_rw·EPI` (verified to residual ~1e-16), so the nodal equation is literally a
discrete diffusion equation with diffusivity `νf`. From this single identity the
engine measures, in TNFR's own variables, a tower of empirically-established
transport phenomena.

- **New module**: `src/tnfr/physics/structural_diffusion.py` — six transport
  layers: diffusion/synchronization (Fourier/Fick/Kuramoto), overdamped drift
  (`q̇ = νf·F`, Stokes/Einstein mobility — corrects the prior "Newton's second
  law" reading: the bare first-order nodal equation is overdamped, νf is mobility
  not inverse mass), discrete standing-wave modes (bounded-manifold Laplacian
  eigenmodes), structural-stability dispersion relation (`σ_k = r − νf·λ_k`, the
  spectral form of U2), random walk + effective resistance (Ohm/Kirchhoff), and
  structural flow (current, Kirchhoff continuity, Ohm).
- **Overdamped-projection bridge**: the nodal equation is the strong-damping
  limit of the substrate wave `q̈ + γq̇ + Lq = 0` with `νf = 1/γ`; the γ-dial
  spans diffusion (γ→∞) to standing waves (γ→0).
- **Honest scope**: the EPI-channel ↔ Laplacian identity is exact; the full ΔNFR
  is multi-channel; `λ_2` is purely topological and does not encode any canonical
  constant (measured negative result).
- **Demonstrations**: `examples/08_emergent_geometry/99`, `113`, `134`, `135`.

### Operator Contracts & Energy — Centralization and Emergence

- **Canonical contract layer**: new `src/tnfr/operators/operator_contracts.py` —
  the single source of truth for what each operator does to node state, anchored
  to the direct `_op_*` effect (TNFR.pdf §2.2.1). Each `OperatorContract` records
  the public English name, the `primary_channel` (one nodal-equation channel:
  EPI / νf / θ / ΔNFR), the `scale` (NODE for twelve operators, NETWORK for the
  U5 operator REMESH), and a verifiable postcondition. The proactive audit
  (`audit_operator_contracts`), the reactive integrity monitor (`POSTCONDITIONS`),
  and the introspection metadata now all derive from this spec — eliminating the
  historical drift where scattered copies disagreed (e.g. AL claiming "positive
  ΔNFR" though `_op_AL` only raises EPI; RA checked for EPI increase though it
  preserves identity; VAL/NUL checked |EPI| though they scale νf).
- **Public English names**: the structural-operator name (Emission, Reception, …)
  is canonical at the public level; the glyph code (AL, EN, …) is the internal
  symbol.
- **Energy/coherence are emergent**: the structural candidate energy
  `E = ½Σ(Φ_s²+|∇φ|²+K_φ²+J_φ²+J_ΔNFR²)` contains no EPI or νf term (measured:
  scaling EPI or νf leaves E unchanged). The per-operator Lyapunov role in
  `physics/lyapunov.py` is therefore re-derived from the canonical grammar U2 role
  (`config.physics_derivation`), not from a hardcoded energy algebra:
  stabilisers {IL, THOL}, destabilisers {OZ, ZHIR, VAL}, the rest neutral. The
  form-channel operators (AL, EN, RA, REMESH) leave a same-snapshot evaluation
  unchanged when all derived fields are held fixed because EPI is absent from
  the formula. This is a dependency statement, not an isometry or a conservation
  law after pressure and fields are recomputed.
- **Dual-lever clarified**: the two levers are the two right-hand-side factors of
  the nodal equation — νf (capacity) and ΔNFR (pressure); operators that write
  the form EPI (the LHS) sit on neither lever.
- **Demonstrations**: `examples/08_emergent_geometry/152`,
  `examples/02_physics_regimes/115`.

### Grammar — Single Canonical Source & Formal-Language Characterization

- **Centralization**: the operator-classification sets (generators, closures,
  stabilizers, destabilizers, transformers, bifurcation triggers/handlers) are
  derived once in `config.physics_derivation` and re-exported by
  `operators/grammar_types.py`. Every grammar consumer — the U1–U6 validator,
  the secondary sequence validator, grammar_dynamics, the runtime preconditions,
  the error factory, and the operator metadata — now reads the single source.
  Parallel hardcoded copies (including a secondary validator that wrongly listed
  NUL as a U2 destabilizer) were removed and pinned by
  `tests/operators/test_grammar_canonical_consistency.py`.
- **Canonical grammar spec**: new `operators/grammar_canon.py` materializes the
  U1–U6 role table, the five-type structural typology, and the canonical glyphic
  macros (anchored to TNFR.pdf §2.3), with a self-consistency check.
- **Formal-language thread** (characterization, demos only): the flat,
  non-nested default-depth history projection is represented by 320 reachable
  states and a 52-state minimal complete DFA. Its numerical growth radius is
  `10.9560791442`; both U2 and U4b reduce the radius. The 944-element syntactic
  monoid is aperiodic (index 4), so this constructed flat language is star-free
  / first-order definable. Nested `THOL[...]` syntax requires a context-free,
  stack-like model, while runtime U3 and reference-dependent U6 remain outside
  the DFA. The Parry process is a selected maximum-entropy Markov policy on the
  dominant component, not a TNFR physical equilibrium or grammar validator.
- **Demonstrations**: `examples/08_emergent_geometry/139`–`152`.

### Number Theory & the Dual-Lever

- Prime families as orbits on the zero-pressure set `{ΔNFR = 0}`; numbers as a
  coupled network (Ω-graded centrality, primes as the transport periphery); the
  nodal flow on numbers (primes as equilibria, not attractors); primality as
  grammatical inertness; numbers as free-monoid words with the dual-lever as the
  two additive gradings (count Ω → ΔNFR pressure, size log → νf capacity); the
  capacity arm carries von Mangoldt and the prime-ladder Hamiltonian P14 is the
  capacity-arm operator — locating the Riemann oscillatory obstruction on the
  capacity axis the per-node substrate is blind to.
- **Honest scope**: these restate classical multiplicative number theory through
  the grammar/dual-lever lens; they close no open problem.
- **Demonstrations**: `examples/07_number_theory/94`–`97`, `100`–`102`,
  `116`, `146`–`149`.

### Millennium Problem Programs (TNFR-native reformulations)

Three new programs join Riemann / Navier–Stokes / Yang–Mills. **None claims a
solution** — each carries an explicit honest-scope statement and classified
obstruction.

- **P vs NP (PNP-1)** — the nodal equation is a gradient flow, so verifying a
  configuration's coherence is `O(|E|)` but synthesizing a globally coherent one
  by relaxation traps in dissonance basins (measured global-optimum hit rate
  drops monotonically with problem size on frustrated MAX-CUT). Mirrors P≠NP;
  Branch B open. `theory/TNFR_P_VS_NP_RESEARCH_NOTES.md`,
  `examples/09_millennium/109`.
- **Birch–Swinnerton-Dyer (BSD-1)** — `a_p = p+1−#E(F_p)` as structural pressure;
  the accumulated product reproduces the original 1965 empirical rank separation
  by brute-force point counting. GL(1)→GL(2) gap open; Branch B.
  `theory/TNFR_BSD_RESEARCH_NOTES.md`, `examples/09_millennium/110`.
- **Hodge (HC-1, scope corrected under Unreleased)** — an auxiliary simplicial
  complex reproduces the standard finite Hodge decomposition and expected Betti
  numbers. Canonical `|∇φ|` and `K_φ` are node summaries, not oriented edge and
  face cochains, so no tetrad cochain tower has been derived. The `(p,p)` and
  algebraicity gap remains outside this baseline. `theory/TNFR_HODGE_RESEARCH_NOTES.md`,
  `examples/09_millennium/111`.

### Documentation, Examples & Repository Hygiene

- **README + core theory docs** refactored to the current engine state: corrected
  a real API note (operators are callable, there is no `.apply()`), added the
  emergent-geometry section to `theory/FUNDAMENTAL_THEORY.md`, rewrote the energy
  classification in `theory/STRUCTURAL_OPERATORS.md` /
  `theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md` to the emergent/grammar-U2 frame,
  and updated counts.
- **Examples reorganized** into 10 thematic subfolders (`01_foundations` …
  `10_applications`), resolving the prior 77–86 numbering collision; each file
  keeps a stable global number. Foundational examples refactored to the canonical
  Kuramoto phase-synchrony physics.
- **Documentation-integrity pass**: repaired all dangling example/source/`.md`
  links repo-wide, pruned 9 obsolete `docs/` files, rebuilt the `theory/` hub,
  and resynced the derived `.github/agents/my-agent.md` mirror.
- **Deep repo cleanup**: removed foreign GraphQL scratch JSONs, a CUDA debug
  script, a backup test, an empty dead CLI module, and a stale task tracker; fixed
  a dangling `tnfr-validate` console-script entry point in `pyproject.toml`.
- **Registry consolidation + lint**: the SDK fluent glyph→operator map and the
  lyapunov operator table now derive from the canonical registries; cleared a
  small set of dead-code lint findings.
- **SDK fixes**: repaired silent no-ops in `auto_optimize` and
  `evolve_grammar_aware`; added a proactive measured operator-contract fidelity
  audit (`net.audit_operators()`).

### Research Program Milestones (Yang–Mills Y1–Y5, REMESH-∞ N15, Navier–Stokes N16–N17)

The Yang–Mills (Y1–Y5) and REMESH-∞ / Navier–Stokes (N15–N17) program
milestones below were developed earlier in the cycle and are part of this
release. Each carries an explicit honest-scope statement; none resolves a Clay
Millennium Problem.

#### Y5 — TNFR–Yang–Mills Closure / Obstruction Classification

- **Verdict**: `BRANCH_B_OBSTRUCTION_CLASSIFIED` — Y1–Y4 establish a finite TNFR `U(1)` structural gauge diagnostic surface, but Clay-strength closure requires a new canonical non-Abelian derivation plus a continuum / thermodynamic lower-bound theorem.
- **New API**: `classify_yang_mills_closure()` in `src/tnfr/yang_mills/closure.py`, exported from `tnfr.yang_mills` with `YangMillsClosureReport`.
- **Finite TNFR branch**: `A_FINITE_U1_DIAGNOSTIC_SURFACE` when Y4 reports stable finite positive gaps.
- **Clay-strength branch**: `B_REQUIRES_NEW_CANONICAL_NONABELIAN_DERIVATION` because Y3 remains `OPEN_DERIVABILITY_GAP`.
- **Scope discipline**: `clay_problem_resolved = False`; the obstruction is localized, not removed.
- **Validation**: 4 new tests in `tests/physics/test_yang_mills_closure.py` cover Branch-B classification, report reuse, sampled collapse handling, and package-root import. Y1–Y5 focused run: `33 passed`.
- **Next target**: Y6 / Branch-B derivation search for a TNFR-native non-Abelian connection and non-commuting generator algebra. If no derivation exists without external group labels, the programme should pause at Branch B.

#### Y4 — TNFR–Yang–Mills Finite Scaling Diagnostic

- **Verdict surface**: `FINITE_SCALING_EVIDENCE` or `GAP_COLLAPSE_OBSERVED` depending on sampled finite graph families. This is a finite diagnostic only, not a continuum theorem.
- **New API**: `run_finite_scaling_study()` in `src/tnfr/yang_mills/scaling.py`, exported from `tnfr.yang_mills` with `FiniteScalingPoint` and `FiniteScalingReport`.
- **Scaling coordinate (historical name corrected under Unreleased)**: graph node count `n` under fixed single-snapshot magnitude ratios `ρ_U6 = max_i |Φ_s(i)|/(π/2)`; this does not assess U6 drift. Grouped reports fit finite log-log slopes of mean gap versus `n`.
- **Scope discipline**: Y4 runs while YMG-4 remains open. Therefore finite positive scaling evidence cannot be promoted to a Clay-strength Yang–Mills mass-gap claim.
- **Validation**: 6 new tests in `tests/physics/test_yang_mills_scaling.py` cover report shape/scope, grouped finite scaling, reproducibility, sampled collapse classification, invalid input rejection, and package-root import. Y1–Y4 focused run: `29 passed`.
- **Next target**: Y5 closure / obstruction classification, likely Branch B unless a later TNFR-native non-Abelian connection and generator algebra are derived.

#### Y3 — TNFR–Yang–Mills Non-Abelian Derivability Audit

- **Verdict**: `OPEN_DERIVABILITY_GAP` — audited candidate routes for deriving a non-Abelian / multi-channel gauge sector from TNFR-internal data only; no route is promoted to canonical status.
- **New API**: `audit_nonabelian_derivability()` in `src/tnfr/yang_mills/derivability.py`, exported from `tnfr.yang_mills` with `NonAbelianCandidateAudit` and `NonAbelianDerivabilityReport`.
- **Routes audited**: U5 nested-EPI multiplets, THOL/REMESH operator-history internal spaces, and graph cycle-basis bundles.
- **Obstruction**: current canonical `Ψ = K_φ + i·J_φ` gauge structure supplies a scalar local `U(1)` connection. Nested EPI or operator-history data do not yet derive component-mixing parallel transport or non-commuting generator algebra; cycle-basis routes require non-canonical basis/orientation selection.
- **Validation**: 5 new tests in `tests/physics/test_yang_mills_derivability.py` cover baseline `U(1)` confirmation, nested-EPI obstruction, cycle-bundle rejection, unsupported route errors, and package-root import. Y1+Y2+Y3 focused run: `23 passed`.
- **Open boundary**: YMG-4 remains open. Y4 scaling can proceed only as a conditional finite diagnostic; it cannot become a Clay-strength claim while non-Abelian derivability is unresolved.

#### Y2 — TNFR–Yang–Mills U6 Confinement Sweep

- **Verdict**: `EMPIRICAL_FINITE_GRAPH_ONLY` — finite sweep surface created for testing how the Y1 structural gauge gap behaves across two groups of the single-snapshot potential-magnitude coordinate; the original U6 wording was corrected under Unreleased.
- **New API**: `run_u6_confinement_sweep()` in `src/tnfr/yang_mills/u6_sweep.py`, exported from `tnfr.yang_mills`.
- **Sweep coordinate**: legacy `ρ_U6 = max_i |Φ_s(i)|/(π/2)`; values below one mean below that magnitude scale, not U6-confined. A U6 drift verdict requires a reference snapshot.
- **Telemetry recorded**: gap statistics, self-adjointness, seeded local-U(1)
  spectral invariance, pure-gauge cycle-closure residuals, applicable snapshot
  checks, legacy magnitude ratios, and finite-scope metadata. The cycle residual
  is numerical error, not curvature activity.
- **Validation**: 5 new tests in `tests/physics/test_yang_mills_u6_sweep.py` cover report shape/scope, U6 target tracking, gap contracts, reproducibility, invalid input rejection, and package-root import. Y1+Y2 focused run: `18 passed`.
- **Open boundary**: Y2 does not prove a U6 lower-bound theorem and does not address non-Abelian derivability (YMG-4) or continuum scaling (YMG-5). Next target: Y3 derivability audit.

#### Y1 — TNFR–Yang–Mills Finite Structural Gauge Gap Diagnostic

- **Verdict**: `DIAGNOSTIC_SURFACE_CREATED` — first TNFR-native Yang–Mills / structural mass-gap attack surface implemented as a finite-graph diagnostic, not a Clay-strength proof.
- **New package**: `src/tnfr/yang_mills/` with `build_structural_gauge_graph()`, `build_structural_gauge_gap_operator()`, and `compute_structural_gauge_gap()`.
- **Operator (current notation)**: `H_structural = L_A + V_F + V_Phi`, where
  `A=d(arg Psi)` makes `L_A` unitarily equivalent to the ordinary weighted graph
  Laplacian, `V_F` is analytically zero apart from numerical cycle-closure
  residuals, and `V_Phi` is the selected single-snapshot magnitude penalty
  `Phi_s²/(pi/2)²`; it is not a U6 test.
- **TNFR scope discipline**: the reported number is the spectral gap of this
  selected finite matrix. It is not evidence of dynamical gauge curvature,
  confinement, a physical mass, or a continuum Yang–Mills mass gap.
- **Validation**: 13 new tests in `tests/physics/test_yang_mills_structural_gap.py` cover graph construction, self-adjointness, non-negative finite gap reporting, seeded local-U(1) spectral invariance, reproducibility, package imports, and no EPI/phase mutation. Focused run: `13 passed`.
- **Open boundaries**: non-Abelian derivability (YMG-4) and continuum / thermodynamic scaling (YMG-5) remain open.
- **Documentation**: `theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md` records the Y-series gap ledger and updates the next target to Y2 (U6 confinement sweep).

#### N17-A — U3+U5 → K41: Analytical Cascade Locality (ANALYTICAL_CONSISTENT_CONDITIONAL)

- **Verdict**: `ANALYTICAL_CONSISTENT_CONDITIONAL` — K41 $k^{-5/3}$ spectrum derived conditionally from TNFR grammar rules U2+U3+U5+CDC; algebraically closed given the Cascade Development Condition.
- **Lemma U5-SS** (U5 + U2 → scale self-similarity): U5-uniformity (same canonical operators and constants at every hierarchy level) + U2 force $u_\ell = C(\varepsilon r_\ell)^{1/3}$ in the inertial range. The K41 scaling emerges from grammar structure (U5 collapses the dimensionless ratio to a level-independent constant), not from external dimensional analysis.
- **Lemma U3-CL** (U3 → cascade locality, conditional): Under Lemma U5-SS, U3 (phase-gated coupling, $|\phi_i - \phi_j| \le \Delta\phi_{\max}$) blocks all inter-level interactions **if and only if** the Cascade Development Condition (CDC) holds → constant energy flux $\Pi_\ell = u_\ell^3 / r_\ell = \varepsilon$ across scales.
- **Theorem** (U2 + U3 + U5 + CDC → K41): $E(k_\ell) \sim \varepsilon^{2/3} k_\ell^{-5/3}$ — proof: $E_\ell \sim u_\ell^2 \sim \varepsilon^{2/3} r_\ell^{2/3}$, $E(k_\ell) = E_\ell / \Delta k_\ell$ with $\Delta k_\ell \sim k_\ell$ (log bands) → $E(k_\ell) \sim \varepsilon^{2/3} k_\ell^{-5/3}$. □
- **CDC (irreducible gap)**: CDC (adjacent cascade levels have $|\phi_{\ell,i} - \phi_{\ell+1,j}| \ge \Delta\phi_{\max}$ for all $i,j$) is not derivable from U3, U5, or the nodal equation. It is the K41 locality hypothesis restated in TNFR language, and the structural analogue of $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$ in the Riemann programme — reachable only by a sufficiently developed turbulent cascade, not from the canonical operator catalog alone.
- **N17-A does not close NS-G1..G4** — those gaps concern continuum-limit, uniform bounds, BKM criterion, and vortex stretching; not cascade locality.
- **N17-B pre-registered** (deferred): empirical energy spectrum via `energy_spectrum_3d()` (to be implemented in `src/tnfr/navier_stokes/operator.py`), n ∈ {32, 48}, ν ∈ {0.01, 0.005}, T = 2.0. Expected verdict: `STEEPER_THAN_K41` (CDC not satisfied at Re_eff ≤ 500).
- **Documentation**: `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §20 (full lemmas, theorem, CDC gap analysis, verdict table, N17-B pre-registration spec).

#### N16 — NS-G5 Closure: 2D-Embedding Lemma

- **Verdict**: NS-G5 **CLOSED** at the discrete-operator level via the **2D-Embedding Lemma (Theorem NS-G5-TNFR)**.
- **Algebraic proof** (three steps using existing `TNFRNavierStokesOperator` methods on z-independent u = (u₀(x,y), u₁(x,y), 0)):
  1. `vorticity_3d`: ω₀ = ω₁ = 0, ω₂ = ∂_x(v) − ∂_y(u)
  2. `vortex_stretching_field`: S_a = ω₀·∂_x(u_a) + ω₁·∂_y(u_a) + ω₂·∂_z(u_a) = ω₂·0 = 0 for all a
  3. `stretching_production`: = 0.0 exactly in IEEE 754
- **TNFR reading**: z-channel decoupling → no cross-channel ΔNFR → enstrophy ≤ viscous dissipation (monotonically non-increasing) → discrete TNFR analogue of 2D NS global regularity.
- **Contrast with 3D**: ∂_z(u_a) ≠ 0 activates cross-channel ΔNFR coupling → stretching production generically positive → U2 (convergence/boundedness) is not guaranteed → vortex stretching amplification is structurally active.
- **Empirical corroborator**: `examples/85_navier_stokes_dimensional_asymmetry.py` — z-independence → `stretching_production` ≈ 0 at machine precision across all tested configurations (commit `1fac358b`).
- **Scope**: NS-G5 closure does NOT affect NS-G1..G4 and does NOT address the Clay Millennium Problem (3D global regularity).
- **Documentation**: `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §19.

#### N15 REMESH fixed-delay surrogate — corrected historical record

- **Historical milestones**: commits `a1f298fd`, `badac156`, and `48b0574a`
  introduced a history-space projection, projected charge/energy claims, and
  spectral comparisons under the label “REMESH-∞ closure.” The corrected
  analysis is maintained in
  [theory/REMESH_INFINITY_DERIVATION.md](theory/REMESH_INFINITY_DERIVATION.md)
  §§1–23.
- **Valid restricted result**: on a finite cyclic history window, the unclipped
  fixed-coefficient filter
  $F=\beta I+\gamma S^{\tau_l}+\delta S^{\tau_g}$ is a normal contraction for
  $0<\alpha<1$. Its Cesàro averages converge to the orthogonal projection onto
  $\ker(I-F)$. The common fixed modes have period
  $\gcd(\tau_l,\tau_g)$; the former `lcm` lattice statement was incorrect.
- **Runtime boundary**: `apply_network_remesh` is history-gated, configurable,
  structurally clipped, and uses a snapshot whose location changes with
  $\tau_g$. The finite cyclic result does not establish its literal
  $\tau_g\to\infty$ limit.
- **Invariant correction**: convex EPI mixing is not an isometry. Orthogonal
  projection contracts the surrogate history norm, but it neither proves
  conservation of TNFR structural charge nor monotonicity of the structural
  candidate energy. The former exact-conservation and universal $O(1/n)$ rate
  claims are superseded.
- **Registry scope**: the surrogate projection can be computed without adding
  an implementation registry entry. This does not prove that the 13 registered
  operators exhaust all admissible TNFR transformations; catalog completeness
  remains open.
- **External-program scope**: the finite surrogate does not advance RH,
  Navier–Stokes regularity, K41, or an RMT correspondence. Historical branch
  labels now apply only to comparisons made inside that declared surrogate.

## [0.0.3.3] - 2026-03-07

### Documentation Audit (Sessions 1-4)

- **Comprehensive tone audit**: Removed speculative/grandiose language across 25+ files
- **TNFR_RIEMANN_RESEARCH_NOTES.md**: Reduced from 2679 to 1499 lines (removed unfounded claims)
- **AGENTS.md**: Fixed 'inevitability' → 'derivation strength', updated conservation test count (62 → 88), verified all 40+ cross-reference links
- **Synced .github/agents/my-agent.md** with AGENTS.md corrections
- **Updated test counts** to 1,655 across 8 files
- **Removed orphaned file**: src/train_gmx_optimizer.py
- **Fixed contradictions** between AGENTS.md and theory/ documents
- **Validated**: 1653 passed, 2 skipped

## [0.0.3.2] - 2026-03-06

### Documentation & Consistency Fixes

- **Corrected false Γ(4/3)/Γ(1/3) derivation** in MINIMAL_STRUCTURAL_DEGREES.md and FUNDAMENTAL_THEORY.md (Γ(4/3)/Γ(1/3) = 1/3, not 0.7711)
- **Synchronized .github/agents/my-agent.md** with AGENTS.md (K_φ threshold, MIN_BUSINESS_COHERENCE, THOL_MIN values)
- **Fixed CHANGELOG version** to match pyproject.toml (0.0.3.2)
- **Fixed MIN_BUSINESS_COHERENCE precision** in ARCHITECTURE.md and CONTRIBUTING.md (0.751 → 0.7506)
- **Resolved phantom docs/TNFR_FORCES_EMERGENCE.md** references across 8+ files
- **Removed dead code** src/tnfr/config.py (shadowed by config/ package)
- **Cleaned unused imports** in sdk/simple.py

## [0.0.3] - 2026-03-05

### Structural Conservation Theorem

- **conservation.py**: Introduced a finite-trajectory structural-balance residual
  and Noether-like vocabulary. Later audits established that U1-U6 do not imply
  a vanishing residual or a universal conservation theorem.
- Added charge-density/current, structural-energy, Ward-residual, stability-policy,
  and spectral diagnostics; these names do not by themselves establish conserved
  charges, Noether hypotheses, or Lyapunov monotonicity.
- Two-sector structure: Potential (Φ_s ↔ J_ΔNFR) and Geometric (K_φ ↔ J_φ) coupled through Ψ = K_φ + i·J_φ
- 62 validation tests, charge drift < 0.03% across topologies

### Dissipative Conservation

- **dissipative_conservation.py**: GPU-accelerated dissipative conservation analysis with PyTorch backend
- Phase field computation, dissipation rate tracking, and energy budget monitoring

### Closed-Loop Integrity Monitor

- **integrity.py**: `StructuralIntegrityMonitor` with complete postconditions for all 13 canonical operators
- Each operator (AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH) has verified pre/postcondition contracts
- Automatic violation detection and reporting

### Grammar-Aware Dynamics

- **grammar_dynamics.py**: Bridge between grammar validation (U1-U6) and dynamic operator selection
- Incremental U1-U6 checks: `validate_candidate()`, `filter_candidates()`, `suggest_alternative()`, `enforce_grammar_on_glyph()`
- Priority-based operator substitution with fallback logic
- **grammar_application.py**: Pre-validation in `apply_glyph_with_grammar()` for grammar enforcement before operator application
- **selectors.py**: `_soft_grammar_prefilter()` wired with grammar_dynamics for operator filtering

### Simple SDK — Research-Grade Access

- **simple.py**: Upgraded with full Structural Field Tetrad, conservation laws, and unified telemetry access
- **TetradSnapshot** dataclass: phi_s, grad_phi, k_phi, xi_c, j_phi, j_dnfr with `is_safe()` and `summary()`
- **ConservationReport** dataclass: noether_charge, energy, lyapunov_stable, lyapunov_derivative, conservation_quality with `summary()`
- **10 new Network methods**: `tetrad()`, `fields()`, `conservation()`, `telemetry()`, `tensor_invariants()`, `emergent_fields()`, `evolve_grammar_aware()`, `integrity_check()`, upgraded `results()` and `info()`
- **TNFR.analyze()**: One-shot comprehensive analysis (coherence, tetrad, conservation, tensor invariants, emergent fields, integrity)
- Feature-gated imports: `_HAS_FIELDS`, `_HAS_CONSERVATION`, `_HAS_INTEGRITY`, `_HAS_GRAMMAR_DYNAMICS`
- 29 new tests in `tests/sdk/test_simple_advanced.py`

### Shared Test Infrastructure

- **tests/conftest.py**: Centralized test fixtures (`make_ring_graph`, `make_node_data`, `ring3`, `ring5`, `small_graph`)
- DRY reduction across 16+ test files that previously duplicated `_make_graph` helpers

### Code Quality

- Fixed bare `except:` clauses in grammar_dynamics.py (now `except Exception:`)
- NAV bypass fix for grammar validation edge case
- Redundancy elimination across physics helpers
- Rich operator postconditions (13/13 coverage)

### Cross-Codebase Constant Unification (Round 1)

- **grammar_types.py**: Eliminated duplicate operator sets (single canonical definition)
- **THOL_MIN_COLLECTIVE_COHERENCE**: Unified to canonical 0.2413 (was 0.3)
- **MIN_BUSINESS_COHERENCE**: Centralized to canonical formula (e×φ)/(π+e) ≈ 0.7506
- **health_analyzer.py / self_organization.py**: Aligned fallback values to canonical

### Phase Gradient Threshold Unification

- **Canonical value**: γ/π ≈ 0.1837 (Kuramoto critical coupling in TNFR units)
- **Unified across 9 code files**: Replaced competing values (0.2904, 0.2886, 0.2915, 0.38) with single canonical derivation
- **Updated 8 documentation files**: Consistent threshold references throughout

### Cross-Codebase Constant Unification (Round 2)

- **compute_structural_potential_field**: Added alias in physics/fields.py (was silently missing, imported in 2 files)
- **SHA_VF_FACTOR comment**: Fixed from ≈ 0.8476 to correct ≈ 0.9015 in defaults_core.py
- **Operator fallback values**: SHA (0.85→0.9015), NUL (0.85→0.9015), VAL (1.05→1.0676) aligned to canonical
- **K_φ hotspot formula**: Fixed in conservation.py from 2π/√5 ≈ 2.8099 to canonical 0.9×π ≈ 2.8274
- **grammar_core.py K_φ default**: Fixed from 3.0 to canonical 2.8274
- **telemetry/constants.py**: Removed dead try/except ImportError fallback; direct canonical imports
- **config.py**: Structural field thresholds now derive from constants.canonical (was hardcoded)
- **pyproject.toml**: Added mpmath to core dependencies (was required but unlisted)
- **Documentation sync**: Updated 7 doc files with correct threshold values and test counts

### Test Suite

- **1,655 tests** (1,646 passing, 9 skipped), 0 failing
- Coverage spans operators, physics, dynamics, grammar, conservation, integrity, SDK, and factorization

## [0.0.2] - 2025-11-29

### TNFR Development Doctrine Establishment

- **Foundational Principle**: Added TNFR Development Doctrine as core methodological commitment
- **Theoretical Integrity**: Commitment to follow mathematics objectively from nodal equation ∂EPI/∂t = νf · ΔNFR(t)
- **Scientific Independence**: Defend conclusions emerging rigorously from TNFR principles regardless of external paradigm alignment
- **Validation Criteria**: Established 4-point validation framework (Derivable, Testable, Reproducible, Coherent)

### Complete Framework Expansion

- **29 New Examples**: Comprehensive examples (11-39) covering physics, biology, cosmology, consciousness studies
- **TNFR-Riemann Program**: Complete theoretical framework connecting discrete operators to Riemann Hypothesis
- **Advanced Physics Modules**: Classical mechanics, quantum mechanics, symplectic integration implementations
- **Extensive Theory Documentation**: 25+ specialized theoretical documents in theory/ directory

### Documentation Academic Modernization

- **Unified Academic Tone**: Systematic elimination of grandilocuent language across all documentation
- **README Gateway**: Transformed main README into coherent documentation entry point
- **Consistent Terminology**: Standardized "Primary theoretical reference" replacing "SINGLE SOURCE OF TRUTH"
- **Professional Presentation**: Enhanced credibility through formal academic language standards

### Test Suite Optimization

- **Major Cleanup**: Removed 58 obsolete test files (82 → ~30 files)
- **100% Pass Rate**: Achieved 173 passing, 7 skipped, 0 failing tests
- **Focused Validation**: Retained only tests validating TNFR theoretical foundations
- **Core Coverage**: Mathematics, operators, physics, validation maintained

### Technical Enhancements

- **Enhanced N-body Dynamics**: Improved TNFR integration with classical mechanics
- **Riemann Operator**: Complete implementation with eigenvalue analysis capabilities
- **Type System**: Enhanced type definitions and structural validation
- **Code Quality**: Significant cleanup removing outdated components

## [9.7.0] - 2025-11-29

### Major Theoretical Enhancements

- **Universal Tetrahedral Correspondence**: Complete mathematical framework establishing exact mapping between four universal constants (φ, γ, π, e) and four structural fields (Φ_s, |∇φ|, K_φ, ξ_C) *(later superseded — see the φ/γ/e purge under [Unreleased]: only π is a genuine structural scale)*
- **Unified Field Framework**: Mathematical unification discovering complex geometric field Ψ = K_φ + i·J_φ with emergent invariants
- **Self-Optimizing Engine**: Self-optimization capabilities with unified field telemetry for automated structural optimization
- **Complete Academic Documentation**: Comprehensive conversion to formal academic tone across entire documentation ecosystem

### Canonical Invariants Optimization

- Consolidated from 10 to 6 canonical invariants based on mathematical derivation from nodal equation
- Optimized invariants: Nodal Equation Integrity, Phase-Coherent Coupling, Multi-Scale Fractality, Grammar Compliance, Structural Metrology, Reproducible Dynamics
- Enhanced theoretical consistency and reduced redundancy

### Documentation Modernization

- **AGENTS.md**: Complete academic conversion maintaining single source of truth status
- **README.md**: Restructured with new Getting Started section and clear learning paths
- **GLOSSARY.md**: Comprehensive expansion with Universal Tetrahedral Correspondence coverage
- Eliminated promotional language and emojis across entire ecosystem
- Updated all version references to 9.7.0

### Structural Field Tetrad

- **Complete Mathematical Foundations**: All four canonical fields now have rigorous mathematical derivations
- **CANONICAL Status**: Φ_s, |∇φ|, K_φ, ξ_C all promoted to canonical status with theoretical validation
- **Unified Complex Geometry**: Integration of curvature and transport via complex field Ψ

### Development Infrastructure

- Updated pyproject.toml to v9.7.0 with current dependency structure
- Modernized CONTRIBUTING.md with academic tone and current 6 invariants
- Enhanced TESTING.md with updated invariant validation framework
- Complete English-only policy implementation

## [9.1.0] - 2025-11-14

### Added

- Phase 3 structural instrumentation:
  - `run_structural_validation` aggregator (grammar U1-U3 + field thresholds Φ_s, |∇φ|, K_φ, ξ_C, optional ΔΦ_s drift).
  - `compute_structural_health` with risk levels and recommendations.
  - `TelemetryEmitter` integration example (`examples/structural_health_demo.py`).
  - Performance guardrails: `PerformanceRegistry`, `perf_guard`, `compare_overhead`.
  - CLI: `scripts/structural_health_report.py` (on-demand health summaries).
  - Docs: README Phase 3 section, CONTRIBUTING instrumentation notes, `docs/STRUCTURAL_HEALTH.md`.
- Glyph-aware grammar error factory (operator glyph → canonical name mapping).

### Tests

- Added unit tests for validation, health, grammar error factory, telemetry emitter, performance guardrails.

### Performance

- Validation instrumentation overhead ~5.8% (moderate workload) below 8% guardrail.

### Internal

- Optional `perf_registry` parameter in `run_structural_validation` (read-only timing).
- Canonical operator registry frozen (removed dynamic auto-registration, cache
  invalidation, metaclass telemetry, reload script). Attempting dynamic
  registration now raises. Ensures strict adherence to unified grammar (U1-U4)
  and prevents non-canonical transformations.

### Deferred

- U4 bifurcation validation excluded pending dedicated handler reintroduction.

### Integrity

- All changes preserve TNFR canonical invariants (no EPI mutation; phase verification intact; read-only telemetry/validation).
- Registry immutability strengthens invariants #1 (EPI only via operators), #4
  (operator closure) and #5 (phase verification untouched). Tests updated:
  removed dynamic registration tests; added `test_canonical_operator_set`.

## [9.0.2]

Previous release (see repository history) with foundational operators, unified grammar, metrics, and canonical field tetrad.

---
