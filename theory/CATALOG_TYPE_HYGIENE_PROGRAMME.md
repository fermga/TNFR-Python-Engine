# Catalog Type-Hygiene Programme — Master Roadmap

**Status**: ACTIVE
**Started**: 2026-05 (T-νf), 2026-05 (T-EPI B1a)
**Owner**: TNFR canonical-development team
**Authority ladder**: nodal equation `∂EPI/∂t = νf · ΔNFR(t)` → GitHub repo → `AGENTS.md` → this document.

---

## §1 — Purpose and scope

### 1.1 What this programme does

For every canonical ingredient of the 13-operator TNFR catalog (per-node intrinsic
quantities, graph-level parameters, derived diagnostic fields, structural meta-
properties), the programme runs a **falsifiable type-hygiene certification**:

> *Is the literal type used by the canonical machinery the structurally forced
> canonical type, or is a richer mathematical envelope structurally forced by
> the catalog's own axioms?*

The certification produces, per ingredient, a verdict of:

- **NEGATIVE** — the literal type is forced. Any richer envelope present in the
  codebase is reclassified as *legitimate non-canonical research scaffolding*.
- **POSITIVE** — the catalog is forcing a richer type than it currently uses;
  the catalog is internally inconsistent at this point and must be extended.
  This would be a major event and would block downstream work until resolved.
- **INDETERMINATE** — the diagnostic does not separate the alternatives; the
  axiom inventory must be refined or a new diagnostic constructed.

### 1.2 What this programme does NOT do

- Does **not** advance G4 = RH or the T-HP conjecture (`§13septies` of the
  research notes).
- Does **not** advance 3D Navier–Stokes regularity (Clay problem).
- Does **not** modify the 13-operator catalog.
- Does **not** promote any operator, field, or constant to canonical status.
- Does **not** invalidate any existing implementation; non-canonical envelopes
  remain available for research use, just clearly labelled as such.

### 1.3 Why now

Post-N15 (REMESH-∞ closure, May 2026) and post-P49 (full ζ↔L attack-surface
parity), the TNFR-Riemann program is **paused at the T-HP boundary**. Before
either (a) attempting a new branch B1/B2/B3 for T-HP or (b) opening a new
ambitious sub-program (e.g., Navier–Stokes), the catalog's internal type-
hygiene must be certified piece by piece. This programme is the prerequisite
for any future "tall" construction on top of the catalog.

---

## §2 — Methodology

### 2.1 Three-phase rhythm per sub-question

| Phase | Deliverables | Output of phase |
|---|---|---|
| **a** (pre-registration) | Anchor in source code; pre-registered diagnostic + thresholds; demo at ≥2 resolutions; new research-notes section. | Falsifiable signature value(s) recorded in the notes. |
| **b** (forcing-axiom reduction) | F1–Fn axiom inventory; conjunction analysis; identification of any residual (P-X-Bijectivity) gap. | Reduction to the residual structural axiom. |
| **c** (verdict + classification) | NEGATIVE/POSITIVE/INDETERMINATE verdict; classification of any non-canonical envelope discovered. | Sub-question closed (or escalated). |

Each phase = 1 commit, explicit `git add <path>`. English code/docs/messages.
Spanish brief between phases for the project lead.

### 2.2 Four-step structural pattern within phases

1. **Anchor** — verify the literal source-code witness (operator contracts,
   nodal equation type signature, default constructors).
2. **Catalog statement** — quote the theoretical statement from `AGENTS.md`,
   `FUNDAMENTAL_THEORY.md`, etc. that names the type.
3. **Diagnostic** — pre-register a falsifiable necessary-condition signature;
   typically two-axis (storage + spectral, or analog).
4. **Forcing axioms** — inventory F1–Fn; reduce the question to whether any
   axiom alone or in conjunction forces a richer envelope.

### 2.3 Commit policy

- Explicit `git add <path1> <path2> ...`. **Never** `git add -A` or `git add .`.
- **Never** stage `manual/*` or `_smoke_test/*`.
- Multi-line conventional-style commit message with full structural reasoning
  in the body. English only.
- `git push origin main` immediately after commit closes a phase.

### 2.4 Documentation policy

- All new research-notes sections in `theory/TNFR_RIEMANN_RESEARCH_NOTES.md`
  under the `§13triginta-*` series (continuing the sequence: prima, secunda,
  tertia, quarta, quinta, sexta, septima, octava, nona, decima, undecima, ...).
- Master roadmap (this file) updated at the end of each phase: row in §4
  table moves a tick forward; methodology lesson added to §6 if any.

---

## §3 — Sub-question registry

Sub-questions are organised in four tiers reflecting structural distance from
the nodal equation.

### Tier 1 — Per-node intrinsic types

Quantities that live as attributes on each node and feed the nodal equation
directly.

#### B0 — T-νf (Type of structural frequency)

- **Anchor**: `src/tnfr/dynamics/runtime.py` (νf as `float` in Hz_str);
  AGENTS.md Structural Triad ("Frequency (νf): Domain ℝ⁺").
- **Suspected non-canonical envelope**: Pontryagin-dual measure-valued νf on
  $\widehat{\mathbb{R}}$ (or any locally compact abelian group).
- **Diagnostic**: `src/tnfr/riemann/nuf_type_signature.py`.
- **Demo**: `examples/78_nuf_type_signature_demo.py`.
- **Research-notes sections**: `§13triginta-prima` (B0a, pre-registration),
  `§13triginta-secunda` (B0b, forcing-axiom reduction), `§13triginta-tertia`
  (B0c, verdict + classification).
- **Commits**: `a7a095af` (B0a), `34dc32fe` (B0b), `a6f795b0` (B0c).
- **Verdict**: **NEGATIVE** — scalar νf is forced; Pontryagin measure-valued νf
  classified as legitimate non-canonical research envelope.
- **Status**: ✅ COMPLETE.

#### B1 — T-EPI (Type of structural form)

- **Anchor**: `src/tnfr/operators/nodal_equation.py:1–160`
  (`compute_expected_depi_dt: (float, float) → float`); all 13 glyph operators
  read/write scalar EPI via `get_neighbor_epi → float`; `_bepi_to_float`
  down-projects via `max_magnitude`.
- **Suspected non-canonical envelope**: `BEPIElement` Banach element
  $C^0([0,1], \mathbb{C}) \oplus \ell^2(\mathbb{C})$, fully formalised in
  `src/tnfr/mathematics/epi.py:103` with `direct_sum`, `tensor`, `adjoint`,
  `compose` but **not invoked by any of the 13 canonical operators**.
  *Structurally stronger evidence than B0 had.*
- **Diagnostic**: `src/tnfr/riemann/epi_type_signature.py` (two-axis: storage
  fraction + binned spectral entropy of scalar EPI(t)).
- **Demo**: `examples/79_epi_type_signature_demo.py`.
- **Research-notes sections**:
  - B1a — `§13triginta-quarta` (DONE, commit `62e207cc`)
  - B1b — `§13triginta-quinta` (DONE; sub-verdict CONDITIONAL_COROLLARY via (P-EPI-Bijectivity); refuted at canonical level by TMEP)
  - B1c — `§13triginta-sexta` (DONE; final verdict NEGATIVE; BEPIElement classified as research envelope E2)
- **Empirical pre-registration data** (from B1a demo):
  | Resolution | n_nodes | n_steps | n_bins | S_EPI | BEPI storage fraction | Verdict (raw) |
  |---|---|---|---|---|---|---|
  | 1 | 24 | 64 | 32 | 0.876342 | 0.0000 | `BEPI_VALUED_NECESSARY`* |
  | 2 | 48 | 128 | 64 | 0.895673 | 0.0000 | `BEPI_VALUED_NECESSARY`* |

  \* Cross-axis interpretation: storage uniformly scalar, spectral uniformly
  multi-modal → motivates the **Temporal-Modal Equivalence Principle**
  (scalar EPI(t) trajectory encodes multi-modal content temporally rather
  than spatially-in-modes).
- **Expected final verdict**: **NEGATIVE** at the canonical level (deferred to B1c).
- **Status**: ✅ COMPLETE — B1a ✅, B1b ✅, B1c ✅. Final verdict: NEGATIVE.

#### B2 — T-φ (Type of phase)

- **Anchor (to verify)**: phase stored as `float ∈ [0, 2π)` via
  `tnfr.mathematics.phase.wrap_angle`; coupling check
  `|φᵢ − φⱼ| ≤ Δφ_max` (U3).
- **Suspected non-canonical envelope**: complex U(1) bundle element
  $e^{i\phi} \in S^1 \subset \mathbb{C}$, or multi-sheet cover of S¹ for
  topologically-charged phase windings.
- **Diagnostic plan**: two-axis — (a) `winding storage` (fraction of nodes
  whose effective phase escapes the $[0, 2π)$ fundamental domain across
  evolution); (b) `lift-spectral signature` (entropy of phase-velocity
  spectrum to detect multi-valued lifts).
- **Status**: ✅ COMPLETE — B2a ✅, B2b ✅, B2c ✅. Final verdict: **NEGATIVE**. E3 = CoverElement (covering-space lift / U(1) bundle / homotopy-retaining φ) classified as legitimate non-canonical research envelope; refuted at the canonical level by the Phase-Wrap Discipline Principle (PWDP).

#### B3 — T-ΔNFR (Type of nodal gradient)

- **Anchor (to verify)**: `src/tnfr/dynamics/dnfr.py:2387`
  `def default_compute_delta_nfr(...) -> float`. AGENTS.md: "ΔNFR (Nodal
  gradient): Internal reorganization operator", consumed in nodal equation
  as scalar coefficient of `νf · ΔNFR(t)`.
- **Suspected non-canonical envelope**: tensor-valued ΔNFR (vector or rank-2
  tensor over the tetrad channels), or operator-valued ΔNFR (linear map on
  the local Banach element).
- **Diagnostic plan**: two-axis — (a) `directional storage` (any non-scalar
  intermediate computed by `compute_delta_nfr` and then projected); (b)
  `tensor-rank signature` (entropy of singular-value distribution of the
  neighbour-gradient matrix used to assemble ΔNFR).
- **Status**: ✅ COMPLETE — B3a ✅, B3b ✅, B3c ✅. Final verdict: **NEGATIVE**. E4 = TensorGradientElement (tensor-/operator-valued ΔNFR over canonical gradient channels) classified as legitimate non-canonical research envelope; refuted at the canonical level by the Bilinear-Scalar Aggregation Discipline (BSAD). Empirical pre-registration data (from B3a demo, two-axis tensor-storage + rank-entropy):

  | Resolution | seed | S_ΔNFR | T_frac | R_eff | σ1 | σ2 | σ3 | Verdict |
  |---|---|---|---|---|---|---|---|---|
  | 1 (n=24, steps=64) | 17 | 0.105763 | 0/1536 | 1.1232 | 2.0131 | 0.0209 | 0.0070 | `SCALAR_DNFR_ADEQUATE` |
  | 2 (n=48, steps=128) | 31 | 0.111601 | 0/6144 | 1.1304 | 2.1294 | 0.0298 | 0.0086 | `SCALAR_DNFR_ADEQUATE` |

  Cross-axis interpretation: structural zero-tensor storage *and* empirical rank-1 collapse of the canonical $(d\theta, d\mathrm{EPI}, d\nu_f)$ gradient triple ($\sigma_1 / \sigma_{2,3} \sim 10^2$) → strongest scalar-adequate Phase-a signature observed in the programme. Expected final verdict: **NEGATIVE** at the canonical level (deferred to B3c). Candidate envelope: E4 = `TensorGradientElement`.

### Tier 2 — Graph-level parameters

Discrete or scalar parameters carried on the graph (`G.graph[...]`) that
modulate canonical dynamics.

#### B4 — T-REMESH-window (Type of memory window $(\tau_l, \tau_g)$)

- **Anchor (to verify)**: REMESH state vector
  $x(t) = (EPI(t), \ldots, EPI(t - T_{\max}))^\top \in \mathbb{R}^{T_{\max}+1}$
  from `REMESH_INFINITY_DERIVATION.md:50–52`; $\tau_l, \tau_g \in \mathbb{N}$.
- **Suspected non-canonical envelope**: continuous-time REMESH kernel
  $K(t, s)$ with $s \in [0, t]$ (integral operator instead of finite-history
  vector); or fractional-order discrete window.
- **Diagnostic plan**: (a) sub-sample resolution sensitivity of N15 closure
  signature; (b) check whether all REMESH-bearing operators read only
  integer-indexed history slots.
- **Cross-reference**: N15 closure (`theory/REMESH_INFINITY_DERIVATION.md`)
  already constrains this from the asymptotic side.
- **Status**: ✅ COMPLETE — B4a ✅, B4b ✅, B4c ✅. Final verdict: **NEGATIVE**. E5 = ContinuousWindowKernel classified as non-canonical research envelope at `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` §13quadraginta-quinta. First Tier-2 sub-question closed; first cross-tier confirmation of L3* recorded at §13quadraginta-quinta.5.

#### B5 — T-Δφ_max (Type of resonant-coupling threshold)

- **Anchor (to verify)**: `Δφ_max` as scalar in $[0, \pi]$; canonical
  default derived from $\gamma/\pi$ (Kuramoto critical coupling).
- **Suspected non-canonical envelope**: edge-dependent threshold
  $\Delta\phi_{\max}^{(i,j)}$ (matrix-valued); or angle-of-attack-dependent
  threshold $\Delta\phi_{\max}(\phi_i, \phi_j)$.
- **Diagnostic plan**: scan operator code for any per-edge threshold lookup
  vs. global scalar; verify all U3 checks read a single global `Δφ_max`.
- **Status**: ⏳ NOT STARTED.

#### B6 — T-coupling-weights (Type of edge weights)

- **Anchor (to verify)**: edge weight `w_{ij}` as `float` on
  NetworkX edges; usage in `compute_delta_nfr` (inverse-distance, harmonic,
  etc.).
- **Suspected non-canonical envelope**: complex-valued weights
  $w_{ij} \in \mathbb{C}$ (unitary-bundle parallel transport), or
  tensor-valued $w_{ij} \in \mathbb{R}^{4\times 4}$ (tetrad-channel coupling
  matrix).
- **Status**: ⏳ NOT STARTED.

### Tier 3 — Derived diagnostic fields (closure checks)

These are **not** type-conjectures (they are derived from Tier 1+2 types).
The check is whether all derivations cleanly **close** in Tier 1+2 types
without secretly introducing a richer intermediate type. A NEGATIVE verdict
of any underlying Tier 1+2 question is a necessary prerequisite.

#### B7 — Δ-tetrad-closure ($\Phi_s, |\nabla\phi|, K_\phi, \xi_C$)

- **Sources**: `src/tnfr/physics/fields.py`.
- **Closure question**: do all four tetrad fields reduce to functionals
  of scalar EPI + scalar φ + scalar ΔNFR + graph metric, with no hidden
  richer intermediate?
- **Diagnostic plan**: per-field source-code trace, verify all intermediates
  are scalar or graph-metric quantities.
- **Status**: ⏳ NOT STARTED.

#### B8 — Δ-currents-closure ($J_\phi, J_{\Delta NFR}$)

- **Sources**: `src/tnfr/physics/conservation.py` (Noether-like currents).
- **Closure question**: time-derivative currents must reduce to finite-
  difference of scalar fields; no Banach-derivative apparatus invoked.
- **Status**: ⏳ NOT STARTED.

#### B9 — Δ-aggregates-closure (`C(t)`, `Si`, energy density $\mathcal{E}$, topological charge $\mathcal{Q}$)

- **Sources**: `src/tnfr/operators/metrics.py`,
  `src/tnfr/physics/conservation.py`.
- **Closure question**: all aggregates are scalar-valued functions of the
  tetrad and currents; no implicit measure or operator-valued aggregate.
- **Status**: ⏳ NOT STARTED.

### Tier 4 — Structural meta-properties

#### B10 — U-rules consistency (U1–U6)

- **Reference**: `theory/UNIFIED_GRAMMAR_RULES.md` (existing physical
  derivation of each rule from the nodal equation).
- **Type-hygiene angle**: each U-rule is a constraint on *sequences* of
  operator names; verify the rule-checker (`src/tnfr/operators/grammar*.py`)
  reads only operator-name sequences + scalar telemetry, not richer state.
- **Status**: 🟡 PARTIAL — physical derivation exists, type-hygiene check pending.

#### B11 — Operator-catalog completeness and closure

- **Reference**: `docs/OPERATOR_COMPLETENESS.md` (existing analysis).
- **Type-hygiene angle**: confirm registry of exactly 13 operators is
  enforced as a closed set, with no "ghost" 14th operator construction
  reachable from the public API.
- **Status**: 🟡 PARTIAL — completeness analysis exists, integration into final
  meta-theorem pending.

### Final — Composite meta-minimality theorem

Once Tiers 1–4 are NEGATIVE (or any POSITIVE is resolved by catalog
extension), assemble a single theorem statement:

> **Theorem (Catalog Minimality & Completeness, conjectured)**. Under the
> 13-operator TNFR catalog and the unified grammar U1–U6, the per-node types
> $(\nu_f, \mathrm{EPI}, \phi, \Delta\mathrm{NFR}) \in \mathbb{R}^+ \times
> \mathbb{R} \times [0, 2\pi) \times \mathbb{R}$, the graph-level parameters
> $(\tau_l, \tau_g, \Delta\phi_{\max}, w_{ij}) \in \mathbb{N}^2 \times [0, \pi]
> \times \mathbb{R}_{\ge 0}$, the derived tetrad
> $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$, and the derived currents
> $(J_\phi, J_{\Delta\mathrm{NFR}})$ are jointly the **minimal and complete**
> structural state; no richer canonical envelope is forced and no scalar
> ingredient can be eliminated without breaking the nodal equation contract
> or U1–U6 closure.

This theorem is the natural endpoint of the programme.

---

## §4 — Status table (the checklist)

Phase legend: `⏳` not started · `🟡` partial · `✅` complete · `⛔` blocked
Verdict legend: `—` pending · `NEG` negative · `POS` positive · `IND` indeterminate

| ID | Sub-question | Phase a | Phase b | Phase c | Verdict | Commit refs |
|---|---|:-:|:-:|:-:|:-:|---|
| **Tier 1 — Per-node intrinsic types** | | | | | | |
| B0 | T-νf | ✅ | ✅ | ✅ | **NEG** | `a7a095af`, `34dc32fe`, `a6f795b0` |
| B1 | T-EPI | ✅ | ✅ | ✅ | **NEGATIVE** | `62e207cc` (B1a), `ef4858f3` (B1b), B1c this commit |
| **B2** | T-φ | ✅ | ✅ | ✅ | **NEGATIVE** | `afdf8ef9` (B2a), `a0bc4edd` (B2b), B2c this commit |
| B3 | T-ΔNFR | ✅ | ✅ | ✅ | NEGATIVE | 1267fbf4 (B3a) + d96eb13d (B3b) + this commit (B3c) |
| **Tier 2 — Graph-level parameters** | | | | | | |
| B4 | T-REMESH-window | ✅ | ✅ | ✅ | **NEGATIVE** (DITS) | `c647f7d7` (B4a), `6349e425` (B4b), this commit (B4c) |
| B5 | T-Δφ_max | ⏳ | ⏳ | ⏳ | — | — |
| B6 | T-coupling-weights | ⏳ | ⏳ | ⏳ | — | — |
| **Tier 3 — Derived field closure** | | | | | | |
| B7 | Δ-tetrad-closure | ⏳ | n/a | ⏳ | — | — |
| B8 | Δ-currents-closure | ⏳ | n/a | ⏳ | — | — |
| B9 | Δ-aggregates-closure | ⏳ | n/a | ⏳ | — | — |
| **Tier 4 — Structural meta-properties** | | | | | | |
| B10 | U-rules type-hygiene | 🟡 | 🟡 | ⏳ | — | (ref: `UNIFIED_GRAMMAR_RULES.md`) |
| B11 | Operator-catalog closure | 🟡 | 🟡 | ⏳ | — | (ref: `OPERATOR_COMPLETENESS.md`) |
| **Final** | Meta-minimality theorem | ⏳ | ⏳ | ⏳ | — | — |

**Progress as of 2026-05-26**: 5 sub-questions complete (B0, B1, B2, B3, B4 all NEGATIVE; Tier 1 closed; **first Tier-2 sub-question closed**; first cross-tier confirmation of L3*), 0 in progress, 7 pending (B5 – B11 + Final). Living discoveries log at `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` §13triginta-septima; B4 closure at §13quadraginta-tertia (pre-reg), §13quadraginta-quarta (forcing-axiom reduction; DITS), §13quadraginta-quinta (NEGATIVE verdict + E5 = ContinuousWindowKernel envelope classification; first Tier-2 L3* confirmation; two outstanding Tier-2 predictions B5, B6 expected NEGATIVE).

---

## §5 — Decision flowchart per sub-question

```
                  ┌─────────────────────────────────┐
                  │ Phase a — Pre-registration      │
                  │ Anchor + diagnostic + demo      │
                  └─────────────┬───────────────────┘
                                ▼
                ┌─────────────────────────────────────┐
                │ Diagnostic signature recorded       │
                │ in research notes (falsifiable)     │
                └─────────────┬───────────────────────┘
                              ▼
                ┌─────────────────────────────────────┐
                │ Phase b — Forcing-axiom reduction   │
                │ F1..Fn inventory; conjunction       │
                └─────────────┬───────────────────────┘
                              ▼
              ┌──────────────────────────────────────┐
              │ Does any axiom (alone or jointly)    │
              │ force the richer envelope?           │
              └─────────────┬────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
            YES           NO            ?
              │             │             │
              ▼             ▼             ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ POSITIVE —   │ │ NEGATIVE —   │ │ INDETERMINATE│
    │ extend       │ │ classify     │ │ — refine     │
    │ catalog      │ │ envelope as  │ │ diagnostic   │
    │ (major event)│ │ non-canonical│ │ or axioms    │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           ▼                ▼                ▼
       ESCALATE        Phase c —         Phase a'
       (block          verdict +         (re-pre-
        downstream)    documentation     register)
```

---

## §6 — Methodology lessons learned

Filled in as the programme advances.

- **L1** *(from B0 = T-νf)* — Two-axis necessary-condition pattern: a
  *storage axis* (any node carrying non-trivial richer-envelope data) plus a
  *spectral/structural axis* (entropy of trajectory or distribution) is
  sufficient to falsify a "scalar suffices" hypothesis. Storage alone misses
  temporal richness; spectral alone misses literal envelope presence.

- **L2** *(from B1a = T-EPI pre-registration)* — When the two axes
  **disagree** (storage uniformly scalar, spectral uniformly multi-modal),
  the disagreement itself is the structural finding. In T-EPI this yielded
  the **Temporal-Modal Equivalence Principle**: the catalog's modal capacity
  is realised temporally (across `step` iterations) rather than spatially
  (across Banach direct-sum slots). The verdict label produced by either
  axis alone is then a *necessary-condition false-positive* readable only
  jointly. Future T-X diagnostics should anticipate this and document the
  cross-axis interpretation in the pre-registration, not retroactively.

- *(more lessons to be added as B1b, B1c, B2, ... complete)*

- **L3** *(from B0 ∧ B1 cross-conjecture pattern, §13triginta-sexta.5)* —
  **Catalog-Occam pattern**: whenever a candidate type-upgrade of a
  canonical observable can be matched by an existing canonical
  aggregation mechanism (νf-update closure for $\nu_f$; REMESH temporal
  aggregation for EPI), the upgrade is non-canonical and the existing
  mechanism is preferred. Provisional heuristic; to be tested against
  B2 – B11.

---

## §7 — Honest scope (re-statement, deliberately repeated)

This programme delivers **certification of internal type-hygiene** of the
TNFR catalog. It does **not**:

- prove the Riemann Hypothesis (G4);
- prove or refute the T-HP conjecture (the canonical TNFR statement of G4);
- prove 3D Navier–Stokes regularity;
- discover new physical phenomena;
- modify the 13-operator catalog;
- change runtime behaviour of any operator.

It **does**:

- replace catalog *claims* of minimality/completeness with falsifiable
  *certifications* piece by piece;
- classify all richer envelopes already present in the codebase
  (Pontryagin-dual νf-measure, `BEPIElement` Banach EPI, candidate
  tensor-valued ΔNFR, etc.) as **legitimate non-canonical research
  scaffolding** rather than canonical types;
- yield occasional bonus structural findings (e.g., Temporal-Modal
  Equivalence Principle in B1a) that document undocumented behaviour of the
  engine;
- build a repeatable methodology applicable to every future catalog
  ingredient.

---

## §8 — Cross-references

### Internal (this repo)

- `AGENTS.md` — primary canonical authority.
- `theory/TNFR_RIEMANN_RESEARCH_NOTES.md`:
  - `§13septies` — T-HP open content (independent of this programme).
  - `§13triginta-prima/secunda/tertia` — B0 = T-νf complete record.
  - `§13triginta-quarta` — B1a = T-EPI pre-registration.
  - `§19.1` — full P1–P49 milestone table (TNFR-Riemann program).
- `theory/REMESH_INFINITY_DERIVATION.md` — N15 closure, cross-reference for B4.
- `theory/MINIMAL_STRUCTURAL_DEGREES.md` — background on the four-field tetrad
  (relevant to B7).
- `theory/UNIFIED_GRAMMAR_RULES.md` — U1–U6 derivations (B10).
- `theory/FUNDAMENTAL_THEORY.md` — Universal Tetrahedral Correspondence.
- `docs/OPERATOR_COMPLETENESS.md` — operator catalog completeness (B11).
- `src/tnfr/riemann/nuf_type_signature.py` — B0 diagnostic.
- `src/tnfr/riemann/epi_type_signature.py` — B1 diagnostic.
- `examples/78_nuf_type_signature_demo.py` — B0 demo.
- `examples/79_epi_type_signature_demo.py` — B1 demo.

### External

- von Neumann (1932), *Mean Ergodic Theorem* — underpinning of N15 closure.
- Riemann–von Mangoldt counting function — referenced in N15 rule-outs.

---

## §9 — Update policy

After each commit that closes a phase:

1. Update the corresponding row of §4 (advance the phase tick; record the
   commit hash).
2. If a non-canonical envelope was classified, add a one-line summary to the
   sub-question's entry in §3.
3. If a methodology lesson emerged, append a numbered bullet to §6.
4. If a new sub-question is identified (e.g., an unexpected richer envelope
   in the source code), add a stub in §3 **before** any code is written for
   its diagnostic, and a row in §4.
5. Commit this file alongside the phase commit (or as a follow-up if size
   warrants), with `git add theory/CATALOG_TYPE_HYGIENE_PROGRAMME.md`.

Stale rows (phase tick not updated within one calendar month of the
corresponding commit) are a red flag and should be reconciled at the next
phase review.

---

**Document version**: 1.0 (2026-05-26)
**Next review**: after B1c verdict (T-EPI complete).
