# TNFR Website — Complete Content & Structure Brief

> Revised 2026-06-24 · aligned with TNFR v0.0.3.4 (repository state as of 2026-06-24)

- **Project**: Public website for TNFR (Resonant Fractal Nature Theory)
- **Audience**: Kaniz and the dev team building the website
- **Repo (source of truth)**: <https://github.com/fermga/TNFR-Python-Engine>
- **PyPI package**: <https://pypi.org/project/tnfr/>
- **DOI (Zenodo)**: 10.5281/zenodo.17602860
- **License**: MIT
- **Current version**: 0.0.3.4 (released 2026-06-17)

> This revision supersedes the 2026-06-21 brief. It is content-identical in
> structure, but every fact, equation, count, and status has been re-checked
> against the current repository. Section **0.1** lists exactly what changed and
> why, so nothing already designed needs to be thrown away — only corrected.

---

## 0. How to read this document

This brief is the single source of truth for the website content. It includes:

1. The non-negotiable editorial rules (tone, terminology, what we never say).
2. The complete sitemap with the purpose of every page.
3. Page-by-page content blocks with the exact text, tables, equations and assets to use.
4. The visual identity guidelines (colors, typography, components).
5. The assets inventory (what diagrams to design and where the text comes from in the repo).

The website is a **presentation of TNFR and of the material already published in
the repository**. It is not an applications showcase. If anything in this document
is unclear, **ask before inventing**. Do not paraphrase the scientific content —
copy the exact wording provided here and in the linked source files. TNFR has a
strict canonical terminology that must be preserved.

**Hosting note**: the production domain `tnfr.info` is already contracted with a
separate hosting provider (IONOS). Deployment to the production environment is
therefore out of scope for Kaniz. The deliverable is the website source code
(frontend + backend) ready to be deployed by the owner.

---

## 0.1 What changed since the 2026-06-21 brief (delta for Kaniz)

The repository moved forward between the two briefs. The website content must
reflect the **current** state. Concrete corrections (old → corrected):

1. **Home, "Verify" card.** "~2,195 tests" → **2,041 tests** (README).
2. **`/learn/tutorials`.** A flat list "`01_hello_world.py` … `10_simplified_sdk_showcase.py`"
   → **162 examples across 10 thematic subfolders** (`examples/01_foundations` …
   `examples/10_applications`).
3. **`/theory/tetrad` equations.** The `|∇φ|` formula was a duplicate of the `K_φ`
   formula → `|∇φ|` is the **1st-order** mean of wrapped neighbour differences,
   while `K_φ` is the **2nd-order** deviation from the neighbour mean (`= L_rw·φ`).
   See § 3.2.2.
4. **`/theory/grammar` U6 and `/theory/correspondence`.** "ΔΦ_s < φ ≈ 1.618 (golden
   ratio)", "adopted drift bound" → **ΔΦ_s < π/2 ≈ 1.571 is the π-derived drift bound
   (half phase-wrap).** Only **π** is a genuine structural scale; φ, γ, e are not
   structural scales. Do not present 1.618 as a derived golden-ratio constant, and do
   not reuse the φ symbol (it denotes phase).
5. **`/theory/operators` table.** ZHIR grammar role "U4a, U4b"; an "effect on |ΔNFR|
   (↑/↓/=)" column → ZHIR is a **destabilizer, so its role now includes U2**, and the
   imprecise |ΔNFR| arrow is replaced by the canonical **primary nodal channel**
   (EPI / νf / θ / ΔNFR), which is how the engine documents operators. See § 3.2.3.
6. **`/research` open programs.** 3 programs (Riemann, Navier–Stokes, Yang–Mills) →
   **6 programs** (add **P vs NP, BSD, Hodge**), with updated statuses (Riemann paused
   at T-HP; Navier–Stokes N1–N17, NS-G5 closed at the discrete level). See § 3.5.6.
7. **`/research` theory documents.** 12 files → add the new canonical docs and the 3
   new research notes. See § 3.5.3.
8. **`/software/sdk`.** TetradSnapshot + ConservationReport → also add
   **SymplecticReport** (`net.symplectic_substrate()`), plus `evolve_grammar_aware`,
   `telemetry()`, `audit_operators()`, and `nfr()`. See § 3.4.3.
9. **`/theory/nodal-equation`.** Structural triad only → add a short note on the
   **fractal-resonant node (NFR)**, now a centralized canonical concept. See § 3.2.1.

Everything else in the original brief (sitemap, visual identity, accessibility,
backend/CMS scope) remains valid.

---

## 1. Editorial rules (mandatory)

These rules are derived from the project's internal `AGENTS.md` (the canonical
specification of the theory). Violations make the website unusable.

### 1.1 Tone

- **Academic and engineering-oriented**, not marketing.
- Every claim must be anchored to either the **nodal equation**, an **operator
  contract**, an **experimentally validated result in the repository**, or
  **recorded telemetry**. Qualitative claims with no math, code or data behind
  them are forbidden.
- **No metaphysical, mystical, spiritual, cosmological, or consciousness-related
  statements.** TNFR is a mathematical framework for coherent patterns on
  graph-coupled networks. It is *not* a theory of everything, a philosophy, a
  worldview, a self-help paradigm, or a description of consciousness.
- **No slogans, no superlatives, no "revolutionary", no "unifies everything", no
  "paradigm shift".** Plain factual descriptions only.
- **No anthropomorphism** of the engine ("the system understands", "knows",
  "feels").

### 1.2 What we never say on the website

- Avoid: "TNFR proves the Riemann Hypothesis". Use instead: "TNFR provides a
  computational research framework related to the Riemann Hypothesis; the
  classical hypothesis itself remains open."
- Avoid: "TNFR solves Navier–Stokes". Use instead: "TNFR offers structural
  diagnostics for the 3D Navier–Stokes problem; global regularity remains open."
- Avoid: "TNFR explains consciousness / the universe / spirituality". Never
  mentioned.
- Avoid: "Quantum-classical unification". Use instead: "Discrete-mode and
  smooth-trajectory regimes of the same nodal equation, demonstrated within the
  framework."
- Avoid: "TNFR replaces / supersedes physics / mathematics". Use instead: "TNFR
  is a specific modeling framework with a defined scope."
- Avoid: presenting **1.618, γ, e, or φ** as derived structural constants. Only
  **π** is a genuine structural scale (the phase-wrap bound); the Φ_s confinement
  bound is π-derived (π/2 drift, π/4 per-node) and ξ_C is set by the spectral gap.
  φ, γ, e are not structural scales — say so.

### 1.3 Canonical terminology (English, must be exact)

Always use these exact names — never translate, abbreviate, or invent synonyms:

| Term | Symbol | Never write |
|------|--------|-------------|
| Nodal equation | ∂EPI/∂t = νf · ΔNFR(t) | "main equation", "TNFR equation" |
| Primary Information Structure | EPI | "node state", "configuration" |
| Structural frequency | νf (units: Hz_str) | "frequency", "Hz" |
| Nodal field response / structural pressure | ΔNFR | "gradient", "error" |
| Phase | φ or θ | — |
| Structural potential | Φ_s | — |
| Phase gradient | \|∇φ\| | — |
| Phase curvature | K_φ | — |
| Coherence length | ξ_C | — |
| Total coherence | C(t), range [0, 1] | — |
| Sense index | Si, range [0, 1+] | — |
| Structural-field tetrad | (Φ_s, \|∇φ\|, K_φ, ξ_C) | "the four constants" |
| Fractal-resonant node | NFR | "the object", "the entity" |
| Unified grammar | U1–U6 | "grammar rules", "syntax" |

**Terminology note.** The four fields above form the **structural-field tetrad** —
the minimal complete description of a network state. The **one genuine structural
scale** is **π**, which bounds the whole phase sector (both `|∇φ|` and `K_φ`). The
constants γ, e, and φ are **not** structural scales and no longer appear in the
engine; everything other than π is derived from the nodal dynamics / spectral gap
or is a free operational parameter. In particular, the Φ_s confinement bound
(per-node `|Φ_s| < π/4 ≈ 0.785`, drift `ΔΦ_s < π/2 ≈ 1.571`) is **π-derived**
(quarter / half phase-wrap). Always refer to the
"structural-field tetrad" (the four fields), not to "four constants".

### 1.4 Language policy

- **Website language: English** for all pages.
- No mixed-language paragraphs.

### 1.5 Math rendering

- All equations must render with **KaTeX** (not MathJax — faster, lighter).
- Inline math: `$...$`. Block math: `$$...$$`.
- Display Greek letters using LaTeX commands (`\varphi`, `\nu_f`, `\xi_C`), never
  Unicode in equations.

---

## 2. Sitemap (final structure)

```text
/
|-- /theory                <- Intermediate-level technical core
|   |-- /theory/nodal-equation
|   |-- /theory/tetrad         (Phi_s, |grad-phi|, K_phi, xi_C)
|   |-- /theory/operators      (13 canonical operators)
|   |-- /theory/grammar        (U1-U6)
|   `-- /theory/correspondence (characteristic field scales)
|-- /learn
|   |-- /learn/glossary
|   `-- /learn/tutorials       (linked example scripts)
|-- /software
|   |-- /software/install
|   |-- /software/quickstart
|   `-- /software/sdk
|-- /research              (DOI, OEIS, theory files, reports, open programs)
|-- /about                 (project history, editorial policy, license)
`-- /contact
```

**Header navigation**: Theory · Learn · Software · Research
**Footer navigation**: About · Contact · GitHub · PyPI · DOI · License · Citation

---

## 3. Page-by-page content

### 3.1 Home (/)

**Purpose**: in 30 seconds, the visitor must understand *what TNFR is, the formal
objects it defines,* and *what they can do next.*

**Layout**: one full-screen hero + three short content blocks below + footer.

#### Hero block

- **Title (H1)**: TNFR — Resonant Fractal Nature Theory
- **Subtitle**: A mathematical framework for modeling coherent patterns on
  graph-coupled networks through the nodal equation ∂EPI/∂t = νf · ΔNFR(t).
- **Three CTAs** (horizontal buttons): Read the theory (→ /theory) · Install the
  SDK (→ /software/install) · Browse the research (→ /research)
- **Background visual**: animated SVG/Canvas of a 20-node graph evolving smoothly
  (subtle, low-saturation). Reference behavior:
  `examples/01_foundations/` network-formation scripts in the repository.

#### Block 1 — "What TNFR defines"

Exact text (do not paraphrase):

> TNFR is a framework for describing the dynamics of coherent patterns on
> graph-coupled networks. It is defined by four formal components:
>
> 1. A **nodal equation** governing the evolution of every node:
>    $$\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$
> 2. A **closed set of 13 canonical operators** (the only mechanism by which a
>    node's state may change).
> 3. A **unified grammar (U1–U6)** that constrains valid operator sequences.
> 4. A **structural-field tetrad** (Φ_s, |∇φ|, K_φ, ξ_C) that completely
>    characterizes the state of a network.

Render this as a compact table below the text:

| Field | Order | Meaning |
|-------|-------|---------|
| Φ_s | 0th — global aggregation | Structural potential (global stability) |
| \|∇φ\| | 1st — local derivative | Phase gradient (local stress) |
| K_φ | 2nd — discrete Laplacian (`K_φ = L_rw·φ`) | Phase curvature (geometric confinement) |
| ξ_C | non-local — correlation | Coherence length (spatial correlations) |

#### Block 2 — "What you can do with it"

Render as three short side-by-side cards:

| Read | Compute | Verify |
|------|---------|--------|
| The complete theory, derived from the nodal equation and 13 operators, with KaTeX-rendered equations. | Install the Python SDK (`pip install tnfr`) and run reproducible network simulations. | Inspect **2,041 tests**, 50 benchmark scripts, and the open-source code on GitHub. |

#### Block 3 — Quick start code

Render as a syntax-highlighted Python block (mirrors `README.md` verbatim):

```python
from tnfr.sdk import TNFR

# Create, connect, evolve
net = TNFR.create(20).ring().evolve(5)
print(net.results().summary())
# -> C=0.987, Si=0.912, N=20, E=20, rho=0.105

# Inspect the four structural fields
tetrad = net.tetrad()
print(tetrad.summary())
# -> Phi_s=0.0312, |grad_phi|=0.0841, |K_phi|=0.1523, xi_C=2.3147 (N=20)
```

Below the code block, three small badges:

- Python 3.10+
- MIT License
- DOI 10.5281/zenodo.17602860

---

### 3.2 /theory — Intermediate technical core

**Purpose**: this is the most important section of the website. A reader who
finishes these five pages must be able to **explain TNFR to another scientist**
without ever opening the source code.

Each sub-page has the same structure:

1. **Lead paragraph** (3–4 sentences, what this concept is).
2. **Formal block** (definitions, equations, tables).
3. **Diagram or worked example.**
4. **Why it matters** (one paragraph linking to the rest of the framework).
5. **References** (links to the canonical source in the repo).

#### 3.2.1 /theory/nodal-equation

**Lead**:

> Every node in a TNFR network evolves under a single first-order differential
> equation. From this equation the 13 operators, the grammar rules U1–U6, and the
> structural tetrad are derived.

**Formal block**:

$$\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f(t) \cdot \Delta\mathrm{NFR}(t)$$

| Symbol | Definition | Domain / Units |
|--------|------------|----------------|
| EPI | Primary Information Structure — coherent state of the node | structural manifold |
| νf | Structural frequency — reorganization capacity | ℝ⁺ (Hz_str) |
| ΔNFR | Nodal field response — local structural pressure | ℝ |
| t | Time | ℝ |

**Structural triad** (each node carries three irreducible attributes):

| Attribute | Symbol | Meaning |
|-----------|--------|---------|
| Form | EPI | Coherent configuration; modified only through canonical operators |
| Frequency | νf | Reorganization rate; νf → 0 means the node deactivates |
| Phase | φ (or θ) | Synchronization parameter in [0, 2π); coupling requires \|φᵢ − φⱼ\| ≤ Δφ_max |

**The fractal-resonant node (NFR)**: the node carrying this triad is a
*fractal-resonant node* — canonically, a **region of structural coherence coupled
to a network**. It is multiscalar (an NFR can nest other NFRs), autopoietic
(emerges by local reorganization), relational (exists only by coupling), and
temporal (persists while it reorganizes). Its internal **nodal topology** is read
from the emergent structural-potential geometry as *radial* (one central nucleus),
*annular* (passive center, peripheral ring), or *multinodal* (several centers).

**Stability criterion**: integrating the nodal equation,

$$\mathrm{EPI}(t_f) = \mathrm{EPI}(t_0) + \int_{t_0}^{t_f} \nu_f(\tau)\,\Delta\mathrm{NFR}(\tau)\,d\tau$$

Coherence is preserved only when the integral converges:

$$\int_{t_0}^{t_f} \nu_f(\tau)\,\Delta\mathrm{NFR}(\tau)\,d\tau < \infty$$

This convergence requirement is the formal basis of grammar rule **U2**
(Convergence and Boundedness).

**Diagram (to be designed by Kaniz)**: a clean schematic showing a single node
with its three irreducible attributes (EPI, νf, φ) and an arrow labelled
∂EPI/∂t = νf · ΔNFR(t) pointing to its updated state at time t + dt.

**References**:

- Source: `theory/FUNDAMENTAL_THEORY.md` § 2; NFR definition in `AGENTS.md` § 2
- Implementation: `src/tnfr/operators/nodal_equation.py`;
  `src/tnfr/physics/fields.py` (`classify_nodal_topology`)

#### 3.2.2 /theory/tetrad

**Lead**:

> The state of any TNFR network is characterized by four scalar fields. Each one
> answers a different structural question, and together they form the minimal
> complete description of a coherent system on a graph, as derived in
> `theory/MINIMAL_STRUCTURAL_DEGREES.md`.

**The four structural questions** (render as a 4-row table):

| Question | Field | Order |
|----------|-------|-------|
| How much pressure accumulates from the network? | Φ_s (structural potential) | 0th — global aggregation |
| How misaligned am I with my neighbours? | \|∇φ\| (phase gradient) | 1st — local derivative |
| How sharply does alignment change direction? | K_φ (phase curvature) | 2nd — discrete Laplacian |
| How far does my state correlate across the system? | ξ_C (coherence length) | non-local — correlation range |

**Equations** (four KaTeX blocks — corrected to the canonical definitions in
`docs/STRUCTURAL_FIELDS_TETRAD.md`):

$$\Phi_s(i) = \sum_{j \neq i} \frac{\Delta\mathrm{NFR}_j}{d(i,j)^2}$$

$$|\nabla\varphi|(i) = \operatorname*{mean}_{j \in \mathcal{N}(i)} \big| \operatorname{wrap}(\varphi_j - \varphi_i) \big|$$

$$K_\varphi(i) = \varphi_i - \frac{1}{\deg(i)} \sum_{j \in \mathcal{N}(i)} \varphi_j \qquad (= L_{\mathrm{rw}}\,\varphi \text{ in the smooth limit})$$

$$C(r) \sim \exp(-r / \xi_C)$$

> **Note for Kaniz (correction):** in the previous brief `|∇φ|` and `K_φ` were
> printed with the *same* formula. They are different orders of the derivative
> tower: `|∇φ|` is the mean of the *absolute wrapped neighbour differences*
> (1st order); `K_φ` is the *signed deviation from the neighbour mean* (2nd
> order, the discrete Laplacian applied to phase).

**Canonical thresholds table**:

| Field | Threshold | Source |
|-------|-----------|--------|
| Φ_s | per-node \|Φ_s\| < π/4 ≈ 0.785; drift ΔΦ_s < π/2 ≈ 1.571 | **π-derived** (quarter / half phase-wrap) |
| \|∇φ\| | \|∇φ\| ≤ π (phase wrap) | geometric bound; sync onset ≈ 0.29 (σ-dependent) |
| K_φ | \|K_φ\| < 0.9π ≈ 2.8274 | 90% of the geometric bound \|K_φ\| ≤ π |
| ξ_C | ξ_C > diameter ⇒ critical | finite-size scaling; ξ_C ∝ 1/√λ₂ (spectral gap) |

**Diagram (to be designed by Kaniz)**: a tetrahedron whose four vertices/edges
carry the four structural fields (Φ_s, |∇φ|, K_φ, ξ_C) and their order
(0th / 1st / 2nd / non-local). π may be shown as the phase scale on the |∇φ| and
K_φ edges. Static SVG; rotatable Three.js version optional.

**Why it matters**:

> The four classes — global aggregation, first derivative, second derivative, and
> correlation range — exhaust the independent structural information available
> from a scalar phase field coupled to a scalar source on a graph. The minimality
> argument is given in detail in `theory/MINIMAL_STRUCTURAL_DEGREES.md`.

**References**:

- Source: `theory/MINIMAL_STRUCTURAL_DEGREES.md`,
  `docs/STRUCTURAL_FIELDS_TETRAD.md`
- Implementation: `src/tnfr/physics/fields.py`

#### 3.2.3 /theory/operators

**Lead**:

> All structural changes in TNFR occur through exactly 13 canonical operators.
> Direct mutation of EPI, νf, or φ outside this operator algebra is not permitted
> by the framework. The constraint is derived from the nodal equation, not from a
> coding convention. Each operator acts on exactly **one nodal channel** — the
> form EPI, the capacity νf, the phase θ, or the pressure ΔNFR — at node or
> network scale.

**The 13 operators** (render as a responsive card grid, 3 columns desktop /
1 column mobile). For each operator, the card must show: operator code (large),
English name, one-line description, **primary nodal channel** (EPI / νf / θ /
ΔNFR), and grammar role tag.

| # | Code | Name | One-line description | Primary channel | Grammar role |
|---|------|------|----------------------|-----------------|--------------|
| 1 | AL | Emission | Creates EPI from a null state; raises νf | EPI (form) | U1a (Generator) |
| 2 | EN | Reception | Captures and integrates incoming structural input | EPI (form) | — |
| 3 | IL | Coherence | Stabilizes form through negative feedback on ΔNFR | ΔNFR (pressure) | U2 (Stabilizer) |
| 4 | OZ | Dissonance | Introduces controlled instability | ΔNFR (pressure) | U2 (Destabilizer), U4a, U1b |
| 5 | UM | Coupling | Creates a structural link via phase synchronization | θ (phase) | U3 |
| 6 | RA | Resonance | Amplifies and propagates patterns coherently | EPI (form) | U3 |
| 7 | SHA | Silence | Freezes evolution temporarily (νf → 0) | νf (capacity) | U1b (Closure) |
| 8 | VAL | Expansion | Increases structural complexity | νf (capacity) | U2 (Destabilizer) |
| 9 | NUL | Contraction | Reduces structural complexity | νf (capacity) | — |
| 10 | THOL | Self-organization | Creates sub-EPIs while preserving global form | ΔNFR (pressure) | U2 (Stabilizer), U4a, U4b |
| 11 | ZHIR | Mutation | Phase transformation at threshold | θ (phase) | U2 (Destabilizer), U4a, U4b |
| 12 | NAV | Transition | Regime shift; activates latent EPI | ΔNFR (pressure) | U1a, U1b |
| 13 | REMESH | Recursivity | Couples EPI(t) with EPI(t − τ) across scales | EPI (form) | U1a, U1b |

> **Scale note**: **REMESH (Recursivity)** is the only operator that acts at
> **NETWORK** scale (it implements operational fractality, grammar U5); the other
> twelve act at **NODE** scale.
>
> **Correction vs the previous brief**: ZHIR is a destabilizer, so its grammar
> role now includes **U2**. The earlier "effect on |ΔNFR| (↑/↓/=)" column has been
> replaced by the canonical **primary nodal channel**, which is how the engine
> itself classifies operators (`src/tnfr/operators/operator_contracts.py`).

**Canonical classification** (used to group operator cards by functional class):

- **Generators (U1a)**: AL, NAV, REMESH
- **Closures (U1b)**: SHA, NAV, REMESH, OZ
- **Stabilizers (U2)**: IL, THOL
- **Destabilizers (U2)**: OZ, ZHIR, VAL
- **Coupling (U3)**: UM, RA
- **Transformers (U4b)**: ZHIR, THOL

**Composition block** (below the card grid):

Operators compose into named **fragments (macros)** that implement typical
workflows. A fragment is *not* a standalone valid word — it becomes valid by
adding grammar glue (a U1a generator prefix and a U1b closure suffix):

| Fragment | Composition | Use case |
|----------|-------------|----------|
| Bootstrap | [Emission, Coupling, Coherence] = [AL, UM, IL] | Initialize a new network |
| Stabilize | [Coherence, Silence] = [IL, SHA] | Consolidate after changes |
| Explore | [Dissonance, Mutation, Coherence] = [OZ, ZHIR, IL] | Move past a local optimum |
| Propagate | [Resonance, Coupling] = [RA, UM] | Spread coherence across the network |

**Diagram (to be designed by Kaniz)**: a single visual catalogue of the 13
operator codes grouped by functional class (Generators / Stabilizers /
Destabilizers / Coupling / Transformers / Closure), using a consistent
iconographic style.

**References**:

- Source: `theory/STRUCTURAL_OPERATORS.md`
- Implementation: `src/tnfr/operators/definitions.py`,
  `src/tnfr/operators/operator_contracts.py`

#### 3.2.4 /theory/grammar

**Lead**:

> Operator sequences must satisfy six grammar rules (U1–U6). Each rule is derived
> from a specific property of the nodal equation. Sequences that violate them
> produce unbounded or fragmented dynamics within the framework.

**The six rules** (render as an accordion or expandable list — one section per
rule):

**U1 — Structural initiation and closure**

- U1a: If EPI = 0, the sequence must start with a generator (AL, NAV, or REMESH).
  Formal basis: the nodal equation is undefined at EPI = 0.
- U1b: Every sequence must end with a closure operator (SHA, NAV, REMESH, or OZ).
  Formal basis: sequences need a defined endpoint that leaves the system in a
  coherent attractor.

**U2 — Convergence and boundedness**

- If the sequence contains a destabilizer (OZ, ZHIR, VAL), it must also contain a
  stabilizer (IL, THOL).
- Formal basis: the integral $\int \nu_f \cdot \Delta\mathrm{NFR}\, dt$ must
  converge; without negative feedback it diverges and coherence is lost.

**U3 — Resonant coupling**

- Coupling operators (UM, RA) require phase compatibility: \|φᵢ − φⱼ\| ≤ Δφ_max.
- Formal basis: antiphase produces destructive interference.

**U4 — Bifurcation dynamics**

- U4a: Triggers (OZ, ZHIR) need handlers (THOL, IL).
- U4b: Transformers (ZHIR, THOL) need a recent destabilizer in context. ZHIR
  additionally requires a prior IL (stable base).

**U5 — Multi-scale coherence**

- Nested EPIs (hierarchical structures) require stabilizers at every scale.
- Formal basis: parent coherence depends on the aggregated child reorganization
  (`C_parent ≥ α · Σ C_child`).

**U6 — Structural potential confinement**

- Telemetry-based safety check: monitor **ΔΦ_s < π/2 ≈ 1.571**, where
  Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)². This is a read-only check, not a sequence
  constraint.
- Formal basis: the emergent field Φ_s must remain bounded for coherence to
  survive. **π/2 ≈ 1.571 is the π-derived drift bound** (half phase-wrap, tied to
  the one genuine structural scale; not a golden-ratio constant).

**Example block** (show side by side, valid vs invalid):

- **Valid sequence**: `[AL, UM, IL, OZ, IL, SHA]` — starts with a generator
  (U1a OK), contains stabilizers for the destabilizer (U2 OK), ends with closure
  (U1b OK).
- **Invalid sequence**: `[OZ, VAL, OZ]` — starts with a destabilizer with no prior
  generator (U1a FAIL), no stabilizer present (U2 FAIL), ends in OZ (closure OK
  but the sequence is still invalid on U1a/U2).

**Diagram (to be designed by Kaniz)**: a state-machine-style diagram showing the
six rules with arrows between operator classes (generators → stabilizers →
closure, etc.).

**References**:

- Source: `theory/UNIFIED_GRAMMAR_RULES.md`
- Implementation: `src/tnfr/operators/grammar.py`,
  `src/tnfr/operators/grammar_canon.py`

#### 3.2.5 /theory/correspondence

**Lead**:

> Each structural field has a characteristic scale that sets or bounds its
> behavior. The two phase fields, |∇φ| and K_φ, share a single geometric scale, π
> (the phase-wrap bound). The potential field Φ_s has an empirical confinement
> bound, and the coherence length ξ_C is set by the network's spectral gap λ₂.

**Field scales** (render as four cards):

| Field | Characteristic scale | Constraint |
|-------|----------------------|------------|
| Φ_s | π-derived confinement (quarter / half phase-wrap) | per-node \|Φ_s\| < π/4 ≈ 0.785; drift ΔΦ_s < π/2 ≈ 1.571 |
| \|∇φ\| | phase-wrap (π) | \|∇φ\| ≤ π; synchronization onset ≈ 0.29 |
| K_φ | phase-wrap (π) | \|K_φ\| < 0.9π ≈ 2.827; K_φ = L_rw·φ |
| ξ_C | spectral gap | ξ_C ∝ 1/√λ₂; ξ_C > diameter ⇒ critical |

**Closing paragraph**:

> **π is the one genuine structural scale of the tetrad**: it scales the whole
> phase sector, bounding both |∇φ| and K_φ. The Φ_s confinement bound is π-derived
> (π/4 per-node, π/2 drift), and ξ_C is set by the spectral gap λ₂. The constants
> γ, e, and φ are **not** structural scales and no longer appear in the engine;
> everything other than π is derived from the nodal dynamics or is a free
> operational parameter. The full map of constants and bounds is documented in
> `src/tnfr/constants/canonical.py`.

**References**:

- Source: `theory/FUNDAMENTAL_THEORY.md` § 4
- Implementation: `src/tnfr/constants/canonical.py`

---

### 3.3 /learn

#### 3.3.1 /learn/glossary

- Render the full content of `theory/GLOSSARY.md` with a top search box.
- Each glossary entry must be linkable by anchor (e.g. `/learn/glossary#EPI`).
- Bonus (if budget allows): tooltips on hover for any canonical term used anywhere
  on the website.

#### 3.3.2 /learn/tutorials

- List the example scripts from `examples/`. The current repository ships **162
  examples organized in 10 thematic subfolders** — present them grouped by folder,
  each entry with title, one-line description, and a "View on GitHub" link. Use
  `examples/README.md` as the index/source.
  - `01_foundations` — nodal equation, operators, network formation
  - `02_physics_regimes` — transport, diffusion, discrete-mode/smooth-trajectory
  - `03_riemann_zeta` — TNFR–Riemann ζ track
  - `04_riemann_L_twisted` — χ-twisted L-function track
  - `05_type_hygiene` — catalog type-hygiene programme
  - `06_navier_stokes` — K_φ cascade, Taylor–Green
  - `07_number_theory` — primality, cyclotomy, prime families
  - `08_emergent_geometry` — symplectic substrate, conservation, grammar geometry
  - `09_millennium` — Millennium-problem reformulations
  - `10_applications` — applied/SDK showcases
- Optional (out of initial scope): embed each example as a runnable Pyodide
  notebook.

---

### 3.4 /software

#### 3.4.1 /software/install

```bash
pip install tnfr                       # stable release
pip install -e ".[dev-minimal]"        # development
pip install -e ".[test-all]"           # full test suite
pip install -e ".[compute-jax]"        # JAX backend
pip install -e ".[compute-torch]"      # PyTorch backend
```

Requirements: Python 3.10+, Linux/macOS/Windows.

#### 3.4.2 /software/quickstart

Mirror the `README.md` "Quick Start" block (already in this brief, § 3.1, Block 3).

#### 3.4.3 /software/sdk

Render the dataclass reference for the Simple SDK (`src/tnfr/sdk/simple.py`):

**Builder / evolution**

- `TNFR.create(n)`, `.ring()`, `.evolve(steps)`
- `.evolve_grammar_aware(steps)` — proactive U1–U6 enforcement during evolution
- `net.results().summary()` → `C`, `Si`, `N`, `E`, `rho`
- `net.telemetry()` → C(t), Si, phase_sync, tetrad
- `net.audit_operators()` → 13/13 operator-contract audit
- `net.nfr()` → whole-NFR read-out (radial / annular / multinodal topology)

**Reports (dataclasses)**

- `TetradSnapshot`: fields `phi_s`, `grad_phi`, `k_phi`, `xi_c`, `j_phi`,
  `j_dnfr`; methods `is_safe()`, `summary()`
- `ConservationReport`: fields `noether_charge`, `energy`, `lyapunov_stable`,
  `lyapunov_derivative`, `conservation_quality`; method `summary()`
- `SymplecticReport`: fields `phase_space_dimension`, `hamiltonian`,
  `background_potential`, `liouville_divergence`, `is_valid_manifold`; method
  `summary()`
- `TNFR.analyze(net)` → comprehensive dict (coherence, tetrad, conservation,
  tensor_invariants, emergent_fields, integrity, features)

Reference: `src/tnfr/sdk/simple.py`

---

### 3.5 /research

**Purpose**: a single hub that catalogues everything that has been **published** as
part of the TNFR project — citation metadata, theory documents, generated reports,
companion labs, and the explicit status of open research programs.

The page should be organized as the six numbered sub-sections below, separated by
visible dividers. Each sub-section gets its own anchor (`/research#citation`,
`/research#theory`, etc.).

#### 3.5.1 Citation

Citation block (BibTeX, render in a code block with a copy-to-clipboard button):

```bibtex
@software{tnfr_python_engine,
  author  = {Martinez Gamo, F. F.},
  orcid   = {0009-0007-6116-0613},
  title   = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year    = {2026},
  version = {0.0.3.4},
  doi     = {10.5281/zenodo.17602860},
  url     = {https://github.com/fermga/TNFR-Python-Engine},
  license = {MIT}
}
```

DOI: 10.5281/zenodo.17602860 → <https://doi.org/10.5281/zenodo.17602860> ·
License: MIT.

> **Important note for Kaniz**: do **not** import the `abstract` field from the
> repository's `CITATION.cff` verbatim. That field still contains "paradigm shift"
> phrasing forbidden by § 1.1. When a short project description is needed (Open
> Graph metadata, search-engine snippet, page `<meta>` description), use the
> wording from § 3.1 Block 1 of this brief instead.

#### 3.5.2 OEIS sequences

No TNFR-original OEIS sequences are registered yet — keep this sub-section
**hidden** until official identifiers are provided. (Note: some examples in
`examples/07_number_theory/` *reproduce* known OEIS sequences such as A005384
(Sophie Germain primes) and A074816 as validation; these are illustrative checks,
not new submissions, and should not be presented as registered TNFR sequences.)

#### 3.5.3 Theory documents

Linked list of the canonical theory files in the repository. Each item: title,
one-line summary, link to GitHub.

Core canon:

- `theory/FUNDAMENTAL_THEORY.md` — nodal equation, structural-field tetrad, field scales.
- `theory/STRUCTURAL_OPERATORS.md` — the 13 canonical operators, contracts, composition.
- `theory/UNIFIED_GRAMMAR_RULES.md` — rules U1–U6 with full derivations.
- `theory/MINIMAL_STRUCTURAL_DEGREES.md` — minimality proof of the tetrad.
- `theory/STRUCTURAL_CONSERVATION_THEOREM.md` — Noether-like conservation derivation.
- `theory/TNFR_VARIATIONAL_PRINCIPLE.md` — Lagrangian/Hamiltonian formulation.
- `theory/GLOSSARY.md` — canonical terminology.

Emergent geometry & extended structure (new since the previous brief):

- `theory/EMERGENT_ONTOLOGY.md` — how patterns/objects emerge from the dynamics.
- `theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md` — the six downstream emergent fields.
- `theory/GAUGE_SYMMETRY_AND_UNIFICATION.md` — U(1)/U(2) gauge & polarization structure.
- `theory/PHYSICAL_REGIME_CORRESPONDENCES.md` — discrete-mode vs smooth-trajectory regimes.
- `theory/MATHEMATICAL_DYNAMICS_BASIS.md` — formal dynamics basis.
- `theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md` — stability analysis.
- `theory/REMESH_INFINITY_DERIVATION.md` — N15 REMESH-∞ closure (catalog-completeness theorem).

Research programs:

- `theory/TNFR_NUMBER_THEORY.md` — primality and factorization in the TNFR formulation.
- `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` — TNFR–Riemann program.
- `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` — TNFR–Navier–Stokes program.
- `theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md` — TNFR–Yang–Mills program.
- `theory/TNFR_P_VS_NP_RESEARCH_NOTES.md` — TNFR–P vs NP program.
- `theory/TNFR_BSD_RESEARCH_NOTES.md` — TNFR–Birch–Swinnerton-Dyer program.
- `theory/TNFR_HODGE_RESEARCH_NOTES.md` — TNFR–Hodge program.

#### 3.5.4 Generated reports

The repository's build tasks generate several HTML/PNG reports from the engine.
Embed each as a card with: title, one-line description, generation command
(`./make.cmd <task>`), link to the generated artefact. (These tasks are defined in
`.vscode/tasks.json` and run through `./make.cmd`.)

- **Atom Atlas** — atomic-scale TNFR signatures. Task: `./make.cmd report-particle-atlas-u6`.
- **Molecule Atlas** — molecular configurations under phase-coupled networks. Task: `./make.cmd molecule-atlas-script`.
- **Periodic Table Atlas** — periodic-table-scale TNFR diagnostics. Task: `./make.cmd report-periodic-table-classic`.
- **Operator Completeness** — exhaustive operator-coverage report. Task: `./make.cmd report-operator-completeness`.
- **Interaction Sequences** — illustrative operator sequences. Task: `./make.cmd report-interaction-sequences`.
- **Emergent Particles** — emergent-pattern detection report. Task: `./make.cmd report-emergent-particles`.
- **Fundamental Particles Atlas** — particle-scale TNFR diagnostics. Task: `./make.cmd report-fundamental-particles`.

> These reports must be treated as **illustrative outputs of the framework**, not
> as claims about chemistry or physics beyond what each report explicitly
> demonstrates. Each card must show the originating `./make.cmd` task so any reader
> can regenerate it.

#### 3.5.5 Companion projects

- `primality-test/` — primality experiments under the TNFR formulation
  (primality as ΔNFR = 0).
- `factorization-lab/` — factorization experiments based on spectral decomposition
  on prime-path graphs.

Render as two cards with a one-line description and a "View on GitHub" link.

#### 3.5.6 Open research programs (mandatory honesty section)

This sub-section must explicitly state the open status of each program. There are
now **six** programs. Use the wording below.

**TNFR–Riemann program**

> The TNFR–Riemann program is a computational research framework relating discrete
> prime-path graph operators to the Riemann Hypothesis. The critical-parameter
> convergence σ_c → 1/2 has been numerically verified within the framework. The
> classical Riemann Hypothesis itself remains open. The work in this repository
> contributes structural diagnostics and machinery, not a proof of RH.
>
> Current status: milestones P1–P49 implemented; the full ζ↔L attack surface is
> shipped. The bridge to RH is the open conjecture T-HP (gap G4), currently
> **paused** at the oscillatory residue $S(T) = (1/\pi)\,\arg\zeta(\tfrac{1}{2}+iT)$.
> Link: `theory/TNFR_RIEMANN_RESEARCH_NOTES.md`

**TNFR–Navier–Stokes program**

> Structural diagnostics for the 3D Navier–Stokes problem via the K_φ cascade and
> Taylor–Green vortex experiments. Milestones N1–N17 implemented; the NS-G5 gap is
> closed at the discrete-operator level. Global regularity of 3D Navier–Stokes
> (the continuum / Clay problem, gaps NS-G1..G4) remains open. TNFR provides
> measurement and diagnostic infrastructure, not a resolution. Link:
> `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md`

**TNFR–Yang–Mills program**

> Exploratory structural-gap diagnostics related to the Yang–Mills mass-gap
> problem (Y1–Y5; finite U(1) diagnostics). The non-Abelian mass gap remains open
> (Branch B). No claims of resolution. Link:
> `theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md`

**TNFR–P vs NP program**

> Structural synthesis-vs-verification programme (PNP-1): coherence verification
> is O(\|E\|), whereas coherent synthesis exhibits trapping. The worst-case
> separation remains open (Branch B). Not a proof. Link:
> `theory/TNFR_P_VS_NP_RESEARCH_NOTES.md`

**TNFR–BSD program**

> Structural-pressure programme for Birch–Swinnerton-Dyer (BSD-1): rank separation
> via structural-pressure accumulation. The link from rank to order of vanishing
> remains open (Branch B). Not a proof. Link: `theory/TNFR_BSD_RESEARCH_NOTES.md`

**TNFR–Hodge program**

> Discrete cochain programme (HC-1): the discrete Hodge decomposition equals
> homology exactly (Eckmann). The (p,p) bigrading and algebraicity are
> structurally blind in the current formulation (Branch B3-leaning, a strong
> negative result). Not a proof. Link: `theory/TNFR_HODGE_RESEARCH_NOTES.md`

**Closing status table** (render at the end of this sub-section):

| Program | Latest milestone | Status | Notes |
|---------|------------------|--------|-------|
| Riemann | P1–P49 (ζ↔L attack surface) | OPEN — G4 = RH not closed; **paused at T-HP** | σ_c → 1/2 verified |
| Navier–Stokes | N1–N17 (NS-G5 closed at discrete level) | OPEN — global regularity / Clay not resolved | K_φ cascade diagnostics |
| Yang–Mills | Y1–Y5 (finite U(1) diagnostics) | OPEN — non-Abelian mass gap (Branch B) | research notes |
| P vs NP | PNP-1 | OPEN — worst-case separation (Branch B) | verification O(\|E\|) vs synthesis trapping |
| BSD | BSD-1 | OPEN — rank ↔ order of vanishing (Branch B) | structural-pressure accumulation |
| Hodge | HC-1 | OPEN — structurally blind (Branch B3-leaning) | discrete Hodge = homology (Eckmann) |

---

### 3.6 /about

- Brief project history (one paragraph — will be provided before launch).
- Editorial policy: summarized version of section 1 of this brief.
- License: MIT.

### 3.7 /contact

- Email contact form (with reCAPTCHA or hCaptcha).
- Direct links: GitHub Issues, GitHub Discussions.
- Note: "For research collaborations, please include affiliation and a short
  description of the proposed work."

---

## 4. Visual identity

### 4.1 Color palette

Sober, scientific, low-saturation. Suggested:

- **Primary** (deep blue): `#0B2545`
- **Accent** (resonant amber): `#E8A33D`
- **Neutral background**: `#FAFAF7`
- **Text**: `#1A1A1A`
- **Muted**: `#6B7280`
- **Success / coherence**: `#1F7A4D`
- **Warning / dissonance**: `#C45A3B`

### 4.2 Typography

- **Headings**: a modern serif (e.g. Source Serif Pro, Lora, or Newsreader).
- **Body**: a high-readability sans-serif (e.g. Inter, IBM Plex Sans).
- **Code / monospace**: JetBrains Mono or IBM Plex Mono.
- **Math**: KaTeX default (Computer Modern).

### 4.3 UI components (minimum set)

- Navigation bar (sticky, with logo + 4 main sections + GitHub icon).
- Footer (4 columns: About, Resources, Community, Legal).
- Equation block (KaTeX rendered, with copy-to-clipboard).
- Code block (syntax highlighted, with copy-to-clipboard).
- Reference block (gray background, "Sources" header, bulleted links).
- Operator card (used in /theory/operators).
- Accordion (used in /theory/grammar).
- Searchable glossary entries.
- Report card (used in /research#reports).
- DOI / version / license badges.

### 4.4 Responsiveness

- Mobile-first.
- Breakpoints: 640 / 768 / 1024 / 1280 px.
- All tables must collapse gracefully on mobile (consider horizontal scroll for
  the larger ones).
- Equations must scroll horizontally on narrow viewports rather than overflow.

### 4.5 Accessibility (mandatory)

- WCAG 2.1 AA contrast ratios.
- All figures must have descriptive alt text (provided in section 5.1).
- Keyboard-navigable accordion and search components.
- KaTeX accessible output enabled.

---

## 5. Assets inventory

### 5.1 Diagrams to be designed by Kaniz

All diagrams are part of the design scope. No external image library is supplied —
Kaniz must produce them in a consistent visual style (line art, two-color palette
derived from § 4.1).

| Diagram | Used on page | Brief | Suggested alt text |
|---------|--------------|-------|--------------------|
| Single node + nodal equation | /theory/nodal-equation | A node showing its three irreducible attributes (EPI, νf, φ) and an arrow labelled ∂EPI/∂t = νf · ΔNFR(t) to its next state. | "Single TNFR node evolving under the nodal equation." |
| Structural-field tetrad | /theory/tetrad, /theory/correspondence | A tetrahedron whose vertices/edges carry the four structural fields (Φ_s, \|∇φ\|, K_φ, ξ_C) and their order. π may appear as the phase scale on the \|∇φ\|/K_φ edges. Static SVG; rotatable Three.js optional. | "The four structural fields of TNFR (the minimal tetrad)." |
| Operator catalogue | /theory/operators | A single panel showing the 13 operator codes (AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH), grouped by functional class. | "The 13 canonical TNFR operators grouped by functional class." |
| Grammar state machine | /theory/grammar | Diagram of allowed transitions between operator classes implementing U1–U6 (generator → stabilizer → closure, etc.). | "Grammar U1–U6 represented as a state machine over operator classes." |

### 5.2 Text sources (Markdown, from the repo)

| Page | Source file |
|------|-------------|
| /theory/nodal-equation | this brief § 3.2.1 + `theory/FUNDAMENTAL_THEORY.md` § 2 |
| /theory/tetrad | this brief § 3.2.2 + `theory/MINIMAL_STRUCTURAL_DEGREES.md` + `docs/STRUCTURAL_FIELDS_TETRAD.md` |
| /theory/operators | this brief § 3.2.3 + `theory/STRUCTURAL_OPERATORS.md` |
| /theory/grammar | this brief § 3.2.4 + `theory/UNIFIED_GRAMMAR_RULES.md` |
| /theory/correspondence | this brief § 3.2.5 + `theory/FUNDAMENTAL_THEORY.md` § 4 |
| /learn/glossary | `theory/GLOSSARY.md` |
| /research (citation / DOI) | `CITATION.cff`, `README.md` |
| /research (theory documents) | all files in `theory/` listed in § 3.5.3 |
| /research (open programs) | the six `theory/TNFR_*_RESEARCH_NOTES.md` files |

### 5.3 External links (always opened in a new tab)

- GitHub: <https://github.com/fermga/TNFR-Python-Engine>
- PyPI: <https://pypi.org/project/tnfr/>
- DOI: <https://doi.org/10.5281/zenodo.17602860>

---

## 6. Backend & CMS scope (database integration)

Since the proposal includes a Next.js / Express backend with MongoDB (running on
Node.js), the **minimum useful schema** is:

### 6.1 Collections

- `posts`: published research updates and changelog entries. Fields: `slug`,
  `title`, `body_mdx`, `published_at`, `tags[]`, `summary`.
- `references`: bibliographic entries for /research. Fields: `id`, `type`
  (paper/zenodo/oeis/software), `title`, `authors[]`, `year`, `url`, `doi`,
  `bibtex`.
- `glossary_overrides`: optional manual overrides for glossary tooltips. Fields:
  `term`, `short_definition`, `link`.

### 6.2 Admin

A protected `/admin` route (basic auth or simple JWT) for:

- Creating / editing / publishing posts.
- Adding / editing references.
- Triggering a re-fetch of glossary terms from GitHub (the canonical source).

### 6.3 Out of scope (keep it simple)

- Out: user accounts, comments, forums (use GitHub Discussions instead).
- Out: newsletter signup (can be added later as a separate feature).
- Out: Google Analytics or Facebook Pixel. Use privacy-respecting analytics if any
  (Plausible or Umami suggested).
- Out: production hosting and DNS configuration. The domain `tnfr.info` is already
  contracted with IONOS and will be deployed by the owner; the deliverable from
  Kaniz is the source code (frontend + backend) ready to be deployed.

---

*End of brief.*
