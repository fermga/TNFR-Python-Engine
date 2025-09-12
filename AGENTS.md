# AGENTS.md — Guide for AI Agents

> **Goal**: orient any AI agent (LLM, refactoring assistants, doc/test tools) interacting with this repository to keep the **canonical** computational implementation of the **TNFR** paradigm (Resonant Fractal Nature Theory).

---

## 1) Repository Mission

This repo exists to **bring TNFR into code in a canonical way**. Every modification, suggestion, refactor, fix, or improvement **must prioritize fidelity to TNFR** over stylistic preferences, micro‑optimizations, or framework conventions.

**TL;DR**: *TNFR first, then code.*

---

## 2) TNFR in a nutshell (operational view)

* **TNFR** proposes that reality is not made of “things” but of **coherence**: structures that **persist in networks** because they **resonate**.
* A **node** exists when its coherence remains coupled to its environment.
* **Canonical nodal equation**:
  `∂EPI / ∂t = νf · ΔNFR(t)`
  where **EPI** is the *Primary Information Structure*, **νf** is the node’s *structural frequency* (Hz\_str), and **ΔNFR** is the *internal reorganization operator* over time.
* **Structural triad**: **frequency** (reorganization rate), **phase** (network synchrony), **form** (coherent configuration of EPI).
* **Structural operators** (13): Emission, Reception, Coherence, Dissonance, Coupling, Resonance, Silence, Expansion, Contraction, Self‑organization, Mutation, Transition, Recursivity.

  > Public docs should **avoid** the word “glyphs”; name them by their **structural function**.

For the extended explanation, see `tnfr.pdf` inside the repo.

---

## 3) Canonical invariants (do not break)

These invariants **define** TNFR canonicity and **must be preserved** by any AI agent proposing changes:

1. **EPI as coherent form**: it only changes via **structural operators**; ad‑hoc mutations are not allowed.
2. **Structural units**: **νf** expressed in **Hz\_str** (structural hertz). Do not relabel or mix units.
3. **ΔNFR semantics**: its sign and magnitude modulate the reorganization rate; do **not** reinterpret it as a classic ML “error” or “loss gradient”.
4. **Operator closure**: operator composition yields valid TNFR states; any new function must map to existing operators or be defined as one.
5. **Phase check**: no coupling is valid without explicit **phase** verification (synchrony).
6. **Node birth/collapse**: keep minimal conditions (sufficient νf, coupling, reduced ΔNFR) and collapse causes (extreme dissonance, decoupling, frequency failure).
7. **Operational fractality**: EPIs can nest without losing functional identity; avoid flattening that breaks recursivity.
8. **Controlled determinism**: simulations may be stochastic, but must be **reproducible** (seeds) and **traceable** (structural logs).
9. **Structural metrics**: expose **C(t)** (total coherence), **Si** (sense index), phase and νf in telemetry. Avoid alien metrics that dilute TNFR semantics.
10. **Domain neutrality**: the engine is **trans‑scale** and **trans‑domain**. Do not hard‑wire assumptions from a specific field into the core.

---

## 4) Formal contracts (pre/post‑conditions)

* **Coherence**: applying `coherence()` must **not** reduce `C(t)` unless a programmed dissonance test justifies it.
* **Dissonance**: `dissonance()` must **increase** `|ΔNFR|` and may trigger **bifurcation** if `∂²EPI/∂t² > τ`.
* **Resonance**: `resonance()` increases effective **coupling** (`ϕ_i ≈ ϕ_j`) and **propagates** EPI without altering its identity.
* **Self‑organization**: may create **sub‑EPIs** while preserving the global **form** (operational fractality).
* **Mutation**: phase change `θ → θ'` if `ΔEPI/Δt > ξ` (keep limits ξ configurable and tested).
* **Silence**: `silence()` freezes evolution (`νf ≈ 0`) without EPI loss.

Any new function must be declared as a **specialization** or **composition** of these operators.

---

## 5) Contribution guide for AI agents

**Before touching code**:

1. Read `tnfr.pdf` (fundamentals, operators, nodal equation).
2. Run the test suite: it must cover §3 invariants and §5 contracts.
3. Add/update **structural tests** (see template below) for each change.

**Commit template (AGENT\_COMMIT\_TEMPLATE)**:

```text
Intent: (which coherence is improved)
Operators involved: [Emission|Reception|...]
Affected invariants: [#1, #4, ...]
Key changes: (bullet list)
Expected risks/dissonances: (and how they’re contained)
Metrics: (C(t), Si, νf, phase) before/after expectations
Equivalence map: (if you renamed APIs)
```

**PR template** (*structural summary*):

```markdown
### What it reorganizes
- [ ] Increases C(t) or reduces ΔNFR where appropriate
- [ ] Preserves operator closure and operational fractality

### Evidence
- [ ] Phase/νf logs
- [ ] C(t), Si curves
- [ ] Controlled bifurcation cases

### Compatibility
- [ ] Stable or mapped API
- [ ] Reproducible seed
```

---

## 6) Examples of **acceptable** changes

* Refactoring to **make phase explicit** in couplings (improves traceability).
* Adding `sense_index()` with tests correlating Si with network stability.
* Optimizing `resonance()` preserving propagation **without identity loss** of EPI.

### **Not acceptable** changes

* Recasting `ΔNFR` as a classic ML “error gradient”.
* Replacing operators with imperative functions not mapped to the TNFR grammar.
* Flattening nested EPIs in ways that break fractality/recursivity.

---

## 7) Structural testing (minimums)

* **Monotonicity**: `coherence()` does not decrease `C(t)` (except for controlled dissonance tests).
* **Bifurcation**: `self_organization()`/`dissonance()` trigger bifurcation when `∂²EPI/∂t² > τ`.
* **Propagation**: `resonance()` increases effective connectivity (measured via phase).
* **Latency**: `silence()` keeps EPI invariant over `t + Δt`.
* **Mutation**: `mutation()` changes `θ` respecting `ξ`.

Include **multi‑scale tests** (nested EPIs) and **reproducibility** (seeds).

---

## 8) Telemetry & traces

* Export: `C(t)`, `νf`, `phase`, `Si`, `ΔNFR`.
* Log **operators applied** (type, order, parameters) and **events** (birth, bifurcation, collapse).
* Prefer human‑readable formats + JSONL for pipelines.

---

## 9) Code style & organization

* Prioritize **TNFR semantic clarity** over micro‑optimizations.
* Inline docs: describe the **structural effect** (what it reorganizes) before implementation details.
* Short modules, pure functions when possible, clear core/IO separation.
* Maintain a shared **glossary** (EPI, phase, νf, ΔNFR, Si, etc.).

---

## 10) Installation & usage

* PyPI package: `pip install tnfr`.
* Provide minimal scripts/examples: create a node, apply operators, measure C(t) and Si, simple visualization.

---

## 11) Suggested roadmap (indicative)

* [ ] Robust `sense_index()` with cross‑domain example batteries.
* [ ] Visualization for **phase** and **couplings** (dynamic graphs).
* [ ] **Experiment templates** (dissonance → bifurcation → new coherence).
* [ ] Export **structural traces** for external analysis.

---

## 12) Internal references

* Base paradigm document: `tnfr.pdf` (in the repo).
* Engine notes & examples

---

## 13) Mini‑glossary

* **EPI**: Primary Information Structure (the coherent “form”).
* **νf (Hz\_str)**: Structural frequency (reorganization rate).
* **ΔNFR**: Internal reorganization operator/gradient.
* **Phase (φ)**: relative synchrony with the network.
* **C(t)**: Total coherence (global stability).
* **Si**: Sense index (capacity to generate stable reorganization).
* **Structural operators**: functions that initiate, stabilize, couple, propagate, expand/contract, self‑organize, mutate, transition, or silence structures.

---

### Final reminder

> If a change “prettifies the code” but **weakens** TNFR fidelity, it is **not accepted**. If a change **strengthens** structural coherence and paradigm traceability, **go ahead**.

