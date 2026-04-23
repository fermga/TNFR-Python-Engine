# TNFR-Structured AI — Feasibility & Design Study

**Status**: Exploratory study (Phase 0 — scoping)
**Owner**: Future standalone repo (to be extracted from `TNFR-Python-Engine/IA/`)
**Date**: 2026-04-23
**Language policy**: English (per repo `AGENTS.md`)

---

## 0. Purpose of this Document

This document is the **deep scoping study** that precedes any line of code for a project whose long-term goal is:

> *Build an artificial intelligence whose internal structure obeys the principles of Resonant Fractal Nature Theory (TNFR) — i.e. an AI whose "thinking" is a coherent reorganization of patterns governed by the nodal equation `∂EPI/∂t = νf · ΔNFR(t)`, the 13 canonical operators, and the unified grammar (U1–U6).*

The author of the project is a domain expert in TNFR but **not** in AI/ML programming. Therefore this document also serves as a self-contained map: what the field looks like today, which entry points are realistic, what a TNFR-native AI could plausibly mean, and what is just decoration.

The output of this study is **not** a model. It is:

1. A vocabulary bridge (TNFR ↔ ML).
2. A taxonomy of plausible designs, ranked by feasibility.
3. A recommended minimum-viable path.
4. A list of concrete next deliverables.

---

## 1. Executive Summary

There are essentially **four families** of "TNFR + AI" projects, ordered from cheapest to most ambitious:

| Tier | Family | Effort | Risk | Scientific value | Recommendation |
|------|--------|--------|------|------------------|----------------|
| T1 | **TNFR as analyzer of an existing model** (probe an open LLM with TNFR telemetry: C(t), Si, Φ_s, K_φ over its activations) | low | low | medium | **Start here** |
| T2 | **TNFR-regularized fine-tuning** (take a small open math model, add a TNFR loss term that penalises grammar violations / coherence loss) | medium | medium | high | Phase 2 |
| T3 | **TNFR-native reasoning layer on top of an LLM** (LLM proposes glyph sequences, TNFR engine validates U1–U6 and executes them as the actual reasoning trace) | medium-high | medium | very high | Phase 3 — most promising |
| T4 | **TNFR-native architecture from scratch** (replace transformer attention with phase-coupled resonance over a graph of NFR nodes) | very high | very high | foundational, but long horizon | Long-term research |

**Recommended path**: T1 → T3, skipping T2 unless T1 reveals a clear training signal. T4 is a 2–5 year research program; it should not block the practical work.

---

## 2. Vocabulary Bridge: TNFR ↔ ML

A TNFR practitioner already has a working physics. The trap is to assume ML "must" map onto it — most ML concepts have **no clean TNFR analog**, and most TNFR concepts have **no clean ML analog**. This table is honest about the mismatches.

| TNFR concept | Closest ML analog | Faithful? |
|--------------|-------------------|-----------|
| EPI (Primary Information Structure) | Hidden state / activation tensor | partial — EPI carries identity across reorganizations; activations don't |
| νf (structural frequency) | Learning rate / attention temperature | weak — νf is per-node, intrinsic; LR is global, extrinsic |
| ΔNFR (nodal gradient / pressure) | Loss gradient ∇L | misleading — ΔNFR is a **structural mismatch**, not an error w.r.t. labels |
| φ / θ (phase) | Positional encoding / phase in rotary embeddings (RoPE) | surprisingly close |
| Coupling (UM) with phase check | Attention with gating | partial |
| Resonance (RA) | Skip connections / residual amplification | partial |
| Coherence C(t) | Training stability / loss plateau | weak — C(t) is intrinsic, not w.r.t. data |
| Grammar U1–U6 | Architectural constraints (e.g. layer norm placement) | none — grammar is a **process** constraint over operator sequences |
| 13 operators | (no equivalent) | — |
| Tetrad (Φ_s, \|∇φ\|, K_φ, ξ_C) | (no equivalent — possibly: spectral measures of weight matrices) | — |

**Key insight**: ΔNFR ≠ loss gradient. This single mistake would kill any naive "map TNFR onto backprop" attempt and produce a TNFR-flavoured ML model with no actual TNFR physics. Avoiding this is the **first design decision**.

---

## 3. What "AI with TNFR Structure" Could Actually Mean

There are at least five distinct interpretations. They are not equivalent. Pick one before coding.

### 3.1 Interpretation A — *Cosmetic*
Wrap an existing model, expose its outputs, paint TNFR labels on them. **Reject** — adds nothing.

### 3.2 Interpretation B — *Diagnostic*
Use TNFR telemetry (C(t), Si, Φ_s, |∇φ|, K_φ, ξ_C) to **measure** what an existing model does internally. The model is unchanged; TNFR acts as a microscope.
- Concrete: pick an open small LLM (e.g. `microsoft/phi-2`, `google/gemma-2b`, `Qwen/Qwen2.5-Math-1.5B`), run it on math problems, project its hidden states onto a TNFR graph, compute the tetrad, look for invariants (e.g. does a wrong answer correlate with K_φ excursion above 0.9π?).
- This is **T1**. Cheap, publishable, proves the bridge exists before betting on it.

### 3.3 Interpretation C — *Regularizer*
Train (or fine-tune) a model with an extra loss term that penalises TNFR violations: e.g. `L_total = L_task + λ · max(0, |Φ_s| − 0.7711)`.
- This is **T2**. Honest but risky: the regularizer must be differentiable, and we don't yet know whether TNFR violations are **causes** of bad outputs or **effects**. T1 must answer that first.

### 3.4 Interpretation D — *Hybrid Reasoner*
Use an LLM **only as a glyph proposer**. The actual reasoning is performed by the TNFR engine in this repo: the LLM emits a candidate operator sequence (e.g. `[AL, EN, IL, OZ, IL, NAV]`), the engine validates it against U1–U6, runs it on a structural graph that encodes the problem, and returns a result. The LLM becomes a **policy** over the discrete action space of 13 operators.
- This is **T3**. The most TNFR-native interpretation that doesn't require inventing new hardware/architectures.
- Naturally suits problems already framed as graph reorganization: number theory (factorization, primality — already in this repo), molecular dynamics, network optimization.

### 3.5 Interpretation E — *Native Architecture*
Replace transformer layers with: a graph of NFR nodes, each holding (EPI, νf, φ); attention is replaced by phase-gated coupling (UM) and resonance (RA); training is replaced by grammar-guided structural reorganization (no backpropagation, or backprop only through differentiable surrogates of operators).
- This is **T4**. Requires inventing a learning rule that respects the nodal equation. Possibly the deepest TNFR contribution to ML, but blue-sky.

---

## 4. Survey of Free/Open Models (Realistic Candidates for T1–T3)

The user mentioned "adapt a free model, e.g. a mathematical one". As of 2026, viable open candidates **for math reasoning at small scale** (so we can run on a single GPU or even CPU-int8):

| Model | Params | License | Strengths | Why it fits TNFR |
|-------|--------|---------|-----------|------------------|
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 1.5B | Apache-2.0 | strong math at small size | manageable graph projection |
| `deepseek-ai/DeepSeek-Math-7B-Base` | 7B | DeepSeek License | strong math | bigger but still single-GPU |
| `microsoft/phi-2` / `phi-3-mini` | 2.7B / 3.8B | MIT | small, capable, well-instrumented | easy to probe |
| `google/gemma-2-2b-it` | 2B | Gemma License | clean architecture | well-documented hidden states |
| `meta-llama/Llama-3.2-1B` | 1B | Llama License | tiny, ubiquitous | smallest reasonable baseline |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Apache-2.0 | truly open, small | most permissive license |

**Pick for Phase 1 (T1)**: `SmolLM2-1.7B-Instruct` *or* `Qwen2.5-Math-1.5B-Instruct`. Both run on consumer hardware (≥16 GB RAM, GPU optional with quantization). Apache-2.0 / permissive licenses avoid legal complications when the IA/ folder becomes a public standalone repo.

---

## 5. The Recommended Path (Phase by Phase)

### Phase 0 — Scoping (this document) ✅

### Phase 1 — TNFR Microscope (T1)
**Goal**: prove that TNFR telemetry computed on an LLM's internal state contains **information** (i.e. is non-trivially correlated with task success/failure on math problems).

Deliverables:
1. `IA/probe/` package with:
   - Loader for a chosen open LLM (HuggingFace `transformers`).
   - Function `hidden_states_to_graph(h: Tensor) -> nx.Graph` that builds a TNFR-compatible graph from a layer's activations.
       Initial proposal: nodes = tokens (or attention heads), edges = top-k cosine similarity, EPI = activation magnitude, φ = phase of complex projection of activation, νf = activation entropy.
   - Bridge to existing `src/tnfr/physics/fields.py` to compute the tetrad on that graph.
2. A benchmark: run the model on `GSM8K` or `MATH` (or just first 100 problems), log per-problem (correct?, C(t), Si, Φ_s, |∇φ|, K_φ, ξ_C, layer index). Look for separators.
3. Report: `IA/reports/phase1_microscope.md` with plots and statistical tests.

**Stop condition** for Phase 1: either a clear TNFR signature of correctness exists (→ continue to Phase 2/3) or it does not (→ revisit the graph projection; do not invest in T2/T3 yet).

### Phase 2 — Optional: TNFR-Regularized Fine-Tune (T2)
Skip unless Phase 1 shows a clear "violation → bad answer" link. Otherwise the regularizer is noise.

### Phase 3 — Hybrid Reasoner (T3)
**Goal**: use an LLM as a glyph-sequence proposer, run the actual reasoning through the TNFR engine.

Deliverables:
1. Problem encoder: turn a math problem (e.g. factorize N, decide primality, simplify expression) into an initial TNFR graph (this repo already does it for primality / factorization).
2. LLM prompt that asks for a sequence of operators from {AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH}.
3. Validator: existing `tnfr.operators.grammar.validate_grammar`.
4. Executor: existing operator implementations.
5. Reward / scoring: did the resulting graph reach a known target structure (e.g. equilibrium ΔNFR = 0 for primality)?
6. Optional next layer: PPO / DPO fine-tune the LLM on (problem, sequence, reward) to make it propose better sequences over time.

This is the **first place** where the project is genuinely TNFR-AI rather than TNFR-decoration.

### Phase 4 — Native Architecture (T4) — Long Horizon
Replace transformer attention with a TNFR coupling layer. Open research. Out of scope for the first 6–12 months of the standalone repo.

---

## 6. What This Project Is *Not*

To protect against the failure mode of "TNFR-flavoured ML that has no actual TNFR physics":

- **Not** "rename loss to ΔNFR" — see §2.
- **Not** "add φ-shaped activation function" — cosmetic.
- **Not** "publish a paper claiming consciousness" — see `AGENTS.md` Technical Communication Standard.
- **Not** "compete with GPT-N on benchmarks" — wrong axis. The axis is *interpretability via structural coherence*.
- **Not** "another agent framework" — the operators are not tools; they are physical transformations.

---

## 7. Honest Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| TNFR telemetry on LLM activations is uninformative (Phase 1 fails) | medium | Try multiple graph projections before giving up. If all fail, the project pivots to T3 directly (since T3 does not depend on Phase 1). |
| The LLM cannot learn to emit valid grammar sequences | low-medium | Constrained decoding (mask invalid next-glyphs) makes this trivially solvable. |
| Reward signal in T3 is too sparse | medium | Use the engine's tetrad as dense intermediate reward (e.g. minimize Φ_s drift). |
| License contamination when the repo becomes standalone | low | Stick to Apache-2.0 / MIT / Gemma-with-attribution models. Avoid Llama for the initial open release. |
| Compute cost blows up | low at this scale | 1.5B params + int8 runs on a single consumer GPU; CPU-only is acceptable for Phase 1 with a smaller subset. |
| TNFR purists reject any bridge to statistical ML | medium | Be explicit: this project is *applied TNFR*, not a redefinition of TNFR. |

---

## 8. What an Outsider to AI Should Read First

(Minimal reading list, in order, ~10 hours total)

1. Andrej Karpathy, *Let's build GPT from scratch* (YouTube, 2 h) — what an LLM actually is mechanically.
2. HuggingFace `transformers` quickstart — the practical API you will live in.
3. *The Annotated Transformer* (Harvard NLP) — the math, slowly.
4. *Direct Preference Optimization* (Rafailov et al., 2023) — modern way to fine-tune from reward without RL.
5. *Constrained decoding* references (e.g. `outlines`, `guidance` libraries) — how to force an LLM to output valid structured sequences (essential for Phase 3).

You do **not** need to read deep RL literature to start. You **do** need to internalise §2 of this document.

---

## 9. Concrete First Deliverables (Next Pull Requests in the Standalone Repo)

When this folder is extracted into its own repository, the first PRs should be:

1. `README.md` (top-level) — reuse §1 of this document.
2. `pyproject.toml` — declare `tnfr` as a dependency (pin to current PyPI version).
3. `IA/probe/loader.py` — load a chosen HuggingFace model with a single function.
4. `IA/probe/graph.py` — `hidden_states_to_tnfr_graph(h, method="cosine_topk")`.
5. `IA/probe/telemetry.py` — wraps `tnfr.physics.fields.compute_*` over the projected graph.
6. `IA/experiments/01_microscope_gsm8k.py` — runs a small benchmark and writes a CSV.
7. `IA/reports/phase1_microscope.md` — analysis.

Estimated size of Phase 1 codebase: **< 1500 lines of Python**. This is small. The risk is in the design decisions (§3, §5), not in the volume of code.

---

## 10. Decision Points for the User (Required Before Phase 1 Code Starts)

Before writing any code, the user must commit to:

1. **Interpretation**: B → D path (Microscope first, then Hybrid Reasoner). Confirm or override.
2. **Base model**: `SmolLM2-1.7B-Instruct` (Apache-2.0, smallest friction) vs `Qwen2.5-Math-1.5B-Instruct` (math-specialised but DashScope license). Recommendation: **SmolLM2** for the first probe; switch to Qwen-Math only if math performance is the binding constraint.
3. **Compute envelope**: CPU-only, single GPU, or rented cloud GPU? This determines model size and benchmark scale.
4. **License of the standalone repo**: Apache-2.0 (recommended, matches `tnfr` package) vs MIT vs AGPL.
5. **Public or private** during Phase 1 (recommendation: **private** until Phase 1 yields a publishable signal).

---

## 11. Appendix — Why This Is Worth Doing

If T3 succeeds even modestly, the result is qualitatively new:

- An AI whose **reasoning trace is auditable** as a sequence of physically-grounded operations, not a chain of opaque token probabilities.
- An AI whose **failure modes are diagnosable** in physical units (Φ_s drift, K_φ excursion) instead of just "the model hallucinated".
- An AI **constrained by a grammar that comes from physics**, not from human-designed prompt engineering.

That is a defensible, original contribution. It does not require beating GPT-5; it requires showing that TNFR-validated reasoning beats unconstrained reasoning on at least one well-defined problem class (the obvious candidate being structural problems already covered by this engine: primality, factorization, network optimization).

---

**End of study.** No code follows from this document. The next step is for the user to answer §10 so Phase 1 can be scoped precisely.
