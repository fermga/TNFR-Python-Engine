# Phase 1 — TNFR Microscope (Concrete Plan)

**Status**: Locked (2026-04-23) after user decisions on §10 of `TNFR_AI_FEASIBILITY_STUDY.md`.
**Compute target**: CPU + NVIDIA RTX 3060 12 GB.
**Model**: `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Apache-2.0).
**License of standalone repo**: MIT.
**Visibility**: private.

---

## 1. Goal (one sentence)

Decide whether **TNFR telemetry computed on SmolLM2's hidden states carries information about answer correctness** on small math problems — i.e. whether the structural fields (Φ_s, |∇φ|, K_φ, ξ_C) computed on a graph projected from activations separate correct from incorrect runs.

If yes → Phase 3 (Hybrid Reasoner) is justified.
If no → revisit the projection (try 2–3 alternatives) before any larger investment.

---

## 2. Hardware budget

| Resource | Available | Used by Phase 1 |
|----------|-----------|-----------------|
| CPU | system CPU | tokenization, graph building, TNFR fields |
| GPU | RTX 3060 12 GB | SmolLM2 forward pass in fp16 (~3.4 GB), batch 1, ~512 tokens |
| Disk | local | model weights (~3.5 GB), GSM8K dataset (~5 MB), CSV outputs |
| Network | once | initial HF download |

SmolLM2-1.7B in fp16 fits comfortably. No quantization needed for Phase 1. Quantization (`bitsandbytes` 8-bit / 4-bit) is reserved for Phase 3 if VRAM becomes binding.

---

## 3. Software stack (pinned, conservative)

```
python >= 3.11
torch >= 2.4 (with CUDA 12.x for the RTX 3060)
transformers >= 4.45
accelerate >= 0.34
datasets >= 3.0
networkx >= 3.0  (already a tnfr dep)
numpy, scipy   (already tnfr deps)
tnfr >= 0.0.3  (local editable install while inside this repo)
matplotlib, pandas  (reports)
```

CUDA install note for the RTX 3060: PyTorch wheel index `https://download.pytorch.org/whl/cu121` (or cu124 if available at install time).

---

## 4. The graph projection (the only non-trivial design choice)

Given a hidden-state tensor `h ∈ ℝ^(T × d)` from one transformer layer for one prompt (T tokens, d = 2048 for SmolLM2):

| TNFR attribute | Mapped from | Formula (initial proposal) |
|----------------|-------------|----------------------------|
| Node = token | row of `h` | one node per token (cap T at 512) |
| Edge | token-token similarity | top-k cosine, k = 8 |
| EPI (per node) | activation magnitude | `‖h_t‖_2 / max_t ‖h_t‖_2` ∈ [0, 1] |
| φ (per node, phase) | complex projection | `arg( h_t · w_re + i · h_t · w_im )` with two fixed random unit vectors `w_re`, `w_im` (seeded for reproducibility) |
| νf (per node) | local entropy proxy | softmax over neighbour cosines → Shannon entropy, normalised to [0, 1] |

This is **one** projection. We will benchmark **three** alternatives in Phase 1 to avoid locking in a bad one:

- **P1**: cosine-topk (above).
- **P2**: thresholded cosine (edge if `cos > 0.5`).
- **P3**: attention-derived (use the model's actual attention matrix at that layer, top-k).

Each projection is a function `hidden_states_to_tnfr_graph(h, method) -> nx.Graph`. The TNFR fields are then computed via `tnfr.physics.fields.compute_unified_telemetry(G)`.

---

## 5. Benchmark dataset

- **GSM8K** (`openai/gsm8k`, `main` split, `test` subset, 1319 problems).
- Phase 1 runs on the **first 100 problems** (sample budget: ~5 minutes per problem × 100 = ~8 h on the 3060; reduce to 50 if too slow).
- Per problem we record:
  - `problem_id`, `gold_answer`, `model_answer`, `correct: bool`
  - For each chosen layer ∈ {4, 12, 23} (early / middle / final of SmolLM2's 24 layers):
    - `C(t), Si, phase_sync, Φ_s, |∇φ|, K_φ, ξ_C`
  - Wall-clock time per problem.

Output: `IA/reports/phase1_microscope.csv` + summary `phase1_microscope.md`.

---

## 6. Statistical test (decision criterion)

For each tetrad field `f ∈ {Φ_s, |∇φ|, K_φ, ξ_C}` and each layer:

- Mann-Whitney U test between the `correct = True` and `correct = False` distributions.
- Effect size: rank-biserial correlation `r`.
- **Pass criterion**: at least one (field, layer) pair with `p < 0.01` and `|r| > 0.2`.

If passed → Phase 3.
If borderline (`0.01 < p < 0.05` or `|r| < 0.2`) → run the other two projections (P2, P3) and re-test.
If all three projections fail → write up null result, pivot to Phase 3 directly (no need for Phase 2).

---

## 7. Deliverables (in order)

1. `IA/probe/loader.py` — `load_smollm2(device="cuda") -> (model, tokenizer)`.
2. `IA/probe/graph.py` — `hidden_states_to_tnfr_graph(h, method, k=8, seed=0)`.
3. `IA/probe/telemetry.py` — `tnfr_telemetry_from_hidden(h, method, layers) -> dict`.
4. `IA/experiments/01_microscope_smoke.py` — runs **5 problems** end-to-end as a sanity check (~5 min).
5. `IA/experiments/02_microscope_gsm8k.py` — full 100-problem run, writes CSV.
6. `IA/reports/phase1_microscope.md` — analysis with plots and the statistical test outcome.
7. `IA/requirements-phase1.txt` — pinned versions.

Estimated code size: **< 600 lines** for steps 1–5, **< 400 lines** for analysis. Total Phase 1 < 1 kLOC of Python. The weight is in physics-aware design, not volume.

---

## 8. Open risks specific to Phase 1

| Risk | Mitigation |
|------|-----------|
| HF model download blocked / slow | pre-download once; cache under `~/.cache/huggingface/` |
| `torch.cuda.OutOfMemoryError` even at fp16 batch=1 | drop to 256 tokens; if still failing, switch to 8-bit via `bitsandbytes` |
| `hidden_states` returned as tuple of (T, d) — must request `output_hidden_states=True` | encoded in `loader.py` defaults |
| Phase projection (`w_re`, `w_im` random vectors) introduces noise | seed = 0; later replace with PCA's first 2 components if needed |
| TNFR `compute_*` functions assume specific node attributes | one adapter file: `graph.py` sets `EPI`, `theta` (= φ), `nu_f` to match `tnfr` conventions |

---

## 9. What this phase does **not** do

- No fine-tuning. No backprop. Model weights are **frozen**.
- No new TNFR operators are introduced.
- No claims about "the model's reasoning" — only about correlations between TNFR fields and correctness.
- No public release.

---

## 10. Stop / Continue gate after Phase 1

| Outcome | Next |
|---------|------|
| ≥1 tetrad field separates correct/incorrect with `p < 0.01`, `|r| > 0.2` | proceed to Phase 3 design (Hybrid Reasoner) |
| All projections fail | write null-result note, pivot directly to Phase 3 (T3 doesn't depend on T1) |
| Mixed signal | run an extra projection (e.g. PCA-based phase) before deciding |

In all cases, document the outcome in `IA/reports/phase1_microscope.md` and update `IA/README.md` status table.
