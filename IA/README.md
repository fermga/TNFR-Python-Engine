# IA/ — TNFR-Structured AI (private, will be extracted)

This subfolder will become a **standalone private repository** later. While inside `TNFR-Python-Engine`, it consumes `tnfr` directly from the local source tree. After extraction it will depend on `tnfr` from PyPI.

## Status

- **Phase 0 — Scoping**: complete → `TNFR_AI_FEASIBILITY_STUDY.md`
- **Phase 1 — TNFR Microscope**: planned → `PHASE_1_PLAN.md`
- **Phase 2 / 3 / 4**: gated on Phase 1 results.

## User decisions (locked, 2026-04-23)

1. **Path**: B (Microscope) → D (Hybrid Reasoner). T2 skipped unless Phase 1 motivates it.
2. **Base model**: `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Apache-2.0).
3. **Compute**: local CPU + 1× NVIDIA RTX 3060 (12 GB VRAM).
4. **License**: MIT (same as `tnfr`).
5. **Visibility**: private until Phase 1 yields a publishable signal.

## Layout (target)

```
IA/
├── README.md                       # this file
├── TNFR_AI_FEASIBILITY_STUDY.md    # Phase 0 — scoping
├── PHASE_1_PLAN.md                 # Phase 1 — concrete plan
├── LICENSE                         # MIT (for the future standalone repo)
├── pyproject.toml                  # for the future standalone repo
├── probe/                          # Phase 1: TNFR microscope on an LLM
│   ├── __init__.py
│   ├── loader.py                   # HuggingFace model loader
│   ├── graph.py                    # hidden_states → TNFR graph
│   └── telemetry.py                # tetrad over the projected graph
├── experiments/
│   └── 01_microscope_smoke.py      # smallest end-to-end run
└── reports/                        # markdown + CSV outputs
```

## Quickstart (after Phase 1 code lands)

```bash
# from the repo root
pip install -e .                            # tnfr (already installed)
pip install -r IA/requirements-phase1.txt   # transformers, datasets, torch+cu121

python IA/experiments/01_microscope_smoke.py
```

## Non-goals

See §6 of `TNFR_AI_FEASIBILITY_STUDY.md`.
