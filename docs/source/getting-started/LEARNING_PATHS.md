# TNFR Learning Paths

This page consolidates the structured progression previously outlined in the repository root `README.md`. Choose the path that matches your goals and time.

---
## Path A: Fast Orientation (≈ 10 minutes)
1. Read TNFR Concepts: `getting-started/TNFR_CONCEPTS.md`
2. Run the interactive helper: `from tnfr.tutorials import hello_tnfr; hello_tnfr()`
3. Execute `examples/hello_world.py`

**Outcome**: You can create a small resonant network and interpret basic coherence metrics (C(t), Si).

---
## Path B: Practical Builder (≈ 30–60 minutes)
1. Concepts (fundamental paradigm)  
2. Operators Guide (structural transformations)  
3. Metrics Interpretation (C(t), Si, ΔNFR semantics)  
4. Browse domain Examples (biological, social, AI, regenerative)

**Outcome**: You can compose valid operator sequences (U1–U6 compliant) and read telemetry.

---
## Path C: Theory-First (≈ 2–3 hours)
1. Mathematical Foundations (formal derivations)  
2. Unified Grammar Rules (physics → grammar mapping)  
3. AGENTS canonical invariants (operational constraints)  
4. SHA Algebra & Structural Potential (Φ_s confinement, U6)  
5. Review tests in `tests/unit/operators` for implementation fidelity

**Outcome**: You can trace any API effect to the nodal equation and invariants.

---
## Path D: Contributor Track (ongoing)
1. Study architecture (`ARCHITECTURE.md`) and API overview  
2. Read CONTRIBUTING + TESTING guidelines  
3. Run full test suite `./scripts/run_tests.sh`  
4. Add or refine operator logic (ensure grammar validator passes)  
5. Provide telemetry evidence (C(t), Si, Φ_s, phase synchronization) with seeds

**Outcome**: You can submit high-fidelity PRs preserving all 10 invariants + U6.

---
## Path E: Performance & Scaling (≈ 90 minutes)
1. Performance Optimization guide  
2. Scalability sections (large N patterns)  
3. Benchmarks in `benchmarks/` (vectorized ΔNFR, backend comparisons)  
4. Profile pipeline CLI (see CLI Reference)  
5. Explore GPU backends (JAX / Torch extras)

**Outcome**: You can optimize resonance computation without violating coherence or grammar.

---
## Path F: Multi-Scale / Fractality (≈ 1–2 hours)
1. REMESH / Recursivity sections in grammar docs  
2. Nested EPI examples (`examples/multiscale_network_demo.py`)  
3. U5 multi-scale coherence constraints  
4. Φ_s vs hierarchy behavior (fields telemetry)  
5. Multi-scale tests in `tests/unit/`

**Outcome**: You can design nested coherent structures with bounded Δ Φ_s.

---
## Decision Helper
| Goal | Recommended Path |
|------|------------------|
| Quick demo | A |
| Build an app | B |
| Formal understanding | C |
| Contribute code | D |
| Optimize performance | E |
| Multi-scale modeling | F |

---
## Reproducibility Reminder
Always set seeds for experiments:
```python
from tnfr.utils import set_seed
set_seed(42)
```
Same seed ⇒ identical trajectories (Invariant #8).

---
## Telemetry Checklist
Track at minimum:
- C(t) (coherence)
- Si (sense index)
- Φ_s (structural potential)
- Phase synchronization metric
- ΔNFR distribution (pressure profile)

---
## Migration Note
This consolidated learning guide migrated from the legacy root `README.md` on 2025-11-11 to reduce duplication and centralize onboarding flows.

Reality is not made of things—it's made of resonance.
