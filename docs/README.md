# TNFR Documentation Hub

Specialized technical documentation for TNFR theory and implementation. The
primary reference is always [AGENTS.md](../AGENTS.md) (complete theory,
operators, grammar, fields); the files here are focused supplements.

All content derives from the nodal equation `∂EPI/∂t = νf · ΔNFR(t)`, the 13
canonical operators, grammar U1–U6, and the structural field tetrad
`(Φ_s, |∇φ|, K_φ, ξ_C)`.

---

## Documents

| Document | Status | Purpose |
|----------|--------|---------|
| [AGENTS.md](../AGENTS.md) | **CANONICAL** | Primary reference — complete TNFR theory |
| [STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md) | **CANONICAL** | Formal field definitions (Φ_s, \|∇φ\|, K_φ, ξ_C) |
| [grammar/PHYSICS_VERIFICATION.md](grammar/PHYSICS_VERIFICATION.md) | **CANONICAL** | U1–U6 grammar proofs from the nodal equation |
| [API_CONTRACTS.md](API_CONTRACTS.md) | Active | Pre/post-condition contracts for the 13 operators |
| [CANONICAL_OZ_SEQUENCES.md](CANONICAL_OZ_SEQUENCES.md) | Active | Dissonance (OZ) operator-sequence patterns |
| [STRUCTURAL_INTERFACE_THEORY.md](STRUCTURAL_INTERFACE_THEORY.md) | Active | Structural-interface programme: pipelines, fair benchmarks, validated results, limitations |

---

## Navigation

**Newcomers** → [AGENTS.md](../AGENTS.md) → [examples/](../examples/) (see
[examples/README.md](../examples/README.md) for the thematic index) → the
Simple SDK ([src/tnfr/sdk/simple.py](../src/tnfr/sdk/simple.py)).

**Developers** → [API_CONTRACTS.md](API_CONTRACTS.md) (operator specs) →
[STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md) (field math) →
[grammar/PHYSICS_VERIFICATION.md](grammar/PHYSICS_VERIFICATION.md) (grammar proofs).

**Theory** → the `theory/` folder holds the research programmes (Riemann,
Navier–Stokes, Yang–Mills, P-vs-NP, BSD, Hodge, number theory, the variational
principle, and the structural conservation theorem).

**Factorization** → the spectral Paley factorization entry point
(`tnfr.factorization.factorize`) lives in the main package; see
[factorization-lab/README.md](../factorization-lab/README.md) for the lab,
CLI usage, and certificate formats.
