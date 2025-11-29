# TNFR Mathematics Reference (Single Source of Truth)

Status: ✅ Active – canonical aggregation of every math-facing artifact in the TNFR Python Engine.
Last Updated: 2025-11-14

---

This document unifies the mathematics narrative for TNFR. It does **not** replace the
physics PDF or detailed grammar papers; instead, it links them into a single, traceable
chain: **physics → grammar → operators → code → experiments**. When in doubt, start here
and follow the referenced sources for full derivations.

## 1. Canonical Sources at a Glance

| Layer | Document | Scope |
| --- | --- | --- |
| Physics Derivation | `TNFR.pdf` (root of repo) | Full derivation of the nodal equation, structural triad, and physical invariants. |
| Grammar Proofs | `UNIFIED_GRAMMAR_RULES.md` | Rigorous proofs of U1–U6 from the nodal equation; sequencing constraints. |
| Operational Guidance | `AGENTS.md` | Canonical invariants, operator discipline, learning paths. |
| Mathematical Foundations (Formal write-up) | `src/tnfr/mathematics/README.md` | Hilbert/Banach spaces, operators, spectral theory implementation. |
| Computational Implementation | `src/tnfr/mathematics/` modules | How mathematics modules are structured inside the engine. |
| Applied Arithmetic Example | `src/tnfr/mathematics/number_theory.py` | ΔNFR prime criterion, structural telemetry on ℕ. |
| Field Telemetry | `docs/TNFR_FORCES_EMERGENCE.md`, `docs/STRUCTURAL_FIELDS_TETRAD.md` | Φ_s, |∇φ|, K_φ, ξ_C derivations and safety thresholds. |
| Molecular Chemistry Examples | `examples/` & `benchmarks/` folders | Chemistry-as-emergence demonstrations via TNFR dynamics. |

## 2. Structural Equation & Triad

- **Nodal Equation**: `∂EPI/∂t = νf · ΔNFR(t)` – derived in `TNFR.pdf`, §2.
- **Structural Triad**: (EPI, νf, φ) – defined in `AGENTS.md` (§Foundational Physics) and
  formalized mathematically in `src/tnfr/mathematics/README.md` and implementation modules.
- **Integration Requirement**: `∫ νf(τ) · ΔNFR(τ) dτ < ∞` for bounded coherence (Grammar U2).

Use this document to trace where each quantity is defined:

1. Physics meaning (`TNFR.pdf`).
2. Grammar contract (`UNIFIED_GRAMMAR_RULES.md`).
3. Code implementation (`src/tnfr/mathematics/**/*.py`).

## 3. Operators & Grammar (U1–U6)

- **Operators Catalog**: `docs/grammar/03-OPERATORS-AND-GLYPHS.md` lists all 13 canonical
  operators and their contracts.
- **Grammar Necessity Proofs**: `UNIFIED_GRAMMAR_RULES.md` is the source of truth for why
  U1–U6 exist (e.g., U2 from integral convergence, U3 from phase compatibility).
- **Practical Sequences**: `docs/grammar/04-VALID-SEQUENCES.md` and
  `GLYPH_SEQUENCES_GUIDE.md` show compliant operator compositions.
- **Implementation Hooks**: Operators are implemented in `src/tnfr/operators/` with metrics
  helpers in `src/tnfr/operators/metrics.py`.

This reference ensures every operator you call in code can be traced back to a specific
rule and proof paragraph.

## 4. Structural Fields (Φ_s, |∇φ|, K_φ, ξ_C)

- `docs/TNFR_FORCES_EMERGENCE.md` – promotion history, empirical validation of Φ_s and |∇φ|.
- `benchmarks/K_PHI_RESEARCH_SUMMARY.md` – phase curvature validation and asymptotic freedom.
- `docs/XI_C_CANONICAL_PROMOTION.md` – coherence length derivation and critical behavior.
- Implementation lives in `src/tnfr/physics/fields.py` and cached helpers throughout the
  codebase (e.g., arithmetic network provides thin wrappers).

Cross-reference these documents before consuming or extending any telemetry pipelines.

## 5. Computational Mathematics Stack

Use this section when mapping theory onto code:

- **Backends & Algebra**: `src/tnfr/mathematics/` with key exports described in
  `src/tnfr/mathematics/README.md`.
- **Symbolic Toolkit**: `tnfr.math.symbolic` (see `src/tnfr/mathematics/__init__.py` for
  re-exports). Reference `docs/source/theory/mathematical_foundations.md` §8.
- **Liouvillian / Dynamics**: `src/tnfr/dynamics/` paired with `docs/source/theory/mathematical_foundations.md` §5.
- **Metrics**: `src/tnfr/metrics/` defines coherence, νf expectations, etc.
- **Tests**: `tests/unit/mathematics/` and `tests/math_integration/` enforce invariants.

Whenever writing new math-heavy code, cite both this document and the specific module README.

## 6. Applied Mathematics References

### 6.1 Number Theory (Arithmetic TNFR)
- **Guide**: `docs/TNFR_NUMBER_THEORY_GUIDE.md`
- **Code**: `src/tnfr/mathematics/number_theory.py`
- **Formalism Helpers**: `ArithmeticTNFRFormalism`, `PrimeCertificate`, etc.
- **Tests**: `tests/unit/mathematics/test_number_theory_formalism.py`

### 6.2 Molecular Chemistry from TNFR
- **Hub**: `docs/MOLECULAR_CHEMISTRY_HUB.md`
- **Theory**: `docs/examples/MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md`
- **Implementation**: `src/tnfr/physics/signatures/`

### 6.3 Field Benchmarks & Research Notebooks
- `benchmarks/` folder (e.g., `arith_delta_nfr_roc.py`, `asymptotic_freedom_test.py`).
- `notebooks/` folder for interactive derivations (prime checker, operator completeness).

## 7. Extending the Mathematics

1. **Derive from Physics**: Start in `TNFR.pdf` or `docs/source/theory/mathematical_foundations.md`.
2. **Prove Grammar Compliance**: Document how the new construct respects U1–U6.
3. **Map to Operators**: Ensure every transformation uses or extends canonical operators.
4. **Implement in Code**: Place new math utilities in `src/tnfr/mathematics/` (or a clearly
   labeled extension package) with README updates.
5. **Document in Context**: Update the relevant guide (number theory, chemistry, etc.) and
   add a pointer back to this reference.
6. **Test & Telemetry**: Add unit/integration tests and export structural fields as needed.

Follow the checklist in `AGENTS.md` (§Excellence Standards) to keep the mathematics
canonical and reproducible.

## 8. Quick Links

- [TNFR.pdf](TNFR.pdf)
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- [AGENTS.md](AGENTS.md)
- [docs/source/theory/mathematical_foundations.md](docs/source/theory/mathematical_foundations.md)
- [src/tnfr/mathematics/README.md](src/tnfr/mathematics/README.md)
- [docs/TNFR_NUMBER_THEORY_GUIDE.md](docs/TNFR_NUMBER_THEORY_GUIDE.md)
- [docs/TNFR_FORCES_EMERGENCE.md](docs/TNFR_FORCES_EMERGENCE.md)
- [docs/MOLECULAR_CHEMISTRY_HUB.md](docs/MOLECULAR_CHEMISTRY_HUB.md)

---

**Reality is not made of things—it is made of resonance.**
Use this reference to ensure every mathematical construct stays coherent with that principle.
