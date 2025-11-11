# TNFR Grammar Documentation - Navigation Guide

<div align="center">

**Centralized and unified documentation for the TNFR grammar system**

[📖 Concepts](#-fundamental-concepts) • [📐 Constraints](#-canonical-constraints) • [⚙️ Operators](#️-operators-and-glyphs) • [🔄 Sequences](#-valid-sequences) • [💻 Implementation](#-implementation) • [🧪 Testing](#-testing) • [📚 Quick Reference](#-quick-reference)

</div>

---

## 🎯 Purpose

This directory contains the **single source of truth** for all TNFR grammar-related documentation. It consolidates previously fragmented information across multiple files into a clear, navigable hierarchical structure.

### Why this reorganization?

**Before:** Documentation fragmented across README.md, UNIFIED_GRAMMAR_RULES.md, GRAMMAR_MIGRATION_GUIDE.md, GLYPH_SEQUENCES_GUIDE.md, source code, and scattered tests.

**Now:** A modular structure where each grammar aspect has a defined place and everything is interconnected.

---

## 📑 Documentation Structure

### 🌊 Abstraction Levels

This documentation follows a **gradual abstraction** model from concepts to implementation:

```
Physical Intuition → Mathematical Formalization → Code Implementation → Test Validation
```

### 📂 Document Organization

#### **Level 1: Fundamental Concepts**

**[01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)**
- TNFR ontology: From objects to resonant patterns
- Paradigm shift: Coherence vs. Causality
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
- Structural triad: Form (EPI), Frequency (νf), Phase (φ)
- Integrated dynamics and convergence
- **Audience:** New users, developers needing to understand "the why"
- **Reading time:** 20-30 minutes

#### **Level 2: Canonical Constraints**

**[02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)**
- **U1: STRUCTURAL INITIATION & CLOSURE**
  - U1a: Initiators (Generators)
  - U1b: Closures
  - Physical derivation: ∂EPI/∂t undefined at EPI=0
- **U2: CONVERGENCE & BOUNDEDNESS**
  - Stabilizers vs. Destabilizers
  - Integral convergence theorem
- **U3: RESONANT COUPLING**
  - Phase verification
  - Interference physics
- **U4: BIFURCATION DYNAMICS**
  - U4a: Triggers need handlers
  - U4b: Transformers need context
- **Each constraint includes:** Intuition → Derivation → Implementation → Tests
- **Audience:** Developers implementing validation, advanced contributors
- **Reading time:** 45-60 minutes

#### **Level 3: Canonical Operators**

**[03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)**
- Catalog of 13 canonical operators
- Standard format for each operator:
  - **Physics:** What transformation does it represent?
  - **Effect:** Impact on ∂EPI/∂t
  - **When to use:** Use cases
  - **Grammar:** Classification (Generator, Stabilizer, etc.)
  - **Contract:** Pre/postconditions
  - **Examples:** Executable code
- **Classification by grammatical role**
- **Operator composition**
- **Audience:** All developers
- **Reading time:** 60-90 minutes (constant reference)

#### **Level 4: Valid Sequences**

**[04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md)**
- **Canonical patterns:**
  - Bootstrap: [Emission, Coupling, Coherence]
  - Stabilize: [Coherence, Silence]
  - Explore: [Dissonance, Mutation, Coherence]
  - Propagate: [Resonance, Coupling]
- **Anti-patterns** (invalid sequences and why)
- **Step-by-step validation logic**
- **Complex sequence examples**
- **Structural pattern detection**
- **Audience:** Developers building sequences, debugging
- **Reading time:** 30-45 minutes

#### **Level 5: Technical Implementation**

**[05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md)**
- **Architecture of `grammar.py`**
- **Operator sets** (GENERATORS, CLOSURES, etc.)
- **Validation functions:**
  - `validate_grammar(sequence, epi_initial)`
  - `validate_resonant_coupling(G, node_i, node_j)`
  - Internal helpers
- **Telemetry and logging**
- **Integration with `definitions.py`**
- **Extension points**
- **Audience:** Developers modifying core
- **Reading time:** 45-60 minutes

#### **Level 6: Validation and Testing**

**[06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md)**
- **Grammar testing strategy**
- **Tests per constraint (U1-U5)**
- **Monotonicity tests (coherence)**
- **Bifurcation tests**
- **Propagation tests**
- **Multi-scale tests (fractality)**
- **Reproducibility tests**
- **Minimum required coverage**
- **How to add tests for new constraints**
- **Audience:** Developers writing tests, QA
- **Reading time:** 30-45 minutes

#### **Level 7: Migration and Evolution**

**[07-MIGRATION-AND-EVOLUTION.md](07-MIGRATION-AND-EVOLUTION.md)**
- **Grammar system history:**
  - C1-C3 (legacy grammar.py)
  - RC1-RC4 (legacy canonical_grammar.py)
  - U1-U5 (current unified grammar)
- **Mapping old → new rules**
- **Deprecations and breaking changes**
- **Procedure for adding new constraints**
- **Maintenance guarantees**
- **Audience:** Maintainers, contributors migrating old code
- **Reading time:** 20-30 minutes

#### **Level 8: Quick Reference**

**[08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md)**
- **Cheat sheet of U1-U5 constraints**
- **Operator table** with glyphs and classification
- **Common sequences lookup table**
- **Validation decision tree**
- **Frequent import commands**
- **Common troubleshooting**
- **Audience:** Everyone (quick reference during development)
- **Reading time:** 5-10 minutes

---

### 📚 Complementary Documents

**[GLOSSARY.md](GLOSSARY.md)**
- Operational definitions of all TNFR terms
- Format: Term → Symbol → Code → Meaning → Reference
- **Audience:** Everyone
- **Use:** Constant reference

**[MASTER-INDEX.md](MASTER-INDEX.md)**
- Global conceptual map of grammar system
- Relationships between concepts
- Dependency diagram
- **Audience:** Developers planning large changes
- **Use:** Holistic system view

**[EXECUTIVE-SUMMARY.md](EXECUTIVE-SUMMARY.md)**
- High-level overview for managers and stakeholders
- Business value and strategic importance
- Current status and roadmap
- **Audience:** Non-technical decision makers
- **Use:** Strategic planning and resource allocation

**[TOOLING-AND-AUTOMATION.md](TOOLING-AND-AUTOMATION.md)**
- Complete guide to validation scripts and tools
- CI/CD integration and pre-commit hooks
- Development workflows and best practices
- **Audience:** Developers and DevOps engineers
- **Use:** Daily development and automation setup

---

### 💡 Executable Examples

**[examples/](examples/)**
- **01-basic-bootstrap.py:** Basic initialization sequence
- **02-intermediate-exploration.py:** Controlled destabilization exploration
- **03-advanced-bifurcation.py:** Bifurcation and mutation handling
- **04-anti-patterns.py:** Invalid sequence examples (commented)
- **05-multi-scale.py:** Nested EPIs and fractality
- All verifiable with `pytest`

---

### 🔧 JSON Schemas

**[schemas/](schemas/)**
- **constraints schema:** Formal constraint definitions (updated to include U5)
- **canonical-operators.json:** Metadata for 13 operators
- **valid-sequences.json:** Catalog of canonical patterns
- **Use:** Programmatic validation, tooling, IDEs

---

## 🚀 How to Use This Documentation

### For New Users

**Recommended learning path:**

1. **[01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)** - Understand TNFR paradigm
2. **[GLOSSARY.md](GLOSSARY.md)** - Familiarize with key terms
3. **[03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)** - Learn the 13 operators
4. **[examples/01-basic-bootstrap.py](examples/01-basic-bootstrap.py)** - Run first example
5. **[08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md)** - Keep handy during development

**Total time:** ~2 hours for operational fundamentals

### For Intermediate Developers

**If you already know TNFR and want to implement sequences:**

1. **[04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md)** - Patterns and anti-patterns
2. **[02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)** - U1-U5 constraints
3. **[examples/](examples/)** - Run intermediate and advanced examples
4. **[08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md)** - Quick lookup

**Total time:** ~90 minutes

### For Advanced Contributors

**If you're modifying core or adding features:**

1. **[05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md)** - Code architecture
2. **[06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md)** - Test strategy
3. **[MASTER-INDEX.md](MASTER-INDEX.md)** - System conceptual map
4. **[07-MIGRATION-AND-EVOLUTION.md](07-MIGRATION-AND-EVOLUTION.md)** - How to evolve system
5. **[schemas/](schemas/)** - Validation schemas

**Total time:** ~2-3 hours for complete mastery

---

## 🔗 External References

### Main Repository Documentation

- **[../../README.md](../../README.md)** - TNFR project overview
- **[../../UNIFIED_GRAMMAR_RULES.md](../../UNIFIED_GRAMMAR_RULES.md)** - Complete formal derivations (original source)
- **[../../AGENTS.md](../../AGENTS.md)** - Canonical invariants and contracts
- **[../../GLOSSARY.md](../../GLOSSARY.md)** - General project glossary
- **[../../TNFR.pdf](../../TNFR.pdf)** - Complete theoretical foundations

### Implementation

- **[../../src/tnfr/operators/grammar.py](../../src/tnfr/operators/grammar.py)** - Canonical implementation
- **[../../src/tnfr/operators/definitions.py](../../src/tnfr/operators/definitions.py)** - Operator definitions
- **[../../tests/unit/operators/test_unified_grammar.py](../../tests/unit/operators/test_unified_grammar.py)** - Test suite

### Documentation Sync Tool

- **[../../tools/sync_documentation.py](../../tools/sync_documentation.py)** - Centralized sync tool
- **[CODE_DOCS_CROSSREF.md](CODE_DOCS_CROSSREF.md)** - Bidirectional cross-references

**Run sync check:**
```bash
python tools/sync_documentation.py --all
```

This validates:
- All functions documented (17/17 ✓)
- All examples execute (8/8 ✓)
- Cross-references accurate (35 documented)
- Schema matches implementation

---

## 📝 Writing Conventions

### Format

- **Language:** English for all technical content
- **Equations:** Standard mathematical notation with LaTeX
- **Code:** Python 3.9+ with type hints
- **References:** Internal relative links, external absolute links

### Section Structure

Each technical document follows this structure:

```markdown
# Document Title

## Purpose
[What this document is for]

## Key Concepts
[Prerequisites needed]

## Main Content
[Development with subsections]

## Examples
[Executable code]

## References
[Links to other documents]
```

### Code

All code examples must:
- ✅ Be executable
- ✅ Include complete imports
- ✅ Have explanatory comments
- ✅ Follow TNFR conventions (don't modify EPI directly, etc.)
- ✅ Include expected telemetry output

---

## 🤝 Contributing to This Documentation

### Principles

1. **Single source of truth:** Don't duplicate, cross-reference
2. **Physics first:** All documentation derives from TNFR physics
3. **Incremental:** Add without breaking existing structure
4. **Validable:** Executable examples, updatable JSON schemas

### Adding New Content

**To add a new constraint:**
1. Document physics in `02-CANONICAL-CONSTRAINTS.md`
2. Implement in `../../src/tnfr/operators/grammar.py`
3. Add tests in `../../tests/unit/operators/test_unified_grammar.py`
4. Update `schemas/constraints-u1-u4.json`
5. Add examples in `examples/`
6. Update `08-QUICK-REFERENCE.md`

**To add a new operator:**
1. Document in `03-OPERATORS-AND-GLYPHS.md`
2. Implement in `../../src/tnfr/operators/definitions.py`
3. Update classification in `../../src/tnfr/operators/grammar.py`
4. Add contract tests
5. Update `schemas/canonical-operators.json`

### Maintaining Coherence

**Before making PR:**
- [ ] All examples are executable
- [ ] Bidirectional links work
- [ ] JSON schemas reflect changes
- [ ] Tests pass
- [ ] Changes documented in 07-MIGRATION-AND-EVOLUTION.md if breaking

---

## 📊 Completion Status

### ✅ Complete
- Directory structure
- Navigation README (this file)
- Main cross-references

### 🚧 In Progress
- 01-FUNDAMENTAL-CONCEPTS.md
- 02-CANONICAL-CONSTRAINTS.md
- 03-OPERATORS-AND-GLYPHS.md
- 04-VALID-SEQUENCES.md
- 05-TECHNICAL-IMPLEMENTATION.md
- 06-VALIDATION-AND-TESTING.md
- 07-MIGRATION-AND-EVOLUTION.md
- 08-QUICK-REFERENCE.md

### 📋 Planned
- GLOSSARY.md (consolidate from ../../GLOSSARY.md)
- MASTER-INDEX.md
- examples/*.py
- schemas/*.json

---

## 🎓 Documentation Philosophy

> **"If a change cannot be traced from TNFR physics to code to tests, it is not canonical."**

This documentation exists to make that traceability **explicit, navigable, and maintainable**.

### Values

- **Clarity over brevity:** Better to explain twice than leave doubts
- **Physics over convention:** Every rule derives inevitably from nodal equation
- **Code over prose:** Executable examples > abstract descriptions
- **Testing over trust:** Everything documented must be testable

---

## 📞 Contact and Support

**Found inconsistencies?**
- Open GitHub issue with label `documentation`

**Need navigation help?**
- Check [08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md) first
- Then consult specific document for your level

**Want to contribute?**
- Read [../../CONTRIBUTING.md](../../CONTRIBUTING.md)
- Then review "Contributing to This Documentation" section above

---

<div align="center">

**Version:** 1.0  
**Last updated:** 2025-11-10  
**Maintainer:** TNFR Core Team

**Reality is not made of things—it's made of resonance. Document accordingly.**

</div>
