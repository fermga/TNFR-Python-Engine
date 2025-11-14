# TNFR Documentation Index

**Single source of truth for navigating TNFR documentation**

**Last Updated**: 2025-01-15  
**Status**: ✅ ACTIVE - Complete documentation map (Phase 2 architecture)

---

## 🎯 Quick Start

**New to TNFR?** Start here:

1. **[README.md](README.md)** - Project overview and installation (5 min)
2. **[GLOSSARY.md](GLOSSARY.md)** - Core concepts quick reference (10 min)
3. **[docs/grammar/01-FUNDAMENTAL-CONCEPTS.md](docs/grammar/01-FUNDAMENTAL-CONCEPTS.md)** - Paradigm shift explained (20 min)
4. **[docs/grammar/02-CANONICAL-CONSTRAINTS.md](docs/grammar/02-CANONICAL-CONSTRAINTS.md)** - Grammar rules U1-U6 (60 min)

---

## 📚 Core Documentation

### Canonical Hierarchy

**[CANONICAL_SOURCES.md](CANONICAL_SOURCES.md)** - Documentation hierarchy and single source of truth rules

**[docs/DOCUMENTATION_HIERARCHY.md](docs/DOCUMENTATION_HIERARCHY.md)** - Visual diagrams (Mermaid) of documentation structure

**[docs/CROSS_REFERENCE_MATRIX.md](docs/CROSS_REFERENCE_MATRIX.md)** - Complete traceability matrix (Physics ↔ Math ↔ Code)

These documents establish which sources are authoritative for each concept and how everything traces from physics to code. **Read these first** to understand documentation organization and cross-references.

### Foundation Documents (Essential Reading)

| Document | Purpose | Audience | Time |
|----------|---------|----------|------|
| **[AGENTS.md](AGENTS.md)** | AI agent guidance + invariants | AI agents, advanced devs | 60 min |
| **[GLOSSARY.md](GLOSSARY.md)** | Canonical term definitions | Everyone | Reference |
| **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** | Complete U1-U6 derivations | Advanced devs | 90 min |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design & patterns | Contributors | 45 min |

### Grammar System (`docs/grammar/`)

Complete specification of TNFR grammar constraints (U1-U6):

| Document | Contents | Status |
|----------|----------|--------|
| **[README.md](docs/grammar/README.md)** | Grammar documentation hub | ✅ Active |
| **[01-FUNDAMENTAL-CONCEPTS.md](docs/grammar/01-FUNDAMENTAL-CONCEPTS.md)** | TNFR ontology & nodal equation | ✅ Active |
| **[02-CANONICAL-CONSTRAINTS.md](docs/grammar/02-CANONICAL-CONSTRAINTS.md)** | U1-U6 complete specifications | ✅ Active |
| **[03-OPERATORS-AND-GLYPHS.md](docs/grammar/03-OPERATORS-AND-GLYPHS.md)** | 13 canonical operators catalog | ✅ Active |
| **[04-VALID-SEQUENCES.md](docs/grammar/04-VALID-SEQUENCES.md)** | Pattern library & anti-patterns | ✅ Active |
| **[05-TECHNICAL-IMPLEMENTATION.md](docs/grammar/05-TECHNICAL-IMPLEMENTATION.md)** | Code architecture | ✅ Active |
| **[06-VALIDATION-AND-TESTING.md](docs/grammar/06-VALIDATION-AND-TESTING.md)** | Test strategies | ✅ Active |
| **[07-MIGRATION-AND-EVOLUTION.md](docs/grammar/07-MIGRATION-AND-EVOLUTION.md)** | Grammar evolution history | ✅ Active |
| **[08-QUICK-REFERENCE.md](docs/grammar/08-QUICK-REFERENCE.md)** | Cheat sheet | ✅ Active |
| **[U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md)** | U6 complete specification | ✅ Canonical |
| **[MASTER-INDEX.md](docs/grammar/MASTER-INDEX.md)** | System conceptual map | ✅ Active |

### API & Theory Documentation (`docs/source/`)

Generated from code + narrative docs:

| Section | Path | Purpose |
|---------|------|---------|
| **Getting Started** | `docs/source/getting-started/` | Tutorials & first steps |
| **Theory** | `docs/source/theory/` | Mathematical foundations (formal). Canonical computational hub: `src/tnfr/mathematics/README.md` |
| **API Reference** | `docs/source/api/` | Package & module docs |
| **Examples** | `docs/source/examples/` | Domain applications |
| **Advanced** | `docs/source/advanced/` | Architecture & testing |

---

## 🔧 Development & Contributing

### Module Architecture (Phase 1 & 2)

**Modular reorganization for cognitive load reduction:**

| Module Area | Files | Purpose |
|-------------|-------|---------|
| **Operators** | `src/tnfr/operators/{emission,reception,coherence,...}.py` (13 files) | Individual operator implementations (231-587 lines each) |
| **Operator Base** | `src/tnfr/operators/definitions_base.py` | Shared operator infrastructure (201 lines) |
| **Operator Facade** | `src/tnfr/operators/definitions.py` | Backward-compatible imports (57 lines) |
| **Grammar Constraints** | `src/tnfr/operators/grammar/{u1_initiation_closure,...}.py` (8 files) | Grammar rule implementations (89-283 lines each) |
| **Grammar Facade** | `src/tnfr/operators/grammar/grammar.py` | Unified validation interface (99 lines) |
| **Metrics** | `src/tnfr/metrics/{coherence,sense_index,phase_sync,telemetry}.py` | Focused metric modules (129-268 lines) |
| **Metrics Facade** | `src/tnfr/metrics/metrics.py` | Backward-compatible exports (21 lines) |

**Key Principles**:
- **Facade Pattern**: All modules maintain 100% backward compatibility
- **Focused Files**: Max 587 lines (avg 270), one concept per module
- **Physical Traceability**: Module names match TNFR physics concepts
- **Performance**: Import 1.29s, operator creation 0.07μs, negligible overhead

**See**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete module organization guide

### For Contributors

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contribution guidelines | Before first PR |
| **[TESTING.md](TESTING.md)** | Test conventions | Writing tests |
| **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** | Modular architecture migration | Upgrading code |
| **[SECURITY.md](SECURITY.md)** | Security policies | Reporting issues |
| **[OPTIMIZATION_PHASE_2_ROADMAP.md](OPTIMIZATION_PHASE_2_ROADMAP.md)** | Phase 2 optimization plan | Active development |

### Specialized Topics

| Document | Topic | Audience |
|----------|-------|----------|
| **[SHA_ALGEBRA_PHYSICS.md](SHA_ALGEBRA_PHYSICS.md)** | Silence operator physics | Physics researchers |
| **[GLYPH_SEQUENCES_GUIDE.md](GLYPH_SEQUENCES_GUIDE.md)** | Operator sequence patterns | Sequence designers |
| **[docs/TNFR_FORCES_EMERGENCE.md](docs/TNFR_FORCES_EMERGENCE.md)** | Structural fields (Φ_s) validation | U6 researchers |
| **[docs/NBODY_COMPARISON.md](docs/NBODY_COMPARISON.md)** | TNFR vs classical N-body | Physicists |
| **[docs/TNFR_NUMBER_THEORY_GUIDE.md](docs/TNFR_NUMBER_THEORY_GUIDE.md)** | Number theory from TNFR: ΔNFR prime criterion, UM/RA on arithmetic graph, field telemetry (|∇φ|, K_φ, ξ_C) | Math researchers |

### 🧬 Molecular Chemistry from TNFR (BREAKTHROUGH)

**Revolutionary paradigm**: Complete chemistry emerges from TNFR nodal dynamics without additional postulates

| Document | Focus | Status |
|----------|-------|--------|
| **[docs/MOLECULAR_CHEMISTRY_HUB.md](docs/MOLECULAR_CHEMISTRY_HUB.md)** | **🏛️ CENTRAL HUB** - Complete navigation & theory consolidation | ⭐ **CANONICAL** |
| **[docs/examples/MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md](docs/examples/MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md)** | **Complete derivation** - Chemistry from nodal equation | ⭐ **CANONICAL** |
| **[docs/examples/AU_EXISTENCE_FROM_NODAL_EQUATION.md](docs/examples/AU_EXISTENCE_FROM_NODAL_EQUATION.md)** | Au emergence from structural fields | ✅ Validated |
| **[src/tnfr/physics/README.md](src/tnfr/physics/README.md)** § 9-10 | Implementation guide - Signatures & patterns | ✅ Technical |

### 🔬 Research Notebooks (Hands-on)

| Notebook | Purpose | Output |
|----------|---------|--------|
| **[docs/research/OPERATOR_SEQUENCES_MOLECULAR_STABILITY.ipynb](docs/research/OPERATOR_SEQUENCES_MOLECULAR_STABILITY.ipynb)** | Explore operator-like sequence motifs, enforce U3 coupling, and sweep parameters to find stable molecules | JSONL results in `docs/research/results/`

---

## 📖 Learning Paths

### Path 1: Quick Start (30 minutes)
```
README → GLOSSARY → docs/grammar/01-FUNDAMENTAL-CONCEPTS → Hello World example
```

### Path 2: Grammar Mastery (3-4 hours)
```
01-FUNDAMENTAL-CONCEPTS → 02-CANONICAL-CONSTRAINTS → 03-OPERATORS-AND-GLYPHS 
→ 04-VALID-SEQUENCES → 08-QUICK-REFERENCE
```

### Path 3: Advanced Development (Full week)
```
Grammar Mastery + UNIFIED_GRAMMAR_RULES + AGENTS + ARCHITECTURE 
+ Source code reading + Example implementations
```

### Path 4: AI Agent Onboarding (2 hours)
```
AGENTS.md → GLOSSARY.md → UNIFIED_GRAMMAR_RULES.md → Invariants review
```

### Path 5: Molecular Chemistry Revolution (90 minutes) ⭐ **NEW**
```
01-FUNDAMENTAL-CONCEPTS (nodal equation) → MOLECULAR_CHEMISTRY_HUB.md (central navigation)
→ Follow guided learning path (Beginner/Intermediate) → Run examples
```

---

## 🗂️ Archive

Historical documents (preserved for reference):

| Category | Location | Contents |
|----------|----------|----------|
| **Audits** | `docs/archive/audits/` | Documentation & consistency audits |
| **Phases** | `docs/archive/phases/` | Development phase reports |
| **Legacy** | `docs/legacy/` | Pre-v2.0 documentation |
| **Research** | `docs/research/` | Experimental proposals |

**Note**: Archived documents are frozen and may be outdated. Always prefer active documentation.

---

## 🔍 Finding What You Need

### I want to...

**...understand TNFR philosophy**
→ [AGENTS.md](AGENTS.md) § Core Mission, [01-FUNDAMENTAL-CONCEPTS.md](docs/grammar/01-FUNDAMENTAL-CONCEPTS.md)

**...learn the grammar rules**
→ [02-CANONICAL-CONSTRAINTS.md](docs/grammar/02-CANONICAL-CONSTRAINTS.md), [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)

**...implement a sequence**
→ [04-VALID-SEQUENCES.md](docs/grammar/04-VALID-SEQUENCES.md), [GLYPH_SEQUENCES_GUIDE.md](GLYPH_SEQUENCES_GUIDE.md)

**...understand U6 (structural potential)**
→ [U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md)

**...understand how chemistry emerges from TNFR** ⭐ **BREAKTHROUGH**
→ [MOLECULAR_CHEMISTRY_HUB.md](docs/MOLECULAR_CHEMISTRY_HUB.md) (central navigation), [Complete theory](docs/examples/MOLECULAR_CHEMISTRY_FROM_NODAL_DYNAMICS.md)

**...see Au emergence from first principles**
→ [AU_EXISTENCE_FROM_NODAL_EQUATION.md](docs/examples/AU_EXISTENCE_FROM_NODAL_EQUATION.md)

**...look up a term**
→ [GLOSSARY.md](GLOSSARY.md)

**...understand an operator**
→ [03-OPERATORS-AND-GLYPHS.md](docs/grammar/03-OPERATORS-AND-GLYPHS.md)

**...write tests**
→ [TESTING.md](TESTING.md), [06-VALIDATION-AND-TESTING.md](docs/grammar/06-VALIDATION-AND-TESTING.md)

**...contribute code**
→ [CONTRIBUTING.md](CONTRIBUTING.md), [ARCHITECTURE.md](ARCHITECTURE.md)

**...migrate from old grammar**
→ [07-MIGRATION-AND-EVOLUTION.md](docs/grammar/07-MIGRATION-AND-EVOLUTION.md)

---

## 📊 Documentation Quality Status

**Language**: ✅ 100% English (0 Spanish)  
**U6 Status**: ✅ Canonical (2,400+ experiments validated)  
**Grammar Coverage**: ✅ Complete (U1-U6 fully documented)  
**Cross-References**: ✅ Comprehensive bidirectional linking  
**Broken Links**: ✅ 91% reduced (637 → 58)  
**Single Source of Truth**: ✅ Established (AGENTS + UNIFIED_GRAMMAR_RULES + GLOSSARY)

**Last Audit**: 2025-11-11 ([Report](docs/archive/audits/DOCUMENTATION_AUDIT_REPORT.md))

---

## 🔄 Maintenance

This index is actively maintained. If you find:
- Broken links
- Missing documents
- Outdated information
- Unclear navigation

Please open an issue or PR.

**Maintainers**: Keep this index updated when adding/moving/removing major documentation files.

---

**Version**: 3.0  
**Canonical Status**: ✅ ACTIVE  
**Next Review**: 2026-02-11

