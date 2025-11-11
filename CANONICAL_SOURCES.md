# Canonical Documentation Sources

**Single Source of Truth Hierarchy for TNFR Documentation**

Version: 3.0  
Last Updated: 2025-11-11  
Status: ✅ CANONICAL - Authoritative documentation hierarchy

---

## 🎯 Purpose

This document establishes the **canonical information hierarchy** for the TNFR-Python-Engine repository. When documentation appears in multiple places, this hierarchy determines which source is authoritative.

**Core Principle**: Every concept should have ONE canonical definition. All other mentions should REFERENCE that canonical source, not replicate it.

---

## 📚 Canonical Hierarchy

### Tier 1: Ultimate Sources (Physics & Philosophy)

These define TNFR theory and paradigm. All implementation must align with these.

1. **[AGENTS.md](AGENTS.md)** - Complete TNFR guide for AI agents and developers
   - Paradigm shift (objects → resonance)
   - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
   - 13 canonical operators
   - Unified grammar (U1-U6)
   - 10 canonical invariants
   - Development workflow
   - **Audience**: All contributors, AI agents
   - **Role**: Master reference for TNFR principles

2. **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** - Mathematical derivations
   - Complete physics derivations for U1-U6
   - Proof of canonicity (Absolute/Strong)
   - Mathematical foundations
   - **Audience**: Advanced developers, theorists
   - **Role**: Rigorous mathematical proofs

3. **[GLOSSARY.md](GLOSSARY.md)** - Terminology reference
   - Quick definitions of all TNFR terms
   - Operator tables
   - Grammar rule summaries
   - **Audience**: All users (quick lookup)
   - **Role**: Rapid terminology reference

### Tier 2: Specialized Documentation (Single Responsibility)

These documents cover specific aspects in depth. They MAY provide context but MUST reference Tier 1 for canonical definitions.

4. **[docs/grammar/02-CANONICAL-CONSTRAINTS.md](docs/grammar/02-CANONICAL-CONSTRAINTS.md)** - U1-U6 technical specification
   - Detailed constraint specifications
   - Implementation examples
   - Test requirements
   - **Must reference**: UNIFIED_GRAMMAR_RULES.md for derivations

5. **[docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md)** - U6 complete specification
   - U6 physics, validation, implementation
   - **Must reference**: UNIFIED_GRAMMAR_RULES.md for derivation

6. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture
   - Module organization
   - Data flow
   - Design patterns
   - **Must reference**: AGENTS.md for invariants, operators, grammar

7. **[TESTING.md](TESTING.md)** - Test strategy
   - Testing requirements
   - Invariant test examples
   - **Must reference**: AGENTS.md for invariant definitions

8. **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development workflow
   - Contribution guidelines
   - Commit templates
   - Code standards
   - **Must reference**: AGENTS.md for invariants, operators

### Tier 3: Supporting Documentation (References Only)

These documents provide guidance, tutorials, examples. They MUST NOT define canonical concepts - only reference and use them.

9. **[README.md](README.md)** - Entry point
   - Project overview
   - Installation
   - Quick examples
   - Navigation links
   - **Role**: Landing page with minimal context

10. **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Navigation hub
    - Complete documentation map
    - Learning paths
    - Document finder
    - **Role**: Pure navigation, no definitions

11. **[docs/grammar/README.md](docs/grammar/README.md)** - Grammar navigation
    - Grammar documentation index
    - Learning sequence
    - **Role**: Navigation within grammar system

12. **docs/source/** - Extended documentation
    - API references
    - Tutorials
    - Theory elaborations
    - **Role**: Detailed guides and API docs

---

## ✅ Compliance Rules

### DO: Reference Canonical Sources

✅ **Correct pattern**:
```markdown
For complete operator definitions, see **[AGENTS.md § Canonical Operators](AGENTS.md#-the-13-canonical-operators)**.

Quick summary:
- Emission (AL): Creates EPI from vacuum
- Reception (EN): Captures incoming resonance
- ... [minimal context only]
```

✅ **Correct in code comments**:
```python
# Validate phase compatibility per U3 (RESONANT COUPLING)
# See UNIFIED_GRAMMAR_RULES.md § U3 for complete derivation
if abs(phase_i - phase_j) > delta_phi_max:
    raise ValidationError("Phase mismatch - violates U3")
```

### DON'T: Replicate Canonical Definitions

❌ **Incorrect pattern**:
```markdown
## The Nodal Equation

The nodal equation is the heart of TNFR:

∂EPI/∂t = νf · ΔNFR(t)

Where:
- EPI is the Primary Information Structure...
- νf is the structural frequency...
- ΔNFR is the nodal reorganization gradient...

[500 lines of content already in AGENTS.md]
```

### Acceptable Context Levels

**Minimal context** (OK everywhere):
- "EPI (structural form)"
- "νf (reorganization frequency in Hz_str)"
- "U3 requires phase verification"

**Brief summary** (OK in specialized docs):
- 2-3 sentence explanation
- Links to canonical source
- Example: ARCHITECTURE.md U1-U6 summary

**Complete definition** (ONLY in canonical sources):
- Full physics derivation
- Mathematical proofs
- Comprehensive examples
- Only in: AGENTS.md, UNIFIED_GRAMMAR_RULES.md, GLOSSARY.md, 02-CANONICAL-CONSTRAINTS.md

---

## 🔍 Verification

### Finding Canonical Source

**Question**: "Where is [concept X] canonically defined?"

**Answer**:
1. Check this document for hierarchy
2. For physics/operators/grammar → **AGENTS.md**
3. For math derivations → **UNIFIED_GRAMMAR_RULES.md**
4. For term lookup → **GLOSSARY.md**
5. For technical specs → **docs/grammar/02-CANONICAL-CONSTRAINTS.md**

### Detecting Redundancy

**Red flags** (indicates potential redundancy):
- ❌ Multiple files with "## The Nodal Equation" section
- ❌ Complete operator definitions in non-canonical docs
- ❌ U1-U6 derivations outside UNIFIED_GRAMMAR_RULES.md
- ❌ Invariant definitions (1-10) outside AGENTS.md

**Green patterns** (correct references):
- ✅ "See AGENTS.md for nodal equation details"
- ✅ "Operators (AGENTS.md § 3.2): emission, reception, ..."
- ✅ "Per U3 (RESONANT COUPLING - UNIFIED_GRAMMAR_RULES.md), ..."
- ✅ "Invariant #5 (AGENTS.md) requires phase verification"

---

## 📋 Audit Checklist

When creating or modifying documentation:

- [ ] Does this define a TNFR concept comprehensively?
  - If YES: Is this a Tier 1/2 canonical source?
  - If NO: Add reference to canonical source
  
- [ ] Am I replicating >100 words from another document?
  - If YES: Replace with reference + minimal context

- [ ] Does the reader need complete understanding of this concept HERE?
  - If NO: Link to canonical source instead of explaining

- [ ] Is this definition consistent with AGENTS.md?
  - If UNCERTAIN: Verify against AGENTS.md

- [ ] Have I indicated this is a SUMMARY not CANONICAL?
  - Add: "For complete details, see [CANONICAL_SOURCE]"

---

## 🔧 Migration from Redundant Documentation

If you find redundant documentation:

1. **Identify canonical source** (use hierarchy above)
2. **Extract unique content** (what's NOT in canonical source?)
3. **Move unique content** to appropriate canonical location
4. **Replace redundant sections** with references
5. **Update links** throughout repository
6. **Archive old versions** in docs/archive/ if historical value

**Example commit**:
```
Consolidate operator definitions to AGENTS.md

- Removed 500-line operator section from ARCHITECTURE.md
- Replaced with reference to AGENTS.md § Canonical Operators
- Kept architecture-specific implementation notes
- Updated all internal links

Reduces redundancy while preserving architectural context.
```

---

## 📊 Current State (as of 2025-11-11)

### ✅ Verified Canonical Sources

- AGENTS.md: Complete, U1-U6 integrated, 10 invariants ✅
- GLOSSARY.md: Centralized terminology ✅
- UNIFIED_GRAMMAR_RULES.md: U1-U6 derivations ✅
- docs/grammar/02-CANONICAL-CONSTRAINTS.md: U1-U6 specs ✅
- docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md: U6 complete ✅

### ✅ Verified Non-Duplicating References

- ARCHITECTURE.md: References AGENTS.md, minimal summaries ✅
- CONTRIBUTING.md: References invariants by number only ✅
- TESTING.md: Test examples, references AGENTS.md ✅
- README.md: Minimal overview, links to canonical sources ✅

### 📚 Documentation Counts

- Canonical sources (Tier 1-2): 8 documents
- Navigation/index documents: 5 documents
- Tutorial/guide documents: ~30 documents
- API documentation: ~15 documents
- READMEs (directory indexes): 26 documents

**Total**: ~84 active documentation files

---

## 🚀 Maintenance

**Update frequency**:
- Review this hierarchy: Quarterly (next: 2026-02-11)
- Audit compliance: When major docs added/modified
- Verify canonical sources: After each release

**Responsibility**:
- Maintainers: Enforce hierarchy in PR reviews
- Contributors: Check this document before modifying docs
- AI agents: Use this hierarchy for all documentation questions

---

## 📖 Related Documents

- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Complete navigation
- [AGENTS.md](AGENTS.md) - Master TNFR reference
- [GLOSSARY.md](GLOSSARY.md) - Quick terminology
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) - Grammar physics

---

**Version History**:
- v3.0 (2025-11-11): Created canonical hierarchy document
- v2.0 (2025-11-09): U6 canonicalization
- v1.0 (2025-11-06): Initial documentation consolidation

---

<div align="center">

**One concept, one source, many references**

</div>
