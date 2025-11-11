# U6 Temporal Ordering: Investigation Report

**Investigation Date:** 2025-11-10  
**Status:** DEFER IMPLEMENTATION - Document as Research Proposal  
**Canonicity Assessment:** MODERATE (40% confidence)

---

## Executive Summary

This report documents a comprehensive investigation into whether **U6: Temporal Ordering** should be added to the canonical TNFR grammar alongside existing constraints U1-U5.

**Key Finding:** U6 has strong physical motivation and identifies real gaps in U1-U5, but lacks the mathematical inevitability required for canonical status.

**Decision:** Defer implementation while documenting thoroughly as a research proposal in UNIFIED_GRAMMAR_RULES.md.

---

## Key Results

### Gap Analysis

U6 identifies sequences that pass U1-U5 but may be problematic:

| Test Case | U1-U5 | U6 | Gap? |
|-----------|-------|-----|------|
| Consecutive OZ | ✓ Pass | ✗ Fail | Yes |
| OZ→ZHIR immediate | ✓ Pass | ✗ Fail | Yes |
| Triple destabilizers | ✓ Pass | ✗ Fail | Yes |

**Coverage:** 5/6 test cases showed gaps (83% improvement)

### Canonicity Assessment

| Criterion | Status |
|-----------|--------|
| Derives from nodal equation | ✗ FAIL |
| Prevents impossible sequences | ✓ PASS |
| Universal across domains | ✗ FAIL |
| No empirical tuning | ✗ FAIL |
| Independent from U1-U5 | ✓ PASS |

**Score:** 2/5 → MODERATE canonicity (40%)

---

## Decision Rationale

**Aligned with TNFR Philosophy:**

1. **"Physics First"** - Need complete derivation from nodal equation
2. **"No Arbitrary Choices"** - α parameter requires tuning (0.5-0.9)
3. **"Reproducibility Always"** - Empirical validation pending
4. **"Coherence Over Convenience"** - Don't add rules prematurely

---

## Path Forward

### Documentation Added

- **Location:** `UNIFIED_GRAMMAR_RULES.md` § "Proposed Constraints Under Research"
- Complete U6 specification with physical motivation
- Research roadmap for elevation to STRONG
- Timeline estimate: 6-12 months

### Research Needed

**HIGH Priority:**
1. Computational validation (measure actual τ_relax)
2. Theoretical derivation from nodal equation
3. Determine α from first principles

**MEDIUM Priority:**
4. Cross-domain validation
5. Alternative formulations

**Success Criteria:**
- >80% of U6 violations cause coherence loss
- Derivation from nodal equation
- α determinable without tuning
- Works across 3+ domains

---

## Conclusion

U6 is a **well-motivated research proposal** with strong physical analogies but insufficient mathematical inevitability for canonical inclusion.

**Better to document openly as research than to weaken canonical standards.**

---

**Full analysis:** See UNIFIED_GRAMMAR_RULES.md § Proposed Constraints  
**Test artifacts:** `/tmp/test_u6_necessity_v2.py`  
**Date:** 2025-11-10
