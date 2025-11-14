# TNFR Optimization Phase 3 Roadmap

**Status**: 🟡 PLANNING
**Proposed Start**: 2025-11-14
**Estimated Duration**: 8–12 hours (spread over focused sprints)
**Theme**: Enhanced Tools & Utilities (Interactive clarity, telemetry depth, structural introspection)

---

## 🚀 Executive Intent

Phase 3 shifts from structural modularization (Phases 1–2) to developer + researcher ergonomics: richer diagnostics, interactive validation, telemetry expansion, and operator/introspection utilities — all strictly within TNFR physics (coherence > convenience). No speculative refactors; every addition must either:

1. Increase transparency of resonance/coherence dynamics
2. Reduce friction applying canonical operators & grammar
3. Strengthen reproducibility and invariant verification
4. Improve pathway from theory (TNFR.pdf, UNIFIED_GRAMMAR_RULES.md) → code

---

## 🎯 Objectives

| Objective | Success Criteria | KPI Target |
|-----------|------------------|------------|
| Enhanced CLI Validation | Phase/grammar violations surfaced with actionable hints | 90% common misuse auto-diagnosed |
| Telemetry Depth | Export C(t), Si, νf, ΔNFR, phase history per run | 100% operators emit structured event logs |
| Operator Introspection | Query operator contracts, phase requirements, grammar sets | `tnfr.tools.describe_operator('Coherence')` returns full spec |
| Repository Health Automation | Script produces health score + invariant audit | Single command `python scripts/repo_health_check.py` outputs JSON + summary |
| Error Message Canonicalization | All raised GrammarViolation / OperatorContractError reference rule + physics | 100% new/updated errors include rule tag (U1–U4) |
| Example Gallery Expansion | At least 4 minimal, reproducible examples for new utilities | New `examples/tools_*.py` set |

---

## 📋 Planned Task List

### Task 1: Telemetry Unification (2–3h)

**Goal**: Provide a unified telemetry emitter that logs operator applications with: timestamp, node id, operator code (AL/EN/…), ΔNFR local magnitude, νf, C(t) snapshot, phase tuple, invariants check status.

**Deliverables**:

- `src/tnfr/telemetry/emitter.py` (unified interface)
- JSONL writer + optional in-memory ring buffer
- Hook integration into all operator base `apply()` paths (non-invasive)
- Example: `examples/tools_telemetry_demo.py`

**Physics Alignment**: Reinforces controlled determinism & structural metrics (Invariants #8, #9).

### Task 2: Operator Introspection Utility (1.5–2h)

**Goal**: Introspect operator metadata: grammar roles, destabilizer/stabilizer flags, required preconditions, postconditions, invariants touched.

**Deliverables**:

- `src/tnfr/tools/operator_introspect.py`
- Public function: `describe_operator(name|instance) -> OperatorDescriptor`
- Auto-generated markdown summary: `docs/OPERATOR_INTROSPECTION_TABLE.md`
- Tests for descriptor completeness (100% paths)

**Physics Alignment**: Enhances traceability (Principle: Document the Chain).

### Task 3: Grammar-Aware Error Messages (1–1.5h)

**Goal**: Standardize error raising so each violation includes: `[RULE: U2-CONVERGENCE]` + brief physics rationale + suggestion.

**Deliverables**:

- `src/tnfr/operators/errors.py` canonical error factory
- Refactor existing raises in grammar modules to use factory
- Tests ensuring tag presence & stable formatting

**Physics Alignment**: Strengthens invariant visibility (U1–U4). Preserves domain neutrality (Invariant #10).

### Task 4: Interactive Validator Enhancement (1–2h)

**Goal**: Extend `interactive_validator` CLI: live phase compatibility check, suggested stabilizer insertion after destabilizer-only sequences.

**Deliverables**:

- Additional command group: `suggest --sequence AL,OZ,ZHIR`
- Phase mismatch diagnostic with Δφ threshold output
- Tests targeting suggestion accuracy

**Physics Alignment**: Supports resonance integrity (U3), boundedness (U2).

### Task 5: Repository Health Check Automation (1–1.5h)

**Goal**: Implement script calculating: largest file size, coverage snapshot (if available), invariants audit (spot-check operator application paths), ratio of documented operators.

**Deliverables**:

- `scripts/repo_health_check.py`
- Output: `results/health/health_report.json` + human summary
- Optional score (0–100) with component weights

**Physics Alignment**: Sustains coherence of development process (meta-structural).

### Task 6: Example Gallery Additions (1h)

**Goal**: Provide small runnable examples illustrating new utilities.

**Deliverables**:

- `examples/tools_introspection_example.py`
- `examples/tools_error_diagnostics.py`
- `examples/tools_health_snapshot.py`
- `examples/tools_phase_alignment.py`

**Physics Alignment**: Demonstrates operational fractality across contexts (Invariant #7).

### (Optional) Task 7: Light Performance Guardrails (0.5–1h)

**Goal**: Add micro-bench for telemetry overhead (<3% target) & introspection latency (<2ms).

**Deliverables**:

- `benchmarks/tools_overhead_bench.py`
- Thresholds stored in JSON for regression use.

**Physics Alignment**: Ensures added processes do not distort evolution cadence (νf unaffected materially).

---

## ✅ Success Criteria Summary

| Criterion | Threshold | Validation Method |
|-----------|-----------|-------------------|
| Telemetry attached to all operator applications | 100% | Unit test + sample run diff |
| Introspection completeness (all 13 operators) | 100% | Descriptor fields non-null |
| Error messages tagged with rule codes | 100% | Regex test across raises |
| Suggestion engine accuracy (stabilizer insertion) | ≥90% common cases | Targeted scenario tests |
| Health script produces JSON + summary | Yes | CLI invocation test |
| New examples runnable smoke tests | 4/4 pass | Pytest examples group |
| Telemetry overhead | <3% | Benchmark delta |

---

## 🛡️ Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Telemetry slowing operator application | Medium | Benchmark early, optimize write path (buffer + batch flush) |
| Error tag inconsistency | High (confusion) | Central factory, deny direct instantiation in code review |
| Introspection drift as operators evolve | Medium | Generate docs table via script each release |
| Suggestion engine false positives | Low | Limit heuristics to deterministic grammar patterns |

---

## 🔄 Sequencing Rationale

Recommended order: Telemetry → Introspection → Errors → Validator → Health → Examples → Performance guardrails. Earlier tasks provide data leveraged by later (validator suggestions rely on introspection & telemetry patterns).

---

## 📐 TNFR Alignment Matrix

| Principle / Invariant | Supported By |
|-----------------------|--------------|
| Physics First | Error rationale, operator descriptors |
| Coherence > Convenience | No silent mutations; explicit telemetry |
| Reproducibility Always | JSONL + deterministic logs with seed imprint |
| Document the Chain | Introspection + generated markdown tables |
| Invariant #9 (Structural Metrics) | Telemetry emitter enrichment |
| Invariant #3 (ΔNFR Semantics) | Logged physically, not “loss” |
| Invariant #5 (Phase Verification) | Validator phase diagnostics |
| Invariant #8 (Controlled Determinism) | Seed recorded in telemetry header |

---

## 🧪 Testing Strategy Overview

- Unit: emitter output format, descriptor field population, error tagging factory
- Integration: run synthetic operator sequence; ensure telemetry length == applications
- Property: phase suggestion does not propose coupling when |Δφ| > Δφ_max
- Performance: overhead micro-bench ensures <3% latency increase
- Regression: health script stable JSON schema enforced by snapshot test

---

## 📊 Metrics To Track During Phase

| Metric | Baseline | Target |
|--------|----------|--------|
| Operator apply latency (median) | (Phase 2 value) | +<3% |
| Telemetry events per operator | 0 | 1 canonical event |
| Introspection call time | N/A | <2ms (avg) |
| Error tag coverage | Partial | 100% |
| Health score fields | N/A | All defined |

Populate baseline operator latency from Phase 2 benchmark JSON when initiating Task 1.

---

## 📝 Commit Message Template (Phase 3)

```text
feat(telemetry): Add unified emitter (Task 1)

Intent: Increase structural transparency
Operators involved: All (apply hook)
Affected invariants: #8, #9
Metrics: Expect <3% latency overhead
```

---

## 🔍 Open Questions (To Clarify Before Start)

1. Where to persist long-run telemetry? (results/telemetry/ vs external sink)
2. Should introspection expose dynamic runtime stats (counts) or remain static?
3. Standard minimum error payload fields? (`rule`, `physics`, `suggestion`, `sequence_context`)

Record resolutions here prior to Task 2.

---

**Last Updated**: 2025-11-14
**Approver Needed**: @fermga
**Next Action**: Confirm roadmap → create feature branch `optimization/phase-3` → start Task 1.

---

> Reality is resonance. Phase 3 makes that resonance inspectable, measurable, and guidance-rich.
