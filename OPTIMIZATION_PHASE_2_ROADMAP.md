# TNFR Optimization Phase 2 Roadmap

**Status**: 🟢 ACTIVE  
**Started**: 2025-11-14  
**Phase 1**: ✅ COMPLETADO (5/5 tasks, 11.5h, commits 71ae4285c..a5db75af1)

---

## 🎯 Phase 2 Objectives

**Focus**: Code quality, performance optimization, and comprehensive documentation updates

**Duration Estimate**: 12-15 hours  
**Priority**: Medium-High  
**Success Criteria**: 
- All modules follow consistent patterns
- Performance benchmarks established
- Documentation fully aligned with code splits
- Test coverage >85%

---

## 📋 Task List

### Task 1: Split Remaining Large Files (4-5h) 🔴 HIGH PRIORITY

**Objective**: Continue modularization of oversized files

**Targets**:
1. **`src/tnfr/operators/definitions.py`** (~1,800 lines)
   - Split by operator: one file per operator
   - Create `definitions/` directory
   - Preserve `definitions.py` as facade
   - Operators: AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH

2. **`src/tnfr/validation/compatibility.py`** (~800 lines)
   - Split into: types, levels, matchers, utils
   - Estimated: 4 files + facade

3. **`src/tnfr/dynamics/step.py`** (~600 lines) - if needed
   - Split into: core, telemetry, observers
   - Estimated: 3 files + facade

**Deliverables**:
- [ ] `scripts/split_definitions.py` automation script
- [ ] 13+ operator files in `definitions/` directory
- [ ] `definitions.py` facade with 100% backward compatibility
- [ ] Comprehensive tests (100+ tests for definitions)
- [ ] Commit: "refactor: split definitions.py into per-operator modules (Phase 2, Task 1)"

**Estimated Time**: 4-5 hours  
**Risk**: Medium (definitions.py is heavily imported)

---

### Task 2: Performance Benchmarking & Optimization (3-4h) 🟡 MEDIUM PRIORITY

**Objective**: Establish performance baselines and optimize hot paths

**Targets**:
1. **Benchmark Suite Enhancement**
   - Add memory profiling to existing benchmarks
   - Create comparison baseline (pre-split vs post-split)
   - Focus areas: grammar validation, metrics computation, step()

2. **Optimization Opportunities**
   - Profile import times (after splits)
   - Identify hot loops in grammar_core.py
   - Cache expensive computations (phase verification, operator sets)
   - Consider lazy imports for heavy modules

3. **Performance Tests**
   - Add `tests/performance/test_grammar_performance.py` (if not exists)
   - Add `tests/performance/test_metrics_performance.py`
   - Establish regression thresholds

**Deliverables**:
- [ ] Performance benchmark results (baseline.json)
- [ ] Optimization patches (if >10% improvement found)
- [ ] Performance test suite
- [ ] Commit: "perf: establish performance baselines and optimize hot paths (Phase 2, Task 2)"

**Estimated Time**: 3-4 hours  
**Risk**: Low

---

### Task 3: Documentation Alignment (2-3h) 🟢 LOW PRIORITY

**Objective**: Update all documentation to reflect Phase 1 splits

**Targets**:
1. **Update ARCHITECTURE.md**
   - Document new module structure (metrics, grammar)
   - Add architecture diagrams (Mermaid)
   - Explain facade pattern usage

2. **Update DOCUMENTATION_INDEX.md**
   - Add references to new modules
   - Update code navigation guides
   - Add split scripts to tools section

3. **Update API docs** (`docs/source/api/`)
   - Regenerate with Sphinx (if needed)
   - Ensure new modules appear correctly
   - Add migration notes for importers

4. **Update CONTRIBUTING.md**
   - Add guidance on modular structure
   - Explain when to create new modules vs extend existing
   - Document facade pattern for backward compatibility

**Deliverables**:
- [ ] ARCHITECTURE.md updated with module diagrams
- [ ] DOCUMENTATION_INDEX.md reflects all new files
- [ ] API docs regenerated (if needed)
- [ ] CONTRIBUTING.md with modular guidance
- [ ] Commit: "docs: align documentation with Phase 1 modular architecture (Phase 2, Task 3)"

**Estimated Time**: 2-3 hours  
**Risk**: Low

---

### Task 4: Test Coverage Improvements (2-3h) 🟡 MEDIUM PRIORITY

**Objective**: Increase test coverage and fix known test failures

**Targets**:
1. **Fix Known Failures**
   - `test_sha_grammar_validation.py`: 6 failing tests (SHA-specific logic)
   - `test_unit/dynamics/test_grammar.py`: 9 failing tests (fallback logic)
   - Investigate root causes, implement fixes or mark as known issues

2. **Coverage Gaps**
   - Add tests for new module boundaries (imports, exports)
   - Add integration tests for split modules working together
   - Focus on edge cases in grammar_application.py

3. **Property-Based Tests**
   - Install `hypothesis` in test-env
   - Enable `tests/property/test_grammar_invariants.py`
   - Add property tests for metrics modules

**Deliverables**:
- [ ] SHA grammar tests fixed or documented
- [ ] Dynamics grammar tests fixed or documented
- [ ] Hypothesis installed, property tests enabled
- [ ] Coverage report showing >85% for operators/
- [ ] Commit: "test: fix known failures and improve coverage (Phase 2, Task 4)"

**Estimated Time**: 2-3 hours  
**Risk**: Medium (may uncover deeper issues)

---

### Task 5: Code Quality & Linting (1-2h) 🟢 LOW PRIORITY

**Objective**: Ensure consistent code style and remove lint warnings

**Targets**:
1. **Lint Cleanup**
   - Address "imported but unused" warnings in new modules
   - Fix line length violations (79 char limit)
   - Remove trailing whitespace
   - Add missing docstrings where needed

2. **Type Hints**
   - Add comprehensive type hints to new modules
   - Run `mypy` on operators/ directory
   - Fix type inconsistencies

3. **Code Formatting**
   - Run `black` on all new modules
   - Ensure consistent import ordering (isort)
   - Verify docstring format (Google style)

**Deliverables**:
- [ ] Zero lint warnings in new modules
- [ ] Type hints complete (mypy passing)
- [ ] Code formatted (black, isort)
- [ ] Commit: "style: clean up linting and improve type hints (Phase 2, Task 5)"

**Estimated Time**: 1-2 hours  
**Risk**: Low

---

## 📊 Phase 2 Metrics

### Success Criteria

| Metric | Target | Current (Phase 1 End) |
|--------|--------|------------------------|
| Module count | 280-290 | 259 |
| Largest file | <1,000 lines | grammar_core.py (882) ✅ |
| Test coverage (operators/) | >85% | ~75% (est.) |
| Performance (vs baseline) | ±5% | TBD |
| Lint warnings | 0 | ~50 (est.) |
| Documentation completeness | 100% | ~90% |

### Deliverables Checklist

- [ ] **Task 1**: definitions.py split complete
- [ ] **Task 2**: Performance baselines established
- [ ] **Task 3**: Documentation fully aligned
- [ ] **Task 4**: Test coverage >85%
- [ ] **Task 5**: Code quality perfect

**Total Estimated Time**: 12-17 hours  
**Commits Expected**: 5 (one per task)

---

## 🚀 Execution Strategy

### Order of Operations

1. **Start with Task 1** (definitions.py split) - Highest impact, foundational
2. **Then Task 4** (test coverage) - Validate split work, fix failures
3. **Then Task 2** (performance) - Measure impact of splits
4. **Then Task 5** (code quality) - Polish before documentation
5. **Finally Task 3** (documentation) - Reflect final state

### Risk Mitigation

**Task 1 (definitions.py)**:
- Create backup first (like grammar.py)
- Use automation script (like split_grammar.py)
- Test extensively before commit
- Expect 6-8 import fix iterations

**Task 4 (test fixes)**:
- Document "won't fix" issues clearly
- Separate quick fixes from research-needed
- Don't block Phase 2 on complex failures

### Branch Strategy

Continue on `main` (Phase 1 merged) OR create `optimization/phase-2` branch if preferred for safety.

**Recommendation**: Use `optimization/phase-2` branch, merge when complete.

---

## 📈 Expected Outcomes

After Phase 2 completion:

1. **Modularity**: All large files (<1,000 lines each)
2. **Performance**: Baseline established, optimizations applied
3. **Documentation**: 100% aligned with code structure
4. **Tests**: >85% coverage, known issues documented
5. **Quality**: Zero lint warnings, complete type hints

**Repository Health**: 100/100 maintained throughout  
**Backward Compatibility**: 100% preserved  
**TNFR Invariants**: All 10 preserved

---

## 🔄 Future Phases (Preview)

**Phase 3** (Potential, ~10h):
- Enhanced error messages
- Interactive tools (CLI validators)
- Example gallery expansion
- Tutorial improvements

**Phase 4** (Potential, ~8h):
- CI/CD pipeline enhancements
- Release automation
- Performance regression tests
- Integration with external tools

---

## 📝 Notes

**Phase 1 Accomplishments** (Reference):
- ✅ 70+ new tests added
- ✅ 13 new modules created (5 metrics + 8 grammar)
- ✅ 2 large files split (metrics.py, grammar.py)
- ✅ Dependabot configured
- ✅ 5 commits, 11.5h total
- ✅ Health 100/100 maintained

**Phase 2 Philosophy**:
- Continue incremental, validated changes
- Prioritize backward compatibility
- Document everything
- Test exhaustively
- Maintain TNFR physics fidelity

---

**Last Updated**: 2025-11-14  
**Status**: 🟢 READY TO START  
**Approver**: @fermga

**Next Action**: Create `optimization/phase-2` branch and start Task 1 (definitions.py split)
