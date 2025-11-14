# Tools Directory Logging Audit

**Date**: 2025-01-XX  
**Context**: Repository-wide print statement migration to structured logging

## Audit Summary

**Total files audited**: 15+ scripts  
**Print statements found**: 200+  
**Recommendation**: **NO CONVERSION** — All prints are intentional CLI output

## Classification

All print statements in `tools/` fall into these categories:

### 1. CLI Progress Indicators

- **Examples**: `sequence_explorer.py`, `sync_documentation.py`
- **Pattern**: Progress messages like `"[1/5] Auditing grammar.py..."`
- **Purpose**: User-facing feedback for long-running operations
- **Status**: ✅ Keep as-is (intentional stdout)

### 2. Formatted Analysis Reports

- **Examples**: `analyze_universality.py`, `test_nested_fractality.py`
- **Pattern**: Tables, headers, scientific results with ASCII art
- **Purpose**: Human-readable terminal output for analysis scripts
- **Status**: ✅ Keep as-is (intentional stdout)

### 3. Interactive CLI Prompts

- **Examples**: `sequence_explorer.py` (interactive mode)
- **Pattern**: User instructions, menu options, input prompts
- **Purpose**: Core functionality of interactive tools
- **Status**: ✅ Keep as-is (required for CLI)

### 4. Demonstration Output

- **Examples**: `fields_demo.py`
- **Pattern**: Summary statistics, visualization results
- **Purpose**: Educational/demonstration script output
- **Status**: ✅ Keep as-is (intentional demonstration)

### 5. Error Messages to stderr

- **Examples**: `bandit_to_sarif.py`, `sequence_explorer.py`
- **Pattern**: `print(..., file=sys.stderr)`
- **Purpose**: Error reporting (already using correct stream)
- **Status**: ✅ Already correct (using stderr for errors)

## Rationale

Unlike core modules (`src/tnfr/`), tools are **user-facing CLI scripts** where:

1. Print statements are **intentional interface** (not debug artifacts)
2. Output is meant for direct human consumption
3. Redirection to logging would **break functionality** (users expect stdout)
4. Scripts document their purpose as "command-line tools" in docstrings

## Comparison with Previous Work

| Category | Print Usage | Recommendation |
|----------|-------------|----------------|
| **Core modules** (`src/tnfr/structural.py`, `src/tnfr/visualization/cascade_viz.py`) | Debug/status messages | ✅ **Converted to logging** |
| **Tutorials** (`src/tnfr/tutorials/`) | Educational narration | ✅ **Documented as intentional** |
| **Tools** (`tools/*.py`) | CLI output/interface | ✅ **Keep as-is** (correct usage) |

## Sample Files Reviewed

1. **sequence_explorer.py** (345 lines)
   - Interactive CLI for sequence analysis
   - Prints: Headers, tables, validation results, visualizations paths
   - Verdict: All prints are CLI interface

2. **sync_documentation.py** (415 lines)
   - Documentation synchronization tool
   - Prints: Progress indicators, audit reports, issue summaries
   - Verdict: All prints are progress feedback

3. **fields_demo.py** (102 lines)
   - Physics demonstration script
   - Prints: Summary statistics for computed fields
   - Verdict: All prints are demonstration output

4. **test_nested_fractality.py**
   - Scientific analysis script
   - Prints: Hypothesis, test steps, formatted tables, conclusions
   - Verdict: All prints are analysis report output

## Conclusion

**NO ACTION REQUIRED** — The tools/ directory correctly uses print statements for their intended purpose (CLI output). Converting these to logging would be inappropriate and break user-facing functionality.

## Documentation Added

This audit serves as documentation that tools/ scripts intentionally use print statements and were **reviewed and approved** during the logging migration effort.

---

**Cross-reference**: See `src/tnfr/tutorials/structural_metabolism.py` and `src/tnfr/tutorials/autonomous_evolution.py` for similar "intentional print" documentation in tutorial modules.
