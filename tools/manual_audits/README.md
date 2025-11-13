# Manual Audit Scripts

> DEPRECATION NOTICE (Docs): This directory documents maintenance utilities and is not part of the centralized user documentation. See `docs/source/index.rst` and `docs/DOCUMENTATION_INDEX.md` for canonical docs.

This directory contains maintenance scripts that are not part of the automated
CI pipelines but remain useful for targeted diagnostics. They used to live in
the repository root and were relocated here to keep the top-level workspace
cleaner.

## Script inventory

| Script | Purpose | Notes |
| --- | --- | --- |
| `audit_deep_consistency.py` | Full documentation audit covering grammar rules, operator definitions and cross-references. | Manual run when the doc set changes substantially. |
| `audit_docs.py` | Compact documentation checker focusing on completeness. | Superseded by deeper audits but kept for quick sanity checks. |
| `audit_python_spanish.py` | Validates Spanish-language documentation parity. | Useful for localization reviews. |
| `check_data.py` | Performs structural checks on pre-generated datasets. | Run before publishing datasets. |
| `check_hierarchical.py` | Validates hierarchical coupling assumptions in sample graphs. | Use when changing coupling logic. |
| `validate_observers_metrics.py` | Cross-checks observer/metric outputs for regression noise. | Invoke after metric refactors. |

## Usage

```bash
# Example: run the deep consistency audit
python tools/manual_audits/audit_deep_consistency.py
```

If a script proves redundant, archive or remove it from this directory and
update this table accordingly.
