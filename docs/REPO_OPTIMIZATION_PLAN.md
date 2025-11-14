# Repository Optimization Plan

This guide consolidates the highest-impact workflows for keeping the TNFR Python Engine
lean and reproducible.

## 1. Generated Artifacts

Use `make clean` (new target) to remove benchmark outputs, notebook exports, and cache
folders. The target calls `scripts/clean_generated_artifacts.py`, which deletes:

- `results/`, `outputs/`, `benchmarks/results/`
- `examples/output/` (generated notebooks/scripts)
- `validation_outputs/`, `artifacts/`, `profiles/`, `dist-test/`, `site/`
- Python caches (`__pycache__`, `*.pyc`, `*.pyo`, `*.pyd`)

### Commands

```bash
make clean
# Windows fallback (PowerShell)
./make.cmd clean
```

This runs cross-platform (the script uses Python) and is safe to execute repeatedly.

## 2. Targeted Test Runs

- **`make smoke-tests`** (PowerShell: `./make.cmd smoke-tests`): runs `pytest` on the curated bundle below (examples + telemetry) in one command; ideal before commits.
- **VS Code task**: `Terminal → Run Task → Run focused tests (examples + telemetry)` simply delegates to `./make.cmd smoke-tests`, so you get the same curated bundle without leaving the editor.
- **Unit mathematics + telemetry**: `pytest tests/unit/mathematics tests/unit/operators/test_telemetry_warnings_extended.py`
- **Example smoke tests**: `pytest tests/examples/test_atom_atlas_minimal.py tests/examples/test_periodic_table_basic.py`
- **Focused U6 suite**: `pytest tests/examples/test_u6_sequential_demo.py`

Documenting these bundles helps keep CI and local runs fast while still covering
high-risk areas.

## 3. Dependency Profiles

- **Core install**: `pip install .`
- **Dev minimal**: `pip install -e ".[dev-minimal]"`
- **Docs**: `pip install -e ".[docs]"`
- **Full test battery**: `pip install -e ".[test-all]"`

Installing only what a task needs reduces environment churn and avoids conflicting
binary wheels, especially on Windows.

## 4. Notebook / Report Generation

When exporting notebooks (nbconvert tasks listed in VS Code), write outputs under
`results/reports/`—already ignored. After verifying a report, use `make clean` or manually
remove the run-specific folder to prevent Git noise.

## 5. Large Benchmarking Runs

Benchmark scripts and notebooks emit JSONL/PNG assets under `benchmarks/results/`.
Keep those out of version control (now enforced via `.gitignore`). Before pushing a
branch that ran benchmarks locally, execute `make clean` or delete the directory to
avoid stray multi-megabyte files.

---

**Reminder**: keep generated assets outside the repo tree or ensure they are ignored
before running long experiments. This keeps clones fast, diffs readable, and makes CI
runs deterministic.
