# Contributing

This project uses English exclusively for code identifiers and docstrings. When contributing:

- Use descriptive English names for all variables, functions, classes and modules.
- Write docstrings and comments in English.
- Update existing code to maintain this convention when modifying files.

## Testing

Run the full quality gate from the project root with:

```bash
./scripts/run_tests.sh
```

The helper sets up `PYTHONPATH` and orchestrates the tooling invoked by the
continuous integration workflow:

- `pydocstyle` for targeted docstring style checks.
- `coverage run --source=src -m pytest` to execute the test suite under
  coverage.
- `coverage report -m` to display the aggregate coverage summary.
- `vulture --min-confidence 80 src tests` to detect unused code paths.

To forward additional flags to `pytest`, append them after `--`, e.g.
`./scripts/run_tests.sh -- -k coherence`.

The [README Tests section](README.md#tests) repeats these instructions so that
contributors can find them quickly while browsing the project overview.

Make sure to honor the patterns in `.gitignore` so that dependency and build
artifacts (e.g., `node_modules/` or `dist/`) are not committed.
