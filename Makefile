.PHONY: docs stubs stubs-check stubs-check-sync stubs-sync verify-refs verify-refs-verbose reproduce reproduce-verify help

help:
	@echo "Available targets:"
	@echo "  docs                - Build Sphinx documentation"
	@echo "  stubs               - Generate missing .pyi stub files"
	@echo "  stubs-check         - Check for missing .pyi stub files"
	@echo "  stubs-check-sync    - Check if .pyi stub files are synchronized with .py files"
	@echo "  stubs-sync          - Regenerate outdated .pyi stub files"
	@echo "  verify-refs         - Verify internal markdown references"
	@echo "  verify-refs-verbose - Verify internal references with verbose output"
	@echo "  reproduce           - Run benchmarks with deterministic seeds and generate checksums"
	@echo "  reproduce-verify    - Verify checksums against existing manifest"

docs:
	@sphinx-build -b html docs/source docs/_build/html

stubs:
	@echo "Generating .pyi stub files..."
	@python scripts/generate_stubs.py

stubs-check:
	@echo "Checking for missing .pyi stub files..."
	@python scripts/generate_stubs.py --check

stubs-check-sync:
	@echo "Checking if .pyi stub files are synchronized..."
	@python scripts/generate_stubs.py --check-sync

stubs-sync:
	@echo "Synchronizing outdated .pyi stub files..."
	@python scripts/generate_stubs.py --sync

verify-refs:
	@echo "Verifying internal markdown references..."
	@python scripts/verify_internal_references.py --ci

verify-refs-verbose:
	@echo "Verifying internal markdown references (verbose)..."
	@python scripts/verify_internal_references.py --verbose

reproduce:
	@echo "Running reproducible benchmarks with deterministic seeds..."
	@python scripts/run_reproducible_benchmarks.py

reproduce-verify:
	@echo "Verifying checksums against manifest..."
	@python scripts/run_reproducible_benchmarks.py --verify artifacts/manifest.json
