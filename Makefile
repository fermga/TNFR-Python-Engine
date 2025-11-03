.PHONY: docs stubs stubs-check stubs-check-sync stubs-sync help

help:
	@echo "Available targets:"
	@echo "  docs             - Build Sphinx documentation"
	@echo "  stubs            - Generate missing .pyi stub files"
	@echo "  stubs-check      - Check for missing .pyi stub files"
	@echo "  stubs-check-sync - Check if .pyi stub files are synchronized with .py files"
	@echo "  stubs-sync       - Regenerate outdated .pyi stub files"

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
