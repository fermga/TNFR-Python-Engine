.PHONY: docs stubs stubs-check stubs-check-sync stubs-sync

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
