.PHONY: docs stubs stubs-check

docs:
@sphinx-build -b html docs/source docs/_build/html

stubs:
@echo "Generating .pyi stub files..."
@python scripts/generate_stubs.py

stubs-check:
@echo "Checking for missing .pyi stub files..."
@python scripts/generate_stubs.py --check
