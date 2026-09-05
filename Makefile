.PHONY: help clean test examples docs all hello music network sdk visualization lint format security security-audit security-setup dev-setup dev-test validate riemann-benchmark factorization-full-spectrum self-optimize self-optimize-validate

SELF_OPT_MANIFEST ?= tests/data/self_optimization/test_run/_manifest.json
SELF_OPT_MANIFEST_SUMMARY ?= tests/data/self_optimization/test_run/_manifest_summary.json
SELF_OPT_OUTPUT ?= results/self_optimization
SELF_OPT_SUMMARY ?= results/self_opt_summary.json
SELF_OPT_VALIDATION_REPORT ?= results/self_optimization_validation.json

help:
	@echo "TNFR development targets"
	@echo "  test        Run the canonical core test areas and Riemann benchmark"
	@echo "  examples    Run the maintained introductory examples"
	@echo "  docs        Validate and build the documentation site"
	@echo "  validate    Run documentation and SDK validation"
	@echo "  clean       Remove generated repository artifacts"
	@echo "  dev-setup   Install the supported minimal development extra"
	@echo "  security    Run dependency and source security checks"

clean:
	@python scripts/clean_repository.py

test:
	@python -m pytest tests/core_physics tests/operators tests/physics -v --tb=short
	@python benchmarks/riemann_program.py

riemann-benchmark:
	@python benchmarks/riemann_program.py

factorization-full-spectrum:
	@python factorization-lab/benchmarks/full_spectrum_factorization.py

self-optimize:
	@python scripts/run_self_optimization.py --manifest $(SELF_OPT_MANIFEST) --manifest-summary $(SELF_OPT_MANIFEST_SUMMARY) --output-dir $(SELF_OPT_OUTPUT) --summary $(SELF_OPT_SUMMARY)

self-optimize-validate:
	@python scripts/run_self_opt_validation.py --payload-root $(SELF_OPT_OUTPUT) --report $(SELF_OPT_VALIDATION_REPORT)

examples: hello music network sdk

hello:
	@python examples/01_foundations/01_hello_world.py

music:
	@python examples/01_foundations/02_musical_resonance.py

network:
	@python examples/01_foundations/03_network_formation.py

sdk:
	@python examples/01_foundations/10_simplified_sdk_showcase.py

visualization:
	@python examples/01_foundations/09_visualization_suite.py

docs:
	@python scripts/verify_internal_references.py --ci
	@python scripts/check_documentation.py
	@python scripts/prepare_docs.py
	@python -m mkdocs build --strict

lint:
	@python -m flake8 src --count --show-source --statistics

format:
	@python -m black src examples scripts

security security-audit:
	@python -m pip_audit
	@python -m bandit -r src -c bandit.yaml

security-setup:
	@python -m pip install -e ".[security]"

all: clean test examples docs

dev-setup:
	@python -m pip install -e ".[dev-minimal]"

dev-test:
	@python -m pytest --cov=src/tnfr --cov-report=html

validate:
	@python -c "import tnfr; print('TNFR import: OK')"
	@python scripts/verify_internal_references.py --ci
	@python scripts/check_documentation.py
	@python -m pytest tests/sdk -q
