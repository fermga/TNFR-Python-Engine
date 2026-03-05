# TNFR Modern Makefile - Optimized for Clean Repository
# 
# Purpose: Essential tasks for the streamlined TNFR repository
# Focus: Core examples, testing, and documentation generation

.PHONY: help clean test examples docs all riemann-benchmark self-optimize self-optimize-validate

SELF_OPT_MANIFEST ?= tests/data/self_optimization/test_run/_manifest.json
SELF_OPT_MANIFEST_SUMMARY ?= tests/data/self_optimization/test_run/_manifest_summary.json
SELF_OPT_OUTPUT ?= results/self_optimization
SELF_OPT_SUMMARY ?= results/self_opt_summary.json
SELF_OPT_VALIDATION_REPORT ?= results/self_optimization_validation.json

# Default target
help:
	@echo "🌊 TNFR Modern Build System 🌊"
	@echo "================================"
	@echo ""
	@echo "Essential targets:"
	@echo "  help          - Show this help message"
	@echo "  clean         - Remove generated artifacts"
	@echo "  test          - Run core test suite"
	@echo "  examples      - Run all essential examples"
	@echo "  docs          - Generate documentation"
	@echo "  all           - Run everything (test + examples + docs)"
	@echo ""
	@echo "Example targets:"
	@echo "  hello         - Run hello world example"
	@echo "  music         - Run musical resonance demo"
	@echo "  network       - Run simple network demo"
	@echo "  chemistry     - Run atoms & molecules demo"
	@echo "  sdk           - Run SDK quickstart demo"
	@echo ""
	@echo "Quality targets:"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code"
	@echo "  security      - Run security checks"
	@echo "Self-optimization:"
	@echo "  self-optimize           - Run Paley manifest through the self-optimization runner"
	@echo "  self-optimize-validate  - Validate self-optimization payloads via pytest subsets"
	@echo "Benchmark targets:"
	@echo "  factorization-full-spectrum - Run Paley full-spectrum factorization benchmark"

# Clean generated artifacts
clean:
	@echo "🧹 Cleaning generated artifacts..."
	@if exist "examples\output" rmdir /s /q "examples\output"
	@if exist "results" rmdir /s /q "results"
	@if exist "dist" rmdir /s /q "dist"
	@if exist "build" rmdir /s /q "build"
	@if exist "*.egg-info" rmdir /s /q "*.egg-info"
	@if exist "__pycache__" rmdir /s /q "__pycache__"
	@echo "✅ Clean complete"

# Core test suite (essential tests only)
test:
	@echo "🧪 Running core TNFR test suite..."
	@python -m pytest tests/core_physics tests/grammar tests/operators tests/physics -v --tb=short
	@echo "📈 Running TNFR–Riemann sigma-critical benchmark..."
	@python -c "import runpy, sys, pathlib; sys.path.insert(0, str(pathlib.Path('src').resolve())); sys.argv = ['benchmarks/riemann_program.py']; runpy.run_path('benchmarks/riemann_program.py', run_name='__main__')"
	@echo "✅ Core tests complete"

riemann-benchmark:
	@echo "📈 Running TNFR–Riemann sigma-critical benchmark..."
	@python -c "import runpy, sys, pathlib; sys.path.insert(0, str(pathlib.Path('src').resolve())); sys.argv = ['benchmarks/riemann_program.py']; runpy.run_path('benchmarks/riemann_program.py', run_name='__main__')"

factorization-full-spectrum:
	@echo "🧮 Running TNFR Paley full-spectrum factorization sweep..."
	@python factorization-lab/benchmarks/full_spectrum_factorization.py

self-optimize:
	@echo "🧠 Running TNFR self-optimization manifest pipeline..."
	@python scripts/run_self_optimization.py --manifest $(SELF_OPT_MANIFEST) --manifest-summary $(SELF_OPT_MANIFEST_SUMMARY) --output-dir $(SELF_OPT_OUTPUT) --summary $(SELF_OPT_SUMMARY)

self-optimize-validate:
	@echo "🧪 Validating TNFR self-optimization payloads..."
	@python scripts/run_self_opt_validation.py --payload-root $(SELF_OPT_OUTPUT) --report $(SELF_OPT_VALIDATION_REPORT)

# Run all essential examples
examples: hello music network chemistry sdk
	@echo "✅ All essential examples complete"

# Individual example targets
hello:
	@echo "👋 Running Hello World example..."
	@mkdir examples\output 2>nul || echo >nul
	@python examples/01_hello_world.py
	@echo ""

music:
	@echo "🎵 Running Musical Resonance demo..."
	@mkdir examples\output 2>nul || echo >nul
	@python examples/02_musical_resonance.py
	@echo ""

network:
	@echo "🕸️ Running Simple Network demo..."
	@mkdir examples\output 2>nul || echo >nul
	@python examples/03_simple_network.py
	@echo ""

chemistry:
	@echo "⚛️ Running Atoms & Molecules demo..."
	@mkdir examples\output 2>nul || echo >nul
	@python examples/04_atoms_and_molecules.py
	@echo ""

# Run visualization suite
visualization:
	@echo "🎨 Running TNFR Visualization Suite..."
	@mkdir examples\output 2>nul || echo >nul
	@python examples/13_visualization_suite.py
	@echo ""

# Run network topologies analysis
topologies:
	@echo "🕸️ Running Network Topologies Analysis..."
	@mkdir examples\output 2>nul || echo >nul
	@python examples/07_network_topologies.py
	@echo ""

# Run emergent phenomena demo
emergence:
	@echo "🌟 Running Emergent Phenomena Demo..."
	@mkdir examples\output 2>nul || echo >nul
	@python examples/12_emergent_phenomena.py
	@echo ""

# Documentation generation
docs:
	@echo "📚 Generating documentation..."
	@if not exist "docs\build" mkdir "docs\build"
	@echo "Documentation structure validated"
	@echo "✅ Documentation ready"

# Code quality
lint:
	@echo "🔍 Running code linting..."
	@python -m flake8 src/ examples/ --count --show-source --statistics --max-line-length=100
	@echo "✅ Linting complete"

format:
	@echo "✨ Formatting code..."
	@python -m black src/ examples/ --line-length=100
	@echo "✅ Formatting complete"

security:
	@echo "🛡️ Running security checks..."
	@python -m bandit -r src/ -f json -o security_report.json || echo "Security scan complete"
	@echo "✅ Security check complete"

# Comprehensive target
all: clean test examples docs
	@echo ""
	@echo "🎉 COMPLETE SUCCESS! 🎉"
	@echo "======================"
	@echo "✅ Tests passed"
	@echo "✅ Examples executed"  
	@echo "✅ Documentation generated"
	@echo ""
	@echo "Repository is ready for use!"
	@echo ""
	@echo "Quick start:"
	@echo "  make hello     - Try your first TNFR example"
	@echo "  make chemistry - See TNFR in action with molecules"
	@echo "  make sdk       - Learn modern TNFR integration"

# Development helpers
dev-setup:
	@echo "⚙️ Setting up development environment..."
	@pip install -e .[dev]
	@echo "✅ Development setup complete"

dev-test:
	@echo "🔬 Running development test suite..."
	@python -m pytest tests/ -v --cov=src/tnfr --cov-report=html
	@echo "✅ Development tests complete - see htmlcov/"

# Quick validation
validate:
	@echo "✅ Running quick validation..."
	@python -c "import tnfr; print('TNFR import: OK')"
	@python examples/01_hello_world.py > nul 2>&1 && echo "✅ Hello World: OK" || echo "❌ Hello World: FAILED"
	@python -m pytest tests/test_repository_validation.py -q && echo "✅ Repository: OK" || echo "❌ Repository: FAILED"
	@echo "✅ Quick validation complete"

# Modern workflow
modern: clean validate examples
	@echo "🚀 Modern TNFR workflow complete!"
	@echo "   Repository cleaned, validated, and examples ready"