.PHONY: clean-scratch docs stubs stubs-check stubs-check-sync stubs-sync verify-refs verify-refs-verbose reproduce reproduce-verify security-audit security-audit-json help clean smoke-tests report-tetrad report-phase-gated report-atoms-molecules report-triatomic-atlas report-molecule-atlas report-operator-completeness-classic report-operator-completeness-print report-interaction-sequences-classic report-interaction-sequences-print report-emergent-particles report-force-study-plots report-fundamental-particles-classic report-fundamental-particles-print report-all-classic report-all-print report-atom-atlas-script report-periodic-table-script report-particle-atlas-u6 report-periodic-table-classic

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
	@echo "  security-audit      - Run pip-audit to scan for dependency vulnerabilities"
	@echo "  security-audit-json - Run pip-audit and generate JSON report"
	@echo "  smoke-tests         - Run curated pytest bundle for fast validation"
	@echo "  clean               - Remove generated artifacts (results/, outputs/, examples/output/, etc.)"
	@echo "  report-*            - Export notebooks or run scripts via nbconvert (see Makefile for list)"
	@echo "  atom-atlas-script   - Run examples/atom_atlas.py via Python"

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

security-audit:
	@echo "Running pip-audit to scan for dependency vulnerabilities..."
	@./scripts/run_pip_audit.sh

security-audit-json:
	@echo "Running pip-audit and generating JSON report..."
	@./scripts/run_pip_audit.sh --json

clean:
	@echo "Removing generated artifacts (results/, outputs/, validation outputs, caches)..."
	@python scripts/clean_generated_artifacts.py

clean-scratch:
	@echo "Cleaning debug/scratch files..."
	@rm -rf debug_scratch/
	@echo "Removed debug_scratch directory"
smoke-tests:
	@echo "Running curated pytest bundle (examples + telemetry)..."
	@python -m pytest -q \
		tests/examples/test_u6_sequential_demo.py \
		tests/unit/operators/test_telemetry_warnings_extended.py \
		tests/examples/test_atom_atlas_minimal.py \
		tests/examples/test_periodic_table_basic.py \
		tests/test_precision_walk_dashboard_artifact.py

report-tetrad:
	@echo "Exporting Force Fields Tetrad notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Force_Fields_Tetrad_Exploration.ipynb

report-atoms-molecules:
	@echo "Exporting Atoms & Molecules Study notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/TNFR_Atoms_And_Molecules_Study.ipynb

report-phase-gated:
	@echo "Exporting Phase-Gated Coupling Demo notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/TNFR_Phase_Gated_Coupling_Demo.ipynb

report-triatomic-atlas:
	@echo "Exporting Triatomic Atlas notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=1200 --output-dir results/reports notebooks/TNFR_Triatomic_Atlas.ipynb

report-molecule-atlas:
	@echo "Exporting Molecule Atlas notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=1200 --output-dir results/reports notebooks/TNFR_Molecule_Atlas.ipynb

report-operator-completeness:
	@echo "Exporting Operator Completeness notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=1200 --output-dir results/reports notebooks/Operator_Completeness_Search.ipynb

report-interaction-sequences:
	@echo "Exporting Interaction Sequences notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Interaction_Sequences.ipynb

report-emergent-particles:
	@echo "Exporting Emergent Particles notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Emergent_Particles_From_TNFR.ipynb

report-fundamental-particles:
	@echo "Exporting Fundamental Particles Atlas notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Fundamental_Particles_TNFR_Atlas.ipynb

atom-atlas-script:
	@echo "Running atom_atlas.py script..."
	@mkdir -p examples/output
	@python examples/atom_atlas.py

periodic-table-script:
	@echo "Running periodic_table_atlas.py script..."
	@mkdir -p examples/output
	@python examples/periodic_table_atlas.py

triatomic-atlas-script:
	@echo "Running triatomic_atlas.py script..."
	@mkdir -p examples/output
	@python examples/triatomic_atlas.py

molecule-atlas-script:
	@echo "Running molecule_atlas.py script..."
	@mkdir -p examples/output
	@python examples/molecule_atlas.py

phase-gated-script:
	@echo "Running phase_gated_coupling_demo.py script..."
	@mkdir -p examples/output
	@python examples/phase_gated_coupling_demo.py

elements-signature-script:
	@echo "Running elements_signature_study.py script..."
	@mkdir -p examples/output
	@python examples/elements_signature_study.py

report-operator-completeness-print:
	@echo "Exporting Operator Completeness notebook (print-friendly)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template lab --HTMLExporter.theme=light --HTMLExporter.exclude_input=True --HTMLExporter.exclude_input_prompt=True --HTMLExporter.exclude_output_prompt=True --ExecutePreprocessor.timeout=1200 --output-dir results/reports notebooks/Operator_Completeness_Search.ipynb

report-interaction-sequences-print:
	@echo "Exporting Interaction Sequences notebook (print-friendly)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template lab --HTMLExporter.theme=light --HTMLExporter.exclude_input=True --HTMLExporter.exclude_input_prompt=True --HTMLExporter.exclude_output_prompt=True --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Interaction_Sequences.ipynb

report-fundamental-particles-print:
	@echo "Exporting Fundamental Particles Atlas notebook (print-friendly)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template lab --HTMLExporter.theme=light --HTMLExporter.exclude_input=True --HTMLExporter.exclude_input_prompt=True --HTMLExporter.exclude_output_prompt=True --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Fundamental_Particles_TNFR_Atlas.ipynb

report-particle-atlas-u6:
	@echo "Exporting Particle Atlas U6 Sequential notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=1200 --output-dir results/reports notebooks/TNFR_Particle_Atlas_U6_Sequential.ipynb

report-periodic-table-classic:
	@echo "Exporting Periodic Table Atlas notebook (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=1500 --output-dir results/reports notebooks/TNFR_Periodic_Table_Atlas.ipynb

force-study-plots:
	@echo "Generating force study plots..."
	@python benchmarks/plot_force_study_summaries.py

report-all-classic:
	@echo "Exporting all TNFR reports (classic template)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Force_Fields_Tetrad_Exploration.ipynb
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Emergent_Particles_From_TNFR.ipynb
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Fundamental_Particles_TNFR_Atlas.ipynb
	@python -m nbconvert --to html --execute --template classic --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Interaction_Sequences.ipynb

report-all-print:
	@echo "Exporting all TNFR reports (print-friendly)..."
	@mkdir -p results/reports
	@python -m nbconvert --to html --execute --template lab --HTMLExporter.theme=light --HTMLExporter.exclude_input=True --HTMLExporter.exclude_input_prompt=True --HTMLExporter.exclude_output_prompt=True --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Force_Fields_Tetrad_Exploration.ipynb
	@python -m nbconvert --to html --execute --template lab --HTMLExporter.theme=light --HTMLExporter.exclude_input=True --HTMLExporter.exclude_input_prompt=True --HTMLExporter.exclude_output_prompt=True --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Emergent_Particles_From_TNFR.ipynb
	@python -m nbconvert --to html --execute --template lab --HTMLExporter.theme=light --HTMLExporter.exclude_input=True --HTMLExporter.exclude_input_prompt=True --HTMLExporter.exclude_output_prompt=True --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Fundamental_Particles_TNFR_Atlas.ipynb
	@python -m nbconvert --to html --execute --template lab --HTMLExporter.theme=light --HTMLExporter.exclude_input=True --HTMLExporter.exclude_input_prompt=True --HTMLExporter.exclude_output_prompt=True --ExecutePreprocessor.timeout=900 --output-dir results/reports notebooks/Interaction_Sequences.ipynb
	@python -m nbconvert --to html --execute --template lab --HTMLExporter.theme=light --HTMLExporter.exclude_input=True --HTMLExporter.exclude_input_prompt=True --HTMLExporter.exclude_output_prompt=True --ExecutePreprocessor.timeout=1200 --output-dir results/reports notebooks/Operator_Completeness_Search.ipynb
