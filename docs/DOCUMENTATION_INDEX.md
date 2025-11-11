# TNFR Documentation Index

**Welcome to TNFR!** This index helps you navigate the reorganized documentation.

## 📖 Main Navigation

**Start Here**: [**Documentation Home**](source/index.md) - Main hub with complete navigation

## 🚀 Quick Start by User Type

### 🔰 For Beginners
→ [**Welcome to TNFR**](source/getting-started/README.md) ⭐ **START HERE**  
→ [Quickstart Tutorial](source/getting-started/quickstart.md)  
→ [Core Concepts](source/getting-started/TNFR_CONCEPTS.md)  
→ [FAQ](source/getting-started/FAQ.md)

### 👤 For Users  
→ [**Operators Guide**](source/user-guide/OPERATORS_GUIDE.md)  
→ [**Pattern Cookbook**](PATTERN_COOKBOOK.md) ⭐ **NEW** - Ready-to-use recipes  
→ [**Canonical OZ Sequences**](CANONICAL_OZ_SEQUENCES.md) ⭐ **NEW** - OZ dissonance patterns  
→ [Metrics Interpretation](source/user-guide/METRICS_INTERPRETATION.md)  
→ [Troubleshooting](source/user-guide/TROUBLESHOOTING.md)  
→ [Examples Catalog](source/examples/README.md)

### 💻 For Developers
→ [**Performance Optimization**](source/advanced/PERFORMANCE_OPTIMIZATION.md)  
→ [**Mathematical Foundations**](source/theory/mathematical_foundations.md) ⭐ **CANONICAL MATH SOURCE**  
→ [API Reference](source/api/overview.md)  
→ [Contributing](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md)

---

## 📚 Complete Documentation Structure

### Getting Started (`source/getting-started/`)
- **[README.md](source/getting-started/README.md)** - What is TNFR? Installation, first network
- [quickstart.md](source/getting-started/quickstart.md) - Step-by-step tutorial  
- [TNFR_CONCEPTS.md](source/getting-started/TNFR_CONCEPTS.md) - Fundamental concepts
- [FAQ.md](source/getting-started/FAQ.md) - Frequently asked questions
- [INTERACTIVE_TUTORIAL.md](source/getting-started/INTERACTIVE_TUTORIAL.md) - Hands-on learning
- [math-backends.md](source/getting-started/math-backends.md) - NumPy/JAX/PyTorch
- [optional-dependencies.md](source/getting-started/optional-dependencies.md) - Extras

### User Guide (`source/user-guide/`)
- **[OPERATORS_GUIDE.md](source/user-guide/OPERATORS_GUIDE.md)** - Complete guide to 13 operators
- [METRICS_INTERPRETATION.md](source/user-guide/METRICS_INTERPRETATION.md) - Understanding C(t), Si, νf, phase, ΔNFR  
- [TROUBLESHOOTING.md](source/user-guide/TROUBLESHOOTING.md) - Common problems & solutions
- **[PATTERN_COOKBOOK.md](PATTERN_COOKBOOK.md)** ⭐ **NEW** - Validated recipes by domain
- **[CANONICAL_OZ_SEQUENCES.md](CANONICAL_OZ_SEQUENCES.md)** ⭐ **NEW** - OZ dissonance operator sequences

### Examples (`source/examples/`)
- **[README.md](source/examples/README.md)** - Categorized example catalog
- [controlled_dissonance.py](source/examples/controlled_dissonance.py) - Basic example
- [optical_cavity_feedback.py](source/examples/optical_cavity_feedback.py) - Advanced

### API Reference (`source/api/`)
- [overview.md](source/api/overview.md) - Package structure
- [operators.md](source/api/operators.md) - Operator API
- [telemetry.md](source/api/telemetry.md) - Metrics API

### Advanced Topics (`source/advanced/`)
- **[PERFORMANCE_OPTIMIZATION.md](source/advanced/PERFORMANCE_OPTIMIZATION.md)** - Backends, caching, factories

### Theory (`source/theory/`)
- **[mathematical_foundations.md](source/theory/mathematical_foundations.md)** - ⭐ **CANONICAL MATHEMATICAL SOURCE**
- **Classical Mechanics Emergence Series:**
  - [07_emergence_classical_mechanics.md](source/theory/07_emergence_classical_mechanics.md) - Direct derivation from TNFR
  - [08_classical_mechanics_euler_lagrange.md](source/theory/08_classical_mechanics_euler_lagrange.md) - Variational formulation
  - [09_classical_mechanics_numerical_validation.md](source/theory/09_classical_mechanics_numerical_validation.md) - Computational validation
- Jupyter notebooks with mathematical examples and visualizations
- Primers and operator/validator notebooks

---

## 🗺️ Quick Learning Paths

**Path 1: Quickest (15 min)**  
`Welcome → Quickstart → First Example`

**Path 2: Comprehensive (2-3 hours)**  
`Welcome → Concepts → Operators → Examples → API`

**Path 3: Theory-First (3-4 hours)**  
`Concepts → Mathematical Foundations → Math Notebooks → Examples`

---

## 🔗 Key Reference Files

- [TNFR.pdf](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR.pdf) - Complete theory
- [AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md) - Canonical invariants and Φ_s status
- [UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md) - Grammar U1-U6 derivations ⭐ **UPDATED: U6 CANONICAL**
- [GLOSSARY.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLOSSARY.md) - Terms
- [ARCHITECTURE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md) - System design

### Physics & Forces
- [docs/TNFR_FORCES_EMERGENCE.md](TNFR_FORCES_EMERGENCE.md) - ⭐ **Φ_s Validation & Four Forces (§ 14-15)**
- [src/tnfr/physics/fields.py](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/physics/fields.py) - Structural potential implementation

### Grammar & Research
- **U6: STRUCTURAL POTENTIAL CONFINEMENT** - ✅ **CANONICAL** (promoted 2025-11-11)
  - Derivation: UNIFIED_GRAMMAR_RULES.md § U6
  - Validation: docs/TNFR_FORCES_EMERGENCE.md § 14-15
  - Implementation: src/tnfr/operators/grammar.py::validate_structural_potential_confinement
  - Physics: src/tnfr/physics/fields.py::compute_structural_potential
- [grammar/U6_TEMPORAL_ORDERING.md](grammar/U6_TEMPORAL_ORDERING.md) - ⚗️ **U7 Research Proposal** (experimental, renamed from U6)
- [research/U6_INVESTIGATION_REPORT.md](research/U6_INVESTIGATION_REPORT.md) - U7 initial investigation (temporal ordering)
- [GLYPH_SEQUENCES_GUIDE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md) - Grammar 2.0 sequences

## ℹ️ Legacy Documentation

Detailed technical documents (factory patterns, cache optimization, dependency analysis) remain in `docs/` for reference.

See [legacy/README.md](legacy/README.md) for details.

---

**Ready?** Go to [**Welcome to TNFR**](source/getting-started/README.md) →
