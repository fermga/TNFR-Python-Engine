# TNFR Python Engine

<div align="center">

**Model reality as coherent resonance, not isolated objects**

[![PyPI](https://img.shields.io/pypi/v/tnfr)](https://pypi.org/project/tnfr/)
[![Python](https://img.shields.io/pypi/pyversions/tnfr)](https://pypi.org/project/tnfr/)
[![License](https://img.shields.io/github/license/fermga/TNFR-Python-Engine)](LICENSE.md)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://fermga.github.io/TNFR-Python-Engine/)

</div>

TNFR (Resonant Fractal Nature Theory) is a physics-grounded computational paradigm: reality is modeled as **coherent patterns that persist through resonance**. Structures reorganize according to the nodal equation (∂EPI/∂t = νf · ΔNFR) under canonical grammar constraints (U1–U6) and invariants.

---
## Quick Install
```bash
pip install tnfr
```
Optional GPU / extras: see Getting Started.

### Minimal Example
```python
from tnfr.sdk import TNFRNetwork
net = TNFRNetwork("hello")
summary = (net.add_nodes(8)
             .connect_nodes(0.35, "random")
             .apply_sequence("basic_activation", repeat=2)
             .measure().summary())
print(summary)
```

---
## Primary Documentation Hubs

**📚 [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation map and navigation guide

**📖 [CANONICAL_SOURCES.md](CANONICAL_SOURCES.md)** - Documentation hierarchy (which source is authoritative for what)

### Quick Navigation

- **Getting Started**: `docs/source/getting-started/README.md` - Tutorials & first steps
- **Learning Paths**: `docs/source/getting-started/LEARNING_PATHS.md` - Guided learning sequences
- **Grammar System**: `docs/grammar/README.md` - U1-U6 constraints hub
- **Glossary**: `GLOSSARY.md` - Canonical term definitions
- **AI Agent Guide**: `AGENTS.md` - Invariants & philosophy
- **Architecture**: `ARCHITECTURE.md` - System design patterns
- **Contributing**: `CONTRIBUTING.md` | **Tests**: `TESTING.md`

### Core References

- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** - Complete U1-U6 physics derivations
- **[Mathematical Foundations](docs/source/theory/mathematical_foundations.md)** - Rigorous formalization
- **[Operators Reference](docs/source/api/operators.md)** - 13 canonical operators
- **[U6 Specification](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md)** - Structural potential confinement

Extended examples: `examples/` (multi-scale, regenerative, performance)  
CLI & profiling: `docs/source/tools/CLI.md`

---
## Key Principles (Snapshot)
- Coherence over objects (EPI evolves only via operators)
- Bounded reorganization (U2 integral convergence, U6 Φ_s confinement)
- Phase-verified coupling (U3)
- Operational fractality (REMESH / multi-scale coherence U5)
- Reproducibility (seeded trajectories, Invariant #8)

---
## Citation & License
MIT License – see `LICENSE.md`.
Please cite: `fermga/TNFR-Python-Engine` and theoretical sources (`TNFR.pdf`, Mathematical Foundations).

---
## Useful Links
- Docs: https://fermga.github.io/TNFR-Python-Engine/
- PyPI: https://pypi.org/project/tnfr/
- Issues: https://github.com/fermga/TNFR-Python-Engine/issues

---
<div align="center">
Reality is not made of things—it's made of resonance.
</div>
