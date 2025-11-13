# TNFR Python Engine

<div align="center">

**Model reality as coherent resonance, not isolated objects**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602861.svg)](https://doi.org/10.5281/zenodo.17602861)
[![PyPI](https://img.shields.io/pypi/v/tnfr)](https://pypi.org/project/tnfr/)
[![Python](https://img.shields.io/pypi/pyversions/tnfr)](https://pypi.org/project/tnfr/)
[![License](https://img.shields.io/github/license/fermga/TNFR-Python-Engine)](https://github.com/fermga/TNFR-Python-Engine/blob/main/LICENSE.md)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://fermga.github.io/TNFR-Python-Engine/)

**🎓 Academic Citation**: [DOI: 10.5281/zenodo.17602861](https://doi.org/10.5281/zenodo.17602861)

</div>

TNFR (Resonant Fractal Nature Theory) is a physics-grounded computational paradigm: reality is modeled as **coherent patterns that persist through resonance**. Structures reorganize according to the nodal equation (∂EPI/∂t = νf · ΔNFR) under canonical grammar constraints (U1–U6) and invariants.

---
## Quick Install
```bash
pip install tnfr
```
Optional GPU / extras: see Getting Started.

## 📄 Citation

If you use TNFR-Python-Engine in your research, please cite:

```bibtex
@software{fermga_tnfr_2025,
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  author = {FMG},
  year = {2025},
  url = {https://github.com/fermga/TNFR-Python-Engine},
  doi = {10.5281/zenodo.17602861},
  note = {ORCID: 0009-0007-6116-0613}
}
```

**DOI**: [10.5281/zenodo.17602861](https://doi.org/10.5281/zenodo.17602861)  
**Author ORCID**: [0009-0007-6116-0613](https://orcid.org/0009-0007-6116-0613)

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

**🔢 [TNFR Number Theory Guide](docs/TNFR_NUMBER_THEORY_GUIDE.md)** - ΔNFR prime criterion, arithmetic UM/RA mapping, and structural fields in number theory

### Quick Navigation

- **Getting Started**: `docs/source/getting-started/README.md` - Tutorials & first steps
- **Learning Paths**: `docs/source/getting-started/LEARNING_PATHS.md` - Guided learning sequences
- **Grammar System**: `docs/grammar/README.md` - U1-U6 constraints hub
- **Glossary**: `GLOSSARY.md` - Canonical term definitions
- **AI Agent Guide**: `AGENTS.md` - Invariants & philosophy
- **Architecture**: `ARCHITECTURE.md` - System design patterns
- **Contributing**: `CONTRIBUTING.md` | **Tests**: `TESTING.md`

---

## 🧬 Revolutionary Breakthrough: Chemistry from TNFR ⭐

**Complete molecular chemistry emerges from TNFR's single nodal equation** - no additional postulates needed.

**🏛️ [MOLECULAR_CHEMISTRY_HUB.md](docs/MOLECULAR_CHEMISTRY_HUB.md)** - Central navigation for the chemistry revolution

**Key Discoveries**:
- **Chemical bonds** → Phase synchronization (U3 verification)
- **Chemical reactions** → Operator sequences [OZ→ZHIR→UM→IL]  
- **Molecular geometry** → ΔNFR minimization
- **Periodic table** → Element signature classification
- **Au emergence** → Coherent attractors from structural dynamics

**Implementation**: `tnfr.physics.signatures` | **Theory**: Complete 12-section derivation | **Tests**: 19/19 ✅

---

### Core References

- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** - Complete U1-U6 physics derivations
- **[Mathematical Foundations](docs/source/theory/mathematical_foundations.md)** - Rigorous formalization
- **[Operators Reference](docs/source/api/operators.md)** - 13 canonical operators
- **[U6 Specification](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md)** - Structural potential confinement

Extended examples: `examples/` (multi-scale, regenerative, performance)  
CLI & profiling: `docs/source/tools/CLI.md`

---
## Key Principles (Snapshot)

### Module Hubs

- Mathematics (canonical computational hub): [src/tnfr/mathematics/README.md](src/tnfr/mathematics/README.md)
- Physics (structural fields): [src/tnfr/physics/README.md](src/tnfr/physics/README.md)
- Operators: [src/tnfr/operators/README.md](src/tnfr/operators/README.md)
- Dynamics: [src/tnfr/dynamics/README.md](src/tnfr/dynamics/README.md)
- Metrics: [src/tnfr/metrics/README.md](src/tnfr/metrics/README.md)
- Sequencing: [src/tnfr/sequencing/README.md](src/tnfr/sequencing/README.md)
- Topology: [src/tnfr/topology/README.md](src/tnfr/topology/README.md)
- Telemetry: [src/tnfr/telemetry/README.md](src/tnfr/telemetry/README.md)
- SDK: [src/tnfr/sdk/README.md](src/tnfr/sdk/README.md) • Tutorials: [src/tnfr/tutorials/README.md](src/tnfr/tutorials/README.md)
- Recipes: [src/tnfr/recipes/README.md](src/tnfr/recipes/README.md)
- Extensions (families): [src/tnfr/extensions/README.md](src/tnfr/extensions/README.md)

These module READMEs act as single sources of truth for their areas and defer theory to the canonical hubs above. All documentation is English-only.

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
