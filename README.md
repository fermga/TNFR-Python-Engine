# TNFR Python Engine

## What is TNFR?

TNFR (**Resonant Fractal Nature Theory**) models reality as a network of resonant nodes that reorganize structurally through coherence. Imagine cells synchronizing in a living organism: each cell maintains its identity while coordinating behavior with others. TNFR is an operational paradigm with concrete mathematical and computational tools for simulating complex adaptive systems.

## What is it for?

TNFR enables modeling and analyzing complex systems where coherent patterns emerge through local interactions:

- 🧬 **Biology**: Cellular communication networks, neuronal synchronization, protein dynamics
- 🌐 **Social systems**: Information propagation, community formation, opinion dynamics
- 🤖 **Artificial Intelligence**: Resonant symbolic systems, structural processing networks
- 🔬 **Network science**: Structural coherence analysis, emergent pattern detection
- 🏗️ **Distributed systems**: Decentralized coordination, self-organization

**Key advantages**: Operational fractality (patterns scale without losing structure), complete traceability (every reorganization is observable), and guaranteed reproducibility.

## Quick Installation

Install from PyPI (Python ≥ 3.9):

```bash
pip install tnfr
```

### Your First TNFR Network (3 lines!)

```python
from tnfr.sdk import TNFRNetwork

# 1. Create a network with an identifying name
network = TNFRNetwork("hello_world")

# 2. Add 10 nodes and connect them randomly (30% probability)
#    Then apply a sequence of structural operators 3 times
#    Finally measure coherence metrics
results = network.add_nodes(10).connect_nodes(0.3, "random").apply_sequence("basic_activation", repeat=3).measure()

# 3. Display results: coherence C(t), sense index Si, and frequencies νf
print(results.summary())
```

🎉 **That's it!** You just created, activated, and measured a TNFR network.

**What just happened?**
- `add_nodes(10)`: Creates 10 resonant nodes (like musical notes that can synchronize)
- `connect_nodes(0.3, "random")`: Connects nodes randomly with 30% probability
- `apply_sequence("basic_activation", repeat=3)`: Applies structural operators (Emission → Coherence → Resonance) 3 times
- `measure()`: Calculates total coherence C(t), sense index Si, and other structural metrics

### Interactive Tutorials

Learn TNFR interactively in 5 minutes:

```python
from tnfr.tutorials import hello_tnfr
hello_tnfr()  # Guided tour of TNFR concepts
```

Or try domain-specific examples:
- `biological_example()` - Cell communication model
- `social_network_example()` - Social dynamics  
- `technology_example()` - Distributed systems
- `team_communication_example()` - Team structure optimization
- `adaptive_ai_example()` - Learning through resonance

📘 **Want a structured learning path?** See the full [**Interactive Tutorial**](docs/source/getting-started/INTERACTIVE_TUTORIAL.md) (60 minutes from zero to first application)

## Key Concepts

**New to TNFR?** 👉 Start with the [**TNFR Fundamental Concepts Guide**](docs/source/getting-started/TNFR_CONCEPTS.md) - understand the paradigm in 10 minutes!

### Quick Reference

Before diving deeper, here's a brief overview of fundamental terms:

#### Resonant Fractal Node (NFR)
Minimum unit of structural coherence in the network. Each node has:
- **EPI**: Primary Information Structure (its coherent "shape")
- **νf**: Structural frequency (reorganization rate, in Hz_str)
- **Phase φ**: Synchrony with other nodes in the network

#### Structural Operators
Functions that reorganize nodes coherently (13 canonical operators):
- **Emission/Reception**: Initiate and capture resonant patterns
- **Coherence/Dissonance**: Stabilize or destabilize structures
- **Resonance**: Propagates coherence without losing EPI identity
- **Self-organization**: Creates emergent sub-structures
- [See complete list in GLOSSARY.md](GLOSSARY.md#structural-operators)

#### Coherence Metrics
- **C(t)**: Total network coherence at time t
- **Si**: Sense index (capacity to generate stable reorganization)
- **ΔNFR**: Internal reorganization operator

#### Fundamental Nodal Equation
```
∂EPI / ∂t = νf · ΔNFR(t)
```
This equation governs how the structure (EPI) of each node evolves according to its frequency (νf) and reorganization gradient (ΔNFR).

**📖 For deeper understanding**: 
- [TNFR Fundamental Concepts](docs/source/getting-started/TNFR_CONCEPTS.md) - Comprehensive introduction to the paradigm
- [GLOSSARY.md](GLOSSARY.md) - Complete reference of all terms, variables, and operators

## Technical Documentation

### User Guides

- 📘 [**TNFR Fundamental Concepts**](docs/source/getting-started/TNFR_CONCEPTS.md) - **START HERE** - Understand the paradigm in 10 minutes!
- 🚀 [**NEW Quick Start Guide**](docs/source/getting-started/QUICKSTART_NEW.md) - Get started in 5 minutes!
- 📚 [Original Quickstart](docs/source/getting-started/quickstart.md) - Python and CLI walkthroughs
- 🎓 [Interactive Tutorials](src/tnfr/tutorials/README.md) - Learn by doing
- 💡 [Hello World Example](examples/hello_world.py) - Simplest possible example

### API Reference and Architecture

- [Documentation Index](docs/source/home.md) — Navigation hub for API chapters and examples
- [API Overview](docs/source/api/overview.md) — Package map, invariants, and structural data flow
- [Structural Operators](docs/source/api/operators.md) — Canonical grammar, key concepts, and typical workflows
- [Glyph Sequences Guide](GLYPH_SEQUENCES_GUIDE.md) — Canonical operator sequences, multi-domain examples
- [Backend System](docs/backends.md) — Vectorized computation with NumPy/JAX/Torch backends
- [Telemetry & Utilities](docs/source/api/telemetry.md) — Coherence metrics, trace capture, locking
- [Examples](docs/source/examples/README.md) — Runnable scenarios, CLI artifacts

### Theoretical and Advanced Resources

- [ARCHITECTURE.md](ARCHITECTURE.md) — Orchestration layers and invariant enforcement
- [TESTING.md](TESTING.md) — Test strategy, organization, and structural fidelity validation
- [SECURITY.md](SECURITY.md) — Security policy and best practices
- [CONTRIBUTING.md](CONTRIBUTING.md) — QA battery (`scripts/run_tests.sh`) and review expectations
- [GLOSSARY.md](GLOSSARY.md) — Unified glossary of TNFR variables, operators, and concepts
- [Factory Documentation](docs/FACTORY_DOCUMENTATION_INDEX.md) — Comprehensive guide to factory patterns
- [TNFR.pdf](TNFR.pdf) — Theoretical foundations, structural operators, and paradigm glossary

## CLI Profiling Tools

Generate Sense Index and ΔNFR profiling artifacts directly from the CLI with the `profile-pipeline` subcommand. This tool reproduces the performance benchmark that captures vectorized and fallback execution traces for the full pipeline:

```bash
tnfr profile-pipeline \
  --nodes 120 --edge-probability 0.28 --loops 3 \
  --si-chunk-sizes auto 48 --dnfr-chunk-sizes auto \
  --si-workers auto --dnfr-workers auto \
  --output-dir profiles/pipeline
```

The command writes `.pstats` and JSON summaries for each configuration/mode pair, making it easy to inspect hot paths with :mod:`pstats`, Snakeviz, or other tools.

## Documentation Build

Netlify now renders the documentation with [Sphinx](https://www.sphinx-doc.org/), so MyST Markdown, doctests, and notebooks share a single pipeline. Reproduce the hosted site locally as follows:

1. Create and activate a virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install the documentation toolchain and project extras:
   `python -m pip install -r docs/requirements.txt && python -m pip install -e .[docs]`.
3. Execute the doctest suite with `sphinx-build -b doctest docs/source docs/_build/doctest` to ensure structural snippets remain coherent.
4. Generate the HTML site with `make docs`, which wraps `sphinx-build -b html docs/source docs/_build/html`.

The Netlify build (`netlify.toml`) runs `python -m pip install -r docs/requirements.txt && make docs` and publishes the resulting `docs/_build/html` directory, keeping the hosted documentation aligned with local verification runs.

## Local Development

Use the helper scripts to keep formatting aligned with the canonical configuration and reproduce the quality gate locally:

```bash
./scripts/format.sh           # Apply Black and isort across src/, tests/, scripts/, and benchmarks/
./scripts/format.sh --check   # Validate formatting without modifying files
./scripts/run_tests.sh        # Execute the full QA battery (type checks, tests, coverage, linting)
```

The formatting helper automatically prefers `poetry run` when a Poetry environment is available and falls back to `python -m` invocations so local runs mirror the tooling invoked in continuous integration.

## Configuration and Secrets Management

TNFR follows security best practices for handling sensitive credentials:

**Quick Start:**

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your credentials (this file is gitignored)
# Never commit .env files with real credentials!
```

**Secure Configuration Loading:**

```python
from tnfr.secure_config import load_redis_config, get_cache_secret

# Load Redis configuration from environment variables
redis_config = load_redis_config()

# Get cache signing secret for hardened mode
cache_secret = get_cache_secret()
```

**Key Principles:**

- All secrets are loaded from environment variables (never hardcoded)
- `.env.example` provides a template with secure placeholder values
- Configuration utilities validate and provide helpful error messages
- Automated tests scan for accidentally hardcoded secrets

See [SECURITY.md](SECURITY.md) for detailed information on secret management, credential rotation, and security best practices.

## Migration Notes

- **Si dispersion keys:** Ensure graph payloads and configuration files use the canonical `dSi_dphase_disp` attribute for Si dispersion sensitivity before upgrading. The runtime now raises `ValueError` listing any unexpected sensitivity keys, and `tnfr.metrics.sense_index.compute_Si_node` rejects unknown keyword arguments.
- Refer to the [release notes](docs/source/releases.md#1100-si-dispersion-legacy-keys-removed) for a migration snippet that rewrites stored graphs in place prior to running the new version.

## License

Released under the [MIT License](LICENSE.md). Cite the TNFR paradigm when publishing research or derived artifacts based on this engine.

---

## Getting Started - Suggested Progression

1. **New to TNFR?** → Read [TNFR Fundamental Concepts](docs/source/getting-started/TNFR_CONCEPTS.md) to understand the paradigm (10 minutes)
2. **Beginners** → Run `examples/hello_world.py` and then `from tnfr.tutorials import hello_tnfr; hello_tnfr()`
3. **Users** → Read [QUICKSTART_NEW.md](docs/source/getting-started/QUICKSTART_NEW.md) and experiment with domain tutorials
4. **Developers** → See [ARCHITECTURE.md](ARCHITECTURE.md), [GLOSSARY.md](GLOSSARY.md), and the [API Overview](docs/source/api/overview.md)
5. **Researchers** → Study [TNFR.pdf](TNFR.pdf) and the [Theoretical Overview Notebook](docs/source/theory/00_overview.ipynb)
