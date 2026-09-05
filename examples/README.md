# TNFR examples

The examples are executable demonstrations organized by subject. Their numbered
filenames are stable discovery aids, not a progression of proven results.

Install the repository in editable mode before running them:

```bash
python -m pip install -e .
python examples/01_foundations/01_hello_world.py
python examples/01_foundations/10_simplified_sdk_showcase.py
```

Examples that require optional libraries should report or skip the missing
backend explicitly.

## Directory index

| Directory | Scope |
| --- | --- |
| `01_foundations` | Nodal state, operators, topology, coherence and public SDK |
| `02_physics_regimes` | Diffusion, modal models, fields, conservation and grammar diagnostics |
| `03_riemann_zeta` | Riemann-program instruments for the zeta track |
| `04_riemann_L_twisted` | Character-twisted and L-function research instruments |
| `05_type_hygiene` | Catalog-extension counterexamples and type checks |
| `06_navier_stokes` | Scoped Navier-Stokes correspondences and cascade diagnostics |
| `07_number_theory` | Arithmetic pressure, residue networks and primality structure |
| `08_emergent_geometry` | Symplectic, spectral, multiscale and structural-geometry models |
| `09_millennium` | Explicitly open reformulations of classical research problems |
| `10_applications` | Data-interface and application demonstrations |

## Interpretation rules

- Read the module docstring before running an example; it states assumptions and
  expected outputs.
- A numerical match applies only to the recorded domain and tolerance.
- Labels such as `classical`, `quantum-like`, `particle`, `atom` or `cosmology`
  denote model comparisons or analogies unless a document states and validates a
  physical identification.
- Riemann, Navier-Stokes, Yang-Mills, P-versus-NP, BSD and Hodge examples do not
  claim solutions to those problems.
- Arithmetic examples must disclose whether known factors enter construction or
  verification.

The governing theory and claim status live in [theory/README.md](../theory/README.md).
Public APIs and package ownership live in [ARCHITECTURE.md](../ARCHITECTURE.md).
Test requirements live in [TESTING.md](../TESTING.md).
