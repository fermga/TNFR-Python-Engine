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

Recent executable runtime and refinement examples in `02_physics_regimes` are
`163_reception_runtime_bridge.py`, `164_resonance_runtime_bridge.py`,
`165_operator_event_relaxation.py`, and
`166_event_remesh_reference_family.py`. Example 166 certifies one finite
event-free effective-P2 runtime family for a fixed nonuniform mode, with
`2/4/8` pressure-refreshed Euler segments; it is not a generic
mesh-convergence experiment.

`167_reversible_eigenmode_reference.py` is the pure exact-rational extension.
It certifies both nonuniform eigenmodes of the nonregular three-node path in
the reversible metric `H=diag(1,2,1)`, including rational exponential
enclosures, exact Euler products and conditional exact-real refinement scope.
It does not execute a binary64 solver, glyph or REMESH operation. The complete
proof is in the
[diffusion stability theorem](../theory/TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem).

`168_runtime_reversible_eigenmode_reference.py` executes three independent
`2/4/8`-segment pressure-refreshed partitions of one nonregular-`P3` exact
mode, then binds their captured binary64 boundaries to that rational reference.
It reports nonzero pressure (`rho`), held-input (`eta`) and combined local
(`epsilon`) defects, propagated through the complete Euler matrices. The family
is a finite offline comparison of individually executor-certified records; it
does not certify runtime mesh convergence, solver accuracy/order, common causal
provenance, glyph/REMESH behavior, repetition or future stability.

`169_event_remesh_causal_runtime.py` executes two declared event/REMESH cycle
specs on one graph in one outer transaction. Its receipts bind each ordinal,
exact spec, schedule identity and cycle result to the same invocation, while
the result retains the offline cycle observation and finite schedule/history
telescope. The lag-one `alpha=1` witness alternates `(2,0) -> (0,2) -> (2,0)`,
showing that causal provenance and atomicity alone do not prove a global gain,
uniform normalized class margin, convergence or future stability.

`170_runtime_remesh_block_margin.py` applies the exact block observer to two
causally executed finite sequences. The contractive block has
`kappa=139/256` and endpoint gain upper bound `117/256`; the lag-one
`alpha=1` identity-schedule block has `kappa=0`. Both results are finite
observations. They do not prove a uniform positive normalized margin over a
declared forward-invariant runtime class, an intrablock runtime
prefix-amplification bound,
repetition or future stability. A positive absolute uniform drop is excluded
by equilibrium and quadratic amplitude scaling.

`171_remesh_schedule_policy_stability.py` treats the separate exact model class
in which every schedule preserves spatial consensus and has one common
fixed-metric disagreement gain bound `q`. It displays the sufficient universal
horizon `active_max_delay+1`, prefix gain upper bound one, normalized block margin `1-q`
and the repeated bound `q^floor(n/L)`. Its strict mixed-delay and pure-delay
witnesses use `q=1/4`; the `q=1` witness has zero certified margin. This example
does not execute or verify a binary64 schedule, control runtime defects or
claim stability of spatially uniform temporal means or full TNFR state.

`172_runtime_remesh_relative_defect.py` adds the conditional robust envelope
`q_eff=q*(1+eta)`, where `eta` bounds the signed pre-schedule centered-energy
defect relative to the Jensen input energy `J`. Its three-cycle dyadic witness
has zero defect and spans one complete universal block; its `alpha=0.4`
witness retains a positive binary64 defect and accepts the exact minimum
`eta`. The causal observer verifies every selected `J`, defect, schedule gain,
history-energy vector and finite endpoint bound. It does not establish a
forward-invariant runtime class, repeated or future binary64 stability, solver
properties or full TNFR stability.
