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

`173_binary64_remesh_relative_defect.py` isolates the sharp represented-number
boundary. Its normal-valued `alpha=1/2` pair requires exactly
`eta=2^210-1/4`, hence strict robust-envelope contraction requires
`q<4/(2^212+3)`; the bounded interval alone cannot promote the implemented
`q=9/16` witness policy. Separately, the sufficient-history `alpha=1`
hard-clip class has uniform `eta=0` and is forward invariant under REMESH alone
on its fixed support and metric. It does not certify a common schedule family,
schedule composition, repeated runtime stability or future execution.

`174_binary64_p2_reception_remesh_stability.py` supplies the first restricted
global numeric EPI-kernel composition. On two mutual singleton neighbors, the
configured binary64 half-Reception Jacobi kernel maps every finite represented
pair in a common hard interval to numeric consensus, so `q=0`. With the `alpha=1`,
`eta=0` REMESH class, active-history spatial disagreement is exactly zero after
`tau_global+1` cycles of the restricted kernels. The example keeps the complete
Reception stage, grammar, live graph execution and solver outside scope.

`175_runtime_p2_reception_stage.py` crosses the finite causal stage boundary.
It executes one grammar-admitted two-phase EN event on a real P2 graph and
binds the sealed targets, runtime neighbors, exact half mix, hard interval,
metric ray and captured endpoints to the global `q=0` kernel by bit-exact
replay. It does not bind REMESH history/configuration to that graph, certify
all auxiliary Reception state, or establish repeated/future runtime stability.

`176_runtime_p2_reception_remesh_sequence.py` binds the two restricted kernels
inside one completed graph-owned causal sequence. Each observed cycle retains
an executor-owned half-Reception EN stage with `q=0` and an applied
`alpha=1` hard-clipped REMESH global-delay copy with `eta=0`. Once
`N >= L = tau_global+1`, the active suffix of `L` history rows and the recorded
post-horizon endpoint have zero spatial disagreement. Older inactive history
rows may remain outside the source interval. The example does not certify
future or unobserved repetition, auxiliary state, solver behavior or full TNFR
stability.

`177_runtime_p2_reception_remesh_policy.py` executes that restricted word
through the reusable transactional policy twice on the same graph. Before each
call it revalidates the live P2 support, metric, exact half-Reception factor,
hard-clipped `alpha=1` REMESH controls, active incoming history and zero-flow
cycle schedules, and it rederives U1a admission at every cycle start. Execution
and finite post-certification share one outer graph transaction. The two returned
certificates have independent finite provenance;
neither certifies a later invocation, auxiliary-state stability, solver behavior
or full TNFR stability.
