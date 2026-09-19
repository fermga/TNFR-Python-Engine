# CLI and SDK usage

The CLI and Python SDK provide two entrances to the same declared study runner.
Use a `Network` for direct manipulation, or a `StudySpec` to retain a reusable
preparation and operator word. Both use the engine's existing operators,
preconditions and diagnostic owners; the interface adds no physical law.

## Install and discover

```bash
python -m pip install tnfr
tnfr --help
python -m tnfr --help
```

For the checked-out repository, install it with `python -m pip install -e .`.
The commands below need the core package only. Plotting and optional numerical
backends have separate extras listed in the [root README](../README.md#installation).
Use the installed command's help to check its available options.

## Create, evolve and diagnose

```python
from tnfr.sdk import TNFR, diagnose_network

network = TNFR.create(6, seed=42).ring()
network.evolve(steps=1, sequence="basic_activation")
diagnostics = diagnose_network(network)
print(diagnostics)
```

This supplies a six-node ring and one registered operator word. Initial EPI is
zero, capacity is one, and phase is zero. A uniform initial state can retain
uniform diagnostics; this preparation does not demonstrate spontaneous pattern
formation. On this direct API, `seed` configures stochastic topology builders;
runtime operators resolve the separate graph `RANDOM_SEED` setting. The
`run_study` interface below supplies the same declared seed to both owners.

`steps=1` executes one complete word, not one second of physical time. Word
admission and live node/edge preconditions still apply. In particular, a word
requiring resonance with a neighbor can fail on an isolated node in a random
graph. A topology seed makes the random construction repeatable within its
runtime; it does not make an inadmissible preparation valid.

`diagnose_network` observes a detached graph copy. It reads the stored pressure;
it does not refresh pressure, evolve the network, infer a phase law, or supply
missing temporal observations. A nodal product `nu_f * DeltaNFR` is a model-rate
read-out, not a measured finite-time change.

Invalid or unavailable diagnostics retain explicit error/availability data
instead of inventing a number. Circular curvature can be unavailable at
individual nodes. Coherence-length provenance distinguishes the static
product-fit length from the separate spectral fallback. See the
[tetrad owner](STRUCTURAL_FIELDS_TETRAD.md) for definitions and units. Diagnostic
flags are observations or policies, not authorization to execute an operator
or a guarantee of future stability.

## Run and export the same study from either interface

The Python declaration is explicit and serializable:

```python
from tnfr.sdk import StudySpec, export_to_json, run_study

spec = StudySpec(
    nodes=6,
    topology="ring",
    seed=42,
    sequence="basic_activation",
    cycles=1,
    name="ring-study",
)
result = run_study(spec)
export_to_json(spec.to_dict(), "study.json")
export_to_json(result, "report.json")
```

The equivalent CLI preparation is:

```bash
tnfr network --nodes 6 --topology ring --seed 42 --sequence basic_activation --steps 1 --name ring-study --export-spec study.json --output report.json
```

The CLI maps `--steps` to the declaration's `cycles`. To run an already exported
declaration:

```bash
python -m tnfr network --spec study.json --output replay-report.json
```

`--spec` is exclusive with preparation options such as `--nodes`, `--seed` and
`--steps`. Edit or construct the declaration before running it; an additional
command-line value does not silently override a recorded input.

Python reads the same file through the strict declaration constructor:

```python
from tnfr.sdk import StudySpec, import_from_json, run_study

spec = StudySpec.from_dict(import_from_json("study.json"))
replayed = run_study(spec)
report = replayed.to_dict()
```

`StudySpec` is immutable and validates its declared inputs. Its fields are
`nodes`, `topology`, `seed`, `sequence`, `cycles`, `probability` and `name`.
Supported topologies are `ring`, `path`, `star`, `complete` and `random`;
`probability` configures random edges. `sequence` names a registered word.
Each call creates its own prepared network rather than continuing another run.

The result records the declaration, runtime provenance, realized initial and
final triad/support, and final diagnostics. `to_dict()` returns detached data;
`export_to_json` uses the shared JSON writer. Without `--output`, the CLI writes
JSON to standard output and sends messages to standard error. An output path
receives the report instead. Retain the declaration together with the report
when comparing studies.

| Report key | Content |
| --- | --- |
| `spec` | Validated declaration, including the seed and requested cycles |
| `execution` | Package/runtime versions, precision mode, registered word and completed cycles |
| `initial_state` | Indexed scalar triad/pressure observations and support before execution |
| `final` | Final state projection, metrics, nodal observations and tetrad with availability/provenance |

Other engine configuration is inherited from the running process. The report
does not capture every effective setting; retain relevant external configuration
with the declaration. Scalar state rows use indices in graph iteration order;
display labels are not serialized node identities, and edge attributes are not
included in this projection.

The executable [reproducible study example](../examples/01_foundations/reproducible_study.py)
writes a declaration, reads it back, runs it and exports the result:

```bash
python examples/01_foundations/reproducible_study.py --output-dir output/study
```

## Discover registered words and operators

```bash
tnfr sequences
tnfr sequences basic_activation
tnfr operators
tnfr operators emission
```

These commands expose existing catalogs rather than a second grammar. They
accept `--output` for JSON export. Python accesses the same named words with
`list_sequences()` or `list_sequences("basic_activation")` from `tnfr.sdk`.
The [operator contracts](API_CONTRACTS.md) own operator metadata, while
[grammar scope](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) distinguishes admission
policies from mathematical stability results.

## Reproducibility and scope

| Retained item | What it establishes | What it does not establish |
| --- | --- | --- |
| Declaration and seed | Supplied preparation, word and requested cycle count | Autonomous selection of those inputs |
| Initial/final triad and support | Observed endpoints of that finite invocation | Every intermediate state or future behavior |
| Runtime provenance | Context for comparing executions | Bit-identical results across arbitrary versions, platforms or numerical backends |
| Diagnostic values and availability | Read-outs of stored state with estimator scope | Complete reconstruction or a new constitutive law |

A study report is not a complete resumable checkpoint. The declaration can be
executed again from its supplied initial state; the report does not restore
all callbacks, caches, operator history or external state. This distinction also
applies to older `export_to_json` payloads and `import_from_json`, which reads
JSON data without reconstructing a live execution.

For experimental interpretation, the
[research plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md) and
[measurement protocol](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md) remain
the owners of hypothesis selection, calibration and reserved evaluation.
Exporting a study does not by itself satisfy those scientific admission gates.

## Existing advanced execution routes

`tnfr run`, `sequence`, `math.run`, `epi.validate` and `metrics` retain their
specialized execution/configuration paths. Their `--help` output owns the
available flags. They are not aliases for `network`, and their history formats
are not `StudySpec` declarations. Prefer `network` for the shared SDK/CLI study
workflow and use a specialized route when its actual configuration is needed.

The [example index](../examples/README.md) classifies the other demonstrations.
The SDK's fluent builders, auxiliary physics adapters and optimizer policies
have their own scopes; this guide does not promote them into autonomous nodal
laws.
