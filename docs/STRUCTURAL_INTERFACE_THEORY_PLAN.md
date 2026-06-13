# TNFR Structural Interface Theory — Integration Plan

## Status

Planning document.  This roadmap describes how to turn the current phase-gate
diagnostics into a reusable TNFR programme for structural-interface analysis on
real graph data.

## Executive summary

TNFR Structural Interface Theory studies graph-local boundaries where a node
state, phase, class, measurement band, or regime is incompatible with the local
topology that supports it.

The intended contribution is not a new fundamental physical law.  The intended
contribution is an operational framework:

```text
real system -> graph construction -> phase/state field -> local interface stress
            -> TNFR tetrad telemetry -> grammar-valid operator prescription
```

This is comparable in role, not in ontological scope, to classical ideas such as
phase boundaries, domain walls, percolation interfaces, graph cuts, and defect
localization.  TNFR's distinctive addition is the last step: the diagnostic is
expressed as canonical operator guidance that can be validated against U1-U6.

## Honest scope and current caveat

The current phase-gate examples are useful proof-of-concept demonstrations:

- controlled local phase compatibility benchmark;
- WDBC biomedical morphology audit;
- UCI Wine Quality chemistry audit.

However, the biomedical and wine examples currently define the review target
from local opposite-label neighbourhood conflicts, while the TNFR phase-gate
stress also uses those conflicts.  This makes the high AUC values a sanity check
for localization, not a fair claim of superiority over classical graph metrics.

The next phase must separate:

1. **Interface detection target** — what counts as a true boundary case.
2. **TNFR score** — phase/tetrad/operator telemetry used to rank nodes.
3. **Classical baselines** — graph disagreement, graph total variation, local
   entropy, label propagation residuals, graph cuts, and uncertainty scores.

Acceptance requires either independent labels/endpoints or a benchmark design
that explicitly treats local disagreement as one baseline, not as hidden ground
truth.

## Core concept

### Structural interface

A structural interface is a graph-local region where neighbouring nodes are
close under the graph relation but differ sharply in phase, state, label,
measurement band, or regime.

Examples:

- tumour samples that are morphologically close but diagnostically different;
- chemical samples that are similar but assigned to different quality bands;
- sensors that are physically adjacent but report incompatible phases;
- ecological sites that are geographically/feature close but belong to
  different habitat states;
- materials microstructures where neighbouring patches cross a damage/failure
  regime boundary.

### TNFR observables

The canonical interface observables should be derived from existing TNFR fields:

- edge phase-gate compliance;
- phase-gradient stress `|∇φ|`;
- phase-curvature stress `|Kφ|`;
- structural potential stress `|Φ_s|`;
- coherence length / neighbourhood persistence where meaningful;
- incident gate-violation pressure;
- grammar-valid operator prescription.

### Operator prescription

Prescriptions must remain read-only recommendations until explicitly applied by
an execution layer.  Every prescribed sequence must pass the repository's
sequence validators:

- `tnfr.operators.grammar_patterns.validate_sequence` for full sequences;
- `tnfr.operators.grammar_dynamics.validate_sequence_incremental` for live
  context.

Current validated patterns:

| Interface state | Sequence | Meaning |
| --- | --- | --- |
| Fully phase-compatible | `UM -> RA -> SHA` | couple, propagate resonance, close |
| Mostly compatible with local hotspots | `IL -> UM -> SHA` | stabilize then guarded coupling |
| Failed interface / boundary hotspot | `IL -> OZ -> THOL -> SHA` | stabilize, open controlled reorganization, self-organize, close |

## Repository architecture

### Current seed modules

- `src/tnfr/validation/phase_gate.py`
  - low-level phase-gate compliance, hotspot ranking, report rendering,
    canonical prescription.
- `examples/90_phase_gate_monitor_demo.py`
  - controlled phase-histogram/locality demonstration.
- `examples/91_breast_cancer_phase_gate_demo.py`
  - real WDBC biomedical proof of concept.
- `examples/92_wine_quality_phase_gate_demo.py`
  - online UCI Wine Quality proof of concept.

### Proposed new module

Create:

```text
src/tnfr/validation/structural_interface.py
```

Suggested public API:

```python
@dataclass(frozen=True)
class StructuralInterfaceProblem:
    graph: Any
    state_key: str
    phase_key: str = "phase"
    domain: str = "generic"
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class StructuralInterfaceScore:
    node: Any
    tnfr_stress: float
    phase_gradient: float
    abs_curvature: float
    structural_potential: float
    incident_gate_pressure: float
    prescription: tuple[str, ...]

@dataclass(frozen=True)
class StructuralInterfaceBenchmarkResult:
    dataset: Mapping[str, Any]
    graph: Mapping[str, Any]
    task: Mapping[str, Any]
    tnfr_results: Mapping[str, Any]
    baseline_results: list[Mapping[str, Any]]
    hotspots: list[Mapping[str, Any]]
```

Functions:

```python
build_knn_graph(records, features, *, k=10) -> Graph
encode_phase_from_binary_state(G, state_key, *, positive_phase=0, negative_phase=pi)
score_structural_interfaces(G, *, gate=DEFAULT_PHASE_GATE) -> list[StructuralInterfaceScore]
evaluate_interface_scores(labels, score_maps) -> dict
render_structural_interface_markdown(result) -> str
render_structural_interface_html(result) -> str
export_structural_interface_report(result, path) -> Path
```

### Keep `phase_gate.py` as the primitive layer

`phase_gate.py` should remain the low-level U3/tetrad diagnostic.  The new
module should orchestrate datasets, graph construction, baseline comparison,
and cross-domain benchmark reporting.

## Fair benchmark design

### Required baselines

Every structural-interface benchmark must compare against:

1. local kNN disagreement / homophily conflict;
2. graph total variation per node;
3. local class entropy;
4. label propagation residual;
5. graph cut contribution;
6. mean neighbour distance;
7. degree / topology-only score;
8. at least one simple domain feature baseline;
9. constant/random baseline.

The local kNN disagreement baseline is especially important because it is the
closest classical analogue to phase-gate violations.  TNFR claims must be framed
relative to this baseline, not only relative to weak baselines.

### Target definitions

Use at least one of the following non-circular targets:

1. **Independent expert/review label**
   - e.g. known borderline class, clinical follow-up, human audit flag.
2. **Downstream model error**
   - train a classifier/regressor and test whether interface hotspots predict
     misclassifications or high residuals on held-out data.
3. **Temporal transition**
   - interface hotspots at time `t` predict state changes at time `t+1`.
4. **Perturbation sensitivity**
   - hotspots are nodes whose label/regime changes under small feature/graph
     perturbations.
5. **Classical interface benchmark target**
   - if the target is local disagreement itself, explicitly compare TNFR to
     local disagreement and avoid superiority claims unless TNFR adds stable
     ranking, robustness, or operator prescription.

### Metrics

Minimum required metrics:

- ROC-AUC;
- average precision / PR-AUC for imbalanced targets;
- precision@N where `N` is the number of review nodes;
- top-k overlap across random seeds;
- ranking stability under graph perturbation;
- sequence validity rate for operator prescriptions;
- report generation integrity.

## Dataset roadmap

### Tier 0 — already integrated proof-of-concept

| Dataset | Sector | Source | Status |
| --- | --- | --- | --- |
| Synthetic phase-gate pairs | controlled graph signals | generated | implemented |
| WDBC breast cancer | biomedicine | scikit-learn bundled | implemented |
| UCI Wine Quality | food chemistry | UCI online | implemented |

### Tier 1 — immediate real-data expansions

| Dataset | Sector | Source | Target idea |
| --- | --- | --- | --- |
| UCI Dry Bean | agriculture / morphology | UCI online | class boundary / model error |
| UCI Seeds | agriculture | UCI online | cultivar boundary samples |
| Iris | botany / morphology | scikit-learn | held-out misclassification |
| Palmer Penguins | ecology | seaborn/open data | species boundary under morphology |
| Human Activity Recognition | sensors | UCI online | transition windows / misclassification |

### Tier 2 — stronger theory comparisons

| Dataset | Sector | Target |
| --- | --- | --- |
| power grid disturbance / fault data | engineering | pre-fault interface detection |
| materials microstructure data | materials science | failure/damage boundary |
| ecological habitat transition data | ecology | transition-zone detection |
| public image embedding datasets | vision | cluster-boundary errors |

## Implementation milestones

### Milestone 1 — formalize the abstraction

Deliverables:

- `src/tnfr/validation/structural_interface.py`;
- dataclasses for problem, score, benchmark result;
- reusable graph builders and phase encoders;
- renderer/export helpers;
- exports from `src/tnfr/validation/__init__.py`.

Acceptance:

- `phase_gate.py` remains backward-compatible;
- all operator prescriptions pass grammar validators;
- tests cover empty graph, binary state, multi-class state, and no-conflict cases.

### Milestone 2 — fair baseline suite

Deliverables:

- `src/tnfr/validation/interface_baselines.py` or private helpers inside
  `structural_interface.py`;
- local disagreement, graph total variation, entropy, label propagation
  residual, graph cut contribution, distance, degree, random/constant baselines;
- deterministic seed handling.

Acceptance:

- baseline names and formulas documented;
- no hidden use of the target label in TNFR-only scores unless declared;
- tests verify baselines on small hand-constructed graphs.

### Milestone 3 — benchmark runner

Deliverables:

- `benchmarks/structural_interface_benchmark.py`;
- command-line options for dataset, k, seed, target type, output path;
- JSON/Markdown/HTML reports under `results/reports`;
- Windows make target:
  - `structural-interface-benchmark`.

Acceptance:

- benchmark can run at least WDBC and Wine Quality;
- benchmark summary explicitly separates proof-of-concept targets from
  independent targets;
- focused tests pass without network where possible, skip gracefully otherwise.

### Milestone 4 — independent target validation

Deliverables:

- held-out model-error target for WDBC/Wine/Iris/Dry Bean;
- optional temporal/perturbation target where data permits;
- report section: "non-circular validation".

Acceptance:

- TNFR compared against local disagreement, entropy, graph-TV, and label
  propagation residual;
- claims only accepted if TNFR adds either performance, robustness, or actionable
  grammar-valid prescription beyond the closest classical baseline.

### Milestone 5 — documentation and examples

Deliverables:

- `docs/STRUCTURAL_INTERFACE_THEORY.md`;
- update `docs/README.md`;
- update `README.md` examples section;
- optional notebook/report export target;
- one small "try it" example and one benchmark example.

Acceptance:

- all prose follows the repository's English-only and non-grandiose policy;
- limitations are explicit;
- examples are reproducible from a clean checkout with documented optional
  dependencies.

## Testing plan

Add focused tests:

```text
tests/test_structural_interface_api.py
tests/test_structural_interface_baselines.py
tests/test_structural_interface_benchmark.py
```

Required checks:

- graph construction preserves sample count;
- phase encoding is deterministic;
- TNFR scores are finite;
- operator prescriptions pass both grammar validators;
- report exporters create JSON/Markdown/HTML;
- online datasets skip cleanly on network failure;
- benchmark results include the closest classical baselines;
- no circular validation is reported as external superiority.

## Command integration

Proposed Windows make targets:

```text
structural-interface-benchmark
structural-interface-wdbc
structural-interface-wine
structural-interface-all
```

Existing related targets:

```text
external-phase-gate-validation
wdbc-phase-gate-demo
wine-quality-phase-gate-demo
```

## Acceptance criteria for a strong TNFR claim

A strong claim requires all of the following:

1. At least three real datasets across two or more sectors.
2. At least one non-circular target per sector.
3. TNFR compared against the closest classical graph-local baselines.
4. TNFR either:
   - improves ranking performance; or
   - matches performance while adding stable grammar-valid prescriptions; or
   - improves robustness/stability under graph perturbations.
5. Full report includes limitations and failure cases.
6. All tests and report-generation targets pass.

## First implementation sequence

1. Promote the reusable parts of examples 91 and 92 into
   `structural_interface.py`.
2. Add baseline functions and tests on toy graphs.
3. Build `benchmarks/structural_interface_benchmark.py` with WDBC + Wine.
4. Add held-out classifier-error target to avoid circularity.
5. Add Dry Bean or Iris as a third dataset.
6. Generate a consolidated report:
   `results/reports/structural_interface_benchmark.html`.
7. Update documentation and README with sober claims.

## Non-goals

- Do not claim clinical diagnosis, food-quality certification, or physical-law
  discovery.
- Do not claim superiority if the target is identical to local disagreement.
- Do not add a new TNFR operator.
- Do not mutate graph state during validation; prescriptions remain read-only
  until a separate execution layer is designed.

## Summary

Structural Interface Theory can become a concrete TNFR contribution if it is
implemented as a fair, reproducible cross-domain benchmark.  The key scientific
discipline is to compare TNFR against the strongest classical graph-local
baselines and to reserve the strongest TNFR claim for the combination of:

```text
local interface detection + tetrad telemetry + grammar-valid prescription
```
