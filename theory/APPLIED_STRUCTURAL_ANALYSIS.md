# Applied structural analysis

**Status: experimental engineering reference; no factorization-complexity or
physical-emergence theorem.** This document connects applications to their
implemented entry points. The [theory index](README.md) owns classification;
the [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) owns priorities.
The preceding claim-heavy version is retained in the [archive](research/archive/README.md).

## 1. Scope

The factorization lab constructs residue graphs, applies spectral and nodal
processing, and scores candidate periodicities. These are implemented search
heuristics. The nodal identity does not prove that a selected periodicity divides
an arbitrary input or prescribe the confidence/endorsement thresholds.
Demonstrations must state which arithmetic information and fallbacks were used.

## 2. Spectral factorization

### 2.1 Statement

The [public wrapper](../src/tnfr/factorization/__init__.py) delegates to the lab's
[API](../factorization-lab/tnfr_factorization/api.py). The
[Paley-Jacobi pipeline](../factorization-lab/tnfr_factorization/spectral_paley.py)
constructs spectral candidates, may process partitions through a decoder,
and emits telemetry and optional certificates. An output named
`tnfr_certified_factors` records the pipeline's structural verification outcome;
the name alone is not an arithmetic proof.

### 2.2 Algorithm and configured verification

The pipeline combines graph construction, spectral analysis, partitioning,
configured operator processing, candidate inference and validation. Read active
configuration and recorded operator words from the actual run; a schematic word
such as UM/RA/IL/THOL is not by itself a standalone grammar certificate.

Pressure gain, coherence ratios, field deltas, periodicity confidence, coverage
and partition endorsements are selected filters. Their numerical cuts are
engineering policies, not constants derived uniquely from the nodal equation.
High confidence is not a substitute for checking `1 < d < n` and `n % d == 0`.

### 2.5 Pure mode

`TNFR_PURE_MODE=1` skips arithmetic hints and gcd refinement in the local
candidate-construction stage. It does not make the complete pipeline
arithmetic-free: analysis unconditionally computes prime-factorization telemetry,
its empty-candidate trial-division fallback has no pure-mode guard, and partition
size hints can use divisibility. `TNFR_PURE_MODE_VERIFY_DIVISIBILITY=1` filters
initial seeds; it is not a universal proof for later heuristic acceptance.
The verifier records `support_divisible` without requiring it in `pass_all`.
Report these dependencies when attributing success to TNFR-specific processing.
The [lab guide](../factorization-lab/README.md) owns current flag and output details.

The lab's spectral quantities named potential, gradient and curvature are proxies,
not the canonical tetrad computed by `tnfr.physics.fields`. Their scale and
meaning must not be substituted for the canonical field bounds.

### 2.6 Output artifacts

`FactorizationResult` exposes candidates, structurally accepted candidates,
telemetry, effective mode and an optional certificate path. Artifact contents
and available provenance depend on the path/options; do not assume every output
contains a seed and hash. Preserve input, source, configuration, candidate list,
arithmetic verification and fallback provenance for reproducible comparisons.

## 3. Implementation

| Component | Owner |
| --- | --- |
| API and result fields | [api.py](../factorization-lab/tnfr_factorization/api.py) |
| Candidate/refinement/verification paths | [spectral_paley.py](../factorization-lab/tnfr_factorization/spectral_paley.py) |
| Partition planning | [partitioning.py](../factorization-lab/tnfr_factorization/partitioning.py) |
| Seed handling | [seed_management.py](../factorization-lab/seed_management.py) |
| Snapshot/replay | [snapshot_system.py](../factorization-lab/snapshot_system.py) |
| CLI and use | [lab README](../factorization-lab/README.md) |

## 4. Test coverage and evaluation boundary

The [lab tests](https://github.com/fermga/TNFR-Python-Engine/tree/main/factorization-lab/tests) exercise finite examples, verification
policies, false-positive controls, seeds, partitioning and replay. They do not
prove universal factor recovery or a complexity improvement. Reserved evaluation
must freeze thresholds before scoring, disclose labels entering construction,
compare against declared baselines and report failures as well as successes.
No new performance benchmark is claimed by this documentation audit.

## Implementation & Examples

Riemann prime-ladder and von Mangoldt examples illustrate separate arithmetic
constructions. Their logarithmic rates are assigned in those models; they are
not a derived universal NFR clock. Dirichlet-series and analytic-continuation
statements retain their domains. Current scope and source inventory are in
[Riemann notes](TNFR_RIEMANN_RESEARCH_NOTES.md), and operator correspondence
limits are in [arithmetic operators](TNFR_ARITHMETIC_OPERATORS.md).

## 6. References

- [Foundations](FUNDAMENTAL_THEORY.md): definitions and state-space obligations.
- [Structural interfaces](../docs/STRUCTURAL_INTERFACE_THEORY.md): engineering
  pipeline and evaluation requirements.
- [Measurement protocol](research/PASSIVE_TRANSPORT_PROTOCOL.md): independent
  admission for the supporting physical bridge.
