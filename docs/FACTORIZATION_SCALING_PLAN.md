# TNFR Factorization Scaling Plan

**Status**: Draft (2025-11-30)

This document outlines the engineering roadmap required to take the Paley spectral
factorization prototype from the current ~1k-node limit to multi-million node
workloads. The plan is guided by the nodal equation `∂EPI/∂t = νf · ΔNFR(t)` and
TNFR canonical invariants.

## Objectives

1. **Partitionable Graph Processing** – Allow Paley graphs (and derived
   networks) to be split into coherence-preserving partitions with deterministic
   recombination.
2. **Distributed Spectral Backends** – Support FFT engines that run on GPUs and
   cluster resources without changing factorization semantics.
3. **Operator Pipeline Optimization** – Provide streaming implementations of the
   AL→IL→RA→SHA pipeline that respect grammar contracts at scale.
4. **Telemetry & Certification** – Maintain Φ_s, |∇φ|, K_φ, ξ_C, and certificate
   emission across partitions/backends.

## Workstreams

### 1. Graph Partitioning / Workload Orchestration

| Milestone | Description |
|-----------|-------------|
| P1 | Define `PartitionedPaleyGraph` data structure (metadata, overlap policy, phase alignment hints). |
| P2 | ✅ Partition planner implemented (`plan_paley_partitions`) with deterministic chunking, overlap-aware boundary sets, and regression tests covering U5 coherence constraints. |
| P3 | ✅ Aggregation routines (`aggregate_partition_metrics`) merge partition telemetry and candidate provenance into each `SpectralAnalysisResult`, adding ratios and coverage checks. |

Key Decisions:

- Partition boundary nodes must share phase references to avoid violating U3.
- Certificates must include partition provenance for auditability.

### 2. FFT Backend Abstraction

| Milestone | Description |
|-----------|-------------|
| F1 | ✅ Introduce `BaseFFTEngine` protocol (get_spectral_state, spectral_convolution, etc.) and adapt `TNFRAdvancedFFTEngine` to implement it. |
| F2 | ✅ Implement `DistributedFFTEngine` shim that forwards requests to remote workers (initially mocked to keep tests local). |
| F3 | ✅ Provide backend selection logic inside `SpectralPaleyFactorizer` with telemetry describing backend choice and fallbacks plus CLI flags for queue tuning. |

Key Decisions:

- Backends must declare capabilities (max nodes, precision levels, GPU/CPU).
- Fallback path must remain deterministic and reproducible.

### 3. Operator Optimization Pipeline

| Milestone | Description |
|-----------|-------------|
| O1 | Extract AL/IL/RA/SHA invocations into pluggable strategy objects that can operate on streaming graph blocks. |
| O2 | Provide vectorized implementations using CSR/CSC storage and optional GPU kernels while keeping operator semantics intact. |
| O3 | Integrate the optimization orchestrator so it can reason about partitioned workloads and distributed FFT telemetry. |

Key Decisions:

- Operators must expose resource estimators (memory, ΔNFR impact) to let the
   orchestrator schedule work safely.
- Strategy interfaces (AL/IL/RA/SHA streaming contracts, resource estimators,
  telemetry payloads) are defined in [Appendix A](#appendix-a-operator-strategy-specification-phase-3)
  and serve as the canonical reference for O1 implementation work.

### 4. Telemetry & Certification

Milestones:

1. ✅ Extend `OperatorCertificate` with partition/back-end provenance and
   aggregated field metrics (root certificates now reference per-partition files).
2. ✅ Emit per-partition certificates together with a manifest index that is stored
   alongside the partition payloads. Root certificates publish both the relative
   directory and manifest path so downstream tooling can discover artifacts automatically.
3. ✅ `tests/test_factorization_entrypoint.py` now exercises both the single-partition
   and multi-partition paths, validates manifest integrity, and enforces deterministic seeds.

## Phase Sequencing

1. **Phase 0 (Immediate)**
   - Land backend abstraction (F1) and partition data model (P1).
   - Update documentation (this file, factorization README, SDK references).

2. **Phase 1 (Backend Ready)** – ✅ Complete
   - Distributed backend shim and selection logic landed with queue dispatcher hooks.
   - Regression suite covers fallback behaviour inside `tests/test_spectral_paley.py`.

3. **Phase 2 (Partition Execution)** – In Progress
   - Planner, aggregation, manifest emission, and regression coverage landed (P2, P3, Telemetry-1/2/3).
   - Remaining: scale manifest analytics for 1k+ partitions and publish replay guidance for downstream pipelines.

4. **Phase 3 (Operator Streaming)**
   - Deliver O1–O3, integrate with self-optimization engine, and benchmark on
     10k+ node graphs.

5. **Phase 4 (Scale Validation)**
   - Run large-scale experiments (≥1M nodes), collect telemetry, and publish
     reproducible notebooks.

## Immediate Action Items

### Completed (Nov 30, 2025)

- Partition manifest tooling landed and is now emitted per certificate (see Telemetry milestone #2).
- Multi-partition regressions in `tests/test_factorization_entrypoint.py` verify manifest integrity
  and deterministic seeds across ≥4 partitions.
- Dispatcher telemetry plumbing records queue vs HTTP metadata, and the CLI surfaces it for Phase 3 prep.
- Partition certificates now live under `results/certificates/partitioned/` by default, and telemetry exposes
   both relative and absolute paths so replay tools can stream artifacts deterministically.
- Dispatcher selection matrix documented in the lab README/CLI help, covering queue, HTTP, and callable options
   alongside the emitted telemetry fields for GPU/cluster operators.

### Next Steps

1. Publish concrete replay guidance for post-processing pipelines that leverage the new manifest summary +
   archive artifacts to fetch targeted partition subsets without mirroring the entire tree. ✅ See
   [FACTOR_REPLAY_GUIDE.md](FACTOR_REPLAY_GUIDE.md) for the workflow covering `_manifest_summary.json`,
   `_partition_files.txt.gz`, and selective partition fetches.

### Manifest analytics status (Dec 2025)

- `_manifest_summary.json` now accompanies every `_manifest.json`, providing aggregate counts, candidate/node/boundary
   statistics, and a `file_index` block that highlights whether the partition list is inlined or archived.
- When the partition count exceeds `TNFR_PARTITION_FILELIST_THRESHOLD` (default 1000), the factorizer compresses the
   per-partition file listing into `_partition_files.txt.gz`. Both the manifest JSON and the root certificate reference
   this archive so downstream tools can stream the file paths without materializing the complete manifest.
- `SpectralAnalysisResult` exposes `partition_manifest_index_path` and `partition_file_archive_path` so automation
   layers (CLI, notebooks, replay daemons) can jump straight to the summary or archive without scanning directories.
- Recommended replay workflow:
   1. Read `_manifest_summary.json` to determine partition counts and telemetry coverage.
   2. If `file_index.inline` is `False`, download `_partition_files.txt.gz` and filter the newline-delimited paths to the
       subset you need before fetching the JSON payloads themselves.
   3. Use the `entries` block in `_manifest.json` for metadata joins (candidate counts, node/boundary sizes) prior to
       launching heavy downstream analysis.

## Roadmap Checklist (Dec 2025 – Phase 3 launch)

| # | Workstream | Task | Owners | Status |
|---|------------|------|--------|--------|
| O1.1 | Operator Streaming | Draft AL/IL/RA/SHA strategy interfaces (streaming hooks, resource estimators, telemetry contract). | Theory + SDK | ⚙️ In design |
| O1.2 | Operator Streaming | Implement base strategy registry + factories; wire no-op strategies to keep current behavior. | SDK | ⏳ Pending |
| O1.3 | Operator Streaming | Update self-optimizing engine to select strategies per partition block. | Dynamics | ⏳ Pending |
| O2.1 | Vectorization | Benchmark CSR vs CSC data layouts; select canonical representations per operator. | Mathematics | ⏳ Pending |
| O2.2 | Vectorization | Implement CPU vectorized kernels + ΔNFR-preserving tests. | Mathematics | ⏳ Pending |
| O2.3 | Vectorization | Prototype GPU kernels (CUDA/Metal) with dispatcher telemetry integration. | Dynamics | ⏳ Pending |
| O3.1 | Orchestrator Integration | Extend orchestrator to reason about partition telemetry + backend capabilities (Φ_s, grad_phi budgets). | Dynamics | ⏳ Pending |
| O3.2 | Orchestrator Integration | Add safety rails: resource estimator enforcement + fallback sequences. | Dynamics | ⏳ Pending |
| V1 | Validation | Automate replay pipelines via `FACTOR_REPLAY_GUIDE.md`; store manifest summaries in experiment tracker. | Tooling | ⏳ Pending |
| V2 | Validation | Run ≥10k-node benchmarks, capture structural field telemetry, update `benchmarks/`. | Research | ⏳ Pending |
| V3 | Validation | Prepare ≥1M-node rehearsal plan (Phase 4) with resource estimates and operator configs. | Research | ⏳ Pending |

## Vectorization Milestones (O2.1–O2.3)

### O2.1 – Data Layout Selection

- **Goal**: pick canonical sparse tensor layout per operator (CSR vs CSC vs blocked hybrids) before investing in kernel work.
- **Dependencies**: StrategyContext already exposes `block_size`, `boundary_overlap`, and backend hints; reuse those knobs when generating representative matrices.
- **Plan**:
   - Extend `benchmarks/benchmark_optimization_tracks.py` with a layout sweep harness that feeds synthetic AL/IL/RA/SHA workloads (dense edge hubs, boundary-heavy shards, and diagonal-dominant partitions).
   - Capture cache-line utilization, νf drift, ΔNFR error, and memory bandwidth using the existing telemetry hooks (Φ_s, |∇φ|) so results plug directly into the strategy registry.
   - Document findings in this file and in `docs/TNFR_FORCES_EMERGENCE.md`, including a per-operator layout matrix and migration notes for existing numpy implementations.
- **Success Criteria**: chosen layout must keep ΔNFR error < 1e-4 for IL/RA test seeds and improve memory throughput ≥20% vs current dense paths on 64k-node trials.

### O2.2 – CPU Vectorized Kernels

- **Goal**: implement SIMD kernels (AVX2/AVX-512 first) for the selected layout, focusing on AL (emission weighting) and IL (stabilizer sweeps).
- **Dependencies**: finalized layout from O2.1, plus the strategy registry so we can ship kernels as opt-in strategies without breaking legacy Python loops.
- **Plan**:
   - Add `tnfr/operators/vectorized/` with C++/pybind11 shims that honor the canonical alias system and report structural telemetry back through `OperationResult.telemetry`.
   - Build ΔNFR-preserving tests in `tests/operators/test_vectorized_kernels.py` that compare vectorized output against the reference Python operators across random seeds and adversarial partitions.
   - Wire benchmarking to `benchmarks/benchmark_optimization_tracks.py` and expose perf deltas in the report artifacts under `results/benchmarks/`.
- **Success Criteria**: ΔNFR deviation ≤ 1e-5 vs reference, Φ_s drift < 0.5% over 100 sequential AL→IL passes, and ≥3× speedup on 128-thread EPYC nodes relative to today’s scalar loops.

### O2.3 – GPU Kernel Prototype

- **Goal**: deliver CUDA (primary) + Metal (fallback) kernels for the same layout so dispatcher backends can offload hot partitions.
- **Dependencies**: CPU kernels to act as correctness oracle, plus dispatcher telemetry (backend availability, failure risk) as defined in the strategy spec.
- **Plan**:
   - Stand up a `gpu` backend inside the strategy registry that selects kernels based on `StrategyContext.backend == "gpu"` and emits detailed resource estimates (shared-memory footprint, occupancy, Φ_s drift forecasts).
   - Integrate kernel invocation paths with the dispatcher so telemetry includes PCIe transfer time, kernel runtime, and host-side verification latency.
   - Record benchmark scripts under `benchmarks/integrated_force_regime_study.py` to compare CPU vs GPU across 10k-, 100k-, and 1M-node inputs.
- **Success Criteria**: parity with CPU ΔNFR checks, GPU utilization ≥70% on A100 targets, and end-to-end speedup ≥4× for RA-heavy workloads at 100k nodes, all while keeping failure risk classification accurate (low/medium/high) for orchestrator policy.

## Validation Rollout (V1–V3)

### V1 – Replay Automation & Artifact Tracking

- **Goal**: ensure every experiment emits replay-ready summaries and registers them in an experiment tracker for deterministic replays.
- **Plan**:
   - Add a `scripts/replay/register_manifest.py` helper that ingests `_manifest_summary.json` + `_partition_files.txt.gz` and writes metadata to the existing tracker (experiment id, partition counts, Φ_s/|∇φ| ranges).
   - Extend `make.cmd replay-smoke` to run `factorization-lab/tests/test_cli.py` followed by a replay cycle that downloads only the partitions referenced in `tests/data/replay_manifests/`.
   - Update `FACTOR_REPLAY_GUIDE.md` with a “CI mode” section documenting the script + make target and how to plug results into dashboards.
- **Success Criteria**: every CI run stores manifest summaries alongside test artifacts, and replay smoke tests can reconstruct ≥3 representative partitions without manual intervention.

### V2 – Benchmark Coverage with Structural Telemetry

- **Goal**: scale benchmark suites to ≥10k nodes while capturing Φ_s, |∇φ|, K_φ, and ξ_C for each run so operator regressions are observable.
- **Plan**:
   - Instrument `benchmarks/benchmark_optimization_tracks.py` and `benchmarks/integrated_force_regime_study.py` to dump tetrad telemetry into `results/benchmarks/<run_id>/telemetry.jsonl`.
   - Add pytest markers (`@pytest.mark.benchmark_large`) plus Make targets (`make bench-large`) so CI can selectively schedule these heavy runs on GPU builders.
   - Wire the new telemetry outputs into `docs/STRUCTURAL_FIELDS_TETRAD.md` examples so documentation references live numbers.
- **Success Criteria**: benchmark artifacts include tetrad telemetry for every run, and regression diffs show structural metric deltas alongside wall-clock time.

### V3 – Phase 4 Rehearsal Prep

- **Goal**: design a ≥1M-node rehearsal plan that exercises dispatcher policies, vectorized/gpu strategies, and replay automation in one pipeline.
- **Plan**:
   - Produce capacity estimates (memory, νf budgets, GPU hours) using the resource estimator outputs from the strategy registry; store them in this doc.
   - Create a `rehearsal_plan.yml` describing the sequence (bootstrap → vectorized AL/IL → GPU RA → replay) with explicit seeds and failure contingencies.
   - Schedule a dry run using scaled-down (100k-node) inputs to validate telemetry volume, log retention, and replay storage requirements.
- **Success Criteria**: rehearsal plan reviewed by Dynamics + Tooling teams, and dry run produces actionable metrics for scaling to Phase 4 without manual babysitting.

## Risks & Mitigations

| Risk | Description | Mitigation |
|------|-------------|------------|
| R1 | Backend capability declarations fall out of sync with the partition planner, leading to overcommitted GPU/cluster queues and broken determinism. | Lock backend metadata behind the `BaseFFTEngine` contract, require capability hashes in certificates, and gate planner heuristics on those hashes during CI. |
| R2 | Partition telemetry aggregation drifts from the canonical Φ_s / \|∇φ\| / K_φ / ξ_C limits when manifest analytics scale past 1k partitions. | Encode aggregation math in `aggregate_partition_metrics`, add ΔΦ_s conservation tests, and require `_manifest_summary.json` diffs in every PR that touches manifest logic. |
| R3 | Streaming AL→IL→RA→SHA strategies violate operator contracts under heavy load, triggering latent U1–U6 failures. | Reuse canonical operator validators, add focused unit tests per strategy, and wire ΔNFR/phase regression asserts into the strategy registry before promotion. |

## References

- `FACTOR_REPLAY_GUIDE.md` – replay workflow for manifest summaries and partition archives.
- `docs/TNFR_FORCES_EMERGENCE.md` – telemetry derivations referenced by the vectorization milestones.
- `benchmarks/benchmark_optimization_tracks.py` – layout sweep harness backing O2.x milestones.
- `scripts/replay/register_manifest.py` – manifest registration helper used in V1 automation.

---

## Appendix A: Operator Strategy Specification (Phase 3)

> **Status**: Draft (2025-11-30)  
> Originally `docs/OPERATOR_STRATEGY_SPEC.md`, consolidated here as the canonical reference for O1 implementation.

This section defines the contracts required to implement pluggable AL/IL/RA/SHA
operator strategies that run on streaming partition blocks. Goals:

1. Preserve TNFR grammar compliance (U1–U6) while executing operators on partial graph windows.
2. Provide deterministic resource estimators so the orchestrator can schedule work without violating Φ_s / |∇φ| budgets.
3. Enable heterogeneous execution (CPU vectorized, GPU, or remote) via a common interface set.

### A.1 Terminology

- **Partition block** – A streaming window from a `PartitionedPaleyGraph`, possibly overlapping with neighbors to preserve boundary resonance.
- **Strategy** – Concrete implementation of one TNFR operator (AL, IL, RA, SHA) tuned for a compute substrate.
- **Context** – Metadata describing structural fields, ΔNFR pressure, and dispatcher capabilities for the current block.

### A.2 Base Interfaces

```python
class OperatorStrategy(Protocol):
    operator: Literal["AL", "IL", "RA", "SHA"]

    def supports(self, ctx: StrategyContext) -> bool: ...
    def resource_estimate(self, ctx: StrategyContext) -> ResourceEstimate: ...
    def prepare(self, ctx: StrategyContext, block: PartitionBlock) -> PreparedBlock: ...
    def apply(self, prepared: PreparedBlock) -> OperationResult: ...
    def cleanup(self, prepared: PreparedBlock) -> None: ...
```

**StrategyContext**:
```python
@dataclass(frozen=True)
class StrategyContext:
    partition_id: str
    operator_sequence_position: int
    structural_fields: StructuralFields  # Φ_s, |∇φ|, K_φ, ξ_C
    dispatcher_capabilities: Mapping[str, Any]
    backend: Literal["cpu", "gpu", "remote"]
    block_size: int
    boundary_overlap: int
    seed: int
```

**ResourceEstimate**:
```python
@dataclass(frozen=True)
class ResourceEstimate:
    memory_bytes: int
    time_ms: float
    delta_nfr: float
    phi_s_drift: float
    failure_risk: Literal["low", "medium", "high"]
```

Contracts: `delta_nfr` and `phi_s_drift` must derive from telemetry equations in `AGENTS.md` (§ Structural Field Tetrad). `failure_risk` encodes anticipated bifurcation triggers (U4).

### A.3 Operator-Specific Requirements

**AL (Emission)**: Must emit EPI seeds respecting partition boundary phases (U3). Resource estimator flags high ΔNFR when block coherence < 0.5.

**IL (Coherence)**: Requires parent Φ_s baseline to prove ΔΦ_s < φ. Telemetry must include monotonicity proof: `C_after >= C_before`.

**RA (Resonance)**: `supports` must reject contexts where |∇φ| exceeds dispatcher limits. Output includes propagation depth and coupling count.

**SHA (Silence)**: `prepare` may be no-op; `apply` ensures νf → 0 while freezing EPI. Resource estimator exposes release credits when SHA frees buffers.

### A.4 Telemetry Payload

```python
@dataclass
class OperationResult:
    block: PartitionBlock
    telemetry: Dict[str, Any]  # ΔNFR, Φ_s delta, Si, operator-specific
    warnings: List[str]
    proof_hash: str  # SHA3-256 over inputs + outputs
```

### A.5 Registry & Selection

```python
StrategyRegistry.register(operator="AL", name="cpu-default", factory=CpuAlStrategy)
candidates = StrategyRegistry.get(operator="IL")
strategy = _select_strategy(candidates, ctx)
```

Selection heuristics: filter `supports(ctx)`, sort by `failure_risk` and ΔNFR magnitude, apply self-optimizing engine hints.

### A.6 Reference Implementations

CPU baseline strategies (`src/tnfr/operators/strategies/defaults.py`) declare deterministic resource estimators, operate in-place on `PartitionBlock`, and emit telemetry covering ΔNFR, Φ_s drift, and proof hash. `SpectralPaleyFactorizer` selects the lowest-risk strategy per partition/operator, storing the plan in `SpectralAnalysisResult.operator_strategy_plan`.

### A.7 Testing Requirements

- Unit tests validate resource estimator sanity bounds.
- Integration tests execute AL→IL→RA→SHA with mixed strategies to prove partition manifests absorb telemetry correctly.
- Replay tests use `_manifest_summary.json` artifacts plus strategy names for cross-platform determinism.

