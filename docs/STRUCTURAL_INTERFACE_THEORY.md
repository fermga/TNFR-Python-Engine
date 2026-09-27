# TNFR Observational Interface Guide

## Status

Implemented diagnostic interfaces for graph and time-series observations,
plus separate frozen-calibration forecast APIs. This guide owns their current
engineering contracts; field definitions belong to the
[tetrad guide](STRUCTURAL_FIELDS_TETRAD.md), and physical admission and priorities
belong to the [research execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#supporting-measurement-bridge).
Historical numerical summaries are retained in the
[reported-observation archive](../theory/research/archive/REPORTED_INTERFACE_OBSERVATIONS.md),
not as independently reproduced validation or current API evidence. This guide
maintains no separate research queue.

This is an **operational framework**, not a new fundamental physical law.  It
reuses the existing TNFR Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and the
13 canonical operators; it adds no new operator. Recommendations do not execute
operators. Graph construction and phase encoding are explicit preparation
steps, while field readers may populate internal caches; the entire interface
is not a transaction that leaves every graph attribute untouched.

## Executive summary

A **structural interface** is a graph-local region where neighbouring nodes are
close under the graph relation but differ sharply in phase, state, label,
measurement band, or regime. The interface implementation ranks such regions
from TNFR phase telemetry and expresses the diagnosis as a configured operator
recommendation for an already-active state:

```text
real system -> graph / proximity construction -> phase or state field
            -> local interface stress (tetrad telemetry)
            -> contextual operator recommendation (not live admission)
```

The framework supplies three observational settings:

| Setting | Module | Native field role | Evaluation boundary |
| --- | --- | --- | --- |
| Static spatial | [structural_interface.py](../src/tnfr/validation/structural_interface.py) | Phase encodes an injected label | Compare against label disagreement and label-propagation residual; disclose label availability |
| Temporal single-series | [temporal_interface.py](../src/tnfr/validation/temporal_interface.py) | Phase is estimated from an analytic signal | Compare against variance and autocorrelation with the same information and split |
| Multi-channel | [multichannel_interface.py](../src/tnfr/validation/multichannel_interface.py) | Per-channel phase estimates on a PLV observation graph | Local phase fields and pressure-product diagnostics have different inputs; predictive independence must be tested |

The distinctive TNFR contribution is the combination
`local interface detection + tetrad telemetry + contextual recommendation`,
not a claim of universal superiority over classical graph metrics.

## Core concept

### Structural interface

Examples of structural interfaces:

- tumour samples that are morphologically close but diagnostically different;
- chemical samples that are similar but assigned to different quality bands;
- sensors that are physically adjacent but report incompatible phases;
- a time series approaching a regime change (bifurcation / transition);
- a network of oscillators crossing a synchronisation transition.

### TNFR observables

The interface combines canonical field read-outs with separate phase gates and
recommendation policies (see
[STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)):

- edge phase-gate compliance (U3 resonant-coupling condition
  `|wrap(φᵢ − φⱼ)| ≤ Δφ_max`);
- phase-gradient stress `|∇φ|` (local desynchronisation);
- phase-curvature stress `|K_φ|` (local circular-mean mismatch);
- structural potential `Φ_s` (global pressure, reported as telemetry, not folded
  into the ranking);
- coherence length `ξ_C`; only its product-fit branch has path-length units.
  The shared estimator can expose provenance, but the multichannel numeric
  series does not retain its fit/fallback branch;
- incident gate-violation pressure;
- a configured operator recommendation with declared grammar rationale.

### Operator prescription

Prescriptions are **read-only recommendations for an already-active state**.
The producer in [phase_gate.py](../src/tnfr/validation/phase_gate.py) selects
tuples and records a grammar rationale; it does not run a sequence validator
or execute those tuples. The
[phase-gate tests](../tests/test_phase_gate_api.py) and
[interface tests](../tests/test_structural_interface_api.py) check representative
patterns using explicit `initial_epi_nonzero=True` context and initialized
graphs. The principal nonempty-support network patterns are:

| Interface state | Sequence | Meaning |
| --- | --- | --- |
| Fully phase-compatible | `UM → RA → SHA` | couple, propagate resonance, close |
| Mostly compatible with local hotspots | `IL → UM → SHA` | stabilize, then guarded coupling |
| Failed interface / boundary hotspot | `IL → OZ → THOL → SHA` | stabilize, open controlled reorganization, self-organize, close |

Additional branches return `SHA` for an edgeless graph or `IL → SHA` for
selected node guidance. None of these recommendations is a standalone
initiation certificate. Validate the actual word/history context and refresh
state-dependent U3 admission before execution. The selection does not prove
causal efficacy, future stress reduction or an autonomous intervention law.

## Setting 1 — static spatial interfaces

### Static pipeline

```text
records -> z-scored k-NN proximity graph -> binary state encoded as phase
        -> per-node interface stress -> ranking vs classical baselines
        -> non-circular target evaluation (ROC-AUC, precision@review)
```

When a binary state is encoded into phase (positive class at φ = 0, negative at
φ = π), the TNFR interface stress is, **by construction**, related to the
classical k-NN label-disagreement baseline.  It is therefore reported *beside*
that baseline, not as an independent discovery.  Non-circular claims require an
independent target through `evaluate_interface_scores`.

### Fair benchmark design

The static benchmark entry point computes the shared classical baseline suite
([interface_baselines.py](../src/tnfr/validation/interface_baselines.py)):

1. local k-NN disagreement (closest classical analogue);
2. graph total variation;
3. local class entropy;
4. label-propagation residual;
5. graph-cut contribution;
6. mean neighbour distance;
7. degree / topology-only;
8. a simple domain-feature baseline;
9. constant / random control.

Non-circular targets (at least one required for any claim): independent
expert/review label, **held-out downstream model error**, temporal transition,
perturbation sensitivity, or an explicit classical-interface target.

### Results (held-out model-error target)

Historical static rankings and their replay limits are retained in the
[reported observations](../theory/research/archive/REPORTED_INTERFACE_OBSERVATIONS.md#reported-static-ranking).
They do not establish superiority or supply a current benchmark result.

## Setting 2 — temporal single-series interfaces

### Temporal pipeline

```text
real time series -> Hilbert instantaneous phase -> delay-embedding proximity graph
                 -> per-window TNFR tetrad -> Kendall-τ trend toward a transition
                 -> comparison vs classical early-warning signals
```

Here phase is estimated through the selected Hilbert transform, rather than
assigned from a class label. This signal coordinate is not automatically the
primitive TNFR phase. The classical baselines are the
standard early-warning signals (EWS) for critical slowing down: rolling variance
and lag-1 autocorrelation (Scheffer et al. 2009; Dakos et al. 2012).

### Result (grid-frequency real data)

The historical grid-frequency trends are retained in the
[reported observations](../theory/research/archive/REPORTED_INTERFACE_OBSERVATIONS.md#reported-grid-frequency-trends).
Variance and autocorrelation remain necessary comparators; no reported trend
alone establishes a transition mechanism.

## Setting 3 — multi-channel coupled oscillators

### Multi-channel pipeline

```text
multi-channel signals -> per-channel Hilbert phase + amplitude
                      -> phase-locking coupling graph (nodes = channels)
                      -> per-window spatial tetrad
                      -> synchrony discrimination vs Kuramoto order parameter R
```

The supplied observation graph permits spatial field read-outs. Relevant
comparators include the Kuramoto order parameter `R`, mean phase-locking value
(PLV) and phase dispersion. PLV construction is an observational choice, not
measured canonical wiring or a complete nodal state map.

### Honest redundancy caveat

The phase-gradient field `|∇φ|` measures local neighbor mismatch, whereas `R`
uses a global phasor sum. They can correlate, but neither generally determines
the other on an arbitrary graph. Two further read-outs are:

- **ξ_C** — a static pressure-coherence product fit or a spectral fallback;
- **K_φ** — phase curvature.

The amplitude-envelope pressure proxy and Hilbert phase are different functions
of the same signals. Different formulas do not prove statistical independence,
causal relevance or additional predictive information. Their dependence and
the fit/fallback branch must be measured on the retained evaluation data.

`MultichannelWindowSeries.xi_c` retains numeric estimates only. It cannot tell
whether a sample came from the product fit or spectral fallback; the current
`SynchronyDiscrimination` report does not restore that evidence. Obtain
`estimate_coherence_length_with_provenance` on the actual prepared graph when
branch interpretation matters, rather than inferring it from the value.

The multichannel evaluator reports `max(AUC, 1-AUC)` after assigning each window
its majority label. This orientation-free score uses the supplied evaluation
labels to choose direction; it is descriptive discrimination, not an oriented
classifier frozen on calibration data or a reserved prediction. No finite
scores or a one-class window sample return the helper's neutral 0.5 convention,
which is not evidence that an informative two-class comparison was available.

### Result (EEG Eye State real data)

The historical EEG Eye State rankings are retained in the
[reported observations](../theory/research/archive/REPORTED_INTERFACE_OBSERVATIONS.md#reported-eeg-eye-state-ranking).
They lack a pinned replay and uncertainty interval and are not reserved forecasts.

## How to run

Install the working repository as described in [TESTING.md](../TESTING.md).
Use explicit dataset/source flags for an offline run: the static CLI defaults
to `--dataset all`, the temporal CLI to `--source grid`, and the multichannel
CLI to `--source eeg`. Those defaults can request network data. Static bundled
datasets additionally require scikit-learn; it is not a core TNFR dependency.

### Try it (offline, deterministic)

```bash
python examples/10_applications/93_structural_interface_demo.py
```

[examples/10_applications/93_structural_interface_demo.py](../examples/10_applications/93_structural_interface_demo.py)
runs the static-spatial pipeline on a synthetic two-cluster graph and
concatenated independently prepared sinusoidal blocks with dispersed and
aligned phases. It prints a baseline comparison and contextual operator
recommendations; the join is not an observed dynamical transition. The
multichannel benchmark's synthetic source similarly concatenates independently
initialized low/high-coupling Kuramoto blocks.

### Benchmark entry points

The historical `make.cmd` wrapper is absent. Run the maintained Python CLIs
from the repository root, selecting the intended inputs explicitly:

```bash
python benchmarks/structural_interface_benchmark.py --dataset offline --target model_error
python benchmarks/temporal_interface_benchmark.py --source synthetic
python benchmarks/multichannel_interface_benchmark.py --source synthetic
```

Inspect each script's `--help` for optional real-data sources and limits. The
static default target is `circular`, a localization check, so the example above
selects the separate model-error target explicitly. Its information/split
limitations still require review before a predictive claim.

All three CLIs default to `results/reports/` and accept `--output`. The static
suite writes per-dataset JSON/Markdown/HTML and a JSON summary; temporal and
multichannel CLIs write JSON. Inspect returned status and skip reasons: missing
data or dependencies can yield skipped reports, which are not validation passes.

## API reference

These APIs distinguish implemented engineering behavior from physical
admission. The [execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md)
owns stage status; the [passive transport protocol](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md)
owns independent measurement conditions. Historical reports are not replayed
by API regression tests.

### Static spatial — `tnfr.validation.structural_interface`

- `StructuralInterfaceProblem`, `StructuralInterfaceScore` — frozen dataclasses;
- `build_knn_graph(records, feature_keys, *, k=10, ...)` — z-scored k-NN graph;
- `encode_phase_from_binary_state(G, state_key, *, positive_value, ...)` —
  in-place phase encoding;
- `score_structural_interfaces(problem_or_graph, *, state_key=None, ...)` —
  list of `StructuralInterfaceScore`;
- `interface_score_maps`, `baseline_score_maps`, `full_baseline_score_maps` —
  node → score maps;
- `evaluate_interface_scores(labels, score_maps)` — ROC-AUC and
  precision@review-count per score;
- `render_structural_interface_markdown` / `_html`,
  `export_structural_interface_report`.

### Temporal — `tnfr.validation.temporal_interface`

- `TemporalInterfaceConfig`, `WindowTetradSeries`, `EarlyWarningComparison`;
- `hilbert_instantaneous_phase`, `delay_embedding`, `local_structural_pressure`,
  `build_temporal_proximity_graph`;
- `window_tetrad_series(signal, *, config=None, mode="retrospective",
  warmup_samples=0, latency_samples=0)`;
- `rolling_variance`, `rolling_lag1_autocorrelation`, `kendall_tau`;
- `evaluate_early_warning(signal, *, transition_index=None, config=None)`.

The default path and `evaluate_early_warning` are retrospective descriptors.
`mode="prospective"` processes each window from its available prefix and
records inclusive `available_at` indices, including declared latency. A
window-end timestamp alone does not establish availability.
`calibrate_temporal_warning` freezes feature configuration and TNFR/baseline
channel selection on a declared calibration run;
`evaluate_prospective_warning` uses those choices without refitting and returns
`descriptive_evaluation` or `unavailable`, with a reason and optional trends.
`transition_index` is a supplied cutoff, not a detected or predicted event.
Independent preparations and their provenance remain protocol obligations;
prefix-only processing by itself establishes no event-prediction performance.

### Multi-channel — `tnfr.validation.multichannel_interface`

- `MultichannelConfig`, `MultichannelWindowSeries`, `SynchronyDiscrimination`;
- `fft_bandpass`, `analytic_phase_amplitude`, `phase_amplitude_matrices`;
- `kuramoto_order_parameter`, `phase_locking_matrix`, `phase_offsets`,
  `amplitude_pressure`, `build_coupling_graph`;
- `multichannel_window_series(signals, *, config=None)`;
- `evaluate_synchrony_discrimination(signals, labels, *, config=None)`.

The PLV graph and amplitude-pressure proxy are observational read-outs, not
canonical wiring, a complete EPI/capacity state map or a U3 certificate.

Labels must be finite binary values, one per signal sample, before window
processing. `SynchronyDiscrimination.metadata` retains `auc_available` and
`auc_unavailable_reason` (`no_windows`, `missing_positive_class` or
`missing_negative_class`). Unavailable scores retain legacy `0.5` values only
for compatibility; the interpretation and summary mark them unavailable,
not chance-level measured performance. The benchmark reuses one computed
series for rankings and block summaries, and preserves these flags in its
strict JSON report. Corrupt input rows are rejected rather than removed from
the sample clock; missing class means are `null`.

### Modal diagnostics — `tnfr.validation.signal_confrontation`

- `confront_signal` returns `SignalConfrontation` with static graph read-outs
  and a `modal_diagnostic`; its coherence assumes `dEPI=0` explicitly. It
  constructs an observational PLV graph, not measured canonical wiring.
- `diagnose_modal_roots` returns `ModalRootDiagnostic`: `resolved`, `unresolved`
  or `failure`, with a reason. Real/complex root majority and fitted
  growing/decaying/unit-boundary behavior are separate statistics.
- Legacy `wave_fraction`/`emergent_wave_fraction` are complex-root energy
  fractions or `None`; deprecated `diffusive_face_valid` is `bool | None` and
  abstains on failure, unresolved fits or growing modes. Neither Boolean
  value certifies a physical regime. Handle `None` explicitly; it must not
  fall through to a wave label.
- `SignalConfrontation.summary()` reports the diagnostic scope; `to_dict()`
  preserves null abstentions and lists nonfinite read-outs for strict JSON.
- `nodal_prediction_skill` remains a same-window descriptive fit with
  `evaluation_scope="same_window_descriptive_fit"` and `capacity_domain`.
  Its unconstrained coefficient is not clipped into physical admissibility;
  its nonnegative training improvement is not held-out forecasting evidence.

For supported finite signals, `confront_signal` projects onto graph `L_sym`
modes and fits affine AR(2) models. The intercept prevents incomplete-period
means from being forced into the homogeneous recurrence; it is a statistical
nuisance term, not an added nodal pressure channel. Short, constant or
degenerate valid data can be `unresolved`; computational exceptions become
`failure` with a reason. Malformed or nonfinite inputs are rejected.

This adapter's `xi_c` is specifically the graph-wave reciprocal scale
`1/omega_0`, where the pulse owner selects the smallest eigenvalue above
`1e-9` and sets `omega_0` to its square root. Thus `xi_c=1/sqrt(lambda_2)`
only on connected support with a resolved gap; when no positive mode is
selected, `xi_c` is infinite. The adapter does not run the tetrad product fit. Its `coherence` is
`structural_coherence(mean(abs(DeltaNFR)), 0)`, not measured dynamic `C(t)` or
a convergence certificate. Graph spectral scales have no automatic conversion
to laboratory frequency or length.

For nonzero observed increments `r` and fitted diffusion direction `d`, the
unconstrained same-window improvement over persistence in exact arithmetic is
`(r dot d)^2 / (||r||^2 * ||d||^2) >= 0`. Its positivity and the legacy
`NodalPredictionSkill.beats_persistence` describe training residuals. A frozen
measurement map, admitted capacity and independent reserved acquisition are
separate obligations.

<a id="frozen-calibration-and-reserved-forecasts"></a>
### Reserved nodal forecasts — `tnfr.validation.nodal_prediction`

These APIs are also exported through `tnfr.validation`:

- `NodalMeasurementRun`: detached observations, channel/time/unit identity
  and declared `acquisition_id`; cropped or renamed records retain their
  preparation identity.
- `calibrate_nodal_prediction`: calibration-only common positive capacity
  and baseline on independently specified support, offsets/scales and clock
  bridge; returns immutable `FrozenNodalCalibration` with detached nested
  content. Nonpositive or unidentifiable capacity is rejected for this model.
- `forecast_nodal_response`: consumes calibration, a disjoint declared
  `evaluation_acquisition_id`, reserved run identity, initial measurement,
  timestamps and fixed budgets. It advances EPI via the shared nodal integrator.
- `write_nodal_forecast`: saves the issued prediction. Retain its
  `content_hash` before reading reserved observations.
- `score_nodal_forecast(..., expected_forecast_hash=issued_hash)`: checks
  that retained digest, calibration, run/acquisition identity, units,
  timestamps and initialization before scoring.
  A passing numerical budget still has physical status
  `not_admitted_by_this_score`.

These are refreshed-Euler engineering forecasts. A fitted finite-increment
coefficient is not automatically a continuous physical rate. Declared
acquisition IDs and hashes do not prove independent laboratory preparation
or trusted chronology.

An [EvidenceSidecar](../src/tnfr/research/evidence_sidecar.py) with a
[CoreExperimentManifest](../src/tnfr/research/core_manifests.py) additionally
checks actual artifact bytes under `root_dir`; those file digests differ from
a model's canonical content hash. The
[integration test](../tests/research/test_nodal_prediction_evidence.py) links
calibration, issued forecast and result artifacts and rejects modified bytes.
[Example 159](../examples/10_applications/159_empirical_confrontation_pipeline.py)
demonstrates prediction-before-response order on a synthetic fixture. It is
not laboratory evidence and does not replay the archived EEG tables.

### Continuous P2 uncertainty — `tnfr.validation.p2_transport`

The [P2 protocol annex](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md#implemented-continuous-time-uncertainty-boundary)
owns `P2MeasurementBounds`, continuous calibration, rational forecast enclosures,
hash/acquisition checks and their admission limits. That model does not reuse
the Euler increment coefficient. Interval overlap is not proof of one joint
latent fit or physical validation. Example 159 demonstrates its software
fixture separately from Euler.

The [Volts model boundary](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md#volts-model-boundary)
owns the earlier fixed-reference comparison and its limits. The separately
declared driven [TCLab protocol](../theory/research/TCLAB_EXPLORATORY_PROTOCOL.md)
owns its calibration/reserved comparisons and source evidence. Neither result
admits the full physical model or replaces the primary research queue.

## Limitations and non-goals

- No clinical diagnosis, food-quality certification, or physical-law claims.
- No superiority claim where the target is identical to local disagreement
  (those cases are reported as localization sanity checks at ≈ 1.0 AUC).
- Historical rankings and small numerical gaps require pinned replay, an
  information-matched evaluation and uncertainty before generalization.
- Single-series and multichannel settings need appropriate comparators;
  graph-local field availability is not a predictive advantage by itself.
- Different field formulas do not certify independent information. Preserve
  ξ_C estimator provenance and distinguish a pressure proxy from nodal pressure.
- No new TNFR operator is introduced. Recommendations do not execute operators;
  preparation writes and internal field caches are distinct from that promise.

## References

- Field definitions: [STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)
- Grammar contracts and verification: [Unified grammar](../theory/UNIFIED_GRAMMAR_RULES.md#9-verification-and-reporting)
- Primary theory: [AGENTS.md](../AGENTS.md)
- Scheffer et al. (2009), *Early-warning signals for critical transitions*, Nature.
- Dakos et al. (2012), *Methods for detecting early warnings of critical
  transitions in time series*, PLoS ONE.
