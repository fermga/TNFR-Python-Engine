# TNFR Structural Interface Theory

## Status

Implemented diagnostic interfaces for graph and time-series observations.
The historical numerical summaries below are reported results, not a completed
validation of the nodal law or a reproducibility certificate. Their data/run
provenance and current physical-admission boundary are centralized in
[the empirical record](EMPIRICAL_CONFRONTATION_EEG.md); the research execution
plan owns further validation, rather than this guide maintaining another queue.

This is an **operational framework**, not a new fundamental physical law.  It
reuses the existing TNFR Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and the
13 canonical operators; it adds no new operator. Recommendations do not execute
operators. Graph construction and phase encoding are explicit preparation
steps, while field readers may populate internal caches; the entire interface
is not a transaction that leaves every graph attribute untouched.

## Executive summary

A **structural interface** is a graph-local region where neighbouring nodes are
close under the graph relation but differ sharply in phase, state, label,
measurement band, or regime.  Structural Interface Theory ranks such regions
from TNFR phase telemetry and expresses the diagnosis as a configured operator
recommendation for an already-active state:

```text
real system -> graph / proximity construction -> phase or state field
            -> local interface stress (tetrad telemetry)
            -> contextual operator recommendation (not live admission)
```

The framework supplies three observational settings. Historical comparisons
are scoped to their reported preparations and targets:

| Setting | Module | Native field role | Honest verdict |
| --- | --- | --- | --- |
| Static spatial | [structural_interface.py](../src/tnfr/validation/structural_interface.py) | Phase encodes an injected label | Reported rankings vary by dataset; label-propagation residual is an essential comparator |
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
- coherence length `ξ_C` with its product-fit, spectral-fallback or unavailable
  provenance; only the fit branch has path-length units;
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

Historically reported ranking power (ROC-AUC) against classifier errors. The
reported target differs from local disagreement; independent replay must still
verify the split, label availability, graph construction and preprocessing.
The circular "local-disagreement" target gives
≈ 1.0 for all local scores and is used only as a localization sanity check.

| Dataset | TNFR | local disagreement | graph TV | local entropy | label-prop residual | errors / N |
| --- | --- | --- | --- | --- | --- | --- |
| WDBC (breast cancer) | **0.9590** | 0.9493 | 0.9493 | 0.9345 | 0.9563 | 12 / 569 |
| Iris | **0.9860** | 0.9820 | 0.9820 | 0.9695 | 0.9850 | 7 / 150 |
| Digits | 0.6984 | 0.6962 | 0.6962 | 0.6980 | **0.8200** | 146 / 1797 |
| Wine quality (red) | 0.8739 | 0.8623 | — | — | **0.9423** | — |

The reported rankings include stronger label-propagation performance on
digits and wine red. Small differences on WDBC and Iris lack an uncertainty
statement here, and the low error counts matter. These summaries establish
neither superiority nor a non-noise mechanism without a pinned replay and
appropriate uncertainty analysis.

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

The historical power-grid report gives a classical variance trend (Kendall-τ ≈
0.255) slightly **beats** the strongest TNFR channel (Φ_s, τ ≈ 0.184), and both
are weak (< 0.26). Those statistics do not establish the physical mechanism
behind the signal or a general limit of either method.

Variance and autocorrelation are necessary comparators for this preparation.
Multichannel data permit additional graph-local observations; their availability
does not itself establish added predictive value.

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

- **ξ_C** — a static pressure-coherence product fit, or the separately
  identified spectral fallback;
- **K_φ** — phase curvature.

The amplitude-envelope pressure proxy and Hilbert phase are different functions
of the same signals. Different formulas do not prove statistical independence,
causal relevance or additional predictive information. Their dependence and
the fit/fallback branch must be measured on the retained evaluation data.

### Result (EEG Eye State real data)

Historically reported discrimination (ROC-AUC) of eyes-open versus eyes-closed
labels; the table does not supply a pinned replay or uncertainty interval:

| Indicator | AUC |
| --- | --- |
| phase dispersion (baseline) | **0.641** |
| ξ_C (TNFR) | 0.615 |
| Kuramoto R (baseline) | 0.559 |
| mean PLV (baseline) | 0.530 |

These reported AUCs order the selected scores in this preparation. The gap
of about 0.026 between phase dispersion and ξ_C does not establish statistical
equivalence or superiority. Neither this ranking nor the formulas alone prove
that ξ_C adds information after controlling for the other observables.

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
runs the static-spatial pipeline on a synthetic two-cluster graph and a
synthetic multi-channel regime switch, printing a baseline comparison and
contextual operator recommendations.

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

The P1 engineering boundary below is implemented; P2 physical admission remains
`not_admitted` and the current programme's empirical outcome is `not_tested`.
The historical numerical tables above were not rerun by these changes. The
[five-stage plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md) owns stage
status; the [passive transport protocol](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md)
owns the proposed measurement conditions.

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

### Modal diagnostics — `tnfr.validation.signal_confrontation`

- `confront_signal` returns `SignalConfrontation` with static graph read-outs
  and a `modal_diagnostic`; its coherence assumes `dEPI=0` explicitly.
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

### Reserved nodal forecasts — `tnfr.validation.nodal_prediction`

These APIs are also exported through `tnfr.validation`:

- `NodalMeasurementRun`: detached observations, channel/time/unit identity
  and declared `acquisition_id`; cropped or renamed records retain their
  preparation identity.
- `calibrate_nodal_prediction`: calibration-only common positive capacity
  and baseline on independently specified support, offsets/scales and clock
  bridge; returns immutable `FrozenNodalCalibration`.
- `forecast_nodal_response`: consumes calibration, a disjoint declared
  `evaluation_acquisition_id`, reserved run identity, initial measurement,
  timestamps and fixed budgets. It advances EPI via the shared nodal integrator.
- `write_nodal_forecast`: saves the issued prediction. Retain its
  `content_hash` before reading reserved observations.
- `score_nodal_forecast(..., expected_forecast_hash=issued_hash)`: checks
  that retained digest, calibration and observation identities before scoring.
  A passing numerical budget still has physical status
  `not_admitted_by_this_score`.

These are refreshed-Euler engineering forecasts, not automatic admission of
a continuous physical model. Declared acquisition IDs and hashes do not prove
independent laboratory preparation or trusted chronology. The
[empirical record](EMPIRICAL_CONFRONTATION_EEG.md#frozen-calibration-and-reserved-forecasts)
describes artifact admission, and
[example 159](../examples/10_applications/159_empirical_confrontation_pipeline.py)
demonstrates prediction-before-response order on a synthetic P2 fixture.

### Continuous P2 uncertainty — `tnfr.validation.p2_transport`

`P2MeasurementBounds` freezes independently supplied sensor and clock bounds.
`calibrate_p2_transport` uses fixed first/last observations to enclose continuous
common capacity on known two-node passive support; it does not reuse the P1
Euler increment coefficient. `forecast_p2_transport` issues exact rational
coordinate/mean/contrast enclosures without reserved response values.
`write_p2_transport_forecast` saves those endpoints exactly, and
`score_p2_transport` requires the retained forecast hash and disjoint acquisition
identity. Its outcomes are `incompatible_with_declared_bounds` or
`not_falsified_by_enclosures`; interval overlap does not prove one joint latent
fit. Common-time sensor alignment and joint error bounds remain declared
assumptions requiring evidence. The
[P2 annex](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md#implemented-continuous-time-uncertainty-boundary)
owns the derivation and physical admission conditions. The public package and
typing exports include these APIs; example 159 demonstrates their synthetic
continuous path separately from Euler.

The separately named `bound_fixed_reference_capacity` and
`bound_fixed_reference_transport` in
[p2_transport_reference.py](../src/tnfr/physics/p2_transport_reference.py)
cover capacities `(nu,0)` and a constant reference. Their contrast decay rate
is `nu`, not `2*nu`; the arithmetic mean is not conserved. They share the
rational log/exp kernels and do not fabricate a second measured channel.
The [Volts benchmark](../benchmarks/volts_fixed_reference_exploration.py)
uses that reference for a predeclared within-acquisition exploration only;
it leaves the strict disjoint-acquisition scoring API unchanged. Its nominal
arithmetic enclosures are not instrument error bounds. Data ingestion requires
Python 3.11 or later and the pinned optional `research-data` extra. Physical
mapping and results are owned by the same P2 annex.

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
- Grammar derivations: [grammar/PHYSICS_VERIFICATION.md](grammar/PHYSICS_VERIFICATION.md)
- Primary theory: [AGENTS.md](../AGENTS.md)
- Scheffer et al. (2009), *Early-warning signals for critical transitions*, Nature.
- Dakos et al. (2012), *Methods for detecting early warnings of critical
  transitions in time series*, PLoS ONE.
