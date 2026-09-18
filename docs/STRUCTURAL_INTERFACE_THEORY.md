# TNFR Structural Interface Theory

## Status

Active.  This document describes a completed, reproducible TNFR programme for
**structural-interface analysis** on real graph and time-series data.  It
consolidates the earlier planning work and reports the validated results,
including the cases where classical baselines win.

This is an **operational framework**, not a new fundamental physical law.  It
reuses the existing TNFR Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and the
13 canonical operators; it adds no new operator and mutates no graph state
during validation.

## Executive summary

A **structural interface** is a graph-local region where neighbouring nodes are
close under the graph relation but differ sharply in phase, state, label,
measurement band, or regime.  Structural Interface Theory ranks such regions
from TNFR phase telemetry and expresses the diagnosis as a grammar-valid
operator prescription:

```text
real system -> graph / proximity construction -> phase or state field
            -> local interface stress (tetrad telemetry)
            -> grammar-valid operator prescription
```

The framework is evaluated in three settings, each with its own module, honest
verdict, and failure cases:

| Setting | Module | Native field role | Honest verdict |
| --- | --- | --- | --- |
| Static spatial | [structural_interface.py](../src/tnfr/validation/structural_interface.py) | Phase encodes an injected label | Competitive with local classical baselines; the strongest global baseline (label-propagation residual) wins on hard data |
| Temporal single-series | [temporal_interface.py](../src/tnfr/validation/temporal_interface.py) | Phase is **measured** (Hilbert) | Classical critical-slowing-down indicators are the right tool for a single scalar series |
| Multi-channel | [multichannel_interface.py](../src/tnfr/validation/multichannel_interface.py) | Phase is **measured** per channel | ξ_C and K_φ are genuinely distinct from the Kuramoto order parameter; ξ_C is competitive on real EEG |

The distinctive TNFR contribution is the combination
`local interface detection + tetrad telemetry + grammar-valid prescription`,
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

All interface observables derive from existing canonical fields (see
[STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)):

- edge phase-gate compliance (U3 resonant-coupling condition
  `|wrap(φᵢ − φⱼ)| ≤ Δφ_max`);
- phase-gradient stress `|∇φ|` (local desynchronisation);
- phase-curvature stress `|K_φ|` (geometric phase torsion);
- structural potential `Φ_s` (global pressure, reported as telemetry, not folded
  into the ranking);
- coherence length `ξ_C` (spatial correlation scale), where meaningful;
- incident gate-violation pressure;
- a grammar-valid operator prescription.

### Operator prescription

Prescriptions are **read-only recommendations**.  Every prescribed sequence
passes the repository's sequence validators
(`tnfr.operators.grammar_patterns` and `tnfr.operators.grammar_dynamics`).  The
three validated patterns are:

| Interface state | Sequence | Meaning |
| --- | --- | --- |
| Fully phase-compatible | `UM → RA → SHA` | couple, propagate resonance, close |
| Mostly compatible with local hotspots | `IL → UM → SHA` | stabilize, then guarded coupling |
| Failed interface / boundary hotspot | `IL → OZ → THOL → SHA` | stabilize, open controlled reorganization, self-organize, close |

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

Every static benchmark compares TNFR against the full classical baseline suite
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

Ranking power (ROC-AUC) of each score against held-out classifier errors.  These
are the **non-circular** numbers; the circular "local-disagreement" target gives
≈ 1.0 for all local scores and is used only as a localization sanity check.

| Dataset | TNFR | local disagreement | graph TV | local entropy | label-prop residual | errors / N |
| --- | --- | --- | --- | --- | --- | --- |
| WDBC (breast cancer) | **0.9590** | 0.9493 | 0.9493 | 0.9345 | 0.9563 | 12 / 569 |
| Iris | **0.9860** | 0.9820 | 0.9820 | 0.9695 | 0.9850 | 7 / 150 |
| Digits | 0.6984 | 0.6962 | 0.6962 | 0.6980 | **0.8200** | 146 / 1797 |
| Wine quality (red) | 0.8739 | 0.8623 | — | — | **0.9423** | — |

**Honest reading.**  TNFR's interface stress edges the *simpler* local baselines
(local disagreement, graph total variation, local entropy) on clean datasets,
and on WDBC and Iris it also edges the label-propagation residual.  It does
**not** dominate that strongest global baseline in general: the
**label-propagation residual beats TNFR on the harder, noisier datasets**
(digits 0.820 vs 0.698; wine red 0.942 vs 0.874).  On Wine red the remaining
baselines are weak (graph cut 0.845; mean neighbour distance 0.472; degree
0.531; feature deviation 0.409; random ≈ 0.5), confirming the target is a real
boundary signal and not noise.

## Setting 2 — temporal single-series interfaces

### Temporal pipeline

```text
real time series -> Hilbert instantaneous phase -> delay-embedding proximity graph
                 -> per-window TNFR tetrad -> Kendall-τ trend toward a transition
                 -> comparison vs classical early-warning signals
```

Here the phase is **measured**, not injected.  The classical baselines are the
standard early-warning signals (EWS) for critical slowing down: rolling variance
and lag-1 autocorrelation (Scheffer et al. 2009; Dakos et al. 2012).

### Result (grid-frequency real data)

On real power-grid frequency data the classical variance trend (Kendall-τ ≈
0.255) slightly **beats** the strongest TNFR channel (Φ_s, τ ≈ 0.184), and both
are weak (< 0.26).  Grid frequency is a fast stochastic signal rather than a
slow bifurcation, so neither approach has a strong pre-transition trend.

**Honest reading.**  For a single scalar series, classical critical-slowing-down
indicators are the appropriate tool.  TNFR's added value appears in the
multi-channel setting, where a coherence *length* and a phase *curvature* exist.

## Setting 3 — multi-channel coupled oscillators

### Multi-channel pipeline

```text
multi-channel signals -> per-channel Hilbert phase + amplitude
                      -> phase-locking coupling graph (nodes = channels)
                      -> per-window spatial tetrad
                      -> synchrony discrimination vs Kuramoto order parameter R
```

This is the tetrad's native setting.  The gold-standard baseline is the
Kuramoto order parameter `R`; secondary baselines are mean phase-locking value
(PLV) and phase dispersion.

### Honest redundancy caveat

The phase-gradient field `|∇φ|` is **partially redundant** with `1 − R`: both
measure global desynchronisation.  The genuinely distinct fields are:

- **ξ_C** — a coherence *length* (spatial correlation scale), which has no
  order-parameter analogue;
- **K_φ** — phase curvature.

Because the structural pressure ΔNFR is derived from the amplitude envelope
(phase-independent), Φ_s and ξ_C are not trivial reproductions of `|∇φ|`.

### Result (EEG Eye State real data)

Discrimination (ROC-AUC) of the eyes-open vs eyes-closed regime:

| Indicator | AUC |
| --- | --- |
| phase dispersion (baseline) | **0.641** |
| ξ_C (TNFR) | 0.615 |
| Kuramoto R (baseline) | 0.559 |
| mean PLV (baseline) | 0.530 |

**Honest reading.**  ξ_C (0.615) beats the Kuramoto order parameter (0.559) and
mean PLV (0.530); phase dispersion (0.641) edges ξ_C.  The gap between the best
TNFR field and the best baseline is ≈ 0.026, which the framework reports as
**comparable** rather than as a TNFR win.  The point is that ξ_C and K_φ carry
information the global order parameter cannot express, not that TNFR dominates.

## How to run

All benchmarks have offline defaults (synthetic fixtures or bundled scikit-learn
data) and skip gracefully when an online dataset is unreachable.  Set
`PYTHONPATH` to `./src` first.

### Try it (offline, deterministic)

```bash
python examples/10_applications/93_structural_interface_demo.py
```

[examples/10_applications/93_structural_interface_demo.py](../examples/10_applications/93_structural_interface_demo.py)
runs the static-spatial pipeline on a synthetic two-cluster graph and a
synthetic multi-channel regime switch, printing the honest baseline comparison
and a grammar-valid prescription.

### Benchmarks (Windows make targets)

| Target | Setting | Data |
| --- | --- | --- |
| `structural-interface-offline` | static spatial | bundled scikit-learn (offline) |
| `structural-interface-all` | static spatial | WDBC + Wine + Iris + Digits |
| `structural-interface-wdbc` | static spatial | WDBC |
| `structural-interface-wine` | static spatial | UCI Wine Quality (online) |
| `structural-interface-model-error` | static spatial | held-out model-error target |
| `temporal-interface-benchmark` | temporal | synthetic fixture (offline) |
| `temporal-interface-grid` | temporal | real grid frequency (online, cached) |
| `multichannel-interface-benchmark` | multi-channel | synthetic Kuramoto (offline) |
| `multichannel-interface-eeg` | multi-channel | real EEG Eye State (online, cached) |

Example:

```bash
.\make.cmd structural-interface-offline
.\make.cmd multichannel-interface-benchmark
```

Reports are written to `results/reports/` as JSON, Markdown, and HTML.

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
- The label-propagation residual is a strong global baseline that **beats TNFR
  on hard static datasets** (digits, wine red); this is reported, not hidden.
- For a single scalar time series, classical critical-slowing-down indicators
  are preferred; TNFR's value is multi-channel.
- In the multi-channel setting `|∇φ|` is partially redundant with `1 − R`; only
  ξ_C and K_φ are genuinely distinct.
- No new TNFR operator is introduced; no graph state is mutated during
  validation (prescriptions are read-only).

## References

- Field definitions: [STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)
- Grammar derivations: [grammar/PHYSICS_VERIFICATION.md](grammar/PHYSICS_VERIFICATION.md)
- Primary theory: [AGENTS.md](../AGENTS.md)
- Scheffer et al. (2009), *Early-warning signals for critical transitions*, Nature.
- Dakos et al. (2012), *Methods for detecting early warnings of critical
  transitions in time series*, PLoS ONE.
