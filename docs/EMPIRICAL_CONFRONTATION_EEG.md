# Reported empirical observations and diagnostic limits

**Status:** P1 engineering complete; P2 physical admission `not_admitted`;
the admitted physical protocol remains `not_tested`. A predeclared Volts
continuation has been computed as an exploration within one acquisition;
its [result and evidence record](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md#volts-exploratory-record)
report lower error for the affine control than for the nodal model. It does
not close physical admission. The historical numerical summaries below have
not been independently reproduced in this
checkout from a pinned dataset/run manifest. They are retained as reported
results, not admitted validation of the complete nodal equation.

## Implemented diagnostic boundary

The [research strategy](../theory/NODAL_RESEARCH_STRATEGY.md) distinguishes
implementation defects from scientific scope limits. The P1 repairs implement
the following boundary:

- Modal-analysis exceptions return `status="failure"` with a reason. Short,
  constant or degenerate valid windows return `status="unresolved"`; neither
  becomes a fabricated zero fraction or favorable regime label. Malformed or
  nonfinite input is rejected before analysis.
- The diffusion score fits graph, normalization and coefficients on the same
  window it evaluates. Its unconstrained training improvement over persistence
  is nonnegative by construction; it is not held-out predictive skill.
- The legacy descriptive coefficient is retained without clipping and exposes
  `capacity_domain`, including negative and inactive cases. The separate frozen
  calibration API rejects nonpositive or unidentifiable capacity for its
  positive-capacity model. Real/complex AR(2) roots and their growing, decaying
  or unit-boundary behavior are distinct statistics; none certifies a physical
  diffusion or conservative wave.
- PLV neighborhoods and amplitude-pressure proxies do not provide a complete
  canonical EPI/capacity state or establish U3-admissible coupling.

The engineering boundary is implemented and tested; it supplies neither a
calibrated laboratory state map nor a physical law validation. Stage status and
the subsequent gates remain owned by the
[five-stage execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md).
The [passive transport protocol](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md)
records P2's unmet measurement conditions. This empirical record has no separate
task queue, and the historical numerical results below were not rerun by P1.

### Public diagnostic migration

[`diagnose_modal_roots`](../src/tnfr/validation/signal_confrontation.py)
returns `ModalRootDiagnostic` with `status`, `reason`, `root_classification`,
`stability` and fitted/unresolved/growing/decaying/boundary mode counts.
`resolved` means the selected numerical fits are identified, not that a TNFR
model is admitted. The affine AR(2) intercept is a statistical nuisance term
that avoids spurious damping from incomplete-period mean removal; it is not
an added TNFR pressure channel.

`SignalConfrontation.modal_diagnostic` retains that result. The legacy names
`wave_fraction` and `emergent_wave_fraction` now return a complex-root energy
fraction or `None` when unavailable. `diffusive_face_valid` is a deprecated
`bool | None` descriptive majority alias: it also returns `None` for any
growing fitted mode. Even `True` is not a diffusion certificate. Consumers must
handle `None` explicitly, never write `"diffusion" if value else "wave"`.
`summary()` labels the diagnostic scope; `to_dict()` preserves null abstentions
and names nonfinite read-outs in `nonfinite_readouts` for strict JSON.

## Purpose

TNFR's structural read-outs (the tetrad `|∇φ|`, `K_φ`, `Φ_s`, `ξ_C`;
the auxiliary graph-wave spectrum `ω_k = √λ_k`; static pressure coherence
`C_static`; structural capacity `νf`) are
meant to be **confronted with real data** — tests that *can* fail. The engine
ships the **data-agnostic instrument**; it does **not** bundle datasets.

- **Instrument (in-engine):** [`confront_signal`](../src/tnfr/validation/signal_confrontation.py)
  maps a supported finite real multichannel signal → a **constructed phase-locking graph**
  ([`build_coupling_graph`](../src/tnfr/validation/multichannel_interface.py),
  PLV observational graph, not a U3 certificate) → the scoped read-outs
  (tetrad, pulse, `ξ_C` via the graph's
  spectral gap, static pressure coherence `C_static`) → a **modal-root
  diagnostic**
  ([`diagnose_modal_roots`](../src/tnfr/validation/signal_confrontation.py):
  project the signal onto `L_sym` eigenmodes and classify fitted AR(2) roots.
  This is a statistical modal diagnostic; root type alone establishes neither
  a conservative wave nor a stable diffusive nodal trajectory).
  The signal window has no TNFR EPI-rate channel. Consequently the public
  ``coherence`` result evaluates ``structural_coherence(mean|DeltaNFR|, 0)``.
  It is a static snapshot read-out, not canonical dynamic ``C(t)`` and not an
  attractor or convergence certificate.
  Demo: [example 159](../examples/10_applications/159_empirical_confrontation_pipeline.py)
  (`python examples/10_applications/159_empirical_confrontation_pipeline.py path/to/signal.npy`).
- **Findings below** come from the **TNFR-IA empirical arm** (a separate,
  downstream repository that consumes this engine) run on the public CHB-MIT
  scalp-EEG corpus. The prior report assigns the data and scripts to TNFR-IA;
  no pinned external commit or complete replay envelope is supplied here.

## Reported CHB-MIT findings (external empirical arm)

| Confrontation | Result | Reading |
|---|---|---|
| **Local phase tetrad** `\|∇φ\|`, `\|K_φ\|` vs seizure/interictal | cross-patient (leave-one-patient-out) **AUC ≈ 0.66–0.67**, higher at ictal | reported association with labels; independent replay remains pending |
| **Global Kuramoto `R`** vs seizure | **AUC ≈ 0.49** (chance) | reported lack of label discrimination in this preparation; does not by itself identify a local mechanism or exclude global synchrony effects |
| **`νf`** (single structural frequency) | tracks the measured dominant EEG frequency (**Spearman ρ ≈ 0.86**); fingerprints subjects | reported association does not identify structural mobility with physical oscillation frequency |
| **Single-`νf` conservative-wave forecaster** (few-shot, leave-one-patient-out) | **ties** a strong Gaussian process and a stabilised DMD/Koopman model (within ~1%); beats a parameter-matched neural net | reported accuracy with 1–3 model parameters and ~20× lower cost than the neural TNFR; physical identification and replay remain pending |
| **Face diagnosis** (`verify_overdamped_projection` at the data-fitted `γ`) | reported under-damped classification | a fitted auxiliary-model result, not certification of physical conservative dynamics |
| **Pre-ictal** (5 min before onset) tetrad | every magnitude reported at **chance** | no predictive evidence in the reported five-minute preparation; concurrent association does not establish a general detection or prediction rule |

## Reported cross-subject confrontation (PhysioNet eegmmidb)

A second, independent confrontation run **through the shipped instrument**
(`confront_signal`, historical data-fitted face) on the PhysioNet **eegmmidb** eyes-open
(R01) vs eyes-closed (R02) baselines — 10 subjects, 64-channel, drift removed
(`fft_bandpass` 1–40 Hz), paired across subjects with a two-sided binomial sign
test.  Data are **not** bundled (fetched to a scratch dir); the run is
not independently reproducible from the supplied description alone: exact
run scripts, subject/file hashes, preprocessing and evaluation records remain
to be pinned and replayed.

| Confrontation (open→closed) | Result | Reading |
|---|---|---|
| **Modal root classifier** (AR(2) on `L_sym`) | reported **WAVE / under-damped in 18/20** conditions | descriptive classification; the reported thermal comparison (97–99% diffusive labels) also needs acquisition/replay provenance and is not a physical certificate |
| **`K_φ`** (phase curvature, local) | rises in **10/10** subjects (paired dz ≈ +1.1, sign-test **p ≈ 0.002**) | reported cross-subject association in this sample; independent replay remains pending |
| **`\|∇φ\|`** (phase gradient, local) | rises in **9/10** (**p ≈ 0.021**) | reported cross-subject association, pending replay |
| **`C_static`** (pressure snapshot) / **`Q`** (quality factor) | both rise (8/10 each) | reported trends; `C_static` is pressure-only (p ≈ 0.11), not dynamic total `C(t)` or proof of a sharper physical rhythm |
| **Global Kuramoto `R`** | **falls** in 9/10 (dz ≈ −0.9) | reported direction differs from the local-field trends in this sample; no general diagnostic ordering follows |
| **Collective pulse** `ω₀=√λ₂` | structural `ω₀ ∈ [0.12, 0.27]` vs measured `1–11 Hz` (Spearman ≈ −0.3); flat open↔closed (dz ≈ −0.2, **p = 1.0**) | graph-derived auxiliary wave scale, without an identified conversion to temporal Hz; the reported absence of discrimination does not prove topology is state-insensitive |

**Reported correction.** The cross-subject summary gives the opposite local-field
direction to the earlier single-subject reading (eyes-closed ⇒ `|∇φ|` down).
If replay confirms those results, they would reject that directional
generalization for this preparation. A local/global diagnostic difference does
not establish a travelling wave, show that one observable has the correct sign
and the other the wrong one, or identify a TNFR mechanism. Fixing k in a PLV
neighbor graph does not fix which neighbors its data-dependent construction
selects. Both acquisition provenance and the actual constructed graphs are
needed to interpret the reported result.

## Dynamics confrontation (the nodal equation as a predictor)

Beyond static read-outs, an in-sample diffusion-direction fit is available
([`nodal_prediction_skill`](../src/tnfr/validation/signal_confrontation.py)):
the EPI channel of `∂EPI/∂t = ν_f·ΔNFR` is graph diffusion, so its one-step
predictor `x̂(t+1) = x(t) − c·L_rw·x(t)` (a single fitted diffusion step
`c = ν_f·dt`) is scored by the one-step increment variance it explains beyond
persistence, against a per-channel AR-1 baseline. Both models are currently
fitted and scored on the same supplied data. The table preserves reported
scores; it does not establish out-of-sample forecasting.

| Signal | nodal 1-step skill | AR-1 | Reading |
|---|---|---|---|
| synthetic graph diffusion | **+0.16–+0.21** (recovers `c`) | +0.03 | reported synthetic same-window recovery; not rerun here or physical validation |
| real EEG (eyes-closed) | +0.02 | **+0.05** | reported same-window AR-1 score exceeds the selected diffusion-direction fit |
| real thermal field (60–120 min steps) | **+0.015–+0.029** | +0.013–+0.025 | reported same-window scores are close; this comparison alone does not certify a diffusive physical regime |

For nonzero increments `r` and diffusion direction `d`, the fitted improvement
is `(r dot d)^2/(||r||^2*||d||^2) >= 0`. A positive score alone therefore
cannot establish that the physical nodal law predicts unseen observations.
The scores are descriptive; a frozen measurement map, admissible capacity,
run-level holdout and controlled intervention are required for the next test.
The legacy `NodalPredictionSkill` now states
`evaluation_scope="same_window_descriptive_fit"`; its `beats_persistence`
property describes only the training residual.

## Frozen calibration and reserved forecasts

The public [`tnfr.validation`](../src/tnfr/validation/__init__.py) exports
the separate [nodal prediction API](../src/tnfr/validation/nodal_prediction.py):

1. `NodalMeasurementRun` retains channel identities, real timestamps, units,
   finite observations and a declared `acquisition_id`. A renamed or cropped
   record from one preparation keeps that acquisition identity.
2. `calibrate_nodal_prediction` takes calibration runs, independently specified
   support, fixed sensor offsets/scales and a finite positive clock bridge.
   `FrozenNodalCalibration` detaches nested content and freezes the common
   positive capacity and comparison baseline. Its finite-increment fit is
   scoped to refreshed Euler dynamics, not automatically a continuous-rate
   identification from laboratory measurements.
3. `forecast_nodal_response` takes that calibration, a disjoint declared
   `evaluation_acquisition_id`, reserved run identity, only the initial
   measurement, timestamps and fixed numerical/error budgets. Each EPI advance
   uses the shared nodal integrator; no future response updates the model.
4. Retain `forecast.content_hash` and save the forecast with
   `write_nodal_forecast` before acquiring or reading the reserved response.
   `score_nodal_forecast(..., expected_forecast_hash=issued_hash)` requires that
   retained digest and checks calibration, run/acquisition identity, units,
   timestamps and initialization. It reports numerical error, with physical
   status `not_admitted_by_this_score`.

Acquisition identities are declarations, and a hash authenticates content,
not the truth of those declarations or human chronology. An
[`EvidenceSidecar`](../src/tnfr/research/evidence_sidecar.py) with a
[`CoreExperimentManifest`](../src/tnfr/research/core_manifests.py) additionally
verifies actual artifact bytes under `root_dir`; these file hashes differ
from a model's canonical content hash. The
[integration test](../tests/research/test_nodal_prediction_evidence.py) links
calibration, issued forecast and result artifacts, and rejects modified bytes.
[Example 159](../examples/10_applications/159_empirical_confrontation_pipeline.py)
demonstrates this order on a nine-sample analytic P2 software fixture. That
fixture is not laboratory evidence and does not replay the EEG tables above.

The same example now includes a separate continuous P2 interval fixture.
[`calibrate_p2_transport`](../src/tnfr/validation/p2_transport.py) encloses the
continuous capacity from fixed calibration endpoints and independently supplied
sensor/clock bounds; it does not relabel the Euler coefficient.
`forecast_p2_transport` issues a rational outer tube before reserved values
exist. `score_p2_transport` checks coordinate, mean and contrast boxes with
the retained forecast hash. Overlap reports `not_falsified_by_enclosures`,
not existence of one common latent fit or physical validation. The
[protocol annex](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md#implemented-continuous-time-uncertainty-boundary)
owns the derivation, assumptions and unchanged physical admission gaps.

The [temporal interface](../src/tnfr/validation/temporal_interface.py) separately
offers `window_tetrad_series(..., mode="prospective")`, with prefix-only
processing, warmup/latency and `available_at` indices.
`calibrate_temporal_warning` freezes channel selection;
`evaluate_prospective_warning` reports descriptive trends or explicit
unavailability. A supplied `transition_index` is a declared evaluation cutoff,
not a predicted event. These prefix descriptors are neither event predictors
nor replacements for the frozen nodal response model.

## Honest scope

- **Reported baseline comparison.** The external wave-model summary claims
  accuracy near GP and stabilized DMD. That comparison needs a replay manifest;
  it does not derive the auxiliary wave from canonical nodal evolution or
  establish physical meanings for its fitted parameters.
- **Concurrent association, not an admitted predictor.** The CHB-MIT summary
  reports concurrent AUC ≈ 0.67 and chance-level pre-ictal results. No clinical
  detection/prediction capability follows without replay and the appropriate
  evaluation protocol.
- **Modest reported margins.** Relative-MSE differences near 1% and the listed
  AUCs need uncertainty and information-matched comparisons; these summaries
  establish no competitive-performance claim.
- **Limited cross-subject preparation.** The eegmmidb report covers ten subjects
  in one eyes-open/closed comparison. Its local/global trends, if reproduced,
  would remain observations in that sample, not identification of spatial alpha
  dynamics or of the complete nodal law.
- **Modal diagnosis has limited scope.** Adequate samples do not turn AR(2)
  root classification into a certificate of diffusion, conservative dynamics
  or TNFR correspondence. Failure and unresolved fits now retain explicit
  statuses, and the old physical face labels are no longer generated.
- **Reported associations need provenance.** Label discrimination, model-fit
  scores and timing claims remain externally reported until their precise
  experiments are independently replayable. Even successful replay would
  establish only the measured association or restricted prediction.

See also [`docs/STRUCTURAL_INTERFACE_THEORY.md`](STRUCTURAL_INTERFACE_THEORY.md)
(the tetrad vs the Kuramoto order parameter on real EEG) and
[`theory/EMERGENT_ONTOLOGY.md`](../theory/EMERGENT_ONTOLOGY.md) §0 (the empirical
caveat this record nuances).
