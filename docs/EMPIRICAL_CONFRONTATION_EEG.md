# Reported empirical observations and diagnostic limits

**Status:** externally reported observations and implemented diagnostics.
The numerical summaries below have not been independently reproduced in this
checkout from a pinned dataset/run manifest. They are retained as reported
results, not admitted validation of the complete nodal equation.

## Current audit boundary

The [2026-09-17 strategic review](../theory/NODAL_RESEARCH_STRATEGY.md)
identifies defects and scope limits that must be resolved before the proposed
laboratory programme uses these APIs as evidence:

- Modal-analysis exceptions currently become `wave_fraction=0` and a favorable
  diffusive label. A failed diagnostic must instead be unresolved.
- The diffusion score fits graph, normalization and coefficients on the same
  window it evaluates. Its unconstrained training improvement over persistence
  is nonnegative by construction; it is not held-out predictive skill.
- The fitted diffusion coefficient is not constrained to positive capacity.
  Real versus complex AR(2) roots do not certify diffusion or conservative waves.
- PLV neighborhoods and amplitude-pressure proxies do not provide a complete
  canonical EPI/capacity state or establish U3-admissible coupling.

These runtime issues remain open. The review corrects the interpretation and
priorities; it does not silently repair or certify the empirical pipeline.
The repair tasks and subsequent measurement/evaluation gates are owned by
the [five-stage execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md),
starting at P1.1. This empirical record does not maintain a separate queue.

## Purpose

TNFR's structural read-outs (the tetrad `|∇φ|`, `K_φ`, `Φ_s`, `ξ_C`;
the pulse `ω_k = √λ_k`; static pressure coherence `C_static`; structural
frequency `νf`) are
meant to be **confronted with real data** — tests that *can* fail. The engine
ships the **data-agnostic instrument**; it does **not** bundle datasets.

- **Instrument (in-engine):** [`confront_signal`](../src/tnfr/validation/signal_confrontation.py)
  maps any real multichannel signal → the **emergent phase-locking graph**
  ([`build_coupling_graph`](../src/tnfr/validation/multichannel_interface.py),
  PLV observational graph, not a U3 certificate) → the scoped read-outs
  (tetrad, pulse, `ξ_C` via the graph's
  spectral gap, static pressure coherence `C_static`) → the **emergent two-face
  diagnosis**
  ([`emergent_wave_fraction`](../src/tnfr/validation/signal_confrontation.py):
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
| **Global Kuramoto `R`** vs seizure | **AUC ≈ 0.49** (chance) | global hypersynchrony does **not** rise under the bipolar montage / focal onset — the state is *local*, not mean-field |
| **`νf`** (single structural frequency) | tracks the measured dominant EEG frequency (**Spearman ρ ≈ 0.86**); fingerprints subjects | reported association does not identify structural mobility with physical oscillation frequency |
| **Single-`νf` conservative-wave forecaster** (few-shot, leave-one-patient-out) | **ties** a strong Gaussian process and a stabilised DMD/Koopman model (within ~1%); beats a parameter-matched neural net | reported accuracy with 1–3 model parameters and ~20× lower cost than the neural TNFR; physical identification and replay remain pending |
| **Face diagnosis** (`verify_overdamped_projection` at the data-fitted `γ`) | reported under-damped classification | a fitted auxiliary-model result, not certification of physical conservative dynamics |
| **Pre-ictal** (5 min before onset) tetrad | every magnitude at **chance** | the tetrad is a **detection** marker (concurrent), **not** a **prediction** marker |

## Reported cross-subject confrontation (PhysioNet eegmmidb)

A second, independent confrontation run **through the shipped instrument**
(`confront_signal`, data-fitted face) on the PhysioNet **eegmmidb** eyes-open
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
| **`C_static`** (pressure snapshot) / **`Q`** (quality factor) | both rise (8/10 each) | a sharper rhythm; `C_static` is a pressure-only trend (p ≈ 0.11), not dynamic total `C(t)` |
| **Global Kuramoto `R`** | **falls** in 9/10 (dz ≈ −0.9) | mean-field synchrony moves **opposite** to the local fields — the local tetrad and `R` **dissociate** |
| **Collective pulse** `ω₀=√λ₂` | structural `ω₀ ∈ [0.12, 0.27]` vs measured `1–11 Hz` (Spearman ≈ −0.3); flat open↔closed (dz ≈ −0.2, **p = 1.0**) | the pulse `ω_k=√λ_k` is the coupling graph's **spatial / topological** standing-wave spectrum — **not** the temporal Hz spectrum, and not a state marker here (the k-NN PLV topology is state-insensitive by construction); the *local* tetrad carries the state |

**Falsification and correction.** The naive single-subject reading (eyes-closed
⇒ `|∇φ|` *down*) did **not** generalise: cross-subject, `|∇φ|` and `K_φ` rise
while `R` falls.  Measured first, then reported — the cross-subject test
corrected a premature single-subject conclusion.  The consistent structural fact
is a **local↔global dissociation**: eyes-closed alpha is a spatially-structured
rhythm (local phase curvature up, global in-phase order down), an established
phenomenon (alpha as a spatial / travelling wave) that the canonical *local*
tetrad reads correctly where the mean-field order parameter mis-signs it.
Recovers a known phenomenon structurally; closes nothing, beats nothing, is not
new physics.

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
| synthetic graph diffusion | **+0.16–+0.21** (recovers `c`) | +0.03 | validated: the predictor recovers a genuine diffusion law |
| real EEG (eyes-closed) | +0.02 | **+0.05** | oscillatory — the EPI *diffusion* channel is weak; AR-1 wins |
| real thermal field (60–120 min steps) | **+0.015–+0.029** | +0.013–+0.025 | diffusive — the nodal diffusion **matches/edges** AR-1 at diffusive timescales |

For nonzero increments `r` and diffusion direction `d`, the fitted improvement
is `(r dot d)^2/(||r||^2*||d||^2) >= 0`. A positive score alone therefore
cannot establish that the physical nodal law predicts unseen observations.
The scores are descriptive; a frozen measurement map, admissible capacity,
run-level holdout and controlled intervention are required for the next test.

## Honest scope

- **Reported baseline comparison.** The external wave-model summary claims
  accuracy near GP and stabilized DMD. That comparison needs a replay manifest;
  it does not derive the auxiliary wave from canonical nodal evolution or
  establish physical meanings for its fitted parameters.
- **Detection, not prediction.** The tetrad tracks the seizure while it happens
  (AUC ≈ 0.67), not its approach (pre-ictal at chance).
- **One substrate, modest margins.** Single corpus; relative-MSE margins over the
  baselines are ~1%; AUC ≈ 0.67 is not a competitive seizure detector.
- **Cross-subject, still one paradigm.** The eegmmidb generalisation is n = 10,
  a single eyes-open/closed paradigm; effect sizes are medium–large (`K_φ`
  p ≈ 0.002) but `C` is only a trend. It recovers a known phenomenon (spatial
  alpha) rather than out-predicting a baseline.
- **Modal diagnosis has limited scope.** Adequate samples do not turn AR(2)
  root classification into a certificate of diffusion, conservative dynamics
  or TNFR correspondence. Diagnostic failure must be distinguished from either
  label before further empirical use.
- **Reported associations need provenance.** Label discrimination, model-fit
  scores and timing claims remain externally reported until their precise
  experiments are independently replayable. Even successful replay would
  establish only the measured association or restricted prediction.

See also [`docs/STRUCTURAL_INTERFACE_THEORY.md`](STRUCTURAL_INTERFACE_THEORY.md)
(the tetrad vs the Kuramoto order parameter on real EEG) and
[`theory/EMERGENT_ONTOLOGY.md`](../theory/EMERGENT_ONTOLOGY.md) §0 (the empirical
caveat this record nuances).
