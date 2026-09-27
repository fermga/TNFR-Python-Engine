# Reported interface observations: historical evidence boundary

**Status:** Historical reported results, not independently reproduced from a
pinned dataset/run manifest in this checkout. These summaries are retained for
provenance and negative findings; they are not current API specifications,
admitted physical validation or a research execution queue.

The numerical tables were consolidated from the former
`docs/EMPIRICAL_CONFRONTATION_EEG.md` and the historical result sections of
`docs/STRUCTURAL_INTERFACE_THEORY.md`. Moving them does not rerun, authenticate
or upgrade any result. Current behavior belongs to the
[observational interface guide](../../../docs/STRUCTURAL_INTERFACE_THEORY.md);
physical admission and priorities belong to the
[execution plan](../FIVE_STAGE_EXECUTION_PLAN.md#supporting-measurement-bridge).
The [passive transport](../PASSIVE_TRANSPORT_PROTOCOL.md) and
[TCLab](../TCLAB_EXPLORATORY_PROTOCOL.md) owners retain their separately scoped
models, protocols and evidence; the summaries below do not replace those records.

## Reported static ranking

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

## Reported grid-frequency trends

The historical power-grid report gives a classical variance trend (Kendall-τ ≈
0.255) slightly stronger than the strongest TNFR channel (Φ_s, τ ≈ 0.184). Both
reported correlations are below 0.26. Those statistics do not establish the physical mechanism
behind the signal or a general limit of either method.

Variance and autocorrelation are necessary comparators for this preparation.
Multichannel data permit additional graph-local observations; their availability
does not itself establish added predictive value.

## Reported EEG Eye State ranking

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

The maintained multichannel adapter now reports orientation-free discrimination
`max(AUC, 1-AUC)` on its supplied labels. The old table has no pinned replay
establishing its exact orientation convention, preprocessing or estimator branch;
do not reconstruct those details from current code or treat its values as a
reserved forecast.

## Reported CHB-MIT findings (external empirical arm)

The prior report attributes these findings to the separate TNFR-IA empirical
arm using the public CHB-MIT scalp-EEG corpus. No pinned external commit or
complete replay envelope is supplied here.

| Confrontation | Result | Reading |
|---|---|---|
| **Local phase tetrad** `\|∇φ\|`, `\|K_φ\|` vs seizure/interictal | cross-patient (leave-one-patient-out) **AUC ≈ 0.66–0.67**, higher at ictal | reported association with labels; independent replay remains pending |
| **Global Kuramoto `R`** vs seizure | **AUC ≈ 0.49** (chance) | reported lack of label discrimination in this preparation; does not by itself identify a local mechanism or exclude global synchrony effects |
| **`νf`** (single structural frequency) | tracks the measured dominant EEG frequency (**Spearman ρ ≈ 0.86**); fingerprints subjects | reported association does not identify structural mobility with physical oscillation frequency |
| **Single-`νf` conservative-wave forecaster** (few-shot, leave-one-patient-out) | **ties** a strong Gaussian process and a stabilised DMD/Koopman model (within ~1%); beats a parameter-matched neural net | reported accuracy with 1–3 model parameters and ~20× lower cost than the neural TNFR; physical identification and replay remain pending |
| **Face diagnosis** (`verify_overdamped_projection` at the data-fitted `γ`) | reported under-damped classification | a fitted auxiliary-model result, not certification of physical conservative dynamics |
| **Pre-ictal** (5 min before onset) tetrad | every magnitude reported at **chance** | no predictive evidence in the reported five-minute preparation; concurrent association does not establish a general detection or prediction rule |

## Reported cross-subject confrontation (PhysioNet eegmmidb)

The earlier summary describes a second confrontation **through the shipped instrument**
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

## Reported same-window diffusion comparisons

These scores concern the supplied isolated EPI-diffusion realization; the bare
nodal identity alone does not select that pressure law.

The reported in-sample diffusion-direction comparison used
[`nodal_prediction_skill`](../../../src/tnfr/validation/signal_confrontation.py):
the predictor `x̂(t+1) = x(t) − c·L_rw·x(t)` fits a single diffusion step
`c = ν_f·dt` and scores the one-step increment variance explained beyond
persistence, against a per-channel AR-1 baseline. Both models were fitted and
scored on the same supplied data. The table preserves reported scores; it does
not establish out-of-sample forecasting.

| Signal | nodal 1-step skill | AR-1 | Reading |
|---|---|---|---|
| synthetic graph diffusion | **+0.16–+0.21** (recovers `c`) | +0.03 | reported synthetic same-window recovery; not rerun here or physical validation |
| real EEG (eyes-closed) | +0.02 | **+0.05** | reported same-window AR-1 score exceeds the selected diffusion-direction fit |
| real thermal field (60–120 min steps) | **+0.015–+0.029** | +0.013–+0.025 | reported same-window scores are close; this comparison alone does not certify a diffusive physical regime |

## Limits of these reports

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
