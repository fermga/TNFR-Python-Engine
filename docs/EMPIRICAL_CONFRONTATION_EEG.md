# Empirical Confrontation — Real EEG (the falsifiable arm)

**Status:** validation record. The value here is a **falsifiable instrument**, not
an out-prediction claim. Closes no open problem.

## Purpose

TNFR's structural read-outs (the tetrad `|∇φ|`, `K_φ`, `Φ_s`, `ξ_C`;
the pulse `ω_k = √λ_k`; static pressure coherence `C_static`; structural
frequency `νf`) are
meant to be **confronted with real data** — tests that *can* fail. The engine
ships the **data-agnostic instrument**; it does **not** bundle datasets.

- **Instrument (in-engine):** [`confront_signal`](../src/tnfr/validation/signal_confrontation.py)
  maps any real multichannel signal → the **emergent phase-locking graph**
  ([`build_coupling_graph`](../src/tnfr/validation/multichannel_interface.py),
  U3/PLV) → the scoped read-outs (tetrad, pulse, `ξ_C` via the emergent
  spectral gap, static pressure coherence `C_static`) → the **emergent two-face
  diagnosis**
  ([`emergent_wave_fraction`](../src/tnfr/validation/signal_confrontation.py):
  project the signal onto the emergent `L_sym` eigenmodes and read the damping
  from each modal coordinate's own AR(2) dynamics — oscillate ⇒ conservative-
  wave face, relax ⇒ diffusive face — so the verdict is set by the emergent
  dynamics, not by a fixed γ or the raw input spectrum).
  The signal window has no TNFR EPI-rate channel. Consequently the public
  ``coherence`` result evaluates ``structural_coherence(mean|DeltaNFR|, 0)``.
  It is a static snapshot read-out, not canonical dynamic ``C(t)`` and not an
  attractor or convergence certificate.
  Demo: [example 159](../examples/10_applications/159_empirical_confrontation_pipeline.py)
  (`python examples/10_applications/159_empirical_confrontation_pipeline.py path/to/signal.npy`).
- **Findings below** come from the **TNFR-IA empirical arm** (a separate,
  downstream repository that consumes this engine) run on the public CHB-MIT
  scalp-EEG corpus. They are recorded here as the engine's validation evidence;
  the raw data and experiment scripts live in TNFR-IA, not here.

## Findings (real CHB-MIT EEG; the empirical arm)

| Confrontation | Result | Reading |
|---|---|---|
| **Local phase tetrad** `|∇φ|`, `|K_φ|` vs seizure/interictal | cross-patient (leave-one-patient-out) **AUC ≈ 0.66–0.67**, higher at ictal | a canonical magnitude carries **labelled clinical state**, and it is the *local* tetrad the theory centres |
| **Global Kuramoto `R`** vs seizure | **AUC ≈ 0.49** (chance) | global hypersynchrony does **not** rise under the bipolar montage / focal onset — the state is *local*, not mean-field |
| **`νf`** (single structural frequency) | tracks the measured dominant EEG frequency (**Spearman ρ ≈ 0.86**); fingerprints subjects | `νf` is canonical *and* provably meaningful (a robust rhythm meter) |
| **Single-`νf` conservative-wave forecaster** (few-shot, leave-one-patient-out) | **ties** a strong Gaussian process and a stabilised DMD/Koopman model (within ~1%); beats a parameter-matched neural net | strong-baseline accuracy from **1–3 physical constants** on emergent geometry, ~20× cheaper than the neural TNFR |
| **Face diagnosis** (`verify_overdamped_projection` at the data-fitted `γ`) | real EEG sits on the **under-damped (conservative / oscillatory) face** | the engine's own certificate locates real data on the correct face |
| **Pre-ictal** (5 min before onset) tetrad | every magnitude at **chance** | the tetrad is a **detection** marker (concurrent), **not** a **prediction** marker |

## Cross-subject confrontation (PhysioNet eegmmidb; reproducible in-engine)

A second, independent confrontation run **through the shipped instrument**
(`confront_signal`, data-fitted face) on the PhysioNet **eegmmidb** eyes-open
(R01) vs eyes-closed (R02) baselines — 10 subjects, 64-channel, drift removed
(`fft_bandpass` 1–40 Hz), paired across subjects with a two-sided binomial sign
test.  Data are **not** bundled (fetched to a scratch dir); the run is
reproducible from the engine alone.

| Confrontation (open→closed) | Result | Reading |
|---|---|---|
| **Emergent face** (modal AR(2) on `L_sym`) | **WAVE / under-damped in 18/20** conditions | the emergent two-face verdict: real EEG is on the conservative (wave) face; a real thermal field (heat diffusion) lands on the **diffusive** face (97–99% of 2-day windows), so the certificate **discriminates** two real systems |
| **`K_φ`** (phase curvature, local) | rises in **10/10** subjects (paired dz ≈ +1.1, sign-test **p ≈ 0.002**) | the phase-curvature field — a canonical TNFR magnitude with no standard analogue — is the **most consistent cross-subject state marker** |
| **`|∇φ|`** (phase gradient, local) | rises in **9/10** (**p ≈ 0.021**) | the local phase field carries the state consistently across subjects |
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

Beyond the static read-outs, the **evolution law** itself is confrontable
in-engine ([`nodal_prediction_skill`](../src/tnfr/validation/signal_confrontation.py)):
the EPI channel of `∂EPI/∂t = ν_f·ΔNFR` is graph diffusion, so its one-step
predictor `x̂(t+1) = x(t) − c·L_rw·x(t)` (a single fitted diffusion step
`c = ν_f·dt`) is scored by the one-step increment variance it explains beyond
persistence, against a per-channel AR-1 baseline.

| Signal | nodal 1-step skill | AR-1 | Reading |
|---|---|---|---|
| synthetic graph diffusion | **+0.16–+0.21** (recovers `c`) | +0.03 | validated: the predictor recovers a genuine diffusion law |
| real EEG (eyes-closed) | +0.02 | **+0.05** | oscillatory — the EPI *diffusion* channel is weak; AR-1 wins |
| real thermal field (60–120 min steps) | **+0.015–+0.029** | +0.013–+0.025 | diffusive — the nodal diffusion **matches/edges** AR-1 at diffusive timescales |

**Diffusion-selective, internally consistent with the two faces.** The nodal EPI
channel is a *diffusion* law, so its one-step skill tracks how diffusive the
system is: positive and AR-1-matching for the thermal (diffusive-face) field,
weak for oscillatory EEG (wave-face, where AR-1 wins).  Absolute skills are
small (~0.02–0.03 on real data) — a **structural-fidelity** result, not superior
forecasting.  This anchors the paradigm's core *evolution law* on real data, not
just its static observables.

## Honest scope

- **Ties, does not beat.** On real EEG the canonical conservative-wave model
  reaches the same accuracy band as strong classical baselines (GP, stabilised
  DMD); it does **not** out-predict them. The distinctive value is *structural*:
  parsimony (few physical constants), a canonical/emergent construction,
  interpretability (`νf`, the local tetrad), stability by construction.
- **Detection, not prediction.** The tetrad tracks the seizure while it happens
  (AUC ≈ 0.67), not its approach (pre-ictal at chance).
- **One substrate, modest margins.** Single corpus; relative-MSE margins over the
  baselines are ~1%; AUC ≈ 0.67 is not a competitive seizure detector.
- **Cross-subject, still one paradigm.** The eegmmidb generalisation is n = 10,
  a single eyes-open/closed paradigm; effect sizes are medium–large (`K_φ`
  p ≈ 0.002) but `C` is only a trend. It recovers a known phenomenon (spatial
  alpha) rather than out-predicting a baseline.
- **The emergent face needs adequate windows.** The two-face verdict reads the
  damping from the emergent modal dynamics (AR(2) per `L_sym` mode), not from
  the input spectrum. It certifies **both** faces on real data — EEG on the
  *wave* face (18/20) and a real thermal field on the *diffusive* face (97–99%),
  completing the discrimination the input-spectrum Q could not — but the per-mode
  fit is under-powered for very short windows (EEG converges to *wave* only for
  windows of roughly a thousand samples or more).
- **The falsifiable positive:** the *canonical, emergent* representation carries
  labelled state and lands exactly where the theory's *local* tetrad fields live
  — the first labelled evidence that a canonical TNFR magnitude carries real
  clinical state. Closes no open problem.

See also [`docs/STRUCTURAL_INTERFACE_THEORY.md`](STRUCTURAL_INTERFACE_THEORY.md)
(the tetrad vs the Kuramoto order parameter on real EEG) and
[`theory/EMERGENT_ONTOLOGY.md`](../theory/EMERGENT_ONTOLOGY.md) §0 (the empirical
caveat this record nuances).
