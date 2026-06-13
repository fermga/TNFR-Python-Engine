# TNFR Structural Interface Theory

## Status

Active.  This document describes a completed, reproducible TNFR programme for
**structural-interface analysis** on real graph and time-series data.  It
consolidates the work planned in
[STRUCTURAL_INTERFACE_THEORY_PLAN.md](STRUCTURAL_INTERFACE_THEORY_PLAN.md) and
reports the validated results, including the cases where classical baselines win.

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

- edge phase-gate compliance (U3 resonant-coupling condition `|φᵢ − φⱼ| ≤ Δφ_max`);
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
python examples/93_structural_interface_demo.py
```

[examples/93_structural_interface_demo.py](../examples/93_structural_interface_demo.py)
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
- `window_tetrad_series(signal, *, config=None)`;
- `rolling_variance`, `rolling_lag1_autocorrelation`, `kendall_tau`;
- `evaluate_early_warning(signal, *, transition_index=None, config=None)`.

### Multi-channel — `tnfr.validation.multichannel_interface`

- `MultichannelConfig`, `MultichannelWindowSeries`, `SynchronyDiscrimination`;
- `fft_bandpass`, `analytic_phase_amplitude`, `phase_amplitude_matrices`;
- `kuramoto_order_parameter`, `phase_locking_matrix`, `phase_offsets`,
  `amplitude_pressure`, `build_coupling_graph`;
- `multichannel_window_series(signals, *, config=None)`;
- `evaluate_synchrony_discrimination(signals, labels, *, config=None)`.

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

- Planning roadmap: [STRUCTURAL_INTERFACE_THEORY_PLAN.md](STRUCTURAL_INTERFACE_THEORY_PLAN.md)
- Field definitions: [STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)
- Grammar derivations: [grammar/PHYSICS_VERIFICATION.md](grammar/PHYSICS_VERIFICATION.md)
- Primary theory: [AGENTS.md](../AGENTS.md)
- Scheffer et al. (2009), *Early-warning signals for critical transitions*, Nature.
- Dakos et al. (2012), *Methods for detecting early warnings of critical
  transitions in time series*, PLoS ONE.
