# TCLab: first frozen nodal response comparison

**Status: 2026-09-21.** One bounded exploratory comparison is complete.
`physical_status=not_admitted`; `measurement_verdict=not_assessed`.
The user authorized an initial approximation despite unresolved physical
admission. This note owns that experiment; the
[execution plan](FIVE_STAGE_EXECUTION_PLAN.md#supporting-measurement-bridge)
owns next actions and the [source review](PASSIVE_TRANSPORT_PROTOCOL.md#terrestrial-source-review-2026-09-21)
owns alternative datasets. No production dynamics or defaults changed.
This is a retained experiment, not an active acquisition or refitting plan.
Any follow-up requires admission through that sole queue; the dated results
and frozen specification below remain unchanged.

## 1. Question, source and physical boundary

Does a declared driven nodal model with latent heater states predict two
recorded sensor temperatures better than a smaller measured-state model when
both are calibrated on the same files and frozen before reserved comparison?
This is a conditional model comparison, not a derivation of thermodynamics or
an experiment on autonomous NFR formation. Both model families also admit a
classical linear thermal-network interpretation; their comparison cannot
establish a uniquely TNFR explanation.

The [authors' TCLab description](https://dowlinglab.github.io/pyomo-doe/notebooks/tclab-model/)
distinguishes heater and sensor temperatures. The
[data directory](https://github.com/dowlinglab/pyomo-doe/tree/d250c5b0625afb35007c075df1e0f7125e74017d/data)
is pinned at `d250c5b0625afb35007c075df1e0f7125e74017d`; repository license is
BSD-3-Clause. No author-fitted coefficients were imported. Git blob identities
bind all five downloaded files (132,055 bytes total).

The [historical acquisition notebook](https://github.com/dowlinglab/pyomo-doe/blob/b114b13f1416c44037daf933e1495110a8168633/notebooks/tclab_experiment_validation.ipynb)
uses hardware `TCLab()`, nominal one-second scheduling and 900-second runs.
Its active loop includes 25-minute cooldowns and sets heater-1 maximum-power
configuration to 200, not 200 watts. The saved notebook does not reproduce
every file: step/five-minute generation is incomplete, heater-2 settings are
not assigned, and `env_2` is unexplained. This supports intended hardware
acquisition, not a complete per-run device/reset/calibration record.

Independent ambient temperature, command-to-watt conversion, sensor/clock
uncertainty and physical repeat independence remain unverified. The model's
nominal time and temperature chart do not resolve those omissions.

## 2. Model frozen before response inspection

EPI is provisionally represented in degrees Celsius; the declared structural
clock is the recorded clock in seconds. Set `a`=leak rate, `k`=inter-heater
coupling rate, `c`=heater-to-sensor backreaction coefficient, `r`=sensor response
rate and `g`=command gain. All are positive effective coefficients, with `g`
in degrees Celsius per second per command percentage point. They are not
independently measured heat capacities or conductances.

For `i=1,2` and the other assembly `j`, the four-state candidate is

```text
H_i' = a*(Ta-H_i) + k*(H_j-H_i) + c*(S_i-H_i) + g*Q_i
S_i' = r*(H_i-S_i).
```

This follows from the selected graph realization of the extended nodal row,
not from the uncompleted nodal identity alone. On support `(H1,H2,S1,S2,B)`,
use weights `w(Hi,Si)=1`, `w(H1,H2)=k/c`, `w(Hi,B)=a/c`; the unit edge fixes
an arbitrary common conductance scale. Set `nu_H=a+k+c`, `nu_S=r`, `nu_B=0`,
`B=Ta`. Then the shared isolated-EPI pressure `p=-L_rw*x` and explicit source
`Gamma_Hi=g*Qi` yield the displayed equations. All other pressure channels
are inactive; there are no operator events. The fixed bath and supplied heater
input are premises, not emergent sources.

The smaller, three-parameter comparison is

```text
S_i' = a*(Ta-S_i) + k*(S_j-S_i) + g*Q_i.
```

It is another grounded nodal graph, not a claim that the real sensor pair is
a complete state. Both candidates assume exchange symmetry, linear commands,
constant coefficients and a constant bath. Before each prediction set `Ta`
to the mean of the first sensor pair and each hidden heater to its corresponding
first sensor value. These are initialization approximations, not measured
ambient/heater states. The entire recorded command schedule is supplied;
the forecast is conditional on those inputs, with no later temperature updates.

Within the four-state model, equal current sensor readings can have different
future derivatives when hidden heater temperatures differ. Eliminating the
heaters therefore introduces memory/hidden-initial-state terms, consistent
with the existing [memory derivation](../DERIVED_EPI_MEMORY.md). This identity
does not establish that four states are the minimal physical TCLab state.

### Variable contract for this approximation

This 2026-09-23 clarification applies the plan's
[definition/identification/evolution contract](FIVE_STAGE_EXECUTION_PLAN.md#variable-definition-identification-and-evolution)
to the already frozen models. It changes neither their equations nor the
reported forecasts, calibration or physical-admission status.
The table details the four-state candidate. The two-state baseline retains
only sensors and bath, with its separately specified rates and no heater states.

| Quantity | Mathematical definition in this model | Operational identification | Prospective law / remaining premise |
| --- | --- | --- | --- |
| EPI | Real scalar temperature chart, nominal degrees Celsius, on the declared nodes | CSV `T1,T2` observe sensors; heater states are hidden; the bath is an initial-sensor proxy. Independent sensor calibration, ambient and hidden initialization remain open. | The equations above evolve heaters/sensors; bath EPI is held. Initialize once, without subsequent sensor correction. |
| `nu_f` | Nonnegative rate per recorded second; heater/sensor values derive from the fitted coefficients as specified above; bath rate is zero | Calibration fits effective rates conditional on the pressure normalization and clock. It does not independently measure a physical reorganization capacity or heat capacity. | Rates are held throughout each run and shared across acquisitions; that constancy is a model premise. |
| `DeltaNFR` | Isolated EPI transport `-L_rw*x`, in the chosen temperature unit | Computed from the supplied graph and modeled state; hidden heater temperatures prevent direct reconstruction from the two sensors alone | Evaluate the frozen graph law on predicted states. No evaluated temperature derivative determines pressure. |
| Clock | Increasing recorded timestamps, nominal seconds | CSV `Time`; calibration, synchronization and clock uncertainty remain unverified | Left-held commands on actual recorded intervals; no fitted structural-to-physical clock correction. |
| Support and weights | Fixed reciprocal graph; common weight scale is a normalization convention | Chosen exchange/bath topology; absolute conductances are not measured or identified by the normalized transport fit | Graph and weights remain fixed. The node/edge realization is a constitutive approximation. |
| Input `Gamma` | Additive heater rate `g*Q_i`, in degrees Celsius per recorded second | Recorded percent commands and calibration-fitted gain; no independent command-to-watt conversion | Entire command schedule is conditioned on; gain is frozen. No output-dependent source correction. |
| Phase and history | No phase channel or additional independent history coordinate is consumed | No physical phase measurement is claimed; hidden heaters carry the model's internal state | Full-state evolution is Markovian; eliminating heaters produces the conditional memory described above. |

Definitions and executable closure are specified for this approximation.
Limited predictive adequacy is assessed in section 4; physical identification
remains open. Resolving the table's physical gaps does not follow automatically
from lowering forecast error, and the existing scored files remain inspected.

## 3. Split, fitting and numerical realization

Only these two named files calibrate each model:

- `validation_experiment_env_2_step_50_run_2.csv`;
- `validation_experiment_env_2_sin_5_50_run_2.csv`.

Reserve whole files `step_50_run_3`, `sin_2_50_run_3`, `sin_50_50_run_3` under
the same prefix. The acquisition code maps `sin_50` to a 0.5-minute input,
not a 50-minute period. Forecasting uses recorded commands, not inferred
filename periods. Source aliases identified by the inventory are excluded.
Different file identities do not prove independent physical preparations.

Fit all post-initial calibration samples of both sensors with equal residual
weight. Use five parameters for the latent model, three for the smaller model,
each bounded to `[1e-6,1]` in its declared units. These bounds are numerical
search settings, not physical constants. Three initializations and a maximum
of 200 evaluations per initialization were frozen before loading responses.
All six fits converged without active bounds; select the minimum calibration
cost separately for each model. No evaluation result selected coefficients,
bounds, initial conditions, horizon, smoothing or a new model. Persistence at
each initial sensor temperature is also scored as a trivial control.

[The benchmark](../../benchmarks/tclab_nodal_exploration.py) constructs its
generator through `structural_diffusion_operator` and propagates the affine
system using the shared `matrix_exponential` with left-held recorded commands
on each actual timestamp interval. This is a detached, approximate matrix
forecast, not a live engine trajectory or an exact-real interval certificate.
All forecast files were saved and content-hashed before converting reserved
post-initial temperatures into scored observations. Whole CSV bytes were
available locally; this is not externally blinded or independently timestamped
prospective acquisition. No missing rows were interpolated.

## 4. Observed comparison

RMSE below combines both sensors and all post-initial samples in degrees
Celsius; per-channel results prevent the larger response hiding the other.

| Reserved file suffix | Scored times | Four-state RMSE | Two-state RMSE | Persistence RMSE | Four-state RMSE by sensor `(S1,S2)` |
| --- | ---: | ---: | ---: | ---: | --- |
| `step_50_run_3` | 900 | 1.285 | 1.545 | 22.356 | `(1.792, 0.303)` |
| `sin_2_50_run_3` | 900 | 2.434 | 3.061 | 21.342 | `(3.286, 1.023)` |
| `sin_50_50_run_3` | 898 | 2.446 | 2.664 | 20.980 | `(3.314, 0.994)` |

The latent model reduces joint RMSE by approximately 16.8%, 20.5% and 8.2%
relative to the two-state baseline. Its calibration RMSE is 0.642 C versus
1.358 C. These are descriptive comparisons, not confidence intervals or an
independence-adjusted significance test. The larger model has two additional
fitted parameters; this experiment does not prove a universal need for memory.

An important remaining error is a positive sensor-1 bias of approximately
1.466, 2.972 and 3.025 C respectively. Sensor-2 MAE is slightly worse than the
two-state baseline on both oscillatory tests, despite lower RMSE. Retain those
mixed results: improvement is not uniform across metrics. Possible effects
of ambient/initialization, symmetry, input conversion and omitted physics
are unresolved explanations, not measured diagnoses or a license to refit.

The selected latent coefficients `(a,k,c,r,g)` are approximately
`(0.00695545,0.00275190,0.01143465,0.04570218,0.00679360)`.
The smaller model's `(a,k,g)` are
`(0.00443295,0.00182855,0.00447633)`.
The latent-state observability rank is 4 within its supplied model. This does
not identify separately every physical primitive or establish noise-robust
reconstruction. Heater 2 remained unexcited in the reserved inputs; its gain
equality remains an untested symmetry premise.

An independent SciPy matrix-exponential propagation differs from the retained
predictions by at most `7.4e-13 C` across both models/all reserved runs. Thus
this solver comparison does not explain the degree-scale residuals; it is
not a rigorous error bound or sensor calibration. The
[synthetic tests](../../tests/test_tclab_nodal_exploration.py) cover analytic
and independent ODE solutions, input boundaries, duplicate splits, exclusion
of future responses and an actual weighted-pressure/Gamma/shared-integrator
step. All 26 checks passed.

## 5. Evidence and next decision

The local retained bundle is `artifacts/research/tclab_exploration_2026_09_21/`:
original source files, frozen specification, executed benchmark snapshot,
calibration, issued forecasts, scores and numerical check. The inspected
six-panel plot is `reserved_comparison.png` (also SVG). These are ignored local
artifacts, not a published independently audited evidence bundle.

- Dataset revision: `d250c5b0625afb35007c075df1e0f7125e74017d`.
- Engine base: `f27413a0a29dddc132e5aafea081b0caee0b0135`; dirty source hash
  `sha256:da7c03dbb402ae560b5fdd0bd8d039836b4b620fde83e79befdcae2e710e2be6`.
- Python 3.13.6, NumPy 2.5.3, SciPy 1.18.1; deterministic fixed starts, no RNG.
- Raw specification SHA-256:
  `b8582e565b5523a864a3fa69f2e029c5a0fd8abe500c61990d9b1a367b458b55`.
  The pre-data console hash `c271ba2f...` identified logical LF text; Windows
  saved CRLF. Both retained raw specification copies agree; no policy changed.
- Forecast hash:
  `sha256:c0d8219f925ada27b2b79f316ef0e9961afa19a57fb270297aaab15579536299`.

The implementation corrected a capacity attribute and CSV admission before
calibration; later graph metadata/formatting changes do not alter the matrices
or retained forecasts. Replaying into a fresh directory is permitted but does
not create new reserved evidence:

```powershell
python benchmarks/tclab_nodal_exploration.py --spec artifacts/research/tclab_exploration_2026_09_21/spec_before_responses.json --data artifacts/research/tclab_exploration_2026_09_21/data --output artifacts/research/tclab_exploration_2026_09_21/replay
```

The useful conclusion is that a specified nodal transport/input model transfers
some response structure across these files, and the latent version improves
the predeclared quadratic metric. Physical admission and the stronger TNFR
ontology remain open. If the sole plan admits another comparison, establish
independently which initialization/ambient/input premises can be supported
and declare any model revision explicitly. Preserve the present residuals; these three runs
are now inspected and cannot serve as fresh reserved tests for that revision.
