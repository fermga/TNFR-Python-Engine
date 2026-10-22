# Admission of an observation-based collective forecast

<a id="organized-observed-forecast-admission"></a>
## Conditional admission result

The [retained observation-to-future certificate](ORGANIZED_OBSERVED_PREDICTION.md#organized-observed-prediction-result)
bounds differences between complete histories without supplying their unknown
initialization. This owner derives the contract for using one proposed carrier
history as a forecast. Sections 1-4 establish its analytical admission;
section 5 declares one operational test on archived synthetic readings.
The [research program](../RESEARCH_PROGRAM.md#research-current-question)
owns its evaluation status and subsequent dependencies.

Retain the complete fine law, source class, command, support, held capacities,
structural clock and sample schedule in the
[original declaration](ORGANIZED_OBSERVED_PREDICTION.md#organized-observed-prediction-declaration).
Local time runs from zero at `tau=200` to `T=16`. There are thirty vector
samples, each with five mean-form coordinates, and arbitrary correlated
coordinate error at most `eta=1e-14`. The initial common form remains unknown.
Neither a zero-sum prior nor the attachment preparation may be supplied to an
observation-only proposal. A common phase rotation can fix `Phi_0=0`; it
does not fix any relative phase or winding.

## 1. The retained gain applies to an arbitrary sample discrepancy

Consider two exact carriers with initial mean forms bounded by `X=1/100`,
arbitrary circular initial phases, and the same complete prescribed input.
Remove the command using the original coordinates `Y=U-z_cmd`, `Psi=Phi-H`.
At all readings and the future endpoint, `z_cmd=0`, so the observed form is Y.
Let their difference be u and suppose

\[
 \max_j\|u(t_j)\|_\infty\le\delta,\qquad\delta\ge0.
 \tag{1}
\]

The [static phasor construction and difference remainder](ORGANIZED_OBSERVED_PREDICTION.md#2-a-static-design-with-unknown-circular-phases)
do not require that delta equal the earlier fine-pair noise allowance. They
use initial-form boundedness, global sine bounds, the shared command and the
maximum two-unit distance from the latest sample. Their estimates are
homogeneous in the carrier differences. In the notation of that proof,

\[
 R_*\le A_* B_*+B_0\delta,\qquad
 B_*\le k(\delta+R_*)+e_0\delta+e_bB_*.
\]
\[
 d=1-e_b-kA_*>0,\qquad
 Q=\frac{k(1+B_0)+e_0}{d},\qquad B_*\le Q\delta.
 \tag{2}
\]

Substitution into the same future residual identity gives

\[
 G=w+n_0+n_bQ+(1+w)(A_*Q+B_0),\qquad
 \boxed{\|u(T)\|_\infty\le G\delta.}
 \tag{3}
\]

The retained interval matrices and rational weights certify this G. Its
displayed approximation `3.9317888454` is descriptive; admission uses the
rebuilt exact/outward bound. Neither an assumed inverse nor a cached verdict
can replace the residual and positive-denominator checks.

The comparison allows arbitrary initial edge phasors, so the two carriers
need not share a winding or satisfy the original acute-gap identity. The
candidate must still solve the same full-circle law. This broader comparison
does not enlarge the class from which the unknown fine truth is assumed to
come, or certify an NFR identity for a candidate outside that class.

## 2. One candidate history and its complete record residual

Let y contain the 150 admitted raw readings. Propose one exact candidate
initial state `(Uhat_0,Phihat_0)` from y and declared model information alone,
with `||Uhat_0||infinity<=X`. Retain its common form and declare its phase
gauge and circular representatives. Rounded proposal coordinates define the
candidate actually validated; they are not exact inverses of an unknown state.

Suppose a validated numerical calculation encloses the same candidate's exact
form at every reading in coordinate intervals `C_jc`, and its future form in
intervals `C_Tc` about a reported center `m_T`, with uniform radius `r_T`.
These enclosures must follow one uninterrupted initial state through the
complete word and tail. Define the upper residual

\[
 \rho=\max_{j,c}\sup_{v\in C_{jc}}|v-y_{jc}|.
 \tag{4}
\]

For symmetric sample bounds with center `m_jc` and radius `r_jc`, this is
`max_(j,c)(abs(m_jc-y_jc)+r_jc)`. It includes historical numerical error.
Intersecting a sample interval with a reading band does not prove that the
exact candidate lies inside that band. Independent per-sample states likewise
do not certify one jointly compatible history.

Assume the record was produced by some fine history in the original class.
Its exact carrier starts at that history's actual unknown collective means.
The unchanged [hidden transport bound](ORGANIZED_OBSERVED_PREDICTION.md#1-fine-source-uncertainty-is-retained-on-both-sides)
gives uniform mean-form error `E_0<2.3e-18` over the entire local horizon.
Rebuild E_0 from the original hidden bounds and laws; the stronger error bound
from known attachment preparation does not apply here. At every reading,

\[
 \|U_{truth\ carrier}(t_j)-U_{candidate}(t_j)\|_\infty
 \le\eta+E_0+\rho.
 \tag{5}
\]

Apply (3), add the truth's future source error and the candidate's numerical
endpoint radius, and obtain

\[
 \boxed{\|U_{fine}(216)-m_T\|_\infty\le
 R:=G(\eta+E_0+\rho)+E_0+r_T.}
 \tag{6}
\]

Historical numerical error is already in rho and is not added again.
Only the truth needs the fine-to-carrier correction: the candidate is the
exact carrier specified by its proposed initial state. A future reading with
allowance `eta_future` has error at most `R+eta_future` against the forecast.

The true-form policy radius `1e-13` is admitted whenever (6) proves
`R<=1e-13`. For example, `rho<=eta+E_0` is a sufficient fitting policy if
the remaining endpoint budget also satisfies this inequality. That residual
cap is not necessary: the complete inequality can admit a larger residual.
A future numerical protocol must fix its policy and budgets before evaluation.

## 3. Conditional accuracy and exact feasible witnesses

Equation (6) is conditional on fine-record membership. A carrier proposal
with a small residual does not establish that a fine-class source generated
the record, or that the carrier itself is a fine-class feasible witness.
The record might come from a different law or violate the supplied source
bounds. Good fitting does not authenticate those premises.

A stronger sufficient witness has a separate admission. Require the candidate
initial collective gaps and winding to satisfy the original circular class,
as well as the initial form bound. Set `u=z=0` and lift

\[
 x_{c,j}=\widehat U_c,\qquad
 \theta_{c,j}=\psi_j+\widehat\Phi_c.
 \tag{7}
\]

The [invariant-carrier proof](ORGANIZED_CONSTITUENT_MEMORY.md#3-the-invariant-carrier-and-its-decoupled-tangent)
shows this is an exact fine history for the same uniform-child input. If
its whole-record upper residual satisfies `rho<=eta`, it is a genuine
fine-class feasible member and proves nonemptiness. The retained fine-future
diameter then bounds its distance from every compatible fine future; add
`r_T` when reporting a numerical endpoint center. A residual allowed up to
`eta+E_0` does not establish this stronger claim.

## 4. Record, numerical and failure contracts

An operational consumer must retain and check:

- All 150 finite signed readings, their five-coordinate node order, exact
  thirty sample times, observation map, units and the declared error allowance.
- The complete law, gain, support, capacity, input and clock associations;
  source-class assumptions; and any preparation information actually supplied.
- The proposal's exact represented initial coordinates, circular phase
  conventions and common gauge, with source/algorithm/budget provenance.
- One validated candidate history, its sample enclosures and future enclosure,
  numerical precision/backend, stage partition, residuals and failure status.
- The retained gain evidence and rebuilt source/model-error budgets. Supplied
  summaries, matching labels or stale bounds cannot replace these premises.

Admission precedes float conversion. Reject nonfinite or Boolean physical
values and malformed records. Circle-valued phase admission and any chosen
continuous lifts remain distinct; an optimizer's representatives do not
authenticate a fine-state identity. Numerical integration must retain the
declared clock in both carrier rows, split actual input stages and preserve
the reached state across their boundaries.

Malformed input is rejection. A failed proposal, unsupported candidate domain,
inconclusive trajectory enclosure or exceeded numerical budget is unavailable
candidate evidence. A failed residual or radius bound rejects that candidate
for the declared policy. None establishes that every candidate fails or that
the raw record is incompatible with the model. A successful bound remains
conditional on its stated truth and numerical premises.

The [shared prediction-transfer arithmetic](../../src/tnfr/mathematics/_prediction_transfer.py)
and [independent budget controls](../../tests/mathematics/test_prediction_transfer.py)
implement the residual/error combination, including invalid domains. That
arithmetic consumes supplied enclosures and a supplied gain; it does not
certify their trajectories, provenance, record membership or model identity.
The shared prescribed-time integrator and retained evidence readers remain
the applicable owners for those obligations. No new nonlinear trajectory,
raw response or physical experiment is evaluated by this analytical result.

<a id="organized-observed-forecast-protocol"></a>
## 5. Fixed record-only software protocol

This protocol fixes one synthetic record, one deterministic proposal and one
validated candidate history before extraction or numerical evaluation. Its
purpose is to make the conditional forecast operational without supplying
generating-state coordinates. The historical archive and its future are
already public: this is a software demonstration, not a reserved response,
independent validation, law comparison or physical experiment.

### Record and prior evidence

The [read-only input owner](../../src/tnfr/research/_organized_forecast_inputs.py)
pins the following complete archive associations, including their attempt,
result, source-tree and source-inventory hashes:

| Input | Frozen producer revision | SHA-256 of `response.zip` |
| --- | --- | --- |
| [Known-preparation carrier history](../../docs/assets/organized_computable_prediction/receipt.json) | `8f5a28c58f23ab1586cc91ac36d49e6ddd9bbce0` | `c4fca529593dbfdf445d314508b78f3f188cd3fa3dce3388fa52df3a1581e4d3` |
| [Static observation certificate](../../docs/assets/organized_observed_prediction/receipt.json) | `de5b500f70b1626d75f12484af0e6381fc9d93af` | `49febee9897b4acf738d803d37128d5e9652a19e59cb082c284de780a66e3592` |

Re-admit the thirty absolute sample times and the first five coordinates of
the original `observed` branch's endpoint centers, beginning with its initial
center at `tau=200`. Use all 150 signed rational readings in time-major,
child-order `0..4`; do not round, add noise or select a subset. The last
reading is at `tau=214.00015`. The command form is zero at these times, so
the archived Y centers observe U. Rebuild the original source correction
and carried metric radii through that last reading and require their sum
to be at most `eta=1e-14` at every sample. Retain extraction errors and
endpoint associations as provenance, outside the proposal interface.

Rebuild the static observation/future matrices from all 2,145 retained
coefficient steps, then reconstruct the dual residuals, positive bootstrap
margin, G and containing-class E_0. The saved verdict or scalar gain cannot
replace this arithmetic. These readers retain the original smooth-field,
Taylor-derivative and source proof premises. Digests associate contents;
they do not authenticate execution or prove arbitrary saved derivatives.
No original coefficient or response producer is rerun.

### One untrusted numerical proposal

The [proposal owner](../../src/tnfr/physics/_organized_forecast_proposal.py)
receives only the thirty times, 150 readings and static `150 x 15` observation
matrix. The law, command and clock are declared model information. Source
phases, source initialization, extraction errors, truth labels and future
values are not inference arguments. The fixed algorithm is:

1. Solve the midpoint linear design with NumPy `linalg.lstsq`, `rcond=None`,
   requiring rank 15. Initialize oriented edge angles with `atan2` of its
   formal sine/cosine coefficients. Fix `Phi_0=0` and obtain the other four
   phase representatives by cumulative sums of the first four edge angles.
   Initialize forms from the first five readings clipped to binary64
   `[-.01,.01]`, preserving the unknown common form.
2. Fit five forms and four phase representatives with one SciPy
   `least_squares` call: method `trf`, `max_nfev=80`, `ftol=None`, `gtol=None`,
   `xtol=1e-13`, `x_scale=(.01,.01,.01,.01,.01,1,1,1,1)` and residual scale
   `1e-4`. Form bounds are binary64 `[-.01,.01]`; phase lifts are unbounded.
   Require optimizer success. Cache paired residual/Jacobian requests at
   identical parameters; there is no restart or alternative initializer.
3. Each trial carries the full ten-coordinate command-removed sine state and
   its analytic `10 x 9` sensitivities. Use DOP853 with `rtol=1e-11`,
   `atol=1e-17`, `max_step=1/8`, splitting every prescribed input stage.
   Integrate only the 15 plateaus and first 14 ramps, stopping at the last
   reading. Both rows retain the same structural clock; no state coordinate
   is reset at a stage boundary. This is a detached numerical proposal,
   not validated trajectory evidence.
4. Convert the returned finite binary64 coordinates to exact represented
   rationals. Clip the five forms again to the exact interval `[-1/100,1/100]`
   and set initial `Phi_0=0` exactly. This represented state is the sole
   candidate. Retain the numerical start, optimizer status, evaluation and
   forward-call counts, diagnostic residual and numerical backend metadata.
   Backend metadata includes NumPy/SciPy versions, compiler/build architecture,
   BLAS/LAPACK configuration and integer widths, and available SIMD features;
   omit machine paths, commands and environment values.

### One validated forecast and decision

The [candidate consumer](../../src/tnfr/physics/_organized_observed_forecast.py)
re-admits the record and exact candidate before integration. Run one carried
candidate from `tau=200` to `216` through the complete word and released tail,
using the shared prescribed-time metric Taylor kernel, order 16, outward
128-bit dyadic intervals, identity metric and growth bound `g=1/3000`.
Split at every stage and use maximum step `1/8`: 143 successful steps cover
the fixed horizon. The initial candidate radius is zero and the propagated
numerical radius may not exceed `1e-15`. Keep every compact tube, polynomial,
remainder, growth and endpoint witness. No refinement or retry follows failure.

Construct all sample enclosures from this single fresh history and calculate
the upper residual (4), then (6) with the rebuilt G, containing-class E_0
and `eta=1e-14`. Admit only if the trajectory is complete and the true-form
radius is at most `1e-13`. Report the five endpoint centers with their true-form
bounds and separate future-reading bounds enlarged by `eta`. Do not inspect
the archived future as an acceptance criterion. The stronger exact feasible
witness in section 3 is not part of this protocol.

Failure of proposal, enclosure or policy yields explicit unavailability
without a new candidate, budget, tolerance or source choice. Malformed inputs
reject before use; unexpected software errors retain an incomplete attempt.
A successful case establishes one operational conditional forecast, not
optimizer convergence for every admissible record, source-class membership
for arbitrary data, or a uniquely justified evolution law.

Run [the single fixed instrument](../../benchmarks/organized_observed_forecast.py)
only after committing this declaration and all consumed source. Its shared
exclusive assessment lifecycle records the source inventory and attempt
before extraction; exact record, proposal and validated response follow in
the result. Archive those bytes and receipt for read-only verification.

<!-- FROZEN DECLARATION AND PROOF END: append evaluated results only below. -->

<a id="organized-observed-forecast-result"></a>
## Retained operational software result

The single evaluation of declaration/source
`53917ac7d05c4fd9c79c8ce37ed1c95c25587f50` returned
`conditional_record_only_forecast_certified`. Its
[attempt](../../docs/assets/organized_observed_forecast/attempt.json),
[exact response archive](../../docs/assets/organized_observed_forecast/response.zip)
and [receipt](../../docs/assets/organized_observed_forecast/receipt.json)
retain the extracted record, input associations, represented candidate,
proposal backend and all 143 validated steps. No old producer was replayed.

| Quantity | Evaluated value, displayed approximately |
| --- | ---: |
| Historical readings / optimizer forward calls | `150 / 3` |
| Maximum archived extraction allowance | `1.489381e-21`, below `eta=1e-14` |
| Whole-record validated residual rho | `2.827128e-19` |
| Rebuilt carrier gain G | `3.9317888454` |
| Containing-class source bound E_0 | `2.164623e-18` |
| Candidate endpoint numerical radius r_T | `9.775683e-24` |
| Certified true-form radius R | `3.932968e-14`, below `1e-13` |
| Certified future-reading radius R+eta | `4.932968e-14` |

The five forecast centers at `tau=216`, in child order and structural form
units, are approximately
`(3.256469903e-6, 1.645762162e-6, -1.253943082e-7, -1.973615505e-6, -2.803222251e-6)`.
Use the exact archived centers and outward bounds for calculations. The
binary64 optimizer's residual is only a diagnostic; the policy decision uses
the separate validated history's upper residual and exact transfer arithmetic.

This establishes a computable conditional forecast for the selected record
without privileged generating-state coordinates or future response. The
first five raw readings initialize the form proposal; generating phases
are not supplied. All retained historical enclosures belong to one candidate. It does
not establish global optimizer convergence, exact fine-feasible membership,
or an independently observed prediction: the record is extracted from an
already published synthetic trajectory. The assumed law, source class,
input and ideal clock still carry the scientific interpretation.

The [read-only evidence audit](../../tests/physics/test_organized_observed_forecast_evidence.py)
rebuilds source/gain inputs, candidate/sample associations, compact radius
propagation and the forecast bounds without fitting or integrating another
trajectory. Retained Picard/Taylor witnesses keep their source proof premises.
This forecast does not resolve whether unknown initialization can mask a
changed constitutive law. That separate question needs joint law/state
admission; tighter forecast arithmetic alone cannot supply it. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone selects further investigation.
