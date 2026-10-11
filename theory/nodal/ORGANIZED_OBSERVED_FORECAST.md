# Admission of an observation-based collective forecast

<a id="organized-observed-forecast-admission"></a>
## Conditional admission result

The [retained observation-to-future certificate](ORGANIZED_OBSERVED_PREDICTION.md#organized-observed-prediction-result)
bounds differences between complete histories without supplying their unknown
initialization. This owner derives the contract for using one proposed carrier
history as a forecast. The analytical admission is established below; it does
not supply a record proposal, a validated candidate trajectory or a new
evaluated prediction. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the status of that operational work.

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
