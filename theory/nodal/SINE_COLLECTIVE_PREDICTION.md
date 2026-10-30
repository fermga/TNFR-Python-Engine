# Collective prediction and temporal error allowances

<a id="collective-prediction-scope"></a>

This chapter owns continuing work on source-associated collective prediction
for the normalized-sine replica model. The existing
[replica chapter](SINE_REPLICA_PULSE.md#replica-internal-comparison) retains
the complete source, internal comparison theorem and all earlier proofs.
Its [resolution result](SINE_REPLICA_PULSE.md#replica-prediction-resolution-result)
supplies the geometric thresholds consumed here. Published routes and
frozen evidence remain with those owners; this chapter does not redefine
the model or reproduce their derivations.

The [research program](../RESEARCH_PROGRAM.md#research-current-question)
owns the sole active task. The first question here is whether the same
positive majorant, evaluated at five declared cell endpoints, sharpens the
retained prediction enough for a temporal decision. No new endpoint
allowance has been evaluated during this editorial preparation.

<a id="replica-endpoint-allowance"></a>

## 1. Whole-cell decisions from five endpoint allowances

### 1.1. The unchanged positive majorant

Retain the memberwise synchronized comparison and its full-source error
theorem from [Section 17](SINE_REPLICA_PULSE.md#replica-internal-comparison).
The source's internal initialization, complete sine law, fixed support,
held capacities, continuous lifts and original structural clock are unchanged.
The actual-source chart and activity premises remain with that theorem;
a comparison tube alone does not establish them.

Let `M` be the same rational nonnegative comparison matrix rebuilt from
the retained source, and `z_0=(0,0,d^2,d*v,v^2)` its same initial envelope.
These are the previously admitted upper coefficients and initial data,
not fitted coefficients or a newly selected complete physical law. Number
its coordinates by `(P,Q,D^2,DU,U^2)` and define the exact comparison flow

\[
z(t)=\exp(tM)z_0,\qquad E(t)=\sqrt2\,z_Q(t),\qquad t\ge0.
\]

Componentwise coefficient and initial-data upper bounds preserve this
comparison: nonnegative matrix products give `M_0^k<=M^k` when
`0<=M_0<=M`, and the exponential series preserves that ordering. Thus the
existing theorem bounds every actual-minus-comparison collective gap by
`E(t)` through time six. Here `E(t)` is an error majorant, not an
observed error, a comparison trajectory or the actual internal motion.
Because `M>=0` and `z_0>=0` componentwise, every term in the exponential
series is nonnegative. Hence `z(t)>=0` and

\[
z'(t)=Mz(t)\ge0,\qquad
z(b)-z(t)=\int_t^b Mz(s)\,ds\ge0\quad(0\le t\le b).
\]

Thus `E(t)<=E(b)`. This monotonicity belongs to the exact positive
majorant. It does not require the physical dynamics to contract, its sine
secants to be positive, or independently computed numerical upper bounds
to increase with their evaluation times.

### 1.2. Retained arithmetic and a valid cap minimum

The shared [positive exponential owner](../../src/tnfr/mathematics/_comparison_flow.py)
returns `K_b>=exp(b*M)` entrywise. Keep its unchanged policy
`exact_nonnegative_series32_resolvent_tail_v1`: exact rational products,
32 polynomial terms beyond the constant term, and the positive resolvent
tail, admitted only when `b*||M||_infinity<=32`. The exponential calculation
uses no floating matrix exponential or dyadic display rounding. The matrix
coefficients and the factor `r_2>=sqrt(2)` retain their earlier outward
source enclosures. With the same `z_0` at structural time zero, set

\[
B_b=r_2(K_bz_0)_Q\ge E(b),\qquad
\widehat E_b=\min\{B_b,E_6\},
\]

where `E_6` is the previously certified horizon allowance. This is not a
restart at the cell's left endpoint or a product of locally reset flows.
For every closed cell `[a,b]` with `0<=a<=b<=6`, the two independent bounds
on the same gap error give

\[
\max_i|g_i(t)-\widehat g_i(t)|
\le E(t)\le B_b,\qquad
\max_i|g_i(t)-\widehat g_i(t)|\le E_6
\quad(t\in[a,b]).
\]

Therefore that error is at most `widehat E_b` throughout the cell,
including both endpoints. The minimum is nonnegative and at most `E_6`,
so it stays inside the already proved geometric-observer domain. This
argument assumes no ordering between separately computed `B_b` values.
Re-admitting the old time-two and time-six kernel records validates prior
arithmetic; it does not evaluate any of the five new endpoint allowances.

### 1.3. Fixed consumers and temporal coverage

Only the following cells receive a newly evaluated allowance:

| Zero-based cell | Right endpoint `b` |
| --- | --- |
| 36 | `37/8` |
| 37 | `19/4` |
| 38 | `39/8` |
| 39 | `5` |
| 40 | `41/8` |

Keep the original allowance on each of the other forty-three cells, their
whole-time tubes and their certified separation. For each selected cell,
consume its exact admitted positive threshold `rho_i` from the
[resolution theorem and result](SINE_REPLICA_PULSE.md#replica-prediction-resolution).
The same closed dilation reader separates the cell exactly when
`widehat E_b<rho_i`. Equality remains unresolved. A displayed decimal is
not substituted for the exact rational comparison. Unavailable evidence
cannot become a separation or a zero allowance.

Reconcile each new label with the existing integer-assignment observer and
rebuild all forty-eight labels through the shared closed-cell time cover.
All cells separated certify full-horizon target absence. A remaining
possible-time component may still be too short, with the appropriate
endpoint inclusion, to contain a closed half-unit residence window.
Residence exclusion and full-horizon absence are therefore separate
outcomes. Possible times establish neither an exact compatible state nor
a visit, and the old full-state absence is not used to force a label.

### 1.4. Prospective implementation and interpretation

The [endpoint owner](../../src/tnfr/research/replica_endpoint_allowance.py)
uses `evaluate_endpoint`, `classify_allowance` and `merge_allowances` for
the arithmetic and fixed schedule above. One frozen finite assessment
admits the anchored source, previous allowance and resolution records,
reconstructs consumed prior arithmetic, and evaluates the five declared
absolute times once. It changes neither the matrix, source weights, norm,
initialization, kernel policy, target nor time horizon. There is no new
trajectory, field evaluation, prior scientific-producer replay, alternate
time, optimized majorant or adaptive retry. Each command has a 300-second
limit, and failures remain retained.

Eight constructed control groups cover an independent positive-envelope
oracle, nonnegative and scalar premises, cap minima, exact threshold
contact, the five-cell schedule with forty-three retained entries, temporal
coverage, bound-evidence corruption and primitive/record admission. No
control evaluates these new times on the actual source. The eight controls
and the single five-time assessment are pending. After evaluation, a
detached audit checks retained polynomial/resolvent identities, envelope
products, minima, threshold decisions and coverage without another kernel
producer call. Associate copied prior evidence directly with its anchored
originals. Records belong to `artifacts/replica-endpoint-allowance-20261027/`.

A recovered decision would show that the existing theorem's temporal
resolution suffices for this source and reader. An unresolved decision
would retain a quantitative limitation of these sufficient bounds, not
measure the actual discrepancy or prove a physical internal effect. This
prior-informed test establishes no autonomous closure, new formation,
constitutive uniqueness or empirical identification. Only the research
program selects a subsequent investigation after this assessment closes.

<a id="replica-endpoint-allowance-result"></a>

## 2. Retained result: residence exclusion with one unresolved cell

All eight constructed controls passed without skips. One frozen assessment
evaluated the five declared endpoints, using the unchanged matrix, initial
envelope, kernel and source association. The detached audit checked each
stored polynomial and resolvent-tail identity, envelope product, cap minimum,
threshold comparison and temporal cover. It called no exponential producer
or inverse and performed no integration. Evidence is retained in
`artifacts/replica-endpoint-allowance-20261027/`.

The result is **`half_unit_residence_excluded_by_endpoint_allowances`**.
The table displays gap allowances rounded **upward** and critical thresholds
rounded **downward**. All decisions use their exact rational values in
`endpoint-result.json` and `result-audit.json`, not these decimal displays.

| Cell | Right endpoint | Gap upper allowance | Threshold lower display | Decision |
| --- | --- | --- | --- | --- |
| 36 | `37/8` | `0.051668878` | `0.149522927` | Separated |
| 37 | `19/4` | `0.058529601` | `0.053970151` | Unresolved |
| 38 | `39/8` | `0.066187396` | `0.095873750` | Separated |
| 39 | `5` | `0.074725781` | `0.137330028` | Separated |
| 40 | `41/8` | `0.084236471` | `0.178356240` | Separated |

Every new bound is below the retained horizon cap, so the minimum selects
the endpoint value in all five cases. Four exact margins `rho_i-B_b` are
positive; cell 37's margin is negative. The other forty-three allowances
and their separated classifications are unchanged. In total, forty-seven
closed cells now certify separation, leaving only

\[
\mathcal P=(37/8,19/4),\qquad |\mathcal P|=1/8.
\]

The adjacent separated cells exclude both endpoints. This open component
cannot contain any closed residence interval of length `1/2`. Consequently,
conditional on the retained source family, validated comparison tubes and
error theorem, **no actual source member has the declared half-unit target
residence wholly within `[0,6]`**. The conclusion covers residence starts
between grid points. It excludes a necessary geometric requirement of the
[operational acquisition criterion](SINE_REPLICA_PULSE.md#replica-fixed-budget-collective-preparation),
not just four aligned positive cells. Brief visits inside the remaining
component are unresolved by this reduced predictor, and behavior beyond
time six is outside this result.

This is a source- and task-specific sufficiency result for the ten-coordinate
comparison **with its source-associated internal-error allowances**. It
retains internal initialization through those bounds and does not establish
exact autonomous closure, internal irrelevance or a universal reduced state.
The stronger [earlier full-state absence](SINE_REPLICA_PULSE.md#replica-target-separation)
remains separate prior evidence. No new actual trajectory, actual-error
measurement, formation event or physical identification was produced here.

The earlier pending-language records the prospective protocol. This section
closes its residence decision; resolving the remaining possible times requires
a separate scientific consumer rather than automatic further refinement.
The [research program](../RESEARCH_PROGRAM.md#research-current-question)
owns the next question and its selection against the global objective.
