# A supplied form probe of the acquired two-port C9 pair

<a id="sine-two-port-probe"></a>

## Question, source handoff and scope

The [capture result](SINE_TWO_PORT_CAPTURE.md#sine-two-port-capture-result)
places the entire prepared thirty-six-coordinate family in the compatible
two-port basin at slow time \(\sigma_*=1025\). This result asks whether
a common, explicitly supplied form input produces a resolvable receiver
response in that acquired pair and an unjoined control, while preserving
the winding identities and accounting for the input's storage jump.

The discriminating mechanism is transport through supplied contacts. A
nonzero receiver response does not distinguish the winding classes, prove
that the acquired organization is necessary for transport, or identify a
physical interaction. Acquisition matters here as the justified source of
the complete pre-probe state and its retained identity. The response theorem
itself also applies to other states meeting its explicit endpoint bounds.
The simpler form-diffusion model \(x_\tau=-Ax\) already has the
positive contact-transfer term proved below, without any phase coupling.
The proposed sign contrast therefore cannot select the sine law against
that alternative. Its additional conclusions concern a controlled input
applied to the previously acquired complete state, bounded work, finite
readout error and preservation of its winding identity.

The runtime certificate is conditional on that endpoint domain. Its fixed
research protocol additionally binds the domain to the previous capture
theorem and its retained execution. Neither a cached verdict nor a small
nominal-reference error alone supplies this handoff. No actual state is
reset to the compatible target.

## Complete rows, endpoint domain and common input

Retain the capture owner's two unit C9 cycles, donor nodes `0,...,8`,
receiver nodes `9,...,17`, and unit contacts `(0,9)` and `(1,10)`. Write
\(L\) for this joined Laplacian, \(M\) for its degree matrix,
\(A=M^{-1}L\), and \(f(\theta)=M^{-1}S(\theta)\), where
\(S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i)\). Each component has
degree mass twenty and the total mass is forty. The ports have degree three;
all other degrees are two. Keep held capacities one, beta one,
\(e=1023/1024\), \(w=1/1024\), and
\[
\gamma=\frac1{1023\pi}<\bar\gamma=\frac1{3069},\qquad
\tau=et,\qquad\sigma=\gamma^2\tau.
\]
Between events the complete law remains
\[
x_\tau=-Ax+\gamma f(\theta),\qquad
\theta_\tau=\gamma Ax. \tag{1}
\]
There is no continuous input during the response window. Both rows are
retained; the heat semigroup used below is only a variation-of-constants
tool for the form row of (1).

Let \(P_M\) remove the full degree-weighted mean and let
\(\theta_*^m\) be the compatible target with the actual state's conserved
continuous phase mean. Its correlated root, winding periods \((2,1,0)\)
and criticality are those of the
[compatibility owner](SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-compatibility).
Use the same continuous lifts; no independent component rotation is made.
At the pre-probe handoff require
\[
\|P_Mx^-\|_M\le X,\qquad
\|\theta^- -\theta_*^m\|_M\le Y,
\qquad X=\frac1{8192},\quad Y=\frac1{1024}. \tag{2}
\]
The previous capture theorem proves strict bounds inside (2) for every
member of its original independent nodewise form and radian phase error
family, with both source radii \(1/65536\). These deliberately larger
endpoint bounds retain that actual family without treating its sharp saved
bounds as new source coordinates.

At the fixed handoff supply the hybrid event
\[
x^+=x^-+a r_D,\qquad \theta^+=\theta^-,\qquad
r_D=\mathbf1_{\{0,\ldots,8\}},\qquad a=\frac1{2048}. \tag{3}
\]
This is a uniform signed form increment on all donor nodes. It is an
external, instantaneous preparation action, not a derived event selector
or a finite-duration pressure flow. The same nodal action is supplied to
the unjoined control. Its internal coefficient rows are given by (1) with
that control's own Laplacian and degree matrix.

The control consists of the two isolated unit C9 cycles from its initial
preparation onward. It starts with the same nominal eighteen nodal phases,
zero nominal form and the same independent source error budgets as the
capture experiment, and evolves to the same original handoff time. Removing
contacts from a previously joined state is not this control preparation.
The argument permits either paired source errors or independently selected
members of the declared family. No control equilibrium reset is needed.

Measure elapsed fast time \(s=\tau-\tau_*\) after (3), with fixed
\(h=1/4\). Thus the added original time is
\(h/e=256/1023\), and the added slow time is
\(1/(4\cdot1023^2\pi^2)\). The handoff itself remains
\[
\sigma_*=1025,\qquad
\tau_*=1025\cdot1023^2\pi^2,\qquad
t_*=1025\cdot1023\cdot1024\pi^2. \tag{4}
\]
These are structural clocks; no laboratory-unit identification is supplied.

## Work and preservation of the acquired identity

The normalized storage and loss are unchanged:
\[
H(x,\theta)=\frac12 x^TLx+
 \sum_{\{i,j\}\in E}[1-\cos(\theta_j-\theta_i)],\qquad
H_\tau=-\|Ax\|_M^2. \tag{5}
\]
Define \(b=e_0+e_1-e_9-e_{10}\). The actual joined support gives
\[
Lr_D=b,\qquad r_D^TLr_D=2,\qquad
\|b\|_{M^{-1}}=\frac2{\sqrt3}<\frac76. \tag{6}
\]
Consequently the exact event work, in these storage units, is
\[
W=H(x^+,\theta^-)-H(x^-,\theta^-)
  =a^2+a b^Tx^-,\qquad
a^2-\frac76aX\le W\le a^2+\frac76aX. \tag{7}
\]
This accounts for the jump separately from continuous loss. It does not
infer a reservoir, event passivity or a physical energy scale. For (2)--(3)
the lower work bound is positive, so the declared action supplies storage.
The upper work bound obeys
\[
W\le\frac{31}{100663296}<W_{\rm allowed}=\frac1{2000000}. \tag{8}
\]

The full form mean increases by \(a/2\). Since
\(\|P_Mr_D\|_M=\sqrt{10}<19/6\), the post-event relative norm is
at most
\[
q_0=X+\frac{19}{6}a=\frac{41}{24576}. \tag{9}
\]
The phase mean and target representative are unchanged. Criticality and
the normalized Laplacian upper bound two give the pre-event estimate
\(H^- -H_*\le X^2+Y^2\). The capture owner's local ball has radius
\(R=1/12\) in the combined relative-form and phase norm and boundary
storage at least \(B=1/648000\), provided the target's minimum acute
margin exceeds \(1/8\) radian. The two strict admissions after (3) are
\[
q_0^2+Y^2<R^2,\qquad
X^2+Y^2+a^2+\frac76aX<B. \tag{10}
\]
For the chosen constants the second margin is exactly
\[
B-\left(X^2+Y^2+a^2+\frac76aX\right)
 =\frac{181201}{679477248000}>0. \tag{11}
\]
Thus the same loss and boundary argument traps the post-event state for
every later uninterrupted evolution. It retains periods \((2,1,0)\)
and converges to \(\theta_*^m\) and uniform form equal to the new
mean. This preserves the established mathematical identity through the
supplied probe; it does not establish autonomous maintenance against
arbitrary or repeated inputs.

For the unjoined control, (3) is constant on each connected component and
therefore lies in its Laplacian kernel. Its work is exactly zero and its
relative motion is unchanged by the input. The original control preparation
also retains each isolated winding identity. To see this directly, each
ring has degree mass eighteen, diameter four, and normalized gap at least
\(1/18\). The uniform class-two twist has acute margin
\(\pi/18>1/6\), and class one's margin is larger. In the relative
radius \(1/12\), the phase Hessian is bounded below by \(L/25\).
The ring boundary storage is consequently at least \(1/129600\).
After subtracting each ring's own means, its initial combined squared norm
and excess storage are at most
\(36/65536^2=9/1073741824\). They are strictly inside that radius and
barrier. The same loss and acute uniqueness argument traps each component;
the donor's common form shift at (3) does not change these conclusions.

## Receiver observation and the exact control result

For the joined graph define
\[
\ell_R u=\frac1{20}\sum_{i=9}^{17}M_{ii}u_i,\qquad
J=\ell_R[x(h)-x(0^+)]. \tag{12}
\]
The event leaves receiver coordinates unchanged, so the same baseline can
be read immediately before (3). Use the corresponding actual-degree
normalization, mass eighteen, for the unjoined receiver, and call its
increment \(J_0\). The observation rule is common: the receiver's
degree-weighted mean form increment. Its normalization is part of the
declared measurement map rather than an invented shared degree matrix.
The fixed coefficient vectors differ between the two supports; this is
the same support-dependent rule, not an identical linear functional.

Summing (1) over an isolated receiver with its actual degrees cancels
both Laplacian and sine terms. Therefore
\[
J_0=0 \tag{13}
\]
for every control initial state and every positive observation duration.
This equality does not use small errors, relaxation, equilibrium or a
linearization. The control identity argument above is a separate claim.

There are four scalar readings: before and after in each model. Allow
independent signed readout errors of magnitude at most
\(\delta=2^{-26}\) per reading. If \(\widehat C\) is the observed
joined-minus-unjoined increment, then
\[
|\widehat C-(J-J_0)|\le4\delta. \tag{14}
\]
No cancellation, independence in a probabilistic sense, or averaging
advantage is assumed. Common form origins and their source errors cancel
from each increment.

## Full-law finite-window bounds

Let \(T(s)=e^{-sA}\). It is a contraction in the degree metric and
preserves constants and weighted means. Globally on continuous lifts,
\(\|A\|_M\le2\), \(f(\theta_*^m)=0\), and
\(\|f(\theta)-f(\theta_*^m)\|_M\le
2\|\theta-\theta_*^m\|_M\). The latter follows from the sine
Jacobian \(-M^{-1}B\operatorname{diag}(\cos)B^T\); its metric
operator norm is at most that of \(A\), even outside the acute chart.

Write \(Q\) and \(P\) for the suprema over \([0,h]\) of the
post-event centered form norm and target phase distance. Variation of
constants for the form row and integration of the phase row give
\[
Q\le q_0+2\bar\gamma hP,\qquad
P\le Y+2\bar\gamma hQ.
\]
For \(c=2\bar\gamma h<1\), use the explicit enclosing bounds
\[
Q\le Q_{\max}=\frac{q_0+cY}{1-c^2},\qquad
P\le P_{\max}=Y+cQ_{\max}. \tag{15}
\]
These estimates retain the nonlinear phase response and original form.
They do not identify phase with the auxiliary gradient trajectory or treat
the compatible state as the actual preparation.

The exact form variation of constants splits (12) into three terms:
\[
\begin{aligned}
J={}&a\ell_RT(h)r_D
 +\ell_R[T(h)-I]x^-\\
 &+\gamma\int_0^h\ell_RT(h-s)f(\theta(s))\,ds. \tag{16}
\end{aligned}
\]
The first is a supplied-kick heat term, the second the pre-existing form
background, and the third the actual nonlinear sine feedback. None is
discarded from the result.

For the first term, \(P_{\rm walk}=I-A\) is the actual nonnegative
random-walk matrix. Since \(\ell_Rr_D=0\) and
\(\ell_RP_{\rm walk}r_D=1/10\), the nonnegative exponential series
proves, for \(0<h\le1\),
\[
\ell_RT(h)r_D\ge\frac{he^{-h}}{10}
 \ge\frac{h(1-h)}{10}. \tag{17}
\]
For an upper bound put \(v=r_D-\tfrac12\mathbf1\). Reversibility
gives
\[
\ell_RT(h)r_D=\frac{10-\langle v,T(h)v\rangle_M}{20}.
\]
Its derivative is nonnegative and its second derivative nonpositive, by
the nonnegative spectrum of the self-adjoint \(A\). Its initial value
is zero and its initial derivative is \(1/10\), hence
\[
0\le\ell_RT(h)r_D\le h/10. \tag{18}
\]

Since \(-\ell_RA=b^T/20\), metric contraction and (6) bound the
second term by
\[
|\ell_R[T(h)-I]x^-|\le\frac{7hX}{120}. \tag{19}
\]
The mean-free restriction of \(\ell_R\) has exact dual norm
\(1/\sqrt{40}<1/6\). Every \(f(\theta)\) has weighted mean zero,
so contraction and (15) bound the last term in (16) by
\[
\frac{\bar\gamma hP_{\max}}{3}. \tag{20}
\]
In particular this estimate does not require cancellation of the actual
contact sine currents. Define
\[
E_{\rm bg}=\frac{7hX}{120},\qquad
E_{\rm sine}=\frac{\bar\gamma hP_{\max}}3.
\]
Equations (13)--(20) give the complete-state response enclosure
\[
\frac{ah(1-h)}{10}-E_{\rm bg}-E_{\rm sine}
\le J-J_0\le
\frac{ah}{10}+E_{\rm bg}+E_{\rm sine}. \tag{21}
\]

For the fixed constants,
\[
Q_{\max}=\frac{128735343}{77158488064},\qquad
P_{\max}=\frac{150742119}{154316976128},
\]
and the lower side of (21) is exactly
\[
L_{\rm response}=\frac{3265940501}{444432891248640}.
\]
With the four readout errors retained, the declared observation threshold
\(C_{\min}=1/262144\) has strictly positive margin
\[
L_{\rm response}-4\delta-C_{\min}
 =\frac{98820691289}{28443705039912960}>0. \tag{22}
\]
Thus every state in (2), including the whole acquired source family,
produces \(\widehat C>C_{\min}\) while meeting the work and identity
requirements, conditional on the supplied input and observation contracts.
Equation (22) is an analytic sufficient prediction, not a measured response
or a trajectory simulation.

## Prospective protocol and admission boundary

<a id="sine-two-port-probe-protocol"></a>

Before the first new report assessment, freeze the following:

- The joined support, complete rows, capacities, coefficients, source and
  three structural clocks are the unchanged capture model. Its entire
  original error family reaches the handoff time (4), and (2) is justified
  by that theorem's retained first execution. The unjoined control is
  prepared separately from the same declared source on its own support.
- The single supplied action is (3), with amplitude \(1/2048\), applied
  to both models at the fixed handoff. There are no resets, contact events,
  further inputs, searches over amplitude, or response-selected times.
- The duration is \(h=1/4\) in fast time. The observation is precisely
  (12)--(14), using actual-degree receiver means, four independent error
  bounds \(2^{-26}\), and strict contrast threshold \(1/262144\).
- The work allowance is \(1/2000000\). Rebuild (7)--(11), including
  both work endpoints, the strict post-event radius and storage margins,
  and the same target's fresh minimum acute margin above \(1/8\) radian.
- Re-admit the primary compatible `(2,1)` root using thirty-two outer and
  sixty-four inner strict-sign refinements. Its enclosure supplies target
  admission; it does not replace the actual endpoint by a root midpoint.
- Rebuild all rational bounds (15)--(22) from admitted primitive constants.
  Preserve each failed strict obligation as unavailable. Equality with a
  strict threshold is unavailable; a cached passing flag cannot supply it.
  There is no trajectory integration, adaptive search or response-fitting
  budget in this analytical certificate.
- Archive the prospective proof, protocol and complete producing source
  before assessment, then retain the report and manifest separately. Link
  the unchanged capture artifacts and declare their retained execution as
  a premise of source acquisition. A read-only consistency audit can
  re-admit source, law, coordinates, work policy and saved evidence without
  rerunning that producer. Such an audit does not independently reproduce
  every validated Taylor remainder or authenticate execution chronology.

The generic endpoint-domain API may be used without a particular acquisition
history, but then certifies only the conditional theorem for (2). The fixed
research record additionally requires the explicit capture-source handoff.
Do not promote one scope to the other through report labels or filenames.

At this prospective declaration no new probe report has been evaluated.
Any retained result must be appended separately. A successful assessment
would certify supplied-contact transmission with a finite readout margin,
bounded supplied work and retained winding identity. It would not establish
a winding-specific response, an autonomous input or contact rule, physical
binding, or an experimental measurement bridge.

## Retained certificate after export recovery

<a id="sine-two-port-probe-result"></a>

The original frozen execution completed its certificate calculations but
failed while exporting the auxiliary control-identity dictionary: the shared
projector rejected a dictionary passed as one value. No primary report was
saved. The [first-attempt record](../../docs/assets/sine_formed_classes/two-port-probe-v1.first-attempt.json)
preserves that software failure; it is neither a failed mathematical
inequality nor an available first response. The original attempt is not
represented as a successful retained assessment.

A separately frozen
[export-recovery wrapper](../../docs/assets/sine_formed_classes/two-port-probe-v1.export-recovery.py.txt)
projects the dictionary's individual values and records the evaluation
history. Because the first process retained no report, the wrapper repeats
the deterministic analytical assessment with the identical scientific
inputs and runtime. It changes neither the prospective inequalities nor
the source, law, input, observation, time, target-refinement budget or
thresholds. The original frozen evaluator, archive and prospective proof
remain unchanged. The earlier capture producer is not rerun.

This retained recovery assessment returned `certified_probe`, with no
unavailable reasons. All ten fixed stopping conditions passed, including
the acquired-source handoff, fresh compatible target, four-error observation
margin, positive supplied work within allowance, joined identity and
recovery, and the original unjoined control's separate identity bound.
The saved report contains exact rational endpoints. The following values
are approximate displays of those bounds, not measured responses or
replacement thresholds:

| Retained quantity | Approximate bound |
| --- | ---: |
| True joined receiver mean increment | `[7.34855715e-6, 1.40137475e-5]` |
| Joined-minus-unjoined contrast with all four readout errors | `[7.28895251e-6, 1.40733522e-5]` |
| Lower contrast margin above `1/262144` | `3.47425524e-6` |
| Supplied joined storage jump | `[1.68879827e-7, 3.07957331e-7]` |
| Post-input storage margin below the capture barrier | `2.66677068e-7` |

The unjoined receiver's true increment and the unjoined event work are
exactly zero. The observed contrast includes its two readout errors as
well as the joined model's two errors. Its observation uses the same
actual-degree mean rule with its own coefficient vector; the two models
are not assigned an identical linear functional.

The [source handoff audit](../../src/tnfr/research/sine_two_port_handoff.py)
re-admits the original source and complete law, reconstructs the target
from strict root brackets, checks the retained reference chain and
recomputes the complete-state handoff bounds. The admitted original-form
and phase norms are approximately `5.39860959e-5` and `3.51060503e-4`,
inside the larger budgets (2). The audit retains the explicit premise
that the earlier Picard inclusions, local remainders and endpoint centers
are valid evidence from the archived execution. It does not independently
reproduce every Taylor remainder or authenticate that execution.

Consequently the conditional probe conclusion covers every member of the
unchanged original thirty-six-coordinate source family after its justified
handoff at \(\sigma=1025\). The joined state retains periods
\((2,1,0)\) through the input and subsequent uninterrupted flow. It
converges to the same phase geometry with its original phase mean and form
mean increased by \(a/2\). Each unjoined component retains its winding;
the control donor's form mean instead increases by \(a\). These are
analytic full-family conclusions, without a probe trajectory, an endpoint
reset or a measured physical response.

The retained
[protocol](../../docs/assets/sine_formed_classes/two-port-probe-v1.protocol.json),
[original producing source archive](../../docs/assets/sine_formed_classes/two-port-probe-v1.source.zip),
[recovery report](../../docs/assets/sine_formed_classes/two-port-probe-v1.json)
and [manifest](../../docs/assets/sine_formed_classes/two-port-probe-v1.manifest.json)
bind the unchanged prior capture artifacts and the separate failure and
recovery records above. The original source archive SHA-256 is
`d54a607c76bdd992bedf26f92618360ed506739e2c2e70ed0b260fc704bda578`,
based on revision `f1ce7f196a50708e5e5910cdf68d0143265059a1` with the
three declared runtime overlays. The prospective text remains a
byte-identical prefix of this owner. Content hashes support association
and source recovery, not independent chronology authentication.

The positive contrast establishes the declared supplied-contact
transmission, bounded work and retained mathematical identity. The pure
form-diffusion alternative already transmits the same donor input, as
(17)--(18) demonstrate. This baseline probe therefore does not select the
sine law, show a winding-specific signature or establish that acquired
organization is necessary for transport. No autonomous input or contact
selection, physical binding or laboratory observation bridge follows.
