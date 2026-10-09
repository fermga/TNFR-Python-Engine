# A changed-input prediction from the nonlinear collective interface

<a id="sine-class-changed-input-prediction"></a>

The [collective interface](SINE_CLASS_COLLECTIVE_INTERFACE.md) supplies
causal central-port equations through cubic amplitude order. This owner
admits one new input and an independently executable prediction from those
equations. The input changes the driven component, rather than scaling
the earlier donor word or reflecting donor and receiver.

The comparison tests whether retaining the internal causal memory matters
for this observation. Its leading separation already exists in the linear
response. Success would not establish that the nonlinear correction is
indispensable, nor would it establish a new class-specific contrast or a
physical measurement law. A complete nonlinear forward evaluation remains
a separate prospective obligation.

<a id="sine-changed-input-complete-protocol"></a>
## Fixed complete law, source and changed word

Retain the three C9 components, contacts \((4,13),(13,22)\), actual degrees,
unit capacities, and the class tuples \((1,k,1)\), separately for \(k=1,2\).
Use the unchanged complete rows and structural clock
\[
 x'=-Ax+\gamma D^{-1}S(\theta),\qquad
 \theta'=\gamma Ax,\qquad
 A=D^{-1}L,\qquad \gamma=\frac1{1023\pi},\qquad
 \tau=\frac{1023}{1024}t.
 \tag{1}
\]
Continuous phase deviations are relative to the aligned target
\(\Theta_k\). The original acquired-source and contact construction is the
[complete two-class preparation](SINE_CLASS_NONLINEAR_ORGANIZATION.md#sine-nonlinear-organization-source):
each component retains its complete form and phase residuals, bounded in
Euclidean norm by \(\epsilon=10^{-32}\), and its original zero component
means before contact. In particular every residual coordinate is bounded
by \(\epsilon\). The joined source is not replaced by its target.

A Cartesian interval cover of these residuals is an outer cover of the
same acquired family. It does not assert independent preparability of
its corners, remove the original zero-sum correlations, or equate
sources between classes or models. A protocol label such as
\(\texttt{independent\_source\_uncertainty}\) denotes this conservative
uncertainty treatment, not an additional preparation mechanism.

Reference common origins remain the declared original zero origins.
The small actual joined weighted means are part of the source uncertainty;
degree-centering this first source is not assumed to preserve its
maximum-coordinate bound. The source at \(0^-\) precedes the impulse.
Contact is followed by the same prescribed event in each class:
\[
 x(0^+)=x(0^-)+q e_{13},\qquad
 \theta(0^+)=\theta(0^-),\qquad
 q=\frac7{10000},\qquad H=1.
 \tag{2}
\]
There are no further impulses or resets. The total central-input variation
is \(q<7/5000\), and the horizon lies in the interface domain.

The observable is one endpoint form value
\[
             r_k=x_{13}(H)
 \tag{3}
\]
in the declared common origin. No baseline is subtracted, and no phase
derivative or unobserved hidden coordinate is measured. Each model's
recording error has magnitude at most
\[
                       \delta=10^{-8}.
 \tag{4}
\]
The two classes are checked separately. A difference between their
predictions is not a stopping requirement.

<a id="sine-changed-input-memory-loss-comparator"></a>
## The grounded tangent port truncation

Retain the six central deviations \(v=(x_4,x_{13},x_{22},
y_4,y_{13},y_{22})\). Partition the full tangent generator as in the
[interface kernel owner](SINE_CLASS_COLLECTIVE_INTERFACE.md#sine-collective-interface-kernels),
\[
 J_k=\begin{pmatrix}E_k&B_k\\C_k^{hv}&M_k\end{pmatrix}.
 \tag{5}
\]
The declared memory-loss comparator is the six-coordinate law
\[
       v'=E_kv,\qquad
       v(0^+)=v(0^-)+(0,q,0,0,0,0).
 \tag{6}
\]
Its own initial visible residual coordinates are bounded by \(\epsilon\).
Its reference origins are supplied and retained externally. The omitted
hidden deviations are grounded at zero: both the convolution and the
hidden-initialization source are absent.

Equation (6) is called the **grounded tangent port truncation**. It is not
the stationary Schur comparator and not a full-state evolution with its
hidden state retained. Its class-dependent instantaneous block \(E_k\)
is rebuilt from the same target and actual degrees. No newly normalized
three-node graph or fitted coupling is supplied.

The complete source, charge, work and identity claims of (1) are not
transferred to (6). In particular \(E_k\) does not annihilate a uniform
port vector when the corresponding hidden values are grounded. It would
be incorrect to evolve accumulated common origins with (6). The present
protocol uses the original zero origins and one window.

<a id="sine-changed-input-prior-memory-gap"></a>
## A response-free finite separation

Put \(g=1/3000>\gamma\) and \(D_H=1-2g^2H^2\). First consider zero
residuals under (2). Let \(V=\{4,13,22\}\) and \(P=I-A\). The full
heat semigroup is
\[
 \exp(-At)=e^{-t}\sum_{n\ge0}\frac{t^nP^n}{n!}.
 \tag{7}
\]
The entries of \(P\) are nonnegative and its rows sum to one. The
grounded port heat law has generator \(-A_{VV}\), so its corresponding
series uses \(P_{VV}\). Every port-only walk occurs in the full series.
Additional walks through hidden nodes have nonnegative weights.

At the mediator port there are exactly two length-two walks that leave
the central set and return through its internal neighbors. Their total
weight is
\[
 (P_{VH}P_{HV})_{13,13}
       =2\left(\frac14\right)\left(\frac12\right)=\frac14.
 \tag{8}
\]
Consequently the difference between the full and grounded heat responses
at this port satisfies
\[
 q\bigl[\exp(-AH)_{13,13}
       -\exp(-A_{VV}H)_{13,13}\bigr]
       \ge \frac{q e^{-H}H^2}{8}.
 \tag{9}
\]
This bound is independent of the mediator class; it does not evaluate a
finite matrix exponential or an observed response.

As an independent local check, the full and grounded nominal port forms
agree in value and first derivative at zero. Their second derivative
difference is
\[
       q\,(\mathcal K_k(0))_{x_{13},x_{13}}
          =\frac q4\left[1-\gamma^2\cos(k\alpha)\right],
          \qquad \alpha=2\pi/9.
 \tag{9a}
\]
This coefficient checks both the hidden return paths and the complete
phase feedback. The finite proof below does not replace its tail by a
formal onset.

Both heat semigroups are maximum-norm contractions. The full sine force
relative to the target and the grounded tangent phase force have
Lipschitz constant at most two. For either model, heat contraction and
its phase row give
\[
 \|x\|_\infty\le\frac q{D_H},\qquad
 \|y(t)\|_\infty\le\frac{2gqt}{D_H},\qquad
 \|x(H)-x_{\rm heat}(H)\|_\infty
       \le\frac{2g^2qH^2}{D_H}.
 \tag{10}
\]
The last estimate follows by integrating the phase bound, rather than
replacing the phase by an independent constant source.

Let \(r_k^{\rm nom}\) denote the full nominal sine endpoint and
\(r_{E,k}^{\rm nom}\) the nominal endpoint of (6). Combining (9)-(10),
\[
 r_k^{\rm nom}-r_{E,k}^{\rm nom}
 \ge qH^2\left(\frac{e^{-H}}8-\frac{4g^2}{D_H}\right)
 >q\left(\frac1{24}-\frac{4g^2}{1-2g^2}\right)
 >\frac1{40000}.
 \tag{11}
\]
Here \(H=1\) and \(\exp(1)<3\). The strict lower bound is fixed before
the new causal coefficients are evaluated. It predicts a positive
memory-retention effect for each class, not a lower bound on their
cross-class difference.

<a id="sine-changed-input-causal-numerics"></a>
## Independent causal computation and numerical budget

The predictor evaluates the six-port Volterra hierarchy, including the
hidden quadratic functional and its cubic feedback, from the
[fixed grounded-path kernels](SINE_CLASS_COLLECTIVE_INTERFACE.md#sine-collective-interface-kernels).
It does not call the complete 54-coordinate coefficient recurrence or use
a full forward response as a fitted kernel. Hidden reconstruction in
these fixed convolutions is part of the reduction; it is not a new
autonomous hidden-state integration.

Use ordinary time coefficients through degree \(N=32\), exact rational
convolution weights, and the shared 128-bit outward interval arithmetic.
There is one interval \([0,1]\), with no subdivision or adaptive retry.
Grounded-path spatial powers and two-channel kernel powers provide the
coefficients of \(G(t)\); multiplication by the exact coupling factors
gives those of \(B G(t)C^{hv}\). The integral of two monomials uses
\[
 \int_0^t(t-s)^i s^j\,ds
       =\frac{i!\,j!}{(i+j+1)!}\,t^{i+j+1}.
 \tag{12}
\]
The construction retains all factors of \(\gamma\) in their original
coordinates and enclosures of the stated exact trigonometric constants.

Substitution into the Volterra equations proves coefficient by
coefficient that this causal computation produces the time series of the
complete amplitude levels. In particular hidden level two is formed
before its forcing enters visible level three. Keeping only the
linearly observable hidden modes would not justify the cubic prediction.

An independent positive coefficient majorant certifies the time tails.
Let \(\mathcal L=201/100\), so
\(\|J_k\|_\infty\le2+2g<\mathcal L\). For the original-coordinate
bilinear and trilinear maps,
\[
 \|\mathbb Q(z,w)\|_\infty\le2g\|z\|_\infty\|w\|_\infty,\qquad
 \|\mathbb T(z,w,u)\|_\infty
       \le\frac43g\|z\|_\infty\|w\|_\infty\|u\|_\infty.
 \tag{13}
\]
Coefficientwise majorants for the three levels are
\[
 \begin{split}
 V_1(t)&=q e^{\mathcal Lt},\\
 V_2(t)&=2gq^2t e^{2\mathcal Lt},\\
 V_3(t)&=\left(\frac43gq^3t+4g^2q^3t^2\right)e^{3\mathcal Lt}.
 \end{split}
 \tag{14}
\]
For example, \(V_2'-\mathcal LV_2-2gV_1^2=\mathcal LV_2\).
Similarly \(V_3'-\mathcal LV_3-4gV_1V_2-(4/3)gV_1^3
=2\mathcal LV_3\). These residuals have nonnegative coefficients.
Their initial values dominate the single first-level jump and the zero
higher levels. This proves domination of absolute exact coefficients,
not just a pointwise solution estimate.

Write \(E_m(z)=\sum_{n\ge m}z^n/n!\). For \(0\le z<m+1\),
\[
 E_m(z)\le
 \frac{z^m}{m!}\frac1{1-z/(m+1)}.
 \tag{15}
\]
The omitted tails after degree 32 are therefore bounded by
\[
 \begin{split}
 {\cal T}_1&=qE_{33}(\mathcal LH),\\
 {\cal T}_2&=2gq^2H E_{32}(2\mathcal LH),\\
 {\cal T}_3&=\frac43gq^3H E_{32}(3\mathcal LH)
                  +4g^2q^3H^2 E_{31}(3\mathcal LH).
 \end{split}
 \tag{16}
\]
All denominators in (15) are positive at the declared values. The
nominal central form uses levels one and three and charges
\({\cal T}_1+{\cal T}_3\). The nominal comparator series charges
\({\cal T}_1\); the same coefficient majorant bounds its generator.
Exact rational substitution in (15)-(16) gives
\({\cal T}_1+{\cal T}_3<10^{-22}\). This bound on the exact time
tail is separate from rounding and interval dependency in the retained
finite coefficients.

The coefficient arithmetic and retained tail bounds together must
produce a numerical interval of radius at most \(10^{-12}\) for each
nominal endpoint. This is an explicit postcondition, not a promise that
the unexecuted interval computation achieves it. Arithmetic widening is
retained separately from the analytic amplitude and source allowances.
No reduction of the reading error is inferred from numerical precision.

<a id="sine-changed-input-source-and-verdict"></a>
## Initialization, approximation and recording allowances

Let \(I_k\) be the computed interval for the nominal cubic central
prediction and \(J_k^{E}\) the computed interval for the nominal grounded
comparator. Both include their interval arithmetic and time tails.
The complete unknown source is retained as in the interface: its linear
source is \(\Pi\exp(J_kt)z_0\), and its additional nonlinear coupling has
the separate initialization defect. It is neither set to zero nor
evaluated from fitted endpoint data.

At \(H=1\), put \(\ell=1-2g\). The original maximum-coordinate source
premise bounds each linear-source contribution by \(\epsilon/\ell\).
The grounded comparator has its own visible-source allowance
\(\epsilon/\ell\); it consumes no hidden source. With the
[interface fidelity bounds](SINE_CLASS_COLLECTIVE_INTERFACE.md#sine-collective-interface-fidelity),
define
\[
 \begin{split}
 B_{\rm full}
  &=\frac{\epsilon}{\ell}
       +R_5(q,1)+E_{\rm init}(q,1,\epsilon),\\
 B_E&=\frac{\epsilon}{\ell},\\
 I_k^{\rm actual}&=I_k+[-B_{\rm full},B_{\rm full}],\\
 J_k^{E,\rm actual}&=J_k^E+[-B_E,B_E].
 \end{split}
 \tag{17}
\]
The exact common origins are retained externally in both predictions.
The initialization defect applies to all admitted residuals, without
assuming nominal reflection parity for the actual source.

There is one scalar recording per model and class. Thus strict disjoint
recorded intervals in the predicted direction require
\[
 \inf I_k-\sup J_k^E-B_{\rm full}-B_E-2\delta>0,
                  \qquad k=1,2.
 \tag{18}
\]
This charges two reading errors, not the eight- or sixteen-reading
allowances of the earlier mixed-response protocols. The two classes
must each pass (18); averaging them cannot replace either result.

The radius ceiling also gives a conditional resolution guarantee.
An interval of radius at most \(\eta=10^{-12}\) containing an exact
coefficient can have an endpoint as far as \(2\eta\) from it.
Consequently (11), containment and the two numerical-radius
postconditions imply that the left side of (18) is strictly greater than
\[
 q\left(\frac1{24}-\frac{4g^2}{1-2g^2}\right)
 -4\eta-2R_5(q,1)-E_{\rm init}(q,1,\epsilon)
 -\frac{2\epsilon}{\ell}-2\delta
       >29\,10^{-6}.
 \tag{18a}
\]
The second amplitude allowance here transports the prior full-sine
gap to the cubic numerical band before that band is expanded in (17).
It is used only in this resolution argument; the direct assessment
in (17)-(18) does not add another amplitude allowance.

The pre-computation lower bound (11) is much larger than the admitted
noise and approximation terms. It supplies a discriminating prediction
before numerical evaluation. The actual certificate still reconstructs
(18) from the computed intervals; it does not intersect those intervals
with (11) or substitute the prior lower bound for numerical evidence.
Failure of a numerical-radius or separation postcondition is an
unavailable or inconclusive computation, not a proof of physical
equivalence. A rigorously inconsistent enclosure requires an integrity
or mathematical audit before interpreting either result.

<a id="sine-changed-input-work-and-identity"></a>
## Work, charge and identity of the changed input

The original acquired source and contacts retain their separate costs.
The original joined excess storage is at most \(22\epsilon^2\).
For the new mediator impulse, its degree is four and the exact work is
\[
 W=q(Lx(0^-))_{13}+2q^2,\qquad
 |(Lx(0^-))_{13}|\le8\epsilon.
 \tag{19}
\]
Hence
\[
 2q^2-8q\epsilon\le W\le2q^2+8q\epsilon<10^{-6}<2\,10^{-6},
 \qquad
 22\epsilon^2+W<10^{-6}<\frac1{388800}.
 \tag{20}
\]
This bound uses the actual full preevent state. It does not replace
pressure by \(4x_{13}\), nor assign a work reserve from future loss.

The Euclidean quotient radius immediately after the jump obeys the
conservative bound
\[
 \|P_0x^+\|_2^2+\|P_0y^+\|_2^2
       \le27(q+\epsilon)^2+27\epsilon^2<\frac1{144}.
 \tag{21}
\]
Together with (20), the unchanged acute trapping barrier at
\(r=1/12\) preserves the full pattern identity. Storage decreases
between events under the declared complete law. The actual
degree-weighted form mean changes by
\[
                     \Delta\mu_x=\frac{4q}{58}=\frac{2q}{29};
 \tag{22}
\]
the phase mean is unchanged. Any subsequent convergence belongs to
that new mean leaf, not to a reset zero-mean state.

Equations (19)-(22) admit this particular changed word. They do not
transfer a general work guarantee to all inputs in the interface
domain, nor prove repeated operation of this new protocol. The
grounded comparator is evaluated only with its own stated six-state
law and source/error premises.

## Prospective evidence boundary

The fixed input, observation, comparator, error allowances and degree-32
policy above must be associated with their implementation before the
first new causal coefficient calculation. That calculation is a reduced
prediction, not an independent complete-law response. Any later reserved
forward comparison must preserve its full source, event, numerical and
recording association separately.

The present admission does not execute a new acquisition, fit hidden
kernels, regenerate an earlier producer, or identify a fundamental
physical law. Its new obligation is an explicit changed-input transfer
test of the nonlinear causal interface on the same supplied support.

<a id="sine-changed-input-prediction-result"></a>
## Retained causal prediction

The first fixed calculation completed for both classes. Its source and
protocol were associated before coefficient evaluation, after the
[design declaration](../../docs/assets/sine_formed_classes/class-collective-prediction-v1.design.json).
The [evidence archive](../../docs/assets/sine_formed_classes/class-collective-prediction-v1.evidence.zip)
contains that design, the protocol, source archive, freeze receipt,
exclusive attempt and complete outcome. The full Git base is
`5b9e721dc6240ea2d21d0690e1cdaa9b4d5b318d`; the source archive declares
the two runtime overlays and preserves this owner's prospective prefix.
Byte associations do not independently authenticate chronology or execution.

The finite-polynomial midpoint summaries below are for orientation only.
The decision uses the retained rational endpoint enclosures, independently
reconstructed time tails, source and analytic defects, and both reading
errors in (18).

| Mediator class | Causal nominal port prediction, approximately | Grounded comparator, approximately | Strict recorded-gap lower bound |
| --- | --- | --- | --- |
| 1 | \(0.000314379329034230502\) | \(0.000279274938096188480\) | \(3.5084390938\,10^{-5}\) |
| 2 | \(0.000314379334629583386\) | \(0.000279274943255797574\) | \(3.5084391373\,10^{-5}\) |

For each mediator-port prediction, the nominal numerical interval radius
is less than \(6.654\,10^{-24}\); the comparator radius is less than
\(8.677\,10^{-31}\). Both satisfy the fixed \(10^{-12}\) ceilings.
These are numerical radii, not recording precision or acquired-source
uncertainty. The separate bounds are
\[
 R_5(q,1)<5.903\,10^{-35},\qquad
 E_{\rm init}(q,1,\epsilon)<3.114\,10^{-42},\qquad
 \epsilon/\ell<1.001\,10^{-32}.
\]
The work, storage and quotient-entry conditions (19)-(22) hold independently
of these computed outputs. No state or source was reset or reacquired.

The compressed archive has 1,016,399 bytes and SHA-256
`01b443211665c027bcd5b86ace8a22a4e7ae051f34d172d5520644e6225ba16a`.
Its outcome retains all causal coefficients and error components. The
[read-only audit](../../tests/physics/test_sine_class_port_prediction_evidence.py)
reconstructs the consumed endpoints and budgets; retained coefficient
generation remains an explicit execution premise. Independent
[algebra controls](../../tests/physics/test_sine_class_port_prediction.py)
check grounded kernels, low-order full-edge substitution, hidden quadratic
recoupling and independent initialization without rerunning this selected
prediction. The [policy owner](../../src/tnfr/research/sine_class_collective_protocol.py)
centralizes the fixed source, new work/mean admission and strict comparison.

This is a successfully evaluated reduced prediction. It is not an
independent full-law forward observation. It separates the specified
grounded tangent alternative, with a leading effect already present in
linear hidden memory. Nonlinear necessity, minimal memory, new class
discrimination and physical identification do not follow. The
[independent forward protocol](SINE_CLASS_COLLECTIVE_FORWARD_PROTOCOL.md#sine-collective-forward-frozen-association)
owns the complete-law comparison and its source association; the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns its evaluation status and next action.
