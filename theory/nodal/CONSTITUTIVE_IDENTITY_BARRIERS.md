# Constitutive freedom in regional identity barriers and work

<a id="constitutive-identity-barriers"></a>

Matching the organized target, its storage, full tangent, quadratic field
and the previously declared registered-composition conditions does not fix
a C9 winding boundary's storage minimum or regional boundary power. The
admitted analytic family below gives exact counterexamples to both claims.
Its true first-boundary minima for winding zero and one vary affinely with
the constitutive parameter on one explicit interval. Regional power also
varies at one fixed complete state whose regional, contact and complete
storage values are unchanged.

These are static and conditional complete-law results. A minimum over a
phase-cell boundary is not the work consumed by an actual trajectory, an
activation law or a proof that a source reaches that boundary. The
[frozen sine-source loss](ORGANIZED_RECEIVER_ENCLOSURE.md#organized-receiver-enclosure-result)
is preserved in its own model. The [research program](../RESEARCH_PROGRAM.md#research-current-question)
remains the sole queue.

<a id="constitutive-barrier-model"></a>
## Actual complete law and the matched properties

Use the fixed equal-contact support of the
[receiver model](ORGANIZED_RECEIVER_MECHANISM.md#organized-receiver-source):
27 labelled nodes, three unit C9 regions, and contacts from oriented region
pairs \((0,1),(1,2),(2,0)\) with offsets zero and one, each of weight
one half. Every node has weighted degree four. All fine form and phase
coordinates, region membership and boundary registrations are retained.
Capacities are held at one and conductances remain fixed. There are no
inputs, resets, support events or operator invocations.

Write \(\alpha=2\pi/9\), \(\gamma=1/(1023\pi)\), and use the
structural clock \(\tau=(1023/1024)t\). The parameter \(\lambda\)
below is the constitutive parameter of the
[composition-compatible counterfamily](COMPOSITION_COMPATIBLE_CONSTITUTIVE_LAWS.md#composition-compatible-counterfamily),
not the slow-clock factor \(\gamma^2\). Restrict it to
\[
 |\lambda|\le E:=1/10000,
 \qquad P(d)=\cos d(1-\cos9d)^2,
 \qquad V_\lambda(d)=1-\cos d+\lambda P(d),
 \qquad j_\lambda(d)=V_\lambda'(d).                         \tag{1}
\]
With the actual weighted Laplacian \(L\), declare both rows:
\[
 S_{\lambda,i}(\theta)
    =\sum_jw_{ij}j_\lambda(\theta_j-\theta_i),\qquad
 x_\tau=\tfrac14[-Lx+\gamma S_\lambda(\theta)],\qquad
 \theta_\tau=\tfrac\gamma4Lx.                              \tag{2}
\]
The form row supplies pressure under unit capacity. It is not inferred
from an observed form response. In the original clock both rows acquire
the factor \(1023/1024\); structural units have no laboratory-unit bridge.

The current is odd, analytic and periodic. Its potential is even and
periodic. The bounds proved for the original family give
\[
 |P(d)|\le162(1-\cos d),\qquad |P(d)|\le4,
 \qquad V_\lambda(d)\ge(1-162E)(1-\cos d)\ge0.             \tag{3}
\]
The five-term Fourier expansion also gives
\[
 |P''(d)|\le328,
 \qquad |j_\lambda'(d)|\le1+328E.                         \tag{4}
\]
Thus the lifted vector field is globally Lipschitz. The fine law retains
common form and phase origins, graph relabellings preserving all marked
data, and simultaneous form/phase sign reversal. Both global means are
conserved; individual regional means need not be.

For the actual support, the complete storage and its loss are
\[
 H_\lambda=\tfrac12x^TLx+
       \sum_{\{i,j\}}w_{ij}V_\lambda(\theta_j-\theta_i),
 \qquad (H_\lambda)_\tau=-\tfrac14\|Lx\|^2.                \tag{5}
\]
This follows directly from \(\nabla_\theta H_\lambda=-S_\lambda\)
and both rows of (2). It re-admits the balance for the changed law rather
than transferring an evaluated sine trajectory.

At every gap \(k\alpha\), the factor \((1-\cos9d)^2\) has
a fourth-order zero. Hence
\[
 P=P'=P''=P'''=0\quad\text{at }d=k\alpha.                 \tag{6}
\]
The internal winding-one twist retains its storage, current, tangent and
quadratic field. The same is true of the equal-contact joined target
\(x_*=0\), \(\theta_{*,r,j}=\alpha(j-4)\): its actual fine gaps
are zero or \(\pm\alpha\) modulo a turn. Substituting (6) preserves
the fine-node current cancellation from the
[target proof](ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md#organized-network-capture-target).
This concerns the full 54-dimensional target field. It does not identify
the aggregate contact phase lag with a fine-edge gap or transfer the
unequal-contact target.

The family also retains its previously proved profile-preservation
properties under the matching and three-offset interfaces defined by its
[classification owner](COMPOSITION_COMPATIBLE_CONSTITUTIVE_LAWS.md#composition-compatible-classification).
That result does not assert profile preservation under every interface,
including the two-offset contacts used here. Nor do the matching target
data transfer source acquisition, nonlinear transport, capture radii or
event contracts. This remains a detached complete-law comparison under the
[NFR admission](../EMERGENT_ONTOLOGY.md#12-can-nfr-properties-justify-a-joint-law),
without a physical identity or a newly selected fundamental law.

<a id="constitutive-exact-boundary-minimum"></a>
## The true C9 first-boundary minima

Retain the principal-branch integers of a marked region at release and
continue its registered gaps until their first boundary contact. For
initial winding \(w\in\{0,1\}\), define the compact set
\[
 \mathcal B_w=\left\{\delta\in[-\pi,\pi]^9:
      \sum_j\delta_j=2\pi w,\quad \max_j|\delta_j|=\pi\right\},
 \qquad B_w(\lambda)=\min_{\delta\in\mathcal B_w}
                         \sum_jV_\lambda(\delta_j).       \tag{7}
\]
Every point of this set represents a consistent circular C9 phase
configuration with the retained winding sum. At least one gap is on the
boundary; winding at that boundary is not assigned. Minimization in (7)
retains all other gaps and permits simultaneous boundary contacts.

Put \(c=\cos(\pi/8)\). The exact result on the full interval (1) is
\[
 \boxed{B_0(\lambda)=B_1(\lambda)
       =2+8(1-c)+\lambda\beta,\qquad
       \beta=8c(1+c)^2-4.}                                \tag{8}
\]
In particular, the number \(9/4\) in the
[regional sine theorem](ORGANIZED_REGIONAL_RETENTION.md#organized-regional-winding-barrier)
remains a valid conservative bound in its stated sine domain; it was not
claimed to be the sharp value now computed at \(\lambda=0\).

To prove (8), select one boundary gap. Its potential is always
\[
 V_\lambda(\pi)=V_\lambda(-\pi)=2-4\lambda.               \tag{9}
\]
If the other eight have sum \(\pi\) or \(-\pi\), their equal-gap
candidate has cost
\[
 C_\lambda=8V_\lambda(\pi/8)
     =8(1-c)+8\lambda c(1+c)^2.                           \tag{10}
\]
The following elementary rational bounds will isolate every minimizer:
\[
 c>23/25,\qquad 1/4<\cos(5\pi/12)<1/3.                   \tag{11}
\]
For the first, \(\sqrt2>7/5\) gives
\(c^2=(2+\sqrt2)/4>17/20>(23/25)^2\). For the second, use
\(\cos(5\pi/12)=(\sqrt6-\sqrt2)/4>1/4\), since
\(\sqrt6>1+\sqrt2\), and
\(\cos(5\pi/12)=\sin(\pi/12)<\pi/12<1/3\).

By (3) and (11),
\[
 C_\lambda<16/25+32E.                                   \tag{12}
\]
If any of the remaining eight gaps has magnitude at least \(5\pi/12\),
its potential alone is at least \(2/3-4E\). The other seven potentials
are nonnegative. Thus their total exceeds the candidate by more than
\[
 2/3-4E-(16/25+32E)
       =2/75-36E=173/7500>0.                              \tag{13}
\]
Every minimizing remaining gap consequently lies in
\((-5\pi/12,5\pi/12)\). Throughout that interval,
\[
 V_\lambda''(d)=\cos d+\lambda P''(d)
       >1/4-328E=543/2500>0.                              \tag{14}
\]
Strict convexity and Jensen's inequality now prove that eight gaps with
sum \(\pm\pi\) have their unique minimum at eight identical gaps
\(\pm\pi/8\). This is a global minimum over the full constrained
face, because (13) excluded every omitted large-gap point.

For winding zero, either selected boundary sign leaves the opposite sum
\(\mp\pi\), so this proves (8). For winding one, a selected
\(+\pi\) leaves sum \(\pi\). The remaining possibility is a selected
\(-\pi\) with sum \(3\pi\). The global inequality
\(1-\cos d\ge2d^2/\pi^2\), followed by Cauchy--Schwarz and (3),
bounds that remaining cost below by
\[
 \frac2{\pi^2}\frac{(3\pi)^2}{8}-32E=9/4-32E.
\]
Its excess over the candidate in (12) is greater than
\[
 9/4-16/25-64E=4009/2500>0.                               \tag{15}
\]
It cannot minimize. This also excludes configurations with further
boundary gaps. The minimizing configurations are therefore one
\(+\pi\) and eight \(+\pi/8\) for winding one; for winding zero
they have one \(\pm\pi\) and eight \(\mp\pi/8\), with arbitrary
choice of the marked boundary edge. Substituting (9)--(10) gives (8).

<a id="constitutive-barrier-discriminator"></a>
## A fixed constitutive discriminator and the donor/receiver trade-off

The expression for \(\beta\) is increasing in \(c>0\). Equation (11)
gives \(23<\beta<28\). In particular the two already declared endpoint
laws satisfy
\[
 B_w(1/10000)-B_w(-1/10000)=\beta/5000>23/5000.             \tag{16}
\]
This is an exact unequal-minimum result, not a sampled comparison or a
comparison of arbitrary boundary-state energies. It refutes selection of
the regional first-boundary minimum by the matched target and registered
composition properties alone.

For a zero-form winding-one internal twist, the internal baseline remains
\(9V_\lambda(\alpha)=9(1-\cos\alpha)\). For a flat zero-form receiver
it remains zero. Their geometric storage differences to the first boundary
are therefore
\[
 R_D^*(\lambda)=B_1(\lambda)-9(1-\cos\alpha),\qquad
 R_R^*(\lambda)=B_0(\lambda).                             \tag{17}
\]
Both have the same slope \(\beta\), and
\(R_R^*-R_D^*=9(1-\cos\alpha)\) is unchanged. Raising this common
barrier increases both donor headroom and the receiver's necessary entry
storage. It does not selectively guarantee transmission with retention.

These baselines are ideal internal states, not source-release certificates.
For independently established bounds \(\|x_r\|\le X\) and
\(\|\theta_r-\vartheta\|\le b\), with
\(\vartheta=\alpha(j-4)\) or zero, (4) and the stationary internal
gradient do give the own-law static upper bound
\[
 H_{\lambda,r}\le U_{\lambda,r}(\vartheta)
                   +2X^2+2(1+328E)b^2.                   \tag{18}
\]
The cycle Laplacian bound \(L_I\preceq4I\) proves both terms. Establishing
that a changed-law trajectory reaches such a state is a separate obligation;
the sine source bounds, velocity defect, release time and capture margins
cannot supply it. A boundary minimum also omits positive form storage,
accumulated loss, contact work and the direction of passage.

<a id="constitutive-regional-work"></a>
## Regional work changes at identical complete state and storage

For each region retain the notation of the
[regional balance](ORGANIZED_REGIONAL_RETENTION.md#organized-regional-work-budget),
now using the actual changed current:
\[
 \ell_r=L_{I,r}x_r,\quad b_r=(L_Cx)_r,\quad
 s_{\lambda,r}=(S_{\lambda,I})_r,\quad
 c_{\lambda,r}=(S_{\lambda,C})_r.
\]
Direct differentiation of the internal storage gives
\[
 \begin{split}
 H_{\lambda,r}&=\tfrac12x_r^TL_{I,r}x_r
                       +\sum_{\text{internal }e}V_\lambda(\delta_e),\\
 D_r&=\tfrac14\|\ell_r\|^2,\\
 P_{\lambda,r}&=-\tfrac14\ell_r\cdot b_r
       +\tfrac\gamma4(\ell_r\cdot c_{\lambda,r}
                              -s_{\lambda,r}\cdot b_r),\\
 (H_{\lambda,r})_\tau&=-D_r+P_{\lambda,r}.
 \end{split}                                             \tag{19}
\]
The internal reciprocal terms cancel by (2); the contact terms retain
their state dependence. Including the contact storage recovers (5).
The first-exit proof applies with the actual \(B_w(\lambda)\) and the
actual integrated power in (19). Neither the sine power history nor a
larger static barrier determines the changed-law history.

There is an exact static witness on the same weighted support. Number
nodes \(9r+j\), and choose
\[
 x=e_9,\qquad \theta=(\pi/2)e_0.                          \tag{20}
\]
Subtracting the common form and phase means puts this witness on the
global mean-zero leaf without changing any following value. Every consumed
phase gap is zero or \(\pm\pi/2\). At these gaps the potentials agree
with sine, while
\[
 j_\lambda(\pi/2)=1-\lambda,
 \qquad S_\lambda=(1-\lambda)S_0.                        \tag{21}
\]
The exact internal storage, internal loss, contact storage and total loss
are independent of \(\lambda\):
\[
 (H_{\lambda,0},H_{\lambda,1},H_{\lambda,2})=(2,1,0),
 \quad (D_0,D_1,D_2)=(0,3/2,0),
 \quad H_{\lambda,C}=3,
 \quad H_\lambda=6,
 \quad D_{\rm all}=19/4.                                 \tag{22}
\]
For example, the sole nonzero form node has degree four, so its complete
form storage is two. Its Laplacian column has squared norm
\(4^2+2\cdot1^2+4\cdot(1/2)^2=19\). The phase node has two internal
and two units of contact conductance, giving internal phase storage two
and contact phase storage two. The remaining contact storage is form.

Nevertheless, evaluation of (19) gives
\[
 (P_{\lambda,0},P_{\lambda,1},P_{\lambda,2})
    =\left(-\frac\gamma8(1-\lambda),
           -1+\frac\gamma8(1-\lambda),0\right),
 \qquad
 P_\lambda-P_0=(\gamma\lambda/8,-\gamma\lambda/8,0).
                                                               \tag{23}
\]
For region zero, \(\ell_0=0\); its contact form gradient is \(-1/2\)
at nodes zero and eight, and its internal current is
\((1-\lambda)(-2,1,0,\ldots,0,1)\). Their dot product gives the
first entry. The other entries follow from the same retained contacts.
The regional storage rates sum to \(-5/2\), and contact storage has
rate \(-9/4\), yielding the unchanged total rate \(-19/4\).

Thus identical complete state, every regional and contact storage, and
complete instantaneous loss do not determine regional power across this
admitted constitutive class. The sign of the difference in (23) changes
with \(\lambda\); this does not claim that the regional power itself
changes sign. The witness is a supplied mathematical state, not a state
shown reachable from the original two-ramp source.

The branch-normal velocity at any identical complete state remains
\(\gamma b_e^TLx/4\), independently of \(\lambda\), because the phase
row is unchanged. The changed current instead changes the form row and
subsequent evolution. A different barrier or instantaneous work value
therefore does not by itself establish a different crossing outcome.

These counterexamples identify nonlinear constitutive information left
free by the matched target and composition requirements. They neither
select a preferred member of the family nor transfer the evaluated sine
source to it. Acquisition, donor maintenance, complete joined capture and
physical identification retain their separate obligations.

## Implementation and independent controls

The [boundary assessment](../../src/tnfr/physics/_constitutive_identity_barriers.py)
rebuilds the primitive constitutive law before evaluating (8), its strict
proof margins and the ideal donor target margin. Laws outside (1) but inside
the family's admitted interval remain valid laws; this narrower theorem
returns unavailable minima for them. It imports no release or source report.
The [independent controls](../../tests/physics/test_constitutive_identity_barriers.py)
evaluate the factored potential, actual equal-network target gaps, boundary
configurations and the inequalities consumed by the global proof.

The [shared regional observer](../../src/tnfr/physics/_organized_regional_retention.py)
accepts the entire already admitted family, \(|\lambda|\le1/324\),
using the [single potential/current owner](../../src/tnfr/physics/_composition_compatible_constitutive.py).
It retains the admitted law and both complete rows. Its default zero member
preserves the sine arithmetic; the separate source-retention assessment
remains sine-only. Signed form follows shared scalar materialization, while
primitive phase lifts retain their admitted represented values. A rounded
quarter turn is not the exact state in (20). The
[regional controls](../../tests/physics/test_organized_regional_retention.py)
therefore use a represented nonzero angle and independently evaluate its
actual potential/current. The shared exact-phasor controls check (21).
None of these controls evaluates a trajectory or regenerates frozen evidence.
