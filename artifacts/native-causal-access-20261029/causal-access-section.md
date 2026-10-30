
<a id="native-causal-access"></a>
## Causal access to the finite receiver-form contract

The FO threshold is retained as a conditional mathematical discriminator.
This audit establishes what its preparation and reference require, and why
the final EPI reading cannot authenticate those requirements. It admits an
explicit ideal mathematical control port, not an independently realized
actuator. The supplied access remains essential; neither another trajectory
nor a smaller numerical error would establish it.

### CA1. A sufficient access record and its symmetries

The following record is sufficient to apply FO. Minimality of its state
dimension or of its measurements is not claimed.

| Information | Role and independent admission required |
| --- | --- |
| Actual graph, unit measure, coefficients, labels and port association | Identify the one-bridge complete law and the donor/receiver coordinates. A drawing or an operator name does not supply them. |
| Initial relative form, phase geometry and capacity profile | Establish the prepared organization, capacity contrast and shared versus mismatched errors. Full initial form does not determine hidden phase. |
| Capacity policy of each trial | Establish whether the actual prepared profile remains held or evolves under the declared row. A nominal stored profile is not necessarily the actual profile. |
| Elapsed clock and readout map | Establish time conversion, gain, offset and error bounds in their native units, independently of the tested terminal contrast. |
| Trial association and input history | Establish which preparations and policies belong to A and B, and whether any intervention or controller adds state or changes another row. |

The complete field is equivariant under a common form translation and a
common phase rotation, and `ell^T*1=0`. These two origins can be removed
from an observation/preparation record by an explicitly declared choice of
representatives before applying FO's coordinate error budget. Each trial
has one common form origin and one common phase origin; separate shifts of
its two connected rings are not those symmetries. The mean-one full state
has dimension 29 and its neutral quotient dimension 27, but this is not a
proof that 27 is the minimal calibration or observable dimension for `y`.

The [response-identification owner](RELATIONAL_RESPONSE_IDENTIFICATION.md#prepared-coefficient-identification)
already retains hidden initialization, excitation, clock and sampling
conditions; the [derived-memory owner](../DERIVED_EPI_MEMORY.md) retains
information discarded by partial observation. Their software controls do
not constitute a physical preparation or an independent measurement bridge.
FO supplies separation once its premises hold, not a procedure for inferring
those premises from its own success.

### CA2. Preparation and exact-hold obstructions

Keep the actual connected support and native capacity row `nu_dot=-B*nu/2`.
Its kernel is exactly the common vector, hence
\[
\dot\nu=0\quad\Longleftrightarrow\quad \nu=b\mathbf1 .
\tag{CA1}
\]
The nonuniform JO/FO profile cannot be held merely by choosing a special
initial condition under this unchanged dynamic law. Conversely, starting
from `nu=b*1`, interventions that alter only form and phase while retaining
this capacity law cannot create `eta!=0`: the capacity subsystem is
autonomous and its uniform state remains invariant. The preparation needs
an independently supplied capacity contrast or a separately admitted
capacity-changing preparation law. This statement does not exclude the
registered operators that directly change capacity under their own contracts.

Nor can exact nonuniform holding be hidden in a finite passive extension of
this capacity subsystem. Suppose finitely many auxiliary nodes obey the
same reciprocal positive linear diffusion on a fixed closed connected unit
graph, with no sources or imposed boundary values. Every coordinate of
`nu(t)=exp(-B_ext*t/2)*nu(0)` is analytic in time. If the original nodes
remain constant on an open interval, analyticity makes those coordinates
constant for all time. CT convergence makes every node in that connected
component tend to one common value. The original constants would therefore
be equal, contradicting the FO profile.

This obstruction concerns the stated capacity-subsystem class. It does not
exclude approximate holding for finite time, active inputs, other laws,
changing support, an infinite environment or imposed boundary values. It
also does not authorize adding auxiliary edges to the full FO model:
ordinary new graph edges would generally change its other native rows.

### CA3. Two explicit ideal access models

A supplied selective transport port can be represented by
\[
\dot x=F_x(x,\theta,\nu),\qquad
\dot\theta=F_\theta(x,\theta,\nu),\qquad
\dot\nu=-sB\nu/2,\qquad \dot s=0,\quad s\in\{0,1\}.
\tag{CA2}
\]
Here `s` is initialized before the trial, not inferred or autonomously
selected by the nodal response. The other rows, graph and actual prepared
profile are unchanged. The complete augmented mathematical family is
specified: `s=0` gives held and `s=1` dynamic. It preserves total capacity
and positivity, with
\[
\dot V_\nu=-s\,\nu^TB\nu/2,\qquad
V_\nu=\tfrac12\|\nu-\bar\nu\mathbf1\|^2.
\]
This is a constitutive access premise. It does not derive a mechanism that
sets `s` or model the reaction/resources of a physical transport switch.
Removing graph edges changes `B,D` and resultants in the other rows; it
does not implement this selective port. The
[operator causal-role contract](../STRUCTURAL_OPERATORS.md#operator-causal-role)
also prevents identifying it with a Coupling invocation without a proof.

An alternative retains the native capacity row and declares an external
source port:
\[
\dot\nu=-B\nu/2+u,\qquad u=B\nu/2.
\tag{CA3}
\]
This ideal memoryless feedback gives exact holding and requires access to
the actual fine capacity contrasts and graph. A uniform offset does not
change the command, although the predictive state still needs its capacity
mean. For an oriented incidence matrix `G` with `B=GG^T`, native edge
flux `j=-G^T*nu/2` is cancelled by `j_c=G^T*nu/2`. Endpoint reactions
are equal and opposite; all regional transport cancels and `1^T*u=0`.
The source's variance contribution is `(nu-bar(nu)*1)^T*u=nu^T*B*nu/2`,
exactly compensating the native loss.

Zero net capacity input is not zero actuator resource cost. Native storage
has no explicit capacity term, so its unchanged loss identity cannot account
for or authenticate an actuator, reservoir or measurement device. CA3 is
complete as an ideal input-output mathematical prescription; no autonomous
physical controller or additional registered event is claimed.

If the controller instead stores a fixed reference, the declared augmented
law is
\[
\dot\rho=0,\quad u=B\rho/2,\qquad
\nu(t)=\rho+e^{-Bt/2}(\nu(0)-\rho).
\tag{CA4}
\]
Exact hold occurs iff `B*(nu(0)-rho)=0`. Matching capacity contrasts suffices;
on the same mean-one leaf it requires `rho=nu(0)`. Otherwise there is drift
throughout the trial, not simply a perturbed held initial profile. Total
capacity is conserved, but the original capacity hull need not be.
Markov contraction gives a sufficient positivity condition
`min(rho)>||nu(0)-rho||_infty` and
`||nu(t)-nu(0)||_infty<=2||nu(0)-rho||_infty`.
FO's initial-error budget does not by itself admit this changed control law.
No stored nominal reference may silently replace its actual-profile premise.

### CA4. A hidden phase mismatch reproduces the terminal contrast

There is a concrete information obstruction at the actual FO preparation,
not only at the blind equilibrium. Let `z_*` mean the nominal prepared
state (FO1), and define
\[
Y(\delta,t)=y_H(t;z_*+\delta e_{\theta_5}),\qquad
f(\delta)=Y(\delta,T),\quad T=2^{-36}.
\tag{CA5}
\]
Only the receiver port's initial phase changes. All initial form and
capacity coordinates, and `Y(delta,0)=0`, remain identical to nominal.
At `delta=0`, the native phase derivatives give
\[
\partial_{\theta_5}g_5=-1/\pi,\qquad
\partial_{\theta_5}g_6=\partial_{\theta_5}g_9=1/(2\pi),\qquad
\partial_\delta\partial_tY(0,0)=-5/(8\pi).
\tag{CA6}
\]
The two other receiver nodes contribute zero, and all receiver capacities
are one. These derivatives use the full one-bridge resultants.

FO2's holomorphic field and Picard bound apply to this new one-parameter
initial family: for `|delta|<=p=1/128` and `|t|<=R=1/8` its initial
distance from the lock is at most `r/4` and `|Y|<=2r=1/16`.
Write `Y=sum c_jk*delta^j*t^k`; Cauchy gives
`|c_jk|<=p^(-j)*R^(-k)/16`. For `b(delta)=partial_t Y(delta,0)`,
the geometric sums for `|delta|<=p/2` and `T<=R/2` yield
\[
|b'(\delta)-b'(0)|\le2^{16}|\delta|,\qquad
|\partial_\delta Y(\delta,T)-Tb'(\delta)|\le2^{12}T^2.
\tag{CA7}
\]
For the first inequality, the sum of `j u^(j-1)` over `j>=2` is at
most `8u` for `u<=1/2`. For the second, use
`sum_(j>=1) j u^(j-1)<=4` and `sum_(k>=2) v^k<=2v^2`.
Thus these are effective bounds from the retained analytic domain.

On `0<=delta<=rho_delta=2^-110` the error in `f'(delta)/T` is at
most `2^-94+2^-24<1/32`. Together with `3<pi<4`, this gives
\[
-T/4<f'(\delta)<-T/8.
\tag{CA8}
\]
The nominal contrast `Delta=y_D(T;z_*)-y_H(T;z_*)<=-S` satisfies
`|Delta|<2^-155+2^-158=9S<2^-154`, where `S=2^-158`:
use `K<1/36` for the cubic coefficient and the FO remainder.
Meanwhile `f(rho_delta)-f(0)<-2^-149<Delta<0`.
Continuity and strict monotonicity therefore give a unique
\[
\delta_*\in[2^{-120},2^{-115}),\qquad
y_H(T;z_*+\delta_*e_{\theta_5})=y_D(T;z_*).
\tag{CA9}
\]
The interval follows from `T*delta_*/8<|Delta|<T*delta_*/4`.
No trajectory, root search or numerical phase value was used.

A held/held pair with this unverified phase mismatch in trial A and the
nominal held reference B has exactly the same terminal readings and contrast
as the nominal dynamic/held pair. Both recover and have identical initial
form, capacity and `y(0)`. The mismatch is within FO's recovery envelope
but far outside its allowed preparation error. This is not a counterexample
to FO under its admitted premises. It proves that these initial-form and
terminal observations cannot authenticate the required phase preparation.
It does not identify whole trajectories: the initial rate changes, and
additional sufficiently precise observations could detect that difference.
It also does not make truly identical held/held preparations distinguishable.

### CA5. Access decision and remaining realization obligation

Retain FO on its original conditional domain. Its preparation and reference
are now explicitly separated from the predicted response. The available
mathematical access models are a supplied selective transport policy or a
supplied compensating-input law; neither is an emergent consequence of the
unforced capacity flow. Form/phase-only initialization does not create the
required capacity contrast; the finite passive extension above does not
provide exact nonuniform holding.
The terminal contrast cannot authenticate phase matching or actuator policy.

The current repository evidence therefore admits the mathematical consumer
but does not provide independent realization of its preparation, held
reference or readout/clock precision. A simulator can check a declared state
and selected law; those checks do not establish a physical actuator or sensor.
The original four prerequisite scopes are not silently extended to an
autonomous control/preparation mechanism. This audit also does not justify
replacing the observable merely because its sufficient error budget is small.

The next decision must specify the exact scope of a possible independent
realization-feasibility study and review its foundational admission before
any physical source search or comparison starts. It must distinguish asking
whether the required access exists from assuming that it does, and may
conclude with a concrete remaining obligation or rejection of this consumer.
The [program](../RESEARCH_PROGRAM.md#research-current-question) owns that
single decision. This is not itself physical readiness.

Analytical source, proof and scope reviews are retained in
`artifacts/native-causal-access-20261029/`. No controller was installed,
no scientific calculation script or trajectory was evaluated, and no
historical producer was replayed.
