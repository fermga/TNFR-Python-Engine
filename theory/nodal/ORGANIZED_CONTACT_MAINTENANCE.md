# Organized identity under continued external contact

<a id="organized-contact-maintenance"></a>

## Declared geometry and actual-source admission

This gate studies continued contact after the
[relative-origin comparison](ORGANIZED_EXTERNAL_REGISTRATION.md). It separates
stationary geometry, stability, source inclusion and loss of a marked identity.
No finite response, target label or carrier restriction supplies the others.

| Item | Fixed scope before source assessment |
| --- | --- |
| Complete law | Keep the 54-node unit support C9 square W6, all 108 fine form/phase coordinates, held unit capacities, x_tau=(-Lx+gamma*S(theta))/4 and theta_tau=gamma*Lx/4, gamma=1/(1023*pi), tau=(1023/1024)*t. No input or event after attachment |
| Stationary geometry | Classify the aligned outer-origin target and the antipodal-donor stationary control on the full fine state, including both conserved means. Retain the twelve-coordinate carrier separately |
| Organized carrier question | Determine whether any carrier stationary state can retain the closed acute outer winding-one chart, and what any obstruction implies for continued carrier motion. Acute-chart exit, winding loss and full-fine source loss are separate claims |
| Full-fine boundary | Retain all six internal winding-one cycles and all nine old outer cycles as marked identities. New wheel triangles constrain reuse of an all-edge acute chart; they do not alone exclude a stable state with some nonacute edges |
| Actual control source | Rebuild all twelve original allowances in [0,1e-10] and the whole sequential observation history. For a separately declared control choose the outer-zero host and donor primitive relative origin beta=0. Keep every fine residual, all 96 relative preparation directions and exact zero joined means; no reset, target substitution or extra waiting |
| Control acceptance | Prove the actual control lies in the joined acute target cell with total initial excess above 6*V9 below 1/100 and strictly below every cell boundary. If admitted, establish all-future marked winding retention and asymptotic convergence; no finite settling deadline |
| Work | Count actual new attachment work and all earlier supplied preparation and observation costs separately. A sharper present-storage bound must not erase historical costs or treat dissipated storage as an event reserve |
| Other registrations and outer-one source | Keep the original family and every beta available as parameters. The beta=0 control does not imply capture of another origin or the outer-one source. Carrier statements do not replace full-fine source inclusion; report unresolved cases explicitly |
| Numerical boundary | Exact rational primitive admission and existing dyadic128 outward arithmetic. No trajectory, source-producer replay, root search, relative-origin scan, precision tuning or new physical-data evaluation |

The geometric proof may classify or obstruct a proposed retained organization.
The actual-source control has the fixed acceptance criterion above before its
assessment. Conditional maintenance and a scoped obstruction both count as
informative outcomes. Neither identifies a fundamental physical property;
the plan retains property-first physical comparison and its independent bridge.

The declaration above was committed as `4c4addf1` before the actual-source
assessment. The following proofs classify the fixed support and complete law;
they evaluate no source trajectory, trial registration or equilibrium search.

## 1. Complete state, identities and stationary equations

Use node order `(c,j)`, with five host constituents `c=0,...,4`, external
constituent `c=5`, and internal slots `j=0,...,8`. The support is the product
of C9 and the wheel W6: six internal cycles, five outer ring edges per slot,
and five spokes per slot. Its 144 unit edges are those already owned by the
[external-contact geometry](ORGANIZED_EXTERNAL_INTERACTION.md#1-complete-state-support-and-balances).
No edge, capacity, phase row or clock is changed here. Write

\[
 \alpha=2\pi/9,\qquad \psi_j=\alpha(j-4),\qquad
 \gamma=1/(1023\pi),\qquad
 S_i(\theta)=\sum_{b\sim i}\sin(\theta_b-\theta_i).
\]

With the unnormalized Laplacian L, the complete law and storage are

\[
 \begin{split}
 x'&=(-Lx+\gamma S(\theta))/4,&
 \theta'&=\gamma Lx/4,\\
 H(x,\theta)&=\tfrac12x^TLx+
       \sum_{\{i,b\}}[1-\cos(\theta_b-\theta_i)],&
 H'&=-\tfrac14\|Lx\|_2^2.
 \end{split}\tag{1}
\]

Prime denotes the declared tau clock. Both global form and lifted phase
means are conserved. Connectedness implies that a stationary state has
constant form and satisfies `S(theta)=0`. On the declared zero-form-mean
leaf its form is zero. A stationary relative phase cannot acquire a
nonzero common angular velocity: its conserved lifted mean forbids it.

The six marked internal winding observations and nine old outer winding
observations use oriented principal phase gaps on their fixed cycles.
An acute chart requires its declared gaps to lie in `(-pi/2,pi/2)`;
crossing its boundary does not by itself change a winding number. Changing
a principal winding requires reaching a branch cut at an absolute gap pi.
New spoke triangles are part of the support but carry no newly prescribed
identity. The complete state retains every fine deviation and both means.

## 2. Two full-state stationary controls and their complete tangents

### Aligned constituent origins

For a common origin Phi and common form m, the aligned state

\[
 x^*_{c,j}=m,\qquad \theta^*_{c,j}=\psi_j+\Phi
 \tag{2}
\]

is stationary: opposite internal currents cancel and every wheel current
vanishes. It retains six internal windings one and nine outer windings zero.
The global means choose m and Phi; these are not independently damped modes.

Let `L9` denote the C9 Laplacian and K the W6 Laplacian. In the chosen node
order, the unweighted form Laplacian and phase-storage Hessian at (2) are

\[
 L=I_6\otimes L_9+K\otimes I_9,\qquad
 M_+=\cos\alpha\,I_6\otimes L_9+K\otimes I_9.
 \tag{3}
\]

Both are positive on the common-mode complement. This is a full-fine
statement, including perturbations that leave the twelve-coordinate carrier.

### Antipodal external constituent

Keep all five host origins at Phi and the external origin at `Phi+pi`.
The spoke currents still vanish, so this state is also stationary. Its
internal and old outer windings agree with (2), but its spoke cosines are
negative. Write `K_minus` for the wheel Laplacian with ring weights one
and spoke weights minus one. Its phase Hessian is

\[
 M_-=\cos\alpha\,I_6\otimes L_9+K_-\otimes I_9.
 \tag{4}
\]

The stationary label and the same marked windings do not imply the same
stability. If a particular conserved lifted mean is required, subtract the
appropriate common phase from all 54 coordinates; this changes neither
current nor tangent.

### Exact joint mode classification

The internal Fourier values are

\[
 \lambda_9(k)=2-2\cos(2\pi k/9),\qquad k=0,\ldots,8.
\]

The wheel has a common mode, a host-versus-external mode proportional to
`(1,1,1,1,1,-5)`, and four ring Fourier modes with external entry zero.
Their simultaneous eigenvalues are:

| Wheel mode | Form value mu | Aligned phase value kappa | Antipodal phase value kappa |
| --- | --- | --- | --- |
| Common | 0 | 0 | 0 |
| Host versus external | 6 | 6 | -6 |
| Ring Fourier r=1,...,4 | 3-2*cos(2*pi*r/5) | 3-2*cos(2*pi*r/5) | 1-2*cos(2*pi*r/5) |

Each pair `r,5-r` contributes two real ring directions. For any product
mode, define `lambda=lambda9(k)+mu` and
`h=cos(alpha)*lambda9(k)+kappa`. The complete two-row tangent is

\[
 \begin{pmatrix}\delta x\\\delta\theta\end{pmatrix}'
 =\frac14
 \begin{pmatrix}-\lambda&-\gamma h\\\gamma\lambda&0\end{pmatrix}
 \begin{pmatrix}\delta x\\\delta\theta\end{pmatrix},
 \qquad
 s^2+\frac\lambda4s+\frac{\gamma^2\lambda h}{16}=0.
 \tag{5}
\]

For the aligned target, all 53 noncommon product modes have positive
lambda and h. Their 106 temporal eigenvalues have negative real parts.
The remaining two neutral directions are exactly the conserved form and
phase means. Thus (2) is locally asymptotically stable on each fixed-mean
leaf; the acute storage argument below supplies a nonlinear version.

For the antipodal target, all nine host-versus-external product modes have
`h=cos(alpha)*lambda9(k)-6<0`, since `lambda9(k)<=4` and `cos(alpha)<1`.
Each has one positive and one negative temporal root. The four ring modes
have positive antipodal phase values `(3-sqrt(5))/2` and
`(3+sqrt(5))/2`, each twice. The other 44 relative product modes therefore
contribute 88 stable directions. Altogether there are **nine unstable,
97 stable and two neutral directions**. This full-state stationary control
is a saddle, not a retained-source capture certificate.

## 3. Triangles obstruct reuse of a wholly acute winding-one cell

At any fixed slot j, let d_c be the principal host ring gap from c to c+1
and sigma_c the principal spoke gap from host c to the external node.
The oriented triangle periods are integers

\[
 n_c=\frac{d_c+\sigma_{c+1}-\sigma_c}{2\pi},\qquad
 \sum_{c=0}^4n_c=\frac1{2\pi}\sum_{c=0}^4d_c.
 \tag{6}
\]

These are topological identities for the actual fine state, with principal
branches away from their cuts. Outer winding one makes their sum one, so
some triangle has a nonzero period. Its three absolute principal gaps
cannot all be strictly below `2*pi/3`. In particular a wholly acute joined
chart cannot retain outer winding one. If all five old ring gaps remain
acute, that triangle additionally forces at least one spoke gap to have
absolute value at least `3*pi/4`: its two spoke magnitudes must sum to at
least `2*pi-|d_c|`, which exceeds `3*pi/2` in the strict chart.

This holds independently at each of the nine slots and requires no carrier
assumption. It explains why the earlier positive-edge-cosine storage trap
cannot simply be reused on the enlarged support. It does **not** establish
that the complete phase Hessian is indefinite: negative edge weights can
be compensated by the rest of a graph. Nor does it prohibit equilibria
with some nonacute edges or prove loss of an actual source winding.

## 4. No stationary acute outer winding on the exact carrier

The invariant carrier retains all twelve coordinates,

\[
 x_{c,j}=X_c,\qquad \theta_{c,j}=\psi_j+\Phi_c,
 \qquad
 X'=(-KX+\gamma S_{W_6}(\Phi))/4,\quad
 \Phi'=\gamma KX/4.
 \tag{7}
\]

It is a justified restriction of (1), not a replacement for the independently
perturbed source. Its exact storage and loss are

\[
 H_{12}=6V_9+\tfrac92X^TKX+
       9\sum_{\{c,b\}\in W_6}[1-\cos(\Phi_b-\Phi_c)],
 \qquad H_{12}'=-\tfrac94\|KX\|_2^2,
 \tag{8}
\]

where `V9=9*(1-cos(alpha))`.

### Closed acute-cell obstruction

Suppose a carrier stationary state has outer winding one and five directed
principal ring gaps `d_j` in `[-pi/2,pi/2]`. Their sum is `2*pi`; consequently
none can be negative, since the other four have sum at most `2*pi`.
Set the hub phase to zero, and denote the ring phases modulo `2*pi` by
phi_j. The stationary ring equations are

\[
 \sin\phi_j=\sin d_j-\sin d_{j-1}.
 \tag{9}
\]

Choose a maximal gap `d_m`. Since sine is increasing on `[0,pi/2]`, its
first endpoint has nonnegative sine and its second endpoint nonpositive
sine. The positive oriented gap is at most `pi/2`, so these endpoints
straddle the hub's antipodal phase. Choose representatives

\[
 \phi_m=\pi-u,\qquad \phi_{m+1}=\pi+v,
 \qquad u,v\ge0,\quad u+v=d_m.
\]

Adding the two adjacent relations in (9) gives

\[
 \sin d_{m-1}+\sin d_{m+1}
 =2\sin d_m-\sin u-\sin v
 \le\sin d_m\le1,
 \tag{10}
\]

where `sin(u)+sin(v)>=sin(u+v)` for these angles. For two angles a,b in
`[0,pi/2]`, `sin(a)+sin(b)<=1` implies `a+b<=pi/2`. For completeness,
at fixed sum `s>=pi/2` the concave sine sum has minimum
`1+sin(s-pi/2)` at an endpoint; equality at sum `pi/2` requires one angle
zero and the other `pi/2`. Equation (10) therefore bounds the sum of the
two gaps adjacent to d_m by `pi/2`.

The maximal gap and the other two gaps are each at most `pi/2`. Since the
five gaps sum to `2*pi`, every bound must be an equality. Thus four gaps
are `pi/2` and one is zero. At the two coincident endpoints of that zero
gap, (9) would require the same phase to have sine minus one and plus one,
respectively. This is impossible. Hence **no carrier stationary state lies
in the closed acute outer-winding-one cell**. Reflection of all phases
proves the corresponding winding-minus-one result.

The argument uses the actual unit ring and spoke weights. It neither
changes the interface nor identifies a constitutive coefficient from a
desired retained organization.

### Dynamical consequence and its precise boundary

Every carrier trajectory starting in this cell leaves its closure after
some finite time. To prove this, suppose instead it remained there for all
future times. Conserved mean form and (8) bound X. The phase torus is
compact, and the closed acute winding cell stays away from principal
branch cuts, so the trajectory has a nonempty compact omega-limit set.
Storage is nonincreasing. Its largest invariant zero-loss subset requires
`KX=0`; the phase row then vanishes, and invariance of the form row requires
`S_W6(Phi)=0`. The closed-cell obstruction excludes that set, contradicting
indefinite retention. The same reasoning applies for every supplied hub
origin and every finite initial carrier form.

This supplies no uniform exit deadline. Leaving an acute cell at absolute
gap `pi/2` is not changing winding at a branch cut of absolute gap pi.
Nor does carrier exit establish exit of every actual fine-state source:
off the carrier, internal C9 torques enter the stationary equations and
cannot be discarded from (9). Broader nonacute winding-one equilibria and
their possible attraction require their own classification.

## 5. A full-fine obstruction to stable retention near the carrier

The previous section's stationary equations are restricted to the carrier.
A different argument applies to the entire fine phase Hessian, even when
internal currents do not cancel. Let

\[
 M(\theta)=B\,\operatorname{diag}(\cos\delta_e)B^T,
 \qquad
 D(\theta)=\sum_{54\ \mathrm{internal\ edges}}
                  [1-\cos(\delta_e-\alpha)].
 \tag{11}
\]

Here B is any consistently oriented incidence matrix. Orient each internal
cycle in increasing slot order; its closing gap retains the declared
`+2*pi` winding representative. The cosine in D makes any equivalent
representative harmless. D measures departure of the internal edge gaps
from the carrier twist. It is not a distance between complete states and
does not replace their form or phase coordinates.

Suppose all nine old outer cycles have strictly acute winding one. With
an arbitrary common reference beta, put

\[
 \chi_{c,j}=\theta_{c,j}-\psi_j-\beta,
 \qquad v_{c,j}=\sin\chi_{c,j},\quad
 w_{c,j}=\cos\chi_{c,j}.
\]

For every edge, the squared differences of these two vectors sum to
`2*(1-cos(chi_b-chi_a))`. Their two phase quadratic forms therefore obey

\[
 v^TMv+w^TMw
 =2\sum_e\cos\delta_e\,[1-\cos(\chi_b-\chi_a)].
 \tag{12}
\]

For same-slot wheel edges, the demodulated difference is the actual gap.
For internal edges it is the actual gap minus alpha, modulo `2*pi`.
Since `cos(delta_e)<=1` and every bracket is nonnegative, the internal
contribution to the sum on the right before its factor two is at most D.
This inequality does not assume positive internal cosines.

To bound a wheel, let its five acute ring gaps be d_i. Winding one makes
them positive and their sum `2*pi`. Put `c5=cos(2*pi/5)` and
`C=sum_i cos(d_i)`. Concavity of cosine and the quadratic-mean inequality
give

\[
 0<C\le5c_5<5/2,\qquad
 \sum_i[\cos d_i-\cos^2d_i]
 \le C-C^2/5\le5c_5(1-c_5).
 \tag{13}
\]

The hub's antipodal point is bracketed by two consecutive ring vertices.
Their hub-relative phases are `pi-a` and `pi+b`, with nonnegative a,b
and `a+b<pi/2`. The two corresponding spoke contributions
`cos(sigma)-cos(sigma)^2` sum to less than minus two, since
`cos(a)+cos(b)>1` and `cos(a)^2+cos(b)^2>1`. Each of the remaining three
spoke contributions is at most `1/4`. Thus every wheel contributes less
than minus d, where

\[
 d=\frac54-5c_5(1-c_5)
   =\frac{5(7-3\sqrt5)}8>0.
 \tag{14}
\]

Combining the nine wheels and the internal edges yields the strict bound

\[
 v^TMv+w^TMw<2[D(\theta)-9d].
 \tag{15}
\]

Consequently **D<=9d excludes a positive-semidefinite phase Hessian** in
this acute outer-winding-one domain. At least one of v,w has negative
quadratic form. Subtracting its global mean preserves that form because
`M*1=0`, so the negative direction belongs to the conserved-phase-mean
tangent space. All fine coordinates are included; no carrier substitution
or assumed cancellation of internal torques enters this test.

At a stationary state, negative phase curvature supplies a positive growth
root for the complete two-row dynamics. On the zero-mean subspace L is
invertible. Eliminating the form perturbation gives

\[
 L^{-1}\delta\theta''+\tfrac14\delta\theta'
       +\frac{\gamma^2}{16}M\delta\theta=0.
 \tag{16}
\]

The symmetric matrix polynomial
`P(s)=s^2*L^{-1}+s*I/4+gamma^2*M/16` has a negative eigenvalue at s=0
and is positive definite for sufficiently large positive s. Continuity
therefore supplies a positive s with singular P(s), and its kernel gives
an exponentially growing complete tangent mode. Such a stationary state
is linearly unstable on the fixed-mean leaf.

This is a local stationary obstruction. A negative Hessian at a moving
source is not a positive Lyapunov exponent for that trajectory, nor proof
of future identity loss. A stable stationary retained outer-winding-one
organization must either have **D>9d**, involving changed internal gaps,
or lie outside the old strictly acute outer chart. Existence or acquisition
of either alternative remains unproved. Instability on the full relative
state space also does not prove that the 96-dimensional prepared family
intersects every growing direction.

For the actual outer-one source at attachment, let Z be its freshly rebuilt
full-phase error about the carrier and m_o the continuous per-node phase
drift bound carried through the entire observation word. The commands are
constant within each child and their final sum is zero; no commanded
internal edge difference is retained. Host internal edge deviations from
alpha are bounded by `2*(Z+m_o)`, while the external constituent's deviations
are bounded by `2*q`, with `q=1e-70`. Thus all original source errors and
all relative origins satisfy the initial bound

\[
 D(\theta(0))\le90(Z+m_o)^2+18q^2.
 \tag{17}
\]

This checks that the actual initial outer-one family lies inside the
negative-curvature domain when the right side is below 9d. It is neither
a stationary-state substitution nor a future bound on D along its motion.

<a id="full-chamber-phase-curvature-tests"></a>
### 5.1. Winding triangles and full-domain necessary curvature tests

The analytical mechanism admission declared at `0aec0fe4` asks whether
phase-Hessian witnesses can cover the complete source-compatible marked
class. The following argument removes the acute outer-gap hypothesis from
a necessary defect bound and excludes every common internal profile. It
does not exclude all nonseparable stationary states. All results use (1),
the original 144 unit edges and the complete phase Hessian; no source,
trajectory, optimizer or numerical certificate is evaluated.

Put `g(a)=cos(a)-cos(a)^2`. For one wheel, define

\[
 F_W=\sum_{e\in W_6}g(\delta_e).
 \tag{17a}
\]

**Winding-triangle bound.** If the five rim edges avoid their principal cuts
and have winding one, then `F_W<=-1/4`, regardless of the hub origin or any
acute-gap assumption.

First avoid spoke cuts as well. Orient each spoke triangle consistently
with its rim edge. Their principal windings sum to the rim winding, because
oppositely oriented spokes cancel. Each triangle winding is `-1`, `0` or
`1`, so at least one is `1`. Its three principal increments `a,b,c` are
strictly between zero and pi and sum to `2*pi`: a nonpositive increment
could not complete that sum with the other two strictly below pi.

For such a triangle, `g(a)+g(b)+g(c)<=-2`. Here is a direct proof that also
admits boundary limits. Relabel so `c=max(a,b,c)>=2*pi/3`, and put
`p=cos(c/2)` and `z=cos((a-b)/2)`. Then `0<=p<=1/2` and `p<=z<=1`.
The sum is

\[
 2(1-2p^2)z^2-2pz-4(1-p^2)^2.
 \tag{17b}
\]

It is convex in z, so its maximum is at an endpoint. At `z=p` it equals
`-2-8(p^2-1/2)^2`; at `z=1` it equals
`-2+2p[2p(1-p^2)-1]`. Both are at most `-2`, since `2p(1-p^2)<=1`.
Each of the seven remaining wheel edges has `g<=1/4`. Thus
`F_W<=-2+7/4=-1/4`. If a spoke is at a cut, perturb only the hub origin
to avoid all spoke cuts and pass to the limit. The marked rim winding is
unchanged and the cosine expression is continuous.

For the complete fine state, orient internal gaps in increasing slot order
and define

\[
 \begin{split}
 W&=\sum_{90\ \mathrm{wheel\ edges}}g(\delta_e),&
 A&=\sum_{54\ \mathrm{internal\ edges}}\cos\delta_e,\\
 B_2&=\sum_{54\ \mathrm{internal\ edges}}\cos^2\delta_e,&
 C_2&=\sum_{54\ \mathrm{internal\ edges}}\cos\delta_e\sin\delta_e.
 \end{split}
 \tag{17c}
\]

For `m=0,...,8`, take the real vectors
`v_m(c,j)=sin(theta_(c,j)-m*alpha*j)` and
`w_m(c,j)=cos(theta_(c,j)-m*alpha*j)`, with `alpha=2*pi/9`.
The phase reference is periodic on C9, including its closing edge.
The squared-difference identity used in (12) gives exactly

\[
 \frac{v_m^TMv_m+w_m^TMw_m}{2}
 =W+A-B_2\cos(m\alpha)-C_2\sin(m\alpha).
 \tag{17d}
\]

There is no assumption of stationarity or a separable phase profile here.
For nine winding-one rims, (17a) implies `W<=-9/4`. Positive semidefiniteness
of M consequently requires

\[
 A-\max_{m=0,\ldots,8}
       [B_2\cos(m\alpha)+C_2\sin(m\alpha)]\ \ge\ \frac94.
 \tag{17e}
\]

For `m=1`, the internal contribution is
`sum_internal cos(delta_e)[1-cos(delta_e-alpha)]<=D`, with D from (11).
Thus **`D<9/4` excludes a positive-semidefinite full phase Hessian**, even
for nonacute outer gaps and reflection-breaking states. The non-strict
boundary `D=9/4` is not excluded by this bound. At a stationary point the
negative direction gives complete-law instability by (16). At a moving
state it remains an instantaneous curvature statement, not a future
identity-loss verdict. The original acute-domain theorem and frozen source
assessment retain their own stated bounds.

### 5.2. No stable stationary contact with a common internal phase profile

Suppose the full phase state has the form
`theta_(c,j)=psi_j+Phi_c` modulo a turn for all six children, and its rim
origins have winding one. The shared profile psi need not be uniform,
acute or a stationary internal cycle. Choose vectors constant within each
child: `v_(c,j)=sin(Phi_c)` and `w_(c,j)=cos(Phi_c)`. Internal edges
contribute exactly zero. The nine wheels have identical gaps, hence

\[
 v^TMv+w^TMw=18F_W\ \le\ -\frac92.
 \tag{17f}
\]

At least one vector has negative quadratic form. Subtracting its global
mean preserves that form and places it on the full relative phase leaf.
Consequently **no stationary state with a shared internal phase profile
and retained outer winding one can be stable under (1)**. This excludes
arbitrary common-profile distortions and arbitrary rim origins, without a
reflection or acute-gap premise. It is broader than the helical carrier
and does not require the constituent internal torques to vanish separately.

A possible stable stationary target must therefore be genuinely
nonseparable: some relative constituent phase difference must depend on
the internal slot. Such dependence is a necessary property, not a supplied
new law or evidence that a stable target exists. It does not establish
formation, capture, a bound state or a physical constituent.

<a id="phase-pair-witness-incompleteness"></a>
### 5.3. An exact blind point of the nine fixed phase-pair sums

The necessary tests (17d) do not form a complete Hessian certificate. An
exact counterexample remains inside the original complete product marked
class, including its mixed cycles and all true cut margins.

Set `psi_0=0` and give its nine successive internal gaps the values

\[
 (\pi/3,\pi/3,\pi/3,\pi/3,\pi/3,\pi/3,\pi/3,-\pi/6,-\pi/6).
 \tag{17g}
\]

Their sum is `2*pi`; all lie strictly inside the acute interval. For hosts
put `Phi_c=(c-2)*pi/3`, `c=0,...,4`, and put `Phi_D=0` for the donor.
Use `theta_(c,j)=psi_j+Phi_c` and zero forms, with a common phase translation
if needed to retain the declared lifted mean. The five rim gaps are four
copies of `pi/3` and the closing `2*pi/3`. The internal closing branch is
`-1`, as is the rim closing branch; all other marked branches are zero.
Every mixed marked square telescopes to zero. Thus this is in the
[original complete class](ORGANIZED_CONTACT_STATIONARY_SEARCH.md#marked-cycle-information-and-connected-chambers),
not merely a configuration sharing fifteen winding labels.

Per wheel the rim contributes `1/4` to `F_W`, and the spokes contribute
`-1`, giving `W=-27/4` over all nine wheels. Per internal child the sums in
(17c) are

\[
 A_0=\frac72+\sqrt3,\qquad B_0=\frac{13}4,\qquad
 C_0=\frac{5\sqrt3}4,\qquad
 \sqrt{B_0^2+C_0^2}=\frac{\sqrt{61}}2.
 \tag{17h}
\]

For every one of the nine m values, the half pair sum in (17d) is at least
`-27/4+6*(A_0-sqrt(61)/2)>1/4`. The strict inequality follows exactly from
`sqrt(3)>5/3` and `sqrt(61)<8`, which give
`A_0-sqrt(61)/2>7/6`. No floating-point trigonometric evaluation is needed.

Nevertheless let u equal one on host child zero and zero elsewhere, then
subtract its global mean. Internal edges again contribute zero. At that
host the two rim cosines are `1/2` and `-1/2`, and the spoke cosine is
`-1/2`, so `u^T M u=-9/2`. The full Hessian is not positive semidefinite.
The state is explicitly nonstationary: at slot `j=1` the equal adjacent
internal sine currents cancel, while `S_(0,1)=sqrt(3)/2`.

This example defeats the **nine pair-sum tests**, not every individual sine
or cosine vector, their linear combinations or the full quadratic form on
their span. A positive pair sum can hide a negative individual direction.
It also does not refute a stronger theorem restricted to stationary states.
The new common-profile test (17f) correctly rejects this same configuration.

The pair tests and common-profile obstruction alone do not admit a
chamber-wide stationary certificate. The following stationarity arguments
further restrict the nonseparable sector without discarding full relative
curvature or replacing it by more evaluations of these nine sums.

<a id="stationary-resultant-and-slot-comparison"></a>
### 5.4. Every stationary PSD candidate has a nonzero aligned resultant

For a stationary phase state, define

\[
 z_i=e^{i\theta_i},\qquad R_i=\sum_{b\sim i}z_b,\qquad
 r_i=\Re(\overline z_iR_i)=\sum_{b\sim i}\cos(\theta_b-\theta_i).
 \tag{17i}
\]

Every node has odd degree, five on a host and seven on the donor. For odd
d, the maximum of `sum a_k^2` on `[-1,1]^d` with `sum a_k=0` is `d-1`.
Indeed a vertex of this polytope has at most one nonsaturated coordinate.
The other `d-1` coordinates sum to an even integer, so the remaining
coordinate must be zero; the saturated coordinates balance. Convexity
puts a maximum at a vertex. Apply this to the stationary sine currents:
`sum sin(delta_ib)=0` implies `sum cos(delta_ib)^2>=1` at every node.

Suppose the full phase Hessian M is positive semidefinite. Edgewise
`cos(delta)<=1` gives `M<=L`. The C9 Laplacian is bounded by `4I`, while
W6 is a subgraph of K6 and its Laplacian is bounded by `6I`. Thus
`0<=M<=10I`, and spectral calculus gives `M^2<=10M`. Since `M_ii=r_i`,

\[
 r_i^2+1\ \le\ (M^2)_{ii}\ \le\ 10r_i,
 \qquad r_i\ge5-2\sqrt6>\frac1{10}.
 \tag{17j}
\]

Stationarity also makes `conj(z_i)*R_i` real. Consequently
`R_i=r_i*z_i` with the strict bound (17j), and every such candidate obeys
the regular fixed-point equation `z_i=R_i/|R_i|`. This excludes zero
resultants and antipodal alignment, without a separable, acute or reflection
premise. It is a necessary condition, not a proof of existence, uniqueness
or full Hessian positivity. The angular fixed-point derivative still has
signed weights `cos(delta_ib)/r_i`; their row sum is one, but negative
entries prevent inference of a maximum principle or contraction.

Nor can unrestricted stress positivity replace tangent-Hessian positivity.
With adjacency matrix A and `T=diag(r)-A`, stationarity gives `T*z=0`,
but `1^T*T*1=-2*V(theta)`. Requiring T to be PSD would force zero potential
and synchronization. The stable aligned-twist control (2) already has
`T*1=(2*cos(alpha)-2)*1<0`, although its M is positive on the full relative
leaf by (3). This rejects that stronger certificate, not the control.

### 5.5. Stationary comparison with a cyclically shifted profile

Use lifts in the original complete product marked chamber: internal and
rim closing branches are `-1`, and all other marked branches are zero.
Let `kappa_(c,j)` be the forward principal internal gap, and define

\[
 \theta^+_{c,j}=\begin{cases}
 \theta_{c,j+1}-\alpha,&j<8,\\
 \theta_{c,0}+2\pi-\alpha,&j=8,
 \end{cases}
 \qquad d_{c,j}=\theta^+_{c,j}-\theta_{c,j}
              =\kappa_{c,j}-\alpha.
 \tag{17k}
\]

This is the slot automorphism followed by a common phase rotation, with
the compatible lift retained. It preserves the complete law, every child
lifted mean and the marked branch vector. Internal and rim principal gaps
are cyclically shifted. Thus a stationary theta gives a stationary theta+
with a permutation-conjugate Hessian M+. The straight lifted segment
`theta+t*d`, `0<=t<=1`, stays in the original marked chamber: each marked
gap interpolates between two values strictly inside the same cut bounds.
This segment is a comparison of configurations, not a solution trajectory.

For each oriented edge, let a and b be its gaps at theta and theta+ using
the same lift/branch convention at both endpoints. Set
`h=(b-a)/2` and `mu=(a+b)/2`. Include all 144 edges. On unmarked spokes
retain these compatible lifted gaps; independently wrapping the two
endpoints would change the secant. On every edge
`2h=d_right-d_left=kappa_right-kappa_left`, hence `|h|<pi`.
Stationarity at both endpoints and the edge quadratic form give

\[
 \begin{split}
 0&=\sum_e(\sin b_e-\sin a_e)(b_e-a_e)
     =4\sum_e\cos\mu_e\,h_e\sin h_e,\\
 \frac{d^TMd+d^TM^+d}{2}
   &=4\sum_e\cos\mu_e\,h_e^2\cos h_e\ \ge\ 0
       \quad\text{if }M\succeq0.
 \end{split}
 \tag{17l}
\]

Subtracting these identities gives the necessary inequality

\[
 \sum_e\cos\mu_e\,h_e(\sin h_e-h_e\cos h_e)\ \le\ 0.
 \tag{17m}
\]

The factor `h*(sin h-h*cos h)` is strictly positive for `0<|h|<pi`:
it is even, and the derivative of `sin h-h*cos h` is `h*sin h>0` for
positive h in that interval. Therefore a PSD stationary candidate must
either have a changing edge with `cos(mu)<0`, or satisfy `cos(mu)=0`
on every changing edge. In the latter case (17l) has zero quadratic form
at both PSD endpoints, so `Md=0`. It also gives `sin a=sin b` separately
on every edge. The next argument excludes this entire flat alternative,
including Hessians with additional zero eigenvalues.

### 5.6. The flat alternative requires nine exceptional slots in six children

Suppose the flat alternative holds. Equality of each edge's sine current
before and after the shift means its current is independent of the slot.
Each internal C9 then has a common positive current: zero current would
force all its cut-free gaps to zero, and negative current would force all
gaps negative, both contradicting winding one. Write its acute solution
as `a` in `(0,pi/2]`; each gap is either a or `pi-a`.
If there are m obtuse gaps, their sum is `9a+m*(pi-2a)=2*pi`.
For `m>=2` this sum is at least `2*pi+5a>2*pi`. Thus the only profiles are

\[
 \begin{array}{ll}
 \text{uniform:}&\kappa_j=2\pi/9\quad\text{for every }j,\\
 \text{one obtuse gap:}&\kappa_{k_c}=6\pi/7,\qquad
                  \kappa_j=\pi/7\quad(j\ne k_c).
 \end{array}
 \tag{17n}
\]

Adjacent children cannot have different types. On a wheel edge connecting
such children, at the eight ordinary internal slots the successive
compatible wheel gaps differ by `+/-5*pi/63`. There are two consecutive
ordinary slots, so three consecutive wheel gaps would have equal sine
and nonzero constant spacing `epsilon=+/-5*pi/63`. This is impossible:
`sin t=sin(t+epsilon)=sin(t+2epsilon)` first implies
`sin(t+epsilon)*(1-cos epsilon)=0` and then `sin epsilon=0`.
The wheel is connected, so all six children have the same type.

If all are uniform, they share an internal profile and section 5.2 gives
a negative relative Hessian direction. In the other case, d from (17k)
has the same two values in every child: its high value occurs only at
slot `k_c`, and the difference between the two values is `q=5*pi/7`.
Use `Md=0` at node `(c,k_c-1)`. Its only internal neighbor with high d is
the forward neighbor, across an edge of weight `cos(pi/7)>0`; its internal
contribution to Md is `-q*cos(pi/7)`, which is nonzero. Cancellation
requires a wheel neighbor whose exceptional slot is `k_c-1`; otherwise
all wheel differences of d vanish. At `(c,k_c+1)`, the unique internal
high neighbor is backward across the obtuse edge. Its contribution is
`+q*cos(pi/7)`, so a wheel neighbor with exceptional slot `k_c+1` is also
required. These arguments need no sign assumption on the wheel weights.

The nonempty set of occupied exceptional slots is therefore closed under
both neighboring slots modulo nine. It must contain all nine slots, but
six children can occupy at most six. This contradiction excludes the
flat PSD alternative exactly, without a strict-positivity assumption.

Consequently **every stationary candidate with a PSD full phase Hessian in
the original complete marked class has a changing edge with strictly
negative midpoint cosine in (17k)**. This is negative curvature of one
changing edge potential at the comparison midpoint. Neither negative total
segment curvature nor a negative candidate-Hessian direction has been
proved; other edges can contribute positively. It is not an assertion
about time evolution. Symmetry and stationarity alone have not forced a
common profile.

For an independent nonstationary control, the phase-pair blind point
(17g) has exactly two changing internal gaps per child under (17k), while
its wheel gaps are unchanged. Its stationarity secant is

\[
 \sum_e(\sin b_e-\sin a_e)(b_e-a_e)
       =3\pi(\sqrt3+1)>0.
 \tag{17o}
\]

It therefore fails the first identity in (17l), although all nine pair
sums pass. The stationary conditions add information missing from those
sums; this example is not evidence about existence of a stationary target.

The remaining sector is regular, genuinely nonseparable and has negative
midpoint cosine on a changing edge. Neither that necessary property nor
the resultant bound certifies stability or excludes this sector. A full
decision still requires a joint stationary/curvature argument or a
constructed stable target on all 53 relative phase directions. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#contact-maintenance-certificate-admission)
owns that unresolved admission. No new law, target, source capture or
physical identification follows from these restrictions.

## 6. Actual aligned outer-zero control after the full observation word

The local target in (2) becomes a maintenance result only with an actual
source inclusion argument. This section uses the separately declared
outer-zero control and primitive external origin beta=0. It retains the
original twelve allowances, recovered fine states and all five sequential
observation blocks. The [source owner](ORGANIZED_OUTER_WINDING.md#6-paired-actual-residual-control-and-interpretation)
supplies the actual residual control, not the exactly stationary zero-error
target. No coordinate is reset after observation and no settling interval
is inserted.

Let r and z be the freshly rebuilt full-form and full-phase error upper
bounds from the host source, and `q=1e-70` the external source bound.
Let `g` be the upper bound for gamma and T the complete observation-word
duration. At the beginning of that word the actual control has
`||x_H||_infinity<=r` and phase offsets from psi of magnitude at most z.
During observation the host has degree four and both rows retain their
uniform quarter factor. The diffusion semigroup contracts the maximum
norm, and `|S_i|<=4`, so throughout the complete word

\[
 B=r+gT,\qquad \|x_H\|_\infty\le B.
 \tag{18}
\]

All fifteen supplied phase events leave form unchanged. Their accumulated
commanded offsets sum to zero at the end of the word, but the continuous
phase drift survives. Because `|theta_i'|<=2*g*B`, the actual reached
host phase offsets satisfy

\[
 Z_c=z+2gBT,\qquad
 |\theta_{H,c,j}-\psi_j|\le Z_c.
 \tag{19}
\]

This bound accounts for motion during every commanded phase offset; it
does not interpret the final inverse command as restoration of state.
The separately recovered external constituent retains form and phase
offset magnitudes at most q. Thus define

\[
 b=\max(B,q),\qquad e=\max(Z_c,q).
 \tag{20}
\]

The declared beta=0 and preserved zero means put the actual joined source
on the same mean leaf as `x*=0, theta*_{c,j}=psi_j`. Each internal gap
is in `[alpha-2e,alpha+2e]`; ring and spoke gaps have absolute value at
most `2e`. Initial inclusion in the joined acute target chart follows by
checking the strict margins

\[
 \alpha-2e>0,\qquad \pi/2-\alpha-2e>0,\qquad \pi/2-2e>0.
 \tag{21}
\]

The first margin is a useful initial property, not an assertion that all
internal edge gaps remain positive forever.

### Present storage and new attachment work

Every form edge difference has magnitude at most 2b. The 144 edges give
form storage at most `288*b^2`. The target phase gradient vanishes, so
the sum of first-order phase terms is zero. Every edge phase perturbation
has magnitude at most 2e, and the absolute second derivative of its
potential is at most one. Taylor's integral remainder therefore bounds
the complete phase excess by `144*(2*e)^2/2=288*e^2`. Consequently

\[
 H(0)-6V_9\le U_0:=288(b^2+e^2).
 \tag{22}
\]

This is the storage of the actual reached source, including the observation
history through (18)--(19). It is not the sum of historical preparation
expenditures. For the new 45-edge attachment event alone,
`1-cos(a)<=a^2/2` gives the separate bound

\[
 0\le W_B\le\frac{45}{2}\big[(B+q)^2+(Z_c+q)^2\big].
 \tag{23}
\]

All earlier acquisition, outer attachment and observation events retain
their own signed work and expenditure accounting. A small present excess
or new attachment cost neither erases these costs nor assigns dissipated
storage to an undeclared event reservoir.

### Full-state trap and convergence

On the joined acute cell, each of the six internal C9 cycles has winding
one and phase cost at least V9 by convexity. All 90 wheel edges have
nonnegative phase cost. An internal acute boundary costs at least
`b9-V9` above the six-cycle minimum, where

\[
 b_9=1+8[1-\cos(3\pi/16)].
 \tag{24}
\]

This is the smaller positive-boundary value. A negative internal face
leaves the other eight gaps summing to `5*pi/2` and has the larger bound
`1+8*(1-cos(5*pi/16))`. A wheel-edge acute boundary costs one while
the six internal cycles retain their minima. Hence every feasible acute
cell boundary has storage excess at least

\[
 \mathcal B=\min\{b_9-V_9,1\}>0.
 \tag{25}
\]

No claim that the separate minimizing cycle configurations can be realized
simultaneously is needed: these are lower bounds on every actual boundary
point. The phase-space cell uses the branches inherited from the aligned
target. Within it all edge cosines are positive, so the phase Hessian is
positive definite on the zero-mean leaf. Its lifted cell is convex, and
the target is its unique stationary phase point.

If the rebuilt actual source satisfies (21) and `U0<min(1/100,Bcal)`,
nonincrease of H forbids a first exit from the cell. Use the preserved
closed sublevel `H-6V9<=U0<Bcal` on the fixed-mean leaf inside the target
cell. Its closure cannot meet the cell boundary because every boundary
point costs at least Bcal. Connectedness bounds form through its quadratic
storage and phase through fixed-branch edge differences. The sublevel is
therefore compact. Its largest invariant zero-loss set requires `Lx=0` and
`S(theta)=0`; it is the single aligned target. The invariance principle
therefore proves **all-future retention of six internal windings one and
nine old outer windings zero, and full-state convergence to the target**.
It supplies no finite settling time. The proof does not extend source
inclusion to beta different from zero or to the outer-one preparation.

## 7. Fixed actual-source assessment and research boundary

The declared exact-rational rebuild at the maximum of all twelve original
allowances certifies source admission and the following conservative bounds:

\[
 \begin{split}
 B&<6.224\cdot10^{-8},& Z_c&<7.746\cdot10^{-15},\\
 U_0&<1.116\cdot10^{-12}<1/100,&
 \mathcal B&>0.2426,\\
 W_B&<8.714\cdot10^{-14},&
 D(\theta_1(0))&<1.395\cdot10^{-16}<8/5<9d.
 \end{split}\tag{26}
\]

The three initial cell margins are strictly positive. All original source
directions survive the argument; arbitrary correlation is permitted, and
smaller admitted allowances satisfy the same monotone bounds. The actual
beta=0 outer-zero control therefore passes the frozen criterion with its
whole observation history. It retains all fifteen marked windings for
all future time and converges to the aligned target under the supplied
continued-contact law. The statement is about an entire admitted source
family, not a fitted nominal trajectory.

For the outer-one family, (26) certifies only initial membership in the
negative phase-curvature domain, uniformly over the supplied relative
origin. Its nonstationary future is not replaced by an equilibrium or an
infinitesimal perturbation calculation. The exact carrier cannot remain
in the old acute chart indefinitely; a stable full-fine stationary
organization retaining that chart would require D to exceed 9d. The
existence, acquisition and maintained identity of such a substantially
reorganized fine state remain unresolved. An outer gap becoming nonacute
would still require a separate winding-loss check. The subsequent
[full-domain curvature result](#full-chamber-phase-curvature-tests) supplies
the stronger necessary condition `D>=9/4` without the acute outer-gap
restriction and excludes every common internal phase profile. It does not
extend the frozen source calculation into a future-retention certificate.

These results separate maintained constituent identity, loss of a convenient
capture chart, stationary instability and actual-source identity loss.
They do not justify another series of short-horizon refinements or transfer
the beta=0 control to another registration. Any following change of
organization, interface, state restriction or complete law needs its own
admission. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns that decision and the property-first physical comparison; none of
the present bounds identifies mass, charge or another fundamental physical
property.

The [reflection classification](ORGANIZED_CONTACT_REFLECTION_OBSTRUCTION.md)
adds a separate obstruction without an acute-gap or small-defect premise.
Its static reflection sector includes nonuniform internal profiles; no
stable stationary state there retains all fifteen winding-one identities.
An explicit nonacute saddle shows that stationary existence remains distinct
from stability. Neither result supplies the actual outer-one source's fate.

## 8. Detached implementation and verification boundary

The [maintenance owner](../../src/tnfr/physics/_organized_contact_maintenance.py)
reuses the existing complete contact support and exact interval arithmetic.
Its geometric derivation retains the separate form and phase eigenvalues
and both complete tangent rows, the two conserved neutral directions and
the constant in the full-fine curvature obstruction. Repeated spectral
coefficients are evaluated once within each derivation, without caching
source premises or verdicts. Its source assessor
admits all twelve original primitive allowances and freshly rebuilds the
source and observation evidence. It accepts no incoming report as a
replacement for those premises and advances no trajectory.

Source inclusion, attachment work, initial defect-domain membership and
the beta=0 capture verdict retain distinct meanings. A passing initial
defect-domain flag is not a future instability, retention or loss verdict.
The original external-contact and registration reports retain their
earlier finite scope and frozen evidence.

The [dedicated tests](../../tests/physics/test_organized_contact_maintenance.py)
exercise complete support and tangent modes, both stationary controls,
primitive admission, source rebuilding and scoped source-bound verdicts.
Source regeneration and trajectory execution remain forbidden in this
detached proof path. Numerical checks verify implemented arithmetic and
wiring; the analytic arguments above establish the conditional mathematical
claims and their domain boundaries.

The later analytical extension in sections 5.1-5.6 uses exact identities
and counterexamples. Static regression checks separately reconstruct the
stationary controls' neighbor resultants and the blind point's lifted
secant from edge sums. These are arithmetic controls, not a stationary
search or a proof by sampled configurations. The extension does not change
the earlier assessor's threshold, flags or frozen evidence, and adds no
numerical decision API. Its necessary curvature, resultant and slot
conditions must not be exposed as sufficient stability or maintenance
verdicts.
