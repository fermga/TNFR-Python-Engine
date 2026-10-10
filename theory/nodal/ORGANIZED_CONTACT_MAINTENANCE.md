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
nonincrease of H forbids a first exit from the cell. The strict sublevel
is separated from its boundary. On the fixed-mean leaf the connected
form quadratic form and bounded fixed-branch phase differences make it
compact. Its largest invariant zero-loss set requires `Lx=0` and
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
would still require a separate winding-loss check.

These results separate maintained constituent identity, loss of a convenient
capture chart, stationary instability and actual-source identity loss.
They do not justify another series of short-horizon refinements or transfer
the beta=0 control to another registration. Any following change of
organization, interface, state restriction or complete law needs its own
admission. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns that decision and the property-first physical comparison; none of
the present bounds identifies mass, charge or another fundamental physical
property.

## 8. Detached implementation and verification boundary

The [maintenance owner](../../src/tnfr/physics/_organized_contact_maintenance.py)
reuses the existing complete contact support and exact interval arithmetic.
Its geometric derivation retains the separate form and phase eigenvalues
and both complete tangent rows, the two conserved neutral directions and
the constant in the full-fine curvature obstruction. Its source assessor
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
