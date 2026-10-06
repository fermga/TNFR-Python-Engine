# Native mediated pattern dynamics

Native one-, two- and three-port mediator reduction, fast-limit scope, memory clocks and the frozen causal-response control; no boundary source is installed.

Part of [Native pattern reduction and memory](RELATIONAL_PATTERN_MEMORY.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="shared-implementation-and-evidence-scope-1"></a>


<a id="mediated-pattern-interaction"></a>
## 9. A nodal intermediary mediates joint form/phase interaction

### Full state and conditional recovery

Replace the direct `0--5` bridge between the prepared C5 rings by
`0--10--5`. All twelve edges have unit conductance. The eleven-node state
retains signed form and primitive phase at every node, including mediator 10.
Hold capacity `nu>0` on both rings and `mu>0` on the mediator, with fixed
`e,w,beta>0` and the same structural clock. Here `e,w` are the effective
normalized channel coefficients. The law is the existing
`RelationalExchangeModel`, not an additional coupling or memory equation.

At uniform form, winding-one phases `theta_i=2*pi*(i mod 5)/5` on each ring
and `theta_10=0`, all sine sums vanish and every edge cosine is positive.
Set `c=cos(2*pi/5)`, `r=1+2*c`. Port phase metrics are `H_p=pi*r`, and
the mediator metric is `H_m=2*pi`. The
[existing local recovery theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
therefore applies to this connected graph: two common-offset freedoms and
twenty stable transverse directions, with a sufficiently small acute recovery
neighborhood. The two rings can have a joint restoring geometry through an
intermediary without a direct ring-to-ring edge. This is local recovery of a
prepared geometry on supplied support, not capture from disconnected support,
a global formation theorem or a physical bound-state identification.

### Derived memory, with two indispensable hidden coordinates

Write `u,v` for form and lifted phase deviations. Retain all twenty ring
coordinates as `y`, and hide only `z_m=(u_10,v_10)`. Define

\[
D_0=\begin{pmatrix}-e&-w/\pi\\w/(\beta\pi)&0\end{pmatrix},\qquad
B_0=-\tfrac12D_0,\qquad
C_0=\begin{pmatrix}e/3&w/(\pi r)\\-w/(\beta\pi r)&0\end{pmatrix}.
\]

The Jacobian of the admitted full field has hidden row
`z_m_dot=mu*D_0*z_m+mu*B_0*(z_0+z_5)`. Each visible port receives
`nu*C_0*z_m`; its other instantaneous terms remain in the full visible block
`A`. Here `z_0,z_5` denote the two form/phase port pairs within `y`, not
closed states of the rings. Let `B` inject `nu*C_0` at both port rows, and
let `C` read both port pairs with `mu*B_0`. Exact tangent elimination gives

\[
\dot y(t)=Ay(t)+B e^{\mu D_0t}z_m(0)
+\int_0^t B e^{\mu D_0(t-s)}C y(s)\,ds.
\]

The cross-region and self-memory port blocks are identical:

\[
\boxed{\quad K_\mu(t)=\mu\nu C_0 e^{\mu D_0t}B_0.\quad}
\]

This kernel follows from the declared field; no delay, fitted gain or primitive
pulse is introduced. It is exact for the tangent model. Nonlinear continuation
retains nonlinear hidden forcing, as in section 3, and cannot substitute this
fixed convolution for the full engine. The initial hidden-state term remains
necessary even when a particular experiment prepares it to zero.

Since `det(C_0)=w^2/(beta*pi^2*r^2)>0`, both mediator coordinates influence
the observed ring rates. For the supplied full-state Jacobian `J` and coordinate
observation `O`, `rank(O)=20` and `rank((O;OJ))=22`. Thus a twenty-coordinate
instantaneous all-state linear closure is impossible; retaining both hidden
coordinates or their exact memory is necessary. This reuses the invariant-row
argument, rather than introducing another state-minimality criterion.

<a id="mediator-pressure-boundary-chart"></a>
### An exact nonlinear pressure chart retains the moving boundary

The degree-two mediator also admits an exact change of coordinates beyond
the tangent approximation. Retain the full visible ring state and the same
fixed support, coefficients and capacities. On a continuous local phase lift,
retain full-field admission, including all ring edges, and write

\[
\bar x=\frac{x_0+x_5}{2},\qquad
\bar\theta=\frac{\theta_0+\theta_5}{2},\qquad
\delta=\theta_5-\theta_0,\qquad
u=x_m-\bar x,\qquad v=\theta_m-\bar\theta.
\]

Here `u,v` are mediator contrasts with the moving endpoints, not absolute
deviations from the equilibrium. Assume both incident gaps remain acute:
`|v+delta/2|<pi/2` and `|v-delta/2|<pi/2`. These imply `|delta|<pi`
and `|v|<pi/2`. The mediator's relative neighbor resultant is exactly

\[
z_m=2\cos(\delta/2)e^{-iv},\qquad
H_m=2\pi\cos(\delta/2)\operatorname{sinc}(v)>0,
\qquad g_m=-v/\pi,\qquad q_m=2u.
\]

Consequently its evaluated pressure, not a stale stored observation, obeys

\[
p_m=-eu-\frac w\pi v,\qquad
v=-\frac\pi w(p_m+eu).
\]

At fixed visible state, `(u,p_m)` is therefore an invertible replacement for
the two mediator coordinates. It preserves their initial information; it
does not remove the node, its capacity or either incident edge. Applying the
unchanged full field and differentiating the moving endpoint averages gives

\[
\dot u=\mu p_m-\dot{\bar x},\qquad
\dot v=\frac{\mu w u}
 {\beta\pi\cos(\delta/2)\operatorname{sinc}(v)}
 -\dot{\bar\theta},\qquad
\dot p_m=-e\dot u-\frac w\pi\dot v.
\]

These are exact nonlinear coordinate identities throughout the admitted
interval. The endpoint rates come from the retained full network; they are
not new external forcing or a closed two-port law. In particular, with held
`mu`, pressure history determines the contrast only after retaining its
initial value and the endpoint motion:

\[
u(t)=u(0)+\mu\int_0^t p_m(s)\,ds
       -[\bar x(t)-\bar x(0)].
\]

Substituting this expression and `v=-pi*(p_m+e*u)/w` into the pressure row
gives a nonlinear memory representation with moving-boundary terms. Dropping
`u(0)`, either endpoint-rate term, or the visible phase contrast changes the
model. This is the same closure principle as the
[P2 pressure chart](JOINT_PARAMETER_RESPONSE.md#pressure-state-closure), with
the additional boundary motion required by the mediator's environment.

The nonzero determinant of `C_0` above also rules out an exact smooth scalar
mediator summary for this full visible-rate observation on an open
neighborhood of the reference. At fixed visible state, the two rates at one
port have derivative `nu*C_0` with respect to the two mediator coordinates,
of rank two; any smooth factorization through one scalar has rank at most
one. This is local minimality for the stated observation, not a claim about
restricted one-dimensional preparations, lossy approximations or histories.
The [finite nonlinear compensation witness](RELATIONAL_EFFECTIVE_CONNECTIONS.md#environmental-capture-domain)
exhibits the corresponding failure of pressure alone inside a recovery
domain. The coordinate identities themselves also hold at `mu=0`, where
the mediator's absolute state freezes but its endpoint-relative coordinates
can still change. Positive-capacity recovery remains a separate theorem.
The [native controls](../../tests/physics/test_relational_mediation.py) check
both moving-boundary rows and an independent pressure derivative at nonuniform
visible states, including a frozen mediator. They test these coordinate
identities, not a reduced nonlinear trajectory solver.

<a id="fast-mediator-reduction"></a>
### A controlled nonlinear reduction when the mediator is fast

Hold the ring capacities at one and vary only the positive mediator capacity
`mu`, with the same unforced law, support, clock and effective coefficients. This is a
time-scale regime of the existing law, not a new interaction. Let `y` retain
all twenty visible ring coordinates and let `z=(u,v)` be the endpoint-relative
mediator coordinates just defined. Reconstruct the full state by
`x_m=bar(x)+u`, `theta_m=bar(theta)+v` and evaluate the native visible field
`F(y,z)`. Set `b(y,z)=(dot(bar(x)),dot(bar(theta)))` using its port rows. Then

\[
\dot y=F(y,z),\qquad \dot z=\mu f_\delta(z)-b(y,z),\qquad
f_\delta(u,v)=
\begin{pmatrix}
-eu-wv/\pi\\
wu/[\beta\pi\cos(\delta/2)\operatorname{sinc}(v)]
\end{pmatrix}.
\]

Both `F` and `b` are independent of `mu`: visible capacities are held, and
this relational pressure does not consume neighbor capacity. The unique
frozen-boundary fast equilibrium is `z=0`. The candidate reduced equation is
`dot(y_*)=F(y_*,0)`, with the same visible initial state. At finite `mu`, however,
`dot(z)=-b(y,0)` on `z=0`; the midpoint constraint is generally not invariant.
Its validity as an approximation requires the estimate below, not merely
instantaneous storage minimization.

**The reduced field retains the actual interface.** Write `q_ring` for the
form gradient from each ring's internal edges alone. At the reconstructed
midpoint,

\[
\widetilde q_0=q_{{\rm ring},0}+\frac{x_0-x_5}{2},\qquad
\widetilde q_5=q_{{\rm ring},5}+\frac{x_5-x_0}{2},
\]

with all other visible gradients unchanged. The relative phase resultants
at the two ports are

\[
\begin{aligned}
\widetilde Z_0&=e^{i(\theta_1-\theta_0)}+
 e^{i(\theta_4-\theta_0)}+e^{i\delta/2},\\
\widetilde Z_5&=e^{i(\theta_6-\theta_5)}+
 e^{i(\theta_9-\theta_5)}+e^{-i\delta/2}.
\end{aligned}
\]

Retain the original port degree three, the other visible degrees two, and
the native definitions `g_tilde=Arg(Z_tilde)/pi` and
`H_tilde=pi*|Z_tilde|*sinc(Arg(Z_tilde))`. Thus the visible rows are
`dot(x_i)=-e*q_tilde_i/d_i+w*g_tilde_i` and
`dot(theta_i)=w*q_tilde_i/(beta*H_tilde_i)`. In particular, this is not the
law on a ten-node graph with an ordinary unit `0--5` edge: that edge would
give the full form contrast and a full-angle neighbor phasor. The native
eleven-node field evaluated at the midpoint already supplies the required
reduction; no second pressure implementation is needed.

Its storage is also inherited, rather than fitted:

\[
\begin{aligned}
S_{\rm eff}={}&\frac12\sum_{\{i,j\}\in E_{\rm ring}}(x_i-x_j)^2
 +\frac{(x_0-x_5)^2}{4}\\
&+\beta\left[\sum_{\{i,j\}\in E_{\rm ring}}
 (1-\cos(\theta_i-\theta_j))+2(1-\cos(\delta/2))\right],\\
\dot S_{\rm eff}={}&-e\sum_{i\ne m}\frac{\widetilde q_i^2}{d_i}\le0.
\end{aligned}
\]

To obtain the last identity, both hidden storage derivatives vanish at the
midpoint, so differentiating its reconstruction contributes no extra term.
The remaining derivatives and form/phase exchange cancellation are the native
ones. The half-angle potential belongs to the declared `|delta|<pi` lift;
it is not a globally defined replacement cosine edge on the phase circle.
The selected phase lift remains part of its domain. Away from the midpoint,
the exact storage difference is

\[
S_{\rm full}(y,z)-S_{\rm eff}(y)
 =u^2+2\beta\cos(\delta/2)(1-\cos v)\ge0.
\]

A fixed nonzero hidden initial state can therefore carry finite excess
storage even when its visible trajectory effect becomes small. Its initial
layer loss is retained by the full law; the reduction does not claim uniform
storage or instantaneous-rate convergence at time zero.

**Uniform attraction for the default coefficients.** The following explicit
bound uses `e=w=1/2`, `beta=1`; it is not a uniform claim over all coefficients.
For `|delta|<=1/2`, `|v|<=1/4`, write `f_delta(z)=A_0*z+R_delta(z)` with

\[
A_0=\begin{pmatrix}-1/2&-1/(2\pi)\\1/(2\pi)&0\end{pmatrix},\qquad
P=\begin{pmatrix}2&\pi\\\pi&2+\pi^2\end{pmatrix}.
\]

Direct multiplication gives `A_0^T*P+P*A_0=-I`. The leading minor and
determinant of `P-I` are one, while `trace(P)=4+pi^2<14`, hence `I<P<14I`.
Moreover,

\[
\cos(\delta/2)\operatorname{sinc}(v)\ge
 \frac{31}{32}\frac{95}{96}=\frac{2945}{3072},\qquad
\|R_\delta(z)\|\le\frac{127}{6\cdot2945}\|z\|.
\]

Here `cos(t)>=1-t^2/2`, `sinc(t)>=1-t^2/6` on the stated intervals and
`pi>3` suffice. Therefore `2*||P||*127/(6*2945)<1/2`, and the fast field obeys

\[
2z^TPf_\delta(z)\le-\tfrac12\|z\|^2.
\]

This common quadratic bound does not differentiate a moving `P(delta)` and
holds for arbitrary admitted changes of the visible phase contrast. It uses
positive form damping: the `e=0` periodic boundary does not justify the same
attracting reduction.

**Domain and continuation hypotheses.** Fix a finite horizon `T` on which
the reduced solution `y_*` exists. Choose constants `d>0`, `0<rho<1/4` such
that its closed visible tube of radius `d`, together with
`W(z)=sqrt(z^T*P*z)<=rho`, is compactly contained in a smooth full-state
chart. Require all reconstructed ring edges to remain acute and
`|delta|<=1/2` there. The mediator edges are then also acute, because
`|v+-delta/2|<=1/2<pi/2`. In this fixed tube choose bounds, independent of
`mu`,

\[
\|b(y,z)\|\le B,\qquad
\|F(y,z)-F(y_*(t),0)\|
 \le L(\|y-y_*(t)\|+\|z\|).
\]

Smoothness supplies finite such bounds on an admitted tube; no numerical
values are inferred merely from a trajectory. Its margin must be justified
for the reduced solution, rather than assuming the full solution stays in
the domain whose preservation is to be proved. Near the aligned winding
reference, its strict acute margins and smooth reduced field provide such
a neighborhood for sufficiently close preparations on any fixed finite
horizon, by ordinary local existence and continuous dependence.

**Initial layer and visible error.** Up to the first exit from this tube,
the previous quadratic inequality and `|b|<=B` give

\[
D^+W\le-\frac\mu{56}W+14B,\qquad
W(t)\le e^{-\mu t/56}W(0)
 +\frac{784B}{\mu}(1-e^{-\mu t/56}).
\]

The upper right derivative formulation includes `W=0`. Since `||z||<=W`,
comparison of the two visible equations and Gronwall's inequality imply

\[
\boxed{\quad
\sup_{0\le t\le T}\|y(t)-y_*(t)\|
 \le\frac{Le^{LT}}{\mu}\,[56W(0)+784BT].\quad}
\]

For `W(0)<rho`, choose `mu` so that `784B/mu<rho` and the displayed visible
bound is strictly below `d`. The estimate for `W` is a convex combination of
two values below `rho`. Both estimates therefore prevent a first exit;
compact smooth continuation extends the solution through `T` and validates
the bounds on that whole interval. These conditions are sufficient, not
optimal, and do not supply a numerical cutoff without actual tube bounds.

A fixed small hidden initial state is allowed. Its visible contribution is
included in the `56W(0)/mu` term, while the hidden state retains an initial
layer of decay `exp(-mu*t/56)`. It is not uniformly small at time zero:
for `mu>=1` it becomes `O(1/mu)` after a time of order `log(mu)/mu`.
Thus finite-memory influence can be approximated in this regime without
denying the exact hidden-state obstruction at fixed capacity. Neither
arbitrary initial environments nor infinite-horizon errors are covered.

This is an ideal continuous-law result. Its field evaluation reuses detached
[`evaluate_relational_exchange`](../../src/tnfr/dynamics/relational.py)
at the reconstructed midpoint and projects the visible rows. The
[native static controls](../../tests/physics/test_relational_mediation.py)
can check its field substitution and storage identities; they do not certify
the tube constants or a reduced trajectory. Fixed-step Euler is not justified
as `mu` grows: resolving the fast initial layer needs a separate numerical
step-size analysis. No new solver, primitive edge, occurrence law or physical
identification follows from eliminating this already supplied mediator.

<a id="two-mediator-composition"></a>
### Two fast intermediaries compose through their inherited interface

Replace the intermediary path by `0--10--11--5`, retaining the same two C5
rings and all thirteen unit edges. There are twenty visible coordinates and
four hidden coordinates. Hold ring capacities at one and initially give both
intermediaries capacity `mu>0`, with `e=w=1/2`, `beta=1`. The unforced law
and clock are unchanged. Choose a local acute path lift and write
`X=x_5-x_0`, `delta=theta_5-theta_0`. Frozen-endpoint hidden equilibrium is

\[
x_{10}^*=x_0+X/3,\quad x_{11}^*=x_0+2X/3,\qquad
\theta_{10}^*=\theta_0+\delta/3,\quad
\theta_{11}^*=\theta_0+2\delta/3.
\]

Let `u=(x_10-x_10^*,x_11-x_11^*)` and
`v=(theta_10-theta_10^*,theta_11-theta_11^*)`. For `z=(u_1,u_2,v_1,v_2)`, set

\[
M=\begin{pmatrix}2&-1\\-1&2\end{pmatrix},\qquad
\begin{aligned}
C_1&=\cos(\delta/3+v_2/2)\operatorname{sinc}(v_1-v_2/2),\\
C_2&=\cos(\delta/3-v_1/2)\operatorname{sinc}(v_2-v_1/2).
\end{aligned}
\]

The hidden metrics are `H_10=2*pi*C_1`, `H_11=2*pi*C_2`, the form gradient
is `M*u`, and the phase source is `-M*v/(2*pi)`. Thus the exact fast map is

\[
f_\delta(z)=
\begin{pmatrix}
-(e/2)Mu-(w/(2\pi))Mv\\
(w/(2\beta\pi))\operatorname{diag}(C_1^{-1},C_2^{-1})Mu
\end{pmatrix}.
\]

The moving-boundary system again has `dot(y)=F(y,z)` and
`dot(z)=mu*f_delta(z)-b(y,z)`. Here `b` consists of the derivatives of the
four interpolated coordinates above, for example
`dot(x_10^*)=(2*dot(x_0)+dot(x_5))/3`. All these endpoint rates come from
the retained visible field. The hidden equilibrium is unique in this chart:
the phase rows first require `M*u=0`, and the form rows then require `M*v=0`.

**Exact static composition and reduced balance.** Evaluating the full native
field at this lift gives

\[
\widetilde q_0=q_{{\rm ring},0}-X/3,\qquad
\widetilde q_5=q_{{\rm ring},5}+X/3,
\]

and port resultants equal to their two internal ring phasors plus
`exp(i*delta/3)` at port 0 and `exp(-i*delta/3)` at port 5. Retain original
port degree three and all native metric/phase-source definitions. The storage
and its derivative under the induced visible law are

\[
S_{\rm eff}=S_{\rm rings}+X^2/6+3\beta(1-\cos(\delta/3)),\qquad
\dot S_{\rm eff}=-e\sum_{i\ne10,11}\widetilde q_i^2/d_i\le0.
\]

Both hidden storage gradients vanish at the lift, so the same envelope
chain rule used for one intermediary proves the balance. Hidden initial
storage is not removed from the full system. Its form excess is
`u^T*M*u/2`; its phase excess is

\[
\beta[3\cos(\delta/3)-\cos(\delta/3+v_1)
 -\cos(\delta/3+v_2-v_1)-\cos(\delta/3-v_2)].
\]

The latter is nonnegative when the three lifted gaps remain acute, by strict
convexity of `1-cos` on that interval and their fixed sum. A fixed hidden
initial displacement can therefore again carry finite initial-layer storage.

Successive static elimination gives the same result if its inherited
interface is retained. Eliminating node 10 first gives
`x_10=(x_0+x_11)/2`, `theta_10=(theta_0+theta_11)/2`. The remaining gradient is
`q_11=(x_11-x_0)/2+(x_11-x_5)`, still with original degree two, while its
phase resultant is

\[
e^{-i(\theta_{11}-\theta_0)/2}+e^{i(\theta_5-\theta_{11})}.
\]

Setting both remaining rows to zero yields the same thirds; reversing the
order does too. By contrast, replacing the eliminated path by an ordinary
unit `0--11` edge would produce the false interpolation
`x_10=x_0+X/4`, `x_11=x_0+X/2`. Reconstructing the original graph gives
`q_11=-X/4`, and the analogous phase preparation has hidden gradient
`sin(delta/4)-sin(delta/2)`, generally nonzero. Static elimination therefore
composes through the inherited field, not through an arbitrary replacement
graph. This equality does not by itself interchange finite-capacity dynamics
or establish nested initial layers for successively separated time scales.

The same static statement holds for a finite path of `ell` unit-conductance
edges, with only degree-two interior nodes and a specified path lift.
Here `ell` counts fine links; it is not the separate graph `length` attribute
or an inferred physical distance:

\[
S_{\rm path}^{\rm eff}(X,\delta)
 =\frac{X^2}{2\ell}+\beta\ell(1-\cos(\delta/\ell)).
\]

Each gap is `delta/ell` in the selected acute sector. Joining lengths
`ell_1,ell_2` and minimizing their common boundary places that boundary at
`x_A+ell_1*(x_B-x_A)/(ell_1+ell_2)`, with the same lifted phase interpolation.
The phase condition follows from the injectivity of sine on the acute
interval. Energy and endpoint messages therefore agree with length
`ell_1+ell_2`, associatively. The message retains length, lift, endpoint form
contrast, adjacent phase and original node degrees. Different allowed lift
sectors cannot be silently identified using endpoint phases modulo full
turns. This is a static series result, not a dynamical estimate uniform in
path length or a result for branching interiors.

For example, a three-edge path whose endpoint circular gap is `3*pi/4`
admits both lifted gaps `pi/4` on every edge and `-5*pi/12` on every edge.
Its two hidden phase pairs are respectively `(pi/4,pi/2)` and
`(-5*pi/12,-5*pi/6)` relative to the left endpoint. Both have zero hidden
phase source and acute fine edges, yet their adjacent port phasors differ.
With uniform form and the maintained ring twists, donor pressure has opposite
signs. Thus length and circular endpoint phases alone are insufficient;
static composition must retain the selected lift sector. This control is
outside the small-`delta` quantitative neighborhood used next.

**Joint fast attraction and a finite-horizon bound.** At the default
coefficients and `delta=v=0`, the joint fast Jacobian and a common quadratic
matrix are

\[
J_0=\begin{pmatrix}-M/4&-M/(4\pi)\\M/(4\pi)&0\end{pmatrix},\qquad
P_4=\begin{pmatrix}2I&\pi I\\\pi I&(2+\pi^2)I\end{pmatrix}.
\]

They satisfy `I<P_4<14I` and
`J_0^T*P_4+P_4*J_0=-diag(M,M)/2<=-I/2`. For
`|delta|<=1/2`, `|v_1|,|v_2|<=1/8`, the cosine arguments have magnitude at
most `11/48` and the sinc arguments at most `3/16`. Hence

\[
C_i\ge\frac{4487}{4608}\frac{509}{512}
 =\frac{2283883}{2359296},\qquad
0\le C_i^{-1}-1\le\frac{75413}{2283883}<\frac1{28}.
\]

Since `||M||=3` and `pi>3`, the nonlinear remainder obeys
`||f_delta(z)-J_0*z||<=||z||/112`. Consequently
`2*z^T*P_4*f_delta(z)<=-||z||^2/4`. This controls the four hidden coordinates
jointly; independent single-node decay estimates are not substituted for the
coupled proof.

Reuse the preceding finite-horizon tube and first-exit construction, now with
`W=sqrt(z^T*P_4*z)`, `rho<1/8`, and bounds `B,L` for this twelve-node
reconstruction. On its admitted interval,

\[
\begin{aligned}
W(t)&\le e^{-\mu t/112}W(0)
 +\frac{1568B}{\mu}(1-e^{-\mu t/112}),\\
\sup_{0\le t\le T}\|y(t)-y_*(t)\|
 &\le\frac{Le^{LT}}\mu[112W(0)+1568BT].
\end{aligned}
\]

Require `W(0)<rho`, `1568B/mu<rho` and a visible error bound strictly below
the tube radius `d`. These close the same first-exit argument. The three
intermediary edge gaps are acute throughout the box, with magnitude at most
`5/12`; the tube must separately retain all ring-edge and visible-state
margins. Thus the reduction has an `O(1/mu)` visible error with its hidden
initial layer retained, on a fixed finite horizon and admitted neighborhood.
Neither a fixed-step Euler certificate nor an infinite-horizon bound follows.

Fixed positive unequal capacity ratios also admit a local stability
certificate. For capacities `mu*a_1,mu*a_2`, put
`D=diag(a_1,a_2)`, `D_4=diag(D,D)` and `P_a=P_4*D_4^-1`. The fast map is
`D_4*f_delta`, and exactly
`2*z^T*P_a*D_4*f_delta=2*z^T*P_4*f_delta<=-||z||^2/4`.
Moreover `I/a_max<P_a<14I/a_min`. The same proof therefore gives constants
depending on these fixed ratios, with its hidden tube rescaled to preserve
`||z||<=sqrt(a_max)*sqrt(z^T*P_a*z)<1/8`. They are not uniform as a ratio
vanishes or becomes unbounded. The explicit `112,1568` bounds above are for
equal capacities only.

The [native static controls](../../tests/physics/test_relational_mediation.py)
check the reconstructed field, inherited balance and the wrong-edge control.
They do not evaluate a reserved trajectory or supply numerical tube constants.
No additional pressure owner, topology mutation or reduced executor is
introduced: composition is performed on the already supplied nodal system.

<a id="three-port-collective-interaction"></a>
### A branching intermediary induces a collective three-port interaction

Take three C5 rings on nodes `0,...,14`, with ports `A=0`, `B=5`, `C=10`,
and connect one intermediary `m=15` to all three ports. All eighteen edges
have unit conductance. Retain all thirty visible form/phase coordinates,
unit visible capacities, a held positive intermediary capacity, and the same
unforced relational law with `e=w=1/2`, `beta=1`. The support is supplied;
stationary elimination does not assert that the full system is stationary.

For the three port phases define

\[
Z=e^{i\theta_A}+e^{i\theta_B}+e^{i\theta_C}=Re^{i\Psi},\qquad
\bar x=(x_A+x_B+x_C)/3.
\]

Work in a local chart with `R>0` and all three gaps
`alpha_p=theta_p-Psi` strictly acute, and retain admission of every ring
edge. A nonzero resultant alone does not guarantee this acute-star premise.
The unique conditional hidden equilibrium is

\[
x_m^*=\bar x,\qquad \theta_m^*=\Psi.
\]

Indeed the hidden phase row first requires `q_m=0`, fixing the form average;
the form row then requires zero phase source. In the admitted chart this
selects the circular resultant direction. The antipodal stationary point of
the phase storage is outside this domain. Both hidden storage gradients
vanish at the admitted lift.

**Inherited field and storage.** At each port the hidden contribution to
the form gradient is `x_p-bar(x)`, and its relative neighbor phasor is
`exp(i*(Psi-theta_p))`. Add these to the port's two internal ring contributions
and retain degree three, the native phase source and the native phase metric.
Other visible rows are unchanged. Consequently the full sixteen-node native
field at the conditional lift supplies the induced visible law without
another pressure formula. Its storage is

\[
S_{\rm eff}=S_{\rm rings}
 +\frac16\sum_{p<q}(x_p-x_q)^2+\beta(3-R),\qquad
\dot S_{\rm eff}=-e\sum_{i\ne m}\widetilde q_i^2/d_i\le0.
\]

The envelope chain rule again removes the hidden reconstruction derivative
because its storage gradients vanish there. This is a closed conditional
field on the full visible state, not a closed state of three rigid rings
or an exact finite-capacity elimination of hidden history.

**The phase storage is not a sum of independent pair interactions.** Define
`V=3-R`. Even allowing arbitrary three-times differentiable pair functions
of their two absolute phases, plus one-port terms, a representation

\[
V=V_{AB}(\theta_A,\theta_B)+V_{AC}(\theta_A,\theta_C)
 +V_{BC}(\theta_B,\theta_C)+\sum_p V_p(\theta_p)
\]

would require `partial_A partial_B partial_C V=0` throughout its domain.
But at `(theta_A,theta_B,theta_C)=(0,0,t)`, with `0<t<pi/4`, direct
differentiation gives

\[
\begin{aligned}
\partial_A\partial_B V
 &=-\frac{(2+\cos t)^2}{(5+4\cos t)^{3/2}},\\
\partial_A\partial_B\partial_C V
 &=-\frac{2\sin t(1-\cos t)(2+\cos t)}{(5+4\cos t)^{5/2}}\ne0.
\end{aligned}
\]

These preparations lie in the acute-star domain and can be arbitrarily
close to alignment. Thus no independent-pair identity holds on an open
neighborhood of that reference. Allowing a nominal pair to consume the
third port's state would instead declare a collective interaction.

Low-order agreement does not remove this obstruction. For local differences
`Delta_pq=theta_p-theta_q`, set `S_k=sum_pairs Delta_pq^k` and
`P_6=product_pairs Delta_pq^2`. Expansion about alignment gives

\[
V=\frac{S_2}{6}-\frac{S_4}{216}-\frac{S_6}{19440}
 +\frac{P_6}{648}+O(\|\Delta\|^8).
\]

Here `S_4=S_2^2/2` and `S_6=S_2^3/4+3*P_6`. General pair potentials can
match the quadratic and quartic terms; the displayed genuinely mixed
storage term first occurs at sixth order. The particular cosine surrogate
`sum_pairs(1-cos(Delta_pq))/3` already fails at fourth order. This order
statement concerns storage, not every component of the nonlinear field.

**The actual port rate is collective too.** Keep all forms zero, hold ring A
in its winding-one reference, and rotate rings B and C by independent `s,t`.
Let `a=2*cos(2*pi/5)`, so `0<a<1`, and define

\[
\Psi(s,t)=\arg(1+e^{is}+e^{it}),\qquad
g(\psi)=\arg(a+e^{i\psi}),\qquad
\dot x_A=\frac w\pi g(\Psi(s,t)).
\]

The complete state and both internal neighbors of A remain fixed in this
comparison. At `s=t>0` sufficiently small, with `Q=5+4*cos(t)`,

\[
\Psi_s=\Psi_t=\frac{2+\cos t}{Q},\qquad
\Psi_{st}=\frac{2(2+\cos t)\sin t}{Q^2}>0,
\]

while

\[
g'(\psi)=\frac{1+a\cos\psi}{1+a^2+2a\cos\psi}>0,\qquad
g''(\psi)=\frac{a(1-a^2)\sin\psi}{(1+a^2+2a\cos\psi)^2}>0
\]

at the resulting positive `Psi`. Hence
`partial_s partial_t dot(x_A)=(w/pi)*(g''*Psi_s*Psi_t+g'*Psi_st)>0`.
An additive rate depending separately on `(A,B)` and `(A,C)`, plus A alone,
has zero mixed derivative under these same variations. This rules out that
independent-pair field representation, even without requiring a pairwise
storage. Shared state-dependent normalizations that read the third ring
would not satisfy the proposed independence contract.

Near alignment, this mixed rate derivative is
`2*w*t*(1+3*a)/(27*pi*(1+a)^3)+O(t^3)`. Thus the native visible field
already has a collective cubic contribution, although the storage's first
genuinely mixed contribution is sixth order. The inherited phase metric
and readout prevent those two order statements from being interchanged.

**A controlled fast regime is available locally.** To justify a dynamical
approximation, set `u=x_m-bar(x)`, `v=theta_m-Psi(y)`, `z=(u,v)` and give
the mediator capacity `mu`. The exact moving-boundary rows are

\[
\dot z=\mu
\begin{pmatrix}-eu-wv/\pi\\3wu/[\beta\pi R\operatorname{sinc}(v)]\end{pmatrix}
-\begin{pmatrix}\dot{\bar x}\\\dot\Psi\end{pmatrix},\qquad
\dot\Psi=\frac1R\sum_{p=A,B,C}\cos(\alpha_p)\dot\theta_p.
\]

The phase-boundary velocity is a weighted circular-mean derivative, not an
arithmetic average. Its port rates are native visible rates. At the default
coefficients, require `|alpha_p|<=1/4` and `|v|<=1/4` in a compact admitted
tube about a reduced visible trajectory. Then

\[
\frac R3\operatorname{sinc}(v)\ge
 \frac{31}{32}\frac{95}{96}=\frac{2945}{3072}.
\]

The same two-dimensional matrix `P` from the fast-mediator theorem therefore
gives exactly its common quadratic estimate and `56,784` bounds. Apply its
first-exit argument with newly justified bounds `B,L`, hidden radius
`rho<1/4`, and visible margin `d` for this sixteen-node reconstruction.
The boundary velocity and visible field remain independent of `mu`, so the
visible trajectory error is `O(1/mu)` on the fixed finite horizon, including
the hidden initial-layer contribution. Neither the earlier graph's tube
constants nor its preparation are silently reused. At finite capacity the
conditional lift is not generally invariant.

The hidden storage excess is exactly
`3*u^2/2+beta*R*(1-cos(v))`; a fixed nonzero hidden initial state can carry
finite storage into that layer. Its loss and initial rate discrepancy are
not erased by the visible approximation. The
[native mediation controls](../../tests/physics/test_relational_mediation.py)
check the induced rows, storage and mixed-response discriminator without a
new response campaign. The result derives a collective conditional interaction
from the supplied nodal law and support; it installs no new connection rule
or reduced executor.

### Capacity sets the memory clock, and the kernel need not be positive

Return here to the native two-port model at the start of section 9, with
`c=cos(2*pi/5)` and its original port quantity `r=1+2*c`.
This `r` is distinct from the resultant ratio `rho=R/k` above.
For that two-port single-intermediary model,
the hidden eigenvalues solve
`s^2+mu*e*s+mu^2*w^2/(beta*pi^2)=0`. They have negative real parts for the
positive premises above. Default `e=w=1/2`, `beta=1` gives two negative real
roots; an oscillating mediator is not required. For dimensionless `s>0`,
`K_(s*mu)(t)=s*K_mu(s*t)` retains the same ring capacity `nu`. Hence

\[
\int_0^\infty K_\mu(t)\,dt=\frac\nu2 C_0\qquad(\mu>0),
\]

because `B_0=-D_0/2`. Changing mediator capacity changes the transient memory
while leaving this integrated tangent kernel unchanged. This is not an
invariance of full trajectories or final nonlinear form offsets. At `mu=0`
the mediator freezes and transmits no donor-induced change from identical
mediator initial states. The positive-capacity integral limit is not uniform
as `mu` tends to zero; it cannot be assigned to the frozen case.

The phase-to-phase cross entry satisfies

\[
[K_\mu(0)]_{vv}=-\frac{\mu\nu w^2}{2\beta\pi^2r}<0,
\qquad \int_0^\infty[K_\mu(t)]_{vv}\,dt=0.
\]

Continuity and exponential decay force a positive contribution at some later
lag. Thus even the overdamped default has a signed memory entry. This describes
feedback from past phase deviations in a chosen tangent chart, not alternating
physical attraction, a receiver trajectory reversal or an autonomous bond.
Pure-diffusion positive-kernel results cannot be transferred to this joint law.

### A prospective onset discriminator in the actual nonlinear field

Prepare only donor form `u_0(0)=epsilon!=0`; all other form deviations and all
phase deviations, including the mediator's, are zero. Direct differentiation
at this preparation gives, for the mediated receiver,

\[
\dot u_5(0)=\dot v_5(0)=0,\qquad
\ddot u_5(0)=\epsilon\mu\nu
 \left[\frac{e^2}{6}-\frac{w^2}{2\beta\pi^2r}\right],\qquad
\ddot v_5(0)=-\frac{\epsilon\mu\nu ew}{2\beta\pi r}.
\]

These initial derivatives are exact for the nonlinear ideal field as well:
the receiver's initial form gradient is zero, so its phase-metric derivative
does not enter the phase acceleration. The full subsequent memory remains
nonlinear. A direct `0--5` bridge instead gives initial receiver rates
`e*nu*epsilon/3` and `-w*nu*epsilon/(beta*pi*r)`. Removing the intermediary
path leaves the independent receiving ring with no donor response.

The distinction is first-order versus second-order onset, not a finite waiting
time before transmission. The phase contribution competes with diffusion in
the form acceleration: at `e=w=1/2`, `beta=1` it reduces but does not reverse
the positive form signal for `epsilon>0`. The displayed coefficient changes
sign below `beta=3*w^2/(e^2*pi^2*r)`; this is a conditional family boundary,
not a reason to fit beta after a response. Varying only mediator capacity
predicts proportional initial accelerations, including a frozen-mediator null.

### Shared implementation and evidence scope

[`derive_coordinate_memory`](../../src/tnfr/mathematics/linear_observation.py)
centralizes the detached exact block split of any supplied fixed generator
`z_dot=J*z`. It retains visible/hidden coordinate order, the four blocks and
`K(0)=B*C`; it does not assume diffusion, positivity, stability or a memory
cutoff. Rationalized trigonometric entries describe the supplied rational
matrix, not exact real pi or the ideal C5 cosine. `derive_linear_observation`
remains the independent owner of invariant-row closure.

The [mediator controls](../../tests/physics/test_relational_mediation.py) compare
the analytic joint Jacobian and prepared response against the production
field. Static finite-difference checks are complemented by a closed-form
two-step receiver control at `dt=1/32`, distinct from the reserved horizon
below. Neither supplies an ODE error enclosure. Model defaults, ordered nodes `0,...,10`,
unit edges, held capacities, explicit donor amplitude and binary64 arithmetic
are retained in the tests. No stochastic preparation, new event selector or
change to the nodal executor is involved. The mechanism is conditional mediated
interaction with local recovery; primitive support origin and a finite capture
prediction remain separate obligations in the sole execution plan.

<a id="mediator-orientation-scope"></a>
### Orientation sensitivity: winding sign is invisible at a single symmetric port

The [magnetic binding comparison](../PHYSICAL_REGIME_CORRESPONDENCES.md#magnetic-binding-comparison)
asks whether a pattern's orientation changes its interaction. The present
one-port-per-ring geometry has an exact limitation. Reflect the donor ring by
`R=(1 4)(2 3)`, fixing port 0, mediator 10 and the entire receiving ring.
This is an automorphism of the supplied graph. With capacities and all other
node data transformed consistently, the ideal full field is equivariant:

\[
F(Rx,R\theta)=R F(x,\theta).
\]

For the uniform-capacity preparation above, this reflection maps the donor's
positive twist to its negative twist modulo full phase turns, while leaving
a donor-port form impulse and the receiving preparation unchanged. Uniqueness
on the admitted smooth domain then gives `z_minus(t)=R*z_plus(t)`.
Mediator and receiving-ring histories are identical throughout their common
admitted interval. This is a nonlinear symmetry statement, not merely the
fact that the tangent coefficients contain the even quantity `cos(kappa)`.

Consequently, the current experiment cannot distinguish equal from opposite
winding signs using its receiver response. Assigning those signs magnetic
polarities would introduce a physical meaning that this interface cannot
resolve. Winding itself depends on a declared cycle orientation; no spatial
magnetic moment follows from its integer value. Nor does a signed memory
entry establish magnetic attraction or repulsion.

This does not rule out orientation-sensitive TNFR interactions under other
admitted preparations or structures. The existing
[hidden form-orientation control](RELATIONAL_PATTERN_COMPOSITION.md#hidden-form-orientation-changes-the-metrics-next-response)
already shows nonlinear dependence on relative form/phase orientation.
An asymmetric internal preparation, two separately distinguished contacts or
a justified geometric observation can remove the independent reflection
symmetry. Their state, support and observation must be declared before such a
claim is tested; new contacts also require fresh phase-domain admission.
They are not an automatic extension of the single-port result. This comparison
does not replace the finite mediator-capacity prediction in the sole queue.

<a id="finite-mediated-response"></a>
## 10. A finite causal-response test of the effective connection

### Influence has an onset order, not a selected activation time

For the section 9 donor preparation, compare the perturbed receiver with the
unperturbed equilibrium at the same supplied support and capacities. Smoothness
and the exact initial acceleration imply

\[
\Delta\theta_5(t)=
-\frac{\epsilon\mu\nu ew}{4\beta\pi r}\,t^2+O(t^3).
\]

With positive coefficients and nonzero epsilon, this difference is nonzero
for every sufficiently small positive time. The first derivative vanishes
at zero; a finite waiting interval does not follow. A threshold-based time
of detection would be an observation policy, not an autonomous support event.
The unperturbed equilibrium itself generates no signal. At zero mediator
capacity, equal mediator initial states stay equal and the receiving dynamics
are independent of the donor. Removing the fine path also removes influence.

This defines the useful causal question without adding a primitive edge:
does an intervention in one region change another through the retained
intermediary? The memory law answers it conditionally. Joint geometric
recovery has its separate positive-capacity theorem; capture from a distant
preparation, support birth and physical binding do not follow from a nonzero
response alone.

### A frozen mediator can organize both rings without coupling their changes

The zero-capacity control is stronger than an absent signal at equilibrium.
With mediator state `(x_m,theta_m)` held fixed, each ring is a separate system
with one anchored boundary. Its acute reference is uniform form `x_m` and
the same ring twist rotated by `theta_m`. Write `B_g,K_g` for the grounded
form Laplacian and cosine-weighted phase Hessian, including the anchoring edge.
Both are positive definite. On each ring, `A=N*D^-1` and `M=N*H_*^-1`
remain positive diagonal, with degrees including that edge. The local Jacobian
is

\[
J_g=\begin{pmatrix}
-eAB_g&-wMK_g\\ (w/\beta)MB_g&0
\end{pmatrix}.
\]

Storage coordinates transform this into
`[[-D_cal,-C_cal],[C_cal^T,0]]`, where
`D_cal=e*B_g^(1/2)*A*B_g^(1/2)>0` and
`C_cal=(w/sqrt(beta))*B_g^(1/2)*M*K_g^(1/2)` is invertible.
The real-part argument in the existing
[local recovery proof](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
then makes this grounded Jacobian Hurwitz. Each ring has local exponential
recovery to its fixed-boundary reference, without an internal common-offset
freedom. This is a separate anchored theorem, not an extension of the
whole-network strictly-positive-capacity theorem to a zero capacity.

Nevertheless, the two systems consume no changing state from each other.
Perturbing one cannot alter the other when mediator and receiver preparations
are held fixed. Thus common geometry and individual restoration can occur
without mutual influence. A claim of a collective bond must retain this
common-boundary control as well as its geometry and recovery observations.

### A symmetric port prediction has no quadratic amplitude error

Reflect both rings around their ports, fixing nodes 0, 5 and 10, and call the
permutation `R`. Let `Q=diag(R,R)` act on the 22 form/phase deviations from
the ideal reference. Simultaneous form/phase inversion and graph equivariance
give, on the admitted lifted chart,

\[
G(-Qz)=-QG(z),\qquad -R\theta_*=\theta_*\pmod {2\pi}.
\]

The donor direction `a=(e_0,0)` is reflection-even. Each simultaneous Euler
map inherits the symmetry, so its N-step state satisfies
`z_N(-epsilon)=-Q*z_N(epsilon)`. A linear observation `O` of the receiver
port or mediator obeys `OQ=O`. Smooth dependence on the initial amplitude
therefore yields, for a fixed admitted grid,

\[
Oz_N(\epsilon)=\epsilon O(I+hJ_\mu)^Na+O(\epsilon^3).
\]

The full 22-coordinate state need not have cubic error: internal
reflection-odd modes can carry quadratic corrections. This result specifies
the approximation order near zero; it supplies neither a numerical cubic
remainder constant at a chosen amplitude nor an exact-ODE error bound.
Native represented reference phases can have tiny residual drift, which
must be recorded rather than subtracted from the response after evaluation.

### Frozen finite-executor protocol

Use the existing law with `e=w=1/2`, `beta=1`, ring capacities one,
donor form `epsilon=1/64`, and otherwise the section 9 equilibrium. Fix mediator
capacities `mu=0,1,2`, horizon `T=1/4`, and 64 and 128 simultaneous Euler
steps. These two grids are numerical controls of the same preparation, not
independent physical trials or validated continuous-time enclosures.

The memory predictor advances its exact tangent blocks with old-state values:

\[
y_{n+1}=y_n+h(Ay_n+Bh_n),\qquad
h_{n+1}=h_n+h(Cy_n+Dh_n).
\]

The hidden initial deviation is zero in this preparation. The memory-omitting
control retains `y_{n+1}=y_n+hAy_n`; its receiving ring remains at zero.
It is a deliberate ablation, not a proposed closed law for arbitrary states.
For positive mediator capacity, a stronger instantaneous comparison sets
`0=Cy+Dh`, hence `h=-D^-1*C*y=(z_0+z_5)/2`. Its visible generator is
`A-B*D^-1*C`. Both the capacity factor and its inverse cancel, so this
approximation predicts identical responses at `mu=1` and `mu=2` on every
matched grid. It also replaces the prepared zero hidden state with its
instantaneous equilibrium and need not match the initial onset. It is not
an exact transient reduction and is undefined by this inverse at `mu=0`.
The capacity intervention therefore discriminates retained memory from a
fixed instantaneous connection even when their integrated kernel agrees.
Observe the receiver port pair `(x_5,theta_5-theta_{*,5})`, retaining the full
state and mediator endpoint as additional evidence.

Before native execution, freeze each tangent prediction and its source/runtime
provenance. The prospective decision requires positive-capacity endpoint and
`mu=2` minus `mu=1` intervention errors in maximum norm no greater than 1%
of their respective predicted signals plus `1e-12`. Each predicted signal
must exceed `100e-12`; the frozen-mediator receiver must stay within `1e-12`.
Also require held support/capacities, the declared fresh-pressure path,
acute margin at least `pi/20`, and actual work-balance residual at most `1e-12`.
These are finite experiment decision tolerances, not emergent constants.
A pass would test the derived interaction without fitting an edge or kernel;
it would not promote the tolerance to a proved nonlinear or solver error bound.

### Reserved response and decision

The [mediator instrument](../../benchmarks/relational_mediation_response.py)
uses the shared coordinate-memory decomposition and Euler arithmetic for the
prediction, and `step_relational_exchange` for the full nonlinear response.
The [prediction](../../docs/assets/relational_mediation_response/result.prediction.json)
and [source archive](../../docs/assets/relational_mediation_response/result.sources.zip)
were frozen before the [reserved response](../../docs/assets/relational_mediation_response/result.json).
All declared gates passed on both grids without changing the preparation,
law, coefficients or decision tolerances.

At 128 steps, the receiver results were:

| Mediator capacity | Predicted form | Observed form | Predicted phase deviation | Observed phase deviation | Pair maximum error |
| --- | --- | --- | --- | --- | --- |
| 1 | `1.4859487561e-5` | `1.4859487442e-5` | `-1.0748178383e-5` | `-1.0748178470e-5` | `1.18910e-13` |
| 2 | `2.8689674324e-5` | `2.8689674024e-5` | `-2.0729626866e-5` | `-2.0729627404e-5` | `5.37972e-13` |

The capacity intervention had predicted maximum norm `1.3830186762e-5`
and observed norm `1.3830186582e-5`; its pair error was `4.51169e-13`.
Instantaneous elimination predicted exactly zero intervention difference.
The omitted-memory control predicted zero receiver response. The frozen
mediator's observed receiver norm stayed below `2.41e-18` on both grids,
consistent with its null prediction and materialized reference drift.

These values describe the retained finite executor comparison. In particular,
the largest positive-capacity receiver error relative to its predicted pair
was `1.88e-8`, but full-state tangent errors reached `7.22e-7`, consistent
with retaining a different approximation scope for hidden internal shape.
The largest 64-to-128-grid full-state difference was `1.884e-6`; this is
not a continuous-time error certificate. No further refinement or amplitude
campaign is needed for the frozen decision.

The minimum observed acute margin was `0.31220729`, the largest instantaneous
work-balance residual was `2.51e-18`, and the largest Euler storage-step
defect was `5.52e-9`. These are different quantities. The static rounded
reference had maximum rate `1.95e-17`; that drift was not subtracted.
Execution used Python 3.13.6, NumPy 2.5.3, NetworkX 3.6.1, Windows AMD64,
binary64 states, ascending node order, sorted unit edges and no randomness.
The archive retains all 606 TNFR Python source files plus this producer;
dependency versions are recorded separately. The
[record tests](../../tests/physics/test_relational_mediation_response.py)
check frozen evidence without rerunning its trajectories.

This closes the bounded F4 question: a derived intermediary memory predicts
a finite response and its capacity intervention better than the stated
memory-omitting controls. It establishes neither a new primitive edge nor
a unique law of physical binding. A ring's response can be causally linked
to another without a direct edge, while the fine paths carrying that
interaction remain premises of the model.
