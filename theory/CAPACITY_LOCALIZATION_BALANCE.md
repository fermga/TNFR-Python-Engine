# Capacity-conditioned EPI balance on a canonical cycle

**Status:** Exact fixed-capacity balance and relaxation, a conditional local
form/capacity classification, flow/event admission criteria, and finite
canonical checks. Independent selection of the capacity law and general
localization remain open.
**Research links:** B2.d.2/O3.a, S3, S8, S9 and S16.

## 1. The canonical channels and the restricted preparation

Consider a fixed simple unit-conductance cycle. Let `L=I-W/2` be its normalized
Laplacian, `B=2L` its conductance Laplacian, `x` its real scalar EPI field and
`nu_i>0` its structural capacities. Keep a regular phase twist whose absolute
adjacent gap is strictly below both the effective U3 gate and `pi/2`, so the
neighbor phasor resultant points along the local phase. Its phase pressure is zero,
as shown in [Coupling winding persistence](COUPLING_WINDING_PERSISTENCE.md).
The cycle's uniform degree makes topology pressure zero as well.

The remaining canonical gradients are the neighbor mean differences already
implemented by [dnfr.py](../src/tnfr/dynamics/dnfr.py). Writing the effective
normalized channel coefficients as `e=w_epi>0`, `f=w_vf>=0`, the nodal equation
becomes

$$
\Delta\mathrm{NFR}=-eLx-fL\nu,\qquad
\dot x=\operatorname{diag}(\nu)\Delta\mathrm{NFR}.
$$

The full four-channel normalization remains in force even when two channel
values vanish; the EPI/capacity pair is not renormalized by itself. The
capacity profile and the phase twist are held fixed during this restricted EPI
flow. Clipping, changing support, an independent phase advance and intervening
capacity-changing operators are separate dynamics. No external potential or
new restoring coefficient has been introduced.

## 2. A shifted structural coordinate gives the exact balance

Set `r=f/e` and define

$$
y=x+r\nu,\qquad H=\operatorname{diag}(2/\nu_i).
$$

Fixed capacity gives the exact homogeneous equation

$$
\dot y=-e\operatorname{diag}(\nu)Ly.
$$

The capacity-pressure channel has been retained in the structural coordinate
`y`, not discarded. In particular, `sum_i H_i*x_i` is conserved, because
`H*diag(nu)*L=B` and the rows and columns of `B` sum to zero. If `x0` is the
initial EPI, let

$$
c=\frac{\sum_i H_i(x_{0i}+r\nu_i)}{\sum_i H_i},\qquad
x_i^*=c-r\nu_i.
$$

The connected cycle has only the constant null mode of `L`. Thus `x*` is the
unique zero-pressure profile with the conserved weighted EPI total selected
by `x0`. All capacities are positive, so rate stationarity and zero pressure
coincide in this fixed model. This implication changes at zero capacity.

Let `z=x-x*`. Its inherited weighted deviation energy satisfies

$$
V=\tfrac12z^\top Hz,\qquad
\dot V=-e z^\top Bz
=-e\sum_i(z_{i+1}-z_i)^2\leq0.
$$

Since `sum_i H_i*z_i=0`, the only zero-dissipation deviation is zero. More
explicitly, write `nu_min=min_i nu_i` and
`lambda2=1-cos(2*pi/n)`. Weighted mean minimization and the cycle Poincare
inequality give

$$
z^\top Bz\geq\nu_{\min}\lambda_2 z^\top Hz,
\qquad V(t)\leq e^{-2e\nu_{\min}\lambda_2t}V(0).
$$

This proves relaxation to the capacity-conditioned profile within the exact
held-field model, not attraction for general multichannel operator schedules.

A second diagnostic is the shifted Dirichlet energy

$$
F=\tfrac12y^\top By,\qquad
\dot F=-e(By)^\top\operatorname{diag}(\nu_i/2)(By)\leq0.
$$

The ordinary EPI Dirichlet energy `x^T*B*x/2` can increase: its gradient is
only one of the two pressure contributions. For example, with
`e=f=1/2`, `nu=(1/2,1,1,1)` and `x=(3/4,1/2,1/2,1/2)`, its derivative is
`1/16`, while both `V'` and `F'` are `-1/16`. This is structural redistribution
under the existing capacity channel, not a violation of the derived balance.

## 3. What a capacity dip produces, and what it does not establish

For initially uniform EPI `x0=b*1`, define the harmonic capacity
`nu_h=n/sum_i(1/nu_i)`. Then

$$
x_i^*=b+r(\nu_h-\nu_i),\qquad
x_i^*-x_j^*=-r(\nu_i-\nu_j).
$$

A capacity dip produces a higher equilibrium EPI at the same location; a
capacity peak produces a lower one. This contrast is fixed by the prepared
structural capacity, not by an independently fitted confinement law. With
uniform capacity the equilibrium is uniform, so an EPI bump alone disperses
under the same positive-capacity flow.

One canonical preparation makes this quantitative. Start with capacity one
and uniform EPI, then apply a single Silence operation at a selected node.
With the ideal repository formulas, its retained capacity is
`q=1-1/(4*pi)`, while the default channel ratio is `r=1/pi`. The subsequent
held-capacity equilibrium has core-minus-background contrast

$$
r(1-q)=\frac1{4\pi^2}\simeq0.0253303.
$$

This contrast is independent of cycle size; the common equilibrium level
depends on the harmonic capacity. Runtime observations instead use the exact
ratio of materialized normalized channel coefficients and the represented
Silence factor. They do not assert bitwise equality with the symbolic formula.

The selected node is a declared structural preparation. As in the existing
[pointed symmetry analysis](../src/tnfr/physics/pointed_symmetry.py), its location
breaks translation symmetry through that selector, not spontaneously. Once
prepared, the EPI contrast develops under the autonomous restricted nodal
flow without continuing external forcing. Its persistence is conditional on
retaining the capacity profile and the other hypotheses. It is not evidence
for an independently generated or universally stable localized entity.

Capacity acts through both nodal levers here. Multiplying a heterogeneous
capacity profile by `a` scales the EPI-pressure rate contribution by `a` and
the capacity-pressure contribution by `a^2`. It also changes the predicted
equilibrium contrast. It is therefore not merely a rescaling of time when
the capacity channel is active and the profile is nonuniform.

## 4. Canonical capacity updates generally release the static profile

The same canonical operators that prepared or synchronize the network can
change its conditional balance. Suppose `x=c-r*nu` initially and preserve
the regular phase twist and unit cycle support. Use target-only UM
(`UM_BIDIRECTIONAL=False`) without functional links
(`UM_FUNCTIONAL_LINKS=False`). Its immutable all-target stage with the
configured capacity synchronization factor has

$$
\nu^+=(I-sL)\nu,\qquad s=\texttt{UM\_vf\_sync}.
$$

UM leaves EPI unchanged. After the canonical pressure is refreshed,

$$
\Delta\mathrm{NFR}^+=f s L^2\nu.
$$

Similarly, uniform Silence attenuation `nu^+=q*nu` gives

$$
\Delta\mathrm{NFR}^+=f(1-q)L\nu.
$$

For a nonconstant capacity profile and positive `f`, these expressions are
generally nonzero. A direct UM pressure write or its local stabilization
telemetry must not replace this refreshed channel sum. The fixed-capacity
profile is not forward invariant under the default capacity-changing stages.
UM's capacity synchronization itself uses the same cycle diffusion map and
smooths the supporting capacity contrast under its own exact fixed-gate
hypotheses.

For continuously varying capacity, even the shifted coordinate carries an
additional term:

$$
\dot y=-e\operatorname{diag}(\nu)Ly+r\dot\nu.
$$

The fixed-capacity conservation law and Lyapunov metric cannot be transferred
without accounting for that forcing and the changing metric. This term is a
consequence of the coordinate definition, not a new physical force.

### A predeclared capacity-law discriminator

The held-capacity model is mathematically closed when its omitted rows are
stated as `nu_dot=0`, `theta_dot=0` and fixed support. Its differentiated
equilibrium is a valid conditional result. The missing justification concerns
these constitutive premises and the origin of the conserved capacity profile;
closure, restoration of the supporting state and autonomous formation are
different claims.

One bounded comparison exposes the specific capacity premise without changing
the pressure formula or fitting a sustaining source. Keep a fixed even unit
cycle, equal zero phases, absent Gamma, and effective coefficients `e,f>0`.
Compare held capacity with the declared alternative `nu_dot=-epsilon*L*nu`,
`epsilon>0`. This uses the same support-averaging direction as section 4's
UM capacity map. Its continuous clock and rate are additional comparison
premises, not a continuous-limit theorem for default UM or an installed engine
law. Work in one fixed nondimensional structural clock; dimensional epsilon
would have inverse-clock units. Other zero pressure channels remain in the
declared full normalization.

For the checkerboard mode `v_i=(-1)^i`, `Lv=lambda*v`, `lambda=2`, write
`nu=a+c(t)*v`, `x=b(t)*1+d(t)*v`, with `a>|c_0|>0`. Multiplication by
capacity produces a mean term as well as a contrast term. The exact nodal
projection is

\[
\dot c=-\epsilon\lambda c,\qquad
\dot d=-a\lambda(ed+fc),\qquad
\dot b=-\lambda c(ed+fc)=\frac ca\dot d.
\]

Start at the same zero-pressure preparation `d_0=-(f/e)*c_0` in both
models. Let `alpha=e*a*lambda`, `beta=epsilon*lambda`. Then

\[
c(t)=c_0e^{-\beta t},\qquad
d(t)=d_0\frac{\alpha e^{-\beta t}-\beta e^{-\alpha t}}{\alpha-\beta}.
\]

At `alpha=beta`, the continuous extension is
`d(t)=d_0*(1+alpha*t)*exp(-alpha*t)`. For every `epsilon>0`, both
contrasts vanish, capacity stays bounded below by `a-|c_0|`, and

\[
b_\infty-b_0=\frac{f c_0^2}{2(ea+\epsilon)}.
\]

This follows by integrating `b_dot=(c/a)*d_dot`; fixing the EPI mean would
discard actual nodal dynamics. At exactly `epsilon=0`, the initial EPI and
capacity instead remain unchanged. Consequently

\[
\lim_{\epsilon\downarrow0}\lim_{t\to\infty}d(t;\epsilon)=0,
\qquad
\lim_{t\to\infty}\lim_{\epsilon\downarrow0}d(t;\epsilon)=d_0\ne0.
\]

The mean limits differ too: the former is `b_0+f*c_0^2/(2*e*a)`, the
latter `b_0`. The new information is this singular dependence on the
conserved-capacity premise, not another proof that mixing can erase contrast.
The [segmented mixing result](CYCLE_SUPPORT_DYNAMICS.md#5-positive-mobility-and-genuine-mixing-give-a-different-boundary)
already establishes that latter boundary under its own reset/flow hypotheses.
Weak capacity averaging can imitate the held preparation on a finite horizon
without inheriting its infinite-time contrast.

Both models have identical initial triad, zero pressure and zero EPI rate.
The released model nevertheless has
`p_dot(0)=f*epsilon*lambda^2*c_0*v` and
`x_ddot(0)=f*epsilon*lambda^2*c_0*(a*v+c_0*1)`; the held model has zero
acceleration. These are prospective consequences of the chosen capacity row,
not a retrospective pressure reconstruction.

**Fixed control.** With `e=f=1/2`, `a=1`, `c_0=1/4`, `b_0=1/2` and
`epsilon=1/4`, put `r=exp(-t/2)`. The released prediction is
`c=r/4`, `d=r^2/4-r/2`,
`b=1/2+(1-r^2)/16-(1-r^3)/24`.
For `t>=0`, its capacities stay in `[3/4,5/4]` and EPI in `[1/4,3/4]`;
the limiting EPI is the uniform value `25/48`. The held model keeps its
initial contrast `d=-1/4` and mean `1/2`. The
[existing boundary test owner](../tests/physics/test_capacity_pressure_boundaries.py)
checks the exact projection, solution, resonance and limit distinction, plus
fresh production pressure and a shared Euler probe at a predeclared analytic
snapshot. That probe is not an executed full trajectory or a physical test.
Neither this comparison nor its outcome independently selects either
capacity law or demonstrates generation of the supporting state.

## 5. Diminishing capacity can retain contrast by exhausting the nodal clock

If capacity is spatially uniform during each EPI segment, its own pressure
gradient is zero. Let each segment have duration `h` and capacity
`nu_k=nu0*q^k`, as in repeated uniform canonical Silence attenuation with
`0<q<1`. The total capacity exposure is finite:

$$
\Theta_\infty=h\sum_{k\geq0}\nu_0q^k=\frac{h\nu_0}{1-q}.
$$

Exact continuous diffusion on these piecewise constant segments multiplies
a Laplacian mode of eigenvalue `lambda>0` by
`exp(-e*lambda*Theta_infinity)>0`. Nonzero initial contrast therefore need
not vanish. Its rate tends to zero because capacity tends to zero; its
pressure can remain nonzero. This is retention through a finite accumulated
reorganization budget, not a restoring mechanism or spontaneous confinement.
The formula concerns the exact segmented flow. A finite Euler trajectory
has its own product of modal factors and numerical error.

At exactly zero capacity, stationarity is weaker still. A node freezes even
when its pressure is nonzero, and `H_i=2/nu_i` is unavailable. If every
capacity vanishes, any EPI field is stationary although `-e*L*x` need not
vanish. With some frozen nodes, the active part instead has a Dirichlet
problem for `y`, with the frozen values as boundary data. Different frozen
values can support residual pressure at those nodes. The executable balance
in this note deliberately requires strictly positive capacity.

## 6. The common-walk condition is essential

The canonical implementation averages EPI with edge conductances but averages
capacity over the unweighted unique neighbor support. The two walks coincide
on the unit cycle considered here; they need not coincide on a weighted
graph. With distinct walks `L_E,L_0`, the pressure is
`-e*L_E*x-f*L_0*nu`, and substituting `x=c-r*nu` leaves
`f*(L_E-L_0)*nu` rather than zero.

Even regular weighted strength is insufficient: a four-cycle with alternating
edge weights `1,3`, capacity `(1,2,3,4)`, `x=5-nu` and `e=f=1/2` has residual
pressure `(-1/4,-1/4,1/4,1/4)`. For a more general graph, the weighted total
can drift as well. On a three-node path with edge weights `1,3`, strengths
`d_E=(1,4,3)` and capacity `(1,2,4)`,
`d_E^T*L_0*nu=3`; consequently the derivative of
`sum_i (d_Ei/nu_i)*x_i` is `-3*f`. No stationary unclipped two-channel state
exists for that fixed forcing when `f>0`. These controls prevent a false
extension of the cycle theorem.

## 7. Executable evidence and the next structural extension

[`capacity_localization.py`](../src/tnfr/physics/capacity_localization.py)
observes exact rational supplied cycle coordinates, the shifted equilibrium,
pressure/rate, weighted conservation and both energy balances. It uses the
shared exact-or-represented real reader and the existing full-channel default
normalization. General positive explicit coefficient pairs are formal
algebraic inputs; actual normalized canonical coefficients and live graph
conditions require separate evidence. No clipping interval, observed phase
loop or operator history is inferred from the detached vectors.

[Exact tests](../tests/physics/test_capacity_localization.py) cover pressure
cancellation, capacity dips, uniform-capacity dispersion, the two-lever
scaling, and EPI-energy growth with decreasing shifted energies.
[Boundary tests](../tests/physics/test_capacity_pressure_boundaries.py) exercise
the weighted-walk mismatch, actual UM/Silence refresh effects and the declared
continuous capacity-law comparison in section 4.
[Clock tests](../tests/physics/test_capacity_clock_retention.py) connect actual
uniform Silence attenuation to the finite-exposure obstruction using existing
rational exponential bounds.
[Runtime tests](../tests/physics/test_capacity_localization_runtime.py)
and the [benchmark](../benchmarks/capacity_localization.py) retain the actual
canonical preparation, pressure-refreshed solver partitions and finite
residuals. These are distinct from a future binary64 or complete-runtime
stability theorem.

The zero-winding control shares the exact prepared two-channel model; its
binary64 endpoints need not equal those of represented winding phases. The
runtime test accounts for their difference with the captured remaining-channel
pressure defect `delta_p` and Euler residual `epsilon`. For each recorded step,
`d=h*diag(nu)*delta_p+epsilon`. The common matrix
`M=I-h*e*diag(nu)*L` has nonnegative entries and unit row sums on these admitted
steps, so the exact discrepancy satisfies
`Delta_next=M*Delta+d_zero-d_winding` and
`||Delta_next||_inf <= ||Delta||_inf+||d_zero-d_winding||_inf`.
This is a finite bound derived from the retained residuals, not an arbitrary
equality tolerance or a prediction of unobserved numerical errors.

The change of coordinate also reuses the existing
[heterogeneous diffusion geometry](../src/tnfr/physics/structural_diffusion.py):
the reference coordinate `y` has effective capacity `e*nu`. Its exact-model
convergence and Dirichlet identities therefore need no new transport law;
identifying a particular materialized runtime generator remains separate.

A further identity suggests how to study localized phase structure without
introducing another primitive. Write `phi_i=2*pi*W*i/n+p_i` with a periodic
phase perturbation `p` and a consistent lift. Require every lifted gap
`2*pi*W/n+p_{i+1}-p_i` to have absolute value strictly below both the effective
U3 gate and `pi/2`. The canonical phase channel is then `-L*p/pi`.
If both this derived coordinate and capacity are
held fixed, the combined pressure is
`-L*(e*x+f*nu+(w_phase/pi)*p)`. The corresponding shifted coordinate is
`x+(f/e)*nu+(w_phase/(pi*e))*p`. A target-only all-node UM stage without
functional links on this fixed cycle evolves `p` with its phase gap
coefficient and evolves `nu` with its separate capacity
coefficient, so this more general balance is likewise conditional on what
structural fields are retained. That algebraic extension is developed in the
joint support/release study linked below; it is not implemented by the
capacity-only observer or a claim of spontaneous physical localization.
The separate [joint cycle observer and proof](CYCLE_SUPPORT_DYNAMICS.md)
now use the normalized coordinate `a=p/pi`, derive the reset/flow energy
budget and distinguish default-factor finite-clock retention from an active
zero-pressure equilibrium. Both observers reuse one private exact cycle
algebra; their hypotheses and public read-outs remain separate.
Neither note opens a new cycle campaign; the
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns resumption.

## 8. A local form-capacity relation: dissipation and restoration criteria

Section 4 leaves primitive capacity evolution undetermined. A different,
explicit constitutive premise removes capacity as an independent state:
`nu_i=g(x_i)`, with one positive `C1` function `g` on an open scalar interval
`I`. This is a conditional class to test, not a relation derived by the nodal
identity or installed in the engine. Keep the fixed connected unit cycle,
equal held phases, absent Gamma, fixed effective `e,f>0` and no events/clipping.
Only preparations satisfying this same relation are admitted. Write
`D=2I_n`, `B=DL`, and apply scalar functions componentwise. The unchanged
pressure formula and the chain rule give

$$
h(s)=es+fg(s),\qquad p=-Lh(x),\qquad
\dot x=-\operatorname{diag}(g(x))Lh(x),\qquad
\dot\nu_i=g'(x_i)\dot x_i.
$$

The capacity rate now follows from the supplied relation; it is not an
independently fitted restoring input. Equal phases and regular support make
the other pressure channels zero without changing their normalization.
Weighted conductances with a different capacity walk, changing support,
phase response and arbitrary independently prepared capacities are outside
this class. The same function `g` belongs to the declared form chart and clock;
a chart change must transform that relation too.

### Conserved coordinate and exact dissipation

Choose any reference point in `I` and define the following primitives:

$$
q(s)=\int^s\frac{du}{g(u)},\qquad
Q(x)=\sum_i d_i q(x_i),\qquad
V(x)=\sum_i d_i\int^{x_i}\frac{h(u)}{g(u)}\,du.
$$

Their additive constants have no dynamical effect. Since `g>0`, `q` is an
invertible scalar coordinate and `dot q=-Lh(x)`. Symmetry of `B` gives

$$
\dot Q=0,\qquad
\dot V=-h(x)^T B h(x)
       =-\sum_{\{i,j\}\in E}(h(x_i)-h(x_j))^2\leq0.
$$

This is a derived balance for the conditional reduced law, not an assertion
about the tetrad energy or a full joint variational completion. The conserved
quantity is generally neither total capacity nor arithmetic EPI mean.
Stationarity is exactly `h(x_i)=c` for every node, by positive capacity and
connectedness. Thus an injective `h` admits only uniform stationary form.

If `h` is strictly increasing on `I`, maxima cannot grow and minima cannot
decrease: the initial EPI hull is forward invariant. Its compactness inside
`I` gives a positive lower capacity bound and a global solution. On this hull,
`V` is bounded below, and its zero-dissipation set consists of uniform states.
The conserved `Q` selects precisely one of them. Consequently every such
solution converges to the uniform value `a` determined by
`q(a)=sum_i d_i*q(x_i(0))/sum_i d_i`. No exponential rate is asserted when
the increasing function has vanishing derivative.

### The response slope distinguishes decay, growth and mere freezing

At a uniform state `x=a*1`, differentiation of the full nodal row gives

$$
J=-g(a)\,[e+fg'(a)]L.
$$

The derivative of the mobility multiplies zero equilibrium pressure and
therefore drops out here. Each nonuniform Laplacian mode has linear rate
`-g(a)*(e+f*g'(a))*lambda`. Positive `e+f*g'(a)` gives local exponential
decay of contrast; a negative value gives local growth. Zero gives only
linear degeneracy. The threshold `g'(a)=-e/f` follows from the two retained
pressure coefficients, not from telemetry or a new selected safety margin.

If `h` is constant throughout `I`, equivalently `g(s)=(c-e*s)/f` on its
positive domain, every admissible field has zero pressure and freezes.
Choosing that cancellation would preserve every preparation without restoring
any particular one. Cancellation alone cannot explain identity selection.

### A sufficient conditional criterion for restoring differentiated form

Suppose an independently specified `g` has an equilibrium `x*` with a common
value `h(x_i*)=c` but nonuniform form. Assume all occupied slopes
`s_i=h'(x_i*)` are strictly positive, and put `g_i=g(x_i*)`, `S=diag(s_i)`.
The exact linearization and positive metric obey

$$
J_*=-\operatorname{diag}(g_i)L S,\qquad
R=\operatorname{diag}(d_i s_i/g_i),\qquad R J_*=-S B S.
$$

Its single null direction is `S^-1*1`. Because `R*S^-1*1=grad Q(x*)`, the
tangent space to `Q=Q(x*)` is precisely the `R`-orthogonal complement of that
direction. All eigenvalues there are strictly negative. The continuously
differentiable reduced flow restores `x*` locally and exponentially for small
perturbations on that same conserved-Q surface. Perturbations changing `Q`
cannot return to exactly `x*`. Positive occupied slopes are a sufficient
criterion, not a classification of every possible stable equilibrium.

For two distinct occupied values `a<b` with the same `h` and positive slopes
at both, the mean value theorem forces `h'<0`, hence `g'<-e/f`, somewhere
between them. This negative response need not occur at an occupied node or
at the initial uniform state. A strictly increasing response cannot provide
this branch. The condition does not select a function, prepare those values,
prove their formation or choose a preferred arrangement on the cycle.

The [constitutive capacity controls](../tests/physics/test_constitutive_capacity_scope.py)
check these balances and linearizations, and compare affine decay, cancellation
and initial growth with fresh production pressure and the shared joint-response
owner. The probes use positive capacities in a declared finite chart; they
neither execute this whole coupled flow nor supply a physically justified `g`.
This gives an admission criterion for a proposed form/capacity mechanism while
leaving its independent constitutive origin open.

<a id="capacity-law-admission"></a>
## 9. Capacity-law admission: a state relation must survive flow and events

Section 8 classifies a **supplied** local relation; it does not establish that
the engine preserves that relation. The admission question is whether one
complete declared law stays on
`M_g={nu_i=g(x_i) for every i}`. Keep a fixed finite support and conductances,
scalar form, one declared clock and fixed pressure coefficients. On a regular consumed
phase chart, let `P(x,theta,nu)` be the prospective configured pressure and
write the continuous completion as

\[
\dot x_i=\nu_iP_i,\qquad \dot\theta_i=\Omega_i,
\qquad \dot\nu_i=A_i.
\]

No Gamma, clipping or event is included in these continuous rows. The local
function `g` is fixed and `C1` on the stated form domain; its parameters do
not change silently after a response is observed. A relation depending on
phase, neighbors or history would need the corresponding additional chain
rule terms and is a different reduction.

### Continuous and event tests have different meanings

Differentiate the actual constraint, not the nodal identity alone:

\[
R_i:=\frac{d}{dt}(\nu_i-g(x_i))
       =A_i-g'(x_i)\nu_iP_i.
\]

For a locally Lipschitz complete vector field, `R=0` at **every** admitted
point of `M_g` is the tangency condition for its local invariance, for as long
as the solution stays in the stated domain. Indeed, solve the restricted
form/phase rows with `nu=g(x)`; the chain rule supplies exactly the full
capacity row, and uniqueness identifies the two solutions. Conversely an
invariant differentiable trajectory has zero constraint derivative. A zero
residual at one preparation alone is not an invariance theorem. In particular,
held capacity (`A=0`) is compatible only where `g'(x_i)nu_iP_i=0`; preparing
the graph initially does not make a generic held-capacity flow remain on it.

For an admitted operator or adaptation event, use its actual endpoints:

\[
J_i:=\nu_i^+-g(x_i^+).
\]

The same constrained model admits that event only if `J=0`. A finite jump is
not a derivative, and dividing it by an invented duration cannot supply
`A`. Changing `g` at the event would instead declare a switched constitutive
model with a separate transition rule. Hybrid invariance requires both the
continuous tangency condition and preservation by every realized event.

### Capacity contributes to pressure as well as mobility

Let `G_W` denote the conductance-weighted neighbor difference and `G_U` the
unique-support neighbor difference, with zero rows at isolates. On fixed
support and conductances the implemented mixture has the form
`P=e*G_W*x+f*G_U*nu+w_phase*g_phase(theta)+fixed_topology_source`.
The complete derivative on its regular phase chart is

\[
\dot P=eG_W(\nu\odot P)+fG_UA
       +w_\phi Dg_\phi(\theta)\Omega,\qquad
\ddot x=A\odot P+\nu\odot\dot P.
\]

The shared [joint-response owner](../src/tnfr/physics/phase_response.py)
already evaluates these contributions along supplied rates. Setting
`nu=g(x)` requires `A=g'(x)*(nu*P)` in **both** places; retaining only its
mobility contribution drops the activated capacity-pressure derivative.
Section 8's `p=-L(ex+fg(x))` is the common-walk specialization, not permission
to replace `G_U` by `G_W` on arbitrary weighted support.

### What the actual capacity writers preserve

The [native adaptation owner](../src/tnfr/dynamics/adaptation.py) reads stored
pressure and Si and counts consecutive qualifying calls. Eligibility means
`abs(p)<=EPS_DNFR_STABLE`, `Si>=si_hi` for `VF_ADAPT_TAU` calls. Its
immutable-snapshot update has the ideal-real form

\[
x^+=x,\qquad \nu^+=\nu+\mu E G_U\nu,
\quad \mu=\texttt{VF_ADAPT_MU}\in[0,1],
\]

where `E` is the diagonal eligibility mask. Thus, from `M_g`,
`J=mu*E*G_U*g(x)`. The actual arithmetic additionally retains represented
means and updates inside their incoming capacity hull. It reads neither
`g` nor an elapsed duration and does not refresh pressure or Si. A fresh
gate therefore requires pressure refresh, then Si refresh, then adaptation;
its counters and stored gate inputs belong to the operational state. The
gate and invocation schedule remain configured policies. Uniform capacity
is an exact represented fixed point, including partial eligibility, as
checked by the [existing adaptation controls](../tests/test_structural_stability_adaptation.py).

Other actual writers have separate event obligations:

| Execution path | Capacity/form action relevant to `M_g` |
| --- | --- |
| [SHA proposal](../src/tnfr/operators/al_sha_stage_proposals.py) | `nu+=q*nu`, `x+=x`; hence `J=(q-1)g(x)` at the affected node. Latency metadata does not repair it. |
| [Target-only UM](../src/tnfr/operators/_coupling_stage_kernel.py), with functional links disabled | Capacity blends toward U3-compatible neighbor capacities while EPI is retained. The same graph-law condition uses that compatible-neighbor mean; section 4 already proves the cycle profile release. |
| [VAL/NUL proposal](../src/tnfr/operators/_scale_operator_kernel.py) | Capacity is multiplied by its admitted factor. With edge awareness disabled EPI is retained; when enabled, use the actual scaled/bounded EPI endpoint in `J`. NUL's inverse stored-pressure write preserves an ideal instantaneous product, not the relation `nu=g(x)` or refreshed pressure. |

These are conditional event comparisons, not a claim that every graph or
operator word passes the separate live/grammar admission. No operator is
modified here to impose an otherwise unjustified `g`.

### One exact live-policy witness and its zero boundary

Use unit P2, held equal phases, absent Gamma and declared normalized
`(phase,EPI,capacity,topology)=(0,1/2,1/2,0)`. In one fixed chart take
`g(x)=x`, initially on its positive domain, and

\[
x=\nu=(5/8,3/8),\quad P=(-1/4,1/4),\quad
\dot x=(-5/32,3/32).
\]

The pressure is freshly computed from the two existing channels, not from
the observed rate. A tangent continuous capacity row is `A=dot x`; a held
row instead gives `R=(5/32,-3/32)`. For the actual adaptation event set
`VF_ADAPT_TAU=2`, `VF_ADAPT_MU=1/4`, `EPS_DNFR_STABLE=1/2` and `si_hi=0`.
These explicit policy settings admit the freshly computed Si without
replacing it by a favorable fixture value. After two qualifying calls from
zero counters, with no intervening evolution,

\[
x^+=x,\quad \nu^+=(9/16,7/16),\quad J=(-1/16,1/16).
\]

Stored pressure is unchanged by the event, while a subsequent pressure
refresh gives `P^+=(-3/16,3/16)`. This is a capacity-source change at fixed
form, not a pressure-refresh error. The dyadic preparation exposes the
different declared laws without a trajectory sweep or a fitted coefficient.

Section 8 requires strictly positive `g`. For a **separate** nonnegative
boundary extension `g(x)=x` on `x>=0`, take `x=nu=(0,1/2)` with the same
pressure and event settings. Initially `P=(1/2,-1/2)` but `dot x_0=0`.
Every finite `C1` local graph-law completion has `A_0=g'(0)*nu_0*P_0=0`
there. This only uses the unforced row; it does not assert zero pressure or
global equilibrium. The native event instead gives `nu^+=(1/8,3/8)` at
unchanged form. After refresh `P^+=(3/8,-3/8)`, so the formerly frozen node
has rate `3/64`. Nonnegative capacity permits this reactivation in an
independent event model; it contradicts invariance of this particular local
graph law, not the nodal equation. The positive-capacity `1/g` charge and
metric from section 8 do not extend through zero by this argument.

The [constitutive capacity controls](../tests/physics/test_constitutive_capacity_scope.py)
retain the tangency and event distinctions through the production pressure,
Si and adaptation owners. They test the declared finite preparation and
separate exact-algebraic conclusions from general continuous existence.

### Admission must transform with the form chart and clock

For `y=a*x+b`, `tau=c*t`, with `a,c>0`, the same constrained model has

\[
\widetilde\nu=\nu/c,\qquad
\widetilde g(y)=g((y-b)/a)/c,\qquad
\widetilde P=aP,\qquad \widetilde A=A/c^2.
\]

Consequently `R_tilde=R/c^2` and `J_tilde=J/c`; zero admission defects are
coordinate-independent. The capacity-pressure coefficient transforms as
`f_tilde=a*c*f`, and every other channel coefficient, bound and gate must
follow the existing [joint-unit rule](NODAL_PARAMETER_FOUNDATIONS.md#3-joint-changes-of-form-and-time-units).
Keeping `g(y)=y` after shifting the form origin would change the model, even
where the unrestricted pressure admits common form-offset symmetry. Common
`g` is compatible with node relabeling; a node-specific family would have
to be relabeled together with the state.

A nonlinear/state-dependent clock also differentiates its rate, as proved
by the [full-state covariance owner](NODAL_PARAMETER_FOUNDATIONS.md#pressure-clock-full-state-closure).
It can turn a local autonomous `g` into a time- or joint-state-dependent
relation. Treating capacity as an activity clock at its zero boundary is
singular; invocation counts still do not become elapsed time. No choice of
units selects a capacity law.

**Admission result.** Held independent capacity, a supplied local graph law
and native gated capacity events are different completed model choices.
Their formulas can be tested against the same prospective pressure, but
none follows uniquely from `dot x=nu*P`. Section 8's classification remains
valid on its original invariant reduced law; it is not a certificate for
the unrelated writers above. A proposed mechanism must state which choice
it uses and satisfy its continuous, event, domain and covariance obligations
before its restoration claim can be transferred to the engine.
