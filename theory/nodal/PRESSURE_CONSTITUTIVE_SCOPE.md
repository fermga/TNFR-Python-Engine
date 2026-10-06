# Pressure premises and constitutive response

Locality, covariance, phase-domain boundaries and the prospective pressure comparison.

Section numbers are stable locators across this document family.
The [parameter reference](../NODAL_PARAMETER_FOUNDATIONS.md) owns the
reading map; the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns active tasks. Each result retains its stated model and scope.

## 4. What locality and symmetry can derive

Consider only an affine local EPI pressure on a finite loopless support:

\[
P_i(x)=b_i+\sum_j A_{ij}x_j,
\qquad A_{ij}=0\ \text{off support for }i\ne j.
\]

Uniform-shift invariance gives `A*1=0`. Requiring every uniform form to be an
equilibrium gives `b=0`. Consequently

\[
P_i(x)=\sum_{j\ne i}a_{ij}(x_j-x_i).
\]

The local maximum principle for every `x` is equivalent to `a_ij>=0`: the
sufficiency follows term by term; for necessity set `x_i=0`, one selected
neighbor `x_j=-1`, and all other coordinates zero. This derives a possibly directed
diffusive generator **under those premises**. It does not select its gains.

Reciprocity needs an additional positive measure `h` satisfying
`h_i*a_ij=h_j*a_ji`. Then `W_ij=h_i*a_ij` is symmetric. With unit off-diagonal
row sums, `d_i=h_i` and `P=-L_rw*x`. Undirected support is insufficient: a
triangle with forward rates `2/3` and reverse rates `1/3` has symmetric
support but violates detailed balance, because the two cycle products are
`8/27` and `1/27`.

There is a restricted way to remove conductance ratios without fitting them.
If reciprocal conductances depend only on support and are invariant under
every support automorphism, they are constant on each undirected edge orbit.
On a connected edge-transitive graph with nonzero conductance all conductances
are one common positive value;
row normalization removes it and yields the unweighted random walk. Multiple
edge orbits leave ratios undetermined. This is a genuine consequence of the
added symmetry premises, not a derivation of why a network must have that
symmetry or why it must emerge.

Fixed phase, capacity and topology sources instead give `b=F`; the full
multichannel pressure need not annihilate a uniform EPI field or satisfy a
maximum principle. Removing those sources would change the model. Self-loops
also change normalization without contributing an EPI difference and require
their own convention. The portable covariance controls test these restricted
matrix identities alongside the existing support owner.

### 4.1 Exact low-degree reduction and its nonlinear boundary

On a loopless support of maximum degree two, suppose all phases have a
common real lift of width strictly below pi. A singleton neighbor has its
own phase; the circular mean of two neighbors is their lifted arithmetic
midpoint. The displacement from the node remains in the same regular
branch. Consequently `g_phi=G_U theta/pi` **exactly**, without a small-angle
approximation. If EPI uses the same row-normalized support, then

\[
p=G_U\left(e x+\frac{a}{\pi}\theta+b\nu+t k\right).
\]

This explains exact compensation families on paths and common-chart cycles.
It does not extend to arbitrary degree, nonmatching transport, or winding
configurations without a common chart. With undirected support the displayed
reduction has `sum_i k_i*p_i=0`; the nonlinear phase law has no such general
support-degree conservation identity.

For an explicit boundary, take unit-conductance `K4`, phases
`(0,0,0,pi/3)`, uniform EPI and common positive capacity. Every edge is
strictly U3-compatible at the configured `pi/2` threshold, and every
resultant is nonzero. Put `beta=atan(sqrt(3)/5)`. Direct phasor addition gives

\[
g_\phi=\frac1\pi(\beta,\beta,\beta,-\pi/3),\qquad
\sum_i g_{\phi,i}=\frac{3\beta-\pi/3}{\pi}<0.
\]

Indeed `tan(3*beta)=9*sqrt(3)/10<sqrt(3)=tan(pi/3)`, with both angles in
`(0,pi/2)`. The other three gradients vanish, so any positive phase
coefficient gives a negative instantaneous EPI mean rate. Reflecting all
phases reverses this rate. Oddness under reflection therefore does not make
the phase channel an antisymmetric pairwise flux. The
[compact pressure controls](../../tests/physics/test_pressure_constitutive_scope.py)
check this analytic result and the low-degree reduction against actual
materialized pressure, with explicit floating-point tolerance.

This is a constitutive boundary, not a demonstrated conservation bug.
The [regular phase metric](../TNFR_VARIATIONAL_PRINCIPLE.md#136-exact-state-dependent-metric-for-canonical-phase-pressure)
already relates this phase reading to a cosine energy with a
state-dependent diagonal metric; that metric generally differs from the
transport degree metric. Neither identity supplies phase motion or a
conserved EPI total. Replacing the phasor law by a conservative edge flux
would require an independent conservation premise and would change the
model. Unit covariance, joint dynamical closure and the identity intended
to persist must be assessed before selecting such a replacement.

### 4.2 When scale covariance and regularity force linear form response

The affine premise in section 4 can itself be derived under stronger, explicit
assumptions. Fix the graph and all non-EPI state. For a source-free pressure
`Q` on a real form chart, suppose

\[
Q(x+c\mathbf1)=Q(x),\qquad Q(a x)=a Q(x)\quad(a>0),
\]

whenever the arguments are admitted, and suppose `Q` has a Frechet derivative
at an interior uniform state. Work in a translated chart containing an open
neighborhood of zero and the segment from zero to each form being considered;
shift invariance transfers the derivative to zero.
Homogeneity gives `Q(0)=0`. For every admitted `v`,

\[
Q(v)=\lim_{\epsilon\downarrow0}\frac{Q(\epsilon v)-Q(0)}{\epsilon}
     =DQ(0)v.
\]

Thus `Q` is exactly linear on that domain, not merely approximately linear near
equilibrium. Locality then restricts its matrix support, and the maximum
principle from section 4 fixes the off-diagonal signs. The resulting family is
`Q_i=sum_j a_ij*(x_j-x_i)`, `a_ij>=0`. This does not select those coefficients,
reciprocity, row normalization or the active support. Differentiability means
one linear total derivative; separate directional derivatives are insufficient.

Row normalization can be justified by another explicit premise. Suppose this
linear row is anonymous and unweighted: only the center and neighbor multiset
are available, with no other structural attributes distinguishing neighbors.
Permutation symmetry then gives `Q_k=c_k*sum_j(x_j-x_i)`. Require invariance
when every neighbor observation is repeated `m` times. It follows that
`m*c_(mk)=c_k`; taking `k=1` yields `c_k=c_1/k`. The arithmetic mean is thereby
selected up to the common single-neighbor gain `c_1`. Fixing that gain defines
a normalization, not a newly derived physical rate. Repetition of observations
is an added premise about this row, not a theorem about cloning physical nodes,
adding parallel transport channels or changing topology. Weighted rows still
need a meaning and aggregation rule for their conductances.

Exact amplitude covariance is an added scale-free constitutive premise. It is
stronger than correctly converting units while also transforming dimensional
parameters. EPI-offset invariance is justified only for a channel whose chosen
form origin is redundant; it is not a theorem about every operator or a
physically distinguished vacuum. For the mixed pressure, this argument can
apply to `Q(x)=P(x)-P(0)` at held non-EPI state if its premises hold. It does not
determine the independent source `P(0)` or eliminate that source.

Regularity is essential. With gaps `d_ij=x_j-x_i`, the comparison response

\[
Q_i^{\rm cmp}(x)=
\begin{cases}
\displaystyle\frac{\sum_{j\in N_i}d_{ij}^3}{\sum_{j\in N_i}d_{ij}^2},
 &\sum_jd_{ij}^2>0,\\
0,&\text{otherwise}
\end{cases}
\]

uses only existing form differences, has no new dimensional parameter, and
satisfies locality, relabeling, affine form-chart covariance and the maximum
principle. It is continuous at uniform form since its magnitude is bounded by
the largest neighbor gap. However, at the center of a two-leaf star, directions
with gaps `(1,0)` and `(0,1)` each give one, while their sum `(1,1)` also gives
one. Its directional derivative at uniform form is not additive, so no total
derivative exists there. This is a countermodel to uniqueness from covariance
and continuity alone, not an installed pressure law. The existing
[covariance controls](../../tests/physics/test_nodal_parameter_covariance.py)
compare it with the shared exact EPI owner.

There is also a useful stationary boundary independent of linearity. On fixed
finite connected undirected support with positive capacities, assume every
local form maximum with a strictly lower neighbor has strictly negative source-free
pressure. A nonuniform stationary form would have a maximum plateau with a
boundary node adjacent to a lower value, contradicting its zero pressure.
Hence only uniform stationary form is possible. Nonlinearity alone cannot
evade this obstruction while keeping that strict response premise. Zero
capacity, disconnected support, zero response on unequal neighbors or an
additional source changes the hypotheses. This is not a convergence theorem
or an exclusion of phase identity, moving patterns or finite-lived regions.
For directed support the same plateau argument needs strong connectivity;
weak connectivity alone permits distinct terminal values.

### 4.3 Phase domain, orientation and a discriminating structural response

The circle supplies periodicity, not a unique real-valued pressure. On regular
branches the implemented `g_phi` respects common rotation and node relabeling.
It is odd under internal phase conjugation `theta->-theta`. That conjugation
is different from a graph permutation representing a spatial reflection.
Holding `x,nu,G,W` fixed gives `P(-theta)-P(theta)=-2*a*g_phi(theta)`.
Consequently it is not a redundancy of the full scalar-form law when `a>0`.
If phase orientation is declared merely a coordinate convention with unchanged
scalar form and pressure, its coefficient must also transform as `a->-a`.
Otherwise conjugation describes a distinct model state. No symmetry claim is
complete without the transformation of every state, output and coefficient.

Two separate global-domain obstructions are explicit:

- On P2, the ideal phase pressure is `wrap(delta)/pi`. Its limits at an
  antipodal gap are `+1` and `-1`, although the neighbor resultant is nonzero.
  No continuous real-valued extension matches both limits. Moreover a
  single-valued periodic odd response must satisfy `f(pi)=0`, since `pi` and
  `-pi` denote the same circle point. Signed binary64 ties are a representation
  convention, not a globally odd circular law.
- With center phase zero and two neighbors at `0` and `pi±epsilon`, the
  neighbor resultant tends to zero. The pressure tends to `-1/2` and `+1/2`
  respectively. Here the limiting target directions are not antipodal to the
  center. No value assigned at zero resultant makes this response continuous.

These controls do not obey a strict U3 edge domain and are not evidence against
the regular-domain theorem. A model may restrict its domain, retain a lift or
extra state, define an event rule, or select a different response; each is an
additional obligation. The pressure reader's numerical fallback supplies none
of those dynamical justifications. The existing regular phase metric and its
[current/curvature identity](../TNFR_VARIATIONAL_PRINCIPLE.md#136-exact-state-dependent-metric-for-canonical-phase-pressure)
remain the owner of local geometry:

\[
J_{\phi,i}=\frac{|S_i|}{k_i}\sin(\pi g_{\phi,i}).
\]

On the regular non-antipodal reciprocal domain, both readings have the same
alignment potential `V_phi`, but different metrics: `g_phi=-H_phi^-1*grad V_phi`
with `H_phi,ii=pi*|S_i|*sinc(pi*g_phi,i)`, while
`J_phi/pi=-(pi*diag(k))^-1*grad V_phi`. The common potential therefore does not
select the response metric or a phase-to-form coupling. Equivalently,
`J_phi/pi=(|S_i|/k_i)*sinc(pi*g_phi,i)*g_phi,i` is a nodewise attenuation on
that domain. This does not order total EPI source work: nodal work factors
can have different signs and the attenuation is not spatially uniform. The
existing forcing/Dirichlet balance remains necessary for that comparison.

This identity exposes a concrete selection question: should form pressure read
only the resultant direction, as the present phase channel does, or also its
amplitude? Both readings already exist in the repository; neither is a phase
clock. Matching their coherent linear response does not answer the question.
For neighbor gaps `epsilon*a_j`, let `m`, `v` and `c3` be respectively the mean,
variance and central third moment of the unscaled `a_j`. On the regular branch
near consensus,

\[
g_\phi=\frac{\epsilon m-\epsilon^3c_3/6}{\pi}+O(\epsilon^5),\qquad
\frac{J_\phi}{\pi}=
\frac{\epsilon m-\epsilon^3(c_3+3mv+m^3)/6}{\pi}+O(\epsilon^5).
\]

The first possibly nonzero discriminating coefficient is
`(m^3+3*m*v)/(6*pi)` at cubic order; it vanishes for `m=0`.
For two neighbors with gaps `m±d`, the comparison is exact: phase pressure is
`m/pi` while the matched-gain current is `sin(m)*cos(d)/pi`, for `|d|<pi/2`
and an admitted principal mean `m`. The
[production controls](../../tests/physics/test_pressure_constitutive_scope.py)
use `m=pi/8`, spreads `pi/16` and `3*pi/16`, and center phase `pi/4`. All edge
gaps are strictly acute. The phase pressure stays fixed while the existing
current changes; no coupling coefficient is fitted and no alternative dynamics
is installed. These are model-discrimination controls, not laboratory evidence
selecting either response. Reflection symmetry or agreement to first order
likewise does not select a unique smooth periodic response function.
The [complete exchange comparison](SINE_CONSTITUTIVE_INFORMATION.md#global-closure-pressure-comparison)
now derives the associated phase row, global continuation and work balance.
It also shows why a bounded phase-only repair cannot preserve global
passivity with native Arg pressure and cosine storage. Normalized pairwise
superposition selects the sine response within an explicit comparison class;
it is not a consequence of circular phase alone or a default pressure change.

#### Prospective finite-response discriminator

The same P3 preparation gives an exact-real prediction for this bounded
comparison. Keep unit conductances, common fixed capacity `kappa>0`, common
initial scalar form, no Gamma and held phases
`(c+m-d, c, c+m+d)`. Require `abs(m)+abs(d)<pi/2`. Let `a>0` and `e>0` be
the same effective phase and EPI coefficients in both models. The comparison
source replaces only `g_phi` by the already defined `J_phi/pi`; doing so is a
declared constitutive hypothesis, not an existing default engine law.

For spreads `d_1,d_2`, the topology source is identical and the capacity
gradient is zero. Thus those full-mixture channels cancel in the difference
of the two form responses, even though the P3 topology source itself need
not vanish. The present Arg source has center `m/pi` and endpoint differences
that are antisymmetric under path reflection. This odd subspace is invariant
under `L_rw`, so the center difference is zero. For the comparison source,
the even part is proportional to `(-1,1,-1)`, a `L_rw` eigenvector of
eigenvalue two. Integration of that forced mode yields

```text
x_center^Arg(t; d_2) - x_center^Arg(t; d_1) = 0,

x_center^J(t; d_2) - x_center^J(t; d_1)
    = a*sin(m)*(cos(d_2)-cos(d_1))/(2*pi*e)
      * (1-exp(-2*kappa*e*t)).
```

The odd component of the current source likewise cannot reach the center in
this fixed linear transport model. Nonzero `sin(m)` and distinct cosines
therefore supply a finite-response discriminator at every positive horizon.
This is a paired exact-real prediction with held supporting coordinates,
before clipping or numerical stepping. It is not an absolute-response formula
with the topology channel silently omitted. A numerical comparison must check
the declared unclipped interval and separate pressure realization, integration
and rounding defects. A frozen second spread is a numerical prospective
control, not an independent physical observation.

**Executed fixed control.** The [shared-engine experiment](../../src/tnfr/research/phase_form_response.py)
now evaluates four trajectories with common initial EPI `1/2`, capacity one,
`c=pi/4`, `m=pi/8`, spreads `pi/16` and `3*pi/16`, and the complete normalized
mixture `(phase, EPI, capacity, topology)=(1/4,1/2,1/8,1/8)`. It refreshes
pressure before each of 256 shared Euler calls with `h=1/256`, holds phase and
support, and declares Gamma absent. Only the alternative phase source is
substituted. The nonzero topology contribution remains in each trajectory;
the initial total central Arg pressure is `-3/32`, despite its positive phase
contribution. This prevents a phase-only interpretation of the absolute motion.

The declaration and predictions are written and hashed before evolution.
For the even mode, Euler replaces `exp(-2*kappa*e*t)` by
`(1-2*kappa*e*h)^256`. The signed central response below is wide minus narrow:

| Phase source | Continuous prediction | Euler prediction | Engine observation |
| --- | --- | --- | --- |
| Arg direction | 0 | 0 | -5.55e-17 |
| `J_phi/pi` | -0.002874319847547102 | -0.002877592338102031 | -0.002877592338101642 |

For the current source the analytic integration defect is about `-3.27249e-6`,
whereas engine minus discrete prediction is about `3.89e-16`. Thus the measured
difference is resolved independently of the known Euler error. The fixed
budgets are `2e-14` for pressure realization, `2e-15` for one represented-step
rounding defect, `1e-11` for the discrete trajectory and `5e-6` for the paired
continuous response. All pass. Positive averaging with source bound `0.203125`
gives the analytic enclosure `[0.296875,0.703125]`; raw represented proposals
and accepted values also remain within the predeclared open interval
`(0.25,0.75)`, inside the hard clip `[0,1]`.

The [regression controls](../../tests/research/test_phase_form_response.py) also
compare against an independent augmented matrix power and check evidence
hashes, full-mixture retention and rejection of changed declarations. Run
`python -m tnfr.research.phase_form_response --output <fresh-directory>` to
produce the declaration, trajectories and evidence sidecar. This is local
prospective recording with binary64 analytical evaluations and selected error
budgets, not an interval certificate or external preregistration. It establishes
the predicted finite implementation distinction; both constitutive hypotheses
remain possible until an independent reduction or physical measurement
criterion selects the information their pressure must preserve.

### 4.4 Constitutive admission ledger and remaining choices

The nodal equation fixes a typed rate product. The following ledger prevents
extra constitutive conditions from being promoted to consequences of that
product. Its mathematical owners above replace parallel research campaigns.

| Requirement | What it justifies | What remains open or conditional |
| --- | --- | --- |
| Relabeling covariance when node names are bookkeeping | Permuting all nodal fields and support must permute pressure | It does not select an aggregation, graph, coefficient or actual state symmetry. |
| Phase periodicity and common rotation with no supplied phase reference | Dependence on circular relative configuration on the admitted domain | Neither the Arg rule, phase orientation, behavior at cancellation nor a phase clock is selected. |
| Form-chart and time-unit covariance | The coefficient transformation in [section 3](../NODAL_PARAMETER_FOUNDATIONS.md#3-joint-changes-of-form-and-time-units) | Fixed normalized numeric weights are not a physical unit law; a vacuum may invalidate form-offset redundancy. |
| Source-free exact amplitude covariance plus regularity at uniform form | The linear family proved in section 4.2 | These are added premises; they do not constrain an independently supplied source or forbid nonlinear models with another scale/domain. |
| Locality plus a local maximum principle in that linear family | Nonnegative neighbor-difference coefficients | Nonlinearity, reciprocity, row speed and the support must be addressed separately. The full sourced pressure does not obey this form-only maximum principle. |
| Reciprocal edge response with a declared measure | Detailed balance and the corresponding Dirichlet identities | Undirected support alone does not suffice. A weighted EPI charge is not thereby conserved in the presence of other channels. |
| Anonymous linear row plus whole-neighborhood observation replication | Selects the arithmetic mean up to a common gain | Replication is an additional premise, not graph cloning. More general row gains can be absorbed into effective capacity for isolated EPI diffusion; replacing primitive capacity everywhere changes its other laws. |
| Regular phase-domain closure or a declared boundary rule | A defined continuation when a trajectory reaches branch/cancellation boundaries | Current-state U3 admission and a numerical fallback do not establish future domain invariance. |
| Full-state identity, source evolution and event laws | A complete candidate for maintenance or formation | Well-posedness still needs a domain and regularity argument. A pressure formula, grammar admission or tetrad snapshot alone does not supply the complete laws. |

For reciprocal transport, positive capacity and held coefficients, the actual
mean balance is `d/dt sum_i(d_i/nu_i)*x_i=sum_i d_i*F_i`, with the metric also
held fixed. It is an accounting identity, not a requirement that the right side
vanish. With `W,nu,e,F` fixed and a compatible stationary target `x*`, the
difference `x-x*` obeys homogeneous diffusion even though absolute EPI does not. Reuse the
[forced-support owner](../FORCED_SUPPORT_BALANCE.md), rather than rejecting the
full pressure merely because a source-free property fails.

The present pressure is therefore retained as a configured, regular-domain map,
with a well-characterized EPI sector and explicit source choices. Neither its
normalized coefficients nor its use of phase direction alone is uniquely
derived. An unresolved constitutive distinction is what information a derived
phase-to-form response must preserve: direction, resultant amplitude and its
source work, with a declared phase domain and orientation. This must be fixed
before promoting an alternative to dynamics. The existing metric, joint
derivative and discrimination controls provide the route; another arbitrary
potential, numerical trajectory or fitted force cannot remove the freedom.

<a id="regular-derived-phase-pressure"></a>
### 4.5 Regular pressure on a state with derived phase

The [S/D reference admission](../FUNDAMENTAL_THEORY.md#reference-s-d-admission-verdict)
distinguishes independently retained primitive phase from the orientation of
a regional form contrast. This distinction matters at zero contrast even
when every positive-amplitude phase lies well inside an acute chart. The
[directed two-region lift](DERIVED_FORM_PHASE.md#derived-phase-identification-admission)
already gives a concrete obstruction and complete mean/contrast budgets.
The following admission results classify the obstruction and the remaining
constitutive freedom; they do not install another pressure or select a
mechanism to sustain a pattern.

**State and equivalence.** Write the regional contrasts as `z in C^r`, with
means `mu`, capacities, support, interfaces and reporting frames retained.
Complex notation represents real form coordinates. In this section these
other data are fixed parameters. A real scalar source `q(z;mu,metadata)` is
an additional regional mean-pressure contribution, not an angular velocity
or a complete nodal law. Its contribution to mean velocity also carries the
declared capacity. If q is to be an observation on the admitted Gram state
`Q=z*z^dagger`, it must satisfy

\[
q(e^{i\alpha}z;\mu,\mathrm{metadata})
=q(z;\mu,\mathrm{metadata}).
\]

This is an active common modal rotation at fixed frames, expressing equality
of the selected observations. It is not a symmetry imposed on every TNFR
form law. A passive reporting-frame change instead transforms the complex
coordinates and interface matrices together. A regional relabeling likewise
permutes the whole source vector and its metadata; it does not leave each
indexed scalar unchanged.

**An amplitude-independent phase source cannot retain directional response
at a vanishing contrast.** Consider two regions with `r_A,r_B>0` and a regular
relative-phase interval I. Suppose the proposed scalar source is `f(delta)`
throughout this domain, independently of the two amplitudes. Fix nonzero
`z_B=R*exp(i*beta)` and choose any `delta_1,delta_2 in I`. The paths

\[
z_A^{(j)}(\epsilon)=\epsilon e^{i(\beta-\delta_j)},\qquad
\epsilon\downarrow0,
\]

have the same limiting real form and retained data. Their source limits are
`f(delta_1)` and `f(delta_2)`. A continuous extension at `(0,z_B)` therefore
requires f to be constant on I. If I is symmetric and f is additionally odd,
that constant is zero. The conclusion requires both paths to belong to the
admitted domain; a model excluding this boundary has a narrower obligation.

For the existing two-triangle lift, `f(delta)=delta/(2*pi)`: the choices
`delta=0,pi/6` yield limits `0,1/12`, or mean-rate contributions `0,nu*w/12`.
The incompatibility occurs away from antipodal or zero-resultant primitive
phase configurations. Replacing Arg by a nonconstant amplitude-independent
sine response retains this zero-contrast obstruction. An assigned angle at
zero changes neither limiting value. Amplitude attenuation would change the
source away from zero; it is a new constitutive choice, not a continuous
completion agreeing with the original rule.

There is no corresponding contradiction for reference S with independently
stored theta: these paths approach different retained phase states there.
Its own primitive-resultant and wrap boundaries still require the section
4.3 admission. Retained justified history can also distinguish approaches,
but then the history belongs to the state; it is not a form-only closure.

**Differentiability plus scale-free contrast response excludes an extra
rotation-invariant mean source.** Let the contrast domain contain an open
neighborhood of zero and the radial segments to all considered states,
and be invariant under common modal rotation. At fixed means and metadata,
assume q has a real Frechet derivative at zero and

\[
q(\lambda z)=\lambda q(z)\quad(\lambda>0).
\]

The section 4.2 argument gives `q(0)=0` and `q(z)=Dq(0)z` wherever these
radial scalings are admitted. Invariance under the rotation `alpha=pi`
also gives `q(-z)=q(z)`, whereas a real linear derivative gives
`Dq(0)(-z)=-Dq(0)z`. Hence

\[
\boxed{q(z)=0.}
\]

This is a theorem about this additional scalar source and these premises.
It does not set the contrast-vector dynamics to zero: complex-linear
transport obeys `F(e^{i alpha}z)=e^{i alpha}F(z)`, not scalar invariance.
The [existing all-state linear criterion](DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state)
already determines when mean/contrast separation and a complex-linear
generator yield a closed Gram law, including zero. Its inherited transport
and complete budgets remain available without an added mean source.

The homogeneity assumption here scales contrasts while holding means fixed.
It is not automatically the homogeneity of the entire fine form. Applying
the section 4.2 argument to a complete mean/contrast state can leave linear
mean-dependent terms; they must not be erased by this contrast-only result.
For a source with nonzero value at zero contrast, the result applies to its
contrast-dependent difference only if that difference meets the premises.
Smoothness at zero, exact homogeneity and the chosen Gram equivalence are
additional admissibility decisions, not consequences of the nodal product.

**Two countermodels expose the freedom.** They are mathematical comparisons,
not proposed engine mechanisms. Let a supplied dimensionless interface
orientation `tau_AB` have unit magnitude, and put
`s_AB=Im(conjugate(z_A)*tau_AB*z_B)`. With the reference D convention
`U_a_new=U_a*R(beta_a)`, reporting coordinates transform as
`z_a_new=exp(-i*beta_a)*z_a`. Transform the interface as
`tau_AB_new=exp(i*(beta_B-beta_A))*tau_AB`; s_AB is then unchanged.
For aligned matched frames the existing interface convention gives
`tau_AB=1`. This datum must not be silently reset after changing frames.

- `q=kappa*s_AB` is smooth, common-rotation invariant and zero when either
  contrast vanishes. It is quadratic, not degree-one homogeneous. For pressure
  units X, kappa has units `X^-1`. A form-unit conversion `z_new=lambda*z`
  transforms `kappa_new=kappa/lambda`, so the pressure converts correctly.
  Correct unit covariance does not imply fixed-kappa homogeneity or derive
  kappa's value.
- `q=s_AB/sqrt(abs(z_A)^2+abs(z_B)^2)` away from total zero, with value zero
  there, is degree-one homogeneous and continuous on the whole contrast
  space: `abs(q)<=sqrt(abs(z_A)^2+abs(z_B)^2)/2`. It is smooth when total
  contrast is nonzero, including either single-zero stratum. It has no real
  total derivative at total zero. For aligned frames, q at `(1,i)` is
  `1/sqrt(2)` and at `(-1,-i)` is the same; its homogeneous directional
  response cannot be a real linear derivative. Continuity alone therefore
  does not imply the vanishing theorem.

The second comparison has a stronger positive regularity property: its
extension is **globally 1-Lipschitz** in the real contrast norm. In aligned
frames put `y=(u_A,v_A,u_B,v_B)`, `r=norm(y)` and
`h=u_A*v_B-v_A*u_B`. Then `norm(grad h)^2=r^2` and `y dot grad h=2h`, so for
`r>0`,

\[
\left\|\nabla\frac{h}{r}\right\|^2
=1-\frac{3h^2}{r^4}\le1.
\]

The same identity holds after the orthogonal coordinate change representing
a unit interface tau. Integration along a line segment proves the Lipschitz
bound; if the segment crosses zero, split it there and use continuity.
Consequently a finite constant-coefficient Cartesian mean/contrast ODE with
this source inserted through a fixed linear map is globally Lipschitz and
has a unique solution for every finite time. Failure of differentiability at
zero does not invalidate this well-posed countermodel. The nodal equation
does not require the optional total-derivative premise of the vanishing
theorem. Continuity by itself would not have established this stronger claim.

These comparisons require no new microscopic EPI component, but neither is
derived by naming its ingredients structural. For example, on the retained
two-region model with common fixed capacity, inserting opposite mean-pressure
sources `(q,-q)` preserves the total mean. With the original linear contrast
row retained, its complete contrast budget remains
`E_dot=-(3*e*nu/2)*E-e*nu*abs(z_A-z_B)^2`: a source uniform within each region
has zero contrast projection. The Lipschitz example therefore provides a
well-defined nonlinear mean response through zero contrast, not internal
contrast maintenance. Its source choice is still independent; capacity,
support and any changed contrast row need their own laws and budgets.

| Explicit premise | Consequence | Remaining admission |
| --- | --- | --- |
| Phase-only source, amplitude independence and continuous single-zero continuation | Relative-phase response must be constant on the admitted interval | Nonconstant response needs amplitude dependence, retained state/history or a restricted domain |
| Gram-state scalar source, positive degree-one contrast homogeneity and a total derivative at zero | The extra contrast-dependent scalar mean source vanishes | Mean-dependent terms and covariant contrast-vector dynamics remain; none of these premises is universal |
| Existing complex-linear transport and its admitted interface/capacity domain | Cartesian and Gram evolution continue through zero with inherited budgets | Fine pressure, graph and held-capacity premises remain supplied |
| Reporting-frame/relabeling covariance | State, interfaces, observations and indexed sources transform together | This does not choose a response function, gain or a physical phase reference |
| Smoothness without fixed-parameter homogeneity, or homogeneity without a derivative | Nontrivial invariant sources remain possible; the norm-normalized comparison is even globally Lipschitz | The displayed comparison can define a well-posed Cartesian law; coefficient/source selection, complete budgets and physical justification remain independent |

The admission result separates a valid derived mechanism from an unjustified
additional feedback. The existing Cartesian interaction is regular at zero;
the attempted amplitude-independent angle-fed pressure is not. No selected
gain, artificial sustaining term, controller threshold or new default law
follows from that distinction.

The [compact pressure controls](../../tests/physics/test_pressure_constitutive_scope.py)
check the countermodels' scale behavior, the exact gradient bound and the
nonlinear directional response. They do not execute the alternative law or
promote its existence to constitutive selection.
