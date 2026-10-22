# Geometry and the formation boundary of the prepared mobile pair

**Scope:** stationary, finite-capture and instantaneous-rate restrictions for
the prepared mobile pair under the unchanged
[complete cycle law](LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-law).
The all-time part reuses the [sector-capture theorem](SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture);
the finite parts reuse the full-state transport bounds. A separate static C12
counterexample tests observation sufficiency, outside the original source
family. Actual-state observations distinguish coincidence from retention.
No trajectory, new reserved response, parameter search, input
or support change is evaluated. The earlier
[nominal interaction result](LOCALIZED_MOBILE_RESPONSE.md#mobile-response-result)
and its unresolved original-family interaction bound remain unchanged.
A separately declared preparation and mechanism would be needed to acquire
a common, finite-lived organization under this law while retaining the original
pair's exclusions. The [research program](../RESEARCH_PROGRAM.md#research-current-question)
alone admits further investigation of that deferred question.

<a id="geometry-and-formation"></a>
## 1. Which geometry enters this model?

Three distinct geometries must be retained:

| Geometry | Role in the present model | Boundary |
| --- | --- | --- |
| Support and coefficients | The supplied unit cycle determines incidence D, Laplacian L and the actual exchange paths; held capacities enter the complete law. | A drawn embedding or changed edge length alone does not change these rate rows. Edge length defines the localization observation, separately from conductance. |
| Relative form and primitive phase | Neighbor form differences drive phase motion; sine phase differences drive form. Phase differences determine the storage curvature and its critical configurations. | The Fourier carrier of `z=x+i*theta` is not primitive circular-phase winding. |
| Organization and boundary | A spatial window, profile, carrier band or partition specifies an observation of the retained full state. Its boundary remains coupled to the complement. | Overlap, compatible grouping and concentration do not establish actual entry into a new identity or its maintenance. |

These distinctions reuse the [relation foundation](RELATION_FOUNDATIONS.md),
[collective phase-offset conditions](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-phase-offset-partition)
and [regional work accounting](FORCED_REGIONAL_RESPONSE.md#7-a-region-and-its-environment-on-the-same-nodal-support).
They do not introduce an extra geometric force or select a new law.

The moving centers in the prepared-transport proof are comparison references,
not imposed trajectories in (1). Actual form and phase evolve under both
complete rows and can alter the observed profiles and their paths within the
proved error bounds. A common organization could involve deflection, changed
relative motion or internal rearrangement of the incoming patterns. It need
not be a third separate object, and its definition must not require that the
original patterns first separate. Any inferred position or velocity remains
an observation of the field on fixed sites, with its own availability and
error; the elementary support vertices do not move in this model.

A proper contiguous window on a cycle induces a path. It has no internal
graph cycle on which to transfer the earlier C9 constituent-winding identity.
Closing that path by ignoring its environmental endpoints would change the
support. A localized identity can instead retain relative form/phase structure
and boundary exchange, but it needs its own prospective formation criterion.

<a id="mobile-zero-sector-basin"></a>
## 2. An all-time basin for the original preparation family

Let D be the oriented incidence matrix of a simple unit cycle `C_N`, `N>=3`,
and `L=D^T D`, `A=L/2`. Retain signed real form, real continuous phase lifts,
held unit capacities, beta one, positive e,w, and no forcing or events.
In the declared clock `tau=w*t/pi`, with `sigma=pi*e/w>0`, the complete rows are

\[
 x'=-\sigma Ax-\tfrac12D^T\sin(D\theta),\qquad \theta'=Ax. \tag{1}
\]

Both ordinary means are conserved. Each source member retains its own means;
they need not equal the nominal zero means. A constant form or phase reference
can be subtracted without changing (1). The following source premise concerns
one consistent real lift relative to such a constant reference:

\[
 \|(x(0),\theta(0))\|_2\le M,\qquad H_0=2M^2<1. \tag{2}
\]

This is a sufficient source condition, not a phase reconstruction from form,
a physical preparation certificate or an admission from a cached verdict.

### Storage and geometric trapping

The existing structural storage and its actual clock-transformed balance are

\[
 H(x,\theta)=\tfrac12x^TLx+\sum_e[1-\cos(D\theta)_e],
 \qquad H'=-\tfrac\sigma2\|Lx\|_2^2\le0. \tag{3}
\]

Here H is the original storage, not the packet localization diagnostic.
For a cycle, `||L||<=4`, and `1-cos u<=u^2/2`, so

\[
 H(0)\le2\|x(0)\|^2+2\|\theta(0)\|^2\le H_0. \tag{4}
\]

Initially every lifted edge difference has absolute value at most
`2M<sqrt(2)<pi/2`. A first exit from this zero-winding acute component would
have an edge at `+/-pi/2`, contributing one to H. All other storage terms
are nonnegative, contradicting `H<=H_0<1`. Therefore every future edge remains
in the same lifted acute component, and in fact

\[
 \cos(D\theta)_e\ge1-H_0>0,\qquad
 (1-H_0)L\preceq\nabla^2V(\theta)\preceq L,\quad
 V=\sum_e[1-\cos(D\theta)_e]. \tag{5}
\]

This bound concerns phase-storage curvature, not the tetrad's diagnostic
curvature. It is valid for all nonnegative structural time, without extending
the finite moving-template error tube. The unwrapped gaps still telescope
to zero around the cycle, so the circular winding remains zero. Changing
that winding would require an edge to cross a circular cut, which is also
excluded by the stronger acute trap.

### The only limiting state on each mean leaf

The bounded gaps and fixed phase mean bound every lifted phase coordinate.
The form term in H bounds centered form because L is positive on its
mean-zero space. The invariant sublevel is thus compact on each mean leaf.
The full vector field is globally smooth with bounded sine currents and
linear form dependence, so forward continuation is available.

In the zero-dissipation set, `Lx=0`, hence form is uniform and `theta'=0`.
An invariant trajectory there must also have `D^T sin(Dtheta)=0`: otherwise
the form row leaves that set, since its ordinary mean derivative is zero.
Taking its inner product with theta gives

\[
 0=\sum_e (D\theta)_e\sin(D\theta)_e.
\]

Every summand is nonnegative and vanishes only at zero in the retained
acute chart. Thus every gap vanishes. The sector-capture/LaSalle argument
therefore yields

\[
 x(\tau)\longrightarrow\overline x(0)\mathbf1,\qquad
 \theta(\tau)\longrightarrow\overline\theta(0)\mathbf1. \tag{6}
\]

This is convergence to one state on each actual source member's conserved
mean leaf. It supplies neither a convergence rate nor a finite settling time.
The original finite packet identities can remain valid during transport
even though they eventually relax. Equation (6) excludes a nonuniform
stationary asymptote in this source family; it does not exclude finite-lived
additional structure before that limit.

### Application without a new response calculation

The two nominal Fejer preparations have orthogonal complex carrier bands
and joint norm `sqrt(2)*a`. Each primitive full-state error is at most epsilon,
and the joint source is their exact sum. Hence the entire original family
satisfies (2) with

\[
 M=(\sqrt2+1/500)a,\qquad a=2^{-12},\qquad
 H_0=2(\sqrt2+1/500)^2a^2<\frac3{12500000}=2.4\,10^{-7}. \tag{7}
\]

For example `sqrt(2)<70711/50000` proves the last strict rational bound.
Neither the computed endpoint defect nor the unresolved uniform interaction
verdict is a premise of (7). The conclusion therefore applies to all original
source errors, despite the distinct endpoint discrimination limitation.
The same estimate gives an all-time phase-Hessian lower factor greater than
`1-2.4e-7`. It does not give an all-time `||z||<2a` bound, preserve the two
carrier projections forever, or quantify the lifetime of another pattern.

## 3. Why the source's geometry cannot be omitted

Small storage alone does not imply consensus for arbitrary circular sources
on this same support. A separately supplied twist with constant principal
edge gap `2*pi/N`, uniform form and `N>4` is a critical configuration:
the incoming sine currents cancel and its Hessian is `cos(2*pi/N)*L`.
It is a strict local minimum on the relative-phase leaf and has the existing
positive-loss local recovery property. On `C_1024` its storage satisfies

\[
 H_{\rm twist}=N[1-\cos(2\pi/N)]\le\frac{2\pi^2}{N}<1. \tag{8}
\]

It has winding one and no all-small zero-component lift around the cycle.
It therefore does not satisfy the source condition (2). This counterexample
shows why support, storage and the phase component must be matched together.
It is an extended prepared pattern, not a localized object formed by the
current encounter. It supplies no route across the inaccessible phase boundary.

## 4. An exact carrier geometry, without a localized envelope closure

For N divisible by four, the repeated period-four family

\[
 \theta=(\varphi,\psi,-\varphi,-\psi,\ldots),\qquad
 x=(r,s,-r,-s,\ldots)
\]

is invariant under both complete rows. On this subspace A acts as identity;
direct neighbor addition gives

\[
 \varphi'=r,\quad \psi'=s,\qquad
 r'=-\sigma r-\sin\varphi\cos\psi,\quad
 s'=-\sigma s-\sin\psi\cos\varphi.
\]

The invertible relative coordinates `u=varphi+psi`, `v=varphi-psi` consequently satisfy

\[
 u''+\sigma u'+\sin u=0,\qquad
 v''+\sigma v'+\sin v=0. \tag{9}
\]

Their initial velocities retain r,s; this exact change of coordinates does
not discard form or install a pendulum as a new law. It exhibits nonlinear
internal dynamics selected by a compatible carrier geometry under the
actual positive loss. The family repeats throughout the support and supplies
no localized envelope, acquisition from outside that family or self-binding.
The finite-width prepared packets are not confined to this four-dimensional
family. Their other modes and boundary exchange cannot be dropped by using (9).

At the consensus tangent, the two modal exponents are
`lambda(k)*[-sigma/2 +/- sqrt(sigma^2/4-1)]`, with `lambda(k)=1-cos k`.
Thus their second k derivatives vanish at the current `+/-pi/2` carriers.
For the frozen underdamped coefficients their third derivatives do not vanish.
Equation (9) retains exact carrier nonlinearity, but neither it nor this
dispersion fact supplies a controlled modulation or localization theorem.

<a id="common-pattern-formation-contract"></a>
## 5. Formation of a common relational pattern

The geometric circumstances already matter in (1)--(8): the support supplies
paths, the prepared phase component determines accessible critical geometry,
and the storage/loss law restricts which regions of state space can be reached.
The unchanged small pair cannot acquire another stationary trapping geometry.
Its certified nominal nonadditivity can coexist with this obstruction.

The generative target is a **common organization of the incoming patterns**:
their relative arrangement and joint evolution acquire an identity absent
initially. It may move, deform or have bounded internal relative motion; neither
unchanged outgoing trajectories nor an extra separate object is required.
Retain constituent and environmental boundary information consumed by that
identity. If the original constituent identities cease to apply, record that
reorganization rather than retaining their labels as evidence of persistence.

An admissible finite-lived claim specifies:

- **Relational identity.** A positive-width class on the actual joint state,
  with a nonvanishing signal floor and specified relative geometry, primitive
  phase information and internal response. Any collective center, trajectory
  or phase registration needs an observation rule, ambiguity handling and
  error bounds; a band observation is not an independent dynamical constituent.
- **Formation.** The original source class lacks that common identity with a
  margin. Both complete rows must carry it into the declared class. Initial
  support connectivity, the presence of two packets or transient overlap
  does not establish this acquisition.
- **Joint retention.** The relation remains in a declared nontrivial domain
  for a specified interval beyond ordinary passage, with environmental work,
  dispersion and loss accounted for. Bounded distance alone is vacuous on a
  finite cycle; a common moving center alone can also follow two separating
  packets. The incoming patterns need not separate before this test begins.
- **Continued response.** The reached organization has a discriminating later
  response without resetting its acquired state. Compare the same-source
  linear passage and separate histories so that overlap, interference and
  independently continuing packets cannot alone pass the formation test.

A third Fourier band or the difference of three histories alone does not
satisfy those conditions. The
[parent-acquisition architecture](ORGANIZED_PARENT_ACQUISITION.md) illustrates
this proof structure; its supports, windings and coefficients do not transfer.

Specify whether the claim concerns the exact nominal preparation or every
member of the original source family. A family claim must carry its correlated
preparation errors through initial absence, entry, retention and response;
the joint source is the sum of the same two uncertain constituent sources.
Retain their actual conserved means. Nominal feasibility does not establish
family-wide formation, and the unresolved earlier interaction bound neither
proves nor forbids a direct formation argument.

Declare the observation's domain and availability through overlap and
deformation. Treat an identification as an exact symmetry only when it
preserves the complete law; a lossy observation retains its stated information
and error limits. Independent registration of each incoming packet must not
discard their relative phase or origin. If an observed region changes
membership, include transport across its moving boundary rather than reusing
a fixed-region balance unchanged.
Loss of the original packet labels is distinct from loss of the retained fine
state. A sufficient full-state description can support an organization claim
without an autonomous reduced law; that stronger claim needs its own closure
or memory argument.

The signal floor, absence margin, entry and retention windows, and later
response discriminator need independent reasons and an error budget before
evaluating a formation response. Declare the same-source passage control's
law and comparison interval. The actual damped tangent and conservative
transport reference are distinct controls; using the latter retains its
damping-error allowance. Finite overlap or an arbitrarily small deformation
alone cannot establish a sustained identity. If linear passage or independently
evolving packets also pass the proposed formation test, it does not yet
discriminate the target. Prior inspected responses remain prior information;
they cannot be relabelled as reserved evidence for a newly chosen criterion.

The existing endpoint bounds still restrict the unchanged source family.
At tau=256, each nonzero band observation has less than `1/25` of its
squared norm outside its radius-96 reference window; the two reference
centers have cycle distance 256. If a common cycle ball of radius r contains
more than `1/25` of **each** band's squared norm, it must intersect both
reference windows. The cycle-distance triangle inequality then gives
`256<=2(96+r)`, hence `r>=32`. This necessary geometric restriction neither
excludes wider or two-lobed composites nor establishes one above that radius.
It concerns the retained band observations, not an additive decomposition of
local total density. Radius 32 is not a newly selected formation threshold.

No such composite identity or binding mechanism is certified here.
Comparing nonlinear coupling with dispersion and loss requires the actual
carrier geometry: at the present `pi/2` carrier the tangent's quadratic
dispersion vanishes. A quadratic-dispersion envelope law cannot be imported
as a formation theorem. Changing preparation, support, loss or events would
define a separately admitted case, preserving the present evidence.

<a id="mobile-relative-capture-obstruction"></a>
## 6. Actual-state coincidence and a finite capture obstruction

This section tests specified geometric requirements for candidate
compact-capture identities.
It defines a capture class, not a complete NFR identity or a physical binding
criterion. The full state, primitive phase lift and environmental influence
remain retained. Every conclusion below covers the **whole original source
family** on `[0,256]`, without consuming the evaluated interaction defect.

### 6.1 One preparation bound for the actual state and passage controls

Write `v=v_A+v_B` for the existing opposite-carrier reference, and let
`U(tau)` be the actual damped tangent propagator. Compare three histories:

- the actual joint flow `z_AB=Phi_tau(u_A+u_B)`;
- the same-source linear control `y=U(tau)(u_A+u_B)`;
- the sum of separate nonlinear flows `z_A+z_B`.

The primitive errors remain shared as originally specified; the joint source
error is at most `2*epsilon`, not epsilon. In the notation of the
[encounter bound](LOCALIZED_MOBILE_INTERACTION.md#mobile-pair-identity),
the first two template errors are at most E and
`2*epsilon+sqrt(2)*a*d`, respectively. For the separate-history sum, contractivity
applied to the summed tangent and the two individual Duhamel remainders gives

\[
 E_s=2\epsilon+\sqrt2\,a d+\frac{64}{3}Ta^3
     =E+\frac{32}{3}Ta^3. \tag{10}
\]

This uses the individual bootstraps as well as the joint bootstrap; it does
not assume that the damped generator is complex-linear or that the bands are
closed nonlinear subsystems. The unchanged inputs give, using the same
outward bounds as the encounter proof,

\[
 E_s/a<\bar h=\frac{13}{250}.
 \tag{11}
\]

For each of these histories Z, put `f=P_+Z`, `g=P_-Z`. Orthogonal projection
therefore gives `||f-v_A||,||g-v_B||<a*hbar` and
`||f||,||g||>237*a/250`. These are observations of two real channels expressed
as `z=x+i*theta`, not independently evolving constituents or complex EPI.

### 6.2 A coincidence class on the actual state

Define normalized diagnostic weights and a relative-distance observation by

\[
 \mu_+(j)=\frac{|f_j|^2}{\|f\|^2},\qquad
 \mu_-(l)=\frac{|g_l|^2}{\|g\|^2},\qquad
 K_\ell(Z)=\sum_{d_N(j,l)\le\ell}\mu_+(j)\mu_-(l).
 \tag{12}
\]

The product defines a deterministic diagnostic, not an assumption of physical
or probabilistic independence. K lies in `[0,1]` and measures relative bulk
proximity without assigning sites to constituents, selecting peaks or
prescribing motion. It is invariant under common cycle translations and
reflection with band exchange. It discards internal phase and shape
information; it supplies neither a closed future law nor a decomposition of
the actual local squared density into two independent energies.

For a declared distance ell, define the **majority-coincidence class** by
the signal floors `||f||,||g||>=237*a/250` and `K_ell>=1/2`.
The threshold means a majority of the diagnostic pair weight. It is a stated
observation policy, not a universal TNFR constant. The definition allows
deformation and displacement and has an interior on the full state domain.
Initial absence, entry into this class and persistence there can now be
tested independently. For any candidate identity that includes this geometric
requirement, additional phase, response or memory conditions cannot rescue a
trajectory that fails it.

To bound K, use an auxiliary reference-window radius r. If each band has
outside fraction at most eta, then two elementary geometric implications are

\[
 \begin{array}{ll}
 D=0,\quad\ell\ge2r &: K_\ell\ge(1-\eta)^2,\\
 D>2r+\ell &: K_\ell\le1-(1-\eta)^2,
 \end{array} \tag{13}
\]

where D is the cycle distance between reference centers. The first line
includes every pair from the two cores; the second excludes every such pair
by the triangle inequality. All remaining pair weight is at most
`1-(1-eta)^2`. Equality in the second line's separation condition is not
admitted by this argument.

Choose the auxiliary proof radius `r=48`, half the existing localization
radius. It changes neither the source nor the law. The transport tail bound
and (11) yield

\[
 H=\frac{16384}{47^3}<\frac4{25},\qquad
 \eta<\left(\frac{2/5+13/250}{1-13/250}\right)^2
       =\left(\frac{113}{237}\right)^2<\frac14.
 \tag{14}
\]

For **every** `96<=ell<160`, the actual joint state and both passage controls
consequently satisfy

\[
 K_\ell(0)<\frac7{16},\qquad
 K_\ell(128)>\frac9{16},\qquad
 K_\ell(256)<\frac7{16}. \tag{15}
\]

The norm floors keep K continuous. Thus the majority-coincidence class is
absent initially with a margin greater than `1/16`, entered during encounter
with that margin, and exited before the original endpoint. This is an
analytical entry/exit result, not a newly sampled response. Both controls
exhibit it, so coincidence acquisition alone fails to discriminate formation.
More generally, `K_ell<7/16` whenever
`|256-2*tau|>96+ell`. For example, at ell=96 it is excluded at every outgoing
checkpoint `224<tau<=256`. Choosing a checkpoint strictly below 256 would
leave time for a later readout, but coincidence already fails there and stays
excluded through 256. No crossing time or behavior at equality is asserted.

### 6.3 Circular registration and a relative-retention bound

A second observation restricts relative motion without identifying a peak.
For any nonzero band y, let

\[
 M(y)=\sum_j e^{2\pi i j/N}|y_j|^2.
 \tag{16}
\]

A nonzero norm does not ensure a nonzero M. Here the Fejer reference at center
c gives an independent availability argument. Character orthogonality and
the adjacent-coefficient autocorrelation imply

\[
 M(v_c)=a^2\kappa_Q e^{2\pi i c/N},\qquad
 \kappa_Q=\frac{\sum_m b_m b_{m+1}}{\sum_m b_m^2}
          =\frac{2Q^2-2}{2Q^2+1}.
 \tag{17}
\]

The admitted bands avoid wraparound aliases in this adjacent sum. For
`||y-v_c||<=a*h`, the norm-one diagonal character matrix gives

\[
 |M(y)-M(v_c)|\le a^2 h(2+h).
 \tag{18}
\]

When `beta=h(2+h)<kappa_Q`, the moment stays nonzero and the disk in (18)
bounds the angular error by `asin(beta/kappa_Q)`. Define the positional bound
`delta=N*asin(beta/kappa_Q)/(2*pi)`. This is a circular first-moment center,
not a center of mass or an autonomous collective coordinate.
For (11), `kappa_16=170/171`, `beta=6669/62500`, and

\[
 \frac\beta{\kappa_{16}}<\frac19-\frac1{6\,9^3}
       \le\sin(1/9),\qquad
 \delta<\frac{1024}{18\pi}<19.
 \tag{19}
\]

Each actual center can be lifted continuously within delta of its own
reference `128-tau` or `-128+tau`. Since `256+2*delta<512`, their signed
relative center s has a single continuous principal branch and

\[
 |s(\tau)-(256-2\tau)|\le2\delta.
 \tag{20}
\]

If s stays in any fixed real interval of width W throughout `[t_0,t_1]`,
subtracting its endpoint bounds proves

\[
 t_1-t_0\le\frac W2+2\delta<\frac W2+38.
 \tag{21}
\]

This allows common translation, deformation and relative oscillation inside
the interval. It excludes only longer confinement of this observed relation;
it does not bound instantaneous velocity or every possible relative identity.
The reference passage traverses an interval of width W in time `W/2`.
Equation (21) limits extra residence without assuming a relative force law.

### 6.4 Exclusion at the preexisting localization scale

The same result also tests a capture class using the **existing** radius 96
and fraction `24/25`: both bands would have to place at least `24/25` of
their squared norms inside one common radius-96 ball. This allows arbitrary
common centers and deformed profiles.

For a band with fraction at least `1-f` in a radius-r ball, `r<N/4`, put
`alpha=2*pi*r/N`. Its core moment lies in the angular cone `[-alpha,alpha]`
about that ball's center, with magnitude at least
`(1-f)*||y||^2*cos(alpha)`. The tail moment has magnitude at most
`f*||y||^2`. If `f/[(1-f)*cos(alpha)]<1`, the observed center is consequently
within

\[
 r+\frac N{2\pi}\arcsin\frac f{(1-f)\cos\alpha}
 \tag{22}
\]

of the ball center. For `r=96,f=1/25`, `alpha=3*pi/16`,
`cos(alpha)>4/5` and `5/96<sin(1/18)`, so the second term is less than 10
sites. Two bands in that common ball would therefore have center separation
less than 212. But (20) makes the actual separation greater than 218 both
initially and at tau=256. The common-ball class is excluded at those times.
At tau=128 the original joint band-tail certificate puts both bands in the
same radius-96 ball with fraction greater than `24/25`; the actual damped
linear control also satisfies that certificate, since its template error is
smaller. Entry alone again cannot identify a new organization.

**Conclusion and scope.** Persistent majority coincidence at the outgoing
checkpoints in (15), prolonged relative confinement forbidden by (21), and
common-ball capture through tau=256 at the preexisting width/fraction are
excluded for the original source family. These are failures of necessary
conditions, not merely unsuccessful sufficient certificates. They leave open
shorter-lived organizations during encounter, wider or two-lobed structures,
internal phase/response identities, and reorganizations for which the old
bands no longer define the relevant relation. No full composite identity,
retention mechanism or discriminating continued response is established.
The nominal nonlinear interaction remains valid alongside these exclusions.

<a id="mobile-relative-rate-obstruction"></a>
## 7. The actual relative rate and its missing internal information

### 7.1 Differentiate the observation under both complete rows

The position bound (20) alone cannot bound a velocity: a small positional
error may oscillate rapidly. Instead differentiate the actual observation
under (1). With `z=x+i*theta`, the exact two-real-channel field is

\[
 z'=iAz-\frac\sigma2 A(z+\bar z)+R(\theta),\qquad
 R(\theta)=-\frac12D^T[\sin(D\theta)-D\theta].
 \tag{23}
\]

The conjugate term retains the form-only loss. Dropping it would replace the
actual positive-loss law by a complex-linear model. Since the opposite
projectors satisfy `P_+bar(z)=bar(P_-z)` and commute with A, their exact rows are

\[
 f'=iAf-\frac\sigma2 A(f+\bar g)+P_+R,\qquad
 g'=iAg-\frac\sigma2 A(g+\bar f)+P_-R.
 \tag{24}
\]

These are observed rates with the full state retained, not autonomous band
equations: R consumes the actual primitive phase, including omitted modes.
Define the real-linear directional derivative

\[
 \mathcal D_y(v)=\sum_j\chi_j(\bar y_jv_j+y_j\bar v_j),
 \qquad \chi_j=e^{2\pi i j/N}.
 \tag{25}
\]

For `M(y)!=0` and a continuous center lift,

\[
 M(y)'=\mathcal D_y(y'),\qquad
 c_y'=\frac N{2\pi}\operatorname{Im}\frac{\mathcal D_y(y')}{M(y)},
 \qquad s'=c_f'-c_g'.
 \tag{26}
\]

The character chi is complex. Replacing (25) by twice a real part outside the
sum would incorrectly force the moment derivative to be real. Its real
linearity permits a conservative, dissipative and nonlinear contribution to
(26), each evaluated at **the same actual state**. The nonlinear contribution
compares the full field with its actual damped tangent at that state. It is
not the difference between observed velocities on the nonlinear trajectory
and `U(tau)u`, whose states and moment denominators generally differ.
Similarly, the curve `z_A+z_B` has remainder `R(theta_A)+R(theta_B)`;
evaluating the joint field on that sum would instead use
`R(theta_A+theta_B)`. The following rate certificate concerns the actual joint
flow only, even though its geometric majorants also cover the passage controls.

### 7.2 Adjacent frequencies control transport without a large carrier error

Use the unitary discrete Fourier transform and
`lambda_k=1-cos(2*pi*k/N)`. For either projected band y, the conservative
part of its moment derivative is exactly

\[
 \mathcal D_y(iAy)=i\sum_k(\lambda_k-\lambda_{k+1})
                         \widehat y_k\overline{\widehat y_{k+1}}.
 \tag{27}
\]

Only adjacent modes with both coefficients present contribute. In `B_+`,
write `k=N/4+m`, with `-Q+1<=m<=Q-2`, and put `Delta=pi/N`. Then

\[
 \lambda_k-\lambda_{k+1}
 =-2\sin\Delta\cos((2m+1)\Delta),\qquad
 v_m=\frac{\sin\Delta}{\Delta}\cos((2m+1)\Delta).
 \tag{28}
\]

The sign reverses on `B_-`. Thus the carrier frequency cancels before the
estimate. Since `|2m+1|<=2Q-3`, elementary sine/cosine inequalities give

\[
 |1-v_m|\le\left[\frac16+\frac{(2Q-3)^2}{2}\right]\Delta^2.
 \tag{29}
\]

Also `sum_k |yhat_k||yhat_{k+1}|<=||y||^2`, by Cauchy--Schwarz. Equations
(26)--(29) compare the two conservative center rates with -1 and +1 without
treating a reference trajectory as the actual motion.

Let h be an admitted common band-template error, `U=1+h`, and
`m_0=kappa_Q-h(2+h)>0`. The previous moment argument gives
`||f||,||g||<=a*U` and `|M(f)|,|M(g)|>=a^2*m_0`. Bounds on the three rate
contributions, per center, are

\[
 C=\left[\frac16+\frac{(2Q-3)^2}{2}\right]
          \left(\frac\pi N\right)^2\frac{U^2}{m_0},\qquad
 D=2N\frac e w\frac{U^2}{m_0},\qquad
 V=\frac{32N}{3\pi}a^2\frac U{m_0}.
 \tag{30}
\]

Here C bounds the conservative deviation from the indicated unit speed.
For D, use both band norms in
`||A(f+bar(g))||<=2(||f||+||g||)` and
`|D_y(v)|<=2||y||||v||`. The clock coefficient `sigma=pi*e/w` cancels the
pi in (26); the conjugate-band loss term is not discarded. For V, the joint
bootstrap `||z||<2a` and `||R||<=(4/3)||theta||^3` give
`||R||<(32/3)*a^3`. Neither the sharper cubic-response radii nor an evaluated
response are needed. V bounds the magnitude of a nonlinear contribution;
it does not determine its sign.

### 7.3 Uniform strict passage of the original source family

Use the original inputs and the already admitted `h=13/250`. Direct rational
inequalities give `U^2/m_0<5/4`, `U/m_0<6/5`. With
`(pi/1024)^2<1/106000`, `1024*e/w<1/1000` and `pi>3`, (30) yields

\[
 C<\frac1{200},\qquad D<\frac1{400},\qquad V<\frac1{3840},\qquad
 C+D+V<\frac{149}{19200}<\frac1{100}.
 \tag{31}
\]

The nonzero moments and the continuous relative branch were proved throughout
`[0,256]` for every original preparation. Consequently, on that whole interval,

\[
 |c_f'+1|<\frac1{100},\qquad |c_g'-1|<\frac1{100},\qquad
 |s'+2|<\frac1{50},\qquad
 -\frac{101}{50}<s'<-\frac{99}{50}.
 \tag{32}
\]

This is an instantaneous-rate theorem, not a derivative inferred from a
position tube or a numerical trajectory. The observed relative center cannot
stop, reverse or remain fixed for a positive interval. Since its initial
value is positive and its endpoint value negative by (20), the two registered
centers coincide exactly once in this interval. This does not assert that
every coincidence observable crosses its threshold only once.

Integrating the strict negative rate strengthens (21): residence of s in a
fixed interval of width W lasts at most `50*W/99`. This is a restriction on
the declared relative coordinate, allowing deformed profiles. It does not
exclude every internal, wider or shorter-lived common identity. Neither the
rate bound nor nonzero-moment availability is extended beyond tau=256.

<a id="band-intensity-rate-nonclosure"></a>
### 7.4 Equal observed geometry can have a different nonlinear rate

The geometry still does not supply a generally autonomous rate law. The
following exact **static counterexample** uses the same complete positive-loss
law on `C_12`, with `Q=2`, `B_+={2,3,4}` and `B_-={8,9,10}`. It is not a
member of the original C1024 source family and makes no claim about closure
restricted to that family.

Choose `0<q<pi/4`, for example `q=1/16`, and set

\[
 u_j=q[\cos(\pi j/3)+\cos(\pi j/2)],\qquad
 Z_F=u\ (x=u,\theta=0),\quad Z_P=iu\ (x=0,\theta=u).
 \tag{33}
\]

Both have zero means and squared norm `12*q^2`; the phase-only state's largest
edge gap is `2q<pi/2`. They can be arbitrarily small. Multiplication by i
preserves every band intensity pointwise, hence both band norms, every K_ell,
both moments and both observed centers agree. Direct character sums give
`M_+=M_-=3*q^2>0`, so both states have the same regular local relative center
`s=0`. Multiplication by i here changes the allocation between form and
primitive phase; it is not a symmetry of the complete law.

At the form-only state, the actual field is `(-sigma+i)Au`, and

\[
 M_\pm'=\frac{3q^2}{2}(-3\sigma\mp i),\qquad
 s_F'=-\frac6\pi.
 \tag{34}
\]

At the phase-only state, the field is the real vector `F=S(u)/2`. For this
finite sum write tilde for `1/N` Fourier coefficients, so
`Fhat_k=sqrt(12)*Ftilde_k`; the profile has `utilde_k=q/2` at modes 2, 3, 9,
10. Its two relevant field coefficients and the adjacent zero coefficient are

\[
 \widetilde F_2=\frac{\sin(q/2)-\sin(2q)-\sin(3q/2)}{12},\qquad
 \widetilde F_3=-\frac{2\sin(q/2)+\sin(2q)+2\sin(3q/2)}{12},\qquad
 \widetilde F_4=0.
 \tag{35}
\]

Substitution in (25), or direct sums in site coordinates, gives

\[
 M_\pm'=\mp\frac{iq}{2}[3\sin(q/2)+\sin(3q/2)],\qquad
 s_P'=-\frac{6\sin(q/2)+2\sin(3q/2)}{\pi q}.
 \tag{36}
\]

Since `sin(t)<t` for positive t,

\[
 s_P'-s_F'=\frac2{\pi q}[3q-3\sin(q/2)-\sin(3q/2)]>0.
 \tag{37}
\]

The actual damped tangent evaluated at either state gives `s'=-6/pi`.
Thus the form-only state has zero nonlinear correction, whereas the
phase-only state has the strictly positive correction (37). Equal observed
geometry does not determine even this nonlinear contribution: the lost
form/phase allocation is consumed by the sine pressure. Both displayed rates
remain negative in the chosen small domain. The positive correction slows
passage; it establishes neither a restoring force nor retention at `s=0`.

**Scope of the completed rate question.** The original source family has
strict one-pass relative motion over the admitted interval, while a general
instantaneous rate law based only on the band intensities/centers fails the
projectability test. The two statements have different source/support scope
and are compatible. Neither supplies a full NFR identity, a physical force
or a new constitutive law. Any further retention claim needs an independently
justified internal relation and complete-state or memory treatment of the
information it consumes.

The [retention-mechanism owner](LOCALIZED_RETENTION_MECHANISMS.md)
examines that boundary: supplied reflection and the uncoupled period-four
energies do not establish formation. A different on-grid carrier admits a
cubic focusing comparison, with separate source, residual and maintenance
obligations. It does not alter the original family's passage theorem.

<a id="6-shared-arithmetic-and-controls"></a>
<a id="7-shared-arithmetic-and-controls"></a>
## 8. Shared arithmetic and controls

The private `_bound_cycle_consensus_basin` reader in the
[transport owner](../../src/tnfr/physics/_localized_identity_transport.py)
re-admits the complete positive-loss model, cycle size and independent initial
norm bound before computing (2)--(5). A failed strict bound yields unavailable
sufficiency; it is not a verdict about an evaluated state or finite formation.
Existing finite transport/source reports and their schemas remain separate.

The [geometry controls](../../tests/physics/test_localized_cycle_geometry.py)
check the actual storage derivative and clock, independent source bounds,
strict admission boundaries, the low-storage winding counterexample and
the period-four reduction from direct neighbor rows.
They do not execute a trajectory or replay the frozen response producer.
The proof and source premises, rather than a passing diagnostic score,
establish the conditional all-time claim.

The private `_bound_localized_mobile_geometry` reader in the
[mobile budget owner](../../src/tnfr/physics/_localized_mobile_interaction.py)
rebuilds the preparation budgets from fresh primitives. It bounds the shared
three-history error, band norms, tails, first moments and relative registration
with outward arithmetic. Moment, branch and proximity availability remain
separate; no old identity or cubic-response verdict is treated as a premise.
The [relative-geometry controls](../../tests/physics/test_localized_mobile_geometry.py)
exercise independent coefficient/moment identities, pair-weight geometry,
lossy observations, arithmetic boundaries and incompatible inputs. The reader
evaluates neither K on a trajectory nor a formation verdict, and preserves the
existing response-budget schema and frozen evidence.

The same owner's `_bound_localized_mobile_relative_rate` rebuilds those
geometric bounds before using (30). It exposes the conservative deviation,
dissipative correction and nonlinear correction separately. Band-rate
availability, a continuous principal relative branch and a certified strictly
negative relative rate are distinct; failure of the last condition does not
prove reversal. The [rate controls](../../tests/physics/test_localized_mobile_relative_rate.py)
derive the moment differentiation, conjugate-band rows, adjacent-frequency
bound and C12 counterexample independently. No state reader, FFT execution
path or new integrator is installed. The rate certificate is for the actual
joint flow; the other histories' geometric bounds do not transfer their rates.
