# Pair geometry under alternative reciprocal mobility

The distinct complete positive-mobility law, grouping discriminator and protected relative geometry; old waveform, mean and volume claims do not transfer automatically.

Part of [Coarse-graining, coherence geometry and bridge results](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

<a id="sine-pairing-constitutive-scope"></a>

## 20. Constitutive scope of the local grouping mechanism

### 20.1. The common work identity leaves a mobility law to specify

Reuse the [reciprocal mobility class](SINE_CONSTITUTIVE_INFORMATION.md#common-exchange-geometry-and-composition)
on fixed simple connected unit support. Let `L_f` be its combinatorial
Laplacian, `q=L_f*x`, and
`S_i=sum_{j~i} sin(theta_j-theta_i)`. In the conservative sector the
complete unforced rows are

\[
\boxed{\dot x_i=w\nu_i m_i(\theta)S_i,\qquad
\dot\theta_i=(w/\beta)\nu_i m_i(\theta)q_i.}
\]

Here `w,beta>0`, capacities are held and strictly positive, and the
structural clock is fixed. For the present local result the supplied
mobility functions are finite, strictly positive, phase-only and locally
Lipschitz on a circular neighborhood of the preparation. These are
sufficient hypotheses for a unique differentiable local solution, not
conditions inferred from an observed response. A model must specify
its mobility functions and their domain; positivity is not a complete
law. The frozen comparison below uses smooth functions of incident
phase differences, preserving global phase-origin and graph-relabeling
symmetries.

The nodal pressure is the independently supplied map
`p_i=w*m_i*S_i`. The common storage is

\[
E=\frac12\sum_{\{i,j\}}(x_i-x_j)^2+
  \beta\sum_{\{i,j\}}[1-\cos(\theta_j-\theta_i)].
\]

Its gradients are `grad_x E=q` and `grad_theta E=-beta*S`.
Consequently each node's two exchange work terms cancel:

\[
q_i\dot x_i-\beta S_i\dot\theta_i
 =w\nu_i m_i q_iS_i-w\nu_i m_i S_iq_i=0.
\]

No derivative of the mobility is needed for this identity. It does
not select `m_i=1/(pi*d_i)`, establish an invariant volume, or
give all candidates the same trajectory or weighted means. The
[existing conditional classification](SINE_CONSTITUTIVE_INFORMATION.md#global-closure-pressure-comparison)
selects normalized sine pressure only after adding its stated
pairwise-superposition or phase-independent-response premise and
normalization. That selection theorem is not being assumed to hold
for every positive reciprocal mobility.

### 20.2. Positive reciprocal response preserves the local direction of grouping

Retain exactly Section 16's twenty-edge doubled-C5 preparation:
`d=u=1/8`, phases `(0,d,2d,3d,1,1,2,2,3,3)`, forms
`(u,-u,u,-u,0,0,0,0,0,0)`, common capacity `nu=1`, and
`w=beta=1`. Keeping `k=w*nu/beta>0` visible below makes the
common rate factor explicit. Every fine degree is four and the
prepared neighbor form sums vanish, so `q=4*x`. Write `m_i`
for the chosen mobility evaluated at this initial phase state.
The complete phase row gives

\[
(\dot\theta_0,\dot\theta_1,\dot\theta_2,\dot\theta_3)
 =4ku\,(m_0,-m_1,m_2,-m_3),\qquad
\dot\theta_i=0\quad(i=4,\ldots,9).
\]

Keep the same circular squared-chord observation
`D_ij=2-2*cos(theta_j-theta_i)` and the two initially tied
preference margins

\[
M_1=D_{12}-D_{10},\qquad M_2=D_{21}-D_{23}.
\]

Differentiating the chords using the actual phase row, rather than
assuming a phase clock, gives

\[
\boxed{\begin{aligned}
\dot M_1(0)&=8ku\sin d\,(m_0+2m_1+m_2)>0,\\
\dot M_2(0)&=8ku\sin d\,(m_1+2m_2+m_3)>0.
\end{aligned}}
\]

For example, `M1_dot=2*sin(d)*(theta_dot2-2*theta_dot1+theta_dot0)`;
the second expression is
`2*sin(d)*(2*theta_dot2-theta_dot1-theta_dot3)`.
The signs follow solely from this preparation, positive capacity
and mobility, and `0<d<pi`. Equal mobilities are unnecessary.

Section 16.3 already checks every other potential partner at these
same phases. Its strictly positive outsider margins remain positive
on some sufficiently small neighborhood of time zero for each
admitted complete field. The two nonzero margin derivatives resolve
the ties. Thus the original preparation has the same complete mutual
matching `(0,1),(2,3),(4,5),(6,7),(8,9)` for all sufficiently
small positive times, and the same nonmutual choices of Section 16.4
for all sufficiently small negative times. The observed organization
does not require the particular constant sine mobility.

Complete form reversal retains the mobility values, negates `q`
and every initial phase rate, and reverses both margin derivatives.
It therefore reverses the local direction of the observation change.
More strongly, because the complete conservative form row is
phase-only and its phase row is linear in form,
`(-x(-t),theta(-t))` is the solution with reversed initial form.
This exact local time-reversal identity uses the same phase-only
mobility law in both rows. Merely assuming positive state-dependent
mobility without its form-reversal symmetry would not justify that
identity, although the displayed initial sign argument still has
its own pointwise version.

These are local conclusions for each specified law. An arbitrarily
large positive common multiplier already changes its clock speed,
and the class has no shared numerical Lipschitz or acceleration
budget. No uniform positive time interval follows from the strict
initial signs alone. In particular, Section 17's uncertainty box
and window `[1/128,1/64]` are not transferred to another mobility.
Support-compatible observed pairs likewise do not license use of
the sine-specific collective rate, pulse or recurrence formulas.

### 20.3. A frozen relative-rate discriminator

Use only the two previously admitted comparison members

\[
m_i^{(\epsilon)}(\theta)
 =\frac{1+\epsilon(S_i/d_i)^2}{\pi d_i},
\qquad \epsilon\in\{0,1\}.
\]

Both complete rows change together when the mobility changes.
The same source, support, held capacities, coefficients and structural
clock are retained. No parameter is fitted. Before evaluating the
alternative, fix the dimensionless initial-rate observation

\[
\boxed{\mathscr D=
 \frac{\dot M_1(0)-\dot M_2(0)}
      {\dot M_1(0)+\dot M_2(0)}.}
\]

The positive source has a strictly positive denominator; the
form-reversed control has a strictly negative one. Both are valid
denominators. A generic numerical consumer must still report the
ratio unavailable whenever its denominator enclosure includes zero.
The raw margin derivatives remain separate evidence.

A common positive clock rescaling multiplies both derivatives by
the same factor and leaves `mathscr D` unchanged. Adding a common
phase-origin velocity also changes neither derivative, because their
phase-rate coefficient sums are zero. Therefore distinct values
exclude an identification by common clock rescaling and common
phase drift at this source. They do not exclude arbitrary state
redefinitions or establish an independent physical measurement bridge.

Put `s_i=S_i/4` at the frozen preparation. The preceding exact
expressions give

\[
\mathscr D_\epsilon=
\frac{\epsilon(s_0^2+s_1^2-s_2^2-s_3^2)}
 {8+\epsilon(s_0^2+3s_1^2+3s_2^2+s_3^2)}.
\]

Hence `mathscr D_0=0` exactly, without comparing rounded intervals.
To evaluate the other member independently, the required fine currents
are explicitly

\[
\begin{aligned}
S_0&=\sin(2d)+\sin(3d)+2\sin3,\\
S_1&=\sin d+\sin(2d)+2\sin(3-d),\\
S_2&=-\sin(2d)-\sin d+2\sin(1-2d),\\
S_3&=-\sin(3d)-\sin(2d)+2\sin(1-3d).
\end{aligned}
\]

The shared outward rational trigonometric arithmetic at `d=1/8`
certifies the strict bound

\[
\boxed{\frac1{500}<\mathscr D_1<\frac1{400},
\qquad \mathscr D_0=0.}
\]

Rounded displays of the enclosed values are:

| Fixed mobility | `M1_dot(0)` | `M2_dot(0)` | `mathscr D` |
| --- | --- | --- | --- |
| `epsilon=0` | `0.0396852001938463` | `0.0396852001938463` | Exactly `0` |
| `epsilon=1` | `0.0417943681701375` | `0.0415967934157799` | Approximately `0.00236925293520505` |

The rational strict interval, rather than those rounded decimal
displays, establishes separation. Form reversal negates both raw
margin derivatives and leaves their ratio unchanged. Its exact
control follows algebraically; no new trajectory or observation
window was evaluated.

The result separates two questions. Both laws create the same local
nearest-partner ordering from the chosen preparation, so observing
only that qualitative ordering cannot select between them. Their
relative rates nevertheless differ on the same complete state.
Reciprocal work cancellation and the qualitative mechanism are
broader than the selected quantitative constitutive response.

### 20.4. Shared comparison and evidence boundary

`SineExchangeComparison.with_current_squared_mobility(epsilon=...)` in
the [comparison owner](../../src/tnfr/physics/relational_sine_comparison.py)
retains the original capture and declares the alternative mobility,
both complete rows and their storage-work evidence. The
`assess_sine_pairing_mobility` reader in the
[scale owner](../../src/tnfr/physics/relational_sine_scale.py)
evaluates two explicitly supplied chord-margin triples and their
rate contrast, sum and available ratio. Its triple order is
`(observer, alternative, preferred)`, so `(1,2,0)` and
`(2,1,3)` represent exactly `M1` and `M2` above.

The reader uses the shared fine capture, phase-current arithmetic and
actual phase rows. Exact cancellation is retained where the law and
gap identities provide it; overlapping intervals are not promoted to
equality. The generic report evaluates specified instantaneous margins,
not a theorem about every possible preparation. The positive-class
conclusion is the conditional argument of Section 20.2.

The [constitutive comparison contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-pairing-mobility)
and [independent replica tests](../../tests/physics/test_relational_sine_replica.py)
retain both frozen controls, invalid-domain and availability cases,
full-node rates and explicit law provenance. Native Arg dispatch and
the selected sine runtime remain separate. No finite-horizon forecast,
unchanged invariant volume, periodic pulse, microscopic law selection
or physical identification follows from this static comparison.

<a id="sine-replica-constitutive-nonselection"></a>

### 20.5. Recursive inheritance and the equilibrium tangent do not select one law

The [existing composition comparison](SINE_CONSTITUTIVE_INFORMATION.md#common-exchange-geometry-and-composition)
already distinguishes equal replication from independent neighbor
superposition. Its counterfamily and the frozen discriminator above give
a stronger combined scope statement: exact recursive inheritance and the
same small-signal exchange can coexist with different complete dynamics.
The following is a corollary of those admitted laws, not a new constitutive
model or evidence that a hierarchy forms itself.

Take any finite connected simple undirected unit base graph with at least
two nodes, positive held capacities `nu_i`, and fixed `w,beta>0`. Keep `e=0`
and either existing member `epsilon=0` or `epsilon=1`. For an integer
`k>=1`, replace each base node `i` by `k` constituents `(i,r)`, put no edges
within a fiber, and replace each base edge by all `k^2` cross edges.
Supply synchronized fiber states
`x_(i,r)=X_i`, `theta_(i,r)=Theta_i` and capacity `nu_(i,r)=nu_i`.
With `q=L X` and `S_i=sum_j sin(Theta_j-Theta_i)` on the base, direct
neighbor counting gives, for every base preparation,

\[
d_{i,r}^{\rm fine}=k d_i,\qquad
q_{i,r}^{\rm fine}=k q_i,\qquad
S_{i,r}^{\rm fine}=k S_i,\qquad
m_{i,r}^{(\epsilon),\rm fine}=\frac1k m_i^{(\epsilon)}.
\]

Consequently both actual fine rows are the copied base rows:

\[
\dot x_{i,r}=w\nu_i m_i^{(\epsilon)}S_i,\qquad
\dot\theta_{i,r}=(w\nu_i/\beta)m_i^{(\epsilon)}q_i.
\]

Uniqueness preserves synchronization; no capacity or clock rescaling is
needed. Every base edge contributes the same storage on each of its `k^2`
copies, so the full storage satisfies `E_fine=k^2 E_base` on this invariant
submanifold. Applying an integer `ell>=1`-fold construction to the result is,
up to relabeling, the `(k*ell)`-fold construction, with storage factor
`(k*ell)^2`.
This proves finite iterative same-law inheritance for both candidates.
The support, fiber partition and synchronized preparation remain supplied;
it proves neither attraction to these fibers nor generic off-fiber closure,
autonomous hierarchical formation or a fractal dimension. Section 7 retains
the internal variables required away from synchronization for the sine law.

The common equilibrium tangent is equally insufficient for selection. At
any full equilibrium with `q=0,S=0`, the difference between these complete
fields is

\[
\dot x_i^{(\epsilon)}-\dot x_i^{(0)}
 =\frac{\epsilon w\nu_i}{\pi d_i^3}S_i^3,\qquad
\dot\theta_i^{(\epsilon)}-\dot\theta_i^{(0)}
 =\frac{\epsilon w\nu_i}{\beta\pi d_i^3}S_i^2q_i.
\]

Both corrections are cubic in a joint form/phase perturbation, including
at nonconsensus critical targets. Let `H_*` be the cosine phase Hessian
there and `K=diag(nu_i/d_i)`. Their full Jacobian is therefore identical:

\[
J_*=
\begin{pmatrix}
0&-(w/\pi)K H_*\\
(w/(\beta\pi))K L&0
\end{pmatrix}.
\]

When `H_*` is positive on the common-phase quotient, the existing
[conservative tangent argument](RESONANCE_FOUNDATIONS.md#finite-conservative-memory)
gives the same free oscillatory modes, with the common origins treated
separately. The same prescribed infinitesimal input and observation also
have the same linear response. This does not transfer a finite-amplitude
periodic family, its period or its transverse stability. In particular,
the bounded positive-frequency work-port peak theorem requires positive
loss; it is not a theorem about a bounded forced response at `e=0`.

Nevertheless, the already frozen non-equilibrium comparison in Section
20.3 gives `mathscr D_0=0` and `1/500<mathscr D_1<1/400`. Thus the candidates
have identical recursive inheritance and equilibrium linearizations but
different relative rates that a common clock rescaling cannot remove.
No frozen response is rerun for this conclusion. To select a microscopic
law one still needs a justified additional restriction or an independent
discriminator; calling either prepared inheritance or an oscillatory
tangent "fractal resonance" cannot supply that restriction.

The [shared comparison tests](../../tests/physics/test_relational_sine_comparison.py)
check both admitted mobilities on a nonregular-degree base with unequal
per-fiber capacities and successive two- and three-fold replication. They
retain every fine row and check complete rates and storage against independent
values; this is implementation evidence for the corollary, not its proof.

<a id="collective-mean-closure-obstruction"></a>

### 20.6. Universal collective-mean closure forces a constant circular current

The [sine counterexample](SINE_PAIR_STATE.md#74-a-nearby-exact-obstruction-to-autonomous-block-means)
has a stronger class-level counterpart. Let `j` be a bounded real kernel on
the full phase circle. Consider the additive form source

\[
\dot x_i=\frac{w\nu}{\pi d_i}\sum_{k\sim i}j(\theta_k-\theta_i),
\qquad w,\nu>0.
\]

Hold capacity, coefficients, support and clock fixed, with no inputs or
events. Use this form row, optionally with an additional linear form-diffusion
term, which vanishes on the uniform-form preparations below. No other form
source is added. The phase row remains separately supplied; the necessary
form-row test below does not select it.

Replace both vertices of `K2` by pairs and its edge by all four cross-edges,
giving `K2,2`. Pair membership and every attachment are known and unchanged.
Suppose an autonomous collective state retains only each pair's arithmetic
form mean `X`, circular mean direction `Theta`, and held capacity, discarding
internal dispersion. Require this description to reproduce the form-mean
rate for every phase center and every sufficiently small internal dispersion.
It need not be assumed in advance that the reduced rate has the original
kernel's formula; the synchronized preparation already fixes that value.

Give all four nodes the same form. Prepare the first pair at
`(Theta_A,Theta_A)` and compare the second pair at `(Theta_B,Theta_B)`
with `(Theta_B+delta,Theta_B-delta)`, where
`|delta|<delta_0<pi/2`. Both preparations have the same retained state.
In particular the second pair's mean phasor is
`exp(i*Theta_B)*cos(delta)`, whose direction remains `Theta_B` because
`cos(delta)>0`. This uses a regular circular mean and compatible local
lifts, not an angle assigned at a zero resultant. The first pair's two
constituents have identical neighbor lists, so its mean-rate equality,
with `Delta=Theta_B-Theta_A`, requires

\[
\boxed{\quad
j(\Delta+\delta)+j(\Delta-\delta)=2j(\Delta)
\quad(\Delta\in\mathbb R/2\pi\mathbb Z,\ |\delta|<\delta_0).
\quad}
\]

**Classification without an extra regularity assumption.** Fix any allowed
`delta`. Applying this identity successively at `Delta+n*delta` gives

\[
j(\Delta+n\delta)=j(\Delta)
 +n\,[j(\Delta+\delta)-j(\Delta)]\qquad(n\in\mathbb Z).
\]

All arguments are circular. Boundedness of `j` therefore forces the bracket
to vanish. Every sufficiently small phase translation leaves `j` unchanged;
subdividing an arbitrary phase displacement into finitely many such
translations shows that `j` is constant on the circle. No continuity,
differentiability or measurability was used. If the kernel is also odd,
that constant is zero. Conversely a constant kernel passes this source-rate
test, but this alone does not establish closure of the other evolution rows.

Thus no nontrivial bounded odd additive circular current admits the proposed
universal state of collective means and held capacities. The obstruction is
already an instantaneous source mismatch; choosing a phase evolution cannot
repair it. It concerns loss of the resultant's length, a stronger compression
than the [first-moment information premise](SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency),
which retains the full complex resultant and degree. Replacing sine by a
different nonconstant kernel does not remove this information requirement.

The result does not forbid closure on a synchronized invariant family,
restricted phase domains, retained internal coordinates or exact memory.
For example a kernel that is affine on a restricted lifted interval can
pass its local midpoint tests without being globally constant. A capacity
or clock that consumes the discarded dispersion adds information and needs
its own law; it is outside the held, common-clock premise. Native Arg
pressure is nonadditive and is not classified here. Nothing creates a
primitive edge or selects a physical particle, support or interaction law.

The existing [global pair state](SINE_PAIR_STATE.md#sine-global-pair-state) and
[state on asymmetric support](SINE_PAIR_STATE.md#sine-mixed-pair-state) retain the information
needed by their declared sine laws; their coordinates are derived from the
constituents rather than added primitive causes. The
[static information controls](../../tests/physics/test_phase_moment_information.py)
reuse exact rational constituent phasors to compare equal form/phase means
with different resultant lengths and different sine/cubic sources. They
check independent full-neighbor sums and the shared reader's scope, not the
universal classification or future trajectories. No new runtime law or
frozen response is introduced.

<a id="sine-mobility-relative-geometry"></a>

## 21. Protected relative geometry without an assumed invariant volume

### 21.1. Quotient only the common origins of the actual complete law

Retain the exact twenty-edge doubled-C5 support and winding-one target
of Section 12, common positive held capacity and the conservative
current-squared mobility of Section 20. The comparison uses only its
already declared members `epsilon=0` and `epsilon=1`, with
`w=beta=nu=1`. The identities below display the existing positive
coefficients and apply to each fixed finite `epsilon>=0`; no parameter
search is needed. All constituent forms, phases and fixed connections
remain in the model. No input, support event or clock change is added.
Reflecting the target gives the same result for winding minus one:
the cosine storage, spectral gap and radius condition use the same
absolute target increment.

Let `n=10`, `d=4`, `P=I-11^T/n`, `L=L_f` and

\[
S_i(\theta)=\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
m_i(\theta)=\frac{1+\epsilon(S_i/d)^2}{\pi d},
\qquad M=\operatorname{diag}(m_i).
\]

Constant common form translation and common circular phase rotation
are exact symmetries: neither changes `q=Lx`, `S`, `M` or either
complete rate row. In Section 12's target chart write

\[
x=\bar x\mathbf1+v,\qquad
\theta=\theta_*+c\mathbf1+h\pmod{2\pi},\qquad
v,h\perp\mathbf1.
\]

The retained relative state is `(v,h)`, with eighteen continuous
coordinates. Only the two common origins are discarded. Pairwise
form differences and circular phase differences, including internal
member state, remain determined. The common phase origin `c` has
a local continuous lift, rather than a globally defined real mean
on the phase torus.

The induced field in the same structural clock is exactly

\[
\boxed{\dot v=w\nu P M(h)S(h),\qquad
\dot h=(w\nu/\beta)P M(h)Lv.}
\]

Here `S(h)` and `M(h)` mean their evaluation at `theta_*+h`;
the omitted common phase cancels from every gap. No projected
mobility or scalar effective capacity replaces these matrix products.
The removed origins obey the independently determined rows

\[
\dot{\bar x}=\frac{w\nu}{n}\mathbf1^TMS,\qquad
\dot c=\frac{w\nu}{\beta n}\mathbf1^TMLv.
\]

Their initial values and these integrals reconstruct the full state
as long as the chart is valid. Discarding them therefore limits
the claim to relative structure; it does not prove that the original
means are constant. Projecting out their computed motion also does
not reparametrize time or supply an independently chosen phase clock.

### 21.2. The existing storage barrier protects the relative pattern

The exact same fine storage descends to the quotient:

\[
E(v,h)=\tfrac12 v^TLv+
\beta\sum_{\{i,j\}}[1-\cos(\theta_{j,*}-\theta_{i,*}+h_j-h_i)].
\]

Since `q=Lv` and `S` both have zero sum, differentiating using the
projected field cancels the exchange terms exactly:

\[
\dot E=w\nu q^TPMS-w\nu S^TPMq=0.
\]

The projections act trivially on the left vectors `q,S`; the
remaining cancellation uses the symmetry of the same diagonal `M`
in both rows. Conservation holds along the changed law itself, not
along a substituted sine trajectory.

Keep every radius, phase-lift and storage hypothesis from Section 12:

\[
\alpha=2\pi/5,\quad E_*=20\beta(1-\cos\alpha),\quad
0<r,\quad \alpha+\sqrt2r<\pi/2,
\]

\[
c_r=\cos(\alpha+\sqrt2r)>0,\qquad
\kappa_f=\frac{5-\sqrt5}{2}\min(1,\beta c_r),\qquad
0<\eta_*<\kappa_f r^2.
\]

The symbol `eta_*` is the existing excess-storage ceiling, renamed
here to distinguish it from the mobility parameter. Put
`Z^2=||v||^2+||h||^2` and `mathcal E=E-E_*`. The unchanged
target Hessian and fine spectral gap give the same geometric bound

\[
\mathcal E\ge\kappa_f Z^2\qquad(Z\le r).
\]

This inequality concerns the storage and target chart; its derivation
does not contain the mobility. Define the relative family

\[
\boxed{\mathcal V=
\{(v,h):Z^2<r^2,\quad \mathcal E<\eta_*\}.}
\]

Every state in `V` is protected for all positive and negative times
under the declared mobility. If there were a first exit at `Z=r`,
the same conserved excess would have to satisfy both
`mathcal E<eta_*` and `mathcal E>=kappa_f*r^2`, a contradiction.
More explicitly,

\[
\boxed{Z(t)^2\le\frac{\mathcal E(0)}{\kappa_f}
<\frac{\eta_*}{\kappa_f}<r^2\qquad(t\in\mathbb R).}
\]

There is no finite-time escape hidden in this statement. On the phase
torus `|S_i|<=d` and
`1/(pi*d)<=m_i<=(1+epsilon)/(pi*d)`. In particular every full
form rate is bounded by `w*nu*(1+epsilon)/pi`. The full form
can grow at most linearly on a finite time interval, and the phase
rates are at most linear in that finite form. Smooth continuation
therefore exists in both time directions. Alternatively, the relative
field is smooth on a neighborhood of the compact trapped closure;
its two origin rows stay bounded there and can be integrated separately.

The family `V` is open, has finite positive eighteen-dimensional
volume, and has compact closure inside the valid relative chart.
Its invariance does not require a bounded interval for the absolute
form mean. Every fine edge remains acute in its target-compatible
lift, and every cycle retains the target winding. Section 14's
phase-separation argument therefore continues to identify the same
five constituent pairs. These consequences use the proved geometry,
not the sine-specific rates or an invariant-volume assumption.

The individual pair swaps remain exact symmetries of this particular
mobility: they permute degrees, currents and both rows together.
Consequently each synchronized tip set is invariant. Smooth uniqueness
also prevents a state outside such a set from reaching it in finite
time. Removing all five tip sets gives the corresponding open invariant
family `V_*`, still of finite positive relative volume. This fact alone
does not establish a nonzero internal velocity, an angular circulation
bound or a repeating waveform; those are separate dynamical questions.

The conclusion is conditional maintenance of an already admitted
organization. The earlier local-transition preparation is not thereby
placed inside this protected family. Since the family is invariant in
both time directions, it is not a finite-time capture target for states
outside it under this same complete autonomous law. This uses the
[two-sided invariance argument](RESONANCE_FOUNDATIONS.md#conservative-formation-boundary),
not the unproved invariant measure for positive `epsilon`. A measure-based
obstruction to asymptotic capture requires separate recurrence hypotheses.

### 21.3. Mean motion and divergence change although storage does not

The original weighted-mean calculation exposes the difference without
choosing a new response preparation. More generally, on the admitted
unit support with held positive capacities put
`rho_i=d_i/nu_i` and `W=sum_i rho_i`. For the same mobility family,
the old weighted form mean and a locally lifted phase mean satisfy

\[
\boxed{\begin{aligned}
\frac{d}{dt}\frac{\sum_i\rho_i x_i}{W}
 &=\frac{w\epsilon}{\pi W}\sum_i\frac{S_i^3}{d_i^2},\\
\frac{d}{dt}\frac{\sum_i\rho_i\theta_i}{W}
 &=\frac{w\epsilon}{\beta\pi W}
       \sum_i\frac{S_i^2q_i}{d_i^2}.
\end{aligned}}
\]

The constant-mobility terms vanish because `sum S_i=sum q_i=0`.
The remaining terms are actual drift, not a fitted correction to a
conserved quantity. The weighted phase expression still depends on
declared continuous lifts; it is not a global scalar observable on
the phase torus. For the doubled-C5 coefficients frozen above these
two right-hand sides become respectively
`epsilon*sum(S_i^3)/(640*pi)` and
`epsilon*sum(S_i^2*q_i)/(640*pi)`.

Write `C_i=sum_{j~i} cos(theta_j-theta_i)`, so
`partial_theta_i S_i=-C_i`. The divergence of the full field is

\[
\boxed{\operatorname{div}F
=-\frac{2w\epsilon}{\beta\pi}
  \sum_i\frac{\nu_i q_i S_i C_i}{d_i^3}.}
\]

The form row has no form derivative; the displayed term is the
diagonal phase derivative of the changed phase row. This is also the
Euclidean divergence of the induced relative field. To see this
without treating constrained coordinates as independent, choose a
constant linear basis for the relative form and locally lifted phase
coordinates and append their two common origins. The complete field
is independent of the two origin values. Their two diagonal derivative
entries are zero, leaving precisely the relative-coordinate trace.
The same argument applies to node-reference differences instead of
the centered basis.

For `epsilon=0` both mean drifts and the divergence vanish identically.
For `epsilon=1` these identities fail even arbitrarily near the
existing protected target, not merely outside its acute domain. An
analytic local check makes this explicit without a new numerical
preparation. Put `c_alpha=cos(alpha)>0` and let `H` be any nonzero
centered direction. At `h=s*H`,

\[
S=-s c_\alpha LH+O(s^2),\qquad
C=d c_\alpha\mathbf1+O(s).
\]

Taking `v=-s*H` in the same open family gives

\[
\operatorname{div}F
=-\frac{2w\nu\epsilon c_\alpha^2}{\beta\pi d^2}
    s^2\|LH\|^2+O(s^3),
\]

which is nonzero for sufficiently small nonzero `s` when `epsilon>0`.
For the form mean, `L` maps the centered subspace invertibly onto
itself. That subspace contains vectors `Q` with `sum Q_i^3!=0`.
Choose the direction with `LH=Q`; then

\[
\begin{aligned}
\dot{\bar x}
&=-\frac{w\nu\epsilon c_\alpha^3}{n\pi d^3}
   s^3\sum_i Q_i^3+O(s^4),\\
\dot c
&=-\frac{w\nu\epsilon c_\alpha^2}{\beta n\pi d^3}
   s^3\sum_i Q_i^3+O(s^4).
\end{aligned}
\]

Both are likewise nonzero locally. These are directional proofs about
the existing open family, not a new selected-amplitude experiment or
radius search. Thus the old absolute-mean slab and ordinary-volume
proof cannot simply be inherited. A zero drift or divergence at a
particular captured state would not restore either all-state identity.

### 21.4. What is still required for an almost-everywhere return claim

The relative trapping theorem concerns **every** admitted initial
state. Recurrence requires a separate measure argument. For
`epsilon=0`, the relative field is divergence-free; ordinary relative
volume restricted to `V` or `V_*` is finite and invariant. The
[existing finite-measure proof](RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
then gives relative-state recurrence for almost every point with
respect to that measure. Full-state sine recurrence retains the
additional absolute-mean and phase-torus premises of its owner.

For `epsilon=1`, nonzero Euclidean divergence prevents the same
volume proof, but does not disprove recurrence or exclude a different
invariant density. To recover the comparable almost-everywhere
statement on the relative family, one sufficient missing object is
a finite invariant measure equivalent to its natural relative volume.
For a positive differentiable density `varrho(v,h)`, its exact
obligation is

\[
\operatorname{div}_{v,h}(\varrho F_{\rm rel})=0,
\qquad 0<\int_{\mathcal V}\varrho\,dv\,dh<\infty,
\]

with equivalence to the preparation measure and compatibility with
the complete two-sided flow. Positivity and smoothness on a
neighborhood of the compact closure would supply useful sufficient
finiteness and equivalence conditions. A singular invariant measure
concentrated at the stationary target exists, but its recurrence says
nothing about almost every initial state in the open family.

Even a density depending on phases alone requires a genuine
integrability check. Let `varrho(theta)` be global-phase-shift invariant,
write

\[
B_i=\partial_{\theta_i}\log m_i,\qquad
a_i=\partial_{\theta_i}\log\varrho.
\]

Because `q=Lx` spans the centered subspace, invariance for every
relative form would require a common value

\[
\nu_i m_i(a_i+B_i)=\gamma(\theta)\quad\hbox{for every }i.
\]

The shift condition `sum_i a_i=0` fixes the only candidate gradient:

\[
\boxed{a_i=-B_i+
\frac{\sum_j B_j}{\sum_j1/(\nu_jm_j)}\frac1{\nu_i m_i}.}
\]

This one-form must be an exact gradient on the relative chart, with
the necessary positivity, normalization and any extension conditions.
No such solution is established here. A product of reciprocal nodal
mobilities cannot be assumed to solve it: differentiating that product
also introduces neighboring mobilities' derivatives. Moreover failure
of this phase-only ansatz would not exclude a density depending on
both relative form and phase. The general equation above, rather than
an unproved invariant-volume label, is the remaining obligation.

This theorem therefore certifies relative geometry, while leaving the
alternative's comparable almost-everywhere recurrence claim unavailable.
It neither labels that dynamics nonrecurrent nor imports the old
internal-pulse period or circulation theorem. Relative return, if
subsequently proved, would itself not ensure return of discarded
common origins or provide a deadline for a selected state.

### 21.5. Shared balance and admission evidence

The existing mobility comparison retains its own complete-law
provenance. Its `relative_balance(reference_node=...)` reader uses captured fine rows,
phase currents and cosine sums to report reference-node coordinate
rates, weighted-mean drift and full/relative divergence. Exact
constant-mobility cancellations remain distinct from finite
state-specific interval enclosures.

`assess_sine_mobility_geometry` in the
[scale owner](../../src/tnfr/physics/relational_sine_scale.py)
reuses the actual doubled-cycle support, symbolic target, supplied
phase lifts and shared source storage/barrier admission. It separates
an admitted relative family, a trapped captured source and membership
in the stricter family with synchronized tips removed. The same
geometry bounds do not relabel the alternative as the original law.
The report retains recurrence as unavailable when its invariant-measure
obligation is unresolved, rather than turning that absence into a
negative trajectory verdict.

The source capture is not treated as the exact symbolic twist merely
because rounded phases are close to it. No new radius, uncertainty
budget, preparation scan or frozen response is used. The
[relational contract](../../docs/contracts/relational/SINE_PAIR_DYNAMICS.md#sine-mobility-geometry)
and [independent replica tests](../../tests/physics/test_relational_sine_replica.py)
own the executable admission and balance checks. Protected relative
organization remains distinct from autonomous formation, physical
identification and selection of the microscopic mobility.
