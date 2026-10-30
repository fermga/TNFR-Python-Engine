# Coherence Observation: Finite Profiles, Lost Information and Dynamics

**Status:** Exact conditional identities and finite counterexamples for the
pressure-derived coherence observation. The implementation controls use fixed
fixtures, not trajectories or empirical data. This chapter justifies the
additional profile readout without introducing a new constitutive law,
universal correlation length or canonical state coordinate.

The [structural field guide](../docs/STRUCTURAL_FIELDS_TETRAD.md#coherence-length)
owns the public contracts. The
[tetrad reconstruction scope](MINIMAL_STRUCTURAL_DEGREES.md) distinguishes four
diagnostic channels from a sufficient state. The [research program](RESEARCH_PROGRAM.md)
owns task selection; this chapter is a result and dependency owner, not a plan.

## 1. The observed quantity and its assumptions

Write `p = DeltaNFR`. For admitted finite stored pressure, the estimator reads

$$c_i=\frac{1}{1+|p_i|}.$$

This is the pressure-only specialization of the canonical coherence kernel,
with the form-rate contribution set to zero. It is not the full runtime
coherence. The fixed numerical pressure scale implicit in this formula belongs
to the coherence definition; changing pressure units without transforming that
scale is not an observation-preserving coordinate change.

Let $P_r$ be the sampled distinct-node pairs at positive represented distance
$r$, and write angle brackets for the arithmetic mean over that finite set.
The fitted profile is

$$q_r=\langle c_i c_j\rangle_{P_r}.$$

Undirected pairs are counted once. Directed pairs retain outgoing source and
target roles. Geometry uses explicit edge `length`, else the compatibility
`weight`, else one, with parallel-edge minimum. Zero-distance and unreachable
pairs do not enter the fit. A directed distance need not be symmetric.
Source selection and the global node population are distinct: above the
documented sampling boundary, only the declared insertion-order source sample
contributes pairs, while the global coherence mean still uses all graph nodes.

The existing policy requires ten positive-distance pairs, at least two pairs
per retained distance bin, and at least three bins whose product means exceed
`1e-9`. It fits

$$y_r=\log q_r\approx a+\beta r,\qquad \xi_C=-1/\beta$$

with equal weight per admitted bin, and accepts a finite negative slope
producing a finite positive length. Exact constant products are rejected.
These counts and the product floor are estimator policies. Acceptance does
not include a residual threshold, a stationary-field hypothesis, a dynamical
model or independent statistical replication.

The fit length uses the declared path-length unit. Its separately reported
normalized-Laplacian fallback is dimensionless and topology/conductance-based.
A fallback is not a measurement of the missing product-decay fit. In
particular, changing only explicit metric lengths does not rescale that
normalized spectral object.

## 2. Exact finite moments and distance-conditioned endpoints

For a nonempty node set, define the global mean and population variance

$$\mu=\frac1N\sum_i c_i,\qquad
v=\frac1N\sum_i(c_i-\mu)^2.$$

For every nonempty pair bin, define

$$M_r=\frac12\langle c_i+c_j\rangle,\qquad
S_r=\frac12\langle c_i^2+c_j^2\rangle,$$

$$K_r=\langle(c_i-\mu)(c_j-\mu)\rangle,\qquad
V_r=\frac12\langle(c_i-\mu)^2+(c_j-\mu)^2\rangle,$$

$$\gamma_r=\frac12\langle(c_i-c_j)^2\rangle.$$

Here $K_r$ is a globally centered pair product, $V_r$ is the bin's endpoint
second moment about the global mean, and $\gamma_r$ is a semivariogram.
These names denote finite deterministic observations. They do not by
themselves establish a random-field covariance or stationarity.

### 2.1 Finite-bin identity

Expanding each product and square gives, without probabilistic assumptions,

$$K_r=q_r-2\mu M_r+\mu^2,$$

$$V_r=S_r-2\mu M_r+\mu^2,$$

$$\boxed{\gamma_r=S_r-q_r=V_r-K_r.}$$

Thus $K_r=q_r-\mu^2$ requires $M_r=\mu$ on this strictly positive coherence
domain. The further replacement $\gamma_r=v-K_r$ requires $V_r=v$.
Neither follows from grouping finite graph pairs by distance. Boundary,
degree, component, direction and source-sampling effects can change the
endpoint population in each bin.

The identities also explain the dependencies: $q_r$, $\mu$, $M_r$ and $S_r$
determine $K_r$, $V_r$ and $\gamma_r$. Reporting all of them can clarify
different structural questions, but does not create independent dimensions.

### 2.2 Endpoint-role centering is a different observation

For directed pairs let $a_r=\langle c_i\rangle$ and
$b_r=\langle c_j\rangle$. The bin-centered endpoint covariance is

$$H_r=\langle(c_i-a_r)(c_j-b_r)\rangle=q_r-a_r b_r.$$

It differs from the globally centered product by

$$K_r=H_r+(a_r-\mu)(b_r-\mu).$$

For an undirected bin, symmetrizing the two endpoint roles makes
$a_r=b_r=M_r$, hence $H_r=q_r-M_r^2$. This convention removes dependence on
the arbitrary order used to enumerate an unordered pair. Directional roles
remain distinct on a directed graph. The shared profile reader declares
this convention explicitly.

A bin with a constant source can have $H_r=0$ while its raw product changes
with distance. This is useful separation of endpoint levels from within-bin
variation, not evidence that an independent dynamic correlation has vanished.

## 3. Exact radial star: a perfect fit with different centered structure

Consider seven nodes: one center and two leaves at each radius $1,2,3$.
Give each edge its radius as explicit length and unit conductance. Fix
zero phase and unit capacity. Prepare

$$c_0=\frac12,\qquad c_i=2^{-\ell_i-1},\qquad p_i=c_i^{-1}-1.$$

The pressures are $1$ at the center and $3,7,15$ on the corresponding leaf
pairs. Every path joins either center to leaf or two leaves through center,
so for every distinct pair,

$$c_i c_j=2^{-d(i,j)-2}.$$

The distance-bin counts are

| Distance | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| Pair count | 2 | 3 | 6 | 5 | 4 | 1 |

The fit uses distances $1$ through $5$ and yields

$$a=-2\log2,\qquad \beta=-\log2,\qquad \xi_C=1/\log2.$$

The exact-real residual is zero. This result follows from a supplied radial
preparation and additive star distances. No stochastic correlation, propagation
mechanism, collective object or dynamic length has been derived.

### 3.1 Centering can reverse the interpretation

The global moments are

$$\mu=\frac{11}{56},\qquad v=\frac{129}{6272}.$$

At distance one,

$$q_1=\frac18,\quad M_1=\frac38,\quad
\gamma_1=\frac1{32},\quad K_1=\frac{51}{3136}.$$

In contrast, subtracting only the global squared mean gives
$q_1-\mu^2=271/3136$. That subtraction omits the endpoint correction.

At distance five, all four pairs join radius-two to radius-three leaves:

$$q_5=\frac1{128},\qquad M_5=\frac3{32},\qquad
K_5=\frac{15}{1568}>0,$$

whereas $q_5-\mu^2<0$. The naive subtraction even gives the wrong sign.
The actual centered products at distances one through six have signs
$+,-,-,-,+,+$, despite the perfect decreasing raw exponential. A scalar
length and the full centered profile therefore answer different questions.

The last bin has only one pair and is excluded from regression. It remains
a valid finite observation and must remain visible with its exclusion reason.

### 3.2 Directed exponential control

On an outward directed star, use three leaves at each radius $1,2,3,4$,
$c_0=1$ and $c_i=2^{-\ell_i}$. The twelve reachable distinct-node pairs
are center-to-leaf pairs. Their products give $q_r=2^{-r}$,
$\xi_C=1/\log2$ and zero exact-real log residual.

Each bin's source coherence is constant, so its endpoint covariance $H_r$
is zero. A perfect raw exponential and identically zero bin-centered endpoint
covariance coexist. Neither observation may silently substitute for the other.

## 4. What the scalar compresses or fails to identify

### 4.1 Coherence amplitude and pressure sign

For any $A>0$ with $A c_i\leq1$, replace $c_i$ by $A c_i$ and independently
prepare pressure magnitude $|\widetilde p_i|=(A c_i)^{-1}-1$. Then

$$\widetilde q_r=A^2q_r,\qquad
\widetilde a=a+2\log A,\qquad
\widetilde\beta=\beta.$$

Hence the mathematical fit length is unchanged on the same admitted bins.
The global mean scales by $A$; variance, centered products and semivariogram
scale by $A^2$. Residuals in log space and $R^2$ are unchanged. The fixed
product floor can change bin admission under arbitrary amplitude changes;
the prescribed factor-one-half directed-star control stays above that floor.

Independently replacing $p_i$ by either sign of its magnitude preserves every
coherence-derived profile, including the nodewise $c_i$. Reversing one
radius-one leaf pressure in the radial star changes its signed products with
the other nonzero pressures while preserving the entire coherence profile.
Signed pressure or the admitted rate $\nu_i p_i$ can detect that lost sign.

This proves added information relative to $\xi_C$ and coherence alone. It
does not prove independence from the full tetrad: its signed potential
$\Phi_s=B_Gp$ can already distinguish those preparations. For a known
invertible $B_G$, the complete potential field determines pressure and every
coherence-derived observation. Conditioning and any noninjective cases
require their own analysis; a global potential mean discards more information.

### 4.2 Constant fields, plateaus and finite observation range

If every $c_i=c$, then $q_r=c^2$, $K_r=H_r=\gamma_r=0$ and the product
fit has no decay slope. A connected graph can still supply a finite spectral
fallback. That fallback describes the separately declared generator and
does not turn a constant measured profile into a finite spatial decay.

For a separately justified smooth profile

$$q(r)=\mu^2+B e^{-r/\ell},\qquad B>0,$$

direct differentiation gives the local inverse log-decay rate

$$\left[-\frac{d}{dr}\log q(r)\right]^{-1}
=\ell\left(1+\frac{\mu^2}{B}e^{r/\ell}\right).$$

It grows with distance even though the supplied connected component has
constant decay parameter $\ell$. Thus an uncentered exponential fit can
confound a plateau with its decay component. This is a conditional algebraic
example, not a claim that every graph profile has this form. Negative
centered products and oscillations likewise cannot be characterized by one
positive decay length. A length much larger than the observed distance span
also extrapolates beyond that span; the scalar alone does not quantify that
extrapolation.

### 4.3 An accepted oscillatory profile with poor exponential fit

Keep the directed-star support of Section 3.2, but assign leaf coherence
$1/2,1/16,1/2,1/16$ at the four successive radii. Its twelve products give

$$\frac{\log q_r}{\log2}=(-1,-4,-1,-4).$$

Ordinary equal-bin least squares has $\bar r=5/2$,
$\overline{\log q}=-(5/2)\log2$ and
$\sum(r-\bar r)^2=5$. Therefore

$$\beta=-\frac35\log2,\qquad a=-\log2,
\qquad \xi_C=\frac5{3\log2}.$$

The log residuals divided by $\log2$ are
$(3/5,-9/5,9/5,-3/5)$. Their squared sum is $36/5$, while the total
centered squared sum is $9$. Consequently

$$R^2=1-\frac{36/5}{9}=\frac15.$$

The existing policy accepts this positive length. Acceptance is consistent
with its documented negative-slope rule; it is not proof that the profile
is approximately exponential. The residuals and nonmonotonic bins expose
what the scalar hides. Introducing a quality threshold would be a separately
justified estimator policy, not a conclusion of this witness.

### 4.4 Metric scaling and exact-bin fragility

Multiplying all positive lengths by $s>0$ preserves pair membership under
exact scaling, moves every bin from $r$ to $sr$, and divides the mathematical
slope by $s$. Thus $\widetilde\xi_C=s\xi_C$. Profile moments and log
residuals are unchanged. Floating arithmetic and floor/rank admission remain
part of an implementation check; the finite control uses a factor of three.

The exact-equality bin rule has another limitation. Perturb the radial-star
edge lengths to

$$\widetilde\ell_i=b_i+2^{-20}2^{i-1},\quad i=1,\ldots,6,
\qquad (b_i)=(1,1,2,2,3,3).$$

Every center-to-leaf distance has an integer base and a one-bit perturbation;
every leaf-to-leaf distance has an integer base and a two-bit perturbation.
The largest perturbation coefficient is $32+16=48<2^{20}$. Distinct integer
bases cannot collide, and equal integer bases have distinct binary subset
sums. All twenty-one positive pair distances are therefore different.
These dyadic edge and path values are exactly representable in binary64.

Repreparing $c_0=1/2$ and $c_i=2^{-\widetilde\ell_i-1}$ still gives the
exact-real identity $c_ic_j=2^{-d(i,j)-2}$. Nevertheless every bin has one
pair, so no bin meets the fit count rule. Floating representations of these
new coherence values need not satisfy the exponential identity exactly;
the bin-count rejection does not depend on that rounding.

This proves that a small metric perturbation can destroy fit availability
without destroying the exact-real exponential preparation. Replacing exact
bins with distance windows or a pairwise regression would change weighting,
resolution and bias. Such an estimator requires an explicit contract and
comparison; the current reader does not silently make that replacement.

## 5. Temporal dependence requires a complete law

On a time interval with fixed support, metric, sampled pairs and pressure
signs, differentiable independently specified pressure gives

$$c_i'=-\operatorname{sign}(p_i)p_i'c_i^2,$$

$$q_r'=\langle c_i'c_j+c_ic_j'\rangle.$$

If the admitted fit bins stay fixed, their products remain positive and the
negative slope remains admitted, then

$$\beta'=
\frac{\sum_r(r-\bar r)(q_r'/q_r)}{\sum_r(r-\bar r)^2},
\qquad \xi_C'=\frac{\beta'}{\beta^2}.$$

The derivative formula follows by differentiating the ordinary least-squares
slope on this fixed branch. The nodal identity $x_i'=\nu_i p_i$ supplies
neither $p_i'$ nor these hypotheses. At a zero-pressure crossing the absolute
value may require one-sided derivatives; changing metric bins, admission or
support introduces a separate nonsmooth or event problem. A measured change
also needs timestamps and an observation protocol, not just a predicted rate.

### 5.1 Same full tetrad and law, different length derivative

Use the radial star of Section 3 and choose dimensionless coordinates and
clock with the complete supplied law

$$x_i'=p_i,\qquad p_i'=-x_i.$$

Capacities are fixed at one; phases are fixed at zero; support, metric and
conductance are held; there is no forcing or event. This regular oscillator
model satisfies the unforced nodal row. It is a conditional counterexample,
not a selected or necessary TNFR pressure law.

Compare two preparations with the same positive pressures from Section 3,
but $x_i=0$ at every node in the first and $x_i=1$ in the second. They have
the same complete initial tetrad and the same initial form rate. Their
different EPI is omitted by that snapshot readout. Positivity of pressure and
admission of the same five fit bins persist for sufficiently small time.

In the first preparation $p_i'=0$, so $q_r'=0$ and $\xi_C'=0$ initially.
In the second $p_i'=-1$ and $c_i'=c_i^2$. Within every bin the initial product
is constant, giving

$$\frac{q_r'}{q_r}=\langle c_i+c_j\rangle
=\left(\frac34,\frac7{12},\frac7{16},\frac3{10},\frac3{16}\right),
\quad r=1,\ldots,5.$$

Since $\bar r=3$ and $\sum(r-3)^2=10$,

$$\beta'=
\frac{-2(3/4)-(7/12)+(3/10)+2(3/16)}{10}
=-\frac{169}{1200}.$$

Therefore the second preparation has

$$\boxed{\xi_C'=-\frac{169}{1200(\log2)^2}}$$

instead of zero. A perfect common initial fit and a fixed complete law do
not make the tetrad, or the expanded static coherence profile, autonomous.
This proof uses only the instantaneous analytic jet; no trajectory or
empirical derivative is claimed. Retaining the already existing EPI state
distinguishes this pair; no new fundamental coordinate is needed.

## 6. Related channels and what each could add

The first comparison baseline matters: additional information relative to
one scalar can already be present in the full tetrad or primitive state.

| Observation | Information recovered or exposed | Dependency and remaining obligation |
| --- | --- | --- |
| Global coherence mean, variance and fit intercept | Amplitude lost by the length | Derived from the same pressure magnitudes; does not recover sign or phase |
| Full product profile, counts and residuals | Oscillation, poor fit, unresolved bins and observed distance span | Derived from stored pressure and geometry; pairs are not independent replicates |
| Centered products, endpoint moments and semivariogram | Level effects versus pair contrast and distance-conditioned endpoint bias | Exact dependencies in Section 2; no independent state dimension or automatic stationarity |
| Signed pressure and capacity-conditioned form rate | Sign and capacity information absent from coherence-only observations | Potential may already encode pressure; rate prediction needs the selected source and complete nodal-row contract |
| Distance-resolved circular phase alignment | Phase organization invisible to pressure-only coherence | Its circular variogram is algebraically redundant; the full tetrad can already distinguish some preparations |
| Oriented phase differences and cycle winding | Ordering and circulation discarded by unsigned/local phase summaries | Needs phase admission, cycle orientation and the declared state-equivalence relation |
| Timestamped pressure/coherence response or independently specified pressure derivative | Temporal distinction absent from a static observation | Requires one complete law, clock, preparation and event domain; observed and predicted rates remain distinct |

### 6.1 An oriented phase observation beyond this full tetrad

Take the unit-length, unit-conductance eight-cycle with fixed node labels
$0,\ldots,7$, unit capacity, and held pressure $p_i=i/8$. Compare

$$\theta_i^+=2\pi i/8,\qquad \theta_i^-=-2\pi i/8.$$

Each node has wrapped neighbor differences of magnitude $\pi/4$ in both
preparations. Its neighbor phasor sum is
$2\cos(\pi/4)e^{i\theta_i}$, so curvature is defined and zero. Pressure and
geometry are identical, hence potential and the coherence-length observation
are identical as well. These are exact-real full-tetrad equalities;
trigonometric implementation checks admit their declared rounding tolerance.

Along the fixed oriented cycle $0\to1\to\cdots\to7\to0$,

$$W=\frac1{2\pi}\sum_{i=0}^7
\operatorname{wrap}(\theta_{i+1}-\theta_i)=+1\ \hbox{or}\ -1.$$

The mean sine current at each node is zero because its two outgoing neighbor
contributions cancel. Thus this nodal current does not replace the oriented
cycle observation. Distinct fixed pressure labels prevent a reflection of
the whole state from identifying the preparations while preserving pressure;
the chosen ports/labels and cycle orientation are part of the comparison.
If orientation reversal is declared a gauge instead, the sign alone is not
an invariant distinction. Winding also requires treatment of the wrapping
cut and does not reconstruct an arbitrary phase field.

This witness supplies additional information relative to the full tetrad on
its declared state space. It does not show that winding is needed for every
model, that the two preparations have different dynamics under an unspecified
law, or that a finite list of added channels is universally sufficient.

### 6.2 Related phase and relaxation observations

On the same declared pair sets, circular phase alignment can be written

$$A_r=\langle\cos(\theta_j-\theta_i)\rangle_{P_r}.$$

It is invariant under global phase rotation and can distinguish consensus
from a twist at identical stored pressure. The mean circular contrast
$\langle1-\cos(\theta_j-\theta_i)\rangle$ is exactly $1-A_r$, so these
are not two independent channels. Both also lose handedness under phase
reversal. Consensus and twist already have different phase-gradient fields;
their separation establishes an increment over coherence alone, unlike the
full-tetrad witness in Section 6.1.

Temporal scales require their own model. Take at least two nodes with fixed,
reciprocal nonnegative effective conductances, positive weighted degrees and
connected positive-conductance support, and uniform positive capacity $\nu$.
For this domain, the explicitly
selected pure-EPI law $x'=-\nu L_{rw}x$ has modal solution
$x_k(t)=x_k(0)e^{-\nu\lambda_k t}$. Its slowest available nonconstant
modal time is $\tau=1/(\nu\lambda_2)$. When the implementation resolves
that gap, the separate spectral reference satisfies
$\tau=\xi_{\mathrm{spectral}}^2/\nu$. An initial state need not excite
that mode. This identity neither identifies the fitted product length with
that reference nor transfers to another pressure law, heterogeneous capacities
or support events. The [pressure owner](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md)
retains the pure-EPI model's premises.

### 6.3 Criterion for a predictive use

For a supplied complete law $z'=F(z)$ and an observation $y=O(z)$ on a smooth
domain, its rate is $DO(z)F(z)$. An autonomous observation law requires this
rate to agree on every admitted fiber with the same $O(z)$, together with
the necessary domain and solution hypotheses. A candidate extra channel is
useful for a particular task when it separates a previously indistinguishable
pair with different specified responses, or proves an appropriate closure
on that model. An extra numerical output or favorable fit score alone
establishes neither condition.

For actual temporal evidence, specify the complete law, pressure freshness,
capacity, support, clock, preparation, observation and horizon before the
reserved response. Compare against existing pressure, EPI, phase and capacity
observations as well as against $\xi_C$. If hidden initialization or memory
remains necessary, retain it explicitly. The
[theory-to-execution map](THEORY_TO_EXECUTION.md) and
[research program](RESEARCH_PROGRAM.md) own those admissions.

## 7. Shared implementation and verification boundary

[`observe_coherence_profile`](../src/tnfr/physics/canonical.py) exposes the
optional detached profile through the canonical field owner. Its shared
implementation in [_coherence_fit.py](../src/tnfr/physics/_coherence_fit.py)
retains the scalar estimator's pair and fit policies. It reports all sampled
positive-distance bins, endpoint conventions, global moments, exclusion
reasons and descriptive regression evidence. Missing fit quantities remain
unavailable. It does not replace an absent fit with the spectral scale, impose
a new goodness-of-fit cutoff or evolve any nodal state.

The optional report keeps node/source order, distance/sample descriptions and
numeric-mode provenance. Its log residuals, $R^2$, numerical design rank and
singular values describe the performed regression. They are not calibrated
confidence intervals, independent pair samples or proof of a physical model.
Finite exact identities in this chapter remain distinct from represented
arithmetic: a represented zero does not prove an exact-real zero.

The fixed controls are owned by
[the profile tests](../tests/physics/test_coherence_profile.py) and
[the observation-scope tests](../tests/physics/test_coherence_observation_scope.py),
alongside the existing
[distance-contract controls](../tests/physics/test_coherence_distance_contract.py).
Their intended checks distinguish exact rational/symbolic targets, floating
agreement, fit rejection and unavailable observations. The proofs above do
not depend on a numerical parameter search. Test execution and its current
outcome belong to the validation record; naming a test is not evidence that
it has run. These finite controls establish their own fixtures and contracts,
not universal empirical performance or a new constitutive selection.

## 8. Spatial observation at a supplied distance resolution

Exact-distance bins preserve a precise finite profile, but their repeated-bin
regression can become unavailable after an arbitrarily small metric change.
A separate observation can declare its distance resolution instead. This is
an additive readout of the same pair population, not a replacement definition
of the historical scalar fit or a dynamically selected TNFR length.

### 8.1 Triangular weights and retained endpoint populations

Keep the shared sampled positive-distance reachable pair set $P$. Supply a
query radius $r\geq0$ and a half-width $h>0$, both in the declared path-length
unit. For each pair define

$$w_{ij}(r,h)=\max\left(0,1-\frac{|d_{ij}-r|}{h}\right),\qquad
W=\sum_{P}w_{ij}.$$

If $W>0$, the observed product is

$$Q(r,h)=\frac{\sum_P w_{ij}c_ic_j}{W}.$$

The numerator and mass remain visible. So do the positive-weight pair count
and normalized mass $W/|P|$, when $P$ is nonempty. These quantities describe
the supplied population and weighting; none is an independent sample count.
If $W=0$, the normalized moments are unavailable, not zero. An empty pair
population also has no normalized support fraction.

The numerical reducer normalizes weights before forming moments, using the
shared stable weighted-mean owner. A tiny positive raw numerator can be
unrepresentable even when its ratio to the mass is representable. In that
case the numerator is separately unavailable with `positive_underflow`
status, while the normalized product remains available. Empty support keeps
a zero numerator with `empty_support` status. This protects the final mean
from an unnecessary intermediate loss; it does not reconstruct information
already rounded in the underlying coherence profile or certify exact-real
zeros from represented zeros.

Apply the same weights to endpoint means, endpoint second moments, globally
centered products and semivariograms. The global mean $\mu$ still refers to
every graph node. The exact identities remain

$$K=Q-2\mu M+\mu^2,\qquad \gamma=S-Q.$$

Directed source and target roles remain distinct. On undirected support,
symmetrizing each pair gives both roles mean $M$. Writing $a,b$ for the
weighted source/target means, the weighted endpoint covariance is
$H=Q-ab$. This is a finite deterministic covariance under the declared
weights; its interpretation does not require or establish stationarity.

The existing exact-bin report is sufficient for this reduction because every
pair in a bin receives the same distance weight. A bin with count $n_b$,
product $q_b$ and weight $w_b$ contributes $n_bw_bq_b$ to the numerator and
$n_bw_b$ to the mass. All sampled bins participate, including singleton bins
and bins excluded by the historical fit's product floor. This differs from
equal-bin regression weighting.

Bin covariances cannot simply be averaged: with normalized bin masses
$\alpha_b=n_bw_b/W$, role means $a_b,b_b$ and bin covariance $H_b$,

$$H=\sum_b\alpha_b\left[H_b+(a_b-a)(b_b-b)\right].$$

The second term retains changes in endpoint composition between bins. The
shared implementation uses this identity and preserves the original profile,
including source order, omitted pairs, geometry and numeric provenance.

### 8.2 Equivariance, perturbation and availability

Uniformly rescaling $d,r,h$ by one positive factor preserves every exact-real
weight and normalized moment. The supplied resolution must transform with
the metric unit. Holding the numerical bandwidth fixed while changing the
distance unit is a different observation policy.

For a perturbation theorem, fix pair identities, reachability and the
positive-distance admission domain. Suppose the two metrics satisfy
$|\widetilde d_{ij}-d_{ij}|\leq\varepsilon$ on this population. With the same
queries and bandwidth, the triangular kernel gives

$$|\widetilde w_{ij}-w_{ij}|\leq\varepsilon/h.$$

Let $m=|P|$, $\Delta=m\varepsilon/h$, and let
$R=\max_P(c_ic_j)-\min_P(c_ic_j)$. For unchanged coherence values,
$|\widetilde W-W|\leq\Delta$. If $W>\Delta$, the perturbed support is
nonempty and

$$\boxed{|\widetilde Q-Q|\leq
\min\left(R,\frac{R\Delta}{W-\Delta}\right).}$$

To prove the bound, put $z_{ij}=c_ic_j$ and subtract the two normalized sums:

$$\widetilde Q-Q=
\frac{\sum_P(\widetilde w_{ij}-w_{ij})(z_{ij}-Q)}{\widetilde W}.$$

Both $Q$ and each $z_{ij}$ lie in the same interval of width $R$; the
denominator is at least $W-\Delta$. The additional bound by $R$ follows
because both weighted means lie in that interval.

A sharper sufficient denominator retains each pair's possible weight range:

$$w^-_{ij}=\max\left(0,1-\frac{|d_{ij}-r|+\varepsilon}{h}\right),$$

$$w^+_{ij}=\max\left(0,1-
\frac{\max(0,|d_{ij}-r|-\varepsilon)}{h}\right).$$

Then $w^-_{ij}\leq\widetilde w_{ij}\leq w^+_{ij}$. Set
$W^-=\sum_Pw^-_{ij}$ and
$D=\sum_P\max(w_{ij}-w^-_{ij},w^+_{ij}-w_{ij})$. If $W^->0$, the same
proof gives $|\widetilde Q-Q|\leq\min(R,RD/W^-)$. Failure of this sufficient
condition means the bound does not guarantee support; it need not mean that
the actual perturbed observation is absent. Nominal availability and a proof
of robustness are different claims.

If products also change by at most $\eta_z$, their contribution adds at most
$\eta_z$ to either bound, with a final bound of one on this coherence domain.
The primitive map satisfies

$$|c(p)-c(\widetilde p)|
=\frac{\big||p|-|\widetilde p|\big|}
{(1+|p|)(1+|\widetilde p|)}
\leq|p-\widetilde p|.$$

Thus a uniform nodal coherence error $\eta_c$ implies the simple product
allowance $\eta_z\leq\min(1,2\eta_c)$. These are deterministic inequalities,
not noise estimates or calibrated confidence intervals.

The fixed-pair hypothesis matters. The current reader omits zero distances;
a zero/positive crossing can alter its pair population. Positive distances
bounded below by more than $\varepsilon$ are a sufficient margin when the
population already contains every sampled reachable pair. Any omitted zero
distances must separately remain omitted. Support or source-sample changes
need their own comparison. On a fixed finite graph with matched nonnegative
edge lengths changed by at most $\delta$, shortest-path distances change by
at most $(N-1)\delta$: each direction of the comparison uses a simple
minimizing path with at most $N-1$ edges. This allows the minimizing path to
switch and preserves reachability, but does not by itself preserve exclusion
of zero-distance pairs.

All these bounds concern exact real quantities under their stated metric
model. Floating path accumulation, kernel evaluation and moment reduction
need their own rounding allowance for an outward-certified conclusion.
The public resolution report is descriptive binary64 output and exposes no
numerically certified robustness flag.

### 8.3 Fixed controls distinguish product continuity from endpoint identity

Use the radial star of Section 3, keep its pressures fixed, and perturb only
its lengths by $\eta(1,2,4,8,16,32)$, where $\eta=2^{-20}$. Query the six
integer radii with half-width $h=1/4$. Every original distance bin remains
inside its own kernel support, while all other original bins stay outside.
Within each original bin, every unchanged product equals $2^{-r-2}$.
Consequently both preparations have exactly

$$Q(r,1/4)=2^{-r-2},\qquad r=1,\ldots,6,$$

with positive mass, although the perturbed exact-distance profile consists
of twenty-one singleton bins and has no admissible historical fit. The last
query also retains its single-pair observation; deterministic observation
availability does not require regression's two-pair count rule.

Endpoint moments need not be preserved. At $r=2$, the three original pairs
have perturbed distance increments $4\eta,8\eta,3\eta$ and hence weights
$1-16\eta,1-32\eta,1-12\eta$. Their endpoint means are $5/16,5/16,1/4$.
Therefore

$$\widetilde M(2,1/4)=\frac{7/8-18\eta}{3-60\eta},\qquad
\widetilde M-\frac7{24}=-\frac{\eta}{2(3-60\eta)}.$$

The unchanged product readout does not authorize dropping this population
change. With the original geometry, increasing the width to $h=3/2$ at
$r=1$ instead combines masses two and one from distances one and two:

$$Q(1,3/2)=\frac{2(1/8)+(1)(1/16)}{3}=\frac5{48}.$$

This explicitly demonstrates resolution dependence. A query at $r=20$ with
$h=1/4$ has no support. Multiplying geometry, queries and bandwidth by three
preserves the exact weights and moments. Independent rational pair sums in
[the resolution tests](../tests/physics/test_coherence_resolution.py) own these
fixed implementation controls; no trajectory or width search is involved.

### 8.4 Scope relative to other estimators

Hard distance windows can move an entire nonzero-weight pair in or out at a
boundary after an arbitrarily small perturbation; they do not have the
triangular kernel's global Lipschitz property. Pairwise log regression avoids
the repeated-bin requirement, but changes weighting and fits log products
rather than the log of bin-mean products. It retains its own signal-domain and
conditioning obligations. Neither alternative is excluded; each defines a
different observation requiring its own admitted comparison.

[`observe_coherence_resolution`](../src/tnfr/physics/coherence_resolution.py)
reads a live graph through the canonical profile owner and accepts only
explicit radii and bandwidth. It adds no quality cut, automatic width,
correlation-length fit, canonical state coordinate or constitutive law.
The scalar estimator and its spectral fallback keep their existing semantics.
