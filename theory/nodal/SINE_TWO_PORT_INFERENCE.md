# Two-port geometry from a calibrated local response

<a id="sine-two-port-inference"></a>

## Question and scope

The [interior phase-dipole result](SINE_TWO_PORT_DIPOLE.md#sine-two-port-dipole-result)
is a forward discrimination certificate for two acquired geometries. It does
not supply an independent observation from which to infer either geometry.
Moreover, the fixed critical `(2,1,0)` problem already has a unique target;
assuming that problem and returning its root would add no inverse information.

This result instead admits a continuous family of complete, generally
nonstationary states on the joined support. Under a known law, input, clock
and positive observation-gain interval, one supplied finite local increment
gives a necessary outer interval for a donor angle. The receiver geometry
remains a nuisance coordinate. The proof retains its nonzero port currents,
all form and phase errors, and the finite response of the full law.
It is a conditional theorem and computational enclosure, not a new frozen
response assessment, independently calibrated experiment or physical inference.

<a id="sine-two-port-inference-domain"></a>
## A realizable transient state family

Use donor nodes `0,...,8`, receiver nodes `9,...,17`, their unit C9 cycles,
and unit contacts `(0,9)` and `(1,10)`. Ports `0,1,9,10` have degree three;
the remaining nodes have degree two. Let \(L\) be the full Laplacian,
\(M\) its degree matrix, \(A=M^{-1}L\), and
\(\|z\|_M=(z^TMz)^{1/2}\). The total degree mass is forty.
Let \(P_M\) remove the single common degree-weighted mean.

Admit any closed parameter intervals within the compact radian rectangle
\[
\frac{11}{8}\le b\le\frac32,\qquad \frac23\le c\le1.
\tag{1}
\]
Define the donor short angle \(s\), receiver bulk angle \(d\), and
contact offset \(\zeta\) by
\[
s=4\pi-8b,\qquad d=\frac{2\pi-c}{8},\qquad
\zeta=\frac{s-c}{2}.
\tag{2}
\]
Before subtracting their one full degree-weighted mean, take the lifts
\[
\begin{aligned}
\theta^0_{D0}&=0,&
\theta^0_{Dj}&=s+(j-1)b &&(1\le j\le8),\\
\theta^0_{R0}&=\zeta,&
\theta^0_{Rj}&=\zeta+c+(j-1)d &&(1\le j\le8).
\end{aligned}
\tag{3}
\]
Thus the donor short arc is \(s\), its eight long arcs are \(b\),
the receiver short arc is \(c\), and its eight long arcs are \(d\).
The increasing cycle orientations use closing-edge lift adjustments
\(4\pi\) and \(2\pi\), respectively. The contact gaps are
\(\zeta,-\zeta\), and the mixed cycle `0->9->10->1->0` has period zero.
All these nominal edges are strictly acute throughout (1). For example,
\(s\le4\pi-11<\pi/2\) follows from \(\pi<22/7\), while
\(s\ge4\pi-12>0\); the other bounds follow directly from (1)-(2).
No critical balance equation is imposed.

At the pre-probe time admit
\[
x^-=m_x\mathbf1+u,\qquad
\theta^-=m_\theta\mathbf1+\theta^0(b,c)+v,
\qquad
P_Mu=u,\quad P_Mv=v,\quad
\|u\|_M\le X,\quad\|v\|_M\le Y.
\tag{4}
\]
The real common form and lifted phase means are arbitrary. Their removal
is a proof coordinate choice, not a reset. Every node can carry a residual;
no reflection or componentwise mean constraint is imposed. This family
is supplied independently of the response. The previous capture and warmup
certificates do not admit every point of this different transient family.

## Complete law, event and observation

Retain unit held capacities, beta one, \(e=1023/1024\), \(w=1/1024\),
and the fast structural clock \(\tau=et\). Define
\[
\gamma=\frac1{1023\pi},\qquad
f_i(\theta)=\frac1{M_{ii}}
 \sum_{j\sim i}\sin(\theta_j-\theta_i).
\]
The full rows after the supplied event are
\[
x'=-Ax+\gamma f(\theta),\qquad \theta'=\gamma Ax,
\tag{5}
\]
where primes mean \(\tau\) derivatives. The graph, capacities and law
remain fixed; no other forcing or event acts during the window.
These are the same complete rows as the
[two-port compatibility model](SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-compatibility),
without its equilibrium premise. Both common means are conserved.

At \(\tau=0\), supply
\[
q=e_4-e_5,\qquad \theta^+=\theta^-+a q,
\qquad x^+=x^-,\qquad 0<a\le1,
\tag{6}
\]
and read at \(0<h\le1\). The event is an external instantaneous phase
input, not a finite pressure pulse or autonomously selected event.
The primitive model increment is
\[
y=q^T[x(h)-x^-]. \tag{7}
\]
Here \(\|q\|_M=2\), \(\|q\|_{M^{-1}}=1\), and the phase
event preserves the weighted mean. Common form cancels in (7) and does
not affect (5).

Each of the two scalar readings measures \(Gq^Tx\) plus a common
constant offset and its own additive error of magnitude at most
\(\delta\). One fixed gain \(G\in[G_-,G_+]\), with \(G_->0\),
applies to both readings. The offset cancels; the two errors need not.
If the recorded increment belongs to the supplied interval
\(\mathcal M=[m_-,m_+]\), then
\[
y\in\mathcal U=
 \frac{\mathcal M+[-2\delta,2\delta]}{[G_-,G_+]}.
\tag{8}
\]
Interval division retains signed form. An unbounded offset drift or
different unknown gains between readings is outside this model. The law,
clock, input and calibration intervals are independent premises, not
quantities fitted to this same increment.

## Port locality and a finite full-law error bound

The degree operator is self-adjoint and nonnegative, with
\(\|A\|_M\le2\). Thus \(e^{-\tau A}\) is a contraction.
The complete sine map is globally Lipschitz with constant two in this
norm: its derivative is a signed-cosine weighted Laplacian whose absolute
quadratic form is bounded by that of \(A\).

Write \(f_0=f(\theta^0(b,c))\). The two equal and opposite bulk
sine currents cancel at every nonport node. Consequently \(f_0\)
is supported only on `0,1,9,10`, and
\[
\|f_0\|_M\le\sqrt{12}<F,\qquad F=4.
\tag{9}
\]
Both readout nodes are at graph distance at least three from those ports.
The diagonal entries of \(A\) do not shorten that distance, so
\[
q^TA^k f_0=0\qquad(k=0,1,2). \tag{10}
\]
This removes only three Taylor moments, not all subsequent port influence.
Using the contraction integral remainder,
\[
\left|q^Te^{-rA}f_0\right|
 \le \frac{r^3}{6}\|A^3f_0\|_M
 \le\frac43 F r^3.
\]
Therefore the nominal noncritical background contribution to (7) is
at most \(\gamma Fh^4/3\). It cannot be dropped by treating the
family as a set of equilibria or by equating local adjacency with the
complete finite response.

Use the rational upper bound \(g=1/3069>\gamma\), and define
\[
Q=\frac{X+gh(F+4a+2Y)}{1-4g^2h^2}.
\tag{11}
\]
The denominator is positive for \(h\le1\). Duhamel's formula and the
global Lipschitz bound give, throughout \(0\le r\le h\),
\[
\|P_Mx(r)\|_M\le Q,\qquad
\|\theta(r)-\theta^+\|_M\le2ghQ.
\tag{12}
\]
Indeed, if \(Q_h\) is the actual supremum form norm, the initial
sine norm is at most \(F+4a+2Y\), and phase motion contributes at
most \(4g^2h^2Q_h\) to its form integral. Solving that scalar
inequality proves (11)-(12). No acute assumption is needed for these
global bounds.

At the nominal phase immediately after (6), the three affected interior
edges have gaps \(b+a,b-2a,b+a\). Their local sine rows give exactly
\[
q^Tf(\theta^0+a q)
 =\sin(b-2a)-\sin(b+a)
 =-2\sin(3a/2)\cos(b-a/2).
\tag{13}
\]
The unknown receiver coordinate does not enter this leading term; its
full finite influence is still covered by (9)-(12).

Let
\[
K=2\gamma h\sin(3a/2)>0,
\qquad
E=2hX+\frac{gFh^4}{3}+4gah^2+2ghY+4g^2h^2Q.
\tag{14}
\]
Then every admitted full trajectory obeys
\[
\boxed{\left|y+K\cos(b-a/2)\right|\le E.}
\tag{15}
\]
The five terms of \(E\) have separate sources. They bound free form
relaxation, the port background, transport of the initial pulse current,
the initial phase residual, and phase motion during the response.
For the pulse term, its sine-map increment has norm at most \(4a\);
\(\|(e^{-rA}-I)z\|_M\le2r\|z\|_M\) and integration give
\(4gah^2\). The other terms follow from (9)-(12) and the same
Duhamel formula. This calculation retains both rows of (5).

## Branch retention and the conditional inverse

Let \(\mu_0\) be a lower bound for the nominal minimum acute edge
margin over the declared parameter rectangle, including both contacts.
Each edge difference has degree-dual norm at most one. The source guard
\(\mu_0-Y>0\) admits the pre-probe acute branches and their geometric
interpretation. A separate sufficient whole-window acute guard is
\[
\mu_0-Y-2a-2ghQ>0. \tag{16}
\]
When it passes, the supplied event and subsequent window preserve the
declared cycle branches and periods. It does not prove all-time recovery,
acquisition or an allowance on the event's supplied work. The bound (15)
and the inverse below remain valid when this additional guard is unresolved:
their full-flow estimate is global. Whole-window acuteness is optional
evidence, not a necessary inverse premise. The pre-probe source guard
suffices for the actual geometric statistic defined below.

The shifted donor branch \(b-a/2\) lies strictly within \((0,\pi)\)
under (1) and (6). Hence cosine is strictly decreasing, and its derivative
has no zero on any admitted closed branch. Let \([b_-,b_+]\) be the
declared donor interval, and form
\[
\mathcal C=
\frac{-\mathcal U+[-E,E]}{K}
\;\cap\;
[\cos(b_+-a/2),\cos(b_--a/2)]
\;\cap\;[-1,1].
\tag{17}
\]
If this intersection is empty, no state satisfying all the family, law,
input, clock, gain and error premises produces the supplied interval.
Touching endpoints must not be discarded as a strict exclusion.
If \(\mathcal C=[C_-,C_+]\) is nonempty, every compatible nominal
donor coordinate lies in
\[
\mathcal B=[\arccos C_++a/2,\arccos C_-+a/2]
 \cap[b_-,b_+]. \tag{18}
\]
Outward interval arithmetic and monotone bisection can enlarge (17)-(18);
unresolved numerical separation must retain the candidate or abstain.
No equality with an interval midpoint or floating residual is a proof of
exclusion. More precision refines an enclosure; it supplies no additional
observation.

With arbitrary residuals, \(b\) is a nominal family coordinate and
need not label the fine state uniquely. A unique actual geometric scalar
at the pre-probe time is the mean of its eight oriented donor long arcs:
\[
B_{\mathrm{actual}}
 =\frac{4\pi-(\theta^-_1-\theta^-_0)}8
 =b-\frac{v_1-v_0}{8}.
\tag{19}
\]
The two port degrees are three, so
\(|v_1-v_0|\le\sqrt{2/3}\,Y\le Y\). Thus
\[
B_{\mathrm{actual}}\in\mathcal B+[-Y/8,Y/8]. \tag{20}
\]
Do not clip (20) back to the nominal prior: the actual fine-state mean
can lie outside that prior while still satisfying (4). This inference is
about the pre-probe geometry. The immediate interior phase jump leaves
the two port phases and this statistic unchanged, but its value can change
during the subsequent nonstationary response. Neither \(c\) nor the full
residual vectors are reconstructed.

<a id="sine-two-port-inference-boundary"></a>
## Meaning of the enclosure and remaining obligations

The [implementation](../../src/tnfr/physics/relational_sine_two_port_inference.py)
reuses the shared affine geometry and outward trigonometric arithmetic;
it performs no equilibrium solve, trajectory integration or old producer
replay. Its [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-two-port-inference)
owns scalar admission, numerical budgets and availability. The
[tests](../../tests/physics/test_sine_two_port_inference.py) exercise domains,
the local algebra, bounds and inverse behavior. Synthetic response intervals
used by those tests are software controls, not independent observations.

A bounded candidate is a necessary outer constraint. It does not prove
that any retained value, or even any complete state, realizes the supplied
reading. Conversely, exclusion rejects the combined premises without
identifying which premise failed. Overlap of outer response intervals is
not a constructed observational collision. This differs from the
[finite orientation/law collision](SINE_PAIR_INTERACTION.md#sine-pair-receiver-constitutive-confounding),
which proves an actual equal finite reading under its own complete laws.
The [hidden-star inverse](SINE_ENVIRONMENTAL_MEMORY.md#sine-hidden-state-observability)
uses known incidence and multiple instantaneous rates; its rank conditions
and coefficients do not transfer to this single finite increment.

The leading coefficient alone combines gain, clock scale and angle.
If these calibration premises are removed, this theorem gives no joint
identification. Equal leading coefficients would not by themselves prove
equal finite full-law responses. A law-selection or physical interpretation
would need independent calibration, admissible alternatives and a justified
measurement bridge, following the
[research strategy](../NODAL_RESEARCH_STRATEGY.md#constitutive-information-justification).

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns whether a later reserved inference experiment is admitted. Such an
experiment must fix its unknown-state family, independent calibration,
recorded observation, horizon and numerical budget before assessing its
reserved response. This conditional inverse neither starts that campaign
nor modifies any earlier frozen prediction, source archive or response.

<a id="sine-two-port-inference-protocol"></a>
## Prospective reserved software inference protocol

The theorem above remains unchanged. This separate experiment evaluates
three fixed complete-law responses and passes only their declared public
observations to the inverse. The full source recipe, calibration, numerical
budget, stopping criteria and producing source must be archived before any
reserved response is generated. This is a software information-exclusion
test with a separately specified forward calculation. It supplies neither
external data nor physical calibration or cryptographic blindness.

### Hidden sources and public preparation information

Use (1)-(7), the same eighteen-node support and full fast-clock law, and
the following public inputs for every case:

| Input | Fixed value |
| --- | --- |
| Nominal donor prior | \([11/8,3/2]\) radian |
| Receiver nuisance prior | \([2/3,1]\) radian |
| Full degree-norm allowances | \(X=Y=2^{-40}\) |
| Supplied phase increment | \(a=2^{-10}\) radian |
| Fast elapsed time | \(h=2^{-16}\) |
| Error of each scalar response reading | \(\delta=2^{-60}\) in recorded units |
| Inverse refinement budget | Forty-eight bisections per boundary |
| Required actual-angle width | Strictly less than \(1/1024\) radian |

For case index \(k=0,1,2\), fix the hidden parameter pairs respectively as
\[
(b_k,c_k)=
 (45/32,3/4),\quad(23/16,5/6),\quad(47/32,11/12).
\tag{21}
\]
These are three concrete transient sources, not critical targets or a
claim that three evaluations cover every point of the theorem's family.
For node index \(i=0,\ldots,17\), define raw residuals
\[
r_i^{(k)}=\frac{(i+1)(k+1)}{2^{52}},\qquad
s_i^{(k)}=\frac{((7i+5k)\bmod19)+1}{2^{52}}.
\tag{22}
\]
Apply the full degree-mean projection to each vector, giving
\(u^{(k)}=P_Mr^{(k)}\) and \(v^{(k)}=P_Ms^{(k)}\). The actual
source is
\[
x^-_k=\frac{k+1}{7}\mathbf1+u^{(k)},\qquad
\theta^-_k=-\frac{k+1}{5}\mathbf1+
 \theta^0(b_k,c_k)+v^{(k)}.
\tag{23}
\]
All eighteen residual coordinates of each channel are nonzero. Before
division by \(2^{52}\), the form mean is \((k+1)183/20\), and
the three phase means are \(48/5,197/20,423/40\); none equals a
corresponding raw integer. Orthogonal mean projection cannot increase
the degree norm. The common conservative bound
\[
\|u^{(k)}\|_M^2,\ \|v^{(k)}\|_M^2
 \le\frac{40\cdot57^2}{2^{104}}<2^{-80}
\tag{24}
\]
admits both public allowances without discarding any node. Common means
remain in the evolved state. Neither an equilibrium reset nor an old
capture or warmup report supplies these sources.

### Independent software calibration and recorded readings

One hidden held gain \(G=3/2\) and offset \(O=5/7\) apply to all
calibration and response readings. Use two known scalar references
\(z_-=-1,z_+=1\), with calibration error bound
\(\delta_c=2^{-14}\), and the frozen realizations
\[
e_-^{\mathrm{cal}}=\delta_c/2,\qquad
e_+^{\mathrm{cal}}=-\delta_c/4,\qquad
R_\pm=Gz_\pm+O+e_\pm^{\mathrm{cal}}.
\]
The public calibration derives only
\[
\mathcal G=
\left[\frac{R_+-R_-}{2}-\delta_c,
      \frac{R_+-R_-}{2}+\delta_c\right]. \tag{25}
\]
Its width is \(2^{-13}\), its lower endpoint exceeds one, and it
contains the true held gain. The constant offset cancels. Neither the
hidden gain nor offset is passed to the inverse, and none of the three
reserved responses is used to determine (25). Exact reference values,
error allowances and a held affine sensor are supplied software premises.

For case \(k\), set
\[
e_k^{\mathrm{before}}=(k-1)\delta/2,\qquad
e_k^{\mathrm{after}}=(1-2k)\delta/4.
\tag{26}
\]
Both magnitudes are at most \(\delta\). The recorded scalar readings
are \(Gq^Tx^-_k+O+e_k^{\mathrm{before}}\) and
\(Gq^Tx_k(h)+O+e_k^{\mathrm{after}}\). Retain both and their
increment enclosure. The produced increment interval must have numerical
width at most \(2^{-60}\), separately from the per-reading sensor
error allowance of the same size. The inverse retains both reading errors
again as unknown bounded errors; it receives no realized error correction.

### A separate full-state response certificate

The [response producer](../../src/tnfr/physics/relational_sine_two_port_readout.py)
evolves all thirty-six coordinates after the phase event using the shared
complete sine rows. It consumes primitive source boxes, not the inverse
formula, its leading coefficient or its response-error bound. Fix one
step of duration \(h\), Taylor order four, at most sixteen Picard tube
iterations, and the shared outward dyadic128 rational/trigonometric backend.
There is no folded symmetry, equilibrium search, adaptive step, changed
precision or response-dependent retry.

The [shared direct source-box kernel](../../src/tnfr/mathematics/_validated_taylor.py)
retains a strict Picard tube \(Z\) for the entire step. Let \(J_j(B)\)
be its interval extension of the normalized order-\(j\) solution derivative
on a state box \(B\), evaluated by the shared formal jet recurrence.
Every solution starting in the post-event source box \(B_0\) satisfies
\[
z(h)\in B_0+\sum_{j=1}^4 h^jJ_j(B_0)+h^5J_5(Z).
\tag{27}
\]
The order-five derivative enclosure is evaluated on the entire certified
tube. All lower coefficients are evaluated on the complete initial box,
so no center substitution discards initial uncertainty. The complete sine
field is globally smooth; strict Picard inclusion and its positive margin
provide the whole-time enclosure independently of an acute-chart claim.

The increment in (27) omits the common order-zero coordinate before
interval arithmetic. Applying the raw form readout \(q^T\) to that
increment retains its correlation with the same initial state, instead
of subtracting unrelated initial and endpoint boxes. All thirty-six
endpoint coordinates, source boxes, tube, coefficients, derivative
remainders, domain margins and achieved horizon remain in the report.
A failed tube or remainder calculation is unavailable evidence, not an
endpoint to substitute with a midpoint or a lower-order approximation.

### Public inverse inputs and prospective controls

Construct each inverse request from an explicit allowlist: the two public
angle priors, \(X,Y,a,h\), the produced recorded increment interval,
\(\delta\), gain bounds and forty-eight refinements. No hidden angle,
state coordinate, common mean, exact gain, offset, realized error, endpoint
coordinate or producer diagnostic enters that request. The posterior truth
audit is separate from the inverse call. This boundary must be checked in
the actual evaluator, not inferred from a report label.

For each case make three requests with the same recorded increment:

1. **Calibrated inference:** use \(\mathcal G\) from (25) and both
   original angle priors. Require a bounded candidate containing the hidden
   nominal \(b_k\) and the actual pre-probe long-arc mean (19), with
   actual-angle width strictly below \(1/1024\).
2. **Reduced calibration information:** replace only the gain bounds by
   \([1,2]\). Require a bounded candidate containing the calibrated
   actual-angle interval and having width strictly above \(1/64\).
   This tests the effect of discarding gain information on a necessary
   outer bound. It does not construct two exact trajectories with equal
   readings or prove that every retained angle is realizable.
3. **False donor prior:** use (25) and replace only the nominal donor prior
   by \([11/8,89/64]\). Require an incompatible result. This tests
   exclusion of the combined false-prior premises; it does not infer which
   premise would have failed for an unknown external source.

The calibration and noise budgets provide a prospective feasibility check
without evaluating any reserved response. For these public inputs,
\(K\ge K_-=ah/1206\), the shifted branch has sine greater than
\(9/10\), and exact rational comparisons give
\(E/K_-<2^{-15}\), \(Q<2^{-25}\). When the produced recorded
interval has width at most \(2^{-60}\), its calibrated width divided
by \(K_-\) is less than \(2^{-21}+2^{-12}\). Equations (17)-(20)
then bound the ideal primary actual-angle width, including the slack of
forty-eight resolved bisections at each boundary, by
\[
\frac{10}{9}(2^{-21}+2^{-12}+2^{-14})
 +\frac{Y}{4}+\frac{2(b_+-b_-)}{2^{48}}
 <2^{-11}<\frac1{1024}, \tag{28}
\]
The required strict signs and outward arithmetic availability must still
be certified; an unresolved bisection cannot claim this numerical slack.
This analytical budget check does not replace any of the reserved stopping
criteria or manufacture the observation supplied to the inverse.

Also retain the complete phase-blind alternative
\[
x'=-Ax,\qquad\theta'=\gamma Ax
\tag{29}
\]
on the same support, source, clock and phase event. Its form flow is
independent of phase. Contraction and \(\|A\|_M\le2\) give
\[
|m_{\mathrm{heat}}|\le2G_+hX+2\delta,
\qquad G_+=\sup\mathcal G.
\tag{30}
\]
Require each retained recorded increment interval to be disjoint from this
closed symmetric interval. This includes its raw form background, held
gain and both reading errors, with no ideal reset or extra numerical
trajectory. Excluding this specified alternative does not select the sine
law among every possible phase-dependent law.

### Stopping rule and retained evidence

Predict that all three fixed cases pass source and calibration admission,
the complete-horizon numerical certificate and numerical-width limit,
public-only input separation, both primary truth-containment checks,
the strict primary width threshold, the reduced-calibration containment
and width control, the false-prior exclusion, and the phase-blind exclusion.
Record every criterion separately; success is their conjunction across
all cases. Preserve the first verdict even if an arithmetic, numerical,
export or discrimination condition fails. A correction needs separately
retained evidence and cannot rewrite the evaluated prediction.

The frozen protocol owns the exact source arrays, hidden and public records,
calibration readings, fixed laws and units, event, backend and budgets.
Archive it with the complete producing source and prospective proof before
the first reserved calculation. Retain the resulting full response reports,
three inverse requests and outputs per case, posterior audits, stopping
verdicts and a manifest of source/configuration hashes. Hashes provide
integrity checks, not provenance authentication. Keep all earlier frozen
experiments and their original producers unchanged. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns the active status and later result closure.

<a id="sine-two-port-inference-result"></a>
## Retained result of the reserved software inference experiment

The first frozen evaluation returned `certified_reserved_inference` with
all forty-three stopping conditions passing. No response was retried and
no source, calibration, horizon, numerical budget or threshold was changed.
The [saved record](../../docs/assets/sine_formed_classes/two-port-inference-v1.json)
retains the three complete response certificates, public packets, inverse
outputs and posterior audits. Its
[protocol](../../docs/assets/sine_formed_classes/two-port-inference-v1.protocol.json),
[source archive](../../docs/assets/sine_formed_classes/two-port-inference-v1.source.zip)
and [manifest](../../docs/assets/sine_formed_classes/two-port-inference-v1.manifest.json)
retain the preparation and producing implementation. The archived prospective
proof has 23,649 bytes, with SHA-256
`488c51e9adc0ecbbf2007cef4c113073e08c61222607cc7a72c8c7a5c630c50f`.
Its content is preserved above; routine Git checkout conversions may change
the working copy's line-ending bytes without changing that archived artifact.

For every fixed source, (22) gives \(v_1-v_0=7/2^{52}\). Its exact
pre-probe long-arc mean is therefore
\[
B_k=b_k-\frac7{2^{55}}.
\tag{31}
\]
The primary nominal interval contains \(b_k\), and the primary actual-angle
interval contains \(B_k\), in all three cases. The displayed interval
endpoints below are rounded outward; widths are approximate displays of
the saved exact rational differences. They do not replace the exact tests.

| Case and hidden nominal \(b_k\) | Primary actual-angle interval, radian | Primary width | Width with gain bounds \([1,2]\) |
| --- | --- | --- | --- |
| 1: \(45/32\) | \([1.4062191227,1.4062800901]\) | \(6.0967316150\times10^{-5}\) | \(0.07278097884\) |
| 2: \(23/16\) | \([1.4374705738,1.4375287176]\) | \(5.8143756231\times10^{-5}\) | \(0.09610233068\) |
| 3: \(47/32\) | \([1.4687219897,1.4687773795]\) | \(5.5389766204\times10^{-5}\) | \(0.07734653861\) |

Each primary actual-angle width is strictly below \(1/1024\). Each
broad-gain actual interval contains the corresponding primary interval and
has width strictly above \(1/64\). All three fixed false-prior requests
return `incompatible`. Each retained recorded increment also lies outside
the complete phase-blind alternative's closed raw-response interval.
These controls used the same recorded response as their respective primary
inference; they did not trigger another full-flow evaluation.

The calibration produced the positive interval
\([196597/131072,196613/131072]\) from its two separate reference
readings. Each full thirty-six-coordinate certificate reached the fixed
endpoint in its one fourth-order step. All source and reading errors were
admitted, and each recorded increment's numerical width met its separate
\(2^{-60}\) limit. Fresh worker processes consumed only the serialized
public inference packets; hidden source and sensor realizations remained
in the separate source generation and posterior audit. The retained packet
hashes and frozen worker code make that information boundary reviewable.

This is finite software evidence for conditional partial inference on the
three stated transient sources. The broad-gain widths quantify this outer
inverse's loss of resolution; they are not exact observational collisions.
Excluding the supplied false prior and phase-blind law does not establish
unique geometry under arbitrary models or a uniquely selected constitutive
law. Every interval still has the theorem's necessary-constraint meaning;
the receiver state and complete nodal state are not reconstructed. Source
preparation, complete law, structural clock and held affine observation
model remain supplied. No physical calibration, external measurement bridge,
source acquisition or post-probe maintenance result follows from this record.
Content hashes support consistency and source recovery, not independent
timing, execution authentication or cryptographic blindness.
