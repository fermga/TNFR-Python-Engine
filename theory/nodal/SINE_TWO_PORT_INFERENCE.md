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
