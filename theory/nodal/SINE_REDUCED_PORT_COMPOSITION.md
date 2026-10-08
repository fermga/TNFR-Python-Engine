# Degree-aware composition of reduced sine-class ports

<a id="sine-reduced-port-composition"></a>

## Prospective composition claim and frozen protocol

The [two-component reduction](SINE_REDUCED_CLASS_PORTS.md#sine-reduced-class-ports)
admits one central contact per component. This gate asks whether the
derived component description composes on a finite simple undirected
**unit** contact graph, with the actual contact degrees, an exact joined
storage balance and a uniform finite-window approximation of the complete
fine law. It does not ask for another donor-class response comparison.

Freeze this single control before evaluating its certificate:

- Use three simple unit C9 components, in component order \((0,1,2)\)
  and local node order \(0,\ldots,8\), with winding classes
  \((1,2,1)\). The supplied contact edges are \((0,1)\) and \((1,2)\),
  each joining the local central nodes \(4\). Their contact degrees
  are \((1,2,1)\); the joined fine port degrees are \((3,4,3)\).
- Retain unit capacities, beta one, \(e=1023/1024\), \(w=1/1024\),
  the structural clock \(\tau=et\), and \(\gamma=1/(1023\pi)\).
  The complete rows on each live support are
  \[
  x'=-KLx+\gamma KS(\theta),\qquad \theta'=\gamma KLx,
  \]
  where \(K\) is inverse fine degree, \(L\) is the unit Laplacian,
  and \(S_a(\theta)=\sum_{b\sim a}\sin(\theta_b-\theta_a)\).
  No edge weights, loss changes, inputs or occurrence law are inferred.
- Retain the original class-specific form sources
  \(k_i m(j-4)\), \(m=(2046/9)(355/113)^2\). Nominal initial
  phases are constant on each component, with supplied origins
  \((o_0,o_1,o_2)=(0,1/1000,0)\). Each component has independent
  per-coordinate initial form and continuous phase-lift errors at most
  \(10^{-10}\), subject to zero sum separately in each channel.
  Retain all sixteen relative source coordinates per component and
  its original storage ceiling \(10^9\); source costs need not agree.
- Freshly admit [formation of both classes](SINE_PATTERN_DYNAMICS.md#sine-formed-class-pair)
  at \(\tau=100\) and relative radius \(r=1/12\). Continue the
  actual unprobed trajectories for \(D=10^{13}\). Require Euclidean
  form and target-phase error bounds \(\epsilon=10^{-32}\),
  separately on every component, using the exact decay bound
  \(2^{-512}\) only if \(512\le\kappa D\le4096\).
  The original common origins are retained; no target reset is made.
- At \(\tau_c=100+D\), add the two unit contacts simultaneously,
  leaving every state coordinate unchanged. Declare total event-work
  allowance \(2\,10^{-6}\), the sum of two declared per-edge
  allowances. Hold the joined support and complete law afterward.
  Require full joined-family acute identity retention and recovery
  on its actual postevent conserved-mean leaf.
- Derive the thirty-coordinate surrogate, ten coordinates per
  component, from the actual postcontact degrees. Retain the class
  tangent only for internal cycle currents and the exact nonlinear
  bridge currents. The full comparison has fifty-four coordinates.
  The nominal surrogate starts at zero form and zero displacement
  from the limiting twists and supplied origins; actual fine-state
  errors remain in the approximation bound, not in a reset.
- Require the maximum absolute error in **each fine form and phase
  coordinate**, relative to the lifted surrogate, to stay strictly
  below \(2\epsilon=2\,10^{-32}\) throughout
  \(0\le\tau-\tau_c\le h=1/100\). This is a full-state
  mathematical approximation allowance in the fixed normalized
  coordinate units, not a sensor error or a new response contrast.
- Retain the exact degree-normalization control: with all phase
  deviations and origins zero, middle port form one and every other
  retained form zero, the middle form rate is \(-1\). Reusing the
  one-contact denominator three gives \(-4/3\) and a nonzero
  actual-degree-weighted form-charge rate \(-4/3\). Require the
  degree-aware assembly's exact storage/loss and charge identities,
  rather than a tolerance-based pass for this rational control.
- Preserve source and protocol before the first reserved control
  evaluation. Use admitted exact primitives and outward rational
  interval bounds. Keep the tiny squared handoff bounds and decay
  as exact rationals. No trajectory integration, topology search,
  horizon/precision sweep or refitted component coefficient is
  admitted. Pass only if fresh formation, every actual handoff,
  contact work, full identity, algebraic controls and the strict
  whole-window approximation allowance all pass unchanged.

The reusable theorem below concerns fixed finite simple undirected unit
central-contact graphs. It does not extend the law to weighted support,
arbitrary port locations or changing contacts during the certified window.
Supplying contacts and organized sources does not derive their occurrence.
The long structural dwell and source accuracy have no laboratory bridge.

## Component coordinates, contact degree and origins

Let \(\mathcal G\) be a fixed finite simple undirected unit graph on
\(m\) components. Give component \(i\) class \(k_i\in\{1,2\}\),
contact degree \(d_i\), and one declared real common phase origin
\(o_i\). Each contact joins the central nodes, so it changes only
their fine degrees, from two to \(2+d_i\). With
\(\alpha=2\pi/9\), use the centered reference twist
\(\Theta_{i,j}=k_i\alpha(j-4)\) and
\(c_i=\cos(k_i\alpha)>0\).

Use the [reflection projection and lift](SINE_REDUCED_CLASS_PORTS.md#the-retained-component-state-and-one-contact-normalization)
\(S,T\) on the five layers \((4),(3,5),(2,6),(1,7),(0,8)\).
They obey \(ST=I_5\), \(TS=(I+R)/2\), and have maximum-norm
operator norm one. For a fine state, the coordinate map is
\[
\bar x_i=Sx_i,\qquad
\bar y_i=S(\theta_i-\Theta_i-o_i\mathbf1).
\]
The reduced trajectory defined below is a surrogate, generally different
from the projection of an evolving nonlinear fine state. Its lifted phase
is \(\Theta_i+o_i\mathbf1+T\bar y_i\). In particular, the common
origin is not also included in \(\bar y_i\).

Let \(L_0=2L_{P_5}\), twice the five-node path Laplacian. The edge
between fine nodes zero and eight has zero difference on an even lift.
The exact reduced mobility mass and internal operator are
\[
M_i=T^{\mathsf T}\operatorname{diag}(d_{\rm fine})T
 =\operatorname{diag}(2+d_i,4,4,4,4),\qquad
A_i=M_i^{-1}L_0. \tag{1}
\]
Thus the first row of \(A_i\) is
\((2,-2,0,0,0)/(2+d_i)\). The other four rows are unchanged from
the one-contact reduction. The class coefficient is inherited from the
cycle; mobility additionally depends on the declared interface. Treating
the one-contact matrix as independent of contact degree would change the
complete law, rather than reuse its reduction.

For each component aggregate the two definite input channels
\[
u_{x,i}=\sum_{j\sim i}(\bar x_{j,0}-\bar x_{i,0}),\qquad
u_{\theta,i}=\sum_{j\sim i}
 \sin(o_j-o_i+\bar y_{j,0}-\bar y_{i,0}).
\]
With \(e_0\) the central layer vector, the component equations are
\[
\begin{aligned}
M_i\bar x_i'&=-L_0\bar x_i-\gamma c_iL_0\bar y_i
             +e_0(u_{x,i}+\gamma u_{\theta,i}),\\
M_i\bar y_i'&=\gamma L_0\bar x_i-\gamma e_0u_{x,i}.
\end{aligned} \tag{2}
\]
Both rows use the same postcontact degree. The state has ten coordinates
per component rather than eighteen. Disconnected contact graphs and an
isolated component also define these equations; a single global recovery
certificate will separately require connected support.

All edge origins must come from the same vector \(o\). Their signed
sum around every contact cycle is zero as a real lift. Independent edge
offsets with a nonzero cycle sum would introduce a different frustrated
model. Common shifts of all \(o_i\) are immaterial to rates. Replacing
\(o_i\) by \(o_i+a_i\) and \(\bar y_i\) by
\(\bar y_i-a_i\mathbf1\) preserves each reconstructed phase and
all rates, since \(L_0\mathbf1=0\). Wrapping arbitrary phase
deviations before applying \(L_0\) is not this coordinate change.

## Exact assembled storage, dissipation and charges

Stack the retained vectors, set \(M=\operatorname{diag}(M_i)\),
and let \(b_{ij}\) be the retained vector with entries plus one and
minus one at the two port layers. Define
\[
\widehat L=\operatorname{diag}(L_0,\ldots,L_0)
             +\sum_{\{i,j\}\in E(\mathcal G)}b_{ij}b_{ij}^{\mathsf T},
\]
\[
V(\bar y)=\frac12\sum_i c_i\bar y_i^{\mathsf T}L_0\bar y_i
 +\sum_{\{i,j\}\in E(\mathcal G)}
  [1-\cos(o_j-o_i+\bar y_{j,0}-\bar y_{i,0})].
\]
The first is exactly \(T^{\mathsf T}L_{\rm fine}T\) for the block
lift. Internal potentials are tangent quadratics; bridge potentials remain
the original nonlinear sine potentials. Equation (2) becomes
\[
\bar x'=-M^{-1}\widehat L\bar x-\gamma M^{-1}\nabla V,
\qquad \bar y'=\gamma M^{-1}\widehat L\bar x. \tag{3}
\]
Consequently the reduced storage
\[
H_{\rm red}=\tfrac12\bar x^{\mathsf T}\widehat L\bar x+V(\bar y)
\]
satisfies, for every retained state and every fixed admitted contact graph,
\[
\boxed{H_{\rm red}'=
 -(\widehat L\bar x)^{\mathsf T}M^{-1}(\widehat L\bar x)\le0.}
\tag{4}
\]
The two mixed terms cancel by symmetry of \(M^{-1}\). This is a
derived balance of this surrogate, not an assertion that its storage
equals the full sine storage away from the reference geometry.

Both \(\mathbf1^{\mathsf T}M\bar x\) and
\(\mathbf1^{\mathsf T}M\bar y\) are constant: the gradients and
Laplacian have zero total sum. The individual component charges instead obey
\[
(\mathbf1^{\mathsf T}M_i\bar x_i)'=u_{x,i}+\gamma u_{\theta,i},
\qquad
(\mathbf1^{\mathsf T}M_i\bar y_i)'=-\gamma u_{x,i}. \tag{5}
\]
They cannot be held separately fixed after attachment.

The observed port values are not automatically energy-conjugate outputs.
For internal storage alone, put \(a_i=L_0\bar x_i\) and
\(b_i=c_iL_0\bar y_i\). Its open-port supply term is
\[
\frac{(a_{i,0}-\gamma b_{i,0})u_{x,i}
          +\gamma a_{i,0}u_{\theta,i}}{2+d_i}.
\]
The internal dissipation is \(-a_i^{\mathsf T}M_i^{-1}a_i\).
Summing internal storages alone omits bridge form storage as well as
bridge potential. Equation (4) includes both and provides the joined
balance; it does not identify \(u_x\bar x_0+u_\theta\bar y_0\)
as physical power.

### A rational control rejects the unchanged one-contact assembly

For the frozen three-component path, set middle port form to one and
every other retained form to zero, with all deviations and origins zero.
The middle port has four neighbors, so its exact form rate is \(-1\).
The adjacent internal layer has rate \(1/2\), and each endpoint
component's port has rate \(1/3\). These give actual-degree-weighted
charge rate \(-4+2+1+1=0\).

Keeping denominator three at the middle port produces \(-4/3\).
Its other rates above remain unchanged, so the actual weighted charge
rate becomes \(-16/3+2+1+1=-4/3\). The corresponding phase-charge
rate is \(4\gamma/3\), rather than zero. The correct storage loss
at this state is \(-17/3\); the erroneous mobility gives \(-7\).
Thus nonincreasing storage under some changed metric would not validate
the declared model. This exact algebraic control is independent of a
numerical trajectory or the reserved formation-family bounds.

## Uniform error against the full nonlinear fine law

Extend equation (2) to all nine coordinates per component solely as a
proof device: linearize internal cycle sine currents about their own
twist, retain exact bridge sine currents, and keep the actual fine
mobility. Independent reflections fix every central port. This extended
surrogate preserves their even subspace, and projection/lift of that
subspace gives (2) exactly. The full nonlinear sine law does not generally
preserve the same even subspace: quadratic internal terms can generate
odd deviations. No such invariance is assumed below.

Write the full ideal comparison as
\(x=\gamma v\), \(\theta=\Theta+o+y\), with \(v(0)=y(0)=0\),
and put \(\eta=\gamma^2\), \(A=KL\). The diffusion semigroup
\(\exp(-At)\) is a maximum-norm contraction, \(\|A\|_\infty\le2\),
and the normalized sine field is globally Lipschitz with constant two.
Its reference forcing is supported at the central ports. Define
\[
\sigma=\|KS(\Theta+o)\|_\infty
 =\max_i\frac{|\sum_{j\sim i}\sin(o_j-o_i)|}{2+d_i}. \tag{6}
\]
This retains the algebraic sum of incident currents, rather than fitting
a forcing from the evaluated response. Suppose
\[
0\le h<1/3,\qquad C_h=1-\tfrac23\eta h^2>0.
\]
Duhamel's formula and \(y'=\eta Av\) give
\[
\|v(t)\|_\infty\le\sigma t+2\int_0^t\|y(s)\|_\infty ds,
\qquad
\|y(t)\|_\infty\le2\eta\int_0^t\|v(s)\|_\infty ds.
\]
For \(B_h=\sup_{0<t\le h}\|v(t)\|_\infty/t\), these imply
\(B_h\le\sigma+(2/3)\eta h^2 B_h\). Smoothness at zero and
global existence justify the finite supremum. Therefore
\[
\|v(t)\|_\infty\le\frac{\sigma t}{C_h},\qquad
\|y(t)\|_\infty\le\frac{\eta\sigma t^2}{C_h}
\quad(0\le t\le h). \tag{7}
\]

On each internal edge the first-order sine remainder has magnitude at
most half the squared deviation gap. Each gap is at most
\(2\|y\|_\infty\), and internal degree divided by actual fine
degree is at most one. Hence the full normalized internal-field defect
is at most \(2\|y\|_\infty^2\), irrespective of the number of
components or central contacts. Bridge fields have no reduction defect.
In physical form coordinates the complete-row defect is thus at most
\(2\gamma^5\sigma^2t^4/C_h^2\).

Both full and extended surrogate fields are globally Lipschitz in the
joint maximum norm with constant at most \(2+2\gamma<3\).
Variation of constants for the error, or the integral Gronwall
inequality, bounds the fine ideal-to-lifted-surrogate discrepancy by
\[
E_{\rm nonlinear}(h)
\le\frac{2\gamma^5\sigma^2 h^5}{5C_h^2(1-3h)}. \tag{8}
\]
For example, take the supremum of the error on \([0,h]\): its
integral inequality is at most the integral of the stated defect plus
\(3h\) times that supremum. This proves (8) without evaluating an
exponential and covers both form and phase errors, including feedback
from generated odd modes. The restriction \(h<1/3\) belongs to this
sufficient bound, not to existence of either flow.

Every actual formation-family endpoint has fine maximum-norm distance
at most \(\epsilon\) from its ideal comparison state. Global
Lipschitz comparison separately bounds its actual full trajectory's
distance from the ideal full trajectory by
\(\epsilon/(1-3h)\). In particular, arbitrary admitted odd initial
errors and their independent component correlations are retained.
The whole-window full-state error certificate is
\[
\boxed{E_{\rm full}(h)=\frac{\epsilon}{1-3h}
 +\frac{2\gamma^5\sigma^2 h^5}{5C_h^2(1-3h)}.} \tag{9}
\]
It applies to every fine coordinate after lifting the surrogate and
restoring each reference twist and origin. At \(h=0\) it is the
initial uncertainty bound. If all origins agree, \(\sigma=0\), so
the ideal zero-deviation comparison is stationary; actual preparation
errors still remain. The bound is uniform in graph size because of
normalized degrees, not because the graph or initial state was omitted.

## Actual source handoff, support work and joined identity

Reuse the [unprobed nonlinear handoff](SINE_FORMED_CLASS_CONTACT.md#the-actual-formation-images-reach-the-contact-tolerance)
with every original formation source re-admitted. Common phase origins
are exact isolated-cycle symmetries. Thus its per-class proof transfers
to each component independently without rotating a reached state or
modifying any source-family width. The report must establish the actual
endpoint bounds before equation (9) becomes a claim about that family.

For the following joined certificate require \(m\ge2\) and connected
\(\mathcal G\), and put \(\Delta=\operatorname{diam}(\mathcal G)\).
Any fine node lies within four cycle edges of its center, so the joined
fine diameter is at most \(\Delta+8\). For \(n=9m\), every
mean-zero real vector \(z\) satisfies
\[
\|z\|_2^2\le\frac n4(\max z-\min z)^2
 \le\frac{n(\Delta+8)}4z^{\mathsf T}Lz.
\]
The first inequality is the bounded-range variance inequality; the second
uses a shortest path and Cauchy--Schwarz. Consequently
\[
\lambda_2(L)\ge\lambda_*:=\frac4{9m(\Delta+8)}. \tag{10}
\]
This proof also permits cycles in the contact graph. No tree-only
stability theorem is being transferred.

The aligned joined target has each component's class twist, a single
common phase origin, zero bridge phase gaps, and common form. Internal
sine forces cancel and bridge currents vanish, so it is critical for
the joined support. In its Euclidean quotient ball of radius \(r\),
with \(0<r\le1/12\), every continuous representative of a reference
edge gap changes by at most \(\sqrt2r\). The fixed closing-edge
integer is retained. Thus all circular gaps remain acute and their
cosines are at least
\[
c_r=\cos(4\pi/9+\sqrt2r)>1/20.
\]
The [acute-chart coercivity argument](SINE_FORMED_CLASS_CONTACT.md#one-joined-state-event-work-and-both-retained-identities)
gives full sine storage excess at least
\(\kappa_*\|z\|_2^2\), where
\[
\kappa_*=\tfrac12\lambda_*c_r. \tag{11}
\]
Here \(z\) removes the arithmetic common form and phase origins for
geometric comparison. Those arithmetic means need not remain constant;
the conserved actual-degree-weighted means are retained below.

At contact write \(x_i\) and \(u_i=\theta_i-\Theta_i-o_i\mathbf1\)
for the actual component deviations. Their Euclidean norms are at most
\(\epsilon\), and their component sums are exactly zero, inherited
from the isolated degree-two law. If \(\bar o=m^{-1}\sum_i o_i\),
the quotient squared distance obeys
\[
Z_0^2\le 2m\epsilon^2+9\sum_i(o_i-\bar o)^2. \tag{12}
\]
The cross terms vanish because the phase residual sum is zero on each
component. Replacing the actual correlated family by unrelated absolute
boxes would lose this equality.

The cycle Laplacian has largest eigenvalue at most four. Its form and
phase storage excesses are each at most \(2\epsilon^2\) per
component; the phase gradient at the twist is zero and its Hessian norm
is at most four. The added bridge form storage is at most
\(d_{\max}\sum_i|x_{i,4}|^2\le d_{\max}m\epsilon^2\).
Using \(1-\cos a\le a^2/2\) on each new bridge gives
\[
H_0-H_*\le B_H:=(4+d_{\max})m\epsilon^2
 +\frac12\sum_{\{i,j\}\in E}
       (|o_j-o_i|+2\epsilon)^2. \tag{13}
\]
The exact event work is the sum of the added edge storages,
\[
W=\sum_{\{i,j\}\in E}\left\{
 \tfrac12(x_{j,4}-x_{i,4})^2+
 1-\cos(o_j-o_i+u_{j,4}-u_{i,4})\right\},
\]
and independently
\[
0\le W\le B_W:=2|E|\epsilon^2
 +\frac12\sum_{\{i,j\}\in E}
       (|o_j-o_i|+2\epsilon)^2. \tag{14}
\]
This supplied work is not a continuous loss reserve. The subsequent
full-law balance is exactly
\[
H'=-(Lx)^{\mathsf T}K(Lx)\le0.
\]
If \(Z_0^2<r^2\) and \(B_H<\kappa_*r^2\), a first-exit
argument traps the actual full family in the acute quotient ball for
all future uninterrupted joined flow. All target cycle periods are
retained, including the individual C9 windings and the zero periods
on the aligned contact cycles. Strict convexity in this chart and the
largest invariant zero-loss set give recovery to its unique target
geometry on each actual conserved-mean leaf, by the
[whole-state recovery argument](SINE_PATTERN_RECOVERY.md#sine-cycle-recovery).
This identity/recovery conclusion uses the actual full sine storage,
not the surrogate balance (4) or its finite error estimate.

Total degree mass is \(M_{\rm tot}=18m+2|E|\). Since each target
twist is centered and its central phase is zero, the exact postevent
conserved origins are
\[
\mu_x=\frac{\sum_i d_i x_{i,4}}{M_{\rm tot}},\qquad
\mu_\theta=\frac{\sum_i(18+d_i)o_i+\sum_i d_i u_{i,4}}
                  {M_{\rm tot}}. \tag{15}
\]
Each residual fraction has magnitude at most
\(2|E|\epsilon/M_{\rm tot}\). Component origins are not conserved
separately after joining. The final common form is \(\mu_x\), and
the final common phase origin relative to the aligned twists is
\(\mu_\theta\), using the retained continuous lifts.

## Frozen-control sufficiency and scope

For the declared path, \(m=3\), \(|E|=2\), \(d_{\max}=2\),
and \(\Delta=2\), so \(\lambda_*=2/135\) and
\(\kappa_*r^2>r^2/2700\). The reference forcing in (6) is
\(\sigma=\sin(1/1000)/2\): the two middle currents add, while
each endpoint current is divided by three. In (12), the origin
contribution is \(6(1/1000)^2\). Equations (13)--(14) retain both
contacts and every actual endpoint error. The source, handoff and
outward strict-margin computations remain separate prerequisites of
the reserved certificate; these formulas do not substitute a cached
prior verdict for any of them.

The result is a context-aware composition rule with one scalar contact
degree per component, not a context-free reuse of the old denominator.
It supplies an exact reduced balance and controlled finite full-state
approximation under the declared unit law. It neither proves a minimal
realization nor an exact nonlinear quotient. It does not install a
support event, select component birth, infer pairwise physical forces,
or establish a new many-body response observation. Failure of a
sufficient finite error or trapping bound must be reported as unavailable,
unless an independent obstruction is proved. No sensor or laboratory
clock is identified by this mathematical coordinate-error allowance.

## Retained evaluation of the unchanged composition protocol

The first reserved evaluation, performed after preserving the protocol,
proof and producing source, returned `certified_sine_port_composition`
with no unavailable reasons. The frozen stopping rule passed unchanged.
Fresh formation and both class handoffs passed; all three components
therefore retain their independent original source uncertainties. The
joined identity/recovery, supplied-work and uniform approximation flags
are all true. These are analytic interval certificates, not observations
of a propagated or measured trajectory.

The saved upper bounds have the following decimal summaries:

| Quantity | Retained value, approximately | Declared comparison |
| --- | --- | --- |
| Whole-window full-coordinate error | \(1.0339346076859747\,10^{-32}\) | Strictly below \(2\,10^{-32}\) |
| Ideal nonlinear reduction defect | \(3.006772634428256\,10^{-35}\) | Included once in the total error |
| Initial joined quotient distance squared | \(6\,10^{-6}\) | Strictly below \(r^2=1/144\) |
| Full sine storage excess | \(1\,10^{-6}\) | Below the barrier \(2.9141691574402282\,10^{-6}\) |
| Total supplied contact work | \(1\,10^{-6}\) | At most the declared \(2\,10^{-6}\) |

The tiny positive endpoint-error contributions in the last three rows
remain in the exact report; the decimal summaries do not round those
contributions away for certification. The following integers divided
by \(2^{128}\) are the exact retained outward **lower** margins:

| Strict margin | Lower numerator over \(2^{128}\) |
| --- | --- |
| Approximation allowance minus total error | `3287350` |
| Squared radius minus initial quotient bound | `2361030298304991476603765637298244192` |
| Full storage barrier minus excess bound | `651358011580819424763819880101342` |
| Work allowance minus event-work bound | `340282366920938463463374607418156` |

The exact algebraic control also passed: the middle port rate is \(-1\),
the actual weighted charge rate is zero, form storage is two, and storage
rate is \(-17/3\). Reusing denominator three gives middle rate
\(-4/3\) and weighted charge rate \(-4/3\), as predicted before
evaluation. The control rejects incorrect contact normalization without
claiming a new receiver-response observation.

Retain the [frozen protocol](../../docs/assets/sine_formed_classes/port-composition-v1.protocol.json),
[complete report](../../docs/assets/sine_formed_classes/port-composition-v1.json),
[producing source archive](../../docs/assets/sine_formed_classes/port-composition-v1.source.zip)
and [evidence manifest](../../docs/assets/sine_formed_classes/port-composition-v1.manifest.json).
The archive preserves this owner's pre-evaluation protocol and derivation;
this retained-result section was added afterward. Exact arithmetic checks
against the saved report confirmed primitive-input equality, the stated
margin endpoints and stopping inequalities without rerunning its producer.

The shared implementation exposes `evaluate_sine_port_composition` and
`assess_sine_port_composition`, with `SinePortCompositionState` and
`SinePortComposition`, in
[`relational_sine_port_composition.py`](../../src/tnfr/physics/relational_sine_port_composition.py).
The [contract](../../docs/contracts/relational/SINE_PATTERNS.md#sine-reduced-port-composition)
and [usage guide](../../docs/guides/relational/SINE_PATTERNS.md#sine-reduced-port-composition)
retain admission, availability and coordinate conventions. The independent
[composition controls](../../tests/physics/test_sine_port_composition.py)
check fine-node projection, mobility, storage/supply, origins, actual-family
bounds and strict numerical boundaries; they do not replace the all-time
proof or execute a trajectory.

The result establishes degree-aware reduced composition under the supplied
unit sine law, with finite full-state accuracy and separate actual identity
retention. The structural dwell remains \(10^{13}\), and the approximation
allowance remains \(2\,10^{-32}\); neither has acquired an empirical
clock, preparation or sensor interpretation. No runtime speedup, autonomous
support selection, minimal state or uniquely selected physical law follows.
