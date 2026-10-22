# Complete port inherited from an organized C9 source

<a id="organized-source-port"></a>

## Declared model, preparation and admission question

The [auxiliary reservoir classification](ORGANIZED_PARENT_RESERVOIR_MEMORY.md)
retains a supplied scalar source law. This separate question asks what
interaction an existing C9 organization actually supplies through a declared
attachment. The earlier reservoir protocols and outcomes remain unchanged.

| Item | Fixed choice and scope |
| --- | --- |
| Host | Original four marked C9s on the matching 36-node support, with child `c=2*g+a`, node `9*c+j`; retain the two original parent markings |
| Donor | One additional marked C9, child `c=4`, node `36+j`; no replacement by a new primitive reservoir |
| Contact | Nine supplied unit edges `(j,36+j)` join donor slots to host child zero; five internal cycles plus the original 36 host matching edges give 90 unit edges total |
| Complete law | `x_tau=(-Lx+gamma*S(theta))/4`, `theta_tau=gamma*Lx/4`, where `L` and `S` use the actual enlarged support and `gamma=1/(1023*pi)` |
| Normalization and capacity | Uniform factor `1/4` in BOTH rows, held unit capacities. Degrees are five on host child zero, four on other host children and three on the donor; do not replace the declared factor by reciprocal degree |
| Clock and units | `tau=(1023/1024)*t`, with declared elapsed origins for each preparation stage and attachment; both rows transform together. Retain existing structural form, phase and storage units |
| Fine state and lift | All 45 signed forms and 45 phases retained. Write `theta[c,j]=psi[j]+delta[c,j]`, `psi[j]=2*pi*(j-4)/9`, with continuous real offsets and the fixed closing-edge turn on every C9 |
| Exact carrier family | `x[c,j]=X[c]`, `theta[c,j]=psi[j]+Phi[c]`; test invariance and all ten collective rows without asserting closure for arbitrary fine states |
| Observations | Five child form means and lifted phase-offset means; also test the eight host child means with the two conserved whole-network totals retained as preparation constants, defined as the sums of the five child means |
| Scalar comparison | On an open domain of independently variable donor form and phase, hold the original host coordinates, support subtraction and clock fixed. Test closure of both port channels from one continuously differentiable donor scalar; state separately what changes when the two conserved totals are retained |
| Host preparation | Rebuild the existing acquired-parent family at its unchanged absolute slow time `s=4`, from the four original independent class-one sources with form/phase allowances `1e-10`; retain every reached coordinate and its fine error |
| Donor preparation | Independently rebuild the original isolated class-one ramp source with form/phase allowances `1e-10` and separate zero-sum errors. The uniform `1/4` isolated law is the old `1/2` law at half speed: formation time is 200 and the fixed recovery dwell is `2*10^13` in the new `tau` clock. No source or reached state is reset |
| Supplied relative origin | At attachment translate the donor's reached phase lift by `1/8`, retaining its zero form mean and all fine deviations; host origins and acquired state are unchanged |
| Attachment work | Count the actual new-edge storage `sum_j[(x[36+j]-x[j])^2/2+1-cos(theta[36+j]-theta[j])]` separately from prior host/donor storage and dissipation; no passive occurrence or funding mechanism is supplied |
| Identity | Admit every initial child winding from its actual source and retained chart. Carrier invariance and initial identity alone do not certify finite maintenance of independently perturbed acquired children |
| Required result | Derive the full port, source reaction, storage and sufficient collective information; prove an exact conditional representation or a scoped obstruction. No finite readout, disruption target, new trajectory or post-attachment lifetime is selected |

The attachment and preparation remain supplied. This is detached law and
source admission, not execution of an operator or selection of its occurrence.
An organized source may change the host's phase row as well as its form row;
its law need not reproduce the previous form-only auxiliary port.

Re-admit primitive source allowances before rebuilding recovery and host
acquisition. Account for all clock changes and initial storage. The prepared
host and donor may be synchronized to the attachment by declared preparation
start times; do not insert an additional unproved waiting interval.

Retain the distinction between the exact carrier restriction, actual fine
means and their unresolved correlations. Conserved preparation constants
are information, not inferred causes or extra evolving coordinates. No
scalar-minimality claim may discard them silently. No physical identification,
autonomous support formation or universal source realization follows.

The declaration above was committed as `02a6415f` before the following
assessment. Its conclusion is an exact two-channel interaction on a
ten-coordinate carrier family, with a separate initial acquired-source
admission. The inherited source does not realize the previous form-only
auxiliary law. The amount of evolving source state needed depends on the
host observations and conserved preparation information retained.

## 1. Fine law, normalization and attachment storage

Let \(H\) denote the original 36-node host, \(D\) the donor C9, and
\(B\) the nine new edges. Each internal cycle has nine unit edges;
the four original matching child interfaces contribute 36 more, and the
donor interface contributes nine. Thus the enlarged support has 45 nodes
and 90 edges. Its degrees are those in the declaration, rather than a
common degree four. With \(e=1023/1024\), its complete rows are

\[
 \begin{split}
 S_i(\theta)&=\sum_{k\sim i}\sin(\theta_k-\theta_i),\\
 x_\tau&=\tfrac14[-Lx+\gamma S(\theta)],\qquad
 \theta_\tau=\tfrac\gamma4Lx,\qquad \tau=e t .
 \end{split}\tag{1}
\]

The uniform mobility \(1/4\) is a supplied coefficient. It is not the
reciprocal of each new degree, a changed capacity, or an inference from
the nodal identity. Both original-time rows are \(e\) times (1).
For unit capacity the independently declared pressure is the form row
in the selected clock, with the corresponding structural units. There
are no continuous external inputs after the supplied attachment.
These equations are the detached comparison law, not the native
resultant-direction pressure or an operator word.

Use continuous phase lifts, retaining the original closing-edge branch
on each child when reading its winding. Sine and cosine are periodic, so
the vector field itself is independent of those branch representatives.
It is globally Lipschitz on the finite-dimensional lifted state: the
linear form rows and globally bounded sine derivatives give a unique
complete solution for every finite initial state. This is existence of
the specified law, not an assertion of future identity.

The actual storage and its derivative are

\[
 \begin{split}
 H_{45}(x,\theta)&=\tfrac12x^TLx+
                 \sum_{\{i,k\}}[1-\cos(\theta_k-\theta_i)],\\
 (H_{45})_\tau&=-\tfrac14\|Lx\|_2^2,\qquad
 (\mathbf1^Tx)_\tau=(\mathbf1^T\theta)_\tau=0 .
 \end{split}\tag{2}
\]

Indeed \(\nabla_xH_{45}=Lx\) and \(\nabla_\theta H_{45}=-S\);
substitution of both rows cancels the reciprocal products. Symmetry and
oddness cancel every edge in the two sums. The phase sum refers to the
declared continuous lifts, not a circular mean or phase-resultant angle.

For each new edge put

\[
 u_j=x_{36+j}-x_j,\qquad
 \delta_j=\theta_{36+j}-\theta_j,\qquad
 H_B=\sum_{j=0}^8[\tfrac12u_j^2+1-\cos\delta_j].
 \tag{3}
\]

The attachment changes no state coordinate. Consequently
\(H_{45}^+-H^- = H_B\), where \(H^-\) includes both the actual host
and donor internal storage. Each edge is counted once. In particular,
the contact form storage depends on a difference of two existing forms;
no independent \(r^2/2\) storage has been added. Positive \(H_B\)
requires a supplied structural-work source. Past dissipation is not a
declared reserve, and (3) supplies neither funding nor event occurrence.

## 2. Actual boundary channels, reaction and regional balance

Let \(L_H,S_H\) contain only the original host edges and \(L_D,S_D\)
only the donor cycle. A selector \(E\) inserts a nine-vector at the
host slots \(j=0,\ldots,8\). Define the two vector channels

\[
 f_j=\frac{u_j+\gamma\sin\delta_j}{4},\qquad
 g_j=-\frac{\gamma u_j}{4}.
 \tag{4}
\]

The full law splits exactly as

\[
 \begin{array}{ll}
 (x_H)_\tau=(-L_Hx_H+\gamma S_H)/4+Ef,
   &(\theta_H)_\tau=\gamma L_Hx_H/4+Eg,\\
 (x_D)_\tau=(-L_Dx_D+\gamma S_D)/4-f,
   &(\theta_D)_\tau=\gamma L_Dx_D/4-g .
 \end{array}\tag{5}
\]

The source reacts in both channels, with the opposite currents under
this common mobility. Its boundary values are evolving fine state; they
are not a prescribed waveform. The host's two charge rates are
\(\sum_j f_j\) and \(\sum_j g_j\), while the donor rates are their
negatives. A zero form gap removes the instantaneous phase channel,
but it does not remove the dependence of the form channel on phase.

To make the regional work explicit, write
\(a_H=L_Hx_H\), \(a_D=L_Dx_D\), \(b_H=-S_H\), \(b_D=-S_D\),
and let \(H_H,H_D\) denote the respective internal storages. Direct
differentiation gives

\[
 \begin{split}
 (H_H)_\tau&=-\tfrac14\|a_H\|_2^2+
                      (E^Ta_H)^Tf+(E^Tb_H)^Tg,\\
 (H_D)_\tau&=-\tfrac14\|a_D\|_2^2-a_D^Tf-b_D^Tg,\\
 (H_B)_\tau&=\sum_j[u_j(u_j)_\tau+
                                  \sin\delta_j(\delta_j)_\tau],\\
 (H_H+H_D+H_B)_\tau&=-\tfrac14\|Lx\|_2^2 .
 \end{split}\tag{6}
\]

These supplies depend on internal form pressures and phase gradients.
They are not form times form input alone. The sum of the two isolated
losses is generally not the complete loss; differentiating the contact
storage restores precisely (2). This is the same component-balance
method as the [earlier causal composition](SINE_CLASS_INTERFACE_COMPOSITION.md#sine-composition-balances),
with the present support, mobility and nine actual channels re-derived.

## 3. Exact ten-coordinate restriction

Let \(c=0,1,2,3\) label the original children and \(c=4\) the donor.
The interchild edge set and its unweighted Laplacian are

\[
 \begin{split}
 \mathcal E_C&=\{\{0,1\},\{2,3\},\{0,2\},\{1,3\},\{0,4\}\},\\
 K&=\begin{pmatrix}
 3&-1&-1&0&-1\\
 -1&2&0&-1&0\\
 -1&0&2&-1&0\\
 0&-1&-1&2&0\\
 -1&0&0&0&1
 \end{pmatrix}.
 \end{split}\tag{7}
\]

On the declared family
\(x_{c,j}=X_c\), \(\theta_{c,j}=\psi_j+\Phi_c\), the two internal
cycle sine currents cancel and internal form currents vanish. Every
matching edge has the same form and phase gaps at every slot. Both
fine rows are therefore constant within each child. Smooth uniqueness
proves invariance, and substitution gives all ten rows:

\[
 q=KX,\qquad
 s_c(\Phi)=\sum_{b:\{c,b\}\in\mathcal E_C}\sin(\Phi_b-\Phi_c),
 \qquad (X_c)_\tau=(-q_c+\gamma s_c)/4,\quad
 (\Phi_c)_\tau=\gamma q_c/4 .
 \tag{8}
\]

Writing \(R=X_4\), \(\Psi=\Phi_4\),
\(u=R-X_0\), \(\delta=\Psi-\Phi_0\), the donor and its increments
to the original host child-zero rows are in particular

\[
 \begin{array}{ll}
 F=(u+\gamma\sin\delta)/4,&G=-\gamma u/4,\\
 R_\tau=-F,&\Psi_\tau=-G=\gamma u/4.
 \end{array}\tag{9}
\]

Thus the constituent supplies a closed two-coordinate reaction on this
family. The ten coordinates include both the donor and the host; the
port has not counted the source again as an independent force.
Every carrier child keeps its winding one because its internal gaps
remain \(2\pi/9\). This exact statement on the carrier does not prove
retention for acquired transverse errors.

The inherited storage is the pullback, not a postulated source energy:

\[
 \begin{split}
 H_{10}={}&45(1-\cos\alpha)+\frac92X^TKX
           +9\sum_{\{c,b\}\in\mathcal E_C}[1-\cos(\Phi_b-\Phi_c)],\\
 (H_{10})_\tau={}&-\frac94\|KX\|_2^2,
 \qquad \alpha=2\pi/9 .
 \end{split}\tag{10}
\]

For example the new contact term is exactly
\(9[u^2/2+1-\cos\delta]\). The donor's internal carrier storage
\(9(1-\cos\alpha)\) is independent of its two common coordinates.
This does not make its interaction storage or reaction independent of
them. Equations (8)--(10) use the actual common factor \(1/4\);
the final-degree normalization of the separate
[pair inheritance model](ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md#organized-inheritance-restriction)
would instead define a different complete law here.

## 4. Conserved preparation information and the scalar obstruction

The carrier totals

\[
 T_x=\sum_{c=0}^4X_c,\qquad T_\phi=\sum_{c=0}^4\Phi_c
 \tag{11}
\]

are conserved. They are the whole fine sums divided by nine because
\(\sum_j\psi_j=0\). If all eight host child coordinates and both
initial totals are retained, then at every time

\[
 R=T_x-\sum_{c=0}^3X_c,\qquad
 \Psi=T_\phi-\sum_{c=0}^3\Phi_c .
 \tag{12}
\]

Substitution into the first eight rows of (8) is an exact autonomous
eight-coordinate law on that fixed information leaf. Summing those
rows reproduces both donor rows, so the reconstruction remains valid.
No extra evolving donor scalar is required there. The two constants
retain initialization information; they are not inferred from current
host observations and do not become an additional force. Changing phase
representatives requires transporting the retained lift and its total.

For the separately declared scalar comparison, hold the host state,
original host support subtraction and clock fixed, and let \(R,\Psi\)
vary independently in an open contact chart with \(|\delta|<\pi/2\).
The derivative of the complete port output (9) is

\[
 D_{(R,\Psi)}(F,G)=
 \begin{pmatrix}1/4&\gamma\cos\delta/4\\-\gamma/4&0\end{pmatrix},
 \qquad \det D(F,G)=\gamma^2\cos\delta/16>0 .
 \tag{13}
\]

If a continuously differentiable scalar observation \(r=h(R,\Psi)\)
and continuously differentiable output maps reconstructed both channels,
their derivative would factor through a one-dimensional space and
have rank at most one. Equation (13) is a contradiction. The obstruction
also holds for arbitrary single-valued output maps: (13) makes the port
locally injective, whereas a continuously differentiable scalar cannot
be injective on an open two-dimensional neighborhood. A nonzero scalar
gradient gives a nontrivial local level curve by the implicit-function
theorem; if the gradient vanishes throughout a ball, the scalar is
constant there. In either case identical scalar observations would have
to reconstruct distinct port values.
Thus no such scalar closes even the instantaneous complete port on this
open domain, and hence it cannot provide a complete smooth autonomous
representation there. This excludes a regular scalar reduction, not discontinuous
encodings, a fixed curve, special singular points or a model retaining
additional preparation constants.

For the particular observation \(r=R\), two donor origins with the
same form give different \(F\) whenever their sine gaps differ.
If the phase total and all host coordinates are separately retained,
the donor phase can instead be recovered and \(R\) suffices. If only
the form total is retained, donor phase is a sufficient remaining
coordinate. With both totals, (12) applies. These distinctions prevent
a dimension claim from silently discarding available initialization.

The old auxiliary law adds \(b r\) only to the host form row and
leaves its phase row unchanged. Relative to precisely that unchanged
host and clock, the organized-source port changes the phase row by
\(-\gamma u/4\) at every contacted slot. For any \(u\ne0\), no
choice of the old scalar reproduces this row. Even at an instantaneous
\(u=0\), the form current depends on the donor phase and its subsequent
reaction. A coincidence at a selected state is not equivalence of
complete laws on an open domain. This is a scoped nonrealization of
that auxiliary port, not a universal obstruction to other reductions,
different interfaces or separately admitted controlled channels.

## 5. Fine means do not remove contact correlations

For arbitrary fine states define the same child means, and write
\(\theta_{c,j}=\psi_j+\Phi_c+\eta_{c,j}\), with
\(\sum_j\eta_{c,j}=0\). All matching form sums reduce exactly to
mean differences. The actual mean rows are consequently

\[
 \begin{split}
 (X_c)_\tau&=\tfrac14[-(KX)_c+
          \gamma\sum_{b:\{c,b\}\in\mathcal E_C}J_{cb}],\\
 (\Phi_c)_\tau&=\tfrac\gamma4(KX)_c,\\
 J_{cb}&=\frac19\sum_j
     \sin(\Phi_b-\Phi_c+\eta_{b,j}-\eta_{c,j}).
 \end{split}\tag{14}
\]

The two totals still reconstruct donor means, but generally not the
currents \(J_{cb}\). For an explicit matched-observation witness,
take all forms zero and all host offsets zero, with donor origin
\(\delta=1/10\). In state A all residuals vanish; in state B give
the donor the offsets \((h,-h,0,\ldots,0)\), \(h=1/100\), keeping
every host fine coordinate unchanged. Both states have the same ten
means, both totals, all child windings and the same host registration.
All internal donor gaps remain acute: their deviations from
\(\alpha\) have magnitude at most \(2h\). The donor contact current
is nevertheless

\[
 J_{04}^{A}=\sin\delta,\qquad
 J_{04}^{B}=\sin\delta\frac{7+2\cos h}{9},\qquad
 (X_0^B-X_0^A)_\tau=
        \frac{\gamma\sin\delta}{18}(\cos h-1)\ne0 .
 \tag{15}
\]

The last notation compares their rates at these initial states, not a
subtraction of evolved trajectories. The witness also works for any
sufficiently small positive \(h\). It proves lack of exact closure on
fine neighborhoods; it does not assert that these particular states
are reached by the fixed acquired source. Fine correlations require
retained coordinates, a justified approximation, or causal elimination
with complete initial state. An earlier model's memory coefficients
cannot be transferred merely because this port also has two channels.

## 6. Actual donor preparation, host handoff and event work

Re-admit the original class-one primitive preparation, independently of
any stored recovery verdict:

\[
 \begin{split}
 m&=(2046/9)(355/113)^2,\\
 x_j(0)&=m(j-4)+\xi_j^x,\qquad
 \theta_j(0)=\xi_j^\theta,\\
 |\xi_j^x|,|\xi_j^\theta|&\le10^{-10},\qquad
 \sum_j\xi_j^x=\sum_j\xi_j^\theta=0 .
 \end{split}\tag{16}
\]

These are signed forms and primitive lifted phases on the original
ordered unit C9, with held unit capacities, sine potential and the same
\(\gamma\). The shared
[acquisition and recovery owner](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-carried-handoff)
rebuilds all consumed source norms, nonlinear capture and recovery from
those primitives at deformation zero. The original donor vector field
has factor \(1/2\) in both rows. If its elapsed clock is \(v\),
then the new isolated solution is exactly

\[
 z_{\rm new}(\tau)=z_{\rm old}(\tau/2),\qquad v=\tau/2 .
 \tag{17}
\]

The chain rule applies to both form and phase, with unchanged initial
state; uniqueness proves the identity. It does not replay an old
trajectory or transplant a coefficient into the enlarged support.
The donor therefore acquires winding one by new elapsed time 200,
and at new elapsed time \(2(100+10^{13})\) its reached residuals obey

\[
 \|x_D\|_2\le q,\qquad
 \|\theta_D-\psi\|_2\le q,\qquad q=10^{-70},
 \tag{18}
\]

with their exact zero means retained. No coordinate is reset. The new
original-clock duration is \((1024/1023)2(100+10^{13})\), not the
old preparation duration. The donor's full initial storage is bounded by

\[
 B_1=36m^2+32m\,10^{-10}+36\,10^{-20}<10^9 .
 \tag{19}
\]

The nine edge differences of the nominal ramp give the first term;
the actual form and phase errors give the remaining terms. Its isolated
law dissipates this storage at its own declared speed. At the recovered
endpoint, stationarity of the winding-one target and
\(\|L_{C9}\|_2\le4\) bound its storage above by
\(9(1-\cos\alpha)+4q^2\). This reached bound does not remove the
initial cost or imply that lost storage can fund attachment.
Its excess above \(9(1-\cos\alpha)\) is also nonnegative: the segment
between the reached phase and the winding target remains in the acute
fixed-winding chart, where the phase Hessian is a positive weighted
Laplacian; the target is stationary and the form term is nonnegative.

Separately rebuild the host's actual acquisition at its unchanged
\(s=4\), using all four independently uncertain sources and the
[host handoff proof](ORGANIZED_PARENT_ACQUISITION.md#2-source-admission-and-full-fine-state-error).
Its reference uses gaps \(B,\Delta\) and forms
\(v=-\gamma Y_B\), \(u=-\gamma Y_\Delta\), where
\(0\le Y_D\le(1+2\gamma^2)D\). At \(s=4\), retain

\[
 B<1/64,\quad \Delta<1/640,\quad
 \|x_H-x_H^*\|_2\le R<10^{-60},\quad
 \|\theta_H-\theta_H^*\|_2\le Z<10^{-60}.
 \tag{20}
\]

The same upper exponential factor bounds both gaps, whose initial
values are \(1/2\) and \(1/20\); hence the second bound follows by
using the host proof's common factor below \(1/32\). No proportionality
of their actual nonlinear trajectories is assumed. In particular the
child-zero reference phase offset is \(-(B+\Delta)/2\), and its
form is \(\gamma(Y_B+Y_\Delta)/2\).

The four host primitive costs, their original 36-contact work and
subsequent loss retain that owner's accounting. Together with (19),
the five original isolated source budgets sum to at most \(5B_1\);
the host assembly work is a separate supplied term. At the new contact,
the pre-event storage is exactly
\(H_H(x_H(s=4),\theta_H(s=4))+H_D(x_D,\theta_D)\), not five target
storages. The independent preparation starts may be offset to reach
their declared endpoints at the same attachment time. No extra waiting
or unproved frozen state is inserted.

Translate the actual disconnected donor phase origin by \(1/8\) as
declared. This is a symmetry of its isolated dynamics and storage; once
contact is added the relative origin is consumed by the interaction.
Both source residuals and all host coordinates are retained. Set

\[
 b=11/1280,\qquad e_\theta=Z+q,\qquad e_x=R+q,
 \quad d_+=1/8+b+e_\theta,\quad
 u_+=\gamma(1+2\gamma^2)b+e_x .
 \tag{21}
\]

The actual host keeps its zero global form and lifted-offset sums, and
the donor's translation changes only its phase mean. Consequently this
entire prepared family has \(T_x=0\), \(T_\phi=1/8\). These constants
reconstruct its donor means from host means, but (14) still retains the
actual fine contact correlations.

Every actual contact gap and form difference obeys
\(1/8-e_\theta<\delta_j<d_+\) and \(|u_j|\le u_+\).
They use the full independently uncertain sources; no matching residual
is assumed. The elementary cosine bound and positivity on this contact
chart then give

\[
 \frac1{16}<
 9\left[\frac12(3/25)^2-\frac1{24}(3/25)^4\right]
 <H_B\le\frac92(d_+^2+u_+^2)<\frac1{10} .
 \tag{22}
\]

For the lower bound, (20) gives \(\delta_j>3/25\), while
\(\delta_j<1/4\); \(1-\cos\delta\) is increasing there and its
quartic Taylor lower bound applies. The upper bound uses
\(1-\cos\delta\le\delta^2/2\), \(\gamma<1/3000\), and the
retained tiny errors. These rationally separated constants are event
work bounds, not a prediction of subsequent disruption.

All five marked windings are initially admitted by their actual source
proofs. The event changes only support, so it preserves those instantaneous
windings and the host's acquired registration. The exact port identity
(5) then applies to the full reached state, whereas (8) is exact only on
the carrier. No finite post-attachment identity, capture, lifetime or
response follows from the event budget or carrier invariance alone.

## 7. Scope of the admitted mechanism

The organized constituent supplies two coupled boundary channels and
their reaction, with storage inherited from the same fine configuration.
On the compatible carrier its own form and phase suffice; with all
eight host coordinates and both conserved totals, they can instead be
reconstructed. Fine states additionally retain contact correlations.
The complete port excludes the old form-only auxiliary law under the
fixed host subtraction and clock, and excludes a smooth scalar port on
the stated independent two-coordinate donor domain.

The primitive preparation, positive loss, relative origin, uniform
mobility, unit capacities, matching interface and attachment occurrence
remain supplied premises. The source is not a new fundamental state
coordinate, and the result selects neither those premises nor a physical
reservoir. It establishes no new finite response or maintenance outcome
for the enlarged support. Any such claim requires its own admitted
source, horizon, work and transverse-error argument in the sole
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).

## 8. Shared implementation and verification boundary

The [shared detached owner](../../src/tnfr/physics/_organized_source_port.py)
builds the declared support, observes both complete fine rows and regional
accounts, evaluates the exact carrier restriction, and reconstructs donor
means from eight host coordinates plus two admitted totals. Its attachment
assessment re-admits primitive allowances and rebuilds the existing host
acquisition and isolated donor recovery before applying (17) and the new
contact bounds. An incoming report or cached verdict is not source evidence.

The [contract tests](../../tests/physics/test_organized_source_port.py)
compare the fine rows and storage with independently constructed graph
calculations, retain both clocks and reaction channels, exercise the
carrier and conserved-information reconstruction, and verify the
matched-mean fine counterexample and invalid scalar/vector admission.
Source and contact checks retain the complete preparation budgets and
actual reached errors. These tests establish implementation behavior;
the invariance, closure obstruction and source transfer arguments are
the conditional proofs above. No observer installs the support, chooses
an event or evaluates a post-attachment trajectory.

<a id="organized-source-emission-realization"></a>
## 9. All-duration obstruction to the fixed Emission endpoint

**Conditional theorem.** The unchanged attached source cannot realize the
nonclipped default Emission (AL) form increment at all nine constituents of
host child zero at any finite elapsed time. A bound on their mean already
excludes that necessary condition, allowing the donor and all other
coordinates to evolve. The result concerns this preparation, support,
complete law and fixed amplitude; it does not exclude other Emission gains
or identify a universal activation mechanism.

### Fixed comparison and evidence boundary

The analytical comparison retains the preceding actual acquired host and
recovered donor, all ten original `1e-10` error allowances, attachment and
continuous lifted law (1). Time zero below is the actual attachment state,
not an earlier nominal carrier or a reset after interaction. The observation is

\[
 X_0(\tau)=\frac19\sum_{j=0}^8 x_j(\tau).
\]

The fixed target is nine constituent Emission increments with the shared
default positive `AL_boost`, denoted \(b_{\rm AL}\), on the hard in-rail
branch. No aggregate operator inheritance is assumed. Its necessary form
condition is

\[
 x_j(T)=x_j(0)+b_{\rm AL}\quad(0\le j<9)
 \quad\Longrightarrow\quad X_0(T)-X_0(0)=b_{\rm AL}.
 \tag{23}
\]

For an instantaneous event, the direct glyph holds capacity, phase and
stored pressure. A finite-duration comparison instead refreshes pressure
with the same complete law at the AL endpoint. That refreshed pressure
need not equal the pre-event stored value. Full realization additionally
requires the appropriate phase, capacity, grammar and lifecycle conditions;
the obstruction below relaxes all such conditions and excludes even (23).
It therefore does not manufacture failure by freezing the donor or by
requiring fresh constitutive pressure to behave like a stored event value.

The horizon is every elapsed \(\tau\ge0\); the positive original-clock
rescaling gives the same reachability conclusion. The calculation uses
exact rational bounds, the shared 128-bit outward trigonometric intervals,
and the exact represented default boost. There is no selected response
time, amplitude search, gain change, numerical trajectory or frozen-producer
replay. Source bounds are rebuilt from admitted primitive allowances.

### A gap for the actual fine support

The 45-node support is the Cartesian product of the unit C9 and the
five-child graph with Laplacian \(K\) in (7):

\[
 L=K\otimes I_9+I_5\otimes L_{C9}.
\]

The child graph has diameter three. For any nonconstant mean-zero real
five-vector \(v\), let \(r=\max v-\min v\). The elementary variance
bound and a shortest path between its extremal nodes give

\[
 \|v\|_2^2\le\frac54r^2,\qquad
 r^2\le3\sum_{\{a,c\}\in\mathcal E_C}(v_a-v_c)^2.
\]

Thus its positive spectral gap is at least \(4/15\). The variance
inequality follows from
\((v_i-\min v)(\max v-v_i)\ge0\), summation and
\(-\min v\max v\le r^2/4\); the path inequality is Cauchy--Schwarz.
For C9 the smallest positive eigenvalue is
\(2-2\cos(2\pi/9)>1/3\), certified with the shared outward cosine
interval. The product eigenvalues are sums, so on the full mean-zero
fine space

\[
 x^TLx\ge\lambda_*\|x\|_2^2,\qquad \lambda_*=4/15.
 \tag{24}
\]

This includes all internal fine modes, not only the ten-coordinate
carrier. The actual degrees are nine copies of five, 27 of four and nine
of three. Therefore the complete sine current obeys

\[
 \mathbf1^TS(\theta)=0,\qquad
 \|S(\theta)\|_2^2\le\sum_i d_i^2=738<28^2
 \tag{25}
\]

for every phase configuration. No short-arc or retained-winding assumption
is needed for this bound. The separate capture theorem remains valid on
its own premises but is not used to suppress possible transients here.

### Retaining the actual preparation in an all-time envelope

Write \(\bar\gamma\) and \(\bar\eta\) for the outward upper bounds on
\(\gamma\) and \(\gamma^2\). Rebuild the host's gap bounds
\(\bar B,\bar\Delta\), form error \(R\), and donor radius \(q\) from
the original preparation. Each host reference form has absolute value
at most \(\bar\gamma(1+2\bar\eta)(\bar B+\bar\Delta)/2\).
There are 36 host coordinates, so the full initial form satisfies

\[
 \|x(0)\|_2\le R_0
 :=3\bar\gamma(1+2\bar\eta)(\bar B+\bar\Delta)+R+q
 <\frac1{50000}.
 \tag{26}
\]

This uses \(\bar B<1/64\), \(\bar\Delta<1/640\),
\(R<10^{-60}\), \(q=10^{-70}\), \(\bar\gamma<1/3000\), and the
rebuilt outward square bound \(\bar\eta<10^{-7}\).
It does not replace the reached forms by zero or discard independent
source errors. Exact zero total form is retained from both preparations
and is conserved by (1).

Variation of constants for the full form row gives

\[
 x(\tau)=e^{-L\tau/4}x(0)+\frac\gamma4
   \int_0^\tau e^{-L(\tau-s)/4}S(\theta(s))\,ds.
\]

Both terms are in the mean-zero space. Using (24)--(26), set
\(A=28\bar\gamma/\lambda_*<7/200\) to obtain

\[
 \|x(\tau)\|_2
 \le e^{-\tau/15}R_0+(1-e^{-\tau/15})A
 \le M:=\max\{R_0,A\}<\frac7{200}.
 \tag{27}
\]

The phase evolution and donor reaction remain those of the coupled law;
only their universally bounded sine contribution was relaxed. This is a
uniform transient estimate, which the separate asymptotic convergence
\(x\to0\) would not by itself supply. The quarter row factor sets the
decay rate and cancels from the steady forcing allowance. A clock change
must transform both rows, but cannot increase this all-duration envelope.

The child-mean observation has Euclidean norm \(1/3\). Hence for every
elapsed time, including all possible endpoint choices,

\[
 |X_0(\tau)-X_0(0)|
 \le\frac{M+R_0}{3}
 <\frac{1751}{150000}<\frac1{80}
 <\frac1{16}<b_{\rm AL}.
 \tag{28}
\]

The last inequality is checked against the resolved represented default,
approximately \(0.07957747\); it is not inferred from an operator name or
a supposed universal constant. The initial target coordinates and their
default AL proposals are well inside the hard rails because
\(|x_j(0)|\le R_0\) and \(R_0+b_{\rm AL}<1\). Clipping cannot remove
the specified increment in this comparison. Thus (23) fails for every
finite duration, proving the stated obstruction. A nonzero instantaneous
jump also cannot be the continuous observation of this smooth unforced
flow under a regular clock; finite-duration reachability is the stronger
substantive comparison tested here.

### Implementation, controls and scope

The existing [detached source owner](../../src/tnfr/physics/_organized_source_port.py)
reuses primitive attachment admission and derives the product gap, form
envelope and fixed default-Emission comparison. Candidate bounds remain
distinct from an admitted-source verdict. Neither an input report nor its
cached status substitutes for rebuilding the consumed preparation.

The [realization controls](../../tests/physics/test_organized_source_emission.py)
independently reconstruct support, spectra and degree bounds; check the
source-inclusive envelope, scalar comparison and mean observation; and
exercise default-operator, invalid-domain and unavailable-source boundaries.
The existing source-port and capture controls retain their earlier evidence.
These are detached analytical assessments and static controls, not executed
continuous responses or proof that a runtime word occurred.

The conclusion permits the already-proved source interaction, identity
maintenance and joined registration. It excludes one stronger claim: that
this unchanged source supplies the specified default Emission endpoint.
It establishes no necessary minimum Emission amplitude, no universal
failure of smaller gains, and no rejection of canonical operator semantics.
A bound that allows another endpoint would still not prove its realization,
grammar admission or occurrence. Native saturation belongs to a different
complete law and is not a premise of this proof.

<a id="organized-source-open-domain-emission"></a>
## 10. No fixed-duration constant Emission map on an open leaf domain

**Conditional theorem.** For every fixed finite duration `T>0` and every
fixed positive gain `b`, the unchanged 45-node autonomous law cannot produce
the nonclipped Emission (AL) increment `b` at all nine child-zero constituents
on any nonempty open subset of its complete conserved leaf. Even their
necessary mean increment cannot hold on such a subset. Donor reaction and
all collateral endpoint coordinates remain unrestricted in this obstruction.

This is a reusable-map question. It differs from the preceding bound on one
acquired family and one default gain: no small-source allowance, default
amplitude, spectral gap estimate or capture conclusion is needed here.
It also has a different limitation: it excludes an open full-state domain,
not every individual endpoint or every constrained preparation family.

### Fixed law, observation and complete domain

Keep the actual support, quarter mobility, held unit capacities and both
rows of (1), with `gamma=1/(1023*pi)` and elapsed structural clock `tau`.
There are no new inputs, events, controllers or support changes. Write
`theta[c,j]=psi[j]+delta[c,j]` on continuous real lifts as before. The
conserved state domain is the affine leaf

\[
 \mathcal M=\{(x,\delta)\in\mathbb R^{45}\times\mathbb R^{45}:
          \mathbf1^Tx=0,\quad\mathbf1^T\delta=9/8\}.
 \tag{29}
\]

These are raw sums; the corresponding sums of five child means are zero
and `1/8`. Eliminating the last coordinate of each vector identifies
`M` with `R^88`, so it is connected. The observation remains the actual
linear child mean `X_0=sum_(j=0)^8 x_j/9`, with all other fine coordinates
retained in the flow. A mean observation does not assert mean-only closure.

The comparison asks for a nonempty relatively open `U` in this leaf on
which the same pair `(T,b)` satisfies

\[
 x_j(T;z)-x_j(0;z)=b\quad(0\le j<9)
 \quad\Longrightarrow\quad
 h_T(z):=X_0(\Phi_T(z))-X_0(z)=b .
 \tag{30}
\]

Strict in-rail and scalar admission conditions may restrict `U`, but must
leave it open in the full leaf for this theorem to apply. The continuous
law itself has no hard form rails or runtime grammar/history coordinates.
Same-law pressure refresh, grammar and lifecycle completion remain separate
from this necessary form condition. No aggregate AL inheritance is assumed.

### Analytic flow and the equilibrium obstruction

The field is linear in form and a finite sum of sines of affine phase
differences. It is real analytic on the entire lifted state. Its real
derivative is globally bounded: the actual support has maximum degree five,
so both `L` and the sine-current Jacobian have operator norm at most ten.
Consequently the full field is globally Lipschitz and has a complete real
flow; edge cancellation preserves (29).

For every fixed finite `T`, the flow is real analytic in its initial
state. One justification uses the local holomorphic extension of the
linear and sine rows: on a small complex neighborhood of a real orbit
segment, the Picard construction converges uniformly and gives holomorphic
dependence. Cover the compact real segment from zero to `T` by finitely
many such intervals and compose their local maps. Restricting back to the
real affine chart proves the claimed local analyticity at every initial
state in `M`. This argument requires no complete complex flow and no
analytic continuation in time to infinity.

The existing [joined equilibrium](ORGANIZED_SOURCE_CAPTURE.md#1-full-lifted-cell-stationary-target-and-storage)
belongs to this same leaf:

\[
 x^*=0,\qquad \delta^*_{c,j}=1/40.
 \tag{31}
\]

Internal cycle sine currents cancel pairwise, matching currents vanish,
and `L*x_star=0`, so both complete rows vanish. In particular
`h_T(z_star)=0` at every fixed finite duration. The
[analytic realization criterion](../STRUCTURAL_OPERATORS.md#analytic-fixed-duration-realization)
now applies to the linear observation `X_0`. If (30) held on any nonempty
open `U`, analyticity and connectedness would force `h_T=b` everywhere on
`M`, including (31), a contradiction. Thus the realization set has empty
interior for every such `(T,b)`.

The continuation is an identity in **initial state**, not a simulated path
through intervening preparations. No initial member of `U` is assumed to
reach the equilibrium; no invariance of `U` or of its winding chamber is
required. The smooth law is defined beyond those chambers, and it is that
law which extends `h_T`. The argument does not assert that AL is admissible
or executed outside `U`, including at the equilibrium. A constant positive
clock rescaling only changes the quantified duration; it cannot remove
the contradiction when both evolution rows transform together.

The same criterion's countable sampling corollary also excludes choosing
the completion time from any prescribed countable schedule, such as the
multiples of one sampling interval, for the same fixed gain on an open
leaf domain. Every fixed-time equality set is closed with empty interior;
their countable union cannot cover that domain. The selector may depend
arbitrarily on the initial state. This concerns observing the unchanged
continuous flow at selected times, not a native step that intervenes with
hybrid operators, adaptation or other distinct laws. It does not exclude
continuously state-dependent stopping times or approximate matches.

### Boundaries that permit different realization questions

The actual acquired-source family retains at most 80 independent original
preparation directions, not an open 88-dimensional leaf domain. The theorem
therefore does not extend the preceding default-gain exclusion to every
smaller gain on that family. It also does not classify which isolated or
lower-dimensional state sets can meet a fixed endpoint.

The following auxiliary countermodels clarify the hypotheses; none is a
new TNFR law, source or installed operator mechanism:

| Changed hypothesis | Exact comparison |
| --- | --- |
| Constrained domain | The analytic oscillator `u_dot=v, v_dot=-u` has quarter-period map `(u,v)->(v,-u)`. On the line `v=u+b`, its `u` increment is `b`; this line has no interior in the two-dimensional state space. |
| Smoothness without analyticity | Put `r(s)=exp(-1/s)` for `s>0` and zero otherwise, and `sigma(s)=r(s)/(r(s)+r(1-s))`. The smooth field `x_dot=b*sigma(x)` has an equilibrium at zero and equals `b` on `x>=1`. At fixed duration one it translates every `x` in `(2,3)` by `b`. Its flat transition prevents real analyticity. |
| State-dependent duration | The analytic field `x_dot=x` has an equilibrium at zero, but for `x>0`, `T(x)=log(1+b/x)>0` gives `x(T(x))=x+b`. That is a stopping map, not one fixed-duration flow map. |
| Zero increment | A full-period oscillator map is the identity, compatible with an open domain and an equilibrium; the criterion requires `b!=0`. |

Even in the actual conserved leaf, a hypothetical nine-target form increase
with an equal opposite change at the nine donor slots preserves total form.
Thus conservation alone would not exclude the requested increment. It is
the analytic fixed-duration realization that fails.

The [static map controls](../../tests/physics/test_organized_source_emission_map.py)
check the complete leaf chart and compensated target, then the stated
countermodels with a fixed rational gain and closed-form maps. They reuse
the existing equilibrium/law owners and block native execution, numerical
flows and frozen producers. Finite algebraic checks are not a numerical
proof of the analytic identity theorem. No new runtime API or boolean
registry substitutes for the proof above.

State-dependent stopping, restricted actuator preparation, nonanalytic
observation and approximate realization remain distinct questions. Each
requires its own state, clock, admission and occurrence argument. This
result does not choose among them or change the declared canonical event
semantics. In particular it does not imply that interacting NFRs cannot
have reusable behavior under other justified realization contracts.

<a id="organized-source-regular-stopping"></a>
## 11. Regular stopping cannot realize Emission on an open leaf domain

**Conditional theorem.** The unchanged law (1) cannot realize a regular
near-identity Emission (AL) family on any nonempty relatively open subset
of the complete conserved leaf (29), even when its duration depends on
the initial state and gain. Preserving the nine target phases imposes nine
independent form constraints before a duration can be chosen. The exact
first-order compatibility locus is nonempty, but membership alone does
not realize the operator: the explicit state below fails at second order.

### Ideal primary action and complete comparison

Retain all 90 fine coordinates, the same 90 unit edges, held unit capacities,
quarter factor, positive `gamma` and clock `tau`. Let `P` select child zero's
nine coordinates. For every `z` in the proposed domain and all sufficiently
small positive gains `b`, the endpoint requirements are

\[
 P[x(T(z,b);z)-x]=b\mathbf1_9,\qquad
 P[\theta(T(z,b);z)-\theta]=0,
 \quad T(z,0)=0,\quad T(z,b)>0.                 \tag{32}
\]

Require C1 dependence on gain at zero, with a finite derivative. Circular
phase equality is equivalent to the displayed lifted equality locally:
continuity and `T->0` exclude a nonzero `2*pi` turn. The domain may impose
strict in-rail conditions but must remain open in the full 88-dimensional
leaf. Donor and all other endpoints may react freely. Pressure is refreshed
by the same law at the endpoint, not held equal to the pre-event value.

These are necessary primary-channel conditions for the ideal nonclipped
Emission map. The formal `b=0` identity is not a zero-gain AL invocation.
The implementation admits positive represented gains and materializes
bounded floating-point proposals; arbitrarily small exact increments are
not a claim about binary64 execution. Grammar, history, activation metadata
and lifecycle completion have separate obligations. Capacities are already
held by this comparison law; no auxiliary controller or phase clamp is added.

### Exact first-order locus and open-domain obstruction

Write `a(z)=partial_b T(z,0)`. The
[regular stopping criterion](../STRUCTURAL_OPERATORS.md#regular-stopping-realization)
gives `a>0` and, using both actual rows,

\[
 a\,P[-Lx+\gamma S(\theta)]/4=\mathbf1_9,
 \qquad a\gamma\,PLx/4=0 .                    \tag{33}
\]

Consequently the **exact first-order compatibility locus** is

\[
 \mathcal C=\{(x,\delta)\in\mathcal M:
       PLx=0,\quad PS(\psi+\delta)=s\mathbf1_9
       \text{ for some }s>0\},\qquad
 a=\frac4{\gamma s}.                           \tag{34}
\]

Conversely, every point in this locus supplies the displayed positive `a`
and satisfies (33). This equivalence concerns first-order equations only,
not existence of a stopping family satisfying (32). It retains individual
target currents, not merely their average. No global independence or
codimension is claimed for the nonlinear current-equality constraints.
Intersect this algebraic locus with the declared strict in-rail domain
when comparing an admitted Emission action.

The nine linear form constraints are independent even after enforcing
zero total form. Let `e_i` be the fine coordinate vectors and put

\[
 v_j=e_{9+j}-e_{27+j},\qquad PLv_j=-e_j
       \quad(0\le j<9).                       \tag{35}
\]

Each `v_j` has zero sum. Child zero is linked to child one at the matching
slot and has no link to child three; its internal edges meet neither
coordinate in `v_j`. This proves (35) directly from the actual support.
Thus `PL` has rank nine on the 44-dimensional zero-sum form space; its
kernel has dimension 35. Including the 44 phase-offset directions gives
a proper 79-dimensional affine constraint set inside the 88-dimensional
leaf. The locus (34) lies within it and has no relatively open full-leaf
domain. This proves the obstruction without a trajectory, source-bound
estimate or equilibrium argument.

The previously acquired family has at most 80 original preparation
directions; neither dimension count identifies its nonlinear image or
which of its members satisfy (34). This gate does not replace those
preparations by a convenient constrained state.

### A tangent-admitted state still fails to preserve phase

Consider the following exact static comparison state on the same leaf:

\[
 x=0,\qquad \delta_{c,j}=0\ (c<4),\qquad
 \delta_{4,j}=1/8.                            \tag{36}
\]

It is a supplied carrier state, not the earlier acquired source or a new
formation result. Put `q=sin(1/8)>0`, and let `w` be one at the nine target
slots, minus one at the nine donor slots and zero elsewhere. Internal
cycle currents cancel and only the donor interface contributes, so

\[
 S=q w,\qquad PS=q\mathbf1_9,\qquad PLw=4\mathbf1_9.
                                                        \tag{37}
\]

The target form velocity is `gamma*q/4` and every phase velocity is zero.
This admits the unique tangent duration derivative `a=4/(gamma*q)`.
However, differentiating the complete phase row gives

\[
 P\theta''(0)=\frac\gamma{16}PL[-Lx+\gamma S]
             =\frac{\gamma^2q}{4}\mathbf1_9>0.           \tag{38}
\]

For any regular candidate duration with that derivative, the local phase
displacement is therefore

\[
 P[\theta(T(z,b);z)-\theta]
      =\frac{2}{q}b^2\mathbf1_9+o(b^2),                 \tag{39}
\]

which contradicts (32) for sufficiently small positive gains. No second
gain derivative of `T` was assumed: `T=a*b+o(b)` suffices. More generally,
any exact candidate in (34) must also satisfy
`PL[-Lx+gamma*S]=0`; that additional constraint has not been classified as
a sufficient condition. The uniform carrier makes the target form rates
equal, but does not supply an independent means of holding their phases.

### Evidence and remaining scope

The [static stopping controls](../../tests/physics/test_organized_source_emission_stopping.py)
use the shared complete-state port observer and exact support multiplication
to check the rank witness, currents and nonzero phase acceleration. They
also retain a regular auxiliary stopping success and a singular boundary:
`x'=x, phi'=0` permits `T=log(1+b/x)` on `x>0`; the analytic system
`x'=2v, v'=1, phi'=0`, from `x=v=0`, permits `T=sqrt(b)` while the hidden
coordinate reacts. The latter has no finite derivative at zero. Neither
auxiliary field is installed as a TNFR law. Frozen producers, native calls
and numerical flow evaluation remain blocked in these controls.

The result closes the stated regular open-domain gate and rejects one
explicit constrained candidate beyond first order. It does not classify
all constrained preparations, singular durations, isolated finite-gain
hits, long excursions or approximate maps. In particular it preserves the
source's earlier interaction and collective-identity results. The failure
belongs to this exact primary-action realization under this supplied
complete law; it neither changes canonical Emission semantics nor excludes
other justified TNFR laws. Solving an endpoint equation would still supply
no occurrence rule, grammatical admission or physical identification.

<a id="organized-source-carrier-persistence"></a>
## 12. Local classification of phase-stationary carrier motion

**Conditional theorem.** Near the joined equilibrium, every trajectory of
the exact carrier law (8) that keeps child zero's phase constant on a time
interval belongs to the two-dimensional antisymmetric family defined below.
Its target form is also constant at zero. Other children can move, but that
motion cannot realize a positive Emission (AL) increment at child zero.
This classifies the nearby constrained states, rather than testing further
isolated candidates from the first-order locus (34).

### Fixed leaf and an exact invariant family

Keep the actual five-child graph, both rows (8), held unit capacities,
`gamma=1/(1023*pi)` and clock `tau`. The ten carrier coordinates lie on

\[
 \sum_cX_c=0,\qquad \sum_c\Phi_c=1/8,\qquad m=1/40.
                                                        \tag{40}
\]

This is an eight-dimensional affine leaf. Every point lifts to the fine
state `x[c,j]=X_c`, `theta[c,j]=psi[j]+Phi_c`; its invariance was proved in
Section 3. The independently acquired fine source is not projected onto
this family. The equilibrium is `X=0`, `Phi=m*1_5`.

Consider the embedded two-dimensional family

\[
 X=(0,u,-u,0,0),\qquad
 \Phi=(m,m+a,m-a,m,m).                         \tag{41}
\]

On the actual support, `KX=(0,2u,-2u,0,0)` and
`s(Phi)=(0,-2*sin(a),2*sin(a),0,0)`. Substitution into both rows gives

\[
 u'=-\tfrac12[u+\gamma\sin a],\qquad
 a'=\tfrac\gamma2u .                          \tag{42}
\]

All remaining coordinates in (41) have zero derivative, proving invariance
by uniqueness. In particular `X0=0` and `Phi0=m` at every time, although
`u` or `a` can change. This is internal motion under the supplied law, not
formation of a new organization or a selected operator occurrence.

The same cancellation is visible in the inherited storage (10). After
subtracting its constant internal-cycle value,

\[
 H_{10}-45(1-\cos\alpha)=18u^2+36(1-\cos a),\qquad
 H_{10}'=-18u^2.                              \tag{43}
\]

Thus the fixed target can coexist with changing, dissipative internal
state. The state and net rows of child zero do not observe this motion;
its individually marked interfaces to children one and two generally do.
Their opposing currents cancel in the net row. This is not invisibility
to a complete boundary observation or a closed target-only state model.

### Finite rank certificate at the equilibrium

Let `f` be the actual analytic carrier field on (40), `h=Phi0`, and freeze
the six-output map

\[
 G=(L_fh,L_f^2h,\ldots,L_f^6h),\qquad L_fg=Dg\,f.
                                                        \tag{44}
\]

At the equilibrium `z_star`, `f=0`. With `A=Df(z_star)` and `c=dh`,
induction gives `d(L_f^j h)(z_star)=c*A^j`: the differentiated higher-order
terms multiply `f(z_star)` and vanish. Thus the Jacobian of the **nonlinear**
map (44) can be computed from the linearization, without substituting a
linear law for the nonlinear trajectory or expanding a nonlinear Lie tower.

Use symmetric coordinates `(X0,Xs,X3,Y0,Ys,Y3)`, where
`Xs=(X1+X2)/2`, `Yc=Phi_c-m`, `Ys=(Y1+Y2)/2`, and antisymmetric coordinates
`u=(X1-X2)/2`, `a=(Y1-Y2)/2`. The totals determine
`X4=-X0-2*Xs-X3` and `Y4=-Y0-2*Ys-Y3`. At equilibrium the symmetric block is

\[
 B=\begin{pmatrix}4&0&1\\-1&2&-1\\0&-2&2\end{pmatrix},\qquad
 A_s=\frac14\begin{pmatrix}-B&-\gamma B\\\gamma B&0\end{pmatrix},
 \qquad c_s=(0,0,0,1,0,0).                    \tag{45}
\]

The matrix `B` is the actual `K` restricted to the sum-zero symmetric
subspace. The two antisymmetric columns of `DG` vanish because (41) has
constant output. Direct polynomial elimination of the six symmetric rows
gives the exact determinant

\[
 \det\begin{pmatrix}c_sA_s\\c_sA_s^2\\\vdots\\c_sA_s^6\end{pmatrix}
 =\frac{16000\gamma^{15}}{4^{21}}P(\gamma^2),\qquad
 P(\eta)=4489\eta^3-2453\eta^2+435\eta-25.     \tag{46}
\]

Here `21=1+...+6` retains the quarter factor in every row. As an independent
coordinate check, eliminating child four in the first-four-child chart
and choosing columns `(X0,X1,X3,Y0,Y1,Y3)` gives the same polynomial with
prefactor `4000` instead of `16000`; the symmetric chart doubles two columns.
This is exact algebra, not a numerical singular-value threshold.

For the actual parameter, `0<gamma<1/3000`, hence `0<eta<10^-6`. Dropping
the negative quadratic term bounds

\[
 P(\eta)<-25+435\,10^{-6}+4489\,10^{-18}<-24. \tag{47}
\]

The determinant is therefore nonzero. The six-output Jacobian has rank
six on the eight-dimensional leaf, with precisely the two antisymmetric
directions in its kernel. The small determinant does not indicate a zero
rank, and no claim is made for arbitrary substituted parameters or for
numerically stable recovery from noisy phase derivatives.

### Nonlinear local classification and realization consequence

Write the leaf coordinates as `(y,u,a)`, with `y` the six symmetric
coordinates in (45). The invariant family (41) is exactly `y=0`, and
`G(0,u,a)=0` for all its states. By (46), `D_yG` is invertible at the
equilibrium. The implicit-function theorem supplies a neighborhood in
which the zero set is a unique graph `y=chi(u,a)`. The known solutions
`y=0` force `chi=0`. Thus, in some open neighborhood `U` of the equilibrium,

\[
 G(z)=0\quad\Longleftrightarrow\quad z\text{ belongs to (41)}.
                                                        \tag{48}
\]

An interval of constant target phase makes all six derivatives vanish at
its initial state. For an initial state in `U`, (48) therefore puts the
whole trajectory on the invariant family, with target form identically
zero. Conversely every member of that family has constant target phase.
This proves the stated local classification for the nonlinear law.
The neighborhood is existential; no quantitative radius or inclusion of
the earlier prepared-source bounds has been certified.

There is also a small-time endpoint consequence beyond differentiable
stopping. At a fixed initial state in `U`, the target phase is analytic in
time. If distinct positive durations tending to zero preserve that phase,
its difference from the initial phase has accumulating zeros and vanishes
on an interval. Circular equality reduces to equality of the continuous
lift near time zero. Equation (48) again forces the target form to remain
zero. Consequently no positive-gain Emission family with completion times
tending to zero is possible there, even with singular gain dependence.
A single finite phase return, distant initial states and arbitrary fine
perturbations are outside this local conclusion.

### Static verification and scientific boundary

The [carrier persistence controls](../../tests/physics/test_organized_source_carrier_persistence.py)
derive the matrices from the actual support, verify the exact symbolic
family and determinant, and compare fine and collective observer rows
with independent fixed controls. They keep the two antisymmetric kernel
directions, the sign-certified polynomial and all clock factors. The
proof of a neighborhood comes from the implicit-function theorem, not
from sampled states, numerical rank or a simulated trajectory.

The complete law and registered Emission semantics are unchanged. The
result identifies a persistent cancellation mechanism and excludes a
specific primary action locally on the carrier; it does not derive an
event selector, a grammar word, a universal TNFR restriction or a physical
particle. Existing formed-source evidence and full-state retention remain
with their own owners and premises.

<a id="organized-source-fine-response"></a>
## 13. Fine internal organization creates a delayed target response

**Conditional result.** Two declared full states have the same nine target
forms, primitive phases and both net evolution rows, but different target
accelerations under the unchanged nonlinear law. Their target means remain
matched through the tested second order. The actual full linearization
keeps this perturbation invisible to the target at every time, whereas the
nonlinear response separates the two fixed fine slots at second order.
Thus neither the carrier nor a matched target state and initial velocity
retains all information needed to predict this response.

### Frozen preparation and observation

The comparison retains all 45 nodes, 90 unit edges, held unit capacities,
quarter mobility, `gamma=1/(1023*pi)` and structural clock `tau`. There are
no inputs, events or support changes. Both initial forms are zero. With
`m=1/40`, `epsilon=1/100` and `v=(1,-1,0,0,0,0,0,0,0)`, set `a_j=epsilon*v_j`.
The control has every offset equal to `m`; the second state has

\[
 \delta_{1,j}=m+a_j,\qquad\delta_{2,j}=m-a_j,\qquad
 \delta_{c,j}=m\quad(c=0,3,4),\qquad x=0.       \tag{49}
\]

These are supplied static states, not the previously acquired source.
The zero sum of `v` preserves every child offset mean and the full totals
`sum x=0`, `sum delta=9/8`. The lifted internal cycle increments are
`alpha`, or `alpha +/- (a_(j+1)-a_j)`, where `alpha=2*pi/9` and indices
are cyclic. Their perturbations have magnitude at most `2/100`, so all
increments lie strictly in `(0,pi/2)` and sum to `2*pi` on each child.
Every initial winding is therefore one. Matching-edge gaps have magnitude
at most `1/100`. Initial admission supplies no finite retention interval
or inclusion in the image of an acquired preparation.

The observations are the nine target forms and primitive phases, their
net rows, the slot-zero-minus-slot-one contrasts and the child means.
At the target, the child-one and child-two matching sine currents are
`sin(a_j)` and `-sin(a_j)`, and all internal currents cancel. Thus the
target current is zero, as in the control. Since all forms are zero,
every phase velocity is zero as well. The target state and both net rows
match exactly. Individually marked interface currents generally differ
and are outside this matched observation; they remain in the complete law.
The initial primitive phase contrast is `-alpha`, not zero. Its change
and the initial offset contrast are zero.

### Differentiating both actual rows

For any admitted fine state of this fixed model, write `q=Lx`,
`J(theta)=DS(theta)`. Differentiation, with primes in `tau`, gives

\[
 q'=Lx',\qquad
 S'_i=\sum_{k\sim i}\cos(\theta_k-\theta_i)
                         (\theta'_k-\theta'_i),\qquad
 x''=(-q'+\gamma S')/4,\quad\theta''=\gamma q'/4. \tag{50}
\]

The source reacts through these same rows. In (49), `theta'=0`, so
`S'=0`, `x'=gamma*S/4` and

\[
 x''=-\gamma LS/16,\qquad\theta''=\gamma^2LS/16. \tag{51}
\]

Define `d_j=a_(j+1)-a_j` and `r_j=cos(d_j)-cos(d_(j-1))`. The matching
currents of children one and two cancel in their sum. Their internal
currents instead satisfy the exact identity

\[
 S_{1,j}+S_{2,j}=2\sin\alpha\,r_j.             \tag{52}
\]

For example, their forward contributions sum to
`sin(alpha+d_j)+sin(alpha-d_j)=2*sin(alpha)*cos(d_j)`;
the backward contributions subtract the same expression at `j-1`.
Since target and donor currents vanish initially, the target component
of `LS` is the negative of (52). Consequently

\[
 x''_{0,j}=\frac{\gamma\sin\alpha}{8}r_j,
 \qquad\theta''_{0,j}=-\frac{\gamma^2\sin\alpha}{8}r_j.
                                                        \tag{53}
\]

This term is the even-in-profile part of the actual sine interaction
on the nonzero winding background. It introduces no additional force,
pressure law or event. Its cyclic sum vanishes, explaining why the mean
readout misses this fine response through second order.

For the frozen profile, every child's mean also agrees with equilibrium
through that order: internal currents telescope, `sum_j sin(a_j)=0`, and
the means of `LS` are `K` times the means of `S`. This stronger all-child
statement uses the particular opposed two-slot values, not merely a zero
arithmetic mean for an arbitrary internal profile.

For the frozen profile, let `D=cos(epsilon)-cos(2*epsilon)>0` and
`E=1-cos(epsilon)>0`. Direct substitution gives

\[
 r=(-D,D,E,0,0,0,0,0,-E),\qquad
 (x_{0,0}-x_{0,1})''=-\frac{\gamma\sin\alpha}{4}D<0,
 \quad(\theta_{0,0}-\theta_{0,1})''=
                      \frac{\gamma^2\sin\alpha}{4}D>0.  \tag{54}
\]

Taylor's theorem therefore gives a negative form contrast and a positive
change in primitive phase contrast at sufficiently small positive times;
their leading coefficients are one half the accelerations in (54).
This is a local existence-of-separation result. No numerical duration,
finite reading margin, observed trajectory or sustained response is certified.
The derivative budget ends at order two; zero mean derivatives alone do
not prove all-time constancy. The separate symmetry proof in
[section 14](#organized-source-reflection-state) supplies that result for
this preparation and its declared reflection sector.

There is a broader algebraic boundary within the same opposed-profile,
zero-form class. If every cyclic increment has `abs(d_j)<pi`, vanishing
of all target accelerations in (53) implies that all `cos(d_j)` are equal.
Cosine is injective in `abs(d_j)` on this domain, so all magnitudes equal
one value `d`. The nine signed increments sum to zero. An odd number of
terms each equal to `+d` or `-d` cannot have zero sum unless `d=0`.
Thus only a spatially constant opposed profile cancels all nine target
accelerations in this class. This corollary uses the same second-derivative
identity; it is not a new preparation scan or a classification of arbitrary
nonzero-form fine states.

### Why the actual tangent does not predict this response

At the joined equilibrium, the fine phase stiffness is

\[
 H=K\otimes I_9+\cos\alpha\,I_5\otimes L_{C9},\qquad
 A=\frac14\begin{pmatrix}-L&-\gamma H\\\gamma L&0\end{pmatrix}.
                                                        \tag{55}
\]

The internal cosine factor is essential: `H` is not the bare support
Laplacian `L`. Let `b=(0,1,-1,0,0)`; then `Kb=2*b`. Both `L` and `H`
preserve the fine subspace `b tensor R^9`. The initial perturbation in
(49) belongs to that subspace in phase and is zero in form. The complete
linearized evolution therefore stays in the same subspace in both
channels, and its target components vanish for every time. This is an
exact invariant-subspace argument, not a finite sample of a linear solver.
It does not exclude the nonlinear term (52), which is quadratic to leading
order in profile amplitude and has nonzero target acceleration.

### Shared implementation and canonical-state implications

The [source-port owner](../../src/tnfr/physics/_organized_source_port.py)
provides `_observe_organized_source_acceleration(form=..., phase_offsets=...)`.
It re-admits raw primitives through the complete first-row observer, then
differentiates every actual weighted edge as in (50). It accepts no cached
report. Both channel accelerations, their child means and the current and
gradient derivatives remain associated with the freshly rebuilt state.
Original and slow-clock accelerations use the squares of the first-rate
conversion factors. These are predicted instantaneous derivatives, not
Taylor coefficients, measured secants, Mutation evidence or finite-time
error certificates. Earlier first-row reports and frozen producers are unchanged.

The [fine-response controls](../../tests/physics/test_organized_source_fine_response.py)
check primitive admission, windings, exact cancellations, independent fine
edge derivatives, contrast signs and the actual tangent subspace. A separate
general-state control retains the nonzero cosine-current derivative in (50),
which the zero-form gate does not exercise. The earlier regular-stopping
control also consumes this shared acceleration owner. All remain detached:
no native step, numerical flow or frozen response producer is invoked.

For a complete NFR description, the useful mechanism is retention of the
constituent phase arrangement and the reacting interfaces. Matching net
rows is insufficient here. The existing
[nonlinear constituent-memory method](ORGANIZED_CONSTITUENT_MEMORY.md#organized-constituent-memory-result)
offers exact projection and elimination identities, but its `C9 x C5`
support, bounds and source certificates do not transfer to this `C9 x
(C4 plus leaf)` support. A new representation must reconstruct the actual
rows and retain discarded initialization or its derived memory. This
result does not itself provide a smaller closed state, an inherited
operator, grammatical admission, autonomous selection or physical identity.

<a id="organized-source-reflection-state"></a>
## 14. An exact nonlinear state with constant child means

**Conditional representation.** The signed-reflection sector containing
(49) admits forty independent real coordinates that reconstruct both
complete nonlinear rows. Each child's form mean remains zero and its
lifted phase/offset mean remains `m=1/40` for every time. Thus the means
have a closed, constant evolution on this sector, while they cannot
determine its fine internal response. The proof retains the same 45-node
support, unit capacities, quarter mobility, gamma and clocks as section 13.
It changes coordinates and restricts initial state; it supplies no new law.

### Declared real lifts and the exact symmetry

Let `r(j)=(1-j) mod 9`, with pairs `(0,1),(2,8),(3,7),(4,6)` and fixed
slot `5`. Let `P` reflect every child by `r`, `psi_j=alpha*(j-4)` and
`eta=delta-m`. The selected sector is

\[
 x=-Px,\qquad \eta=-P\eta,\qquad
 \theta=\psi+m\mathbf1+\eta.                         \tag{56}
\]

In particular, `x_(c,5)=eta_(c,5)=0`. This is a sector of the declared
continuous real offsets, not a choice of wrapped representatives on every
circular phase orbit. Define `b_j=1` at slots zero and one, and zero at
the other seven slots. Direct substitution gives

\[
 \psi_j+\psi_{r(j)}=2\alpha-2\pi b_j,\qquad
 \widetilde\theta_{c,j}=2m+2\alpha-\theta_{c,r(j)}-2\pi b_j.
                                                               \tag{57}
\]

The affine reflection of primitive phase therefore has common circular
origin `m+alpha`, with two nonzero lift branches. Omitting these full
turns would give a false real-lift identity. Equation (57) is exactly
`tilde delta=2m-P delta`; since `b_r(j)=b_j`, it is an involution.
The circular fixed-point equation also allows an antipodal choice at
slot five; (56) selects the component and lifts used here.

The actual support consists of internal cycles and matching slots, so
`P` is its weighted automorphism and `LP=PL`. Sine is odd and ignores
integer turns. Hence

\[
 L(-Px)=-PLx,\qquad S(\widetilde\theta)=-P S(\theta).
                                                               \tag{58}
\]

Both rows, including `theta'=gamma*Lx/4`, are equivariant under the same
affine map. The field is globally Lipschitz on the finite-dimensional real
lift: its form dependence is linear and its sine derivatives are bounded.
Existence and uniqueness of the complete flow therefore preserve the
fixed set (56) for every time. This transfers the
[signed-reflection proof method](ORGANIZED_CONTACT_DYNAMIC_CANCELLATION.md#organized-contact-dynamic-cancellation)
after rebuilding this support and lift; no coefficient, source enclosure
or maintenance certificate from that different model is transferred.

### Reconstruction and both reduced rows

For one child use representatives `(0,2,3,4)` and define

\[
 E_9q=(q_0,-q_0,q_1,q_2,q_3,0,-q_3,-q_2,-q_1)^T,
 \qquad E=I_5\otimes E_9,\qquad x=Eu,\quad\delta=m\mathbf1+Ea.
                                                               \tag{59}
\]

Here `u,a` each have twenty coordinates, ordered by child then representative.
Let `R` select those representatives. Then `RE=I_20`, `E^T E=2I_20`,
and `ER` is identity on the odd subspace. Restriction of phase coordinates
first subtracts `m`. Lift and restriction are inverse on (56); restriction
outside that sector is undefined. Averaging an arbitrary fine state into
this chart would discard information and is not this operation.

With the actual five-child Laplacian `K` from section 3, edge incidence gives

\[
 A_r=RLE=K\otimes I_4+I_5\otimes B,\qquad
 B=\begin{pmatrix}3&1&0&0\\1&2&-1&0\\0&-1&2&-1\\0&0&-1&2\end{pmatrix},
 \qquad LE=EA_r.                                         \tag{60}
\]

For a child's four phase deviations, its internal current on the
representatives is

\[
 J(a_c)=\begin{pmatrix}
 \sin(\alpha-2a_{c,0})-\sin(\alpha+a_{c,0}+a_{c,1})\\
 \sin(\alpha+a_{c,2}-a_{c,1})-\sin(\alpha+a_{c,0}+a_{c,1})\\
 \sin(\alpha+a_{c,3}-a_{c,2})-\sin(\alpha+a_{c,2}-a_{c,1})\\
 \sin(\alpha-a_{c,3})-\sin(\alpha+a_{c,3}-a_{c,2})
 \end{pmatrix}.                                          \tag{61}
\]

Add `sum_(d~c) sin(a_(d,k)-a_(c,k))` at every representative `k`, using
exactly the child matching edges, to obtain `j(a)`. Equations (58) and
(61) give `S(psi+m+Ea)=E j(a)`, including the zero fixed-slot current.
Consequently the autonomous reduced law is exactly

\[
 u'=\frac{-A_ru+\gamma j(a)}4,\qquad
 a'=\frac{\gamma A_ru}4.                                \tag{62}
\]

Lifting (62) reconstructs every fine form and phase row, not just the
selected representatives. This is a nonlinear intertwining on the whole
sector, not a tangent reduction or a finite trajectory fit. Both rows
retain the factors `1023/1024` for original time and `2/gamma^2` for slow
time. The positive off-diagonal entries and nonzero row sums of `B`
also matter: these twenty signed coordinates are not twenty ordinary
NFR nodes with an ordinary positive-conductance graph Laplacian.

### Storage, means and the retained response

Let `W(a)` be the original ninety-edge cosine potential evaluated at (59).
The inherited storage and balance are

\[
 H_r(u,a)=u^TA_ru+W(a),\qquad \nabla_a W=-2j(a),\qquad
 H_r'=-\frac12\|A_ru\|^2.                              \tag{63}
\]

The factor two follows from `E^T E=2I`, and (63) equals the original
`-||Lx||^2/4` exactly. It introduces neither a new reservoir nor an event
passivity premise. The cosine potential and all marked interfaces remain
those of the fine state.

Every column of `E_9` sums to zero, and the winding background `psi` has
mean zero. Invariance of (56) therefore proves all five child form means
zero and all five lifted phase and offset means `m` for all time. The
frozen profile (49) is represented by `a_(1,0)=epsilon`,
`a_(2,0)=-epsilon`, with every other coordinate and every form zero.
Its locally nonzero target contrasts (54) thus coexist with exactly
constant means. This is an exact failure of those means to determine the
fine observation, not a failure of their trivial closed mean evolution.

### Implementation boundary and remaining identity obligations

The [existing source-port owner](../../src/tnfr/physics/_organized_source_port.py)
provides `_lift_organized_source_reflection`,
`_restrict_organized_source_reflection` and
`_observe_organized_source_reflection`. All accept raw coordinates, admit
them through the shared represented-real reader and normalize before
equality checks or arithmetic. `phase_coordinates` receives `eta`; the
restriction argument `phase_offsets` receives `delta=m+eta`. Neither is
primitive phase. Restriction checks every pair and fixed slot exactly, rejecting
off-sector values rather than wrapping, recentering or averaging them.
For example, binary64 `0.025` is not exactly rational `1/40`; constructing
the offsets from admitted deviations avoids silently equating the two.

The observation lifts through the shared full-law reader and selects both
reduced rows in every clock. It retains the full fine observation for
support, storage, interface and provenance consumers. There is no duplicate
sine kernel, numerical integrator or new runtime execution path. Optional
acceleration still uses the raw lifted state and the section 13 owner.
The [representation controls](../../tests/physics/test_organized_source_reflection.py)
independently derive support, branch integers, signed incidence and both
rows; interval checks exercise the implementation but do not replace the
exact equivariance proof.

This is a sufficient state on a declared symmetry sector, not a minimum
dimension theorem or a projection of the acquired-source family. Its
unrestricted coordinates do not certify winding one or acute edges.
Indeed, symbolically taking every child's
`a=(5*alpha,3*alpha,2*alpha,alpha)` and `u=0` makes all primitive phases
equal to `m+alpha` modulo full turns: this is a stationary winding-zero
state in the same sector. It does not show that (49) reaches that state.
Winding retention, attraction from outside the sector and formation need
their own arguments. Continuous invariance also supplies no operator
history, grammar context, event descent or occurrence law. A registered
event must be admitted and its successor checked separately against the
same lifted-state and complete-law boundary.

<a id="organized-source-reflection-reception"></a>
## 15. Reception on the retained reflection state

### Fixed event question and numerical boundary

The question is whether one supplied atomic Reception (EN) stage on targets
`(0,1)` has a nontrivial admitted successor in the section 14 sector, and
how both rows of this same law change at that successor. The mathematical
domain keeps all forty coordinates and independently admitted runtime
context. The numerical control retains the section 13 delayed phase
offsets, supplies `x_0=1/64`, `x_1=-1/64` and zero other forms, and uses
the actual ordered 45-node support with its ninety unit edges. The old
zero-form profile alone would make this Reception numerically trivial.

Before executing the response, fix the following protocol. Nodes and edges
use the source-port owner's order, capacities are one, the structural clock
is `tau` with `_t=7`, the seed is `17`, and the common semantic kind is
`organized-source`. Use the resolved default `EN_mix`, symmetric hard rails
`[-1,1]`, strict operator preconditions and grammar rejection `raise`.
Every target has independently supplied Emission (AL)/Coherence (IL)
history with ordinal two, zero debt and prior coherence. This is supplied
context, not a claim that those events produced the current forms.
Stored `dEPI=0` is a diagnostic input, not measured temporal evidence.
The runtime phase encoding is binary64 materialization of
`alpha_midpoint*(j-4)+delta`; the exact real offsets are retained separately.
They are never reconstructed by subtracting approximate pi from runtime
phase. A reader must check the unchanged represented-phase association,
support and capacities before using those offsets for the complete-law
observation. Initial stored pressure is a binary64 projection of the fresh
`tau` form row at unit capacity, with its projection recorded separately.

Use the actual network stage with the step-zero context of the validated
word `EN -> IL -> SHA` under `core` admission. Only Reception is executed:
the declared suffix does not constitute a completed word or an executed
future handler. Source tracking remains enabled at its default distance
two. All forms are below the activity cut, so an empty source observation
is expected; it does not waive or replace live admission. Built-in operator
metrics are enabled to observe their pre-refresh boundary; no external
monitor or custom metrics callback is configured. The declared pressure-refresh callback
reuses this model's fine reader after the commit, checking held-state
association before writing projected fresh pressure.

The sole corresponding schedule control calls public Reception on zero
then one, with the same contextual boundary and refresh after each accepted
call. It deliberately reads the second neighborhood after the first write.
Additional invalid-input controls check strict rejection and transaction
rollback; they are not new source or parameter searches. The computation
budget is one finite stage and its serial control, exact rational endpoint
accounting and the existing rational transcendental enclosures. No numerical
trajectory, frozen source producer, acquired-source reconstruction or
physical identification belongs to this gate.

### Ideal primary action, same-law refresh and event work

Let `beta` be the resolved mixing fraction interpreted as a real number,
with the runtime admission `0 < beta <= 1`. An ideal evaluation of that
coefficient is distinct from the successive rounded runtime operations.
Write `q=Lx` and `e=e_0-e_1`. Each target has five bare neighbors, including
the other target, so its unclipped change from the common snapshot is
`-beta*q_i/5`. In the reflection sector `q_1=-q_0`; hence

\[
 x^+=x+d e,\qquad d=-\frac{\beta q_0}{5},\qquad
 \delta^+=\delta.                                      \tag{64}
\]

Only reduced form coordinate `u_(0,0)` changes. The reflected neighbor
sets have equal size, and a common mix and symmetric hard clipping are
odd under form sign reversal. Thus the primary action preserves the sector
whenever both targets are independently admitted. Inputs inside `[-1,1]`
give convex proposals inside those rails, so clipping is inactive on the
domain of (64). This argument does not prove universal strict-pressure or
grammar admission, nor symmetry of arbitrary kind/source/history records.
Those records are retained as contextual state, not compressed into the
forty physical coordinates.

For any actual form endpoints with phase, support and capacity held, let
`z=x^+-x`. The unchanged sine current gives

\[
 q^+=q+Lz,\qquad
 p_\tau^+-p_\tau=-\frac14Lz,\qquad
 \theta_\tau^{\prime+}-\theta_\tau'=
                         \frac\gamma4Lz.                \tag{65}
\]

Here `p_tau=(-q+gamma*S)/4` is the specified pressure law; at unit capacity
it equals the form row. This is fresh constitutive pressure in the declared
clock, not the stored field retained by the EN
jump. Both rows change. In particular, the unchanged-pressure result of
the [separate conservative Reception model](SINE_COLLECTIVE_RECEPTION.md#4-fresh-rows-means-and-pressure-provenance)
does not transfer to this diffusive form law. Untargeted neighbors can
change both rates without any form or phase jump of their own. Rate
differences transform by the first-order clock factors from section 14;
the event itself has no inferred duration.

The exact storage jump, independently of how the endpoints were obtained, is

\[
 \Delta H=q^Tz+\frac12 z^TLz.                           \tag{66}
\]

The phase potential cancels because the same lift is retained. For this
pair, `e^T L e=12`: the shared edge contributes the cross term missing
from a sum of two isolated-target calculations. Inserting (64) yields

\[
 \Delta H=2q_0d+6d^2
    =-\frac{2\beta}{5}\left(1-\frac{3\beta}{5}\right)q_0^2
    \leq0.                                              \tag{67}
\]

It is strictly negative when `q_0` is nonzero. This theorem applies to the
admitted, unclipped, unit-weight pair, rather than arbitrary weighted
Reception or out-of-rail states. The jump account is separate from the
continuous loss `H'=-||Lx||^2/4`; no dissipated quantity is assigned to an
undeclared event reservoir.

For the frozen forms, set `h=1/64` and `r=1-6*beta/5`. The atomic ideal
endpoint is `(r*h,-r*h)`. The corresponding ideal serial zero-then-one
endpoint is instead

\[
 (x_0^+,x_1^+)_{\rm serial}
    =\left(rh,-rh-\frac{6\beta^2h}{25}\right).          \tag{68}
\]

The nonzero pair sum leaves the exact sector; its child-zero form mean
is `-2*beta^2*h/75`. This is a different read schedule, not a different
Reception formula. Fine-state continuation remains defined outside the
sector, while a strict chart restriction must reject it. Neither schedule
has a universal form-total conservation contract.

Equations (56), (64) and (67) also give a conditional hybrid composition
result without another response experiment: any finite sequence of such
independently admitted ideal atomic stages interleaved with this exact continuous
flow preserves the physical reflection sector. If every event's inputs
remain inside the rails, storage is nonincreasing across events and along
the flows. Stage times, grammar, consumed metadata and any future admission
remain explicit premises. This does not establish a completed grammatical
word, infinite-event execution, autonomous occurrence or winding retention
along the intervening flow.

### Actual represented successor and shared endpoint accounting

The [fixed runtime controls](../../tests/operators/test_organized_source_reception.py)
execute the existing public stage with its immutable all-target snapshot.
The admitted atomic endpoint has exactly opposite represented forms, with
positive amplitude strictly below `1/64`; strict restriction to the exact
reflection chart succeeds. The amplitude differs from the ideal real
formula: the two-stage binary64 mean and blend must not be replaced by an
affine matrix product. Both actual endpoints agree with their respective
ideal formulas within the declared conservative `1e-17` absolute allowance.
The fixed normal-range operations admit a roundoff budget below
`5*2^-53/64 < 1e-17`, including propagation of the first serial error;
the allowance is not fitted to the evaluated discrepancy.
This is a finite represented control, not universal floating-point
equivariance. All represented forms are re-admitted before endpoint work
or fresh rates are calculated.

The serial public control has the predicted negative pair sum and fails
the exact chart check. The full fine observer still admits it. Its actual
work is also negative in this control; storage decrease alone therefore
does not imply preservation of the reflection sector. Both controls retain
the declared phases, capacities, support and structural time. The immediate
phase windings are unchanged at Reception; later retention is a separate
flow question.

The stage records both ordered five-neighbor inputs, their actual means,
unchanged common kinds and empty tracked-source records. Each target's
history advances from `AL -> IL` to `AL -> IL -> EN`, with ordinal two to
three, zero debt and prior coherence retained. Untargeted histories and
the supplied zero stored derivatives are unchanged. The stage's sealed
Reception observations describe its accepted endpoint; they do not certify
future stability. Built-in metrics read the old stored pressure before
the explicit callback writes fresh projected pressure. The serial second
call instead consumes the pressure refreshed after its first event.
Second-target form, pressure and grammar rejections restore the complete
atomic graph-owned state. A rejected second public call leaves its already
accepted first event in place; separate calls are not one transaction.

The [shared source-port owner](../../src/tnfr/physics/_organized_source_port.py)
now supplies `_observe_organized_source_form_jump(form_before=...,
form_after=..., phase_offsets=...)`. It admits all primitive endpoint vectors
before any fine-law observation, rebuilds both states, and retains the exact
form displacement, gradient change and storage jump (66). Correlated rate
differences use the unchanged sine current in (65), with both channels in
all three clocks. This avoids subtracting independent phase-current bounds
for a quantity that cancels exactly. The same calculation accepts a serial
endpoint outside the reflection sector; it does not silently project it.

The helper takes one declared held phase lift and the fixed support and
unit-capacity model. It does not authenticate an event or inspect a graph.
An execution consumer must first verify those held-field premises and the
represented-phase association, as the fixed control does. Stored pressure,
operator names, cached bounds and grammar labels cannot replace these
checks. No new Reception implementation or pressure law was introduced.

<a id="organized-source-hybrid-retention"></a>
## 16. Joint winding and form-rail maintenance

**Conditional result.** One explicit domain containing the section 15
preparation preserves all five winding-one cycles and the form limits needed
by the ideal paired Reception action. It is invariant under the unchanged
continuous law and any finite list of independently admitted ideal atomic
Reception stages on `(0,1)`. This joins the continuous and event arguments
without assuming in advance that future forms stay inside the rails.
The statement is about maintenance of a supplied organization; stage
occurrence, full grammatical completion and formation remain separate.

### The domain is an intersection, not an energy test alone

Keep the exact reflection sector of section 14, the full support, held
unit capacities, declared clocks and the fixed continuous phase lifts. Let
`A` be the acute cell of the
[same-support boundary proof](ORGANIZED_SOURCE_CAPTURE.md#1-full-lifted-cell-stationary-target-and-storage):
every internal angle is `alpha+delta_(c,j+1)-delta_(c,j)`, including the
fixed closing-edge turn, and every matching angle is the corresponding
offset difference. All ninety angles must lie strictly in `(-pi/2,pi/2)`.
The internal sums then remain `2*pi`, so each marked cycle has winding one.
Define

\[
 H_*=45(1-\cos\alpha),\qquad E=H-H_*,\qquad
 \mathcal D=\{(x,\delta)\text{ in the reflection sector and }\mathcal A:
                         E<1/100\}.                    \tag{69}
\]

On the closed acute cell, Jensen's inequality on each nine-edge cycle
gives phase storage at least `H_*`. Matching storage is nonnegative.
Writing `F=x^T Lx/2`, this proves `0<=F<=E` on that cell. The
[full boundary bound](ORGANIZED_SOURCE_CAPTURE.md#2-every-acute-boundary-face-has-a-positive-excess)
requires `E>=beta_face>6/25` at every phase face. These geometric facts
use the actual ninety-edge support. They do not require replaying or
importing the different acquired source from that earlier capture study.

Every part of (69) is necessary for this certificate. In particular, the
winding-zero stationary state from section 14 has `H-H_*<0` outside the
declared acute cell. Small storage excess alone does not certify winding
identity. Nor does a failed sufficient check demonstrate identity loss.

### A form bound before invoking event passivity

The reflection sector fixes `x_(c,5)=0`. Every other slot in its own C9
connects to that slot along a path of at most four unit edges. Telescoping
and Cauchy's inequality give

\[
 |x_{c,j}|^2
 \leq4\sum_{\text{path}}(x_a-x_b)^2
 \leq8F\leq8E<\frac{2}{25}<\left(\frac3{10}\right)^2.
                                                               \tag{70}
\]

Thus every form in `D` has `abs(x)<3/10<1`. This bound follows from the
state domain, before applying Reception. All inputs to the pair's convex
mixture therefore lie strictly inside the hard rails `[-1,1]`; its ideal
proposal is unclipped. The event-work theorem (67) is now applicable
without assuming the very rail retention it is meant to help establish.

### Entry of the fixed preparation

For `h=1/64`, `epsilon=1/100` and the section 15 phase profile, form
storage is `6*h^2`. Only children one and two have nonuniform internal
phase increments. Their combined excess is the even-in-profile term
below; the four host matching interfaces contribute eight nonzero
single-slot costs. The donor contact contributes zero at this preparation.
Consequently the full initial excess is

\[
 E_0=6h^2+2\cos\alpha\bigl[(1-\cos2\epsilon)
                         +2(1-\cos\epsilon)\bigr]
                         +8(1-\cos\epsilon).
                                                               \tag{71}
\]

Using `1-cos(t)<=t^2/2` and `cos(alpha)<=1` gives the prospective rational bound

\[
 E_0\leq6h^2+10\epsilon^2
       =\frac{631}{256000}<\frac1{100}.                 \tag{72}
\]

The preparation has the exact reflection constraints. Its internal angles
lie between `alpha-2*epsilon` and `alpha+2*epsilon`, within `(0,pi/2)`;
matching gaps have magnitude at most `epsilon`. It therefore belongs to
the intersection (69), rather than merely passing its energy inequality.
All fine coordinates are retained. This is a static admission of the
declared preparation, not new formation or acquired-source evidence.

### Flow, events and finite composition

Start in `D`. Sector invariance and `H'=-||Lx||^2/4<=0` preserve the
reflection constraints and initial excess along an exact flow segment.
A first exit through an acute face would require excess above `6/25`,
contradicting `E<1/100`. Thus the phase cell remains intact, and (70)
continues to bound every form.

At the next requested ideal pair-Reception stage, independent live policy
and grammar admission are still required. If admitted, the already proved
rails make (64) unclipped. The pair preserves reflection, holds phases
and decreases or preserves storage by (67). Its successor is again in
`D`. Induction covers any finite number of these events and intervening
exact flows, followed by all future flow after the final event. In
particular,

\[
 \operatorname{wind}(C_9^{(c)})=1\quad(c=0,\ldots,4),
 \qquad |x_i|<3/10,
 \qquad E(\tau)\leq E(0)<1/100                         \tag{73}
\]

hold throughout such a continuation; the frozen preparation has `E(0)=E_0`
from (71). This does not assert universal
acceptance of future requests: stored pressure, grammar context and source
metadata are separate consumed state. Infinite or accumulating event
schedules, arbitrary additional operators and changed capacity laws are
outside this theorem.

After the last event, the
[same-law convergence argument](ORGANIZED_SOURCE_CAPTURE.md#4-permanent-winding-retention-and-full-state-convergence)
also applies: the trapped sublevel is compact on the fixed-mean leaf,
the acute phase storage is strictly convex there, and the largest invariant
zero-loss set is `x=0, delta=m`. Thus the remaining exact flow converges to
that target while retaining all five windings. Persistent winding identity
does not mean persistent nonzero motion. This supplies no finite settling
time and does not extend convergence to an infinite event schedule.

For a represented event, the ideal theorem alone is insufficient. Its
actual successor requires verification of the unchanged phase-lift,
support and capacity association with this model. Given that association,
it can re-enter the argument after checking exact sector preservation and
nonpositive actual work, or by directly re-admitting that successor to `D`.
Raw-vector admission alone cannot authenticate the runtime graph. Section 15
supplies those checks for its one frozen atomic control. No further
runtime event is executed or inferred here, and the serial control has
no exact-sector certificate.

### Shared admission and meaningful unavailable outcomes

The [capture owner](../../src/tnfr/physics/_organized_source_capture.py)
provides `_assess_organized_source_reflection_retention(form=...,
phase_offsets=...)`. It first normalizes the raw vectors and applies the
strict reflection restriction. Invalid primitives and off-sector states
reject. It then rebuilds the source-free boundary and full fine storage,
checks every fixed-branch angle and compares the full excess upper bound
with the frozen `1/100` ceiling. No incoming report, cached certificate,
attachment producer or acquisition calculation replaces those premises.

An admitted report carries conditional winding and form bounds. A valid
sector state outside the acute cell or above the sufficient ceiling gives
`unavailable`, retains its calculated evidence and supplies explicit unmet
conditions; its future winding and form bounds remain absent. A small
negative lower interval endpoint near equilibrium is ordinary enclosure
dependency and does not imply negative physical excess inside the cell.
The domain test uses the upper bound without silently clipping its evidence.

The [independent controls](../../tests/physics/test_organized_source_hybrid_retention.py)
derive the entry cost, actual grounded paths, Cauchy bound and ideal pair
work. They also distinguish low storage outside the acute cell, excessive
form storage inside it, and states outside the exact reflected lift.
They invoke neither numerical flow nor another runtime event. The claim
is a conditional maintained organization under a specified hybrid model,
not physical identification, an autonomous controller or a completed NFR
occurrence law.

<a id="organized-source-capacity-continuation"></a>
## 17. Transferring maintenance through a capacity change

**Conditional specialization of established exchange geometry.** The
[reciprocal form/phase balance](RESONANCE_FOUNDATIONS.md#reciprocal-exchange)
already covers a common diagonal mobility in both rows, including form
dissipation. The
[common exchange family](SINE_CONSTITUTIVE_INFORMATION.md#common-exchange-geometry-and-composition)
also distinguishes that balance from selection of a unique constitutive law.
The task here is to transfer section 16's particular reflected maintenance
domain to changed capacities on its actual support, not to rediscover the
form/phase relation. The sine coefficients and quarter normalization remain
those of this owner; native pressure and degree-normalized comparison
coefficients cannot replace them.

### Complete continuation and the supplied phase premise

Silence (SHA) preserves form, phase, stored pressure and support at its
primary jump, while attenuating capacity. Keep this model's independent
pressure `p_tau=(-q+gamma*S)/4`, with `q=Lx` on the ninety unit edges.
Write `D=diag(nu)` for the normalized capacities held during a continuous
segment. The selected pair reset has `nu_0=nu_1=kappa`, other capacities
one, and `0<kappa<1`; one is the no-reset baseline and zero is a separate
boundary considered below. The form row must now be

\[
 x_\tau=Dp_\tau=\tfrac14D[-q+\gamma S(\theta)].          \tag{74}
\]

This is the same declared pressure map, not pressure reconstructed from
the response. It contains no capacity-contrast channel. Native pressure
generally does contain such a channel: after this local reset it need not
vanish and is even under the reflection, unlike q and S. Substituting that
different pressure would change both the model and its symmetry argument.

Keeping the old phase row `theta_tau=gamma*q/4` is a possible complete
law, but its storage derivative is

\[
 H_\tau=-\tfrac14q^TDq+\tfrac\gamma4S^T(D-I)q.           \tag{75}
\]

In contrast, the established reciprocal continuation specializes to

\[
 x_\tau=\tfrac14D[-q+\gamma S],\qquad
 \theta_\tau=\tfrac\gamma4Dq,\qquad \nu_\tau=0,
 \qquad H_\tau=-\tfrac14q^TDq\leq0.                    \tag{76}
\]

The block cancellation proof in the linked exchange owner applies with
mobility `K=D/4`, `e=beta=1` and `a=b=gamma`. This is substitution into
its algebra, not identification of the actual unequal degrees with four.
All rows use the declared tau clock; original-time rates multiply by
`1023/1024` and slow-time rates by `2/gamma^2`. A local capacity change is
not a common reparameterization of time.

Equation (76) supplies the phase premise explicitly. Neither the nodal
product nor the Silence label uniquely imposes it. For example, a phase
row `gamma*Dq/4+R` changes the balance by `-S^T R`; exact preservation of
the same loss only constrains `S^T R=0`. Regularity, symmetries, retained
coordinates and event compatibility impose further independent obligations.
This freedom does not invalidate the earlier conditional exchange results.

### Why an unchanged phase row cannot inherit the loss theorem

The failure of automatic cancellation in (75) has a witness inside the
maintained domain, rather than merely outside its hypotheses. Let
`v=e_0-e_1`, `delta=m*1+epsilon*v`, `x=h*v`. Define

\[
 A(\epsilon)=\sin(\alpha+\epsilon)
              -\sin(\alpha-2\epsilon)+3\sin\epsilon.
\]

The actual support gives `q_0=6h`, `q_1=-6h`, `S_0=-A`, `S_1=A` and
`q^TDq=(72*kappa+8)*h^2`. Thus (75) becomes

\[
 H_\tau=-(18\kappa+2)h^2
          +3\gamma(1-\kappa)hA(\epsilon).              \tag{77}
\]

For any fixed `0<kappa<1`, sufficiently small positive h makes this
positive whenever A is positive. The static control fixes, before
evaluation, `epsilon=1/100`, `h=1/10^7`, `kappa=1/2`. On these angles
`cos>1/2` and `sin(epsilon)>epsilon/2`, so `A>3*epsilon`; also
`gamma>1/4000`. Consequently

\[
 H_\tau>\frac9{800000}h-11h^2>0.                       \tag{78}
\]

Both vectors satisfy the exact reflection constraints. Internal angles
are alpha, `alpha-2*epsilon` or `alpha+epsilon`, and matching gaps are
zero or `+/-epsilon`, all acute. At the target, `grad V=0` and
`Hess V<=L`, since every cosine weight is at most one. Along this phase
segment Taylor's integral identity therefore gives
`V-V_*<=epsilon^2*v^TLv/2=6*epsilon^2`. The entry excess obeys
`E<=6*h^2+6*epsilon^2<1/100`. This is an admitted small-state failure of
nonpositive storage rate for the unchanged phase row. It does not prove
eventual winding loss or absence of some other maintenance certificate.
Under (76), the same state instead has `H_tau=-11*h^2`.

### What transfers, and the zero-capacity boundary

The pair capacity matrix commutes with the reflection P. Section 14's
odd q and S therefore stay odd after multiplication by D: both versions
of the phase row preserve the exact sector, although only (76) has the
established nonpositive storage balance. For (76), the same geometric
phase-face barrier and grounded-path form bound apply without alteration.
The intersection `D_domain` in (69) is invariant, all five windings remain
one and every form stays below `3/10` in absolute value. No source
acquisition or new trajectory is needed for this transfer.

A capacity-only jump has `Delta H=0`, because H depends on form, phase
and support, which are held at this event. It also leaves the domain's
entry state unchanged. This statement requires those actual held-field
associations; a runtime label or latent-form record does not establish them.

For fixed strictly positive kappa, `q^TDq>=kappa*||q||^2`. The compact
trapped sector and strict acute phase convexity from section 16 then give
the same limiting target `x=0, delta=m`. No uniform convergence rate as
`kappa` tends to zero is asserted. Generic fine states have different
conserved leaves: during each fixed positive-capacity segment of (76),

\[
 \mathbf1^TD^{-1}x=\text{constant},\qquad
 \mathbf1^TD^{-1}\delta=\text{constant}.                \tag{79}
\]

These are raw weighted sums of continuous offsets, not circular means or
the old sum-of-child-means normalization. A capacity jump changes their
weights and can change their values without changing form or phase. On
the reflected sector, every ordinary child form mean remains zero and
offset mean remains m directly by symmetry; no inverse-capacity argument
is needed for that fact.

At `kappa=0`, (76) is still a regular finite-dimensional law with
nonpositive storage rate. The reflection, winding and form-rail certificate
survive. The targeted form and phase-offset coordinates are then frozen.
For a nonzero targeted form, convergence to the previous zero-form target
is impossible; the positive-capacity convergence theorem cannot transfer.
Inverse-capacity totals are unavailable. With the other phase premise,
zero capacity freezes only the unforced form row. Freezing phase is a
property of (76), not a consequence of the nodal product alone.

### Shared observations and the execution boundary

The [source-port owner](../../src/tnfr/physics/_organized_source_port.py)
adds `_observe_organized_source_capacity(form=..., phase_offsets=...,
capacities=..., phase_mobility=...)`. The required phase selector is
`shared_capacity` for (76) or `unit_capacity` for (75). All primitive
vectors and nonnegative capacities are admitted before fine observation.
Fresh currents, pressure, both rows, all clocks and the storage cross term
are rebuilt; zero capacities are not divided out. The nested
`unit_capacity_reference` retains its explicit original-law meaning and
must not be read as the new continuation's rates or conserved totals.

The [capture owner](../../src/tnfr/physics/_organized_source_capture.py)
adds `_assess_organized_source_shared_capacity_retention(form=...,
phase_offsets=..., pair_capacity=...)`. It reuses the raw reflected
domain check and rebuilds the actual common-capacity rows. The pair value
is admitted in `[0,1]`; the selected law is explicit in the function's
name. Valid states outside the sufficient domain remain unavailable.
At zero, an available retention certificate does not populate target
convergence. Neither reader authenticates a runtime graph or installs a law.

The [independent controls](../../tests/physics/test_organized_source_capacity.py)
check both balances, the fixed counterexample, actual edge currents,
weighted versus ordinary totals, clock changes, strict admission and the
zero-capacity boundary. They execute no Silence event, grammatical suffix,
continuous trajectory or frozen producer. The declared `EN -> IL -> SHA`
suffix remains unperformed: this capacity-law admission neither bypasses
Coherence nor supplies its phase work or live grammatical context.

<a id="organized-source-coherence-domain"></a>
## 18. Coherence preserves the ideal domain but fails default pair admission

**Two distinct conclusions.** The ideal common-snapshot Coherence (IL)
phase map preserves the entire reflected domain (69), for every common
phase-locking coefficient in `[0,1]`. This specializes the existing
[local-resultant method](SINE_COLLECTIVE_PRESSURE_HANDOFF.md#collective-il-local-resultants)
and [simultaneous shared-edge work identity](SINE_COLLECTIVE_PRESSURE_HANDOFF.md#collective-il-adjacent-work)
to the present degree-five pair. However, the current strict default form
policy cannot admit both reflected targets. The mathematical jump theorem
therefore does not execute the pending word step or authorize bypassing
its live guard.

### Actual phase map and a whole-domain local bound

For targets zero and one, use the same immutable pre-event state and
define their relative, bare-neighbor resultants

\[
 z_i=\sum_{j\sim i}e^{i(\theta_j-\theta_i)},\qquad
 R_i=|z_i|,\qquad a_i=\operatorname{Arg}z_i,
 \qquad d_i=\lambda a_i,\quad0\leq\lambda\leq1.        \tag{80}
\]

The ideal event holds forms, capacities and support, contracts each
target's stored pressure by the independently resolved IL factor, and
adds d_i to its continuous phase lift. Its stored circular representative
may wrap. The public/staged phase map is different from the pressure-only
low-level glyph. The ordinary phase coefficient is read from the shared
kernel's `DEFAULT_PHASE_LOCKING_COEFFICIENT`, whose represented value is
0.3; the proof covers its whole admitted interval.

Let `u_e=delta_b-delta_a` on an oriented edge. For an internal edge,
the target angle is alpha and the actual fixed-branch angle is alpha+u.
Concavity of cosine throughout the acute interval gives the integral
Taylor bound

\[
 \begin{split}
 B_\alpha(u)
 &=\cos\alpha-\cos(\alpha+u)-\sin\alpha\,u\\
 &=u^2\int_0^1(1-s)\cos(\alpha+su)\,ds
 \geq\frac{\cos\alpha}{3}u^2\geq\frac14u^2.
 \end{split}                                           \tag{81}
\]

Here `cos(alpha)>3/4`; the last bound also holds at u=0. Indeed the
concavity chord bounds the integrand cosine below by
`(1-s)*cos(alpha)+s*cos(alpha+u)`, with the endpoint cosine nonnegative.
For matching edges the same argument starts at zero and gives
`1-cos(u)>=u^2/3`. Internal linear terms cancel on every marked cycle.
Since form storage is nonnegative, every state in (69) satisfies

\[
 \sum_{e\in E}u_e^2\leq4(H-H_*)<\frac1{25}.            \tag{82}
\]

At the target, either node has two cycle neighbors and three matching
neighbors, so `z_*=3+2*cos(alpha)>9/2`. The elementary phasor difference
bound and Cauchy's inequality now give, for either actual five-neighbor
set,

\[
 |z_i-z_*|\leq\sum_{j\sim i}|\delta_j-\delta_i|
 \leq\sqrt5\left(\sum_{j\sim i}|\delta_j-\delta_i|^2\right)^{1/2}
 <\frac{\sqrt5}{5}<\frac12.                            \tag{83}
\]

Consequently `Re(z_i)>4`, `R_i>4` and
`abs(a_i)<atan(1/8)<1/8`. The local argument is unique and excludes
zero-resultant fallbacks. These constants have been re-derived on the
actual ninety-edge support; the earlier degree-seven contact bounds were
not imported. Global phase resultants need not define an angle here.

### Reflection, joint work and the jump path

The reflected neighbor sets and the primitive-phase turn relation (57)
give `z_1=conjugate(z_0)`. On the regular local branch this implies
`a_1=-a_0`, and hence `d_1=-d_0`. Adding the two opposite increments
preserves the exact lifted reflection sector and every child offset mean.
It does not supply a symmetry theorem for arbitrary represented `atan2`,
modulo or addition operations.

Reuse the established two-target identity with
`Delta=theta_1-theta_0`:

\[
 \begin{split}
 G_i&=R_i[\cos a_i-\cos((1-\lambda)a_i)],\\
 C_{01}&=\cos(\Delta-d_0)+\cos(\Delta+d_1)
          -\cos\Delta-\cos(\Delta+d_1-d_0),\\
 \Delta H&=G_0+G_1+C_{01}.                             \tag{84}
 \end{split}
\]

The shared edge must occur once in the actual work. Its correction has
`abs(C_01)<=abs(d_0*d_1)<=(d_0^2+d_1^2)/2` and need not be negative.
With `R_i>4`, `abs(a_i)<1/8` and `sin(s)>=3*s/4` on `[0,1/8]`, the
same integral argument as in the linked owner yields, for `0<lambda<=1`,

\[
 G_i\leq-\frac32\left(\frac2\lambda-1\right)d_i^2,
 \qquad
 \Delta H\leq-\left(\frac3\lambda-2\right)(d_0^2+d_1^2)
 \leq0.                                                \tag{85}
\]

Nonzero displacement gives strict decrease; lambda=0 gives an identity
phase map. Neither case eliminates stored-pressure or contextual changes.

Endpoint nonincrease alone would not exclude a jump across an acute face.
Consider the comparison path `theta(s)=theta+s*d`, `0<=s<=1`. Every
prefix is the same snapshot formula with coefficient s*lambda. Thus (85)
bounds its storage without assuming that the path remains in the acute
cell. A first boundary crossing would require excess above `6/25`, in
contradiction to the initial excess below `1/100`. The whole path and
endpoint therefore remain in (69), retaining all five windings and the
form rails. The path is a proof device for the event, not a finite-duration
pressure law or an executed numerical trajectory.

### Fresh rates and the frozen prefix

Coherence holds x, capacity and support, so q is unchanged. At unit
capacity its complete-law handoff has

\[
 \Delta p_\tau=\Delta x_\tau=\frac\gamma4\Delta S,
 \qquad\Delta\theta_\tau=0.                            \tag{86}
\]

Every touched edge contributes to Delta S, including the shared target
edge and the untargeted neighbors. Both clock conversions apply to the
rate jumps. The contracted stored pressure is distinct from this fresh
constitutive pressure; contracting a stored value does not change its
declared law.

In the frozen section 15 profile, the opposed child-one and child-two
phase deviations cancel each target's imaginary resultant; the cycle
contributions cancel as well. Therefore a_0=a_1=0 and ideal IL makes no
phase jump, even after Reception changes the forms. On this special
profile fresh pressure returns to its old constitutive value after any
stored-pressure contraction. The whole-domain proof above has nontrivial
content beyond that quiet profile. The previously fixed local phase
profile from section 17 supplies its nonzero-displacement control, with
no new preparation search or trajectory.

### A strict-default admission obstruction

The [strict IL validator](../../src/tnfr/operators/preconditions/coherence.py)
requires `EPI > min_epi` and `nu_f > min_vf`. The
[threshold owner](../../src/tnfr/config/thresholds.py) explicitly classifies
these as selected policies and sets `EPI_IL_MIN=0`. For the reflected pair,

\[
 x_0=-x_1\quad\Longrightarrow\quad
 \neg(x_0>0\ \text{and}\ x_1>0).                        \tag{87}
\]

The zero pair fails both strict inequalities. Thus no state in this
reflection sector admits both targets under the current strict default
form gate. In particular, the actual accepted Reception endpoint does not
admit the pending paired IL step, despite its mathematical phase map being
defined and domain-preserving. This is a live-policy obstruction, not a
failure of grammar syntax, storage geometry, the nodal identity or numerical
integration. No IL or SHA stage is executed here.

Negative EPI is valid signed form. Failure of this configured positive
threshold does not show that a negative target has no structure. Changing
the threshold, taking absolute values, adding a common form offset,
disabling strict checks or invoking only the pressure glyph would change
the admitted task. None is used to turn (87) into a successful execution.
The policy's foundational justification and any general migration require
their own contract assessment, preserving existing callers and evidence.

### Shared conditional assessment

The [capture owner](../../src/tnfr/physics/_organized_source_capture.py)
provides `_assess_organized_source_reflection_coherence(form=...,
phase_offsets=..., phase_locking_coefficient=...)`. It admits the coefficient
and raw state, rebuilds (69), and calculates local resultants, joint work
and both fresh rate changes. Invalid primitives and off-sector inputs
reject; valid states outside the sufficient domain give unavailable
endpoint and retention conclusions.

The ideal endpoint generally has transcendental offsets. Their outward
intervals feed the shared admitted-edge algebra directly; interval
midpoints are not converted into fictitious exact endpoints. The proved
reflection relation preserves the correlation between the two phase
increments. A conditional work upper bound of zero is distinct from a
rounded interval enclosure that may straddle zero at a quiet profile.
The assessment installs no law and asserts no live grammar or policy
admission. Actual represented execution would require its own held-field,
phase-lift association and successor checks.

The [independent controls](../../tests/physics/test_organized_source_coherence.py)
cover the actual resultants, shared-edge correction, whole endpoint work,
all affected currents and clocks, the quiet and nontrivial profiles,
invalid admission and the strict-default pair obstruction. They execute
no operator stage, source producer or continuous flow. The result admits
the ideal primary map while retaining the actual obstacle to completing
the pending `EN -> IL -> SHA` word.

<a id="organized-source-coherence-handoff"></a>
## 19. Represented Coherence handoff under explicit event-domain admission

### Frozen preparation and separate observations

The [general activity contract](../STRUCTURAL_OPERATORS.md#coherence-activity-admission)
admits the signed targets through explicit `event_domain` without changing
the event map or the default-policy rejection in section 18. This gate uses
the [frozen protocol](../../docs/assets/organized_source_coherence_handoff/protocol.json).
It declares one new atomic Reception prerequisite on the fixed section 15
input, then two atomic Coherence responses: one on that actual EN output and
one on the previously fixed section 17 nontrivial phase control. The older
Reception test producer is not invoked. Its original source, serial control
and evaluated conclusions remain separate.

Both IL stages target `(0,1)` on the same ordered ninety-edge support, with
unit capacities, the registered default pressure retention and represented
phase coefficient 0.3. The complete declared word is `EN -> IL -> SHA`, with
IL at index one. Initial `AL -> IL` history is supplied for the EN preparation;
the other control supplies `AL -> IL -> EN` history at its two targets.
Such supplied history is admission context, not evidence of those events.
No Silence, serial alternative, flow step, acquired-source reconstruction,
coefficient search or autonomous selector is part of this protocol.

The exact ideal offsets and runtime angles are distinct inputs. Initial
angles are the specified binary64 rendering of
`alpha*(j-4)+delta_(c,j)`. For the runtime state, interpret every stored
angle as its represented rational value on the mathematical circle of
period `2*pi`. The post-IL continuous lift uses turn minus one at the two
targets and zero elsewhere; its nearest-lift admission is independently
checked rather than inferred from the operator label.

The new shared `_observe_organized_source_phase_state(form=..., phases=...)`
reader in the [port owner](../../src/tnfr/physics/_organized_source_port.py)
re-admits all raw coordinates and reuses the admitted edge algebra with zero
carrier offset. It computes the same declared full law directly from
`sin(theta_j-theta_i)`, not from offsets reconstructed with approximate pi.
It shares form costs, row assembly and clock conversion with the ideal-offset
reader. Thus stored-pressure contraction and the later raw-phase constitutive
refresh remain separate observations. The callback records the contracted
pressure before one refresh; metrics preceding that callback do not become
fresh constitutive readings.

### Prospective error and scope boundaries

The immutable protocol fixes initial rendering error `epsilon=2^-48`, actual
input phase-reset error `b=2^-46`, exact-source endpoint comparison `B=2^-45`
and pressure materialization error `b` in tau units. The existing
[`bound_phase_reset`](../../src/tnfr/research/phase_reset_certificate.py)
rebuilds each relative resultant from the actual rational input angles with
outward interval arithmetic. Its strict nearest-lift test and error criterion
remain independent of grammar and future stability. An 80-decimal-digit oracle
checks the finite interval calculations separately.

These budgets precede response evaluation. If every initial angle differs
from its declared ideal lift by at most epsilon, each relative resultant
differs by at most `10*epsilon`. Since section 18 gives ideal magnitude above
four, its argument changes by at most
`10*epsilon/(4-10*epsilon) < 3*epsilon`. With the declared coefficient below
one third, initial-angle and ideal-reset errors contribute less than
`2*epsilon`; adding b stays below B. This comparison does not assume the
represented input has exact reflected phase relations.

The nine touched edges are counted once. Their target incidences total ten
and their untargeted incidences eight. The cosine Lipschitz bound therefore
gives source-to-runtime work discrepancy at most `10*B+26*epsilon`: eighteen
initial incidences and eight held endpoint incidences consume epsilon.
Relative to the ideal reset of the actual input, the work budget is `10*b`.
The moving control must have strictly negative represented work; the quiet
control receives only an absolute error bound, never automatic nonincrease.
Immediate winding/acute-cell membership, storage excess below `1/100`, held
fields, histories, and both fresh rows in all three clocks have their own
checks.

An error bound is not exact reflection. Already before IL, two rational
runtime angles cannot sum modulo `2*pi` to the exact reflected value
`1/20+4*pi/9` required for the first pair. The earlier EN result retained a
separately declared ideal lift because EN held its rendering unchanged.
After IL, bounded agreement with a new ideal endpoint does not identify the
literal represented state with that endpoint. Neither averaging increments
nor subtracting a binary64 alpha repairs this distinction. Exact reflected
future maintenance remains unavailable without its own state association;
an immediate geometric certificate is a different conclusion.

### Retained response and interpretation

The [source receipt](../../docs/assets/organized_source_coherence_handoff/freeze.json)
binds the complete current source tree, instrument and protocol to the
[source archive](../../docs/assets/organized_source_coherence_handoff/source.zip).
The [exclusive attempt](../../docs/assets/organized_source_coherence_handoff/attempt.json)
retains ideal predictions before the three declared events. The
[response](../../docs/assets/organized_source_coherence_handoff/response.json)
records their primitive before/after states and the pre-refresh callback
boundary. These byte associations are not independent chronology or provenance
authentication. No earlier producer was replayed or overwritten.

Both atomic IL stages passed the frozen reset, lift, work and immediate
domain criteria. The table gives rounded displays of retained outward
intervals, not replacement acceptance bounds:

| Observation | Actual Reception handoff | Supplied moving control |
| --- | --- | --- |
| Maximum reset defect against the ideal map of the actual input | `2.257e-16` | `4.305e-16` |
| Maximum endpoint discrepancy against the separate exact source | `1.971e-16` | `5.119e-16` |
| Joint IL storage work | approximately `+4.096e-33` | approximately `-3.070111795e-4` |
| Endpoint storage excess above H* | approximately `0.001598573` | approximately `0.000223437` |

All ninety fixed-branch angles remain acute, the five cycle windings remain
one, total form is exactly zero and the form rails hold. IL preserves form,
capacity, support, structural time and untargeted histories; both target
histories advance from `AL, IL, EN` to `AL, IL, EN, IL`. No IL warning occurs.
The EN prerequisite retains two no-emission-source telemetry advisories:
supplied history does not authenticate preceding Emission events or source
selection. These advisories neither replace live admission nor select events.

The quiet work enclosure is strictly positive at the represented endpoint.
It satisfies the declared absolute comparison budget; it is not a finite
nonincrease result. The moving control has strictly negative joint work,
including the positive shared-edge correction. Both preserve the fresh phase
rate at the jump because form is held, while fresh form rates respond to the
changed sine currents. Stored pressure contraction is separately verified
before refresh. The two paired lift increments have a nonzero sum of about
`-4.899e-16`; numerical closeness is not exact antisymmetry.

The [read-only evidence controls](../../tests/research/test_organized_source_coherence_handoff_evidence.py)
rebuild source predictions with an independent 80-digit oracle, re-admit raw
states, reconstruct all complete-law rows and clocks, and independently count
the nine touched edges. They forbid event, source-response and trajectory
execution. Generic controls also exercise raw-state admission and common-phase
covariance. Cached analysis flags are not the evidence for these conclusions.

This closes the represented reset and immediate-domain question. It does not
complete the declared word, prove source occurrence or transfer exact reflected
future maintenance to rational runtime angles. The existing full-graph
spectral and acute-boundary bounds may instead support a maintenance theorem
on the complete mean-zero leaf, with its own raw-state and lift admission.
That possibility requires a separate argument, not tighter reset tolerances.

<a id="organized-source-raw-maintenance"></a>
## 20. Continuous maintenance from raw phase state without reflection

### Fixed model and exact lift admission

This conditional theorem extends maintenance to the full zero-total-form
leaf of the unchanged ninety-edge, unit-capacity model. It starts from
primitive state, not an ideal reflected state associated within a tolerance.
The support, the quarter factor in both rows, gamma, clocks and absence of
later inputs or events remain the premises of section 1. In particular,
pressure is freshly defined by the complete constitutive law; the binary64
stored-pressure snapshot is not assumed to satisfy that law exactly.

Let `r_i` be the finite raw phase coordinate and `k_i` its explicitly supplied
integer turn. Retain the exact affine-in-pi lift and its mean:

\[
 \Theta_i=r_i+2\pi k_i,\qquad
 \mu=\frac{\sum_i r_i+2\pi\sum_i k_i}{45}.
\]

No rational offset is reconstructed by subtracting an approximate carrier
angle. For every oriented actual edge `a -> b`, use

\[
 a_e=(r_b-r_a)+2\pi(k_b-k_a+c_e),
 \qquad
 c_e=\begin{cases}1&\text{on each internal closing edge }8\to0,\\
                  0&\text{otherwise.}\end{cases}
\]

Cancel the integer differences before interval evaluation. Each marked
cycle then has exact sum `2*pi`. If all ninety angles are strictly acute,
these are its principal edge differences and its winding is one. The
connected fixed-branch cell is convex. The future solution uses the initial
lift continuously; it does not select new integer turns at later snapshots.
Its equilibrium has `Theta_(c,j)=psi_j+mu`, where
`psi_j=(2*pi/9)*(j-4)` and the carrier sum over all nodes is zero.

The admitted domain is

\[
 \mathcal D_0=\{(x,\Theta):\ \mathbf1^Tx=0,\quad
             |a_e|<\pi/2\ \text{for every edge},\quad
             E=H-H_*<1/100\}.
\]

It imposes no childwise form mean, fixed reflection origin or antisymmetry.
Finite scalars and exact zero total form are necessary domain admissions.
Wrong local turns can leave the acute cell even when sine currents and
physical storage are unchanged. Small storage alone cannot replace the
fixed-branch check: the winding-zero equilibrium has negative excess
relative to this different winding-one reference.

### Barrier, form rails and all-future winding

On the closed acute cell, the existing cycle Jensen inequality gives
`F=x^T Lx/2 <= E`. The actual full-graph spectral bound (24), which includes
all fine modes, therefore yields

\[
 \|x\|_2^2\le\frac{2F}{4/15}\le\frac{15}{2}E
 <\frac3{40}<\left(\frac3{10}\right)^2.
\]

This derives `abs(x_i)<3/10<1` without a reflection-fixed zero slot.
The zero-total premise is essential for this estimate: a common nonzero
form is invisible to `Lx` and can have arbitrarily large magnitude at the
same form-storage cost. No centering or projection silently removes it.

Edge cancellation in both complete rows conserves `sum x` and `sum Theta`.
The inherited identity `H_tau=-||Lx||^2/4` makes E nonincreasing. Every
acute-cell face requires `E>=beta_face>6/25`, by the
[same full-support boundary proof](ORGANIZED_SOURCE_CAPTURE.md#2-every-acute-boundary-face-has-a-positive-excess).
A first exit from a state in `D_0` would contradict the strict separation
`E<=E_initial<1/100<beta_face`. Global existence follows from the finite
dimensional globally Lipschitz complete vector field. Thus the solution
remains in `D_0` for all future tau, all five marked windings remain one,
and every form remains within the stated rails.

This is a domain theorem, not a verdict inferred by sampling a trajectory.
Positive rescalings to original time t and slow time s preserve the
conclusion while transforming both rows and the storage rate together.
The reported form bound is sufficient and deliberately not claimed sharp.

### Full-state asymptotic limit and its scope

The conserved phase mean is the actual mu above, rather than `1/40` supplied
by the earlier ideal reflection chart. Connected support and bounded
fixed-branch edge differences bound all phase coordinates at fixed mean.
The form estimate and strict boundary separation make the trapped sublevel
compact. Its phase Hessian is a connected graph Laplacian with weights
`cos(a_e)>0`, positive definite on the fixed-mean tangent space. Hence the
phase potential is strictly convex there, and `psi+mu` is its unique
stationary point: internal currents cancel and matching currents vanish.

In the zero-loss set, `Lx=0` and the zero total imply `x=0`. The phase row
then vanishes, and invariance of this set further requires `S(Theta)=0`.
The largest invariant subset is the single equilibrium on this mean leaf.
The [existing compact-sublevel argument](ORGANIZED_SOURCE_CAPTURE.md#4-permanent-winding-retention-and-full-state-convergence)
therefore gives

\[
 x_i(\tau)\longrightarrow0,\qquad
 \Theta_{c,j}(\tau)\longrightarrow\psi_j+\mu.
\]

No finite settling time or permanent nonzero motion follows. Individual
child means may vary. This convergence and winding retention concern the
ideal continuous complete law starting from the represented endpoint;
they do not certify native time stepping or numerical error for future
times. Nor do they derive source formation, autonomous event selection or
physical identification.

Arbitrary later Reception stages do not inherit the earlier reflected
hybrid theorem. Outside that sector a pair can change total form by
`-beta*(q_0+q_1)/5`, where beta is its mixing factor and `q=Lx`.
Silence changes capacity and therefore requires a separately declared
complete continuation law and its own admission. Neither event follows
merely because its predecessor belongs to `D_0`.

### Shared certificate and retained-endpoint result

The [capture owner](../../src/tnfr/physics/_organized_source_capture.py) provides
`_assess_organized_source_raw_retention(form=..., phases=..., phase_turns=...)`.
It reuses the raw complete-law observer, full-support gap check and source-free
boundary owner. It fixes the model rather than accepting an arbitrary graph or
capacity law. A consumer associating an external runtime snapshot must first
admit its support, node order, capacities and clock. No incoming report,
cached verdict, acquired source or event history substitutes for that work.

All ninety raw scalars, forty-five ordinary integer turns and exact zero
total form are admitted before geometry. Invalid primitives and nonzero
total form reject; valid states outside the strict acute cell or energy
ceiling retain observations but receive no future rail, winding or limiting
phase certificate. The exact raw phase sum and integer turn sum accompany
their outward interval mean. The barrier's old template origin is only a
translated reference for energy and face bounds, not the endpoint's limit.

The [independent controls](../../tests/physics/test_organized_source_raw_retention.py)
derive a positive rational LDL certificate for `L-(4/15)I` on the entire
44-dimensional zero-sum fine space, including its nonorthonormal basis metric.
An asymmetric admitted state has moving child phase means, demonstrating why
those means cannot be claimed invariant. Separate controls reject nonzero
form totals before observation and distinguish high energy, a wrong local
turn and a low-energy winding-zero equilibrium. A common integer-turn shift
changes the lifted origin while leaving edge angles, storage and rates intact.

**Retained result.** Both section 19 IL endpoints pass this new certificate
after their primitive snapshot/model association is re-admitted. Their excesses
remain approximately `0.001598573` and `0.000223437`, below `1/100`, with the
same ninety acute angles and exact zero form total. Thus both have conditional
all-future winding-one retention, `abs(x_i)<3/10`, and convergence to their own
lifted mean under the unchanged continuous law. The interval means exclude
exact equality to `1/40`; neither endpoint is projected onto that origin.
Both complete rows in all three clocks agree with the retained observations.

This is a new mathematical admission applied read-only to old observations.
The earlier response's `future_retention_certified=false` and
`exact_reflection_certified=false` flags remain byte-identical: that response
did not establish this broader theorem. No event, source producer or flow
was replayed, and no error budget or evaluated prediction was rewritten.

<a id="organized-source-silence-handoff"></a>
## 21. Silence suffix and positive-capacity continuation

### Weighted-leaf transfer of the existing reciprocal law

Silence (SHA) changes capacity at its jump. Continue with the explicit
`shared_capacity` law (76), the same ninety unit edges and no later input or
event. This is the phase premise already studied in section 17, now applied
to raw phases without an exact reflection assumption. It is not selected by
the nodal identity or by the name of the closing operator.

For positive held capacities write `D=diag(nu)`, `w_i=1/nu_i`, and retain
the raw lift `Theta_i=r_i+2*pi*k_i` from section 20. The independent pressure
and complete rows are

\[
 p_\tau=\tfrac14[-Lx+\gamma S(\Theta)],\qquad
 x_\tau=Dp_\tau,\qquad
 \Theta_\tau=\tfrac\gamma4DLx,\qquad \nu_\tau=0.
\]

In this particular pressure law, changing capacity leaves p unchanged while
changing both rates. A callback storing the new form rate as pressure would
multiply by capacity again at the next nodal step. Pressure must be evaluated
from its independent formula, never reconstructed from that step.

The existing exchange cancellation and edge sums give

\[
 H_\tau=-\tfrac14(Lx)^TD(Lx),\qquad
 (w^Tx)_\tau=0,\qquad (w^T\Theta)_\tau=0.
\]

Ordinary totals need not remain constant after a nonuniform reset. Require
`w^T x=0` explicitly, every fixed-branch angle acute, and `E=H-H_*<1/100`.
The same phase barrier proves all-future acute-cell and winding retention.
For the form bound, put `y=x-mean(x)*1` and
`rho=nu_max/nu_min=w_max/w_min`. Weighted centering and the full spectral
gap (24) imply

\[
 w_{\min}\|x\|_2^2\le\sum_iw_i x_i^2
   =\min_c\sum_iw_i(x_i-c)^2
   \le\sum_iw_i y_i^2\le w_{\max}\|y\|_2^2,
 \qquad
 \|x\|_2^2\le\rho\,\frac{15}{2}F\le\rho\,\frac{15}{2}E
              <\rho\,\frac3{40}.
\]

Here `F=x^T Lx/2=y^T Ly/2`. This bounds the full form, not just its
centered component. Geometric maintenance does not depend on the bound
being less than one; an explicit unit-rail claim does. A capacity ratio
whose sufficient bound fails that rail test is not evidence of actual exit.

The phase origin of the limiting carrier is

\[
 c_D=\frac{\sum_iw_i(\Theta_i-\psi_i)}{\sum_iw_i},\qquad
 \psi_i=\frac{2\pi}{9}(i\bmod9-4).
\]

Subtract the carrier before taking this weighted mean: `sum(w_i*psi_i)`
need not be zero. Preserve its exact affine-in-pi numerator
`sum(w_i*r_i) + pi*(2*sum(w_i*k_i) - (2/9)*sum(w_i*(i%9-4)))`.
A capacity jump can change c_D even though no phase moves at that instant.

Positive held capacity makes the zero-loss set `Lx=0`. The weighted-zero
form leaf then forces `x=0`. Connected acute geometry and the fixed weighted
phase total make the trapped sublevel compact. The phase Hessian is positive
on the weighted-mean tangent space. At a constrained critical point,
`S=c*w`; its zero ordinary sum implies `c=0`, since `sum(w)>0`. The unique
stationary carrier translate is therefore `psi+c_D`. The same invariance
argument yields `x->0` and `Theta->psi+c_D`. No uniform settling rate as
capacity approaches zero, or future numerical trajectory, is certified.

Zero capacity is a separate boundary and cannot enter these inverse-capacity
totals. The existing observer still admits its finite complete rows. The
`unit_capacity` phase alternative retains the cross term (75); it cannot
borrow this theorem merely because the nodal form row agrees. Both complete
rows and storage rates transform under all three declared clocks.

### Prospective suffix protocol

The [frozen protocol](../../docs/assets/organized_source_silence_handoff/protocol.json)
uses the two retained section 19 IL endpoints, their exact raw phases and
turns, and their original ninety-edge order. The targets remain `(0,1)`.
The recorded target history is `AL, IL, EN, IL`, at ordinal four. A new
`ValidatedSequence` for `EN -> IL -> SHA` supplies step-two context; the live
history is separately admitted. Only two atomic SHA stages are authorized:
no preceding event, source producer, flow segment or additional control is
replayed. The ordinary strict SHA capacity floor remains its declared
represented default 0.01.

The registered factor is frozen as its represented binary64 value
`kappa=0.9204225284540524`, not as an exact irrational formula. Both incoming
capacities are one, so multiplication preserves this represented factor
exactly. The predicted capacities are `kappa,kappa,1,...,1`. Each source has
opposite target forms and zero form elsewhere, hence its new weighted form
total is exactly zero. Since `kappa>5/6`, the bound above is
`rho*3/40=3/(40*kappa)<9/100`; the old strict `abs(x_i)<3/10` rails therefore
remain available without reflection.

Reconstruction copies the recorded consumed structural coordinates, source
metadata, history, ordinal and grammar context. Fresh monitor/cache/telemetry
sinks and absent optional lifecycle/controller state are declared explicitly.
This association is not a complete checkpoint or an uninterrupted full-runtime
replay of the whole word. The source receipts and exact copied-field checks
bind the suffix preparation to the earlier primitive observations.

Acceptance requires unchanged form, phase, kind, support, stored dEPI and
structural tau; exact target capacity attenuation; unchanged non-target
capacity; valid live grammar and ordinal progression; and the actual five
latency fields, including the signed preserved form. The stage's common UTC
wall timestamp is recorded telemetry, not elapsed structural time. Capture
held stored pressure before the one callback refresh; the fresh constitutive
pressure has budget `2^-46` in tau units. Both complete rows in all clocks,
weighted totals, limiting origin and positive-capacity maintenance have
separate checks. Neither `latent=true` nor `was_initial_on_silence` certifies
vanishing evolution; the frozen sources predict nonzero form and phase rates.

At the event, held form/phase/support imply exact `Delta H=0`, separately
from continuous dissipation afterward. Built-in metrics are retained as
telemetry. Missing temporal history cannot certify temporal variance or
preservation, and nonfinite diagnostic estimates receive an explicit
unavailable marker with their original representation rather than a JSON
number or a zero substitute. The structural certificate uses raw state and
the declared complete law, not those diagnostic estimates.

### Retained represented response

The [source freeze](../../docs/assets/organized_source_silence_handoff/freeze.json)
and [exclusive attempt](../../docs/assets/organized_source_silence_handoff/attempt.json)
preceded the two stages. The frozen evaluator and attempt supplement explicitly
declare `epi_history_during_silence` absent in the reconstructed runtime; the
older snapshot does not establish that absence. The
[response](../../docs/assets/organized_source_silence_handoff/response.json)
records exactly two accepted atomic SHA stages, one per preparation, with
all 64 checks passing in each. No EN/IL event or continuous step was replayed.

Both target capacities become the frozen kappa; all other capacities remain
one. Signed form, represented phase, support and structural tau are held.
The histories advance to `AL, IL, EN, IL, SHA`, at ordinal five, with no
grammar substitution. Each stage has one shared UTC lifecycle timestamp and
one fresh constitutive-pressure callback. The exact event work is zero.

| Preparation | Excess storage after SHA, approximately | Continuous loss in tau, approximately | Conditional continuation |
| --- | ---: | ---: | --- |
| Retained Reception/Coherence handoff | 0.001598572987237434 | 0.002286789838860984 | Five windings, form rails and convergence admitted |
| Retained moving control | 0.00022343667977782226 | 1.856760551217294e-13 | Five windings, form rails and convergence admitted |

Each primitive state has exactly zero inverse-capacity weighted form total.
The common sufficient squared-form bound is `0.08148431582391892`, below
the frozen `0.09` ceiling; its outward square root is below `0.286`.
The two weighted limiting phase origins are near 0.025, computed from their
actual raw lifts rather than rounded to that value. Both target form and phase
rates remain nonzero, including the control marked `was_initial_on_silence`.
Lifecycle labels therefore do not imply stationarity.

Read-only evidence tests independently rebuild all edge sums and complete
rows at 80-digit precision, all three clocks, storage balance, weighted totals
and the carrier subtraction. They verify archive receipts and exact consumed
endpoint association while prohibiting event and flow execution. Two interval
expressions for the same scaled rate can differ by outward rounding; each
must contain the independent rate, without requiring identical endpoints.
The earlier frozen evidence remains unchanged.

This closes the represented suffix gate with a conditional continuous
maintenance theorem. It supplies a checked consumed-state handoff through
the selected word, not an uninterrupted runtime checkpoint or an autonomous
occurrence law. An open admission domain for the complete word, with
independent asymmetric perturbations and nonzero conserved form means,
remains a separate mathematical obligation.

<a id="organized-source-robust-word"></a>
## 22. A complete word on an open form-and-phase domain

### Arbitrary conserved form means

Reception (EN), Coherence (IL) and Silence (SHA) need not preserve the
zero-weighted-form condition used in section 21. That condition therefore
cannot define a neighborhood allowing independent form perturbations.
For the same positive held-capacity reciprocal continuation, retain instead

\[
 m_D=\frac{\sum_i w_i x_i}{\sum_i w_i},\qquad z=x-m_D\mathbf1,
 \qquad w_i=1/\nu_i.
\]

The edge sums still give `m_D'=0`. Centering is a proof coordinate, not a
replacement for the supplied state: the observer continues to consume raw x.
Since `Lz=Lx` and `sum(w*z)=0`, the weighted least-squares and spectral
argument from section 21 yields

\[
 \|z\|_2^2\le\frac{2\rho}{\lambda_2}F
 \le\frac{2\rho}{\lambda_2}E(0),\qquad
 |x_i(t)|\le |m_D|+\sqrt{\frac{2\rho}{\lambda_2}E(0)}.
\]

An outward upper bound on initial excess can replace `E(0)`. The strict
acute-cell energy ceiling still guarantees geometric retention. A separate
mean-plus-radius test is needed to certify unit form rails; failure of this
sufficient bound does not prove an exit. On the invariant zero-loss set,
`Lx=0` gives `x=m_D*1`. Invariance then requires `LDS=0`, so `S=c*w`;
the ordinary sum of S is zero, hence c=0. The previous compactness and
strict-convexity argument gives

\[
 x(t)\longrightarrow m_D\mathbf1,\qquad
 \Theta(t)\longrightarrow\psi+c_D\mathbf1 .
\]

The phase origin c_D retains the weighted carrier subtraction from section
21. The additive `_assess_organized_source_weighted_mean_retention` reader
implements this domain; the older weighted-zero reader keeps its original
rejection contract and serialized results.

### Fixed execution context and the open domain

Use the raw pre-Reception primitive state retained in stage zero of the
section 19 response as center `(x^0,theta^0)`. Declare zero input lift turns.
For every node, independently, admit

\[
 |x_i-x_i^0|<r,\qquad |\Theta_i-\theta_i^0|<r,\qquad r=10^{-5}.
\]

This is an open subset of ninety real coordinates. Support, initial unit
capacities, clocks, targets `(0,1)`, factors, kinds and grammar context are
fixed; the binary64 subset is not itself an open real domain. The primary
maps act successively, with both targets reading one snapshot per stage.
There is no continuous segment between the three events and no whole-word
atomic transaction is asserted.

All nodes have the same nonempty `organized-source` kind and supplied
`AL, IL` history, ordinal two, zero debt and prior Coherence. Use initialized
core grammar, rejection mode `raise`, hard unit rails and the explicit IL
`event_domain` policy. The historical default signed-positive policy still
does not admit this pair. Strict EN checks use their actual represented
signed-form and signed-pressure ceilings. The SHA floor is also an explicitly
represented policy value. Controllers, custom integrity monitors and the
optional held-step nodal validator are absent. A callback may only refresh
the independent pressure law; no callback state mutation is covered.

The exact represented factors are read from their shared owners: EN mix
beta, IL pressure retention b, phase coefficient lambda and SHA factor
kappa. The word reader reuses canonical factor validation and rejects an
unrepresented factor or policy ceiling instead of silently changing it.
The phase coefficient lies in `[0,1]`; the other three domains are owned by
`operators/factor_contracts.py`. Positive continuation additionally excludes
the otherwise valid zero-capacity SHA boundary.

The live trace becomes `AL, IL, EN, IL, SHA`, with target ordinals two
through five. Kind selection is harmless here because every candidate kind
agrees. Source discovery, latency thresholds and metrics remain telemetry:
they do not select the numeric EN input set or the continuous law. The SHA
wall timestamp does not advance the declared structural clock.

### Compositional bounds from primitive state

The detached `_assess_organized_source_word_domain` reader reconstructs
the exact primary-map center. Reception is the unweighted convex neighbor
blend; at the targets,

\[
 \bar x_i=(1-\beta)x_i^0+
            \frac{\beta}{d_i}\sum_{j\sim i}x_j^0,
 \qquad d_i=5.
\]

Other forms are held. With admitted Reception form defect `e_x`, the full
successor form differs from this center by at most `A=r+e_x`. This includes
the two-stage binary64 mean/blend error if a represented execution is being
assessed; a represented matrix multiplication is not substituted for that
kernel.

For Coherence, rebuild the relative center resultant
`Z_i^0=sum_j exp(i*(theta_j^0-theta_i^0))`. An independent input phase
perturbation changes it by at most `2*d_i*r`. Require
`M_i=Re(Z_i^0)-2*d_i*r>0`. Every segment between the center and perturbed
resultant remains in that right half-plane; integrating the derivative of
Arg therefore gives

\[
 |\operatorname{Arg}Z_i-\operatorname{Arg}Z_i^0|
 \le\frac{2d_i r}{M_i},\qquad
 B=\max_{i\in\{0,1\}}\left(r+\lambda\frac{2d_i r}{M_i}\right)+e_\theta .
\]

The center phase after IL is its initial lift plus
`lambda*Arg(Z_i^0)` at each target. B bounds all final lifted-phase
deviations, including an admitted IL lift/reset defect `e_theta`.
Untargeted phases retain their input deviation r. SHA holds form and phase;
its unit input capacities make the represented output exactly
`kappa,kappa,1,...,1`.

Let `W=sum_edges weight=90` and
`V_x=sum_edges weight*abs(bar_x_j-bar_x_i)`. The exact quadratic form
storage and the global Lipschitz constant one of cosine imply

\[
 E_{\rm successor}\le
 \bar E+2A V_x+2A^2W+2BW=:E_{\max}.
\]

Here the interval center excess `bar E` is rebuilt from primitive forms and
phases and the exact primary maps. It is not a cached historical verdict.
The reader also checks every initial edge angle enlarged by `2r` and every
final center edge angle enlarged by `2B` against the acute boundary.
Integer turn differences cancel before interval construction.

For the EN pressure precondition, its independent constitutive formula gives
`abs(p_i-p_i^0)<=d_i*(r+gamma*r)/2+e_p`, where `e_p` bounds stored-pressure
materialization. The signed upper ceiling is checked with that bound;
pressure is never recovered by dividing a measured form rate by capacity.
IL's finite pressure contraction and each subsequent refresh are separate
runtime obligations; they do not alter the geometric estimates above.

Finally, with the exact center weighted mean `bar m_D`,

\[
 |m_D|\le|\bar m_D|+A,\qquad
 |x_i(t)|\le|\bar m_D|+A+
       \sqrt{\frac{2E_{\max}}{\kappa\lambda_2}},\qquad
 \lambda_2\ge4/15.
\]

All these are joint sufficient bounds on the whole box. They do not infer
entry by intersecting unrelated marginal state budgets.

### Constructive radius and represented scope

At `r=10^-5`, the rebuilt ideal center and resultant bounds give
`B<0.000016620`, `E_max<0.004593` and an all-future absolute form bound
below `0.194`. Every initial and final edge remains acute, and both signed
EN admission margins are strict. The whole ideal primary word is therefore
admitted on this initialized domain; all five windings persist under the
declared subsequent continuous law.

The same conclusions follow with the independently declared defect budgets
`e_x=e_theta=e_p=2^-46`; the energy and rail bounds remain below the same
rounded upper limits. These are conditional defect-transfer statements.
The degree-five IL runtime uses ordinary platform phasor/trigonometric
operations, not the certified two-neighbor branch. The old finite reset
certificate does not prove a uniform backend error bound on this open box.
Each future represented use must establish its actual map/lift defects,
held fields, factors, policy, grammar and pressure-refresh association.
No new runtime response or uninterrupted historical word is claimed here.

An independent analytic check avoids reliance on the calculated center:
the original four offset deviations have amplitude `a=1/100` and total
edge squared difference `20*a^2`. The carrier's linear potential term
cancels, so their phase excess is at most `10*a^2`. The ideal quiet IL
center has zero phase displacement. With `h=1/64`, EN center form storage
is at most `6*h^2=3/2048`. The retained raw-center rendering error is at
most `epsilon=2^-48`. Put `eta=r+epsilon`; then the looser bounds
`B<=2*eta+e_theta` and

\[
 E_{\max}\le3/2048+1/1000+360hA+180A^2+180B<7/1000
\]

independently verify the energy margin. No new trajectory, source search,
operator response or parameter scan supplies this proof.

The mean extension is necessary even when the input ordinary mean is zero.
Perturb only `x_2` by `+delta` and `x_3` by `-delta`. The declared neighbor
support makes EN add `beta*delta/5` only to target one. After SHA,

\[
 m_D=\frac{\beta\delta}{5(43\kappa+2)}\ne0 .
\]

Independent incidence and high-precision map checks verify this control,
the full box bounds and invalid-domain rejection. It exhibits a permitted
asymmetric word whose eventual form differs from zero while its geometric
organization remains retained.

<a id="organized-source-word-reuse"></a>
## 23. Complete-word passivity and finite reuse

Reception (EN), Coherence (IL) and Silence (SHA) can be composed repeatedly
on a larger ideal domain than the reflected source sector. This result
uses the same ninety unit edges, targets `(0,1)`, primitive phase law and
declared clocks as sections 21–22. It neither selects these events nor
changes the held inter-event capacity law. The exact primary maps, the
scalar rounded capacity recurrence and a complete represented execution
have separate obligations.

The detached
[`_assess_organized_source_word_passivity`](../../src/tnfr/physics/_organized_source_passivity.py)
re-admits every form, phase, lift turn, positive capacity and event factor.
It rebuilds the raw acute-cell excess `E=H-H_*<1/100`, independently of
reflection, zero form mean or a sufficient form-rail certificate.
Its work bounds are conditional mathematical certificates. They do not
invoke operators, authorize live admission or certify numerical map errors.

### Joint work without reflection

Write `q=Lx`, `beta=EN_mix` and `a=beta/5`. Reception proposes
`x'=x-a*P*q`, where P retains only the two target coordinates. Their
principal Laplacian block is `[[5,-1],[-1,5]]`; both targets read the same
snapshot. For `Q=q_0+q_1` and `R=q_0-q_1`, direct expansion gives

\[
 \begin{split}
 \Delta H_{\rm EN}
 &=-a(q_0^2+q_1^2)
   +\frac{a^2}{2}\{5(q_0^2+q_1^2)-2q_0q_1\}\\
 &=-\frac a2\{(1-2a)Q^2+(1-3a)R^2\}\\
 &\leq-a(1-3a)(q_0^2+q_1^2)\leq0 .
 \end{split}
\]

This holds for every admitted `0<beta<=1`, without symmetry or a mean
constraint. Replacing beta by s*beta gives the same sign for each reset
prefix, `0<=s<=1`. The phase is held, and the convex Reception map also
preserves the instantaneous form extrema. The two loss modes must remain
distinct when budgeting subsequent mean drift.

The local-resultant proof (81)–(83) also does not require reflection.
On each marked cycle the linear term cancels around the cycle; matching
edges have zero carrier gap. Consequently the unrestricted acute excess
still implies `sum(u_e^2)<=4E<1/25`, `Re(z_i)>4` and
`abs(Arg(z_i))<1/8` at both targets. Only the separate claim
`z_1=conjugate(z_0)` in section 18 needed reflection.

Let `a_i=Arg(z_i)` independently and `d_i=lambda*a_i`, with
`0<=lambda<=1`. Keep both actual increments and the shared-edge correction
in (84). The same isolated-work and mixed-edge bounds yield

\[
 \Delta H_{\rm IL}
 \leq-\lambda(3-2\lambda)(a_0^2+a_1^2)\leq0 .
\]

There is no projection to opposite increments or discarded mean phase.
Lambda zero is an identity phase map; otherwise a nonzero angle gives
strict decrease. Replacing lambda by s*lambda bounds every comparison
prefix without assuming it stays acute. The first acute boundary would
cost more than `6/25`, whereas the initial excess is below `1/100`.
Thus the entire comparison path stays in the same cell and retains all
five windings. This is an event proof, not a finite-duration flow.

Silence holds form, phase and support, so `DeltaH_SHA=0`. It multiplies
both target capacities by the declared `0<kappa<1`. Stored pressure and
latency metadata have their own event contracts; they are not terms in H.
Between complete words, the declared reciprocal continuation gives
`H_tau=-q^T D q/4<=0` with positive held D. Any finite admitted ideal
composition with arbitrary finite waiting intervals therefore retains
`E<1/100` and the five winding identities.

This conclusion does not turn the earlier tiny positive represented
Coherence work into exact nonincrease. Actual rounded words need their own
stage/lift defects and cumulative work budget. The frozen responses and
their verdicts are unchanged.

### A joint mean budget that survives capacity contrast

Geometric retention alone does not bound absolute form. For reuse, impose
equal target capacities `v_j=v_0*kappa^j`, with all other capacities one.
Let `u=(x_0+x_1)/2`, `b=sum(x_2,...,x_44)/43` and

\[
 m(v)=\frac{x_0+x_1+v\sum_{i=2}^{44}x_i}{2+43v}
     =c(v)u+(1-c(v))b,\qquad c(v)=\frac2{2+43v}.
\]

This is the actual inverse-capacity weighted mean. It is conserved during
each held-capacity flow interval, changes during Reception and Silence,
and is held by Coherence. There is no centering of the raw form.
Reception and Silence have the exact resets

\[
 \Delta m_{\rm EN}=-\frac{aQ}{2+43v},\qquad
 \Delta m_{\rm SHA}=[c(\kappa v)-c(v)](u-b),
\]

where the latter consumes the post-Reception form. The support diameter
is seven: the C9 distance is at most four and the child-graph distance
at most three. Along a shortest path, with form storage `F=x^T Lx/2`,

\[
 |x_i-x_j|\leq\sqrt{14F}\leq\sqrt{14E_0}=B ,
\]

when the initial excess is bounded by E0. The phase excess is nonnegative
in the retained cell. Hence `abs(u-b)<=B`, and every positive weighted
mean lies in the form's convex hull. This bound has no inverse minimum
capacity factor.

All Reception losses together are at most E0: the same storage budget
pays for every passive event and continuous loss. Applying Cauchy's
inequality to the exact Q-mode loss above gives, for N words,

\[
 \begin{split}
 A_N&=\left\{\frac{2aE_0}{1-2a}
           \sum_{j=0}^{N-1}(2+43v_0\kappa^j)^{-2}\right\}^{1/2},\\
 \sum|\Delta m_{\rm EN}|&\leq A_N,\\
 \sum|\Delta m_{\rm SHA}|&\leq[c(v_N)-c(v_0)]B,\\
 |x_i(t)|&\leq |m(v_0)|+A_N+[1+c(v_N)-c(v_0)]B .
 \end{split}
\]

The c increments telescope even though the forms at successive events
differ: their common range bound is B. The last line covers every
intermediate stage and flow interval through word N, and all future
held-capacity flow after it. It does not spend E0 independently at every
stage or combine overlapping marginal budgets.

The shared
[`_assess_organized_source_reuse_budget`](../../src/tnfr/physics/_organized_source_reuse_budget.py)
retains the admitted `(v_0,kappa,N)` that define the finite sum and computes
outward rational-interval bounds. It avoids constructing that sum's large
common denominator. The graph diameter is derived from shared support.
The word count is an explicit finite work budget, not event admission.

For the registered default Reception factor beta, `a<1/20`.
With `v_0=1`, `N<=56`, `E_0<=1/100` and
actual initial excess strictly below `1/100`,

\[
 A_N<1/8,\qquad B<3/8,\qquad
 c(v_N)-c(1)<43/45.
\]

Thus `abs(m(1))<1/30` suffices for
`abs(x_i(t))<107/120<0.9<1`. This proves both unit rails and the stricter
signed Reception form ceiling with margin. It includes the open source
neighborhood of section 22. Its sharper evaluated finite-sum bound is
available from the shared reader; the simple constants above provide an
independent proof.

Fresh target pressure also remains bounded without a mean assumption.
Each target has degree five, so

\[
 |q_i|\leq\sqrt{10E_0},\quad |S_i|\leq5,\quad
 |p_{\tau,i}|\leq\frac{\sqrt{10E_0}+5\gamma}{4}
                  <0.081<0.15 .
\]

These comparisons retain the strict represented EN policy ceilings from
section 22. The pressure bound applies only when the stored pre-Reception
pressure is actually associated with this independent law and clock;
pressure contraction by IL or stale storage is not that association.

### Finite capacity admission and contextual grammar

Under the present law capacity is held between events. Repetition supplies
no replenishment. With initial unit targets, the registered default represented
factor is exactly

\[
 \kappa=\frac{518151819521135}{562949953421312}.
\]

For the enabled inclusive SHA pre-call floor represented by `0.01`,
`kappa^55` is approximately `0.010454378039190186` and `kappa^56` is
approximately `0.009622445068245949`. Therefore 56 ideal SHA calls satisfy
that gate; the next call fails it. The last accepted jump may end below
the floor because admission checks the pre-state.

The separate
[`_assess_organized_source_reuse_capacity`](../../src/tnfr/physics/_organized_source_reuse_capacity.py)
re-admits represented initial capacity, factor and floor. It distinguishes
exact products from sequential correctly rounded binary64 multiplication.
The latter also permits 56 calls for these parameters, ending at
approximately `0.009622445068245946`. This is a scalar calculation, not 56
executed words. For other parameters rounding can change the count;
underflow rejection, represented stagnation and exhausted finite budgets
are explicit outcomes, not unlimited-reuse certificates.

The bound of 56 is specific to the declared factor, floor, enabled policy
and held-capacity model. It is neither a universal TNFR constant nor
evidence that the organization disappears at that boundary. The final
positive capacity still admits the stated unforced continuation.
H contains no capacity-storage term: attenuation is not expenditure of H,
and zero H work of a pure capacity reset does not derive its supply.
Removing the floor would not prove sustained activity as capacities tend
to zero; limits for positive held capacities do not automatically transfer
to a changing-capacity law with finite accumulated activity. The
[capacity-horizon distinction](../CAPACITY_LOCALIZATION_BALANCE.md) remains.
Moreover, EN and IL of request 57 could execute before SHA rejects.
Each stage is atomic separately; a word is not a rollback transaction.

Reuse retains section 22's supplied established-form history, prior
Coherence, zero destabilizer debt, common kind, explicit IL `event_domain`
and absent optional controllers. Core contextual grammar admits the
SHA-to-EN boundary, including zero instantaneous form with nonempty
established history. Its finite repeated-word check is separate from live
state checks. The optional legacy preference discourages immediate
EN-to-SHA; it does not forbid SHA-to-EN or EN-to-IL-to-SHA. Reception does
not erase Silence's latency fields; latency is not a freezing law.
Continuous phase lifts accumulate actual increments across wrapped
representatives, rather than retaining invalid fixed turns.

The resulting domain supports finite ideal reuse with explicit geometry,
form, pressure and capacity limits. A prescribed schedule still supplies
occurrence. Renewing capacity would require an additional admitted event
or continuous law and its full work, grammar and state consequences;
continuous loss alone is not a replenishment reservoir.

<a id="organized-source-occurrence-nonselection"></a>
### Conditional non-selection of the first word's time

The preceding bounds imply a limited occurrence statement without another
trajectory or a new selector. **They admit more than one timing of the same
word; they do not determine its time.** This is a corollary of the ideal
domain, not a new empirical result or a certificate for rounded executions.

Fix one common initial state in the section 23 domain with initial unit
capacities, `E(0)<1/100` and `abs(m(1))<1/30`. Retain all ninety raw form and
lifted-phase coordinates, their node/port associations, the same ninety
edges, targets `(0,1)`, factors and structural clock tau. Stored pressure is
associated with the independent formula in section 21. Give both alternatives
the same section 22 context: common kind, established `AL, IL` history,
ordinal two, zero debt, prior Coherence, core grammar and explicit IL
`event_domain` policy. Keep the same represented Reception ceilings and
inclusive Silence capacity floor. No controller, custom monitor, expiry
rule or additional event is present.

Waiting means the exact reciprocal flow of section 21 with capacity held.
It advances tau and the continuous phase lifts, without generating glyphs
or changing the supplied grammar context. Before admission, pressure is
evaluated from its own law. A timestamp or a held stale pressure value does
not substitute for that association. No rule in this selected EN/IL/SHA
domain requires a maximum age of the supplied glyph history. This assertion
does not extend to Mutation's measured temporal evidence or another action
that consumes a different history.

For any two fixed finite times `0<=a<b`, define two completions:

1. Continue to tau=a, apply the three ideal maps Reception, Coherence and
   Silence in that order, then continue with the changed held capacities.
2. Continue to tau=b, apply the identical word, then continue with its
   changed held capacities.

Each word has zero structural duration here; its three separately admitted
stages remain ordered, without an intervening flow segment. It is not a
single rollback transaction. Its actual context updates belong to the chosen
maps. Silence's wall-clock telemetry is not the structural timing rule.

Both alternatives are admitted by the stated ideal domain. Before the first
word, the unit-capacity flow preserves the mean and decreases excess, so
waiting preserves geometric identity. The joint mean/form bound for one
word applies to an arbitrary finite initial wait and all its reset prefixes;
the independent fresh-pressure bound retains the strict Reception ceiling.
Coherence uses the declared event-domain policy and the existing nonzero
resultant bounds. Silence sees unit target capacities before its jump, above
the inclusive floor. Waiting leaves the stipulated grammar context intact,
and the admitted word supplies its successive updates. After the word the
same estimates retain identity and form rails for all subsequent flow with
positive held capacity. No infinite event sequence, limiting zero capacity,
or renewal mechanism enters this argument.

The two full states differ at every time `a<tau<b`: target capacity equals

```text
kappa = 518151819521135 / 562949953421312
```

in the first completion and one in the second. These are the same declared
Silence multiplier and held-capacity laws used above; `0<kappa<1`. Capacity
is a retained state coordinate that enters both continued rows. The
difference is therefore not merely a different event label or an unobserved
history association. It proves that the shared continuous law, word maps,
grammar admission and identity requirements do not uniquely determine
occurrence, even from this fully specified common initial state.

The alternatives are **different complete hybrid models**, completed by their
different supplied schedules. This does not establish two trajectories of
one fully specified occurrence law, nor rule out an independently justified
law that selects one schedule. The corollary says nothing yet about equality
of later responses after both words have completed, or equality modulo a
declared observation and symmetry. Ideal waiting and passivity also supply
no new uniform floating-point defect certificate. The frozen represented
handoffs, source archives and their verdicts remain unchanged.

<a id="organized-source-flow-word-timing"></a>
## 24. Timing changes the response after both words have completed

**Conditional counterexample.** Waiting before Reception (EN), Coherence
(IL) and Silence (SHA) need not give the same later response as applying
the word first and waiting under its successor capacity. The distinction
survives common form and phase offsets and is present in both continuous
rows. It holds on the ideal organized-source domain of section 23; it is
not a numerical trajectory, a prescribed finite-horizon separation or a
law selecting when an operator occurs.

### Fixed state, neighborhood and observation

Retain the 45 labeled nodes, ninety unit edges, targets `(0,1)`, positive
capacities, primitive phase lifts and structural clock tau from sections
21–23. Initially every capacity is one. The comparison uses one word,
the shared default represented factors and the section 22 context:
common `organized-source` kind, supplied `AL, IL` history, ordinal two,
zero debt, prior Coherence, initialized core grammar, `raise`, hard unit
rails and explicit IL `event_domain` policy. Retain the actual Reception
ceilings and inclusive Silence floor. There is no extra controller,
custom monitor, held-step nodal validator or state-mutating callback.

Write `h=1/64`, `alpha=2*pi/9` and choose the exact primitive state

\[
 x^*=h(e_0-e_1),\qquad
 \Theta^*_{9c+j}=\psi_j=\alpha(j-4),\quad 0\le c<5,\ 0\le j<9.
\]

This is the winding-one organized carrier, not a globally constant phase.
The closing-edge lift branches are those of the existing acute cell.
Declare the open neighborhood, with the discrete context held fixed,

\[
 U=\{(x,\Theta):\|x-x^*\|_\infty<\varepsilon,
                  \|\Theta-\psi\|_\infty<\varepsilon\},
 \qquad\varepsilon=10^{-5}.
\]

The retained node/port associations are part of this comparison. Its
coordinate symmetries are common form translation and common circular
phase rotation, with the corresponding continuous lifts. They are used
locally where admission holds; arbitrary translations need not preserve
the absolute form rails. Do not rescale the structural clock or relabel
ports to compare endpoints. Fix the observation

\[
 \mathcal O(x,\Theta)=
 \bigl(x_8-x_2,\ (\Theta_8-\psi_8)-(\Theta_2-\psi_2)\bigr).
\]

It removes both common offsets. The section 14 affine signed reflection
also cannot remove the counterexample: the chosen state, both flows and
the word remain in its fixed sector with offset origin zero.

At the center, `S(psi)=0`, `E=6h^2=3/2048` and the form mean is zero.
Each of the five cycles has its retained winding, and every marked angle
is acute. The form edge differences obey `sum_edges abs(Delta x*)=10h`.
For the whole box, quadratic expansion of form storage gives
`F<=6h^2+20h*epsilon+180*epsilon^2`. The carrier phase is stationary:
its linear phase terms telescope around each cycle and vanish on the
matching edges. The second derivative of each cosine potential is at
most one, so its excess is at most `180*epsilon^2`. Hence

\[
 E\le6h^2+20h\varepsilon+360\varepsilon^2<1/100,
 \qquad |m(1)|<\varepsilon<1/30.
\]

Each marked edge angle differs from its carrier angle by less than
`2*epsilon`, retaining the acute branch. Thus U lies in section 23's
larger domain. It is not the section 22 box centered on a frozen response;
no historical response is needed to specify these primitives. Section 23
admits arbitrary finite initial waiting, the complete ideal word and its
positive-capacity continuation with all geometric, form and fresh-pressure
bounds. Silence initially sees unit capacities, above its floor. Each
stage reads one common snapshot at its two targets; the stages are ordered
and separately atomic, with no intervening continuous step.

### Complete local difference

Let L be the actual fine Laplacian, P the diagonal selector of nodes zero
and one, `a=beta/5`, `A=I-a*P*L` and `D=I+(kappa-1)*P`. Set

\[
 S_i(\Theta)=\sum_{j\sim i}\sin(\Theta_j-\Theta_i),\qquad
 z_i(\Theta)=\sum_{j\sim i}\exp(i(\Theta_j-\Theta_i)).
\]

The complete primary word sends `(x,Theta,1)` to `(Ax,C(Theta),D*1)`,
where `C_i(Theta)=Theta_i+lambda*Arg(z_i)` on the targets and C is the
identity elsewhere. On the admitted domain `Re(z_i)>4`, so this local
lifted map is smooth. With `J=DC(Theta)`, its actual derivative is

\[
 (Jv)_i=v_i+\lambda\operatorname{Re}
 \frac{\sum_{j\sim i}e^{i(\Theta_j-\Theta_i)}(v_j-v_i)}{z_i(\Theta)}
 \quad(i=0,1),\qquad (Jv)_i=v_i\quad(i\notin\{0,1\}).
\]

The two flows use the same independently defined pressure and both rows:

\[
 F_D(x,\Theta)=\frac14
 \begin{pmatrix}D[-Lx+\gamma S(\Theta)]\\ \gamma DLx\end{pmatrix},
 \qquad\gamma=(1023\pi)^{-1}.
\]

Here the flow parameter is elapsed structural time tau; capacity is held
on each segment. Differentiate the two endpoints `W(Phi_t(s))` and
`Phi_t^kappa(W(s))` at zero elapsed time. Their complete primary difference
`K=DW(s)F_1(s)-F_D(W(s))` is

\[
 \begin{split}
 K_x&=\tfrac14\{(DLA-AL)x+
                  \gamma[A S(\Theta)-D S(C(\Theta))]\},\\
 K_\Theta&=\tfrac\gamma4[J Lx-DLAx],\\
 K_\nu&=0,\qquad K_{\rm clock}=0.
 \end{split}
\]

In particular, Coherence's derivative cannot be replaced by the identity
merely because its phase displacement vanishes at one state. The IL
pressure retention factor b still has its specified stored-pressure event
effect. Before continuation, refresh `p=(-Lx+gamma*S(Theta))/4` from its
independent law. It is an algebraic state association, not a freely chosen
forcing or an extra pressure evolution law. If pressure is also retained
as a refreshed derived coordinate, its difference is
`K_p=(-L*K_x+gamma*DS(C(Theta))*K_Theta)/4`.

Do not center either endpoint. Its form mean is computed with its actual
inverse capacities; its phase-origin mean first subtracts the carrier
from the primitive lifts and uses those same weights. These are derived
coordinates of the retained state, not disposable integration constants.
The histories, ordinals and lifecycle records follow their separate update
maps on the stipulated discrete context: both completed histories are
`AL, IL, EN, IL, SHA`, with ordinal five. This tangent is taken on the
primary continuous state; those record update maps remain separate.
Silence's wall timestamp is not the structural clock.

### An exact projected coefficient

At the carrier put `c=cos(alpha)` and `R=3+2c`. Each target has two cycle
neighbors and three matching neighbors, so `z_0=z_1=R>9/2`. Let `L_c`
denote the Laplacian with cycle-edge weights c and matching-edge weights
one. Although `C(psi)=psi`,

\[
 J=I-\frac\lambda R P L_c.
\]

For `q=Lx*`, the actual incidence gives

\[
 \begin{gathered}
 q_0=6h,\quad q_1=-6h,\quad q_2=h,\quad q_8=-h,\\
 q_9=q_{18}=q_{36}=-h,\qquad q_{10}=q_{19}=q_{37}=h,
 \end{gathered}
\]

with all other coordinates zero. In particular `P*L*q=40*x*`.
Write `r=1-6a` and `T=(21+19c)/(3+2c)`. Then `Ax*=r*x*` and
`Jq=q-lambda*T*x*`. The complete difference at this state is

\[
 \begin{split}
 K_x&=\tfrac14[-6a q+\{40a-6r(1-\kappa)\}x^*],\\
 K_\Theta&=\tfrac\gamma4[6a q+
                         \{6r(1-\kappa)-\lambda T\}x^*].
 \end{split}
\]

Both inverse-capacity form means, before and after the word at this
center, are exactly zero, and both carrier-subtracted phase-origin means
are zero. This does not impose a zero-mean restriction on U. The fixed
observation yields

\[
 D\mathcal O\,K=(3ah,-3\gamma ah),\qquad
 3ah=\frac{815555754282159}{360287970189639680}>0,
\]

using the shared represented default
`beta=271851918094053/1125899906842624`. The other factors are
`lambda=5404319552844595/18014398509481984` and
`kappa=518151819521135/562949953421312`; their target-row contributions
were retained even though this observation eliminates them at the center.

Both flows and W are smooth locally. Their endpoints agree at `t=0`,
and Taylor expansion therefore gives

\[
 \mathcal O(W\Phi_t(s^*))-\mathcal O(\Phi_t^\kappa W(s^*))
       =(3ah,-3\gamma ah)t+O(t^2).
\]

The first component is strictly positive for all sufficiently small
positive t. Both words have then completed, both capacities equal D and
both structural clocks have advanced by the same amount. The difference
is an internal contrast, not a common offset or the earlier interval's
capacity mismatch. Continuity also retains a nonzero projected derivative
on some neighborhood of the center inside U; no numerical size for that
smaller neighborhood or time interval is asserted. The result does not
claim persistent asymptotic separation or a finite observation margin.

### Detached implementation and scope

The private
[`_assess_organized_source_carrier_timing`](../../src/tnfr/physics/_organized_source_timing.py)
reader computes the local difference at the exact carrier for admitted
rational or represented form coordinates. It reuses the shared support,
factor admission and rational interval arithmetic. It retains the exact
form difference, the full phase difference as rational coefficients of
`cos(alpha)`, its outward enclosure, both form means and successor
capacities. It does not approximate the carrier by binary64 phases or
hold phases fixed during either hypothetical flow.

[Independent tests](../../tests/physics/test_organized_source_timing.py)
rebuild the incidence, verify both rows and the nonidentity phase
derivative, and exercise translation and zero-form controls, an asymmetric
nonzero successor mean, invalid domains and prohibited runtime execution.
No event, integrator, source producer or frozen response is evaluated.
These checks support the algebraic implementation; smoothness and the
admission proof above supply the local mathematical conclusion.

Thus event timing can be necessary predictive information even after the
same complete word has occurred. The supplied state, continuous law and
grammar do not select that timing. A quantified later response or an
occurrence mechanism would need its own bounded question and premises;
neither follows from this counterexample.
