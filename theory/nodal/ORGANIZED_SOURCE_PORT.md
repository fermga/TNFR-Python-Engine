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
