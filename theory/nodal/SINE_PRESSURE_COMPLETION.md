# Pressure completion: compatibility, balances and operator scope

This owner classifies extensions of the maintained weighted conservative
contact after the [IL/OZ handoff](SINE_COLLECTIVE_PRESSURE_HANDOFF.md).
The result is a conditional incompatibility between specified requirements,
with explicit countermodels when a requirement is dropped. It neither selects
a pressure-memory law nor requires every organization to realize every operator.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns task status. No trajectory or new operator response is evaluated here.

<a id="pressure-completion-premises"></a>
## 1. State, units and the question being tested

Use the [actual-law state](SINE_COLLECTIVE_PRESSURE_HANDOFF.md#collective-pressure-handoff-model),
including member-associated boundary weights, fixed bare support U, held
positive capacities N=diag(nu), signed form x and circular phase theta with
declared continuous lifts. Put D=diag(d), with positive bare degrees, and

\[
 M=N(\pi D)^{-1},\quad q=L_Wx,\quad
 S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i),\quad
 \Phi=(\pi D)^{-1}S,\quad \rho_i=d_i/\nu_i.
\]

The current contact has connected positive symmetric conductances. Algebraic
balances below only need symmetric fixed conductances and reciprocal bare
support; connected positive weighted support is used explicitly for kernel
and equilibrium conclusions. Zero capacity is outside the inverse-capacity
mean argument. These formulas do not substitute the native neighbor-argument
pressure or the default mixed pressure for this selected sine law.

An independently specified driver is a law specified before evaluating its
response. It need not be an independent dynamical coordinate. The current
model independently specifies `p=Phi(y)` for `y=(x,theta,nu,U,W)`. Conversely,
declaring an independent pressure coordinate requires its own evolution and
initialization. The [parameter foundation](../NODAL_PARAMETER_FOUNDATIONS.md)
and [pressure-state closure boundary](../FUNDAMENTAL_THEORY.md#environment-substrate-and-pressure)
retain this distinction; a change of coordinates alone adds no new freedom.

To study that freedom conditionally, define the pressure deviation
`r=p-Phi(y)` and consider

\[
 \dot x=MS+Nr=Np,\qquad \dot\theta=Mq,\qquad \dot r=f(y,r).
 \tag{1}
\]

The nodal row is retained. The phase row, clock, support and capacity premises
are also retained; f is not selected. A complete extended model would still
need f, its admissible domain, event maps, initialization and any inputs.
Operator context k remains separate. If a future law consumes k, it must
declare that dependency rather than assuming pressure memory from a history
record. This is a fine-state analysis, not a closed scalar macro-node law.

Use the original normalized structural chart and clock. In general
`[r]=[p]=[x]/([nu][t])` and `[f]=[r]/[t]`. Added storage must have the units
of H below. A quadratic pressure storage needs its own dimensional coefficient;
pressure squared is not automatically physical energy. Every new evolution
row must transform under a clock change. No laboratory-unit bridge is supplied.

## 2. Constitutive graph and pressure-coordinate evolution

Suppose an event starts on `p=Phi(y)`, leaves y unchanged and changes p by a
nonzero amount. Its output cannot remain on that same single-valued graph:
`Phi(y^+)=Phi(y^-)=p^-`, whereas `p^+!=p^-`. An off-graph event could instead
land on the graph; the obstruction concerns preserving it from a point on it.
Refreshing pressure projects back onto this constraint, as already proved for
OZ. This is an algebraic statement about the chosen constitutive law.

The [existing derivative](SINE_COLLECTIVE_PRESSURE_HANDOFF.md#collective-pressure-memory-boundary)
uses the bare cosine Hessian K(theta) and gives, under (1),

\[
 \dot p=f(y,r)-(\pi D)^{-1}K(\theta)Mq.
 \tag{2}
\]

Thus `r_dot=-gamma*r` and `p_dot=-gamma*(p-Phi(y))` are different laws.
The second has
`r_dot=-gamma*r+(pi*D)^(-1)*K*M*q` and generally leaves `r=0`.
Held pressure `p_dot=0` has the same constitutive-drift term. Holding pressure
is not holding its deviation from an evolving Phi. Extending the derivative
of Phi away from its graph without accounting for this term would silently
change the model. Unique invariant-flow claims additionally require appropriate
well-posedness; the pointwise tangency condition alone is `f(y,0)=0`.

<a id="pressure-completion-balances"></a>
## 3. Exact means, power and the retained OZ endpoint

Let `R=sum_i rho_i`. Reciprocal support gives `sum S=0` and `sum q=0`.
On a continuous phase lift, (1) therefore implies

\[
 \frac{d}{dt}\frac{\rho^Tx}{R}=\frac{d^Tr}{R},\qquad
 \frac{d}{dt}\frac{\rho^T\theta}{R}=0.
 \tag{3}
\]

Keeping the original weighted form mean requires `r in E=d^perp` at every
time. An evolution intended to preserve E must have `d^T f=0` there, and an
event must land in E. Neither property follows from operator admission.

For the existing structural storage

\[
 H=\tfrac12x^TL_Wx+\sum_{\{i,j\}\in U}(1-\cos(\theta_j-\theta_i)),
\]

the gradients are `H_x=q`, `H_theta=-S`. Hence

\[
 \dot H=q^T(MS+Nr)-S^TMq=q^TNr.
 \tag{4}
\]

This is the [existing off-graph work identity](SINE_COLLECTIVE_PRESSURE_HANDOFF.md#collective-pressure-memory-boundary),
now combined with the mean constraint. It identifies a power pair within this
model; it is not an electrical or physical-energy identification. Preserving
the nodal identity does not by itself preserve these means or storage.

### Detached consequence of the already retained action

The [frozen response](../../docs/assets/collective_pressure_handoff/correction_v2/response.json)
contains the same y before and after raw OZ. Re-admit its primitives and
rebuild Phi, degrees and capacities through the weighted-law reader. With
`j=p_OZ_raw-p_IL_completed`, exact represented arithmetic gives

\[
 R=104,\qquad
 d^Tj=\frac{33762929380696061}{147573952589676412928}>0.
\]

Stored pressure has a materialization residual, so j must not be substituted
for the ideal deviation without that residual. Independently enclosing
`r=p_OZ_raw-Phi(y)` gives the hypothetical rate

\[
 \frac{d^Tr}{R}\ \in\
 [2.1998702728\,10^{-6},\ 2.1998702782\,10^{-6}],
 \tag{5}
\]

where displayed endpoints round the retained rational bounds outwards.
Consequently consuming this raw pressure in (1) would immediately change the
old weighted form mean. A mean-preserving extension cannot admit this output
unchanged. Projecting it onto E, adding a compensating input or redefining the
conserved quantity would be additional premises, not the recorded OZ action.
The actual form gradient is nonzero after the inherited contact word; the
preparation's phase-axis name cannot replace inspection of that endpoint.

Equation (5) is a new read-only consequence of saved state, not an executed
held-pressure interval. The actual experiment immediately refreshed pressure
and did not follow (1) off the constitutive graph. Its original **8/9** verdict
and unresolved ideal-phase accuracy certificate remain unchanged. The
[retained evidence reader](../../tests/research/test_collective_pressure_handoff_evidence.py)
owns source associations and primitive reconstruction; hashes associate bytes,
not execution authenticity.

<a id="pressure-completion-obstruction"></a>
## 4. An obstruction to smooth passive extensions preserving the old law

Fix y0 with `q(y0)!=0`. Suppose all of the following are required:

1. The two form/phase rows in (1), with held support, capacity and clock.
2. A two-sided relative neighborhood of `r=0` in `E=d^perp` is admitted,
   and f takes values in E there.
3. `f(y,0)=0` and f is continuous near `(y0,0)`.
4. Added storage V(y,r) is C2, locally nonnegative, and `V(y,0)=0` for every
   nearby y. Derivatives in r are relative to E.
5. Autonomous passivity `d(H+V)/dt<=0` throughout that neighborhood.

**These requirements are incompatible.** Passivity and zero added storage on
the original graph are additional hypotheses, not universal TNFR axioms.

**Proof.** Since zero minimizes V in the pressure directions and
`V(y,0)` vanishes identically, `V_y(y,0)=V_r(y,0)=V_yr(y,0)=0`.
At fixed y0 and any v in E, C2 regularity gives

\[
 V_y(y_0,\epsilon v)=o(\epsilon),\qquad
 V_r(y_0,\epsilon v)=O(\epsilon).
\]

Continuity and `f(y0,0)=0` give `f(y0,epsilon*v)=o(1)`. The y rows are
bounded nearby, so `V_dot=o(epsilon)`. Let P be the orthogonal projector
onto E and take `v=PNq(y0)`. This v is nonzero: otherwise `Nq=c*d`, and
`0=sum q=c*sum(d_i/nu_i)` forces c=0 and then q=0. Therefore (4) gives

\[
 \frac{d}{dt}(H+V)(y_0,\epsilon v)
 =\epsilon\|PNq(y_0)\|^2+o(\epsilon)>0
\]

for sufficiently small positive epsilon, contradicting requirement 5.
The C2 argument needs only the displayed little-o remainder, not an assumed
quadratic remainder for V_y. No quadratic ansatz or positive-definite storage
Hessian is necessary. The pointwise contradiction does not require integrating
a trajectory. For actual flow invariance, supplement tangency with suitable
local uniqueness assumptions. The same proof applies without the mean
constraint by taking E to be the full pressure space. QED.

For a smaller permitted linear space E, the first-order obstruction occurs
when `P_E Nq!=0`. It does not automatically apply to one-sided cones, isolated
event outputs, nonsmooth storage, independent reservoirs, domains lacking those
directions or a changed phase row. It proves neither a unique extension nor
the impossibility of pressure memory in TNFR.

## 5. Zero instantaneous work and moving constraints

At q=0 the linear power vanishes; full equilibrium at r=0 additionally needs
S=0. On connected positive weighted support, `q_dot=L_W M S` cannot vanish
when S is nonzero. Indeed `L_W M S=0` would give `MS=c*1`, and `sum S=0`
forces c=0. A constant-form state with S!=0 thus leaves q=0 under the
original law; its phase rate is zero at the initial instant while form starts
moving. The theorem above does not claim a positive first-order
power at such an instant.

Instantaneously work-neutral, mean-preserving deviations lie in

\[
 E(y)=\ker d^T\cap\ker(Nq)^T.
\]

Its dimension is n-2 for q!=0 and n-1 at q=0. The two constraints remain
true along a differentiable solution only if

\[
 d^Tf=0,\qquad
 [L_W(MS+Nr)]^TNr+(Nq)^Tf=0.
 \tag{6}
\]

This is an additional evolution obligation. At a base-law equilibrium q=S=0,
any nonzero r in `d^perp` makes the second expression strictly positive on
connected positive weighted support: `(Nr)^T L_W(Nr)>0`. Otherwise Nr would
be constant, and `d^Tr=0` would force r=0. Equivalently, H has zero first but
strictly positive second derivative at that state. Instantaneous neutrality
does not certify a persistent work-neutral channel. Disconnected conductance
support has a larger kernel and needs separate component constraints.

<a id="pressure-completion-alternatives"></a>
## 6. Explicit alternatives and the premise each changes

The following are mathematical controls delimiting the obstruction, not
installed laws or newly selected TNFR coefficients. They retain (1) unless
an altered row is stated. Let `P=I-dd^T/(d^Td)` and r lie in E.

**Retain the constitutive graph and permit exchange work.** Setting `f=0`
preserves r and both means but leaves `H_dot=q^TNr` of either sign.
Residual relaxation `f=-gamma*r`, gamma>=0, also preserves r=0. With
`V=kappa*||r||^2/2`, kappa>0, its total storage rate is

\[
 \dot H+\dot V=q^TNr-\kappa\gamma\|r\|^2.
\]

No fixed finite damping gain controls every sufficiently small two-sided
deviation at q!=0. Adding damping alone does not resolve the stated conflict.

**Permit reciprocal exchange to leave the original constitutive graph.** Set

\[
 f=-\kappa^{-1}PNq-\gamma r,\qquad
 V=\tfrac\kappa2\|r\|^2.
\]

Both means are preserved and `d(H+V)/dt=-kappa*gamma*||r||^2<=0`, but
`f(y,0)!=0` at q!=0. The original conservative flow is not an invariant
subsystem. This transfers the cancellation method of the
[existing auxiliary reservoir](ORGANIZED_PARENT_RESERVOIR_EXCHANGE.md#organized-parent-reservoir-exchange),
with a fresh derivation for N and E. Its 36-node positive-loss coefficients,
clock, initialization and frozen negative response do not transfer here.

**Change the phase law on a restricted domain.** When S!=0, put
`a=P_(rho^perp)*S`. Then a!=0 since `sum S=0` and `sum rho>0`.
Adding `h=(q^TNr)*a/||a||^2` to the phase row preserves its weighted mean and
supplies `-S^Th=-q^TNr`. With residual relaxation and the quadratic V above,
total storage is nonincreasing and r=0 is invariant. This has changed the
phase law and is undefined at S=0. It supplies no global smooth completion;
at S=0 a finite phase correction cannot cancel nonzero q^TNr through H_theta.

**Retain an independently initialized reservoir.** Adding e>0 with
`e_dot=-q^TNr`, `r_dot=0` conserves H+e while e stays in its admitted domain.
This leaves the zero-baseline storage premise and introduces initial content,
depletion and event-work obligations. Neither global positivity nor a natural
reservoir follows. Previous dissipative loss is not such a reservoir without
a declared retained coordinate and law.

**Consume pressure over a declared interval.** The
[THOL handoff calculation](../THOL_PRESSURE_FEEDBACK.md#4-refresh-order-determines-whether-the-pressure-increment-reaches-epi)
shows how a held-pressure interval can leave a form change after later refresh.
Its duration, complete rows and work are part of a different hybrid model.
It needs no permanently independent pressure coordinate, but does not change
the outcome of the immediate-refresh IL/OZ word. Held pressure generally
leaves r=0 and need not preserve the old weighted form mean.

## 7. Events, identity and symmetry obligations

For an admitted y-preserving event starting at r=0, nonnegative zero-baseline added
storage obeys `Delta(H+V)=V(y,r^+)>=0`. Strict positivity requires an accounted
event source, another changing coordinate or reservoir depletion. A
semidefinite V can have nonzero zero-cost directions, whose subsequent
admission is still separate. Continuous passivity neither pays event work nor
selects its occurrence. OZ's context changes must remain available to later
grammar and event consumers even when continuous futures coincide.

The existing identity barrier uses the original H and its conservative
continuation. It does not transfer merely because an extension preserves the
nodal row. If an extension independently establishes nonincreasing H+V,
V>=0, initial identity membership `Z(0)<1/16` and
`H(y0)+V(y0,r0)-H_*<1/6400`, the unchanged pointwise barrier excludes a
first exit while the admitted extended solution exists continuously. Keep
its support, reference, phase-lift conventions and applicable domain/mean
hypotheses unchanged. Semidefinite V does not ensure pressure boundedness;
global continuation needs separate bounds on all added state. Conservation of H+V
alone does not bound H by its old initial value when V initially stores work.
Formation, attraction and a new collective identity still need their own proof.

Common form shifts and common phase rotations leave q,S and Phi unchanged.
A full extension must make f and storage compatible with any claimed symmetry.
Joint relabeling transports d,N,W,r, constituent marks and event context; the
projector P transforms with them. An added coordinate, reservoir or held
interval also requires its own preparation and clock declaration. None of
these conditional symmetries identifies physical gauge freedom.

## 8. Consolidated inference and verification

The current algebraic pressure law and its OZ refresh neutrality are consistent.
An independent pressure extension is possible, but it cannot freely retain
all of the theorem's rows, mean constraints, invariant constitutive graph,
smooth zero-baseline added storage and autonomous passivity at once. The
retained raw OZ output additionally fails the old form-mean constraint if
consumed unchanged. These are useful model restrictions, not a reason to force
OZ persistence or replace the nodal equation.

Other operator mechanisms can be studied under the existing complete law
without selecting one of the alternative memory laws above, provided their
actual consumed state and continuation requirements are admitted. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone selects such work; this classification introduces no queue.

The [exact algebra and retained-state controls](../../tests/physics/test_collective_pressure_completion.py)
compare whole-edge derivatives with complete nodal rows on varied rational
supports/capacities, test the projected positive-power direction and explicit
countermodels, and retain equilibrium/disconnected-support boundaries. Exact
quarter-turn phase fixtures keep separate coefficients of 1 and 1/pi, so
the clock is not silently rescaled. The read-only OZ control rebuilds Phi
from captured primitives and retains its materialization residual. These
controls support the derivations and finite consequence; samples do not prove
the universal theorem. [Testing](../../TESTING.md) owns the combined selection.
