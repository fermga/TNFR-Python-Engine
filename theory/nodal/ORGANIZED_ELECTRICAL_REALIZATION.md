# Complete-state electrical comparison of the organized sine law

<a id="organized-electrical-realization"></a>

## Question, reference and scope

The [charge observation owner](ORGANIZED_CHARGE_OBSERVATION.md) excludes a
specified charge-density-only closure. Here phase remains part of the state.
The question is whether the same organized quarter law, its retained-reference
charge inventory and its full storage can be realized by a conventional passive
network of capacitors, resistors and ideal sinusoidal Josephson junctions.

This is a conditional constitutive comparison, not a physical identification
or a derivation of the Josephson relations. The reference uses those relations
as independently supplied physical laws. A formal positive quadratic storage
is not sufficient to establish electrostatic realizability in physical island
charge and voltage coordinates.

The result below excludes a declared **direct island-charge and phase map**
on C9 and the same-slot 54-node contact, even allowing a dense passive
capacitance network, a ground reference, unobserved common phase and statically
eliminated passive capacitor nodes. A separate constructive result realizes
the exact organized six-child carrier by passive components after retaining
one collective charge and phase per child. The direct obstruction does not
classify arbitrary collective charge maps, ideal-transformer
coordinate mixing, dynamic hidden circuitry, active synthesis, or a map
defined only on a lower-dimensional prepared trajectory. Those alternatives
need their own independent observation, law and work admissions.

This detached proof changes no execution law, consumes no newly evaluated
trajectory and installs no operator or event selector. Its organized states
remain conditional test cases of the supplied complete law, as required by
the [NFR ontology](../EMERGENT_ONTOLOGY.md#research-target-physical-properties-of-collective-patterns).
The [research program](../RESEARCH_PROGRAM.md#research-current-question)
alone selects subsequent work.

## Independent passive electrical reference

Orient each junction edge once, and let \(B\) be its node-edge incidence
matrix, so \(\delta=B^T\vartheta\). There is no supplied magnetic flux,
phase frustration, external current source or time-dependent component.
Electrical island phases are circular; a local continuous lift defines their
derivatives without making the lift an additional observable charge.
Geometric inductance, electromagnetic retardation, thermal noise and quantum
fluctuations are omitted in this classical lumped reference. Component laws
and that approximation require separate physical admission for an apparatus.

The ideal single-junction reference is

\[
 I=I_c\sin\delta+\frac VR+C\frac{dV}{dt},
 \qquad \frac{d\delta}{dt}=\frac V{a_0},
 \qquad a_0=\frac{\hbar}{2e}>0.
 \tag{1}
\]

Equations (1) and (2) of [Babenko et al., *Characterization of a Josephson
Junction Comb Generator* (2020)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=929317)
give the current balance and voltage-phase relation. The exact sine law is
an ideal reference premise; admitting it does not establish that every
physical junction obeys it on every regime. The quantum constant in (1)
comes from the reference and is not obtained from TNFR.

Kirchhoff balance and capacitor storage assemble the reference into

\[
 q=C V,\qquad
 \dot q=-G V-B\mathop{\rm diag}(I_{c,e})\sin(B^T\vartheta),
 \qquad
 \dot\vartheta=V/a_0,
 \tag{2}
\]

with the appropriate common-potential gauge for a floating network.
Here \(q\) is physical island charge relative to any independently fixed
background offset, \(V\) is voltage, \(C\) is a symmetric capacitance
matrix and \(G\) a symmetric passive conductance matrix. Dots in this owner
mean laboratory time \(t\), in contrast to the structural prime below.
All junction critical currents satisfy \(I_{c,e}>0\). The storage and loss
are

\[
 E=\frac12 V^T C V+
       a_0\sum_e I_{c,e}[1-\cos\delta_e],
 \qquad
 \dot E=-V^T G V\le0.
 \tag{3}
\]

Indeed \(\dot E=V^T\dot q+a_0\sum_e I_{c,e}\sin\delta_e\,
\dot\delta_e\); substituting (2) cancels the junction terms. This derivation
fixes the orientation and restoring sign rather than relying on a visually
similar sine term.

Ordinary passive two-terminal capacitors give

\[
 C_{ij}=-c_{ij}\le0\quad(i\ne j),\qquad
 C\mathbf1=c_g\ge0,
 \tag{4}
\]

where \(c_{ij}\ge0\) is a mutual capacitor and \(c_g\) contains capacitors
to the reference. The same sign, symmetry and row-sum conditions hold for
the grounded Maxwell capacitance matrix of conductors in the reference's
fixed linear electrostatic regime; see
[Díaz and Herrera, *The positivity and other properties of the matrix of
capacitance: physical and mathematical implications*](https://arxiv.org/pdf/0903.0600v4),
equations (3)-(8). A floating complete matrix has zero row sums and the
common-potential null mode. A grounded restricted matrix is positive definite
under the reference's usual connectedness assumptions. These are conditions
on physical conductor charges and potentials, not on arbitrary linear
coordinates of a positive energy.

The two primary sources were reviewed on 2026-10-10; the capacitance reference
is pinned to arXiv version 4. Only their constitutive relations and matrix
conditions are used here, not their measured responses as TNFR evidence.

The proof below only needs symmetry, positive energy, (4), and the exact
laws (2). No numerical material value, apparatus or experimental dataset is
admitted.

## Complete TNFR state and the declared direct map

The separate [operational identification](ORGANIZED_ELECTRICAL_IDENTIFICATION.md)
derives the remaining component relation after independent calibration and
the finite-work boundary for the source's phase interventions. It changes
neither this reference nor the exact construction below.

On a fixed connected simple unit graph with \(n\ge2\), write

\[
 L=BB^T,\quad S(\theta)=-B\sin(B^T\theta),\quad
 x'=\frac{-Lx+\gamma S(\theta)}4,\quad
 \theta'=\frac{\gamma Lx}4,\quad
 \gamma=\frac1{1023\pi}.
 \tag{5}
\]

Prime denotes the structural clock \(\tau\). Forms are signed real,
phases are circular, capacities are held at one, and there is no forcing or
event. The [organized interaction owner](ORGANIZED_EXTERNAL_INTERACTION.md#1-complete-state-support-and-balances)
proves

\[
 H=\frac12 x^T Lx+\sum_e[1-\cos(B^T\theta)_e],\qquad
 H'=-\frac14\|Lx\|^2,\qquad (\mathbf1^Tx)'=0.
 \tag{6}
\]

Retain a fixed reference \(r\), and specify before comparing responses

\[
 q=\kappa(x-r\mathbf1),\qquad
 \vartheta=\theta,\qquad
 \tau=\tau_0+b t,\qquad \kappa>0,\quad b>0.
 \tag{7}
\]

These are supplied observation and clock scales, not fitted consequences.
With the normalized coordinates of (5), \(\kappa\) has units coulombs,
\(b\) inverse seconds and \(a_0\) volt-seconds. An independently declared
form unit must be retained in \(\kappa\); the numerical normalization is
not a laboratory-unit bridge.

Start on the invariant neutral leaf \(\mathbf1^Tq=0\), equivalently
\(r=\bar x\) for that preparation, and let
\(P=I-\mathbf1\mathbf1^T/n\). This is a stated restriction, not universal
neutrality. Keeping the reference \(r\) is necessary when comparing inventory
between regions or supports. The map on this leaf is invertible after the
same common-phase convention is adopted on both sides. Simultaneously
reversing charge and phase, or consistently permuting nodes and support,
does not change the conclusions. An independent sign flip of only one
coordinate changes the required capacitance sign or junction restoring sign.

The map is tested on an open set of form variations in the neutral leaf and
independent circular phase variations. It is not merely a match at one
equilibrium or along one response trace. The linear phase-row restriction
therefore applies in an arbitrarily small full-state neighborhood of an
organized twist as well as on the full leaf.

## Necessary voltage, capacitance, conductance and energy scales

The phase row and the reference's voltage-phase relation force

\[
 V=d Lq,\qquad d=\frac{a_0 b\gamma}{4\kappa}>0.
 \tag{8}
\]

Let \(L^+\) denote the Moore-Penrose inverse, with
\(LL^+=L^+L=P\) and \(L^+\mathbf1=0\). A floating capacitance satisfying
\(q=CV\) for every neutral \(q\) must be exactly

\[
 C_* = c_*L^+,\qquad c_* = \frac1d
                =\frac{4\kappa}{a_0 b\gamma}>0.
 \tag{9}
\]

This is necessary, not a convenient choice: multiplying \(q=dCLq\)
on the neutral subspace by its inverse gives (9). Symmetry and the zero
common mode fix its full floating extension. Positive definiteness of
\(L^+\) on \(\mathbf1^\perp\) establishes positive quadratic energy,
but does not yet establish (4).

The form row becomes

\[
 \dot q=-\frac b4 Lq-\frac{b\kappa\gamma}4 B\sin(B^T\vartheta).
 \tag{10}
\]

Consequently each original junction needs
\(I_c=b\kappa\gamma/4\), and the conductance must act on every
relative voltage as

\[
 G|_{\mathbf1^\perp}=g_*I,\qquad
 g_* = \frac{\kappa}{a_0\gamma}>0.
 \tag{11}
\]

For a floating resistor network this gives \(G_*=g_*P\), implemented
formally by an all-pairs resistor graph with conductance \(g_*/n\) on
each pair. Unless the original support is complete, this dissipative
support differs from the TNFR adjacency. Equality of nodal divergence or
regional total current does not identify its individual physical wires
with the original structural edges. Ground shunts \(G=g_*I\) give the
same neutral voltage action but also require explicit ground-current
accounting. A same-sparse-support resistor realization is already excluded
by (11), independently of the capacitance issue.

If one temporarily permits the formal positive matrix (9), the complete
field and storage match exactly:

\[
 E=K H,\qquad K=\frac{a_0 b\kappa\gamma}4>0,\qquad
 \dot E=-\frac{Kb}4\|Lx\|^2=-V^TG_*V.
 \tag{12}
\]

The capacitor term follows from \(q^TC_*^+q/2=dq^TLq/2\); the junction
term has the same factor \(K\). The scales retain two unselected positive
parameters, \(\kappa\) and \(b\), even with \(a_0\) fixed by the physical
reference. Neither complete field matching nor energy matching determines
a measurement calibration.

Thus (9)-(12) are an exact **formal** charge-phase representation of (5).
The remaining question is whether its capacitance is a physical passive
Maxwell capacitance in these particular charge coordinates.

## C9 gives an exact passive-capacitance obstruction

<a id="direct-island-capacitance-obstruction"></a>

For the unit cycle \(C_n\), the entry of \(L^+\) at graph distance
\(d_{ij}\) is

\[
 (L^+)_{ij}=\frac{n^2-1}{12n}
                -\frac{d_{ij}(n-d_{ij})}{2n}.
 \tag{13}
\]

This identity is checked without a spectral approximation: the displayed
circulant has zero row sums, and applying the cycle's second difference
gives \(LL^+=P\). For C9 its diagonal and distances one through four are

\[
 \frac{20}{27},\quad \frac8{27},\quad
 -\frac1{27},\quad -\frac7{27},\quad -\frac{10}{27}.
 \tag{14}
\]

In particular, every adjacent off-diagonal entry of \(C_*\) is
\(8c_*/27>0\), whereas every passive Maxwell off-diagonal entry is
nonpositive. No positive charge or clock calibration can remove that sign.
Node relabeling only relocates it; simultaneous charge and phase reversal
leaves the capacitance relation unchanged.

**Conditional obstruction.** The direct map (7), with the complete quarter
law on C9 and independent neutral form variations, has no realization of
(2) by ordinary passive capacitors or electrostatic conductor capacitance.
The contradiction already occurs in the phase/voltage/charge relations;
fitting a current response, changing resistor values, or adding more response
samples cannot repair it. It also holds near an acute organized C9 twist,
because the required linear relation (8) does not depend on its phase.

This is not an obstruction on every graph. On \(K_n\),
\(L=nP\) and \(L^+=P/n\); all off-diagonal entries are
\(-1/n^2\). Then \(C_*=c_*P/n\) is a passive all-pairs capacitor network,
\(G_*=g_*P\) is a passive all-pairs resistor network and the positive
junction values above satisfy the complete equations and storage on the
neutral leaf. This mathematical control shows why positivity, graph support
and coordinate interpretation must be checked rather than globally rejecting
the whole reference family. It is a supplied circuit construction, not a
derivation of matter or a transfer of C9 acquisition certificates to \(K_n\).

The same direct fine-node obstruction applies to the actual same-slot support
\(C_9\square W_6\), without transferring a C9 subgraph conclusion by
assumption. Let \(J\) copy a nine-vector into all six children. Then
\(L_{54}J=JL_9\), and uniqueness of the centered inverse gives
\(L_{54}^+J=JL_9^+\). In node labels \((c,j)\), for every child \(c\),

\[
 \sum_{d=0}^{5}(L_{54}^+)_{(c,j),(d,j+1)}=\frac8{27}>0.
 \tag{15}
\]

Every summand is off-diagonal, so at least one is positive. Thus the necessary
fine-node capacitance again violates (4). This product argument applies to
the same-slot support; it is not asserted for every alternative registration.

## Grounding, an unobserved common phase and eliminated capacitor nodes

<a id="ground-and-neutral-capacitance-boundary"></a>

Suppose a grounded passive capacitance \(C\) obeys (4), and retain only
relative electrical phase. Its voltage may then contain an unobserved common
component: \(V=v+v_0\mathbf1\), with \(v=PV=dLq\). Write
\(c=C\mathbf1\ge0\), \(s=\mathbf1^Tc\). On the neutral charge leaf,

\[
 0=\mathbf1^Tq=c^Tv+s v_0.
 \tag{16}
\]

If \(s>0\), eliminating \(v_0\) gives

\[
 q=C_0v,\qquad C_0=C-\frac{cc^T}s,\qquad
 C_0\mathbf1=0,\qquad (C_0)_{ij}\le0\quad(i\ne j).
 \tag{17}
\]

The last inequality follows directly from \(c_i,c_j\ge0\) and (4).
Equation (8) again forces \(C_0=c_*L^+\), contradicting (14).
If \(s=0\), nonnegative \(c\) implies \(c=0\), so \(C\) is already
floating and (9) applies. Thus an unobserved common electrical rotation
and a passive ground capacitance do not evade the neutral-sector result.
This argument is stronger than fixing zero common voltage in advance.

Under strict absolute phase matching, symmetry separately permits only
\(C=c_*L^++c_0\mathbf1\mathbf1^T/n\); positive energy requires
\(c_0\ge0\), which makes the positive C9 off-diagonal entries no smaller.
The relative-phase proof above does not rely on that stricter convention.

Static removal of hidden passive capacitor nodes also preserves the
obstruction. Partition a passive matrix into retained and hidden nodes.
When its hidden block \(C_{hh}\) is invertible, eliminating held hidden
charges gives the incremental retained matrix

\[
 C_{\rm eff}=C_{rr}-C_{rh}C_{hh}^{-1}C_{hr}.
 \tag{18}
\]

A grounded principal block of a connected passive capacitor network has
an entrywise nonnegative inverse: the discrete maximum principle rules out
a negative minimum of the solution to \(C_{hh}u=f\ge0\), and invertibility
removes a constant isolated null component. Since \(C_{rh}\) and
\(C_{hr}\) have nonpositive entries, the subtracted matrix in (18) is
entrywise nonnegative, so the reduced off-diagonal entries remain nonpositive.
The reduced row sums remain nonnegative; this also follows by treating the
ground as a retained vertex of the augmented zero-row-sum capacitor graph.
Then (17) applies if its common potential is removed. Held nonzero hidden
charges change only an affine offset, not this incremental matrix.

An isolated singular hidden block must first have its gauge or disconnected
component declared; an inverse cannot be silently substituted for it.
Dynamic hidden charges are different: eliminating them can introduce state
or memory and is outside the static same-state realization established here.

## Exact collective carrier: a passive realization exists

<a id="organized-collective-electrical-realization"></a>

The direct fine-node obstruction is not the complete collective decision.
On the same-slot support, retain the already admitted exact carrier

\[
 x_{c,j}=X_c,\qquad
 \theta_{c,j}=\alpha j+\Phi_c\pmod{2\pi},\qquad
 \alpha=2\pi/9,\qquad c=0,\ldots,5,\quad j=0,\ldots,8.
 \tag{19}
\]

The first five children form the host ring and child five is the donor.
Their contact graph is the wheel \(W_6\). Each child retains its supplied
nine-node identity, marking and internal twist. This is a restricted invariant
manifold, not a claim that every recovered fine state lies on it.

Internal form differences vanish and the two internal sine contributions
cancel at every vertex. Each same-slot external neighbor contributes the
same term at every slot. Therefore the **full nonlinear** fine law restricts
exactly to

\[
 X'=\frac{-L_WX+\gamma S_W(\Phi)}4,\qquad
 \Phi'=\frac{\gamma L_WX}4.
 \tag{20}
\]

Here \(S_W=-B_W\sin(B_W^T\Phi)\). There is no tangent approximation,
discarded memory or degree renormalization in (20). Common form and phase
translations, circular covariance and wheel relabelings retain the declared
mean/reference and marked-child conventions. Internal C9 winding stays one
exactly on (19). Maintenance of an outer winding under a particular response
is a separate condition; closure does not prove that every outer winding is
preserved.

For each child, preserve the already declared **aggregate** inventory,

\[
 q_c=\kappa\sum_{j=0}^{8}(x_{c,j}-r)
        =9\kappa(X_c-r),\qquad
 \vartheta_c=\Phi_c.
 \tag{21}
\]

The electrical island now represents a whole organized child, not one of its
fine nodes. Its phase is the retained common phase relative to the supplied
slot carrier \(\alpha j\), defined circularly on (19). It is not an
arbitrary arithmetic average of nine independently chosen phase
representatives. The physical observation of such a collective phase still
needs independent justification. On the neutral leaf of (21), the same
derivation as (8)-(12) gives

\[
 \begin{split}
 C_W&=\frac{36\kappa}{a_0 b\gamma}L_W^+,
 &G_W&=\frac{9\kappa}{a_0\gamma}P_6,\\
 I_{c,W}&=\frac{9b\kappa\gamma}4,
 &V_W&=\frac{a_0b\gamma}{36\kappa}L_Wq,
 \qquad P_6=I_6-\frac{\mathbf1\mathbf1^T}{6}.
 \end{split}
 \tag{22}
\]

Unlike the direct C9 inverse, the centered wheel inverse has strictly
negative off-diagonal entries. Its exact entries are

| Wheel entry | Value of \(L_W^+\) |
| --- | --- |
| Host diagonal | \(103/396\) |
| Donor diagonal | \(55/396\) |
| Adjacent host pair | \(-5/396\) |
| Nonadjacent host pair | \(-41/396\) |
| Host-donor pair | \(-11/396\) |

The symmetric matrix with these entries has zero row sums and satisfies
\(L_WL_W^+=P_6\), which verifies the inverse by exact arithmetic. It is
positive definite on the neutral subspace. Every pair of distinct islands
can therefore be joined by an ordinary positive mutual capacitor:

\[
 c_{cd}=\frac{36\kappa}{a_0b\gamma}
    \begin{cases}
      5/396,&c,d\text{ adjacent host children},\\
      41/396,&c,d\text{ nonadjacent host children},\\
      11/396,&\text{one child is the donor}.
    \end{cases}
 \tag{23}
\]

Each of the fifteen island pairs also has resistor conductance
\(9\kappa/(6a_0\gamma)\). The ten original wheel edges carry positive
sinusoidal junctions with critical current (22). Thus this is an explicit
finite passive circuit construction in the ideal reference class, not just
a positive matrix lacking a passive component realization. Extra capacitor
and resistor connections between nonadjacent host children are part of the
construction; the mapping does not equate those wires with original TNFR
support edges. No spatial conductor geometry, apparatus or component
calibration is inferred.

The fine storage on (19) is exactly

\[
 H_{54}=H_{\rm int}+9H_W,\qquad
 H_{\rm int}=54[1-\cos\alpha],\qquad
 H_W=\frac12X^TL_WX+
       \sum_{\{c,d\}\in E_W}[1-\cos(\Phi_d-\Phi_c)].
 \tag{24}
\]

The internal constant is retained explicitly: the circuit measures the
variable collective storage, not the absolute full fine storage. With
\(K=a_0b\kappa\gamma/4\) as before,

\[
 E_W=9K H_W=K(H_{54}-H_{\rm int}),\qquad
 \dot E_W=-\frac{9Kb}4\|L_WX\|^2=-V_W^TG_WV_W.
 \tag{25}
\]

Every regional sum over whole children has the same total charge transfer,
because (20)-(22) match its full nodal divergence. The graph of resistor
connections differs, so the representation does not reproduce each original
edge current or assign its individual work to the same physical wire.
Likewise, it does not automatically represent arbitrary fine forcing,
carrier-breaking observations or support events.

One supplied port extension is exact. Add the same form input `f_c` to all
nine nodes of child `c`, leave the phase row unchanged and require
`sum_c f_c=0`. This preserves the carrier and its neutral leaf. With
`kappa_c=9*kappa` and `K_c=9*K`, the electrical current input and work are

\[
 I_{\rm ext}=b\kappa_c f,\qquad
 V_W^TI_{\rm ext}=K_c b(L_WX)^Tf.                       \tag{25a}
\]

Thus the complete driven collective energy law has this input power in
addition to the loss in (25). The input remains supplied; phase forcing,
nonuniform fine-node injection and support switching have no such admission
from (25a).

**Conditional positive realization.** On the exact invariant carrier (19),
with its marking, retained reference, neutral inventory, fixed same-slot
support and clock scale, (21)-(23) give an invertible collective charge-phase
description by the ideal passive reference (2). Its variable storage and
total dissipation match (25). This does not identify an actual physical
system, select the free scales \(\kappa,b\), or imply that a general nearby
fine state has this closed electrical evolution. The generic fine-node
obstruction and this positive collective construction have different state
and information domains and are jointly consistent.

## Inventory, work and interpretation boundaries

The charge map retains the exact regional continuity equation

\[
 \dot Q_A=\frac{b\kappa}4
 \sum_{\substack{i\in A,\ j\notin A\\i\sim j}}
 [x_j-x_i+\gamma\sin(\theta_j-\theta_i)].
 \tag{26}
\]

This is a structural transport identity under (5), not physical charge
identification. The formal circuit reproduces its nodal divergence but has
the additional capacitance and wire-realizability obligations above.

A nonzero conserved total is not silently projected away. Under strict
absolute phase matching, a uniform nonzero \(q\) produces \(V=0\) in (8),
which contradicts \(q=CV\) unless that uniform charge is an independently
retained background offset. A grounded common-charge mode can instead carry
a common voltage and phase rotation; admitting it changes the state/gauge
comparison and retains its initial total. The proof already uses the neutral
leaf most favorable to the direct map, where this separate issue is absent.
Fixed background offsets do not alter the incremental contradiction (14).

For a supplied state jump on **unchanged** support with unchanged scales,
the formal equality (12) gives \(\Delta E=K\Delta H\). This is an
accounting identity, not a passive event selector. An arbitrary external
input must transform both rows and its work pairing; the unforced proof
does not supply a current source, phase drive or controller. When support
changes, (9) and (11) generally change their capacitance and conductance
operators, including entries away from the newly attached edge. Switching
such a circuit requires an independent component, charge and work model.
The old support-event TNFR storage budget does not by itself prove that
electrical switching is passive or that it has occurred.

The result therefore separates three conclusions. The complete organized law
has an exact formal passive-energy representation with retained phase. The
declared direct fine-node charge/phase interpretation on C9 and the same-slot
54-node contact fails the physical capacitance sign test, including its
grounded and statically reduced variants. The exact six-child carrier admits
the explicitly constructed passive collective circuit (22)-(25). The
[source-inclusive comparison](ORGANIZED_ELECTRICAL_SOURCE.md#source-electrical-admission-result)
extends that collective result to actual prepared families with circular
observation and uniform finite error bounds. No elementary-charge quantization, microscopic
constituent formation, electromagnetic emergence or empirical validation
follows from either conclusion.

## Exact departure identities for a later source-inclusive comparison

<a id="collective-electrical-departure-identities"></a>

The exact carrier result does not reset the acquired fine state. Its
extension has a reusable symbolic starting point. On a declared coherent
phase chart, write

\[
 x_{c,j}=X_c+u_{c,j},\qquad
 \theta_{c,j}=\alpha j+\Phi_c+z_{c,j},\qquad
 \sum_j u_{c,j}=\sum_j z_{c,j}=0.                       \tag{27}
\]

These are actual child means and deviations. Their phase lifts must follow
one admitted circular observation. For example, let
`a_c=Arg(sum_j exp(i*(theta_cj-alpha*j)))` and require every wrapped
residual about `a_c` to have magnitude strictly less than `pi/2`. The
observation is unavailable if the resultant is zero; no angle is supplied
there. On the strict collar set `v_cj=wrap(theta_cj-alpha*j-a_c)`, and use

\[
 \Phi_c=a_c+\frac19\sum_jv_{c,j}\pmod{2\pi},\qquad
 z_{c,j}=v_{c,j}-\frac19\sum_kv_{c,k}.                 \tag{28}
\]

Independent full turns of fine representatives do not change (28).
Crossing the argument's display cut changes only a common lift; on the
strict collar (28) is smooth as a circular observation. Locally its
derivative is the arithmetic mean of the consistent fine phase derivatives.
The raw `Arg` alone generally has a different derivative. The collar,
marking and lift continuity are hypotheses, not yet a certificate that the
actual acquired family satisfies them throughout contact.

Define, on each wheel edge,

\[
 A_{cd}=\frac19\sum_j
   \sin(\Phi_d-\Phi_c+z_{d,j}-z_{c,j}),\qquad
 \mathcal R_c=\frac\gamma4\sum_{d\sim c}
       [A_{cd}-\sin(\Phi_d-\Phi_c)].                  \tag{29}
\]

Equitable same-slot support and internal edge cancellation give the exact
mean equations

\[
 X'=\frac{-L_WX+\gamma S_W(\Phi)}4+\mathcal R,\qquad
 \Phi'=\frac\gamma4L_WX,\qquad \sum_c\mathcal R_c=0.           \tag{30}
\]

No phase remainder occurs in the second row for this specific observation.
The phase-covariance remainder is not an independent external drive: its
evolution consumes the hidden fine state. Since each edge departure has
zero mean, Taylor's theorem with `abs(sin''(t))<=1` cancels the linear term
and yields

\[
 |A_{cd}-\sin(\Phi_d-\Phi_c)|
 \le\frac1{18}\sum_j(z_{d,j}-z_{c,j})^2.               \tag{31}
\]

This quadratic bound is uniform in the collective phase difference. It
does not require an acute external contact and does not replace the
independent internal collar or identity premises.

Let `T` repeat each of six coordinates over its nine fine nodes. The exact
orthogonality `T^T u=T^T z=0` and `L_54 T=T L_W` give

\[
 H_{54}=H_{\rm int}+9H_W+R_H,\qquad
 R_H=\tfrac12u^TL_{54}u+R_\phi,\qquad
 |R_\phi|\le\tfrac12z^TL_{54}z,                       \tag{32}
\]

where `R_phi` is the actual difference of phase potentials. Internal
first-order terms telescope around each cycle; external first-order terms
vanish by the zero child means. The cosine Hessian bound then gives (32).
The signed remainder need not be nonnegative; it is not an event reserve.
The loss and mean storage obey

\[
 \begin{split}
 H_{54}'&=-\tfrac94\|L_WX\|^2-\tfrac14\|L_{54}u\|^2,\\
 H_W'&=-\tfrac14\|L_WX\|^2+(L_WX)^T\mathcal R.
 \end{split}                                         \tag{33}
\]

Equations (27)-(33) isolate the actual observation, closure and hidden-work
obligations. The [source-inclusive owner](ORGANIZED_ELECTRICAL_SOURCE.md)
uses them to prove and assess the finite comparison on the acquired families,
retaining hidden storage and work. Physical calibration and support events
remain separate obligations. The
[exact algebra and field controls](../../tests/physics/test_organized_electrical_realization.py)
verify the carrier construction and fine obstruction without replaying a
frozen producer; the general conditional proofs remain in this owner.
