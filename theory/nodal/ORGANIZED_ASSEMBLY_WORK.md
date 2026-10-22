# Work required to assemble recovered organizations

**Status:** conditional endpoint obstruction for the three recovered sine
winding-one C9 objects and the two fixed oriented contact supports. Every
simultaneous form/phase reset that retains their acute winding-one internal
geometry requires positive net storage work. Allowing internal deformation
and arbitrary relative phase origins does not remove that requirement.
This is a lower-bound theorem, not the sharp minimum, an event selector or a
claim about physical binding. The [research program](../RESEARCH_PROGRAM.md#research-current-question)
alone assigns subsequent work.

The [acquired network capture theorem](ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md)
already establishes convergence after a supplied, unchanged-state contact.
Its positive contact work is a separate premise. The present result tests
whether simultaneous internal reorganization can pay that cost for the
actual recovered children, using the reset-accounting principle from
[support events](RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact).
The weighted sine storage below is its own model; native unique-neighbor
phase storage and registered operator execution are not substituted for it.

<a id="organized-assembly-state"></a>
## Complete endpoint model and retained identity

Retain nodes \((r,j)\), ordered as \(9r+j\), with \(r=0,1,2\) and
\(j=0,\ldots,8\). Each marked child retains all nine original
unit-conductance cycle edges. Write \(\alpha=2\pi/9\),
\(x=\mathrm{EPI}\in\mathbb R^{27}\), and let \(\theta\) denote
primitive circular phase. Capacity is held at one. The before support
\(E_0\) is the disjoint union of the three cycles. The after support
\(E_s\) additionally contains, for each oriented contact
\(e=(r,t)\in\{(0,1),(1,2),(2,0)\}\),
\[
 (r,j)--(t,j)\quad\text{with weight }1-s_e,\qquad
 (r,j)--(t,j+1)\quad\text{with weight }s_e . \tag{1}
\]
All indices are modulo nine, all added edges are undirected, and
\[
 s^{\rm eq}=(1/2,1/2,1/2),\qquad
 s^{\rm un}=(1/4,1/2,3/4). \tag{2}
\]
No old edge is removed or reweighted. Each final node has weighted degree
four. The complete support-dependent storage, with its absolute edge
baselines retained, is
\[
 H_E(x,\theta)=\frac12\sum_{\{i,j\}\in E}w_{ij}(x_i-x_j)^2
              +\sum_{\{i,j\}\in E}w_{ij}[1-\cos(\theta_j-\theta_i)] .
 \tag{3}
\]
The phase weights in (3) are actual conductances. A comparison that replaces
them with unique-neighbor weights changes this model.

This is detached endpoint accounting for the same sine model as the capture
owner. That owner declares the complete continuous laws, the structural
clock \(\tau=(1023/1024)t\), pressure, held capacity and loss. No finite
duration is assigned to a reset, and no new continuous row is introduced.
The units of work below are the declared structural storage units of (3);
there is no laboratory energy identification.

The initial endpoint is an actual member of the original three-source
family, after acquisition time \(100\) and uninterrupted recovery dwell
\(10^{13}\). At the unchanged handoff \(\tau_h=100+10^{13}\),
the [shared recovery](ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md#organized-inheritance-acquired-state)
at sine deformation zero gives
\[
 \begin{split}
 \Theta_{r,j}&=\alpha(j-4),\qquad
       \theta^-_{r,j}=\Theta_{r,j}+y^-_{r,j},\\
 \|x^-_r\|_2&\le\varepsilon,\qquad
       \|y^-_r\|_2\le\varepsilon,
       \qquad \varepsilon=10^{-70},\\
 \mathbf1^Tx^-_r&=\mathbf1^Ty^-_r=0 .
 \end{split}\tag{4}
\]
The three copies have independent original coordinate errors at most
\(10^{-10}\), with their declared separate zero means. No source is reset
to \((0,\Theta)\), no new dwell is selected, and no favorable correlation
between children is imposed.

Admit hypothetical finite after coordinates \((x^+,\theta^+)\) subject to
the following endpoint identity restriction. For each marked child let
\(\delta^+_{r,j}\) be the principal circular gap from node \((r,j)\) to
\((r,j+1)\). Require
\[
 -\pi/2<\delta^+_{r,j}<\pi/2,\qquad
       \sum_{j=0}^8\delta^+_{r,j}=2\pi . \tag{5}
\]
Thus every retained child is internally strictly acute and has winding
one. Its internal shape, absolute and relative phase origins, and every
finite signed form may change. Endpoint means need not equal the source
means. Proving an obstruction in this larger class also covers any subclass
that conserves them. New contact gaps need not be acute.

Condition (5) is an explicit endpoint class, not a consequence of winding
alone or a declaration that any such state was acquired dynamically. In
particular, the capture theorem permits nonacute transient gaps in its
larger trapping sector. Its no-reset capture certificate is not transferred
to arbitrary endpoints here. An admitted algebraic reset is not yet a
registered operator: live preconditions, collective descent and occurrence
remain separate obligations of the
[NFR ontology](../EMERGENT_ONTOLOGY.md#nfr-participation-in-form) and
[operator contracts](../STRUCTURAL_OPERATORS.md#22-operator-as-transformation).

Common form translation, common phase rotation and simultaneous cyclic
relabeling preserve (3)--(5). Separate child phase rotations preserve
internal storage but generally change contact storage; they are retained
coordinates, not independently removable gauges of the joined model.

<a id="organized-assembly-source-storage"></a>
## The actual source has only a small internal excess

Put
\[
 V(z)=1-\cos z,\qquad B_0=27V(\alpha). \tag{6}
\]
On the acute interval, \(V''(z)=\cos z>0\). Jensen's inequality and the
winding constraint in (5) give, separately for each child,
\[
 \sum_{j=0}^8V(\delta^+_{r,j})
       \ge9V\!\left(\frac19\sum_j\delta^+_{r,j}\right)
       =9V(\alpha). \tag{7}
\]
Thus every after endpoint has old-support storage at least \(B_0\),
including its nonnegative form contribution. Equality in phase requires
all internal gaps to be \(\alpha\).

The source in (4) is also in that acute winding sector. Let \(L_0\) be the
Laplacian of the three unit cycles. Its largest eigenvalue is at most four.
The internal phase gradient vanishes at \(\Theta\); this is a fine-node
identity because each node has opposite sine currents from its two cycle
neighbors. Taylor's integral formula and \(\nabla^2 U\preceq L_0\) give
\[
 \begin{split}
 H_{E_0}(x^-,\theta^-)-B_0
 &\le\tfrac12(x^-)^TL_0x^-
                     +\tfrac12(y^-)^TL_0y^-\\
 &\le2\|x^-\|_2^2+2\|y^-\|_2^2
 \le12\varepsilon^2.
 \end{split}\tag{8}
\]
Combining (7) with (8),
\[
 \boxed{B_0\le H_{E_0}(x^-,\theta^-)\le B_0+12\varepsilon^2.} \tag{9}
\]
All residual coordinates remain in this bound. The exact rational term
\(12\varepsilon^2\) must remain separate from an outward interval for
the irrational baseline \(B_0\); rounding it away would incorrectly
replace a reached state with the ideal minimum.

<a id="organized-assembly-contact-bound"></a>
## Every retained winding forces positive contact storage

Consider one oriented contact \(e=(r,t)\) and one node index \(j\).
Set \(a=\theta^+_{t,j}-\theta^+_{r,j}\), interpreted modulo a turn,
and \(\delta=\delta^+_{t,j}\). The two port contributions are
\[
 \begin{split}
 C_{e,j}
 &=(1-s_e)V(a)+s_eV(a+\delta)\\
 &=1-\operatorname{Re}\{e^{ia}(1-s_e+s_e e^{i\delta})\}\\
 &\ge 1-|1-s_e+s_e e^{i\delta}|=:f_{s_e}(\delta).
 \end{split}\tag{10}
\]
Seam integers do not change the cosines or the phasor identity. This
inequality minimizes over one source phase independently; it remains a
lower bound when the actual 27 phases impose further correlations.

For \(u=s(1-s)\) and \(0<s<1\), write
\[
 \rho_s(z)=\sqrt{1-2u(1-\cos z)},\qquad f_s(z)=1-\rho_s(z).
 \tag{11}
\]
On \([-\pi/2,\pi/2]\), \(\rho_s^2\ge1-2u\ge1/2\), so the
derivatives are regular and
\[
 f_s''(z)=\frac{u\cos z}{\rho_s(z)}
            +\frac{u^2\sin^2z}{\rho_s(z)^3}>0 . \tag{12}
\]
The endpoint positivity follows from the second term; at zero the first
term is positive. Applying Jensen to the nine target-child gaps gives
\[
 \sum_{j=0}^8 C_{e,j}
 \ge\sum_j f_{s_e}(\delta^+_{t,j})
 \ge9f_{s_e}(\alpha)=9[1-\rho_{s_e}(\alpha)]. \tag{13}
\]
Every contact has its own target child with winding one, so summing these
three inequalities is valid on the same complete endpoint. Define
\[
 G(s)=9\sum_{e}[1-\rho_{s_e}(\alpha)]. \tag{14}
\]
The added form storage is nonnegative, hence (7) and (13) imply the
full-state lower bound
\[
 \boxed{H_{E_s}(x^+,\theta^+)\ge B_0+G(s).} \tag{15}
\]
No invariant registered-family assumption or collective tangent reduction
appears in this argument.

A simpler independent lower bound follows from
\(1-\sqrt{1-z}\ge z/2\) for \(0\le z\le1\):
\[
 G(s)\ge 9V(\alpha)\sum_es_e(1-s_e)>0. \tag{16}
\]
For the two fixed controls,
\[
 \begin{array}{c|c|c}
 &G(s)&\text{simpler lower bound}\\ \hline
 s^{\rm eq}&27[1-\cos(\pi/9)]&\tfrac{27}{4}V(\alpha)\\
 s^{\rm un}&9[3-2\sqrt{1-\tfrac38V(\alpha)}-\cos(\pi/9)]
                &\tfrac{45}{8}V(\alpha)
 \end{array} . \tag{17}
\]
The two values of \(G\) are approximately \(1.62829924\) and
\(1.35048908\), respectively. These decimals are explanatory;
the certificate uses outward rational trigonometric and square-root bounds.
The same enclosures give \(V(\alpha)>1/5\). Thus even the simpler
bounds give \(G(s^{\rm eq})>27/20\) and
\(G(s^{\rm un})>9/8\), far above the exact source excess in (9).

Equations (14)--(17) are not claims of a sharp joined minimum. Simultaneous
equality in the child Jensen bounds requires uniform internal twist. At
such a state, equality in every contact's phasor bound would require
\(\Phi_t-\Phi_r=-\arg(1-s_e+s_e e^{i\alpha})\) modulo a turn.
The left sides sum to zero around the three-contact cycle, while the
sum of those arguments is \(\pi/3\) for both controls, as established
by the [network compatibility theorem](ORGANIZED_NETWORK_COMPATIBILITY.md).
Thus the separately relaxed contact minima are incompatible. The positive
lower bound remains valid without solving the sharper coupled minimization.

<a id="organized-assembly-work-obstruction"></a>
## Required work and passive path obstruction

The exact storage jump for a simultaneous reset and attachment is
\[
 W_{\rm event}=H_{E_s}(x^+,\theta^+)-H_{E_0}(x^-,\theta^-).
 \tag{18}
\]
It includes both internal reorganization and all added-edge costs. The
counterfactual algebraic decomposition
\[
 \begin{split}
 W_{\rm event}={}&H_{E_0}(x^+,\theta^+)-H_{E_0}(x^-,\theta^-)\\
 &+H_{E_s}(x^+,\theta^+)-H_{E_0}(x^+,\theta^+)
 \end{split}\tag{19}
\]
does not impose an execution order. Equations (9) and (15) show that
contemporaneous internal release is insufficient:
\[
 \boxed{W_{\rm event}\ge G(s)-12\varepsilon^2>0.} \tag{20}
\]
In particular, the internal term can release at most \(12\varepsilon^2\),
whereas the new contacts cost at least \(G(s)\). The source has already
relaxed close to its minimum within the retained acute winding sector.
The obstruction applies to every original admitted source and every after
endpoint in (5), not merely a nominal target or one attempted reset.

The unchanged-state attachment from the capture theorem costs approximately
\(3.15840002\), with its own exact residual correction. Equation (20)
allows a reset to reduce that cost but excludes making it storage-passive
under the declared endpoint restrictions. It neither constructs an optimal
reset nor certifies capture from a proposed reset endpoint.

The finite hybrid balance also excludes an evasion by passive staging.
Suppose every continuous segment has its separately admitted storage law
\(H'=-D\), \(D\ge0\), every event uses the actual complete storage
jump \(J_k\), and all segments and jumps use the same storage normalization
(3). For any such finite hybrid path from (4) to an endpoint in (5),
\[
 H_{\rm final}-H_{\rm initial}
       =-\int D\,d\tau+\sum_kJ_k,
 \qquad
 \sum_kJ_k\ge G(s)-12\varepsilon^2+\int D\,d\tau>0.
 \tag{21}
\]
Consequently no path with every event passive, \(J_k\le0\), has these
endpoints. Even permitting individually positive jumps while demanding a
nonpositive total jump budget cannot satisfy (21). If independently
declared supplied-work budgets obey \(J_k\le W_k\), their sum must also
obey the same positive lower bound. Past loss is not an available reserve
without a separately specified storage coordinate and reservoir law.

This uses telescoping accounting, not a theorem admitting an arbitrary
intermediate support or trajectory. Each intermediate law, regularity
domain and event must be admitted separately. The result does not supply
a schedule, finite stopping time, non-Zeno proof or autonomous occurrence
argument. Temporary edge removal cannot evade the endpoint inequality if
the declared final support is restored and the same complete storage is
used throughout. Permanently removing or reweighting original edges changes
the final-support premise instead.

<a id="organized-assembly-deletion"></a>
## Deletion and limits of the conclusion

At any final endpoint in (5), removing all added contacts without changing
the coordinates changes storage by
\[
 H_{E_0}(x^+,\theta^+)-H_{E_s}(x^+,\theta^+)
       =-H_{\rm contacts}(x^+,\theta^+)\le-G(s)<0. \tag{22}
\]
The three marked cycles and their acute winding-one endpoint identities
remain. This is frozen-state endpoint accounting, not a post-deletion
recovery theorem or an assertion that the preceding acquired trajectory
is internally acute at a specified finite time.

Thus held-support convergence does not establish storage binding against
an admitted contact-deletion action. A negative deletion jump also does
not select that action's occurrence. Both attachment and detachment still
require an event model; no physical binding energy is identified here.

The obstruction retains its declared acute endpoint identity class, late
recovered source, original internal edges, two fixed contact controls and
weighted sine storage. It does not exclude attachment earlier during
acquisition, endpoints outside the acute class, an identity change,
removal or reweighting of original edges, or a separately justified storage
or reservoir law. Those would change premises and require their own
source, event and capture analysis. No such extension is executed here.

## Implementation and independent controls

The [assembly-work certificate](../../src/tnfr/physics/_organized_assembly_work.py)
re-admits primitive inputs, rebuilds the selected weighted support and
original source recovery, and derives the contact lower bounds from the
shared contact-amplitude owner. It retains the exact source excess
separately from interval-rounded nominal terms and makes source-dependent
claims only when that recovery is certified. Incoming report verdicts are
not premises.

The [independent controls](../../tests/physics/test_organized_assembly_work.py)
exercise the actual support, nonuniform acute winding endpoints,
finite form/phase reset accounting, strict work signs, original residual
scope, deletion, and invalid or unavailable source inputs. These are
analytic and static checks. No original source producer, trajectory,
reserved response or physical dataset is regenerated.
