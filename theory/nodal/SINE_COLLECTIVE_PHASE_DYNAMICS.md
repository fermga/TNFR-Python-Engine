# Collective phase families and live contact feedback

Unit and weighted phase-offset partitions, nearby full-state moving windows, contact/storage exchange and complete receiver-rigidity conditions.

Part of [Sine pattern geometry and dissipative capture](SINE_PATTERN_DYNAMICS.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 39. Compatible internal geometry and collective phase dynamics

<a id="sine-phase-offset-partition"></a>

A region can retain phase differences while its common form and phase
continue to evolve. The full nodal rows determine when this is possible;
an instantaneous equality of phase rates does not suffice. This section
classifies exact collective families under Section 35's conservative unit
law, fixed finite connected simple support, unit held capacities and
`tau=t/pi`. No input, event or change of support is introduced.

### A declared partition with fixed internal phase offsets

Partition every fine node into nonempty blocks `B_a`. Supply fixed circular
offsets `psi_i`; a consistent real lift can be used for differentiation.
Consider the **whole family**

\[
\mathcal M_\psi:
\qquad x_i=X_a,\qquad \theta_i=\Theta_a+\psi_i
\quad(i\in B_a),
\]

with freely varying collective forms `X_a` and phases `Theta_a`. Internal
phase differences are fixed, but different nodes need not share a phase.
The block partition and offsets are supplied geometry, not inferred
membership or a newly created node. Define, using full graph degrees,

\[
q_{ib}=\frac{|N(i)\cap B_b|}{d_i},\qquad
Z_{ib}=\frac1{d_i}
       \sum_{j\in N(i)\cap B_b}e^{\mathrm i(\psi_j-\psi_i)}.
\]

The family is invariant under the complete law **if and only if**, in
each block `a`, the following conditions hold:

1. For every external block `b!=a`, `q_ib=q_ab` is independent of `i`.
2. For every external block `b!=a`, the full complex value `Z_ib=Z_ab`
   is independent of `i`.
3. Every internal sine sum vanishes: `Im Z_ia=0` for all `i in B_a`.

These are conditions on support and phase offsets for all collective
states. Failure does not exclude an accidental special trajectory with
additional relations between its collective coordinates. In particular,
equal neighbor counts alone do not settle the nonlinear form row.

### Necessity and sufficiency use both consumed rows

On the proposed family the phase row at `i in B_a` is

\[
\theta_i'=\sum_{b\ne a}q_{ib}(X_a-X_b).
\]

For its value to be independent of `i` for every independent choice of
`X`, each external coefficient must agree. The form row is

\[
x_i'=\operatorname{Im}Z_{ia}
 +\sum_{b\ne a}\operatorname{Im}
       \left[Z_{ib}e^{\mathrm i(\Theta_b-\Theta_a)}\right].
\]

Independent variation of the collective phase differences makes the
external sine and cosine coefficients independent Fourier components.
Their equality forces equality of the external complex values `Z_ib`.
The remaining internal imaginary parts must be one common constant
`c_a`. Undirected internal edges give

\[
\sum_{i\in B_a}d_i\operatorname{Im}Z_{ia}
=\sum_{\substack{i,j\in B_a\\j\sim i}}
  \sin(\psi_j-\psi_i)=0.
\]

Since `sum_(i in B_a) d_i>0`, that common constant is zero. There is no
additional condition on `Re Z_ia`: the form row does not consume it and
the phase row has zero internal form differences. For example a one-block
P3 with offsets `(0,0,pi)` has internal real values `(1,0,-1)`, zero
internal sine current and uniform-form equilibria. Requiring the complete
internal complex sums to agree would wrongly reject this family.

Conversely, the three conditions make both fine rows constant within
their respective blocks. The field is tangent to `M_psi` everywhere;
smooth uniqueness therefore preserves the family. Its exact closed law is

\[
\boxed{\begin{aligned}
X_a'&=\sum_{b\ne a}\operatorname{Im}
       \left[Z_{ab}e^{\mathrm i(\Theta_b-\Theta_a)}\right],\\
\Theta_a'&=\sum_{b\ne a}q_{ab}(X_a-X_b).
\end{aligned}}
\]

Thus phase-offset geometry and support determine whether all constituents
maintain compatible rates while the collective coordinates move. This is
an invariant-family reduction, not a quotient for every fine state.

### Reciprocity, inherited storage and conserved means

Let `m_a=sum_(i in B_a) d_i`, `N_ab=m_a q_ab` and
`P_ab=m_a Z_ab`. Counting the same undirected cross edges in reverse gives

\[
N_{ab}=N_{ba},\qquad
P_{ab}=\overline{P_{ba}},\qquad |Z_{ab}|\le q_{ab}.
\]

The full fine storage restricted to the family is exactly

\[
\begin{aligned}
E_{\mathcal M}
&=E_{\mathrm{int}}(\psi)
 +\sum_{a<b}\left[
 \frac{N_{ab}}2(X_a-X_b)^2+N_{ab}
 -\operatorname{Re}\left(
     P_{ab}e^{\mathrm i(\Theta_b-\Theta_a)}\right)\right],\\
E_{\mathrm{int}}(\psi)
&=\sum_a\sum_{\{i,j\}\subset B_a,\ i\sim j}
       [1-\cos(\psi_j-\psi_i)].
\end{aligned}
\]

The closed rows consequently satisfy

\[
X_a'=-\frac1{m_a}\frac{\partial E_{\mathcal M}}{\partial\Theta_a},
\qquad
\Theta_a'=\frac1{m_a}\frac{\partial E_{\mathcal M}}{\partial X_a}.
\]

This is a restriction of the existing storage and reciprocal field, not
an independently supplied energy. It preserves that storage and the
collective weighted sums `sum_a m_a X_a` and `sum_a m_a Theta_a` in
continuous lifts. The fixed contribution `sum_i d_i psi_i` restores the
full lifted phase sum.

In general the inherited form coupling `Z_ab` and phase coupling `q_ab`
are different. Phase cancellation can give `|Z_ab|<q_ab`, and complex
offsets can remain in the interaction. Replacing both by the weights of
a bare normalized-sine graph would then change the complete law. This
is the same retained-information boundary as in
[nonlinear replica inheritance](SINE_PAIR_STATE.md#sine-replica-inheritance).

For **each block separately**, internal form storage is zero and internal
phase storage is constant. In fact `q_R=S_R=0` at every point of this
family, not only initially. Section 36's internal channels and the net
regional power `P_R` therefore vanish throughout its motion. This does
not remove interaction: the collective forms and phases, cross-edge
storage and common regional rates can change. Internal regional storage
does not count that cross-edge organization.

### One moving realization on the retained private-leaf support

Use the same C5 plus five private leaves as Section 37, with blocks
`R=(0,...,4)` and `Q=(5,...,9)`. Supply offsets
`psi_i=psi_(5+i)=sigma*2pi*i/5`, for `sigma=+1` or `-1`.
The external coefficients are

\[
q_{RQ}=Z_{RQ}=\frac13,\qquad
q_{QR}=Z_{QR}=1.
\]

Every receiver internal sine sum cancels between the two cycle directions;
the leaves have no internal edges. Write the block forms as `a,b` and
phases as `Theta,Psi`. The complete reduced rows are

\[
a'=\tfrac13\sin(\Psi-\Theta),\qquad
b'=-\sin(\Psi-\Theta),\qquad
\Theta'=\tfrac13(a-b),\qquad
\Psi'=b-a.
\]

Putting `u=a-b`, `phi=Psi-Theta` gives

\[
u'=\tfrac43\sin\phi,\qquad
\phi'=-\tfrac43u,\qquad
\tfrac12u^2+1-\cos\phi=\text{constant}.
\]

The receiver retains its exact acute winding and internal storage `V_5`
while interacting with its moving environment. Its collective weighted
form sum is `15a+5b`; the analogous lifted phase sum has the same weights.
The common regional phase motion is compensated by the environment, in
accord with Section 38's global phase balance. The inherited pendulum
reuses the [reciprocal exchange mechanism](RESONANCE_FOUNDATIONS.md#reciprocal-exchange);
it is not another fundamental oscillator, frequency identification or
universal permanent-pulse postulate.

### Exact compatibility is not finite-time formation into the family

The complete smooth field has a unique flow in both time directions.
If a trajectory met this invariant family at a finite time, the unique
backward solution through that point would also lie in the family.
Therefore a preparation outside `M_psi` cannot enter it exactly at finite
time under this unchanged autonomous law. In particular, globally flat
phases cannot enter the displayed nonzero-twist family exactly.

This does not exclude entry into a neighborhood with nonzero width, nor
finite retention of a less restrictive identity. Those require an actual
entry state, declared finite error bounds and a comparison under the
same complete law. The existing
[full-state finite-window argument](SINE_PAIR_GROUPING.md#sine-joint-identity-window)
illustrates that separate obligation on its own support; it is not
automatically a certificate for these arbitrary partitions. No acquisition
trajectory or attraction theorem is supplied by the exact classification.

### Shared admission distinguishes proof from undecided equality

The [partition owner](../../src/tnfr/physics/relational_sine_partition.py)
provides `assess_sine_phase_offset_partition(source, blocks,
phase_offset_turns)` and `SinePhaseOffsetPartition`. Declared rational-turn
offsets specify the phase geometry independently of a response.
Its `.evaluate(...)` returns a `SinePhaseOffsetState` for admitted collective
coordinates, retaining fine reconstruction, the inherited rows and storage.
It does not install a coarse runtime, select a partition or certify that
the source already belongs to the supplied family.

The all-real mathematical criterion is stronger than any particular finite
equality recognizer. Exact cancellations admitted by the implementation,
including sine oddness and angle reflection, can prove the required
identities. Certified interval exclusion of zero can prove a failed
equality. An unrecognized symbolic cancellation or an interval containing
zero remains unavailable; it is not evidence of inequality and must not
be converted into a failed mathematical criterion.

The [held native joint quotient](../TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#joint-constitutive-reduction-with-inherited-support-counts),
the [replica family](SINE_PAIR_STATE.md#sine-replica-inheritance)
and the [independent-swap criterion](SINE_PAIR_STATE.md#sine-pair-support-symmetry)
retain their different models and quantifiers. This theorem concerns
supplied fixed offsets and all collective states on their invariant family;
it does not claim closure of arbitrary fine states, autonomous formation,
support selection or physical identification.

## 39a. Weighted form and held nonuniform capacities

<a id="sine-weighted-phase-offset-partition"></a>

Section 39 retains its unit law and rescaled clock. The following extension
uses the [weighted form, bare-sine law](SINE_FORM_CAPACITY_ACTION.md#1-state-law-and-execution-boundary)
in the **original structural time** `t`. Let the finite bare support `U` be
connected, simple and undirected, with strictly positive symmetric
conductances `W_ij` on every edge. Its bare degrees are `d_i>0`. Hold every
capacity `nu_i>0`, conductance and edge fixed during continuous evolution;
there is no forcing, continuous loss or event on that segment. The complete
rows and independently specified pressure are

\[
 p_i=\frac1{\pi d_i}\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
 \dot x_i=\frac{\nu_i}{\pi d_i}
                 \sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
 \dot\theta_i=\frac{\nu_i}{\pi d_i}
                 \sum_{j\sim i}W_{ij}(x_i-x_j).
\]

The sine row and degree use bare neighbors; only the form-gradient channel
uses conductances. The pressure is not inferred by dividing an observed
rate by capacity. These are selected constitutive premises, not a unique
consequence of the nodal identity or the native neighbor-resultant law.

Supply a partition into `k` nonempty blocks, with `2<=k<n`, and fixed circular
offsets with declared real representatives `psi_i`. Consider the entire
family `M_psi` of Section 39: `x_i=X_a` and
`theta_i=Theta_a+psi_i` for `i in B_a`, with freely varying collective forms
and phases. Set

\[
 \kappa_i=\nu_i/d_i,\qquad \rho_i=d_i/\nu_i,\qquad
 A_{ib}=\kappa_i\sum_{j\in N(i)\cap B_b}W_{ij},\qquad
 Z_{ib}=\kappa_i\sum_{j\in N(i)\cap B_b}
                          e^{\mathrm i(\psi_j-\psi_i)}.
\]

### Necessary and sufficient conditions on the whole family

The complete field preserves `M_psi` **if and only if**, for each block `a`:

1. Every external coefficient `A_ib=A_ab`, `b!=a`, is independent of
   `i in B_a`.
2. Every external complex coefficient `Z_ib=Z_ab`, `b!=a`, is independent
   of `i in B_a`.
3. `Im Z_ia=0` for every `i in B_a`.

No equality of the internal `A_ia` or `Re Z_ia` is required. Nonuniform
capacities can compensate different degrees; equality of the capacities
within a block is not a premise.

To prove necessity, restriction of the fine rows gives

\[
 \dot\theta_i=\frac1\pi\sum_{b\ne a}A_{ib}(X_a-X_b),\qquad
 \dot x_i=\frac1\pi\left\{\operatorname{Im}Z_{ia}
 +\sum_{b\ne a}\operatorname{Im}
       [Z_{ib}e^{\mathrm i(\Theta_b-\Theta_a)}]\right\}.
\]

Independent variation of the forms forces the external `A` coefficients
to agree. Independent collective phase differences separate the sine and
cosine coefficients, forcing equality of each external complex `Z`. The
remaining internal imaginary part must be a common constant `c_a`. But

\[
 c_a\sum_{i\in B_a}\rho_i
 =\sum_{i\in B_a}\rho_i\operatorname{Im}Z_{ia}
 =\sum_{\substack{i,j\in B_a\\j\sim i}}\sin(\psi_j-\psi_i)=0.
\]

Each internal undirected edge cancels with its reverse, and the sum of
positive `rho_i` is nonzero, so `c_a=0`. Conversely, these three conditions
make both rows constant within each block at every collective state.
The complete smooth field is tangent to the family; uniqueness preserves
it. The descended law, still in the original clock, is exactly

\[
 \boxed{\begin{aligned}
 \dot X_a&=\frac1\pi\sum_{b\ne a}\operatorname{Im}
                  [Z_{ab}e^{\mathrm i(\Theta_b-\Theta_a)}],\\
 \dot\Theta_a&=\frac1\pi\sum_{b\ne a}A_{ab}(X_a-X_b).
 \end{aligned}}
\]

Fine pressures need not agree within a block: when capacities differ, the
products `nu_i p_i` can agree without equality of `p_i`. Failure of the
criterion excludes preservation of this **whole family**, not every
special trajectory with further coordinate constraints. As in Section 39,
smooth backward uniqueness also prevents exact finite-time entry from
outside under the unchanged law. Neither invariance nor its classification
proves attraction, formation or robustness to transverse errors.

### Two inherited couplings, storage and lifted means

Define `m_a=sum_(i in B_a) rho_i`. For different blocks put

\[
 \begin{aligned}
 N^W_{ab}&=\sum_{\substack{i\in B_a,j\in B_b\\j\sim i}}W_{ij}
             =m_aA_{ab},\\
 N^0_{ab}&=|\{\{i,j\}\in U:i\in B_a,j\in B_b\}|,\\
 P_{ab}&=\sum_{\substack{i\in B_a,j\in B_b\\j\sim i}}
                 e^{\mathrm i(\psi_j-\psi_i)}=m_aZ_{ab}.
 \end{aligned}
\]

Reversing the same cross edges gives the reciprocal identities

\[
 N^W_{ab}=N^W_{ba},\qquad N^0_{ab}=N^0_{ba},\qquad
 P_{ab}=\overline{P_{ba}},\qquad |P_{ab}|\leq N^0_{ab}.
\]

The bound uses the **bare edge count**, not the weighted total `N^W`.
The inherited sine coupling and weighted form coupling are distinct data.
In particular, replacing both by one coarse adjacency matrix would
generally change the law.

The fine structural storage and its exact restriction are

\[
 \begin{aligned}
 H&=\frac12\sum_{\{i,j\}\in U}W_{ij}(x_i-x_j)^2
       +\sum_{\{i,j\}\in U}[1-\cos(\theta_j-\theta_i)],\\
 H_{\mathcal M}
 &=H_{\mathrm{int}}(\psi)
   +\sum_{a<b}\left[\frac{N^W_{ab}}2(X_a-X_b)^2+N^0_{ab}
       -\operatorname{Re}(P_{ab}e^{\mathrm i(\Theta_b-\Theta_a)})\right],\\
 H_{\mathrm{int}}(\psi)
 &=\sum_a\sum_{\substack{\{i,j\}\in U\\i,j\in B_a}}
                              [1-\cos(\psi_j-\psi_i)].
 \end{aligned}
\]

Internal form storage vanishes on the family. Direct differentiation yields

\[
 \dot X_a=-\frac1{\pi m_a}\frac{\partial H_{\mathcal M}}{\partial\Theta_a},
 \qquad
 \dot\Theta_a=\frac1{\pi m_a}\frac{\partial H_{\mathcal M}}{\partial X_a}.
\]

Thus `H_M`, `sum_a m_a X_a` and `sum_a m_a Theta_a` are conserved during
held-capacity flow. This is restriction of the existing storage, not a
new independently supplied energy or an identification of `m_a` with
physical mass. To recover the full fine lifted phase mean, retain

\[
 \eta_a=\frac1{m_a}\sum_{i\in B_a}\rho_i\psi_i,\qquad
 \mu_\theta=
 \frac{\sum_a m_a(\Theta_a+\eta_a)}{\sum_a m_a}.
\]

The offsets contribute even though their derivatives vanish. Changing
phase representatives changes a lifted mean unless the associated lift
convention is changed consistently. The continuous reduction retains
`X`, `Theta`, `m`, `N^W`, `N^0`, `P` and the declared offset registration;
these data do not by themselves retain operator histories or admission
context. This is an invariant-family law, not an all-state quotient or
automatically another bare normalized-sine graph.

### Which primary capacity resets preserve the family dynamics?

<a id="sine-weighted-capacity-descent"></a>

Assume the preceding criterion holds. Consider a zero-duration primary
reset that preserves forms, phases, support, conductances and structural
time, with strictly positive capacities

\[
 \nu_i^+=r_i\nu_i,\qquad r_i>0.
\]

Every such reset preserves instantaneous membership in `M_psi`: its defining
forms and phases did not change. The **post-reset vector field** preserves
the entire same family if and only if the **actual ratios** `r_i=r_a` are
constant within each block.

Indeed, the new external rows are `A_ib^+=r_i A_ab` and
`Z_ib^+=r_i Z_ab`; the internal imaginary rows remain zero. Block-constant
ratios are sufficient. For necessity, connected support and a proper
partition give each block an edge to some other block `b`. Strictly
positive conductances and original closure then give `A_ab>0` for every
node in that block. Equality of the new external phase coefficients
`r_i A_ab` forces equality of all its `r_i`. The connected positive-support
and proper-partition premises matter: one block, zero conductances or
zero capacities require separate analysis and are not covered by this iff.

For an admitted block-constant reset,

\[
 A_{ab}^+=r_a A_{ab},\qquad Z_{ab}^+=r_a Z_{ab},\qquad
 m_a^+=m_a/r_a.
\]

The two rows of block `a` acquire the common factor `r_a`, whereas
`N^W`, `N^0`, `P` and `H_M` are unchanged. The storage jump is exactly zero
because `H` has no capacity term. There is a single constant clock scaling
of the **whole family law** precisely when all block factors agree: each
block has a nonzero external phase coefficient, so comparison with one
common multiplier forces the same multiplier everywhere. Different block
factors preserve the family while changing its relative dynamics.

Mean leaves generally change. For `q_a=X_a` or `q_a=Theta_a`, let
`mu_q=sum_a m_a q_a / sum_a m_a`. The unchanged coordinates give

\[
 \mu_q^+-\mu_q=
 \frac{\sum_a(m_a^+-m_a)(q_a-\mu_q)}{\sum_a m_a^+}.
\]

For the full fine phase mean use `q_a=Theta_a+eta_a`. Block-constant ratios
leave each `eta_a` fixed, but change its contribution to the global mean
through the new block weights. Subsequent conservation uses those new
weights. Zero event storage does not transfer the old mean leaves or a
future trajectory certificate. These are the collective consequences of
the existing [form-capacity event identities](SINE_FORM_CAPACITY_ACTION.md#2-full-refreshed-rows-and-event-work).

### Weighted star: positive family and local-reset obstruction

Take `A={0,1,2,3}` and `B={4}`. The internal star edges `0--1`, `0--2`,
`0--3` have weight `3`; each node in `A` has one edge to `4` of weight `2`.
Choose zero offsets and

\[
 d=(4,2,2,2,4),\qquad \nu=(4,2,2,2,8).
\]

Then `kappa_i=1` throughout `A` and `kappa_4=2`. The exact inherited data are

\[
 \begin{gathered}
 A_{AB}=2,\quad A_{BA}=16,\quad Z_{AB}=1,\quad Z_{BA}=8,\\
 m_A=4,\quad m_B=\tfrac12,\quad
 N^W_{AB}=8,\quad N^0_{AB}=P_{AB}=4.
 \end{gathered}
\]

Every internal imaginary row vanishes. The internal real `Z` value is `3`
at the center and `1` at a leaf, so requiring equality of full internal
moments would reject this valid example. The center and leaves also have
different degrees: this block is not an orbit requiring equal capacities
under a graph automorphism.

Writing `u=X_A-X_B` and `Delta=Theta_B-Theta_A`, the original-clock rows are

\[
 \dot X_A=\frac{\sin\Delta}{\pi},\quad
 \dot X_B=-\frac{8\sin\Delta}{\pi},\quad
 \dot\Theta_A=\frac{2u}{\pi},\quad
 \dot\Theta_B=-\frac{16u}{\pi},\qquad
 H_{\mathcal M}=4u^2+4(1-\cos\Delta).
\]

A common capacity factor `s>0` on all of `A` multiplies its two rows by
`s` and changes `m_A` to `4/s`; the rows of `B` remain unchanged. This
preserves the family. In contrast, resetting only node `0` by `s!=1`
gives it external coefficients `(A_0B,Z_0B)=(2s,s)`, while the other
members retain `(2,1)`. At `u=1`, `Delta=pi/2`, its form/phase rates are
`(s/pi,2s/pi)`, instead of `(1/pi,2/pi)` at the leaves. Tangency fails
immediately, despite unchanged instantaneous family membership and zero
event storage. This counterexample requires no numerical trajectory.

### Represented SHA, full operator context and implementation scope

For a mathematical common attenuation `0<s<1`, the preceding primary map
is compatible with [SHA's capacity action](../STRUCTURAL_OPERATORS.md#101-silence-sha).
For an implemented proposal, use the ratio of the **actual represented
output capacity** to the represented input capacity, not merely its
configured factor. The shared
[AL/SHA proposal owner](../../src/tnfr/operators/al_sha_stage_proposals.py)
admits those outputs. Multiplication by the same binary64 factor does not
in general give exactly equal rational ratios on arbitrary nonuniform
capacities; a tolerance cannot prove exact family invariance.

For this star, the capacities in `A` are powers of two. Multiplication by
the normal represented default SHA factor, approximately `0.9204225284540524`,
is exact for `4,2,2,2`, with finite normal outputs. Therefore its actual
ratios agree exactly. This is a statement about the represented factor,
not exact evaluation of the real expression `1-1/(4*pi)`. No SHA event is
executed by this analytical example.

The complete canonical operator still needs live fine-node preconditions,
grammar, target order, histories, operator steps, debt and lifecycle state,
as required by the [event contract](../../docs/contracts/OPERATOR_EVENTS.md).
SHA also writes latent/lifecycle metadata, including wall-clock metadata
distinct from structural time. Fine states with the same continuous
collective coordinates can have different admission contexts. The present
primary-map descent therefore does not establish a complete collective
SHA operator, an autonomous occurrence rule, a newly acquired history
sample or future freezing of both continuous rows.

The [weighted partition owner](../../src/tnfr/physics/relational_sine_weighted_partition.py)
provides `assess_sine_weighted_phase_offset_partition(source, blocks=...,
phase_offset_turns=...)`, returning `SineWeightedPhaseOffsetPartition`;
`.evaluate(...)` returns `SineWeightedPhaseOffsetState`.
`assess_sine_weighted_capacity_reset(partition, capacity_after=...)` and
`.capacity_reset(...)` assess actual positive represented output capacities.
Consumers re-admit primitive source state and rebuild coefficients rather
than trusting cached verdicts. They install no event or coarse runtime and
do not claim the supplied source already lies in the declared family.

The theorem allows real offsets; the implementation admits rational-turn
offsets and reuses Section 39's sound but incomplete symbolic/interval
recognizer. Its `certified`, `excluded` and `unavailable` results distinguish
proved identities, proved obstruction and undecided equality. This finite
recognizer is not a complete decision procedure for the mathematical iff.
The [independent controls](../../tests/physics/test_sine_weighted_phase_offset_partition.py)
check the weighted star, the two fine/coarse channels, inherited storage
and means, represented reset ratios, invalid primitive domains and
rebuilding of report-derived quantities.

The [unit partition owner](../../src/tnfr/physics/relational_sine_partition.py)
and the [support-involution owner](../../src/tnfr/physics/relational_sine_symmetry.py)
retain their narrower domains. This extension allows arbitrary supplied
blocks with capacity/degree compensation and distinct weighted/bare
couplings; it does not replace their results or the native joint quotient.
The star is a supplied mathematical control, not evidence that the retained
formed organization has this partition, that such a family forms
autonomously, or that its structural coefficients are physical observables.

## 39b. A transverse linear-stability band for the weighted star

<a id="sine-weighted-star-transverse-band"></a>

Retain exactly Section 39a's five-node weighted star, zero offsets and
complete conservative law, with held capacities

\[
 \nu=(4s,2s,2s,2s,8),\qquad \frac{23}{25}\leq s\leq1.
\]

This is a fixed band of supplied models and preparations. The parameter
`s` is constant along each segment, not a new capacity evolution law.
The question is whether all **infinitesimal transverse perturbations** of
its collective librations remain bounded. It differs from invariance of
the exactly prepared family, finite-error retention and formation.

### Full fine variation and the retained transverse directions

Let `L` be the unweighted four-node internal-star Laplacian. Along a
collective solution write `Delta=Theta_B-Theta_A` and `c=cos Delta`.
Use `a,phi in R^4` for fine form/phase perturbations on `A`, and `b,psi`
for the two perturbations on node `B`. Direct differentiation of **both**
consumed rows in the original time gives the full ten-dimensional system

\[
 \begin{aligned}
 \dot a&=-\frac{s}{\pi}[(L+cI)\phi-c\psi\mathbf1],&
 \dot b&=\frac{2c}{\pi}(\mathbf1^T\phi-4\psi),\\
 \dot\phi&=\frac{s}{\pi}[(3L+2I)a-2b\mathbf1],&
 \dot\psi&=\frac4\pi(4b-\mathbf1^Ta).
 \end{aligned}
\]

The factors `3` and `2` in the phase rows are the internal and external
conductances. The form rows instead use bare sine derivatives. They cannot
be obtained by replacing both channels with one weighted Laplacian.

Put `bar_a=1^T a/4`, `bar_phi=1^T phi/4`. Their rows with `b,psi` form the
four-dimensional collective variation:

\[
 \dot{\bar a}=\frac{sc}{\pi}(\psi-\bar\phi),\qquad
 \dot b=\frac{8c}{\pi}(\bar\phi-\psi),\qquad
 \dot{\bar\phi}=\frac{2s}{\pi}(\bar a-b),\qquad
 \dot\psi=\frac{16}{\pi}(b-\bar a).
\]

The complementary subspace

\[
 \mathbf1^Ta=\mathbf1^T\phi=0,\qquad b=\psi=0
\]

is invariant: the sums and the two `B` rows remain zero. Every perturbation
splits uniquely into these collective and transverse components. The
internal `L` eigenvalues on the zero-sum subspace are `1,1,4`. The
two-dimensional `lambda=1` eigenspace consists of leaf contrasts with zero
center component; `(3,-1,-1,-1)` spans the `lambda=4` eigenspace. Thus all
six transverse directions, including both copies of the first mode, obey
independent two-dimensional systems

\[
 \boxed{\quad
 \dot a_\lambda=-\frac{s}{\pi}(\lambda+\cos\Delta)\phi_\lambda,
 \qquad
 \dot\phi_\lambda=\frac{s}{\pi}(3\lambda+2)a_\lambda,
 \qquad \lambda=1,1,4.
 \quad}
\]

This is an exact decomposition of the full variational equation. It does
not discard a fine direction or replace the evolving collective state
by an independently forced waveform.

### The reference libration and its declared clock

For `u=X_A-X_B`, the collective rows are

\[
 \dot u=\frac{s+8}{\pi}\sin\Delta,\qquad
 \dot\Delta=-\frac{2(s+8)}\pi u,\qquad
 H_{\mathcal M}=4[u^2+1-\cos\Delta].
\]

Transform every row with the positive constant clock

\[
 \tau=\frac{\sqrt2(s+8)}\pi t.
\]

Primes in this subsection denote `d/dtau`. Then
`u'=sin Delta/sqrt(2)`, `Delta'=-sqrt(2)u` and
`Delta''+sin Delta=0`. On a nonconstant libration around consensus choose
the continuous lift with `|Delta|<=Delta_max<pi` and set

\[
 k=\sin(\Delta_{\max}/2),\qquad
 0<k\leq\tfrac14,\qquad
 u^2+1-\cos\Delta=2k^2,\qquad H_{\mathcal M}=8k^2.
\]

The conserved energy gives the period directly by the substitution
`sin(Delta/2)=k sin v` in a quarter-cycle quadrature:

\[
 P(k)=4\int_0^{\pi/2}\frac{dv}{\sqrt{1-k^2\sin^2v}},\qquad
 2\pi\leq P(k)\leq\frac{2\pi}{\sqrt{1-k^2}}<\frac{13}{2}.
\]

For the last strict bound, `k<=1/4`, `pi<22/7` and
`sqrt(15)>387/100` give `P<=8pi/sqrt(15)<17600/2709<13/2`.
The original-time period is `T=P*pi/[sqrt(2)(s+8)]`. No trajectory or
sampled period is needed for this enclosure. The full reference period
is used; no shorter coefficient period or relabeling is assumed.

Eliminating `a_lambda` using the constant coefficient in its phase row
gives, for `y=phi_lambda`,

\[
 y''+\alpha_\lambda(\lambda+\cos\Delta(\tau))y=0,\qquad
 \alpha_\lambda=\frac{s^2(3\lambda+2)}{2(s+8)^2}>0.
\]

The potential is periodic because the independently derived reference
librates. Its positivity alone would not prove stability: a periodic
coefficient can transfer amplitude between the two tangent coordinates.

### A rigorous rotation comparison for the whole return

Set

\[
 r_\lambda=\sqrt{\alpha_\lambda(\lambda+1)},\qquad
 q_\lambda(\tau)=\alpha_\lambda(\cos\Delta(\tau)-1),\qquad
 |q_\lambda|\leq2\alpha_\lambda k^2.
\]

For `z=(r_lambda y,y')`, its fundamental equation is

\[
 z'=(A_0+E(\tau))z,\quad
 A_0=\begin{pmatrix}0&r_\lambda\\-r_\lambda&0\end{pmatrix},\quad
 E=\begin{pmatrix}0&0\\-q_\lambda/r_\lambda&0\end{pmatrix}.
\]

The unperturbed fundamental matrix `R(tau)=exp(A_0 tau)` is orthogonal.
Writing the actual fundamental matrix as `R(tau)V(tau)` gives
`V'=R^{-1}ER V`, with `V(0)=I`. Its spectral-norm integral inequality
and Gronwall's inequality imply

\[
 \|V(P)-I\|_2\leq e^{B_\lambda}-1,\qquad
 B_\lambda=\frac{2\alpha_\lambda k^2P}{r_\lambda}
           =\frac{2r_\lambda k^2P}{\lambda+1}.
\]

This bound retains the time ordering of the full equation; it does not
replace the return by an exponential of an averaged stiffness. For its
monodromy matrix `M_lambda`, orthogonality and
`|tr D|<=2||D||_2` give

\[
 \boxed{\quad
 \left|\operatorname{tr}M_\lambda-2\cos(r_\lambda P)\right|
 \leq2(e^{B_\lambda}-1).
 \quad}
\]

The generator has trace zero, so the Wronskian identity gives
`det M_lambda=1` exactly. Constant changes of modal coordinates and the
positive clock conversion preserve this determinant and the return
eigenvalues. The comparison applies to every starting phase of the same
libration, since its amplitude and period bounds do not depend on that
phase.

### Uniform rational margins on the fixed band

Monotonicity of `s/(s+8)` and squaring positive rational endpoints give

\[
 r_1=\frac{s\sqrt5}{s+8}\in\left(\frac{23}{100},\frac14\right),
 \qquad
 r_4=\frac{s\sqrt{35}}{s+8}
             \in\left(\frac{61}{100},\frac{33}{50}\right).
\]

Together with the fixed amplitude and period ceilings, these imply

\[
 B_1<\frac{13}{128}<\frac19,\qquad
 B_4<\frac{429}{4000}<\frac19,\qquad
 e^{B_\lambda}-1\leq\frac{B_\lambda}{1-B_\lambda}<\frac18.
\]

The exponential inequality follows by comparison of its power series
with the geometric series for `0<=B<1`. The two reference-rotation
angles have elementary enclosures

\[
 \frac{23\pi}{50}<r_1P<\frac{13}{8},\qquad
 \frac{61\pi}{50}<r_4P<\frac{429}{100}.
\]

The first interval is within distance `1/5` of `pi/2`, using
`3<pi<22/7`; hence `|cos(r_1P)|<1/5`. The second lies strictly inside
`(pi,3pi/2)`, and its distance above `pi` is greater than `33/50`.
Monotonicity there and the alternating cosine upper bound give

\[
 |\cos(r_4P)|<\cos(33/50)
 \leq1-\frac{(33/50)^2}{2}+\frac{(33/50)^4}{24}
 =\frac{118515921}{150000000}<\frac45.
\]

Consequently the actual, unevaluated returns satisfy the strict bounds

\[
 \boxed{\qquad
 |\operatorname{tr}M_1|<\frac{13}{20}<2,
 \qquad
 |\operatorname{tr}M_4|<\frac{37}{20}<2.
 \qquad}
\]

These are uniform sufficient margins, not computed multipliers. Each
real two-dimensional return has determinant one and trace strictly
between `-2` and `2`, so its two eigenvalues are distinct conjugates of
unit modulus. It is diagonalizable, and all its positive and negative
integer powers are bounded. The fundamental matrix on one finite period
is bounded as well, proving bounded continuous-time motion of each
infinitesimal transverse mode. The repeated `lambda=1` sector is a direct
sum of two identical uncoupled blocks, so repetition does not introduce
a linear Jordan block. All six transverse directions are covered.

The boundedness can also be made uniform on the declared band. Write
`d=tr M_lambda/2=cos varphi`. The same comparison gives
`||M_lambda||_2<9/8` and `|d|<37/40`. Cayley--Hamilton gives, for every
integer `n`,

\[
 M_\lambda^n=\cos(n\varphi)I
 +\frac{\sin(n\varphi)}{\sin\varphi}(M_\lambda-dI),\qquad
 \|M_\lambda^n\|_2
 \leq1+\frac{9/8+37/40}{\sqrt{1-(37/40)^2}}
 =1+\frac{82}{\sqrt{231}}.
\]

The one-period propagator has norm below `9/8` at every intervening time,
so its composition with this bound controls all real times. The modal
coordinate changes and their inverses are uniformly bounded on the fixed
positive `s` band. This supplies uniform linear transverse boundedness;
it does not bound nonlinear errors or the separate collective variation.

### Equilibrium, represented SHA and the boundary of the conclusion

At `k=0`, `u=0` and `Delta=0` modulo `2pi`. This reference is stationary;
it has no nonzero minimal period. The transverse equations directly reduce
to `y''+r_lambda^2 y=0`, with `r_lambda>0`, so their solutions are bounded
without assigning a fictitious reference period or using a limiting
monodromy as measured data.

The supplied default SHA factor has the exact represented value

\[
 \widehat s=\frac{518151819521135}{562949953421312},\qquad
 \widehat s-\frac{23}{25}
 =\frac{5946559338199}{14073748835532800}>0,\qquad
 1-\widehat s=\frac{44798133900177}{562949953421312}>0.
\]

For this star's source capacities `4,2,2,2`, its represented products are
exact and have the common ratio `widehat s`, as established in Section
39a. Thus both the original `s=1` law and this hypothetical post-SHA law
lie in the proved band. A capacity-only reset keeps `u`, `Delta` and
`H_M=8k^2`, hence the libration amplitude parameter, unchanged. The new
period and transverse coefficients have nevertheless been re-admitted
above; they are not copied from the old law. This establishes conditional
linear stability on either held-capacity segment, not execution of SHA
or admissibility of its fine-node grammar and lifecycle.

Collective variations remain separate. Changing a libration's amplitude
can change its period and produce tangential phase shear; common form and
phase origins also belong to the collective sector. No boundedness claim
for a fixed full ten-dimensional reference trajectory follows from the
transverse result. Nor do unit-modulus linear multipliers establish
nonlinear orbital stability: finite perturbations can couple modes, and
the two equal-frequency leaf modes retain a possible nonlinear resonance.
There is no attraction, finite-error retention, acquisition, autonomous
formation or physical identification theorem here.

The [unit moving-window estimate](#sine-moving-pattern-window) and the
[replica-pulse variation](SINE_REPLICA_PULSE.md#sine-replica-pulse-variation)
supply related methods under their own supports and complete laws. Their
coefficients and verdicts were not transferred to this star. The present
result follows from its full fine linearization and scalar bounds on the
declared band, without a trajectory, return-matrix scan or fitted period.

The [shared transverse owner](../../src/tnfr/physics/relational_sine_weighted_transverse.py)
exposes `assess_sine_weighted_star_transverse_band`, returning
`SineWeightedStarTransverseBand`. It recomputes its period, frequency,
rotation and exponential bounds from the supplied capacity interval and
amplitude ceiling. Its generic outward cosine enclosure can be wider than
the hand-derived rational margins above; it independently tests whether
the resulting complete trace intervals lie strictly inside `(-2,2)`.
An inconclusive interval remains unavailable, not evidence of instability.
At zero amplitude, positive constant oscillators supply the verdict;
any reported period bound describes the nonzero-libration limit.
The [independent controls](../../tests/physics/test_sine_weighted_star_transverse.py)
check the full fine/modal equations, multiplicities, scalar bounds, clock,
representation and rejection boundaries without integrating a trajectory.

## 39c. Sufficient state of the weighted collective NFR and its boundary

<a id="sine-weighted-star-state-sufficiency"></a>

Retain the same weighted star, original structural time and held capacities
`(4s,2s,2s,2s,8)`, with `23/25<=s<=1`. The marked region
`A={0,1,2,3}` is the candidate collective NFR; node `B={4}` remains its
**dynamical boundary**, not an imposed waveform. Support, conductances and
the complete conservative law remain supplied. No forcing, capacity
evolution, event or trajectory is added. The following identities hold
for any fixed `s>0`, without requiring the exact collective family.

Following the
[ontology's state and language distinction](../EMERGENT_ONTOLOGY.md#collective-grammar-readiness),
a complete study has several responsibilities:

| Responsibility | Result or boundary in this model |
| --- | --- |
| Identity and boundary | Marked four-node `A` with distinguished center and three leaves, interacting with retained dynamical `B`; membership and support are supplied |
| Continuous state and law | The full internal coordinate extension below is invertible and sufficient for the specified ten-dimensional `A+B` dynamics |
| Maintenance | Sections 39a–39b prove exact-family invariance and the specified linear transverse band; [Section 39d](#sine-weighted-star-identity-retention) separately certifies finite-amplitude identity retention |
| Canonical actions | Fine histories, temporal eligibility, grammar, targets and lifecycle still require the shared event/word admission contract |
| Occurrence and interpretation | Autonomous formation, event selection and physical identification remain open |

A sufficient chart is not itself a newly formed NFR. The present question
is which observations retain enough information to predict the existing
fine law, not whether every NFR must have this support or representation.

### Declared lifts, collective coordinates and internal state

Use continuous real lifts of the primitive circular phases, recording
their initial representatives. Define arithmetic means on `A`, the
single-node boundary coordinates and zero-sum internal deviations:

\[
 \begin{gathered}
 X_A=\tfrac14\sum_{i\in A}x_i,\quad X_B=x_4,\quad
 \Theta_A=\tfrac14\sum_{i\in A}\theta_i,\quad\Theta_B=\theta_4,\\
 \xi_i=x_i-X_A,\quad\eta_i=\theta_i-\Theta_A,\quad
 \sum_{i\in A}\xi_i=\sum_{i\in A}\eta_i=0,\\
 u=X_A-X_B,\quad\Delta=\Theta_B-\Theta_A,\quad
 M=\tfrac14\sum_{i\in A}e^{\mathrm i\eta_i},\quad
 F=\operatorname{Im}(e^{\mathrm i\Delta}\overline M)
   =\tfrac14\sum_{i\in A}\sin(\Delta-\eta_i).
 \end{gathered}
\]

Here `M` uses the **positive** internal phase exponent and has modulus at
most one. `Theta_A` is the declared arithmetic lift mean, not `arg M` or
an inferred primitive phase. No angle is assigned to a vanishing
resultant. The lift registration makes these coordinates unambiguous even
when a circular mean is unavailable.

Let `L` be the bare internal-star Laplacian and `K=3L+2I`. Its nonlinear
internal sine row is

\[
 G_i(\eta)=\sum_{\substack{j\sim i\\j\in A}}\sin(\eta_j-\eta_i),
 \qquad \sum_{i\in A}G_i=0.
\]

The coefficients `3` and `2` in `K` come from internal and boundary
conductances. They do not weight `G_i` or `F`.

### Exact nonlinear collective and internal equations

Summing the fine form rows cancels every internal sine pair. Summing the
phase rows cancels the internal weighted form gradient. Therefore

\[
 \boxed{\quad
 \dot X_A=\frac{s}{\pi}F,\quad
 \dot X_B=-\frac8\pi F,\quad
 \dot\Theta_A=\frac{2s}{\pi}u,\quad
 \dot\Theta_B=-\frac{16}{\pi}u.
 \quad}
\]

Subtracting these means from the fine equations gives every internal row:

\[
 \boxed{\quad
 \dot\xi_i=\frac{s}{\pi}
       [G_i(\eta)+\sin(\Delta-\eta_i)-F],\qquad
 \dot\eta=\frac{s}{\pi}K\xi.
 \quad}
\]

Their sums vanish, so both zero-sum constraints persist. In particular,

\[
 \dot u=\frac{s+8}{\pi}F,\qquad
 \dot\Delta=-\frac{2(s+8)}{\pi}u.
\]

Section 39b's collective pendulum is recovered when `xi=eta=0`, hence
`M=1` and `F=sin Delta`. Away from that family the same complete law
supplies the correction; the boundary node and its reaction remain
part of the state.

Differentiating the internal moment exposes the next information need:

\[
 \boxed{\quad
 \dot M=\frac{\mathrm i s}{4\pi}
               \sum_{i\in A}e^{\mathrm i\eta_i}(K\xi)_i.
 \quad}
\]

Since `sum_i(K xi)_i=0`, the exponential can equivalently be replaced by
`exp(i eta_i)-M` in that sum. The derivative consumes a **mixed internal
form/phase moment**, including which form belongs to which phase and
internal edge. An unassociated phase distribution or form norm does not
retain that information.

The independently specified fine pressures are

\[
 p_i=\frac{G_i(\eta)+\sin(\Delta-\eta_i)}{\pi d_i}
 \quad(i\in A),\qquad p_B=-\frac{F}{\pi}.
\]

Consequently the aggregate form row preserves the nodal identity:

\[
 \dot X_A=\tfrac14\sum_{i\in A}\nu_i p_i=\frac{sF}{\pi},
 \qquad \dot X_B=8p_B.
\]

An average of nodal products is generally not the product of average
capacity and average pressure. For coarse nodal notation, one consistent
**declared normalization** is `nu_A=s`, `nu_B=8`,
`P_A=F/pi`, `P_B=-F/pi`. These pressures are defined from the fine
constitutive map before evaluating a response. This choice neither
selects a unique collective capacity nor removes the internal state
needed to evolve `F`; it is not pressure reconstructed from a measured
derivative.

### Storage, conserved means and lift covariance

The full fine storage becomes

\[
 \begin{aligned}
 H&=4u^2+\tfrac12\xi^TK\xi+V_{\mathrm{int}}(\eta)
           +4[1-\operatorname{Re}(e^{\mathrm i\Delta}\overline M)],\\
 V_{\mathrm{int}}(\eta)
   &=\sum_{\substack{\{i,j\}\in U\\i,j\in A}}
                              [1-\cos(\eta_j-\eta_i)].
 \end{aligned}
\]

Expanding the four weighted cross-edge form differences uses
`sum_i(u+xi_i)^2=4u^2+sum_i xi_i^2`. The phase storage retains the
full complex moment. Both `H` and the weighted sums

\[
 Q_x=\frac4sX_A+\frac12X_B,\qquad
 Q_\theta=\frac4s\Theta_A+\frac12\Theta_B
\]

are conserved. Dividing by `4/s+1/2` gives the full fine normalized
means. No internal term is omitted: the fine inverse-capacity weights
are `1/s` throughout `A` and both internal deviations have zero sum.

For comparison let `H_col=4[u^2+1-cos Delta]`. Its exact off-family
remainder and exchange are

\[
 \begin{aligned}
 R&=H-H_{\mathrm{col}}
   =\tfrac12\xi^TK\xi+V_{\mathrm{int}}(\eta)
       +4[\cos\Delta-\operatorname{Re}(e^{\mathrm i\Delta}\overline M)],\\
 \dot H_{\mathrm{col}}
   &=\frac{8(s+8)}{\pi}u(F-\sin\Delta),\qquad
 \dot R=-\dot H_{\mathrm{col}}.
 \end{aligned}
\]

This is exchange between a chosen collective observation and its
internal remainder, without an added reservoir or changed total storage.
Neither `R>=0` nor conservation of `H_col` holds in general.
In particular, `H/8` is the squared libration amplitude parameter only
on the exact family; it does not identify an off-family amplitude or
supply Section 39b's reference orbit.

For integer changes of lift `theta_i -> theta_i+2pi n_i`, with
`bar_n=sum_(i in A) n_i/4` and boundary integer `n_B`, one obtains

\[
 \begin{gathered}
 \Theta_A^+=\Theta_A+2\pi\bar n,\quad
 \eta_i^+=\eta_i+2\pi(n_i-\bar n),\quad
 M^+=e^{-2\pi\mathrm i\bar n}M,\\
 \Delta^+=\Delta+2\pi(n_B-\bar n).
 \end{gathered}
\]

Thus `exp(i Delta) conj(M)`, fine pressures and rates, and full `H` are
unchanged. The registered lifted mean instead changes by
`2pi[(sum_A n_i)/s+n_B/2]/(4/s+1/2)`. The observation `H_col` and its
remainder can change too. For example, let every form and physical phase
be zero, but use `theta_0=2pi` and every other phase lift zero. Then
`H=0`, `Theta_A=pi/2`, `Delta=-pi/2`, `M=-i`, `H_col=4` and `R=-4`.
A globally nonnegative remainder claim would be false even for this
physical equilibrium. Positivity needs an independently admitted phase
chart and its convexity premises.

### Matched means and full storage do not determine collective response

For an autonomous first-order law on a proposed observation `Y=O(z)`,
`DO(z)f(z)` must agree for every fine state with the same `Y`. This is
the continuous part of the
[shared descent criterion](../STRUCTURAL_OPERATORS.md#collective-operator-descent).
The following comparisons keep support, law, capacities, clock and
lift convention fixed; equality of their observations is exact.

Set all forms and both collective phase means to zero. For `0<a<pi/2`,
compare

\[
 \xi=0,\qquad \eta^\pm=\pm a(3,-1,-1,-1).
\]

Both states have the same four collective coordinates, conserved fine
means and full storage

\[
 H^\pm=3(1-\cos4a)+(1-\cos3a)+3(1-\cos a).
\]

But the triple-angle identities give

\[
 M_\pm=\cos^3a\mp\mathrm i\sin^3a,\quad
 F_\pm=\pm\sin^3a,\quad
 \dot X_A^\pm=\pm\frac{s}{\pi}\sin^3a,\quad
 \dot X_B^\pm=\mp\frac8\pi\sin^3a.
\]

The proposed state `(X_A,X_B,Theta_A,Theta_B,H)` has two different
derivatives at the same value. No single-valued autonomous first-order
vector field on that observation represents all these fine states.
As `a` approaches zero the witnesses approach the exact consensus
family; this is not a large-amplitude exception.

### Adding the complex moment and storage still does not close

Let `e=(0,1,-1,0)`, so `L e=e` and `K e=5e`. For `a,b>0` with
`a<=pi/2` prepare

\[
 X_A=X_B=\Theta_A=0,\quad\Theta_B=\delta,\qquad
 \eta=ae,\quad\xi^\pm=\pm be .
\]

The phase vector is identical in both states. Their moment and full
storage are

\[
 M=\frac{1+\cos a}{2},\qquad
 H^\pm=5b^2+2(1-\cos a)+4-2(1+\cos a)\cos\delta .
\]

Thus collective coordinates, both components of `M`, full `H`,
conserved means and all instantaneous collective rates agree. The
exact mixed moment nevertheless gives

\[
 \boxed{\qquad \dot M_\pm=\mp\frac{5sb}{2\pi}\sin a\ne0.\qquad}
\]

The augmented observation
`(X_A,X_B,Theta_A,Theta_B,Re M,Im M,H)` also fails projectability.
With `delta=0` and `a,b` tending to zero, both states approach the same
equilibrium family while their derivative contrast remains nonzero for
every positive parameter choice. The distinction is internal form/phase
association, not a different law, operator or environment.

These instantaneous comparisons exclude the two named reduced states.
They do not measure accumulated future separation, show that every
finite moment hierarchy fails, or identify a minimal sufficient state.
Linear transverse boundedness does not erase the missing information.

### An invertible sufficient extension and its symmetries

Retain the four collective coordinates and every `xi_i,eta_i` subject
to the two zero-sum constraints. The inverse is explicit:

\[
 x_i=X_A+\xi_i,\quad\theta_i=\Theta_A+\eta_i\quad(i\in A),
 \qquad x_4=X_B,\quad\theta_4=\Theta_B.
\]

There are four collective and six independent internal coordinates,
exactly the original ten continuous degrees of freedom in the declared
lift chart. The equations above are closed and reconstruct both fine
rows, pressures and storage. This is a sufficient organization of the
complete state, not dimension reduction or a minimality proof. Retaining
the dynamical `B` is essential; this extension does not make `A` autonomous.

The continuous law respects a common form translation, a common phase
rotation and simultaneous permutations of the three identical leaves.
The center and boundary remain distinguished. A leaf-label quotient
must keep the joint form/phase pairing and edge/target associations
consumed by the intended observations or actions. Permuting forms
independently of phases is not that symmetry. Common origins may be
discarded for an admitted relative-flow question; actions consuming
absolute form or marked constituents need their own descent proof.

The obstructions identify information lost by smaller observations.
One can retain more state as above, or derive a causal memory description
with its hidden initialization and boundary premises. No memory kernel
is supplied by the witnesses, and a stored operator history cannot
replace it. This is the
[ontology's distinction between dynamical memory and execution history](../EMERGENT_ONTOLOGY.md#collective-memory-and-operators),
not evidence that the fine law needs a new force or operator.

Continuous sufficiency likewise supplies no measured temporal eligibility,
grammar debt, glyph history, lifecycle state, target marks, stored-pressure
provenance or configuration consumed by canonical actions. The
[ordered-word admission contract](../STRUCTURAL_OPERATORS.md#collective-grammar-descent)
must hold for the selected actions and reachable successors. This state
analysis executes none and selects no autonomous event. Continuous
sufficiency alone proves no nonlinear stability or finite-error retention;
[Section 39d](#sine-weighted-star-identity-retention) supplies a separate
identity certificate. Neither proves formation or physical identification.

### Shared observation and reconstruction

The [weighted-state owner](../../src/tnfr/physics/relational_sine_weighted_state.py)
provides `assess_sine_weighted_star_state` and
`reconstruct_sine_weighted_star_state`. Exact rational-turn phase lifts
retain registration; the equations above use radians and original-time
rates. Reconstruction enforces zero-sum constraints rather than
projecting incompatible supplied state.
The [independent controls](../../tests/physics/test_sine_weighted_star_state.py)
compare fine and collective/internal equations, storage, lift covariance
and both matched-state obstructions. These are detached computations
without a trajectory, operator event or replay of frozen evidence.

## 39d. All-time retention of a declared finite-width internal identity

<a id="sine-weighted-star-identity-retention"></a>

The complete state of Section 39c permits a separate nonlinear geometric
question. Retain the same positive weighted support, conservative
weighted-form/bare-sine law, original clock and strictly positive held
capacities. Declare the following identity of the marked four-node
region `A`:

\[
 |x_0-x_j|<\tfrac12,\qquad
 d_{\mathbb S^1}(\theta_0,\theta_j)<\tfrac\pi2,
 \qquad j=1,2,3.
\]

The phase distance is the absolute principal circular difference.
These tolerances are a supplied identity policy for this mathematical
organization, not universal NFR constants or a replacement for other
identity definitions. The boundary node `B` remains dynamical and
contributes to the full storage. Internal storage alone is not the
certificate.

### The exact form barrier uses the whole weighted network

Fix one center-to-leaf form difference `a=x_0-x_1`. Common translation
allows `x_1=0`, `x_0=a` when minimizing the nonnegative form storage

\[
 H_x=\tfrac12\sum_{\{i,j\}\in U}W_{ij}(x_i-x_j)^2.
\]

The free-coordinate quadratic is strictly convex. Its unique minimizer
has equal other-leaf forms `x_2=x_3=v` and boundary form `x_4=w`.
The stationary equations are

\[
 5v=3a+2w,\qquad 4w=a+2v,
 \qquad
 v=\tfrac78a,\quad w=\tfrac{11}{16}a.
\]

The weighted gradient at node `0` at this minimizer is

\[
 3a+6(a-v)+2(a-w)=\tfrac{35}{8}a.
\]

Equivalently the effective conductance between the fixed terminals is
`35/8`. Substitution into the quadratic gives the exact constrained
minimum and, by leaf symmetry, the inequality for every internal arm:

\[
 \min H_x=\frac{35}{16}a^2,\qquad
 H\geq H_x\geq\frac{35}{16}(x_0-x_j)^2
 \quad(j=1,2,3).
\]

Reaching `|x_0-x_j|=1/2` therefore costs at least `35/64` in full storage.
This improves on keeping only the weight-three internal edge, because
the remaining positive support also constrains the contrast.

For the phase condition, one offending bare edge contributes

\[
 d_{\mathbb S^1}(\theta_0,\theta_j)\geq\tfrac\pi2
 \quad\Longrightarrow\quad
 1-\cos(\theta_j-\theta_0)\geq1.
\]

All other terms in `H` are nonnegative. This phase lower bound alone
exceeds the form barrier; it need not be the sharp constrained phase
minimum.

### Strict full-storage admission gives all-time identity retention

Consequently

\[
 \boxed{\qquad H<\frac{35}{64}\quad\Longrightarrow\quad
 |x_0-x_j|<\frac12,\quad
 d_{\mathbb S^1}(\theta_0,\theta_j)<\frac\pi2
 \quad(j=1,2,3).\qquad}
\]

The phase conclusion actually holds on every bare edge because
`35/64<1`. The form bound can be read quantitatively as
`|x_0-x_j|<=4 sqrt(H/35)`. The full fine field in continuous lifts is
globally Lipschitz: its form row is a finite sine sum and its phase row
is linear in form, with fixed finite coefficients. Solutions exist
uniquely for all real times and conserve `H`. The displayed implication
therefore holds for all time from any admitted state, including finite
internal and boundary deviations from the exact collective family.
No linearization, sampled endpoint or phase-chart assumption is needed.

The strict inequality matters. With equal phases and
`(x_0,x_1,x_2,x_3,x_4)=(1/2,0,7/16,7/16,11/32)`, one has exactly
`H=35/64` and an internal form difference on the excluded boundary.
Thus this is the exact minimum storage outside the declared strict
identity. It is a sufficient all-time certificate, not a claim that
states with greater storage lose identity or ever reach its boundary.
An unresolved numerical storage enclosure likewise cannot certify loss.

This is retention, not acquisition: the same inequality already places
the initial state inside the identity, and its conservation applies in
both time directions. It does not say that a preparation lacking the
identity evolves into it.

### An explicit open neighborhood of the entire coherent band

Every exact collective reference in Section 39b's band has
`H_*=8k^2<=1/2`, leaving the positive gap `3/64` to the barrier.
The neighborhood need not be restricted to an internal mode or to a
fixed collective phase. A uniform bound follows from the squared-norm
representation

\[
 H=\|\mathcal V(x,\theta)\|_2^2,\qquad
 \mathcal V=
 \left(
  \left(\sqrt{W_{ij}/2}(x_i-x_j)\right)_{\{i,j\}\in U},
  \left(\sqrt2\sin((\theta_j-\theta_i)/2)\right)_{\{i,j\}\in U}
 \right).
\]

Keep the edge orientations and phase lifts fixed when comparing a
reference and its perturbation. If every one of the five form errors
has magnitude at most `epsilon_x` and every lifted phase error at most
`epsilon_theta` radians, the global sine Lipschitz bound gives

\[
 \|\mathcal V-\mathcal V_*\|_2^2
 \leq34\epsilon_x^2+14\epsilon_\theta^2,\qquad
 \sqrt H\leq\sqrt{H_*}
                   +\sqrt{34\epsilon_x^2+14\epsilon_\theta^2}.
\]

The constants use `sum_e W_e=17` and seven bare edges: each edge
difference error is at most twice the corresponding nodal error.
This retains both fine channels and the boundary perturbations.

For `epsilon_x=1/256` in the declared form units and
`epsilon_theta=1/256` radians, any reference with `H_*<=1/2` satisfies

\[
 H\leq\left(\frac1{\sqrt2}+\frac{\sqrt3}{64}\right)^2
 =\frac12+\frac{\sqrt6}{64}+\frac3{4096}
 <\frac{2211}{4096}
 <\frac{2240}{4096}=\frac{35}{64}.
\]

The strict rational comparison uses `sqrt(6)<5/2` and leaves a storage
margin greater than `29/4096`. In exact-turn coordinates, nodal phase
errors at most `1/2048` turn are sufficient, since
`2pi/2048<1/256` radians. The box includes all ten fine form/phase
coordinates and has nonempty interior around every reference in the
coherent band. In fact the same estimate applies to any full reference
with `H_*<=1/2`, whether or not it belongs to the exact family.

Every state in these boxes retains the declared internal identity for
all time under its admitted complete law. Its internal motion, collective
amplitude and accumulated phase can differ from those of the reference.
This proves finite-amplitude geometric retention, not nonlinear tracking
of a chosen periodic orbit, attraction or convergence.

### Capacity-only changes and the execution boundary

Storage contains no capacity term. A hypothetical capacity-only reset
preserving forms, phases, support and conductances therefore preserves
`H` and this certificate. Each resulting well-defined segment with
strictly positive held capacities and the same conservative rows again
retains the identity. Exact-family preservation is a different condition:
Section 39a requires block-constant actual capacity ratios, whereas this
storage argument also covers a singleton reset that breaks that family.
The structural geometry can remain bounded while its internal motion
ceases to be a synchronized collective waveform.

This statement neither executes SHA nor bypasses its represented output,
live grammar, target and lifecycle requirements. Other operator maps can
change forms, phases or support and must account for their own storage
jumps. No previous trajectory certificate is transferred through an
unspecified event.

The [shared weighted-state owner](../../src/tnfr/physics/relational_sine_weighted_state.py)
provides `assess_sine_weighted_star_retention(state)` and
`SineWeightedStarRetention`. It rebuilds the complete primitive state and
full storage before comparing its upper enclosure strictly with `35/64`.
This reader retains its declared capacity family `(4s,2s,2s,2s,8)`;
applying the broader capacity-reset corollary requires independent
admission of the actual successor's complete law.
The [independent controls](../../tests/physics/test_sine_weighted_star_state.py)
cover the form minimizer, strict boundary, stored-report rebuilding and
capacity independence. The result complements the sufficient state and
linear band: it supplies the declared finite-width identity guarantee,
without proving autonomous formation or physical identification.

## 39e. Composition of maintained identities with a dynamic boundary bridge

<a id="sine-weighted-star-maintenance-composition"></a>

Two individually maintained regions need not inherit their isolated
certificates after contact. This section supplies one exact composition
law, accounts for its support work and proves a nonempty preparation
domain with nonzero exchange and retention of both declared identities.
It retains all constituent and boundary state; it does not infer a new
autonomous organization from contact.

### Joined support, complete state and actual normalization

Take two copies `r=1,2` of Section 39c's five-node weighted star, both
with held scale `s=1`. Each has its marked center, three leaves and
dynamic boundary `B_r`. Keep every original weight and capacity and
supply one undirected boundary edge `{B_1,B_2}` of conductance one.
The joined graph has ten nodes, fifteen bare edges and total
conductance `35`. Its capacities are two copies of `(4,2,2,2,8)`,
but its bare degrees are two copies of `(4,2,2,2,5)`.

On this actual support the complete conservative law is

\[
 S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i),\quad
 q_i=(L_Wx)_i,\quad
 p_i=\frac{S_i}{\pi d_i},\qquad
 \dot x_i=\nu_i p_i,\quad
 \dot\theta_i=\frac{\nu_i}{\pi d_i}q_i .
\]

Use original structural time `t` and registered continuous phase lifts.
Capacity is held positive; there is no forcing, loss or further support
event. The bridge is bare phase support as well as weighted form
support. In particular, reducing a hypothetical conductance would not
remove its sine term or undo the degree change. This result concerns
the declared unit bridge, without a weak-contact limit.

For each copy retain the complete chart
`(X_Ar,X_Br,Theta_Ar,Theta_Br,xi_r,eta_r)` from Section 39c, with
both internal deviations summing to zero. Define

\[
 \begin{gathered}
 u_r=X_{Ar}-X_{Br},\quad
 \Delta_r=\Theta_{Br}-\Theta_{Ar},\quad
 M_r=\tfrac14\sum_{i\in A_r}e^{\mathrm i\eta_{ri}},\quad
 F_r=\operatorname{Im}(e^{\mathrm i\Delta_r}\overline M_r),\\
 f=X_{B1}-X_{B2},\quad
 \delta=\Theta_{B2}-\Theta_{B1},\qquad
 \sigma_1=1,\quad\sigma_2=-1 .
 \end{gathered}
\]

The two charts have twenty independent real coordinates in the chosen
lifts. The displayed `f,delta,u_r,Delta_r,M_r,F_r` are functions of
that state, not extra independent coordinates or a substitute for
the internal form/phase associations.

The four `A_r` nodes have their old degrees and mobility one. Therefore
their mean and internal rows remain

\[
 \begin{aligned}
 \dot X_{Ar}&=F_r/\pi,&\qquad
 \dot\Theta_{Ar}&=2u_r/\pi,\\
 \dot\xi_{ri}
   &=[G_i(\eta_r)+\sin(\Delta_r-\eta_{ri})-F_r]/\pi,&
 \dot\eta_r&=K\xi_r/\pi ,
 \end{aligned}
\]

with the same `G` and `K=3L+2I` as before. The boundary mobility is
now `8/5`, so its independent pressure and two rows are

\[
 \boxed{\quad
 p_{Br}=\frac{-4F_r+\sigma_r\sin\delta}{5\pi},\qquad
 \dot X_{Br}=\frac8{5\pi}[-4F_r+\sigma_r\sin\delta],\qquad
 \dot\Theta_{Br}=\frac8{5\pi}[-8u_r+\sigma_r f].
 \quad}
\]

Both rows must change together. Reusing the old `d_B=4` normalization
and simply adding a bridge current would be a different law. Useful
consequences of the actual rows are

\[
 \begin{aligned}
 \dot u_r&=\frac{37F_r-8\sigma_r\sin\delta}{5\pi},&
 \dot\Delta_r&=\frac{-74u_r+8\sigma_r f}{5\pi},\\
 \dot f&=\frac{32(F_2-F_1)+16\sin\delta}{5\pi},&
 \dot\delta&=\frac{64(u_1-u_2)-16f}{5\pi}.
 \end{aligned}
\]

These equations with the internal rows are a sufficient exact chart
of the joined fine law. They neither discard the two dynamic boundaries
nor close on the displayed collective variables alone.

### Insertion work, component supply and conserved joint storage

Let `H_r` be the full five-node storage of copy `r`, including its
four original boundary edges. At the same primitive forms and phases,
the joined storage is exactly

\[
 \boxed{\qquad
 H_{\mathrm{join}}=H_1+H_2+E_b,\qquad
 E_b=\tfrac12f^2+1-\cos\delta .
 \qquad}
\]

Thus `E_b>=0` is the support-insertion storage work for this supplied
comparison. The degree changes do not add a term to storage, but they
do change its subsequent field. No unrecorded loss or reservoir pays
this work, and this comparison is not a runtime operator event.

For any complete joined state,
\(\partial H_{\mathrm{join}}/\partial x_i=q_i\) and
\(\partial H_{\mathrm{join}}/\partial\theta_i=-S_i\). Consequently
the derivative cancels separately at each node:

\[
 \dot H_{\mathrm{join}}
 =\sum_i\frac{\nu_i}{\pi d_i}(q_iS_i-S_iq_i)=0 .
\]

The individual `H_r` are generally not conserved. At their boundary
their own gradients are `q_{Br}^{(r)}=-8u_r` and
`S_{Br}^{(r)}=-4F_r`. Internal cancellation still holds with the
actual boundary mobility; the bridge supplies the remaining terms:

\[
 \boxed{\qquad
 \dot H_r=\frac{8\sigma_r}{5\pi}
                 [-8u_r\sin\delta+4F_rf].
 \qquad}
\]

This is a full-storage supply balance, not just a form-channel balance.
The bridge storage has derivative

\[
 \dot E_b
   =f\dot f+\sin\delta\,\dot\delta
   =\frac{32f(F_2-F_1)+64(u_1-u_2)\sin\delta}{5\pi}
   =-\dot H_1-\dot H_2 .
\]

The two supplies need not be negatives of one another: the bridge can
temporarily store the difference. The original isolated certificates
therefore cannot be applied as if `H_1` and `H_2` remained invariant.

The actual inverse-capacity weights `d_i/nu_i` are one on each
`A_r` node and `5/8` at each boundary. With

\[
 Q_{xr}=4X_{Ar}+\tfrac58X_{Br},\qquad
 Q_{\theta r}=4\Theta_{Ar}+\tfrac58\Theta_{Br},
\]

their two exchange channels are

\[
 \dot Q_{xr}=\frac{\sigma_r}{\pi}\sin\delta,\qquad
 \dot Q_{\theta r}=\frac{\sigma_r}{\pi}f .
\]

Both joined sums `Q_x1+Q_x2` and `Q_theta1+Q_theta2` are conserved;
division by the total weight `37/4` gives the normalized fine means.
The phase mean retains its declared lift registration. The former
isolated boundary weight `1/2` does not transfer through the support
change.

### The exact joint identity sublevel

Declare both copies' three center-leaf form and circular phase
inequalities from Section 39d. This is the conjunction of six internal
arm identities; it adds no condition on the bridge itself. Nonnegative
storage gives, for each copy and each of its leaves,

\[
 H_{\mathrm{join}}\geq H_r
       \geq \frac{35}{16}(x_{r0}-x_{rj})^2,\qquad j=1,2,3 .
\]

Any internal phase gap at least `pi/2` also contributes at least one
to the joined storage. Hence

\[
 \boxed{\qquad
 H_{\mathrm{join}}<\frac{35}{64}
 \quad\Longrightarrow\quad
 \text{both declared internal identities persist for all real time}.
 \qquad}
\]

As before the complete lifted field is globally Lipschitz, so the
conserved storage and pointwise inclusion give a global result. The
phase bound additionally keeps every bare edge, including the bridge,
strictly within a circular gap of `pi/2`. Quantitatively every internal
form gap is at most `4 sqrt(H_join/35)`.

The joint threshold remains sharp despite adding support. Take the
first copy's forms as `(1/2,0,7/16,7/16,11/32)`, all five forms of
the second copy equal to `11/32` and every phase equal. Then the
bridge and second copy have zero storage, while the first copy reaches
one form boundary with exactly `H_join=35/64`. This establishes the
minimum storage outside the joint strict identity. It does not say
that greater storage causes escape or that every identity-retaining
state must pass this sufficient certificate.

### A strict source with exchange and a full twenty-coordinate neighborhood

At the reference preparation, both copies have all four `A` forms
equal to `1/16` and their boundary forms zero. All phases in copy one
are zero, and all phases in copy two are `pi/8`, namely `1/16` turn.
Thus

\[
 u_1=u_2=\tfrac1{16},\quad f=0,\quad
 \Delta_1=\Delta_2=0,\quad F_1=F_2=0,\quad
 \delta=\tfrac\pi8,\qquad
 H_1=H_2=\tfrac1{64}.
\]

The source storage and initial supply are

\[
 \begin{gathered}
 H_*=\tfrac1{32}+1-\cos(\pi/8)<\tfrac18<\tfrac{35}{64},\\
 \dot H_1=-\frac4{5\pi}\sin(\pi/8)<0,\qquad
 \dot H_2=\frac4{5\pi}\sin(\pi/8)>0,\qquad
 \dot E_b=0 .
 \end{gathered}
\]

For the first strict inequality use
`1-cos z<=z^2/2` and `pi^2<12`. The zero instantaneous bridge-storage
derivative does not eliminate its nonzero sine current or the transfer
between component storages.

The witness is not confined to exact internal coherence. Perturb every
one of the ten forms by at most `epsilon_x` and every one of the ten
phase lifts by at most `epsilon_theta` radians. Section 39d's edge-norm
representation on the joined support yields

\[
 \sqrt{H_{\mathrm{join}}}
 \leq\sqrt{H_*}
       +\sqrt{70\epsilon_x^2+30\epsilon_\theta^2}.
\]

The constants are twice the total conductance `35` and twice the
fifteen bare-edge count, respectively. For both errors at most `1/256`,
`sqrt(H_*)<3/8` gives the strict rational bound

\[
 H_{\mathrm{join}}<
       \left(\frac38+\frac5{128}\right)^2
       =\frac{2809}{16384}
       <\frac{8960}{16384}=\frac{35}{64}.
\]

The same box preserves the initial supply signs without a trajectory.
For both copies,

\[
 u_r\geq\tfrac7{128},\qquad
 |F_r|\leq\tfrac1{128},\qquad |f|\leq\tfrac1{128},\qquad
 |\delta-\pi/8|\leq\tfrac1{128}.
\]

The contact-sine bound follows from its four initially zero differences
and `|sin z|<=|z|`. For the bridge,
`47/128<delta<pi/2`. Monotonicity of sine and its elementary lower
bound on this interval give

\[
 \sin\delta>\sin(47/128)
 \geq \frac{47}{128}-\frac16\left(\frac{47}{128}\right)^3
 >\frac13 .
\]

Consequently each component's bracket satisfies

\[
 -8u_r\sin\delta+4F_rf
 <-\frac7{48}+\frac1{4096}<-\frac18,
 \qquad
 \dot H_1<-\frac1{5\pi},\quad
 \dot H_2>\frac1{5\pi}.
\]

All twenty coordinates can vary independently in the interior of this
box. Phase errors at most `1/2048` turn suffice for the radian bound.
Every such initial state has nonzero exchange and retains both
identities for all time under the joined law. The supply signs are
initial-time conclusions; neither their future persistence nor
monotonic component-storage transfer is asserted.

### Individually certified does not imply jointly certified

Keep the same forms and coherent phase within each copy, but shift the
entire second copy by `pi/2` instead. Both isolated source storages
remain `1/64`, and both internal identities are initially present.
The supplied bridge adds one, giving

\[
 H_{\mathrm{join}}=\tfrac1{32}+1=\tfrac{33}{32}
                       >\tfrac{35}{64}.
\]

The individual old-law certificates pass while the sufficient joint
storage certificate is unavailable. In this particular control, the
internal rows give an independent exact answer: `xi_r=eta_r=0` is
invariant for each copy, since `G=0`, `F_r=sin Delta_r` and `K xi_r=0`.
Every center-leaf form and phase difference therefore stays zero for
all time, whatever the retained boundary motion. Thus the joint
storage certificate is demonstrably sufficient but not necessary for
retention. Its unavailability is not identity loss or a solver failure.
The storage comparison isolates the additional joint preparation and
support-work obligation of this particular certificate.

### Shared assessment and canonical boundary

The [contact owner](../../src/tnfr/physics/relational_sine_weighted_contact.py)
provides `assess_sine_weighted_star_contact(left,right)` and
`SineWeightedStarContact`. It re-admits both full source states at
exact scale one, reconstructs the joined field and pressure from the
new support, and independently checks them against the sufficient
charts. The old source rate fields stay associated with their isolated
laws. Joined storage, supplies, means and the strict identity bound
belong to the new law. The
[independent controls](../../tests/physics/test_sine_weighted_star_contact.py)
cover the fine chain rule, degree change, insertion work, strict
barrier and unavailable control.

This establishes compatibility of maintained identities with nonzero
exchange on a supplied complete support. It does not establish
autonomous bridge formation, a larger NFR identity, attraction or
physical identification. An actual UM or another canonical event must
separately admit its targets, grammar, live configuration, history and
complete output, following the
[collective action contract](../STRUCTURAL_OPERATORS.md#collective-operator-descent)
and [event owner](../../docs/contracts/OPERATOR_EVENTS.md).
An event producing different capacities, conductances or support must
re-admit that successor law; neither this detached bridge nor the
old isolated certificates can substitute for those premises.

## 40. A finite-width window around compatible moving geometry

<a id="sine-moving-pattern-window"></a>

Section 39's exact family is not an acquisition target that can be entered
from outside at finite time. A neighborhood with nonzero width has a
different obligation: errors in both the receiver and its environment must
remain controlled while the reference moves. The following sufficient
certificate uses the actual storage and complete conservative field. It
proves finite retention for an independently supplied entry neighborhood;
it does not produce that neighborhood from an earlier preparation.

### Reference motion, error chart and declared duration

Retain the C5 with five private leaves, unit capacities, `e=0`, `w=beta=1`,
no input or events, and `tau=t/pi`. Use Section 39's certified two-block
family with receiver winding `sigma=+1` or `-1`, matching receiver/leaf
phase offsets and `alpha=2pi/5`. Reference block forms are `a,b`, their
phases are `Theta,Psi`, and

\[
u=a-b,\qquad \phi=\Psi-\Theta,\qquad
h=\tfrac12u^2+1-\cos\phi.
\]

The inherited pendulum preserves `h` and satisfies
`|phi'|<=4sqrt(2h)/3`. Choose a radian contact envelope
`0<Gamma<pi/2`, with `|phi(0)|<Gamma` and
`h<1-cos(Gamma)`. Energy and continuity then keep the reference contact
phase in `(-Gamma,Gamma)` for all positive and negative times. This
includes the zero-pulse reference; a nonzero pulse is not required.

Fix a finite duration `T>0` in `tau` and two radian error radii:

\[
0<\rho_R<\frac\pi{10},\qquad
0<\rho_C<\frac\pi2-\Gamma.
\]

Set `kappa_R=cos(alpha+rho_R)>0` and
`kappa_C=cos(Gamma+rho_C)>0`. The internal receiver and contact errors
use different margins. Contact convexity is a sufficient premise of this
certificate, not a defining property of a regional identity. Exact
compatible families can also move through nonacute contacts; this estimate
does not assess their transverse robustness.

For an actual full solution and the reference, use continuously matched
phase lifts. On oriented edge `i->j`, write
`f_e=x_i-x_j`, `delta_e=theta_j-theta_i`,
`xi_e=f_e-f_e^*` and `eta_e=delta_e-delta_e^*`.
Admit the initial chart separately: `|eta_e(0)|<rho_R` on receiver
edges and `<rho_C` on contacts. A small scalar storage value alone does
not establish these phase-chart premises.

### The moving-reference relative-storage identity

For `U(delta)=1-cos(delta)`, define

\[
D=\frac12\sum_e\xi_e^2
 +\sum_e\left[U(\delta_e^*+\eta_e)-U(\delta_e^*)
                    -\sin(\delta_e^*)\eta_e\right].
\]

This is the Bregman remainder of the existing full storage, not another
physical energy or a new evolution rule. The complete field has the
constant reciprocal matrix with blocks `0,-K;K,0`. Differentiating the
remainder along both actual solutions cancels the quadratic form terms
and gives exactly

\[
\boxed{D'
=\sum_e(\delta_e^*)'
 [\sin(\delta_e^*+\eta_e)-\sin(\delta_e^*)
                          -\cos(\delta_e^*)\eta_e].}
\]

Internal reference gaps are fixed at `sigma*alpha`; only the five
contacts contribute, each with reference gap `phi`. While the admitted
chart holds, convexity and Taylor's remainder imply

\[
D\ge\frac12\sum_e\xi_e^2
 +\frac{\kappa_R}2\sum_{e\in R}\eta_e^2
 +\frac{\kappa_C}2\sum_{e\in C}\eta_e^2,
\qquad
|D'|\le\frac{|\phi'|}{\kappa_C}D\le\lambda D,
\]

where

\[
\lambda=\frac{4\sqrt{2h}}{3\kappa_C},\qquad
d_* =\frac12\min\{\kappa_R\rho_R^2,\kappa_C\rho_C^2\}.
\]

Thus a proved initial upper bound `D(0)<=D_0`, together with the initial
chart and the prospective strict inequality

\[
\boxed{D_0e^{\lambda T}<d_*}
\]

retains the complete error chart for every `|tau|<=T`. Indeed a first
chart exit would require `D>=d_*`, contradicting the integrated bound.
This is a full-interval proof, not a claim inferred from sampled endpoints.

For independent initial nodal form/phase error radii `r_i,p_i>=0`,
the shared sufficient initial bound is

\[
D_0=\frac12\sum_{\{i,j\}\in E}
       [(r_i+r_j)^2+(p_i+p_j)^2].
\]

The same component sums must pass the separate initial edge-chart tests.
During the window, every edge form error is at most `sqrt(2D_max)`;
receiver and contact phase errors are at most
`sqrt(2D_max/kappa_R)` and `sqrt(2D_max/kappa_C)`, respectively, where
`D_max=D_0 exp(lambda T)`. Initial weighted form/phase mean errors remain
their conserved values. Retaining those origins with the edge bounds
controls the relative full state on this connected support; no hidden
environmental coordinate has been replaced by an imposed boundary drive.

### Internal identity and actual work remain compatible

The reference receiver has uniform form and equal oriented principal gaps
`sigma*alpha`. Its internal linear phase term cancels because
`sum_(e in R) eta_e=0`. The receiver's contribution to `D` is therefore
exactly `E_R-V_5`. Nonnegative contact contributions give

\[
0\le E_R-V_5\le D\le D_{\max}.
\]

The receiver retains acute winding `sigma` throughout the window, with
bounded internal form as well as phase. The environment continues its own
nodal motion. The regional work identity still determines its actual net
work; zero work for the exact reference does not imply zero work for every
nearby trajectory. This result supplies a positive-width continuation
criterion without assuming that an arbitrary environment is harmless.

### Conserved storage and backward time restrict acquisition claims

The full reference storage is `H^*=V_5+5h`. At the declared entry instant,
the exact relation to an actual state is

\[
H-H^*=D+u\sum_{e\in C}\xi_e
             +\sin\phi\sum_{e\in C}\eta_e.
\]

With the component error radii above, put

\[
L_0=|u|\sum_{\{i,j\}\in C}(r_i+r_j)
   +|\sin\phi|\sum_{\{i,j\}\in C}(p_i+p_j).
\]

The whole declared entry box has the conserved-storage enclosure
`H in [H^*-L_0,H^*+L_0+D_0]`. It stays valid along every covered
trajectory because the full law conserves `H`.

If its upper bound is below `7/2`, the
[phase-flat acquisition barrier](SINE_REGIONAL_FORMATION.md#sine-regional-channel-accessibility)
excludes every phase-flat source from reaching this entry box. In the
vanishing-width limit, a necessary compatibility condition is

\[
h\ge\frac{7/2-V_5}{5}
  =\frac{5\sqrt5-11}{20}.
\]

This floor neither proves entry nor uniquely selects a pulse. The
[larger sector proof](SINE_REGIONAL_FORMATION.md#sine-cycle-sector-barrier) also excludes any initial
zero-winding receiver from a target box whose conserved full storage is
strictly below `7/2`. Initial winding and the actual full budget remain
separate admitted observations; a nonacute unit-winding source is not
automatically covered by that particular zero-winding test.

A named acquisition source must also have a conserved storage compatible
with the declared box. For example Section 38's private-leaf ramp
`x_(5+i)=-3omega(i-2)` has exact full storage `45omega^2`.
Its large-rate acquisition cannot be joined to a small-error reference
with `h=1/50` and full storage near `V_5+1/10`: the budgets are disjoint.
Discarding that discrepancy would remove the actual environmental state
or silently introduce loss. Budget overlap is necessary, not sufficient,
for a new favorable preparation to acquire into this neighborhood.

The error estimate also holds backward for the same finite duration.
Every admitted entry-box state already had the acute identity over its
certified preceding window. Hence this certificate cannot describe first
appearance at the central instant; an initially identity-absent preparation
must precede that window if the same support and law apply throughout.
Formation followed by later stabilization remains a separate possibility.

The example `u=1/5`, `phi=0`, `Gamma=1/4`, `rho_R=rho_C=1/10`,
`T=5` and nodal form/phase error radii `1/1000` gives a nonzero-width
finite retention certificate with a moving reference. These declared
values illustrate the sufficient inequalities without optimizing a
threshold or executing a new trajectory. Setting the reference pulse to
zero gives a complementary low-storage control: maintenance can be
certified while phase-flat acquisition is excluded.

[`assess_sine_moving_pattern_window`](../../src/tnfr/physics/relational_sine_partition.py)
rebuilds the declared reference and applies these independent chart,
reference-motion, storage-growth and acquisition-budget checks. It also
encloses full storage directly from the primitive edge uncertainty boxes.
When the initial receiver chart is certified, the lower bound additionally
retains its correlated cycle minimum `V_5`, rather than independently
minimizing all five phase gaps. The primitive enclosure remains valid when
the relative-storage certificate is unavailable. The
[contract](../../docs/contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-moving-pattern-window)
and [usage](../../docs/guides/relational/SINE_REGIONAL_DYNAMICS.md#moving-pattern-window)
retain the distinction between a certified finite neighborhood and an
independently demonstrated formation-to-retention handoff.

## 41. Actual contact motion can exchange storage with internal structure

<a id="sine-collective-pulse-transfer"></a>

The contact pendulum of Section 39 conserves its own storage only on the
exact compatible family. Away from that family, the same full nodal law
supplies a correction from unequal contact phases. This identifies an
actual exchange mechanism to examine, rather than adding a controller or
assuming that a prepared compatible pattern has already formed.

### Mean-contact rows from the complete state

Keep the unchanged conservative unit C5/private-leaf support of Sections
39–40. Match receiver node `r_i` with its actual private leaf `l_i`.
No uniformity of the observed forms or phases is assumed. Define

\[
a=\frac15\sum_i x_{r_i},\qquad
b=\frac15\sum_i x_{l_i},\qquad u=a-b,
\]

and, in declared continuous phase lifts,

\[
\phi_i=\theta_{l_i}-\theta_{r_i},\qquad
\bar\phi=\frac15\sum_i\phi_i,\qquad
\eta_i=\phi_i-\bar\phi,\qquad \sum_i\eta_i=0.
\]

Internal cycle currents cancel in the receiver mean. Summing the complete
phase rows cancels its internal form Laplacian as well. In `tau=t/pi`,
the exact mean-contact laws are therefore

\[
\boxed{u'=\frac43\left\langle\sin\phi_i\right\rangle,
\qquad \bar\phi'=-\frac43u,}
\]

where brackets denote the mean over the five contacts. Equivalently, if
`Z=<exp(i eta_i)>`, then
`u'=(4/3) Im[exp(i bar_phi) Z]`. The contact deviations are live fine-state
coordinates. Replacing `Z` by one recovers the exact compatible family,
but it is not justified for an arbitrary acquired state.

Define the chart-dependent collective contact storage

\[
h_c=\tfrac12u^2+1-\cos\bar\phi.
\]

Its exact derivative is

\[
\boxed{h_c'
=\frac43u\left[
 \left\langle\sin(\bar\phi+\eta_i)\right\rangle
 -\sin\bar\phi\right].}
\]

Thus collective contact storage is not a second general invariant. Its
exchange with the remaining structure can have either sign. The formula
does not select an initial source, a direction of long-time transfer or a
new constitutive law.

### The storage split retains its phase-lift dependence

Let `d_i=x_(r_i)-x_(l_i)`, and use `F_R,V_R` for the actual internal
receiver stores. Direct edge summation gives

\[
\begin{aligned}
H&=5h_c+\mathcal R,\\
\mathcal R
&=F_R+V_R+\frac12\sum_i(d_i-u)^2\\
&\quad+\sum_i\left[
 U(\bar\phi+\eta_i)-U(\bar\phi)
                    -\sin\bar\phi\,\eta_i\right],
\qquad U(s)=1-\cos s.
\end{aligned}
\]

Consequently `R'=-5h_c'` because the full storage `H` is conserved.
The remainder is signed in general. If all segments joining `bar_phi`
to the five contact phases lie in one convex cosine-potential chart, its
contact terms are nonnegative. Even in that chart, nonnegativity does not
make the remainder an independently controlled reservoir or specify when
it feeds the receiver. Its actual transfer remains fixed by the full rows.

The contact mean uses retained **real lifts**. Independently wrapping each
contact and then averaging can change `bar_phi`, `h_c` and the signed
remainder, while leaving the actual full storage unchanged. A single
snapshot can declare a starting lift, but it does not reconstruct discarded
phase history. These collective coordinates are not new globally intrinsic
physical energies. The common-real-phase preparation below fixes their
initial lift unambiguously, and the complete smooth phase rows continue it.

### A phase-flat preparation has a discriminating fourth derivative

At a preparation with one common real phase, set
`v_i=phi_i'(0)`, `m=<v_i>=-4u/3`, and `delta_i=v_i-m`.
All form rates vanish initially. The first three derivatives of `h_c`
are zero. Differentiating the exact transfer row gives

\[
\boxed{h_c^{(4)}(0)
=\frac{16u^2}{3}\left\langle\delta_i^2\right\rangle
 -\frac{4u}{3}\left\langle\delta_i^3\right\rangle.}
\]

To check the cancellation, write
`g=<sin(phi_i)>-sin(bar_phi)`. At zero phase,
`g=g'=g''=0`, and
`g'''=-(<v_i^3>-m^3)`; mean higher phase jets cancel because
`bar_phi=<phi_i>` exactly. Expanding `<(m+delta_i)^3>` gives the displayed
variance and third-moment terms. This is a local derivative of the full
law, not a finite-time approximation substituted for it.

### One phase-flat preparation above the necessary phase-path budget

On receiver nodes `(0,...,4)` and private leaves `(5,...,9)`, supply

\[
u=\frac{119}{100},\qquad \epsilon=\frac1{20},\qquad
x_{r_i}=\frac u4+\epsilon(\mathbf1_{i=0}-\mathbf1_{i=1}),
\qquad x_{l_i}=-\frac{3u}{4},
\]

with all phases zero. The weighted form and phase means are zero. The
initial receiver has zero winding, so the proposed acute unit-winding
identity is absent. Exact edge summation gives

\[
H=\frac52u^2+4\epsilon^2
 =\frac{14201}{4000}=3.55025.
\]

This is above the necessary `7/2` phase-path barrier. It does not make
Section 40's particular small-error reference box accessible. That box
retains acute receiver winding, hence `V_R>=V_5`; its five contact form
gaps are each at least `1/5-2/1000=99/500`. Consequently every state in
that box satisfies the sharper, correlation-aware lower bound

\[
H\ge V_5+\frac52\left(\frac{99}{500}\right)^2
 =V_5+\frac{9801}{100000}
 >\frac{14201}{4000}.
\]

An independent componentwise edge enclosure can overlap the source budget
while forgetting this exact cycle constraint. Such outer-interval overlap
is not accessibility. Conservation excludes a handoff to that fixed box;
the control tests finite acute identity only. Any later maintenance
neighborhood requires its own source-compatible storage admission, without
post hoc damping or removal of environmental state.

Removing the transverse dipole gives an exactly invariant zero-winding
contact-pulse control. The control is not claimed to have the same energy
as the nonzero-dipole source. Its invariant geometry supplies the analytic
comparison without a second numerical response or amplitude search.

For the nonzero dipole, the actual centered contact velocities are

\[
(\delta_0,\ldots,\delta_4)
=\frac1{60}(-7,7,-1,0,1),\qquad
\langle\delta_i^2\rangle=\frac1{180},\qquad
\langle\delta_i^3\rangle=0.
\]

Therefore `h_c^(4)(0)=4u^2/135>0`. Time reversal at common initial phase
makes form even and phase odd, so locally

\[
h_c(\tau)=h_c(0)+\frac{u^2}{810}\tau^4+O(\tau^6).
\]

The first nonzero transfer is **toward the collective contact pulse**.
It does not give immediate transfer into the desired organized receiver.
This also does not mean every receiver field is stationary: internal
form and phase can exchange while the aggregate remainder decreases.
That local result does not decide whether later modulation produces the
target identity; the separate finite response below evaluates that question.

### Retained full-law control excludes acute winding on its declared horizon

One bounded control used exactly this preparation, unchanged law and
support. Its declared original-clock horizon was `t in [0,24]`, time step
`3/32`, Taylor order `12` and at most `256` steps. The observation asked
whether acute receiver winding `+1` or `-1` holds for an original-clock
duration of at least `1/2`, with strict positive acute margin; the
configured margin threshold is zero.

The [declaration](../../docs/assets/collective_pulse_organization/declaration.json)
and [frozen protocol](../../docs/assets/collective_pulse_organization/response-v1.protocol.json)
preceded evaluation. Outward dyadic intervals contain the declared exact
rational preparation; their centers do not replace it. The
[retained response](../../docs/assets/collective_pulse_organization/response-v1.json)
completed all `256` steps without an evaluation error, covering the full
original-clock interval `[0,24]`. Every whole-time tube certifies receiver
winding zero. The verdict is therefore
`acute_winding_excluded_on_horizon`, not incomplete or unresolved evidence.
The [source archive](../../docs/assets/collective_pulse_organization/response-v1.source.zip)
retains the evaluated implementation and preparation.

Initial regional storage is `3/400=0.0075`; its final enclosure is contained
in `[0.0061989419859282,0.0061989419859330]`. The endpoint full-storage
enclosure contains the conserved exact value `14201/4000=3.55025`.
Thus exceeding the necessary phase-path budget and breaking the unseeded
family's reflection symmetry did not produce the target identity in this
control. The local fourth derivative did not supply that finite verdict;
the whole-time proof did.

The exclusion concerns this source, complete law, support and horizon.
It excludes neither later acquisition nor other admitted preparations.
There is no response-driven retry, extended horizon, retuning or added loss.
Even a positive acute-entry response would still need source-compatible
full-state evidence for Section 40's subsequent maintenance window; this
negative response supplies no such handoff.

`observe_sine_collective_pulse` in the shared
[partition owner](../../src/tnfr/physics/relational_sine_partition.py)
rebuilds these actual-state means, feedback and optional flat-phase jet.
It retains explicit integer contact-turn offsets and the signed remainder;
it does not simulate the declared finite response. The
[contract](../../docs/contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-collective-pulse-transfer)
and [usage example](../../docs/guides/relational/SINE_REGIONAL_DYNAMICS.md#collective-pulse-balance)
describe its detached evidence and availability.

### A structural alternative separates storage release from acquisition

The same law admits a different initial transfer sign without changing its
coefficients or waiting longer. Keep all receiver forms equal to `a`, all
leaf forms equal to `b`, and all receiver phases equal. Write `u=a-b>0`
and supply contact lifts `phi_i=gamma+eta_i`, with `mean(eta_i)=0`.
For a symmetric, nonconstant deviation multiset with
`abs(eta_i)<min(gamma,pi/2-gamma)` and `0<gamma<pi/2`,
`S=mean(sin eta_i)=0` and `C=mean(cos eta_i)<1`. Then

\[
h_c'(0)=\frac43u(C-1)\sin\gamma<0.
\]

This is an exact release from collective contact storage into its signed
remainder. It is not merely a different scalar bookkeeping observation:
the full law also forces an internal receiver response. Put
`s_i=sin(phi_i)` and let indices follow the C5 cycle. Initially,

\[
x_{r_i}'=\frac{s_i}{3},\qquad
\theta_{r_i}'=\frac u3,\qquad
\theta_{r_i}''=\frac{6s_i-s_{i-1}-s_{i+1}}9.
\]

The receiver starts with zero internal form and phase storage. Direct
differentiation of its edge sums gives

\[
F_R''(0)=\frac19\sum_i(s_{i+1}-s_i)^2,\qquad
V_R^{(4)}(0)=3\sum_i
 (\theta_{r_{i+1}}''-\theta_{r_i}'')^2.
\]

For phase storage all lower derivatives vanish: the initial relative
phases and their first rates are zero. Both displayed quantities are
strictly positive when `s` is nonconstant, since `6I-A_C5` has strictly
positive eigenvalues and preserves the constant/nonconstant decomposition.
Thus internal form grows at order `tau^2` and internal phase storage at
order `tau^4` for a sufficiently short nonzero interval. No acute winding,
directed full-cycle passage or finite-width handoff follows from that local
onset.

In particular, choose `0<rho<min(gamma,pi/2-gamma)` and the cyclic
deviations `(rho,-rho,-rho,rho,0)`. Reflection `i -> 3-i (mod 5)`,
including the matched private leaves, preserves this source. The complete
law preserves that symmetry by the
[same equivariance and uniqueness argument](SINE_REGIONAL_FORMATION.md#sine-conservative-source-geometry),
so the receiver cannot acquire nonzero winding
whenever no receiver edge is antipodal; in particular, acute unit winding
is excluded. Nevertheless this source has `h_c'(0)<0`
and both positive internal derivatives above. Permuting the same multiset
to `(rho,rho,-rho,-rho,0)` removes that particular reflection obstruction
while leaving the initial contact resultant, transfer rate and total
storage unchanged:

\[
H=\frac52u^2+5-(1+4\cos\rho)\cos\gamma.
\]

Removing a symmetry obstruction is not a formation theorem. This exact
comparison explains the next admission question: which ordered contact
geometry transfers into a directed receiver phase configuration while
retaining a source-compatible full-state maintenance budget? A negative
collective transfer rate or a growing internal store cannot answer it
alone. These conditional preparations are not an additional scheduled
response, parameter search or modification of the frozen control.

## 45. Complete contact feedback distinguishes cancellation from rigidity

<a id="sine-receiver-rigidity"></a>
<a id="sine-relative-phase-feedback"></a>

The relative-rate identity at the end of Section 44 gives an instantaneous
condition. Its persistence depends on both the receiver's internal current
and the actual contact response. The same conservative unit C5/private-leaf
law supplies the needed derivatives and admits a precise classification of
**exactly rigid** receiver geometry. A finite retained identity need not be
rigid, so this classification does not replace the broader acquisition and
retention question.

### The complete centered rows and their derivatives

Use `L=L_R`, the five-node cycle Laplacian, and let `P` remove the receiver
mean. In matched receiver/leaf order put

\[
\vartheta=P\theta_R,\qquad q=Px_R,\qquad r=x_R-x_Q,
\qquad u=\langle r\rangle,\qquad
\phi=\theta_Q-\theta_R,
\]

in continuous phase lifts. Let `s=S_C5(vartheta)` be the incoming internal sine
current and `f=sin(phi)` componentwise. The full normalized rows give

\[
q'=\frac13(s+Pf),\qquad
r'=\frac13s+\frac43f,\qquad u'=\frac43\langle f\rangle,
\]

\[
\boxed{\vartheta'=\frac13(Lq+Pr),\qquad
\phi'=-\frac13(Lq+4r).}
\]

All derivatives in this section use `tau=t/pi`. Differentiating the first
phase row and retaining the actual contact current gives

\[
\boxed{\vartheta''=\frac19\bigl[(L+I_5)s+(L+4I_5)Pf\bigr].}
\]

Define `L_cos(vartheta)` as the cycle Laplacian whose edge weights are
`cos(vartheta_j-vartheta_i)`. Since `s'=-L_cos(vartheta)vartheta'`, one more derivative is

\[
\boxed{\vartheta'''=\frac19\left[-(L+I_5)L_{\cos}(\vartheta)\vartheta'
 +(L+4I_5)P\bigl(\cos\phi\mathbin{\odot}\phi'\bigr)\right],}
\]

where `odot` denotes componentwise multiplication. None of these rows
assumes that a small regional storage, a zero measured rate or a
collective mean determines the environmental coordinates.

### Vanishing first and second rates do not imply a fixed geometry

Prepare receiver phases at a uniform unit twist, copy each phase to its
private leaf, and choose any nonzero centered form vector `q`. Supply
`r=u*1-Lq`; receiver and leaf forms can then be reconstructed with any
declared common form origin. Initially `s=0`, `phi=0`, and

\[
\vartheta'=0,\qquad \vartheta''=0,\qquad
\boxed{\vartheta'''=\frac19(L+4I_5)Lq\ne0.}
\]

Both factors on the centered subspace are invertible. In particular, for
`q=kappa*(1,-1,0,0,0)`, the exact third derivative is

\[
\vartheta'''=\kappa(22/9,-22/9,1,0,-1).
\]

The contact phase differences have rates `Lq-4u*1/3` at that instant
(the absolute leaf phase rates are `Lq-u*1`). Their subsequent
sine currents break the initial compensation and produce the cubic
receiver response. This is an initial derivative of the complete field,
not a finite-time trajectory estimate. It excludes the inference that
even simultaneous zero first and second relative rates prove rigidity.

### Exact rigidity has no additional moving compensation family

Suppose the receiver has a fixed relative phase geometry on a nonempty
open time interval, so `vartheta'=0` throughout that interval. The full finite
sine vector field is real analytic, its solutions exist for all real
time, and their storage is finite and conserved. The global bound
`|x_i'|<=1` and the linear phase row permit at most linear form growth
and quadratic lifted-phase growth on finite intervals, precluding finite-time
blowup in either direction. Consequently the
analytic identity `vartheta'=0` extends to the entire solution. In particular
`vartheta`, and hence `s`, are constant.

The first two vanishing phase rows imply

\[
Pr=-Lq,\qquad
Pf=-(L+4I_5)^{-1}(L+I_5)s,
\qquad q'=(L+4I_5)^{-1}s.
\]

Thus `q'` is constant. But the receiver form storage
`q^T Lq/2` is bounded above by the conserved full storage `H`; since
`L` is positive definite on centered vectors, `q` is bounded for all
real time. A nonzero constant `q'` is impossible. Therefore

\[
s=0,\qquad Pf=0,\qquad q=\text{constant},\qquad
\phi_i'=a_i-4u/3,\quad a=Lq,\quad\sum_i a_i=0.
\]

Every contact sine equals one common analytic function `m(tau)`.
For any pair of contacts the analytic identity

\[
0=\sin\phi_i-\sin\phi_j
 =2\sin\frac{\phi_i-\phi_j}{2}
    \cos\frac{\phi_i+\phi_j}{2}
\]

forces at least one factor to vanish identically: if one analytic factor
is nonzero somewhere, the other vanishes on an open interval and hence
everywhere. The corresponding continuous lifts either differ by a fixed
multiple of `2pi`, or sum to a fixed odd multiple of `pi`.

In the supplementary case, summing their phase rates gives
`a_i+a_j-8u/3=0`, forcing `u` to be constant. Since `u'=4m/3`,
this forces `m=0` identically. If `m` is not identically zero, all
contacts must instead agree modulo `2pi`. Their equal derivatives imply
that all `a_i` agree, hence `a=0` and `q=0`. Both form blocks are
uniform and all contact phases agree: this is precisely the moving
compatible family of Section 39, with its actual contact pendulum.

If `m=0` identically, each contact phase is a fixed integer multiple
of `pi`. Its zero derivative gives `a_i=4u/3` at every node, hence
`u=0`, `a=0` and `q=0`. All forms are equal throughout the full graph,
and every phase is stationary. Individual contacts can be aligned or
antialigned; these mixed-contact states are equilibria, not another moving
compensation mechanism.

Finally, at an acute cycle critical point `s=0` forces all oriented edge
sines to agree. Sine is injective on the acute interval, so unit winding
forces the uniform gaps `+2pi/5` or `-2pi/5`. Thus an exactly rigid
acute unit-winding receiver belongs either to the known uniform-block
moving family or to a static equilibrium with zero contact torques.
This classifies individual rigid trajectories, rather than only families
required to close for every supplied collective state.

### A retained second-order description exposes the same feedback

The contact row also gives the exact equation

\[
\boxed{\phi''=-\frac19
 [(L+4I_5)s+(L+16I_5)\sin\phi].}
\]

Together with the displayed equation for `vartheta''`, this is a closed
second-order description of four centered receiver phase coordinates and
five contact phases. It retains nine initial velocities, rather than
claiming a first-order closure from instantaneous phases alone. If
`v=vartheta'` and `c=phi'`, the original relative forms reconstruct exactly by

\[
Lq=Pc+4v,\qquad Pr=-Pc-v,\qquad
u=-\frac34\langle c\rangle.
\]

The inverse of `L` is restricted to the centered subspace. The two
conserved degree-weighted form and lifted-phase means recover the common
origins: if they are `m_x,m_theta`, the receiver form mean is
`m_x+u/4` and its phase mean is `m_theta-<phi>/4`. The leaf state then
follows from `x_Q=x_R-r` and `theta_Q=theta_R+phi`.

These eighteen relative coordinates plus the two origins carry the same
twenty fine form/phase coordinates. This is a support-specific instance of
the [exact form/phase representation](SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence),
not an information loss, a new microscopic law or an additional research
branch. It makes explicit where the environmental response re-enters a
putatively compensated receiver.

### Consequence for the longer retention question

Exact entry from outside either rigid invariant family at a finite time
remains excluded by uniqueness. That fact does not rule out acute identity
with bounded internal evolution. In the compensated example with `u=0`,
the full storage is `V_5+13kappa^2`; sufficiently small nonzero `kappa`
gives `H<B`, where `B=5-4cos(3pi/8)` is the **acute-face** barrier.
The regional barrier then protects its already prepared
acute identity even though the nonzero third derivative proves it is not
rigid. This is maintenance of an initially present identity, not formation
from zero winding.

The prospective formation-to-retention requirement of a scaled duration
at least one is not closed by these derivative or rigidity results.
Its permissible targets can evolve internally. They still require actual
full-state entry, bounded accumulated relative transport and compatible
regional work over that duration. Instantaneous compensation is a useful
quantity to inspect, but it is not a substitute for those interval bounds.

In particular, each target edge `i->j` needs a continuous lift satisfying

\[
\left|\delta_{ij}(0)+\int_0^s[v_j(\tau)-v_i(\tau)]\,d\tau\right|
 <\frac\pi2\qquad\text{for every }0\le s\le1,
\]

with the declared winding sector and full-state uncertainty retained.
Cancellation only at the endpoint cannot establish this prefix condition.
The generic gap-acceleration bound of four gives a phase-error radius of
two at this horizon, already greater than `pi/2`; that estimate alone
cannot certify the required acute interval, even from zero initial relative
rate. The actual restoring and contact feedback, or existing validated
full-state enclosures, must supply tighter state-dependent control. These
are consequences of the selected complete sine law, not a proof that it
is the uniquely emerging law of TNFR.
