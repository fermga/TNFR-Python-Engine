# Directional maintenance after a complete form-capacity word

The [form-capacity analysis](SINE_FORM_CAPACITY_ACTION.md) showed why a scalar
storage sublevel cannot decide retention of the specified local resets. This
owner supplies an exact reduction of the retained source and a conditional
nonlinear enclosure method. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone selects further investigation and its dependencies. No change of
nodal law, radius, source origin or operator gains is part of this comparison.

## 1. Actual state and the complete word

Use the `OZ_completed` primitive state and associated execution context from
the [retained IL/OZ response](../../docs/assets/collective_pressure_handoff/correction_v2/response.json).
Keep the twenty nodes, all 45 bare edges, actual conductances, signed form,
positive capacities and primitive phase. The starting structural time is 7.
The source has nine exact twin equalities, but nodes 0 and 1 have different
forms and need not have the same boundary weights. The generic full-source
neighborhood does not share the nominal source's exact symmetry.

The selected bounded extension is `contraction,coherence,silence` on ordered
targets `(0,1)`. NUL uses its default factor; IL and SHA retain the source's
configuration, including the supplied `SHA_vf_factor=1/2`. This is not a pair
of reciprocal scale maps. NUL's stored-pressure rescaling, IL's phase reset,
SHA's capacity change, live histories and every pressure refresh belong to
the full preparation. A result for the isolated primary NUL map cannot be
substituted for this word's endpoint.

Read-only sequence admission retains the existing OZ debt of one on each
target. The proposed debt progression is `1 -> 1 -> 0 -> 0`: NUL supplies no
stabilization, IL pays the retained debt, and SHA closes the word. Shadow
grammar checks do not execute the word or admit its later physical states;
each actual stage must still pass live preconditions without replacement.

Canonical U6 is separately observed from before/after structural potential
snapshots, using the shared full-graph distance convention and the declared
drift threshold. It is not a token-order rule or a future confinement theorem.
Record its baseline, raw/refreshed stored-pressure convention and finite
coverage; a passing sequence does not imply a passing U6 observation.

The [retained-state reader](../../src/tnfr/research/retained_graph_state.py)
reconstructs explicitly recorded persistent inputs, including bounded history
deques, attribute associations and neighbor order. It neither authenticates a
previous execution nor invents omitted cache state. Recorded rebuildable
caches remain absent until their owners rebuild them. This is a new declared
preparation from retained inputs, not an unrestricted checkpoint facility.

Between events use the same complete conservative law and original clock:

\[
 \dot x_i=\frac{\nu_i}{\pi d_i}
    \sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
 \dot\theta_i=\frac{\nu_i}{\pi d_i}
    \sum_{j\sim i}W_{ij}(x_i-x_j).
 \tag{1}
\]

Support, conductances and post-word capacities are held, and forcing and
continuous loss are absent. The supplied operator word does not establish
an autonomous event-selection law. Future phases are continuous lifts, not
modulo-wrapped jumps between integration steps.

## 2. An exact eleven-cell reduction

Let sigma fix nodes 0 and 1 and exchange each pair `(2,3),...,(18,19)`.
Admit the complete bare support, each conductance, held capacities and initial
form/phase lifts under this involution. Exact equivariance and smooth uniqueness
then preserve its fixed-point family under (1). Event preservation is separate:
re-admit membership and symmetry after the complete word.

There are eleven cells: two singletons and nine pairs. Let m_a be their
multiplicities, n_ab the number of neighbors in b of one member of a, and
w_ab its sum of conductances to b. These are per-member quantities. Internal
neighbors still contribute to the original degree d_a, even though their
form and phase differences vanish on the fixed family. Zero-conductance
edges still contribute to n_ab and d_a. Reciprocity gives

\[
 m_a n_{ab}=m_b n_{ba},\qquad m_a w_{ab}=m_b w_{ba}.
 \tag{2}
\]

The reduced field is exactly

\[
 \dot X_a=\frac{\nu_a}{\pi d_a}
    \sum_b n_{ab}\sin(\Theta_b-\Theta_a),\qquad
 \dot\Theta_a=\frac{\nu_a}{\pi d_a}
    \sum_b w_{ab}(X_a-X_b).
 \tag{3}
\]

This is an exact 22-coordinate representation of the specified invariant
family. It is not an eleven-node simple graph with recomputed degrees, nor
an identification of eleven independent collective NFRs. Reconstructing
fine observables requires multiplicities and the retained member association.
In particular the unchanged identity is

\[
 Z^2=\sum_a m_a(X_a-\bar X)^2+
     \sum_a m_a(\Theta_a-\Theta_{*,a}-\bar\eta)^2,
 \quad \bar X=\frac1{20}\sum_a m_aX_a,
 \quad \bar\eta=\frac1{20}\sum_a m_a(\Theta_a-\Theta_{*,a}).
 \tag{4}
\]

The radius remains r=1/16. An independent uncertainty box around the original
fine source need not lie in this family; no robust full-family result follows
merely from reducing its nominal member.

The [weighted symmetry owner](../../src/tnfr/physics/relational_sine_symmetry.py)
re-admits primitive source and law before constructing `SineSupportInvolution`.
It distinguishes invariant family, source membership and trajectory reduction.
Cached verdicts cannot replace these premises. Its interval/Taylor field
reuses the shared sine rate evaluator, retaining the actual normalization.

## 3. A metric that retains directional error

Let K be the symmetric reduced form Hessian: off-diagonal entries are
`K_ab=-m_a w_ab`, with row sums zero. Let C_0 be the reduced phase comparison
Laplacian whose fine old edges have coefficient 309/1000 and whose five new
edges have coefficient one. The true reference coefficient on old edges is
`cos(2*pi/5)`, strictly between 309/1000 and 310/1000. C_0 is a rational
comparison metric, not a replacement of the sine field in (3).

Define

\[
 \rho_a=\frac{m_a d_a}{\nu_a},\quad R=\sum_a\rho_a,
 \quad B=\operatorname{diag}\frac1{\pi\rho_a},\qquad
 Q=\operatorname{diag}\left(K+\frac{\rho\rho^T}{R},
                            C_0+\frac{\rho\rho^T}{R}\right).
 \tag{5}
\]

Connected positive support makes each Laplacian positive on relative
coordinates; the added rank-one term removes the common null direction.
Thus Q is positive definite, including common origins. The two rho-weighted
means are constant along the continuing law, and `rho^T B=1^T/pi`, so the
added terms contribute zero to the field's metric-Jacobian balance. They
make the error metric invertible without changing the actual dynamics.

For the true phase Hessian C(Theta), the Jacobian is

\[
 J=\begin{pmatrix}0&-B C(\Theta)\\BK&0\end{pmatrix},\qquad
 QJ+J^TQ=
 \begin{pmatrix}0&KB(C_0-C(\Theta))\\
 (C_0-C(\Theta))BK&0\end{pmatrix}.
 \tag{6}
\]

The large conservative tangent terms cancel. Only the change of the phase
Hessian relative to C_0 enters the error-growth estimate. This is why a
directional enclosure can remain useful after the scalar energy-sublevel
certificate has failed.

On the larger validation domain `Z<1/8`, every fine edge perturbation from
its reference gap has magnitude below `sqrt(2)/8<5/28`. The cosine Lipschitz
bound and `0<cos(2*pi/5)-309/1000<1/1000` give the relative old-edge bound

\[
 \frac{5/28+1/1000}{309/1000}<\frac35.
\]

For new zero-reference-gap edges, `1-cos(delta)<=delta^2/2<1/64`.
Consequently

\[
 -\kappa C_0\preceq C(\Theta)-C_0\preceq\kappa C_0,
 \qquad \kappa=3/5.
 \tag{7}
\]

Actual conductances and capacities must not exceed one for the following
comparison. The normalized-Laplacian upper bounds imply
`||K^(1/2) B C_0^(1/2)||<=2/pi`. Combining (6)--(7) and the quadratic
inequality `2ab<=a^2+b^2` yields the logarithmic norm bound

\[
 \mu_Q(J)\leq\kappa/\pi<1/5.
 \tag{8}
\]

This bound applies to the true nonlinear Jacobian throughout the admitted
convex domain, not just the stationary tangent. It does not assert that Q is
conserved or that the complete flow contracts. It permits the existing
[metric Taylor validator](../../src/tnfr/mathematics/_validated_metric.py)
to carry one full uncertainty ball with growth rate 1/5 rather than replacing
the carried uncertainty by independent coordinate boxes at each step.

## 4. Bounded nonlinear evaluation and its possible conclusions

The numerical comparison uses the exact reduced nonlinear field (3), a
Picard tube for each whole interval, order-12 interval Taylor bounds and the
shared outward 128-bit arithmetic. The fixed mesh has duration
1/4 and 160 steps, covering elapsed structural time `[0,40]` after the word
at time 7. This is `[7,47]` in the retained absolute clock; operator counts
are not elapsed seconds. Source, laws, observation, horizon, precision and
failure policy must be frozen before the response is evaluated.

The horizon has an independent scale justification. An untouched internal
twin-antisymmetric tangent direction has form and phase Hessian eigenvalues
4 and `4*cos(2*pi/5)`, and mobility `1/(4*pi)`. Its angular frequency is
`sqrt(cos(2*pi/5))/pi`, with period `2*pi^2/sqrt(cos(2*pi/5))<40`.
This calibrates a nontrivial structural horizon without looking at a response;
it does not make the actual nonlinear trajectory periodic or transfer that
mode's dynamics to the excited directions.

The validation radius 1/8 and observation radius 1/16 have separate roles.
The former supplies a domain for (8); it does not relax the identity target.
Whole-step bounds `Z^2<1/256` prove finite maintenance on the
covered intervals. A certified endpoint lower bound `Z^2>1/256` proves that
this particular identity has been left by that time. A straddling interval,
failed Picard tube or exhausted fixed budget remains unresolved. None may
be turned into a dynamical failure or a passing flag. Finite maintenance is
not all-time stability, formation or autonomous invocation of the word.

The [existing scalar-sublevel obstruction](SINE_FORM_CAPACITY_ACTION.md#same-storage-sublevel-obstruction)
remains valid. A successful directional enclosure would answer a different
question using more information: the complete initial direction, actual
nonlinear evolution and certified accumulated error, rather than all states
sharing a scalar storage bound and two means.

<a id="directional-maintenance-result"></a>
## 5. Frozen complete-word result: finite departure

The [protocol](../../docs/assets/collective_form_capacity/maintenance_v1/protocol.json),
[source freeze](../../docs/assets/collective_form_capacity/maintenance_v1/freeze.json)
and archived implementation preceded the single evaluation. Its
[response](../../docs/assets/collective_form_capacity/maintenance_v1/response.json)
and [verdict](../../docs/assets/collective_form_capacity/maintenance_v1/report.json)
retain all three executed stages and 57 certified continuation steps. The
prospective stop rule ended the calculation at elapsed time 57/4, before the
maximum forty-unit horizon, because the endpoint was provably outside the
unchanged identity radius. No numerical retry, alternate word, radius change
or source/gain adjustment was performed.

**Grammar and events.** The actual `ValidatedSequence` and all six live
target-stage checks passed. Each atomic stage executed its requested glyph
with strict preconditions and no replacement. Both targets retained the
debt progression `1 -> 1 -> 0 -> 0`; IL compensated the inherited OZ debt.
All eight canonical U6 comparisons passed on their declared snapshots.
The complete-word mean absolute potential drift was approximately
`1.137030752337261e-5`, below the represented pi/2 threshold. These are
finite stored-state observations, not U6 bounds on the continuing trajectory.

**Future identity.** Whole-interval bounds certify `Z<1/16` through elapsed
time 113/8. At time 57/4 the outward endpoint bounds imply

\[
 0.003923728663717655 < Z^2 < 0.003923728663717657,
 \qquad Z^2>1/256=0.00390625.
\]

Equivalently `0.06263967324082760 < Z < 0.06263967324082762`, whereas the
frozen radius was 0.0625. Continuity therefore places the first boundary
hit in elapsed `(14.125,14.25)`, or absolute structural time
`(21.125,21.25)`. The final metric error radius is below `1.54e-22`; the
departure margin is not an unresolved enclosure overlap.

This rejects the specified finite-retention hypothesis for this complete
word and nominal preparation. It does not establish destruction of the
organization, loss of constituent winding, eventual escape from every
neighborhood or failure of the TNFR operator contracts. A valid grammar word
and passing finite U6 observations do not imply preservation of this finer
geometric identity under the chosen subsequent law. Conversely, an isolated
NUL reset or a different complete word cannot inherit this trajectory verdict.

The [read-only evidence tests](../../tests/research/test_collective_form_capacity_maintenance_evidence.py)
verify the frozen source associations, reconstruct primitive event states,
recheck live grammar/debt and U6, rebuild the exact reduction/metric and
recompute endpoint identity bounds from retained centers and error radii.
They execute neither the reserved word nor the continuing trajectory.

## 6. Retained organization through the certified departure

The same evidence supports a separate interpretation without a new forecast
or a changed identity criterion. Every certified Picard tube remains in
`Z<1/8` through elapsed 57/4. Therefore each fine phase-edge perturbation
from the original reference has magnitude below `sqrt(2)/8`. Old reference
gaps have principal magnitude `2*pi/5`, and new reference gaps are zero;
`2*pi/5+sqrt(2)/8<pi/2`. All 45 edges consequently stay acute throughout
that interval. No original cycle edge crosses a principal cut, so each
constituent's originally oriented C5 winding remains +1. Held support,
conductances and positive capacities are premises of this continuation;
the exact eleven-cell symmetry is preserved by its law.

Let A and B denote the original ten-node constituents, and let
`eta=theta-theta_star` use the same fixed lifts as (4). Define Z_A and Z_B
by centering form and eta separately within each constituent. The exact
variance decomposition is

\[
 Z^2=Z_A^2+Z_B^2+
 5(\bar x_A-\bar x_B)^2+5(\bar\eta_A-\bar\eta_B)^2.
\]

Reconstructing the final metric ball gives the following approximate
contributions; the read-only evidence controls retain outward bounds.

| Contribution at elapsed 57/4 | Squared distance |
| --- | ---: |
| Internal constituent A | 0.002749906304 |
| Internal constituent B | 0.001020360841 |
| Relative form means | 0.000126069321 |
| Relative phase-error means | 0.000027392199 |

Each internal contribution is below 1/256 at this endpoint, while their sum
with relative registration exceeds that threshold. The failed joint-tube
claim thus coexists with retained acute geometry and winding. Neither a
winding label nor these endpoint component bounds define a closed collective
state or prove indefinite constituent maintenance. The separate
[identity-transition contract](SINE_COLLECTIVE_IDENTITY_TRANSITION.md)
applies the existing operator and grammar descent criteria to this marked
contact, with explicit source and successor domains. Its ideal event-sector
and finite U6 results preserve this negative original observation and do not
extend the certified future interval.
