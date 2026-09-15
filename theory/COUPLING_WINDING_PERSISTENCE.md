# Canonical Coupling preserves winding in a restricted cycle regime

**Status:** Exact target-only UM gap transport and persistence theorem;
finite production checks and a canonical Transition loss witness.
**Research links:** B2.d/O3.a, S3, S8, S9 and S16.
**Scope:** Fixed simple undirected cycles with explicit phase coordinates.

## 1. The structural state and the actual operator realization

Let `phi_i` be the phase at node `i` of an oriented cycle `C_n`, `n>=3`, and
let its oriented shortest-arc gaps be

$$
d_i=\operatorname{wrap}(\phi_{i+1}-\phi_i),\qquad
-\pi<d_i<\pi,\qquad
\sum_i d_i=2\pi W.
$$

Indices are cyclic and `W` is the integer winding of this declared cycle.
The final identity follows from a consistent closed phase loop, not from
arbitrarily supplied real gap coordinates. Reversing orientation changes its
sign. Retaining a positive wrap-branch margin makes the observation defined;
it does not by itself prove that a future operation preserves that margin.

The following theorem uses the existing canonical Coupling implementation
in [_coupling_stage_kernel.py](../src/tnfr/operators/_coupling_stage_kernel.py),
with `UM_BIDIRECTIONAL=False` and `UM_FUNCTIONAL_LINKS=False`. The first
configuration makes each target average only its neighbors and write only
its own phase. The second keeps the simple cycle support fixed. Other
preconditions and grammar admission remain necessary at execution time.

Let `g` be the effective UM gate, including any tightening of the hard U3
limit, and suppose every gap lies in one interval

$$
-g<m\leq d_i\leq M<g,\qquad 0<g\leq\pi/2.
$$

The interval may be positive, negative or contain zero. Both cycle neighbors
are therefore compatible with every target. Their separation in the local
lift is `d_(i-1)+d_i`, whose absolute value is strictly less than `pi`.
The neighbor phasor sum is nonzero, so the circular midpoint is unambiguous.
The UM phase factor `eta=UM_theta_push` is the already configured operator
coefficient, with `0<eta<=1`. Its repository default is `1/(pi+1)` as
materialized by `UM_THETA_PUSH`. The theorem covers the declared factor
interval; it does not derive that particular default uniquely from the nodal
equation or add an independent phase-evolution law.

The default bidirectional UM configuration has a different effect: it includes
the target in the phasor mean and writes neighboring phases too. It is outside
this theorem. Even on a uniform twisted cycle, one bidirectional target shrinks
the two inner gaps by `1-eta` and expands the adjacent outer gaps by `1+eta`.
The interval invariance proved below cannot simply be transferred to it.

## 2. The midpoint identity and its canonical pressure connection

In the target's local lift, its neighbor phases are
`phi_i-d_(i-1)` and `phi_i+d_i`. Their phasor sum is

$$
2\cos\frac{d_{i-1}+d_i}{2}\;
\exp\left(i\left[\phi_i+\frac{d_i-d_{i-1}}2\right]\right).
$$

The cosine is positive under the stated strict gate. Consequently the actual
target-only UM shortest-arc update has displacement

$$
u_i=\frac\eta2(d_i-d_{i-1}).
$$

This identity connects directly to the existing canonical phase-pressure
channel in [dnfr.py](../src/tnfr/dynamics/dnfr.py). On this same simple cycle,
its unweighted neighbor-phasor mean is the midpoint above, hence

$$
\partial\phi_i
=-\frac1\pi\operatorname{wrap}(\phi_i-\bar\phi_i)
=\frac{d_i-d_{i-1}}{2\pi},\qquad
u_i=\eta\pi\,\partial\phi_i.
$$

The configured phase-channel weight belongs to the aggregation of
`DeltaNFR`; it is separate from the unweighted channel identity here. The
nodal integrator subsequently uses `nu_f*DeltaNFR` to evolve EPI. Reading the
same structural gradient in UM does not identify a phase jump with that EPI
evolution or introduce a second continuous-time law. The identity excludes
vanishing resultants, arbitrary neighborhood profiles and wrap-crossing
configurations. Binary64 phasor evaluation retains its separate residual.

## 3. One target: conserved circulation and interval protection

Updating target `i` changes just its incident gaps. Put
`a=d_(i-1)`, `b=d_i`. Then

$$
\begin{pmatrix}a'\\b'\end{pmatrix}
=\begin{pmatrix}1-\eta/2&\eta/2\\\eta/2&1-\eta/2\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}.
$$

Each new gap is a convex combination of the old pair, and their sum is
unchanged. Every gap remains in `[m,M]`, strictly inside U3 and away from the
wrap branch. The same is true along the affine shortest-arc completion of
the declared phase jump. Thus the closed cycle retains `W`. This completion
is a mathematical path witness; the engine event itself is a discrete update.

For the gap mean `d_bar=sum(d)/n`, define the diagnostic squared spread
`V(d)=sum((d_i-d_bar)^2)/2`. Direct expansion gives

$$
V(d)-V(d')=\frac{\eta(2-\eta)}4(a-b)^2\geq0.
$$

Every finite sequence of these admitted target-only updates preserves the
same interval and winding, even if its target order varies. Arbitrary target
selection does not imply convergence to uniform gaps: a sequence can keep
updating an already equal pair while leaving other differences untouched.

## 4. A simultaneous stage is the canonical cycle diffusion map

An immutable all-target UM stage proposes the displacement in section 2 for
every node. Since target-only UM has no overlapping neighbor writes, the
new edge gaps are

$$
d_i'=d_i+u_{i+1}-u_i
=(1-\eta)d_i+\frac\eta2(d_{i-1}+d_{i+1}).
$$

Equivalently,

$$
d'=(I-\eta L_{\rm rw,C_n})d.
$$

This is precisely the normalized cycle Laplacian already used by the canonical
pure-EPI diffusion channel. Here it acts on an oriented gap observation under
a UM stage; the index counts operator stages, not elapsed physical time.
Its nonnegative, doubly stochastic matrix preserves the gap sum, the initial
gap interval, U3 compatibility and winding in the same exact model.

The exact nonnegative spread drop is

$$
V(d)-V(d')=
\frac{\eta(1-\eta)}2\sum_i(d_{i+1}-d_i)^2
+\frac{\eta^2}{8}\sum_i(d_{i+1}-d_{i-1})^2.
$$

The Fourier eigenvalues are
`1-eta+eta*cos(2*pi*k/n)`. For fixed `0<eta<1`, their absolute values are
strictly below one for all `k!=0`, so repeated exact all-target stages converge
to `d_bar=2*pi*W/n`. At `eta=1`, an even cycle has an alternating gap mode with
eigenvalue `-1`; its spread is preserved, although winding and interval
protection still hold. An odd cycle has no such nonconstant mode. These are
exact-model statements, not asymptotic binary64 execution certificates.

For same-sign gaps and nonzero total `S=sum(d)=2*pi*W`, circulation concentration
can be read without another dynamical parameter:

$$
I(d)=\sum_i\left(\frac{d_i}{S}\right)^2
=\frac1n+\frac{2V(d)}{S^2}.
$$

It cannot increase under the protected maps. Under the strictly contracting
all-target regime it tends to `1/n`. The preserved winding therefore becomes
an evenly distributed twist; this class supplies no mechanism for sustained
localization of circulation. It neither constructs a localized physical
entity nor supplies a restoring law for localized EPI or capacity support.

## 5. Nonzero winding can coexist with zero canonical pressure

A uniform twist `phi_i=2*pi*W*i/n` with `|2*pi*W/n|<g` is fixed by target-only
UM. Its phase-pressure channel vanishes because its two neighbor phasors
have their mean at the target phase. With uniform EPI, uniform positive
capacity and the uniform degree of the simple cycle, the other three
canonical gradient channels also vanish. Therefore the complete canonical
`DeltaNFR` is zero in this exact restricted state, despite nonzero `W`.

Pressure equilibrium thus need not mean phase consensus. A positive-winding
example with the strict canonical gate requires `n>4*W`. The spectral
convergence above concerns uniformity of gaps, not uniformity of phases.
It supplies neither attraction for arbitrary multichannel trajectories nor
a dynamic preparation of nonzero winding from a zero-winding state. The
protected UM class cannot perform that preparation because it conserves `W`.

## 6. Canonical loss, branch crossings and observation limits

Outside the protected class, canonical events can change winding. For the
regular unit-winding eight-cycle, change only the phase at node zero from its
initial value zero to `theta`. The edge from node seven to node zero reaches
the wrap branch when `theta=3*pi/4`. Immediately below this value the winding
is one; immediately above it, before the next incident branch crossing, it
is zero. The cycle support can remain unchanged throughout.

The production Transition (`NAV`) word supplies a finite instance. On the
prepared regular eight-cycle with repository default regime policy and seed
17, repeated admitted single-Transition words give `theta_0=2.35` after 13
steps and `theta_0=2.5500000000000003` after 14. Their winding observations are
respectively one and zero. The reported minimum branch margins are
approximately `0.006194490192344748` and `0.19380550980765499`. The actual
EPI, capacity and pressure effects of Transition remain part of those events;
the example does not impose an auxiliary phase trajectory. Its endpoint
change requires a branch crossing in any continuous fixed-cycle completion,
but does not certify an unobserved continuous path inside a discrete event.

U3 admission applies to the operator's required compatible relations; it is
not a universal promise that every cycle edge stays admissible under every
canonical operator. Equal endpoint winding likewise cannot establish a
protected intervening path: opposite edge slips can cancel. Neither a
phase-independent EPI operation nor a static snapshot is a proof about all
future phase writes. Cycle deletion, changed support, absent phase values,
branch hits and nonzero signed branch slips must remain distinguishable.

## 7. Exact companion, runtime checks and remaining work

[`coupling_winding.py`](../src/tnfr/physics/coupling_winding.py) implements the
single-target and all-target exact gap maps with conserved sum, interval and
spread identities. It reuses the shared exact-or-represented real reader:
integer and rational inputs remain exact, including NumPy integers promoted
to Python integers; other real values retain their binary64 rational value.
The observer accepts signed gap coordinates and does not round their sum
into a purported winding integer. A separate declared-cycle phase observation
must establish closure and winding. Its factor, state and gate hypotheses
also do not admit a live operator word by themselves.

[Exact tests](../tests/physics/test_coupling_winding.py) cover convex transport,
Jensen dissipation, finite compositions, the even-cycle boundary and the
shared pure-EPI pressure map.
[Pressure tests](../tests/physics/test_coupling_pressure_bridge.py) compare the
actual scalar and vectorized canonical phase channel with the midpoint
identity, then use the shared nodal integrator for EPI evolution. Their
regular twisted controls bound observed binary64 pressure residuals; a
selected finite tolerance is not a rigorous bound on every libm evaluation.

[Runtime tests](../tests/physics/test_coupling_winding_runtime.py) and the
[benchmark](../benchmarks/canonical_winding_persistence.py) compare actual
canonical UM histories with the exact gap companion and record endpoint
residuals, branch/U3 margins, winding and actual operator histories. They
use eight UM/SHA pairs on perturbed C8 and C16 with W=0 and W=1. SHA leaves
phase unchanged and attenuates capacity by its existing default factor;
the word uses no phase-factor override. Both sequence validators and live
operator requirements remain active. Separate DERIVED and MEASURED manifests
record the working-source digest and the corresponding evidence boundary.
The word observer now uses live primary-EPI admission, shared adjacency
validation, runtime U3 limits and node/edge identity comparisons; its
[integrity tests](../tests/physics/test_winding_word_integrity.py) cover these
boundaries without changing committed-prefix behavior after later failure.
The production controls
also exercise canonical winding loss through Transition. These finite
checks are separate from the exact repeated-map theorem and do not prove
future binary64 protection, arbitrary grammar stability or a physical
memory experiment.

The [capacity-localization study](CAPACITY_LOCALIZATION_BALANCE.md) now tests
physical nodal flow in addition to the discrete phase words above. It finds
EPI diffusion under uniform capacity and a conditional nonuniform equilibrium
when a retained capacity profile balances that diffusion. The UM/SHA word
above has no physical EPI-flow interval, so retention of an EPI shape under
that word alone could not establish resistance to diffusion. The remaining
question is joint maintenance when canonical operators evolve the supporting
phase and capacity fields. That requires their complete effects and cannot
be inferred from integer winding alone.
