# Collective response inherited from constituent deviations

<a id="organized-constituent-memory-declaration"></a>
## Analytical question and fixed comparison

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#organized-inherited-response-admission)
asks which collective response follows from the existing fine dynamics,
instead of a separately supplied response matrix. The
[electrical protocol](ORGANIZED_ELECTRICAL_PROTOCOL.md) owns the complete
five-child law and source. Its sine-covariance remainder, quadratic norm bound
and uniform-child input/work pairing are starting results, not this question's
answer. The [ontology](../EMERGENT_ONTOLOGY.md#nfr-participation-in-form)
requires the effective object to retain its constituents' actual influence.

| Item | Fixed analytical scope |
| --- | --- |
| Fine support | Unit undirected `C9 x C5`, node `9*c+j`, five children and nine slots, 90 edges; no donor, shifted registration or support event |
| Complete law | `x'=(-L45*x+gamma*S(theta))/4+R*f`, `theta'=gamma*L45*x/4`; full-circle pairwise sine, held unit capacities, `gamma=1/(1023*pi)` |
| Clock and input | Prime in `tau=(1023/1024)*t`; R repeats each child input over its nine slots. Compare prescribed neutral piecewise-continuous f under the unchanged input map; use a common differentiable input for derivative witnesses |
| Observation | Actual arithmetic child form means U and the existing circular collar observer's continuous lifted child phases Phi. Write `x=R*U+u`, `theta=psi+R*Phi+z`, `psi_j=2*pi*(j-4)/9`, `R^T*u=R^T*z=0` |
| Actual source | Retain the acquired hosts and their proved norm bounds. General identities apply on their admitted observation intervals; no reached state is replaced or regenerated |
| Counterexample domain | The existing finite-observation larger inference class: `abs(x_i)<=1/100`, `abs(z_cj)<=1e-6`, principal mean gaps at most `7/5`, outer winding in `{-1,0,1}`. Membership in this class does not prove acquisition |
| Time/domain | Exact identities on finite intervals with the admitted continuous lift. Norm certificates additionally require `s=2*g*T<1`, `g>=gamma`, and `2*Z<pi/2`, with Z rebuilt from primitive hidden-state bounds. General new inputs do not inherit the old word's full winding-retention certificate |
| Comparator | The supplied `W_r=I+r*L5^2` law in the same U, Phi, input and clock. Chart changes, altered input maps, fitted coefficients and changed fine laws are separate comparisons |
| Decision | Characterize nonlinear hidden-history dependence beyond the existing norm bound; distinguish carrier tangent behavior from actual constituent feedback; prove or exclude fixed positive-r response inheritance on the declared input/domain class |
| Evidence | Exact projected identities, independent edge sums and static derivative/sign controls. No trajectory, physical dataset, parameter scan, source assessment, reserved response or measured derivative is evaluated |

An implicit hidden-state integral equation is not by itself a reduced solver
or a finite-dimensional closure. Any approximation must retain its hidden
initialization, input dependence and own residual. Earlier memory identities
and source transport remain valid on their own models. This comparison
neither derives primitive support nor establishes a new physical property.

<a id="organized-constituent-memory-result"></a>
## Result: nonlinear feedback without a carrier tangent kernel

Declaration `78d460df` fixes the comparison above. In its natural means and
unchanged input map, no fixed positive-r response matrix is an exact
all-input inheritance. The internal phase deviations instead generate a
nonlinear current correction with a leading quadratic covariance term.
Their evolution retains hidden initialization and collective phase history.
On the exact carrier the hidden state stays zero, and the full tangent has
no collective-hidden coupling. That zero tangent kernel does not remove
the nonlinear correction away from the carrier.

An explicit pair in the larger inference class has equal collective state,
initial current correction, full storage and instantaneous loss, but opposite
derivatives of that correction. This identifies information missing even
after adding the instantaneous correction to the means. It is a local
model obstruction, not a claim that both states were reached by the actual
source preparation or that their finite responses were observed.

## 1. Exact projected law and source domain

Write `L_f=I5 tensor L9+L5 tensor I9`, `P=RR^T/9`, `Q=I-P`. The fixed
template psi is repeated in each child. The observation gives
`U=R^T*x/9`, `Phi=R^T*(theta-psi)/9` on its declared continuous lift.
Since `L_f R=R L5` and `Q L_f=L_f Q`, the complete projected rows are

\[
 \begin{aligned}
 U'&=(-L_5U+\gamma S_5(\Phi))/4+f+\mathcal R(\Phi,z),
 &\Phi'&=\gamma L_5U/4,\\
 u'&=-L_fu/4+(\gamma/4)Q S(\psi+R\Phi+z),
 &z'&=\gamma L_fu/4.
 \end{aligned} \tag{1}
\]

The exact correction is the one in the
[protocol](ORGANIZED_ELECTRICAL_PROTOCOL.md), not an inferred external force.
Uniform-child input obeys `Q R f=0`. This removes direct hidden forcing;
it does not make the hidden dynamics independent of the collective phases.
The input changes U and Phi, which enter its nonlinear sine map.

For primitive bounds `||u0||<=a0`, `||z0||<=b0`, let `g>=gamma`,
`s=2gT<1`, and retain the existing transport bounds

\[
 A=\frac{a_0+s b_0}{1-s^2},\qquad
 Z=\frac{b_0+s a_0}{1-s^2}.
 \tag{2}
\]

The [shared transport](../../src/tnfr/physics/_organized_electrical_source.py)
rebuilds these from the primitive bounds. They apply to (1) because the
heat part contracts, `||L_f||<=8`, and the sine map is globally Lipschitz
with that bound. Requiring `2Z<pi/2` retains the circular child observer.
The earlier actual source satisfies its proved bounds under the matching
law, support and input hypotheses. The identities below need no source
replay. They do not assert that every point of its norm enclosure was formed.
In particular, generic new input histories require their own outer-identity
and physical observation arguments.

## 2. Exact hidden-history dependence

Put `E(t)=exp(-t*L_f/4)` and
`G(Phi,z)=Q S(psi+R Phi+z)`. Variation of constants in (1) gives

\[
 u(t)=E(t)u_0+\frac\gamma4\int_0^t E(t-s)G(\Phi(s),z(s))\,ds,
 \tag{3}
\]
\[
 z(t)=z_0+\gamma[I-E(t)]u_0+
       \frac{\gamma^2}{4}\int_0^t[I-E(t-s)]G(\Phi(s),z(s))\,ds.
 \tag{4}
\]

Equation (4) follows by integrating the live z row and exchanging the finite
integrals in (3); `integral_0^t L_f E(s) ds=4[I-E(t)]` supplies its coefficient.
No inverse on the common mode or phase reconstruction from a response is used.
For a prescribed continuous Phi history the globally Lipschitz hidden rows
have a unique solution. Therefore (4) defines an implicit causal functional
`z[Phi;u0,z0]`, and the actual mean correction is
`R(Phi(t),z[Phi;u0,z0](t))`. Uniform input affects this functional through
the coupled collective history. Setting either hidden initial term to zero
without its preparation premise changes the problem.

This is a model-specific reuse of
[hidden-state elimination](RELATIONAL_PATTERN_MEMORY.md#3-exact-hidden-memory-retains-its-initial-condition).
It still requires solving the hidden history. It is neither a known linear
convolution of U nor a reduced numerical method, positive memory kernel,
finite cutoff or new delayed operator. The new restrictions below concern
this product support and its nonlinear feedback, beyond that generic identity.

## 3. The invariant carrier and its decoupled tangent

For every Phi, internal helical sine currents cancel and same-slot external
currents are constant within each child, so `G(Phi,0)=0`. Thus u=z=0 is
invariant for every admitted uniform-child input. Such an input cannot
generate hidden departures from exactly zero in this law.

At the carrier, let `K(Phi)=-D S(psi+R Phi)`. Its exact form is

\[
 K(\Phi)=\cos(2\pi/9)(I_5\otimes L_9)
            +L_5^{\cos(\Phi)}\otimes I_9,
 \tag{5}
\]

where the weighted outer Laplacian uses the cosine of each current outer
phase gap. It can be time dependent and need not be positive for arbitrary
Phi. Both K and L_f commute with P and Q. Consequently `R^T K Q=0` and
`Q K R=0`: the full collective and hidden variational equations decouple
along any carrier trajectory. The hidden tangent generator is

\[
 \mathcal A_h(\Phi)=\frac14
 \begin{pmatrix}-L_f&-\gamma K(\Phi)\\ \gamma L_f&0\end{pmatrix}
 \quad\hbox{on }\operatorname{im}Q\oplus\operatorname{im}Q.
 \tag{6}
\]

Its state transition may depend on the entire carrier history. Nevertheless,
both couplings to the collective tangent vanish, so linear elimination
produces an identically zero collective memory kernel there. This is a
symmetry result for the selected observation and input, not a general
absence of memory or a transverse stability theorem. Freezing K or replacing
its trigonometric coefficients by rationals would require a separate model
or controlled approximation.

## 4. The first nonlinear contribution and its retained information

Orient a C5 edge `c -> d` and put
`alpha=Phi_d-Phi_c`, `d_j=z_dj-z_cj`, `m_cd=(1/9)sum_j d_j^2`.
Because `sum_j d_j=0`, Taylor's theorem gives

\[
 \frac19\sum_j\sin(\alpha+d_j)-\sin\alpha
   =-\tfrac12\sin\alpha\,m_{cd}+r_{cd},\qquad
 |r_{cd}|\le\frac1{54}\sum_j|d_j|^3.
 \tag{7}
\]

Choose incidence B with a -1 at each edge's tail and +1 at its head, so
`S5=-B sin(B^T Phi)`. The leading correction and its error satisfy

\[
 \mathcal R_2=\frac\gamma8 B
      [\sin(B^T\Phi)\mathbin\odot m],\qquad
 \|\mathcal R-\mathcal R_2\|_1
       \le\frac\gamma{108}\sum_{\{c,d\},j}|d_j|^3
       \le\frac{2\gamma}{27}Z^3.
 \tag{8}
\]

The last bound uses `max|d_j|<=2Z` and
`sum_edges,j d_j^2=z^T(L5 tensor I9)z<=4Z^2`. It is a uniform pointwise
error when the actual hidden state satisfies (2); it is not a forecast
obtained from a tangent approximation to that state. The quadratic term
may vanish when outer sines or the incidence sum cancel. At nonzero outer
phase gaps it is generically present even though the carrier tangent
kernel is zero. The edge variance contains correlations between constituent
slots, rather than a freely fitted response coefficient.

To see the next information obligation, write `xi=(u,z)` and
`C=xi xi^T`. Along the hidden tangent, exactly

\[
 C'=\mathcal A_h C+C\mathcal A_h^T.
 \tag{9}
\]

For the actual nonlinear rows, an additional
`n xi^T+xi n^T` is present, where
`n=((gamma/4)[G(Phi,z)+K(Phi)z],0)=O(||z||^2)` on a bounded regular domain.
Its contribution is of third or higher order in the joint hidden amplitude.
The phase covariance block alone consumes the form-phase cross blocks:
`Czz'=(gamma/4)(L_f Cuz+Czu L_f)`. The same edge variances therefore do
not close their own evolution. A covariance approximation needs its own
propagated error, initial covariance and collective-history model; (9)
alone is not an exact nonlinear moment closure or a dimension reduction.

There is also an exact obstruction to replacing all hidden state by C.
At U=0 and the outer winding-one twist, take u=0 and let child zero's
hidden phases be `(a,a,-2a,0,0,0,0,0,0)`, with other children zero and
`a=2^-40`. Replacing z by -z preserves `xi xi^T` and R_2. However,
`mean_j sin(z_0j)=[2 sin a-sin(2a)]/9`, so

\[
 \mathcal R_0(z)=-\frac\gamma9\cos(2\pi/5)\sin a(1-\cos a)<0,
 \qquad \mathcal R_0(-z)=-\mathcal R_0(z)>0.
 \tag{10}
\]

Both preparations lie in the same declared inference class. They have the
same full second moment but different instantaneous collective correction.
Thus second moments are a controlled approximation target, not an exact
state representation for the nonlinear law. The cubic remainder in (8)
retains precisely information that a covariance-only description can lose.

## 5. Equal current correction, storage and loss, different acceleration

Let `kappa=2*pi/5`, `Phi_c=(c-2)*kappa`, U=0 and
`h=(1,-1,0,0,0,0,0,0,0)`. Choose `epsilon=2^-40`, put
`z_0j=epsilon*h_j` with all other children zero, and set `u^+=z`, `u^-=-z`.
Use the same held zero input. These are algebraic states, not new source
responses. They belong to the declared larger inference class, with zero
common means, the same winding-one cycles and strict initial collars.
No inclusion in the image of the actual acquired source is inferred.

Both states have identical phases, U, Phi, R, U' and Phi'. Their exact
correction is

\[
 \mathcal R=\frac{\gamma\sin\kappa(1-\cos\epsilon)}{18}
                (0,1,0,0,-1).
 \tag{11}
\]

Fine form storage is `u^T L_f u/2=5epsilon^2` in both; phase storage
agrees because the phases are identical. Also
`||L_f u||^2/4=14epsilon^2` for both and the supplied work is zero.
Thus total storage and instantaneous loss do not remove their ambiguity.

The full phase row gives opposite hidden velocities,
`(z')^+ = gamma*L_f*z/4` and `(z')^- = -(z')^+`, while Phi'=0.
For the edge from child 1 to child 0,
`(L9+3I)h=(6,-6,1,0,0,0,0,0,-1)`. Differentiating its actual sine average
gives `6[cos(kappa-epsilon)-cos(kappa+epsilon)]`
`=12 sin(kappa) sin(epsilon)`, hence

\[
 (\mathcal R'_1)^\pm
    =\pm\frac{\gamma^2\epsilon\sin\kappa\sin\epsilon}{12},
 \qquad (U''_1)^+-(U''_1)^-
    =\frac{\gamma^2\epsilon\sin\kappa\sin\epsilon}{6}>0.
 \tag{12}
\]

Other terms in the differentiated mean row agree. Even the enlarged
instantaneous observation `(U,Phi,R,H_f,loss_f)` therefore has no autonomous
law valid for this whole class. The missing form-phase association determines
how the same initial phase variance will evolve. This is a strict local
derivative obstruction; no finite readout margin, time horizon for a resolved
signal, or acquisition of this pair is claimed.

## 6. Fixed matrix inheritance and actual work

At any fixed admitted fine state, change the prescribed neutral input by d.
The fine row in (1) changes by R d, and the observed mean derivative changes
by exactly d. The proposed positive-r law in the same means and clock instead
changes it by `M_r d`, with `M_r=(I+r L5^2)^-1`. Equality for every neutral d
forces `M_r=I` on that subspace, hence r=0 because L5 is invertible there.
A bounded input switch leaves the state continuous and tests this channel
without assigning an artificial duration to an event.

This obstruction holds even at an actual acquired state. It excludes exact
all-input inheritance of the positive-r matrix under the fixed observation
and input map; it does not exclude accidental equality for one history,
a separately justified approximation, another chart, boundary actuation or
a changed constitutive law. At r=0, the hidden correction still has to be
retained away from the exact carrier.

With `H_f=x^T L_f x/2+sum_edges(1-cos Delta theta)` and
`H_5=U^T L5 U/2+sum_C5(1-cos Delta Phi)`, the existing storage split and (1)
give

\[
 \begin{aligned}
 H_f'&=-\tfrac94\|L_5U\|^2-\tfrac14\|L_fu\|^2
                      +9(L_5U)^Tf,\\
 H_5'&=-\tfrac14\|L_5U\|^2+(L_5U)^T(f+\mathcal R),\\
 R_H'&=-\tfrac14\|L_fu\|^2-9(L_5U)^T\mathcal R.
 \end{aligned} \tag{13}
\]

The correction exchanges signed storage between retained and hidden state.
It supplies no additional external work and need not dissipate the collective
storage by itself. R_H remains signed, rather than a nonnegative reserve.
The matrix law has a different own storage; comparing them cannot replace
this inherited balance or silently count the same interaction twice.

## Verification and remaining boundary

The [static controls](../../tests/physics/test_organized_constituent_memory.py)
reconstruct fine edges, projections, fields and chain-rule derivatives.
They check the heat-integral normalization, tangent blocks, quadratic
correction and cubic remainder, storage/work, input obstruction and the
factored nonzero derivative witness. They reuse shared exact matrix and
rational interval arithmetic. No new solver or public API is installed;
the existing source transport remains the conditional norm owner.

These results distinguish a derived nonlinear feedback mechanism from a
supplied matrix law without identifying a physical force or selecting a
unique fundamental dynamics. The
[finite compression result](ORGANIZED_COMPRESSED_MEMORY.md#organized-compressed-memory-result)
retains five initial interface variances and propagates their nonlinear
forecast error under the same fine law. Its conditional intrinsic improvement
does not supply those statistics or initial phases from source norm bounds,
nor reduce measurement uncertainty. The
[sole queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the remaining initialization obligation.
