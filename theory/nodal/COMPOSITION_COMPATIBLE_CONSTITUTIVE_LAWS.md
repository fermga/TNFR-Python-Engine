# Constitutive profiles preserved by registered C9 composition

<a id="composition-compatible-constitutive-laws"></a>

The two prescribed C9 interfaces restrict an analytic edge law, but do not
select sine. Under the complete-law comparison below, the admissible current
has Fourier harmonics congruent to either one or minus one modulo nine.
Matching the organized target storage, tangent and quadratic field imposes
three further linear conditions. Matching individual edge currents imposes
a fourth, stronger condition. An explicit nonnegative potential satisfies
all of them while changing the nonlinear law. These are conditional
classification and counterexample results, not acquisition or physical
identification results.

The [registered inheritance owner](ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md)
defines the two interfaces and their source-specific certificates. This
note changes the constitutive class and therefore re-establishes the
consumed algebra and regularity. The original acquired sources, recovery
dwell, finite contact bounds and evaluated evidence are not reused as
certificates for these new laws.

<a id="composition-compatible-state"></a>
## Complete law, identity and the comparison class

Retain two labelled C9 components, node order \(9r+j\), with
\(r=0,1\), \(j=0,\ldots,8\), nine unit cycle edges per component and
\(\alpha=2\pi/9\). Choose exactly one supplied unit cross interface:

\[
 \mathcal S_1=\{0\},\qquad \mathcal S_3=\{-1,0,1\},\qquad
 (0,j)--(1,j+s),\quad s\in\mathcal S_q.
 \tag{1}
\]

Indices are modulo nine. There are \(9q\) cross edges and every final
degree is \(d=2+q\). Constituent membership, local labels, orientation
and attachment registration are retained boundary data. In particular,
the first phase resultant of an ideal winding-one C9 vanishes, and its
ninth moment loses the registration modulo \(\alpha\). Neither replaces
the retained collective phase used here.

The complete fine state is signed real form and primitive circular phase
with declared continuous lifts. Capacities are held at one, the support is
fixed between any separately prescribed events, and the unforced clock is
\(\tau=(1023/1024)t\). Supply an odd real-analytic \(2\pi\)-periodic
current \(j\), with even periodic potential
\(V(\delta)=\int_0^\delta j(s)\,ds\), normalized by \(V(0)=0\).
Require \(V\ge0\) independently. The reciprocal coefficient is
\(\gamma>0\), with \(1/(1023\pi)\) the earlier fine-model value.
The comparison class permits the mapped positive coefficient derived below;
holding it fixed would be a different equivalence question.

With the actual support Laplacian \(L\), let

\[
 \begin{split}
 S_{j,i}(\theta)&=\sum_{\ell\sim i}j(\theta_\ell-\theta_i),\\
 x'&=-d^{-1}Lx+\gamma d^{-1}S_j(\theta),\\
 \theta'&=\gamma d^{-1}Lx,\\
 H_V&=\tfrac12x^TLx+
                  \sum_{\{i,\ell\}}V(\theta_\ell-\theta_i).
 \end{split}\tag{2}
\]

In original time both evolution rows acquire the factor \(1023/1024\).
The first row supplies the pressure in the unit-capacity nodal identity;
it is not reconstructed from a response. The structural chart fixes its
units, with \([p]=[x]/([\nu_f][t])\); no laboratory time, sensor or energy
unit follows. Pair antisymmetry and reciprocal cancellation give

\[
 H_V'=-d^{-1}\|Lx\|_2^2,\qquad
 (\mathbf1^Tx)'=(\mathbf1^T\theta)'=0.
 \tag{3}
\]

The candidate registered organization is the embedded family

\[
 x_{r,j}=X_r,\qquad
 \theta_{r,j}=\Phi_r+\alpha(j-4).
 \tag{4}
\]

It retains two forms, two registered phases, the internal winding-one
identity and the specified boundary support. Common form-origin and
phase-origin shifts, graph relabellings preserving the full supplied data,
and simultaneous form/phase sign reversal preserve the corresponding law.
The phase-origin symmetry does not discard relative registration.
Away from (4), constituent means need not close; no arbitrary fine state
is identified with an exact collective state.

This is the detached continuous/proof path under the
[NFR property admission](../EMERGENT_ONTOLOGY.md#12-can-nfr-properties-justify-a-joint-law).
It neither runs nor changes the native relational engine or the general
operator runtime. A supplied contact is not an automatically admitted
Coupling event. Named operators still need their own live admission,
[collective descent](../STRUCTURAL_OPERATORS.md#collective-operator-descent)
and occurrence arguments. Continuous closure of (4) establishes none of
those event claims.

<a id="composition-compatible-classification"></a>
## Necessary and sufficient harmonic and target conditions

Define the interface action and geometric factors

\[
 (A_q j)(\delta)=\frac1q\sum_{s\in\mathcal S_q}
                          j(\delta+\alpha s),\qquad
 k_q=\frac q{2+q},\qquad
 r_1=1,\quad r_3=\frac{1+2\cos\alpha}{3}.
 \tag{5}
\]

The internal currents on (4) cancel by oddness and periodicity. Every
cross sum depends only on \(\Delta=\Phi_1-\Phi_0\), so both rows are
tangent to (4). Writing \(u=X_1-X_0\), the exact restriction is

\[
 \begin{array}{ll}
 X_0'=k_q[u+\gamma A_qj(\Delta)],&
 X_1'=-k_q[u+\gamma A_qj(\Delta)],\\
 \Phi_0'=-\gamma k_qu,&\Phi_1'=\gamma k_qu.
 \end{array}\tag{6}
\]

Analytic periodic currents and their first derivatives are bounded on the
real circle. Together with the linear form rows this gives a globally
Lipschitz lifted field and unique global continuation. Thus the tangent
calculation proves invariance of (4), not merely instantaneous matching.

Here **profile preservation** means that a positive constant common clock
rescaling, a positive common form-unit chart \(X_r=aZ_r\), unchanged collective
phase, and one positive reciprocal coefficient put both rows into the
normalized reciprocal pair law with the *same* function \(j\). Diffusion
fixes the clock to \(\sigma=k_q\tau\); equality of the two reciprocal
coefficients then forces

\[
 A_qj=\rho_qj,\qquad a^2=\rho_q>0,
 \qquad\gamma_{\rm eff}=\gamma\sqrt{\rho_q}.
 \tag{7}
\]

This defines the equivalence class; it does not exclude all nonlinear
conjugacies with different inputs or observations. Matching contacts have
\(A_1=I\), so their profile condition is automatic.

Require the organized storage and local derivative data to agree with sine
at the consumed gaps
\(\delta=0,\alpha,2\alpha\):

\[
 V(\delta)=1-\cos\delta,\quad
 j'(\delta)=\cos\delta,\quad
 j''(\delta)=-\sin\delta.
 \tag{8}
\]

Negative gaps then follow by parity. On the declared target supports,
these conditions match target storage, the complete tangent and the full
quadratic field. The target vector field itself vanishes for every odd
current: opposite currents cancel at cycle nodes and symmetric contacts,
and aligned bridge currents are zero. Matching those sums therefore does
not require matching each nonzero edge current individually. Both nonzero
winding gaps are included even though the
composition in (4) uses winding one. They admit the same local target
comparison on the earlier supports consuming winding two and aligned
bridges; they do not transfer that model's acquisition proof.

Odd real-analytic periodicity supplies a sine series
\(j(\delta)=\sum_{n\ge1}b_n\sin(n\delta)\) with real exponentially
decaying coefficients. Equivalently, use a holomorphic extension to some
open strip about the real circle and any smaller closed strip to bound
the coefficients. This justifies the termwise shifts, integrals and all
finite-order derivatives used below. Conversely, exponential decay
supplies the required analytic current.

For each sine harmonic,

\[
 A_3\sin(n\delta)=\frac{1+2\cos(n\alpha)}3\sin(n\delta).
 \tag{9}
\]

Differentiating (7) at zero and applying (8) gives

\[
 \rho_3j'(0)=\tfrac13[j'(0)+2j'(\alpha)]
             =\frac{1+2\cos\alpha}{3}=r_3.
 \tag{10}
\]

Since \(j'(0)=1\), necessarily \(\rho_3=r_3\). Uniqueness of Fourier
coefficients in (9) now gives

\[
 b_n=0\quad\text{unless }n\equiv1\text{ or }-1\pmod9.
 \tag{11}
\]

Let \(\mathcal N\) be those positive integers and put
\(s_n=1\) for \(n\equiv1\pmod9\), \(s_n=-1\) for
\(n\equiv-1\pmod9\). For \(k=0,1,2\), the retained frequencies obey
\(\cos(nk\alpha)=\cos(k\alpha)\) and
\(\sin(nk\alpha)=s_n\sin(k\alpha)\). The normalized potential is

\[
 V(\delta)=\sum_{n\in\mathcal N}\frac{b_n}{n}
                                         (1-\cos n\delta).
 \tag{12}
\]

Consequently (8) is equivalent to three moment conditions:

\[
 \boxed{\quad
 \sum_{n\in\mathcal N}\frac{b_n}{n}=1,\qquad
 \sum_{n\in\mathcal N}nb_n=1,\qquad
 \sum_{n\in\mathcal N}s_nn^2b_n=1.
 \quad}\tag{13}
\]

For necessity, use the potential value at \(\alpha\),
the derivative at zero and the second derivative at \(\alpha\);
\(1-\cos\alpha\) and \(\sin\alpha\) are nonzero. For sufficiency,
substitution gives every value in (8), including the entire set at
\(2\alpha\). The second winding therefore adds no independent moment
after the interface restriction (11), although it remains part of the
matched-target claim.

Equations (11)--(13), real exponential coefficient decay and
\(V(\delta)\ge0\) for every real \(\delta\) are jointly necessary
and sufficient for this comparison. Nonnegativity is a separate global
condition; these equalities do not imply it. This classification never
assumes first-neighbor-resultant sufficiency. That
[different information premise](SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency)
would remove the higher harmonics by a separate argument.

The stronger edge-current requirement \(j(k\alpha)=\sin(k\alpha)\)
for \(k=1,2\) is equivalent to the additional condition
\(\sum_ns_nb_n=1\). It is not redundant. For example, the perturbation
current
\[
 h(\delta)=57\sin\delta-708\sin8\delta
                         +185\sin10\delta+221\sin17\delta
\]
has zero perturbations of all three moments in (13), but its signed
current moment is 729. Its normalized potential
\(Q=\int_0^\delta h(s)\,ds\) obeys
\(|Q(\delta)|\le11328(1-\cos\delta)\): apply
\(1-\cos(n\delta)\le n^2(1-\cos\delta)\) to each coefficient.
Thus \(V=1-\cos\delta+eQ\), \(|e|\le1/22656\), is a nonnegative
counterexample to deducing that fourth condition from the actual matched
target fields. The next family satisfies even the stronger condition.

<a id="composition-compatible-counterfamily"></a>
## An explicit nonnegative nonsine family

For the exact real parameter \(|\epsilon|\le1/324\), define

\[
 P(\delta)=\cos\delta(1-\cos9\delta)^2,\qquad
 V_\epsilon(\delta)=1-\cos\delta+\epsilon P(\delta),\qquad
 j_\epsilon=V_\epsilon'.
 \tag{14}
\]

These functions are analytic, periodic and have the required parity.
The elementary bound \(|\sin(9z)|\le9|\sin z|\) for real \(z\)
gives \(1-\cos9\delta\le81(1-\cos\delta)\). Combining it with
\(1-\cos9\delta\le2\) proves

\[
 |P(\delta)|\le162(1-\cos\delta),\qquad
 \tfrac12(1-\cos\delta)\le V_\epsilon(\delta)
                         \le\tfrac32(1-\cos\delta).
 \tag{15}
\]

Thus the potential is nonnegative globally and vanishes only at full
turns. Its positivity is proved without sampling phase or selecting a
trajectory. Expanding its five harmonics gives

\[
 P(\delta)=\tfrac32\cos\delta-\cos8\delta-\cos10\delta
                     +\tfrac14\cos17\delta+\tfrac14\cos19\delta.
 \tag{16}
\]

| Frequency \(n\) | Current coefficient \(b_n\) | Residue sign \(s_n\) |
| --- | --- | --- |
| 1 | \(1-3\epsilon/2\) | +1 |
| 8 | \(8\epsilon\) | -1 |
| 10 | \(10\epsilon\) | +1 |
| 17 | \(-17\epsilon/4\) | -1 |
| 19 | \(-19\epsilon/4\) | +1 |

All other coefficients vanish. Each is in (11), and direct substitution
satisfies (13) and the additional edge-current moment. There is also a
local verification:
at every \(\delta=k\alpha\), the factor
\((1-\cos9\delta)^2\) has a zero of order four. Hence

\[
 P=P'=P''=P'''=0,\qquad
 P''''(k\alpha)=6\,9^4\cos(k\alpha).
 \tag{17}
\]

The organized target, storage, full tangent and quadratic field therefore
agree with sine. The cubic field generally changes:

\[
 j_\epsilon'''(k\alpha)
       =[-1+39366\epsilon]\cos(k\alpha).
 \tag{18}
\]

Every nonzero admitted \(\epsilon\) is genuinely nonsine in the declared
chart. In particular, \(P'(\pi/2)=-1\) gives the exact static witness

\[
 j_\epsilon(\pi/2)=1-\epsilon\ne1.
 \tag{19}
\]

The unchanged derivative \(j_\epsilon'(0)=1\) excludes a mere global
current multiplier. This is a different complete nonlinear vector field,
not a finite measured response or a claim excluding every possible
nonlinear state conjugacy.

The coefficient identities hold algebraically outside the admitted interval,
but positivity need not survive. For example, taking \(\epsilon=2\) in
(14) gives \(V_2(\pi)=2-8=-6\), while preserving all three target moments
and the additional edge-current moment. This explicitly separates the global
storage condition from the linear matching conditions; that value is rejected
by the static family's admission.

This freedom has arbitrarily many independent directions. For each
integer \(m\ge2\), the even trigonometric polynomial
\(P_m(\delta)=\cos\delta(1-\cos9\delta)^m\) has only the same
residue frequencies, and has a zero of order \(2m\) at every target
gap. Any finite sum

\[
 V=1-\cos\delta+\sum_{m=2}^M a_mP_m(\delta),\qquad
 \sum_{m=2}^M81\,2^{m-1}|a_m|\le\tfrac12
 \tag{20}
\]

obeys (11)--(15)'s nonnegative lower bound and all target conditions.
The bound follows from
\((1-\cos9\delta)^m\le81\,2^{m-1}(1-\cos\delta)\).
The distinct polynomial powers are linearly independent. There is thus
no finite count of alternatives selected by these constraints. The
uniform derivative constants below apply only to (14), not automatically
to every finite sum in (20).

<a id="composition-compatible-local-identity"></a>
## Own-law regularity and local target identity

For (14), summing absolute Fourier derivative coefficients gives

\[
 \begin{split}
 |j_\epsilon'|&\le1+328|\epsilon|\le163/81,\\
 |j_\epsilon''|&\le1+\tfrac{8913}{2}|\epsilon|
                        \le3187/216<15.
 \end{split}\tag{21}
\]

At any edge target \(0,\pm\alpha,\pm2\alpha\), its potential curvature
is exactly the sine curvature, whose minimum is
\(\cos(4\pi/9)>1/6\). One algebraic proof of the latter bound sets
\(c=\cos(4\pi/9)\in(0,1/2)\); it is the root there of
\(8c^3-6c+1=0\), a strictly decreasing polynomial with positive value
at \(c=1/6\). The mean value estimate from (21) consequently gives

\[
 V_\epsilon''(\delta)>\frac{11}{120}>0
 \quad\text{when the edge is within }1/200
 \text{ of one of those target gaps}.
 \tag{22}
\]

This re-admits a local convex cell for the actual potential. On any of
the connected target supports with these edge gaps, opposite cycle
currents and zero aligned bridge currents still make the target an
equilibrium. On the two-C9 three-offset support the extra opposite
cross currents likewise cancel at alignment. Its full tangent is

\[
 \mathcal J=
 \begin{pmatrix}
 -D^{-1}L&-\gamma D^{-1}L_c\\
 \gamma D^{-1}L&0
 \end{pmatrix},
 \qquad
 Q=\tfrac12x^TLx+\tfrac12y^TL_cy,
 \tag{23}
\]

where \(D\) is the actual positive degree matrix and \(L_c\) the
target curvature-weighted Laplacian. Both \(L\) and \(L_c\) have only
the common-origin kernel on a connected support, because every target
curvature is positive.
Restricting to fixed conserved \(D\)-weighted form and lifted phase
means removes those two origins. For regular interfaces these are the
ordinary means. Along the linear tangent,
\(Q'=-x^TLD^{-1}Lx\). The only trajectory in its zero-loss set has
\(Lx=0\), then \(L_cy=0\), so the relative tangent is asymptotically
stable and, in finite dimension, Hurwitz. Smooth nonlinear continuation
therefore gives local asymptotic stability for the actual law; (22) also
provides its own strictly convex storage neighborhood.

This is an independently admitted local identity, conditional on starting
sufficiently close with sufficiently small full excess storage. It gives
neither the old radius \(1/12\), a source-to-cell route, a uniform recovery
dwell, nor a contact certificate for a previously acquired fine state.
Nonlinear remainders, source transport and event bounds must be rebuilt
before asserting any of those properties. The entire invariant family
(4) need not be transversely stable at every collective phase.

<a id="composition-compatible-complete-map"></a>
## Both evolution rows, storage and observations inherit together

For every current admitted by the classification, not only (14),
\(A_qj=r_qj\). Integrating this equality and evaluating at zero gives

\[
 A_qV(\Delta)=(1-r_q)+r_qV(\Delta).
 \tag{24}
\]

Indeed, \(A_1V(0)=0\), while
\(A_3V(0)=2V(\alpha)/3=1-r_3\) by target matching.
Let \(B_q=18(1-\cos\alpha)+9q(1-r_q)\). The full fine storage and
loss on (4) are exactly

\[
 H_V|_{\mathcal M}=B_q+9q[\tfrac12u^2+r_qV(\Delta)],
 \qquad\frac{dH_V}{d\tau}=-18qk_qu^2.
 \tag{25}
\]

The inherited chart and clock are forced within the declared map class:

\[
 \sigma=k_q\tau,\qquad X_r=\sqrt{r_q}Z_r,
 \qquad\gamma_{\rm eff}=\gamma\sqrt{r_q}.
 \tag{26}
\]

For \(z=Z_1-Z_0\), both complete rows and the full storage become

\[
 \begin{array}{ll}
 \displaystyle\frac{dZ_0}{d\sigma}=z+\gamma_{\rm eff}j(\Delta),&
 \displaystyle\frac{dZ_1}{d\sigma}=-z-\gamma_{\rm eff}j(\Delta),\\
 \displaystyle\frac{d\Phi_0}{d\sigma}=-\gamma_{\rm eff}z,&
 \displaystyle\frac{d\Phi_1}{d\sigma}=\gamma_{\rm eff}z,
 \end{array}
 \tag{27}
\]

\[
 H_V|_{\mathcal M}=B_q+9qr_q[\tfrac12z^2+V(\Delta)],
 \qquad\frac{dH_V}{d\sigma}=-18qr_qz^2.
 \tag{28}
\]

For (14), \(\epsilon\) is unchanged by either interface. The reciprocal
coefficient is unchanged only for matching contacts. Three-offset
contacts still require its change and the form chart; a fixed-\(\gamma\)
family is not closed under this complete comparison. Unit fine capacities
are retained, and \(k_q\) is a derived common rate, not an inferred
capacity law or an inverse mass.

Uniform constituent form inputs \(b_r(\tau)\) and independently supplied
phase inputs \(c_r(\tau)\) preserve (4). They transform respectively to
\(b_r/(k_q\sqrt{r_q})\) and \(c_r/k_q\) in (27), with their original
time argument transformed as well. The full continuous balance is

\[
 \frac{dH_V}{d\tau}=-18qk_qu^2+
            9q[u(b_1-b_0)+r_qj(\Delta)(c_1-c_0)].
 \tag{29}
\]

Dividing by \(k_q\) and using (26) reproduces the same work in (28).
A uniform form jump \(a_r\) becomes \(a_r/\sqrt{r_q}\); a uniform
phase jump keeps its angle. Their work is the difference of the full
storage. A raw form observation reads \(X_r=\sqrt{r_q}Z_r\), not
\(Z_r\). Any more general observation is composed with the same fine
embedding and chart. Independent sensor calibration and discarded
constituent information remain separate obligations.

Finally, attaching the specified support without changing state requires
the exact event work

\[
 \mathcal W_q=\sum_{j,s\in\mathcal S_q}
 \left[\tfrac12(x_{1,j+s}-x_{0,j})^2+
                 V(\theta_{1,j+s}-\theta_{0,j})\right]\ge0.
 \tag{30}
\]

On (4) it is \(9q[\tfrac12u^2+(1-r_q)+r_qV(\Delta)]\).
At the aligned target it is zero for matching and
\(18(1-\cos\alpha)>0\) for three offsets. The additive constant in
(24) is therefore retained even though it disappears from continuous
derivatives. Positive work needs a declared external supply, and neither
passivity nor the continuous law selects event occurrence.

<a id="composition-compatible-scope"></a>
## What the classification establishes

The two named interfaces constrain the harmonic content and the matched
targets constrain three coefficient moments. Even imposing the fourth
moment for individual edge-current equality, together with nonnegative
storage, these requirements still admit the explicit nonsine laws above.
Thus same-profile organized composition does not select a unique current
in the stated analytic class. Repeating either proved algebraic interface
map leaves those profiles unchanged; it does not construct the nested
geometries, sources or events needed to repeat the physical organization.

Composition at different phase registrations, other cycle sizes, every
possible interface, a first-resultant information restriction or a different
operator-descent requirement would add premises. None follows from this
two-interface calculation. Any further selection principle must be justified
against retained NFR and boundary state rather than chosen to remove the
counterexample. The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns that decision; this proof creates no parallel research queue.

The [oriented weighted-interface comparison](ORIENTED_INTERFACE_PROFILE_INHERITANCE.md)
now isolates one such extra premise: unchanged profiles across its declared
open interface interval select sine within this class. The other currents
still inherit exact boundary-dependent laws, so that conditional selector
does not follow from the NFR ontology or invalidate the present counterfamily.

## Implementation and independent controls

The detached static owner
[`_composition_compatible_constitutive.py`](../../src/tnfr/physics/_composition_compatible_constitutive.py)
admits the exact parameter domain of (14), derives its finite harmonic
coefficients and observes edge jets and both complete collective rows.
The static pair instrument fixes the fine reciprocal coefficient to
\(1/(1023\pi)\), while the proof permits any positive fine coefficient
with its correspondingly mapped comparison value.
It reuses the shared registered-interface geometry and complete chart
data without transferring a sine acquisition certificate. It represents
the explicit stronger counterfamily, not a decision procedure for global
nonnegativity of every analytic series in the classification.

Its [independent controls](../../tests/physics/test_composition_compatible_constitutive.py)
rebuild the potential and derivatives, target moments, exact nonsine
witness, full fine-state restriction, storage and transformed observations.
These algebraic and static tests check implemented behavior; they do not
evaluate a trajectory, form an organization under the new law, freeze a
physical protocol or regenerate previous scientific evidence.
