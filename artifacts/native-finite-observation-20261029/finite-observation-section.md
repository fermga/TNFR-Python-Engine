
<a id="native-finite-observation"></a>
## Finite receiver-form separation with explicit uncertainty

This analytical result closes the finite-observation obligation for the
**one-bridge JO preparation**. It proves a sufficient, nonempty error domain
and an explicit paired decision threshold. It does not evaluate a trajectory,
assert achievable instrument precision, or supply an independent physical
realization. All constants below are bounds derived from the complete field,
not estimates fitted to a receiver signal.

### FO1. Preparation and complete recovery admission

Retain the two unit C5 rings with the single bridge `(0,5)`, all thirty
coordinates, `e=w=1/2`, `beta=b=kappa=1`, equal constituent measure and the
JO structural clock. Use the aligned winding-one phase `theta_*` and
\[
a=\eta=2^{-20},\qquad
x(0)=a e_1,\qquad \nu(0)=\mathbf1+\eta(e_0-e_1).
\tag{FO1}
\]
The dynamic law has `nu_dot=-B*nu/2`; the held law has `nu_dot=0` and
retains its actually prepared profile. Their form and phase rows are
identical functions of state. No event, additional pressure or support
change is introduced. The observation is
`y=x_5-(x_6+x_7+x_8+x_9)/4=ell^T*x`, with `||ell||_1=2`.

The actual one-bridge graph has diameter five and maximum degree three.
Summing pathwise Cauchy inequalities gives
`10*||P_0*v||^2<=225*v^T*B*v`, so `lambda_2(B)>=2/45`;
`lambda_max(B)<=6`. Re-admit the local energy argument of RB6 on this
support: for `r_rec=1/16`, each target principal edge gap is at most
`2*pi/5`, and
\[
c_r>\,1-(387/280)^2/2=7031/156800>1/32.
\]
Here `sqrt(2)*r_rec<1/8` and `pi<22/7` bound the perturbed edge gap.
Consequently `k_r>=1/1440`, and a sufficient recovery guard is
\[
Z<1/16,\qquad E_{\rm rel}<1/368640,
\quad
Z^2=\|P_0x\|_2^2+\|P_0(\theta-\theta_*)\|_2^2.
\tag{FO2}
\]
The energy is the same `E_rel=x^T*B*x/2+V(theta)-V(theta_*)` as RB6;
this is a fresh support bound, not a transfer of two-bridge capture data.
At (FO1), `Z=a*sqrt(9/10)` and `E_rel=a^2=2^-40`. More generally,
a full coordinate perturbation of sup norm at most `a`, with zero total
capacity perturbation, has `Z<6a<2^-17` and
`E_rel<=3Z^2<3*2^-34<2^-32<1/368640`.
Its capacities lie in `[1-2a,1+2a]`.

Thus both nominal and admitted perturbed trials recover. The native energy
loss retains the invariant local sublevel. Dynamic capacity stays in its
positive heat-flow hull; LaSalle for native energy plus capacity variance
in the neutral quotient identifies the equilibrium shape. JO's eventual
local exponential tail reconstructs finite common-origin limits. Held
capacity uses the positive-profile recovery theorem. This comparatively
wide recovery envelope is not itself an observable-separation budget.

### FO2. A uniform analytic remainder from the complete field

Complex coordinates here are solely a device for bounding derivatives of
the real signed-form model; they introduce no complex EPI realization.
Set `r=1/32` and consider the complex cube
`|x_i|, |theta_i-theta_*,i|, |nu_i-1|<=r`. Write
\[
Z_i^\pm=\sum_{j\sim i}e^{\pm i(\theta_j-\theta_i)},\quad
C_i=(Z_i^++Z_i^-)/2,\quad S_i=(Z_i^+-Z_i^-)/(2i).
\]
At the lock both resultants equal a real `C_i^*>1/2`. Each edge perturbation
has modulus at most `2r=1/16`. The exponential series gives
`|exp(z)-1|<=|z|/(1-|z|)<=1/15`, hence
`|Z_i^+-C_i^*|,|Z_i^--C_i^*|<=d_i/15<=1/5`.
For `0<=s<=1`, both `C_i+isS_i` and `C_i-isS_i` are convex combinations
of `Z_i^+,Z_i^-` and have modulus strictly greater than `3/10`.
The principal logarithms and reciprocals below are therefore holomorphic
on a neighborhood of the cube:
\[
g_i=\frac{\Log Z_i^+-\Log Z_i^-}{2\pi i},\qquad
h_i=\frac1{2\pi}\int_0^1
 \left((C_i+isS_i)^{-1}+(C_i-isS_i)^{-1}\right)\,ds .
\tag{FO3}
\]
On real states these are exactly the native `g_i` and `H_i^{-1}`,
including the removable zero-angle case. The integral of the logarithm's
derivative along the segment from `C_i^*` bounds each log difference by
`(1/5)/(3/10)=2/3`. Thus `|g_i|<2/9` and `|h_i|<10/9`.
With `q=Bx`, the complete rows are
\[
F_{x_i}=\nu_i[-q_i/(2d_i)+g_i/2],\qquad
F_{\theta_i}=\nu_i h_i q_i/2,\qquad
F_\nu=0\ \hbox{or}\ -B\nu/2 .
\tag{FO4}
\]
Since `|nu_i|<=33/32` and `|q_i|<=6r`,
\[
|F_{x_i}|<(1353/288)r<5r,\quad
|F_{\theta_i}|<(55/16)r<5r,\quad |F_{\nu_i}|\le3r.
\tag{FO5}
\]
The last bound uses `B*1=0`. In particular `||F||_infty<=M=5r`
for both complete laws, including their capacity directions.

For the nominal family JO6 with complex `|a|,|eta|<=p=1/128=r/4`,
Picard iteration on `|t|<=R=1/8` stays within
`r/4+R*M=7r/8<r` of the lock. The holomorphic field has a finite
Lipschitz bound on that compact convex inner cube. Successive differences
obey the usual factorial Picard bound, so the iterations converge uniformly
and define a solution jointly holomorphic in `a,eta,t`. This argument
does not require `R*L<1`. Each `|y|<=2r`, so the contrast
`Delta=y_dynamic-y_held` has `|Delta|<=4r=1/8`.

The contrast vanishes identically for `a=0` (native lock, even while
capacity redistributes) and for `eta=0` (uniform capacity). JO also gives
zero coefficients at time orders zero, one and two for this family.
These identities persist in the analytic extension. Cauchy therefore gives
\[
\Delta=\sum_{i,j\ge1,\ k\ge3}c_{ijk}a^i\eta^jt^k,\qquad
|c_{ijk}|\le\tfrac18p^{-i-j}R^{-k}.
\tag{FO6}
\]
For `|a|,|eta|<=p/2=1/256`, summing the two parameter series contributes
at most `4|a*eta|/p^2`. For `0<=t<=R/8=1/64`, the fourth time derivative
series uses `sum_(k>=4)(k)_4 u^(k-4)=24/(1-u)^5` and `(8/7)^5<2`.
It follows that
\[
|\Delta^{(4)}(t)|\le24\,2^{26}|a\eta|,\qquad
\left|\Delta(T)+\frac{a\eta(1-\eta)K}{6}T^3\right|
 \le2^{26}|a\eta|T^4 .
\tag{FO7}
\]
This is uniform over the stated nominal family. Arbitrary full-state
preparation errors need not preserve its cubic jet or its factorization;
the uncertainty contract below transports their entire response instead.

### FO3. An explicit finite nominal margin

At the actual port `h=(1+sqrt(5))/2>8/5` and `pi>3`, so
\[
K=\frac1{36}-\frac1{4\pi^2h^2}>
\frac1{36}-\frac{25}{2304}=\frac{13}{768}>\frac1{64}.
\]
For `0<eta<=1/256` this implies
`a*eta*(1-eta)*K/6>2^-9*a*eta` when `a>0`.
Choose, independently of the preparation amplitudes,
\[
T=2^{-36},\qquad H=2T=2^{-35}.
\]
The remainder coefficient in (FO7) then contributes at most
`2^26*T=2^-10` times `a*eta*T^3`, less than half the cubic lower bound.
At the concrete preparation (FO1),
\[
\Delta(T)\le-S,\qquad
S=2^{-10}a\eta T^3=2^{-158}>0.
\tag{FO8}
\]
This finite lower bound is not a sampled response or an estimate of the
actual signal size. No numerical trajectory or derivative was evaluated.

### FO4. Full preparation, clock and readout errors

Use two trials A and B. The reference B is held; A is either held
(hypothesis HH) or dynamic (hypothesis DH). Their complete initial states
are centered on the nominal prepared state z_* from (FO1), not the
unperturbed lock used as the analytic cube center. They
are `z_A=z_*+u+v_A` and `z_B=z_*+u+v_B`. The shared perturbation `u`
and separate mismatches `v_A,v_B` each have zero total capacity component.
They retain all form, phase and capacity coordinates, including both common
origins. Each held trial keeps its own actual initial capacity profile.

On the real cube used above, `C_i>1/4`, `|x_i|<=1/32`, `nu_i<2`.
The complete Jacobian estimates in RB7, now with `d<=3,m=1/4`, give
`L_x<19,L_theta<21,L_nu<=3`. Hence use `L=32` for both fields.
FO5 also gives `|y'|<=2M<1`, so use `V=1`.
Initial full-state perturbations considered here are inside `r/4`;
the same field bound supplies continuation through `H`. Straight segments
between compared real states stay in the cube. Therefore
`exp(L*H)<2`, and Gronwall plus `||ell||_1=2` bounds a prepared-state
error `epsilon_0` by `4*epsilon_0` in the scalar output.

Let the actual elapsed times from preparation be
`t_m=T+tau_s+tau_m` and the readouts be `Y_m=y_m(t_m)+n_m`.
Use the following equal numerical bounds in their respective native
coordinate, structural-time and scalar-form units:
\[
\|u\|_\infty,\|v_A\|_\infty,\|v_B\|_\infty,
|\tau_s|,|\tau_A|,|\tau_B|,|n_A|,|n_B|
\le\epsilon=2^{-166}.
\tag{FO9}
\]
Each initial state differs from nominal by at most `2epsilon` and lies
strictly inside the recovery envelope of FO1. Both sample times lie in
`[0,H]`. Each output differs from its nominal value at `T` by less than
`8epsilon+2epsilon+epsilon=11epsilon`: preparation, timing and readout,
respectively. Shared errors are retained in both radii rather than
cancelled without proof. Lower-order effects of imperfect preparation are
included in this bound.

The paired error is less than
`22epsilon=(11/128)S<S/8`. For `Q=Y_A-Y_B`,
\[
\mathrm{HH}:\quad |Q|<S/8,\qquad
\mathrm{DH}:\quad Q<-7S/8.
\tag{FO10}
\]
The explicit threshold `q_*=-S/2=-2^-159` therefore distinguishes these
two hypotheses with margin greater than `3S/8=3*2^-161` on either side.
No unknown absolute nominal output is needed for this paired threshold.
A single scalar reading without its admitted reference is not thereby
classified, and HH/DH do not exhaust all possible constitutive laws.

Timing bounds concern elapsed time, not an arbitrary absolute date. A
known exact clock change transforms every evolution row. An uncertain
affine conversion `t=alpha(s-s_0)+j` at `s=T` must instead satisfy its
budget through, for example,
`|t-T|<=|alpha-1|T+|alpha||s_0|+|j|`. An unknown clock scale is not removed
by calling the observation origin-independent. In FO9 the readout budget
is for the scalar `y`; separate node readings with individual error at most
`2^-167` suffice because `||ell||_1=2`. Only an exactly common additive
node offset cancels automatically; general gain and offset errors do not.

### FO5. Admission, abstention and remaining access

FO8-FO10 establish a nonempty sufficient finite observation contract with
explicit positive errors. The exact blind controls `a=0`, `eta=0`,
`kappa=0` and native lock remain: no positive capacity discriminator is
licensed there. Outside the proven support, complete laws, preparation,
regularity, clock or error bounds, this certificate abstains. Decreasing
the observation time does not absorb fixed noise. Neither the uncertainty
budget nor the nominal remainder is inferred from the evaluated signal.

The bounds are deliberately conservative. In particular, `2^-166` as a
capacity perturbation near one is below ordinary binary64 spacing; this
is not an implementation or experimental precision claim. It also does
not show that such precision is necessary: the actual margin and less
conservative admissible budgets have not been estimated. The observed
quantity is a recoverable scalar deformation, not an autonomous reduced
state, a new identity, or the two-bridge selection geometry.

The reference preparation, ability to hold or release capacity while
preserving the other rows, and independent clock/readout accuracy remain
supplied mathematical access assumptions. This theorem does not establish
their physical realization or account for a controller's extra state,
reaction or resources. Such access must be justified separately before
consuming this paired contract in a physical comparison.

Analytical derivation and independent reviews are retained in
`artifacts/native-finite-observation-20261029/`. No trajectory, scientific
calculation script, historical producer or new engine executor was run.
The [program](../RESEARCH_PROGRAM.md#research-current-question) owns the
next access-admission question; physical readiness remains absent.
