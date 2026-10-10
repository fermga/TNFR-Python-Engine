# Full-fine stationary contact in a declared winding chamber

## Prospective declaration

This is a bounded, source-free stationary search under the complete law in
[continued contact](ORGANIZED_CONTACT_MAINTENANCE.md#1-complete-state-identities-and-stationary-equations).
It does not evolve the acquired family or change the supplied interface.
The [reflection obstruction](ORGANIZED_CONTACT_REFLECTION_OBSTRUCTION.md)
leaves possible stable targets outside its sectors unresolved.

The following declaration and its executable instrument are committed before
generating any candidate. Their commit is recorded in the exclusive attempt
record. The first outcome is retained; no budget extension or replacement is
admitted by this declaration.

| Item | Fixed rule |
| --- | --- |
| Complete law | 54 nodes `(c,j)`, `c=0,...,5`, `j=0,...,8`; the existing 144 unit edges of `C9 square W6`; held unit capacity; `x'=(-Lx+gamma*S)/4`, `theta'=gamma*Lx/4`, `gamma=1/(1023*pi)`, prime in `tau=(1023/1024)*t`; no input or event |
| Stationary reduction | Connectedness and the two conserved means give `x=0`, `S(theta)=0` on the zero-form-mean leaf. Ground phase node 0 only to choose coordinates. All other 53 coordinates, including relative donor origin, remain free; recenter a certified target to the zero lifted-phase mean |
| Observed identity | Six oriented internal C9 windings one and nine oriented old outer C5 windings one; all 99 marked edges remain away from principal cuts. No spoke winding is prescribed |
| Search domain | One lifted chamber: internal forward edges have branch integer 0 except the closing edge in each C9, which has -1; outer forward edges have 0 except those from child 4 to child 0, which have -1. Constrain each marked gap `theta_b-theta_a-2*pi*k` to `[-pi+1/1024,pi-1/1024]`. Nonacute gaps are allowed. Spokes are unconstrained |
| Domain limitation | These branch integers fix the complete marked-graph cycle class, including mixed cycles not specified by the fifteen windings. No reflection, carrier or internal-shape condition is imposed. Other cycle classes and the omitted margin strips are not searched |
| Starts | In order, four donor origins `beta=k*pi/2`, `k=0,1,2,3`, each with amplitudes `0,1/8,1/4,1/2,1,3/2`. Base phase is `2*pi*j/9+2*pi*c/5` on hosts and `2*pi*j/9+beta` on donor. One NumPy PCG64 stream with seed 20261010 draws 53 uniform `[-1,1)` values for every attempt, including amplitude zero; node 0 perturbation is zero. Multiply by amplitude, then by `t=min(1,min_e slack_e/(2*abs(q_b-q_a)))`, ignoring zero denominators, where `slack_e=pi-1/1024-abs(base_gap_e)`. This retains at least half the base slack |
| Candidate generator | Python 3.13.6, NumPy 2.5.3, SciPy 1.18.1, binary64, BLAS/OpenMP thread environment 1. One SLSQP minimization of `V=sum_all_144_edges(1-cos(delta))` per start, analytic gradient `-S`, the 99 linear constraints, `ftol=1e-12`, at most 500 iterations and 5000 value/gradient oracle calls. Every attempt and endpoint is retained, including failure and budget exhaustion |
| Screening and polish | Only endpoints with gradient infinity norm below `1e-7` and tightened-chart slack above `2e-7` receive one `root(method='hybr')` call, analytic grounded Hessian, `xtol=1e-10`, at most 600 residual and 600 Jacobian calls. A polished endpoint must satisfy the same two screens and have numerical grounded-Hessian minimum eigenvalue above `1e-7` before rigorous certification. Numerical eigenvalues and solver success labels are screens only |
| Certification budget | One box of exact radius `1/10^7` in all 53 grounded coordinates, existing dyadic128 outward arithmetic. Admit binary64 centers as their exact represented rationals. One binary64 inverse rounded to rational multiples of `2^-40` by Python nearest/ties-to-even `round(value*2**40)`; verify it by the contraction inequalities below. Round edge cosine lower bounds downward to multiples of `2^-32` and use shared exact strict LDL. No radius, precision, inverse or root retry |
| Acceptance | Strict root inclusion, strict full relative-Hessian positivity and all marked branch intervals inside the tightened chart. These imply a stationary target locally asymptotically stable on the fixed-mean leaf for the complete two-row law. An optimizer flag or small residual alone never passes |
| Stopping | Stop at the first certified target; otherwise finish the 24 attempts and stop. An exception is retained as an incomplete attempt, with no automatic rerun. A boundary-constrained minimum is not an equilibrium of the unforced law. A finite nonfinding is neither nonexistence nor actual winding loss |
| Evidence | Retain the full Git source revision, source blob associations, runtime and numeric backend, exact binary64 start/endpoint coordinates, support order and branches, oracle counts, optimizer messages, residuals, boundary slack and numerical curvature. Write exclusive attempt/outcome records with the shared artifact writer |

No trajectory horizon is introduced: optimizer iterations are not physical
time. No acquisition, source capture, spontaneous symmetry breaking or
fundamental physical identification is inferred from this search.

## Root and stability acceptance

For a grounded phase center `q`, let `F=-S` on its 53 free nodes and let `J`
be the grounded cosine Laplacian. The absent row follows from the exact
zero sum of all currents. Every primitive coordinate and edge is re-admitted;
reported residuals or Hessians are not accepted as premises.

Let `R` be the supplied rational approximate inverse. Outward trigonometric
arithmetic at `q` gives bounds on `F(q)` and `J(q)`. With exact midpoint
matrix `J0`, interval radius row sums `e_k`, and radius `r`, define

\[
 \eta\geq\|RF(q)\|_\infty,\qquad
 q_0=\max_i\left[\sum_j|\delta_{ij}-(RJ_0)_{ij}|
                    +\sum_k|R_{ik}|e_k\right],
 \qquad q_B=q_0+4d_{\max}r\|R\|_\infty.
\]

Here `d_max=7`. The cosine is 1-Lipschitz and each edge difference changes
by at most `2r`; the diagonal and off-diagonal row changes together are at
most `4*d_max*r`. Thus `q_B<1` and `eta+q_B*r<r` make `y -> y-RF(y)` a
strict self-map and contraction on the closed box. Its fixed point is unique
there. Also `q0<1` makes `R*J(q)` invertible, so `R` is invertible and the
fixed point is a zero of `F`, not merely of a singular projection.

For each edge, bound its cosine throughout the box and round the lower bound
down to the declared dyadic grid. The exact grounded Laplacian assembled
from these lower weights is a Loewner lower bound for every `J` in the box:
the difference is a sum of nonnegative multiples of edge outer products.
This remains valid with negative individual edge weights. Strict exact LDL
therefore certifies positive definiteness of the entire 53-dimensional phase
quotient. Grounding and restricting to mean zero are congruent descriptions;
their numerical eigenvalues need not coincide. The complete two-row tangent
pencil from the contact theorem then has 106 stable relative directions and
the two retained neutral means. A nonpassing LDL check alone proves neither
instability nor nonexistence; extra zero directions remain unresolved.

For completeness, positivity suffices without simultaneous diagonalization
of `L` and `M`. Linearization gives
`theta''=-L*theta'/4-gamma^2*L*M*theta/16`. On the mean-zero subspace,
any complex eigenpair satisfies
`(s^2*L^-1+s*I/4+gamma^2*M/16)v=0`. Multiplying by the conjugate transpose
of `v` gives `a*s^2+b*s+c=0` with strictly positive real
`a=v*L^-1*v`, `b=||v||^2/4` and `c=gamma^2*v*M*v/16`.
Both real roots are negative, and nonreal roots have real part `-b/(2*a)<0`.
The smooth complete law therefore has local exponential stability on each
fixed-mean leaf; neither common mean is damped.

For every marked edge, its interval gap minus `2*pi*k` must lie strictly
between `-pi+1/1024` and `pi-1/1024`. The integer branch sum is -1 around
each of the fifteen cycles, so telescoping gives winding one exactly.
Interval overlap of a sum with `2*pi` is not used as a winding proof.

## Marked-cycle information and connected chambers

The fifteen prescribed winding numbers do not specify every winding on the
marked support. Remove the 45 unmarked spokes. The resulting marked graph has
99 edges, 54 vertices and two connected components: the 45-node host product
`C9 square C5` and the donor C9. Its real cycle-space dimension is therefore
`99-54+2=47`. The six internal cycles and nine outer cycles are independent:
they are pairwise edge-disjoint, and every one has an edge with nonzero
coefficient that appears in no other prescribed cycle. Their span has
dimension 15, leaving `47-15=32` cycle-space directions not specified by those
observations. This dimension count does not assert that arbitrary integer
assignments to the remaining directions are simultaneously realizable by
phases with principal gaps.

For an oriented marked edge `a -> b`, choose real phase lifts and write

\[
 \delta_{ab}=\theta_b-\theta_a-2\pi k_{ab},
 \qquad -\pi<\delta_{ab}<\pi,\qquad k_{ab}\in\mathbb Z.
\]

Changing a vertex lift by `2*pi*m_a` changes `k_ab` by `m_b-m_a`, an integer
coboundary. Its circulation around every cycle is unchanged, and the
principal winding of an oriented cycle is minus that circulation. Along a
continuous path avoiding all marked cuts, each principal gap is continuous;
every integer cycle winding is therefore constant. Thus a marked-cycle
class is invariant under such paths, even when only fifteen of its values
are retained as the declared identity.

Conversely, suppose two phase configurations have equal windings on every
marked cycle. Their branch difference has zero circulation, so integrating
it along a spanning tree in each marked component supplies integer vertex
shifts that make their branch vectors equal. Linear interpolation of these
adjusted lifts keeps every marked gap strictly between its two cuts, because
each gap interpolates between two points in the same open interval. Hence
the complete realizable cycle class, modulo integer vertex shifts, specifies
a connected component of the cut-free phase domain. This is a statement
about continuous paths in the configuration space, not reachability under
the supplied contact law.

There are two independent common marked-component angles. After quotienting
the common phase of the whole connected contact graph, the relative donor
angle remains a free circle coordinate; no marked edge constrains it.
Spokes enter the stationary equations and storage but introduce no prescribed
cut or winding here. Fixing the global lifted mean only chooses a common
representative and does not remove this relative angle. The search above
retains it as a free optimization coordinate. Its four supplied initial
angles are starts, not four fixed-origin subproblems.

The declared branch chamber has zero winding on every elementary marked
host square. It is strictly narrower than the class defined by the fifteen
observations alone. An exact counterexample uses phases in turns,
`u=theta/(2*pi)`. For the five hosts set

\[
 u_{c,j}=j/9+c/5+(9/25)s_c t_j,\qquad
 s=(1,-1,1,-1,1),\qquad
 t=(1,-1,1,0,0,0,0,0,0),
 \tag{1}
\]

and set `u_(5,j)=j/9` for the donor. All marked principal gaps lie strictly
between `-1/2` and `1/2` turns. On each perturbed internal cycle, the two
changes of size `18/25` cross opposite cuts, so their winding changes cancel.
The other changed gaps have principal representatives `1/9 +/- 9/25`, both
inside the interval. Every internal winding remains one. At each of the
three perturbed outer slots, two positive and two negative `18/25` changes
likewise give cancelling branch crossings. The remaining outer slots are
uniform. Thus all nine outer windings also remain one.

Nevertheless, the oriented square
`(0,2) -> (1,2) -> (1,3) -> (0,3) -> (0,2)` has principal gaps

\[
 (12/25,\ 106/225,\ -1/5,\ 56/225)
\]

in turns, whose sum is one. The unperturbed product state has sum zero on
this square. These states therefore share the fifteen prescribed windings
but lie in different marked-cut-free components, even after arbitrary
integer vertex shifts. Both may be recentered to the same lifted mean
without changing this conclusion.

This counterexample concerns retained information and chamber scope only.
It is not a stationary or stable target: at node `(0,3)`, the wheel currents
cancel, while the internal current is
`sin(2*pi/9)+sin(112*pi/225)>0`. No candidate search or trajectory is needed
for these exact identities. The tightened margin and finite start budget
further restrict the numerical search; neither expands it to the other
32 unspecified cycle-space directions or proves exclusion there.

For uninterrupted maintenance from the original product chamber, the complete
marked class is a necessary target condition: every marked edge belongs to
one of the fifteen observed cycles, and the all-times contract excludes its
cut. Simultaneous compensating crossings can restore sampled winding values
but fail that cut-free contract at the crossing. This is a kinematic
restriction on a possible source-to-target connection, not a claim that the
actual source reaches any stationary state.

Membership of the actual preparation in this chamber follows from the
[retained source bounds](ORGANIZED_EXTERNAL_INTERACTION.md#2-actual-source-and-observation-history-transport),
rather than from those fifteen labels. The host is
`psi_j+Phi_c+error` with coordinate error at most `Z+m`, `Z<7*10^-70`,
`m=2*g*B_o*T`, `T=1/5000`, `g<1/3000` and `B_o<1/50`; thus `m<10^-8`.
Its carrier outer gaps are bounded by `7/5`. The donor's corresponding
error is at most `10^-70`; its supplied common origin does not affect a
marked gap. Scaling these errors to zero avoids every marked cut. The
carrier's winding-one outer gaps can then interpolate to `2*pi/5`, while
internal gaps remain `2*pi/9`. This proves membership using the existing
admitted source contract, without rerunning its producers or implying that
this interpolation is a trajectory of the law. The chosen zero-mean target
representative can be translated to any retained common phase mean.
