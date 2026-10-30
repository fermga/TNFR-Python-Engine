# A source-compatible outward facet of the buffered reference domain

2026-10-29. Analyst: `/root/admission_foundation`.

**Result.** The complete buffered acute cell intersected with the M4 energy
sublevel is not forward invariant under M2. An explicit state on one positive
donor facet has a strictly outward M2 derivative, while satisfying all cycle
constraints, the joined weighted mean and even the smaller exact nominal-source
energy sublevel. It is connected to the actual nominal reference preparation
by an energy-decreasing geometric path within the buffered cell. This is an
obstruction to this proposed domain certificate, not evidence that the actual
reference trajectory reaches the witness or exits the cell.

This is a symbolic analytical derivation under the parent's current task.
No field implementation, trajectory, target, interval diagnostic, numerical
producer or scientific test was executed. The failed diagnostic supplied no
edge information. Its source and every prior result remain untouched.

## Sources and contribution boundary

The complete law, original labels, offsets, degree metric, source and comparison
are CAPTURE, `sine-formed-two-port-entry-method`, M1--M9, especially lines
481--530 and 546--600. COMPATIBILITY,
`sine-two-port-handoff-obstruction`, equations (14)--(19), lines 439--527,
already supplies a low-energy, cycle-compatible path to an acute boundary.
Its formed-source section, A1--A3, supplies the original ports and gauge.
The live PROGRAM, `research-current-question`, admits the present analytical
domain/comparison question, not another reference attempt.

I initially found a different elementary outward witness and reported it to
the parent. The parent then identified the more useful reuse of the existing
(15)--(19) path below. I independently checked its cycle relations, transport
to original labels, energy comparison, nodal degrees and outward derivative.
This note adopts that shared construction rather than adding a duplicate
canonical family. It is not a blind review of the parent's idea.

| Consumed source, relative to repository | Bytes | SHA-256 |
| --- | ---: | --- |
| `theory/nodal/SINE_TWO_PORT_CAPTURE.md` | 47760 | `6502c13c716e89630b242236c273ac36f992e3995aa6ea9587167510802489d4` |
| `theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md` | 42842 | `b6a17ba53681a17111c2cb960ac01224596907ffa77a733229c115c322cc4607` |
| `theory/RESEARCH_PROGRAM.md` | 92570 | `5d30da78647fb5ec1c53f00970e5e93c45fbdd0ba3e36e37766d7889f9885665` |
| `src/tnfr/research/formed_two_port_reference.py` | 20488 | `0a141ab0ec38b468a96a4ddc65e09884fe2687e34e26e2f341f99eb950cf1345` |
| `artifacts/formed-two-port-attempt-20261029/diagnostic-failure-review.md` | 6077 | `35b37d43d594083c2e501f180f3f093cc70b290edad988fc0c3b9b6d39ff2768` |

## The domain and exact facet conditions

Write `q=4*pi/9`, `r=2*pi/9`, `delta=(q-r)/2=pi/9`,
`rho=1/2048`, `a_rho=pi/2-rho`, and `f(v)=1-cos(v)`.
Keep the original cycles, oriented in increasing local index including their
closing edges, and contacts `(4,13)` and `(5,14)`, donor to receiver.
The four port degrees are three; every other degree is two; total mass is forty.
Keep the original closing offsets `4*pi` and `2*pi`.

The nominal edge vector has nine donor gaps `q`, nine receiver gaps `r`, and
contact gaps `phi` and `phi-r`, where `phi=1/1000`. With the unchanged incidence
matrix `B` whose rows are `e_right-e_left`, set

    g = g0+B*xi,
    xi' = -D^{-1} B^T sin(g),
    U(g) = sum_e f(g_e),
    U_floor = 9*f(q)+9*f(r).

The proposed closed domain, in the zero joined-mean gauge, is

    C_E = {xi : 1^T D xi=0, |g_e|<=a_rho for all 20 edges,
                  U(g)<=U_floor+E},  E=1/4.

The reference starts at `xi=0`. Its gaps have strict buffered margins. Its
phase energy, independently of any complete-state form contribution, obeys

    U(g0)-U_floor = f(phi)+f(phi-r)
                 < 1/2000000 + 20/81 < 1/4,

using `0<phi<r`, `f(v)<=v^2/2` and `pi^2<10`. Thus nominal reference membership
does not require identifying actual full-state storage with reference energy.

Let `K=B D^{-1} B^T`. The exact edge derivative is

    g' = -K sin(g).

For a facet `g_e=s*a_rho`, `s` equal to +1 or -1, its inward-pointing condition
is precisely

    s * (K sin(g))_e >= 0.                                  (D1)

This is necessary for invariance at every feasible boundary point. The
remaining exact constraints already have tangent or inward derivatives:

    (1^T D xi)' = 0,
    (z^T g)' = 0 whenever z^T B=0,
    U' = -(B^T sin(g))^T D^{-1}(B^T sin(g)) <= 0.             (D2)

In particular, energy decay alone supplies no implication for (D1). On this
acute affine cell U is convex. With the strictly feasible source, the facet
conditions and energy inequality describe the inward tangent-cone test for
this convex intersection. The following single failure of (D1) is enough to
settle the proposed invariance claim; no enumeration of all forty signed
facets is required.

## Reuse the established geometric path at the buffered facet

For `0<=s<=s_b=pi/144`, use the COMPATIBILITY (15) path transported from old
ports 0/1 to original ports 4/5. Its gaps in the original chart are:

* Donor edge `5->6`: `q+8*s`.
* The other eight donor edges: `q-s`.
* All nine receiver edges: `r`.
* Contact `4->13`: `c_s=delta-s/2`; contact `5->14`: `-c_s`.

The two ring sums are `4*pi` and `2*pi`, and the mixed cycle satisfies

    c_s + r - (-c_s) - (q-s) = 0.                           (D3)

For explicit source-compatible affine coordinates, let donor increments be
`h_5=8*s` and `h_j=-s` for the other eight indices. Define

    eta_D,0 = 0,
    eta_D,j = sum_{k=0}^{j-1} h_k,  j=1,...,8,
    eta_R,j = delta-phi-(9/2)*s,  j=0,...,8,
    xi = eta - (1^T D eta/40)*1.                            (D4)

Since `eta_D,4=-4*s` and `eta_D,5=-5*s`, substitution into the original g0
gives exactly the two contact gaps above. The donor closing increment is
`-s` because `sum_j h_j=0`; both inherited offsets are retained. The common
subtraction enforces the same joined mean and changes no gap. It does not
preserve the two component means separately or reset the original source.
The original `phi` occurs explicitly in this affine representation.

Now select the exact symbolic parameter

    s_rho = (pi/18-rho)/8 > 0,  with s_rho<s_b.

At this point write

    a = q+8*s_rho = pi/2-rho,
    b = q-s_rho = 7*pi/16+rho/8,
    c = delta-s_rho/2 = 31*pi/288+rho/16.                   (D5)

Exactly one buffered facet is active: donor edge `5->6` has gap a. All other
gaps lie strictly between `-a` and `a`. This follows directly from
`0<s_rho<s_b`, `0<b<q<a`, `0<r<q` and `0<c<delta<r`.

Equations (17)--(18) of the existing owner give, throughout this same path,

    d/ds [U(s)-U_min] < -17/288,
    U_min = U_floor+2*f(delta).

Therefore the buffered witness obeys the strict energy bounds

    U(s_rho) < U_min < U_floor+pi^2/81
                       < U_floor+10/81 < U_floor+1/4.     (D6)

This imports the proved geometric path and energy inequality only. No old
reference trajectory, prepared-source capture or target root is consumed.

## Exact outward derivative under the current M2 law

At the witness, node 5 has degree three and node 6 degree two. Node 5 sees
the preceding donor gap b, following gap a and outgoing contact gap -c.
Node 6 sees incoming gap a and following gap b. Their reference rates are

    psi'_5 = (sin(a)-sin(b)-sin(c))/3,
    psi'_6 = (sin(b)-sin(a))/2.

Hence the active donor gap has the exact derivative

    g'_{5->6} = (5/6)*(sin(b)-sin(a))+(1/3)*sin(c).        (D7)

The already used elementary inequalities give a strictly positive rational
lower bound without evaluating any trigonometric number:

    sin(b) >= cos(pi/16),    sin(a)<=1,
    1-cos(pi/16) <= pi^2/512 < 5/256,
    sin(c) >= sin(31*pi/288) >= 31/144.

It follows that

    g'_{5->6} > -25/1536 + 31/432 = 767/13824 > 0.        (D8)

Thus (D1) fails at a point of C_E. Smoothness of M2 implies that its solution
from this witness immediately crosses the buffered facet. The whole proposed
domain is not forward invariant. This statement is about the exact reference
vector field, not an interval enclosure or the actual full-law phase row.

## The obstruction also belongs to the nominal-source sublevel component

The example is not excluded by replacing `E=1/4` with the sharper exact level
`U(g0)`. First vary the geometric relative origin `v` from the original `phi`
to `delta`, keeping both uniform rings. All gaps remain buffered, while

    U_nom(v) = U_floor+2-2*cos(delta)*cos(v-delta),
    dU_nom/dv = 2*cos(delta)*sin(v-delta) < 0

for `phi<=v<delta`. Remove only the joined mean along this geometric path.
It starts at the actual nominal reference `xi=0` and ends at the s=0 member
of (D4). Then follow (D4) from zero to `s_rho`; its storage strictly decreases
by (18). Consequently the witness lies in the same connected component of

    {xi : 1^T D xi=0, |g_e|<=a_rho, U(g)<=U(g0)}.

This path is an auxiliary geometric construction. Neither of its two segments
is asserted to solve M2, and the intermediate midpoint origin is not installed
as a different initial condition. The source remains `phi=1/1000`.

The witness has strict room below U(g0) and all other facets. Therefore the
failure is not just a closed-boundary convention: smooth local backward
continuation of M2 from the witness gives nearby strictly buffered initial
points still below U(g0) which subsequently cross this facet. These are
existence statements about other states in this domain, not replacements for
the one fixed reference source or its correlated actual source family.

## Closed conclusion and remaining dependency

The specific proposed theorem, invariance of the full buffered affine cell
intersected with the source-compatible scalar energy sublevel, is false for
M2. It remains false for the exact nominal-source sublevel and its source
component. This sharpens the existing energy-boundary obstruction by checking
the actual degree-normalized reference vector field on a buffered facet.

The result does not determine the nominal source trajectory's behavior, name
the limiting edge of the failed candidate box, diagnose its inflation, prove
non-entry or alter the four certified steps. It neither proves that a wider
transient domain is needed nor admits another computation.

M4's acute Jensen floor and M6--M7's first-exit comparison remain valid in
their stated scopes. A successful original-domain certificate must retain
more source-specific dynamical information than membership in this entire
energy sublevel. Alternatively a wider-domain proposal must re-establish the
floor/comparison/bootstrap dependencies. Neither branch is supplied by this
negative result; in particular, M8--M9 actual-source entry remains open and
physical mapping remains blocked.
