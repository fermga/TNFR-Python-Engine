# Nucleus B — Conditional Symmetry Obstructions

**Status**: current internal mathematical scope reference; the former
publication proposal and universal RH no-go claim are withdrawn.
**Review**: 2026-09-19. No new numerical campaign.
**Companion**: [Nucleus A](NUCLEUS_A_PRIME_LADDER_ATLAS.md) records the declared
prime-ladder construction; the [Riemann memo](TNFR_RIEMANN_RESEARCH_NOTES.md)
owns the external-target boundary.

## 1. Supported statement

Let a finite group act on a specified state space through $P_\sigma$. If a
linear generator $L$ commutes with every $P_\sigma$, its flow and every
defined function of that generator also commute with the action:

$$
[L,P_\sigma]=0
\ \Longrightarrow\
[e^{-tL},P_\sigma]=0,\qquad [f(L),P_\sigma]=0.
$$

For general maps, equivariant factors compose when the group actions and
domains match. An invariant initial state then remains in the fixed set.
These are conditional algebraic results. Applying them to a live TNFR model
requires checking the full state, capacity, pressure law, support, histories,
selectors and ordering. Shared scalar configuration values alone do not
establish those premises.

No result here places the analytic zeta argument $S(T)$ in a finite symmetry
complement or proves that all canonical constructions fail to reproduce it.

## 2. Reusable owners and exact scope

| Result / tool | Owner | Supported use |
|---|---|---|
| Fixed weighted-graph diffusion equivariance | [equivariance.py](../src/tnfr/physics/equivariance.py), [symmetry_sectors.py](../src/tnfr/physics/symmetry_sectors.py) | Automorphisms preserve weights/direction; orbit-constant capacity commutes with the action |
| Reynolds projection | [symmetry_sectors.py](../src/tnfr/physics/symmetry_sectors.py) | Image is the fixed subspace; linear equivariant generators preserve its orthogonal complement |
| Pointed selectors | [pointed_symmetry.py](../src/tnfr/physics/pointed_symmetry.py) | Marking an origin reduces the group to its stabilizer; this is a declared symmetry break |
| Finite pointed operator probes | [operator_equivariance.py](../src/tnfr/physics/operator_equivariance.py) | Compare corresponding node actions and selected scalar channels on supplied fixtures |
| Finite word probes | [word_equivariance.py](../src/tnfr/physics/word_equivariance.py) | Compare selected words/prefixes; do not prove an arbitrary-history theorem |
| Linear flow intertwiners | [structural_morphism.py](../src/tnfr/physics/structural_morphism.py) | $ML_s=L_tM$ transports every solution of the corresponding fixed linear models |
| Arithmetic product controls | [crt_multiscale.py](../src/tnfr/mathematics/crt_multiscale.py), [padic_tower.py](../src/tnfr/mathematics/padic_tower.py) | Exact generator identities for supplied product/reduction-compatible families |
| Current joint observation boundaries | [TNFR_SCALE_GEOMETRY_AND_BRIDGE.md](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) | State and geometric output closure must be checked separately |

[TNFR_STRUCTURAL_OBSERVABILITY.md](TNFR_STRUCTURAL_OBSERVABILITY.md)
centralizes the diffusion proof, finite test scope and pointed/global distinction.

## 3. Why the old universal argument fails

The former proposal combined individually useful observations into stronger
claims that do not follow.

1. **Covariance is not a fixed-selector symmetry.** The family identity
   $O_{\sigma v}(P_\sigma x)=P_\sigma O_v(x)$ permits a localized emission to
   break a symmetric seed. It does not show that $O_v$ commutes with every
   automorphism while $v$ is held fixed.

2. **Graph-level gains do not erase nodal data.** Capacities, phases, EPI,
   weighted edges and histories can differ by node. A group preserving a bare
   unweighted graph need not preserve its full physical state or generator.
   Absence of one configuration slot is not a theorem excluding every
   structurally derived weighting law.

3. **A finite residual is not a universal base case.** The operator audit
   reads selected scalar channels on original nodes of two fixtures.
   Created support, arbitrary histories, all inputs and every selector are
   not certified by a zero residual in that test.

4. **Nonlinear equivariance is not linear block decomposition.** It preserves
   the fixed set but need not preserve its orthogonal complement. For the
   exchange action on two coordinates, the equivariant map
   $(x_1,x_2)\mapsto(x_1^2,x_2^2)$ sends $(1,-1)$ to $(1,1)$.
   Tensor lifts, functional calculus and linear combinations require suitable
   linear carriers; they cannot be applied indiscriminately to the catalog.

5. **Relabel-invariant does not mean information-free.** A matrix spectrum is
   invariant under simultaneous relabeling yet can encode an unordered set
   of node frequencies. Swapping frequency values while holding other data
   fixed is a different intervention. Spectral permutation invariance alone
   cannot establish blindness to arithmetic or analytic $S(T)$.

6. **A field map needs invariant input.** An equivariant tetrad read-out is
   orbit-constant on an invariant state. It need not lie in the fixed sector
   for arbitrary phase/pressure input; unavailable curvature also cannot be
   replaced with a numeric invariant.

7. **A bundle dichotomy is not an exhaustive no-go.** Showing that one
   selected invariant connection fails one diagnostic does not classify all
   possible state-derived connections or derive their dynamics. A
   non-invariant connection need not be externally supplied merely because
   the bare graph has a symmetry.

## 4. Historical finite controls

The May proposal recorded finite augmented-graph, product, line-graph and
tensor-lift comparisons. Those records are preserved in
[NUCLEUS_B_PROPOSAL_PRE_DOCUMENTATION_CLEANUP_2026-09-19.txt](research/archive/NUCLEUS_B_PROPOSAL_PRE_DOCUMENTATION_CLEANUP_2026-09-19.txt)
and the full
[Riemann notebook archive](research/archive/RIEMANN_NOTEBOOK_PRE_DOCUMENTATION_CLEANUP_2026-09-19.txt).

A verdict named INDETERMINATE_DEGENERATE_CONSTRUCTION means that the selected
statistic failed to distinguish that construction from its control within
the recorded tolerance. It is not an analytic proof that the whole candidate
family, every state on it or all future refinements are impossible.
The former count of approximately fifteen universally closed families is
therefore superseded.

The conditional product lemma survives: equivariant factors give equivariant
Kronecker products under the product action, and matching Kronecker sums
inherit the same symmetry. An induced line-graph action also needs the
selected edge weights and observables to respect that action. These facts
do not identify an analytic zeta representation.

## 5. Current use and boundary

This reference is useful for avoiding information loss in a proposed coarse
state or read-out. Specify the action, determine its fixed sector, check the
complete evolution/observation premises and produce a witness if a discarded
direction changes a retained output. Reuse the existing exact closure and
geometry owners rather than treating a symmetry slogan as a proof.

There is no active external-publication schedule, novelty claim, exhaustive
catalog-closure program or new operator requirement attached to this note.
The historical reading list and effort estimates are preserved only in the
archive. Any future publication would first need a valid self-contained
theorem and its actual source audit; a literature review cannot repair an
unproved premise.

The [single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) determines research
priorities. This arithmetic reference does not create a parallel queue or
close the ontological TNFR objective.
