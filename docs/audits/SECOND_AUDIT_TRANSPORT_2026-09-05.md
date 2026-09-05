# Second audit: weighted transport and random walks

Date: 2026-09-05. Scope: the random-walk and structural-flow section of
[`structural_diffusion.py`](../../src/tnfr/physics/structural_diffusion.py), plus
[`test_random_walk_consistency.py`](../../tests/physics/test_random_walk_consistency.py).
Root diffusion-generator, spectral, cache and SDK changes are reported separately
in the [second repository audit](SECOND_REPOSITORY_AUDIT_2026-09-05.md).

## Reproduced defects and corrections

| Counterexample | Previous result | Corrected behavior |
|---|---|---|
| Parallel conductances 2 and 3 between nodes 0 and 1, with conductance 1 between 0 and 2 | The manually assembled adjacency gave transition probabilities 1/2 and 1/2 | Shared weighted adjacency sums parallel edges; probabilities are 5/6 and 1/6 |
| Isolated node, directed sink, or node with only zero-weight outgoing edges | Zero transition row; stochasticity and `L_rw = I - P` disagreed | Zero-strength nodes are absorbing; every transition row sums to one |
| Two disconnected two-node components, with edge weights 2 and 4 | Commute time in the first component was 6 instead of 2; unreachable resistance was finite | Reachable commute uses its component's volume; unreachable resistance and commute are infinite |
| Nonempty edgeless graph | Stationary vector summed to zero | Uniform stationary distribution; empty graph retains an empty vector |
| Weighted edge 2 and EPI difference 2 | Current omitted conductance, violating the weighted Kirchhoff identity | `J_ij = W_ij * (EPI_i - EPI_j)`; parallel conductances also contribute |
| Pseudoinverse deliberately replaced with zero in a regression test | The Ohm certificate accepted zero voltage because it compared expressions from the same broken inverse | A sampled injection must satisfy `L @ V = b` as well as the independently exposed resistance drop |

The random-walk matrix now delegates to the canonical diffusion operator.
Resistance is computed separately within each positive-conductance component,
which also prevents a distant component's conductance scale from changing the
pseudoinverse rank of a local component. Tests cover components with conductances
`1e-20` and `1e20`.

Current and certificate field extraction use the shared strict EPI accessor.
The documented continuity identity now matches the nodal equation: for positive
row strength `d_i` and frequency `nu_f_i`,
`(d_i / nu_f_i) * dEPI_i/dt + div(J)_i = 0`. A frozen node may retain pressure
and this constitutive current while its nodal derivative vanishes.

## Compatibility and supported scope

Public function signatures and certificate dataclass fields are unchanged.
The historical `commute_equals_2m_resistance` field now refers to the pair's
component volume; this equals `2m` for a connected, unweighted, loopless graph.
Self-loop conductance is counted once in row strength, matching the transition
matrix. A weight-6 self-loop and weight-2 connecting edge give resistance `1/2`
and commute time `5`, including the loop's holding time.

`random_walk_matrix` supports directed graphs, including sinks. Degree-based
stationarity, resistance, commute, antisymmetric current and their certificates
require symmetric conductance and now explicitly reject asymmetric adjacency.
Symmetric directed adjacency remains supported. General directed stationary
measures belong to the separate directed-diffusion solver; the degree formula
does not establish them.

For disconnected graphs, cross-component transport and `max_resistance` can be
infinite. Self-resistance and self-commute are zero. The selected stationary
measure is normalized degree when conductance exists and uniform when all nodes
are absorbing; stationarity does not establish uniqueness or convergence.

## Validation and limits

The new file contains **32 regression cases**. The first 19 exposed **16 failures**
before the random-walk corrections. Seven subsequent flow cases failed before
the flow corrections. The final source-tree run was:

```powershell
.venv312/Scripts/python.exe -m pytest tests/physics/test_random_walk_consistency.py tests/physics/test_structural_diffusion.py tests/physics/test_diffusion_generator_consistency.py -q --tb=short
```

Result: **126 passed in 0.75 seconds**. This is the recorded focused integration
run, including concurrent root diffusion tests, not a full-suite or coverage
claim. `git diff --check` passed for the edited source.

Independent assertions include closed-form two-node and parallel-edge results,
self-loop holding times, the first-step hitting equations
`H(i,t) = 1 + sum_j P(i,j) H(j,t)`, heterogeneous-frequency continuity, and
rejection of the deliberately broken inverse. Fixtures are deterministic;
certificate sampling uses seed 0.

These checks remain numerical diagnostics. Pseudoinverse rank and accuracy
depend on conditioning and floating-point precision; component isolation does
not resolve every nearly singular connected network. Resistance triangle
checks sample 200 triples, and Ohm checks sample at most 20 distinct reachable
pairs; the latter is vacuous when none exists. Neither stationary measures nor
these sampled identities prove finite-time or asymptotic convergence of every
walk. In particular, disconnected and periodic walks need not converge to one
unique global distribution. The root finite-time diffusion certificate has its
separate scope and limitations in the repository audit.
