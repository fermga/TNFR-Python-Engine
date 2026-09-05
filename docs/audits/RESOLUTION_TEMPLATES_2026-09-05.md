# Resolution of template parameter contradictions — 2026-09-05

## Scope and reproduced behavior

This change resolves the documented operational parameter gaps in
[sdk/templates.py](../../src/tnfr/sdk/templates.py). It adds label-preserving
support-graph helpers to [sdk/_topology.py](../../src/tnfr/sdk/_topology.py),
reusing the existing SDK rewiring default and integer/probability validation.
It does not modify a physical threshold, canonical operator, U3 phase gate or
U5 interpretation. Graph layers are topology metadata, not nested EPIs.

At seed 0 with 12 nodes, the previous implementation produced these results:

- Social contact requests 2, 5 and 8 all gave mean degree 4; the argument changed
  rewiring probability instead of contact count.
- Inspiration values 0 and 1 produced identical edges.
- Hierarchy depths 1 and 4 produced identical edges and no hierarchy metadata.
- Ecosystem budgets 1, 9, 10, 20 and 30 scheduled respectively 0, 0, 3, 8 and 10
  canonical word applications.

The first new regression run had 37 failures and three passes. The existing
focused baseline passed 49 tests with one warning in 0.47 seconds. The root's
full baseline completed before source edits: 3,082 passed, 11 skipped and 97
warnings in 104.54 seconds.

## Defined operational behavior

| Existing parameter | Implemented meaning |
|---|---|
| `connections_per_person` | Exact initial mean degree `k`: the simple support graph contains `people*k/2` edges. |
| `inspiration_level` | Rewiring probability of a sparse small-world lattice, independent of edge count. |
| `hierarchy_depth` | Number of nonempty graph layers, recorded by each node's `hierarchy_level`. |
| `evolution_steps` | Exact number of canonical word applications when execution completes. |

The social scaffold starts with a ring lattice. An odd requested degree adds
opposite-node matching on an even-sized ring. Edge rewiring uses the existing
`SDK_REWIRING_PROB_DEFAULT = 0.16` and the network's own RNG. Each successful
rewire replaces one edge, preserving mean degree while allowing individual
degrees to differ. A realizable integer degree requires `0 <= k < people` and
an even product `people*k`; impossible requests are rejected before node
creation. Zero contacts and a single node with zero contacts are supported.
Connectedness is not guaranteed by rewiring or by a low contact count.

Inspiration zero preserves the small-world ring lattice; one attempts every
eligible rewire using the existing SDK small-world implementation. Its fixed
edge count is independent of inspiration. Tiny or complete scaffolds can lack
alternative edges, so the parameter need not change every particular graph.
At 12 nodes and seed 7, values zero and one produce different edge sets with
identical node initialization and edge counts.

Hierarchy depth one is a peer ring. Otherwise the first node is the root and
the remaining nodes are spread as evenly as possible over subsequent layers,
with earlier layers receiving any remainder. Each child has one parent in the
preceding layer; nodes within each layer form a ring. The initial graph is
connected, depth cannot exceed the node count, and depth equal to the node
count gives a chain. The support graph remains undirected. These properties
describe the initial scaffold and do not certify multiscale physical coherence.

Ecosystem steps now cycle through `creative_mutation`, `network_sync` and
`consolidation`, one word per step, without discarding a remainder. A word
contains several canonical operators; this budget is neither an individual
operator count nor elapsed physical time. Social, creative and organizational
phase allocations retain their existing division/remainder rules, with
preflight validation and explicit word-budget documentation.

The graph's `template_topology` metadata records the contact degree/rewiring
choice, inspiration rewiring choice or hierarchy depth. Population sizes must
be positive integers; cycle budgets must be non-negative integers. All these
controls are validated before node creation.

## Validation and compatibility

The [new regression module](../../tests/sdk/test_template_parameter_semantics.py)
contains 56 cases. It checks contact counts at zero, odd, even and complete
degrees; rewiring and hierarchy effects; mixed node identities; exact scheduled
budgets; rejection before node creation; and reproducibility.

Twelve cases execute actual one-, two- and three-word template runs with
explicitly coherent initial phases. They verify exact executed operator order,
identical repeated histories, topology, `C(t)` and per-node sense indices at
seed 7. Operators and their hard phase checks remain active in these tests.
A separate naturally sampled case confirms that
`creative_process_model(ideas=12, inspiration_level=0.4,
development_cycles=3, random_seed=7)` still rejects an inadmissible Resonance
step at U3. Successful coherent fixtures do not establish arbitrary sampled
phase admissibility.

The combined parameter, initialization and topology run passed 105 tests with
one existing warning in 1.88 seconds. The full SDK, hard-U3 and validated-word
integration then passed **222 tests, with two warnings, in 11.44 seconds**.
Warnings concern the existing Silence/Coherence pattern and unavailable
optional JAX. Interpreter: Python 3.12.10 through
`.venv312/Scripts/python.exe`, with NumPy 2.3 and NetworkX 3.5. Commands:

```sh
python -m pytest tests/sdk/test_template_parameter_semantics.py -q --tb=short
python -m pytest tests/sdk/test_template_parameter_semantics.py tests/sdk/test_experiment_reproducibility.py tests/sdk/test_topology_identity.py -q --tb=short
python -m pytest tests/sdk tests/operators/test_u3_hard_invariant.py tests/operators/test_validated_sequence_execution.py -q --tb=short
```

Public signatures and return types are unchanged. Intended behavior changes are
visible: previously ignored controls now change the initial topology; impossible
contact/layer requests fail explicitly; and the ecosystem budget can execute
more work than before because it no longer silently drops or rescales steps.
The shared SDK defaults remain operational choices, not structural constants.

Live phase or grammar rejection still raises and stops the run. A template
returns no result in that case, and already executed operations are not rolled
back. Callers requiring access to partially evolved state should construct and
retain a `TNFRNetwork` explicitly. No automatic phase adjustment, gate bypass,
domain-model validation or universal convergence claim is introduced.
