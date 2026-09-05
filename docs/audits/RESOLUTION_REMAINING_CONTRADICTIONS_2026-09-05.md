# Resolution of remaining contradictions — 2026-09-05

## Scope

**Subsequent work:** the [nodal-synergy pass](NODAL_SYNERGY_AUDIT_2026-09-05.md)
addresses the node-lookup bottleneck recorded below and adds shared sparse
transport and diagnostic-field reuse. This report retains its historical
measurements and limits.

This pass implements the remaining operational corrections documented by the
four audits and reconciles their mathematical scope with the canonical guide.
The starting default suite passed **3,082 tests, 11 skipped, 97 warnings**, in
104.54 seconds. Existing uncommitted changes are preserved; `manual/` is outside
the work. No release, commit, or remote publication is performed.

## Resolution map

| Previous finding | Corrected behavior or statement | Remaining boundary |
|---|---|---|
| Fourth audit: `RANDOM_SEED=None` works only during initialization | One shared resolver validates seeds and records a realized integer for initialization and runtime consumers. | Replay needs the same initial graph, node ordering and operation history. |
| Runtime jitter depends on graph identity | Random draws derive from the recorded seed, node offset and persisted per-node progress. | Old identity-dependent jitter streams intentionally change. |
| Social contact parameter changes rewiring instead of contacts | The initial graph has the requested exact mean degree when realizable. | Rewiring can change individual degrees and connectedness; impossible parity requests raise. |
| Inspiration and hierarchy parameters are ignored | Inspiration controls rewiring; hierarchy creates explicit graph layers and metadata. | These are operational scaffolds, not validated domain models or nested EPIs. |
| Ecosystem evolution budget silently drops/rescales work | Each step applies one complete canonical word, cycling transformation, synchronization and consolidation. | A live grammar rejection stops execution; a word is not a unit of physical time. |
| T02: tetrad potential gradient does not give canonical EPI pressure | An exact restricted Dirichlet gradient-flow balance is implemented and proved under explicit assumptions. | The full multichannel/tetrad/isotropic-substrate correspondence remains unproved. |
| T07: phase wrapping supposedly enforces potential confinement | The exact topology/source-dependent bound replaces that assertion; π/4 and π/2 remain selected safety policies. | A trajectory bound needs pressure and graph assumptions. |
| T08: three operations supposedly relax every graph mode | The guide and derivation docs identify the mean-rate calibration; the P21 witness needs 231 steps. | Runtime window 3 and debt 2 remain unchanged; no general convergence certificate is implied. |
| T10: U1/U2 supposedly follow as universal analytic theorems | Initialization/closure and stabilization contracts are distinguished from existence, boundedness and integrability. | Infinite-horizon convergence still needs a pressure law and bounds on actual evolution. |
| T11: tetrad supposedly reconstructs all independent structural information | The tetrad remains the canonical diagnostic interface; algebra generation is separated from state reconstruction. | Minimal complete reconstruction is not established. |

Implementation detail and reproducible evidence are recorded in the
[seed resolution](RESOLUTION_SEEDS_2026-09-05.md),
[template resolution](RESOLUTION_TEMPLATES_2026-09-05.md), and
[theory-scope resolution](RESOLUTION_THEORY_SCOPE_2026-09-05.md).
The original [counterexamples](THEORY_CONTRADICTIONS_2026-09-05.md) remain useful
historical evidence; their old open-status summaries describe their own audit.

## Restricted variational result

For fixed finite symmetric nonnegative conductance `W`, define `d_i=Σ_j W_ij`,
`B=D−W`, `x=EPI` and nonnegative capacity `νf_i`. Then

```text
E_D = ½ xᵀ B x = ¼ Σ_ij W_ij (x_i−x_j)²,
∇E_D = Bx,
M_ii = νf_i/d_i for d_i>0, otherwise 0,
x' = −M∇E_D = −diag(νf)L_rw x,
E_D' = −(∇E_D)ᵀ M∇E_D ≤ 0.
```

[`compute_diffusion_energy`](../../src/tnfr/physics/structural_diffusion.py)
returns the energy, gradient, actual mobility, EPI rate and energy rate without
evolving or mutating the graph. It reuses the diffusion adjacency and state
readers. Only actual weighted pairs contribute to its edge-difference evaluation,
avoiding cancellation under a common field offset and spurious differences
between disconnected extreme values. Unrepresentable floating-point balances
raise explicitly. The routine retains the module's dense adjacency construction;
it is a read-out, not a new sparse solver or an integration method.

Self-loops add strength but no energy, parallel edges contribute conductance,
isolates have zero flow, and heterogeneous capacities remain per node. Positive
strength and capacity give metric `diag(d_i/νf_i)`; zero capacity gives degenerate
mobility and can preserve nonuniform positive energy. Asymmetric adjacency is
rejected for this symmetric identity. The theorem does not identify Dirichlet
energy with tetrad potential, prove energy decay under arbitrary discrete step
sizes, or establish a Hamiltonian lift of the full nodal equation.

The [19 regression cases](../../tests/physics/test_diffusion_energy_balance.py)
independently evaluate edge energy, finite-difference its gradient and directional
derivative, compare EPI rates with the canonical pressure callback, and exercise
weighted/multiple edges, loops, capacity heterogeneity, zero capacity, isolates,
large offsets, invalid inputs, numerical overflow and positive mobility underflow.
Independent review caught the underflow case: weight `1e200` and capacity `1e-200`
give an unrepresentable mobility `1e-400`, even though the true EPI rate and
dissipation can be represented. The read-out rejects it instead of reporting
false stationarity. The one-edge tetrad
counterexample remains explicit: `E_D=1/2`, whereas the tetrad potential is 1
for `EPI=[1,0]`, unit capacities and equal phases. The final integrated suite
below includes all 19 cases after independent review.

## Centralization and compatibility

The shared [scope note](../../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) supplies
the mathematical assumptions and counterexamples used by the grammar, tetrad
and fundamental theory entry points. The canonical [agent guide](../../AGENTS.md)
and its required [mirror](../../.github/agents/my-agent.md) carry the same closed
synthesis. These changes correct unsupported statements while preserving all
13 operator roles, U1–U6 validation rules and numerical thresholds.

Recorded seeds and per-node jitter progress centralize runtime reproducibility.
Existing deterministic integer initialization and sampling streams are preserved;
the formerly process-identity-dependent jitter is intentionally corrected.
Template signatures and result types are unchanged, but the corrected controls
now change topology and the ecosystem budget can execute more work than before.
U3 rejection remains active and is tested; no universal coherence increase or
successful execution for arbitrary random phases is claimed.

The jitter progress change avoids new whole-network scans per draw. Timings
in the seed report also expose an existing expensive node-order/checksum path
at 2,000 nodes in both the previous wheel and corrected source. That separate
lookup path remains a measured optimization opportunity; these changes do not
claim an end-to-end runtime speedup.

## Integrated verification

The final default suite passed **3,218 tests, 11 skipped, 97 warnings**, in
110.22 seconds: **136 additional passing cases** compared with the baseline.
These are the 19 diffusion-energy, 56 template-parameter and 61 seed-resolution
cases. The unchanged skips cover seven unavailable JAX cases, three unavailable
scikit-learn modules and one extended-longdouble case unsupported on this
Windows runtime. Warnings remain optional JAX and operator anti-pattern
diagnostics. The default suite excludes marked slow tests.

Laboratory snapshot and seed suites outside the default test tree passed
**30 tests**, with no warnings, in 1.06 seconds. Independent review also checked
continued jitter on detached SDK copies for Graph, DiGraph, MultiGraph and
MultiDiGraph with mixed node identities. Review caught and corrected a missing
`typing.cast` import in the public legacy jitter settings property; the final
seed suite exercises both direct and manager access to that property.

Both the wheel and source distribution build with the project's declared
isolated backend. All **454 Python modules and 107 typing files** parse and
match both archives byte for byte. Package name/version, Python and dependency
requirements, and console entry points match the immutable fourth-audit wheel.
An initial non-isolated build used an older local setuptools that could not
parse the license metadata; using the declared isolated backend resolved the
environment mismatch without changing project metadata.

Whitespace checks pass. All local links in the 20 audit documents and the five
reconciled theory/field guides resolve. AGENTS.md and its required mirror are
byte-identical, SHA-256
`6ad6fb1712871a3a2de631dc716d90fc8888cf012a82ba30cd753070550d141a`.
The two calibration/constant modules retain identical executable syntax trees
after docstrings are removed. No optional backend or general mathematical
conjecture is certified by these counts.

Local verification artifacts: `tmp/resolution-baseline.log`,
`tmp/resolution-final-tests.log`, `tmp/resolution-build.log`,
`tmp/resolution-dist/`, and `tmp/verify_resolution_artifacts.py`.
Runtime: Python 3.12.10, NumPy 2.3.3, NetworkX 3.5, pytest 9.0.2.
Reproduction from the repository root:

```powershell
.venv312/Scripts/python.exe -m pytest -q --tb=short -rs
.venv312/Scripts/python.exe -m pytest factorization-lab/tests/test_snapshot_system.py factorization-lab/tests/test_seed_management.py -q
.venv312/Scripts/python.exe -m build --outdir tmp/resolution-dist
.venv312/Scripts/python.exe tmp/verify_resolution_artifacts.py
```

Focused counts overlap and must not be added together. These are local
verification artifacts, not a published release.
