# Third audit: numerical integrators — 2026-09-05

## Scope and evidence

This pass examined `dynamics/symplectic.py`, `integrators.py`,
`structural_clip.py` and the constant-field integration wrapper in
`dynamics/canonical.py`. It preserves the existing nodal and mechanical
representations, force equations, units and operator catalog. It introduces no
new physical threshold, force law or operator. The tests initialize fixtures
explicitly and evolve them through the existing numerical integration APIs.

The focused baseline passed 24 tests:

```sh
python -m pytest tests/test_classical_mechanics.py tests/core_physics/test_nodal_equation.py tests/core_physics/test_structural_triad.py -q --tb=short
```

The new [numerical regression module](../../tests/test_integrator_numerics.py)
contains 62 cases. Its first 42-case run exposed the defective Yoshida
composition, zero-step changes, default timestep validation, extended dynamics
and graph-size-dependent divergence. Ten subsequent canonical-wrapper cases
initially failed at an unconditional optional GPU import; after isolating that
import, nine independently reproduced parameter/defaulting errors. The final
empty-graph pair independently reproduced a vectorized/scalar clock mismatch.

Final focused checks passed **64 tests**, including the two existing mechanics
tests. The broader run passed **1693 tests, one skipped, 23 warnings** in
38.31 seconds before adding the final two empty-graph cases:

```sh
python -m pytest tests/test_integrator_numerics.py tests/test_classical_mechanics.py -q --tb=short
python -m pytest tests/test_integrator_numerics.py tests/test_classical_mechanics.py tests/core_physics tests/operators tests/physics tests/sdk -q --tb=short
```

The warnings concerned existing repeated-Coherence detection and unavailable
optional JAX. Interpreter: `.venv312/Scripts/python.exe`, Python 3.12.10,
NumPy 2.3.3 and NetworkX 3.5. Standalone measurements explicitly inserted the
working `src` directory into `sys.path`. No random seed is needed for the
deterministic graphs and analytic trajectories below.

## I01 — Yoshida advanced the wrong interval and had no fourth-order convergence

The former method executed three partial drift/kick stages, omitted the last
drift, then executed a full Verlet step. For a force-free particle initialized
at `q=0`, `q_dot=1`, a requested `dt=0.1` returned
`q=0.13243964040201714`; the exact result is `0.1`. This is a 32.44% displacement
error, independently of force approximation. Halving the timestep of the
harmonic oscillator produced measured orders about 0.00149 and 0.00036.

The method now composes the existing synchronized velocity-Verlet kernel:

```text
S4(h) = S2(w1*h) S2(w0*h) S2(w1*h)
w1 = 1 / (2 - 2**(1/3))
w0 = -2**(1/3) * w1
2*w1 + w0 = 1
2*w1**3 + w0**3 = 0
```

The first identity preserves the interval; the second cancels the leading
cubic error of the symmetric second-order map. Leapfrog continues to share
the synchronized Verlet kernel. The incomplete private drift/kick helper was
removed. Signed finite steps remain supported because the composition and
time-reversal check require them; nonfinite steps fail before mutation, and
zero steps neither mutate the node nor call the force evaluator.

For `q''=-q`, `q(0)=1`, `q_dot(0)=0`, integrated to `T=1`, the exact state is
`[cos(1), -sin(1)]`. Euclidean state errors were:

| Method | 10 steps | 20 steps | 40 steps | Measured orders |
|---|---:|---:|---:|---|
| Velocity Verlet | 8.988604426e-4 | 2.245515720e-4 | 5.612768516e-5 | 2.00105, 2.00026 |
| Yoshida | 5.586328050e-6 | 3.487656866e-7 | 2.179195285e-8 | 4.00157, 4.00039 |

Order is `log2(error(h)/error(h/2))`. All three public mechanical steppers
also pass forward/backward recovery with absolute tolerance `1e-14`.

These claims require an autonomous separable Hamiltonian, a conservative force
depending only on position, constant inverse inertia during each step and an
initial stored force consistent with the initial position. They do not certify
arbitrary callbacks, U2, or boundedness for arbitrary potentials or timesteps.

The mechanical representation remains `EPI=[q,q_dot]`, with
`q_dot'=nu_f*force`. Existing velocity can therefore persist at zero inverse
inertia. This is explicitly distinguished from the bare first-order nodal
equation's zero-capacity condition; the implementation does not silently change
one representation into the other.

## I02 — Clipping moved stationary nodes and zero-duration integrations

In ordinary vectorized integration, `EPI=0.4`, zero nodal derivative and soft
clipping produced `0.7807144353592678`, even when `dt=0`. The scalar fallback
produced `0.8077432421346754`. In extended integration, a zero step changed
`EPI=-0.5` to zero through an unconditional `[0,1]` clamp.

Zero-duration calls now return before state, derivative telemetry or clock
writes. During a nonzero step, clipping applies only when the integrated EPI
actually changes. Thus `nu_f=0` or `DeltaNFR=0` with zero external forcing
preserves EPI. This check covers scalar and vectorized paths. It does not
discard an explicitly nonzero optional external forcing term.

Array clipping now uses the boundary policy centralized in
`structural_clip.py`, including the scalar soft-map coefficients, instead of
the integrator's divergent formula. Active soft-clipping outputs can therefore
change on the vectorized path to match the scalar path. The existing soft map
is a nonlinear remapping, not the identity throughout the allowed interval;
unconstrained ODE convergence orders are claimed only where clipping is
inactive. No global coherence improvement is inferred from these corrections.

Explicit and graph-default timesteps now share finite, nonnegative validation.
Previously graph `DT=-0.1`, NaN and infinity bypassed the explicit-argument
checks. Initial time and the subdivision minimum must also be finite; the
subdivision calculation no longer adds an epsilon that can violate its stated
minimum-step condition. The vectorized clock on an empty graph now advances
by the same interval as the scalar path; zero-duration identity still applies.

## I03 — Extended Euler used synthetic fields and asynchronous node states

The old extended path called `compute_phase_current` and `compute_dnfr_flux`
with unsupported keyword arguments. A broad exception handler concealed the
resulting `TypeError` and replaced canonical fields with synthetic values.
It also wrote each node immediately, so later nodes saw a partially advanced
graph. On a three-node path with EPI 0.4, frequency 1, pressures
`[0.1,0.2,0.3]`, phases `[0,0.4,0.8]` and `dt=0.1`, reversing node insertion
order changed pressure by as much as `2.89850625e-5` in one step.

The corrected path computes the canonical field maps once per substep, using
their actual signatures, and stages every derivative before writing nodes.
Zero flux is retained as a valid value. Field errors are no longer silently
converted into synthetic dynamics. The resulting Euler update is independent
of node insertion order in the regression, including phase and pressure.

The extended API now shares ordinary timestep preparation and configured EPI
bounds. Its default timestep previously raised `AttributeError` because the
mapping `DEFAULTS` was accessed as `DEFAULTS.dt`. Negative valid EPI values no
longer encounter a hardcoded `[0,1]` domain. Phase reads and writes use the
shared alias mechanism, and advanced phases are wrapped to `[0,2*pi)`.

An analytic two-node case independently verifies the field and time integration.
With equal zero pressure, phase separation `delta(0)=pi/2`, and the existing
local coupling `kappa`, canonical phase currents are `(+sin(delta),-sin(delta))`.
The implemented extended equation gives

```text
delta' = -2 * 0.135 * kappa * sin(delta)
delta(t) = 2 * atan(exp(-2 * 0.135 * kappa * t))
```

Its Euler trajectory has first-order convergence for 10, 20 and 40 steps to
`T=5`. The one-step analytic current check also passes after a `4*pi` phase
shift and when phases are supplied through the `phase` alias.

Extended dynamics supports Euler only. Requesting `rk4` now raises the existing
`NetworkConfigError` before changing node state, instead of silently executing
Euler under a different method name. This is an intentional compatibility
change; a coupled RK4 implementation has not been certified or substituted.

## I04 — Flux divergence changed with unrelated graph padding

The documented scalar discretization is

```text
divergence(i) = sqrt(k_i) * (J_i - mean(J_j for j in G.neighbors(i)))
```

At more than 100 nodes, a separate sparse branch used weighted degree and
counted parallel edges. It also assigned nonzero divergence to isolated nodes.
An edge of weight four with fluxes `(2,1)` gave `(1,-1)` at 100 nodes but
`(2,-2)` at 101. An unrelated isolate with flux one changed from zero to one.

The new implementation follows the existing scalar unique-neighbor semantics
at every size, including directed successors, one contribution per parallel
neighbor and zero divergence without outgoing neighbors. Weights are not
interpreted as metric spacing in this diagnostic. It avoids sparse dependency
dispatch, dense distance/index work and repeated `nodes.index` calls.

A deterministic timing comparison used unweighted paths, `J_i=sin(i)`, seven
batches of 30 calls, and median seconds per call. The previous function was
read from `HEAD:src/tnfr/dynamics/integrators.py`; the final function came from
the working source. Their outputs agreed within `1e-14` on these inputs.

| Path nodes | Previous median | Corrected median |
|---|---:|---:|
| 100 | 0.593 ms | 0.128 ms |
| 101 | 0.904 ms | 0.130 ms |
| 1000 | 12.929 ms | 1.229 ms |

These are local measurements, not general workload speed guarantees.

## I05 — The constant-field CPU wrapper required optional GPU dependencies

`integrate_canonical_nodal_equation(use_gpu=False)` unconditionally imported
the GPU subsystem and failed when optional `psutil` was unavailable. The GPU
import is now confined to the requested GPU branch. The CPU regression
explicitly rejects that import, rather than depending on the installed extras.

The wrapper also used `value or default` for timestep, step count and tolerance.
Consequently explicit `dt=0` advanced EPI, zero tolerance was replaced with
`1e-6`, and zero steps silently selected the default maximum. Resolution now
uses `is None`. Time and tolerance must be finite and nonnegative, and the
maximum step count must be a positive integer. A zero step returns unchanged
state and `steps=0`; `converged=False` records that no convergence assessment
occurred, and `backend_used="none"` records that no backend executed.

The wrapper freezes the stored frequency and pressure. Its four identical
Runge-Kutta stages were therefore redundant; Euler and RK4 now share the exact
constant-derivative update. Both method names remain accepted. Documentation
now identifies that restricted behavior and explains that `final_error` and
`converged` concern the last step-change norm, not structural equilibrium.
It no longer claims that every runtime integrator delegates to this wrapper.

## Reproducible mechanical example

Run from the repository root, using the working source explicitly:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
import numpy as np
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY
from tnfr.dynamics.symplectic import TNFRSymplecticIntegrator

node = {
    EPI_PRIMARY: np.array([0.0, 1.0]),
    VF_PRIMARY: 1.0,
    DNFR_PRIMARY: np.zeros(2),
}
TNFRSymplecticIntegrator.yoshida_4th_order(node, 0.1, lambda _: np.zeros(2))
np.testing.assert_allclose(node[EPI_PRIMARY], [0.1, 1.0], atol=1e-14, rtol=0)
```

The regression module supplies the oscillator, harmonic forcing, coupled
phase, field-padding, zero-step and invalid-input cases in runnable form.
No GPU execution, arbitrary force-callback certification, general stability
proof, or measured C(t)/Si improvement is claimed by this pass.

## Independent cross-review: numeric timestep compatibility

The independent review reran the 64 focused tests and checked the staged
extended update, zero-step behavior, reversible Yoshida coefficients and its
autonomous-force assumptions. It reproduced one compatibility regression:
`G.graph['DT']=np.float32(0.1)` previously resolved to
`0.10000000149011612`, but the new built-in `(int, float)` guard rejected it
as nonnumeric; `np.int64(1)` was rejected as well. Timestep resolution now
accepts `numbers.Real` before the existing finite/nonnegative validation.
The previous Python-bool policy is preserved.

Fourteen added cases verify constant-derivative updates with explicit and
graph-default NumPy real scalars, reject negative/NaN/infinite scalar steps
before node or clock changes, and preserve bool behavior. The final focused
run passed **78 tests**, with no skips or warnings. No broader integration
semantics were changed by this cross-review correction.
